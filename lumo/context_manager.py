from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .features import memory as memorylib
from .tool_executor import strip_tool_hints
from .workspace import AGENT_STATE_DIR, now


DEFAULT_TOTAL_BUDGET = 260000
CURRENT_REQUEST_SECTION = "current_request"
CURRENT_REQUEST_BUDGET = 30000
CURRENT_REQUEST_OVERFLOW_FILE = "prompt.txt"
CONTEXT_COMPRESSION_TEMPLATE = "lumo/prompt/context_compress.md"
CONTEXT_SUMMARY_KIND = "context_summary"
CONTEXT_SUMMARY_MAX_TOKENS = 10000
CONTEXT_COMPRESSION_RETRY_DROP_CHARS = 10000
SECTION_ORDER = ("prefix", "durable_memory", "history", CURRENT_REQUEST_SECTION)
STALE_READ_MESSAGE = (
    "This earlier read_file output is stale because the file freshness no longer matches "
    "or the file was modified later in the transcript. "
    "Read the file again before relying on its contents."
)


_ASCII_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _is_cjk_char(char):
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
    )


def _context_units(text):
    text = str(text)
    units = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if _is_cjk_char(char):
            units += 1
            index += 1
            continue
        match = _ASCII_WORD_PATTERN.match(text, index)
        if match:
            units += 1
            index = match.end()
            continue
        units += 1
        index += 1
    return units


@dataclass
class SectionRender:
    raw: str
    budget: int | None
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)

    @property
    def raw_units(self):
        return _context_units(self.raw)

    @property
    def rendered_units(self):
        return _context_units(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = {}
        self.section_floors = {}
        self.reduction_order = ("history",)
        self._current_request_details = {}
        self._current_user_message = ""
        self._last_context_compression = self._empty_compression_metadata()

    def build(self, user_message):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.build_async(user_message))

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(self.build_async(user_message))
            except BaseException as exc:
                result["error"] = exc

        import threading

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def build_async(self, user_message):
        user_message = str(user_message)
        durable_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            durable_memory_enabled = self.agent.feature_enabled("memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        current_request_text, current_request_details = self._prepare_current_request(user_message)
        self._current_request_details = current_request_details
        self._current_user_message = user_message
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "durable_memory": "Durable memory:\n- disabled" if not durable_memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: current_request_text,
        }
        rendered = self._render_sections(section_texts)
        prompt = self._assemble_prompt(rendered)
        compression_metadata = self._empty_compression_metadata()
        if context_reduction_enabled and _context_units(prompt) > self.total_budget:
            rendered, prompt, compression_metadata = await self._compress_history_until_fit(section_texts, rendered, prompt)
        self._last_context_compression = compression_metadata
        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            user_message=user_message,
            section_texts=section_texts,
            compression_metadata=compression_metadata,
        )
        return prompt, metadata

    def _current_request_prefix(self):
        return (
            "Progress:\n"
            f"- Current task completion score: {int(getattr(self.agent, 'last_completion_score', 0) or 0)}\n"
            "Current user request:\n"
            "Before answering, check whether the accumulated durable memory helps you interpret the user's intent or project context.\n"
        )

    def _prepare_current_request(self, user_message):
        prefix = self._current_request_prefix()
        inline_text = prefix + str(user_message)
        user_units = _context_units(user_message)
        inline_units = _context_units(inline_text)
        details = {
            "externalized": False,
            "externalized_path": "",
            "raw_user_chars": len(str(user_message)),
            "raw_user_units": user_units,
            "budget_units": CURRENT_REQUEST_BUDGET,
        }
        if user_units <= CURRENT_REQUEST_BUDGET and inline_units <= CURRENT_REQUEST_BUDGET:
            return inline_text, details

        root = Path(getattr(self.agent, "root", "."))
        relative_path = Path(AGENT_STATE_DIR) / CURRENT_REQUEST_OVERFLOW_FILE
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(user_message), encoding="utf-8")
        display_path = "./" + relative_path.as_posix()
        details.update(
            {
                "externalized": True,
                "externalized_path": display_path,
            }
        )
        rendered = (
            "Current user request:\n"
            f"用户的指令长度超限，完整内容已经保存到 {display_path}。\n"
            f"原始用户指令约 {user_units} 个上下文单位，超过 current_request 上限 {CURRENT_REQUEST_BUDGET}。\n"
            f"请先使用 read_file 从头到尾完整读取 {relative_path.as_posix()}；如果 read_file 返回 has_more 或 system-reminder，"
            "请继续按 next_offset 读取，直到完整读完整个文件后，再回答用户的问题。\n"
            "Do not answer before reading the full file."
        )
        return rendered, details

    def _history_without_externalized_current_request(self, history):
        details = getattr(self, "_current_request_details", {}) or {}
        if not details.get("externalized") or not history:
            return history
        current_user_message = getattr(self, "_current_user_message", None)
        last_item = history[-1]
        if last_item.get("role") == "user" and str(last_item.get("content", "")) == str(current_user_message):
            return history[:-1]
        return history

    def _prompt_history(self):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history = self._history_without_externalized_current_request(history)
        summary_index = self._latest_context_summary_index(history)
        if summary_index is None:
            return history
        return history[summary_index:]

    def _latest_context_summary_index(self, history):
        for index in range(len(history) - 1, -1, -1):
            if self._is_context_summary(history[index]):
                return index
        return None

    def _is_context_summary(self, item):
        return (
            isinstance(item, dict)
            and item.get("role") == "system"
            and item.get("kind") == CONTEXT_SUMMARY_KIND
        )

    def _render_sections(self, section_texts):
        history = self._prompt_history()
        history_raw = self._raw_history_text(history)
        background_task_text = str(getattr(self.agent, "recent_background_tasks_text", lambda: "")() or "").strip()
        if background_task_text:
            history_raw = "\n".join([background_task_text, "", history_raw]).strip()
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=None, rendered=section_texts["prefix"], details={}),
            "durable_memory": SectionRender(raw=section_texts["durable_memory"], budget=None, rendered=section_texts["durable_memory"], details={}),
            "history": SectionRender(
                raw=history_raw,
                budget=None,
                rendered=history_raw,
                details=self._history_details(history),
            ),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=CURRENT_REQUEST_BUDGET,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _history_details(self, history):
        stale_read_indexes = self._stale_read_indexes(history)
        return {
            "summary_index": self._latest_context_summary_index(history),
            "history_entries": len(history),
            "stale_read_replacement_count": len(stale_read_indexes),
        }

    async def _compress_history_until_fit(self, section_texts, rendered, prompt):
        metadata = self._empty_compression_metadata()
        metadata.update(
            {
                "triggered": True,
                "status": "running",
                "before_prompt_chars": len(prompt),
                "before_prompt_units": _context_units(prompt),
                "before_history_chars": rendered["history"].rendered_chars,
                "before_history_units": rendered["history"].rendered_units,
                "template_path": CONTEXT_COMPRESSION_TEMPLATE,
                "max_summary_tokens": CONTEXT_SUMMARY_MAX_TOKENS,
                "retry_drop_chars": CONTEXT_COMPRESSION_RETRY_DROP_CHARS,
            }
        )
        rounds = []
        total_retry_count = 0
        total_discarded_chars = 0
        failures = []
        while _context_units(prompt) > self.total_budget:
            history_text = rendered["history"].rendered
            if not history_text.strip() or history_text.strip() == "Transcript:\n- empty":
                raise RuntimeError("Prompt exceeds context budget, but there is no history to compress.")
            before_prompt_units = _context_units(prompt)
            before_history_units = rendered["history"].rendered_units
            summary, attempt_metadata = await self._compress_history_with_retries(history_text)
            total_retry_count += int(attempt_metadata.get("retry_count", 0))
            total_discarded_chars += int(attempt_metadata.get("discarded_chars", 0))
            failures.extend(list(attempt_metadata.get("failures", [])))
            round_metadata = dict(metadata)
            round_metadata.update(attempt_metadata)
            round_metadata.update(
                {
                    "before_prompt_units": before_prompt_units,
                    "before_history_units": before_history_units,
                }
            )
            summary_record = self._append_context_summary(summary, round_metadata)
            rendered = self._render_sections(section_texts)
            prompt = self._assemble_prompt(rendered)
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "summary_history_index": len(getattr(self.agent, "session", {}).get("history", [])) - 1,
                    "summary_created_at": summary_record.get("created_at", ""),
                    "summary_chars": len(summary),
                    "summary_units": _context_units(summary),
                    "retry_count": int(attempt_metadata.get("retry_count", 0)),
                    "discarded_chars": int(attempt_metadata.get("discarded_chars", 0)),
                    "compression_input_chars": int(attempt_metadata.get("compression_input_chars", 0)),
                    "compression_input_units": int(attempt_metadata.get("compression_input_units", 0)),
                    "before_prompt_units": before_prompt_units,
                    "after_prompt_units": _context_units(prompt),
                    "before_history_units": before_history_units,
                    "after_history_units": rendered["history"].rendered_units,
                }
            )
            if len(rounds) > 50:
                raise RuntimeError("Context compression did not converge after 50 rounds.")
        metadata.update(
            {
                "status": "ok",
                "round_count": len(rounds),
                "rounds": rounds,
                "retry_count": total_retry_count,
                "discarded_chars": total_discarded_chars,
                "failures": failures,
                "summary_history_index": rounds[-1]["summary_history_index"] if rounds else None,
                "summary_created_at": rounds[-1]["summary_created_at"] if rounds else "",
                "after_prompt_chars": len(prompt),
                "after_prompt_units": _context_units(prompt),
                "after_history_chars": rendered["history"].rendered_chars,
                "after_history_units": rendered["history"].rendered_units,
                "over_budget_after_compression": _context_units(prompt) > self.total_budget,
            }
        )
        return rendered, prompt, metadata

    async def _compress_history_with_retries(self, history_text):
        template = self._load_compression_template()
        remaining = str(history_text)
        discarded_chars = 0
        retry_count = 0
        failures = []
        while remaining:
            compression_prompt = self._render_compression_prompt(template, remaining)
            try:
                summary = await self._complete_compression_prompt(compression_prompt)
                summary = str(summary).strip()
                if not summary:
                    raise RuntimeError("context compression returned empty summary")
                return summary, {
                    "status": "ok",
                    "retry_count": retry_count,
                    "discarded_chars": discarded_chars,
                    "compression_input_chars": len(remaining),
                    "compression_input_units": _context_units(remaining),
                    "compression_prompt_chars": len(compression_prompt),
                    "compression_prompt_units": _context_units(compression_prompt),
                    "failures": failures,
                }
            except Exception as exc:
                failures.append(str(exc))
                drop_chars = min(CONTEXT_COMPRESSION_RETRY_DROP_CHARS, len(remaining))
                remaining = remaining[drop_chars:]
                discarded_chars += drop_chars
                retry_count += 1
        raise RuntimeError(
            "Context compression failed after discarding all retry input: "
            + (failures[-1] if failures else "unknown compression error")
        )

    def _load_compression_template(self):
        path = Path(getattr(self.agent, "root", ".")) / CONTEXT_COMPRESSION_TEMPLATE
        if not path.is_file():
            raise RuntimeError(f"Context compression template not found: {CONTEXT_COMPRESSION_TEMPLATE}")
        return path.read_text(encoding="utf-8")

    def _render_compression_prompt(self, template, history_text):
        return (
            str(template)
            .replace("{{MAX_SUMMARY_TOKENS}}", str(CONTEXT_SUMMARY_MAX_TOKENS))
            .replace("{{CONTEXT}}", str(history_text))
        )

    async def _complete_compression_prompt(self, prompt):
        complete_async = getattr(self.agent.model_client, "complete_async", None)
        if complete_async is not None:
            return await complete_async(prompt, CONTEXT_SUMMARY_MAX_TOKENS)
        return await asyncio.to_thread(self.agent.model_client.complete, prompt, CONTEXT_SUMMARY_MAX_TOKENS)

    def _append_context_summary(self, summary, compression_metadata):
        history = self.agent.session.setdefault("history", [])
        record = {
            "role": "system",
            "kind": CONTEXT_SUMMARY_KIND,
            "content": str(summary).strip(),
            "created_at": now(),
            "metadata": {
                "source": "context_compression",
                "template_path": CONTEXT_COMPRESSION_TEMPLATE,
                "max_summary_tokens": CONTEXT_SUMMARY_MAX_TOKENS,
                "retry_count": int(compression_metadata.get("retry_count", 0)),
                "discarded_chars": int(compression_metadata.get("discarded_chars", 0)),
                "before_prompt_units": int(compression_metadata.get("before_prompt_units", 0)),
                "before_history_units": int(compression_metadata.get("before_history_units", 0)),
            },
        }
        history.append(record)
        self.agent.session_path = self.agent.session_store.save(self.agent.session)
        return record

    def _empty_compression_metadata(self):
        return {
            "triggered": False,
            "status": "skipped",
            "template_path": CONTEXT_COMPRESSION_TEMPLATE,
            "max_summary_tokens": CONTEXT_SUMMARY_MAX_TOKENS,
            "retry_drop_chars": CONTEXT_COMPRESSION_RETRY_DROP_CHARS,
            "retry_count": 0,
            "discarded_chars": 0,
            "failures": [],
        }

    def render_history_for_delegate(self):
        return self._raw_history_text(self._prompt_history())

    def _read_history_key(self, item):
        args = item.get("args", {})
        path = str(args.get("path", "")).strip()
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
        offset = read_window.get("start_line", args.get("offset", args.get("start", "")))
        limit = read_window.get("requested_lines", args.get("limit", args.get("end", "")))
        return (path, str(offset), str(limit))

    def _history_path_key(self, item):
        args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
        path = str(args.get("path", "")).strip()
        if not path:
            return ""
        root = getattr(self.agent, "root", None)
        try:
            raw_path = Path(path)
            if root is not None and not raw_path.is_absolute():
                raw_path = Path(root) / raw_path
            resolved = raw_path.resolve()
            if root is not None:
                root_path = Path(root).resolve()
                try:
                    return resolved.relative_to(root_path).as_posix()
                except ValueError:
                    pass
            return resolved.as_posix()
        except Exception:
            return path.replace("\\", "/").lstrip("./")

    def _stale_read_indexes(self, history):
        stale_indexes = set()
        latest_reads_by_path = {}
        for index, item in enumerate(history):
            if item.get("role") != "tool":
                continue
            name = item.get("name")
            path_key = self._history_path_key(item)
            if not path_key:
                continue
            if name == "read_file":
                if self._read_freshness_stale(item, path_key):
                    stale_indexes.add(index)
                    continue
                latest_reads_by_path.setdefault(path_key, []).append(index)
            elif name in {"write_file", "patch_file"}:
                stale_indexes.update(latest_reads_by_path.get(path_key, []))
                latest_reads_by_path[path_key] = []
        return stale_indexes

    def _read_freshness_stale(self, item, path_key=None):
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        expected = metadata.get("freshness")
        if not expected:
            return False
        path_key = path_key or self._history_path_key(item)
        if not path_key:
            return False
        current = memorylib.file_freshness(path_key, getattr(self.agent, "root", None))
        return expected != current

    def _render_stale_read_history_item(self, item):
        prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True, ensure_ascii=False)}"
        return [prefix, STALE_READ_MESSAGE]

    @staticmethod
    def _history_item_tool_reminder(item):
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        return str(metadata.get("tool_reminder", "")).strip()

    @staticmethod
    def _history_item_display_content(item):
        return strip_tool_hints(item.get("content", ""))

    @staticmethod
    def _latest_tool_reminder_indexes(history):
        latest = {}
        for index, item in enumerate(history):
            if item.get("role") != "tool":
                continue
            key = ContextManager._tool_reminder_key(item)
            if key is None:
                continue
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            if str(metadata.get("tool_reminder", "")).strip():
                latest[key] = index
        return latest

    @staticmethod
    def _tool_reminder_key(item):
        name = str(item.get("name", "")).strip()
        if name in {"read_file", "grep", "git_status"}:
            return (name,)
        if name == "git_diff":
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            path = str(args.get("path", ".")).strip() or "."
            mode = str(args.get("mode", "workspace")).strip() or "workspace"
            return (name, path, mode)
        if name == "task_list":
            return (name,)
        if name == "task_output":
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            task_id = str(args.get("task_id", "")).strip()
            stream = str(args.get("stream", "stdout")).strip() or "stdout"
            if task_id:
                return (name, task_id, stream)
        return None

    def _raw_history_text(self, history):
        if not history:
            return "Transcript:\n- empty"
        filtered_history = list(self.agent._iter_history_items_for_prompt(history))
        stale_read_indexes = self._stale_read_indexes(filtered_history)
        latest_tool_reminders = self._latest_tool_reminder_indexes(filtered_history)
        lines = []
        for index, item in enumerate(filtered_history):
            if self._is_context_summary(item):
                lines.append("[context_summary]")
                lines.append(str(item.get("content", "")))
                continue
            if item.get("role") == "tool":
                if index in stale_read_indexes:
                    lines.extend(self._render_stale_read_history_item(item))
                    continue
                lines.append(f"[tool:{item.get('name', '')}] {json.dumps(item.get('args', {}), sort_keys=True, ensure_ascii=False)}")
                if item.get("name") == "read_file":
                    summary = str(item.get("summary", "")).strip()
                    if not summary:
                        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                        summary = str(metadata.get("archive_summary", "")).strip()
                    lines.append(summary or str(item.get("content", "")))
                    if latest_tool_reminders.get(self._tool_reminder_key(item)) == index:
                        reminder = self._history_item_tool_reminder(item)
                        if reminder:
                            lines.append(f"<tool_reminder>{reminder}</tool_reminder>")
                else:
                    if item.get("name") in {"git_status", "git_diff"}:
                        lines.append(self._history_item_display_content(item))
                        if latest_tool_reminders.get(self._tool_reminder_key(item)) == index:
                            reminder = self._history_item_tool_reminder(item)
                            if reminder:
                                lines.append(f"<tool_reminder>{reminder}</tool_reminder>")
                    elif item.get("name") in {"grep", "task_output", "task_list", "run_shell_bg", "task_stop"}:
                        summary = str(item.get("summary", "")).strip()
                        if not summary:
                            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                            summary = str(metadata.get("archive_summary", "")).strip()
                        lines.append(summary or str(item.get("content", "")))
                        if latest_tool_reminders.get(self._tool_reminder_key(item)) == index:
                            reminder = self._history_item_tool_reminder(item)
                            if reminder:
                                lines.append(f"<tool_reminder>{reminder}</tool_reminder>")
                    else:
                        lines.append(str(item.get("content", "")))
            else:
                lines.append(f"[{item.get('role', 'unknown')}] {item.get('content', '')}")
        return "\n".join(["Transcript:", *lines])

    def _assemble_prompt(self, rendered):
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["durable_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(self, prompt, rendered, user_message, section_texts, compression_metadata):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": None,
                "rendered_chars": rendered[section].rendered_chars,
                "raw_units": rendered[section].raw_units,
                "budget_units": None,
                "rendered_units": rendered[section].rendered_units,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            "raw_units": _context_units(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_units": CURRENT_REQUEST_BUDGET,
            "rendered_units": rendered[CURRENT_REQUEST_SECTION].rendered_units,
        }
        prompt_units = _context_units(prompt)
        current_request_details = dict(getattr(self, "_current_request_details", {}) or {})
        current_request_externalized = bool(current_request_details.get("externalized"))
        current_request_text = (
            f"<externalized to {current_request_details.get('externalized_path', '')}>"
            if current_request_externalized
            else user_message
        )
        compression_metadata = dict(compression_metadata or self._empty_compression_metadata())
        return {
            "prompt_chars": len(prompt),
            "prompt_units": prompt_units,
            "prompt_budget_chars": None,
            "prompt_budget_units": self.total_budget,
            "prompt_budget_unit": "ascii_word_cjk_char_or_punctuation",
            "prompt_over_budget": prompt_units > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                "prefix": None,
                "durable_memory": None,
                "history": None,
                CURRENT_REQUEST_SECTION: CURRENT_REQUEST_BUDGET,
            },
            "section_budget_unit": "ascii_word_cjk_char_or_punctuation",
            "sections": section_metadata,
            "budget_reductions": [compression_metadata] if compression_metadata.get("triggered") else [],
            "context_compression": compression_metadata,
            "reduction_order": ["history"],
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "raw_units": rendered["history"].raw_units,
                "rendered_units": rendered["history"].rendered_units,
                "history_entries": int(rendered["history"].details.get("history_entries", 0)),
                "summary_index": rendered["history"].details.get("summary_index"),
                "stale_read_replacement_count": int(rendered["history"].details.get("stale_read_replacement_count", 0)),
            },
            "current_request": {
                "text": current_request_text,
                "externalized": current_request_externalized,
                "externalized_path": str(current_request_details.get("externalized_path", "")),
                "budget_units": CURRENT_REQUEST_BUDGET,
                "raw_chars": int(current_request_details.get("raw_user_chars", len(user_message))),
                "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
                "raw_units": int(current_request_details.get("raw_user_units", _context_units(user_message))),
                "rendered_units": _context_units(user_message) if not current_request_externalized else rendered[CURRENT_REQUEST_SECTION].rendered_units,
                "section_units": rendered[CURRENT_REQUEST_SECTION].rendered_units,
            },
        }
