"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、durable memory、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .features import memory as memorylib


DEFAULT_TOTAL_BUDGET = 300000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 4000,
    "durable_memory": 14000,
    "history": 250000,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 2000,
    "durable_memory": 500,
    "history": 1,
}
# 当 prompt 超预算时，会优先压缩这些 section。
DEFAULT_REDUCTION_ORDER = ("history", "durable_memory", "prefix")
SECTION_ORDER = ("prefix", "durable_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
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
    """Count prompt budget units: ASCII words, CJK characters, and punctuation."""
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


def _clip_to_units(text, limit):
    text = str(text)
    if limit <= 0:
        return ""
    if _context_units(text) <= limit:
        return text
    suffix = "..."
    suffix_units = _context_units(suffix)
    if limit <= suffix_units:
        return "." * limit

    units = 0
    index = 0
    target = limit - suffix_units
    while index < len(text) and units < target:
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
    return text[:index].rstrip() + suffix


def _tail_clip(text, limit):
    return _clip_to_units(text, int(limit))


@dataclass
class SectionRender:
    raw: str
    budget: int
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
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            # Accept old config keys, but the prompt no longer renders a relevant_memory section.
            for key, value in section_budgets.items():
                key = str(key)
                if key == "memory":
                    key = "durable_memory"
                if key == "relevant_memory":
                    continue
                self.section_budgets[key] = int(value)
        self._section_floor_overrides = {}
        for key, value in (section_floors or {}).items():
            key = str(key)
            if key == "memory":
                key = "durable_memory"
            if key == "relevant_memory":
                continue
            self._section_floor_overrides[key] = int(value)
        self.section_floors = self._compute_section_floors()
        normalized_reduction_order = []
        for section in reduction_order or DEFAULT_REDUCTION_ORDER:
            section = str(section)
            if section == "memory":
                section = "durable_memory"
            if section == "relevant_memory" or section == CURRENT_REQUEST_SECTION:
                continue
            if section in SECTION_ORDER and section not in normalized_reduction_order:
                normalized_reduction_order.append(section)
        self.reduction_order = tuple(normalized_reduction_order or DEFAULT_REDUCTION_ORDER)

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 持久记忆 +
        历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `Pico.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，durable
        memory 提供跨会话事实，history 提供本会话事实。
        """
        user_message = str(user_message)
        # 给每个 prompt section 算一个“最小保留长度下限”。
        self.section_floors = self._compute_section_floors()
        durable_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            durable_memory_enabled = self.agent.feature_enabled("memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "durable_memory": "Durable memory:\n- disabled" if not durable_memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: (
                "Current user request:\n"
                "Before answering, check whether the accumulated durable memory helps you interpret the user's intent or project context.\n"
                f"{user_message}"
            ),
        }
        # Checkpoint state is still evaluated and recorded in metadata, but the
        # rendered checkpoint block is temporarily omitted from the prompt to
        # avoid repeating task state already present in memory/history.
        # checkpoint_text = ""
        # if hasattr(self.agent, "render_checkpoint_text"):
        #     checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        # if checkpoint_text:
        #     section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text

        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 如果 prompt 超预算，就按固定顺序不断压缩。
        # 这里的顺序体现了平台偏好：
        # 先压缩 history，再压缩 durable memory，最后才动 prefix。
        # 最新用户请求永远不裁剪，因为那是本轮最重要的输入。
        prompt_units = _context_units(prompt)
        while prompt_units > self.total_budget:
            overflow = prompt_units - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_units": current_budget,
                        "after_units": new_budget,
                        "overflow_units": overflow,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets)
                prompt = self._assemble_prompt(rendered)
                prompt_units = _context_units(prompt)
                reduced = True
                break
            if not reduced:
                break

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    def _render_sections_without_reduction(self, section_texts):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._raw_history_text(history)
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=_context_units(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "durable_memory": SectionRender(raw=section_texts["durable_memory"], budget=_context_units(section_texts["durable_memory"]), rendered=section_texts["durable_memory"], details={}),
            "history": SectionRender(raw=history_raw, budget=_context_units(history_raw), rendered=history_raw, details={"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _compute_section_floors(self):
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0))
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_history_section(self, budget):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self._raw_history_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_read_summary_count": 0,
                    "summarized_tool_count": 0,
                    "stale_read_replacement_count": 0,
                },
            )

        # 优先保留最近的历史，因为下一步决策通常最依赖刚刚发生的工具结果。
        recent_window = 6
        recent_start = max(0, len(history) - recent_window)
        stale_read_indexes = self._stale_read_indexes(history)
        history_entries, history_details = self._compressed_history_entries(history, recent_start, stale_read_indexes)
        rendered_entries = []
        for entry in reversed(history_entries):
            recent = bool(entry.get("recent", False))
            candidate_lines = list(entry.get("lines", []))
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            if _context_units(candidate_rendered) <= budget:
                rendered_entries = candidate_entries
                continue
            if recent:
                available = budget - _context_units("Transcript:")
                if rendered_entries:
                    available -= sum(_context_units(line) + 1 for line in rendered_entries)
                available = max(20, available - 1)
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if _context_units(candidate_rendered) <= budget:
                    rendered_entries = candidate_entries
            else:
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if _context_units(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
        rendered = "\n".join(["Transcript:", *rendered_entries])

        if _context_units(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "recent_window": recent_window,
                "recent_start": recent_start,
                "rendered_entries": rendered_entries,
                **history_details,
            },
        )

    def render_history_for_delegate(self):
        """Render parent history with the same compression policy as prompts."""
        budget = int(self.section_budgets.get("history", 0) or 0)
        return self._render_history_section(budget).rendered

    def _compressed_history_entries(self, history, recent_start, stale_read_indexes=None):
        entries = []
        seen_older_reads = set()
        stale_read_indexes = set(stale_read_indexes or set())
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_read_summary_count": 0,
            "summarized_tool_count": 0,
            "stale_read_replacement_count": 0,
        }

        for index, item in enumerate(history):
            recent = index >= recent_start
            is_stale_read = index in stale_read_indexes
            if recent:
                lines = self._render_stale_read_history_item(item) if is_stale_read else self._render_history_item(item)
                if is_stale_read:
                    details["stale_read_replacement_count"] += 1
                entries.append(
                    {
                        "recent": True,
                        "lines": lines,
                    }
                )
                continue

            if item["role"] == "tool" and item["name"] == "read_file":
                read_key = self._read_history_key(item)
                if read_key in seen_older_reads:
                    details["collapsed_duplicate_reads"] += 1
                    continue
                seen_older_reads.add(read_key)
                if is_stale_read:
                    entries.append({"recent": False, "lines": self._render_stale_read_history_item(item)})
                    details["older_entries_count"] += 1
                    details["stale_read_replacement_count"] += 1
                    continue
                path = str(item["args"].get("path", "")).strip()
                summary = self._history_item_summary(item)
                if summary:
                    entries.append({"recent": False, "lines": [f"{path} -> {summary}"]})
                    details["older_entries_count"] += 1
                    details["reused_read_summary_count"] += 1
                    continue

            if item["role"] == "tool":
                summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line]})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue

            entries.append({"recent": False, "lines": self._render_history_item(item, 60)})

        return entries, details

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
        prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
        return [prefix, STALE_READ_MESSAGE]

    def _history_item_summary(self, item):
        summary = str(item.get("summary", "")).strip()
        if summary:
            return summary
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        return str(metadata.get("archive_summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        summary = self._history_item_summary(item)
        if summary:
            return f"{item['name']} -> {summary}"
        if item["name"] == "run_shell":
            command = str(item["args"].get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        return self._render_history_item(item, 60)[0]

    def _raw_history_text(self, history):
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(str(item["content"]))
            else:
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])

    def _render_history_item(self, item, line_limit=None):
        if item["role"] == "tool":
            prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
            content = str(item["content"]) if line_limit is None else _tail_clip(item["content"], max(20, line_limit))
            return [prefix, content]
        content = str(item["content"]) if line_limit is None else _tail_clip(item["content"], line_limit)
        return [f"[{item['role']}] {content}"]

    def _assemble_prompt(self, rendered):
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["durable_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, user_message, section_texts):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": None,
                "rendered_chars": rendered[section].rendered_chars,
                "raw_units": rendered[section].raw_units,
                "budget_units": int(budgets.get(section, 0)),
                "rendered_units": rendered[section].rendered_units,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            "raw_units": _context_units(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_units": None,
            "rendered_units": rendered[CURRENT_REQUEST_SECTION].rendered_units,
        }
        prompt_units = _context_units(prompt)
        return {
            "prompt_chars": len(prompt),
            "prompt_units": prompt_units,
            "prompt_budget_chars": None,
            "prompt_budget_units": self.total_budget,
            "prompt_budget_unit": "ascii_word_cjk_char_or_punctuation",
            "prompt_over_budget": prompt_units > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "section_budget_unit": "ascii_word_cjk_char_or_punctuation",
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "raw_units": rendered["history"].raw_units,
                "rendered_units": rendered["history"].rendered_units,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_read_summary_count": int(rendered["history"].details.get("reused_read_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
                "stale_read_replacement_count": int(rendered["history"].details.get("stale_read_replacement_count", 0)),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
                "raw_units": _context_units(user_message),
                "rendered_units": _context_units(user_message),
                "section_units": rendered[CURRENT_REQUEST_SECTION].rendered_units,
            },
        }
