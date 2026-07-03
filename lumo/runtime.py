"""Agent 运行时核心逻辑。

Pico 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import asyncio
import hashlib
import os
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from . import checkpoint as checkpointlib
from .background_tasks import (
    BackgroundTaskManager,
    format_task_list_text,
    format_task_output_text,
    format_task_start_text,
    format_task_stop_text,
)
from .features import memory as memorylib
from . import security as securitylib
from .context_manager import ContextManager, _context_units
from .checkpoint import CHECKPOINT_NONE_STATUS
from .prompt_prefix import build_prompt_prefix, tool_signature
from .run_store import RunStore
from .security import REDACTED_VALUE
from .session_store import SessionStore
from .tool_context import ToolContext
from .tool_executor import ToolExecutor, ToolExecutionResult, strip_tool_hints
from . import tools as toolkit
from .workspace import AGENT_STATE_DIR, IGNORED_PATH_NAMES, WorkspaceContext, clip, now

DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "context_reduction": True,
    "prompt_cache": True,
    "request_rewrite": False,
}
STALE_READ_MESSAGE = (
    "This earlier read_file output is stale because the file freshness no longer matches "
    "or the file was modified later in the transcript. "
    "Read the file again before relying on its contents."
)
DURABLE_MEMORY_HISTORY_UNIT_LIMIT = 150000
DURABLE_MEMORY_SCHEMA_VERSION = 1
DURABLE_MEMORY_TOPICS = tuple(memorylib.DURABLE_TOPIC_DEFAULTS.keys())
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")
REQUEST_REWRITE_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt" / "request_rewrite.md"
REQUEST_REWRITE_MAX_NEW_TOKENS = 600
REQUEST_REWRITE_MAX_CHAR_MULTIPLIER = 8
FINAL_TAG_PATTERN = re.compile(r"</?final>", re.I)
TOOL_BLOCK_PATTERN = re.compile(r"<tool\b[\s\S]*?(?:</tool>|$)", re.I)
TODO_UPDATE_BLOCK_PATTERN = re.compile(r"<todo_update\b[\s\S]*?</todo_update>", re.I)
REQUEST_PLAN_BLOCK_PATTERN = re.compile(r"<request_plan\b[\s\S]*?</request_plan>", re.I)
DISPLAY_BLOCK_PATTERN = re.compile(r"<display\b[\s\S]*?</display>", re.I)
WHITESPACE_PATTERN = re.compile(r"\s+")
ASSISTANT_TERMINAL_PREVIEW_LIMIT = 120
ASSISTANT_TERMINAL_FULL_TEXT_PATTERN = re.compile(
    r"(?i)(\?|\uFF1F|\b(error|failed|failure|blocker|blocked|permission denied|need your input|choose|which option|prefer)\b|错误|失败|报错|需要你|是否|要不要|哪一种|选哪个)"
)
FIRST_SENTENCE_PATTERN = re.compile(r"(.+?(?:[。！？!?](?=\s|$)|\.(?=\s|$)))")

__all__ = ["Pico", "SessionStore"]


class Pico:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=12,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        allowed_tools=None,
        tool_call_reporter=None,
        assistant_message_reporter=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.tool_call_reporter = tool_call_reporter
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / AGENT_STATE_DIR / "runs")
        self.background_tasks = BackgroundTaskManager(self.run_store, self.root)
        self.assistant_message_reporter = assistant_message_reporter
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self._apply_tool_allowlist(self.build_tools())
        self.tool_executor = ToolExecutor(self)
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self.last_user_request_rewrite = {}
        self.transient_todo_state = {
            "rewritten_request": "",
            "todos": [],
            "active_todo_id": "",
            "blocked_todo_id": "",
        }
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}
        durable_memory_evolution = self.session.setdefault("durable_memory_evolution", {})
        if not isinstance(durable_memory_evolution, dict):
            self.session["durable_memory_evolution"] = {}

    def current_runtime_identity(self):
        return checkpointlib.current_runtime_identity(self)

    def checkpoint_state(self):
        return checkpointlib.checkpoint_state(self)

    def current_checkpoint(self):
        return checkpointlib.current_checkpoint(self)

    def invalidate_stale_memory(self):



        return []

    def evaluate_resume_state(self):
        return checkpointlib.evaluate_resume_state(self)

    def render_checkpoint_text(self):
        return checkpointlib.render_checkpoint_text(self)

    def durable_memory_evolution_state(self):
        state = self.session.setdefault("durable_memory_evolution", {})
        if not isinstance(state, dict):
            state = {}
            self.session["durable_memory_evolution"] = state
        state.setdefault("schema_version", DURABLE_MEMORY_SCHEMA_VERSION)
        state.setdefault("history_hash", "")
        state.setdefault("workspace_snapshot_hash", "")
        state.setdefault("last_evolved_at", "")
        state.setdefault("last_reason", "")
        state.setdefault("last_promotions", [])
        state.setdefault("last_superseded", [])
        state.setdefault("last_history_units", 0)
        state.setdefault("last_history_excerpt_units", 0)
        state.setdefault("last_workspace_fingerprint", "")
        return state

    def durable_memory_index_path(self):
        return Path(self.root) / AGENT_STATE_DIR / "memory" / "MEMORY.md"

    def durable_memory_topics_dir(self):
        return Path(self.root) / AGENT_STATE_DIR / "memory" / "topics"

    def durable_memory_text(self):
        durable_store = memorylib.DurableMemoryStore(Path(self.root) / AGENT_STATE_DIR / "memory")
        lines = ["Durable memory:"]
        topics = durable_store.load_index()
        if not topics:
            lines.append("- none")
            return "\n".join(lines)
        for topic in topics:
            topic_name = str(topic.get("topic", "")).strip()
            title = str(topic.get("title", "")).strip() or topic_name
            lines.append(f"- {topic_name}: {title}")
            summary = str(topic.get("summary", "")).strip()
            if summary:
                lines.append(f"  - summary: {summary}")
            tags = [str(tag).strip() for tag in topic.get("tags", []) if str(tag).strip()]
            if tags:
                lines.append(f"  - tags: {', '.join(tags)}")
            notes = durable_store.load_topic_notes(topic_name)
            if notes:
                lines.append("  - notes:")
                for note in notes:
                    note_text = str(note.get("text", "")).strip()
                    if note_text:
                        lines.append(f"    - {note_text}")
        return "\n".join(lines)

    def _durable_memory_store(self):
        return memorylib.DurableMemoryStore(Path(self.root) / AGENT_STATE_DIR / "memory")

    def _durable_memory_topics_text(self):
        durable_store = self._durable_memory_store()
        topics = durable_store.load_index()
        if not topics:
            return "Existing durable memory:\n- none"
        lines = ["Existing durable memory:"]
        for topic in topics:
            lines.append(f"- {topic['topic']}: {topic['title']}")
            summary = str(topic.get("summary", "")).strip()
            if summary:
                lines.append(f"  - summary: {summary}")
            tags = [str(tag).strip() for tag in topic.get("tags", []) if str(tag).strip()]
            if tags:
                lines.append(f"  - tags: {', '.join(tags)}")
            notes = durable_store.load_topic_notes(topic["topic"])
            if notes:
                lines.append("  - notes:")
                for note in notes:
                    note_text = str(note.get("text", "")).strip()
                    if note_text:
                        lines.append(f"    - {note_text}")
        return "\n".join(lines)

    def _durable_memory_history_excerpt(self):
        history = list(self.session.get("history", []))
        if not history:
            return "Transcript:\n- empty"
        rendered = json.dumps(history, indent=2, ensure_ascii=False)
        return self._tail_clip_units(rendered, DURABLE_MEMORY_HISTORY_UNIT_LIMIT).strip() or "[]"

    @staticmethod
    def _tail_clip_units(text, limit):
        text = str(text)
        if _context_units(text) <= limit:
            return text
        prefix = "...[truncated to recent durable-memory history window]\n"
        target = max(1, int(limit) - _context_units(prefix))
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high) // 2
            if _context_units(text[mid:]) > target:
                low = mid + 1
            else:
                high = mid
        return prefix + text[low:].lstrip()

    def _workspace_snapshot_hash(self):
        snapshot = self.capture_workspace_snapshot()
        payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _text_hash(text):
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def _durable_memory_evolution_prompt(self, reason, history_excerpt, durable_text):
        return "\n\n".join(
            [
                "You are reviewing a coding agent session for durable memory evolution.",
                "Only promote stable facts that should survive into future sessions.",
                "Permanent memory should be short, ideally one sentence each.",
                "Return JSON only in this shape:",
                '{"updates":[{"topic":"project-conventions","note":"Use pytest for tests."}]}',
                'If nothing should be added or changed, return {"updates":[]}.',
                'If durable_text already contains the same or a clearly similar durable fact, return {"updates":[]}.',
                "Allowed topics: project-conventions, key-decisions, dependency-facts, user-preferences.",
                "Update rules:",
                "- project-conventions: repo-wide or workflow rules that stay valid.",
                "- key-decisions: design choices and rationale anchors.",
                "- dependency-facts: stable dependency, backend, toolchain, or environment facts.",
                "- user-preferences: stable preferences the user will likely want remembered.",
                "Reject transient task state, temporary blockers, file positions, raw or noisy tool outputs, secrets, and long summaries, but keep stable facts learned from them.",
                "Good examples:",
                "- {\"topic\":\"project-conventions\",\"note\":\"Use pytest for tests.\"}",
                "- {\"topic\":\"key-decisions\",\"note\":\"Keep durable memory topic-based and lightweight.\"}",
                "- {\"topic\":\"user-preferences\",\"note\":\"The user prefers Chinese explanations with concrete examples.\"}",
                "Bad examples:",
                "- {\"topic\":\"key-decisions\",\"note\":\"Current blocker is a 401 when calling the API.\"}",
                "- {\"topic\":\"dependency-facts\",\"note\":\"stdout: FAIL test_one FAIL test_two.\"}",
                f"Trigger reason: {reason}",
                durable_text,
                "Session history excerpt:",
                history_excerpt,
            ]
        )

    @staticmethod
    def _extract_json_object(text):
        text = str(text).strip()
        if not text:
            return {}
        if "<final>" in text:
            text = Pico.extract(text, "final").strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _normalize_durable_updates(self, payload):
        updates = []
        allowed_topics = set(DURABLE_MEMORY_TOPICS)
        for item in payload.get("updates", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            topic = str(item.get("topic", "")).strip()
            note_text = str(item.get("note", "")).strip()
            if topic not in allowed_topics:
                continue
            if self.reject_durable_reason(note_text):
                continue
            if note_text:
                updates.append((topic, note_text))
        return updates

    def should_evolve_durable_memory(self, reason=None):
        state = self.durable_memory_evolution_state()
        history_excerpt = self._durable_memory_history_excerpt()
        history_hash = self._text_hash(history_excerpt)
        workspace_snapshot_hash = self._workspace_snapshot_hash()
        if not list(self.session.get("history", [])):
            return False, {
                "reason": reason or "",
                "history_hash": history_hash,
                "workspace_snapshot_hash": workspace_snapshot_hash,
                "history_excerpt": history_excerpt,
            }
        changed = (
            history_hash != str(state.get("history_hash", "")).strip()
            or workspace_snapshot_hash != str(state.get("workspace_snapshot_hash", "")).strip()
        )
        return changed, {
            "reason": reason or "",
            "history_hash": history_hash,
            "workspace_snapshot_hash": workspace_snapshot_hash,
            "history_excerpt": history_excerpt,
        }

    def evolve_durable_memory(self, reason="session_end", force=False):
        state = self.durable_memory_evolution_state()
        changed, snapshot = self.should_evolve_durable_memory(reason)
        if not force and not changed:
            state["last_reason"] = reason
            state["last_status"] = "skipped"
            self.session["durable_memory_evolution"] = state
            self.session_path = self.session_store.save(self.session)
            return {
                "status": "skipped",
                "reason": reason,
                "changed": False,
                "promotions": [],
                "superseded": [],
            }

        prompt = self._durable_memory_evolution_prompt(
            reason=reason,
            history_excerpt=snapshot["history_excerpt"],
            durable_text=self._durable_memory_topics_text(),
        )
        try:
            raw = self.model_client.complete(prompt, self.max_new_tokens)
            payload = self._extract_json_object(raw)
            updates = self._normalize_durable_updates(payload)
            if not updates:
                state.update(
                    {
                        "schema_version": DURABLE_MEMORY_SCHEMA_VERSION,
                        "history_hash": snapshot["history_hash"],
                        "workspace_snapshot_hash": snapshot["workspace_snapshot_hash"],
                        "last_evolved_at": now(),
                        "last_reason": reason,
                        "last_status": "no_changes",
                        "last_promotions": [],
                        "last_superseded": [],
                        "last_history_units": int(_context_units(json.dumps(self.session.get("history", []), ensure_ascii=False))),
                        "last_history_excerpt_units": int(_context_units(snapshot["history_excerpt"])),
                        "last_workspace_fingerprint": getattr(self.workspace, "fingerprint", lambda: "")(),
                    }
                )
                self.session["durable_memory_evolution"] = state
                self.session_path = self.session_store.save(self.session)
                return {
                    "status": "no_changes",
                    "reason": reason,
                    "changed": True,
                    "promotions": [],
                    "superseded": [],
                    "prompt": prompt,
                    "raw": raw,
                }
            promoted, superseded = self.memory.promote_durable(updates)
        except Exception as exc:
            state.update(
                {
                    "schema_version": DURABLE_MEMORY_SCHEMA_VERSION,
                    "last_evolved_at": now(),
                    "last_reason": reason,
                    "last_status": "failed",
                    "last_error": str(exc),
                    "history_hash": snapshot["history_hash"],
                    "workspace_snapshot_hash": snapshot["workspace_snapshot_hash"],
                }
            )
            self.session["durable_memory_evolution"] = state
            self.session_path = self.session_store.save(self.session)
            return {
                "status": "failed",
                "reason": reason,
                "changed": True,
                "error": str(exc),
                "promotions": [],
                "superseded": [],
            }
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_superseded = superseded
        state.update(
            {
                "schema_version": DURABLE_MEMORY_SCHEMA_VERSION,
                "history_hash": snapshot["history_hash"],
                "workspace_snapshot_hash": snapshot["workspace_snapshot_hash"],
                "last_evolved_at": now(),
                "last_reason": reason,
                "last_status": "updated",
                "last_promotions": list(promoted),
                "last_superseded": list(superseded),
                "last_history_units": int(_context_units(json.dumps(self.session.get("history", []), ensure_ascii=False))),
                "last_history_excerpt_units": int(_context_units(snapshot["history_excerpt"])),
                "last_workspace_fingerprint": getattr(self.workspace, "fingerprint", lambda: "")(),
            }
        )
        self.session["durable_memory_evolution"] = state
        self.session_path = self.session_store.save(self.session)
        if self.current_task_state is not None:
            self.run_store.write_report(self.current_task_state, self.redact_artifact(self.build_report(self.current_task_state)))
        return {
            "status": "updated",
            "reason": reason,
            "changed": True,
            "promotions": promoted,
            "superseded": superseded,
            "prompt": prompt,
            "raw": raw,
        }

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self.tool_context())

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    def _apply_tool_allowlist(self, tools):
        if self.allowed_tools is None:
            return tools
        legal_names = toolkit.legal_tool_names()
        unknown = [name for name in self.allowed_tools if name not in legal_names]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {
            name: tool
            for name, tool in tools.items()
            if name in allowed
        }

    def tool_signature(self):
        return tool_signature(self.tools)

    def build_prefix(self):
        return build_prompt_prefix(workspace=self.workspace, tools=self.tools)

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    """
    prefix 是 prompt 前面那段比较稳定的内容，主要包括：
    系统规则 Rules
    工具说明 Tools
    示例 Valid response examples
    工作区摘要 Workspace
    """
    """
    refresh_prefix() 的职责就是：
    看看上一次的 prefix 指纹是什么
    重新读取当前工作区信息
    判断工作区是否变了
    如果变了，重建 prefix
    记录这次刷新结果
    """
    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)



        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        return self.durable_memory_text()

    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        history = list(self._iter_history_items_for_prompt(history))
        lines = []
        seen_reads = set()
        stale_read_indexes = self._stale_read_indexes(history)
        latest_tool_reminders = self._latest_tool_reminder_indexes(history)
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                read_key = self._history_read_key(item)
                if read_key in seen_reads:
                    continue
                seen_reads.add(read_key)

            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                if index in stale_read_indexes:
                    lines.append(STALE_READ_MESSAGE)
                else:
                    if item["name"] in {"git_status", "git_diff"}:
                        lines.append(self._history_item_display_content(item))
                    else:
                        summary = self._history_item_summary(item)
                        lines.append(summary or str(item["content"]))
                    if latest_tool_reminders.get(self._tool_reminder_key(item)) == index:
                        reminder = self._history_item_tool_reminder(item)
                        if reminder:
                            lines.append(f"<tool_reminder>{reminder}</tool_reminder>")
            else:
                lines.append(f"[{item['role']}] {item['content']}")

        return "\n".join(lines)

    @staticmethod
    def _history_item_summary(item):
        summary = str(item.get("summary", "")).strip()
        if summary:
            return summary
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        return str(metadata.get("archive_summary", "")).strip()

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
            key = Pico._tool_reminder_key(item)
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
        if name == "task_output":
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            task_id = str(args.get("task_id", "")).strip()
            stream = str(args.get("stream", "stdout")).strip() or "stdout"
            if task_id:
                return (name, task_id, stream)
        return None

    @staticmethod
    def _history_read_key(item):
        args = item.get("args", {})
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
        offset = read_window.get("start_line", args.get("offset", args.get("start", "")))
        limit = read_window.get("requested_lines", args.get("limit", args.get("end", "")))
        return (str(args.get("path", "")).strip(), str(offset), str(limit))

    def _history_path_key(self, item):
        args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
        path = str(args.get("path", "")).strip()
        if not path:
            return ""
        try:
            raw_path = Path(path)
            if not raw_path.is_absolute():
                raw_path = Path(self.root) / raw_path
            resolved = raw_path.resolve()
            try:
                return resolved.relative_to(Path(self.root).resolve()).as_posix()
            except ValueError:
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
        current = memorylib.file_freshness(path_key, self.root)
        return expected != current

    def prompt_visible_history(self):
        return self.context_manager._prompt_history()

    @staticmethod
    def _merge_line_ranges(ranges):
        ordered = sorted(
            (
                (int(start), int(end), bool(has_more), int(known_line_floor))
                for start, end, has_more, known_line_floor in ranges
                if int(start) >= 1 and int(end) >= int(start)
            ),
            key=lambda item: (item[0], item[1]),
        )
        merged = []
        for start, end, has_more, known_line_floor in ordered:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end, has_more, known_line_floor])
                continue
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = merged[-1][2] or has_more
            merged[-1][3] = max(merged[-1][3], known_line_floor)
        return [tuple(item) for item in merged]

    def _prompt_visible_read_coverage(self, path):
        target_path = self.path(path)
        path_key = self._history_path_key({"args": {"path": str(target_path)}})
        current_freshness = memorylib.file_freshness(path_key, self.root)
        history = self.prompt_visible_history()
        ranges = []
        for item in history:
            if item.get("role") != "tool" or item.get("name") != "read_file":
                continue
            if self._history_path_key(item) != path_key:
                continue
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            if str(metadata.get("freshness", "")).strip() != current_freshness:
                continue
            read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
            start_line = int(read_window.get("start_line", 0) or 0)
            end_line = int(read_window.get("end_line", 0) or 0)
            if start_line < 1 or end_line < start_line:
                continue
            ranges.append(
                (
                    start_line,
                    end_line,
                    bool(read_window.get("has_more")),
                    int(read_window.get("known_line_floor", end_line) or end_line),
                )
            )
        merged = self._merge_line_ranges(ranges)
        fully_read = bool(merged) and len(merged) == 1 and merged[0][0] == 1 and not merged[0][2]
        return {
            "path": target_path.relative_to(self.root).as_posix(),
            "path_key": path_key,
            "freshness": current_freshness,
            "ranges": merged,
            "fully_read": fully_read,
        }

    @staticmethod
    def _first_unread_after_overlap(ranges, start_line):
        candidate = int(start_line)
        changed = True
        while changed:
            changed = False
            for covered_start, covered_end, _has_more, _known_line_floor in ranges:
                if covered_start <= candidate <= covered_end:
                    candidate = covered_end + 1
                    changed = True
        return candidate

    def auto_continue_read_file_args(self, args):
        args = dict(args or {})
        path = str(args.get("path", "")).strip()
        if not path:
            return {"status": "noop", "args": args}
        offset, limit = toolkit._read_window_args(args)
        coverage = self._prompt_visible_read_coverage(path)
        if not coverage["ranges"]:
            return {"status": "noop", "args": args, "coverage": coverage}
        next_offset = self._first_unread_after_overlap(coverage["ranges"], offset)
        if next_offset == offset:
            return {"status": "noop", "args": args, "coverage": coverage}
        if coverage["fully_read"]:
            return {
                "status": "fully_read",
                "args": args,
                "coverage": coverage,
                "requested_offset": offset,
                "effective_offset": next_offset,
                "limit": limit,
            }
        rewritten = dict(args)
        rewritten["offset"] = next_offset
        rewritten["limit"] = limit
        rewritten.pop("start", None)
        rewritten.pop("end", None)
        return {
            "status": "continued",
            "args": rewritten,
            "coverage": coverage,
            "requested_offset": offset,
            "effective_offset": next_offset,
            "limit": limit,
        }

    def synthetic_fully_read_result(self, path, requested_offset, limit, coverage):
        relative_path = coverage.get("path") or self.path(path).relative_to(self.root).as_posix()
        ranges = coverage.get("ranges", [])
        covered_until = max((item[1] for item in ranges), default=max(int(requested_offset) - 1, 0))
        freshness = str(coverage.get("freshness", "")).strip()
        content = "\n".join(
            [
                f"# {relative_path}",
                f"# already fully read through line {covered_until}",
                (
                    f"<tool_reminder>{relative_path} has already been fully read in the current prompt-visible transcript "
                    f"through line {covered_until}. Requested reread from line {requested_offset} with limit {limit} was skipped.</tool_reminder>"
                ),
                (
                    f"<summary-for-history>{relative_path} was already fully read in the visible transcript; "
                    f"skipped duplicate reread request from line {requested_offset}.</summary-for-history>"
                ),
            ]
        )
        metadata = {
            "tool_status": "ok",
            "tool_error_code": "",
            "security_event_type": "",
            "risk_level": "low",
            "read_only": True,
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
            "workspace_fingerprint": self.workspace.fingerprint(),
            "archive_summary": (
                f"{relative_path} was already fully read in the visible transcript; "
                f"skipped duplicate reread request from line {requested_offset}."
            ),
            "tool_reminder": (
                f"{relative_path} has already been fully read in the current prompt-visible transcript through line {covered_until}. "
                f"Requested reread from line {requested_offset} with limit {limit} was skipped."
            ),
            "read_window": {
                "start_line": int(requested_offset),
                "end_line": int(covered_until),
                "known_line_floor": int(covered_until),
                "returned_lines": 0,
                "requested_lines": int(limit),
                "has_more": False,
            },
            "freshness": freshness,
            "auto_read_handling": "fully_read_skip",
        }
        return ToolExecutionResult(content=content, metadata=metadata)

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def is_runtime_notice_text(text):
        return str(text or "").startswith("Runtime notice:")

    def should_record_history_item(self, item):
        if not isinstance(item, dict):
            return True
        if item.get("role") != "assistant":
            return True
        content = str(item.get("content", ""))
        if not self.is_runtime_notice_text(content):
            return True
        history = self.session.get("history", [])
        if not history:
            return True
        last_item = history[-1]
        if not isinstance(last_item, dict):
            return True
        if last_item.get("role") != "assistant":
            return True
        return str(last_item.get("content", "")) != content

    def record_history_item(self, item):
        if not self.should_record_history_item(item):
            return False
        self.record(item)
        return True

    def _iter_history_items_for_prompt(self, history):
        previous_runtime_notice = None
        for item in history:
            if not isinstance(item, dict):
                previous_runtime_notice = None
                yield item
                continue
            if item.get("role") == "assistant":
                content = str(item.get("content", ""))
                if self.is_runtime_notice_text(content):
                    if content == previous_runtime_notice:
                        continue
                    previous_runtime_notice = content
                    yield item
                    continue
            previous_runtime_notice = None
            yield item

    @staticmethod
    def looks_sensitive_env_name(name):
        return securitylib.looks_sensitive_env_name(name)

    def is_secret_env_name(self, name):
        return securitylib.is_secret_env_name(name, secret_env_names=self.secret_env_names)

    def configured_secret_env_items(self):
        return securitylib.configured_secret_env_items(secret_env_names=self.secret_env_names)

    def detected_secret_env_items(self):
        return securitylib.detected_secret_env_items(secret_env_names=self.secret_env_names)

    def secret_env_summary(self):
        return securitylib.secret_env_summary(secret_env_names=self.secret_env_names)

    def detected_secret_env_summary(self):
        return securitylib.detected_secret_env_summary(secret_env_names=self.secret_env_names)

    def redact_text(self, text):
        return securitylib.redact_text(text, secret_env_names=self.secret_env_names)

    def redact_artifact(self, value, key=None):
        return securitylib.redact_artifact(value, key=key, secret_env_names=self.secret_env_names)

    def tool_call_summary(self, name, args):
        args = args if isinstance(args, dict) else {}
        redacted_args = self.redact_artifact(args)
        if name == "read_file":
            return self._read_file_tool_target_summary(redacted_args)
        if name in {"write_file", "patch_file", "run_shell", "run_shell_bg", "task_stop", "delegate"}:
            return self.redact_text(self.approval_summary(name, redacted_args))
        return json.dumps(redacted_args, ensure_ascii=False, sort_keys=True)

    def report_tool_call(self, name, args):
        reporter = getattr(self, "tool_call_reporter", None)
        if reporter is None:
            return
        reporter(name, self.tool_call_summary(name, args))

    def report_tool_result(self, name, args, metadata, content=""):
        reporter = getattr(self, "tool_call_reporter", None)
        if reporter is None:
            return
        summary = self.tool_result_summary(name, args, metadata, content=content)
        summary = self.redact_text(str(summary or "").strip())
        if not summary:
            return
        reporter(name, f"-> {summary}")

    def tool_result_summary(self, name, args, metadata, content=""):
        args = args if isinstance(args, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        if name == "read_file":
            return self._read_file_tool_result_summary(args, metadata, content=content)
        return ""

    def _display_tool_path(self, path):
        raw = str(path or "").strip()
        if not raw:
            return ""
        try:
            resolved = self.path(raw)
            return resolved.relative_to(self.root).as_posix()
        except Exception:
            return raw.replace("\\", "/").lstrip("./")

    def _read_file_tool_target_summary(self, args):
        path = self._display_tool_path(args.get("path", ""))
        return path or json.dumps(args, ensure_ascii=False, sort_keys=True)

    def _read_file_tool_result_summary(self, args, metadata, content=""):
        if str(metadata.get("tool_status", "")).strip() not in {"", "ok"}:
            return ""
        auto_handling = str(metadata.get("auto_read_handling", "")).strip()
        if auto_handling == "fully_read_skip":
            return "already fully read, skipped reread"

        normalized_content = str(content or "").strip().lower()
        if "unchanged since last read" in normalized_content:
            return "unchanged since last read"

        read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
        returned_lines = int(read_window.get("returned_lines", 0) or 0)
        start_line = int(read_window.get("start_line", args.get("offset", args.get("start", 1)) or 1) or 1)
        end_line = int(read_window.get("end_line", 0) or 0)
        has_more = bool(read_window.get("has_more", False))
        path = self._display_tool_path(args.get("path", ""))
        if returned_lines > 0 and end_line >= start_line:
            summary = f"{returned_lines} lines from {path} ({start_line}-{end_line})" if path else f"{returned_lines} lines ({start_line}-{end_line})"
        elif path:
            summary = f"0 lines from {path}"
        else:
            summary = "0 lines"
        if has_more:
            summary += ", more remains"
        return summary

    def report_assistant_message(self, text, *, compact=True):
        reporter = getattr(self, "assistant_message_reporter", None)
        message = self._normalize_assistant_terminal_message(text)
        message = self.redact_text(message)
        if reporter is None or not message:
            return
        if compact and not self._should_keep_full_assistant_terminal_message(message):
            message = self._compact_assistant_terminal_message(message)
        if not message:
            return
        reporter(message)

    @staticmethod
    def _normalize_assistant_terminal_message(text):
        value = str(text or "").strip()
        if not value:
            return ""
        if "<final>" in value:
            extracted = Pico.extract_answer_text(value)
            if extracted:
                value = extracted
        value = TOOL_BLOCK_PATTERN.sub(" ", value)
        value = TODO_UPDATE_BLOCK_PATTERN.sub(" ", value)
        value = REQUEST_PLAN_BLOCK_PATTERN.sub(" ", value)
        value = DISPLAY_BLOCK_PATTERN.sub(" ", value)
        value = FINAL_TAG_PATTERN.sub(" ", value)
        value = WHITESPACE_PATTERN.sub(" ", value).strip()
        return value

    @staticmethod
    def _compact_assistant_terminal_message(text):
        value = str(text or "").strip()
        if not value:
            return ""
        sentence_match = FIRST_SENTENCE_PATTERN.match(value)
        preview = sentence_match.group(1).strip() if sentence_match else value
        if len(preview) > ASSISTANT_TERMINAL_PREVIEW_LIMIT:
            return preview[: ASSISTANT_TERMINAL_PREVIEW_LIMIT - 3].rstrip() + "..."
        return preview

    @staticmethod
    def _should_keep_full_assistant_terminal_message(text):
        value = str(text or "").strip()
        if not value:
            return False
        return bool(ASSISTANT_TERMINAL_FULL_TEXT_PATTERN.search(value))

    def set_transient_todo_state(self, rewritten_request="", todos=None, active_todo_id="", blocked_todo_id=""):
        self.transient_todo_state = {
            "rewritten_request": str(rewritten_request or ""),
            "todos": list(todos or []),
            "active_todo_id": str(active_todo_id or ""),
            "blocked_todo_id": str(blocked_todo_id or ""),
        }

    def clear_transient_todo_state(self):
        self.set_transient_todo_state()

    def current_todo_request_text(self):
        state = dict(getattr(self, "transient_todo_state", {}) or {})
        rewritten_request = str(state.get("rewritten_request", "")).strip()
        todos = list(state.get("todos", []) or [])
        active_todo_id = str(state.get("active_todo_id", "")).strip()
        if not rewritten_request and not todos:
            return ""
        lines = []
        if rewritten_request:
            lines.append("Rewritten request:")
            lines.append(rewritten_request)
        if todos:
            done_count = sum(1 for item in todos if str(item.get("status", "")).strip() == "done")
            lines.append(f"Todo progress: Done {done_count}/{len(todos)}")
        unfinished_todos = [item for item in todos if str(item.get("status", "")).strip() != "done"]
        active_todo = None
        for item in unfinished_todos:
            if str(item.get("id", "")).strip() == active_todo_id:
                active_todo = item
                break
        if active_todo is not None:
            lines.append("")
            lines.append("Current focus todo:")
            lines.append(f'- [{active_todo_id}] {str(active_todo.get("text", "")).strip()}')
        remaining_todos = [
            item for item in unfinished_todos if str(item.get("id", "")).strip() != active_todo_id
        ]
        if remaining_todos:
            lines.append("")
            lines.append("Remaining todos:")
            for item in remaining_todos:
                todo_id = str(item.get("id", "")).strip()
                todo_text = str(item.get("text", "")).strip()
                lines.append(f"- [{todo_id}] {todo_text}")
        blocked_todo_id = str(state.get("blocked_todo_id", "")).strip()
        if blocked_todo_id:
            lines.append(f"Blocked todo: {blocked_todo_id}")
        return "\n".join(lines).strip()

    @staticmethod
    def _artifact_like_path(path):
        normalized = str(path or "").strip().replace("\\", "/").lstrip("./")
        return normalized.endswith(".patch") and "/runs/" in normalized

    def render_obligation_progress_message(self, obligation):
        if not isinstance(obligation, dict):
            return ""
        required_tool = str(obligation.get("required_tool", "")).strip()
        if not required_tool:
            return ""
        args = obligation.get("suggested_args", {})
        args = args if isinstance(args, dict) else {}
        reason = str(obligation.get("reason", "")).strip()

        if required_tool == "git_status":
            path = str(args.get("path", ".")).strip() or "."
            return f"继续执行：git_status(path={path})" if path != "." else "继续执行：git_status"

        if required_tool == "git_diff":
            path = str(args.get("path", ".")).strip() or "."
            mode = str(args.get("mode", "workspace")).strip() or "workspace"
            details = [f"mode={mode}"]
            if path != ".":
                details.append(f"path={path}")
            return f"继续执行：git_diff({', '.join(details)})"

        if required_tool == "read_file":
            path = str(args.get("path", "")).strip()
            offset = args.get("offset")
            if reason == "git_diff_externalized" or self._artifact_like_path(path):
                return f"继续核对补丁文件：{path}" if path else "继续核对补丁文件"
            if path and offset:
                return f"继续读取：{path}（从第 {offset} 行）"
            if path:
                return f"继续读取：{path}"
            return "继续执行：read_file"

        if required_tool == "task_output":
            task_id = str(args.get("task_id", "")).strip()
            if task_id:
                return f"继续等待后台任务：{task_id}"
            return "继续执行：task_output"

        if args:
            summary = self.tool_call_summary(required_tool, args)
            return f"继续执行：{required_tool} {summary}"
        return f"继续执行：{required_tool}"

    def shell_env(self):
        return securitylib.shell_env(allowlist=self.shell_env_allowlist, root=self.root)

    async def complete_text_async(self, prompt, max_new_tokens=None, prompt_cache_key=None, prompt_cache_retention=None):
        token_limit = int(max_new_tokens or self.max_new_tokens)
        complete_async = getattr(self.model_client, "complete_async", None)
        if complete_async is not None:
            return await complete_async(
                prompt,
                token_limit,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
        return await asyncio.to_thread(
            self.model_client.complete,
            prompt,
            token_limit,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

    def _request_rewrite_template(self):
        return REQUEST_REWRITE_TEMPLATE_PATH.read_text(encoding="utf-8")

    def _request_rewrite_prompt(self, user_message):
        user_text = str(user_message or "")
        max_chars = max(len(user_text), len(user_text) * REQUEST_REWRITE_MAX_CHAR_MULTIPLIER)
        return (
            self._request_rewrite_template()
            .replace("{{MAX_CHARS}}", str(max_chars))
            .replace("{{USER_REQUEST}}", user_text)
        )

    @staticmethod
    def _fallback_request_plan(original_text, error=""):
        text = str(original_text or "").strip()
        return {
            "rewritten_request": text,
            "todos": [{"id": "t1", "status": "active", "text": text or "Handle the current user request"}],
            "active_todo_id": "t1",
            "valid": False,
            "error": str(error or "").strip(),
            "raw_text": "",
        }

    @staticmethod
    def _normalize_request_plan_text(text, max_chars):
        normalized = str(text or "").strip()
        fenced = re.search(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", normalized, re.S)
        if fenced:
            normalized = fenced.group(1).strip()
        if len(normalized) > max_chars:
            normalized = normalized[:max_chars].rstrip()
        return normalized

    @staticmethod
    def parse_request_plan(raw_text, original_text):
        original_text = str(original_text or "")
        fallback = Pico._fallback_request_plan(original_text)
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            fallback["error"] = "empty_request_plan"
            return fallback
        match = REQUEST_PLAN_BLOCK_PATTERN.search(raw_text)
        if not match:
            fallback["error"] = "missing_request_plan_block"
            fallback["raw_text"] = raw_text
            return fallback
        try:
            root = ET.fromstring(match.group(0))
        except Exception as exc:
            fallback["error"] = f"invalid_request_plan_xml:{exc}"
            fallback["raw_text"] = raw_text
            return fallback
        rewritten_node = root.find("rewritten_request")
        todo_list_node = root.find("todo_list")
        max_chars = max(len(original_text), len(original_text) * REQUEST_REWRITE_MAX_CHAR_MULTIPLIER)
        rewritten_request = Pico._normalize_request_plan_text(
            rewritten_node.text if rewritten_node is not None else original_text,
            max_chars=max_chars,
        ) or original_text
        todos = []
        active_ids = []
        seen_ids = set()
        if todo_list_node is not None:
            for todo_node in todo_list_node.findall("todo"):
                todo_id = str(todo_node.attrib.get("id", "")).strip()
                status = str(todo_node.attrib.get("status", "")).strip().lower()
                todo_text = Pico._normalize_request_plan_text("".join(todo_node.itertext()), max_chars=max_chars)
                if not todo_id or todo_id in seen_ids:
                    fallback["error"] = "duplicate_or_missing_todo_id"
                    fallback["raw_text"] = raw_text
                    return fallback
                if status not in {"active", "pending"}:
                    fallback["error"] = "invalid_initial_todo_status"
                    fallback["raw_text"] = raw_text
                    return fallback
                if not todo_text:
                    fallback["error"] = "empty_todo_text"
                    fallback["raw_text"] = raw_text
                    return fallback
                seen_ids.add(todo_id)
                todos.append({"id": todo_id, "status": status, "text": todo_text})
                if status == "active":
                    active_ids.append(todo_id)
        if not todos or len(active_ids) != 1:
            fallback["error"] = "invalid_initial_todo_shape"
            fallback["raw_text"] = raw_text
            return fallback
        return {
            "rewritten_request": rewritten_request,
            "todos": todos,
            "active_todo_id": active_ids[0],
            "valid": True,
            "error": "",
            "raw_text": raw_text,
        }

    async def rewrite_user_message_async(self, user_message):
        original_text = str(user_message or "")
        metadata = {
            "enabled": bool(self.feature_enabled("request_rewrite")),
            "applied": False,
            "changed": False,
            "original_chars": len(original_text),
            "rewritten_chars": len(original_text),
            "max_chars": len(original_text) * REQUEST_REWRITE_MAX_CHAR_MULTIPLIER,
            "template_path": str(REQUEST_REWRITE_TEMPLATE_PATH),
            "error": "",
            "todo_count": 1,
            "active_todo_id": "t1",
        }
        self.last_user_request_rewrite = dict(metadata)
        if not metadata["enabled"] or not original_text.strip():
            return self._fallback_request_plan(original_text)
        try:
            prompt = self._request_rewrite_prompt(original_text)
            raw = await self.complete_text_async(prompt, max_new_tokens=REQUEST_REWRITE_MAX_NEW_TOKENS)
            plan = self.parse_request_plan(raw, original_text)
            metadata.update(
                {
                    "applied": True,
                    "changed": str(plan.get("rewritten_request", "")) != original_text,
                    "rewritten_chars": len(str(plan.get("rewritten_request", ""))),
                    "todo_count": len(plan.get("todos", []) or []),
                    "active_todo_id": str(plan.get("active_todo_id", "")).strip(),
                    "error": str(plan.get("error", "")).strip(),
                }
            )
            self.last_user_request_rewrite = dict(metadata)
            return plan
        except Exception as exc:
            metadata["error"] = str(exc)
            self.last_user_request_rewrite = dict(metadata)
            return self._fallback_request_plan(original_text, error=str(exc))

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._build_prompt_and_metadata_async(user_message))

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(self._build_prompt_and_metadata_async(user_message))
            except BaseException as exc:
                result["error"] = exc

        import threading

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def _build_prompt_and_metadata_async(self, user_message):
        """
        {
            "workspace_changed": True/False,
            "prefix_changed": True/False,
        }"""
        refresh = self.refresh_prefix()
        """
        先让 memory 里过期的 file summaries 失效
        取当前 checkpoint
        如果有 checkpoint：
            检查 schema version 是否匹配
            检查 key files 的 freshness 是否还一致
            检查 runtime identity 是否变化
        最后给出一个 resume_state
        {
            "status": "partial-stale",
            "stale_paths": ["pico/cli.py"],
            "runtime_identity_mismatch_fields": [],
            "stale_summary_invalidations": 1
        }
        """
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = await self.context_manager.build_async(user_message)


        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "durable_memory_chars": len(self.memory_text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "request_rewrite": dict(self.last_user_request_rewrite or {}),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()

        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        return checkpointlib.create_checkpoint(self, task_state, user_message, trigger)

    def infer_next_step(self, task_state):
        return checkpointlib.infer_next_step(task_state)

    def update_memory_after_tool(self, name, args, result, metadata=None):
        """Compatibility hook.

        Short-term tool facts now live in session history. Durable cross-session
        facts are promoted by the durable-memory evolution flow, so this method
        intentionally no longer mirrors read_file summaries or process events
        into session memory.
        """
        return None

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):


        return None

    def reject_durable_reason(self, note_text):
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return []
        promotions = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        break
                    promotions.append((topic, note_text))
                break
        return promotions

    def promote_durable_memory(self, user_message, final_answer):
        promotions = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_superseded = superseded
        return promoted, superseded

    def ask(self, user_message):
        from .agent_loop import AgentLoop

        return AgentLoop(self).run(user_message)

    async def ask_async(self, user_message):
        from .agent_loop import AgentLoop

        return await AgentLoop(self).run_async(user_message)

    def execute_tool(self, name, args, execution_context=None):
        result = self.tool_executor.execute(name, args, execution_context=execution_context)
        self._last_tool_result_metadata = dict(result.metadata)
        return result

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        return self.execute_tool(name, args).content

    def repeated_tool_call(self, name, args):
        if name == "task_output":
            return False


        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_background_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        background_tasks = self.background_tasks.summarize_run_tasks(task_state.run_id, limit=8)
        logical_steps = int(getattr(task_state, "logical_steps", getattr(task_state, "tool_steps", 0)) or 0)
        raw_tool_calls = int(getattr(task_state, "raw_tool_calls", getattr(task_state, "tool_steps", 0)) or 0)
        raw_attempts = int(getattr(task_state, "raw_attempts", getattr(task_state, "attempts", 0)) or 0)
        last_progress_chain = str(getattr(task_state, "last_progress_chain", "") or "")
        last_progress_cursor = str(getattr(task_state, "last_progress_cursor", "") or "")
        last_stall_reason = str(getattr(task_state, "last_stall_reason", "") or "")
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "logical_steps": logical_steps,
            "raw_tool_calls": raw_tool_calls,
            "raw_attempts": raw_attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "todo_state": {
                "rewritten_request": str(getattr(task_state, "rewritten_request", "") or ""),
                "todos": list(getattr(task_state, "todos", []) or []),
                "active_todo_id": str(getattr(task_state, "active_todo_id", "") or ""),
                "todo_version": int(getattr(task_state, "todo_version", 0) or 0),
                "last_todo_update": str(getattr(task_state, "last_todo_update", "") or ""),
                "blocked_todo_id": str(getattr(task_state, "blocked_todo_id", "") or ""),
            },
            "progress_state": {
                "logical_steps_used": logical_steps,
                "raw_tool_calls": raw_tool_calls,
                "raw_attempts": raw_attempts,
                "last_progress_chain": last_progress_chain,
                "last_progress_cursor": last_progress_cursor,
            },
            "stall_summary": {
                "last_stall_reason": last_stall_reason,
            },
            "durable_promotions": list(self.last_durable_promotions),
            "durable_superseded": list(self.last_durable_superseded),
            "background_tasks": {
                "total": int(background_tasks.get("total", 0)),
                "counts": dict(background_tasks.get("counts", {})),
                "recent": [
                    {
                        "task_id": item.task_id,
                        "status": item.status,
                        "return_code": item.return_code,
                        "pid": item.pid,
                        "started_at": item.started_at,
                        "finished_at": item.finished_at,
                        "timeout": item.timeout,
                        "command": clip(item.command, 160),
                    }
                    for item in background_tasks.get("recent", [])
                ],
            },
            "redacted_env": self.detected_secret_env_summary(),
        }

    def write_git_diff_artifact(self, content):
        if self.current_task_state is None:
            return ""
        path = self.run_store.write_latest_git_diff(self.current_task_state.run_id, content)
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except Exception:
            return str(path)

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self.tool_context(), name, args)

    def tool_context(self):
        return ToolContext(
            root=self.root,
            path_resolver=self.path,
            shell_env_provider=self.shell_env,
            depth=self.depth,
            max_depth=self.max_depth,
            spawn_delegate=self.spawn_delegate,
            background_task_starter=self.start_background_task,
            background_task_reader=self.read_background_task,
            background_task_stopper=self.stop_background_task,
            background_task_lister=self.list_background_tasks,
            background_task_lookup=self.find_background_task,
            git_diff_artifact_writer=self.write_git_diff_artifact,
        )

    def spawn_delegate(self, args):
        task = str(args.get("task", "")).strip()
        inherit_context = args.get("inherit_context", True)
        if isinstance(inherit_context, str):
            inherit_context = inherit_context.strip().lower() in {"true", "1", "yes"}
        child = Pico(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            run_store=self.run_store,
            approval_policy=self.approval_policy,
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=self.read_only,
            secret_env_names=self.secret_env_names,
            shell_env_allowlist=self.shell_env_allowlist,
            tool_call_reporter=self.tool_call_reporter,
            assistant_message_reporter=self.assistant_message_reporter,
        )
        if inherit_context:
            parent_history = self.context_manager.render_history_for_delegate()
            task = "\n\n".join(
                [
                    "Parent context:",
                    parent_history,
                    "Delegated task:",
                    task,
                ]
            )
        return "delegate_result:\n" + child.ask(task)

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self.tool_context(), args)

    def tool_glob(self, args):
        return toolkit.tool_glob(self.tool_context(), args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self.tool_context(), args)

    def tool_grep(self, args):
        return toolkit.tool_grep(self.tool_context(), args)

    def tool_git_status(self, args):
        return toolkit.tool_git_status(self.tool_context(), args)

    def tool_git_diff(self, args):
        return toolkit.tool_git_diff(self.tool_context(), args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self.tool_context(), args)

    def _background_task_run_id(self):
        if getattr(self, "current_task_state", None) is not None:
            return self.current_task_state.run_id
        if getattr(self, "current_run_dir", None) is not None:
            return Path(self.current_run_dir).name
        run_id = self.new_run_id()
        self.run_store.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        return run_id

    def start_background_task(self, args):
        command = str(args.get("command", "")).strip()
        timeout = int(args.get("timeout", 3600))
        task_id = self.new_background_task_id()
        record = self.background_tasks.start(
            run_id=self._background_task_run_id(),
            task_id=task_id,
            command=command,
            cwd=self.root,
            env=self.shell_env(),
            timeout=timeout,
        )
        return format_task_start_text(record)

    def read_background_task(self, args):
        output_data = self.background_tasks.read_output(
            task_id=args.get("task_id", ""),
            offset=int(args.get("offset", 0)),
            limit=int(args.get("limit", 4000)),
            stream=str(args.get("stream", "stdout")).strip() or "stdout",
        )
        return format_task_output_text(output_data)

    def stop_background_task(self, args):
        record, stopped = self.background_tasks.stop(args.get("task_id", ""))
        return format_task_stop_text(record, stopped)

    def list_background_tasks(self, args):
        listing = self.background_tasks.list_tasks(
            run_id=self._background_task_run_id(),
            offset=int(args.get("offset", 0)),
            limit=int(args.get("limit", 20)),
            status=str(args.get("status", "all")).strip() or "all",
        )
        return format_task_list_text(listing)

    def find_background_task(self, task_id):
        task_id = str(task_id).strip()
        if not task_id:
            return None
        try:
            return self.background_tasks.get(task_id).to_dict()
        except FileNotFoundError:
            return None

    def tool_run_shell_bg(self, args):
        return toolkit.tool_run_shell_bg(self.tool_context(), args)

    def tool_task_output(self, args):
        return toolkit.tool_task_output(self.tool_context(), args)

    def tool_task_stop(self, args):
        return toolkit.tool_task_stop(self.tool_context(), args)

    def tool_task_list(self, args):
        return toolkit.tool_task_list(self.tool_context(), args)

    def recent_background_tasks_text(self, limit=3):
        run_id = self._background_task_run_id()
        summary = self.background_tasks.summarize_run_tasks(run_id, limit=limit)
        recent_history = list(self._iter_history_items_for_prompt(self.session.get("history", [])))
        used_recently = any(
            isinstance(item, dict)
            and item.get("role") == "tool"
            and item.get("name") in {"run_shell_bg", "task_output", "task_stop", "task_list"}
            for item in recent_history[-6:]
        )
        if not used_recently and int(summary.get("counts", {}).get("running", 0)) <= 0:
            return ""
        recent = list(summary.get("recent", []))
        if not recent:
            return ""
        lines = ["Recent background tasks:"]
        for item in recent:
            return_code = item.return_code if item.return_code is not None else "(running)"
            stdout_log = self._relative_task_log_path(item.stdout_path)
            stderr_log = self._relative_task_log_path(item.stderr_path)
            lines.append(
                f"- {item.task_id}: status={item.status}, return_code={return_code}, "
                f"command={clip(item.command, 100)}, stdout_log={stdout_log}, stderr_log={stderr_log}"
            )
        lines.append(
            "If you need the full original log, exact ordering, or details beyond paged task_output, "
            "use read_file on the corresponding stdout_log or stderr_log path."
        )
        return "\n".join(lines)

    def _relative_task_log_path(self, path):
        path = str(path or "").strip()
        if not path:
            return "-"
        try:
            relative = Path(path).resolve().relative_to(self.root.resolve())
            return relative.as_posix()
        except Exception:
            return path

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self.tool_context(), args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self.tool_context(), args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self.tool_context(), args)

    @staticmethod
    def _approval_preview(value, limit=80):
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = " | ".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) <= limit:
            return text
        return text[:limit] + f"... ({len(text) - limit} more chars)"

    def approval_summary(self, name, args):
        args = args if isinstance(args, dict) else {}
        path = str(args.get("path", "")).strip()
        if name == "patch_file":
            old_text = str(args.get("old_text", ""))
            new_text = str(args.get("new_text", ""))
            replace_all = args.get("replace_all", False)
            anchor = self._approval_preview(old_text, 70) or "<empty old_text>"
            if old_text and new_text.endswith(old_text):
                inserted = new_text[: -len(old_text)]
                action = f"insert before matched block: {self._approval_preview(inserted, 80)}"
            elif old_text and new_text.startswith(old_text):
                inserted = new_text[len(old_text) :]
                action = f"insert after matched block: {self._approval_preview(inserted, 80)}"
            elif old_text and old_text in new_text:
                action = "replace matched block with surrounding edits"
            else:
                action = "replace one exact matched block"
            return (
                f"path={path or '-'}; {action}; "
                f"replace_all={replace_all}; "
                f"anchor={json.dumps(anchor, ensure_ascii=False)}; "
                f"old={len(old_text.splitlines())} lines/{len(old_text)} chars; "
                f"new={len(new_text.splitlines())} lines/{len(new_text)} chars"
            )
        if name == "write_file":
            content = str(args.get("content", ""))
            action = "overwrite" if path and (self.root / path).is_file() else "create"
            return f"path={path or '-'}; {action}; write {len(content.splitlines())} lines/{len(content)} chars"
        if name == "run_shell":
            command = self._approval_preview(args.get("command", ""), 120)
            timeout = args.get("timeout", 20)
            return f"command={json.dumps(command, ensure_ascii=False)}; timeout={timeout}"
        if name == "run_shell_bg":
            command = self._approval_preview(args.get("command", ""), 120)
            timeout = args.get("timeout", 3600)
            return f"command={json.dumps(command, ensure_ascii=False)}; background=true; timeout={timeout}"
        if name == "task_stop":
            task_id = str(args.get("task_id", "")).strip()
            return f"task_id={json.dumps(task_id, ensure_ascii=False)}"
        if name == "delegate":
            task = self._approval_preview(args.get("task", ""), 120)
            return f"task={json.dumps(task, ensure_ascii=False)}; max_steps={args.get('max_steps', 3)}; inherit_context={args.get('inherit_context', True)}"
        return json.dumps(args, ensure_ascii=False, sort_keys=True)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {self.approval_summary(name, args)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse_todo_update(raw):
        raw = str(raw or "")
        match = TODO_UPDATE_BLOCK_PATTERN.search(raw)
        if not match:
            return None
        try:
            root = ET.fromstring(match.group(0))
        except Exception:
            return None
        if root.tag != "todo_update":
            return None
        operations = []
        active_targets = []
        for child in list(root):
            tag = str(child.tag or "").strip()
            if tag not in {"complete", "activate", "append", "drop", "block"}:
                return None
            todo_id = str(child.attrib.get("id", "")).strip()
            if tag == "append":
                if not todo_id:
                    return None
                text = " ".join(part.strip() for part in child.itertext() if str(part or "").strip()).strip()
                if not text:
                    return None
                operations.append({"op": "append", "id": todo_id, "text": text})
                continue
            if not todo_id:
                return None
            operations.append({"op": tag, "id": todo_id})
            if tag == "activate":
                active_targets.append(todo_id)
        if len(active_targets) > 1:
            return None
        return {"raw": match.group(0), "operations": operations}

    @staticmethod
    def extract_display_text(raw):
        raw = str(raw or "")
        match = DISPLAY_BLOCK_PATTERN.search(raw)
        if not match:
            return ""
        try:
            root = ET.fromstring(match.group(0))
        except Exception:
            return ""
        if root.tag != "display":
            return ""
        text = " ".join(part.strip() for part in root.itertext() if str(part or "").strip()).strip()
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        todo_update = Pico.parse_todo_update(raw)
        if todo_update is None:
            stripped = raw.strip()
            if stripped:
                return "retry", Pico.retry_payload("model response is missing a valid <todo_update> block", raw)
            return "retry", Pico.retry_payload("model returned an empty response", "")

        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = Pico.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", Pico.retry_payload("model returned malformed tool JSON", raw)
            if not isinstance(payload, dict):
                return "retry", Pico.retry_payload("tool payload must be a JSON object", raw)
            if not str(payload.get("name", "")).strip():
                return "retry", Pico.retry_payload("tool payload is missing a tool name", raw)
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", Pico.retry_payload(raw_text=raw)
            payload["todo_update"] = todo_update
            payload["display_text"] = Pico.extract_display_text(raw)
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = Pico.parse_xml_tool(raw)
            if payload is not None:
                payload["todo_update"] = todo_update
                payload["display_text"] = Pico.extract_display_text(raw)
                return "tool", payload
            return "retry", Pico.retry_payload(raw_text=raw)
        answer_text = Pico.extract_answer_text(raw)
        display_text = Pico.extract_display_text(raw)
        if answer_text:
            return "answer", {
                "text": answer_text,
                "todo_update": todo_update,
                "raw_text": raw.strip(),
                "display_text": display_text,
            }
        if todo_update.get("operations"):
            return "todo_only", {
                "todo_update": todo_update,
                "raw_text": raw.strip(),
                "display_text": display_text,
            }
        raw = raw.strip()
        if raw:
            return "retry", Pico.retry_payload("model returned text without a usable tool call or answer", raw)
        return "retry", Pico.retry_payload("model returned an empty response", "")

    @staticmethod
    def retry_payload(problem=None, raw_text=""):
        return {
            "notice": Pico.retry_notice(problem),
            "problem": str(problem or "").strip(),
            "raw_text": str(raw_text or "").strip(),
        }

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with exactly one <todo_update>...</todo_update> block and then either one valid <tool> call or a usable plain-text answer. "
            "If you only need to switch todo state, return the <todo_update> block by itself. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def extract_answer_text(text):
        raw = str(text or "")
        if "<final>" in raw:
            final = Pico.extract(raw, "final").strip()
            if final:
                final = TODO_UPDATE_BLOCK_PATTERN.sub(" ", final)
                final = DISPLAY_BLOCK_PATTERN.sub(" ", final)
                return WHITESPACE_PATTERN.sub(" ", final).strip()
            return ""
        stripped = TODO_UPDATE_BLOCK_PATTERN.sub(" ", raw).strip()
        stripped = re.sub(r"<tool(?P<attrs>[^>]*)>.*?</tool>", " ", stripped, flags=re.S).strip()
        stripped = DISPLAY_BLOCK_PATTERN.sub(" ", stripped).strip()
        stripped = FINAL_TAG_PATTERN.sub(" ", stripped).strip()
        return WHITESPACE_PATTERN.sub(" ", stripped).strip()

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = Pico.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = Pico.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()


        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved
