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
from datetime import datetime
from pathlib import Path

from . import checkpoint as checkpointlib
from .features import memory as memorylib
from . import security as securitylib
from .context_manager import ContextManager, _context_units
from .checkpoint import CHECKPOINT_NONE_STATUS
from .prompt_prefix import build_prompt_prefix, tool_signature
from .run_store import RunStore
from .security import REDACTED_VALUE
from .session_store import SessionStore
from .tool_context import ToolContext
from .tool_executor import ToolExecutor
from . import tools as toolkit
from .workspace import AGENT_STATE_DIR, IGNORED_PATH_NAMES, WorkspaceContext, clip, now

DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "context_reduction": True,
    "prompt_cache": True,
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
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        allowed_tools=None,
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
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / AGENT_STATE_DIR / "runs")
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

        lines = []
        seen_reads = set()
        stale_read_indexes = self._stale_read_indexes(history)
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
                    summary = self._history_item_summary(item)
                    lines.append(summary or str(item["content"]))
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

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

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

    def shell_env(self):
        return securitylib.shell_env(allowlist=self.shell_env_allowlist, root=self.root)

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

    def execute_tool(self, name, args):
        result = self.tool_executor.execute(name, args)
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


        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):


        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_superseded": list(self.last_durable_superseded),
            "redacted_env": self.detected_secret_env_summary(),
        }

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

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self.tool_context(), args)

    def tool_search(self, args):
        return toolkit.tool_search(self.tool_context(), args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self.tool_context(), args)

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



        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = Pico.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", Pico.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", Pico.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", Pico.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", Pico.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = Pico.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", Pico.retry_notice()
        if "<final>" in raw:
            final = Pico.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", Pico.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", Pico.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

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
