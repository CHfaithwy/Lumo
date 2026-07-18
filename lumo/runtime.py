

import json
import asyncio
import hashlib
import os
import re
import uuid
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
from . import skills as skilllib
from .legacy_protocol import migrate_history
from .context_manager import ContextManager, _context_units
from .checkpoint import CHECKPOINT_NONE_STATUS
from .prompt_prefix import build_prompt_prefix, tool_signature
from .python_environment import PythonEnvironmentManager
from .run_store import RunStore
from .security import REDACTED_VALUE
from .session_store import SessionStore
from .tool_context import ToolContext
from .tool_executor import ToolExecutor, ToolExecutionResult, strip_tool_hints
from .tool_output import (
    PersistedToolOutput,
    TOOL_RESULT_PREVIEW_LIMIT_CHARS,
    build_externalized_output_message,
    preview_text,
)
from . import tools as toolkit
from .workspace import AGENT_STATE_DIR, IGNORED_PATH_NAMES, WorkspaceContext, clip, now

DEFAULT_SHELL_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "PWD",
    "SHELL",
    "SYSTEMROOT",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "USER",
)
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
SKILL_ROUTING_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt" / "skill_routing.md"
SHELL_REPAIR_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt" / "shell_repair.md"
SKILL_ROUTING_MAX_NEW_TOKENS = 64
SKILL_ROUTING_RECENT_CONTEXT_CHARS = 2000
SHELL_REPAIR_MAX_NEW_TOKENS = 256
SKILL_ROUTING_REASONING_EFFORT = "medium"
PYTHON_MISSING_MODULE_PATTERN = re.compile(r"\b(?:ModuleNotFoundError|ImportError):\s+No module named\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
ASSISTANT_TERMINAL_PREVIEW_LIMIT = 200
SHELL_REPAIR_INLINE_LIMIT = 2000
SHELL_REPAIR_TAIL_LIMIT = 1600
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
        self.skills_dir = skilllib.ensure_skills_dir(self.root)
        self.python_environment = PythonEnvironmentManager(self.root)
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
        self._workspace_fingerprint = self.workspace.fingerprint()
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_model_request = {"instructions": "", "messages": [], "tools": []}
        self.last_provider_request = {}
        self.last_provider_response = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self.last_skill_routing = {}
        self._pending_task_skill = None
        self.transient_task_skills = {}
        self._injected_task_skill_call_ids = set()
        self._skill_catalog = None
        self.transient_skill_state = {
            "catalog": skilllib.SkillCatalog(),
            "skill_categories": [],
        }
        self.transient_todo_state = {
            "todos": [],
            "active_todo_id": "",
            "blocked_todo_id": "",
        }
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }
        self.shell_repair_stats = {
            "requested": 0,
            "planned": 0,
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "skipped": 0,
        }
        self.shell_repair_artifacts = []
        self.tool_output_stats = {
            "externalized": 0,
            "per_result_limit": 0,
            "message_budget": 0,
            "persistence_failed": 0,
            "original_bytes": 0,
            "stored_bytes": 0,
            "artifact_truncated": 0,
        }
        self.archive_stats = {
            "requested": 0,
            "archived": 0,
            "passthrough": 0,
            "externalized": 0,
        }
        self._auxiliary_call_counts = {}

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
        migrated_history, migrated = migrate_history(self.session.get("history", []))
        if migrated:
            self.session["history"] = migrated_history
        self.session["protocol_version"] = "native-v1"
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


        self.session.pop("active_skills", None)

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
            payload = asyncio.run(
                self.complete_structured_async(
                    {"instructions": "Extract only stable durable-memory updates.", "messages": [{"role": "user", "content": prompt}]},
                    {
                        "type": "object",
                        "properties": {"updates": {"type": "array", "items": {"type": "object", "properties": {"topic": {"type": "string"}, "note": {"type": "string"}}, "required": ["topic", "note"], "additionalProperties": False}}},
                        "required": ["updates"],
                        "additionalProperties": False,
                    },
                    max_new_tokens=self.max_new_tokens,
                    name="durable_memory_updates",
                )
            )
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
                    "raw": payload,
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
            "raw": payload,
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

    def refresh_skill_catalog(self):
        self._skill_catalog = skilllib.discover_skill_catalog(self.root)
        return self._skill_catalog

    def current_skill_catalog(self):
        catalog = getattr(self, "_skill_catalog", None)
        if isinstance(catalog, skilllib.SkillCatalog):
            return catalog
        return self.refresh_skill_catalog()

    def set_transient_skill_route(self, catalog=None, skill_categories=None):
        catalog = catalog if isinstance(catalog, skilllib.SkillCatalog) else self.current_skill_catalog()
        categories = skilllib.normalize_routed_categories(skill_categories, catalog)
        self.transient_skill_state = {
            "catalog": catalog,
            "skill_categories": categories,
        }

    def clear_transient_skill_route(self):
        self.transient_skill_state = {
            "catalog": skilllib.SkillCatalog(),
            "skill_categories": [],
        }

    def available_skills_text(self):
        state = dict(getattr(self, "transient_skill_state", {}) or {})
        catalog = state.get("catalog")
        if not isinstance(catalog, skilllib.SkillCatalog):
            catalog = skilllib.SkillCatalog()
        categories = list(state.get("skill_categories", []) or [])
        return skilllib.render_skill_listing(
            catalog.skills_for_categories(categories),
            no_match=not categories,
        )

    def start_task_skills(self, task_state):
        self._pending_task_skill = None
        self.transient_task_skills = {}
        self._injected_task_skill_call_ids = set()
        task_state.update_loaded_skills([])

    def _task_skill_audit_record(self, record):
        return {
            key: record.get(key, "")
            for key in ("call_id", "qualified_name", "path", "skill_root", "args", "loaded_at")
        }

    def register_pending_task_skill(self, call_id, task_state):
        pending = self._pending_task_skill
        self._pending_task_skill = None
        if not isinstance(pending, dict) or not str(call_id or "").strip():
            return None
        record = dict(pending)
        record["call_id"] = str(call_id)
        qualified_name = str(record.get("qualified_name", "")).strip()
        if not qualified_name:
            return None
        for old_call_id, old_record in list(self.transient_task_skills.items()):
            if str(old_record.get("qualified_name", "")).strip() == qualified_name:
                self.transient_task_skills.pop(old_call_id, None)
                self._injected_task_skill_call_ids.discard(old_call_id)
        self.transient_task_skills[record["call_id"]] = record
        task_state.update_loaded_skills(
            [self._task_skill_audit_record(item) for item in self.transient_task_skills.values()]
        )
        self.emit_trace(task_state, "task_skill_loaded", self._task_skill_audit_record(record))
        return self._task_skill_audit_record(record)

    def task_skill_contexts(self):
        return skilllib.render_task_skill_contexts(list(self.transient_task_skills.values()))

    def new_task_skill_injections(self, call_ids):
        new = []
        for call_id in call_ids or []:
            if call_id in self._injected_task_skill_call_ids:
                continue
            record = self.transient_task_skills.get(call_id)
            if record:
                self._injected_task_skill_call_ids.add(call_id)
                new.append(self._task_skill_audit_record(record))
        return new

    def clear_task_skills(self, task_state=None, reason="task_finished"):
        loaded = [self._task_skill_audit_record(item) for item in self.transient_task_skills.values()]
        sanitized_last_request = self.redact_task_skill_artifact(
            dict(getattr(self, "last_model_request", {}) or {})
        )
        sanitized_provider_request = self.redact_task_skill_artifact(
            dict(getattr(self, "last_provider_request", {}) or {})
        )
        self._pending_task_skill = None
        self.transient_task_skills = {}
        self._injected_task_skill_call_ids = set()
        last_request = sanitized_last_request
        if last_request:
            last_request["messages"] = [
                dict(item)
                for item in list(last_request.get("messages", []) or [])
                if str(item.get("role", "")) != "skill_context"
            ]
            self.last_model_request = last_request
        self.last_provider_request = sanitized_provider_request
        if hasattr(self.model_client, "last_request_payload"):
            self.model_client.last_request_payload = self.redact_task_skill_artifact(
                dict(getattr(self.model_client, "last_request_payload", {}) or {})
            )
        if task_state is not None and loaded:
            self.emit_trace(task_state, "task_skills_cleared", {"reason": reason, "loaded_skills": loaded})

    def redact_task_skill_artifact(self, value):

        contexts = [
            str(item.get("content", ""))
            for item in self.task_skill_contexts().values()
            if str(item.get("content", ""))
        ]
        if not contexts:
            return value

        def redact(node):
            if isinstance(node, dict):
                return {key: redact(item) for key, item in node.items()}
            if isinstance(node, list):
                return [redact(item) for item in node]
            if isinstance(node, str):
                for content in contexts:
                    node = node.replace(content, "[task skill content omitted from artifact]")
                return node
            return node

        return redact(value)

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



    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(self, "_workspace_fingerprint", "")
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace
            self._workspace_fingerprint = refreshed_workspace_fingerprint

        prefix_state = self.build_prefix()
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
            return ""
        if compact and not self._should_keep_full_assistant_terminal_message(message):
            message = self._compact_assistant_terminal_message(message)
        if not message:
            return ""
        reporter(message)
        return message

    def assistant_progress_message(self, text):
        message = self._normalize_assistant_terminal_message(text)
        message = self.redact_text(message)
        return message

    @staticmethod
    def _normalize_assistant_terminal_message(text):
        value = str(text or "").strip()
        if not value:
            return ""
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

    def set_transient_todo_state(self, todos=None, active_todo_id="", blocked_todo_id="", **_ignored):
        self.transient_todo_state = {
            "todos": list(todos or []),
            "active_todo_id": str(active_todo_id or ""),
            "blocked_todo_id": str(blocked_todo_id or ""),
        }

    def clear_transient_todo_state(self):
        self.set_transient_todo_state()

    def current_todo_request_text(self):
        state = dict(getattr(self, "transient_todo_state", {}) or {})
        todos = list(state.get("todos", []) or [])
        active_todo_id = str(state.get("active_todo_id", "")).strip()
        if not todos:
            return ""
        lines = []
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

    def prepare_shell_command(self, command):
        return self.python_environment.prepare(command, self.shell_env())

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

    async def complete_messages_async(
        self,
        system,
        messages,
        max_new_tokens=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
    ):
        token_limit = int(max_new_tokens or self.max_new_tokens)
        complete_messages_async = getattr(self.model_client, "complete_messages_async", None)
        if complete_messages_async is not None:
            return await complete_messages_async(
                system,
                messages,
                token_limit,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
        complete_messages = getattr(self.model_client, "complete_messages", None)
        if complete_messages is not None:
            return await asyncio.to_thread(
                complete_messages,
                system,
                messages,
                token_limit,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
        fallback = self.render_model_messages(system, messages)
        return await self.complete_text_async(
            fallback,
            max_new_tokens=token_limit,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

    async def complete_turn_async(
        self,
        turn_request,
        prompt_cache_key=None,
        prompt_cache_retention=None,
    ):
        from .model_protocol import ModelTurnRequest

        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(
            turn_request,
            max_output_tokens=self.max_new_tokens,
        )
        client_method = getattr(self.model_client, "complete_turn_async", None)
        if client_method is None:
            raise RuntimeError("native_tool_calling_unsupported: model client does not implement complete_turn_async")
        try:
            response = await client_method(
                request,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
        except Exception:
            self.last_provider_request = dict(getattr(self.model_client, "last_request_payload", {}) or {})
            self.last_provider_response = {}
            raise
        self.last_provider_request = dict(getattr(self.model_client, "last_request_payload", {}) or {})
        self.last_provider_response = dict(getattr(self.model_client, "last_response_payload", {}) or {})
        return response

    async def complete_structured_async(self, request, schema, max_new_tokens=256, name="structured_output", reasoning_effort=None):
        method = getattr(self.model_client, "complete_structured_async", None)
        if method is None:
            raise RuntimeError("native_structured_output_unsupported: model client does not implement complete_structured_async")
        result = await method(
            request,
            schema,
            max_new_tokens=max_new_tokens,
            name=name,
            reasoning_effort=reasoning_effort,
        )
        task_state = getattr(self, "current_task_state", None)
        if task_state is not None:
            count = int(self._auxiliary_call_counts.get(name, 0)) + 1
            self._auxiliary_call_counts[name] = count
            request_path, response_path = self.run_store.write_auxiliary_exchange(
                task_state,
                name,
                count,
                self.redact_artifact(
                    {
                        "provider": self.model_client.__class__.__name__,
                        "request": dict(request or {}),
                        "schema": dict(schema or {}),
                        "provider_request": dict(getattr(self.model_client, "last_request_payload", {}) or {}),
                    }
                ),
                self.redact_artifact(
                    {
                        "result": result if isinstance(result, dict) else {},
                        "provider_response": dict(getattr(self.model_client, "last_response_payload", {}) or {}),
                    }
                ),
            )
            self.emit_trace(
                task_state,
                "auxiliary_model_completed",
                {"kind": name, "request_path": str(request_path), "response_path": str(response_path)},
            )
        return result

    @staticmethod
    def render_model_messages(system, messages):
        blocks = []
        if str(system or "").strip():
            blocks.append("[system]\n" + str(system).strip())
        for item in list(messages or []):
            role = str((item or {}).get("role", "user")).strip() or "user"
            content = str((item or {}).get("content", ""))
            if content.strip():
                blocks.append(f"[{role}]\n{content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _shell_exit_code(content):
        match = re.search(r"^exit_code:\s*(-?\d+)\s*$", str(content), re.MULTILINE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _shell_streams(content):
        text = str(content or "")
        match = re.search(r"^stdout:\s*\n([\s\S]*?)\n\s*stderr:\s*\n([\s\S]*)$", text, re.MULTILINE)
        if not match:
            return "", text
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _shell_repair_eligibility(stdout, stderr):
        detail = "\n".join([str(stdout or ""), str(stderr or "")])
        if PYTHON_MISSING_MODULE_PATTERN.search(detail):
            return True, "missing_python_module"
        if re.search(r"\bSyntaxError\b", detail):
            return False, "python_syntax_error"
        if re.search(r"\bhere-document\b|\bheredoc\b", detail, re.IGNORECASE):
            return False, "unsupported_shell_syntax"
        return False, "non_dependency_python_failure"

    def _skipped_shell_repair_result(self, tool_result, reason, *, task_id=""):
        metadata = dict(tool_result.metadata or {})
        metadata.update(
            {
                "repair_attempted": False,
                "repair_eligible": False,
                "repair_action": "none",
                "repair_status": "skipped",
                "repair_reason": str(reason),
            }
        )
        self.shell_repair_stats["skipped"] += 1
        payload = {"reason": str(reason)}
        if task_id:
            payload["task_id"] = str(task_id)
        self._trace_current_task("shell_repair_skipped", payload)
        content = "\n".join(
            [
                str(tool_result.content),
                "repair_status: skipped",
                f"repair_reason: {reason}",
                "next_action: Automatic repair only handles missing Python modules; diagnose and correct this command in the main agent loop.",
            ]
        )
        capture = getattr(tool_result, "output_capture", None)
        if capture is not None:
            capture.cleanup()
        return ToolExecutionResult(content=content, metadata=metadata)

    def _shell_repair_prompt(self, original_command, executed_command, exit_code, stdout, stderr):
        environment = self.python_environment.summary()
        return (
            SHELL_REPAIR_TEMPLATE_PATH.read_text(encoding="utf-8")
            .replace("{{ENVIRONMENT}}", json.dumps(environment, ensure_ascii=False, sort_keys=True))
            .replace("{{ORIGINAL_COMMAND}}", self.redact_text(clip(original_command, 2000)))
            .replace("{{EXECUTED_COMMAND}}", self.redact_text(clip(executed_command, 2000)))
            .replace("{{EXIT_CODE}}", str(exit_code if exit_code is not None else "unknown"))
            .replace("{{STDOUT}}", self.redact_text(clip(stdout, 1000)))
            .replace("{{STDERR}}", self.redact_text(clip(stderr, 4000)))
        )

    @staticmethod
    def parse_shell_repair(payload):
        payload = payload if isinstance(payload, dict) else {}
        action = str(payload.get("action", "")).strip().lower()
        command = str(payload.get("command", "")).strip()
        reason = " ".join(str(payload.get("reason", "")).split())
        if action not in {"repair", "none"}:
            return {"valid": False, "action": "none", "command": "", "reason": "invalid_shell_repair_action"}
        if action == "repair" and not command:
            return {"valid": False, "action": "none", "command": "", "reason": "missing_repair_command"}
        if action == "none" and command:
            return {"valid": False, "action": "none", "command": "", "reason": "none_action_must_not_include_command"}
        return {"valid": True, "action": action, "command": command, "reason": reason}

    def _trace_current_task(self, event, payload=None):
        task_state = getattr(self, "current_task_state", None)
        if task_state is not None:
            self.emit_trace(task_state, event, payload or {})

    async def _request_shell_repair_async(self, original_command, executed_command, exit_code, stdout, stderr):
        self.shell_repair_stats["requested"] += 1
        self._trace_current_task(
            "shell_repair_requested",
            {"command": original_command, "exit_code": exit_code},
        )
        try:
            request = {
                "instructions": "Return a validated shell environment repair decision. Only repair the Lumo Python environment.",
                "messages": [{"role": "user", "content": self._shell_repair_prompt(original_command, executed_command, exit_code, stdout, stderr)}],
            }
            schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["repair", "none"]},
                    "command": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["action", "command", "reason"],
                "additionalProperties": False,
            }
            raw = await self.complete_structured_async(request, schema, max_new_tokens=SHELL_REPAIR_MAX_NEW_TOKENS, name="shell_repair")
        except Exception as exc:
            self.shell_repair_stats["failed"] += 1
            self._trace_current_task("shell_repair_planned", {"valid": False, "error": str(exc)})
            return {"valid": False, "action": "none", "command": "", "reason": f"repair_model_error:{exc}"}
        plan = self.parse_shell_repair(raw)
        if plan.get("valid") and plan.get("action") == "repair":
            try:
                self.python_environment.validate_repair_command(plan.get("command", ""))
            except ValueError as exc:
                plan = {"valid": False, "action": "none", "command": "", "reason": f"repair_command_rejected:{exc}"}
                self.shell_repair_stats["rejected"] += 1
            else:
                self.shell_repair_stats["planned"] += 1
        elif not plan.get("valid"):
            self.shell_repair_stats["rejected"] += 1
        self._trace_current_task("shell_repair_planned", plan)
        return plan

    @staticmethod
    def _shell_repair_tail(text, limit=SHELL_REPAIR_TAIL_LIMIT):
        value = str(text or "").strip()
        if len(value) <= int(limit):
            return value
        return "...[showing final repair output]\n" + value[-int(limit):]

    def _write_shell_repair_artifact(self, content):
        if self.current_task_state is None:
            return ""
        index = len(self.shell_repair_artifacts) + 1
        redacted = self.redact_text(str(content or ""))
        path = self.run_store.write_shell_repair_log(self.current_task_state.run_id, index, redacted)
        try:
            display_path = path.resolve().relative_to(self.root.resolve()).as_posix()
        except Exception:
            display_path = str(path)
        self.shell_repair_artifacts.append(display_path)
        return display_path

    def _failed_shell_repair_result(
        self,
        tool_result,
        *,
        repair_command,
        repair_exit_code,
        repair_stdout="",
        repair_stderr="",
        repair_reason="",
    ):
        stdout = self.redact_text(str(repair_stdout or "").strip())
        stderr = self.redact_text(str(repair_stderr or "").strip())
        reason = self.redact_text(str(repair_reason or stderr or stdout or "repair failed").strip())
        full_repair_output = "\n".join(
            [
                f"command: {self.redact_text(str(repair_command or '').strip()) or '(unavailable)'}",
                "status: failed",
                f"exit_code: {repair_exit_code if repair_exit_code is not None else 'unknown'}",
                f"stdout:\n{stdout or '(empty)'}",
                f"stderr:\n{stderr or '(empty)'}",
            ]
        )
        repair_log_path = ""
        if len(full_repair_output) > SHELL_REPAIR_INLINE_LIMIT:
            repair_log_path = self._write_shell_repair_artifact(full_repair_output)
            visible_output = "\n".join(
                [
                    f"command: {self.redact_text(str(repair_command or '').strip()) or '(unavailable)'}",
                    "status: failed",
                    f"exit_code: {repair_exit_code if repair_exit_code is not None else 'unknown'}",
                    f"full_log: {repair_log_path or '(unavailable)'}",
                    f"error_tail:\n{self._shell_repair_tail(stderr or stdout or reason)}",
                ]
            )
        else:
            visible_output = full_repair_output

        metadata = dict(tool_result.metadata)
        metadata.update(
            {
                "repair_attempted": True,
                "repair_action": "repair",
                "repair_status": "failed",
                "repair_reason": self._shell_repair_tail(reason, 1000),
                "repair_log_path": repair_log_path,
            }
        )
        content = "\n".join(
            [
                "original_execution:",
                self.redact_text(str(tool_result.content)),
                "repair_execution:",
                visible_output,
                "next_action:",
                "Automatic repair failed; diagnose the reported repair error in the main agent loop.",
            ]
        )
        capture = getattr(tool_result, "output_capture", None)
        if capture is not None:
            capture.cleanup()
        return ToolExecutionResult(content=content, metadata=metadata)

    async def maybe_repair_shell_failure_async(self, name, args, tool_result):
        metadata = dict(tool_result.metadata or {})
        if name == "run_shell":
            if not metadata.get("python_env_used"):
                return tool_result
            if metadata.get("python_env_error"):
                return tool_result
            self._trace_current_task(
                f"python_environment_{metadata.get('python_env_status') or 'reused'}",
                {"path": metadata.get("python_env_path", ".lumo/python-env")},
            )
            if str(metadata.get("tool_status", "")) not in {"error", "partial_success"}:
                return tool_result
            stdout, stderr = self._shell_streams(tool_result.content)
            eligible, reason = self._shell_repair_eligibility(stdout, stderr)
            if not eligible:
                return self._skipped_shell_repair_result(tool_result, reason)
            return await self._repair_foreground_shell_async(args, tool_result)
        if name == "task_output" and str(metadata.get("background_task_status", "")) == "failed":
            return await self._repair_background_shell_async(args, tool_result)
        return tool_result

    async def _repair_foreground_shell_async(self, args, tool_result):
        original_command = str(args.get("command", "")).strip()
        executed_command = str(tool_result.metadata.get("executed_command", original_command)).strip()
        stdout, stderr = self._shell_streams(tool_result.content)
        exit_code = self._shell_exit_code(tool_result.content)
        plan = await self._request_shell_repair_async(original_command, executed_command, exit_code, stdout, stderr)
        if not plan.get("valid") or plan.get("action") != "repair":
            metadata = dict(tool_result.metadata)
            metadata.update(
                {
                    "repair_attempted": True,
                    "repair_action": str(plan.get("action", "none")),
                    "repair_status": "not_run",
                    "repair_reason": str(plan.get("reason", "")),
                }
            )
            capture = getattr(tool_result, "output_capture", None)
            if capture is not None:
                capture.cleanup()
            return ToolExecutionResult(content=tool_result.content, metadata=metadata)

        before_snapshot = self.capture_workspace_snapshot()
        repair_started = datetime.now()
        try:
            repair_result, repair_argv = await asyncio.to_thread(
                self.python_environment.run_repair,
                plan["command"],
                self.shell_env(),
                300,
            )
        except Exception as exc:
            self.shell_repair_stats["failed"] += 1
            failed_result = self._failed_shell_repair_result(
                tool_result,
                repair_command=plan.get("command", ""),
                repair_exit_code=None,
                repair_stderr=str(exc),
                repair_reason=str(exc),
            )
            self._trace_current_task(
                "shell_repair_executed",
                {
                    "status": "failed",
                    "error": self._shell_repair_tail(str(exc), 1000),
                    "repair_log_path": failed_result.metadata.get("repair_log_path", ""),
                },
            )
            return failed_result
        failed_result = None
        if repair_result.returncode != 0:
            failed_result = self._failed_shell_repair_result(
                tool_result,
                repair_command=" ".join(repair_argv),
                repair_exit_code=repair_result.returncode,
                repair_stdout=repair_result.stdout,
                repair_stderr=repair_result.stderr,
                repair_reason=repair_result.stderr or repair_result.stdout,
            )
        self._trace_current_task(
            "shell_repair_executed",
            {
                "status": "succeeded" if repair_result.returncode == 0 else "failed",
                "command": " ".join(repair_argv),
                "return_code": repair_result.returncode,
                "duration_ms": int((datetime.now() - repair_started).total_seconds() * 1000),
                "repair_log_path": (
                    failed_result.metadata.get("repair_log_path", "") if failed_result is not None else ""
                ),
            },
        )
        if repair_result.returncode != 0:
            self.shell_repair_stats["failed"] += 1
            return failed_result

        retry_raw = toolkit.tool_run_shell(self.tool_context(), args)
        retry_capture = getattr(retry_raw, "returncode", None)
        if retry_capture is not None and hasattr(retry_raw, "diagnostic_text"):
            retry_exit_code = int(retry_capture)
            retry_content = retry_raw.diagnostic_text()
            retry_raw.cleanup()
        else:
            retry_content = str(retry_raw)
            retry_exit_code = self._shell_exit_code(retry_content)
        after_snapshot = self.capture_workspace_snapshot()
        affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
        retry_status = "succeeded" if retry_exit_code == 0 else "failed"
        if retry_status == "succeeded":
            self.shell_repair_stats["succeeded"] += 1
        else:
            self.shell_repair_stats["failed"] += 1
        self._trace_current_task(
            "shell_command_retried",
            {"status": retry_status, "command": original_command, "return_code": retry_exit_code},
        )
        metadata = dict(tool_result.metadata)
        metadata.update(
            {
                "tool_status": "ok" if retry_status == "succeeded" else "error",
                "tool_error_code": "" if retry_status == "succeeded" else "tool_failed",
                "affected_paths": affected_paths,
                "diff_summary": diff_summary,
                "workspace_changed": bool(affected_paths),
                "repair_attempted": True,
                "repair_action": "repair",
                "repair_status": "succeeded",
                "repair_command": " ".join(repair_argv),
                "retry_status": retry_status,
            }
        )
        content = "\n".join(
            [
                "original_execution:",
                tool_result.content,
                "repair_execution:",
                f"exit_code: {repair_result.returncode}",
                f"stdout:\n{repair_result.stdout.strip() or '(empty)'}",
                f"stderr:\n{repair_result.stderr.strip() or '(empty)'}",
                "retry_execution:",
                str(retry_content),
            ]
        )
        capture = getattr(tool_result, "output_capture", None)
        if capture is not None:
            capture.cleanup()
        return ToolExecutionResult(content=content, metadata=metadata)

    async def _repair_background_shell_async(self, args, tool_result):
        task_id = str(args.get("task_id", "")).strip()
        try:
            record = self.background_tasks.get(task_id)
        except Exception:
            return tool_result
        if not record.python_env_used or not record.repair_authorized or record.repair_attempted:
            return tool_result
        record = self.background_tasks.update(record, repair_attempted=True)
        stdout = Path(record.stdout_path).read_text(encoding="utf-8", errors="replace") if Path(record.stdout_path).is_file() else ""
        stderr = Path(record.stderr_path).read_text(encoding="utf-8", errors="replace") if Path(record.stderr_path).is_file() else ""
        eligible, reason = self._shell_repair_eligibility(stdout, stderr)
        if not eligible:
            return self._skipped_shell_repair_result(tool_result, reason, task_id=record.task_id)
        plan = await self._request_shell_repair_async(
            record.original_command or record.command,
            record.executed_command or record.command,
            record.return_code,
            stdout,
            stderr,
        )
        if not plan.get("valid") or plan.get("action") != "repair":
            metadata = dict(tool_result.metadata)
            metadata.update(
                {
                    "repair_attempted": True,
                    "repair_action": str(plan.get("action", "none")),
                    "repair_status": "not_run",
                    "repair_reason": str(plan.get("reason", "")),
                }
            )
            return ToolExecutionResult(content=tool_result.content, metadata=metadata)
        try:
            repair_result, repair_argv = await asyncio.to_thread(
                self.python_environment.run_repair,
                plan["command"],
                self.shell_env(),
                300,
            )
        except Exception as exc:
            self.shell_repair_stats["failed"] += 1
            self._trace_current_task("shell_repair_executed", {"status": "failed", "error": str(exc), "task_id": task_id})
            return ToolExecutionResult(
                content=tool_result.content,
                metadata={**dict(tool_result.metadata), "repair_attempted": True, "repair_status": "failed", "repair_reason": str(exc)},
            )
        self._trace_current_task(
            "shell_repair_executed",
            {"status": "succeeded" if repair_result.returncode == 0 else "failed", "command": " ".join(repair_argv), "return_code": repair_result.returncode, "task_id": task_id},
        )
        if repair_result.returncode != 0:
            self.shell_repair_stats["failed"] += 1
            return ToolExecutionResult(
                content=tool_result.content,
                metadata={**dict(tool_result.metadata), "repair_attempted": True, "repair_status": "failed"},
            )
        prepared = self.prepare_shell_command(record.original_command or record.command)
        replacement_id = self.new_background_task_id()
        replacement = self.background_tasks.start(
            run_id=record.run_id,
            task_id=replacement_id,
            command=prepared.command,
            original_command=record.original_command or record.command,
            cwd=self.root,
            env=prepared.env,
            timeout=record.timeout,
            python_env_used=True,
            python_env_path=prepared.python_env_path,
            repair_authorized=True,
            repair_attempted=True,
            retry_of_task_id=record.task_id,
        )
        self.background_tasks.update(record, replacement_task_id=replacement.task_id, repair_attempted=True)
        self.shell_repair_stats["succeeded"] += 1
        self._trace_current_task(
            "shell_command_retried",
            {"status": "running", "task_id": task_id, "replacement_task_id": replacement.task_id},
        )
        stream = str(args.get("stream", "stdout")).strip() or "stdout"
        limit = int(args.get("limit", 4000) or 4000)
        metadata = dict(tool_result.metadata)
        metadata.update(
            {
                "background_task_id": replacement.task_id,
                "background_task_status": "running",
                "background_task_return_code": None,
                "repair_attempted": True,
                "repair_action": "repair",
                "repair_status": "succeeded",
                "repair_command": " ".join(repair_argv),
                "replacement_task_id": replacement.task_id,
                "followup_tool": "task_output",
                "followup_args": {"task_id": replacement.task_id, "offset": 0, "limit": limit, "stream": stream},
                "followup_reason": "background_task_restarted_after_repair",
                "followup_key": f"task_output_wait:{replacement.task_id}:{stream}",
                "chain_key": f"task_output:{replacement.task_id}:{stream}",
                "completion_block_policy": "until_terminal",
                "followup_is_blocking": True,
                "blocks_completion": True,
                "archive_summary": (
                    f"Background task {record.task_id} failed; repaired the managed Python environment and "
                    f"restarted the original command as {replacement.task_id}."
                ),
                "tool_reminder": (
                    f"Continue with task_output using task_id {replacement.task_id}, offset 0, "
                    f"stream {stream}, and limit {limit}. Do not poll the failed task {record.task_id} again."
                ),
            }
        )
        content = "\n".join(
            [
                tool_result.content,
                "repair_status: succeeded",
                f"replacement_task_id: {replacement.task_id}",
                f"repair_command: {' '.join(repair_argv)}",
            ]
        )
        return ToolExecutionResult(content=content, metadata=metadata)

    async def route_skill_categories_async(self, user_message, skill_catalog=None, recent_context=None):
        original_text = str(user_message or "")
        catalog = skill_catalog if isinstance(skill_catalog, skilllib.SkillCatalog) else self.current_skill_catalog()
        should_route = skilllib.should_route_skill_categories(catalog)
        visible_categories = [category.name for category in catalog.categories]
        context_text = (
            self.skill_routing_recent_context(original_text)
            if recent_context is None
            else str(recent_context or "").strip()
        )
        metadata = {
            "mode": "categories" if should_route else "all",
            "attempted": False,
            "valid": True,
            "retry": False,
            "retry_valid": False,
            "selected_categories": [],
            "available_skill_category_count": len(catalog.categories),
            "available_skill_count": len(catalog.skills),
            "recent_context_chars": len(context_text),
            "error": "",
        }
        if not should_route:
            metadata["selected_categories"] = visible_categories
            self.last_skill_routing = dict(metadata)
            return visible_categories

        metadata["attempted"] = True
        try:
            schema = {
                "type": "object",
                "properties": {"categories": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
                "required": ["categories"],
                "additionalProperties": False,
            }
            raw = await self.complete_structured_async(
                {"instructions": "Classify the request into the supplied skill categories only.", "messages": [{"role": "user", "content": self._skill_routing_prompt(original_text, catalog, recent_context=context_text)}]},
                schema,
                max_new_tokens=SKILL_ROUTING_MAX_NEW_TOKENS,
                name="skill_routing",
                reasoning_effort=SKILL_ROUTING_REASONING_EFFORT,
            )
            categories, valid = self.parse_skill_routing(raw, catalog)
            if not valid:
                metadata["retry"] = True
                retry_raw = await self.complete_structured_async(
                    {"instructions": "Return only valid category identifiers from the supplied list.", "messages": [{"role": "user", "content": self._skill_routing_prompt(original_text, catalog, recent_context=context_text)}]},
                    schema,
                    max_new_tokens=SKILL_ROUTING_MAX_NEW_TOKENS,
                    name="skill_routing",
                    reasoning_effort=SKILL_ROUTING_REASONING_EFFORT,
                )
                categories, valid = self.parse_skill_routing(retry_raw, catalog)
                metadata["retry_valid"] = valid
            metadata["valid"] = valid
            metadata["selected_categories"] = categories if valid else []
            if not valid:
                metadata["error"] = "invalid_skill_routing_response"
            self.last_skill_routing = dict(metadata)
            return list(metadata["selected_categories"])
        except Exception as exc:
            metadata["valid"] = False
            metadata["error"] = str(exc)
            self.last_skill_routing = dict(metadata)
            return []

    def skill_routing_recent_context(self, current_request):

        current_request = str(current_request or "")
        history = list(self.session.get("history", []) or [])
        lines = []
        skipped_current = False
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            if role == "user" and not skipped_current and str(item.get("content", "")) == current_request:
                skipped_current = True
                continue
            if role == "tool":
                content = self._history_item_summary(item) or str(item.get("content", ""))
                line = f"tool {item.get('name', '')}: {content}"
            else:
                content = str(item.get("content", ""))
                line = f"{role}: {content}"
            line = " ".join(line.split())
            if line:
                lines.append(line)
            if len("\n".join(lines)) >= SKILL_ROUTING_RECENT_CONTEXT_CHARS:
                break
        return clip("\n".join(reversed(lines)), SKILL_ROUTING_RECENT_CONTEXT_CHARS)

    @staticmethod
    def _skill_routing_prompt(user_message, catalog, recent_context=""):
        categories = skilllib.render_category_catalog(catalog)
        return (
            SKILL_ROUTING_TEMPLATE_PATH.read_text(encoding="utf-8")
            .replace("{{SKILL_CATEGORIES}}", categories)
            .replace("{{RECENT_CONTEXT}}", str(recent_context or "(none)"))
            .replace("{{USER_REQUEST}}", str(user_message or ""))
        )

    @staticmethod
    def parse_skill_routing(payload, catalog):
        categories = list(payload.get("categories", []) or []) if isinstance(payload, dict) else []
        return skilllib.validate_routed_categories(categories, catalog)

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

        refresh = self.refresh_prefix()

        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = await self.context_manager.build_async(user_message)
        model_request = self.context_manager.model_request()


        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "durable_memory_chars": len(self.memory_text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_directory_tree_lines": len(self.workspace.directory_tree.splitlines()),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "model_request_mode": "messages",
                "model_message_count": len(model_request.get("messages", [])),
                "model_instruction_chars": len(str(model_request.get("instructions", ""))),
                "skill_routing": dict(self.last_skill_routing or {}),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        self.last_model_request = model_request
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
            "protocol_version": "native-v1",
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "completion_mode": str(
                getattr(task_state, "completion_mode", "")
                or (
                    "todo_list_completed"
                    if str(getattr(task_state, "status", "")) == "completed"
                    and str(getattr(task_state, "stop_reason", "")) == "todo_list_completed"
                    else ""
                )
            ),
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "logical_steps": logical_steps,
            "raw_tool_calls": raw_tool_calls,
            "raw_attempts": raw_attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "loaded_skills": [dict(item) for item in list(getattr(task_state, "loaded_skills", []) or []) if isinstance(item, dict)],
            "prompt_metadata": self.last_prompt_metadata,
            "todo_state": {
                "skill_categories": list(getattr(task_state, "skill_categories", []) or []),
                "todos": list(getattr(task_state, "todos", []) or []),
                "active_todo_id": str(getattr(task_state, "active_todo_id", "") or ""),
                "todo_version": int(getattr(task_state, "todo_version", 0) or 0),
                "last_todo_update": str(getattr(task_state, "last_todo_update", "") or ""),
                "blocked_todo_id": str(getattr(task_state, "blocked_todo_id", "") or ""),
                "planning_mode": str(getattr(task_state, "planning_mode", "direct") or "direct"),
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
            "skill_routing": {
                "selected_categories": list(getattr(task_state, "skill_categories", []) or []),
                "available_category_count": int(
                    (getattr(self, "last_skill_routing", {}) or {}).get("available_skill_category_count", 0) or 0
                ),
                "mode": str((getattr(self, "last_skill_routing", {}) or {}).get("mode", "")),
                "attempted": bool((getattr(self, "last_skill_routing", {}) or {}).get("attempted", False)),
                "valid": bool((getattr(self, "last_skill_routing", {}) or {}).get("valid", False)),
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
                        "original_command": clip(item.original_command, 160),
                        "python_env_used": item.python_env_used,
                        "repair_attempted": item.repair_attempted,
                        "retry_of_task_id": item.retry_of_task_id,
                        "replacement_task_id": item.replacement_task_id,
                    }
                    for item in background_tasks.get("recent", [])
                ],
            },
            "python_environment": self.python_environment.summary(),
            "shell_repair": {
                **dict(self.shell_repair_stats),
                "artifacts": list(self.shell_repair_artifacts),
            },
            "tool_output": dict(self.tool_output_stats),
            "tool_archive": dict(self.archive_stats),
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

    @staticmethod
    def tool_result_is_externalization_exempt(name, metadata=None):
        metadata = metadata if isinstance(metadata, dict) else {}
        if name in {"read_file", "task_output", "run_shell_bg"}:
            return True
        return name == "git_diff" and bool(metadata.get("externalized_patch_path"))

    @staticmethod
    def tool_result_visible_text(tool_result):
        capture = getattr(tool_result, "output_capture", None)
        if capture is not None:
            return capture.diagnostic_text()
        return strip_tool_hints(str(getattr(tool_result, "content", "")))

    def tool_result_model_visible_chars(self, tool_result):
        capture = getattr(tool_result, "output_capture", None)
        if capture is not None:
            return capture.total_chars()
        return len(self.redact_text(self.tool_result_visible_text(tool_result)))

    def externalize_tool_result(self, call_id, name, tool_result, reason):

        metadata = dict(getattr(tool_result, "metadata", {}) or {})
        capture = getattr(tool_result, "output_capture", None)
        content = self.tool_result_visible_text(tool_result)
        exempt = self.tool_result_is_externalization_exempt(name, metadata)
        if not reason or (exempt and reason == "per_result_limit"):
            if capture is not None:
                try:
                    content = capture.full_text()
                finally:
                    capture.cleanup()
            return ToolExecutionResult(content=content, metadata=metadata)
        if self.current_task_state is None:
            if capture is not None:
                capture.cleanup()
            return ToolExecutionResult(content=content, metadata=metadata)

        try:
            if capture is not None:
                persisted = self.run_store.write_captured_tool_result(self.current_task_state.run_id, call_id, capture)
            else:
                persisted = self.run_store.write_tool_result(self.current_task_state.run_id, call_id, content)
            artifact_path = Path(persisted.relative_path)
            try:
                display_path = artifact_path.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                display_path = str(artifact_path)
            preview_limit = 0 if reason == "message_budget" else TOOL_RESULT_PREVIEW_LIMIT_CHARS
            visible_preview = self.redact_text(preview_text(content, preview_limit))
            metadata.update(
                {
                    "externalized_output_path": display_path,
                    "externalization_reason": str(reason),
                    "original_output_chars": persisted.original_chars,
                    "original_output_bytes": persisted.original_bytes,
                    "stored_output_bytes": persisted.stored_bytes,
                    "preview_chars": len(visible_preview),
                    "artifact_truncated": persisted.artifact_truncated,
                }
            )
            self.tool_output_stats["externalized"] += 1
            self.tool_output_stats[str(reason)] = self.tool_output_stats.get(str(reason), 0) + 1
            self.tool_output_stats["original_bytes"] += persisted.original_bytes
            self.tool_output_stats["stored_bytes"] += persisted.stored_bytes
            if persisted.artifact_truncated:
                self.tool_output_stats["artifact_truncated"] += 1
            self.emit_trace(
                self.current_task_state,
                "tool_output_externalized",
                {
                    "call_id": str(call_id),
                    "name": str(name),
                    "path": display_path,
                    "reason": str(reason),
                    "original_chars": persisted.original_chars,
                    "original_bytes": persisted.original_bytes,
                    "stored_bytes": persisted.stored_bytes,
                    "artifact_truncated": persisted.artifact_truncated,
                },
            )
            return ToolExecutionResult(
                content=build_externalized_output_message(
                    PersistedToolOutput(
                        relative_path=display_path,
                        original_chars=persisted.original_chars,
                        original_bytes=persisted.original_bytes,
                        stored_bytes=persisted.stored_bytes,
                        artifact_truncated=persisted.artifact_truncated,
                    ),
                    visible_preview,
                ),
                metadata=metadata,
            )
        except Exception as exc:
            visible_preview = self.redact_text(preview_text(content, TOOL_RESULT_PREVIEW_LIMIT_CHARS))
            metadata.update(
                {
                    "artifact_persistence_error": self.redact_text(str(exc)),
                    "externalization_reason": str(reason),
                    "original_output_chars": len(content),
                    "preview_chars": len(visible_preview),
                }
            )
            self.tool_output_stats["persistence_failed"] += 1
            self.emit_trace(
                self.current_task_state,
                "tool_output_externalization_failed",
                {"call_id": str(call_id), "name": str(name), "reason": str(reason)},
            )
            return ToolExecutionResult(
                content=(
                    "Tool output was too large to inline and could not be persisted.\n\n"
                    f"Preview (first {len(visible_preview)} chars):\n{visible_preview or '(empty)'}"
                ),
                metadata=metadata,
            )
        finally:
            if capture is not None:
                capture.cleanup()

    def validate_tool(self, name, args):

        toolkit.validate_tool(self.tool_context(), name, args)

    def normalize_tool_arguments(self, name, args):
        return toolkit.normalize_tool_arguments(name, args)

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
            skill_loader=self.load_skill,
            skill_catalog_provider=self.current_skill_catalog,
            todo_writer=self.write_todos,
            git_diff_artifact_writer=self.write_git_diff_artifact,
            shell_command_preparer=self.prepare_shell_command,
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
            max_steps=int(args.get("max_steps", toolkit.DELEGATE_DEFAULT_MAX_STEPS)),
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

    def load_skill(self, args):
        name = str(args.get("name", "")).strip()
        skill_args = str(args.get("args", "") or "")
        state = dict(getattr(self, "transient_skill_state", {}) or {})
        catalog = state.get("catalog")
        if not isinstance(catalog, skilllib.SkillCatalog) or not catalog.categories:
            catalog = self.current_skill_catalog()
        skill, content = skilllib.load_skill_content(self.root, name, catalog=catalog)
        self._pending_task_skill = skilllib.task_skill_record(skill, content, skill_args)
        return skilllib.format_use_skill_result(skill, skill_args)

    def write_todos(self, args):
        task_state = self.current_task_state
        if task_state is None:
            raise ValueError("todo_write requires an active task")
        todos = toolkit.normalize_todo_items(args)
        active_todo_id = next(
            (item["id"] for item in todos if item["status"] == "active"),
            "",
        )
        blocked_todo_id = next(
            (item["id"] for item in todos if item["status"] == "blocked"),
            "",
        )
        task_state.update_todo_state(
            todos=todos,
            active_todo_id=active_todo_id,
            todo_version=int(task_state.todo_version or 0) + 1,
            last_todo_update=json.dumps(todos, ensure_ascii=False),
            blocked_todo_id=blocked_todo_id,
            planning_mode="planned",
        )
        self.set_transient_todo_state(
            todos=todos,
            active_todo_id=active_todo_id,
            blocked_todo_id=blocked_todo_id,
        )
        done_count = sum(1 for item in todos if item["status"] == "done")
        summary = (
            f"Todo list updated: {done_count}/{len(todos)} done; "
            f"active={active_todo_id or 'none'}; blocked={blocked_todo_id or 'none'}."
        )
        return f"{summary}\n<summary-for-history>{summary}</summary-for-history>"

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
        prepared = self.prepare_shell_command(command)
        task_id = self.new_background_task_id()
        record = self.background_tasks.start(
            run_id=self._background_task_run_id(),
            task_id=task_id,
            command=prepared.command,
            original_command=command,
            cwd=self.root,
            env=prepared.env,
            timeout=timeout,
            python_env_used=prepared.python_env_used,
            python_env_path=prepared.python_env_path,
            repair_authorized=prepared.python_env_used,
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
            repair = (
                "; managed_python=.lumo/python-env; one restricted environment repair may run on failure"
                if self.python_environment.is_managed_python_command(args.get("command", ""))
                else ""
            )
            return f"command={json.dumps(command, ensure_ascii=False)}; timeout={timeout}{repair}"
        if name == "run_shell_bg":
            command = self._approval_preview(args.get("command", ""), 120)
            timeout = args.get("timeout", 3600)
            repair = (
                "; managed_python=.lumo/python-env; one restricted environment repair may run on failure"
                if self.python_environment.is_managed_python_command(args.get("command", ""))
                else ""
            )
            return f"command={json.dumps(command, ensure_ascii=False)}; background=true; timeout={timeout}{repair}"
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
