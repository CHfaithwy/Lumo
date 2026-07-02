"""Agent control loop extracted from the runtime facade."""

import asyncio
from dataclasses import dataclass, field
import re
import threading
import time

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .features import memory as memorylib
from .task_state import TaskState
from .tool_executor import strip_tool_hints
from . import tools as toolkit
from .workspace import IGNORED_PATH_NAMES, clip, now


NEGATED_TOOL_PREFIX_PATTERN = re.compile(
    r"(?:do\s+not\s+use|don't\s+use|never\s+use|avoid\s+using|\u4e0d\u8981\u7528|\u4e0d\u8981\u8c03\u7528|\u4e0d\u8981\u8c03|\u522b\u7528|\u7981\u6b62\u4f7f\u7528)\s*$",
    re.I,
)
CONDITIONAL_TOOL_PREFIX_PATTERN = re.compile(r"(?:if|when|unless|\u5982\u679c|\u82e5|\u5982)\s*$", re.I)
CONDITIONAL_TOOL_CLAUSE_PATTERN = re.compile(r"(?:\bif\b|\bwhen\b|\bunless\b|\u5982\u679c|\u82e5|\u5982)", re.I)
FULL_FILE_READ_REQUEST_PATTERN = re.compile(
    r"(?i)(fully\s+read|read\s+the\s+(?:full|whole|entire)\s+file|read\s+.*from\s+start\s+to\s+finish|read\s+it\s+all|\u5b8c\u6574(?:\u5730)?\u8bfb(?:\u53d6|\u5b8c)?|\u5b8c\u6574\u8bfb\u5b8c|\u8bfb\u5b8c\u6574\u4e2a|\u901a\u8bfb|\u4ece\u5934\u5230\u5c3e\u8bfb|\u8bfb\u5b8c(?:\u8fd9\u4e2a|\u8be5)?\u6587\u4ef6)",
    re.I,
)
TASK_POLL_UNTIL_DONE_PATTERN = re.compile(
    r"(?i)(poll\s+until|keep\s+polling|until\s+the\s+(?:background\s+)?task\s+(?:finishes|completes|exits)|wait\s+until\s+the\s+(?:background\s+)?task\s+(?:finishes|completes|exits)|\u8f6e\u8be2.*\u76f4\u5230.*(?:\u7ed3\u675f|\u5b8c\u6210|\u9000\u51fa)|\u76f4\u5230.*(?:\u540e\u53f0)?\u4efb\u52a1(?:\u7ed3\u675f|\u5b8c\u6210|\u9000\u51fa)|\u6301\u7eed\u67e5\u770b.*\u76f4\u5230.*(?:\u7ed3\u675f|\u5b8c\u6210))",
    re.I,
)
FILE_PATH_TOKEN_PATTERN = re.compile(
    r"`([^`\n]+?\.[A-Za-z0-9_+-]+)`|(?<![A-Za-z0-9_])([A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.[A-Za-z0-9_+-]+)(?![A-Za-z0-9_])"
)
TASK_ID_PATTERN = re.compile(r"\b(task_\d{8}[A-Za-z0-9_-]*)\b")


@dataclass(frozen=True)
class PendingObligation:
    key: str
    source: str
    reason: str
    required_tool: str
    suggested_args: dict
    blocks_completion: bool
    chain_key: str = ""
    completion_block_policy: str = ""


@dataclass
class ProgressState:
    logical_steps_used: int = 0
    raw_model_attempts: int = 0
    raw_tool_calls: int = 0
    no_progress_turn_streak: int = 0
    same_chain_no_progress_streak: int = 0
    same_action_no_progress_streak: int = 0
    last_active_chain_key: str = ""
    started_chain_keys: set = field(default_factory=set)
    last_chain_cursor_by_key: dict = field(default_factory=dict)


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _obligation_to_dict(obligation):
        return {
            "key": obligation.key,
            "source": obligation.source,
            "reason": obligation.reason,
            "required_tool": obligation.required_tool,
            "suggested_args": dict(obligation.suggested_args or {}),
            "blocks_completion": bool(obligation.blocks_completion),
            "chain_key": str(getattr(obligation, "chain_key", "") or ""),
            "completion_block_policy": str(getattr(obligation, "completion_block_policy", "") or ""),
        }

    def _run_history(self, history_start_index):
        return list(self.agent.session.get("history", [])[history_start_index:])

    def _tool_history(self, history_items):
        return [
            item
            for item in history_items
            if isinstance(item, dict) and item.get("role") == "tool"
        ]

    @staticmethod
    def _read_followup_key_from_item(item):
        if not isinstance(item, dict) or item.get("role") != "tool" or item.get("name") != "read_file":
            return ""
        args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        path = str(args.get("path", "")).strip()
        freshness = str(metadata.get("freshness", "")).strip()
        if not path or not freshness:
            return ""
        return f"read_file_continue:{path}:{freshness}"

    @staticmethod
    def _task_output_followup_key_from_item(item):
        if not isinstance(item, dict) or item.get("role") != "tool" or item.get("name") != "task_output":
            return ""
        args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
        task_id = str(args.get("task_id", "")).strip()
        stream = str(args.get("stream", "stdout")).strip() or "stdout"
        if not task_id:
            return ""
        return f"task_output_wait:{task_id}:{stream}"

    def _clear_satisfied_followup_obligations(self, active, item):
        if not isinstance(item, dict) or item.get("role") != "tool":
            return
        name = str(item.get("name", "")).strip()
        if name == "read_file":
            read_key = self._read_followup_key_from_item(item)
            if read_key:
                active.pop(read_key, None)
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            current_path = str(args.get("path", "")).strip()
            if current_path:
                for key, obligation in list(active.items()):
                    if obligation.reason != "git_diff_externalized":
                        continue
                    suggested_path = str((obligation.suggested_args or {}).get("path", "")).strip()
                    if suggested_path and suggested_path == current_path:
                        active.pop(key, None)
        elif name == "task_output":
            task_key = self._task_output_followup_key_from_item(item)
            if task_key:
                active.pop(task_key, None)

    def _followup_pending_obligations(self, history_items):
        active = {}
        for item in self._tool_history(history_items):
            self._clear_satisfied_followup_obligations(active, item)
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            followup_tool = str(metadata.get("followup_tool", "")).strip()
            followup_key = str(metadata.get("followup_key", "")).strip()
            is_blocking = bool(metadata.get("followup_is_blocking", metadata.get("blocks_completion", False)))
            if not followup_tool or not followup_key or not is_blocking:
                continue
            active[followup_key] = PendingObligation(
                key=followup_key,
                source="tool_followup",
                reason=str(metadata.get("followup_reason", "")).strip() or "tool_followup",
                required_tool=followup_tool,
                suggested_args=dict(metadata.get("followup_args", {}) or {}),
                blocks_completion=is_blocking,
                chain_key=str(metadata.get("chain_key", "")).strip(),
                completion_block_policy=str(metadata.get("completion_block_policy", "")).strip(),
            )
        return list(active.values())

    @staticmethod
    def _tool_occurrence_is_guarded(text, start_index):
        prefix = str(text or "")[max(0, int(start_index) - 24) : int(start_index)]
        compact = prefix.replace("`", "").replace('"', "").replace("'", "").strip()
        if NEGATED_TOOL_PREFIX_PATTERN.search(compact):
            return True
        if CONDITIONAL_TOOL_PREFIX_PATTERN.search(compact):
            return True
        full_prefix = str(text or "")[: int(start_index)]
        boundary = max(
            full_prefix.rfind("\n"),
            full_prefix.rfind("。"),
            full_prefix.rfind("！"),
            full_prefix.rfind("？"),
            full_prefix.rfind("；"),
            full_prefix.rfind(";"),
            full_prefix.rfind("."),
        )
        clause_fragment = full_prefix[boundary + 1 :].strip()
        if CONDITIONAL_TOOL_CLAUSE_PATTERN.search(clause_fragment):
            return True
        return False

    @staticmethod
    def _looks_like_step_line(text):
        stripped = str(text or "").strip()
        if not stripped:
            return False
        return bool(
            re.match(
                r"^(?:[-*•]\s+|\d+\s*[.)]\s+|[A-Za-z]\.\s+|(?:first|next|then|finally)\b|(?:先|再|然后|最後|最后)\b)",
                stripped,
                re.I,
            )
        )

    @staticmethod
    def _is_completion_requirement_line(text):
        stripped = str(text or "").strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        phrases = (
            "final answer",
            "answer should include",
            "when answering",
            "completion condition",
            "completed when",
            "最终回答",
            "最后回答",
            "回答时请",
            "完成条件",
            "最终请说明",
            "请明确说明",
        )
        return any(phrase in lowered for phrase in phrases)

    def _extract_first_tool_name(self, text):
        text = str(text or "")
        if not text.strip():
            return ""
        matches = []
        for name in sorted(toolkit.legal_tool_names()):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
            for match in pattern.finditer(text):
                if self._tool_occurrence_is_guarded(text, match.start()):
                    continue
                matches.append((match.start(), match.end(), name))
        if not matches:
            return ""
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return matches[0][2]

    def _extract_request_tool_sequence(self, text):
        text = str(text or "")
        if not text.strip():
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        step_lines = [
            line
            for line in lines
            if self._looks_like_step_line(line) and not self._is_completion_requirement_line(line)
        ]
        if step_lines:
            ordered = []
            for line in step_lines:
                name = self._extract_first_tool_name(line)
                if name and name not in ordered:
                    ordered.append(name)
            return ordered
        filtered_text = "\n".join(line for line in lines if not self._is_completion_requirement_line(line))
        ordered = []
        matches = []
        for name in sorted(toolkit.legal_tool_names()):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
            for match in pattern.finditer(filtered_text):
                if self._tool_occurrence_is_guarded(filtered_text, match.start()):
                    continue
                matches.append((match.start(), match.end(), name))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        for _start, _end, name in matches:
            if name not in ordered:
                ordered.append(name)
        return ordered

    @staticmethod
    def _dedupe_preserve_order(values):
        ordered = []
        for value in values:
            if value and value not in ordered:
                ordered.append(value)
        return ordered

    def _workspace_file_candidates(self, basename):
        basename = str(basename or "").strip()
        if not basename:
            return []
        matches = []
        try:
            for candidate in self.agent.root.rglob(basename):
                try:
                    relative = candidate.relative_to(self.agent.root)
                except ValueError:
                    continue
                if any(part in IGNORED_PATH_NAMES for part in relative.parts):
                    continue
                if candidate.is_file():
                    matches.append(relative.as_posix())
                if len(matches) >= 2:
                    break
        except Exception:
            return []
        return matches

    def _normalize_request_path_token(self, token, history_items):
        token = str(token or "").strip().strip("`").strip()
        if not token:
            return ""
        normalized = token.replace("\\", "/").lstrip("./")
        try:
            normalized = self.agent.path(token).relative_to(self.agent.root).as_posix()
        except Exception:
            normalized = token.replace("\\", "/").lstrip("./")
        if "/" in normalized:
            return normalized
        basename = normalized.rsplit("/", 1)[-1]
        for item in reversed(self._tool_history(history_items)):
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            candidate_path = str(args.get("path", "")).strip().replace("\\", "/").lstrip("./")
            if candidate_path and candidate_path.rsplit("/", 1)[-1] == basename:
                return candidate_path
        workspace_matches = self._workspace_file_candidates(basename)
        if len(workspace_matches) == 1:
            return workspace_matches[0]
        return normalized

    def _extract_request_file_targets(self, text, history_items):
        text = str(text or "")
        targets = []
        for match in FILE_PATH_TOKEN_PATTERN.finditer(text):
            token = match.group(1) or match.group(2) or ""
            normalized = self._normalize_request_path_token(token, history_items)
            if normalized:
                targets.append(normalized)
        return self._dedupe_preserve_order(targets)

    @staticmethod
    def _looks_like_full_read_request(text):
        text = str(text or "")
        if FULL_FILE_READ_REQUEST_PATTERN.search(text):
            return True
        lowered = text.lower()
        extra_markers = (
            "完整读完",
            "完整阅读",
            "读完整个",
            "通读",
            "read all",
            "read it all",
            "read the entire",
            "read the whole",
            "fully inspect",
        )
        return any(marker in text or marker in lowered for marker in extra_markers)

    @staticmethod
    def _path_matches_target(path_value, target):
        path_value = str(path_value or "").strip().replace("\\", "/").lstrip("./")
        target = str(target or "").strip().replace("\\", "/").lstrip("./")
        if not path_value or not target:
            return False
        if path_value == target:
            return True
        return path_value.rsplit("/", 1)[-1] == target.rsplit("/", 1)[-1]

    def _latest_read_state_for_target(self, target, history_items):
        for item in reversed(self._tool_history(history_items)):
            if item.get("name") != "read_file":
                continue
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            path_value = str(args.get("path", "")).strip()
            if not self._path_matches_target(path_value, target):
                continue
            path_key = self.agent._history_path_key(item)
            current_freshness = ""
            if path_key:
                try:
                    current_freshness = memorylib.file_freshness(path_key, self.agent.root)
                except Exception:
                    current_freshness = ""
            freshness = str(metadata.get("freshness", "")).strip()
            if current_freshness and freshness and freshness != current_freshness:
                continue
            read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
            return {
                "path": path_key or path_value.replace("\\", "/").lstrip("./"),
                "has_more": bool(read_window.get("has_more")),
                "metadata": metadata,
                "item": item,
            }
        return None

    def _request_full_read_pending_obligations(self, request_text, history_items):
        text = str(request_text or "")
        if not self._looks_like_full_read_request(text):
            return []
        targets = self._extract_request_file_targets(text, history_items)
        pending = []
        for target in targets:
            latest_state = self._latest_read_state_for_target(target, history_items)
            if latest_state and not latest_state.get("has_more", False):
                continue
            pending.append(
                PendingObligation(
                    key=f"request:full_read:{target}",
                    source="request",
                    reason="explicit_request_full_read",
                    required_tool="read_file",
                    suggested_args={"path": target, "offset": 1, "limit": 200},
                    blocks_completion=True,
                    chain_key=f"request_full_read:{target}",
                    completion_block_policy="until_eof",
                )
            )
        return pending

    @staticmethod
    def _extract_request_task_ids(text):
        return AgentLoop._dedupe_preserve_order(match.group(1) for match in TASK_ID_PATTERN.finditer(str(text or "")))

    def _latest_background_task_id(self, history_items):
        for item in reversed(self._tool_history(history_items)):
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            task_id = str(metadata.get("background_task_id", "")).strip()
            if task_id:
                return task_id
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            task_id = str(args.get("task_id", "")).strip()
            if task_id:
                return task_id
        return ""

    @staticmethod
    def _task_is_terminal(status):
        return str(status or "").strip() in {"exited", "failed", "stopped"}

    def _terminal_task_output_seen(self, history_items, task_id):
        task_id = str(task_id or "").strip()
        if not task_id:
            return False
        for item in reversed(self._tool_history(history_items)):
            if item.get("name") != "task_output":
                continue
            args = item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}
            if str(args.get("task_id", "")).strip() != task_id:
                continue
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            if self._task_is_terminal(metadata.get("background_task_status", "")):
                return True
        return False

    def _request_task_poll_pending_obligations(self, request_text, history_items):
        text = str(request_text or "")
        if not TASK_POLL_UNTIL_DONE_PATTERN.search(text):
            return []
        task_ids = self._extract_request_task_ids(text)
        if not task_ids:
            latest_task_id = self._latest_background_task_id(history_items)
            if latest_task_id:
                task_ids = [latest_task_id]
        pending = []
        for task_id in task_ids:
            if self._terminal_task_output_seen(history_items, task_id):
                continue
            pending.append(
                PendingObligation(
                    key=f"request:task_poll:{task_id}",
                    source="request",
                    reason="explicit_request_task_poll_until_done",
                    required_tool="task_output",
                    suggested_args={"task_id": task_id, "offset": 0, "limit": 4000, "stream": "stdout"},
                    blocks_completion=True,
                    chain_key=f"request_task_poll:{task_id}:stdout",
                    completion_block_policy="until_terminal",
                )
            )
        return pending

    def _request_pending_obligations(self, request_text, history_items):
        required_sequence = self._extract_request_tool_sequence(request_text)
        pending = []
        if required_sequence:
            executed_tools = [str(item.get("name", "")).strip() for item in self._tool_history(history_items)]
            matched_count = 0
            for name in executed_tools:
                if matched_count >= len(required_sequence):
                    break
                if name == required_sequence[matched_count]:
                    matched_count += 1
            for index, name in enumerate(required_sequence[matched_count:], start=matched_count):
                pending.append(
                    PendingObligation(
                        key=f"request:{index}:{name}",
                        source="request",
                        reason="explicit_request_tool_order",
                        required_tool=name,
                        suggested_args={},
                        blocks_completion=True,
                        chain_key=f"request_tool_order:{index}:{name}",
                        completion_block_policy="until_action",
                    )
                )
        pending.extend(self._request_full_read_pending_obligations(request_text, history_items))
        pending.extend(self._request_task_poll_pending_obligations(request_text, history_items))
        return pending

    def _current_pending_obligations(self, request_text, history_items):
        request_obligations = self._request_pending_obligations(request_text, history_items)
        followup_obligations = self._followup_pending_obligations(history_items)
        combined = []
        seen_keys = set()
        for obligation in [*request_obligations, *followup_obligations]:
            dedupe_key = str(obligation.chain_key or obligation.key)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            combined.append(obligation)
        return combined

    @staticmethod
    def _active_blocking_obligation(obligations):
        for item in obligations:
            if item.blocks_completion:
                return item
        return None

    @staticmethod
    def _completion_block_priority(policy):
        order = {
            "until_action": 1,
            "until_opened": 2,
            "until_terminal": 3,
            "until_eof": 4,
        }
        return int(order.get(str(policy or "").strip(), 0))

    def _matching_execution_obligation(self, name, args, obligations):
        matches = [
            obligation
            for obligation in list(obligations or [])
            if obligation.blocks_completion and self._tool_matches_obligation(name, args, obligation)
        ]
        if not matches:
            return None
        matches.sort(
            key=lambda obligation: (
                self._completion_block_priority(obligation.completion_block_policy),
                1 if obligation.source == "tool_followup" else 0,
            ),
            reverse=True,
        )
        return matches[0]

    @staticmethod
    def _obligation_missing_identity_reason(obligation):
        if not isinstance(obligation, PendingObligation):
            return ""
        args = dict(obligation.suggested_args or {})
        if obligation.required_tool == "read_file" and (
            obligation.source == "tool_followup" or obligation.reason == "explicit_request_full_read"
        ):
            if not str(args.get("path", "")).strip():
                return "missing_path"
        if obligation.required_tool == "task_output" and (
            obligation.source == "tool_followup" or obligation.reason == "explicit_request_task_poll_until_done"
        ):
            if not str(args.get("task_id", "")).strip():
                return "missing_task_id"
        return ""

    def _required_tool_unavailable_reason(self, obligation):
        if not isinstance(obligation, PendingObligation):
            return ""
        required_tool = str(obligation.required_tool or "").strip()
        if not required_tool:
            return ""
        if required_tool not in self.agent.tools:
            return "tool_not_registered"
        if self.agent.allowed_tools is not None and required_tool not in self.agent.allowed_tools:
            return "tool_not_allowed"
        return self._obligation_missing_identity_reason(obligation)

    def _tool_matches_obligation(self, name, args, obligation):
        if not isinstance(obligation, PendingObligation):
            return False
        if str(name or "").strip() != str(obligation.required_tool or "").strip():
            return False
        args = args if isinstance(args, dict) else {}
        suggested_args = dict(obligation.suggested_args or {})
        if obligation.required_tool == "read_file":
            suggested_path = str(suggested_args.get("path", "")).strip()
            if suggested_path and not self._path_matches_target(args.get("path", ""), suggested_path):
                return False
        if obligation.required_tool == "task_output":
            suggested_task_id = str(suggested_args.get("task_id", "")).strip()
            if suggested_task_id and str(args.get("task_id", "")).strip() != suggested_task_id:
                return False
            suggested_stream = str(suggested_args.get("stream", "")).strip()
            if suggested_stream and str(args.get("stream", "stdout")).strip() != suggested_stream:
                return False
        return True

    @staticmethod
    def _artifact_like_path(path):
        normalized = str(path or "").strip().replace("\\", "/").lstrip("./")
        return normalized.endswith(".patch") and "/runs/" in normalized

    def _derive_progress_chain_key(self, name, args, metadata, active_obligation=None):
        name = str(name or "").strip()
        args = args if isinstance(args, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        if self._tool_matches_obligation(name, args, active_obligation):
            return str(active_obligation.chain_key or active_obligation.key or "").strip()
        metadata_chain_key = str(metadata.get("chain_key", "")).strip()
        if metadata_chain_key:
            return metadata_chain_key
        if name == "read_file":
            path = str(args.get("path", "")).strip()
            if self._artifact_like_path(path):
                return f"git_diff_artifact:{path.replace('\\', '/').lstrip('./')}"
            freshness = str(metadata.get("freshness", "")).strip()
            if freshness:
                return f"read_file:{path}:{freshness}"
            return f"read_file:{path}"
        if name == "task_output":
            task_id = str(args.get("task_id", "")).strip()
            stream = str(args.get("stream", "stdout")).strip() or "stdout"
            return f"task_output:{task_id}:{stream}"
        if name == "git_diff":
            path = str(args.get("path", ".")).strip() or "."
            mode = str(args.get("mode", "workspace")).strip() or "workspace"
            return f"git_diff:{path}:{mode}"
        if name == "git_status":
            return f"git_status:{str(args.get('path', '.')).strip() or '.'}"
        if name == "grep":
            path = str(args.get("path", ".")).strip() or "."
            pattern = str(args.get("pattern", "")).strip()
            glob = str(args.get("glob", "")).strip()
            mode = str(args.get("output_mode", "content")).strip() or "content"
            return f"grep:{path}:{pattern}:{glob}:{mode}"
        if name == "glob":
            path = str(args.get("path", ".")).strip() or "."
            pattern = str(args.get("pattern", "")).strip()
            return f"glob:{path}:{pattern}"
        if name == "list_files":
            return f"list_files:{str(args.get('path', '.')).strip() or '.'}"
        if name == "run_shell_bg":
            return f"run_shell_bg:{str(args.get('command', '')).strip()}"
        if name == "run_shell":
            return f"run_shell:{str(args.get('command', '')).strip()}"
        if name == "task_list":
            status = str(args.get("status", "all")).strip() or "all"
            offset = int(args.get("offset", 0) or 0)
            return f"task_list:{status}:{offset}"
        if name == "write_file":
            return f"write_file:{str(args.get('path', '')).strip()}"
        if name == "patch_file":
            return f"patch_file:{str(args.get('path', '')).strip()}"
        return f"{name}:{str(args)}"

    def _progress_cursor(self, name, args, metadata):
        name = str(name or "").strip()
        args = args if isinstance(args, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        if name == "read_file":
            read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
            path = str(args.get("path", "")).strip()
            freshness = str(metadata.get("freshness", "")).strip()
            end_line = int(read_window.get("end_line", 0) or 0)
            has_more = bool(read_window.get("has_more", False))
            return f"{path}:{freshness}:{end_line}:{int(has_more)}"
        if name == "task_output":
            task_id = str(args.get("task_id", "")).strip()
            stream = str(args.get("stream", "stdout")).strip() or "stdout"
            status = str(metadata.get("background_task_status", "")).strip()
            next_offset = int(metadata.get("background_task_next_offset", args.get("offset", 0) or 0) or 0)
            return_code = metadata.get("background_task_return_code")
            return f"{task_id}:{stream}:{status}:{next_offset}:{return_code}"
        if name == "git_diff":
            externalized_path = str(metadata.get("externalized_patch_path", "")).strip()
            if externalized_path:
                return f"externalized:{externalized_path}"
            offset = int(args.get("offset", 1) or 1)
            limit = int(args.get("limit", 300) or 300)
            return f"inline:{offset}:{limit}"
        if name == "git_status":
            offset = int(args.get("offset", 0) or 0)
            limit = int(args.get("limit", 200) or 200)
            return f"status:{offset}:{limit}"
        if name == "grep":
            offset = int(args.get("offset", 0) or 0)
            head_limit = int(args.get("head_limit", 200) or 200)
            return f"grep:{offset}:{head_limit}"
        if name == "glob":
            return f"glob:{str(args.get('path', '.')).strip()}:{str(args.get('pattern', '')).strip()}"
        if name == "list_files":
            return f"list_files:{str(args.get('path', '.')).strip()}"
        if name == "run_shell_bg":
            return str(metadata.get("background_task_id", "")).strip()
        if name == "run_shell":
            return f"run_shell:{str(args.get('command', '')).strip()}:{str(metadata.get('tool_status', '')).strip()}"
        if name == "task_list":
            offset = int(args.get("offset", 0) or 0)
            status = str(args.get("status", "all")).strip() or "all"
            return f"task_list:{status}:{offset}"
        if name == "write_file":
            return f"write_file:{str(args.get('path', '')).strip()}"
        if name == "patch_file":
            return f"patch_file:{str(args.get('path', '')).strip()}:{int(bool(args.get('replace_all', False)))}"
        return ""

    def _detect_progress_event(self, name, args, metadata, active_obligation, progress_state):
        tool_status = str((metadata or {}).get("tool_status", "")).strip()
        if tool_status not in {"ok", "partial_success"}:
            return {"progressed": False, "chain_key": self._derive_progress_chain_key(name, args, metadata, active_obligation), "cursor": "", "opened_new_chain": False}
        chain_key = self._derive_progress_chain_key(name, args, metadata, active_obligation)
        cursor = self._progress_cursor(name, args, metadata)
        opened_new_chain = chain_key not in progress_state.started_chain_keys
        last_cursor = progress_state.last_chain_cursor_by_key.get(chain_key)
        progressed = opened_new_chain or cursor != last_cursor
        return {
            "progressed": bool(progressed),
            "chain_key": chain_key,
            "cursor": cursor,
            "opened_new_chain": bool(opened_new_chain and progressed),
        }

    @staticmethod
    def _tool_action_signature(name, args, chain_key):
        return f"{str(chain_key or '')}|{str(name or '')}|{repr(dict(args or {}))}"

    @staticmethod
    def _tool_failure_signature(name, args, metadata, result, chain_key=""):
        metadata = metadata if isinstance(metadata, dict) else {}
        tool_status = str(metadata.get("tool_status", "")).strip()
        if tool_status in {"", "ok", "partial_success"}:
            return ""
        tool_error_code = str(metadata.get("tool_error_code", "")).strip()
        return f"{str(chain_key or '')}|{str(name or '')}|{tool_error_code}|{repr(dict(args or {}))}|{clip(result, 200)}"

    @staticmethod
    def _progress_state_dict(progress_state):
        return {
            "logical_steps_used": int(progress_state.logical_steps_used),
            "raw_model_attempts": int(progress_state.raw_model_attempts),
            "raw_tool_calls": int(progress_state.raw_tool_calls),
            "no_progress_turn_streak": int(progress_state.no_progress_turn_streak),
            "same_chain_no_progress_streak": int(progress_state.same_chain_no_progress_streak),
            "same_action_no_progress_streak": int(progress_state.same_action_no_progress_streak),
            "last_active_chain_key": str(progress_state.last_active_chain_key or ""),
            "started_chain_keys": sorted(str(key) for key in progress_state.started_chain_keys),
            "last_chain_cursor_by_key": {str(key): str(value) for key, value in progress_state.last_chain_cursor_by_key.items()},
        }

    async def _force_summary_reply_async(self, prompt):
        fallback_prompt = "\n\n".join(
            [
                prompt,
                "Runtime fallback:",
                "The runtime decided to stop normal planning because it cannot make reliable progress from the current state.",
                "Stop using tools and write the best direct reply to the user now using only the evidence already available in this prompt.",
                "Respond in the user's language. Be concise and concrete.",
                "If the evidence is sufficient, answer directly.",
                "If the evidence is insufficient, clearly state what can already be concluded and what remains uncertain.",
                "Return plain answer text only. Do not use <tool> tags, <final> tags, or <completion> tags.",
            ]
        )
        return await self._complete_model_async(prompt=fallback_prompt)

    async def _finish_with_forced_summary(
        self,
        task_state,
        original_user_message,
        prompt,
        *,
        reason,
        attempts,
        tool_steps,
        run_started_at,
        previous_completion_score=None,
        current_completion_score=None,
        missing_completion_streak=0,
        obligation=None,
        unavailable_detail="",
    ):
        agent = self.agent
        agent.emit_trace(
            task_state,
            "forced_summary_requested",
            {
                "attempts": attempts,
                "tool_steps": tool_steps,
                "reason": reason,
                "previous_completion_score": previous_completion_score,
                "current_completion_score": current_completion_score,
                "missing_completion_streak": missing_completion_streak,
                "pending_obligation": self._obligation_to_dict(obligation) if obligation else None,
                "required_tool_unavailable_detail": str(unavailable_detail or ""),
            },
        )
        fallback_started_at = time.monotonic()
        fallback_raw = await self._force_summary_reply_async(prompt)
        fallback_final = self._coerce_fallback_final(fallback_raw)
        agent.record({"role": "assistant", "content": fallback_final, "created_at": now()})
        task_state.update_progress_state(
            chain=str(getattr(obligation, "chain_key", "") or task_state.last_progress_chain or ""),
            cursor=str(task_state.last_progress_cursor or ""),
            stall_reason=reason,
        )
        task_state.finish_success(fallback_final)
        checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="forced_summary")
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": "forced_summary",
            },
        )
        agent.emit_trace(
            task_state,
            "forced_summary_finished",
            {
                "reason": reason,
                "duration_ms": int((time.monotonic() - fallback_started_at) * 1000),
                "final_answer": fallback_final,
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": fallback_final,
                "forced_summary_reason": reason,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        agent.clear_transient_runtime_requirements()
        return fallback_final

    def _coerce_fallback_final(self, raw):
        raw = str(raw or "").strip()
        if not raw:
            return "I could not produce a properly formatted final answer, and the fallback summary was empty."
        if "<final>" in raw:
            final = self.agent.extract(raw, "final").strip()
            if final:
                return self.agent.strip_completion_tags(final).strip()
        return self.agent.strip_completion_tags(raw).strip()

    def run(self, user_message):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(user_message))

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(self.run_async(user_message))
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def run_async(self, user_message):
        return await self._run(user_message)

    async def _complete_model_async(self, prompt, prompt_cache_key=None, prompt_cache_retention=None):
        agent = self.agent
        return await agent.complete_text_async(
            prompt,
            max_new_tokens=agent.max_new_tokens,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

    async def _run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        original_user_message = str(user_message)
        agent.memory.set_task_summary(original_user_message)
        run_history_start_index = len(agent.session.get("history", []))
        agent.record({"role": "user", "content": original_user_message, "created_at": now()})

        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=original_user_message,
        )

        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)

        agent.current_task_state = task_state

        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(original_user_message, 300),
            },
        )
        effective_user_message = await agent.rewrite_user_message_async(original_user_message)
        rewrite_metadata = dict(getattr(agent, "last_user_request_rewrite", {}) or {})
        if rewrite_metadata.get("enabled"):
            agent.emit_trace(
                task_state,
                "user_request_rewritten",
                {
                    **rewrite_metadata,
                    "rewritten_request": clip(effective_user_message, 300),
                },
            )

        progress_state = ProgressState()
        raw_tool_call_backstop = max(agent.max_steps * 8, agent.max_steps + 24)
        raw_attempt_backstop = max(agent.max_steps * 12, agent.max_steps + 36)
        forced_summary_used = False
        previous_scored_completion = None
        missing_completion_streak = 0
        request_text_for_obligations = effective_user_message if str(effective_user_message or "").strip() else original_user_message
        obligation_notice_rounds = 0
        retry_streak = 0
        obligation_non_action_streak = 0
        last_active_obligation_key = ""
        last_action_signature = ""
        last_failure_signature = ""
        same_chain_failure_streak = 0
        last_prompt = ""
        budget_reason = ""

        while True:
            prompt_pending_obligations = self._current_pending_obligations(
                request_text_for_obligations,
                self._run_history(run_history_start_index),
            )
            active_budget_obligation = self._active_blocking_obligation(
                [item for item in prompt_pending_obligations if item.blocks_completion]
            )
            allow_same_chain_continuation = bool(
                active_budget_obligation
                and str(active_budget_obligation.chain_key or "").strip()
                and str(active_budget_obligation.chain_key).strip() in progress_state.started_chain_keys
            )
            if progress_state.logical_steps_used >= agent.max_steps and not allow_same_chain_continuation:
                budget_reason = "logical_step_budget_exhausted"
                break
            if progress_state.raw_tool_calls >= raw_tool_call_backstop:
                budget_reason = "raw_tool_call_backstop_exhausted"
                break
            if progress_state.raw_model_attempts >= raw_attempt_backstop:
                budget_reason = "raw_attempt_backstop_exhausted"
                break

            progress_state.raw_model_attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            agent.set_transient_runtime_requirements(
                [self._obligation_to_dict(item) for item in prompt_pending_obligations]
            )
            obligation_notice_rounds = obligation_notice_rounds + 1 if prompt_pending_obligations else 0

            prompt, prompt_metadata = await agent._build_prompt_and_metadata_async(effective_user_message)
            last_prompt = prompt
            prompt_metadata["pending_obligations"] = [self._obligation_to_dict(item) for item in prompt_pending_obligations]
            prompt_metadata["pending_obligations_count"] = len(prompt_pending_obligations)
            prompt_path = agent.run_store.write_prompt(task_state, progress_state.raw_model_attempts, agent.redact_text(prompt))
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "prompt_path": str(prompt_path),
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="freshness_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            active_prompt_blocking_obligation = self._active_blocking_obligation(prompt_pending_obligations)
            required_tool_unavailable_reason = self._required_tool_unavailable_reason(active_prompt_blocking_obligation)
            if active_prompt_blocking_obligation and required_tool_unavailable_reason:
                if not forced_summary_used:
                    forced_summary_used = True
                    return await self._finish_with_forced_summary(
                        task_state,
                        original_user_message,
                        prompt,
                        reason="required_tool_unavailable",
                        attempts=progress_state.raw_model_attempts,
                        tool_steps=progress_state.logical_steps_used,
                        run_started_at=run_started_at,
                        previous_completion_score=previous_scored_completion,
                        current_completion_score=None,
                        missing_completion_streak=missing_completion_streak,
                        obligation=active_prompt_blocking_obligation,
                        unavailable_detail=required_tool_unavailable_reason,
                    )
                final = "Stopped because the required tool for the current obligation is unavailable."
                task_state.stop_retry_limit(final)
                agent.record({"role": "assistant", "content": final, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "run_finished",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "final_answer": final,
                        "forced_summary_reason": "required_tool_unavailable",
                        "required_tool_unavailable_detail": required_tool_unavailable_reason,
                        "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                    },
                )
                agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
                agent.clear_transient_runtime_requirements()
                return final
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = await self._complete_model_async(
                prompt,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind, payload = agent.parse(raw)
            current_completion_score = None
            if isinstance(payload, dict):
                current_completion_score = payload.get("completion_score")
            if current_completion_score is not None:
                current_completion_score = int(current_completion_score)
                agent.last_completion_score = current_completion_score
                missing_completion_streak = 0
            else:
                missing_completion_streak += 1
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "previous_completion_score": previous_scored_completion,
                    "current_completion_score": current_completion_score,
                    "missing_completion_streak": missing_completion_streak,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )
            current_pending_obligations = self._current_pending_obligations(
                request_text_for_obligations,
                self._run_history(run_history_start_index),
            )
            blocking_obligations = [item for item in current_pending_obligations if item.blocks_completion]
            active_blocking_obligation = self._active_blocking_obligation(blocking_obligations)
            active_obligation_key = str(active_blocking_obligation.key) if active_blocking_obligation else ""

            if kind == "tool":
                retry_streak = 0
                name = payload.get("name", "")
                args = payload.get("args", {})
                original_args = dict(args) if isinstance(args, dict) else {}
                execution_obligation = self._matching_execution_obligation(name, args, blocking_obligations)
                auto_read_decision = None
                if name == "read_file":
                    auto_read_decision = agent.auto_continue_read_file_args(args)
                    if auto_read_decision.get("status") == "continued":
                        args = dict(auto_read_decision.get("args", {}))
                progress_state.raw_tool_calls += 1
                task_state.record_raw_tool_call(name)
                tool_started_at = time.monotonic()
                execution_context = {
                    "active_obligation": self._obligation_to_dict(execution_obligation) if execution_obligation else {},
                }
                agent.report_tool_call(name, args)
                if name == "read_file" and auto_read_decision and auto_read_decision.get("status") == "fully_read":
                    tool_result = agent.synthetic_fully_read_result(
                        path=original_args.get("path", ""),
                        requested_offset=auto_read_decision.get("requested_offset", 1),
                        limit=auto_read_decision.get("limit", 1),
                        coverage=auto_read_decision.get("coverage", {}),
                    )
                else:
                    tool_result = agent.execute_tool(name, args, execution_context=execution_context)
                if name == "read_file":
                    agent.report_tool_result(name, args, tool_result.metadata, content=tool_result.content)
                result = tool_result.content
                archive_summary = str((tool_result.metadata or {}).get("archive_summary", "")).strip()
                stored_content = result
                if name in {"read_file", "run_shell_bg", "task_output", "task_list", "task_stop"} and archive_summary:
                    stored_content = archive_summary
                elif name in {"git_status", "git_diff"}:
                    stored_content = strip_tool_hints(result)
                tool_record = {
                    "role": "tool",
                    "name": name,
                    "args": args,
                    "content": stored_content,
                    "created_at": now(),
                    "metadata": dict(tool_result.metadata or {}),
                }
                if archive_summary:
                    tool_record["summary"] = archive_summary
                agent.record(tool_record)
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "requested_args": original_args if name == "read_file" and auto_read_decision else args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )
                progress_event = self._detect_progress_event(
                    name,
                    args,
                    tool_result.metadata,
                    execution_obligation,
                    progress_state,
                )
                chain_key = str(progress_event.get("chain_key", "")).strip()
                cursor = str(progress_event.get("cursor", "")).strip()
                action_signature = self._tool_action_signature(name, args, chain_key)
                failure_signature = self._tool_failure_signature(name, args, tool_result.metadata, result, chain_key=chain_key)
                stall_reason = ""
                if progress_event.get("progressed"):
                    if progress_event.get("opened_new_chain"):
                        progress_state.logical_steps_used += 1
                        task_state.record_logical_step(name)
                        progress_state.started_chain_keys.add(chain_key)
                    progress_state.last_chain_cursor_by_key[chain_key] = cursor
                    progress_state.no_progress_turn_streak = 0
                    progress_state.same_chain_no_progress_streak = 0
                    progress_state.same_action_no_progress_streak = 0
                    progress_state.last_active_chain_key = chain_key
                    obligation_non_action_streak = 0
                    last_active_obligation_key = ""
                    last_action_signature = action_signature
                    last_failure_signature = ""
                    same_chain_failure_streak = 0
                    task_state.update_progress_state(chain=chain_key, cursor=cursor, stall_reason="")
                else:
                    progress_state.no_progress_turn_streak += 1
                    if chain_key and chain_key == progress_state.last_active_chain_key:
                        progress_state.same_chain_no_progress_streak += 1
                    else:
                        progress_state.same_chain_no_progress_streak = 1 if chain_key else 0
                        progress_state.last_active_chain_key = chain_key
                    if action_signature and action_signature == last_action_signature:
                        progress_state.same_action_no_progress_streak += 1
                    else:
                        progress_state.same_action_no_progress_streak = 1 if action_signature else 0
                    last_action_signature = action_signature
                    if failure_signature and chain_key:
                        if failure_signature == last_failure_signature:
                            same_chain_failure_streak += 1
                        else:
                            same_chain_failure_streak = 1
                        last_failure_signature = failure_signature
                    else:
                        same_chain_failure_streak = 0
                        if not failure_signature:
                            last_failure_signature = ""
                    stall_reason = "same_chain_no_progress" if progress_state.same_chain_no_progress_streak >= 1 else ""
                    if active_obligation_key:
                        last_active_obligation_key = active_obligation_key
                    task_state.update_progress_state(chain=chain_key, cursor=cursor, stall_reason=stall_reason)
                    if (
                        same_chain_failure_streak >= 2
                        and failure_signature
                        and not forced_summary_used
                    ):
                        forced_summary_used = True
                        return await self._finish_with_forced_summary(
                            task_state,
                            original_user_message,
                            prompt,
                            reason="repeated_unrecoverable_tool_failure",
                            attempts=progress_state.raw_model_attempts,
                            tool_steps=progress_state.logical_steps_used,
                            run_started_at=run_started_at,
                            previous_completion_score=previous_scored_completion,
                            current_completion_score=current_completion_score,
                            missing_completion_streak=missing_completion_streak,
                            obligation=active_blocking_obligation,
                        )
                    if (
                        progress_state.same_chain_no_progress_streak >= 3
                        and chain_key
                        and not forced_summary_used
                    ):
                        forced_summary_used = True
                        return await self._finish_with_forced_summary(
                            task_state,
                            original_user_message,
                            prompt,
                            reason="no_progress_on_same_chain",
                            attempts=progress_state.raw_model_attempts,
                            tool_steps=progress_state.logical_steps_used,
                            run_started_at=run_started_at,
                            previous_completion_score=previous_scored_completion,
                            current_completion_score=current_completion_score,
                            missing_completion_streak=missing_completion_streak,
                            obligation=active_blocking_obligation,
                        )
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                retry_streak += 1
                retry_payload = payload if isinstance(payload, dict) else {"notice": str(payload or ""), "problem": "", "raw_text": ""}
                retry_raw_text = str(retry_payload.get("raw_text", "")).strip()
                if blocking_obligations:
                    obligation_non_action_streak += 1
                    if active_obligation_key and active_obligation_key != last_active_obligation_key:
                        obligation_non_action_streak = 1
                    last_active_obligation_key = active_obligation_key
                    if active_blocking_obligation is not None:
                        agent.report_obligation_progress(
                            self._obligation_to_dict(active_blocking_obligation)
                        )
                    progress_state.no_progress_turn_streak += 1
                    task_state.update_progress_state(
                        chain=active_blocking_obligation.chain_key if active_blocking_obligation else "",
                        cursor="",
                        stall_reason="pending_obligation_non_action",
                    )
                    agent.emit_trace(
                        task_state,
                        "retry_suppressed_due_to_pending_obligations",
                        {
                            "current_completion_score": current_completion_score,
                            "missing_completion_streak": missing_completion_streak,
                            "retry_excerpt": clip(retry_raw_text, 300),
                            "pending_obligations": [self._obligation_to_dict(item) for item in blocking_obligations],
                            "obligation_non_action_streak": obligation_non_action_streak,
                            "progress_state": self._progress_state_dict(progress_state),
                        },
                    )
                    if (
                        obligation_non_action_streak >= 2
                        and obligation_notice_rounds > 0
                        and not forced_summary_used
                    ):
                        forced_summary_used = True
                        return await self._finish_with_forced_summary(
                            task_state,
                            original_user_message,
                            prompt,
                            reason="repeated_refusal_to_act_on_obligations",
                            attempts=progress_state.raw_model_attempts,
                            tool_steps=progress_state.logical_steps_used,
                            run_started_at=run_started_at,
                            previous_completion_score=previous_scored_completion,
                            current_completion_score=current_completion_score,
                            missing_completion_streak=missing_completion_streak,
                            obligation=active_blocking_obligation,
                        )
                else:
                    obligation_non_action_streak = 0
                    last_active_obligation_key = ""
                    task_state.update_progress_state(chain="", cursor="", stall_reason="unusable_model_output")
                    if retry_streak >= 2 and not forced_summary_used:
                        forced_summary_used = True
                        return await self._finish_with_forced_summary(
                            task_state,
                            original_user_message,
                            prompt,
                            reason="consecutive_unusable_model_outputs",
                            attempts=progress_state.raw_model_attempts,
                            tool_steps=progress_state.logical_steps_used,
                            run_started_at=run_started_at,
                            previous_completion_score=previous_scored_completion,
                            current_completion_score=current_completion_score,
                            missing_completion_streak=missing_completion_streak,
                        )
                    if retry_raw_text:
                        agent.report_assistant_message(retry_raw_text)
                        agent.record_history_item({"role": "assistant", "content": retry_raw_text, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            answer_payload = payload if isinstance(payload, dict) else {"text": str(payload or "").strip(), "completion_score": current_completion_score, "raw_text": raw}
            answer_text = str(answer_payload.get("text", "")).strip()
            retry_streak = 0

            if blocking_obligations:
                obligation_non_action_streak += 1
                if active_obligation_key and active_obligation_key != last_active_obligation_key:
                    obligation_non_action_streak = 1
                last_active_obligation_key = active_obligation_key
                if active_blocking_obligation is not None:
                    agent.report_obligation_progress(
                        self._obligation_to_dict(active_blocking_obligation)
                    )
                progress_state.no_progress_turn_streak += 1
                task_state.update_progress_state(
                    chain=active_blocking_obligation.chain_key if active_blocking_obligation else "",
                    cursor="",
                    stall_reason="pending_obligation_non_action",
                )
                agent.emit_trace(
                    task_state,
                    "answer_suppressed_due_to_pending_obligations",
                    {
                        "current_completion_score": current_completion_score,
                        "missing_completion_streak": missing_completion_streak,
                        "answer_excerpt": clip(answer_text, 300),
                        "pending_obligations": [self._obligation_to_dict(item) for item in blocking_obligations],
                        "obligation_non_action_streak": obligation_non_action_streak,
                        "progress_state": self._progress_state_dict(progress_state),
                    },
                )
                if (
                    obligation_non_action_streak >= 2
                    and obligation_notice_rounds > 0
                    and not forced_summary_used
                ):
                    forced_summary_used = True
                    return await self._finish_with_forced_summary(
                        task_state,
                        original_user_message,
                        prompt,
                        reason="repeated_refusal_to_act_on_obligations",
                        attempts=progress_state.raw_model_attempts,
                        tool_steps=progress_state.logical_steps_used,
                        run_started_at=run_started_at,
                        previous_completion_score=previous_scored_completion,
                        current_completion_score=current_completion_score,
                        missing_completion_streak=missing_completion_streak,
                        obligation=active_blocking_obligation,
                    )
                agent.run_store.write_task_state(task_state)
                continue

            stop_reason = ""
            obligation_non_action_streak = 0
            last_active_obligation_key = ""
            if current_completion_score is not None and current_completion_score >= 95:
                stop_reason = "completion_score_threshold"

            if current_completion_score is not None:
                previous_scored_completion = current_completion_score

            if not stop_reason:
                if answer_text:
                    agent.report_assistant_message(answer_text)
                    agent.record_history_item({"role": "assistant", "content": answer_text, "created_at": now()})
                task_state.update_progress_state(chain="", cursor="", stall_reason="")
                agent.run_store.write_task_state(task_state)
                continue

            final = answer_text or self._coerce_fallback_final(raw)
            agent.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="run_finished")
            agent.run_store.write_task_state(task_state)
            agent.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            agent.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "completion_stop_reason": stop_reason,
                    "current_completion_score": current_completion_score,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
            agent.clear_transient_runtime_requirements()
            return final

        if not forced_summary_used and last_prompt:
            forced_summary_used = True
            return await self._finish_with_forced_summary(
                task_state,
                original_user_message,
                last_prompt,
                reason=budget_reason,
                attempts=progress_state.raw_model_attempts,
                tool_steps=progress_state.logical_steps_used,
                run_started_at=run_started_at,
                previous_completion_score=previous_scored_completion,
                current_completion_score=None,
                missing_completion_streak=missing_completion_streak,
            )
        if budget_reason == "raw_attempt_backstop_exhausted":
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        elif budget_reason == "raw_tool_call_backstop_exhausted":
            final = "Stopped after too many tool calls without reaching a reliable conclusion."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        agent.clear_transient_runtime_requirements()
        return final
