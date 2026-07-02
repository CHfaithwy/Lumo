"""Structured tool execution for the agent runtime."""

from dataclasses import dataclass
import re

from .features import memory as memorylib
from .workspace import clip

SUMMARY_FOR_HISTORY_PATTERN = re.compile(r"<summary-for-history>(.*?)</summary-for-history>", re.DOTALL)
TOOL_REMINDER_PATTERN = re.compile(r"<tool_reminder>(.*?)</tool_reminder>", re.DOTALL)
READ_FILE_HEADER_PATTERN = re.compile(
    r"# lines\s+(\d+)-(\d+)\s+of at least\s+(\d+)\s+\(returned\s+(\d+),\s+requested\s+(\d+)\)"
)
BACKGROUND_TASK_ID_PATTERN = re.compile(r"^task_id:\s*(.+)$", re.MULTILINE)
BACKGROUND_TASK_STATUS_PATTERN = re.compile(r"^status:\s*(.+)$", re.MULTILINE)
BACKGROUND_TASK_RETURN_CODE_PATTERN = re.compile(r"^return_code:\s*(.+)$", re.MULTILINE)
BACKGROUND_TASK_NEXT_OFFSET_PATTERN = re.compile(r"^next_offset:\s*(\d+)\s*$", re.MULTILINE)
EXTERNALIZED_PATCH_PATH_PATTERN = re.compile(r"^externalized_patch_path:\s*(.+)$", re.MULTILINE)
TOOL_HINT_LINE_PATTERNS = (SUMMARY_FOR_HISTORY_PATTERN, TOOL_REMINDER_PATTERN)


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    metadata: dict


def _metadata(
    tool_status,
    tool_error_code="",
    security_event_type="",
    risk_level="low",
    read_only=True,
    affected_paths=None,
    workspace_changed=False,
    workspace_fingerprint="",
    diff_summary=None,
    archive_summary="",
    tool_reminder="",
    read_window=None,
    freshness=None,
    background_task_id="",
    background_task_status="",
    background_task_return_code=None,
    background_task_next_offset=None,
    externalized_patch_path="",
    followup_tool="",
    followup_args=None,
    followup_reason="",
    followup_key="",
    chain_key="",
    completion_block_policy="",
    followup_is_blocking=False,
    blocks_completion=False,
):
    result = {
        "tool_status": tool_status,
        "tool_error_code": tool_error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level,
        "read_only": read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
        "followup_tool": str(followup_tool or ""),
        "followup_args": dict(followup_args or {}),
        "followup_reason": str(followup_reason or ""),
        "followup_key": str(followup_key or ""),
        "chain_key": str(chain_key or ""),
        "completion_block_policy": str(completion_block_policy or ""),
        "followup_is_blocking": bool(followup_is_blocking),
        "blocks_completion": bool(blocks_completion),
    }
    if workspace_fingerprint:
        result["workspace_fingerprint"] = workspace_fingerprint
    if archive_summary:
        result["archive_summary"] = str(archive_summary)
    if tool_reminder:
        result["tool_reminder"] = str(tool_reminder)
    if read_window:
        result["read_window"] = dict(read_window)
    if freshness:
        result["freshness"] = str(freshness)
    if background_task_id:
        result["background_task_id"] = str(background_task_id)
    if background_task_status:
        result["background_task_status"] = str(background_task_status)
    if background_task_return_code is not None:
        result["background_task_return_code"] = background_task_return_code
    if background_task_next_offset is not None:
        result["background_task_next_offset"] = int(background_task_next_offset)
    if externalized_patch_path:
        result["externalized_patch_path"] = str(externalized_patch_path)
    return result


def _extract_archive_summary(content):
    match = SUMMARY_FOR_HISTORY_PATTERN.search(_extract_tool_hint_region(content))
    if not match:
        return ""
    return clip(" ".join(match.group(1).split()), 500)


def _extract_tool_reminder(content):
    match = TOOL_REMINDER_PATTERN.search(_extract_tool_hint_region(content))
    if not match:
        return ""
    return clip(" ".join(match.group(1).split()), 500)


def _extract_tool_hint_region(content):
    text = str(content)
    lines = text.splitlines()
    suffix = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if suffix:
                continue
            continue
        if any(pattern.fullmatch(stripped) for pattern in TOOL_HINT_LINE_PATTERNS):
            suffix.append(stripped)
            continue
        break
    if not suffix:
        return ""
    suffix.reverse()
    return "\n".join(suffix)


def strip_tool_hints(content):
    text = str(content)
    lines = text.splitlines()
    end = len(lines)
    saw_hint = False
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped:
            if saw_hint:
                end -= 1
                continue
            break
        if any(pattern.fullmatch(stripped) for pattern in TOOL_HINT_LINE_PATTERNS):
            saw_hint = True
            end -= 1
            continue
        break
    return "\n".join(lines[:end]).rstrip()


def _extract_read_window(content):
    text = str(content)
    header = READ_FILE_HEADER_PATTERN.search(text)
    hint_region = _extract_tool_hint_region(text)
    next_offset = re.search(r'"offset":(\d+)', hint_region)
    result = {}
    if header:
        result.update(
            {
                "start_line": int(header.group(1)),
                "end_line": int(header.group(2)),
                "known_line_floor": int(header.group(3)),
                "returned_lines": int(header.group(4)),
                "requested_lines": int(header.group(5)),
            }
        )
    if next_offset:
        result["next_offset"] = int(next_offset.group(1))
    has_more = bool(TOOL_REMINDER_PATTERN.search(hint_region))
    if not has_more and header:
        end_line = int(header.group(2))
        known_line_floor = int(header.group(3))
        has_more = end_line < known_line_floor
    if not has_more and next_offset and header:
        requested_lines = int(header.group(5))
        has_more = int(next_offset.group(1)) > int(header.group(1)) and int(header.group(4)) >= requested_lines
    result["has_more"] = has_more
    return result


def _extract_background_task_id(content):
    match = BACKGROUND_TASK_ID_PATTERN.search(str(content))
    if not match:
        return ""
    return str(match.group(1)).strip()


def _extract_background_task_status(content):
    match = BACKGROUND_TASK_STATUS_PATTERN.search(str(content))
    if not match:
        return ""
    return str(match.group(1)).strip()


def _extract_background_task_return_code(content):
    match = BACKGROUND_TASK_RETURN_CODE_PATTERN.search(str(content))
    if not match:
        return None
    value = str(match.group(1)).strip()
    if value in {"", "(running)", "(none)"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _extract_background_task_next_offset(content):
    match = BACKGROUND_TASK_NEXT_OFFSET_PATTERN.search(str(content))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_externalized_patch_path(content):
    match = EXTERNALIZED_PATCH_PATH_PATTERN.search(str(content))
    if not match:
        return ""
    return str(match.group(1)).strip()


def _build_followup_metadata(
    name,
    args,
    *,
    content="",
    read_window=None,
    freshness=None,
    background_task_id="",
    background_task_status="",
    execution_context=None,
):
    args = args if isinstance(args, dict) else {}
    read_window = read_window if isinstance(read_window, dict) else {}
    execution_context = execution_context if isinstance(execution_context, dict) else {}
    active_obligation = execution_context.get("active_obligation", {}) if isinstance(execution_context.get("active_obligation", {}), dict) else {}
    active_required_tool = str(active_obligation.get("required_tool", "")).strip()
    active_chain_key = str(active_obligation.get("chain_key", "")).strip()
    active_block_policy = str(active_obligation.get("completion_block_policy", "")).strip()
    if name == "read_file" and read_window.get("has_more"):
        path = str(args.get("path", "")).strip()
        next_offset = int(read_window.get("next_offset", 0) or 0)
        requested_lines = int(read_window.get("requested_lines", args.get("limit", 0) or 0) or 0)
        if path and next_offset >= 1 and requested_lines >= 1 and freshness:
            is_blocking = active_required_tool == "read_file" and active_block_policy == "until_eof"
            return {
                "followup_tool": "read_file",
                "followup_args": {
                    "path": path,
                    "offset": next_offset,
                    "limit": requested_lines,
                },
                "followup_reason": "read_window_incomplete",
                "followup_key": f"read_file_continue:{path}:{freshness}",
                "chain_key": active_chain_key or f"read_file:{path}:{freshness}",
                "completion_block_policy": "until_eof" if is_blocking else "",
                "followup_is_blocking": is_blocking,
                "blocks_completion": is_blocking,
            }
    if name == "task_output" and str(background_task_status).strip() == "running":
        task_id = str(background_task_id or args.get("task_id", "")).strip()
        stream = str(args.get("stream", "stdout")).strip() or "stdout"
        limit = int(args.get("limit", 4000) or 4000)
        next_offset = _extract_background_task_next_offset(content)
        if task_id:
            return {
                "followup_tool": "task_output",
                "followup_args": {
                    "task_id": task_id,
                    "offset": int(next_offset if next_offset is not None else int(args.get("offset", 0) or 0)),
                    "stream": stream,
                    "limit": limit,
                },
                "followup_reason": "background_task_still_running",
                "followup_key": f"task_output_wait:{task_id}:{stream}",
                "chain_key": active_chain_key or f"task_output:{task_id}:{stream}",
                "completion_block_policy": "until_terminal",
                "followup_is_blocking": True,
                "blocks_completion": True,
            }
    if name == "git_diff":
        externalized_path = _extract_externalized_patch_path(content)
        if externalized_path:
            return {
                "followup_tool": "read_file",
                "followup_args": {
                    "path": externalized_path,
                    "offset": 1,
                },
                "followup_reason": "git_diff_externalized",
                "followup_key": f"git_diff_artifact:{externalized_path}",
                "chain_key": active_chain_key
                or f"git_diff:{str(args.get('path', '.')).strip()}:{str(args.get('mode', 'workspace')).strip() or 'workspace'}",
                "completion_block_policy": "until_opened",
                "followup_is_blocking": True,
                "blocks_completion": True,
            }
    return {
        "followup_tool": "",
        "followup_args": {},
        "followup_reason": "",
        "followup_key": "",
        "chain_key": active_chain_key,
        "completion_block_policy": "",
        "followup_is_blocking": False,
        "blocks_completion": False,
    }


class ToolExecutor:
    def __init__(self, agent):
        self.agent = agent
    """
    |-- 1. allowed_tools 白名单检查
    |-- 2. 工具是否存在检查
    |-- 3. 参数合法性检查
    |-- 4. 重复调用检查
    |-- 5. 高风险工具审批
    |-- 6. 执行前 workspace 快照
    |-- 7. 真正运行工具
    |-- 8. 执行后 workspace 快照
    |-- 9. 判断是否改动文件、是否 partial success
    diff_summary = [
        "created:new_file.py",
        "modified:pico/tools.py",
    ]
    |-- 10. 更新 memory / process note 工具执行成功后，会把少量信息写进工作记忆。
    |-- 11. 返回 ToolExecutionResult

    最近两次工具事件如果和当前调用完全一样，直接拒绝:

    """
    def execute(self, name, args, execution_context=None):
        agent = self.agent
        execution_context = execution_context if isinstance(execution_context, dict) else {}
        if agent.allowed_tools is not None and name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        tool = agent.tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                ),
            )

        try:
            agent.validate_tool(name, args)
        except Exception as exc:
            example = agent.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            return ToolExecutionResult(
                content=message,
                metadata=_metadata(
                    "rejected",
                    tool_error_code="invalid_arguments",
                    security_event_type=security_event_type,
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        if tool["risky"] and not agent.approve(name, args):
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block" if agent.read_only else "approval_denied",
                    risk_level="high",
                    read_only=False,
                ),
            )

        before_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            raw_content = tool["run"](args)
            content = str(raw_content) if name in {"read_file", "task_output", "git_status", "git_diff"} else clip(raw_content)
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", content)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            archive_summary = _extract_archive_summary(content)
            tool_reminder = _extract_tool_reminder(content)
            read_window = _extract_read_window(content) if name == "read_file" else {}
            freshness = memorylib.file_freshness(args.get("path", ""), agent.root) if name == "read_file" else None
            background_task_id = _extract_background_task_id(content) if name in {"run_shell_bg", "task_output", "task_stop"} else ""
            background_task_status = _extract_background_task_status(content) if name in {"run_shell_bg", "task_output", "task_stop"} else ""
            background_task_return_code = _extract_background_task_return_code(content) if name in {"task_output", "task_stop"} else None
            background_task_next_offset = _extract_background_task_next_offset(content) if name == "task_output" else None
            externalized_patch_path = _extract_externalized_patch_path(content) if name == "git_diff" else ""
            followup = _build_followup_metadata(
                name,
                args,
                content=content,
                read_window=read_window,
                freshness=freshness,
                background_task_id=background_task_id,
                background_task_status=background_task_status,
                execution_context=execution_context,
            )
            metadata = _metadata(
                tool_status,
                tool_error_code=tool_error_code,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                archive_summary=archive_summary,
                tool_reminder=tool_reminder,
                read_window=read_window,
                freshness=freshness,
                background_task_id=background_task_id,
                background_task_status=background_task_status,
                background_task_return_code=background_task_return_code,
                background_task_next_offset=background_task_next_offset,
                externalized_patch_path=externalized_patch_path,
                followup_tool=followup.get("followup_tool", ""),
                followup_args=followup.get("followup_args", {}),
                followup_reason=followup.get("followup_reason", ""),
                followup_key=followup.get("followup_key", ""),
                chain_key=followup.get("chain_key", ""),
                completion_block_policy=followup.get("completion_block_policy", ""),
                followup_is_blocking=followup.get("followup_is_blocking", False),
                blocks_completion=followup.get("blocks_completion", False),
            )
            agent.update_memory_after_tool(name, args, content, metadata=metadata)
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            metadata = _metadata(
                "partial_success" if workspace_changed else "error",
                tool_error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)
