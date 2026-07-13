"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from functools import partial
from pathlib import Path

from . import git_tools as gitlib
from . import skills as skilllib
from .workspace import IGNORED_PATH_NAMES
from .tool_output import ShellOutputCapture

READ_FILE_MAX_LIMIT = 2000
READ_FILE_DEFAULT_LIMIT = 200
READ_FILE_DEFAULT_UNSPECIFIED_LIMIT = READ_FILE_MAX_LIMIT
READ_FILE_ARCHIVE_SUMMARY_LINES = 12
GLOB_MAX_RESULTS = 200
GREP_DEFAULT_HEAD_LIMIT = 200
GREP_MAX_HEAD_LIMIT = 2000
GREP_MAX_CONTEXT_LINES = 200
GREP_OUTPUT_MODES = {"content", "files", "count"}
GREP_TIMEOUT_SECONDS = 20
GREP_MAX_TIMEOUT_SECONDS = 120
RUN_SHELL_DEFAULT_TIMEOUT = 20
RUN_SHELL_MAX_TIMEOUT = 300
RUN_SHELL_BG_DEFAULT_TIMEOUT = 3600
RUN_SHELL_BG_MAX_TIMEOUT = 86400
TASK_OUTPUT_DEFAULT_LIMIT = 4000
TASK_OUTPUT_MAX_LIMIT = 20000
TASK_OUTPUT_STREAMS = {"stdout", "stderr", "both"}
TASK_LIST_DEFAULT_LIMIT = 20
TASK_LIST_MAX_LIMIT = 100
TASK_LIST_STATUSES = {"all", "running", "exited", "failed", "stopped"}
TODO_WRITE_MAX_ITEMS = 50
DELEGATE_DEFAULT_MAX_STEPS = 3
DELEGATE_MAX_STEPS = 12
TODO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
TODO_STATUSES = {"pending", "active", "done", "blocked"}
BASH_HEREDOC_PATTERN = re.compile(r"(?:^|\s)<<-?\s*(?:['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?)")
GIT_STATUS_DEFAULT_LIMIT = gitlib.GIT_STATUS_DEFAULT_LIMIT
GIT_STATUS_MAX_LIMIT = gitlib.GIT_STATUS_MAX_LIMIT
GIT_DIFF_DEFAULT_LIMIT = gitlib.GIT_DIFF_DEFAULT_LIMIT
GIT_DIFF_MAX_LIMIT = gitlib.GIT_DIFF_MAX_LIMIT
GIT_DIFF_MODES = gitlib.GIT_DIFF_MODES


def _nullable_integer(*, minimum=None, maximum=None, description=""):
    definition = {"type": ["integer", "null"]}
    if minimum is not None:
        definition["minimum"] = int(minimum)
    if maximum is not None:
        definition["maximum"] = int(maximum)
    if description:
        definition["description"] = str(description)
    return definition


class UnsupportedShellSyntaxError(ValueError):
    tool_error_code = "unsupported_shell_syntax"


def _uses_windows_shell():
    return os.name == "nt"


def validate_shell_command_syntax(command):
    text = str(command or "")
    if _uses_windows_shell() and BASH_HEREDOC_PATTERN.search(text):
        raise UnsupportedShellSyntaxError(
            "Bash here-documents such as <<'PY' or <<EOF are unsupported by the Windows shell. "
            "Use python -c only for compact code, or write a script file and run it."
        )

BASE_TOOL_SPECS = {
    "list_files": {
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": ["string", "null"], "description": "Workspace directory."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "concurrency_safe": True,
        "risky": False,
        "description": "List files in the workspace.",
    },
    "glob": {
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": ["string", "null"]},
            },
            "required": ["pattern", "path"],
            "additionalProperties": False,
        },
        "concurrency_safe": True,
        "risky": False,
        "description": "Find paths by glob pattern. Use it to discover candidate files; narrow pattern or path if results are truncated.",
    },
    "read_file": {
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": _nullable_integer(minimum=1, description="One-based line offset."),
                "limit": _nullable_integer(minimum=1, maximum=READ_FILE_MAX_LIMIT, description="Maximum lines to return."),
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": False,
        },
        "concurrency_safe": True,
        "risky": False,
        "description": (
            "Read a UTF-8 file by line range. Use it for known files, raw artifacts/logs, or context after search. "
            "When a tool result gives an externalized output path, read that artifact before rerunning the tool. "
            "Omit limit to read as much as fits; use offset/limit for targeted or next_offset reads."
        ),
    },
    "grep": {
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"}, "path": {"type": ["string", "null"]},
                "output_mode": {"type": ["string", "null"], "enum": ["content", "files", "count", None]},
                "head_limit": _nullable_integer(minimum=1, maximum=GREP_MAX_HEAD_LIMIT, description="Maximum matching entries."),
                "offset": _nullable_integer(minimum=0, description="Zero-based result offset."),
                "glob": {"type": ["string", "null"]},
                "-A": _nullable_integer(minimum=0, maximum=GREP_MAX_CONTEXT_LINES, description="Context lines after each match."),
                "-B": _nullable_integer(minimum=0, maximum=GREP_MAX_CONTEXT_LINES, description="Context lines before each match."),
                "-C": _nullable_integer(minimum=0, maximum=GREP_MAX_CONTEXT_LINES, description="Context lines before and after each match."),
                "timeout": _nullable_integer(minimum=1, maximum=GREP_MAX_TIMEOUT_SECONDS, description="Search timeout in seconds."),
            },
            "required": ["pattern", "path", "output_mode", "head_limit", "offset", "glob", "-A", "-B", "-C", "timeout"],
            "additionalProperties": False,
        },
        "concurrency_safe": True,
        "risky": False,
        "description": (
            "Search file contents by a reliable pattern such as a symbol, config key, error, path fragment, or exact phrase. "
            "Prefer content mode with a small -C window before read_file; use files/count for discovery or totals. "
            "Do not guess a pattern: use glob, list_files, or read_file when it is unclear."
        ),
    },
    "git_status": {
        "parameters": {
            "type": "object", "properties": {
                "path": {"type": ["string", "null"]},
                "offset": _nullable_integer(minimum=0, description="Zero-based result offset."),
                "limit": _nullable_integer(minimum=1, maximum=GIT_STATUS_MAX_LIMIT, description="Maximum changed paths to return."),
            }, "required": ["path", "offset", "limit"], "additionalProperties": False,
        },
        "concurrency_safe": True,
        "risky": False,
        "description": (
            "Show changed-file scope in the current Git working tree, including staged, unstaged, untracked, and deleted paths. "
            "Use after edits before git_diff to confirm the expected file set."
        ),
    },
    "git_diff": {
        "parameters": {
            "type": "object", "properties": {
                "path": {"type": ["string", "null"]},
                "mode": {"type": ["string", "null"], "enum": ["workspace", "staged", "unstaged", None]},
                "offset": _nullable_integer(minimum=1, description="One-based patch line offset."),
                "limit": _nullable_integer(minimum=1, maximum=GIT_DIFF_MAX_LIMIT, description="Maximum patch lines to return."),
            }, "required": ["path", "mode", "offset", "limit"], "additionalProperties": False,
        },
        "concurrency_safe": False,
        "risky": False,
        "description": (
            "Show the Git patch for the workspace or a path. Use it after edits or when the user asks for changes or a patch. "
            "Large patches may be externalized; read the returned artifact path for raw patch text."
        ),
    },
    "use_skill": {
        "parameters": {
            "type": "object", "properties": {
                "name": {"type": "string"}, "args": {"type": ["string", "null"]},
            }, "required": ["name", "args"], "additionalProperties": False,
        },
        "concurrency_safe": False,
        "risky": False,
        "description": (
            "Load a routed reusable workflow from .lumo/skills/<category>/<name>/SKILL.md. Pass the qualified "
            "category/name shown in Skills. Use when an available skill matches the task; "
            "its full instructions apply only to the current task after this tool result."
        ),
    },
    "todo_write": {
        "parameters": {
            "type": "object", "properties": {
                "todos": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "active", "done", "blocked"]},
                }, "required": ["id", "text", "status"], "additionalProperties": False}, "maxItems": TODO_WRITE_MAX_ITEMS},
            }, "required": ["todos"], "additionalProperties": False,
        },
        "concurrency_safe": False,
        "risky": False,
        "description": (
            "Create or replace the task todo list for genuinely multi-step work. Each item needs id, text, and "
            "status=pending|active|done|blocked; use one active item at most. Do not use for simple questions or one-step work."
        ),
    },
    "run_shell": {
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": _nullable_integer(minimum=1, maximum=RUN_SHELL_MAX_TIMEOUT, description="Foreground timeout in seconds.")}, "required": ["command", "timeout"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": True,
        "description": (
            "Run a short foreground command in the repo root and use its result immediately. "
            f"Use for quick inspection, one-shot scripts, tests, or lint; timeout is 1-{RUN_SHELL_MAX_TIMEOUT} seconds. "
            "Use run_shell_bg for longer work. On Windows this is not a Bash shell: do not use <<'PY' or <<EOF. "
            "Bare Python commands use .lumo/python-env; "
            "after one failure runtime may perform one restricted environment repair and retry."
        ),
    },
    "run_shell_bg": {
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": _nullable_integer(minimum=1, maximum=RUN_SHELL_BG_MAX_TIMEOUT, description="Background timeout in seconds.")}, "required": ["command", "timeout"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": True,
        "description": (
            f"Start a long-running command and return a task_id; timeout is 1-{RUN_SHELL_BG_MAX_TIMEOUT} seconds. "
            "Use for builds, full tests, servers, or benchmarks. On Windows this is not a Bash shell: do not use <<'PY' or <<EOF; "
            "inspect it with task_output and stop it with task_stop. Bare Python commands use .lumo/python-env; "
            "a failed task may be repaired and restarted once when task_output observes the failure."
        ),
    },
    "task_output": {
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}, "offset": _nullable_integer(minimum=0, description="Zero-based output offset."), "limit": _nullable_integer(minimum=1, maximum=TASK_OUTPUT_MAX_LIMIT, description="Maximum output characters to return."), "stream": {"type": ["string", "null"], "enum": ["stdout", "stderr", "both", None]}}, "required": ["task_id", "offset", "limit", "stream"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": False,
        "description": (
            "Read paginated stdout or stderr from a run_shell_bg task. Use next_offset to continue without replaying output."
        ),
    },
    "task_list": {
        "parameters": {"type": "object", "properties": {"offset": _nullable_integer(minimum=0, description="Zero-based task offset."), "limit": _nullable_integer(minimum=1, maximum=TASK_LIST_MAX_LIMIT, description="Maximum tasks to return."), "status": {"type": ["string", "null"], "enum": ["all", "running", "exited", "failed", "stopped", None]}}, "required": ["offset", "limit", "status"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": False,
        "description": (
            "List current-run background tasks. Use it to discover task_id values for task_output or task_stop."
        ),
    },
    "task_stop": {
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": True,
        "description": "Stop a background task previously started with run_shell_bg.",
    },
    "write_file": {
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": True,
        "description": (
            "Create a text file or intentionally replace an entire existing file. Use patch_file for small changes."
        ),
    },
    "patch_file": {
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "replace_all": {"type": ["boolean", "null"]}}, "required": ["path", "old_text", "new_text", "replace_all"], "additionalProperties": False},
        "concurrency_safe": False,
        "risky": True,
        "description": (
            "Replace exact text for a small edit in an existing file. old_text must be unique unless replace_all=true; "
            "otherwise add context to make the target unique."
        ),
    },
}

DELEGATE_TOOL_SPEC = {
    "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "max_steps": _nullable_integer(minimum=1, maximum=DELEGATE_MAX_STEPS, description="Maximum child-agent steps."), "inherit_context": {"type": ["boolean", "null"]}}, "required": ["task", "max_steps", "inherit_context"], "additionalProperties": False},
    "concurrency_safe": False,
    "risky": False,
    "description": (
        "Ask a bounded child agent to handle a subtask. Set inherit_context=true when it needs parent context; "
        "false for independent work."
    ),
}


def legal_tool_names():
    return set(BASE_TOOL_SPECS) | {"delegate"}


def normalize_tool_arguments(name, args):
    """Clamp only valid integer arguments that exceed a declared tool maximum.

    Lower-bound violations and incorrect JSON types remain validation errors: silently
    changing those values could turn a malformed model call into a different action.
    """
    if not isinstance(args, dict):
        return args, []
    spec = BASE_TOOL_SPECS.get(name, DELEGATE_TOOL_SPEC if name == "delegate" else {})
    schema = spec.get("parameters", {}) if isinstance(spec, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    normalized = {key: value for key, value in args.items() if value is not None}
    changes = []
    for key, definition in properties.items():
        if key not in normalized or not isinstance(definition, dict):
            continue
        value = normalized[key]
        maximum = definition.get("maximum")
        if (
            maximum is not None
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > int(maximum)
        ):
            effective = int(maximum)
            normalized[key] = effective
            changes.append(
                {
                    "argument": key,
                    "requested": value,
                    "effective": effective,
                    "reason": "above_maximum",
                }
            )
    return normalized, changes


def _read_window_request(args):
    """Return one-based offset, effective line limit, and whether limit/end was explicit."""
    args = args or {}
    offset = int(args.get("offset", args.get("start", 1)))
    limit_explicit = "limit" in args or "end" in args
    if "limit" in args:
        limit = int(args.get("limit", READ_FILE_DEFAULT_LIMIT))
    elif "end" in args:
        end = int(args.get("end", offset + READ_FILE_DEFAULT_LIMIT - 1))
        if end < offset:
            raise ValueError("invalid line range")
        limit = end - offset + 1
    else:
        limit = READ_FILE_DEFAULT_UNSPECIFIED_LIMIT
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > READ_FILE_MAX_LIMIT:
        raise ValueError(f"limit must be <= {READ_FILE_MAX_LIMIT}")
    return offset, limit, limit_explicit


def _read_window_args(args):
    """Return one-based offset and effective line limit, accepting legacy start/end."""
    offset, limit, _limit_explicit = _read_window_request(args)
    return offset, limit


def _glob_search_root(context, args):
    return context.path(args.get("path", "."))


def _grep_search_target(context, args):
    return context.path(args.get("path", "."))


def _glob_match(pattern, relative_path):
    normalized_pattern = str(pattern or "").replace("\\", "/")
    normalized_path = str(relative_path or "").replace("\\", "/")
    pattern_segments = [segment for segment in normalized_pattern.split("/") if segment]
    path_segments = [segment for segment in normalized_path.split("/") if segment]

    if not pattern_segments:
        return False

    memo = {}

    def match(pattern_index, path_index):
        key = (pattern_index, path_index)
        if key in memo:
            return memo[key]

        if pattern_index >= len(pattern_segments):
            result = path_index >= len(path_segments)
            memo[key] = result
            return result

        segment = pattern_segments[pattern_index]
        if segment == "**":
            result = match(pattern_index + 1, path_index)
            if not result and path_index < len(path_segments):
                result = match(pattern_index, path_index + 1)
            memo[key] = result
            return result

        if path_index >= len(path_segments):
            memo[key] = False
            return False

        result = re.match(fnmatch.translate(segment), path_segments[path_index]) is not None and match(
            pattern_index + 1,
            path_index + 1,
        )
        memo[key] = result
        return result

    return match(0, 0)


def _iter_glob_matches(context_root, search_root, pattern):
    matched = []
    stack = [search_root]

    while stack:
        current = stack.pop()
        try:
            entries = sorted(
                [entry for entry in current.iterdir() if entry.name not in IGNORED_PATH_NAMES],
                key=lambda item: (item.is_file(), item.name.lower()),
            )
        except (OSError, PermissionError):
            continue

        child_dirs = []
        for entry in entries:
            if entry.is_dir():
                if entry.is_symlink():
                    continue
                child_dirs.append(entry)
                continue
            if not entry.is_file():
                continue
            relative_to_search_root = entry.relative_to(search_root).as_posix()
            if _glob_match(pattern, relative_to_search_root):
                matched.append(entry.relative_to(context_root).as_posix())

        stack.extend(reversed(child_dirs))

    matched.sort(key=str.lower)
    return matched


def _grep_output_mode(args):
    mode = str(args.get("output_mode", "content")).strip().lower() or "content"
    if mode not in GREP_OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of: {', '.join(sorted(GREP_OUTPUT_MODES))}")
    return mode


def _grep_head_limit(args):
    value = int(args.get("head_limit", GREP_DEFAULT_HEAD_LIMIT))
    if value < 1:
        raise ValueError("head_limit must be >= 1")
    if value > GREP_MAX_HEAD_LIMIT:
        raise ValueError(f"head_limit must be <= {GREP_MAX_HEAD_LIMIT}")
    return value


def _grep_offset(args):
    value = int(args.get("offset", 0))
    if value < 0:
        raise ValueError("offset must be >= 0")
    return value


def _grep_timeout_seconds(args):
    value = int(args.get("timeout", GREP_TIMEOUT_SECONDS))
    if value < 1:
        raise ValueError("timeout must be >= 1")
    if value > GREP_MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be <= {GREP_MAX_TIMEOUT_SECONDS}")
    return value


def _run_shell_bg_timeout(args):
    value = int(args.get("timeout", RUN_SHELL_BG_DEFAULT_TIMEOUT))
    if value < 1:
        raise ValueError("timeout must be >= 1")
    if value > RUN_SHELL_BG_MAX_TIMEOUT:
        raise ValueError(f"timeout must be <= {RUN_SHELL_BG_MAX_TIMEOUT}")
    return value


def _task_output_limit(args):
    value = int(args.get("limit", TASK_OUTPUT_DEFAULT_LIMIT))
    if value < 1:
        raise ValueError("limit must be >= 1")
    if value > TASK_OUTPUT_MAX_LIMIT:
        raise ValueError(f"limit must be <= {TASK_OUTPUT_MAX_LIMIT}")
    return value


def _task_output_stream(args):
    value = str(args.get("stream", "stdout")).strip().lower() or "stdout"
    if value not in TASK_OUTPUT_STREAMS:
        raise ValueError(f"stream must be one of: {', '.join(sorted(TASK_OUTPUT_STREAMS))}")
    return value


def _task_list_limit(args):
    value = int(args.get("limit", TASK_LIST_DEFAULT_LIMIT))
    if value < 1:
        raise ValueError("limit must be >= 1")
    if value > TASK_LIST_MAX_LIMIT:
        raise ValueError(f"limit must be <= {TASK_LIST_MAX_LIMIT}")
    return value


def _task_list_status(args):
    value = str(args.get("status", "all")).strip().lower() or "all"
    if value not in TASK_LIST_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(TASK_LIST_STATUSES))}")
    return value


def _tool_bool_arg(args, key, default=False):
    value = args.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{key} must be a boolean")


def _grep_optional_nonnegative_int(args, key):
    value = args.get(key, None)
    if value in (None, ""):
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    if value > GREP_MAX_CONTEXT_LINES:
        raise ValueError(f"{key} must be <= {GREP_MAX_CONTEXT_LINES}")
    return value


def _display_search_path(search_path, root):
    if search_path == root:
        return "."
    return search_path.relative_to(root).as_posix()


def _grep_target_argument(search_path, root):
    if search_path == root:
        return "."
    return search_path.relative_to(root).as_posix()


def _is_ignored_candidate(path, root):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in IGNORED_PATH_NAMES for part in parts)


def _iter_grep_candidate_files(root, search_path, glob_pattern=None):
    if search_path.is_file():
        if _is_ignored_candidate(search_path, root):
            return []
        if glob_pattern and not _glob_match(glob_pattern, search_path.name):
            return []
        return [search_path]

    matched = []
    stack = [search_path]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(
                [entry for entry in current.iterdir() if entry.name not in IGNORED_PATH_NAMES],
                key=lambda item: (item.is_file(), item.name.lower()),
            )
        except (OSError, PermissionError):
            continue

        child_dirs = []
        for entry in entries:
            if entry.is_dir():
                if entry.is_symlink():
                    continue
                child_dirs.append(entry)
                continue
            if not entry.is_file():
                continue
            relative_to_search = entry.relative_to(search_path).as_posix()
            if glob_pattern and not _glob_match(glob_pattern, relative_to_search):
                continue
            matched.append(entry)
        stack.extend(reversed(child_dirs))
    matched.sort(key=lambda item: item.relative_to(root).as_posix().lower())
    return matched


def _page_entries(entries, limit, offset):
    total = len(entries)
    shown = entries[offset:offset + limit]
    truncated = offset + limit < total
    return shown, total, truncated


def _normalize_rg_relative_path(value):
    return str(value).replace("\\", "/")


def _parse_rg_count_lines(lines, search_path=None, root=None):
    entries = []
    total_matches = 0
    for line in lines:
        if not str(line).strip():
            continue
        path_text, _, count_text = str(line).rpartition(":")
        if not path_text or not count_text.strip():
            try:
                count = int(str(line).strip())
            except ValueError:
                continue
            if search_path is None or root is None or not search_path.is_file():
                continue
            path_text = search_path.relative_to(root).as_posix()
        else:
            try:
                count = int(count_text.strip())
            except ValueError:
                continue
            path_text = _normalize_rg_relative_path(path_text.strip())
        entries.append((path_text, count))
        total_matches += count
    return entries, total_matches


def _build_grep_summary(display_path, pattern, output_mode, shown_count, offset, total_matches, file_count, glob_pattern=None):
    scope = f"{display_path} with glob {glob_pattern}" if glob_pattern else display_path
    if output_mode == "files":
        return (
            f"Grepped {scope} for pattern {pattern} in files mode; "
            f"matched {file_count} files; showing {shown_count} from offset {offset}."
        )
    if output_mode == "count":
        return (
            f"Grepped {scope} for pattern {pattern} in count mode; "
            f"matched {total_matches} lines across {file_count} files; showing {shown_count} counts from offset {offset}."
        )
    return (
        f"Grepped {scope} for pattern {pattern} in content mode; "
        f"matched {total_matches} lines across {file_count} files; showing {shown_count} from offset {offset}."
    )


def _build_grep_header(pattern, display_path, output_mode, total_matches, file_count, shown_count, offset):
    header = [
        f"# grep pattern: {pattern}",
        f"# search path: {display_path}",
        f"# output mode: {output_mode}",
    ]
    if output_mode == "files":
        header.append(f"# matched {file_count} files (showing {shown_count} from offset {offset})")
    elif output_mode == "count":
        header.append(
            f"# matched {total_matches} lines across {file_count} files (showing {shown_count} counts from offset {offset})"
        )
    else:
        header.append(
            f"# matched {total_matches} lines across {file_count} files (showing {shown_count} lines from offset {offset})"
        )
    return header


def _grep_result_text(pattern, display_path, output_mode, body_lines, shown_count, offset, total_matches, file_count, truncated, glob_pattern=None):
    header = _build_grep_header(pattern, display_path, output_mode, total_matches, file_count, shown_count, offset)
    body = body_lines or ["(no matches)"]
    hints = []
    if truncated:
        hints.append(
            f"<tool_reminder>This grep result is only a partial page, not the full search result. "
            f"Showing {shown_count} visible entries from offset {offset}. "
            f"If you need more matches, continue with grep using a larger offset such as {offset + shown_count}, "
            f"or narrow the pattern, path, or glob before retrying.</tool_reminder>"
        )
    summary = _build_grep_summary(display_path, pattern, output_mode, shown_count, offset, total_matches, file_count, glob_pattern=glob_pattern)
    hints.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join([*header, *body, *hints])


def _raise_grep_timeout(pattern, display_path, timeout_seconds):
    raise RuntimeError(
        "grep timed out before completing the search. "
        "This does not mean there were no matches. "
        f"Pattern: {pattern!r}. "
        f"Path: {display_path}. "
        f"Timeout: {timeout_seconds}s. "
        "Narrow the pattern or path, increase timeout, or retry with paging after a narrower search."
    )


def _grep_deadline(timeout_seconds):
    return time.monotonic() + float(timeout_seconds)


def _check_grep_deadline(deadline, pattern, display_path, timeout_seconds):
    if time.monotonic() >= float(deadline):
        _raise_grep_timeout(pattern, display_path, timeout_seconds)


def _git_status_offset(args):
    offset = int(args.get("offset", 0))
    if offset < 0:
        raise ValueError("offset must be >= 0")
    return offset


def _git_status_limit(args):
    limit = int(args.get("limit", GIT_STATUS_DEFAULT_LIMIT))
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > GIT_STATUS_MAX_LIMIT:
        raise ValueError(f"limit must be <= {GIT_STATUS_MAX_LIMIT}")
    return limit


def _git_diff_mode(args):
    mode = str(args.get("mode", "workspace")).strip() or "workspace"
    if mode not in GIT_DIFF_MODES:
        raise ValueError(f"mode must be one of {', '.join(sorted(GIT_DIFF_MODES))}")
    return mode


def _git_diff_offset(args):
    offset = int(args.get("offset", 1))
    if offset < 1:
        raise ValueError("offset must be >= 1")
    return offset


def _git_diff_limit(args):
    limit = int(args.get("limit", GIT_DIFF_DEFAULT_LIMIT))
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > GIT_DIFF_MAX_LIMIT:
        raise ValueError(f"limit must be <= {GIT_DIFF_MAX_LIMIT}")
    return limit


def _run_rg_command(context, args):
    search_path = _grep_search_target(context, args)
    pattern = str(args.get("pattern", "")).strip()
    output_mode = _grep_output_mode(args)
    head_limit = _grep_head_limit(args)
    offset = _grep_offset(args)
    timeout_seconds = _grep_timeout_seconds(args)
    glob_pattern = str(args.get("glob", "")).strip() or None
    context_after = _grep_optional_nonnegative_int(args, "-A")
    context_before = _grep_optional_nonnegative_int(args, "-B")
    context_around = _grep_optional_nonnegative_int(args, "-C")

    if search_path.is_file() and glob_pattern and not _glob_match(glob_pattern, search_path.name):
        display_path = _display_search_path(search_path, context.root)
        return _grep_result_text(
            pattern,
            display_path,
            output_mode,
            [],
            0,
            offset,
            0,
            0,
            False,
            glob_pattern=glob_pattern,
        )

    common_args = ["rg", "--smart-case", "--hidden", "--no-heading"]
    for ignored in sorted(IGNORED_PATH_NAMES):
        common_args.extend(["--glob", f"!{ignored}/**"])
        common_args.extend(["--glob", f"!**/{ignored}/**"])
    if glob_pattern:
        common_args.extend(["--glob", glob_pattern])
    target_arg = _grep_target_argument(search_path, context.root)
    display_path = _display_search_path(search_path, context.root)

    count_result = subprocess.run(
        [*common_args, "-c", pattern, target_arg],
        cwd=context.root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if count_result.returncode not in (0, 1):
        error_text = count_result.stderr.strip() or count_result.stdout.strip() or "rg failed"
        raise RuntimeError(error_text)
    count_lines = [line for line in count_result.stdout.splitlines() if line.strip()]
    count_entries, total_matches = _parse_rg_count_lines(count_lines, search_path=search_path, root=context.root)
    file_count = len(count_entries)

    if output_mode == "files":
        file_result = subprocess.run(
            [*common_args, "-l", pattern, target_arg],
            cwd=context.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if file_result.returncode not in (0, 1):
            error_text = file_result.stderr.strip() or file_result.stdout.strip() or "rg failed"
            raise RuntimeError(error_text)
        entries = [_normalize_rg_relative_path(line.strip()) for line in file_result.stdout.splitlines() if line.strip()]
        shown, _, truncated = _page_entries(entries, head_limit, offset)
        return _grep_result_text(
            pattern,
            display_path,
            output_mode,
            shown,
            len(shown),
            offset,
            total_matches=file_count,
            file_count=file_count,
            truncated=truncated,
            glob_pattern=glob_pattern,
        )

    if output_mode == "count":
        entries = [f"{path}:{count}" for path, count in count_entries]
        shown, _, truncated = _page_entries(entries, head_limit, offset)
        return _grep_result_text(
            pattern,
            display_path,
            output_mode,
            shown,
            len(shown),
            offset,
            total_matches=total_matches,
            file_count=file_count,
            truncated=truncated,
            glob_pattern=glob_pattern,
        )

    content_args = [*common_args, "-n", "-H"]
    if context_around is not None:
        content_args.extend(["-C", str(context_around)])
    else:
        if context_before is not None:
            content_args.extend(["-B", str(context_before)])
        if context_after is not None:
            content_args.extend(["-A", str(context_after)])
    content_result = subprocess.run(
        [*content_args, pattern, target_arg],
        cwd=context.root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if content_result.returncode not in (0, 1):
        error_text = content_result.stderr.strip() or content_result.stdout.strip() or "rg failed"
        raise RuntimeError(error_text)
    entries = [_normalize_rg_relative_path(line.rstrip()) for line in content_result.stdout.splitlines() if line.rstrip()]
    shown, _, truncated = _page_entries(entries, head_limit, offset)
    return _grep_result_text(
        pattern,
        display_path,
        output_mode,
        shown,
        len(shown),
        offset,
        total_matches=total_matches,
        file_count=file_count,
        truncated=truncated,
        glob_pattern=glob_pattern,
    )


def _grep_fallback_content_lines(relative_path, lines, matched_indexes, before, after):
    included_indexes = set()
    for index in matched_indexes:
        start = max(0, index - before)
        end = min(len(lines) - 1, index + after)
        included_indexes.update(range(start, end + 1))
    rendered = []
    for index in sorted(included_indexes):
        rendered.append(f"{relative_path}:{index + 1}:{lines[index]}")
    return rendered


def _compile_grep_fallback_pattern(pattern):
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise RuntimeError(f"invalid grep pattern for fallback search: {exc}") from exc


def _run_grep_fallback(context, args):
    search_path = _grep_search_target(context, args)
    pattern = str(args.get("pattern", "")).strip()
    output_mode = _grep_output_mode(args)
    head_limit = _grep_head_limit(args)
    offset = _grep_offset(args)
    timeout_seconds = _grep_timeout_seconds(args)
    glob_pattern = str(args.get("glob", "")).strip() or None
    context_after = _grep_optional_nonnegative_int(args, "-A") or 0
    context_before = _grep_optional_nonnegative_int(args, "-B") or 0
    context_around = _grep_optional_nonnegative_int(args, "-C")
    if context_around is not None:
        context_before = context_around
        context_after = context_around

    display_path = _display_search_path(search_path, context.root)
    deadline = _grep_deadline(timeout_seconds)
    files = _iter_grep_candidate_files(context.root, search_path, glob_pattern=glob_pattern)
    compiled_pattern = _compile_grep_fallback_pattern(pattern)
    file_matches = []
    total_matches = 0
    content_entries = []
    count_entries = []

    for file_path in files:
        _check_grep_deadline(deadline, pattern, display_path, timeout_seconds)
        lines = []
        matched_indexes = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw_line in enumerate(handle):
                _check_grep_deadline(deadline, pattern, display_path, timeout_seconds)
                line = raw_line.rstrip("\n").rstrip("\r")
                lines.append(line)
                if compiled_pattern.search(line):
                    matched_indexes.append(index)
        if not matched_indexes:
            continue
        relative_path = file_path.relative_to(context.root).as_posix()
        file_matches.append(relative_path)
        total_matches += len(matched_indexes)
        count_entries.append(f"{relative_path}:{len(matched_indexes)}")
        if output_mode == "content":
            content_entries.extend(
                _grep_fallback_content_lines(relative_path, lines, matched_indexes, context_before, context_after)
            )

    if output_mode == "files":
        shown, _, truncated = _page_entries(file_matches, head_limit, offset)
        return _grep_result_text(
            pattern,
            display_path,
            output_mode,
            shown,
            len(shown),
            offset,
            total_matches=len(file_matches),
            file_count=len(file_matches),
            truncated=truncated,
            glob_pattern=glob_pattern,
        )

    if output_mode == "count":
        shown, _, truncated = _page_entries(count_entries, head_limit, offset)
        return _grep_result_text(
            pattern,
            display_path,
            output_mode,
            shown,
            len(shown),
            offset,
            total_matches=total_matches,
            file_count=len(file_matches),
            truncated=truncated,
            glob_pattern=glob_pattern,
        )

    shown, _, truncated = _page_entries(content_entries, head_limit, offset)
    return _grep_result_text(
        pattern,
        display_path,
        output_mode,
        shown,
        len(shown),
        offset,
        total_matches=total_matches,
        file_count=len(file_matches),
        truncated=truncated,
        glob_pattern=glob_pattern,
    )


def _summarize_read_window(relative_path, offset, selected_lines, has_more, next_offset):
    facts = []
    for line in selected_lines:
        text = line.strip()
        if not text:
            continue
        facts.append(text)
        if len(facts) >= READ_FILE_ARCHIVE_SUMMARY_LINES:
            break
    if facts:
        summary = " | ".join(facts)
    else:
        summary = "(empty window)"
    suffix = f" If needed, continue from line {next_offset}." if has_more else ""
    return f"Read {relative_path} from line {offset}: {summary}.{suffix}"


def _read_file_window(path, offset, limit):
    end_line = offset + limit - 1
    selected_lines = []
    returned_lines = 0
    last_seen_line = 0
    has_more = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, raw_line in enumerate(handle, start=1):
            last_seen_line = number
            if number < offset:
                continue
            if number > end_line:
                has_more = True
                break

            line = raw_line.rstrip("\n").rstrip("\r")
            rendered = f"{number:>4}: {line}"

            selected_lines.append(rendered)
            returned_lines += 1

    if selected_lines:
        next_offset = offset + returned_lines
    else:
        next_offset = offset
    if not has_more and last_seen_line >= end_line:
        next_offset = end_line + 1

    return {
        "lines": selected_lines,
        "returned_lines": returned_lines,
        "end_line": offset + max(returned_lines - 1, 0),
        "has_more": has_more,
        "next_offset": next_offset,
        "last_seen_line": last_seen_line,
    }


def build_tool_registry(context):


    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }


    if context.depth < context.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, context)}
    return tools


def _schema_types(definition):
    expected = definition.get("type") if isinstance(definition, dict) else None
    return expected if isinstance(expected, list) else [expected]


def _validate_native_schema_value(value, definition, label, *, allow_omitted_nullable=True):
    allowed_types = _schema_types(definition)
    if value is None and "null" in allowed_types:
        return
    valid = (
        ("string" in allowed_types and isinstance(value, str))
        or ("integer" in allowed_types and isinstance(value, int) and not isinstance(value, bool))
        or ("boolean" in allowed_types and isinstance(value, bool))
        or ("array" in allowed_types and isinstance(value, list))
        or ("object" in allowed_types and isinstance(value, dict))
    )
    if not valid:
        raise ValueError(f"argument '{label}' has an invalid JSON type")
    if isinstance(value, int) and not isinstance(value, bool):
        minimum = definition.get("minimum") if isinstance(definition, dict) else None
        maximum = definition.get("maximum") if isinstance(definition, dict) else None
        if minimum is not None and value < int(minimum):
            raise ValueError(f"argument '{label}' must be >= {int(minimum)}")
        if maximum is not None and value > int(maximum):
            raise ValueError(f"argument '{label}' must be <= {int(maximum)}")
    if isinstance(value, list):
        item_definition = definition.get("items", {}) if isinstance(definition, dict) else {}
        max_items = definition.get("maxItems") if isinstance(definition, dict) else None
        if max_items is not None and len(value) > int(max_items):
            raise ValueError(f"argument '{label}' must contain at most {int(max_items)} items")
        for index, item in enumerate(value):
            _validate_native_schema_value(item, item_definition, f"{label}[{index}]", allow_omitted_nullable=False)
    elif isinstance(value, dict):
        _validate_native_schema_object(value, definition, label, allow_omitted_nullable=allow_omitted_nullable)


def _validate_native_schema_object(args, schema, label="arguments", *, allow_omitted_nullable=True):
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(args, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(args) - set(properties))
    if unknown:
        if label == "arguments":
            raise ValueError(f"unknown argument(s): {', '.join(unknown)}")
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")
    required = set(schema.get("required", []) or [])
    missing = sorted(
        key
        for key in required
        if key not in args
        and not (allow_omitted_nullable and "null" in _schema_types(properties.get(key, {})))
    )
    if missing:
        raise ValueError(f"missing required {label} field(s): {', '.join(missing)}")
    for key, value in args.items():
        _validate_native_schema_value(
            value,
            properties.get(key, {}),
            f"{label}.{key}" if label != "arguments" else key,
            allow_omitted_nullable=allow_omitted_nullable,
        )


def _validate_native_schema(name, args):
    spec = BASE_TOOL_SPECS.get(name, DELEGATE_TOOL_SPEC if name == "delegate" else {})
    schema = spec.get("parameters", {}) if isinstance(spec, dict) else {}
    # Strict provider schemas require every property, while nullable properties
    # encode Lumo defaults. The loop removes nulls before local execution.
    _validate_native_schema_object(args, schema)


def validate_tool(context, name, args):
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be an object")
    _validate_native_schema(name, args)

    if name == "list_files":
        path = context.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    if name == "glob":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = _glob_search_root(context, args)
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        _read_window_args(args)
        return

    if name == "grep":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = _grep_search_target(context, args)
        if not path.exists():
            raise ValueError("path does not exist")
        if not path.is_file() and not path.is_dir():
            raise ValueError("path must be a file or directory")
        _grep_output_mode(args)
        _grep_head_limit(args)
        _grep_offset(args)
        _grep_timeout_seconds(args)
        _grep_optional_nonnegative_int(args, "-A")
        _grep_optional_nonnegative_int(args, "-B")
        _grep_optional_nonnegative_int(args, "-C")
        return

    if name == "git_status":
        context.path(args.get("path", "."))
        gitlib.ensure_git_repository(context.root)
        _git_status_offset(args)
        _git_status_limit(args)
        return

    if name == "git_diff":
        context.path(args.get("path", "."))
        gitlib.ensure_git_repository(context.root)
        _git_diff_mode(args)
        _git_diff_offset(args)
        _git_diff_limit(args)
        return

    if name == "use_skill":
        skill_name = str(args.get("name", "")).strip()
        if not skill_name:
            raise ValueError("name must not be empty")
        catalog = context.skill_catalog()
        try:
            skilllib.load_skill_content(context.root, skill_name, catalog=catalog)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return

    if name == "todo_write":
        normalize_todo_items(args)
        return

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        validate_shell_command_syntax(command)
        timeout = int(args.get("timeout", RUN_SHELL_DEFAULT_TIMEOUT))
        if timeout < 1 or timeout > RUN_SHELL_MAX_TIMEOUT:
            raise ValueError(f"timeout must be in [1, {RUN_SHELL_MAX_TIMEOUT}]")
        return

    if name == "run_shell_bg":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        validate_shell_command_syntax(command)
        _run_shell_bg_timeout(args)
        return

    if name == "task_output":
        task_id = str(args.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("task_id must not be empty")
        if context.find_background_task(task_id) is None:
            raise ValueError("task_id does not exist")
        _grep_offset(args)
        _task_output_limit(args)
        _task_output_stream(args)
        return

    if name == "task_list":
        _grep_offset(args)
        _task_list_limit(args)
        _task_list_status(args)
        return

    if name == "task_stop":
        task_id = str(args.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("task_id must not be empty")
        if context.find_background_task(task_id) is None:
            raise ValueError("task_id does not exist")
        return

    if name == "write_file":
        path = context.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    if name == "patch_file":


        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        replace_all = _tool_bool_arg(args, "replace_all", False)
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count == 0:
            raise ValueError(f"old_text not found in file.\nString: {old_text}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"Found {count} matches for old_text, but replace_all is false. "
                "Set replace_all=true to replace all occurrences, or provide more surrounding context in old_text to make the target unique."
            )
        return


    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        inherit_context = args.get("inherit_context", True)
        if isinstance(inherit_context, str):
            if inherit_context.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("inherit_context must be a boolean")
        return


def tool_list_files(context, args):

    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"


def tool_glob(context, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    search_root = _glob_search_root(context, args)
    if not search_root.is_dir():
        raise ValueError("path is not a directory")

    matched = _iter_glob_matches(context.root, search_root, pattern)
    shown = matched[:GLOB_MAX_RESULTS]
    if search_root == context.root:
        rendered_search_path = "."
    else:
        rendered_search_path = search_root.relative_to(context.root).as_posix()
    header = [
        f"# glob pattern: {pattern}",
        f"# search path: {rendered_search_path}",
    ]
    if len(matched) > GLOB_MAX_RESULTS:
        header.append(f"# matched {len(matched)} files (showing first {GLOB_MAX_RESULTS})")
    else:
        header.append(f"# matched {len(matched)} files (showing {len(shown)})")
    body = shown or ["(no matches)"]
    hints = []
    if len(matched) > GLOB_MAX_RESULTS:
        hints.append(
            f"<system-reminder>Results were truncated at {GLOB_MAX_RESULTS} files. Narrow the pattern or path before retrying.</system-reminder>"
        )
    if len(matched) > GLOB_MAX_RESULTS:
        summary = (
            f"Globbed {rendered_search_path} with pattern {pattern}; matched {len(matched)} files; "
            f"showing first {GLOB_MAX_RESULTS}."
        )
    else:
        summary = (
            f"Globbed {rendered_search_path} with pattern {pattern}; matched {len(matched)} files; "
            f"showing {len(shown)}."
        )
    hints.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join([*header, *body, *hints])




def tool_read_file(context, args):
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    offset, limit, _limit_explicit = _read_window_request(args)
    relative_path = path.relative_to(context.root).as_posix()
    window = _read_file_window(path, offset, limit)
    body = "\n".join(window["lines"]) or "(empty)"
    header = (
        f"# {relative_path}\n"
        f"# lines {offset}-{window['end_line']} of at least {window['last_seen_line']} "
        f"(returned {window['returned_lines']}, requested {limit})"
    )
    hints = []
    if window["has_more"]:
        remaining_lines = max(0, int(window["last_seen_line"]) - int(window["end_line"]))
        hints.append(
            f"<tool_reminder>You already read {relative_path} lines {offset}-{window['end_line']}. "
            f"There are at least {remaining_lines} unread lines left based on the current file snapshot. "
            f"Do not reread the same window. If you need to continue, call "
            f'read_file with {{"path":"{relative_path}","offset":{window["next_offset"]},"limit":{limit}}} '
            f"to continue from the next unread line, or jump to a later targeted window only if you know where to inspect.</tool_reminder>"
        )
    summary = _summarize_read_window(
        relative_path,
        offset,
        window["lines"],
        window["has_more"],
        window["next_offset"],
    )
    hints.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join([header, body, *hints])



def tool_grep(context, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")

    if shutil.which("rg"):
        search_path = _grep_search_target(context, args)
        display_path = _display_search_path(search_path, context.root)
        timeout_seconds = _grep_timeout_seconds(args)
        try:
            return _run_rg_command(context, args)
        except subprocess.TimeoutExpired:
            _raise_grep_timeout(pattern, display_path, timeout_seconds)
    return _run_grep_fallback(context, args)


def tool_git_status(context, args):
    path = context.path(args.get("path", "."))
    offset = _git_status_offset(args)
    limit = _git_status_limit(args)
    return gitlib.git_status_text(context.root, path, offset=offset, limit=limit)


def tool_git_diff(context, args):
    path = context.path(args.get("path", "."))
    mode = _git_diff_mode(args)
    offset = _git_diff_offset(args)
    limit = _git_diff_limit(args)
    return gitlib.git_diff_text(
        context.root,
        path,
        mode=mode,
        offset=offset,
        limit=limit,
        artifact_writer=context.write_git_diff_artifact,
    )


def tool_use_skill(context, args):
    return context.load_skill(args)


def normalize_todo_items(args):
    raw_items = (args or {}).get("todos")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("todos must be a non-empty list")
    if len(raw_items) > TODO_WRITE_MAX_ITEMS:
        raise ValueError(f"todos must contain at most {TODO_WRITE_MAX_ITEMS} items")

    normalized = []
    seen_ids = set()
    active_count = 0
    blocked_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("each todo must be an object")
        todo_id = str(raw_item.get("id", "")).strip()
        text = str(raw_item.get("text", "")).strip()
        status = str(raw_item.get("status", "")).strip().lower()
        if not TODO_ID_PATTERN.fullmatch(todo_id):
            raise ValueError("todo id must be 1-32 letters, numbers, underscores, or hyphens")
        if todo_id in seen_ids:
            raise ValueError(f"duplicate todo id: {todo_id}")
        if not text:
            raise ValueError(f"todo text must not be empty: {todo_id}")
        if status not in TODO_STATUSES:
            raise ValueError(f"invalid todo status for {todo_id}: {status}")
        seen_ids.add(todo_id)
        active_count += int(status == "active")
        blocked_count += int(status == "blocked")
        normalized.append({"id": todo_id, "text": text, "status": status})

    if active_count > 1:
        raise ValueError("todo list may contain at most one active item")
    if blocked_count > 1:
        raise ValueError("todo list may contain at most one blocked item")
    if active_count and blocked_count:
        raise ValueError("todo list cannot contain both active and blocked items")
    if not active_count and not blocked_count:
        for item in normalized:
            if item["status"] == "pending":
                item["status"] = "active"
                break
    return normalized


def tool_todo_write(context, args):
    return context.write_todos(args)


def tool_run_shell(context, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    validate_shell_command_syntax(command)
    timeout = int(args.get("timeout", RUN_SHELL_DEFAULT_TIMEOUT))
    if timeout < 1 or timeout > RUN_SHELL_MAX_TIMEOUT:
        raise ValueError(f"timeout must be in [1, {RUN_SHELL_MAX_TIMEOUT}]")
    try:
        prepared = context.prepare_shell_command(command)
    except Exception as exc:
        return textwrap.dedent(
            f"""\
            python_env_used: true
            python_env_error: {exc}
            exit_code: 1
            stdout:
            (empty)
            stderr:
            {exc}
            """
        ).strip()
    environment_lines = [f"python_env_used: {'true' if prepared.get('python_env_used') else 'false'}"]
    if prepared.get("python_env_used"):
        environment_lines.extend(
            [
                f"python_env_path: {prepared.get('python_env_path', '')}",
                f"python_executable: {prepared.get('python_executable', '')}",
                f"python_env_status: {prepared.get('environment_status', '')}",
                f"executed_command: {prepared.get('command', '')}",
            ]
        )
    stdout_fd, stdout_path = tempfile.mkstemp(prefix="lumo-shell-", suffix=".stdout.log")
    stderr_fd, stderr_path = tempfile.mkstemp(prefix="lumo-shell-", suffix=".stderr.log")
    timed_out = False
    try:
        with os.fdopen(stdout_fd, "wb") as stdout_handle, os.fdopen(stderr_fd, "wb") as stderr_handle:
            try:
                result = subprocess.run(
                    prepared["command"],
                    cwd=context.root,
                    shell=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=timeout,
                    env=prepared["env"],
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                timed_out = True
        return ShellOutputCapture(
            stdout_path=Path(stdout_path),
            stderr_path=Path(stderr_path),
            environment_lines=tuple(environment_lines),
            returncode=returncode,
            timed_out=timed_out,
        )
    except Exception:
        for path in (stdout_path, stderr_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def tool_run_shell_bg(context, args):
    validate_shell_command_syntax(args.get("command", ""))
    return context.start_background_task(args)


def tool_task_output(context, args):
    return context.read_background_task(args)


def tool_task_list(context, args):
    return context.list_background_tasks(args)


def tool_task_stop(context, args):
    return context.stop_background_task(args)


def tool_write_file(context, args):
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context, args):
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    replace_all = _tool_bool_arg(args, "replace_all", False)
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        raise ValueError(f"old_text not found in file.\nString: {old_text}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"Found {count} matches for old_text, but replace_all is false. "
            "Set replace_all=true to replace all occurrences, or provide more surrounding context in old_text to make the target unique."
        )
    updated = (
        text.replace(old_text, str(args["new_text"]))
        if replace_all
        else text.replace(old_text, str(args["new_text"]), 1)
    )
    path.write_text(updated, encoding="utf-8")
    return f"patched {path.relative_to(context.root)}"



def tool_delegate(context, args):
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "glob": tool_glob,
    "read_file": tool_read_file,
    "grep": tool_grep,
    "git_status": tool_git_status,
    "git_diff": tool_git_diff,
    "use_skill": tool_use_skill,
    "todo_write": tool_todo_write,
    "run_shell": tool_run_shell,
    "run_shell_bg": tool_run_shell_bg,
    "task_output": tool_task_output,
    "task_list": tool_task_list,
    "task_stop": tool_task_stop,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
