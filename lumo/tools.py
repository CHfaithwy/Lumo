"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import fnmatch
import re
import shutil
import subprocess
import textwrap
import time
from functools import partial

from .workspace import IGNORED_PATH_NAMES

READ_FILE_DEFAULT_LIMIT = 200
READ_FILE_MAX_LIMIT = 2000
READ_FILE_ARCHIVE_SUMMARY_LINES = 12
GLOB_MAX_RESULTS = 200
GREP_DEFAULT_HEAD_LIMIT = 200
GREP_OUTPUT_MODES = {"content", "files", "count"}
GREP_TIMEOUT_SECONDS = 20

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "glob": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": (
            "Find files by glob pattern under a directory. Use this to discover candidate files before "
            "calling read_file or grep. If results are truncated, narrow the pattern or path and retry."
        ),
    },
    "read_file": {
        "schema": {"path": "str", "offset": "int=1", "limit": "int=200"},
        "risky": False,
        "description": (
            "Read a UTF-8 file by line window. Use offset/limit for long files; "
            'if has_more is true, continue with next_offset or a targeted window like {"offset":8600,"limit":300}. '
            "After each chunk, summarize the relevant facts before deciding whether to read more."
        ),
    },
    "grep": {
        "schema": {
            "pattern": "str",
            "path": "str='.'",
            "output_mode": "str='content'",
            "head_limit": f"int={GREP_DEFAULT_HEAD_LIMIT}",
            "offset": "int=0",
            "glob": "str|None=None",
            "-A": "int|None=None",
            "-B": "int|None=None",
            "-C": "int|None=None",
            "timeout": f"int={GREP_TIMEOUT_SECONDS}",
        },
        "risky": False,
        "description": (
            "Search file contents with rg or a simple fallback. Use output_mode='files' to list matching files, "
            "output_mode='count' for per-file match counts, glob to narrow file candidates, head_limit+offset to page "
            "results, -A/-B/-C for surrounding lines in content mode, and timeout to bound slow searches. "
            "If grep times out, treat it as incomplete search rather than proof of no matches."
        ),
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}

DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3", "inherit_context": "bool=True"},
    "risky": False,
    "description": (
        "Ask a bounded child agent to work on a subtask. With inherit_context=true, "
        "the child receives the parent's compressed history; use this when the subtask depends on prior conversation, files, or decisions. "
        "With inherit_context=false, the child starts without parent history; use this for independent subtasks unrelated to the main agent's context."
    ),
}


def legal_tool_names():
    return set(BASE_TOOL_SPECS) | {"delegate"}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "glob": '<tool>{"name":"glob","args":{"pattern":"**/*.py","path":"lumo"}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","offset":1,"limit":80}}</tool>',
    "grep": '<tool>{"name":"grep","args":{"pattern":"binary_search","path":"lumo","output_mode":"content","head_limit":20,"offset":0,"timeout":20}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3,"inherit_context":true}}</tool>',
}


def _read_window_args(args):
    """Return one-based offset and line limit, accepting legacy start/end."""
    args = args or {}
    offset = int(args.get("offset", args.get("start", 1)))
    if "limit" in args:
        limit = int(args.get("limit", READ_FILE_DEFAULT_LIMIT))
    elif "end" in args:
        end = int(args.get("end", offset + READ_FILE_DEFAULT_LIMIT - 1))
        if end < offset:
            raise ValueError("invalid line range")
        limit = end - offset + 1
    else:
        limit = READ_FILE_DEFAULT_LIMIT
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > READ_FILE_MAX_LIMIT:
        raise ValueError(f"limit must be <= {READ_FILE_MAX_LIMIT}")
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
    if value > 120:
        raise ValueError("timeout must be <= 120")
    return value


def _grep_optional_nonnegative_int(args, key):
    value = args.get(key, None)
    if value in (None, ""):
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
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
            "<system-reminder>Results were paginated. Increase offset or narrow the pattern, path, or glob before retrying.</system-reminder>"
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
    lowered_pattern = pattern.lower()
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
                if lowered_pattern in line.lower():
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


def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(context, name, args):
    args = args or {}

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

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
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
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
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
    offset, limit = _read_window_args(args)
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
        hints.append(
            f"<system-reminder>The lines above are the current read window. If they are not enough to answer "
            f"the user's question or support your judgment, continue with "
            f'read_file args {{"path":"{relative_path}","offset":{window["next_offset"]},"limit":{limit}}} '
            f'or jump to a targeted window like {{"offset":8600,"limit":300}} if you know where to inspect.</system-reminder>'
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


def tool_run_shell(context, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    result = subprocess.run(
        command,
        cwd=context.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,


        env=context.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


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
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
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
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
