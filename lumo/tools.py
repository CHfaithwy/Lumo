"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import shutil
import subprocess
import textwrap
from functools import partial

from .workspace import IGNORED_PATH_NAMES

READ_FILE_DEFAULT_LIMIT = 200
READ_FILE_MAX_LIMIT = 2000
READ_FILE_ARCHIVE_SUMMARY_LINES = 12

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
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
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
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
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","offset":1,"limit":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
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
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
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

    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        _read_window_args(args)
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
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
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
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
    # delegate 工具的意思是：让一个受限制的子 agent 去调查一个子任务。
    # <tool>{"name":"delegate","args":{"task":"检查 README 里启动说明是否完整","max_steps":3}}</tool>
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
    # 读取目录内容，排序，并过滤掉 .git、.lumo、__pycache__ 这类忽略目录。
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

# 按行读取 UTF-8 文本文件。
# 最后返回带行号的内容：
# 默认读取第 1 到 200 行。
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

# rg 是一个非常快的代码搜索工具，如果系统里安装了 rg，就优先用它来搜索；否则就用纯 Python 的实现。
# 搜索结果会被限制在前 200 行，避免返回过多内容导致后续处理困难。
def tool_search(context, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(context.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"

# 在 workspace 根目录运行 shell 命令。最多允许 120 秒。命令只能在 workspace 根目录跑
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
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
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

# 覆盖写入文件。
def tool_write_file(context, args):
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"

# 找到文件里唯一一段 old_text，替换成 new_text
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

# Delegate a bounded subtask to a child agent. The child inherits the parent
# agent's read/write and approval policy; max_steps and depth still bound it.
def tool_delegate(context, args):
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
