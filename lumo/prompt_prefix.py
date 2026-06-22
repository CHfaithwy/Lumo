"""Stable prompt prefix construction."""

import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .workspace import now


LUMO_INSTRUCTIONS_FILE = "lumo.md"
MAX_LUMO_INSTRUCTION_UNITS = 3000
MAX_LUMO_INSTRUCTIONS_CHARS = 10000
_CJK_CHAR_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")


@dataclass
class PromptPrefix:


    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


def tool_signature(tools):
    payload = []
    for name in sorted(tools):
        tool = tools[name]
        payload.append(
            {
                "name": name,
                "schema": tool["schema"],
                "risky": tool["risky"],
                "description": tool["description"],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _is_cjk_char(char):
    return bool(_CJK_CHAR_PATTERN.fullmatch(char))


def _truncate_lumo_instructions(content):
    """Limit lumo.md by English words/CJK chars, with a hard char cap."""
    char_limit = min(len(content), MAX_LUMO_INSTRUCTIONS_CHARS)
    index = 0
    units = 0

    while index < char_limit and units < MAX_LUMO_INSTRUCTION_UNITS:
        char = content[index]
        if _is_cjk_char(char):
            units += 1
            index += 1
            continue

        match = _WORD_PATTERN.match(content, index)
        if match:
            units += 1
            index = min(match.end(), char_limit)
            continue

        index += 1

    if index >= len(content):
        return content

    trimmed = content[:index].rstrip()
    limit_reasons = []
    if units >= MAX_LUMO_INSTRUCTION_UNITS:
        limit_reasons.append(f"{MAX_LUMO_INSTRUCTION_UNITS} words/CJK chars")
    if index >= MAX_LUMO_INSTRUCTIONS_CHARS:
        limit_reasons.append(f"{MAX_LUMO_INSTRUCTIONS_CHARS} chars")
    reason = " and ".join(limit_reasons) if limit_reasons else "configured limit"
    return f"{trimmed}\n...[truncated {len(content) - index} chars after {reason}]"


def _load_lumo_instructions(workspace):
    path = Path(workspace.repo_root) / LUMO_INSTRUCTIONS_FILE
    if not path.is_file():
        return "Project instructions from lumo.md:\n- none"
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return "Project instructions from lumo.md:\n- empty"
    content = _truncate_lumo_instructions(content)
    return f"Project instructions from lumo.md:\n{content}"


def build_prompt_prefix(workspace, tools, built_at=None):
    tool_lines = []
    for name, tool in tools.items():
        fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
        risk = "approval required" if tool["risky"] else "safe"
        tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
    tool_text = "\n".join(tool_lines)
    examples = "\n".join(
        [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","offset":1,"limit":80}}</tool>',
            '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "<final>Done.</final>",
        ]
    )
    lumo_instructions = _load_lumo_instructions(workspace)


    text = textwrap.dedent(
        f"""\
        You are LUMO, a small coding agent working inside a local repository.
        You are an interactive coding agent. Help the user edit, inspect, and understand the local codebase.
        Use tools when needed. Be careful with destructive operations.
        Follow the user's instructions.

        Rules:
        - Use tools instead of guessing about the workspace.
        - Use Read before editing files you have not seen.
        - Prefer small, targeted edits.
        - Return exactly one <tool>...</tool> or one <final>...</final>.
        - Tool calls must look like:
          <tool>{{"name":"tool_name","args":{{...}}}}</tool>
        - For write_file and patch_file with multi-line text, prefer XML style:
          <tool name="write_file" path="file.py"><content>...</content></tool>
        - Final answers must look like:
          <final>your answer</final>
        - Never invent tool results.
        - Keep answers concise and concrete.
        - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
        - Before writing tests for existing code, read the implementation first.
        - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
        - New files should be complete and runnable, including obvious imports.
        - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
        - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.

        Language:
        Respond in the same language as the user unless instructed otherwise.

        {lumo_instructions}

        Tools:
        {tool_text}

        Valid response examples:
        {examples}

        {workspace.text()}
        """
    ).strip()
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
