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
        - Every response must include exactly one <completion>0-100</completion> tag, where the score represents how complete the current user request or instruction is overall.
        - Return at most one primary action: either one <tool>...</tool> or one plain answer.
        - Tool calls must look like:
          <tool>{{"name":"tool_name","args":{{...}}}}</tool>
        - For write_file and patch_file with multi-line text, prefer XML style:
          <tool name="write_file" path="file.py"><content>...</content></tool>
        - Judge completion against the current request and transcript. If the task is not actually complete yet, keep working and give a lower completion score.
        - When you are not calling a tool, write the answer directly as plain text after the completion tag.
        - Never invent tool results.
        - Keep answers concise and concrete.
        - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
        - Before writing tests for existing code, read the implementation first.
        - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
        - When the user asks about repository-local implementation details such as a function, class, file, config key, or code path, prefer answering from repository evidence instead of guessing.
        - Valid evidence can come from code the user pasted, files already read in this session, transcript summaries, or other repository-local context already in the prompt.
        - When searching for repository evidence, prefer grep in content mode with a small -C window before jumping into full-file reads, so you can inspect the local context around matches.
        - If the user asked to fully read, fully inspect, or completely review a file, and the transcript still shows unread file content remains, do not output a high completion score yet.
        - New files should be complete and runnable, including obvious imports.
        - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool.
        - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.

        Completion scoring:
        - 0-25: You have barely started, have very limited evidence, or are still identifying what to inspect.
        - 25-50: You have some useful evidence, but large parts of the user's request are still unresolved.
        - 50-95: You have substantial partial coverage, but important reading, verification, or explanation is still missing.
        - 95-100: Only use this range when the current tool results and transcript are already sufficient to support all of the user's requirements. If anything important is still missing, do not output a score in this range.

        Language:
        Respond in the same language as the user unless instructed otherwise.

        {lumo_instructions}

        Tools:
        {tool_text}

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
