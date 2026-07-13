"""Stable prompt prefix construction."""

import hashlib
import json
import re
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
    tool_signature: str
    built_at: str


def tool_signature(tools):
    payload = []
    for name in sorted(tools):
        tool = tools[name]
        payload.append(
            {
                "name": name,
                "parameters": tool.get("parameters", {}),
                "risky": tool["risky"],
                "concurrency_safe": bool(tool.get("concurrency_safe", False)),
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
        return ""
    content = _truncate_lumo_instructions(content)
    return f"Project instructions from lumo.md:\n{content}"


def _join_sections(*sections):
    cleaned = [str(section).strip() for section in sections if str(section).strip()]
    return "\n\n".join(cleaned)


def build_prompt_prefix(workspace, tools, built_at=None):
    lumo_instructions = _load_lumo_instructions(workspace)
    text = _join_sections(
        "\n".join(
            [
                "You are LUMO, a small coding agent working inside a local repository.",
                "You are an interactive coding agent. Help the user edit, inspect, and understand the local codebase.",
                "Use tools when needed. Be careful with destructive operations.",
                "Follow the user's instructions.",
            ]
        ),
        "\n".join(
            [
                "Rules:",
                "- Use tools for workspace facts instead of guessing; read files before editing unseen code.",
                "- Prefer small, targeted edits.",
                "- Simple questions and one-step tasks do not need a todo list; answer directly or call the needed tool.",
                "- Use todo_write only for genuinely multi-step work, such as multiple files, verification phases, background tasks, or an explicit planning request.",
                "- Once a todo list exists, keep its statuses current with todo_write and advance the active item first.",
                "- Use native function tools when workspace evidence or an action is needed; tool arguments are validated by runtime.",
                "- Assistant text accompanying native tool calls is optional. When present, use it only for context the tool call itself does not show; do not restate the imminent tool action. Without a tool call, text is the final answer.",
                "- Never invent tool results.",
                "- Keep answers concise and concrete.",
                "- If your next sentence says a specific tool is still needed, call that tool instead of describing the next step in plain text.",
                "- Workspace snapshots, directory trees, and prefix summaries are hints, not substitutes for explicit tool evidence.",
                "- Before writing tests for existing code, read the implementation first.",
                "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
                "- For repository-local details, answer from evidence: pasted code, read files, transcript summaries, or visible repo context.",
                "- If an available skill clearly matches the task, call use_skill first and then follow that workflow.",
                "- New files should be complete and runnable, including obvious imports.",
                "- Do not repeat an unhelpful identical tool call; choose a different tool or narrower arguments.",
                "- Required tool arguments must not be empty.",
            ]
        ),
        "\n".join(
            [
                "Language:",
                "Respond in the same language as the user unless instructed otherwise.",
            ]
        ),
        "\n".join(
            [
                "Native response behavior:",
                "- Use the provider's native function-call interface for tools. Do not emit XML tool, display, final, or todo tags.",
                "- Call todo_write only for genuinely multi-step work; use status=done when the plan is complete.",
                "- A text response without a native tool call is returned to the user as the final answer.",
            ]
        ),
        lumo_instructions,
    )
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
