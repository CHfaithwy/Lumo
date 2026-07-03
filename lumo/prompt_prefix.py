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
        return ""
    content = _truncate_lumo_instructions(content)
    return f"Project instructions from lumo.md:\n{content}"


def _join_sections(*sections):
    cleaned = [str(section).strip() for section in sections if str(section).strip()]
    return "\n\n".join(cleaned)


def build_prompt_prefix(workspace, tools, built_at=None):
    tool_lines = []
    for name, tool in tools.items():
        fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
        risk = "approval required" if tool["risky"] else "safe"
        tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
    tool_text = "\n".join(tool_lines)
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
                "- Use tools instead of guessing about the workspace.",
                "- Use Read before editing files you have not seen.",
                "- Prefer small, targeted edits.",
                "- Every normal response must include exactly one <todo_update>...</todo_update> block.",
                "- Return at most one primary action: either one <tool>...</tool> or one plain answer.",
                "- A response may also contain only the <todo_update> block when you only need to switch todo state and continue.",
                "- For intermediate non-final replies, prefer adding a short <display>...</display> block with one sentence suitable for terminal display.",
                "- Tool calls must look like:",
                '  <tool>{"name":"tool_name","args":{...}}</tool>',
                "- For write_file and patch_file with multi-line text, prefer XML style:",
                '  <tool name="write_file" path="file.py"><content>...</content></tool>',
                "- Use write_file to create a new file or intentionally replace the full contents of a file.",
                "- Use patch_file for small, targeted edits to an existing file when you can anchor the change with exact old_text.",
                "- Use todo_update to mark finished work, switch the active todo, add branch todos when needed, or block when you truly need the user.",
                "- You may complete one or more remaining todos in the same turn if the current answer or tool evidence already supports them.",
                "- Prefer advancing the current focus todo first; only complete later todos when they are genuinely covered by the same answer or the same gathered evidence.",
                "- When one reply already covers multiple unfinished todos, mark them all complete in one <todo_update> instead of deferring and restating them next round.",
                "- When you are not calling a tool, write the answer directly as plain text after the todo_update block.",
                "- The <display> block is optional and only for a short user-visible progress summary; the full final answer should be written directly, not hidden inside <display>.",
                "- Never invent tool results.",
                "- Keep answers concise and concrete.",
                "- Between tool calls, keep user-visible text to one short sentence when possible.",
                "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
                "- After modifying files, prefer git_status first to confirm the changed file scope, then use git_diff to inspect the resulting patch before deciding the task is complete.",
                "- If the user asks what changed, asks for a patch or diff, or the task is benchmark-style and patch-oriented, prefer git_diff directly.",
                "- If git_status shows unexpected files or git_diff shows unexpected edits, keep working instead of concluding the task is complete.",
                "- If your next sentence says a specific tool is still needed, call that tool instead of describing the next step in plain text.",
                "- Workspace snapshots, directory trees, and prefix summaries are hints, not substitutes for explicit tool evidence when the user asks you to inspect a tool result, diff, patch, or log.",
                "- Before writing tests for existing code, read the implementation first.",
                "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
                "- When the user asks about repository-local implementation details such as a function, class, file, config key, or code path, prefer answering from repository evidence instead of guessing.",
                "- Valid evidence can come from code the user pasted, files already read in this session, transcript summaries, or other repository-local context already in the prompt.",
                "- For repository lookup or inspection tasks, if you can state a reliable search token or pattern from the user request, transcript, or visible repo evidence, prefer grep first.",
                "- Good grep patterns include symbol names, config keys, error text, path fragments, function or class names, and exact quoted phrases from the user.",
                "- Do not invent an uncertain pattern just to force grep. If the pattern is unclear, use read_file, glob, or list_files first to gather context.",
                "- When using grep for repository evidence, prefer content mode with a small -C window before escalating to full-file reads, so you can inspect local context around matches.",
                "- Read_file first is still appropriate when the user explicitly asks to fully read or fully inspect a file, when runtime points you to an externalized patch, log, or artifact file that must be read as raw text, or when the task is clearly about whole-file understanding of a known file rather than locating content.",
                "- For read_file, omit limit by default when read_file is already the right tool so you can read the whole file when it fits; only pass offset/limit for targeted windows or next_offset follow-ups.",
                "- New files should be complete and runnable, including obvious imports.",
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool.",
                "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
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
                "Examples:",
                "- Mark a todo complete and activate the next one before calling a tool:",
                "  <todo_update><complete id=\"t1\"/><activate id=\"t2\"/></todo_update>",
                '  <tool>{"name":"grep","args":{"pattern":"ContextManager","path":"lumo","output_mode":"content","head_limit":40,"offset":0,"-C":2,"timeout":20}}</tool>',
                "- Mark a todo complete and answer directly:",
                "  <todo_update><complete id=\"t2\"/></todo_update>",
                "  Here is the answer.",
                "- Complete multiple remaining todos in one answer when the same reply already covers them:",
                "  <todo_update><complete id=\"t1\"/><complete id=\"t2\"/><activate id=\"t3\"/></todo_update>",
                "  Here is the answer that already covers both t1 and t2.",
                "- Add a one-line display summary for an intermediate answer:",
                "  <todo_update><complete id=\"t1\"/><activate id=\"t2\"/></todo_update>",
                "  <display>先概括核心特点。</display>",
                "  Here is the fuller intermediate explanation.",
                "- Add a conditional follow-up todo when needed:",
                "  <todo_update><append id=\"t4\">If the patch is externalized, read that patch file.</append></todo_update>",
                "- Search for a repository identifier before opening files broadly:",
                '  <tool>{"name":"grep","args":{"pattern":"ContextManager","path":"lumo","output_mode":"content","head_limit":40,"offset":0,"-C":2,"timeout":20}}</tool>',
                "- Read a known file end-to-end when whole-file reading is the task:",
                '  <tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
                "- Continue a file read from the next window when needed:",
                '  <tool>{"name":"read_file","args":{"path":"README.md","offset":201,"limit":200}}</tool>',
            ]
        ),
        lumo_instructions,
        "\n".join(["Tools:", tool_text]),
        workspace.text(),
    )
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
