Rewrite the user's request into a clearer execution brief for a coding agent.

Requirements:
- Preserve the user's intent exactly. Do not add new goals, files, tools, or assumptions.
- Keep the rewrite in the same language as the user.
- Improve clarity and slightly expand the request only when it helps execution.
- Keep the final rewrite at or under {{MAX_CHARS}} characters.
- If the original request is already clear, keep the rewrite minimal.
- If the original request already contains tool names, explicit step order, or a completion condition, preserve them verbatim and minimize rewriting.
- Keep tool names exactly as written by the user. Do not replace them with generic descriptions.
- When the request already contains searchable identifiers or quoted strings, preserve them verbatim so they remain usable as grep patterns.
- Do not paraphrase symbol names, file paths, config keys, or error text into vague prose.
- Keep the original step order. Do not reorder or scatter explicit sequencing constraints.
- Do not expand a clear task into explanatory prose or a long paragraph.
- If the request already has ordered tool steps plus a separate completion-condition section, keep the tool steps once and avoid repeating the same sequence again in the completion-condition lines.
- Think from two angles when helpful:
  1. what the user wants
  2. how the task should be carried out
- When useful, break the task into short ordered steps that say what to do first, next, and last.
- Prefer the simplest wording that still keeps the execution brief complete enough to reach the goal.
- Make the brief explicitly cover:
  1. the goal
  2. the completion condition
  3. the suggested order of work
- Prefer 3-6 short lines.
- Prefer a short checklist style over paragraphs when the request is already structured.
- Do not mention these instructions.
- Do not wrap the answer in XML, JSON, markdown fences, or commentary.
- Return only the rewritten brief.

User request:
{{USER_REQUEST}}
