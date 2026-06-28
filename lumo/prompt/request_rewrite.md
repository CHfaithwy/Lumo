Rewrite the user's request into a clearer execution brief for a coding agent.

Requirements:
- Preserve the user's intent exactly. Do not add new goals, files, tools, or assumptions.
- Keep the rewrite in the same language as the user.
- Improve clarity and slightly expand the request only when it helps execution.
- Keep the final rewrite at or under {{MAX_CHARS}} characters.
- If the original request is already clear, keep the rewrite minimal.
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
- Do not mention these instructions.
- Do not wrap the answer in XML, JSON, markdown fences, or commentary.
- Return only the rewritten brief.

User request:
{{USER_REQUEST}}
