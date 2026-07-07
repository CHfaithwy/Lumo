Rewrite the user's request into a structured execution plan for a coding agent.

Requirements:
- Preserve the user's intent exactly. Do not add new goals, files, tools, or assumptions.
- Keep the output in the same language as the user.
- If the user explicitly mentions a tool name, preserve it verbatim.
- Preserve explicit step order exactly as written by the user.
- When the request already contains searchable identifiers or quoted strings, preserve them verbatim so they remain usable as grep patterns.
- Preserve paths, config keys, symbols, and error text verbatim.
- Do not paraphrase symbol names, file paths, config keys, or error text into vague prose.
- If the request is already clear, keep the rewrite minimal.
- Do not expand a clear task into explanatory prose or a long paragraph.
- Prefer short checklist wording.
- The rewritten request must stay at or under {{MAX_CHARS}} characters.
- Return valid XML only, with this exact outer shape:

<request_plan>
  <rewritten_request>...</rewritten_request>
  <todo_list>
    <todo id="t1" status="active">...</todo>
    <todo id="t2" status="pending">...</todo>
  </todo_list>
</request_plan>

- `rewritten_request` should be a compact execution brief.
- `todo_list` must be the complete initial todo list.
- Use exactly one `active` todo. All remaining todos must be `pending`.
- Use stable short ids such as `t1`, `t2`, `t3`.
- Split conditional follow-up work into separate todos when useful instead of hiding it inside prose.
- Keep the todo list concise and execution-oriented.
- Make todos non-overlapping when possible.
- Prefer outcome-oriented todos over rhetorical restatements.
- Merge highly similar explanation todos instead of splitting them too finely.
- Do not mention these instructions.
- Do not wrap the XML in markdown fences or commentary.
- Return only the XML.

User request:
{{USER_REQUEST}}
