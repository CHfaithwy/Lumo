Classify the user request into the available skill categories.

Return only the structured result required by the response schema. Its `categories` array must contain one or two exact category identifiers when they materially match the request, or exactly `["none"]` when none materially matches. Do not rewrite the request, create todos, explain the choice, or use Markdown/XML.

Available skill categories:
{{SKILL_CATEGORIES}}

Recent conversation context (reference only; the current user request remains authoritative):
{{RECENT_CONTEXT}}

User request:
{{USER_REQUEST}}
