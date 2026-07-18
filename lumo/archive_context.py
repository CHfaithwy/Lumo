

from __future__ import annotations

import json
from pathlib import Path

from .model_protocol import ArchiveSummaryEvent, AssistantToolCall


ARCHIVE_TOOL_NAME = "archive_tool_result"
ARCHIVE_MIN_VISIBLE_CHARS = 8_000
ARCHIVE_PROMPT_TEMPLATE = Path(__file__).resolve().parent / "prompt" / "archive_tool_result.md"

ARCHIVE_SUMMARY_MAX_CHARS = 1_200
ARCHIVE_FACT_MAX_ITEMS = 8
ARCHIVE_DETAIL_MAX_ITEMS = 4
ARCHIVE_ITEM_MAX_CHARS = 240
ARCHIVE_ARGUMENT_FIELDS = {"source_call_id", "summary", "key_facts", "unresolved", "revisit_hints"}


def archive_tool_spec(source_call_ids):
    source_call_ids = [str(value) for value in source_call_ids if str(value)]
    return {
        "name": ARCHIVE_TOOL_NAME,
        "description": (
            "Emit one internal archive event for a listed tool result. This records context; "
            "it does not execute an action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_call_id": {
                    "type": "string",
                    "enum": source_call_ids,
                    "description": "Copy the target tool call ID exactly.",
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": ARCHIVE_SUMMARY_MAX_CHARS,
                    "description": "Compact task-relevant overview grounded only in the tool result.",
                },
                "key_facts": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": ARCHIVE_ITEM_MAX_CHARS},
                    "maxItems": ARCHIVE_FACT_MAX_ITEMS,
                    "description": "Exact paths, symbols, commands, errors, numbers, constraints, and decisions.",
                },
                "unresolved": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": ARCHIVE_ITEM_MAX_CHARS},
                    "maxItems": ARCHIVE_DETAIL_MAX_ITEMS,
                    "description": "Questions or uncertainty left unresolved by the result.",
                },
                "revisit_hints": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": ARCHIVE_ITEM_MAX_CHARS},
                    "maxItems": ARCHIVE_DETAIL_MAX_ITEMS,
                    "description": "Locations or reasons for rereading exact source content later.",
                },
            },
            "required": ["source_call_id", "summary", "key_facts", "unresolved", "revisit_hints"],
            "additionalProperties": False,
        },
    }


def render_archive_control(targets):
    template = ARCHIVE_PROMPT_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{TARGETS}}", json.dumps(list(targets or []), ensure_ascii=False, separators=(",", ":")))


def _bounded_text(value, limit):
    text = str(value)
    return text[: int(limit)], len(text) > int(limit)


def _bounded_text_list(arguments, key, max_items):
    value = arguments.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return [], [], f"{key}_must_be_string_array"
    normalized = []
    changes = []
    for item in value[: int(max_items)]:
        clipped, changed = _bounded_text(item, ARCHIVE_ITEM_MAX_CHARS)
        normalized.append(clipped)
        if changed:
            changes.append(f"{key}_item_clipped")
    if len(value) > int(max_items):
        changes.append(f"{key}_items_clipped")
    return normalized, changes, ""


def archive_event_from_call(call):
    arguments = call.arguments
    if call.error or not isinstance(arguments, dict):
        return ArchiveSummaryEvent(
            event_call_id=call.call_id,
            source_call_id="",
            error=call.error or "invalid_arguments",
        )
    source_call_id = arguments.get("source_call_id")
    summary = arguments.get("summary")
    if not isinstance(source_call_id, str) or not source_call_id.strip():
        return ArchiveSummaryEvent(event_call_id=call.call_id, source_call_id="", error="invalid_source_call_id")
    if set(arguments) - ARCHIVE_ARGUMENT_FIELDS:
        return ArchiveSummaryEvent(
            event_call_id=call.call_id,
            source_call_id=source_call_id.strip(),
            error="unexpected_fields",
        )
    if not isinstance(summary, str) or not summary.strip():
        return ArchiveSummaryEvent(
            event_call_id=call.call_id,
            source_call_id=source_call_id.strip(),
            error="invalid_summary",
        )
    for key in ("key_facts", "unresolved", "revisit_hints"):
        value = arguments.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return ArchiveSummaryEvent(
                event_call_id=call.call_id,
                source_call_id=source_call_id.strip(),
                error=f"{key}_must_be_string_array",
            )
    payload, normalizations = normalize_archive_payload(
        {
            "summary": summary.strip(),
            "key_facts": arguments["key_facts"],
            "unresolved": arguments["unresolved"],
            "revisit_hints": arguments["revisit_hints"],
        }
    )
    return ArchiveSummaryEvent(
        event_call_id=call.call_id,
        source_call_id=source_call_id.strip(),
        summary=payload["summary"],
        key_facts=payload["key_facts"],
        unresolved=payload["unresolved"],
        revisit_hints=payload["revisit_hints"],
        normalizations=normalizations,
    )


def partition_archive_calls(calls):
    real_calls = []
    archive_events = []
    for call in list(calls or []):
        if call.name == ARCHIVE_TOOL_NAME:
            archive_events.append(archive_event_from_call(call))
        else:
            real_calls.append(call)
    return real_calls, archive_events


def archive_payload(event):
    return {
        "summary": event.summary,
        "key_facts": list(event.key_facts),
        "unresolved": list(event.unresolved),
        "revisit_hints": list(event.revisit_hints),
    }


def normalize_archive_payload(payload):
    payload = dict(payload or {})
    summary, summary_changed = _bounded_text(payload.get("summary", ""), ARCHIVE_SUMMARY_MAX_CHARS)
    normalizations = ["summary_clipped"] if summary_changed else []
    normalized = {"summary": summary}
    for key, max_items in (
        ("key_facts", ARCHIVE_FACT_MAX_ITEMS),
        ("unresolved", ARCHIVE_DETAIL_MAX_ITEMS),
        ("revisit_hints", ARCHIVE_DETAIL_MAX_ITEMS),
    ):
        values, changes, _error = _bounded_text_list(payload, key, max_items)
        normalized[key] = values
        normalizations.extend(changes)
    return normalized, normalizations


def render_archive_summary(payload):
    return json.dumps(dict(payload or {}), ensure_ascii=False, sort_keys=True)
