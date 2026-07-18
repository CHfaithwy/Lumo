

from __future__ import annotations

import json
import re


_JSON_TOOL = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)


def migrate_history(history):

    items = [dict(item) if isinstance(item, dict) else {"role": "assistant", "content": str(item)} for item in list(history or [])]
    migrated = []
    pending_calls = []
    changed = False
    for index, item in enumerate(items):
        role = str(item.get("role", ""))
        if role == "assistant" and not item.get("tool_calls"):
            match = _JSON_TOOL.search(str(item.get("content", "")))
            if match:
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and isinstance(payload.get("args", {}), dict) and str(payload.get("name", "")).strip():
                    call_id = f"legacy_{index}"
                    item["tool_calls"] = [{"call_id": call_id, "name": str(payload["name"]), "arguments": dict(payload["args"]), "raw_arguments": match.group(1), "error": ""}]
                    item["content"] = _JSON_TOOL.sub("", str(item.get("content", ""))).strip()
                    pending_calls.append(item["tool_calls"][0])
                    changed = True
        elif role == "tool" and not item.get("call_id"):
            if pending_calls:
                call = pending_calls.pop(0)
                if str(item.get("name", "")) == call["name"]:
                    item["call_id"] = call["call_id"]
                    item.setdefault(
                        "result",
                        {
                            "status": str((item.get("metadata") or {}).get("tool_status", "ok")),
                            "content": str(item.get("content", "")),
                            "metadata": dict(item.get("metadata", {}) or {}),
                            "error": str((item.get("metadata") or {}).get("tool_error_code", "")),
                        },
                    )
                    changed = True
                else:
                    pending_calls.clear()
    return migrated + items, changed
