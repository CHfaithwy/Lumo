"""Provider-neutral native model and tool-call protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AssistantToolCall:
    """One model-requested function invocation with a stable provider call ID."""

    call_id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments or {}),
            "raw_arguments": self.raw_arguments,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssistantToolCall":
        arguments = value.get("arguments", {})
        return cls(
            call_id=str(value.get("call_id", "")),
            name=str(value.get("name", "")),
            arguments=dict(arguments) if isinstance(arguments, dict) else None,
            raw_arguments=str(value.get("raw_arguments", "")),
            error=str(value.get("error", "")),
        )


@dataclass(frozen=True)
class ToolResultMessage:
    """A local tool result that must be returned to the matching model call."""

    call_id: str
    name: str
    output: dict[str, Any]

    def to_history_item(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "call_id": self.call_id,
            "name": self.name,
            "result": dict(self.output),
        }


@dataclass
class ModelTurnRequest:
    """Canonical request consumed by all provider adapters."""

    instructions: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_output_tokens: int = 0
    reasoning_effort: str | None = None
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    structured_schema: dict[str, Any] | None = None
    structured_name: str = "structured_output"
    force_structured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            "messages": list(self.messages),
            "tools": list(self.tools),
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "structured_schema": self.structured_schema,
            "structured_name": self.structured_name,
            "force_structured": self.force_structured,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], max_output_tokens: int = 0) -> "ModelTurnRequest":
        return cls(
            instructions=str(value.get("instructions", value.get("system", ""))),
            messages=[dict(item) for item in list(value.get("messages", []) or [])],
            tools=[dict(item) for item in list(value.get("tools", []) or [])],
            max_output_tokens=int(value.get("max_output_tokens", max_output_tokens) or max_output_tokens),
            reasoning_effort=(str(value.get("reasoning_effort", "")).strip() or None),
            tool_choice=value.get("tool_choice", "auto"),
            parallel_tool_calls=bool(value.get("parallel_tool_calls", True)),
            structured_schema=value.get("structured_schema") if isinstance(value.get("structured_schema"), dict) else None,
            structured_name=str(value.get("structured_name", "structured_output")),
            force_structured=bool(value.get("force_structured", False)),
        )


@dataclass
class ModelTurnResponse:
    """Normalized provider response used by the agent loop."""

    text: str = ""
    tool_calls: list[AssistantToolCall] = field(default_factory=list)
    provider_output_items: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    refusal: str = ""
    parse_errors: list[str] = field(default_factory=list)
    structured_output: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "provider_output_items": list(self.provider_output_items),
            "raw_response": dict(self.raw_response),
            "usage": dict(self.usage),
            "refusal": self.refusal,
            "parse_errors": list(self.parse_errors),
            "structured_output": self.structured_output,
        }


class NativeToolCallingUnsupportedError(RuntimeError):
    """Raised when a configured provider cannot satisfy the native contract."""
