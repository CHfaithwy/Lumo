

import asyncio
import json
import random
import time
from email.utils import parsedate_to_datetime

import httpx

from ..archive_context import ARCHIVE_TOOL_NAME, partition_archive_calls
from ..model_protocol import (
    ArchiveSummaryEvent,
    AssistantToolCall,
    ContextWindowExceededError,
    ModelTurnRequest,
    ModelTurnResponse,
    NativeToolCallingUnsupportedError,
    ProviderRequestError,
)

OPENAI_COMPATIBLE_USER_AGENT = "lumo/0.1"
MAX_RETRY_AFTER_SECONDS = 120.0
MAX_LOCAL_RETRY_DELAY_SECONDS = 30.0
LOCAL_RETRY_JITTER_RATIO = 0.25


def _retry_delay(attempt, retry_after_seconds=None):
    if retry_after_seconds is not None:
        return min(MAX_RETRY_AFTER_SECONDS, max(0.0, float(retry_after_seconds)))
    base_delay = min(MAX_LOCAL_RETRY_DELAY_SECONDS, float(2**attempt))
    return min(MAX_LOCAL_RETRY_DELAY_SECONDS, base_delay * (1.0 + random.random() * LOCAL_RETRY_JITTER_RATIO))


def _format_retry_delay(delay_seconds):
    delay_seconds = float(delay_seconds)
    if delay_seconds >= 10 or delay_seconds.is_integer():
        return f"{round(delay_seconds)}s"
    return f"{delay_seconds:.1f}s"


def _report_retry(retry_reporter, status_code, attempt, attempts, delay_seconds):
    if retry_reporter is None:
        return
    if status_code is None:
        reason = "connection error"
    elif status_code == 429:
        reason = "HTTP 429 (rate limited)"
    else:
        reason = f"HTTP {status_code}"
    try:
        retry_reporter(f"{reason}; retry {attempt + 1}/{attempts - 1} in {_format_retry_delay(delay_seconds)}")
    except Exception:
        pass


def _parse_retry_after(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                return None
            seconds = parsed.timestamp() - time.time()
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    return max(0.0, seconds) if seconds >= 0 else None


def _retry_after_from_body(body):
    try:
        payload = json.loads(str(body or ""))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("retry_after")
    if value is None and isinstance(payload.get("error"), dict):
        value = payload["error"].get("retry_after")
    return _parse_retry_after(value)


def _response_retry_after(response):
    if response is None:
        return None
    header_delay = _parse_retry_after(response.headers.get("Retry-After"))
    return header_delay if header_delay is not None else _retry_after_from_body(response.text)


def _is_context_window_error(status_code, body):
    if status_code not in {400, 413}:
        return False
    text = str(body or "").lower()
    if any(
        marker in text
        for marker in (
            "prompt-too-long",
            "prompt too long",
            "context_length_exceeded",
            "context_window_exceeded",
            "model_context_window_exceeded",
        )
    ):
        return True
    overflow_terms = ("too long", "too large", "exceed", "exceeded", "maximum", "limit", "window", "length")
    if ("context" in text or "prompt" in text) and any(term in text for term in overflow_terms):
        return True
    return "input" in text and "token" in text and any(term in text for term in overflow_terms)


def _is_retryable_status(status_code):
    return status_code == 429 or status_code >= 500


def _http_status_error_message(prefix, exc):
    response = exc.response
    body = response.text if response is not None else str(exc)
    status_code = response.status_code if response is not None else "unknown"
    return f"{prefix} failed with HTTP {status_code}: {body}"


def _provider_http_error(prefix, exc):
    response = exc.response
    body = response.text if response is not None else str(exc)
    status_code = response.status_code if response is not None else None
    error_type = ContextWindowExceededError if _is_context_window_error(status_code, body) else ProviderRequestError
    return error_type(
        _http_status_error_message(prefix, exc),
        status_code=status_code,
        response_body=body,
        retry_after_seconds=_response_retry_after(response),
    )


def _provider_connection_error(message):
    return ProviderRequestError(message, status_code=None, response_body="", retry_after_seconds=None)


def _is_native_tool_capability_error(error):
    text = str(error).lower()
    return any(token in text for token in ("tool", "function", "strict", "schema", "responses"))


def _optional_reasoning_effort(value):
    if value is None:
        return None
    return str(value).strip() or None


def _post_json_with_retries(
    url,
    payload,
    headers,
    timeout,
    http_error_prefix,
    connection_error_message,
    attempts=3,
    retry_reporter=None,
):
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(attempts):
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.text, response.headers.get("Content-Type", "")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if _is_retryable_status(status_code) and attempt < attempts - 1:
                    delay_seconds = _retry_delay(attempt, _response_retry_after(exc.response))
                    _report_retry(retry_reporter, status_code, attempt, attempts, delay_seconds)
                    time.sleep(delay_seconds)
                    continue
                raise _provider_http_error(http_error_prefix, exc) from exc
            except httpx.RequestError as exc:
                if attempt < attempts - 1:
                    delay_seconds = _retry_delay(attempt)
                    _report_retry(retry_reporter, None, attempt, attempts, delay_seconds)
                    time.sleep(delay_seconds)
                    continue
                raise _provider_connection_error(connection_error_message) from exc


async def _post_json_with_retries_async(
    url,
    payload,
    headers,
    timeout,
    http_error_prefix,
    connection_error_message,
    attempts=3,
    retry_reporter=None,
):
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(attempts):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.text, response.headers.get("Content-Type", "")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if _is_retryable_status(status_code) and attempt < attempts - 1:
                    delay_seconds = _retry_delay(attempt, _response_retry_after(exc.response))
                    _report_retry(retry_reporter, status_code, attempt, attempts, delay_seconds)
                    await asyncio.sleep(delay_seconds)
                    continue
                raise _provider_http_error(http_error_prefix, exc) from exc
            except httpx.RequestError as exc:
                if attempt < attempts - 1:
                    delay_seconds = _retry_delay(attempt)
                    _report_retry(retry_reporter, None, attempt, attempts, delay_seconds)
                    await asyncio.sleep(delay_seconds)
                    continue
                raise _provider_connection_error(connection_error_message) from exc


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.message_requests = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}
        self.last_request_payload = {}
        self.last_response_payload = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    async def complete_async(self, prompt, max_new_tokens, **kwargs):
        return self.complete(prompt, max_new_tokens, **kwargs)

    def complete_messages(self, system, messages, max_new_tokens, **kwargs):
        request = {
            "system": str(system or ""),
            "messages": [dict(item) for item in list(messages or [])],
        }
        self.message_requests.append(request)
        rendered = _render_message_request(system, messages)
        return self.complete(rendered, max_new_tokens, **kwargs)

    async def complete_messages_async(self, system, messages, max_new_tokens, **kwargs):
        return self.complete_messages(system, messages, max_new_tokens, **kwargs)

    async def complete_turn_async(self, turn_request, **kwargs):
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        self.last_request_payload = request.to_dict()
        raw = self.complete(_render_message_request(request.instructions, request.messages), request.max_output_tokens, **kwargs)
        if isinstance(raw, ModelTurnResponse):
            response = raw
        elif isinstance(raw, dict):
            real_calls, archive_events = partition_archive_calls(
                [AssistantToolCall.from_dict(item) for item in list(raw.get("tool_calls", []) or [])]
            )
            archive_events.extend(
                ArchiveSummaryEvent.from_dict(item)
                for item in list(raw.get("archive_events", []) or [])
                if isinstance(item, dict)
            )
            response = ModelTurnResponse(
                text=str(raw.get("text", "")),
                tool_calls=real_calls,
                archive_events=archive_events,
                raw_response=dict(raw),
                structured_output=(
                    dict(raw)
                    if request.structured_schema and "text" not in raw and "tool_calls" not in raw
                    else None
                ),
            )
        else:
            response = ModelTurnResponse(text=str(raw), raw_response={"fake_text": str(raw)})
        self.last_response_payload = response.to_dict()
        return response

    async def complete_structured_async(self, request, schema, **kwargs):
        response = await self.complete_turn_async(
            ModelTurnRequest(
                instructions=str(request.get("instructions", "")) if isinstance(request, dict) else "",
                messages=list(request.get("messages", []) or []) if isinstance(request, dict) else [{"role": "user", "content": str(request)}],
                max_output_tokens=int(kwargs.get("max_new_tokens", 256)),
                structured_schema=dict(schema),
                force_structured=True,
                reasoning_effort=_optional_reasoning_effort(kwargs.get("reasoning_effort")),
            )
        )
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return response.structured_output or {}


def _normalize_message_request(system, messages):
    normalized = []
    if str(system or "").strip():
        normalized.append({"role": "system", "content": str(system).strip()})
    for item in list(messages or []):
        role = str((item or {}).get("role", "user")).strip().lower()
        if role not in {"user", "assistant"}:
            role = "user"
        content = str((item or {}).get("content", ""))
        if content.strip():
            normalized.append({"role": role, "content": content})
    return normalized


def _render_message_request(system, messages):
    return "\n\n".join(
        f"[{item['role']}]\n{item['content']}"
        for item in _normalize_message_request(system, messages)
    )


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout, retry_reporter=None):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retry_reporter = retry_reporter
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):


        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        body_text, _ = _post_json_with_retries(
            self.host + "/api/generate",
            payload,
            {"Content-Type": "application/json"},
            self.timeout,
            "Ollama request",
            (
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ),
            retry_reporter=self.retry_reporter,
        )
        data = json.loads(body_text)

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    async def complete_async(self, prompt, max_new_tokens, **kwargs):
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        body_text, _ = await _post_json_with_retries_async(
            self.host + "/api/generate",
            payload,
            {"Content-Type": "application/json"},
            self.timeout,
            "Ollama request",
            (
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ),
            retry_reporter=self.retry_reporter,
        )
        data = json.loads(body_text)
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def _build_messages_payload(self, system, messages, max_new_tokens):
        return {
            "model": self.model,
            "messages": _normalize_message_request(system, messages),
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

    def _build_turn_payload(self, turn_request):
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        messages = []
        if request.instructions.strip():
            messages.append({"role": "system", "content": request.instructions})
        for message in list(request.messages or []):
            role = str(message.get("role", "user"))
            if role == "tool":
                output = message.get("result", {}) if isinstance(message.get("result", {}), dict) else {}
                output = dict(output)
                output["source_call_id"] = str(message.get("call_id", ""))
                messages.append({"role": "tool", "tool_name": str(message.get("name", "")), "content": json.dumps(output, ensure_ascii=False)})
                continue
            if role == "skill_context":
                messages.append({"role": "user", "content": str(message.get("content", ""))})
                continue
            if role == "archive_control":
                messages.append({"role": "user", "content": str(message.get("content", ""))})
                continue
            item = {"role": role if role in {"user", "assistant"} else "user", "content": str(message.get("content", ""))}
            if role == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = [
                    {"function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in list(message.get("tool_calls", []) or [])
                ]
            messages.append(item)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": request.max_output_tokens, "temperature": self.temperature, "top_p": self.top_p},
        }
        if request.tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"]}}
                for tool in request.tools
            ]
        if request.structured_schema:
            payload["format"] = request.structured_schema
        return payload

    async def complete_turn_async(self, turn_request, **kwargs):
        del kwargs
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        payload = self._build_turn_payload(request)
        self.last_request_payload = payload
        body_text, _ = await _post_json_with_retries_async(
            self.host + "/api/chat", payload, {"Content-Type": "application/json"}, self.timeout,
            "Ollama request", f"Could not reach Ollama.\nHost: {self.host}\nModel: {self.model}",
            retry_reporter=self.retry_reporter,
        )
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama error: backend returned non-JSON content") from exc
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        message = data.get("message") or {}
        calls = []
        for index, raw_call in enumerate(list(message.get("tool_calls", []) or [])):
            function = raw_call.get("function", {}) if isinstance(raw_call, dict) else {}
            arguments = function.get("arguments", {}) if isinstance(function.get("arguments", {}), dict) else None
            calls.append(AssistantToolCall(call_id=f"ollama_{len(request.messages)}_{index}", name=str(function.get("name", "")), arguments=arguments, raw_arguments=json.dumps(function.get("arguments", {}), ensure_ascii=False), error="" if arguments is not None else "invalid_arguments"))
        calls, archive_events = partition_archive_calls(calls)
        response = ModelTurnResponse(
            text=str(message.get("content", "")), tool_calls=calls, archive_events=archive_events, raw_response=data,
            usage=dict(data.get("eval_count") and {"output_tokens": data.get("eval_count")} or {}),
        )
        if request.structured_schema and response.text:
            try:
                response.structured_output = json.loads(response.text)
            except json.JSONDecodeError:
                response.parse_errors.append("structured_output_not_json")
        self.last_response_payload = response.to_dict()
        self.last_completion_metadata = dict(response.usage)
        return response

    async def complete_structured_async(self, request, schema, **kwargs):
        response = await self.complete_turn_async(ModelTurnRequest(
            instructions=str(request.get("instructions", "")), messages=list(request.get("messages", []) or []),
            max_output_tokens=int(kwargs.get("max_new_tokens", 256)), structured_schema=dict(schema), force_structured=True,
        ))
        return response.structured_output or {}

    def complete_messages(self, system, messages, max_new_tokens, **kwargs):
        del kwargs
        self.last_completion_metadata = {}
        body_text, _ = _post_json_with_retries(
            self.host + "/api/chat",
            self._build_messages_payload(system, messages, max_new_tokens),
            {"Content-Type": "application/json"},
            self.timeout,
            "Ollama request",
            f"Could not reach Ollama.\nHost: {self.host}\nModel: {self.model}",
            retry_reporter=self.retry_reporter,
        )
        data = json.loads(body_text)
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str((data.get("message") or {}).get("content", ""))

    async def complete_messages_async(self, system, messages, max_new_tokens, **kwargs):
        del kwargs
        self.last_completion_metadata = {}
        body_text, _ = await _post_json_with_retries_async(
            self.host + "/api/chat",
            self._build_messages_payload(system, messages, max_new_tokens),
            {"Content-Type": "application/json"},
            self.timeout,
            "Ollama request",
            f"Could not reach Ollama.\nHost: {self.host}\nModel: {self.model}",
            retry_reporter=self.retry_reporter,
        )
        data = json.loads(body_text)
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str((data.get("message") or {}).get("content", ""))


def _normalize_versioned_base_url(base_url):
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_text(data):
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""


def _extract_openai_text_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        text = _extract_openai_text(event)
        if text:
            return text
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data):


    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


class OpenAICompatibleModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout, reasoning_effort=None, retry_reporter=None):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.retry_reporter = retry_reporter


        self.supports_prompt_cache = True
        self.last_completion_metadata = {}
        self.last_request_payload = {}
        self.last_response_payload = {}

    def _build_payload(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}


        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        return payload

    def _build_messages_payload(self, system, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": item["role"],
                    "content": [{"type": "input_text", "text": item["content"]}],
                }
                for item in _normalize_message_request(system, messages)
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        return payload

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _openai_input_items(messages):
        items = []
        for message in list(messages or []):
            role = str((message or {}).get("role", "user")).strip().lower()
            if role == "tool":
                result = message.get("result", {}) if isinstance(message.get("result", {}), dict) else {}
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("call_id", "")),
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
                continue
            if role == "skill_context":
                items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": str(message.get("content", ""))}],
                    }
                )
                continue
            if role == "archive_control":
                items.append(
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": str(message.get("content", ""))}],
                    }
                )
                continue
            if role == "assistant" and isinstance(message.get("provider_output_items"), list) and message.get("provider_output_items"):
                items.extend(dict(item) for item in message["provider_output_items"] if isinstance(item, dict))
                continue
            if role == "assistant" and message.get("tool_calls"):
                if str(message.get("content", "")).strip():
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": str(message["content"])}]})
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("call_id", "")),
                        "name": str(call.get("name", "")),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    }
                    for call in list(message.get("tool_calls", []) or [])
                )
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                items.append({"role": role if role in {"user", "assistant", "developer"} else "user", "content": content})
            elif str(content).strip():
                items.append(
                    {
                        "role": role if role in {"user", "assistant", "developer"} else "user",
                        "content": [{"type": "input_text", "text": str(content)}],
                    }
                )
        return items

    @staticmethod
    def _openai_tools(tools):
        return [
            {
                "type": "function",
                "name": str(tool["name"]),
                "description": str(tool.get("description", "")),
                "parameters": dict(tool["parameters"]),
                "strict": True,
            }
            for tool in list(tools or [])
        ]

    def _build_turn_payload(self, turn_request, prompt_cache_key=None, prompt_cache_retention=None):
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        payload = {
            "model": self.model,
            "instructions": request.instructions,
            "input": self._openai_input_items(request.messages),
            "max_output_tokens": request.max_output_tokens,
            "stream": False,
            "store": False,
        }
        if request.tools:
            payload["tools"] = self._openai_tools(request.tools)
            payload["tool_choice"] = request.tool_choice
            payload["parallel_tool_calls"] = bool(request.parallel_tool_calls)
        if request.structured_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.structured_name,
                    "schema": request.structured_schema,
                    "strict": True,
                }
            }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        reasoning_effort = request.reasoning_effort or self.reasoning_effort
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        return payload

    @staticmethod
    def _parse_openai_turn(data):
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI-compatible error: response must be a JSON object")
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        output_items = [dict(item) for item in list(data.get("output", []) or []) if isinstance(item, dict)]
        text_parts = []
        calls = []
        errors = []
        refusal = ""
        for item in output_items:
            if item.get("type") == "function_call":
                raw_arguments = str(item.get("arguments", ""))
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError as exc:
                    arguments = None
                    errors.append(f"invalid function arguments for {item.get('name', '')}: {exc}")
                calls.append(
                    AssistantToolCall(
                        call_id=str(item.get("call_id", "")),
                        name=str(item.get("name", "")),
                        arguments=arguments if isinstance(arguments, dict) else None,
                        raw_arguments=raw_arguments,
                        error="" if isinstance(arguments, dict) else "invalid_arguments_json",
                    )
                )
                continue
            for content in list(item.get("content", []) or []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and str(content.get("text", "")):
                    text_parts.append(str(content["text"]))
                elif content.get("type") == "refusal":
                    refusal = str(content.get("refusal", content.get("text", "")))
        if not text_parts and data.get("output_text"):
            text_parts.append(str(data["output_text"]))
        calls, archive_events = partition_archive_calls(calls)
        replay_output_items = [
            item
            for item in output_items
            if not (item.get("type") == "function_call" and item.get("name") == ARCHIVE_TOOL_NAME)
        ]
        return ModelTurnResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            archive_events=archive_events,
            provider_output_items=replay_output_items,
            raw_response=data,
            usage=_extract_usage_cache_details(data),
            refusal=refusal,
            parse_errors=errors,
        )

    @staticmethod
    def _openai_response_artifact(data):

        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "output": list(data.get("output", []) or []),
            "usage": dict(data.get("usage", {}) or {}),
            "error": data.get("error"),
        }

    async def complete_turn_async(self, turn_request, prompt_cache_key=None, prompt_cache_retention=None):
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        payload = self._build_turn_payload(request, prompt_cache_key, prompt_cache_retention)
        self.last_request_payload = payload
        try:
            body_text, content_type = await _post_json_with_retries_async(
                self.base_url + "/responses", payload, self._headers(), self.timeout,
                "OpenAI-compatible request", self._connection_error_message(),
                retry_reporter=self.retry_reporter,
            )
        except RuntimeError as exc:
            if request.tools and _is_native_tool_capability_error(exc):
                raise NativeToolCallingUnsupportedError(f"native_tool_calling_unsupported: {exc}") from exc
            raise
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            _, data = _extract_openai_response_from_sse(body_text)
        else:
            try:
                data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("OpenAI-compatible error: backend returned non-JSON content") from exc
        try:
            response = self._parse_openai_turn(data)
        except RuntimeError as exc:
            if request.tools and _is_native_tool_capability_error(exc):
                raise NativeToolCallingUnsupportedError(f"native_tool_calling_unsupported: {exc}") from exc
            raise
        self.last_response_payload = self._openai_response_artifact(data)
        self.last_completion_metadata = dict(response.usage)
        if request.structured_schema and response.text:
            try:
                response.structured_output = json.loads(response.text)
            except json.JSONDecodeError:
                response.parse_errors.append("structured_output_not_json")
        return response

    async def complete_structured_async(self, request, schema, **kwargs):
        turn = ModelTurnRequest(
            instructions=str(request.get("instructions", "")),
            messages=list(request.get("messages", []) or []),
            max_output_tokens=int(kwargs.get("max_new_tokens", 256)),
            structured_schema=dict(schema),
            structured_name=str(kwargs.get("name", "structured_output")),
            force_structured=True,
            reasoning_effort=_optional_reasoning_effort(kwargs.get("reasoning_effort")),
        )
        response = await self.complete_turn_async(turn)
        return response.structured_output or {}

    def _connection_error_message(self):
        return (
            "Could not reach the OpenAI-compatible backend.\n"
            f"Base URL: {self.base_url}\n"
            f"Model: {self.model}"
        )

    def _parse_response(self, body_text, content_type, prompt_cache_key=None, prompt_cache_retention=None):



        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            metadata = {}
            if isinstance(response_data, dict) and response_data:



                metadata = {
                    "prompt_cache_supported": self.supports_prompt_cache,
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,

                    **_extract_usage_cache_details(response_data),
                }
            if text:
                return text, metadata
            raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_extract_usage_cache_details(data),
        }
        return _extract_openai_text(data), metadata

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):

        self.last_completion_metadata = {}
        body_text, content_type = _post_json_with_retries(
            self.base_url + "/responses",
            self._build_payload(prompt, max_new_tokens, prompt_cache_key, prompt_cache_retention),
            self._headers(),
            self.timeout,
            "OpenAI-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        text, metadata = self._parse_response(body_text, content_type, prompt_cache_key, prompt_cache_retention)
        self.last_completion_metadata = metadata
        return text

    async def complete_async(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        self.last_completion_metadata = {}
        body_text, content_type = await _post_json_with_retries_async(
            self.base_url + "/responses",
            self._build_payload(prompt, max_new_tokens, prompt_cache_key, prompt_cache_retention),
            self._headers(),
            self.timeout,
            "OpenAI-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        text, metadata = self._parse_response(body_text, content_type, prompt_cache_key, prompt_cache_retention)
        self.last_completion_metadata = metadata
        return text

    def complete_messages(self, system, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        self.last_completion_metadata = {}
        body_text, content_type = _post_json_with_retries(
            self.base_url + "/responses",
            self._build_messages_payload(
                system, messages, max_new_tokens, prompt_cache_key, prompt_cache_retention
            ),
            self._headers(),
            self.timeout,
            "OpenAI-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        text, metadata = self._parse_response(body_text, content_type, prompt_cache_key, prompt_cache_retention)
        self.last_completion_metadata = metadata
        return text

    async def complete_messages_async(self, system, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        self.last_completion_metadata = {}
        body_text, content_type = await _post_json_with_retries_async(
            self.base_url + "/responses",
            self._build_messages_payload(
                system, messages, max_new_tokens, prompt_cache_key, prompt_cache_retention
            ),
            self._headers(),
            self.timeout,
            "OpenAI-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        text, metadata = self._parse_response(body_text, content_type, prompt_cache_key, prompt_cache_retention)
        self.last_completion_metadata = metadata
        return text


def _extract_anthropic_text(data):
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                return text
    return ""


class AnthropicCompatibleModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout, retry_reporter=None):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.retry_reporter = retry_reporter
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}
        self.last_request_payload = {}
        self.last_response_payload = {}

    def _build_payload(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def _build_messages_payload(self, system, messages, max_new_tokens):
        normalized = _normalize_message_request("", messages)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": item["role"],
                    "content": [{"type": "text", "text": item["content"]}],
                }
                for item in normalized
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if str(system or "").strip():
            payload["system"] = [{"type": "text", "text": str(system).strip()}]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _anthropic_messages(messages):
        rendered = []
        pending_tool_blocks = []

        def flush_tool_blocks():
            nonlocal pending_tool_blocks
            if pending_tool_blocks:
                rendered.append({"role": "user", "content": pending_tool_blocks})
                pending_tool_blocks = []

        for message in list(messages or []):
            role = str(message.get("role", "user"))
            if role == "tool":
                output = message.get("result", {}) if isinstance(message.get("result", {}), dict) else {}
                pending_tool_blocks.append({"type": "tool_result", "tool_use_id": str(message.get("call_id", "")), "content": json.dumps(output, ensure_ascii=False)})
                continue
            if role == "skill_context":
                block = {"type": "text", "text": str(message.get("content", ""))}
                if pending_tool_blocks:
                    pending_tool_blocks.append(block)
                else:
                    rendered.append({"role": "user", "content": [block]})
                continue
            if role == "archive_control":
                block = {"type": "text", "text": str(message.get("content", ""))}
                if pending_tool_blocks:
                    pending_tool_blocks.append(block)
                else:
                    rendered.append({"role": "user", "content": [block]})
                continue
            flush_tool_blocks()
            if role == "assistant" and message.get("tool_calls"):
                content = []
                if str(message.get("content", "")).strip():
                    content.append({"type": "text", "text": str(message["content"])})
                content.extend({"type": "tool_use", "id": call["call_id"], "name": call["name"], "input": call["arguments"]} for call in list(message.get("tool_calls", []) or []))
                rendered.append({"role": "assistant", "content": content})
                continue
            content = message.get("content", "")
            if str(content).strip():
                rendered.append({"role": role if role in {"user", "assistant"} else "user", "content": [{"type": "text", "text": str(content)}]})
        flush_tool_blocks()
        return rendered

    def _build_turn_payload(self, turn_request):
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        payload = {
            "model": self.model,
            "messages": self._anthropic_messages(request.messages),
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.instructions.strip():
            payload["system"] = [{"type": "text", "text": request.instructions}]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if request.tools:
            payload["tools"] = [{"name": tool["name"], "description": tool.get("description", ""), "input_schema": tool["parameters"], "strict": True} for tool in request.tools]
            payload["tool_choice"] = request.tool_choice if isinstance(request.tool_choice, dict) else {"type": str(request.tool_choice or "auto")}
        return payload

    @staticmethod
    def _parse_anthropic_turn(data):
        if not isinstance(data, dict):
            raise RuntimeError("Anthropic-compatible error: response must be a JSON object")
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        text_parts = []
        calls = []
        for item in list(data.get("content", []) or []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif item.get("type") == "tool_use":
                arguments = item.get("input", {})
                calls.append(AssistantToolCall(call_id=str(item.get("id", "")), name=str(item.get("name", "")), arguments=dict(arguments) if isinstance(arguments, dict) else None, raw_arguments=json.dumps(arguments, ensure_ascii=False), error="" if isinstance(arguments, dict) else "invalid_arguments"))
        calls, archive_events = partition_archive_calls(calls)
        provider_output_items = [
            dict(item)
            for item in data.get("content", [])
            if isinstance(item, dict)
            and not (item.get("type") == "tool_use" and item.get("name") == ARCHIVE_TOOL_NAME)
        ]
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        return ModelTurnResponse(text="\n".join(text_parts).strip(), tool_calls=calls, archive_events=archive_events, provider_output_items=provider_output_items, raw_response=data, usage={"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")})

    async def complete_turn_async(self, turn_request, **kwargs):
        del kwargs
        request = turn_request if isinstance(turn_request, ModelTurnRequest) else ModelTurnRequest.from_dict(turn_request)
        payload = self._build_turn_payload(request)
        self.last_request_payload = payload
        try:
            body_text, _ = await _post_json_with_retries_async(
                self.base_url + "/messages",
                payload,
                self._headers(),
                self.timeout,
                "Anthropic-compatible request",
                self._connection_error_message(),
                retry_reporter=self.retry_reporter,
            )
        except RuntimeError as exc:
            if request.tools and _is_native_tool_capability_error(exc):
                raise NativeToolCallingUnsupportedError(f"native_tool_calling_unsupported: {exc}") from exc
            raise
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Anthropic-compatible error: backend returned non-JSON content") from exc
        try:
            response = self._parse_anthropic_turn(data)
        except RuntimeError as exc:
            if request.tools and _is_native_tool_capability_error(exc):
                raise NativeToolCallingUnsupportedError(f"native_tool_calling_unsupported: {exc}") from exc
            raise
        self.last_response_payload = response.to_dict()
        self.last_completion_metadata = dict(response.usage)
        return response

    async def complete_structured_async(self, request, schema, **kwargs):
        schema_tool = {"name": "structured_output", "description": "Return the required structured result.", "parameters": dict(schema)}
        turn = ModelTurnRequest(instructions=str(request.get("instructions", "")), messages=list(request.get("messages", []) or []), tools=[schema_tool], max_output_tokens=int(kwargs.get("max_new_tokens", 256)), tool_choice={"type": "tool", "name": "structured_output"}, force_structured=True)
        response = await self.complete_turn_async(turn)
        if not response.tool_calls:
            return {}
        return dict(response.tool_calls[0].arguments or {})

    def _connection_error_message(self):
        return (
            "Could not reach the Anthropic-compatible backend.\n"
            f"Base URL: {self.base_url}\n"
            f"Model: {self.model}"
        )

    def _parse_response(self, body_text):
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Anthropic-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        text = _extract_anthropic_text(data)
        if text:
            return text
        raise RuntimeError("Anthropic-compatible error: could not extract text from response")

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):


        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        body_text, _ = _post_json_with_retries(
            self.base_url + "/messages",
            self._build_payload(prompt, max_new_tokens),
            self._headers(),
            self.timeout,
            "Anthropic-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        return self._parse_response(body_text)

    async def complete_async(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        body_text, _ = await _post_json_with_retries_async(
            self.base_url + "/messages",
            self._build_payload(prompt, max_new_tokens),
            self._headers(),
            self.timeout,
            "Anthropic-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        return self._parse_response(body_text)

    def complete_messages(self, system, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        body_text, _ = _post_json_with_retries(
            self.base_url + "/messages",
            self._build_messages_payload(system, messages, max_new_tokens),
            self._headers(),
            self.timeout,
            "Anthropic-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        return self._parse_response(body_text)

    async def complete_messages_async(self, system, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        body_text, _ = await _post_json_with_retries_async(
            self.base_url + "/messages",
            self._build_messages_payload(system, messages, max_new_tokens),
            self._headers(),
            self.timeout,
            "Anthropic-compatible request",
            self._connection_error_message(),
            retry_reporter=self.retry_reporter,
        )
        return self._parse_response(body_text)
