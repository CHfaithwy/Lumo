"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。
"""

import asyncio
import json
import time

import httpx

OPENAI_COMPATIBLE_USER_AGENT = "lumo/0.1"
PROMPT_CACHE_COMPATIBLE_HOSTS = ("openai.com", "right.codes", "codex2api.com")


def _retry_delay(attempt):
    return 0.5 * (attempt + 1)


def _http_status_error_message(prefix, exc):
    response = exc.response
    body = response.text if response is not None else str(exc)
    status_code = response.status_code if response is not None else "unknown"
    return f"{prefix} failed with HTTP {status_code}: {body}"


def _post_json_with_retries(url, payload, headers, timeout, http_error_prefix, connection_error_message, attempts=3):
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(attempts):
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.text, response.headers.get("Content-Type", "")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code >= 500 and attempt < attempts - 1:
                    time.sleep(_retry_delay(attempt))
                    continue
                raise RuntimeError(_http_status_error_message(http_error_prefix, exc)) from exc
            except httpx.RequestError as exc:
                if attempt < attempts - 1:
                    time.sleep(_retry_delay(attempt))
                    continue
                raise RuntimeError(connection_error_message) from exc


async def _post_json_with_retries_async(url, payload, headers, timeout, http_error_prefix, connection_error_message, attempts=3):
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(attempts):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.text, response.headers.get("Content-Type", "")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code >= 500 and attempt < attempts - 1:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise RuntimeError(_http_status_error_message(http_error_prefix, exc)) from exc
            except httpx.RequestError as exc:
                if attempt < attempts - 1:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise RuntimeError(connection_error_message) from exc


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    async def complete_async(self, prompt, max_new_tokens, **kwargs):
        return self.complete(prompt, max_new_tokens, **kwargs)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        # Ollama 当前不支持我们这里接入的 prompt cache 语义，
        # 所以 runtime 传下来的缓存参数会被忽略。
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
        )
        data = json.loads(body_text)
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")


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
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
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
    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # 当前只在明确支持 prompt cache 语义的后端上启用这条链路，
        # 避免对不支持的后端传一个“看起来统一、其实没意义”的伪参数。
        self.supports_prompt_cache = any(host in self.base_url for host in PROMPT_CACHE_COMPATIBLE_HOSTS)
        self.last_completion_metadata = {}

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
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
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

    def _connection_error_message(self):
        return (
            "Could not reach the OpenAI-compatible backend.\n"
            f"Base URL: {self.base_url}\n"
            f"Model: {self.model}"
        )

    def _parse_response(self, body_text, content_type, prompt_cache_key=None, prompt_cache_retention=None):
        # 有些兼容后端返回普通 JSON，有些返回 SSE。
        # 这里两种都接住，并尽量统一抽取文本和 usage/cache 元数据。
        """
        JSON：一次性返回一个完整 JSON 对象
        SSE：按很多行 data: {...} 流式返回事件  

        {
            "id": "resp_123",
            "output": [
                {
                "type": "message",
                "content": [
                    {
                    "type": "output_text",
                    "text": "你好！有什么我可以帮你处理的？"
                    }
                ]
                }
            ],
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 20,
                "total_tokens": 1220
            }
        }

        SSE
        data: {"type":"response.created","response":{"id":"resp_123"}}

        data: {"type":"response.output_text.delta","delta":"你"}

        data: {"type":"response.output_text.delta","delta":"好"}

        data: {"type":"response.output_text.delta","delta":"！"}

        data: {"type":"response.output_text.done","text":"你好！"}

        data: {"type":"response.completed","response":{"id":"resp_123","output_text":"你好！","usage":{"input_tokens":1200,"output_tokens":20,"total_tokens":1220}}}

        data: [DONE]
        """
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            metadata = {}
            if isinstance(response_data, dict) and response_data:
                # 这些元数据会一路传回 runtime，进入 trace 和 report，
                # 用来观察 prompt cache 是否真的命中。
                """
                    _extract_usage_cache_details
                    {
                        "input_tokens": 1200,
                        "output_tokens": 80,
                        "total_tokens": 1280,
                        "cached_tokens": 900,
                        "cache_hit": True,
                    }
                    """
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
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / cached_tokens 等元数据写进
          `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `Pico.ask()` 的模型调用阶段，是稳定前缀缓存复用链路真正
        落到 provider API 的地方。
        """
        self.last_completion_metadata = {}
        body_text, content_type = _post_json_with_retries(
            self.base_url + "/responses",
            self._build_payload(prompt, max_new_tokens, prompt_cache_key, prompt_cache_retention),
            self._headers(),
            self.timeout,
            "OpenAI-compatible request",
            self._connection_error_message(),
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
    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

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

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

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
        # 为了保持统一接口，runtime 仍然会传缓存参数进来；
        # 这里只是显式丢弃，因为当前 Anthropic-compatible 路径没有接缓存复用。
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        body_text, _ = _post_json_with_retries(
            self.base_url + "/messages",
            self._build_payload(prompt, max_new_tokens),
            self._headers(),
            self.timeout,
            "Anthropic-compatible request",
            self._connection_error_message(),
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
        )
        return self._parse_response(body_text)
