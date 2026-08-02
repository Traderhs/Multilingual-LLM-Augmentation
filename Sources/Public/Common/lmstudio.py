"""Reusable LM Studio OpenAI-compatible generation settings and request client."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


DEFAULT_LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL = "http://127.0.0.1:1234/v1/embeddings"
EndpointConfig = tuple[str, str, int]


class LmStudioRequestError(RuntimeError):
    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class LmStudioGenerationSettings:
    temperature: float | None = 0.8
    top_p: float | None = 0.9
    top_k: int | None = None
    max_tokens: int | None = 256
    repeat_penalty: float | None = 1.0
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: str | tuple[str, ...] | None = None
    logit_bias: Mapping[str, float] | None = None
    enable_thinking: bool | None = True
    stream: bool = True

    def request_parameters(self, seed: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": list(self.stop) if isinstance(self.stop, tuple) else self.stop,
            "logit_bias": dict(self.logit_bias) if self.logit_bias is not None else None,
            "enable_thinking": self.enable_thinking,
            "stream": self.stream,
            "seed": int(seed),
        }
        return {key: value for key, value in values.items() if value is not None}

    def report_parameters(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": list(self.stop) if isinstance(self.stop, tuple) else self.stop,
            "logit_bias": dict(self.logit_bias) if self.logit_bias is not None else None,
            "enable_thinking": self.enable_thinking,
            "stream": self.stream,
            "seed": "per_request",
        }


STANDARD_LMSTUDIO_GENERATION_SETTINGS = LmStudioGenerationSettings()


def parse_lmstudio_endpoints(
    endpoint_args: list[list[str]] | None,
    default_model: str,
    default_api_url: str = DEFAULT_LMSTUDIO_API_URL,
) -> list[EndpointConfig]:
    if not endpoint_args:
        return [(default_api_url, default_model, 1)]

    endpoints: list[EndpointConfig] = []
    for api_url, model, max_concurrency_text in endpoint_args:
        try:
            max_concurrency = int(max_concurrency_text)
        except ValueError as exc:
            raise ValueError(
                f"Endpoint concurrency must be an integer: {max_concurrency_text}"
            ) from exc
        if not api_url.strip():
            raise ValueError("Endpoint API URL must not be empty")
        if not model.strip():
            raise ValueError(f"Endpoint model must not be empty: {api_url}")
        if max_concurrency < 1:
            raise ValueError(
                f"Endpoint concurrency must be >= 1: {api_url} {max_concurrency}"
            )
        endpoints.append((api_url, model, max_concurrency))
    return endpoints


def _management_url(api_url: str, path: str) -> str:
    parsed = urlsplit(api_url)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _request_json(
    *,
    api_url: str,
    method: str,
    body: Mapping[str, Any] | None,
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    data = (
        json.dumps(dict(body), ensure_ascii=False).encode("utf-8")
        if body is not None
        else None
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace").strip()
        raise LmStudioRequestError(
            f"HTTP {exc.code}: {response_text or exc}", response_text or None
        ) from exc
    except urllib.error.URLError as exc:
        raise LmStudioRequestError(f"url_error: {exc}") from exc
    except TimeoutError as exc:
        raise LmStudioRequestError(
            f"timeout after {timeout_seconds}s: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise LmStudioRequestError(f"response_error: {exc}") from exc
    if not isinstance(payload, dict):
        raise LmStudioRequestError("response is not a JSON object")
    return payload


@contextmanager
def loaded_lmstudio_models(
    endpoints: Sequence[EndpointConfig],
    *,
    api_key: str | None = None,
    timeout_seconds: int = 600,
    log: Callable[[str], None] | None = None,
) -> Iterator[None]:
    """Load each endpoint model synchronously and always unload its instance."""
    loaded: list[tuple[str, str, str]] = []
    unique = list(dict.fromkeys((url, model) for url, model, _ in endpoints))
    try:
        for api_url, model in unique:
            if log:
                log(f"LM Studio model load start | model={model} | endpoint={api_url}")
            payload = _request_json(
                api_url=_management_url(api_url, "/api/v1/models/load"),
                method="POST",
                body={"model": model},
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            instance_id = payload.get("instance_id")
            if payload.get("status") != "loaded" or not isinstance(instance_id, str):
                raise LmStudioRequestError(f"model load did not report loaded: {payload}")
            loaded.append((api_url, model, instance_id))
            if log:
                log(
                    f"LM Studio model load complete | model={model} | "
                    f"instance_id={instance_id}"
                )
        yield
    finally:
        body_failed = sys.exc_info()[0] is not None
        unload_errors: list[Exception] = []
        for api_url, model, instance_id in reversed(loaded):
            if log:
                log(
                    f"LM Studio model unload start | model={model} | "
                    f"instance_id={instance_id}"
                )
            try:
                _request_json(
                    api_url=_management_url(api_url, "/api/v1/models/unload"),
                    method="POST",
                    body={"instance_id": instance_id},
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                unload_errors.append(exc)
                if log:
                    log(
                        f"LM Studio model unload failed | model={model} | "
                        f"instance_id={instance_id} | error={type(exc).__name__}: {exc}"
                    )
            else:
                if log:
                    log(
                        f"LM Studio model unload complete | model={model} | "
                        f"instance_id={instance_id}"
                    )
        if unload_errors and not body_failed:
            raise unload_errors[0]


def request_lmstudio_chat_content(
    *,
    api_url: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    settings: LmStudioGenerationSettings,
    seed: int,
    response_format: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 600,
) -> str:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        **settings.request_parameters(seed),
    }
    if response_format is not None:
        request_body["response_format"] = dict(response_format)

    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if settings.stream:
                content = _read_stream_content(response)
            else:
                content = _read_json_content(response)
    except LmStudioRequestError:
        raise
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace").strip()
        raise LmStudioRequestError(
            f"HTTP {exc.code}: {response_text or exc}", response_text or None
        ) from exc
    except urllib.error.URLError as exc:
        raise LmStudioRequestError(f"url_error: {exc}") from exc
    except TimeoutError as exc:
        raise LmStudioRequestError(
            f"timeout after {timeout_seconds}s: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeError, KeyError, TypeError) as exc:
        raise LmStudioRequestError(f"response_error: {exc}") from exc

    if not content.strip():
        raise LmStudioRequestError("empty_response", content)
    return content


def request_lmstudio_embeddings(
    *,
    api_url: str,
    model: str,
    inputs: Sequence[str],
    api_key: str | None = None,
    timeout_seconds: int = 600,
) -> list[list[float]]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not inputs:
        return []
    request_body = {"model": model, "input": list(inputs)}
    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace").strip()
        raise LmStudioRequestError(
            f"HTTP {exc.code}: {response_text or exc}", response_text or None
        ) from exc
    except urllib.error.URLError as exc:
        raise LmStudioRequestError(f"url_error: {exc}") from exc
    except TimeoutError as exc:
        raise LmStudioRequestError(
            f"timeout after {timeout_seconds}s: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise LmStudioRequestError(f"response_error: {exc}") from exc

    rows = payload.get("data")
    if not isinstance(rows, list):
        raise LmStudioRequestError("embedding response data is not a list")
    indexed: dict[int, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("index"), int):
            raise LmStudioRequestError("embedding response row has invalid index")
        vector = row.get("embedding")
        if not isinstance(vector, list) or not all(
            isinstance(value, (int, float)) for value in vector
        ):
            raise LmStudioRequestError("embedding response row has invalid vector")
        if row["index"] in indexed:
            raise LmStudioRequestError(
                f"embedding response has duplicate index {row['index']}"
            )
        indexed[row["index"]] = [float(value) for value in vector]
    expected_indices = set(range(len(inputs)))
    if set(indexed) != expected_indices:
        raise LmStudioRequestError(
            f"embedding response indices {sorted(indexed)} != {sorted(expected_indices)}"
        )
    return [indexed[index] for index in range(len(inputs))]


def _read_stream_content(response: Any) -> str:
    content_parts: list[str] = []
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
    except Exception as exc:
        partial = "".join(content_parts)
        raise LmStudioRequestError(f"stream_error: {exc}", partial or None) from exc
    return "".join(content_parts)


def _read_json_content(response: Any) -> str:
    payload = json.loads(response.read().decode("utf-8", errors="replace"))
    content = payload.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise LmStudioRequestError("response message.content is not a string")
    return content
