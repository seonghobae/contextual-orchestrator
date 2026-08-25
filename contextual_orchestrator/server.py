"""HTTP server exposing chat, admin, governance, and evaluation endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import secrets
import struct
import threading
import time
import urllib.parse
from typing import Any, Callable
import uuid

from .admin import ADMIN_HTML, ADMIN_TRANSLATIONS
from .api_contract import OPENAPI_SPEC
from .cost_ledger import ATTRIBUTION_DIMENSIONS, dimension_catalog
from .cost_router import CostRoutingCoordinator
from .batch_routing import BatchRequest
from .orchestrator import (
    BudgetExceededError,
    MAX_LOCAL_CONCURRENCY,
    TaskOrchestrator,
    _is_local_provider_url,
    _new_chat_completion_id,
    chat_completion_chunks,
    chat_completion_response,
    text_completion_response,
    redact_value,
    sse_stream_body,
)
from .tool_fallback import ToolFallbackStoppedError

# OpenAI request params forwarded verbatim to the provider on passthrough.
OPENAI_PASSTHROUGH_PARAM_KEYS = {
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "n", "stop",
    "seed", "presence_penalty", "frequency_penalty", "logit_bias", "logprobs",
    "top_logprobs", "user", "metadata", "parallel_tool_calls", "reasoning_effort",
    "response_format", "tools", "tool_choice", "functions", "function_call",
    "modalities", "prediction", "store", "service_tier", "stream_options",
    # Chat-era surfaces accepted only for explicit unsupported errors.
    "audio", "web_search_options",
    # Modern OpenAI SDK control fields — accepted only for named unsupported errors.
    "prompt_cache_key", "safety_identifier", "verbosity", "prompt_cache_retention",
    # Responses-style reasoning object on chat — named unsupported (not effort string).
    "reasoning",
    # Async background mode — not supported on this gateway.
    "background",
    "include",
    # Assistants-style tool_resources — named unsupported (not unknown_fields).
    "tool_resources",
}
# Provider features the multi-agent verifier cannot merge -> single-agent passthrough.
PASSTHROUGH_TRIGGER_KEYS = {"response_format", "tools", "tool_choice", "functions", "function_call"}
ALLOWED_CHAT_KEYS = {
    "model", "messages", "orchestration", "orchestration_mode", "mode",
    "include_orchestration_trace", "stream", "attribution", "routing",
    # Tool-loop budget — accepted only for named unsupported error (no multi-step tool loop).
    "max_tool_calls",
} | OPENAI_PASSTHROUGH_PARAM_KEYS
# Responses API body keys (`input` replaces `messages`).
ALLOWED_RESPONSES_KEYS = {
    "model", "input", "instructions", "stream", "metadata", "reasoning",
    "prompt_cache_key", "client_metadata",
    # OpenAI Responses native output budget (not max_tokens on this surface).
    "max_output_tokens",
    # Tool-loop budget — accepted only for explicit unsupported error (no multi-step tool loop).
    "max_tool_calls",
    # Gateway cost/routing control plane (stripped before provider passthrough).
    "attribution", "routing",
    # previous_response_id / conversation / truncation / include fail closed
    # with named unsupported errors. Official text.format is validated
    # (omit-real optionals), not rejected wholesale.
    "previous_response_id", "conversation", "truncation", "include", "text",
} | OPENAI_PASSTHROUGH_PARAM_KEYS
ALLOWED_BATCH_KEYS = {"requests", "attribution", "routing", "model"}
ALLOWED_EMBEDDINGS_BATCH_KEYS = {"model", "input", "inputs", "endpoint", "metadata", "attribution", "user", "encoding_format", "dimensions", "routing"}
ALLOWED_EMBEDDINGS_KEYS = {
    "model", "input", "encoding_format", "dimensions", "user", "metadata", "attribution", "routing",
}
ALLOWED_COMPLETIONS_KEYS = {
    "model", "prompt", "stream", "stream_options", "echo", "suffix", "best_of",
    "logprobs", "top_logprobs", "n", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "user", "seed",
    "presence_penalty", "frequency_penalty", "logit_bias", "service_tier", "metadata",
    "store",
    # Chat-era tool surfaces — accepted only for explicit unsupported errors.
    "tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
    # Tool-loop budget (chat/Responses-native) — named unsupported, not unknown_fields.
    "max_tool_calls",
    "response_format",
    # Chat-era structured/output controls — accepted only for explicit migration errors.
    "modalities", "prediction", "reasoning_effort",
    # Chat-era multimodal/search — accepted only for named unsupported errors.
    "audio", "web_search_options",
    # Modern OpenAI SDK control fields — named unsupported errors.
    "prompt_cache_key", "safety_identifier", "verbosity", "prompt_cache_retention",
    "reasoning", "background", "include",
    "tool_resources",
} | {"attribution", "routing"}
ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
# Chat message object keys this gateway interprets. Anything else fails closed
# with unknown_message_fields (named error, not silent strip/smuggle).
ALLOWED_MESSAGE_KEYS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "refusal",
    "annotations",
    "audio",
    "function_call",
    "weight",
    "prefix",
}
ALLOWED_MODES = {"auto", "route", "conduct"}
ALLOWED_SIMULATE_KEYS = {"prompt", "mode", "include_orchestration_trace"}
ALLOWED_WORKFLOW_KEYS = {"prompt_text", "run_mode", "include_orchestration_trace"}
ALLOWED_EVALUATION_KEYS = {"prompts", "prompt_text", "run_mode", "include_orchestration_trace"}
ALLOWED_AGENT_PATCH_KEYS = {"status", "priority", "tags", "provider_exclusions"}
ALLOWED_AGENT_CREATE_KEYS = {
    "id",
    "model",
    "base_url",
    "api_key_env",
    "credential_key",
    "tags",
    "priority",
    "disabled",
    "provider_name",
    "provider_exclusions",
}


class RequestError(Exception):
    """HTTP-safe request failure."""

    def __init__(self, status: int, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail or {}


def _request_body_size(headers: Any, max_body_bytes: int) -> int:
    """Return a safe JSON body length or reject ambiguous HTTP framing.

    The stdlib handler does not decode transfer codings for this API. A single
    ASCII decimal ``Content-Length`` is therefore the only accepted framing
    signal; duplicate, comma-joined, negative, malformed, oversized, or
    transfer-coded requests fail closed before any body read.
    """
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        transfer_values = get_all("transfer-encoding")
        length_values = get_all("content-length")
    else:  # pragma: no cover - production uses email.message.Message headers
        transfer_value = headers.get("transfer-encoding")
        length_value = headers.get("content-length")
        transfer_values = None if transfer_value is None else [transfer_value]
        length_values = None if length_value is None else [length_value]

    if transfer_values is not None:
        raise RequestError(
            400,
            "invalid_request_framing",
            "transfer-encoding request framing is not supported",
        )
    if length_values is None:
        return 0
    if len(length_values) != 1 or "," in length_values[0]:
        raise RequestError(
            400,
            "invalid_request_framing",
            "content-length must appear exactly once",
        )
    value = length_values[0].strip()
    if not value or not value.isascii() or not value.isdecimal():
        raise RequestError(
            400,
            "invalid_request_framing",
            "content-length must be a non-negative decimal value",
        )
    normalized = value.lstrip("0") or "0"
    maximum = str(max_body_bytes)
    if len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    ):
        raise RequestError(413, "request_too_large", "request body exceeds configured limit")
    body_size = int(normalized)
    return body_size


@dataclass
class SecurityConfig:
    """Runtime safety controls for the stdlib HTTP server."""

    auth_token: str = ""
    admin_token: str = ""
    inference_token: str = ""
    allow_public_bind: bool = False
    expose_trace_by_default: bool = False
    max_body_bytes: int = 64 * 1024
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    max_concurrent_runs: int = 8
    # Deployment may inject a real OIDC/JWT verifier (for example a Keyverse
    # relying-party adapter). The core deliberately does not decode JWTs with
    # an unsafe hand-rolled parser or own Keycloak admin credentials.
    bearer_verifier: Callable[[str, str], bool] | None = None
    _rate_buckets: dict[str, tuple[int, float]] = field(default_factory=dict, init=False, repr=False)
    _rate_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _run_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.auth_token and (self.admin_token or self.inference_token):
            raise ValueError("single auth_token cannot be combined with split tokens")
        if (self.admin_token or self.inference_token) and not (self.admin_token and self.inference_token):
            raise ValueError("split token mode requires both admin_token and inference_token")
        if type(self.max_concurrent_runs) is not int or not 1 <= self.max_concurrent_runs <= MAX_LOCAL_CONCURRENCY:
            raise ValueError(
                f"max_concurrent_runs must be an integer in 1..{MAX_LOCAL_CONCURRENCY}"
            )
        self._run_semaphore = threading.BoundedSemaphore(self.max_concurrent_runs)

    def check_bind(self, host: str) -> None:
        """Require explicit opt-in before binding the API to public interfaces."""
        if host in {"0.0.0.0", "::", ""} and not self.allow_public_bind:  # nosec B104 - comparison rejects public bind unless explicitly opted in.
            raise ValueError("public bind requires --allow-public-bind")

    def authorize(self, headers: Any, scope: str, client_address: str) -> None:
        """Validate bearer token for admin or inference scope."""
        if not (self.auth_token or self.admin_token or self.inference_token or self.bearer_verifier):
            raise RequestError(401, "unauthorized", "bearer token is required")
        raw = headers.get("authorization", "")
        if not raw.lower().startswith("bearer "):
            raise RequestError(401, "unauthorized", "bearer token is required")
        token = raw.split(" ", 1)[1].strip()
        if self.bearer_verifier is not None:
            try:
                valid = bool(self.bearer_verifier(token, scope))
            except Exception:  # noqa: BLE001 - an auth adapter failure is an auth denial
                valid = False
        else:
            if scope == "admin":
                expected = self.admin_token or self.auth_token
            elif scope == "inference":
                expected = self.inference_token or self.auth_token
            else:
                expected = ""
            valid = bool(expected) and secrets.compare_digest(token, expected)
        if not valid:
            raise RequestError(401, "unauthorized", "bearer token is invalid for this scope")

    def check_rate_limit(self, key: str) -> None:
        """Apply a simple per-client fixed-window request budget."""
        now = time.monotonic()
        with self._rate_lock:
            count, reset_at = self._rate_buckets.get(key, (0, now + self.rate_limit_window_seconds))
            if now >= reset_at:
                count, reset_at = 0, now + self.rate_limit_window_seconds
            if count >= self.rate_limit_requests:
                raise RequestError(429, "rate_limit_exceeded", "request rate limit exceeded")
            self._rate_buckets[key] = (count + 1, reset_at)

    def acquire_run_slot(self) -> None:
        """Reserve a run slot, rejecting quickly when the process is saturated."""
        if not self._run_semaphore.acquire(blocking=False):
            raise RequestError(503, "concurrency_limit_exceeded", "too many concurrent orchestration runs")

    def release_run_slot(self) -> None:
        """Release a run slot acquired by acquire_run_slot."""
        self._run_semaphore.release()

    def readiness_profile(self) -> dict[str, Any]:
        """Return a secret-free security profile for sales-readiness evidence."""
        if self.bearer_verifier is not None:
            auth_mode = "external_bearer_verifier"
        elif self.admin_token and self.inference_token:
            auth_mode = "split_token"
        elif self.auth_token:
            auth_mode = "single_token"
        else:
            auth_mode = "auth_not_configured"
        return {
            "auth_mode": auth_mode,
            "allow_public_bind": self.allow_public_bind,
            "expose_trace_by_default": self.expose_trace_by_default,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "max_concurrent_runs": self.max_concurrent_runs,
        }


def _error_payload(error_code: str, error_message: str, error_detail: dict[str, Any] | None = None) -> dict[str, Any]:
    detail = error_detail or {}
    return {
        "error": {"code": error_code, "message": error_message, "detail": detail},
        "error_code": error_code,
        "error_message": error_message,
        "error_detail": detail,
    }


MAX_JSON_NESTING_DEPTH = 32


def _reject_excessive_json_nesting(payload: bytes, max_depth: int = MAX_JSON_NESTING_DEPTH) -> None:
    """Reject JSON with object/array nesting deeper than max_depth before parsing.

    json.loads() has no built-in depth cap, so a deeply nested payload well
    under max_body_bytes can still burn disproportionate CPU/stack during
    parsing (JSON-bomb DoS). Structural brackets are always single ASCII
    bytes and UTF-8 continuation/lead bytes are always >= 0x80, so a raw
    byte scan that only toggles on an unescaped '"' is safe without decoding.
    """
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        char = chr(byte)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth > max_depth:
                raise RequestError(400, "invalid_json", "request body JSON nesting exceeds the allowed depth")
        elif char in "}]":
            depth -= 1


TOOL_FALLBACK_STOPPED_STATUS = 409
TOOL_FALLBACK_STOPPED_CODE = "tool_execution_stopped"
TOOL_FALLBACK_STOPPED_MESSAGE = (
    "tool execution stopped because no safe retry or failover was available"
)


def _tool_fallback_error_detail(error: ToolFallbackStoppedError) -> dict[str, Any]:
    """Return secret-free structured evidence for one fail-closed tool decision."""
    decision = error.decision
    detail = {
        "action": decision.action.value,
        "failure_kind": decision.kind.value,
        "reason_code": decision.reason_code,
    }
    observed_kind = decision.observed_kind or decision.kind
    if observed_kind is not decision.kind:
        detail["observed_failure_kind"] = observed_kind.value
    return detail


def _coerce_json(payload: bytes) -> dict[str, Any]:
    _reject_excessive_json_nesting(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RequestError(400, "invalid_json", "request body must be a JSON object")
    return value




def _coerce_optional_bool(
    value: Any,
    *,
    error_code: str,
    message: str,
) -> bool | None:
    """Treat null/empty as omit; accept bool, 0/1 int or whole float, and true/false strings.

    ``True``/``False`` are not accepted via the int branch (``bool`` is a
    subclass of ``int`` in Python), so only bare ``0``/``1`` coerce. Whole
    floats (``0.0``/``1.0``) and whole-float strings (``"0.0"``/``"1.0"``) from
    form/JS SDKs coerce the same way. Other strings are case-insensitive
    ``true``/``false`` (with incidental whitespace stripped).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is float and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        # Whole-float digit strings ("0.0", "1.00") from form encodings.
        try:
            as_float = float(lowered)
        except ValueError as exc:
            raise RequestError(400, error_code, message) from exc
        if as_float in (0.0, 1.0) and as_float.is_integer():
            return bool(int(as_float))
    raise RequestError(400, error_code, message)


def _coerce_optional_int(
    value: Any,
    *,
    error_code: str,
    message: str,
) -> int | None:
    """Treat null/empty as omit; accept int, digit strings, and whole-number floats.

    JS JSON and some SDKs serialize integers as strings (``"1"``), whole floats
    (``1.0``), or whole-float *strings* (``"1.0"`` / ``"0.0"`` from form encodings).
    All coerce to ``int``; non-integral floats/strings and bools fail closed.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise RequestError(400, error_code, message)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit() and stripped not in {"-", ""}:
            return int(stripped)
        # Whole-number float strings ("1.0", "0.0", " 2.00 ") from JS form SDKs.
        try:
            as_float = float(stripped)
        except ValueError as exc:
            raise RequestError(400, error_code, message) from exc
        if as_float.is_integer() and abs(as_float) <= 2**53:
            return int(as_float)
        raise RequestError(400, error_code, message)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and abs(value) <= 2**53:
            return int(value)
        raise RequestError(400, error_code, message)
    raise RequestError(400, error_code, message)


def _coerce_optional_float(
    value: Any,
    *,
    error_code: str,
    message: str,
) -> float | None:
    """Treat null/empty as omit; accept int/float and numeric strings."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise RequestError(400, error_code, message)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise RequestError(400, error_code, message) from exc
    raise RequestError(400, error_code, message)


def _validate_completion_prompt(prompt: Any) -> list[dict[str, str]]:
    """Legacy Completions ``prompt`` → single user message list.

    Accepts OpenAI shapes:

    - non-empty string
    - non-empty array of non-empty strings (at most 128 items; joined with newlines)
    - non-empty array of non-negative token integers (whole floats like ``1.0`` ok)
    - non-empty array of token-integer arrays (joined like string arrays)

    Token sequences are re-encoded to a stable text surrogate for string
    completion backends (same encoding as embeddings). Bools, negatives,
    non-integral floats, and mixed token/string batches fail closed.
    """
    if isinstance(prompt, str):
        if not prompt.strip():
            raise RequestError(400, "invalid_prompt", "prompt must be a non-empty string or array")
        if len(prompt) > 32_000:
            raise RequestError(400, "invalid_prompt", "prompt must be at most 32000 characters")
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        if not prompt:
            raise RequestError(400, "invalid_prompt", "prompt must be a non-empty string or array")
        if len(prompt) > 128:
            raise RequestError(
                400,
                "invalid_prompt",
                "prompt array must contain at most 128 items",
            )
        # Single token sequence: [1, 2, 3] / [1.0, 2.0] → one user message.
        coerced_tokens = _coerce_embedding_token_sequence(prompt)
        if coerced_tokens is not None:
            text = _embedding_token_sequence_to_text(coerced_tokens)
            if len(text) > 32_000:
                raise RequestError(400, "invalid_prompt", "prompt must be at most 32000 characters")
            return [{"role": "user", "content": text}]
        # Batch of token sequences: [[1,2],[3]] — join surrogates like string arrays.
        if isinstance(prompt[0], list):
            batch_tokens: list[list[int]] = []
            for item in prompt:
                coerced = _coerce_embedding_token_sequence(item)
                if coerced is None:
                    raise RequestError(
                        400,
                        "invalid_prompt",
                        "token-id prompt arrays must contain non-negative token integers",
                    )
                batch_tokens.append(coerced)
            parts = [_embedding_token_sequence_to_text(item) for item in batch_tokens]
            joined = "\n".join(parts)
            if len(joined) > 32_000:
                raise RequestError(400, "invalid_prompt", "prompt must be at most 32000 characters")
            return [{"role": "user", "content": joined}]
        # Apparent token sequence with bools/negatives/non-integral floats — fail closed.
        if all(_is_token_id_shaped(item) for item in prompt):
            raise RequestError(
                400,
                "invalid_prompt",
                "token-id prompts must be non-negative integers",
            )
        parts = []
        for item in prompt:
            if not isinstance(item, str):
                raise RequestError(400, "invalid_prompt", "prompt array items must be strings")
            if not item.strip():
                raise RequestError(
                    400,
                    "invalid_prompt",
                    "prompt array items must be non-empty strings",
                )
            parts.append(item)
        joined = "\n".join(parts)
        if not joined.strip():
            raise RequestError(400, "invalid_prompt", "prompt must be a non-empty string or array")
        if len(joined) > 32_000:
            raise RequestError(400, "invalid_prompt", "prompt must be at most 32000 characters")
        return [{"role": "user", "content": joined}]
    raise RequestError(400, "invalid_prompt", "prompt must be a non-empty string or array")


def _validate_completions_stream(body: dict[str, Any]) -> bool | None:
    """Legacy Completions ``stream`` — strict boolean honesty contract.

    OpenAI Completions accepts streaming. This gateway:
    - accepts omit and ``stream=false`` as the non-streaming text_completion path
    - rejects ``stream=true`` with a clear redirect to chat completions
    - rejects non-boolean values fail-closed (no silent coercion)
    """
    if "stream" not in body:
        return None
    stream = body.get("stream")
    stream = _coerce_optional_bool(
        stream, error_code="invalid_stream", message="stream must be a boolean"
    )
    if stream is None:
        return None
    if stream is True:
        raise RequestError(
            400,
            "invalid_stream",
            "stream is not supported on /v1/completions; use /v1/chat/completions",
        )
    return stream


def _validate_completions_echo(body: dict[str, Any]) -> bool | None:
    """Legacy Completions ``echo`` — boolean / JS 0/1; ``true`` is not supported.

    OpenAI can prepend the prompt to the completion when ``echo`` is true. This
    gateway does not implement that behaviour, so ``echo=true`` fails closed with
    a clear ``invalid_echo`` error. ``false``/``0`` and omit remain valid.
    """
    if "echo" not in body:
        return None
    echo = _coerce_optional_bool(
        body.get("echo"),
        error_code="invalid_echo",
        message="echo must be a boolean",
    )
    if echo is None:
        return None
    if echo is True:
        raise RequestError(
            400,
            "invalid_echo",
            "echo=true is not supported on /v1/completions",
        )
    return echo














def _coerce_logit_bias_value(value: Any) -> float:
    """Coerce a logit_bias map value to float in [-100, 100].

    Accepts int/float and numeric strings (JS form SDKs); bools fail closed.
    """
    number = _coerce_optional_float(
        value,
        error_code="invalid_logit_bias",
        message="logit_bias values must be numbers in [-100, 100]",
    )
    if number is None or isinstance(value, bool):
        raise RequestError(
            400,
            "invalid_logit_bias",
            "logit_bias values must be numbers in [-100, 100]",
        )
    if number < -100 or number > 100:
        raise RequestError(
            400,
            "invalid_logit_bias",
            "logit_bias values must be numbers in [-100, 100]",
        )
    return float(number)


def _coerce_logit_bias_token_key(key: Any) -> str:
    """Normalize a logit_bias map key to a digit token id string.

    Form/JS SDKs often pad numeric keys with incidental whitespace (``" 100 "``).
    Strip before the digit check so type validation matches OpenAI token-id
    maps; empty-after-strip and non-digit keys fail closed.
    """
    token = str(key).strip()
    if not token.isdigit():
        raise RequestError(400, "invalid_logit_bias", "logit_bias keys must be digit token ids")
    return token


def _validate_completions_logit_bias(body: dict[str, Any]) -> dict[str, float] | None:
    """Legacy Completions ``logit_bias`` — empty object is a no-op; non-empty fails closed.

    OpenAI uses logit_bias to bias token sampling. This gateway does not apply
    token biases on the Completions route. An empty object is an honest no-op
    (SDK clients often send ``{}``). Any non-empty map is type-checked then
    rejected so clients never believe sampling bias was applied.
    """
    if "logit_bias" not in body:
        return None
    bias = body.get("logit_bias")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if bias is None:
        return None
    if not isinstance(bias, dict):
        raise RequestError(400, "invalid_logit_bias", "logit_bias must be an object of token biases")
    # Empty object: no tokens to bias — treat as omit (honest no-op).
    if len(bias) == 0:
        return {}
    if len(bias) > 300:
        raise RequestError(400, "invalid_logit_bias", "logit_bias must contain at most 300 entries")
    for key, value in bias.items():
        _coerce_logit_bias_token_key(key)
        _coerce_logit_bias_value(value)
    raise RequestError(
        400,
        "invalid_logit_bias",
        "logit_bias is not supported on /v1/completions",
    )



def _validate_service_tier(body: dict[str, Any], *, endpoint_path: str) -> str | None:
    """OpenAI ``service_tier`` — known tier names are no-ops; unknown fail closed.

    OpenAI uses service_tier for capacity priority (auto/default/flex/priority).
    This gateway has no separate tiered capacity plane, so recognised OpenAI
    names are accepted as default-capacity no-ops (SDK clients often send flex).
    Explicit JSON null or empty string is treat-as-omit. Unknown values fail
    closed so clients cannot invent tier labels.
    """
    if "service_tier" not in body:
        return None
    service_tier = body.get("service_tier")
    # Explicit JSON null or empty string is treat-as-omit (SDK optional default).
    if service_tier is None or (isinstance(service_tier, str) and not service_tier.strip()):
        return None
    if not isinstance(service_tier, str):
        raise RequestError(400, "invalid_service_tier", "service_tier must be a string")
    # Strip incidental whitespace and casefold so " AUTO " / " Flex " match.
    service_tier = service_tier.strip().lower()
    if service_tier not in {"auto", "default", "flex", "priority"}:
        raise RequestError(
            400,
            "invalid_service_tier",
            "service_tier must be one of auto, default, flex, priority "
            f"on {endpoint_path}",
        )
    body["service_tier"] = service_tier
    return service_tier


def _validate_completions_user(body: dict[str, Any]) -> str | None:
    """OpenAI ``user`` end-user id — optional string, max 64 characters.

    Explicit JSON null is treat-as-omit (SDK optional default). Empty or
    whitespace-only strings still fail closed so clients cannot attribute spend
    to a blank identity. Scalar bool/int/float values coerce to strings (JS/form
    SDKs often send numeric account ids); objects/arrays fail closed. Coerced
    values are written back so proxy/egress sees an honest string identity.
    """
    if "user" not in body:
        return None
    user = body.get("user")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if user is None:
        return None
    if isinstance(user, bool):
        # JSON bool → lowercase OpenAI-style string form (parity with metadata).
        user = "true" if user else "false"
    elif type(user) is int:
        user = str(user)
    elif isinstance(user, float):
        # Whole floats stringify compactly (1.0 → "1"); others use str().
        if user.is_integer() and abs(user) <= 2**53:
            user = str(int(user))
        else:
            user = str(user)
    elif not isinstance(user, str):
        raise RequestError(400, "invalid_user", "user must be a string of at most 64 characters")
    if not user.strip():
        raise RequestError(400, "invalid_user", "user must be a non-empty string of at most 64 characters")
    if len(user) > 64:
        raise RequestError(400, "invalid_user", "user must be a string of at most 64 characters")
    body["user"] = user
    return user

def _validate_completions_n(body: dict[str, Any]) -> int | None:
    """Legacy Completions ``n`` — positive integer; only ``n=1`` is supported.

    OpenAI can return multiple completions when ``n > 1``. This gateway always
    returns a single choice, so ``n > 1`` fails closed. ``n=1`` and omit remain
    valid. Cap 128 is retained for clear range errors before the support check.
    Digit strings and whole-number floats (JS JSON) coerce.
    """
    if "n" not in body:
        return None
    n = _coerce_optional_int(
        body.get("n"),
        error_code="invalid_n",
        message="n must be a positive integer",
    )
    if n is None:
        return None
    body["n"] = n
    if n < 1:
        raise RequestError(400, "invalid_n", "n must be a positive integer")
    if n > 128:
        raise RequestError(400, "invalid_n", "n must be at most 128")
    if n > 1:
        raise RequestError(
            400,
            "invalid_n",
            "n greater than 1 is not supported on /v1/completions",
        )
    return n


def _validate_responses_n(body: dict[str, Any]) -> int | None:
    """Responses ``n`` — only omit or 1; multi-choice is not framed on passthrough.

    OpenAI may request multiple samples via ``n``. This gateway's Responses
    passthrough returns a single completion shape, so ``n`` greater than 1
    fails closed. ``n=1`` and omit remain valid.
    Digit strings and whole-number floats (JS JSON) coerce.
    """
    if "n" not in body:
        return None
    n = _coerce_optional_int(
        body.get("n"),
        error_code="invalid_n",
        message="n must be an integer",
    )
    if n is None:
        return None
    body["n"] = n
    if n < 1:
        raise RequestError(400, "invalid_n", "n must be a positive integer")
    if n > 1:
        raise RequestError(
            400,
            "invalid_n",
            "n greater than 1 is not supported on /v1/responses",
        )
    return n



def _validate_responses_logit_bias(body: dict[str, Any]) -> dict[str, float] | None:
    """Responses ``logit_bias`` — digit-token map values in [-100, 100]; pass through.

    Invalid shapes fail closed before provider egress. Valid maps (including empty)
    are forwarded on Responses passthrough. Numeric strings coerce (JS form SDKs);
    padded digit keys strip before write-back so providers see clean token ids.
    """
    if "logit_bias" not in body:
        return None
    bias = body.get("logit_bias")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if bias is None:
        return None
    if not isinstance(bias, dict):
        raise RequestError(400, "invalid_logit_bias", "logit_bias must be an object of token biases")
    if len(bias) > 300:
        raise RequestError(400, "invalid_logit_bias", "logit_bias must contain at most 300 entries")
    cleaned: dict[str, float] = {}
    for key, value in bias.items():
        token = _coerce_logit_bias_token_key(key)
        cleaned[token] = _coerce_logit_bias_value(value)
    body["logit_bias"] = cleaned
    return cleaned


def _validate_responses_logprobs(body: dict[str, Any]) -> None:
    """Responses ``logprobs`` / ``top_logprobs`` — OpenAI shape; invalid fail closed.

    ``logprobs`` must be boolean when present. ``top_logprobs`` requires
    ``logprobs=true`` and must be an integer in [0, 20].
    Explicit JSON null for either field is treat-as-omit (SDK optional default).
    """
    if "logprobs" in body:
        lp = body.get("logprobs")
        if lp is not None:
            coerced = _coerce_optional_bool(
                lp,
                error_code="invalid_logprobs",
                message="logprobs must be a boolean",
            )
            if coerced is None:
                pass
            else:
                body["logprobs"] = coerced
    if "top_logprobs" in body:
        tlp = body.get("top_logprobs")
        if tlp is None or (isinstance(tlp, str) and not tlp.strip()):
            return
        if body.get("logprobs") is not True:
            raise RequestError(
                400,
                "invalid_top_logprobs",
                "top_logprobs requires logprobs=true on /v1/responses",
            )
        # Digit strings / whole floats coerce (JS JSON integer-as-string).
        coerced_tlp = _coerce_optional_int(
            tlp,
            error_code="invalid_top_logprobs",
            message="top_logprobs must be an integer in [0, 20]",
        )
        if coerced_tlp is None:
            return
        if coerced_tlp < 0 or coerced_tlp > 20:
            raise RequestError(400, "invalid_top_logprobs", "top_logprobs must be an integer in [0, 20]")
        body["top_logprobs"] = coerced_tlp



def _validate_responses_parallel_tool_calls(body: dict[str, Any]) -> bool | None:
    """Responses ``parallel_tool_calls`` — strict boolean when present.

    OpenAI uses this flag to allow concurrent tool invocations. Invalid types
    fail closed before provider passthrough so clients never believe a coerced
    value was applied. ``true`` requires a non-empty ``tools`` array (chat parity).
    """
    if "parallel_tool_calls" not in body:
        return None
    value = body.get("parallel_tool_calls")
    value = _coerce_optional_bool(
        value,
        error_code="invalid_parallel_tool_calls",
        message="parallel_tool_calls must be a boolean",
    )
    if value is None:
        return None
    if value is True:
        tools = body.get("tools") if "tools" in body else None
        if not isinstance(tools, list) or not tools:
            raise RequestError(
                400,
                "invalid_parallel_tool_calls",
                "parallel_tool_calls=true requires tools on /v1/responses",
            )
    return value


def _validate_responses_seed(body: dict[str, Any]) -> int | None:
    """Responses ``seed`` — signed int64; valid values pass through to the provider.

    Unlike Completions (where seed is not applied), Responses passthrough forwards
    seed to the selected agent. Invalid types/ranges fail closed before egress.
    Digit strings and whole-number floats (JS JSON) coerce.
    """
    if "seed" not in body:
        return None
    seed = _coerce_optional_int(
        body.get("seed"),
        error_code="invalid_seed",
        message="seed must be an integer",
    )
    if seed is None:
        return None
    body["seed"] = seed
    if seed < -(2**63) or seed > (2**63 - 1):
        raise RequestError(400, "invalid_seed", "seed must fit in a signed 64-bit integer")
    return seed


def _validate_responses_stop(body: dict[str, Any]) -> str | list[str] | None:
    """Responses ``stop`` — string or ≤4 non-empty strings (≤256 chars); pass through.

    Shape matches OpenAI. Valid stop values are forwarded on Responses passthrough;
    invalid shapes fail closed so clients never believe a broken stop list was applied.
    """
    if "stop" not in body:
        return None
    stop = body.get("stop")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if stop is None:
        return None
    if isinstance(stop, str):
        # Empty/whitespace string is omit-equivalent (no stop sequences).
        if not stop.strip():
            return None
        if len(stop) > 256:
            raise RequestError(400, "invalid_stop", "each stop sequence must be at most 256 characters")
        return stop
    if isinstance(stop, list):
        # Drop whitespace-only items; empty result is omit-equivalent.
        stop = [item for item in stop if not (isinstance(item, str) and not item.strip())]
        if not stop:
            return None
        if len(stop) > 4:
            raise RequestError(400, "invalid_stop", "stop must be a string or array of up to 4 non-empty strings")
        for item in stop:
            if not isinstance(item, str) or not item:
                raise RequestError(400, "invalid_stop", "stop sequences must be non-empty strings")
            if len(item) > 256:
                raise RequestError(400, "invalid_stop", "each stop sequence must be at most 256 characters")
        return stop
    raise RequestError(400, "invalid_stop", "stop must be a string or array of up to 4 non-empty strings")


def _validate_completions_stop(body: dict[str, Any]) -> str | list[str] | None:
    """Legacy Completions ``stop`` — type-checked then rejected (not applied).

    OpenAI uses stop sequences to cut generation early. This gateway validates
    shape (string or ≤4 non-empty strings, each ≤256 chars) but does not apply
    stop sequences on the Completions path, so any provided non-empty ``stop`` fails closed. Empty string/array/null are omit no-ops.
    """
    if "stop" not in body:
        return None
    stop = body.get("stop")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if stop is None:
        return None
    if isinstance(stop, str):
        # Empty/whitespace string is omit-equivalent (no stop sequences).
        if not stop.strip():
            return None
        if len(stop) > 256:
            raise RequestError(400, "invalid_stop", "each stop sequence must be at most 256 characters")
    elif isinstance(stop, list):
        # Drop whitespace-only items; empty result is omit-equivalent.
        stop = [item for item in stop if not (isinstance(item, str) and not item.strip())]
        if not stop:
            return None
        if len(stop) > 4:
            raise RequestError(400, "invalid_stop", "stop must be a string or array of up to 4 non-empty strings")
        for item in stop:
            if not isinstance(item, str) or not item:
                raise RequestError(400, "invalid_stop", "stop sequences must be non-empty strings")
            if len(item) > 256:
                raise RequestError(400, "invalid_stop", "each stop sequence must be at most 256 characters")
    else:
        raise RequestError(400, "invalid_stop", "stop must be a string or array of up to 4 non-empty strings")
    raise RequestError(
        400,
        "invalid_stop",
        "stop sequences are not supported on /v1/completions",
    )



def _validate_completions_seed(body: dict[str, Any]) -> int | None:
    """Legacy Completions ``seed`` — type-checked then rejected (not applied).

    OpenAI uses seed for best-effort deterministic sampling. This gateway validates
    signed int64 integers but does not apply seed on the Completions route path,
    so any provided ``seed`` fails closed. Omit remains valid.
    Digit strings and whole-number floats (JS JSON) coerce before the support reject.
    """
    if "seed" not in body:
        return None
    seed = _coerce_optional_int(
        body.get("seed"),
        error_code="invalid_seed",
        message="seed must be an integer",
    )
    if seed is None:
        return None
    body["seed"] = seed
    if seed < -(2**63) or seed > (2**63 - 1):
        raise RequestError(400, "invalid_seed", "seed must fit in a signed 64-bit integer")
    raise RequestError(
        400,
        "invalid_seed",
        "seed is not supported on /v1/completions",
    )



def _validate_completions_frequency_penalty(body: dict[str, Any]) -> float | None:
    """Legacy Completions ``frequency_penalty`` — number in [-2, 2]."""
    if "frequency_penalty" not in body:
        return None
    value = body.get("frequency_penalty")
    value = _coerce_optional_float(
        value,
        error_code="invalid_frequency_penalty",
        message="frequency_penalty must be a number in [-2, 2]",
    )
    if value is None:
        return None
    number = float(value)
    if number < -2 or number > 2:
        raise RequestError(400, "invalid_frequency_penalty", "frequency_penalty must be a number in [-2, 2]")
    body["frequency_penalty"] = number
    return number

def _validate_completions_presence_penalty(body: dict[str, Any]) -> float | None:
    """Legacy Completions ``presence_penalty`` — number in [-2, 2]."""
    if "presence_penalty" not in body:
        return None
    value = body.get("presence_penalty")
    value = _coerce_optional_float(
        value,
        error_code="invalid_presence_penalty",
        message="presence_penalty must be a number in [-2, 2]",
    )
    if value is None:
        return None
    number = float(value)
    if number < -2 or number > 2:
        raise RequestError(400, "invalid_presence_penalty", "presence_penalty must be a number in [-2, 2]")
    body["presence_penalty"] = number
    return number

def _validate_completions_temperature(body: dict[str, Any]) -> float | None:
    """Legacy Completions ``temperature`` — number in [0, 2]."""
    if "temperature" not in body:
        return None
    temperature = body.get("temperature")
    temperature = _coerce_optional_float(
        temperature,
        error_code="invalid_temperature",
        message="temperature must be a number in [0, 2]",
    )
    if temperature is None:
        return None
    value = float(temperature)
    if value < 0 or value > 2:
        raise RequestError(400, "invalid_temperature", "temperature must be a number in [0, 2]")
    body["temperature"] = value
    return value

def _validate_completions_top_p(body: dict[str, Any]) -> float | None:
    """Legacy Completions ``top_p`` — number in (0, 1] (OpenAI nucleus sampling)."""
    if "top_p" not in body:
        return None
    top_p = body.get("top_p")
    top_p = _coerce_optional_float(
        top_p,
        error_code="invalid_top_p",
        message="top_p must be a number in (0, 1]",
    )
    if top_p is None:
        return None
    value = float(top_p)
    if value <= 0 or value > 1:
        raise RequestError(400, "invalid_top_p", "top_p must be a number in (0, 1]")
    body["top_p"] = value
    return value

def _validate_completions_model(body: dict[str, Any]) -> str:
    """Legacy Completions ``model`` — required non-empty string (OpenAI parity).

    Incidental leading/trailing whitespace is stripped and written back so
    tools/response_format passthrough (``proxy_completion``) matches the same
    pool model id as the orchestration path. Form/JS SDKs often pad model names.
    """
    if "model" not in body:
        raise RequestError(400, "invalid_model", "model is required")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RequestError(400, "invalid_model", "model must be a non-empty string")
    model = model.strip()
    if len(model) > 256:
        raise RequestError(400, "invalid_model", "model must be at most 256 characters")
    body["model"] = model
    return model

def _validate_completions_max_tokens(body: dict[str, Any]) -> int | None:
    """Legacy Completions ``max_tokens`` — positive integer capped at 1_048_576."""
    if "max_tokens" not in body:
        return None
    max_tokens = body.get("max_tokens")
    max_tokens = _coerce_optional_int(
        max_tokens,
        error_code="invalid_max_tokens",
        message="max_tokens must be a positive integer",
    )
    if max_tokens is None:
        return None
    if max_tokens < 1:
        raise RequestError(400, "invalid_max_tokens", "max_tokens must be a positive integer")
    if max_tokens > 1_048_576:
        raise RequestError(
            400,
            "invalid_max_tokens",
            "max_tokens must be at most 1048576",
        )
    body["max_tokens"] = max_tokens
    return max_tokens

def _validate_chat_max_completion_tokens(body: dict[str, Any]) -> int | None:
    """Chat Completions ``max_completion_tokens`` — positive integer capped at 1_048_576.

    OpenAI prefers this over legacy ``max_tokens`` for chat. When both are set,
    ``max_completion_tokens`` wins so clients get a single honest budget.
    """
    if "max_completion_tokens" not in body:
        return None
    max_completion_tokens = body.get("max_completion_tokens")
    max_completion_tokens = _coerce_optional_int(
        max_completion_tokens,
        error_code="invalid_max_completion_tokens",
        message="max_completion_tokens must be a positive integer",
    )
    if max_completion_tokens is None:
        return None
    if max_completion_tokens < 1:
        raise RequestError(
            400,
            "invalid_max_completion_tokens",
            "max_completion_tokens must be a positive integer",
        )
    if max_completion_tokens > 1_048_576:
        raise RequestError(
            400,
            "invalid_max_completion_tokens",
            "max_completion_tokens must be at most 1048576",
        )
    body["max_completion_tokens"] = max_completion_tokens
    return max_completion_tokens


def _validate_responses_max_output_tokens(body: dict[str, Any]) -> int | None:
    """Responses ``max_output_tokens`` — OpenAI-native output budget (positive int).

    Official Responses clients send ``max_output_tokens`` rather than chat-era
    ``max_tokens``. Accept and type-check so the field is not opaque
    ``unknown_fields``; value is left on the body for provider passthrough.
    Cap matches ``max_tokens`` (1_048_576). Digit strings and whole-number
    floats (JS JSON) coerce.
    """
    if "max_output_tokens" not in body:
        return None
    value = _coerce_optional_int(
        body.get("max_output_tokens"),
        error_code="invalid_max_output_tokens",
        message="max_output_tokens must be a positive integer",
    )
    if value is None:
        return None
    body["max_output_tokens"] = value
    if value < 1:
        raise RequestError(
            400,
            "invalid_max_output_tokens",
            "max_output_tokens must be a positive integer",
        )
    if value > 1_048_576:
        raise RequestError(
            400,
            "invalid_max_output_tokens",
            "max_output_tokens must be at most 1048576",
        )
    return value



def _validate_max_tool_calls(
    body: dict[str, Any],
    *,
    endpoint_path: str,
) -> None:
    """Reject ``max_tool_calls`` — no multi-step tool loop on this gateway.

    OpenAI may cap tool-call rounds via ``max_tool_calls`` (Responses-native;
    some chat SDKs also send it). This gateway proxies a single completion and
    does not run a tool loop, so any provided value fails closed with a named
    error rather than opaque ``unknown_fields``. Explicit JSON null, empty
    / whitespace strings, and zero (int/float/digit or whole-float string
    ``"0"`` / ``"0.0"``) are treat-as-omit (SDK optional defaults / no tool
    rounds requested).
    """
    if "max_tool_calls" not in body:
        return
    value = body.get("max_tool_calls")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    # Zero is omit-equivalent (no tool-call rounds). Digit/"0"/0.0/"0.0" coerce first.
    if type(value) is int and value == 0:
        return
    if isinstance(value, float) and value == 0.0:
        return
    if isinstance(value, str) and value.strip() == "0":
        return
    coerced = _coerce_optional_int(
        value,
        error_code="invalid_max_tool_calls",
        message="max_tool_calls must be an integer",
    )
    if coerced is None or coerced == 0:
        return
    raise RequestError(
        400,
        "invalid_max_tool_calls",
        f"max_tool_calls is not supported on {endpoint_path}",
    )


def _validate_responses_max_tool_calls(body: dict[str, Any]) -> None:
    """Responses ``max_tool_calls`` — named reject; null/empty omit."""
    _validate_max_tool_calls(body, endpoint_path="/v1/responses")


def _validate_completions_logprobs(body: dict[str, Any]) -> int | bool | None:
    """Legacy Completions ``logprobs`` — token logprobs are not supported.

    This gateway always returns ``logprobs: null`` on text completions, so
    boolean ``true`` and nonzero integer logprobs fail closed. ``false``, omit,
    integer ``0``, and string ``false``/``0`` (JS form defaults) are
    omit-equivalent no-ops.
    """
    if "logprobs" not in body:
        return None
    logprobs = body.get("logprobs")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if logprobs is None:
        return None
    # Integer 0 is historical OpenAI "no logprobs" — omit-equivalent.
    if type(logprobs) is int and logprobs == 0:
        return None
    # Whole-float 0.0 (JS) and digit/float-string "0"/"0.0" are omit-equivalent.
    if isinstance(logprobs, float) and logprobs == 0.0:
        return None
    if isinstance(logprobs, str) and logprobs.strip():
        try:
            as_zero = _coerce_optional_int(
                logprobs,
                error_code="invalid_logprobs",
                message="logprobs must be false; token logprobs are not supported on /v1/completions",
            )
            if as_zero == 0:
                return None
        except RequestError:
            pass  # fall through to bool coerce / fail-closed
    coerced = _coerce_optional_bool(
        logprobs,
        error_code="invalid_logprobs",
        message="logprobs must be false; token logprobs are not supported on /v1/completions",
    )
    if coerced is None or coerced is False:
        return None if coerced is None else False
    raise RequestError(
        400,
        "invalid_logprobs",
        "logprobs must be false; token logprobs are not supported on /v1/completions",
    )


def _validate_completions_top_logprobs(body: dict[str, Any]) -> None:
    """Reject non-zero ``top_logprobs`` on legacy Completions.

    OpenAI Completions historically used integer ``logprobs`` (0–5); modern
    chat uses boolean ``logprobs`` + ``top_logprobs``. This gateway never returns
    token logprobs on /v1/completions, so non-zero ``top_logprobs`` fails closed
    with ``invalid_top_logprobs`` rather than opaque ``unknown_fields``.
    Explicit JSON null, empty/whitespace string, or zero (int/float/digit or
    whole-float string ``"0"`` / ``"0.0"``) is treat-as-omit — digit coerce
    then nonzero reject (parity with ``max_tool_calls``).
    """
    if "top_logprobs" not in body:
        return
    value = body.get("top_logprobs")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    # Digit / whole-float coerce first; zero is omit-equivalent (no top alts).
    coerced = _coerce_optional_int(
        value,
        error_code="invalid_top_logprobs",
        message="top_logprobs must be an integer",
    )
    if coerced is None or coerced == 0:
        return
    raise RequestError(
        400,
        "invalid_top_logprobs",
        "top_logprobs is not supported on /v1/completions",
    )


def _validate_completions_suffix(body: dict[str, Any]) -> str | None:
    """Legacy Completions ``suffix`` — optional string; non-empty is not supported.

    OpenAI appends ``suffix`` after the model completion. This gateway does not
    implement that insertion, so a non-empty suffix fails closed. Empty string
    and omit remain valid. Non-string values and oversized strings still fail.
    """
    if "suffix" not in body:
        return None
    suffix = body.get("suffix")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if suffix is None:
        return None
    if not isinstance(suffix, str):
        raise RequestError(400, "invalid_suffix", "suffix must be a string")
    # Empty/whitespace-only is treat-as-omit (SDK optional blank).
    if not suffix.strip():
        return None
    if len(suffix) > 8_000:
        raise RequestError(400, "invalid_suffix", "suffix must be at most 8000 characters")
    raise RequestError(
        400,
        "invalid_suffix",
        "non-empty suffix is not supported on /v1/completions",
    )


def _validate_completions_best_of(body: dict[str, Any]) -> int | None:
    """Legacy Completions ``best_of`` — positive integer, ``best_of >= n``, max 1.

    OpenAI generates ``best_of`` candidates server-side and returns the top ``n``.
    This gateway runs a single completion path, so ``best_of > 1`` fails closed
    rather than silently returning one unranked candidate. ``best_of=1`` (and
    omit) remain valid. Boolean ``True``/``False`` are rejected.
    Digit strings and whole-number floats (JS JSON) coerce.
    """
    if "best_of" not in body:
        return None
    best_of = _coerce_optional_int(
        body.get("best_of"),
        error_code="invalid_best_of",
        message="best_of must be a positive integer",
    )
    if best_of is None:
        return None
    body["best_of"] = best_of
    if best_of < 1:
        raise RequestError(400, "invalid_best_of", "best_of must be a positive integer")
    if best_of > 128:
        raise RequestError(400, "invalid_best_of", "best_of must be at most 128")
    if best_of > 1:
        raise RequestError(
            400,
            "invalid_best_of",
            "best_of greater than 1 is not supported on /v1/completions",
        )
    n = body.get("n", 1)
    if n is None or (isinstance(n, str) and not str(n).strip()):
        n = 1
    else:
        n = _coerce_optional_int(
            n,
            error_code="invalid_n",
            message="n must be a positive integer",
        )
        if n is None:
            n = 1
    body["n"] = n
    if n < 1:
        raise RequestError(400, "invalid_n", "n must be a positive integer")
    if best_of < n:
        raise RequestError(
            400,
            "invalid_best_of",
            "best_of must be greater than or equal to n",
        )
    return best_of


def _validate_completions_stream_options(body: dict[str, Any]) -> dict[str, Any] | None:
    """Legacy Completions ``stream_options`` — object with boolean flags; requires stream=true.

    Mirrors OpenAI chat Completions: ``stream_options`` is only valid when streaming.
    This gateway rejects Completions streaming, so a well-formed ``stream_options``
    still fails closed once ``stream`` is checked (or here if ``stream`` is not true).
    Explicit JSON null on *allowed* flag keys is treat-as-omit (SDK optional defaults).
    Unknown keys fail closed even when their value is null so clients cannot smuggle
    unsupported flags past the allow-list via null serialization.
    """
    if "stream_options" not in body:
        return None
    opts = body.get("stream_options")
    # Explicit JSON null or empty object is treat-as-omit (SDK optional default).
    if opts is None:
        return None
    if not isinstance(opts, dict):
        raise RequestError(400, "invalid_stream_options", "stream_options must be an object")
    if not opts:
        return None
    allowed = {"include_usage", "include_obfuscation"}
    # Reject unknown keys before dropping nulls (null is not a free pass for unknowns).
    unknown = sorted(set(opts) - allowed)
    if unknown:
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options contains unsupported fields",
            {"fields": unknown},
        )
    # Drop null flag values (SDK optional defaults) before further checks.
    opts = {key: value for key, value in opts.items() if value is not None}
    if not opts:
        return None
    # Coerce string/0-1 bool forms before all-false omit checks.
    coerced_opts: dict[str, Any] = {}
    for key, value in opts.items():
        coerced = _coerce_optional_bool(
            value,
            error_code="invalid_stream_options",
            message=f"stream_options.{key} must be a boolean",
        )
        if coerced is not None:
            coerced_opts[key] = coerced
    opts = coerced_opts
    if not opts:
        return None
    # All-false boolean flags are omit-equivalent no-ops (SDK optional defaults).
    if set(opts) <= allowed and all(v is False for v in opts.values()):
        return None
    if body.get("stream") is not True:
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options requires stream=true",
        )
    return opts




def _validate_chat_stream_options(body: dict[str, Any], stream: bool) -> dict[str, Any] | None:
    """Chat Completions ``stream_options`` — requires stream=true; include_usage unsupported.

    Shape matches OpenAI (include_usage / include_obfuscation booleans). This
    gateway's SSE route path does not emit a final usage chunk and does not
    apply stream obfuscation, so include_usage/include_obfuscation=true fail closed.
    Explicit JSON null on *allowed* flag keys is treat-as-omit (SDK optional defaults).
    Unknown keys fail closed even when their value is null so clients cannot smuggle
    unsupported flags past the allow-list via null serialization.
    """
    if "stream_options" not in body:
        return None
    opts = body.get("stream_options")
    # Explicit JSON null or empty object is treat-as-omit (SDK optional default).
    if opts is None:
        return None
    if not isinstance(opts, dict):
        raise RequestError(400, "invalid_stream_options", "stream_options must be an object")
    if not opts:
        return None
    allowed = {"include_usage", "include_obfuscation"}
    # Reject unknown keys before dropping nulls (null is not a free pass for unknowns).
    unknown = sorted(set(opts) - allowed)
    if unknown:
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options contains unsupported fields",
            {"fields": unknown},
        )
    # Drop null flag values (SDK optional defaults) before further checks.
    opts = {key: value for key, value in opts.items() if value is not None}
    if not opts:
        return None
    # Coerce string/0-1 bool forms before all-false omit and true reject.
    coerced_opts: dict[str, Any] = {}
    for key, value in opts.items():
        coerced = _coerce_optional_bool(
            value,
            error_code="invalid_stream_options",
            message=f"stream_options.{key} must be a boolean",
        )
        if coerced is not None:
            coerced_opts[key] = coerced
    opts = coerced_opts
    if not opts:
        return None
    # All-false boolean flags are omit-equivalent no-ops (SDK optional defaults).
    if set(opts) <= allowed and all(v is False for v in opts.values()):
        return None
    if stream is not True:
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options requires stream=true on /v1/chat/completions",
        )
    if opts.get("include_usage") is True:
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options.include_usage=true is not supported on /v1/chat/completions",
        )
    if opts.get("include_obfuscation") is True:
        # SSE obfuscation is not applied by this gateway; fail closed.
        raise RequestError(
            400,
            "invalid_stream_options",
            "stream_options.include_obfuscation=true is not supported on /v1/chat/completions",
        )
    return opts


def _reject_unknown_keys(body: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestError(400, "unknown_fields", "request contains unsupported fields", {"fields": unknown})



def _validate_responses_text(body: dict[str, Any]) -> dict[str, Any] | None:
    """Official Responses ``text`` — ``format`` shapes, omit-real optionals.

    Official SDKs send ``text: {format: {type: text}}`` as the default
    structured-output plane (OpenAI, 2024). Accept ``text`` / ``json_object``
    / ``json_schema`` formats, pop JSON-null or blank ``description`` and
    JSON-null ``strict`` so passthrough matches omit, and fail closed on
    unknown keys. ``verbosity`` is not applied: JSON null / blank is popped;
    any other value is ``invalid_text``. ``text`` and ``response_format``
    cannot both be set — accepting the official default must not open a
    dual-plane passthrough. Flat ``json_schema`` ``name`` matches
    ``[a-zA-Z0-9_-]{1,64}`` (ASCII only).
    """
    if "text" not in body:
        return None
    text = body.get("text")
    # Explicit JSON null, empty object, or empty/whitespace string is omit.
    if (
        text is None
        or (isinstance(text, dict) and not text)
        or (isinstance(text, str) and not text.strip())
    ):
        return None
    if not isinstance(text, dict):
        raise RequestError(400, "invalid_text", "text must be an object")
    unknown_text = sorted(set(text) - {"format", "verbosity"})
    if unknown_text:
        raise RequestError(
            400,
            "invalid_text",
            "text accepts only format and verbosity",
            {"fields": unknown_text},
        )
    if "verbosity" in text:
        verbosity = text.get("verbosity")
        if verbosity is None or (isinstance(verbosity, str) and not verbosity.strip()):
            text.pop("verbosity")
        elif isinstance(verbosity, str) and verbosity.strip().lower() in {
            "low",
            "medium",
            "high",
        }:
            # Known OpenAI levels are default-length no-ops (no verbosity plane).
            text["verbosity"] = verbosity.strip().lower()
        else:
            raise RequestError(
                400,
                "invalid_text",
                "text.verbosity must be one of low, medium, high",
            )
    response_format = body.get("response_format")
    response_format_present = not (
        response_format is None
        or (isinstance(response_format, dict) and not response_format)
        or (isinstance(response_format, str) and not response_format.strip())
    )
    if "format" not in text:
        if not text:
            return None
        raise RequestError(
            400,
            "invalid_text",
            "text.format is required when text is provided",
        )
    fmt = text.get("format")
    if (
        fmt is None
        or (isinstance(fmt, dict) and not fmt)
        or (isinstance(fmt, str) and not fmt.strip())
    ):
        text.pop("format", None)
        if not text:
            return None
        raise RequestError(
            400,
            "invalid_text",
            "text.format is required when text is provided",
        )
    if not isinstance(fmt, dict):
        raise RequestError(400, "invalid_text", "text.format must be an object")
    if response_format_present:
        raise RequestError(
            400,
            "invalid_text",
            "text and response_format cannot both be set on /v1/responses; "
            "use official text.format only",
        )
    fmt_type = fmt.get("type")
    # Explicit JSON null or blank type alone is treat-as-omit (SDK optional default).
    if fmt_type is None or (isinstance(fmt_type, str) and not fmt_type.strip()):
        remaining_fmt = {key: value for key, value in fmt.items() if key != "type"}
        if not remaining_fmt:
            text.pop("format", None)
            if not text:
                return None
            raise RequestError(
                400,
                "invalid_text",
                "text.format is required when text is provided",
            )
        raise RequestError(
            400,
            "invalid_text",
            "text.format.type must be one of text, json_object, json_schema",
        )
    # Strip + casefold so " JSON_OBJECT " / "Text" match official types; write back.
    if isinstance(fmt_type, str):
        fmt_type = fmt_type.strip().lower()
        fmt["type"] = fmt_type
    if fmt_type not in ("text", "json_object", "json_schema"):
        raise RequestError(
            400,
            "invalid_text",
            "text.format.type must be one of text, json_object, json_schema",
        )
    if fmt_type in ("text", "json_object"):
        unknown_fmt = sorted(set(fmt) - {"type"})
        if unknown_fmt:
            raise RequestError(
                400,
                "invalid_text",
                f"text.format with type {fmt_type} accepts only the type field",
                {"fields": unknown_fmt},
            )
        return text
    unknown_fmt = sorted(set(fmt) - {"type", "name", "schema", "description", "strict"})
    if unknown_fmt:
        raise RequestError(
            400,
            "invalid_text",
            "text.format json_schema accepts only type, name, schema, description, and strict",
            {"fields": unknown_fmt},
        )
    name = fmt.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestError(
            400,
            "invalid_text",
            "text.format.name must be a non-empty string",
        )
    # Strip incidental whitespace before length/charset (SDK pad).
    name = name.strip()
    fmt["name"] = name
    # OpenAI Structured Outputs: name is [a-zA-Z0-9_-]{1,64}. Fail closed
    # so buyers get invalid_text instead of a provider 400.
    if len(name) > 64:
        raise RequestError(
            400,
            "invalid_text",
            "text.format.name must be at most 64 characters",
        )
    if not name.isascii() or not all(ch.isalnum() or ch in "_-" for ch in name):
        raise RequestError(
            400,
            "invalid_text",
            "text.format.name must match [a-zA-Z0-9_-]",
        )
    schema_body = fmt.get("schema")
    if not isinstance(schema_body, dict):
        raise RequestError(
            400,
            "invalid_text",
            "text.format.schema must be an object",
        )
    if "description" in fmt:
        description_value = fmt.get("description")
        if description_value is None or (
            isinstance(description_value, str) and not description_value.strip()
        ):
            fmt.pop("description")
        elif not isinstance(description_value, str):
            raise RequestError(
                400,
                "invalid_text",
                "text.format.description must be a string when provided",
            )
    if "strict" in fmt:
        strict_value = fmt.get("strict")
        if strict_value is None or (
            isinstance(strict_value, str) and not strict_value.strip()
        ):
            fmt.pop("strict")
        else:
            coerced_strict = _coerce_optional_bool(
                strict_value,
                error_code="invalid_text",
                message="text.format.strict must be a boolean when provided",
            )
            if coerced_strict is None:
                fmt.pop("strict")
            else:
                fmt["strict"] = coerced_strict
    return text


def _validate_responses_conversation_controls(body: dict[str, Any]) -> None:
    """Fail closed on OpenAI conversation-control fields this gateway does not apply.

    ``previous_response_id``, ``conversation``, ``truncation``, and
    ``include`` are real OpenAI Responses controls this gateway does not
    apply. Accepting them as unknown fields yields opaque 400s; named
    unsupported errors let buyers migrate cleanly. Explicit JSON null or
    empty string for string fields is treat-as-omit (SDK optional default).
    Empty include structures remain omit no-ops. Official ``text.format``
    is validated by ``_validate_responses_text`` (OpenAI, 2024).
    """
    def _present_nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    if "previous_response_id" in body and _present_nonempty(body.get("previous_response_id")):
        raise RequestError(
            400,
            "invalid_previous_response_id",
            "previous_response_id is not supported on /v1/responses",
        )
    if "conversation" in body and _present_nonempty(body.get("conversation")):
        raise RequestError(
            400,
            "invalid_conversation",
            "conversation is not supported on /v1/responses",
        )
    if "truncation" in body and _present_nonempty(body.get("truncation")):
        trunc = body.get("truncation")
        # OpenAI truncation auto|disabled are honest no-ops here: this gateway
        # has no multi-turn conversation window to truncate. Other values fail
        # closed so clients never believe an unsupported policy applied.
        if isinstance(trunc, str) and trunc.strip().lower() in {"auto", "disabled"}:
            pass
        else:
            raise RequestError(
                400,
                "invalid_truncation",
                "truncation must be auto or disabled on /v1/responses "
                "(or omit; multi-turn truncation is not applied)",
            )
    if "include" in body:
        include = body.get("include")
        # Explicit JSON null, empty/omit-only array, or empty/whitespace string.
        if (
            include is None
            or (isinstance(include, list) and _is_omit_equivalent_list(include))
            or (isinstance(include, str) and not include.strip())
        ):
            pass
        else:
            raise RequestError(
                400,
                "invalid_include",
                "include is not supported on /v1/responses",
            )
    _validate_responses_text(body)


def _validate_responses_stream_options(body: dict[str, Any]) -> None:
    """Responses ``stream_options`` — not supported (Responses streaming is off).

    OpenAI pairs stream_options with stream=true. This gateway rejects
    stream=true on /v1/responses, so any present stream_options would be a
    silent no-op; fail closed instead. Explicit JSON null (object or *allowed*
    flag values) is treat-as-omit. Unknown keys fail closed even when null so
    clients cannot smuggle unsupported flags past the allow-list via nulls.
    """
    if "stream_options" not in body:
        return
    opts = body.get("stream_options")
    # Explicit JSON null or empty object is treat-as-omit (SDK optional default).
    if opts is None or (isinstance(opts, dict) and not opts):
        return
    if isinstance(opts, dict):
        allowed_flags = {"include_usage", "include_obfuscation"}
        # Reject unknown keys before treating null flags as omit.
        unknown = sorted(set(opts) - allowed_flags)
        if unknown:
            raise RequestError(
                400,
                "invalid_stream_options",
                "stream_options contains unsupported fields",
                {"fields": unknown},
            )
        # Null flag values alone are omit-equivalent (SDK optional defaults).
        non_null = {key: value for key, value in opts.items() if value is not None}
        if not non_null:
            return
        # All-false allowed flags are also omit-equivalent.
        if set(non_null) <= allowed_flags and all(v is False for v in non_null.values()):
            return
    raise RequestError(
        400,
        "invalid_stream_options",
        "stream_options is not supported on /v1/responses (stream is not supported)",
    )


def _validate_mode(mode: Any) -> str:
    # Strip + casefold so " ROUTE " / "Conduct" match official aliases.
    if isinstance(mode, str):
        mode = mode.strip().lower()
    if not isinstance(mode, str) or mode not in ALLOWED_MODES:
        raise RequestError(400, "invalid_mode", "mode must be auto, route, or conduct")
    return mode



def _require_pool_model(
    orchestrator: Any, model_name: str, *, required_capability: str | None = None
) -> None:
    """Fail closed when ``model_name`` is not served by any enabled agent.

    OpenAI clients treat ``model`` as the deployment they paid for. Silently
    answering with a different pool agent hides capacity/routing mismatches.
    """
    agents = getattr(orchestrator, "agents", None) or []
    for agent in agents:
        if getattr(agent, "disabled", False):
            continue
        if getattr(agent, "model", None) == model_name and (
            required_capability is None or required_capability in getattr(agent, "tags", ())
        ):
            return
    raise RequestError(
        400,
        "invalid_model",
        f"model {model_name!r} is not available in the agent pool",
    )



def _validate_message_content_parts(content: list[Any]) -> list[dict[str, Any]]:
    """OpenAI multimodal content-parts array (text + image_url) for vision callers.

    Parts are shape-checked and returned for provider passthrough. Unsupported
    part types fail closed with a named error so clients never believe audio or
    other modalities were processed. Empty/whitespace text and image URLs fail
    closed; bare-string ``image_url`` is normalized to ``{"url": ...}``; optional
    ``detail`` must be auto/low/high when present.
    """
    if not content:
        raise RequestError(
            400,
            "invalid_message_content",
            "multipart content arrays must be non-empty",
        )
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise RequestError(
                400,
                "invalid_message_content",
                "message content part must be an object",
            )
        part_type = part.get("type")
        # Strip + casefold so " TEXT " / "Image_Url" match official part types.
        if isinstance(part_type, str):
            part_type = part_type.strip().lower()
            # Responses-style aliases used by some SDKs on chat histories.
            if part_type in {"input_text", "output_text"}:
                part_type = "text"
            elif part_type == "input_image":
                part_type = "image_url"
            if part.get("type") != part_type:
                part = {**part, "type": part_type}
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise RequestError(
                    400,
                    "invalid_message_content",
                    "text content part requires a string text field",
                )
            if not text.strip():
                raise RequestError(
                    400,
                    "invalid_message_content",
                    "text content part text must be a non-empty string",
                )
            parts.append(part)
        elif part_type == "image_url":
            image_url = part.get("image_url")
            # OpenAI SDKs occasionally send image_url as a bare URL string.
            if isinstance(image_url, str):
                if not image_url.strip():
                    raise RequestError(
                        400,
                        "invalid_message_content",
                        "image_url content part requires a non-empty url string",
                    )
                image_url = {"url": image_url}
                part = {**part, "image_url": image_url}
            if not isinstance(image_url, dict):
                raise RequestError(
                    400,
                    "invalid_message_content",
                    "image_url content part requires image_url.url as a string",
                )
            url = image_url.get("url")
            if not isinstance(url, str) or not url.strip():
                raise RequestError(
                    400,
                    "invalid_message_content",
                    "image_url content part requires image_url.url as a non-empty string",
                )
            if "detail" in image_url:
                detail = image_url.get("detail")
                # Explicit null / empty string: treat as omit (SDK optional default).
                if detail is None or (isinstance(detail, str) and not detail.strip()):
                    cleaned = {key: value for key, value in image_url.items() if key != "detail"}
                    part = {**part, "image_url": cleaned}
                else:
                    if not isinstance(detail, str):
                        raise RequestError(
                            400,
                            "invalid_message_content",
                            "image_url.detail must be a string",
                        )
                    detail_normalized = detail.strip().lower()
                    if detail_normalized not in {"auto", "low", "high"}:
                        raise RequestError(
                            400,
                            "invalid_message_content",
                            "image_url.detail must be one of auto, low, high",
                        )
                    if detail != detail_normalized:
                        part = {
                            **part,
                            "image_url": {**image_url, "detail": detail_normalized},
                        }
            parts.append(part)
        else:
            raise RequestError(
                400,
                "invalid_message_content",
                "content part type must be text or image_url",
            )
    return parts


def _reject_unknown_message_keys(message: dict[str, Any]) -> None:
    """Fail closed on chat message keys outside the OpenAI surface we honor.

    Named ``unknown_message_fields`` (with the key list) beats silent strip on
    the orchestration path or silent smuggle on tools passthrough.
    """
    unknown = sorted(set(message) - ALLOWED_MESSAGE_KEYS)
    if unknown:
        raise RequestError(
            400,
            "unknown_message_fields",
            "message contains unsupported fields",
            {"fields": unknown},
        )


def _validate_chat_message_known_fields(body: dict[str, Any]) -> None:
    """Reject unknown message keys and legacy function role before passthrough."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        # Strip + casefold so "Function" / " FUNCTION " hit the migration reject.
        if isinstance(role, str) and role.strip().lower() == "function":
            raise RequestError(
                400,
                "invalid_message_role",
                "function role is not supported on /v1/chat/completions; use tool instead",
            )
        _reject_unknown_message_keys(message)


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "invalid_message", "messages must be a non-empty array")
    validated: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError(400, "invalid_message", "each message must be an object")
        role = message.get("role")
        content = message.get("content")
        # Form/JS SDKs sometimes send "User" / " Assistant " — casefold + strip.
        if isinstance(role, str):
            role = role.strip().lower()
            message["role"] = role
        if isinstance(role, str) and role == "developer":
            # Newer OpenAI clients send developer in place of system. This
            # gateway has no separate developer plane — alias to system so
            # instructions still apply (parity with common OpenAI gateways).
            role = "system"
            message["role"] = "system"
        if isinstance(role, str) and role == "function":
            # Legacy Completions function-calling role; tool replaces it.
            raise RequestError(
                400,
                "invalid_message_role",
                "function role is not supported on /v1/chat/completions; use tool instead",
            )
        # Named error for unsupported keys — never silent strip or passthrough smuggle.
        _reject_unknown_message_keys(message)
        if not isinstance(role, str) or role not in ALLOWED_MESSAGE_ROLES:
            raise RequestError(400, "invalid_message", "message role or content is invalid")
        # OpenAI assistant tool turns often send content:null with tool_calls; treat
        # explicit JSON null as empty string on assistant/tool (SDK optional default).
        if content is None and role in {"assistant", "tool"}:
            content = ""
        if isinstance(content, list):
            # Vision/omni callers send OpenAI content-parts arrays. Shape-check and
            # passthrough text+image_url; other part types fail closed.
            content = _validate_message_content_parts(content)
        elif not isinstance(content, str):
            raise RequestError(400, "invalid_message", "message role or content is invalid")
        # User/system turns drive the prompt — empty string content is never applied.
        # Multimodal arrays are non-empty after parts validation.
        if role in {"user", "system"} and isinstance(content, str) and not content.strip():
            raise RequestError(
                400,
                "invalid_message_content",
                "user and system message content must be a non-empty string",
            )
        entry: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            # OpenAI tool messages bind results to a prior tool_call via tool_call_id.
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise RequestError(
                    400,
                    "invalid_message",
                    "tool messages require a non-empty tool_call_id string",
                )
            # Strip incidental whitespace so form/JS SDKs that pad IDs still bind.
            tool_call_id = tool_call_id.strip()
            if len(tool_call_id) > 128:
                raise RequestError(
                    400,
                    "invalid_message",
                    "tool_call_id must be at most 128 characters",
                )
            entry["tool_call_id"] = tool_call_id
        if "name" in message:
            # OpenAI optional participant name on system/user/assistant (not tool).
            msg_name = message.get("name")
            # Explicit JSON null or empty/whitespace string is treat-as-omit
            # (SDK optional default / blank participant).
            if msg_name is None or (isinstance(msg_name, str) and not msg_name.strip()):
                pass
            else:
                if role == "tool":
                    raise RequestError(
                        400,
                        "invalid_message_name",
                        "name is not valid on tool role messages",
                    )
                if not isinstance(msg_name, str):
                    raise RequestError(
                        400,
                        "invalid_message_name",
                        "message name must be a non-empty string",
                    )
                # Strip incidental whitespace before length/charset (SDK pad).
                msg_name = msg_name.strip()
                if len(msg_name) > 64:
                    raise RequestError(
                        400,
                        "invalid_message_name",
                        "message name must be at most 64 characters",
                    )
                # OpenAI participant names: [a-zA-Z0-9_-]{1,64}.
                # str.isalnum() alone accepts Unicode letters/digits (café, 名前, ١٢٣).
                if not msg_name.isascii() or not all(
                    ch.isalnum() or ch in "_-" for ch in msg_name
                ):
                    raise RequestError(
                        400,
                        "invalid_message_name",
                        "message name must match [a-zA-Z0-9_-]",
                    )
                entry["name"] = msg_name
        if "refusal" in message:
            # OpenAI assistant refusal plane — null/empty omit; non-empty fails closed
            # (this gateway does not surface or apply refusal content).
            refusal = message.get("refusal")
            if refusal is None or (isinstance(refusal, str) and not refusal.strip()):
                pass
            elif role != "assistant":
                raise RequestError(
                    400,
                    "invalid_message_refusal",
                    "refusal is only valid on assistant messages",
                )
            elif not isinstance(refusal, str):
                raise RequestError(
                    400,
                    "invalid_message_refusal",
                    "refusal must be a string",
                )
            else:
                raise RequestError(
                    400,
                    "invalid_message_refusal",
                    "non-empty refusal is not supported on /v1/chat/completions",
                )
        if "annotations" in message:
            # OpenAI message annotations — null/empty omit; non-empty fails closed.
            annotations = message.get("annotations")
            if annotations is None or (isinstance(annotations, list) and not annotations):
                pass
            else:
                raise RequestError(
                    400,
                    "invalid_message_annotations",
                    "non-empty annotations are not supported on /v1/chat/completions",
                )
        if "audio" in message:
            # OpenAI assistant audio payload — null/empty omit; non-empty fails closed
            # (this text gateway has no speech plane on chat message history).
            audio = message.get("audio")
            if audio is None or (isinstance(audio, dict) and not audio):
                pass
            else:
                raise RequestError(
                    400,
                    "invalid_message_audio",
                    "non-empty message audio is not supported on /v1/chat/completions",
                )
        if "function_call" in message:
            # Legacy assistant function_call on messages — null/empty omit; non-empty
            # fails closed (use tool_calls; body-level function_call is also rejected).
            function_call = message.get("function_call")
            if function_call is None or (isinstance(function_call, dict) and not function_call):
                pass
            else:
                raise RequestError(
                    400,
                    "invalid_message_function_call",
                    "non-empty message function_call is not supported on /v1/chat/completions; "
                    "use tool_calls instead",
                )
        if "weight" in message:
            # OpenAI fine-tune style message weight (0 or 1). Explicit null is
            # treat-as-omit. 0/1 (int/float/digit strings) are honest no-ops
            # (no fine-tune plane here). Other values fail closed so clients
            # never believe weighting applied.
            weight = message.get("weight")
            if weight is None or (isinstance(weight, str) and not weight.strip()):
                pass
            elif isinstance(weight, bool):
                raise RequestError(
                    400,
                    "invalid_message_weight",
                    "message weight must be 0 or 1",
                )
            else:
                coerced_weight = _coerce_optional_int(
                    weight,
                    error_code="invalid_message_weight",
                    message="message weight must be 0 or 1",
                )
                if coerced_weight is not None and coerced_weight not in (0, 1):
                    raise RequestError(
                        400,
                        "invalid_message_weight",
                        "message weight must be 0 or 1",
                    )
        if "prefix" in message:
            # OpenAI partial-assistant / predicted-outputs style prefix flag.
            # null/false (and 0/"false"/"0.0") are honest no-ops; true fails
            # closed (no prefix plane).
            prefix = message.get("prefix")
            if prefix is None or (isinstance(prefix, str) and not prefix.strip()):
                pass
            else:
                coerced_prefix = _coerce_optional_bool(
                    prefix,
                    error_code="invalid_message_prefix",
                    message="message prefix must be a boolean",
                )
                if coerced_prefix is True:
                    raise RequestError(
                        400,
                        "invalid_message_prefix",
                        "message prefix=true is not supported on /v1/chat/completions",
                    )
        validated.append(entry)
    return validated


def _validate_chat_message_audio_function_call(body: dict[str, Any]) -> None:
    """Message-level ``audio`` / ``function_call`` — null/empty omit; else fail closed.

    Runs before tools passthrough so multi-turn histories with SDK-default
    null slots stay honest even when the body is proxied verbatim.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "audio" in message:
            audio = message.get("audio")
            if audio is None or (isinstance(audio, dict) and not audio):
                pass
            else:
                raise RequestError(
                    400,
                    "invalid_message_audio",
                    "non-empty message audio is not supported on /v1/chat/completions",
                )
        if "function_call" in message:
            function_call = message.get("function_call")
            if function_call is None or (isinstance(function_call, dict) and not function_call):
                pass
            else:
                raise RequestError(
                    400,
                    "invalid_message_function_call",
                    "non-empty message function_call is not supported on /v1/chat/completions; "
                    "use tool_calls instead",
                )


def _validate_chat_tool_message_ids(body: dict[str, Any]) -> None:
    """Fail closed on role=tool messages missing a usable tool_call_id.

    Runs before tools passthrough so multi-turn tool results are shape-checked
    even when the body is proxied verbatim to a single provider agent.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise RequestError(
                400,
                "invalid_message",
                "tool messages require a non-empty tool_call_id string",
            )
        # Strip + write back so tools passthrough sees the canonical id.
        tool_call_id = tool_call_id.strip()
        if len(tool_call_id) > 128:
            raise RequestError(
                400,
                "invalid_message",
                "tool_call_id must be at most 128 characters",
            )
        message["tool_call_id"] = tool_call_id


def _validate_chat_logprobs_surface(body: dict[str, Any]) -> None:
    """Fail-closed chat ``logprobs`` / ``top_logprobs`` before any proxy.

    Chat route and tools passthrough do not return token logprobs. Explicit
    JSON null, empty/whitespace string, or zero (int/float/digit or
    whole-float string) on ``top_logprobs`` is treat-as-omit and popped so the
    upstream payload matches an omitted field. ``logprobs=true`` and nonzero
    ``top_logprobs`` stay named 400s even when ``tools`` would otherwise take
    the passthrough return.
    """
    if "logprobs" not in body and "top_logprobs" not in body:
        return
    if "logprobs" in body:
        lp = body.get("logprobs")
        if isinstance(lp, str) and not lp.strip():
            lp = None
        if lp is not None:
            coerced = _coerce_optional_bool(
                lp,
                error_code="invalid_logprobs",
                message="logprobs must be a boolean",
            )
            if coerced is None:
                pass
            elif coerced is True:
                raise RequestError(
                    400,
                    "invalid_logprobs",
                    "logprobs=true is not supported on /v1/chat/completions",
                )
            else:
                body["logprobs"] = False
    if "top_logprobs" in body:
        tlp = body.get("top_logprobs")
        # Explicit JSON null or empty/whitespace string is treat-as-omit.
        if tlp is None or (isinstance(tlp, str) and not tlp.strip()):
            body.pop("top_logprobs", None)
            return
        # Digit / whole-float coerce first; zero is omit-equivalent (no top alts).
        coerced = _coerce_optional_int(
            tlp,
            error_code="invalid_top_logprobs",
            message="top_logprobs must be an integer",
        )
        if coerced is None or coerced == 0:
            body.pop("top_logprobs", None)
            return
        raise RequestError(
            400,
            "invalid_top_logprobs",
            "top_logprobs is not supported on /v1/chat/completions",
        )


def _validate_chat_assistant_tool_calls(body: dict[str, Any]) -> None:
    """OpenAI assistant ``tool_calls`` array shape on chat messages.

    Each entry must be a function tool call with non-empty ``id``,
    ``function.name``, and string ``function.arguments`` (JSON text).
    Explicit JSON null or empty ``tool_calls`` arrays are treat-as-omit.
    Validated before passthrough so multi-turn tool histories fail closed.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "tool_calls" not in message:
            continue
        if message.get("role") != "assistant":
            raise RequestError(
                400,
                "invalid_message",
                "tool_calls is only valid on assistant messages",
            )
        tool_calls = message.get("tool_calls")
        # Explicit JSON null or empty array is treat-as-omit (SDK optional default /
        # no-op history slot). Non-empty arrays are shape-checked below.
        if tool_calls is None or (isinstance(tool_calls, list) and not tool_calls):
            continue
        if not isinstance(tool_calls, list):
            raise RequestError(
                400,
                "invalid_message",
                "tool_calls must be a non-empty array",
            )
        if len(tool_calls) > 128:
            raise RequestError(
                400,
                "invalid_message",
                "tool_calls must contain at most 128 entries",
            )
        for call in tool_calls:
            if not isinstance(call, dict):
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls entry must be an object",
                )
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls entry requires a non-empty id string",
                )
            # Strip incidental whitespace; length after strip (SDK pad).
            call_id = call_id.strip()
            if len(call_id) > 128:
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls id must be at most 128 characters",
                )
            call["id"] = call_id
            call_type = call.get("type")
            # Strip + casefold so "Function" / " FUNCTION " match OpenAI type.
            if isinstance(call_type, str):
                call_type = call_type.strip().lower()
                call["type"] = call_type
            if call_type != "function":
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls entry type must be function",
                )
            function = call.get("function")
            if not isinstance(function, dict):
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls entry requires a function object",
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls function.name must be a non-empty string",
                )
            # Strip before length/charset so " lookup_item " is honest wire form.
            name = name.strip()
            if len(name) > 64:
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls function.name must be at most 64 characters",
                )
            # OpenAI function names: [a-zA-Z0-9_-]{1,64}. Fail closed so buyers
            # get invalid_message instead of a provider 400.
            # str.isalnum() alone accepts Unicode letters/digits (café, 名前, ١٢٣).
            if not name.isascii() or not all(ch.isalnum() or ch in "_-" for ch in name):
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls function.name must match [a-zA-Z0-9_-]",
                )
            function["name"] = name
            arguments = function.get("arguments")
            # Explicit JSON null / missing is treat-as-omit → empty JSON-text.
            # Write back so proxy_completion forwards a string, not JSON null.
            if arguments is None:
                function["arguments"] = ""
                arguments = ""
            # Some SDKs send already-parsed objects/arrays; serialize to JSON text
            # so the OpenAI wire shape (string) is preserved on passthrough.
            elif isinstance(arguments, (dict, list)):
                function["arguments"] = json.dumps(
                    arguments, separators=(",", ":"), ensure_ascii=False
                )
                arguments = function["arguments"]
            if not isinstance(arguments, str):
                raise RequestError(
                    400,
                    "invalid_message",
                    "each tool_calls function.arguments must be a string",
                )


def _validate_openai_metadata(body: dict[str, Any]) -> dict[str, str] | None:
    """OpenAI ``metadata`` — object of string pairs, at most 16 entries.

    Keys must be non-empty (no leading/trailing pad) and ≤64 characters; values
    ≤512 characters. Explicit JSON null values are treat-as-omit for that key
    and written back onto ``body`` so ``proxy_completion`` does not forward
    non-string values. Scalar bool/int/float values coerce to strings (JS SDK
    form encodings often send numbers); objects/arrays fail closed so clients
    cannot store nested junk that cost or observability consumers would drop.
    """
    if "metadata" not in body:
        return None
    metadata = body.get("metadata")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if metadata is None:
        body.pop("metadata", None)
        return None
    if not isinstance(metadata, dict):
        raise RequestError(400, "invalid_metadata", "metadata must be an object")
    if len(metadata) > 16:
        raise RequestError(400, "invalid_metadata", "metadata must contain at most 16 entries")
    validated: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise RequestError(400, "invalid_metadata", "metadata keys must be strings")
        # Empty/whitespace keys are not omit-equivalent attribute names — fail
        # closed so cost/observability consumers never index blank labels.
        if not key.strip():
            raise RequestError(
                400,
                "invalid_metadata",
                "metadata keys must be non-empty strings",
            )
        # Leading/trailing whitespace changes key identity vs strip(); reject so
        # clients cannot smuggle padded labels past exact-key attribution joins.
        if key != key.strip():
            raise RequestError(
                400,
                "invalid_metadata",
                "metadata keys must not include leading or trailing whitespace",
            )
        if len(key) > 64:
            raise RequestError(400, "invalid_metadata", "metadata keys must be at most 64 characters")
        # Explicit JSON null value is treat-as-omit for that key (SDK optional).
        if value is None:
            continue
        if isinstance(value, bool):
            # JSON bool → lowercase OpenAI-style string form.
            value = "true" if value else "false"
        elif type(value) is int:
            value = str(value)
        elif isinstance(value, float):
            # Whole floats stringify compactly (1.0 → "1"); others use str().
            if value.is_integer() and abs(value) <= 2**53:
                value = str(int(value))
            else:
                value = str(value)
        elif not isinstance(value, str):
            raise RequestError(
                400,
                "invalid_metadata",
                "metadata values must be strings (or scalar bool/number)",
            )
        if len(value) > 512:
            raise RequestError(
                400,
                "invalid_metadata",
                "metadata values must be at most 512 characters",
            )
        validated[key] = value
    if not validated:
        if any(value is None for value in metadata.values()):
            body.pop("metadata", None)
        return None
    body["metadata"] = validated
    return validated


def _validate_attribution(attribution: Any) -> dict[str, Any] | None:
    if attribution is None:
        return None
    if not isinstance(attribution, dict):
        raise RequestError(400, "invalid_attribution", "attribution must be an object")
    allowed = set(ATTRIBUTION_DIMENSIONS) | {"provider"}
    unknown = sorted(set(attribution) - allowed)
    if unknown:
        raise RequestError(400, "invalid_attribution", "attribution contains unsupported dimensions", {"fields": unknown})
    # Explicit JSON null or empty/whitespace values are treat-as-omit for each
    # known dimension (SDK optional keys); non-empty values stringify.
    cleaned: dict[str, Any] = {}
    for key, value in attribution.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        cleaned[key] = str(value)
    return cleaned or None


def _validate_routing(routing: Any) -> dict[str, Any] | None:
    """OpenAI-adjacent routing hints for sync vs batch channel selection.

    Fail closed on shape so callers cannot smuggle non-boolean latency flags or
    free-form priority values that RoutingPolicy would silently misread via
    loose coercion (``bool(x)`` / ``str(x)``).
    """
    if routing is None:
        return None
    if not isinstance(routing, dict):
        raise RequestError(400, "invalid_routing", "routing must be an object")
    unknown = sorted(set(routing) - {"channel", "latency_tolerant", "priority"})
    if unknown:
        raise RequestError(400, "invalid_routing", "routing contains unsupported keys", {"fields": unknown})
    channel = routing.get("channel")
    # Explicit JSON null or empty/whitespace is treat-as-omit for optional keys.
    if channel is None or (isinstance(channel, str) and not channel.strip()):
        channel = None
    elif not isinstance(channel, str) or channel.strip().lower() not in {"sync", "batch"}:
        raise RequestError(400, "invalid_routing", "routing.channel must be sync or batch")
    else:
        channel = channel.strip().lower()
    latency_tolerant: bool | None = None
    if "latency_tolerant" in routing:
        # Null/empty omit; bool, int 0/1, and "true"/"false" strings coerce
        # (SDK form/query parity with stream/store).
        latency_tolerant = _coerce_optional_bool(
            routing.get("latency_tolerant"),
            error_code="invalid_routing",
            message="routing.latency_tolerant must be a boolean",
        )
    if "priority" in routing:
        priority = routing.get("priority")
        if priority is None or (isinstance(priority, str) and not priority.strip()):
            pass  # omit
        elif not isinstance(priority, str) or priority.strip().lower() not in {
            "interactive",
            "normal",
            "bulk",
        }:
            raise RequestError(
                400,
                "invalid_routing",
                "routing.priority must be one of interactive, normal, bulk",
            )
    # Rebuild without omitted null optional keys for honest passthrough shape.
    cleaned: dict[str, Any] = {}
    if channel is not None:
        cleaned["channel"] = channel
    if latency_tolerant is not None:
        cleaned["latency_tolerant"] = latency_tolerant
    if "priority" in routing:
        priority = routing.get("priority")
        if isinstance(priority, str) and priority.strip():
            cleaned["priority"] = priority.strip().lower()
    return cleaned if cleaned else {}


def _validate_batch_requests(body: dict[str, Any], expose_trace: bool) -> list[BatchRequest]:
    raw_requests = body.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise RequestError(400, "invalid_request", "requests must be a non-empty array")
    default_attribution = _validate_attribution(body.get("attribution")) or {}
    default_model = str(body.get("model", "contextual-orchestrator"))
    batch: list[BatchRequest] = []
    seen_custom_ids: set[str] = set()
    for item in raw_requests:
        if not isinstance(item, dict):
            raise RequestError(400, "invalid_request", "each batch request must be an object")
        messages = _validate_messages(item.get("messages"))
        attribution = _validate_attribution(item.get("attribution"))
        merged = {**default_attribution, **(attribution or {})}
        mode = _validate_mode(item.get("mode", "auto"))
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": str(item.get("model", default_model)),
            "attribution": merged,
            "mode": mode,
        }
        # Caller-supplied custom_id: without it, results cannot be mapped
        # back to requests on backends that do not preserve submission
        # order (the OpenAI Batch contract explicitly does not), because
        # the submit response never discloses the generated ids. Same
        # bounds as the OpenAI Batch API's custom_id.
        custom_id = item.get("custom_id")
        if custom_id is not None:
            if not isinstance(custom_id, str) or not custom_id.strip():
                raise RequestError(
                    400, "invalid_request", "custom_id must be a non-empty string"
                )
            if len(custom_id) > 64:
                raise RequestError(
                    400, "invalid_request", "custom_id must be at most 64 characters"
                )
            if custom_id in seen_custom_ids:
                raise RequestError(
                    400, "invalid_request", "custom_id values must be unique within a batch"
                )
            seen_custom_ids.add(custom_id)
            kwargs["custom_id"] = custom_id
        batch.append(BatchRequest(**kwargs))
    return batch


def _is_token_id_shaped(value: Any) -> bool:
    """True for numeric token-id shapes (int / whole float), including negatives and bools.

    Used to distinguish failed token sequences from string arrays so clients get a
    named token-id error instead of a string-item error.
    """
    if isinstance(value, bool):
        return True
    if type(value) is int:
        return True
    if type(value) is float and value.is_integer():
        return True
    return False


def _coerce_token_id(value: Any) -> int | None:
    """Coerce one non-negative OpenAI token id, or None if not a valid token id.

    Accepts bare ``int`` and whole floats (``1.0``) from JS/form SDKs. Bools,
    negatives, and non-integral floats are not valid token ids.
    """
    if isinstance(value, bool):
        return None
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is float:
        if value < 0 or not value.is_integer():
            return None
        return int(value)
    return None


def _coerce_embedding_token_sequence(value: Any) -> list[int] | None:
    """Coerce a non-empty list of non-negative token ids, or None if not a sequence."""
    if not isinstance(value, list) or not value:
        return None
    tokens: list[int] = []
    for item in value:
        token_id = _coerce_token_id(item)
        if token_id is None:
            return None
        tokens.append(token_id)
    return tokens


def _is_embedding_token_sequence(value: Any) -> bool:
    """True when value is a non-empty list of non-negative token ids (int/whole float)."""
    return _coerce_embedding_token_sequence(value) is not None


def _embedding_token_sequence_to_text(tokens: list[int]) -> str:
    """Stable text surrogate for token-id inputs on string embedding/completion backends."""
    return "\x1etokens:" + ",".join(str(token) for token in tokens)


def _normalize_embedding_input_item(item: Any) -> str:
    """Normalize one embeddings unit to a non-empty string for the backend."""
    if isinstance(item, str):
        if not item.strip():
            raise RequestError(
                400,
                "invalid_input",
                "each embedding input must be a non-empty string",
            )
        return item
    coerced = _coerce_embedding_token_sequence(item)
    if coerced is not None:
        return _embedding_token_sequence_to_text(coerced)
    raise RequestError(
        400,
        "invalid_input",
        "each embedding input must be a string or array of non-negative token integers",
    )


def _validate_embeddings_inputs(body: dict[str, Any]) -> list[str]:
    """Validate embeddings ``input``/``inputs`` for sync and batch paths.

    Accepts OpenAI shapes:

    - non-empty string
    - non-empty array of non-empty strings
    - non-empty array of non-negative token integers / whole floats (one embedding)
    - non-empty array of token-integer arrays (batch)

    Token arrays are re-encoded to a stable text surrogate for string embedding
    backends. Blank string items fail closed.
    """
    raw = body.get("inputs")
    if raw is None:
        raw = body.get("input")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RequestError(
            400,
            "invalid_input",
            "input/inputs must be a non-empty string, string array, token array, "
            "or array of token arrays",
        )
    # Single token sequence: [1, 2, 3] / [1.0, 2.0] → one embedding unit.
    coerced_tokens = _coerce_embedding_token_sequence(raw)
    if coerced_tokens is not None:
        return [_embedding_token_sequence_to_text(coerced_tokens)]
    # Batch of token sequences: [[1,2],[3]] — first element is a list.
    if isinstance(raw[0], list):
        batch_tokens: list[list[int]] = []
        for item in raw:
            coerced = _coerce_embedding_token_sequence(item)
            if coerced is None:
                raise RequestError(
                    400,
                    "invalid_input",
                    "each embedding input must be a string or array of non-negative "
                    "token integers",
                )
            batch_tokens.append(coerced)
        return [_embedding_token_sequence_to_text(item) for item in batch_tokens]
    # Apparent flat token sequence with invalid ids (negatives/bools/1.5).
    if all(_is_token_id_shaped(item) for item in raw):
        raise RequestError(
            400,
            "invalid_input",
            "token-id inputs must be non-negative integers",
        )
    inputs: list[str] = []
    for item in raw:
        inputs.append(_normalize_embedding_input_item(item))
    return inputs



def _validate_chat_store(body: dict[str, Any]) -> bool | None:
    """Chat Completions ``store`` — strict boolean; ``true`` is not supported.

    OpenAI can persist completions when ``store=true``. This gateway does not
    implement that persistence surface, so ``store=true`` fails closed.
    ``store=false`` and omit remain valid (explicit no-store is honest).
    """
    if "store" not in body:
        return None
    store = body.get("store")
    store = _coerce_optional_bool(
        store, error_code="invalid_store", message="store must be a boolean"
    )
    if store is None:
        return None
    if store is True:
        raise RequestError(
            400,
            "invalid_store",
            "store=true is not supported on /v1/chat/completions",
        )
    return store




def _validate_chat_sampling_and_control_fields(
    body: dict[str, Any],
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Validate chat sampling knobs and fail-closed unsupported controls.

    Must run before tools/response_format ``proxy_completion`` passthrough so
    buyers never receive 200 when invalid or unsupported OpenAI controls would
    only have been checked on the multi-agent route path.
    """
    sampling: dict[str, Any] = {
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "presence_penalty": None,
        "frequency_penalty": None,
    }
    if "temperature" in body:
        sampling["temperature"] = _validate_completions_temperature(body)
    if "top_p" in body:
        sampling["top_p"] = _validate_completions_top_p(body)
    # OpenAI: max_completion_tokens takes precedence over max_tokens.
    if "max_completion_tokens" in body:
        sampling["max_tokens"] = _validate_chat_max_completion_tokens(body)
    elif "max_tokens" in body:
        sampling["max_tokens"] = _validate_completions_max_tokens(body)
    if "presence_penalty" in body:
        sampling["presence_penalty"] = _validate_completions_presence_penalty(body)
    if "frequency_penalty" in body:
        sampling["frequency_penalty"] = _validate_completions_frequency_penalty(body)
    if "seed" in body:
        # Type-check then fail closed: chat route does not apply seed.
        # Explicit JSON null or empty/whitespace string is treat-as-omit.
        seed_raw = body.get("seed")
        if seed_raw is not None and not (
            isinstance(seed_raw, str) and not seed_raw.strip()
        ):
            try:
                _validate_completions_seed(body)
            except RequestError as exc:
                if exc.code == "invalid_seed" and "not supported" in exc.message:
                    raise RequestError(
                        400,
                        "invalid_seed",
                        "seed is not supported on /v1/chat/completions",
                    ) from exc
                raise
            raise RequestError(
                400,
                "invalid_seed",
                "seed is not supported on /v1/chat/completions",
            )
    if "logit_bias" in body:
        # Empty {} is an honest no-op (shared Completions helper).
        # Non-empty maps fail closed with a chat-path message.
        try:
            _validate_completions_logit_bias(body)
        except RequestError as exc:
            if (
                exc.code == "invalid_logit_bias"
                and "not supported" in exc.message
            ):
                raise RequestError(
                    400,
                    "invalid_logit_bias",
                    "logit_bias is not supported on /v1/chat/completions",
                ) from exc
            raise
    if "stop" in body:
        # Explicit JSON null, empty/whitespace string, empty [], or
        # all-whitespace array items is treat-as-omit (SDK optional default).
        stop_val = body.get("stop")
        if isinstance(stop_val, str) and not stop_val.strip():
            stop_val = ""
        if isinstance(stop_val, list):
            stop_val = [
                s for s in stop_val if not (isinstance(s, str) and not s.strip())
            ]
            if not stop_val:
                stop_val = []
        if stop_val is not None and stop_val != [] and stop_val != "":
            try:
                _validate_completions_stop(body)
            except RequestError as exc:
                # Completions helper fails closed with a Completions path message;
                # re-surface for chat with the chat endpoint string.
                if exc.code == "invalid_stop" and "not supported" in exc.message:
                    raise RequestError(
                        400,
                        "invalid_stop",
                        "stop sequences are not supported on /v1/chat/completions",
                    ) from exc
                raise
            raise RequestError(
                400,
                "invalid_stop",
                "stop sequences are not supported on /v1/chat/completions",
            )
    if "n" in body:
        try:
            _validate_completions_n(body)
        except RequestError as exc:
            if exc.code == "invalid_n" and "not supported" in exc.message:
                raise RequestError(
                    400,
                    "invalid_n",
                    "n greater than 1 is not supported on /v1/chat/completions",
                ) from exc
            raise
    if "store" in body:
        _validate_chat_store(body)
    if "modalities" in body:
        _validate_chat_modalities(body)
    if "prediction" in body:
        _validate_chat_prediction(body)
    if "reasoning_effort" in body:
        _validate_chat_reasoning_effort(body)
    if "service_tier" in body:
        _validate_service_tier(body, endpoint_path="/v1/chat/completions")
    if "user" in body:
        _validate_completions_user(body)
    if "stream_options" in body:
        _validate_chat_stream_options(body, stream)
    return sampling


def _validate_completions_tools_surface(body: dict[str, Any]) -> None:
    """Reject chat-era tool fields on legacy Completions with a migration path.

    OpenAI Completions has no tools surface. Clients migrating from chat often
    still send tools/tool_choice. Named unsupported errors beat opaque
    unknown_fields for commercial honesty.

    Honest no-ops (omit-equivalent SDK defaults):
    - empty ``tools: []``
    - empty ``functions: []``
    - ``parallel_tool_calls=false`` / null
    - ``tool_choice`` none/auto/empty-string/empty-object/null
    - ``function_call`` none/auto/empty-string/null

    Non-empty tools/functions, non-default tool_choice/function_call, or
    ``parallel_tool_calls=true`` fail closed with a chat migration path.
    """
    tools = body.get("tools") if "tools" in body else None
    # Empty array and explicit JSON null are omit-equivalent SDK defaults.
    if tools is None or (isinstance(tools, list) and not tools):
        tools_present = False
    else:
        tools_present = "tools" in body

    functions = body.get("functions") if "functions" in body else None
    if functions is None or (isinstance(functions, list) and not functions):
        functions_present = False
    else:
        functions_present = "functions" in body

    parallel = body.get("parallel_tool_calls") if "parallel_tool_calls" in body else None
    if "parallel_tool_calls" in body:
        parallel = _coerce_optional_bool(
            parallel,
            error_code="invalid_parallel_tool_calls",
            message="parallel_tool_calls must be a boolean",
        )
    if parallel is False or parallel is None:
        # false or omit-equivalent SDK defaults (no-ops).
        parallel_present = False
    else:
        parallel_present = True

    def _tool_control_present(key: str) -> bool:
        if key not in body:
            return False
        value = body.get(key)
        # null, empty string, empty object, none/auto (whitespace-padded) are omit-equivalent.
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip().lower()
            if not stripped or stripped in ("none", "auto"):
                return False
        if isinstance(value, dict) and not value:
            return False
        return True

    if (
        tools_present
        or functions_present
        or parallel_present
        or _tool_control_present("tool_choice")
        or _tool_control_present("function_call")
    ):
        raise RequestError(
            400,
            "invalid_tools",
            "tools, tool_choice, functions, function_call, and parallel_tool_calls "
            "are not supported on /v1/completions; use /v1/chat/completions instead",
        )


def _validate_completions_response_format_surface(body: dict[str, Any]) -> None:
    """Reject response_format on legacy Completions with a migration path.

    Structured outputs are a chat/Responses surface. Completions has no
    response_format plane — fail closed so clients migrate to chat.
    """
    if "response_format" in body:
        fmt = body.get("response_format")
        # Explicit JSON null, empty object, or empty/whitespace string is treat-as-omit.
        if (
            fmt is None
            or (isinstance(fmt, dict) and not fmt)
            or (isinstance(fmt, str) and not fmt.strip())
        ):
            return
        raise RequestError(
            400,
            "invalid_response_format",
            "response_format is not supported on /v1/completions; use /v1/chat/completions instead",
        )


def _validate_completions_chat_era_fields_surface(body: dict[str, Any]) -> None:
    """Reject chat-era modalities/prediction/reasoning_effort on Completions.

    Legacy Completions has no multi-modal output, Predicted Outputs, or o-series
    reasoning_effort plane. Named unsupported errors beat opaque unknown_fields
    so clients migrate to /v1/chat/completions.
    Explicit JSON null, empty list/object, or empty/whitespace string is treat-as-omit.
    """
    for key in ("modalities", "prediction", "reasoning_effort"):
        if key not in body:
            continue
        value = body.get(key)
        # Explicit JSON null, empty list/object, nested-omit object, or empty string.
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, dict) and not _non_omit_object_entries(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        # Text-only modalities ["text"] is an honest no-op on this text gateway
        # (parity with chat Completions allowing modalities ["text"]).
        # Strip + casefold so [" TEXT "] matches text-only.
        if key == "modalities" and isinstance(value, list):
            stripped_items = [
                item.strip().lower() if isinstance(item, str) else item for item in value
            ]
            if stripped_items == ["text"]:
                continue
        # Known reasoning_effort levels are default-effort no-ops (no effort plane).
        # Strip + casefold so " NONE " / " Medium " match (chat parity).
        if (
            key == "reasoning_effort"
            and isinstance(value, str)
            and value.strip().lower() in _OPENAI_REASONING_EFFORT_LEVELS
        ):
            body["reasoning_effort"] = value.strip().lower()
            continue
        raise RequestError(
            400,
            "invalid_chat_era_field",
            "modalities, prediction, and reasoning_effort are not supported on "
            "/v1/completions; use /v1/chat/completions instead",
        )


def _validate_completions_store(body: dict[str, Any]) -> bool | None:
    """Legacy Completions ``store`` — strict boolean; ``true`` is not supported.

    OpenAI may persist completions when ``store=true``. This gateway has no
    Completions persistence surface, so ``store=true`` fails closed rather than
    silently ignoring a buyer-visible storage control. ``store=false``/omit stay valid.
    """
    if "store" not in body:
        return None
    store = body.get("store")
    store = _coerce_optional_bool(
        store, error_code="invalid_store", message="store must be a boolean"
    )
    if store is None:
        return None
    if store is True:
        raise RequestError(
            400,
            "invalid_store",
            "store=true is not supported on /v1/completions",
        )
    return store


def _validate_responses_store(body: dict[str, Any]) -> bool | None:
    """Responses API ``store`` — strict boolean; ``true`` is not supported.

    OpenAI may persist Responses when ``store=true``. This gateway's Responses
    path is a single-agent passthrough without a persistence plane, so
    ``store=true`` fails closed rather than silently dropping a buyer-visible
    storage control. ``store=false`` and omit remain valid.
    """
    if "store" not in body:
        return None
    store = body.get("store")
    store = _coerce_optional_bool(
        store, error_code="invalid_store", message="store must be a boolean"
    )
    if store is None:
        return None
    if store is True:
        raise RequestError(
            400,
            "invalid_store",
            "store=true is not supported on /v1/responses",
        )
    return store




# OpenAI o-series reasoning_effort levels. Without an effort plane this gateway
# treats known levels as default-effort no-ops (parity with verbosity low/medium/high).
_OPENAI_REASONING_EFFORT_LEVELS = frozenset(
    {"none", "minimal", "low", "medium", "high"}
)


def _validate_chat_reasoning_effort(body: dict[str, Any]) -> None:
    """Chat Completions ``reasoning_effort`` — known levels are default-effort no-ops.

    OpenAI o-series models accept ``reasoning_effort`` (none/minimal/low/medium/high).
    This gateway never threads the knob into ``ModelClient`` on the orchestration
    path. Known levels are accepted as default-effort no-ops (no effort plane) so
    o-series SDK defaults (often ``medium``) do not 400; unknown values fail closed.
    Explicit JSON null or empty/whitespace string is treat-as-omit.
    """
    if "reasoning_effort" not in body:
        return
    effort = body.get("reasoning_effort")
    if effort is None:
        return
    if isinstance(effort, str):
        # Strip + casefold so " NONE " / " Medium " match known levels.
        stripped = effort.strip().lower()
        if not stripped:
            return
        if stripped in _OPENAI_REASONING_EFFORT_LEVELS:
            body["reasoning_effort"] = stripped
            return
    raise RequestError(
        400,
        "invalid_reasoning_effort",
        "reasoning_effort must be one of none, minimal, low, medium, high "
        "on /v1/chat/completions",
    )






def _non_omit_object_entries(value: dict[str, Any]) -> dict[str, Any]:
    """Drop nested null / blank / empty-object entries (SDK optional defaults).

    Parity with ``web_search_options`` nested omit: when every entry is an omit
    equivalent, the parent object is treat-as-omit rather than a present value.
    """
    return {
        key: item
        for key, item in value.items()
        if item is not None
        and not (isinstance(item, str) and not item.strip())
        and not (isinstance(item, dict) and not item)
    }


def _is_omit_equivalent_list(value: list[Any]) -> bool:
    """True when list is empty or every item is null / blank string."""
    if not value:
        return True
    return all(
        item is None or (isinstance(item, str) and not item.strip()) for item in value
    )


def _validate_chat_audio_web_search_surface(
    body: dict[str, Any],
    *,
    endpoint_path: str = "/v1/chat/completions",
) -> None:
    """Reject ``audio`` / ``web_search_options`` with named migration errors.

    This text gateway has no speech synthesis plane and no web-search tool
    harness on chat or Completions. Named unsupported errors beat opaque
    ``unknown_fields`` so SDK clients can migrate deliberately.
    Explicit JSON null, empty object, or nested-omit-only object is treat-as-omit
    (SDK optional default).
    """
    if "audio" in body:
        audio = body.get("audio")
        # Explicit JSON null, empty object, or object of only nested omit
        # values (voice/format null) is treat-as-omit (SDK optional default).
        if audio is None or (
            isinstance(audio, dict) and not _non_omit_object_entries(audio)
        ):
            pass
        else:
            raise RequestError(
                400,
                "invalid_audio",
                f"audio is not supported on {endpoint_path}",
            )
    if "web_search_options" in body:
        web = body.get("web_search_options")
        # Explicit JSON null, empty object, or nested-omit-only object.
        if web is None or (
            isinstance(web, dict) and not _non_omit_object_entries(web)
        ):
            pass
        elif isinstance(web, dict):
            raise RequestError(
                400,
                "invalid_web_search_options",
                f"web_search_options is not supported on {endpoint_path}",
            )
        else:
            raise RequestError(
                400,
                "invalid_web_search_options",
                f"web_search_options is not supported on {endpoint_path}",
            )



def _validate_tool_resources(body: dict[str, Any], *, endpoint_path: str) -> None:
    """Reject Assistants-style ``tool_resources`` with a named unsupported error.

    OpenAI Assistants/Responses SDKs may send ``tool_resources`` (file_search,
    code_interpreter bindings). This gateway has no tool-resource plane, so any
    non-omit value fails closed. Nested null/blank/empty-object entries are omit.
    """
    if "tool_resources" not in body:
        return
    value = body.get("tool_resources")
    # Explicit JSON null, empty object, or nested-omit-only object is treat-as-omit.
    if value is None or (
        isinstance(value, dict) and not _non_omit_object_entries(value)
    ):
        return
    raise RequestError(
        400,
        "invalid_tool_resources",
        f"tool_resources is not supported on {endpoint_path}",
    )


def _validate_openai_sdk_control_fields(body: dict[str, Any], *, endpoint_path: str) -> None:
    """Reject modern OpenAI SDK control fields not applied on this gateway.

    ``prompt_cache_key``, ``safety_identifier``, ``verbosity``, and ``prompt_cache_retention`` appear in
    recent OpenAI SDK clients. This gateway has no prompt-cache affinity plane,
    no safety-identifier side channel, and no verbosity sampling control — named
    unsupported errors beat opaque ``unknown_fields``.
    """
    def _sdk_control_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    # Explicit JSON null or empty string is treat-as-omit (SDK optional default).
    if "prompt_cache_key" in body and _sdk_control_present(body.get("prompt_cache_key")):
        raise RequestError(
            400,
            "invalid_prompt_cache_key",
            f"prompt_cache_key is not supported on {endpoint_path}",
        )
    if "safety_identifier" in body and _sdk_control_present(body.get("safety_identifier")):
        raise RequestError(
            400,
            "invalid_safety_identifier",
            f"safety_identifier is not supported on {endpoint_path}",
        )
    if "verbosity" in body and _sdk_control_present(body.get("verbosity")):
        verbosity = body.get("verbosity")
        # Known OpenAI verbosity levels are default-length no-ops (no sampling plane).
        if isinstance(verbosity, str) and verbosity.strip().lower() in {
            "low",
            "medium",
            "high",
        }:
            body["verbosity"] = verbosity.strip().lower()
        else:
            raise RequestError(
                400,
                "invalid_verbosity",
                "verbosity must be one of low, medium, high "
                f"on {endpoint_path}",
            )
    if "prompt_cache_retention" in body and _sdk_control_present(body.get("prompt_cache_retention")):
        raise RequestError(
            400,
            "invalid_prompt_cache_retention",
            f"prompt_cache_retention is not supported on {endpoint_path}",
        )



def _validate_chat_reasoning_object(body: dict[str, Any]) -> None:
    """Reject Responses-style ``reasoning`` object on chat Completions.

    OpenAI Responses accepts a ``reasoning`` object; chat Completions uses
    ``reasoning_effort`` (already fail-closed). Clients that send ``reasoning``
    on chat must get a named error, not opaque unknown_fields.
    Explicit JSON null, empty object, or empty/whitespace string is treat-as-omit
    (SDK optional default / stringified empty control).
    """
    if "reasoning" not in body:
        return
    value = body.get("reasoning")
    # Explicit JSON null, empty/nested-omit object, or empty/whitespace string.
    if (
        value is None
        or (isinstance(value, dict) and not _non_omit_object_entries(value))
        or (isinstance(value, str) and not value.strip())
    ):
        return
    raise RequestError(
        400,
        "invalid_reasoning",
        "reasoning is not supported on /v1/chat/completions; use /v1/responses or omit",
    )



def _validate_openai_background(body: dict[str, Any], *, endpoint_path: str) -> bool | None:
    """OpenAI ``background`` — ``false``/omit are honest no-ops; ``true`` fails closed.

    OpenAI may run long jobs asynchronously when ``background=true``. This
    gateway is request-scoped with no background job plane, so ``true`` fails
    closed. ``false`` is a deliberate no-op (SDK defaults often send it).
    """
    if "background" not in body:
        return None
    value = _coerce_optional_bool(
        body.get("background"),
        error_code="invalid_background",
        message="background must be a boolean",
    )
    if value is None:
        return None
    if value is True:
        raise RequestError(
            400,
            "invalid_background",
            f"background=true is not supported on {endpoint_path}",
        )
    return False



def _validate_chat_include_field(body: dict[str, Any], *, endpoint_path: str = "/v1/chat/completions") -> None:
    """Reject OpenAI ``include`` outside Responses (where it is also unsupported).

    Some SDKs send ``include`` on chat/Completions. Named error beats opaque
    unknown_fields so clients know the surface is unsupported here.
    Explicit JSON null, empty array, or empty/whitespace string is treat-as-omit.
    """
    if "include" not in body:
        return
    include = body.get("include")
    # Explicit JSON null, empty/omit-only array, or empty/whitespace string.
    if (
        include is None
        or (isinstance(include, list) and _is_omit_equivalent_list(include))
        or (isinstance(include, str) and not include.strip())
    ):
        return
    raise RequestError(
        400,
        "invalid_include",
        f"include is not supported on {endpoint_path}",
    )


def _validate_completions_reasoning_object(body: dict[str, Any]) -> None:
    """Reject Responses-style ``reasoning`` object on legacy Completions.

    Explicit JSON null, empty object, or empty/whitespace string is treat-as-omit
    (SDK optional default / stringified empty control).
    """
    if "reasoning" not in body:
        return
    value = body.get("reasoning")
    if (
        value is None
        or (isinstance(value, dict) and not _non_omit_object_entries(value))
        or (isinstance(value, str) and not value.strip())
    ):
        return
    raise RequestError(
        400,
        "invalid_reasoning",
        "reasoning is not supported on /v1/completions; use /v1/responses or omit",
    )


def _validate_responses_modalities(body: dict[str, Any]) -> list[str] | None:
    """Responses ``modalities`` — omit or ``["text"]`` only (text gateway)."""
    if "modalities" not in body:
        return None
    modalities = body.get("modalities")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if modalities is None or (isinstance(modalities, str) and not modalities.strip()):
        return None
    if not isinstance(modalities, list):
        raise RequestError(
            400,
            "invalid_modalities",
            "modalities must be a non-empty array of strings",
        )
    # Empty array is omit-equivalent (SDK optional default).
    if not modalities:
        return None
    if any(not isinstance(item, str) for item in modalities):
        raise RequestError(
            400,
            "invalid_modalities",
            "modalities must be a non-empty array of strings",
        )
    # Strip + casefold so [" TEXT "] matches text-only; write back lowercased.
    modalities = [item.strip().lower() for item in modalities]
    if modalities != ["text"]:
        raise RequestError(
            400,
            "invalid_modalities",
            'only modalities ["text"] is supported on /v1/responses',
        )
    body["modalities"] = modalities
    return modalities


def _validate_responses_prediction(body: dict[str, Any]) -> None:
    """Responses ``prediction`` (Predicted Outputs) — not supported on this gateway.

    Explicit JSON null, empty/nested-omit object, or empty/whitespace string is omit.
    """
    if "prediction" not in body:
        return
    value = body.get("prediction")
    if (
        value is None
        or (isinstance(value, dict) and not _non_omit_object_entries(value))
        or (isinstance(value, str) and not value.strip())
    ):
        return
    raise RequestError(
        400,
        "invalid_prediction",
        "prediction is not supported on /v1/responses",
    )


def _validate_chat_modalities(body: dict[str, Any]) -> list[str] | None:
    """Chat Completions ``modalities`` — omit or ``["text"]`` only.

    OpenAI selects output types (text/audio) via modalities. This gateway is
    text-only; non-text modalities fail closed so clients cannot silently
    believe audio (or other) output was applied.
    """
    if "modalities" not in body:
        return None
    modalities = body.get("modalities")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if modalities is None or (isinstance(modalities, str) and not modalities.strip()):
        return None
    if not isinstance(modalities, list):
        raise RequestError(
            400,
            "invalid_modalities",
            "modalities must be a non-empty array of strings",
        )
    # Empty array is omit-equivalent (SDK optional default).
    if not modalities:
        return None
    if any(not isinstance(item, str) for item in modalities):
        raise RequestError(
            400,
            "invalid_modalities",
            "modalities must be a non-empty array of strings",
        )
    # Strip + casefold so [" TEXT "] matches text-only; write back lowercased.
    modalities = [item.strip().lower() for item in modalities]
    if modalities != ["text"]:
        raise RequestError(
            400,
            "invalid_modalities",
            'only modalities ["text"] is supported on /v1/chat/completions',
        )
    body["modalities"] = modalities
    return modalities


def _validate_chat_prediction(body: dict[str, Any]) -> None:
    """Chat Completions ``prediction`` (Predicted Outputs) — not supported.

    OpenAI Predicted Outputs lets clients supply expected completion content for
    latency wins. This gateway does not apply ``prediction`` on the multi-agent
    route path, so any non-empty present value fails closed rather than silently
    ignoring a buyer-visible optimization hint.
    Explicit JSON null, empty object, or empty/whitespace string is treat-as-omit.
    """
    if "prediction" not in body:
        return
    value = body.get("prediction")
    if (
        value is None
        or (isinstance(value, dict) and not _non_omit_object_entries(value))
        or (isinstance(value, str) and not value.strip())
    ):
        return
    raise RequestError(
        400,
        "invalid_prediction",
        "prediction is not supported on /v1/chat/completions",
    )


def _validate_chat_response_format(body: dict[str, Any]) -> dict[str, Any] | None:
    """OpenAI chat ``response_format`` — object with type text/json_object/json_schema.

    Shape is validated before passthrough so malformed payloads fail closed
    rather than reaching a provider with an unusable format object.

    OpenAI type-only forms are strict: ``text`` and ``json_object`` accept only
    the ``type`` key. ``json_schema`` accepts only ``type`` and ``json_schema``.
    Extra sibling keys fail closed so clients cannot smuggle unsupported fields
    into a provider-shaped object that this gateway never interpreted.
    Inside ``json_schema``, ``name`` must match ``[a-zA-Z0-9_-]{1,64}``
    (ASCII only — ``str.isalnum()`` is not sufficient). Nested keys are
    limited to ``name`` / ``schema`` / ``description`` / ``strict``;
    JSON-null or blank ``description`` and JSON-null ``strict`` are popped
    omit-real before passthrough (parity with Responses ``text.format``).
    """
    if "response_format" not in body:
        return None
    fmt = body.get("response_format")
    # Explicit JSON null, empty object, or empty string is treat-as-omit
    # (SDK optional default / stringified empty control).
    if (
        fmt is None
        or (isinstance(fmt, dict) and not fmt)
        or (isinstance(fmt, str) and not fmt.strip())
    ):
        return None
    if not isinstance(fmt, dict):
        raise RequestError(
            400,
            "invalid_response_format",
            "response_format must be an object",
        )
    fmt_type = fmt.get("type")
    # Explicit JSON null or blank type is treat-as-omit when no other payload
    # remains (SDK optional default). Non-empty unknown types still fail closed.
    if fmt_type is None or (isinstance(fmt_type, str) and not fmt_type.strip()):
        remaining = {key: value for key, value in fmt.items() if key != "type"}
        if not remaining:
            return None
        raise RequestError(
            400,
            "invalid_response_format",
            "response_format.type must be one of text, json_object, json_schema",
        )
    # Strip + casefold so " JSON_OBJECT " / "Text" match official types; write back.
    if isinstance(fmt_type, str):
        fmt_type = fmt_type.strip().lower()
        fmt["type"] = fmt_type
    if fmt_type not in ("text", "json_object", "json_schema"):
        raise RequestError(
            400,
            "invalid_response_format",
            "response_format.type must be one of text, json_object, json_schema",
        )
    if fmt_type in ("text", "json_object"):
        # OpenAI: {"type": "json_object"} / {"type": "text"} — no siblings.
        unknown = sorted(set(fmt) - {"type"})
        if unknown:
            raise RequestError(
                400,
                "invalid_response_format",
                f"response_format with type {fmt_type} accepts only the type field",
                {"fields": unknown},
            )
        return fmt
    if fmt_type == "json_schema":
        unknown = sorted(set(fmt) - {"type", "json_schema"})
        if unknown:
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format with type json_schema accepts only type and json_schema",
                {"fields": unknown},
            )
        schema = fmt.get("json_schema")
        if not isinstance(schema, dict):
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema must be an object when type is json_schema",
            )
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema.name must be a non-empty string",
            )
        # Strip incidental whitespace before length/charset (SDK pad).
        name = name.strip()
        schema["name"] = name
        # OpenAI Structured Outputs: name is [a-zA-Z0-9_-]{1,64}. Fail closed
        # so buyers get invalid_response_format instead of a provider 400.
        # str.isalnum() alone accepts Unicode letters/digits (café, 名前, ١٢٣).
        if len(name) > 64:
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema.name must be at most 64 characters",
            )
        if not name.isascii() or not all(ch.isalnum() or ch in "_-" for ch in name):
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema.name must match [a-zA-Z0-9_-]",
            )
        # Nested json_schema accepts only the official Structured Outputs keys.
        # Unknown siblings fail closed so clients cannot smuggle unsupported
        # fields into a provider-shaped object this gateway never interpreted.
        unknown_schema = sorted(
            set(schema) - {"name", "schema", "description", "strict"}
        )
        if unknown_schema:
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema accepts only name, schema, "
                "description, and strict",
                {"fields": unknown_schema},
            )
        # OpenAI requires json_schema.schema as the actual JSON Schema object.
        # Fail closed when missing or non-object so clients cannot silently
        # believe structured-output enforcement applied without a schema body.
        schema_body = schema.get("schema")
        if not isinstance(schema_body, dict):
            raise RequestError(
                400,
                "invalid_response_format",
                "response_format.json_schema.schema must be an object",
            )
        # Explicit JSON null / blank description is omit-equivalent: pop so
        # passthrough matches omit (parity with Responses text.format).
        if "description" in schema:
            description_value = schema.get("description")
            if description_value is None or (
                isinstance(description_value, str) and not description_value.strip()
            ):
                schema.pop("description")
            elif not isinstance(description_value, str):
                raise RequestError(
                    400,
                    "invalid_response_format",
                    "response_format.json_schema.description must be a string "
                    "when provided",
                )
        # Explicit JSON null is omit-equivalent: pop so passthrough matches omit.
        # Bool, int 0/1, whole-float, and string true/false forms coerce.
        if "strict" in schema:
            strict_value = schema.get("strict")
            if strict_value is None or (
                isinstance(strict_value, str) and not strict_value.strip()
            ):
                schema.pop("strict")
            else:
                coerced_strict = _coerce_optional_bool(
                    strict_value,
                    error_code="invalid_response_format",
                    message=(
                        "response_format.json_schema.strict must be a boolean "
                        "when provided"
                    ),
                )
                if coerced_strict is None:
                    schema.pop("strict")
                else:
                    schema["strict"] = coerced_strict
    return fmt


def _omit_null_tool_function_field(
    function: dict[str, Any],
    field_name: str,
    *,
    expected_types: tuple[type, ...],
    error_message: str,
) -> None:
    """Drop a JSON-null optional ``tool.function`` field or fail-closed.

    Official OpenAI SDKs serialize omitted optional fields as JSON ``null``.
    Leaving those keys on the body is not omit-equivalent: ``proxy_completion``
    forwards the request verbatim and several providers reject ``null``
    ``parameters``, ``description``, or ``strict``. Pop the key in place so the
    upstream payload matches an omitted field. Non-null values of the wrong
    type stay ``invalid_tools``.
    """
    if field_name not in function:
        return
    value = function.get(field_name)
    if value is None:
        function.pop(field_name)
        return
    if not isinstance(value, expected_types):
        raise RequestError(400, "invalid_tools", error_message)


def _coerce_tool_function_strict(function: dict[str, Any], *, name_prefix: str) -> None:
    """Null/empty omit; bool and form/JS 0/1/"true" forms coerce for ``strict``."""
    if "strict" not in function:
        return
    strict_value = function.get("strict")
    if strict_value is None or (isinstance(strict_value, str) and not strict_value.strip()):
        function.pop("strict", None)
        return
    coerced_strict = _coerce_optional_bool(
        strict_value,
        error_code="invalid_tools",
        message=f"{name_prefix}.strict must be a boolean when provided",
    )
    if coerced_strict is None:
        function.pop("strict", None)
    else:
        function["strict"] = coerced_strict


def _validate_tool_function_fields(
    function: dict[str, Any],
    *,
    name_prefix: str,
) -> None:
    """Shared name/description/parameters/strict checks for chat or flat tools."""
    unknown_fn = sorted(set(function) - {"name", "description", "parameters", "strict"})
    if unknown_fn:
        raise RequestError(
            400,
            "invalid_tools",
            f"{name_prefix} accepts only name, description, parameters, and strict",
            {"fields": unknown_fn},
        )
    _coerce_tool_function_strict(function, name_prefix=name_prefix)
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestError(
            400,
            "invalid_tools",
            f"{name_prefix}.name must be a non-empty string",
        )
    name = name.strip()
    if len(name) > 64:
        raise RequestError(
            400,
            "invalid_tools",
            f"{name_prefix}.name must be at most 64 characters",
        )
    if not name.isascii() or not all(ch.isalnum() or ch in "_-" for ch in name):
        raise RequestError(
            400,
            "invalid_tools",
            f"{name_prefix}.name must match [a-zA-Z0-9_-]",
        )
    function["name"] = name
    _omit_null_tool_function_field(
        function,
        "parameters",
        expected_types=(dict,),
        error_message=f"{name_prefix}.parameters must be an object",
    )
    _omit_null_tool_function_field(
        function,
        "description",
        expected_types=(str,),
        error_message=f"{name_prefix}.description must be a string when provided",
    )
    description = function.get("description")
    if isinstance(description, str) and len(description) > 1024:
        raise RequestError(
            400,
            "invalid_tools",
            f"{name_prefix}.description must be at most 1024 characters",
        )


def _validate_chat_tools(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """OpenAI chat/Responses ``tools`` — nested or flat function tool objects.

    Empty array is omit-equivalent. Accepts:

    - Chat nested: ``{"type":"function","function":{"name":...}}``
    - Responses flat: ``{"type":"function","name":...,"parameters":...}``

    Shape is preserved for passthrough (nested stays nested; flat stays flat).
    Optional ``description`` / ``parameters`` / ``strict`` nulls are popped.
    """
    if "tools" not in body:
        return None
    tools = body.get("tools")
    # Explicit JSON null is treat-as-omit (SDK optional default).
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError(
            400,
            "invalid_tools",
            "tools must be an array",
        )
    # Empty array: honest no-op (same as omitting tools).
    if not tools:
        return []
    if len(tools) > 128:
        raise RequestError(
            400,
            "invalid_tools",
            "tools must contain at most 128 entries",
        )
    validated: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            raise RequestError(400, "invalid_tools", "each tool must be an object")
        tool_type = item.get("type")
        # Strip + casefold so "Function" / " FUNCTION " match OpenAI type.
        if isinstance(tool_type, str):
            tool_type = tool_type.strip().lower()
            item["type"] = tool_type
        if tool_type != "function":
            raise RequestError(
                400,
                "invalid_tools",
                "each tool type must be function",
            )
        # Responses flat shape: name/parameters at top level (no nested function).
        flat_keys = {"name", "description", "parameters", "strict"} & set(item)
        if "function" in item and flat_keys:
            raise RequestError(
                400,
                "invalid_tools",
                "each tool must use either nested function or flat name/parameters, not both",
            )
        if "function" in item:
            # Chat nested shape: type + function only.
            unknown_tool = sorted(set(item) - {"type", "function"})
            if unknown_tool:
                raise RequestError(
                    400,
                    "invalid_tools",
                    "each tool accepts only type and function fields",
                    {"fields": unknown_tool},
                )
            function = item.get("function")
            if not isinstance(function, dict):
                raise RequestError(
                    400,
                    "invalid_tools",
                    "each tool.function must be an object",
                )
            _validate_tool_function_fields(function, name_prefix="each tool.function")
        elif flat_keys or "name" in item:
            # Official Responses flat function tool.
            unknown_tool = sorted(
                set(item) - {"type", "name", "description", "parameters", "strict"}
            )
            if unknown_tool:
                raise RequestError(
                    400,
                    "invalid_tools",
                    "each flat function tool accepts only type, name, description, "
                    "parameters, and strict",
                    {"fields": unknown_tool},
                )
            # Validate name/description/parameters/strict without treating type as a
            # function field (type stays on the tool object for passthrough).
            function_fields = {
                key: item[key]
                for key in ("name", "description", "parameters", "strict")
                if key in item
            }
            _validate_tool_function_fields(
                function_fields, name_prefix="each tool"
            )
            # Write stripped/coerced fields back onto the flat tool object.
            for key, value in function_fields.items():
                item[key] = value
            for key in ("name", "description", "parameters", "strict"):
                if key not in function_fields and key in item:
                    item.pop(key, None)
        else:
            raise RequestError(
                400,
                "invalid_tools",
                "each tool.function must be an object",
            )
        validated.append(item)
    return validated


def _tool_choice_declared_names(tools: Any) -> set[str]:
    """Collect stripped tool names from nested or flat ``tools`` entries."""
    tool_names: set[str] = set()
    if not isinstance(tools, list):
        return tool_names
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if isinstance(fn, dict):
            tool_name = fn.get("name")
            if isinstance(tool_name, str):
                tool_names.add(tool_name.strip())
        # Responses flat function tools put name at the top level.
        elif isinstance(item.get("name"), str):
            tool_names.add(item["name"].strip())
    return tool_names


def _validate_chat_tool_choice(body: dict[str, Any]) -> str | dict[str, Any] | None:
    """OpenAI chat/Responses ``tool_choice`` — none/auto/required or named function.

    ``none`` / ``auto`` without tools remain honest no-ops. ``required`` demands
    a non-empty ``tools`` array (parity with ``parallel_tool_calls=true``).

    Named selection accepts both wire shapes (shape preserved for passthrough):

    - Chat nested: ``{"type":"function","function":{"name":...}}``
    - Responses flat: ``{"type":"function","name":...}``

    Mixed nested+flat on one object fails closed. The resolved name must match
    a tools entry so clients cannot force a tool the request did not declare.
    """
    if "tool_choice" not in body:
        return None
    choice = body.get("tool_choice")
    # Explicit JSON null, empty object, or empty/whitespace string is
    # treat-as-omit (SDK optional default / stringified empty control).
    if (
        choice is None
        or (isinstance(choice, dict) and not choice)
        or (isinstance(choice, str) and not choice.strip())
    ):
        return None
    if isinstance(choice, str):
        # Strip + casefold so " REQUIRED " / " Auto " match honest controls.
        choice = choice.strip().lower()
        if choice not in ("none", "auto", "required"):
            raise RequestError(
                400,
                "invalid_tool_choice",
                "tool_choice string must be one of none, auto, required",
            )
        # required forces at least one tool call — meaningless without tools.
        # Fail closed (parity with parallel_tool_calls=true) so clients cannot
        # believe tool use was mandated when no tools were declared.
        if choice == "required":
            tools = body.get("tools") if "tools" in body else None
            if not isinstance(tools, list) or not tools:
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice=required requires a non-empty tools array",
                )
        return choice
    if isinstance(choice, dict):
        has_function = "function" in choice
        has_flat_name = "name" in choice
        if has_function and has_flat_name:
            raise RequestError(
                400,
                "invalid_tool_choice",
                "tool_choice must use either nested function or flat name, not both",
            )
        if has_function:
            # Chat nested: {type, function}; extra siblings fail closed.
            unknown = sorted(set(choice) - {"type", "function"})
            if unknown:
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice object accepts only type and function fields",
                    {"fields": unknown},
                )
        elif has_flat_name:
            # Responses flat: {type, name}; extra siblings fail closed.
            unknown = sorted(set(choice) - {"type", "name"})
            if unknown:
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "flat tool_choice object accepts only type and name fields",
                    {"fields": unknown},
                )
        else:
            # type-only or other keys without a name source.
            unknown = sorted(set(choice) - {"type", "function", "name"})
            if unknown:
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice object accepts only type and function fields, "
                    "or type and name for the flat shape",
                    {"fields": unknown},
                )
            raise RequestError(
                400,
                "invalid_tool_choice",
                "tool_choice.function must be an object with a name",
            )
        choice_type = choice.get("type")
        # Strip + casefold so "Function" / " FUNCTION " match OpenAI type.
        if isinstance(choice_type, str):
            choice_type = choice_type.strip().lower()
            choice["type"] = choice_type
        if choice_type != "function":
            raise RequestError(
                400,
                "invalid_tool_choice",
                "tool_choice object type must be function",
            )
        if has_function:
            function = choice.get("function")
            if not isinstance(function, dict):
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice.function must be an object with a name",
                )
            unknown_fn = sorted(set(function) - {"name"})
            if unknown_fn:
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice.function accepts only name",
                    {"fields": unknown_fn},
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice.function.name must be a non-empty string",
                )
            # Strip so padded names match tools[].function.name after tools strip.
            name = name.strip()
            function["name"] = name
            name_error = "tool_choice.function.name must match a tools entry"
        else:
            name = choice.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RequestError(
                    400,
                    "invalid_tool_choice",
                    "tool_choice.name must be a non-empty string",
                )
            name = name.strip()
            choice["name"] = name
            name_error = "tool_choice.name must match a tools entry"
        tool_names = _tool_choice_declared_names(body.get("tools"))
        if name not in tool_names:
            raise RequestError(
                400,
                "invalid_tool_choice",
                name_error,
            )
        return choice
    raise RequestError(
        400,
        "invalid_tool_choice",
        "tool_choice must be a string or object",
    )






def _validate_responses_model(body: dict[str, Any]) -> str:
    """Responses API ``model`` — required non-empty string ≤256 chars.

    OpenAI requires model on Responses. Missing/empty/non-string values fail
    closed so clients cannot hit passthrough with an implicit mock default and
    believe a named deployment was selected. Strip + write back so
    ``proxy_completion`` pool match sees the same id as form/JS padded names.
    """
    model = body.get("model")
    if model is None:
        raise RequestError(400, "invalid_model", "model is required on /v1/responses")
    if not isinstance(model, str) or not model.strip():
        raise RequestError(400, "invalid_model", "model must be a non-empty string")
    model = model.strip()
    if len(model) > 256:
        raise RequestError(400, "invalid_model", "model must be at most 256 characters")
    body["model"] = model
    return model


def _validate_responses_instructions(body: dict[str, Any]) -> str | None:
    """Responses API ``instructions`` — optional non-empty string ≤32000 chars.

    OpenAI system-style instructions for the Responses surface. Explicit
    JSON null or empty/whitespace strings are treat-as-omit and popped so
    ``proxy_completion`` does not forward a blank system prompt. Non-strings
    fail closed so clients cannot ship a silent no-op that looks like a
    configured system prompt.
    """
    if "instructions" not in body:
        return None
    value = body.get("instructions")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if value is None or (isinstance(value, str) and not value.strip()):
        body.pop("instructions", None)
        return None
    if not isinstance(value, str):
        raise RequestError(400, "invalid_instructions", "instructions must be a string")
    if len(value) > 32_000:
        raise RequestError(
            400,
            "invalid_instructions",
            "instructions must be at most 32000 characters",
        )
    return value


def _validate_responses_reasoning(body: dict[str, Any]) -> None:
    """Responses API ``reasoning`` — known effort levels are default-effort no-ops.

    OpenAI Responses accepts a ``reasoning`` object (effort/summary controls).
    This gateway proxies Responses but does not interpret or enforce reasoning
    controls. Known ``effort`` levels (none/minimal/low/medium/high) with blank
    or omit ``summary`` are accepted as default-effort no-ops (chat
    ``reasoning_effort`` parity). Explicit JSON null, empty object, or empty
    string is treat-as-omit. Unknown effort/summary values fail closed.
    """
    if "reasoning" not in body:
        return
    value = body.get("reasoning")
    if (
        value is None
        or (isinstance(value, dict) and not value)
        or (isinstance(value, str) and not value.strip())
    ):
        return
    if isinstance(value, dict):
        # Known effort levels + null/blank summary are default-effort no-ops.
        unknown = sorted(set(value) - {"effort", "summary"})
        if not unknown:
            effort = value.get("effort") if "effort" in value else None
            summary = value.get("summary") if "summary" in value else None
            effort_ok = (
                "effort" not in value
                or effort is None
                or (
                    isinstance(effort, str)
                    and (
                        not effort.strip()
                        or effort.strip().lower() in _OPENAI_REASONING_EFFORT_LEVELS
                    )
                )
            )
            summary_omit = (
                "summary" not in value
                or summary is None
                or (isinstance(summary, str) and not summary.strip())
            )
            if effort_ok and summary_omit:
                if isinstance(effort, str) and effort.strip():
                    value["effort"] = effort.strip().lower()
                    body["reasoning"] = value
                return
    raise RequestError(
        400,
        "invalid_reasoning",
        "reasoning.effort must be one of none, minimal, low, medium, high "
        "on /v1/responses",
    )



def _validate_batch_embeddings_endpoint(body: dict[str, Any]) -> str | None:
    """Batch embeddings ``endpoint`` — optional non-empty string alias ≤256 chars.

    naruon and OpenAI-compatible clients may tag the upstream embeddings route
    (e.g. ``/v1/embeddings``). Explicit JSON null or empty/whitespace string is
    treat-as-omit (SDK optional default). Non-string values fail closed so the
    gateway never records a blank endpoint alias as if a route was selected.
    """
    if "endpoint" not in body:
        return None
    value = body.get("endpoint")
    # Explicit JSON null or empty/whitespace string is treat-as-omit.
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise RequestError(
            400,
            "invalid_endpoint",
            "endpoint must be a non-empty string on /v1/batch/embeddings",
        )
    if len(value) > 256:
        raise RequestError(
            400,
            "invalid_endpoint",
            "endpoint must be at most 256 characters",
        )
    return value


def _validate_embeddings_model(body: dict[str, Any], orchestrator: Any | None = None) -> str:
    """Validate or auto-select an OpenAI embeddings model.

    Strip + write back (parity with chat/Completions/Responses) so padded
    form/JS model names bind to the pool id on every surface. An omitted model
    is resolved by the orchestrator's explicit ``embedding`` capability pool;
    no consumer-side sentinel model is accepted.
    """
    if body.get("model") is None:
        if orchestrator is None:
            raise RequestError(400, "invalid_model", "model is required outside an orchestrator request")
        try:
            model = orchestrator.select_capability_agent("embedding").model
        except (RuntimeError, ValueError) as exc:
            raise RequestError(
                503,
                "embedding_unavailable",
                "no enabled embedding-capable agent is available",
            ) from exc
        body["model"] = model
        return model
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RequestError(400, "invalid_model", "model must be a non-empty string")
    model = model.strip()
    if len(model) > 256:
        raise RequestError(400, "invalid_model", "model must be at most 256 characters")
    body["model"] = model
    return model


def _validate_embeddings_encoding_format(body: dict[str, Any]) -> str | None:
    """OpenAI ``encoding_format`` — omit/null/empty, ``float``, or ``base64``.

    ``float`` (default) returns numeric vectors; ``base64`` returns OpenAI-style
    little-endian float32 base64 strings. Explicit JSON ``null`` or empty
    whitespace string is treat-as-omit. Case-insensitive values are written back
    lowercased.
    """
    if "encoding_format" not in body:
        return None
    value = body.get("encoding_format")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise RequestError(400, "invalid_encoding_format", "encoding_format must be a string")
    # Strip incidental whitespace and casefold so " FLOAT " / "Base64" match.
    value = value.strip().lower()
    if value not in {"float", "base64"}:
        raise RequestError(
            400,
            "invalid_encoding_format",
            'encoding_format must be "float" or "base64"',
        )
    body["encoding_format"] = value
    return value


def _validate_embeddings_dimensions(body: dict[str, Any]) -> None:
    """OpenAI ``dimensions`` — not applied; non-null values fail closed.

    Explicit JSON ``null`` or empty/whitespace string is treat-as-omit. Digit
    strings and whole floats coerce to int for type honesty, then still fail
    closed so clients cannot believe reduced dimensionality was applied.
    """
    if "dimensions" not in body:
        return
    value = body.get("dimensions")
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    # Coerce digit/float forms so type errors surface as invalid_dimensions with
    # the same unsupported message (not a silent string-vs-int split).
    coerced = _coerce_optional_int(
        value,
        error_code="invalid_dimensions",
        message="dimensions must be an integer",
    )
    if coerced is None or coerced == 0:
        # Zero is omit-equivalent (no reduced-dimension request).
        return
    raise RequestError(
        400,
        "invalid_dimensions",
        "dimensions is not supported on embeddings endpoints",
    )


def _encode_embedding_base64(vector: list[Any]) -> str:
    """OpenAI base64 embedding: little-endian float32 binary, ASCII base64."""
    floats = [float(x) for x in vector]
    packed = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(packed).decode("ascii")


def _openai_embeddings_response(
    document: dict[str, Any],
    *,
    model: str,
    encoding_format: str | None = None,
) -> dict[str, Any]:
    """Map batch document vectors to the OpenAI ``/v1/embeddings`` list shape."""
    items = document.get("embeddings") or []
    use_base64 = encoding_format == "base64"
    data = []
    for item in items:
        vector = list(item.get("embedding") or [])
        data.append(
            {
                "object": "embedding",
                "index": int(item.get("index", 0)),
                "embedding": _encode_embedding_base64(vector) if use_base64 else vector,
            }
        )
    total_tokens = int(document.get("total_tokens") or 0)
    return {
        "object": "list",
        "data": data,
        "model": model or document.get("model") or "contextual-orchestrator",
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


def _embeddings_attribution(body: dict[str, Any]) -> dict[str, Any]:
    """Build ledger attribution from the explicit ``attribution`` field merged
    with any attribution dimensions carried inside ``metadata``.

    naruon sends full cost attribution (service, team, group, company, plus the
    provider alias) inside ``metadata`` alongside observability-only keys
    (source, organization_id, user_id). Only recognised dimension keys feed the
    ledger; the rest are ignored here but still accepted.
    """
    attribution = _validate_attribution(body.get("attribution")) or {}
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise RequestError(400, "invalid_request", "metadata must be an object")
    known = set(ATTRIBUTION_DIMENSIONS) | {"provider"}
    merged: dict[str, Any] = {}
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in known and value not in (None, ""):
                merged[key] = str(value)
    # An explicit attribution field wins over metadata-derived dimensions.
    merged.update(attribution)
    return merged


def _strip_trace(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_strip_trace(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _strip_trace(value) for key, value in payload.items() if key != "trace"}
    return payload


def _response_payload(payload: dict[str, Any], include_trace: bool) -> dict[str, Any]:
    safe_payload = redact_value(payload)
    if include_trace:
        return safe_payload
    return _strip_trace(safe_payload)


def responses_sse_body(response: dict[str, Any]) -> str:
    """Frame a completed Responses object as a valid SSE response."""
    sequence = 0
    frames: list[str] = []

    def emit(event_type: str, **values: Any) -> None:
        nonlocal sequence
        payload = {"type": event_type, "sequence_number": sequence, **values}
        sequence += 1
        frames.append(
            f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        )

    in_progress = {**response, "status": "in_progress", "output": []}
    emit("response.created", response=in_progress)
    for output_index, item in enumerate(response.get("output", [])):
        if not isinstance(item, dict):
            continue
        item_in_progress = {**item, "status": "in_progress"}
        emit("response.output_item.added", output_index=output_index, item=item_in_progress)
        if item.get("type") == "message":
            for content_index, part in enumerate(item.get("content", [])):
                if not isinstance(part, dict):
                    continue
                part_in_progress = {**part, "text": ""}
                emit(
                    "response.content_part.added",
                    item_id=item.get("id"),
                    output_index=output_index,
                    content_index=content_index,
                    part=part_in_progress,
                )
                if part.get("type") == "output_text":
                    emit(
                        "response.output_text.delta",
                        item_id=item.get("id"),
                        output_index=output_index,
                        content_index=content_index,
                        delta=part.get("text", ""),
                    )
                    emit(
                        "response.output_text.done",
                        item_id=item.get("id"),
                        output_index=output_index,
                        content_index=content_index,
                        text=part.get("text", ""),
                    )
                emit(
                    "response.content_part.done",
                    item_id=item.get("id"),
                    output_index=output_index,
                    content_index=content_index,
                    part=part,
                )
        elif item.get("type") == "function_call":
            arguments = str(item.get("arguments", "{}"))
            emit(
                "response.function_call_arguments.delta",
                item_id=item.get("id"),
                output_index=output_index,
                delta=arguments,
            )
            emit(
                "response.function_call_arguments.done",
                item_id=item.get("id"),
                output_index=output_index,
                name=item.get("name", ""),
                arguments=arguments,
            )
        emit("response.output_item.done", output_index=output_index, item=item)
    emit("response.completed", response=response)
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def build_server(
    orchestrator: TaskOrchestrator,
    host: str = "127.0.0.1",
    port: int = 8000,
    security: SecurityConfig | None = None,
    clearfolio_url: str | None = None,
    coordinator: CostRoutingCoordinator | None = None,
) -> ThreadingHTTPServer:
    """Build, but do not start, the orchestration HTTP server.

    ``coordinator`` is the cost-review + routing hub. When omitted a default
    one is built around ``orchestrator`` with an in-memory KV config store, so
    every completion is priced, recorded, and sync/batch routed.
    """
    security = security or SecurityConfig()
    security.check_bind(host)
    coordinator = coordinator or CostRoutingCoordinator(orchestrator)
    if clearfolio_url is not None:
        parsed_viewer = urllib.parse.urlparse(clearfolio_url)
        if parsed_viewer.scheme not in {"http", "https"} or not parsed_viewer.netloc:
            raise ValueError("clearfolio_url must be an http(s) URL")
        clearfolio_url = clearfolio_url.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        """Handle authenticated orchestration, administration, and health routes."""

        def do_GET(self) -> None:  # noqa: N802
            """Dispatch GET requests after applying the route's authorization scope."""
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if path == "/openapi.json":
                    self._send(OPENAPI_SPEC)
                    return
                if path == "/healthz":
                    # Unauthenticated liveness probe for containers/orchestrators.
                    self._send({
                        "status": "ok",
                        "service": "contextual-orchestrator",
                        "agent_count": len(orchestrator.agents),
                        "candidate_count": len(orchestrator.candidates),
                        "enabled_agent_count": len(orchestrator.agents),
                        "batch_backend": coordinator.batch_backend.name,
                        "embedding_batch_backend": coordinator.embedding_batch_backend.name,
                        "provider_readiness": "unprobed",
                        "usage_record_count": len(coordinator.ledger.records()),
                    })
                    return
                if path == "/v1/models" or path.startswith("/v1/models/"):
                    # OpenAI model discovery is inference-scope (same bearer as chat).
                    self._authorize("inference")
                    if path == "/v1/models":
                        self._send(orchestrator.list_openai_models())
                        return
                    model_id = urllib.parse.unquote(path[len("/v1/models/") :])
                    if not model_id or "/" in model_id:
                        raise RequestError(400, "invalid_model", "model id path must be a single segment")
                    try:
                        self._send(orchestrator.get_openai_model(model_id))
                    except KeyError:
                        self._send_error(404, "model_not_found", f"model {model_id!r} not found")
                    return
                if path.startswith("/v1/batch/embeddings/"):
                    # Embeddings batch polling is an inference-scope surface, so
                    # it is authorized here before the admin gate below.
                    self._authorize("inference")
                    batch_id = path[len("/v1/batch/embeddings/"):]
                    try:
                        self._send(coordinator.embeddings_batch_document(batch_id))
                    except KeyError:
                        self._send_error(404, "embeddings_batch_not_found", f"embeddings batch {batch_id} not found")
                    return
                self._authorize("admin")
                if path == "/api/v1/cost_attribution_dimensions":
                    self._send({"items": dimension_catalog(), "total_count": len(ATTRIBUTION_DIMENSIONS)})
                    return
                if path == "/api/v1/cost_reports/rollup":
                    dimension = (query.get("dimension") or ["model_name"])[0]
                    start = self._parse_optional_int(query, "start")
                    end = self._parse_optional_int(query, "end")
                    try:
                        self._send(coordinator.cost_report(dimension, start, end))
                    except ValueError as exc:
                        self._send_error(400, "invalid_dimension", str(exc))
                    return
                if path == "/api/v1/llm_usage_records":
                    start = self._parse_optional_int(query, "start")
                    end = self._parse_optional_int(query, "end")
                    records = coordinator.ledger.records(start, end)
                    page_number, page_size = self._parse_paging(query, default_size=50, max_size=500)
                    window = records[(page_number - 1) * page_size : page_number * page_size]
                    self._send({
                        "items": window,
                        "total_count": len(records),
                        "page_number": page_number,
                        "page_size": page_size,
                    })
                    return
                if path.startswith("/api/v1/batch_routing_jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    try:
                        self._send(coordinator.poll_batch(job_id))
                    except KeyError:
                        self._send_error(404, "batch_job_not_found", f"batch job {job_id} not found")
                    return
                if path in ("/", "/admin"):
                    self._send_text(ADMIN_HTML, "text/html; charset=utf-8")
                    return
                if path == "/admin/state":
                    state = orchestrator.admin_state()
                    state["document_viewer"] = (
                        {"provider": "clearfolio", "url": clearfolio_url} if clearfolio_url else None
                    )
                    self._send(_response_payload(state, security.expose_trace_by_default))
                    return
                if path == "/api/v1/agent_pools":
                    page_number, page_size = self._parse_paging(query, default_size=20, max_size=100)
                    items = orchestrator.list_agents(page_number=page_number, page_size=page_size)
                    self._send({
                        "items": items,
                        "total_count": len(orchestrator.candidates),
                        "page_number": page_number,
                        "page_size": page_size,
                    })
                    return
                if path == "/api/v1/orchestration_policies/default_policy":
                    self._send(orchestrator.admin_state()["policy"])
                    return
                if path == "/api/v1/provider_readiness/latest":
                    raw_refresh = (query.get("refresh") or ["false"])[0].lower()
                    if raw_refresh not in {"true", "false"}:
                        raise ValueError("refresh must be true or false")
                    self._send(orchestrator.provider_readiness_report(refresh=raw_refresh == "true"))
                    return
                if path == "/api/v1/analytics_snapshots/latest":
                    self._send(orchestrator.analytics_snapshot(locale_bundles=ADMIN_TRANSLATIONS))
                    return
                if path == "/api/v1/spend_analytics/latest":
                    self._send(orchestrator.spend_analytics())
                    return
                if path == "/api/v1/sales_readiness/latest":
                    self._send(orchestrator.sales_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_readiness/latest":
                    self._send(orchestrator.commercial_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/buyer_evidence_manifests/latest":
                    self._send(orchestrator.buyer_evidence_manifest_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/buyer_handoff_bundles/latest":
                    self._send(orchestrator.buyer_handoff_bundle_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/saleability_decisions/latest":
                    self._send(orchestrator.saleability_decision_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_evidence_exports/latest":
                    self._send(orchestrator.commercial_evidence_export_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_acceptance_checks/latest":
                    self._send(orchestrator.commercial_acceptance_check_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_release_candidates/latest":
                    self._send(orchestrator.commercial_release_candidate_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_gap_registers/latest":
                    self._send(orchestrator.commercial_gap_register_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_procurement_readiness/latest":
                    self._send(orchestrator.commercial_procurement_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_contract_readiness/latest":
                    self._send(orchestrator.commercial_contract_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_onboarding_readiness/latest":
                    self._send(orchestrator.commercial_onboarding_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_operations_readiness/latest":
                    self._send(orchestrator.commercial_operations_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_security_attestations/latest":
                    self._send(orchestrator.commercial_security_attestation_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_value_readiness/latest":
                    self._send(orchestrator.commercial_value_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_close_readiness/latest":
                    self._send(orchestrator.commercial_close_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_go_to_market_readiness/latest":
                    self._send(orchestrator.commercial_go_to_market_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_launch_readiness/latest":
                    self._send(orchestrator.commercial_launch_readiness_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_completion_scorecards/latest":
                    self._send(orchestrator.commercial_completion_scorecard_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_buyer_acceptance_workflows/latest":
                    self._send(orchestrator.commercial_buyer_acceptance_workflow_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_demo_scenarios/latest":
                    self._send(orchestrator.commercial_demo_scenario_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_proposal_packets/latest":
                    self._send(orchestrator.commercial_proposal_packet_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_purchase_approval_packets/latest":
                    self._send(orchestrator.commercial_purchase_approval_packet_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_due_diligence_rooms/latest":
                    self._send(orchestrator.commercial_due_diligence_room_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/commercial_investment_committee_memos/latest":
                    self._send(orchestrator.commercial_investment_committee_memo_report(
                        locale_bundles=ADMIN_TRANSLATIONS,
                        security_profile=security.readiness_profile(),
                    ))
                    return
                if path == "/api/v1/workflow_runs":
                    page_number, page_size = self._parse_paging(query, default_size=20, max_size=200)
                    self._send(_response_payload({
                        "items": orchestrator.list_recent_runs(page_number=page_number, page_size=page_size),
                        "total_count": len(getattr(orchestrator, "_workflow_runs", {})),
                        "page_number": page_number,
                        "page_size": page_size,
                    }, security.expose_trace_by_default))
                    return
                if path.startswith("/api/v1/workflow_runs/"):
                    workflow_run_id = path.rsplit("/", 1)[-1]
                    try:
                        self._send(_response_payload(orchestrator.get_workflow_run(workflow_run_id), security.expose_trace_by_default))
                        return
                    except KeyError:
                        self._send_error(404, "workflow_run_not_found", f"workflow_run {workflow_run_id} not found")
                        return
                if path.startswith("/api/v1/access_reports/"):
                    workflow_run_id = path.rsplit("/", 1)[-1]
                    try:
                        orchestrator.record_analytics_event(
                            "access_report_viewed",
                            {
                                "endpoint_path": "/api/v1/access_reports/{workflow_run_id}",
                                "workflow_run_id": workflow_run_id,
                                "actor_scope": "admin",
                                "status_code": 200,
                            },
                        )
                        self._send(_response_payload(orchestrator.get_access_report(workflow_run_id), security.expose_trace_by_default))
                        return
                    except KeyError:
                        self._send_error(404, "workflow_run_not_found", f"workflow_run {workflow_run_id} not found")
                        return
                if path.startswith("/api/v1/evaluation_runs/"):
                    evaluation_run_id = path.rsplit("/", 1)[-1]
                    runs = getattr(orchestrator, "_evaluation_runs", {})
                    if evaluation_run_id in runs:
                        self._send(_response_payload(runs[evaluation_run_id], security.expose_trace_by_default))
                        return
                    self._send_error(404, "evaluation_run_not_found", f"evaluation_run {evaluation_run_id} not found")
                    return
                if path.startswith("/api/v1/agent_pools/"):
                    segments = [part for part in path.split("/") if part]
                    if len(segments) == 6 and segments[:3] == ["api", "v1", "agent_pools"] and segments[4] == "worker_agents":
                        agent_pool_id = segments[3]
                        worker_agent_id = segments[-1]
                        try:
                            payload = orchestrator._agent_to_admin_payload(orchestrator._agent(worker_agent_id))
                            payload["agent_pool_id"] = agent_pool_id
                            self._send(payload)
                            return
                        except KeyError:
                            self._send_error(404, "agent_not_found", f"agent {worker_agent_id} not found")
                            return
                    raise RequestError(
                        400,
                        "bad_path",
                        "agent path must be /api/v1/agent_pools/{agent_pool_id}/worker_agents/{worker_agent_id}",
                    )
                if path.startswith("/api/v1/locale_bundles/"):
                    locale_code = path.rsplit("/", 1)[-1]
                    bundle = ADMIN_TRANSLATIONS.get(locale_code)
                    if not bundle:
                        self._send_error(404, "locale_not_found", f"locale {locale_code} not found")
                        return
                    orchestrator.record_analytics_event(
                        "locale_bundle_loaded",
                        {
                            "endpoint_path": "/api/v1/locale_bundles/{locale_code}",
                            "locale_code": locale_code,
                            "actor_scope": "admin",
                            "status_code": 200,
                        },
                    )
                    self._send({"locale_code": locale_code, "messages": bundle})
                    return
                self._send_error(404, "route_not_found", "not found")
            except RequestError as exc:
                self._send_error(exc.status, exc.code, exc.message, exc.detail)
            except (TypeError, ValueError) as exc:
                self._send_error(400, "invalid_request", str(exc))
            except Exception:
                self._send_error(500, "internal_error", "internal server error")

        def do_PATCH(self) -> None:  # noqa: N802
            """Apply an authenticated agent-pool worker update."""
            try:
                self._authorize("admin")
                path = urllib.parse.urlparse(self.path).path
                if path.startswith("/api/v1/agent_pools/") and "/worker_agents/" in path:
                    segments = [part for part in path.split("/") if part]
                    if len(segments) != 6 or segments[:3] != ["api", "v1", "agent_pools"] or segments[4] != "worker_agents":
                        raise RequestError(400, "bad_path", "agent patch path missing worker agent")
                    body = self._read_json()
                    _reject_unknown_keys(body, ALLOWED_AGENT_PATCH_KEYS)
                    updated = orchestrator.patch_agent(segments[3], segments[-1], body)
                    self._send(updated, 200)
                    return
                self._send_error(404, "route_not_found", "not found")
            except RequestError as exc:
                self._send_error(exc.status, exc.code, exc.message, exc.detail)
            except (ValueError, TypeError) as exc:
                self._send_error(400, "invalid_request", str(exc))
            except KeyError as exc:
                self._send_error(404, "resource_not_found", str(exc))
            except Exception:
                self._send_error(500, "internal_error", "internal server error")

        def do_DELETE(self) -> None:  # noqa: N802
            """Delete an authenticated agent-pool worker resource."""
            try:
                self._authorize("admin")
                path = urllib.parse.urlparse(self.path).path
                if path.startswith("/api/v1/agent_pools/") and "/worker_agents/" in path:
                    segments = [part for part in path.split("/") if part]
                    if len(segments) != 6 or segments[:3] != ["api", "v1", "agent_pools"] or segments[4] != "worker_agents":
                        raise RequestError(400, "bad_path", "agent delete path missing worker agent")
                    self._send(orchestrator.remove_agent(segments[3], segments[-1]), 200)
                    return
                self._send_error(404, "route_not_found", "not found")
            except RequestError as exc:
                self._send_error(exc.status, exc.code, exc.message, exc.detail)
            except (ValueError, TypeError) as exc:
                self._send_error(400, "invalid_request", str(exc))
            except KeyError as exc:
                self._send_error(404, "resource_not_found", str(exc))
            except Exception:
                self._send_error(500, "internal_error", "internal server error")

        def do_POST(self) -> None:  # noqa: N802
            """Dispatch authenticated completion, agent, and simulation writes."""
            try:
                path = urllib.parse.urlparse(self.path).path
                scope = "admin" if path == "/admin/simulate" or path.startswith("/api/v1/agent_pools/") else "inference"
                self._authorize(scope)
                body = self._read_json()

                if path.startswith("/api/v1/agent_pools/") and path.endswith("/worker_agents"):
                    segments = [part for part in path.split("/") if part]
                    if len(segments) != 5 or segments[:3] != ["api", "v1", "agent_pools"]:
                        raise RequestError(400, "bad_path", "agent create path must be /api/v1/agent_pools/{pool}/worker_agents")
                    _reject_unknown_keys(body, ALLOWED_AGENT_CREATE_KEYS)
                    if _is_local_provider_url(str(body.get("base_url", ""))):
                        raise RequestError(
                            400,
                            "local_provider_registration_forbidden",
                            "Configure local providers at trusted process startup instead.",
                        )
                    self._send(orchestrator.add_agent(segments[3], body), 201)
                    return

                if path == "/v1/completions":
                    # Legacy OpenAI Completions: prompt → route → text_completion.
                    _reject_unknown_keys(body, ALLOWED_COMPLETIONS_KEYS)
                    _validate_completions_tools_surface(body)
                    _validate_completions_response_format_surface(body)
                    _validate_completions_chat_era_fields_surface(body)
                    _validate_chat_audio_web_search_surface(
                        body, endpoint_path="/v1/completions"
                    )
                    _validate_openai_sdk_control_fields(body, endpoint_path="/v1/completions")
                    _validate_tool_resources(body, endpoint_path="/v1/completions")
                    _validate_max_tool_calls(body, endpoint_path="/v1/completions")
                    _validate_completions_reasoning_object(body)
                    _validate_openai_background(body, endpoint_path="/v1/completions")
                    _validate_chat_include_field(body, endpoint_path="/v1/completions")
                    _validate_completions_stream(body)
                    _validate_completions_stream_options(body)
                    _validate_completions_best_of(body)
                    _validate_completions_echo(body)
                    _validate_completions_suffix(body)
                    _validate_completions_logprobs(body)
                    _validate_completions_top_logprobs(body)
                    # OpenAI chat-era clients sometimes send max_completion_tokens
                    # on Completions; prefer it over legacy max_tokens when both set.
                    if "max_completion_tokens" in body:
                        max_tokens = _validate_chat_max_completion_tokens(body)
                    else:
                        max_tokens = _validate_completions_max_tokens(body)
                    model_name = _validate_completions_model(body)
                    _require_pool_model(orchestrator, model_name)
                    if "store" in body:
                        _validate_completions_store(body)
                    top_p = _validate_completions_top_p(body)
                    temperature = _validate_completions_temperature(body)
                    presence_penalty = _validate_completions_presence_penalty(body)
                    frequency_penalty = _validate_completions_frequency_penalty(body)
                    _validate_completions_seed(body)
                    _validate_completions_stop(body)
                    _validate_completions_n(body)
                    end_user_id = _validate_completions_user(body)
                    _validate_completions_logit_bias(body)
                    _validate_service_tier(body, endpoint_path="/v1/completions")
                    if "metadata" in body:
                        _validate_openai_metadata(body)
                    if "prompt" not in body:
                        raise RequestError(400, "invalid_prompt", "prompt is required")
                    messages = _validate_completion_prompt(body.get("prompt"))
                    attribution = _validate_attribution(body.get("attribution"))
                    attribution = dict(attribution or {})
                    # OpenAI ``user`` → cost-ledger account when attribution.account is unset.
                    if end_user_id is not None and not attribution.get("account"):
                        attribution["account"] = end_user_id
                    # Request model id → model_name dimension when unset (cost rollups).
                    if model_name and not attribution.get("model_name"):
                        attribution["model_name"] = model_name
                    # Endpoint product surface → service dimension when unset.
                    if not attribution.get("service"):
                        attribution["service"] = "completions_api"
                    routing = _validate_routing(body.get("routing"))
                    started_at = time.perf_counter()
                    model_client = orchestrator.client
                    with model_client.request_options(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                    ):
                        result = self._run(lambda: coordinator.complete(
                            messages,
                            mode="route",
                            attribution=attribution,
                            hints=routing,
                            model_name=model_name,
                            workflow_run_id=f"run_{uuid.uuid4().hex}",
                        ))
                    # Batch-channel Completions return a job handle (202), not a
                    # text_completion body — match chat Completions honesty so
                    # clients never receive a 500 on a valid batch routing hint.
                    if isinstance(result, dict) and result.get("channel") == "batch":
                        orchestrator.record_analytics_event(
                            "text_completion_batched",
                            {
                                "endpoint_path": "/v1/completions",
                                "actor_scope": "inference",
                                "status_code": 202,
                                "batch_job_id": result.get("job_id"),
                                "batch_backend": result.get("backend"),
                                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            },
                        )
                        self._send(result, 202)
                        return
                    orchestrator.record_analytics_event(
                        "text_completion_requested",
                        {
                            "endpoint_path": "/v1/completions",
                            "actor_scope": "inference",
                            "status_code": 200,
                            "run_mode": "route",
                            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        },
                    )
                    self._send(text_completion_response(
                        result, model=model_name, usage=result.get("usage"),
                    ))
                    return
                if path == "/v1/chat/completions":
                    _reject_unknown_keys(body, ALLOWED_CHAT_KEYS)
                    _validate_chat_audio_web_search_surface(body)
                    _validate_openai_sdk_control_fields(body, endpoint_path="/v1/chat/completions")
                    _validate_tool_resources(body, endpoint_path="/v1/chat/completions")
                    _validate_chat_reasoning_object(body)
                    _validate_openai_background(body, endpoint_path="/v1/chat/completions")
                    _validate_chat_include_field(body)
                    _validate_max_tool_calls(body, endpoint_path="/v1/chat/completions")
                    # functions/function_call: null or empty functions[] are omit no-ops
                    # (SDK optional defaults); non-empty or any function_call fail closed.
                    functions_raw = body.get("functions") if "functions" in body else None
                    function_call_raw = body.get("function_call") if "function_call" in body else None
                    functions_present = (
                        "functions" in body
                        and functions_raw is not None
                        and not (isinstance(functions_raw, list) and not functions_raw)
                    )
                    # function_call none/auto/empty-string (whitespace-padded) without functions
                    # are omit-equivalent no-ops; any other function_call or non-empty functions
                    # fail closed.
                    function_call_present = (
                        "function_call" in body
                        and function_call_raw is not None
                        and not (
                            isinstance(function_call_raw, str)
                            and (
                                not function_call_raw.strip()
                                or function_call_raw.strip().lower() in ("none", "auto")
                            )
                        )
                    )
                    if functions_present or function_call_present:
                        # OpenAI deprecated functions/function_call in favor of tools/tool_choice.
                        # Fail closed with a migration message rather than silent passthrough of
                        # a deprecated surface clients may still send from old SDKs.
                        raise RequestError(
                            400,
                            "invalid_functions",
                            "functions and function_call are not supported on /v1/chat/completions; "
                            "use tools and tool_choice instead",
                        )
                    tools_list = body.get("tools") if isinstance(body.get("tools"), list) else None
                    # tool_choice null is omit-equivalent; alone / empty tools: only "none" is a valid no-op.
                    if (
                        "tool_choice" in body
                        and body.get("tool_choice") is not None
                        and not tools_list
                    ):
                        tc = body.get("tool_choice")
                        tc_norm = tc.strip().lower() if isinstance(tc, str) else tc
                        # none/auto/empty-object/empty-string without tools are omit-equivalent no-ops.
                        if (
                            tc_norm not in ("none", "auto")
                            and not (isinstance(tc, dict) and not tc)
                            and not (isinstance(tc, str) and not tc.strip())
                        ):
                            raise RequestError(
                                400,
                                "invalid_tool_choice",
                                "tool_choice requires tools on /v1/chat/completions",
                            )
                    # Shape-check tool results and message audio/function_call before
                    # passthrough or orchestration (named errors, not silent drop).
                    _validate_chat_message_known_fields(body)
                    _validate_chat_tool_message_ids(body)
                    _validate_chat_assistant_tool_calls(body)
                    _validate_chat_message_audio_function_call(body)
                    _validate_chat_logprobs_surface(body)
                    if "metadata" in body:
                        _validate_openai_metadata(body)
                    if "response_format" in body:
                        _validate_chat_response_format(body)
                    if "tools" in body:
                        _validate_chat_tools(body)
                    if "tool_choice" in body:
                        _validate_chat_tool_choice(body)
                    if "parallel_tool_calls" in body:
                        # Always type-check. With tools, true/false both valid for
                        # provider passthrough; without tools, true fails closed.
                        # Explicit JSON null is treat-as-omit (SDK optional default).
                        ptc = body.get("parallel_tool_calls")
                        ptc = _coerce_optional_bool(
                            ptc,
                            error_code="invalid_parallel_tool_calls",
                            message="parallel_tool_calls must be a boolean",
                        )
                        if ptc is not None:
                            if ptc is True and not tools_list:
                                raise RequestError(
                                    400,
                                    "invalid_parallel_tool_calls",
                                    "parallel_tool_calls=true requires tools on /v1/chat/completions",
                                )
                            body["parallel_tool_calls"] = ptc
                    # Strip+writeback model before tools/response_format passthrough so
                    # proxy_completion pool match sees the same id as form/JS padded names.
                    _validate_completions_model(body)
                    # Coerce stream early so stream_options fail-closed matches route path
                    # and tools/response_format passthrough cannot skip type checks.
                    stream = body.get("stream", False)
                    if stream is None or (isinstance(stream, str) and not stream.strip()):
                        stream = False
                    else:
                        coerced_stream = _coerce_optional_bool(
                            stream,
                            error_code="invalid_request",
                            message="stream must be a boolean",
                        )
                        stream = False if coerced_stream is None else coerced_stream
                    body["stream"] = stream
                    # Sampling + unsupported controls before passthrough (honesty parity
                    # with the multi-agent route path).
                    sampling = _validate_chat_sampling_and_control_fields(
                        body, stream=bool(stream)
                    )
                    temperature = sampling["temperature"]
                    top_p = sampling["top_p"]
                    max_tokens = sampling["max_tokens"]
                    presence_penalty = sampling["presence_penalty"]
                    frequency_penalty = sampling["frequency_penalty"]
                    # Explicit JSON null on trigger keys is omit-equivalent (SDK optional
                    # defaults) — do not force single-agent passthrough for null-only keys.
                    if any(
                        key in body and body.get(key) is not None
                        for key in PASSTHROUGH_TRIGGER_KEYS
                    ):
                        # response_format / tools cannot be merged across agents;
                        # proxy the full request to one agent and return it verbatim.
                        started_at = time.perf_counter()
                        proxied = self._run(
                            lambda: orchestrator.proxy_completion(body, endpoint="chat/completions")
                        )
                        orchestrator.record_analytics_event(
                            "chat_completion_passthrough",
                            {
                                "endpoint_path": "/v1/chat/completions",
                                "actor_scope": "inference",
                                "status_code": 200,
                                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            },
                        )
                        self._send(proxied)
                        return
                    messages = _validate_messages(body.get("messages"))
                    mode = _validate_mode(body.get("orchestration") or body.get("orchestration_mode") or body.get("mode") or "auto")
                    if "include_orchestration_trace" in body:
                        # Null/empty omit; bool, int 0/1, and "true"/"false"/"0"/"1"
                        # strings coerce (SDK form/query parity with stream/store).
                        coerced_trace = _coerce_optional_bool(
                            body.get("include_orchestration_trace"),
                            error_code="invalid_include_orchestration_trace",
                            message="include_orchestration_trace must be a boolean",
                        )
                        if coerced_trace is None:
                            include_trace = bool(security.expose_trace_by_default)
                        else:
                            include_trace = coerced_trace
                    else:
                        include_trace = bool(security.expose_trace_by_default)
                    # stream + stream_options already coerced/validated before passthrough.
                    attribution = _validate_attribution(body.get("attribution"))
                    routing = _validate_routing(body.get("routing"))
                    # Require model — silent default to contextual-orchestrator hid
                    # which deployment the buyer selected on the chat Completions path.
                    model_name = _validate_completions_model(body)
                    _require_pool_model(orchestrator, model_name)
                    attribution = dict(attribution or {})
                    # OpenAI chat ``user`` → account when unset.
                    # Same fail-closed rules as Completions: present key must be a
                    # non-empty string ≤64 chars (null omit; scalars coerce; empty reject).
                    end_user_id = _validate_completions_user(body)
                    if end_user_id is not None and not attribution.get("account"):
                        attribution["account"] = end_user_id
                    if model_name and not attribution.get("model_name"):
                        attribution["model_name"] = model_name
                    if not attribution.get("service"):
                        attribution["service"] = "chat_completions_api"
                    # sampling/controls already validated before passthrough branch.
                    if "metadata" in body:
                        _validate_openai_metadata(body)
                    started_at = time.perf_counter()
                    model_client = orchestrator.client
                    with model_client.request_options(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                    ):
                        if stream and orchestrator.would_route(messages, mode):
                            self._stream_route_completion(orchestrator, security, messages, model_name)
                            orchestrator.record_analytics_event(
                                "chat_completion_requested",
                                {
                                    "endpoint_path": "/v1/chat/completions",
                                    "actor_scope": "inference",
                                    "status_code": 200,
                                    "run_mode": "route",
                                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                                    "response_streamed": True,
                                },
                            )
                            return
                        result = self._run(lambda: coordinator.complete(
                            messages,
                            mode=mode,
                            attribution=attribution,
                            hints=routing,
                            model_name=model_name,
                            workflow_run_id=f"run_{uuid.uuid4().hex}",
                        ))
                    # Latency-tolerant requests get dispatched to the batch backend.
                    if result.get("channel") == "batch":
                        orchestrator.record_analytics_event(
                            "chat_completion_batched",
                            {
                                "endpoint_path": "/v1/chat/completions",
                                "actor_scope": "inference",
                                "status_code": 202,
                                "batch_job_id": result["job_id"],
                                "batch_backend": result["backend"],
                            },
                        )
                        self._send(result, 202)
                        return
                    orchestrator.record_analytics_event(
                        "chat_completion_requested",
                        {
                            "endpoint_path": "/v1/chat/completions",
                            "actor_scope": "inference",
                            "status_code": 200,
                            "run_mode": result["mode"],
                            "workflow_run_id": result["workflow_run_id"],
                            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            "response_streamed": stream,
                        },
                    )
                    if stream:
                        chunks = chat_completion_chunks(result, model=model_name, include_trace=include_trace)
                        self._send_sse(sse_stream_body(chunks))
                        return
                    self._send(chat_completion_response(
                        result, model=model_name, include_trace=include_trace, usage=result.get("usage"),
                    ))
                    return
                if path == "/v1/embeddings":
                    # OpenAI sync embeddings: input → vectors as list object.
                    # Reuses the embedding batch backend (local path completes
                    # synchronously) and frames an OpenAI-shaped response so
                    # SDKs that call /v1/embeddings work without the batch path.
                    _reject_unknown_keys(body, ALLOWED_EMBEDDINGS_KEYS)
                    model_name = _validate_embeddings_model(body, orchestrator)
                    # Same pool honesty as chat/Completions: do not silently serve
                    # a different embedding deployment than the client requested.
                    _require_pool_model(orchestrator, model_name, required_capability="embedding")
                    encoding_format = _validate_embeddings_encoding_format(body)
                    _validate_embeddings_dimensions(body)
                    end_user_id = _validate_completions_user(body)
                    if "routing" in body:
                        routing = _validate_routing(body.get("routing"))
                        # Sync embeddings has no batch channel job plane.
                        if routing and routing.get("channel") == "batch":
                            raise RequestError(
                                400,
                                "invalid_routing",
                                "routing.channel=batch is not supported on /v1/embeddings; use /v1/batch/embeddings",
                            )
                        if routing and routing.get("latency_tolerant") is True:
                            raise RequestError(
                                400,
                                "invalid_routing",
                                "routing.latency_tolerant=true is not supported on /v1/embeddings; use /v1/batch/embeddings",
                            )
                    if "metadata" in body and not isinstance(body.get("metadata"), dict):
                        # OpenAI-shaped string metadata is preferred for this
                        # surface; non-objects fail closed before attribution merge.
                        raise RequestError(400, "invalid_metadata", "metadata must be an object")
                    if "metadata" in body:
                        # When all values are strings, enforce OpenAI ≤16 pairs;
                        # naruon-style attribution-in-metadata still uses
                        # _embeddings_attribution below for known dimensions.
                        meta = body.get("metadata") or {}
                        if meta and all(isinstance(v, str) for v in meta.values()):
                            _validate_openai_metadata(body)
                    if "input" not in body and "inputs" not in body:
                        # OpenAI only documents ``input``; accept nothing else.
                        raise RequestError(400, "invalid_input", "input is required on /v1/embeddings")
                    # Prefer OpenAI ``input``; do not accept ``inputs`` on this path
                    # (batch endpoint owns ``inputs``) so clients get a clear split.
                    if "inputs" in body and "input" not in body:
                        raise RequestError(
                            400,
                            "invalid_input",
                            "use input on /v1/embeddings; inputs is only for /v1/batch/embeddings",
                        )
                    inputs = _validate_embeddings_inputs({"input": body.get("input")})
                    attribution = _embeddings_attribution(body)
                    attribution = dict(attribution or {})
                    if end_user_id is not None and not attribution.get("account"):
                        attribution["account"] = end_user_id
                    if model_name and not attribution.get("model_name"):
                        attribution["model_name"] = model_name
                    if not attribution.get("service"):
                        attribution["service"] = "embeddings_api"
                    started_at = time.perf_counter()
                    document = self._run(lambda: coordinator.complete_embeddings_batch(
                        inputs,
                        model=model_name,
                        attribution=attribution,
                        metadata={"actor_scope": "inference", "endpoint_alias": "embeddings"},
                    ))
                    if document.get("status") != "completed" or document.get("embeddings") is None:
                        # Async backends return a job handle; fail closed on the
                        # sync OpenAI path rather than inventing vectors.
                        raise RequestError(
                            503,
                            "embeddings_unavailable",
                            "sync /v1/embeddings is unavailable for this backend; use /v1/batch/embeddings",
                        )
                    orchestrator.record_analytics_event(
                        "embeddings_requested",
                        {
                            "endpoint_path": "/v1/embeddings",
                            "actor_scope": "inference",
                            "status_code": 200,
                            "input_count": len(inputs),
                            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        },
                    )
                    self._send(
                        _openai_embeddings_response(
                            document,
                            model=model_name,
                            encoding_format=encoding_format,
                        )
                    )
                    return
                if path == "/v1/batch/embeddings":
                    _reject_unknown_keys(body, ALLOWED_EMBEDDINGS_BATCH_KEYS)
                    inputs = _validate_embeddings_inputs(body)
                    model_name = _validate_embeddings_model(body, orchestrator)
                    _require_pool_model(orchestrator, model_name, required_capability="embedding")
                    _validate_embeddings_encoding_format(body)
                    _validate_embeddings_dimensions(body)
                    # OpenAI ``user`` end-user id — same fail-closed shape as sync embeddings.
                    end_user_id = _validate_completions_user(body)
                    if "routing" in body:
                        _validate_routing(body.get("routing"))
                    attribution = _embeddings_attribution(body)
                    attribution = dict(attribution or {})
                    if end_user_id is not None and not attribution.get("account"):
                        attribution["account"] = end_user_id
                    if model_name and not attribution.get("model_name"):
                        attribution["model_name"] = model_name
                    if not attribution.get("service"):
                        attribution["service"] = "embeddings_batch_api"
                    submit_metadata: dict[str, Any] = {"actor_scope": "inference"}
                    endpoint_alias = _validate_batch_embeddings_endpoint(body)
                    if endpoint_alias is not None:
                        submit_metadata["endpoint_alias"] = endpoint_alias
                    document = self._run(lambda: coordinator.complete_embeddings_batch(
                        inputs,
                        model=model_name,
                        attribution=attribution,
                        metadata=submit_metadata,
                    ))
                    is_complete = document.get("status") == "completed"
                    orchestrator.record_analytics_event(
                        "embeddings_batch_created",
                        {
                            "endpoint_path": "/v1/batch/embeddings",
                            "actor_scope": "inference",
                            "status_code": 200 if is_complete else 202,
                            "batch_id": document.get("batch_id"),
                            "batch_backend": document.get("backend"),
                            "input_count": len(inputs),
                        },
                    )
                    self._send(document, 200 if is_complete else 202)
                    return
                if path == "/api/v1/batch_routing_jobs":
                    _reject_unknown_keys(body, ALLOWED_BATCH_KEYS)
                    batch_requests = _validate_batch_requests(body, security.expose_trace_by_default)
                    metadata = {"actor_scope": "inference"}
                    job = self._run(lambda: coordinator.submit_batch(batch_requests, metadata=metadata))
                    orchestrator.record_analytics_event(
                        "batch_routing_job_created",
                        {
                            "endpoint_path": "/api/v1/batch_routing_jobs",
                            "actor_scope": "inference",
                            "status_code": 201,
                            "batch_job_id": job.job_id,
                            "batch_backend": job.backend,
                            "request_count": job.request_count,
                        },
                    )
                    self._send({
                        "job_id": job.job_id,
                        "backend": job.backend,
                        "status": job.status,
                        "request_count": job.request_count,
                    }, 201)
                    return
                if path.startswith("/api/v1/batch_routing_jobs/") and path.endswith("/results"):
                    job_id = path[len("/api/v1/batch_routing_jobs/"):-len("/results")]
                    try:
                        retrieved = self._run(lambda: coordinator.retrieve_batch(job_id))
                    except KeyError:
                        self._send_error(404, "batch_job_not_found", f"batch job {job_id} not found")
                        return
                    self._send(_response_payload(retrieved, include_trace=True))
                    return
                if path == "/v1/responses":
                    # The Responses API has no chat-completions verifier equivalent,
                    # so every request is proxied to one agent verbatim.
                    _reject_unknown_keys(body, ALLOWED_RESPONSES_KEYS)
                    # Fail-closed shape checks before passthrough so buyers never
                    # get a 200 after shipping invalid OpenAI-shaped metadata/input.
                    _validate_responses_model(body)
                    _validate_responses_conversation_controls(body)
                    if "store" in body:
                        _validate_responses_store(body)
                    # OpenAI ``user`` end-user id — same fail-closed shape as chat/Completions.
                    if "user" in body:
                        _validate_completions_user(body)
                    if "service_tier" in body:
                        _validate_service_tier(body, endpoint_path="/v1/responses")
                    if "stream_options" in body:
                        _validate_responses_stream_options(body)
                    # Sampling knobs: type/range fail-closed before provider passthrough.
                    if "temperature" in body:
                        _validate_completions_temperature(body)
                    if "top_p" in body:
                        _validate_completions_top_p(body)
                    if "presence_penalty" in body:
                        _validate_completions_presence_penalty(body)
                    if "frequency_penalty" in body:
                        _validate_completions_frequency_penalty(body)
                    if "n" in body:
                        _validate_responses_n(body)
                    if "seed" in body:
                        _validate_responses_seed(body)
                    if "stop" in body:
                        _validate_responses_stop(body)
                    if "logit_bias" in body:
                        _validate_responses_logit_bias(body)
                    if "logprobs" in body or "top_logprobs" in body:
                        _validate_responses_logprobs(body)
                    if "max_tokens" in body:
                        _validate_completions_max_tokens(body)
                    if "max_completion_tokens" in body:
                        _validate_chat_max_completion_tokens(body)
                    if "max_output_tokens" in body:
                        _validate_responses_max_output_tokens(body)
                    if "max_tool_calls" in body:
                        _validate_responses_max_tool_calls(body)
                    _validate_openai_sdk_control_fields(body, endpoint_path="/v1/responses")
                    _validate_tool_resources(body, endpoint_path="/v1/responses")
                    _validate_openai_background(body, endpoint_path="/v1/responses")
                    if "parallel_tool_calls" in body:
                        _validate_responses_parallel_tool_calls(body)
                    # Tools surface: same OpenAI function-tool shape as chat; fail closed.
                    functions_raw = body.get("functions") if "functions" in body else None
                    function_call_raw = body.get("function_call") if "function_call" in body else None
                    functions_present = (
                        "functions" in body
                        and functions_raw is not None
                        and not (isinstance(functions_raw, list) and not functions_raw)
                    )
                    # function_call none/auto/empty-string (whitespace-padded) without functions
                    # are omit-equivalent no-ops.
                    function_call_present = (
                        "function_call" in body
                        and function_call_raw is not None
                        and not (
                            isinstance(function_call_raw, str)
                            and (
                                not function_call_raw.strip()
                                or function_call_raw.strip().lower() in ("none", "auto")
                            )
                        )
                    )
                    if functions_present or function_call_present:
                        raise RequestError(
                            400,
                            "invalid_functions",
                            "functions and function_call are not supported on /v1/responses; "
                            "use tools and tool_choice instead",
                        )
                    tools_list = body.get("tools") if isinstance(body.get("tools"), list) else None
                    # tool_choice null is omit-equivalent; alone / empty tools: only "none" is a valid no-op.
                    if (
                        "tool_choice" in body
                        and body.get("tool_choice") is not None
                        and not tools_list
                    ):
                        tc = body.get("tool_choice")
                        tc_norm = tc.strip().lower() if isinstance(tc, str) else tc
                        # none/auto/empty-object/empty-string without tools are omit-equivalent no-ops.
                        if (
                            tc_norm not in ("none", "auto")
                            and not (isinstance(tc, dict) and not tc)
                            and not (isinstance(tc, str) and not tc.strip())
                        ):
                            raise RequestError(
                                400,
                                "invalid_tool_choice",
                                "tool_choice requires tools on /v1/responses",
                            )
                    if "tools" in body:
                        _validate_chat_tools(body)
                    if "tool_choice" in body:
                        _validate_chat_tool_choice(body)
                    if "response_format" in body:
                        _validate_chat_response_format(body)
                    if "modalities" in body:
                        _validate_responses_modalities(body)
                    if "prediction" in body:
                        _validate_responses_prediction(body)
                    if "reasoning_effort" in body and body.get("reasoning_effort") is not None:
                        raise RequestError(
                            400,
                            "invalid_reasoning_effort",
                            "reasoning_effort is not supported on /v1/responses",
                        )
                    if "reasoning" in body:
                        _validate_responses_reasoning(body)
                    if "instructions" in body:
                        _validate_responses_instructions(body)
                    if "metadata" in body:
                        _validate_openai_metadata(body)
                    if "attribution" in body:
                        _validate_attribution(body.get("attribution"))
                    if "routing" in body:
                        routing = _validate_routing(body.get("routing"))
                        # Responses passthrough has no batch channel plane yet.
                        if routing and routing.get("channel") == "batch":
                            raise RequestError(
                                400,
                                "invalid_routing",
                                "routing.channel=batch is not supported on /v1/responses",
                            )
                        if routing and routing.get("latency_tolerant") is True:
                            raise RequestError(
                                400,
                                "invalid_routing",
                                "routing.latency_tolerant=true is not supported on /v1/responses",
                            )
                    if "input" not in body:
                        raise RequestError(400, "invalid_input", "input is required on /v1/responses")
                    input_value = body.get("input")
                    if not isinstance(input_value, (str, list)) or (
                        isinstance(input_value, str) and not input_value.strip()
                    ) or (isinstance(input_value, list) and len(input_value) == 0):
                        raise RequestError(
                            400,
                            "invalid_input",
                            "input must be a non-empty string or non-empty array on /v1/responses",
                        )
                    # stream=false / omit → non-SSE JSON response (honest no-stream path).
                    # stream=true is not implemented for Responses passthrough.
                    # String/0-1 forms coerce via shared bool helper (parity with chat).
                    if "stream" in body:
                        stream = _coerce_optional_bool(
                            body.get("stream"),
                            error_code="invalid_stream",
                            message="stream must be a boolean",
                        )
                        if stream is True:
                            raise RequestError(
                                400,
                                "invalid_stream",
                                "stream is not supported on /v1/responses",
                            )
                    started_at = time.perf_counter()
                    proxied = self._run(
                        lambda: orchestrator.proxy_completion(body, endpoint="responses")
                    )
                    orchestrator.record_analytics_event(
                        "responses_passthrough",
                        {
                            "endpoint_path": "/v1/responses",
                            "actor_scope": "inference",
                            "status_code": 200,
                            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        },
                    )
                    if body.get("stream") is True:
                        self._send_sse(responses_sse_body(proxied))
                    else:
                        self._send(proxied)
                    return

                if path == "/admin/simulate":
                    _reject_unknown_keys(body, ALLOWED_SIMULATE_KEYS)
                    prompt = body.get("prompt", "")
                    if not isinstance(prompt, str):
                        raise RequestError(400, "invalid_request", "prompt must be a string")
                    mode = _validate_mode(body.get("mode", "auto"))
                    include_trace = bool(body.get("include_orchestration_trace", security.expose_trace_by_default))
                    result = self._run(lambda: orchestrator.run([{"role": "user", "content": prompt}], mode=mode))
                    self._send(_response_payload(result, include_trace))
                    return
                if path == "/api/v1/workflow_runs":
                    _reject_unknown_keys(body, ALLOWED_WORKFLOW_KEYS)
                    prompt = body.get("prompt_text", "")
                    if not isinstance(prompt, str) or not prompt:
                        raise RequestError(400, "invalid_request", "prompt_text is required")
                    mode = _validate_mode(body.get("run_mode", "auto"))
                    include_trace = bool(body.get("include_orchestration_trace", security.expose_trace_by_default))
                    result = self._run(lambda: orchestrator.run([{"role": "user", "content": prompt}], mode=mode))
                    self._send(_response_payload(result, include_trace), 201)
                    return
                if path == "/api/v1/evaluation_runs":
                    _reject_unknown_keys(body, ALLOWED_EVALUATION_KEYS)
                    prompts = body.get("prompts")
                    if prompts is None and "prompt_text" in body:
                        prompts = [body["prompt_text"]]
                    if not isinstance(prompts, list) or not prompts:
                        raise RequestError(400, "invalid_request", "prompts must be a non-empty array")
                    mode = _validate_mode(body.get("run_mode", "auto"))
                    include_trace = bool(body.get("include_orchestration_trace", security.expose_trace_by_default))
                    evaluation_run = self._run(lambda: orchestrator.run_evaluation([str(item) for item in prompts], mode=mode))
                    self._send(_response_payload(evaluation_run, include_trace), 201)
                    return
                self._send_error(404, "route_not_found", "not found")
            except json.JSONDecodeError:
                self._send_error(400, "invalid_json", "request body is not valid JSON")
            except ToolFallbackStoppedError as exc:
                self._send_error(
                    TOOL_FALLBACK_STOPPED_STATUS,
                    TOOL_FALLBACK_STOPPED_CODE,
                    TOOL_FALLBACK_STOPPED_MESSAGE,
                    _tool_fallback_error_detail(exc),
                )
            except BudgetExceededError as exc:
                self._send_error(429, "budget_exceeded", str(exc), exc.detail)
            except RequestError as exc:
                self._send_error(exc.status, exc.code, exc.message, exc.detail)
            except (TypeError, ValueError) as exc:
                self._send_error(400, "invalid_request", str(exc))
            except Exception:
                self._send_error(500, "internal_error", "internal server error")

        def _authorize(self, scope: str) -> None:
            security.check_rate_limit(self.client_address[0])
            security.authorize(self.headers, scope, self.client_address[0])

        def _run(self, callback: Any) -> dict[str, Any]:
            security.acquire_run_slot()
            try:
                return callback()
            finally:
                security.release_run_slot()

        def _parse_positive_int(self, raw: str | None, field_name: str, default: int, max_value: int | None = None) -> int:
            value = default if raw is None else int(raw)
            if value < 1:
                raise ValueError(f"{field_name} must be >= 1")
            if max_value is not None and value > max_value:
                raise ValueError(f"{field_name} must be <= {max_value}")
            return value

        def _parse_paging(
            self,
            query: dict[str, list[str]],
            default_size: int = 10,
            max_size: int = 100,
        ) -> tuple[int, int]:
            page_number = self._parse_positive_int((query.get("page_number") or [None])[0], "page_number", 1)
            page_size = self._parse_positive_int((query.get("page_size") or [None])[0], "page_size", default_size, max_size)
            return page_number, page_size

        def _parse_optional_int(self, query: dict[str, list[str]], field_name: str) -> int | None:
            raw = (query.get(field_name) or [None])[0]
            if raw is None or raw == "":
                return None
            return int(raw)

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise RequestError(415, "unsupported_media_type", "content-type must be application/json")
            try:
                body_size = _request_body_size(self.headers, security.max_body_bytes)
            except RequestError:
                # Do not let a peer reuse a connection after an ambiguous frame.
                self.close_connection = True
                raise
            raw = self.rfile.read(body_size)
            if len(raw) != body_size:
                self.close_connection = True
                raise RequestError(
                    400,
                    "invalid_request_framing",
                    "request body ended before content-length",
                )
            return _coerce_json(raw) if raw else {}

        def log_message(self, format: str, *args: object) -> None:
            """Suppress default request logging to keep service output structured."""
            return

        def _send_error(
            self,
            status: int,
            code: str,
            message: str,
            detail: dict[str, Any] | None = None,
        ) -> None:
            self._send(_error_payload(code, message, {"request_id": uuid.uuid4().hex, **(detail or {})}), status)

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(raw)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _send_text(self, payload: str, content_type: str, status: int = 200) -> None:
            raw = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(raw)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _send_sse(self, body: str, status: int = 200) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("content-length", str(len(raw)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _begin_sse(self) -> None:
            # Incremental SSE: no content-length; the connection close delimits the body.
            self.send_response(200)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self._send_security_headers()
            self.end_headers()

        def _write_sse(self, frame: str) -> None:
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        def _stream_route_completion(self, orchestrator: Any, security: Any, messages: Any, model_name: str) -> None:
            """Pipe a worker's live deltas out as OpenAI chat.completion.chunk SSE frames."""
            run_id = f"run_{uuid.uuid4().hex}"
            completion_id = _new_chat_completion_id()
            created = int(time.time())

            def frame(delta: dict[str, Any], finish: str | None = None) -> str:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            security.acquire_run_slot()
            try:
                self._begin_sse()
                self._write_sse(frame({"role": "assistant"}))
                try:
                    for delta in orchestrator.stream_route(messages, workflow_run_id=run_id):
                        self._write_sse(frame({"content": delta}))
                    self._write_sse(frame({}, finish="stop"))
                except ToolFallbackStoppedError as exc:
                    detail = {
                        "request_id": uuid.uuid4().hex,
                        **_tool_fallback_error_detail(exc),
                    }
                    payload = _error_payload(
                        TOOL_FALLBACK_STOPPED_CODE,
                        TOOL_FALLBACK_STOPPED_MESSAGE,
                        detail,
                    )
                    self._write_sse(
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )
                    self._write_sse(frame({}, finish="error"))
                except Exception:  # noqa: BLE001 - headers already sent; surface as a terminal error frame
                    self._write_sse(frame({}, finish="error"))
                self._write_sse("data: [DONE]\n\n")
            finally:
                security.release_run_slot()

        def _send_security_headers(self) -> None:
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("cache-control", "no-store")
            self.send_header("x-frame-options", "DENY")

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    orchestrator: TaskOrchestrator,
    host: str = "127.0.0.1",
    port: int = 8000,
    security: SecurityConfig | None = None,
    clearfolio_url: str | None = None,
) -> None:
    """Serve the admin console and resource-oriented orchestration API."""
    server = build_server(orchestrator, host=host, port=port, security=security, clearfolio_url=clearfolio_url)
    print(f"listening on http://{host}:{port}")
    server.serve_forever()
