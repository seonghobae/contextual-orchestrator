"""Runtime orchestration, workflow trace, governance, and audit primitives."""

from __future__ import annotations

from collections import Counter, deque, OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import wraps
import http.client
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import ssl
import sqlite3
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse, urlunsplit
import urllib.error
import urllib.request

from jsonschema.validators import validator_for

from .chat_capability import (
    is_chat_compatible_model_id,
    is_general_chat_agent_model_id,
)
from .conventions import require_object_name
from .credentials import NotConfigured, get_credential
from .model_group import ModelGroupRouter, canonical_group_name
from .endpoint_race import EndpointAttempt, EndpointEquivalenceContract, race_first_valid
from .telemetry import inject_trace_context, traced
from .pii_protection import (
    DEFAULT_PII_KEY_NAME,
    ENCRYPTED_FIELDS_KEY,
    PiiFieldEncryptor,
    PiiProtectionError,
    is_encrypted_detail,
    load_pii_encryptor,
)
from .tool_fallback import (
    MAX_TOOL_RETRY_ATTEMPTS,
    ToolExecutionError,
    ToolFailureDecision,
    ToolFailureKind,
    ToolFallbackAction,
    ToolFallbackStoppedError,
    classify_tool_failure,
    downgrade_to_failover,
)
from .response_cache import ResponseCacheProvider, build_response_cache_key
from .reasoning_effort_profile import (
    ReasoningEffortProfile,
    apply_request_profile,
    snapshot_role_effort_catalog,
)


# content is usually str; multimodal vision messages use OpenAI content-parts lists.
ChatMessage = dict[str, Any]
ProviderDestination = tuple[int, tuple[Any, ...]]
MAX_LOCAL_CONCURRENCY = 64
_PASSTHROUGH_UNAVAILABLE_STATUS = frozenset({404, 410})
_PROVIDER_ERROR_CHAIN_LIMIT = 8
DEFAULT_PROVIDER_PROBE_TIMEOUT = 5.0
MODEL_CAPABILITIES = frozenset(
    {"text", "image", "video", "speech", "transcription", "embedding", "rerank", "audio"}
)
MAX_PROVIDER_PROBE_TIMEOUT = 30.0
_SAFE_PROVIDER_PROBE_ERROR_TYPES = frozenset({
    "ConnectionError",
    "HTTPError",
    "OSError",
    "RuntimeError",
    "SSLError",
    "TimeoutError",
    "TypeError",
    "UnknownError",
    "URLError",
    "ValueError",
})


def _safe_provider_probe_error_type(exc: Exception) -> str:
    """Keep provider diagnostics package-owned instead of echoing exception classes."""
    name = type(exc).__name__
    return name if name in _SAFE_PROVIDER_PROBE_ERROR_TYPES else "UnknownError"


def _validate_provider_probe_timeout(timeout: float) -> float:
    """Validate the finite, bounded timeout used by explicit readiness probes."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("provider probe timeout must be a finite number")
    value = float(timeout)
    if not math.isfinite(value) or not 0.1 <= value <= MAX_PROVIDER_PROBE_TIMEOUT:
        raise ValueError(
            f"provider probe timeout must be between 0.1 and {MAX_PROVIDER_PROBE_TIMEOUT:g} seconds"
        )
    return value


class BudgetExceededError(RuntimeError):
    """Raised when an operator-configured spend budget is already exhausted."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class ProviderResponseError(RuntimeError):
    """Raised for a provider response that cannot become a safe completion."""


def _structured_output_error(
    content: str, response_format: object
) -> str | None:
    """Return a bounded contract error for strict JSON Schema output."""
    if not isinstance(response_format, Mapping) or response_format.get("type") != "json_schema":
        return None
    specification = response_format.get("json_schema")
    schema = specification.get("schema") if isinstance(specification, Mapping) else None
    if not isinstance(schema, Mapping):
        return "schema_missing"
    try:
        instance = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return "invalid_json"
    validator_type = validator_for(schema)
    try:
        validator_type.check_schema(schema)
        validator_type(schema).validate(instance)
    except Exception:  # noqa: BLE001 - untrusted schema/output boundary
        return "schema_violation"
    return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). ponytail: heuristic, not a real tokenizer.

    Honest floor for spend analytics on mock/runtime text; replace with provider-reported
    usage when real workers return it.
    """
    return (len(text) + 3) // 4 if text else 0


def _step_output_tokens(step: Mapping[str, Any]) -> tuple[int, bool]:
    """Return provider-reported output tokens or the existing text estimate."""
    usage = step.get("usage")
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            reported = usage.get(key)
            if type(reported) is int and reported >= 0:
                return reported, True
    return estimate_tokens(step.get("output", "")), False


def _step_output_token_count(step: Mapping[str, Any]) -> int:
    """Return the output count used by in-flight structured budget checks."""
    return _step_output_tokens(step)[0]


def _cost_usd_decimal(output_tokens: int, price_per_million: float) -> Decimal:
    """Return exact decimal USD for tokens at a USD-per-million price."""
    return Decimal(output_tokens) * Decimal(str(price_per_million)) / Decimal(1_000_000)


_COMMERCIAL_REPORT_CACHE: ContextVar[dict[tuple[Any, Any, Any], dict[str, Any]] | None] = ContextVar(
    "commercial_report_cache",
    default=None,
)

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
)

DEFAULT_COMMERCIAL_TARGET_VALUE_KRW = 2_000_000_000
MAX_MODEL_JUDGE_REPLY_CHARACTERS = 32_000
CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1 = "contextual-orchestrator-contract-v1"


@dataclass(frozen=True)
class FastMLSIRMJudgeComponents:
    """Resolved fast-mlsirm symbols used by model verification."""

    judge_cls: type[Any]
    criterion_cls: type[Any]
    format_error: type[Exception]


def _resolve_fast_mlsirm_components() -> FastMLSIRMJudgeComponents | None:
    """Resolve the fast-mlsirm adapter symbols without importing unconditionally."""
    try:
        from fast_mlsirm import ContextualOrchestratorJudge, JudgeCriterion, JudgeFormatError
    except ModuleNotFoundError as exc:
        if exc.name == "fast_mlsirm":
            return None
        raise
    return FastMLSIRMJudgeComponents(ContextualOrchestratorJudge, JudgeCriterion, JudgeFormatError)


@dataclass
class _FastMLSIJudgeAdapter:
    """Adapter that exposes `complete()` for `ContextualOrchestratorJudge`."""

    orchestrator: "TaskOrchestrator"
    text: str
    judge: str
    served_agent_id: str | None = None
    mode: str = "auto"
    allowed_agent_ids: set[str] | None = None

    @property
    def contextual_orchestrator_contract(self) -> str:
        """Declare the versioned gateway boundary required by fast-mlsirm."""
        return CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    @property
    def client(self) -> ModelClient:
        """Expose the existing gateway client capability to fast-mlsirm."""
        return self.orchestrator.client

    def complete(self, messages: list[ChatMessage], mode: str | None = None) -> dict[str, Any]:
        """Return one judge completion through the constrained adapter."""
        if mode is not None and (type(mode) is not str or mode not in {"auto", "route", "conduct"}):
            raise ValueError("mode must be auto, route, or conduct")
        output, served_id, usage = self.orchestrator._invoke(
            self._agent(),
            messages,
            text=self.text,
            role="judge",
            allowed_agent_ids=self.allowed_agent_ids,
            eligibility_role="verifier",
        )
        return self._completion_payload(output, served_id, usage, self.mode if mode is None else mode)

    def complete_structured(
        self,
        messages: list[ChatMessage],
        mode: str | None = None,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a Judge JSON-schema request through the existing gateway proxy."""
        if mode is not None and (type(mode) is not str or mode not in {"auto", "route", "conduct"}):
            raise ValueError("mode must be auto, route, or conduct")
        if not isinstance(response_format, dict):
            raise TypeError("response_format must be a mapping")
        agent = self._agent()
        request = {
            "model": agent.model,
            "messages": messages,
            "temperature": self.orchestrator.client.temperature,
            "max_tokens": self.orchestrator.client.max_output_tokens,
            "response_format": response_format,
        }
        effort_profile = self.orchestrator._role_effort_profile("judge")
        if effort_profile is not None:
            request = self.orchestrator.client.apply_effort_profile(
                agent, request, effort_profile
            )
        request["stream"] = False
        response = self.orchestrator.client.proxy_send(
            agent, "chat/completions", request
        )
        output = ModelClient._response_content(agent, response)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return self._completion_payload(output, agent.id, usage, self.mode if mode is None else mode)

    def _completion_payload(
        self,
        output: str,
        served_id: str,
        usage: dict[str, Any] | None,
        mode: str,
    ) -> dict[str, Any]:
        """Build the bounded adapter response shared by normal and structured calls."""
        self.served_agent_id = served_id
        trace = [
            {
                "id": 0,
                "role": "verifier",
                "agent_id": served_id,
                "subtask": "LLM-as-a-Judge evaluation",
                "output": output,
            }
        ]
        if usage is not None:
            trace[0]["usage"] = usage
        return {
            "answer": output,
            "mode": mode,
            "trace": trace,
        }

    def _agent(self) -> ModelAgent:
        return self.orchestrator._agent(self.judge)


def _parse_triage_reply(reply: str) -> bool:
    """Parse one exact ``{"workflow_required": bool}`` triage verdict.

    Rejects oversize replies, duplicate object keys, extra fields, and any
    non-boolean value so a chatty model can never smuggle a decision through.
    Raises ``ValueError`` on every violation; callers fail closed to the
    orchestrated path.
    """
    if not isinstance(reply, str) or len(reply) > MAX_MODEL_JUDGE_REPLY_CHARACTERS:
        raise ValueError("triage response is missing or exceeds the maximum size")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("triage response contains duplicate object keys")
            result[key] = value
        return result

    try:
        parsed = json.loads(reply, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("triage response is not valid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"workflow_required"}:
        raise ValueError("triage response must contain exactly workflow_required")
    if type(parsed["workflow_required"]) is not bool:
        raise ValueError("workflow_required must be a boolean")
    return parsed["workflow_required"]


def _parse_model_judge_reply(reply: str) -> tuple[str, str]:
    """Parse one exact, duplicate-free model-judge verdict."""
    if not isinstance(reply, str) or len(reply) > MAX_MODEL_JUDGE_REPLY_CHARACTERS:
        raise ValueError("judge response is missing or exceeds the maximum size")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("judge response contains duplicate object keys")
            result[key] = value
        return result

    try:
        decision = json.loads(reply.strip(), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, RecursionError, TypeError):
        raise ValueError("judge response is not valid JSON") from None
    if not isinstance(decision, dict) or set(decision) != {"decision", "reason"}:
        raise ValueError("judge response must match the exact verdict schema")
    decision_value = decision["decision"]
    if not isinstance(decision_value, str) or decision_value not in {"ACCEPT", "REJECT"}:
        raise ValueError("judge decision is not an allowed enum value")
    reason = decision["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("judge reason is missing")
    return decision_value, reason.strip()


@dataclass(frozen=True)
class ModelAgent:
    """Configuration for one model-backed worker in the agent pool."""

    id: str
    model: str
    base_url: str = "mock://local"
    # Legacy field: kept for back-compat. When set, its STRING is treated as the
    # KV credential NAME (never read as an environment variable). Prefer
    # ``credential_key``. See docs/kv-credentials.md.
    api_key_env: str = ""
    credential_key: str = "OPENAI_API_KEY"
    tags: tuple[str, ...] = ()
    priority: int = 0
    disabled: bool = False
    provider_name: str = ""
    provider_exclusions: tuple[str, ...] = ()
    # Explicit KV credential for an authenticated loopback gateway. Keep this
    # separate from ``credential_key`` so mlx:// workers remain keyless.
    local_credential_key: str = ""
    # Authorization header scheme, e.g. "Bearer" (OpenAI-compatible default) or
    # "Key" (Bytez). Sent as f"{auth_scheme} {api_key}".
    auth_scheme: str = "Bearer"
    # Optional measured-routing group: agents sharing a canonical group name are
    # one logical model whose members are ordered by observed speed/stability
    # (see model_group.ModelGroupRouter and planning ADR 0032).
    # Empty string means the agent is ungrouped.
    group_name: str = ""
    # ``None`` means provider support is unproven. Opt-in effort profiles then
    # fail closed unless the profile explicitly requests the safe ``omit`` fallback.
    reasoning_effort_supported: bool | None = None
    # Explicit reviewed replica contract. A group never races when this is absent.
    endpoint_equivalence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        require_object_name(self.id, "agent.id")
        if self.group_name:
            object.__setattr__(self, "group_name", canonical_group_name(self.group_name))
        if type(self.local_credential_key) is not str:
            raise TypeError("local_credential_key must be a string")
        if self.local_credential_key and urlparse(self.base_url).scheme != "local":
            raise ValueError("local_credential_key requires a local:// gateway URL")
        if not self.auth_scheme or type(self.auth_scheme) is not str:
            raise ValueError("auth_scheme must be a non-empty string")
        if self.reasoning_effort_supported not in (None, True, False):
            raise TypeError("reasoning_effort_supported must be true, false, or null")
        if self.endpoint_equivalence is not None:
            contract = EndpointEquivalenceContract(**self.endpoint_equivalence)
            object.__setattr__(self, "endpoint_equivalence", dict(contract.__dict__))

    def to_config(self) -> dict[str, Any]:
        """Round-trippable agent configuration (from_dict(to_config(a)) == a)."""
        return {
            "id": self.id,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_key": self.credential_key,
            "tags": list(self.tags),
            "priority": self.priority,
            "disabled": self.disabled,
            "provider_name": self.provider_name,
            "provider_exclusions": list(self.provider_exclusions),
            "local_credential_key": self.local_credential_key,
            "auth_scheme": self.auth_scheme,
            "group_name": self.group_name,
            "reasoning_effort_supported": self.reasoning_effort_supported,
            "endpoint_equivalence": self.endpoint_equivalence,
        }

    @property
    def credential_name(self) -> str:
        """KV credential name for this agent's provider secret.

        Back-compat: a legacy ``api_key_env`` value is treated as the credential
        NAME (its string), not as an environment variable to read. Otherwise the
        modern ``credential_key`` is used.
        """
        return self.api_key_env or self.credential_key

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelAgent":  # pragma: no cover
        """Build an agent from JSON configuration with naming validation."""
        require_object_name(value["id"], "agent.id")
        return cls(
            id=value["id"],
            model=value["model"],
            base_url=value.get("base_url", "mock://local"),
            api_key_env=value.get("api_key_env", ""),
            credential_key=value.get("credential_key", "OPENAI_API_KEY"),
            tags=tuple(value.get("tags", ())),
            priority=int(value.get("priority", 0)),
            disabled=bool(value.get("disabled", False)),
            provider_name=value.get("provider_name", ""),
            provider_exclusions=tuple(value.get("provider_exclusions", value.get("provider_exclusion", ()))),
            local_credential_key=value.get("local_credential_key", ""),
            auth_scheme=value.get("auth_scheme", "Bearer"),
            group_name=value.get("group_name", ""),
            reasoning_effort_supported=value.get("reasoning_effort_supported"),
            endpoint_equivalence=value.get("endpoint_equivalence"),
        )


def _validate_batch_results(
    requests: Mapping[str, list[ChatMessage]],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reject incomplete batch output before it can become an accepted run."""
    if not isinstance(results, Mapping):
        raise TypeError("batch provider returned an invalid result map")
    if set(requests) != set(results):
        raise RuntimeError(
            "batch provider returned an incomplete or unexpected result set "
            f"(requested={len(requests)}, received={len(results)})"
        )
    invalid_count = sum(
        not isinstance(result, Mapping) or not isinstance(result.get("content"), str)
        for result in results.values()
    )
    if invalid_count:
        raise RuntimeError(
            f"batch provider returned {invalid_count} result(s) without assistant content"
        )
    return {custom_id: dict(result) for custom_id, result in results.items()}


@dataclass(frozen=True)
class WorkflowStep:
    """One visible step in a conducted orchestration workflow."""

    id: int
    role: str
    agent_id: str
    subtask: str
    access: tuple[int, ...] = ()
    latency_ms: float | None = None
    output: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the workflow step for API and trace responses."""
        return {
            "id": self.id,
            "role": self.role,
            "agent_id": self.agent_id,
            "subtask": self.subtask,
            "access": list(self.access),
            "latency_ms": self.latency_ms,
            "output": self.output,
        }


@dataclass(frozen=True)
class OrchestrationPolicy:
    """Policy knobs that govern routing, verification, and admin visibility.

    Route-vs-conduct selection is evidence-based: one exact-schema model
    triage call (cached by content hash) replaces the former keyword-hint and
    character-length rules, which were hand-tuned heuristics with no
    literature or measured grounding.
    """

    route_p95_seconds: float = 2.5
    # Real-time answer judging on single-shot route paths. Verdicts feed the
    # quality ledger so measured accuracy steers subsequent routing.
    realtime_judge: bool = True
    verifier_required: bool = True
    # Conductor-style planning (arXiv:2512.04388): "generated" asks the planner model to
    # emit the workflow (subtasks, worker assignment, access lists); "template" keeps the
    # fixed 4-step plan. Generated plans that fail validation fall back to the template.
    workflow_planning: str = "template"
    max_workflow_steps: int = 6
    # Verifier verdicts are structured model judgments. Keyword matching is intentionally
    # unsupported: it cannot handle negation, language, or a report that quotes a risk.
    verifier_judge: str = "model"

    def __post_init__(self) -> None:
        if self.verifier_judge != "model":
            raise ValueError("keyword-based verifier_judge modes are unsupported; use 'model'")
        if isinstance(self.realtime_judge, bool) is False:
            raise ValueError("realtime_judge must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        """Return the API-safe policy snapshot for workflow records."""
        return {
            "route_p95_seconds": self.route_p95_seconds,
            "realtime_judge": self.realtime_judge,
            "verifier_required": self.verifier_required,
            "workflow_planning": self.workflow_planning,
            "verifier_judge": self.verifier_judge,
            "max_workflow_steps": self.max_workflow_steps,
            "workflow_steps": ["thinker", "worker", "verifier", "synthesizer"],
            "supported_locales": ["en", "ko"],
        }


# HTTP statuses worth retrying: request timeout, conflict, too-early, rate limit,
# and the standard upstream/gateway failures. Everything else (400/401/403/404 ...)
# is a caller or configuration error and must not be retried.
TRANSIENT_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
LOCAL_PROVIDER_SCHEMES = frozenset({"mlx", "local"})
LOCAL_PROVIDER_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_tool_execution_stopped(error: urllib.error.HTTPError) -> bool:
    """Return whether an HTTP error carries the terminal tool-stop contract."""
    cache_key = "_contextual_orchestrator_tool_execution_stopped"
    cached = getattr(error, cache_key, None)
    if isinstance(cached, bool):
        return cached
    try:
        payload = json.loads(error.read(65536).decode("utf-8"))
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        result = False
    else:
        details = payload.get("error") if isinstance(payload, dict) else None
        result = isinstance(details, dict) and details.get("code") == "tool_execution_stopped"
    try:
        setattr(error, cache_key, result)
    except (AttributeError, TypeError):  # pragma: no cover - HTTPError is mutable
        pass
    return result


def _provider_tool_execution_stopped(agent: ModelAgent) -> ToolFallbackStoppedError:
    """Convert the provider's terminal tool-stop contract to the public safe error."""
    decision = classify_tool_failure(
        ToolExecutionError(
            "provider reported terminal tool execution state",
            tool_name="provider_tool_runtime",
            kind=ToolFailureKind.TRANSPORT_ERROR,
            outcome_unknown=True,
        )
    )
    return ToolFallbackStoppedError(agent.id, decision)


def _is_local_provider_url(base_url: str) -> bool:
    """Return whether a provider uses the explicit loopback-only local scheme."""
    parsed = urlparse(base_url)
    try:
        parsed.port
    except ValueError:
        return False
    return parsed.scheme in LOCAL_PROVIDER_SCHEMES and parsed.hostname in LOCAL_PROVIDER_HOSTS


def _is_direct_mlx_provider_url(base_url: str) -> bool:
    """Return whether a loopback provider is the direct mlx-lm transport."""
    return _is_local_provider_url(base_url) and urlparse(base_url).scheme == "mlx"


def _provider_credential_name(agent: ModelAgent) -> str | None:
    """Return the credential name allowed for this provider transport."""
    if not _is_local_provider_url(agent.base_url):
        return agent.credential_name
    # mlx-lm is intentionally keyless; only the explicit local:// gateway
    # transport may opt into a separately named loopback bearer credential.
    if urlparse(agent.base_url).scheme != "local":
        return None
    return agent.local_credential_key or None


def _provider_credential(agent: ModelAgent) -> str | None:
    """Resolve the transport-specific credential from the KV registry."""
    name = _provider_credential_name(agent)
    return get_credential(name) if name else None


class _LocalProviderState:
    """Coordinate model switching and bounded concurrency for one local endpoint."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.active_model: str | None = None
        self.active = 0
        self.capacity = 1


_LOCAL_PROVIDER_STATES: dict[str, _LocalProviderState] = {}
_LOCAL_PROVIDER_STATES_GUARD = threading.Lock()


def _local_provider_state(base_url: str) -> _LocalProviderState:
    """Return the shared in-process coordinator for one loopback provider endpoint."""
    parsed = urlparse(base_url)
    key = urlunsplit(("http", (parsed.netloc or base_url).lower(), parsed.path.rstrip("/"), "", ""))
    with _LOCAL_PROVIDER_STATES_GUARD:
        return _LOCAL_PROVIDER_STATES.setdefault(key, _LocalProviderState())


@contextmanager
def _local_provider_slot(
    agent: ModelAgent,
    capacity: int,
    timeout: float,
):
    """Bound local requests and serialize model switches on a shared endpoint."""
    if not _is_local_provider_url(agent.base_url):
        yield
        return

    state = _local_provider_state(agent.base_url)
    deadline = time.monotonic() + max(float(timeout), 0.0)
    with state.condition:
        while True:
            if state.active == 0:
                state.active_model = agent.model
                state.capacity = capacity
            elif state.active_model == agent.model:
                state.capacity = min(state.capacity, capacity)

            if state.active_model == agent.model and state.active < state.capacity:
                state.active += 1
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local provider endpoint is busy past its request deadline")
            state.condition.wait(remaining)

    try:
        yield
    finally:
        with state.condition:
            state.active -= 1
            if state.active == 0:
                state.active_model = None
                state.capacity = 1
            state.condition.notify_all()


def _responses_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def _responses_to_chat_payload(request: dict[str, Any]) -> dict[str, Any]:
    # ADR 0002: keep Codex Responses compatibility at the public control-plane
    # boundary; mlx-lm remains a local Chat Completions worker provider.
    messages: list[dict[str, Any]] = []
    instructions = _responses_text(request.get("instructions"))
    if instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = request.get("input", "")
    if isinstance(raw_input, list):
        items = raw_input
    elif isinstance(raw_input, str):
        items = [{"type": "message", "role": "user", "content": raw_input}]
    else:
        raise ValueError("local Responses input must be a string or item list")
    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError("local Responses input items must be objects")
        item_type = item.get("type", "message")
        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"unsupported local Responses message role: {role}")
            raw_content = item.get("content")
            content: str | list[dict[str, Any]] = _responses_text(raw_content)
            if isinstance(raw_content, list):
                parts: list[dict[str, Any]] = []
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in {"input_text", "output_text", "text"} and isinstance(
                        part.get("text"), str
                    ):
                        parts.append({"type": "text", "text": part["text"]})
                    elif part_type in {"input_image", "image_url"}:
                        image_url = part.get("image_url")
                        if isinstance(image_url, str):
                            image_url = {
                                "url": image_url,
                                **(
                                    {"detail": part["detail"]}
                                    if isinstance(part.get("detail"), str)
                                    else {}
                                ),
                            }
                        if isinstance(image_url, dict):
                            parts.append({"type": "image_url", "image_url": image_url})
                if any(part.get("type") == "image_url" for part in parts):
                    content = parts
            if content:
                messages.append({"role": role, "content": content})
        elif item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": str(item.get("call_id", "")),
                "content": _responses_text(item.get("output", item.get("content", ""))),
            })
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": str(item.get("call_id", "")),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name", "")),
                        "arguments": str(item.get("arguments", "{}")),
                    },
                }],
            })
        elif item_type in {"reasoning", "item_reference"}:
            continue
        else:
            raise ValueError(f"unsupported local Responses input item: {item_type}")

    payload: dict[str, Any] = {
        "model": request.get("model", "local-model"),
        "messages": messages,
        "stream": False,
    }
    for key in (
        "temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty",
        "frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "user",
        "parallel_tool_calls", "tool_choice",
    ):
        if key in request:
            payload[key] = request[key]
    if "max_output_tokens" in request and "max_tokens" not in payload:
        payload["max_tokens"] = request["max_output_tokens"]

    response_format = _responses_text_format_to_chat_response_format(request.get("text"))
    if response_format is None and isinstance(request.get("response_format"), dict):
        response_format = request["response_format"]
    if response_format is not None:
        payload["response_format"] = response_format

    tools: list[dict[str, Any]] = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = {
            key: tool[key]
            for key in ("name", "description", "parameters", "strict")
            if key in tool
        }
        tools.append({"type": "function", "function": function})
    if tools:
        payload["tools"] = tools

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    return payload


def _responses_text_format_to_chat_response_format(
    text: Any,
) -> dict[str, Any] | None:
    """Translate a validated Responses text format for workflow evidence calls."""
    if not isinstance(text, dict) or not isinstance(text.get("format"), dict):
        return None
    fmt = text["format"]
    if fmt.get("type") in {"text", "json_object"}:
        return {"type": fmt["type"]}
    if fmt.get("type") != "json_schema":
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            key: fmt[key]
            for key in ("name", "schema", "description", "strict")
            if key in fmt
        },
    }


def _canonical_provider_usage(
    usage: dict[str, Any], *, responses: bool
) -> dict[str, Any]:
    """Copy provider usage with aliases consumed by existing spend accounting."""
    canonical = dict(usage)
    if responses:
        for responses_key, chat_key in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            value = canonical.get(responses_key)
            if type(value) is int and value >= 0:
                canonical.setdefault(chat_key, value)
    return canonical


def _chat_to_responses_payload(data: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    if not isinstance(content, str):
        content = message.get("reasoning") if isinstance(message.get("reasoning"), str) else ""

    output: list[dict[str, Any]] = []
    if content or not message.get("tool_calls"):
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    for tool_call in message.get("tool_calls", []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        output.append({
            "id": f"fc_{tool_call.get('id', uuid.uuid4().hex)}",
            "type": "function_call",
            "status": "completed",
            "call_id": str(tool_call.get("id", uuid.uuid4().hex)),
            "name": str(function.get("name", "")),
            "arguments": str(function.get("arguments", "{}")),
        })

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    response: dict[str, Any] = {
        "id": f"resp_{data.get('id', uuid.uuid4().hex)}",
        "object": "response",
        "created_at": int(data.get("created", time.time())),
        "model": data.get("model", request.get("model", "local-model")),
        "output": output,
        "output_text": content,
        "status": "completed" if choice.get("finish_reason") != "length" else "incomplete",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
        },
    }
    if isinstance(request.get("metadata"), dict):
        response["metadata"] = request["metadata"]
    return response


def is_transient_error(exc: BaseException) -> bool:
    """Return True when a provider call failure is worth retrying with backoff."""
    if isinstance(exc, urllib.error.HTTPError):
        if _is_tool_execution_stopped(exc):
            return False
        return exc.code in TRANSIENT_HTTP_STATUS
    # Network-level failures (DNS, connection reset, read timeout) are transient.
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    # A VPN/socket path can surface as an SSL EOF or SSL_ERROR_SYSCALL. Keep
    # certificate verification failures non-transient so a bad trust boundary
    # is never retried as if it were a network fault.
    if isinstance(exc, ssl.SSLError):
        return not isinstance(exc, ssl.SSLCertVerificationError)
    return False


_PASSTHROUGH_TRIGGER_KEYS = (
    "response_format",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
)


def _is_omit_equivalent_control(key: str, value: Any) -> bool:
    """Mirror the HTTP-boundary treat-as-omit rules for optional provider controls.

    ``response_format`` / ``tool_choice`` / ``function_call`` treat JSON null,
    empty objects, and blank strings as omit; the choice keys additionally treat
    the honest no-op keywords ``"none"`` / ``"auto"`` as omit. ``tools`` /
    ``functions`` treat JSON null and empty arrays as omit. A trigger key
    carrying only such a value is omit-equivalent to the key being absent, so it
    must not select the conducted-evidence + synthesis path on its own (parity
    with the ``_validate_chat_*`` boundary rules in ``server.py``).
    """
    if value is None:
        return True
    if key in ("tools", "functions"):
        return isinstance(value, list) and not value
    if isinstance(value, dict):
        return not value
    if isinstance(value, str):
        stripped = value.strip().casefold()
        if not stripped:
            return True
        return key in ("tool_choice", "function_call") and stripped in ("none", "auto")
    return False


def _is_passthrough_failover_error(exc: BaseException) -> bool:
    """Recognize failures proving that a passthrough request was not accepted."""
    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(_PROVIDER_ERROR_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        if (
            isinstance(current, urllib.error.HTTPError)
            and current.code
            in (_PASSTHROUGH_UNAVAILABLE_STATUS | TRANSIENT_HTTP_STATUS)
        ):
            return True
        if isinstance(current, socket.gaierror) and current.errno == socket.EAI_AGAIN:
            return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            return False
        else:
            current = current.__context__
    return False


class ModelClient:
    """Small chat-completions client with retry, backoff, and mock support."""

    def __init__(
        self,
        timeout: int = 90,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
        local_max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_backoff_cap: float = 8.0,
        temperature: float = 0.2,
        local_concurrency: int = 1,
        chat_template_args: dict[str, Any] | None = None,
        ca_bundle: str | None = None,
        verify_tls: bool = True,
        allowed_provider_hosts: Iterable[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        if isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.default_temperature = temperature
        self.default_top_p: float | None = None
        self.default_presence_penalty: float | None = None
        self.default_frequency_penalty: float | None = None
        self.max_retries = max_retries
        if isinstance(local_max_retries, bool) or local_max_retries < 0:
            raise ValueError("local_max_retries must be >= 0")
        self.local_max_retries = int(local_max_retries)
        self.retry_backoff = retry_backoff
        self.retry_backoff_cap = retry_backoff_cap
        self.temperature = temperature
        if type(local_concurrency) is not int or not 1 <= local_concurrency <= MAX_LOCAL_CONCURRENCY:
            raise ValueError(
                f"local_concurrency must be an integer in 1..{MAX_LOCAL_CONCURRENCY}"
            )
        self.local_concurrency = local_concurrency
        self.chat_template_args = dict(chat_template_args or {})
        self.allowed_provider_hosts = self._normalize_allowed_provider_hosts(allowed_provider_hosts)
        # Seam so tests can observe/skip real sleeping during backoff.
        self._sleep = time.sleep
        # Per-thread usage from the most recent chat() (the server is threaded).
        self._local = threading.local()
        if not verify_tls:
            raise ValueError("provider TLS verification cannot be disabled; configure a trusted ca_bundle")
        # TLS trust for provider egress. The system trust store is the default;
        # ca_bundle points at a custom CA for a reviewed corporate gateway.
        self._ssl_context = self._build_ssl_context(ca_bundle)

    @staticmethod
    def _build_ssl_context(ca_bundle: str | None) -> ssl.SSLContext:
        if ca_bundle:
            if not os.path.isfile(ca_bundle):
                raise ValueError(f"provider CA bundle does not exist: {ca_bundle}")
            try:
                return ssl.create_default_context(cafile=ca_bundle)
            except OSError as exc:
                raise ValueError(f"provider CA bundle could not be loaded: {ca_bundle}") from exc
        return ssl.create_default_context()

    @staticmethod
    def _normalize_allowed_provider_hosts(hosts: Iterable[str] | None) -> frozenset[str]:
        """Normalize an explicit provider-host policy once at client construction."""
        if hosts is None:
            return frozenset()
        if isinstance(hosts, (str, bytes)):
            raise ValueError("allowed_provider_hosts must be an iterable of host strings")
        normalized: set[str] = set()
        try:
            values = iter(hosts)
        except TypeError as exc:
            raise ValueError("allowed_provider_hosts must be an iterable of host strings") from exc
        for host in values:
            if type(host) is not str:
                raise ValueError("allowed_provider_hosts must contain only strings")
            value = host.strip().lower()
            if not value or any(character in value for character in "/?#"):
                raise ValueError("allowed_provider_hosts entries must be bare host names")
            normalized.add(value)
        return frozenset(normalized)

    def take_usage(self) -> dict[str, Any] | None:
        """Return and clear provider-reported usage from the most recent chat() on this thread."""
        usage = getattr(self._local, "usage", None)
        self._local.usage = None
        return usage

    def request_settings_snapshot(self) -> dict[str, Any]:
        """Return this thread's effective request-scoped provider settings."""
        scoped = getattr(self._local, "request_settings", {})
        return {
            "temperature": scoped.get("temperature", self.default_temperature),
            "top_p": scoped.get("top_p", self.default_top_p),
            "presence_penalty": scoped.get("presence_penalty", self.default_presence_penalty),
            "frequency_penalty": scoped.get("frequency_penalty", self.default_frequency_penalty),
            "max_output_tokens": scoped.get("max_output_tokens", self.max_output_tokens),
        }

    @contextmanager
    def request_settings(self, **overrides: Any):
        """Apply provider settings to only the current server request thread."""
        previous = getattr(self._local, "request_settings", None)
        current = self.request_settings_snapshot()
        current.update({key: value for key, value in overrides.items() if value is not None})
        self._local.request_settings = current
        try:
            yield
        finally:
            if previous is None:
                del self._local.request_settings
            else:
                self._local.request_settings = previous

    #: Deterministic vector dimension for mock-provider embeddings (test fixture
    #: only; production providers always return their own dimensionality).
    MOCK_EMBEDDING_DIMENSION = 8

    def embed(self, agent: ModelAgent, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string from an OpenAI-compatible endpoint.

        Real providers receive a standard ``/embeddings`` request. The
        ``mock://`` transport derives a deterministic BLAKE2b-based unit-sphere
        vector so unit tests can exercise cosine ordering without network
        access; this fixture is never used against production traffic.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings")
        for item in texts:
            if not isinstance(item, str) or not item:
                raise ValueError("texts entries must be non-empty strings")
        if agent.base_url.startswith("mock://"):
            vectors: list[list[float]] = []
            for item in texts:
                digest = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
                raw = [byte for byte in digest[: self.MOCK_EMBEDDING_DIMENSION]]
                centered = [(value / 255.0) * 2.0 - 1.0 for value in raw]
                vectors.append(centered)
            return vectors
        destination = self._validate_provider(agent)  # pragma: no cover
        payload = {"model": agent.model, "input": texts}  # pragma: no cover
        response = self._send_raw(agent, "embeddings", payload, destination)  # pragma: no cover
        data = response.get("data") if isinstance(response, dict) else None  # pragma: no cover
        if not isinstance(data, list) or len(data) != len(texts):  # pragma: no cover
            raise RuntimeError(  # pragma: no cover
                f"provider {agent.id} returned an invalid embeddings payload"
            )
        vectors = []  # pragma: no cover
        for entry in data:  # pragma: no cover
            vector = entry.get("embedding") if isinstance(entry, dict) else None  # pragma: no cover
            if not isinstance(vector, list) or not all(  # pragma: no cover
                isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector
            ):
                raise RuntimeError(  # pragma: no cover
                    f"provider {agent.id} returned a non-numeric embedding vector"
                )
            vectors.append([float(value) for value in vector])  # pragma: no cover
        return vectors  # pragma: no cover

    def chat(
        self,
        agent: ModelAgent,
        messages: list[ChatMessage],
        temperature: float | None = None,
        top_p: float | None = None,
        effort_profile: ReasoningEffortProfile | None = None,
    ) -> str:
        """Send messages to a mock or OpenAI-compatible chat endpoint with retries.

        When ``temperature``/``top_p`` are omitted, ``default_temperature`` and
        ``default_top_p`` are used so request-scoped Completions sampling can be
        applied without threading kwargs through every orchestrator hop.
        """
        if not is_chat_compatible_model_id(agent.model):
            raise ValueError("model is not chat-compatible and cannot serve a chat request")
        self._local.usage = None
        # Expose the effective sampling knobs for request-path tests / diagnostics.
        settings = self.request_settings_snapshot()
        effective_temperature = settings["temperature"] if temperature is None else temperature
        effective_top_p = settings["top_p"] if top_p is None else top_p
        effective_presence = settings["presence_penalty"]
        effective_frequency = settings["frequency_penalty"]
        self._local.last_temperature = effective_temperature
        self._local.last_top_p = effective_top_p
        self._local.last_presence_penalty = effective_presence
        self._local.last_frequency_penalty = effective_frequency
        if agent.base_url.startswith("mock://"):
            return self._mock(agent, messages)

        destination = self._validate_provider(agent)  # pragma: no cover
        api_key = _provider_credential(agent)  # pragma: no cover
        credential_name = _provider_credential_name(agent)  # pragma: no cover
        if credential_name and not api_key:  # pragma: no cover
            raise NotConfigured(
                f"{agent.id} requires a resolvable credential '{credential_name}' in the KV"
            )

        payload = {  # pragma: no cover
            "model": agent.model,
            "messages": messages,
            "temperature": effective_temperature,
            "stream": False,
            "max_tokens": settings["max_output_tokens"],
        }
        if effective_top_p is not None:  # pragma: no cover
            payload["top_p"] = effective_top_p
        if effective_presence is not None:  # pragma: no cover
            payload["presence_penalty"] = effective_presence
        if effective_frequency is not None:  # pragma: no cover
            payload["frequency_penalty"] = effective_frequency
        if _is_direct_mlx_provider_url(agent.base_url) and self.chat_template_args:
            payload["chat_template_kwargs"] = self.chat_template_args
        payload = self.apply_effort_profile(agent, payload, effort_profile)
        parsed_provider = urlparse(agent.base_url)
        with traced(
            f"chat {agent.model}",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": agent.provider_name or parsed_provider.hostname or agent.id,
                "gen_ai.request.model": agent.model,
                "contextual_orchestrator.agent_id": agent.id,
                "server.address": parsed_provider.hostname or "",
                "server.port": parsed_provider.port or (443 if parsed_provider.scheme == "https" else 80),
            },
        ), _local_provider_slot(agent, self.local_concurrency, self.timeout):
            return self._send_with_retry(agent, payload, destination)

    def apply_effort_profile(
        self,
        agent: ModelAgent,
        payload: dict[str, Any],
        profile: ReasoningEffortProfile | None,
    ) -> dict[str, Any]:
        """Apply an opt-in profile while proving provider support before egress."""
        supports = (
            agent.reasoning_effort_supported is True
            or (agent.reasoning_effort_supported is None and agent.base_url.startswith("mock://"))
        )
        return apply_request_profile(
            payload,
            profile,
            supports_reasoning_effort=supports,
            default_max_output_tokens=self.max_output_tokens,
        )

    def probe(self, agent: ModelAgent, *, timeout: float = DEFAULT_PROVIDER_PROBE_TIMEOUT) -> dict[str, Any]:
        """Verify a local model registry, then run one bounded completion probe.

        ``/health`` and ``/v1/models`` only prove process/model-registry liveness;
        this verifies the configured local model and deliberately exercises the
        chat path with one output token. It never retries, so a stuck local queue
        cannot be multiplied by the readiness check.
        """
        probe_timeout = _validate_provider_probe_timeout(timeout)
        started = time.monotonic()
        if not is_chat_compatible_model_id(agent.model):
            return {
                "agent_id": agent.id,
                "model": agent.model,
                "status": "not_ready",
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "error_type": "ValueError",
                "failure_code": "non_chat_model",
            }
        self._local.usage = None
        failure_code = "provider_probe_failed"
        try:
            if agent.base_url.startswith("mock://"):
                content = self._mock(agent, [{"role": "user", "content": "Reply with exactly OK."}])
                usage = None
            else:
                destination = self._validate_provider(agent)
                if _is_local_provider_url(agent.base_url):
                    registry_request = urllib.request.Request(
                        self._provider_url(agent, "/models"),
                        method="GET",
                    )
                    with self._open_provider(
                        registry_request, destination, timeout=probe_timeout
                    ) as registry_response:
                        registry = json.loads(
                            registry_response.read().decode("utf-8")
                        )
                    model_ids = {
                        item.get("id")
                        for item in registry.get("data", [])
                        if isinstance(item, dict) and type(item.get("id")) is str
                    }
                    if agent.model not in model_ids:
                        failure_code = "provider_model_not_registered"
                        raise RuntimeError(
                            f"provider {agent.id} model registry does not contain {agent.model!r}"
                        )
                payload: dict[str, Any] = {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "temperature": 0.0,
                    "stream": False,
                    "max_tokens": 1,
                }
                if _is_direct_mlx_provider_url(agent.base_url) and self.chat_template_args:
                    payload["chat_template_kwargs"] = self.chat_template_args
                with _local_provider_slot(agent, self.local_concurrency, probe_timeout):
                    content = self._send(agent, payload, destination, timeout=probe_timeout)
                usage = self.take_usage()
            if not content.strip():
                failure_code = "provider_empty_probe_response"
                raise RuntimeError(f"provider {agent.id} returned empty probe content")
            return {
                "agent_id": agent.id,
                "model": agent.model,
                "status": "ready",
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "usage": usage,
            }
        except Exception as exc:  # noqa: BLE001 - readiness reports failures, it does not serve them
            return {
                "agent_id": agent.id,
                "model": agent.model,
                "status": "not_ready",
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "error_type": _safe_provider_probe_error_type(exc),
                "failure_code": failure_code,
            }

    def _send_with_retry(
        self,
        agent: ModelAgent,
        payload: dict[str, Any],
        destination: ProviderDestination | None = None,
        *,
        timeout: float | None = None,
    ) -> str:
        """Call the provider, retrying transient failures with exponential backoff + jitter."""
        last_error: Exception | None = None
        retry_limit = self._retry_limit(agent)
        for attempt in range(retry_limit + 1):  # pragma: no branch - retry limits are validated non-negative
            try:
                return (
                    self._send(agent, payload, destination)
                    if timeout is None
                    else self._send(agent, payload, destination, timeout=timeout)
                )
            except Exception as exc:  # noqa: BLE001 - classify then decide
                last_error = exc
                if attempt >= retry_limit or not is_transient_error(exc):
                    break
                self._sleep(self._backoff_delay(attempt))
        if isinstance(last_error, urllib.error.HTTPError) and _is_tool_execution_stopped(last_error):
            raise _provider_tool_execution_stopped(agent) from None
        if isinstance(last_error, ProviderResponseError):
            raise last_error
        raise RuntimeError(f"provider {agent.id} request failed") from None

    def _retry_limit(self, agent: ModelAgent) -> int:
        """Return a retry budget without multiplying an expensive local queue by default."""
        return self.local_max_retries if _is_local_provider_url(agent.base_url) else self.max_retries

    def _backoff_delay(self, attempt: int) -> float:
        """Full-jitter exponential backoff, capped, so retries do not thundering-herd a provider."""
        ceiling = min(self.retry_backoff_cap, self.retry_backoff * (2 ** attempt))
        return random.uniform(0.0, ceiling)

    def _send(
        self,
        agent: ModelAgent,
        payload: dict[str, Any],
        destination: ProviderDestination | None = None,
        *,
        timeout: float | None = None,
    ) -> str:
        """Perform one provider HTTP request (isolated so retry/backoff stays testable)."""
        api_key = _provider_credential(agent)
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"{agent.auth_scheme} {api_key}"
        inject_trace_context(headers)
        request = urllib.request.Request(
            self._provider_url(agent, "/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opened = (
            self._open_provider(request, destination)
            if timeout is None
            else self._open_provider(request, destination, timeout=timeout)
        )
        with opened as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._local.usage = usage
        return self._response_content(agent, data)

    @staticmethod
    def _response_content(agent: ModelAgent, data: dict[str, Any]) -> str:
        """Extract text and explain provider responses that contain reasoning only."""
        choices = data.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(message, dict) and message.get("reasoning"):
            raise ProviderResponseError(
                f"provider {agent.id} returned reasoning without content; "
                "for mlx-lm set chat_template_args={\"enable_thinking\": false} or increase max_output_tokens"
            )
        raise ProviderResponseError(f"provider {agent.id} response did not contain assistant content")

    @staticmethod
    def _connect_validated(
        destination: ProviderDestination, timeout: float | None, source_address: tuple[str, int] | None
    ) -> socket.socket:
        """Connect to one already-resolved address without performing another DNS lookup."""
        family, sockaddr = destination
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            connection.settimeout(timeout)
            if source_address is not None:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _resolve_addresses(hostname: str, port: int) -> list[ProviderDestination]:
        try:
            addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise RuntimeError(f"provider host {hostname!r} could not be resolved") from exc
        resolved = [(family, sockaddr) for family, _type, _proto, _canonname, sockaddr in addresses]
        if not resolved:
            raise RuntimeError(f"provider host {hostname!r} has no stream address")
        return resolved

    def _open_provider(
        self,
        request: urllib.request.Request,
        destination: ProviderDestination | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Open a validated HTTP(S) request without generic URL-handler dispatch."""
        parsed = urlparse(request.full_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise RuntimeError("provider request URL must be an HTTP(S) URL without userinfo or fragments")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise RuntimeError("provider request URL has an invalid port") from exc
        if destination is None:
            destination = self._resolve_addresses(parsed.hostname, port)[0]
        connection_timeout = self.timeout if timeout is None else timeout
        connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            # The explicit verifying context is the security control for this reviewed API.
            connection = http.client.HTTPSConnection(  # nosemgrep: python.lang.security.audit.httpsconnection-detected.httpsconnection-detected
                parsed.hostname,
                port,
                timeout=connection_timeout,
                context=self._ssl_context,
            )
        else:
            connection = http.client.HTTPConnection(parsed.hostname, port, timeout=connection_timeout)
        connection._create_connection = (  # type: ignore[attr-defined]
            lambda _address, timeout, source_address: self._connect_validated(
                destination, timeout, source_address
            )
        )
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            connection.request(
                request.get_method(),
                target,
                body=request.data,
                headers=dict(request.header_items()),
            )
            response = connection.getresponse()
            if response.status >= 400:
                body = response.read()
                status = response.status
                reason = response.reason
                headers = response.headers
                response.close()
                connection.close()
                raise urllib.error.HTTPError(
                    request.full_url,
                    status,
                    reason,
                    headers,
                    io.BytesIO(body),
                )
            return response
        except Exception:
            connection.close()
            raise

    def stream_chat(
        self,
        agent: ModelAgent,
        messages: list[ChatMessage],
        temperature: float | None = None,
        effort_profile: ReasoningEffortProfile | None = None,
    ):
        """Yield content deltas from a mock or OpenAI-compatible streaming endpoint.

        Real token streaming: the provider is called with stream=true and its SSE deltas
        are yielded as they arrive (not computed-then-framed). The mock path yields its
        answer in fixed chunks so behavior shape stays testable and unchanged.
        """
        if not is_chat_compatible_model_id(agent.model):
            raise ValueError(
                f"model {agent.model!r} is not chat-compatible and cannot serve {agent.id!r}"
            )
        if agent.base_url.startswith("mock://"):
            answer = self._mock(agent, messages)
            for start in range(0, len(answer), 24):
                yield answer[start : start + 24]
            return

        destination = self._validate_provider(agent)  # pragma: no cover
        settings = self.request_settings_snapshot()
        payload = {  # pragma: no cover
            "model": agent.model,
            "messages": messages,
            "temperature": settings["temperature"] if temperature is None else temperature,
            "stream": True,
            "max_tokens": settings["max_output_tokens"],
        }
        if _is_direct_mlx_provider_url(agent.base_url) and self.chat_template_args:
            payload["chat_template_kwargs"] = self.chat_template_args
        payload = self.apply_effort_profile(agent, payload, effort_profile)
        parsed_provider = urlparse(agent.base_url)
        with traced(
            f"chat {agent.model}",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": agent.provider_name or parsed_provider.hostname or agent.id,
                "gen_ai.request.model": agent.model,
                "contextual_orchestrator.agent_id": agent.id,
                "server.address": parsed_provider.hostname or "",
                "server.port": parsed_provider.port or (443 if parsed_provider.scheme == "https" else 80),
            },
        ), _local_provider_slot(agent, self.local_concurrency, self.timeout):  # pragma: no cover
            yield from self._stream_send(agent, payload, destination)

    def _stream_send(
        self, agent: ModelAgent, payload: dict[str, Any], destination: ProviderDestination | None = None
    ):
        """Stream content deltas from a provider SSE response (real transport, testable)."""
        api_key = _provider_credential(agent)
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if api_key:
            headers["authorization"] = f"{agent.auth_scheme} {api_key}"
        inject_trace_context(headers)
        request = urllib.request.Request(
            self._provider_url(agent, "/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        stream_error: RuntimeError | None = None
        try:
            with self._open_provider(request, destination) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or [{}]
                    delta = (choices[0] or {}).get("delta", {}).get("content")
                    if delta:
                        yield delta
        except Exception as exc:  # noqa: BLE001 - provider error boundary (CWE-209)
            # The gateway's own terminal tool-stop contract must survive the
            # boundary: convert the provider HTTP shape into the package-owned
            # stop error so callers keep the 409 semantics they rely on.
            if _is_tool_execution_stopped(exc):
                raise _provider_tool_execution_stopped(agent) from None
            if isinstance(exc, ToolFallbackStoppedError):
                raise
            # A stream may already have emitted bytes, so it can neither be retried
            # nor failed over to another provider. Keep the provider status, body,
            # and exception cause inside the gateway; callers get one stable,
            # package-owned error instead of raw provider diagnostics.
            stream_error = RuntimeError(f"provider {agent.id} streaming request failed")
        if stream_error is not None:
            raise stream_error

    # -- Full OpenAI passthrough (transport) ------------------------------------
    # Requests that carry provider features the multi-agent verifier cannot merge
    # (response_format / tools / the Responses API) are proxied to a single agent
    # so the full provider response shape (tool_calls, parsed structured output,
    # Responses output items) survives verbatim. Agent selection lives on the
    # orchestrator; this is the agent-level transport.
    def proxy_send(
        self, agent: ModelAgent, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Passthrough a full request to one agent, returning the raw provider JSON."""
        return self._proxy_send(agent, endpoint, payload, allow_transient_retries=True)

    def proxy_send_once(
        self, agent: ModelAgent, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Send one passthrough attempt so cross-provider failover cannot amplify load."""
        return self._proxy_send(agent, endpoint, payload, allow_transient_retries=False)

    def _proxy_send(
        self,
        agent: ModelAgent,
        endpoint: str,
        payload: dict[str, Any],
        *,
        allow_transient_retries: bool,
    ) -> dict[str, Any]:
        """Apply the shared passthrough contract with a selectable retry policy."""
        normalized_endpoint = endpoint.strip("/")
        if normalized_endpoint.startswith("v1/"):
            normalized_endpoint = normalized_endpoint[3:]
        if (
            normalized_endpoint in {"chat/completions", "completions", "responses"}
            and not is_chat_compatible_model_id(agent.model)
        ):
            raise ValueError(
                f"model {agent.model!r} is not chat-compatible and cannot serve {agent.id!r}"
            )
        if agent.base_url.startswith("mock://"):
            return self._mock_raw(agent, normalized_endpoint, payload)
        destination = self._validate_provider(agent)  # pragma: no cover
        parsed_provider = urlparse(agent.base_url)
        operation_name = {
            "chat/completions": "chat",
            "completions": "text_completion",
            "responses": "generate_content",
        }.get(normalized_endpoint, "generate_content")
        with traced(
            f"{operation_name} {agent.model}",
            {
                "gen_ai.operation.name": operation_name,
                "gen_ai.provider.name": agent.provider_name or parsed_provider.hostname or agent.id,
                "gen_ai.request.model": agent.model,
                "contextual_orchestrator.agent_id": agent.id,
                "server.address": parsed_provider.hostname or "",
                "server.port": parsed_provider.port or (443 if parsed_provider.scheme == "https" else 80),
            },
        ):
            if (
                normalized_endpoint == "chat/completions"
                and _is_local_provider_url(agent.base_url)
            ):
                # Preserve caller ownership while supplying the configured cap
                # that local OpenAI-compatible servers require when SDKs omit it.
                payload = dict(payload)
                payload.setdefault(
                    "max_tokens",
                    self.request_settings_snapshot()["max_output_tokens"],
                )
            if normalized_endpoint == "responses" and _is_local_provider_url(agent.base_url):
                chat_payload = _responses_to_chat_payload(payload)
                if "response_format" in chat_payload and not (
                    "response_format" in agent.tags
                    or "capability:response_format" in agent.tags
                ):
                    raise ValueError(
                        "selected model does not support the requested response format"
                    )
                chat_payload.setdefault("max_tokens", self.request_settings_snapshot()["max_output_tokens"])
                if _is_direct_mlx_provider_url(agent.base_url) and self.chat_template_args:
                    chat_payload["chat_template_kwargs"] = self.chat_template_args
                with _local_provider_slot(agent, self.local_concurrency, self.timeout):
                    chat_response = self._send_raw_with_retry(
                        agent,
                        "chat/completions",
                        chat_payload,
                        destination,
                        allow_transient_retries=allow_transient_retries,
                    )
                return _chat_to_responses_payload(chat_response, payload)
            with _local_provider_slot(agent, self.local_concurrency, self.timeout):  # pragma: no cover
                return self._send_raw_with_retry(
                    agent,
                    normalized_endpoint,
                    payload,
                    destination,
                    allow_transient_retries=allow_transient_retries,
                )

    def proxy_send_bytes(
        self, agent: ModelAgent, endpoint: str, payload: dict[str, Any]
    ) -> tuple[bytes, str]:
        """Passthrough a provider response whose body is binary media."""
        if agent.base_url.startswith("mock://"):
            return b"mock audio", "audio/mpeg"
        api_key = _provider_credential(agent)  # pragma: no cover
        headers = {"content-type": "application/json"}  # pragma: no cover
        if api_key:  # pragma: no cover
            headers["authorization"] = f"{agent.auth_scheme} {api_key}"
        request = urllib.request.Request(  # pragma: no cover
            self._provider_url(agent, f"/{endpoint.lstrip('/')}"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._open_provider(request, self._validate_provider(agent)) as response:  # pragma: no cover
            return response.read(), response.headers.get_content_type()

    def _send_raw_with_retry(
        self,
        agent: ModelAgent,
        endpoint: str,
        payload: dict[str, Any],
        destination: ProviderDestination | None = None,
        *,
        allow_transient_retries: bool = True,
    ) -> dict[str, Any]:  # pragma: no cover
        """Passthrough transport with the same transient-failure retry policy as _send."""
        last_error: Exception | None = None
        retry_limit = self._retry_limit(agent) if allow_transient_retries else 0
        for attempt in range(retry_limit + 1):
            try:
                return self._send_raw(agent, endpoint, payload, destination)
            except Exception as exc:  # noqa: BLE001 - classify then decide
                last_error = exc
                if attempt >= retry_limit or not is_transient_error(exc):
                    break
                self._sleep(self._backoff_delay(attempt))
        if isinstance(last_error, urllib.error.HTTPError) and _is_tool_execution_stopped(last_error):
            raise _provider_tool_execution_stopped(agent) from None
        if not allow_transient_retries and last_error is not None:
            raise last_error
        raise RuntimeError(f"provider {agent.id} passthrough request failed") from None

    def _send_raw(
        self,
        agent: ModelAgent,
        endpoint: str,
        payload: dict[str, Any],
        destination: ProviderDestination | None = None,
    ) -> dict[str, Any]:  # pragma: no cover
        """One provider HTTP request returning the FULL provider JSON (for passthrough)."""
        api_key = _provider_credential(agent)
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"{agent.auth_scheme} {api_key}"
        inject_trace_context(headers)
        request = urllib.request.Request(
            self._provider_url(agent, f"/{endpoint.lstrip('/')}"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._open_provider(request, destination) as response:
            return json.loads(response.read().decode("utf-8"))

    def _mock_raw(
        self, agent: ModelAgent, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock full provider response for tests; echoes forwarded params so passthrough is assertable."""
        response_format = payload.get("response_format")
        if response_format is None and isinstance(payload.get("text"), dict):
            response_format = payload["text"].get("format")
        mock_content = (
            "{}"
            if isinstance(response_format, dict)
            and str(response_format.get("type", "")).strip().lower() == "json_schema"
            else f"[{agent.id}] chat-mock"
        )
        echoed = {
            key: payload[key]
            for key in (
                "model",
                "response_format",
                "tools",
                "tool_choice",
                "temperature",
                "max_tokens",
                "instructions",
                "metadata",
                "messages",
                "top_logprobs",
                "text",
            )
            if key in payload
        }
        if endpoint.strip("/") == "responses":
            return {
                "id": f"resp_mock_{agent.id}",
                "object": "response",
                "model": agent.model,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": mock_content}],
                    }
                ],
                "echo": echoed,
            }
        return {
            "id": f"chatcmpl_mock_{agent.id}",
            "object": "chat.completion",
            "model": agent.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "echo": echoed,
        }

    def _validate_provider(self, agent: ModelAgent) -> ProviderDestination:
        """Reject unsafe model endpoints and return the exact address to connect to."""
        # Runtime secret must be resolvable from the KV — never an env var name,
        # never a silent os.getenv fallback. (Legacy api_key_env, if set, is used
        # only as the credential NAME; see ModelAgent.credential_name.)
        if _is_local_provider_url(agent.base_url):
            parsed = urlparse(agent.base_url)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise RuntimeError(f"{agent.id} local provider URL must not contain credentials or query data")
            addresses = self._resolve_addresses(parsed.hostname or "", parsed.port or 80)
            if any(not ipaddress.ip_address(sockaddr[0]).is_loopback for _family, sockaddr in addresses):
                raise RuntimeError(f"{agent.id} local provider resolves to a non-loopback address")
            return addresses[0]
        credential_name = _provider_credential_name(agent)
        if credential_name and get_credential(credential_name) is None:
            raise NotConfigured(
                f"{agent.id} requires a resolvable credential '{credential_name}' in the KV "
                "(this replaces the legacy api_key_env environment pattern)"
            )
        parsed = urlparse(agent.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RuntimeError(f"{agent.id} base_url must use https")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError(f"{agent.id} base_url must not contain credentials, query data, or fragments")
        hostname = parsed.hostname.lower()
        if self.allowed_provider_hosts and hostname not in self.allowed_provider_hosts:
            raise RuntimeError(f"{agent.id} provider host is not allowlisted")
        addresses = self._resolve_addresses(hostname, parsed.port or 443)
        for _family, sockaddr in addresses:
            ip_address = ipaddress.ip_address(sockaddr[0])
            if (
                ip_address.is_private
                or ip_address.is_loopback
                or ip_address.is_link_local
                or ip_address.is_multicast
                or ip_address.is_reserved
            ):
                raise RuntimeError(f"{agent.id} provider resolves to non-public address")
        return addresses[0]

    def _provider_url(self, agent: ModelAgent, path: str) -> str:
        """Build a provider URL while rejecting urllib-supported local schemes."""
        parsed = urlparse(agent.base_url)
        if _is_local_provider_url(agent.base_url):
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise RuntimeError(f"{agent.id} local provider URL must not contain credentials or query data")
            base_url = urlunsplit(("http", parsed.netloc, parsed.path.rstrip("/"), "", ""))
        elif parsed.scheme in {"http", "https"} and parsed.hostname:
            base_url = agent.base_url.rstrip("/")
        else:
            raise RuntimeError(f"{agent.id} base_url must be an http(s) provider URL")
        if not path.startswith("/") or path.startswith("//") or "\r" in path or "\n" in path:
            raise RuntimeError("provider path must be a single absolute URL path")
        return f"{base_url}{path}"

    def _mock(self, agent: ModelAgent, messages: list[ChatMessage]) -> str:
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        role = "worker"
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        match = re.search(r"Role: ([a-z]+)", system)
        if match:
            role = match.group(1)
        return f"[{agent.id}:{role}] {last[:220]}"

    # --- OpenAI Batch API (async, ~50% provider discount; NOT for latency-sensitive chat) ---

    def batch_chat(
        self,
        agent: ModelAgent,
        requests: dict[str, list[ChatMessage]],
        temperature: float | None = None,
        poll_interval: float = 5.0,
        poll_timeout: float = 3600.0,
        effort_profile: ReasoningEffortProfile | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run many chat requests through the provider's Batch API and return results by id.

        ``requests`` maps a caller custom_id to its messages. Suited to eval/benchmark
        workloads (24h completion window, ~half the price); real-time chat should keep
        using ``chat``. The mock path answers synchronously so tests and local runs work.
        """
        if not is_chat_compatible_model_id(agent.model):
            raise ValueError(
                f"model {agent.model!r} is not chat-compatible and cannot serve {agent.id!r}"
            )
        if agent.base_url.startswith("mock://"):
            results = {
                custom_id: {"content": self._mock(agent, messages), "usage": None}
                for custom_id, messages in requests.items()
            }
        elif _is_local_provider_url(agent.base_url):
            results = self._local_batch_chat(agent, requests, temperature, effort_profile)
        else:
            destination = self._validate_provider(agent)  # pragma: no cover
            batch_error: RuntimeError | None = None
            try:
                results = self._batch_run(  # pragma: no cover
                    agent, requests, temperature, poll_interval, poll_timeout, destination, effort_profile
                )
            except Exception:  # noqa: BLE001 - provider batch boundary (CWE-209)
                # Batch upload, polling, and output retrieval all cross the same
                # public gateway boundary; provider bodies and exception text stay
                # inside the authorized provider observability system.
                batch_error = RuntimeError(f"provider {agent.id} batch request failed")
            if batch_error is not None:
                raise batch_error
        return _validate_batch_results(requests, results)

    def _local_batch_chat(
        self,
        agent: ModelAgent,
        requests: dict[str, list[ChatMessage]],
        temperature: float | None,
        effort_profile: ReasoningEffortProfile | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run local OpenAI-compatible requests concurrently through mlx-lm."""
        request_settings = self.request_settings_snapshot()

        def complete(custom_id: str, messages: list[ChatMessage]) -> tuple[str, dict[str, Any]]:
            with self.request_settings(**request_settings):
                if effort_profile is not None:
                    content = self.chat(
                        agent,
                        messages,
                        temperature=temperature,
                        effort_profile=effort_profile,
                    )
                else:
                    content = self.chat(agent, messages, temperature=temperature)
                return custom_id, {"content": content, "usage": self.take_usage()}

        if self.local_concurrency == 1 or len(requests) <= 1:
            return dict(complete(custom_id, messages) for custom_id, messages in requests.items())
        with ThreadPoolExecutor(max_workers=min(self.local_concurrency, len(requests))) as pool:
            futures = [
                pool.submit(copy_context().run, complete, custom_id, messages)
                for custom_id, messages in requests.items()
            ]
            return dict(future.result() for future in futures)

    def _batch_run(
        self,
        agent: ModelAgent,
        requests: dict[str, list[ChatMessage]],
        temperature: float | None,
        poll_interval: float,
        poll_timeout: float,
        destination: ProviderDestination | None = None,
        effort_profile: ReasoningEffortProfile | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Upload, create, poll, and parse one batch (isolated so the flow stays testable)."""
        settings = self.request_settings_snapshot()
        lines = [
            json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self.apply_effort_profile(agent, {
                    "model": agent.model,
                    "messages": messages,
                    "temperature": settings["temperature"] if temperature is None else temperature,
                    "max_tokens": settings["max_output_tokens"],
                }, effort_profile),
            }, ensure_ascii=False)
            for custom_id, messages in requests.items()
        ]
        input_file_id = self._batch_upload(agent, "\n".join(lines).encode("utf-8"), destination)
        batch_id = self._batch_json(agent, "POST", "/batches", {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }, destination)["id"]
        deadline = time.monotonic() + poll_timeout
        while True:
            batch = self._batch_json(agent, "GET", f"/batches/{batch_id}", destination=destination)
            status = batch.get("status")
            if status == "completed":
                break
            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"batch {batch_id} ended with status {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"batch {batch_id} still {status} after {poll_timeout}s")
            self._sleep(poll_interval)
        raw = self._batch_raw(agent, f"/files/{batch['output_file_id']}/content", destination)
        results: dict[str, dict[str, Any]] = {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = (row.get("response") or {}).get("body") or {}
            choices = body.get("choices") or [{}]
            results[row.get("custom_id", "")] = {
                "content": (choices[0].get("message") or {}).get("content"),
                "usage": body.get("usage"),
            }
        return results

    def _batch_upload(
        self, agent: ModelAgent, payload: bytes, destination: ProviderDestination | None = None
    ) -> str:
        """Upload a JSONL batch input via multipart/form-data; returns the file id."""
        boundary = f"co-batch-{uuid.uuid4().hex}"
        api_key = get_credential(agent.credential_name) or ""
        body = (
            f"--{boundary}\r\ncontent-disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"batch.jsonl\"\r\n"
            "content-type: application/jsonl\r\n\r\n"
        ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
        request = urllib.request.Request(
            self._provider_url(agent, "/files"),
            data=body,
            headers={
                "authorization": f"{agent.auth_scheme} {api_key}",
                "content-type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with self._open_provider(request, destination) as response:
            return json.loads(response.read().decode("utf-8"))["id"]

    def _batch_json(
        self,
        agent: ModelAgent,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        destination: ProviderDestination | None = None,
    ) -> dict[str, Any]:
        api_key = get_credential(agent.credential_name) or ""
        request = urllib.request.Request(
            self._provider_url(agent, path),
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={
                "authorization": f"{agent.auth_scheme} {api_key}",
                "content-type": "application/json",
            },
            method=method,
        )
        with self._open_provider(request, destination) as response:
            return json.loads(response.read().decode("utf-8"))

    def _batch_raw(self, agent: ModelAgent, path: str, destination: ProviderDestination | None = None) -> bytes:
        api_key = get_credential(agent.credential_name) or ""
        request = urllib.request.Request(
            self._provider_url(agent, path),
            headers={"authorization": f"{agent.auth_scheme} {api_key}"},
            method="GET",
        )
        with self._open_provider(request, destination) as response:
            return response.read()


def _coerce_input_text(value: Any) -> str:
    """Best-effort text (for agent selection) from a Responses API ``input`` field."""
    if isinstance(value, str):
        return value
    parts: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # OpenAI content-parts: {"type": "text", "text": "..."}
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                            parts.append(chunk["text"])
    return " ".join(parts)


def _coerce_message_content_text(content: Any) -> str:
    """Best-effort plain text from chat message content (string or content-parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _coerce_input_text(content)
    return ""


def load_agents(path: str) -> list[ModelAgent]:  # pragma: no cover
    """Load model agent definitions from an agents JSON file."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return [ModelAgent.from_dict(item) for item in data["agents"]]


class _AgentPoolStore:
    """Durable, normalized agent-pool storage used across process restarts.

    Agent scalar attributes live in ``agent_pool``. Ordered tags and provider
    exclusions live in child tables, so a database row never hides a second
    unqueryable JSON document. Legacy ``agent_pool(agent_id, payload)`` files
    migrate transactionally on first open; malformed or ambiguous data fails
    closed without discarding the old tab    Model-group membership lives in normalized
    ``model_group``/``model_group_member`` relations beside the pool.
    """

    _AGENT_TABLE_NAME = "agent_pool"
    _TAG_TABLE_NAME = "agent_pool_tags"
    _EXCLUSION_TABLE_NAME = "agent_pool_provider_exclusions"
    _LEGACY_TABLE_NAME = "agent_pool_legacy_payloads"
    _AGENT_COLUMNS = frozenset(
        {
            "agent_id",
            "model_name",
            "base_url",
            "api_key_env",
            "credential_key",
            "priority",
            "disabled",
            "provider_name",
            "local_credential_key",
            "auth_scheme",
            "reasoning_effort_supported",
        }
    )

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        """Return whether the exact application-owned SQLite table exists."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _connect(path: str) -> sqlite3.Connection:
        """Open a pool connection with relationship integrity enabled first."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @classmethod
    def _create_normalized_schema(cls, conn: sqlite3.Connection) -> None:
        """Create the 3NF parent and ordered child tables if absent."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_pool (
                agent_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key_env TEXT NOT NULL,
                credential_key TEXT NOT NULL,
                priority INTEGER NOT NULL,
                disabled INTEGER NOT NULL,
                provider_name TEXT NOT NULL,
                local_credential_key TEXT NOT NULL,
                auth_scheme TEXT NOT NULL,
                reasoning_effort_supported INTEGER,
                CONSTRAINT agent_pool_disabled_flag_check CHECK (disabled IN (0, 1)),
                CONSTRAINT agent_pool_reasoning_effort_flag_check
                    CHECK (reasoning_effort_supported IS NULL OR reasoning_effort_supported IN (0, 1))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_pool_tags (
                agent_id TEXT NOT NULL,
                tag_position INTEGER NOT NULL,
                tag_name TEXT NOT NULL,
                CONSTRAINT agent_pool_tags_primary_key PRIMARY KEY (agent_id, tag_position),
                CONSTRAINT agent_pool_tags_agent_foreign_key
                    FOREIGN KEY (agent_id) REFERENCES agent_pool(agent_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_pool_provider_exclusions (
                agent_id TEXT NOT NULL,
                exclusion_position INTEGER NOT NULL,
                provider_name TEXT NOT NULL,
                CONSTRAINT agent_pool_provider_exclusions_primary_key
                    PRIMARY KEY (agent_id, exclusion_position),
                CONSTRAINT agent_pool_provider_exclusions_agent_foreign_key
                    FOREIGN KEY (agent_id) REFERENCES agent_pool(agent_id) ON DELETE CASCADE
            )
            """
        )

    @classmethod
    def _insert_agent(cls, conn: sqlite3.Connection, agent: "ModelAgent") -> None:
        """Insert one agent and its ordered multi-valued attributes."""
        config = agent.to_config()
        conn.execute(
            """
            INSERT INTO agent_pool (
                agent_id, model_name, base_url, api_key_env, credential_key,
                priority, disabled, provider_name, local_credential_key, auth_scheme,
                reasoning_effort_supported
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config["id"],
                config["model"],
                config["base_url"],
                config["api_key_env"],
                config["credential_key"],
                config["priority"],
                int(config["disabled"]),
                config["provider_name"],
                config["local_credential_key"],
                config["auth_scheme"],
                config["reasoning_effort_supported"],
            ),
        )
        conn.executemany(
            "INSERT INTO agent_pool_tags (agent_id, tag_position, tag_name) VALUES (?, ?, ?)",
            [(agent.id, position, tag) for position, tag in enumerate(agent.tags)],
        )
        conn.executemany(
            """
            INSERT INTO agent_pool_provider_exclusions
                (agent_id, exclusion_position, provider_name)
            VALUES (?, ?, ?)
            """,
            [
                (agent.id, position, provider)
                for position, provider in enumerate(agent.provider_exclusions)
            ],
        )

    @classmethod
    def _initialize_schema(cls, conn: sqlite3.Connection) -> None:
        """Create or transactionally migrate the agent-pool schema."""
        agent_exists = cls._table_exists(conn, cls._AGENT_TABLE_NAME)
        tag_exists = cls._table_exists(conn, cls._TAG_TABLE_NAME)
        exclusion_exists = cls._table_exists(conn, cls._EXCLUSION_TABLE_NAME)
        if not agent_exists:
            if tag_exists or exclusion_exists:
                raise RuntimeError("agent-pool child tables exist without agent_pool")
            cls._create_normalized_schema(conn)
            return

        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_pool)")}
        if "payload" in columns:
            if tag_exists or exclusion_exists:
                raise RuntimeError("legacy agent_pool conflicts with normalized child tables")
            conn.execute("ALTER TABLE agent_pool RENAME TO agent_pool_legacy_payloads")
            cls._create_normalized_schema(conn)
            rows = conn.execute(
                "SELECT payload FROM agent_pool_legacy_payloads ORDER BY agent_id"
            ).fetchall()
            for (payload,) in rows:
                cls._insert_agent(conn, ModelAgent.from_dict(json.loads(payload)))
            # The legacy table is kept until _migrate_legacy_groups promotes its
            # group_name fields into model_group_member; that caller drops it.
            return

        if "reasoning_effort_supported" not in columns:
            conn.execute(
                "ALTER TABLE agent_pool ADD COLUMN reasoning_effort_supported INTEGER "
                "CHECK (reasoning_effort_supported IS NULL OR reasoning_effort_supported IN (0, 1))"
            )
            columns.add("reasoning_effort_supported")
        if not cls._AGENT_COLUMNS.issubset(columns):
            missing = ", ".join(sorted(cls._AGENT_COLUMNS - columns))
            raise RuntimeError(f"unsupported agent_pool schema; missing columns: {missing}")
        cls._create_normalized_schema(conn)

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._path = path
        conn = self._connect(self._path)
        try:
            conn.execute("BEGIN")
            self._initialize_schema(conn)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS model_group (group_name TEXT PRIMARY KEY)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS model_group_member ("
                "agent_id TEXT PRIMARY KEY REFERENCES agent_pool(agent_id) ON DELETE CASCADE, "
                "group_name TEXT NOT NULL REFERENCES model_group(group_name) ON DELETE CASCADE)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS endpoint_equivalence_contract ("
                "contract_id TEXT PRIMARY KEY, model_revision TEXT NOT NULL, "
                "reasoning_effort_profile TEXT NOT NULL, structured_output_contract TEXT NOT NULL, "
                "accuracy_class TEXT NOT NULL, data_residency_policy TEXT NOT NULL, "
                "retention_policy TEXT NOT NULL, context_limit INTEGER NOT NULL CHECK(context_limit > 0), "
                "pricing_evidence_id TEXT NOT NULL, hedge_eligible INTEGER NOT NULL CHECK(hedge_eligible IN (0,1)), "
                "cancellation_supported INTEGER NOT NULL CHECK(cancellation_supported IN (0,1)), "
                "execution_policy TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS endpoint_equivalence_capability ("
                "contract_id TEXT NOT NULL REFERENCES endpoint_equivalence_contract(contract_id) ON DELETE CASCADE, "
                "capability_name TEXT NOT NULL, PRIMARY KEY(contract_id, capability_name))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS endpoint_equivalence_member ("
                "agent_id TEXT PRIMARY KEY REFERENCES agent_pool(agent_id) ON DELETE CASCADE, "
                "contract_id TEXT NOT NULL REFERENCES endpoint_equivalence_contract(contract_id) ON DELETE RESTRICT)"
            )
            self._migrate_legacy_groups(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _migrate_legacy_groups(conn: sqlite3.Connection) -> None:
        """Move legacy payload group names into the normalized membership relation.

        Legacy payload rows live in ``agent_pool_legacy_payloads`` only while
        ``_initialize_schema`` is mid-migration; when that table is present its
        ``group_name`` fields are promoted into ``model_group_member`` before it
        is dropped. Fresh normalized databases have no legacy table and this
        becomes a no-op.
        """
        legacy_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_pool_legacy_payloads'"
        ).fetchone() is not None
        if not legacy_exists:
            return
        for agent_id, raw_payload in list(
            conn.execute("SELECT agent_id, payload FROM agent_pool_legacy_payloads")
        ):
            try:
                group_name = json.loads(raw_payload).get("group_name", "")
            except (TypeError, ValueError):
                continue
            if not group_name:
                continue
            canonical = canonical_group_name(group_name)
            conn.execute(
                "INSERT OR IGNORE INTO model_group (group_name) VALUES (?)", (canonical,)
            )
            conn.execute(
                "INSERT OR REPLACE INTO model_group_member (agent_id, group_name) VALUES (?, ?)",
                (agent_id, canonical),
            )
        conn.execute("DROP TABLE agent_pool_legacy_payloads")

    def save(self, agent: "ModelAgent") -> None:
        """Persist one normalized model-agent definition."""
        with self._lock:
            conn = self._connect(self._path)
            try:
                config = agent.to_config()
                conn.execute(
                    """
                    UPDATE agent_pool SET
                        model_name = ?, base_url = ?, api_key_env = ?, credential_key = ?,
                        priority = ?, disabled = ?, provider_name = ?,
                        local_credential_key = ?, auth_scheme = ?,
                        reasoning_effort_supported = ?
                    WHERE agent_id = ?
                    """,
                    (
                        config["model"],
                        config["base_url"],
                        config["api_key_env"],
                        config["credential_key"],
                        config["priority"],
                        int(config["disabled"]),
                        config["provider_name"],
                        config["local_credential_key"],
                        config["auth_scheme"],
                        config["reasoning_effort_supported"],
                        agent.id,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0] == 0:
                    self._insert_agent(conn, agent)
                else:
                    conn.execute("DELETE FROM agent_pool_tags WHERE agent_id = ?", (agent.id,))
                    conn.execute(
                        "DELETE FROM agent_pool_provider_exclusions WHERE agent_id = ?",
                        (agent.id,),
                    )
                    conn.executemany(
                        "INSERT INTO agent_pool_tags (agent_id, tag_position, tag_name) VALUES (?, ?, ?)",
                        [(agent.id, position, tag) for position, tag in enumerate(agent.tags)],
                    )
                    conn.executemany(
                        """
                        INSERT INTO agent_pool_provider_exclusions
                            (agent_id, exclusion_position, provider_name)
                        VALUES (?, ?, ?)
                        """,
                        [
                            (agent.id, position, provider)
                            for position, provider in enumerate(agent.provider_exclusions)
                        ],
                    )
                # Model-group membership is a normalized relation beside the pool.
                conn.execute("DELETE FROM model_group_member WHERE agent_id = ?", (agent.id,))
                if agent.group_name:
                    conn.execute(
                        "INSERT OR IGNORE INTO model_group (group_name) VALUES (?)",
                        (agent.group_name,),
                    )
                    conn.execute(
                        "INSERT INTO model_group_member (agent_id, group_name) VALUES (?, ?)",
                        (agent.id, agent.group_name),
                    )
                conn.execute(
                    "DELETE FROM model_group WHERE NOT EXISTS ("
                    "SELECT 1 FROM model_group_member "
                    "WHERE model_group_member.group_name = model_group.group_name)"
                )
                conn.execute("DELETE FROM endpoint_equivalence_member WHERE agent_id = ?", (agent.id,))
                conn.execute(
                    "DELETE FROM endpoint_equivalence_contract WHERE NOT EXISTS ("
                    "SELECT 1 FROM endpoint_equivalence_member "
                    "WHERE endpoint_equivalence_member.contract_id = "
                    "endpoint_equivalence_contract.contract_id)"
                )
                if agent.endpoint_equivalence is not None:
                    contract = EndpointEquivalenceContract(**agent.endpoint_equivalence)
                    conn.execute(
                        "INSERT INTO endpoint_equivalence_contract VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(contract_id) DO UPDATE SET model_revision=excluded.model_revision, "
                        "reasoning_effort_profile=excluded.reasoning_effort_profile, "
                        "structured_output_contract=excluded.structured_output_contract, "
                        "accuracy_class=excluded.accuracy_class, data_residency_policy=excluded.data_residency_policy, "
                        "retention_policy=excluded.retention_policy, context_limit=excluded.context_limit, "
                        "pricing_evidence_id=excluded.pricing_evidence_id, hedge_eligible=excluded.hedge_eligible, "
                        "cancellation_supported=excluded.cancellation_supported, "
                        "execution_policy=excluded.execution_policy",
                        (
                            contract.contract_id, contract.model_revision,
                            contract.reasoning_effort_profile, contract.structured_output_contract,
                            contract.accuracy_class, contract.data_residency_policy,
                            contract.retention_policy, contract.context_limit,
                            contract.pricing_evidence_id, int(contract.hedge_eligible),
                            int(contract.cancellation_supported), contract.execution_policy,
                        ),
                    )
                    conn.execute(
                        "DELETE FROM endpoint_equivalence_capability WHERE contract_id = ?",
                        (contract.contract_id,),
                    )
                    conn.executemany(
                        "INSERT INTO endpoint_equivalence_capability (contract_id, capability_name) VALUES (?, ?)",
                        [(contract.contract_id, name) for name in contract.capability_set],
                    )
                    conn.execute(
                        "INSERT INTO endpoint_equivalence_member (agent_id, contract_id) VALUES (?, ?)",
                        (agent.id, contract.contract_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def load_all(self) -> list["ModelAgent"]:
        """Load every persisted model-agent definition."""
        with self._lock:
            conn = self._connect(self._path)
            try:
                rows = conn.execute(
                    """
                    SELECT agent_id, model_name, base_url, api_key_env, credential_key,
                           priority, disabled, provider_name, local_credential_key, auth_scheme,
                           reasoning_effort_supported
                    FROM agent_pool ORDER BY agent_id
                    """
                ).fetchall()
                tags = conn.execute(
                    "SELECT agent_id, tag_position, tag_name FROM agent_pool_tags "
                    "ORDER BY agent_id, tag_position"
                ).fetchall()
                exclusions = conn.execute(
                    "SELECT agent_id, exclusion_position, provider_name "
                    "FROM agent_pool_provider_exclusions ORDER BY agent_id, exclusion_position"
                ).fetchall()
                groups = conn.execute(
                    "SELECT agent_id, group_name FROM model_group_member ORDER BY agent_id"
                ).fetchall()
                contracts = conn.execute(
                    "SELECT endpoint_equivalence_member.agent_id, endpoint_equivalence_contract.* "
                    "FROM endpoint_equivalence_member JOIN endpoint_equivalence_contract USING (contract_id)"
                ).fetchall()
                contract_capabilities = conn.execute(
                    "SELECT contract_id, capability_name FROM endpoint_equivalence_capability "
                    "ORDER BY contract_id, capability_name"
                ).fetchall()
            finally:
                conn.close()
        tags_by_agent: dict[str, list[str]] = {}
        for agent_id, _position, tag_name in tags:
            tags_by_agent.setdefault(agent_id, []).append(tag_name)
        exclusions_by_agent: dict[str, list[str]] = {}
        for agent_id, _position, provider_name in exclusions:
            exclusions_by_agent.setdefault(agent_id, []).append(provider_name)
        group_by_agent: dict[str, str] = dict(groups)
        capabilities_by_contract: dict[str, list[str]] = {}
        for contract_id, capability_name in contract_capabilities:
            capabilities_by_contract.setdefault(contract_id, []).append(capability_name)
        contract_by_agent = {
            row[0]: {
                "contract_id": row[1], "model_revision": row[2],
                "reasoning_effort_profile": row[3],
                "structured_output_contract": row[4], "accuracy_class": row[5],
                "data_residency_policy": row[6], "retention_policy": row[7],
                "context_limit": row[8], "pricing_evidence_id": row[9],
                "hedge_eligible": bool(row[10]), "cancellation_supported": bool(row[11]),
                "execution_policy": row[12],
                "capability_set": tuple(capabilities_by_contract.get(row[1], ())),
            }
            for row in contracts
        }
        return [
            ModelAgent(
                id=row[0],
                model=row[1],
                base_url=row[2],
                api_key_env=row[3],
                credential_key=row[4],
                tags=tuple(tags_by_agent.get(row[0], ())),
                priority=row[5],
                disabled=bool(row[6]),
                provider_name=row[7],
                provider_exclusions=tuple(exclusions_by_agent.get(row[0], ())),
                local_credential_key=row[8],
                auth_scheme=row[9],
                reasoning_effort_supported=(None if row[10] is None else bool(row[10])),
                group_name=group_by_agent.get(row[0], ""),
                endpoint_equivalence=contract_by_agent.get(row[0]),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Compatibility no-op: agent-pool operations use short-lived sqlite handles."""


class _StateStore:
    """Minimal write-through sqlite persistence for orchestrator runtime state.

    ponytail: one generic table, no ORM. Keyed kinds (workflow_run, evaluation_run)
    upsert by key; stream kinds append. Streams saved as durable commit synchronously and use the same bounded
    retention as their in-memory deques so request traffic cannot grow the DB forever.
    Runtime values (kind, key, payload, limit) are always bound through SQLite
    placeholders so persisted prompts and identifiers cannot become SQL syntax.
    """

    _KEYED = {"workflow_run", "evaluation_run"}
    _TABLE_NAME = "orchestration_records"
    _LEGACY_TABLE_NAME = "records"
    _LEGACY_INDEX_NAME = "records_kind_seq"
    _INDEX_NAME = "orchestration_records_kind_seq"
    _STREAM_LIMITS = {"audit": 256, "authorization": 256, "analytics": 256}
    _CREATE_RECORDS_SQL = (
        "CREATE TABLE IF NOT EXISTS orchestration_records ("
        "seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, key TEXT, payload TEXT NOT NULL)"
    )
    _CREATE_RECORDS_KIND_SEQ_INDEX_SQL = (
        f"CREATE INDEX IF NOT EXISTS {_INDEX_NAME} ON {_TABLE_NAME}(kind, seq)"
    )
    _DELETE_KEYED_SQL = "DELETE FROM orchestration_records WHERE kind = ? AND key = ?"
    _INSERT_SQL = "INSERT INTO orchestration_records (kind, key, payload) VALUES (?, ?, ?)"
    _PRUNE_STREAM_SQL = (
        "DELETE FROM orchestration_records WHERE kind = ? AND seq NOT IN ("
        "SELECT seq FROM orchestration_records WHERE kind = ? ORDER BY seq DESC LIMIT ?)"
    )
    _SELECT_ALL_SQL = "SELECT payload FROM orchestration_records WHERE kind = ? ORDER BY seq"
    _SELECT_LIMIT_SQL = "SELECT payload FROM orchestration_records WHERE kind = ? ORDER BY seq DESC LIMIT ?"

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._migrate_legacy_table()
            self._conn.execute(self._CREATE_RECORDS_SQL)
            self._conn.execute(self._CREATE_RECORDS_KIND_SEQ_INDEX_SQL)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            self._conn.close()
            raise
        # Non-durable streams are best-effort, but each keeps its own newest
        # retention window so an authorization flood cannot evict audit data.
        self._stream_events: dict[str, deque[tuple[str | None, dict[str, Any]]]] = {
            kind: deque(maxlen=limit) for kind, limit in self._STREAM_LIMITS.items()
        }
        self._stream_condition = threading.Condition()
        self._stream_closing = False
        self._stream_writing = False
        self._next_stream_index = 0
        self._stream_worker = threading.Thread(
            target=self._drain_stream_queue,
            name="contextual-orchestrator-state-store",
            daemon=True,
        )
        self._stream_worker.start()

    def _migrate_legacy_table(self) -> None:
        """Rename the pre-policy table without discarding persisted state."""
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", ("table",)
            ).fetchall()
        }
        has_legacy = self._LEGACY_TABLE_NAME in tables
        has_current = self._TABLE_NAME in tables
        if has_legacy and has_current:
            raise RuntimeError(
                "state database contains both legacy and current persistence tables"
            )
        if has_legacy:
            # _LEGACY_TABLE_NAME/_TABLE_NAME/_LEGACY_INDEX_NAME are fixed
            # class-level string literals, never derived from request or
            # database content -- no injection surface despite the f-string shape.
            rename_sql = f"ALTER TABLE {self._LEGACY_TABLE_NAME} RENAME TO {self._TABLE_NAME}"
            drop_index_sql = f"DROP INDEX IF EXISTS {self._LEGACY_INDEX_NAME}"
            self._conn.execute(rename_sql)  # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query,python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
            self._conn.execute(drop_index_sql)  # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query,python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query

    def save(self, kind: str, key: str | None, payload: dict[str, Any], *, durable: bool = False) -> None:
        """Persist one typed state record, optionally on the durable path."""
        if kind in self._STREAM_LIMITS and not durable:
            with self._stream_condition:
                if self._stream_closing:
                    raise RuntimeError("state store is closed")
                self._stream_events[kind].append((key, payload))
                self._stream_condition.notify()
            return
        self._save_sync(kind, key, payload)

    def _save_sync(self, kind: str, key: str | None, payload: dict[str, Any]) -> None:
        blob = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            if kind in self._KEYED:
                self._conn.execute(self._DELETE_KEYED_SQL, (kind, key))
            self._conn.execute(self._INSERT_SQL, (kind, key, blob))
            if kind in self._STREAM_LIMITS:
                limit = self._STREAM_LIMITS[kind]
                self._conn.execute(self._PRUNE_STREAM_SQL, (kind, kind, limit))
            self._conn.commit()

    def _drain_stream_queue(self) -> None:
        while True:
            with self._stream_condition:
                while not self._stream_closing and not any(self._stream_events.values()):
                    self._stream_condition.wait()
                event = self._next_stream_event()
                if event is None:
                    return
                kind, key, payload = event
                self._stream_writing = True
            try:
                self._save_sync(kind, key, payload)
            except Exception:  # noqa: BLE001 - a best-effort stream write must not stop later persistence.
                pass
            finally:
                with self._stream_condition:
                    self._stream_writing = False
                    self._stream_condition.notify_all()

    def _next_stream_event(self) -> tuple[str, str | None, dict[str, Any]] | None:
        """Return one pending event fairly; caller holds ``_stream_condition``."""
        kinds = tuple(self._STREAM_LIMITS)
        for offset in range(len(kinds)):
            index = (self._next_stream_index + offset) % len(kinds)
            kind = kinds[index]
            if self._stream_events[kind]:
                self._next_stream_index = (index + 1) % len(kinds)
                key, payload = self._stream_events[kind].popleft()
                return kind, key, payload
        return None

    def _flush_streams(self) -> None:
        with self._stream_condition:
            while self._stream_writing or any(self._stream_events.values()):
                self._stream_condition.wait()

    def load(self, kind: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Load typed state records in insertion order."""
        self._flush_streams()
        with self._lock:
            if limit is None:
                rows = self._conn.execute(self._SELECT_ALL_SQL, (kind,)).fetchall()
            else:
                rows = self._conn.execute(self._SELECT_LIMIT_SQL, (kind, limit)).fetchall()
                rows = list(reversed(rows))
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        """Close the sqlite handle so Windows can release the database file."""
        self._flush_streams()
        with self._stream_condition:
            self._stream_closing = True
            self._stream_condition.notify_all()
        self._stream_worker.join()
        with self._lock:
            self._conn.close()


class _ResponseCache:
    """Exact-match TTL + LRU cache for orchestration results.

    ponytail: stdlib OrderedDict, no cache library. Deep-copies on the way in and out
    so callers can never mutate a cached entry. Thread-safe (the HTTP server is threaded).
    """

    def __init__(self, ttl: float, max_entries: int = 256, clock: Any = time.monotonic) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a detached cached response when it remains valid."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self._clock() - stored_at >= self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)  # LRU: most-recently used at the end
            return copy.deepcopy(value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a detached response and evict least-recently-used entries."""
        with self._lock:
            self._data[key] = (self._clock(), copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict least-recently used


class TaskOrchestrator:
    """Coordinate model routing, conducted workflows, governance, and audit state.

    Routing evidence policy (ADR 0034): ordering inputs are limited to
    (a) operator-declared configuration -- ``priority``, capability tags,
    provider exclusions, model groups, (b) literature-standard similarity --
    cosine similarity between task and agent-metadata embeddings
    (Karpukhin et al., 2020; Ong et al., 2024), and (c) measured evidence --
    the Beta-Bernoulli stability posterior (Laplace rule of succession) and
    Jacobson-style EWMA throughput ledgers in :mod:`.model_group`. Keyword
    hint tables and hand-tuned integer weights are intentionally absent.
    """

    # Role-to-capability-tag mapping over operator-maintained agent tags. Used
    # as a binary eligibility preference (fit / no fit), never weighted.
    ROLE_TAGS = {
        "thinker": ("planning", "reasoning", "research"),
        "worker": ("coding", "implementation", "reasoning"),
        "verifier": ("verification", "security", "review", "debugging"),
        "judge": ("verification", "security", "review", "debugging"),
        "synthesizer": ("writing", "reasoning", "planning"),
        "embedding": ("embedding",),
    }
    AUTO_MODEL = "orchestrator/auto"
    FREE_MODEL = "orchestrator/free"

    #: Bound on cached task/descriptor embedding vectors and triage verdicts.
    #: Operational memory bound mirroring ``cache_max_entries``; not a routing weight.
    EVIDENCE_CACHE_MAX_ENTRIES = 512

    def __init__(
        self,
        agents: list[ModelAgent],
        client: ModelClient | None = None,
        price_per_million: dict[str, float] | None = None,
        budget_max_output_tokens: int | None = None,
        budget_max_cost_usd: float | None = None,
        state_db: str | None = None,
        agents_db: str | None = None,
        cache_ttl: float = 0.0,
        cache_max_entries: int = 256,
        tool_retry_attempts: int = 1,
        tool_retry_backoff_seconds: float = 0.25,
        cache_provider: ResponseCacheProvider | None = None,
        role_effort_catalog: dict[str, ReasoningEffortProfile] | None = None,
        pii_key_name: str = DEFAULT_PII_KEY_NAME,
    ) -> None:
        # Optional durable model-group management: stored operator changes overlay the
        # seed agents file at startup (stored rows win by id; stored-new rows append).
        self._pool_store = _AgentPoolStore(agents_db) if agents_db else None
        if self._pool_store is not None:
            stored = {agent.id: agent for agent in self._pool_store.load_all()}
            agents = [stored.pop(agent.id, agent) for agent in agents] + list(stored.values())
        self.candidates = list(agents)
        self.agents = [agent for agent in self.candidates if not agent.disabled]
        if not self.agents:  # pragma: no cover
            raise ValueError("at least one enabled agent is required")
        # Measured speed/stability routing inside model groups (global: every
        # selection path below funnels through _ranked_agents). Ledger state is
        # process-local by design: it reflects this instance's observed traffic
        # and resets on restart, never carrying stale evidence across pools.
        self._group_router = ModelGroupRouter()
        # Quality ledger: identical estimator family as the transport ledger but
        # fed by real-time fast-mlsirm judge verdicts on final answers, so
        # measured accuracy -- not transport success -- steers future routing.
        self._quality_router = ModelGroupRouter()
        for grouped in self.candidates:
            self._group_router.register_member(grouped.id)
            self._quality_router.register_member(grouped.id)
        # Evidence caches (bounded, thread-safe): semantic-affinity vectors for
        # task text and agent metadata, plus strict triage verdicts keyed by
        # content hash. Bounds are operational memory limits, never weights.
        self._evidence_lock = threading.Lock()
        self._task_vector_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._descriptor_vector_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._triage_cache: OrderedDict[str, bool] = OrderedDict()
        # Strict structured verdict parser seam; tests may substitute it, and
        # production always uses the exact-schema implementation below.
        self._triage_fn = self._triage_workflow_required
        self.client = client or ModelClient()
        # The cost coordinator installs this optional sink. Direct orchestrator
        # callers still retain audit evidence without inventing price or usage.
        self._race_usage_sink: Callable[[str, Any], None] | None = None
        if (
            isinstance(tool_retry_attempts, bool)
            or not isinstance(tool_retry_attempts, int)
            or tool_retry_attempts < 0
            or tool_retry_attempts > MAX_TOOL_RETRY_ATTEMPTS
        ):
            raise ValueError(
                "tool_retry_attempts must be a nonnegative integer at most "
                f"{MAX_TOOL_RETRY_ATTEMPTS}"
            )
        self.tool_retry_attempts = tool_retry_attempts
        if (
            isinstance(tool_retry_backoff_seconds, bool)
            or not isinstance(tool_retry_backoff_seconds, (int, float))
            or not math.isfinite(float(tool_retry_backoff_seconds))
            or tool_retry_backoff_seconds < 0
        ):
            raise ValueError(
                "tool_retry_backoff_seconds must be a finite nonnegative number"
            )
        self.tool_retry_backoff_seconds = float(tool_retry_backoff_seconds)
        # Injectable seams keep retry timing deterministic in tests while
        # production uses full jitter to avoid synchronized retry bursts.
        self._tool_retry_sleep = time.sleep
        self._tool_retry_jitter = random.uniform
        self.policy = OrchestrationPolicy()
        # Opt-in issue #568 catalog. None keeps production answers and payload
        # keys unchanged. Operator next action: pass default_role_effort_catalog()
        # to attach a replayable snapshot; do not treat that as a default change.
        self.role_effort_catalog = role_effort_catalog
        # Operator-supplied USD price per 1M tokens, keyed by model. Empty => cost not computed.
        self.price_per_million = dict(price_per_million or {})
        # Operator spend caps; None => disabled (no behavior change). Enforced in run().
        self.budget_max_output_tokens = budget_max_output_tokens
        self.budget_max_cost_usd = budget_max_cost_usd
        self._workflow_runs: dict[str, dict[str, Any]] = {}
        self._budget_spend_lock = threading.Lock()
        self._budget_spent_output_tokens = 0
        self._budget_spent_cost_usd = Decimal(0)
        self._budget_model_output_tokens: dict[str, int] = {}
        self._evaluation_runs: dict[str, dict[str, Any]] = {}
        self._analytics_events: deque[dict[str, Any]] = deque(maxlen=256)
        self._audit_events: deque[dict[str, Any]] = deque(maxlen=256)
        self._authorization_events: deque[dict[str, Any]] = deque(maxlen=256)
        self._run_order: deque[str] = deque(maxlen=128)
        # Per-agent circuit breaker: consecutive failures trip an agent "open"
        # so a persistently failing provider is skipped until it cools down.
        self._circuit: dict[str, dict[str, float]] = {}
        self._circuit_lock = threading.Lock()
        self._provider_readiness_lock = threading.Lock()
        self.circuit_failure_threshold = 3
        self.circuit_reset_seconds = 30.0
        # Optional exact-match response cache: default ttl 0 disables it (no behavior change).
        if cache_provider is not None and cache_ttl:
            raise ValueError("cache_provider and cache_ttl cannot both be configured")
        self._cache_provider = cache_provider
        self._cache = _ResponseCache(cache_ttl, cache_max_entries) if cache_ttl and cache_ttl > 0 else None
        # Optional durable persistence: default None keeps all state purely in-memory
        # (zero behavior change). When set, runs/audit/analytics survive restart.
        self._store = _StateStore(state_db) if state_db else None
        if not isinstance(pii_key_name, str) or not pii_key_name:
            raise ValueError("pii_key_name must be a non-empty string")
        self._pii_key_name = pii_key_name
        self._pii_encryptors: dict[str, PiiFieldEncryptor] = {}
        self._commercial_report_cache_local = threading.local()
        if self._store is not None:
            self._reload_state()

    def close(self) -> None:
        """Release optional durable resources owned by this orchestrator."""
        if self._pool_store is not None:
            self._pool_store.close()
        if self._store is not None:
            self._store.close()

    def provider_readiness_report(
        self,
        *,
        refresh: bool = False,
        timeout: float = DEFAULT_PROVIDER_PROBE_TIMEOUT,
    ) -> dict[str, Any]:
        """Report provider liveness separately from an explicit chat readiness probe."""
        if type(refresh) is not bool:
            raise ValueError("refresh must be a boolean")
        probe_timeout = _validate_provider_probe_timeout(timeout)
        items: list[dict[str, Any]] = []
        with self._provider_readiness_lock:
            for agent in self.candidates:
                provider = agent.provider_name or self._infer_provider_name(agent.base_url)
                if agent.disabled:
                    items.append({
                        "agent_id": agent.id,
                        "model": agent.model,
                        "provider": provider,
                        "status": "disabled",
                    })
                    continue
                if refresh:
                    item = dict(self.client.probe(agent, timeout=probe_timeout))
                    item["provider"] = provider
                    items.append(redact_value(item))
                else:
                    items.append({
                        "agent_id": agent.id,
                        "model": agent.model,
                        "provider": provider,
                        "status": "unprobed",
                    })
        active = [item for item in items if item["status"] != "disabled"]
        status = "unprobed" if not refresh else (
            "ready" if active and all(item["status"] == "ready" for item in active) else "not_ready"
        )
        return {
            "status": status,
            "probe": "refresh" if refresh else "none",
            "timeout_seconds": probe_timeout,
            "checked_at": int(time.time()) if refresh else None,
            "agent_count": len(active),
            "ready_agent_count": sum(item["status"] == "ready" for item in active),
            "items": items,
        }

    def _reload_state(self) -> None:
        for record in self._store.load("workflow_run"):
            self._replace_workflow_run(record)
            self._run_order.appendleft(record["workflow_run_id"])
        for evaluation in self._store.load("evaluation_run"):
            self._evaluation_runs[evaluation["evaluation_run_id"]] = evaluation
        for event in self._store.load("analytics", self._analytics_events.maxlen):
            self._analytics_events.append(event)
        for event in self._store.load("audit", self._audit_events.maxlen):
            self._audit_events.append(event)
        for event in self._store.load("authorization", self._authorization_events.maxlen):
            self._authorization_events.append(event)

    # Orchestration-only body keys that must not be forwarded to the provider.
    _ORCHESTRATION_ONLY_KEYS = frozenset(
        {
            "orchestration",
            "orchestration_mode",
            "mode",
            "include_orchestration_trace",
            "attribution",
            "routing",
        }
    )

    def proxy_completion(
        self,
        body: dict[str, Any],
        *,
        endpoint: str = "chat/completions",
        effort_profile: ReasoningEffortProfile | None = None,
        single_agent: bool = True,
    ) -> dict[str, Any]:
        """Serve provider-shaped requests through orchestration or explicit passthrough.

        Structured and Responses requests conduct the normal evidence workflow
        when the HTTP boundary opts in. Omit-equivalent controls (JSON nulls,
        empty objects/arrays, blank strings, and the honest no-op keywords such
        as ``tool_choice="auto"`` without tools) never opt a request in on their
        own — they take the plain passthrough path exactly like an absent key.
        Direct callers retain the established single-provider passthrough
        contract.
        """
        normalized_endpoint = endpoint.strip("/")
        if not single_agent and (
            normalized_endpoint == "responses"
            or any(
                key in body
                and not _is_omit_equivalent_control(key, body.get(key))
                for key in _PASSTHROUGH_TRIGGER_KEYS
            )
        ):
            return self._orchestrated_provider_completion(
                body,
                endpoint=normalized_endpoint,
                effort_profile=effort_profile,
            )
        messages = body.get("messages")
        if isinstance(messages, list):
            text = self._latest_user_text(messages)
        else:
            text = _coerce_input_text(body.get("input"))
        requested_model = body.get("model")
        # When the client names a model, resolve a pool agent that actually serves
        # that model id (never silently rewrite to an unrelated agent.model --
        # a commercial honesty failure for OpenAI SDK passthrough tools/Responses
        # paths). _requested_agent already fails closed on unconfigured/empty
        # model ids; disabled-agent rejection happens here so the error message
        # is specific to why the request can't be served.
        agent = self._requested_agent(requested_model)
        if agent is not None and agent.disabled:
            raise RuntimeError(f"requested model {requested_model!r} is disabled")
        if agent is None:
            agent = self._select_agent(
                text, "worker", free_only=requested_model == self.FREE_MODEL
            )
        upstream = {
            key: value
            for key, value in body.items()
            if key not in self._ORCHESTRATION_ONLY_KEYS
        }
        upstream["model"] = agent.model
        # v1 passthrough returns the full JSON body; SSE stream passthrough is a
        # follow-up, so force a non-streamed upstream response here.
        upstream["stream"] = False
        if requested_model not in (
            None,
            "contextual-orchestrator",
            self.AUTO_MODEL,
            self.FREE_MODEL,
        ):
            if effort_profile is not None:
                upstream = self.client.apply_effort_profile(agent, upstream, effort_profile)
            measured = bool(agent.group_name or requested_model == self.FREE_MODEL)
            started_at = time.perf_counter()
            try:
                result = self.client.proxy_send(agent, endpoint, upstream)
            except Exception:
                if measured:
                    self._group_router.observe_failure(agent.id)
                raise
            if measured:
                self._group_router.observe_success(
                    agent.id, time.perf_counter() - started_at
                )
            return result

        allowed_agent_ids = (
            {candidate.id for candidate in self.agents if self._is_free_agent(candidate)}
            if requested_model == self.FREE_MODEL
            else (
                {candidate.id for candidate in self.agents}
                if requested_model == self.AUTO_MODEL
                else None
            )
        )
        # Cross-provider failover lives ONLY on this plain virtual passthrough
        # path (and the virtual tools path reached with single_agent=True).
        # Conducted structured synthesis never replays across providers — see
        # _orchestrated_provider_completion.
        ranked_candidates = self._failover_candidates(
            agent, text, "worker", allowed_agent_ids=allowed_agent_ids
        )
        if (
            effort_profile is not None
            and effort_profile.unsupported_provider_fallback != "omit"
        ):
            supported = [
                candidate
                for candidate in ranked_candidates
                if candidate.reasoning_effort_supported is True
                or (
                    candidate.reasoning_effort_supported is None
                    and candidate.base_url.startswith("mock://")
                )
            ]
            if supported:
                ranked_candidates = supported
        candidates: list[ModelAgent] = []
        seen_providers: set[str] = set()
        for candidate in ranked_candidates:
            provider_key = (
                f"provider:{candidate.provider_name.casefold()}"
                if candidate.provider_name.strip()
                else f"endpoint:{candidate.base_url.rstrip('/').casefold()}"
            )
            if provider_key in seen_providers:
                continue
            seen_providers.add(provider_key)
            candidates.append(candidate)
        last_error: Exception | None = None
        for candidate in candidates:
            started_at = time.perf_counter()
            candidate_payload = dict(upstream)
            candidate_payload["model"] = candidate.model
            if effort_profile is not None:
                candidate_payload = self.client.apply_effort_profile(
                    candidate, candidate_payload, effort_profile
                )
            try:
                send_once = getattr(self.client, "proxy_send_once", None)
                if not callable(send_once):
                    send_once = self.client.proxy_send
                result = send_once(candidate, endpoint, candidate_payload)
            except Exception as exc:  # noqa: BLE001 - provider trust boundary
                if not _is_passthrough_failover_error(exc):
                    raise
                last_error = exc
                self._record_failure(candidate.id)
                if candidate.group_name:
                    self._group_router.observe_failure(candidate.id)
                continue
            self._record_success(candidate.id)
            if candidate.group_name:
                self._group_router.observe_success(
                    candidate.id, time.perf_counter() - started_at
                )
            return result
        raise RuntimeError(
            f"all {len(candidates)} candidate agents failed for passthrough endpoint={endpoint}"
        ) from last_error

    def _orchestrated_provider_completion(
        self,
        body: dict[str, Any],
        *,
        endpoint: str,
        effort_profile: ReasoningEffortProfile | None,
    ) -> dict[str, Any]:
        """Conduct evidence work, then preserve the caller's provider contract.

        Structured synthesis here is INTENTIONALLY single-provider: the final
        synthesized response is produced by exactly one selected synthesizer
        agent with no cross-provider retry loop. Cross-provider failover is a
        property of the plain virtual passthrough / virtual tools paths only
        (see ``proxy_completion``), where a raw request can be replayed on a
        different provider without changing its meaning; replaying a conducted
        synthesis would mix evidence and attribution across providers, so it
        stays single-shot and fails closed instead.
        """
        response_request = endpoint == "responses"
        chat_body = _responses_to_chat_payload(body) if response_request else dict(body)
        messages = chat_body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("structured completion requires non-empty messages")
        task = self._latest_user_text(messages)
        required_tags = ("vision",) if self._source_image_parts(messages) else ()
        selection_tags = (
            *required_tags,
            *(("response_format",) if chat_body.get("response_format") else ()),
        )
        requested_model = body.get("model")
        free_only = requested_model == self.FREE_MODEL
        final_agent = self._requested_agent(requested_model)
        if final_agent is None:
            try:
                final_agent = self._select_agent(
                    task,
                    "synthesizer",
                    required_tags=selection_tags,
                    free_only=free_only,
                )
            except RuntimeError as exc:
                if selection_tags:
                    raise ValueError(
                        "no enabled model supports required tags: "
                        + ", ".join(selection_tags)
                    ) from exc
                raise
        elif any(tag not in final_agent.tags for tag in required_tags):
            raise ValueError(
                f"requested model {requested_model!r} lacks required tags: "
                + ", ".join(required_tags)
            )
        if final_agent.disabled:
            raise RuntimeError(f"requested model {requested_model!r} is disabled")

        self._raise_if_spend_budget_exceeded()
        workflow = self.conduct(
            messages,
            model_name=self.FREE_MODEL if free_only else "contextual-orchestrator",
        )
        in_flight_tokens = sum(_step_output_token_count(step) for step in workflow["trace"])
        model_by_agent = {agent.id: agent.model for agent in self.agents}
        in_flight_cost = sum(
            _step_output_token_count(step)
            / 1_000_000
            * self.price_per_million[model]
            for step in workflow["trace"]
            if (
                model := model_by_agent.get(
                    step.get("served_agent_id") or step.get("agent_id")
                )
            )
            in self.price_per_million
        )
        self._raise_if_spend_budget_exceeded(
            additional_output_tokens=in_flight_tokens,
            additional_cost_usd=round(in_flight_cost, 6),
        )

        evidence = "\n\n".join(
            f"Workflow step {step['id']} ({step['role']}):\n{step['output']}"
            for step in workflow["trace"]
        )
        guidance = (
            "You are the final synthesizer in a multi-agent workflow. "
            "Use the original request and verified workflow evidence. Return only "
            "the requested provider response; do not mention the workflow or invent "
            f"evidence.\n\nVerified workflow evidence:\n{evidence}"
        )
        if response_request:
            upstream = {
                key: value
                for key, value in body.items()
                if key not in self._ORCHESTRATION_ONLY_KEYS and key != "model"
            }
            upstream.pop("max_tokens", None)
            upstream.pop("max_completion_tokens", None)
            original_instructions = _responses_text(body.get("instructions")).strip()
            upstream["instructions"] = (
                f"{original_instructions}\n\n{guidance}"
                if original_instructions
                else guidance
            )
            upstream["model"] = final_agent.model
            upstream["stream"] = False
        else:
            synthesis_messages = copy.deepcopy(messages)
            guidance_index = next(
                (
                    index
                    for index in range(len(synthesis_messages) - 1, -1, -1)
                    if synthesis_messages[index].get("role") == "user"
                ),
                None,
            )
            if guidance_index is None:
                synthesis_messages.insert(0, {"role": "system", "content": guidance})
            else:
                content = synthesis_messages[guidance_index].get("content")
                if isinstance(content, list):
                    synthesis_messages[guidance_index]["content"] = [
                        *content,
                        {"type": "text", "text": guidance},
                    ]
                else:
                    synthesis_messages[guidance_index]["content"] = (
                        f"{content}\n\n{guidance}" if isinstance(content, str) else guidance
                    )
            upstream = {
                key: value
                for key, value in chat_body.items()
                if key not in self._ORCHESTRATION_ONLY_KEYS
                and key not in {"model", "messages"}
            }
            upstream.update(
                {
                    "model": final_agent.model,
                    "messages": synthesis_messages,
                    "stream": False,
                }
            )
        active_profile = effort_profile or self._role_effort_profile("synthesizer")
        if active_profile is not None:
            upstream = self.client.apply_effort_profile(
                final_agent,
                upstream,
                active_profile,
            )
        synthesis_started = time.perf_counter()
        # Single-provider by design: no cross-provider failover on this
        # conducted synthesis call (see the docstring above) — only the plain
        # virtual passthrough / virtual tools paths fail over across providers.
        raw = self.client.proxy_send(final_agent, endpoint, upstream)
        def provider_output(response: Mapping[str, Any]) -> str:
            if not response_request:
                try:
                    return ModelClient._response_content(final_agent, response)
                except RuntimeError:
                    return ""
            output = response.get("output_text")
            if isinstance(output, str):
                return output
            return "".join(
                    _responses_text(item.get("content"))
                    for item in response.get("output", [])
                    if isinstance(item, dict) and item.get("type") == "message"
                )

        synthesis_output = provider_output(raw)
        synthesis_step: dict[str, Any] = {
            "id": len(workflow["trace"]),
            "role": "synthesizer",
            "agent_id": final_agent.id,
            "subtask": "Provider-facing structured synthesis",
            "access": [step["id"] for step in workflow["trace"]],
            "latency_ms": round((time.perf_counter() - synthesis_started) * 1000, 2),
            "output": synthesis_output,
        }
        if isinstance(raw.get("usage"), dict):
            synthesis_step["usage"] = _canonical_provider_usage(
                raw["usage"], responses=response_request
            )
        repair_step: dict[str, Any] | None = None
        response_format = chat_body.get("response_format")
        contract_error = _structured_output_error(synthesis_output, response_format)
        if contract_error == "schema_missing":
            raise ProviderResponseError(
                "response_format.json_schema is missing a schema"
            )
        if contract_error is not None:
            repair_output_tokens = _step_output_token_count(synthesis_step)
            repair_output_cost = 0.0
            if final_agent.model in self.price_per_million:
                repair_output_cost = round(
                    repair_output_tokens / 1_000_000 * self.price_per_million[final_agent.model],
                    6,
                )
            self._raise_if_spend_budget_exceeded(
                additional_output_tokens=in_flight_tokens + repair_output_tokens,
                additional_cost_usd=round(in_flight_cost + repair_output_cost, 6),
            )
            repair_upstream = copy.deepcopy(upstream)
            repair_instruction = (
                "The prior synthesis violated the caller's strict JSON Schema "
                f"({contract_error}). Regenerate the complete answer and return only "
                "JSON that satisfies the supplied response_format."
            )
            if response_request:
                current = repair_upstream.get("instructions")
                repair_upstream["instructions"] = (
                    f"{current}\n\n{repair_instruction}"
                    if isinstance(current, str) and current
                    else repair_instruction
                )
            else:
                repair_messages = repair_upstream.get("messages")
                if not isinstance(repair_messages, list):
                    raise ProviderResponseError("structured synthesis omitted messages")
                repair_upstream["messages"] = [
                    *repair_messages,
                    {"role": "system", "content": repair_instruction},
                ]
            repair_started = time.perf_counter()
            repaired = self.client.proxy_send(final_agent, endpoint, repair_upstream)
            repaired_output = provider_output(repaired)
            if _structured_output_error(repaired_output, response_format) is not None:
                raise ProviderResponseError(
                    "structured synthesis and repair violated response_format"
                )
            repair_step = {
                "id": synthesis_step["id"] + 1,
                "role": "repair",
                "agent_id": final_agent.id,
                "subtask": "Strict JSON Schema repair",
                "access": [synthesis_step["id"]],
                "latency_ms": round((time.perf_counter() - repair_started) * 1000, 2),
                "output": repaired_output,
            }
            if isinstance(repaired.get("usage"), dict):
                repair_step["usage"] = _canonical_provider_usage(
                    repaired["usage"], responses=response_request
                )
            raw = repaired
            synthesis_output = repaired_output
        if response_request:
            raw.setdefault("output_text", synthesis_output)
        echo = raw.get("echo")
        if isinstance(echo, dict):
            if response_request:
                original_instructions = body.get("instructions")
                if isinstance(original_instructions, str) and original_instructions.strip():
                    echo["instructions"] = original_instructions
                else:
                    echo.pop("instructions", None)
            elif "messages" in echo:
                echo["messages"] = copy.deepcopy(messages)
        workflow_run_id = f"run_{uuid.uuid4().hex}"
        trace = [
            *workflow["trace"],
            synthesis_step,
            *([repair_step] if repair_step is not None else []),
        ]
        record = self._with_effort_snapshot(
            {
                "workflow_run_id": workflow_run_id,
                "created_at": int(time.time()),
                "mode": "conduct",
                "policy_mode": "conduct",
                "prompt_text": task,
                "answer": synthesis_output,
                "cache_status": "bypass",
                "trace": trace,
                "policy_snapshot": self.policy.as_dict(),
                "verification": workflow.get("verification"),
            }
        )
        self._workflow_runs[workflow_run_id] = record
        self._run_order.appendleft(workflow_run_id)
        if self._store is not None:
            self._store.save("workflow_run", workflow_run_id, record)
        self._append_audit_event(
            "workflow_run_created",
            {"workflow_run_id": workflow_run_id, "mode": "conduct", "agent_count": len(trace)},
        )
        self.record_analytics_event(
            "workflow_run_created",
            {
                "workflow_run_id": workflow_run_id,
                "run_mode": "conduct",
                "policy_mode": "conduct",
                "trace_step_count": len(trace),
                "trace_complete": self._is_trace_complete(record),
            },
        )
        raw["orchestration"] = {
            "workflow_run_id": workflow_run_id,
            "mode": "conduct",
            "agent_count": len(trace),
            "plan_source": workflow.get("plan_source"),
        }
        return raw

    def _requested_agent(self, requested_model: Any) -> ModelAgent | None:
        """Resolve an explicit model without silently serving a different model."""
        if requested_model is None or requested_model in {
            "contextual-orchestrator", self.AUTO_MODEL, self.FREE_MODEL
        }:
            return None
        if type(requested_model) is not str or not requested_model:
            raise ValueError("requested model must be a configured non-empty string")
        matches = [candidate for candidate in self.candidates if candidate.model == requested_model]
        if not matches:
            try:
                requested_group = canonical_group_name(requested_model)
            except ValueError:
                requested_group = ""
            matches = [
                candidate
                for candidate in self._ranked_agents("", "worker")
                if candidate.group_name
                and canonical_group_name(candidate.group_name) == requested_group
            ]
        if not matches:
            raise ValueError(f"requested model {requested_model!r} is not configured")
        return next((candidate for candidate in matches if not candidate.disabled), matches[0])

    def complete(
        self,
        messages: list[ChatMessage],
        mode: str = "auto",
        *,
        bypass_cache: bool = False,
        model_name: str = "contextual-orchestrator",
        cache_partition: str | None = None,
    ) -> dict[str, Any]:
        """Return a route or conducted completion without persisting a workflow run."""
        if not isinstance(bypass_cache, bool):
            raise TypeError("bypass_cache must be a boolean")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if cache_partition is not None and (not isinstance(cache_partition, str) or not cache_partition.strip()):
            raise ValueError("cache_partition must be a non-empty string when provided")
        cache = self._cache_provider if self._cache_provider is not None else self._cache
        if cache is None or bypass_cache:
            result = self._dispatch(messages, mode, model_name)
            result["cache_status"] = "bypass" if bypass_cache else "disabled"
            return result
        try:
            key = self._cache_key(messages, mode, model_name, cache_partition)
        except (TypeError, ValueError):
            # Cache key serialization is an optimization boundary; unusual but
            # valid caller objects must still reach the live provider path.
            result = self._dispatch(messages, mode, model_name)
            result["cache_status"] = "miss"
            return result
        try:
            cached = cache.get(key)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            cached = None
        if (
            isinstance(cached, Mapping)
            and isinstance(cached.get("mode"), str)
            and isinstance(cached.get("answer"), str)
            and isinstance(cached.get("trace"), list)
        ):
            result = copy.deepcopy(dict(cached))
            result["cache_status"] = "hit"
            return result
        result = self._dispatch(messages, mode, model_name)
        try:
            cache.put(key, result)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            pass
        result["cache_status"] = "miss"
        return result

    def _dispatch(
        self,
        messages: list[ChatMessage],
        mode: str,
        model_name: str = "contextual-orchestrator",
    ) -> dict[str, Any]:
        text = self._latest_user_text(messages)
        if mode == "route" or (
            mode == "auto"
            and (
                model_name not in {"contextual-orchestrator", self.AUTO_MODEL, self.FREE_MODEL}
                or not self._needs_workflow(text)
            )
        ):
            return self.route_once(messages, model_name=model_name)
        return self.conduct(messages, model_name=model_name)

    def would_route(
        self,
        messages: list[ChatMessage],
        mode: str = "auto",
        model_name: str = "contextual-orchestrator",
    ) -> bool:
        """True when this request takes the single-worker route path (vs the conduct workflow)."""
        text = self._latest_user_text(messages)
        return mode == "route" or (
            mode == "auto"
            and (
                model_name not in {"contextual-orchestrator", self.AUTO_MODEL, self.FREE_MODEL}
                or not self._needs_workflow(text)
            )
        )

    def stream_route(
        self,
        messages: list[ChatMessage],
        workflow_run_id: str | None = None,
        *,
        model_name: str = "contextual-orchestrator",
        owner_id: str | None = None,
    ):
        """Stream a single worker's content deltas as they arrive, then persist the run.

        True streaming for the route path. ponytail: no cross-agent failover here — bytes
        already sent can't be recalled, so a mid-stream provider failure surfaces to the caller.
        """
        text = self._latest_user_text(messages)
        agent = self._requested_agent(model_name) or self._select_agent(
            text, "worker", free_only=model_name == self.FREE_MODEL
        )
        parts: list[str] = []
        effort_profile = self._role_effort_profile("worker")
        stream = (
            self.client.stream_chat(agent, messages, effort_profile=effort_profile)
            if effort_profile is not None
            else self.client.stream_chat(agent, messages)
        )
        started_at = time.perf_counter()
        try:
            for delta in stream:
                parts.append(delta)
                yield delta
        except Exception:
            if agent.group_name or model_name == self.FREE_MODEL:
                self._group_router.observe_failure(agent.id)
            raise
        if agent.group_name or model_name == self.FREE_MODEL:
            self._group_router.observe_success(agent.id, time.perf_counter() - started_at)
        answer = "".join(parts)
        # Real-time judging after the stream: already-sent bytes cannot be
        # recalled, so the verdict never changes this response -- it feeds the
        # quality ledger so measured accuracy steers future member ordering,
        # and it is persisted for audit.
        latency_seconds = time.perf_counter() - started_at
        verification = self._realtime_route_judge(
            text=text,
            answer=answer,
            served_id=agent.id,
            latency_seconds=latency_seconds,
            usage=None,
            free_only=model_name == self.FREE_MODEL,
        )
        record = self._with_effort_snapshot(
            {
                "workflow_run_id": workflow_run_id or f"run_{uuid.uuid4().hex}",
                "created_at": int(time.time()),
                "mode": "route",
                "policy_mode": "route",
                "prompt_text": text,
                "answer": answer,
                "trace": [
                    {"id": 0, "role": "worker", "agent_id": agent.id, "subtask": "Direct route (streamed)",
                     "access": [], "output": answer}
                ],
                "policy_snapshot": self.policy.as_dict(),
                "verification": {**verification, "verifier_output": answer},
            }
        )
        if owner_id is not None:
            record["owner_id"] = owner_id
        self._replace_workflow_run(record)
        self._run_order.appendleft(record["workflow_run_id"])
        self._append_audit_event(
            "workflow_run_created",
            {"workflow_run_id": record["workflow_run_id"], "mode": "route", "agent_count": 1},
        )
        self.record_analytics_event(
            "workflow_run_created",
            {"workflow_run_id": record["workflow_run_id"], "run_mode": "route", "policy_mode": "route",
             "trace_step_count": 1, "trace_complete": self._is_trace_complete(record)},
        )

    def _cache_key(
        self,
        messages: list[ChatMessage],
        mode: str,
        model_name: str = "contextual-orchestrator",
        cache_partition: str | None = None,
    ) -> str:
        snapshot = getattr(self.client, "request_settings_snapshot", None)
        parameters = snapshot() if callable(snapshot) else {
            "temperature": getattr(self.client, "default_temperature", None),
            "top_p": getattr(self.client, "default_top_p", None),
            "presence_penalty": getattr(self.client, "default_presence_penalty", None),
            "frequency_penalty": getattr(self.client, "default_frequency_penalty", None),
            "max_output_tokens": getattr(self.client, "max_output_tokens", None),
        }
        return build_response_cache_key(
            messages,
            mode,
            model=model_name,
            parameters=parameters,
            partition=cache_partition,
        )

    def run(
        self,
        messages: list[ChatMessage],
        mode: str = "auto",
        workflow_run_id: str | None = None,
        *,
        bypass_cache: bool = False,
        model_name: str = "contextual-orchestrator",
        cache_partition: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute completion and persist a workflow run with trace and policy evidence."""
        if self.budget_max_output_tokens is not None or self.budget_max_cost_usd is not None:
            budget = self.budget_status()
            if budget["exceeded"]:
                raise BudgetExceededError("spend budget exceeded", detail=budget)
        result = self.complete(
            messages,
            mode=mode,
            bypass_cache=bypass_cache,
            model_name=model_name,
            cache_partition=cache_partition,
        )
        prompt = self._latest_user_text(messages)
        record = self._with_effort_snapshot(
            {
                "workflow_run_id": workflow_run_id or f"run_{uuid.uuid4().hex}",
                "created_at": int(time.time()),
                "mode": result["mode"],
                "policy_mode": mode,
                "prompt_text": prompt,
                "answer": result["answer"],
                "cache_status": result.get("cache_status", "disabled"),
                "trace": result["trace"],
                "policy_snapshot": self.policy.as_dict(),
                "verification": result.get("verification"),
            }
        )
        if owner_id is not None:
            record["owner_id"] = owner_id
        self._replace_workflow_run(record)
        self._run_order.appendleft(record["workflow_run_id"])
        if self._store is not None:
            self._store.save("workflow_run", record["workflow_run_id"], record)
        self._append_audit_event(
            "workflow_run_created",
            {
                "workflow_run_id": record["workflow_run_id"],
                "mode": record["mode"],
                "agent_count": len(record["trace"]),
            },
        )
        self.record_analytics_event(
            "workflow_run_created",
            {
                "workflow_run_id": record["workflow_run_id"],
                "run_mode": record["mode"],
                "policy_mode": record["policy_mode"],
                "trace_step_count": len(record["trace"]),
                "trace_complete": self._is_trace_complete(record),
            },
        )
        for step in record["trace"]:
            self.record_analytics_event(
                "workflow_step_completed",
                {
                    "workflow_run_id": record["workflow_run_id"],
                    "run_mode": record["mode"],
                    "step_id": step["id"],
                    "agent_id": step["agent_id"],
                    "role": step["role"],
                    "duration_ms": step.get("latency_ms"),
                },
            )
        return record

    def _raise_if_spend_budget_exceeded(
        self,
        *,
        additional_output_tokens: int = 0,
        additional_cost_usd: float = 0.0,
    ) -> None:
        """Fail before another provider call would cross an operator budget."""
        with self._budget_spend_lock:
            spent_output_tokens = self._budget_spent_output_tokens
            spent_cost_decimal = self._budget_spent_cost_usd
            budget = self._budget_block(
                spent_output_tokens,
                float(spent_cost_decimal) if self.price_per_million else None,
            )
            spent_tokens = spent_output_tokens + additional_output_tokens
            spent_cost = budget["spent_cost_usd"]
            effective_cost = (
                spent_cost + additional_cost_usd if spent_cost is not None else None
            )
            if budget["exceeded"] or (
                budget["max_output_tokens"] is not None
                and spent_tokens >= budget["max_output_tokens"]
            ) or (
                budget["max_cost_usd"] is not None
                and effective_cost is not None
                and effective_cost >= budget["max_cost_usd"]
            ):
                raise BudgetExceededError("spend budget exceeded", detail=budget)

    def _trace_budget_spend(self, trace: list[dict[str, Any]]) -> tuple[int, float]:
        """Return completed provider-call spend for a workflow budget checkpoint."""
        model_by_agent = {agent.id: agent.model for agent in self.agents}
        output_tokens = sum(_step_output_token_count(step) for step in trace)
        output_cost = sum(
            _step_output_token_count(step) / 1_000_000 * self.price_per_million[model]
            for step in trace
            if (
                model := model_by_agent.get(
                    step.get("served_agent_id") or step.get("agent_id")
                )
            )
            in self.price_per_million
        )
        return output_tokens, round(output_cost, 6)

    def batch_route(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Route many prompts through the provider's Batch API and persist each run.

        The cheap lane for bulk/eval workloads (~50% provider discount, async window) —
        not for latency-sensitive chat. Each prompt gets the same worker selection as
        ``route_once``; results are persisted as normal route runs (with provider usage
        when reported) so spend analytics and the admin console see them unchanged.
        """
        if self.budget_max_output_tokens is not None or self.budget_max_cost_usd is not None:
            budget = self.budget_status()
            if budget["exceeded"]:
                raise BudgetExceededError("spend budget exceeded", detail=budget)
        selected = [(prompt, self._select_agent(prompt, "worker")) for prompt in prompts]
        agents_by_id = {agent.id: agent for _, agent in selected}
        requests_by_agent: dict[str, dict[str, list[ChatMessage]]] = {}
        for index, (prompt, agent) in enumerate(selected):
            requests_by_agent.setdefault(agent.id, {})[f"task_{index}"] = [{"role": "user", "content": prompt}]

        answers: dict[int, dict[str, Any]] = {}
        for agent_id, requests in requests_by_agent.items():
            effort_profile = self._role_effort_profile("worker")
            batch = (
                self.client.batch_chat(
                    agents_by_id[agent_id], requests, effort_profile=effort_profile
                )
                if effort_profile is not None
                else self.client.batch_chat(agents_by_id[agent_id], requests)
            )
            results = _validate_batch_results(requests, batch)
            for custom_id, result in results.items():
                # _validate_batch_results already pinned every result key to the
                # canonical requested task_{index} identifiers, so hostile or
                # duplicate identifiers cannot reach this loop. These guards only
                # document that contract and are unreachable today.
                prefix, suffix = custom_id.rsplit("_", 1)  # pragma: no cover - contract pinned above
                index = int(suffix)  # pragma: no cover
                if (  # pragma: no cover - contract pinned above
                    prefix != "task"
                    or custom_id != f"task_{index}"
                    or not 0 <= index < len(selected)
                ):
                    raise RuntimeError("batch provider returned an invalid request identifier")
                if index in answers:  # pragma: no cover - results keys are unique
                    raise RuntimeError("batch provider returned a duplicate request identifier")
                answers[index] = result

        records: list[dict[str, Any]] = []
        for index, (prompt, agent) in enumerate(selected):
            result = answers[index]
            row: dict[str, Any] = {
                "id": 0, "role": "worker", "agent_id": agent.id,
                "subtask": "Direct route (batched)", "access": [], "output": result["content"],
            }
            if result.get("usage") is not None:
                row["usage"] = result["usage"]
            record = self._with_effort_snapshot(
                {
                    "workflow_run_id": f"run_{uuid.uuid4().hex}",
                    "created_at": int(time.time()),
                    "mode": "route",
                    "policy_mode": "route",
                    "prompt_text": prompt,
                    "answer": result["content"],
                    "trace": [row],
                    "policy_snapshot": self.policy.as_dict(),
                    "verification": {"accepted": True, "reason": "single route path (batched)", "verifier_output": ""},
                }
            )
            self._replace_workflow_run(record)
            self._run_order.appendleft(record["workflow_run_id"])
            if self._store is not None:
                self._store.save("workflow_run", record["workflow_run_id"], record)
            self._append_audit_event(
                "workflow_run_created",
                {"workflow_run_id": record["workflow_run_id"], "mode": "route", "agent_count": 1},
            )
            self.record_analytics_event(
                "workflow_run_created",
                {"workflow_run_id": record["workflow_run_id"], "run_mode": "route", "policy_mode": "route",
                 "trace_step_count": 1, "trace_complete": self._is_trace_complete(record)},
            )
            records.append(record)
        return records

    def run_evaluation(
        self, prompts: list[str], mode: str = "auto", owner_id: str | None = None
    ) -> dict[str, Any]:
        """Replay prompts through the runtime and persist an evaluation record."""
        if not prompts:  # pragma: no cover
            raise ValueError("evaluation requires at least one prompt")
        workflow_run_ids: list[str] = []
        results: list[dict[str, Any]] = []
        for prompt in prompts:
            record = self.run(
                [{"role": "user", "content": prompt}], mode=mode, owner_id=owner_id
            )
            workflow_run_ids.append(record["workflow_run_id"])
            results.append({
                "workflow_run_id": record["workflow_run_id"],
                "answer": record["answer"],
            })

        evaluation_run_id = f"eval_{uuid.uuid4().hex}"
        evaluation = {
            "evaluation_run_id": evaluation_run_id,
            "created_at": int(time.time()),
            "mode": mode,
            "prompt_count": len(prompts),
            "workflow_run_ids": workflow_run_ids,
            "results": results,
            "success_count": len([r for r in results if r["answer"]]),
        }
        if owner_id is not None:
            evaluation["owner_id"] = owner_id
        self._evaluation_runs[evaluation_run_id] = evaluation
        if self._store is not None:
            self._store.save("evaluation_run", evaluation_run_id, evaluation)
        self._append_audit_event(
            "evaluation_run_created",
            {
                "evaluation_run_id": evaluation_run_id,
                "workflow_run_count": len(workflow_run_ids),
                "success_count": evaluation["success_count"],
            },
        )
        self.record_analytics_event(
            "evaluation_run_created",
            {
                "evaluation_run_id": evaluation_run_id,
                "run_mode": mode,
                "workflow_run_count": len(workflow_run_ids),
                "success_count": evaluation["success_count"],
            },
        )
        return evaluation

    def compare_to_baseline(self, prompts: list[str], mode: str = "auto") -> dict[str, Any]:
        """Measure the orchestration engine against a single-worker baseline.

        For each prompt: run the full orchestration (route/conduct per mode) and a
        single-agent baseline (one worker call, no verifier/synthesizer), then report
        latency and a structural coverage proxy plus the delta.

        This is a MEASURED report, not a quality claim: the proxy is structural
        (contributing steps + verifier-pass presence, computable from mock/runtime
        outputs), NOT human-judged answer quality. Read-only — it does not persist runs.
        """
        results: list[dict[str, Any]] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            start = time.perf_counter()
            # Evaluation must measure provider work, not a cache hit from a prior request.
            # The normal completion path still honors the configured response cache.
            orchestrated = self._dispatch(messages, mode)
            orchestrated_latency = round((time.perf_counter() - start) * 1000, 2)

            start = time.perf_counter()
            baseline = self.route_once(messages)
            baseline_latency = round((time.perf_counter() - start) * 1000, 2)

            orchestrated_steps = len(orchestrated["trace"])
            baseline_steps = len(baseline["trace"])
            results.append({
                "prompt": prompt[:120],
                "orchestrated": {
                    "mode": orchestrated["mode"],
                    "latency_ms": orchestrated_latency,
                    "steps": orchestrated_steps,
                    "verified": bool(orchestrated.get("verification", {}).get("accepted")),
                    "answer_length": len(orchestrated["answer"]),
                },
                "baseline": {
                    "mode": baseline["mode"],
                    "latency_ms": baseline_latency,
                    "steps": baseline_steps,
                    "answer_length": len(baseline["answer"]),
                },
                "latency_overhead_ms": round(orchestrated_latency - baseline_latency, 2),
                "structural_coverage_delta": orchestrated_steps - baseline_steps,
            })

        count = len(results)

        def avg(select: Any) -> float:
            return round(sum(select(row) for row in results) / count, 2) if count else 0.0

        aggregate = {
            "orchestrated_avg_latency_ms": avg(lambda row: row["orchestrated"]["latency_ms"]),
            "baseline_avg_latency_ms": avg(lambda row: row["baseline"]["latency_ms"]),
            "avg_latency_overhead_ms": avg(lambda row: row["latency_overhead_ms"]),
            "orchestrated_avg_steps": avg(lambda row: row["orchestrated"]["steps"]),
            "baseline_avg_steps": avg(lambda row: row["baseline"]["steps"]),
            "avg_structural_coverage_delta": avg(lambda row: row["structural_coverage_delta"]),
            "verified_share": round(sum(1 for row in results if row["orchestrated"]["verified"]) / count, 2) if count else 0.0,
        }
        return {
            "mode": mode,
            "prompt_count": count,
            "results": results,
            "aggregate": aggregate,
            "quality_proxy": (
                "structural proxy from mock/runtime outputs (contributing steps + verifier-pass presence); "
                "measures the latency-for-verification tradeoff, NOT human-judged quality"
            ),
        }

    def get_workflow_run(
        self, workflow_run_id: str, owner_id: str | None = None
    ) -> dict[str, Any]:
        """Return a persisted workflow run by identifier."""
        if workflow_run_id not in self._workflow_runs:  # pragma: no cover
            raise KeyError(workflow_run_id)
        record = self._workflow_runs[workflow_run_id]
        if owner_id is not None and record.get("owner_id") != owner_id:
            raise KeyError(workflow_run_id)
        return record

    def get_evaluation_run(
        self, evaluation_run_id: str, owner_id: str | None = None
    ) -> dict[str, Any]:
        """Return an evaluation only when it belongs to the requested owner."""
        record = self._evaluation_runs[evaluation_run_id]
        if owner_id is not None and record.get("owner_id") != owner_id:
            raise KeyError(evaluation_run_id)
        return record

    def get_access_report(
        self, workflow_run_id: str, owner_id: str | None = None
    ) -> dict[str, Any]:
        """Return per-step visibility and accessed output evidence for a run."""
        run = self.get_workflow_run(workflow_run_id, owner_id=owner_id)
        access_report = []
        for step in run["trace"]:
            access_report.append({
                "step_id": step["id"],
                "role": step["role"],
                "agent_id": step["agent_id"],
                "access": step["access"],
                "accessed_outputs": [
                    run["trace"][index]["output"] for index in step["access"] if index < len(run["trace"])
                ],
            })
        return {
            "workflow_run_id": workflow_run_id,
            "policy_snapshot": run["policy_snapshot"],
            "steps": access_report,
            "verifier": run.get("verification"),
        }

    def patch_agent(self, agent_pool_id: str, worker_agent_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply governance updates to an agent and emit an audit event."""
        if not patch:  # pragma: no cover
            raise ValueError("patch request body must contain updates")
        current = self._agent_in_pool(agent_pool_id, worker_agent_id)
        patched = current
        if "status" in patch:
            status = str(patch["status"]).lower()
            if status in {"active", "enabled"}:
                patched = replace(patched, disabled=False)
            elif status in {"disabled", "excluded", "inactive", "quarantine"}:
                patched = replace(patched, disabled=True)
            else:  # pragma: no cover
                raise ValueError("status must be active, enabled, disabled, excluded, inactive, or quarantine")
        if "priority" in patch:
            patched = replace(patched, priority=int(patch["priority"]))
        if "tags" in patch:
            patched = replace(patched, tags=tuple(patch["tags"]))
        if "provider_exclusions" in patch:
            patched = replace(patched, provider_exclusions=tuple(patch["provider_exclusions"]))
        if "group_name" in patch:
            group_name = str(patch["group_name"])
            patched = replace(patched, group_name=canonical_group_name(group_name) if group_name else "")
        if "endpoint_equivalence" in patch:
            value = patch["endpoint_equivalence"]
            if value is not None and not isinstance(value, dict):
                raise ValueError("endpoint_equivalence must be an object or null")
            patched = replace(patched, endpoint_equivalence=value)

        updated_candidates = [patched if agent.id == worker_agent_id else agent for agent in self.candidates]
        updated_agents = [agent for agent in updated_candidates if not agent.disabled]
        if not updated_agents:
            raise ValueError("cannot disable the last enabled agent")
        self.candidates = updated_candidates
        self.agents = updated_agents
        self._rebuild_budget_meter()
        if patched.group_name != current.group_name:
            self._routers_reset_members({worker_agent_id})
        candidate_ids = {agent.id for agent in updated_candidates}
        for agent_id in candidate_ids:
            self._routers_register_member(agent_id)
        self._routers_forget_members(candidate_ids)
        if self._pool_store is not None:
            self._pool_store.save(patched)
        self._append_audit_event(
            "agent_patched",
            {
                "agent_pool_id": agent_pool_id,
                "worker_agent_id": worker_agent_id,
                "updated_fields": sorted(patch.keys()),
            },
        )
        if "status" in patch:
            self.record_analytics_event(
                "agent_status_changed",
                {
                    "agent_pool_id": agent_pool_id,
                    "agent_id": worker_agent_id,
                    "status": self._agent_to_admin_payload(patched)["status"],
                },
            )
        if "provider_exclusions" in patch:
            self.record_analytics_event(
                "provider_exclusion_changed",
                {
                    "agent_pool_id": agent_pool_id,
                    "agent_id": worker_agent_id,
                    "provider_exclusions": list(patched.provider_exclusions),
                },
            )
        return self._agent_to_admin_payload(patched)

    def list_model_groups(self) -> list[dict[str, Any]]:
        """Return operator-defined logical models and measured member evidence."""
        names = sorted({canonical_group_name(agent.group_name) for agent in self.candidates if agent.group_name})
        return [self.get_model_group(name) for name in names]

    def get_model_group(self, group_name: str) -> dict[str, Any]:
        """Return one logical model group or raise ``KeyError`` when absent."""
        name = canonical_group_name(group_name)
        members = [agent for agent in self.candidates if agent.group_name and canonical_group_name(agent.group_name) == name]
        if not members:
            raise KeyError(name)
        ranked_ids = self._measured_member_order([agent.id for agent in members])
        return {
            "group_name": name,
            "member_agent_ids": ranked_ids,
            "enabled_member_count": sum(1 for agent in members if not agent.disabled),
            "capability_coverage": {
                capability: sum(capability in agent.tags for agent in members)
                for capability in sorted(MODEL_CAPABILITIES)
                if any(capability in agent.tags for agent in members)
            },
            "members": [self._agent_to_admin_payload(self._agent(agent_id)) for agent_id in ranked_ids],
        }

    def set_model_group(self, group_name: str, member_agent_ids: list[str]) -> dict[str, Any]:
        """Create or replace a group membership using configured agent identifiers."""
        name = canonical_group_name(group_name)
        if not member_agent_ids or any(type(agent_id) is not str for agent_id in member_agent_ids):
            raise ValueError("member_agent_ids must be a non-empty list of strings")
        if len(member_agent_ids) != len(set(member_agent_ids)):
            raise ValueError("member_agent_ids must not contain duplicates")
        requested = set(member_agent_ids)
        known = {agent.id for agent in self.candidates}
        previous = {
            agent.id
            for agent in self.candidates
            if agent.group_name and canonical_group_name(agent.group_name) == name
        }
        missing = sorted(requested - known)
        if missing:
            raise KeyError(",".join(missing))
        previous_candidates = self.candidates
        updated = [
            replace(agent, group_name=name)
            if agent.id in requested
            else replace(agent, group_name="")
            if agent.group_name and canonical_group_name(agent.group_name) == name
            else agent
            for agent in self.candidates
        ]
        self.candidates = updated
        self.agents = [agent for agent in updated if not agent.disabled]
        changed = {
            before.id
            for before, after in zip(previous_candidates, updated)
            if before.group_name != after.group_name
        }
        self._routers_reset_members(changed)
        for agent_id in changed:
            self._routers_register_member(agent_id)
        for agent in updated:
            if agent.id in requested:
                if self._pool_store is not None:
                    self._pool_store.save(agent)
            elif agent.id in previous and self._pool_store is not None:
                self._pool_store.save(agent)
        self._routers_forget_members({agent.id for agent in updated})
        self._append_audit_event("model_group_set", {"group_name": name, "member_agent_ids": sorted(requested)})
        return self.get_model_group(name)

    def delete_model_group(self, group_name: str) -> dict[str, Any]:
        """Delete a logical group while retaining its provider agents."""
        current = self.get_model_group(group_name)
        name = current["group_name"]
        member_ids = set(current["member_agent_ids"])
        self._routers_reset_members(member_ids)
        for agent_id in member_ids:
            self._routers_register_member(agent_id)
        self.candidates = [replace(agent, group_name="") if agent.id in member_ids else agent for agent in self.candidates]
        self.agents = [agent for agent in self.candidates if not agent.disabled]
        if self._pool_store is not None:
            for agent in self.candidates:
                if agent.id in member_ids:
                    self._pool_store.save(agent)
        self._routers_forget_members({agent.id for agent in self.candidates})
        self._append_audit_event("model_group_deleted", {"group_name": name})
        return {"group_name": name, "deleted": True}

    def add_agent(self, agent_pool_id: str, value: dict[str, Any]) -> dict[str, Any]:
        """Register a new worker agent (model group member) at runtime; persists when agents_db is set."""
        if agent_pool_id != "default":  # pragma: no cover
            raise KeyError(agent_pool_id)
        if "id" not in value or "model" not in value:
            raise ValueError("agent requires id and model")
        agent = ModelAgent.from_dict(value)
        if any(existing.id == agent.id for existing in self.candidates):
            raise ValueError(f"agent {agent.id} already exists")
        if not agent.base_url.startswith("mock://"):
            parsed = urlparse(agent.base_url)
            if not _is_local_provider_url(agent.base_url) and (parsed.scheme != "https" or not parsed.hostname):
                raise ValueError("non-mock remote agents must use an https base_url; local agents use mlx://loopback")
            if not _is_local_provider_url(agent.base_url) and not agent.credential_name:
                raise ValueError("non-mock agents require credential_key or legacy api_key_env")
        self.candidates = [*self.candidates, agent]
        self.agents = [candidate for candidate in self.candidates if not candidate.disabled]
        self._rebuild_budget_meter()
        self._routers_register_member(agent.id)
        if self._pool_store is not None:
            self._pool_store.save(agent)
        self._append_audit_event(
            "agent_added",
            {"agent_pool_id": agent_pool_id, "worker_agent_id": agent.id, "model": agent.model},
        )
        self.record_analytics_event(
            "agent_added",
            {"agent_pool_id": agent_pool_id, "agent_id": agent.id, "model": agent.model},
        )
        return self._agent_to_admin_payload(agent)

    def sync_discovered_agents(self, discovered_agents: list[ModelAgent]) -> dict[str, list[str]]:
        """Upsert auto-discovered agents into the pool; persists when agents_db is set.

        Unlike :meth:`add_agent`, an id that already exists is replaced in place
        (re-running discovery is idempotent) instead of raising. New agents are
        appended disabled (see ``model_discovery.agent_from_discovered``) so a
        freshly discovered model never starts serving traffic before an operator
        (or the cost router) opts it in via ``patch_agent``.
        """
        existing_by_id = {agent.id: index for index, agent in enumerate(self.candidates)}
        updated_candidates = list(self.candidates)
        added: list[str] = []
        updated: list[str] = []
        for agent in discovered_agents:
            index = existing_by_id.get(agent.id)
            if index is None:
                existing_by_id[agent.id] = len(updated_candidates)
                updated_candidates.append(agent)
                added.append(agent.id)
            else:
                agent = replace(
                    agent,
                    group_name=updated_candidates[index].group_name,
                )
                updated_candidates[index] = agent
                updated.append(agent.id)
            if self._pool_store is not None:
                self._pool_store.save(agent)
        self.candidates = updated_candidates
        self.agents = [candidate for candidate in self.candidates if not candidate.disabled]
        self._rebuild_budget_meter()
        for agent in discovered_agents:
            self._routers_register_member(agent.id)
        if added or updated:
            self._append_audit_event(
                "agents_discovered",
                {"added": added, "updated": updated},
            )
            self.record_analytics_event(
                "agents_discovered",
                {"added_count": len(added), "updated_count": len(updated)},
            )
        return {"added": added, "updated": updated}

    def remove_agent(self, agent_pool_id: str, worker_agent_id: str) -> dict[str, Any]:
        """Remove a worker agent from the pool; the pool must keep at least one enabled agent."""
        target = self._agent_in_pool(agent_pool_id, worker_agent_id)
        remaining_enabled = [agent for agent in self.candidates if agent.id != worker_agent_id and not agent.disabled]
        if not remaining_enabled:
            raise ValueError("cannot remove the last enabled agent")
        self.candidates = [agent for agent in self.candidates if agent.id != worker_agent_id]
        self.agents = [agent for agent in self.candidates if not agent.disabled]
        self._rebuild_budget_meter()
        self._routers_forget_members({agent.id for agent in self.candidates})
        if self._pool_store is not None:
            # Disabled tombstone (not a row delete): it overlays the seed file on restart
            # and startup drops disabled agents, so removal survives even for seed agents.
            self._pool_store.save(replace(target, disabled=True, group_name=""))
        self._append_audit_event(
            "agent_removed",
            {"agent_pool_id": agent_pool_id, "worker_agent_id": worker_agent_id, "model": target.model},
        )
        self.record_analytics_event(
            "agent_removed",
            {"agent_pool_id": agent_pool_id, "agent_id": worker_agent_id},
        )
        return {"removed": worker_agent_id}

    def route_once(
        self,
        messages: list[ChatMessage],
        *,
        model_name: str = "contextual-orchestrator",
    ) -> dict[str, Any]:
        """Route a prompt to one selected worker agent and return a single-step trace.

        When ``policy.realtime_judge`` is on, every candidate answer is judged
        in real time by the fast-mlsirm judge before it is returned; rejected
        answers fail over to the next measured candidate within the configured
        tool-retry budget, and every verdict updates the quality ledger so
        measured accuracy steers future routing. Speed is explicitly not a
        design constraint at this layer -- correctness is.
        """
        text = self._latest_user_text(messages)
        free_only = model_name == self.FREE_MODEL
        requested = self._requested_agent(model_name)
        ranked_pool: list[ModelAgent] = (
            [requested] if requested is not None else []
        ) or self._ranked_agents(text, "worker", free_only=free_only)
        free_ids = {candidate.id for candidate in self.agents if self._is_free_agent(candidate)}
        allowed_agent_ids = free_ids if free_only else None

        max_attempts = 1 + min(self.tool_retry_attempts, MAX_TOOL_RETRY_ATTEMPTS)
        trace_rows: list[dict[str, Any]] = []
        answer = ""
        served_id = ""
        usage: dict[str, Any] | None = None
        verification: dict[str, Any] = {
            "accepted": False,
            "reason": "no candidate attempted",
            "verifier_output": "",
            "judge": "model",
        }
        tried_ids: set[str] = set()
        for attempt_index, candidate in enumerate(ranked_pool):
            if len(tried_ids) >= max_attempts:
                break
            tried_ids.add(candidate.id)
            start = time.perf_counter()
            attempt_answer, attempt_served_id, attempt_usage = self._invoke(
                candidate,
                messages,
                text=text,
                role="worker",
                allowed_agent_ids=allowed_agent_ids,
            )
            latency_seconds = time.perf_counter() - start
            row = {
                "id": attempt_index,
                "role": "worker",
                "agent_id": candidate.id,
                "subtask": "Direct route",
                "access": [],
                "latency_ms": round(latency_seconds * 1000, 2),
                "output": attempt_answer,
            }
            if attempt_usage is not None:
                row["usage"] = attempt_usage
            if attempt_served_id != candidate.id:
                row["served_agent_id"] = attempt_served_id
                row["failover_from"] = candidate.id
            answer, served_id, usage = attempt_answer, attempt_served_id, attempt_usage
            verification = self._realtime_route_judge(
                text=text,
                answer=answer,
                served_id=served_id,
                latency_seconds=latency_seconds,
                usage=attempt_usage,
                free_only=free_only,
            )
            row["realtime_judge"] = {
                "accepted": verification["accepted"],
                "reason": verification["reason"],
            }
            trace_rows.append(row)
            if verification["accepted"]:
                break
            # Rejected answers already recorded a quality-ledger failure in
            # _realtime_route_judge; keep the last (best-available) answer but
            # fail over to the next measured candidate while budget remains.

        final_row = trace_rows[-1] if trace_rows else {
            "id": 0,
            "role": "worker",
            "agent_id": "",
            "subtask": "Direct route",
            "access": [],
            "latency_ms": None,
            "output": "",
        }
        return self._with_effort_snapshot(
            {
                "mode": "route",
                "answer": answer,
                "verification": {**verification, "verifier_output": answer},
                "trace": [final_row],
            }
        )

    def _realtime_route_judge(
        self,
        *,
        text: str,
        answer: str,
        served_id: str,
        latency_seconds: float,
        usage: dict[str, Any] | None,
        free_only: bool,
    ) -> dict[str, Any]:
        """Judge one direct-route answer now and feed the quality ledger.

        Accepted answers record one success observation (with provider token
        counts when reported); rejected or unjudgeable answers record one
        failure, so measured accuracy -- not just transport success -- steers
        subsequent member ordering inside model groups.
        """
        output_tokens = self._usage_completion_tokens(usage)

        def _record(accepted: bool) -> None:
            if accepted:
                self._quality_router.observe_success(
                    served_id, latency_seconds, output_tokens=output_tokens
                )
            else:
                self._quality_router.observe_failure(served_id)

        if not self.policy.realtime_judge:
            return {
                "accepted": True,
                "reason": "single route path",
                "verifier_output": answer,
                "judge": "model",
            }
        fallback_report = {"verifier_output": answer}
        base = self._model_judge_verification(
            text, fallback_report, free_only=free_only
        )
        accepted = bool(base.get("accepted"))
        _record(accepted)
        return base

    @staticmethod
    def _usage_completion_tokens(usage: dict[str, Any] | None) -> int | None:
        """Provider-reported completion token count, or None when absent/invalid."""
        if not isinstance(usage, dict):
            return None
        tokens = usage.get("completion_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            return None
        return tokens

    def conduct(
        self,
        messages: list[ChatMessage],
        *,
        model_name: str = "contextual-orchestrator",
        progress: Any = None,
    ) -> dict[str, Any]:
        """Run a workflow, optionally reporting safe stage summaries (never hidden reasoning)."""
        self._raise_if_spend_budget_exceeded()
        task = self._latest_user_text(messages)
        source_images = self._source_image_parts(messages)
        required_tags = ("vision",) if source_images else ()
        caller_instructions = "\n\n".join(
            message["content"]
            for message in messages
            if message.get("role") == "system" and isinstance(message.get("content"), str)
        )
        plan_source = "template"
        if model_name not in {"contextual-orchestrator", self.AUTO_MODEL}:
            steps = self._plan(task, model_name=model_name)
        elif self.policy.workflow_planning == "generated":
            try:
                steps = self._plan_generated(task)
                plan_source = "generated"
            except BudgetExceededError:
                raise
            except Exception:  # noqa: BLE001 - invalid plans must not break the request
                steps = self._plan(task)
                plan_source = "template_fallback"
        else:
            steps = self._plan(task)
        outputs: dict[int, str] = {}
        trace: list[dict[str, Any]] = []
        free_ids = {candidate.id for candidate in self.agents if self._is_free_agent(candidate)}
        requested_agent = self._requested_agent(model_name)
        judge_agent_ids = (
            {
                candidate.id
                for candidate in self.agents
                if candidate.group_name == requested_agent.group_name
            }
            if requested_agent is not None and requested_agent.group_name
            else {requested_agent.id}
            if requested_agent is not None
            else free_ids
            if model_name == self.FREE_MODEL
            else None
        )

        for step in steps:
            if plan_source == "generated":
                in_flight_tokens, in_flight_cost = self._trace_budget_spend(trace)
                self._raise_if_spend_budget_exceeded(
                    additional_output_tokens=in_flight_tokens,
                    additional_cost_usd=in_flight_cost,
                )
            agent = self._agent(step.agent_id)
            if any(tag not in agent.tags for tag in required_tags):
                try:
                    capable = self._ranked_agents(
                        step.subtask,
                        step.role,
                        required_tags=required_tags,
                        free_only=model_name == self.FREE_MODEL,
                    )
                except RuntimeError:
                    capable = []
                if capable:
                    agent = capable[0]
            if progress is not None:
                progress(step.role, "started")
            prior = "\n\n".join(f"Step {i}: {outputs[i]}" for i in step.access)
            instruction = (
                f"Original task:\n{task}\n\nAccessed prior work:\n{prior}\n\n"
                f"Subtask:\n{step.subtask}"
            )
            user_content: str | list[dict[str, Any]] = instruction
            if source_images:
                user_content = [
                    {"type": "text", "text": instruction},
                    *copy.deepcopy(source_images),
                ]
            step_messages = [
                {
                    "role": "system",
                    "content": (
                        f"Role: {step.role}\n"
                        "Use only the original task and the accessed prior steps. "
                        "Return concise, directly useful work."
                        + (f"\n\nCaller instructions:\n{caller_instructions}" if caller_instructions else "")
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            start = time.perf_counter()
            output, served_id, usage = self._invoke(
                agent,
                step_messages,
                text=task,
                role=step.role,
                allowed_agent_ids=free_ids if model_name == self.FREE_MODEL else None,
            )
            elapsed = (time.perf_counter() - start) * 1000
            outputs[step.id] = output
            row = step.as_dict()
            row["agent_id"] = agent.id
            row["latency_ms"] = round(elapsed, 2)
            row["output"] = output
            if usage is not None:
                row["usage"] = usage
            if served_id != agent.id:  # pragma: no cover
                row["served_agent_id"] = served_id
                row["failover_from"] = agent.id
            trace.append(row)
            if progress is not None:
                progress(step.role, "completed")

        if plan_source == "generated":
            # Generated plans have variable shape: locate roles instead of fixed indices.
            def last_output(role: str) -> str:
                ids = [step.id for step in steps if step.role == role]
                return outputs.get(ids[-1], "") if ids else ""

            # Generated plans may omit a thinker; the first step's output is the upstream evidence.
            upstream = last_output("thinker") or outputs.get(steps[0].id, "")
            verification = self._judge_verifier_output(last_output("verifier"), upstream, last_output("worker"))
            if self.policy.verifier_judge == "model":  # pragma: no branch - OrchestrationPolicy validates this to be constant
                verification = self._model_judge_verification(
                    task,
                    verification,
                    free_only=model_name == self.FREE_MODEL,
                    allowed_agent_ids=judge_agent_ids,
                )
            answer = outputs[steps[-1].id]
            if not verification["accepted"] and self.policy.verifier_required and last_output("worker"):
                answer = last_output("worker")
        else:
            verification = self._judge_verifier_output(outputs.get(2, ""), outputs.get(0, ""), outputs.get(1, ""))
            if self.policy.verifier_judge == "model":  # pragma: no branch - OrchestrationPolicy validates this to be constant
                verification = self._model_judge_verification(
                    task,
                    verification,
                    free_only=model_name == self.FREE_MODEL,
                    allowed_agent_ids=judge_agent_ids,
                )
            answer = outputs[steps[2].id] if not self.policy.verifier_required else outputs[steps[-1].id]
            if not verification["accepted"] and self.policy.verifier_required:
                answer = outputs[steps[1].id]

        return self._with_effort_snapshot(
            {
                "mode": "conduct",
                "answer": answer,
                "trace": trace,
                "verification": verification,
                "plan_source": plan_source,
            }
        )

    def _role_effort_profile(self, role: str) -> ReasoningEffortProfile | None:
        """Return the opt-in profile bound to one workflow role."""
        if self.role_effort_catalog is None:
            return None
        return self.role_effort_catalog.get(role)

    def _with_effort_snapshot(self, result: dict[str, Any]) -> dict[str, Any]:
        """Attach a replayable role-effort snapshot when the operator opted in.

        Buyer next action: compare ``reasoning_effort_snapshot.snapshot_hash``
        on ``complete``, ``run``, ``stream_route``, and ``batch_route``. Omit
        the constructor catalog to keep today's payload.
        """
        if self.role_effort_catalog is None:
            return result
        snapshot = snapshot_role_effort_catalog(self.role_effort_catalog)
        result["reasoning_effort_snapshot"] = {
            "profile_version": snapshot.profile_version,
            "snapshot_hash": snapshot.snapshot_hash,
            "role_profiles": snapshot.role_profiles,
        }
        return result

    def _plan_generated(self, task: str) -> list[WorkflowStep]:
        """Ask the planner model to generate the workflow (Conductor, arXiv:2512.04388).

        The plan is JSON: natural-language subtasks, a worker assignment, and an access
        list of prior step outputs per step. Anything invalid raises so conduct() falls
        back to the fixed template — a bad plan must never break the request.
        """
        planner = self._select_agent(task, "thinker")
        pool = "\n".join(
            f"- {agent.id}: model={agent.model}, tags={', '.join(agent.tags) or 'none'}"
            for agent in self.agents
            if is_general_chat_agent_model_id(agent.model)
        )
        system = (
            "You are the workflow conductor. Decompose the user's task into a short workflow.\n"
            'Return ONLY a JSON object, no prose: {"steps": [{"id": 0, "role": "thinker|worker|verifier|synthesizer", '
            '"agent_id": "<agent id>", "subtask": "natural-language instruction", "access": [prior step ids]}]}\n'
            f"Rules: 2 to {self.policy.max_workflow_steps} steps; ids sequential from 0; access may list only earlier "
            "step ids (each step sees ONLY the outputs it lists); the final step must produce the answer; include a "
            "verifier step when correctness matters.\n"
            f"Available agents:\n{pool}"
        )
        planner_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        effort_profile = self._role_effort_profile("planner")
        raw = (
            self.client.chat(planner, planner_messages, effort_profile=effort_profile)
            if effort_profile is not None
            else self.client.chat(planner, planner_messages)
        )
        return self._parse_workflow_plan(raw)

    def _parse_workflow_plan(self, raw: str) -> list[WorkflowStep]:
        """Validate a generated plan strictly; raise ValueError on any structural problem."""
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("plan contains no JSON object")
        data = json.loads(raw[start : end + 1])
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not (2 <= len(raw_steps) <= self.policy.max_workflow_steps):
            raise ValueError(f"plan must have 2..{self.policy.max_workflow_steps} steps")
        known_agents = {agent.id: agent for agent in self.agents}
        steps: list[WorkflowStep] = []
        for index, item in enumerate(raw_steps):
            if int(item.get("id", -1)) != index:
                raise ValueError("step ids must be sequential from 0")
            role = str(item.get("role", ""))
            if role not in {"thinker", "worker", "verifier", "synthesizer"}:
                raise ValueError(f"unknown role {role!r}")
            subtask = str(item.get("subtask", "")).strip()
            if not subtask:
                raise ValueError("step subtask must be non-empty")
            access = tuple(sorted({int(value) for value in item.get("access", [])}))
            if any(value < 0 or value >= index for value in access):
                raise ValueError("access may reference only earlier steps")
            agent_id = item.get("agent_id")
            assigned = known_agents.get(agent_id)
            if assigned is None or not is_general_chat_agent_model_id(assigned.model):
                # Unknown or stale ineligible assignments are reselected honestly.
                agent_id = self._select_agent(subtask, role).id
            steps.append(WorkflowStep(index, role, agent_id, subtask, access))
        if steps[-1].role not in {"synthesizer", "worker"}:
            raise ValueError("final step must produce the answer")
        return steps

    def _plan(
        self, task: str, *, model_name: str = "contextual-orchestrator"
    ) -> list[WorkflowStep]:
        requested = self._requested_agent(model_name)
        free_only = model_name == self.FREE_MODEL
        thinker = (requested or self._select_agent(task, "thinker", free_only=free_only)).id
        worker = (requested or self._select_agent(task, "worker", free_only=free_only)).id
        verifier = (requested or self._select_agent(task, "verifier", free_only=free_only)).id
        synthesizer = (requested or self._select_agent(task, "synthesizer", free_only=free_only)).id
        return [
            WorkflowStep(0, "thinker", thinker, "Decompose the task and identify the best execution strategy."),
            WorkflowStep(1, "worker", worker, "Execute the core task using the plan.", (0,)),
            WorkflowStep(2, "verifier", verifier, "Find concrete errors, gaps, and unsupported claims.", (0, 1)),
            WorkflowStep(3, "synthesizer", synthesizer, "Produce the final answer, incorporating only verified work.", (0, 1, 2)),
        ]

    def _static_rank_key(
        self,
        agent: ModelAgent,
        role: str,
        affinity: float | None,
    ) -> tuple[int, int, int, float, str]:
        """Operator-declared static ordering key (ascending sort = best first).

        Inputs are exclusively operator configuration and literature-standard
        semantic similarity -- no invented weights:

        1. ``-role_fit``: agents whose operator-maintained capability tags
           include a tag declared for the role lead their tier; a high
           ``priority`` never promotes a capability-mismatched agent over a
           matching one.
        2. ``-priority``: the operator's explicit per-agent ranking.
        3. Semantic bucket + ``-cosine``: cosine similarity between task and
           agent-metadata embeddings (Karpukhin et al., 2020; Ong et al.,
           2024). Agents without an affinity vector sort after all measured
           ones within the same declaration tier.
        4. ``agent.id``: deterministic final tiebreak.
        """
        role_fit = 1 if set(agent.tags) & set(self.ROLE_TAGS.get(role, ())) else 0
        priority = agent.priority
        if isinstance(priority, bool) or not isinstance(priority, (int, float)):
            priority = 0
        has_affinity = 1 if affinity is None else 0
        negated_affinity = 0.0 if affinity is None else -float(affinity)
        return (-role_fit, -int(priority), has_affinity, negated_affinity, agent.id)

    def _ranked_agents(
        self,
        text: str,
        role: str,
        *,
        required_tags: tuple[str, ...] = (),
        free_only: bool = False,
        chat_only: bool = True,
    ) -> list[ModelAgent]:
        """Rank logical model groups, then measured provider members within each group.

        Ordering ladder, all evidence-based:
        1. Role-ineligible members (operator ``provider_exclusions``) always
           follow every eligible one.
        2. Within a partition, operator declarations order statically via
           :meth:`_static_rank_key` (priority -> capability fit -> cosine).
        3. Inside one logical model group, measured ledgers refine member
           order (:meth:`_measured_member_order`: judged quality first, then
           successful responses per second).
        """
        candidates = [
            agent
            for agent in self.agents
            if (not free_only or self._is_free_agent(agent))
            and (not chat_only or is_general_chat_agent_model_id(agent.model))
            and all(tag in agent.tags for tag in required_tags)
        ]
        if not candidates:
            if free_only:
                raise RuntimeError("no enabled zero-cost model is available")
            if chat_only:
                raise RuntimeError("no chat-compatible agent available")
            raise RuntimeError("no enabled zero-cost model is available")
        affinities = self._semantic_affinities(text, candidates)
        static = sorted(
            candidates,
            key=lambda agent: self._static_rank_key(agent, role, affinities.get(agent.id)),
        )
        eligible = [agent for agent in static if role not in agent.provider_exclusions]
        excluded = [agent for agent in static if role in agent.provider_exclusions]
        return self._refine_partition(eligible, role) + self._refine_partition(excluded, role)

    def _refine_partition(self, partition: list[ModelAgent], role: str) -> list[ModelAgent]:
        """Group-refine one role-partition with measured intra-group ordering."""
        groups: dict[str, list[ModelAgent]] = {}
        for agent in partition:
            key = canonical_group_name(agent.group_name) if agent.group_name else f"agent:{agent.id}"
            groups.setdefault(key, []).append(agent)
        ordered: list[ModelAgent] = []
        for members in groups.values():
            if not members[0].group_name:
                ordered.extend(members)
                continue
            eligible = [member for member in members if role not in member.provider_exclusions]
            excluded = [member for member in members if role in member.provider_exclusions]
            for sub_partition in (eligible, excluded):
                by_id = {member.id: member for member in sub_partition}
                ordered.extend(
                    by_id[member_id] for member_id in self._measured_member_order(list(by_id))
                )
        return ordered

    def _measured_member_order(self, member_ids: list[str]) -> list[str]:
        """Order same-declaration members by measured evidence, quality first.

        Evidence ladder: judged-answer observations (real-time fast-mlsirm
        verdicts) govern when any member has them; otherwise the transport
        throughput/stability ledger decides; with no evidence at all the
        caller's input order survives untouched. No synthetic scores.
        """
        if any(
            self._quality_router.member_observation_count(member_id) > 0
            for member_id in member_ids
        ):
            return self._quality_router.ranked_member_ids(member_ids)
        return self._group_router.ranked_member_ids(member_ids)

    # --- dual-ledger membership maintenance ---------------------------------

    def _routing_ledgers(self) -> tuple[ModelGroupRouter, ModelGroupRouter]:
        """Both measured ledgers (transport throughput and judged quality)."""
        return self._group_router, self._quality_router

    def _routers_register_member(self, member_id: str) -> None:
        """Register one member in every routing ledger (idempotent)."""
        for router in self._routing_ledgers():
            router.register_member(member_id)

    def _routers_reset_members(self, member_ids: set[str]) -> None:
        """Drop ledger rows whose group context changed in every ledger."""
        for router in self._routing_ledgers():
            router.reset_members(member_ids)

    def _routers_forget_members(self, member_ids: set[str]) -> None:
        """Forget members that left the pool in every ledger."""
        for router in self._routing_ledgers():
            router.forget_members(member_ids)

    def _is_free_agent(self, agent: ModelAgent) -> bool:
        """Return true only for explicitly zero-priced configured models."""
        if "cost:free" in agent.tags or self.price_per_million.get(agent.id) == 0:
            return True
        return self.price_per_million.get(agent.model) == 0 and sum(
            candidate.model == agent.model for candidate in self.candidates
        ) == 1

    # --- semantic-affinity evidence (cosine similarity; no keyword lists) ---

    @staticmethod
    def _agent_descriptor_text(agent: ModelAgent) -> str:
        """Operator-declared metadata joined as the agent's embedding document."""
        return " ".join([agent.model, *sorted(agent.tags)])

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float], vector_b: list[float]
    ) -> float | None:
        """Cosine of two equal-length vectors; None when either norm is zero."""
        if len(vector_a) != len(vector_b) or not vector_a:
            return None
        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return None
        return dot / (norm_a * norm_b)

    def _cache_put(self, cache: OrderedDict[str, Any], key: str, value: Any) -> None:
        """Insert into one bounded LRU evidence cache under the evidence lock."""
        with self._evidence_lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > self.EVIDENCE_CACHE_MAX_ENTRIES:
                cache.popitem(last=False)

    def _embedding_agent_id(self) -> str | None:
        """First measured embedding-capable member id, or None when unconfigured."""
        try:
            return self.select_capability_agent("embedding").id
        except (RuntimeError, ValueError):
            return None

    def _embed_cached(self, text: str) -> list[float] | None:
        """Embedding vector for text via the configured embedding member; None on failure."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._evidence_lock:
            cached = self._task_vector_cache.get(digest)
        if cached is not None:
            return cached
        embedding_member = self._embedding_agent_id()
        if embedding_member is None:
            return None
        try:
            vectors = self.client.embed(self._agent(embedding_member), [text])
        except Exception:  # noqa: BLE001 - similarity is best-effort evidence
            return None
        vector = vectors[0] if vectors else None
        if vector is not None:
            self._cache_put(self._task_vector_cache, digest, vector)
        return vector

    def _descriptor_vector_cached(self, agent: ModelAgent) -> list[float] | None:
        """Cached embedding of one agent's operator-declared metadata document."""
        fingerprint = hashlib.sha256(
            "\x1f".join([agent.id, self._agent_descriptor_text(agent)]).encode("utf-8")
        ).hexdigest()
        with self._evidence_lock:
            cached = self._descriptor_vector_cache.get(fingerprint)
        if cached is not None:
            return cached
        embedding_member = self._embedding_agent_id()
        if embedding_member is None:
            return None
        try:
            vectors = self.client.embed(
                self._agent(embedding_member), [self._agent_descriptor_text(agent)]
            )
        except Exception:  # noqa: BLE001 - similarity is best-effort evidence
            return None
        vector = vectors[0] if vectors else None
        if vector is not None:
            self._cache_put(self._descriptor_vector_cache, fingerprint, vector)
        return vector

    def _semantic_affinities(
        self, text: str, agents: list[ModelAgent]
    ) -> dict[str, float | None]:
        """Cosine similarity between task text and every agent's metadata document.

        Returns ``{agent_id: float|None}``; all values are None whenever there
        is no task text, no embedding-capable member, or embedding transport
        fails -- callers then fall back to declaration-only ordering.
        """
        stripped = text.strip() if isinstance(text, str) else ""
        if not stripped or not agents:
            return {agent.id: None for agent in agents}
        task_vector = self._embed_cached(stripped)
        if task_vector is None:
            return {agent.id: None for agent in agents}
        affinities: dict[str, float | None] = {}
        for agent in agents:
            descriptor_vector = self._descriptor_vector_cached(agent)
            affinities[agent.id] = (
                None
                if descriptor_vector is None
                else self._cosine_similarity(task_vector, descriptor_vector)
            )
        return affinities

    # --- structured complexity triage (replaces keyword hint tables) -------

    #: Exact-schema instruction for the single structured triage call.
    TRIAGE_SYSTEM_PROMPT = (
        "You classify whether a user task requires an orchestrated multi-step "
        "workflow (planning plus verification across steps) or one direct answer. "
        'Reply with exactly one JSON object {"workflow_required": true} or '
        '{"workflow_required": false} and nothing else.'
    )

    def _triage_workflow_required(self, text: str) -> bool:
        """Decide route-vs-conduct with one strict JSON verdict; fail to conduct.

        Evidence policy: the decision is made by a model under an exact output
        schema, never by keyword matching. Any failure of the triage call or
        parse fails closed toward the orchestrated path, which carries verifier
        assurance; an absent triage agent degrades to the direct path because
        no evidence source exists at all. Verdicts are cached by content hash.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._evidence_lock:
            cached = self._triage_cache.get(digest)
        if cached is not None:
            return cached
        verdict = self._compute_triage_verdict(text)
        self._cache_put(self._triage_cache, digest, verdict)
        return verdict

    def _compute_triage_verdict(self, text: str) -> bool:
        """One uncached triage decision for :meth:`_triage_workflow_required`."""
        try:
            candidates = self._ranked_agents(text, "worker", free_only=True)
        except RuntimeError:
            candidates = []
        if not candidates:
            candidates = list(self.agents)
        if not candidates:
            return False
        triage_agent = candidates[0]
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            reply = self.client.chat(triage_agent, messages, temperature=0.0)
            return _parse_triage_reply(reply)
        except Exception:  # noqa: BLE001 - fail closed toward verified orchestration
            return True

    def _select_agent(
        self,
        text: str,
        role: str,
        *,
        free_only: bool = False,
        required_tags: tuple[str, ...] = (),
    ) -> ModelAgent:
        """Select one general-chat agent for a conversational role.

        Non-chat discovery rows (embeddings, rerank, transcription, ...) are
        excluded by the capability contract enforced by
        :func:`is_general_chat_agent_model_id`; this is an endpoint-compatibility
        gate, not a task-keyword heuristic.
        """
        ranked = [
            agent
            for agent in self._ranked_agents(
                text, role, free_only=free_only, required_tags=required_tags
            )
            if is_general_chat_agent_model_id(agent.model)
            and all(tag in agent.tags for tag in required_tags)
        ]
        if not ranked:
            raise RuntimeError(f"no chat-compatible agent available for role={role}")
        selected = ranked[0]
        if selected.disabled:  # pragma: no cover
            raise RuntimeError(f"no enabled agent available for role={role}")
        if role in selected.provider_exclusions:  # pragma: no cover
            raise RuntimeError(f"no eligible agent available for role={role}")
        return selected

    def _capability_agents(self, capability: str, model_name: str | None = None) -> list[ModelAgent]:
        """Return measured candidates supporting a capability, optionally within one group."""
        capability = capability.strip().lower()
        capability = {"embeddings": "embedding"}.get(capability, capability)
        if not capability:
            raise ValueError("capability must be a non-empty string")
        virtual_model = model_name in {self.AUTO_MODEL, self.FREE_MODEL}
        free_only = model_name == self.FREE_MODEL
        exact_models = {agent.model for agent in self.candidates}
        requested_group = (
            canonical_group_name(model_name)
            if model_name is not None and model_name not in exact_models and not virtual_model
            else None
        )
        if model_name is not None and not virtual_model and not any(
            agent.model == model_name
            or (
                agent.group_name
                and requested_group is not None
                and canonical_group_name(agent.group_name) == requested_group
            )
            for agent in self.candidates
        ):
            raise ValueError(f"requested model {model_name!r} is not configured")
        ranked = [
            agent
            for agent in self._ranked_agents(
                "", capability, free_only=free_only, chat_only=False
            )
            if not agent.disabled
            and capability in agent.tags
            and capability not in agent.provider_exclusions
            and (
                model_name is None or virtual_model or agent.model == model_name
                or (
                    agent.group_name
                    and requested_group is not None
                    and canonical_group_name(agent.group_name) == requested_group
                )
            )
        ]
        if not ranked:
            raise RuntimeError(f"no enabled agent available for capability={capability}")
        return ranked

    def select_capability_agent(self, capability: str, model_name: str | None = None) -> ModelAgent:
        """Select a measured member supporting a capability, optionally within one group."""
        return self._capability_agents(capability, model_name)[0]

    @staticmethod
    def _equivalent_race_members(
        candidates: list[ModelAgent], *, capability: str
    ) -> list[ModelAgent]:
        """Return replicas proven equivalent for the requested capability."""
        if len(candidates) < 2 or not candidates[0].group_name:
            return []
        declared = [agent for agent in candidates if agent.endpoint_equivalence is not None]
        if len(declared) < 2:
            return []
        first = declared[0]
        contract = EndpointEquivalenceContract(**first.endpoint_equivalence)  # type: ignore[arg-type]
        if (
            capability not in contract.capability_set
            or not contract.hedge_eligible
            or contract.execution_policy != "immediate_race"
        ):
            return []
        peers = [
            agent
            for agent in declared
            if agent.group_name == first.group_name
            and EndpointEquivalenceContract(**agent.endpoint_equivalence) == contract  # type: ignore[arg-type]
        ]
        return peers if len(peers) >= 2 else []

    def _record_endpoint_race(self, outcome: Any, *, capability: str) -> None:
        """Persist secret-free winner and cancellation provenance."""
        self._append_audit_event(
            "equivalent_endpoint_race_completed",
            {
                "capability": capability,
                "winner_endpoint_id": outcome.winner_endpoint_id,
                "attempted_endpoint_ids": list(outcome.attempted_endpoint_ids),
                "cancellation_outcomes": dict(outcome.cancellation_outcomes),
                "completion_ms": outcome.completion_ms,
            },
        )

    def _record_endpoint_attempt(
        self,
        endpoint_id: str,
        value: Any | None,
        error: BaseException | None,
        *,
        capability: str,
    ) -> None:
        """Record reported duplicate usage without treating missing usage as free."""
        usage = None
        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[2], dict):
            usage = value[2]
        elif isinstance(value, dict) and isinstance(value.get("usage"), dict):
            usage = value["usage"]
        self._append_audit_event(
            "equivalent_endpoint_attempt_completed",
            {
                "capability": capability,
                "endpoint_id": endpoint_id,
                "validation_outcome": "provider_error" if error is not None else "completed",
                "usage": usage,
                "duplicate_cost_evidence": (
                    "provider_reported_usage" if usage is not None
                    else "unavailable_requires_provider_invoice"
                ),
            },
        )

    def _record_race_attempt(
        self,
        endpoint_id: str,
        value: Any | None,
        error: BaseException | None,
        *,
        capability: str,
    ) -> None:
        """Share race completion evidence with normal stability/circuit ledgers."""
        self._record_endpoint_attempt(endpoint_id, value, error, capability=capability)
        if error is not None:
            self._group_router.observe_failure(endpoint_id)
            self._record_failure(endpoint_id)

    def _race_attempt_collector(
        self, capability: str
    ) -> tuple[
        Callable[[str, Any | None, BaseException | None], None],
        Callable[[str | None], None],
    ]:
        """Return callbacks that ledger completed loser usage after winner selection."""
        state_lock = threading.Lock()
        pending: list[tuple[str, Any]] = []
        state: dict[str, Any] = {"finalized": False, "winner": None}

        def emit(endpoint_id: str, value: Any) -> None:
            sink = self._race_usage_sink
            if sink is not None:
                sink(endpoint_id, value)

        def completed(
            endpoint_id: str,
            value: Any | None,
            error: BaseException | None,
        ) -> None:
            self._record_race_attempt(
                endpoint_id, value, error, capability=capability
            )
            if error is not None or value is None:
                return
            with state_lock:
                if not state["finalized"]:
                    pending.append((endpoint_id, value))
                    return
                winner = state["winner"]
            if winner is None or endpoint_id != winner:
                emit(endpoint_id, value)

        def finalize(winner_endpoint_id: str | None) -> None:
            with state_lock:
                state["finalized"] = True
                state["winner"] = winner_endpoint_id
                ready = list(pending)
                pending.clear()
            for endpoint_id, value in ready:
                if winner_endpoint_id is None or endpoint_id != winner_endpoint_id:
                    emit(endpoint_id, value)

        return completed, finalize

    def proxy_capability(
        self,
        body: dict[str, Any],
        *,
        capability: str,
        endpoint: str,
        binary: bool = False,
    ) -> dict[str, Any] | tuple[bytes, str]:
        """Route one capability request with measured group-member failover."""
        requested_model = body.get("model")
        candidates = self._capability_agents(capability, requested_model)
        race_members = self._equivalent_race_members(candidates, capability=capability)
        if race_members:
            if len(race_members) > MAX_LOCAL_CONCURRENCY:
                raise ValueError(
                    "immediate_race endpoint count exceeds the supported concurrency capacity"
                )
            def call(agent: ModelAgent) -> dict[str, Any] | tuple[bytes, str]:
                payload = {
                    key: value for key, value in body.items()
                    if key not in self._ORCHESTRATION_ONLY_KEYS
                }
                payload["model"] = agent.model
                provider_endpoint = (
                    "images"
                    if agent.provider_name == "openrouter" and endpoint == "images/generations"
                    else endpoint
                )
                return (
                    self.client.proxy_send_bytes(agent, provider_endpoint, payload)
                    if binary else self.client.proxy_send(agent, provider_endpoint, payload)
                )

            contract = EndpointEquivalenceContract(**race_members[0].endpoint_equivalence)  # type: ignore[arg-type]
            attempt_completed, finalize_attempts = self._race_attempt_collector(capability)
            try:
                outcome = race_first_valid(
                    [
                        EndpointAttempt(
                            agent.id,
                            contract,
                            lambda agent=agent: call(agent),
                            cancellation_supported=False,
                        )
                        for agent in race_members
                    ],
                    validate=(
                        (
                            lambda value: isinstance(value, tuple)
                            and len(value) == 2
                            and isinstance(value[0], bytes)
                            and bool(value[0])
                        )
                        if binary
                        else (lambda value: isinstance(value, dict) and bool(value))
                    ),
                    deadline_seconds=self.client.timeout,
                    max_concurrency=len(race_members),
                    on_attempt_complete=attempt_completed,
                )
            except RuntimeError:
                outcome = None
            finalize_attempts(
                None if outcome is None else outcome.winner_endpoint_id
            )
            if outcome is not None:
                self._record_endpoint_race(outcome, capability=capability)
                self._group_router.observe_success(
                    outcome.winner_endpoint_id, outcome.completion_ms / 1000
                )
                return outcome.value
        last_error: Exception | None = None
        for agent in candidates:
            payload = {
                key: value
                for key, value in body.items()
                if key not in self._ORCHESTRATION_ONLY_KEYS
            }
            payload["model"] = agent.model
            provider_endpoint = (
                "images"
                if agent.provider_name == "openrouter" and endpoint == "images/generations"
                else endpoint
            )
            started_at = time.perf_counter()
            try:
                result = (
                    self.client.proxy_send_bytes(agent, provider_endpoint, payload)
                    if binary
                    else self.client.proxy_send(agent, provider_endpoint, payload)
                )
            except Exception as exc:  # noqa: BLE001 - fail over to the next measured member
                last_error = exc
                self._group_router.observe_failure(agent.id)
                continue
            self._group_router.observe_success(agent.id, time.perf_counter() - started_at)
            return result
        raise RuntimeError(f"all {capability} providers failed") from last_error

    def _invoke(
        self,
        primary: ModelAgent,
        messages: list[ChatMessage],
        *,
        text: str,
        role: str,
        allowed_agent_ids: set[str] | None = None,
        eligibility_role: str | None = None,
    ) -> tuple[str, str, dict[str, Any] | None]:
        """Call an agent with bounded, safety-aware tool retry and failover.

        ``ModelClient`` handles provider transport retries. This layer classifies
        agent/tool-runtime failures: missing tools move to a compatible agent,
        explicitly idempotent transient calls retry the same agent, and ambiguous
        side effects or policy/permission/argument errors fail closed.

        ``eligibility_role`` keeps operator exclusions tied to the role used to
        select the primary when the call's effort profile has a distinct name.
        """
        required_tags = ("vision",) if self._source_image_parts(messages) else ()
        candidates = self._failover_candidates(
            primary,
            text,
            eligibility_role or role,
            required_tags=required_tags,
            allowed_agent_ids=allowed_agent_ids,
        )
        if not candidates and required_tags:
            candidates = self._failover_candidates(
                primary,
                text,
                eligibility_role or role,
                allowed_agent_ids=allowed_agent_ids,
            )
        if not candidates:
            raise RuntimeError(f"no chat-compatible agent available for role={role}")
        race_members = self._equivalent_race_members(candidates, capability="text")
        if race_members:
            if len(race_members) > MAX_LOCAL_CONCURRENCY:
                raise ValueError(
                    "immediate_race endpoint count exceeds the supported concurrency capacity"
                )
            effort_profile = self._role_effort_profile(role)
            request_settings = self.client.request_settings_snapshot()

            def call(agent: ModelAgent) -> tuple[str, str, dict[str, Any] | None]:
                with self.client.request_settings(**request_settings):
                    output = (
                        self.client.chat(agent, messages, effort_profile=effort_profile)
                        if effort_profile is not None
                        else self.client.chat(agent, messages)
                    )
                    usage = self.client.take_usage() if hasattr(self.client, "take_usage") else None
                return output, agent.id, usage

            contract = EndpointEquivalenceContract(**race_members[0].endpoint_equivalence)  # type: ignore[arg-type]
            attempt_completed, finalize_attempts = self._race_attempt_collector("text")
            try:
                outcome = race_first_valid(
                [
                    EndpointAttempt(
                        agent.id,
                        contract,
                        lambda agent=agent: call(agent),
                    )
                    for agent in race_members
                    ],
                    validate=lambda value: isinstance(value[0], str) and bool(value[0]),
                    deadline_seconds=self.client.timeout,
                    max_concurrency=len(race_members),
                    on_attempt_complete=attempt_completed,
                )
            except RuntimeError:
                outcome = None
            finalize_attempts(
                None if outcome is None else outcome.winner_endpoint_id
            )
            if outcome is not None:
                self._record_endpoint_race(outcome, capability="text")
                self._record_success(outcome.winner_endpoint_id)
                usage = outcome.value[2]
                output_tokens = None
                if isinstance(usage, dict):
                    reported = usage.get("completion_tokens", usage.get("output_tokens"))
                    if type(reported) is int and reported > 0:
                        output_tokens = reported
                self._group_router.observe_success(
                    outcome.winner_endpoint_id,
                    outcome.completion_ms / 1000,
                    output_tokens=output_tokens,
                )
                return outcome.value
        retry_limit = min(self.tool_retry_attempts, MAX_TOOL_RETRY_ATTEMPTS)
        for agent in candidates:
            retry_attempt = 0
            while True:
                try:
                    attempt_start = time.perf_counter()
                    effort_profile = self._role_effort_profile(role)
                    output = (
                        self.client.chat(agent, messages, effort_profile=effort_profile)
                        if effort_profile is not None
                        else self.client.chat(agent, messages)
                    )
                except Exception as exc:
                    if agent.group_name or allowed_agent_ids is not None:
                        self._group_router.observe_failure(agent.id)
                    if isinstance(exc, (ProviderResponseError, ToolFallbackStoppedError)):
                        raise
                    decision = classify_tool_failure(exc)
                    action = decision.action
                    # A failed attempt is one Bernoulli stability observation
                    # for measured group routing regardless of what happens next.
                    if (
                        action is ToolFallbackAction.RETRY_SAME_AGENT
                        and retry_attempt < retry_limit
                    ):
                        retry_attempt += 1
                        self._record_tool_fallback(agent.id, decision, retry_attempt)
                        if decision.circuit_failure:  # pragma: no branch - retry-classified failures always trip the circuit
                            self._record_failure(agent.id)
                        if self.tool_retry_backoff_seconds:
                            retry_ceiling = min(
                                self.tool_retry_backoff_seconds
                                * (2.0 ** min(retry_attempt - 1, 16)),
                                30.0,
                            )
                            retry_delay = self._tool_retry_jitter(0.0, retry_ceiling)
                            self._tool_retry_sleep(retry_delay)
                        continue
                    if action is ToolFallbackAction.RETRY_SAME_AGENT:
                        decision = downgrade_to_failover(decision)
                        action = decision.action
                    self._record_tool_fallback(agent.id, decision, retry_attempt)
                    if decision.circuit_failure:
                        self._record_failure(agent.id)
                    if action is ToolFallbackAction.FAIL_CLOSED:
                        raise ToolFallbackStoppedError(agent.id, decision) from None
                    break
                # Success: one Bernoulli observation plus measured latency, and
                # provider-reported completion tokens when available feeding the
                # tokens-per-second EWMA (Jacobson 1988 estimator). Token counts
                # are never inferred from text length or chunk counts.
                usage = self.client.take_usage() if hasattr(self.client, "take_usage") else None
                output_tokens = self._usage_completion_tokens(usage)
                if agent.group_name or allowed_agent_ids is not None:
                    self._group_router.observe_success(
                        agent.id,
                        time.perf_counter() - attempt_start,
                        output_tokens=output_tokens,
                    )
                self._record_success(agent.id)
                return output, agent.id, usage
        raise RuntimeError(f"all {len(candidates)} candidate agents failed for role={role}") from None

    def _record_tool_fallback(
        self,
        agent_id: str,
        decision: ToolFailureDecision,
        retry_attempt: int,
    ) -> None:
        """Record a secret-free audit event for one tool fallback decision."""
        event_detail = {
            "agent_id": agent_id,
            "action": decision.action.value,
            "failure_kind": decision.kind.value,
            "reason_code": decision.reason_code,
            "retry_attempt": retry_attempt,
        }
        observed_kind = (
            decision.kind
            if decision.observed_kind is None
            else decision.observed_kind
        )
        if observed_kind is not decision.kind:
            event_detail["observed_failure_kind"] = observed_kind.value
        self._append_audit_event("tool_fallback_decision", event_detail)

    def _failover_candidates(
        self,
        primary: ModelAgent,
        text: str,
        role: str,
        *,
        required_tags: tuple[str, ...] = (),
        allowed_agent_ids: set[str] | None = None,
    ) -> list[ModelAgent]:
        try:
            ranked = self._ranked_agents(text, role, required_tags=required_tags)
        except RuntimeError:
            if required_tags:
                return []
            raise
        if allowed_agent_ids is not None:
            ranked = [agent for agent in ranked if agent.id in allowed_agent_ids]
        if allowed_agent_ids is None:
            ranked = [
                agent
                for agent in ranked
                if (
                    canonical_group_name(agent.group_name)
                    == canonical_group_name(primary.group_name)
                    if primary.group_name and agent.group_name
                    else not primary.group_name and not agent.group_name
                )
            ]
        ordered = (
            [primary]
            if allowed_agent_ids is None or primary.id in allowed_agent_ids
            else []
        ) + [agent for agent in ranked if agent.id != primary.id]
        ordered = [
            agent
            for agent in ordered
            if is_general_chat_agent_model_id(agent.model)
            and all(tag in agent.tags for tag in required_tags)
        ]
        eligible = [agent for agent in ordered if not agent.disabled and role not in agent.provider_exclusions]
        healthy = [agent for agent in eligible if not self._circuit_open(agent.id)]
        # If every eligible agent is circuit-open, still probe them rather than fail with no attempt.
        return healthy or eligible

    def _circuit_open(self, agent_id: str) -> bool:
        with self._circuit_lock:
            state = self._circuit.get(agent_id)
            if not state or state["failures"] < self.circuit_failure_threshold:
                return False
            if time.monotonic() - state["opened_at"] >= self.circuit_reset_seconds:
                state["failures"] = 0.0
                state["opened_at"] = 0.0
                return False
            return True

    def _record_failure(self, agent_id: str) -> None:
        with self._circuit_lock:
            state = self._circuit.setdefault(agent_id, {"failures": 0.0, "opened_at": 0.0})
            state["failures"] += 1.0
            if state["failures"] >= self.circuit_failure_threshold and not state["opened_at"]:
                state["opened_at"] = time.monotonic()

    def _record_success(self, agent_id: str) -> None:
        with self._circuit_lock:
            self._circuit.pop(agent_id, None)

    def _agent(self, agent_id: str) -> ModelAgent:
        for agent in self.candidates:
            if agent.id == agent_id:
                return agent
        raise KeyError(agent_id)  # pragma: no cover

    def _agent_in_pool(self, agent_pool_id: str, worker_agent_id: str) -> ModelAgent:
        """Resolve an agent only through the pool boundary it can belong to.

        The current persistence model has one ``default`` pool and stores
        agents by ID. Keeping the pool check beside the lookup prevents a
        future multi-pool change from turning separately validated path
        parameters into an object-authorization bypass.
        """
        if agent_pool_id != "default":
            raise KeyError(agent_pool_id)
        return self._agent(worker_agent_id)

    def _needs_workflow(self, text: str) -> bool:
        """Route-vs-conduct decision from a strict structured triage verdict.

        Keyword hint tables are intentionally absent: keyword matching cannot
        handle negation, mixed language, or tasks that quote trigger words, and
        hand-tuned thresholds are not evidence. The verdict comes from one
        exact-schema model call (cached by content hash) and fails closed to
        the orchestrated path on any uncertainty.
        """
        return bool(self._triage_fn(text))

    def _latest_user_text(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            text = _coerce_message_content_text(message.get("content", ""))
            if text:
                return text
        return ""  # pragma: no cover

    @staticmethod
    def _source_image_parts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Copy validated image parts so evidence steps receive source pixels."""
        return [
            copy.deepcopy(part)
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]

    def _model_judge_verification(
        self,
        task: str,
        fallback: dict[str, Any],
        *,
        free_only: bool = False,
        allowed_agent_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Ask a model for a strict structured verdict and fail closed on uncertainty."""
        verifier_output = fallback.get("verifier_output", "")
        if not verifier_output:
            return {
                "accepted": False,
                "reason": "model judge requires a non-empty verifier report",
                "verifier_output": verifier_output,
                "judge": "model",
            }
        try:
            components = _resolve_fast_mlsirm_components()
        except Exception:  # noqa: BLE001 - a broken installed judge must not bypass the required path
            return {
                "accepted": False,
                "reason": "fast-mlsirm judge could not be loaded; verification failed closed",
                "verifier_output": verifier_output,
                "judge": "model",
            }
        if components is None:
            return {
                "accepted": False,
                "reason": "fast-mlsirm judge is unavailable; verification failed closed",
                "verifier_output": verifier_output,
                "judge": "model",
            }
        try:
            judge = next(
                agent
                for agent in self._ranked_agents(task, "verifier", free_only=free_only)
                if allowed_agent_ids is None or agent.id in allowed_agent_ids
            )
            # The judge is one bounded provider call.  Do not pass the
            # planning strategy ("template"/"generated") as an
            # orchestration mode or recursively conduct another workflow.
            judge_adapter = _FastMLSIJudgeAdapter(
                self,
                task,
                judge.id,
                mode="route",
                allowed_agent_ids=allowed_agent_ids,
            )
            fast_judge = components.judge_cls(
                judge_adapter,
                mode="route",
                accept_threshold=0.7,
            )
            result = fast_judge.judge(
                task=task,
                answer=verifier_output,
                criteria=(
                    components.criterion_cls(
                        criterion_id="evidence_quality",
                        description="Does the verifier output identify concrete evidence and caveats with actionable impact?",
                        weight=1.0,
                    ),
                    components.criterion_cls(
                        criterion_id="risk_signal",
                        description="Does the verifier output mention substantive risks and constraints with support?",
                        weight=1.0,
                    ),
                ),
            )
            verification = {
                "accepted": result.accepted,
                "reason": result.rationale,
                "verifier_output": verifier_output,
                "judge": "model",
            }
            if judge_adapter.served_agent_id is not None and judge_adapter.served_agent_id != judge.id:
                verification["judge_agent_id"] = judge_adapter.served_agent_id
            if result.usage:
                verification["judge_usage"] = result.usage
            verification["judge_orchestration_mode"] = result.orchestration_mode
            criterion_scores = getattr(result, "criterion_scores", None)
            to_irt_row = getattr(result, "to_irt_row", None)
            if isinstance(criterion_scores, Mapping) and callable(to_irt_row):
                try:
                    irt_row = to_irt_row(item_type="dichotomous")
                except Exception:  # noqa: BLE001 - invalid IRT projection must not be published
                    return {
                        "accepted": False,
                        "reason": "model judge returned an invalid multi-item IRT projection; verification failed closed",
                        "verifier_output": verifier_output,
                        "judge": "model",
                    }
                if (
                    len(criterion_scores) < 2
                    or type(irt_row) not in (tuple, list)
                    or len(irt_row) != len(criterion_scores)
                ):
                    return {
                        "accepted": False,
                        "reason": "model judge returned an invalid multi-item IRT projection; verification failed closed",
                        "verifier_output": verifier_output,
                        "judge": "model",
                    }
                verification["judge_criterion_scores"] = dict(criterion_scores)
                verification["judge_irt_item_type"] = "dichotomous"
                verification["judge_irt_row"] = list(irt_row)
            return verification
        except components.format_error:
            return {
                "accepted": False,
                "reason": "model judge returned an invalid structured verdict; verification failed closed",
                "verifier_output": verifier_output,
                "judge": "model",
            }
        except Exception:  # noqa: BLE001 - judge failure must not break the request
            return {
                "accepted": False,
                "reason": "model judge unavailable; verification failed closed",
                "verifier_output": verifier_output,
                "judge": "model",
            }

    def _judge_verifier_output(self, verifier_output: str, thinker_output: str, worker_output: str) -> dict[str, Any]:
        """Prepare evidence for the model judge without making a heuristic decision."""
        del thinker_output, worker_output
        return {
            "accepted": False,
            "reason": "model judgment required; keyword matching is disabled",
            "verifier_output": verifier_output,
        }

    def _protected_event_detail(self, detail: dict[str, Any], pii_fields: Iterable[str]) -> dict[str, Any]:
        """Encrypt explicitly declared PII fields before an event enters memory or storage."""
        fields = tuple(pii_fields)
        if not fields:
            return detail
        encryptor = self._pii_encryptors.get(self._pii_key_name)
        if encryptor is None:
            encryptor = load_pii_encryptor(self._pii_key_name)
            self._pii_encryptors[self._pii_key_name] = encryptor
        return encryptor.encrypt_fields(detail, fields)

    def _append_audit_event(
        self,
        event_type: str,
        detail: dict[str, Any],
        *,
        pii_fields: Iterable[str] = (),
        stream: str = "audit",
        durable: bool = True,
    ) -> None:
        """Append a durable event to a bounded audit stream by default."""
        event = {
            "created_at": int(time.time()),
            "event_type": event_type,
            "event_detail": self._protected_event_detail(detail, pii_fields),
        }
        events = self._authorization_events if stream == "authorization" else self._audit_events
        events.append(event)
        if self._store is not None:
            self._store.save(stream, None, event, durable=durable)

    def record_authorization_decision(
        self,
        *,
        scope: str,
        purpose: str,
        allowed: bool,
        reason: str,
        durable: bool = False,
    ) -> None:
        """Record a secret-free role/purpose authorization decision."""
        self._append_audit_event(
            "authorization_decision",
            {
                "scope": scope,
                "purpose": purpose,
                "allowed": bool(allowed),
                "reason": reason,
            },
            stream="authorization",
            durable=durable,
        )

    def _infer_provider_name(self, base_url: str) -> str:
        if base_url.startswith("mock://"):
            return f"mock-{base_url.removeprefix('mock://')}"
        if "://" in base_url:
            return base_url.split("//", 1)[-1].split("/", 1)[0]
        return base_url  # pragma: no cover

    def _agent_to_admin_payload(self, agent: ModelAgent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "model": agent.model,
            "base_url": agent.base_url,
            "provider_name": agent.provider_name or self._infer_provider_name(agent.base_url),
            "priority": agent.priority,
            "tags": list(agent.tags),
            "status": "disabled" if agent.disabled else "active",
            "provider_exclusions": list(agent.provider_exclusions),
            "group_name": agent.group_name,
            "group_routing": self._group_router.member_report(agent.id) if agent.group_name else None,
        }

    def list_agents(self, page_number: int = 1, page_size: int = 10) -> list[dict[str, Any]]:
        """Return a paginated admin-safe view of configured agents."""
        if page_number < 1 or page_size < 1:  # pragma: no cover
            raise ValueError("page_number/page_size must be >= 1")
        start = (page_number - 1) * page_size
        end = start + page_size
        return [self._agent_to_admin_payload(agent) for agent in self.candidates[start:end]]

    def list_openai_models(self) -> dict[str, Any]:
        """Return an OpenAI-compatible ``/v1/models`` list from the agent pool.

        Buyers discover selectable model ids without admin-scope agent pool access.
        Each enabled agent model appears once; gateway default
        ``contextual-orchestrator`` is always first. Disabled models are omitted
        deliberately (matching real OpenAI API behavior: you only see models you
        can actually call) rather than listed with a "disabled" status -- showing
        an inference-scope caller a model it cannot use is its own kind of
        dishonesty. Operators get disabled-agent visibility through the
        admin-scope ``list_agents``/``/admin`` surface instead.
        """
        created = 1_700_000_000  # stable epoch so list responses are deterministic
        data: list[dict[str, Any]] = [
            {
                "id": "contextual-orchestrator",
                "object": "model",
                "created": created,
                "owned_by": "contextual-orchestrator",
            }
        ]
        data.append({
            "id": self.AUTO_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "contextual-orchestrator",
        })
        if any(self._is_free_agent(agent) for agent in self.agents):
            data.append({
                "id": self.FREE_MODEL,
                "object": "model",
                "created": created,
                "owned_by": "contextual-orchestrator",
            })
        seen: set[str] = {item["id"] for item in data}
        # Model-group aliases are addressable model ids (a logical name routes
        # to the best measured member), so advertise them like real models.
        for group in self.list_model_groups():
            if not group.get("enabled_member_count"):
                continue
            group_alias = str(group["group_name"])
            if group_alias in seen:
                continue
            seen.add(group_alias)
            data.append(
                {
                    "id": group_alias,
                    "object": "model",
                    "created": created,
                    "owned_by": "model_group",
                }
            )
        # ``self.agents`` is the enabled-only projection of ``self.candidates``
        # (maintained at every pool mutation), so no disabled agent can appear
        # in this loop.
        for agent in self.agents:
            model_id = str(agent.model).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": agent.provider_name
                    or self._infer_provider_name(agent.base_url)
                    or "agent_pool",
                }
            )
        return {"object": "list", "data": data}

    def get_openai_model(self, model_id: str) -> dict[str, Any]:
        """Return one OpenAI model object or raise ``KeyError`` when unknown."""
        wanted = (model_id or "").strip()
        if not wanted:
            raise KeyError(model_id)
        for item in self.list_openai_models()["data"]:
            if item["id"] == wanted:
                return item
        raise KeyError(model_id)

    def list_recent_runs(
        self,
        page_number: int = 1,
        page_size: int = 10,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a paginated list of recent workflow run records."""
        if page_number < 1 or page_size < 1:  # pragma: no cover
            raise ValueError("page_number/page_size must be >= 1")
        start = (page_number - 1) * page_size
        end = start + page_size
        run_ids = [
            run_id
            for run_id in self._run_order
            if owner_id is None or self._workflow_runs[run_id].get("owner_id") == owner_id
        ][start:end]
        return [self._workflow_runs[run_id] for run_id in run_ids]

    def count_workflow_runs(self, owner_id: str | None = None) -> int:
        """Count only workflow runs visible to the requested owner."""
        return sum(
            owner_id is None or record.get("owner_id") == owner_id
            for record in self._workflow_runs.values()
        )

    def list_recent_audit_events(
        self,
        page_number: int = 1,
        page_size: int = 25,
        *,
        role: str | None = None,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent audit events, decrypting PII only for authorized replay."""

        if page_number < 1 or page_size < 1:  # pragma: no cover
            raise ValueError("page_number/page_size must be >= 1")
        events = list(self._audit_events)
        start = (page_number - 1) * page_size
        end = start + page_size
        total = len(events)
        left = max(0, total - end)
        right = max(0, total - start)
        selected = list(reversed(events[left:right]))
        if role != "admin" or purpose != "audit_replay":
            return selected
        restored: list[dict[str, Any]] = []
        encryptors: dict[str, Any] = {}
        for event in selected:
            detail = event.get("event_detail")
            if not is_encrypted_detail(detail):
                restored.append(event)
                continue
            restored_event = dict(event)
            try:
                metadata = detail.get(ENCRYPTED_FIELDS_KEY)
                key_name = metadata.get("key_name") if isinstance(metadata, dict) else self._pii_key_name
                if not isinstance(key_name, str) or not key_name:
                    raise PiiProtectionError("encrypted field metadata has no valid key name")
                encryptor = encryptors.get(key_name)
                if encryptor is None:
                    encryptor = load_pii_encryptor(key_name)
                    encryptors[key_name] = encryptor
                restored_event["event_detail"] = encryptor.decrypt_fields(detail)
            except PiiProtectionError:
                restored_event["event_detail"] = {
                    **detail,
                    "__pii_protection_error__": "unavailable",
                }
            restored.append(restored_event)
        return restored

    def list_recent_authorization_decisions(self, page_number: int = 1, page_size: int = 25) -> list[dict[str, Any]]:
        """Return recent secret-free authorization decisions in newest-first order."""
        if page_number < 1 or page_size < 1:  # pragma: no cover
            raise ValueError("page_number/page_size must be >= 1")
        events = list(self._authorization_events)
        start = (page_number - 1) * page_size
        end = start + page_size
        total = len(events)
        left = max(0, total - end)
        right = max(0, total - start)
        return list(reversed(events[left:right]))

    def record_analytics_event(
        self,
        event_name: str,
        detail: dict[str, Any],
        *,
        pii_fields: Iterable[str] = (),
    ) -> None:
        """Record a compact in-memory analytics event without prompt or output text."""
        require_object_name(event_name, "analytics.event_name")
        event = {
            "event_time": int(time.time()),
            "event_name": event_name,
            "event_detail": redact_value(self._protected_event_detail(detail, pii_fields)),
        }
        self._analytics_events.append(event)
        if self._store is not None:
            self._store.save("analytics", None, event)

    def _run_budget_output_by_model(self, record: Mapping[str, Any]) -> dict[str, int]:
        """Return the exact per-model output-token contribution of one run."""
        model_by_agent = {agent.id: agent.model for agent in self.candidates}
        output_by_model: dict[str, int] = {}
        for step in record.get("trace", []):
            model = step.get("model_name") or model_by_agent.get(
                step.get("served_agent_id") or step.get("agent_id"), "unknown"
            )
            output_tokens, _reported = _step_output_tokens(step)
            output_by_model[model] = output_by_model.get(model, 0) + output_tokens
        return output_by_model

    def _replace_workflow_run(self, record: dict[str, Any]) -> None:
        """Store one run and update its constant-time budget meter atomically."""
        model_by_agent = {agent.id: agent.model for agent in self.candidates}
        for step in record.get("trace", []):
            if not step.get("model_name"):
                agent_id = step.get("served_agent_id") or step.get("agent_id")
                step["model_name"] = model_by_agent.get(agent_id, "unknown")
        run_id = record["workflow_run_id"]
        with self._budget_spend_lock:
            previous = self._workflow_runs.get(run_id)
            for sign, run in ((-1, previous), (1, record)):
                if run is None:
                    continue
                for model, output_tokens in self._run_budget_output_by_model(run).items():
                    before = self._budget_model_output_tokens.get(model, 0)
                    after = before + sign * output_tokens
                    price = self.price_per_million.get(model)
                    if price is not None:
                        self._budget_spent_cost_usd += _cost_usd_decimal(
                            after, price
                        ) - _cost_usd_decimal(before, price)
                    if after:
                        self._budget_model_output_tokens[model] = after
                    else:
                        self._budget_model_output_tokens.pop(model, None)
                    self._budget_spent_output_tokens += sign * output_tokens
            self._workflow_runs[run_id] = record

    def _rebuild_budget_meter(self) -> None:
        """Reconcile the meter after a rare agent-pool identity change."""
        with self._budget_spend_lock:
            output_by_model: dict[str, int] = {}
            for run in self._workflow_runs.values():
                for model, output_tokens in self._run_budget_output_by_model(run).items():
                    output_by_model[model] = output_by_model.get(model, 0) + output_tokens
            self._budget_model_output_tokens = output_by_model
            self._budget_spent_output_tokens = sum(output_by_model.values())
            self._budget_spent_cost_usd = sum(
                (
                    _cost_usd_decimal(output_tokens, self.price_per_million[model])
                    for model, output_tokens in output_by_model.items()
                    if model in self.price_per_million
                ),
                start=Decimal(0),
            )

    def spend_analytics(self, price_per_million: dict[str, float] | None = None) -> dict[str, Any]:
        """Estimated token and cost spend per model, aggregated from workflow runs.

        Tokens are ESTIMATED from runtime output text (~4 chars/token), not provider-reported
        usage. Cost is computed only for models with an operator-supplied price; models without
        one are reported under ``unpriced_models`` with a null cost. This is the honest local
        floor for spend observability, not a billing system.
        """
        prices = {**self.price_per_million, **(price_per_million or {})}
        model_by_agent = {agent.id: agent.model for agent in self.candidates}
        by_model: dict[str, dict[str, Any]] = {}
        total_output_tokens = 0
        total_prompt_tokens = 0
        reported_prompt_tokens = 0
        any_reported_prompt = False

        for run in self._workflow_runs.values():
            total_prompt_tokens += estimate_tokens(run.get("prompt_text", ""))
            for step in run["trace"]:
                model = step.get("model_name") or model_by_agent.get(
                    step.get("served_agent_id") or step.get("agent_id"), "unknown"
                )
                estimated = estimate_tokens(step.get("output", ""))
                usage = step.get("usage")
                reported_prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                if isinstance(reported_prompt, int):
                    reported_prompt_tokens += reported_prompt
                    any_reported_prompt = True
                effective, is_reported = _step_output_tokens(step)
                bucket = by_model.setdefault(
                    model, {"estimated_output_tokens": 0, "output_tokens": 0, "step_count": 0, "reported_steps": 0}
                )
                bucket["estimated_output_tokens"] += estimated
                bucket["output_tokens"] += effective
                bucket["step_count"] += 1
                bucket["reported_steps"] += 1 if is_reported else 0
                total_output_tokens += effective

        rows: list[dict[str, Any]] = []
        unpriced: list[str] = []
        total_cost_usd = Decimal(0)
        for model, bucket in sorted(by_model.items()):
            price = prices.get(model)
            cost_decimal = (
                _cost_usd_decimal(bucket["output_tokens"], price)
                if price is not None
                else None
            )
            cost = float(cost_decimal) if cost_decimal is not None else None
            if price is None:
                unpriced.append(model)
            else:
                total_cost_usd += cost_decimal
            if bucket["reported_steps"] == 0:
                usage_source = "estimated"
            elif bucket["reported_steps"] == bucket["step_count"]:
                usage_source = "reported"
            else:
                usage_source = "mixed"
            rows.append({
                "model": model,
                "estimated_output_tokens": bucket["estimated_output_tokens"],
                "output_tokens": bucket["output_tokens"],
                "usage_source": usage_source,
                "step_count": bucket["step_count"],
                "price_per_million_usd": price,
                "estimated_cost_usd": cost,
            })

        return {
            "measurement_status": "local_runtime_estimate",
            "source_note": (
                "output_tokens use provider-reported usage when available (usage_source=reported/mixed) and "
                "fall back to a ~4 chars/token estimate otherwise; cost = output_tokens x operator-supplied price only."
            ),
            "pricing_configured": bool(prices),
            "totals": {
                "run_count": len(self._workflow_runs),
                "estimated_output_tokens": total_output_tokens,
                "estimated_prompt_tokens": total_prompt_tokens,
                "reported_prompt_tokens": reported_prompt_tokens,
                "prompt_tokens_source": "reported" if any_reported_prompt else "estimated",
                "estimated_cost_usd": float(total_cost_usd) if prices else None,
                "currency": "USD",
            },
            "by_model": rows,
            "unpriced_models": unpriced,
            "budget": self._budget_block(
                total_output_tokens,
                float(total_cost_usd) if prices else None,
            ),
        }

    def _budget_block(self, spent_tokens: int, spent_cost: float | None) -> dict[str, Any]:
        token_limit = self.budget_max_output_tokens
        cost_limit = self.budget_max_cost_usd
        exceeded = bool(
            (token_limit is not None and spent_tokens >= token_limit)
            or (cost_limit is not None and spent_cost is not None and spent_cost >= cost_limit)
        )
        return {
            "enabled": token_limit is not None or cost_limit is not None,
            "max_output_tokens": token_limit,
            "max_cost_usd": cost_limit,
            "spent_output_tokens": spent_tokens,
            "spent_cost_usd": spent_cost,
            "remaining_output_tokens": max(0, token_limit - spent_tokens) if token_limit is not None else None,
            "remaining_cost_usd": (
                round(max(0.0, cost_limit - spent_cost), 6)
                if cost_limit is not None and spent_cost is not None else None
            ),
            "exceeded": exceeded,
        }

    def budget_status(self) -> dict[str, Any]:
        """Current spend-budget state (limits, spent, remaining, exceeded)."""
        with self._budget_spend_lock:
            spent_tokens = self._budget_spent_output_tokens
            spent_cost = float(self._budget_spent_cost_usd)
        return self._budget_block(
            spent_tokens,
            spent_cost if self.price_per_million else None,
        )

    def analytics_snapshot(self, locale_bundles: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
        """Return source-backed local KPI definitions from in-memory runtime state."""
        runs = list(self._workflow_runs.values())
        conducted_runs = [run for run in runs if run["mode"] == "conduct"]
        trace_complete_count = sum(1 for run in conducted_runs if self._is_trace_complete(run))
        policy_safe_count = sum(1 for run in runs if self._is_policy_safe_run(run))
        event_counts = Counter(event["event_name"] for event in self._analytics_events)
        successful_chat_requests = sum(
            1
            for event in self._analytics_events
            if event["event_name"] == "chat_completion_requested"
            and event["event_detail"].get("status_code") == 200
        )
        route_count = sum(1 for run in runs if run["mode"] == "route")
        conduct_count = sum(1 for run in runs if run["mode"] == "conduct")
        step_count = sum(len(run["trace"]) for run in runs)
        provider_exclusion_misses = sum(self._provider_exclusion_miss_count(run) for run in runs)
        locale_parity = self._locale_key_parity(locale_bundles or {})

        return {
            "measurement_status": "local_runtime_snapshot",
            "source_note": "Metrics are measured from this process in-memory runtime, not production telemetry.",
            "event_counts": dict(sorted(event_counts.items())),
            "kpis": [
                {
                    "metric_name": "compatible_api_adoption",
                    "label": "Compatible API adoption",
                    "value": successful_chat_requests,
                    "unit": "successful_requests",
                    "source": "chat_completion_requested events",
                },
                {
                    "metric_name": "trace_complete_workflow_rate",
                    "label": "Trace-complete workflow rate",
                    "numerator": trace_complete_count,
                    "denominator": len(conducted_runs),
                    "value_percent": self._percent(trace_complete_count, len(conducted_runs)),
                    "source": "workflow_runs conduct traces",
                },
                {
                    "metric_name": "policy_safe_routing_rate",
                    "label": "Policy-safe routing rate",
                    "numerator": policy_safe_count,
                    "denominator": len(runs),
                    "value_percent": self._percent(policy_safe_count, len(runs)),
                    "source": "workflow_runs policy snapshots",
                },
            ],
            "drivers": [
                {
                    "metric_name": "route_versus_conduct_mix",
                    "label": "Route-versus-conduct mix",
                    "counts": {"route": route_count, "conduct": conduct_count},
                    "source": "workflow_runs mode",
                },
                {
                    "metric_name": "evaluation_replay_usage",
                    "label": "Evaluation replay usage",
                    "value": event_counts.get("evaluation_run_created", 0),
                    "unit": "runs",
                    "source": "evaluation_run_created events",
                },
                {
                    "metric_name": "agent_health_coverage",
                    "label": "Agent health coverage",
                    "numerator": len([agent for agent in self.agents if agent.id and agent.model and agent.base_url]),
                    "denominator": len(self.agents),
                    "value_percent": self._percent(
                        len([agent for agent in self.agents if agent.id and agent.model and agent.base_url]),
                        len(self.agents),
                    ),
                    "source": "agent pool configuration",
                },
            ],
            "guardrails": [
                {
                    "metric_name": "provider_exclusion_miss_rate",
                    "label": "Provider exclusion miss rate",
                    "value": provider_exclusion_misses,
                    "denominator": step_count,
                    "value_percent": self._percent(provider_exclusion_misses, step_count),
                    "source": "workflow trace agent selections",
                },
                {
                    "metric_name": "locale_key_parity",
                    "label": "Locale key parity",
                    **locale_parity,
                    "source": "admin locale bundles",
                },
            ],
        }

    def sales_readiness_report(
        self,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a local, evidence-backed sales-readiness gate for enterprise pilots."""
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        runs = list(self._workflow_runs.values())
        conducted_runs = [run for run in runs if run["mode"] == "conduct"]
        trace_complete_count = sum(1 for run in conducted_runs if self._is_trace_complete(run))
        event_counts = analytics["event_counts"]
        criteria = [
            self._criterion(
                "api_compatibility",
                "OpenAI-compatible API",
                "pass" if event_counts.get("chat_completion_requested", 0) > 0 else "warn",
                f"{event_counts.get('chat_completion_requested', 0)} compatible chat requests recorded",
                "Run a /v1/chat/completions smoke test before an enterprise evaluation.",
            ),
            self._criterion(
                "admin_evidence",
                "Operator evidence surface",
                "pass" if admin_state["agents"] and admin_state["policy"] else "fail",
                f"{len(admin_state['agents'])} agents, {len(admin_state['recent_audit_events'])} audit events exposed",
                "Expose agent pool, policy, and audit state before positioning the product as sellable.",
            ),
            self._criterion(
                "trace_evidence",
                "Workflow trace evidence",
                "pass" if trace_complete_count > 0 else "warn",
                f"{trace_complete_count} complete conducted traces across {len(conducted_runs)} conducted runs",
                "Run a conduct-mode workflow so access-list and verifier evidence are visible.",
            ),
            self._criterion(
                "evaluation_replay",
                "Evaluation replay",
                "pass" if event_counts.get("evaluation_run_created", 0) > 0 else "warn",
                f"{event_counts.get('evaluation_run_created', 0)} evaluation replay runs recorded",
                "Run at least one evaluation replay before customer-facing pilot review.",
            ),
            self._security_posture_criterion(security_profile or {}),
            self._criterion(
                "analytics_truthfulness",
                "Analytics truthfulness",
                "pass" if analytics["measurement_status"] == "local_runtime_snapshot" else "fail",
                analytics["source_note"],
                "Label metrics as proposed definitions unless backed by measured runtime telemetry.",
            ),
            self._locale_readiness_criterion(analytics),
            self._provider_egress_criterion(),
        ]
        summary = self._criteria_summary(criteria)
        readiness_summary = {"pass": summary["pass"], "warn": summary["warn"], "fail": summary["fail"]}
        if summary["fail"]:
            readiness_status = "not_ready"
        elif summary["warn"]:
            readiness_status = "pilot_ready_with_warnings"
        else:
            readiness_status = "sales_ready"

        return {
            "readiness_status": readiness_status,
            "measurement_status": "local_runtime_snapshot",
            "source_note": (
                "Sales readiness is based on this process-local runtime, configuration, and "
                "documentation evidence; it is not a production compliance certificate."
            ),
            "summary": readiness_summary,
            "readiness_summary": readiness_summary,
            "criteria": criteria,
        }

    def commercial_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a diligence-oriented readiness gate for high-value enterprise sales."""
        sales_readiness = self.sales_readiness_report(
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        sales_rows = self._criteria_by_name(sales_readiness["criteria"])
        analytics_guardrails = self._metrics_by_name(analytics["guardrails"])
        documentation = self._commercial_documentation_profile()
        security_profile = security_profile or {}
        policy_safe_metric = self._metrics_by_name(analytics["kpis"])["policy_safe_routing_rate"]
        provider_metric = analytics_guardrails["provider_exclusion_miss_rate"]
        locale_metric = analytics_guardrails["locale_key_parity"]

        criteria = [
            self._criterion(
                "product_capability_evidence",
                "Product capability evidence",
                "pass" if sales_readiness["readiness_status"] == "sales_ready" else "warn",
                (
                    f"sales_readiness={sales_readiness['readiness_status']}; "
                    f"{sales_readiness['readiness_summary']['pass']} sales criteria passing"
                ),
                "Resolve all sales-readiness warnings before presenting the product for a high-value diligence review.",
            ),
            self._criterion(
                "security_and_access_control",
                "Security and access control",
                "pass"
                if sales_rows["security_posture"]["status"] == "pass"
                and sales_rows["provider_egress_safety"]["status"] == "pass"
                else "fail",
                (
                    f"{sales_rows['security_posture']['evidence']}; "
                    f"{sales_rows['provider_egress_safety']['evidence']}"
                ),
                "Keep split admin/inference tokens, private bind defaults, hidden traces, and safe provider egress.",
            ),
            self._criterion(
                "operational_resilience",
                "Operational resilience",
                "pass"
                if int(security_profile.get("rate_limit_requests") or 0) > 0
                and int(security_profile.get("max_concurrent_runs") or 0) > 0
                and policy_safe_metric.get("value_percent") == 100.0
                else "warn",
                (
                    f"rate_limit_requests={security_profile.get('rate_limit_requests')}; "
                    f"max_concurrent_runs={security_profile.get('max_concurrent_runs')}; "
                    f"policy_safe_routing_rate={policy_safe_metric.get('value_percent')}%"
                ),
                "Publish production SLOs, backup policy, and incident runbooks before a production sale.",
            ),
            self._criterion(
                "audit_and_compliance_evidence",
                "Audit and compliance evidence",
                "pass"
                if sales_rows["trace_evidence"]["status"] == "pass"
                and provider_metric.get("value") == 0
                else "warn",
                (
                    f"{sales_rows['trace_evidence']['evidence']}; "
                    f"provider_exclusion_misses={provider_metric.get('value')}"
                ),
                "Capture customer-specific access reports and compliance exceptions during paid pilot onboarding.",
            ),
            self._criterion(
                "buyer_due_diligence_packet",
                "Buyer due-diligence packet",
                "pass" if not documentation["missing_documents"] else "warn",
                (
                    f"{documentation['present_count']}/{documentation['required_count']} required documents present; "
                    f"missing={', '.join(documentation['missing_documents']) or 'none'}"
                ),
                "Complete README, security, API, analytics, product, and commercial readiness documents.",
            ),
            self._criterion(
                "support_and_localization",
                "Support and localization",
                "pass" if locale_metric.get("value_percent") == 100.0 and documentation["has_security_policy"] else "warn",
                (
                    f"locale_key_parity={locale_metric.get('value_percent')}%; "
                    f"security_policy={documentation['has_security_policy']}"
                ),
                "Keep Korean and English operator copy aligned and publish support ownership for customer operations.",
            ),
            self._criterion(
                "commercial_value_case",
                "Commercial value case",
                "pass" if target_contract_value_krw >= DEFAULT_COMMERCIAL_TARGET_VALUE_KRW else "warn",
                (
                    f"target_contract_value_krw={target_contract_value_krw:,}; "
                    "value case uses compatibility API, evidence control plane, replay, and audit controls"
                ),
                "Anchor high-value sales review at KRW 2,000,000,000 or higher with buyer-specific ROI evidence.",
            ),
        ]
        summary = self._criteria_summary(criteria)
        commercial_summary = {"pass": summary["pass"], "warn": summary["warn"], "fail": summary["fail"]}
        if commercial_summary["fail"]:
            commercial_status = "not_commercial_ready"
        elif commercial_summary["warn"]:
            commercial_status = "commercial_ready_with_warnings"
        else:
            commercial_status = "commercial_ready"

        return {
            "commercial_status": commercial_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_due_diligence_snapshot",
            "source_note": (
                "Commercial readiness is based on process-local runtime, repository documentation, "
                "security configuration, and analytics evidence; it is not a valuation guarantee, "
                "purchase commitment, or production compliance certificate."
            ),
            "summary": commercial_summary,
            "commercial_summary": commercial_summary,
            "criteria": criteria,
            "documentation": documentation,
            "sales_readiness": sales_readiness,
        }

    def commercial_evidence_manifest_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the evidence index for commercial readiness review."""
        commercial = self.commercial_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        commercial_rows = self._criteria_by_name(commercial["criteria"])
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        items = [
            self._buyer_evidence_item(
                "product_scope",
                "Product scope",
                "Economic buyer",
                ["README.md", "docs/product_planning.md", "docs/commercial_readiness.md"],
                "repository_artifact",
                "ready" if all(has_file(path) for path in ("README.md", "docs/product_planning.md", "docs/commercial_readiness.md")) else "blocked",
                "Single enterprise orchestration control plane is documented.",
                "Keep product scope unified for buyer review.",
            ),
            self._buyer_evidence_item(
                "compatible_inference_api",
                "Compatible inference API",
                "Platform reviewer",
                ["/v1/chat/completions", "docs/rest_api_design.md", "tests/test_api_contract.py"],
                "repository_artifact",
                "ready" if has_file("docs/rest_api_design.md") and has_file("tests/test_api_contract.py") else "blocked",
                "OpenAI-compatible endpoint and API contract tests are present.",
                "Restore API contract docs and tests before buyer review.",
            ),
            self._buyer_evidence_item(
                "admin_evidence_control_plane",
                "Admin evidence control plane",
                "Platform operator",
                ["/admin", "/admin/state", "docs/screen_design.md"],
                "repository_artifact",
                "ready" if has_file("docs/screen_design.md") else "blocked",
                "Admin screen design and runtime state endpoint are present.",
                "Restore admin evidence design before buyer review.",
            ),
            self._buyer_evidence_item(
                "sales_readiness",
                "Sales readiness",
                "Product owner",
                ["/api/v1/sales_readiness/latest", "tests/test_sales_readiness.py"],
                "measured_local",
                "ready" if commercial["sales_readiness"]["readiness_summary"]["fail"] == 0 else "blocked",
                f"sales_readiness={commercial['sales_readiness']['readiness_status']}",
                "Resolve sales-readiness failures before commercial review.",
            ),
            self._buyer_evidence_item(
                "commercial_readiness",
                "Commercial readiness",
                "Economic buyer",
                ["/api/v1/commercial_readiness/latest", "tests/test_commercial_readiness.py"],
                "measured_local",
                "ready" if commercial["commercial_summary"]["fail"] == 0 else "blocked",
                f"commercial_status={commercial['commercial_status']}",
                "Resolve commercial-readiness failures before buyer review.",
            ),
            self._buyer_evidence_item(
                "analytics_honesty",
                "Analytics honesty",
                "Analytics reviewer",
                ["/api/v1/analytics_snapshots/latest", "docs/analytics_spec.md"],
                "measured_local",
                "ready" if analytics["measurement_status"] == "local_runtime_snapshot" else "blocked",
                analytics["source_note"],
                "Keep measured local evidence separate from production KPI proposals.",
            ),
            self._buyer_evidence_item(
                "access_list_evidence",
                "Access-list evidence",
                "Security and compliance reviewer",
                ["/api/v1/access_reports/{workflow_run_id}", "docs/product_planning.md"],
                "repository_artifact",
                "ready" if has_file("docs/product_planning.md") else "blocked",
                "Workflow trace and access-report evidence are documented.",
                "Restore access-list evidence docs before compliance review.",
            ),
            self._buyer_evidence_item(
                "evaluation_replay",
                "Evaluation replay",
                "Quality reviewer",
                ["/api/v1/evaluation_runs", "docs/screen_design.md"],
                "repository_artifact",
                "ready" if has_file("docs/screen_design.md") else "blocked",
                "Evaluation replay surface is documented.",
                "Restore evaluation replay docs before quality review.",
            ),
            self._buyer_evidence_item(
                "security_posture",
                "Security posture",
                "Security reviewer",
                ["SECURITY.md", "tests/test_security_hardening.py", "CodeQL", "Dependency review", "Trivy"],
                "measured_local",
                "ready" if commercial_rows["security_and_access_control"]["status"] == "pass" else "blocked",
                commercial_rows["security_and_access_control"]["evidence"],
                "Resolve concrete security failures before buyer review.",
            ),
            self._buyer_evidence_item(
                "visual_stakeholder_evidence",
                "Visual stakeholder evidence",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable Figma, FigJam, and Slides artifacts are recorded.",
                "Record editable Figma artifacts before stakeholder review.",
            ),
            self._buyer_evidence_item(
                "buyer_diligence_packet",
                "Buyer diligence packet",
                "Procurement reviewer",
                ["docs/commercial_buyer_diligence_packet.md"],
                "repository_artifact",
                "ready" if has_file("docs/commercial_buyer_diligence_packet.md") else "blocked",
                "Buyer questions map to evidence paths and caveats.",
                "Restore the buyer diligence packet before procurement review.",
            ),
            self._buyer_evidence_item(
                "buyer_acceptance_runbook",
                "Buyer acceptance runbook",
                "Procurement reviewer",
                ["docs/commercial_buyer_acceptance_runbook.md"],
                "repository_artifact",
                "ready" if has_file("docs/commercial_buyer_acceptance_runbook.md") else "blocked",
                "Go, warning, and no-go rules are documented.",
                "Restore acceptance runbook before procurement review.",
            ),
            self._buyer_evidence_item(
                "buyer_evidence_manifest",
                "Buyer evidence manifest",
                "Deal owner",
                ["docs/commercial_buyer_evidence_manifest.md", "/api/v1/commercial_evidence_manifests/latest"],
                "measured_local",
                "ready" if has_file("docs/commercial_buyer_evidence_manifest.md") else "blocked",
                "Buyer evidence is indexed by owner, source, evidence type, and completion state.",
                "Restore the manifest document and endpoint before buyer review.",
            ),
            self._buyer_evidence_item(
                "packaging_decision",
                "Packaging decision",
                "Procurement and security reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                "Single repo and one deployable product remain the current decision.",
                "Document extraction triggers before changing package boundaries.",
            ),
            self._buyer_evidence_item(
                "production_slo_support",
                "Production SLO and support proof",
                "Customer operations reviewer",
                ["production telemetry", "incident drill records", "support ownership"],
                "proposed_until_production",
                "warning",
                "Production SLO, incident, and support evidence require a deployed customer environment.",
                "Collect production telemetry during paid onboarding.",
            ),
            self._buyer_evidence_item(
                "buyer_specific_roi_legal",
                "Buyer-specific ROI and legal proof",
                "Economic buyer and procurement",
                ["ROI model", "legal questionnaire", "data-processing terms", "support plan"],
                "proposed_until_buyer_specific",
                "warning",
                "ROI, legal, procurement, and deployment evidence require a named buyer.",
                "Collect buyer-specific inputs during account diligence.",
            ),
        ]
        summary = self._buyer_manifest_summary(items)
        if summary["by_completion_state"].get("blocked", 0):
            manifest_status = "buyer_review_blocked"
        elif summary["by_completion_state"].get("warning", 0):
            manifest_status = "buyer_review_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            manifest_status = "buyer_review_ready"  # pragma: no cover

        return {
            "manifest_status": manifest_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_buyer_evidence_manifest",
            "source_note": (
                "Buyer evidence manifest combines process-local runtime reports, repository documents, "
                "Figma artifact records, and explicit production or buyer-specific caveats; it is not a "
                "valuation guarantee, purchase commitment, or production compliance certificate."
            ),
            "summary": summary,
            "items": items,
            "related_runtime_reports": {
                "commercial_status": commercial["commercial_status"],
                "sales_readiness_status": commercial["sales_readiness"]["readiness_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
        }

    def commercial_handoff_bundle_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the commercial handoff bundle for sale-readiness evidence."""
        manifest = self.commercial_evidence_manifest_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        manifest_summary = manifest["summary"]["by_completion_state"]
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        runtime_state = "blocked" if manifest_summary.get("blocked", 0) else "ready"
        included_artifacts = [
            self._buyer_evidence_item(
                "runtime_reports",
                "Runtime reports",
                "Deal owner",
                [
                    "/api/v1/sales_readiness/latest",
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_evidence_manifests/latest",
                    "/api/v1/analytics_snapshots/latest",
                ],
                "measured_local",
                runtime_state,
                (
                    f"buyer_manifest_status={manifest['manifest_status']}; "
                    f"commercial_status={manifest['related_runtime_reports']['commercial_status']}"
                ),
                "Resolve runtime report blockers before buyer handoff.",
            ),
            self._buyer_evidence_item(
                "repository_packet",
                "Repository packet",
                "Procurement reviewer",
                [
                    "README.md",
                    "docs/commercial_buyer_diligence_packet.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "docs/commercial_buyer_evidence_manifest.md",
                    "docs/commercial_buyer_handoff_bundle.md",
                ],
                "repository_artifact",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/commercial_buyer_diligence_packet.md",
                        "docs/commercial_buyer_acceptance_runbook.md",
                        "docs/commercial_buyer_evidence_manifest.md",
                        "docs/commercial_buyer_handoff_bundle.md",
                    )
                )
                else "blocked",
                "Buyer-facing diligence, acceptance, manifest, and handoff documents are present.",
                "Restore missing buyer packet documents before procurement review.",
            ),
            self._buyer_evidence_item(
                "figma_stakeholder_artifacts",
                "Figma stakeholder artifacts",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable Figma, FigJam, and Slides artifacts are recorded without Code Connect.",
                "Record editable stakeholder artifacts before buyer handoff.",
            ),
            self._buyer_evidence_item(
                "verification_commands",
                "Verification commands",
                "Technical reviewer",
                [
                    "tests/test_buyer_handoff_bundle.py",
                    "tests/test_buyer_evidence_manifest.py",
                    "tests/test_plugin_driven_artifacts.py",
                    "tests/test_api_contract.py",
                    "pytest -q",
                ],
                "measured_local",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "tests/test_buyer_handoff_bundle.py",
                        "tests/test_buyer_evidence_manifest.py",
                        "tests/test_plugin_driven_artifacts.py",
                        "tests/test_api_contract.py",
                    )
                )
                else "blocked",
                "Focused contract tests and full pytest verification are named for buyer review.",
                "Restore focused tests before technical buyer handoff.",
            ),
            self._buyer_evidence_item(
                "packaging_decision",
                "Packaging decision",
                "Procurement and security reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                "Single repository and one deployable product remain the current decision.",
                "Only extract a library after a second product, independent release cadence, or provenance trigger exists.",
            ),
        ]
        follow_up_items = [
            self._buyer_evidence_item(
                "production_handoff_readiness",
                "Production handoff readiness",
                "Customer operations reviewer",
                ["production SLO", "incident drill", "support rota", "deployment history"],
                "proposed_until_production",
                "warning",
                "Production SLO, incident, deployment, and support evidence require a live customer environment.",
                "Collect production telemetry and support evidence during paid onboarding.",
            ),
            self._buyer_evidence_item(
                "buyer_specific_commercial_close",
                "Buyer-specific commercial close",
                "Economic buyer and legal reviewer",
                ["ROI model", "legal questionnaire", "data-processing terms", "support plan"],
                "proposed_until_buyer_specific",
                "warning",
                "ROI, legal, procurement, and deployment commitments require a named buyer.",
                "Collect buyer-specific inputs during account diligence.",
            ),
        ]
        all_items = included_artifacts + follow_up_items
        summary = self._buyer_manifest_summary(all_items)
        if summary["by_completion_state"].get("blocked", 0):
            bundle_status = "buyer_handoff_blocked"
        elif summary["by_completion_state"].get("warning", 0):
            bundle_status = "buyer_handoff_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            bundle_status = "buyer_handoff_ready"  # pragma: no cover

        return {
            "bundle_status": bundle_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_buyer_handoff_bundle",
            "source_note": (
                "Buyer handoff bundle packages local runtime reports, repository documents, "
                "Figma artifact records, verification commands, and explicit production or "
                "buyer-specific caveats; it is not a valuation guarantee, purchase commitment, "
                "or production compliance certificate."
            ),
            "summary": summary,
            "included_artifacts": included_artifacts,
            "follow_up_items": follow_up_items,
            "acceptance_gates": [
                {
                    "gate_name": "go",
                    "rule": "no blocked included artifacts and concrete security checks have no failure",
                },
                {
                    "gate_name": "warning",
                    "rule": "production or buyer-specific evidence remains proposed and explicitly caveated",
                },
                {
                    "gate_name": "blocked",
                    "rule": "security failure, API contract regression, document mismatch, product defect, or Code Connect usage",
                },
            ],
            "related_runtime_reports": {
                "buyer_manifest_status": manifest["manifest_status"],
                **manifest["related_runtime_reports"],
            },
            "library_split_decision": {
                "decision": "keep_single_product",
                "reason": "No second product, independent release cadence, or security provenance trigger exists.",
                "allowed_future_triggers": [
                    "second product requires core only",
                    "independent release cadence is needed",
                    "buyer security provenance requires package extraction",
                ],
            },
            "plugin_traceability": {
                "figma": "editable stakeholder artifacts and FigJam workflow",
                "product_design": "buyer handoff surface and admin evidence workflow",
                "superpowers": "implementation plan and verification checklist",
                "ponytail": "single-product packaging and no new dependency",
                "data_analytics": "measured versus proposed evidence separation",
            },
        }

    def buyer_evidence_manifest_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the deprecated manifest alias for existing Python consumers."""
        return self.commercial_evidence_manifest_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )

    def buyer_handoff_bundle_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the deprecated handoff alias for existing Python consumers."""
        return self.commercial_handoff_bundle_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )

    def saleability_decision_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the buyer-facing saleability decision for high-value review."""
        handoff = self.commercial_handoff_bundle_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        # Blockers must be hashable, operator-readable identifiers: downstream
        # readiness reports deduplicate inherited blocker lists via
        # ``dict.fromkeys``, which crashes on unhashable evidence-item dicts.
        concrete_blockers = [
            item["item_name"]
            for item in handoff["included_artifacts"]
            if item["completion_state"] == "blocked"
        ]
        warning_conditions = [
            item
            for item in handoff["follow_up_items"]
            if item["completion_state"] == "warning"
        ]
        if concrete_blockers:
            saleability_status = "saleability_blocked"
            decision_label = "Blocked by concrete defect"
        elif warning_conditions:
            saleability_status = "saleability_ready_with_warnings"
            decision_label = "Ready for buyer diligence with explicit warnings"
        else:  # pragma: no cover - unreachable while handoff follow-up warnings are literal report sections
            saleability_status = "saleability_ready"  # pragma: no cover
            decision_label = "Ready for buyer diligence"  # pragma: no cover

        return {
            "saleability_status": saleability_status,
            "decision_label": decision_label,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_saleability_decision",
            "source_note": (
                "Saleability decision is a local buyer due-diligence gate based on runtime "
                "reports, repository documents, Figma artifacts, verification commands, and "
                "explicit caveats; it is not a valuation guarantee, purchase commitment, "
                "or production compliance certificate."
            ),
            "decision_summary": {
                "included_artifact_count": len(handoff["included_artifacts"]),
                "blocked_count": len(concrete_blockers),
                "warning_count": len(warning_conditions),
                "review_process_is_blocker": False,
            },
            "decision_basis": [
                {
                    "basis_name": "buyer_handoff_bundle",
                    "status": handoff["bundle_status"],
                    "source": "/api/v1/commercial_handoff_bundles/latest",
                },
                {
                    "basis_name": "buyer_evidence_manifest",
                    "status": handoff["related_runtime_reports"]["buyer_manifest_status"],
                    "source": "/api/v1/commercial_evidence_manifests/latest",
                },
                {
                    "basis_name": "commercial_readiness",
                    "status": handoff["related_runtime_reports"]["commercial_status"],
                    "source": "/api/v1/commercial_readiness/latest",
                },
                {
                    "basis_name": "sales_readiness",
                    "status": handoff["related_runtime_reports"]["sales_readiness_status"],
                    "source": "/api/v1/sales_readiness/latest",
                },
            ],
            "concrete_blockers": concrete_blockers,
            "warning_conditions": warning_conditions,
            "review_process_policy": {
                "is_blocker": False,
                "non_blocker_examples": [
                    "reviewer delay",
                    "review bot delay",
                    "queued model review",
                    "pending check without concrete failure",
                ],
                "blocker_definition": "concrete security, API contract, document, or product defect",
            },
            "related_runtime_reports": {
                "buyer_handoff_status": handoff["bundle_status"],
                **handoff["related_runtime_reports"],
            },
            "library_split_decision": handoff["library_split_decision"],
            "plugin_traceability": handoff["plugin_traceability"],
        }

    def commercial_evidence_export_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a portable buyer due-diligence export index for commercial review."""
        saleability = self.saleability_decision_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = saleability["concrete_blockers"]
        required_external_evidence = [
            {
                "evidence_name": item["item_name"],
                "label": item["label"],
                "reviewer": item["reviewer"],
                "sources": item["sources"],
                "evidence_type": item["evidence_type"],
                "evidence": item["evidence"],
                "next_action": item["next_action"],
            }
            for item in saleability["warning_conditions"]
        ]
        saleability_state = "blocked" if saleability["saleability_status"] == "saleability_blocked" else "ready"
        export_sections = [
            self._buyer_evidence_item(
                "saleability_decision",
                "Saleability decision",
                "Deal owner",
                ["/api/v1/saleability_decisions/latest", "docs/commercial_saleability_decision.md"],
                "measured_local",
                saleability_state,
                f"saleability_status={saleability['saleability_status']}",
                "Resolve concrete saleability blockers before exporting buyer evidence.",
            ),
            self._buyer_evidence_item(
                "runtime_reports",
                "Runtime reports",
                "Technical reviewer",
                [
                    "/api/v1/sales_readiness/latest",
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_evidence_manifests/latest",
                    "/api/v1/commercial_handoff_bundles/latest",
                    "/api/v1/saleability_decisions/latest",
                    "/api/v1/analytics_snapshots/latest",
                ],
                "measured_local",
                "blocked" if concrete_blockers else "ready",
                (
                    f"buyer_handoff_status={saleability['related_runtime_reports']['buyer_handoff_status']}; "
                    f"buyer_manifest_status={saleability['related_runtime_reports']['buyer_manifest_status']}"
                ),
                "Resolve blocked runtime reports before buyer export.",
            ),
            self._buyer_evidence_item(
                "buyer_packet_documents",
                "Buyer packet documents",
                "Procurement reviewer",
                [
                    "docs/commercial_buyer_diligence_packet.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "docs/commercial_buyer_evidence_manifest.md",
                    "docs/commercial_buyer_handoff_bundle.md",
                    "docs/commercial_saleability_decision.md",
                    "docs/commercial_evidence_export.md",
                ],
                "repository_artifact",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "docs/commercial_buyer_diligence_packet.md",
                        "docs/commercial_buyer_acceptance_runbook.md",
                        "docs/commercial_buyer_evidence_manifest.md",
                        "docs/commercial_buyer_handoff_bundle.md",
                        "docs/commercial_saleability_decision.md",
                        "docs/commercial_evidence_export.md",
                    )
                )
                else "blocked",
                "Buyer diligence, acceptance, manifest, handoff, decision, and export documents are present.",
                "Restore missing buyer packet documents before export.",
            ),
            self._buyer_evidence_item(
                "figma_stakeholder_artifacts",
                "Figma stakeholder artifacts",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable stakeholder artifacts are recorded and Code Connect is excluded.",
                "Record Figma artifacts before exporting buyer evidence.",
            ),
            self._buyer_evidence_item(
                "verification_commands",
                "Verification commands",
                "Technical reviewer",
                [
                    "tests/test_commercial_evidence_export.py",
                    "tests/test_saleability_decision.py",
                    "tests/test_plugin_driven_artifacts.py",
                    "tests/test_api_contract.py",
                    "pytest -q",
                ],
                "measured_local",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "tests/test_commercial_evidence_export.py",
                        "tests/test_saleability_decision.py",
                        "tests/test_plugin_driven_artifacts.py",
                        "tests/test_api_contract.py",
                    )
                )
                else "blocked",
                "Focused commercial export, saleability, plugin artifact, and API contract tests are named.",
                "Restore focused tests before buyer export.",
            ),
            self._buyer_evidence_item(
                "review_process_policy",
                "Review process policy",
                "Deal owner",
                ["docs/commercial_saleability_decision.md", "/api/v1/saleability_decisions/latest"],
                "repository_artifact",
                "ready",
                "Reviewer delay, review bot delay, and queued model review are not concrete blockers.",
                "Escalate only concrete security, API contract, document, or product defects.",
            ),
            self._buyer_evidence_item(
                "packaging_decision",
                "Packaging decision",
                "Procurement and security reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                saleability["library_split_decision"]["reason"],
                "Only extract a library after a second product, independent release cadence, or provenance trigger exists.",
            ),
        ]
        export_section_summary = self._buyer_manifest_summary(export_sections)
        blocked_count = export_section_summary["by_completion_state"]["blocked"] + len(concrete_blockers)
        warning_count = len(required_external_evidence)
        if blocked_count:
            export_status = "commercial_export_blocked"
        elif warning_count:
            export_status = "commercial_export_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            export_status = "commercial_export_ready"  # pragma: no cover

        return {
            "export_status": export_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_evidence_export",
            "source_note": (
                "Commercial evidence export packages local runtime decisions, repository documents, "
                "Figma artifact records, verification commands, review-process policy, packaging decision, "
                "and explicit production or buyer-specific evidence gaps; it is not a valuation guarantee, "
                "purchase commitment, or production compliance certificate."
            ),
            "export_summary": {
                "section_count": len(export_sections),
                "blocked_count": blocked_count,
                "warning_count": warning_count,
                "review_process_is_blocker": saleability["review_process_policy"]["is_blocker"],
            },
            "export_sections": export_sections,
            "required_external_evidence": required_external_evidence,
            "concrete_blockers": concrete_blockers,
            "review_process_policy": saleability["review_process_policy"],
            "related_runtime_reports": {
                "saleability_status": saleability["saleability_status"],
                **saleability["related_runtime_reports"],
            },
            "library_split_decision": saleability["library_split_decision"],
            "plugin_traceability": saleability["plugin_traceability"],
            "export_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_evidence_exports/latest",
                "documentation": "docs/commercial_evidence_export.md",
            },
        }

    def commercial_acceptance_check_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the buyer acceptance check over the commercial evidence export."""
        evidence_export = self.commercial_evidence_export_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = evidence_export["concrete_blockers"]
        export_blocked = evidence_export["export_status"] == "commercial_export_blocked"
        runtime_state = "blocked" if export_blocked or concrete_blockers else "ready"
        acceptance_items = [
            self._buyer_evidence_item(
                "runtime_endpoint_chain",
                "Runtime endpoint chain",
                "Technical reviewer",
                [
                    "/api/v1/analytics_snapshots/latest",
                    "/api/v1/sales_readiness/latest",
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_evidence_manifests/latest",
                    "/api/v1/commercial_handoff_bundles/latest",
                    "/api/v1/saleability_decisions/latest",
                    "/api/v1/commercial_evidence_exports/latest",
                ],
                "measured_local",
                runtime_state,
                f"commercial_export_status={evidence_export['export_status']}",
                "Resolve blocked runtime report chain before buyer acceptance.",
            ),
            self._buyer_evidence_item(
                "buyer_packet_documents",
                "Buyer packet documents",
                "Procurement reviewer",
                [
                    "docs/commercial_buyer_diligence_packet.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "docs/commercial_buyer_evidence_manifest.md",
                    "docs/commercial_buyer_handoff_bundle.md",
                    "docs/commercial_saleability_decision.md",
                    "docs/commercial_evidence_export.md",
                    "docs/commercial_acceptance_check.md",
                ],
                "repository_artifact",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "docs/commercial_buyer_diligence_packet.md",
                        "docs/commercial_buyer_acceptance_runbook.md",
                        "docs/commercial_buyer_evidence_manifest.md",
                        "docs/commercial_buyer_handoff_bundle.md",
                        "docs/commercial_saleability_decision.md",
                        "docs/commercial_evidence_export.md",
                        "docs/commercial_acceptance_check.md",
                    )
                )
                else "blocked",
                "Buyer packet documents cover diligence, acceptance, manifest, handoff, decision, export, and check.",
                "Restore missing buyer packet documents before buyer acceptance.",
            ),
            self._buyer_evidence_item(
                "admin_operator_surface",
                "Admin operator surface",
                "Platform operator",
                ["/admin", "contextual_orchestrator/admin.py", "/api/v1/commercial_acceptance_checks/latest"],
                "repository_artifact",
                "ready" if has_file("contextual_orchestrator/admin.py") else "blocked",
                "Admin observability surface exposes the commercial acceptance check status with bilingual labels.",
                "Expose acceptance check status in admin observability before buyer acceptance.",
            ),
            self._buyer_evidence_item(
                "verification_evidence",
                "Verification evidence",
                "Technical reviewer",
                [
                    "tests/test_commercial_acceptance_check.py",
                    "tests/test_commercial_evidence_export.py",
                    "tests/test_saleability_decision.py",
                    "tests/test_plugin_driven_artifacts.py",
                    "tests/test_api_contract.py",
                    "pytest -q",
                ],
                "measured_local",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "tests/test_commercial_acceptance_check.py",
                        "tests/test_commercial_evidence_export.py",
                        "tests/test_saleability_decision.py",
                        "tests/test_plugin_driven_artifacts.py",
                        "tests/test_api_contract.py",
                    )
                )
                else "blocked",
                "Focused commercial acceptance, export, saleability, plugin artifact, and API contract tests are named.",
                "Restore focused tests before buyer acceptance.",
            ),
            self._buyer_evidence_item(
                "figma_stakeholder_artifacts",
                "Figma stakeholder artifacts",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable stakeholder artifacts are recorded and Code Connect is excluded.",
                "Record editable Figma artifacts before buyer acceptance.",
            ),
            self._buyer_evidence_item(
                "review_process_policy",
                "Review process policy",
                "Deal owner",
                ["docs/commercial_saleability_decision.md", "/api/v1/saleability_decisions/latest"],
                "repository_artifact",
                "ready",
                "Reviewer delay, review bot delay, queued model review, and pending checks without concrete failure are not blockers.",
                "Block only on concrete security, API contract, document, or product defects.",
            ),
            self._buyer_evidence_item(
                "packaging_decision",
                "Packaging decision",
                "Procurement and security reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                evidence_export["library_split_decision"]["reason"],
                "Only extract a library after a second product, independent release cadence, or provenance trigger exists.",
            ),
        ]
        follow_up_items = [
            self._buyer_evidence_item(
                item["evidence_name"],
                item["label"],
                item["reviewer"],
                item["sources"],
                item["evidence_type"],
                "warning",
                item["evidence"],
                item["next_action"],
            )
            for item in evidence_export["required_external_evidence"]
        ]
        all_items = acceptance_items + follow_up_items
        summary = self._buyer_manifest_summary(all_items)
        blocked_count = summary["by_completion_state"]["blocked"] + len(concrete_blockers)
        warning_count = summary["by_completion_state"]["warning"]
        if blocked_count:
            acceptance_status = "commercial_acceptance_blocked"
        elif warning_count:
            acceptance_status = "commercial_acceptance_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            acceptance_status = "commercial_acceptance_ready"  # pragma: no cover

        return {
            "acceptance_status": acceptance_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_acceptance_check",
            "source_note": (
                "Commercial acceptance check evaluates local commercial evidence export, admin visibility, "
                "repository packet, Figma artifacts, verification commands, review-process policy, packaging "
                "decision, and explicit production or buyer-specific gaps; it is not a valuation guarantee, "
                "purchase commitment, or production compliance certificate."
            ),
            "acceptance_summary": {
                "item_count": len(all_items),
                "blocked_count": blocked_count,
                "warning_count": warning_count,
                "review_process_is_blocker": evidence_export["review_process_policy"]["is_blocker"],
            },
            "acceptance_items": acceptance_items,
            "follow_up_items": follow_up_items,
            "concrete_blockers": concrete_blockers,
            "required_external_evidence": evidence_export["required_external_evidence"],
            "acceptance_gates": [
                {
                    "gate_name": "go",
                    "rule": "no blocked acceptance items and no required external evidence gaps",
                },
                {
                    "gate_name": "warning",
                    "rule": "only production or buyer-specific evidence remains explicitly caveated",
                },
                {
                    "gate_name": "blocked",
                    "rule": "security failure, API contract regression, document mismatch, product defect, or Code Connect usage",
                },
            ],
            "review_process_policy": evidence_export["review_process_policy"],
            "related_runtime_reports": {
                "commercial_export_status": evidence_export["export_status"],
                **evidence_export["related_runtime_reports"],
            },
            "library_split_decision": evidence_export["library_split_decision"],
            "plugin_traceability": evidence_export["plugin_traceability"],
            "acceptance_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_acceptance_checks/latest",
                "documentation": "docs/commercial_acceptance_check.md",
            },
        }

    def commercial_release_candidate_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the local buyer-facing commercial release-candidate manifest."""
        acceptance = self.commercial_acceptance_check_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = acceptance["concrete_blockers"]
        acceptance_blocked = acceptance["acceptance_status"] == "commercial_acceptance_blocked"
        runtime_state = "blocked" if acceptance_blocked or concrete_blockers else "ready"
        release_artifacts = [
            self._buyer_evidence_item(
                "commercial_acceptance_check",
                "Commercial acceptance check",
                "Deal owner",
                ["/api/v1/commercial_acceptance_checks/latest", "docs/commercial_acceptance_check.md"],
                "measured_local",
                runtime_state,
                f"acceptance_status={acceptance['acceptance_status']}",
                "Resolve blocked acceptance checks before tagging a release candidate.",
            ),
            self._buyer_evidence_item(
                "runtime_endpoint_chain",
                "Runtime endpoint chain",
                "Technical reviewer",
                [
                    "/api/v1/analytics_snapshots/latest",
                    "/api/v1/sales_readiness/latest",
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_evidence_manifests/latest",
                    "/api/v1/commercial_handoff_bundles/latest",
                    "/api/v1/saleability_decisions/latest",
                    "/api/v1/commercial_evidence_exports/latest",
                    "/api/v1/commercial_acceptance_checks/latest",
                    "/api/v1/commercial_release_candidates/latest",
                ],
                "measured_local",
                runtime_state,
                "Commercial release candidate endpoint is chained after acceptance, export, decision, handoff, manifest, readiness, and analytics reports.",
                "Restore blocked runtime endpoint evidence before release-candidate handoff.",
            ),
            self._buyer_evidence_item(
                "repository_distribution_packet",
                "Repository distribution packet",
                "Procurement reviewer",
                [
                    "README.md",
                    "docs/rest_api_design.md",
                    "docs/commercial_buyer_diligence_packet.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "docs/commercial_buyer_evidence_manifest.md",
                    "docs/commercial_buyer_handoff_bundle.md",
                    "docs/commercial_saleability_decision.md",
                    "docs/commercial_evidence_export.md",
                    "docs/commercial_acceptance_check.md",
                    "docs/commercial_release_candidate.md",
                ],
                "repository_artifact",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/rest_api_design.md",
                        "docs/commercial_buyer_diligence_packet.md",
                        "docs/commercial_buyer_acceptance_runbook.md",
                        "docs/commercial_buyer_evidence_manifest.md",
                        "docs/commercial_buyer_handoff_bundle.md",
                        "docs/commercial_saleability_decision.md",
                        "docs/commercial_evidence_export.md",
                        "docs/commercial_acceptance_check.md",
                        "docs/commercial_release_candidate.md",
                    )
                )
                else "blocked",
                "Repository packet contains the README, REST API contract notes, and commercial buyer documents.",
                "Restore missing distribution documents before buyer release-candidate review.",
            ),
            self._buyer_evidence_item(
                "security_package_metadata",
                "Security and package metadata",
                "Security reviewer",
                [
                    "LICENSE",
                    "SECURITY.md",
                    "pyproject.toml",
                    "requirements.lock",
                    ".github/workflows/security.yml",
                    ".github/dependabot.yml",
                    "ContextualWisdomLab/.github central required security workflows",
                ],
                "repository_artifact",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "LICENSE",
                        "SECURITY.md",
                        "pyproject.toml",
                        "requirements.lock",
                        ".github/workflows/security.yml",
                        ".github/dependabot.yml",
                    )
                )
                else "blocked",
                "License, security policy, package metadata, locked requirements, local supply-chain workflow, Dependabot metadata, and central required security workflows are present.",
                "Restore missing security or package metadata before release-candidate handoff.",
            ),
            self._buyer_evidence_item(
                "admin_operator_surface",
                "Admin operator surface",
                "Platform operator",
                ["/admin", "contextual_orchestrator/admin.py", "/api/v1/commercial_release_candidates/latest"],
                "repository_artifact",
                "ready" if has_file("contextual_orchestrator/admin.py") else "blocked",
                "Admin observability surface exposes the release-candidate status with bilingual labels.",
                "Expose release-candidate status in admin observability before buyer handoff.",
            ),
            self._buyer_evidence_item(
                "verification_evidence",
                "Verification evidence",
                "Technical reviewer",
                [
                    "tests/test_commercial_release_candidate.py",
                    "tests/test_commercial_acceptance_check.py",
                    "tests/test_commercial_evidence_export.py",
                    "tests/test_saleability_decision.py",
                    "tests/test_plugin_driven_artifacts.py",
                    "tests/test_api_contract.py",
                    "pytest -q",
                ],
                "measured_local",
                "ready"
                if all(
                    has_file(path)
                    for path in (
                        "tests/test_commercial_release_candidate.py",
                        "tests/test_commercial_acceptance_check.py",
                        "tests/test_commercial_evidence_export.py",
                        "tests/test_saleability_decision.py",
                        "tests/test_plugin_driven_artifacts.py",
                        "tests/test_api_contract.py",
                    )
                )
                else "blocked",
                "Focused release-candidate, acceptance, export, saleability, plugin artifact, and API contract tests are named.",
                "Restore focused verification before release-candidate handoff.",
            ),
            self._buyer_evidence_item(
                "figma_stakeholder_artifacts",
                "Figma stakeholder artifacts",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable stakeholder artifacts are recorded and Code Connect is excluded.",
                "Record editable Figma artifacts before buyer release-candidate review.",
            ),
            self._buyer_evidence_item(
                "review_process_policy",
                "Review process policy",
                "Deal owner",
                ["docs/commercial_saleability_decision.md", "docs/commercial_release_candidate.md"],
                "repository_artifact",
                "ready",
                "Reviewer delay, review bot delay, queued model review, and pending checks without concrete failure are not blockers.",
                "Block only on concrete security, API contract, document, or product defects.",
            ),
            self._buyer_evidence_item(
                "packaging_decision",
                "Packaging decision",
                "Procurement and security reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                acceptance["library_split_decision"]["reason"],
                "Only extract a library after a second product, independent release cadence, or provenance trigger exists.",
            ),
        ]
        external_release_gaps = [
            self._buyer_evidence_item(
                item["item_name"],
                item["label"],
                item["reviewer"],
                item["sources"],
                item["evidence_type"],
                "warning",
                item["evidence"],
                item["next_action"],
            )
            for item in acceptance["follow_up_items"]
        ]
        summary = self._buyer_manifest_summary(release_artifacts + external_release_gaps)
        blocked_count = summary["by_completion_state"]["blocked"] + len(concrete_blockers)
        warning_count = summary["by_completion_state"]["warning"]
        if blocked_count:
            release_status = "commercial_release_blocked"
        elif warning_count:
            release_status = "commercial_release_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            release_status = "commercial_release_ready"  # pragma: no cover

        return {
            "release_status": release_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_release_candidate",
            "source_note": (
                "Commercial release candidate packages local acceptance, runtime endpoints, repository "
                "distribution documents, security metadata, admin visibility, verification commands, "
                "Figma artifact records, review-process policy, packaging decision, and explicit external "
                "release gaps; it is not a valuation guarantee, purchase commitment, or production "
                "compliance certificate."
            ),
            "release_summary": {
                "artifact_count": len(release_artifacts),
                "blocked_count": blocked_count,
                "warning_count": warning_count,
                "review_process_is_blocker": acceptance["review_process_policy"]["is_blocker"],
            },
            "release_artifacts": release_artifacts,
            "external_release_gaps": external_release_gaps,
            "concrete_blockers": concrete_blockers,
            "release_gates": [
                {
                    "gate_name": "package",
                    "rule": "runtime endpoint chain, repository packet, security metadata, admin surface, tests, Figma artifacts, review policy, and packaging decision are present",
                },
                {
                    "gate_name": "warning",
                    "rule": "only production or buyer-specific external evidence remains explicitly caveated",
                },
                {
                    "gate_name": "blocked",
                    "rule": "security failure, API contract regression, missing distribution artifact, document mismatch, product defect, or Code Connect usage",
                },
            ],
            "review_process_policy": acceptance["review_process_policy"],
            "related_runtime_reports": {
                "commercial_acceptance_status": acceptance["acceptance_status"],
                **acceptance["related_runtime_reports"],
            },
            "library_split_decision": acceptance["library_split_decision"],
            "plugin_traceability": acceptance["plugin_traceability"],
            "release_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_release_candidates/latest",
                "documentation": "docs/commercial_release_candidate.md",
            },
        }

    def commercial_gap_register_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an owner/action register for commercial release-candidate gaps."""
        release = self.commercial_release_candidate_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        concrete_blockers = release["concrete_blockers"]
        release_blocked = release["release_status"] == "commercial_release_blocked"
        gap_items = []
        for item in release["external_release_gaps"]:
            source_type = item["evidence_type"]
            if source_type == "proposed_until_production":
                gap_status = "production_input_required"
                gap_type = "production_evidence_gap"
                owner = "Operations and support owner"
            else:
                gap_status = "buyer_input_required"
                gap_type = "buyer_specific_gap"
                owner = "Buyer and deal owner"
            gap_items.append({
                "gap_name": item["item_name"],
                "label": item["label"],
                "gap_type": gap_type,
                "gap_status": gap_status,
                "owner": owner,
                "reviewer": item["reviewer"],
                "sources": item["sources"],
                "source_evidence_type": source_type,
                "current_evidence": item["evidence"],
                "required_input": item["next_action"],
                "is_blocker": False,
            })

        blocked_count = len(concrete_blockers) + (1 if release_blocked else 0)
        if blocked_count:
            gap_register_status = "commercial_gap_register_blocked"
        elif gap_items:
            gap_register_status = "commercial_gap_register_open"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            gap_register_status = "commercial_gap_register_clear"  # pragma: no cover

        production_gap_count = sum(1 for item in gap_items if item["gap_type"] == "production_evidence_gap")
        buyer_specific_gap_count = sum(1 for item in gap_items if item["gap_type"] == "buyer_specific_gap")
        return {
            "gap_register_status": gap_register_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_gap_register",
            "source_note": (
                "Commercial gap register converts local release-candidate warning gaps into owner, action, "
                "source, and required-input rows for buyer due diligence; it is not a valuation guarantee, "
                "purchase commitment, or production compliance certificate."
            ),
            "gap_summary": {
                "total_gap_count": len(gap_items),
                "production_gap_count": production_gap_count,
                "buyer_specific_gap_count": buyer_specific_gap_count,
                "blocked_count": blocked_count,
                "review_process_is_blocker": release["review_process_policy"]["is_blocker"],
            },
            "gap_items": gap_items,
            "concrete_blockers": concrete_blockers,
            "gap_status_rules": [
                {
                    "gap_status": "production_input_required",
                    "rule": "production deployment, support, SLO, or operational evidence must be supplied before production claim",
                },
                {
                    "gap_status": "buyer_input_required",
                    "rule": "buyer-specific legal, procurement, ROI, or deployment context must be supplied before buyer-specific claim",
                },
                {
                    "gap_status": "blocked",
                    "rule": "concrete security, API contract, document, product defect, or Code Connect usage blocks commercial release",
                },
            ],
            "review_process_policy": release["review_process_policy"],
            "related_runtime_reports": {
                "commercial_release_status": release["release_status"],
                **release["related_runtime_reports"],
            },
            "library_split_decision": release["library_split_decision"],
            "plugin_traceability": release["plugin_traceability"],
            "gap_register_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_gap_registers/latest",
                "documentation": "docs/commercial_gap_register.md",
            },
        }

    def commercial_procurement_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a procurement/legal readiness gate over commercial evidence."""
        gap_register = self.commercial_gap_register_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        gap_by_status = {item["gap_status"]: item for item in gap_register["gap_items"]}
        production_gap = gap_by_status.get("production_input_required")
        buyer_gap = gap_by_status.get("buyer_input_required")
        concrete_blockers = gap_register["concrete_blockers"]
        procurement_items = [
            {
                "item_name": "license_and_rights",
                "label": "License and rights",
                "owner": "Procurement reviewer",
                "sources": ["LICENSE", "pyproject.toml"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready" if has_file("LICENSE") and has_file("pyproject.toml") else "blocked",
                "evidence": "MIT license and package metadata are present for buyer rights review.",
                "required_input": "Restore license or package metadata before procurement review.",
            },
            {
                "item_name": "security_package_metadata",
                "label": "Security package metadata",
                "owner": "Security reviewer",
                "sources": [
                    "SECURITY.md",
                    "requirements.lock",
                    ".github/workflows/security.yml",
                    ".github/dependabot.yml",
                    "ContextualWisdomLab/.github central required security workflows",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "SECURITY.md",
                        "requirements.lock",
                        ".github/workflows/security.yml",
                        ".github/dependabot.yml",
                    )
                )
                else "blocked",
                "evidence": "Security policy, locked dependencies, local supply-chain workflow, Dependabot metadata, and central required security workflows are present.",
                "required_input": "Restore missing security metadata before procurement review.",
            },
            {
                "item_name": "distribution_packet",
                "label": "Distribution packet",
                "owner": "Deal owner",
                "sources": [
                    "README.md",
                    "docs/rest_api_design.md",
                    "docs/commercial_release_candidate.md",
                    "docs/commercial_gap_register.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/rest_api_design.md",
                        "docs/commercial_release_candidate.md",
                        "docs/commercial_gap_register.md",
                    )
                )
                else "blocked",
                "evidence": "Repository overview, REST contract, release candidate, and gap register documents are present.",
                "required_input": "Restore missing distribution documents before procurement review.",
            },
            {
                "item_name": "admin_evidence_surface",
                "label": "Admin evidence surface",
                "owner": "Platform operator",
                "sources": ["/admin", "contextual_orchestrator/admin.py", "/api/v1/commercial_procurement_readiness/latest"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready" if has_file("contextual_orchestrator/admin.py") else "blocked",
                "evidence": "Admin observability surface exposes procurement readiness with bilingual labels.",
                "required_input": "Expose procurement readiness in admin observability before buyer review.",
            },
            {
                "item_name": "production_support_slo_input",
                "label": "Production support and SLO input",
                "owner": production_gap["owner"] if production_gap else "Operations and support owner",
                "sources": production_gap["sources"] if production_gap else ["docs/commercial_gap_register.md"],
                "evidence_type": "proposed_until_production",
                "completion_state": "warning" if production_gap else "ready",
                "source_gap_status": production_gap["gap_status"] if production_gap else "resolved",
                "evidence": production_gap["current_evidence"] if production_gap else "No production evidence gap is open.",
                "required_input": production_gap["required_input"] if production_gap else "No production input required.",
            },
            {
                "item_name": "buyer_legal_roi_procurement_input",
                "label": "Buyer legal, ROI, and procurement input",
                "owner": buyer_gap["owner"] if buyer_gap else "Buyer and deal owner",
                "sources": buyer_gap["sources"] if buyer_gap else ["docs/commercial_gap_register.md"],
                "evidence_type": "proposed_until_buyer_specific",
                "completion_state": "warning" if buyer_gap else "ready",
                "source_gap_status": buyer_gap["gap_status"] if buyer_gap else "resolved",
                "evidence": buyer_gap["current_evidence"] if buyer_gap else "No buyer-specific evidence gap is open.",
                "required_input": buyer_gap["required_input"] if buyer_gap else "No buyer input required.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_saleability_decision.md", "docs/commercial_procurement_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Reviewer delay, review bot delay, queued model review, and pending checks without concrete failure are not blockers.",
                "required_input": "Block only on concrete security, API contract, document, or product defects.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": "Procurement and security reviewer",
                "sources": ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready" if has_file("docs/library_research.md") and has_file("docs/commercial_plugin_operating_model.md") else "blocked",
                "evidence": gap_register["library_split_decision"]["reason"],
                "required_input": "Only extract a library after a second product, independent release cadence, or provenance trigger exists.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in procurement_items)
        production_gap_count = 1 if production_gap else 0
        buyer_specific_gap_count = 1 if buyer_gap else 0
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            procurement_status = "commercial_procurement_blocked"
        elif warning_count:
            procurement_status = "commercial_procurement_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            procurement_status = "commercial_procurement_ready"  # pragma: no cover

        return {
            "procurement_status": procurement_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_procurement_readiness",
            "source_note": (
                "Commercial procurement readiness packages local license, security, distribution, admin, "
                "gap-register, review-process, and packaging evidence for buyer due diligence; it is not "
                "a valuation guarantee, purchase commitment, or production compliance certificate."
            ),
            "procurement_summary": {
                "item_count": len(procurement_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "production_gap_count": production_gap_count,
                "buyer_specific_gap_count": buyer_specific_gap_count,
                "review_process_is_blocker": gap_register["review_process_policy"]["is_blocker"],
            },
            "procurement_items": procurement_items,
            "concrete_blockers": concrete_blockers,
            "procurement_status_rules": [
                {
                    "procurement_status": "commercial_procurement_ready",
                    "rule": "license, security, distribution, admin, support, legal, ROI, review, and packaging evidence are ready",
                },
                {
                    "procurement_status": "commercial_procurement_ready_with_warnings",
                    "rule": "local packet is ready while production or buyer-specific inputs remain explicit warnings",
                },
                {
                    "procurement_status": "commercial_procurement_blocked",
                    "rule": "missing packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks procurement",
                },
            ],
            "review_process_policy": gap_register["review_process_policy"],
            "related_runtime_reports": {
                "commercial_gap_register_status": gap_register["gap_register_status"],
                **gap_register["related_runtime_reports"],
            },
            "library_split_decision": gap_register["library_split_decision"],
            "plugin_traceability": gap_register["plugin_traceability"],
            "procurement_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_procurement_readiness/latest",
                "documentation": "docs/commercial_procurement_readiness.md",
            },
        }

    def commercial_contract_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a contract-readiness gate over procurement evidence."""
        procurement = self.commercial_procurement_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        procurement_by_name = {item["item_name"]: item for item in procurement["procurement_items"]}
        license_item = procurement_by_name["license_and_rights"]
        security_item = procurement_by_name["security_package_metadata"]
        support_item = procurement_by_name["production_support_slo_input"]
        buyer_item = procurement_by_name["buyer_legal_roi_procurement_input"]
        packaging_item = procurement_by_name["packaging_decision"]
        concrete_blockers = procurement["concrete_blockers"]
        support_slo_gap_count = 1 if support_item["completion_state"] == "warning" else 0
        buyer_order_form_gap_count = 1 if buyer_item["completion_state"] == "warning" else 0
        contract_items = [
            {
                "item_name": "license_commercial_rights",
                "label": "License and commercial rights terms",
                "owner": "Legal reviewer",
                "sources": license_item["sources"],
                "evidence_type": license_item["evidence_type"],
                "completion_state": license_item["completion_state"],
                "evidence": license_item["evidence"],
                "required_input": license_item["required_input"],
            },
            {
                "item_name": "security_privacy_terms",
                "label": "Security and privacy terms",
                "owner": "Security and legal reviewer",
                "sources": [*security_item["sources"], "docs/commercial_procurement_readiness.md"],
                "evidence_type": security_item["evidence_type"],
                "completion_state": security_item["completion_state"],
                "evidence": (
                    f"{security_item['evidence']} Runtime readiness profile uses "
                    f"auth_mode={security_profile.get('auth_mode', 'unknown') if security_profile else 'unknown'}, "
                    f"public_bind={security_profile.get('allow_public_bind', 'unknown') if security_profile else 'unknown'}, "
                    "and trace exposure controls."
                ),
                "required_input": security_item["required_input"],
            },
            {
                "item_name": "audit_export_obligations",
                "label": "Audit and export obligations",
                "owner": "Compliance reviewer",
                "sources": [
                    "/api/v1/commercial_evidence_exports/latest",
                    "docs/commercial_evidence_export.md",
                    "docs/rest_api_design.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "docs/commercial_evidence_export.md",
                        "docs/rest_api_design.md",
                    )
                )
                else "blocked",
                "evidence": "Commercial evidence export and REST API contract describe buyer-readable audit evidence.",
                "required_input": "Restore evidence export docs and REST contract before contract review.",
            },
            {
                "item_name": "contract_packet_docs",
                "label": "Contract packet documents",
                "owner": "Deal owner",
                "sources": [
                    "README.md",
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_saleability_decision.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/commercial_contract_readiness.md",
                        "docs/commercial_procurement_readiness.md",
                        "docs/commercial_saleability_decision.md",
                    )
                )
                else "blocked",
                "evidence": "Contract packet, procurement gate, and saleability blocker policy are documented.",
                "required_input": "Restore buyer contract packet docs before legal review.",
            },
            {
                "item_name": "support_slo_terms",
                "label": "Support and SLO terms",
                "owner": support_item["owner"],
                "sources": support_item["sources"],
                "evidence_type": support_item["evidence_type"],
                "completion_state": support_item["completion_state"],
                "source_gap_status": support_item.get("source_gap_status", "resolved"),
                "evidence": support_item["evidence"],
                "required_input": support_item["required_input"],
            },
            {
                "item_name": "buyer_order_form_input",
                "label": "Buyer order-form input",
                "owner": buyer_item["owner"],
                "sources": buyer_item["sources"],
                "evidence_type": buyer_item["evidence_type"],
                "completion_state": buyer_item["completion_state"],
                "source_gap_status": buyer_item.get("source_gap_status", "resolved"),
                "evidence": buyer_item["evidence"],
                "required_input": buyer_item["required_input"],
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_saleability_decision.md", "docs/commercial_contract_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review process delay is not a contract blocker unless a concrete failure is produced.",
                "required_input": "Block only on concrete security, API contract, document, or product defects.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": packaging_item["owner"],
                "sources": packaging_item["sources"],
                "evidence_type": packaging_item["evidence_type"],
                "completion_state": packaging_item["completion_state"],
                "evidence": packaging_item["evidence"],
                "required_input": packaging_item["required_input"],
            },
        ]
        state_counts = Counter(item["completion_state"] for item in contract_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            contract_status = "commercial_contract_blocked"
        elif warning_count:
            contract_status = "commercial_contract_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            contract_status = "commercial_contract_ready"  # pragma: no cover

        return {
            "contract_status": contract_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_contract_readiness",
            "source_note": (
                "Commercial contract readiness packages local license, security/privacy, audit export, "
                "support/SLO, buyer order-form, review-process, and packaging evidence for legal and "
                "procurement due diligence; it is not a valuation guarantee, purchase commitment, or "
                "production compliance certificate."
            ),
            "contract_summary": {
                "item_count": len(contract_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "support_slo_gap_count": support_slo_gap_count,
                "buyer_order_form_gap_count": buyer_order_form_gap_count,
                "review_process_is_blocker": procurement["review_process_policy"]["is_blocker"],
            },
            "contract_items": contract_items,
            "concrete_blockers": concrete_blockers,
            "contract_status_rules": [
                {
                    "contract_status": "commercial_contract_ready",
                    "rule": "license, security/privacy, audit/export, support/SLO, buyer order-form, review, and packaging terms are ready",
                },
                {
                    "contract_status": "commercial_contract_ready_with_warnings",
                    "rule": "local contract packet is ready while production support/SLO or buyer order-form inputs remain explicit warnings",
                },
                {
                    "contract_status": "commercial_contract_blocked",
                    "rule": "missing contract packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks contract readiness",
                },
            ],
            "review_process_policy": procurement["review_process_policy"],
            "related_runtime_reports": {
                "commercial_procurement_status": procurement["procurement_status"],
                **procurement["related_runtime_reports"],
            },
            "library_split_decision": procurement["library_split_decision"],
            "plugin_traceability": procurement["plugin_traceability"],
            "contract_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_contract_readiness/latest",
                "documentation": "docs/commercial_contract_readiness.md",
            },
        }

    def commercial_onboarding_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a paid-onboarding readiness gate over contract evidence."""
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        contract_by_name = {item["item_name"]: item for item in contract["contract_items"]}
        support_item = contract_by_name["support_slo_terms"]
        buyer_item = contract_by_name["buyer_order_form_input"]
        packaging_item = contract_by_name["packaging_decision"]
        concrete_blockers = contract["concrete_blockers"]
        support_slo_action_count = 1 if support_item["completion_state"] == "warning" else 0
        buyer_input_action_count = 1 if buyer_item["completion_state"] == "warning" else 0
        onboarding_items = [
            {
                "item_name": "buyer_kickoff_packet",
                "label": "Buyer kickoff packet",
                "owner": "Deal owner",
                "sources": [
                    "README.md",
                    "docs/commercial_onboarding_readiness.md",
                    "docs/commercial_contract_readiness.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/commercial_onboarding_readiness.md",
                        "docs/commercial_contract_readiness.md",
                    )
                )
                else "blocked",
                "evidence": "Buyer kickoff packet connects product overview, contract readiness, and onboarding plan.",
                "action": "Use the packet to start paid onboarding with named buyer stakeholders.",
                "exit_criteria": "Buyer confirms kickoff owner, onboarding dates, and evidence review cadence.",
            },
            {
                "item_name": "support_slo_kickoff",
                "label": "Support and SLO kickoff",
                "owner": support_item["owner"],
                "sources": support_item["sources"],
                "evidence_type": support_item["evidence_type"],
                "completion_state": support_item["completion_state"],
                "source_gap_status": support_item.get("source_gap_status", "resolved"),
                "evidence": support_item["evidence"],
                "action": "Collect support rota, escalation path, SLO target, and incident drill evidence during paid onboarding.",
                "exit_criteria": "Buyer and operator approve support owner, response target, escalation path, and first incident drill record.",
            },
            {
                "item_name": "buyer_order_form_kickoff",
                "label": "Buyer order-form kickoff",
                "owner": buyer_item["owner"],
                "sources": buyer_item["sources"],
                "evidence_type": buyer_item["evidence_type"],
                "completion_state": buyer_item["completion_state"],
                "source_gap_status": buyer_item.get("source_gap_status", "resolved"),
                "evidence": buyer_item["evidence"],
                "action": "Collect buyer order-form, ROI, legal questionnaire, deployment, and support inputs.",
                "exit_criteria": "Buyer-specific order form and legal/procurement inputs are attached to the diligence packet.",
            },
            {
                "item_name": "telemetry_capture_plan",
                "label": "Telemetry capture plan",
                "owner": "Data analytics owner",
                "sources": ["/api/v1/analytics_snapshots/latest", "docs/analytics_spec.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready" if has_file("docs/analytics_spec.md") else "blocked",
                "evidence": "Analytics spec separates measured local evidence from proposed production metrics.",
                "action": "Capture production onboarding telemetry without mixing it with local prototype metrics.",
                "exit_criteria": "First buyer environment records adoption, latency, verification, trace completeness, and support events.",
            },
            {
                "item_name": "acceptance_exit_criteria",
                "label": "Acceptance exit criteria",
                "owner": "Technical buyer reviewer",
                "sources": [
                    "/api/v1/commercial_acceptance_checks/latest",
                    "docs/commercial_acceptance_check.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if has_file("docs/commercial_acceptance_check.md")
                and has_file("docs/commercial_buyer_acceptance_runbook.md")
                else "blocked",
                "evidence": "Acceptance check and buyer runbook define go/no-go review gates.",
                "action": "Run the buyer acceptance checklist after kickoff evidence is attached.",
                "exit_criteria": "Acceptance check has no concrete blockers and warnings are explicitly owned.",
            },
            {
                "item_name": "security_legal_handoff",
                "label": "Security and legal handoff",
                "owner": "Security and legal reviewer",
                "sources": ["SECURITY.md", "docs/commercial_contract_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if has_file("SECURITY.md") and has_file("docs/commercial_contract_readiness.md")
                else "blocked",
                "evidence": "Security policy and contract readiness packet are available for buyer handoff.",
                "action": "Attach security policy, dependency lock, and contract readiness rows to buyer diligence.",
                "exit_criteria": "Buyer security/legal reviewer accepts the packet or opens concrete findings.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_saleability_decision.md", "docs/commercial_onboarding_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review delay is not an onboarding blocker unless a concrete failure is produced.",
                "action": "Continue onboarding work while queued reviews are pending.",
                "exit_criteria": "Only concrete security, API contract, document, or product defects block progress.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": packaging_item["owner"],
                "sources": packaging_item["sources"],
                "evidence_type": packaging_item["evidence_type"],
                "completion_state": packaging_item["completion_state"],
                "evidence": packaging_item["evidence"],
                "action": "Keep one deployable enterprise control-plane product through onboarding.",
                "exit_criteria": "Extract only after a second product, independent release cadence, or buyer provenance trigger exists.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in onboarding_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            onboarding_status = "commercial_onboarding_blocked"
        elif warning_count:
            onboarding_status = "commercial_onboarding_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            onboarding_status = "commercial_onboarding_ready"  # pragma: no cover

        return {
            "onboarding_status": onboarding_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_onboarding_readiness",
            "source_note": (
                "Commercial onboarding readiness converts local contract and procurement warnings into "
                "paid-onboarding owners, actions, and exit criteria; it is not a valuation guarantee, "
                "purchase commitment, or production compliance certificate."
            ),
            "onboarding_summary": {
                "item_count": len(onboarding_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "support_slo_action_count": support_slo_action_count,
                "buyer_input_action_count": buyer_input_action_count,
                "review_process_is_blocker": contract["review_process_policy"]["is_blocker"],
            },
            "onboarding_items": onboarding_items,
            "concrete_blockers": concrete_blockers,
            "onboarding_status_rules": [
                {
                    "onboarding_status": "commercial_onboarding_ready",
                    "rule": "kickoff packet, support/SLO, buyer input, telemetry, acceptance, security/legal, review, and packaging actions are ready",
                },
                {
                    "onboarding_status": "commercial_onboarding_ready_with_warnings",
                    "rule": "local onboarding plan is ready while production support/SLO or buyer order-form actions remain explicit warnings",
                },
                {
                    "onboarding_status": "commercial_onboarding_blocked",
                    "rule": "missing onboarding packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks onboarding",
                },
            ],
            "review_process_policy": contract["review_process_policy"],
            "related_runtime_reports": {
                "commercial_contract_status": contract["contract_status"],
                **contract["related_runtime_reports"],
            },
            "library_split_decision": contract["library_split_decision"],
            "plugin_traceability": contract["plugin_traceability"],
            "onboarding_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_onboarding_readiness/latest",
                "documentation": "docs/commercial_onboarding_readiness.md",
            },
        }

    def commercial_operations_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an operations-handoff readiness gate over onboarding evidence."""
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        onboarding_by_name = {item["item_name"]: item for item in onboarding["onboarding_items"]}
        support_item = onboarding_by_name["support_slo_kickoff"]
        telemetry_item = onboarding_by_name["telemetry_capture_plan"]
        acceptance_item = onboarding_by_name["acceptance_exit_criteria"]
        security_item = onboarding_by_name["security_legal_handoff"]
        packaging_item = onboarding_by_name["packaging_decision"]
        concrete_blockers = onboarding["concrete_blockers"]
        operations_items = [
            {
                "item_name": "deployment_runbook",
                "label": "Deployment runbook",
                "owner": "Platform operator",
                "sources": [
                    "README.md",
                    "docs/commercial_operations_readiness.md",
                    "docs/commercial_onboarding_readiness.md",
                    "docs/rest_api_design.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "docs/commercial_operations_readiness.md",
                        "docs/commercial_onboarding_readiness.md",
                        "docs/rest_api_design.md",
                    )
                )
                else "blocked",
                "evidence": "Repository overview, REST contract, onboarding plan, and operations handoff plan are present.",
                "action": "Use existing stdlib server and documented endpoints for buyer operations handoff.",
                "exit_criteria": "Buyer operator can start, authenticate, inspect readiness endpoints, and run verification commands.",
            },
            {
                "item_name": "monitoring_telemetry_capture",
                "label": "Monitoring and telemetry capture",
                "owner": telemetry_item["owner"],
                "sources": telemetry_item["sources"],
                "evidence_type": "proposed_until_production",
                "completion_state": "warning",
                "source_gap_status": "production_input_required",
                "evidence": telemetry_item["evidence"],
                "action": "Capture adoption, latency, verifier outcomes, trace completeness, support events, and deployment health in the buyer environment.",
                "exit_criteria": "First production telemetry snapshot is attached without mixing it with local prototype metrics.",
            },
            {
                "item_name": "incident_rollback_plan",
                "label": "Incident and rollback plan",
                "owner": "Operations and support owner",
                "sources": ["docs/commercial_onboarding_readiness.md", "docs/commercial_buyer_acceptance_runbook.md"],
                "evidence_type": "proposed_until_production",
                "completion_state": "warning",
                "source_gap_status": "production_input_required",
                "evidence": "Incident drill and rollback proof require a buyer deployment or paid onboarding environment.",
                "action": "Run the first incident drill and rollback exercise during onboarding.",
                "exit_criteria": "Incident owner, escalation path, rollback steps, and drill record are attached.",
            },
            {
                "item_name": "backup_recovery_plan",
                "label": "Backup and recovery evidence",
                "owner": "Operations and data owner",
                "sources": ["docs/commercial_onboarding_readiness.md", "docs/commercial_buyer_diligence_packet.md"],
                "evidence_type": "proposed_until_production",
                "completion_state": "warning",
                "source_gap_status": "production_input_required",
                "evidence": "Backup and recovery evidence depends on the buyer deployment topology and persistence choices.",
                "action": "Define backup scope, retention, restore owner, and first restore proof during onboarding.",
                "exit_criteria": "Buyer accepts backup scope and a restore proof is attached or explicitly waived.",
            },
            {
                "item_name": "support_slo_ownership",
                "label": "Support rota and SLO ownership",
                "owner": support_item["owner"],
                "sources": support_item["sources"],
                "evidence_type": support_item["evidence_type"],
                "completion_state": support_item["completion_state"],
                "source_gap_status": support_item.get("source_gap_status", "resolved"),
                "evidence": support_item["evidence"],
                "action": support_item["action"],
                "exit_criteria": support_item["exit_criteria"],
            },
            {
                "item_name": "acceptance_handoff",
                "label": "Acceptance handoff",
                "owner": acceptance_item["owner"],
                "sources": acceptance_item["sources"],
                "evidence_type": acceptance_item["evidence_type"],
                "completion_state": acceptance_item["completion_state"],
                "evidence": acceptance_item["evidence"],
                "action": acceptance_item["action"],
                "exit_criteria": acceptance_item["exit_criteria"],
            },
            {
                "item_name": "security_legal_handoff",
                "label": "Security and legal handoff",
                "owner": security_item["owner"],
                "sources": security_item["sources"],
                "evidence_type": security_item["evidence_type"],
                "completion_state": security_item["completion_state"],
                "evidence": security_item["evidence"],
                "action": security_item["action"],
                "exit_criteria": security_item["exit_criteria"],
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_saleability_decision.md", "docs/commercial_operations_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review delay is not an operations blocker unless a concrete failure is produced.",
                "action": "Continue operations handoff work while queued reviews are pending.",
                "exit_criteria": "Only concrete security, API contract, document, or product defects block progress.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": packaging_item["owner"],
                "sources": packaging_item["sources"],
                "evidence_type": packaging_item["evidence_type"],
                "completion_state": packaging_item["completion_state"],
                "evidence": packaging_item["evidence"],
                "action": packaging_item["action"],
                "exit_criteria": packaging_item["exit_criteria"],
            },
        ]
        state_counts = Counter(item["completion_state"] for item in operations_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        production_evidence_action_count = sum(
            1 for item in operations_items if item.get("source_gap_status") == "production_input_required"
        )
        if blocked_count:
            operations_status = "commercial_operations_blocked"
        elif warning_count:
            operations_status = "commercial_operations_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            operations_status = "commercial_operations_ready"  # pragma: no cover

        return {
            "operations_status": operations_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_operations_readiness",
            "source_note": (
                "Commercial operations readiness converts local onboarding evidence and production "
                "operations gaps into handoff owners, actions, and exit criteria; it is not a valuation "
                "guarantee, purchase commitment, or production compliance certificate."
            ),
            "operations_summary": {
                "item_count": len(operations_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "production_evidence_action_count": production_evidence_action_count,
                "review_process_is_blocker": onboarding["review_process_policy"]["is_blocker"],
            },
            "operations_items": operations_items,
            "concrete_blockers": concrete_blockers,
            "operations_status_rules": [
                {
                    "operations_status": "commercial_operations_ready",
                    "rule": "deployment, monitoring, incident, backup, support, acceptance, security/legal, review, and packaging evidence are ready",
                },
                {
                    "operations_status": "commercial_operations_ready_with_warnings",
                    "rule": "local operations plan is ready while production telemetry, incident, backup, or SLO evidence remains explicit warnings",
                },
                {
                    "operations_status": "commercial_operations_blocked",
                    "rule": "missing operations packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks operations handoff",
                },
            ],
            "review_process_policy": onboarding["review_process_policy"],
            "related_runtime_reports": {
                "commercial_onboarding_status": onboarding["onboarding_status"],
                **onboarding["related_runtime_reports"],
            },
            "library_split_decision": onboarding["library_split_decision"],
            "plugin_traceability": onboarding["plugin_traceability"],
            "operations_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_operations_readiness/latest",
                "documentation": "docs/commercial_operations_readiness.md",
            },
        }

    def commercial_security_attestation_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a buyer security-review attestation gate over operations evidence."""
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        operations_by_name = {item["item_name"]: item for item in operations["operations_items"]}
        runtime_profile = security_profile or {}
        concrete_blockers = operations["concrete_blockers"]
        security_attestation_items = [
            {
                "item_name": "security_policy",
                "label": "Security policy",
                "owner": "Security owner",
                "sources": ["SECURITY.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready" if has_file("SECURITY.md") else "blocked",
                "evidence": "Repository security disclosure and support policy is present.",
                "action": "Attach SECURITY.md to the buyer security review packet.",
                "exit_criteria": "Buyer can identify the vulnerability reporting path and supported scope.",
            },
            {
                "item_name": "dependency_lock_package_metadata",
                "label": "Dependency lock and package metadata",
                "owner": "Release owner",
                "sources": ["requirements.lock", "pyproject.toml"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if has_file("requirements.lock") and has_file("pyproject.toml")
                else "blocked",
                "evidence": "Pinned dependency lock and Python package metadata are present for supply-chain review.",
                "action": "Use the pinned lockfile and package metadata as the buyer dependency baseline.",
                "exit_criteria": "Buyer can inspect package metadata and reproduce the dependency installation path.",
            },
            {
                "item_name": "security_workflow_metadata",
                "label": "Security workflow metadata",
                "owner": "Security owner",
                "sources": [
                    ".github/dependabot.yml",
                    ".github/workflows/security.yml",
                    "ContextualWisdomLab/.github central required security workflows",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        ".github/dependabot.yml",
                        ".github/workflows/security.yml",
                    )
                )
                else "blocked",
                "evidence": "Dependabot plus local CodeQL and pip-audit/SBOM workflows are defined; dependency review, Trivy, OSV, and Scorecard are delegated to central required workflows.",
                "action": "Attach workflow definitions and latest passing run evidence when the buyer review requests hosted CI proof.",
                "exit_criteria": "Buyer can inspect the configured security workflow controls and their latest run status separately.",
            },
            {
                "item_name": "runtime_access_control_profile",
                "label": "Runtime access-control profile",
                "owner": "Platform operator",
                "sources": ["contextual_orchestrator/server.py", "/api/v1/commercial_operations_readiness/latest"],
                "evidence_type": "runtime_configuration",
                "completion_state": "ready",
                "evidence": (
                    f"Runtime profile uses auth_mode={runtime_profile.get('auth_mode', 'unknown')}, "
                    f"allow_public_bind={runtime_profile.get('allow_public_bind', False)}, "
                    f"expose_trace_by_default={runtime_profile.get('expose_trace_by_default', False)}, "
                    f"rate_limit_requests={runtime_profile.get('rate_limit_requests', 'unknown')}, "
                    f"max_concurrent_runs={runtime_profile.get('max_concurrent_runs', 'unknown')}."
                ),
                "action": "Use the secret-free runtime profile as buyer-visible access-control evidence.",
                "exit_criteria": "Buyer can verify admin and inference scopes, public bind opt-in, trace exposure default, rate limit, and concurrency controls.",
            },
            {
                "item_name": "audit_export_evidence",
                "label": "Audit and evidence export",
                "owner": "Evidence owner",
                "sources": [
                    "docs/commercial_evidence_export.md",
                    "docs/commercial_operations_readiness.md",
                    "/api/v1/commercial_evidence_exports/latest",
                    "/api/v1/commercial_operations_readiness/latest",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if has_file("docs/commercial_evidence_export.md")
                and has_file("docs/commercial_operations_readiness.md")
                else "blocked",
                "evidence": "Commercial evidence export and operations readiness documents are present for buyer audit review.",
                "action": "Package runtime evidence export with operations readiness for the security review data room.",
                "exit_criteria": "Buyer can trace security claims to runtime endpoints and Markdown artifacts.",
            },
            {
                "item_name": "vulnerability_scan_evidence",
                "label": "Vulnerability scan evidence",
                "owner": "Security owner",
                "sources": [
                    ".github/workflows/security.yml",
                    "ContextualWisdomLab/.github central required security workflows",
                ],
                "evidence_type": "external_attestation_required",
                "completion_state": "warning",
                "source_gap_status": "external_attestation_required",
                "evidence": "Local supply-chain workflow metadata and central security scan workflow metadata exist, but the buyer packet still needs the latest hosted scan result or buyer-accepted equivalent.",
                "action": "Attach latest CodeQL, pip-audit, Trivy, SBOM, and Scorecard results when CI completes or the buyer requests evidence.",
                "exit_criteria": "Hosted scan outputs are attached, or the buyer explicitly accepts workflow definitions as sufficient for this stage.",
            },
            {
                "item_name": "third_party_attestation_pen_test",
                "label": "Third-party attestation or penetration test",
                "owner": "Security owner",
                "sources": ["buyer security review", "external assessor"],
                "evidence_type": "external_attestation_required",
                "completion_state": "warning",
                "source_gap_status": "external_attestation_required",
                "evidence": "Independent SOC 2, ISO 27001, penetration-test, or buyer security assessment evidence is outside the repo-local prototype.",
                "action": "Provide the buyer-requested attestation, schedule an assessment, or document an explicit waiver.",
                "exit_criteria": "Buyer accepts the third-party security evidence, scheduled assessment, or waiver.",
            },
            {
                "item_name": "buyer_privacy_dpa_questionnaire",
                "label": "Buyer privacy, DPA, and questionnaire input",
                "owner": "Deal owner",
                "sources": ["buyer DPA", "buyer privacy questionnaire", "buyer order form"],
                "evidence_type": "buyer_input_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_input_required",
                "evidence": "Privacy, DPA, subprocessors, data residency, and questionnaire answers depend on buyer-specific terms.",
                "action": "Collect buyer privacy questionnaire, DPA requirements, subprocessors, and data residency constraints.",
                "exit_criteria": "Buyer-specific privacy inputs are completed or explicitly waived in the deal packet.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": operations_by_name["review_process_policy"]["sources"],
                "evidence_type": operations_by_name["review_process_policy"]["evidence_type"],
                "completion_state": operations_by_name["review_process_policy"]["completion_state"],
                "evidence": operations_by_name["review_process_policy"]["evidence"],
                "action": operations_by_name["review_process_policy"]["action"],
                "exit_criteria": operations_by_name["review_process_policy"]["exit_criteria"],
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": operations_by_name["packaging_decision"]["owner"],
                "sources": operations_by_name["packaging_decision"]["sources"],
                "evidence_type": operations_by_name["packaging_decision"]["evidence_type"],
                "completion_state": operations_by_name["packaging_decision"]["completion_state"],
                "evidence": operations_by_name["packaging_decision"]["evidence"],
                "action": operations_by_name["packaging_decision"]["action"],
                "exit_criteria": operations_by_name["packaging_decision"]["exit_criteria"],
            },
        ]
        state_counts = Counter(item["completion_state"] for item in security_attestation_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        external_attestation_gap_count = sum(
            1 for item in security_attestation_items if item.get("source_gap_status") == "external_attestation_required"
        )
        buyer_privacy_gap_count = sum(
            1 for item in security_attestation_items if item.get("source_gap_status") == "buyer_input_required"
        )
        if blocked_count:
            security_attestation_status = "commercial_security_attestation_blocked"
        elif warning_count:
            security_attestation_status = "commercial_security_attestation_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            security_attestation_status = "commercial_security_attestation_ready"  # pragma: no cover

        return {
            "security_attestation_status": security_attestation_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_security_attestation",
            "source_note": (
                "Commercial security attestation separates repo-local security evidence from external "
                "attestation, hosted scan, and buyer privacy inputs; it is not a valuation guarantee, "
                "purchase commitment, production compliance certificate, or third-party security audit."
            ),
            "security_attestation_summary": {
                "item_count": len(security_attestation_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "external_attestation_gap_count": external_attestation_gap_count,
                "buyer_privacy_gap_count": buyer_privacy_gap_count,
                "review_process_is_blocker": operations["review_process_policy"]["is_blocker"],
            },
            "security_attestation_items": security_attestation_items,
            "concrete_blockers": concrete_blockers,
            "security_attestation_status_rules": [
                {
                    "security_attestation_status": "commercial_security_attestation_ready",
                    "rule": "security policy, dependency metadata, workflow metadata, access controls, audit export, external attestation, buyer privacy input, review policy, and packaging evidence are ready",
                },
                {
                    "security_attestation_status": "commercial_security_attestation_ready_with_warnings",
                    "rule": "repo-local security packet is ready while hosted scan evidence, third-party attestation, or buyer privacy input remains explicit warnings",
                },
                {
                    "security_attestation_status": "commercial_security_attestation_blocked",
                    "rule": "missing local security packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks security attestation",
                },
            ],
            "review_process_policy": operations["review_process_policy"],
            "related_runtime_reports": {
                "commercial_operations_status": operations["operations_status"],
                **operations["related_runtime_reports"],
            },
            "library_split_decision": operations["library_split_decision"],
            "plugin_traceability": operations["plugin_traceability"],
            "security_attestation_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_security_attestations/latest",
                "documentation": "docs/commercial_security_attestation.md",
            },
        }

    def commercial_value_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a buyer economic-review gate over value and ROI evidence."""
        commercial = self.commercial_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        export = self.commercial_evidence_export_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        criteria_by_name = self._criteria_by_name(commercial["criteria"])
        value_case = criteria_by_name["commercial_value_case"]
        kpis = self._metrics_by_name(analytics["kpis"])
        guardrails = self._metrics_by_name(analytics["guardrails"])
        security_items = {item["item_name"]: item for item in security["security_attestation_items"]}
        value_items = [
            {
                "item_name": "commercial_value_case_basis",
                "label": "Commercial value-case basis",
                "owner": "Deal owner",
                "sources": ["/api/v1/commercial_readiness/latest", "docs/commercial_readiness.md"],
                "evidence_type": "local_due_diligence_snapshot",
                "completion_state": "ready" if value_case["status"] == "pass" else "warning",
                "evidence": value_case["evidence"],
                "action": "Use commercial readiness as the value-case baseline without presenting it as a valuation guarantee.",
                "exit_criteria": "Buyer sees the KRW target as a review anchor, not as a guaranteed valuation.",
            },
            {
                "item_name": "local_analytics_evidence",
                "label": "Local analytics evidence",
                "owner": "Product analytics owner",
                "sources": ["/api/v1/analytics_snapshots/latest", "docs/analytics_spec.md"],
                "evidence_type": "measured_local",
                "completion_state": "ready",
                "evidence": (
                    f"compatible_api_adoption={kpis['compatible_api_adoption'].get('value')}; "
                    f"trace_complete_workflow_rate={kpis['trace_complete_workflow_rate'].get('value_percent')}%; "
                    f"policy_safe_routing_rate={kpis['policy_safe_routing_rate'].get('value_percent')}%; "
                    f"provider_exclusion_miss_rate={guardrails['provider_exclusion_miss_rate'].get('value')}"
                ),
                "action": "Use local measured adoption, trace, policy, and provider-safety metrics as evidence only for this prototype.",
                "exit_criteria": "Buyer understands these are local measured signals, not production revenue or customer usage claims.",
            },
            {
                "item_name": "buyer_evidence_export",
                "label": "Buyer evidence export",
                "owner": "Evidence owner",
                "sources": [
                    "/api/v1/commercial_evidence_exports/latest",
                    "/api/v1/commercial_security_attestations/latest",
                    "docs/commercial_evidence_export.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready" if has_file("docs/commercial_evidence_export.md") else "blocked",
                "evidence": (
                    f"commercial_export_status={export['export_status']}; "
                    f"security_attestation_status={security['security_attestation_status']}"
                ),
                "action": "Package value evidence with export and security attestation outputs in the buyer data room.",
                "exit_criteria": "Buyer can trace economic claims back to runtime endpoints and repo artifacts.",
            },
            {
                "item_name": "pricing_package_rationale",
                "label": "Pricing and package rationale",
                "owner": "Deal owner",
                "sources": [
                    "docs/commercial_readiness.md",
                    "docs/commercial_saleability_decision.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_value_readiness.md",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "docs/commercial_readiness.md",
                        "docs/commercial_saleability_decision.md",
                        "docs/commercial_procurement_readiness.md",
                    )
                )
                else "blocked",
                "evidence": "Commercial readiness, saleability, and procurement documents anchor the package rationale.",
                "action": "Keep the KRW 2B package rationale tied to API compatibility, evidence control plane, replay, audit, security, and operations readiness.",
                "exit_criteria": "Buyer can inspect which product capabilities support the package rationale.",
            },
            {
                "item_name": "roi_model_inputs",
                "label": "ROI model inputs",
                "owner": "Buyer sponsor and deal owner",
                "sources": ["buyer ROI model", "customer discovery", "procurement value case"],
                "evidence_type": "buyer_input_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_financial_input_required",
                "evidence": "Buyer-specific baseline cost, workflow volume, error/rework cost, compliance cost, and time-saving assumptions are not repo-local facts.",
                "action": "Collect buyer baseline metrics and map them to API compatibility, trace audit, replay, and operations savings.",
                "exit_criteria": "Buyer accepts the ROI model inputs or marks them waived for the commercial review.",
            },
            {
                "item_name": "reference_customer_or_case_study",
                "label": "Reference customer or proof",
                "owner": "Deal owner",
                "sources": ["reference customer", "case study", "paid pilot result"],
                "evidence_type": "external_value_proof_required",
                "completion_state": "warning",
                "source_gap_status": "external_value_proof_required",
                "evidence": "Reference customer, paid pilot, or production proof is external to the repo-local prototype.",
                "action": "Attach a reference, pilot result, or explicit buyer waiver before treating the value case as externally proven.",
                "exit_criteria": "Buyer accepts the reference proof, pilot result, or waiver.",
            },
            {
                "item_name": "procurement_budget_owner",
                "label": "Procurement budget owner",
                "owner": "Buyer sponsor and procurement owner",
                "sources": ["buyer order form", "procurement process", "budget approval"],
                "evidence_type": "buyer_input_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_financial_input_required",
                "evidence": "Budget owner, approval path, and order-form authority are buyer-specific inputs.",
                "action": "Identify sponsor, budget owner, procurement path, and order-form authority.",
                "exit_criteria": "Buyer confirms the budget owner and approval path for the KRW 2B review.",
            },
            {
                "item_name": "implementation_payback_assumption",
                "label": "Implementation payback assumption",
                "owner": "Buyer sponsor and onboarding owner",
                "sources": ["buyer onboarding plan", "implementation estimate", "operations handoff"],
                "evidence_type": "buyer_input_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_financial_input_required",
                "evidence": "Implementation timeline, staffing, opportunity cost, and payback window depend on buyer deployment scope.",
                "action": "Estimate implementation effort and payback window during paid onboarding or buyer diligence.",
                "exit_criteria": "Buyer accepts the payback assumptions or marks them out of scope.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": security_items["review_process_policy"]["sources"],
                "evidence_type": security_items["review_process_policy"]["evidence_type"],
                "completion_state": security_items["review_process_policy"]["completion_state"],
                "evidence": security_items["review_process_policy"]["evidence"],
                "action": security_items["review_process_policy"]["action"],
                "exit_criteria": security_items["review_process_policy"]["exit_criteria"],
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": security_items["packaging_decision"]["owner"],
                "sources": security_items["packaging_decision"]["sources"],
                "evidence_type": security_items["packaging_decision"]["evidence_type"],
                "completion_state": security_items["packaging_decision"]["completion_state"],
                "evidence": security_items["packaging_decision"]["evidence"],
                "action": security_items["packaging_decision"]["action"],
                "exit_criteria": security_items["packaging_decision"]["exit_criteria"],
            },
        ]
        state_counts = Counter(item["completion_state"] for item in value_items)
        concrete_blockers = security["concrete_blockers"]
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        buyer_financial_gap_count = sum(
            1
            for item in value_items
            if item.get("source_gap_status") in {"buyer_financial_input_required", "external_value_proof_required"}
        )
        external_value_proof_gap_count = sum(
            1 for item in value_items if item.get("source_gap_status") == "external_value_proof_required"
        )
        if blocked_count:
            value_status = "commercial_value_blocked"
        elif warning_count:
            value_status = "commercial_value_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            value_status = "commercial_value_ready"  # pragma: no cover

        return {
            "value_status": value_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_value_readiness",
            "source_note": (
                "Commercial value readiness separates repo-local measured evidence from buyer-specific "
                "ROI, reference, budget, and payback inputs; it is not a valuation guarantee, purchase "
                "commitment, revenue proof, or financial advice."
            ),
            "value_summary": {
                "item_count": len(value_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "buyer_financial_gap_count": buyer_financial_gap_count,
                "external_value_proof_gap_count": external_value_proof_gap_count,
                "review_process_is_blocker": security["review_process_policy"]["is_blocker"],
            },
            "value_items": value_items,
            "concrete_blockers": concrete_blockers,
            "value_status_rules": [
                {
                    "value_status": "commercial_value_ready",
                    "rule": "commercial value case, local analytics, evidence export, pricing rationale, ROI inputs, reference proof, budget owner, payback assumptions, review policy, and packaging evidence are ready",
                },
                {
                    "value_status": "commercial_value_ready_with_warnings",
                    "rule": "repo-local value evidence is ready while buyer ROI inputs, reference proof, budget owner, or payback assumptions remain explicit warnings",
                },
                {
                    "value_status": "commercial_value_blocked",
                    "rule": "missing local value packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks value readiness",
                },
            ],
            "review_process_policy": security["review_process_policy"],
            "related_runtime_reports": {
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_export_status": export["export_status"],
                "commercial_status": commercial["commercial_status"],
                **security["related_runtime_reports"],
            },
            "library_split_decision": security["library_split_decision"],
            "plugin_traceability": security["plugin_traceability"],
            "value_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_value_readiness/latest",
                "documentation": "docs/commercial_value_readiness.md",
            },
        }

    def commercial_close_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the final buyer-close gate over commercial readiness evidence."""
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        export = self.commercial_evidence_export_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = [
            *value["concrete_blockers"],
            *security["concrete_blockers"],
            *contract["concrete_blockers"],
            *onboarding["concrete_blockers"],
            *operations["concrete_blockers"],
            *export["concrete_blockers"],
        ]
        concrete_blockers = list(dict.fromkeys(concrete_blockers))
        close_items = [
            {
                "item_name": "sellable_product_packet",
                "label": "Sellable product packet",
                "owner": "Deal owner",
                "sources": [
                    "/api/v1/commercial_value_readiness/latest",
                    "/api/v1/commercial_security_attestations/latest",
                    "/api/v1/commercial_evidence_exports/latest",
                    "docs/commercial_value_readiness.md",
                    "docs/commercial_security_attestation.md",
                    "docs/commercial_evidence_export.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if value["value_status"] != "commercial_value_blocked"
                and security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and export["export_status"] != "commercial_export_blocked"
                and has_file("docs/commercial_value_readiness.md")
                and has_file("docs/commercial_security_attestation.md")
                and has_file("docs/commercial_evidence_export.md")
                else "blocked",
                "evidence": (
                    f"value_status={value['value_status']}; "
                    f"security_attestation_status={security['security_attestation_status']}; "
                    f"commercial_export_status={export['export_status']}"
                ),
                "action": "Attach the repo-local product, security, value, and evidence export packet to buyer close review.",
                "exit_criteria": "Buyer can inspect the sellable packet without treating it as a purchase commitment or valuation guarantee.",
            },
            {
                "item_name": "contract_close_packet",
                "label": "Contract close packet",
                "owner": "Legal and procurement owner",
                "sources": [
                    "/api/v1/commercial_contract_readiness/latest",
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if contract["contract_status"] != "commercial_contract_blocked"
                and has_file("docs/commercial_contract_readiness.md")
                else "blocked",
                "evidence": f"contract_status={contract['contract_status']}",
                "action": "Use contract readiness as the local legal/procurement packet and track final signatures separately.",
                "exit_criteria": "Buyer legal/procurement sees local contract evidence and the remaining signature inputs.",
            },
            {
                "item_name": "onboarding_operations_packet",
                "label": "Onboarding and operations packet",
                "owner": "Customer success and platform owner",
                "sources": [
                    "/api/v1/commercial_onboarding_readiness/latest",
                    "/api/v1/commercial_operations_readiness/latest",
                    "docs/commercial_onboarding_readiness.md",
                    "docs/commercial_operations_readiness.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and operations["operations_status"] != "commercial_operations_blocked"
                and has_file("docs/commercial_onboarding_readiness.md")
                and has_file("docs/commercial_operations_readiness.md")
                else "blocked",
                "evidence": (
                    f"onboarding_status={onboarding['onboarding_status']}; "
                    f"operations_status={operations['operations_status']}"
                ),
                "action": "Attach onboarding and operations readiness as the go-live support packet.",
                "exit_criteria": "Buyer can identify implementation, support, operations, and acceptance owners.",
            },
            {
                "item_name": "buyer_evidence_export_packet",
                "label": "Buyer evidence export packet",
                "owner": "Evidence owner",
                "sources": [
                    "/api/v1/commercial_evidence_exports/latest",
                    "docs/commercial_evidence_export.md",
                    "docs/figma_artifacts.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if export["export_status"] != "commercial_export_blocked"
                and has_file("docs/commercial_evidence_export.md")
                and has_file("docs/figma_artifacts.md")
                else "blocked",
                "evidence": f"commercial_export_status={export['export_status']}",
                "action": "Use the portable export packet as the buyer data-room index.",
                "exit_criteria": "Buyer can trace close evidence to runtime endpoints, docs, and Figma/FigJam artifacts.",
            },
            {
                "item_name": "signed_order_form_msa",
                "label": "Signed order form or MSA",
                "owner": "Buyer sponsor, procurement owner, and deal owner",
                "sources": ["buyer order form", "MSA", "signature packet"],
                "evidence_type": "buyer_signature_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": "Signed order form, MSA, commercial terms, and authority confirmation are buyer-side close inputs.",
                "action": "Collect final signed order form or MSA, or attach an explicit buyer waiver.",
                "exit_criteria": "Buyer and seller signature authority accept the order form or MSA.",
            },
            {
                "item_name": "dpa_security_acceptance",
                "label": "DPA and security acceptance",
                "owner": "Buyer security, privacy, and legal owner",
                "sources": ["buyer DPA", "security review acceptance", "privacy questionnaire"],
                "evidence_type": "buyer_signature_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": "DPA, security acceptance, privacy questionnaire, and attestation waivers are buyer-specific close inputs.",
                "action": "Collect DPA/security acceptance or documented waiver from buyer security and legal reviewers.",
                "exit_criteria": "Buyer signs or waives DPA/security acceptance requirements.",
            },
            {
                "item_name": "budget_approval_purchase_order",
                "label": "Budget approval and purchase order",
                "owner": "Buyer finance and procurement owner",
                "sources": ["budget approval", "purchase order", "finance approval"],
                "evidence_type": "buyer_signature_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": "Budget approval, purchase order, and finance authority are external buyer procurement evidence.",
                "action": "Collect buyer budget approval and PO or attach approved alternative payment authority.",
                "exit_criteria": "Buyer procurement confirms budget authority and payment path for KRW 2B.",
            },
            {
                "item_name": "go_live_authorization",
                "label": "Go-live authorization",
                "owner": "Buyer business sponsor and implementation owner",
                "sources": ["go-live approval", "implementation authorization", "acceptance signoff"],
                "evidence_type": "buyer_signature_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": "Go-live authorization and implementation acceptance require named buyer approval.",
                "action": "Collect go-live authorization or mark production activation out of scope for the signed deal.",
                "exit_criteria": "Buyer authorizes go-live, paid onboarding, or a scoped post-signature implementation plan.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_saleability_decision.md", "docs/commercial_close_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review process delay is not a close blocker unless a concrete product, security, API-contract, or document failure is produced.",
                "action": "Keep commercial close work moving while queued review processes are pending.",
                "exit_criteria": "Only concrete failures block close readiness.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": "Procurement and security reviewer",
                "sources": ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if value["library_split_decision"]["decision"] == "keep_single_product"
                else "warning",
                "evidence": value["library_split_decision"]["reason"],
                "action": "Keep one deployable enterprise control-plane product until extraction triggers are real.",
                "exit_criteria": "Do not create a separate library, Git submodule, or extracted package for this close gate.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in close_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        buyer_signature_gap_count = sum(
            1 for item in close_items if item.get("source_gap_status") == "buyer_signature_required"
        )
        if blocked_count:
            close_status = "commercial_close_blocked"
        elif warning_count:
            close_status = "commercial_close_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            close_status = "commercial_close_ready"  # pragma: no cover

        return {
            "close_status": close_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_close_readiness",
            "source_note": (
                "Commercial close readiness separates repo-local sellable product evidence from buyer "
                "signature, legal, procurement, security acceptance, and go-live authorization inputs; "
                "it is not a valuation guarantee, purchase commitment, signed order, legal opinion, "
                "or production compliance certificate."
            ),
            "close_summary": {
                "item_count": len(close_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "buyer_signature_gap_count": buyer_signature_gap_count,
                "review_process_is_blocker": value["review_process_policy"]["is_blocker"],
            },
            "close_items": close_items,
            "concrete_blockers": concrete_blockers,
            "close_status_rules": [
                {
                    "close_status": "commercial_close_ready",
                    "rule": "sellable product packet, contract packet, onboarding/operations packet, evidence export, signatures, DPA/security acceptance, budget/PO, go-live authorization, review policy, and packaging evidence are ready",
                },
                {
                    "close_status": "commercial_close_ready_with_warnings",
                    "rule": "repo-local close packet is ready while buyer signatures, DPA/security acceptance, budget/PO, or go-live authorization remain explicit warnings",
                },
                {
                    "close_status": "commercial_close_blocked",
                    "rule": "missing local close evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks close readiness",
                },
            ],
            "review_process_policy": value["review_process_policy"],
            "related_runtime_reports": {
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_contract_status": contract["contract_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_operations_status": operations["operations_status"],
                "commercial_export_status": export["export_status"],
                **value["related_runtime_reports"],
            },
            "library_split_decision": value["library_split_decision"],
            "plugin_traceability": value["plugin_traceability"],
            "close_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_close_readiness/latest",
                "documentation": "docs/commercial_close_readiness.md",
            },
        }

    def commercial_go_to_market_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a buyer-facing GTM readiness index over commercial evidence."""
        close = self.commercial_close_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        export = self.commercial_evidence_export_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        handoff = self.commercial_handoff_bundle_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        saleability = self.saleability_decision_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = [
            *close["concrete_blockers"],
            *value["concrete_blockers"],
            *security["concrete_blockers"],
            *export["concrete_blockers"],
            *saleability["concrete_blockers"],
        ]
        concrete_blockers = list(dict.fromkeys(concrete_blockers))
        gtm_items = [
            {
                "item_name": "commercial_close_packet",
                "label": "Commercial close packet",
                "owner": "Deal owner",
                "sources": ["/api/v1/commercial_close_readiness/latest", "docs/commercial_close_readiness.md"],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if close["close_status"] != "commercial_close_blocked"
                and has_file("docs/commercial_close_readiness.md")
                else "blocked",
                "evidence": f"close_status={close['close_status']}",
                "action": "Use close readiness as the buyer-facing final readiness packet.",
                "exit_criteria": "Buyer sees local close packet status and remaining buyer-side signature gaps.",
            },
            {
                "item_name": "economic_value_packet",
                "label": "Economic value packet",
                "owner": "Deal owner and analytics owner",
                "sources": [
                    "/api/v1/commercial_value_readiness/latest",
                    "/api/v1/analytics_snapshots/latest",
                    "docs/commercial_value_readiness.md",
                    "docs/analytics_spec.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if value["value_status"] != "commercial_value_blocked"
                and has_file("docs/commercial_value_readiness.md")
                and has_file("docs/analytics_spec.md")
                else "blocked",
                "evidence": (
                    f"value_status={value['value_status']}; "
                    f"kpi_count={len(analytics['kpis'])}; guardrail_count={len(analytics['guardrails'])}"
                ),
                "action": "Show value evidence with measured-local versus buyer-specific metric separation.",
                "exit_criteria": "Buyer can inspect value claims without treating them as revenue proof or financial advice.",
            },
            {
                "item_name": "security_trust_packet",
                "label": "Security trust packet",
                "owner": "Security owner",
                "sources": [
                    "/api/v1/commercial_security_attestations/latest",
                    "docs/commercial_security_attestation.md",
                    "SECURITY.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and has_file("docs/commercial_security_attestation.md")
                and has_file("SECURITY.md")
                else "blocked",
                "evidence": f"security_attestation_status={security['security_attestation_status']}",
                "action": "Use security attestation as the buyer trust packet and keep external attestations separate.",
                "exit_criteria": "Buyer can inspect local security controls and external attestation gaps.",
            },
            {
                "item_name": "buyer_evidence_packet",
                "label": "Buyer evidence packet",
                "owner": "Evidence owner",
                "sources": [
                    "/api/v1/commercial_evidence_exports/latest",
                    "/api/v1/commercial_handoff_bundles/latest",
                    "docs/commercial_evidence_export.md",
                    "docs/commercial_buyer_handoff_bundle.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if export["export_status"] != "commercial_export_blocked"
                and handoff["bundle_status"] != "buyer_handoff_blocked"
                and has_file("docs/commercial_evidence_export.md")
                and has_file("docs/commercial_buyer_handoff_bundle.md")
                else "blocked",
                "evidence": f"commercial_export_status={export['export_status']}; buyer_handoff_status={handoff['bundle_status']}",
                "action": "Attach evidence export and handoff bundle as the buyer data-room index.",
                "exit_criteria": "Buyer can trace GTM claims to runtime endpoints, docs, tests, and Figma artifacts.",
            },
            {
                "item_name": "saleability_decision_packet",
                "label": "Saleability decision packet",
                "owner": "Deal owner",
                "sources": ["/api/v1/saleability_decisions/latest", "docs/commercial_saleability_decision.md"],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if saleability["saleability_status"] != "saleability_blocked"
                and has_file("docs/commercial_saleability_decision.md")
                else "blocked",
                "evidence": f"saleability_status={saleability['saleability_status']}",
                "action": "Use saleability decision as the GTM go/no-go baseline.",
                "exit_criteria": "Buyer and stakeholder review can distinguish warnings from concrete blockers.",
            },
            {
                "item_name": "admin_operator_evidence",
                "label": "Admin operator evidence",
                "owner": "Product design owner",
                "sources": ["/admin", "contextual_orchestrator/admin.py", "docs/screen_design.md"],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if has_file("contextual_orchestrator/admin.py") and has_file("docs/screen_design.md")
                else "blocked",
                "evidence": "Admin surface exposes readiness status, source notes, measurement status, and warning/blocker summaries.",
                "action": "Use the existing admin observability surface instead of creating a separate sales dashboard.",
                "exit_criteria": "Operator can inspect GTM readiness from the current admin surface.",
            },
            {
                "item_name": "analytics_truthfulness_packet",
                "label": "Analytics truthfulness packet",
                "owner": "Data analytics owner",
                "sources": ["docs/analytics_spec.md", "/api/v1/analytics_snapshots/latest"],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready" if has_file("docs/analytics_spec.md") else "blocked",
                "evidence": "Analytics spec separates measured local evidence from proposed production or buyer-specific inputs.",
                "action": "Keep GTM metrics from claiming production revenue, signed buyer proof, or unmeasured telemetry.",
                "exit_criteria": "Stakeholders can see which KPI fields are measured and which are proposed inputs.",
            },
            {
                "item_name": "stakeholder_artifacts_packet",
                "label": "Stakeholder artifacts packet",
                "owner": "Figma and Product Design owner",
                "sources": [
                    "docs/figma_artifacts.md",
                    "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                    "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                ],
                "evidence_type": "figma_artifact",
                "completion_state": "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "evidence": "Editable Figma/FigJam stakeholder artifacts are recorded and Figma Code Connect is excluded.",
                "action": "Use editable stakeholder artifacts for GTM review instead of screenshot-only evidence.",
                "exit_criteria": "Stakeholders can open the design file and FigJam board for GTM review.",
            },
            {
                "item_name": "buyer_signature_budget_follow_up",
                "label": "Buyer signature and budget follow-up",
                "owner": "Buyer sponsor, procurement owner, and deal owner",
                "sources": ["buyer order form", "MSA", "DPA", "security acceptance", "purchase order", "go-live approval"],
                "evidence_type": "buyer_input_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": (
                    f"buyer_signature_gap_count={close['close_summary']['buyer_signature_gap_count']}; "
                    "signed order/MSA, DPA/security acceptance, budget/PO, and go-live authorization are buyer inputs."
                ),
                "action": "Collect buyer signatures, approvals, or waivers before representing the packet as closed-won.",
                "exit_criteria": "Buyer accepts or waives all signature, budget, security acceptance, and go-live inputs.",
            },
            {
                "item_name": "production_external_proof_follow_up",
                "label": "Production and external proof follow-up",
                "owner": "Security, operations, and deal owner",
                "sources": ["hosted scan output", "third-party attestation", "reference proof", "production telemetry"],
                "evidence_type": "external_or_production_input_required",
                "completion_state": "warning",
                "source_gap_status": "external_or_production_input_required",
                "evidence": (
                    f"security_warning_count={security['security_attestation_summary']['warning_count']}; "
                    f"value_warning_count={value['value_summary']['warning_count']}; "
                    f"export_warning_count={export['export_summary']['warning_count']}"
                ),
                "action": "Attach hosted scan, third-party attestation, reference proof, and production telemetry when available.",
                "exit_criteria": "Buyer accepts external proof, production proof, or an explicit waiver.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_go_to_market_readiness.md", "docs/commercial_saleability_decision.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review process delay is not a GTM blocker unless a concrete failure is produced.",
                "action": "Continue GTM readiness work while queued reviews are pending.",
                "exit_criteria": "Only concrete product, security, API contract, or document failures block GTM readiness.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": "Procurement and security reviewer",
                "sources": ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if close["library_split_decision"]["decision"] == "keep_single_product"
                else "warning",
                "evidence": close["library_split_decision"]["reason"],
                "action": "Keep one deployable enterprise control-plane product until extraction triggers are real.",
                "exit_criteria": "Do not create a separate library, Git submodule, or extracted package for this GTM gate.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in gtm_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        buyer_signature_gap_count = close["close_summary"]["buyer_signature_gap_count"]
        external_or_production_gap_count = (
            security["security_attestation_summary"]["external_attestation_gap_count"]
            + value["value_summary"]["external_value_proof_gap_count"]
            + export["export_summary"]["warning_count"]
        )
        if blocked_count:
            gtm_status = "commercial_go_to_market_blocked"
        elif warning_count:
            gtm_status = "commercial_go_to_market_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            gtm_status = "commercial_go_to_market_ready"  # pragma: no cover

        return {
            "go_to_market_status": gtm_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_go_to_market_readiness",
            "source_note": (
                "Commercial go-to-market readiness indexes repo-local sellable product, evidence, "
                "admin, analytics, and stakeholder artifacts separately from buyer signatures, "
                "external proof, and production telemetry; it is not a valuation guarantee, purchase "
                "commitment, signed order, legal opinion, production compliance certificate, or revenue proof."
            ),
            "go_to_market_summary": {
                "item_count": len(gtm_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "buyer_signature_gap_count": buyer_signature_gap_count,
                "external_or_production_gap_count": external_or_production_gap_count,
                "review_process_is_blocker": close["review_process_policy"]["is_blocker"],
            },
            "go_to_market_items": gtm_items,
            "concrete_blockers": concrete_blockers,
            "go_to_market_status_rules": [
                {
                    "go_to_market_status": "commercial_go_to_market_ready",
                    "rule": "close, value, security, evidence, saleability, admin, analytics, stakeholder artifacts, buyer inputs, external proof, review policy, and packaging evidence are ready",
                },
                {
                    "go_to_market_status": "commercial_go_to_market_ready_with_warnings",
                    "rule": "repo-local GTM packet is ready while buyer signatures, budget/PO, DPA/security acceptance, production telemetry, reference proof, hosted scan, or third-party attestation remain explicit warnings",
                },
                {
                    "go_to_market_status": "commercial_go_to_market_blocked",
                    "rule": "missing local GTM packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks GTM readiness",
                },
            ],
            "review_process_policy": close["review_process_policy"],
            "related_runtime_reports": {
                "commercial_close_status": close["close_status"],
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_export_status": export["export_status"],
                "buyer_handoff_status": handoff["bundle_status"],
                "saleability_status": saleability["saleability_status"],
                **close["related_runtime_reports"],
            },
            "library_split_decision": close["library_split_decision"],
            "plugin_traceability": close["plugin_traceability"],
            "go_to_market_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_go_to_market_readiness/latest",
                "documentation": "docs/commercial_go_to_market_readiness.md",
            },
        }

    def commercial_launch_readiness_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the buyer launch/trial readiness gate over commercial evidence."""
        gtm = self.commercial_go_to_market_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        acceptance = self.commercial_acceptance_check_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = [
            *gtm["concrete_blockers"],
            *operations["concrete_blockers"],
            *onboarding["concrete_blockers"],
            *acceptance["concrete_blockers"],
        ]
        concrete_blockers = list(dict.fromkeys(concrete_blockers))
        launch_items = [
            {
                "item_name": "go_to_market_packet",
                "label": "Go-to-market packet",
                "owner": "Deal owner",
                "sources": [
                    "/api/v1/commercial_go_to_market_readiness/latest",
                    "docs/commercial_go_to_market_readiness.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if gtm["go_to_market_status"] != "commercial_go_to_market_blocked"
                and has_file("docs/commercial_go_to_market_readiness.md")
                else "blocked",
                "evidence": f"go_to_market_status={gtm['go_to_market_status']}",
                "action": "Use the GTM packet as the launch/trial entry evidence.",
                "exit_criteria": "Buyer can inspect the launch packet without treating it as a signed deal or production proof.",
            },
            {
                "item_name": "runtime_launch_path",
                "label": "Runtime launch path",
                "owner": "Platform operator",
                "sources": [
                    "README.md",
                    "contextual_orchestrator/server.py",
                    "contextual_orchestrator/api_contract.py",
                    "docs/rest_api_design.md",
                    "/v1/chat/completions",
                    "/admin",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "README.md",
                        "contextual_orchestrator/server.py",
                        "contextual_orchestrator/api_contract.py",
                        "docs/rest_api_design.md",
                    )
                )
                else "blocked",
                "evidence": "Stdlib server, OpenAI-compatible endpoint, admin console, and REST contract are present.",
                "action": "Run the existing server and admin/API smoke tests for buyer trial setup.",
                "exit_criteria": "Buyer can start the runtime, authenticate admin calls, and inspect launch readiness JSON.",
            },
            {
                "item_name": "acceptance_test_packet",
                "label": "Acceptance test packet",
                "owner": "Technical reviewer",
                "sources": [
                    "/api/v1/commercial_acceptance_checks/latest",
                    "tests/test_commercial_acceptance_check.py",
                    "tests/test_commercial_go_to_market_readiness.py",
                    "tests/test_commercial_launch_readiness.py",
                    "pytest -q",
                ],
                "evidence_type": "measured_local",
                "completion_state": "ready"
                if acceptance["acceptance_status"] != "commercial_acceptance_blocked"
                and all(
                    has_file(path)
                    for path in (
                        "tests/test_commercial_acceptance_check.py",
                        "tests/test_commercial_go_to_market_readiness.py",
                        "tests/test_commercial_launch_readiness.py",
                    )
                )
                else "blocked",
                "evidence": f"acceptance_status={acceptance['acceptance_status']}",
                "action": "Use focused acceptance and launch tests as the local verification packet.",
                "exit_criteria": "Focused launch, GTM, acceptance, API, and artifact tests pass before buyer handoff.",
            },
            {
                "item_name": "operator_runbook_packet",
                "label": "Operator runbook packet",
                "owner": "Customer success and platform owner",
                "sources": [
                    "/api/v1/commercial_operations_readiness/latest",
                    "/api/v1/commercial_onboarding_readiness/latest",
                    "docs/commercial_operations_readiness.md",
                    "docs/commercial_onboarding_readiness.md",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if operations["operations_status"] != "commercial_operations_blocked"
                and onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and has_file("docs/commercial_operations_readiness.md")
                and has_file("docs/commercial_onboarding_readiness.md")
                else "blocked",
                "evidence": (
                    f"operations_status={operations['operations_status']}; "
                    f"onboarding_status={onboarding['onboarding_status']}"
                ),
                "action": "Attach onboarding and operations readiness as the launch runbook.",
                "exit_criteria": "Buyer sees implementation, support, telemetry, incident, backup, and acceptance owners.",
            },
            {
                "item_name": "admin_observability_packet",
                "label": "Admin observability packet",
                "owner": "Product design owner",
                "sources": ["/admin", "/admin/state", "contextual_orchestrator/admin.py", "docs/screen_design.md"],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if admin_state["agents"] and has_file("contextual_orchestrator/admin.py") and has_file("docs/screen_design.md")
                else "blocked",
                "evidence": (
                    f"agent_count={len(admin_state['agents'])}; "
                    "admin surface exposes launch, source, measurement, and warning summaries."
                ),
                "action": "Use the current admin observability surface rather than a separate sales dashboard.",
                "exit_criteria": "Operator can review launch readiness from the existing admin console.",
            },
            {
                "item_name": "buyer_environment_inputs",
                "label": "Buyer environment inputs",
                "owner": "Buyer implementation owner and platform operator",
                "sources": ["buyer environment URL", "deployment topology", "admin token handoff", "data retention decision"],
                "evidence_type": "buyer_environment_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_environment_required",
                "evidence": "Buyer deployment URL, topology, credentials handoff, retention, and network policy are not repo-local evidence.",
                "action": "Collect buyer environment details or attach explicit trial-scope waivers.",
                "exit_criteria": "Buyer provides environment inputs or agrees the launch is limited to repo-local/demo execution.",
            },
            {
                "item_name": "production_telemetry_inputs",
                "label": "Production telemetry inputs",
                "owner": "Operations and analytics owner",
                "sources": [
                    "/api/v1/commercial_operations_readiness/latest",
                    "/api/v1/analytics_snapshots/latest",
                    "production request logs",
                    "incident drill record",
                    "backup restore proof",
                ],
                "evidence_type": "proposed_until_production",
                "completion_state": "warning",
                "source_gap_status": "production_input_required",
                "evidence": (
                    f"operations_production_evidence_action_count="
                    f"{operations['operations_summary']['production_evidence_action_count']}; "
                    f"analytics_measurement_status={analytics['measurement_status']}"
                ),
                "action": "Capture production telemetry, SLO evidence, incident drill, and restore proof in the buyer environment.",
                "exit_criteria": "First production telemetry snapshot and operations proof are attached or explicitly waived.",
            },
            {
                "item_name": "commercial_signature_inputs",
                "label": "Commercial signature inputs",
                "owner": "Buyer sponsor, procurement owner, and deal owner",
                "sources": ["signed order/MSA", "DPA/security acceptance", "purchase order", "go-live authorization"],
                "evidence_type": "buyer_signature_required",
                "completion_state": "warning",
                "source_gap_status": "buyer_signature_required",
                "evidence": (
                    f"buyer_signature_gap_count={gtm['go_to_market_summary']['buyer_signature_gap_count']}; "
                    "signed order, DPA/security acceptance, budget/PO, and go-live authorization are buyer inputs."
                ),
                "action": "Collect signatures, approvals, or waivers before representing launch readiness as closed-won.",
                "exit_criteria": "Buyer accepts all signature, DPA/security, budget, and go-live inputs.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_launch_readiness.md", "docs/commercial_go_to_market_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review process delay is not a launch blocker unless a concrete failure is produced.",
                "action": "Continue launch readiness work while queued review processes are pending.",
                "exit_criteria": "Only concrete product, security, API contract, or document failures block launch readiness.",
            },
            {
                "item_name": "packaging_decision",
                "label": "Packaging decision",
                "owner": "Procurement and security reviewer",
                "sources": ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if gtm["library_split_decision"]["decision"] == "keep_single_product"
                else "warning",
                "evidence": gtm["library_split_decision"]["reason"],
                "action": "Keep one deployable enterprise control-plane product until extraction triggers are real.",
                "exit_criteria": "Do not create a separate library, Git submodule, or extracted package for this launch gate.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in launch_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        buyer_environment_gap_count = sum(
            1 for item in launch_items if item.get("source_gap_status") == "buyer_environment_required"
        )
        production_telemetry_gap_count = sum(
            1 for item in launch_items if item.get("source_gap_status") == "production_input_required"
        )
        commercial_signature_gap_count = sum(
            1 for item in launch_items if item.get("source_gap_status") == "buyer_signature_required"
        )
        external_input_group_count = (
            buyer_environment_gap_count + production_telemetry_gap_count + commercial_signature_gap_count
        )
        if blocked_count:
            launch_status = "commercial_launch_blocked"
        elif warning_count:
            launch_status = "commercial_launch_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            launch_status = "commercial_launch_ready"  # pragma: no cover

        return {
            "launch_status": launch_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_launch_readiness",
            "source_note": (
                "Commercial launch readiness packages repo-local GTM, runtime, acceptance, operator, admin, "
                "analytics, Figma, review-process, and packaging evidence separately from buyer environment, "
                "production telemetry, and commercial signature inputs; it is not a valuation guarantee, "
                "purchase commitment, signed order, legal opinion, production compliance certificate, or revenue proof."
            ),
            "launch_summary": {
                "item_count": len(launch_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "external_input_group_count": external_input_group_count,
                "buyer_environment_gap_count": buyer_environment_gap_count,
                "production_telemetry_gap_count": production_telemetry_gap_count,
                "commercial_signature_gap_count": commercial_signature_gap_count,
                "review_process_is_blocker": gtm["review_process_policy"]["is_blocker"],
            },
            "launch_items": launch_items,
            "concrete_blockers": concrete_blockers,
            "launch_status_rules": [
                {
                    "launch_status": "commercial_launch_ready",
                    "rule": "GTM, runtime, acceptance, operator, admin, buyer environment, production telemetry, commercial signature, review policy, and packaging evidence are ready",
                },
                {
                    "launch_status": "commercial_launch_ready_with_warnings",
                    "rule": "repo-local launch packet is ready while buyer environment, production telemetry, or commercial signature inputs remain explicit warnings",
                },
                {
                    "launch_status": "commercial_launch_blocked",
                    "rule": "missing local launch packet evidence, concrete product defect, API contract failure, document mismatch, security failure, or Code Connect usage blocks launch readiness",
                },
            ],
            "review_process_policy": gtm["review_process_policy"],
            "related_runtime_reports": {
                "commercial_go_to_market_status": gtm["go_to_market_status"],
                "commercial_operations_status": operations["operations_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_acceptance_status": acceptance["acceptance_status"],
                "analytics_measurement_status": analytics["measurement_status"],
                **gtm["related_runtime_reports"],
            },
            "library_split_decision": gtm["library_split_decision"],
            "plugin_traceability": gtm["plugin_traceability"],
            "launch_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_launch_readiness/latest",
                "documentation": "docs/commercial_launch_readiness.md",
            },
        }

    def commercial_completion_scorecard_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the final KRW 2B commercial completion scorecard."""
        commercial = self.commercial_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        gtm = self.commercial_go_to_market_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        launch = self.commercial_launch_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        concrete_blockers = list(launch["concrete_blockers"])
        if commercial["commercial_status"] == "not_commercial_ready":
            concrete_blockers.append("commercial_readiness_failed")
        if launch["launch_status"] == "commercial_launch_blocked":
            concrete_blockers.append("commercial_launch_blocked")
        concrete_blockers = list(dict.fromkeys(concrete_blockers))
        scorecard_items = [
            {
                "item_name": "product_design_evidence",
                "label": "Product Design evidence",
                "owner": "Product design owner",
                "sources": [
                    "docs/plugin_driven_design_brief.md",
                    "docs/commercial_plugin_operating_model.md",
                    "docs/screen_design.md",
                    "/admin",
                    "/api/v1/commercial_readiness/latest",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "docs/plugin_driven_design_brief.md",
                        "docs/commercial_plugin_operating_model.md",
                        "docs/screen_design.md",
                    )
                )
                and admin_state["agents"]
                else "blocked",
                "evidence": "Buyer, operator, security/compliance, and procurement workflows map to admin and readiness evidence.",
                "action": "Keep buyer evidence paths visible in the existing admin control plane.",
                "exit_criteria": "Every persona has a product/API/docs evidence path.",
            },
            {
                "item_name": "figma_artifacts",
                "label": "Figma artifacts",
                "owner": "Figma owner",
                "sources": [
                    "docs/figma_artifacts.md",
                    "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                    "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                ],
                "evidence_type": "figma_artifact",
                "completion_state": "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "evidence": "Editable design, FigJam diagrams, and stakeholder artifact records exist without Code Connect.",
                "action": "Use Figma/FigJam artifacts for stakeholder review without generating Code Connect metadata.",
                "exit_criteria": "Figma artifacts are recorded and Code Connect remains unused.",
            },
            {
                "item_name": "superpowers_plan_evidence",
                "label": "Superpowers plan evidence",
                "owner": "Implementation owner",
                "sources": [
                    "docs/superpowers/plans/2026-07-02-commercial-completion-scorecard-runtime.md",
                    "docs/superpowers/plans/2026-07-02-commercial-launch-readiness.md",
                    "tests/test_commercial_completion_scorecard.py",
                ],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if has_file("docs/superpowers/plans/2026-07-02-commercial-completion-scorecard-runtime.md")
                and has_file("tests/test_commercial_completion_scorecard.py")
                else "blocked",
                "evidence": "Dated plans and focused tests define files, expected failures, implementation, and verification commands.",
                "action": "Keep TDD plans and verification commands committed with the scorecard.",
                "exit_criteria": "Plan and focused test exist for the runtime scorecard.",
            },
            {
                "item_name": "ponytail_packaging_decision",
                "label": "Ponytail packaging decision",
                "owner": "Procurement and security reviewer",
                "sources": ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready"
                if launch["library_split_decision"]["decision"] == "keep_single_product"
                else "warning",
                "evidence": launch["library_split_decision"]["reason"],
                "action": "Keep one repository and one deployable product until extraction triggers are real.",
                "exit_criteria": "No separate library, Git submodule, or extracted package is created for this increment.",
            },
            {
                "item_name": "data_analytics_truthfulness",
                "label": "Data Analytics truthfulness",
                "owner": "Analytics owner",
                "sources": [
                    "docs/analytics_spec.md",
                    "/api/v1/analytics_snapshots/latest",
                    "/api/v1/commercial_launch_readiness/latest",
                ],
                "evidence_type": "measured_local",
                "completion_state": "ready"
                if has_file("docs/analytics_spec.md") and analytics["measurement_status"] == "local_runtime_snapshot"
                else "blocked",
                "evidence": "Measured local evidence and proposed production or buyer-specific inputs are separated.",
                "action": "Do not present proposed buyer or production inputs as measured product results.",
                "exit_criteria": "Every commercial KPI has an evidence type and source expectation.",
            },
            {
                "item_name": "runtime_endpoint_chain",
                "label": "Runtime endpoint chain",
                "owner": "Platform operator",
                "sources": [
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_go_to_market_readiness/latest",
                    "/api/v1/commercial_launch_readiness/latest",
                    "/api/v1/commercial_completion_scorecards/latest",
                ],
                "evidence_type": "repository_and_runtime_artifact",
                "completion_state": "ready"
                if commercial["commercial_status"] != "not_commercial_ready"
                and gtm["go_to_market_status"] != "commercial_go_to_market_blocked"
                and launch["launch_status"] != "commercial_launch_blocked"
                else "blocked",
                "evidence": (
                    f"commercial_status={commercial['commercial_status']}; "
                    f"go_to_market_status={gtm['go_to_market_status']}; "
                    f"launch_status={launch['launch_status']}"
                ),
                "action": "Expose completion status through the same admin-protected runtime API chain.",
                "exit_criteria": "Runtime chain has no blocked local evidence gate.",
            },
            {
                "item_name": "verification_packet",
                "label": "Verification packet",
                "owner": "Technical reviewer",
                "sources": [
                    "tests/test_commercial_completion_scorecard.py",
                    "tests/test_commercial_launch_readiness.py",
                    "tests/test_plugin_driven_artifacts.py",
                    "tests/test_api_contract.py",
                    "pytest -q",
                ],
                "evidence_type": "measured_local",
                "completion_state": "ready"
                if all(
                    has_file(path)
                    for path in (
                        "tests/test_commercial_completion_scorecard.py",
                        "tests/test_commercial_launch_readiness.py",
                        "tests/test_plugin_driven_artifacts.py",
                        "tests/test_api_contract.py",
                    )
                )
                else "blocked",
                "evidence": "Focused completion, launch, artifact, and API contract tests are present.",
                "action": "Run focused tests and full pytest before presenting completion status.",
                "exit_criteria": "Focused tests, compileall, full pytest, and diff hygiene pass.",
            },
            {
                "item_name": "review_process_policy",
                "label": "Review process policy",
                "owner": "Deal owner",
                "sources": ["docs/commercial_completion_scorecard.md", "docs/commercial_launch_readiness.md"],
                "evidence_type": "repository_artifact",
                "completion_state": "ready",
                "evidence": "Review delay, model-review delay, and queued review automation are not product blockers.",
                "action": "Block only on concrete security, API contract, document, or functional defects.",
                "exit_criteria": "Review process delay remains non-blocking without concrete failure evidence.",
            },
            {
                "item_name": "production_buyer_followups",
                "label": "Production and buyer follow-ups",
                "owner": "Buyer sponsor, operations owner, and deal owner",
                "sources": [
                    "/api/v1/commercial_launch_readiness/latest",
                    "buyer environment",
                    "production telemetry",
                    "commercial signatures",
                ],
                "evidence_type": "external_input_required",
                "completion_state": "warning",
                "source_gap_status": "external_input_required",
                "evidence": (
                    f"external_input_group_count={launch['launch_summary']['external_input_group_count']}; "
                    f"buyer_signature_gap_count={gtm['go_to_market_summary']['buyer_signature_gap_count']}"
                ),
                "action": "Collect buyer environment, production telemetry, and signature inputs or explicit waivers.",
                "exit_criteria": "Buyer supplies or waives remaining external inputs.",
            },
        ]
        state_counts = Counter(item["completion_state"] for item in scorecard_items)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            completion_status = "commercial_completion_blocked"
        elif warning_count:
            completion_status = "commercial_completion_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            completion_status = "commercial_completion_ready"  # pragma: no cover

        return {
            "completion_status": completion_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_completion_scorecard",
            "source_note": (
                "Commercial completion scorecard aggregates repo-local product design, Figma, Superpowers, "
                "Ponytail, Data Analytics, runtime, verification, review-process, and packaging evidence "
                "separately from buyer and production follow-ups; it is not a valuation guarantee, purchase "
                "commitment, signed order, legal opinion, production compliance certificate, or revenue proof."
            ),
            "completion_summary": {
                "item_count": len(scorecard_items),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "external_input_group_count": launch["launch_summary"]["external_input_group_count"],
                "review_process_is_blocker": launch["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "completion_items": scorecard_items,
            "concrete_blockers": concrete_blockers,
            "completion_status_rules": [
                {
                    "completion_status": "commercial_completion_ready",
                    "rule": "product design, Figma, Superpowers, Ponytail, Data Analytics, runtime, verification, review policy, packaging, and external inputs are ready",
                },
                {
                    "completion_status": "commercial_completion_ready_with_warnings",
                    "rule": "repo-local program completion evidence is ready while buyer environment, production telemetry, commercial signatures, or other external inputs remain explicit warnings",
                },
                {
                    "completion_status": "commercial_completion_blocked",
                    "rule": "security failure, API contract regression, document mismatch, reproducible product defect, missing local completion evidence, or Code Connect usage blocks completion",
                },
            ],
            "review_process_policy": launch["review_process_policy"],
            "related_runtime_reports": {
                "commercial_readiness_status": commercial["commercial_status"],
                "commercial_go_to_market_status": gtm["go_to_market_status"],
                "commercial_launch_status": launch["launch_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": launch["library_split_decision"],
            "plugin_traceability": launch["plugin_traceability"],
            "completion_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_completion_scorecards/latest",
                "documentation": "docs/commercial_completion_scorecard.md",
            },
        }

    def commercial_buyer_acceptance_workflow_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return owner-scoped buyer acceptance workflow evidence."""
        acceptance = self.commercial_acceptance_check_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        completion = self.commercial_completion_scorecard_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        handoff = self.commercial_handoff_bundle_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def step(
            step_name: str,
            label: str,
            owner: str,
            sources: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            decision_rule: str,
            next_action: str,
        ) -> dict[str, Any]:
            require_object_name(step_name, "buyer_acceptance_workflow.step_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("buyer acceptance workflow state must be ready, warning, or blocked")
            return {
                "step_name": step_name,
                "label": label,
                "owner": owner,
                "sources": sources,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "decision_rule": decision_rule,
                "next_action": next_action,
            }

        concrete_blockers = list(dict.fromkeys(acceptance["concrete_blockers"] + completion["concrete_blockers"]))
        acceptance_blocked = acceptance["acceptance_status"] == "commercial_acceptance_blocked"
        completion_blocked = completion["completion_status"] == "commercial_completion_blocked"
        local_runtime_state = "blocked" if acceptance_blocked or completion_blocked or concrete_blockers else "ready"
        workflow_steps = [
            step(
                "confirm_product_scope",
                "Confirm product scope",
                "Product owner",
                ["README.md", "docs/product_planning.md", "docs/commercial_readiness.md"],
                "repository_artifact",
                "ready" if all_files("README.md", "docs/product_planning.md", "docs/commercial_readiness.md") else "blocked",
                "Product remains one enterprise orchestration control plane.",
                "Proceed when the buyer reviews one compatible API plus one admin evidence surface.",
                "Restore product scope docs before buyer acceptance.",
            ),
            step(
                "confirm_integration_surface",
                "Confirm integration surface",
                "Platform reviewer",
                ["/v1/chat/completions", "docs/rest_api_design.md", "tests/test_api_contract.py"],
                "repository_artifact",
                "ready" if all_files("docs/rest_api_design.md", "tests/test_api_contract.py") else "blocked",
                "OpenAI-compatible API and API contract tests are present.",
                "Proceed when API compatibility evidence is present and tests pass.",
                "Restore REST docs or API contract tests before buyer acceptance.",
            ),
            step(
                "confirm_operator_evidence",
                "Confirm operator evidence",
                "Platform operator",
                ["/admin", "/admin/state", "docs/screen_design.md", "contextual_orchestrator/admin.py"],
                "repository_artifact",
                "ready" if all_files("docs/screen_design.md", "contextual_orchestrator/admin.py") else "blocked",
                "Admin console exposes operator evidence for pool, policy, trace, access, replay, analytics, and readiness.",
                "Proceed when operator state and commercial readiness surfaces are visible.",
                "Restore admin evidence docs or implementation before buyer acceptance.",
            ),
            step(
                "confirm_readiness_endpoints",
                "Confirm readiness endpoints",
                "Product owner",
                [
                    "/api/v1/sales_readiness/latest",
                    "/api/v1/commercial_readiness/latest",
                    "/api/v1/commercial_acceptance_checks/latest",
                    "/api/v1/commercial_completion_scorecards/latest",
                ],
                "measured_local",
                local_runtime_state,
                (
                    f"commercial_acceptance_status={acceptance['acceptance_status']}; "
                    f"commercial_completion_status={completion['completion_status']}"
                ),
                "Proceed when local runtime gates have no concrete blockers.",
                "Resolve blocked readiness, acceptance, or completion gates before buyer acceptance.",
            ),
            step(
                "confirm_security_posture",
                "Confirm security posture",
                "Security reviewer",
                ["SECURITY.md", "tests/test_security_hardening.py", ".github/workflows/security.yml"],
                "repository_artifact",
                "ready" if all_files("SECURITY.md", "tests/test_security_hardening.py", ".github/workflows/security.yml") else "blocked",
                "Security policy, hardening tests, and hosted security workflow metadata are present.",
                "Proceed when concrete security failures are absent.",
                "Fix concrete security failures; queued security checks alone are not blockers.",
            ),
            step(
                "confirm_metric_honesty",
                "Confirm metric honesty",
                "Analytics reviewer",
                ["/api/v1/analytics_snapshots/latest", "docs/analytics_spec.md"],
                "measured_local",
                "ready" if has_file("docs/analytics_spec.md") and analytics["measurement_status"] == "local_runtime_snapshot" else "blocked",
                "Analytics spec and local snapshot separate measured local evidence from proposed production or buyer inputs.",
                "Proceed when measured and proposed claims are not mixed.",
                "Restore analytics source labels before buyer acceptance.",
            ),
            step(
                "confirm_visual_review_path",
                "Confirm visual review path",
                "Stakeholder reviewer",
                ["docs/figma_artifacts.md", "Figma design file", "FigJam board", "Figma Slides deck"],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "Editable design, FigJam, and stakeholder artifacts are recorded without Code Connect.",
                "Proceed when visual artifacts are available for stakeholder review.",
                "Record Figma artifacts before buyer acceptance.",
            ),
            step(
                "confirm_packaging_decision",
                "Confirm packaging decision",
                "Procurement reviewer",
                ["docs/library_research.md", "docs/commercial_plugin_operating_model.md"],
                "repository_artifact",
                "ready" if completion["library_split_decision"]["decision"] == "keep_single_product" else "warning",
                completion["library_split_decision"]["reason"],
                "Proceed with one repository and one deployable product.",
                "Extract only after a second product, independent release cadence, or provenance trigger exists.",
            ),
            step(
                "confirm_production_inputs",
                "Confirm production inputs",
                "Operations owner",
                ["production telemetry", "support plan", "SLO evidence", "incident drill"],
                "proposed_until_production",
                "warning",
                "Production telemetry, support, SLO, and incident evidence require a deployment or paid onboarding environment.",
                "Proceed with warning when production inputs are explicitly caveated.",
                "Collect production evidence during buyer onboarding or mark an explicit waiver.",
            ),
            step(
                "confirm_buyer_specific_inputs",
                "Confirm buyer-specific inputs",
                "Buyer and account team",
                ["ROI model", "security questionnaire", "legal review", "deployment target"],
                "proposed_until_buyer_specific",
                "warning",
                "ROI, legal, procurement, and deployment inputs require a named buyer.",
                "Proceed with warning when buyer-specific inputs are explicit follow-ups.",
                "Collect buyer-specific inputs during account diligence.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in workflow_steps)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            workflow_status = "buyer_acceptance_workflow_blocked"
        elif warning_count:
            workflow_status = "buyer_acceptance_workflow_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            workflow_status = "buyer_acceptance_workflow_ready"  # pragma: no cover

        return {
            "workflow_status": workflow_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_buyer_acceptance_workflow",
            "source_note": (
                "Commercial buyer acceptance workflow maps runbook owners, runtime evidence, "
                "Figma artifacts, analytics truthfulness, review-process policy, and packaging "
                "decision into Go, Warning, and No-Go steps; it is not a valuation guarantee, "
                "purchase commitment, signed order, legal opinion, production compliance certificate, "
                "or revenue proof."
            ),
            "workflow_summary": {
                "step_count": len(workflow_steps),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "production_follow_up_count": sum(
                    1 for item in workflow_steps if item["evidence_type"] == "proposed_until_production"
                ),
                "buyer_specific_follow_up_count": sum(
                    1 for item in workflow_steps if item["evidence_type"] == "proposed_until_buyer_specific"
                ),
                "review_process_is_blocker": acceptance["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "acceptance_steps": workflow_steps,
            "concrete_blockers": concrete_blockers,
            "go_warning_no_go_rules": [
                {
                    "workflow_status": "buyer_acceptance_workflow_ready",
                    "rule": "all acceptance owners have ready evidence and no external production or buyer-specific inputs remain open",
                },
                {
                    "workflow_status": "buyer_acceptance_workflow_ready_with_warnings",
                    "rule": "repo-local buyer acceptance evidence is ready while production or buyer-specific inputs remain explicit warnings",
                },
                {
                    "workflow_status": "buyer_acceptance_workflow_blocked",
                    "rule": "security failure, API contract regression, document mismatch, reproducible product defect, missing acceptance path, or Code Connect usage blocks acceptance",
                },
            ],
            "review_process_policy": acceptance["review_process_policy"],
            "related_runtime_reports": {
                "commercial_acceptance_status": acceptance["acceptance_status"],
                "commercial_completion_status": completion["completion_status"],
                "buyer_handoff_status": handoff["bundle_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": completion["library_split_decision"],
            "plugin_traceability": completion["plugin_traceability"],
            "workflow_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_buyer_acceptance_workflows/latest",
                "documentation": "docs/commercial_buyer_acceptance_runbook.md",
            },
        }

    def commercial_demo_scenario_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return buyer-demo scenarios for the KRW 2B completion standard."""
        completion = self.commercial_completion_scorecard_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        buyer_workflow = self.commercial_buyer_acceptance_workflow_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def step(
            step_name: str,
            label: str,
            persona: str,
            sources: list[str],
            runtime_endpoints: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            demo_action: str,
            expected_evidence: str,
        ) -> dict[str, Any]:
            require_object_name(step_name, "commercial_demo.step_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("commercial demo step state must be ready, warning, or blocked")
            return {
                "step_name": step_name,
                "label": label,
                "persona": persona,
                "sources": sources,
                "runtime_endpoints": runtime_endpoints,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "demo_action": demo_action,
                "expected_evidence": expected_evidence,
            }

        concrete_blockers = list(
            dict.fromkeys(completion["concrete_blockers"] + buyer_workflow["concrete_blockers"])
        )
        local_runtime_state = (
            "blocked"
            if completion["completion_status"] == "commercial_completion_blocked"
            or buyer_workflow["workflow_status"] == "buyer_acceptance_workflow_blocked"
            or concrete_blockers
            else "ready"
        )
        event_counts = analytics.get("event_counts", {})
        recent_runs = admin_state.get("recent_workflow_runs", [])
        demo_steps = [
            step(
                "compatible_api_smoke",
                "Compatible API smoke",
                "Economic buyer",
                ["README.md", "docs/rest_api_design.md", "/v1/chat/completions"],
                ["/v1/chat/completions"],
                "measured_local",
                "ready"
                if all_files("README.md", "docs/rest_api_design.md")
                and event_counts.get("chat_completion_requested", 0) > 0
                else "blocked",
                f"successful_chat_completion_events={event_counts.get('chat_completion_requested', 0)}",
                "Run the OpenAI-compatible chat completion call used by the buyer application.",
                "The buyer sees a compatible response and the runtime records a local analytics event.",
            ),
            step(
                "conducted_workflow_trace",
                "Conducted workflow trace",
                "Platform operator",
                ["/admin", "/api/v1/workflow_runs", "docs/screen_design.md"],
                ["/admin", "/api/v1/workflow_runs"],
                "measured_local",
                "ready" if recent_runs and has_file("docs/screen_design.md") else "blocked",
                f"recent_workflow_run_count={len(recent_runs)}",
                "Open the admin trace view for a conducted workflow run.",
                "The operator can inspect mode, policy mode, selected agents, and run trace evidence.",
            ),
            step(
                "access_list_inspection",
                "Access-list inspection",
                "Compliance reviewer",
                ["docs/product_planning.md", "/api/v1/access_reports/{workflow_run_id}", "/admin"],
                ["/api/v1/access_reports/{workflow_run_id}", "/admin"],
                "repository_and_runtime_artifact",
                "ready" if has_file("docs/product_planning.md") else "blocked",
                "access reports are scoped to workflow_run_id and exposed through the admin surface",
                "Open the access report for the conducted workflow run.",
                "The reviewer sees why each agent had access to context, tools, and trace evidence.",
            ),
            step(
                "evaluation_replay",
                "Evaluation replay",
                "Quality reviewer",
                ["docs/screen_design.md", "/api/v1/evaluation_runs", "/admin"],
                ["/api/v1/evaluation_runs", "/admin"],
                "measured_local",
                "ready" if event_counts.get("evaluation_run_created", 0) > 0 else "blocked",
                f"evaluation_run_created_events={event_counts.get('evaluation_run_created', 0)}",
                "Replay the buyer prompt through the evaluation endpoint.",
                "The reviewer sees replay status and trace-backed verification evidence.",
            ),
            step(
                "admin_readiness_console",
                "Admin readiness console",
                "Economic buyer",
                [
                    "/admin",
                    "/api/v1/commercial_completion_scorecards/latest",
                    "/api/v1/commercial_buyer_acceptance_workflows/latest",
                    "/api/v1/commercial_demo_scenarios/latest",
                ],
                [
                    "/admin",
                    "/api/v1/commercial_completion_scorecards/latest",
                    "/api/v1/commercial_buyer_acceptance_workflows/latest",
                    "/api/v1/commercial_demo_scenarios/latest",
                ],
                "repository_and_runtime_artifact",
                local_runtime_state,
                (
                    f"commercial_completion_status={completion['completion_status']}; "
                    f"buyer_acceptance_workflow_status={buyer_workflow['workflow_status']}"
                ),
                "Show the readiness card chain in the admin console.",
                "The buyer sees completion, buyer acceptance, and demo status without leaving one control plane.",
            ),
            step(
                "metric_truthfulness",
                "Metric truthfulness",
                "Compliance reviewer",
                ["docs/analytics_spec.md", "/api/v1/analytics_snapshots/latest"],
                ["/api/v1/analytics_snapshots/latest"],
                "measured_local",
                "ready"
                if has_file("docs/analytics_spec.md") and analytics["measurement_status"] == "local_runtime_snapshot"
                else "blocked",
                "measured local metrics remain separate from proposed production and buyer-specific metrics",
                "Review the analytics spec and local analytics endpoint.",
                "The reviewer can distinguish measured runtime data from proposed KPI definitions.",
            ),
            step(
                "figma_stakeholder_review",
                "Figma stakeholder review",
                "Stakeholder reviewer",
                [
                    "docs/figma_artifacts.md",
                    "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                    "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                ],
                [],
                "figma_artifact",
                "ready" if has_file("docs/figma_artifacts.md") else "blocked",
                "editable Figma and FigJam stakeholder artifacts are recorded without Code Connect",
                "Walk through the design file and FigJam diagram packet.",
                "Stakeholders review the product narrative, admin surface, and runtime flow as editable artifacts.",
            ),
            step(
                "buyer_acceptance_decision",
                "Buyer acceptance decision",
                "Economic buyer",
                [
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "/api/v1/commercial_buyer_acceptance_workflows/latest",
                ],
                ["/api/v1/commercial_buyer_acceptance_workflows/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state,
                f"buyer_acceptance_workflow_status={buyer_workflow['workflow_status']}",
                "Use the buyer acceptance workflow as the Go/Warning/No-Go decision record.",
                "The buyer can sign off on repo-local evidence while external follow-ups stay explicit.",
            ),
            step(
                "production_buyer_followups",
                "Production and buyer follow-ups",
                "Economic buyer",
                ["production telemetry", "ROI model", "security questionnaire", "support plan"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Production telemetry, named-buyer ROI, security questionnaire, and support plan require buyer input.",
                "Capture buyer-specific production, ROI, legal, and support inputs after the local demo.",
                "External inputs are tracked as warnings, not hidden as measured product evidence.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in demo_steps)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            demo_status = "commercial_demo_blocked"
        elif warning_count:
            demo_status = "commercial_demo_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            demo_status = "commercial_demo_ready"  # pragma: no cover
        required_runtime_endpoints = list(
            dict.fromkeys(
                endpoint
                for item in demo_steps
                for endpoint in item["runtime_endpoints"]
                if endpoint.startswith("/")
            )
        )

        return {
            "demo_status": demo_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_demo_scenarios",
            "source_note": (
                "Commercial demo scenarios package repo-local runtime, admin, analytics, Figma, "
                "buyer acceptance, review-policy, and packaging evidence for KRW 2,000,000,000 "
                "saleability review; it is not a valuation guarantee, purchase commitment, "
                "signed order, legal opinion, production compliance certificate, or revenue proof."
            ),
            "demo_narrative": {
                "title": "KRW 2B commercial control-plane buyer demo",
                "promise": (
                    "Show one enterprise orchestration control plane with a compatible inference API, "
                    "operator/admin evidence, trace and access visibility, evaluation replay, and truthful metrics."
                ),
                "audience": [
                    "Economic buyer",
                    "Platform operator",
                    "Compliance reviewer",
                    "Quality reviewer",
                    "Stakeholder reviewer",
                ],
            },
            "demo_summary": {
                "step_count": len(demo_steps),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "persona_count": len({item["persona"] for item in demo_steps}),
                "endpoint_count": len(required_runtime_endpoints),
                "review_process_is_blocker": completion["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "demo_steps": demo_steps,
            "required_runtime_endpoints": required_runtime_endpoints,
            "concrete_blockers": concrete_blockers,
            "demo_status_rules": [
                {
                    "demo_status": "commercial_demo_ready",
                    "rule": "all demo steps are ready and no production or buyer-specific follow-ups remain open",
                },
                {
                    "demo_status": "commercial_demo_ready_with_warnings",
                    "rule": "repo-local demo evidence is ready while production, ROI, legal, support, or buyer-specific inputs remain explicit warnings",
                },
                {
                    "demo_status": "commercial_demo_blocked",
                    "rule": "security failure, API contract regression, document mismatch, runtime defect, missing local demo evidence, or Code Connect usage blocks the demo",
                },
            ],
            "review_process_policy": completion["review_process_policy"],
            "related_runtime_reports": {
                "commercial_completion_status": completion["completion_status"],
                "buyer_acceptance_workflow_status": buyer_workflow["workflow_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": completion["library_split_decision"],
            "plugin_traceability": completion["plugin_traceability"],
            "demo_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_demo_scenarios/latest",
                "documentation": "docs/commercial_demo_scenarios.md",
            },
        }

    def commercial_proposal_packet_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return buyer proposal sections for the KRW 2B saleability standard."""
        completion = self.commercial_completion_scorecard_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        demo = self.commercial_demo_scenario_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        buyer_workflow = self.commercial_buyer_acceptance_workflow_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def section(
            section_name: str,
            label: str,
            owner: str,
            sources: list[str],
            runtime_endpoints: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            buyer_message: str,
            next_action: str,
        ) -> dict[str, Any]:
            require_object_name(section_name, "commercial_proposal.section_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("commercial proposal section state must be ready, warning, or blocked")
            return {
                "section_name": section_name,
                "label": label,
                "owner": owner,
                "sources": sources,
                "runtime_endpoints": runtime_endpoints,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "buyer_message": buyer_message,
                "next_action": next_action,
            }

        concrete_blockers = list(
            dict.fromkeys(
                completion["concrete_blockers"]
                + demo["concrete_blockers"]
                + buyer_workflow["concrete_blockers"]
            )
        )
        local_runtime_state = (
            "blocked"
            if completion["completion_status"] == "commercial_completion_blocked"
            or demo["demo_status"] == "commercial_demo_blocked"
            or buyer_workflow["workflow_status"] == "buyer_acceptance_workflow_blocked"
            or concrete_blockers
            else "ready"
        )
        proposal_sections = [
            section(
                "executive_summary",
                "Executive summary",
                "Deal owner",
                ["README.md", "docs/commercial_completion_scorecard.md", "/api/v1/commercial_completion_scorecards/latest"],
                ["/api/v1/commercial_completion_scorecards/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state if all_files("README.md", "docs/commercial_completion_scorecard.md") else "blocked",
                f"commercial_completion_status={completion['completion_status']}",
                "One enterprise orchestration control plane is ready for a KRW 2B buyer review with explicit caveats.",
                "Use the completion scorecard as the proposal cover evidence.",
            ),
            section(
                "product_scope",
                "Product scope",
                "Product owner",
                ["docs/product_planning.md", "docs/commercial_plugin_operating_model.md", "docs/library_research.md"],
                [],
                "repository_artifact",
                "ready"
                if all_files("docs/product_planning.md", "docs/commercial_plugin_operating_model.md", "docs/library_research.md")
                and completion["library_split_decision"]["decision"] == "keep_single_product"
                else "blocked",
                completion["library_split_decision"]["reason"],
                "The offer is one compatible API plus one admin evidence surface, not a split product suite.",
                "Keep proposal language centered on a single deployable control plane.",
            ),
            section(
                "buyer_value_case",
                "Buyer value case",
                "Economic buyer",
                ["docs/commercial_value_readiness.md", "docs/analytics_spec.md", "/api/v1/commercial_value_readiness/latest"],
                ["/api/v1/commercial_value_readiness/latest", "/api/v1/analytics_snapshots/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if value["value_status"] != "commercial_value_blocked"
                and has_file("docs/commercial_value_readiness.md")
                and analytics["measurement_status"] == "local_runtime_snapshot"
                else "blocked",
                f"value_status={value['value_status']}; analytics_measurement_status={analytics['measurement_status']}",
                "Measured local value evidence is separated from buyer-specific ROI inputs.",
                "Attach buyer ROI assumptions only after buyer discovery validates them.",
            ),
            section(
                "demo_and_acceptance_path",
                "Demo and acceptance path",
                "Product design owner",
                [
                    "docs/commercial_demo_scenarios.md",
                    "docs/commercial_buyer_acceptance_runbook.md",
                    "/api/v1/commercial_demo_scenarios/latest",
                    "/api/v1/commercial_buyer_acceptance_workflows/latest",
                ],
                ["/api/v1/commercial_demo_scenarios/latest", "/api/v1/commercial_buyer_acceptance_workflows/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state
                if all_files("docs/commercial_demo_scenarios.md", "docs/commercial_buyer_acceptance_runbook.md")
                else "blocked",
                f"commercial_demo_status={demo['demo_status']}; buyer_acceptance_workflow_status={buyer_workflow['workflow_status']}",
                "Buyer review can move from demo script to Go/Warning/No-Go acceptance without leaving the control plane.",
                "Run the demo packet and record the acceptance decision.",
            ),
            section(
                "technical_evidence",
                "Technical evidence",
                "Platform reviewer",
                ["docs/rest_api_design.md", "docs/screen_design.md", "contextual_orchestrator/api_contract.py", "/admin"],
                ["/v1/chat/completions", "/admin", "/api/v1/workflow_runs", "/api/v1/access_reports/{workflow_run_id}"],
                "repository_and_runtime_artifact",
                "ready"
                if all_files("docs/rest_api_design.md", "docs/screen_design.md", "contextual_orchestrator/api_contract.py")
                and admin_state["agents"]
                else "blocked",
                f"agent_count={len(admin_state['agents'])}; recent_workflow_run_count={len(admin_state['recent_workflow_runs'])}",
                "The buyer can verify API compatibility, trace evidence, and access-list evidence.",
                "Include endpoint list and admin review screenshots or live walkthrough in the proposal.",
            ),
            section(
                "security_and_compliance",
                "Security and compliance",
                "Security reviewer",
                ["SECURITY.md", "docs/commercial_security_attestation.md", "/api/v1/commercial_security_attestations/latest"],
                ["/api/v1/commercial_security_attestations/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and all_files("SECURITY.md", "docs/commercial_security_attestation.md")
                else "blocked",
                f"security_attestation_status={security['security_attestation_status']}",
                "Repo-local security evidence is present while external attestation and buyer DPA inputs stay explicit.",
                "Do not claim third-party attestation until supplied.",
            ),
            section(
                "implementation_and_operations",
                "Implementation and operations",
                "Operations owner",
                [
                    "docs/commercial_onboarding_readiness.md",
                    "docs/commercial_operations_readiness.md",
                    "/api/v1/commercial_onboarding_readiness/latest",
                    "/api/v1/commercial_operations_readiness/latest",
                ],
                ["/api/v1/commercial_onboarding_readiness/latest", "/api/v1/commercial_operations_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and operations["operations_status"] != "commercial_operations_blocked"
                and all_files("docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md")
                else "blocked",
                f"onboarding_status={onboarding['onboarding_status']}; operations_status={operations['operations_status']}",
                "Implementation and operations evidence is proposal-ready with production environment follow-ups separated.",
                "Convert buyer environment details into an onboarding checklist after selection.",
            ),
            section(
                "proposal_review_packet",
                "Proposal review packet",
                "Stakeholder reviewer",
                [
                    "docs/commercial_proposal_packet.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-proposal-packet-runtime.md",
                ],
                ["/api/v1/commercial_proposal_packets/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if all_files(
                    "docs/commercial_proposal_packet.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-proposal-packet-runtime.md",
                )
                else "blocked",
                "Proposal docs, FigJam artifact record, and implementation plan are committed as repo artifacts.",
                "Stakeholders can review the proposal packet as docs, runtime JSON, and FigJam flow.",
                "Keep Figma Code Connect out of the proposal workflow.",
            ),
            section(
                "commercial_terms_followups",
                "Commercial terms follow-ups",
                "Deal owner",
                ["docs/commercial_contract_readiness.md", "buyer order form", "legal review", "pricing approval"],
                ["/api/v1/commercial_contract_readiness/latest"],
                "proposed_until_buyer_specific",
                "warning",
                f"contract_status={contract['contract_status']}",
                "Order-form, legal, pricing approval, and signature inputs need named-buyer review.",
                "Collect buyer-specific legal and commercial terms or record explicit waiver.",
            ),
            section(
                "production_buyer_inputs",
                "Production and buyer inputs",
                "Buyer sponsor",
                ["production telemetry", "buyer ROI model", "support plan", "security questionnaire"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Production telemetry, buyer ROI model, support plan, and security questionnaire are external inputs.",
                "The proposal is locally ready while buyer-specific inputs remain caveated.",
                "Collect external evidence during proposal negotiation or paid onboarding.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in proposal_sections)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            proposal_status = "commercial_proposal_blocked"
        elif warning_count:
            proposal_status = "commercial_proposal_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            proposal_status = "commercial_proposal_ready"  # pragma: no cover
        required_runtime_endpoints = list(
            dict.fromkeys(
                endpoint
                for item in proposal_sections
                for endpoint in item["runtime_endpoints"]
                if endpoint.startswith("/")
            )
        )

        return {
            "proposal_status": proposal_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_proposal_packet",
            "source_note": (
                "Commercial proposal packet packages repo-local completion, demo, acceptance, value, "
                "security, contract, onboarding, operations, analytics, Figma, review-policy, and packaging "
                "evidence for KRW 2,000,000,000 buyer proposal review; it is not a valuation guarantee, "
                "purchase commitment, signed order, legal opinion, production compliance certificate, or revenue proof."
            ),
            "proposal_narrative": {
                "title": "KRW 2B commercial buyer proposal packet",
                "promise": (
                    "Present one enterprise orchestration control plane with compatible API integration, "
                    "operator evidence, buyer demo path, acceptance workflow, and truthful commercial caveats."
                ),
                "audience": [
                    "Economic buyer",
                    "Platform reviewer",
                    "Security reviewer",
                    "Operations owner",
                    "Stakeholder reviewer",
                    "Deal owner",
                ],
            },
            "proposal_summary": {
                "section_count": len(proposal_sections),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "endpoint_count": len(required_runtime_endpoints),
                "review_process_is_blocker": completion["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "proposal_sections": proposal_sections,
            "required_runtime_endpoints": required_runtime_endpoints,
            "concrete_blockers": concrete_blockers,
            "proposal_status_rules": [
                {
                    "proposal_status": "commercial_proposal_ready",
                    "rule": "all proposal sections are ready and no buyer-specific commercial or production inputs remain open",
                },
                {
                    "proposal_status": "commercial_proposal_ready_with_warnings",
                    "rule": "repo-local proposal evidence is ready while pricing, legal, ROI, production, support, or signature inputs remain explicit warnings",
                },
                {
                    "proposal_status": "commercial_proposal_blocked",
                    "rule": "security failure, API contract regression, document mismatch, runtime defect, missing local proposal evidence, or Code Connect usage blocks the proposal",
                },
            ],
            "review_process_policy": completion["review_process_policy"],
            "related_runtime_reports": {
                "commercial_completion_status": completion["completion_status"],
                "commercial_demo_status": demo["demo_status"],
                "buyer_acceptance_workflow_status": buyer_workflow["workflow_status"],
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_contract_status": contract["contract_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_operations_status": operations["operations_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": completion["library_split_decision"],
            "plugin_traceability": completion["plugin_traceability"],
            "proposal_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_proposal_packets/latest",
                "documentation": "docs/commercial_proposal_packet.md",
            },
        }

    def commercial_purchase_approval_packet_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return buyer-side purchase approval gates for the KRW 2B standard."""
        proposal = self.commercial_proposal_packet_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        close = self.commercial_close_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        procurement = self.commercial_procurement_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def gate(
            gate_name: str,
            label: str,
            owner: str,
            sources: list[str],
            runtime_endpoints: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            approval_question: str,
            next_action: str,
        ) -> dict[str, Any]:
            require_object_name(gate_name, "commercial_purchase_approval.gate_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("commercial purchase approval gate state must be ready, warning, or blocked")
            return {
                "gate_name": gate_name,
                "label": label,
                "owner": owner,
                "sources": sources,
                "runtime_endpoints": runtime_endpoints,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "approval_question": approval_question,
                "next_action": next_action,
            }

        concrete_blockers = list(dict.fromkeys(proposal["concrete_blockers"] + close["concrete_blockers"]))
        local_runtime_state = (
            "blocked"
            if proposal["proposal_status"] == "commercial_proposal_blocked"
            or close["close_status"] == "commercial_close_blocked"
            or concrete_blockers
            else "ready"
        )
        approval_gates = [
            gate(
                "proposal_packet_ready",
                "Proposal packet ready",
                "Deal owner",
                ["docs/commercial_proposal_packet.md", "/api/v1/commercial_proposal_packets/latest"],
                ["/api/v1/commercial_proposal_packets/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state if has_file("docs/commercial_proposal_packet.md") else "blocked",
                f"commercial_proposal_status={proposal['proposal_status']}",
                "Can the buyer review one coherent proposal packet?",
                "Use the proposal packet as the approval cover artifact.",
            ),
            gate(
                "procurement_path_ready",
                "Procurement path ready",
                "Procurement owner",
                ["docs/commercial_procurement_readiness.md", "/api/v1/commercial_procurement_readiness/latest"],
                ["/api/v1/commercial_procurement_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if procurement["procurement_status"] != "commercial_procurement_blocked"
                and has_file("docs/commercial_procurement_readiness.md")
                else "blocked",
                f"commercial_procurement_status={procurement['procurement_status']}",
                "Can procurement validate license, rights, distribution, admin, and caveat evidence?",
                "Route the packet to procurement with buyer-specific inputs still marked as warnings.",
            ),
            gate(
                "contract_legal_packet_ready",
                "Contract and legal packet ready",
                "Legal owner",
                ["docs/commercial_contract_readiness.md", "/api/v1/commercial_contract_readiness/latest"],
                ["/api/v1/commercial_contract_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if contract["contract_status"] != "commercial_contract_blocked"
                and has_file("docs/commercial_contract_readiness.md")
                else "blocked",
                f"commercial_contract_status={contract['contract_status']}",
                "Can legal review support, privacy, audit, license, and order-form obligations?",
                "Collect final legal edits and buyer order-form fields outside the local runtime claim.",
            ),
            gate(
                "financial_value_case_ready",
                "Financial value case ready",
                "Economic buyer",
                ["docs/commercial_value_readiness.md", "docs/analytics_spec.md", "/api/v1/commercial_value_readiness/latest"],
                ["/api/v1/commercial_value_readiness/latest", "/api/v1/analytics_snapshots/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if value["value_status"] != "commercial_value_blocked"
                and all_files("docs/commercial_value_readiness.md", "docs/analytics_spec.md")
                and analytics["measurement_status"] == "local_runtime_snapshot"
                else "blocked",
                f"commercial_value_status={value['value_status']}; analytics_measurement_status={analytics['measurement_status']}",
                "Can finance separate measured local evidence from buyer ROI assumptions?",
                "Attach buyer ROI and payback assumptions only after buyer discovery.",
            ),
            gate(
                "security_acceptance_ready",
                "Security acceptance ready",
                "Security owner",
                ["SECURITY.md", "docs/commercial_security_attestation.md", "/api/v1/commercial_security_attestations/latest"],
                ["/api/v1/commercial_security_attestations/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and all_files("SECURITY.md", "docs/commercial_security_attestation.md")
                else "blocked",
                f"commercial_security_attestation_status={security['security_attestation_status']}",
                "Can security approve repo-local controls while external attestations remain caveated?",
                "Collect buyer DPA, privacy, and third-party attestation evidence separately.",
            ),
            gate(
                "implementation_readiness_ready",
                "Implementation readiness ready",
                "Implementation owner",
                [
                    "docs/commercial_onboarding_readiness.md",
                    "docs/commercial_operations_readiness.md",
                    "/api/v1/commercial_onboarding_readiness/latest",
                    "/api/v1/commercial_operations_readiness/latest",
                ],
                ["/api/v1/commercial_onboarding_readiness/latest", "/api/v1/commercial_operations_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and operations["operations_status"] != "commercial_operations_blocked"
                and all_files("docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md")
                else "blocked",
                f"commercial_onboarding_status={onboarding['onboarding_status']}; commercial_operations_status={operations['operations_status']}",
                "Can implementation owners see onboarding and operations evidence before purchase approval?",
                "Turn buyer environment details into the paid onboarding plan.",
            ),
            gate(
                "close_readiness_ready",
                "Close readiness ready",
                "Deal owner",
                ["docs/commercial_close_readiness.md", "/api/v1/commercial_close_readiness/latest"],
                ["/api/v1/commercial_close_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if close["close_status"] != "commercial_close_blocked"
                and has_file("docs/commercial_close_readiness.md")
                else "blocked",
                f"commercial_close_status={close['close_status']}",
                "Can the buyer see final signature, budget, security, and go-live caveats before approval?",
                "Use close readiness as the bridge between local product evidence and buyer approvals.",
            ),
            gate(
                "approval_runtime_packet_ready",
                "Approval runtime packet ready",
                "Stakeholder reviewer",
                [
                    "docs/commercial_purchase_approval_packet.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-purchase-approval-packet-runtime.md",
                ],
                ["/api/v1/commercial_purchase_approval_packets/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if all_files(
                    "docs/commercial_purchase_approval_packet.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-purchase-approval-packet-runtime.md",
                )
                and admin_state["agents"]
                else "blocked",
                "Purchase approval docs, FigJam artifact record, plan, and admin runtime are present.",
                "Can stakeholders review the approval packet as docs, runtime JSON, and FigJam flow?",
                "Keep Figma Code Connect out of the approval workflow.",
            ),
            gate(
                "buyer_signature_authority",
                "Buyer signature authority",
                "Buyer sponsor and legal owner",
                ["signed order form", "MSA", "DPA", "security acceptance"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Signed order form, MSA, DPA, and final security acceptance require buyer authority.",
                "Does the buyer have an identified signer and legal approval path?",
                "Collect named signer and final legal/security approvals.",
            ),
            gate(
                "buyer_budget_po_authority",
                "Buyer budget and PO authority",
                "Finance and procurement owner",
                ["budget approval", "purchase order", "finance authority", "go-live authorization"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Budget approval, purchase order, finance authority, and go-live authorization require buyer input.",
                "Can finance issue the KRW 2B purchase order and approve go-live?",
                "Collect budget owner, PO path, and go-live authorization or explicit waiver.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in approval_gates)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            purchase_approval_status = "commercial_purchase_approval_blocked"
        elif warning_count:
            purchase_approval_status = "commercial_purchase_approval_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            purchase_approval_status = "commercial_purchase_approval_ready"  # pragma: no cover
        required_runtime_endpoints = list(
            dict.fromkeys(
                endpoint
                for item in approval_gates
                for endpoint in item["runtime_endpoints"]
                if endpoint.startswith("/")
            )
        )

        return {
            "purchase_approval_status": purchase_approval_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_purchase_approval_packet",
            "source_note": (
                "Commercial purchase approval packet packages repo-local proposal, close, procurement, "
                "contract, value, security, onboarding, operations, analytics, Figma, review-policy, and "
                "packaging evidence for KRW 2,000,000,000 buyer purchase approval; it is not a valuation "
                "guarantee, purchase commitment, signed order, legal opinion, production compliance "
                "certificate, or revenue proof."
            ),
            "approval_narrative": {
                "title": "KRW 2B buyer purchase approval packet",
                "promise": (
                    "Give finance, procurement, legal, security, and implementation owners one runtime "
                    "approval packet that separates local product evidence from buyer-specific authority inputs."
                ),
                "audience": [
                    "Economic buyer",
                    "Finance owner",
                    "Procurement owner",
                    "Legal owner",
                    "Security owner",
                    "Implementation owner",
                ],
            },
            "approval_summary": {
                "gate_count": len(approval_gates),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "endpoint_count": len(required_runtime_endpoints),
                "review_process_is_blocker": proposal["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "approval_gates": approval_gates,
            "required_runtime_endpoints": required_runtime_endpoints,
            "concrete_blockers": concrete_blockers,
            "purchase_approval_status_rules": [
                {
                    "purchase_approval_status": "commercial_purchase_approval_ready",
                    "rule": "all approval gates are ready and no buyer signature, budget, PO, or go-live inputs remain open",
                },
                {
                    "purchase_approval_status": "commercial_purchase_approval_ready_with_warnings",
                    "rule": "repo-local purchase approval evidence is ready while buyer signature authority, budget, PO, or go-live authorization remain explicit warnings",
                },
                {
                    "purchase_approval_status": "commercial_purchase_approval_blocked",
                    "rule": "security failure, API contract regression, document mismatch, runtime defect, missing local approval evidence, or Code Connect usage blocks purchase approval",
                },
            ],
            "review_process_policy": proposal["review_process_policy"],
            "related_runtime_reports": {
                "commercial_proposal_status": proposal["proposal_status"],
                "commercial_close_status": close["close_status"],
                "commercial_procurement_status": procurement["procurement_status"],
                "commercial_contract_status": contract["contract_status"],
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_operations_status": operations["operations_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": proposal["library_split_decision"],
            "plugin_traceability": proposal["plugin_traceability"],
            "approval_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_purchase_approval_packets/latest",
                "documentation": "docs/commercial_purchase_approval_packet.md",
            },
        }

    def commercial_due_diligence_room_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return buyer due diligence room sections for the KRW 2B standard."""
        purchase = self.commercial_purchase_approval_packet_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        proposal = self.commercial_proposal_packet_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        completion = self.commercial_completion_scorecard_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        demo = self.commercial_demo_scenario_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        buyer_workflow = self.commercial_buyer_acceptance_workflow_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        close = self.commercial_close_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        procurement = self.commercial_procurement_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def section(
            section_name: str,
            label: str,
            reviewer: str,
            sources: list[str],
            runtime_endpoints: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            diligence_question: str,
            next_action: str,
        ) -> dict[str, Any]:
            require_object_name(section_name, "commercial_due_diligence.section_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("commercial due diligence section state must be ready, warning, or blocked")
            return {
                "section_name": section_name,
                "label": label,
                "reviewer": reviewer,
                "sources": sources,
                "runtime_endpoints": runtime_endpoints,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "diligence_question": diligence_question,
                "next_action": next_action,
            }

        concrete_blockers = list(
            dict.fromkeys(
                purchase["concrete_blockers"]
                + proposal["concrete_blockers"]
                + completion["concrete_blockers"]
                + demo["concrete_blockers"]
                + buyer_workflow["concrete_blockers"]
            )
        )
        local_runtime_state = (
            "blocked"
            if purchase["purchase_approval_status"] == "commercial_purchase_approval_blocked"
            or proposal["proposal_status"] == "commercial_proposal_blocked"
            or completion["completion_status"] == "commercial_completion_blocked"
            or demo["demo_status"] == "commercial_demo_blocked"
            or buyer_workflow["workflow_status"] == "buyer_acceptance_workflow_blocked"
            or concrete_blockers
            else "ready"
        )
        diligence_sections = [
            section(
                "purchase_approval_packet",
                "Purchase approval packet",
                "Purchase committee",
                ["docs/commercial_purchase_approval_packet.md", "/api/v1/commercial_purchase_approval_packets/latest"],
                ["/api/v1/commercial_purchase_approval_packets/latest", "/api/v1/commercial_proposal_packets/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state if has_file("docs/commercial_purchase_approval_packet.md") else "blocked",
                f"commercial_purchase_approval_status={purchase['purchase_approval_status']}",
                "Can the buyer review one approval packet before diligence sign-off?",
                "Use the approval packet as the diligence room cover index.",
            ),
            section(
                "runtime_api_evidence",
                "Runtime API evidence",
                "Platform reviewer",
                [
                    "docs/rest_api_design.md",
                    "contextual_orchestrator/api_contract.py",
                    "README.md",
                    "/v1/chat/completions",
                ],
                [
                    "/v1/chat/completions",
                    "/api/v1/workflow_runs",
                    "/api/v1/access_reports/{workflow_run_id}",
                    "/api/v1/commercial_due_diligence_rooms/latest",
                ],
                "repository_and_runtime_artifact",
                "ready"
                if local_runtime_state == "ready"
                and all_files("docs/rest_api_design.md", "contextual_orchestrator/api_contract.py", "README.md")
                and admin_state["agents"]
                else "blocked",
                f"agent_count={len(admin_state['agents'])}; api_contract_present={has_file('contextual_orchestrator/api_contract.py')}",
                "Can platform reviewers verify compatible API and evidence endpoints?",
                "Keep API compatibility and evidence endpoints in the diligence index.",
            ),
            section(
                "admin_trace_evidence",
                "Admin trace evidence",
                "Operator reviewer",
                ["docs/screen_design.md", "/admin", "/admin/state", "workflow trace", "access report"],
                ["/admin", "/admin/state", "/api/v1/workflow_runs", "/api/v1/access_reports/{workflow_run_id}"],
                "repository_and_runtime_artifact",
                "ready"
                if all_files("docs/screen_design.md", "contextual_orchestrator/admin.py")
                and admin_state["agents"]
                and admin_state["recent_workflow_runs"]
                else "blocked",
                f"recent_workflow_run_count={len(admin_state['recent_workflow_runs'])}",
                "Can the operator show trace and access-list evidence in the admin console?",
                "Run one conduct workflow before a live diligence walkthrough.",
            ),
            section(
                "security_and_compliance",
                "Security and compliance",
                "Security reviewer",
                ["SECURITY.md", "docs/commercial_security_attestation.md", "/api/v1/commercial_security_attestations/latest"],
                ["/api/v1/commercial_security_attestations/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and all_files("SECURITY.md", "docs/commercial_security_attestation.md")
                else "blocked",
                f"commercial_security_attestation_status={security['security_attestation_status']}",
                "Can security separate repo-local controls from external attestation gaps?",
                "Do not claim third-party certification until supplied.",
            ),
            section(
                "commercial_terms",
                "Commercial terms",
                "Legal and procurement reviewers",
                [
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_close_readiness.md",
                ],
                [
                    "/api/v1/commercial_contract_readiness/latest",
                    "/api/v1/commercial_procurement_readiness/latest",
                    "/api/v1/commercial_close_readiness/latest",
                ],
                "repository_and_runtime_artifact",
                "ready"
                if contract["contract_status"] != "commercial_contract_blocked"
                and procurement["procurement_status"] != "commercial_procurement_blocked"
                and close["close_status"] != "commercial_close_blocked"
                and all_files(
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_close_readiness.md",
                )
                else "blocked",
                (
                    f"commercial_contract_status={contract['contract_status']}; "
                    f"commercial_procurement_status={procurement['procurement_status']}; "
                    f"commercial_close_status={close['close_status']}"
                ),
                "Can legal and procurement review terms, rights, and close caveats together?",
                "Attach buyer-specific order-form language outside the local runtime claim.",
            ),
            section(
                "value_and_analytics",
                "Value and analytics",
                "Economic reviewer",
                ["docs/commercial_value_readiness.md", "docs/analytics_spec.md", "/api/v1/analytics_snapshots/latest"],
                ["/api/v1/commercial_value_readiness/latest", "/api/v1/analytics_snapshots/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if value["value_status"] != "commercial_value_blocked"
                and analytics["measurement_status"] == "local_runtime_snapshot"
                and all_files("docs/commercial_value_readiness.md", "docs/analytics_spec.md")
                else "blocked",
                f"commercial_value_status={value['value_status']}; analytics_measurement_status={analytics['measurement_status']}",
                "Can finance tell measured local evidence apart from buyer ROI assumptions?",
                "Keep proposed KPI targets separate from measured local runtime data.",
            ),
            section(
                "implementation_readiness",
                "Implementation readiness",
                "Implementation and operations reviewers",
                ["docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md"],
                ["/api/v1/commercial_onboarding_readiness/latest", "/api/v1/commercial_operations_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and operations["operations_status"] != "commercial_operations_blocked"
                and all_files("docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md")
                else "blocked",
                f"commercial_onboarding_status={onboarding['onboarding_status']}; commercial_operations_status={operations['operations_status']}",
                "Can implementation owners see onboarding, operations, incident, and handoff evidence?",
                "Convert buyer environment specifics into the paid onboarding plan.",
            ),
            section(
                "figma_and_design_review",
                "Figma and design review",
                "Product design reviewer",
                [
                    "docs/commercial_due_diligence_room.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-due-diligence-room-runtime.md",
                ],
                ["/api/v1/commercial_due_diligence_rooms/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if all_files(
                    "docs/commercial_due_diligence_room.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-due-diligence-room-runtime.md",
                )
                else "blocked",
                "Due diligence doc, FigJam artifact record, and implementation plan are committed.",
                "Can stakeholders inspect the diligence room as docs, runtime JSON, and FigJam?",
                "Use FigJam only; do not use Figma Code Connect.",
            ),
            section(
                "buyer_authority_documents",
                "Buyer authority documents",
                "Buyer sponsor",
                ["named buyer signer", "budget owner and purchase order", "buyer DPA or privacy acceptance"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Named signer, budget owner, PO, and buyer privacy acceptance require buyer authority.",
                "Does the buyer have authority artifacts ready for final diligence?",
                "Collect signer, PO, DPA/privacy acceptance, or explicit waiver.",
            ),
            section(
                "production_external_attestations",
                "Production and external attestations",
                "Production and security owners",
                ["production telemetry", "third-party security attestation", "hosted scan evidence"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Production telemetry and third-party attestations are external evidence, not local repo measurements.",
                "Can the buyer distinguish local readiness from production and third-party evidence?",
                "Collect hosted telemetry and external attestation after environment selection.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in diligence_sections)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            due_diligence_status = "commercial_due_diligence_blocked"
        elif warning_count:
            due_diligence_status = "commercial_due_diligence_ready_with_warnings"
        else:  # pragma: no cover - unreachable while this report carries literal buyer-specific warning sections
            due_diligence_status = "commercial_due_diligence_ready"  # pragma: no cover
        required_runtime_endpoints = list(
            dict.fromkeys(
                endpoint
                for item in diligence_sections
                for endpoint in item["runtime_endpoints"]
                if endpoint.startswith("/")
            )
        )

        return {
            "due_diligence_status": due_diligence_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_due_diligence_room",
            "source_note": (
                "Commercial due diligence room packages repo-local purchase approval, proposal, runtime, "
                "admin trace, security, contract, value, onboarding, operations, analytics, Figma, "
                "review-policy, and packaging evidence for KRW 2,000,000,000 buyer diligence; it is "
                "not a valuation guarantee, purchase commitment, signed order, legal opinion, production "
                "compliance certificate, third-party attestation, or revenue proof."
            ),
            "diligence_narrative": {
                "title": "KRW 2B commercial due diligence room",
                "promise": (
                    "Give finance, procurement, legal, security, product, and implementation reviewers "
                    "one evidence room that separates measured local product evidence from buyer-specific "
                    "and external production artifacts."
                ),
                "audience": [
                    "Economic buyer",
                    "Finance owner",
                    "Procurement owner",
                    "Legal owner",
                    "Security owner",
                    "Platform reviewer",
                    "Implementation owner",
                ],
            },
            "diligence_summary": {
                "section_count": len(diligence_sections),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "endpoint_count": len(required_runtime_endpoints),
                "review_process_is_blocker": purchase["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "diligence_sections": diligence_sections,
            "required_runtime_endpoints": required_runtime_endpoints,
            "buyer_missing_artifacts": [
                "named buyer signer",
                "budget owner and purchase order",
                "buyer DPA or privacy acceptance",
                "production telemetry",
                "third-party security attestation",
            ],
            "concrete_blockers": concrete_blockers,
            "due_diligence_status_rules": [
                {
                    "due_diligence_status": "commercial_due_diligence_ready",
                    "rule": "all diligence sections are ready and no buyer authority, production, or third-party evidence remains open",
                },
                {
                    "due_diligence_status": "commercial_due_diligence_ready_with_warnings",
                    "rule": "repo-local diligence room evidence is ready while buyer authority, production telemetry, or external attestations remain explicit warnings",
                },
                {
                    "due_diligence_status": "commercial_due_diligence_blocked",
                    "rule": "security failure, API contract regression, document mismatch, runtime defect, missing local diligence evidence, or Code Connect usage blocks buyer diligence",
                },
            ],
            "review_process_policy": purchase["review_process_policy"],
            "related_runtime_reports": {
                "commercial_purchase_approval_status": purchase["purchase_approval_status"],
                "commercial_proposal_status": proposal["proposal_status"],
                "commercial_completion_status": completion["completion_status"],
                "commercial_demo_status": demo["demo_status"],
                "buyer_acceptance_workflow_status": buyer_workflow["workflow_status"],
                "commercial_close_status": close["close_status"],
                "commercial_procurement_status": procurement["procurement_status"],
                "commercial_contract_status": contract["contract_status"],
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_operations_status": operations["operations_status"],
                "analytics_measurement_status": analytics["measurement_status"],
            },
            "library_split_decision": purchase["library_split_decision"],
            "plugin_traceability": purchase["plugin_traceability"],
            "due_diligence_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_due_diligence_rooms/latest",
                "documentation": "docs/commercial_due_diligence_room.md",
            },
        }

    def commercial_investment_committee_memo_report(
        self,
        target_contract_value_krw: int = DEFAULT_COMMERCIAL_TARGET_VALUE_KRW,
        locale_bundles: dict[str, dict[str, str]] | None = None,
        security_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return executive investment committee memo sections for the KRW 2B standard."""
        due_diligence = self.commercial_due_diligence_room_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        purchase = self.commercial_purchase_approval_packet_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        proposal = self.commercial_proposal_packet_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        completion = self.commercial_completion_scorecard_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        demo = self.commercial_demo_scenario_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        buyer_workflow = self.commercial_buyer_acceptance_workflow_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        close = self.commercial_close_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        procurement = self.commercial_procurement_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        contract = self.commercial_contract_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        value = self.commercial_value_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        security = self.commercial_security_attestation_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        onboarding = self.commercial_onboarding_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        operations = self.commercial_operations_readiness_report(
            target_contract_value_krw=target_contract_value_krw,
            locale_bundles=locale_bundles,
            security_profile=security_profile,
        )
        analytics = self.analytics_snapshot(locale_bundles=locale_bundles)
        admin_state = self.admin_state()
        root = Path(__file__).resolve().parents[1]

        def has_file(path: str) -> bool:
            return (root / path).is_file()

        def all_files(*paths: str) -> bool:
            return all(has_file(path) for path in paths)

        def section(
            section_name: str,
            label: str,
            reviewer: str,
            sources: list[str],
            runtime_endpoints: list[str],
            evidence_type: str,
            completion_state: str,
            evidence: str,
            committee_question: str,
            next_action: str,
        ) -> dict[str, Any]:
            require_object_name(section_name, "commercial_investment_committee.section_name")
            if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
                raise ValueError("commercial investment committee section state must be ready, warning, or blocked")
            return {
                "section_name": section_name,
                "label": label,
                "reviewer": reviewer,
                "sources": sources,
                "runtime_endpoints": runtime_endpoints,
                "evidence_type": evidence_type,
                "completion_state": completion_state,
                "evidence": evidence,
                "committee_question": committee_question,
                "next_action": next_action,
            }

        concrete_blockers = list(
            dict.fromkeys(
                due_diligence["concrete_blockers"]
                + purchase["concrete_blockers"]
                + proposal["concrete_blockers"]
                + completion["concrete_blockers"]
                + demo["concrete_blockers"]
                + buyer_workflow["concrete_blockers"]
            )
        )
        local_runtime_state = (
            "blocked"
            if due_diligence["due_diligence_status"] == "commercial_due_diligence_blocked"
            or purchase["purchase_approval_status"] == "commercial_purchase_approval_blocked"
            or proposal["proposal_status"] == "commercial_proposal_blocked"
            or completion["completion_status"] == "commercial_completion_blocked"
            or demo["demo_status"] == "commercial_demo_blocked"
            or buyer_workflow["workflow_status"] == "buyer_acceptance_workflow_blocked"
            or concrete_blockers
            else "ready"
        )
        memo_sections = [
            section(
                "executive_recommendation",
                "Executive recommendation",
                "Investment committee chair",
                [
                    "docs/commercial_investment_committee_memo.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-investment-committee-memo-runtime.md",
                ],
                ["/api/v1/commercial_investment_committee_memos/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state
                if all_files(
                    "docs/commercial_investment_committee_memo.md",
                    "docs/figma_artifacts.md",
                    "docs/superpowers/plans/2026-07-02-commercial-investment-committee-memo-runtime.md",
                )
                else "blocked",
                "Investment committee memo, FigJam artifact record, and implementation plan are committed.",
                "Can the committee recommend the KRW 2B purchase path with explicit conditions?",
                "Use this memo as the executive decision cover artifact.",
            ),
            section(
                "diligence_room_ready",
                "Due diligence room ready",
                "Diligence owner",
                ["docs/commercial_due_diligence_room.md", "/api/v1/commercial_due_diligence_rooms/latest"],
                ["/api/v1/commercial_due_diligence_rooms/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state if has_file("docs/commercial_due_diligence_room.md") else "blocked",
                f"commercial_due_diligence_status={due_diligence['due_diligence_status']}",
                "Can the memo point to a complete diligence room?",
                "Reference due diligence room sections instead of duplicating evidence.",
            ),
            section(
                "purchase_approval_ready",
                "Purchase approval ready",
                "Purchase sponsor",
                ["docs/commercial_purchase_approval_packet.md", "/api/v1/commercial_purchase_approval_packets/latest"],
                ["/api/v1/commercial_purchase_approval_packets/latest"],
                "repository_and_runtime_artifact",
                local_runtime_state if has_file("docs/commercial_purchase_approval_packet.md") else "blocked",
                f"commercial_purchase_approval_status={purchase['purchase_approval_status']}",
                "Can the committee see finance, procurement, legal, security, and implementation gates?",
                "Use the purchase approval packet as committee appendix A.",
            ),
            section(
                "financial_case",
                "Financial case",
                "Economic buyer",
                ["docs/commercial_value_readiness.md", "docs/analytics_spec.md", "/api/v1/analytics_snapshots/latest"],
                ["/api/v1/commercial_value_readiness/latest", "/api/v1/analytics_snapshots/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if value["value_status"] != "commercial_value_blocked"
                and analytics["measurement_status"] == "local_runtime_snapshot"
                and all_files("docs/commercial_value_readiness.md", "docs/analytics_spec.md")
                else "blocked",
                f"commercial_value_status={value['value_status']}; analytics_measurement_status={analytics['measurement_status']}",
                "Does the memo separate measured local evidence from buyer ROI assumptions?",
                "Attach buyer ROI model only after buyer discovery supplies it.",
            ),
            section(
                "risk_and_security_summary",
                "Risk and security summary",
                "Security reviewer",
                ["SECURITY.md", "docs/commercial_security_attestation.md", "/api/v1/commercial_security_attestations/latest"],
                ["/api/v1/commercial_security_attestations/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if security["security_attestation_status"] != "commercial_security_attestation_blocked"
                and all_files("SECURITY.md", "docs/commercial_security_attestation.md")
                else "blocked",
                f"commercial_security_attestation_status={security['security_attestation_status']}",
                "Are security risks explicit without claiming external certification?",
                "Keep third-party attestation outside measured local evidence.",
            ),
            section(
                "commercial_terms_summary",
                "Commercial terms summary",
                "Legal and procurement reviewers",
                [
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_close_readiness.md",
                ],
                [
                    "/api/v1/commercial_contract_readiness/latest",
                    "/api/v1/commercial_procurement_readiness/latest",
                    "/api/v1/commercial_close_readiness/latest",
                ],
                "repository_and_runtime_artifact",
                "ready"
                if contract["contract_status"] != "commercial_contract_blocked"
                and procurement["procurement_status"] != "commercial_procurement_blocked"
                and close["close_status"] != "commercial_close_blocked"
                and all_files(
                    "docs/commercial_contract_readiness.md",
                    "docs/commercial_procurement_readiness.md",
                    "docs/commercial_close_readiness.md",
                )
                else "blocked",
                (
                    f"commercial_contract_status={contract['contract_status']}; "
                    f"commercial_procurement_status={procurement['procurement_status']}; "
                    f"commercial_close_status={close['close_status']}"
                ),
                "Can legal and procurement conditions be approved or tracked?",
                "Attach order-form details only after buyer legal review.",
            ),
            section(
                "implementation_readiness_summary",
                "Implementation readiness summary",
                "Implementation owner",
                ["docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md"],
                ["/api/v1/commercial_onboarding_readiness/latest", "/api/v1/commercial_operations_readiness/latest"],
                "repository_and_runtime_artifact",
                "ready"
                if onboarding["onboarding_status"] != "commercial_onboarding_blocked"
                and operations["operations_status"] != "commercial_operations_blocked"
                and all_files("docs/commercial_onboarding_readiness.md", "docs/commercial_operations_readiness.md")
                else "blocked",
                f"commercial_onboarding_status={onboarding['onboarding_status']}; commercial_operations_status={operations['operations_status']}",
                "Can implementation start after buyer environment details are supplied?",
                "Convert buyer environment details into the paid onboarding plan.",
            ),
            section(
                "design_and_figma_review",
                "Design and Figma review",
                "Product design reviewer",
                ["docs/figma_artifacts.md", "docs/commercial_investment_committee_memo.md"],
                ["/api/v1/commercial_investment_committee_memos/latest"],
                "repository_and_runtime_artifact",
                "ready" if all_files("docs/figma_artifacts.md", "docs/commercial_investment_committee_memo.md") else "blocked",
                "FigJam memo flow and Product Design scope are recorded without Code Connect.",
                "Can stakeholders inspect the memo flow in FigJam and runtime JSON?",
                "Keep Figma Code Connect out of the committee workflow.",
            ),
            section(
                "buyer_final_authority",
                "Buyer final authority",
                "Buyer sponsor",
                ["executive sponsor approval", "named signer", "budget owner", "purchase order"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Executive sponsor approval, named signer, budget owner, and purchase order require buyer authority.",
                "Can the buyer approve the final KRW 2B purchase authority?",
                "Collect final buyer authority artifacts or explicit waiver.",
            ),
            section(
                "production_external_evidence",
                "Production and external evidence",
                "Production and security owners",
                ["production telemetry", "third-party security attestation", "hosted scan evidence"],
                [],
                "proposed_until_buyer_specific",
                "warning",
                "Production telemetry, third-party security attestation, and hosted scan evidence are external inputs.",
                "Can the committee approve with production and external evidence still tracked as conditions?",
                "Collect hosted evidence after environment selection.",
            ),
        ]
        state_counts = Counter(item["completion_state"] for item in memo_sections)
        blocked_count = state_counts.get("blocked", 0) + len(concrete_blockers)
        warning_count = state_counts.get("warning", 0)
        if blocked_count:
            investment_committee_status = "commercial_investment_committee_blocked"
            recommendation_status = "do_not_recommend_until_blockers_cleared"
        elif warning_count:
            investment_committee_status = "commercial_investment_committee_ready_with_warnings"
            recommendation_status = "recommend_with_buyer_conditions"
        else:  # pragma: no cover - unreachable while external-evidence warning sections remain literal
            investment_committee_status = "commercial_investment_committee_ready"  # pragma: no cover
            recommendation_status = "recommend"  # pragma: no cover
        required_runtime_endpoints = list(
            dict.fromkeys(
                endpoint
                for item in memo_sections
                for endpoint in item["runtime_endpoints"]
                if endpoint.startswith("/")
            )
        )

        return {
            "investment_committee_status": investment_committee_status,
            "target_contract_value_krw": target_contract_value_krw,
            "target_contract_value_display": f"KRW {target_contract_value_krw:,}",
            "measurement_status": "local_commercial_investment_committee_memo",
            "source_note": (
                "Commercial investment committee memo packages repo-local due diligence, purchase approval, "
                "proposal, runtime, admin trace, security, contract, value, onboarding, operations, analytics, "
                "Figma, review-policy, and packaging evidence for KRW 2,000,000,000 executive review; it is "
                "not a valuation guarantee, purchase commitment, signed order, legal opinion, production "
                "compliance certificate, third-party attestation, or revenue proof."
            ),
            "executive_recommendation": {
                "title": "KRW 2B commercial investment committee memo",
                "recommendation_status": recommendation_status,
                "recommendation": (
                    "Recommend committee review with buyer authority, production telemetry, and external "
                    "attestation conditions tracked separately from measured local product evidence."
                    if recommendation_status == "recommend_with_buyer_conditions"
                    else "Do not recommend until concrete blockers are cleared."
                    if recommendation_status == "do_not_recommend_until_blockers_cleared"
                    else "Recommend committee approval with no open local or buyer-specific conditions."
                ),
                "audience": [
                    "Investment committee chair",
                    "Economic buyer",
                    "Finance owner",
                    "Procurement owner",
                    "Legal owner",
                    "Security owner",
                    "Implementation owner",
                ],
            },
            "memo_summary": {
                "section_count": len(memo_sections),
                "ready_count": state_counts.get("ready", 0),
                "warning_count": warning_count,
                "blocked_count": blocked_count,
                "endpoint_count": len(required_runtime_endpoints),
                "review_process_is_blocker": due_diligence["review_process_policy"]["is_blocker"],
                "code_connect_used": False,
            },
            "memo_sections": memo_sections,
            "required_runtime_endpoints": required_runtime_endpoints,
            "committee_decision_questions": [
                "Is the product evidence sufficient for KRW 2B buyer review?",
                "Are buyer authority documents named and tracked?",
                "Are production and third-party evidence gaps explicit warnings?",
                "Is any concrete blocker present?",
            ],
            "buyer_missing_artifacts": due_diligence["buyer_missing_artifacts"]
            + ["executive sponsor approval", "investment committee sign-off"],
            "concrete_blockers": concrete_blockers,
            "investment_committee_status_rules": [
                {
                    "investment_committee_status": "commercial_investment_committee_ready",
                    "rule": "all memo sections are ready and no buyer authority, production, or third-party evidence remains open",
                },
                {
                    "investment_committee_status": "commercial_investment_committee_ready_with_warnings",
                    "rule": "repo-local committee memo evidence is ready while buyer authority, production telemetry, or external attestations remain explicit warnings",
                },
                {
                    "investment_committee_status": "commercial_investment_committee_blocked",
                    "rule": "security failure, API contract regression, document mismatch, runtime defect, missing local memo evidence, or Code Connect usage blocks committee recommendation",
                },
            ],
            "review_process_policy": due_diligence["review_process_policy"],
            "related_runtime_reports": {
                "commercial_due_diligence_status": due_diligence["due_diligence_status"],
                "commercial_purchase_approval_status": purchase["purchase_approval_status"],
                "commercial_proposal_status": proposal["proposal_status"],
                "commercial_completion_status": completion["completion_status"],
                "commercial_demo_status": demo["demo_status"],
                "buyer_acceptance_workflow_status": buyer_workflow["workflow_status"],
                "commercial_close_status": close["close_status"],
                "commercial_procurement_status": procurement["procurement_status"],
                "commercial_contract_status": contract["contract_status"],
                "commercial_value_status": value["value_status"],
                "commercial_security_attestation_status": security["security_attestation_status"],
                "commercial_onboarding_status": onboarding["onboarding_status"],
                "commercial_operations_status": operations["operations_status"],
                "analytics_measurement_status": analytics["measurement_status"],
                "admin_agent_count": len(admin_state["agents"]),
            },
            "library_split_decision": due_diligence["library_split_decision"],
            "plugin_traceability": due_diligence["plugin_traceability"],
            "committee_links": {
                "figma_design_file": "https://www.figma.com/design/vsZMd8WAv42HDRgcZuNcWk",
                "figjam_board": "https://www.figma.com/board/Wr8iMlB9SHkerHSjv0Pe0M",
                "runtime_endpoint": "/api/v1/commercial_investment_committee_memos/latest",
                "documentation": "docs/commercial_investment_committee_memo.md",
            },
        }

    def admin_state(
        self,
        owner_id: str | None = None,
        *,
        role: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Build the admin console state payload from agents, policy, and audit data."""
        agent_page_size = max(1, len(self.candidates))
        return {
            "agents": self.list_agents(page_size=agent_page_size),
            "policy": {
                **self.policy.as_dict(),
                "roles": list(self.ROLE_TAGS),
            },
            "routing_evidence": {
                "transport": self._group_router.snapshot(),
                "quality": self._quality_router.snapshot(),
            },
            "recent_workflow_runs": [
                self._shorten_run(run)
                for run in self.list_recent_runs(
                    page_size=max(1, len(self._run_order)), owner_id=owner_id
                )
            ],
            "recent_audit_events": self.list_recent_audit_events(role=role, purpose=purpose),
            "recent_authorization_decisions": self.list_recent_authorization_decisions(),
            "spend": self.spend_analytics(),
        }

    def _shorten_run(self, run: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        return {
            "workflow_run_id": run["workflow_run_id"],
            "mode": run["mode"],
            "policy_mode": run["policy_mode"],
            "created_at": run["created_at"],
        }

    def _is_trace_complete(self, run: dict[str, Any]) -> bool:
        trace = run.get("trace", [])
        if not trace:
            return False
        for step in trace:
            if not all(key in step for key in ("id", "role", "agent_id", "subtask", "access", "output")):
                return False
            if not isinstance(step["access"], list) or step["output"] is None:
                return False
        verification = run.get("verification") or {}
        return bool(run.get("answer") and "accepted" in verification and "reason" in verification)

    def _is_policy_safe_run(self, run: dict[str, Any]) -> bool:
        if run["mode"] == "conduct" and run["policy_snapshot"].get("verifier_required") and not run.get("verification"):
            return False
        return self._provider_exclusion_miss_count(run) == 0

    def _provider_exclusion_miss_count(self, run: dict[str, Any]) -> int:
        misses = 0
        for step in run.get("trace", []):
            try:
                agent = self._agent(step["agent_id"])
            except KeyError:
                misses += 1
                continue
            if step["role"] in agent.provider_exclusions:
                misses += 1
        return misses

    def _locale_key_parity(self, locale_bundles: dict[str, dict[str, str]]) -> dict[str, Any]:
        english = locale_bundles.get("en", {})
        other_locale_codes = sorted(code for code in locale_bundles if code != "en")
        denominator = len(english) * len(other_locale_codes)
        missing = [
            f"{locale_code}.{key}"
            for locale_code in other_locale_codes
            for key in sorted(english)
            if not locale_bundles[locale_code].get(key)
        ]
        numerator = denominator - len(missing)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value_percent": self._percent(numerator, denominator),
            "missing_keys": missing,
        }

    def _percent(self, numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return round((numerator / denominator) * 100, 2)

    def _criterion(
        self,
        criterion_name: str,
        label: str,
        status: str,
        evidence: str,
        remediation: str,
    ) -> dict[str, str]:
        require_object_name(criterion_name, "sales_readiness.criterion_name")
        if status not in {"pass", "warn", "fail"}:  # pragma: no cover
            raise ValueError("sales readiness status must be pass, warn, or fail")
        return {
            "criterion_name": criterion_name,
            "status": status,
            "label": label,
            "evidence": evidence,
            "remediation": remediation,
        }

    def _buyer_evidence_item(
        self,
        item_name: str,
        label: str,
        reviewer: str,
        sources: list[str],
        evidence_type: str,
        completion_state: str,
        evidence: str,
        next_action: str,
    ) -> dict[str, Any]:
        require_object_name(item_name, "buyer_evidence_manifest.item_name")
        if evidence_type not in {
            "measured_local",
            "repository_artifact",
            "figma_artifact",
            "proposed_until_production",
            "proposed_until_buyer_specific",
        }:  # pragma: no cover
            raise ValueError("buyer evidence type is invalid")
        if completion_state not in {"ready", "warning", "blocked"}:  # pragma: no cover
            raise ValueError("buyer evidence completion state must be ready, warning, or blocked")
        return {
            "item_name": item_name,
            "label": label,
            "reviewer": reviewer,
            "sources": sources,
            "evidence_type": evidence_type,
            "completion_state": completion_state,
            "evidence": evidence,
            "next_action": next_action,
        }

    def _buyer_manifest_summary(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        completion_counts = Counter(item["completion_state"] for item in items)
        evidence_counts = Counter(item["evidence_type"] for item in items)
        return {
            "total_items": len(items),
            "by_completion_state": {
                "ready": completion_counts.get("ready", 0),
                "warning": completion_counts.get("warning", 0),
                "blocked": completion_counts.get("blocked", 0),
            },
            "by_evidence_type": {
                "measured_local": evidence_counts.get("measured_local", 0),
                "repository_artifact": evidence_counts.get("repository_artifact", 0),
                "figma_artifact": evidence_counts.get("figma_artifact", 0),
                "proposed_until_production": evidence_counts.get("proposed_until_production", 0),
                "proposed_until_buyer_specific": evidence_counts.get("proposed_until_buyer_specific", 0),
            },
        }

    def _security_posture_criterion(self, security_profile: dict[str, Any]) -> dict[str, str]:
        auth_mode = security_profile.get("auth_mode", "loopback_no_auth")
        issues: list[str] = []
        warnings: list[str] = []
        if auth_mode == "single_token":
            warnings.append("single bearer token shared by admin and inference scopes")
        elif auth_mode not in {"split_token", "external_bearer_verifier"}:
            issues.append("no bearer token configured outside loopback-only development")
        if security_profile.get("allow_public_bind"):
            issues.append("public bind is enabled")
        if security_profile.get("expose_trace_by_default"):
            issues.append("trace exposure is enabled by default")
        if int(security_profile.get("rate_limit_requests") or 0) <= 0:
            issues.append("request rate limiting is disabled")
        if int(security_profile.get("max_concurrent_runs") or 0) <= 0:
            issues.append("run concurrency limiting is disabled")

        if issues:
            status = "fail"
            evidence = "; ".join(issues)
            remediation = "Require bearer auth, private bind defaults, hidden traces, rate limits, and run limits."
        elif warnings:
            status = "warn"
            evidence = "; ".join(warnings)
            remediation = "For enterprise pilots, split admin and inference tokens before customer evaluation."
        else:
            status = "pass"
            auth_evidence = (
                "external bearer verifier"
                if auth_mode == "external_bearer_verifier"
                else "split tokens"
            )
            evidence = f"{auth_evidence}, private bind default, hidden traces, rate limits, and run limits are configured"
            remediation = "Keep these controls enabled for customer-facing pilots."

        return self._criterion("security_posture", "Security posture", status, evidence, remediation)

    def _locale_readiness_criterion(self, analytics: dict[str, Any]) -> dict[str, str]:
        locale_metric = next(
            metric for metric in analytics["guardrails"] if metric["metric_name"] == "locale_key_parity"
        )
        missing = locale_metric.get("missing_keys", [])
        parity = locale_metric.get("value_percent")
        if parity == 100.0:
            status = "pass"
            evidence = "English and Korean admin locale keys are aligned"
            remediation = "Keep locale parity tests updated when adding operator copy."
        else:
            status = "warn" if missing else "fail"
            evidence = f"{parity}% locale key parity; missing keys: {', '.join(missing) or 'locale bundles absent'}"
            remediation = "Fill missing Korean and English operator labels before customer review."
        return self._criterion("locale_readiness", "Locale readiness", status, evidence, remediation)

    def _provider_egress_criterion(self) -> dict[str, str]:
        unsafe = []
        remote = []
        for agent in self.agents:
            if agent.base_url.startswith("mock://"):
                continue
            if _is_local_provider_url(agent.base_url):
                continue
            remote.append(agent.id)
            parsed = urlparse(agent.base_url)
            if parsed.scheme != "https" or not agent.credential_name:
                unsafe.append(agent.id)
        if unsafe:
            status = "fail"
            evidence = f"unsafe provider egress config for agents: {', '.join(sorted(unsafe))}"
            remediation = "Use https provider endpoints with a named KV credential before enabling remote egress."
        else:
            status = "pass"
            evidence = (
                "mock providers only"
                if not remote
                else f"{len(remote)} remote providers use https and a named KV credential"
            )
            remediation = "Keep provider allow-list enforcement enabled for non-mock providers."
        return self._criterion("provider_egress_safety", "Provider egress safety", status, evidence, remediation)

    def _criteria_by_name(self, criteria: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        return {row["criterion_name"]: row for row in criteria}

    def _metrics_by_name(self, metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {row["metric_name"]: row for row in metrics}

    def _commercial_documentation_profile(self) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        required_documents = [
            "README.md",
            "SECURITY.md",
            "docs/product_planning.md",
            "docs/rest_api_design.md",
            "docs/analytics_spec.md",
            "docs/commercial_readiness.md",
        ]
        missing_documents = [
            document_path
            for document_path in required_documents
            if not (root / document_path).is_file()
        ]
        return {
            "required_documents": required_documents,
            "missing_documents": missing_documents,
            "present_count": len(required_documents) - len(missing_documents),
            "required_count": len(required_documents),
            "has_security_policy": (root / "SECURITY.md").is_file(),
            "source": "repository documentation files",
        }

    def _criteria_summary(self, criteria: list[dict[str, str]]) -> dict[str, int]:
        counts = Counter(row["status"] for row in criteria)
        return {
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
        }


def _report_cache_token(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_report_cache_token(item) for item in value)
    return ("id", id(value))


def _commercial_report_cached(method: Any) -> Any:
    @wraps(method)
    def wrapper(self: TaskOrchestrator, *args: Any, **kwargs: Any) -> dict[str, Any]:
        local = self._commercial_report_cache_local
        depth = getattr(local, "depth", 0)
        if depth == 0:
            local.cache = {}
        local.depth = depth + 1
        try:
            key = (
                method.__name__,
                _report_cache_token(args),
                tuple(sorted((name, _report_cache_token(value)) for name, value in kwargs.items())),
            )
            if key not in local.cache:
                local.cache[key] = method(self, *args, **kwargs)
            return local.cache[key]
        finally:
            local.depth -= 1
            if depth == 0:
                local.cache = {}

    return wrapper


for _report_name, _report_method in list(TaskOrchestrator.__dict__.items()):
    if _report_name.startswith("commercial_") and _report_name.endswith("_report"):
        setattr(TaskOrchestrator, _report_name, _commercial_report_cached(_report_method))


def redact_text(text: str) -> str:
    """Mask credential shapes (API keys, tokens, passwords, bearer headers) from traces.

    Does not mask PII (email addresses, names, etc.): per governance-risk-compliance policy,
    PII is protected by purpose-limited authorization, encryption, and audit logging, not by
    destroying it in every response -- blanket PII masking here broke every downstream
    consumer that needs the real content (e.g. an email client rendering actual addresses).
    """
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)(api"):
            redacted = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
        else:
            redacted = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", redacted)
    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact string values while preserving response shape."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value

def _freeze_report_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _freeze_report_cache_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_report_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_report_cache_value(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _cached_commercial_report(method: Any) -> Any:
    @wraps(method)
    def wrapper(self: TaskOrchestrator, *args: Any, **kwargs: Any) -> dict[str, Any]:
        cache = _COMMERCIAL_REPORT_CACHE.get()
        token = None
        if cache is None:
            cache = {}
            token = _COMMERCIAL_REPORT_CACHE.set(cache)
        try:
            key = (
                method.__name__,
                _freeze_report_cache_value(args),
                _freeze_report_cache_value(kwargs),
            )
            if key not in cache:
                cache[key] = method(self, *args, **kwargs)
            return cache[key]
        finally:
            if token is not None:
                _COMMERCIAL_REPORT_CACHE.reset(token)

    return wrapper


for _commercial_report_name, _commercial_report_method in list(TaskOrchestrator.__dict__.items()):
    if (
        _commercial_report_name.startswith("commercial_")
        and _commercial_report_name.endswith("_report")
        and callable(_commercial_report_method)
    ):
        setattr(
            TaskOrchestrator,
            _commercial_report_name,
            _cached_commercial_report(_commercial_report_method),
        )


def _pareto_front(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Configs not dominated on (quality up, cost down)."""
    front: list[dict[str, Any]] = []
    for a in results:
        dominated = any(
            b is not a
            and b["quality"] >= a["quality"]
            and b["cost_usd"] <= a["cost_usd"]
            and (b["quality"] > a["quality"] or b["cost_usd"] < a["cost_usd"])
            for b in results
        )
        if not dominated:
            front.append(a)
    return front


def _recommend_config(results: list[dict[str, Any]], cost_budget_usd: float | None) -> dict[str, Any] | None:
    if not results:
        return None
    if cost_budget_usd is not None:
        affordable = [r for r in results if r["cost_usd"] <= cost_budget_usd]
        if affordable:
            best = max(affordable, key=lambda r: (r["quality"], -r["cost_usd"]))
            reason = "highest quality within cost budget"
        else:
            best = min(results, key=lambda r: r["cost_usd"])
            reason = "no config within budget; cheapest instead"
    else:
        # Maximize performance first, minimize cost as the tie-break (cheapest among the
        # best-quality configs) — the honest reading of "max quality while min cost".
        best = max(results, key=lambda r: (r["quality"], -r["cost_usd"]))
        reason = "highest quality; cheapest among equal-quality configs"
    return {"name": best["name"], "quality": best["quality"], "cost_usd": best["cost_usd"], "reason": reason}


def _score_config(orchestrator: Any, tasks: list[dict[str, Any]], quality_fn: Any, mode: str, use_batch: bool) -> float:
    """Mean quality of one config over the task set; route configs may evaluate via Batch."""
    if use_batch and mode == "route":
        records = orchestrator.batch_route([task["prompt"] for task in tasks])
        scores = [float(quality_fn(task, record["answer"] or "")) for task, record in zip(tasks, records)]
    else:
        scores = [
            float(quality_fn(task, orchestrator.run([{"role": "user", "content": task["prompt"]}], mode=mode)["answer"]))
            for task in tasks
        ]
    return sum(scores) / len(scores) if scores else 0.0


def optimize_orchestration(
    candidates: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    quality_fn: Any,
    cost_budget_usd: float | None = None,
    use_batch: bool = False,
) -> dict[str, Any]:
    """Search orchestration configs for maximum quality at minimum cost (the Fugu tradeoff).

    - ``candidates``: ``[{"name": str, "orchestrator": TaskOrchestrator, "mode": str}]``.
      Each orchestrator should be fresh so its spend reflects only this search.
    - ``tasks``: ``[{"prompt": str, ...}]`` — the eval set; each task + the produced
      answer is passed to ``quality_fn``.
    - ``quality_fn(task, answer_text) -> float`` in [0, 1] — the caller's real quality
      signal (e.g. checkable answers or a judge). This function does not fabricate quality.
    - ``cost_budget_usd``: optional cap. Recommendation = highest-quality config within
      budget, else the cheapest; with no budget, the best quality-per-USD.

    Returns per-config measured quality + real cost, the Pareto front, and a recommendation.
    """
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        orchestrator = candidate["orchestrator"]
        mode = candidate.get("mode", "auto")
        quality = _score_config(orchestrator, tasks, quality_fn, mode, use_batch)
        cost = orchestrator.spend_analytics()["totals"]["estimated_cost_usd"] or 0.0
        results.append({
            "name": candidate["name"],
            "mode": mode,
            "quality": round(quality, 4),
            "cost_usd": round(cost, 6),
            "quality_per_usd": round(quality / cost, 2) if cost > 0 else None,
            "task_count": len(tasks),
        })

    return {
        "objective": "maximize quality, minimize cost",
        "cost_budget_usd": cost_budget_usd,
        "results": sorted(results, key=lambda r: (-r["quality"], r["cost_usd"])),
        "pareto_front": [r["name"] for r in _pareto_front(results)],
        "recommended": _recommend_config(results, cost_budget_usd),
    }


def evolve_orchestration(
    build_orchestrator: Any,
    search_space: dict[str, list[Any]],
    tasks: list[dict[str, Any]],
    quality_fn: Any,
    generations: int = 4,
    population: int = 6,
    cost_budget_usd: float | None = None,
    seed: int = 7,
    use_batch: bool = False,
) -> dict[str, Any]:
    """Evolve orchestration configs toward max quality at min cost (TRINITY-style search).

    For search spaces too large to enumerate: a seeded mutation+selection loop over
    ``search_space`` (param -> candidate values). ``build_orchestrator(config)`` returns a
    fresh TaskOrchestrator for a config (the caller owns provider wiring); each config is
    evaluated ONCE and cached — critical when evaluation costs real API money.

    Fitness maximizes measured quality, then minimizes measured cost; configs whose cost
    exceeds ``cost_budget_usd`` rank below all affordable ones. Quality comes from the
    caller's ``quality_fn(task, answer) -> [0,1]`` — never fabricated.
    """
    rng = random.Random(seed)
    params = sorted(search_space)

    def random_config() -> dict[str, Any]:
        return {p: rng.choice(search_space[p]) for p in params}

    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        child = dict(config)
        gene = rng.choice(params)
        choices = [v for v in search_space[gene] if v != config[gene]] or search_space[gene]
        child[gene] = rng.choice(choices)
        return child

    def key(config: dict[str, Any]) -> str:
        return json.dumps({p: config[p] for p in params}, sort_keys=True, ensure_ascii=False)

    evaluated: dict[str, dict[str, Any]] = {}

    def evaluate(config: dict[str, Any]) -> dict[str, Any]:
        config_key = key(config)
        if config_key in evaluated:
            return evaluated[config_key]
        orchestrator = build_orchestrator(config)
        mode = config.get("mode", "auto")
        quality = _score_config(orchestrator, tasks, quality_fn, mode, use_batch)
        cost = orchestrator.spend_analytics()["totals"]["estimated_cost_usd"] or 0.0
        result = {
            "name": config_key,
            "config": dict(config),
            "quality": round(quality, 4),
            "cost_usd": round(cost, 6),
            "quality_per_usd": round(quality / cost, 2) if cost > 0 else None,
            "task_count": len(tasks),
        }
        evaluated[config_key] = result
        return result

    def fitness(row: dict[str, Any]) -> tuple[int, float, float]:
        affordable = 1 if cost_budget_usd is None or row["cost_usd"] <= cost_budget_usd else 0
        return (affordable, row["quality"], -row["cost_usd"])

    # Seed population (dedup keys so a tiny space doesn't waste evaluations).
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(pool) < population and len(seen) < min(population * 4, _space_size(search_space)):
        config = random_config()
        if key(config) not in seen:
            seen.add(key(config))
            pool.append(config)

    history: list[dict[str, Any]] = []
    for generation in range(generations):
        rows = [evaluate(config) for config in pool]
        rows.sort(key=fitness, reverse=True)
        survivors = rows[: max(1, len(rows) // 2)]
        history.append({
            "generation": generation,
            "best": {"config": survivors[0]["config"], "quality": survivors[0]["quality"], "cost_usd": survivors[0]["cost_usd"]},
            "evaluated_total": len(evaluated),
        })
        pool = [dict(row["config"]) for row in survivors]
        while len(pool) < population:
            pool.append(mutate(rng.choice(survivors)["config"]))

    results = sorted(evaluated.values(), key=fitness, reverse=True)
    return {
        "objective": "maximize quality, minimize cost (evolutionary search)",
        "cost_budget_usd": cost_budget_usd,
        "generations": generations,
        "evaluations": len(evaluated),
        "space_size": _space_size(search_space),
        "results": results,
        "pareto_front": [row["name"] for row in _pareto_front(results)],
        "recommended": _recommend_config(results, cost_budget_usd),
        "history": history,
    }


def _space_size(search_space: dict[str, list[Any]]) -> int:
    size = 1
    for values in search_space.values():
        size *= max(1, len(values))
    return size


def chat_completion_response(
    result: dict[str, Any],
    model: str = "contextual-orchestrator",
    include_trace: bool = False,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:  # pragma: no cover
    """Wrap orchestration output in an OpenAI-compatible chat completion response.

    ``usage`` carries the token counts recorded by the cost ledger; when absent
    the response reports zeros (the count could not be computed).
    """
    orchestration = {
        "workflow_run_id": result.get("workflow_run_id"),
        "mode": result["mode"],
        "verification": result.get("verification"),
        "channel": result.get("channel"),
        "routing_reason": result.get("routing_reason"),
        "usage_record_id": result.get("usage_record_id"),
        "cost": result.get("cost"),
    }
    if include_trace:
        orchestration["trace"] = redact_value(result["trace"])
    return {
        "id": _new_chat_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "orchestration": {key: value for key, value in orchestration.items() if value is not None},
    }


def text_completion_response(
    result: dict[str, Any],
    model: str = "contextual-orchestrator",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:  # pragma: no cover
    """Wrap orchestration output as OpenAI legacy ``text_completion`` (``/v1/completions``)."""
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": result["answer"],
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


_STREAM_CHUNK_SIZE = 32


def chat_completion_chunks(
    result: dict[str, Any],
    model: str = "contextual-orchestrator",
    include_trace: bool = False,
) -> list[dict[str, Any]]:
    """Frame an orchestration result as OpenAI-compatible ``chat.completion.chunk`` deltas.

    The engine produces the full answer before framing, so this yields a correct-shape
    SSE stream (role delta, content deltas, terminal stop delta) rather than true
    token-by-token streaming — real token streaming requires a streaming ModelClient.
    """
    answer = result.get("answer", "")
    completion_id = _new_chat_completion_id()
    created = int(time.time())
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}

    chunks: list[dict[str, Any]] = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    ]
    for start in range(0, len(answer), _STREAM_CHUNK_SIZE):
        piece = answer[start : start + _STREAM_CHUNK_SIZE]
        chunks.append({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})

    orchestration = {
        "workflow_run_id": result.get("workflow_run_id"),
        "mode": result.get("mode"),
        "verification": result.get("verification"),
    }
    if include_trace and "trace" in result:
        orchestration["trace"] = redact_value(result["trace"])
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    final["orchestration"] = {key: value for key, value in orchestration.items() if value is not None}
    chunks.append(final)
    return chunks


def _new_chat_completion_id() -> str:
    """Create a collision-resistant OpenAI-compatible completion identifier."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def sse_stream_body(chunks: list[dict[str, Any]]) -> str:
    """Serialize chat completion chunks as a Server-Sent Events body terminated by ``[DONE]``."""
    frames = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    frames.append("data: [DONE]\n\n")
    return "".join(frames)
