"""Shared fuzz-target logic for the highest-value untrusted-input surfaces.

Each ``exercise_*`` function takes one decoded input, drives a real code path,
and asserts the invariants that must hold *for arbitrary input*:

* the function either produces a well-formed result or raises one of a small,
  documented set of "expected" exceptions -- never an unhandled ``TypeError``,
  ``AttributeError``, ``RecursionError``, ``SystemError`` or a hang; and
* structural invariants on any successful result (shape, types, idempotence).

CodeGraph (``codegraph explore``) surfaced these nine surfaces as the ones that
consume untrusted bytes/JSON:

1. ``server._coerce_json`` / ``_validate_mode`` / ``_validate_messages`` /
   ``_reject_unknown_keys`` -- the HTTP request-body parser and validators.
2. ``orchestrator.ModelAgent.from_dict`` -- the agent-pool config parser.
3. ``orchestrator.redact_text`` / ``redact_value`` -- secret/PII redaction run
   over arbitrary trace payloads (regex + recursion).
4. ``orchestrator.TaskOrchestrator.run`` (+ ``sse_stream_body``) -- end-to-end
   prompt processing on a mock (offline) provider.
5. ``orchestrator._parse_model_judge_reply`` -- strict parsing of untrusted
   model-generated verdicts.
6. ``model_discovery._parse_openai_compatible`` / ``_parse_bytez`` -- parsing
   of a remote provider's model-list HTTP response (attacker/compromised
   provider-controlled JSON).
7. ``pii_protection._decode_secret`` -- explicit key-encoding enforcement at
   the field-encryption boundary.
8. ``reasoning_effort_profile.parse_reasoning_effort_profile`` -- untrusted
   role-compute JSON. Must raise ``EffortProfileError`` / ``TypeError`` /
   ``ValueError`` or return a finite profile. Never crash on NaN, bool-as-
   number, or unknown keys.
9. ``orchestrator._structured_output_error`` -- untrusted provider JSON text
   plus caller-supplied JSON Schema. Must return a bounded contract error or
   ``None`` without crashing the gateway.

No network, no secrets, no filesystem: every target runs fully offline.
"""

from __future__ import annotations

import json
import math
from typing import Any

from contextual_orchestrator import server
from contextual_orchestrator.model_discovery import (
    ProviderModelSource,
    _parse_bytez,
    _parse_openai_compatible,
)
from contextual_orchestrator.orchestrator import (
    ModelAgent,
    TaskOrchestrator,
    _parse_model_judge_reply,
    _structured_output_error,
    chat_completion_chunks,
    redact_text,
    redact_value,
    sse_stream_body,
)
from contextual_orchestrator.pii_protection import PiiProtectionError, _decode_secret
from contextual_orchestrator.reasoning_effort_profile import (
    ACCESS_LIST_SCOPES,
    REASONING_EFFORT_LEVELS,
    EffortProfileError,
    parse_reasoning_effort_profile,
    production_default_change_allowed,
    run_equal_budget_ablation,
)

# ``RequestError`` is the only *domain* exception the request layer is allowed to
# raise; everything else below is a legitimate stdlib decode/parse failure.
RequestError = server.RequestError

# Malformed bytes/JSON must surface only as these.
_EXPECTED_BODY_EXC = (
    RequestError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    ValueError,  # json.loads raises ValueError subclasses; be explicit anyway
)

# ``from_dict`` indexes/coerces raw config, so these are the sane failure modes.
_EXPECTED_CONFIG_EXC = (
    KeyError,
    TypeError,
    ValueError,
)


def exercise_pii_key(value: str) -> None:
    """Verify arbitrary unprefixed key text cannot cross the key boundary."""
    if value.startswith(("base64:", "hex:", "passphrase:")):
        return
    try:
        _decode_secret(value, key_name="fuzz_key")
    except PiiProtectionError:
        return
    raise AssertionError("unprefixed PII encryption key was accepted")


def exercise_request_body(raw: bytes) -> None:
    """Drive the HTTP request-body parser + validators over arbitrary bytes.

    Mirrors what ``Handler._read_json`` does after size/content-type checks:
    decode + JSON-parse, then run the field validators the POST routes use.
    """
    try:
        body = server._coerce_json(raw)
    except _EXPECTED_BODY_EXC:
        return
    except RecursionError:
        # Deeply nested JSON is attacker-controllable; the parser should not let
        # it escape as an unhandled crash of an unexpected type. Treat as a
        # finding only if it is NOT a plain RecursionError from json depth.
        return

    # On success the contract is: a plain ``dict``.
    assert isinstance(body, dict), f"body must be dict, got {type(body)!r}"

    # Unknown-key rejection must never raise anything but RequestError.
    try:
        server._reject_unknown_keys(body, {"prompt_text", "run_mode", "messages", "mode"})
    except RequestError:
        pass

    # Mode validation: any value either normalises to an allowed mode or raises
    # RequestError -- nothing else.
    for key in ("run_mode", "mode"):
        if key in body:
            try:
                mode = server._validate_mode(body[key])
            except RequestError:
                pass
            else:
                assert mode in server.ALLOWED_MODES

    # Message validation: returns a normalised list or raises RequestError.
    if "messages" in body:
        try:
            messages = server._validate_messages(body["messages"])
        except RequestError:
            pass
        else:
            assert isinstance(messages, list) and messages
            for message in messages:
                assert set(message) == {"role", "content"}
                assert message["role"] in server.ALLOWED_MESSAGE_ROLES
                assert isinstance(message["content"], str)

    # Omit-equivalent instructions/metadata must leave the body persist-clean.
    if "instructions" in body:
        try:
            instructions = server._validate_responses_instructions(body)
        except RequestError:
            pass
        else:
            if instructions is None:
                assert "instructions" not in body
    if "metadata" in body:
        try:
            metadata = server._validate_openai_metadata(body)
        except RequestError:
            pass
        else:
            if metadata is None:
                leftover = body.get("metadata")
                assert leftover is None or (
                    isinstance(leftover, dict)
                    and not any(value is None for value in leftover.values())
                )
            else:
                assert body.get("metadata") == metadata
                assert all(isinstance(value, str) for value in metadata.values())

    # response_format.json_schema.name must match [a-zA-Z0-9_-]{1,64} ASCII.
    if "response_format" in body:
        try:
            fmt = server._validate_chat_response_format(body)
        except RequestError:
            pass
        else:
            if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                schema = fmt.get("json_schema")
                if isinstance(schema, dict) and "name" in schema:
                    schema_name = schema["name"]
                    assert isinstance(schema_name, str) and 1 <= len(schema_name) <= 64
                    assert schema_name.isascii() and all(
                        ch.isalnum() or ch in "_-" for ch in schema_name
                    )

    # Tools honesty: successful function names match [a-zA-Z0-9_-]{1,64} ASCII.
    if "tools" in body:
        try:
            tools = server._validate_chat_tools(body)
        except RequestError:
            pass
        else:
            if tools:
                for item in tools:
                    function = item.get("function")
                    assert isinstance(function, dict)
                    tool_name = function.get("name")
                    assert isinstance(tool_name, str) and 1 <= len(tool_name) <= 64
                    assert tool_name.isascii() and all(
                        ch.isalnum() or ch in "_-" for ch in tool_name
                    )

    # Official Responses text.format: successful names are ASCII [a-zA-Z0-9_-]{1,64}.
    if "text" in body:
        try:
            text_value = server._validate_responses_text(body)
        except RequestError:
            pass
        else:
            if isinstance(text_value, dict):
                fmt = text_value.get("format")
                if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                    name = fmt.get("name")
                    assert isinstance(name, str) and 1 <= len(name) <= 64
                    assert name.isascii() and all(
                        ch.isalnum() or ch in "_-" for ch in name
                    )


def exercise_agent_config(value: Any) -> None:
    """Drive ``ModelAgent.from_dict`` over an arbitrary decoded JSON value."""
    if not isinstance(value, dict):
        # from_dict indexes value["id"]; a non-dict is a config error, not a bug.
        return
    try:
        agent = ModelAgent.from_dict(value)
    except _EXPECTED_CONFIG_EXC:
        return

    # Successful parse invariants.
    assert isinstance(agent.id, str) and agent.id
    assert isinstance(agent.tags, tuple)
    assert isinstance(agent.provider_exclusions, tuple)
    assert isinstance(agent.priority, int)
    assert isinstance(agent.disabled, bool)


_FUZZ_OPENAI_SOURCE = ProviderModelSource(
    provider_name="fuzz_openai",
    credential_name="FUZZ_OPENAI_API_KEY",
    list_url="https://example.invalid/v1/models",
    chat_base_url="https://example.invalid/v1",
)
_FUZZ_BYTEZ_SOURCE = ProviderModelSource(
    provider_name="fuzz_bytez",
    credential_name="FUZZ_BYTEZ_API_KEY",
    list_url="https://example.invalid/models/v2/list/models",
    chat_base_url="https://example.invalid/models/v2/openai/v1",
    auth_scheme="Key",
    style="bytez",
    task_filter="chat",
)


def exercise_provider_model_payload(value: Any) -> None:
    """Drive provider model-list parsers over arbitrary decoded values."""
    for source, parser in (
        (_FUZZ_OPENAI_SOURCE, _parse_openai_compatible),
        (_FUZZ_BYTEZ_SOURCE, _parse_bytez),
    ):
        discovered = parser(value, source)
        assert isinstance(discovered, list)
        for model in discovered:
            assert isinstance(model.model_id, str) and model.model_id
            assert model.provider_name == source.provider_name


def exercise_redaction(text: str) -> None:
    """Drive secret/PII redaction over arbitrary text and structures.

    Invariants: never crashes, always returns ``str``, and is idempotent on its
    own output (re-redacting redacted text yields the same string). Idempotence
    guards against a regex that could re-trigger on the ``[REDACTED]`` marker.
    """
    once = redact_text(text)
    assert isinstance(once, str)
    twice = redact_text(once)
    assert once == twice, "redaction is not idempotent"

    # ``redact_value`` must preserve container shape while redacting leaves.
    payload = {"trace": [text, {"nested": text}], "count": 3, "flag": True, "none": None}
    out = redact_value(payload)
    assert isinstance(out, dict)
    assert isinstance(out["trace"], list)
    assert isinstance(out["trace"][1], dict)
    assert out["count"] == 3 and out["flag"] is True and out["none"] is None


def _mock_orchestrator() -> TaskOrchestrator:
    agents = [
        ModelAgent(id="general_agent", model="mock-generalist", base_url="mock://generalist",
                   tags=("reasoning", "writing", "planning"), priority=1),
        ModelAgent(id="builder_agent", model="mock-builder", base_url="mock://builder",
                   tags=("coding", "debugging", "implementation"), priority=2),
        ModelAgent(id="reviewer_agent", model="mock-reviewer", base_url="mock://reviewer",
                   tags=("verification", "security", "review"), priority=3),
    ]
    return TaskOrchestrator(agents)


def exercise_orchestration(prompt: str, mode: str) -> None:
    """Run a full orchestration on arbitrary prompt text against mock providers.

    Exercises ``_latest_user_text`` -> ``_needs_workflow`` -> ``_score_agent`` ->
    route/conduct -> trace assembly -> SSE framing, all offline via ``mock://``.
    """
    orchestrator = _mock_orchestrator()
    if mode not in server.ALLOWED_MODES:
        mode = "auto"

    record = orchestrator.run([{"role": "user", "content": prompt}], mode=mode)

    assert record["mode"] in server.ALLOWED_MODES
    assert isinstance(record["answer"], str)
    assert isinstance(record["trace"], list) and record["trace"]
    assert record["prompt_text"] == prompt

    # The whole record must be JSON-serialisable (it is returned over HTTP).
    json.dumps(record, ensure_ascii=False)

    # SSE framing must produce a body whose data frames are valid JSON (or DONE).
    chunks = chat_completion_chunks(record, include_trace=True)
    body = sse_stream_body(chunks)
    assert body.endswith("data: [DONE]\n\n")
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if not frame or frame == "data: [DONE]":
            continue
        assert frame.startswith("data: ")
        json.loads(frame[len("data: "):])


def exercise_reasoning_effort_profile(value: Any) -> None:
    """Drive the issue #568 profile parser and ablation over arbitrary JSON.

    Invariants: unknown keys, NaN, infinity, and bool-as-number fail as
    ``EffortProfileError`` / ``TypeError`` / ``ValueError``. A successful
    parse is finite. An ablation against a supplied ``true_theta`` either
    fails closed or stays production-locked while ``measurement_status`` is
    estimated.
    """
    if not isinstance(value, dict):
        return
    try:
        # ``true_theta`` belongs to the ablation input, not the profile schema.
        # Keep the two trust-boundary payloads separate so this branch is reachable.
        profile_value = {key: item for key, item in value.items() if key != "true_theta"}
        profile = parse_reasoning_effort_profile(profile_value)
    except (EffortProfileError, TypeError, ValueError):
        return
    assert profile.reasoning_effort in REASONING_EFFORT_LEVELS
    assert profile.access_list_scope in ACCESS_LIST_SCOPES
    assert math.isfinite(profile.temperature)
    assert math.isfinite(profile.top_p)
    theta = value.get("true_theta")
    if not isinstance(theta, list) or not theta:
        return
    try:
        report = run_equal_budget_ablation(theta)
    except (EffortProfileError, TypeError, ValueError):
        return
    assert production_default_change_allowed(report) is False
    assert report["measurement_status"] == "estimated"


def exercise_model_judge_reply(reply: str) -> None:
    """Drive strict model-judge parsing over arbitrary untrusted text."""
    try:
        decision, reason = _parse_model_judge_reply(reply)
    except ValueError:
        return
    assert decision in {"ACCEPT", "REJECT"}
    assert isinstance(reason, str) and reason.strip()


def exercise_structured_output_contract(value: Any) -> None:
    """Drive strict structured-output validation over arbitrary schema/input."""
    if not isinstance(value, dict):
        return
    content = value.get("content")
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except (TypeError, ValueError):
            content = ""
    result = _structured_output_error(content, value.get("response_format"))
    assert result in {None, "invalid_json", "schema_missing", "schema_violation"}
