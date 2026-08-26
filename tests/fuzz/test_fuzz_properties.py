"""Property-based (Hypothesis) fuzz tests for untrusted-input surfaces.

These run in the normal ``pytest`` suite on every platform and Python version --
no native toolchain required -- and share the exact ``exercise_*`` invariant
checks used by the Atheris coverage-guided harnesses in ``fuzz/``. Hypothesis is
MPL-2.0 (permissive, no copyleft on your code).

The Atheris harnesses (``fuzz/fuzz_*.py``) provide coverage-guided fuzzing in CI;
this module provides fast, deterministic, always-on regression coverage of the
same invariants and shrinks any counterexample to a minimal repro.
"""

from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

from fuzz.targets import (
    exercise_agent_config,
    exercise_model_judge_reply,
    exercise_orchestration,
    exercise_pii_key,
    exercise_provider_model_payload,
    exercise_reasoning_effort_profile,
    exercise_redaction,
    exercise_request_body,
    exercise_structured_output_contract,
)

# Keep per-test wall time small so the suite stays cheap in CI.
_SETTINGS = settings(max_examples=200, deadline=None)

# JSON-ish values Hypothesis can build without recursion blowups.
_json_scalars = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text()
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=12), children, max_size=6),
    max_leaves=25,
)


@_SETTINGS
@given(st.binary(max_size=4096))
def test_request_body_never_crashes_on_raw_bytes(raw: bytes) -> None:
    exercise_request_body(raw)


@_SETTINGS
@given(_json_values.map(lambda v: json.dumps(v).encode("utf-8")))
def test_request_body_never_crashes_on_valid_json(raw: bytes) -> None:
    exercise_request_body(raw)


@_SETTINGS
@given(
    st.fixed_dictionaries(
        {
            "run_mode": st.sampled_from(["auto", "route", "conduct", "bogus", "", 3, None]),
            "messages": st.lists(
                st.fixed_dictionaries(
                    {
                        "role": st.sampled_from(["user", "system", "assistant", "tool", "root", 1]),
                        "content": st.text() | st.integers() | st.none(),
                    }
                ),
                max_size=5,
            ),
            "extra_field": st.text(max_size=8),
        }
    ).map(lambda v: json.dumps(v).encode("utf-8"))
)
def test_request_body_validators_on_structured_input(raw: bytes) -> None:
    exercise_request_body(raw)


def test_request_body_rejects_unhashable_message_role() -> None:
    exercise_request_body(b'{"messages":[{"role":[],"":[],"modnt":""}]}')


@_SETTINGS
@given(_json_values)
def test_agent_config_parser(value: object) -> None:
    exercise_agent_config(value)


@_SETTINGS
@given(
    st.builds(
        dict,
        id=st.text(),
        model=st.text() | st.integers() | st.none(),
        base_url=st.text(),
        priority=st.integers() | st.text() | st.none(),
        tags=st.lists(st.text(), max_size=5) | st.none(),
        disabled=st.booleans() | st.integers(),
    )
)
def test_agent_config_parser_shaped(value: dict) -> None:
    exercise_agent_config(value)


@_SETTINGS
@given(_json_values)
def test_provider_model_payload_parser_never_crashes(value: object) -> None:
    exercise_provider_model_payload(value)


@_SETTINGS
@given(st.text(max_size=4096))
def test_unprefixed_pii_keys_are_rejected(value: str) -> None:
    exercise_pii_key(value)


@_SETTINGS
@given(st.text(max_size=4096))
def test_redaction_never_crashes_and_is_idempotent(text: str) -> None:
    exercise_redaction(text)


@settings(max_examples=100, deadline=None)
@given(
    st.text(max_size=2048),
    st.sampled_from(["auto", "route", "conduct", "unknown"]),
)
def test_orchestration_on_arbitrary_prompt(prompt: str, mode: str) -> None:
    exercise_orchestration(prompt, mode)


@_SETTINGS
@given(st.text(max_size=4096))
def test_model_judge_parser_rejects_or_validates_arbitrary_text(reply: str) -> None:
    exercise_model_judge_reply(reply)


@_SETTINGS
@given(_json_values)
def test_reasoning_effort_profile_never_crashes(value: object) -> None:
    exercise_reasoning_effort_profile(value)


@_SETTINGS
@given(_json_values)
def test_structured_output_contract_never_crashes(value: object) -> None:
    exercise_structured_output_contract(value)
