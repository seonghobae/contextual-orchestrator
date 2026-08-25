"""Durable batch-job registries: Valkey-backed state must survive a restart.

The real defect these tests pin down: batch job registries lived in
per-process dicts, so a server restart between submit and retrieve turned
paid-for work into a 404. With a Valkey-backed registry, a second
coordinator (standing in for the restarted process) sharing the same
Valkey client must see the first coordinator's submitted jobs and serve
their results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextual_orchestrator.batch_job_registry import (
    DEFAULT_RETENTION_SECONDS,
    JobRegistryFactory,
    ValkeyJsonMapping,
    build_job_registry,
)
from contextual_orchestrator.batch_routing import (
    BatchJob,
    BatchRequest,
    BatchResultItem,
    LocalBatchBackend,
)
from contextual_orchestrator.kv_config import InMemoryConfigStore


class FakeValkeyClient:
    """In-memory stand-in for redis.Redis limited to the hash surface used."""

    def __init__(self) -> None:
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.expirations: Dict[str, int] = {}

    def hget(self, key: str, field: str) -> Any:
        return self.hashes.get(key, {}).get(field)

    def hset(self, key: str, field: str, value: str) -> int:
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hdel(self, key: str, field: str) -> int:
        bucket = self.hashes.get(key, {})
        if field in bucket:
            del bucket[field]
            return 1
        return 0

    def hkeys(self, key: str) -> list:
        return list(self.hashes.get(key, {}))

    def hlen(self, key: str) -> int:
        return len(self.hashes.get(key, {}))

    def expire(self, key: str, seconds: int) -> bool:
        self.expirations[key] = seconds
        return True


def test_mapping_round_trips_dataclasses_and_plain_values() -> None:
    """Dataclasses, dataclass lists, and JSON scalars all survive the trip."""
    client = FakeValkeyClient()
    jobs = ValkeyJsonMapping(client, "jobs", decode=lambda raw: BatchJob(**raw))
    job = BatchJob(job_id="batch_1", backend="local", status="completed", request_count=2)
    jobs["batch_1"] = job
    assert jobs["batch_1"] == job

    results = ValkeyJsonMapping(client, "results", decode=lambda raw: BatchResultItem(**raw))
    items = [BatchResultItem(custom_id="a", answer="A"), BatchResultItem(custom_id="b", answer="B")]
    results["batch_1"] = items
    assert results["batch_1"] == items

    counts = ValkeyJsonMapping(client, "counts")
    counts["batch_1"] = 7
    assert counts["batch_1"] == 7
    assert counts.get("missing") is None
    assert "batch_1" in counts and len(counts) == 1


def test_mapping_delete_and_iteration_match_dict_semantics() -> None:
    """The registry honors the MutableMapping contract call sites rely on."""
    client = FakeValkeyClient()
    mapping = ValkeyJsonMapping(client, "jobs")
    mapping["one"] = {"n": 1}
    mapping["two"] = {"n": 2}
    assert sorted(mapping) == ["one", "two"]
    del mapping["one"]
    assert "one" not in mapping
    try:
        del mapping["one"]
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_writes_refresh_the_registry_retention_window() -> None:
    """Every write pushes the hash expiry forward so live registries persist."""
    client = FakeValkeyClient()
    mapping = ValkeyJsonMapping(client, "jobs", retention_seconds=123)
    mapping["job"] = {"ok": True}
    assert client.expirations["batch_job_registry:jobs"] == 123


def test_factory_without_client_hands_out_plain_dicts() -> None:
    """No Valkey configured -> the historical in-process dict behavior."""
    factory = JobRegistryFactory(None)
    assert factory.durable is False
    mapping = factory.mapping("jobs")
    assert isinstance(mapping, dict)


def test_build_job_registry_defaults_to_in_process_without_the_secret() -> None:
    """An unconfigured store must not change existing deployments."""
    factory = build_job_registry(InMemoryConfigStore())
    assert factory.durable is False


def test_jobs_submitted_before_a_restart_are_retrievable_after_it() -> None:
    """Two backends sharing one Valkey client model a process restart."""
    client = FakeValkeyClient()

    def runner(messages, mode):
        return {"answer": messages[-1]["content"].upper(), "mode": mode}

    first = LocalBatchBackend(runner=runner, job_registry=JobRegistryFactory(client))
    job = first.submit([BatchRequest(messages=[{"role": "user", "content": "hi"}])])

    restarted = LocalBatchBackend(runner=runner, job_registry=JobRegistryFactory(client))
    items = restarted.retrieve(job)
    assert [item.answer for item in items] == ["HI"]


def test_default_retention_is_a_week() -> None:
    """Documented default: abandoned jobs expire after seven days."""
    assert DEFAULT_RETENTION_SECONDS == 7 * 24 * 3600


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("test_batch_job_registry: all direct-run checks passed")
