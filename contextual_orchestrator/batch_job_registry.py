"""Durable, Valkey-managed state for batch-routing job registries.

Why this exists: every batch-routing job registry — the coordinator's
``job_id -> BatchJob`` map, the pg-llm-batch backends' tracked-request
maps, and the local backend's result stash — lived in per-process Python
dicts. A process restart between submit and retrieve silently discarded
every queued and completed job, turning ``GET /api/v1/batch_routing_jobs
/{id}/results`` into a 404 for work a caller already paid for. This
module lets those registries live in Valkey instead, so the queue
survives restarts and can be shared by more than one server process.

Design: :class:`ValkeyJsonMapping` is a ``MutableMapping[str, Any]``
backed by one Valkey hash, storing JSON documents. Dataclass values are
serialized with :func:`dataclasses.asdict`; a per-mapping ``decode``
callable rebuilds them on read. :func:`build_job_registry` returns a
factory that hands out either Valkey-backed mappings (when a Valkey URL
is configured and the ``redis`` client is importable) or plain dicts —
so every call site keeps identical mapping semantics and the default
deployment is unchanged.

The Valkey URL is deployment configuration with credentials, so it is
resolved through the KV credential registry
(``get_credential("batch_job_registry_valkey_url")``, with the config
store's secret surface as the injectable test fallback), never from
``os.getenv``.

The durable-row/registry split follows the transactional-outbox shape:
the durable record is the source of truth and the queue entry is only a
wake-up (Richardson, C. (2018). *Microservices patterns: With examples
in Java*. Manning; Kleppmann, M. (2017). *Designing data-intensive
applications*. O'Reilly).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, Optional

# Registry entries expire after this many seconds so abandoned jobs do
# not accumulate forever. Seven days comfortably outlives every batch
# backend's own completion window.
DEFAULT_RETENTION_SECONDS = 7 * 24 * 3600


def _encode(value: Any) -> str:
    """Serialize one registry value (dataclasses included) to JSON."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload: Any = {"__dataclass__": True, "value": dataclasses.asdict(value)}
    elif isinstance(value, list) and value and all(
        dataclasses.is_dataclass(item) and not isinstance(item, type) for item in value
    ):
        payload = {"__dataclass_list__": True, "value": [dataclasses.asdict(item) for item in value]}
    else:
        payload = {"value": value}
    return json.dumps(payload, ensure_ascii=False)


class ValkeyJsonMapping(MutableMapping):
    """One named registry stored as JSON documents in a Valkey hash.

    ``decode`` rebuilds rich values (for example dataclasses) from the
    stored JSON dict; when omitted, values come back as plain JSON data.
    Reads and writes go straight to Valkey — there is no local cache —
    so every process sharing the URL sees one consistent registry.
    """

    def __init__(
        self,
        client: Any,
        name: str,
        *,
        decode: Optional[Callable[[Any], Any]] = None,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    ) -> None:
        self._client = client
        self._key = f"batch_job_registry:{name}"
        self._decode = decode
        self._retention_seconds = retention_seconds

    def _decode_document(self, raw: Any) -> Any:
        document = json.loads(raw)
        value = document["value"]
        if self._decode is None:
            return value
        if document.get("__dataclass_list__"):
            return [self._decode(item) for item in value]
        if document.get("__dataclass__"):
            return self._decode(value)
        return value

    def __getitem__(self, job_id: str) -> Any:
        raw = self._client.hget(self._key, job_id)
        if raw is None:
            raise KeyError(job_id)
        # Reads refresh retention too: a registry that is only polled
        # after submission must not expire mid-use.
        self._client.expire(self._key, self._retention_seconds)
        return self._decode_document(raw)

    def __setitem__(self, job_id: str, value: Any) -> None:
        self._client.hset(self._key, job_id, _encode(value))
        # Retention is per-registry: any write pushes the whole hash's
        # expiry forward, so an active registry never expires mid-flight.
        self._client.expire(self._key, self._retention_seconds)

    def __delitem__(self, job_id: str) -> None:
        if not self._client.hdel(self._key, job_id):
            raise KeyError(job_id)

    def __iter__(self) -> Iterator[str]:
        for field_name in self._client.hkeys(self._key):
            yield field_name.decode() if isinstance(field_name, bytes) else str(field_name)

    def __len__(self) -> int:
        return int(self._client.hlen(self._key))


class JobRegistryFactory:
    """Hands out named job registries, Valkey-backed when configured."""

    def __init__(self, client: Any = None, *, retention_seconds: int = DEFAULT_RETENTION_SECONDS) -> None:
        self._client = client
        self._retention_seconds = retention_seconds

    @property
    def durable(self) -> bool:
        """True when registries survive a process restart."""
        return self._client is not None

    def mapping(self, name: str, *, decode: Optional[Callable[[Any], Any]] = None) -> MutableMapping:
        """Return the registry called ``name`` — a dict unless Valkey is configured."""
        if self._client is None:
            return {}
        return ValkeyJsonMapping(
            self._client, name, decode=decode, retention_seconds=self._retention_seconds
        )


def build_job_registry(config_store: Any) -> JobRegistryFactory:
    """Build the registry factory from the config store's secret surface.

    Resolves ``batch_job_registry_valkey_url`` through the KV credential
    registry first, then the config store's secret surface (the injectable
    test path). When it is unset,
    or the ``redis`` client package is not installed (it ships in the
    ``queue`` extra), registries stay in-process dicts — exactly the
    pre-Valkey behavior — so nothing changes for deployments that have
    not opted in.
    """
    from .credentials import get_credential

    try:
        url = get_credential("batch_job_registry_valkey_url")
    except Exception:  # noqa: BLE001 - no credential backend configured
        url = None
    if not url:
        get_secret = getattr(config_store, "get_secret", None)
        url = get_secret("batch_job_registry_valkey_url", None) if callable(get_secret) else None
    if not url:
        return JobRegistryFactory(None)
    try:
        import redis
    except ImportError:
        return JobRegistryFactory(None)
    client = redis.Redis.from_url(str(url))
    retention = DEFAULT_RETENTION_SECONDS
    get = getattr(config_store, "get", None)
    if callable(get):
        configured = get("routing", "batch_job_retention_seconds", DEFAULT_RETENTION_SECONDS)
        if type(configured) is int and configured >= 1:
            retention = configured
    return JobRegistryFactory(client, retention_seconds=retention)
