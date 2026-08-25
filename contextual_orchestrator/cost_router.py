"""Cost-aware routing coordinator: the LLM cost-review + routing hub.

This composes the existing :class:`~contextual_orchestrator.orchestrator.TaskOrchestrator`
with the cost ledger and the sync-vs-batch router, so the orchestrator becomes
the single control point for:

1. **Cost review** — every completion (sync *and* batch) writes a
   :class:`~contextual_orchestrator.cost_ledger.UsageRecord` with token counts +
   computed cost and full multi-dimensional attribution.
2. **Routing** — :class:`~contextual_orchestrator.batch_routing.RoutingPolicy`
   picks sync vs batch; the batch path is dispatched to a
   :class:`~contextual_orchestrator.batch_routing.BatchBackend` (pg-llm-batch in
   production, local in-process for the mock/standalone path).

All config (prices, thresholds, endpoints) is read from the injected KV config
store, never ``os.getenv``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .batch_routing import (
    BatchBackend,
    BatchJob,
    BatchRequest,
    BatchResultItem,
    EmbeddingBatchBackend,
    EmbeddingBatchRequest,
    EmbeddingBatchResultItem,
    LocalBatchBackend,
    LocalEmbeddingBatchBackend,
    RoutingHints,
    RoutingPolicy,
)
from .batch_job_registry import JobRegistryFactory, build_job_registry
from .cost_ledger import CostLedger, PriceBook
from .kv_config import InMemoryConfigStore
from .token_counting import HeuristicTokenCounter, build_token_counter

_EMBEDDING_CONFIG_CATEGORY = "routing"
_DEFAULT_EMBEDDING_MAX_TOKENS_PER_REQUEST = 280_000
_DEFAULT_EMBEDDING_MAX_CHARS_PER_PART = 240_000
_EMBEDDING_UNIT_RE = re.compile(r"\S+\s*|\s+", re.UNICODE)


class CostRoutingCoordinator:
    """Wire routing + cost accounting around a ``TaskOrchestrator``."""

    def __init__(
        self,
        orchestrator: Any,
        config_store: Any = None,
        *,
        price_book: Optional[PriceBook] = None,
        ledger: Optional[CostLedger] = None,
        token_counter: Any = None,
        routing_policy: Optional[RoutingPolicy] = None,
        batch_backend: Optional[BatchBackend] = None,
        embedding_batch_backend: Optional[EmbeddingBatchBackend] = None,
        postgres_dsn: Optional[str] = None,
        job_registry: Optional[JobRegistryFactory] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config_store or InMemoryConfigStore()
        self.price_book = price_book or PriceBook(self.config)
        self.ledger = ledger or CostLedger(self.price_book)
        self.token_counter = token_counter or (
            build_token_counter(postgres_dsn) if postgres_dsn else HeuristicTokenCounter()
        )
        self.policy = routing_policy or RoutingPolicy(self.config)
        # Job registries live in Valkey when the credential registry carries
        # batch_job_registry_valkey_url, so submitted jobs survive a process
        # restart; otherwise they are the historical in-process dicts. Built
        # before the backends so the default local backends share it and
        # their results survive a restart too.
        registry = job_registry if job_registry is not None else build_job_registry(self.config)
        self.job_registry = registry
        if batch_backend is None:
            client = getattr(orchestrator, "client", None)
            local_concurrency = getattr(client, "local_concurrency", 1)
            self.batch_backend = LocalBatchBackend(
                runner=lambda messages, mode: orchestrator.complete(messages, mode=mode),
                max_concurrency=local_concurrency,
                job_registry=registry,
            )
        else:
            self.batch_backend = batch_backend
        self.embedding_batch_backend: EmbeddingBatchBackend = (
            embedding_batch_backend
            or LocalEmbeddingBatchBackend(
                token_counter=self.token_counter, job_registry=registry
            )
        )
        # job_id -> submitted BatchJob (so poll/retrieve can be driven by id)
        self._batch_jobs = registry.mapping("batch_jobs", decode=lambda raw: BatchJob(**raw))
        # embeddings batch state: job handle + submitted requests + cached doc,
        # keyed by batch id so poll/retrieve is idempotent (usage recorded once).
        self._embedding_jobs = registry.mapping("embedding_jobs", decode=lambda raw: BatchJob(**raw))
        self._embedding_models = registry.mapping("embedding_models")
        self._embedding_requests = registry.mapping(
            "embedding_requests", decode=lambda raw: EmbeddingBatchRequest(**raw)
        )
        self._embedding_input_counts = registry.mapping("embedding_input_counts")
        self._embedding_part_counts = registry.mapping("embedding_part_counts")
        self._embedding_part_limits = registry.mapping("embedding_part_limits")
        self._embedding_documents = registry.mapping("embedding_documents")

    # ------------------------------------------------------------------
    # Provider / model resolution
    # ------------------------------------------------------------------
    def _served_provider_model(self, result: Dict[str, Any], fallback_model: str) -> tuple[str, str]:
        """Derive ``(provider, model)`` from the served agent in the trace."""
        trace = result.get("trace") or []
        agent_id = ""
        for row in trace:
            agent_id = row.get("served_agent_id") or row.get("agent_id") or agent_id
        if agent_id:
            try:
                agent = self.orchestrator._agent(agent_id)
                provider = agent.provider_name or _provider_from_base_url(agent.base_url)
                return provider or "unknown", agent.model or fallback_model
            except Exception:
                pass
        return "unknown", fallback_model

    # ------------------------------------------------------------------
    # Sync + batch completion
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        mode: str = "auto",
        attribution: Optional[Dict[str, Any]] = None,
        hints: Optional[Dict[str, Any]] = None,
        model_name: str = "contextual-orchestrator",
        workflow_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a request (sync or batch) and record its usage + cost.

        Sync requests run the orchestrator immediately and return the completion
        augmented with ``channel``, ``routing_reason``, ``usage``, and the
        ``usage_record_id``. Batch requests are dispatched to the batch backend
        and return a job envelope; their cost is recorded on retrieval.
        """
        routing_hints = hints if isinstance(hints, RoutingHints) else RoutingHints.from_mapping(hints)
        prompt_tokens_estimate = self.token_counter.count_messages(messages, model_name)
        decision = self.policy.decide(routing_hints, prompt_tokens_estimate)

        if decision.channel == "batch":
            request = BatchRequest(
                messages=messages,
                model=model_name,
                attribution=dict(attribution or {}),
                mode=mode,
            )
            job = self.submit_batch([request], metadata={"routing_reason": decision.reason})
            return {
                "channel": "batch",
                "routing_reason": decision.reason,
                "job_id": job.job_id,
                "backend": job.backend,
                "status": job.status,
                "request_count": job.request_count,
            }

        result = self.orchestrator.run(messages, mode=mode, workflow_run_id=workflow_run_id)
        record = self._record_completion(
            messages=messages,
            answer=result.get("answer", ""),
            route_mode=result.get("mode"),
            request_channel="sync",
            attribution=attribution,
            model_name=model_name,
            provider_model=self._served_provider_model(result, model_name),
            workflow_run_id=result.get("workflow_run_id"),
        )
        result["channel"] = "sync"
        result["routing_reason"] = decision.reason
        result["usage_record_id"] = record.usage_record_id
        result["usage"] = {
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
        }
        result["cost"] = {"cost_amount": record.cost_amount, "currency_code": record.currency_code}
        return result

    def _record_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        answer: str,
        route_mode: Optional[str],
        request_channel: str,
        attribution: Optional[Dict[str, Any]],
        model_name: str,
        provider_model: tuple[str, str],
        workflow_run_id: Optional[str],
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        provider, model = provider_model
        if prompt_tokens is None:
            prompt_tokens = self.token_counter.count_messages(messages, model)
        if completion_tokens is None:
            completion_tokens = self.token_counter.count_text(answer, model)
        return self.ledger.record_usage(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            request_channel=request_channel,
            route_mode=route_mode,
            workflow_run_id=workflow_run_id,
            attribution=attribution,
        )

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------
    def submit_batch(
        self,
        requests: List[BatchRequest],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Submit a batch of requests to the configured batch backend."""
        job = self.batch_backend.submit(requests, metadata=metadata)
        self._batch_jobs[job.job_id] = job
        return job

    def poll_batch(self, job_id: str) -> Dict[str, Any]:
        """Poll a previously submitted batch job by id."""
        job = self._require_job(job_id)
        return self.batch_backend.poll(job)

    def retrieve_batch(self, job_id: str) -> Dict[str, Any]:
        """Retrieve batch results and record usage + cost for each completion."""
        job = self._require_job(job_id)
        items: List[BatchResultItem] = self.batch_backend.retrieve(job)
        recorded: List[Dict[str, Any]] = []
        for item in items:
            provider_model = self._resolve_batch_provider_model(item)
            record = self._record_completion(
                messages=[{"role": "user", "content": ""}],
                answer=item.answer,
                route_mode=item.mode,
                request_channel="batch",
                attribution=item.attribution,
                model_name=item.model,
                provider_model=provider_model,
                workflow_run_id=job.job_id,
                prompt_tokens=item.prompt_tokens or None,
                completion_tokens=item.completion_tokens or None,
            )
            recorded.append(
                {
                    "custom_id": item.custom_id,
                    "answer": item.answer,
                    "usage_record_id": record.usage_record_id,
                    "cost_amount": record.cost_amount,
                    "currency_code": record.currency_code,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                }
            )
        return {
            "job_id": job_id,
            "backend": job.backend,
            "result_count": len(recorded),
            "results": recorded,
        }

    def _resolve_batch_provider_model(self, item: BatchResultItem) -> tuple[str, str]:
        provider = str(item.attribution.get("provider") or item.attribution.get("upstream_api") or "")
        if not provider:
            provider = "unknown"
        return provider, item.model

    def _require_job(self, job_id: str) -> BatchJob:
        job = self._batch_jobs.get(job_id)
        if job is None:
            raise KeyError(f"batch job {job_id!r} not found")
        return job

    # ------------------------------------------------------------------
    # Embeddings batch lifecycle
    # ------------------------------------------------------------------
    def submit_embeddings_batch(
        self,
        inputs: List[str],
        *,
        model: str = "contextual-orchestrator",
        attribution: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Submit a bulk embeddings batch to the configured embeddings backend.

        This is the surface naruon's batch embedding service submits to. Each
        input becomes one :class:`EmbeddingBatchRequest`; routing + cost stay
        owned by the orchestrator. Returns the backend job handle; the vectors
        and recorded cost are produced by :meth:`embeddings_batch_document`.
        """
        shared_attribution = dict(attribution or {})
        requests, part_counts, part_limits = self._build_embedding_requests(
            inputs, model=model, attribution=shared_attribution
        )
        job = self.embedding_batch_backend.submit(requests, metadata=metadata)
        self._embedding_jobs[job.job_id] = job
        self._embedding_models[job.job_id] = model
        self._embedding_requests[job.job_id] = requests
        self._embedding_input_counts[job.job_id] = len(inputs)
        self._embedding_part_counts[job.job_id] = part_counts
        self._embedding_part_limits[job.job_id] = part_limits
        return job

    def _build_embedding_requests(
        self,
        inputs: List[str],
        *,
        model: str,
        attribution: Dict[str, Any],
    ) -> tuple[List[EmbeddingBatchRequest], List[int], Dict[str, int]]:
        """Map original embedding inputs into token-budgeted provider parts."""
        max_tokens, max_chars = self._embedding_request_limits()
        requests: List[EmbeddingBatchRequest] = []
        part_counts: List[int] = []
        for source_index, text in enumerate(inputs):
            source_text = str(text)
            parts = self._split_embedding_input(
                source_text, model=model, max_tokens=max_tokens, max_chars=max_chars
            )
            part_count = len(parts)
            part_counts.append(part_count)
            for part_index, (part_text, token_count) in enumerate(parts):
                requests.append(
                    EmbeddingBatchRequest(
                        input_text=part_text,
                        model=model,
                        attribution=dict(attribution),
                        source_index=source_index,
                        part_index=part_index,
                        part_count=part_count,
                        token_count=token_count,
                    )
                )
        return requests, part_counts, {
            "max_tokens_per_part": max_tokens,
            "max_chars_per_part": max_chars,
        }

    def _embedding_request_limits(self) -> tuple[int, int]:
        """Return configured per-provider-call embedding ceilings.

        Azure's current embeddings limit is surfaced by LiteLLM as a 300,000
        token request cap. The default stays below that ceiling and also applies
        a character guard so heuristic token counters cannot accidentally send a
        very long no-whitespace string as one provider request.
        """
        max_tokens = _positive_int(
            self.config.get(
                _EMBEDDING_CONFIG_CATEGORY,
                "embedding_max_tokens_per_request",
                _DEFAULT_EMBEDDING_MAX_TOKENS_PER_REQUEST,
            ),
            _DEFAULT_EMBEDDING_MAX_TOKENS_PER_REQUEST,
        )
        max_chars = _positive_int(
            self.config.get(
                _EMBEDDING_CONFIG_CATEGORY,
                "embedding_max_chars_per_part",
                _DEFAULT_EMBEDDING_MAX_CHARS_PER_PART,
            ),
            _DEFAULT_EMBEDDING_MAX_CHARS_PER_PART,
        )
        return max_tokens, max_chars

    def _split_embedding_input(
        self,
        text: str,
        *,
        model: str,
        max_tokens: int,
        max_chars: int,
    ) -> List[tuple[str, int]]:
        """Split one original embedding input into provider-safe map parts."""
        if text == "":
            return [("", 0)]
        parts = self._force_token_safe_chunks(
            text, model=model, max_tokens=max_tokens, max_chars=max_chars
        )
        return parts or [("", 0)]

    def _force_token_safe_chunks(
        self,
        text: str,
        *,
        model: str,
        max_tokens: int,
        max_chars: int,
    ) -> List[tuple[str, int]]:
        """Recursively split text until each chunk fits token and char budgets."""
        if text == "":
            return [("", 0)]
        if len(text) > max_chars:
            chunks: List[tuple[str, int]] = []
            for start in range(0, len(text), max_chars):
                chunks.extend(
                    self._force_token_safe_chunks(
                        text[start : start + max_chars],
                        model=model,
                        max_tokens=max_tokens,
                        max_chars=max_chars,
                    )
                )
            return chunks

        token_count = self._count_embedding_tokens(text, model)
        if token_count <= max_tokens or len(text) <= 1:
            return [(text, token_count)]

        units = _EMBEDDING_UNIT_RE.findall(text)
        if len(units) > 1:
            chunks = []
            current = ""
            for unit in units:
                candidate = f"{current}{unit}"
                if current and (
                    len(candidate) > max_chars
                    or self._count_embedding_tokens(candidate, model) > max_tokens
                ):
                    chunks.extend(
                        self._force_token_safe_chunks(
                            current,
                            model=model,
                            max_tokens=max_tokens,
                            max_chars=max_chars,
                        )
                    )
                    current = unit
                else:
                    current = candidate
            if current:
                chunks.extend(
                    self._force_token_safe_chunks(
                        current,
                        model=model,
                        max_tokens=max_tokens,
                        max_chars=max_chars,
                    )
                )
            if len(chunks) > 1 or (chunks and chunks[0][0] != text):
                return chunks

        midpoint = max(1, len(text) // 2)
        return self._force_token_safe_chunks(
            text[:midpoint],
            model=model,
            max_tokens=max_tokens,
            max_chars=max_chars,
        ) + self._force_token_safe_chunks(
            text[midpoint:],
            model=model,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )

    def _count_embedding_tokens(self, text: str, model: str) -> int:
        """Count tokens for embedding split decisions, tolerating adapters."""
        try:
            value = int(self.token_counter.count_text(text, model))
        except Exception:
            value = len(text.split())
        if text and value <= 0:
            return 1
        return max(0, value)

    def embeddings_batch_document(self, batch_id: str) -> Dict[str, Any]:
        """Return the naruon-shaped batch document for ``batch_id``.

        Polls the backend; once complete, retrieves the vectors, records one
        usage record per embedding in the cost ledger (full attribution), and
        returns ``{batch_id, status, embeddings, cost_micro_usd, token_counts,
        total_tokens, part_count, model}``. Idempotent: the completed document is
        cached so a poll after completion never double-records cost.
        """
        cached = self._embedding_documents.get(batch_id)
        if cached is not None:
            return cached

        job = self._require_embedding_job(batch_id)
        requests = self._embedding_requests.get(batch_id, [])
        model_name = self._embedding_models.get(batch_id, "contextual-orchestrator")
        status = self.embedding_batch_backend.poll(job)
        if not status.get("is_complete"):
            return {
                "batch_id": batch_id,
                "status": status.get("status") or job.status,
                "backend": job.backend,
                "model": model_name,
                "embeddings": None,
            }

        items: List[EmbeddingBatchResultItem] = self.embedding_batch_backend.retrieve(job)
        request_by_custom_id = {request.custom_id: request for request in requests}
        input_count = self._embedding_input_counts.get(batch_id, len(requests))
        part_counts = self._embedding_part_counts.get(batch_id, [1] * input_count)
        part_limits = self._embedding_part_limits.get(batch_id, {})
        ordered = sorted(items, key=lambda item: item.index)
        parts_by_source: Dict[int, List[Dict[str, Any]]] = {index: [] for index in range(input_count)}
        for item in ordered:
            request = request_by_custom_id.get(item.custom_id)
            source_index = request.source_index if request else item.index
            prompt_tokens = int(item.prompt_tokens)
            if prompt_tokens <= 0 and request is not None:
                prompt_tokens = request.token_count or int(
                    self.token_counter.count_text(request.input_text, item.model)
                )
            parts_by_source.setdefault(source_index, []).append(
                {
                    "part_index": request.part_index if request else 0,
                    "embedding": item.embedding,
                    "prompt_tokens": max(0, prompt_tokens),
                    "model": item.model,
                    "attribution": dict(request.attribution) if request else {},
                }
            )

        embeddings: List[Dict[str, Any]] = []
        token_counts: List[int] = []
        total_cost_amount = 0.0
        currency_code = "USD"
        for source_index in range(input_count):
            parts = sorted(parts_by_source.get(source_index, []), key=lambda item: item["part_index"])
            if not parts:
                embeddings.append({"index": source_index, "embedding": []})
                token_counts.append(0)
                continue
            attribution = dict(parts[0]["attribution"])
            prompt_tokens = sum(int(part["prompt_tokens"]) for part in parts)
            model_name = str(parts[0]["model"])
            provider = str(
                attribution.get("provider") or attribution.get("upstream_api") or "unknown"
            )
            record = self.ledger.record_usage(
                provider=provider,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                request_channel="batch",
                route_mode="embedding",
                workflow_run_id=batch_id,
                attribution=attribution,
            )
            total_cost_amount += float(record.cost_amount)
            currency_code = record.currency_code
            token_counts.append(record.prompt_tokens)
            embeddings.append(
                {
                    "index": source_index,
                    "embedding": _weighted_average_embedding(
                        [(part["embedding"], int(part["prompt_tokens"])) for part in parts]
                    ),
                }
            )

        document = {
            "batch_id": batch_id,
            "status": "completed",
            "backend": job.backend,
            "model": model_name,
            "embeddings": embeddings,
            "token_counts": token_counts,
            "total_tokens": sum(token_counts),
            "part_count": len(requests),
            "input_part_counts": part_counts,
            "map_reduce": {
                "strategy": "token_budgeted_embedding_parts_weighted_average",
                **part_limits,
            },
            "cost_amount": round(total_cost_amount, 6),
            "currency_code": currency_code,
            "cost_micro_usd": int(round(total_cost_amount * 1_000_000)),
        }
        self._embedding_documents[batch_id] = document
        return document

    def complete_embeddings_batch(
        self,
        inputs: List[str],
        *,
        model: str = "contextual-orchestrator",
        attribution: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit an embeddings batch and return its document (one round-trip).

        For the local/in-process backend the batch completes synchronously, so
        this returns the finished ``completed`` document with vectors and cost.
        For an async backend (pg-llm-batch) it returns a ``{batch_id, status}``
        envelope the caller then polls via :meth:`embeddings_batch_document`.
        """
        job = self.submit_embeddings_batch(
            inputs, model=model, attribution=attribution, metadata=metadata
        )
        return self.embeddings_batch_document(job.job_id)

    def _require_embedding_job(self, batch_id: str) -> BatchJob:
        job = self._embedding_jobs.get(batch_id)
        if job is None:
            raise KeyError(f"embeddings batch job {batch_id!r} not found")
        return job

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def cost_report(
        self,
        dimension: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a cost rollup report grouped by ``dimension`` over a window."""
        return self.ledger.report(dimension, start, end)


def _provider_from_base_url(base_url: str) -> str:
    """Best-effort provider label from a base URL scheme/host."""
    if base_url.startswith("mock://"):
        return "mock"
    try:
        from urllib.parse import urlparse

        host = urlparse(base_url).hostname or ""
    except Exception:
        return ""
    return host


def _positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` when invalid."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _weighted_average_embedding(parts: List[tuple[List[float], int]]) -> List[float]:
    """Reduce mapped chunk vectors into one deterministic embedding vector."""
    vectors = [vector for vector, _weight in parts if vector]
    if not vectors:
        return []
    dimension = max(len(vector) for vector in vectors)
    total_weight = sum(max(1, int(weight)) for _vector, weight in parts)
    if total_weight <= 0:
        total_weight = len(parts)
    reduced: List[float] = []
    for offset in range(dimension):
        weighted_sum = 0.0
        for vector, weight in parts:
            weighted_sum += (vector[offset] if offset < len(vector) else 0.0) * max(1, int(weight))
        reduced.append(round(weighted_sum / total_weight, 8))
    return reduced
