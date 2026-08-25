"""HTTP surface for the cost-review + routing hub: /healthz, rollup, batch routing."""

from __future__ import annotations

from pathlib import Path
import json
import threading
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextual_orchestrator import (  # noqa: E402
    CostRoutingCoordinator,
    InMemoryConfigStore,
    ModelAgent,
    PriceBook,
    PriceEntry,
    TaskOrchestrator,
)
from contextual_orchestrator.server import SecurityConfig, build_server  # noqa: E402


def _serve():
    agents = [ModelAgent(id="mock_worker", model="mock-a", base_url="mock://a", provider_name="mock",
                         tags=("reasoning", "coding", "writing"), priority=1)]
    orchestrator = TaskOrchestrator(agents)
    config = InMemoryConfigStore()
    price_book = PriceBook(config)
    price_book.set_price(PriceEntry("mock", "mock-a", prompt_price_per_1k=1.0, completion_price_per_1k=2.0))
    coordinator = CostRoutingCoordinator(orchestrator, config, price_book=price_book)
    token = "cost_token"
    server = build_server(orchestrator, port=0, security=SecurityConfig(auth_token=token), coordinator=coordinator)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], token


def _request(method, url, token=None, body=None, status_ok=(200, 201, 202)):
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:  # pragma: no cover - surfaced in asserts
        return exc.code, json.loads(exc.read())


def test_healthz_is_unauthenticated_and_ok() -> None:
    server, port, _token = _serve()
    try:
        status, body = _request("GET", f"http://127.0.0.1:{port}/healthz")
    finally:
        server.shutdown()
    assert status == 200
    assert body["status"] == "ok"
    assert body["service"] == "contextual-orchestrator"
    assert "batch_backend" in body


def test_chat_completion_reports_real_usage_and_records_cost() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, body = _request("POST", f"{base}/v1/chat/completions", token,
                                 {"model": "mock-a",
                                  "messages": [{"role": "user", "content": "hello there world"}],
                                  "attribution": {"team": "alpha", "company": "acme"}})
        assert status == 200
        assert body["usage"]["total_tokens"] > 0
        assert body["orchestration"]["channel"] == "sync"

        status, report = _request("GET", f"{base}/api/v1/cost_reports/rollup?dimension=team", token)
        assert status == 200
        values = {item["dimension_value"]: item for item in report["items"]}
        assert "alpha" in values
        assert values["alpha"]["record_count"] == 1
    finally:
        server.shutdown()


def test_batch_routing_via_chat_completion_and_results_retrieval() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, submitted = _request("POST", f"{base}/v1/chat/completions", token,
                                     {"model": "mock-a",
                                      "messages": [{"role": "user", "content": "batch this"}],
                                      "routing": {"latency_tolerant": True},
                                      "attribution": {"company": "acme"}})
        assert status == 202
        assert submitted["channel"] == "batch"
        job_id = submitted["job_id"]

        status, polled = _request("GET", f"{base}/api/v1/batch_routing_jobs/{job_id}", token)
        assert status == 200
        assert polled["is_complete"] is True

        status, retrieved = _request("POST", f"{base}/api/v1/batch_routing_jobs/{job_id}/results", token)
        assert status == 200
        assert retrieved["result_count"] == 1

        status, report = _request("GET", f"{base}/api/v1/cost_reports/rollup?dimension=company", token)
        assert report["grand_total"]["record_count"] == 1
    finally:
        server.shutdown()


def test_batch_routing_jobs_endpoint_submits_multiple_requests() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, job = _request("POST", f"{base}/api/v1/batch_routing_jobs", token, {
            "attribution": {"company": "acme"},
            "requests": [
                {"messages": [{"role": "user", "content": "one"}]},
                {"messages": [{"role": "user", "content": "two"}], "attribution": {"team": "beta"}},
            ],
        })
        assert status == 201
        assert job["request_count"] == 2

        status, retrieved = _request("POST", f"{base}/api/v1/batch_routing_jobs/{job['job_id']}/results", token)
        assert retrieved["result_count"] == 2

        status, records = _request("GET", f"{base}/api/v1/llm_usage_records", token)
        assert records["total_count"] == 2
    finally:
        server.shutdown()


def test_batch_routing_jobs_round_trip_caller_supplied_custom_ids() -> None:
    """Without caller custom_ids, results cannot be mapped back to requests
    on backends that do not preserve submission order (the OpenAI Batch
    contract does not) -- the submit response never discloses generated ids.
    """
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, job = _request("POST", f"{base}/api/v1/batch_routing_jobs", token, {
            "requests": [
                {"messages": [{"role": "user", "content": "one"}], "custom_id": "pair-7"},
                {"messages": [{"role": "user", "content": "two"}], "custom_id": "pair-42"},
            ],
        })
        assert status == 201

        status, retrieved = _request(
            "POST", f"{base}/api/v1/batch_routing_jobs/{job['job_id']}/results", token
        )
        assert status == 200
        assert {item["custom_id"] for item in retrieved["results"]} == {"pair-7", "pair-42"}
    finally:
        server.shutdown()


def test_batch_routing_jobs_reject_invalid_custom_ids() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        for bad_requests in (
            [
                {"messages": [{"role": "user", "content": "a"}], "custom_id": "dup"},
                {"messages": [{"role": "user", "content": "b"}], "custom_id": "dup"},
            ],
            [{"messages": [{"role": "user", "content": "a"}], "custom_id": "  "}],
            [{"messages": [{"role": "user", "content": "a"}], "custom_id": "x" * 65}],
            [{"messages": [{"role": "user", "content": "a"}], "custom_id": 7}],
        ):
            status, body = _request(
                "POST", f"{base}/api/v1/batch_routing_jobs", token,
                {"requests": bad_requests},
            )
            assert status == 400
            assert body["error"]["code"] == "invalid_request"
    finally:
        server.shutdown()


def test_cost_report_rejects_unknown_dimension() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, body = _request("GET", f"{base}/api/v1/cost_reports/rollup?dimension=bogus", token)
    finally:
        server.shutdown()
    assert status == 400
    assert body["error"]["code"] == "invalid_dimension"


def test_dimension_catalog_endpoint_lists_all_dimensions() -> None:
    server, port, token = _serve()
    base = f"http://127.0.0.1:{port}"
    try:
        status, body = _request("GET", f"{base}/api/v1/cost_attribution_dimensions", token)
    finally:
        server.shutdown()
    assert status == 200
    assert body["total_count"] == 7


if __name__ == "__main__":  # pragma: no cover
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok {_name}")
    print("ok")
