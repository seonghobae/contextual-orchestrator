"""End-to-end durable provider catalog bootstrap contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time

import pytest

from contextual_orchestrator.credentials import (
    InMemoryCredentialBackend,
    get_credential,
    register_credential,
    set_backend,
)
from contextual_orchestrator.model_discovery import (
    DiscoveredModel,
    ProviderDiscoveryError,
    ProviderModelSource,
)
from contextual_orchestrator.privacy_policy_analysis import PrivacyPolicyAssessment
from contextual_orchestrator.provider_bootstrap import PROVIDER_CREDENTIAL_NAMES
from contextual_orchestrator.provider_catalog_bootstrap import (
    bootstrap_provider_catalog_runtime,
)
from contextual_orchestrator.provider_catalog_store import (
    InMemoryProviderCatalogStore,
)


def _environment() -> dict[str, str]:
    return {
        name: f"value-for-{name.casefold()}"
        for name in PROVIDER_CREDENTIAL_NAMES
    }


def _source(provider: str, credential: str) -> ProviderModelSource:
    return ProviderModelSource(
        provider_name=provider,
        credential_name=credential,
        list_url=f"https://{provider}.example/v1/models",
        chat_base_url=f"https://{provider}.example/v1",
    )


def _model(source: ProviderModelSource, model_id: str) -> DiscoveredModel:
    return DiscoveredModel(
        provider_name=source.provider_name,
        model_id=model_id,
        credential_name=source.credential_name,
        chat_base_url=source.chat_base_url,
        auth_scheme=source.auth_scheme,
        prompt_price_per_1k=1.0,
        completion_price_per_1k=2.0,
    )


def test_failed_provider_uses_persisted_last_known_good_model() -> None:
    """A later provider outage keeps its last successful compatible model."""
    set_backend(InMemoryCredentialBackend())
    try:
        openai = _source("openai", "OPENAI_API_KEY")
        openrouter = _source("openrouter", "OPENROUTER_API_KEY")
        store = InMemoryProviderCatalogStore()

        first = bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(openai, openrouter),
            discovery=lambda _sources: (
                [_model(openai, "gpt-live"), _model(openrouter, "router-live")],
                [],
            ),
            model_limit=4,
        )
        assert first.catalog_model_count == 2
        assert first.last_known_good_model_count == 0

        second = bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(openai, openrouter),
            discovery=lambda _sources: (
                [_model(openrouter, "router-new")],
                [ProviderDiscoveryError("openai", "secret-bearing detail")],
            ),
            model_limit=4,
        )
        assert second.live_discovered_model_count == 1
        assert second.catalog_model_count == 2
        assert second.last_known_good_model_count == 1
        assert second.catalog_refresh_failure_count == 1
        assert second.providers_with_errors == ("openai",)
        refreshes = second.as_dict()["catalog_refreshes"]
        assert isinstance(refreshes, list)
        assert [row["provider_account_id"] for row in refreshes] == [
            "openai_openai_api_key",
            "openrouter_openrouter_api_key",
        ]
        assert [row["refresh_status"] for row in refreshes] == [
            "failed",
            "succeeded",
        ]
        assert refreshes[0]["error_code"] == "provider_discovery_error"
        assert refreshes[1]["error_code"] is None
        assert all(row["finished_at"].endswith("+00:00") for row in refreshes)
        assert set(second.selected_agent_ids) == {
            "openai_gpt_live",
            "openrouter_router_new",
        }
        assert "secret-bearing detail" not in str(second.as_dict())
    finally:
        set_backend(None)


def test_empty_catalog_preserves_lkg_but_nonchat_success_withdraws_it() -> None:
    """Empty refresh is failure; authoritative non-chat success is withdrawal."""
    set_backend(InMemoryCredentialBackend())
    try:
        openai = _source("openai", "OPENAI_API_KEY")
        store = InMemoryProviderCatalogStore()
        bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(openai,),
            discovery=lambda _sources: ([_model(openai, "gpt-live")], []),
            model_limit=1,
        )

        empty = bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(openai,),
            discovery=lambda _sources: ([], []),
            model_limit=1,
        )
        assert empty.last_known_good_model_count == 1
        assert empty.catalog_model_count == 1

        try:
            bootstrap_provider_catalog_runtime(
                environ=_environment(),
                catalog_store=store,
                sources=(openai,),
                discovery=lambda _sources: (
                    [_model(openai, "text-embedding-3-small")],
                    [],
                ),
                model_limit=1,
            )
        except RuntimeError as error:
            assert "no persisted chat-compatible model" in str(error)
        else:
            raise AssertionError("non-chat-only authoritative catalog must fail")
    finally:
        set_backend(None)


def test_privacy_analysis_success_persists_and_empty_failure_preserves_lkg() -> None:
    """The opt-in bootstrap stores grounded evidence without erasing it on failure."""
    set_backend(InMemoryCredentialBackend())
    try:
        source = _source("openai", "OPENAI_API_KEY")
        model = _model(source, "gpt-live")
        model = type(model)(
            **{
                **model.__dict__,
                "privacy_policy_urls": ("https://provider.example/privacy",),
            }
        )
        evidence = PrivacyPolicyAssessment(
            subject_provider=source.provider_name,
            subject_credential=source.credential_name,
            subject_model=model.model_id,
            source_url=model.privacy_policy_urls[0],
            zero_data_retention_available=True,
            supports_no_training=True,
            supports_no_prompt_retention=True,
            evidence_quote="Prompts are not retained.",
            analyzer_provider="openrouter",
            analyzer_model="zdr-analyzer",
            observed_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
        )
        store = InMemoryProviderCatalogStore()

        first = bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(source,),
            discovery=lambda _sources: ([model], []),
            analyze_privacy_policies=True,
            privacy_analysis=lambda models: (list(models), [evidence]),
            model_limit=1,
        )
        assert first.privacy_assessment_count == 1
        assert store.privacy_assessments(source) == (evidence,)

        second = bootstrap_provider_catalog_runtime(
            environ=_environment(),
            catalog_store=store,
            sources=(source,),
            discovery=lambda _sources: ([model], []),
            analyze_privacy_policies=True,
            privacy_analysis=lambda models: (list(models), []),
            model_limit=1,
        )
        assert second.privacy_assessment_count == 1
        assert store.privacy_assessments(source) == (evidence,)
    finally:
        set_backend(None)


def test_unexpected_discovery_failure_restores_entire_credential_inventory() -> None:
    """An unclassified bootstrap failure must not leave unvalidated secrets promoted."""
    set_backend(InMemoryCredentialBackend())
    try:
        previous = {
            name: f"previous-value-for-{name.casefold()}"
            for name in PROVIDER_CREDENTIAL_NAMES
        }
        for name, value in previous.items():
            register_credential(name, value)

        def fail_discovery(_sources):
            raise RuntimeError("unexpected discovery parser failure")

        with pytest.raises(RuntimeError, match="unexpected discovery parser failure"):
            bootstrap_provider_catalog_runtime(
                environ=_environment(),
                catalog_store=InMemoryProviderCatalogStore(),
                sources=(_source("openai", "OPENAI_API_KEY"),),
                discovery=fail_discovery,
                model_limit=1,
            )

        assert {
            name: get_credential(name)
            for name in PROVIDER_CREDENTIAL_NAMES
        } == previous
    finally:
        set_backend(None)


def test_concurrent_bootstraps_report_only_their_own_refresh_evidence() -> None:
    """A shared store must not mix concurrent bootstrap refresh evidence."""
    set_backend(InMemoryCredentialBackend())
    try:
        openai = _source("openai", "OPENAI_API_KEY")
        openrouter = _source("openrouter", "OPENROUTER_API_KEY")
        store = InMemoryProviderCatalogStore()
        discoveries_ready = threading.Barrier(2)

        def run(source: ProviderModelSource):
            def discover(_sources):
                discoveries_ready.wait()
                return [_model(source, f"{source.provider_name}-live")], []

            return bootstrap_provider_catalog_runtime(
                environ=_environment(),
                catalog_store=store,
                sources=(source,),
                discovery=discover,
                model_limit=1,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            reports = tuple(executor.map(run, (openai, openrouter)))

        assert [
            [row.provider_account_id for row in report.catalog_refreshes]
            for report in reports
        ] == [
            ["openai_openai_api_key"],
            ["openrouter_openrouter_api_key"],
        ]
        assert len(store.refresh_evidence()) == 2
    finally:
        set_backend(None)


def test_concurrent_bootstrap_cannot_restore_stale_privacy_evidence() -> None:
    """A later refresh must win over an older delayed privacy write."""
    set_backend(InMemoryCredentialBackend())
    try:
        source = _source("openai", "OPENAI_API_KEY")
        stale_refresh_recorded = threading.Event()

        class _LaggyStore(InMemoryProviderCatalogStore):
            def record_success(
                self,
                source: ProviderModelSource,
                models: list[DiscoveredModel],
                *,
                eligible_model_ids: set[str],
                serving_tags: dict[str, tuple[str, ...]],
            ) -> None:
                super().record_success(
                    source,
                    models,
                    eligible_model_ids=eligible_model_ids,
                    serving_tags=serving_tags,
                )
                if any(model.privacy_policy_urls for model in models):
                    stale_refresh_recorded.set()

            def record_privacy_assessment_success(
                self,
                source: ProviderModelSource,
                assessments: list[PrivacyPolicyAssessment],
            ) -> None:
                if any(item.evidence_quote == "Stale quote." for item in assessments):
                    time.sleep(0.05)
                super().record_privacy_assessment_success(source, assessments)

        stale_model = _model(source, "gpt-live")
        stale_model = type(stale_model)(
            **{
                **stale_model.__dict__,
                "privacy_policy_urls": ("https://provider.example/privacy",),
            }
        )
        current_model = _model(source, "gpt-live")
        stale_evidence = PrivacyPolicyAssessment(
            subject_provider=source.provider_name,
            subject_credential=source.credential_name,
            subject_model=stale_model.model_id,
            source_url="https://provider.example/privacy",
            zero_data_retention_available=True,
            supports_no_training=True,
            supports_no_prompt_retention=True,
            evidence_quote="Stale quote.",
            analyzer_provider="openrouter",
            analyzer_model="zdr-analyzer",
            observed_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        )
        store = _LaggyStore()

        def bootstrap_stale() -> None:
            bootstrap_provider_catalog_runtime(
                environ=_environment(),
                catalog_store=store,
                sources=(source,),
                discovery=lambda _sources: ([stale_model], []),
                analyze_privacy_policies=True,
                privacy_analysis=lambda models: (list(models), [stale_evidence]),
                model_limit=1,
            )

        def bootstrap_current() -> None:
            def discover(_sources):
                assert stale_refresh_recorded.wait(timeout=5)
                return [current_model], []

            bootstrap_provider_catalog_runtime(
                environ=_environment(),
                catalog_store=store,
                sources=(source,),
                discovery=discover,
                analyze_privacy_policies=True,
                privacy_analysis=lambda models: (list(models), []),
                model_limit=1,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            tuple(executor.map(lambda fn: fn(), (bootstrap_stale, bootstrap_current)))

        assert store.privacy_assessments(source) == ()
    finally:
        set_backend(None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
