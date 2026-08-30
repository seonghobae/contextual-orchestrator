"""Trusted provider bootstrap with durable normalized model-catalog persistence.

This command registers the complete credential inventory, performs provider-
isolated discovery, persists successful model metadata in PostgreSQL, retains
last-known-good models for failed providers, and constructs a bounded candidate
pool from the persisted catalog.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import threading
from typing import Callable, Mapping, Sequence

from .cost_ledger import PriceBook
from .credentials import (
    InMemoryCredentialBackend,
    PostgresCredentialBackend,
    get_backend,
    get_credential,
)
from .kv_config import InMemoryConfigStore
from .model_discovery import (
    PROVIDER_MODEL_SOURCES,
    DiscoveredModel,
    ProviderDiscoveryError,
    ProviderModelSource,
    agent_id_for,
    discover_all_models,
    refresh_price_book,
)
from .privacy_policy_analysis import (
    PrivacyPolicyAssessment,
    analyze_discovered_privacy_policies,
)
from .provider_bootstrap import (
    ProviderBootstrapError,
    _synchronize_durable_agent_pool,
    collect_provider_credentials,
    is_chat_serving_candidate,
    register_provider_credentials_atomically,
    select_provider_diverse_models,
    serving_tags_for_discovered,
)
from .provider_catalog_store import (
    CatalogRefreshEvidence,
    InMemoryProviderCatalogStore,
    PostgresProviderCatalogStore,
    ProviderCatalogStore,
)


_CATALOG_REFRESH_EVIDENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProviderCatalogSnapshot:
    """Effective persisted model snapshot after provider-isolated refresh."""

    models: tuple[DiscoveredModel, ...]
    live_model_count: int
    last_known_good_model_count: int
    refresh_failure_count: int
    providers_with_errors: tuple[str, ...]


@dataclass(frozen=True)
class ProviderCatalogBootstrapReport:
    """Secret-free evidence for one durable provider-catalog bootstrap.

    ``registered_credentials`` contains the credential names that remain in the
    credential registry after provider-isolated rollback has completed. It is
    therefore safe for a workflow to use as durable-registration evidence.
    """

    registered_credentials: tuple[str, ...]
    restored_credentials: tuple[str, ...]
    live_discovered_model_count: int
    catalog_model_count: int
    eligible_model_count: int
    last_known_good_model_count: int
    selected_agent_ids: tuple[str, ...]
    enabled_agent_ids: tuple[str, ...]
    durable_agent_pool: bool
    catalog_backend: str
    catalog_refresh_failure_count: int
    providers_with_errors: tuple[str, ...]
    priced_model_count: int
    privacy_assessment_count: int
    catalog_refreshes: tuple[CatalogRefreshEvidence, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON evidence contract without secret values."""
        return {
            "registered_credentials": list(self.registered_credentials),
            "restored_credentials": list(self.restored_credentials),
            "live_discovered_model_count": self.live_discovered_model_count,
            "catalog_model_count": self.catalog_model_count,
            "eligible_model_count": self.eligible_model_count,
            "last_known_good_model_count": self.last_known_good_model_count,
            "selected_agent_ids": list(self.selected_agent_ids),
            "enabled_agent_ids": list(self.enabled_agent_ids),
            "durable_agent_pool": self.durable_agent_pool,
            "catalog_backend": self.catalog_backend,
            "catalog_refresh_failure_count": self.catalog_refresh_failure_count,
            "providers_with_errors": list(self.providers_with_errors),
            "priced_model_count": self.priced_model_count,
            "privacy_assessment_count": self.privacy_assessment_count,
            "catalog_refreshes": [
                {
                    "provider_account_id": evidence.provider_account_id,
                    "refresh_status": evidence.refresh_status,
                    "observed_model_count": evidence.observed_model_count,
                    "eligible_model_count": evidence.eligible_model_count,
                    "error_code": evidence.error_code,
                    "started_at": evidence.started_at.isoformat(),
                    "finished_at": evidence.finished_at.isoformat(),
                }
                for evidence in self.catalog_refreshes
            ],
        }


def build_provider_catalog_store() -> ProviderCatalogStore:
    """Build a catalog store colocated with the active credential backend."""
    backend = get_backend()
    if isinstance(backend, PostgresCredentialBackend):
        return PostgresProviderCatalogStore(backend.connection_dsn)
    if isinstance(backend, InMemoryCredentialBackend):
        return InMemoryProviderCatalogStore()
    raise ProviderBootstrapError(
        "provider catalog requires a built-in atomic credential backend"
    )


def _restore_provider_credentials_atomically(
    previous_credentials: Mapping[str, str | None],
) -> tuple[str, ...]:
    """Restore one credential snapshot in a single built-in backend transaction."""
    backend = get_backend()
    ordered = tuple(sorted(previous_credentials))
    if isinstance(backend, InMemoryCredentialBackend):
        with backend._lock:  # noqa: SLF001 - package-internal rollback transaction
            for name in ordered:
                previous = previous_credentials[name]
                if previous is None:
                    backend._store.pop(name, None)  # noqa: SLF001
                else:
                    backend._store[name] = previous  # noqa: SLF001
        return ordered
    if isinstance(backend, PostgresCredentialBackend):
        with backend._connect() as connection:  # noqa: SLF001 - package transaction
            backend._ensure_schema(connection)  # noqa: SLF001
            with connection.cursor() as cursor:
                for name in ordered:
                    previous = previous_credentials[name]
                    if previous is None:
                        cursor.execute(
                            "DELETE FROM provider_credentials WHERE credential_name = %s",
                            (name,),
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO provider_credentials "
                            "(credential_name, encrypted_value, updated_at) "
                            "VALUES (%s, pgp_sym_encrypt(%s, %s), now()) "
                            "ON CONFLICT (credential_name) DO UPDATE SET "
                            "encrypted_value = EXCLUDED.encrypted_value, updated_at = now()",
                            (name, previous, backend._passphrase),  # noqa: SLF001
                        )
            connection.commit()
        return ordered
    raise ProviderBootstrapError(
        "provider credential rollback requires an atomic built-in backend"
    )


def _source_key(source: ProviderModelSource) -> tuple[str, str]:
    """Return the provider-account key shared by sources and model rows."""
    return (source.provider_name, source.credential_name)


def _model_key(model: DiscoveredModel) -> tuple[str, str]:
    """Return the provider-account key carried by one discovered model."""
    return (model.provider_name, model.credential_name)


def refresh_persisted_provider_catalog(
    store: ProviderCatalogStore,
    *,
    sources: Sequence[ProviderModelSource],
    registered_credentials: Sequence[str],
    discovered: Sequence[DiscoveredModel],
    errors: Sequence[ProviderDiscoveryError],
) -> ProviderCatalogSnapshot:
    """Persist account-local refreshes and return the effective LKG snapshot."""
    registered = set(registered_credentials)
    live_by_account: dict[tuple[str, str], list[DiscoveredModel]] = {}
    for model in discovered:
        live_by_account.setdefault(_model_key(model), []).append(model)

    failed_names = {error.provider_name for error in errors}
    effective: list[DiscoveredModel] = []
    last_known_good_count = 0
    refresh_failures = 0
    providers_with_errors: set[str] = set(failed_names)

    for source in sources:
        if source.credential_name not in registered:
            continue
        account_models = live_by_account.get(_source_key(source), [])
        failed = source.provider_name in failed_names
        if failed:
            store.record_failure(source, error_code="provider_discovery_error")
            refresh_failures += 1
        elif not account_models:
            store.record_failure(source, error_code="empty_provider_catalog")
            refresh_failures += 1
            providers_with_errors.add(source.provider_name)
        else:
            eligible_ids = {
                model.model_id
                for model in account_models
                if is_chat_serving_candidate(model)
            }
            tags = {
                model.model_id: serving_tags_for_discovered(model)
                for model in account_models
                if model.model_id in eligible_ids
            }
            store.record_success(
                source,
                account_models,
                eligible_model_ids=eligible_ids,
                serving_tags=tags,
            )

        persisted = store.serving_models(source)
        effective.extend(persisted)
        if failed or not account_models:
            last_known_good_count += len(persisted)

    unique: dict[tuple[str, str, str], DiscoveredModel] = {}
    for model in effective:
        unique[(model.provider_name, model.credential_name, model.model_id)] = model
    ordered = tuple(unique[key] for key in sorted(unique))
    return ProviderCatalogSnapshot(
        models=ordered,
        live_model_count=len(discovered),
        last_known_good_model_count=last_known_good_count,
        refresh_failure_count=refresh_failures,
        providers_with_errors=tuple(sorted(providers_with_errors)),
    )


DiscoveryFunction = Callable[
    [tuple[ProviderModelSource, ...]],
    tuple[list[DiscoveredModel], list[ProviderDiscoveryError]],
]
PrivacyAnalysisFunction = Callable[
    [Sequence[DiscoveredModel]],
    tuple[list[DiscoveredModel], list[PrivacyPolicyAssessment]],
]


def bootstrap_provider_catalog_runtime(
    *,
    environ: Mapping[str, str],
    require_all_credentials: bool = True,
    agents_db: str | None = None,
    model_limit: int = 16,
    catalog_store: ProviderCatalogStore | None = None,
    sources: Sequence[ProviderModelSource] = PROVIDER_MODEL_SOURCES,
    discovery: DiscoveryFunction | None = None,
    analyze_privacy_policies: bool = False,
    privacy_analysis: PrivacyAnalysisFunction = analyze_discovered_privacy_policies,
) -> ProviderCatalogBootstrapReport:
    """Register secrets, persist catalogs, and build the effective serving pool."""
    credentials = collect_provider_credentials(
        environ,
        require_all=require_all_credentials,
    )
    previous_credentials = {
        name: get_credential(name) for name in credentials
    }
    registered = register_provider_credentials_atomically(credentials)
    try:
        store = catalog_store or build_provider_catalog_store()
        source_tuple = tuple(sources)
        discover = discovery or (
            lambda requested_sources: discover_all_models(requested_sources)
        )
        live_models, errors = discover(source_tuple)
        privacy_assessments: list[PrivacyPolicyAssessment] = []
        if analyze_privacy_policies:
            live_models, privacy_assessments = privacy_analysis(live_models)
        assessments_by_account: dict[
            tuple[str, str], list[PrivacyPolicyAssessment]
        ] = {}
        for assessment in privacy_assessments:
            assessments_by_account.setdefault(
                (assessment.subject_provider, assessment.subject_credential), []
            ).append(assessment)
        # The store evidence log and privacy evidence writes are shared process
        # state. Keep the offset, refresh writes, privacy upserts, and tail
        # capture in one atomic boundary so concurrent bootstraps cannot
        # resurrect stale policy evidence after a newer refresh removed its
        # source rows.
        with _CATALOG_REFRESH_EVIDENCE_LOCK:
            evidence_offset = len(store.refresh_evidence())
            snapshot = refresh_persisted_provider_catalog(
                store,
                sources=source_tuple,
                registered_credentials=registered,
                discovered=live_models,
                errors=errors,
            )
            for source in source_tuple:
                account_assessments = assessments_by_account.get(
                    _source_key(source), []
                )
                if account_assessments:
                    store.record_privacy_assessment_success(
                        source, account_assessments
                    )
            catalog_refreshes = store.refresh_evidence()[evidence_offset:]
        privacy_assessment_count = (
            sum(
                len(store.privacy_assessments(source))
                for source in source_tuple
                if source.credential_name in registered
            )
            if analyze_privacy_policies
            else 0
        )
        failed_provider_names = {error.provider_name for error in errors}
        failed_credentials = {
            source.credential_name
            for source in source_tuple
            if source.credential_name in registered
            and (
                source.provider_name in failed_provider_names
                or not any(
                    _model_key(model) == _source_key(source)
                    for model in live_models
                )
            )
        }
        restored_credentials = _restore_provider_credentials_atomically(
            {
                name: previous_credentials.get(name)
                for name in failed_credentials
            }
        ) if failed_credentials else ()

        usable_models = tuple(
            model
            for model in snapshot.models
            if get_credential(model.credential_name)
        )
        if not usable_models:
            raise ProviderBootstrapError(
                "provider bootstrap has no persisted chat-compatible model with a usable credential"
            )

        price_book = PriceBook(InMemoryConfigStore())
        priced_count = refresh_price_book(list(usable_models), price_book)
        selected = select_provider_diverse_models(
            usable_models,
            limit=model_limit,
        )
        if not selected:
            raise ProviderBootstrapError(
                "provider bootstrap selected no persisted chat-compatible model"
            )
        selected_ids = tuple(agent_id_for(model) for model in selected)
        enabled_ids = (
            _synchronize_durable_agent_pool(agents_db, selected)
            if agents_db
            else ()
        )
        durable_registered_credentials = tuple(
            name for name in registered if get_credential(name) is not None
        )

        return ProviderCatalogBootstrapReport(
            registered_credentials=durable_registered_credentials,
            restored_credentials=tuple(restored_credentials),
            live_discovered_model_count=snapshot.live_model_count,
            catalog_model_count=len(snapshot.models),
            eligible_model_count=len(snapshot.models),
            last_known_good_model_count=snapshot.last_known_good_model_count,
            selected_agent_ids=selected_ids,
            enabled_agent_ids=enabled_ids,
            durable_agent_pool=bool(agents_db),
            catalog_backend=store.backend_name,
            catalog_refresh_failure_count=snapshot.refresh_failure_count,
            providers_with_errors=snapshot.providers_with_errors,
            priced_model_count=priced_count,
            privacy_assessment_count=privacy_assessment_count,
            catalog_refreshes=catalog_refreshes,
        )
    except Exception:
        try:
            _restore_provider_credentials_atomically(previous_credentials)
        except Exception as rollback_error:
            raise ProviderBootstrapError(
                "provider bootstrap failed and credential rollback could not complete"
            ) from rollback_error
        raise


def main(argv: Sequence[str] | None = None) -> None:
    """Run trusted durable provider bootstrap and print secret-free evidence."""
    parser = argparse.ArgumentParser(
        description=(
            "Register provider secrets, persist provider models, and refresh "
            "the effective serving pool."
        )
    )
    parser.add_argument(
        "--agents-db",
        default=os.environ.get("CONTEXTUAL_ORCHESTRATOR_AGENTS_DB") or None,
    )
    parser.add_argument("--model-limit", type=int, default=16)
    parser.add_argument(
        "--analyze-privacy-policies",
        action="store_true",
        help="Persist grounded provider-policy provenance; may incur provider charges.",
    )
    parser.add_argument(
        "--allow-partial-credentials",
        action="store_true",
        help="Permit a subset of the declared provider inventory (development only).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = bootstrap_provider_catalog_runtime(
        environ=os.environ,
        require_all_credentials=not args.allow_partial_credentials,
        agents_db=args.agents_db,
        model_limit=args.model_limit,
        analyze_privacy_policies=args.analyze_privacy_policies,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - subprocess/CLI boundary
    main()
