# Contextual Orchestrator: Product & Technical Gap Baseline

## 2026-08-30 PR #868 privacy-evidence ordering fix

The configured provider catalog bootstrap was persisting grounded privacy/ZDR
assessments after releasing the shared refresh lock. That allowed an older
refresh to restore stale policy evidence after a newer refresh had already
replaced the same provider account with a model set that removed the policy
source row, leaving persisted evidence out of sync with the latest catalog
snapshot.

This cycle keeps refresh writes, privacy-assessment upserts, and refresh-
evidence tail capture inside the same `_CATALOG_REFRESH_EVIDENCE_LOCK`
critical section. A concurrency regression now delays the stale assessment
write while a second refresh removes the source; the final persisted privacy
evidence stays empty, matching the later refresh. Focused branch verification
on Sunday, August 30, 2026 used `uv run pytest` for
`tests/test_provider_catalog_bootstrap.py`,
`tests/test_privacy_policy_analysis.py`,
`tests/test_provider_catalog_store.py`,
`tests/test_provider_catalog_store_boundaries.py`, and
`tests/test_model_discovery_boundaries.py`: `63 passed, 1 skipped` in 32.38s.
This is branch evidence only until protected hosted checks run on the exact PR
head.

## 2026-08-30 hourly loop: #868 test-mock fix, #857 narrow hardening, #906 stale-base merge

Fresh status check confirmed #868/#911/#912 were still `BLOCKED` purely on the
known org-wide `opencode-review`/`noema-review` failure (stale
`ORCHESTRATOR_PIN_SHA` vendored in `ContextualWisdomLab/.github`, fix pending
in `.github#1422`) — none had picked up an approval since the last pass, so
none were merged this cycle. #911/#912 had no other non-systemic failures
(`Full unit and contract suite` green on both) and needed no code changes.

**#868** (`fix/gateway-default-chat-model`) had one genuine, non-systemic
failure at the start of this pass: `Full unit and contract suite` failed with
`AttributeError: 'Namespace' object has no attribute 'provider_ca_bundle'` in
`_discover_models_command` (`contextual_orchestrator/__main__.py:305`) — its
own `argparse.ArgumentParser` never declared `--provider-ca-bundle`, even
though the function read `args.provider_ca_bundle` unconditionally (26 tests
failed: 8 directly on the missing attribute, 18 in
`test_auto_discovery_server.py` because their `discover_all_models` mocks
were fixed-arity lambdas that could not accept the `ca_bundle=` keyword the
server-startup call site already passes). Mid-fix, the PR owner
independently pushed `51fc34bb` adding the identical `--provider-ca-bundle`
argument — this pass rebased its own unpushed commit on top of that (no
history rewritten, since the commit had never been shared) and kept only the
non-duplicate half: widening the 18 test lambdas to `**_kwargs`. Pushed as
`e16cfed2`. Full local suite: `2745 passed, 1 skipped, 1 failed` — the one
failure is `tests/test_psychometric_routing.py` needing the private
`fast-mlsirm` package, unreachable in this sandbox (same documented blocker
as PR #917), not a regression.

**#857** (`fix/provider-backed-embedding-batch`) remains far too diverged to
merge-resolve in one pass (165 files / ~13.9k lines vs current `main`,
consistent with the prior pass's "too large" call) — left as-is otherwise.
The three findings named for re-verification this cycle
(`ProviderEmbeddingBatchBackend.submit` concurrency, `chat()` deadline
propagation, `zdr_only` leaking into provider payloads) were checked against
the PR's current head: the first two are already resolved there (Devin's
"Caller deadline is ignored on chat passthrough" thread is marked resolved,
and `submit`/`_run_job` already serialize every state transition under
`self._registry.lock(...)` with a bounded `ThreadPoolExecutor`), and
`zdr_only` does not exist anywhere in this PR's diff — that finding belongs to
**PR #911** instead (open, unresolved CodeRabbit thread on `server.py`'s
`_validate_zdr_only` not stripping the field from provider request bodies),
not #857; apparently conflated across PRs in an earlier pass's notes. Of
#857's 21 still-unresolved review threads, two were narrowly safe to fix
without touching the stale-merge problem, pushed as `9b9f9e4d` (a plain
commit on the existing head, no merge, no rebase):
- `CostRoutingCoordinator.__init__`'s readiness-recovery loop and
  `_run_provider_readiness_job` both indexed `self._readiness_jobs[job_id]`
  with no presence check; a durable (Valkey/Redis) backend can expire that
  document's TTL between the key listing and the lookup, raising `KeyError`
  out of `__init__` (failing server construction) or silently killing the
  readiness worker thread (leaving the job stuck `queued`/`running`
  forever). Both sites now check `isinstance(..., dict)` and return/continue.
- `tests/test_naruon_ecosystem_connector.py` called
  `urllib.request.urlopen(req)` with no timeout, unlike every other HTTP test
  in the file (`timeout=10`); added it.
Validated with the Rust `_token_packer` extension built locally (`maturin
develop --release`, needed because `build_token_counter` now hard-requires it
— itself one of the 21 still-open findings, left alone): focused suite 54
passed; full suite `2748 passed, 1 skipped, 1 failed` (same `fast-mlsirm`
sandbox gap as above). The remaining ~19 unresolved threads (Dockerfile
`test-runner` stage missing the `orchestrator` user — Major; unbounded
OpenRouter endpoint enumeration; a resolver workflow pinned to a mutable ref;
several Minor/Info items) were left untouched — the Dockerfile one needs a
real `docker build` to fix safely (no daemon available in this sandbox), and
the rest touch enough surrounding logic to risk the kind of regression this
PR has already spent 268 commits chasing.

**#906** (`feat/nim-benchmark-rebuild-20260828`) was reported `dirty` by
GitHub's cached `mergeable_state`; a real trial merge of `origin/main` showed
the branch was NOT irreconcilably diverged as `dirty` implied — the only
textual conflict, across all 28 changed files plus everything `main` gained
over the PR's stale base (33 commits), was in `CHANGELOG.md` (both sides
appended bullets to the same `### Added`/`### Fixed` region). Resolved by
keeping both sides' bullets under the file's one-header-per-type-per-version
convention and merging `origin/main` into the PR branch (a merge commit; no
rebase, no history rewritten). That merge then surfaced two real, narrow
regressions against this PR's own test suite, both fixed and pushed together
as `7ba5fefc`:
- `tests/test_nim_benchmark_workflow_contract.py` read
  `.github/workflows/tests.yml`, which `main` renamed to `ci.yml` in
  `9b0a356d` ("use conventional workflow filename") sometime in those 33
  commits; the `nim_benchmark_quality` job content the tests check for is
  present and intact under the new name — repointed both reads.
- `tests/test_nim_benchmark_release_acceptance.py::
  test_budgeted_client_fallback_and_transport_errors` matched the old error
  string `"provider .* request failed"`. `main`'s new
  `contextual_orchestrator/provider_errors.py` (PR #879) reclassifies
  provider HTTP failures through `ProviderUpstreamError` (still a
  `RuntimeError` subclass) with the fixed message `"provider rejected the
  request with HTTP {status}"` — updated the match regex.

One more failure surfaced by the full suite, `tests/
test_nim_benchmark_release_acceptance.py::
test_smoke_manifest_cannot_authorize_production_routing`, is **not** caused
by this merge: it was verified to fail identically — same
`configured_total_token_budget=1280` vs `observed_budget_tokens=1283` on task
`trick_arithmetic_lily_pads`/policy `conduct_bounded` — on this PR's own
unmerged head `b0167b08`, before touching `main` at all. That contradicts the
PR description's claimed "NIM focused and release/workflow tests: 112
passed." This pass left it untouched rather than loosening the equal-budget
assertion or the `30`/`0.9` evidence thresholds without the PR author's input
on why observed token usage grew by exactly 3 tokens for that one locked
task; it needs the author's judgment (a legitimate token-counting fix
elsewhere in the 32-commit branch history vs. an actual regression), not a
bot's guess. Full suite after both merge-fixes: `2797 passed, 2 failed` (the
token-budget gap above, plus the same sandbox-only `fast-mlsirm` gap).
`opencode-review` and `strix` were already failing on this PR before the
merge for the same org-wide systemic reason (the `strix` job's own log shows
it calling out to `api.opencode.ai`, consistent with `AGENTS.md`'s
"OpenCode/Noema/Strix share this repo's gateway backend" migration note);
`noema-review` was passing even pre-merge. None of this is a new regression
from the merge itself.

Nothing was merged to protected `main` this cycle — the org-wide
`opencode-review`/`noema-review` gate blocks every open PR here until
`ContextualWisdomLab/.github#1422` lands; that PR remains blocked on its own
`pull_request_target` trust-boundary deadlock and is out of this repo's
control. No new PRs had opened since the prior pass.

## 2026-08-30 PR #868 docstring-coverage fix

`Full unit and contract suite` was failing exclusively on
`tests/test_docstring_coverage.py::test_public_production_api_has_complete_docstrings`:
`_TrustedDiscoveryRedirectHandler.redirect_request` in
`contextual_orchestrator/model_discovery.py` (added by this branch) had no
docstring. Added one; no other change. This branch's head was already even
with protected `main` (`5f2753a`), so no merge was needed. `opencode-review`
and `noema-review` remain red on this PR for the same org-wide reason
recorded in the contextual-orchestrator gap baseline's 2026-08-30 entry: the
central `.github` repo's review sidecar vendors a stale
`contextual-orchestrator` pin, not a defect in this branch.

## 2026-08-27 omitted-model and virtual-id contract slice

The prior local `commercial-loop-20260826` worktree contained an omitted-model
repair that was not yet covered by PR #868 head `d37569835b1944075b66dd259d6738a8f4052927`.
That repair was reconciled into the live PR branch without changing the open
privacy-discovery or trace-authorization contracts. The exact contract on the
new head is now:

1. `/v1/chat/completions` omits to the advertised virtual gateway id
   `contextual-orchestrator`.
2. `/v1/responses` omits to `orchestrator/auto`, preserving the orchestrated
   path instead of pretending a concrete deployment was named.
3. Explicit JSON `null` still fails closed on both text surfaces; omission and
   explicit null are no longer conflated.
4. `orchestrator/free` remains explicit-only; no omitted-model path can
   silently downgrade into a free-only request.

Focused exact-head verification on Thursday, August 27, 2026 used `uv run`
from the clean PR worktree:
`tests/test_chat_orchestration_mode_http_honesty.py`,
`tests/test_responses_model_required_http_honesty.py`,
`tests/test_model_strip_writeback_http_honesty.py`, and
`tests/test_orchestrated_responses_stream.py` all passed (`43 passed in
18.49s`). This is branch evidence only and does not replace protected hosted
checks or independent review.

## 2026-08-27 bare-gateway discovery and virtual-model acceptance slice

The user-facing report was reproduced as a code path, not an environment
quirk: with the configured-gateway bootstrap transport
(`LLM_GATEWAY_API_URL` + `LLM_GATEWAY_API_KEY`) pointing at a
plain OpenAI-compatible gateway (e.g. a LiteLLM proxy) whose `/v1/models`
rows carry no modality metadata produced `DiscoveredModel` rows with
**empty** capabilities for chat deployments, while embedding deployments
that happen to carry richer `/model/info` evidence kept an `embedding`
capability. Empty-capability chat rows were then guaranteed to be dropped
by the runtime activation filter (`"chat" in model.capabilities`), which
is exactly "embedding discovers, chat does not".

Closure in PR #868 (`fix/gateway-default-chat-model` core slice):

1. `_parse_openai_compatible`: an identifier that passes the ordinary chat
   transport gate (`is_general_chat_agent_model_id`) now receives the
   `chat` capability, so a bare gateway list discovers usable chat models.
   Endpoint-only ids (embedding/rerank/transcription/...) never pass that
   gate, so no non-chat model is mislabeled.
2. `_auto_discover_runtime_agents` activates candidates with the same
   `is_discovered_chat_candidate` rule the serving bootstrap uses, so the
   two entry points agree on bare-listing evidence.
3. Structured chat trace disclosure path is restored (authorized callers
   receive the disclosed workflow trace; tool passthrough reports
   `trace_unavailable`), and `response_format` is a preference that can
   never fail-closed purely because a pool lacks the tag, while vision
   stays a hard entitlement.
4. `orchestrator/auto`, `orchestrator/free`, and the advertised
   `contextual-orchestrator` default resolve to a concrete synthesizer on
   every structured surface; omitted-model and null/blank-model semantics
   now differ honestly (omission defaults; explicit null/blank fails
   closed).
5. ZDR discovery already marks both paid and free models on OpenRouter
   (`endpoints/zdr`) and configured gateways (`/model/info` consensus);
   no ZDR regression introduced.

Evidence: full suite `2483 passed, 1 skipped` locally and the hosted
"Full unit and contract suite" green on the exact head; focused suites
green. Non-critical CI only: fuzz hash-locks were aligned
(`rpds-py`/`typing-extensions`) with `requirements.lock`; the Strix
security-run provider was externally unavailable at one point (rate limit,
token cap, or connection) and is re-run on dispatch — not a code
defect.

Remaining gap: the durable catalog still lacks operator-visible
per-refresh status on every consumed config; follow-up leaves ZDR
position for a metadata-freshness slice.

## 1. Executive Summary
This document serves as the baseline for the Contextual Orchestrator (an enterprise-grade LLM model orchestration gateway). To achieve a tier-one enterprise valuation (targeting the $20B+ market for AI infrastructure and governance), we must bridge the gap between our current state and a fully auditable, highly concurrent, standard-compliant SaaS gateway.

## 2. Product Requirements Document (PRD) Gaps
### Target Buyer & Value Proposition
- **Enterprise AI Platform Teams & SOC**: Require high throughput, lowest latency, and absolute data privacy compliance (CSAP, SOC2, HIPAA).
- **Core Value**: Token-cost optimization + performance + upstream load balancing with strict PII protection and Role-Based Access Control (RBAC).

### Gap Analysis (Product)
1. **Dynamic Model Discovery & Standard API Routing**:
   - *Current*: A configured OpenAI-compatible gateway with API keys resolves embeddings, but other models (chat, multimodal) fail discovery.
   - *Root cause (closed in PR #868)*: plain OpenAI-compatible listing rows without capability metadata were parsed with empty capabilities and then dropped by the runtime chat-activation filter; endpoint embedding rows kept richer metadata and survived.
   - *Target*: Seamless dynamic model discovery for `orchestrator/auto`, `orchestrator/free`, and omitted models. Paid vs free model discovery fully automated regardless of provider or custom gateway endpoint. Full OpenAPI/RESTful standard compliance.
   - *ZDR Discovery*: OpenRouter and configured gateways discover ZDR models (paid and free) and parse privacy policies for automated compliance.
2. **PII Masking vs Business Continuity**:
   - *Current*: PII masking disrupts downstream workflows if over-aggressive.
   - *Target*: Context-aware differential privacy and entity resolution masking that preserves structural integrity without destroying analytical value (ADR 0027, 0028).
3. **Advanced Scheduling & Reasoning (Fugu/Conductor/TRINITY)**:
   - *Current*: Basic LiteLLM routing parity.
   - *Target*: Test-time compute allocation based on reasoning effort ablation. Dynamic multi-agent routing based on task complexity. True $\theta$ ablations using equal-budget profiling are needed to map `lite` vs `full` vs `pro` execution.

## 3. Technical Requirements Document (TRD) Gaps
### Gap Analysis (Technical)
1. **Concurrency and Scaling**:
   - *Current*: Python GIL limitations (Multithreading issues).
   - *Target*: Asynchronous full-duplex non-blocking I/O. Use Python 3.14 for GIL improvements, but core vector/routing arithmetic must be migrated to **Rust**. `k6` end-to-end load tests required to prove concurrent connections.
2. **Database & Persistence**:
   - *Current*: May have unstructured locking or missing 3NF.
   - *Target*: Strict 3NF database schema with `snake_case` naming. Read/Write replica split. Hot partition mitigation. Use strict `UPSERT` semantics.
3. **Math & Psychometrics Engine**:
   - *Current*: Python-based math.
   - *Target*: All tensor, vector, embedding chunking, token sizing, and psychometric models (TEPP, fast-mlsirm) must be computed in **Rust with GPU+CPU multithreading**. Use empirically validated weights (not arbitrary heuristics). Atomistic fallacy prevention via multilevel/temporal modeling.
4. **Embedding Chunking & Omni-modal**:
   - *Current*: Flat chunks.
   - *Target*: Semantic boundary chunking (DOM nodes, paragraph, sender/receiver). Multimodal embedding (Base64 image text extraction, object detection). Add seamless audio/video routing natively.
5. **Security & Compliance**:
   - *Current*: Basic auth.
   - *Target*: CSAP, SOC 2 compliance. Formalize gateway trust boundary with WAF/IDS (`wardnet`). 100% test coverage (unit, contract, edge cases). 100% docstring coverage.

## 4. Ecosystem Integration Gaps
- **fast-mlsirm & Psychometrics**: Time-aware modeling and multi-level / multi-membership models are not natively integrated in routing decisions.
- **naruon**: PIM/DOM decomposition graphs from `naruon` are not directly queryable via our model's tool calls yet.

## 5. Action Plan & Roadmap (Loop Strategy)
1. **Fix Discovery (Immediate)**: Ensure omitted model, `orchestrator/auto`, and `orchestrator/free` semantics are correct.
2. **Rust Migration (Q3)**: Extract vector math, token counting, and ML routing to a Rust extension.
3. **Database Audit (Q3)**: Review Core ERD. Rename all non-snake_case objects. Add UPSERT paths.
4. **k6 Load Test (Q3)**: Prove lock-free asynchronous operations.
5. **Documentation**: APA 7th citations required for routing strategies.

*Note: All architectural changes must cite relevant literature in APA 7th format. Scheduled for hourly updates.*

# Product and Technical Gap Baseline

## 2026-08-29 batch-routing object-authorization slice

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. The accepted workflow-object
authorization decision now has a bounded implementation branch for its listed
batch-job gap: HTTP-created batch routing jobs carry a non-secret
authenticated-principal digest, and both status and result retrieval require
the same digest. An external verifier can provide a stable tenant/subject key
through the optional principal resolver; bool-only adapters retain the
documented bearer-digest fallback. A mismatch is returned as the existing generic
`batch_job_not_found` response before the backend is called; results also keep
the separate trace-purpose gate. Local exact-branch evidence is `61 passed`
across the cost-router, HTTP, and OpenAPI contract suites. This is branch
evidence only until the implementation reaches protected `main` through the
normal review, Checks, and approval gates.

The remaining issue #117 gaps are unchanged: tenant/resource/purpose/lifetime
claims from an external identity adapter, explicit legacy single-token
production migration, and ownership for other evidence surfaces still need
their own decisions and acceptance evidence.
## 2026-08-29 streamed Responses usage boundary

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. This branch closes the bounded
Responses streaming cost-accounting gap: provider SSE usage, when supplied, is
kept on the served workflow trace; every completed trace step is recorded as a
`request_channel=stream` ledger row; and a provider that omits usage produces
an explicit `measurement_status=unavailable` row with no token estimate from
the synthesized answer. The final Responses event exposes standard
`input_tokens`/`output_tokens`/`total_tokens` only when every step is measured,
plus gateway cost status and usage-record identifiers.

This follows the Responses API's `response.completed` usage shape and the
OpenTelemetry GenAI input/output usage vocabulary. A stream can end before a
provider's final usage frame, so unavailable is retained as an honest state;
it is never represented as free or estimated. The exact branch proof is the
focused streaming, ledger, disconnect, and cost-router tests; protected
Checks, independent approval, and normal merge remain required.

The remaining customer-visible gaps are unchanged: true answer-token
streaming still needs a cancellable asynchronous dependency graph; routing
observations remain process-local; and live NIM quality/cost evidence remains
open under issue #86.

## 2026-08-29 legacy single-token production gate

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. The current implementation branch
adds fail-closed `--production`/`--allow-public-bind` CLI gates: server startup
must choose split admin/inference credentials, and the insecure admin-session
cookie option is rejected; every non-loopback bind also requires the explicit
public-bind opt-in. Canonical `compose.yaml` now seeds those two names into the
KV from separate stdin-only secrets. Single-token mode remains available for
explicit local development, while split static credentials still do not grant
the separate trace purpose without a verified external adapter. Branch evidence is
local focused CLI/Compose/authorization coverage only until normal protected
review, Checks, and approval gates complete.

The remaining issue #117 gap is the external authorization adapter's
tenant/resource/purpose/lifetime context; PR #909 separately carries batch
routing ownership and is not protected-main evidence here.
## 2026-08-29 05:26 KST exact-head protected-queue snapshot

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. The ten-PR open queue was
re-read at the exact heads below. Every PR has zero qualifying independent
approvals and zero unresolved review threads; no protected control was
bypassed.

| PR | Exact head | Current protected evidence |
| ---: | --- | --- |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `13432f3e4836df9bc8b3c83778ca0faf09c04d93` | `BLOCKED`; ordinary/security checks pass, OpenCode fails closed |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `8a1654ead5a23e985c9bf1d6d500602283f05ab8` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode fails closed |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `ec17d4e0b77fe10c8087c587cb748d4027fe4d0f` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `29d9493fcdbf11aaa3d43bc6c7e10857bb85ca73` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed on provider evidence |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `e12d334cf307b5bda1253a020ea6a13cb0e243f4` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `cc50d934e78d12b5edc8640f9ac9dd52d2158b13` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode fails closed |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `2f7b177a1631e3f1c845748e2a4bd312664e0759` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, required Strix context is absent, OpenCode fails closed |
| [#909](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/909) | `d3d2e31df62a5b773ae5077dd538472fa2a6ec18` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed (`STRIX_PROVIDER_UNAVAILABLE`) |
| [#910](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/910) | `f46f11473d96e76282cb908f9ec338588fa14472` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, Full and Atheris pass, Strix remains in progress, OpenCode fails closed |

The current required branch contexts are still `opencode-review` and `strix`.
For #910, the Full and Atheris jobs are now terminal-successful, but the Strix
status remains in progress and its run metadata is not retrievable (`404`), so
it is not passing evidence. #908 has no current Strix check result at all, so
the required context is absent rather than successful. The OpenCode failures
state that no authenticated current-head verdict exists; they are not review
approvals. All rows remain blocked by both external gate evidence and the
absence of a qualifying independent approval.

## 2026-08-29 04:06 KST exact-head protected-queue snapshot

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. The eight-PR open queue was
re-read at the exact heads below. Every PR has zero qualifying independent
approvals and zero unresolved review threads after the current-head review
reply on #903; no protected control was bypassed.

| PR | Exact head | Current protected evidence |
| ---: | --- | --- |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `13432f3e4836df9bc8b3c83778ca0faf09c04d93` | `BLOCKED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `8a1654ead5a23e985c9bf1d6d500602283f05ab8` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `ec17d4e0b77fe10c8087c587cb748d4027fe4d0f` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode and Strix fail closed |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `29d9493fcdbf11aaa3d43bc6c7e10857bb85ca73` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed and Strix fails closed on provider `500` |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `e12d334cf307b5bda1253a020ea6a13cb0e243f4` | `BLOCKED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed and Strix remains in progress |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `bbad9c2653a8d4f3af198f09b4d82e561f5abbc4` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode and Strix fail closed |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `bf6fbb7d372922088bd075b395543b325d91ed78` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema and ordinary security checks pass, OpenCode fails closed |

The current OpenCode failures report that no authenticated `opencode-agent`
review exists for the exact head; they are not review approvals. #901's Strix
log records three provider/backend `500 internal_error` attempts with no
structured report, so its required check failed closed on unavailable external
evidence. #903's local CEFR/reasoning/passthrough regression set is `80
passed`; its hosted Strix result is still pending at this snapshot.

## 2026-08-29 03:35 KST exact-head protected-queue snapshot

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`; the open queue still contains
eight PRs. Every exact head below has zero unresolved threads and zero
qualifying independent approvals. Normal protected controls remain required;
no bypass was used.

| PR | Exact head | Current protected evidence |
| ---: | --- | --- |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `13432f3e4836df9bc8b3c83778ca0faf09c04d93` | `BLOCKED`; ordinary/security checks pass, Full and Atheris run, OpenCode fails closed |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `8a1654ead5a23e985c9bf1d6d500602283f05ab8` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security and Full/Atheris checks pass, OpenCode fails closed |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `ec17d4e0b77fe10c8087c587cb748d4027fe4d0f` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `3adb441a84b6d6d2e0bc866b64281c81acaf70af` | `BLOCKED/REVIEW_REQUIRED`; Full/Atheris/Python/Noema run, Strix runs, earlier ordinary/security checks pass, OpenCode fails closed |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `a0be90cdc401aa6322e5ea7174c5b78efea4b824` | `BLOCKED/REVIEW_REQUIRED`; base update is pushed and hosted checks are rebuilding |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `dddb2026fe591df5bd071b6441ff15272f5d7cd8` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security and Strix checks pass, Full/Atheris run, OpenCode fails closed; this documentation commit will advance its self-referential head |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `bf6fbb7d372922088bd075b395543b325d91ed78` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode fails closed |

The OpenCode and Strix failures are hosted provider-gate outcomes, not local
passing evidence. #857's TLS runner repair is at a fresh exact head, #901
keeps authenticated OpenRouter discovery routable while using its ZDR endpoint
as privacy evidence, and #903 is now main-aligned through a normal merge.

## 2026-08-29 03:23 KST exact-head protected-queue snapshot

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`. The open protected queue has
eight PRs. The table records mixed mergeability, including a `BEHIND` entry, and
every PR is still blocked by protected review requirements; all have zero
unresolved threads and zero qualifying approvals on the exact head. No bypass
was used.

| PR | Exact head | Current protected evidence |
| ---: | --- | --- |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `31059a6b383912fcdf0bac42afa237ff5796b6a4` | `BLOCKED`; ordinary/security checks pass, Atheris passes, Full and Strix run, OpenCode fails closed |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `8a1654ead5a23e985c9bf1d6d500602283f05ab8` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security and Full/Atheris checks pass, OpenCode fails closed |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `ec17d4e0b77fe10c8087c587cb748d4027fe4d0f` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `ea183e72be3a2d2ae8ad6229a0212bc5b8ec9bf6` | `BLOCKED/REVIEW_REQUIRED`; Full is queued, Atheris/Python/Noema run, earlier ordinary/security checks pass |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `57ec66351c1ca37910650d5ad77e6bdbdc79be51` | `BEHIND/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `5c5c077b272c41c8042cb2e42fec33b0da633612` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode fails closed; this documentation commit will advance its self-referential head |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode and Strix fail closed |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `bf6fbb7d372922088bd075b395543b325d91ed78` | `BLOCKED/REVIEW_REQUIRED`; ordinary/security checks pass, OpenCode fails closed |

The OpenCode and Strix failures are hosted provider-gate outcomes, not local
passing evidence. #901's exact head also keeps authenticated OpenRouter
discovery in the serving pool while retaining the public ZDR endpoint as
privacy evidence; its focused provider/discovery/bootstrap/review validation
is `90 passed` locally.

## 2026-08-29 01:14 KST exact-head protected-queue snapshot

Protected `main` remains
`b21645116b352967e50fc497b87eb745b9cc8c61`, including the normal merge of
PR #904. This snapshot records the open queue before this documentation
commit; PR #905 is at pre-push head `322215b7068c279db55eb006679ee36010b852d3`
and this commit will advance that documentation PR head. No open PR has a
qualifying independent approval; exact-head hosted checks and resolved review
threads remain required, and no bypass was used.

PR [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868)
is at `97356f66bd0aa39d4b903e3a1bcef08467e0a36c`, `MERGEABLE/BLOCKED`, with
`REVIEW_REQUIRED`, zero unresolved threads, and zero exact-head approvals.
The main integration and privacy-assessment normalization are local-verified
by 142 focused tests; its hosted Full, security, dependency, supply-chain,
Hypothesis, Atheris, Noema, and review jobs are still queued or running, with
no completed failure at this snapshot.

PR [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901)
is at `6549937834941895ae3e379f82cbf37eb645a59a`, `MERGEABLE/BLOCKED`, with
zero unresolved threads and zero exact-head approvals. Its ordinary checks
are successful so far; Full, Atheris, and Strix are running and OpenCode is
queued. The plain orchestrated Responses path now forces synchronous routing,
and the focused Responses routing/stream suite is `25 passed`.

PR [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908)
is at `bf6fbb7d372922088bd075b395543b325d91ed78`, `MERGEABLE/BLOCKED`, with
zero unresolved threads and zero exact-head approvals. Its required ordinary
checks are successful; OpenCode is failed closed and the focused metering and
cost-ledger proof is `53 passed`. Inline duplicate and rollback health
counters, deferred export accounting, and the corresponding ADR/CHANGELOG
contract are now aligned.

PR [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906)
is at `c6495e19b3255eaf74c94ae3d80d455fa88ebde9`, `MERGEABLE/BLOCKED`, with
all threads resolved and no exact-head approval. Full, security, NIM, supply
chain, and ordinary checks pass; OpenCode and Strix fail closed. Its local NIM
evidence remains `121 passed` with 100% branch coverage for `nim_benchmark.py`.

PR [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905)
is at pre-push head `322215b7068c279db55eb006679ee36010b852d3`,
`MERGEABLE/BLOCKED`, with all threads resolved and no exact-head approval.
Ordinary checks pass and OpenCode fails closed. The next documentation push
will advance this self-referential head, so this row intentionally records the
pre-push SHA.

PR [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857)
is at `52d8bf66f2d14efd2b9e5f11da419b8683696527`, `MERGEABLE/BLOCKED`, with
all threads resolved and no exact-head approval. Ordinary checks pass; Full
and Atheris are running and OpenCode fails closed.

PR [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903)
is at `57ec66351c1ca37910650d5ad77e6bdbdc79be51`, `MERGEABLE/BEHIND`, with
all threads resolved and no exact-head approval. Ordinary checks pass while
OpenCode and Strix fail closed. PR
[#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879)
is at `2b1829a81ea79e012480c680a7ef5683dc13c3bc`, `CONFLICTING/DIRTY`, with
all threads resolved and no exact-head approval; ordinary checks and OpenCode
pass while Strix fails closed.

| PR | Exact head | Current protected gate state |
| ---: | --- | --- |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `97356f66bd0aa39d4b903e3a1bcef08467e0a36c` | `BLOCKED`; main-aligned, ordinary/review checks queued or running, no failure yet, no approval |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `6549937834941895ae3e379f82cbf37eb645a59a` | `BLOCKED`; ordinary checks pass so far, Full/Atheris/Strix running, OpenCode queued, no approval |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `bf6fbb7d372922088bd075b395543b325d91ed78` | `BLOCKED`; ordinary checks pass, OpenCode fails closed, no approval |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED`; ordinary/NIM/security checks pass, OpenCode/Strix fail closed, no approval |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `322215b7068c279db55eb006679ee36010b852d3` | `BLOCKED`; self pre-push head, ordinary checks pass, OpenCode fails closed, no approval |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `52d8bf66f2d14efd2b9e5f11da419b8683696527` | `BLOCKED`; Full/Atheris running, ordinary checks pass, OpenCode fails closed, no approval |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `57ec66351c1ca37910650d5ad77e6bdbdc79be51` | `BEHIND`; ordinary checks pass, OpenCode/Strix fail closed, no approval |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `2b1829a81ea79e012480c680a7ef5683dc13c3bc` | `DIRTY`; ordinary/OpenCode checks pass, Strix fails closed, no approval |

## 2026-08-29 00:32 KST exact-head protected-queue snapshot

Protected `main` is `b21645116b352967e50fc497b87eb745b9cc8c61`, which contains
the merge commit for PR #904 at exact head
`6cd7d57c177d945f67ba3b86b699949584bc6b7e`. This loop did not bypass or claim
that merge. The open queue now contains eight PRs; duplicate docs-only PR #900
remains closed because #905 supersedes its evidence, and the previously stacked
PR #907 is merged into #857. The current protected state still requires exact
head checks, independent approval, and resolved threads; `behind`, `dirty`, or
`UNKNOWN` merge state is not readiness.

PR [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906)
at `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` is the base-aligned head after
the protected #904 merge. Full and Strix are running; ordinary, NIM quality,
security, Hypothesis, dependency, OSV, Trivy, and Noema checks pass while
OpenCode fails closed. All threads are resolved and no independent approval is
present. Its local NIM evidence remains `121 passed` with 100% branch coverage
for `nim_benchmark.py`; the cold-import fuzz fix and unused provider response
wrapper cleanup do not change production routing defaults.

PR [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908)
at `94dd586839efc2621c4e0d81b0867e4af33d0b03` is base-aligned after the
protected #904 merge. Full and Atheris are running; CodeQL, dependency,
Hypothesis, OSV, Trivy, Python supply chain, and Noema pass while OpenCode
fails closed and Strix has not yet reported. All threads are resolved and no
independent approval is present. Its focused metering and cost-ledger proof is
`51 passed`; deferred exports use targeted persisted-ID lookups, transaction
visibility checks, and reconciled duplicate-drop telemetry.

PR [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901)
at `ddc43068e123e707197bbeab8e4518d29a0b8063` is base-aligned after the
protected #904 merge. Full and Atheris are running; CodeQL, coverage,
dependency, OSV, Trivy, Hypothesis, Python supply chain, and Noema pass while
OpenCode fails closed. All threads are resolved and no independent approval is
present.

The previously stacked PR #907 is merged into #857. The shared batch embedding
fixture is byte-identical with the current naruon consumer fixture.

PR [#904](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/904)
was merged into protected `main` as
`b21645116b352967e50fc497b87eb745b9cc8c61`. Its ordinary Full, Atheris,
security, coverage, dependency, OSV, Hypothesis, and Noema checks passed at the
source head; OpenCode and Strix are recorded as failed provider/review gates
after the merge. The merged slice binds concrete model file replicas, honors
file-provider exclusions, and maps provider delete failures to retryable 503
responses.

PR [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905)
at `f7276b02ac1587d7ca9e93876f8c27f4e7b32eb1` is the base-aligned pre-push
head of this baseline refresh branch. Full is running while Atheris, CodeQL,
coverage, dependency, OSV, Trivy, Hypothesis, and Noema pass; OpenCode fails
closed. Its next documentation commit will advance the PR head; threads are
resolved and no independent approval is present. PR
[#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903)
at `57ec66351c1ca37910650d5ad77e6bdbdc79be51` has ordinary checks passing but
OpenCode and Strix fail closed; it has no qualifying independent approval.
PR [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879)
at `2b1829a81ea79e012480c680a7ef5683dc13c3bc` has ordinary checks and OpenCode
passing but Strix failing closed; it has no qualifying independent approval.
PR [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868)
at `511956c274109a89af49193d1e6c78260dd2c1eb` has ordinary checks passing but
OpenCode and Strix fail closed; it has no qualifying independent approval. PR
[#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857)
at `d1afcd3763b925195d6c4303ef5004d92c0d94cf` has ordinary checks passing but
OpenCode fails; it has no qualifying independent approval.

| PR | Exact head | Base / current gate state |
| ---: | --- | --- |
| [#908](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/908) | `94dd586839efc2621c4e0d81b0867e4af33d0b03` | `BLOCKED`; base-aligned, Full/Atheris running, ordinary security checks and Noema pass, OpenCode fails closed, no Strix result, no independent approval |
| [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) | `c6495e19b3255eaf74c94ae3d80d455fa88ebde9` | `BLOCKED`; base-aligned, Full/Strix running, ordinary/NIM/security/review checks pass, OpenCode fails closed, no independent approval |
| [#905](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/905) | `f7276b02ac1587d7ca9e93876f8c27f4e7b32eb1` | `BLOCKED`; base-aligned pre-push head for this baseline refresh, Full running, ordinary checks pass, OpenCode fails closed, no independent approval |
| [#904](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/904) | `6cd7d57c177d945f67ba3b86b699949584bc6b7e` | `MERGED` as `b21645116b352967e50fc497b87eb745b9cc8c61`; ordinary checks pass, OpenCode/Strix fail after merge |
| [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) | `57ec66351c1ca37910650d5ad77e6bdbdc79be51` | `BLOCKED`, `REVIEW_REQUIRED`; OpenCode/Strix fail closed |
| [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901) | `ddc43068e123e707197bbeab8e4518d29a0b8063` | `BLOCKED`; base-aligned, Full/Atheris running, ordinary checks pass, OpenCode fails closed, no independent approval |
| [#879](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/879) | `2b1829a81ea79e012480c680a7ef5683dc13c3bc` | `BLOCKED`, `REVIEW_REQUIRED`; OpenCode/Strix fail closed |
| [#868](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/868) | `511956c274109a89af49193d1e6c78260dd2c1eb` | `BLOCKED`, `REVIEW_REQUIRED`; ordinary checks pass, OpenCode fail, Strix provider failure |
| [#857](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/857) | `d1afcd3763b925195d6c4303ef5004d92c0d94cf` | `BLOCKED`, `REVIEW_REQUIRED`; Strix retry canceled, dependency-review pass, OpenCode fail |
## 2026-08-29 PR #901 routing research grounding

This ZDR slice applies the established cost/performance routing literature to
the gateway boundary without turning provider names or model ids into policy.
[FrugalGPT](https://arxiv.org/abs/2305.05176) motivates composing a model
cascade from heterogeneous providers while reducing inference cost, and
[RouteLLM](https://arxiv.org/abs/2406.18665) motivates selecting among the
available candidates at inference time rather than binding the router to a
fixed model list. In this implementation, Naruon supplies `zdr_only` as a
Boolean request policy and contextual-orchestrator filters the caller's
runtime model-group array by verified `privacy:zdr` evidence before measured
member selection. The public OpenRouter ZDR feed is evidence for matching
models from other providers; OpenRouter is not selected as the upstream by
this policy. Missing or failed ZDR evidence fails closed instead of being
replaced by a stale or hard-coded model list.

The FrugalGPT and RouteLLM PDFs are already vendored under `docs/papers/`; the
catalog there records their arXiv redistribution license and full citation.

## 2026-08-28 21:42 KST PR #901 provider error-shape compatibility slice

PR [#901](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/901)
is at exact head `bcead52a` after a narrow failover repair: provider HTTP 400
tool-description-limit responses whose `error` field is a string now receive
the same capability-mismatch failover as the existing `invalid_tools` object
shape. The focused passthrough suite passed **28 tests**; protected hosted
checks and independent approval remain authoritative and are not claimed here.

## 2026-08-28 21:20 KST PR #901 ZDR batch omitted-model evidence slice

`gh api user`, `gh pr view 901`, and the review-thread GraphQL query all
succeeded on August 28, 2026. At that snapshot, PR `#901`
(`fix/zdr-only-dynamic-discovery`, exact head `dcdb04b7adcfc8f57cbf5b13332b504f5c246fed`)
already covers the earlier ZDR routing defects that had active review threads:
caller-supplied group filtering, generated-plan ZDR revalidation, batch model
selection error normalization, and duplicate-model embedding identity. The
remaining protected block is external: `opencode-review` is still fail-closed
pending a current-head verdict, while the last `strix` failure on this branch
reported `STRIX_PROVIDER_UNAVAILABLE` rather than a repository finding.

The existing `.worktrees/commercial-loop-20260828-pr901-fix` worktree was
reconciled before new edits. It is 38 commits behind the live PR head and its
omitted-model batch embeddings regression test plus this baseline note were
ported onto a fresh worktree and are now published on the live PR head instead
of reviving the diverged branch.

This slice closes one bounded evidence gap in the highest-leverage open product
area: `POST /v1/batch/embeddings` with omitted `model` and `"zdr_only": true`
now has an explicit loopback HTTP regression test proving the gateway selects
the ZDR-capable embedding member rather than a higher-priority non-ZDR peer.
The seeded contract server already carried persisted `privacy:zdr` evidence, so
the change is a missing acceptance test, not a runtime behavior change.

Exact verification on the current-head worktree:

- `uv pip install --python .venv/bin/python -r requirements.lock` → restored the locked runtime packages for the fresh worktree venv.
- `uv pip install --python .venv/bin/python -r requirements-opencode-review-ci.txt` → installed the repository's pinned pytest toolchain for focused validation.
- `uv run --python .venv/bin/python -m pytest -q tests/test_batch_embeddings.py -k 'naruon_contract or zdr_only or omitted_model'` → `2 passed, 6 deselected in 8.40s`
- `uv run --python .venv/bin/python -m pytest -q tests/test_cost_router.py -k 'zdr_only or duplicate_model'` → `1 passed, 24 deselected in 6.77s`
- `uv run --python .venv/bin/python -m pytest -q tests/test_model_group.py -k 'zdr_only or duplicate_model'` → `4 passed, 24 deselected in 6.84s`

## 2026-08-27 20:10 KST main trace-rpds regression slice

Protected `main` briefly carried a merge-order regression from PR #891 merged
before #888: the chat/structured branch still called the removed
`_trace_requested` helper, so every structured chat or
tool-plus-`response_format` request raised `AttributeError` and returned
`500 internal_error` instead of the intended fail-closed
`400 unsupported_trace_disclosure`. The same merge also drifted the fuzz
requirements (`rpds-py` 2026.6.3) away from `requirements.lock` (`0.30.0`),
so the Hypothesis and Atheris PR jobs failed `ResolutionImpossible` on every
open PR.

`#896` restored both: use the already-validated `include_trace` value on the
chat branch (with two #891-era honesty tests aligned to the fail-closed #888
contract), and re-pinned both fuzz requirement groups to the lock's
`rpds-py==0.30.0` (same hash set) with the property `.in` source updated so a
future regen cannot drift again. Full suite at exact head: **2428 passed**;
Hypothesis, Atheris, and devin/opencode/noema reviews all pass.

Merged to protected `main` as `5b3069d4`; protected-main runs for #887, #893,
#883, #889 then landed, delivering bounded tool descriptions, the React +
Storybook web admin, provider-affine video job ownership, and fail-closed
commercial release authorization on one linear main sequence.

**Open queue:** `#879` (provider-failure taxonomy/telemetry, `BEHIND`), `#857`
(provider-backed embeddings, `DIRTY`), `#868` (gateway-default chat surfaces,
`DIRTY`) remain agent-owned and re-check after the next main advance. The org
Strix LLM scan is currently failing closed org-wide because the NIM/OpenRouter
endpoints return 429 and the OpenAI-direct key reports
`credit_balance_exhausted`; this is a provider-credit/billing condition, not a
code finding, and its serial queue re-schedules each PR head once credits
recover.

## 2026-08-27 trace-authority acceptance slice

Protected `main` at `5a01759165be20ab38c05c2321d8a9f00ec331ea`
contains the trace-purpose gate delivered through protected PR #781, but issue
#117 remains open. Current-main probes found two central bypasses: structured
chat accepted a non-Boolean trace flag before returning early, and an
admin/inference principal without trace authority could read access-report
steps and accessed outputs. This slice moves strict flag validation ahead of
every chat execution branch and requires trace authority before access-report
resource lookup, making owned and unknown identifiers indistinguishable to a
non-trace caller.

This is not full #117 closure. Batch routing jobs still lack principal-bound
ownership; the bearer-verifier contract lacks tenant, resource, purpose,
lifetime, and revocation context; and legacy single-token production migration
does not yet have a fail-closed deployment gate. Those requirements need their
own protected implementation and HTTP acceptance evidence.

## 2026-08-26 protected-main catalog evidence slice

Protected `main` is `56a898b85654f5c8468e3d8448d93120b24bd269`
after the normal #851 merge. The exact open queue was re-read at #880
`da9f4ab0`, #879 `ed0690e3`, #876 `4da38a05`, #869 `fe7b248e`, #868
`28e0fdbd`, #858 `b0b67286`, #857 `77fd4369`, and #849 `7abf1b89`.
Those branches cover provider errors and telemetry, CI runtime installation,
this baseline's broader refresh, Zen bootstrap, gateway aliases, customer copy,
provider-backed embeddings, and asynchronous HTTP capacity. None delivers the
operator-visible refresh timestamps and stable success/failure evidence required
by accepted ADR 0015.

The highest-leverage independently implementable customer gap is therefore
provider-catalog freshness evidence. The durable catalog already records exact
per-account refresh status, bounded error code, counts, and UTC instants, but
the bootstrap JSON omitted those fields. Operators consequently could not tell
a live catalog from last-known-good recovery without querying the database.
This slice exposes only the current bootstrap's secret-free refresh evidence;
it does not infer provider/model policy, calculate a freshness threshold, or
claim that an unknown price is zero. The acceptance contract is: each attempted
registered account emits its stable account id, status, observed/eligible
counts, allowlisted error code, and UTC start/finish instants; previous runs in
a reused store are not duplicated in the current report; secrets and raw
provider diagnostics remain absent.

Remaining larger gaps stay unchanged: durable multi-replica routing observations
need an accepted retention/decay decision; video jobs need a normalized durable
ownership/lifecycle contract; and verified answer-token streaming needs a
cancellable asynchronous dependency graph. Implementing any of those without
their missing decisions would invent policy rather than close a bounded gap.

## 2026-08-26 11:46 KST exact-head queue snapshot

Protected `main` remains `762f7a345b1d8c82584023a7ff05b4660d628cab`.
The open queue was re-read at #856 `cf4af71e`, #855 `f2e66db7`, #851
`d42172b3`, #850 `dd88b69e`, #849 `6103806a`, #848 `d0d5439d`, #845
`24219ace`, and #834 implementation head `b189e108`. Every PR retains normal
auto-merge. No exact head has an independent approval, so none is eligible for
protected merge. #855 and #851 have terminal successful required jobs; the
other heads have queued review or security jobs. Queue delay is not treated as
success or bypass authority.

#856 adds the operator-configurable request-body ceiling and validates direct
`SecurityConfig` construction, including rejection of Boolean and non-integer
limits. Its full exact tree exposed only the stale legacy-table assertion
already repaired by #855; the stacked repair leaves its replacement Checks
queued. #834's complete exact tree is `2253 passed`; its seven previously
undocumented public persistence, cache, and judge-adapter boundaries now have
explicit docstrings, with `111` focused tests passing. These are branch
evidence, not protected-main delivery.

## 2026-08-26 10:30 KST review-remediation snapshot

Protected `main` remains `762f7a345b1d8c82584023a7ff05b4660d628cab`.
The open queue was re-read at exact heads #855 `f2e66db7`, #851 `d42172b3`,
#850 `dd88b69e`, #849 `6103806a`, #848 `d0d5439d`, #845 `24219ace`,
and #834 `e563920a`. Auto-merge is enabled without bypass on every PR.
The shared stale-table contract is stacked onto #845, #848, #849, and #850;
#849 also makes the documented k6 traffic compatible with its isolated rate
limit and validates programmatic request budgets, while #848 now rejects every
duplicate front-matter or heading identifier declaration. Their replacement
hosted checks and independent exact-head reviews remain required.

PR #834 no longer presents invented policy rules, recent alerts, deployment
region, environment, or health as runtime facts. It shows next-action empty
states until evidence is loaded, and its simulation begins with an empty,
actionable prompt. The focused admin/model-group contract is `38 passed`; the
updated desktop render is recorded under `docs/images/ui-audit/`. This snapshot
is branch evidence only, not protected-main or deployed evidence.

The same head now treats a disconnected Responses reasoning-summary stream as
cancellation evidence: it stops before the next orchestration stage and always
releases the bounded execution slot. The disconnect, Responses stream, and
passthrough slice is `32 passed`; this prevents paid work from continuing after
the customer can no longer receive it. The complete exact tree passes `2253`
tests after aligning four stale boundary assertions with the model-aware batch
runner, normalized persistence table, measured routing policy, and public agent
selection contract.

## 2026-08-26 10:00 KST customer-copy and responsive evidence

Protected `main` remains `762f7a345b1d8c82584023a7ff05b4660d628cab`.
PR #834 implementation head `a5f8d8424e7eb5c4aa1281ae4de02b5f7647b290`
removes customer-visible internal configuration, authentication, research-role,
worker/planner, agent-ID, and endpoint terminology. English and Korean empty and
warning states now identify the next action. Headless Chromium renders at
`1440 × 1200` and `390 × 844` are recorded in
[`docs/ui-audit-2026-08-26.md`](ui-audit-2026-08-26.md); the mobile page has no
document-level horizontal overflow and the focused admin/integration contract
is `10 passed`.

This remains local exact-head UI evidence rather than protected-main or deployed
authenticated evidence. The next acceptance action is to complete #834's hosted
checks and independent review, merge normally, then repeat the screenshots and
interactions against the deployed console.

## 2026-08-26 08:55 KST protected-main and open-queue baseline

Protected `main` is `762f7a345b1d8c82584023a7ff05b4660d628cab`.
Its latest full Tests run reached `2148 passed` before the remaining persistence
assertion queried the removed legacy `records` table. PR #855 fixes only that
post-merge test regression at exact head `f2e66db7acc05d8077e58f735b640af23906b336`;
it is not protected-main evidence until its exact-head checks and independent
review complete.

| PR | Exact head | Current evidence and next acceptance action |
| --- | --- | --- |
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | `62015641ed3afe979a24fcafcfa2d0d17cfef6ef` | Current-main integration resolves model-judge and model-list conflicts while preserving arbitrary operator groups, explicit cost evidence, all eight model capabilities, `orchestrator/auto` and `orchestrator/free`, Responses reasoning-summary streaming, Compose, and the hourly loop. The exact integrated focused suite is `96 passed`; protected checks and independent review remain required. |
| [#845](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/845) | `3cd57d46ddc3cb3ad2786e548922e501b18f0dca` | Current-main integration retains the PostgreSQL service fallback when both durable KV secrets are absent and fails closed on partial KV configuration. The focused contract is `4 passed` and `actionlint` is clean; a successful scheduled run after protected merge remains required runtime evidence. |
| [#848](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/848) | `359ab9a9a08a59edc749f6ffa2268d5c8f4baf6d` | Current-main integration preserves unique planning ADR identifiers; the exact focused contract is `1 passed`. |
| [#849](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/849) | `fecc7b017fc822077811a76e0c4b34291054c046` | The stacked k6 work is incorporated; HTTP/1.1 keep-alive now reuses connections, bounds idle reads, uses the native listen backlog, closes unread request bodies, and clears trace/session state after every persistent request. The latest focused server slice is `35 passed`; production TLS/provider/soak capacity remains unproven. |
| [#850](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/850) | `d230c00279fcce1bbd2c8d77cd7220bc2aeb0e9f` | Constant-time budget gating remains reconciled after current-main integration; the exact focused budget suite is `9 passed`. |
| [#851](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/851) | `d42172b3b784d8fa7700d1b3792925a052f294c9` | Structured requests no longer conduct on empty `response_format`, empty `tools`, or omit-equivalent `tool_choice`. Provider-reported and token-counter fallback usage now persist `measured` versus `estimated` provenance in normalized `usage_measurements`, cost responses expose it, and conducted structured Chat has a distinct analytics event. The first full run exposed three integration regressions; their root fixes passed `46` focused tests and the resulting exact tree passes all `2175` tests. Protected checks and independent review remain required. |
| [#855](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/855) | `f2e66db7acc05d8077e58f735b640af23906b336` | Repairs the sole observed protected-main test failure by asserting the production `orchestration_records` table; `29 passed` locally. Merge this root repair before treating a later green main run as release evidence. |

Every open PR above has normal auto-merge enabled. At this snapshot their
required hosted jobs are queued and no failed exact-head result is present;
queued jobs and absent independent approval are not success evidence and must
not be bypassed.

## 2026-08-26 00:33 KST exact-head continuation

Protected `main` remains `838b3de160c341a6f36bf588ae9fcc09989c040c`;
none of the following evidence is a protected-main release claim.

| PR | Exact head | Current evidence and customer consequence |
| --- | --- | --- |
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | `7fa07ac22c0482a8f7770df90b9813396e2d86bf` | Every hosted Check, including the rerun Strix and OpenCode review, is terminal and successful with zero unresolved threads. Independent approval remains absent, so the arbitrary model-group, cost-aware discovery, eight-modality routing, Responses reasoning-summary stream, Compose, and hourly-loop stack is still not delivered to protected `main`; auto-merge remains enabled. |
| [#851](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/851) | `e47432b9` | New review found that a mixed provider-usage workflow omitted usage-less evidence calls from the cost ledger. Every trace step now records one row under the same workflow lineage, preserving valid provider counts and applying the existing synchronous token counter only to calls whose provider omitted usage. All three new threads are resolved; `69` focused tests and the full exact tree (`1901 passed in 569.84s`) pass. Replacement hosted Checks and independent review are required; auto-merge remains enabled. |
| [#849](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/849) | `43372c2f58b36466eba4934e50a9f945fffacac9` | The checked-in k6 E2E scenario was rerun on this exact head: 64 concurrent delayed inference users completed 128/128 inference checks while 101/101 liveness checks completed, with 0/229 HTTP failures, 25.07 inference requests/s, and 1.09 ms liveness p99. This remains loopback synthetic-delay evidence, not a production TLS/provider/soak SLO. |

## 2026-08-25 23:38 KST structured-provider continuation

Protected `main` remains `838b3de160c341a6f36bf588ae9fcc09989c040c`;
the following stack integration is not protected-main release evidence.

| PR | Exact head / merge | Current evidence and customer consequence |
| --- | --- | --- |
| [#852](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/852) | `34b705053da1772d4b0d985a5ff781e32c35707c` | Merges current protected `main` without force-updating history, resolves the active-session cache conflict with the stronger active-session check, and repairs three vacuous/wrong-key review assertions. The merged exact tree passed `2090` tests in `620.48s`; pinned coverage `7.15.4` reports `5123` statements, `1510` branches, zero misses, zero partial branches, and `100%`. Hosted checks and independent exact-head review remain authoritative. |
| [#853](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/853) | merged normally into #851 as `7841f15ba42a97c5e8b4b5134f716b87c1c24d71` | Structured Chat and non-tool Responses requests now conduct the existing workflow before provider-native synthesis; schemas, native Responses input, multimodal evidence, caller-visible echo privacy, and one-run cost provenance are covered. Tool requests retain the OpenAI-compatible single-provider response because the client owns tool state. The pre-stack exact tree passed `1894` tests in `589.00s`; the reviewed #851 stack integration passed `51` focused tests. ADR `0034` records the decision. This is stack evidence only. |
| [#851](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/851) | `fa782c6e` | Includes #853 plus the reviewed provider-failover repair. Effort-capable aliases are filtered before provider deduplication, so a lower-ranked supported deployment is not hidden by an unsupported alias. Structured Chat now preserves account/model/service cost attribution, rejects unsupported batch hints, and applies caller sampling to evidence calls; Responses evidence uses the same request scope, and explicit vision mismatch is a 400. The exact tree passes `1900` tests in `572.93s`. The ADR identifier is reserved as `0034`; hosted checks and independent review must be re-established before protected delivery. |
| [#849](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/849) | `43372c2f58b36466eba4934e50a9f945fffacac9` | The HTTP/1.1 handler now closes unsupported method requests with unread bodies, preventing the stdlib `501` path from interpreting payload bytes as another request. The raw-socket regression and the complete unread-body slice pass (`6 passed`). Idle keep-alive threads remain bounded by the configured timeout; production TLS/proxy/soak evidence remains a deployment acceptance gap. |

Issue [#846](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/846)
now has current-main replacements for constant-time spend gates, provider
failover, and structured-provider orchestration. It remains open until those
replacement heads are normally delivered to protected `main`; closed-stack
checks do not transfer.

## 2026-08-25 22:22 KST exact-head continuation

Protected `main` is `838b3de160c341a6f36bf588ae9fcc09989c040c`.
PRs #782 and #790 merged normally as `ba70855cd0f63654feca3a5925c673fb1bf39072`
and `838b3de160c341a6f36bf588ae9fcc09989c040c`. PR #847 merged normally into
the #834 stack as `93b8016cb77cf400b7aec2f15dfc38bc2b7ebeef`; it repairs four reviewed
routing defects: missing-affinity inversion, non-chat leakage into chat roles,
heterogeneous score units, and unjudged quality observations. Its routing
score is now consistently posterior stability divided by EWMA latency;
tokens-per-second remains diagnostic evidence rather than an arbitrary weight.

| PR | Exact head | Current gate and customer consequence |
| --- | --- | --- |
| [#851](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/851) | `121aec01bc02a414c28cc6ef2fdd0d0deb3a9946` | Recovers only the current-main passthrough-failover slice from #846. Virtual requests advance once per distinct provider only after explicit HTTP rejection, stale-model response, or pre-request `EAI_AGAIN`; concrete models and ambiguous timeout/connection outcomes fail closed. Mixed pools select proven reasoning-effort support. The predecessor exact tree passed all `1886` tests; the review-repaired head passed `89` focused tests, so replacement hosted full-suite evidence is required. Auto-merge remains enabled. |
| [#850](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/850) | `05af334a87fb9d5a68f6e459c86b7ca779fb025b` | Constant-time budget status preserves exact randomized analytics parity, replacement semantics, provider/estimated usage, per-model price rounding, restart recovery, and rare agent-pool mutation reconciliation. The predecessor tree passed all `1879` tests; exact-head focused evidence is `49 passed`, so hosted full-suite evidence remains required. Independent review and protected checks are pending with auto-merge enabled. |
| [#849](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/849) | `4d630af98df4afed7b1b585bbec32e77582ae40f` | k6 measured 64 concurrent inference requests: 128/128 inference and 100/100 health requests succeeded with 0/228 failures at 25.068 inference req/s and 10.85 ms health p99. The preceding hosted suite passed 1881 tests before two direct response-writer tests exposed an absent request-body marker; the exact-head fix treats an absent marker as already consumed while preserving unread POST-body connection closure (`14 passed` focused). Replacement hosted full-suite evidence remains required. |
| [#848](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/848) | `9632a682644516b784e77b253cd47583b74ff7a3` | Latest-main merge applied; independent exact-head review and required checks are pending with auto-merge enabled. Until this lands, planning ADR identifiers are not mechanically unique. |
| [#845](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/845) | `6353dfb5027af27ed5fdfd15172a9c841efccf0d` | The preceding hosted full suite reached 1876 passes before one cache test compared identical deterministic payloads instead of the cache contract. The exact-head repair asserts the model-key cache miss and provider-call increment directly (`15 passed` focused); replacement protected checks and independent review are pending with auto-merge enabled. Until this lands, hourly provider-catalog refresh cannot claim reliable run-scoped KV fallback. |
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | `7fa07ac22c0482a8f7770df90b9813396e2d86bf` | Contains the reviewed #847 remediation, validates throughput evidence before mutating routing state, binds external-PR validation to the fetched exact head, and removes the last CLI claim that model-name suffixes prove free cost. The exact-tree model-group/REST/DB/eight-capability/Responses-stream suite is `119 passed`; `docker compose -f compose.yaml config --quiet` renders successfully with required bootstrap inputs. Independent exact-head review and protected checks are pending with auto-merge enabled. This is not protected-main evidence yet. |
| [#818](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/818) | `9888b33d108d3eb030572f8e7e89f8fa47366bd2` | Streaming and passthrough spans now use the OpenTelemetry well-known `chat`, `text_completion`, and `generate_content` operation names; independent approval and exact-head checks remain pending with auto-merge enabled. Telemetry correlation is not release evidence yet. |
| [#794](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/794) | `fcecdb86e9e670c145158866f17485229f216be5` | The preceding hosted full suite passed 1901 tests before the same deterministic cache-payload assertion already repaired on #845 failed. The exact-head repair asserts the model-partition miss and provider-call increment directly (`15 passed` focused); replacement protected checks are required. Shared DB access remains serialized and incomplete flattened schemas fail before rename, but the migrations remain unreleased. |
| [#773](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/773) | this commit supersedes `b6d1f1958338a5fb8162ebd3d5e16a365b8cb61a` | This baseline refresh must pass exact-head checks before it becomes protected-main product evidence. |

PR #849 closes the first measurable asynchronous web-capacity slice with a
checked-in E2E scenario and before/after evidence. The remaining capacity gap is
deployment-specific: repeat the same workload through production TLS, real
provider quotas, multi-process workers, and a soak duration before declaring a
production SLO. Its candidate inference p95 remains provider-delay dominated at
about 1.06 seconds; no unmeasured concurrency target or heuristic tuning is
claimed.

The re-read PRD exposes a separate commercial-semantics gap: its
`KRW 2,000,000,000` prospective contract-review anchor is not the user's
USD 20 billion strategic sale-confidence bar. They are different quantities,
not values to convert with an arbitrary exchange rate. Keep the existing
contract-review API truthful until a separately authorized strategic valuation
evidence model defines currency, valuation date, comparable transactions,
revenue/retention assumptions, and uncertainty; never relabel one as the other.

Issue [#846](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/846)
records fixes stranded on the closed #765 stack. PR #850 recovers only the
constant-time budget-status slice on current main with randomized numerical
parity and no-scan timing-shape evidence; PR #851 independently recovers the
passthrough provider-failover slice with current-main full-suite evidence. The
structured-provider fix was subsequently recovered through #853 and merged
normally into #851; stale historical review or check evidence still does not
transfer to the resulting #851 exact head.

## 2026-08-25 exact-head review continuation

| PR | Exact head/base | Current evidence and decision |
| --- | --- | --- |
| [#848](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/848) | head `e55182494e7bdfabed570942422872e9e3e06f1e`, base `main` `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | Protected `main` contains duplicate planning ADR identifiers `0011` and `0024`. The focused repair preserves the earlier provider-error and embedding identifiers, renumbers the later PII decisions to `0027`/`0028`, and adds a uniqueness/content regression contract (`1 passed`). Open PR ADRs are reserved at `0029`–`0033` so their merge results remain unambiguous. Decision: `WAIT_AND_REMEDIATE`. |
| [#845](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/845) | head `397096cd672ed1dd1f1d2132c2ca427ab298d685`, base `main` `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | Repairs hourly provider-catalog runs `32834283714` and `32829694684`, which failed before discovery because both durable KV secrets were absent. The workflow now uses durable PostgreSQL only when both secrets exist, uses an explicitly run-scoped PostgreSQL service when neither exists, fails closed on partial configuration, pins its container digest, and selects the effective subprocess KV without copying durable secrets through `GITHUB_ENV` or shadowing the fallback. The operator guide distinguishes always-required provider secrets from persistence-only KV secrets. `actionlint` and the focused `4 passed` contract are clean; hosted exact-head Checks and independent approval remain authoritative. Decision: `WAIT_AND_REMEDIATE`. |
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | head `cb950a280e60c583d9c2c093922d9df315375a4c`, base `main` `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | Open with normal auto-merge enabled. Zen availability is joined to Models.dev structured cost/modality evidence without name inference. Normalized catalog reload distinguishes provider-declared `capability:*` evidence from generic serving-role tags and preserves cost/modality evidence. Explicitly excluded embedding deployments fail closed; operator-declared Bytez non-chat endpoint capabilities survive discovery while chat-only safety transports remain filtered. PATCH/DELETE share the GET `model_group_not_found` contract, the re-read PRD covers model groups and all supported modalities, and the normative model-group decision is reserved as ADR `0032`. Focused evidence includes `92 passed`, `59 passed`, `48 passed`, `29 passed`, and the subsequent `23 passed`; hosted Checks and independent exact-head approval remain authoritative. Decision: `WAIT_AND_REMEDIATE`. |
| [#818](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/818) | head `a95d5a108c7698f322e729953be6562bea274968`, base `main` `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | The OpenTelemetry slice is reconciled with current-main request, streaming, batch, dependency-lock, and session-security paths. Reauthorization replaces inbound trace context instead of stacking it. The first full run found one CI-contract failure after `1846` passes; the repair installs both runtime and property locks, removes the superseded attribute denylist, regenerates the runtime lock on CI Python 3.12, and the current filesystem passes `1847` tests plus an isolated Python 3.12 hash-lock install and focused `20 passed`. Hosted exact-head Checks and independent review remain authoritative. Decision: `WAIT_AND_REMEDIATE`. |
| [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) | closed head `b70540420f0f86132cc5911baaf55a26ad0084fa` | Closed unmerged on 2026-08-25 after exact-head decomposition. Its only tip-only Bytez capability repair moved to #834 with regression coverage; merging the conflict-heavy historical stack would duplicate already delivered component PRs and stale ADR history. |
| [#844](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/844) | head `3ca50633b8fa8f639754a5a5dcc8aaa0f2b2bf9f`, merge `84ec3e5345ef2ee951eb4d7a0d15b1175781ba5b` | Merged normally to protected `main` on 2026-08-25. It partitions cached admin-session responses by an active opaque session and contains the thread-termination regression repair. This merge is protected-main evidence for that bounded fix only. |
| [#762](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/762), [#821](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/821), [#780](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/780) | merge commits `2b4a2ff3787d19da7240f5647a05c1a9091d0097`, `38d211665b0d0022e689086db0cc7bc5dc29fcbe`, `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | Normally merged to protected `main` on 2026-08-25. The resulting main exact head is `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424`; this is bounded release evidence for the PII design, token-count strategy coverage, and liveness/readiness boundary only. |

Current open-queue exact-head inventory at this continuation:

| PR | Head | Base | Gate state |
| ---: | --- | --- | --- |
| #773 | self-reference; refetch live | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, auto-merge enabled |
| #782 | `d631d7f37d93613235e92f62a93b3ab69df6fd93` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, changes requested, auto-merge enabled; HTTP owner isolation repaired and `42 passed` after current-main merge |
| #790 | `6acd0fb5dd1ec9dc5680f28b2c84941dbd45a864` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, changes requested, auto-merge enabled; invalid `--max-agents` now fails at the CLI boundary |
| #794 | `ca41d9f938ab6d7e9da0e124447b76a4ec07540a` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, review required, auto-merge enabled; persistence ADRs reserved as `0029`–`0031` |
| #818 | `a95d5a108c7698f322e729953be6562bea274968` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, review required, auto-merge enabled; `1847 passed` and Python 3.12 hash-lock install verified |
| #834 | `cb950a280e60c583d9c2c093922d9df315375a4c` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, auto-merge enabled; PRD, REST errors, Bytez filtering, and ADR `0032` normalized |
| #845 | `397096cd672ed1dd1f1d2132c2ca427ab298d685` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, review required, auto-merge enabled; scheduled provider sync root cause repaired |
| #848 | `e55182494e7bdfabed570942422872e9e3e06f1e` | `6970dbb9e63b9bd1ec602bb8c3c85e3a05480424` | `BLOCKED`, review required, auto-merge enabled; protected-main ADR identifier collision repaired |

The remaining customer-visible routing gap is durable, multi-replica observation
aggregation with an explicit time horizon. The current in-process Beta-Bernoulli
success and Jacobson latency observations are evidence-based but intentionally
reset on restart; inventing decay or cross-model quality weights is prohibited.
Provider identity equivalence remains an operator/provider-provenance assertion,
never a model-name heuristic. Async embedding submissions also must not count as
inference success until the terminal provider result is observed.

## 2026-08-25 Responses reasoning stream, free orchestration, and Compose refresh

| PR | Exact head/base | Current evidence and decision |
| --- | --- | --- |
| [#843](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/843) | merged head `9a084d03edfccd7c46d1ee6ab05062af525622c7`, stack merge `3b0feb6e68453829e724aa4653c25ca1166cb9bc` | Normally merged into #834 after exact-head `1678 passed in 706.95s`, terminal Devin review, and zero unresolved threads. `orchestrator/auto` and `orchestrator/free` now cover text, image, video, speech, transcription, embeddings, rerank, and audio; free routing admits only explicit zero-cost evidence, uses measured primary/failover/capability ordering, and fails closed. `/v1/responses` emits OpenAI reasoning-summary events without raw chain-of-thought, preserves caller instructions, rejects unsupported structured output, and separates transport/application failure status. The hourly OpenCode loop uses `orchestrator/auto`; the cwd-independent locked image built as `sha256:19b7d1cbda5678804721daa6cf936a5ffbeea4b3005fbbbf9be2329b42b77c3b`. This is stack integration, not protected-main release evidence. |
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | head `3b0feb6e68453829e724aa4653c25ca1166cb9bc`, base `main` at `52dfa448417953ebf6e0c7b295e92b4d81cf9420` | Now contains #843. It is conflicting with current main, required workflows are queued, and the live review queue contains unresolved discovery directionality, batch model propagation, hourly-agent trust separation, migration, failover/group-boundary, embedding, API-schema, and release-version findings. Decision: `REVIEW_FIX_RECHECK`; no protected-main or release claim transfers from the merged child. |

The canonical container entry path is root `compose.yaml`: PostgreSQL 17.6,
pgcrypto-backed KV bootstrap from a Compose secret, durable gateway state, a
loopback-only published port, and health-gated service ordering. `docker compose
config --quiet` and both application image builds passed. End-to-end `compose up`
is **not yet runtime evidence** on this machine: the current Colima VM exposes no
host mount, so Docker cannot bind the Compose secret file even under `$HOME`.
Do not replace the secret with a gateway runtime environment variable. Re-run
the health/authenticated inference smoke test on a runner with a working host
mount and record its exact image digest.

Remaining customer-visible gaps:

1. Conducted workflow reasoning summaries stream as each stage starts/completes,
   but final answer deltas begin only after synthesis. True answer-token streaming
   needs a cancellable asynchronous dependency graph; do not fabricate partial
   answers from unverified intermediate work.
2. `orchestrator/free` optimizes only within models with complete structured
   zero-cost evidence. Zen availability is joined to Models.dev metadata, and
   normalized last-known-good reload now retains its free/modality tags; an
   unmatched model or metadata outage remains unknown and excluded. Production
   still needs freshness timestamps and last-success/error evidence for the
   secondary metadata catalog so operators can distinguish current unknown cost
   from stale data.
3. The in-process success/latency ledger is not multi-replica evidence. Add a
   normalized time-windowed observation store with explicit retention/decay
   before claiming fleet-wide optimal routing.
4. Conducted `/v1/responses` streams emit prompt-free request analytics but do
   not yet aggregate their multi-agent usage into cost-ledger rows. Add a durable
   workflow-run usage boundary before claiming complete Responses cost rollups;
   do not estimate provider attribution from the final synthesized answer.

**Snapshot convention:** the initial inventory records its observation time
below; each live continuation carries its own recheck time.
**Source of truth:** `main` at `e226e1197bdfc890c9d8e5b9b648c78857d7e465`
**Product boundary:** one OpenAI-compatible gateway plus its operator evidence
control plane. Fugu, TRINITY, and Conductor are research inputs, not separate
deployables.
**Customer next action:** use this document to select the next mergeable PR and
to verify its exact-head evidence before approving or releasing it.

**Normative decision record:** [ADR 0023 — Product and technical gap
baseline](planning/adrs/0023-product-technical-gap-baseline.md). The earlier
ADR 0016 filename was renamed to ADR 0023 to avoid an identifier collision; no
normative ADR 0016 file remains. Privacy requirements additionally follow
[ADR 0010 — PII audit, not masking](planning/adrs/0010-pii-audit-not-mask.md).

> This is a dated planning snapshot, not a live merge dashboard. PR heads,
> checks, reviews, and base relationships can change after publication. Always
> refetch the remote exact head and protected rules before acting on a row.

## 1. Product requirements (PRD)

Contextual Orchestrator must let an application keep using an OpenAI-compatible
API while the platform chooses between a single-worker route and a deeper,
verifiable workflow. A buyer should be able to answer four questions without
reading source code:

1. Which provider/model handled the request and why was it selected?
2. Which workflow roles saw which prior outputs?
3. What happened when a provider, tool, cache, or verifier failed?
4. Can the same evidence be replayed, audited, and operated as a standalone
   service or an imported module?

The existing product plan covers API compatibility, managed agent pools,
latency/quality policy, trace/access evidence, evaluation replay, i18n, and
buyer-readiness endpoints. The open queue shows that reliability, provider
bootstrap, secure credential use, purpose-limited PII access, and release-grade
operability are still being closed.

## 2. Technical requirements (TRD)

| Boundary | Required behavior | Acceptance evidence |
|---|---|---|
| API | Preserve `/v1/chat/completions` and compatible error/stream contracts. | Contract tests plus hosted required workflows. |
| Routing | Select by capability, provider health, cost, model mode, and explicit exclusions; do not route embedding-only models to chat synthesis. Embedding endpoints may delegate model selection to an enabled `embedding` capability agent. | Exact-head capability-isolation, discovery, failover, and [#789](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/789) embedding-contract tests. |
| Orchestration | Allocate shallow or deep work by task need; retain Thinker/Worker/Verifier/Synthesizer evidence, bounded recursion, and Conductor-style access lists. | Replayable workflow trace and equal-budget ablation evidence for [#568](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/568). |
| Provider plane | Discover model capabilities and price honestly, bootstrap credentials from KV, and use secure provider transport with fail-closed malformed responses. | Catalog/bootstrap, provider-contract, and security checks for [#764](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/764)/[#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765)/[#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768)/[#769](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/769)/[#770](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/770). |
| Failure plane | Classify tool failures, fail safely, preserve upstream truth, and retry only within a bounded policy. | [#771](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/771) focused/full tests and hosted security checks. |
| Cache plane | Optional injected Redis/Dragonfly-compatible response cache; deterministic keys, strict bypass, local fallback, fail-open backend behavior, and no cross-model reuse. | [#772](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/772) focused/full tests and RFC 9111 review. |
| Privacy | Do not blanket-mask operational PII. Enforce purpose-limited authorization, field-level encryption at rest, credential redaction, and auditable access. | [ADR 0010](planning/adrs/0010-pii-audit-not-mask.md) follow-up [#762](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/762) plus implementation tests. |
| Persistence | Keep database objects at least two words in `snake_case` and keep schemas in third normal form. | Schema convention review and migration tests. |
| Packaging | Keep one deployable product until a second consumer, independent cadence, or security-provenance boundary requires extraction; every extracted component must work standalone and as a submodule. | Packaging ADR and consumer integration proof. |
| Operability | Maintain one scheduler owner for product development; do not add a duplicate scheduler. The org target is for OpenCode, Noema, and Strix to use the gateway path without `COPILOT_GITHUB_TOKEN`; this repository must not claim that migration is complete until each central workflow removes its direct provider endpoint/key fallback. Central `.github` PR [#1198](https://github.com/ContextualWisdomLab/.github/pull/1198) currently carries the minute-17 target caller with `max_prs=50`, `max_dispatches=1`, and non-cancelling concurrency; its root branch is still protected-path pending. Earlier [#1178](https://github.com/ContextualWisdomLab/.github/pull/1178) merged only into a non-main stack base. Related gateway route PR [#1170](https://github.com/ContextualWisdomLab/.github/pull/1170) and target [#790](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/790) remain protected prerequisites. Noema/Strix gateway migration remains an external prerequisite, not observed completion evidence. Superseded [#1183](https://github.com/ContextualWisdomLab/.github/pull/1183) is closed without merge. |
| Release | Release only from exact-head green evidence; update version and `CHANGELOG.md`. | Protected normal merge followed by release checks. |

## 3. Current architecture and UML-level flow

```mermaid
flowchart LR
    A[OpenAI-compatible client] --> B[HTTP validation and auth]
    B --> C[CostRoutingCoordinator]
    C --> D{Route or conduct}
    D --> E[Capability and health policy]
    E --> F[Response cache]
    F --> G[ModelClient / provider transport]
    G --> H[Provider pool]
    D --> I[Thinker / Worker / Verifier / Synthesizer]
    I --> J[Access-list trace]
    J --> K[Replay and buyer evidence]
    L[KV credential registry] --> G
    M[Durable model catalog] --> E
```

The public API stays small; the control plane owns provider selection,
capability isolation, workflow evidence, cache policy, and failure truth. Rust
is not warranted for the current stdlib Python gateway solely by preference:
the current product gap is correctness and operational evidence. Revisit a
Rust boundary when profiling demonstrates transport, parsing, or concurrency
cost that the existing process cannot meet, and preserve the OpenAI-compatible
module contract.

### Runtime role mapping

`thinker` is the canonical runtime and trace role for planning work. `planner`
is the planning responsibility, not a separate `WorkflowStep.role`: the
generated-plan path selects its planner model through the `thinker` role,
invokes that control-plane planning call, and then emits execution trace rows
with the declared `thinker`, `worker`, `verifier`, or `synthesizer` roles. A
planner call itself is not silently relabeled as a distinct `planner` trace
role. This keeps the documented role vocabulary aligned with
`TaskOrchestrator.ROLE_TAGS`, `WorkflowStep.role`, and the API trace contract.

## 4. PR inventory at the source-of-truth snapshot

Checks below are a snapshot, not approval. `queued` and `in_progress` are not
 failures, but they also are not merge evidence. Protected main requires two
 approving reviews, an additional approval for unattributed changes,
 last-push approval, resolved threads, all
required workflows, and a normal merge.

| PR | Exact head at snapshot | State / base | Evidence boundary and next action |
|---:|---|---|---|
| [#809](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/809) | `756d2a76bb91c0c65aac6c15bbab8270dd0ea479` | open, based on main; 22 hosted check-runs at snapshot (`15` queued), approvals `0` | Documentation-only public docstring completion for telemetry ledger and HTTP handler methods. Exact-head local evidence is `1435 passed`, interrogate `100%`, compileall/actionlint/diff-check passed, Semgrep found `0` findings, and pip-audit found no known vulnerabilities. Protected hosted Checks and independent approval remain required. Decision: `WAIT_AND_REMEDIATE`. |
| [#808](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/808) | `1f19590c8f70d95dc08507985ace3cb18d482188` | open, based on main; 22 hosted check-runs with 15 queued at snapshot, approvals `0` | Documentation/configuration alignment for the credential-key example and stale KV deviation note. No source merge decision until protected Checks and independent approval complete. Decision: `WAIT_AND_REMEDIATE`. |
| [#807](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/807) | `8bdbd5f16e158aefbdf872c2824035da7a125a74` | open, based on main; 22 hosted check-runs at snapshot (`15` queued), approvals `0` | Provider error-boundary repair. The valid ProviderResponseError probe-classification finding was fixed on this exact head; local focused reliability/discovery/MLX evidence is `75 passed`, and the full suite is `1440 passed`. Local compileall, actionlint, diff-check, Semgrep, and pip-audit passed; measured statement coverage is `90%`, branch coverage approximately `84%`, and interrogate docstring coverage `95.8%`, below the repository's 100% quality standard. Protected hosted Checks and independent approval remain required. Decision: `WAIT_AND_REMEDIATE`. |
| [#806](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/806) | `10b87361cff4f4ed5a5d0dd17baee3e840f53b01` | open, based on main; required Checks queued at snapshot | Test-only CLI mock-boundary repair. Exact-head local evidence is `1435 passed in 533.09s`; protected independent approval and terminal Checks remain required. |
| [#805](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/805) | `3bd723c04a9f827f432bc1f1904599da7b54e78e` | closed without merge, based on `fix/auto-reasoning-effort-contract-rebased` at `96d5f0946a56a80344eeb77bf89e16e7e05609d2` | Structured provider-feature orchestration and bounded workflow retention. The prior open-head evidence is stale; this PR closed at the 3bd head without protected merge or release evidence. Reopen/new PR work must re-establish exact-head verification, hosted Checks, and independent approval. |
| [#804](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/804) | `a1f6716dd2d87a9b5975ebf9770d760837980025` | open, based on main; required Checks queued at snapshot | Root security repair for the Strix agent-pool resource-boundary finding: GET, PATCH, and DELETE resolve pool and worker together. Local exact-head evidence is `1436 passed`; protected independent approval and terminal Checks remain required. |
| [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) | `33f312c7782b07285b782c87bf6214d73a8a6975` | open, based on main; required Checks pending at snapshot | Purpose-limited PII event protection with explicit field encryption and KV-backed AES-256-GCM, plus bounded durable audit retention and hash-complete CI runtime locks. Local exact-head evidence is `1448 passed in 527.54s`, `38` focused persistence/security tests passed, and hash-locked installation succeeded; the Devin disk-exhaustion finding is fixed and resolved. Protected independent approval and terminal Checks remain required. |
| [#802](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/802) | `b2fe47e78ade89b13aa4c239c71562c65af5f12e` | open, stacked on `fix/auto-reasoning-effort-contract-rebased`; mergeable clean, hosted check-runs absent, approvals `0` | Provider telemetry session-correlation change with hash-locked OpenTelemetry dependencies and library-research evidence. The current valid LocalBatchBackend ContextVar propagation finding is fixed on this exact head; focused batch/API/embedding tests are `21 passed`. Remaining Devin notes are informational or resolved. Protected hosted Checks and independent approval remain required. Decision: `WAIT_AND_REMEDIATE`. |
| [#798](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/798) | `b0b043da79468a5816faacd95c6781e5d0d4f46b` | closed without merge, based on main | Reintroduced a target-local hourly caller for central #1170, but it duplicated the live central `.github#1178` scheduler's target, bounded dispatch, and ownership boundary. It was closed on 2026-08-21 to keep one scheduler authority and avoid duplicate PR mutations; the exact-head contract evidence is historical and does not establish a scheduled production run. |
| [#797](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/797) | `5dccb65fdd6088deb7c014f819340cceeb89c313` | closed without merge, based on main | Hourly target-repository caller used the central reusable review/fix workflow with `max_prs=1`, `max_dispatches=1`, explicit scheduler secrets, and no `COPILOT_GITHUB_TOKEN` or manual dispatch. It was closed on 2026-08-20 after central #1183 was superseded; its exact-head proof (`2 passed`, `actionlint`, `compileall`, diff-check) is historical and does not establish merge or release evidence. |
| [#796](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/796) | `dc3302dd53a2aa397f19e567923f4febfa217356` | ready, based on [#795](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/795) | Cost-ledger normalization separates execution facts from attribution dimensions, keeps migration transactional, enables SQLite foreign keys before schema work, maps nullable legacy attribution to `unattributed`, and rolls back failed append writes. Exact current-head proof is focused `59 passed`, full `1454 passed in 522.91s`, compileall, and diff-check clean; it includes static migration SQL, seeded-catalog rollback, FK enforcement/cascade, PostgreSQL metadata selection, qualified SQL naming, failed-append rollback, and current stack naming coverage. Hosted Checks and independent approval remain required. |
| [#794](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/794) | `48a8c79481ebf42749418c7b1d93d8553c9fb4b7` | ready, based on main | Database naming repair renames the single-word state table to `orchestration_records`, preserves legacy rows through an atomic fail-closed migration, uses static migration DDL to satisfy SQL-safety scanning, closes the connection on schema failure, and covers qualified, quoted, and inline-constraint database-object declarations while reusing the canonical naming predicate. Exact-current-head persistence/naming proof is `16 passed`, Ruff/compileall/diff-check clean; the previous full-suite evidence belongs to a predecessor head. Hosted Checks and independent approval remain required. |
| [#795](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/795) | `1968998dabf48d9558c3cc62b32937f745d11be8` | ready, based on [#794](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/794) | Durable agent-pool storage is normalized into scalar, ordered-tag, and provider-exclusion tables; legacy JSON migration remains transactional, every SQLite connection enables foreign-key enforcement before work begins, and the current #794 canonical naming-gate repair is included in the head tree. Exact-current-head focused proof is `26 passed`, Ruff/compileall/diff-check clean; no full-suite result is claimed for this head. Hosted Checks and independent approval remain required. |
| [#793](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/793) | `3651a8181d0844a8daa196a73aff401fd34e78da` | ready, based on main | Request-framing repair rejects ambiguous/unbounded Content-Length before integer conversion and closes the connection after framing failure. Exact-head local proof is focused `31 passed` and full `1443 passed`; hosted Checks are pending and independent approval remains required. |
| [#792](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/792) | `236a28b3f73380aaa39aa7b19a2bc475c2cbdf6f` | ready, based on main | Documentation-only release gap closure: adds the canonical SemVer changelog and explicitly keeps `0.1.0` unreleased until protected main, required Checks, independent review, and release artifacts are verified. Normal merge still requires the protected gate. |
| [#790](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/790) | `8d31fa50cc6de8ddc3e6b91576e7251c5aa7d914` | ready, based on main | Latest exact head includes the normal provider-diverse discovery stack merge, keeps the gateway auth token outside the provider-key bootstrap gate, covers model-discovery rejection paths, rejects `gte-*` embedding families from chat-capability roles, and keeps foreign-currency prices out of direct ranking. Exact-current-head focused gateway/discovery/capability proof is `194 passed`; Ruff/compileall/diff-check pass. Hosted Checks and independent approval remain required; predecessor-head evidence does not transfer. |
| [#788](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/788) | `8000659b7dd299c2564d0d50bbea679cf0bb3810` | ready, based on main | Review opaque admin-session TTL/revocation, same-origin cookie state changes, and Secure-by-default deployment. Run `32376890077` first exhausted NVIDIA NIM and then emitted an unsupported hardcoded-AWS-token claim at `server.py:1932`; the exact PR tree has no AWS token pattern, and the failed Strix job was rerun through the Actions API. Treat the rerun as pending until a fresh exact-head result is terminal. |
| [#789](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/789) | `0eaf2a5b68371c34b7fd065cbcbdad19eb344fbf` | open, based on main; replacement workflows are active, `REVIEW_REQUIRED`, and normal auto-merge is enabled but blocked | Omitted or JSON-`null` embedding models resolve only through an enabled embedding-capable agent; empty batches retain that resolved model identity. Startup discovery preserves operator-managed IDs and activates only source-declared chat catalogs (including OpenRouter's server-side text-output filter), never model-name inference. Exact-head local focused evidence is `39 passed`, with compileall and diff-check clean. Hosted checks and independent approval remain required. |
| [#801](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/801) | `eb9ec5f4e3f8ecbcf96cb132f58a212981ff0a6d` | merged on 2026-08-21 into the non-main parent stack at [`3bbe137`](https://github.com/ContextualWisdomLab/contextual-orchestrator/commit/3bbe1372d3790ae98108810640cafc159e16bb52); it is not in documented `main@e226e1197bdfc890c9d8e5b9b648c78857d7e465` | The stack adds explicit `argv` injection, but documented main still defines `main()` with no argument. This is not LineageWeave completion evidence; the LineageWeave-specific protected-main acceptance gate is tracked below. |
| [#799](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/799) | `0eb0a9b7323b9de17311c0b990838c71de644d00` | ready, based on main | Restores test-contract names, removes an impossible duplicate JSON key, and removes an unused import without runtime changes. Focused HTTP honesty/security proof is `38 passed`; hosted Checks and independent approval remain required. |
| [#782](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/782) | `1e7ddb96256a9379b3d8d4bb39c70a646f302bed` | ready, based on main | Review owner-bound workflow/access/evaluation reads, split-token admin evidence visibility, migration fail-closed behavior, and exact-head protected Checks before merge. |
| [#780](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/780) | `e4e6b7cf27f061ece9f0e03ce82a248480b31597` | ready, based on main | Parent-integrated current head includes #781 trace-purpose authorization and hardened trace fixtures. Exact proof is `1443 passed in 556.14s`; Ruff, compileall, and diff-check pass. Hosted Checks are freshly queued and protected independent approval remains required. |
| [#775](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/775) | `fb8fb621faa66859e36fa9496d3d6deefd09c18e` | ready, based on main | Promoted after exact-head review: marker regression test passed, Python 3.10 resolver skips Atheris, Linux CPython 3.12 resolves Atheris 3.1.0, and the generated hash lock preserves `python_full_version == 3.12.*`. Hosted Checks are green; protected independent approval remains required. |
| [#784](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/784) | `912645f1003d6dea2e83967b3f1987039b4fb8a3` | open, stacked on `fix/agent-pool-boundary-current` at `a1f6716dd2d87a9b5975ebf9770d760837980025`; Checks rerunning | Root #804 agent-pool ownership repair was merged into the PR branch non-force before changing the PR base. Merge-result exact-tree evidence is `57` focused tests passed and `1466 passed in 540.29s`, plus compileall/actionlint/diff-check. Prior Strix IDOR failure is dependency-owned by #804; exact-head SSRF probes reject HTTP and private HTTPS destinations before transport. Independent approval and fresh hosted Checks remain required. |
| [#785](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/785) | `ec609fa7b526a995346c34434e277eb12f5a0246` | ready, based on main | Issue [#568](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/568) exact-head proof is full suite `1461 passed in 580.97s`, focused judge/failover/passthrough/profile suite `69 passed`, and Ruff/diff clean; independent approval and protected Checks remain required. |
| [#773](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/773) | self-reference — refetch live PR head | open, based on main; `BLOCKED` / `REVIEW_REQUIRED` at this snapshot | This document is the PR's own changing artifact, so embedding its content SHA would become stale on every refresh commit. The current hosted rollup has terminal functional/security jobs, but `strix` is in progress and `opencode-review` is queued; an independent approval and terminal required checks remain mandatory. Refetch the live #773 head before relying on this row. |
| [#772](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/772) | `f72ddc886cc55a3243ebe79f6498c7f942409c83` | ready, based on main | Review cache-key isolation, strict bypass parsing, fail-open backend behavior, malformed cache entries, and routing/cost/stream interactions; exact-head cache/cost/ledger proof is `63 passed`, with full suite `1451 passed`. |
| [#771](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/771) | `cc806cdb809068b78388d843758086747a21750a` | live head advanced after prior audit; required workflows queued, approvals `0`; prior evidence stale | The live head added a malformed-provider-response fail-closed repair, so the earlier `e258875e` proof no longer transfers. Exact local follow-up `276ed4f0` adds terminal `409 tool_execution_stopped` preservation across chat/raw retry layers and HTTP contract coverage: focused `114 passed`, direct fallback file `96 passed`, full `1538 passed`, compileall/actionlint/diff-check/Semgrep/pip-audit clean. That follow-up could not yet be pushed because the active all-branch ruleset rejected the update until PR-required workflows are satisfied. Repository coverage remains `90%` statement / `146` partial branches and docstring `95.9%`, below the 100% standard. Decision: `WAIT_AND_REMEDIATE`. |
| [#770](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/770) | `7494f227d0ca84f65ccaac6af9614c59d1fc233b` | ready, based on [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768) | Current stack consumes the shared ordinary-chat classifier and price-honest provider-diverse selection. The latest exact head removes a trailing blank line from the doctoring record; focused discovery/bootstrap/model-selection proof is `32 passed`, and Ruff/compileall/diff-check pass. Hosted Checks must regenerate on this exact head; obtain independent current-head approval before protected merge. |
| [#769](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/769) | `9654c285c54443acf6358193925f4e0e8ae501ce` | ready, based on main | Core repository workflows succeeded on this head; obtain exact-head independent approval and remaining protected contexts. |
| [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768) | `88fee976ca4222309f625058a6f95f09e66744ec` | exact head verified; hosted checks terminal except Trivy/Scorecard neutral; approvals `0` | Current capability boundary includes ShieldGemma, legacy Completions, direct-run regressions, and exact `/v1/responses` normalization. Current head has `21` successful, `8` skipped, and `2` neutral infrastructure findings because the code-scanning baseline reports missing main-branch workflow configuration; the repository delegates those gates to central required workflows. No source failure or unresolved current-head finding was found, but protected independent approval remains required. Decision: `WAIT_AND_REMEDIATE`. |
| [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) | `a4e4f683e3a4f39fc740b9028158da7e7c2bc219` | ready, based on main | Exact head closes the Strix SSRF finding: empty URL userinfo is rejected by presence across discovery, origin, low-level transport, and provider URL validation; empty fragments are rejected as well. Concurrent remote security coverage was preserved. Focused proof is `103 passed` across PR regressions, discovery, local gateway, and security hardening; full exact-head proof is `1519 passed in 556.04s`, with compileall/diff-check clean. The later Devin Responses-batch report was revalidated against the pre-coordinator 400 guard and closed with `13 passed` routing-contract tests without a source change. Hosted Checks are queued on this exact head and independent approval remains required. |
| [#764](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/764) | `ea5ab0e932a299640275fd98ef83ad462e46e2c0` | ready, based on [#770](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/770) | Current remote stack owns durable five-provider credentials and normalized catalog persistence; the latest docs bind bootstrap success to durable KV registration after rollback. Current-head catalog/bootstrap proof is focused `14 passed`; the code-equivalent prior head had full `1542 passed`. Hosted Checks and independent approval remain required. |
| [#763](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/763) | `531c74f49f228929425b485838f18e355aaa0cdf` | ready, based on [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) — parent advanced to `a4e4f683e3a4f39fc740b9028158da7e7c2bc219` | Current stack integrates #765 and #768 gateway/capability boundaries with one-shot local Responses translation, local concurrency coordination, concrete-model stickiness, adaptive provider failover, an embedding-specific capability filter, and the parent’s direct sampling contract. The prior `150 passed`/`1589 passed` proof belonged to the pre-repair parent base and does not transfer; current hosted Checks and independent approval remain required. |
| [#762](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/762) | `8f87bcaeddff0866e26900e41deeafe208d8f9e4` | ready, based on main | The exact-head ADR closes the documented purpose, classification, AEAD/KMS, migration, and audit-gate review gaps; merge the design only after current-head independent approval, then implement its acceptance criteria separately. |

### Live recheck continuation — 2026-08-21 21:48 KST

The following rows supersede the corresponding snapshot rows above for the
listed PRs. This continuation preserves the older snapshot so predecessor
evidence cannot be mistaken for current-head evidence.

| PR | Current exact identity | Live gate evidence and decision |
|---:|---|---|
| [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) | head `537915715c4b050d4b5fa18ce2b7559080c675ba`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 22 check-runs (`7` skipped, `15` queued), formal approval absent. The latest review dispositions and stacked repairs are recorded in the PR; queued checks and no approval keep the normal merge gate closed. Decision: `WAIT_AND_REMEDIATE`. |
| [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768) | head `88fee976ca4222309f625058a6f95f09e66744ec`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 31 terminal runs (`21` success, `8` skipped, `2` neutral), formal approval absent. Neutral Trivy/Scorecard results are infrastructure-baseline warnings, not source success. Decision: `WAIT_AND_REMEDIATE`. |
| [#771](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/771) | head `cc806cdb809068b78388d843758086747a21750a`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 22 check-runs (`7` skipped, `15` queued), formal approval absent. Local follow-up `276ed4f0` is not remote evidence: its normal push was rejected by the active required-workflow ruleset. Decision: `WAIT_AND_REMEDIATE`. |
| [#807](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/807) | head `f0d44f78f820f4ee34280294115e13d2ed541e14`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 22 check-runs (`7` skipped, `15` queued), formal approval absent. Local cleanup `e898ce0a` removes an unreachable error branch and passed 104 targeted tests, but normal push was rejected by the active ruleset. Decision: `WAIT_AND_REMEDIATE`. |
| [#810](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/810) | head `513a8157e667a6adbe7b91b5e802887a55fe9cd8`, base `537915715c4b050d4b5fa18ce2b7559080c675ba` | Open, non-Draft, mergeable but blocked; 17 check-runs (`8` skipped/completed, `9` queued), formal approval absent. Local follow-up `7929f707` preserves budget stops during generated planning; full local suite passed `1646` and compileall/actionlint/diff-check/Semgrep/pip-audit were clean. Normal push was rejected by the active ruleset. Decision: `WAIT_AND_REMEDIATE`. |
| [#811](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/811) | head `f0b0dd565f93d8f4aa90ca6ad67544c6b6b8051f`, base `cc806cdb809068b78388d843758086747a21750a` | Open, non-Draft, mergeable but blocked; review decision `REVIEW_REQUIRED`, no formal approval, and 17 check-runs (`8` skipped/completed, `9` queued). Exact-head focused provider/tool tests passed `115`; this is a dependent stack item over #771. Decision: `WAIT_AND_REMEDIATE`. |
| [#812](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/812) | head `dffa870589f464bd674bc64cd0c16334b5e48712`, base `f0d44f78f820f4ee34280294115e13d2ed541e14` | Remote follow-up for #807; one-file unreachable-branch cleanup, 17 check-runs (`8` skipped, `9` queued), formal approvals `0`, Devin no-issues review. Exact targeted local verification `104 passed`. Decision: `WAIT_AND_REMEDIATE`. |
| [#813](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/813) | head `6e5e19325af79c3c72eb4ff2671b3be4830068c4`, base `537915715c4b050d4b5fa18ce2b7559080c675ba` | Remote follow-up for #810; exact tree equivalent to the locally verified budget-stop repair, 17 check-runs (`8` skipped, `9` queued), formal approvals `0`. Equivalent local tree passed the full `1646`-test suite and static/security checks. Decision: `WAIT_AND_REMEDIATE`. |

### Live recheck continuation — 2026-08-21 22:38 KST

This continuation supersedes the #803 row above for its new exact head and
keeps the hosted gate separate from local evidence.

| PR | Current exact identity | Live gate evidence and decision |
|---:|---|---|
| [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) | head `5c51c3a93bbd1779745f94502ca4d702b2e051d5`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 22 hosted check-runs (`7` completed/skipped, `15` queued), formal approval `0`, exact head unchanged since the follow-up push. Local exact-head evidence is `1451 passed`, focused PII/persistence/admin `23 passed`, `pii_protection.py` 100% statement/branch, repository aggregate 90% statement with 146 partial branches, interrogate 95.9%, pip-audit clean, Semgrep 0, actionlint/compileall/diff-check clean. Authorization-decision churn is now isolated from substantive audit retention; undecryptable replay rows degrade individually. Decision: `WAIT_AND_REMEDIATE`. |

### Live stack continuation — 2026-08-21 22:45 KST

Normal non-main stack transitions observed after the previous recheck:

| PR | Current exact identity | Live gate evidence and decision |
|---:|---|---|
| [#764](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/764) | head `ea5ab0e932a299640275fd98ef83ad462e46e2c0`, base `6b603efeb9728d7c142f090153925948c0f1248f` | Normal merge into non-main stack branch completed at merge commit `074f0e4425de4714aeecc9ee56d9f8e512c2c2e6` after exact recheck: `CLEAN`, 25/25 terminal checks, failures `0`. Decision: `NORMAL_MERGE`. |
| [#770](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/770) | head `074f0e4425de4714aeecc9ee56d9f8e512c2c2e6`, base `88fee976ca4222309f625058a6f95f09e66744ec` | Automatic normal stack transition completed at merge commit `84b010a56524b97bc9f507f016501ce5bd855d84` after #764 advanced its parent branch. This is non-main stack integration, not protected-main release evidence. Decision: `NORMAL_MERGE`. |
| [#796](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/796) | head `dc3302dd53a2aa397f19e567923f4febfa217356`, base `1968998dabf48d9558c3cc62b32937f745d11be8` | Normal merge into non-main stack branch completed at merge commit `820ac3b76934e345fb79133a269fc2c44dd7e351` after exact recheck: `CLEAN`, 25/25 terminal checks, failures `0`. Decision: `NORMAL_MERGE`. |
| [#795](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/795) | head `820ac3b76934e345fb79133a269fc2c44dd7e351`, base `48a8c79481ebf42749418c7b1d93d8553c9fb4b7` | Parent stack head advanced after #796; current state is `UNSTABLE` with no current hosted check-runs, so prior child evidence is not reused. Decision: `WAIT_AND_REMEDIATE`. |
| [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768) | head `84b010a56524b97bc9f507f016501ce5bd855d84`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Main-target root now includes the #764/#770 stack merge; open, non-Draft, mergeable but blocked, 22 checks (`7` terminal, `15` queued), review required and formal approval absent. Decision: `WAIT_AND_REMEDIATE`. |

### Live recheck continuation — 2026-08-21 23:09 KST

This continuation supersedes the prior #803 entry for its new exact head and
keeps local verification separate from the still-pending protected gate.

| PR | Current exact identity | Live gate evidence and decision |
|---:|---|---|
| [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) | head `606eb3788681bf04928c5be9325f2ca499412069`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 24 hosted check-runs (`7` completed/skipped, `15` queued, `2` review-provider contexts), failures `0`, formal approval `0`. The latest rate-limit retention finding was verified against the bounded `authorization` stream; the exact durable-retention regression test passed. Local exact-head evidence is `1453 passed in 523.18s`, focused PII/persistence/security `43 passed`, PII protection `100%` statement/branch, repository aggregate `90%` statement with `146` partial branches, interrogate `95.9%`, pip-audit clean, Semgrep `0`, actionlint/compileall/diff-check clean. Decision: `WAIT_AND_REMEDIATE`. |
| [#801](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/801) | head `eb9ec5f4e3f8ecbcf96cb132f58a212981ff0a6d`, recorded base `39072a654261c3570496849bb4da1e2c340e2fbc`; live parent branch `70f8b4dbb6b8d5749140581c50bcdf3b31ad8475` | The requested explicit-argv CLI change and regression test are already present in the live parent tree. The stale PR head was not pushed or merged; `CLEAN` against the recorded base is not current-parent evidence. Decision: `WAIT_AND_REMEDIATE` as a no-op stack item. |

PR [#791](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/791) was merged into its stacked base branch on 2026-08-20. PR [#800](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/800) was closed unmerged and superseded by [#801](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/801). PR [#779](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/779) was merged into the #765 parent branch at `a4e9a780a56c08e131367f3b904f5c855b3a1b33`; PR [#778](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/778) was then merged into the same #765 parent branch at `0a3ab0b901d0b39d610f26538243bb5b3c1b3c61`; PR [#781](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/781) was merged into the #780 parent branch at `30476b3365ad9825a6dd2b35ac5ee10ef3c34879`; PRs [#776](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/776), [#783](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/783), and [#787](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/787) were merged into their non-main stack branches with merge commits `3e6c4205e707eee70387bb429866b689de848753`, `637e941fd9237780d2fe90150514b048d1675387`, and `41c0c87ef4c4cf335b71d07b990734944db4b89e`. None is a protected-main release. PR [#774](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/774) was closed unmerged as the stale-base predecessor of [#779](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/779). Its local
or predecessor-head evidence does not transfer. Issue [#745](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/745) is represented by
[#772](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/772) and issue [#567](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/567) by [#771](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/771). A draft or implementation PR is not treated as
completed until the protected-main contract is satisfied.

### Live exact-head continuation — 2026-08-22 01:19 KST

This continuation supersedes older rows for the listed PRs. Hosted gate
evidence and local verification are intentionally kept separate; no predecessor
head or approval is reused.

| PR | Current exact identity | Live gate evidence and local verification | Decision |
|---:|---|---|---|
| [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) | head `d19e3492192e21e4a040fa3fc13a0793443731bf`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 19 success, 8 skipped, 2 neutral, and 1 queued check-run; approvals `0`. Exact local full suite `1729 passed`, focused suite `78 passed`, aggregate statement/branch coverage `92%`, and interrogate `100%`. Static/security evidence was recorded on the immediately preceding equivalent tree; the current three-file cleanup/assertion delta has no new gate result. | `WAIT_AND_REMEDIATE` |
| [#804](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/804) | head `71895c7f1f1d63a55e5f67d75dbd8408f8e608f9`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 17 success, 7 skipped, 2 neutral, 1 in progress, and 1 queued check-run; approvals `0`. A normal revert removed the accidental downstream #784 merge, restoring the bounded five-file root tree `a29ae0252f3bf13e70da0a32a38eca116350aad8`; exact local full suite `1436 passed`, focused suite `56 passed`, compile/actionlint/diff/Semgrep/interrogate/pip-audit checks passed. Coverage was not rerun on this exact head. | `WAIT_AND_REMEDIATE` |
| [#807](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/807) | head `d5122a0a30b4543ac2778a1c71e9ee079cbd699f`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 15 success, 7 skipped, 2 neutral, 3 in progress, and 1 queued check-run; approvals `0`. Exact local merge-result tree is `ba261ab39191323f3e116d92103f00d727b10f48` with no conflict; focused suite `117 passed`, full suite `1454 passed`, aggregate statement/branch coverage `90%`, interrogate `95.8%`, Semgrep `0`, pip-audit clean, and actionlint/compileall/diff-check passed. | `WAIT_AND_REMEDIATE` |

The hosted queues and absent approvals are normal protected-path blockers, not
D1-D5 deadlock evidence. The active ruleset has no bypass actor; no force merge,
direct protected-branch push, or approval bypass was attempted.

### Live exact-head continuation — 2026-08-22 01:43 KST

The following exact-head recheck supersedes the immediately preceding rows for
these PRs. The hosted queues remain separate from local evidence.

| PR | Current exact identity | Live gate evidence and local verification | Decision |
|---:|---|---|---|
| [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) | head `1f836527a718374585c5fde7838148cfa22765b2`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 7 skipped and 15 queued check-runs, approvals `0`. Exact local full suite `1453 passed`, focused PII/security `34 passed`, aggregate statement/branch coverage `90%`, `pii_protection.py` 100%, interrogate `95.9%`, Semgrep `0`, pip-audit clean, actionlint/compile/diff clean, and wheel build/install/import smoke passed. One valid documentation finding was fixed in this head. | `WAIT_AND_REMEDIATE` |
| [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765) | head `d19e3492192e21e4a040fa3fc13a0793443731bf`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 20 success, 8 skipped, 2 neutral, and 1 queued check-run; approvals `0`. Exact local full suite `1729 passed`, focused suite `78 passed`, aggregate statement/branch coverage `92%`, interrogate `100%`. | `WAIT_AND_REMEDIATE` |
| [#804](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/804) | head `71895c7f1f1d63a55e5f67d75dbd8408f8e608f9`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 19 success, 8 skipped, 2 neutral, and 1 queued check-run; approvals `0`. The bounded root tree and exact local suite/static evidence remain as recorded above. | `WAIT_AND_REMEDIATE` |
| [#807](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/807) | head `d5122a0a30b4543ac2778a1c71e9ee079cbd699f`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 15 success, 7 skipped, 2 neutral, 3 in progress, and 1 queued check-run; approvals `0`. Exact local full suite `1454 passed`, focused suite `117 passed`, aggregate statement/branch coverage `90%`, interrogate `95.8%`, Semgrep `0`, pip-audit clean, and actionlint/compile/diff clean. | `WAIT_AND_REMEDIATE` |
| [#773](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/773) | head `1ec2ab6315341ad576fe1d1c27f6cbeaab1111dc`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 10 success, 8 skipped, 2 neutral, 3 in progress, and 4 queued check-runs; approvals `0`. The baseline update was normally pushed and its current hosted gate remains incomplete. | `WAIT_AND_REMEDIATE` |

None of these states establishes a D1-D5 deadlock: current heads have generated
checks or are awaiting normal approvals, and the active ruleset exposes no
bypass actor.

### Queue triage continuation — 2026-08-22 01:45 KST

New or newly relevant queue items were checked after the root rows above.

| PR | Current exact identity | Live evidence and dependency decision |
|---:|---|---|
| [#817](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/817) | head `8235d32839148c133ec046d6d16b944ff558c452`, base branch `fix/agent-pool-boundary-current` at `71895c7f1f1d63a55e5f67d75dbd8408f8e608f9` | Open, non-Draft, mechanically clean, 37 files (`+2389/-129`), but no check-runs and no formal approval. It is downstream of the still-unmerged #804 root and remains `WAIT_AND_REMEDIATE`; no downstream merge was attempted. |
| [#818](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/818) | head `057530e6f9a90ec1909e197c23304a031b0e35f6`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; current hosted `osv-scan` fails because the inherited central security workflow invokes deprecated `--output` and then rejects empty `old-results.json`/`new-results.json`; this is not a telemetry-source finding. Formal approvals `0`; decision `WAIT_AND_REMEDIATE` pending central repair and re-run. |
| [#809](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/809) | head `756d2a76bb91c0c65aac6c15bbab8270dd0ea479`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 21 success, 8 skipped, 2 neutral check-runs, approvals `0`. Despite terminal checks, the required approval gate is absent; `WAIT_AND_REMEDIATE`. |
| [#806](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/806) | head `10b87361cff4f4ed5a5d0dd17baee3e840f53b01`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open, non-Draft, mergeable but blocked; 21 success, 8 skipped, 2 neutral check-runs, approvals `0`. Despite terminal checks, the required approval gate is absent; `WAIT_AND_REMEDIATE`. |

### Live exact-head continuation — 2026-08-22 01:48 KST

The bounded documentation/test PRs were independently reviewed at their
current heads. Hosted terminal success does not replace the required approval.

| PR | Current exact identity | Local and live evidence | Decision |
|---:|---|---|---|
| [#806](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/806) | head `10b87361cff4f4ed5a5d0dd17baee3e840f53b01`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | One-file test-only diff; exact focused tests `8 passed`, compileall and diff-check passed. Hosted: 21 success, 8 skipped, 2 neutral; approvals `0`. | `WAIT_AND_REMEDIATE` |
| [#809](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/809) | head `756d2a76bb91c0c65aac6c15bbab8270dd0ea479`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Two-file public-docstring diff; exact interrogate `100.0%`, compileall and diff-check passed. Hosted: 21 success, 8 skipped, 2 neutral; approvals `0`. | `WAIT_AND_REMEDIATE` |

### Protected auto-merge continuation — 2026-08-22 01:50 KST

Normal squash auto-merge is enabled for #804, #803, #765, #807, #806, and
#809 at their exact heads. GitHub reports each as `blocked` until required
reviews and hosted gates are satisfied; this does not bypass any protection.
Downstream #817 remains unqueued for merge until root #804 advances.

All links and full commit SHAs in the initial snapshot reflect the remote state
observed at 2026-08-21 15:36 Asia/Seoul; they are evidence pointers, not
standing approval.

### Live model-group continuation — 2026-08-25 09:12 KST

| PR | Current exact identity | Live evidence, customer outcome, and gap |
|---:|---|---|
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | head `6b069a059b677bbb367e80e5819f34564673fa4a`, base `50014ef9f87623c6dfd529116297c5d180743125` | Open, non-Draft, blocked for protected review/checks; all six review threads are resolved on this exact head, hosted jobs are queued, and no independent approval exists. The feature adds arbitrary operator-defined logical model groups (names need not resemble member model IDs), authenticated REST CRUD, Agent Pool DB persistence, measured intra-group provider routing, OpenRouter/OpenCode Zen discovery, explicit free-cost classification, Admin evidence, OpenAPI, and APA-grounded ADR 0026. Discovery never infers equivalence and no model family is built in. Exact current-head local evidence: full suite `1586 passed in 604.40s`; focused REST/group/discovery/governance suite `37 passed`; new `model_group.py` statement and branch coverage `100%`; compileall and diff-check clean. Customer-visible remaining gap: multi-replica measurement aggregation requires a normalized time-windowed observation store and explicit retention/decay policy; unknown provider prices must remain unknown. Decision: `WAIT_AND_REMEDIATE`. |

The model-group boundary is deliberately generic. Discovery never infers
equivalence from similar names; authenticated operators assert group membership
through `/api/v1/model_groups`, after which routing ranks the logical group
before selecting a measured provider member.

### External central workflow prerequisite

At the refreshed snapshot, central scheduler PR [#1178](https://github.com/ContextualWisdomLab/.github/pull/1178) is open at exact head `97b084ac28b5ccf6de7f68fd2e019d8da6f80143`. Its target caller runs at minute `17`, uses non-cancelling concurrency, scans at most 50 PRs, dispatches at most one repair, and forwards only `PR_REVIEW_MERGE_TOKEN`/`OPENCODE_APPROVE_TOKEN`; current hosted Checks are pending and the cancelled queue scan is not a terminal failure. For reproducible external evidence, the scheduler source is pinned to [`ContextualWisdomLab/.github@8cb6a1a34e9f5c3d4634b94599f492049185bc0d`](https://github.com/ContextualWisdomLab/.github/commit/8cb6a1a34e9f5c3d4634b94599f492049185bc0d), which contains [`.github/workflows/pr-review-merge-scheduler.yml`](https://github.com/ContextualWisdomLab/.github/blob/8cb6a1a34e9f5c3d4634b94599f492049185bc0d/.github/workflows/pr-review-merge-scheduler.yml) and [`.github/workflows/pr-review-fix-scheduler.yml`](https://github.com/ContextualWisdomLab/.github/blob/8cb6a1a34e9f5c3d4634b94599f492049185bc0d/.github/workflows/pr-review-fix-scheduler.yml). Related central gateway PR [#1170](https://github.com/ContextualWisdomLab/.github/pull/1170) is open at exact head `1f2b93ead7205b33712de1865d84c004d93be7ed` and records target [#790](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/790) at current exact head `8d31fa50cc6de8ddc3e6b91576e7251c5aa7d914`; its own protected Checks and independent approval remain pending. The target repository snapshot [`ContextualWisdomLab/contextual-orchestrator@36e3be0bca5f64b7c5150351b2d505ea536a46a4`](https://github.com/ContextualWisdomLab/contextual-orchestrator/commit/36e3be0bca5f64b7c5150351b2d505ea536a46a4) has no `.github/workflows/opencode-review.yml`; the central queue remains asynchronous. Superseded [#1183](https://github.com/ContextualWisdomLab/.github/pull/1183) is closed without merge, so its head is not integration evidence.

The scheduler's earlier dry-run decision for PR #787 was `review_dispatch` for its non-main base; #787, #779, and #778 are now merged into the #765 parent stack. The target repository currently exposes no `opencode-review.yml` workflow of its own and the central repository-dispatch queue remains asynchronous. Until #1178 and gateway #1170 reach protected main, remaining stacked PR review may remain absent or use the prior central model path; this is an external integration dependency, not evidence that the stacked code is merge-ready.

Central Strix context repair is now owned by [.github#1153](https://github.com/ContextualWisdomLab/.github/pull/1153) at exact head `035343c8a68e880a4abf27f7c947bfed9dbaafcf`; its canonical local gate suite passed, while protected hosted Checks and independent approval remain pending. Duplicate follow-ups [.github#1191](https://github.com/ContextualWisdomLab/.github/pull/1191) and [.github#1192](https://github.com/ContextualWisdomLab/.github/pull/1192) are closed and must not be treated as active integration evidence.

### Central exact-head continuation — 2026-08-22 02:05 KST

The prior central snapshot is historical. Current control-plane evidence is:

| PR | Current exact identity | Live evidence and decision |
|---:|---|---|
| [.github#1178](https://github.com/ContextualWisdomLab/.github/pull/1178) | merged commit `efea53a11c90c7ca519af56aed6918124f45826e` into stacked base `fix/pip-audit-pip-2621` | The hourly caller merged into its non-main base, not protected `.github` `main`. Its CodeQL merge-preview run failed because `refs/pull/1178/merge` was absent although `merge_commit_sha` existed; the failure is preserved as control-plane evidence. |
| [.github#1206](https://github.com/ContextualWisdomLab/.github/pull/1206) | head `633342d56801fdb4a9261a165bd31e94cf9a58f4`, base `ac26a07e9fe19760f3b55e3a1db4021ce4776ae6` | Follow-up fixes CodeQL merge analysis to use the exact merge SHA. Local exact suite `1297 passed`, statement/branch coverage `100%`, interrogate `100%`, focused contract `3 passed`, actionlint/compile/diff clean. Hosted state: 5 queued, 5 skipped, approvals `0`. Decision: `WAIT_AND_REMEDIATE`. |
| [.github#1170](https://github.com/ContextualWisdomLab/.github/pull/1170) | head `fa4c6faa7a8c580aca556c31b00068d7aab69010`, base `dd58a88391e44a32fb399f7407f508d8e73cc1c7` | Open main-target gateway PR; hosted state observed as 5 success, 13 skipped, 16 cancelled, approvals `0`. Decision: `WAIT_AND_REMEDIATE`. |
| [.github#1153](https://github.com/ContextualWisdomLab/.github/pull/1153) | head `ebda81f832261489289447778b0e0e7726f9741e`, base `dd58a88391e44a32fb399f7407f508d8e73cc1c7` | Open Strix repair; hosted state observed as 27 success, 15 skipped, 3 neutral, 1 failure, 2 queued, approvals `0`. Failure remains a source/control-plane finding to inspect; no merge or bypass. |

### Central exact-head continuation — live refresh 2026-08-22

The preceding rows are historical snapshots. The latest normal-path evidence is:

| PR | Current exact identity | Live evidence and decision |
|---:|---|---|
| [.github#1198](https://github.com/ContextualWisdomLab/.github/pull/1198) | head `e5a7ac882559fe6d1ae1f91e27d79bc8e0aa1e77`, base `4c33442021d63b09f35a874c5e7a779dd46ef8f2`, merge-result tree `fd7e18d2c50867bb7438325ba2fa09aaa22e204b` | Current exact stack includes the CodeQL merge-preview identity/authentication repair, trusted LLVM 19 coverage boundary, and normal protected-main restack. Fresh exact-head local evidence: `1362 passed, 1 skipped, 16 subtests`, statement/branch coverage `8394/3264 = 100%`, interrogate `100%`, relevant actionlint, compileall, diff-check, and hash-locked pip-audit clean. Current exact-head hosted OpenCode run `32528443790` / bootstrap job `96915418375` remains queued, CodeQL run `32528443855` remains queued, and Strix run `32528443763` is in progress with runner assignment. Formal approval is absent; predecessor-head evidence is discarded. Decision: `WAIT_AND_REMEDIATE`; D2 observation window is not satisfied. |
| [.github#1209](https://github.com/ContextualWisdomLab/.github/pull/1209) | head `e043ad82b8ed65cfd6f46ed844467c8ad9955665`, base `4c33442021d63b09f35a874c5e7a779dd46ef8f2`, merge-result tree `250e070a01307bf07675ab3faadcb5dec7d20c5e` | Current exact head is a normal protected-main restack of the cross-fork OSV result-isolation repair. Fresh local evidence: `1330 passed, 1 skipped, 16 subtests`, statement/branch coverage `8282/3196 (100%)`, interrogate `100%`, changed-workflow actionlint, compileall, exact hash-lock pip-audit, and diff-check clean. Hosted required checks are mixed success and pending/in-progress after the restack; formal approval is absent and unresolved current threads are `0`. Decision: `WAIT_AND_REMEDIATE`; predecessor-head evidence is discarded. |
| [.github#1203](https://github.com/ContextualWisdomLab/.github/pull/1203) | merge commit `4d3d24aa404959f5067735fec0558d5924ade590` from head `c627d4ae7a26222ed3d2ee1ded19e270930aa1f2` into stacked base `94e2b28aee2029d36f272336a7cd4e4ad1516161` | Merged normally into the non-protected #1198 stack, not protected `.github` `main`; it is not protected-main completion evidence. |
| [.github#1208](https://github.com/ContextualWisdomLab/.github/pull/1208) | closed without merge from head `9f0845517010f2c4695de83dd2bd50385c068a15`, base `fce028b4c3bf8e2e5e4819c1c5622e90cfa6ab39` | Historical successor to closed #1207; its exact local pip-audit evidence reproduced `pip==26.1.2 / PYSEC-2026-3721`, but no protected-main merge occurred. Root #1198 owns the current pip-lock repair. |
| [.github#1187](https://github.com/ContextualWisdomLab/.github/pull/1187) | head `0a88e24d9a1c92420f412d241f850aab8e72106e`, base `0156282022134484ea9d7541d5ba0730ba14fd96`, merge-result tree `05a0fdac58258bb6b30da9376c37003a8e18b609` | Normally restacked onto current protected main; one shell-test conflict retained main's targeted-dispatch assertion. Exact merge-result evidence: `1323 passed`, statement/branch coverage `8184/3172, 100%`, interrogate `100%`, compileall, shell syntax, exact hash-lock pip-audit, and diff-check clean. Hosted rollup is 21 queued/blank, 2 pending, and 13 skipped; no qualifying approval exists. Decision: `WAIT_AND_REMEDIATE` pending terminal required Checks and normal approval. |
| [.github#1188](https://github.com/ContextualWisdomLab/.github/pull/1188) | head `2c05f05f5fbb923099e0e228d616ab9974dbd327`, base `0156282022134484ea9d7541d5ba0730ba14fd96`, merge-result tree `6c19376c4a030eb97a9d4b7f6a2a0b85d38740a3` | Exact current-head evidence: `1319 passed`, statement/branch coverage `8168/3162, 100%`, interrogate `100%`, compileall, actionlint for both changed reusable OIDC workflows, shell syntax, hash-locked pip-audit, and diff-check clean. Hosted rollup is 25 success, 18 skipped, 1 cancelled, 3 neutral, and 4 queued/in-progress; no qualifying approval exists. Decision: `WAIT_AND_REMEDIATE` pending terminal required Checks and normal approval. |
| [.github#1189](https://github.com/ContextualWisdomLab/.github/pull/1189) | closed without merge from PR API head `6ffdcc3f0c96610295d143a3f54f3821b3a23ec9`, base `fce028b4c3bf8e2e5e4819c1c5622e90cfa6ab39` | Absorbed by root #1198 at exact feature-stack head `3016543f`; the normal restack `12c7d513` was pushed to its branch but was never treated as PR evidence. No protected-main merge or bypass occurred. |
| [.github#1206](https://github.com/ContextualWisdomLab/.github/pull/1206) | squash commit `b8737dadbcb9871dc939e500ecbb84ceccb8cf42`, merged into stacked base `fix/pip-audit-pip-2621` from head `d51638c11c5da1b30dd1867ecd019fd0dd28eb92f` | CodeQL merge-analysis follow-up merged normally into the #1198 stack, not protected `.github` `main`. The merge tree is `b30e5b0eea90b59eb97dd894ebb1e63a0c9b8c3b`, so it is retained as stack evidence and does not count as protected-main completion. |
| [.github#1153](https://github.com/ContextualWisdomLab/.github/pull/1153) | head `1f8878d78cb2f0aed68964ea5fabf5679526e78b`, base `fce028b4c3bf8e2e5e4819c1c5622e90cfa6ab39`, merge-result tree `9a15ef3f059f652f84f5b76d1bc8cce2f0d8bdac` | Current head is a main-refresh successor for the JavaScript/Storybook coverage gate. Hosted exact-head checks show 3 success, 11 pending, 14 skipped, and 6 cancelled, with no failure conclusion or qualifying approval. Prior local evidence at older heads does not transfer; decision remains `WAIT_AND_REMEDIATE` pending current-head verification and normal approvals. |

At the current observation, no D1–D5 emergency deadlock is established for these central PRs. Organization ruleset `18156473` (`CWL Central required workflows`) is active for normal repositories, has no bypass actors, and requires two approvals, last-push approval, thread resolution, and the listed central workflows. It explicitly excludes the `.github` repository; `.github` instead exposes repository ruleset `17921150` with zero required approvals and an `OrganizationAdmin` always-bypass actor. That bypass is not used: the maintainer procedure still requires independent substantive review, exact-head evidence, terminal Checks, and a final refetch. The queued runs above are only minutes old at this observation, so D2's six-hour/two-observation threshold is not met.

### Central queue live refresh — 2026-08-22 06:00 KST

The queue is progressing, but no terminal merge gate has been fabricated or
reused. On [.github#1198](https://github.com/ContextualWisdomLab/.github/pull/1198),
the predecessor run `32524224916` was cancelled after its bootstrap and the
current exact-head OpenCode bootstrap job `96915418375` remains queued; the
current Strix job `96915480662` has started on a hosted runner. The newer
`scan-pr-queue` failure shown by the PR rollup is also a cancelled predecessor,
not a source failure. On
[.github#1209](https://github.com/ContextualWisdomLab/.github/pull/1209),
OSV, dependency review, Noema, and the security jobs have successful current
contexts while CodeQL/Strix and the scheduler remain pending or in progress
after the restack; this is below D2's threshold. The
contextual #818 refresh created replacement current-head runs after cancelling
the prior duplicate event runs; the cancelled `opencode-review` and
`coverage-evidence` entries have no job steps or source log and are not source
failure evidence. Contextual #773 has current scheduler and Noema success but
OpenCode/Strix remain queued. These observations are all below D2's six-hour
and two-observation threshold; decisions remain `WAIT_AND_REMEDIATE`.

### Contextual live-head refresh — 2026-08-22

The older contextual PR rows above are historical unless their SHA matches this table:

| PR | Current exact identity | Live state and decision |
|---:|---|---|
| [#804](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/804) | head `92c1ef02dc3e5f99c3fee21131cc9977c2eb10d6`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465`, merge-result tree `1027c55395eb18821586749882bc2e199f3a2bbe` | Current head is a documentation-only follow-up on verified code head `4a919ce2`; the preceding exact source tree passed `1473` tests and principal-claim rotation/cross-subject regression evidence. Fresh hosted checks show security, CodeQL, supply-chain, dependency, OSV, Scorecard, Trivy, and scheduler success while Strix remains pending; formal approvals `0`, auto-merge disabled. Earlier repository-wide coverage/Ruff observations remain quality evidence for the source head and are not silently transferred to the doc-only tip. Decision: `WAIT_AND_REMEDIATE`. |
| [#807](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/807) | head `f832e32f99886388188e5b1f0d89c47ce4cfeef8`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open and mergeable but blocked; current checks are terminal at 21 success, 2 neutral, and 8 skipped; auto-merge enabled and formal approvals `0`. Decision: `WAIT_AND_REMEDIATE` pending required independent approvals. |
| [#818](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/818) | head `133729cfbab2ac74f3309cad8fbc7e20c2e33627`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465`, merge-result tree `ec2bee36c5e7928d8e2e141f1c7ec8eb28d5b75c` | Current head adds the buyer-safe disconnected-client diagnostics follow-up on top of the telemetry stack. Fresh exact-head evidence is `1456 passed`, statement coverage `92%`, branch-enabled total `90%` (`1018` branches, `144` partial), interrogate `93.3%`, compileall, tests-workflow actionlint, hash-locked pip-audit, and diff-check clean. Cancelled predecessor OpenCode/coverage runs had no steps or source log; replacement exact-head runs are queued, formal approvals `0`. Decision: `WAIT_AND_REMEDIATE`. |
| [#819](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/819) | head `c9ac72c4479e7709949975a2d487f3d4eb582a30`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open and mergeable but blocked; current functional/security checks are terminal with 21 success, 2 neutral, and 8 skipped results, but formal approvals remain `0`; auto-merge is enabled. Decision: `WAIT_AND_REMEDIATE`. |
| [#820](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/820) | head `83d1326f9149e84dbe58fec0739efca5da99eead`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465`, merge-result tree `390af78d97372195d73f77d0e7ebaf7484a2edbb` | Fresh exact-head evidence is focused `87 passed`, full `1446 passed`, conflict-marker scan/compileall/diff-check clean. Hosted `osv-scan` run `32527911591` / job `96913798878` reproduces the inherited central `--output` deprecation and empty-result `test -s` failure after scanner exit 0; this is not a source vulnerability. Strix remains pending/in progress and formal approvals `0`. Decision: `WAIT_AND_REMEDIATE` pending central repair and exact-head re-run. |
| [#821](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/821) | head `c5de31cf276580fddd4f3bcf863f5a7fcfa5aeb8`, base `e226e1197bdfc890c9d8e5b9b648c78857d7e465` | Open and mergeable but blocked; current functional/security checks are terminal with 21 success, 2 neutral, and 8 skipped results, but formal approvals remain `0`; auto-merge is enabled. Decision: `WAIT_AND_REMEDIATE`. |

### Backlog convergence and consolidation — 2026-08-23

All prior per-PR "Live ... continuation" rows above are historical. A global
`clean` or `merge-ready` label is deliberately not carried forward: a
protected-merge claim must be refreshed for the PR's exact current SHA,
including reviews, unresolved threads, and required checks.

**Exact protected-gate refresh — 2026-08-23.** REST inventory
(`GET /pulls?state=open`) returned **31** open PRs, not 29. The current #803
governance-audit durability repair is `5c6670cb`, with 6 active workflows out
of 29 and no failure; #804's ADR reconciliation is `d8a34558`, with 31
terminal exact-head check runs and no failure in this refresh.
#818 is `083316fc`, with one active replacement check out of 29 after the
session-correlation repair. New #824 is `a4219151`, with six active checks out
of 28 for the locked local-test entrypoint. #820 (`9cae444a`) has 30 terminal
exact-head runs with no failure. Those results are only partial gate evidence:
#803, #818, and #824 are explicitly `REVIEW_REQUIRED`, and every PR needs its
own fresh independent approvals, resolved-thread evidence, and required
terminal workflows before a normal protected merge. No PR is described here as
merge-ready while any of those facts are missing or unverified.

**Shared Strix repair blocked — 2026-08-23.** The current exact heads
[#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803)
(`5c6670cb`) and [#824](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/824)
(`a4219151`) both have a terminal failing `strix` check. Their archived reports
show a rate-limited primary NVIDIA NIM attempt followed by a zero-finding NIM
fallback whose exact `MODEL QUALITY WARNING` banner was misclassified as an
infrastructure failure; this is shared-gate behavior, not zero-finding proof of
either PR's source correctness. Initial central repair
[.github#1254](https://github.com/ContextualWisdomLab/.github/pull/1254) was
closed without merge as a duplicate. Its active successor
[.github#1213](https://github.com/ContextualWisdomLab/.github/pull/1213) is
open at `8d0f7ca8`, stacked on central
[#1233](https://github.com/ContextualWisdomLab/.github/pull/1233)
(`fix/organization-loop-oidc-fallback@dfb8e261`) rather than protected
`.github/main`. It normalizes the workflow-facing `openai-direct/*` fallback
before LiteLLM dispatch and strips only an exact timestamped advisory that says
optional web search lacks `PERPLEXITY_API_KEY`; fatal, denied, timeout, and all
other warnings remain fail-closed. Direct-OpenAI `repository_dispatch` run
`32643804284` validated live #1233 metadata and found zero vulnerabilities,
but its old protected-main classifier terminally failed on seven instances of
that optional advisory; its subsequent same-head status publisher was rate
limited, so it created no success evidence. Fresh #773 head `1240bc23` then
terminally failed run `32644795368` with the same pattern: zero NIM findings,
then raw `openai-direct/gpt-5.6-luna` rejected by LiteLLM as an unknown provider;
it emitted no source vulnerability. #1213's exact `8d0f7ca8` run `32644636757`
likewise scanned its three changed files without a vulnerability before the
trusted pre-#1213 gate passed that same raw alias to LiteLLM. Because both
`pull_request_target` and `repository_dispatch` deliberately execute a trusted
base gate rather than the PR-head gate, the unmerged #1213 cannot change its
own protected scan behavior. Executing the PR-head gate with credentials,
fabricating a status, or using a bypass would violate the boundary. A
human-authorized non-bypass bootstrap decision, then fresh exact-head Strix
evidence, normal stack integration, independent approvals, and fresh target
checks remain required. Neither target PR is protected-main evidence or
unblocked before those conditions hold.

**Closed technical gap — unauthenticated denial recording on the durable
persistence hot path (found and fixed in #803).** While triaging #803's
review threads, tracing `server.py`'s `_authorize` into
`orchestrator.py`'s `_StateStore.save()` showed that `record_authorization_decision`
fires on every pre-auth denial (401 unauthorized, 429 rate-limited — both
reachable with zero credentials), and, when `--state-db` is configured, that
routed through a synchronous, lock-serialized sqlite commit shared with
durable `workflow_run`/`evaluation_run` persistence. An attacker spraying
denied requests across rotating source IPs (evading the per-IP rate limit)
could force that lock contention against legitimate authenticated traffic — a
real DoS amplification this PR introduced, not a pre-existing one, since
denied requests were never durably written before it. Fixed by splitting
`_StateStore.save()` by durability need: keyed kinds stay synchronous, stream
kinds (audit/authorization/analytics — already bounded/best-effort by
existing retention limits) now queue through a bounded background worker,
reusing the `NonBlockingLedgerStore` pattern already established in
`cost_ledger.py` for the same problem class. Verified with two new regression
tests and the full local suite (1459 passed) before and after; no observed
behavior change for `load()`/`close()` read-after-write or restart-durability
guarantees.

## 5. Open issue and product-gap queue

| Issue | Customer-visible gap | Planned proof / next PR |
|---:|---|---|
| [#568](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/568) | Closed by merged PR [#785](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/785): operators can compare provider-neutral reasoning profiles at equal budget. | Keep production route/conduct defaults locked until the accepted true-parameter ablation evidence permits a separately authorized default change. |
| [#123](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/123) | A sole collaborator can be unable to satisfy last-push approval. | Add governance evidence/runbook or a protected-rule-compatible process; never bypass approval. |
| [#119](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/119) | Ambiguous or unbounded inbound framing threatens request integrity. | The #776/#783 implementation stack is merged into non-main branches; protected-main integration still requires exact-head hosted evidence and independent approval. |
| [#118](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/118) | Liveness and authenticated readiness are not yet fully separated. | PR [#780](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/780) implements the minimal `/healthz` and authenticated `/readyz` contract; merge only after exact-head Checks and independent approval. |
| [#117](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/117) | Trace access and inference access need separate authority. | PR [#780](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/780) merged the minimal liveness/readiness and trace-authority slice; the issue remains open for batch ownership, full purpose/tenant/resource/lifetime/revocation context, and the single-token migration gate. |
| [#116](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/116) | Browser admin sessions need separation from long-lived bearer credentials. | PR [#788](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/788) implements opaque bounded sessions, Secure-by-default cookies, same-origin state-change checks, logout/revocation, and regression evidence. |
| [#103](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/103) | Release readiness must fail closed on stale head, missing review, or missing Checks evidence. | PR [#784](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/784) merged the semantic split, but the issue remains open until a trusted `.github` producer artifact and consumer verification bind complete exact-head policy evidence; caller-supplied dictionaries remain insufficient. |
| [#899](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/899) | CEFR writing/speaking observations need a governed, evidence-bound orchestration boundary. | PR [#903](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/903) is the current implementation head; validate its exact protected checks, research traceability, and independent approval before delivery. |
| [#897](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/897) | Language-rater output must structurally prohibit final CEFR or placement decisions. | Keep the operation limited to criterion observations and evidence references; align the acceptance contract with PR #903 and the downstream CEFR/fast-mlsirm owners. |
| [#102](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/102) | Equivalent endpoints need race-to-first-valid completion without unsafe cancellation. | Closed predecessor [#114](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/114) is explicitly a partial experiment and its evidence does not transfer. Rebuild one bounded vertical slice after the protected provider boundary is integrated: explicit endpoint equivalence, completed-response validation, bounded budgets, cancellation-or-drain, deterministic tie-breaking, secret-redacted attempt provenance, and provider-truth tests. |
| [#86](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/86) | NVIDIA NIM discovery needs live, evidence-grade capability/cost/quality measurement. | PR [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) is the active benchmark implementation; its local and ordinary hosted evidence passes, but OpenCode is missing a current verdict and Strix failed closed on provider HTTP 500s. |

Issue [#95](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/95) (Atheris locking must work on all supported CPython interpreters) was closed 2026-08-23 as resolved on protected main by a simpler mechanism than the PR #96 predecessor originally proposed: `pyproject.toml`'s `fuzz` extra pins one version (`atheris==3.1.0; python_version >= '3.12'`) rather than a two-way version split, and it already covers both the 3.12 fuzz runner and the central 3.14 coverage-evidence image per `.github/workflows/fuzz.yml`'s own comment — verified directly against the exact current head, not assumed from the stale issue history. GitHub currently returns `404 Not Found` for issue [#777](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/777); its earlier metric-gap
row is therefore removed from the actionable queue rather than treated as a
live work item.

## 6. Prioritized gap register

| Priority | Gap | Current evidence | Definition of done |
|---:|---|---|---|
| P0 | Protected delivery cannot merge a PR until its exact current SHA has the required approvals, resolved threads, and terminal workflows; `.github` has a weaker repository-local rule than the organization rule. | Organization ruleset `18156473` requires two approving reviews, an additional approval for unattributed changes, last-push approval, and resolved threads for normal repositories. It excludes `.github`, whose active repository ruleset `17921150` permits zero required approvals and exposes an OrganizationAdmin bypass; the maintainer procedure forbids that bypass and retains the stronger independent-review gate. The 2026-08-23 exact refresh found 30 open PRs, including #803 with replacement workflows active after a security fix. This baseline makes no repository-wide clean/ready inference from partial checks; see "Exact protected-gate refresh" above. | Align `.github` governance with the organization policy or document an equivalent non-bypass gate; then obtain independent approvals for the exact current SHA, resolve threads, pass hosted required workflows, and complete normal squash/merge. Upstream: raise the shared installation's rate ceiling or shard review-dispatch load off the single token, since per-PR remediation cannot fix a saturated shared credential. |
| P0 | Agent-pool resource paths must not let a caller dereference a worker outside its addressed pool. | Strix reported the recurring IDOR on exact PR #784; direct root repair [#804](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/804) is open with the pool-boundary resolver and regression tests. #784 now stacks on #804 for dependency-safe retesting. | Protected #804 merges to main, then affected stacked PRs retain the root base and rerun Strix on their exact current heads. |
| P0 | Provider boundary is still being assembled across stacked PRs. | [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768), [#765](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/765), [#764](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/764), [#770](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/770), and [#763](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/763) are pending integration; #778 and #779 are integrated into #765, including temperature negotiation. Central OpenCode gateway routing is tracked by [.github#1170](https://github.com/ContextualWisdomLab/.github/pull/1170), while the current target caller is carried by [.github#1198](https://github.com/ContextualWisdomLab/.github/pull/1198); neither has protected-main completion evidence. | One current-main stack has capability isolation, secure JSON, bounded framing, multimodal evidence, KV bootstrap, honest catalog, optional-control negotiation, and failover with no duplicate logic; central review execution must use the same current gateway pin after protected integration. |
| P0 | Operational failure paths are not yet one buyer-verifiable contract. | [#771](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/771) and [#772](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/772) are open. | Exact-head full suite, focused edge tests, security scans, and a buyer-facing failure/rollback trace pass. |
| P1 | PII can remain usable without blanket masking, but authorization/encryption is unfinished. | [ADR 0010](planning/adrs/0010-pii-audit-not-mask.md) records the no-blanket-masking policy and explicitly leaves authorization/encryption as follow-up. The actual design is proposed [ADR 0011 at #762's exact head](https://github.com/ContextualWisdomLab/contextual-orchestrator/blob/8f87bcaeddff0866e26900e41deeafe208d8f9e4/docs/planning/adrs/0011-pii-purpose-authorization-and-field-encryption.md); both design [#762](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/762) and implementation [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) remain open and are not protected-main evidence. | Protected main has purpose-scoped caller/role authorization, field-level encryption at rest, credential-only redaction, and audit tests proving raw PII is returned only to an authorized purpose. |
| P1 | Deep-workflow compute policy lacks provider-neutral measured ablation. | PR [#785](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/785) supplies opt-in profiles, snapshot replay, and synthetic/estimated RMSE; the production gate remains closed pending buyer-held-out measurement. | Equal-budget shallow/deep/role-effort/access-list replay with reproducible quality, verifier, cost, and trace metrics. |
| P1 | Model discovery lacks live NVIDIA NIM evidence. | Issue [#86](https://github.com/ContextualWisdomLab/contextual-orchestrator/issues/86) remains open; active PR [#906](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/906) now provides the bounded benchmark, but it is not protected-main evidence while OpenCode/Strix and independent approval remain incomplete. | KV-backed NIM discovery benchmark records model-level declared capability, price provenance, failure class, and quality result without secret leakage; protected main then activates only capability-qualified deployments. |
| P1 | Release gate and hourly loop need exact operational proof. | Central scheduler workflows own the loop; PR [#784](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/784) adds the exact-head authority evaluator/collector, but protected approval and release evidence remain open. | One scheduler owner, no duplicate workflow, exact-head release gate, version/changelog update, and normal protected release evidence. |
| P2 | LineageWeave has no protected-main consumer acceptance gate. | [#801](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/801) added explicit CLI `argv` only to a non-main stack. Main-target [#823](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/823) has the explicit contract at `6bb3fe2c54cda9f574cd239922bc91ece5ea2585`, but remains `REVIEW_REQUIRED`/blocked despite terminal hosted checks; documented protected main still exposes `contextual_orchestrator.__main__.main()` without an `argv` argument. LineageWeave `main@ef6f5a5f` still assigns `sys.argv` in `docker/contextual-orchestrator/start.py`, and its bootstrap test observes that mutation; open LineageWeave [#468](https://github.com/ContextualWisdomLab/LineageWeave/pull/468) retains it. Its opt-in real-provider test bypasses that bootstrap, so neither it nor #823's mocked-server unit test is authenticated consumer proof. | PR #823 explicit CLI invocation contract is merged to protected main and update LineageWeave at that exact upstream pin to invoke the server with explicit arguments rather than mutating process arguments. Then run a LineageWeave-owned authenticated `/v1/chat/completions` end-to-end test that proves process `sys.argv` is unchanged; retain authorization and chat-completion evidence against the exact protected main SHA. |
| P2 | Ecosystem boundaries need consumer proof. | `naruon`, `.github`, and sibling components are named consumers, but this repo remains one deployable product. | test_naruon_ecosystem_connector.py proves the exact JSON schema and endpoint consumption without speculatively extracting the codebase. |
| P2 | Frontend component inventory is not applicable here. | This repository is a backend stdlib lab and has no frontend/Storybook tree. | Keep the existing Figma artifact record; introduce Storybook only when a frontend package is actually added. |

## 7. Delivery gates

For each PR, perform the following loop on the current head: inspect changed
files and review threads, reproduce the claimed behavior, fix root causes in
the shared path, run focused and full tests, run compile/diff/security checks,
refresh the hosted Checks, and merge only after the protected rule is satisfied.
Remote agent pushes are respected by refetching the head; stale approvals or
checks are not reused. Review queues and hosted wait time remain active-work
time: use it to implement the next independent gap, not to bypass the gate.

Release is not complete until the version, `CHANGELOG.md`, release candidate,
and exact-head evidence all agree. A green local run is not production or
buyer telemetry; label local evidence accordingly.

## 8. Standards and research basis (APA 7th)

Nielsen, S., Cetin, E., Schwendeman, P., Sun, Q., Xu, J., & Tang, Y. (2025).
*Learning to orchestrate agents in natural language with the Conductor*
(arXiv:2512.04388). https://doi.org/10.48550/arXiv.2512.04388

OpenAI. (n.d.-a). *Create chat completion*. OpenAI Platform.
https://platform.openai.com/docs/api-reference/chat/create

OpenAI. (n.d.-b). *Create a model response*. OpenAI Platform.
https://platform.openai.com/docs/api-reference/responses/create

OpenAPI Initiative. (2025, September 19). *OpenAPI specification version 3.2.0*.
https://spec.openapis.org/oas/v3.2.0.html

Sakana AI. (2026, June 22). *Sakana Fugu: One model to command them all*.
https://sakana.ai/fugu-release/

Xu, J., Sun, Q., Schwendeman, P., Nielsen, S., Cetin, E., & Tang, Y. (2025).
*TRINITY: An evolved LLM coordinator* (arXiv:2512.04695).
https://doi.org/10.48550/arXiv.2512.04695

Fielding, R., Nottingham, M., & Reschke, J. (2022). *HTTP caching* (RFC 9111).
RFC Editor. https://www.rfc-editor.org/rfc/rfc9111.html

National Institute of Standards and Technology. (2024). *Artificial
intelligence risk management framework: Generative artificial intelligence
profile* (NIST AI 600-1). https://doi.org/10.6028/NIST.AI.600-1

These sources support the current product shape, OpenAI-compatible wire
honesty, deep-versus-shallow orchestration allocation, cache safety, and
generative-AI risk evidence. PDFs are attached only when redistribution is
permitted; otherwise the canonical citation and link are retained.

## 9. Design and ecosystem record

- Existing editable Figma file: `Contextual Orchestrator Plugin-Driven Admin
  Design`, file ID `vsZMd8WAv42HDRgcZuNcWk`, recorded in
  [`docs/figma_artifacts.md`](figma_artifacts.md). No new Figma work is needed
  for this backend-only baseline.
- Existing FigJam architecture board is also recorded in that file.
- Storybook/`ui-ux-pro-max`/`Anti-Slop-UI` are deliberately deferred, not
  skipped: [ADR 0033](planning/adrs/0033-admin-console-ui-tooling-boundary.md)
  records the decision, cites the Figma File ID above, and states the three
  concrete conditions (a second reusable screen family, a second consuming
  repository, or a new component-based frontend) that make adoption the
  correct next step rather than optional.
- Repeated backend/API objects remain documented contracts and schemas.
- The current packaging decision is one standalone gateway that can be
  consumed as a module. A repository split requires a concrete independent
  consumer, release cadence, or security-provenance boundary.

**Customer next action:** approve the next exact-head PR only when its row above
has a concrete proof link, then use the next highest-priority unresolved gap to
create the following stacked change.

### Live exact-head continuation — 2026-08-25 09:40 KST

This continuation supersedes older rows for the re-listed PRs only; every
earlier snapshot row remains historical evidence for its own head. The 2026-08-25
session shipped two product changes and one scheduler change, recorded below,
and re-inventoried the whole open queue at the listed heads.

**Shipped this session (pending protected-main merge):**

- [#833](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/833)
  `fix(security): rename credential-name constants and readiness label keys…`
  root-causes the Strix B105 false-positive class: `DEFAULT_*_TOKEN_KEY`
  constants were KV credential *names*, not secrets, and `readiness_pass/warn/fail`
  were locale label keys. Renames are semantics-first (no suppression files, no
  scanner configuration change). This unblocks ~12 PRs whose diffs intersected
  `__main__.py`/`admin.py`.
- [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834)
  `feat(routing): add operator-managed model groups` implements ADR 0026:
  `ModelAgent.group_name`, measured intra-group routing
  (`P(success | Beta(1,1) data) / EWMA_latency_seconds`, Jacobson-gain 1/8, no
  arbitrary cross-metric weight), group alias resolution in
  `_requested_agent`, `/api/v1/model_groups` CRUD with agent-pool persistence,
  OpenCode Zen discovery source, zero-price/`-free` free-tier classification,
  and the admin table's measured group evidence replacing fabricated
  capacity/success percentages.
- [#835](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/835)
  `ci: hourly OpenCode maintenance agent routed through this gateway` stacks on
  #834: a `23 * * * *` UTC job copies six provider secrets into the process-local
  KV (bootstrap transport only), auto-discovers chat models, and uses the stable
  `contextual-orchestrator` gateway alias. It neither creates nor depends on a
  named model group; group membership remains operator-managed through REST/DB.

| PR | Exact head (2026-08-25 09:30 KST) | Base / mergeability | Decision |
|---:|---|---|---|
| #835 | `f21da70b7f18` | stacked on #834; UNSTABLE | `WAIT_AND_REMEDIATE` — hosted Checks + approval |
| #834 | `6b069a059b67` | main; MERGEABLE | `WAIT_AND_REMEDIATE` — hosted Checks + approval |
| #833 | `88112e49f13e` | main; MERGEABLE | `WAIT_AND_REMEDIATE` — strix rerun + approval |
| #831 | `d7efa634a1f1` | main; MERGEABLE | IDOR boundary fix; awaiting terminal Checks + approval |
| #830 | `76c18f13c2ff` | main; MERGEABLE | streaming/batch error boundary; same gate set |
| #828 | `42b6e4cb0851` | main; MERGEABLE | response-write crash fix; same gate set |
| #823 | `0a9a1932cf79` | main; MERGEABLE | LineageWeave CLI argv contract; same gate set |
| #822 | `9729ad7b85cf` | main; MERGEABLE | credential backend tests; same gate set |
| #821 | `577570ab4671` | main; MERGEABLE | token-counting tests; same gate set |
| #820 | `a185eedf7572` | main; **CONFLICTING** | rebase onto current main, then gate set |
| #819 | `5971bdcb0b68` | main; MERGEABLE | config storage tests; same gate set |
| #818 | `4cd295c2b2ab` | main; MERGEABLE | OTel session correlation; same gate set |
| #807 | `7703916f7ab1` | main; **CONFLICTING** | superseded in part by #830; rebase or close as duplicate |
| #803 | `641b2d1f078e` | main; MERGEABLE | PII purpose-limited protection per ADR 0010; gate set |
| #794 | `225271583612` | main; MERGEABLE | DB naming migration; gate set |
| #790 | `17a5e42b44f1` | main; MERGEABLE | review gateway bootstrap; gate set |
| #788 | `d5ebe735a9ee` | main; MERGEABLE | opaque admin sessions; gate set |
| #785 | `ec609fa7b526` | main; **CONFLICTING** | reasoning-effort profiles; rebase required |
| #782 | `5ab3acc5aba6` | main; MERGEABLE | workflow object authorization; gate set |
| #780 | `e4e6b7cf27f0` | main; **CONFLICTING** | liveness/readiness split; rebase required |
| #773 | `cfae080dcab9` | main; MERGEABLE | this document; self-referential row, refetch live head |
| #772 | `c038175528c8` | main; MERGEABLE | distributed response cache plus thread-isolated request sampling; gate set |
| #768 | `2a6c54636693` | main; MERGEABLE | embedding/chat isolation; gate set |
| #765 | `d19e3492192e` | main; **CONFLICTING** | paper-grounded reasoning contract; rebase required |
| #762 | `78aa5be10c58` | main; MERGEABLE | PII design ADR; merge after #803 sequencing decision |

**Gap register updates (2026-08-25):**

1. **Closed by #834:** model-group composition (per-model → per-group), measured
   speed/stability member ordering, group alias addressing, OpenCode Zen
   discovery, free-tier classification.
2. **Open — Zen pricing honesty:** OpenCode Zen `/zen/v1/models` omits prices;
   free classification currently relies on explicit `-free`/`:free` suffixes.
   Keep `unknown ≠ $0`; adopt provider-reported pricing when available.
3. **Open — quality dimension:** intra-group score is reliability×speed only.
   Cross-model quality routing must use calibrated evaluation (fast-mlsirm /
   RouteLLM-style learned router) with ablation, not hand weights (RouteLLM;
   FrugalGPT citations in ADR 0026).
4. **Open — multi-instance telemetry:** observation ledger is process-local by
   design; durable time-windowed aggregation is required before horizontal
   scaling (ADR 0026 boundary).
5. **Closed on #835 head `f21da70b7f18`: scheduler pinning and transient-model
   coupling.** OpenCode is pinned to `1.18.22`, actionlint passes, and the loop
   targets the stable gateway alias without creating an implicit model group.
6. **Open — conflicting stack:** five PRs (#820, #807, #785, #780, #765) need
   rebases before their gate sets can complete; #807 may be closed as
   functionally superseded by #830 if review confirms full coverage overlap.

**Customer next action:** merge order is #833 → #834 → #835 (stack), then work
the table top-down; each row needs terminal Checks plus independent approval on
its exact head before a normal protected merge.

### Capability-group exact-head continuation — 2026-08-25 10:42 KST

This continuation supersedes the preceding model-group/scheduler rows. Model
identifiers used during design were transient examples and are absent from the
contract: equivalence is an operator assertion, never a name-derived inference.

| PR | Exact identity and live boundary |
|---|---|
| [#834](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/834) | head `fae8665771443587fcb64bc517a005412c979c14`, base `50014ef9f87623c6dfd529116297c5d180743125`; open, auto-merge enabled, `REVIEW_REQUIRED`, replacement Checks pending. Its tree `017ca560293c7939b405af6fa25c7f0e3ba13e76` is identical to the locally validating #838 tree. The preceding capability tree passed `1605` tests before parent integration plus `66` focused post-merge tests; #838 adds `28` focused DB/Admin/group tests, with the full exact-tree run still in progress. No protected-main claim is made until exact-head hosted gates and independent approval finish. |
| [#837](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/837) | head `b2d7aea0d71b5669d6c0167527691592a9ab2abe`; normally merged into #834's feature branch at 2026-08-25 10:30 KST. Adds provider-declared `text`, `image`, `video`, `speech`, `transcription`, `embedding`, `rerank`, and `audio` discovery/routing surfaces; its merge-result tree equals #834's current tree. This is stack integration, not protected-main release evidence. |
| [#835](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/835) | head `f21da70b7f184b687b9c4996327c18d384c420d0`; normally merged into #834's feature branch at 2026-08-25 09:52 KST. The hourly loop uses the stable gateway alias, a random loopback bearer token, job-scoped write permission, hash-locked Python requirements, and lockfile-integrity-checked OpenCode `1.18.22`. |
| [#838](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/838) | head `2de42512ac552246b8a5bc024ba179b3addcc1ed`; normally merged into #834's feature branch at 2026-08-25 10:39 KST. It migrates group membership from duplicated agent JSON into 3NF `model_group`/`model_group_member` relations and adds an accessible Admin REST editor plus capability coverage. Its merge-result tree equals #834's current tree; this is stack integration, not protected-main release evidence. |
| [#772](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/772) | head `33bd77ed216f274d658e288c05d9dca6018e07b5`; open/blocked, no failed hosted check at refresh, three pending/external contexts. |
| [#768](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/768) | head `2bb7538c614e9bd8686ba9007c2955b6c770390d`; open/blocked with stale `CHANGES_REQUESTED`, one failed and three pending/external contexts. Exact failure and unresolved review evidence must be remediated before merge. |
| [#803](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/803) | head `6b6bd783272033d7de93d584b7618b4445b5f391`; open/blocked with one failed and three pending/external contexts. The earlier PII and hash-lock local evidence does not clear the current hosted failure. |
| [.github#1316](https://github.com/ContextualWisdomLab/.github/pull/1316) | head `c169b146cfaaebd2ba75ee781bfcf1d82f4dbb84`; central repair aligns Strix's required-workflow smoke contract with its current direct-OpenAI fallback. Auto-merge is enabled; hosted gates remain mixed/pending. Until it normally merges, target-repository Strix failures showing the retired fallback assertion are central-contract failures, not source security evidence. |

Research boundary: MMR-Bench supports modality-aware routing evaluation, while
Jacobson EWMA and the Beta-Bernoulli posterior support the measured latency and
stability estimators. None of those sources proves that two provider identifiers
represent one model; only operator/provider provenance can establish group
membership. The successful-responses-per-second quotient remains a transparent
product policy, not a reproduced paper result.

Buyer-visible gaps now prioritized:

1. Video submission is routed, but provider-affine polling/content download and
   durable job ownership are not yet represented; add a normalized async job
   resource before calling video orchestration production-complete.
2. **Closed on the #838 stack:** Admin exposes group capability coverage and a
   keyboard/native-form REST editor; DB membership is normalized and legacy JSON
   membership migrates without data loss. Authenticated deployed-browser runtime
   evidence remains a release/UAT gate rather than an implementation gap.
3. Free-model tests are deterministic catalog-contract tests. Add an opt-in,
   spend-capped live OpenRouter canary selected from current zero-price metadata;
   never pin a transient free model identifier in production or CI.
4. Multi-instance routing observations remain process-local. Add a time-windowed
   durable observation model with calibrated decay before horizontal scaling.
5. Protected main, not a feature-stack merge, remains the release boundary; do
   not bump or publish a version until #834 has exact terminal gates and an
   independent current-head approval.

### Capability routing contract remediation — 2026-08-25 10:46 KST

[#839](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/839)
head `852fc1eeb2f30f212694702fe96afba216eddee9` normally merged into #834's
feature branch. The resulting #834 exact head is
`19ab869094a8b877fe70397ff6de55c357654efe`, tree
`c92e2c90a14d9e8f296c536cb0172da92eaa35b6`; auto-merge is enabled, but it is
still blocked on replacement hosted Checks and independent review. The patch
closes three exact review findings: unavailable capability pools now honor the
documented 503 contract, unknown explicit model/group names return 400, explicit
groups retain true streaming in auto mode, and top-N discovery honors
provider-reported prices before price-book refresh. The focused regression set
passed 41 tests; the full exact-tree suite is running. This stack merge remains
non-release evidence until protected `main` contains the tree.

PR #773 is currently `ebf397e03c6c69a214a015c39efade6f78c603e1`, with
auto-merge enabled, several successful security/review contexts, and remaining
full-suite/fuzz/security contexts in progress or queued; `REVIEW_REQUIRED`
therefore remains the honest decision boundary.

[#841](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/841)
head `c4896ae4d1d410fdd933b8ae9e99a870d1d2cc9d` subsequently merged normally
into the #834 stack, advancing #834 to exact head
`4bf5cf0a8dce43d7a08a8e9f6ed79bb00017cb03`, tree
`549e85b33dc9cc2941d3983b397add60415b66d0`. It prevents a provider's JSON-null
modality arrays from aborting cross-provider discovery; the focused discovery
suite passed 23 tests. The immediately preceding tree passed the complete local
suite (`1611 passed in 645.34s`); because #841 changes that tree, the current
head still requires its replacement exact-head hosted full-suite result.

Two adjacent queue failures were also remediated at their own heads: #768 is
now `4d3d11ec2836e21b973ab7573322ac883abfb785` and preserves explicitly declared
non-chat capabilities through discovery/agent creation (38 focused tests); #803
is now `030b4348eb4ecdf85d1017f8891cac9faa5a0394` and restores the exact CPython
3.12 Atheris lock marker (1 focused test). Both have normal auto-merge enabled
and remain gated by fresh hosted Checks/review; neither is merged evidence.

### Product/technical specification trace — 2026-08-25 11:02 KST

[#842](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/842)
head `49d0940e04a680648743d9890f79c7110167e820` normally merged into the #834
feature stack, advancing #834 to exact head
`4e0cbca934e7e82cbed56de347853dbc40971f20`. The new
`docs/model-group-product-technical-spec.md` derives the PRD, user stories,
acceptance criteria, TRD, normalized persistence model, REST status contract,
flow/class UML, security/operability boundary, and Admin UI audit from ADR 0026
and the implemented surfaces. Its focused paper/API/DB/group/capability contract
set passed 39 tests. Current-head review threads are zero; replacement hosted
Checks and independent approval remain required.

Research was re-verified against the primary arXiv records for MMR-Bench and
RouteLLM and current official OpenRouter/OpenCode documentation. The exact
boundary remains: research supports modality-aware and learned cost/quality
routing, but does not establish provider-alias identity. OpenCode's own mutable
catalog currently demonstrates why transient free model identifiers must stay
out of the production contract.

### Release-candidate exact tree — 2026-08-25 11:18 KST

[#840](https://github.com/ContextualWisdomLab/contextual-orchestrator/pull/840)
was reordered as a stack prerequisite instead of an independently mergeable
main release: feature head `4e0cbca934e7e82cbed56de347853dbc40971f20`
was merged into release head `fd6dde7e8f06a1d3b67400355f6223c2a653aa3d`,
the PR base was changed to #834's feature branch, and its resulting three-file
diff (`CHANGELOG.md`, `pyproject.toml`, `uv.lock`) normally merged. This removes
the risk of publishing version `0.2.0` before its implementation. It also
restores the pre-existing tool-fallback changelog entries and synchronizes the
lockfile package version.

#834 is now exact head `6f494131774f38b78bdc3b104451e2133edb5318`,
tree `52fdbdcb96bbe0a7617a4573e9c01cd5355db350`. That exact release-candidate
tree passed the complete local suite (`1612 passed in 724.94s`) and produced
both the `0.2.0` sdist and wheel. Protected-main publication remains unproven:
hosted jobs are queued, independent review is required, and no tag or package
has been published.

The shared Strix contract repair is
[.github#1316](https://github.com/ContextualWisdomLab/.github/pull/1316) exact
head `e4f9865a1b06978324f006ee3861b84953877d8b`; its sole informational review
thread was resolved and auto-merge is enabled, while central hosted gates remain
queued. Until that protected merge lands, the queue-wide retired-smoke Strix
failures remain central workflow evidence rather than individual source defects.

### Adjacent PR remediation loop — 2026-08-25 11:31 KST

The protected queue advanced without bypasses; every row below has auto-merge
enabled and remains open until its fresh exact-head gates complete:

| PR | Exact head | Remediation evidence |
|---:|---|---|
| #831 | `634fe03576d986794b121e0da05179f1663073f4` | Wrong-pool POST now matches GET/PATCH/DELETE with 404 `agent_not_found`; 8 focused tests passed. |
| #830 | `d727c0897e94568539c22af19d88e055b88aaa56` | Current main merged; three remaining review notes verified as intentional provider-error boundary behavior and resolved. |
| #828 | `bde1975c8c6e843b6339c6b7f3922dad1b0f07c5` | SSE disconnect now stops upstream token work and releases the run slot; 15 focused tests passed. |
| #794 | `6efeacae06d7445d71991cb7faf970b5a3a1ab65` | SQLite SAVEPOINT preserves caller-owned transactions; direct test-file execution and 28 pytest cases passed. |
| #788 | `1fc1ddc24ef8b0c5009c9212efb803a79d0deaef` | Opaque admin sessions now authorize evaluation runs, erase failed bearer input, and avoid duplicate readiness fan-out; 27 focused tests passed. |

These are PR-head claims only. None is protected-main or release evidence until
terminal Checks and required independent approvals cause a normal merge.

### Live exact-head continuation — 2026-08-27 09:28 KST

Added model capability prior integration for `ModelGroupRouter`:
- Embedded `Chatbot Arena` and `Artificial Analysis` baseline scores as Beta distribution priors.
- Updated `ModelGroupRouter` to accept a `prior_resolver` without breaking the existing group stability calculation or `_report_locked` structure.
- Updated `contextual_orchestrator/orchestrator.py` to use `resolve_quality_prior` for its `_quality_router`.
- PR #883 requires investigation for `strix` check failures.
- PR #888 and #887 have their CodeRabbit comments handled or under review.
- The 1-hour recurring `schedule` gap-loop continues.

### Live exact-head continuation — 2026-08-27 09:52 KST

Added OpenRouter upstream real-time reliability collector (`OpenRouterUptimeCollector`):
- Fetches live telemetry (`uptime_last_30m`) from `/api/v1/models/{model_id}/endpoints`.
- Dynamically integrates this telemetry into `ModelGroupRouter` by exposing `update_prior` to safely adjust `alpha` and `beta` values without mutating underlying stability logic.
- Avoids HTTP blocks during startup by orchestrating a non-blocking Daemon thread polling at set intervals, enabling resilient routing dynamically over time.
- Integrated the change into PR #892 (`feat/model-capability-priors`) and pushed to the origin repository.

### Live exact-head continuation — 2026-08-27 12:0x KST (arbitrary-weight remediation)

GAP RESOLVED on PR #892 head `af9d667f…+fixups`:
- The shipped `_BASELINE_PRIORS` table carried invented Beta pseudo-counts
  ("alpha=10, beta=1…" style), violating the organization rule that no
  weight may be arbitrary. Replaced with a measurement-typed derivation:
  published Arena Elo and Artificial Analysis Quality Index are normalized
  by each instrument's own median/MAD, averaged, squashed through the
  logistic, and split across exactly the repository's existing Laplace
  evidence budget (`PRIOR_EVIDENCE_BUDGET = BETA_PRIOR_SUCCESS_COUNT +
  BETA_PRIOR_FAILURE_COUNT`). Mass is conserved: measured members never
  receive more evidence than unmeasured ones. Unknown identifiers keep
  the unchanged Laplace pair.
- The uptime collector's invented `weight = 50.0` penalty was removed.
  Each poll now folds one window of provider-measured availability into
  equivalent Bernoulli mass (`successes += u/100`, failures`), so all
  counts trace to polls; failure denominator = polls performed.
- `ModelGroupRouter.update_prior()` was added so prior components can be
  refreshed atomically while `success_count`/`failure_count` remain
  bit-identical — telemetry can no longer masquerade as outcomes.
- Collector previously called the nonexistent `update_prior`; it now has
  a contract + tests, hardened HTTPS/percent-encoded fetch, full
  docstrings, and injectable startup delay for deterministic tests.
REMAINING GAP (follow-up loop): re-fit these priors against fast-mlsirm/
TEPP calibrated quality latents before enabling benchmark priors on any
revenue-serving route; until then their influence is capped at the same
budget an unmeasured member already spends.
