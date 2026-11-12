# Performance migration acceptance audit

The original request is the 16-section performance-migration specification in
`/home/omz/.codex/attachments/71142196-41fb-4b92-bbfd-5fa7c39e11b7/pasted-text-1.txt`.
This audit preserves its complete scope. Performance-migration acceptance is
**complete**, supported by terminal jobs and inspected artifacts on 2026-10-14.
This is implementation/performance acceptance, not learning-protocol approval.
Artifact paths in this document are relative to `artefacts/performance-migration`.

| Requirement | Evidence inspected | Status / limit |
|---|---|---|
| 1. Faster persistent-fly training; independent batched accounting | `20261014-event-v1-sustained/batch-{1,8}.json`; current repeated/full CLI results in `20261014-final-v1` | Verified: sustained B and current full C; current median single-fly CLI gain 36.408×; all 109 final native checkpoints audited |
| 2. Repository inspection; preserve existing changes; no competing GPU jobs | Work record; preserved dirty worktree; serial suite commands/PIDs; explicit source-change checks in A/B/D harnesses | Inspected; no stashing/reverting/unrelated process termination |
| 3. Preserve anatomy, mappings, reinforcement, learning equations, terminal/reset order, legal actions, independent RNG, duration/dt, safety, zero external parameters | `execution.py`, `plastic_backend.py`, native kernels, training factory; full-state comparison and component identities; frozen full transitions | Source-complete 100 ms / 0.1 ms / FP32 primary workload preserved; versioned reduction-order differences disclosed |
| 4. A–E harnesses; warmup/repetitions/snapshots; widths 1/2/4/8/16; aggregate/per-learner rates, latency/memory/identities; CUDA events/timeline; genuine closed loop | `benchmarks/{migration,neural}.py`, `run_performance_suite.py`, `benchmark_frozen_workloads.py`, preparation qualification, separate timeline | Verified: all measurement scopes completed; startup stage and process timing distinguished; short and sustained windows labeled |
| 5. Explicit factory for initialization/layout/state/reset/decision/efficacy/readouts/recording/snapshot; same selection in training/evaluation/calibration; separate topology/implementation identities; no silent fallback | `execution.py`, `graph_backend.py`, `learning/config.py`, `analysis/evidence.py`, `analysis/preflight.py`; execution/native tests | Implemented all nine explicit modes; reference remains default; graphs explicitly reject detailed event/component recording |
| 6. Reused buffers/generators/indices/metadata; reduced transfers; GPU eligibility and whole-state health; retained detailed diagnostics | Native buffer reuse tests; lazy owned readout tests; injected silent-neuron NaN tests; measured memory ownership changes | Verified: targeted, full CPU and GPU regression passed; same memory and final learned hashes across restored sustained repetitions |
| 7. Checked CSR32; COO/CSR, SpMV/SpMM, dense layouts, algorithms, descriptors/workspaces/preprocessing; plastic separation/canonical hashes; whole-training selection | `20261014-final-v1/{sparse-layouts,cusparse-algorithms}.json`, `20261012-cusparse-training`, execution/native tests | 92 production cuSPARSE probe cases measured without errors; selected event mode from complete training evidence |
| 8. Optional packaged C++/CUDA extension; supported toolchain, installation/errors/reference; fused FP32 updates/signs/delay/reset/refractory/counting, conservative math | Native/sparse/event build helpers, `native/*.cu` and `native/*.cpp`, `pyproject.toml`, `20261014-public-audit-v2/installed-native-qualification.json`, final native tests, guide | Verified: offline wheel installation outside checkout; all three extensions freshly compile/load from installed sources; all 137 packaged Python/native sources match worktree; GPU tests passed |
| 9. Whole decision and block graphs; stable addresses, sensory/efficacy/reset/delay/RNG/restore/shape changes; reordered/independent learners; no future-game speculation | Parametrized graph replay tests; native-v2/full/block real qualifications; `20261014-independent-event-v1`; current 100-step replay/resume | Qualified; measured lower block capture cost; distinct learner RNG streams retained |
| 10. Reprofile faster runtime; bounded ordered logging/errors/barriers/interruption/safety; dashboard/action history; bounded episode statistics; GPU health | `buffered.py`, `learning/cli.py`, trainer/history code, tests; current timeline; real interrupt; CLI artifact verifier | Verified: full CLI completed with dashboard/action history and checkpoints; artifact audits and current regression passed |
| 11. Consistent immutable checkpoints; profile capture/transfer/serialization/hash; atomic commit/audit barriers; reference legacy loading; honest cross-backend compatibility | `checkpoints.py`, checkpoint stage profile, legacy/resume tests, real interrupt, final CLI verifier, compatibility table | Provenance reuse implemented; background work rejected on measured cost; strict cross-backend exact-resume rejection retained |
| 12. Frozen eval, representation/reachability/motor/plasticity, relevant resume/replay/audit; independent batches/exposure; safe immutable preparation and reward-free caching | Unified factory call sites, `20261014-frozen-v2`, fresh calibration/preflight-v2, `20261014-preparation-v2`, cache tests; simulator-only replay code inspected; public resume v2 and trained/initial evaluations | Verified: actual public full-state native resume passes; frozen learned and initial weights evaluated independently; simulator replay needs no neural inference or backend conversion |
| 13. Whole-network sparsity; outgoing prototype with exactness/degree/atomic/batching/index costs; high-activity limits; separate duration/precision assessment | `20261012-native-v2-sparsity`, event probe, event execution/tests; FP16 update-resolution assessment; `20261014-duration-50ms-v1` | Verified: event qualified with integral bounds; high-density regression reported; 50 ms fresh calibration, 1,000-decision learning run and held-out evaluation completed; no FP16 gain claimed |
| 14. Small-circuit and real-source validation; signs/separation/delay/refractory/reset/eligibility/terminal/bounds/RNG/batching/frozen trajectories/resume/logs/interrupt/errors/health; sustained memory | Relevant existing tests plus native/CSR/buffer/cache tests; full frozen audit, independent learners, real interrupt, sustained B | Verified: 521 CPU tests passed, ten CUDA tests skipped and covered separately by eleven-test GPU suite; final artifact audits passed; historical neural mismatch retained and disclosed |
| 15. Explicit rollout; correctness/resume/calibration/throughput/memory/build/dashboard gates before default; retained reference; blind clears/wins separated | Default remains `torch_reference`; explicit qualified event configs; guide/versioned identities; trained/initial/50 ms evaluation summaries | Verified: qualified explicit configs provided; no default switch; all three 16-seed held-out evaluations have zero blind clears and wins, reported separately from speed |
| 16. Implementations/build/CLI/runtime/checkpoints/evaluation/tests/commands/guide/compatibility/report; hardware/rates/gains/memory/numerics/rejections/bottlenecks/long single and batched commands | `PERFORMANCE_MIGRATION_GUIDE.md`, `PERFORMANCE_REPORT.md`, work record, implemented files and qualification directories | Verified: deliverables complete; exact qualified single/eight-fly commands ran to declared budgets; final numbers and limitations inspected |

Final current CLI suite and followup suites are terminal and completed. The strict
historical comparison's exit 2 is explicitly labeled a comparison mismatch,
not a passing equivalence claim. Its differing fields are solely neural buffers;
native initial replay and checkpoint continuations are exact. The first public
resume audit failed because deferred host readouts were not JSON serializable;
its evidence is retained. The fix hashes actual array values and passes both a
new regression and the real full-state public resume audit. Final full regression
after this fix is in `20261014-public-audit-v2/final-cpu-regression.stdout`.

An opt-in warm preparation cache and FP16 efficacy were not adopted as speed
optimizations: measured preparation wall time regressed, and FP16 loses small
efficacy updates. Background checkpoint serialization was not justified by the
measured complete save cost. Sixteen-fly batching and dense activity event
propagation regressions are reported. Scientific protocol warnings remain
visible; no safety gate, mapping or reward was weakened to obtain acceptance.

## Verification scope

`tests/test_native_cuda.py` exercises exact neuron state/delay ordering,
refractory transitions, fresh/reordered RNG, persistent buffers and generators,
updated efficacy/sensory values, batch/duration changes, signed fixed/plastic
separation, persistent cuSPARSE plans, binary outgoing recurrence across five
widths, and deterministic many-to-one fractional plastic reduction.

`tests/test_execution_backends.py` covers CSR signs/separation, reset/independence,
index overflow and explicit identity. `tests/test_prepared_cache.py` verifies
source/payload corruption, read-only ownership, layout ranges/width/separation,
warm bypass of COO preparation and private backend anatomy.

`tests/test_buffered_runtime.py` covers ordered Unicode/partial writes, failure
latching/no duplicate retry, checkpoint barriers and interruption boundaries.
Training safety tests cover nonfinite/bound/activity failures, bad individual
learners, forensic stop checkpoint, staged/strict resume and exposure limits.
Existing plasticity/training tests cover learning-only state, terminal clearing,
bounds and exact future updates. The real SIGINT qualification checks CLI logs
and checkpoint alignment beyond the unit tests.

The CLI artifact verifier checks every checkpoint's file and weight hashes,
learner counters, next audited observations when available, ordered exposure in
action/reinforcement/metrics logs, episode counts, and final summary/manifest
hashes. It is an artifact consistency check; full neural replay/resume is proven
by the separate trajectory and actual trainer-resume audits.

Frozen-v2 compares complete actions, rewards, before/after state hashes,
reinforcement and terminal flags, rather than only episode summaries. Real
native qualification hashes all neural buffers, plastic tensors, RNG and
gameplay state. Historical reference neural-buffer differences are explicitly
outside any bitwise equivalence claim.
