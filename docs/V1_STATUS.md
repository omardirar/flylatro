# Plastic-brain V1 implementation status

This is the live completion ledger for the authoritative `PLAN.MD`. Statuses
mean `TODO`, `IN PROGRESS`, `IMPLEMENTED`, `LOCALLY VALIDATED`,
`NEEDS HEAVY VALIDATION`, or `BLOCKED`. A green synthetic test is software
evidence, not evidence that real FlyWire dynamics learn Balatro.

| Requirement | Status | Evidence / remaining gate |
|---|---|---|
| Authoritative plan and legacy isolation | LOCALLY VALIDATED | `PLAN.MD` is plastic-brain V1; the reservoir plan is archived; legacy commands are explicitly named; an import-boundary test proves train/evaluate/replay do not load PPO/policy modules |
| Architecture ADRs | LOCALLY VALIDATED | ADRs 0004-0007 cover learned state, population rules, plasticity/dopamine timing, reset policy, learner identity, output modes, and fixed interfaces |
| Versioned KC/MBON/DAN populations | NEEDS HEAVY VALIDATION | Schema-v2 builder/loader and tests store exact KC, MBON, DAN, PAM, PPL1, ALPN, APL, DPM and descending indices/root IDs, rules, counts and hashes; the full v783 census must run on supplied source data |
| Sparse plastic-edge representation | LOCALLY VALIDATED | Shared immutable KC->MBON topology plus per-learner one-dimensional efficacy/eligibility arrays; no dense whole-brain matrix |
| Eligibility and dopamine traces | LOCALLY VALIDATED | Three-factor unit tests cover zero gates, signs, decay, bounds, reset persistence, independent learners and exact round trips |
| Fixed seeded sensory interface | LOCALLY VALIDATED | Observable-state-only ALPN random projection; stable seed/hash and zero-parameter encoding tests; real response calibration remains heavy work |
| Fixed zero-parameter motor interface | LOCALLY VALIDATED | Stable structured action/card/count/target pools, optional consumable card targets, legal masks, deterministic decoding, explicit fixed exploration and hash tests |
| MBON-direct mode | NEEDS HEAVY VALIDATION | Synthetic training changes internal weights and behavior; real v783 activity/learning remains stage 5-9 of the runbook |
| Whole-brain/descending mode | NEEDS HEAVY VALIDATION | Tiny sparse graph proves KC->MBON changes reach MBON and downstream state; synthetic Mode B train/evaluate/replay passes; full-v783 Mode B has an explicit benchmark command |
| Plastic-learning trainer | LOCALLY VALIDATED | Exposure-budgeted primary trainer has no actor, critic, gradient or optimizer; weights persist across episodes; action/dopamine/plasticity/runtime metrics are logged |
| One fly versus independent flies | LOCALLY VALIDATED | One sequential fly is the documented default; vector rows own disjoint plastic state and non-divisible exposure budgets are rejected; learned states are never merged |
| Curriculum and matched exposure | LOCALLY VALIDATED | Held-out seed stream, promotion cadence/state, decision budgets and CLI overrides have tests; real promotion behavior remains an experiment result |
| Frozen evaluation | LOCALLY VALIDATED | Evaluation preserves weight, eligibility and dopamine state hashes and reports win/Ante/length/decision/score metrics |
| Plastic checkpoint/resume | LOCALLY VALIDATED | Hash-verified atomic checkpoint captures current/initial efficacy, traces, environment snapshot, RNGs, curriculum, config, exact mapping payloads/hashes, Git and hardware/software sidecar metadata; future updates reproduce exactly |
| Synaptic-change analysis | LOCALLY VALIDATED | Reports count, distribution, sign, absolute/relative change and groups by MBON root/type, KC type and inferred compartment; per-decision drift metrics and periodic checkpoints expose stage progression |
| Representation/plasticity diagnostics | LOCALLY VALIDATED | Repeated-state trials measure KC sparsity/activity, MBON/descending silence/saturation, variability and separability; training reports eligibility, dopamine, drift, bounds, action entropy/diversity and dominance |
| No-plasticity control | LOCALLY VALIDATED | Updates are disabled; a lazy source schedule enforces identical actions, curriculum Antes and before/after state hashes while weights retain their initial hash |
| Shuffled-topology control | LOCALLY VALIDATED | Seeded population-preserving postsynaptic permutation preserves edge/weight counts, pre-degree and post-degree distribution and keeps KC->MBON endpoints in population |
| Shuffled-reward control | LOCALLY VALIDATED | Persisted hash-verified dopamine permutation plus a lazy source schedule enforces identical actions, curriculum Antes and before/after state hashes; synthetic end-to-end smoke passed |
| Sensory-map replicates | LOCALLY VALIDATED | Mapping seed override changes the mapping hash reproducibly while leaving other components fixed |
| Legacy and conventional baselines | LOCALLY VALIDATED | Existing reservoir/PPO and external baseline implementations remain available only through explicitly labelled legacy/baseline commands and configs |
| Deterministic replay | LOCALLY VALIDATED | Plastic replay identities carry weight/rule/population/interface hashes; a synthetic frozen-evaluation bundle reverified every state/action transition |
| Neural/plasticity recording | LOCALLY VALIDATED | Selected runs record KC, MBON, DAN, descending, dopamine and learned-weight snapshots; opt-in training records sparse changed edges/root IDs rather than full matrices |
| Fixed-coordinate visualisation | LOCALLY VALIDATED | Persisted whole-artifact transform, faint anatomy, distinct roles, absolute signed-change ranking and slowdown labels have fast tests; a full render remains heavy validation |
| Plastic benchmarks | LOCALLY VALIDATED | Sparse-update and plastic end-to-end entry points report throughput/RSS/GPU metrics and enforce heavy guards; dedicated-machine measurements remain pending |
| Heavy-machine runbook | LOCALLY VALIDATED | `docs/GPU_RUNBOOK.md` has all 18 requested stages, parser-checked commands, measured-budget placeholders and explicit success criteria |
| Fast tests | LOCALLY VALIDATED | 102 tests pass, including plasticity, mappings, controls, outcome mapping, exact resume, deterministic learning, Mode B propagation, frozen evaluation, replay imports and visualisation |

## Current verification boundary

Local validation used only mock Balatro, tiny/synthetic plastic circuits and
short runs. It proved the software path can learn only by changing internal
KC->MBON efficacy, checkpoint/resume exactly, freeze for evaluation, emit
learning-relevant recordings and replay deterministically.

No full v783 artifact build/census, compiled real Balatro simulator run, CUDA
whole-brain calibration, long curriculum, live Balatro replay, 10,000-seed
final test or full final render was run on this development machine. Those are
not blockers to code completion; they remain empirical/scientific validation
gates in `docs/GPU_RUNBOOK.md` and must not be described as completed results.
