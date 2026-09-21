# Plastic-brain V1 implementation status

Statuses are deliberately limited to `LOCALLY VALIDATED`,
`NEEDS HEAVY VALIDATION`, and `BLOCKED`. Local synthetic/CPU evidence proves
software semantics, not that real FlyWire dynamics learn Balatro.

| Requirement | Status | Evidence / dedicated-machine gate |
|---|---|---|
| Primary zero-external-policy architecture | LOCALLY VALIDATED | Primary train/evaluate imports remain isolated from legacy PPO; encoder and motor report zero trainable parameters; efficacy is the only persistent learned behavioural state |
| Field-aware observable sensory contract | LOCALLY VALIDATED | Every pinned field is classified/versioned/hashed; semantic tests preserve phase, Ante, rank, shop and boss indicators at amplitude one; no strategy features |
| ALPN projection and collisions | LOCALLY VALIDATED | Seeded replicates, clipped-sum policy, ALPN assignment quantiles/use/class collisions and isolated rate reports have tests; real firing response still needs calibration |
| Episode trace lifecycle | LOCALLY VALIDATED | Terminal reinforcement updates efficacy before learner-local eligibility/reinforcement reset; multi-row and cross-episode contamination tests pass |
| Fixed-reference eligibility | LOCALLY VALIDATED | Configured KC/MBON Hz references and maximum eligibility replace per-decision max normalization; weak/zero/strong/decay/clipping tests pass |
| Plasticity calibration | LOCALLY VALIDATED | Controlled fixed-Hz/bidirectional-reinforcement and integrated modes report efficacy median/mean, absolute/relative drift, measured per-update magnitudes, changes, safety flags and bounds with configurable PASS/FAIL gates |
| Representation units and separation | NEEDS HEAVY VALIDATION | Metrics are explicit Hz for real backend and include population distributions, within/between variance/distances, separation ratio, motor coverage/range and diagnostic-only observable probes; real reports pending |
| Reward-free motor calibration | NEEDS HEAVY VALIDATION | Persisted artifact, root IDs, state seed/hash set, criteria, width>1 default, hash and deterministic tests exist; real motor artifact must be produced and pass dynamic-range gates |
| Canonical motor controls | LOCALLY VALIDATED | Root universe derives from unshuffled artifact; config rejects mismatched artifacts; matched-manifest and RNG row-order tests pass |
| Separate topology controls | LOCALLY VALIDATED | `real`, `kc_mbon_shuffled` and `whole_brain_shuffled` have distinct procedures and invariant tests |
| Weak-edge sensitivity | LOCALLY VALIDATED | Threshold reports at 1/2/5/10 and optional hashed `minimum_synapse_count` have fixture tests; real counts pending |
| Population/data-quality report | NEEDS HEAVY VALIDATION | Exact 5,177 KC gate retained; named population root/type/rule/hash and unresolved-sign/dropped-synapse/coordinate reporting implemented; rebuild report pending |
| Synthetic reinforcement semantics | LOCALLY VALIDATED | Public logs/replay/Parquet/visualization distinguish synthetic appetitive/aversive channels from genuine DAN anatomy; compatibility aliases are internal only |
| Compartmental extension path | LOCALLY VALIDATED | `three-factor-global-v1` remains primary; `EdgeModulationAssignment` supports channel/sign/compartment and refuses incomplete `compartmental-dan-v2` assignments |
| Per-learner exploration RNG | LOCALLY VALIDATED | Derived motor seed + learner ID + decision ID makes insertion/reordering invariant |
| Device plastic state | NEEDS HEAVY VALIDATION | No-autograd Torch efficacy/eligibility/reinforcement state and NumPy checkpoint round-trip pass CPU tests; CUDA residency/performance pending |
| Batched independent flies | NEEDS HEAVY VALIDATION | One shared fixed sparse graph plus batched KC->MBON scatter contribution replaces serialized real learners; measured sequential/batched comparison pending |
| Streaming neural/plastic recording | LOCALLY VALIDATED | One Parquet row group per decision, dictionary roles/kinds, sparse edge detail and bounded recorder memory have tests |
| Selective visualization reads | LOCALLY VALIDATED | Row-group statistics select requested decisions; fixed transform/background retained |
| Optional time-resolved showcase | NEEDS HEAVY VALIDATION | Frozen real evaluation can opt into input stimulation and actual backend spike events; full recording/render intentionally not run locally |
| Visualization role/detail semantics | LOCALLY VALIDATED | Input, KC, MBON, DAN anatomy, descending, both synthetic channels and plasticity have distinct labels; strongest edge pre/post/old/new/delta is supported |
| Strict matched controls | LOCALLY VALIDATED | Action/state schedule, Ante, seed, sensory/motor, budget and artifact/population assertions exist; topology conditions are explicit |
| Replicate protocols | LOCALLY VALIDATED | First-class manifests expand every replicate into paired condition arms with plasticity/sensory/motor/environment/fly/topology/reward seeds and exposure |
| Formal preflight | LOCALLY VALIDATED | `flylatro-preflight` emits PASS/WARN/FAIL reasons for artifact, data, sensory, representation, motor, plasticity, reward, reset, resume, benchmarks, controls and external-policy state |
| Checkpoint/resume | LOCALLY VALIDATED | NumPy and Torch plastic states serialize without gradients; synthetic future updates reproduce exactly |
| GPU runbook and budgets | LOCALLY VALIDATED | Mandatory 14-stage order implements artifact through curriculum; real template has a refused budget placeholder and fixed Ante-1 target, not arbitrary exposure/promotion recommendations |
| Fast regression suite | LOCALLY VALIDATED | 115 lightweight tests passed locally after the hardening pass; compileall also passed |

## Current verification boundary

No full artifact build, CUDA sweep, long Balatro training, large evaluation,
long spike recording or final video was run here. The real v783 dataset is not
present in this workspace. Therefore the following are not empirical claims:

- that chosen durations produce useful real KC/MBON/descending activity;
- that the reward-free motor pools cover all actions dynamically;
- that the default learning/reference rates avoid bounds over real exposures;
- that batched execution improves throughput or fits desired GPU memory;
- that Ante-1 behaviour improves relative to matched controls.

Those are the dedicated-machine gates in `docs/GPU_RUNBOOK.md`. A failed long
experiment is scientifically interpretable only after the strict second
preflight has no scientific/readiness FAIL.

## Known biological simplifications

Balatro-to-ALPN assignments and fly-to-action pools are synthetic interfaces.
Connections are aggregate neuron-pair weights. The primary rule has one global
synthetic reinforcement difference, not evidence-derived compartmental DAN
spiking, receptor kinetics, consolidation or synapse-level timing. Compartment
labels from MBON types are descriptive; `compartmental-dan-v2` stays
uninstantiated until defensible assignments exist.
