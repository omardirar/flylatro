# Plastic-brain V1 implementation status

Statuses are deliberately limited to `LOCALLY VALIDATED`,
`NEEDS HEAVY VALIDATION`, and `BLOCKED`. Local synthetic/CPU evidence proves
software semantics, not that real FlyWire dynamics learn Balatro.

| Requirement | Status | Evidence / dedicated-machine gate |
|---|---|---|
| Primary zero-external-policy architecture | LOCALLY VALIDATED | Primary train/evaluate imports remain isolated from legacy PPO; encoder and motor report zero trainable parameters; efficacy is the only persistent learned behavioural state; every final metadata product records `external_trainable_parameter_count = 0` |
| Field-aware observable sensory contract | LOCALLY VALIDATED | Every pinned field is classified/versioned/hashed; semantic tests preserve phase, Ante, rank, shop and boss indicators at amplitude one; no strategy features |
| Structural ALPN assignment metrics | LOCALLY VALIDATED | Seeded replicates, clipped-sum policy, assignment quantiles, ALPN use and class collisions are reported and explicitly labelled informational, not a readiness gate |
| State-conditioned sensory health | NEEDS HEAVY VALIDATION | Silent/active/saturated ALPN fractions, simultaneous active contributors per ALPN, collision-caused saturation, encoder determinism, same-state equality and distinct-state separation are computed on the frozen corpus and gated; a saturating tiny mapping is shown to fail. Real v783 report pending |
| Frozen reward-free calibration corpus | LOCALLY VALIDATED | Deterministic, versioned, SHA-256-hashed observable states and masks only; seeds, per-state hashes, phase labels, generation method and coverage persisted; determinism, round-trip and reward-free-content tests pass. Real corpus and its phase coverage pending |
| Episode trace lifecycle | LOCALLY VALIDATED | Terminal reinforcement updates efficacy before learner-local eligibility/reinforcement reset; multi-row and cross-episode contamination tests pass |
| Fixed-reference eligibility | LOCALLY VALIDATED | Configured KC/MBON Hz references and maximum eligibility replace per-decision max normalization; weak/zero/strong/decay/clipping tests pass |
| Plasticity calibration | LOCALLY VALIDATED | Controlled fixed-Hz/bidirectional-reinforcement and integrated modes report efficacy median/mean, absolute/relative drift, measured per-update magnitudes, changes, safety flags and bounds with configurable PASS/FAIL gates and evidence provenance |
| Representation pre/post stages | NEEDS HEAVY VALIDATION | Distinct `pre` (no motor claim) and `post` (persisted final mapping) stages; metrics are explicit Hz including population distributions, within/between variance and distance, separation ratio, and diagnostic-only observable probes. Real reports pending |
| Motor population measured | LOCALLY VALIDATED | Diagnostics take `motor_activity` explicitly and record `motor_activity_population`; regression tests use differing MBON/descending counts and statistics so the previous mbon_direct/descending defect cannot recur |
| Contextual motor routing | LOCALLY VALIDATED | 35 pools instead of 61; contextual heads share the slot group, simultaneously compared heads stay disjoint; routing is versioned, hashed and non-trainable |
| Reserved action slots | LOCALLY VALIDATED | Types 0-12 are represented, 13-24 own no population, score `-inf`, are counted when reported legal, and cannot be selected; the artifact still spans the full 25-wide contract |
| Canonical plastic-reachable motor candidates | NEEDS HEAVY VALIDATION | Candidates derive from the real unshuffled KC->MBON topology at the canonical threshold, with per-candidate root ID, KC edge count, synapse weight, type and compartment; candidate-set hash persisted and checked on load. Real candidate census pending |
| Fixed reward-free motor normalization | NEEDS HEAVY VALIDATION | `reward-free-median-iqr-v1` baseline/scale persisted and hashed, applied in `decode` before comparison; offset-pool fairness, determinism and hash stability tested. Real calibration pending |
| Motor calibration quality gates | NEEDS HEAVY VALIDATION | Candidate/selected counts, pool width, reuse, baseline distribution, dynamic range, variance, normalized range, silent/high-rate fractions, competing-pool correlation and effective distinct signals are reported and gated; a degenerate-diversity case is shown to fail. Real gate pending |
| Canonical motor controls | LOCALLY VALIDATED | Universe derives from the real unshuffled artifact and is invariant to topology shuffling and weak-edge thresholds; config rejects a mismatched candidate-set hash |
| Separate topology controls | LOCALLY VALIDATED | `real`, `kc_mbon_shuffled` and `whole_brain_shuffled` have distinct procedures and invariant tests |
| Matched-control semantics | LOCALLY VALIDATED | Exact action/state matched (`no_plasticity`, `shuffled_reward`) and behaviourally independent topology controls are distinguished in code, report and documentation; a topology control that did not change topology fails |
| Weak-edge sensitivity | LOCALLY VALIDATED | Threshold reports at 1/2/5/10 and optional hashed `minimum_synapse_count` have fixture tests; real counts pending |
| Population/data-quality report | NEEDS HEAVY VALIDATION | Exact 5,177 KC gate retained; named population root/type/rule/hash and unresolved-sign/dropped-synapse/coordinate reporting implemented; rebuild report pending |
| KC subtype reachability | NEEDS HEAVY VALIDATION | Per-subtype counts, active fractions, Hz quantiles, plastic-edge reachability, ever-eligible fraction, eligibility mass, plastic-MBON and motor-output coverage implemented and tested on synthetic data. The real answer for ALPN-only drive is a dedicated-machine measurement |
| Chosen-action plasticity specificity | NEEDS HEAVY VALIDATION | Per-decision eligibility and weight-change attribution to chosen pool, competing pools, unconsulted motor pools and non-motor MBONs; four columns sum to the total. Real measurement pending; the rule is unchanged |
| Synthetic reinforcement semantics | LOCALLY VALIDATED | Public logs/replay/Parquet/visualization distinguish synthetic appetitive/aversive channels from genuine DAN anatomy; compatibility aliases are internal only |
| Predeclared reinforcement sensitivity | LOCALLY VALIDATED | `primary-progress`, `reduced-progress` and `terminal-or-clear-only` are named, hashed and refuse ad-hoc retuning; configs shipped. Controlled comparison is a later experiment |
| Compartmental extension path | LOCALLY VALIDATED | `three-factor-global-v1` remains primary; `EdgeModulationAssignment` supports channel/sign/compartment and refuses incomplete `compartmental-dan-v2` assignments |
| Per-learner exploration RNG | LOCALLY VALIDATED | Derived motor seed + learner ID + decision ID makes insertion/reordering invariant |
| Device-resident plastic hot path | NEEDS HEAVY VALIDATION | Routine CUDA steps keep efficacy/eligibility/reinforcement on device; one small scalar block per learning step and one per telemetry step; a monkeypatched transfer test proves no full vector crosses the boundary unless detail is requested. CUDA residency/throughput pending |
| Lightweight vs detailed plastic events | LOCALLY VALIDATED | `PlasticityEvent` is scalar by default; `PlasticityEventDetail` (hashes, changed edge IDs, per-edge deltas) is opt-in; weight hashing is checkpoint/audit-time |
| Batched independent flies | NEEDS HEAVY VALIDATION | One shared fixed sparse graph plus batched KC->MBON scatter contribution; measured sequential/batched comparison pending |
| Benchmark component attribution | NEEDS HEAVY VALIDATION | Poisson, fixed recurrence, plastic contribution, LIF update and device sync are timed and reported with provenance; two reproducible Poisson methods exist and the chunked one is opt-in. Real CUDA measurement pending |
| Streaming neural/plastic recording | LOCALLY VALIDATED | One Parquet row group per decision, dictionary roles/kinds, sparse edge detail and bounded recorder memory have tests |
| Selective visualization reads | LOCALLY VALIDATED | Row-group statistics select requested decisions; fixed transform/background retained |
| Optional time-resolved showcase | NEEDS HEAVY VALIDATION | Frozen real evaluation can opt into input stimulation and actual backend spike events; full recording/render intentionally not run locally |
| Visualization role/detail semantics | LOCALLY VALIDATED | Input, KC, MBON, DAN anatomy, descending, both synthetic channels and plasticity have distinct labels; strongest edge pre/post/old/new/delta is supported |
| Replicate protocols and seed audit | LOCALLY VALIDATED | Every stored seed has a declared stochastic effect; the no-op `plasticity_seed` was removed; `SEED_EFFECTS` and per-condition usage are in the manifest and asserted by tests |
| Executable protocols | LOCALLY VALIDATED | `flylatro-materialize-protocol` writes one validated configuration per arm plus commands and dependency order; each run re-validates its configuration against its protocol arm |
| Evidence provenance | LOCALLY VALIDATED | Population, sensory mapping, sensory health, representation pre/post, motor calibration, reachability, plasticity, specificity, benchmark, preflight and control reports all carry an explicit evidence identity |
| Provenance-verifying preflight | LOCALLY VALIDATED | Gates refuse a `PASS` report whose duration, sensory mapping, motor mapping, artifact, topology, plasticity rule or calibration corpus differs, and refuse a report with no identity at all |
| Initial and strict preflight profiles | LOCALLY VALIDATED | `initial` downgrades benchmark/control/tiny-run/protocol/specificity to WARN; `ante1` requires them; both tested |
| Checkpoint/resume | LOCALLY VALIDATED | NumPy and Torch plastic states serialize without gradients; synthetic future updates reproduce exactly |
| Final run metadata identity | LOCALLY VALIDATED | One canonical completed component identity is written to run manifest, run summary, final checkpoint and checkpoint manifest; a test asserts every identity field agrees |
| Run readiness pathway | LOCALLY VALIDATED | `flylatro-readiness` reports `READY_FOR_ARTIFACT_BUILD` / `READY_FOR_CALIBRATION` / `READY_FOR_TINY_REAL_RUN` / `READY_FOR_ANTE1` with blocking reasons and never claims Ante-1 readiness from synthetic evidence |
| GPU runbook and budgets | LOCALLY VALIDATED | 24-stage corrected dependency order from artifact build to curriculum; executable commands matching real CLI options; explicit stop conditions |
| Source licence and dependency lock | LOCALLY VALIDATED | MIT `LICENSE` with an explicit non-redistribution note; `uv.lock` pins the resolved graph including the pinned simulator revision; ADR 0013 records the decision |
| Fast regression suite | LOCALLY VALIDATED | 219 lightweight tests pass locally; `compileall` and `git diff --check` pass |

## Current verification boundary

No full artifact build, CUDA sweep, long Balatro training, large evaluation,
long spike recording or final video was run here. The real v783 dataset and the
compiled `balatro_sim` extension are both absent from this workspace, so every
local result uses the synthetic mushroom-body double and the mock environment.
Therefore the following are **not** empirical claims:

- that the real ALPN mapping preserves observable-state information at the
  configured `max_rate_hz` and population width;
- that chosen durations produce useful real KC/MBON/descending activity;
- that ALPN-only drive reaches enough of the plastic KC population;
- that the canonical plastic-reachable MBON population is large enough to
  supply 35 pools of at least width two with sufficient diversity;
- that the reward-free motor pools cover all actions dynamically;
- that the default learning/reference rates avoid bounds over real exposures;
- that `three-factor-global-v1` is sufficiently action-specific;
- that batched execution improves throughput or fits desired GPU memory;
- that Poisson generation is or is not the dominant runtime component;
- that Ante-1 behaviour improves relative to matched controls.

Those are the dedicated-machine gates in `docs/GPU_RUNBOOK.md`. A failed long
experiment is scientifically interpretable only after the strict `ante1`
preflight has no scientific/readiness FAIL.

Two gate failures are known and expected on this machine, because they are
honest failures of the development double rather than defects:

- the synthetic circuit's outputs are near-perfectly correlated, so the
  post-motor `competing_pools_distinguishable` gate fails with default
  thresholds;
- the synthetic topology is complete-bipartite, so every edge is eligible and
  the default `maximum_modified_fraction` is exceeded.

Neither says anything about the real sparse connectome.

## Known biological simplifications

Balatro-to-ALPN assignments and fly-to-action pools are synthetic interfaces;
reusing one neural pool for shop, pack, joker and consumable slot two is an
additional declared interface convention. Connections are aggregate neuron-pair
weights. The primary rule has one global synthetic reinforcement difference, not
evidence-derived compartmental DAN spiking, receptor kinetics, consolidation or
synapse-level timing. Compartment labels from MBON types are descriptive;
`compartmental-dan-v2` stays uninstantiated until defensible assignments exist.
Diagnostic classifiers and probes are never policy components.
