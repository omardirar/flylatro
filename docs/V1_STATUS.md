# Plastic-brain V1 implementation status

Statuses are deliberately limited to `LOCALLY VALIDATED`,
`NEEDS HEAVY VALIDATION`, and `BLOCKED`. Local software tests and bounded GPU
gates establish software semantics and measured calibration, not that real
FlyWire dynamics learn Balatro.

## Current research status

Overview updated 12 November 2026 from completed studies through 6 November.
The primary model's calibration history is recorded below. Later experimental
rules and interfaces have separate identities and do not replace primary V1.

- Native CUDA execution, independent learners and exact native resume are
  qualified for the recorded workloads. The reference remains the default;
  see [measured performance and limits](PERFORMANCE_REPORT.md).
- The [repaired native long run](REPAIRED_LONG_RESULTS.md) stopped at 509,922
  decisions per plastic learner on the saturation guard. Fresh frozen
  evaluation found partial first-blind progress and no Ante-1 wins.
- The [single-seed experiment](SINGLE_SEED_RESULTS.md) completed six million
  decisions across six learners. No saved greedy evaluation cleared the
  repeated game's first blind or won Ante 1. Card membership remained close
  to same-count random selection.
- [Local teacher curricula](LOCAL_TEACHER_CURRICULUM.md),
  [exchange credit](EXCHANGE_CREDIT_PILOT.md),
  [relational inputs](RELATIONAL_SENSORY_PILOT.md) and
  [anatomical sensory controls](SENSORY_CONTROL_RESULTS.md) have not qualified
  a new long-training candidate. The next proposed diagnostic is direct
  KC-afferent stimulation to isolate sensory routing from output control.

Reliable autonomous Ante-1 learning remains unproven. The
[documentation index](README.md) links methods and results; the
[learning roadmap](LEARNING_ROADMAP.md) orders further research.

## Primary calibration and implementation evidence

The table records specific software properties and measured studies. A
`LOCALLY VALIDATED` entry does not imply gameplay competence. Historical test
counts and benchmark timings apply to their recorded source and workload.

Source-coverage audit (October 2026): the historical artifact used a threshold-5
export and omitted 40,567 weak KC->MBON pairs. Its earlier readiness PASS
does not prove PLAN.MD's minimum-count-1 anatomy requirement. The separately
built source-complete artifact has 67,504 plastic pairs; its initial 500 ms
representation fails the unchanged high-rate/separation gates.
Gain 0.18 passes representation/reachability at 100/250/500 ms but fails
action-type motor diversity. Pearson correlation estimator repairs are versioned
in motor v5 and representation v4; historical reports are preserved and current
readiness requires newly generated versions. The prospectively ordered fixed
gain follow-up selects 0.225 mV / 100 ms: pre/post representation, reachability,
motor and integrated plasticity pass. The source-complete LR 0.002 pilot
completes 250 decisions safely with full learned-checkpoint health, exact
control and actual GPU resume PASS. All six strict preflights pass. The first
primary and matched control complete 1000 decisions of training. The primary
passes its final neural audits and wins 0/32 independent
validation episodes; the driver was stopped at the user's request during
control evaluation, with the remaining two pairs unrun. Frozen confirmatory evaluation is
still unrun, and improved Ante-1 performance remains unproven.
See `docs/PROGRESS_LOG.md` and ADR 0015. Two historical-graph LR pilots completed
250 decisions and passed learned-checkpoint audits, but neither won Ante 1.

| Requirement | Status | Evidence / dedicated-machine gate |
|---|---|---|
| Primary zero-external-policy architecture | LOCALLY VALIDATED | Primary train/evaluate imports remain isolated from legacy PPO; encoder and motor report zero trainable parameters; efficacy is the only persistent learned behavioural state; every final metadata product records `external_trainable_parameter_count = 0` |
| Field-aware observable sensory contract | LOCALLY VALIDATED | Every pinned field is classified/versioned/hashed; semantic tests preserve phase, Ante, rank, shop and boss indicators at amplitude one; no strategy features |
| Structural ALPN assignment metrics | LOCALLY VALIDATED | Seeded replicates, clipped-sum policy, assignment quantiles, ALPN use and class collisions are reported and explicitly labelled informational, not a readiness gate |
| State-conditioned sensory health | LOCALLY VALIDATED | Real v783 sensory health passes on the balanced 96-state corpus; a saturating tiny mapping is shown to fail. Generalization beyond that frozen corpus remains unmeasured |
| Frozen reward-free calibration corpus | LOCALLY VALIDATED | Real balanced corpus has 96 unique states from 93 runs and passes default phase/context coverage, including competing inventory targets; arrays, snapshots and source provenance are SHA-256 bound |
| Episode trace lifecycle | LOCALLY VALIDATED | Terminal reinforcement updates efficacy before learner-local eligibility/reinforcement reset; multi-row and cross-episode contamination tests pass |
| Fixed-reference eligibility | LOCALLY VALIDATED | Configured KC/MBON Hz references and maximum eligibility replace per-decision max normalization; weak/zero/strong/decay/clipping tests pass |
| Plasticity calibration | LOCALLY VALIDATED | Controlled fixed-Hz/bidirectional-reinforcement and integrated modes report efficacy median/mean, absolute/relative drift, measured per-update magnitudes, changes, safety flags and bounds with configurable PASS/FAIL gates and evidence provenance |
| Representation pre/post stages | LOCALLY VALIDATED | Source-complete gain 0.225 / 100 ms passes full 96-state, three-repeat pre/post representation and learned 250-decision checkpoint health; default 500 ms and lower-gain failures remain preserved |
| Motor population measured | LOCALLY VALIDATED | Diagnostics take `motor_activity` explicitly and record `motor_activity_population`; regression tests use differing MBON/descending counts and statistics so the previous mbon_direct/descending defect cannot recur |
| Contextual motor routing | LOCALLY VALIDATED | 35 pools instead of 61; contextual heads share the slot group, simultaneously compared heads stay disjoint; routing is versioned, hashed and non-trainable |
| Reserved action slots | LOCALLY VALIDATED | Types 0-12 are represented, 13-24 own no population, score `-inf`, are counted when reported legal, and cannot be selected; the artifact still spans the full 25-wide contract |
| Canonical plastic-reachable motor candidates | LOCALLY VALIDATED | Source-complete universe has all 96 MBONs, versus historical 84; eligibility is measured for each named gain/window. Candidate-set hash is persisted and checked on load |
| Fixed reward-free motor normalization | LOCALLY VALIDATED | Fixed median/IQR normalization, 35 pools, width two and zero trainable decoder; historical v4 maps remain preserved. Current v5 uses the corrected Pearson estimator |
| Motor calibration quality gates | LOCALLY VALIDATED | Source-complete gain 0.225 / 100 ms motor v5 passes: 55 eligible, 34 unique neurons, 70 memberships and effective action-type fraction 0.511395 > 0.50. Selection uses no rewards or performance labels |
| Canonical motor controls | LOCALLY VALIDATED | Universe derives from the real unshuffled artifact and is invariant to topology shuffling and weak-edge thresholds; config rejects a mismatched candidate-set hash |
| Separate topology controls | LOCALLY VALIDATED | `real`, `kc_mbon_shuffled` and `whole_brain_shuffled` have distinct procedures and invariant tests |
| Matched-control semantics | LOCALLY VALIDATED | Exact action/state matched (`no_plasticity`, `shuffled_reward`) and behaviourally independent topology controls are distinguished in code, report and documentation; a topology control that did not change topology fails |
| Weak-edge sensitivity | LOCALLY VALIDATED | Source-complete real pair counts at 1/2/5/10 are 67,504/52,463/26,937/6,973; fixture proves weak pairs survive and existing artifacts are preserved |
| Population/data-quality report | LOCALLY VALIDATED | New source-complete build and report verify 139,255 neurons, exact 5,177 KCs, full named populations and source/sign/coordinate accounting; zero source synapse drops |
| KC subtype reachability | LOCALLY VALIDATED | Source-complete gain 0.225 / 100 ms full-corpus pre/post and learned 250-decision reachability pass, including the persisted selected motor mapping |
| Chosen-action plasticity specificity | NEEDS HEAVY VALIDATION | Real 20-decision diagnostic measures 4.83% chosen-motor, 40.58% competing-motor and 40.54% non-motor update attribution. The diffuse global rule is unchanged; effective action-specific learning remains unproven |
| Synthetic reinforcement semantics | LOCALLY VALIDATED | Public logs/replay/Parquet/visualization distinguish synthetic appetitive/aversive channels from genuine DAN anatomy; compatibility aliases are internal only |
| Predeclared reinforcement sensitivity | LOCALLY VALIDATED | `primary-progress`, `reduced-progress` and `terminal-or-clear-only` are named, hashed and refuse ad-hoc retuning; configs shipped. Controlled comparison is a later experiment |
| Compartmental extension path | LOCALLY VALIDATED | `three-factor-global-v1` remains primary; `EdgeModulationAssignment` supports channel/sign/compartment and refuses incomplete `compartmental-dan-v2` assignments |
| Per-learner exploration RNG | LOCALLY VALIDATED | Derived motor seed + learner ID + decision ID makes insertion/reordering invariant |
| Device-resident plastic hot path | LOCALLY VALIDATED | Routine CUDA steps keep efficacy/eligibility/reinforcement on device; native execution retains device-resident learning and compact health transfers. Current qualified workload measurements are in [the performance report](PERFORMANCE_REPORT.md); the earlier 0.767 decisions/s pilot used the reference path |
| Lightweight vs detailed plastic events | LOCALLY VALIDATED | `PlasticityEvent` is scalar by default; `PlasticityEventDetail` (hashes, changed edge IDs, per-edge deltas) is opt-in; weight hashing is checkpoint/audit-time |
| Batched independent flies | LOCALLY VALIDATED | Four independent native closed-loop learners match separate learners; frozen reference/native transitions agree at widths 1/2/4/8/16 in the recorded qualification. Eight-fly CLI throughput is 55.592 aggregate decisions/s for the measured workload; see [qualification limits](PERFORMANCE_REPORT.md) |
| Benchmark component attribution | LOCALLY VALIDATED | Reference and native component profiles are retained separately from throughput runs; the [performance report](PERFORMANCE_REPORT.md) documents dispatch, plastic reduction, recurrence, LIF and random-generation costs. Preflight binds full model identity |
| Streaming neural/plastic recording | LOCALLY VALIDATED | One Parquet row group per decision, dictionary roles/kinds, sparse edge detail and bounded recorder memory have tests |
| Selective visualization reads | LOCALLY VALIDATED | Row-group statistics select requested decisions; fixed transform/background retained |
| Optional time-resolved showcase | NEEDS HEAVY VALIDATION | Frozen real evaluation can opt into input stimulation and actual backend spike events; full recording/render intentionally not run locally |
| Visualization role/detail semantics | LOCALLY VALIDATED | Input, KC, MBON, DAN anatomy, descending, both synthetic channels and plasticity have distinct labels; strongest edge pre/post/old/new/delta is supported |
| Replicate protocols and seed audit | LOCALLY VALIDATED | Every stored seed has a declared stochastic effect; the no-op `plasticity_seed` was removed; `SEED_EFFECTS` and per-condition usage are in the manifest and asserted by tests |
| Executable protocols | LOCALLY VALIDATED | `flylatro-materialize-protocol` writes one validated configuration per arm plus commands and dependency order; each run re-validates its configuration against its protocol arm |
| Evidence provenance | LOCALLY VALIDATED | Population, sensory mapping, sensory health, representation pre/post, motor calibration, reachability, plasticity, specificity, benchmark, preflight and control reports all carry an explicit evidence identity |
| Provenance-verifying preflight | LOCALLY VALIDATED | Gates refuse a `PASS` report whose duration, sensory mapping, motor mapping, artifact, topology, plasticity rule or calibration corpus differs, and refuse a report with no identity at all |
| Initial and strict preflight profiles | LOCALLY VALIDATED | `initial` downgrades benchmark/control/tiny-run/protocol/specificity to WARN; `ante1` requires them; both tested |
| Checkpoint/resume | LOCALLY VALIDATED | Actual source-complete GPU probe passes all 13 future steps: actions, neural activity, environment, weights/traces, RNG and safety, including first-resume reward credit. Raw file identity and pinned-format canonical RNG-map identity are distinct |
| Configurable frozen Ante evaluation | LOCALLY VALIDATED | Ante 1-8 targets, per-episode streaming, explicit free-action metadata, and same-seed paired comparisons; 32 real held-out seeds yielded 0 wins for both 100-decision checkpoints |
| Learned-checkpoint neural health | LOCALLY VALIDATED | Full-corpus diagnostics load actual efficacy and bind checkpoint/weight SHA-256; learned reports cannot replace initial calibration. The real 100-decision checkpoint passes representation and reachability |
| Fail-stop staged training | LOCALLY VALIDATED | Default-on worst-learner finite/bound/drift/silence guards, forensic stop checkpoint, stage caps preserving exact resume, and protocol executor halting all remaining work on failed health |
| Live training dashboard | LOCALLY VALIDATED | Responsive coloured dashboard with per-fly current/highest/average rounds, directly labelled completed-run performance bars with zero baseline/shared scale, named action history and separate game/round/Ante milestones. Keyboard/mouse browsing keeps history steady while training proceeds; clear metric groups and help explain reward, chips, scores and limits. Fixed Red Deck / White Stake and Ante win target are explicit. Monitoring survives checkpoints and leaves tested actions/weights identical; absent legacy history/scores are labelled |
| Final run metadata identity | LOCALLY VALIDATED | One canonical completed component identity is written to run manifest, run summary, final checkpoint and checkpoint manifest; a test asserts every identity field agrees |
| Run readiness pathway | LOCALLY VALIDATED | `flylatro-readiness` reports `READY_FOR_ARTIFACT_BUILD` / `READY_FOR_CALIBRATION` / `READY_FOR_TINY_REAL_RUN` / `READY_FOR_ANTE1` with blocking reasons and never claims Ante-1 readiness from synthetic evidence |
| GPU runbook and budgets | LOCALLY VALIDATED | Corrected dependency order and executable commands with explicit stop conditions. User-authorized unlimited local GPU duration supersedes the original six-hour ceiling; measured accounting and safety guards remain |
| Source licence and dependency lock | LOCALLY VALIDATED | MIT `LICENSE` with an explicit non-redistribution note; `uv.lock` pins the resolved graph including the pinned simulator revision; ADR 0013 records the decision |
| Fast regression suite | LOCALLY VALIDATED | Historical pre-migration snapshot: 488 lightweight tests passed in 31.55 s; bound parity, source coverage, corrected Pearson correlation, confirmatory inference, real-state resume semantics, frozen batch qualification, Ante-target replay, unlimited-duration authorization, dashboard histories/round metrics/navigation and terminal restoration included; compileall and git diff checks pass |

## Historical threshold-5 calibration measured on 2026-10-09

The real v783 artifact and pinned compiled Balatro simulator are available on
this machine. The artifact reports 139,255 neurons, 5,177 KCs, 96 MBONs,
685 ALPNs and 26,937 plastic KC->MBON edges. CUDA diagnostics ran on the
RTX 4080. The historical 50 ms profile is
`configs/plastic-v783-calibration.toml`; the working 500 ms profile is
`configs/plastic-v783-resolution.toml`.

Longer legality-only navigation plus the opt-in inventory-context rule produced
the source corpus `artefacts/calibration-corpus-v783-long-navigation.npz`.
The new `flylatro-select-calibration-corpus` command retained 96 unique states
from 93 source runs, preserving snapshots and per-state provenance. Every
required phase and motor context satisfies the default coverage policy,
including two states with competing Joker targets. The frozen selected corpus
is `artefacts/calibration-corpus-v783-balanced-long.npz`, SHA-256
`020a14f202d5615df220c88cf206dcb92d4542a1d6e0d6c3bdeab6586358b43f`.

The sensory-health gate passes on that corpus. The original 50 ms profile was
blocked:

- `representation-pre-balanced-long-50ms.json` fails state separation:
  ratio 1.0544 versus the required 1.10, using three repeats per state.
- `kc-reachability-balanced-long-50ms.json` finds only 46.4% of the plastic
  MBON/output population ever active; MBON and motor coverage gates fail.
- `motor-map-v783-balanced-long-report.json` finds 18 usable candidates from
  the canonical 84-MBON universe, versus 70 required for 35 pools of width two.
  No calibrated motor artifact is written.
- `preflight-v783-calibration-initial.json` is FAIL. Post-motor representation
  and integrated plasticity calibration cannot proceed without a valid motor
  mapping.

These are measurements of the original ALPN-only, 50 ms configuration, not a
failure of the simulator installation. Changing the sensory route, plastic
population or motor architecture is a versioned model decision; thresholds were
not relaxed to authorize training. Earlier eight-state pilots at 50 and 100 ms
remain exploratory evidence and do not authorize a run on the final corpus.

### Resolved GPU calibration and tiny real training

ADR 0014 versions group-local allocation and selected-output reachability.
Competing pools stay disjoint, while independent comparison groups may reuse
neurons. Reward-free pair selection evaluates the averages actually read by
the decoder rather than assuming diverse individual neurons yield diverse
pools. The fixed neural window is now 500 ms; anatomy, sensory mapping and
plasticity rule are unchanged. No quality threshold was lowered.

The full 96-state, three-repeat pre/post representation reports pass with
separation ratio 1.321. Motor calibration passes with 47 eligible candidates,
31 unique selected neurons and action-type effective signal fraction 0.519.
Reachability passes, including 100% selected-motor activity. The 20-decision
integrated plasticity calibration passes every original safety gate.

`runs/tiny-real-resolution-20261009` completed 20 real Balatro decisions and two
episodes, recording eight update events. Its checkpoint changed 9,135 of
26,937 synapses (33.9%); all changes were decreases, mean absolute efficacy
change was 0.02338 and no synapse reached either bound. All metadata reports
zero external trainable parameters. Measured throughput was 0.735 decisions/s,
with 228 MiB peak Torch allocation.

`runs/tiny-real-resolution-no-plasticity-20261009` reproduced the exact
action/state trajectory with zero update events. The paired validation report
`artefacts/control-validation-resolution.json` passes. Checkpoint diagnostics
are in `artefacts/tiny-real-resolution-synapses.json`.

The independent 20-decision specificity diagnostic assigns 4.83% of absolute
weight change to chosen motor populations, 40.58% to competitors and 40.54%
to non-motor MBONs; the remainder is unconsulted motor populations. This
documents diffuse global modulation, not biological action-specific credit
assignment. These bounded development measurements do not demonstrate improved
Balatro performance or long-exposure safety.

Strict Ante-1 preflight for the 100-decision, one-replicate development pilot
passes with no warnings, and readiness reports `READY_FOR_ANTE1`.
`artefacts/ante1-resolution-pilot-protocol.json` binds the primary and exact
no-plasticity control. Materialization now anchors inherited artifact, corpus,
report, protocol and matched-source paths, so configs under nested
`artefacts/ante1-resolution-pilot-arms/` resolve the same inputs as their base.
The isolated profiled ten-decision benchmark measures 0.435 decisions/s;
unprofiled tiny training measures approximately 0.735 decisions/s. A 100-decision
arm therefore budgets about four minutes conservatively, with checkpoints every
20 decisions. Batch size one provides no evidence of a multi-fly speedup.

Both pilot arms completed 100 decisions and ten episodes. The real arm recorded
41 update events and changed 9,760 synapses (36.23%); mean absolute efficacy
drift was 0.10949, maximum drift 0.47495, all state was finite and bound occupancy
was zero. Runtime was 134.44 seconds (0.744 decisions/s). The control recorded
zero update events; `artefacts/control-validation-ante1-resolution-pilot.json`
passes exact matching. Checkpoints are under
`runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/`, with
synaptic diagnostics in `artefacts/ante1-resolution-pilot-synapses.json`.
Both training win rates were zero over ten episodes. Independent-action frozen
Ante-1 evaluation subsequently used the same 32 held-out seeds (20,000,000
through 20,000,031) for each checkpoint. Both won 0/32; trained/control decision
counts were 330/266. The paired mean return difference was -0.15434 (trained
minus control), not evidence of improvement. Episode scores are unavailable
from this simulator, not measured as zero. Results are in
`evaluations/ante1-resolution-{trained,untrained}-32/` and the comparability
report is `artefacts/paired-evaluation-ante1-resolution-32-audited.json`.

The learned 100-decision checkpoint also passed frozen full-corpus diagnostics
on its actual efficacy: `artefacts/representation-checkpoint-ante1-resolution-100.json`
and `artefacts/kc-reachability-checkpoint-ante1-resolution-100.json`. These
reports bind both checkpoint and weight hashes and use separate report kinds;
they cannot silently stand in for initial calibration.

The next protocol, `artefacts/ante1-guarded-1000-protocol.json`, declares three
stochastic replicates, paired exact no-plasticity controls and 1,000 decisions
per arm. Its validated execution plan uses 100/250/500/1,000-decision stages,
full-corpus learned-checkpoint audits after every primary stage, default-on
routine safety stops, and independent 32-seed evaluation for each completed
pair. Execution status is recorded in
`artefacts/ante1-guarded-1000-execution.json`; declaring the protocol does not
claim that all arms completed or that efficacy remains safe over long exposure.

The first replicate passed its 100-decision checkpoint representation and
reachability audits, then the resumed stage stopped at decision 172.
Lower-bound occupancy was 11.007% (2,965 of 26,937 edges), above the unchanged
10% safety limit; mean absolute drift was 0.17718. All measured state remained
finite. The trainer returned exit code 2, retained the routine-safe
160-decision checkpoint, and quarantined the stopped state in
`runs/ante1-guarded-1000-20261009/mapping-000-replicate-000-plastic_real/plastic-checkpoint-safety-stop.pkl`.
The execution report is `halted`; no controls, later replicates or later stages
were started. This validates real GPU stop propagation, not a completed
three-replicate experiment. The quarantined-weight audit is
`artefacts/ante1-guarded-1000-safety-synapses.json`.
Investigate the rule's mostly depressive response
before proposing a separately versioned sensitivity experiment; do not relax
the gate or resume the quarantined state to reach the declared budget.

Synthetic integration training did complete: 1,000 decisions, 492 episodes,
367 changed-weight events, checkpoint and 32 frozen evaluation episodes under
`runs/plastic-development-20261009` and `evaluations/plastic-development-20261009`.
Its exact matched no-plasticity control reproduced the action/state schedule
with zero plasticity events; the control-validation report passes. These mock
results establish software behavior only, not real Balatro learning. Evaluation
and synaptic analysis now accept a CLI training-budget override while model and
mapping checks remain strict; exact resume still checks the exposure budget.

## Current verification boundary

Real GPU calibration, native performance qualification and long experimental
training have now been run. Their evidence supports bounded execution,
checkpoint and calibration claims, with partial first-blind improvements in
some experiments. It does not establish reliable frozen Ante-1 competence or
complete the primary scientific control protocol.

The following remain outside the demonstrated boundary:

- generalisation of calibration beyond its corpus and model identity, or
  preservation of those properties after arbitrary long plastic exposure;
- downstream descending-mode readiness inferred from MBON-direct results;
- long-exposure stability for every rule: the repaired native long run stopped
  on saturation, while the separately normalised single-seed run completed;
- reliable membership learning, useful autonomous discard strategy, or a
  consistent local update that steers the desired card output;
- improved final-test Ante-1 performance under the declared primary protocol;
- a rendered real time-resolved showcase or final video.

See [the GPU runbook](GPU_RUNBOOK.md) for primary gates and
[the learning roadmap](LEARNING_ROADMAP.md) for experimental follow-ups.
Execution qualification and a scientific learning result are separate claims.

Two gate failures remain known in the synthetic development double:

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
