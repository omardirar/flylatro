# V1 research progress log

## 2026-10-09: authorized six GPU-hour continuation

Authority: attached end-to-end V1 request and PLAN.MD. Preserve the dirty
worktree, historical reports and checkpoints. No rule architecture changes,
new biological assumptions, relaxed safety gates, final-test tuning or cloud
resources are authorized. Only existing KC->MBON efficacies learn.

Initial inspection: working profile is `configs/plastic-v783-resolution.toml`.
The RTX 4080 is available; no compute job is running (`nvidia-smi`). Sandboxed
NVML cannot access it; the host check succeeds. The pre-change regression
suite passes 406 tests in 25.40 s. This continuation's GPU budget starts at
zero; previous experiments remain historical evidence.

Acceptance audit and outstanding evidence:

- Software: audit every status-table row against implementation/tests and real
  evidence; finish data-quality/weak-edge verification, CUDA residency/batching
  benchmark, real exact resume, recording/replay/render, compile and diff checks.
- Correct the offline bound comparison and regenerate safety/160-decision
  checkpoint evidence under new names. Regression must agree with live
  NumPy/Torch guards without counting adjacent interior floating-point values.
- Diagnose outcome pulse balance, eligibility at terminal update, update sizes
  and motor behavior. Check signs/order against an independent equation before
  attributing failure to the model.
- Predeclare LR 0.002/0.005 sensitivity against failed 0.02, first using exactly
  the same recorded neural trajectory, then independent actions. Recalibrate
  rule-dependent evidence. Select solely using declared stability gates.
- Three independently seeded 1,000-decision primaries and exact no-plasticity
  trajectory controls must complete with provenance and checkpoint audits.
- Freeze configuration and paired analysis before confirmatory held-out tests.
  Independent-action performance, uncertainty and reproducible Ante-1 winning
  replay are required. Survival, a single win or a negative/inconclusive
  comparison cannot establish learning or complete this goal.
- Update README/status/runbook with measured paths, commands and limitations.

Hypothesis H1: float32 efficacy clipped to 0.2 is promoted to float64 before the
offline bound comparison, yielding 0.20000000298023224 > Python 0.2. Fix:
compare exact dtype-representable bounds before promotion. Safety limits and
training updates remain unchanged. H2: frequent terminal failure pulses acting
on accumulated eligibility dominate small progress pulses; inspect and replay
the logged trajectory to distinguish correct depressive dynamics from a defect.

Next experiment: deterministic 172-decision guarded-trajectory reconstruction
from unit efficacy, preserving source actions, states and neural seeds. Record
activity once, compare all three rates on identical activity/pulses/resets, then
run bounded independent-action pilots. GPU usage will be recorded per command.

Corrected read-only forensic reports are under
`artefacts/lr-sensitivity-20261009/*-corrected-synapses.json`, bound to verified
checkpoint and weight hashes. The 160-decision checkpoint has 0 bound edges;
the quarantined 172-decision checkpoint has exactly 2,965 / 26,937 lower-bound
edges (11.0071649%), agreeing with live telemetry. Historical reports remain.
NumPy and CPU Torch exact-bound/adjacent-interior regressions pass.

Logged failed-run evidence: 46 progress pulses sum to 0.0927500 appetitive;
17 failure pulses sum to 10.2 aversive. Positive updates total 39.8479 absolute
efficacy units across edges; negative updates total 4,812.6582. Largest positive
edge update is 0.00085336; largest negative is 0.04800001, as predicted by
0.02 * 0.60 * 4. No blind/Ante clear pulse occurred. Play/discard/skip/select
counts are 68/58/28/17, plus one skip-pack; 39 of 68 plays selected four cards.
These data support H2 but equation/reconstruction verification is still needed.

Before measuring the sensitivity, `artefacts/lr-sensitivity-20261009/predeclaration.json`
locks LR 0.002/0.005 against 0.02, 250-decision independent pilots, unchanged
guards and full-corpus health requirements. Selection prefers 0.002 if stable;
0.005 is the fallback only if it passes and 0.002 does not. Reward/performance
is excluded from selection. It also declares three 1,000-decision primary/control
pairs, protocol seed 910009, stages 100/250/500/1,000, and a separate once-only
128-final-test-seed confirmatory comparison with seed-block paired uncertainty.
The smaller rate remains a stability hypothesis, never learning evidence.

Commands so far: `.venv/bin/python -m pytest -q` (406 pre-change PASS),
`.venv/bin/python -m pytest tests/test_plastic_analysis.py -q` (5 PASS),
`.venv/bin/python scripts/lr_trajectory_audit.py --forensic-only` (PASS),
`.venv/bin/python scripts/lr_trajectory_audit.py` (live GPU handle 36631).
Sandbox CUDA is unavailable; the same GPU command runs on the host after
approval. Never infer a stopped job from that sandbox restriction.

Trajectory reconstruction PASS: 172 exact action/state transitions; every
checked historical metric is identical (maximum error 0); the reconstructed
Torch final efficacy hash exactly matches the historical stopped checkpoint.
Independent NumPy/equation discrepancy is at most 1.1921e-7 (float32 rounding).
On identical recorded activity/outcomes/resets, LR 0.002/0.005/0.02 produces
mean absolute drift 0.0178141/0.0445352/0.1771842 and bound fractions
0/0/0.11007165. All three have 10,163 net-decreased and zero net-increased
edges. Correct implementation therefore does not resolve the depressive signal
imbalance. The lower rates address exposure stability only.

Resource accounting: reconstruction used 234.223 GPU wall seconds. The
conservative ledger `artefacts/lr-sensitivity-20261009/gpu-budget.json` records
each subsequent GPU subprocess (start, PID, command, log, duration, exit),
refuses unresolved live entries, and enforces the 21,600-second budget.

The initial orchestration mistakenly required the controlled all-edge stress
sequence to pass the real sparse-modification guard: both controlled reports
correctly FAIL with modified fraction 1.0 versus 0.80. They remain FAIL and are
preserved. The original real readiness gate uses integrated calibration; the
documented orchestration correction applies that gate without altering any
threshold. A second orchestration stop exposed the documented WARN exit code 1
for initial preflight without a protocol/control; its report has no FAILs.
Verified completed measurements are reused by the corrected continuation,
without replacing live processes or overwriting prior execution reports.
The active independent pilot handle is 98050, recorded execution is
`pilot-execution-independent-v3.json`.

Additional implementation gap found: preflight checked control pairs against
one another but did not bind their reference model to the new configuration.
It now rejects old-rule controls when LR/rule/anatomy/mappings/dynamics differ;
a real synthetic-pair regression proves matching controls PASS before, and FAIL
after, a rule change. New pilot controls will supply current-rule evidence.

Confirmatory analysis is implemented before viewing final-test outcomes:
three-fly effects are grouped by shared seed, an exact integer sign distribution
computes the one-sided p-value, and 20,000 fixed-seed bootstrap draws give a
95% interval. Zero discordance and a single lucky win fail the learning gate.
Inference is conditional on the three trained flies; it does not claim
population-level training-replicate uncertainty. Six inference tests pass.

Historical-graph independent pilots both completed 250 decisions safely and
passed full 96-state, three-repeat checkpoint representation/reachability:
LR 0.002 drift 0.030884, maximum drift 0.127719, 27 episodes; LR 0.005 drift
0.071859, maximum drift 0.297362, 25 episodes. Both have zero bound occupancy
and zero training wins. The declared stability preference selects 0.002.
Its 250-decision no-plasticity control passes exact action/state matching.
Pilot training used 340.388/340.739 seconds; checkpoint audits each used
165.08 seconds; the control used 333.469 seconds. Completed GPU charges total
1,697.861 seconds before the source correction's calibration. These are
historical-graph measurements, not final V1 anatomical-readiness claims.

Source audit uncovered a substantive V1 requirement gap. The pinned legacy
`connections_princeton.csv.gz` yields only threshold-5 KC->MBON pairs:
26,937 edges / 226,932 synapses. The already available unthresholded file
(SHA-256 `62c2e562a7470bfd32cbb98e36af3daa607ba0822b37223b34f2aa45250a920e`)
yields 67,504 / 314,975, with threshold-2/5/10 edge counts
52,463 / 26,937 / 6,973. Thus 40,567 weak anatomical pairs and 88,043 synapses
were absent despite `minimum_synapse_count=1`. Evidence:
`artefacts/lr-sensitivity-20261009/source-edge-coverage.json`.

PLAN.MD explicitly requires all source KC->MBON anatomy and primary minimum
count 1. Correcting the input is an implementation repair, not a new
biological/plasticity-rule assumption. The builder now defaults to the pinned
unthresholded source, records its threshold, supports explicit historical
`--connection-source threshold5`, and refuses nonempty output directories.
A fixture build proves weak edges survive and existing artifacts cannot be
overwritten. Preflight now rejects a filtered source as full V1 coverage.
The old artifact/calibration/checkpoints remain untouched; old readiness PASS
does not prove this newly audited source requirement.

Fresh CPU build: `.venv/bin/python -m flylatro.fly.build_flywire --source-dir data
--output-dir data/flywire-source-complete-v783 --connection-source unthresholded
--full`. It contains 19,773,733 whole-brain pairs, 76,944,499 synapses and all
67,504 KC->MBON pairs, with zero invalid-endpoint/unresolved-sign synapse drops.
The neuron/KC/MBON/ALPN/DAN censuses are unchanged. Artifact SHA-256 is
`656c7b2dffec24598629bee06f7d622c1dc6b84768f062f250f8345145ecb3a1`.
`populations-source-complete.json` revalidates the full census and weak edges.
The population report's duplicate version key was corrected: report schema and
materialization now have separate fields.

Fresh source calibration uses the same frozen 96-state corpus, 500 ms window,
three repeats, activity seed 12,000,000, batch 8, and all original thresholds.
The source-correction declaration names/hashes the config/artifact and retains
the original primary/confirmatory scope. Its initial command was already
running when the declaration was persisted; no source-corrected neural results
had been observed or used to select these parameters. Source-dependent reports
and motor mapping must be regenerated; source-independent LR equation findings
remain separate. Unit activity caching was extended to pre/post diagnostics
with exact provenance; learned checkpoints explicitly refuse that cache.
Active source-calibration handle: 3280. Script:
`.venv/bin/python scripts/run_source_complete_calibration.py`.

That source-calibration process completed and is not live. Full-corpus
representation at the original 0.275 mV / 500 ms fails unchanged high-rate
and separation gates (0.302120 > 0.25; ratio 1.0844 < 1.10). Reachability and
motor calibration pass, but no primary training follows this failure.
Conservative cumulative GPU charge at that boundary: 2,359.740 seconds.

Named fixed-dynamics sensitivity exposes the existing fixed LIF synapse gain
in configuration and provenance, preserving its default and the primary
plasticity architecture. `fixed-dynamics-predeclaration.json` fixes 0.18 mV
and windows 100/250/500 ms before measurements, with reward-free readiness
selection and no threshold changes. `.venv/bin/python
scripts/run_dynamics_sensitivity.py` is monitored through session 49049;
execution and resource evidence are `fixed-dynamics-execution.json` and
`gpu-budget.json`. The 100 and 250 ms representation/reachability reports PASS.
Their motor reports FAIL only action-type signal diversity: effective fractions
0.400035 and 0.486185 against 0.50. Other motor checks PASS. The declared 500 ms
full-corpus measurement remains running at this entry; no replacement job was
launched. Completed GPU charges: 2,832.343 seconds, with the active job charged
on completion or bounded termination.

Confirmatory inference correction, before any final-test observations:
`confirmatory-pairing-correction.json` preassigns one trained fly per seed using
RNG seed 910011. Its binary paired differences support an exact one-sided
McNemar/sign-permutation test; the previous test on a three-fly mean required
an unjustified symmetry assumption. All three flies still receive all 128
final-test seeds, and their mean plus seed-block bootstrap remain descriptive.
The primary paired mean must be positive, p < 0.05, and its bootstrap lower
bound > 0, with a positive all-three mean and all other acceptance gates.
Zero discordance reports an exact 95% discordance upper bound rather than a
misleading zero-width interval; it fails improvement. Six inference tests PASS.
Configuration/checkpoint/statistical lock will precede final evaluation.

Session 49049 completed: 500 ms pre-representation/reachability PASS, motor
FAIL (action-type effective signal fraction 0.465331). Thus gain 0.18 has no
ready candidate. Conservative GPU charge: 3,500.066 seconds.

Implementation repair found during diagnosis: motor and representation
correlation divided covariance by N-1 while standardizing with population
standard deviation. A four-sample regression against `numpy.corrcoef` failed
before repair (e.g. -0.8 incorrectly became -1); consistent N denominators
now pass. Motor schema is v5 and representation v4, forcing fresh affected
evidence. Thresholds are unchanged. Motor calibration also refuses existing
mapping/report destinations before expensive work.

`fixed-gain-followup-predeclaration.json` records corrected-cache regeneration
for gain 0.18's three windows, then fixed gains 0.20/0.225/0.25/0.275 at 100 ms.
The first all-gates passing configuration is selected by reward-free readiness.
Session 13650 runs the declared order through `run_dynamics_sensitivity.py`
with explicit `--plan`/`--output`; existing actual-GPU cache identities/hashes
are verified, not replaced. Corrected gain-0.18 audits still fail diversity;
gain 0.20 pre-representation PASS and motor FAIL. Gain 0.225 is running.

Full regression suite: 425 PASS in 28.08 seconds, compilation and diff checks
PASS. The new `flylatro.analysis.resume_cli` checks full future trainer states
after a fresh checkpoint: actions, environment state hashes, neural activity,
efficacy/eligibility/reinforcement, RNG and safety. Synthetic regression PASS;
source-complete real GPU audit remains required after calibration.

Session 13650 is terminal. The first passing declared candidate is gain 0.225 /
100 ms: full pre/post representation and reachability PASS; motor v5 PASS with
55 eligible candidates, 34 unique neurons, 70 memberships and minimum effective
signal fraction 0.511395. Integrated plasticity PASS. Initial preflight has no
FAILs; its expected development WARNs concern missing pilot/control/protocol.
The ten-decision benchmark measures 0.667 decisions/s, 477.54 MiB peak Torch
allocation. Lower-gain failures and unrun later candidates remain explicit.

Fresh independent source-complete pilot is predeclared in
`source-complete-gain0p225-100ms-lr0p002-pilot-predeclaration.json`, using the
original 250-decision pilot seeds and LR 0.002 selected by earlier stability.
Active session 53389 runs `scripts/run_source_complete_pilot.py`, monitored
without replacement. Primary training completed 250 decisions / 27 episodes /
106 update events, with no safety stop and zero wins. Wall time 325.870 seconds,
0.767177 decisions/s. Full learned-checkpoint health runs next, then exact
no-plasticity control and the new actual GPU full-trainer resume probe.

Additional benchmark reporting repair: backend component timers reset every
simulation, but the end-to-end report used only the last simulation divided by
all-decision wall time. v3 now accumulates all measured decisions and excludes
optional later comparison simulations. Batch/sequential comparison reports
actual activity agreement and maximum Hz difference as well as timings.
Regression proves accumulation and detects an intentionally different batched
output. Preflight now binds benchmarks to the full anatomy/interface/rule
identity; fresh current-model v3 benchmark is required before primary stages.
Protocol materialization refuses nonempty destinations, preserving evidence.

Source-complete pilot execution is terminal. The actual 250-decision primary
checkpoint passes full 96-state, three-repeat learned representation and
reachability. Its exact 250-decision no-plasticity control records zero updates
and PASS action/state matching. The first full-trainer resume probe failed raw
snapshot equality, despite all eight future steps matching. Pinned simulator
source confirms arbitrary HashMap serialization order in the leading RNG map.
Canonical audit identity sorts only this verified map and preserves all other
bytes, values and raw stored checkpoints. A separate adapter defect discarded
pre-step observations on restore, losing first-resume reward decomposition;
restore now initializes that state. The new prefix-3 / future-13 actual GPU
probe PASS includes positive progress on its first resumed step and terminal
failure/reset. Old failed evidence remains intact. CPU historical checkpoint
forensics show raw snapshot inequality but canonical equality and the original
first-resume pulse (decision 101) is correctly zero; it does not explain the
historical depressive collapse.

`source-complete-gain0p225-100ms-lr0p002-pilot-validation-v2.json` passes.
Session 9496 runs the original predeclared three paired 1000-decision primary
protocol, seed 910009, stages 100/250/500/1000. Fresh benchmark v3 measures
single-row 0.668335 decisions/s, 477.54 MiB; four-row 1.42364 decisions/s,
545.04 MiB and 2.1226x comparison speedup, with exact measured activity.
Recurrence accounts for 85.91% of profiled decision time. All six strict
materialized preflights PASS. The first primary passes its 100/250 health
audits; 500 decisions are complete and its full health audit is running.
Mean efficacy drift is 0.146826, bound occupancy zero, and training wins zero.
No held-out learning claim follows from this stability evidence.

`frozen-batch-execution-predeclaration.json` prospectively declares width four
as an execution optimization. Every distinct completed checkpoint must match
its already-required sequential 32-seed validation evaluation exactly in
complete per-seed actions, rewards, state hashes and episode outcomes. Frozen
weights, configuration, code and evidence files are bound before final use.
A disagreement rejects batching; final 128 seeds per trained fly, prospective
fly assignment, statistics and acceptance gates remain unchanged. Synthetic
regressions PASS: independent frozen replicas preserve trajectories and source
checkpoint identity; qualification detects changed actions/rewards/states,
incomplete trajectories and changed evidence. Current real qualification and
final evaluation remain unrun. Combined targeted tests: 28 PASS.

Full software suite PASS: 441 tests in 29.16 seconds after batch qualification
and executable documentation additions. CPU read-only 500-decision synaptic
audit verifies actual file/weight/topology identities: 41,996 net-decreased
synapses, zero net-increased, mean absolute drift 0.146826, no efficacy bounds.
This confirms that lowering LR slows persistent net depression; it is not
performance evidence.

Replay review found a target defect: the CLI forced Ante 8 even for bundles
recording Ante-1 evaluation. Verification now applies the persisted target and
shaping beta, checks step rewards/decomposition, and checks the declared win
against the actual terminal episode. Synthetic target-sensitive and reward
mutation regressions PASS. Spiking recordings may use one already-observed
unlocked final-test seed so a genuine winning episode can be recorded without
a new search or tuning. `scripts/produce_heldout_win.py` selects the first
trained replicate/declared seed with a real recorded win, checks locked files,
replays it in the actual simulator, optionally records real spikes, verifies
unchanged full trajectory, packages checkpoint/config and renders brain video.
It explicitly reports absence of genuine final-test wins and never invents a
replay. Current scientific replay/visualization remains conditional on an
actual win; the tool has not consumed any final-test outcomes.
Compilation and diff checks PASS.

Full expanded regression suite PASS: 444 tests in 29.39 seconds, including
all executable runbook commands and new Ante-target/reward replay checks.
The active session 9496 is confirmed live. First primary approaches its final
1000-decision checkpoint (995 observed), still zero bound occupancy and no
training wins; full final checkpoint health remains required before its control.

2026-10-09 17:35 UTC user steering: "ignore the gpu budget. i can run as long
as is needed." The local GPU duration ceiling is superseded by explicit
`gpu-duration-authorization.json`. `run_lr_pilots.gpu_command` now uses no
budget timeout for subsequent commands, while retaining accounting, existing
run/output refusal and unresolved-live-job protection. Original budget/history
are preserved. The already-running wrapper captured its original timeout; it
is kept running and is expected to complete within that deadline. No system,
cloud, publication or biological-architecture authority is inferred.
New duration-authority and held-out-win selection tests: 5 PASS.

The first current-source primary completes 1000 decisions / 110 episodes /
431 update events without a safety stop. Full final representation and
reachability PASS. Independent frozen validation finishes 0/32 Ante-1 wins
(253 decisions, 327.625 s, 0.7722 decisions/s). Its exact control is now running.
This is stable exposure with passing health, still no performance improvement.
Host process PID 370533 is confirmed live. Host `nvidia-smi` measures RTX 4080,
100% utilisation, 2979 MiB used of 16376 MiB during validation. Spare memory
alone does not establish simultaneous-process throughput. Current orchestrator
is serial; multi-process campaigns are unqualified. Existing four-row neural
benchmark has 2.12x measured speedup with exact sampled activity; full frozen
batch trajectory qualification is the next authorized execution optimization.
Primary/control seeds, exposure, scientific gates and final inference remain
unchanged. Longer exposure or more simultaneous jobs cannot by themselves
repair the observed depressive learning balance.

The training CLI now opens a live stderr dashboard in interactive terminals,
with `--dashboard`, `--no-dashboard` and `--dashboard-refresh` controls. It
shows each fly's current/completed runs, decisions, wins, game overs, Ante,
observed chip peaks and returns, alongside throughput, elapsed time, ETA,
plasticity health, whole-device GPU utilisation/memory/temperature/power and
this process's Torch memory. GPU sampling runs asynchronously and unavailable
telemetry does not block training. The pinned simulator omits final episode
scores, so those display as unavailable rather than inferred high scores.
Counters persist separately from scientific trainer state in checkpoints;
atomic `training-dashboard.json` snapshots and `training-episodes.jsonl`
records support inspection and staged resumes. Legacy checkpoints label
missing history. Final JSON remains on stdout and cursor/screen restoration
works on interruption.

Two-fly synthetic comparison verifies identical actions and learned weights
with display enabled or disabled; staged resumes preserve exact counters.
Resize, GPU failure, unavailable scores, auto-reset accounting and terminal
cleanup checks pass. A real PTY synthetic smoke run completes with the live
display and intact final JSON. The exact GPU sampling command also succeeds
on the host RTX 4080. These are dashboard/software checks, not real game-win
evidence. Full software suite PASS: 458 tests in 30.39 seconds. The existing
audited GPU protocol continues without launching a competing training job.

2026-10-09T18:02:50.708854+00:00 user requested all ongoing project runs be stopped and
a command for manually starting long training. Sent SIGINT to the verified
protocol process group 370533 (driver and control evaluation child 390611).
Host process scan confirms no remaining Flylatro jobs; remaining GPU users
are desktop applications. Existing checkpoints and completed reports are
preserved. Driver/execution status now records user interruption, and the
GPU ledger running entry is reconciled conservatively through this timestamp.
First primary and exact control each completed 1000 training decisions; the
control held-out evaluation and remaining replicate pairs are incomplete.
Validated CLI/configuration and local data paths for a fresh 100000-decision
Ante-1 single-fly run with checkpoints every 250 decisions and the dashboard.
No replacement training run was launched; the user will start it manually.

2026-10-09T19:20:13.774288+00:00 dashboard clarification and unattended-run preparation
Dashboard uses zero-baseline bars with a shared scale and explicit played/skipped
round counts; milestones remain separate from decisions, with scrollable history.
Full lightweight suite: 466 passed in 29.34 s; PTY CPU training and pinned
simulator skip checks passed; dashboard instrumentation preserves actions/weights.
Prepared batch4-training config: four independent flies, 100000 total decisions,
checkpoints every 1000 decisions, existing health stops and fixed Ante-1 target.
Red Deck / White Stake remains fixed. Local configuration/artifact checks passed.
User requested the command to keep the dashboard open; no new job launched.
This is exploratory exposure, not qualification of learning or the paired protocol.

2026-10-10T10:48:59.053613+00:00 exploratory training improvements
Exploratory recovery candidates after batch4 safety stop: preserved all original
protocol reinforcement identities and added separately named weak-failure
condition (failure pulse 0.01 vs 0.60; progress/clear/win pulses unchanged).
Five one-fly 1000-decision configs compare original, LR 0.0005, weaker
failure, combined, and exact-action no-plasticity control. Common seed and
motor settings, Ante-1 target, checkpoint cadence 250, and health limits
are preserved. JSON plan supplies train/eval argument lists, frozen
128-episode tuning validation, qualification requirements, and acceptance
criteria. New candidates have no benchmark/plasticity qualification
reports attached; no scientific protocol is rebound or overwritten.
Regression suite: 470 passed in 29.71 s (75 sandbox NVML warnings).
Tests preserve primary hash, enforce fixed candidate parameters, retain
terminal progress/failure and milestone semantics, and check weaker
depression on eligible edges. Compileall/config/CLI checks pass.
No GPU training or evaluation started; learning benefit remains untested.

2026-10-10T11:31:10.671446+00:00 within-game credit-window refinement
Confirmed terminal learning occurs before per-learner eligibility and
reinforcement reset; no cross-game trace leakage found. Added isolated
short-trace and combined-short-trace candidate configs with eligibility
decay 0.5 versus original 0.9. Plan now has seven matched 1000-decision
arms; within-game credit duration and reward/LR changes remain separable.
Passive NumPy/Torch telemetry reports net strengthened/weakened fractions
and worst-learner fraction within 5% of efficacy range of either bound.
No safety gate, plastic state format, trainable parameter, or original
rule changed. Tests verify direction/near-bound semantics and passive
state preservation across backends, plus shorter credit retention.
All seven config/command lists validate. Suite: 472 passed in 29.75 s
(75 expected sandbox NVML warnings); compileall and diff checks pass.
Candidates remain unevaluated; no new GPU run started.

2026-10-10T12:56:35.157211+00:00 autonomous recovery goal started
User authorized bounded GPU training/evaluation until a long-run candidate
qualifies, with no final long session launched automatically. Added frozen
initial-efficacy baseline mode, explicit source-versus-evaluated identities,
and per-episode actual clears/skips/played-round metrics. Source checkpoint
is verified and preserved; baseline replay bundles are suppressed because
the trained source checkpoint alone cannot reproduce the overridden state.
Full regression suite: 476 passed in 30.77 s, 78 sandbox NVML warnings.
Milestone tests cover in-place observations, unshaped clears, terminal
auto-reset and separate skips. CPU/Torch state/identity behavior preserved.
GPU first-evaluations stage live under driver PID 641441; real sequential
versus batch-eight parity passed for all 8 episodes and complete traces.
128-seed unchanged-weight then refined-checkpoint screening is running.
New qualification plan preserves final-test lock and all health stops;
requires original/unchanged-weight comparisons, independent training-seed
confirmation and 5000-decision durability before recommending a long run.
Prepared separate screening pilot stage; it has not been launched yet.

2026-10-10T13:02:13.163287+00:00 frozen baseline complete, pilots queued
Verified first-evaluation driver PID 641441 and live child on host.
Initial-efficacy baseline completed 128 identical-cohort validation
episodes: 0 wins, 0 actual blinds beaten, 1008 decisions; frozen state
and source checkpoint identities preserved. Refined evaluation is live.
Queued screening driver PID 646049 (exec session 81437) waits on the
verified live prerequisite and starts only after successful completion;
no duplicate/restarted jobs or concurrent GPU phases were launched.
Five serial separate-intervention pilots remain capped at 1000 decisions
each, with integrated calibration and 128-seed frozen comparisons.
Comparison now validates static anatomy/motor/dynamics and simulator
seed identities, independent actions and finite outcomes, while allowing
the explicitly declared rule changes. Seven focused comparison tests
passed; compileall and diff checks pass. No candidate is qualified yet.

2026-10-10T13:12:10.859667+00:00 refined frozen comparison complete
Verified exec session 44737 completed normally; baseline/refined each
finished all 128 identical-cohort held-out games with 0 wins and 0 actual
blinds beaten. Mean return difference refined minus initial: -0.0057075.
Actions/episodes differ, so tiny weight changes do affect behaviour, but
the combined-short-trace candidate is explicitly NOT QUALIFIED.
Saved immutable candidate assessment with no success claim. Goal remains
active. Queued driver 646049/session 81437 advanced to original-settings
integrated calibration after successful prerequisite completion.
Separate reward/LR/trace interventions continue serially; no final long
session or final-test seeds launched.

2026-10-10T13:19:50.883172+00:00 CPU frozen trace audit and behavioural diagnosis
Verified screening driver live via session 81437; original-settings
1000-decision training advances with finite activity and no saturation.
Added CPU-only trace replay audit, independent of learning. Reproduced
all 2026 transitions from 256 initial/refined frozen games with exact
pre/post observable and terminal hashes. No hidden identity was read
into policy, and no actions, rewards or original traces were changed.
Initial/refined each made 512 plays. Available visible rank pairs:457/456;
selected visible rank pairs:72/61. Two/three-card plays dominate. These
are descriptive visible patterns, not optimal-play labels or scored
poker categories. Further credit refinement remains an experiment, not
an asserted fix. Pinned source constructor confirms Red Deck/White Stake.
Saved audit artifacts and behaviour diagnostics; nine focused audit/
comparison tests pass, including refusal of corrupted state evidence.
Compileall and diff checks pass. No long-run candidate qualified yet.

2026-10-10T13:25:18.573824+00:00 reference monitored; contingent exploration hypotheses
Verified live screening exec session 81437; reference training reached
566/1000 decisions, 62 completed games, 0 wins, bound fraction zero,
mean absolute drift 0.1625 (original stop limit 0.5). No restart or
long-session launch. Other separate interventions remain queued.
Inspected decoder: stochastic exploration applies to action types,
counts and target heads; card slots always use greedy ranking. This is
a candidate exploration limitation, not an asserted protocol defect.
Recorded conditional hypotheses for separately identified card-slot
exploration, selected-motor credit, and activity-reference diagnostics.
They are not implemented or qualified; current runs and original
protocol identities remain untouched. Existing milestone failures
continue to require independent testing, not a stability success claim.

2026-10-10T13:42:49.116898+00:00 frozen learner selection and reference reproducibility
Original-settings training completed all 1000 decisions without a health stop.
Verified all 1000 actions/state hashes and final weights exactly match the
first lane of the earlier four-fly training exposure; this is reproducibility,
not independent-seed confirmation. Its held-out frozen evaluation is live.
Added explicit zero-based learner selection for multi-fly frozen evaluation;
source checkpoint verification and full-source versus selected-row hashes are
preserved. No weight averaging; unsupported replay bundles are suppressed.
Focused tests pass and full regression suite: 488 passed in 31.55 seconds
(81 sandbox NVML warnings). Compileall and diff checks pass. Real GPU row
selection/parity remains pending; no long-training candidate qualified.

2026-10-10T13:44:18.313977+00:00 original frozen reference complete; weaker failure live
Original 128 held-out games: zero actual blinds and zero Ante-1 wins.
Original mean return gain against initial weights 0.00402344 does not
qualify learning. Refined versus original mean actual blinds gain zero.
Saved completed-reference-assessment.json; neither candidate qualifies.
Supervisor session 81437 is verified live and advanced normally to weak-
failure calibration (passed) and its 1000-decision training. Short trace,
lower LR and combined remain queued. All health stops and final-test lock
remain in force; no final long session launched.

2026-10-10T13:50:05.552004+00:00 weak-failure first checkpoint diagnostic
Verified live supervisor session 81437; weak-failure training exceeded 250
decisions and saved its immutable checkpoint. Verified checkpoint SHA256
for both original and weak-failure at identical 250-decision exposure.
Saved weak-failure-checkpoint-250-diagnostic.json with finite activity/state,
weight drift, bound fractions and observed action/state divergence. Actions
first diverge at decision 10, observed pre-action state at 11. This is rule
sensitivity, not proof of better play or independent-seed confirmation.
Frozen 128-seed comparison remains pending; goal active, no final long run.

2026-10-10T13:51:06.457203+00:00 original frozen CPU replay and card-selection comparison
Verified all 1007 transitions across 128 original-settings frozen games,
including exact pre/post state hashes and terminal boundaries. Original
plays 211 one-card, 178 four-card and 123 five-card hands; 118 selected
plays contain visible rank pairs versus 470 opportunities. Initial/refined
were dominated by two/three-card plays and selected 72/61 pairs. All three
beat zero blinds, so simply increasing card count/pair selection is not
proven sufficient. Recorded cross-arm diagnostic without changing rewards
or policy, accessing hidden identities, or treating replay as learning.
Weaker-failure pilot remains live under session 81437.

2026-10-10T13:58:25.553761+00:00 pulse-to-weight credit magnitude diagnostic
Recorded matched 500-decision exposure diagnostics for original, weaker-
failure and combined-short-trace arms. Checkpoint hashes verified; measured
signed event-update sums agree with independently summed checkpoint weight
changes (rounding differences below 0.000001). Original positive/negative
pulses total 0.435/32.4 and weight changes 114.23/9665.15. Weak-failure pulses
0.493/0.560 produce changes 123.76/166.13; combined-short-trace pulses
0.564/0.580 produce changes 19.02/20.58. Pulse balance therefore does not
equate update balance. Trajectories differ, so no activity-matched causal
claim or success inferred. Separate pilots continue; no health guards or
scientific protocol changed. Supervisor session 81437 verified live.

2026-10-10T14:06:30.923703+00:00 weaker failure completed; frozen evaluation live
Supervisor session 81437 verified live throughout the final training phase;
training finished with exit 0 at exactly 1000 decisions and completed status.
Final checkpoint SHA256 verified; finite weights with no health stop.
Saved weak-failure-training-completion-audit.json. Frozen 128-game validation
started normally on the matched cohort; no gameplay success inferred from
weight health. Next shorter-trace/lower-LR/combined arms remain queued.
No final long session or locked final-test seed used; goal remains active.

2026-10-10T14:06:46.806639+00:00 weaker-failure late training blind clear
Verified a blind_clear reinforcement event at decision 984 (progress plus
0.4 clear pulse), with no other clear/success event in all 1000 records.
Final healthy weights: 51.29% strengthened, 9.99% weakened relative to
initialization; mean absolute drift 0.00051719. This late training event
is not held-out improvement or independent replication. Frozen validation
is live; first 8/128 games reported zero wins. Kept milestone evidence
separate from long-run qualification; all health stops remain unchanged.

2026-10-10T14:15:39.827253+00:00 weaker-failure held-out screen failed; short trace training live
All 128 matched validation games completed with zero actual blinds beaten
and zero Ante-1 wins. Primary gain versus original and unchanged weights
is zero; return gains 0.00969184/0.01371528 do not qualify a candidate.
Verified evaluation source checkpoint hash against completed training.
CPU replay reproduces all 1004 transitions and terminal boundaries exactly.
Visible pair opportunities 462 versus 65 selected pair plays; descriptive
only, no policy teacher or reward changes. Saved immutable candidate
assessment and replay audit. Supervisor session 81437 advanced normally:
short-trace integrated calibration passed, 1000-decision training live.
Lower-LR and combined remain queued. Goal active; no final long run.

2026-10-10T14:21:47.097739+00:00 shorter-trace first checkpoint verified
Supervisor session 81437 verified live across 240-second monitoring window.
Short-trace training passed 250 decisions without a health stop. Verified
checkpoint SHA256 and internal 250-decision counter, finite weights and
saved equal-exposure comparison to original in short-trace-checkpoint-250-
diagnostic.json. Reduced drift alone is not qualification; actual blind
gains and independent-seed confirmation remain required. Training continues.

2026-10-10T14:27:26.126438+00:00 shorter-trace halfway credit diagnostic
Verified live supervisor across 280-second monitoring interval; pilot passed
500 decisions without health stop or wins. Checkpoint SHA256, internal count
and finite weights verified. Saved pulse/update magnitude diagnostics and
independent checkpoint signed-weight sum; measured totals agree within
rounding tolerance. Shorter traces still produce punishment-dominated updates;
no gameplay gain inferred. Frozen screening and other queued arms remain
required; scientific protocol and health guards unchanged.

2026-10-10T14:35:17.418703+00:00 all project sessions stopped at user request
Cancelled monitoring exec cell 288; suspended supervisor to prevent queue
advancement, interrupted active training child, then let supervisor record
exit -2 and finish. Host process scan confirms no project training,
evaluation or experiment supervisor remains. Project GPU allocation released.
GPU ledger has no unresolved command. Short-trace stopped at decision 884;
saved checkpoints through 750 and all prior artifacts preserved. Pending
lower-LR/combined arms cancelled. Recorded user-requested-stop.json and
annotated execution status stopped-by-user; no candidate qualified.

2026-10-24: card-decision experiments and learning roadmap

Added `docs/LEARNING_ROADMAP.md` with short-, medium- and long-term work and
explicit advancement criteria. Added the isolated `scripts/card_learning/`
harness and `scripts/run_card_learning.py`: native replay audit, stateless
card-membership exploration, independent plastic/frozen factorial controls,
rank-balanced native one-pair curriculum, frozen pair benchmark and full-game
transfer evaluation. Actual executed selections receive choice credit; forced
task heads do not. Task boundaries clear traces without a false failure pulse.
Checkpoint identity includes sampler/task/corpus and research/source hashes.

Verified 84 new and related tests, including exact resume and native scoring.
Completed a four-condition synthetic-CPU orchestration smoke run. Replayed all
5,736 transitions in the three completed 20k validation evaluations and
reproduced the prior mistake counts. Prepared 104 training/52 validation pair
snapshots and both real-FlyWire pilot declarations under
`artefacts/card-learning-20261024-v1/`; no new FlyWire training was launched.
The existing separated long run continued, and its guarded source hash remained
unchanged. Separate PLAY/DISCARD preferences and reward/clipping changes remain
planned experiments, not implemented changes to the current learner.

2026-10-26: completed card-credit direction and fixed-hand probes

Added independent `scripts/audit_card_credit_direction.py` and
`scripts/probe_fixed_pair_learning.py`, leaving production and the hash-bound
completed pair harness unchanged. The real direction audit covered 13 training
hands, two source seeds, initial/trained weights and actual/scaled positive and
negative updates. Exact-repeat, no-update and weight-restoration controls all
passed; source plastic state stayed unchanged. Every selected MBON contribution
clipped at 20 Hz, and roughly 65–68% of selected KC contributions clipped.
Actual positive updates left 103/104 source-seed score gaps and 104/104 held-seed
gaps unchanged. Amplified responses were mixed rather than globally inverted.

The real fixed-hand probe used matched plastic/frozen copies, 5k decisions per
lane and 128 before/after evaluation seeds. Plastic/frozen lanes encountered
the pair 51/50 times during the probe; final greedy evaluation was 2/128 versus
3/128, against 3/128 initially for both. No memorization gain was demonstrated
on this one hand/budget. Frozen choices/weights and source state remained exact.
Nine new diagnostic tests and related regression checks passed (63 total).
Details and limitations are in `docs/CARD_CREDIT_DIRECTION_AUDIT.md`; artifacts
are `artefacts/card-credit-direction-20261026-v1/` and
`artefacts/fixed-pair-learning-20261026-v1/`. Reference scaling with update size
matched and a separate outcome-contrast test remain proposed experiments.

2026-10-26: completed matched reference-scaling comparison

Added independent `scripts/compare_card_reference_scaling.py`, preserving the
completed pair harness, core implementation and checkpoint identities. Fixed
reward-free calibration used all 20 legal card-play states and two spike seeds
per state. Unique-neuron 95th-percentile references were KC 60 Hz / MBON 170 Hz;
exact expected uniform-pair coactivity matching set learning rate 0.00423708.
Rewards, mappings, exploration and initial unit efficacies stayed fixed.

After native GPU qualification, the declared three-lane 5k fixed-hand comparison
completed in 344 seconds. Initial evaluation was 3/128 for each lane; final
baseline/rescaled/frozen counts were 2/1/3. Training pair plays were 51/49/50.
Clipping decreased substantially, but no learning gain was demonstrated on this
hand/budget. The rescaled lane accumulated 20.4% more actual update L1 despite
matching calibration expectation; report this limitation rather than claim
perfect trajectory matching. Frozen choices/weights and source state/files
were exact, and the baseline reproduced prior blocks and final choices.

Fourteen new calibration/comparison tests and related regressions passed
(77 total). Artifacts: `artefacts/card-reference-scaling-20261026-v1/`.
Method, command, measurements and limits: `docs/CARD_REFERENCE_SCALING.md`.
The roadmap now prioritizes a separate native-score reward-contrast experiment,
with references/exploration fixed. Rescaled references are not promoted.

2026-10-26: completed separate native-score reward-contrast comparison

Added `scripts/compare_card_reward_contrast.py`, preserving existing source,
harness and checkpoint identities. The declared exploratory transform centers
native progress around the mean of 128 frozen-policy native outcomes and
matches mean absolute calibration pulse with a fixed scalar. This is explicitly
outcome-based feedback calibration, separate from reward-free motor/sensory
calibration; no pair labels or unchosen-action scoring enter feedback. Original
20/20 Hz references, 0.002 learning rate and 5% exploration remain fixed.

After native GPU qualification, the three-lane 5k experiment completed in
332 seconds. Initial evaluation was 3/128 for each lane; final native/centered/
frozen counts were 2/3/3. Training pair plays were 51/60/50. Centered feedback
delivered 2,990 positive and 2,010 negative pulses and changed weights in both
directions, but did not demonstrate gain over its initial/frozen policy.
Realized absolute pulse sum was 26.3% larger and cumulative weight-change L1
18.3% larger than native, despite matching calibration expectation. Post-run
diagnostics found no pair discoveries in the frozen calibration sample;
report its limited reward-tail coverage, not perfect magnitude matching.

Native baseline blocks/final choices reproduced the previous probe. Frozen
weights/choices and source state/files remained exact; no pulse/efficacy bound
was hit. Sixteen new tests plus related regressions passed (93 total), including
independence of feedback/learning from diagnostic pair labels. Artifacts:
`artefacts/card-reward-contrast-20261026-v1/`; method/results/command:
`docs/CARD_REWARD_CONTRAST.md`. Next: broader uniform subset exploration with
original native feedback and references fixed; neither reward centering nor
rescaled references is promoted to long training.

2026-10-27: separate PLAY/DISCARD card preferences and choice credit

Added versioned `action-conditioned-card-routing-v2`, with independent DISCARD
membership/count heads and matching executed-action credit. The mapping rejects
overlap between all core groups and isolates both card populations from
noncard pools. Existing serialized mappings retain unchanged identities and
semantics; old weights cannot be loaded as the new layout. Source/configuration
files were preserved under `artefacts/source-snapshots/pre-action-conditioned-20261027-v1/`.

Width-two real allocation failed: only 59 context-eligible candidates were
available across the active core, which needed 86 active memberships (90 total).
The new exploratory mapping explicitly declares single-MBON options and 45
exclusive core neurons. Minimum pool-width qualification is explicitly one;
response, diversity, dynamic-range, silence and coverage limits remain
unchanged. Primary legacy/preflight width defaults remain two, with a separate
explicit preflight override provided for this layout.

All 691 regression tests passed, including native CUDA and new routing,
NumPy/Torch update isolation, serialization, legacy checkpoint rejection and
exact synthetic training/resume checks. Final native qualification:
`artefacts/play-discard-motor-20261027-v4/`. Reward-free calibration passed;
13 opposing-choice fixtures passed native actions; 104 inactive-action update
checks passed. Actual same-state/same-noise choices differed on 51/52 trials,
and same-count top-two sets differed on 44/52. All plastic state was restored.
These are interface/credit tests, not a learned discard-strategy result.
Real FlyWire/native trainer resume also passed exactly for an eight-decision
prefix and eight future decisions; its test-only report/checkpoint are in the
same qualification directory.

Also repaired `.venv` by pinning its base configuration and interpreter link
to Python 3.14.8; the moving `latest` link had switched it to 3.15 while installed
packages remained under 3.14. Method, configuration and migration boundary:
`docs/PLAY_DISCARD_INTERFACE.md`. Fresh mixed PLAY/DISCARD learning now takes
priority over another legacy shared-head exploration run.

2026-10-27: assessed first separate PLAY/DISCARD learning pilot

The user-run `runs/play-discard-learning-pilot-20261027-v1` completed all four
conditions: 5k decisions per learner, three learners each, 60k decisions total.
Final checkpoint hashes, exact learner counts, finite state, zero efficacy
bound hits and unchanged frozen weights were verified. Initial policies had
3 blind clears/zero wins per arm across repeated 128-root evaluations.
Subset-plastic cleared 5 and won once; top-k-plastic cleared 2 and won zero.

The win was subset learner 2, root 20070062 / native seed PYYEE9EM, with three
blinds beaten in 26 decisions. Native replay verified it. It used zero discards.
A broader CPU audit verified 4,688 transitions across six trained plastic
evaluations plus a frozen baseline, and 7,630 alternative native plays. All
763 audited first-blind failures among plastic learners left discards unused.
The interface fix therefore has one encouraging observed win, without a
reliable learning/discard-quality or layout-causality claim.

Saved `assessment-v1.json`, `decision-audit-v1/` and `winning-episode.jsonl`
under the run. Prepared, but did not execute, a frozen confirmation plan for
all six trained plastic learners on 512 fresh roots (20120000–20120511), with
one equivalent initial-unit-weight control; every pilot initial trajectory was
byte-identical. It tests seven policies/3,584 episodes without more training.
Command: `bash runs/play-discard-learning-pilot-20261027-v1/confirmation-plan-v1/run.sh`.
Results/limits are in `docs/PLAY_DISCARD_LEARNING_RESULTS.md` and the roadmap.

2026-10-27: assessed completed fresh separate-head validation

The user executed the prepared frozen confirmation. All six trained policies
completed 512 shared fresh roots plus one equivalent initial control (3,584
episodes total). Trained policies had zero Ante-1 wins in 3,072 episodes;
subset cleared 4 blinds, top-k 6, initial zero across its 512 episodes. Mean
return gains were small (+0.002448 subset, +0.009452 top-k), with one subset
learner declining. The original single win did not reproduce on this block.

Native replay verified every transition in all seven evaluations. Among
trained policies, 3,061/3,063 first-blind failures left discards unused.
Training-log joins show DISCARD was actually tried on ~28% of PLAY/DISCARD
decisions, versus 3.49–4.23% in greedy frozen evaluation. All training DISCARD
events had zero immediate synthetic pulses; delayed eligibility credit remains
possible. Training samples softmax outside 5% uniform epsilon exploration,
whereas evaluation uses argmax. These identify sampling/credit hypotheses,
not an established unique cause of weak performance.

Saved fresh `assessment-v1.json`, replay audit and execution status. Next:
sampled-versus-greedy frozen diagnostics and action-score margins, followed by
a short native discard→play capacity task with traces spanning both actions.
No further training, reward tuning or GPU experiments were launched here.

2026-10-27: implemented and ran both discard-policy studies

Added independent scripts `evaluate_motor_sampling.py`,
`train_discard_play_curriculum.py`, `assess_discard_curriculum.py` and
`evaluate_discard_curriculum_transfer.py`, plus delayed-credit, frozen-state,
label-isolation and transfer-state tests. Eight new tests and related
regressions passed (73 total). Production sources, the hash-bound pilot
harness, mappings and original checkpoints were unchanged.

Frozen greedy/training-sampled evaluation completed 2,048 episodes on 128
shared new roots. Trained discard fractions were ~4% greedy versus ~28–29%
sampled; initial weights show the same difference. Sampled trained mean
returns were below their own sampled initial controls in both arms. No Ante-1
wins; native replay verified all 12,105 transitions. Results:
`artefacts/motor-sampling-20261027-v1/`.

The curriculum completed 5,000 DISCARD→PLAY trials per learner with three
plastic/frozen pairs (60k native decision exposures), fresh unit weights,
104 training and 52 validation hands. The subsequent PLAY policy stayed at
initial weights; only delayed DISCARD card/count credit could learn. Frozen
weights/choices and all non-DISCARD weights stayed exact, no efficacy bounds
were hit, and the trace-cut negative control eliminated updates. Three
learners' greedy held-out progress declined; sampled progress changes were
mixed/flat. Mean changes were −0.00223 greedy, −0.00101 sampled, without a
consistent advantage over same-count random membership. Counts increased by
~0.21 cards; pair retention fell. Post-hoc native replay verified 4,992 PLAY
outcomes. The fixed PLAY policy captured ~30% of available immediate progress,
limiting the outcome signal; best-hand values never entered training.
Results: `artefacts/discard-play-curriculum-20261027-v1/`.

Free-action transfer completed 512 episodes on 64 further shared roots, both
policies, initial weights and all three trained learners. Trained mean return
was 0.24877 versus 0.23567 initial greedy, 0.25530 versus 0.25099 sampled;
one learner declined in each mode. Greedy clears were 4/192 versus 1/64
initial; sampled zero; all Ante-1 wins zero. Greedy discard fraction remained
3.58%. Native replay verified 3,025 transitions and every episode return/end.
Results: `artefacts/discard-curriculum-transfer-20261027-v1/`.

The sampling discrepancy is established, but useful discard learning is not.
Keep both labelled frozen evaluations; do not extend the current runs or tune
punishment on this evidence. Next isolate PLAY competence, feedback
informativeness and discard count versus membership. Full methods, limitations
and reproduction: `docs/DISCARD_LEARNING_STUDIES.md`. Roadmap updated.

2026-10-27: worked through mechanism, exploration and stronger-PLAY screens

Added independent `scripts/run_training_bottlenecks.py`,
`scripts/assess_training_bottlenecks.py` and
`scripts/confirm_subset_exploration.py`, plus 14 new tests. The final targeted
regression set passed 61 tests. Original production sources, completed harnesses,
mapping and source checkpoints stayed unchanged. All screens started fresh
unit efficacy with three replicate IDs, two treatments and paired frozen lanes.

The simple two-choice action-head diagnostic completed 3k decisions per lane.
Counterbalanced PLAY/DISCARD targets each received an artificial 0.05 pulse
only when chosen; this is separate from native rewards. All six trained copies
increased their target's probability on held neural seeds. Mean PLAY target
probability rose 72.44% → 82.24%, DISCARD 27.56% → 34.37%. This qualifies a
bounded action-head learning effect, not card membership or native gameplay.

The native pair screen completed 5k trials per lane at 5%/25% uniform subset
exploration, fixed PLAY/two cards and unchanged native feedback. The 25% arm
had a small greedy validation gain (11 → 18 pairs / 312), versus 14 at 5%.
Training discoveries were 534 at 5%, 516 at 25%, with mixed case coverage.
All policies then received frozen greedy confirmation on 104 further hands,
four repeats. Initial/5%/25% totals were 47/37/48 pairs out of 1,248 trials
each. Higher exploration beat its 5% trained counterpart in all learners,
but did not improve over initial weights in aggregate; native mean progress
was also slightly below initial. Do not promote the higher rate from this.

The stronger-follow-up screen completed 2.5k DISCARD→PLAY trials per lane.
Only delayed DISCARD card/count credit learned. A visible-current-hand
reference PLAY policy raised greedy held-out progress 0.36835 → 0.41660 and
discard count 2.67 → 4.92; all three learners' progress increased. Same-count
random membership improved 0.36498 → 0.40793, almost as much. Membership
advantage changes were mixed and whole-case intervals included zero, as did
the aggregate progress-change interval. The reference's native feedback
produced ~25–26 times the cumulative update L1 of weak neural PLAY; outcome
quality and realized update strength are not separated. Neural follow-up
greedy progress declined 0.07038 → 0.06718. No reference player was installed
in production and no failure punishment was applied in these capacity tasks.

Independent replay verified 19,968 native actions across the main pair screen,
discard screen and fresh confirmation; native scoring also checked 39,984
follow-up PLAY outcomes during the discard run. All source/frozen/unrelated
weight checks passed, final NPZ counts/drift/traces verified and zero bounds
hit. One initial binary startup failed before model construction because of
a mistyped CXX path; its status is retained under `binary-startup-failed`.
All completed runs used the qualified GCC 14 compiler paths.

Outputs: `artefacts/training-bottlenecks-20261027-v1/`.
Methods, scripts, limits and commands: `docs/TRAINING_BOTTLENECK_STUDIES.md`.
Roadmap: isolate actual card-output reward effects and fixed-count native
membership, with update-matched controls, before long training or reward tuning.

2026-10-27: completed direct card-output and fixed-count membership tests

Added independent run/assessment/fresh-confirmation scripts for card
membership, plus activity and motor-learning-opportunity audits. All 72
targeted tests passed. Original production sources, completed harnesses,
mapping/configuration and source checkpoints were unchanged.

A tool-server restart killed the original binary job after its 1,750-trial
report with no saved learner state. Preserved v1 partial logs/status and
restarted the identical protocol under
`artefacts/card-membership-20261027-v2/` via a transient user service independent
of the tool server. Repeated prefixes were not counted as extra replicates.
The uninterrupted binary run completed all 3k decisions per lane (909 seconds).
Five of six DISCARD-card output targets learned: aggregate exact target
probability 50.00% → 77.58%. One copy received 1,070 rewards and zero updates.

Post-hoc forced-target probes identified why: its MBON10 output root
720575940629529722 fired at 35 Hz mean, but all 121 plastic KC afferents were
inactive and fresh eligibility stayed zero. The same active DISCARD option
had zero coactivity across all 15 applicable cases/two repeats in the existing
96-state reward-free calibration corpus. Its efficacies stayed unit in all
nine native learners. Raw audit flags for disabled SKIP_BLIND/unassigned
candidates are not active policy defects. Added tests distinguish silent
outputs from inactive plastic inputs and verify actual fresh-rule aggregation.

The native screen completed 2k DISCARD→PLAY trials per actor, count fixed at 2,
three learners per condition: neural feedback, reference feedback, reference
attenuated by a frozen initial expected update match, plus shared frozen
actors. The matched gain was 0.0904121; actual update L1 was 21.84–24.18 weak,
294.60–315.34 reference and 24.21–25.54 matched. Source/frozen/count/head/state
checks passed with zero bounds. Native screen replay verified 19,584 actions.

Fresh greedy confirmation on 104 new hands/two repeats verified 7,488 more
native actions. Reference-feedback actors scored with the reference improved
0.36199 → 0.37807, positive in all three, with conditional case interval
[−0.00344, +0.03469]. Matched feedback changed reference progress only +0.00195,
with mixed learners. None improved mean progress under the weak neural PLAY
policy; the earlier positive cross-policy screen did not reproduce. These
controlled results are not qualified full-game competence.

Methods/results: `docs/CARD_MEMBERSHIP_LEARNING.md`; output root above.
The concrete next priority is reward-free active-plastic-input qualification
for each active motor option, followed by a versioned mapping/fresh learners.
Output firing alone is insufficient. Keep primary checkpoints/defaults and
do not substitute punishment tuning or long training for fixing zero credit.

### 2026-10-28 — Active-input-qualified motor repair and fresh validation

Implemented `scripts/learnable_motor.py`, the generation/validation CLI
`scripts/qualify_learnable_motor.py`, and independent saved-state/referee audit
`scripts/assess_learnable_motor.py`. New top-level research modules preserve
all existing production/harness identity hashes and checkpoint resumes.

The reward-free rule requires positive fresh-rule KC/MBON eligibility in at
least half of competing legal samples per option and interpretation context.
It excludes disabled blind skipping, forced-only heads and padding. The
width-one repair ranks unused candidates by neural response, filters before
allocation and retains all original response/diversity/coverage/exclusivity
gates. It never uses game outcomes or diagnostic learning results to select.

Versioned artifact: `artefacts/learnable-motor-20261028-v2/`. Replaced DISCARD
card slot 3 root 720575940629529722 with 720575940620715224 and shared target
slot 4 root 720575940614595218 with 720575940628783363. Other assignments and
normalizations are unchanged. The target change removes one action/target
reuse, raising unique assigned neurons 46 → 47 while retaining 45 exclusive
core neurons. The unvalidated v1 draft is preserved and superseded; it is not
an extra training replicate.

Validated the fixed candidate with fresh unit-weight learners and the original
binary corpus/seeds: 1,500 choices, three counterbalanced plastic/frozen pairs
per target, 64 evaluation neural repeats, +0.05 diagnostic reward, original 5%
subset exploration. The old stuck slot remains 32.19% → 32.19%, update L1 0;
new slot improves 52.97% → 90.08%, update L1 86.1554. All six new target
preferences improve, with slot 1 small. Counterbalanced mean rises 50.00% →
69.79% new versus 50.00% → 64.10% old. Different initial preferences around
slots 2/3 require own-initial/frozen comparisons. Other two slot-pair replicate
states are bit-identical across mappings; source/unrelated/frozen weights and
all fast-state/bound checks pass.

Reward-free validation covers 192 independent game states plus 96 predetermined
calibration-display permutations for inventory contexts, two neural repeats.
Separate scope checks pass: 28 active competing roles in fresh games, 58 in
inventory display fixtures. Original checkpoint load is rejected under the
new mapping; it remains valid with original components. Native legality and
credit isolation pass; independent referee replay verifies 18 interface actions.
All 83 targeted tests pass. Both original subset/top-k checkpoint SHAs remain
unchanged. The transient user service completed; no study remains running.

Methods/results: `docs/LEARNABLE_MOTOR_MAPPING.md`; configuration:
`artefacts/learnable-motor-20261028-v2/config.toml`. Next compare native
fixed-count discard membership with fresh learners on this mapping. This fixes
the demonstrated output blockage, not yet native strategy or full-game wins.

### 2026-10-28 — Repaired mapping native comparison completed; scale-up gate fails

User authorized the short native comparison and asked when long training would
be viable. Added independent runner `scripts/run_repaired_native_membership.py`
and full training/weight/decision auditor
`scripts/assess_repaired_native_membership.py`. Historical production and
research modules remain unchanged; no checkpoint was loaded under new routing.

Run: `artefacts/repaired-native-membership-20261028-v1/`. Retained the completed
legacy native study's 2,000 trials, training corpus/seeds, fixed DISCARD count 2,
native rewards, 5% subset exploration and three seeds. New mapping starts with
unit weights. Feedback roles are neural-native/reference-native/reference
attenuated by a training-only initial L1 match/shared-frozen. New matching gain
0.0893301; realized L1 ranges 24.24–24.84 / 313.16–328.52 / 26.11–28.03.
Training completed in 1,539 seconds under a transient user service.

All nine plastic actors update 896/942 afferents at the repaired output;
frozen actors update none. Every plastic trial has positive native feedback
and positive eligibility. The initial screen's strong-reference neural-PLAY
gain (0.06899 → 0.07806) does not reproduce on the fresh set.

Fresh comparison: 104 balanced hands at new validation offset 420000, disjoint
previous screen/confirmation cases, two repeats, same roots for both mappings,
all actors and own initial controls. Repaired-map neural PLAY progress:
neural-native 0.07327 → 0.07047, reference-native → 0.06628, matched → 0.06850.
All nine observed changes are negative. Primary neural-native mean change is
−0.00280, all three negative, conditional case interval [−0.00813, +0.00203].
Reference-scored means also decline in every new-map feedback condition.
Old-map neural-native mean change is +0.00119, with one of three positive;
neither setup demonstrates the required native learning signal. New-map
initial progress is higher, which is not a training gain.

Independent verification replays 72,000 training actions, 19,584 new screen/
calibration actions and 7,488 fresh actions per mapping: 106,560 in total.
Saved states, reward mapping, pulse assignment, bounds, fast traces and source/
unrelated/frozen weights pass. All 91 targeted tests pass. A descriptive fresh
decision audit finds neural-native PLAY chooses high card despite available
pair-or-better in 457/624 cases (73.24%), and misses an immediate clear in
50/624 (8.01%). Actual training neural/reference pulse correlations are only
0.044–0.123; mean pulses 0.003430/0.042601. At the repaired output, 95.98–96.32%
of changed afferents have the same final delta within each plastic actor.
These are diagnostic observations, not a causal proof of clipping or absent
card information.

Predeclared advance gate fails: native gain <0.005, zero positive learners,
conditional interval crosses zero. No full-game or long training was started.
Keep the legitimate motor repair and original mappings/checkpoints; do not
reallocate using these fresh outcomes. Next isolate card-dependent choice
learning and the information retained by actual choice-gated eligibility in
a balanced two-option diagnostic. Existing pair/window/centering/scaling null
results remain valid. Long training becomes worth scaling after repeatable
held-out native gains and a bounded free-action pilot with improving progress/
clears and stable controls; frequent wins are not a prerequisite.

Methods, full old/new table and reproduction:
`docs/REPAIRED_NATIVE_MEMBERSHIP.md`. The study and both fresh comparisons are
complete; no GPU study remains running.

### 2026-10-28 — Local-only teaching implementation and bounded development

The user rejected gradient training and retained local plasticity only, while
authorizing the proposed learner redesign and teacher curriculum. Implemented
`scripts/local_teaching.py`, `teacher_curriculum_data.py`,
`run_local_teacher_curriculum.py`, `local_rate_teaching.py`,
`run_local_rate_teacher.py`, and independent
`assess_local_teacher_curriculum.py`. Existing production/historical research
code, primary configuration, mappings and checkpoints remain unchanged.

The new sparse rule uses each synapse's smooth presynaptic activity, local mean
trace, measured postsynaptic activity and a synthetic output-neuron teaching
signal. Only existing KC→MBON efficacies learn. Candidate 1 teaches a signed
desired-minus-chosen card/count discrepancy. Candidate 2 teaches desired firing
levels relative to fixed calibration, with a minimum one-spike/decision margin.
Neither computes derivatives, backpropagation or an optimizer update, and no
external actor/decoder/critic parameters are fitted. Exact equations/routing
are experimental engineering conventions, not validated biological mechanisms.

Implemented staged higher-card, pair-member, pair, best-hand and native
Small-Blind card-play curricula, true/random-instruction/frozen actors, label
separation, checkpoint generations and atomic manifests, torn-log recovery,
exact resume, and teacher-free native evaluation. First-blind evaluation forces
PLAY and lets the fly choose count/cards; it does not claim autonomous DISCARD,
blind-selection or shop behaviour. The visible-state teacher cleared 21/32
training Small Blinds without discards; this is reference performance only.

Candidate 1 (`artefacts/local-teacher-curriculum-20261028-v1/`) completed 5k
frames per actor in 1,303 seconds, three seeds × true/random/frozen. Fresh
true-instruction means: higher-card accuracy 53.04% → 54.65%, pair-member
51.12% → 53.37%, pair 3.85% → 4.81%, exact best-hand 0.64% → 1.60%.
Mean best-hand scored chips declined 19.41 → 16.61. Native first-blind
clears 0/192 → 1/192, equal to random instruction. Initial/final roots are
64 shared fresh game seeds across three learners, not 192 training replicates.
The native milestone failed, conditional case interval [0, +0.015625].

Candidate 2 (`artefacts/local-rate-teacher-20261028-v1/`) completed 3k frames
per actor in 900 seconds, using higher-card → best-hand → Small-Blind stages,
new final game roots and rate-target instruction. Fresh higher-card accuracy
49.04% → 50.48%, exact pair 2.56% → 4.49% (random instruction 6.73%),
best-hand imitation 0.64% → 0.96% (random 1.92%). Mean best-hand chips
declined 19.04 → 14.06. Native clears 1/192 → 0/192; random also 0/192,
frozen remains 1/192. Conditional interval [−0.015625, 0]; milestone fails.
The designs/budgets/fresh roots differ, so these are bounded development
attempts, not a matched causal rule comparison.

Both conditional higher-card training curves stayed near chance. Teaching
smaller card counts without mastering membership did not establish useful play.
The code/control checks are successful; the native learning milestone is not.
Neither model was promoted and long RL, DISCARD timing, full Ante-1 and shop
training were not started. The next substantive work requires a coherent local
learning/model-interface redesign, not another assumed punishment/duration fix.

All 100 targeted tests pass. Locality tests cover unrelated-synapse independence,
signed updates, no teacher action substitution, NumPy/Torch parity, complete
native curriculum execution and exact checkpoint continuation after a torn log.
Independent audits verify anatomical/frozen/source weights, exposure, updates,
checkpoints and 19,796 + 18,392 = 38,188 native evaluation actions. A real GPU
restoration from candidate 2's 250-frame checkpoint reproduces the exact next
actions and every local-update event; saved in `gpu-resume-verification.json`.

Methods, equations, limitations, commands and both negative results:
`docs/LOCAL_TEACHER_CURRICULUM.md`. Both transient user services completed;
no development training remains running. Original checkpoint hashes and legacy
resume identity are preserved.
