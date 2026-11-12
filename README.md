# Flylatro

Flylatro tests whether reinforcement-gated plasticity inside a connectome-derived
model of the adult *Drosophila* mushroom body can acquire useful Balatro
behaviour. The authoritative design is [PLAN.MD](PLAN.MD), implementation
evidence is tracked in [docs/V1_STATUS.md](docs/V1_STATUS.md), and dedicated
machine commands live in [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

Upcoming learning experiments are tracked in the
[learning roadmap](docs/LEARNING_ROADMAP.md), with runnable
[card-learning diagnostics and pilots](docs/CARD_LEARNING_EXPERIMENTS.md).
The [PLAY/DISCARD interface fix](docs/PLAY_DISCARD_INTERFACE.md) provides a new
mapping with separate card/count preferences and learning credit; old
checkpoints retain their original shared interpretation.

The experimental [local-only teacher curriculum](docs/LOCAL_TEACHER_CURRICULUM.md)
adds versioned local synaptic rules, visible-state demonstrations, checkpoint
resume and teacher-free first-blind card-play evaluation. Two bounded candidates
completed without a reliable native gameplay gain; neither changes primary V1
or qualifies longer training.

The primary V1 path is:

```text
Balatro state
-> fixed field-aware, seeded synthetic sensory mapping
-> fixed FlyWire topology and simplified LIF dynamics
-> sparse plastic KC->MBON efficacies
-> MBON-direct or downstream descending activity
-> fixed zero-parameter structured motor interface + legal mask
-> Balatro outcome
-> fixed synthetic appetitive/aversive reinforcement mapping
-> eligibility-gated KC->MBON update
```

There is no trainable external actor, decoder, critic, PPO update, or gradient
descent in this path. The persistent learned parameters are the efficacy values
of existing KC->MBON edges. Neural, eligibility and internal reinforcement traces are dynamic fly
state. Balatro sensing and motor interpretation are synthetic, fixed interfaces
and are not claimed to be natural fly biology.

## Safe local plastic-brain smoke run

The smoke profile uses a tiny synthetic mushroom-body circuit and mock Balatro.
It does not load FlyWire, compile the real simulator, use a GPU, or make a
scientific performance claim.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
.venv/bin/python -m pytest
.venv/bin/flylatro-train \
  --config configs/plastic-smoke.toml \
  --run-dir /tmp/flylatro-plastic-smoke \
  --no-tensorboard

.venv/bin/flylatro-evaluate \
  --config configs/plastic-smoke.toml \
  --checkpoint /tmp/flylatro-plastic-smoke/plastic-checkpoint-final.pkl \
  --output-dir /tmp/flylatro-plastic-eval \
  --win-ante 1 \
  --episodes 8

.venv/bin/flylatro-analyze-synapses \
  --config configs/plastic-smoke.toml \
  --checkpoint /tmp/flylatro-plastic-smoke/plastic-checkpoint-final.pkl \
  --output /tmp/flylatro-plastic-smoke/synaptic-analysis.json
```

Training opens a colourful, responsive terminal dashboard automatically. It
shows each fly's decisions, runs, wins/game overs, current/highest/average round,
Ante, blind chips and training reward. A bar chart compares rounds reached in
recent completed runs, labelling played rounds and skipped blinds separately.
Solid bars show played rounds, patterned bars show skips, and both have exact
counts. Bars start at zero, share the same scale across flies, and run from oldest at the top to
newest at the bottom. The average uses all tracked completed runs; performance
history retains 64 runs while the visible bars adapt to available screen space.
Round 0 means no blind has been selected yet. Round/highest/average statistics
use played rounds; skips add to the separate bar segment without counting as
rounds beaten. Missing legacy skip counts show `?`; fresh runs track them fully.
Real training stays fixed to **Red Deck / White Stake**, with the
current Ante win target and fresh run seeds shown in the header.

Each fly has an action feed with named cards, Jokers and purchases. Game over,
round beaten, Ante beaten and run won appear as separate coloured milestones
and add no decisions. Up/Down or the mouse wheel scroll through up to 256
recent activity entries; PgUp/PgDn jump five entries, Tab or Left/Right select
a fly, Home goes to the oldest entry, and End or L returns to the live feed.
Browsing holds the selected history steady while training and counters continue.
Press H for metric definitions and controls. Ctrl+C still checkpoints and exits.
The feed describes submitted choices; simulator card/consumable legalisation
can adjust what takes effect. Hidden card identities remain hidden.

Training reward is the accumulated learning signal; highest blind chips is
the largest observed blind total. Neither is a final game score. Throughput,
elapsed time, ETA, plasticity guards and GPU utilisation/memory/temperature/power
are shown separately. GPU readings cover the whole device; Torch allocation
is shown separately for this process.

The display follows [Nielsen's usability guidance](https://www.nngroup.com/articles/ten-usability-heuristics/)
for visible status, familiar labels, consistency and accessible help, and
[ONS chart guidance](https://service-manual.ons.gov.uk/data-visualisation/guidance/axes-and-gridlines)
for zero baselines and consistent scales. Gameplay, hardware and learning
limits are grouped separately; status and milestone text remains readable
without colour. Performance line charts and speed sparklines are omitted.
Use `--dashboard-refresh 2` to refresh every two seconds, `--no-dashboard` for
quiet terminal output, or `--dashboard` to force periodic plain snapshots in a
redirected log. Final JSON stays on stdout; the display uses stderr. Set
`NO_COLOR=1` to disable colours; non-Unicode terminals use ASCII.
Final scores unavailable from the pinned simulator are shown as `--`;
observed chip peaks are labelled separately. Monitoring counters survive new
checkpoint resumes and are recorded in `training-dashboard.json` and
`training-episodes.jsonl`. Older checkpoints retain available aggregate
history and label missing per-fly/score/round history. Round averages and graphs
begin with newly tracked runs when resuming a checkpoint without round history.

Real Balatro, a v783 artifact, CUDA, more than eight independent flies, or more
than 1,000 training decisions require `--heavy`. Full artifact construction
separately requires `--full`; final-test seeds require
`--unlock-final-test`. These guards keep this development machine from starting
the experiments intended for the dedicated GPU machine.

Frozen evaluation and synaptic analysis accept checkpoints trained with a CLI
exposure-budget override while still verifying every model and mapping identity.
Evaluation metadata retains the checkpoint's original training exposure. Exact
training resume continues to require a matching exposure budget.
`--win-ante 1` evaluates the initial target explicitly; the backward-compatible
default is Ante 8. Evaluation uses each frozen fly's own choices, never its
matched training action schedule. `flylatro-compare-evaluations` compares
episode rows on identical held-out seeds; a comparison PASS establishes
comparability, not improved performance.

Training has default-on `[safety]` stops for non-finite state/activity,
out-of-range efficacy or eligibility, more than 10% bound occupancy, mean
absolute efficacy drift above 0.50, and eight consecutive steps with less than
1% active KCs or selected motor neurons. Bounds/drift/activity are checked for
the worst learner, not hidden in a batch average. A stop returns exit code 2,
writes `safety-stop.json` and a quarantined checkpoint, and does not write a
successful final checkpoint. It never alters weights, rewards or choices.

`flylatro-run-staged-protocol` validates a materialized paired Ante-1 protocol;
`--execute --heavy` runs it. Default stages are 100/250/500/1,000 decisions,
with full frozen-corpus diagnostics on the actual learned weights after every
primary stage. Any failed stage halts all later stages and arms. Stage caps
preserve the original declared budget, so checkpoints resume exactly.
Exact training resume also refuses a changed safety policy; frozen diagnostics
can load older checkpoints without changing their learned state.

On 2026-10-09, `configs/plastic-v783-resolution.toml` passed all initial real
calibration gates on the RTX 4080 at 500 ms, without lowering thresholds. A
20-decision real training run changed 9,135 KC->MBON synapses, saved a checkpoint
and passed its exact matched no-plasticity control. A subsequent seed-bound
100-decision Ante-1 pilot passed strict preflight and its matched control,
changing 9,760 synapses without hitting efficacy bounds. This is a working GPU
training path, not evidence of improved Balatro performance. Frozen Ante-1
evaluation on 32 identical held-out seeds found zero wins for both the trained
100-decision fly and its untrained control. The learned checkpoint passed
full-corpus representation and reachability checks. See
[the measured status](docs/V1_STATUS.md) and
[the motor allocation decision](docs/adr/0014-group-local-motor-populations.md).
The guarded longer run then halted automatically at decision 172: 11.01% of
efficacies reached their lower bound, above the 10% safety limit. Its
160-decision checkpoint is retained; later stages/replicates were not run.
GPU access is working, but long-exposure stability is not established.

The continuation corrected the offline bound audit and completed independent
250-decision LR 0.002/0.005 pilots with passing neural audits and no bounds.
The 0.002 pilot's exact control passes. Neither pilot won Ante 1; lower drift
does not establish learning.

A source audit found that the historical artifact omitted 40,567 weak KC->MBON
pairs. Primary minimum-count-1 V1 now requires the separately built unthresholded
artifact under `data/flywire-source-complete-v783/`; existing artifacts and
checkpoints are preserved. Its initial 500 ms representation fails unchanged
high-rate and state-separation gates. Prospectively ordered reward-free
calibration selects fixed gain 0.225 mV / 100 ms, passing the unchanged gates
with motor v5 and representation v4. A new LR 0.002 pilot completes 250
decisions safely, passes full learned-checkpoint health, its exact control and
a real GPU full-trainer resume audit. All six strict primary/control preflights
pass. The first 1000-decision primary and matched control finish training;
the primary passes neural health and wins 0/32 validation episodes. The user
stopped the driver during control evaluation; remaining pairs are unrun.
This does not establish Ante-1 learning. Current evidence and frozen final-test
declarations are tracked in [the progress log](docs/PROGRESS_LOG.md)
and [ADR 0015](docs/adr/0015-source-coverage-and-fixed-gain.md).

## Primary commands

```text
flylatro-readiness               what remains before a real Ante-1 run
flylatro-build-calibration-corpus frozen reward-free observable-state corpus
flylatro-select-calibration-corpus compact corpus retaining rare phases/contexts
flylatro-create-sensory-mapping  field-aware ALPN mapping + state-conditioned health
flylatro-diagnose-representation neural representation, --stage pre or post
flylatro-calibrate-motor         reward-free canonical motor-pool artifact
flylatro-calibrate-plasticity    short stability report and configurable gates
flylatro-diagnose-specificity    chosen-action vs diffuse plasticity attribution
flylatro-preflight               PASS/WARN/FAIL gate, --profile initial or ante1
flylatro-create-protocol         paired replicate/control manifest
flylatro-materialize-protocol    one exact runnable configuration per arm
flylatro-run-staged-protocol     checkpoint-audited training with fail-stop stages
flylatro-train                   internal KC->MBON plastic learning
flylatro-evaluate                frozen plastic-fly evaluation
flylatro-compare-evaluations     descriptive independent-action paired evaluation
flylatro-validate-controls       matched-control semantics report
flylatro-analyze-synapses        learned-weight distributions and biological groups
flylatro-shuffle-reward          deterministic shuffled-reinforcement schedule
bench-plasticity                 sparse plastic-edge update benchmark
bench-plastic-end-to-end         plastic-fly decision throughput and components
flylatro-replay                  deterministic simulator and optional live replay
flylatro-visualize               fixed-coordinate neural/plasticity rendering
```

The dedicated-machine order is fixed and dependency-driven: build the artifact,
freeze a reward-free calibration corpus, map and audit the sensory route on that
corpus, measure representation **before** any motor artifact exists, choose the
neural duration, calibrate the motor interface, re-measure representation
**with** the persisted mapping, calibrate plasticity, run the initial preflight,
run a tiny real experiment and its matched control, benchmark, build and
materialize the protocol, run the strict preflight, then Ante-1.
`flylatro-readiness` reports which of those stages is blocking and why.

## Evidence provenance

Every generated report carries an evidence identity: git commit and dirty state,
config SHA, simulator version, FlyWire artifact/population/connectivity SHAs,
plastic topology SHA and minimum synapse threshold, sensory feature-contract and
mapping SHAs, calibration corpus SHA, motor candidate-set and mapping SHAs,
reinforcement and plasticity rule SHAs, fly dynamics SHA, decision duration,
output mode, backend and device.

`flylatro-preflight` does not trust a report's own `PASS`. It checks that the
report's identity matches the current experiment and fails with an exact
mismatch message otherwise, so evidence measured at another duration, with
another sensory or motor mapping, or against another artifact or topology cannot
silently authorize a run.

For a deliberately short showcase-training run,
`flylatro-train --record-plasticity-events` records changed KC->MBON edge IDs,
root IDs and old/new/delta efficacy in streamed Parquet row groups. It is opt-in
because detailed edge events are large on the real graph and because routine
training deliberately keeps the whole plastic state on the simulation device:
a normal CUDA step copies only a small scalar block to the host, never a full
efficacy, eligibility or reinforcement vector.

`configs/plastic-real-template.toml` is deliberately a template whose budget
basis is a refused placeholder. Real training must provide a measured budget
and `--budget-basis`; its single-level curriculum fixes the initial target at
Ante 1 without promotion checks. The mandatory order is in the GPU runbook.

## Legacy reservoir/PPO baseline

The earlier fixed-fly architecture is preserved as a comparison, not the main
project direction. Its plan is archived at
[docs/archive/RESERVOIR_V1_PLAN.md](docs/archive/RESERVOIR_V1_PLAN.md) and its
entry points are explicitly named:

```text
flylatro-legacy-ppo-train
flylatro-legacy-ppo-evaluate
bench-policy
bench-end-to-end
```

Legacy code still provides useful environment, seed, replay, benchmark, and
visualisation infrastructure. A green legacy test is not evidence that the
plastic-brain experiment is complete.

## Licence

Flylatro's own source is MIT licensed (see `LICENSE` and ADR 0013). That licence
covers this repository's code only. FlyWire connectome data, Balatro itself, the
pinned `balatroagent` simulator and every third-party Python dependency carry
their own terms and are not redistributed here; see
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md). `uv.lock` pins the resolved
dependency graph, including the pinned simulator revision.

## External data and assets

FlyWire FAFB v783 data are user-supplied, checksum-verified, and not vendored.
The artifact builder records exact KC, MBON, DAN, PAM, PPL1, ALPN, APL, DPM and
descending indices/root IDs plus sparse KC->MBON edge positions, weak-edge
sensitivity and explicit dropped-data counters. Balatro is a
proprietary user-owned installation used only for final replay; no game assets
are distributed here. See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) and the
ADRs under `docs/adr/`.

## Verification boundary

Local synthetic tests can establish software properties such as deterministic
mappings, legal actions, bounded plasticity, exact checkpoint resume, frozen
evaluation, and propagation through tiny graphs. They cannot establish that
the v783 populations are correctly censused on disk, that whole-brain dynamics
are useful, that learning succeeds in real Balatro, or that GPU throughput is
adequate. Those claims require the runbook on the dedicated machine.

## Accelerated execution

Performance migration adds explicit CSR, native CUDA, CUDA Graph and outgoing
spike execution while retaining the reference backend as the default. The
source-complete model, mappings and internal learning rule are preserved. See
[the migration guide](docs/PERFORMANCE_MIGRATION_GUIDE.md) for installation,
backend/checkpoint compatibility, runtime controls and reproducible commands,
the [performance report](docs/PERFORMANCE_REPORT.md) for measured results and
limits, and [the work record](docs/PERFORMANCE_MIGRATION.md) for qualification
artifacts. Throughput improvements do not establish better learning.
