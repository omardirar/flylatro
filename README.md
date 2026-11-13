# Flylatro

Flylatro tests whether reinforcement-gated plasticity inside a connectome-derived
model of the adult *Drosophila* mushroom body can acquire useful Balatro
behaviour. The authoritative design is [PLAN.MD](PLAN.MD), implementation
evidence is tracked in [docs/V1_STATUS.md](docs/V1_STATUS.md), and dedicated
machine commands live in [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

## Current status

The GPU execution and checkpoint path is qualified, but reliable Balatro
learning remains unproven. The latest completed studies are dated 6 November
2026; this overview was updated 12 November 2026.

| Completed work | Result | Evidence |
| --- | --- | --- |
| Native CUDA execution | About 28 decisions/s for one fly and 56 aggregate for eight on the measured RTX 4080; the reference backend remains the default | [Performance report](docs/PERFORMANCE_REPORT.md) |
| Repaired native long run | Stopped at 509,922 decisions per plastic learner on the configured saturation guard; fresh evaluation found some first-blind progress and no Ante-1 wins | [Long-run review](docs/REPAIRED_LONG_RESULTS.md) |
| Repeated-game capacity experiment | Six learners completed one million decisions each; no saved greedy evaluation won Ante 1 or cleared the repeated game's first blind | [Single-seed results](docs/SINGLE_SEED_RESULTS.md) |
| Local teacher curricula | Two bounded candidates failed to establish reliable native gameplay improvement | [Curriculum and results](docs/LOCAL_TEACHER_CURRICULUM.md) |
| Exchange credit, relational inputs and sensory controls | No follow-up qualified a new long-training candidate; the latest cue-association control did not improve | [Learning roadmap](docs/LEARNING_ROADMAP.md) |

Long exposure mainly strengthened the preference to play five cards. Useful
card membership selection and discard strategy remain unresolved. The current
research priority is a controlled test of whether sensory input and local
KC→MBON changes can reliably steer the intended card outputs. The proposed
direct KC-afferent bypass control has not been reported as completed.

Start with the [documentation index](docs/README.md) for setup, experiment
methods, results and historical records. [PLAN.MD](PLAN.MD) defines primary V1;
experimental interfaces and local rules retain separate identities and do not
replace its model or historical checkpoints.

## Model and learning boundary

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

Training opens an interactive terminal dashboard with per-fly progress, action
history, learning health and GPU readings. Press H for help; Ctrl+C checkpoints
and exits. Use `--no-dashboard` for quiet output or `--dashboard-refresh 2` for
a two-second refresh. See the [dashboard guide](docs/TRAINING_DASHBOARD.md) for
controls, metric definitions, resumed history and logging behaviour.

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

## Calibration and experiment history

Primary V1 uses the source-complete v783 artifact: 139,255 neurons and 67,504
plastic KC→MBON pairs. The historical threshold-5 export omitted 40,567 weak
pairs and remains available for reproducing its original checkpoints. The
source-complete reward-free calibration selected fixed gain 0.225 mV and a
100 ms decision window, with versioned motor and representation reports.

Early source-complete pilots passed bounded neural-health, control and resume
checks without establishing Ante-1 learning. Later performance qualification,
long runs and card-learning studies are linked above. Their different rules,
interfaces and evaluation conditions must be compared using each study's own
methods and controls. See [implementation status](docs/V1_STATUS.md),
[the progress log](docs/PROGRESS_LOG.md), and
[ADR 0015](docs/adr/0015-source-coverage-and-fixed-gain.md) for the development
history and provenance requirements.

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
