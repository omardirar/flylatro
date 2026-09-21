# Flylatro

Flylatro tests whether reinforcement-gated plasticity inside a connectome-derived
model of the adult *Drosophila* mushroom body can acquire useful Balatro
behaviour. The authoritative design is [PLAN.MD](PLAN.MD), implementation
evidence is tracked in [docs/V1_STATUS.md](docs/V1_STATUS.md), and dedicated
machine commands live in [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

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
  --episodes 8

.venv/bin/flylatro-analyze-synapses \
  --config configs/plastic-smoke.toml \
  --checkpoint /tmp/flylatro-plastic-smoke/plastic-checkpoint-final.pkl \
  --output /tmp/flylatro-plastic-smoke/synaptic-analysis.json
```

Real Balatro, a v783 artifact, CUDA, more than eight independent flies, or more
than 1,000 training decisions require `--heavy`. Full artifact construction
separately requires `--full`; final-test seeds require
`--unlock-final-test`. These guards keep this development machine from starting
the experiments intended for the dedicated GPU machine.

## Primary commands

```text
flylatro-train                 internal KC->MBON plastic learning
flylatro-evaluate              frozen plastic-fly evaluation
flylatro-analyze-synapses      learned-weight distributions and biological groups
flylatro-create-sensory-mapping field-aware ALPN mapping and collision audit
flylatro-calibrate-motor       reward-free canonical motor-pool artifact
flylatro-calibrate-plasticity  short stability report and configurable gates
flylatro-preflight             formal PASS/WARN/FAIL readiness report
flylatro-create-protocol       paired replicate/control manifest
flylatro-shuffle-reward        deterministic shuffled-reinforcement schedule
bench-plasticity               sparse plastic-edge update benchmark
bench-plastic-end-to-end       plastic-fly decision throughput
flylatro-replay                deterministic simulator and optional live replay
flylatro-visualize             fixed-coordinate neural/plasticity rendering
```

For a deliberately short showcase-training run,
`flylatro-train --record-plasticity-events` records only changed KC->MBON edge
IDs, root IDs and old/new/delta efficacy in streamed Parquet row groups. It is opt-in because detailed edge events can be large
on the real graph.

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
