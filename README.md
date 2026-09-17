# Flylatro

Flylatro is a reinforcement-learning experiment that uses the fixed adult
*Drosophila* connectome as the neural processor for learning Balatro from
scratch. The authoritative research design is [PLAN.MD](PLAN.MD); the current
implementation ledger is [docs/V1_STATUS.md](docs/V1_STATUS.md).

The V1 execution path is:

```text
BalatroVecEnv observation
→ fixed full-state sensory encoder
→ reset-per-decision sparse FlyWire LIF simulation
→ fixed descending-neuron rate/voltage features
→ small structured actor + critic
→ exact legal-action masks
→ PPO
```

The encoder, FlyWire topology, neural dynamics, and feature extraction are
fixed. Only the structured policy/readout and critic are trained. Training
starts from random parameters and consumes environment reward only. The
upstream Balatro heuristic is isolated to an explicit evaluation baseline and
is never available to PPO.

## Safe local validation

Python 3.11+ is required. These commands use only mock Balatro and a fixed
synthetic reservoir; they do not load FlyWire, use a GPU, or run long training.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
.venv/bin/python -m pytest
.venv/bin/flylatro-train \
  --config configs/smoke-v1.toml \
  --run-dir /tmp/flylatro-smoke \
  --no-tensorboard
```

Real Balatro, real/shuffled FlyWire, more than 64 environments, or more than
10 PPO updates are rejected unless `--heavy` is supplied. Full connectome
construction separately requires `--full`. Final-test seeds require
`--unlock-final-test`. Detailed spike recording is off unless
`--record-neural` is explicitly requested for one episode.

## Main commands

```text
flylatro-train              PPO training and checkpointing
flylatro-evaluate           frozen policy evaluation + replay candidates
flylatro-baseline-evaluate  random/legal or isolated heuristic baseline
flylatro-replay             simulator verification, optional live replay
flylatro-visualize          FlyWire-coordinate SVG/timeline generation
bench-balatro               environment throughput
bench-fly                   fly simulation throughput
bench-policy                structured policy throughput
bench-end-to-end            rollout throughput
```

The default development profiles are `configs/dev-v1.toml` and
`configs/smoke-v1.toml`. Dedicated-machine profiles cover benchmarks, Ante 1,
curriculum, full training, controls, final evaluation, and showcase recording.
Exact commands and success criteria are in
[docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

## External data and assets

FlyWire FAFB v783 data are user-supplied and checksum-verified; they are not
vendored. Balatro is proprietary and is used only from a user-owned local
installation for the final visual replay. Flylatro does not recreate or
distribute Balatro assets. See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) and
[ADR 0002](docs/adr/0002-flywire-populations-and-encoding.md).

## Experimental status

The lightweight V1 stack is locally testable. Real PyO3 simulator build,
full-connectome CPU/GPU dynamics, throughput sizing, training, final held-out
evaluation, live Balatro replay, and final rendering remain dedicated-machine
validation—not completed scientific results.
