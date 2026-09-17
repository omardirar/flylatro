# Dedicated GPU-machine runbook

Run all commands from the repository root. Heavy operations require `--heavy`
or `--full`. Replace checkpoint and bundle placeholders with outputs from the
preceding stage. Keep the final-test command sealed until model selection is
finished.

## 0. Install and verify

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,training,flywire,recording,benchmark,balatro]'
.venv/bin/python -m pytest
mkdir -p artefacts
```

Success: the pinned `balatro_sim` extension imports and the fast suite passes.
If the machine needs a particular CUDA Torch wheel, install that official wheel
before the editable install; checkpoint manifests record the resolved version.

Download FlyWire FAFB v783 manually from the Codex portal into
`data/flywire-v783-source/`. Select `neurons.csv.gz`, `classification.csv.gz`,
`consolidated_cell_types.csv.gz`, `connections_princeton.csv.gz`, and
`coordinates.csv.gz`, then run:

```bash
cd data/flywire-v783-source
sha256sum -c ../../docs/flywire-v783.sha256
cd ../..
.venv/bin/python -m flylatro.fly.build_flywire \
  --source-dir data/flywire-v783-source \
  --output-dir data/flywire \
  --full
```

Success: every checksum reports `OK`; the builder emits
`data/flywire/flywire_fafb_v783.npz` and a manifest with 139,255 neurons,
expected population counts, and content hashes.

## 1. Balatro benchmark

```bash
.venv/bin/bench-balatro \
  --backend real --num-envs 32 --steps 200 --heavy \
  | tee artefacts/bench-balatro.json
```

Success: 6,400 masked legal decisions finish without rejection; JSON reports
environment decisions/sec, wall time, and peak RAM.

## 2. Real fly benchmark

```bash
.venv/bin/bench-fly \
  --backend flywire \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --device cuda \
  --batch-sizes 1,8,16,32 \
  --durations-ms 10,25,50 \
  --readout-size 1305 \
  --heavy \
  | tee artefacts/bench-fly.json
```

Success: every matrix entry completes without CUDA OOM and reports fly
decisions/sec, wall time, peak RAM, allocated VRAM, and peak VRAM. Use this
evidence to tune `microbatch_size`; environment count is independent.

## 3. End-to-end benchmark

```bash
.venv/bin/bench-end-to-end \
  --config configs/benchmark-gpu.toml --heavy \
  | tee artefacts/bench-end-to-end.json
```

Success: real simulator→encoder→FlyWire→policy rollout completes and reports
environment, fly, and end-to-end decisions/sec plus episodes and memory. Use
completed episodes and wall time to calculate episodes/hour.

## 4. Ante-1 training

```bash
.venv/bin/flylatro-train \
  --config configs/ante1.toml \
  --run-dir runs/ante1-real-fly \
  --heavy
```

Success: metrics/TensorBoard update, hash-manifested checkpoints appear every
100 updates, held-out promotion metrics are logged, and losses remain finite.

## 5. Full curriculum training

```bash
.venv/bin/flylatro-train \
  --config configs/curriculum.toml \
  --run-dir runs/curriculum-real-fly \
  --heavy
```

Success: held-out curriculum results promote `1 → 2 → 3 → 5 → 8`; training,
validation, curriculum, final-test, and showcase seed ranges remain disjoint.

## 6. Held-out evaluation

```bash
.venv/bin/flylatro-evaluate \
  --config configs/heldout-eval.toml \
  --checkpoint runs/curriculum-real-fly/checkpoint-XXXXXXXXXXXX.pt \
  --output-dir evaluations/held-out \
  --episodes 4096 \
  --heavy
```

Success: `summary.json` says `learning_disabled: true` and reports win rate,
mean/median Ante, episode length, decisions, score when available, and complete
successful-run replay candidates.

## 7. Controls

```bash
.venv/bin/flylatro-train \
  --config configs/control-shuffled.toml \
  --run-dir runs/control-shuffled --heavy

.venv/bin/flylatro-train \
  --config configs/control-conventional.toml \
  --run-dir runs/control-conventional --heavy

.venv/bin/flylatro-evaluate \
  --config configs/control-shuffled.toml \
  --checkpoint runs/control-shuffled/checkpoint-XXXXXXXXXXXX.pt \
  --output-dir evaluations/control-shuffled \
  --episodes 4096 --heavy

.venv/bin/flylatro-evaluate \
  --config configs/control-conventional.toml \
  --checkpoint runs/control-conventional/checkpoint-XXXXXXXXXXXX.pt \
  --output-dir evaluations/control-conventional \
  --episodes 4096 --heavy

.venv/bin/flylatro-baseline-evaluate \
  --config configs/heldout-eval.toml \
  --condition random_legal --episodes 4096 \
  --output evaluations/random-legal.json --heavy

.venv/bin/flylatro-baseline-evaluate \
  --config configs/heldout-eval.toml \
  --condition heuristic_external --episodes 4096 \
  --output evaluations/heuristic-external.json --heavy
```

Success: shuffled metadata records
`postsynaptic-neuron-label-permutation-v2` and seed 1701;
the conventional model logs its modest parameter count; action baselines use
the same environment/seeds/rewards and never expose heuristic actions to PPO.

## 8. Final untouched test

Run once, only after freezing the checkpoint and every modelling choice:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/final-eval.toml \
  --checkpoint runs/FROZEN_RUN/checkpoint-FROZEN_STEP.pt \
  --output-dir evaluations/final-10000 \
  --episodes 10000 \
  --heavy \
  --unlock-final-test
```

Success: exactly 10,000 reserved final-test seeds run with learning disabled;
the summary records the checkpoint/config hashes and final metrics. Never use
these results for model selection.

## 9. Neural recording

Choose a successful candidate from the completed final evaluation and rerun its
recorded integer `simulator_seed` with the frozen checkpoint. This is a
certified rerun of an already-selected untouched success, not seed searching:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/final-eval.toml \
  --checkpoint runs/FROZEN_RUN/checkpoint-FROZEN_STEP.pt \
  --output-dir evaluations/showcase \
  --simulator-seed SELECTED_FINAL_SIMULATOR_SEED \
  --record-neural \
  --heavy \
  --unlock-final-test
```

Success: the winning bundle contains `neural-events.parquet` with decision,
simulated milliseconds, FlyWire root ID, role, and activity. Event capture is
confined to this episode.

Materialise the video-associated reproducibility package:

```bash
.venv/bin/flylatro-package-showcase \
  --bundle evaluations/showcase/replay-candidates/seed-SELECTED_FINAL_SIMULATOR_SEED \
  --checkpoint runs/FROZEN_RUN/checkpoint-FROZEN_STEP.pt \
  --config configs/final-eval.toml \
  --output-dir artefacts/showcase-bundle
```

Success: the verified bundle also contains the exact checkpoint, experiment
config, and generated replay README, all covered by manifest SHA-256 values.

## 10. Real Balatro replay

Install balatrobot v1.5.2 and its Lua mod into a user-owned Balatro install,
launch the local JSON-RPC server on port 12346, and isolate gameplay-changing
mods. Verify headless replay before opting into live mutation:

```bash
.venv/bin/flylatro-replay \
  --bundle artefacts/showcase-bundle \
  --config artefacts/showcase-bundle/experiment.toml \
  --heavy

.venv/bin/flylatro-replay \
  --bundle artefacts/showcase-bundle \
  --config artefacts/showcase-bundle/experiment.toml \
  --live --url http://127.0.0.1:12346 \
  --heavy
```

Success: simulator hashes pass first; the bridge starts the canonical Balatro
seed, verifies the shared live-state signature before every full composite
action, and stops at the first mismatch. Capture only the locally owned game
window. Generate the exact capture command for the machine's window geometry:

```bash
.venv/bin/flylatro-video-command capture \
  --output artefacts/balatro_raw.mp4 \
  --display :0.0 --offset-x 0 --offset-y 0 \
  --width 1280 --height 720 --fps 60
```

Inspect and run the printed ffmpeg command immediately before the live replay;
stop it after the victory screen. It captures pixels from the installed game
without copying assets into Flylatro.

## 11. Final rendering

```bash
.venv/bin/flylatro-visualize \
  --bundle artefacts/showcase-bundle \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --output-dir artefacts/showcase-brain \
  --full

ffmpeg -y -f concat -safe 0 \
  -i artefacts/showcase-brain/frames.ffconcat \
  -c:v libx264 -pix_fmt yuv420p \
  artefacts/showcase-brain/brain.mp4
```

Generate the final composition command:

```bash
.venv/bin/flylatro-video-command compose \
  --game artefacts/balatro_raw.mp4 \
  --brain artefacts/showcase-brain/brain.mp4 \
  --output artefacts/flylatro_win.mp4
```

Run the printed ffmpeg command. Success: real FlyWire coordinates render active
input/internal/readout neurons, timeline metadata includes action probabilities
and cursor cues, and the visual labels the neural-time slowdown. No proprietary
game asset belongs in the repository.
