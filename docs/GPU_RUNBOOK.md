# Dedicated GPU-machine runbook: plastic-brain V1

Run from the repository root. These stages separate code/artifact verification,
calibration, training, held-out evaluation, the final test, and live
publication. Local synthetic success is not real FlyWire or Balatro evidence.

Do not choose long-run exposure budgets until stages 3-6 measure throughput,
memory, representation health, and update magnitude. Then set:

```bash
export FLYLATRO_ANTE1_DECISIONS=REPLACE_FROM_BENCHMARKS
export FLYLATRO_CURRICULUM_DECISIONS=REPLACE_FROM_BENCHMARKS
export FLYLATRO_CHECKPOINT_CADENCE=REPLACE_FROM_BENCHMARKS
export FLYLATRO_CURRICULUM_EVAL_CADENCE=REPLACE_FROM_BENCHMARKS
export FLYLATRO_CURRICULUM_EVAL_EPISODES=REPLACE_FROM_BENCHMARKS
export FLYLATRO_EVAL_EPISODES=REPLACE_FROM_BENCHMARKS
```

All values must be positive integers. They are intentionally not guessed here.

## 1. Build and verify the FlyWire artifact

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,training,flywire,recording,benchmark,balatro]'
.venv/bin/python -m pytest -q
mkdir -p artefacts data/flywire-v783-source data/flywire
cd data/flywire-v783-source
sha256sum -c ../../docs/flywire-v783.sha256
cd ../..
.venv/bin/python -m flylatro.fly.build_flywire \
  --source-dir data/flywire-v783-source \
  --output-dir data/flywire \
  --full
```

Success: tests pass, source checksums say `OK`, and the builder emits the NPZ,
neuron Parquet table, and schema-v2 manifest. It rejects a KC census other than
5,177, empty required populations, or an empty KC->MBON edge set.

## 2. Validate the KC/MBON/DAN census

```bash
.venv/bin/flylatro-verify-populations \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --output artefacts/flywire-v783-population-census.json \
  --full
```

Success: the report contains exact counts, rules, root IDs, artifact hash and
population hash. Review MBON, PAM, PPL1, APL, DPM, ALPN and descending counts
against the pinned tables before training.

## 3. Benchmark naive whole-brain simulation

```bash
.venv/bin/bench-fly \
  --backend flywire \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --device cuda \
  --batch-sizes 1,2,4,8 \
  --durations-ms 10,25,50 \
  --readout-size 1305 \
  --heavy | tee artefacts/bench-naive-whole-brain.json

.venv/bin/bench-plastic-end-to-end \
  --config configs/plastic-real-template.toml \
  --output-mode whole_brain \
  --steps 10 \
  --heavy | tee artefacts/bench-plastic-end-to-end.json
```

Success: all selected cells and ten plastic Mode-B decisions through descending
outputs finish without OOM or non-finite activity, reporting decisions/sec,
RAM and VRAM. Use this evidence to select the neural window and experiment
duration. V1 still defaults to one sequential learner even if larger neural
batches fit.

## 4. Benchmark sparse plastic-edge updates

Read `n_kc_mbon_edges` from the census and substitute it:

```bash
.venv/bin/bench-plasticity \
  --edges REPLACE_WITH_N_KC_MBON_EDGES \
  --learners 1 \
  --iterations 1000 \
  --heavy | tee artefacts/bench-plasticity.json
```

Success: 1,000 update cycles finish with finite bounded weights and report
edge-updates/sec and memory. Confirm storage is a sparse-edge vector, not a
dense neuron-by-neuron matrix.

## 5. Run naive representation diagnostics

```bash
.venv/bin/flylatro-diagnose-representation \
  --config configs/plastic-real-template.toml \
  --samples 16 \
  --repeats 3 \
  --output artefacts/naive-representation.json \
  --heavy
```

Success: KCs are sparse but not all silent; MBON and descending populations are
not all silent/constant/saturated; different observed states are separable.
Calibrate input rates or neural duration before long training if this fails.

## 6. Run a tiny real-FlyWire plastic experiment

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/real-fly-tiny \
  --condition plastic_real \
  --max-environment-decisions 20 \
  --checkpoint-every-decisions 10 \
  --record-plasticity-events \
  --no-tensorboard \
  --heavy

.venv/bin/flylatro-analyze-synapses \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/real-fly-tiny/plastic-checkpoint-final.pkl \
  --output artefacts/real-fly-tiny-synapses.json \
  --heavy
```

Success: all actions are legal; eligible reinforcement changes at least one
KC->MBON efficacy; weights remain finite and mostly away from bounds; the
manifest says zero external trainable parameters; resume reproduces the next
update exactly. `plasticity-events.jsonl` contains sparse edge/root-ID deltas,
not a full efficacy matrix at every decision.

## 7. Train the Ante-1 calibration experiment

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/plastic-real-ante1-map0 \
  --condition plastic_real \
  --sensory-mapping-seed 0 \
  --curriculum-ladder 1,8 \
  --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
  --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
  --max-environment-decisions "$FLYLATRO_ANTE1_DECISIONS" \
  --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
  --heavy
```

Success: the exact decision budget is consumed, one fly's weights persist
across episode resets, held-out promotion metrics are separate, and no external
actor or critic appears. Behavioural improvement is measured, not assumed.

## 8. Train the full curriculum

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/plastic-real-curriculum-map0 \
  --condition plastic_real \
  --sensory-mapping-seed 0 \
  --curriculum-ladder 1,2,3,5,8 \
  --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
  --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
  --max-environment-decisions "$FLYLATRO_CURRICULUM_DECISIONS" \
  --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
  --heavy
```

Success: seed streams stay disjoint; promotion reads only frozen held-out
episodes; decisions, episodes and plasticity events are logged; every checkpoint
restores with matching topology, rule, sensory, motor and reinforcement hashes.

## 9. Evaluate the frozen plastic fly

```bash
.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/plastic-real-curriculum-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/plastic-real-map0-heldout \
  --episodes "$FLYLATRO_EVAL_EPISODES" \
  --seed-stream validation \
  --heavy
```

Success: `learning_disabled` is true; the plastic-weight hash never changes;
metrics include win rate, Ante, episode length, decisions and score; successful
episodes become hash-verified replay candidates.

## 10. Run the no-plasticity control

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/control-no-plasticity-map0 \
  --condition no_plasticity \
  --action-schedule runs/plastic-real-curriculum-map0/training-actions.jsonl \
  --sensory-mapping-seed 0 \
  --curriculum-ladder 1,2,3,5,8 \
  --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
  --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
  --max-environment-decisions "$FLYLATRO_CURRICULUM_DECISIONS" \
  --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
  --heavy

.venv/bin/flylatro-analyze-synapses \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/control-no-plasticity-map0/plastic-checkpoint-final.pkl \
  --output artefacts/control-no-plasticity-map0-synapses.json \
  --condition no_plasticity \
  --action-schedule runs/plastic-real-curriculum-map0/training-actions.jsonl \
  --heavy
```

Success: seeds, curriculum Antes, actions and before/after state hashes match
stage 8 exactly, while final and initial weight hashes match and
changed-synapse count is zero.

## 11. Run the shuffled-topology control

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/control-shuffled-topology-map0 \
  --condition shuffled_topology \
  --sensory-mapping-seed 0 \
  --curriculum-ladder 1,2,3,5,8 \
  --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
  --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
  --max-environment-decisions "$FLYLATRO_CURRICULUM_DECISIONS" \
  --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
  --heavy
```

Success: metadata records the population-preserving postsynaptic permutation
and seed 1701. Edge/weight counts and degree distributions match, KC plastic
edges still target MBONs, and exposure matches stage 8.

## 12. Run the shuffled-reward control

```bash
.venv/bin/flylatro-shuffle-reward \
  --events runs/plastic-real-curriculum-map0/dopamine-events.jsonl \
  --source-checkpoint runs/plastic-real-curriculum-map0/plastic-checkpoint-final.pkl \
  --seed 1701 \
  --output artefacts/shuffled-dopamine-map0.json

.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/control-shuffled-reward-map0 \
  --condition shuffled_reward \
  --dopamine-schedule artefacts/shuffled-dopamine-map0.json \
  --action-schedule runs/plastic-real-curriculum-map0/training-actions.jsonl \
  --sensory-mapping-seed 0 \
  --curriculum-ladder 1,2,3,5,8 \
  --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
  --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
  --max-environment-decisions "$FLYLATRO_CURRICULUM_DECISIONS" \
  --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
  --heavy
```

Success: the recorded schedule hash is stable; the appetitive/aversive pulse
multiset matches the source while order differs; every scheduled pulse is used
once and current outcomes do not supply dopamine. The forced action log makes
the state/action trajectory match the source run exactly; compare transition
hashes before treating this as a valid causal control.

## 13. Run sensory-mapping replicates

```bash
for mapping_seed in 1 2; do
  .venv/bin/flylatro-train \
    --config configs/plastic-real-template.toml \
    --run-dir "runs/plastic-real-curriculum-map$mapping_seed" \
    --condition plastic_real \
    --sensory-mapping-seed "$mapping_seed" \
    --curriculum-ladder 1,2,3,5,8 \
    --curriculum-evaluation-every-decisions "$FLYLATRO_CURRICULUM_EVAL_CADENCE" \
    --curriculum-evaluation-episodes "$FLYLATRO_CURRICULUM_EVAL_EPISODES" \
    --max-environment-decisions "$FLYLATRO_CURRICULUM_DECISIONS" \
    --checkpoint-every-decisions "$FLYLATRO_CHECKPOINT_CADENCE" \
    --heavy
done
```

Success: mapping hashes differ while features, motor/reinforcement maps, seed
order, curriculum and decision budgets remain identical.

## 14. Run the final matched held-out comparison

Evaluate stages 8, 10, 11, 12 and 13 with `--seed-stream validation`, the same
episode count and separate output directories:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/plastic-real-curriculum-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/plastic-real-map0-heldout \
  --episodes "$FLYLATRO_EVAL_EPISODES" \
  --seed-stream validation \
  --heavy

.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/control-no-plasticity-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/control-no-plasticity-map0-heldout \
  --condition no_plasticity \
  --action-schedule runs/plastic-real-curriculum-map0/training-actions.jsonl \
  --episodes "$FLYLATRO_EVAL_EPISODES" \
  --seed-stream validation \
  --heavy

.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/control-shuffled-topology-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/control-shuffled-topology-map0-heldout \
  --condition shuffled_topology \
  --episodes "$FLYLATRO_EVAL_EPISODES" \
  --seed-stream validation \
  --heavy

.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/control-shuffled-reward-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/control-shuffled-reward-map0-heldout \
  --condition shuffled_reward \
  --dopamine-schedule artefacts/shuffled-dopamine-map0.json \
  --action-schedule runs/plastic-real-curriculum-map0/training-actions.jsonl \
  --episodes "$FLYLATRO_EVAL_EPISODES" \
  --seed-stream validation \
  --heavy

for mapping_seed in 1 2; do
  .venv/bin/flylatro-evaluate \
    --config configs/plastic-real-template.toml \
    --checkpoint "runs/plastic-real-curriculum-map$mapping_seed/plastic-checkpoint-final.pkl" \
    --output-dir "evaluations/plastic-real-map$mapping_seed-heldout" \
    --sensory-mapping-seed "$mapping_seed" \
    --episodes "$FLYLATRO_EVAL_EPISODES" \
    --seed-stream validation \
    --heavy
done
```

Success: all conditions use identical seeds and frozen weights; summaries retain
component hashes and exposures. Compare paired outcomes and report all
replicates, not only the best.

## 15. Unlock the final untouched test

Only after all modelling choices and the checkpoint are frozen:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/plastic-real-curriculum-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/plastic-real-map0-final-10000 \
  --episodes 10000 \
  --seed-stream final_test \
  --heavy \
  --unlock-final-test
```

Success: exactly 10,000 reserved seeds run once with frozen plasticity and all
hashes/metrics recorded. Never use these results for model selection.

## 16. Rerun a selected success with detailed recording

Choose a successful stage-15 seed and use its offset; do not search new seeds:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/plastic-real-curriculum-map0/plastic-checkpoint-final.pkl \
  --output-dir evaluations/showcase-selected-final-seed \
  --episodes 1 \
  --seed-stream final_test \
  --seed-offset REPLACE_WITH_SELECTED_FINAL_OFFSET \
  --record-neural \
  --heavy \
  --unlock-final-test
```

Success: the trajectory matches the selected final-test result and emits a
hash-verified replay candidate plus KC, MBON, DAN, descending, dopamine and
learned-weight snapshot events. Evaluation does not alter weights.

## 17. Replay in real Balatro

In a capture terminal, inspect and run the printed X11/ffmpeg command using the
actual game-window coordinates:

```bash
.venv/bin/flylatro-video-command capture \
  --output artefacts/showcase-game.mp4 \
  --offset-x REPLACE_WITH_WINDOW_X \
  --offset-y REPLACE_WITH_WINDOW_Y \
  --width REPLACE_WITH_WINDOW_WIDTH \
  --height REPLACE_WITH_WINDOW_HEIGHT \
  --fps 60
```

Then run the verified replay in the replay terminal:

```bash
.venv/bin/flylatro-replay \
  --bundle evaluations/showcase-selected-final-seed/replay-candidates/seed-SELECTED_SIMULATOR_SEED \
  --config configs/plastic-real-template.toml \
  --heavy

.venv/bin/flylatro-replay \
  --bundle evaluations/showcase-selected-final-seed/replay-candidates/seed-SELECTED_SIMULATOR_SEED \
  --config configs/plastic-real-template.toml \
  --live --url http://127.0.0.1:12346 \
  --heavy
```

Success: headless hashes pass first; live replay starts the canonical Balatro
seed, checks state before every complete action, and stops at first divergence.
Live publication remains distinct from headless verification.

## 18. Generate the fixed-coordinate visualisation

```bash
.venv/bin/flylatro-visualize \
  --bundle evaluations/showcase-selected-final-seed/replay-candidates/seed-SELECTED_SIMULATOR_SEED \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --output-dir artefacts/showcase-brain \
  --full

ffmpeg -y -f concat -safe 0 \
  -i artefacts/showcase-brain/frames.ffconcat \
  -c:v libx264 -pix_fmt yuv420p \
  artefacts/showcase-brain/brain.mp4

.venv/bin/flylatro-video-command compose \
  --game artefacts/showcase-game.mp4 \
  --brain artefacts/showcase-brain/brain.mp4 \
  --output artefacts/flylatro-plastic-learning-final.mp4
```

Success: one persisted anatomical transform is used for every frame; inactive
anatomy stays faint; KC/MBON/DAN/descending/plasticity roles differ; slowdown is
displayed; reward and appetitive/aversive pulses are labelled; all activity
comes from the recording. The printed composition command combines the brain
view with a user-owned Balatro capture; no proprietary asset enters this
repository.
