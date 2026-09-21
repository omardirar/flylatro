# Dedicated GPU runbook: plastic-brain V1

This sequence is mandatory. Do not choose an Ante-1 or curriculum exposure
budget before measuring the dedicated machine. `plastic-real-template.toml`
contains a parser/calibration placeholder and `flylatro-train` refuses it.
Record every command, report, mapping hash, hardware description and chosen
threshold file in the experiment group.

Set paths only; do not set decision budgets yet:

```bash
export FLYLATRO_FLYWIRE_SOURCE=/absolute/path/to/flywire-v783-codex
mkdir -p data/flywire artefacts runs evaluations
```

## 1. Build and verify the v783 artifact

```bash
.venv/bin/python -m flylatro.fly.build_flywire \
  --source-dir "$FLYLATRO_FLYWIRE_SOURCE" \
  --output-dir data/flywire \
  --full

.venv/bin/flylatro-verify-populations \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --output artefacts/populations-and-data-quality.json \
  --full
```

Accept only the exact 5,177-KC census, valid source/artifact hashes, non-empty
MBON/DAN/PAM/PPL1/ALPN populations, explicit APL/DPM/descending reports, and
explicit unresolved-neurotransmitter, dropped-synapse and missing-coordinate
counters. Counts without a reliable reference are reported, not guessed.

## 2. Inspect population and weak-edge sensitivity

The preceding report contains KC->MBON thresholds `1`, `2`, `5`, and `10` with
retained neuron-pair edges, synapses, and both fractions. Keep
`minimum_synapse_count = 1` for the primary model. Any alternative threshold is
a separately named sensitivity experiment and changes the topology hash.

## 3. Create and audit the field-aware sensory map

```bash
.venv/bin/flylatro-create-sensory-mapping \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --mapping-seed 0 \
  --population-width 3 \
  --max-rate-hz 150 \
  --output artefacts/sensory-map-v783-seed0.json \
  --full
```

Accept only the versioned field-aware feature contract and clipped-sum
collision policy. Inspect ALPN use, assignment min/median/p90/p95/p99/max,
feature-class collisions, and isolated binary/scalar effective rates. Mapping
replicates change only `sensory_mapping_seed`.

## 4. Calibrate neural activity at multiple durations

The motor artifact intentionally does not exist yet; this command uses a
clearly marked bootstrap decoder only to traverse reward-free observable
states. Its decoder output is not an experiment result.

```bash
for duration in 10 25 50 100; do
  .venv/bin/flylatro-diagnose-representation \
    --config configs/plastic-real-template.toml \
    --duration-ms "$duration" \
    --samples 16 \
    --repeats 3 \
    --high-rate-hz 200 \
    --maximum-silent-fraction 0.95 \
    --minimum-separation-ratio 1.10 \
    --minimum-motor-dynamic-range-hz 1 \
    --output "artefacts/representation-${duration}ms.json" \
    --heavy
done
```

Choose a duration and thresholds from these measured Hz distributions. Require
finite activity, non-silent KC/MBON/descending populations, repeatable
same-state responses, useful different-state separation, and non-constant
candidate outputs. Phase, Ante, hands-remaining, blind-progress,
rank-presence and shop-presence probes are diagnostics only and never policy
inputs.

## 5. Calibrate and persist canonical motor pools

After selecting the duration in the config, run:

```bash
.venv/bin/flylatro-calibrate-motor \
  --config configs/plastic-real-template.toml \
  --samples 64 \
  --calibration-seed 91001 \
  --high-rate-hz 200 \
  --output artefacts/motor-map-v783.json \
  --heavy
```

Accept only reward-free calibration (`reward_used=false`), pool width at least
two, sufficient option dynamic range, the exact state seed/hash set, canonical
root IDs, and a stable mapping hash. The same artifact is mandatory for real,
no-plasticity, both topology controls, shuffled reinforcement and sensory-map
replicates.

## 6. Calibrate eligibility and plasticity stability

Choose a deliberately small decision count; it is a calibration exposure, not
a training recommendation:

```bash
.venv/bin/flylatro-calibrate-plasticity \
  --config configs/plastic-real-template.toml \
  --decisions 20 \
  --maximum-modified-fraction 0.80 \
  --maximum-bound-fraction 0.10 \
  --maximum-mean-absolute-change 0.50 \
  --output artefacts/plasticity-calibration.json \
  --heavy
```

Review mean/median efficacy, absolute/relative drift, mean/max eligibility,
mean/max update size, changed/positive/negative fractions and both bound
fractions. Tune configured reference rates, learning rate, decay or bounds;
never derive them from win rate or reward performance.

## 7. Run initial formal preflight

Store threshold choices in `artefacts/preflight-thresholds.json`; omitted
benchmark/control evidence is a WARN at this stage because those measurements
occur later in this sequence.

```bash
.venv/bin/flylatro-preflight \
  --config configs/plastic-real-template.toml \
  --representation-report artefacts/representation-50ms.json \
  --plasticity-report artefacts/plasticity-calibration.json \
  --thresholds artefacts/preflight-thresholds.json \
  --output artefacts/preflight-initial.json
```

Proceed only with no FAIL. Artifact, sensory, representation, motor,
plasticity, reward mapping, learner-local reset, checkpoint and zero-external-
policy gates must be PASS; explain any tooling-only WARN.

## 8. Run a tiny real plastic experiment

Choose this tiny gate explicitly, not from the template:

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/tiny-real-plastic \
  --condition plastic_real \
  --budget-basis tiny-real-gate \
  --max-environment-decisions 20 \
  --checkpoint-every-decisions 10 \
  --record-plasticity-events \
  --no-tensorboard \
  --heavy

.venv/bin/flylatro-analyze-synapses \
  --config configs/plastic-real-template.toml \
  --checkpoint runs/tiny-real-plastic/plastic-checkpoint-final.pkl \
  --output artefacts/tiny-real-plastic-synapses.json \
  --heavy
```

Require legal actions, finite state, at least one eligible update, sparse
Parquet changes with pre/post/old/new/delta, low bound occupancy, exact resume,
and `external_trainable_parameter_count = 0`.

## 9. Compare the matched no-plasticity gate

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/tiny-no-plasticity \
  --condition no_plasticity \
  --action-schedule runs/tiny-real-plastic/training-actions.jsonl \
  --budget-basis tiny-matched-control \
  --max-environment-decisions 20 \
  --checkpoint-every-decisions 10 \
  --no-tensorboard \
  --heavy
```

Require identical action schedule hash, before/after state hashes, curriculum
Ante, seed order, sensory/motor mappings and exposure; weights must not change.

## 10. Benchmark throughput and memory

```bash
.venv/bin/bench-fly \
  --backend flywire \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --device cuda \
  --batch-sizes 1,2,4,8 \
  --durations-ms 10,25,50,100 \
  --heavy | tee artefacts/bench-fly.json

.venv/bin/bench-plastic-end-to-end \
  --config configs/plastic-real-template.toml \
  --steps 10 \
  --compare-sequential \
  --heavy | tee artefacts/bench-plastic-end-to-end.json
```

Record measured decisions/sec, peak CPU/GPU memory, batch size, duration and
the measured sequential-versus-batched comparison. The memory model is one
shared fixed sparse graph plus per-learner neural state and batched KC->MBON
efficacy/eligibility/reinforcement traces; do not infer speedups before this.

## 11. Choose budgets and create the paired replicate protocol

Derive environment decisions, checkpoint cadence and evaluation sizes from the
measurements and available wall-clock/storage budget. Record the rationale:

```bash
.venv/bin/flylatro-create-protocol \
  --name ante1-v1 \
  --base-seed 20260921 \
  --replicates REPLACE_WITH_JUSTIFIED_REPLICATE_COUNT \
  --conditions plastic_real,no_plasticity,kc_mbon_shuffled,whole_brain_shuffled,shuffled_reward \
  --exposure-budget-decisions REPLACE_WITH_MEASURED_ANTE1_BUDGET \
  --curriculum-ladder 1 \
  --motor-mapping-id REPLACE_WITH_MOTOR_MAPPING_SHA256 \
  --output artefacts/ante1-protocol.json
```

Do not proceed while any `REPLACE_WITH_...` remains.

## 12. Rerun strict preflight with measured evidence

Prepare two tiny run manifests flattened or retaining their `components`
objects, then require benchmark and control evidence in the threshold JSON:

```json
{
  "require_benchmark_report": true,
  "require_control_manifest_check": true
}
```

```bash
.venv/bin/flylatro-preflight \
  --config configs/plastic-real-template.toml \
  --representation-report artefacts/representation-50ms.json \
  --plasticity-report artefacts/plasticity-calibration.json \
  --benchmark-report artefacts/bench-plastic-end-to-end.json \
  --control-manifest runs/tiny-real-plastic/run-manifest.json \
  --control-manifest runs/tiny-no-plasticity/run-manifest.json \
  --thresholds artefacts/preflight-thresholds.json \
  --output artefacts/preflight-before-ante1.json
```

All scientific/readiness gates must PASS before Ante-1. A visualization-only
WARN is acceptable if final media is not being produced on this machine.

## 13. Run the paired Ante-1 experiment

For every protocol replicate, run the real condition and matched controls with
its recorded seeds. Example real arm after substituting measured values:

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/ante1-replicate-000-real \
  --condition plastic_real \
  --budget-basis artefacts/bench-plastic-end-to-end.json \
  --max-environment-decisions REPLACE_WITH_MEASURED_ANTE1_BUDGET \
  --checkpoint-every-decisions REPLACE_WITH_MEASURED_CHECKPOINT_CADENCE \
  --heavy
```

Primary topology control is `--condition kc_mbon_shuffled`; the distinct
wider control is `--condition whole_brain_shuffled`. Both must reuse the same
motor artifact. Shuffled reinforcement is created from
`synthetic-reinforcement-events.jsonl` and passed with
`--reinforcement-schedule`. Compare paired replicates; never report only the
best seed.

## 14. Proceed to curriculum only after Ante-1 acceptance

The real template's single-level `[1]` ladder fixes the target at Ante 1 and
performs no promotion evaluations. Advance to `1,2,3,5,8` only if Ante-1 runs
preserve all matched hashes, show stable finite plasticity away from widespread
bounds, exhibit non-degenerate actions/neural activity, resume exactly, and
complete within measured resource budgets. Supply measured
`--curriculum-evaluation-every-decisions` and
`--curriculum-evaluation-episodes` values when enabling the multi-level ladder;
the template values are explicitly unused for its fixed target. Behavioural
improvement is an experimental result, not a software gate.

## Optional frozen time-resolved showcase

This is never enabled during routine training:

```bash
.venv/bin/flylatro-evaluate \
  --config configs/plastic-real-template.toml \
  --checkpoint REPLACE_WITH_FROZEN_CHECKPOINT \
  --output-dir evaluations/showcase \
  --episodes 1 \
  --seed-stream showcase \
  --record-neural \
  --record-spikes \
  --heavy
```

The Parquet file has one row group per decision and distinguishes input
stimulation, real KC/MBON/DAN-anatomy/descending spikes, synthetic appetitive
and aversive outcome channels, and selected plastic changes. The synthetic
channels are not PAM/PPL1 spikes. Visualization reads only requested decision
row groups and shows strongest edge pre/post/old/new/delta details.
