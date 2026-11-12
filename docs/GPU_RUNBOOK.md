# Dedicated GPU runbook: plastic-brain V1

This sequence is mandatory and its order is a dependency order, not a
preference. Calibration of the motor interface depends on a frozen reward-free
state corpus and on a chosen neural duration; the motor readiness claim depends
on the motor interface actually existing. Running these out of order produces
evidence that `flylatro-preflight` will refuse, because every report records the
exact configuration it was measured under.

Source correction (2026-10-09): historical `data/flywire/` uses a threshold-5
export. It cannot establish primary minimum-count-1 coverage; current preflight
rejects it for V1. Preserve it for historical checkpoints. The corrected build
is `data/flywire-source-complete-v783/`. Its 500 ms default-gain representation
failed; calibration is ongoing under named configurations and unchanged gates.
Recorded commands/results and the six-hour GPU ledger are in
`docs/PROGRESS_LOG.md` and `artefacts/lr-sensitivity-20261009/`.

`plastic-real-template.toml` contains a parser/calibration placeholder and
`flylatro-train` refuses it. Record every command, report, mapping hash,
hardware description and threshold file in the experiment group.

Historical threshold-5 profile (2026-10-09):
`configs/plastic-v783-resolution.toml` binds the balanced 96-state real corpus,
500 ms duration and the v4 calibrated mapping. Pre/post representation,
reachability, motor and integrated plasticity gates pass with default
thresholds. `runs/tiny-real-resolution-20261009` contains the 20-decision real
run; its exact control is `runs/tiny-real-resolution-no-plasticity-20261009`.
Current source-coverage and report-version gates reject these historical
artifacts for a new V1 run. They document the original development measurements.
The original 50 ms calibration reports remain historical failed evidence.

The seed-bound 100-decision pilot uses
`artefacts/ante1-resolution-pilot-protocol.json` and generated configs under
`artefacts/ante1-resolution-pilot-arms/`. Its strict report
`artefacts/preflight-ante1-resolution-pilot.json` is PASS with no warnings;
readiness is `READY_FOR_ANTE1`. This is one development replicate with an exact
no-plasticity control, not the complete topology/reward-control experiment.
Generated configs anchor inherited inputs and evidence paths before relocation.

Set paths only; do not set decision budgets yet:

```bash
export FLYLATRO_FLYWIRE_SOURCE=/absolute/path/to/flywire-v783-codex
mkdir -p data/flywire artefacts runs evaluations
```

At any point, ask what remains:

```bash
.venv/bin/flylatro-readiness --config configs/plastic-real-template.toml
```

It prints one of `READY_FOR_ARTIFACT_BUILD`, `READY_FOR_CALIBRATION`,
`READY_FOR_TINY_REAL_RUN` or `READY_FOR_ANTE1`, the blocking reason for each
stage, and the next command. It never reports `READY_FOR_ANTE1` from synthetic
or mock evidence.

## 1. Build and verify the v783 artifact

```bash
.venv/bin/python -m flylatro.fly.build_flywire \
  --source-dir "$FLYLATRO_FLYWIRE_SOURCE" \
  --output-dir data/flywire-source-complete-v783 \
  --connection-source unthresholded \
  --full
```

The builder refuses nonempty output directories. The path above is already
built on this machine; verify/reuse its pinned hashes, or choose a fresh output
directory for a new build. Set the configuration's `fly.artifact_path` to that
artifact before generating any source-dependent evidence. Historical reproduction
uses a fresh directory and explicit `--connection-source threshold5`.

## 2. Inspect the population census and weak-edge statistics

```bash
.venv/bin/flylatro-verify-populations \
  --artifact data/flywire-source-complete-v783/flywire_fafb_v783.npz \
  --output artefacts/populations-and-data-quality.json \
  --full
```

Accept only the exact 5,177-KC census, valid source/artifact hashes, non-empty
MBON/DAN/PAM/PPL1/ALPN populations, explicit APL/DPM/descending reports, and
explicit unresolved-neurotransmitter, dropped-synapse and missing-coordinate
counters. The same report carries KC->MBON thresholds `1`, `2`, `5` and `10`.
Keep `minimum_synapse_count = 1` for the primary model; any other threshold is a
separately named sensitivity experiment and changes the topology hash.

## 3. Generate the frozen reward-free calibration state corpus

Every later calibration reads these exact states. The corpus contains
observable states and legal masks only: no reward, no best action, no expected
value, no strategy annotation. Scripted legal navigation is used only to reach
diverse states and its actions are discarded.

```bash
.venv/bin/flylatro-build-calibration-corpus \
  --config configs/plastic-real-template.toml \
  --environment-seeds 9000001,9000002,9000003,9000004,9000005,9000006,9000007,9000008 \
  --states-per-seed 8 \
  --sample-every 3 \
  --navigation-seed 770001 \
  --store-snapshots \
  --output artefacts/calibration-corpus-v783.npz \
  --heavy
```

Inspect `phases_observed`, `phases_missing` and `motor_context_coverage` in the
manifest. Deeper phases (`SHOP`, `PACK`, `ROUND_EVAL`) need longer navigation:
increase `--states-per-seed` and/or `--sample-every`, and raise
`--maximum-decisions-per-seed` if that ceiling would prevent collection. The
generator stops once the requested states are collected, so raising the ceiling
alone does not extend navigation. Record which phases and contexts the corpus
does and does not reach. Its SHA-256 is part of every downstream report.

For rare inventory contexts, opt into
`--navigation-rule seeded-inventory-context-navigation-v1`. This collection-only
rule postpones selling a lone Joker or consuming/selling a lone consumable when
another supported legal action exists. Once multiple targets exist, all legal
actions are sampled uniformly again. It reads only legality masks, never card
quality, reward or score; its version is recorded and hashed in the corpus.
The original uniform navigation remains the default.

Avoid running neural diagnostics over tens of thousands of common states just
to retain a few rare shop or inventory states. Collect a larger navigation
corpus first, then freeze a compact subset:

```bash
.venv/bin/flylatro-select-calibration-corpus \
  --source artefacts/calibration-corpus-v783-navigation.npz \
  --output artefacts/calibration-corpus-v783-balanced.npz \
  --states 96 \
  --states-per-stratum 12
```

Selection prioritizes rare phases and relevant/competing legal motor contexts,
deduplicates observable states, spreads samples over each source trajectory,
and fills remaining slots from the full source. It preserves exact masks,
snapshots and run provenance. The source SHA, selection settings and coverage
policy SHA are included in the selected corpus identity. The selection report
lists remaining coverage deficiencies and the command returns exit code 2 when
they exist; it cannot manufacture evidence absent from the source. Existing
output files are refused. Set `[calibration].corpus_path` to the selected corpus
before producing any downstream evidence, and reuse that exact subset for every
calibration stage.

Coverage is formally gated, not merely reported. `flylatro-preflight` evaluates
the corpus against the versioned `calibration-corpus-coverage-policy-v1`
requirements — minimum states, minimum unique states, minimum distinct source
runs, minimum states and unique states per required phase, and for each motor
interpretation context (`action_type`, `card_count`, `card_slot`, `shop_target`,
`pack_target`, `joker_target`, `consumable_target`) a minimum number of relevant
states, states with competing legal options, and active legal options. In the
`initial` profile a shortfall is a WARN; in `ante1` it is a FAIL, and
deficiencies are reported exactly, for example
`SHOP: 0 states, required >= 4`. Override the defaults deliberately by writing a
JSON policy and setting `[calibration].coverage_policy_path`; do not weaken it
to make a gate pass.

Preflight also binds the corpus to *this* environment. A real experiment
(`fly.backend = flywire`, `environment.backend = balatro_sim`) requires the
corpus manifest to record `environment_backend = "BalatroSimAdapter"` and
`simulator_version = "balatroagent-38ae21431700"`. A mock-generated corpus is
refused even when the configuration points at its SHA-256. The corpus arrays
themselves are re-hashed on load, so an edited manifest cannot authorize a run,
and `--store-snapshots` snapshots are hashed into the corpus identity
(`snapshot_identity_policy = "snapshots-hashed-into-corpus-identity-v1"`).

Each state additionally records the root reset-stream seed, the environment's
actual current `run_seed()`, the episode index inside that stream, the decision
index and a monotonic collection index, so auto-reset cannot make two states
from different runs look like one.

## 4. Create the field-aware sensory mapping

```bash
.venv/bin/flylatro-create-sensory-mapping \
  --artifact data/flywire-source-complete-v783/flywire_fafb_v783.npz \
  --mapping-seed 0 \
  --population-width 3 \
  --max-rate-hz 150 \
  --calibration-corpus artefacts/calibration-corpus-v783.npz \
  --output artefacts/sensory-map-v783-seed0.json \
  --health-report artefacts/sensory-health-v783.json \
  --full
```

## 5. Audit the sensory mapping on the actual calibration corpus

The command above writes both artifacts. Structural assignment counts
(`assignments_per_alpn`, class collisions, ALPN use) are **informational**: they
count every feature channel that *could* share an ALPN across the whole pinned
contract. Readiness comes from the state-conditioned report, which counts only
the channels that are simultaneously non-zero in an observed state:

- `fraction_state_alpn_at_max_rate` — is the input population saturating?
- `fraction_state_alpn_saturated_by_collision` — is saturation caused by
  simultaneous collisions rather than a single full-amplitude channel?
- `active_contributors_per_active_alpn` — median/p90/p95/p99/max simultaneous
  contributors per driven ALPN;
- `distinct_state_pairs_with_identical_vector_fraction` — the decisive number.
  If two different observable states produce the same ALPN vector, collisions
  have destroyed the information the experiment depends on.

If the state-conditioned gate fails, lower `--max-rate-hz`, lower
`--population-width`, or re-examine the feature contract. Do **not** relax it by
raising a structural threshold. Mapping replicates change only
`--mapping-seed`.

## 6. Run neural representation PRE diagnostics at candidate durations

No motor artifact exists yet, and this stage makes no motor readiness claim.

```bash
for duration in 10 25 50 100; do
  .venv/bin/flylatro-diagnose-representation \
    --config configs/plastic-real-template.toml \
    --stage pre \
    --duration-ms "$duration" \
    --repeats 3 \
    --batch-size 1 \
    --high-rate-hz 200 \
    --maximum-silent-fraction 0.95 \
    --minimum-separation-ratio 1.10 \
    --output "artefacts/representation-pre-${duration}ms.json" \
    --reachability-output "artefacts/kc-reachability-${duration}ms.json" \
    --heavy
done
```

Both reports are measured over the frozen corpus. The representation report
names the population the motor interface will consume
(`motor_activity_population` is `mbon` in `mbon_direct` and `descending` in
`whole_brain`) and measures exactly that population.

The reachability report answers whether ALPN-only drive at this duration
actually reaches the plastic pool: per KC subtype it reports neuron count,
active fraction, mean/median/p95 Hz, ever-active fraction, plastic-edge counts,
ever-eligible edges and eligibility mass; plus the fraction of plastic MBONs and
of selected motor outputs that are ever active.

## 7. Choose the neural duration from measured representation evidence

Require finite activity, non-silent KC/MBON/descending populations, repeatable
same-state responses, and useful different-state separation. Set the chosen
`duration_ms` in the configuration. If a large plastic subpopulation is
unreachable, that is a documented model decision for the experimenter —
restrict the primary plastic population, or add another anatomically legitimate
sensory route as a separate versioned model change. Do not silently retune the
biology to make a gate pass.

## 8. Calibrate canonical reward-free motor candidates and pools

Candidates are the MBONs that are postsynaptic in the **real unshuffled**
KC->MBON topology at the canonical minimum synapse count, so every motor output
is plastic-reachable and the universe cannot move when a control shuffles
topology. Permanently reserved action slots receive no neural population at all.

```bash
.venv/bin/flylatro-calibrate-motor \
  --config configs/plastic-real-template.toml \
  --calibration-corpus artefacts/calibration-corpus-v783.npz \
  --calibration-seed 91001 \
  --high-rate-hz 200 \
  --minimum-candidate-robust-scale-hz 0.5 \
  --maximum-within-group-correlation 0.95 \
  --minimum-effective-signal-fraction 0.50 \
  --minimum-normalized-option-range 0.25 \
  --minimum-context-states 4 \
  --minimum-competing-context-states 2 \
  --output artefacts/motor-map-v783.json \
  --report artefacts/motor-map-v783-report.json \
  --heavy
```

The artifact is written only when the quality gates pass. Review the report:
candidate/selected counts, pool width, pool reuse structure, baseline firing
distribution, dynamic range, variance, normalized dynamic range, silent and
high-rate fractions, pairwise correlation between competing pools, and the
effective number of distinct signals per competition group. A group that cannot
obtain enough usable neural diversity fails loudly rather than quietly assigning
near-identical populations to competing alternatives.

Selection and quality are **context-aware**
(`reward-free-neural-motor-calibration-v5`). Each routing group is ranked and
decorrelated in the corpus states where its pools are actually read, using the
corpus's own legality masks: `selection_evidence_by_group` records how many
states that was. `quality.by_context` then reports, separately for every motor
interpretation context, the relevant state count, the number of states with
competing legal options, the active option count, the raw pool dynamic range,
the normalized dynamic range, pairwise correlation, the effective number of
distinct signals and contract-option coverage. A context with fewer than
`--minimum-context-states` relevant states or fewer than
`--minimum-competing-context-states` competing states is reported as
`insufficient_evidence` and fails the `context_evidence_sufficient` gate; it is
never averaged into a passing number. Pearson correlation uses a consistent
population denominator in selection and diagnostics; v4 motor reports and v3
representation reports are preserved historical evidence and are refused by
current preflight. Because the four contextual slot heads
share one pool group by design, this is the evidence that the shared pools carry
usable activity in each context in which they are interpreted. Setting either
threshold to `0` is an explicit opt-out and is only appropriate for a
development double.

Set `[fly].motor_mapping_path = "artefacts/motor-map-v783.json"`. The same
artifact is mandatory for real, no-plasticity, both topology controls, shuffled
reinforcement and sensory-map replicates. Record its `structure_sha256`: that is
the motor identity every matched condition must share, and the value the
replicate protocol binds.

Neurons remain distinct between competing pools within a routing group, while
separate comparison groups may reuse candidates (ADR 0014). Width two therefore
requires 26 candidates for the largest comparison, rather than 70 globally
distinct neurons. Candidate eligibility is evaluated in each group's own legal
contexts. The report records both total memberships and unique selected count;
all diversity and activity thresholds remain unchanged.

Once the artifact exists, regenerate reachability with `--stage post` and
`--reachability-output` on the same frozen corpus. This measures the actual
selected motor neurons and binds their mapping hash. A pre-motor reachability
report cannot establish selected-output coverage.

## 9. Confirm the persisted motor normalization

The artifact stores a fixed reward-free baseline and scale per pool
(`reward-free-median-iqr-v1`: baseline = median, scale = IQR/1.349 floored at
`--minimum-scale-hz`). `FixedMotorInterface.decode` subtracts the baseline and
divides by the scale before comparing alternatives, so two pools with very
different absolute firing rates but equivalent relative modulation compete
fairly. Check `normalization.statistics` for pools whose scale was floored:
those pools have little usable dynamic range.

## 10. Run representation POST diagnostics with the final motor mapping

Only this report may satisfy motor-related preflight gates.

```bash
.venv/bin/flylatro-diagnose-representation \
  --config configs/plastic-real-template.toml \
  --stage post \
  --repeats 3 \
  --minimum-normalized-option-range 0.25 \
  --minimum-action-coverage-fraction 0.50 \
  --maximum-competing-pool-correlation 0.99 \
  --minimum-context-states 4 \
  --minimum-competing-context-states 2 \
  --output artefacts/representation-post-50ms.json \
  --heavy
```

It reports normalized motor-pool activity, per-head option dynamic ranges,
action-option coverage, pairwise competition behaviour and motor score
distributions for the exact persisted mapping.

`motor_interface.by_group` and `by_head` remain descriptive whole-corpus
numbers. The readiness figures come from `motor_interface.by_context`, which
evaluates each head only in the states where it is read and only over the
options legal there: eligible states, states with more than one competing legal
option, normalized option dynamic range, options above the minimum range,
distinct argmax options, argmax coverage, pairwise score correlation and the
effective signal count. The gates `motor_context_evidence_sufficient`,
`motor_option_dynamic_range`, `action_option_coverage` and
`competing_pools_distinguishable` are all computed from those contexts, so a
pool that swings widely in irrelevant states and is constant inside its own
context cannot produce a PASS.

## 11. Calibrate eligibility and plasticity stability

A deliberately small decision count; this is a calibration exposure, not a
training recommendation.

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

## 12. Run the initial preflight

`flylatro-preflight` reads the evidence paths declared in the config's
`[calibration]` section, so the explicit flags below are only needed to override
them. It verifies that every report's recorded provenance matches this
configuration — duration, sensory mapping, motor mapping, artifact, topology,
plasticity rule and calibration corpus — and fails with an exact mismatch
message if a report belongs to another configuration.

```bash
.venv/bin/flylatro-preflight \
  --config configs/plastic-real-template.toml \
  --profile initial \
  --thresholds artefacts/preflight-thresholds.json \
  --output artefacts/preflight-initial.json
```

Proceed only with no FAIL. At this profile the benchmark, matched-control,
tiny-run, protocol and specificity gates are WARN because those measurements
happen later in this sequence, a calibration-corpus coverage shortfall is a
WARN, and a *failed* KC-reachability measurement is a WARN because the duration
and the sensory route are still being chosen. The strict `ante1` profile in
step 21 turns all of those into FAIL.

Every gate also checks that the supplied artifact is the kind and version of
report it asked for: a `sensory_health` report supplied as `motor_calibration`,
a `representation_pre` report supplied as `representation_post`, or a report
from an older incompatible version is a FAIL, not a silent pass.

## 13. Run a tiny real plastic experiment

Choose this tiny gate explicitly, not from the template:

Interactive `flylatro-train` sessions open the dashboard automatically.
`--dashboard` also works when output is redirected, using plain periodic
snapshots; `--no-dashboard` disables display and `--dashboard-refresh 2`
sets a two-second interval. The display goes to stderr and final JSON remains
on stdout. The coloured dashboard reports per-fly decisions/runs/wins/game
overs, current/best Ante, current/highest round and mean round reached over
tracked completed runs. Each fly's bar chart labels played rounds and skipped
blinds as distinct solid/patterned segments with exact counts, oldest run at
the top and newest at the bottom. Skips never count as beaten rounds or inflate
the played-round average. Unknown legacy skip counts show `?` until fresh runs.
Bars use zero
baselines and the same scale across flies; 64 runs are retained, while the
number of displayed bars adapts to terminal height. Round 0 precedes the
first selected blind; skipped blinds do not count as played rounds. Real
training uses the pinned **Red Deck / White Stake** setup, shown with the
current Ante win target. Each new run gets a new seed.

The action feed names submitted cards and inventory choices, with separate
coloured game-over/round-beaten/Ante-beaten/run-won milestones. Up/Down or
mouse wheel browse up to 256 activity entries, PgUp/PgDn jump five, Tab or
Left/Right select a fly, Home selects oldest, and End or L returns to live.
History stays still during browsing while the fly continues training. H opens
metric definitions; Ctrl+C still checkpoints and exits. Hidden cards remain
hidden, and simulator legalisation may adjust submitted selections.

Training reward means accumulated learning signal; highest blind chips means
largest observed blind total, rather than a final game score. Throughput, ETA,
allocation, GPU utilisation/memory/temperature/power and plasticity guards
are shown separately. `NO_COLOR=1` disables colour; ASCII fallback works on
non-Unicode terminals. Whole-device
GPU activity includes other processes. `nvidia-smi` absence is displayed as
unavailable and does not prevent training. Terminal scores omitted by the
simulator show `--`; peak chips are observed values, not complete final scores.
Counters are checkpoint metadata and persist across stages. Current counters
are saved atomically to `training-dashboard.json`, with per-fly terminal
events in `training-episodes.jsonl`. Older checkpoints label any missing history;
round averages and bars start with newly tracked runs if round history is absent.

```bash
.venv/bin/flylatro-train \
  --config configs/plastic-real-template.toml \
  --run-dir runs/tiny-real-plastic \
  --dashboard --dashboard-refresh 2 \
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
and `external_trainable_parameter_count = 0`. `--record-plasticity-events` turns
on detailed per-edge events and weight hashes; routine training leaves them off
so a CUDA step never copies a full plastic vector to the host.

Also measure how action-specific the global rule actually is:

```bash
.venv/bin/flylatro-diagnose-specificity \
  --config configs/plastic-real-template.toml \
  --decisions 20 \
  --output artefacts/chosen-action-specificity.json \
  --heavy
```

This reports the fraction of eligibility and of weight change attributable to
the chosen motor pool, to the competing pools in the same comparison, to motor
pools the chosen action never consulted, and to non-motor MBONs. It is a
measurement of whether `three-factor-global-v1` is sufficiently action-specific.
It is not a licence to bias plasticity toward the chosen output in V1.

## 14. Run the exact matched no-plasticity control

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

## 15. Validate the control manifests

```bash
.venv/bin/flylatro-validate-controls \
  --manifest runs/tiny-real-plastic/run-manifest.json \
  --manifest runs/tiny-no-plasticity/run-manifest.json \
  --output artefacts/control-validation-tiny.json
```

`--manifest` order is irrelevant: the reference is located by
`condition = plastic_real`. Exactly one manifest must declare it; none or
several is a clear failure rather than a silently wrong comparison.

The report states each arm's matching rule explicitly:

- `no_plasticity` and `shuffled_reward` are **exact action/state matched**: the
  executed action schedule, before/after state hashes, seed stream, Ante,
  sensory map, motor map and environment budget must agree;
- `kc_mbon_shuffled` and `whole_brain_shuffled` are **behaviourally independent
  topology controls**: a shuffled topology changes behaviour, so they cannot
  share the real fly's future trajectory. They are matched on the initial seed
  sequence, exposure budget, curriculum, sensory map, canonical motor map,
  reinforcement rule, simulation duration and hardware protocol — and their
  plastic topology hash must actually differ from the real one.

## 16. Benchmark throughput, memory and synchronization cost

```bash
.venv/bin/bench-fly \
  --backend flywire \
  --artifact data/flywire/flywire_fafb_v783.npz \
  --device cuda \
  --batch-sizes 1,2,4,8 \
  --durations-ms 10,25,50,100 \
  --profile-components \
  --heavy | tee artefacts/bench-fly.json

.venv/bin/bench-plastic-end-to-end \
  --config configs/plastic-real-template.toml \
  --steps 10 \
  --compare-sequential \
  --profile-components \
  --output artefacts/bench-plastic-end-to-end.json \
  --heavy
```

`component_seconds` attributes wall time to Poisson input generation, fixed
sparse recurrence, the plastic KC->MBON contribution, the LIF state update and
device synchronization (`recurrent` is the total; `recurrent_fixed` and
`recurrent_plastic` are its nested parts). Record measured decisions/sec, peak
CPU/GPU memory, batch size, duration and the sequential-versus-batched
comparison.

## 17. Tune implementation performance only where measured necessary

If and only if Poisson generation is a materially dominant component, switch
`fly.poisson_method` to `chunked-per-row-v2`, which draws each fly's own stream
in timestep blocks. It preserves independent per-learner streams,
reproducibility and row-order independence, but it **changes
`fly_dynamics_sha256`** and therefore invalidates every report measured under
the previous method. Re-run steps 6-12 if you switch it. Make no performance
claim that is not in a recorded benchmark.

## 18. Choose exposure, checkpoint and evaluation budgets from the measurements

Derive environment decisions, checkpoint cadence and evaluation sizes from the
benchmark and the available wall-clock/storage budget. Record the rationale as
the `--budget-basis`.

## 19. Create the paired replicate protocol

```bash
.venv/bin/flylatro-create-protocol \
  --name ante1-v1 \
  --base-seed 20260921 \
  --replicates REPLACE_WITH_JUSTIFIED_REPLICATE_COUNT \
  --conditions plastic_real,no_plasticity,kc_mbon_shuffled,whole_brain_shuffled,shuffled_reward \
  --exposure-budget-decisions REPLACE_WITH_MEASURED_ANTE1_BUDGET \
  --curriculum-ladder 1 \
  --reinforcement-condition primary-progress \
  --sensory-mapping-seed 0 \
  --motor-mapping-id REPLACE_WITH_MOTOR_STRUCTURE_SHA256 \
  --output artefacts/ante1-protocol.json
```

`REPLACE_WITH_MOTOR_STRUCTURE_SHA256` is the `structure_sha256` field of
`artefacts/motor-map-v783.json`, not its `sha256`. The structure hash covers the
candidate universe, pool structure, routing and normalization; the artifact hash
additionally covers the exploration settings, which legitimately differ per
replicate. Matched conditions must share the structure hash, and a run whose
loaded motor artifact does not match its protocol arm refuses to start.

`--replicates` is the number of **ordinary stochastic learning replicates**
inside one sensory-mapping block. They vary only the environment/training seed
stream, the fly Poisson seed and the motor exploration seed; they all reuse the
one fixed sensory mapping (`--sensory-mapping-seed`, default `0`) and the one
calibrated motor mapping. Arm IDs are therefore
`mapping-000-replicate-000:plastic_real` and so on.

A sensory-mapping change is a separate experimental factor, never replicate
noise. Add an explicit mapping-sensitivity block with

```bash
  --mapping-sensitivity 7:MOTOR_STRUCTURE_SHA256_CALIBRATED_THROUGH_SEED_7
```

repeated once per additional mapping. Each block needs its **own** motor mapping
SHA-256, because motor calibration, sensory health and both representation
stages are all measured through the sensory mapping and must be regenerated when
it changes; the protocol refuses two mapping blocks that share a motor mapping.

Do not proceed while any `REPLACE_WITH_...` remains. Every seed the protocol
stores controls a named stochastic mechanism; the manifest carries that audit
table in `seed_effects`, which quantities vary per replicate in `seed_audit`,
and the block-level factors in `block_level_factors`.

## 20. Materialize the protocol arms

```bash
.venv/bin/flylatro-materialize-protocol \
  --protocol artefacts/ante1-protocol.json \
  --base-config configs/plastic-real-template.toml \
  --output-dir artefacts/ante1-arms \
  --run-root runs/ante1 \
  --budget-basis artefacts/bench-plastic-end-to-end.json \
  --checkpoint-every-decisions REPLACE_WITH_MEASURED_CHECKPOINT_CADENCE
```

This writes one complete, already-validated configuration per arm plus
`protocol-plan.json` with the exact command and dependency order for each. No
JSON seed has to be translated into a CLI flag by hand, and every generated run
re-validates its configuration against the protocol arm before it starts.

Each `shuffled_reward` arm additionally carries `prerequisite_commands`: the
exact `flylatro-shuffle-reward` invocation, already carrying that arm's own
`reward_seed` and its source arm IDs. Run it verbatim; the arm refuses a
schedule shuffled with a different seed or derived from another reference run.

## 21. Run the strict Ante-1 preflight

```bash
.venv/bin/flylatro-preflight \
  --config configs/plastic-real-template.toml \
  --profile ante1 \
  --benchmark-report artefacts/bench-plastic-end-to-end.json \
  --protocol artefacts/ante1-protocol.json \
  --control-manifest runs/tiny-real-plastic/run-manifest.json \
  --control-manifest runs/tiny-no-plasticity/run-manifest.json \
  --thresholds artefacts/preflight-thresholds.json \
  --output artefacts/preflight-ante1.json
```

All scientific/readiness gates must PASS before Ante-1. A visualization-only
WARN is acceptable if final media is not being produced on this machine.

This profile is strict where `initial` was permissive:

- a **failed** KC/plastic-edge reachability report is a FAIL, not a WARN — the
  configured ALPN route and neural duration must actually satisfy the
  predeclared reachability gates. Do not lower the reachability thresholds;
  either restrict the primary plastic population or add another anatomically
  legitimate sensory route as a separate versioned model change;
- calibration-corpus coverage must satisfy the coverage policy for every
  required phase and every motor interpretation context;
- the corpus must have been generated by the real Balatro adapter at the pinned
  simulator revision;
- every report must be the requested kind at a compatible version;
- benchmark, matched-control, tiny-run, protocol and specificity evidence must
  all be present and provenance-matched.

## 22. Run the paired Ante-1 experiments

Execute `protocol-plan.json` in dependency order: every `plastic_real` arm
first, then its matched controls. `shuffled_reward` additionally needs its
schedule, which the plan emits verbatim as that arm's `prerequisite_commands`:

```bash
.venv/bin/flylatro-shuffle-reward \
  --events runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/synthetic-reinforcement-events.jsonl \
  --source-checkpoint runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/plastic-checkpoint-final.pkl \
  --source-run-manifest runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/run-manifest.json \
  --source-arm-id mapping-000-replicate-000:plastic_real \
  --target-arm-id mapping-000-replicate-000:shuffled_reward \
  --seed REPLACE_WITH_ARM_REWARD_SEED_FROM_protocol-plan.json \
  --output runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/shuffled-reinforcement.jsonl
```

Take `--seed` from the arm's own `reward_seed` in `protocol-plan.json`; do not
transcribe a seed by hand. The schedule records the source arm, the source
checkpoint and plastic-weight hashes, the reinforcement event-log SHA-256, the
executed action-schedule SHA-256 and the shuffle seed, and the
`shuffled_reward` arm refuses to start if any of them disagrees with its
protocol arm — including a schedule of exactly the right length taken from a
different replicate.

Compare paired replicates; never report only the best seed.

## 23. Analyze behaviour, plasticity specificity and stability

```bash
.venv/bin/flylatro-validate-controls \
  --manifest runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/run-manifest.json \
  --manifest runs/ante1/ante1-v1/mapping-000-replicate-000-no_plasticity/run-manifest.json \
  --manifest runs/ante1/ante1-v1/mapping-000-replicate-000-kc_mbon_shuffled/run-manifest.json \
  --output artefacts/control-validation-ante1-000.json

.venv/bin/flylatro-analyze-synapses \
  --config artefacts/ante1-arms/mapping-000-replicate-000-plastic_real.toml \
  --checkpoint runs/ante1/ante1-v1/mapping-000-replicate-000-plastic_real/plastic-checkpoint-final.pkl \
  --output artefacts/ante1-000-synapses.json \
  --heavy
```

The predeclared reinforcement-shaping sensitivity conditions in
`configs/reinforcement-sensitivity/` answer whether apparent learning depends on
dense blind-progress shaping. They do not all need to run before the first tiny
gate, but they must be run as a controlled comparison before any claim that
learning is driven by outcomes rather than by shaping. Choosing among them by
whichever scores best would invalidate them.

## 24. Only then decide whether to proceed to curriculum

The real template's single-level `[1]` ladder fixes the target at Ante 1 and
performs no promotion evaluations. Advance to `1,2,3,5,8` only if Ante-1 runs
preserve all matched hashes, show stable finite plasticity away from widespread
bounds, exhibit non-degenerate actions/neural activity, resume exactly, and
complete within measured resource budgets. Supply measured
`--curriculum-evaluation-every-decisions` and
`--curriculum-evaluation-episodes` when enabling the multi-level ladder.
Behavioural improvement is an experimental result, not a software gate.

## Guarded Ante-1 development experiments

Evaluate a previously exposed fly and its untrained control with independent
choices on identical validation seeds. Training controls replay actions to
test update semantics; evaluation deliberately does not replay them.
`--win-ante 1` is essential here because the evaluation default remains Ante 8.

```bash
.venv/bin/flylatro-evaluate \
  --config artefacts/ante1-resolution-pilot-arms/mapping-000-replicate-000-plastic_real.toml \
  --checkpoint runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/mapping-000-replicate-000-plastic_real/plastic-checkpoint-final.pkl \
  --output-dir evaluations/ante1-resolution-trained-32 \
  --win-ante 1 --episodes 32 --seed-stream validation --progress --heavy

.venv/bin/flylatro-evaluate \
  --config artefacts/ante1-resolution-pilot-arms/mapping-000-replicate-000-no_plasticity.toml \
  --checkpoint runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/mapping-000-replicate-000-no_plasticity/plastic-checkpoint-final.pkl \
  --condition no_plasticity \
  --output-dir evaluations/ante1-resolution-untrained-32 \
  --win-ante 1 --episodes 32 --seed-stream validation --progress --heavy

.venv/bin/flylatro-compare-evaluations \
  --trained evaluations/ante1-resolution-trained-32/summary.json \
  --control evaluations/ante1-resolution-untrained-32/summary.json \
  --output artefacts/paired-evaluation-ante1-resolution-32-audited.json
```

These are the recorded 2026-10-09 paths: both checkpoints won 0/32, so no
win-rate improvement is established. Existing output paths are refused; choose
fresh paths for another experiment. The comparison checks model identity,
target, streams, exact seed list and episode rows. PASS means the comparison is
valid, not that learning helped. Missing episode scores are unavailable, not
zero scores.

Initial calibration measures unit efficacy. Before increasing exposure, measure
the **actual learned checkpoint** without further updates:

```bash
.venv/bin/flylatro-diagnose-representation \
  --config configs/plastic-v783-resolution.toml --stage post \
  --checkpoint runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/mapping-000-replicate-000-plastic_real/plastic-checkpoint-final.pkl \
  --repeats 3 --batch-size 8 \
  --output artefacts/representation-checkpoint-ante1-resolution-100.json \
  --reachability-output artefacts/kc-reachability-checkpoint-ante1-resolution-100.json \
  --heavy
```

The recorded checkpoint passes both reports. Checkpoint reports carry distinct
`representation_checkpoint`/`kc_reachability_checkpoint` kinds and checkpoint
and weight hashes. They cannot authorize initial calibration, and a stale
checkpoint report cannot authorize a later training stage.

Routine training now enables these safety defaults even without an explicit
section in an older config:

```toml
[safety]
enabled = true
maximum_bound_fraction = 0.10
maximum_mean_absolute_drift = 0.50
minimum_kc_active_fraction = 0.01
minimum_motor_active_fraction = 0.01
silence_patience_steps = 8
```

The monitor uses device-reduced finite/bound/drift telemetry and per-learner
KC/selected-motor activity. It also stops for out-of-range efficacy/eligibility.
The worst learner governs; a batch average cannot hide a failed fly. Failure
returns exit code 2 and records `safety-stop.json` plus
`plastic-checkpoint-safety-stop.pkl`, with no successful final checkpoint.
The latest safe checkpoint is retained. Do not resume the quarantined state or
relax thresholds merely to finish the budget. These monitors never change the
biological update rule, rewards, weights or action choices.

Measured unprofiled throughput supports roughly 25 minutes per 1,000-decision
arm; six arms plus diagnostics/evaluation can take about three hours or more.
The historical three-replicate paired protocol is
`artefacts/ante1-guarded-1000-protocol.json`, with generated configs/plan under
`artefacts/ante1-guarded-1000-arms/`. Its old-rule/source/report evidence is
refused by current preflight. The following records its original validation
command; a new experiment must regenerate every affected input under fresh paths:

```bash
.venv/bin/flylatro-run-staged-protocol \
  --plan artefacts/ante1-guarded-1000-arms/protocol-plan.json \
  --health-report artefacts/representation-checkpoint-ante1-resolution-100.json \
  --reachability-report artefacts/kc-reachability-checkpoint-ante1-resolution-100.json \
  --control-manifest runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/mapping-000-replicate-000-plastic_real/run-manifest.json \
  --control-manifest runs/ante1-resolution-pilot-20261009/ante1-resolution-pilot-20261009/mapping-000-replicate-000-no_plasticity/run-manifest.json \
  --stages 100,250,500,1000 \
  --output artefacts/ante1-guarded-1000-plan.json --heavy
```

To execute, use a fresh `--output` path and add `--execute`. The recorded live
status is `artefacts/ante1-guarded-1000-execution.json`. Every stage preserves
the declared 1,000-decision budget and seed identity; `--stop-after-decisions`
is a cumulative stage cap, not a different budget. Each primary checkpoint
receives full-corpus representation/reachability audits before continuing. Any
nonzero training or audit exit halts all remaining stages and arms. Completed
pairs additionally require exact control validation and independent 32-seed
evaluation. A failed/staged run cannot pass as a completed control.

Recorded execution outcome: the first replicate's 100-decision audits passed,
but routine training stopped at decision 172 with 11.01% lower-bound occupancy
against the 10% limit. Its 160-decision checkpoint is preserved, the stopped
state is quarantined, and all later stages and arms were skipped. The protocol
did not complete. This is a learning-stability failure, not a CUDA access failure.

## Current source-complete primary and frozen final test

The historical threshold-5 paths above are retained for reproduction. Current
V1 uses `configs/plastic-v783-source-complete-gain0p225-100ms-lr0p002-primary-base.toml`,
the separately built unthresholded artifact, LR 0.002, fixed gain 0.225 mV,
100 ms and the reward-free motor v5 artifact. Initial full-corpus calibration,
250-decision learned health, exact pilot control and actual GPU resume pass.
All six strict preflights pass. The three paired primary runs are in progress;
no final-test outcomes have been observed and no learning claim is established.

The fresh resume probe creates a new checkpoint, rather than resuming a
quarantined training state. Its prefix-three/future-thirteen window includes
reward on the first resumed step:

```bash
.venv/bin/python -m flylatro.analysis.resume_cli \
  --config configs/plastic-v783-source-complete-gain0p225-100ms-lr0p002-pilot.toml \
  --checkpoint REPLACE_WITH_NEW_PROBE_CHECKPOINT \
  --prefix-decisions 3 --future-decisions 13 \
  --output REPLACE_WITH_NEW_RESUME_REPORT --heavy
```

The driver `scripts/run_primary_protocol.py` checks the terminal pilot evidence,
regenerates current benchmarks, validates all six arm configurations and starts
fail-stop staged execution. Its active status is
`artefacts/lr-sensitivity-20261009/source-complete-gain0p225-100ms-lr0p002-primary-driver.json`
and the nested execution status ends in `-1000-execution.json`. Inspect the
live handle and GPU ledger before considering another invocation; existing
destinations are refused. GPU commands retain conservative duration accounting
in `artefacts/lr-sensitivity-20261009/gpu-budget.json`. On 2026-10-09 the user
removed the six-hour ceiling for existing local hardware; subsequent commands
read `gpu-duration-authorization.json` and have no duration-budget timeout.
Original accounting and safety/scientific gates remain intact. The already
running wrapper retains its captured timeout and is expected to finish within
it; editing files does not change an existing Python process's timeout.

Current orchestration serializes GPU jobs and refuses an unresolved live
ledger entry. Separate processes on one GPU require a measured aggregate
throughput comparison and distinct configurations, seeds, checkpoints and
outputs before a simultaneous campaign is qualified. Multi-fly batching
shares fixed anatomy and is the measured acceleration path; a short neural
benchmark does not yet certify long batched training trajectories. Removing
the time ceiling does not relax a safety stop or extend a checkpoint's locked
exposure budget.

Only after all three primary/control pairs complete 1000 decisions, pass their
actual checkpoint health and exact control validation, and independently
evaluate all 32 validation seeds, qualify the predeclared frozen batches:

```bash
.venv/bin/python scripts/run_batch_validation.py \
  --primary-driver artefacts/lr-sensitivity-20261009/source-complete-gain0p225-100ms-lr0p002-primary-driver.json \
  --tag source-complete-gain0p225-100ms-lr0p002
```

This checks every completed primary checkpoint and each distinct frozen
baseline against sequential validation, using identical seed/decision actions,
rewards, state hashes and complete episode results. Qualification binds the
checkpoint, weights, configuration, implementation and evidence files. A
disagreement rejects batching. Width four replicates one frozen fly for
independent seeds; it introduces no learning and never replays training actions.

Write the final lock before observing the final 128 seeds per trained fly:

```bash
.venv/bin/python scripts/run_confirmatory_evaluation.py \
  --primary-driver artefacts/lr-sensitivity-20261009/source-complete-gain0p225-100ms-lr0p002-primary-driver.json \
  --tag source-complete-gain0p225-100ms-lr0p002 \
  --batch-validation artefacts/lr-sensitivity-20261009/source-complete-gain0p225-100ms-lr0p002-batch-validation.json
```

Inspect the concrete lock and remaining budget. Add `--execute` to this same
command to run its exact locked evaluation and analysis. A saved lock is reused
only if every derived identity matches. Omit `--batch-validation` for sequential
execution if qualification fails. The prospective common-baseline optimization
requires all three no-plasticity checkpoints to have exact unit weights, zero
updates and identical independent validation trajectories. It never substitutes
for any exact training control.

`confirmatory-pairing-correction.json` assigns one of the three trained flies
to each final seed before outcomes. The primary binary paired comparison uses
an exact one-sided McNemar/sign-permutation test; the three-fly mean and
seed-block bootstrap are descriptive, conditional on these trained flies.
Acceptance requires positive primary and overall means, p < 0.05 and primary
95% interval lower bound > 0, plus all scientific gates and a reproducible
genuine held-out win. Zero discordance and one lucky win do not qualify.
The analysis code and complete fixed model are locked before final execution.
Negative or inconclusive results remain failures to demonstrate learning;
do not reduce seeds, retune on final outcomes or label survival as success.

After the locked evaluations finish, reproduce a genuine winning final-test
episode and optionally record its actual spikes and brain video:

```bash
.venv/bin/python scripts/produce_heldout_win.py \
  --lock artefacts/lr-sensitivity-20261009/source-complete-gain0p225-100ms-lr0p002-confirmatory-lock.json \
  --output-dir artefacts/source-complete-ante1-heldout-win --execute
```

The deterministic selection uses the first trained replicate and declared
winning seed. The simulator first verifies the existing candidate, then the
recorded episode must match its complete original trajectory and frozen
weights. Ante target and shaping beta come from the bundle. Step rewards and
the actual terminal win are checked. The package contains checkpoint/config,
replay hashes and neural Parquet; rendering keeps fixed anatomy and separates
synthetic reinforcement from real anatomical spiking. A report explicitly
records no genuine win when none exists. A replayed win remains separate from
the confirmatory improvement test. Recording uses the same GPU ledger; video
composition runs on the CPU using the installed SVG decoder.

## Optional frozen time-resolved showcase

Never enabled during routine training:

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
channels are not PAM/PPL1 spikes.

## Stop conditions

Stop and investigate rather than relaxing a threshold if:

- the sensory state-conditioned gate shows distinct observable states producing
  identical ALPN vectors;
- reachability shows that most plastic edges can never become eligible at the
  chosen duration;
- motor calibration cannot find enough usable neural diversity for a head;
- preflight reports an evidence provenance mismatch (the evidence is stale:
  regenerate it, do not override the gate);
- the tiny real run produces zero eligible updates, non-finite state, or
  widespread bound saturation;
- a matched control's action schedule or state hashes diverge;
- a topology control reports the same plastic topology hash as the real arm;
- the calibration corpus cannot reach a required observable phase or a required
  motor interpretation context (extend navigation; do not lower the policy);
- a shuffled-reward schedule is refused because its seed, source arm, event log
  or action schedule does not match the protocol arm (regenerate it from the
  plan's own `prerequisite_commands`);
- `flylatro-validate-controls` cannot find exactly one `plastic_real` reference.

## Exploratory recovery pilots after the 2026-10-09 saturation stop

The unattended four-fly run stopped at 6028 total decisions (1507 per fly),
with 0 Ante-1 wins in 666 completed games. Fly 4 had 33.6% of its plastic
synapses at the lower bound. Total negative/positive pulse magnitude was
399.6/6.67; all four flies showed net weakening and no net strengthening
relative to initial weights. These observations motivate new engineering
experiments, rather than changes to the existing confirmatory protocol.

`configs/exploratory-20261010-plan.json` contains executable command argument
lists for seven matched 1000-decision pilots and 128-episode frozen validation:

| Arm | Learning rate | Failure pulse | Plasticity |
| --- | ---: | ---: | --- |
| Original reference | 0.002 | 0.60 | Enabled |
| Lower learning rate | 0.0005 | 0.60 | Enabled |
| Weaker failure | 0.002 | 0.01 | Enabled |
| Combined | 0.0005 | 0.01 | Enabled |
| Exact-action control | 0.002 | 0.60 | Disabled |
| Shorter trace | 0.002 | 0.60 | Enabled |
| Combined with shorter trace | 0.0005 | 0.01 | Enabled |

The first five arms retain eligibility decay 0.9. The last two use 0.5:
compare shorter trace with original, and combined with shorter trace with
combined. This reduces the trace half-life from about 6.58 decisions to one
decision. Eligibility and reinforcement traces already reset after each
completed game. Shorter traces test within-game credit duration; they may
also lose useful delayed credit and reduce total update magnitude. They do
not implement action-specific or compartment-specific credit assignment.

New JSONL and dashboard-snapshot metrics report `net_strengthened_fraction`
and `net_weakened_fraction` relative to initialization, plus
`max_learner_near_bound_fraction`: the worst fly's fraction of weights within
5% of the efficacy range of either bound, including clamped weights. This
last metric provides advance evidence of saturation; the actual bound and
drift stop rules remain unchanged. These fields are diagnostics, not proof
of learning, and do not add trainable parameters.

The named `exploratory-weak-failure-v1` condition is explicitly post-hoc.
The 60x punishment reduction roughly matches the observed pulse imbalance;
equal pulse magnitudes do not ensure equal weight changes or correct credit
assignment. Progress and milestone rewards remain fixed. The lower learning
rate tests update speed separately; it cannot correct a biased update sign.

Each screening arm uses one fly so its frozen outcome is unambiguous.
Train the original reference first: the disabled-plasticity control
consumes its exact action schedule. Configs retain the same training seeds,
motor exploration, anatomy, Red Deck / White Stake, fixed Ante-1 target, and
health thresholds. Checkpoints are every 250 decisions. Do not resume a
safety-stopped checkpoint to force a pilot to complete.

New benchmark/plasticity qualification report paths are empty intentionally;
old reports are not evidence for new candidate rules. Run short calibration,
health, representation, and reachability checks before extending exposure.
Evaluate all arms on the same validation cohort (offset 10000), inspect actual
blinds beaten and Ante-1 wins, and repeat with matched independent training
seeds before selecting a candidate. Tuning on this cohort requires a separate
confirmation cohort afterward; final-test seeds remain locked. Stability,
reward magnitude, and weight changes alone do not establish improved play.

For a multi-fly checkpoint, the frozen evaluation CLI requires an explicit
zero-based `--learner-index`. It loads and verifies the full source checkpoint,
then copies the selected learner's state without averaging weights or modifying
the source. Reports distinguish the full source checkpoint weight hash from
the selected learner's frozen weight hash. `--initial-weights` can restore that
learner's initial efficacies for a separately labelled baseline. Transition
records remain available; replay bundles are suppressed because the generic
replayer does not yet reproduce selection or baseline overrides. Unit tests
pass, but real GPU selection and sequential/batched parity still need checking
before using this path for independent-seed confirmation.
