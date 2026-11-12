# Fixed-pool normalization at 250 ms

This study tests whether fresh reward-free normalization reduces the strong
slot preference seen when 250 ms neural responses used the 100 ms motor
artifact. It changes only fixed pool baseline/scale statistics. Synaptic
efficacies, anatomy, sensory inputs, pool memberships, routing and exploration
settings remain fixed. No training or gameplay evaluation is launched.

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.motor_normalization_cli \
  --output-dir artefacts/card-normalization-new \
  --states 20 --repeats 4 --fit-repeats 4 --duration-ms 250 --heavy
```

The defaults select the completed pilot's choice-credit configuration and
checkpoint, containing three independent trained learners. Override them with
`--config PATH --checkpoint PATH`. Use a fresh output directory.
`--prepare-only` collects the reserved simulator observations and writes the
declaration without neural simulation. Package installation provides the
`flylatro-test-motor-normalization` entry point.

## Calibration and held-out data

Calibration uses the existing 96-state reward-free corpus. One shared
normalization is fitted from four independent neural repeats under the saved
**initial unit efficacies**, giving 384 neural samples at 250 ms. The fit accepts
only raw fixed-pool rates. It uses the existing median baseline and
`max(IQR / 1.349, source minimum scale)` estimator for all 35 pools. Pool
membership is never reselected. No trained learner's rewards, choices, hand
quality or win results enter fitting, and no hyperparameters are selected using
the test results. This is a normalization experiment, not a full qualification
of every contextual action head.

The reserved corpus contains 512 observations collected by seeded uniform legal
navigation over eight new validation root seeds, 20010000–20010007. Navigation
never reads action quality or outcomes. Actual run identities, observations,
masks and snapshots are recorded through the existing corpus implementation.

The test selects 20 distinct playable runs with at least six visible selectable
cards. Selected state hashes and run identities are excluded if present in
calibration. Root seed streams and derived neural seeds are also disjoint. One
state per run is selected using legality and corpus order only. The first,
sixth and last eligible card are each probed with isolated rank and suit edits.
This directly tests the previously dominant sixth slot as well as the edge
slots. Final-test seeds remain reserved. These states are held out from this
calibration, not claimed as a confirmatory final test.

`declaration.json` pins source config/checkpoint/motor/corpus hashes, source code,
all fit/test seeds, the exact case plan, fit policy and follow-up screen before
neural calibration begins. The reserved corpus is saved as
`reserved-states.npz`. Reproduction checks duplicate the first fit/test neural
seed and require exact outputs. Every plastic-state field and all source files
must remain unchanged.

## Paired readout comparison

The source checkpoint is strictly verified with its original 100 ms model and
motor artifact before a scoped 250 ms intervention. Each of the three trained
learners and its initial-weight control is simulated once per test input/seed.
Both normalizations read **the same cached neural responses**. Original scores
must reproduce the saved source decoder scores exactly before candidate scores
are computed. The candidate therefore cannot gain from a different neural RNG
draw, changed weights, or a different set of motor neurons.

`fresh-motor.json` stores the separate candidate artifact, including its new
normalization/calibration hashes and a membership hash equal to the source.
It is not substituted into the source checkpoint or training configuration.
Generic checkpoint compatibility remains strict. `report.json` records source
identity and experimental 250 ms dynamics separately.

Raw fit/test neural arrays and derived candidate scores have recorded hashes.
The latter contain only new card scores and legal-card contrasts; they reference
the source neural file rather than duplicating the identical KC/MBON data.
The report is updated after each learner, and only `status: completed` marks a
successful run. `assessment.json` records the screen and all learner outcomes.

## Predeclared descriptive screen

A follow-up signal requires a strict majority of trained choice learners to
meet both conditions on held-out states:

- The most common top-card slot becomes less dominant and accounts for at most
  50% of original-input probes.
- Both rank and suit edits have positive relative-card-score effect/noise power
  ratios exceeding the old normalization, with mean ratios improving for both
  edit types across the learners.

State/seed disjointness, unchanged plastic state, and neural health must also
pass. KC/card-pool active fractions must meet source safety minima; at most
25% of either population may exceed 200 Hz. A reduction in slot preference
alone is insufficient. Signed cross-repeat estimates and the limitations of
four repeats are retained as in the preceding information-flow study.

The top-card statistic is a readout preference, not a complete played subset
or a measure of action correctness. Input edits are visible-input interventions,
not simulator-world modifications. Neither normalization relief nor a passed
screen establishes learning or improved gameplay. Hand-quality and gameplay
validation would still be needed before committing to longer training.

Tests cover the unchanged median/IQR estimator, fixed membership/routing,
calibration/test state/run/seed separation, direct sixth-slot probes, exact
agreement with the existing decoder formula, rejection of bias relief without
content sensitivity, frozen-state handling and hashed cached-response artifacts.
A real native smoke run also passes calibration and test repeat checks.

## Completed results, 2026-10-24

The study completed in 348.5 seconds in
`artefacts/card-normalization-20261024-v1`. All source files, nine plastic-state
fields and motor pool memberships remained unchanged. The fit used 384
initial-unit-weight neural samples. The held-out comparison covered 20 distinct
new runs, 60 state/slot cases and 120 rank/suit edits, with four neural repeats
and each of three learners' initial/trained weights.

All **936 exact-repeat checks** passed. Selected test state hashes, actual run
identities, root seeds and neural seeds are disjoint from calibration. Initial
neural arrays agree across all three learners. All 13 fit/raw/derived neural
array files passed their stored hash checks. The full regression passed
**599 tests**, including native CUDA checks; see `regression.stdout`.

`report.json` contains the complete fit and paired results; `assessment.json`
records comparability PASS, all health checks passing and **predeclared
follow-up signal false**. `summary.json` is a compact extract. The source and
candidate motor manifests prove identical membership/routing; the separate
candidate is `fresh-motor.json`. Execution logs are `run.stdout` and `run.stderr`.

| Trained learner | Dominant-slot share, old → fresh | Rank effect/noise ratio, old → fresh | Suit effect/noise ratio, old → fresh | Full learner screen |
| --- | --- | --- | --- | --- |
| 0 | 98.75% → 50.00% | +0.02667 → -0.01530 | -0.00278 → -0.02417 | Fail |
| 1 | 97.50% → 57.50% | -0.01044 → -0.05633 | +0.02666 → -0.03372 | Fail |
| 2 | 98.75% → 48.75% | -0.00099 → +0.01727 | +0.02663 → +0.04877 | Pass |

Only one learner passes, below the required majority. Mean rank effect/noise
falls from +0.00508 to -0.01812; mean suit effect/noise falls from +0.01684 to
-0.00304. Negative estimates are retained finite-sample values, not negative
biological signal power. Four neural repeats provide limited precision; no
confirmatory learning claim follows from either the passing learner or the
group averages.

The sixth slot is top-ranked in 236/240 original-input probes under the old
normalization (98.33%) and only 11/240 under fresh normalization (4.58%). The
new dominant position is the second slot, at 125/240 probes (52.08%). Thus the
old sixth-slot lock is substantially relieved while a remaining position
preference persists. Mean top-card agreement across repeats drops from 96.67%
to 29.44%. These are highest-card-score preferences, not complete played hands
or measures of card quality. This new held-out cohort differs from the earlier
window study, so comparisons should use the paired old/fresh results here.

The result supports a contribution from normalization to the observed slot
preference. It does not demonstrate better card sensitivity across learners or
improved gameplay. Reinterpreting identical neural activity with a different
affine normalization cannot create new neural information. It can change score
weighting and choices, which is why slot relief and content sensitivity were
screened separately.

Retain the existing 100 ms training setup. The fresh artifact remains a
diagnostic candidate; neither it nor 250 ms has been installed into training.
The evidence does not justify another long training commitment yet.

The follow-up [offline identity probe](CARD_IDENTITY_DECODING.md) now finds clear
rank and suit information in KCs, with perfect decoding from sensory drive.
MBON rank decoding fails the held-out comparison and suit decoding is weak.
Investigate the KC-to-MBON pathway and credit/readout before changing the encoder
or committing to more training.
