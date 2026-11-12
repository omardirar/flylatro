# Frozen neural integration-window comparison

This experiment compares the same frozen flies at 100 ms and 250 ms per neural
decision. It tests whether longer integration reduces neural variability while
strengthening repeatable single-card responses. It performs no training and
makes no gameplay or hand-quality claim.

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.card_window_pilot \
  --pilot-dir artefacts/choice-credit-20261016-pilot-v1 \
  --output-dir artefacts/card-window-new \
  --durations-ms 100 250 --states 20 --cards-per-state 2 --repeats 4 \
  --reference artefacts/card-information-20261024-v1 --heavy
```

Use a fresh output directory. `--prepare-only` writes the fixed declaration and
commands without launching neural work. Additional increasing windows can be
explicitly supplied, but the first must equal the source training duration.
The package entry point is `flylatro-compare-card-windows`.

The runner writes `declaration.json` before neural work begins. It pins source
config/checkpoint hashes, the implementation, corpus identity, state/slot plan,
neural seeds, exact commands and the descriptive follow-up criterion. Jobs run
serially. `suite.json` records commands, PIDs, status and timing; each window
has its own stdout/stderr and complete information-flow report and raw arrays.

## Identity and controls

Each checkpoint is verified against its original 100 ms configuration and
component identities before any duration intervention. The analysis then changes
only `processor.duration_ms` in a scope that restores the original value on
success or failure. Candidate duration must be a positive finite whole number
of simulation steps. Rates use the actual duration in their denominator.

The source components continue to describe the trained checkpoint. A separate
`analysis_window` records the experimental duration and its dynamics hash. The
source configuration and checkpoint are never rewritten, and generic checkpoint
compatibility is never relaxed to load a differently timed model.

The integration window changes; timestep, anatomy, gain, plastic efficacies,
sensory mapping, motor pools, normalization and neural-reset policy stay fixed.
The fixed motor artifact was calibrated at 100 ms. Its use at 250 ms is an
explicit frozen diagnostic, not a claim of new motor calibration validity.
Skipping remains disabled and failure pulse remains 0.01 in source identity;
neither outcomes nor reinforcement are consumed by this experiment.

Both arms and all six source learners receive the same 20 states, two visible
card slots per state, original/rank/suit inputs, four neural seeds and their own
initial/trained weights. Each edited input uses the same neural seed as its
original. Neural state resets at every call. Longer windows run more simulation
steps; they do not receive additional environment decisions or learning updates.

The original-window rerun is checked against the earlier experiment with exact
comparison of all 12 neural array files. This compares array contents rather
than ZIP-container metadata. Candidate comparisons require equal source hashes,
corpus/cases, seeds, learner rows, population roots, all non-duration dynamics,
motor calibration and normalization. Sensory arrays and every neural file are
hash-verified. All plastic fields must stay unchanged.

## Declared follow-up screen

The choice-credit arm is primary; the global arm provides context. A longer
window supplies a descriptive follow-up signal only if:

- Both windows have finite, nonnegative rates and unchanged plastic state.
  Every measured KC/card-pool sample must meet the source safety minimum active
  fraction; at most 25% of either population may exceed 200 Hz.
- In a strict majority of the three choice learners, **both** rank and suit
  edits have lower KC and relative-card-score repeat RMS, and a positive
  card-score effect/noise power ratio exceeding the original window.
- The mean choice-arm card-score effect/noise ratio improves for both edit
  types, and mean top-card agreement across unchanged-input neural repeats does
  not decrease.

Reducing variance while also erasing the card response does not pass. Signed
cross-repeat effect estimates are preserved; negative values are finite-sample
noise estimates. Missing zero-noise ratios cannot provide a positive signal.
See [the information-flow experiment](CARD_INFORMATION_FLOW.md) for the precise
response/noise definitions and their limits.

`assessment.json` reports the comparison, health details, per-learner results
and `predeclared_follow_up_signal`. Passing is a reason for another controlled
experiment, not authorization to claim learning or to deploy a timing change.
Failing does not establish that every longer window is ineffective. These are
curated input interventions and only four neural repeats; they are not held-out
gameplay or tests of nonlinear/timing-based decodability.

The implementation tests duration restoration including exceptions, validation
of whole simulation steps, source verification before duration override,
rejection of non-duration model/seed/state changes, array hash checks,
predeclaration without GPU work, and rejection of quieter responses without
stronger content sensitivity. A real 250 ms native smoke run passes exact
repeats and frozen-state checks.

## Completed results, 2026-10-24

The matched study completed in `artefacts/card-window-20261024-v1`.
`assessment.json` reports **comparability PASS**, all health checks passing and
**predeclared follow-up signal false**. The 100 ms rerun exactly reproduces all
12 neural response files from the earlier information-flow experiment; see
`reference-audit.json`. All 2,400 exact-repeat checks passed, all plastic fields
and source files remained unchanged, and initial responses agreed across the six
learners within each duration. Both actual case plans, corpus identities and
seeds match the declaration. All 26 sensory/neural activity files passed hash
verification. The full regression passes **594 tests**, including native CUDA
checks; see `regression.stdout`.

Each duration used the same 20 states, 40 selected state/slot cases and 80
rank/suit edits, four neural repeats and both initial/trained weights. Diagnostic
wall time was 210.2 seconds at 100 ms and 448.2 seconds at 250 ms, including
loading, verification, repeated neural calls and artifact writing. These times
are not training throughput measurements.

Means below average the three trained choice learners. Relative-card-score
noise measures legal-card contrast vectors, not action error.

| Original-input repeat variability | 100 ms | 250 ms | Reduction |
| --- | --- | --- | --- |
| KC rate RMS | 4.211 Hz | 2.321 Hz | 44.9% |
| MBON rate RMS | 7.430 Hz | 4.475 Hz | 39.8% |
| Relative card-score RMS | 1.181 | 0.757 | 35.9% |

Longer integration reduces variability. The declared content-response criterion
fails in every choice learner:

| Learner | Rank card-effect/noise ratio, 100 → 250 ms | Suit card-effect/noise ratio, 100 → 250 ms | Both edits pass |
| --- | --- | --- | --- |
| 0 | +0.00192 → +0.03787 | -0.00498 → -0.04328 | No |
| 1 | -0.01485 → -0.02468 | +0.02725 → -0.05760 | No |
| 2 | -0.01259 → +0.02790 | +0.06729 → +0.01633 | No |

The mean rank ratio rises from -0.00851 to +0.01369; the mean suit ratio falls
from +0.02985 to -0.02818. Signed estimates close to zero have limited precision
with four repeats. No learner improves the criterion for both edits, versus
the required majority of two out of three.

Top-card repeat agreement rises from **18.3% to 93.1%**. A further descriptive
inspection shows a strong position preference: the sixth slot (zero-based
index 5) is top-ranked in **231 of 240 original-input probes, 96.25%**, pooled
over the three choice learners, 20 states and four seeds. The most common top
slot at 100 ms accounts for only 65/240 probes, 27.1%. These probes measure the
highest-scoring individual card, not the complete played subset or its quality.
The input edits target the first and last eligible cards, so they do not
establish sensitivity at every card position. The slot histogram is an
additional descriptive observation, not a substituted screen.

The supported decision is to **retain the 100 ms training setup for now**.
The 250 ms condition is quieter and more position-biased under the fixed
100 ms motor normalization; it does not pass the declared card-response screen.
There is no evidence here of improved hand quality or gameplay.

A plausible contributor to the position preference is the transfer of fixed
100 ms normalization to a different duration's neural-rate distribution. That
causal explanation is untested. A useful follow-up would keep the 250 ms pool
memberships fixed, recalibrate only baseline/scale using reward-free neural
data, and test on separately reserved states and neural seeds. Card-quality
scores must remain outside calibration. A successful calibration check would
still need frozen hand-quality and matched gameplay validation before a longer
training commitment.

The [fixed-pool normalization follow-up](MOTOR_NORMALIZATION_EXPERIMENT.md)
has now completed on new held-out runs. It relieves the sixth-slot preference,
but only one of three learners passes the card-response screen and the mean
rank/suit response estimates worsen. The 100 ms training setup remains current.
