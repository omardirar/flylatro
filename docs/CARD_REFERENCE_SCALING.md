# Matched reference-scaling capacity experiment

Implemented 26 October 2026. This follows the
[credit-direction audit and fixed-hand probe](CARD_CREDIT_DIRECTION_AUDIT.md).
The experiment changes plasticity normalization in disposable learner copies.
The completed checkpoints, production code, legacy harness and reward mapping
remain unchanged.

## Declared comparison

The independent entry point is `scripts/compare_card_reference_scaling.py`.
It writes the design and source hashes before observing training outcomes.
There are three lanes: original plasticity, rescaled plasticity, and frozen
control. All start from source learner 0's initial unit efficacies, rather than
its trained efficacies. Only the rescaled lane changes the two reference rates
and its matched learning rate. Anatomical edges, efficacy limits, motor mapping,
input encoder, native progress reward and 5% subset exploration stay fixed.

Reference selection uses the existing frozen calibration corpus
`artefacts/calibration-corpus-v783-balanced-long.npz`. Include all 20 states
where PLAY is legal and at least three cards are selectable. Use two frozen
spike seeds per state. No environment scoring, rewards, correct-pair labels,
validation outcomes or learned parameters enter calibration. The launcher
rejects overlap between calibration-state hashes and the pair corpus.

For each observation, count each connected KC/MBON once rather than weighting
reference selection by connection degree. Choose the 95th percentile of
positive rates, floored at the original 20 Hz. Then enumerate every legal
two-card choice to obtain the exact probability that each edge is credited
under uniform pair selection, accounting for overlapping motor pools.

Set the candidate learning rate to the baseline rate multiplied by the ratio
of baseline to candidate mean normalized coactivity, weighted by those edge
inclusion probabilities. This matches expected first-update L1 **per unit
reward pulse**, with zero traces and before float32 rounding, on the frozen
calibration states. It does not match every edge or each training update.
Reward/activity correlations, initial-hand differences and diverging policies
can produce different realized update budgets; the report records actual
weight-change L1 and native pulse sums for each lane.

Train all lanes for 5,000 decisions on the first training hand, case 0, native
seed 6000, pair rank six. Use the same display, spike seeds and exploration
IDs across lanes. The seed recipe is identical to the previous fixed-hand
probe. Score the actual submitted cards in the native simulator, preserve
ordinary positive progress feedback, and clear traces after each trial.
Pair labels are measurements only. The frozen lane observes its ordinary
reward but applies no weight update.

Evaluate 128 matched held spike seeds before and after training with greedy
selection. These are repeated evaluations of one training hand, not 128
independent games or held-out-hand generalization. Require identical initial
choices across lanes, exact frozen weights and individual choices, unchanged
source state/files, restored caller environment and no efficacy-bound hits.
Log clipping on the first 32 training decisions as a separate distribution
from calibration.

## Calibration and qualification

The native GPU qualification run completed with 16 training steps and eight
evaluation seeds: `artefacts/card-reference-scaling-20261026-smoke-v1/`.
The new tests and related regressions passed 77 tests, including independent
actual-update verification of the coactivity calculation, NumPy/Torch CPU
state, reward-free calibration and unchanged native reward mapping.

The fixed calibration chose:

| Parameter | Baseline | Rescaled |
| --- | ---: | ---: |
| KC reference | 20 Hz | 60 Hz |
| MBON reference | 20 Hz | 170 Hz |
| Learning rate | 0.002 | 0.004237078809252032 |
| Expected mean coactivity | 0.684247 | 0.322980 |
| Expected update / selected edge / unit pulse | 0.001368493 | 0.001368493 |
| Expected KC clipped fraction | 65.81% | 11.33% |
| Expected MBON clipped fraction | 99.95% | 17.37% |

Clipping fractions count rates at or above the reference and use expected
selected-edge weights. Percentiles count unique positive neuron observations.
Those are different measures; ties and connection degree mean a 95th
percentile does not imply precisely 5% clipped edges.

## Completed native GPU result

The declared run completed in 344 seconds:
`artefacts/card-reference-scaling-20261026-v1/report.json`. An independent
`assessment.json` reconstructs the visible target cards from the native snapshot,
checks the reported evaluation counts and records paired transitions.

| Measurement | Baseline plastic | Rescaled plastic | Frozen |
| --- | ---: | ---: | ---: |
| Training pair plays / 5,000 | 51 | 49 | 50 |
| Initial correct / 128 matched spike seeds | 3 | 3 | 3 |
| Final correct / same 128 spike seeds | 2 | 1 | 3 |
| Actual cumulative update L1 | 93.1132 | 112.1158 | 0 |
| Mean absolute efficacy drift | 0.001379 | 0.001661 | 0 |
| Maximum absolute efficacy drift | 0.015943 | 0.035402 | 0 |

The first 32 actual training decisions showed mean selected-edge clipping of
66.29% KC / 100% MBON for baseline and 14.23% KC / 45.08% MBON for rescaled.
Unit coactivity fell from 66.29% to 6.41% of selected edges. Thus rescaling did
change the saturation behavior on this hand; it did not remove all clipping.

The rescaled lane accumulated **20.4% more actual weight change**, despite
matching expected update magnitude on separate calibration states. Its pulse
sum was only about 0.37% greater (12.942 versus 12.894), so report the update
budget difference explicitly. Frozen calibration matching is not exact
trajectory matching and does not isolate clipping perfectly.

There was no demonstrated fixed-hand learning gain. The very small evaluation
counts do not establish a performance decrease or prove clipping is irrelevant
on other hands, initializations or budgets. This does not qualify the rescaled
references for long training. The baseline reproduced every prior 250-step
pair-frequency block and each final evaluation choice from the earlier
fixed-hand probe. Frozen weights/individual choices, original plastic state
and all hashed source files remained unchanged; no efficacy bound was hit.

The separate [native-score reward-contrast experiment](CARD_REWARD_CONTRAST.md)
is now implemented, using fixed scalar centering with original references and
exploration. Its algorithm is declared before training and pair labels stay
out of feedback. Increasing exploration is another separate factor, since
only about 50 correct choices were encountered in 5,000 trials here.

## Reproduction

Use a new output directory:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/compare_card_reference_scaling.py \
  --condition artefacts/card-learning-20261024-v1/pair-pilot/conditions/subset-plastic/condition.json \
  --checkpoint artefacts/card-learning-20261024-v1/pair-pilot/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/card-reference-scaling-repeat \
  --steps 5000 --evaluation-repeats 128 --heavy
```

Outputs include `design.json`, `calibration.json`, `status.json` and
`report.json`. Reports are diagnostics, not resumable training checkpoints.
Outcome-contrast feedback has its own separate experiment, with original
references held fixed, so it is not mixed into this comparison.
