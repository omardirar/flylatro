# Native-score reward-contrast capacity experiment

Implemented 26 October 2026 after the
[reference-scaling comparison](CARD_REFERENCE_SCALING.md) reduced clipping
without demonstrating fixed-hand learning. This is an exploratory feedback
condition on disposable copies, not a change to the primary reinforcement
contract or existing checkpoints.

## Question and fixed design

Does signed feedback relative to a fixed native-score baseline improve the
fixed-hand capacity test? Under the original mapping, every scoring play gets
positive progress reinforcement. Its magnitude already varies with native
score; the question is whether centering this signal helps, not whether the
original feedback contains no score contrast.

`scripts/compare_card_reward_contrast.py` declares the feedback algorithm and
source hashes in `design.json` before collecting outcomes. Three copies start
from source learner 0's initial unit efficacies: native feedback, centered
feedback, and a frozen control. Keep the original 20/20 Hz reference rates,
0.002 learning rate, choice-gated rule, mappings, neural dynamics, 5% subset
exploration, hand/display and spike/exploration seed recipes. No rescaled
references or exploration increase is combined with this condition.

Use training case 0, native seed 6000, pair rank six. The pair curriculum fixes
PLAY/two cards and restores the same valid opening snapshot after each action.
Source states, the legacy harness and production code stay unchanged. Only
anatomical KC→MBON efficacies can acquire policy behavior.

## Feedback rule, declared before outcomes

First collect 128 actual native progress pulses from the frozen initial policy
on this training hand. Calibration spike seeds and exploration IDs are separate
from training and evaluation. Score only submitted choices; do not enumerate
unchosen actions, identify a best action or consult correct-pair labels.
All lanes get the same frozen calibration exposure. Freeze these two scalar
statistics for the entire subsequent experiment:

```text
b = mean(native calibration pulse)
a = mean(abs(native calibration pulse - b))
g = b / a
centered signed pulse = clip(g * (current native pulse - b), -2, +2)
```

The cap is the source reinforcement mapper's `max_total_pulse` (2 here).
Positive signed values reach the appetitive channel; negative values reach
the aversive channel by their absolute magnitude. These are relative-score
experimental signals, not actual run failures or claims about DAN biology.
Equal-to-baseline choices receive zero. High-scoring non-pairs can still get
positive feedback; there is no pair-specific teacher or hand-type bonus.

This sets mean signed calibration feedback to zero and matches its mean
absolute magnitude to the original mean native pulse. It changes valence and
contrast while controlling one pulse-size statistic. It does not match every
pulse, eligibility-weighted update, or realized training trajectory. Record
actual signed/absolute pulse sums, weight-change L1, weight drift and pulse
sign/cap counts. Refuse constant/nonfinite calibration or a cap that prevents
the declared calibration magnitude match. Do not retune after seeing results.

**This is outcome-based feedback calibration.** It is deliberately separate
from the project's reward-free sensory/motor calibration and must not be used
to fit those mappings. The scalar transform does not predict state/action
values, update during training or enter policy observations. It is not an
external trained policy/value network. Primary absolute-outcome feedback
remains unchanged.

## Training, controls and interpretation

Run 5,000 learning trials per lane. Native feedback remains the original
positive progress pulse. The centered lane applies only its fixed transformed
pulse. The frozen lane observes native feedback but applies zero update.
Clear eligibility and dopamine after every trial; task boundaries emit no
failure/win pulses. Pair labels are diagnostic counts after learning only.

Use the same 128 held spike seeds before and after training, with greedy
evaluation. These are repeated responses to one training hand, not independent
training replicates or held-out-hand generalization. The initial-policy reward
calibration adds 128 frozen trials per lane, in addition to the 5,000 training
trials and 256 before/after frozen evaluations. No learning occurs during
calibration/evaluation. Evaluation outcomes never select feedback parameters.

Require equal initial choices, exact frozen weights/individual choices,
unchanged source state/files, restored caller environment and no efficacy
bound hits. Check that the native lane reproduces the earlier fixed-hand
baseline. Any apparent improvement needs fresh training hands/seeds and an
independent evaluation block before long training or policy transfer claims.

## Qualification and frozen feedback parameters

Sixteen new tests and related regressions passed (93 tests total). The native
GPU qualification completed with 16 training steps, eight before/after
evaluation seeds and the full 128 feedback-calibration trials:
`artefacts/card-reward-contrast-20261026-smoke-v1/`.

The declared calibration produced:

| Statistic | Value |
| --- | ---: |
| Fixed native-pulse center | 0.0025364583765622234 |
| Mean absolute deviation | 0.00016886395997062198 |
| Fixed scale | 15.020720685476654 |
| Original / centered mean absolute calibration pulse | 0.00253645837656222 / 0.00253645837656222 |
| Centered mean signed calibration pulse | Approximately zero |
| Pulse cap | 2 |

The 128 calibration native pulses span only 0.00183333–0.00266667. The fixed
scale is large because those frozen-policy outcomes have little variation;
unseen larger training rewards can produce much larger centered pulses.
Calibration pulse matching is not a guarantee of matched training feedback
or synaptic update budgets. Preserve the declared scale and report those
differences rather than tune it after inspecting training outcomes.

## Completed native GPU result

The declared comparison completed in 332 seconds:
`artefacts/card-reward-contrast-20261026-v1/report.json`. An independent
`assessment.json` reconstructs the visible target cards, verifies evaluation
counts, and compares the native lane with the previous baseline.

| Measurement | Native plastic | Centered plastic | Frozen |
| --- | ---: | ---: | ---: |
| Training pair plays / 5,000 | 51 | 60 | 50 |
| Initial correct / 128 matched spike seeds | 3 | 3 | 3 |
| Final correct / same 128 spike seeds | 2 | 3 | 3 |
| Applied positive / negative pulse counts | 5,000 / 0 | 2,990 / 2,010 | 0 / 0 |
| Applied absolute pulse sum | 12.8940 | 16.2790 | 0 |
| Applied signed pulse sum | 12.8940 | 3.8284 | 0 |
| Actual cumulative update L1 | 93.1132 | 110.1961 | 0 |
| Mean absolute efficacy drift | 0.001379 | 0.000921 | 0 |
| Mean signed efficacy drift | 0.001379 | 0.000411 | 0 |
| Maximum absolute efficacy drift | 0.015943 | 0.011526 | 0 |

Centering changed the learning signal as intended: the centered lane increased
12.72% of edges and decreased 7.92%; native feedback only increased edges.
Centered efficacy ranged from 0.99560 to 1.01153. No pulse cap or efficacy bound
was hit. Smaller final drift does not imply a smaller update budget, because
positive/negative updates can cancel: cumulative update L1 was **18.3% larger**
than native, and absolute pulse sum was **26.3% larger**.

The centered lane did not improve over its initial evaluation or frozen control.
Its 60 training pair plays versus 51 native are not a demonstrated frozen-policy
gain. Counts are small, on one training hand and one initialization; this is
not evidence that every centered-feedback rule fails. The native lane exactly
reproduced all previous 250-step pair-frequency blocks and individual final
evaluation choices. Frozen weights/choices, original plastic state and all
hashed source files stayed unchanged.

Post-run diagnostic reconstruction found **zero correct-pair choices in the
128 feedback-calibration trials**. This label check occurred only in assessment,
never in feedback fitting. The initial-policy calibration missed the high-score
tail, helping explain why calibration pulse matching did not carry over to
training. Do not describe the comparison as perfectly magnitude controlled.

This result does not qualify centered feedback for long training or combining
with rescaled references. Next test broader uniform subset exploration as a
separate capacity condition, holding original native feedback and references
fixed. More discoveries are needed before interpreting weak learning as a
pure reward, representation or credit-direction failure.

## Reproduction

Use a new output directory:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/compare_card_reward_contrast.py \
  --condition artefacts/card-learning-20261024-v1/pair-pilot/conditions/subset-plastic/condition.json \
  --checkpoint artefacts/card-learning-20261024-v1/pair-pilot/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/card-reward-contrast-repeat \
  --steps 5000 --evaluation-repeats 128 --calibration-trials 128 --heavy
```

Outputs: `design.json`, `feedback-calibration.json`, `status.json` and
`report.json`. These are diagnostic reports, not resumable training checkpoints.
Tests cover sign and magnitude calculations, pulse caps, invalid calibration,
NumPy/Torch CPU state, unchanged source/frozen controls, failed-run restoration,
and independence of feedback/weight changes from diagnostic pair labels.
