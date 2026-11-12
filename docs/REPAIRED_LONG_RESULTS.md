# Repaired native long run: partial progress and saturation stop

Reviewed 31 October 2026. Source run:
`runs/repaired-native-long-20261028-v1`. Review artifacts:
`artefacts/repaired-native-long-review-20261031-v1/`.

## What completed

The declared million-decision budget **did not complete**. Both arms finished
the 10k, 30k, 100k and 300k milestones per learner. Plastic training then stopped
on 29 October at 03:23 UTC, after **509,922 decisions per learner** (1,529,766
across three learners), because learner index 1 crossed the configured 10%
bound-saturation threshold. The frozen arm remains at 300k decisions per learner.
Total training exposure was 2,429,766 decisions, of 6,000,000 planned.

The stop was `bound_saturation`, not a CUDA failure or nonfinite activity. At
the stop, 10.0098% of one learner's efficacies were at the upper bound of 2.0;
the global upper-bound fraction was 3.3366%. All efficacies remained finite and
within [0.2, 2.0]. Maximum learner mean absolute drift was 0.15694. The 10%
threshold is an engineering guard, not a proof that further learning is
impossible. The launcher intentionally refuses automatic resume after this stop.
Changing that threshold would require a declared continuation.

The last healthy periodic checkpoint is:
`training/subset-plastic/stage-1000000/attempt-000/plastic-checkpoint-000001500000.pkl`
under the source run, representing **500k decisions per learner**. The complete
safety-stop checkpoint is also retained. All eight completed-stage checkpoint
hashes, the last healthy checkpoint hash and the stop checkpoint hash verified.
All four frozen checkpoint states remain exactly equal to initial efficacies.

## Learning curve

Each row evaluates three learners on the same 256 validation roots: 768 policy
episodes, not 768 independent game roots or training replicates. Counts below
are total blinds cleared, not complete Ante-1 wins or necessarily first blinds.

| Decisions per learner | Plastic blind clears | Frozen blind clears | Plastic Ante-1 wins |
| --- | ---: | ---: | ---: |
| Initial | 3 | 3 | 0 |
| 10k | 1 | 3 | 0 |
| 30k | 2 | 3 | 0 |
| 100k | 19 | 3 | 0 |
| 300k | 24 | 3 | 0 |
| 500k | 23 | 3* | 0 |

The review newly evaluated all three 500k learners, without training. Their
clear counts were [5, 10, 8], and mean return gains versus initial were
[0.13124, 0.14506, 0.14550]. At 300k the gains were
[0.14280, 0.13443, 0.14771]. Progress emerged after the tiny pilots, but the
300k–500k observations show a plateau rather than an accelerating breakthrough.

*The frozen arm was not trained to 500k. Its verified unchanged weights and
the prior frozen evaluations provide this constant baseline; no additional
500k frozen exposure is claimed.*

There were **two Ante-1 wins in 262,409 training episodes**, at roughly 123k and
272k decisions for the respective learners. Training uses stochastic choices;
these rare events do not establish learned competence or greedy validation wins.

## Independent fresh check

The review selected the last healthy periodic checkpoint by its position before
the stop, not by evaluation outcomes. It froze all three 500k policies and
evaluated 256 new validation roots, 20900000–20900255, disjoint from the curve's
20850000–20850255. The equivalent initial unit-weight policy was evaluated once
on the same new roots; initial policies are identical across learners.

Trained clear counts were **[6, 3, 5]**, totaling **14/768 episodes**. Initial
weights cleared **0/256**. All were first Small-Blind clears; there were zero
Ante-1 wins. Mean return gains were [0.14467, 0.11370, 0.14071]. A paired
case bootstrap gave a conditional 95% interval [0.10991, 0.15713] for mean return
gain averaged over these three learners. This resamples 256 shared game roots;
it does not estimate uncertainty across a population of independently trained
flies. Final-test seeds remain unused.

## What the decisions show

Two native replay audits verified **22,756 transitions across 17 evaluation
files**, including **7,412 native counterfactual play checks**. Saved report
paths are `decision-audit/report.json` and
`checkpoint-500k-decision-audit/report.json` under the review directory.

Diagnostics below cover plain first-Small-Blind states only. Chip values are
immediate score, and the random comparator chooses a subset of the fly's recorded
card count on the same visited state. It is not a random full-game policy.

| Diagnostic | Initial curve baseline | 300k, three learners | 500k, three learners | 500k fresh, three learners |
| --- | ---: | ---: | ---: | ---: |
| Five-card fraction of PLAYs | 2.54% | 99.97% | 99.90% | 99.93% |
| Mean chosen PLAY score | 22.15 | 33.15 | 33.83 | 33.73 |
| Mean same-count random subset score | 20.57 | 34.99 | 36.12 | 36.46 |
| DISCARD decisions | 28 | 0 | 14 | 9 |
| Missed available immediate clears | 147 | 404 | 431 | 435 |
| First-blind failures with discards unused | 255/255 | 745/745 | 745/745 | 754/754 |

The initial column has 256 episodes, the others 768, so raw event counts are
not directly comparable as rates. At 500k there were 1,441 High Card plays with
a pair available on the curve roots, and 1,394 on the fresh roots. A pair's
availability alone is not always a strategic error, but missing guaranteed
clears and losing a final hand with unused discards are stronger evidence.

An illustrative audited 300k decision on root 20850004 passed up a 284-chip
diamond flush that would clear the blind, selecting a 15-chip High Card instead.
The hand was Queen C, Jack D, 9 D, 8 D, 6 D, 3 D, 2 C, 2 D; the fly selected
2 C, 6 D, Queen C, 9 D, 3 D. Four discards remained. The state and submitted
action were replayed; the alternative plain-hand score uses the audit scorer,
which was separately checked against native counterfactuals.

Long exposure has mostly produced a strong preference for playing five cards.
The same-count comparison does not demonstrate useful card membership learning.
Different checkpoints visit different states, so these diagnostics cannot
attribute every unit of progress causally to card count.

## Interpretation and next experiment

The short tests did not reveal all of the later partial improvement. However,
the result is not reliable Ante-1 competence, and provides no observed grokking
transition. The million-decision question remains open because the run stopped
early. Saturation and the five-card preference are concrete obstacles to address.

At the stop, 44.34% of plastic edges were stronger than initialization and 3.70%
were weaker. This supports investigating cumulative gain growth; it does not
prove the cause of poor card selection. Further reducing failure punishment is
not supported by these observations. An earlier centered-feedback fixed-hand
test also failed to improve selection, so centering alone is not an established
solution: see `CARD_REWARD_CONTRAST.md`.

Preserve these checkpoints as the baseline. The next proposed intervention is
a separately versioned **local postsynaptic strength-normalisation/competition
rule**, keeping only KC→MBON efficacies learnable. Hold the other learning,
reinforcement and exploration settings fixed. Evaluate card membership at a
fixed card count, alongside full-game progress and saturation. Preventing
saturation is a stability result; it must also improve conditional card choices
to count as gameplay progress. This is a hypothesis, not an implemented or
validated fix. No new training or rule changes were made during the review.
