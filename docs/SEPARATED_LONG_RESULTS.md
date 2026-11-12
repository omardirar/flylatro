# Separated-pool long run: completed results

Run: `runs/separated-long-20261024T182315Z`. Training completed on 24 October
2026 at 20:27:13 UTC. All ten scheduled stages and frozen evaluations completed.
The final checkpoint contains three independent learners, each exposed to
100,000 decisions: 300,000 total. No further training has been launched.

## Outcome and health

| Decisions per learner | Total blind clears / 768 validation games | Ante-1 wins |
| --- | ---: | ---: |
| 10k | 6 | 0 |
| 20k | 5 | 0 |
| 30k | 9 | 0 |
| 40k | 13 | 0 |
| 50k | 18 | 0 |
| 60k | 16 | 0 |
| 70k | 34 | 0 |
| 80k | 19 | 0 |
| 90k | 27 | 0 |
| 100k | 31 | 0 |

At 100k, the individual learners cleared [5, 13, 13] blinds on 256 games each.
All clears in these final evaluations were first Small Blinds. The initial
unit-weight control cleared one blind in 256 games. Final mean return gains
against that control were [+0.18689, +0.23701, +0.23131]. The curve shows delayed
improvement in partial progress and some blind clears; it does not establish
full Ante-1 competence. The 70k checkpoint's higher clear count is a validation
observation, not an independently confirmed best checkpoint.

The training run completed 40,056 episodes with zero Ante-1 wins and no safety
stop. Final efficacy bound fractions were zero; maximum learner mean absolute
drift was 0.10202. Activity/plastic state were finite and card-credit alias
metrics remained zero at the final decision. The final checkpoint's file
SHA-256 matched its manifest. The final checkpoint is retained as the baseline.

The same 256 validation roots are reused across learners and stages. There are
three training replicates, not 768 independent training replicates. These are
exploratory observations; final-test seeds remain unused.

## Final decision audit

The reusable CPU audit replayed all 4,409 transitions from the final 768 games,
checking recorded states, rewards and termination. Diagnostics below cover plain
first-Small-Blind states only. It independently checked 1,308 alternative plays
in the native simulator. Local machine-readable output:
`artefacts/card-learning-20261024-v1/audit-100k/report.json`.

| First-blind behaviour | 20k | 100k |
| --- | ---: | ---: |
| Mean selected play score | 22.73 chips | 37.92 chips |
| High Card fraction | 79.4% | 52.7% |
| Mean selected card count | 2.845 | 4.930 |
| Five-card play fraction | 20.1% | 95.7% |
| Uniform subset of the recorded count, same states | 23.65 chips | 42.84 chips |
| Failures with discards unused | 515/763 (67.5%) | 735/737 (99.7%) |
| Non-winning play with a winning subset available | 508 turns | 594 turns |
| Discard with a winning play available | 308 turns | 78 turns |

At 100k, the flies selected the maximum immediately available score on 323 of
3,059 plays (10.6%). There were 1,355 High Card plays with a pair present.

Choosing more cards can discover pairs by chance. The same-count comparator
shows that card membership remains weak even while scores improve. These are
different visited states at different checkpoints; the table alone cannot
attribute improvement entirely to card count. The random comparator is an
offline same-state calculation, not a random full-game policy evaluation.
Lower immediate score can sometimes be useful for drawing cards, whereas
missing a guaranteed clear or making a losing final play with unused discards
is stronger evidence of a mistake.

## Next experiment

Run the prepared one-pair capacity test first:
`artefacts/card-learning-20261024-v1/pair-pilot`.
Its four conditions compare top-k/subset exploration with plasticity on/off,
using three fresh learners per condition and 5,000 decisions per learner
(60,000 total training decisions). PLAY/count two are fixed, so pair selection
cannot improve merely by increasing card count. Chance is 1/28.

Evaluate on disjoint held-out pair snapshots against each learner's own initial
weights and frozen controls, then test transfer to normal Ante-1 games with free
action/count heads. A success must transfer; a failure directs work toward card
representation and credit. Keep failure punishment unchanged for this test.
The normal-game subset-exploration pilot remains prepared as the subsequent
controlled intervention. Commands: [Card learning experiments](CARD_LEARNING_EXPERIMENTS.md).
