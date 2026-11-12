# Single-seed long training: completed results

Reviewed 4 November 2026. Experiment:
`runs/single-seed-4000-long-20261101-v1/`.
[Methods and launch command](SINGLE_SEED_TRAINING.md).
Independent review: `artefacts/single-seed-review-20261104-v1/summary.json`.

**Conclusion:** both learning rules completed one million decisions per learner
and showed modest scoring progress. Neither learned to reliably solve the
repeated game, and no evidence of grokking emerged. Most score improvement
accompanied a shift toward playing five cards; membership selection remained
close to the mean of random subsets with the same count. Normalisation preserved
its strength budgets and reduced saturation, but has not established reliable
strategy learning.

## Completion and learning curves

Two rules × three independent learners × one million decisions = **six million
training decisions**, completed without a safety stop. Training took about
35.17 hours, excluding evaluation and process setup. Each game reset to the
same opening and hidden RNG state: stream seed 4000, Balatro seed `11HK6BPY`.
Actions and exploration could still produce different trajectories. The target
was Ante 1, which requires three blinds; this was not an Ante-8 experiment.

Saved frozen greedy evaluation results:

| Decisions per learner | Repeated seed, normalised | Repeated seed, original | Unseen first-blind clears, normalised | Unseen first-blind clears, original |
| ---: | ---: | ---: | ---: | ---: |
| Initial | 0/3 | 0/3 | 0/192 | 0/192 |
| 10k | 0/3 | 0/3 | 2/192 | 0/192 |
| 30k | 0/3 | 0/3 | 1/192 | 1/192 |
| 100k | 0/3 | 0/3 | 0/192 | 2/192 |
| 300k | 0/3 | 0/3 | 4/192 | 7/192 |
| 1m | 0/3 | 0/3 | 6/192 | 4/192 |

Each unseen evaluation uses the same 64 seeds across three learners. These are
192 trajectories, not 192 independent trained flies. Endpoints are 3.13% versus
2.08% first-blind success. Normalised per-learner clear counts are [3, 2, 1];
original counts are [1, 2, 1], each out of 64. A few additional clears do not
establish a general advantage of normalisation. No saved evaluation produced
an Ante-1 win. Endpoint total blinds cleared were seven normalised and five
original, including occasional second-blind clears.

There was some late progress, but no transition from reliable training-game
performance to reliable unseen-game performance. In particular, the original
rule's unseen clear count fell from seven at 300k to four at 1m. This finite run
cannot rule out a later transition; it supplies no compelling evidence that
simply extending the current setup will produce one.

## What the flies actually learned

All figures below concern plain first-Small-Blind states, before Jokers or
upgrades can alter scoring. The random comparison enumerates uniformly sampled
subsets of the **same count on the same visible hand**. It is a selection
diagnostic, not a random policy playing complete games.

| Diagnostic on unseen games | Initial | Final normalised | Final original |
| --- | ---: | ---: | ---: |
| Mean chips of selected play | 22.05 | 38.32 | 35.95 |
| Mean chips of random same-count subset | 20.64 | 37.93 | 37.25 |
| Five-card fraction of plays | 3.52% | 100.00% | 99.87% |
| DISCARD actions | 30 | 17 | 0 |
| Failed first blinds with unused discards | 192/192 | 186/186 | 188/188 |
| Plays missing an available immediate clear | 102 | 131 | 121 |

The larger chips total is useful partial progress. It largely accompanies
playing more cards; useful membership selection beyond the same-count random
comparison is not established. Extra nonscoring cards can also cycle the deck,
so a five-card play is not inherently a mistake.

The recurring opening gives a concrete membership failure. The visible hand is
K♥, K♣, J♦, 10♦, 9♦, 6♣, 3♠, 3♦. All six endpoint greedy learners select the
Kings and only one Three, scoring 60 chips, although retaining both pairs scores
92. On the original learners' next hand, two Jacks and two Eights are available;
they select only the Eights and score 52, although two pairs score 112. Repeated
exposure did not remove these mistakes. All six endpoint trajectories use no
discards on the repeated game's first blind.

Stochastic training did occasionally find successful paths:

| Training result | Normalised | Original |
| --- | ---: | ---: |
| Completed game attempts | 479,816 | 538,431 |
| Attempts clearing the first blind | 1,733 (0.36%) | 906 (0.17%) |
| Ante-1 wins | 9 | 8 |

These 17 training wins are real native outcomes, but exploration was active and
learning continued during those trajectories. They do not demonstrate a frozen
policy that can repeat the success. First-blind training clear rates did not
rise continually: the normalised rate was 0.55% in the 500–600k decision window
and 0.31% in the final 100k; the original rate was 0.41% in the 200–300k window
and 0.14% in the final 100k. The windows use completed attempts grouped by their
terminal decision, with shared game seed and persistent learning throughout.

## Weight stability and feedback

Normalised incoming-strength budget errors were at most 5.15e-8 relative,
independently recomputed from the saved efficacies and anatomical synapse
counts. Its maximum bound fraction was 1.35% in one learner, at the lower bound;
there were no upper-bound efficacies. The original rule reached 8.69% at the
upper bound in one learner, and some postsynaptic neurons' total incoming
strength grew about 84.9%. All efficacies remained finite and within bounds.
The safety threshold was not reached.

Reducing failure punishment further has no support here. Among final failed
unseen first blinds, the recorded sum of appetitive minus aversive pulses was
positive in 164/186 normalised and 166/188 original trajectories. This measures
supplied reinforcement, not net synaptic change or proof of causality. It does
show that many failures already receive positive cumulative feedback.

## What to do next

Preserve both checkpoints. Normalisation is the more stable platform for
further local-rule work, but neither version is a demonstrated strategic
learner. Prioritise understanding why the available two-pair selections do not
gain preference: inspect their sensory activity, selected-card eligibility and
resulting motor-score change under matched native feedback, with action type
and card count fixed. Existing fixed-count experiments were also weak, so this
must explain a learning failure rather than just repeat an old small pilot.
This run does not justify another unchanged long extension or weaker punishment.

## Verification

`scripts/review_single_seed.py` verifies the sealed plan, implementation/model
inputs, evaluation receipts and milestone checkpoint hashes. It independently
replays **all 12,305 saved evaluation actions across 2,340 complete games** on
the native CPU simulator, checking state hashes, rewards, terminal flags,
clear counts and win counts. It also checks all **1,018,247 training episode
records** for contiguous per-learner game counters and consistency with stage
summaries and final checkpoint totals. Native neural training updates were not
replayed. Final anatomical weight budgets and bounds are checked directly.

The review preserves replay inputs, per-learner audits, repeated-game decisions
and hashes under `artefacts/single-seed-review-20261104-v1/`. No new training was
launched and the historical learning implementation was not modified.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/review_single_seed.py \
  --run-dir runs/single-seed-4000-long-20261101-v1 \
  --output-dir artefacts/single-seed-review-20261104-recheck
```
