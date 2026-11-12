# Sampling and delayed-discard learning studies

Implemented 27 October 2026 after fresh validation of the
[separate-head pilot](PLAY_DISCARD_LEARNING_RESULTS.md) produced no Ante-1 wins.
The independent scripts leave production code, the hash-bound pilot harness,
mapping artifacts and source checkpoints unchanged.

## Frozen greedy versus training-sampled policy

Entry point: `scripts/evaluate_motor_sampling.py`.
Output: `artefacts/motor-sampling-20261027-v1/`.

The declaration precedes outcomes. Each initial policy and all three trained
learners in both plastic arms use 128 validation roots, 20180000–20180127.
The policies are:

* **Greedy:** highest legal action/count scores, greedy card membership; no
  subset exploration.
* **Training-sampled:** the source softmax action/count selection, original
  5% uniform epsilon and each condition's original card-subset sampler.

Neural seed recipes, source weights, mapping, native environment, outcome
mapping and all learning parameters stay fixed. The sampled policy uses fresh
stateless evaluation IDs rather than replaying training choices. Every plastic
state field is hash-checked before/after. A 128-decision cap records unfinished
games as truncated rather than silently forcing an action. None truncated.

This comparison changes action/count/card sampling together, matching the
source training policy. It does not isolate discarding from those other
selection effects. Every policy shares the same 128 roots; these are repeated
responses of six trained policies, not thousands of independent trained flies.
No fitting, training or final-test seeds are used.

The study completed in 268 seconds, 2,048 evaluation episodes. Independent
native replay verified all 12,105 transitions, rewards and episode endings.
All source plastic state and source files stayed unchanged.

| Arm / policy | Initial mean return | Trained mean return, three learners | Trained discard fraction |
| --- | ---: | ---: | ---: |
| Subset, greedy | 0.21721 | 0.22010 | 4.06% |
| Subset, training-sampled | 0.27480 | 0.26629 | 29.21% |
| Top-k, greedy | 0.21721 | 0.22941 | 4.58% |
| Top-k, training-sampled | 0.27513 | 0.26674 | 27.68% |

Discard fractions use PLAY/DISCARD actions as the denominator. Initial greedy
discard fraction was 3.58%; initial sampled fractions were 28.21% subset and
27.71% top-k. PLAY's normalized score was generally higher than DISCARD's in
visited states, including at initial weights. Sampling therefore explains much
of the discrepancy between training and greedy discard frequency.

No policy won Ante 1. Trained greedy models cleared zero subset / one top-k
blind in 384 episodes per arm; sampled models cleared one subset / four top-k.
Each sampled initial policy cleared one blind in 128 games. Those sparse counts
do not show a robust learned improvement. In both arms, average sampled trained
return was below its sampled initial control. More frequent discards and higher
sampled scores are present before learning too; changing evaluation policy is
not a demonstrated learning fix. Retain both explicitly labelled evaluations
rather than replacing the greedy benchmark after inspecting results.

## Native discard → fixed-PLAY curriculum

Entry point: `scripts/train_discard_play_curriculum.py`.
Output: `artefacts/discard-play-curriculum-20261027-v1/`.

The declared capacity probe starts fresh unit weights with three independent
plastic learners and three paired frozen controls. Each receives 5,000 trials
(one DISCARD plus one PLAY each), 60,000 total native decision exposures.
Opening native states contain exactly one pair, with balanced pair-rank quotas:
104 training snapshots from training offset 9000, 52 disjoint validation
snapshots from validation offset 160000. Displays are independently shuffled;
case selection, source states and random seeds are paired across controls.
Pair labels never enter the runtime policy, feedback or synaptic update.

The first action is forced to DISCARD; the fly chooses its card count and
membership using its existing sampling rule and separate DISCARD outputs.
The actual native discard yields zero immediate reinforcement. Eligibility is
retained, then decays by the source factor 0.9 across the next decision.

The subsequent PLAY is chosen greedily by the frozen **initial-weight** neural
policy on the actual resulting hand. PLAY supplies no fresh eligibility. Its
native progress/clear pulse updates only the previous DISCARD traces in plastic
lanes. This freezes the whole follow-up policy, including its neural response,
so an improvement cannot come from changing PLAY weights instead. Source
learning rate 0.002, 20/20 Hz references, efficacy bounds, rewards and 5% subset
sampler remain unchanged. Artificial task resets add no failure/win pulse.
Traces reset only after the PLAY reward has been applied.

Before/after frozen evaluations use every validation hand with four neural /
display repeats and both greedy and sampled DISCARD selection. Report native
follow-up progress/clears, count, pair preservation and free-action preference.
Pair preservation is diagnostic, not a universal best-discard label.
An additional uniformly random DISCARD with the **same count** is evaluated
offline using the same fixed follow-up policy. This helps distinguish learned
membership from a change in count; its outcomes never train the fly.

Only DISCARD card/count edges may change. Require exact frozen efficacy and
choices, unchanged non-DISCARD efficacy, no bounds hit and unchanged source
state. A negative-control test that clears eligibility after DISCARD produces
zero updates, verifying that the delayed reward actually reaches the first
choice. This is a capacity test of which cards/count to discard, not training
when to choose DISCARD: the action-type head is forced and receives no fresh
choice credit.

The native qualification used 16 trials per lane and passed source/frozen
checks. Eight new tests and related regressions passed (73 total), including
NumPy/Torch delayed credit, trace cutting, identical frozen outcomes, same-count
controls, transfer-state validation and independence of learning from diagnostic
pair labels.

### Completed curriculum results

The full run completed all 5,000 trials per lane in 1,380 seconds. Before/after
validation included 208 trials per learner per selection policy (52 cases ×
four repeats). An independent CPU audit replayed all 4,992 native follow-up
plays across plastic/frozen lanes, stages and policies. All frozen choices and
weights stayed exact. Only DISCARD card/count efficacy changed; original source
state and files stayed exact. Plastic mean absolute efficacy drift was
0.00180–0.00183, maximum 0.02643–0.02694; no bounds were hit.

The primary outcome is the fraction of the 300-chip opening blind scored by
the subsequent native PLAY, averaged over the three plastic learners:

| DISCARD policy | Before progress | After progress | Paired change | Mean discard count, before → after |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.06996 | 0.06772 | −0.00223 | 2.79 → 3.00 |
| Training-sampled | 0.07196 | 0.07094 | −0.00101 | 3.60 → 3.81 |

Greedy progress changes by learner were −0.00333, −0.00144 and −0.00192.
Sampled changes were −0.00444, +0.00104 and +0.00035. Case-bootstrap 95%
intervals for the mean change were [−0.00637, +0.00182] greedy and
[−0.00737, +0.00533] sampled. These resample whole validation hands and are
conditional on these three learners; they do not estimate variability over
new training/mapping replicates. No follow-up PLAY cleared the blind.

Post-training progress also lacked a consistent advantage over uniformly
random membership with the same discard count. Per-learner differences were
[−0.00205, +0.00226, −0.00526] greedy and
[−0.00218, −0.00981, +0.00548] sampled. Membership changed on 125–141 of 208
greedy trials and 148–156 sampled trials per learner. The updates therefore
changed behavior without demonstrating better choices. Pair retention fell
from 47.60% to 43.59% greedy and 33.17% to 27.88% sampled. Pair retention alone
does not establish discard quality, especially when discard counts change.

The post-hoc observable-hand audit found a limited follow-up policy: it used
about 30–31% of the immediate progress available from each resulting hand.
Mean best available plain-hand chips changed 122.92 → 121.42 greedy and
120.09 → 125.61 sampled. The sampled potential increase was mixed across
learners and did not become better native PLAY progress. This is a reason to
probe PLAY competence and reward contrast, not evidence that the learner
cannot learn discards. No future draw information or best-hand labels entered
training. `scripts/assess_discard_curriculum.py` reproduces this assessment.

The delayed update path is functioning, but this budget/task did not demonstrate
useful DISCARD learning. It does not qualify a longer curriculum or a change
to punishment. Forced action timing remains a separate limitation.

## Free-action transfer probe

`scripts/evaluate_discard_curriculum_transfer.py` reads the isolated curriculum
state, preserves the source checkpoint, and restores ordinary native action
choice. Evaluate the three plastic replicates and initial weights under both
selection policies on 64 additional validation roots, 20200000–20200063.
No learning occurs. Because action timing was forced during the curriculum,
transfer is a diagnostic and cannot be described as trained action timing.

The probe completed all 512 episodes in 70 seconds, with zero truncations.
Independent native replay verified all 3,025 transitions, rewards, episode
endings and returns. Source states/files and the isolated curriculum NPZ stayed
unchanged.

| Free-action policy | Initial return | Trained mean return | Initial clears / episodes | Trained clears / episodes | Ante-1 wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 0.23567 | 0.24877 | 1 / 64 | 4 / 192 | 0 |
| Training-sampled | 0.25099 | 0.25530 | 0 / 64 | 0 / 192 | 0 |

These are the same 64 game roots for all models, not independent 192-root
validation blocks. Return changes by learner were [+0.00863, +0.03402,
−0.00333] greedy and [−0.00693, +0.00745, +0.01240] sampled. Small positive
mean changes on this block coexist with a declining learner in each mode and
the negative/flat primary curriculum outcome. They do not establish reliable
transfer. Greedy discard fraction was 4.06% initial versus 3.58% trained;
sampled was 27.27% versus 26.58%. The curriculum did not solve free-action
discard timing. Output: `artefacts/discard-curriculum-transfer-20261027-v1/`.

## Next decision

The proposed follow-ups have since been implemented and completed in
[learning-mechanism, exploration and follow-up-policy screens](TRAINING_BOTTLENECK_STUDIES.md).
The action head learned simple preferences; fresh exploration confirmation
matched initial weights; stronger PLAY primarily changed discard count.

Keep the interface fix and both labelled frozen evaluations. Next isolate the
actual card outputs and fixed-count membership, with update-magnitude controls.
The test-only reference player remains outside the production policy. Useful
membership learning should precede action-timing training and longer budgets.
Punishment reductions are not supported by these results; the capacity tasks
emit no failure punishment. Reward contrast, representation, trace duration and
mapping replicates remain separate factors.

## Reproduction

Use the established compiler environment and new output directories:

```bash
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/evaluate_motor_sampling.py \
  --pilot-dir runs/play-discard-learning-pilot-20261027-v1 \
  --output-dir artefacts/motor-sampling-repeat --episodes 128 --heavy

.venv/bin/python scripts/train_discard_play_curriculum.py \
  --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
  --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/discard-play-curriculum-repeat \
  --trials 5000 --replicates 3 --evaluation-repeats 4 --heavy

.venv/bin/python scripts/assess_discard_curriculum.py \
  --curriculum-dir artefacts/discard-play-curriculum-repeat

.venv/bin/python scripts/evaluate_discard_curriculum_transfer.py \
  --curriculum-dir artefacts/discard-play-curriculum-repeat \
  --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
  --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/discard-curriculum-transfer-repeat --episodes 64 --heavy
```

Reports bind their declarations, corpus and source hashes. The curriculum NPZ
is an isolated diagnostic state, not a generic checkpoint under the original
full-game training identity. Existing source checkpoints are never overwritten.
