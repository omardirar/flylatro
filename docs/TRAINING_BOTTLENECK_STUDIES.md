# Learning mechanism, exploration and follow-up-policy screens

Implemented 27 October 2026 following the
[discard-learning studies](DISCARD_LEARNING_STUDIES.md). The user requested all
three recommended checks. Entry points are `scripts/run_training_bottlenecks.py`
and `scripts/assess_training_bottlenecks.py`; no production source or completed
experiment implementation is changed.

## Shared design

All three screens load the current separate PLAY/DISCARD pilot strictly for
anatomy, neural dynamics, mapping, normalization and source parameters. Temporary
learners start at fresh unit efficacy, not the trained source weights. Each
screen has three replicate IDs, two treatments, and a plastic/frozen pair per
treatment: 12 lanes. Cases, neural seeds, display permutations and sampler keys
are paired across treatments and frozen controls within a replicate. Rewards
and later choices may diverge naturally. Counterbalanced reward targets are
paired copies, not six independent training initializations.

The native tasks use one corpus with 104 training snapshots and 52 disjoint
validation snapshots, eight/four per pair rank. Construction uses training
offset 10500 and validation offset 240000. All are ordinary plain opening
eight-card hands with exactly one pair. They are new snapshots relative to the
earlier curriculum. The corpus was generated during qualification and is
hash-bound and reused unchanged by the full screens. Cases are chosen without
scoring candidate actions. Training is position-shuffled for native card tasks.

Keep the original 0.002 learning rate, 20/20 Hz activity references, 0.9
eligibility decay, decision-local dopamine, 100 ms neural window, gain,
anatomical edges, efficacy bounds and reward mapper. Only the explicitly
selected motor head's KC→MBON edges can learn. Check all original plastic state
fields and source files, exact frozen weights/individual evaluation choices,
unrelated efficacies, clear fast traces at trial boundaries and zero bound hits.
Temporary NPZs are diagnostic states, not generic resumable checkpoints under
the source training identity. Final-test seeds remain unused.

These are bounded descriptive screens, not population-level significance
tests. All replicate policies share validation hands; shuffled displays and
neural repeats are repeated measurements. Report all learners and both
greedy/training-sampled evaluations, including unsuccessful results. Decisions
depend on before/after changes against frozen/initial controls, not selecting
the treatment with the highest raw post-training score.

## 1. Simple preference learning

3,000 decisions per lane, three fixed native training observations (one per
replicate); no native action is executed. Allow PLAY/DISCARD and let the
existing epsilon/softmax selector choose normally. Only action-type posts
receive fresh choice credit; card and count outputs cannot learn here.

One treatment rewards PLAY, the other rewards DISCARD. Correct choices receive
a declared diagnostic appetitive pulse 0.05; all others zero. This is an
artificial two-choice capacity task, not Balatro reward shaping or a proposal
to reward every discard. Targets stay fixed and are balanced across copies.
Frozen lanes receive no synaptic update. Clear eligibility/dopamine each trial.

Before/after evaluation uses 128 matched held neural seeds per replicate on
its fixed training observation. Record greedy choices, matched sampled choices
and the exact epsilon/softmax probability of the rewarded action. Probability
is important because an update may shift preference without crossing argmax.
Memorizing these states would demonstrate only elementary learning capacity,
not hand recognition, new-state generalization or game competence.

## 2. Broader uniform card exploration

5,000 trials per lane, 5% versus 25% uniform two-card subset exploration.
PLAY and count two are fixed in both treatments; only PLAY membership posts
can learn. The non-exploration choice is ordinary top-two neural scores, as in
the earlier pair task. The subset sampler, RNG key recipe and selected-choice
credit are unchanged apart from exploration probability. Both treatments use
the same balanced native states and shuffled displays.

Native PLAY progress supplies feedback; pair labels are counted only as a
diagnostic and never affect updates. Artificial resets emit no failure pulse.
Count discoveries during training, but require frozen before/after improvement
to describe learning. Frozen evaluation covers all 52 validation hands with two
display/neural repeats and both selection policies. Greedy evaluation is the
primary comparison of acquired preference; a higher sampled pair frequency
can come directly from the changed exploration probability.

## 3. Competent follow-up in discard learning

2,500 DISCARD→PLAY trials per lane. Compare the frozen initial neural PLAY
policy with a test-only reference that enumerates legal one-to-five-card
subsets of the **current visible hand after DISCARD** and chooses the highest
plain-hand immediate chip score. Ties prefer fewer cards, then slot order.
It does not inspect hidden cards, upcoming draws or alternate discard futures.
It is an explicit hand-scoring reference policy, not a learned fly or a change
to production observations/decoder. Native scoring verifies the selected play.

The fly chooses DISCARD membership/count using the unchanged source sampler.
Native DISCARD gives zero immediate feedback. Retain its eligibility across
the fixed-policy PLAY, decay once, and apply the actual native PLAY
progress/clear pulse only to previous DISCARD traces. PLAY adds no fresh
eligibility. Both follow-up policies use initial weights for any neural
activity. Artificial resets emit no failure pulse. Reference rewards and
realized update magnitudes may be larger; log these rather than claim an
update-matched comparison. There is no discard-specific bonus or pair teacher.

Evaluate both DISCARD selection policies on 52 validation hands × two repeats,
before/after. Also evaluate uniform random DISCARD membership with the same
count and the corresponding follow-up policy, without learning. Compare
learning changes within each follow-up treatment; the reference's higher raw
score alone is expected and is not discard learning. The task still forces
DISCARD timing and cannot establish when to discard in a free game.

## Reproduction

From the repository root, use new output directories:

```bash
set -e
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for study in binary exploration followup; do
  .venv/bin/python scripts/run_training_bottlenecks.py \
    --study "$study" \
    --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
    --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
    --output-dir "artefacts/training-bottlenecks-repeat/$study" --heavy
  .venv/bin/python scripts/assess_training_bottlenecks.py \
    --run-dir "artefacts/training-bottlenecks-repeat/$study"
done

.venv/bin/python scripts/confirm_subset_exploration.py \
  --run-dir artefacts/training-bottlenecks-repeat/exploration \
  --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
  --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/training-bottlenecks-repeat/exploration-fresh-confirmation \
  --heavy
```

Default corpus construction is deterministic. `--corpus PATH` reuses an exact
existing corpus. Reports record its path/hash and all bound source identities.
Assessments independently restore native snapshots and replay every recorded
before/after native action and progress outcome.

Fourteen new tests cover NumPy/Torch local updates, balanced targets and exact
epsilon/softmax probabilities, source/frozen/unrelated-weight invariants,
native reference score correctness, helper restoration on errors, diagnostic
label isolation, exploration-only membership changes, replay divergence,
saved-state corruption and frozen confirmation reproducibility. The new tests
and related targeted regressions passed 61 tests.

## Results

Full runs are in `artefacts/training-bottlenecks-20261027-v1/`.

### Simple preference learning: completed

The declared 3,000-decision run completed in 917 seconds. All source fields and
files stayed unchanged, frozen choices/weights were exact, only action-type
efficacies changed, no bounds were hit, and saved-state drift/count/trace checks
passed. Before/after evaluation used 128 held neural seeds for each fixed
training observation.

| Reward target | Mean choice probability before → after | Probability gains by learner | Greedy target frequency before → after |
| --- | ---: | --- | ---: |
| PLAY | 72.44% → 82.24% | +9.24, +10.06, +10.10 percentage points | 96.09% → 99.48% |
| DISCARD | 27.56% → 34.37% | +5.10, +9.54, +5.80 percentage points | 3.91% → 7.03% |

All six trained copies increased their assigned target's probability. This
establishes a bounded elementary learning effect for the current action head,
despite the earlier weak native card-task results. It does not establish card
membership learning or new-state generalization. The ordinary greedy policy
still rarely chooses DISCARD even when its probability has increased. Initial
PLAY preference also means the DISCARD-reward treatment experiences fewer
rewarded choices; conditional pulse size is equal, realized update budget is
not. Maximum edge drift was 0.221–0.245 PLAY-reward and 0.075–0.111
DISCARD-reward. Keep the artificial diagnostic reward outside native training.

### Broader exploration: completed screen

All 5,000 trials per lane completed in 1,542 seconds. Native replay verified
4,992 before/after PLAY actions. Frozen choices/weights and unrelated efficacies
stayed exact, source identities were unchanged, and no bounds were hit.

| Subset exploration | Training pairs, three plastic learners | Initial greedy pairs | Final greedy pairs | Mean native greedy progress before → after |
| --- | ---: | ---: | ---: | ---: |
| 5% | 534 / 15,000 | 11 / 312 (3.53%) | 14 / 312 (4.49%) | 0.05178 → 0.05300 |
| 25% | 516 / 15,000 | 11 / 312 (3.53%) | 18 / 312 (5.77%) | 0.05178 → 0.05446 |

The 312 evaluations per treatment use 52 shared hands × two repeats × three
learners. Initial greedy counts by learner were [3, 4, 4] / 104; final 5%
counts [3, 5, 6], final 25% counts [5, 8, 5]. All broader-exploration learners
improved greedy pair fraction and native progress on this block. This is a
small positive screen with sparse successes, not a robust gameplay improvement.
Broader exploration did **not** discover more pairs during training and had
slightly smaller cumulative update L1 (50.16–50.98 versus 52.17–53.37).
Do not attribute the validation result to greater successful-choice exposure.
Post-hoc training-log coverage was also mixed: the three 5% learners discovered
the pair on [78, 86, 91] of 104 distinct cases, versus [86, 83, 87] at 25%.
Every case received 48–49 exposures. `discovery-coverage.json` binds this
diagnostic to the exact training log; it supplies no consistent coverage
explanation for the small validation difference.

Sampled evaluation was mixed: 5% initial/final pair counts 10/14 of 312,
25% 11/13. Each treatment had one learner's sampled native progress decline.
Maximum edge drift was 0.01492–0.01524 at 5%, 0.01306–0.01341 at 25%.

Because this promising screen contains only 18 broader-policy pair selections,
`scripts/confirm_subset_exploration.py` declares an additional **frozen**
confirmation before considering promotion. It evaluates every trained policy
and paired initial control, greedy only, on 104 further validation hands
(eight per rank, validation offset 260000) with four display/neural repeats.
No further training or final-test data is used. It compares within-learner
25%/5%/initial choices and reports whole-case bootstrap intervals conditional
on these three trained learners. Fresh cases are disjoint from screen cases.

### Stronger follow-up: completed

All 2,500 two-decision trials per lane completed in 1,854 seconds. During the
run, 39,984 native follow-up scores were checked against plain-hand scoring.
Independent replay verified another 9,984 recorded before/after DISCARD/PLAY
actions. Source fields/files, frozen choices/weights, unrelated efficacy and
saved-state checks passed; no bounds were hit.

| Follow-up / DISCARD policy | Native progress before → after | Mean discard count before → after |
| --- | ---: | ---: |
| Initial neural PLAY / greedy | 0.07038 → 0.06718 | 2.67 → 2.92 |
| Visible-hand reference / greedy | 0.36835 → 0.41660 | 2.67 → 4.92 |
| Initial neural PLAY / sampled | 0.07603 → 0.07172 | 3.44 → 3.48 |
| Visible-hand reference / sampled | 0.39903 → 0.43657 | 3.44 → 4.81 |

Reference-policy progress increased in all three learners in both evaluation
modes. Greedy changes were [+0.02285, +0.03067, +0.09122]; sampled
[+0.05099, +0.05160, +0.01003]. Reference follow-up blind clears rose 25 → 36
of 312 greedy trials and 32 → 46 sampled. These are opening PLAY outcomes
using an explicit reference player, not full-game wins by a learned fly.
The weak neural follow-up had no greedy clear and one sampled clear both
before and after.

The clearest learned change is **count**: reference learners almost always
choose five discards greedily. Pair retention drops 46.15% → 14.10% greedy,
but changing count makes retention alone unsuitable as a quality label.

Random membership with those same counts improved almost as much: reference
greedy progress 0.36498 → 0.40793, a +0.04295 change versus the fly's +0.04825.
The change in same-count advantage by learner was
[−0.00003, +0.01862, −0.00269]; sampled [+0.00292, +0.02138, −0.01000].
Useful membership learning is therefore not established. A post-hoc
whole-case bootstrap, conditional on these three learners, gives 95% intervals
[−0.01591, +0.11152] for reference greedy progress change and
[−0.04474, +0.05513] for change in its same-count advantage. Both span zero;
the sample does not establish a population-level progress gain.
`paired-case-assessment.json` records both modes and all per-learner results.

The reference received 182–194 summed native appetitive pulse per learner,
versus 8.8–9.3 with neural PLAY. Cumulative update L1 was 1,578–1,690 versus
62–64, about 25–26 times larger. Maximum edge drift was 0.292–0.324 versus
0.0134–0.0142. The comparison changes both outcome quality and realized reward
strength; it does not isolate these explanations. Keep the reference outside
production and do not promote its rewards or weights as a qualified strategy.

### Fresh exploration confirmation: completed

All policies completed the 104 new hands × four repeats in 119 seconds.
Independent replay verified all 4,992 native actions. There was no learning,
source files/state were unchanged, and paired frozen controls agreed exactly.
Each treatment has 1,248 trials across the same 104 hands and three learners:

| Policy | Correct pairs | Pair fraction | Mean native progress |
| --- | ---: | ---: | ---: |
| Initial unit weights | 47 / 1,248 | 3.77% | 0.05184 |
| Trained with 5% exploration | 37 / 1,248 | 2.96% | 0.05027 |
| Trained with 25% exploration | 48 / 1,248 | 3.85% | 0.05135 |

Counts by learner were initial [18, 11, 18], 5% [13, 10, 14], 25% [19, 13, 16],
each out of 416 trials. The 25% models beat their 5% counterpart in all three
learners, but exceeded their own initial pair count in only two, and their own
initial native progress in only one. They produced just one additional pair
selection in aggregate versus initial weights.

The conditional whole-case 95% bootstrap interval for 25% minus 5% pair
fraction was [+0.00080, +0.01683]; for 25% minus initial it was
[−0.01282, +0.01362]. Native progress at 25% was slightly below initial
(−0.00049). These intervals are conditional on this mapping and three trained
learners; they do not establish population significance. More exploration
performed better than the lower-exploration trained condition on this block,
without demonstrating useful improvement over the untrained control.

## Decision after all screens

The current action head can acquire a simple rewarded preference. Extending
that claim to card membership would be premature. The 25% exploration
screen's small absolute gain did not reproduce against initial weights, and
the stronger follow-up mostly produced a change in discard count. No primary
training defaults or source checkpoints were changed; the reference player
and binary diagnostic reward remain outside normal training.

The subsequent [card-output and fixed-count studies](CARD_MEMBERSHIP_LEARNING.md)
have completed. Five of six simple card targets learned; the sixth had zero
eligibility despite repeated rewards. Native fixed-count gains stayed
preliminary on fresh cases and did not improve neural-PLAY performance.
Next add reward-free active-plastic-input qualification to the motor mapping,
then validate a versioned replacement with fresh learners and controls. A
further punishment reduction is not supported: these capacity tasks applied
no failure punishment. Neither a global sign reversal nor an unqualified
learning-rate increase follows from these results.
