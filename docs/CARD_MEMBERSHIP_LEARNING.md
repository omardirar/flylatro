# Card-output capacity and fixed-count discard learning

27 October 2026. Follow-up to the
[learning-bottleneck screens](TRAINING_BOTTLENECK_STUDIES.md), which found that
the action head learns simple preferences while stronger PLAY mostly teaches
larger discard counts. This study isolates the DISCARD membership outputs.

Entry points: `scripts/run_card_membership.py` and
`scripts/assess_card_membership.py`. The existing binary assessment script also
handles the new card-output diagnostic. All code is outside production and the
hash-bound completed training harness. Source checkpoints/configurations stay
unchanged; every learner starts at unit efficacy in temporary state.

## Shared conditions

Use the current single-MBON, separate PLAY/DISCARD mapping, original 100 ms
neural window, gain, anatomy, normalization, learning rate 0.002, 20/20 Hz
references, eligibility decay 0.9 and decision-local dopamine. Only DISCARD
card-output KC→MBON edges may change; type/count/PLAY edges remain exact.
No artificial failure punishment is emitted. Final-test seeds remain locked.

One deterministic corpus has 104 training and 52 validation opening snapshots,
balanced over the 13 pair ranks (eight/four per rank). New construction offsets
are training 14500, validation 280000. All hands are plain eight-card native
Small-Blind openings with exactly one pair. Quota labels do not enter the
policy or native feedback. The corpus generated during qualification is reused
by both full studies, with its path/hash recorded.

## Direct DISCARD-card preference control

Three replicate observations, each with a different pair of legal card slots.
Fix DISCARD and count one; restrict selection to those two otherwise legal
cards. Two plastic/frozen pairs per replicate reward opposite slot choices.
There are 12 lanes, 3,000 trials each. No native action is executed: a chosen
target slot receives an explicitly diagnostic appetitive pulse 0.05, all other
choices zero. It is a fixed-output memorization control, not teaching which
poker cards to discard or proposing a discard bonus in normal training.

Use the actual source membership sampler: 95% greedy top-card selection plus
5% uniform legal-subset sampling. This differs from the action head's softmax.
Given a neural response and two legal cards, the target's exact sampled
probability is 0.975 if it is greedy-first and 0.025 otherwise. Evaluate both
greedy and matched sampled choices over 128 held neural seeds before/after;
average the exact probability across seeds as well. Pair targets and source
neural/exploration keys within each replicate. Only the executed card output
receives eligibility. Clear traces after each diagnostic choice.

Success here would establish only that a DISCARD motor output can acquire a
preference on fixed observations. It would not establish card-content tracking,
held-out-hand generalization or a better game policy. Report low-reward
discovery and individual target failures rather than pooling away asymmetry.

## Native membership with count fixed at two

Run 2,000 DISCARD→PLAY trials for each of three training replicate IDs. Every
native DISCARD selects **exactly two** of eight cards. The source 5% subset
sampler and actual membership scores determine which cards. Type and count
receive no fresh eligibility; only selected DISCARD card posts can learn.
Keep eligibility across the next decision, decay it once, add no PLAY credit,
then apply the chosen branch's native progress/clear pulse. Clear fast traces
after the two-decision trial. Resetting the task supplies no failure/win pulse.

Each replicate has four local states:

1. Plastic DISCARD actor; native feedback from frozen initial neural PLAY.
2. Plastic DISCARD actor; native feedback from a visible-hand reference PLAY.
3. Plastic DISCARD actor; the reference feedback attenuated by a frozen scalar
   chosen to match initial expected update magnitude.
4. One shared frozen DISCARD actor, scored with both follow-up policies.

Sharing the frozen actor is explicit: its DISCARD weights, observations,
neural seeds and sampler keys are identical in all follow-up conditions, and
it never learns. Its single choice is scored under both PLAY policies; those
are two outcomes of one control choice, not independent controls. Before
training, every plastic choice must match it exactly. Frozen choices and both
outcome streams must remain exact after training.

Both PLAY branches score the **same actual hand after the chosen DISCARD**.
The reference enumerates legal current-hand subsets and picks the highest
immediate plain-hand chip score, with fewer cards and then slot order breaking
ties. It never sees hidden cards or alternative discard futures. It is a
test-only reference, not a learned fly or a production decoder change. Neural
PLAY uses initial unit weights throughout. Both branches execute in the native
simulator from the identical post-DISCARD snapshot. Only the branch assigned
to an actor supplies its learning signal; the other outcome is diagnostic.
The selected plays' analytic scores are checked against native progress.

### Initial update matching

Before training, collect 128 training-only frozen trials per replicate, with
independent display, neural and sampler keys. No efficacy changes occur.
For each choice, measure selected eligibility mass after the source 0.9 decay
and obtain both native appetitive pulses. Freeze one global scalar:

```text
weak budget = sum(learning_rate × weak_native_pulse × eligibility_mass)
reference budget = sum(learning_rate × reference_native_pulse × eligibility_mass)
matched gain = weak budget / reference budget
```

Use one copy per replicate rather than counting duplicate lanes. Require
finite nonnegative calibration, positive budgets and gain in (0, 1]. The
matched actor subsequently receives `gain × actual reference pulse`. Do not
retune the gain using training/evaluation outcomes. Calibration state is
discarded; all training states start fresh with zero counters/traces.

This matches **expected initial update L1**, before float32 weight rounding.
It does not match each pulse, reward variance or realized training update
budgets after choices diverge. Record actual pulse sums, cumulative update L1
and efficacy drift in every arm. The attenuation is an experimental feedback
condition, not the original native reward contract.

### Evaluation and checks

Before/after frozen evaluation covers all 52 validation hands with two
independent display/neural repeats, greedy and source-sampled membership.
Score every actor's actual DISCARD with both PLAY policies. Also score uniform
two-card discards with the reference, without learning. Compare within each
actor against its exact initial/frozen control, and report every learner.
Count cannot explain gains because it is fixed at two in every condition.

The assessment restores native snapshots and independently replays calibration
and both evaluation stages, including both PLAY branches. It checks saved
weights, counters, bounds, clear traces and calibration reconstruction.
Whole-case bootstrap intervals use 2,000 draws, conditional on the three
trained learners and this mapping; repeated seeds are not new training
replicates. The task still forces DISCARD timing, and neither its reference
nor binary reward is a qualified full-game improvement.

All three native treatment actors also receive a preplanned frozen greedy
confirmation on 104 additional validation hands (eight per rank, offset 300000)
with two display/neural repeats. `scripts/confirm_card_membership.py` preserves
every local/source plastic field, compares each actor with the same replicate's
shared initial actor, scores both PLAY branches and independently replays all
recorded actions. All treatments are evaluated regardless of screen outcomes;
no trained model or reward parameter is selected using these fresh cases.

Tests cover NumPy/Torch source/frozen/head isolation, forced counts and source
sampler behavior, eligibility-weighted calibration, trace-cut negative control,
gain isolation between learners, shared control equality and replay divergence.

## Reproduction

Use new output directories and the qualified compiler:

```bash
set -e
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for study in binary native; do
  .venv/bin/python scripts/run_card_membership.py \
    --study "$study" \
    --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
    --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
    --output-dir "artefacts/card-membership-repeat/$study" --heavy
done

.venv/bin/python scripts/assess_training_bottlenecks.py \
  --run-dir artefacts/card-membership-repeat/binary
.venv/bin/python scripts/assess_card_membership.py \
  --run-dir artefacts/card-membership-repeat/native

.venv/bin/python scripts/confirm_card_membership.py \
  --run-dir artefacts/card-membership-repeat/native \
  --condition runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json \
  --checkpoint runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/card-membership-repeat/fresh-confirmation --heavy
```

## Results

Outputs: `artefacts/card-membership-20261027-v2/`.

### Card-output preference: completed

The uninterrupted version-2 run completed all 3,000 trials per lane in 909
seconds. Source state/files, frozen choices/weights, unrelated efficacy, final
counters/traces and zero-bound checks passed. Across six trained copies,
mean exact target-selection probability increased 50.00% → 77.58%; five copies
improved and one was unchanged. Individual probabilities on 128 held neural
seeds were:

| Replicate / target slot | Initial probability | Trained probability |
| --- | ---: | ---: |
| 0 / slot 0 | 75.23% | 96.76% |
| 0 / slot 1 | 24.77% | 52.23% |
| 1 / slot 2 | 65.59% | 96.02% |
| 1 / slot 3 | 34.41% | 34.41% |
| 2 / slot 4 | 49.26% | 92.30% |
| 2 / slot 5 | 50.74% | 93.79% |

The unchanged copy received 1,070 rewarded choices but had exactly zero
efficacy updates. The other five had a nonzero update on every rewarded
choice. This demonstrates bounded preference learning on several actual
DISCARD card outputs and exposes a failure that pooling the six targets would
hide. It does not demonstrate card-content recognition or game competence.

### Post-hoc zero-eligibility diagnosis

`scripts/audit_card_membership_activity.py` reconstructs the initial fixed
observations and first 16 held neural seeds. It forces each diagnostic target
in disposable eligibility/weight copies, records actual selected-edge pre/post
activity and applies the same positive pulse. No native actions or runtime
teacher changes occur. Source plastic state/files remain exact.

For target slot 3, MBON root `720575940629529722` fired at a mean 35 Hz, but
**none of its 121 plastic KC inputs fired** in those probes. Fresh eligibility
mass and actual update L1 were zero in every probe. The other targets had
active plastic inputs and positive updates. The unit policy selected this
target greedily on 7/16 of the same seeds, so it can be chosen without any
eligible synapses participating. Its anatomical inputs comprise 73
`KCapbp-ap1`, 47 `KCapbp-ap2` and one `KCapbp-m` KC; no claim about the
biological cause of their model inactivity follows from those labels.

`scripts/audit_motor_learning_opportunities.py` then tested all 96 states in
the existing **reward-free calibration corpus**, with two neural repeats at
initial weights. It aggregates fresh eligibility by anatomical post neuron
and current motor pool, and reports coverage only where the corresponding
head has competing legal options. It applies no rewards, performs no learning
or native scoring, and changes no mappings. The same DISCARD option had zero
fresh eligibility on all 15 applicable cases / 30 case-repeat samples. Its KC
inputs were inactive across the full corpus even while the MBON fired.

The raw audit also flags the inactive SKIP_BLIND contract output; skipping is
disabled in actual training, so that is not an active-policy learning defect.
Other unused candidates with zero coactivity likewise are not automatically
policy failures. The actionable finding is the active DISCARD option.
Anatomical connection count alone would miss it: every assigned motor neuron
has at least ten plastic KC inputs, and this target has 121.

This identifies a concrete learning-opportunity gap in the current
encoder/dynamics/readout combination. It does not prove why those inputs are
inactive, that the neuron is globally unlearnable under every stimulus, or
that removing it alone solves card learning. A future reward-free mapping
qualification should check participating plastic inputs in relevant contexts,
in addition to output response/diversity. Any replacement needs a new mapping
identity, fresh learners and separate validation; existing checkpoints keep
their original meanings. Audit outputs are `card-activity-audit/` (including
anatomical coverage) and `motor-opportunity-audit/` under the version-2 root.

The target's 121 efficacies also remained exactly at unit value in **all nine
native learners** after the full fixed-count experiment, including the strong
feedback arm. `card-activity-audit/native-target-weights.json` records this
additional check. It is a concrete starting-state/observed-input learning
limitation, not evidence that the entire model or biological neuron cannot
learn under other stimulation.

### Fixed-count native screen: completed

All 2,000 DISCARD→PLAY trials per lane completed in 1,462 seconds. Only
DISCARD membership efficacy changed; counts stayed exactly two. Frozen choices
and both native outcome streams, source files/state, unrelated efficacy and
final counters/traces stayed exact, with zero bounds hit. Independent replay
verified 19,584 calibration/evaluation native actions, including both PLAY
branches from each actual post-DISCARD snapshot.

Calibration selected a fixed reference gain **0.0904121**. Initial expected
weak update L1 was 4.25064, raw reference 47.01404, attenuated reference 4.25064.
Realized cumulative training L1 was 21.84–24.18 weak, 294.60–315.34 reference,
24.21–25.54 matched. The attenuation matched the initial expectation, with
roughly 6–11% larger realized matched-arm updates than corresponding weak
learners. Maximum edge drift was 0.0088–0.0097 weak, 0.105–0.114 reference,
0.00945–0.00952 matched.

Greedy evaluation under each actor's assigned follow-up policy:

| Learning feedback / scorer | Mean initial progress | Mean trained progress | Paired change |
| --- | ---: | ---: | ---: |
| Neural native / neural PLAY | 0.06653 | 0.06946 | +0.00293 |
| Reference native / reference PLAY | 0.35660 | 0.36505 | +0.00845 |
| Reference matched / reference PLAY | 0.35660 | 0.36067 | +0.00407 |

These changes were mixed across learners and their conditional whole-case 95%
intervals all span zero. The stronger-reference actors scored with **neural**
PLAY had a secondary increase 0.06653 → 0.07418, positive in all three learners
with a conditional interval [+0.00043, +0.01618]. This is one cross-policy
measure in an exploratory screen; it is not a full-game improvement or a
population-level learning claim.

Sampled membership changes under assigned scorers were −0.00066 weak,
+0.02427 reference and +0.00525 matched, with mixed learner directions.
The unchanged-count setup removes the earlier count-learning explanation;
the preplanned fresh confirmation below checks these small/noisy changes.

### Fresh greedy confirmation: completed

All actors completed 104 new validation hands × two repeats, with both fixed
PLAY branches. Each treatment has 624 observations across three learners on
the same 104 roots. No source or local plastic field changed. Independent
replay verified 7,488 additional native DISCARD/PLAY actions.

| Trained DISCARD actor | Neural PLAY progress, initial → trained | Reference PLAY progress, initial → trained |
| --- | ---: | ---: |
| Neural native feedback | 0.06650 → 0.06532 | 0.36199 → 0.36334 |
| Reference native feedback | 0.06650 → 0.06631 | 0.36199 → 0.37807 |
| Reference matched feedback | 0.06650 → 0.06556 | 0.36199 → 0.36394 |

Raw reference feedback produced a small reference-scored membership signal:
all three changes were positive [+0.02867, +0.01607, +0.00349], mean +0.01608.
Its conditional case-bootstrap 95% interval [−0.00344, +0.03469] still spans
zero. Matching the initial expected update budget left a mixed change of
+0.00195. Neither reference condition improved mean progress with the neural
PLAY policy; the earlier positive neural cross-policy screen did not reproduce.
All fresh conditional intervals span zero. These data do not qualify longer
full-game training or a reward/learning-rate change.

### Decision

Simple card-output preferences can learn, while native membership gains remain
preliminary and do not transfer to stronger performance with the fly's own
PLAY policy in these tests. The concrete new priority is **reward-free motor
learnability qualification**: require an active plastic KC pathway in relevant
contexts rather than accepting output firing alone. The existing active
DISCARD option that never received eligible input should be addressed with a
new, separately validated mapping and fresh learners. Keep original checkpoints
and primary defaults unchanged. Reward/punishment tuning and longer training
cannot repair a zero-eligibility update in the tested states.

The subsequent repair is now implemented and validated:
[active-plastic-input motor mapping](LEARNABLE_MOTOR_MAPPING.md). The new
reward-free mapping replaces the stuck DISCARD output and one intermittent
shared target output. At 1,500 matched diagnostic choices, the repaired target
improves 52.97% → 90.08% with real updates; the old target remains flat with
zero updates. Original checkpoints stay unchanged. Native membership learning
on the repaired mapping is the next controlled comparison.

All 72 targeted tests for this original study passed, including the two new activity-audit checks.
The full native screen and fresh confirmation independently replayed 27,072
native actions in total. The diagnostic binary task executes no native actions.

### Interruption and recovery

The tool server restarted during the first binary run, after its 1,750-trial
progress report. No experiment process or saved learner state remained. Its
partial data and interruption record are preserved under
`artefacts/card-membership-20261027-v1/`. Version 2 restarts the identical
protocol from unit weights; the repeated partial prefix is not an independent
learner or extra completed exposure. A transient user service runs the suite
independently of the tool server. Completed qualification data and the exact
corpus are reused with unchanged source hashes.
