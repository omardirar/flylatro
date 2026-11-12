# Native discard learning after the active-input repair

This study tests whether the repaired motor mapping helps learn useful native
DISCARD membership. The preceding binary preference test established that the
formerly stuck output can update and learn a rewarded choice; it did not
establish a better poker strategy. See
[mapping qualification](LEARNABLE_MOTOR_MAPPING.md).

Runner: `scripts/run_repaired_native_membership.py`.
Independent full-training audit: `scripts/assess_repaired_native_membership.py`.
Artifact directory: `artefacts/repaired-native-membership-20261028-v1/`.

## Fixed comparison

The declared design retains the old native study's 2,000 DISCARD→PLAY trials
per learner, corpus, training/neural seeds, rewards, learning rule, 5% card-subset
exploration and action-conditioned motor architecture. DISCARD count is fixed
at two, so improvements cannot come from discarding more cards. Only DISCARD
membership efficacy changes. PLAY always uses initial weights.

Three independent training seeds each have four actors:

| Actor | Training feedback | Evaluation |
| --- | --- | --- |
| Neural native | Native outcome after the fly's frozen PLAY | Both neural and reference PLAY |
| Reference native | Native outcome after immediate-score reference PLAY | Both policies |
| Reference matched | Reference feedback attenuated by a training-only initial expected update match | Both policies |
| Frozen | Zero applied pulse; chooses its own actions | Shared initial control for all three treatments |

The new mapping starts from unit weights and never loads an old checkpoint.
The old completed 2,000-trial study is preserved and reused; it is not counted
as another replicate. The comparison reports each mapping's learned change
against its own initial/frozen policy, since the mapping changes initial choices.
Initial expected update matching is not a guarantee of identical realized
update trajectories; both are reported.

The training screen retains 52 validation hands/two repeats under greedy and
training-sampled membership. A separate **104-hand fresh set**, balanced over
pair ranks, uses new validation offset 420000. Its seeds are checked against
both previous screen and previous fresh-confirmation cases. Both old and new
trained actors are evaluated greedily on this same set, with no updates or
outcome-based actor selection. The reference player is diagnostic and does
not become a learned or deployable policy.

Native replay verifies calibration, screen and fresh evaluation choices. The
additional independent audit replays every recorded training transition,
checks per-lane pulse assignment and update totals, aligns saved efficacies
to anatomical post roots, and verifies that the repaired output actually
received native updates. It also describes neural PLAY's immediate-score
choices from fresh logs. These descriptions are not training labels or claims
of optimal long-term play.

## When longer training becomes worthwhile

Readiness is declared before examining new results. For the new mapping's
**neural-native** actor evaluated with **neural PLAY** on fresh hands, require:

* Mean blind-progress improvement of at least 0.005 against its own initial
  and frozen policy.
* Positive changes in at least two of the three training seeds.
* A positive lower endpoint of the conditional case-bootstrap 95% interval.
* Exact frozen/source/unrelated-weight controls, legal actions, and healthy
  eligibility and efficacy bounds.

The case interval is conditional on these three learners and one mapping. It
does not treat repeated neural samples as independent training replicates.

Passing advances to **one bounded free-action Ante-1 pilot**, with initial and
frozen controls, checkpoints and fresh game roots. Long training becomes worth
scaling when that pilot shows a repeatable learning curve in actual games:
better blind progress or clears, useful action timing and stable plasticity.
Frequent wins are not a prerequisite. A reference-only gain or artificial
preference-learning gain does not establish that the current neural PLAY and
native feedback form a useful long-run learning loop.

Failing this native test means the current setup has not earned scale-up.
Use the measured bottleneck to choose the next change rather than assume
that more exposure or weaker punishment will fix it. This is a finite
decision path: native check, bounded full-game pilot if justified, then scale
or change the setup based on measured progress.

## Reproduction

Run from the repository root with a new output directory. The prepared
declaration binds all inputs and the runner's source before learning.

```bash
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/run_repaired_native_membership.py \
  --stage prepare --output-dir artefacts/repaired-native-reproduction
.venv/bin/python -u scripts/run_repaired_native_membership.py \
  --stage train --output-dir artefacts/repaired-native-reproduction --heavy
.venv/bin/python scripts/assess_card_membership.py \
  --run-dir artefacts/repaired-native-reproduction/native
.venv/bin/python -u scripts/run_repaired_native_membership.py \
  --stage confirm-new --output-dir artefacts/repaired-native-reproduction --heavy
.venv/bin/python -u scripts/run_repaired_native_membership.py \
  --stage confirm-old --output-dir artefacts/repaired-native-reproduction --heavy
.venv/bin/python scripts/run_repaired_native_membership.py \
  --stage report --output-dir artefacts/repaired-native-reproduction
.venv/bin/python scripts/assess_repaired_native_membership.py \
  --run-dir artefacts/repaired-native-reproduction
```

The current run uses a transient user service so it survives tool-server
interruptions. `status.json` records trial progress and `suite-status.json`
records the active stage. Original source, mappings and checkpoints remain
unchanged.

## Completed results, October 28

The run completed all 2,000 native trials per actor, three training seeds and
all feedback/frozen conditions. Training took 1,539 seconds. The mapping's
previously blocked output received native updates in every plastic actor:
896 of its 942 plastic KC afferents changed in all nine, versus zero changes
in frozen copies. None of the 18,000 positively reinforced plastic-actor trials
had zero total eligibility. Source, frozen and unrelated weights, fast traces,
efficacy bounds, declared pulse assignment and native reinforcement checks pass.

Initial calibration's reference multiplier was 0.0893301. Actual cumulative
update L1 ranges were 24.24–24.84 neural-native, 313.16–328.52 reference-native
and 26.11–28.03 reference-matched. The attenuated reference budget is close,
but not identical, to the native trajectory's realized budget.

The screen's reference-native actors improved neural-PLAY progress
0.06899 → 0.07806, positive in all three seeds. **This did not reproduce on
fresh hands.** Fresh greedy results follow. These are average fractions of
the first blind's score target, not win rates or card-choice accuracy.

| Mapping / feedback | Neural PLAY: initial → trained | Reference PLAY: initial → trained |
| --- | --- | --- |
| Old / neural native | 0.06938 → 0.07057 | 0.40611 → 0.40454 |
| Old / reference native | 0.06938 → 0.06762 | 0.40611 → 0.40732 |
| Old / reference matched | 0.06938 → 0.06874 | 0.40611 → 0.40477 |
| **Repaired / neural native** | **0.07327 → 0.07047** | **0.42505 → 0.41544** |
| Repaired / reference native | 0.07327 → 0.06628 | 0.42505 → 0.41856 |
| Repaired / reference matched | 0.07327 → 0.06850 | 0.42505 → 0.41201 |

All nine repaired-map actors have negative observed neural-PLAY changes on
fresh hands. For the predeclared primary neural-native condition, changes are
−0.00282, −0.00415 and −0.00144; mean −0.00280, conditional case-bootstrap
95% interval [−0.00813, +0.00203]. It fails all three advance criteria. This is
no demonstrated gain, rather than proof that this architecture can never learn.
The repaired mapping starts from a stronger initial policy on this set, but
that initial difference is not training improvement. The learned changes are
weaker than the old mapping's changes in all six descriptive policy/feedback
comparisons. Do not select a mapping or actor using these fresh outcomes.

Independent audit replayed **106,560 native actions**: 72,000 training,
19,584 calibration/screen and 7,488 fresh actions for each mapping. All **91
targeted tests** pass. Both original source-checkpoint hashes and their resume
identities remain unchanged. Detailed reports, final states, training logs,
anatomical weight audit, fresh decisions and hashes are saved under the
artifact directory; `report.json` records the readiness decision and
`assessment.json` records the independent audit.

### What the decisions show

On the repaired neural-native actors' 624 fresh decisions:

* Neural PLAY chose a high-card hand despite an available pair-or-better hand
  in **457/624 (73.24%)** decisions.
* It missed an immediately available blind clear in **50/624 (8.01%)**.
* It achieved mean progress 0.07047 where immediate-score reference PLAY from
  the same post-discard hands achieved 0.41544.

These describe the first immediate PLAY on plain opening hands, not optimal
long-term actions. PLAY is frozen in this experiment, so they do not measure
training a new PLAY policy. The lack of reference-scored improvement also
means the result cannot be explained solely by weak PLAY execution.

Every native training trial supplied positive feedback; mean neural pulse was
0.003430 versus 0.042601 reference. Neural/reference pulse correlations across
the same post-discard hands were only 0.044–0.123 across lanes. Positive reward
availability therefore was not the missing mechanism, while feedback quality
and magnitude remain distinct questions. Stronger reference feedback did not
produce a fresh improvement here.

A post-hoc saved-weight check at the repaired output finds that about 96% of
changed afferents have exactly the same final change within each plastic actor.
This is consistent with mostly uniform gain changes at that output; it does
not prove that all card information is absent or identify clipping as the sole
cause. Previous identity decoding and reference-scaling/centering results must
be retained when interpreting it.

### Decision and next step

**Do not scale this setup unchanged based on this study.** The mapping repair
fixes a real zero-update defect, and remains qualified for subsequent fresh
experiments. It does not demonstrate learning which cards are useful to discard.
Keep its original calibration/qualification and historical mappings intact;
do not retune the replacement against this validation set.

The next focused diagnostic should change the rewarded slot with the visible
hand: a balanced, two-option conditional card-choice task with fresh cases,
own-initial/frozen controls, and a labelled diagnostic teacher separated from
native training. Examine the information in the actual clipped, choice-gated
eligibility patterns as well as rates. This separates learning a fixed output
preference from learning a useful card-dependent choice. Existing broad pair,
longer-window, centered-feedback and reference-scaling null results are not
erased by the repair, and should not be repeated as presumed fixes.

Long training is technically runnable now, but this study does not justify
expecting better strategy from duration alone. The practical scale-up signal
remains repeatable held-out native progress, then a bounded free-action pilot
showing progress/clears with stable controls. Frequent wins are not required.
