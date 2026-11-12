# Executed-choice eligibility pilot

The experimental rule `three-factor-choice-gated-v1` tests whether outcome
reinforcement becomes more useful when new eligibility follows the selected
motor choices. The reference rule remains `three-factor-global-v1`. This is an
engineering action-feedback hypothesis, not an evidence-backed DAN compartment
assignment. It adds no external trainable actor, decoder, critic or optimizer.

## Exact rule

Let `c` be the existing clipped KC/MBON coactivity and `g` a binary mask of
plastic edges whose MBON target belongs to a selected motor pool. The variant
records `eligibility = clip(decay * eligibility + c * g, 0, max_eligibility)`.
The signed reinforcement update, reference rates, efficacy bounds, learning
rate and terminal ordering are unchanged. Earlier selected traces decay normally;
they are not erased when another action is chosen. Terminal outcomes update
weights before clearing that learner's traces.

The selected pools include the chosen action type, card count, selected card
slots and active contextual target, when that head has a genuine competing
choice. Forced one-option heads add no new credit. Selecting every available
card does not create a card-membership choice, although card count can still
be a choice. Inactive heads never contribute. Shared pools/neurons are unioned,
so a connection is not counted twice. A selected but inactive biological edge
still has zero new coactivity; the rule does not invent spikes.

Credit is calculated from actions actually executed. Matched experience
schedules therefore credit their scheduled actions, even if the neural decoder
proposed something else. The rule currently requires MBON-direct output and a
motor pool with anatomical KC→MBON inputs. Whole-brain motor output is rejected
instead of silently applying a different anatomical interpretation.

On CUDA, compact post-selection masks cross from the CPU; device index arrays,
post masks and full edge masks are reused. Eligibility and efficacy remain on
the simulation device. No poker-hand oracle or scoring diagnostic enters the
rule, the masks, or the motor decoder.

## Identity and compatibility

The rule version is part of `plasticity_rule_sha256`. The component manifest
also records `credit_assignment`, including the fixed mapping/assignment hash,
forced-choice policy and engineering assumption. Checkpoint loading rejects a
different plasticity rule even for frozen analysis; start a fresh experiment.
Existing global-rule configurations and their identities stay unchanged.
Frozen evaluation disables eligibility recording and leaves learned state
unchanged in both variants. A distinct experimental comparator checks that
the only model difference is the predeclared credit rule; the generic matched
evaluation comparator continues to require identical learning rules.

## Matched pilot

Run from the repository root with the qualified toolchain:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.learning.choice_credit_pilot \
  --output-dir artefacts/choice-credit-new \
  --steps-per-learner 10000 --replicates 3 \
  --evaluation-episodes 256 --evaluation-batch-size 4 --heavy
```

The runner writes `declaration.json` and both exact configurations before any
training. Each arm has three independent learners and a 30,000 aggregate
decision budget, equal to 10,000 per learner. Learners share immutable anatomy
and GPU execution but do not share experience, weights, traces or RNG streams.
The arms have matching training seeds, base fly seeds and motor exploration.
They run serially on the GPU. The sole model change is the plasticity version.

Both retain failure pulse **0.01**, learning rate **0.002**, temperature **1.0**,
epsilon **0.05**, disabled blind skips, original mappings/normalization, complete
anatomy, 100 ms/dt 0.1 ms and all safety checks. Checkpoints are written every
1,000 decisions per learner. The runner first performs a real native full-state
resume audit. A safety stop is preserved as an experimental outcome and prevents
the scale-up signal from passing.

After training, each learner is independently evaluated with frozen weights on
the same 256 validation seeds, using four replicas of that learner and an Ante-1
target. Replication shares no learned experience. Final-test seeds remain
reserved. There are **three independent training replicates per arm**, not 768
independent training replicates. Complete action/state/reward transitions and
checkpoint identities accompany each evaluation. Every training checkpoint is
audited against durable logs.

The predeclared descriptive scale-up signal requires:

- Every training budget completed without a safety stop.
- More blind clears in a majority of matched training replicates and in total.
- Total wins do not decrease.

This is a screening signal, not a confirmatory statistical learning claim.
Comparability passing does not imply improvement. The assessment records both
arms' blind clears, wins and paired return differences by learner. Source/config
hash checks reject a changed implementation or configuration during the run.

For inspection without launching GPU work, add `--prepare-only`. Use a fresh
output directory. The standalone single-fly candidate config is
`configs/exploratory-event-choice-credit-v1.toml`; running it alone is not the
matched three-seed comparison.

## Verification and results

Tests cover selected versus unselected pools, forced/inactive heads, each
contextual target, card membership/count distinctions, overlapping pool unions,
NumPy/Torch update equality, malformed-mask rejection before mutation, delayed
traces, terminal update-before-clear and learner isolation, executed schedule
credit, unchanged frozen state, strict cross-rule rejection and exact resume.
The full CPU regression passes **558 tests**, with ten CUDA tests skipped.
A real native three-learner smoke run passes 32 decisions per learner; the
pilot's native resume qualification also passes.

The first pilot completed in `artefacts/choice-credit-20261016-pilot-v1`.
Both arms completed 10,000 decisions per learner for three matched learners,
without a safety stop. All six frozen evaluations, the native resume
qualification and checkpoint/log audit passed. `suite.json` records exact
commands and job status; `declaration.json` records the fixed design and
`assessment.json` records the comparison.

| Training replicate | Baseline blind clears / wins | Choice-gated blind clears / wins | Paired mean return change |
| --- | --- | --- | --- |
| 0 | 1 / 0 | 0 / 0 | -0.007203 |
| 1 | 0 / 0 | 1 / 0 | -0.007676 |
| 2 | 0 / 0 | 0 / 0 | +0.007370 |
| Total | 1 / 0 | 1 / 0 | |

Each replicate used the same 256 validation episodes in both arms, for 768
evaluations per arm. The choice rule improved blind clears in only one of three
matched replicates and tied the total. The predeclared scale-up signal is
**false**. This pilot demonstrates no clear performance gain; it does not
establish that the rule is worse or could never improve with more experience.

The recommended next step is a controlled card-selection diagnostic before
another long training run. On identical simulator snapshots and neural/RNG
states, compare initial and trained card choices with the immediate scores of
legal plays, and test whether moving cards between slots moves the preference
with the cards. Scoring stays outside the agent and reward implementation.
This can distinguish sensitivity to card content from slot preference and
check whether weight changes improve hand selection. The earlier small
diagnostic in `artefacts/no-skip-20261016-evaluation-v2/hand-diagnostic.json`
suggests substantial missed immediate reward, but uses a curated corpus and
does not establish a long-term optimal policy. Keep failure pulse 0.01 and blind
skips disabled while diagnosing; this pilot provides no evidence for reducing
punishment further.

The controlled diagnostic is now implemented and has been run on both arms and
all six learners. See [the card-selection diagnostic](CARD_SELECTION_DIAGNOSTIC.md)
for its command, controls and completed results. It finds a small conditional
card-quality improvement in all three choice learners, alongside weak exact
card tracking under permutation. This does not change the failed held-out
scale-up result above.
