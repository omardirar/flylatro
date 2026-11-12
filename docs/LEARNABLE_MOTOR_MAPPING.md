# Motor calibration with active plastic inputs

Implementation: `scripts/learnable_motor.py` and
`scripts/qualify_learnable_motor.py`. Candidate and validation artifacts:
`artefacts/learnable-motor-20261028-v2/`.

The old calibration accepted an MBON that fired but had no active plastic KC
afferents. Its DISCARD card option 3 consequently received 1,070 rewards with
zero updates in the previous diagnostic. See
[the original diagnosis](CARD_MEMBERSHIP_LEARNING.md).

The new version checks actual fresh-rule coactivity, summed over incoming
plastic edges after applying the source KC/MBON reference rates and eligibility
clipping. Each assigned neuron must have positive fresh eligibility in at least
half the reward-free samples in which its option is legal and competes with
another option. The 50% threshold was fixed before learning validation.
Output firing and an anatomical connection count alone do not qualify it.

Checks use each head's interpretation contexts. Shared shop/pack/joker/
consumable outputs must meet the requirement in every exercised interpretation.
Disabled SKIP_BLIND, padding and forced-only options are explicitly unexercised,
not failed learning targets. Existing coverage gates still apply.

The migration retains the existing width-one architecture and replaces only
failed pools. It ranks unused replacements by reward-free neural response,
filters candidates for active inputs **before** allocation and requires the
existing response/diversity/coverage gates afterward. It never uses game scores,
reward labels, or learning-test results to select a replacement. If no candidate
passes, it raises an error and writes no replacement mapping. Production source
and prior experiment scripts are unchanged so their hash-bound resumes remain
valid.

## Calibration result

The first proposed combination passed every gate. Two pools changed:

| Role | Old root | New root | Coactive fraction before → after |
| --- | --- | --- | --- |
| DISCARD card option 3 | 720575940629529722 | 720575940620715224 | 0% → 100% in its competing calibration samples |
| Shared target option 4 | 720575940614595218 | 720575940628783363 | Shop 29.17% → 100%; pack 72.22% → 100% |

The shared target had intermittent eligibility, rather than the card output's
complete absence of eligibility. The robustness threshold catches both.
Other pool assignments and normalization values are unchanged. Canonical output
roots, routing, separate PLAY/DISCARD populations, credit rule, rewards,
exploration, and qualified pool width are retained. Changed calibration identity
means old trained weights cannot be resumed under the new mapping.

The old shared target neuron also served an action-type pool. Using an unused
replacement removes that particular reuse: unique assigned output neurons rise
from 46 to 47, with 45 exclusive core neurons in both layouts. This is a declared
layout difference; the binary card diagnostic never exercises the target head.

## Validation protocol

The mapping is fixed before running any learning checks:

1. Strictly load the old checkpoint with its original components. Run a
   1,500-choice binary card-output diagnostic using fresh unit-weight local
   states, three learner seeds and counterbalanced targets. Each target has a
   plastic/frozen pair: twelve lanes. Use the original pair corpus, seeds, 5%
   subset exploration, fixed DISCARD/count-one interface and diagnostic +0.05
   target reward. Evaluate on 64 shared neural repeats before and after.
2. Build a separate fresh stack with the new mapping. Verify that loading the
   old checkpoint rejects the changed identity.
3. Collect 192 reward-free states from eight independent environment seeds,
   and add 96 predetermined display permutations of the original calibration
   corpus for inventory-context coverage. The random-navigation states alone
   do not reach shops at this budget. Save the scopes separately; the inventory
   fixtures are perturbations of calibration states, not independent games.
   Recheck active plastic inputs with fresh neural seeds and unit weights.
4. Run the identical binary diagnostic on the new mapping, then verify native
   action legality, opposing PLAY/DISCARD expressivity and immediate credit
   isolation using three pair cases and two neural repeats.
5. Independently inspect saved states, frozen controls, source hashes, gate
   results and the old/new option-3 update. The diagnostic executes no native
   game transitions; native interface checks are separate.

This tests whether the mapping supports learning. It cannot establish better
native DISCARD strategy, stronger neural PLAY, or full-game wins.

## Completed validation

All source identities and both original source-checkpoint hashes remain
unchanged. The new mapping passes the original calibration gates and the new
active-input gate. Separately assessed input coverage passes on 28 competing
roles in independently seeded games and 58 in the inventory display fixtures;
unexercised options are reported separately.

At the matched 1,500-choice budget, exact target-selection probability on 64
evaluation neural repeats changed as follows. Each row is one target/learner,
with a matched frozen copy; these are not six independent mappings.

| Rewarded DISCARD card slot | Old mapping: before → after | New mapping: before → after |
| --- | --- | --- |
| 0 | 67.81% → 85.63% | 67.81% → 85.63% |
| 1 | 32.19% → 33.67% | 32.19% → 33.67% |
| 2 | 67.81% → 85.63% | 47.03% → 61.88% |
| **3, repaired output** | **32.19% → 32.19%** | **52.97% → 90.08%** |
| 4 | 54.45% → 78.20% | 54.45% → 78.20% |
| 5 | 45.55% → 69.30% | 45.55% → 69.30% |
| Counterbalanced mean | 50.00% → 64.10% | 50.00% → 69.79% |

The formerly stuck target has cumulative absolute update L1 **86.1554** under
the new mapping, versus exactly **0** under the old mapping. All six new targets
improve against their own initial policy, although slot 1's gain is small.
Slot 2 shares its binary competition with the repaired slot 3 and has a changed
initial preference; compare each map with its own initial and frozen controls.
Every state field for the other two slot-pair replicates is bit-identical
between maps. Frozen choices and weights remain exact; source and unrelated
weights remain unchanged; efficacy bounds are not hit. The independent assessor
reproduces the input gate from saved activity, verifies saved learner states,
and replays all **18** native interface actions.

All **83 targeted tests** pass. Completed data and hashes are in
`assessment.json`, with separate old/new reports and final states under
`validation/`. The replacement fixes the demonstrated learning blockage in
these contexts. It is ready for a short **native fixed-count DISCARD-learning
comparison with fresh learners**; it does not yet justify another long
full-game run or reward/punishment changes.

The subsequent [native repaired-mapping comparison](REPAIRED_NATIVE_MEMBERSHIP.md)
has now completed. The repaired output receives native updates in all nine
plastic actors, but none of the feedback conditions improves fresh mean
progress. Neural-native progress changes 0.07327 → 0.07047, with all three
learners declining. The qualification fixes output learnability; it does not
establish conditional card strategy or justify scaling unchanged training.

## Reproduction

Use a new output directory; existing candidates and completed validation
directories cannot be overwritten:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/qualify_learnable_motor.py \
  --output-dir artefacts/learnable-motor-reproduction --steps 1500

CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -u scripts/qualify_learnable_motor.py \
  --output-dir artefacts/learnable-motor-reproduction --validate --heavy

.venv/bin/python scripts/assess_learnable_motor.py \
  --run-dir artefacts/learnable-motor-reproduction
```

The generated `config.toml` points to the new `candidate-motor.json`. Start fresh
learners when using it. The CLI deliberately separates reward-free allocation
from validation; failed validation never triggers outcome-guided reallocation.
