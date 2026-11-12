# Card-credit direction and clipping audit

Implemented 26 October 2026 after the [pair-learning pilot](PAIR_LEARNING_RESULTS.md)
showed no held-out pair-learning gain. New entry points are independent scripts:

* `scripts/audit_card_credit_direction.py`
* `scripts/probe_fixed_pair_learning.py`

Neither edits `src/flylatro`, the completed pair harness, its condition identities,
or any source checkpoint. Both use isolated efficacy/state copies and verify
the complete source plastic state remains unchanged. They write diagnostic
artifacts, not resumed training checkpoints.

## Direction experiment

Source: subset-exploration plastic learner 0 from
`artefacts/card-learning-20261024-v1/pair-pilot`. Compare its initial and trained
weights on 13 distinct training hands, one per pair rank, with two independent
source neural seeds. Case selection uses corpus rank metadata, not outcomes.
There are 13 hands with repeated measurements, not 52 independent learners.

For each stimulus, inspect both the greedy card choice and a forced correct pair.
The latter is an explicit diagnostic intervention; it never becomes a runtime
teacher or observation feature. Score the submitted cards in the native simulator
and map ordinary progress to a reward pulse. Use the actual choice-gated
three-factor update from zero traces in a temporary state. Check that no
unselected edge changes and that efficacy changes have the requested sign.

Probe a positive native reward and a symmetric negative-pulse sign control.
Evaluate the actual update and 100×/1,000× copies of its efficacy delta, bounded
by the original efficacy limits. Scaled copies are sensitivity probes, not
1,000 successive training decisions or a recommended learning rate.

Compare output preferences using fixed before/after seeds in three contexts:
the source spike seed, a different spike seed, and a shuffled display with
targets mapped to the same physical cards. Report target-vs-competitor normalized
score gap, top-k margin, raw rate gap, membership and actual pair selection.
Repeat/no-update/restoration controls check determinism and backend weight reset.

## Measured results

The real-FlyWire audit completed in 87 seconds. Local output:
`artefacts/card-credit-direction-20261026-v1/report.json`.

* All 52 exact neural repeats, 156 no-update-copy checks and 52 restoration
  checks passed. Source plastic state was unchanged. No probe reached an
  efficacy bound.
* **Every selected MBON edge activity was clipped at the 20 Hz reference** in
  both initial and trained probes, for both action-selection conditions.
* Mean selected-edge KC clipping was 65–68%, depending on weight/action
  condition. A similar fraction of selected coactivity increments was exactly
  one; about 27–31% was zero. This loses graded activity distinctions but does
  not establish that all card information is lost.
* Of 104 positive actual-update comparisons in the original stimulus/seed,
  103 had unchanged target score gap and one improved. Under a held neural seed,
  all 104 were unchanged. The correct-pair maximum edge changes were around
  0.0000164; greedy-choice changes were around 0.0000047–0.0000051.
* Amplified updates had mixed directions. For example, with trained weights and
  the greedy target, 1,000× positive probes worsened the same-seed score gap on
  20/26 measurements and improved it on 2/26. Correct-pair targets behaved
  differently, and held-seed effects also differed. This is not a clean global
  sign inversion or evidence that simply reversing the rule is correct.

The results identify poor one-update observability and substantial clipping.
They do not establish clipping as the sole cause of the failed pair pilot.
Changing references also changes update size, so any follow-up scaling study
must control that confound. Reward contrast, exploration/discovery and circuit
feedback remain alternative explanations.

## Fixed-hand probe

The second script isolates one training hand and one seeded display order.
Two temporary lanes start at identical unit weights and use matched spike seeds
and sampler IDs; one applies reward updates and the other remains frozen.
The original 5% card sampler, native progress feedback and 20 Hz references
remain unchanged. Pair labels are diagnostics only. Traces reset each decision;
the curriculum emits no run-failure pulse.

Use 5,000 decisions per lane and 128 paired frozen evaluation seeds before/after.
The report logs discovered pair plays, block accuracy and efficacy drift; a low
discovery count limits interpretation of failure. Memorizing this one training
hand would not establish generalization or normal-game competence.

The real probe completed in 257 seconds on training seed 6000 (pair rank six):

| Measurement | Plastic copy | Frozen copy |
| --- | ---: | ---: |
| Training pair plays / 5,000 decisions | 51 | 50 |
| Initial frozen evaluation / 128 seeds | 3 | 3 |
| Final frozen evaluation / the same 128 seeds | 2 | 3 |
| Mean absolute efficacy change | 0.001379 | 0 |
| Maximum efficacy change | 0.015943 | 0 |

The frozen lane's individual evaluation choices stayed exact, and the original
source plastic state remained unchanged. The budget did not demonstrate
memorization. This is one hand with few successful choices and a small frozen
evaluation, not a statistically established performance decrease or a proof
that the circuit cannot learn. Data: `artefacts/fixed-pair-learning-20261026-v1/report.json`.

## Next controlled experiments

1. Choose new KC/MBON reference rates from frozen reward-free responses and
   measure how much graded coactivity they retain. Compare original and new
   references with average update magnitude matched, so reduced clipping is
   not confused with a smaller effective learning rate. Keep mappings and
   feedback fixed. The [matched reference-scaling comparison](CARD_REFERENCE_SCALING.md)
   has completed: clipping decreased, but pair counts were 1/128 rescaled,
   2/128 baseline and 3/128 frozen. These audits do not establish a replacement
   rate for long training.
2. Separately assess outcome contrast and exploration on the fixed-hand task.
   Every incorrect scoring play currently receives positive progress feedback.
   The [native-score reward-contrast probe](CARD_REWARD_CONTRAST.md) now tests
   fixed scalar centering with original references/exploration. It completed
   with native/centered/frozen final counts 2/3/3 of 128 against 3 initially,
   so no gain over initial/frozen policy was demonstrated. It uses no pair
   teacher and leaves the primary reward contract unchanged.
3. Treat a global sign reversal as unqualified: the direction probes have mixed
   effects rather than a consistently inverted response. Keep independent
   noise/permutation checks in subsequent sensitivity tests.

The direction and fixed-hand diagnostics are implemented and complete. The
reference and reward-contrast comparisons operate on separate disposable copies.
Neither changes existing runs.

## Reproduction

Both commands require the same compiler environment as the existing native run:

```bash
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/audit_card_credit_direction.py \
  --condition artefacts/card-learning-20261024-v1/pair-pilot/conditions/subset-plastic/condition.json \
  --checkpoint artefacts/card-learning-20261024-v1/pair-pilot/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/card-credit-direction-repeat \
  --cases 13 --repeats 2 --heavy

.venv/bin/python scripts/probe_fixed_pair_learning.py \
  --condition artefacts/card-learning-20261024-v1/pair-pilot/conditions/subset-plastic/condition.json \
  --checkpoint artefacts/card-learning-20261024-v1/pair-pilot/subset-plastic/plastic-checkpoint-final.pkl \
  --output-dir artefacts/fixed-pair-learning-repeat \
  --steps 5000 --evaluation-repeats 128 --heavy
```

Output directories must be new. Reports contain script, checkpoint, condition
and corpus provenance; the older harness's strict identity checks remain active.
Tests cover update signs, selected-edge gating, exact controls, physical-card
mapping under permutations, source/environment restoration and frozen-lane
invariance, including Torch CPU state. The new and related regression suites
passed 63 tests; both native GPU diagnostics and smaller qualification runs
completed successfully.
