# First learning pilot with separate PLAY/DISCARD outputs

Run: `runs/play-discard-learning-pilot-20261027-v1`. All four conditions
completed 5,000 decisions per learner, three learners per condition, 60,000
total training decisions. No condition safety-stopped. New mapping:
[separate PLAY/DISCARD interface](PLAY_DISCARD_INTERFACE.md).

## Held-out result

Each learner used the same 128 validation roots, with greedy frozen evaluation
of trained and initial weights. These are three training replicates per arm
and 128 shared game roots, not 384 independent training replicates.

| Condition | Initial blind clears | Trained blind clears | Initial Ante-1 wins | Trained Ante-1 wins |
| --- | ---: | ---: | ---: | ---: |
| Top-k, frozen | 3 | 3 | 0 | 0 |
| Top-k, plastic | 3 | 2 | 0 | 0 |
| Subset exploration, frozen | 3 | 3 | 0 | 0 |
| Subset exploration, plastic | 3 | 5 | 0 | 1 |

Blind counts include every blind beaten, so completing Ante 1 contributes
three clears. Subset-plastic learner clears were [1, 0, 4], with wins [0, 0, 1].
Top-k-plastic clears were [1, 0, 1]. Initial policies all cleared one Small
Blind each. The six plastic learners all had positive descriptive mean-return
deltas against their initial policies. Arm averages were +0.060587 for subset
and +0.009515 for top-k; the subset average is strongly affected by its one win.
Comparability PASS in `results.json` checks matched evaluation identity, not
learning success or statistical significance.

## Verified winning episode

Subset-plastic learner 2 won on validation root **20070062**, native Balatro
seed **PYYEE9EM**. The recorded episode has 26 decisions, clears all three
Ante-1 blinds, and ends at Ante 2 with `won=true`. Its initial policy failed
on that same root. Native replay verified the states, rewards and termination.
The episode used eight PLAY actions, shop/pack actions and **zero DISCARDs**.
It therefore does not establish learned discard quality.

The preserved trace is `winning-episode.jsonl` under the run directory.
Machine-readable assessment: `assessment-v1.json`. This is one observed win
after selecting among several learners/conditions; it needs fresh validation
before extrapolation or further long training.

## Health and decision audit

All four final checkpoint SHA-256 hashes matched their manifests. Every learner
recorded exactly 5,000 decisions. Plastic state was finite, no efficacy bound
was hit, and frozen efficacies remained exactly equal to initial values.
Card alias admission was zero in the final training metrics. Maximum learner
mean absolute drift was 0.001743; maximum individual edge drift was 0.017613.

The read-only native audit covered all six trained plastic evaluations and one
equivalent frozen baseline: **4,688 transitions**, including the winning
episode. All before/after state hashes, rewards and completion checks passed.
It also verified 7,630 alternative plays in the native simulator across its
opening-state spot checks. Outputs: `decision-audit-v1/`.

The first-Small-Blind decision audit still finds substantial weaknesses:

* All **763/763** audited first-blind failures across the six plastic learners
  left discards unused. This means at least one discard remained, not that no
  discard was used anywhere in those episodes.
* Subset-plastic learners made 64 first-blind discards against 1,536 plays;
  top-k-plastic learners made 53 discards against 1,536 plays.
* Subset learners missed available immediate clears on 236 plays, and top-k
  learners on 223. These are turns, not independent games.
* There were 1,107 subset and 1,117 top-k High Card plays with a pair present.
  Merely having a pair does not by itself establish an error in every context.

Separate preferences/credit work mechanically, but useful discard behavior is
not yet demonstrated. The winning learner is an encouraging candidate, not a
qualified general policy. This pilot also lacks a width-matched shared-head
control, so it cannot isolate the architectural fix from the pool-width and
reward-free mapping changes.

## Prepared next step: fresh frozen validation

Preserve the current weights and test **all six plastic learners**, not just
the winning learner, on 512 fresh validation roots: 20120000–20120511. No
overlap was found in 119 existing evaluation summary files when this plan was
prepared. Locked final-test roots remain unused.

Reuse one equivalent initial-unit-weight control: all pilot initial evaluation
transition logs were byte-identical across conditions/learners, initial
weights are identical, mappings/dynamics match, and evaluation is greedy.
This reuse is explicit in the plan; it does not create independent controls.
The plan runs seven frozen policies, 3,584 episodes total, on 512 shared roots.
It performs no further training and evaluates both top-k and subset arms.

Run from the repository root:

```bash
bash runs/play-discard-learning-pilot-20261027-v1/confirmation-plan-v1/run.sh
```

The script verifies condition/checkpoint identities before launching, uses the
same compiler environment as training, and writes comparisons to
`runs/play-discard-learning-pilot-20261027-v1/fresh-validation-v1/comparisons.json`.
Its design is saved in `confirmation-plan-v1/plan.json`. The user subsequently
executed it; completion is recorded in `confirmation-plan-v1/execution-status.json`.

## Completed fresh validation

All seven frozen evaluations completed on the declared 512 fresh roots.

| Policy | Episodes | Blind clears | Ante-1 wins |
| --- | ---: | ---: | ---: |
| Equivalent initial-unit-weight control | 512 | 0 | 0 |
| Subset-plastic, three learners | 1,536 | 4 | 0 |
| Top-k-plastic, three learners | 1,536 | 6 | 0 |

The six policies share the same 512 roots; there are not 3,072 distinct game
roots or training replicates. Subset learner clears were [1, 0, 3], top-k clears
[2, 3, 1]. Mean return changes versus initial were +0.002448 subset and
+0.009452 top-k. Five learners improved this descriptive return measure and
one declined. The earlier single Ante-1 win did not reproduce on the fresh
block. Small partial-progress gains do not yet establish reliable competence
or a benefit of subset exploration.

The new native replay audit verified every transition in all seven evaluations.
Among trained learners, 3,061/3,063 first-blind failures left discards unused.
Outputs and hashes are in `fresh-validation-v1/decision-audit-v1/` and
`fresh-validation-v1/assessment-v1.json`. Existing checkpoints are unchanged.

### Discard diagnosis and next experiments

The fly did discard during training: 3,516 of 12,631 PLAY/DISCARD decisions in
subset-plastic (27.84%), and 3,504 of 12,604 in top-k-plastic (27.80%). Under
greedy fresh evaluation, individual trained policies discarded on only
3.49–4.23% of PLAY/DISCARD decisions; the initial policy's rate was 4.34%.
Discard legality was also verified at all eight PLAY choices in the original
winning episode, with discards available throughout.

Training's 5% epsilon is not 95% greedy: outside uniform epsilon exploration,
`FixedMotorInterface._choose` samples a softmax over scores. Frozen evaluation
instead takes their argmax. This is a measured policy-selection difference,
not proof that stochastic evaluation will improve wins. The completed frozen
comparison evaluated both policies with the same initial controls and recorded
action-score margins, preserving the established greedy result.

Recorded immediate synthetic feedback on every training DISCARD was zero.
PLAY received progress/clear feedback and terminal failure pulses. Discards
can still receive delayed credit through eligibility (decay 0.9 per decision);
zero immediate reward does not imply zero eventual updates. Whether that
credit improves DISCARD preferences has not been demonstrated.

Both follow-ups are now implemented and completed. The frozen comparison ran
2,048 episodes: sampled discarding rose to ~28–29%, but initial policies did
so too, and neither arm's average sampled trained return exceeded its initial
control. No Ante-1 wins occurred.

The native discard→fixed-PLAY task ran 5,000 trials for each of three plastic
learners and paired frozen controls, retaining eligibility across both actions.
Only DISCARD card/count weights changed. Greedy held-out progress declined in
all three learners; sampled changes were mixed and nearly flat. Same-count
random controls showed no consistent membership advantage. Fixed follow-up
PLAY captured only ~30% of available immediate hand progress. Restoring free
actions on 64 fresh roots produced small mixed return gains and zero wins;
greedy discarding remained rare. This tests free-action transfer, not learned
discard timing, because action type was forced during training.

[Full methods, numerical results, scripts and reproduction commands](DISCARD_LEARNING_STUDIES.md).
The follow-up [mechanism, exploration and stronger-PLAY screens](TRAINING_BOTTLENECK_STUDIES.md)
have also completed: simple action preferences learned, fresh 25%-exploration
pair performance matched initial weights, and stronger PLAY mostly changed
discard count. Retain the interface fix and next isolate actual card outputs
and fixed-count membership. Punishment, trace and sampling changes remain
separate factors; these studies do not qualify reducing punishment, increasing
the learning rate or extending full-game training.
