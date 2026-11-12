# Local normalisation: native fixed-count results

Date: 31 October 2026. Main study:
`artefacts/local-normalisation-20261031-v1/`.
Implementation and declared methods: [local normalisation](LOCAL_NORMALISATION.md).

## Completed experiment

The real-GPU comparison completed its predeclared 20,000 trials per learner.
There are three independent learners in each of the original-rule, normalised
and frozen conditions: nine actors and 180,000 training actions in total.
Every native action plays exactly two cards. Only PLAY membership can learn;
reward parameters and exploration are unchanged. This is a capacity experiment,
not a full-game training result. The experiment took 3,748 seconds (62.5 minutes),
with no safety stop. The transient service exited successfully and is inactive.

**Conclusion:** normalisation controls cumulative incoming-strength growth as
intended. Useful card-selection improvement has not been established. The fresh
result has a small positive mean, but the advantage is inconsistent across
learners and its conditional uncertainty interval includes zero.

## Intermediate learning curve

Rates below average three learners on 104 hands with two display/neural repeats
per hand. Repeats and shared hands are not independent training replicates.
Uniform two-card chance is 1/28, or 3.57%.

| Step | Set | Original rule | Normalised rule | Frozen |
| --- | --- | ---: | ---: | ---: |
| Initial | Training hands | 4.97% | 4.97% | 4.97% |
| Initial | Held-out hands | 4.01% | 4.01% | 4.01% |
| 5k | Training hands | 4.17% | 4.49% | 4.97% |
| 5k | Held-out hands | 4.01% | 4.17% | 4.01% |
| 20k | Training hands | 4.17% | 4.17% | 4.97% |
| 20k | Held-out hands | 2.88% | 3.53% | 4.01% |

At 5k the normalised held-out result is only 0.16 percentage points above its
own initial value and the original-rule control. At 20k both plastic conditions
are below their initial values on this held-out set. No checkpoint was selected
from these intermediate outcomes; the declared endpoint was used for the fresh
confirmation.

## Fresh confirmation

All actors were evaluated with learning disabled on 104 separately declared new
hands, with two matched display/neural repeats per hand. Each condition therefore
has 624 choices across three learners, but only 104 distinct game cases. Initial
and trained evaluations use the same cases, displays and neural seeds.

| Condition | Initial correct pairs | Final correct pairs | Initial accuracy | Final accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original rule | 22/624 | 25/624 | 3.53% | 4.01% |
| Normalised rule | 22/624 | 28/624 | 3.53% | 4.49% |
| Frozen | 22/624 | 22/624 | 3.53% | 3.53% |

The normalised rule gained 0.96 percentage points versus initial and 0.48 points
versus the original rule. Per-learner initial counts were [5, 6, 11]; final
original counts [7, 9, 9]; normalised counts [10, 9, 9], each out of 208.
Two normalised learners improved versus initial, but only one exceeded the
original-rule counterpart; the third declined versus its own initial policy.

Conditional case-bootstrap 95% intervals were:

- Pair gain versus initial: **−0.64 to +2.72 percentage points**.
- Pair advantage versus original rule: **−1.28 to +2.24 points**.
- Native progress gain versus initial: **−0.000978 to +0.002772**.
- Native progress advantage versus original: **−0.002046 to +0.002169**.

The bootstrap averages the two repeats before resampling the 104 shared cases.
Intervals are conditional on these three trained learners; they do not measure
population-level variation in training. All intervals include zero.

Mean native progress rose from 0.050614 to 0.051490 under normalisation and to
0.051437 under the original rule. The normalised versus original difference is
about 0.0000534 of the blind target, or 0.016 chips per opening play. This is not
a demonstrated useful score advantage. The predeclared two-percentage-point
pair gain/advantage criterion and cross-learner consistency criterion failed.

## Strength budgets and independent audit

Every normalised learner preserved every postsynaptic strength budget. Maximum
relative budget errors were [5.02e-8, 3.58e-8, 3.46e-8], independently recomputed
from anatomical synapse counts and saved efficacies. Original-rule maximum
neuronal total-strength increases were [3.62%, 3.62%, 3.63%]. No actor had an
efficacy at its lower or upper bound at the endpoint. The test therefore shows
budget conservation, not avoidance of an observed saturation failure in this
particular short task; the strong uniform-gain stress separately tests that case.

Mean absolute efficacy drift was about 0.00205 under normalisation versus
0.00312 under the original rule. Cumulative net-update L1 was approximately
[140.26, 139.58, 140.12] versus [210.55, 211.03, 211.17]. Thus nominal learning
rate/reward settings match, while realised update magnitudes differ; they were
not retuned to match after observing outcomes.

The independent CPU assessor verified **194,976 recorded actions**: 180,000
training and 14,976 evaluation records, against **8,463 unique native
snapshot/action replays**. Identical combinations are cached after one native
check. It verified native rewards/pulses, pair diagnostics, training/display
schedules, evaluation coverage, weighted budgets, accumulated updates, exact
frozen choices and unrelated weights. Evidence: `assessment.json`, bound to the
final report, checkpoint and assessor hashes. Final-test seeds were unused.

## Completed implementation checks

- Weighted incoming-strength budgets are conserved separately per postsynaptic
  neuron and learner, with efficacy bounds respected.
- Original-treatment lanes match the unchanged three-factor rule exactly over
  150 signed updates, including clipping and disabled learning.
- NumPy/Torch CPU/CUDA paths, bounded projection, neuron independence, controls,
  event accounting and cross-rule checkpoint rejection pass.
- A transparent toy circuit learns two context-dependent choices with constant
  neuronal budgets; its frozen control remains at chance. This is a rule
  capacity check, not evidence about real-fly card recognition.
- Exact native task resume recovers a torn log tail. Real-GPU checkpoint restore
  also reproduced three subsequent trials and every local state array exactly.
- The real-GPU smoke assessment independently checks 258 recorded actions
  against 47 unique native snapshot/action outcomes, including diagnostic pair
  labels and unchanged rewards. It verifies the anatomy-weighted budget directly.
- Thirteen new checks and 64 related historical regressions have passed.
- Historical long-run and PLAY/DISCARD condition identities remain valid.

The rule remains experimental. These results do not qualify it as a gameplay
improvement or establish that longer training will help. They also do not prove
that later learning is impossible. The implementation, controls, learning
curves and checkpoints are retained for future comparisons; the original model
and long-run checkpoints are preserved. No new full-game or million-decision
run was launched.
