# Causal exchange-update probe

Completed 6 November 2026 after the [exchange-credit pilot](EXCHANGE_CREDIT_PILOT.md).
Code: `scripts/probe_exchange_direction.py`.
Evidence: `artefacts/exchange-direction-20261106-v1/report.json`.

## Question and intervention

Does reinforcing a beneficial card exchange increase the entering card's
preference over the card it replaces? The native simulator verifies that
replacing 6♣ with the missing 3♦ in the recurring five-card selection improves
60 chips to 92. This is a labelled diagnostic intervention, not normal training.

Two temporary weight conditions are tested: unit initial efficacies and the
first signed-exchange learner from the completed pilot. The source NPZ is
hash-checked before/after and never modified. For each condition, one update
uses either selected-good-set credit or signed entering/leaving credit, followed
by the existing anatomical incoming-strength normalisation.

The scalar pulse is the **paired native score advantage**, 0.05 × (92−60)/300.
This deliberately supplies a clean directional signal; it is not the pilot's
moving-average baseline. Updates are generated using one fixed neural seed.
Their resulting directions are rescaled on disposable copies to approximately
matched L1 magnitudes 0.1, 1 and 10, with bounds and strength budgets enforced.
Actual magnitudes are recorded; projection/float rounding means they are not
exactly equal. The actual single-update magnitude is also evaluated.

Positive and negative direction controls are measured with eight shared neural
seeds, including the source seed, on the original display and a reversed display.
On the reversed display the measured outputs follow the same physical cards.
Every measurement restores the base efficacy and must exactly reproduce its
original neural output. These amplified directions are sensitivity probes, not
hundreds or thousands of successive learning steps.

## Results

The actual signed update had total absolute efficacy change about **0.005**;
selected-set credit had about **0.0334**. At the actual magnitudes, positive
updates produced no entering-minus-leaving score-gap change in any tested seed,
for either weight condition or display.

For positive signed updates, both weight conditions produced:

| Total absolute weight change | Original display: improved / worse / unchanged | Reversed display: improved / worse / unchanged |
| ---: | ---: | ---: |
| Approximately 0.1 | 0 / 0 / 8 | 0 / 0 / 8 |
| Approximately 1 | 0 / 0 / 8 | 0 / 0 / 8 |
| Approximately 10 | 1 / 2 / 5 | 0 / 1 / 7 |

At magnitude 1, about 200 times the actual signed update, all measured
preferences remained unchanged. At magnitude 10, about 2,000 times the actual
update, responses were mixed; original-display mean change was numerically
zero, and reversed-display mean change was negative. Simply reversing the
update also did not establish a consistent improvement.

Selected-set credit likewise had mixed amplified responses: at magnitude 1,
two of eight original-display responses worsened and none improved; at 10,
two improved and one worsened. The reversed display remained unchanged in
those positive selected-credit conditions.

All **512 exact restoration checks** passed. Candidate efficacies remained
bounded and normalised strength budgets passed the 1e-6 relative tolerance.
The saved pilot checkpoint remained unchanged.

## Interpretation and next priority

The signed pilot's small realised updates are real, but increasing their size
alone is not justified by this probe. Even a clean reward for the beneficial
exchange did not give a consistent improvement in the intended output gap.
The measurement does not establish that all possible local rules fail: there
is one hand, one trained lane, one direction-generation seed and eight response
seeds. No generalisation or full-game learning claim follows.

Prioritise the mapping from visible card relationships through KC activity to
the card-output response. Compare the exact two-pair hand against rank-matched
controls, test whether the relevant patterns reach afferents of the intended
outputs, and measure the output response to bounded local interventions.
Fixed visible relational features are a candidate if native sensory coding
does not provide a usable distinction. The original reward rule, signed-credit
pilot and such a sensory intervention should remain separate research arms.
Another unchanged long run or a large arbitrary learning-rate increase is not
supported by the current results.
