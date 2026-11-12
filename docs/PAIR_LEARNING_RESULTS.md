# Pair-learning pilot results

Assessed 26 October 2026. Run:
`artefacts/card-learning-20261024-v1/pair-pilot`.
All four conditions completed 5,000 decisions per learner across three
learners: 60,000 total training decisions. No condition safety-stopped.

## Held-out pair selection

Each learner was evaluated on 52 validation hands, with four seeded display
permutations/neural repeats per hand. Initial and trained weights used identical
stimuli and seeds. These are 52 hands with repeated measurements, not 208
independent hands. Learners and conditions share this validation set.

| Condition | Initial accuracy | Trained learner accuracies |
| --- | ---: | --- |
| Top-k, frozen | 7.21% | 7.21%, 7.21%, 7.21% |
| Subset exploration, frozen | 7.21% | 7.21%, 7.21%, 7.21% |
| Top-k, plastic | 7.21% | 2.88%, 3.85%, 3.85% |
| Subset exploration, plastic | 7.21% | 3.85%, 3.85%, 3.85% |

Uniform two-card selection has chance 1/28 = 3.57%. Initial accuracy was
15/208 trials; trained learners selected the pair on 6–8/208. All six plastic
learners declined against their own initial weights, and their mean native
scoring reward also declined. Neither condition demonstrated pair learning
at this exposure budget. This does not establish impossibility at other
budgets or under different rules; the corpus and training replicate count are
small and no formal significance claim is made.

## Controls, health and transfer

All four final checkpoint file hashes matched their manifests. Every learner
recorded exactly 5,000 decisions. Frozen weights exactly equalled initial
weights. Plastic weights changed, eligibility was cleared at the final task
boundary, and all final evaluations reported unchanged frozen plastic state.
There was no final efficacy-bound saturation. Maximum learner mean absolute
drift was about 0.00139 (0.139% of initial efficacy); the largest edge increase
was 0.01465. No plastic efficacy decreased, consistent with the pair task's
positive-progress-only feedback and absence of failure pulses.

In full-game transfer on 256 validation roots per learner, the initial/frozen
policy cleared one blind per learner. Top-k plastic learners cleared [2, 1, 2];
subset plastic learners cleared [4, 1, 3]. All had zero Ante-1 wins. Those small
clear-count changes do not override the failed pair-learning objective or
establish transferable pair recognition.

Machine-readable assessment:
`artefacts/card-learning-20261024-v1/pair-pilot/assessment-20261026.json`.
Existing trial reports and checkpoints were preserved. No new training was
launched during this assessment.

## Next work

Follow-up: the [direction and clipping audit](CARD_CREDIT_DIRECTION_AUDIT.md)
has been implemented and run. It preserves these source checkpoints and adds
a separate fixed-hand plastic/frozen capacity probe.

Prioritize a short learning-direction/eligibility audit before extending the
pair budget or launching the larger normal-game exploration factorial:

1. Hold observation and neural seed fixed. Measure card-pool scores before and
   after the actual selected-card reward update. Use exact-repeat/no-update
   controls, independent neural seeds, and a declared update-size sweep so an
   undetectably small single update is not mistaken for a reversed effect.
   Does positive feedback increase selected-card preference relative to its
   competitors, and does that effect transfer across display permutations?
2. Measure KC/MBON clipping specifically on the separated card pathways. The
   current references clip rates at 20 Hz, while mean MBON activity is around
   59 Hz; that is a reason to measure saturation, not a measured fraction of
   clipped selected edges. If distinctions are lost, test reward-free reference
   scaling with update magnitude controlled as a separate condition.
3. Check a tiny fixed-hand learning problem to distinguish basic update/credit
   failures from generalization across ranks and positions. Keep diagnostics
   and any experimental teaching signals explicitly separate from the primary
   runtime interface.

Changing run-failure punishment is not the immediate intervention: this
one-decision curriculum deliberately emits no run-failure pulse. All scored
choices receive positive progress feedback; whether a contrastive signal is
needed remains a hypothesis, not an established cause.
