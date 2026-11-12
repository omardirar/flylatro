# Frozen pair-relationship representation probe

Completed 6 November 2026 after the [causal update probe](EXCHANGE_DIRECTION_RESULTS.md).
Code: `scripts/probe_pair_representation.py`.
Evidence: `artefacts/pair-representation-20261106-v1/report.json` and hash-bound
`responses.npz`. No learning or checkpoint modification occurs.

## Factorial intervention

Two visible card slots have ranks A or B, yielding AA, AB, BA and BB. The
matching contrast is `(AA + BB - AB - BA) / 2`, cancelling purely additive
effects of the individual ranks. Both cards retain their suits. All other hand
features, global observations and deck summaries stay fixed. These synthetic
visible inputs are not altered native simulator worlds; they may disagree
with the deck summary or normal sorting.

Rank pairs are (2,3) and (7,8). Conditions use the recurring opening background,
the original and reversed card displays, unit weights and the signed-exchange
pilot's first trained learner. Eight matched neural seeds per condition isolate
input effects from independent-seed variability. Outputs follow the physical
target card across display reversal. The reference output is the physical 6♣.

The probe measures ALPN drive, KC rates, unique KCs with anatomical inputs to
the target card output, clipped target-edge coactivity, target output rate,
and the target-minus-reference normalised motor score. Cross-repeat contrast
power is retained even when negative; it is descriptive and not a decoding
test or a significance test. No matching labels enter the neural input.

## Results

For the trained learner:

| Rank pair | Display | Matching effect on target/reference score gap | Independent-repeat score noise RMS |
| --- | --- | ---: | ---: |
| 2/3 | Original | +0.253 | 1.316 |
| 2/3 | Reversed | −0.675 | 1.089 |
| 7/8 | Original | +0.084 | 1.265 |
| 7/8 | Reversed | −0.506 | 1.004 |

The 2/3 reversed-display matching contrast is negative in seven of eight seeds
and never positive. The 7/8 reversed contrast is negative in five and positive
in two. These are conditional measurements on one background, not independent
game-level observations.

The original target output receives inputs from 684 unique KCs; the reversed
target uses a different motor neuron with 1,719 unique KC inputs. Matching
contrasts in these afferent rates have RMS 0.82–1.15 Hz, versus trial noise
3.13–3.44 Hz. Cross-repeat power is positive in three of four trained conditions
and negative in one. Clipped coactivity contrast RMS is 0.019–0.032 versus noise
0.072–0.090. Its cross-repeat power is small and changes sign across conditions.
There is no consistently directed match bonus at the card-output score.

The deterministic sensory drive has exactly zero matching interaction in all
tested conditions. This is expected for an additive identity encoder in the
absence of relevant clipping interactions; it does **not** imply that rank
identities were erased or that a downstream nonlinear circuit cannot compute
matching. The nonzero KC contrasts show some downstream interaction, but the
observed output response is weak or incorrectly directed for pair selection.

All 32 exact neural repeat controls passed. The saved source checkpoint hash
remained unchanged. Three new tests verify cancellation of additive identity
effects, correct matching contrast, retention of negative cross-repeat power,
and isolated rank edits.

## Decision

The evidence supports prioritising an explicit, fixed visible relationship
encoding: for each card, expose the count of other visible cards sharing its
rank, with analogous suit information as a separately tested feature. This is
engineered game knowledge, not a learned discovery. It can be encoded as fixed
sensory stimulation while keeping all learned weights anatomical KC→MBON.

A useful comparison must include an equal-drive control with relationship
channels assigned to wrong card slots: otherwise extra stimulation alone can
look beneficial. Evaluate shuffled displays, unseen hands and rank combinations,
and compare frozen weights with local plasticity. First require useful card
membership learning at fixed count before full-game strategy training.

This probe does not establish that adding those features will improve learning.
It does narrow the current failure: the existing path does not deliver a stable,
correctly directed matching preference in the tested cases, and neither more
duration nor larger updates has repaired that behaviour so far.
