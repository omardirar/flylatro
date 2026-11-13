# Local learning after the single-seed experiment

Research and diagnostic exploration, 6 November 2026. No training rule or
historical checkpoint was changed. [Completed run](SINGLE_SEED_RESULTS.md).

## Follow-up status

This page preserves the initial exploration/credit proposal. Subsequent
[exchange-credit training](EXCHANGE_CREDIT_PILOT.md) and
[causal update-direction tests](EXCHANGE_DIRECTION_RESULTS.md) did not qualify
that mechanism for long training. The later [relational-input pilot](RELATIONAL_SENSORY_PILOT.md)
and [sensory-control study](SENSORY_CONTROL_RESULTS.md) also failed to establish
a useful learning advantage. The current [roadmap](LEARNING_ROADMAP.md)
prioritises functional sensory/output control and a direct KC-afferent bypass
test. The exhaustive subset calculation below remains a conditional diagnostic.

## Main recommendation

Investigate exploration and credit assignment together. Prioritise a sampler
that makes small changes to a card set and a local rule that distinguishes the
exploratory change from the unchanged selected cards. Keep local postsynaptic
normalisation as a stability control. This is a candidate mechanism, not a
demonstrated gameplay improvement.

The existing rule multiplies clipped presynaptic and postsynaptic activity by
a boolean selected-output mask and then scalar reinforcement. The card sampler
usually chooses the top-ranked slots; 5% of decisions replace the whole subset
uniformly. This is not, by construction, an unbiased policy-gradient estimator
for that sampler. Normalising each neuron's incoming strength does not supply
the missing action-credit relationship.

## New exhaustive native diagnostic

`scripts/explore_selection_credit.py` loads the recorded endpoint opening
choices of all six trained flies. All choose the same five-card subset from
K♥, K♣, J♦, 10♦, 9♦, 6♣, 3♠, 3♦: the Kings plus 9♦, 6♣ and 3♠.
It scores 60 chips; keeping both Kings and both Threes scores 92.

The script independently evaluates every one of the 56 legal five-card subsets
in the native simulator. It then calculates exact expectations under three
explicit sampler distributions. The 95% greedy component is held fixed;
neural noise, changing weights and other states are outside this calculation.

| Quantity | 95% fixed choice + 5% random whole subset | 95% fixed choice + 5% random one-card swap |
| --- | ---: | ---: |
| Mean chips | 58.911 | 59.427 |
| Probability of a best-scoring subset | 0.357% | 0.667% |
| Mean score when missing 3♦ is selected, minus policy mean | −19.625 | +2.173 |
| Covariance of score with selection of 3♦ | −0.6133 | +0.0362 |

Whole-subset exploration often introduces 3♦ while removing a King. Consequently
the card that would complete two pairs looks harmful under this simple centred
selection-credit statistic. A one-card swap preserves more of the original
context and changes this statistic's sign. Uniform exploration over all 56
subsets gives the missing Three a positive conditional advantage of 1.071 chips,
so the negative sign is a property of the concentrated sampler mixture.

These are selection-credit **proxies**, not measured synaptic updates. They omit
KC/MBON coactivity, local normalisation, eligibility history and the stochastic
neural ranking. The negative sign is not proof that the actual trained fly
depresses this card's synapses. One-card swaps can also get stuck at local optima;
retain some broader exploration in any subsequent practical design.

Evidence: `artefacts/selection-credit-20261106-v1/report.json`, with native scores,
source evaluation hashes and per-card expectations. The code performs no
learning and uses no external trained policy.

## Proposed implementation order

1. **Match local exploratory credit to the sampler.** At fixed PLAY count,
   compare the existing subset sampler with single-card exchanges at matched
   exploration probability. Separately compare the existing rule with signed
   local eligibility that tags the entering/leaving output neurons and a
   centred outcome signal. A factorial comparison is needed: changing both at
   once cannot attribute an improvement. A recent-reward trace would be local
   bookkeeping, explicitly checkpointed; it must not become an external fitted
   critic. Derive and test the rule's assumptions instead of calling any signed
   equation a policy gradient. First measure whether actual updates increase
   the preference for a higher-scoring exchange across independent neural seeds.

2. **Resolve the terminal feedback objective.** The replay shows positive
   accumulated reinforcement in many failed games and zero native progress on
   losing final plays. Decide whether the objective is learning immediate hand
   score or learning blind success. For blind success, test potential-difference
   shaping with proper terminal accounting. Do not erase native rewards or
   silently retrofit this change into existing experiments. A potential-shaping
   theorem preserves an objective under its assumptions; it does not guarantee
   convergence of this local rule or solve delayed credit assignment.

3. **Improve relational input if causal learning still fails.** Older probes
   found surviving but noisy card signals, and even local teacher curricula
   struggled. Recheck the repaired mapping before assigning blame entirely to
   reinforcement. An explicit candidate is fixed per-card visible features
   such as same-rank companions and suit counts, routed into sensory neurons.
   These are engineered poker priors, not a discovered strategy. No best-hand
   labels or future draws should enter normal observations. Test with shuffled
   positions and unseen ranks/suits so fixed slot preference cannot pass.

4. **Train discard strategy after card competence.** Separate the decision to
   discard from the choice of cards. Resample unseen draws during strategy
   puzzles and reward subsequent success. Assistance must remain labelled;
   success with reference PLAY is insufficient unless the fly can perform the
   subsequent PLAY itself. Current own-PLAY transfer is a known bottleneck.

We have already tested isolated reward centring, reference-rate rescaling,
larger neural windows, more uniform exploration, fixed-count tasks and local
teaching. Their weak results argue against repeating a parameter sweep. The new
priority is a causal connection between an explored card replacement, its
outcome and the subsequent change in preference.

## Research basis and limits

- [Florian, 2007](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)
  derives reward-modulated local rules for stochastic spiking models and tests
  delayed eligibility. The eligibility is tied to the stochastic model; the
  paper does not validate this project's boolean selected-pool rule.
- [Legenstein, Pecevski and Maass, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2543108/)
  analyses conditions for reward-modulated STDP and the role of activity/reward
  relationships. It supports examining the learning signal rather than assuming
  any Hebbian update times reward optimises behaviour.
- [Bennett et al., 2021](https://www.nature.com/articles/s41467-021-22592-4)
  models reinforcement prediction errors using mushroom-body feedback. This
  motivates investigating relative outcomes; its model is not evidence that
  a single scalar baseline is sufficient for Balatro or that the current fly
  implements that circuit.
- [Ng, Harada and Russell, 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
  establishes conditions under which potential-based reward shaping preserves
  the optimal policy. Terminal handling and discounting must match the stated
  objective. The result is about reward objectives, not a guarantee for our
  local-plasticity learning dynamics.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/explore_selection_credit.py \
  runs/single-seed-4000-long-20261101-v1 \
  artefacts/selection-credit-new
```
