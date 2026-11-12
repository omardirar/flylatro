# Exploring discard strategy with native delayed rewards

Completed 1 November 2026. Main run:
`artefacts/discard-strategy-20261101-v1/`. All 2,000 trials per learner completed
in 467 seconds. The transient user service
`flylatro-discard-strategy-20261101-v1.service` exited successfully and is inactive.

**Finding:** this benchmark contains a clear, useful discard strategy. A fixed
visible-information heuristic reaches 84.15% estimated fresh blind clears with
assistance. The local-plasticity actors remain around 49–50% and have not learned
the conditional strategy. Removing assistance from PLAY produces no clears on
the tested last-hand transfer cases. This is a bounded negative learning result,
not proof that longer training or every local learning rule must fail.

## Rules and research basis

Discarding exchanges immediate scoring for a chance at a better future hand.
The objective here is clearing the blind, rather than discarding frequently or
maximising the frequency of one poker-hand name. A discard can replace 1–5
cards for one resource; unselected cards stay in hand. Playing can also cycle
non-scoring cards alongside a scoring combination. Resources, remaining cards,
hand levels and Jokers can change the preferred choice.

Relevant sources:

- [Official Balatro FAQ](https://www.playbalatro.com/faq): blinds and build-dependent Joker effects.
- [Wisker, LIACS thesis](https://theses.liacs.nl/pdf/2025-2026-WiskerJ.pdf): discard probabilities and an evaluator for desired hand types. Its deck comparison omits Joker-slot disadvantages; it is not an optimal full-game policy.
- [Associated discard-calculation mod](https://github.com/justinw202/BalatroDiscardMod).
- [Eligibility traces and three-factor learning](https://arxiv.org/abs/1801.05219): delayed outcomes can modulate previously marked synapses; this does not validate our particular rule or establish Balatro learning.
- [Pinned simulator](https://github.com/jahankazimi078/balatroagent/blob/38ae214317009952db4d22a98dc0765cef79370a/sim/core/src/run.rs): Red Deck/White Stake starts with four hands and four discards. Its native `redeterminize` API changes only unobserved deck order and future RNG streams.

For illustration, on a fresh standard deck, an eight-card hand containing four
hearts has nine hearts among 44 unseen cards. Replacing four off-suit cards gives
`1 - C(35,4)/C(44,4) = 61.43%` chance of completing a flush. That is a flush
probability, not automatically the chance of clearing the blind. The actual
task uses the remaining native deck after its setup actions, not this formula's
fresh-deal assumptions.

## Native puzzle construction and assistance

`scripts/strategy_puzzles.py` constructs reachable first-Small-Blind states using
only native actions: select the blind, then three one-card discards and three
one-card plays. These low-scoring setup plays cannot clear 300 chips. Each
snapshot has exactly one hand and one discard remaining. The setup is a way to
construct late-blind recovery exercises, not a proposed good opening strategy.

Cases are selected using visible features only:

- **Play now:** a visible five-card suit combination already clears the target.
- **Draw needed:** four cards of a suit, no current clear, and sufficient visible
  card values that completing a plain flush can clear the remaining target.

Every split has three examples of each family for each suit: 24 training cases,
24 held-out cases and 24 fresh-confirmation cases. No snapshot or root overlaps
the splits. All cases are kept after the reference evaluation; none is dropped
because a sampled outcome is unfavourable.

The assisted action space is explicit: PLAY now, or either of two curated discard
sets. Both sets discard exactly four cards. One tends to preserve the dominant
suit and the other damages it; their order is randomised per case. The fly scores
them through its actual DISCARD-card outputs; candidate indices are not labels
fed into its sensory input. This is a restricted-choice capacity task, not a
demonstration of unrestricted card selection.

After the fly chooses, the unseen deck order and future RNG are resampled. Every
observable array and legal mask must remain unchanged. Training has new futures
every trial, paired across treatment/control actors within a replicate. The
neural decision occurs before resampling, making the new future unavailable to
the policy. No future-order information or reference action values enter the
encoder or learning rule. A separate test compares empirical flush draws with
the exact without-replacement probability from the public deck counts.

A fixed PLAY helper selects the highest immediate-scoring subset of the current
visible hand, before PLAY or after DISCARD. It never sees the future draw before
the discard. Both task aids are labelled in the declaration and reports; helper
success is not reported as the fly's independent playing competence.

## Learning and controls

`scripts/discard_strategy.py` and `scripts/run_discard_strategy.py` compare three
fresh learners each under original three-factor updates, local-normalised
updates, and frozen weights. The main run has **18,000 learned/frozen actor
trials**, with additional native helper actions when DISCARD is chosen.

Only PLAY/DISCARD type posts and DISCARD membership posts can learn. PLAY card
selection and both card-count heads are excluded from new credit. DISCARD
eligibility persists until the helper's PLAY outcome, decaying by 0.9 for that
extra decision; immediate PLAY traces do not take that extra decay. Native
progress/clear/failure supplies reinforcement, with no direct discard bonus,
hand-type teacher, gradient or learned external critic. The existing learning
rate, activity references, dynamics and reinforcement magnitudes remain fixed.

Inactive vector-environment rows need legal auxiliary actions while other rows
finish their helper PLAY. Those auxiliary actions never provide learning or
count as the fly's strategy choices. The CPU replay checks only genuine trial
actions and verifies the per-learner decision clock against the number of
discard/helper steps.

Source/frozen/unrelated weights, efficacy bounds and normalised budgets are
checked throughout. NPZ checkpoint generations bind all state, the cursor,
declared source/input hashes and the training-log length. Resume truncates an
uncheckpointed tail. A real-GPU restore reproduced three subsequent trials,
including hidden draws, outcomes and every state array exactly. Existing
production and historical research files/checkpoints are preserved.

## Reference benchmark

The reference samples 128 independent hidden futures per visible case, shared
across the three choices. All outcomes are executed in the native simulator.
Values describe the assisted one-step task, not an optimal complete Balatro run.

| Fixed policy | Training cases | Held-out cases | Fresh cases |
| --- | ---: | ---: | ---: |
| Always PLAY now | 50.00% | 50.00% | 50.00% |
| Uniform random among three choices | 43.15% | 43.20% | 44.62% |
| Play if already able to clear; otherwise retain dominant suit | 82.26% | 84.08% | 84.15% |
| Best of three choices by estimated reference values | 82.26% | 84.08% | 84.15% |

The fixed heuristic is an exploratory baseline, not fitted or selected from a
family of policies using outcomes. It uses only visible card/score information
and matches the reference on this deliberately simple curriculum. It is not
installed in the fly's policy. The fresh baseline demonstrates about 34 points
of strategy headroom beyond always playing.

## Learning result

Policies are evaluated with two matched neural/display repeats per visible case.
For each actor choice, its expected clear rate comes from the 128-future
reference bank. Rates below average three learners over 24 shared fresh cases.
Neither repeats, future draws nor actors create additional independent cases.

| Actor condition | Initial assisted expected clears | Final assisted expected clears |
| --- | ---: | ---: |
| Original rule | 48.82% | 50.00% |
| Normalised rule | 48.82% | 48.89% |
| Frozen | 48.82% | 48.82% |

The original-rule actors finish by choosing PLAY on every fresh case, including
every draw-needed case. Their small gain reflects eliminating some bad discards
in already-winning situations, not acquiring the conditional drawing strategy.

Normalised actors choose PLAY in 97.22% of cases on average in **both** families.
Their mean gain is only 0.076 percentage points. Per-learner changes are
[−1.94, +0.96, +1.20] points. The conditional case-bootstrap 95% interval for mean
gain is [−1.19, +1.02] points; its advantage over original-rule actors is
[−3.09, +0.33] points. These intervals condition on the three learners and finite
reference draw bank. The predeclared five-point gain and ten-point conditional
type-contrast criteria fail.

Normalised strength budgets remain conserved within 4.7e-8 relative. Original
neurons' maximum total incoming-strength gains reach about 55–57%. No actor has
an efficacy at a bound at the endpoint, and no safety stop occurs. Stability is
therefore distinct from acquiring strategic decisions.

## Separate transfer and skill checks

Removing the curated two-discard menu, while retaining fixed discard count four
and the reference PLAY helper, gives estimated fresh clears of 50.00% original,
48.91% normalised and 48.13% frozen. This does not establish useful raw
discard-membership learning.

Removing the PLAY helper as well produces **0/576** observed clears per trained
condition (24 cases × eight independent futures × three learners), also zero
for initial weights. The fly chooses its own PLAY count/cards and raw DISCARD
membership in this check; discard count remains fixed. This is transfer on these
last-hand puzzles, not a full-game evaluation. PLAY heads were intentionally not
trained by this experiment, so the zero rate confirms dependence on assistance
rather than showing that every possible PLAY training procedure must fail.

Evaluation-only decomposition shows that giving correct visible-rule timing to
the existing forced-discard preferences would yield about 71–73% expected
assisted clears, compared with 84.15% for the fixed heuristic. These are assisted
counterfactuals, not skills learned by the fly. They distinguish timing from
discard selection; detailed values are in `skill-decomposition.json`.

## Independent audit and next interpretation

`scripts/assess_discard_strategy.py` verifies all 18,000 training records,
27,648 reference outcome records, 864 free-membership policy records and 3,456
fly-PLAY transfer records. It independently executes the recorded native actions
and fixed helper choices, caching identical replay combinations. It checks the
natural prefixes, hidden-future invariance, native rewards, equal-count options,
frozen individual decisions, unrelated weights, weighted budgets, checkpoint
hashes and exact delayed-credit clocks. Counts distinguish records from unique
native actions. The assessment is bound to the final report and assessor hashes.

The benchmark and the fixed heuristic establish a concrete trainable objective;
the learner has not acquired it at this budget. The next focused investigation
would isolate action timing with a fixed competent discard selector, and
separately address PLAY competence. That would test the coupled-skill bootstrap
problem without claiming another reward-magnitude adjustment is a proven fix.
Longer learning remains an empirical possibility; this pilot does not establish
that additional duration will produce strategy.

Reproduce in a new directory using the GCC14/CUDA environment documented for
[local normalisation](LOCAL_NORMALISATION.md):

```bash
.venv/bin/python scripts/run_discard_strategy.py \
  --output-dir artefacts/discard-strategy-new --heavy
.venv/bin/python scripts/assess_discard_strategy.py \
  --output-dir artefacts/discard-strategy-new
```

The main directory is completed; rerunning it returns its saved report. No
long full-game training or change to the primary model was made.
