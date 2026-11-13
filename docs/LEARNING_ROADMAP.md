# Learning roadmap

Overview updated 12 November 2026; completed research through 6 November.
[PLAN.MD](../PLAN.MD) remains the scientific authority.
This roadmap orders exploratory work; it does not promote experimental rules,
interfaces, curricula or checkpoints into the primary model. Implementation
and commands: [Card learning experiments](CARD_LEARNING_EXPERIMENTS.md).

## Current priority

First establish functional sensory-to-output control and a repeatable local
positive-learning control. The latest anatomical ALPN cue study did not improve
choice accuracy, and a larger dose did not produce reliable directional control.
The next proposed test compares direct KC-afferent stimulation with ALPN cues
to isolate upstream routing from KC→MBON/readout controllability. It has not
been reported as completed. See [the sensory-control results](SENSORY_CONTROL_RESULTS.md).

Require functional calibration on separate confirmation seeds before proposing
another full-game long run. The studies below explain how this priority evolved;
their earlier “next” suggestions are historical and some have since been tested.
No new long-training candidate is qualified. The [documentation index](README.md)
provides methods and results by task.

## Long-exposure and stability evidence

The repaired native long experiment stopped at 509,922 decisions per plastic
learner after crossing the configured saturation limit. Partial progress
generalised to fresh roots, but card membership and discard use remain weak:
[completed review](REPAIRED_LONG_RESULTS.md). The next isolated intervention,
[local postsynaptic normalisation](LOCAL_NORMALISATION.md), completed its declared
20k-trial fixed-two-card comparison against unchanged and frozen controls.
[Results](LOCAL_NORMALISATION_RESULTS.md): strength budgets conserved, but fresh
pair accuracy 4.49% versus 4.01% original and 3.53% initial has uncertainty
intervals including zero and fails the declared advancement criteria. Preserve
the old checkpoints and reinforcement settings. Budget stability alone does
not establish useful card-choice learning.

[Native discard-strategy puzzles](DISCARD_STRATEGY_PUZZLES.md) now have a
completed 2k-trial pilot per learner, with independently resampled hidden draws,
equal-count discard alternatives and explicit PLAY assistance. A visible-rule
baseline reaches 84.15% expected fresh clears; trained original/normalised
actors remain at 50.00%/48.89%, and own-PLAY transfer gives zero clears on the
tested last-hand cases. Preserve the new benchmark. Next isolate timing from
discard selection and address the existing PLAY bottleneck; this is not a
qualified long-training improvement.

## Completed diagnostics and earlier hypotheses

The [anatomical sensory-control study](SENSORY_CONTROL_RESULTS.md) is complete.
Twelve actors attempted a synthetic cue association for 1k trials; neither
aligned nor reversed mappings improved. Frozen anatomy-selected ALPN cues were
poorly selective; a 3/12/24-neuron dose grid reached only 37.5–50% aligned
accuracy. Structural reachability alone does not qualify a sensory route.
Next use direct KC-afferent stimulation as a bypass control to distinguish
upstream sensory routing from KC→MBON/readout controllability. No new
full-game long-training candidate has been qualified.

The [fixed relational sensory pilot](RELATIONAL_SENSORY_PILOT.md) is implemented
and complete: 18 actors, 1k trials each, correct/off/wrong-slot features paired
with frozen controls. Shuffled-display validation pair counts changed 6→7/156
with correct hints, 8→10/156 without hints and 5→4/156 with wrong-slot hints.
No useful feature advantage is established. All 21,744 recorded decisions were
checked against native replay; hints identify the actual pair and frozen choices
are exact. Nineteen tests pass. Next prioritise a calibrated local
sensory-to-output controllability/positive-learning control, not deployment of
these random-projection hints or another unchanged long run.

The [pair-relationship representation probe](PAIR_REPRESENTATION_RESULTS.md) is
complete: a rank-matching factorial contrast survives into target-output KC
afferents but does not produce a consistently directed card preference.
Reversed-display matching contrasts are usually negative. All 32 exact-repeat
checks and three new analysis tests passed; source checkpoints are preserved.
Next test fixed per-card same-rank relationship inputs with equal-drive wrong-slot
controls, shuffled positions and unseen hands. This is an engineered sensory
hypothesis, not a demonstrated learning improvement.

The [causal exchange-update probe](EXCHANGE_DIRECTION_RESULTS.md) is complete.
A verified 60→92-chip exchange supplied clean paired native feedback to
disposable selected/signed update directions. Actual updates changed no tested
card preference; signed updates amplified to ~200× remained unchanged and
~2,000× produced mixed effects. All 512 exact restoration checks passed and
source checkpoints were preserved. Update magnitude alone is not a qualified
fix. Next prioritise relational sensory activity and its causal effect at the
intended card outputs before increasing learning rate or duration.

The [exchange-credit factorial pilot](EXCHANGE_CREDIT_PILOT.md) is implemented
and complete: 2k trials × five conditions × three learners, 30k native actions.
Repeated-opening scores did not improve under signed credit. A small unseen
gain came from only one learner. Realised signed updates were 70–150 times
smaller than selected-credit updates, despite equal nominal learning rates.
Native feedback/EMA accounting, local budgets and exact frozen choices passed.
Next inspect causal preference changes at matched bounded update magnitudes;
the new rule is not qualified for long training.

[New research and exhaustive native exploration](LOCAL_LEARNING_NEXT_STEPS.md)
identifies an exploration/credit interaction on the repeated opening: under a
fixed 95% greedy + 5% whole-subset mixture, selecting the missing Three has a
negative reward-selection covariance; replacing only one card changes the sign.
All 56 five-card actions were checked natively. This is a conditional diagnostic,
not a neural training result. Prioritise matched exploration and signed local
credit, then terminal reward accounting and relational sensory inputs. Avoid
repeating already unsuccessful isolated centring/reference-rate sweeps.

The [single-seed long-training runner](SINGLE_SEED_TRAINING.md) is implemented
for a separate capacity/delayed-generalisation experiment. It repeats the same
native game after every terminal outcome, preserves independent learner IDs,
compares normalised/original plasticity and evaluates frozen policies on fixed
unseen seeds. The default is 1m decisions per learner, three learners per arm;
the long experiment has now completed all six million decisions. The
[completed review](SINGLE_SEED_RESULTS.md) finds no reliable repeated-game
mastery or evidence of grokking: zero first-blind clears in all saved greedy
training-seed checks, final unseen clears 6/192 normalised and 4/192 original,
and no greedy Ante-1 wins. Stochastic training found nine/eight Ante-1 wins
over 479,816/538,431 attempts. Normalisation preserved strength budgets; card
membership remained close to same-count random selection while count moved to
five. Prioritise the recurring two-pair selection/credit failure over another
unchanged long extension. Full CPU evaluation replay and training-episode
accounting are preserved with the review.

The separated-pool long run completed its declared 100k decisions per learner
without a safety stop. It used three independent learners, 100 ms decisions,
choice-gated KC→MBON plasticity, no blind skips and failure pulse 0.01. The final
checkpoint remains the baseline. [Completed results](SEPARATED_LONG_RESULTS.md)
show 31 clears in 768 validation games, zero Ante-1 wins, and a strong shift
toward five-card plays. The [pair capacity pilot](PAIR_LEARNING_RESULTS.md) has
also completed: all six plastic learners declined from 7.21% initial pair
selection to 2.88–3.85%, near 3.57% chance. The
[learning-direction and clipping audit](CARD_CREDIT_DIRECTION_AUDIT.md) now
confirms substantial selected-edge clipping and mostly invisible single updates;
use these measurements to design controlled follow-ups before long training.

The completed 20k checkpoint had five blind clears in 768 validation games and
zero Ante-1 wins. In the first Small Blind, 79.4% of plays were High Card;
508 plays missed an immediate clear and 308 discards occurred with a clear
available. These are turns, not independent games. Of 763 first-blind failures,
515 had discards unused. In the training prefix, 51.2% of failed runs had a
positive summed reinforcement pulse; this is not net synaptic change.

Detailed local evidence: `artefacts/run-decision-audit-20261024/README.md`.
Reusable audit code: `scripts/card_learning/audit.py`.

## Short term: basic card-choice learning

| Work | Status | Question and completion evidence |
| --- | --- | --- |
| Replay-based decision audit | Implemented | Verify hashes/rewards; measure missed clears, hand types, same-count random comparison and unused discards. Refuse divergent/incomplete logs. |
| Complete delayed-learning curve | Completed; final decisions audited | Delayed partial improvement observed; card membership and discard use remain weak. Preserve the final checkpoint. |
| Card-subset exploration | Legacy/current-layout screens complete; no fresh gain over initial | Preserve type/count exploration and native feedback. The 25% condition beats trained 5% on fresh pair cases but matches untrained initial performance; it is not qualified for full-game scale-up. |
| Plasticity controls | Validated in real pair pilot | Four conditions completed; frozen weights stayed exact, plastic weights changed. Frozen controls act independently, unlike matched-action replay controls. |
| One-pair curriculum | Completed; pair learning not demonstrated | Eight cards, exactly one pair, forced PLAY/two cards. Trained accuracy declined to near chance; no Ante-1 transfer wins. |
| Pair corpus and benchmark | Implemented | Disjoint game roots, equal pair-rank quotas, seeded shuffled positions, native scoring. No pair target enters motor inputs or reinforcement. |
| Resume and transfer checks | Implemented | Preserve future choices, environments and synaptic state on resume. Restore free action/count heads for full-game transfer. |
| Reward-to-output direction and clipping | Implemented and run on real FlyWire | Exact controls pass; selected MBON activity clips completely at 20 Hz. Amplified effects are mixed, not a simple global sign error. |
| Fixed-hand capacity probe | Implemented and run; no gain demonstrated | After 5k decisions, plastic/frozen copies selected the pair on 2/128 versus 3/128 held-seed evaluations. One hand; preserve discovery-count and uncertainty limits. |
| Matched reference scaling | Implemented and run; no gain demonstrated | Calibrated 60/170 Hz references reduce clipping, but final pair counts are 1/128 rescaled versus 2/128 baseline and 3/128 frozen. Expected calibration update matched; realized rescaled update L1 is 20.4% larger. [Results](CARD_REFERENCE_SCALING.md). |
| Native-score reward contrast | Implemented and run; no gain over initial/frozen policy | Native/centered/frozen final counts 2/3/3 of 128, versus 3 initially. Centered feedback produces both weight signs, but actual update L1 is 18.3% larger despite calibration pulse matching. [Results](CARD_REWARD_CONTRAST.md). |
| Separate PLAY/DISCARD preferences and credit | Implemented; code/regression and native interface checks | New versioned card/count heads with exclusive neurons and action-specific credit. Explicit single-MBON option width fits the responsive candidate budget. Legacy checkpoints retain old semantics. [Implementation/tests](PLAY_DISCARD_INTERFACE.md). |
| First separate-head learning pilot and fresh validation | Completed; original win not reproduced | Fresh 512-root evaluation: zero wins across all six trained policies; subset cleared 4 blinds and top-k 6 across 1,536 episodes each, versus zero in the 512-episode initial control. Discard use remains weak. [Results](PLAY_DISCARD_LEARNING_RESULTS.md). |
| Frozen greedy versus training-sampled evaluation | Completed; sampling explains frequency gap | 2,048 episodes on 128 shared fresh roots. Trained discards ~4% greedy versus 28–29% sampled; initial policies show the same gap. No wins or consistent sampled learned advantage. [Results](DISCARD_LEARNING_STUDIES.md). |
| Native delayed DISCARD → fixed-PLAY curriculum | Completed; useful discard learning not demonstrated | 5k trials per learner, three plastic/frozen pairs; only DISCARD card/count weights changed. Held-out greedy progress declined in all learners; sampled changes mixed/flat. Same-count random advantage inconsistent. [Results](DISCARD_LEARNING_STUDIES.md). |
| Free-action transfer from discard curriculum | Completed; exploratory partial gains, no wins | 512 episodes on 64 shared roots; mean returns slightly higher, one declining learner per mode, greedy discard use still 3.58%. Primary curriculum outcome did not improve. [Results](DISCARD_LEARNING_STUDIES.md). |
| Simple current-layout preference learning | Completed; action-head learning demonstrated | Three counterbalanced plastic/frozen pairs per reward target, 3k choices each. Target probability improved in all six trained copies: PLAY 72.44% → 82.24%, DISCARD 27.56% → 34.37%. Artificial test reward, not native training. [Results](TRAINING_BOTTLENECK_STUDIES.md). |
| Broader card-subset exploration | Completed; no fresh gain over initial | 5k two-card trials per learner at 5%/25%. Initial screen positive; fresh 104-hand confirmation: initial 47, 5% 37, 25% 48 pairs / 1,248 trials each. More exploration beats the trained 5% condition but does not demonstrate improvement over initial. [Results](TRAINING_BOTTLENECK_STUDIES.md). |
| PLAY competence and feedback informativeness | Completed; stronger follow-up mostly changes count | 2.5k DISCARD→PLAY trials per learner. Reference-player progress rises 0.368 → 0.417, discard count 2.67 → 4.92. Same-count random improves almost as much; membership evidence mixed and case intervals span zero. Reward/update strength differs materially. [Results](TRAINING_BOTTLENECK_STUDIES.md). |
| Card-output preference and fixed-count discard learning | Completed; partial output capacity, membership gains preliminary | Five of six card-output targets learned; one had 1,070 rewards/zero updates. Fixed-count strong-reference feedback showed a small fresh hand-potential signal, without improvement using neural PLAY; matched-budget gains mixed. [Results](CARD_MEMBERSHIP_LEARNING.md). |
| Reward-free plastic-input coverage | Implemented and audited; active option fails | DISCARD slot 3's MBON fires, but none of its 121 plastic KC inputs were active in the probes or calibration contexts. No efficacy change in any of nine native learners. Output response alone misses this. [Diagnosis](CARD_MEMBERSHIP_LEARNING.md). |
| Learnability-qualified motor mapping | Implemented and validated on fresh learners | Two reward-free pool repairs; original gates retained. Previously stuck DISCARD slot 3 now learns 52.97% → 90.08%, versus old 32.19% → 32.19%, at 1,500 choices. Independent saved-state/input-gate checks, 18 native replays and 83 tests pass. Existing checkpoints retain original routing. [Results and config](LEARNABLE_MOTOR_MAPPING.md). |
| Native membership learning with repaired mapping | Completed; no fresh gain | 2k trials, three seeds/feedback mode. On 104 fresh hands, neural-native progress 0.07327 → 0.07047; all three learners decline. Stronger and matched reference feedback also fail to improve fresh means. Restored updates are verified; 106,560 native actions replayed and 91 tests pass. [Methods/results and long-training criteria](REPAIRED_NATIVE_MEMBERSHIP.md). |
| Conditional card-choice capacity | Implemented in local teacher curriculum; no dependable native gain | Balanced higher-card/pair-member tasks, full pair/hand targets, fresh controls; small card-metric shifts did not establish first-blind competence. The standalone diagnostic was superseded by the user's request to build a new learner. [Implementation/results](LOCAL_TEACHER_CURRICULUM.md). |
| Local-only teacher-guided learning redesign | Implemented; two bounded candidates completed | Smooth/centered local eligibility, neuron-specific signed choice/rate instructions, staged visible-state demonstrations, checkpoint/resume and teacher-free PLAY-only native evaluation. First-blind clears: choice instruction 1/192 (same as random instruction); rate targets 0/192. 100 tests and 38,188 replayed native actions pass; no promotion. [Details](LOCAL_TEACHER_CURRICULUM.md). |
| Local learning/model-interface redesign | Next substantive design decision | The user retained local plasticity only. Neither candidate reliably fits conditional card-choice demonstrations or improves native first-blind play. Revisit representation/anatomical input coverage/recurrent response and the local rule as a coherent model; do not assume another duration or punishment sweep solves it. |

The pair capacity test used 5k decisions per learner. The prepared normal-game
exploration pilot uses 20k, with three learners per condition. These are
descriptive screens, not power calculations. Require improvement against each
condition's own initial weights and frozen controls; confirm with additional
training seeds and a fresh validation block. Shared validation roots are not
independent training replicates. Leave final-test seeds locked.

Report pair accuracy, blind clears, missed clears, unused discards at failure,
same-state score regret and slot-permutation tracking. Pair-task success alone
does not establish Balatro competence. The capacity pilot did not demonstrate
learning, so investigate representation and credit before another long
full-game budget. Reward-to-output direction and clipping have now been measured;
preserve current checkpoints and control update magnitude when testing new
references. A clipped coactivity signal and weak exploration/feedback contrast
are hypotheses to distinguish, not established exclusive causes.

The fixed-hand probe also found no memorization gain at the tested budget.
The reward-free reference-scaling comparison completed with calibration update
magnitude matched. Reduced clipping did not produce a fixed-hand learning gain
at 5k decisions; it does not qualify new references for long training. The
separate native-score reward-contrast comparison also completed without a gain
over its initial/frozen policy. Pair labels were excluded from feedback;
outcome-based feedback calibration remained separate from reward-free
sensory/motor calibration. The new PLAY/DISCARD layout's first learning pilot
has completed: one subset-plastic learner won a held-out Ante-1 game, verified
by native replay. Fresh frozen validation has now completed with zero wins
across six policies on 512 shared roots; small partial-progress gains remain.
The sampled-versus-greedy comparison and native discard→play curriculum have
now completed. Sampling explains the discard-frequency gap but does not rescue
learned performance. The curriculum verified delayed updates without useful
held-out gains; its fixed initial PLAY policy captured only ~30% of available
immediate progress. Free-action transfer had small mixed partial gains and no
wins. Follow-up capacity screens have now completed: the action head learns a
simple rewarded choice, 25% subset exploration has no fresh improvement over
initial weights, and a stronger reference PLAY partner mostly drives discard
count toward five. Direct card-output and fixed-count screens have now also
completed: five of six fixed-output targets learned, while native membership
gains remained preliminary on fresh hands and did not improve results with
neural PLAY. One active DISCARD option had no eligible KC input despite 1,070
rewarded selections. A new versioned mapping now requires active plastic-input
support: two pool replacements pass reward-free qualification and restore
learning of that target. The matched native fixed-count comparison is now
complete: the repaired output receives real native updates, but no feedback
condition improves fresh mean progress and all neural-native learners decline.
The user then retained local-plasticity-only learning and requested a learner
redesign. Two teacher-guided local candidates have completed, with signed
synaptic updates and correct controls but no reliable first-blind improvement.
Their conditional training curves remain near chance. The next step requires
a substantive local learning/model-interface design change; neither longer
exposure nor punishment tuning is qualified by these results. See
[local-only teacher implementation and bounded results](LOCAL_TEACHER_CURRICULUM.md),
[native repair comparison and scale-up criteria](REPAIRED_NATIVE_MEMBERSHIP.md),
[mapping implementation and validation](LEARNABLE_MOTOR_MAPPING.md),
[new card-output tests and eligibility diagnosis](CARD_MEMBERSHIP_LEARNING.md),
[new screens and fresh confirmation](TRAINING_BOTTLENECK_STUDIES.md) and
[methods, results and reproduction](DISCARD_LEARNING_STUDIES.md).
Training discarded on about 28% of card actions, versus 3.5–4.2% under greedy
evaluation; 3,061/3,063 fresh first-blind failures left discards unused. The old
checkpoints retain shared preferences. Broader uniform subset exploration was
tested separately with native feedback and references fixed; retain its null
comparison against initial weights rather than promote the higher rate. Report discovery counts
and compare frozen evaluations with each condition's own initial weights and a
frozen control. Do not combine unqualified reference/feedback changes. A blanket
sign reversal is not qualified by the mixed direction-audit results.

## Medium term: interface and credit

1. **Action-conditioned cards and counts.** Implemented and tested for new
   mappings: separate PLAY/DISCARD rankings, counts and fresh credit; native
   opposing-choice fixtures pass. Old checkpoints still share a ranking.
   The first full-game learning pilot produced one verified Ante-1 win, with
   no discards in that winning episode. Fresh validation did not reproduce
   wins. Diagnose action sampling and delayed discard credit before claiming
   learned discard competence or extending training. Single-MBON
   option width is an explicit architectural change. Assess consumable targeting
   separately; it retains the PLAY interpretation in this version.
2. **Progress-reward ablation.** Compare progress 0.05 versus 0.01 with failure
   held at 0.01 and clear reward at 0.4; later consider clear-only feedback.
   Keep the decoder fixed. Do not reduce punishment solely because wins are rare.
3. **Eligibility scaling.** Clipping measurements and a reward-free calibrated
   comparison are implemented. Match expected calibration update magnitude and
   report realized trajectory differences. Then isolate trace duration and more
   selective credit as separate experiments.
4. **First-blind curriculum.** Progress from card selection to the Small Blind,
   then the three-blind Ante 1. `ladder=[1]` changes the run goal, not these
   prerequisite tasks. Define boundaries and reward semantics explicitly.
5. **Representation and mapping replicates.** Repeat identity/permutation probes
   on new mappings. Vary sensory/motor mappings independently of learner seeds.
   Rank/suit-relation features are a labelled feature-engineering ablation,
   not a silent change to the primary strategy-neutral encoder.
6. **Discard quality.** Measure broken combinations and observable draw odds.
   Breaking the current best hand is not automatically an error. Do not use
   hidden next draws as a teacher or equate immediate score with long-term value.

Beyond the implemented reference-scaling, reward-contrast and action-conditioned
interface probes, these
changes remain planned.
Test each factor separately before combining successful decoder, reward and
credit changes. The current pair corpus balances
pair rank and randomizes position; explicit suit-combination quotas and broader
task corpora remain additional controls to explore.

## Long term: whole-run strategy and biological questions

* Add shop/interest, Joker combinations/order, consumable targeting and boss
  scenarios once enough later-game exposure exists to evaluate them.
* Extend the Ante ladder toward 8, with declared snapshot curricula and transfer
  to fresh complete games without artificial task resets.
* Explore persistent neural state after basic card learning. It requires backend
  state per learner and correct reset/checkpoint handling. Eligibility is a
  learning trace, not policy working memory.
* Compare MBON-direct outputs with downstream whole-brain/descending outputs,
  keeping zero external trained policy parameters and matched controls.
* Test compartment/sign/timing rules with defensible anatomical evidence.
  Keep unknown DAN compartments unknown; neuron-pair edges do not establish
  synapse-level compartment identity.
* Confirm findings with more training seeds, mapping replicates, uncertainty
  estimates respecting shared evaluation roots, a locked final test and replay
  packages. Report unsuccessful experiments too.

## Experiment contract

Only existing KC→MBON efficacies learn. Record sampler, task, source, mapping,
corpus, seed and reward identities. Changing exploration or tasks requires a
fresh condition; exact resume cannot change them. Keep calibration/probe labels
out of runtime policy inputs and preserve safety gates.

Research code was kept outside `src/` to protect the completed long run. A
versioned refactor can promote qualified code into the package while retaining
strict checkpoint identity checks. Prepared pilot declarations are also bound
to the current implementation and must be regenerated after code changes.
