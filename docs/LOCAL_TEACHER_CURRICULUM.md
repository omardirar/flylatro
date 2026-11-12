# Local-plasticity teacher curriculum

The user explicitly retained **local-plasticity-only** learning. This implementation
keeps all trainable parameters on existing anatomical KC→MBON edges and introduces
no optimizer, backpropagation, fitted external policy, decoder or critic. It is a
separate experimental path; primary V1 source, configurations, calibration and
legacy checkpoints remain unchanged.

## Implemented components

* `scripts/local_teaching.py`: bounded local covariance plasticity and output
  teaching-pulse routing, with NumPy and Torch implementations.
* `scripts/teacher_curriculum_data.py`: visible-state demonstration generation,
  balanced two-option scaffolds and teacher-free native card-play evaluation.
* `scripts/run_local_teacher_curriculum.py`: choice-discrepancy teaching,
  staged training, controls, durable checkpoints and exact resume.
* `scripts/local_rate_teaching.py` and `scripts/run_local_rate_teacher.py`:
  neuron-local rate-target instruction as the second bounded candidate.
* `scripts/assess_local_teacher_curriculum.py`: independent anatomical weight,
  exposure, checkpoint, control and native-action replay verification.

The development budget is two candidates, not an open-ended hyperparameter
search. Candidate 1 uses 1,000 frames in each of five stages, candidate 2 uses
1,000 frames in each of three stages. Each has three learner seeds and three
actors per seed: real instruction, random instruction and frozen. Random
instructions preserve cardinality priors drawn only from training demonstrations.
Every actor is evaluated; no favourable learner is selected for reporting.

Training compares greedy neural choices with the instruction; inference is
greedy as well. The inherited subset-exploration wrapper is present for mapping
compatibility but is disabled by these deterministic calls. Teaching reaches
legal output neurons directly, so action discovery is not supplied by a hidden
exploration schedule. Native reinforcement/punishment settings are unchanged
and are not the teaching signal used during this pretraining.

## Local rule

Each synapse uses its own presynaptic rate `pre`, postsynaptic rate `post`,
presynaptic activity-mean trace `mu`, eligibility and efficacy:

```text
x = pre / (pre + 20 Hz)
y = post / (post + 20 Hz)
eligibility = clip(lambda * eligibility + (x - 0.99 * mu) * y, -1, 1)
mu = mu + 0.01 * (x - mu)
delta = clip(0.05 * eligibility * teaching_signal_at_this_MBON, -0.01, 0.01)
efficacy = clip(efficacy + delta, 0.2, 2.0)
```

Means are calibrated over 64 label-free training frames at each stage before
updates begin. `lambda=0` for immediate teaching; eligibility is cleared after
each instruction. The 1% uncentered component permits a common response bias
to learn, while most of the input term retains activity differences. Smooth
compression preserves differences above 20 Hz that the old hard cutoff erased.
Means and counters are local activity state; inference reads only efficacy.

Candidate 1's output instruction is desired card/count membership minus actual
membership. Correct choices produce no discrepancy. Candidate 2 instead supplies
a desired firing level at each legal output:

```text
margin = max(2 * fixed_calibration_scale, one_spike_per_decision)
desired_rate = clip(fixed_baseline + sign * margin, 0, 200 Hz)
teaching_signal = clip((desired_rate - own_measured_rate) / margin, -1, 1)
```

`sign` is positive for a teacher-selected option, negative otherwise. At 100 ms,
one spike corresponds to 10 Hz. The margin remains instructive when a choice
is accidentally correct but has weak/noisy separation. Illegal and uninterpreted
options receive zero signals. Neither rule computes a network derivative or
propagates errors through other neurons.

The teacher delivers a synthetic, neuron-specific instruction, not a fitted
policy parameter or an inferred anatomical dopamine assignment. This exact
equation and routing are engineering hypotheses. General three-factor and
neuron-specific teaching frameworks are described in
[Gerstner et al., 2018](https://arxiv.org/abs/1801.05219); this does not validate
the implementation as a natural fly mechanism.

## Curriculum and inference boundary

1. **Higher card:** choose the higher immediate chip-value card from two
   legal candidates, with the correct side balanced by display permutation.
2. **Pair member:** choose a member of the visible pair over a singleton.
   This is a relational scaffold, not a claim about optimal one-card play.
3. **Pair:** choose both members of the unique pair among eight cards.
4. **Best hand:** imitate the highest immediate-scoring visible plain hand;
   card count and membership are learned together.
5. **Small Blind:** train from teacher-played native first-blind states,
   then evaluate the fly's card counts and selections in complete first blinds.

Candidate 2 uses higher-card → best-hand → Small-Blind stages, avoiding the
intermediate objective changes. Earlier scaffolds are not separate simultaneous
tasks in the sensory input; retention is reported, not assumed. The final
objective is native card play.

Demonstrations use current visible cards only. The reference chooses the best
immediate play and never sees future draws. It cleared 21/32 training Small
Blinds without discarding. Labels and reference actions are stored separately
from observations and never enter sensory encoding or evaluation decoding.

**The first-blind benchmark is PLAY-only.** Blind entry is scripted, PLAY is
forced and the fly controls the number and identity of cards. No reference
action is substituted during evaluation. DISCARD timing, blind selection,
shops and full Ante-1 autonomy remain later milestones; a first-blind result
does not establish them.

Both candidates use the qualified repaired mapping, full native FlyWire neural
processing and actual sparse anatomical efficacy modulation. No trained rate
readout replaces the neural simulation. Held-out game roots and neural seeds
are fixed before updates. Candidate 2 uses a new fresh set and does not reuse
candidate 1's final evaluation roots. These are development candidates, not a
magnitude-matched head-to-head experiment.

## Checkpoints and safety

Every 250 frames and stage boundary writes an immutable checkpoint generation.
An atomic manifest binds its state hash, declaration and cursor. Resume restores
efficacy, eligibility, presynaptic means and counters, reproduces stage schedules,
and truncates uncheckpointed log tails. Old trainer checkpoints are not used
under the new rule. Evaluation freezes all local state fields.

The declared stops retain maximum bound occupancy 10% and mean absolute drift
0.5 for the worst actor, with per-step bounded increments and nonnegative
efficacy limits. Bounds/events are recorded rather than concealed. Source,
frozen and unrelated weights are independently checked after training.

The next curriculum is enabled only if fresh teacher-free first-blind clear
rate rises more than five percentage points against its own initial policy,
beats random instruction by five points, improves in at least two of three
learners, and has a positive conditional case-bootstrap lower endpoint. This
qualifies another bounded curriculum stage, not an overnight full-game run.

## Reproduction

Use a new output directory. Candidate 1:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/run_local_teacher_curriculum.py --stage prepare \
  --output-dir artefacts/local-teacher-repeat --steps-per-stage 1000

CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -u scripts/run_local_teacher_curriculum.py --stage train \
  --output-dir artefacts/local-teacher-repeat --heavy

.venv/bin/python scripts/assess_local_teacher_curriculum.py \
  --run-dir artefacts/local-teacher-repeat
```

For candidate 2 use `run_local_rate_teacher.py` and a different directory.
Add `--resume` to a training command to continue from the bound checkpoint.
`status.json` records progress; final `report.json` contains every actor's
before/after card metrics and actual native first-blind outcomes.

## Candidate 1 result

Artifact: `artefacts/local-teacher-curriculum-20261028-v1/`. Completed all 5,000
frames per actor. On the fresh set, real-instruction means changed:

| Task | Initial | Trained |
| --- | ---: | ---: |
| Higher-card accuracy | 53.04% | 54.65% |
| Pair-member accuracy | 51.12% | 53.37% |
| Exact pair selection | 3.85% | 4.81% |
| Exact best-hand imitation | 0.64% | 1.60% |
| Native first-blind clears | 0/192 | 1/192 |

Random instruction also cleared 1/192 first blinds; frozen cleared 0/192.
Thus the small card-metric shifts do not qualify useful native competence or
longer training. All actors and seeds are retained in the report.

The independent audit replayed 19,796 native evaluation actions and verified
that only intended PLAY-associated anatomical synapses changed. Real-instruction
actors potentiated about 1,700 edges and depressed about 9,100; frozen and
unrelated efficacies stayed exact. The latest durable checkpoint matches the
final state. This is a successful implementation/control result with an
unsuccessful first-blind learning milestone.

## Candidate 2 result and development decision

Artifact: `artefacts/local-rate-teacher-20261028-v1/`. Completed all 3,000
frames per actor with rate-target teaching. Fresh-set results:

| Task | Initial real-instruction actor | Trained real-instruction actor | Trained random instruction |
| --- | ---: | ---: | ---: |
| Higher-card accuracy | 49.04% | 50.48% | 50.80% |
| Pair-member accuracy (not a training stage) | 52.88% | 51.60% | 50.48% |
| Exact pair selection | 2.56% | 4.49% | 6.73% |
| Exact best-hand imitation | 0.64% | 0.96% | 1.92% |
| Native first-blind clears | 1/192 | 0/192 | 0/192 |

Frozen outcomes remain exact, including 1/192 first-blind clears. The 192
trajectories use three learner seeds on 64 shared fresh game roots, not 192
independent training seeds. Conditional case intervals for first-blind change
are [0, +0.015625] for candidate 1 and [−0.015625, 0] for candidate 2. Both
fail the declared native advancement criteria. The two candidates have
different fresh sets and budgets; do not interpret this table as a matched
causal comparison between their rules.

Best-hand **scored chips also declined**, despite occasional exact-imitation
gains: candidate 1's mean changed 19.41 → 16.61 chips, candidate 2's
19.04 → 14.06. Learning a smaller card count without learning the matching
card membership is insufficient. The conditional higher-card training task
also stayed near chance: successive 250-frame blocks were
49.1/49.7/48.8/48.8% for candidate 1 and 49.2/48.0/51.3/46.7% for candidate 2.
There is no demonstrated sustained learning curve to extrapolate into a long run.

Candidate 2's independent audit replayed 18,392 native actions; combined with
candidate 1, **38,188 native evaluation actions** were independently checked.
True-instruction actors depressed roughly 10,600–10,800 edges and potentiated
roughly 160–210, with exact frozen/unrelated weights. All safety checks and
checkpoint/final-state comparisons passed. A real-GPU restoration from its
250-frame checkpoint reproduced the exact next choices and every local-update
event; the verification is saved as `gpu-resume-verification.json`.

**All 100 targeted tests pass.** Implementation checks establish locality,
signed updates, no teacher action substitution, native legality and correct
checkpointing. They do not establish useful gameplay learning. Both bounded
runs are complete, and neither was promoted. Long reinforcement training,
DISCARD timing, full Ante-1 and shop training have not been started because
the card-play foundation did not meet its milestone.

The next decision is a substantive local-learning/model-interface redesign,
not another punishment adjustment or automatic duration increase. These data
do not distinguish all possible causes: representation, sparse anatomical
input coverage, recurrent response and the chosen local teaching equations
can each matter. Preserve the negative results and existing V1 baselines;
do not describe the new curriculum implementation as a solved learning task.
