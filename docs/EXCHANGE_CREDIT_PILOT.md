# Exchange exploration and signed local credit

Implemented 6 November 2026 following the
[exploration diagnostic](LOCAL_LEARNING_NEXT_STEPS.md). This research runner
uses the repaired full FlyWire neural model with fresh anatomical KC→MBON
efficacies. There is no learned external actor, critic or decoder.

## Comparison

All conditions perform one native PLAY of five cards on the opening of seed
4000, then restore that opening. Three independent neural seeds per condition
give 15 actors, including the frozen controls. All learning conditions use the
same local incoming-strength normalisation and original learning rate.

| Condition | Exploration | Eligibility and reinforcement |
| --- | --- | --- |
| subset-selected | 5% random five-card subset | Existing selected-card coactivity and native pulse |
| swap-selected | 5% single-card exchange | Existing selected-card coactivity and native pulse |
| subset-signed-centred | 5% random five-card subset | Signed changed-card coactivity and centred pulse |
| swap-signed-centred | 5% single-card exchange | Signed changed-card coactivity and centred pulse |
| frozen | 5% random subset | Zero weight updates |

The signed rule labels an entering card +1 and a leaving card −1. Cards retained
in both the greedy and actual selection receive zero new eligibility. The tag
multiplies the existing clipped pre/post coactivity at the corresponding
anatomical card-output synapses. Illegal/padding slots, count/type outputs and
other motor heads receive no new credit. Eligibility is cleared after each
isolated trial.

Each lane maintains a scalar recent-reward trace, initialised from its first
outcome. The signed conditions use `native_pulse - previous_trace`; after the
update, `trace += 0.01 * (native_pulse - trace)`. Positive/negative values feed
the existing reinforcement channels. The trace is local bookkeeping stored
with the checkpoint, not a learned external value function. The first signed
update is zero. Native feedback supplies no best-hand labels.

The experiment changes signed eligibility and reward centring as a combined
treatment. It can distinguish the sampler effect and the combined credit effect;
it cannot separately attribute signed eligibility versus centring. It is an
engineering hypothesis, not a derived unbiased policy-gradient estimator.

## Evaluation and limits

The fixed budget is 2,000 trials per actor: 30,000 native training actions.
Frozen greedy evaluations before/after use 16 matched neural repeats of the
training opening and 16 distinct unseen game openings per actor. Card count
remains five. This first stage deliberately tests a fixed display; shuffled
positions and multi-state training are later tests if useful learning appears.
Opening score on unseen games is capped at 300 chips by native progress.

The repeated seed is the known diagnostic case, not an independently selected
confirmation set. No run-success, discard or Ante-1 gameplay improvement can be
inferred from this task. No run-failure punishment is generated at the artificial
trial boundary. The historical long-run code and checkpoints are preserved.

The runner writes the declaration before neural training, logs every sampled
set/native pulse/modulator/reward trace, checks bound/drift safety every 250
trials, and checks final strength budgets and exact frozen efficacy/choices.
This bounded pilot writes a final checkpoint for analysis; it does not implement
automatic interrupted-training resume or accept historical checkpoints under
the new rule. Output directories must be new.

`tests/test_exchange_credit.py` checks legal single exchanges, signed entry/exit
tags, unchanged-set zero credit and opposite weight changes with conserved local
budgets. Together with normalisation regressions, 16 checks pass.

`scripts/assess_exchange_credit.py` independently enumerates all 56 native
opening actions to verify every logged training reward. It reconstructs all
baseline/modulator calculations and verifies exchange sizes, final bounds,
checkpoint identity and exact frozen evaluation choices. It does not replay
the neural weight updates.

## Completed pilot results

`artefacts/exchange-credit-20261106-v1/` completed 30,000 native training
actions in about 747 seconds including the final evaluation. All 15 actors
completed 2,000 trials. The independent assessor checked every recorded native
reward and baseline/modulator calculation against an exhaustive native bank of
56 actions. Frozen individual evaluation scores and weights remained exact.
Maximum strength-budget relative error was 5.39e-8.

| Condition | Repeated-opening mean chips, initial → final | Unseen-opening mean chip change |
| --- | ---: | ---: |
| Whole subset, selected credit | 57.54 → 56.60 | +1.04 |
| Single exchange, selected credit | 57.54 → 57.54 | +1.04 |
| Whole subset, signed/centred | 57.54 → 57.54 | +1.33 |
| Single exchange, signed/centred | 57.54 → 57.54 | +1.33 |
| Frozen | 57.54 → 57.54 | 0 |

The signed treatments' +1.33-chip unseen mean comes entirely from learner 0
gaining four chips averaged over its 16 openings; the other two learners are
unchanged. These small descriptive differences establish no useful learning
advantage. The training opening's best available score is 92, so unchanged
57.54 does not demonstrate mastery.

Every actor encountered 92-chip plays during training (18–33 occurrences).
The signed actors experienced only 96–99 changed-set trials each because
exploration occurs on 5% of trials. Cumulative update L1 was about 102–105 for
selected credit, 1.52–1.64 for whole-subset signed credit, and 0.67–0.71 for
exchange signed credit. Nominal learning rates match; realised updates do not.
This pilot therefore cannot distinguish weak update exposure from a poor
update direction. It does not qualify the new rule for a longer gameplay run.

Next measure actual before/after neural preference changes for known beneficial
exchanges, including independent neural seeds, using matched bounded update
magnitudes. A larger arbitrary learning rate would confound that diagnosis.
The fixed display, single training hand, compound signed/centring treatment
and small evaluation set remain limitations.

## Commands

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_exchange_credit.py \
  --output-dir artefacts/exchange-credit-new --steps 2000 --heavy

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/assess_exchange_credit.py artefacts/exchange-credit-new
```
