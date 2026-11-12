# Matched pilot of separated motor populations

The completed pilot removes the targeted credit cross-talk but fails its
predeclared blind-learning screen. Separated trained learners clear 0, 2, and 2
blinds on 256 held-out games each, compared with 2 for that layout's initial
policy. Both arms remain numerically healthy. The result leaves delayed learning
under a larger budget unresolved; the conservative screen is not evidence that
longer training cannot work.

This tests the cross-head credit interference identified by the
[KC-to-MBON credit audit](CARD_CREDIT_AUDIT.md). Card outputs in the previous
interface share MBON neurons with action-type and card-count outputs. As a
result, choosing an action or count can admit new eligibility on unchosen card
outputs. The candidate assigns distinct neurons to these three core groups
and excludes card neurons from every other motor group.

The experiment is exploratory and bounded: three fresh learners per arm, 10,000
environment decisions per learner, 256 reserved validation games per frozen
evaluation, and a predeclared screen for further work. It does not change the
default training configuration or reuse weights trained under another mapping.

## Calibration and interface

Both arms use the same newly recorded neural responses to the existing
96-state reward-free calibration corpus, with four independent neural repeats,
unit initial efficacies, and the original 100 ms decision window. The first
repeat is duplicated exactly for all 96 states. No outcome, hand-quality,
identity-probe label, reward, or recommended action enters mapping selection.

The baseline retains the existing pool memberships and receives fresh fixed
median/IQR normalization. The separated candidate uses the same normalization
procedure on its own new pools. Therefore the comparison changes pool layout
under a common calibration procedure; the baseline normalization is freshly
estimated rather than copied from the older calibration artifact.

The candidate uses width-two pools and 58 distinct MBONs across 13 action-type,
6 card-count, and 10 card-slot pools. Card MBONs do not occur in any noncard pool.
The six contextual-slot pools retain the existing routing shared among mutually
exclusive shop, pack, joker, and consumable heads. They can reuse noncard MBONs,
so this experiment does not eliminate every form of cross-head sharing.

The allocator uses eligible candidates from each group's own legality contexts,
capacity-preserving matching, refinement of neuron pairs, and exchanges between
groups scored by pooled neural covariance. It uses up to 32 seeded search starts,
four exchange/pair-refinement rounds, and up to 3,000 annealing proposals per
start. The objective contains only neural covariance. Every active selected
candidate must pass the original reward-free quality
thresholds, including context coverage, width, signal diversity, silence, and
normalized option range. No threshold is weakened to obtain a separated mapping.
The first passing candidate is persisted and frozen before either arm trains.

Calibration masks apply the experiment's disabled blind-skip constraint. The
older saved corpus predates that constraint and marks skipping as legal. The
first calibration attempt used those older masks and did not qualify a separated
mapping; no pilot training occurred in that attempt. The final calibration
corrects legality for both arms. The still-present SKIP_BLIND contract pool uses
two quiet unused neurons in the candidate, remains structurally disjoint, and
has no responsiveness claim because it cannot be selected. All active option
quality gates remain unchanged.

Keeping the old card pools fixed was tested during allocator development but
left replacement action outputs too redundant to pass the existing diversity
gate. The final allocator can reassign card, action, and count populations
jointly. Mapping choice is therefore a change in the initial policy as well as
in the credit assignment. Initial-weight evaluation is necessary to distinguish
those effects from learning.

## Matched training and evaluation

The declaration is written before calibration and records implementation/source
hashes, seeds, budgets, methods, and the scale-up screen. Calibration reports
and both mapping/configuration hashes are bound into a separate training plan
before the first training job. Source and implementation hashes are checked
before jobs and at assessment.

Both arms keep:

- Choice-gated three-factor plasticity with learning rate 0.002.
- Failure punishment 0.01 and blind skipping disabled.
- Original anatomy, sensory encoder, synaptic gain, 100 ms duration, routing,
  exploration settings, and synthetic outcome-to-dopamine mapping.
- Fresh unit weights, independent learner state, matched training seeds
  4000/4001/4002, and equal exposure budgets.

Each arm qualifies exact checkpoint resume before its full run. The qualification
uses a separate fresh probe and does not seed or resume the pilot learners.
Ordinary checkpoint component checks continue to reject a different mapping or
credit-assignment identity.

Frozen evaluations use validation roots 20,030,000 through 20,030,255, with
Ante 1 as the target. Each trained learner chooses its own actions. One initial
unit-weight control per arm uses the same validation games and neural seed
schedule; those initial efficacies are identical across that arm's learner rows.
No training actions are replayed and no locked final-test seeds are used.

New telemetry records the fraction of admitted card-targeting edges outside
explicitly selected card posts. This measures gate admission, not historical
reward allocation or coactivity-weighted update magnitude. The separated arm
must record exactly zero admission aliasing throughout training.

## Predeclared screen

Further training requires all of the following:

1. Both arms complete their equal budgets without a safety stop.
2. The separated arm clears more held-out blinds in a majority of matched
   learners and in total.
3. Its trained policies have positive blind-clear gains over their own initial
   policy and larger gains than baseline in a majority of learners and in total.
4. Its total trained wins do not decrease.

An improved initial policy alone cannot pass this screen. Thousands of evaluation
episodes are repeated measurements of three trained learners, not thousands of
independent training replicates. A passing screen supports a larger experiment;
it does not establish a general learning improvement.

## Run

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m flylatro.learning.separated_pilot \
  --output-dir artefacts/separated-motor-20261024-pilot-v3 --heavy
```

The output directory must be new. `--prepare-only` writes the declaration without
collecting neural responses or training. Development counts can be specified
with `--steps-per-learner`, `--replicates`, and `--evaluation-episodes`.

The full run is saved under `artefacts/separated-motor-20261024-pilot-v2`.
`suite.json` identifies the running or completed job; each job has captured
stdout/stderr. Training folders include dashboard snapshots, metrics, action
and reinforcement audits, and checkpoints. The calibration report binds the
reward-free data and both qualified mappings. `assessment.json` reports matched
trained outcomes, initial-policy controls, learning gains, and the predeclared
screen. `checkpoint-audit.json` checks durable logs against every saved training
checkpoint.

## Completed results, October 24

Both arms finish 30,000 environment decisions: exactly 10,000 per independent
learner, using training seeds 4000/4001/4002. All 60,000 decisions are accounted
for in durable action, reinforcement, and metric logs. No safety stop occurs.
There are zero external trainable parameters in either arm.

The separated mapping passes every unchanged reward-free gate on the first
search start after applying the disabled-skip constraint. Its structural core
uses 58 distinct MBONs, including two assigned to the inactive skip contract
pool. It has 56 responsive neurons across active card, action, and count options.
The original context-slot routing is retained. Calibration passes 96 exact
repeat checks, and both arms pass exact checkpoint-resume qualification.

The mechanism intervention succeeds: mean per-step card-admission aliasing
across learners falls from 17.81% in baseline training to exactly zero throughout
separated training. This statistic concerns admission of new eligibility, not
the amount of historical reward or weight change.

| Frozen policy | Baseline blind clears | Separated blind clears |
| --- | ---: | ---: |
| Initial weights, 256 games | 0 | 2 |
| Trained learner 0, 256 games | 1 | 0 |
| Trained learner 1, 256 games | 0 | 2 |
| Trained learner 2, 256 games | 0 | 2 |
| Trained total, 768 games | 1 | 4 |

Neither arm clears Ante 1 in evaluation. All evaluations keep skips disabled.
The initial policy is evaluated once per layout because initial weights and
the frozen evaluation RNG schedule are identical across that layout's learner
rows. That control is reused for comparison, not counted as three independent
sets of initial games.

The separated layout has more trained blind clears than baseline in two of
three learners and in total. Its own blind-clear learning gains are -2, 0, and
0, whereas baseline gains are +1, 0, and 0. Thus zero separated learners satisfy
the required larger positive gain over their own initial policy, and the
predeclared scale-up signal is **false**.

Mean returns are a secondary descriptive result. Baseline initial return is
0.19779 and its trained mean is 0.20091; separated initial return is 0.21414 and
its trained mean is 0.22068. One separated learner improves return noticeably
while keeping the same blind-clear count. These secondary values do not change
the predeclared screen or establish a general learning improvement.

The final audit verifies 22 training checkpoints, equal per-learner exposure,
unit initial efficacies, all 20,000 vector-step metric rows, eight frozen
evaluations totaling 2,048 episodes, source/config/mapping hashes, and unchanged
implementation identity. All 615 tests pass, including native CUDA checks.
The completed `audit.json`, `assessment.json`, `checkpoint-audit.json`, regression
logs, and both qualified mapping artifacts are saved with the experiment.

## Meaning for longer training

The predeclared screen is a conservative decision rule. Failing it means this
pilot does not support an improvement claim or automatic promotion of the new
layout. It does not demonstrate that learning cannot emerge after 10,000
decisions per learner. Stable weights and removal of cross-talk make a bounded
longer exploratory run a defensible next experiment.

A longer follow-up should declare its larger budget explicitly, retain the
current sensory/learning/reinforcement settings, save intermediate checkpoints,
and measure a learning curve on reserved games against its own initial policy.
Comparing layouts would additionally require equal longer exposure in both arms.
Expectations should remain experimental: a better starting policy and clean
credit assignment do not by themselves guarantee useful delayed learning.

The candidate remains a separately versioned experimental artifact. The default
training configuration and source checkpoints have not been replaced.
