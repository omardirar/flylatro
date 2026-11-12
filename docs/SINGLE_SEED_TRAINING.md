# Long single-seed training

Implemented 1 November 2026. This is a capacity and delayed-generalisation
experiment, using native Red Deck/White Stake games through **Ante 1** (three
blinds), with all action types and card counts chosen by the fly. It does not
yet test completion of Ante 8. Only anatomical KC→MBON efficacies learn.

The default experiment compares normalised local plasticity with the original
local rule, three independent learners per arm, fresh unit efficacies, one
million decisions per learner: **six million training decisions total**.
Rewards, punishment, neural dynamics, repaired PLAY/DISCARD motor mapping and
5% explicit card-subset exploration retain their existing settings. No helper
policy chooses training actions. Normalisation is an experimental stability
mechanism, not an established improvement in card learning.

## Launch

From the repository root:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_single_seed.py \
  --output-dir runs/single-seed-4000-long-20261101-v1 \
  --seed 4000 --steps-per-learner 1000000 --replicates 3 \
  --evaluation-seeds 64 --arms normalised original --heavy
```

The dashboard is enabled. Run the same command to resume. A completed run
returns its existing results; use a new destination for a different seed,
budget or treatment. Add `--prepare-only` instead of `--heavy` to declare the
experiment without training. Do not run two launchers against one destination;
the runner enforces an exclusive process lock.

## What one seed means

`--seed 4000` is the native reset-stream seed; its first Balatro game is
`11HK6BPY`. The simulator normally advances its seed stream at every game over.
`scripts/single_seed.py` captures each row's initial native snapshot and restores
it after **every terminal outcome**, preserving rewards, terminal infos and
done flags. Nonterminal rows keep their current games. This repeats both the
visible opening and hidden future RNG state; it does not redeterminise the deck.
Different actions can still lead to different trajectories.

Distinct learner IDs and decision-dependent neural/exploration seeds remain
active. Their weights and exploration histories are independent despite sharing
the game seed. Terminal reinforcement is applied before the existing trainer
clears each finished learner's eligibility and reinforcement traces.

A CPU reference check established that the seed's first blind can be cleared
using four best-visible plain-hand plays scoring 92, 112, 36 and 60 chips.
This proves only that opening is solvable, not that a complete Ante-1 win is
guaranteed. The reference helper is not installed in training or evaluation.

## Evaluation and interpretation

Frozen greedy evaluations run before training and at 10k, 30k, 100k, 300k and
1m decisions per learner. Every checkpoint is evaluated on the repeated game
and the same 64 disjoint validation seeds. Each learner is explicitly labelled;
shared validation games are not independent training replicates. Initial
evaluations supply the frozen-weight baseline without spending millions of
decisions retraining an unchanged control.

Evaluations use a separate native adapter, never the repeated-seed wrapper.
Weights, local traces, training RNG and in-flight games are restored afterward.
Reports include complete decisions, first-blind clear fraction, mean number of
blinds cleared and Ante-1 win rate. Seed ordering is fixed before training;
final-test seeds remain unused. No score threshold ends training early; safety
gates for silence, nonfinite values, drift and saturation remain enabled.

Improvement on the repeated seed demonstrates seed-specific learning.
Improvement on unseen seeds after a prolonged delay would be evidence
consistent with grokking. Flat results over a finite budget do not prove that
later learning is impossible. This local learning rule has no established
grokking guarantee.

## Recovery and verification

Training uses the existing strict checkpoint/trainer/dashboard implementation
through a process-local factory. The experiment seals source/model/config
hashes and its plan. Periodic checkpoints occur every 10k decisions per learner;
stage boundaries also save a checkpoint. Recovery selects fully hash-verified
generations, writes a new attempt directory and retries incomplete evaluations.
A recorded safety stop prevents automatic continuation. Existing historical
source files and checkpoints are preserved.

Useful files under the chosen output directory:

- `single-seed-plan.json`: declared seed, conditions, budget and evaluation set.
- `single-seed-status.json`: completed stages and current checkpoints.
- `evaluations/<arm>-<stage>.json`: frozen repeated/unseen learning curves.
- `training/<arm>/stage-*/attempt-*/training-dashboard.json`: current dashboard.
- `training/<arm>/stage-*/attempt-*/training-actions.jsonl`: actual actions.

Regression checks cover row-specific native resets, exact repeated hidden
draws, unchanged terminal feedback, checkpoint continuation, evaluation
isolation, different-rule rejection, interrupted-stage recovery, sealed inputs
and idempotent completion. GPU smoke results are under
`artefacts/single-seed-20261101-smoke-v2/`; the 20-step smoke is functional
verification, not evidence of learning.
