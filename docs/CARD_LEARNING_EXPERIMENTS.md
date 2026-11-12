# Card learning experiments

`scripts/run_card_learning.py` implements a replay audit, explicit membership
sampling, a one-pair curriculum, frozen pair benchmarking and four-condition
pilots. [Learning roadmap](LEARNING_ROADMAP.md) tracks later work. Separate
PLAY/DISCARD heads and reward/clipping changes use separately versioned
implementations rather than changing historical pilot declarations.

Completed follow-ups are documented in [Pair learning results](PAIR_LEARNING_RESULTS.md)
and [Card-credit direction and clipping](CARD_CREDIT_DIRECTION_AUDIT.md).
The independent [matched reference-scaling probe](CARD_REFERENCE_SCALING.md)
compares reduced clipping with calibration update magnitude matched; it keeps
feedback fixed and preserves this harness's source identity.
The independent [native-score reward-contrast probe](CARD_REWARD_CONTRAST.md)
instead holds references/exploration fixed and changes only its experimental
feedback transform, with no pair labels in reinforcement.
The [PLAY/DISCARD interface fix](PLAY_DISCARD_INTERFACE.md) adds separate
card/count preferences and action-specific credit in new mapping artifacts.
Historical pilots and checkpoints retain their original shared interpretation.

The harness reuses the existing trainer, simulator, mappings and checkpoints
through process-local stack factories. It does not edit production files on
disk. This preserved the long run's `src/flylatro` hash guard and keeps the
completed pair pilot's source identity unchanged.

## CPU diagnostics and preparation

Audit complete frozen Ante-1 validation logs; repeat `--transitions` for other
learners/stages. Before/after hashes, terminal hashes, rewards and episode
completion must match. Poker diagnostics cover only plain first-Small-Blind
states. Native counterfactual checks default to 20 opening states per input.

```bash
.venv/bin/python scripts/run_card_learning.py audit \
  --transitions runs/separated-long-20261024T182315Z/evaluation-020000-learner-0/transitions.jsonl \
  --output-dir artefacts/card-learning-audit-20k-learner0
```

Build 104 training and 52 validation opening snapshots with exactly one pair.
Game-root streams are disjoint and pair-rank quotas equal. Display positions
are shuffled per trial; suit combinations are natural draws, not balanced
quotas. This is a task curriculum, separate from reward-free motor calibration.

```bash
.venv/bin/python scripts/run_card_learning.py build-pairs \
  --output artefacts/card-learning-pairs-v1.json
```

Prepare either pilot without launching GPU training. All destinations must be
new. Defaults use three learners per condition and 5% subset exploration.

```bash
.venv/bin/python scripts/run_card_learning.py pilot \
  --config artefacts/separated-motor-20261024-pilot-v2/configs/separated.toml \
  --output-dir runs/card-subset-pilot-v1 \
  --steps-per-learner 20000 --replicates 3 \
  --episodes 256 --batch-size 4 --prepare-only

.venv/bin/python scripts/run_card_learning.py pilot \
  --config artefacts/separated-motor-20261024-pilot-v2/configs/separated.toml \
  --pair-corpus artefacts/card-learning-pairs-v1.json \
  --output-dir runs/card-pair-pilot-v1 \
  --steps-per-learner 5000 --replicates 3 \
  --episodes 256 --batch-size 4 --prepare-only
```

Four conditions are declared: `topk-plastic`, `topk-frozen`, `subset-plastic`,
`subset-frozen`. Mappings, rewards, roots, base neural seed and exposure budget
are shared. Frozen controls choose their own actions; these are not the primary
protocol's matched-action replay controls. The normal pilot totals 240k decisions
and the pair pilot 60k; neither budget is a scientific power calculation.

## Executing a prepared pilot

Execute one prepared pilot serially with normal safety gates. The harness refuses changed
source, research modules, conditions, configs and pair corpora. It does not
automatically restart partial pilot orchestration.

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_card_learning.py run \
  --pilot-dir runs/card-subset-pilot-v1 --heavy
```

Use `--pilot-dir runs/card-pair-pilot-v1` for the capacity test. Each learner gets
greedy full-game evaluation against its own initial weights on 256 validation
roots at offset 70000. Pair pilots additionally benchmark held-out pairs.
Full-game transfer restores free action/count heads; forced PLAY/two cards
applies only inside the pair curriculum and pair benchmark. `results.json`
contains descriptive comparisons; nothing is automatically promoted and
final-test seeds remain unused. Confirm selected conditions on a fresh block.

## Credit, task boundaries and resume

Sampling uses an independent stateless learner/decision seed, preserving the
existing type/count draws. It samples a legal subset of the chosen size without
replacement. Consumable selections and targets remain unchanged. Greedy
evaluation bypasses sampling; credit uses the actual executed cards.

The pair task restores a valid opening snapshot each decision, shuffles display
order and translates selections back before native scoring. PLAY/count two
are fixed; only competing card pools receive new eligibility. Native progress
is the only learning feedback. Pair recognition is diagnostic info, never a
learning target. Task boundaries clear traces without failure or win pulses.
Generic dashboard completed-run/game-over counters are not pair accuracy; use
the dedicated benchmark report.

Sampler, task, corpus and implementation identities are checkpoint components.
Resume via the condition worker, with the compiler environment above for real
FlyWire execution:

```bash
.venv/bin/python scripts/run_card_learning.py train \
  --condition runs/card-subset-pilot-v1/conditions/subset-plastic/condition.json \
  --run-dir runs/card-subset-pilot-v1/subset-plastic \
  --resume runs/card-subset-pilot-v1/subset-plastic/plastic-checkpoint-interrupted.pkl \
  --dashboard --no-tensorboard --heavy
```

Probability, task, corpus and the declared budget cannot change on resume.
An explicit frozen pair check is available:

```bash
.venv/bin/python scripts/run_card_learning.py pair-evaluate \
  --condition runs/card-pair-pilot-v1/conditions/subset-plastic/condition.json \
  --checkpoint runs/card-pair-pilot-v1/subset-plastic/plastic-checkpoint-final.pkl \
  --learner-index 0 --repeats 4 \
  --output artefacts/card-pair-learner0.json --heavy
```

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/test_card_learning_harness.py -q
```

Tests exercise native scores/snapshots when installed, synthetic CPU training,
exact resume, independent frozen controls, greedy evaluation, slot translation,
executed-card credit and full-game transfer. The real pair pilot has completed
without demonstrating pair learning; the normal-game pilot remains pending.

## Prepared local artifacts, 24 October 2026

`artefacts/card-learning-20261024-v1/` contains the generated 156-hand pair
corpus, `subset-pilot/` (20k per learner) and `pair-pilot/` (5k per learner).
The pair pilot has completed; preserve its outputs rather than restarting it.
The normal-game pilot remains prepared, pending the credit/feedback follow-ups
in the roadmap. All declarations are bound to harness/source files and refuse
later changes.

The same directory contains a completed four-condition synthetic-CPU
`cpu-factorial-smoke/` and a native replay `audit-20k/`. The replay verified all
5,736 transitions from the three 20k evaluations and reproduced 508 missed
winning plays, 308 discards with a clear available, and 515 failures with
discards unused. The focused new and related regression suites passed 84 tests.
