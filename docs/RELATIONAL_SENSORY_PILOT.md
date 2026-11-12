# Fixed relational sensory input pilot

Implemented 6 November 2026 after the
[frozen pair representation probe](PAIR_REPRESENTATION_RESULTS.md).

`scripts/relational_sensory.py` supplies the count of other visible non-Stone
cards sharing each card's rank, capped at three. Face-down cards contribute no
rank identity. These are fixed engineered game features. In a unique-pair
curriculum they directly identify the pair, so success would be learning to use
a supplied hint, not discovering the poker concept independently.

## Stimulation and controls

Each card slot owns three distinct, randomly selected anatomical ALPN input
neurons. Three projection seeds are fixed before training, one per replicate.
No mapping is selected using outcomes. The existing drive on the 30 reserved
neurons is halved **in all conditions**, creating headroom. The count adds
25 Hz per companion on each of the three neurons, up to 75 Hz. There is no
new clipping. Every other input retains the base encoder's drive.

The modes are:

- `off`: shared headroom change, no companion signal.
- `true`: counts assigned to the visible card's current slot.
- `wrong`: counts cyclically shifted among occupied slots, after display
  permutation; exactly the same additional total drive as `true`.

The off condition is a shared-headroom baseline, not an exact reproduction of
the historical unmodified encoder. The wrong condition controls stimulation
energy and feature count, but can accidentally retain part of the useful hint;
it is not an independent noise source. Channels and their sampling seed are
fixed and untrained. All learned parameters remain anatomical KC→MBON efficacies.

## Task and learning

`scripts/run_relational_sensory.py` compares each mode with plasticity enabled
and disabled. Three independent neural/projection replicates produce 18 actors.
Each plastic actor starts at unit efficacy, uses existing choice-gated local
learning and incoming-strength normalisation, and receives unchanged native
progress reinforcement. PLAY and count two are fixed. Membership remains a
neural choice, with 5% uniform card-subset exploration.

Training samples 104 native one-pair opening hands with equal rank quotas.
Complete card rows are shuffled independently each trial, and executed actions
are translated back to native slots. The pilot's fixed budget is 1,000 trials
per actor, or 18,000 native actions total. Artificial trial resets emit no run
failure. Native snapshots are restored directly in a batch before refreshing
the adapter's observation/reward cache.

Before/after frozen evaluations use the first two cases per rank in each split,
two display/neural repeats each: 26 training and 26 validation hands, 104
decisions per actor per checkpoint. Validation game roots are absent from
training, but this is a previously used research corpus, not fresh confirmation.
All ranks appear in training; no unseen-rank generalisation claim follows.
No output teacher, hidden draws, best-hand actions or reward labels enter the
sensory encoder. The supplied visible companion count is nevertheless explicit
domain knowledge.

## Completed results

The completed study is `artefacts/relational-sensory-20261106-v2/`.
The first launcher attempt stopped before training because the inherited exposure
budget was not divisible by 18; the runner now explicitly declares 18×steps.
The completed declaration binds the corrected code.

All 18 actors completed 1,000 trials, for 18,000 native actions. Pair-selection
counts below pool three learners × 26 cases × two repeats = 156 decisions per
condition/split. There are only three independent trained learners; shared cases
and repeated measurements do not increase that count.

| Input mode, plastic | Training initial → final | Validation initial → final |
| --- | ---: | ---: |
| Off, shared headroom | 2/156 → 3/156 | 8/156 → 10/156 |
| Correct relationships | 5/156 → 7/156 | 6/156 → 7/156 |
| Wrong-slot relationships | 4/156 → 4/156 | 5/156 → 4/156 |

All three corresponding frozen conditions retained their individual choices
exactly. Correct-feature validation gains per learner were [+1, +1, −1] choices
out of 52 each. The no-feature condition gained [+1, 0, +1]. Correct-feature
validation accuracy rose from 3.85% to 4.49%; the shared-headroom baseline rose
from 5.13% to 6.41%. These small descriptive changes do not establish a learning
advantage from the relationship features.

The independent assessor verified **18,000 training and 3,744 evaluation
records** against **2,956 unique native case/action replays**. It also verified
that companion features identify the actual native pair on every replayed hand.
Thus the hint-construction positive control passes. Frozen choices/weights,
source hashes, checkpoint bounds and the runner's normalisation checks pass.
Three new encoder tests and 16 related regressions passed (19 total).

This result limits the original recommendation: merely adding same-rank counts
through this fixed random sensory projection is not a qualified improvement.
The short budget, moderate feature drive, three projections, existing clipped
local rule and reused corpus all limit interpretation. The study does not show
that relational features are generally unhelpful or that longer exposure could
never matter. It does show no reason to promote this encoder to long training.

The next question is whether a visible hint can reliably control the intended
output through a calibrated local pathway. That needs a direct sensory-to-KC-to-
output controllability test and a positive learning control, before further
reward tuning or strategy training. A stronger arbitrary input or learning rate
should not be installed without measuring what it changes.

## Verification and artifacts

The runner seals source/config/corpus hashes before building the neural stack,
stores the sensory root allocation, records native actions and display
permutations, checks weight bounds/drift, checks final weighted strength budgets,
and verifies exact frozen efficacy and individual evaluation choices.
`scripts/assess_relational_sensory.py` independently replays the native
case/action combinations, checking pair diagnostics and progress for every
training/evaluation record. Repeated identical native combinations are cached;
record and unique replay counts are reported separately. Neural updates are
not replayed. This bounded runner writes a final research checkpoint and has
no automatic partial-run resume.

Tests verify companion multiplicity, hidden/Stone exclusion, permutation
equivariance, deterministic encoding, clipping-free headroom and exactly equal
additional drive for true/wrong hints. All three new tests pass.

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_relational_sensory.py \
  --output-dir artefacts/relational-sensory-new --steps 1000 --heavy

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/assess_relational_sensory.py artefacts/relational-sensory-new
```
