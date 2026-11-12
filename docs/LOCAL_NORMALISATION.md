# Local postsynaptic normalisation experiment

Implemented 31 October 2026 after the repaired native long run stopped at
509,922 decisions per plastic learner. That run improved partial progress but
strongly favoured five-card plays and did not demonstrate useful membership
selection: [long-run review](REPAIRED_LONG_RESULTS.md).

## What changes

The experimental rule gives each MBON a fixed budget for its incoming anatomical
KC connections. It retains the original choice-gated eligibility and native
reinforcement update, then normalises the affected neuron's incoming strengths.
An input can become relatively stronger while other inputs to that same neuron
become weaker. Different neurons and different learners do not share a budget.

The budget is anatomical synapse count times learned efficacy, rather than the
unweighted sum of efficacy multipliers. The original gain, sensory mapping,
motor mapping, 100 ms neural dynamics, reward/punishment parameters and learning
rate remain unchanged. Only existing KC→MBON efficacies learn. Budgets, sorting
indices and treatment assignments are fixed metadata, not fitted parameters.
There are no gradients, teacher actions, external decoder or critic.

`scripts/local_normalisation.py` implements
`local-postsynaptic-strength-budget-v1` as a research-only extension of the
existing rule. Its configuration hash differs from the original rule, so the
standard checkpoint loader rejects cross-rule loading. Existing production
files, historical harness files, source conditions and checkpoints are preserved.
`normalisation_stack.install_normalisation(stack)` integrates the rule with the
existing choice-gated trainer and its strict checkpoints through a research
stack factory. This integration also binds both new research-module file hashes,
so a code change cannot silently resume under the same configuration hash.
`build_normalised_stack(config, condition=...)` preserves an existing research
condition's subset sampler when constructing a complete stack. The standalone
fixed-count study independently binds its rule and runner sources in its own
declaration and checkpoint manifests.

For each postsynaptic neuron j, learner l, anatomical edge weight a, and initial
efficacy w0:

```text
B[l,j] = sum over incoming edges e of a[e] * w0[l,e]
u = original three-factor update, clipped to [0.2, 2.0]
w[l,e] = clip(s[l,j] * u[l,e], 0.2, 2.0)
choose s[l,j] so sum(a[e] * w[l,e]) = B[l,j]
```

Only posts affected by an actual update are normalised. Frozen learners,
untouched posts and unrelated synapses remain exact. The common case uses a
direct multiplicative scale; cases involving efficacy bounds use a monotone
bounded projection. Grouped deterministic reductions avoid unordered scatter
atomics in the normaliser. NumPy and Torch CPU/CUDA paths are supported without
autograd. Events report the net change after normalisation; the inherited
bound-hit counters still describe clipping of the original proposal.

Fixed total strength does not guarantee that no individual edge will reach a
bound, or that card selection will improve. It specifically removes the option
of increasing every incoming connection together. It is an engineering
hypothesis about local competition, not a validated biological mechanism.

## Native learning comparison

`scripts/run_local_normalisation.py` declares and runs three conditions:

1. Original three-factor updates, without normalisation.
2. Identical updates followed by local normalisation.
3. Frozen unit weights, observing the native outcomes without weight updates.

Each condition has three independent learner seeds. The main declaration is
`artefacts/local-normalisation-20261031-v1/declaration.json`: 20,000 trials per
learner, 180,000 native training actions across nine actors. Each actor uses the
real 139,255-neuron FlyWire model and 67,504 plastic KC→MBON edges. All conditions
start fresh with unit efficacies under the repaired action-conditioned mapping.

The task always performs PLAY with exactly **two cards**, using 5% uniform
card-subset exploration. Only PLAY membership posts receive eligibility;
action type and count cannot acquire a shortcut. Training hands contain exactly
one pair, with eight natural opening hands per pair rank: 104 training hands.
Display order changes independently per trial. Native simulator progress is the
only feedback. Pair labels are computed after choices/rewards/updates and used
only to measure learning. One opening play ends an artificial task trial: traces
are cleared without emitting a failure or win. Failure punishment is unchanged
in configuration but is not exercised by this nonterminal task.

Initial, 5k and 20k checkpoints are evaluated greedily on both training hands and
104 held-out hands, with two display/neural repeats per hand. A separately
declared fresh set of 104 hands is used after training to compare **all** trained
actors with their own initial weights. Cases, neural seeds and exploration IDs
are matched across conditions within each replicate; realised choices, rewards
and update magnitudes can diverge as learning proceeds.

The predeclared primary criterion is at least a two-percentage-point fresh pair
selection gain versus both initial weights and the original-rule actors, with
positive gain and advantage in at least two of three learners. Fresh native
progress must also improve. This is an engineering criterion for a follow-up,
not a power calculation or a theorem about possible later learning. Cases and
repeats are not independent training replicates. Final-test seeds stay unused.

The 20k study has completed. Budgets were conserved; fresh pair selection was
4.49% normalised, 4.01% original and 3.53% initial/frozen. The small advantage
has conditional intervals including zero and failed the declared advancement
criteria. [Complete results and limitations](LOCAL_NORMALISATION_RESULTS.md).

## Checkpoints, audit and reproduction

The study checkpoints every 1,000 trials and at milestones. Each immutable NPZ
generation is bound by an atomic manifest to the declared inputs, rule hash,
cursor and training-log length. Resume restores all local state and removes an
uncheckpointed log tail. A missed milestone evaluation is retried after resume.
Budgets and frozen/unrelated weights are checked during the run; the source
saturation/drift limits also apply. The source implementation is never changed
to bypass the historical identity guards.

Prepare without GPU training:

```bash
.venv/bin/python scripts/run_local_normalisation.py \
  --output-dir artefacts/local-normalisation-20261031-v1 --prepare-only
```

Execute or resume the already declared study:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_local_normalisation.py \
  --output-dir artefacts/local-normalisation-20261031-v1 --heavy
```

The completed main study was managed by the transient user service
`flylatro-local-normalisation-20261031-v1.service`, now inactive with a successful
exit. `status.json` records completion and `evaluation-*.json` contains the
learning curves. Rerunning a completed declaration returns the saved report;
preparing a different budget or treatment requires a new directory and
declaration. Never launch two processes against the same active study directory.

Independent CPU analysis checks native progress, rewards, unchanged reinforcement
pulses and diagnostic pair labels for every recorded action. It caches repeated
identical snapshot/action combinations after one independent native replay, and
reports recorded-action counts separately from unique simulator replays. It
also reads the anatomical artifact directly to check the final weighted budgets,
frozen individual choices, unrelated efficacies and accumulated update totals.

```bash
.venv/bin/python scripts/assess_local_normalisation.py \
  --output-dir artefacts/local-normalisation-20261031-v1
```

The real-GPU smoke study is
`artefacts/local-normalisation-20261031-smoke-v2/`. Its independent assessment
checks 258 action records against 47 unique native actions. Normalised weighted
budget error was below 3.4e-8 relative. A separate real-GPU restore reproduced the
next three trials and every local state array exactly; evidence is under
`resume-verification/report.json`.

Regression coverage in `tests/test_local_normalisation.py` includes weighted
budgets, bounded projection, local competition, neuron/learner independence,
unchanged controls, NumPy/Torch CPU/CUDA parity, uniform-gain stress, event
accounting, strict cross-rule checkpoint rejection, exact resume with a torn log
tail, and independent native replay rejection of corrupted pair diagnostics.
The uniform-gain stress drives the original rule to its upper bounds while the
normalised rule remains at its initial strengths. This establishes the intended
mechanical effect; native conditional-choice learning is assessed separately.
