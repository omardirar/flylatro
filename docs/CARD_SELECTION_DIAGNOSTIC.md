# Frozen card-selection diagnostic

This diagnostic asks whether the fixed sensory/neural/motor path follows card
content when visible hand slots move, and whether trained weights select plays
with higher immediate simulator reward than the same learner's initial weights.
It changes no training rule, mapping, punishment, exploration or saved weights.

Run both arms and every learner of the completed matched pilot:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.card_selection_cli \
  --pilot-dir artefacts/choice-credit-20261016-pilot-v1 \
  --output-dir artefacts/card-selection-new \
  --states 20 --repeats 4 --permutations 4 --heavy
```

For another checkpoint, replace `--pilot-dir` with `--config PATH --checkpoint
PATH`. All learner rows are tested separately by default; repeat
`--learner-index N` to select particular rows. `--corpus PATH` overrides the
configured corpus. The corpus must have hash-verified simulator snapshots. Use
a fresh output directory. An installed command entry point is also available
as `flylatro-diagnose-card-selection`.

The runner selects distinct states offering competing legal card choices and
PLAY, spaced through corpus order without looking at outcomes. The current
96-state corpus provides 20 such states. It checks that each snapshot restores
the recorded observations and constrained legal masks. It exhaustively scores
every legal card subset of size one through five, restoring the same simulator
snapshot and RNG state before each play and the caller's state afterwards.
Scores are the real simulator's scalar immediate reward, including any clear
or win bonus. Every subset, including decoded choices, uses ascending original
slot order. The diagnostic therefore measures membership and count rather than
the strategy of ordering cards inside a play.

Each learner is compared with its own verified initial efficacy vector. Both
weight conditions use identical neural seeds, visible states, permutations and
fixed greedy decoding. Each decision resets neural state through the existing
processor. An exact duplicate call must reproduce the full motor activity and
decoded action. No eligibility, dopamine or learning method is called. Hashes
of all nine plastic-state fields must remain unchanged, and source config and
checkpoint files are checked again at completion.

For each neural response, the ordinary greedy action type is recorded. A second
decode with only PLAY enabled exposes the existing card-count and card-selection
heads even when the ordinary policy chooses DISCARD. This conditional diagnostic
does not replace the game policy and does not claim to improve game results.

For each intervention, every feature of each occupied hand card and its legal
selection bit move together. Padding stays fixed. Unique seeded permutations
move every occupied slot; fewer are used when the small hand has fewer distinct
permutations. The inverse mapping identifies selected original cards. This is
an intervention on the visible input, not a mutation of the simulator's hand.
Mapped selections are evaluated on the original snapshot. Such reordered inputs
may differ from the simulator's ordinary presentation distribution.

`report.json` preserves source identities, the corpus and snapshot hashes, every
subset reward, explicit learner indices, per-trial seeds, card scores, selected
slots, mapped original cards and paired quality changes. It is updated after
each learner. A failed check leaves a report with `status: failed` and an error;
only a fully verified report has `status: completed`.

Interpret these descriptive summaries together:

- **Same card set** measures exact selection of the same original physical
  cards after permutation. **Same feature multiset** also allows indistinguishable
  observed cards. **Same slot set** measures preference for unchanged numeric
  positions. The denominators include only equal-count, non-forced membership
  choices; count agreement and trial counts are separately reported.
- **Card-aligned versus slot-aligned score error** compares the preference
  vectors under the two alignments. Lower card-aligned error is consistent with
  content tracking; lower slot-aligned error is consistent with slot preference.
  Fixed anatomical and sensory asymmetries can produce slot preferences, so this
  is an engineering diagnostic rather than a required biological symmetry.
- **Paired reward change and regret** compare initial and trained selections
  on identical states. Normalized regret and best-subset fraction exclude states
  where all legal plays receive the same reward. In particular, zero-reward
  states do not count as successful quality discrimination. Normalized regret
  also prevents large terminal bonuses from alone determining the comparison.

The corpus is curated calibration material, not held-out gameplay. Four neural
repeats or four permutations are not four independent game states. Immediate
reward is not long-term value, and neither this diagnostic nor its exact-repeat
checks establish learning or a win-rate gain. Scoring results never enter the
encoder, motor decoder, reward implementation or training corpus.

Tests include deliberately content-following and slot-biased processors,
controlled improvement in a selected learner, forced membership, uninformative
reward, legality-bit permutation, exact-repeat and frozen-state failures,
restore-on-error scoring, and a complete real-simulator CLI run with a small
synthetic neural backend and two independent checkpoint learners.

## Completed pilot diagnostic, 2026-10-24

The full run completed in 201.2 seconds. Artifacts are in
`artefacts/card-selection-20261024-v1`: `report.json` contains every trial and
scored subset, `summary.json` is a compact extract, and `run.stdout` and
`run.stderr` preserve execution logs. All 960 exact-repeat checks passed, all
nine plastic-state fields remained unchanged for every learner, and both source
configs and checkpoints retained their original hashes. The full regression
passed **577 tests**, including ten native CUDA tests, using the qualified GCC
14 toolchain; see `regression.stdout`. An earlier regression invocation omitted
that toolchain and hit the system GCC 16/CUDA compiler incompatibility; its log
is preserved as `regression-initial-compiler-mismatch.stdout`.

There were 20 distinct playable corpus states and 4,581 legal subsets in total.
Thirteen states had different rewards for different subsets; seven had no
immediate-reward discrimination. Each learner received four matched neural
repeats and four card permutations per state, under both initial and trained
weights. All initial efficacy vectors and initial results agreed across arms
and learners, as expected with the common controlled neural seeds.

| Rule | Learner | Initial mean immediate reward | Trained mean immediate reward | Paired change | Trained normalized regret on informative states |
| --- | --- | --- | --- | --- | --- |
| Global baseline | 0 | 0.093229 | 0.086542 | -0.006688 | 0.8701 |
| Global baseline | 1 | 0.093229 | 0.084243 | -0.008986 | 0.8551 |
| Global baseline | 2 | 0.093229 | 0.085194 | -0.008035 | 0.8247 |
| Choice gated | 0 | 0.093229 | 0.106979 | +0.013750 | 0.8277 |
| Choice gated | 1 | 0.093229 | 0.094500 | +0.001271 | 0.8154 |
| Choice gated | 2 | 0.093229 | 0.101236 | +0.008007 | 0.8405 |

The choice rule has a small positive change on this conditional PLAY diagnostic
in all three trained learners, averaging +0.007676. The baseline averages
-0.007903. Initial normalized regret is 0.8546; all three choice learners reduce
it, though their remaining average regret is still 0.8154–0.8405 on the
best-to-worst reward range. They choose a best-reward subset in only 2–3 of 52
informative state/repeat trials per learner.

Permutation results show weak exact card tracking. Among equal-count,
non-forced membership comparisons, the choice learners retain the same original
card set in 2.1–7.9% of interventions and the same numeric slot set in
5.2–10.3%. Initial exact card tracking is 1.8%. Slot-aligned card-score error is
lower than card-aligned error for every initial and trained policy. Selections
also change count frequently. Together these measurements show slot asymmetry
and substantial sensitivity to rearranged inputs; they do not localize the
cause to the encoder, recurrent circuit or fixed decoder.

This adds a modest card-quality signal for the choice rule on curated states.
The completed held-out pilot still has one blind clear and zero wins in each
arm and failed its predeclared scale-up threshold. Preserve both findings:
there is no demonstrated gameplay gain to justify a long run yet. The next
investigation should isolate the sensory representation and fixed card readout
while retaining this frozen diagnostic for comparison. The diagnostic provides
no basis for reducing punishment further.

The follow-up [single-card information-flow experiment](CARD_INFORMATION_FLOW.md)
has now completed. Every tested rank/suit edit survives sensory encoding, while
the measured neural-rate effects are smaller than independent-seed variability.
Its report documents the controls, limitations and proposed next hypothesis.
