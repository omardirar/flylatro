# Single-card information-flow experiment

This read-only experiment traces a change in one visible card's rank or suit
through the existing encoder, Kenyon cells, MBONs and fixed card readout. The
training rule, mappings, reinforcement and saved plastic state stay fixed.

Run both completed pilot arms and every learner independently:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.card_information_cli \
  --pilot-dir artefacts/choice-credit-20261016-pilot-v1 \
  --output-dir artefacts/card-information-new \
  --states 20 --cards-per-state 2 --repeats 4 --heavy
```

Alternatively use `--config PATH --checkpoint PATH`. All learner rows are tested
separately by default; repeat `--learner-index N` to select particular rows.
`--corpus PATH` overrides the configured frozen corpus. The stack must be
FlyWire with MBON-direct output. Package installation also provides the
`flylatro-diagnose-card-information` entry point. Use a fresh output directory.

## Controls and measurements

The runner selects distinct competing PLAY states, spaced through corpus order
without consulting rewards. Within each state it selects visible, legal cards
spaced across occupied slots. Face-down and Stone cards, and cards lacking
ordinary one-hot rank/suit identity, are excluded.

For each selected slot, three inputs are compared: original, rank changed, and
suit changed. Rank/suit categories advance cyclically by one. Each edit changes
exactly two one-hot feature channels. Other cards, enhancements, global state,
deck aggregates, legality bits and every other field remain fixed. These are
isolated visible-input interventions, not altered simulator worlds. They may
be inconsistent with deck summaries or normal hand sorting. No reward or
simulator outcome enters this experiment.

The encoder audit reconstructs the exact unclipped drive and checks that
clipping it reproduces the live encoder bit for bit. It records the affected
input roots, drive before/after clipping, saturation, and the fraction of the
input change retained. Complete projection aliases are distinguished from
changes erased by clipping. Unclipped rates are never passed to the simulator.

Each learner uses its own saved initial and trained efficacy vectors. Original,
rank and suit inputs share the same neural seed for each repeat. Different
repeats use independent derived seeds. The original response is reused across
the selected slots of the same state, with this reuse explicit in the arrays.
The first seed of each evaluated condition is repeated exactly and must reproduce
all recorded neural/readout responses. If encoded inputs are identical, their
matched responses must also be identical. All nine plastic-state fields,
source files and diagnostic implementation hashes are checked at completion.
Eligibility and learning methods are never called.

Recorded populations are:

- KC and full MBON spike rates in Hz over the configured decision window.
- Individual MBONs in card pools, MBONs outside card pools, and MBONs used by no
  fixed motor pool. Each group's root IDs are recorded.
- Mean rate of each fixed card pool, normalized card scores, and card scores
  relative to the mean over legal cards. Illegal-card contrast entries are zero.
  Centering removes common score shifts that cannot change card ranking.

Per-case metrics compare the matched input effect with variability between
independent neural repeats of the original input. A positive response distance
alone is not evidence of a stable representation. A signed cross-repeat effect
estimate also asks whether the change points in a similar direction across seeds:

```text
d[r] = changed_response[r] - original_response[r]
effect_power = mean over repeat pairs r<s and population columns of d[r] * d[s]
noise_power = mean over columns of (Var(original) + Var(changed)) / 2
```

Negative finite-sample effect estimates are retained. Summaries weight states
equally after averaging their selected slots. The reported ratio of mean effect
power to mean trial noise power avoids unstable averaging of ratios near zero
noise. Zero-noise denominators are reported as null. These are descriptive
rate-response measurements, not significance tests or tests of rank/suit
decodability. Four repeats or two card slots are not extra independent states.

The report can identify an input distinction definitively lost to clipping. A
response above or below repeat variability later in the circuit only suggests
where further investigation may help. The measurements exclude spike timing
and do not establish whether a downstream circuit could decode the changed
attribute. Responses outside card pools do not alone prove that replacing
those pools would improve action selection.

## Artifacts and verification

`report.json` records exact cases, edits, seeds, sensory effects, identities,
per-learner results and state-balanced summaries. It updates after each learner;
only `status: completed` denotes success. Failures preserve their error and
completed work. Each learner/weight condition has a hash-bound compressed NPZ
file containing responses with axes `[case, condition, repeat, population]`.
`sensory-activity.npz` stores deterministic ALPN drives with axes
`[case, condition, available ALPN]` and the corresponding root IDs.

Tests cover feature-isolated edits, hidden-card exclusion, complete clipping
versus projection aliasing, known signal/noise estimates, signed negative
estimates, common-mode score cancellation, controlled signals inside versus
outside the card readout, empty populations, exact-repeat and frozen-state
failures, equal state weighting, erased-input controls and CLI artifact hashes.
A small real native checkpoint smoke run also verifies the actual stack.

## Completed results, 2026-10-24

The experiment completed in 221.0 seconds across both pilot arms and all six
learners. It used 20 distinct states, two selected card slots per state, 80
rank/suit edits, four neural repeats and each learner's initial/trained weights.
All 1,200 exact-repeat checks passed. Every plastic-state field and source file
remained unchanged. Initial response arrays agreed exactly across all six
learners. All 13 sensory/neural NPZ files passed their report-bound hash checks.
The full regression passed **588 tests**, including the native CUDA tests.

Artifacts are in `artefacts/card-information-20261024-v1`. `report.json` contains
the complete measurements, `summary.json` is a compact extract, the NPZ files
contain raw responses, and `run.stdout`, `run.stderr` and `regression.stdout`
preserve execution/verification evidence.

| Input intervention | Edits | Mean change retained after clipping | Range retained | Completely erased |
| --- | --- | --- | --- | --- |
| Rank | 40 | 64.94% | 44.74–78.97% | 0 |
| Suit | 40 | 64.53% | 28.08–84.23% | 0 |

Clipping removes about 35% of the absolute drive change on average, but every
tested input distinction survives. The baseline has 24.70% of ALPN drives at
maximum rate; this includes full-strength categorical indicators as well as
collisions and is not itself a clipping-failure fraction. These results rule
out complete encoder erasure for the tested edits. They do not rule out partial
clipping as a contributor to weak responses.

The following ranges cover both edit types and all six trained learners. RMS
effects compare edited and original inputs under matching neural seeds; repeat
variability compares the original input across independent seeds.

| Measured population | Mean matched-input effect RMS | Mean original-input repeat RMS |
| --- | --- | --- |
| KC rates | 3.53–3.66 Hz | 4.17–4.31 Hz |
| Full MBON rates | 5.93–6.17 Hz | 7.38–7.55 Hz |
| Relative legal-card scores | 0.907–1.020 normalized units | 1.129–1.219 normalized units |

The measured input effects are smaller than repeat variability at every one of
these stages. Cross-repeat mean-effect power is also small relative to trial
noise power. For the three choice-gated learners:

| Edit | KC effect/noise power | MBON effect/noise power | Relative card-score effect/noise power |
| --- | --- | --- | --- |
| Rank | -0.0051 to -0.0012 | -0.0319 to -0.0049 | -0.0148 to +0.0019 |
| Suit | -0.0073 to +0.0182 | -0.0142 to +0.0185 | -0.0050 to +0.0673 |

Negative values are retained finite-sample estimates, not negative biological
signal power. Four repeats provide limited precision. In the MBONs unused by
any motor pool, corresponding choice-learner estimates range from -0.0328 to
-0.0058 for rank and -0.0242 to +0.0065 for suit. These summaries do not expose
a stronger stable mean-rate signal waiting in unused outputs.

The highest-scoring legal card also varies across seeds on an unchanged input.
Across the three choice learners, pairwise top-card agreement is 14.2–22.5%,
averaged equally over the 20 states. This is a descriptive measure of preference
stability, not accuracy or a comparison against a random-choice null.

The strongest supported conclusion is that input distinctions reach the
circuit while seed variability dominates these population-rate summaries,
already at the first measured neural population, KCs. The experiment does not
identify an exact anatomical loss point or prove that card identity is
undecodable: a sparse, nonlinear or timing-based representation could escape
these summaries. It also supplies no evidence that replacing only the fixed
readout would solve the problem.

Before another long training run, a reasonable next hypothesis is that the
100 ms neural decision window provides too little averaging for this task.
A declared frozen comparison of integration windows could test that hypothesis
against the present content-response and repeat-variability measurements. This
is a proposed follow-up, not an established fix or a change made by this run.
Keep the choice-credit rule, punishment 0.01 and disabled blind skips fixed
while isolating that factor.

The [100 ms versus 250 ms comparison](CARD_WINDOW_EXPERIMENT.md) has now
completed. Longer integration reduces rate variability, while no choice learner
passes the predeclared rank-and-suit response criterion. The report records a
strong top-card slot preference under the unchanged 100 ms normalization and
retains the 100 ms training setup pending further evidence.
