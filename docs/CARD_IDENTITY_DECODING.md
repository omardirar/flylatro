# Offline card-identity decoding

The completed study finds clear linear card-identity information in KCs and
perfect recovery from sensory inputs. MBON rank decoding does not pass the
primary comparison; MBON suit decoding passes with a small average effect.
This changes the next investigation toward the KC-to-MBON pathway, reward
credit, and motor readout. It does not establish which of those causes the
gameplay failure.

This study asks whether visible card rank and suit can be recovered by a linear
probe from the current sensory drive, KC rates, or MBON rates. The fitted probes
are diagnostic artifacts. They never choose actions, change rewards, enter the
motor mapping, or get saved in an agent checkpoint.

The source is the three frozen choice-credit learners from the October 16 pilot.
Their original 100 ms duration, sensory mapping, motor mapping, failure pulse of
0.01, and disabled blind skips are retained. Source checkpoint loading uses the
normal strict component checks. Initial unit weights are measured once as a
secondary control; all three trained weight sets are measured separately.

## Design

The declaration is written before collecting neural responses or fitting probes.
It records source and implementation hashes, seed schedules, selection records,
regularization candidates, controls, and the primary comparison. The artifact
directory is `artefacts/card-decoding-20261024-v1`.

- Collect real simulator observations using seeded uniform legal navigation,
  without using rewards, action quality, or hand strength to select states.
- Select one PLAY state with eight legal, visible cards per actual game run:
  192 runs for fitting, 64 for tuning, and 64 held out for evaluation.
- Keep actual game identities, original observation hashes, and root reset
  streams disjoint across partitions. Use validation-range root seeds starting
  at 20,020,000, 20,020,100, and 20,020,200. The locked final-test stream is unused.
- Independently shuffle the eight complete visible card rows in each selected
  observation. This removes the ordinary association between sorted rank and
  slot. It preserves each hand's multiset. It is an input intervention, not a
  mutation of simulator state or a gameplay trajectory.
- Record four independent neural repeats per observation. Use matched seeds
  across weight sets and duplicate the first repeat exactly. Average repeats
  within each run before fitting. Cards, slots, and repeats are not additional
  independent game runs.
- Fit slot-specific multi-output ridge classifiers for rank (13 classes) and
  suit (4 classes). Train-only standardization drops constant training features.
  A shared feature matrix predicts the eight slots with separate output weights.
  Choose the ridge penalty from 0.01, 0.1, 1, and 10 using tuning balanced
  accuracy. Do not refit on tuning data. Evaluate held-out data once.

Inputs are the live encoder's clipped ALPN drive; KC and MBON inputs are raw
window-average rates in Hz. The direct visible rank/suit one-hots provide a
positive control for labels, partitioning, and the linear fitting machinery.
The positive control must achieve at least 98% accuracy on both tasks for the
comparison to pass.

## Controls and statistics

Report ordinary accuracy, class-balanced accuracy, class support, and per-slot
accuracy. The theoretical uniform baselines are 1/13 for rank and 1/4 for suit.
A second baseline predicts each slot's most frequent training label. It never
sees tuning or evaluation labels when choosing predictions.

For each of 199 null fits, shuffle whole eight-card training-label blocks across
runs and repeat the same tuning procedure. All populations and learners use the
same shuffle schedule for a given task. Evaluation labels remain fixed. The
reported permutation p-value includes the observed arrangement in its denominator.
Per-probe p-values are exploratory. The six primary tests cover rank and suit at
sensory, KC, and MBON stages, with neural statistics averaged across the three
trained learners. Apply Holm correction across these six tests. A primary result
must pass the correction and exceed both uniform chance and the slot-majority
baseline. Initial-weight and direct-input results are controls, outside this
primary family.

The 95% bootstrap interval for excess over the slot-majority baseline resamples
whole evaluation runs, keeping all eight cards together. It describes uncertainty
over these sampled contexts conditional on a fitted probe; it is not an interval
over a large population of independently trained flies.

## Run and artifacts

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.card_decoding_cli \
  --output-dir artefacts/card-decoding-20261024-v2 --heavy
```

The output directory must be new. `--prepare-only` collects observations and
writes the declaration without building the neural stack. Counts and repeats
can be reduced for development, but those runs do not implement the full design.

`declaration.json` contains the predeclared methods and case selection.
The three corpus files retain observations and run provenance. `labels-and-inputs.npz`
contains the diagnostic labels and direct/sensory inputs. Each response file
contains raw repeats, neuron root IDs, and neural seeds. Each probe file contains
held-out predictions and null scores; `report.json` binds files to SHA-256 hashes
and records exact-repeat checks, source identities, and hashes of all nine plastic
state fields. `assessment.json` contains the six primary comparisons and positive
control result. No fitted classifier is loaded into the agent.

## Results, October 24

All values below are held-out class-balanced accuracy. Neural rows average the
three trained learners; they share the same 64 evaluation runs. All six primary
tests use Holm correction. The positive control achieves 100% ordinary and
balanced accuracy on both tasks.

| Stage or baseline | Rank | Suit |
| --- | ---: | ---: |
| Uniform chance | 7.69% | 25.00% |
| Training-only slot majority | 9.16% | 25.31% |
| Direct identity one-hots | 100.00% | 100.00% |
| Encoded ALPN drive | 100.00% | 100.00% |
| KC rates, initial weights | 31.57% | 66.64% |
| KC rates, trained mean | 32.60% | 66.08% |
| MBON rates, initial weights | 8.79% | 30.54% |
| MBON rates, trained mean | 8.65% | 28.77% |

Both sensory tasks, both KC tasks, and mean MBON suit decoding pass the primary
screen (Holm-adjusted p = 0.03). MBON rank does not (adjusted p = 0.14) and falls
below the slot-majority baseline. Individual trained KC results are consistent:
rank 32.77%, 30.56%, and 34.46%; suit 66.61%, 66.24%, and 65.39%. Individual
trained MBON suit results vary: 30.38%, 24.75%, and 31.18%. The aggregate MBON suit
finding therefore should not be described as a robust improvement in every fly.
The 199-shuffle design resolves p-values no smaller than 0.005 before correction.

These results do not show a training improvement: KC accuracy is similar to its
initial-weight control, and mean MBON suit accuracy is lower. The earlier tiny
population-distance estimates did not imply that KCs lacked identity; an offline
linear readout can recover it on held-out runs. Perfect sensory decoding also
means these data do not justify changing the sensory encoder first.

The full run passed 1,280 exact neural duplicate checks. All nine plastic state
fields and source files remained unchanged. File hashes and all 20 reported
task metrics were independently checked from saved predictions; three corpora
passed their identity checks. All 605 tests pass, including native CUDA tests.
The run and regression logs, `audit.json`, and full per-probe metrics are saved
with the experiment.

Keep the 100 ms training configuration, failure pulse 0.01, and blind skipping
disabled. Before longer training, inspect how identity-bearing KCs connect to
MBONs and whether choice-gated eligibility/reward updates reach the relevant
fixed motor pools. A supervised diagnostic classifier should not become the
agent's policy as a consequence of this result.

The completed [KC-to-MBON and credit audit](CARD_CREDIT_AUDIT.md) follows this
recommendation. It finds that identity survives fresh choice-gated eligibility,
while shared card/action/count MBONs admit substantial credit to unchosen card
outputs. It recommends a short matched test of separate motor populations.

## Interpretation

Above-control decoding establishes that a linear readout can recover identity
under this study's conditions. It does not establish that the existing motor
interface uses it, that dopamine associates it with useful choices, or that the
agent can win games. Weak decoding can reflect insufficient fitting contexts,
inappropriate model class, noise, or the chosen measurement window. It cannot
prove identity is absent: nonlinear and spike-timing codes remain untested.

Shuffling visible inputs can disagree with other observation fields that encode
ordering or summaries. These results apply to that intervention and the selected
eight-card PLAY contexts, not every phase or hand size. All three trained learners
come from the same pilot design; held-out runs are not extra training replicates.
