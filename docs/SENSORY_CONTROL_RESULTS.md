# Anatomical sensory control and local association test

Completed 6 November 2026 after the
[relational sensory pilot](RELATIONAL_SENSORY_PILOT.md).

Code: `scripts/probe_sensory_control.py`, `scripts/probe_sensory_dose.py`.
Evidence: `artefacts/sensory-control-20261106-v1/` and
`artefacts/sensory-dose-20261106-v1/`.

## Anatomical routing

For card outputs 7 and 5, calculate a fixed two-hop ALPN→KC→MBON score from
positive anatomical contacts. Each KC's target/reference contacts are normalised
by the respective output neuron's total incoming strength; each ALPN's signed
sum is normalised by its total positive contacts onto KCs. Choose the three
highest-scoring and three lowest-scoring ALPNs, plus three seeded random controls.
Selection uses no neural responses, game rewards or labels.

All conditions halve base sensory drive on the union of reserved neurons to
create headroom. A cue adds 75 Hz on three neurons, for 225 Hz additional drive.
Target, reference and random cues have equal additional drive and no clipping.
The original visible opening remains the fixed background.

This is anatomical route selection, not a demonstrated functional calibration.
Indirect recurrent/inhibitory paths and the fixed readout may change its effect.

## Synthetic positive-learning control

Twelve fresh actors compare aligned/reversed cue associations with plastic/frozen
weights, three independent neural replicates each. Only card slots 5 and 7 are
legal choices; PLAY and count one are fixed. Target and reference cues alternate.
Aligned actors receive +0.05 for target cue→slot 7 and reference cue→slot 5;
reversed actors receive +0.05 for the opposite mapping. Wrong choices receive
−0.05. This is explicitly **synthetic correctness reinforcement**, not Balatro
progress or a gameplay result.

Only existing KC→MBON efficacies learn, using the existing selected-card local
rule and postsynaptic strength normalisation. Fast traces reset each trial.
Every actor completes 1,000 trials. Before/after frozen evaluations use sixteen
matched neural seeds per cue, including off and random input controls. These
repeated cues are not independent games.

| Association | Initial accuracy per learner | Final accuracy per learner |
| --- | --- | --- |
| Aligned, plastic | 46.88%, 46.88%, 50.00% | 46.88%, 46.88%, 50.00% |
| Reversed, plastic | 53.13%, 53.13%, 50.00% | 53.13%, 53.13%, 50.00% |
| Aligned, frozen | 46.88%, 46.88%, 50.00% | unchanged |
| Reversed, frozen | 53.13%, 53.13%, 50.00% | unchanged |

The positive control did not pass: neither association improved. Weight updates
were real (maximum final efficacy displacement about 0.00363), while the
evaluated choices remained unchanged. The maximum relative incoming-strength
budget error was 3.02e-8. All efficacies remained finite/bounded and frozen
weights/choices stayed exact. All 12,000 logged synthetic outcomes were checked
for the declared cue mapping and chosen-card correctness.

The frozen three-neuron cues were already poorly selective: mean normalised
target/reference score gap was −1.2647 for both target and reference cues.
Thus failure of learning cannot be attributed solely to the update rule. The
input did not initially deliver a reliable directional choice signal.

## Frozen dose response

A separate probe ranks the same anatomical pathways and compares populations
of 3, 12 and 24 neurons, with 25 or 75 Hz additional drive per neuron. A larger
common reserved population means its baseline differs from the learning test.
Routes remain fixed from anatomy; no neural outcome selects a route or setting.
Eight shared neural seeds measure each cue, without plastic updates.

| Neurons per cue | Added Hz/neuron | Intended aligned choice accuracy |
| ---: | ---: | ---: |
| 3 | 25 | 37.50% |
| 3 | 75 | 43.75% |
| 12 | 25 | 50.00% |
| 12 | 75 | 50.00% |
| 24 | 25 | 50.00% |
| 24 | 75 | 50.00% |

Score gaps did respond, but their changes were not consistently directed or
large enough to yield reliable choices. More drive and more ALPNs did not
automatically improve selectivity. This is an exploratory grid on one background,
not a global impossibility result or validated calibration procedure.

## Interpretation

Pure anatomical reachability is insufficient to choose a useful sensory route
for this fixed readout. The existing background, recurrent circuitry, spike-count
resolution and readout biases can all contribute. The study does not identify
which is dominant or prove that no sensory route can control these outputs.

The next useful distinction is a bypass control: direct stimulation of selected
KC afferents versus ALPN stimulation. If direct KC cues steer the outputs, focus
on sensory routing/competition upstream. If they do not, inspect KC→MBON
modulation and readout calibration. Use measured functional responses on
separate calibration/confirmation seeds before any new full-game training.
Neither the relational encoder nor these anatomical cues are qualified for
long training by the current results.

Two tests pass for signed two-hop path scores and equal-energy, clipping-free
cues. Source production/historical harness files were not changed.
