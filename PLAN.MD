# Flylatro plastic-brain V1

## Authority and research question

This is the authoritative plan for Flylatro. The superseded fixed-reservoir/PPO
plan is preserved at `docs/archive/RESERVOIR_V1_PLAN.md` and its implementation
is a legacy comparison only.

Plastic-brain V1 asks whether reward-modulated plasticity at anatomically
identified Kenyon-cell to mushroom-body-output-neuron (KC->MBON) connections in
an adult FlyWire-derived network can acquire useful Balatro behaviour. A second
experiment asks whether those local changes can alter downstream whole-brain
and descending-neuron activity enough to change fixed action outputs.

Flylatro is connectome-derived and biologically motivated. It is not a faithful
biophysical model. Synthetic game sensing, simplified LIF dynamics, aggregate
neuron-pair weights, eligibility traces, synthetic scalar reinforcement, and fixed game
motor mappings are engineering abstractions and must be labelled as such.

## Primary architecture

```text
observable Balatro state
  -> fixed, seeded sensory mapping
  -> fixed FlyWire topology and neural dynamics
  -> Kenyon-cell activity
  -> plastic KC->MBON edge efficacies
  -> MBON-direct or downstream descending activity
  -> fixed, zero-parameter motor interface + legal mask
  -> Balatro action and outcome
  -> fixed outcome-to-synthetic-reinforcement transform
  -> eligibility-gated KC->MBON update
```

The only persistent trainable parameters in this path are efficacy values for
existing KC->MBON edges. Membrane, conductance, spike, refractory, eligibility,
and internal reinforcement traces are dynamic state, not an external policy. There is no PPO
actor, trainable decoder, GAE, policy gradient, or external learned critic in
the primary experiment.

## Biological populations and data contract

The pinned source is the FlyWire FAFB materialization 783 / Codex static export.
The artifact builder must derive populations from documented annotation fields,
never informal substring guesses over community labels:

| Population | V783 rule | Use |
|---|---|---|
| KC | `class == Kenyon_Cell` | eligible presynaptic population |
| MBON | `class == MBON` | plastic postsynaptic and Mode A output population |
| DAN | `class == DAN` | census and reinforcement routing |
| PAM | DAN whose resolved `primary_type` begins `PAM` | appetitive channel |
| PPL1 | DAN whose resolved `primary_type` begins `PPL1` | aversive channel |
| AL projection neurons | `class == ALPN` | preferred synthetic input route |
| APL / DPM | exact resolved-type family where present | census / future recurrence |
| descending | `super_class == descending` | Mode B output population |

The builder records exact indices, root IDs, resolved types, rules, counts,
source hashes, and all KC->MBON anatomical edges. A full-data census mismatch
must fail loudly. Compartment labels may be parsed only from curated resolved
MBON/DAN type names and must retain the unmodified source type; unknown values
remain unknown. Neuron-pair edges do not establish synapse-level compartment
location, so V1 must not claim they do.

The builder also records unresolved neurotransmitter neurons, connections and
synapses dropped for invalid endpoints or unresolved sign, their source-synapse
fraction, and missing coordinates. Only the published exact 5,177-KC census is
a hard biological count here; other named populations are fully reported and
hashed unless a reliable reference justifies a hard gate. KC->MBON diagnostics
report minimum pair-synapse thresholds 1/2/5/10. The primary default remains
`minimum_synapse_count = 1`; changing it is an explicit sensitivity experiment
and changes topology identity.

## Fixed sensory interface

The plastic encoder is `plastic-balatro-field-aware-v2`. Every pinned upstream
position is classified as categorical/boolean, bounded ratio, raw count,
already-log scalar, signed-log scalar, or reserved zero. Categorical and
boolean values retain amplitude 0/1, raw counts use a documented log bound,
signed values use dual rails, and already-log values are not divided again by
a blanket group constant. It encodes observable simulator fields only: never
hand strength, recommended moves, expected value, or strategy.

Each feature drives a fixed-width sample of annotated ALPN projection neurons.
Assignment is a seeded deterministic random projection. Collisions use clipped
accumulation rather than division by total assignments, so unrelated channels
cannot attenuate a sparse indicator. The mapping artifact stores every
channel's source/class/transform, exact indices/root IDs, ALPN collision
quantiles/class distribution, version and SHA-256. Matched comparisons reuse
the mapping; mapping replicates change only `sensory_mapping_seed`.

Synthetic Balatro features have no natural fly-stimulus interpretation. Using
anatomical input routes constrains where artificial drive enters but does not
make a card rank biologically meaningful.

## Plastic edges and state

Only existing KC->MBON neuron-pair connections are plastic in V1. Fixed edge
indices and baseline signed anatomical weights are shared. Each independent
learning fly owns a one-dimensional efficacy vector over those sparse edges;
no dense neuron-by-neuron matrix is permitted. The effective edge strength is
the fixed anatomical magnitude multiplied by bounded non-negative efficacy.

Per-fly plastic state contains:

- initial and current efficacy;
- eligibility trace per plastic edge;
- internal trace for synthetic appetitive and aversive channels;
- update and bound-hit statistics;
- decision/episode counters.

Experiment checkpoints additionally carry the environment, motor, NumPy,
Python and Torch RNG states that drive each learner.

Fast neural state resets at every decision by default. Eligibility and
reinforcement traces persist within an episode, are used by its terminal
update, and then reset only for the learner that terminated. Efficacy persists
across episodes. Evaluation freezes plasticity and leaves all traces unchanged.

## Learning rule

V1 uses the named `three-factor-global-v1` rule. KC and MBON activity are
scaled by configured fixed `kc_reference_hz` and `mbon_reference_hz`, clipped,
multiplied edge-wise, decayed and bounded by `max_eligibility`. There is no
per-decision maximum normalization, so weak activity stays weak. A later
synthetic reinforcement pulse updates only eligible anatomical connections:

```text
e_ij <- clip(lambda_e e_ij + clip(KC_i/KC_ref) * clip(MBON_j/MBON_ref))
R_c  <- lambda_R R_c + synthetic_pulse_c
dw_ij = eta * e_ij * (R_appetitive - R_aversive)
w_ij <- clip(w_ij + dw_ij, w_min, w_max)
```

The default engineering convention strengthens eligible pathways toward active
MBONs after positive outcomes and weakens them after negative outcomes. It does
not attempt to assign biological valence to individual MBON types; alternate
compartment-specific or anti-Hebbian timing rules are separate versioned
experiments. `EdgeModulationAssignment` supplies the extension contract for
`compartmental-dan-v2`: compartment identity, edge-specific channel and
edge-specific sign. V1 does not populate it without defensible annotated
compartment-to-DAN assignments and never infers synapse locations from a
neuron-pair edge.

This captures the three-factor idea and the established role of compartmental
dopamine-gated KC->MBON plasticity, but simplifies synaptic timing, receptor
biology, heterogeneous compartments, consolidation, forgetting, and feedback.
The precise equation is a computational modelling convention, not direct
biological evidence.

Biological basis and modelling context:

- Schlegel et al. (2024), DOI 10.1038/s41586-024-07686-5: v783 annotation
  hierarchy and MB population census.
- Aso et al. (2014), DOI 10.7554/eLife.04577 and 10.7554/eLife.04580:
  compartmental DAN/KC/MBON architecture and MBON-guided action selection.
- Handler et al. (2019), DOI 10.1016/j.cell.2019.05.040: timing-dependent,
  dopamine-gated KC->MBON plasticity.
- Bennett et al. (2021), DOI 10.1038/s41467-021-22592-4: a reduced MB model
  with KC->MBON plasticity and reinforcement prediction errors. V1 deliberately
  starts with absolute outcome reinforcement and no learned external critic.
- Gerstner et al. (2018), DOI 10.3389/fncir.2018.00053: eligibility traces as a
  modelling bridge between activity and delayed neuromodulation.

## Fixed synthetic reinforcement mapping

Reward components are transformed by a versioned, non-trainable mapping into
synthetic appetitive and aversive scalar channels:

| Outcome | Default pulse |
|---|---:|
| bounded blind progress | weak appetitive |
| blind cleared | appetitive |
| Ante cleared | strong appetitive |
| failed run | aversive |
| Ante-8 success | strongest appetitive |

These are not simulated PAM/PPL1 spikes; genuine anatomical DAN activity is
recorded separately. The transform clips and logs pulses. It never rewards pairs, flushes, Jokers,
card identities, or a hand-authored action. The shuffled-reward control uses a
pre-generated deterministic permutation/buffer so outcomes and reinforcement have
the same marginal stream but broken temporal causality.

## Fixed motor interface

The motor interface contains no tensors marked trainable and no fitted state.
It maps named neural population scores to the simulator's structured heads:
action type, card count, card slots, and contextual target slots. Legal masks
set illegal logits to negative infinity. Deterministic mode uses stable argmax
with index tie-breaking; exploration is optional seeded softmax-temperature
sampling with an epsilon-uniform branch and is recorded as non-learned
interface state. Scientific mappings are selected by reward-free neural
calibration, not consecutive round-robin or single-neuron defaults. Calibration
rejects silent/high-rate-only outputs, ranks variance and dynamic range, and
persists population pools (default width two where available), exact root IDs,
sample seeds/state hashes, criteria and a mapping hash. Canonical output root
IDs come from the real unshuffled artifact; every matched control reuses the
same artifact.

Two output modes share the same interface:

- `mbon_direct`: scores come directly from configured MBON pools drawn only
  from MBONs that are postsynaptic in the identified plastic edge set. This is
  the basic plastic-learning experiment.
- `whole_brain`: changed KC->MBON weights participate in recurrent FlyWire
  simulation and scores come from fixed descending-neuron pools.

The mapping version, exact population root IDs, context rules, exploration
configuration, and hash are checkpointed.

## Learner identity and batching

The default scientific unit is one plastic fly experiencing episodes
sequentially. Environment vectorization may gather experience but must not mix
independent eligibility or weight state. Parallel experiments use a controlled
batch of independent flies with one shared fixed sparse whole-brain matrix and
a batched gather/scatter KC->MBON plastic contribution. No-autograd Torch
efficacy, eligibility and reinforcement traces can remain on the simulation
device and are separate per learner. Exploration derives from
`(motor_seed, learner_id, decision_id)`, so row ordering cannot perturb another
fly. Learned weights are never averaged or merged.

## Decision lifecycle

1. Read observable state and legal mask.
2. Apply the fixed sensory mapping.
3. Reset or retain fast state according to recorded configuration.
4. Simulate the configured neural window.
5. Measure Mode A MBON or Mode B descending populations.
6. Select a legal action with the fixed motor interface.
7. Step Balatro and decompose strategy-neutral outcomes.
8. Transform the outcome into fixed synthetic reinforcement channels.
9. The just-completed decision's KC/MBON activity has already extended its
   eligibility trace; apply the resulting outcome pulse to that trace.
10. If terminal, reset that learner's eligibility/reinforcement traces after
    the update while preserving efficacy; persist state and diagnostics.

This decision-scale eligibility ordering is versioned because it compresses
within-trial biological timing into one game transition and is a material
experimental assumption.

## Training, curriculum, and evaluation

Ante-1 uses a single-level `[1]` target with no promotion evaluation. Only
after its acceptance does training enable the retained Ante ladder
`1, 2, 3, 5, 8`, disjoint training, curriculum-validation, validation,
final-test, and showcase seeds, and held-out promotion. Budgets are environment
decisions and seed sequences, not optimiser updates. Evaluation loads the same
fly and disables every plastic update while freezing the plastic-weight hash.

Report win rate, Ante distribution, episode length, decisions, available score
metrics, action entropy, neural activity, plasticity events, changed-synapse
counts, weight drift, and bound fractions.

## Controls and matched comparisons

Required conditions are:

1. real topology + KC->MBON plasticity;
2. identical real fly with learning rate zero;
3. `kc_mbon_shuffled`, changing only KC->MBON postsynaptic labels while
   preserving all other edges, edge weights/count, KC out-degree and the MBON
   in-degree distribution;
4. `whole_brain_shuffled`, the distinct broader population-preserving label
   permutation;
5. real fly + deterministically shuffled synthetic reinforcement timing;
6. deterministic sensory-mapping replicates;
7. archived fixed-reservoir/PPO baseline;
8. conventional ML and random-legal external references.

Primary comparisons match environment decisions, seed order, curriculum,
reinforcement map, motor map, and (except mapping replicates) sensory map.
Record wall time, hardware, episodes, plasticity events, and GPU usage.

## Checkpoints and replay

A plastic checkpoint contains current and initial efficacy, eligibility and
internal reinforcement traces, optional fast neural state, counters, curriculum, all RNG
states, environment snapshot where supported, fixed topology/artifact hash,
population census/hash, plasticity rule/config hash, sensory/motor/reinforcement
mapping payloads and hashes, experiment configuration, Git revision, and
software/hardware metadata. Restore must reproduce future weight changes.

Existing deterministic simulator/live replay remains. Selected recordings add
KC, MBON, genuine DAN anatomy and descending activity, honestly labelled
synthetic reinforcement, and sparse plasticity events or periodic snapshots.
The optional frozen showcase also records true simulated spike times and input
stimulation. Parquet is streamed one decision per row group with dictionary
roles/kinds, and visualization reads selected row groups. Full plastic matrices are never recorded
per timestep. Visual coordinates use one persisted whole-artifact transform;
frames may change activity but never rescale or move anatomy.

## Diagnostics and safety gates

Cheap synthetic tests precede real-data runs. Real representation metrics are
explicit spike rates in Hz and include KC/MBON/descending distributions,
within/between-state variance and distance, separation ratio, silent/high-rate
fractions, motor option coverage/dynamic range, eligibility magnitude, update sizes,
bound fractions, drift, action entropy, and dominance. Fail on non-finite
state, zero updates despite eligible reinforcement, uncontrolled saturation,
or a silent/constant motor interface.

`flylatro-preflight` combines artifact/data quality, sensory collisions,
representation, motor, plasticity, reward mapping, trace reset, checkpoint,
benchmark, control and zero-external-policy evidence into PASS/WARN/FAIL with
configured thresholds. Full artifact builds, whole-brain simulations, long training, large recording,
and final tests require `--full`, `--heavy`, or the existing final-test unlock.
Local completion is based on tiny circuits and short mock-environment runs;
scientific claims remain gated on the dedicated GPU machine.

## Implementation sequence and completion gates

1. Archive legacy plan; replace documentation/status/config authority.
2. Extend the v783 artifact schema and validate MB populations/edges.
3. Implement sparse plastic state, eligibility, dopamine and checkpoints.
4. Implement seeded sensory and zero-parameter structured motor interfaces.
5. Implement MBON-direct synthetic and real-artifact paths.
6. Integrate plastic edges into the Torch whole-brain backend.
7. Implement sequential plastic trainer, frozen evaluator and curriculum.
8. Add controls, matched-budget manifests and mapping replicates.
9. Add neural/plastic diagnostics and synaptic analysis.
10. Extend replay/visualisation and benchmarks.
11. Validate all fast tests and publish exact heavy-machine commands.

Plastic-brain V1 is code-complete only when every item in `docs/V1_STATUS.md`
is at least `LOCALLY VALIDATED` or explicitly `NEEDS HEAVY VALIDATION`, with
real commands and success criteria in `docs/GPU_RUNBOOK.md`. Heavy validation
itself is not required on this development machine.

The target technical statement is:

> Flylatro learns by modifying synapses inside a connectome-derived model of
> the fly mushroom-body learning circuit; its Balatro sensory and motor
> interfaces are fixed and contain no learned policy.
