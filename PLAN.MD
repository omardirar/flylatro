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

Structural assignment counts are informational. They count every feature channel
that could share an ALPN across the whole pinned contract, active or not.
Readiness is decided instead by state-conditioned sensory health measured on the
frozen calibration corpus: silent/active/saturated ALPN fractions, simultaneous
active contributors per driven ALPN, the fraction of saturation actually caused
by simultaneous collisions, same-state vector equality, and above all whether
two distinct observable states ever produce the same ALPN vector. Structural
collision load and simultaneous active collision load are different quantities
and only the second can destroy observable-state information.

Synthetic Balatro features have no natural fly-stimulus interpretation. Using
anatomical input routes constrains where artificial drive enters but does not
make a card rank biologically meaningful.

## Frozen reward-free calibration corpus

Calibration must not depend on trajectories produced by whatever bootstrap
decoder happened to exist. A first-class, versioned, SHA-256-hashed corpus holds
observable Balatro states and legal masks only. It contains no reward label, best
action, heuristic recommendation, expected value or strategy annotation.
Deterministic scripted legal navigation from fixed simulator seeds reaches
diverse observable phases; those actions are discarded and are never training
labels or policy targets. The corpus persists its version, the Balatro simulator
version, per-state hashes, seeds, phase labels, generation method, coverage and
optional exact environment snapshots, and is reused by sensory-health
diagnostics, representation diagnostics, reward-free motor calibration, mapping
replicates and neural-duration comparisons. It is part of experimental
provenance.

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

Routine training keeps efficacy, eligibility and reinforcement traces resident on
the simulation device. Per-decision metrics are device reductions delivered as one
small scalar block, so a normal CUDA step never copies a full plastic vector to
the host. Per-edge detail — before/after weight hashes, changed edge IDs and
per-edge deltas — is opt-in and belongs to short gate runs, audits, selected
recordings and diagnostics; weight hashing is checkpoint, periodic-audit or
end-of-run work. The scientific semantics are unchanged and no autograd graph or
optimizer exists anywhere in this path.

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

Three shaping conditions are predeclared: `primary-progress` (the V1 default),
`reduced-progress` and `terminal-or-clear-only`. They differ only in the
blind-progress magnitude and exist to determine whether apparent learning depends
on dense progress reinforcement. They are named before any result is seen and are
never selected by whichever eventually scores best.

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
interface state.

Pools are distinct only where options compete inside one comparison. The fixed
`contextual-motor-routing-v1` scheme defines four competition groups — supported
action types, exact card counts, hand card slots, and one shared contextual
target slot group. Joker, consumable, shop and pack slots share the contextual
group because the already-selected action type makes them mutually exclusive;
card slots stay separate because a consumable reads both at once. The routing is
zero-parameter, non-trainable, deterministic, versioned and hashed. It needs 35
pools rather than the 61 that summing every head width would demand.

Action types 0-12 are the scientifically represented set. `MOVE_JOKER` needs a
source and destination the pinned action contract cannot carry, and 14-24 are
index-stability placeholders. Reserved slots own no neural population, score
negative infinity, are counted whenever the contract reports one legal, and can
never be selected; the serialized artifact still spans the full mask/action
contract and records full width, represented actions and reserved slots.

Motor candidates are the MBONs that are postsynaptic in the real unshuffled
KC->MBON plastic topology at the canonical minimum synapse count, so every motor
output is plastic-reachable. The universe does not move because topology was
shuffled, a weak-edge threshold changed or the control condition changed; its
hash is persisted and checked when a configuration loads a motor artifact. Each
candidate records root ID, KC plastic input edge count, total KC->MBON
anatomical synapse weight, MBON type and compartment label where known.

Scientific mappings are selected by reward-free neural calibration over the
frozen corpus, not round-robin or single-neuron defaults. Calibration rejects
silent and high-rate-only outputs, ranks robust scale and dynamic range, and
applies a within-group correlation ceiling so near-identical populations are not
assigned to competing alternatives. It reports candidate/selected counts, pool
width, pool reuse, baseline firing distribution, dynamic range, variance,
normalized dynamic range, silent and high-rate fractions, competing-pool
correlation and the effective number of distinct signals, and fails clearly when
a head cannot obtain enough usable diversity.

Each pool also stores fixed reward-free normalization
(`reward-free-median-iqr-v1`): baseline is the corpus median and scale is
`max(IQR / 1.349, minimum_scale_hz)`. `decode` subtracts the baseline and divides
by the scale before comparing alternatives, so two pools with different absolute
firing rates but equivalent relative modulation compete fairly. Both vectors are
frozen after calibration, contain no Balatro outcome information, have zero
trainable parameters, and are versioned, hashed and persisted.

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

Controls fall into two kinds that must not be confused. `no_plasticity` and
`shuffled_reward` are **exact action/state matched**: they replay the real fly's
executed action schedule and must reproduce its before/after state hashes, seed
stream, Ante, sensory map, motor map and environment budget. `kc_mbon_shuffled`
and `whole_brain_shuffled` are **behaviourally independent topology controls**:
changing topology changes policy behaviour, so they cannot share the real fly's
future trajectory. They are matched instead on the initial environment seed
sequence, exposure budget, curriculum, sensory map, canonical motor map,
reinforcement rule, simulation duration and hardware protocol — and their plastic
topology hash must actually differ from the real one. The canonical motor root
IDs, pool structure and normalization parameters are identical across every
matched condition.

Replicate protocols are executable, not descriptive: every arm materializes into
one exact validated configuration plus its command and dependency order, and each
run re-validates its configuration against its protocol arm before starting.
Every stored protocol seed controls a named stochastic mechanism; a seed with no
effect is removed rather than implied.

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

Representation calibration has two stages. `representation-pre` measures neural
representation only and makes no claim about the final motor interface.
`representation-post` re-runs with the persisted final motor mapping and reports
normalized motor-pool activity, per-head option dynamic ranges, action-option
coverage, competition behaviour and motor score distributions. Only the
post-motor report may satisfy a motor-related readiness gate. The motor
diagnostic always measures the exact population the motor interface consumes:
MBON output activity in `mbon_direct`, descending output activity in
`whole_brain`.

Three further diagnostics are measurements, never policy components. KC-subtype
reachability asks whether ALPN-only drive and the configured window actually
reach the plastic population. Chosen-action specificity attributes eligibility
and weight change to the chosen motor pool, the competing pools, motor pools the
chosen action never consulted, and non-motor MBONs; it determines whether
`three-factor-global-v1` is sufficiently action-specific and is never a reason to
bias the rule in V1. Observable-category probes exclude singleton classes so a
held-out sample cannot be its own centroid.

Every generated evidence artifact carries an explicit experiment identity: git
commit and dirty state, config SHA, Balatro simulator version, FlyWire artifact,
population and connectivity SHAs, plastic topology SHA, minimum synapse
threshold, sensory feature-contract and mapping SHAs, calibration corpus SHA,
motor candidate-set and mapping SHAs, reinforcement and plasticity rule SHAs, fly
dynamics SHA, decision duration, output mode, backend and device.

`flylatro-preflight` combines artifact/data quality, calibration corpus identity,
sensory contract and state-conditioned sensory health, representation pre/post,
KC reachability, motor artifact/candidate/normalization validity, reserved-slot
safety, fixed-reference eligibility, episode-boundary reset, plasticity stability
and bound occupancy, chosen-action specificity availability, checkpoint/resume,
device-state consistency, benchmark provenance, matched-control readiness,
protocol identity and zero-external-policy evidence into PASS/WARN/FAIL with
configured thresholds. It never trusts a report's own status: each gate verifies
that the report's recorded identity matches the current configuration and fails
with an exact mismatch otherwise. Two profiles exist — `initial` authorises the
first tiny real experiments and downgrades measurements that can only happen
later; `ante1` is strict. Every threshold is a documented engineering gate, not a
biological fact. Full artifact builds, whole-brain simulations, long training, large recording,
and final tests require `--full`, `--heavy`, or the existing final-test unlock.
Local completion is based on tiny circuits and short mock-environment runs;
scientific claims remain gated on the dedicated GPU machine.

## Scientific distinctions that must never be blurred

Three categories are kept separate throughout the code, the reports and the
documentation: **biological evidence**, **computational modelling assumptions**
and **engineering interface choices**. The following statements are load-bearing
and must survive every future change:

- The Balatro-to-ALPN mapping is **synthetic**. Using an anatomical input route
  constrains where artificial drive enters; it does not make a card rank a
  biologically meaningful stimulus.
- Motor action semantics are **synthetic**. Which MBON drives "shop slot 2", and
  the convention that the same pool also means "pack slot 2" in a different
  context, are fixed interface choices with no fly-behavioural claim.
- Synthetic appetitive and aversive reinforcement is **not** simulated PAM/PPL1
  spiking. Genuine annotated DAN anatomy and activity are recorded separately
  and labelled differently everywhere.
- The global three-factor rule is a **modelling convention**. It compresses
  within-trial biological timing into one game transition and uses one global
  scalar difference rather than evidence-derived compartmental DAN signals.
- KC->MBON anatomy is **connectome-derived**. The edge set, its endpoints and
  its aggregate synapse counts come from the pinned v783 export.
- Learning resides in **KC->MBON efficacy**. That vector is the entire
  persistent learned behavioural state; the external trainable parameter count
  is zero.
- Legal action masks are **environment constraints**, not learned strategy. The
  interface never adds preferences to a mask and never treats a mask as advice.
- Diagnostic classifiers and probes — observable-category centroid probes,
  reachability, specificity attribution, representation geometry — are
  **measurements only**. None of them feeds the policy, and none of them may be
  optimised against.

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
