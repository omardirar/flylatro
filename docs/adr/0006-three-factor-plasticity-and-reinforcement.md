# ADR 0006: fixed-reference three-factor plasticity and reinforcement semantics

Status: accepted and locally validated; numerical parameters need real calibration.

## Decision

The primary rule is named `three-factor-global-v1`. KC and MBON firing rates
are divided by configured fixed `kc_reference_hz` and `mbon_reference_hz`,
clipped, multiplied edge-wise, decayed, and clipped to `max_eligibility`.
Decision-relative maximum normalization is prohibited because it promotes
arbitrarily weak noise to full eligibility.

At a terminal transition the ordering is: record the last activity, apply its
outcome reinforcement and efficacy update, then clear eligibility and the two
reinforcement traces for that learner only. Efficacy persists. Cross-episode
trace carryover is not a V1 mode.

Balatro outcomes drive fixed **synthetic appetitive reinforcement** and
**synthetic aversive reinforcement** scalar channels. They are not simulated
PAM/PPL1 spikes. Logs, Parquet roles, replay metadata and visualization say so;
genuine DAN anatomy/activity remains distinct. No strategy labels enter the
mapping and no critic is learned.

`EdgeModulationAssignment` defines a versioned `compartmental-dan-v2` extension
with edge compartment, channel index, and plasticity sign. It refuses missing
assignments. V1 retains the global channel because v783 neuron-pair edges do
not by themselves establish synapse-level compartments or a complete
compartment-to-DAN rule.

## Evidence and assumptions

- Biological evidence: KC->MBON synapses are compartmentally modulated by DAN
  systems and can exhibit timing-dependent plasticity.
- Computational modelling assumption: a decision-scale eligibility trace and
  signed global scalar reinforcement approximate delayed game outcomes.
- Engineering choice: fixed rate references, explicit bounds, terminal reset,
  and configurable safety gates make magnitude and lifecycle reproducible.

## Consequences

Plasticity calibration must report drift, update magnitude, eligibility and
bound occupancy before training. Changing reference rates, trace bounds, rule
version, compartment assignment, or minimum anatomical edge threshold changes
the experiment manifest.
