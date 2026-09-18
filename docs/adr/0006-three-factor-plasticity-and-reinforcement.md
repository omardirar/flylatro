# ADR 0006: three-factor plasticity, dopamine, and reset timing

Status: accepted for V1; parameters need experimental calibration.

## Decision

Active KC and MBON pairs create a decaying edge-local eligibility trace.
Fixed appetitive and aversive dopamine channels gate bounded efficacy changes
on those traces. The implementation is modular, versioned, and exposes exact
decay, learning rate, bounds, event ordering and the
appetitive-minus-aversive sign convention.

Outcome components map to dopamine through a fixed table: bounded progress,
blind clear, Ante clear and final success are appetitive at increasing scales;
run failure is aversive. Strategy concepts and action labels do not enter the
reinforcement value. V1 uses no learned critic. A deterministic shuffled-event
buffer implements the causal reward control.

Fast LIF state resets before each decision by default. Plastic efficacy and
eligibility persist across decisions and episodes. Evaluation freezes updates.

## Biological basis and simplification

KC->MBON connections are a principal site of compartmental dopamine-gated
associative plasticity, with experimentally observed timing-dependent and
bidirectional effects. Eligibility-gated scalar pulses are a modelling
convention that bridges delayed game outcomes; they omit receptor kinetics,
heterogeneous consolidation, feedback-computed prediction error and detailed
compartment dynamics. The default update sign is selected for an interpretable
action-learning experiment, not presented as a universal fly synaptic law.

## Alternatives

PPO, a learned value baseline and gradient descent through the brain were
rejected from core V1. A more detailed anti-Hebbian timing kernel remains an
alternate rule once basic learning and real-data dynamics are measured.
