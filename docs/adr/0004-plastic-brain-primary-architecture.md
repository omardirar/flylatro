# ADR 0004: plastic-brain primary architecture

Status: accepted and locally validated; real FlyWire execution remains heavy validation.

## Context

The original V1 held the fly fixed and trained a PPO actor/critic outside it.
That experiment cannot answer whether learning localized to mushroom-body
synapses can acquire behaviour.

## Decision

Plastic-brain V1 makes sparse KC->MBON efficacy the only persistent trainable
model state. The encoder, topology, neural dynamics, reinforcement transform,
and structured motor interface are fixed. PPO remains under the legacy
namespace/commands only and cannot be imported by the new trainer.

The same motor contract supports `mbon_direct` for isolating basic learning and
`whole_brain` for testing propagation to fixed descending outputs. Mode B is a
required architecture and locally testable on synthetic graphs, but full FAFB
behaviour is a heavy-machine validation gate rather than a prerequisite for
debugging Mode A.

The default scientific unit is one learned fly with sequential experience.
Parallel learners, when enabled, have independent weights, traces, RNG and
curriculum state. Learned synaptic states are never averaged.

## Alternatives

- Training an external actor is retained as a legacy baseline, not the primary
  experiment.
- Backpropagating through all 139k neurons was rejected because it changes the
  research question and makes the learned state non-local.
- One shared brain updated from a vector of simultaneous environments was
  rejected because reward/eligibility ordering would be ambiguous.

## Consequences

Training throughput may be lower than PPO vectorization, but matched exposure
and interpretable learned state take priority. Experiment manifests must say
whether a run is sequential-single-fly or independent-parallel-fly.
