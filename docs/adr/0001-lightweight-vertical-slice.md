# ADR 0001: lightweight vertical-slice boundaries

Status: accepted for development; scientific backends remain unvalidated.

## Context

The initial repository contained only the project plan. The first useful
increment needed to prove component boundaries and replay semantics without
installing or executing the real Balatro simulator, adult connectome, PyTorch,
or PPO on the development machine.

## Decisions

- Flylatro owns simulator-neutral observation, complete composite-action,
  conditional-mask, vector-step, snapshot, and restore contracts.
- The mock environment is explicitly not Balatro. It supplies deterministic
  seeded state transitions and only strategy-neutral progress/clear/win reward
  components for contract testing.
- `EncoderSpec` and `FeatureSpec` are immutable, canonicalised, and SHA-256
  hashed. The development encoder uses named neuron assignments. A real
  sensory-neuron mapping must be a new version, never an in-place change.
- The tiny backend shares one fixed connectivity matrix across a batch and is
  reset for each decision. It is an inexpensive LIF-like test double, not a
  biologically credible adult-fly simulation.
- Neural output is compressed by an explicit fixed feature extractor before it
  reaches the policy.
- The first readout uses direct linear action-type, card-count, card-selection,
  target, and value heads. There is no conventional hidden network. Its
  trainable parameter count is recorded in every run manifest.
- Conditional masks enforce rules after the action type is chosen. The fly runs
  once per state; it is never rerun for individual action candidates.
- JSONL transitions include the full action and before/after state hashes.
  Episode IDs are excluded from state hashes so replays can use another vector
  slot without false divergence.
- NumPy is the only runtime dependency for this slice. An autodiff policy for
  PPO should preserve the same structured interface rather than expanding the
  current readout implicitly.

## Consequences

This slice can validate batching, determinism, masks, component dimensions,
and trace replay cheaply. It cannot validate real Balatro fidelity, biological
plausibility, GPU throughput, trainability, or the main research hypothesis.
Those claims remain gated on the licensed upstream adapters and measured adult
connectome backend.

