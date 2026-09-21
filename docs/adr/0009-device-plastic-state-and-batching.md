# ADR 0009: no-autograd device plastic state and independent batching

Status: architecture locally validated on CPU Torch; CUDA performance pending.

## Decision

The real path keeps efficacy, eligibility and reinforcement traces as Torch
tensors on the simulation device under `torch.no_grad()`. No parameter,
autograd graph, optimizer or gradient descent is introduced. Checkpoints
serialize NumPy arrays and restore them to the configured device.

One fixed sparse whole-brain matrix is shared by every learner. KC->MBON values
are zeroed in that fixed matrix and computed as a batched gather/multiply/
scatter-add using separate efficacy rows. Neural and plastic states remain
independent; no dense whole-brain matrix is created per learner. Event hashes
and checkpoints may require explicit device-to-host snapshots at audit edges.

Routine training keeps the plastic state on device end to end. Per-decision
metrics are Torch reductions and reach the host as one small scalar block:
changed synapses, absolute and maximum update magnitude, eligible edge count,
eligibility mean/max, bound hits and both reinforcement channels. A normal CUDA
step never copies a full efficacy, eligibility or reinforcement vector.

`PlasticityEvent` is therefore a scalar summary by default.
`PlasticityEventDetail` — before/after weight hashes, changed edge IDs and
per-edge deltas — is opt-in and belongs to short gate runs, audits, selected
recordings and diagnostics. Weight hashing moves to checkpoint time, explicit
audit time and end of run rather than every decision. The scientific semantics
are unchanged: the same update is applied in the same order with the same
bounds, and no autograd graph or optimizer is introduced.

The dedicated benchmark compares the same batched operation with row-by-row
execution and reports measured throughput and peak memory. It can also attribute
wall time to Poisson input generation, fixed sparse recurrence, the plastic
KC->MBON contribution, the LIF state update and device synchronization, so a
performance decision rests on a measurement. `fly.poisson_method` selects
between the per-step per-row draw and a chunked per-row draw that preserves
independent per-learner streams, reproducibility and row-order independence;
switching it changes `fly_dynamics_sha256` and invalidates evidence measured
under the previous method. No speedup claim is made before those measurements.

## Classification

Sparse KC->MBON locality is the modelling constraint. Device residency,
scatter-add decomposition and NumPy serialization are engineering choices.
