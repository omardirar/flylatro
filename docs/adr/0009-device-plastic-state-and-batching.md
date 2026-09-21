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

The dedicated benchmark compares the same batched operation with row-by-row
execution and reports measured throughput and peak memory. No speedup claim is
made before those measurements.

## Classification

Sparse KC->MBON locality is the modelling constraint. Device residency,
scatter-add decomposition and NumPy serialization are engineering choices.
