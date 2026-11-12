# Accelerated execution and checkpoint compatibility

Execution selection is explicit. `[fly].backend = "flywire"` still describes the
anatomical model; `[fly].execution` selects its implementation. The default is
`torch_reference`. The source-complete configuration retains 139,255 neurons,
19,773,733 anatomical connections, the original sensory/motor mappings,
100 ms per decision and a 0.1 ms timestep. Learning remains internal KC→MBON
plasticity with zero external trainable policy, decoder or critic parameters.

## Build on the qualified machine

The measured stack is Python 3.14.8, Torch 2.14.1+cu130, CUDA 13.0, nvcc
13.0.88, GCC/G++ 14.3.1 and an RTX 4080 (compute capability 8.9). The system
GCC 16 is unsupported by this CUDA compiler. A compatible GCC 14 is extracted
inside the workspace; it does not replace the system compiler.

Install a Torch wheel matching the intended CUDA toolkit first. The `native`
extra supplies the optional extension build dependencies; it does not select a
CUDA toolkit or silently switch to a CPU implementation.

```bash
.venv/bin/python -m pip install -e '.[training,native]'
.venv/bin/python -m pip install \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.96 \
  nvidia-cuda-cccl==13.0.85

export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9
.venv/bin/python -m flylatro.fly.native_build
.venv/bin/python -m flylatro.fly.sparse_build
.venv/bin/python -m flylatro.fly.event_build
```

The pinned GCC archive commands are in [the work record](PERFORMANCE_MIGRATION.md).
Use an equivalent supported compiler if the workspace bundle is absent.
`CUDA_HOME` can explicitly select a compatible toolkit; otherwise the builder
discovers the installed NVIDIA `cu13` package. Toolkit and Torch CUDA versions
must agree. Builds are content-addressed under `artefacts/native-build`, including
CUDA source hashes, compiler/nvcc/Torch versions, architecture and flags. The
native neuron kernels use FP32, `--fmad=false` and no fast math. A requested
native implementation fails clearly if its GPU/toolchain is unavailable.

## Backends

| Execution | Device | Fixed recurrence | Detailed events / component profiling | Resume |
|---|---|---|---|---|
| `torch_reference` | CPU or CUDA | FP32 COO, int64 indices | Yes | Legacy reference checkpoints supported; atomic plastic accumulation prevents a blanket bitwise neural-replay guarantee |
| `torch_csr` | CPU or CUDA | FP32 CSR, checked int32 indices | Yes | Same backend identity required; reference checkpoints are rejected as exact resumes |
| `cuda_native` | CUDA | PyTorch cuSPARSE CSR plus fused LIF | Yes | Native-v2 decision-boundary resume qualified |
| `cuda_native_graph` | CUDA | Native-v2, complete neural-decision graph | No | Same identity required; exact native-v2 replay/resume qualified |
| `cuda_native_graph_blocks` | CUDA | Native-v2, delay-aligned graph blocks | No | Same identity required; exact native-v2 replay/resume qualified |
| `cuda_native_cusparse` | CUDA | Reusable cuSPARSE descriptors/workspaces; row-major batched SpMM | Yes | Explicit separate version; native-v2 trajectory/resume audit passes |
| `cuda_native_cusparse_graph_blocks` | CUDA | Reusable cuSPARSE plus graph blocks | No | Same identity required; exact trajectory/resume audit passes |
| `cuda_native_event` | CUDA | Active outgoing edges, validated integral fixed weights | Yes | Explicit separate version; native-v2 trajectory/resume audit passes |
| `cuda_native_event_graph_blocks` | CUDA | Active outgoing edges plus graph blocks | No | Same identity required; exact trajectory/resume and independent-learner audits pass |

Every native variant keeps a separate deterministic, fixed-tree FP32 plastic
reduction. Anatomical KC→MBON positions are zeroed in fixed recurrence and added
once through per-learner efficacy. The event variants require integral fixed
weights and each destination's incoming absolute sum below 2²⁴. Neither weights
nor connections are rounded or pruned. This makes their binary-spike FP32 atomic
sums independent of accumulation order. Invalid weights fail explicitly.

Backend/build identities are distinct from canonical anatomical hashes. Even
where trajectory checks agree, changing an execution identity is not advertised
as exact cross-backend checkpoint resume. Start a fresh run for a different
implementation; retain source provenance when constructing a deliberate fork.
Changing batch size also changes independent learner state shape and requires
a fresh training run. Frozen evaluation can replicate one checkpoint's weights
into independent lanes without sharing experience.

For detailed recording, select the corresponding eager backend explicitly.
CUDA Graphs accelerate neural timesteps within one decision and cannot capture
future game decisions or their unknown weight updates.

## Training and monitoring

The configuration
`configs/plastic-v783-source-complete-event-qualified-v1.toml` retains the
original model and mappings and points at fresh calibration evidence. Its
initial preflight has no failed gates. Warnings about matched controls and a
learning protocol are recorded separately; these performance measurements do
not establish learning success or authorize a different scientific protocol.

After exporting the compiler/architecture variables above, a full 1,100-decision
single-fly run is:

```bash
.venv/bin/python -m flylatro.learning.cli \
  --config configs/plastic-v783-source-complete-event-qualified-v1.toml \
  --run-dir runs/performance-event-single-new \
  --heavy --dashboard --dashboard-refresh 1 --no-tensorboard
```

Choose a fresh run directory. To end a stage at a checkpoint boundary while
retaining the declared budget, add `--stop-after-decisions 1000`. Resume with the
same configuration and run directory plus `--resume <checkpoint.pkl>`.

The eight-fly configuration declares 8,800 aggregate decisions, or 1,100 per
independent learner, and checkpoints every 400 aggregate decisions:

```bash
.venv/bin/python -m flylatro.learning.cli \
  --config configs/plastic-v783-source-complete-event-eight-v1.toml \
  --run-dir runs/performance-event-eight-new \
  --heavy --dashboard --dashboard-refresh 1 --no-tensorboard
```

Each fly owns its efficacy, traces, environment and random stream. Report
aggregate and per-learner exposure separately. Eight flies do not provide one
fly with 8,800 decisions of learned experience.

For the subsequent action/punishment experiments, see
[blind skipping and failure punishment](NO_SKIP_EXPERIMENT.md). These are
separately named training variants, with their own checkpoint action policy.

`--log-buffer-bytes` defaults to 262,144 and `--log-flush-seconds` to 1. Ordered
durable logs flush before checkpoints and before final hashes. Writer errors
propagate. SIGINT finishes the current decision/audit boundary, flushes logs and
writes an interrupted checkpoint, returning exit code 130. Health stops retain
their failure report and last safe checkpoint. Software/Git provenance is
captured once per CLI invocation; state, counters and weight hashes are fresh
for every checkpoint. Serialization remains synchronous and atomically
published because measured snapshot sizes do not justify background ownership
and queue complexity.

Population diagnostics materialize their owned GPU activity on request. Routine
training copies motor activity and compact GPU health reductions; eligibility
stays on-device. Whole fast-state finiteness is checked, including silent
neurons. Recent episode histories are bounded by `[training].episode_history_limit`
(512 by default), while lifetime totals/best results and durable episode records
remain resumable. Legacy histories are summarized before trimming.

## Preparation and calibration caches

Preparation caches are opt-in and affect storage/loading, not neural semantics:

```bash
export FLYLATRO_ARTIFACT_CACHE_DIR="$PWD/artefacts/prepared-artifacts"
export FLYLATRO_LAYOUT_CACHE_DIR="$PWD/artefacts/prepared-layouts"
```

Artifact preparation retains read-only mapped arrays. Original source bytes,
manifest identity and every cached array hash are verified. Fixed-layout caches
include source/population/plastic topology, shuffle procedure, backend/build,
Torch/NumPy/CUDA/device capability, precision, index width and direction. Loaded
CSR ranges and plastic separation are checked. Device tensors own their storage;
one learner/backend cannot mutate another's cached anatomy. No learned efficacy,
RNG state or stochastic training activity is cached. Corruption fails explicitly.

The measured warm preparation cache reduces startup GPU peak substantially but
does not improve startup wall time on this machine. It remains disabled by
default. Reward-free activity reuse is separate: `--activity-cache` is verified
against model/mapping/build, corpus, sampling and fixed-unit-efficacy identities.
It cannot be applied to learned-checkpoint diagnostics.

`flylatro.analysis.motor_calibration_cli --validate-existing` requalifies the
configured pools and normalization using fresh reward-free activity. It leaves
the mapping identity intact. Ordinary calibration still selects a new mapping;
that selection is a different experiment and is not part of the primary
performance comparison.

## Reproduction and evidence

Repeated training measurements use warmup, immutable restored snapshots,
production safety checks and fresh outputs:

```bash
.venv/bin/python -m flylatro.benchmarks.migration \
  --config configs/plastic-v783-source-complete-event-performance-v1.toml \
  --execution torch_reference --batch-size 1 --warmup 10 \
  --steps 1000 --repetitions 3 --heavy \
  --output artefacts/performance-migration/new-reference/training.json
.venv/bin/python scripts/run_performance_suite.py \
  --config configs/plastic-v783-source-complete-event-qualified-v1.toml \
  --execution cuda_native_event_graph_blocks --batches 1,8 \
  --steps 1000 --repetitions 3 \
  --output-dir artefacts/performance-migration/new-event-training
```

Use `--workload neural` for controlled workload A and `--workload cli` for the
complete runtime. `scripts/run_frozen_performance_suite.py` measures workload D;
`scripts/profile_execution.py` records intrusive timelines, spike sparsity and
checkpoint stages separately. CUDA trial events include dispatch idle gaps and
are not summed kernel duration. Final throughput measurements do not enable
the synchronizing component profiler. Run GPU jobs serially.

See [the performance report](PERFORMANCE_REPORT.md) for measured gains, memory
and numerical limits, [the acceptance audit](PERFORMANCE_ACCEPTANCE.md) for coverage,
and [the work record](PERFORMANCE_MIGRATION.md)
for executed commands and qualification artifacts.
