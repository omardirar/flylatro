# Performance migration work record

Status: implementation and performance acceptance complete on 2026-10-14.
The reference remains the default; qualified acceleration is explicit. This
file preserves historical work snapshots, including earlier live/pending notes.
Use the [performance report](PERFORMANCE_REPORT.md), [acceptance audit](PERFORMANCE_ACCEPTANCE.md)
and [migration guide](PERFORMANCE_MIGRATION_GUIDE.md) for final results and limits.
The full scope is the user attachment at
`/home/omz/.codex/attachments/71142196-41fb-4b92-bbfd-5fa7c39e11b7/pasted-text-1.txt`.
Existing experiment changes were present before this task and are preserved.
No AGENTS.md was found under the project parent.

## Measurement foundation (2026-10-10)

`python -m flylatro.benchmarks.migration` now measures complete training
**decisions**, including the production trainer's health checks. Unlike the
historical `bench-plastic-end-to-end`, it uses `trainer.train`, which enforces
the safety monitor. It warms up, takes an immutable serialized trainer snapshot,
and restores that same state for each measured repetition. Each restore gets
fresh mutable arrays. Warm-up and measured exposure must fit the configured
budget. Outputs refuse overwrites.

It reports startup, snapshot capture/serialization, per-trial wall throughput,
aggregate and per-learner accounting, vector-step latency samples/quantiles,
process peak RSS, GPU allocation/reservation peaks, configuration/components,
Python source digest, Git state, software and GPU model. CUDA events span an
entire trial: their elapsed time includes dispatch idle gaps and must **not**
be interpreted as summed GPU kernel time. No additional per-operation CUDA
synchronization is introduced. Existing backend synchronization remains.

This is workload B, not a replacement for the required neural-only, full CLI,
evaluation/calibration, timeline and startup breakdown measurements. Dashboard,
durable logging, periodic checkpoints and component profiling are off and
explicitly labelled. Repeat final-weight agreement is a reproducibility check,
not a full trajectory or cross-backend correctness qualification.

Commands executed:

```bash
.venv/bin/python -m pytest tests/test_migration_benchmark.py tests/test_benchmark_accounting.py -q
# 5 passed
.venv/bin/python -m flylatro.benchmarks.migration \
  --config configs/plastic-smoke.toml --warmup 2 --steps 4 --repetitions 3 \
  --output artefacts/performance-migration/20261010-harness-v1/smoke.json
.venv/bin/python -m flylatro.benchmarks.migration \
  --config configs/plastic-v783-source-complete-gain0p225-100ms-lr0p002-pilot.toml \
  --warmup 10 --steps 100 --repetitions 3 --heavy \
  --output artefacts/performance-migration/20261010-harness-v1/reference-training.json
```

The GPU baseline was launched in exec session **70364**. Revalidate that session
and the output before starting another GPU workload. At launch the only listed
compute GPU user was PID 264223, Zed (188 MiB), and host process inspection found
no live training run. Sandbox CUDA detection returns false; host GPU execution
requires the approved escalation. Python is 3.14.8; Torch is 2.14.1+cu130.
`nvcc` was not on PATH or at `/opt/cuda/bin/nvcc` or
`/usr/local/cuda/bin/nvcc`; full compatible toolchain discovery remains pending.

## Remaining scope

1. Finish architecture and training-path inspection; preserve canonical model,
   reward, RNG and topology semantics. Record current artifact counts/hashes.
2. Complete baseline workloads A–E, sustained repetitions and practical batch
   sizes 1/2/4/8/16. Include timeline evidence and CPU/GPU memory stability.
3. Explicit common backend factory/configuration and implementation identity,
   with reference retained; prototype and measure int32 CSR after plastic-edge
   separation. Qualify numerical/behavioural differences instead of assuming
   equivalence.
4. Buffer/index/generator reuse, compact activity transfer, optional C++/CUDA
   extension, compatible reproducible installation, fused FP32 neuron updates,
   cuSPARSE options/workspaces, then graph capture with independent fresh RNG.
5. Buffered durable logging, responsive existing dashboard, immutable bounded
   asynchronous checkpoint work where measured worthwhile; failure/interrupt
   consistency and exact resume validation.
6. Accelerated frozen evaluation, calibration and safe cache identity; practical
   batching and persistent artifact reuse.
7. Measure spike sparsity and assess advanced propagation only if justified;
   altered timestep/duration/precision remain separate model variants.
8. Full meaningful tests, real GPU qualifications, matched before/after
   measurements, migration/build documentation, compatibility/resume table,
   final performance report and qualified long-run commands. Default rollout
   requires all user-specified gates. No speedup or completion is claimed yet.

## Explicit CSR prototype

`[fly] execution = "torch_csr"` selects FP32 PyTorch CSR recurrence with checked
signed-int32 columns and row offsets. `torch_reference` remains the default.
The factory is used by the shared stack builder, so training and its frozen
analysis/evaluation consumers select the same engine. Conversion occurs after
plastic KC→MBON values are removed from fixed recurrence; independent plastic
contributions remain unchanged. Canonical connectivity hashes are retained.
CSR has a distinct dynamics version and implementation metadata because no
real-connectome numerical-equivalence qualification has yet passed. Cross-engine
exact checkpoint resume is therefore deliberately rejected by dynamics checks.
Reference component identities remain unchanged for legacy loading.

Tests: 19 passed across execution backends, dynamics parameters, plastic backend,
checkpoint budget/resume checks and migration benchmark. Tiny-circuit tests cover
signed propagation, no plastic double-counting, index overflow, neural reset,
reordered independent RNG rows, spike counts and voltages. This is not real GPU
qualification or a speedup claim.

The running baseline loaded the pre-CSR code before these edits. Its original
harness writes source identity at completion, so the sidecar
`reference-source-before-edits.json` records the actual pre-edit Python digest.
Use that sidecar when interpreting this first baseline. New harness invocations
capture source identity before construction and flag changes during measurement.
The harness also accepts `--execution torch_csr`; it saves effective config as
well as the source TOML. Do not overlap its GPU measurements with session 70364.

## Measured CSR training decision throughput

Both jobs completed successfully (sessions 70364 and 38762 are terminal).
On the RTX 4080, identical source-complete settings, 10 warm-up decisions,
three restored repetitions of 100 decisions each:

| Execution | Decisions/s across trials | Peak allocated MiB | Repeat final weights |
|---|---|---|---|
| Reference COO | 0.7702, 0.7734, 0.7787 | 477.54 | identical |
| PyTorch CSR32 | 2.6818, 2.6990, 2.7409 | 483.84 | identical |

CSR delivers approximately 3.5× throughput for workload B. Both implementations
also ended with the same weight SHA-256 (`af4c6f5418f1c02235bc9eac46b7b8b219c5d13c7a028fa28dc9e01f2393562b`).
This does not prove full trajectory equality or qualify default rollout.
CSR peak allocation is slightly higher despite smaller matrix storage; further
workspace/memory profiling is required. Evidence is in
`artefacts/performance-migration/20261010-csr-v1/training.json` and the reference
file above. The CSR harness flags source edits made during measurement; its
process had loaded the original CSR/reference loop before the native hook edit.

## Native extension build and installation (experimental)

Implemented an optional PyTorch CUDA extension and `cuda_native` selection.
The first native stage fuses neuron updates, preserving FP32 operation order,
int16 refractory behavior, delay timing, threshold/reset and counts. Fixed CSR
recurrence remains PyTorch/cuSPARSE; orchestration remains in Python pending
measurements and further implementation. `--fmad=false`; no fast-math or reduced
precision. Native requests fail explicitly if the device/toolchain is unavailable.
The `native` extra supplies Torch and Ninja; compiler packages remain explicitly
matched to Torch's CUDA version. CUDA source is included in package data.

Current reproducible dependencies:

```bash
.venv/bin/python -m pip install -e '.[native]'
.venv/bin/python -m pip install \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-cuda-cccl==13.0.85
```

CUDA 13.0 rejects the system GCC 16. The official Arch GCC 14 compiler and
matching support package were extracted into `artefacts/toolchains/gcc14`,
without changing the system installation:

```bash
curl -fsSL 'https://archive.archlinux.org/packages/g/gcc14/gcc14-14.3.1%2Br416%2Bg44d5743651c4-2-x86_64.pkg.tar.zst' -o /tmp/flylatro-gcc14.pkg.tar.zst
curl -fsSL 'https://archive.archlinux.org/packages/g/gcc14-libs/gcc14-libs-14.3.1%2Br416%2Bg44d5743651c4-2-x86_64.pkg.tar.zst' -o /tmp/flylatro-gcc14-libs.pkg.tar.zst
mkdir -p artefacts/toolchains/gcc14
tar -xf /tmp/flylatro-gcc14.pkg.tar.zst -C artefacts/toolchains/gcc14
tar -xf /tmp/flylatro-gcc14-libs.pkg.tar.zst -C artefacts/toolchains/gcc14
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9
.venv/bin/python -m flylatro.fly.native_build
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_native_cuda.py -q
```

Other platforms should supply their own compatible compiler and CUDA_HOME;
this archived compiler extraction is specific to this Arch machine. The loader
also discovers the installed CUDA 13 pip toolkit. It uses a content-addressed
local build directory, records compiler/CUDA/Torch/source/flags, and creates a
local link to the versioned CUDA runtime when the pip wheel omits the unversioned
link required by the linker. A build succeeded with GCC 14.3.1 and NVCC 13.0.88.
See [PyTorch extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
for toolkit and build requirements. Native GPU tests compare all six state
buffers for 150 updates and full decisions with reordered independent RNG rows;
real-connectome native qualification remains required.

Native GPU tests completed: **3 passed**. The first full-connectome native
training benchmark is live in exec session **67130**, writing
`artefacts/performance-migration/20261010-native-v1/training.json` (same config,
10 warm-up, 100 decisions × 3 repetitions, `--execution cuda_native`, and the
CC/CXX/TORCH_CUDA_ARCH_LIST environment above). Revalidate this handle/output
before starting another GPU measurement. No native throughput result yet.

## Native result and buffered runtime

Session 67130 completed: native fused-LIF training measured **3.0461, 3.0630,
3.0165 decisions/s** (roughly 3.9× the reference), peak allocation 483.84 MiB.
All trial final weight hashes match CSR and reference. Evidence:
`artefacts/performance-migration/20261010-native-v1/training.json`.
These measurements precede the subsequent buffer/index reuse changes below;
repeat final sustained comparisons after those changes are complete. No GPU
measurement remains live as of this update.

Implemented bounded synchronous log buffering (default 256 KiB per stream,
1-second opportunity to flush, configurable `--log-buffer-bytes` and
`--log-flush-seconds`). Oversized records write directly after pending records;
no unbounded queue or silent writer thread errors. A checkpoint explicitly
flushes metrics/action/reinforcement/episode/plasticity text streams first.
ExitStack closes all audit writers even if another close fails. Partial write
failures are latched and never retried on close, avoiding duplicated prefixes;
an I/O failure propagates and cannot produce a successful final manifest.
Final hashes follow closure. TensorBoard's own bounded async writer retains its
buffering and is flushed at checkpoint barriers and close.

Dashboard snapshots and persisted dashboard JSON now update at the configured
refresh interval; durable per-decision actions and episode events remain intact.
SIGINT during training is deferred to the completed/audited decision boundary,
then produces `plastic-checkpoint-interrupted.pkl`, status `interrupted`, and
exit code 130. Previous code did not implement that explicit checkpoint path.
The signal handler is scoped and restored. Production safety stops are retained.
Tests across buffered runtime, safety, checkpoint budget and dashboard: **43
passed**, including a signal raised mid-step, checkpoint audit barriers,
Unicode ordering, bounded buffers and partial-write failure handling.

Native reset now retains neural-buffer addresses and generator objects for a
stable batch, reseeds each row exactly as before, and copies efficacies into
owned stable storage. Batch-size changes reallocate explicitly. Device-side
readout/root-ID tensors and plastic edge readout positions are cached rather
than recreated each decision. Native GPU state/RNG/buffer qualification: **4
passed**. These are prerequisites for graph capture, not a graph implementation.
Full CLI timing, background checkpoint assessment, graph capture, stronger
source-complete trajectory/resume qualification and remaining workload/batch
measurements remain outstanding.

## CUDA Graph prototypes and stronger qualification

Added explicit `cuda_native_graph` (complete decision) and
`cuda_native_graph_blocks` (four delay-ring cycles per block, plus final tail).
Both replay only within one decision. Fixed device probability and efficacy
buffers receive current inputs; neural reset and per-row seed derivation remain
outside replay. Generators are registered for capture (Torch 2.14 reports that
explicit registration is now a deprecated no-op because capture tracks them).
Capture warm-up restores all neural buffers and RNG states before actual replay.
Batch/efficacy shape changes invalidate graphs; delay-aligned blocks restart at
the correct ring slot. Unsupported event recording/component timing fails
explicitly and directs the caller to `cuda_native`, rather than silently falling
back. See [PyTorch RNG graph-state documentation](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html).

GPU qualification: **6 passed**, including both graph strategies, full neural
state and RNG-state equality to eager execution, changed sensory inputs and
efficacies, fresh decision seeds, row reorder, batch resize and recapture.
Real graph throughput is not yet reported: the block-graph 10-warm-up,
100-decision × 3 training benchmark is live in exec session **25240**, output
`artefacts/performance-migration/20261010-graph-blocks-v1/training.json`.
Do not start overlapping GPU work before revalidating this session.

Prepared `scripts/qualify_execution.py` for diagnostic full-connectome per-step
action, environment-hash, neural-buffer, learned-state and RNG comparisons,
plus actual file checkpoint resume and initial-state restore after graph reuse.
It is not yet executed. Added `python -m flylatro.benchmarks.neural` for controlled
workload A, excluding encoding/environment/learning with explicit warm-up,
repetitions, independent seeds, reset/simulation timing and graph capture cost.
This harness is not yet qualified or measured; do not treat its existence as
workload-A completion.

Block-graph session 25240 completed: **3.1186, 3.1070, 3.1064 decisions/s**,
peak allocated 475.16 MiB, one capture taking 0.0319 seconds; identical final
weight hashes to all prior implementations. This is a small additional gain
above eager native, not evidence of a large dispatch bottleneck. The full-decision
graph benchmark is now live in session **20055**, writing
`artefacts/performance-migration/20261010-graph-full-v1/training.json`; same config,
10 warm-up and 100 × 3 measured decisions. Revalidate it before further GPU work.

Inspection follow-up: `EvidenceIdentity`/reward-free cache currently include
backend dynamics version but not the native build identity stored in stack
components. Add explicit native implementation hashing to evidence/cache keys
before declaring native calibration caching qualified. Preserve legacy reference
cache readability where the new optional identity is absent.

## Source-complete numerical audit (2026-10-12)

Full-decision graph measurement completed: **3.1088, 3.1052, 3.1018 decisions/s**,
475.16 MiB peak allocated and 0.5081 seconds capture. Block graphs have comparable
throughput with much lower capture cost and are the current graph candidate.

The stronger 16-decision audit at `20261010-trajectory-v1/qualification.json`
returned exit 2. All five engines matched actions, observable simulator state
hashes, metrics, learning-state hashes, RNG state and spike counts. However,
conductance/delay/voltage buffers were not bitwise identical from decision 5
onward. Even reference self-resume and initial restore fail the full-state exact
comparison. This contradicts any claim of complete bitwise neural equivalence;
previous final-weight agreement is narrower evidence.

A controlled repeat probe at `20261012-repeat-v1/diagnosis.json` restored the
same learned checkpoint, inputs and seed three times. The reference had up to
1.53e-5 conductance differences and 3.81e-6 delay differences; native-v1 also
had small delay/conductance differences. Spike, refractory and count buffers
were identical in that probe. Floating-point plastic scatter accumulation is
the suspected source; no comparison tolerance was relaxed to claim exactness.

A versioned native-v2 prototype replaces per-edge atomic plastic scatter with
one fixed reduction tree per postsynaptic neuron/learner. The shared fixed graph
still uses cuSPARSE. This intentionally changes the FP32 accumulation order and
requires fresh native-v2 numerical/behavioral/calibration/performance evidence.
Reference/CSR remain unchanged. Extension build is live in exec session **80318**;
GPU tests and repeated full-state resume qualification are pending.

Calibration provenance now carries an optional implementation hash, including
native build identity, and accelerated preflight checks it. Legacy reference
cache metadata remains unchanged when the optional hash is absent. Native cache
reuse rejects a missing or different build hash. Related cache, execution and
provenance tests: **29 passed**. Native build description can be queried without
compiling or loading the connectome. Preflight now selects the configured backend
version rather than hardcoding the reference backend.

Native-v2 build completed successfully, and **7 GPU tests passed**, including a
2,048-edge many-to-one reduction checked against a float64 oracle and 20 bitwise
identical repetitions. Eager and both graph variants remain exact on the small
circuits. Full-connectome native-v2 eager/graph/resume qualification is now live
in session **58684**, output directory
`artefacts/performance-migration/20261012-trajectory-native-v2`, 16 prefix steps
plus three future steps and full prefix replay for each engine. The comparison
reference in this run is explicitly **cuda_native v2**, not historical COO.
Wait for this result before claiming that native-v2 fixes full-state resume.
Next work includes native-v2 versus historical behavior/calibration, fresh
throughput, full CLI and neural/evaluation workloads, batch sizes and profiling.

## Native-v2 audit and batch measurement suite

Session 58684 finished successfully. In
`artefacts/performance-migration/20261012-trajectory-native-v2/qualification.json`,
all three native-v2 engines have exact per-decision neural/learning/RNG state,
actions and environment hashes relative to eager native-v2, exact file-checkpoint
resume, and exact initial-state replay. This is 16 prefix decisions plus three
future decisions and prefix replay per engine. It does not claim bitwise
agreement with reference-v1's nondeterministic plastic sum.

Full regression suite: **505 passed, 6 skipped** in the CUDA-hidden sandbox;
all skips are native GPU tests. The separate GPU run already passed all seven
native tests, so skips do not substitute for GPU validation.

A serial training suite is live in session **58667**, writing
`artefacts/performance-migration/20261012-native-v2-training/suite.json` and one
JSON per batch. It measures 1/2/4/8/16 independent flies at 100 decisions per fly,
three restored repetitions, plus 10 warm-up steps. Budget overrides preserve
per-learner budget, and the report includes total/per-learner throughput.
Batch 1 finished; subsequent batches are pending/running. Revalidate the session
or recorded live child PID before launching other GPU work.

Added `scripts/run_performance_suite.py` to execute training, neural or complete
CLI workloads serially, preserve exact commands/stdout/stderr and process PID,
and stop on the first failed job. Training CLI now accepts explicit `--execution`.
Prepared `scripts/profile_execution.py` for isolated timeline, timestep sparsity
and checkpoint-stage measurements, and `scripts/benchmark_frozen_workloads.py`
for repeated frozen validation episodes and complete reward-free corpus activity.
These newly prepared diagnostic/frozen scripts have passed syntax checks only;
actual runs and result inspection remain necessary. No speedup is attributed
to an unexecuted workload.

## Reusable cuSPARSE and outgoing-event qualification (2026-10-12)

The serial native-v2 suite finished successfully at all five batch sizes. With
100 vector steps × 3 restored repetitions after ten warm-up steps, aggregate
rates were: batch 1 **3.191–3.202**, batch 2 **1.402–1.426**, batch 4
**2.656–2.657**, batch 8 **3.375–3.405**, batch 16 **3.432–3.439 decisions/s**.
Final weight hashes repeat exactly within each batch; peak allocations repeat
exactly. This exposed a substantial regression for small batches, rather than
supporting an unconditional batched-speedup claim.

`20261012-cusparse-v1/probe.json` measured reusable descriptors, workspaces,
preprocessing and explicit SpMV/SpMM algorithms/layouts. Row-major SpMM CSR ALG1
is materially faster for batches 2/4/8 than PyTorch's selected path. The probe
has 14 layout-validation failures for degenerate batch-one strides; these are
not unsupported CUDA algorithms. The production `sparse_plan.cpp` accepts valid
degenerate strides, separately hashes its source, and passed GPU tests.
`cuda_native_cusparse` and `cuda_native_cusparse_graph_blocks` are explicit
candidates; no reference/native-v2 implementation was silently replaced.
`20261012-trajectory-cusparse-v1/qualification.json` passes all 16-decision
full-state trajectory, checkpoint/future resume and initial-restore checks
against native-v2. Whole-training batch-4 results in
`20261012-cusparse-training/batch-4.json` are **7.071–7.135 total decisions/s**,
**530.54 MiB** peak allocated, with identical repeated final weights.

`20261012-native-v2-timeline` identifies fixed cuSPARSE recurrence as about 92%
of actual CUDA kernel time (the profiler's aggregated parent/child entries must
not be added together). `20261012-native-v2-sparsity/profile.json` measured
**0.02364% mean firing neurons per timestep**, 0.04025% p95, 0.04883% maximum.
This justified the outgoing-event prototype rather than further LIF tuning.

`20261012-event-probe-v1/probe.json` includes zeroing, compaction, outgoing-edge
traversal, atomic contention and graph replay. It matches CSR on all **97 real
timestep samples**, and on controlled densities through 50%, at batch sizes
1/2/4/8/16. Actual batched timing samples duplicate one fly, so these kernel
measurements do not establish independent closed-loop batch throughput. For
one fly, the timed actual sample is **7.18 us** versus **288.22 us** CSR. At 50%
activity the event kernel is slower (**476.78 us** versus **288.27 us**); this
approach is conditional on activity and has no silent activity-based fallback.

The fixed anatomical weights are exact integers and the maximum incoming
absolute sum is **88,389**, below 2^24. The event boundary checks both properties,
without rounding or pruning. Every binary-spike partial FP32 atomic sum is
therefore exactly representable irrespective of compaction or accumulation
order. Fractional plastic contributions still use native-v2's separate fixed
reduction tree. Nonintegral fixed weights or excessive incoming sums produce
an explicit error. `propagate_once` rejects nonbinary activity in this backend.

`cuda_native_event` and `cuda_native_event_graph_blocks` remain explicit
experimental candidates until calibration and sustained runtime qualification.
`20261012-trajectory-event-v1/qualification.json` passes all neural buffers,
learning/RNG states, actions, environment hashes and exact resume/restore against
native-v2 over the same 16+3-decision audit. Native GPU tests: **11 passed**.
The event batch suite is live in session **35712**, output
`20261012-event-v1-training`; inspect its live handle before other GPU work.
Initial complete-training rates are 28.96–29.11 decisions/s for one fly,
42.78–42.86 total for two, and 53.59–53.62 total for four. These are 100-step
repetitions, not yet a long-run acceptance claim; fresh calibration, long CLI
runs and reference behavioral comparison still remain.

Routine processing now retains owned population readouts on-device and lazily
materializes diagnostic NumPy arrays when requested. Motor readout and five
compact GPU health reductions cross to CPU; edge activity remains device-side.
Constant population indices and trainer motor columns are cached. A regression
checks that retained diagnostics survive reuse of neural buffers and still
match the compact health values. NumPy protocol recursion found by this new test
was fixed before subsequent qualification. Training histories now retain at
most 512 episodes by default, with lifetime return sum, best return/Ante and
history offset in resumable state; durable CLI episode records remain complete.
Legacy checkpoints derive these summaries before trimming. The related trainer,
dashboard and pipeline tests passed (39 tests).

Checkpoint-stage profiling in `20261012-native-v2-checkpoint/profile.json` found
about **0.20 ms snapshot capture**, **2.4 ms top-level serialization**, **0.70 ms
file hashing**, **27 ms complete save**. The serialization array also contains
nested Torch pickle calls because the diagnostic wraps the shared pickle module;
only the outer call represents complete serialization. Background serialization
is not justified by these current sizes/cadence (about 0.1% overhead at 100
single-fly decisions between checkpoints). Atomic synchronous commit and audit
flush barriers remain the reliable path. Reprofile larger/long-history snapshots
before adding background writers.

## Event suite completion and sparse storage ownership (2026-10-14)

The event suite in session 35712 completed successfully with no source changes
reported during measurements. Batch 1: **28.958–29.107**, batch 2:
**42.784–42.857**, batch 4: **53.593–53.618**, batch 8: **57.347–57.378**, batch 16:
**40.228–40.234 aggregate decisions/s**. Per-learner rates are the aggregate
rate divided by batch size, not shared learning experience. All repeated weight
hashes and peak allocations agree. Batch 16 regresses versus batch 8; an unlimited
batch-speedup claim is not supported.

`20261012-sparse-storage-v1/probe.json` found that values views retained old COO
index arrays even after CSR conversion. Incoming values ownership reduced cold
allocation from **959,311,360 to 641,527,296 bytes**; outgoing ownership reduced
it again to **323,453,952 bytes**. Both matrix payloads are **158,746,888 bytes**,
unchanged by ownership. The earlier event measurements' 932–1,199 MiB peak
allocation is genuine but predates this correction. Both CSR and outgoing
constructors now clone only the values payload, releasing discarded layouts;
all **11 native GPU tests pass** after the change. Fresh memory and sustained
performance measurement are still required before substituting lower memory
figures into a training performance claim.

The next sustained suite is live in session **11146**, directory
`artefacts/performance-migration/20261014-event-v1-sustained`, 1,000 vector steps
× three restored repetitions after ten warm-up steps for batch sizes 1 and 8.
It uses the fresh `configs/plastic-v783-source-complete-event-performance-v1.toml`:
only execution, experiment name and the measurement exposure budget differ from
the original pilot. Original connectome, sensory/motor mappings, learning rule,
RNG method, 100 ms duration, timestep, rewards and safety thresholds are retained.
The default backend remains the reference.

A new `--validate-existing` option in motor calibration remeasures the configured
pool and normalization quality under current neural execution without selecting
new pools or changing normalization. This is needed for performance migration
qualification: rerunning pool selection would silently change the experiment.
It writes a fresh evidence report and, on success, an identical mapping artifact
copy. The API returns the same mapping object, retains all existing gates, and
fails on silent activity. Unit tests and the actual accelerated corpus run are
pending at this point.

## Fresh evidence and full-runtime results (2026-10-14)

`20261014-event-v1-calibration` finished representation pre/post, KC reachability,
existing-motor revalidation, integrated plasticity and chosen-action specificity.
All measured gates pass, preserving the original motor pools and normalization.
Initial preflight first failed solely because the generated timing-summary
adapter omitted `batch_size`, `execution_mode` and `peak_memory_bytes`. The adapter
and derived summary were corrected using the original measured trial fields;
the original malformed summary remains `benchmark-schema-v1.json`.
`preflight-v2.json` has **no FAIL gates**. Its WARN gates explicitly concern absent
matched controls, tiny real learning-run evidence and a learning protocol; they
are not silently claimed as new scientific learning qualification.

`20261014-independent-event-v1/qualification.json` passes exact comparison of
four genuinely independent, closed-loop learners with four separately trained
flies, preserving learner training/motor/fly-seed identity. All 16 decisions
match actions, environment hashes, every plastic state tensor, all six neural
buffers and each row's generator state. Aggregate metrics inherently differ and
are not confused with per-learner state comparisons.

`20261014-event-v1-cli/suite.json` completed three 1,000-decision CLI invocations.
Each enables durable buffered logging, dashboard snapshots (2-second refresh)
and periodic checkpoints (50 decisions). Runtime is **28.204–28.245 decisions/s**;
whole subprocess wall time including startup is **37.406–37.482 seconds** per
1,000 decisions. All final weights have SHA-256
`fcc8d99ddb1fe6c570a51dfcab70bd237179104dc9685e9e47480d01ccb6df58`.
The CLI's cold-start-inclusive GPU peak is **1,664.09 MiB**, whereas end allocation
is **323.94 MiB**; the lower workload-B peak resets after warmup and must not be
substituted for a cold CLI peak.

The sustained batch-8 suite also completed: **53.791–54.540 aggregate decisions/s**,
**450.70 MiB** repeated peak allocated, **433.49 MiB** end allocated, identical
repeated final weights and no source changes during timing. Batch-1 sustained
rates were **28.881–29.102** at **326.10 MiB** peak. These are 1,000 vector steps
per restored repetition, preserving independent learner budgets and safety.

A serial frozen-workload suite is now live in session **51761**, directory
`20261014-frozen-v1`, measuring reference batch 1 plus native-event batches
1/2/4/8/16 for evaluation and reward-free corpus activity (three repetitions).
Check its live handle before any competing GPU job. Preparation-cache work is
being staged outside source files while these measurements run; benchmark code
and model execution are not changed underneath the live trials.

## Full frozen audit, preparation caches and interruption (2026-10-14)

`20261014-frozen-v2/qualification.json` passes exact complete frozen transitions
(actions, rewards, observable state hashes, synthetic reinforcement, terminal
flags) against the reference at native batch sizes 1/2/4/8/16. Every native
repetition has the same transition hash. Complete recorded corpus activity also
matches reference exactly at all five widths. No weights or eligibility traces
change during frozen evaluation. Version 1 timing supplied three reference
repetitions; version 2 adds complete transition hashing, CUDA trial events,
reservation peaks, full configuration/software and source-change detection.
Its reference has one additional functional/audit repetition; native trials
still have three. Do not treat episode-summary equality alone as this stronger
trajectory qualification.

Opt-in `FLYLATRO_ARTIFACT_CACHE_DIR` and `FLYLATRO_LAYOUT_CACHE_DIR` now provide
verified read-only artifact preparation and fixed incoming/outgoing layouts.
Source bytes and cache payloads are verified; keys bind relevant immutable
source/topology/shuffle/backend/build/precision/index-width/Torch/NumPy/device
identities. Device layouts own their values, and CSR bounds/plastic separation
are checked. Mutable hand-built artifacts are not cached. No learned values,
noise or stochastic training activity are stored. New meaningful tests cover
corruption, stale source, read-only ownership, range/width/separation, and warm
construction that bypasses COO/coalescing without sharing mutable anatomy.

`20261014-preparation-v2` compares disabled, cold and warm preparation in one
persistent process and passes exact 16-decision full-state trajectories. Warm
cache reuse covers both graph directions. Startup GPU peak falls from 1,664 MiB
to about 311 MiB. The verified warm cache was slightly slower than uncached
preparation on this machine (~1.35 vs ~1.11 seconds); it remains opt-in and is
not included in a claimed speedup. Cold population requires extra writing and
hashing. Preparation-stage timing is not cold-process startup timing.

CLI checkpoint provenance reuse captures fixed software/Git identity once per
invocation while preserving fresh state/weight hashes and counters for every
save. `20261014-checkpoint-provenance-v1` measured about 23.7 ms complete saves,
a few milliseconds below earlier complete-save profiling. Serialization remains
synchronous; no mutable GPU tensor is serialized while training continues.

`20261014-interrupt-v1/qualification.json` passes a real native-GPU SIGINT at
decision 51 (exit 130), durable audit/checkpoint alignment, and exact continuation
to decision 100. Action/reinforcement/episode logs are byte-identical to the
uninterrupted run; final weights agree exactly. Whole-fast-state finite checks
now include conductance, voltage and delay buffers, so silent NaN neurons cannot
escape activity checks. Three injected-corruption tests pass.

The final serial suite is live in session **82181**, directory
`20261014-final-v1`, with exact commands/stdout/stderr/PIDs in `suite.json`.
It includes reference workload A, native eager-event training, sparse-layout and
corrected production cuSPARSE probes, actual event-graph timeline, strict
100-decision historical comparison, reference/current full CLI, complete
single/eight-fly CLI runs and artifact audits. Inspect its live handle before
starting another GPU job. A separate named 50 ms model-variant runner is prepared
but not launched; it requires fresh calibration and learning/frozen evaluation
and its results will not count toward equivalent-model acceleration.

## Final acceptance (2026-10-14)

The final serial suite completed. Current reference CLI rates (100 decisions,
three repetitions) are 0.765400 / 0.765424 / 0.766046 decisions/s. Current native
CLI rates (1,000 decisions, three repetitions) are 27.832936 / 27.867536 /
28.027826, with identical final learned hashes. Both use the same 100 ms model,
dashboard every two seconds, buffered logs and checkpoints every 50 decisions.
The median rate ratio is 36.408×. Different exposure window lengths are explicit.

Complete native single/eight-fly runs reach their declared 1,100 / 8,800 decision
budgets at 27.992871 / 55.591655 aggregate decisions/s. Eight-fly throughput is
6.948957 per learner. Dashboard refresh is one second; all durable logs and
checkpoint cadence are enabled. `20261014-final-v1/cli-audit.json` passes all
109 checkpoints across five native CLI runs. Allocation remains 323.94 /
433.49 MiB for one/eight flies; startup-inclusive peak is 1,664.09 MiB.

Strict 100-decision historical comparison differs only in neural-buffer hashes
on 96 decisions; actions, observable game state, metrics, learned state and RNG
agree throughout. Native replay of the entire prefix and checkpoint continuation
are exact. The reference itself has neural-buffer self-replay differences.
All 92 final cuSPARSE algorithm/layout/preprocessing probes have exact oracle
outputs and zero maximum error. These facts do not justify claiming identical
historical neural floating-point state or exact cross-backend checkpoint resume.

The separately named 50 ms variant completes fresh unchanged-mapping calibration,
1,000 decisions of CLI learning and 16-seed frozen trained evaluation. Its initial
preflight has no failed gates; scientific protocol warnings remain. B pilot rates
are 46.233–46.386 and full CLI rate is 44.354 decisions/s. It has zero blind clears
and wins. The primary 100 ms trained candidate and initial-weight baseline also
have zero blind clears/wins on the same 16 validation seeds (Ante-1 target).
No claim of better learning or equivalent-model gain is attached to 50 ms.

The first public resume audit exposed deferred-readout hashing incompatibility;
its failure remains in `20261014-public-audit-v1`. The fix materializes actual
diagnostic values for hashing, with a regression that matches eager array digests
and detects changed values. The real source-complete public full-state resume
audit passes in `20261014-public-audit-v2/resume-report.json`. This is an analysis
fix and does not change measured neural execution or learning.

Final CPU regression after the fix: **521 passed, 10 skipped**, in 30.41 seconds.
The separate GPU-native suite: **11 passed**. The rebuilt wheel installs offline
outside the checkout, all 137 packaged Python/native source files match the
worktree, and all three extensions compile/load from installed sources in fresh
directories. Build/load times are ~29.46 / 12.67 / 27.90 seconds for neuron,
cuSPARSE and event extensions. Compiler/build identities are in
`20261014-public-audit-v2/installed-native-qualification.json`; final wheel is in
`20261014-wheel-v2`. Final whitespace checks pass.
