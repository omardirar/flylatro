# Flylatro performance migration

The migration and its acceptance checks are complete as of 2026-10-14. Backend
selection remains explicit and the reference remains the default. The recommended
qualified acceleration is `cuda_native_event_graph_blocks` for this integral
source-complete anatomy and measured activity range.

## What changed

Explicit execution factories now serve training, evaluation and calibration.
Checked CSR32 storage reduces index traffic. Optional C++/CUDA extensions fuse
FP32 neuron updates, make plastic accumulation deterministic, reuse cuSPARSE
descriptors/workspaces, and propagate anatomical activity through outgoing edges
of firing neurons. CUDA Graph blocks remove repeated Python timestep dispatch.
The custom outgoing kernel followed measured whole-network spike sparsity;
cuSPARSE remains available for comparison and denser activity.

Neural buffers, generators, index selections and motor metadata are reused.
Eligibility remains on the GPU. Routine training transfers motor outputs and
compact health summaries, including whole-fast-state finiteness. Detailed
diagnostics retain owned activity and materialize it on request.

Bounded logging buffers preserve ordered audit records and flush before
checkpoints/final hashes. The dashboard receives compact updates at a configured
cadence. Episode histories are bounded with resumable lifetime statistics and
complete durable episode logs. Checkpoints capture consistent decision-boundary
state and publish atomically; fixed software/Git provenance is reused within an
invocation. Optional verified preparation/layout caches reduce startup GPU peak.

The existing Rust simulator, complete 139,255-neuron / 19,773,733-connection
anatomy, mappings, reward/punishment, eligibility/efficacy equations, 100 ms
duration, 0.1 ms timestep, FP32 precision and independent learners are retained.
There are zero external trainable policy, decoder or critic parameters.

## Machine and measurement

The measured machine has an AMD Ryzen 7 7800X3D (8 cores / 16 threads) and an
NVIDIA RTX 4080 (16,376 MiB, capability 8.9), driver 610.57.04. Software is Python
3.14.8, Torch 2.14.1+cu130, NumPy 2.5.3, CUDA 13.0, nvcc 13.0.88 and GCC/G++
14.3.1. The native kernels use `-O3`, `--fmad=false`, and no fast math.
Hardware capture: `artefacts/performance-migration/20261014-hardware/hardware.json`.

GPU jobs run serially. A/B measurements use explicit warmup and three restored
repetitions. B uses production training/safety checks, without logging, dashboard,
periodic checkpoints, detailed recording or the synchronizing component profiler.
C enables the dashboard, durable logs and periodic checkpoints. D has frozen
weights and measures actual active decisions separately from padded lanes.
CUDA trial events include CPU dispatch idle gaps; intrusive timelines are separate
from throughput measurements. Artifact JSON records configuration, implementation,
source identity, memory and latency distributions where applicable.

## Completed equivalent-model results

Rates are decisions per second. Batched rates are aggregate independent exposure,
not progress by one shared learner.

| Workload | Reference, one fly | Event graph, one fly | Event graph, eight flies |
|---|---:|---:|---:|
| A: fixed-input neural simulation | 0.773–0.774 | 38.620–38.725 | 61.750–61.942 |
| B: training, 1,000 vector steps × 3 | Historical baseline 0.770–0.779 (100 steps × 3) | 28.881–29.102 | 53.791–54.540 |
| C: current CLI, 1,000 decisions × 3 | 0.765–0.766 (100 decisions × 3) | 27.833–28.028 | 55.592 (one full 8,800-decision run) |
| D: frozen evaluation, full transition audit | 0.768 | 29.321–29.388 | 45.576–45.587 |
| D: reward-free calibration corpus | 0.767 | 29.746–29.858 | 56.076–56.165 |

The completed B result is approximately **37.5×** the median historical baseline
at identical model settings. The eight-fly sustained result is **6.724–6.818
decisions/s per learner**. These measurements predate the final extra whole-state
finite reductions and checkpoint-provenance reuse; final C acceptance measures
the current implementation at **36.4×** the median current reference CLI rate.
The repeated C reference/native runs use the same model, seeds, dashboard (2 s),
log buffering and 50-decision checkpoint cadence, with different measured
exposure lengths (100 versus 1,000). The historical 1,000-decision CLI corroborates the
shorter repeated reference B baseline, but these are different workload scopes.

Source artifacts are `20261014-event-v1-neural`, `20261014-event-v1-sustained`,
`20261014-event-v1-cli`, `20261014-frozen-v2`, `20261010-harness-v1`, and the
original `runs/exploratory-20261010-weak-failure/training-dashboard.json`, all
under the repository's artifact/run roots. The final current reference A and
eager-event B results are in `20261014-final-v1`. A complete current 1,100-decision
single-fly CLI runs at 27.993 decisions/s; the eight-fly CLI runs at 55.592
aggregate / 6.949 per learner. Both enable dashboard refresh every second,
buffered durable logs and checkpoints every 50 decisions per learner. Process
wall times including import/startup are 41.331 / 160.316 seconds; training-path
wall times are 39.296 / 158.297 seconds. The measured single-fly CLI retains
~323.94 MiB allocated GPU memory; cold construction peak is 1,664.09 MiB.
The complete CLI artifact audit passes all 109 checkpoints across five native
runs, including exposure, weights, next observations and durable log hashes.

Short B measurements (100 vector steps × 3) cover all requested widths:

| Flies | Aggregate decisions/s | Decisions/s per learner |
|---:|---:|---:|
| 1 | 28.958–29.107 | 28.958–29.107 |
| 2 | 42.784–42.857 | 21.392–21.428 |
| 4 | 53.593–53.618 | 13.398–13.404 |
| 8 | 57.347–57.378 | 7.168–7.172 |
| 16 | 40.228–40.234 | 2.514–2.515 |

Evidence: `20261012-event-v1-training`. Sustained eight-fly measurements are
slower than this short sample and take precedence for planning. Sixteen flies
regress in aggregate throughput; larger batches are not automatically better.

## Memory, startup and checkpoint costs

At the end of each sustained B repetition, allocated GPU memory is exactly
323.94 MiB for one fly and 433.49 MiB for eight flies. Measurement peaks are
326.10 and 450.70 MiB respectively; reserved memory is 700 and 894 MiB. CPU
process peak RSS is approximately 1,846 MiB. Final learned hashes agree across
all three restored repetitions. Stability is established over these measured
1,000-step windows, not an unlimited lifetime run.

Each fixed CSR32 direction has a 158,746,888-byte payload. Values now own their
storage instead of retaining discarded COO/int64 arrays. Both incoming and
outgoing layouts are retained for the explicit event implementation. Payload size
is distinct from total allocated/reserved GPU memory and from startup peak.

E: in one persistent process, uncached / cold cache / verified warm cache startup
takes 1.110 / 2.440 / 1.347 seconds. Warm startup GPU peak drops from 1,664.09 to
310.93 MiB. All three produce identical 16-decision full-state trajectories.
The cache is slower on this machine and remains opt-in. These preparation-stage
numbers exclude fresh-process imports/build and are not a cold-process speedup.
Evidence: `20261014-preparation-v2/qualification.json`.

Checkpoint median stages are 0.153 ms plastic capture including device transfer,
0.211 ms full snapshot capture, approximately 2.40 ms outer serialization,
0.685 ms file hashing and 23.70 ms complete save. Nested serialization calls
are included in the raw profiler; their microsecond median is not the cost of
serializing the whole checkpoint. Fixed provenance reuse saves a few milliseconds
relative to earlier ~27 ms complete saves. At a 50-decision cadence, this does
not justify background serialization/queue complexity. Evidence:
`20261014-checkpoint-provenance-v1/profile.json`.

## Numerical and behavioral qualification

The native-v2 plastic reduction uses a fixed tree in place of the reference's
floating-point atomic accumulation. This is a versioned numerical implementation
change. Exact cross-backend checkpoint resume is rejected. Legacy checkpoints
remain supported by the reference backend. See the [compatibility table and
migration guide](PERFORMANCE_MIGRATION_GUIDE.md).

The strict current 100-decision historical comparison finds neural-buffer hash
differences on 96 decisions. Actions, observable game states, metrics, persistent
learning state and RNG hashes agree on all 100. The reference itself has small
neural-buffer replay/resume differences; native replay of all 100 decisions and
three checkpoint continuations is exact. Evidence:
`20261014-final-v1/historical-comparison/qualification.json`.

Native eager/full/block modes, reusable cuSPARSE, and outgoing event execution
pass real source-complete full-state trajectory/resume comparisons against the
versioned native baseline. Four actual independent closed-loop learners match
four separate learners. Frozen actions, rewards, state transitions, reinforcement
and terminals match reference at widths 1/2/4/8/16 in every repetition; corpus
activity also agrees exactly. No acceptance test was weakened to conceal neural
differences. Real SIGINT at decision 51 returns 130 and resumes to 100 with
byte-identical gameplay/reinforcement/episode logs and identical final weights.

Fresh representation, reachability, unchanged motor mapping/normalization,
integrated plasticity and specificity calibration pass. Preflight has no failed
gates; matched controls and scientific learning-protocol qualification remain
warnings. Performance qualification does not prove improved learning.

On 16 held-out validation seeds with frozen weights and an Ante-1 target, the
1,100-decision trained 100 ms candidate completes 126 decisions and has **zero
blind clears and zero wins**. Its checkpoint's initial-weight baseline completes
122 decisions and also has zero blind clears/wins. Mean skipped blinds differ
(1.5625 trained versus 0.625 initial), so learning changed behavior without
establishing task success. Evidence is in `20261014-final-followup-v1` under
`trained-100ms-evaluation` and `initial-100ms-evaluation`. During training, the
single-fly run completes 120 episodes and the eight-fly run 964 episodes; all
learners have zero wins and best Ante 1.

## Limits and rejected approaches

CSR and fused neurons alone deliver about 2.7–3.2 decisions/s for one fly.
PyTorch CSR batched SpMM regresses at widths 2/4; reusable cuSPARSE improves
four-fly B to 7.071–7.135 aggregate decisions/s, but event execution is faster
at measured activity. Full-decision native-v2 graph capture costs ~0.508 seconds
versus ~0.032 seconds for blocks, with little throughput difference, so blocks
are preferred. Current eager event B is 17.443–17.526 decisions/s; graph replay
substantially reduces dispatch overhead.

Only ~0.02364% of neurons fire per timestep on average. Actual sparse-input
outgoing propagation takes ~7.18 μs versus ~288.22 μs for the compared CSR call.
At 50% firing density it takes ~476.8 μs versus ~288.3 μs for CSR and is slower.
Outgoing FP32 atomics are order-independent only because fixed weights are
integral and each incoming absolute sum is checked below 2²⁴ (actual maximum
88,389). Fractional plastic contributions retain the deterministic native tree.
Unsupported fractional fixed anatomy is rejected, never rounded or pruned.

The current intrusive event timeline's largest neural kernels are deterministic
plastic reduction (~7.56 ms per decision), outgoing recurrence (~4.33 ms), LIF
(~2.83 ms) and random generation (~2.47 ms). Synchronization, graph replay and
CPU work remain visible. The profile is a diagnostic, not a throughput run.
Rust encoding/interface rewrites were not justified by the remaining neural
cost. FP16 learned efficacy would erase representative 1e-5 updates near 1.0;
no reduced-precision speedup is claimed.

## Separately named 50 ms model variant

`configs/plastic-v783-source-complete-event-50ms-qualified-v1.toml` changes only
the primary model's decision duration and explicit experiment name, plus fresh
calibration evidence paths. It retains the original mappings/normalization,
reward, timestep, gain and learning rule. Representation, reachability, motor
revalidation, integrated plasticity and specificity pass; initial preflight has
no failed gates and the same three scientific-protocol warnings.

Its B pilot (100 steps × 3 after warmup) measures 46.233–46.386 decisions/s.
The full CLI stage (1,000 decisions, dashboard/logs/checkpoints enabled) measures
44.354 decisions/s, approximately 1.59× the current 100 ms CLI. This is a changed
model and is excluded from equivalent-model acceleration. Frozen trained
evaluation on 16 validation seeds completes 178 decisions at 90.875 decisions/s
with **zero blind clears and zero wins**. Fresh calibration is satisfactory but
these results do not justify adopting the shorter duration for learning.
Evidence: `20261014-duration-50ms-v1`.

## Reproduction and final acceptance

The [migration guide](PERFORMANCE_MIGRATION_GUIDE.md) provides supported compiler,
native installation, explicit backend selection, cache controls, benchmark and
long single-fly/eight-fly commands. A built wheel has been verified to include
all CUDA/C++ sources, all three build helpers, and optional `native` dependencies.
The final serial suite records exact commands and exit codes in
`artefacts/performance-migration/20261014-final-v1/suite.json`.

After native installation, the exact qualified long-run commands are:

```bash
export CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14"
export CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=1
.venv/bin/python -m flylatro.learning.cli \
  --config configs/plastic-v783-source-complete-event-qualified-v1.toml \
  --run-dir runs/performance-event-single-new \
  --heavy --dashboard --dashboard-refresh 1 --no-tensorboard
.venv/bin/python -m flylatro.learning.cli \
  --config configs/plastic-v783-source-complete-event-eight-v1.toml \
  --run-dir runs/performance-event-eight-new \
  --heavy --dashboard --dashboard-refresh 1 --no-tensorboard
```

Use fresh destinations and run GPU jobs serially. These commands declare 1,100
decisions per learner, matching the complete acceptance runs; they do not waive
exposure or safety checks for a larger budget. Repeated before/after B commands
and A/C/D profiler commands are in the migration guide.

The migration-acceptance CPU regression passes 521 tests with ten CUDA tests skipped; the
separate GPU suite passes all eleven native tests. The public native full-state
resume CLI initially exposed a diagnostic hashing incompatibility with deferred
readouts. The fix materializes those values for hashing, and a meaningful new
regression test plus seven relevant tests pass. The actual public audit now
passes exact saved/restored full trainer state and all future actions/activity/
learning/RNG/safety state in `20261014-public-audit-v2/resume-report.json`.
The failed audit is retained in `20261014-public-audit-v1`. The diagnostic fix
does not change neural execution; throughput measurements retain their recorded
source identity.

The rebuilt wheel installs offline outside the checkout. All three native
extensions compile and load from its installed sources with fresh build
directories. All 137 packaged Python/native files matched the worktree at acceptance;
build identities and timings are preserved in
`20261014-public-audit-v2/installed-native-qualification.json`. Final regression
output is `20261014-public-audit-v2/final-cpu-regression.stdout`.
The [requirement-by-requirement audit](PERFORMANCE_ACCEPTANCE.md) records coverage
and the limits of each claim. No default change or better-learning claim is made.
