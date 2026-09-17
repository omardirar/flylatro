# Flylatro V1 implementation status

This ledger tracks the continuation brief's 18 requirements. `PLAN.MD` remains
the research specification. `NEEDS HEAVY VALIDATION` means implementation and
cheap substitutes exist, but the real dataset, compiled game simulator, GPU
scale, installed game, or long experiment has deliberately not run here.

| # | V1 requirement | Status | Local evidence | Dedicated-machine validation |
|---:|---|---|---|---|
| 1 | Project foundations | IMPLEMENTED | Typed package, strict TOML profiles, structured JSONL/TensorBoard metrics, disjoint seed plan, licence inventory, fast suite | Recreate environment from clean machine |
| 2 | Real Balatro integration | NEEDS HEAVY VALIDATION | `BalatroSimAdapter` validates the exact pinned array contract; fake upstream contract tests cover reset/step/masks/snapshots/seeds | Build pinned PyO3 extension; small real benchmark and deterministic snapshot/replay smoke |
| 3 | Real adult-fly backend | NEEDS HEAVY VALIDATION | Checksum-guarded v783 builder, sparse batched Torch LIF, independent seeds/reset, CPU tiny-artifact tests, real/shuffled topology hashes | Build full artifact; compare dynamics and benchmark CPU/CUDA batches |
| 4 | Fixed Balatro→fly encoder | NEEDS HEAVY VALIDATION | All upstream fields map deterministically to 4,420 bounded features and fixed disjoint sensory populations; hash includes root IDs | Encode real simulator batches and inspect stimulation statistics |
| 5 | Fly feature extraction | NEEDS HEAVY VALIDATION | Descending-neuron rate/voltage representation is fixed and root-ID/hash bound | Inspect full-v783 activity sparsity and feature scaling |
| 6 | Structured policy + critic | IMPLEMENTED | Autodiff actor/critic covers 25 stable type slots and all active composite heads; exact mask tests; parameter count logged | Throughput/learning behaviour at real feature scale |
| 7 | Parallel inference | NEEDS HEAVY VALIDATION | Environment batch and fly microbatch are independent; one fly call per state; lightweight vector tests | Tune microbatch from VRAM/throughput matrix |
| 8 | PPO | IMPLEMENTED | Rollout, GAE, returns, clipped loss, critic, entropy, gradient clip, Adam, tiny learning smoke and exact resume | Long-run numerical stability and learning curve |
| 9 | Strategy-neutral rewards | NEEDS HEAVY VALIDATION | Progress/clear/win decomposition only; components logged separately; mock and upstream-derived decomposition tests/path | Cross-check component sums against compiled real simulator episodes |
| 10 | Ante curriculum | IMPLEMENTED | Configurable ladder and held-out curriculum stream; promotion/exclusion tests; training orchestration | Long promotion run |
| 11 | Checkpoint/reproducibility | IMPLEMENTED | Atomic checkpoint + hash manifest includes model/critic/optimiser/env/RNG/config/curriculum/reward/components/Git/software; resume test | Restore real simulator + GPU run across process/machine |
| 12 | Frozen evaluation | IMPLEMENTED | Batched fixed seeds, learning disabled, metrics, successful-run detection, final-test lock and `final-eval.toml` | Full Ante-8 held-out and final 10,000 seeds |
| 13 | Baselines/controls | NEEDS HEAVY VALIDATION | Real/shuffled/conventional conditions plus random legal and isolated heuristic evaluator; controlled shuffle invariants tested | Train/evaluate all conditions with matched protocol |
| 14 | Complete episode recording | IMPLEMENTED | Versioned tamper-evident bundle: both seed forms, composite actions, phase/Ante/round, reward components, value/probabilities, hashes, neural reference, checkpoint/components | Read back replay bundles from real successful runs |
| 15 | Neural recording | NEEDS HEAVY VALIDATION | Opt-in typed Zstd Parquet events with decision/time/root ID/role/activity; disabled by default and one-episode CLI guard | Record one full-v783 showcase run and size output |
| 16 | Deterministic/live replay | NEEDS HEAVY VALIDATION | Simulator verifier fails at first hash mismatch; live balatrobot adapter uses resolved direct actions, stable-state waits, encoded signatures, and a rich parallel `CrossvalRun` comparison | Install v1.5.2 mod, isolate content mods, run live replay |
| 17 | Visualisation foundations | NEEDS HEAVY VALIDATION | Shared slowed timeline, action probabilities/cursor cues, activity-only FlyWire SVGs, X11 capture and ffmpeg composition commands | Render/capture/synchronise a chosen successful run |
| 18 | Benchmarks/runbook | NEEDS HEAVY VALIDATION | Four safe/guarded benchmarks, scalable profiles, 11-stage exact runbook with success criteria | Execute real benchmark matrix and tune configs |

## Immutable experiment boundaries

- V1 trains only the structured actor/readout and critic. The encoder,
  connectivity, fly dynamics, resets, and neural feature extractor are fixed.
- The default fly policy is linear after fly features. The conventional
  control has one explicitly configured 64-unit hidden layer and its parameter
  count is logged.
- PPO never imports or calls the upstream heuristic. The heuristic adapter is
  available only in `flylatro-baseline-evaluate`.
- Final-test seeds begin at 40,000,000 and require `--unlock-final-test`.
  Training, validation, curriculum, final-test, and showcase ranges do not
  overlap.
- Routine training never records full-brain events. Selected one-episode
  evaluation must opt in with `--record-neural`.

## Pinned external choices

### Balatro

- `jahankazimi078/balatroagent`, commit
  `38ae214317009952db4d22a98dc0765cef79370a`, MIT.
- Internal boundary: `balatro_sim.BalatroVecEnv`; the rest of Flylatro sees
  only `ArrayBalatroEnv` and validated NumPy dictionaries.
- Live bridge: balatrobot JSON-RPC v1.5.2. Live replay uses the canonical
  Balatro seed string, while simulator replay retains its integer seed-stream
  input separately.

### Adult fly

- FlyWire FAFB v783: 139,255 neurons, 3,732,460 source neuron-pair
  connections, 50,666,648 source synapses; CC BY-NC-SA 4.0 external data.
- Dynamics: fixed Shiu-style LIF constants and sparse Torch execution.
- Input: deterministic ascending-root-ID blocks from all 16,938 neurons with
  `super_class == sensory`, three neurons per encoded feature by default.
- Readout: all 1,305 neurons with `super_class == descending`; per-neuron spike
  rate and terminal voltage.
- Shuffled control: bijective postsynaptic neuron-label permutation. It
  preserves every edge/weight, exact pre out-degree, the post in-degree
  distribution, and presynaptic transmitter identity without sparse-edge
  coalescing, while disrupting biological topology.

## Latest lightweight validation

Run locally without the full simulator, FlyWire artifact, GPU, or long jobs:

```text
.venv/bin/python -m pytest -q                         69 passed
.venv/bin/python -m compileall -q src tests           passed
smoke PPO                                              1 update / 8 decisions
frozen smoke evaluation                               2 episodes / 2 wins
generated replay re-verification                      zero divergences
four safe benchmark entry points                      completed
random-legal baseline smoke                           completed
checkpoint CLI resume smoke                           8 → 16 decisions exactly
```

The full command sequence is maintained in [GPU_RUNBOOK.md](GPU_RUNBOOK.md).
