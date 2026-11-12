# Blind skipping and failure-punishment experiments

The completed 100,000-decision followup, corrected held-out comparison and next
recommendation are in [the 2026-10-16 results](NO_SKIP_RESULTS_20261016.md).

The 100,000-decision run launched after performance acceptance stopped safely
at 1,509 decisions: `runs/plastic-performance-event-source-complete-v1-20261014T172134Z`.
It completed 169 episodes with zero wins. Lower-bound efficacy fraction reached
0.339254 (threshold 0.1); mean efficacy fell to 0.573742 and mean absolute drift
was 0.426258. The failure pulse in that command's configuration was **0.6**.

Durable action/reinforcement logs contain 212 blind skips, 676 plays, 442
discards, 169 blind selections and 10 cash-outs. Direct appetitive and aversive
reinforcement for skips were both zero. Total appetitive magnitude was 1.348111;
total aversive magnitude was 101.4, about 75.2 times larger. The current learning
rule applies signed reinforcement to every eligible KC→MBON edge, so this
imbalance produces broad depression rather than selectively teaching good plays.
The observed lower-bound stop supports addressing that imbalance first.

An earlier separate `runs/exploratory-20261010-weak-failure` trial used failure
pulse **0.01**. At 1,000 decisions it had no bound saturation, mean efficacy
1.000383, mean absolute drift 0.000517 and zero wins. This is evidence for a more
stable update scale, not evidence of successful learning or a controlled native
comparison. Reducing below 0.01 has not been justified by these results.

## Explicit action restriction

Set `[environment].allow_skip_blinds = false` to remove only `SKIP_BLIND` from
legal action masks. The real adapter applies it after reset, step, observe and
restore, without mutating upstream masks. It rejects submitted skip actions
before stepping any environment, including when upstream strict mode is off.
Blind selection remains available; all other game actions are preserved.

Training and frozen evaluation both construct the environment through the same
factory. The option defaults to true for existing experiments and legacy
checkpoints. Restricted runs record `action_constraints` in their component
identity. Checkpoint loading rejects a changed skip policy in either direction,
including frozen evaluation; start a fresh experiment for different rules.
Existing stopped runs and checkpoint bytes are unchanged.

These are new exploratory settings. They force exposure to playing blinds and
change the decision task. They do not count as equivalent-model performance
migration or demonstrate better learning. Persisted motor pools/normalization,
complete anatomy, 100 ms/dt 0.1 ms, gain, learning rule and safety remain intact.
Old calibration report paths are cleared instead of claiming fresh qualification;
the existing corpus and fixed motor artifact are retained.

## Prepared comparisons

| Config | Blind skipping | Failure pulse |
|---|---|---:|
| `plastic-v783-source-complete-event-qualified-v1.toml` | Allowed | 0.6 |
| `exploratory-event-no-skip-v1.toml` | Disabled | 0.6 |
| `exploratory-event-weak-failure-v1.toml` | Allowed | 0.01 |
| `exploratory-event-no-skip-weak-failure-v1.toml` | Disabled | 0.01 |

The three exploratory configs declare a 100,000-decision cap and checkpoint
every 1,000 decisions. The original qualification config retains its original
1,100-decision budget; its long-run CLI overrides were recorded in the stopped
run. Compare matched seeds and exposure, retaining safety stops as outcomes.
The combined no-skip/weak-failure config is a practical next candidate; the
individual variants distinguish which change affects stability and performance.
No new long training run was automatically launched.

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 \
.venv/bin/python -m flylatro.learning.cli \
  --config configs/exploratory-event-no-skip-weak-failure-v1.toml \
  --heavy --dashboard --dashboard-refresh 1 --no-tensorboard
```

Use a fresh timestamped run and initial efficacy. Do not resume the saturated
checkpoint with different punishment/action rules. Ctrl+C preserves a consistent
interrupted checkpoint; safety gates stay enabled.

## Verification and next questions

Full CPU regression: **529 passed, 10 CUDA tests skipped**, recorded in
`artefacts/no-skip-20261014-v1/regression.stdout`. New tests cover batched masks
after reset/step/observe/restore, ownership of upstream masks, rejection before
any simulator mutation, checkpoint incompatibility in both directions and
legacy default loading. A real source-complete native/Rust 64-decision CLI
smoke has zero skip actions and no safety stop. The public native 16-prefix /
16-future full-state resume audit passes in the same artifact directory.
These functional checks do not establish long-run learning outcomes.

First measure blind clears, held-out wins, efficacy drift and saturation under
the new variants. Avoid interpreting survival alone as learning. Additional
independent candidates include motor temperature 1.0→0.5 to increase the effect
of score differences while retaining epsilon exploration, and a lower learning
rate if updates still cause rapid drift after punishment is balanced. Neither
change was applied here. If weights remain healthy without improving decisions,
measure whether outcome updates change relative action scores: global
reinforcement of all coactive edges may be a credit-assignment limitation that
requires a separate, explicitly designed learning-rule investigation.
