# ADR 0003: PPO, controls, and replay certification

## Status

Accepted for V1.

## Decision

PPO stores fixed processor outputs in each rollout. Gradients therefore update
only the structured actor/critic; they cannot enter the Balatro encoder or fly
dynamics. The real-fly condition uses no post-fly hidden layer. A conventional
control may use one configured 64-unit `tanh` layer, with its parameter count
recorded in manifests.

The experiment uses five disjoint seed ranges: training, validation,
curriculum selection, final test, and showcase. Curriculum promotion reads only
the curriculum range. Final evaluation requires a separate explicit unlock.

Every replay candidate is a versioned, hash-verified directory. Its identity
stores both the integer simulator seed-stream input and the canonical Balatro
seed string because they are not interchangeable. The decision stream stores
complete structured actions and encoded-state hashes. Simulator replay must
pass before live replay. Live replay compares a state signature shared by the
encoded simulator observation and balatrobot (`state`, Ante, round, money,
hands, discards, chips). It also runs the pinned `CrossvalRun` beside the game
and compares a richer normalized state (blinds and card/inventory/shop areas)
before each action. Either comparison stops on the first difference.

Detailed full-brain spike events are a separate opt-in Parquet member. Routine
PPO checkpoints and rollouts never persist them.

## Consequences

- From-scratch RL and the fixed-fly boundary are mechanically enforced by the
  rollout representation and optimiser parameter set.
- External heuristic actions cannot enter PPO; they implement a different,
  evaluation-only action-source interface.
- Replay bundles are suitable for both scientific reproduction and the final
  live presentation without relying on filenames for identity.
- The encoded signature gives a stable trace-level check while the richer
  simulator mirror catches visible deck/inventory/shop divergence. Live
  cross-validation still requires the installed game and is therefore a
  dedicated-machine validation task.
