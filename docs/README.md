# Flylatro documentation

Updated 12 November 2026. The latest completed research results linked here
are dated 6 November. Reliable frozen Ante-1 competence remains unproven.
[PLAN.MD](../PLAN.MD) defines primary V1; exploratory studies keep their own
rules, mappings, budgets and checkpoint identities.

## Setup and operation

| Task | Start here |
| --- | --- |
| Understand the model and run a synthetic smoke test | [Project README](../README.md) |
| Prepare real data, calibrate and run primary V1 | [GPU runbook](GPU_RUNBOOK.md) |
| Build native extensions and choose an execution backend | [Acceleration and checkpoint compatibility](PERFORMANCE_MIGRATION_GUIDE.md) |
| Read the measured speed, memory and numerical limits | [Performance report](PERFORMANCE_REPORT.md), [acceptance audit](PERFORMANCE_ACCEPTANCE.md) |
| Use the interactive training display | [Dashboard controls and metrics](TRAINING_DASHBOARD.md) |
| Check implementation coverage and evidence boundaries | [V1 implementation status](V1_STATUS.md) |
| Check licences and external data requirements | [Dependencies](DEPENDENCIES.md) |

The synthetic smoke run needs neither FlyWire data nor the real simulator.
Real runs require user-supplied data and the pinned simulator; native CUDA
execution additionally requires a compatible GPU/compiler/toolkit. Generated
`artefacts/`, `runs/`, `evaluations/` and data paths in study documents refer
to local evidence, not files supplied by a fresh repository checkout. Use fresh
output destinations and retain the study's declared configuration and hashes.

## Current research and completed long runs

Read the [learning roadmap](LEARNING_ROADMAP.md) for the current priority and
its evidence. The most recent sensory-control study did not establish a useful
cue-to-choice association. A direct KC-afferent bypass test is proposed to
separate upstream routing from output controllability; it is not a completed
result or a qualified long-training model.

| Experiment | Methods | Completed result |
| --- | --- | --- |
| Repaired native long exposure | [Run design](REPAIRED_LONG_TRAINING.md) | [Saturation stop and fresh evaluation](REPAIRED_LONG_RESULTS.md) |
| Repeated-game capacity, six million decisions total | [Single-seed training](SINGLE_SEED_TRAINING.md) | [No reliable repeated-game mastery](SINGLE_SEED_RESULTS.md) |
| Local incoming-strength normalisation | [Rule and comparison](LOCAL_NORMALISATION.md) | [Stable budgets; advancement criteria failed](LOCAL_NORMALISATION_RESULTS.md) |
| Local teacher curricula | [Methods, equations and both candidate results](LOCAL_TEACHER_CURRICULUM.md) | Native improvement milestone failed for both candidates |
| Discard strategy with resampled hidden draws | [Puzzles and completed pilot](DISCARD_STRATEGY_PUZZLES.md) | Own-PLAY transfer remains a bottleneck |

Blinds cleared, full Ante-1 wins, stochastic training outcomes and frozen greedy
evaluation are different measurements. Shared game roots across several
learners do not constitute independent training replicates. Stability or faster
execution alone does not establish useful card selection.

## Card selection, credit and sensory diagnostics

| Question | Documentation |
| --- | --- |
| How are the card-learning harness and pilots run? | [Card learning experiments](CARD_LEARNING_EXPERIMENTS.md) |
| How were PLAY/DISCARD preferences separated? | [Interface change](PLAY_DISCARD_INTERFACE.md), [learning results](PLAY_DISCARD_LEARNING_RESULTS.md) |
| What happened after repairing motor reachability? | [Repaired native membership](REPAIRED_NATIVE_MEMBERSHIP.md) |
| How do exploration and reward-selection credit interact? | [Exhaustive subset diagnostic](LOCAL_LEARNING_NEXT_STEPS.md), [exchange-credit factorial pilot](EXCHANGE_CREDIT_PILOT.md) |
| Does a beneficial exchange update steer the intended output? | [Causal update-direction probe](EXCHANGE_DIRECTION_RESULTS.md) |
| Does pair information reach card-output afferents? | [Pair representation probe](PAIR_REPRESENTATION_RESULTS.md) |
| Do fixed relational hints help? | [Relational sensory pilot](RELATIONAL_SENSORY_PILOT.md) |
| Can anatomical sensory cues control and teach output choices? | [Sensory control and dose-response results](SENSORY_CONTROL_RESULTS.md) |
| Did longer windows, rescaling or centred feedback fix selection? | [Neural windows](CARD_WINDOW_EXPERIMENT.md), [reference scaling](CARD_REFERENCE_SCALING.md), [reward contrast](CARD_REWARD_CONTRAST.md) |
| How are recorded choices checked? | [Selection diagnostic](CARD_SELECTION_DIAGNOSTIC.md), [credit audit](CARD_CREDIT_AUDIT.md), [direction audit](CARD_CREDIT_DIRECTION_AUDIT.md) |

## Design decisions and historical evidence

[The progress log](PROGRESS_LOG.md) records the earlier calibration, primary
pilots and subsequent development. Later completed-run reviews above carry
their own evidence paths and dates. [Architecture decisions](adr/) explain
versioned contracts, including [motor population allocation](adr/0014-group-local-motor-populations.md)
and [source coverage and fixed gain](adr/0015-source-coverage-and-fixed-gain.md).
[Performance migration records](PERFORMANCE_MIGRATION.md) preserve the measured
implementation work. The [archived reservoir/PPO plan](archive/RESERVOIR_V1_PLAN.md)
describes the legacy baseline.

Historical test counts, calibration passes and throughput numbers apply to
their recorded source and workload. Reproduction requires the same evidence
identity; a past readiness PASS does not authorize a different experiment.
