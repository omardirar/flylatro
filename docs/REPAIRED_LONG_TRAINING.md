# Repaired native long-training experiment

The selected candidate retains native synthetic outcome reinforcement and the
choice-gated three-factor rule. Only existing anatomical KC→MBON efficacies
learn. It uses the repaired action-conditioned PLAY/DISCARD mapping, 5% explicit
card-subset exploration, stochastic training, and disabled blind skipping.
Teacher-discrepancy and teacher-rate rules are not used.

`scripts/run_repaired_long.py` prepares fresh conditions through the unchanged
card-learning harness, then executes only `subset-plastic` and `subset-frozen`.
The inherited declaration also contains two unused top-k conditions; the selected
arms and actual budget are explicit in `long-plan.json`.

Each arm has three learners receiving 1,000,000 decisions each: **6,000,000
training decisions including frozen controls**. Checkpoints are evaluated at
10k, 30k, 100k, 300k and 1M decisions per learner on 256 shared validation roots,
20850000–20850255. Each learner is compared with its own initial efficacies and
the corresponding frozen learner. Evaluation uses greedy choices with learning
disabled. Shared game roots are not independent training replicates.

The budget is fixed: a performance plateau does not stop training. Existing
safety gates remain enabled. Original checkpoints are preserved; a changed
mapping starts from fresh unit efficacies. The experiment measures delayed
learning, and does not assume or establish grokking. Final-test seeds remain
untouched for later confirmation.

Run the prepared experiment from the project root:

```bash
bash runs/repaired-native-long-20261028-v1/run.sh
```

The launcher displays the existing training dashboard. It writes progress to
`long-status.json`, comparisons to `learning-curve-*.json`, and native evaluation
transition logs under `evaluations/`. Training attempts have separate log
directories. Rerunning the same command resumes completed stages and the latest
available checkpoint without appending replayed decisions to old log files.
After an abrupt termination, any uncheckpointed tail remains in the old attempt
for inspection and is superseded by the new attempt. Safety stops require
inspection and are not automatically resumed. Failed evaluations are retried
in a new output directory. A completed experiment is not trained again.

Launcher verification exercises actual synthetic CPU training, stage checkpoint
resume, recovery after an evaluation interruption, initial-weight restoration,
unchanged frozen weights, paired comparisons, and declaration integrity. It is
an orchestration check, not evidence that real FlyWire gameplay will improve.
