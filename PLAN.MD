
# Flylatro

## End-to-End Project Plan and Technical Specification

## 1. Project summary

**Flylatro** is an experimental reinforcement-learning project that will test whether a computational model based on the adult *Drosophila* connectome can learn to play Balatro from scratch.

The central experiment is not to teach the model Balatro using human demonstrations or an existing bot. Instead, the agent will begin with no strategy and learn exclusively through interaction with a simulated Balatro environment and reward signals.

The initial version will use the fly connectome as a **fixed neural-processing reservoir**. Balatro game state will be encoded into stimulation of selected fly neurons, activity will propagate through the fixed connectome, and a small trainable policy/readout will convert that activity into legal Balatro actions.

Training will occur against many headless Balatro environments running simultaneously. Once a trained policy is frozen, it will be evaluated on previously unseen seeds. A successful run from this untouched test set will then be replayed through the real Balatro game while the corresponding fly neural activity is visualised alongside it.

The final result should support both:

**Research output:** quantitative comparison of the real fly connectome against controls such as a shuffled connectome, conventional neural network and random policy.

**Demonstration output:** a video of a genuine previously unseen Balatro seed being beaten by the trained fly-based agent, accompanied by a synchronised visualisation of its neural activity.

---

# 2. Primary research question

The initial research question is:

> Can an agent using the fixed adult Drosophila connectome as its internal neural-processing structure learn to play Balatro through reinforcement learning alone?

A secondary and scientifically more important question is:

> Does the biological topology of the real fly connectome provide a useful inductive bias compared with an equivalently sized or constrained randomly wired network?

The project should therefore be designed from the beginning to support controlled comparisons.

---

# 3. Experimental principles

The project must distinguish carefully between the biological connectome and the machine-learning components surrounding it.

For V1:

| Component                    | Trainable? |
| ---------------------------- | ---------: |
| Balatro rules                |         No |
| Balatro → neural encoder    |         No |
| Fly connectome topology      |         No |
| Fly synaptic weights         |         No |
| Fly neural dynamics          |         No |
| Fly → action policy/readout |        Yes |
| Value estimator              |        Yes |

This means V1 should be described as a:

> **fly-connectome-based reinforcement-learning agent**

rather than claiming that biological synapses inside the fly brain themselves learned Balatro.

Later experiments may introduce plasticity inside the connectome.

### No strategic demonstrations

The fly must not receive:

* heuristic actions;
* human gameplay;
* expert trajectories;
* labelled "correct" poker decisions;
* behaviour-cloning pretraining;
* hand-strength rewards such as "flush = +5";
* explicit strategy rules.

An existing heuristic Balatro bot may be retained solely as an **external benchmark**.

### Permitted learning signals

The agent may receive generic environmental reinforcement such as:

* progress towards clearing the current blind;
* clearing a blind;
* progressing to later Antes;
* completing the game.

These rewards indicate whether the agent is succeeding at the environment; they do not tell it which strategy to use.

---

# 4. High-level architecture

```text
             HEADLESS BALATRO ENVIRONMENTS
             256–4096 games concurrently
                         │
                         ▼
                  game observations
                         │
                         ▼
                  fixed encoder
                         │
                         ▼
                neural stimulation
                         │
                         ▼
            ┌────────────────────────┐
            │ ADULT FLY CONNECTOME   │
            │                        │
            │ ~139k neurons          │
            │ sparse connectivity    │
            │ spiking/LIF dynamics   │
            │ fixed during V1        │
            └───────────┬────────────┘
                        │
                   spike activity
                        │
                        ▼
                feature extraction
                        │
                        ▼
             trainable actor + critic
                        │
                        ▼
                masked legal action
                        │
                        ▼
                Balatro simulator
                        │
                        ▼
                      reward
                        │
                        ▼
                       PPO
```

The Balatro environment and fly simulator should remain separate modules with explicit interfaces between them.

---

# 5. Existing components to reuse

## 5.1 Balatro simulator

Use `jahankazimi078/balatroagent` as the initial Balatro environment and RL harness.

Its Rust/PyO3 environment already supports multiple independent Balatro environments, action masks, automatic resets, rewards, snapshots and Python integration. The wrapper explicitly describes the Rust environment as running independent games in parallel and exposes both environment seeds and replay-oriented information.

The repository already uses large vectorised configurations such as 4,096 simultaneous environments.

It also deliberately produces valid Balatro seeds suitable for live replay in the real game.

Reuse from this project:

```text
Balatro game simulation
Observation encoding contract
Action representation
Legal-action masks
Parallel environment runner
PPO infrastructure where practical
Reward machinery
Curriculum mechanism
Snapshots
Seed management
Evaluation framework
Replay metadata
```

Do **not** use its heuristic actions as training targets.

The heuristic remains useful as an evaluation baseline only.

---

# 6. Fly-brain implementation

Use an adult FlyWire-based connectome.

The fly module must provide a GPU-capable batched simulation interface.

Conceptually:

```python
activity = fly.forward(
    stimulus=batch_stimulus,
    duration_ms=50
)
```

where:

```text
batch_stimulus:
[B, N_NEURONS]

activity:
[B, N_READOUT_FEATURES]
```

The available PyTorch implementations of the Shiu-style fly model already represent independent trials using a batch dimension and share one sparse connectome matrix between those trials.

That is exactly the execution pattern required for Flylatro.

---

# 7. Hardware requirements

Do not purchase hardware until the fly benchmark is completed.

The project should first measure actual neural decisions per second on the available machine.

### Development machine

| Resource | Minimum practical |                   Preferred |
| -------- | ----------------: | --------------------------: |
| CPU      | 6–8 modern cores |                12–16 cores |
| RAM      |             16 GB |                   32–64 GB |
| GPU      |       NVIDIA 8 GB |            NVIDIA 12–24 GB |
| Storage  |        20 GB free |                      50+ GB |
| OS       |             Linux | WSL2 Ubuntu or native Linux |

An NVIDIA GPU is strongly preferred because the fly simulation will probably dominate computational cost.

CPU-only development is acceptable for correctness testing and very small experiments.

### Potential cloud training

Only consider cloud GPU instances after benchmarking.

If local hardware produces insufficient throughput, desirable cloud options would have:

```text
16–24+ GB VRAM
good CUDA performance
8–16+ CPU cores
32+ GB system RAM
```

The exact GPU should be chosen from measured VRAM consumption and decisions/second rather than theoretical estimates.

---

# 8. Software environment

Recommended development environment:

```text
Windows 11
└── WSL2 Ubuntu
    ├── Git
    ├── Rust toolchain
    ├── Python 3.11/3.12
    ├── uv
    ├── PyTorch
    ├── CUDA
    ├── NumPy
    ├── pandas
    ├── PyArrow
    ├── TensorBoard
    ├── pytest
    └── VS Code Remote WSL
```

Additional software for final visualisation:

```text
Balatro
Steamodded / required mod loader
BalatroBot or equivalent live-game bridge
OBS and/or ffmpeg
FlyWire geometry/visualisation tooling
optional Blender / Three.js / custom renderer
```

---

# 9. Suggested project repository

Keep Flylatro separate from upstream repositories.

```text
flylatro/
│
├── pyproject.toml
├── README.md
│
├── configs/
│   ├── dev.yaml
│   ├── benchmark.yaml
│   ├── ante1.yaml
│   ├── curriculum.yaml
│   └── final.yaml
│
├── src/flylatro/
│
│   ├── env/
│   │   ├── balatro.py
│   │   └── observations.py
│   │
│   ├── fly/
│   │   ├── connectome.py
│   │   ├── dynamics.py
│   │   ├── encoder.py
│   │   ├── readout.py
│   │   └── batching.py
│   │
│   ├── policy/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── actions.py
│   │   └── masking.py
│   │
│   ├── training/
│   │   ├── ppo.py
│   │   ├── rollout.py
│   │   ├── curriculum.py
│   │   ├── rewards.py
│   │   └── checkpoints.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── baselines.py
│   │   └── controls.py
│   │
│   ├── replay/
│   │   ├── recorder.py
│   │   ├── live_replay.py
│   │   ├── brain_export.py
│   │   ├── synchronise.py
│   │   └── compose.py
│   │
│   └── telemetry/
│       ├── metrics.py
│       └── profiling.py
│
├── tests/
├── notebooks/
├── runs/
└── artefacts/
```

Upstream repositories should preferably be dependencies, submodules or clearly isolated vendor components rather than heavily modified copies.

---

# 10. Phase M0 — Balatro environment validation

The first milestone contains no fly model.

The objective is to prove the game environment works independently.

Run:

```text
1 environment
16 environments
64 environments
256 environments
1024 environments
```

with randomly selected legal actions.

Measure:

```text
environment steps/second
CPU utilisation
RAM usage
invalid-action count
episode completion rate
average episode length
```

Verify deterministic replay:

```text
same seed
+
same action sequence

→ same observable trajectory
```

### M0 acceptance criteria

```text
✓ simulator builds
✓ Python interface works
✓ vectorised stepping works
✓ legal-action masks work
✓ episodes reset correctly
✓ seeds are logged
✓ same seed/actions replay deterministically
✓ throughput benchmark is recorded
```

---

# 11. Phase M1 — Fly simulation benchmark

The second milestone contains no reinforcement learning.

Load the adult fly connectome and run controlled neural simulations.

Test simulation durations:

```text
10 ms
25 ms
50 ms
100 ms
```

Test batch sizes:

```text
1
2
4
8
16
32
64
128
```

For every combination record:

| Metric                    | Purpose                |
| ------------------------- | ---------------------- |
| Wall-clock time           | Training feasibility   |
| Simulated biological time | Interpretation         |
| Decisions/sec             | Main throughput metric |
| VRAM                      | Batch-size ceiling     |
| CPU RAM                   | System requirement     |
| GPU utilisation           | Optimisation           |
| Spike count               | Sanity check           |

The result should be a benchmark table showing the most efficient operating point.

### M1 acceptance criteria

```text
✓ deterministic neural simulation with fixed seeds
✓ batch dimension works
✓ sparse connectome executes on GPU
✓ stable spike outputs
✓ maximum usable batch size known
✓ recommended simulation duration selected
```

Do not proceed to large RL experiments until this benchmark exists.

---

# 12. Phase M2 — Balatro state encoder

Balatro observations must be converted into neural stimulation.

For V1 the encoder is deterministic and fixed.

Possible encoded information:

```text
current hand
rank of each card
suit of each card
card enhancement / edition where supported
money
current Ante
current blind type
current score
blind score requirement
hands remaining
discards remaining
Jokers
consumables
deck composition
shop contents
```

Each numerical feature should be normalised into a bounded range.

Example:

```text
normalised value 0.00 → 0 Hz
normalised value 0.50 → 50 Hz
normalised value 1.00 → 100 Hz
```

Categorical features should use separate populations.

Example:

```text
card suit:

Spades   → population A
Hearts   → population B
Clubs    → population C
Diamonds → population D
```

Input populations should ideally correspond to biologically meaningful sensory/input neurons rather than arbitrary neurons.

The mapping must be versioned.

Example:

```text
encoder_version = "sensory-v1"
```

It must remain frozen during an experiment.

---

# 13. Phase M3 — Fly feature extraction

After stimulation, neural activity must be compressed into features suitable for the trainable policy.

Potential V1 feature:

```text
spike count / firing rate for selected neurons
during the complete decision window
```

Possible readout populations should be tested later, but V1 should select one and freeze it.

Candidate choices:

```text
selected descending neurons
output-associated populations
random fixed sample of neurons
regional aggregates
1,000–5,000 selected neurons
```

Avoid exposing all ~139k neuron activities to a large unconstrained trainable network unless necessary.

The goal is to ensure the connectome remains responsible for substantial representation transformation.

---

# 14. Phase M4 — Action policy

Balatro has a composite action space.

Do not enumerate every possible complete action and simulate the fly separately for every option.

Instead run the fly once per game state and use structured action heads.

Conceptually:

```text
                   fly representation
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
        action type     cards      target
             │           │           │
       play/discard   indices     shop/etc.
```

Legal-action masks from the Balatro environment must be applied before sampling.

Illegal actions receive probability zero.

The policy should contain as little conventional neural-network capacity as practical.

Initial architecture:

```text
Fly representation
      ↓
small hidden layer, if necessary
      ↓
action heads
```

The critic/value head may use the same fly representation.

---

# 15. Neural-state behaviour

For V1, reset the simulated fly between game decisions.

```text
Balatro state
     ↓
reset fly
     ↓
stimulate for 25–50 ms
     ↓
record neural response
     ↓
choose action
```

Advantages:

```text
easy batching
deterministic inference
no cross-environment state management
simpler replay
simpler PPO
```

A later experiment should test persistent neural state:

```text
decision 1
↓
brain state retained
↓
decision 2
↓
brain state retained
...
```

This should not be part of the initial MVP.

---

# 16. Phase M5 — First integrated agent

Connect:

```text
Balatro
→ encoder
→ fly
→ readout
→ legal action
→ Balatro
```

No learning yet.

Randomly initialise the policy/readout.

Run complete episodes.

The purpose is purely integration validation.

### M5 acceptance criteria

```text
✓ state is encoded
✓ fly produces activity
✓ policy generates action probabilities
✓ masks prevent illegal actions
✓ environment accepts actions
✓ complete episodes execute
✓ episode logs contain all required fields
```

---

# 17. Training algorithm

Use PPO initially.

One training cycle:

```text
N Balatro environments
        ↓
collect observations
        ↓
encode observations
        ↓
split into fly microbatches
        ↓
simulate fly
        ↓
produce action distribution + value
        ↓
sample legal actions
        ↓
step all Balatro environments
        ↓
record transition
        ↓
repeat for rollout length
        ↓
calculate returns / advantages
        ↓
PPO update
```

Example starting configuration:

```yaml
balatro_envs: 256
fly_batch_size: 32
rollout_length: 128

gamma: 0.999
gae_lambda: 0.95

ppo_clip: 0.2
learning_rate: 0.0003

fly_duration_ms: 50

fly_reset_each_decision: true
```

These are starting values, not fixed final hyperparameters.

The environment count and fly batch size do not need to match.

For 256 environments and fly batch size 32:

```text
256 states
↓
8 fly microbatches
↓
256 decisions
```

---

# 18. Reward design

Reward shaping should remain strategy-neutral.

A suitable structure is:

```text
incremental blind progress
    small positive reward

clear current blind
    larger positive reward

complete run
    large positive reward
```

Avoid:

```text
pair reward
flush reward
Joker reward
money-management reward
specific-card reward
```

unless later experiments explicitly investigate additional shaping.

The final full-game reward should remain significant enough that the optimisation target eventually reflects actual winning rather than merely accumulating shaped rewards.

Reward-shaping strength should eventually be reduced as the agent becomes competent.

---

# 19. Curriculum learning

The agent should not initially be required to beat Ante 8.

Use progressively harder objectives.

Recommended ladder:

```text
Ante 1
 ↓
Ante 2
 ↓
Ante 3
 ↓
Ante 5
 ↓
Ante 8
```

Promotion occurs when performance on held-out evaluation episodes reaches a threshold, initially perhaps:

```text
70% win rate
```

for the current curriculum level.

The existing Balatro training infrastructure already implements an Ante-based curriculum of this general form.

Crucially, curriculum changes the difficulty of the goal; it does not provide demonstrations or strategy.

---

# 20. Training seed separation

Use independent deterministic seed streams.

```text
TRAIN
large changing seed stream

VALIDATION
fixed held-out seed set

FINAL TEST
completely untouched seed set
```

Example:

```text
training seeds:
millions of generated episodes

validation:
1,000 fixed seeds

final test:
10,000 fixed previously untouched seeds
```

Final-test seeds must never be used for:

```text
training
hyperparameter selection
curriculum promotion
model selection
debugging strategy
```

---

# 21. Metrics

Log at least:

```text
training step
episodes completed
mean episode return
mean Ante reached
median Ante reached
win rate
blind-clear rate
average decisions/episode
entropy
policy loss
value loss
KL divergence
fly decisions/sec
environment steps/sec
GPU memory
GPU utilisation
```

Curriculum-specific:

```text
Ante-1 win rate
Ante-2 win rate
Ante-3 win rate
Ante-5 win rate
Ante-8 win rate
```

Use TensorBoard initially.

---

# 22. Checkpoints

Every checkpoint should contain:

```text
actor parameters
critic parameters
optimizer state
training step
curriculum state
reward configuration
encoder version
connectome version/hash
fly simulator version
random seeds
configuration file
Git commit
```

This is necessary for reproducibility.

---

# 23. Controls and baselines

The final study should compare at least four systems.

| Agent                      | Purpose                     |
| -------------------------- | --------------------------- |
| Real fly connectome        | Main experiment             |
| Shuffled connectome        | Tests topology contribution |
| Conventional neural policy | ML reference                |
| Random legal policy        | Floor                       |

Optional fifth baseline:

| Agent                          | Purpose                   |
| ------------------------------ | ------------------------- |
| Existing heuristic Balatro bot | Hand-engineered reference |

The heuristic must never generate training targets for the fly.

---

# 24. Shuffled-connectome control

The shuffled network should preserve as many non-topological properties as practical.

Ideally preserve:

```text
number of neurons
number of edges
edge-weight distribution
excitatory/inhibitory composition
approximate in/out degree
input/output population sizes
neural dynamics
training algorithm
```

Only network topology should differ.

This makes:

> real connectome vs shuffled connectome

a meaningful comparison.

---

# 25. Conventional-network control

The conventional neural-network baseline should receive the same Balatro information and be trained with the same:

```text
seeds
rewards
curriculum
training steps
PPO algorithm
evaluation protocol
```

Parameter count should be documented.

Avoid deliberately crippling or overpowered baselines.

---

# 26. Performance optimisation

Do correctness first.

After integration is stable, profile:

```text
Balatro simulation time
encoder time
fly simulation time
feature extraction time
policy inference time
PPO update time
```

The likely bottleneck is the fly simulation.

Optimisation order:

```text
1. shorten fly simulation duration
2. increase fly GPU batch size
3. optimise sparse matrix operations
4. reduce readout-neuron count
5. torch.compile where beneficial
6. investigate lower precision carefully
7. investigate alternative GPU simulators only if required
```

Do not optimise Balatro before profiling demonstrates that it matters.

---

# 27. Episode recording

Every evaluation episode should optionally produce a complete deterministic trace.

Run-level record:

```json
{
  "run_id": "...",
  "balatro_seed": "...",
  "result": "win",
  "max_ante": 9,
  "checkpoint": "...",
  "connectome_hash": "...",
  "encoder_version": "...",
  "fly_simulator_version": "...",
  "balatro_simulator_version": "..."
}
```

Decision-level log:

```json
{
  "decision_id": 47,
  "ante": 6,
  "round": 17,
  "game_state": "selecting_hand",

  "action": {
    "type": "PLAY_HAND",
    "cards": [1, 4],
    "target": null
  },

  "reward": 0.42,

  "policy": {
    "action_probability": 0.81,
    "value": 7.92
  },

  "state_hash_before": "...",
  "state_hash_after": "...",

  "brain": {
    "duration_ms": 50,
    "spike_file": "brain/decision_0047.parquet"
  }
}
```

Always record full composite actions, not merely the action type.

---

# 28. Neural activity logging

Do not save full brain spikes for every training episode; storage would become excessive.

During training:

```text
aggregate neural diagnostics only
```

During selected evaluation runs:

```text
full spike timeline
```

For each decision store:

```text
timestamp
neuron/FlyWire ID
spike event
decision ID
```

Parquet is appropriate.

Possible structure:

```text
run_00481/
│
├── manifest.json
├── decisions.jsonl
│
└── brain/
    ├── decision_0001.parquet
    ├── decision_0002.parquet
    ├── decision_0003.parquet
    └── ...
```

---

# 29. Final evaluation

After training is finished:

```text
freeze all parameters
disable PPO
disable curriculum shortcuts
set target = Ante 8
```

Run the agent on approximately:

```text
10,000 untouched seeds
```

Measure:

```text
win rate
mean Ante
median Ante
score distribution
run-length distribution
decision count
```

Successful seeds are saved automatically.

Do not manually search training seeds for a visually impressive win and present that as the result.

The showcase run should come from the untouched final test population.

---

# 30. Selecting the showcase run

A valid showcase candidate must:

```text
come from final held-out evaluation
use frozen parameters
beat the defined game objective
replay deterministically
have complete action logs
have complete spike logs
```

You may choose the most visually interesting successful run among those valid successes.

For example:

```text
interesting Joker combination
close Boss Blind
unusual discard
dramatic final hand
```

Selection for presentation is acceptable after the success criterion has already been objectively satisfied.

---

# 31. Real Balatro replay

Do not rebuild Balatro's UI.

Use an installed copy of Balatro.

The final replay controller will:

```text
launch / connect to Balatro
start recorded seed
wait until actionable
verify expected state
send recorded action
wait for resulting game state
verify state hash
continue
```

The simulator deliberately exposes game-valid seed strings for live replay.

Use BalatroBot or an equivalent mod/API bridge to submit actions programmatically.

This gives you:

```text
real Balatro graphics
real cards
real Jokers
real animations
real sounds
```

without extracting or duplicating Balatro assets.

---

# 32. Replay verification

Before video capture, replay the complete successful episode automatically.

At every decision:

```text
expected state
      ↓
actual Balatro state
      ↓
compare
```

If they differ:

```text
STOP REPLAY
log mismatch
```

Do not continue after divergence.

This prevents silent desynchronisation.

---

# 33. Screen recording

Two possible approaches:

### Development

Use OBS.

This is convenient for manually validating the replay.

### Production

Use ffmpeg or another scripted capture workflow.

Automated workflow:

```text
start Balatro
load seed
start capture
execute replay
wait for victory screen
stop capture
```

Output:

```text
balatro_raw.mp4
```

No Balatro assets need to be separately packaged into Flylatro.

---

# 34. Fly-brain visualisation

Use real FlyWire neuron coordinates where available.

The visual should represent:

```text
brain anatomy
stimulated/input neurons
internal propagation
readout/output neurons
spike intensity
```

A separate renderer should consume the recorded spike-event data.

For every Balatro decision:

```text
Decision 47
      ↓
load decision_0047.parquet
      ↓
animate recorded 50 ms neural episode
```

Do not synthesise activity that was not recorded.

---

# 35. Visual timing

The biological neural event will be too fast to understand at normal speed.

For example:

```text
actual simulation:
50 ms

visual presentation:
1.0 sec

slowdown:
20×
```

Clearly label this:

> Neural activity shown 20× slower than simulated time.

The rendering should therefore represent actual spikes with altered playback speed, not fabricated activity.

---

# 36. Final video layout

Recommended default layout:

```text
┌────────────────────────────────┬──────────────┐
│                                │              │
│                                │              │
│         REAL BALATRO           │  FLY BRAIN   │
│             ~70%               │     ~30%     │
│                                │              │
│               🪰               │   activity   │
│                                │              │
├────────────────────────────────┴──────────────┤
│ Ante 6 | Decision 47 | PLAY K♠ K♥            │
│ PLAY 82% | DISCARD 11% | Other 7%            │
└───────────────────────────────────────────────┘
```

Balatro remains visually dominant.

The brain panel shows the agent's internal processing.

---

# 37. Brain visual layers

The visualiser should distinguish:

```text
Input neurons
Internal activity
Readout neurons
```

Potential encoding:

```text
input neurons      distinct marker
active neurons     brightness based on recent spikes
readout neurons    distinct outline
inactive neurons   low opacity
```

If appropriate, connections may briefly illuminate when spike propagation occurs.

Avoid rendering all tens of millions of synapses continuously.

That would be visually unreadable and computationally unnecessary.

---

# 38. Action-probability visualisation

Alongside neural activity, display the agent's action confidence.

Example:

```text
PLAY       ████████████████ 82%
DISCARD    ██               11%
REROLL     █                 4%
OTHER                        3%
```

For card selection, optionally show:

```text
Card 1    95%
Card 2    13%
Card 3    87%
...
```

This provides an interpretable bridge between:

```text
neural activity
↓
policy
↓
action
```

---

# 39. Fly cursor

Add a stylised fly cursor as a presentation overlay.

Example:

```text
action log says:
PLAY cards [2, 5, 7]

visual layer:
🪰 moves toward those cards
↓
cards highlight
↓
API executes actual action
```

The fly cursor is decorative.

It should not be represented as the mechanism by which the agent actually controls Balatro.

The API/replay controller remains the real execution path.

---

# 40. Replay synchronisation

Create a master timeline.

Example:

```text
Decision 47

0.000 s    Balatro becomes actionable
0.200 s    freeze / focus decision
0.300 s    neural replay starts
1.300 s    neural replay ends
1.450 s    probabilities appear
1.700 s    fly cursor moves
2.000 s    recorded API action executes
2.000–4.8  Balatro animations
4.800 s    next state becomes actionable
```

The actual simulation remains 50 ms.

Presentation timing may be slower.

---

# 41. Video production

Render components independently.

```text
balatro_raw.mp4
brain_raw.mp4
decision_timeline.json
```

Then compose them offline.

Final compositor handles:

```text
layout
brain panel
fly cursor
decision labels
probabilities
Ante counter
titles
captions
timing
```

Output:

```text
flylatro_win.mp4
```

This approach means the brain rendering can be redesigned without replaying Balatro.

---

# 42. Reproducibility bundle

Every final video should have an associated reproducibility bundle.

```text
showcase_run/
│
├── manifest.json
├── decisions.jsonl
├── state_hashes.jsonl
├── checkpoint.pt
├── config.yaml
├── brain/
│   └── ...
└── README.md
```

The README should document:

```text
Balatro seed
model checkpoint
connectome version
training step
software versions
how to replay
final result
```

---

# 43. Testing strategy

Testing should include four levels.

### Unit tests

Examples:

```text
encoder normalisation
card feature encoding
action-mask handling
spike-feature aggregation
episode logger
state hashing
```

### Integration tests

Examples:

```text
Balatro → encoder
encoder → fly
fly → policy
policy → valid action
```

### Determinism tests

Example:

```text
same model
same Balatro seed
same neural RNG seed

→ identical action trajectory
```

where deterministic operation is expected.

### Replay tests

Example:

```text
record episode
restore seed
replay actions

→ identical state hashes
```

---

# 44. Performance tests

Maintain benchmark scripts.

```text
bench_balatro.py
bench_fly.py
bench_policy.py
bench_end_to_end.py
```

Report:

```text
env steps/sec
fly decisions/sec
full decisions/sec
episodes/hour
GPU memory
CPU memory
```

Keep benchmark history so optimisation changes can be evaluated objectively.

---

# 45. Scientific experiment matrix

Once the pipeline works, run controlled experiments such as:

| Experiment                            | Variable                |
| ------------------------------------- | ----------------------- |
| Real vs shuffled connectome           | Network topology        |
| 10 vs 25 vs 50 ms                     | Neural computation time |
| 500 vs 2,000 vs 5,000 readout neurons | Readout capacity        |
| Fixed vs persistent fly state         | Neural memory           |
| Different input populations           | Sensory mapping         |
| Fixed vs trainable encoder            | Encoder learning        |
| Fixed vs partially plastic connectome | Internal plasticity     |

Do not attempt all of these before the base experiment works.

---

# 46. Future V2: persistent fly memory

After V1:

```text
FlyState[i]
```

can persist for each Balatro environment.

Instead of:

```text
reset brain
state → brain → action
reset
```

use:

```text
state 1 → brain → action 1
             ↓
       neural state retained
             ↓
state 2 → brain → action 2
```

This tests whether short-term neural dynamics improve sequential decision-making.

---

# 47. Future V3: plastic fly synapses

A later experiment may permit selected synapses to adapt.

Potential directions:

```text
reward-modulated STDP
Hebbian plasticity
neuromodulatory reward signals
surrogate-gradient learning
limited plastic neuron populations
```

This changes the research question to:

> Can the simulated fly connectome itself learn Balatro through internal synaptic plasticity?

That should remain separate from V1.

---

# 48. Risks

| Risk                                | Mitigation                                             |
| ----------------------------------- | ------------------------------------------------------ |
| Fly simulation too slow             | Benchmark early; microbatch; shorten simulation window |
| GPU memory too high                 | Reduce batch/readout size                              |
| Fixed reservoir cannot learn task   | Gradually add encoder capacity or plasticity           |
| Sparse rewards prevent learning     | Curriculum + strategy-neutral shaping                  |
| Simulator differs from real Balatro | Replay cross-validation                                |
| Live replay diverges                | State hashes and hard-stop verification                |
| Brain visual unreadable             | Render active subset / aggregated activity             |
| Training result is unstable         | Multiple seeds and repeated training runs              |
| Connectome effect is illusory       | Shuffled-network control                               |
| Conventional head learns everything | Keep policy/readout deliberately small                 |
| Upstream dependency changes         | Pin commits and record hashes                          |

---

# 49. Licensing and publishing

Before publishing source code:

```text
verify Balatro-simulator licence
verify fly-simulator licence
verify connectome-data terms
verify any copied implementation licence
```

If permissive licensing is important, prefer implementing the required batched PyTorch fly dynamics from permissively licensed source material rather than copying code from a more restrictive implementation.

Balatro itself should remain an external installed dependency; do not distribute proprietary Balatro game assets.

---

# 50. Milestone roadmap

| Milestone     | Outcome                                   |
| ------------- | ----------------------------------------- |
| **M0**  | Headless parallel Balatro validated       |
| **M1**  | Fly GPU simulation benchmarked            |
| **M2**  | Fixed Balatro → fly encoder              |
| **M3**  | Fly neural features extracted             |
| **M4**  | Composite action decoder implemented      |
| **M5**  | Random fly agent completes valid episodes |
| **M6**  | PPO learns Ante 1                         |
| **M7**  | Curriculum reaches later Antes            |
| **M8**  | Non-zero Ante-8 held-out success          |
| **M9**  | Real vs shuffled vs conventional controls |
| **M10** | Final untouched test evaluation           |
| **M11** | Successful run replay bundle              |
| **M12** | Real Balatro + neuron visual video        |

---

# 51. Go/no-go gates

The project should deliberately include stopping points.

### Gate A — after fly benchmark

Proceed only if fly inference throughput appears feasible.

If not:

```text
optimise simulator
reduce neural window
reduce batch
investigate alternative backend
```

before doing RL work.

### Gate B — after Ante 1

If the fixed-connectome readout cannot learn Ante 1 despite reasonable optimisation, investigate whether:

```text
encoding is inadequate
readout capacity is inadequate
reward signal is inadequate
simulation window is inadequate
```

before scaling training.

### Gate C — before full training

Only launch long-running experiments after:

```text
determinism works
checkpoints work
evaluation works
metrics work
replay works
```

---

# 52. Definition of MVP

The minimum successful research prototype is:

```text
✓ real adult fly connectome
✓ fixed neural dynamics
✓ fixed Balatro encoder
✓ trainable readout
✓ PPO from random initialisation
✓ no demonstrations
✓ parallel training
✓ Ante-based curriculum
✓ held-out evaluation
✓ at least one successful Ante-8 test run
✓ recorded neural activity
✓ deterministic replay
```

---

# 53. Definition of full project success

The full project succeeds when it produces:

### Technical result

A reproducible fly-connectome agent capable of completing Balatro runs on previously unseen seeds.

### Experimental result

A controlled comparison of:

```text
real connectome
shuffled connectome
conventional neural policy
random policy
```

under equivalent training conditions.

### Quantitative result

A report containing:

```text
learning curves
Ante progression
win rates
sample efficiency
training throughput
run-length distributions
control comparisons
```

### Demonstration result

A deterministic replay of a genuine held-out winning run showing:

```text
real Balatro gameplay
fly-shaped cursor overlay
actual agent decisions
actual recorded fly neural activity
neuron activity slowed for visibility
action probabilities
Ante / decision metadata
```

---

# 54. Recommended first implementation sequence

The immediate implementation order should be:

```text
Balatro simulator
       ↓
Balatro benchmark
       ↓
fly simulator
       ↓
fly benchmark
       ↓
fixed encoder
       ↓
random integrated agent
       ↓
Ante-1 PPO
       ↓
curriculum
       ↓
full training
       ↓
controls
       ↓
final test
       ↓
winning-run recording
       ↓
live Balatro replay
       ↓
brain rendering
       ↓
final video
```

Do not begin with the visual layer.

Do not begin with large RL runs.

The first two measurements to obtain are:

```text
Balatro environment steps/sec

and

fly decisions/sec
```

Those numbers determine the practical scale of the entire project.

---

# 55. Final conceptual description

The completed system should be explainable in one sentence:

> **Flylatro takes the current state of Balatro, converts it into stimulation of a simulated adult fruit-fly connectome, observes the resulting neural activity, and learns through reinforcement alone how to convert that activity into actions that eventually beat the game.**

And the final demonstration should be equally straightforward:

> **A previously unseen Balatro seed is replayed from the fly's recorded winning test run while the exact neural activity that produced each decision is shown alongside the game.**
