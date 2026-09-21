# ADR 0010: decision-streamed recording and optional spike showcase

Status: accepted and locally validated; a real showcase remains heavy work.

## Decision

Detailed recording is opt-in. `NeuralEventRecorder` buffers only one decision
and writes one compressed Parquet row group per decision with dictionary-coded
`role` and `event_kind`. Routine evaluation stores population rates; selected
plastic changes carry pre/post root ID, old/new efficacy and delta. Training
can stream sparse changes and synthetic reinforcement without writing a full
matrix.

A frozen real-FlyWire showcase may additionally enable backend spike events.
It records input stimulation, true simulated spike time, KC, MBON, genuine DAN
anatomy, downstream/descending activity, action, honestly labelled synthetic
appetitive/aversive outcome channels and selected plastic snapshots. Routine
training never enables spike recording.

Visualization reads only relevant decision row groups, retains one persisted
anatomical transform and faint background, and labels genuine DAN anatomy
separately from synthetic reinforcement. It displays at most selected strongest
KC->MBON changes with pre/post/old/new/delta details.

## Classification

Spike times and anatomical coordinates are simulation/data outputs. Playback
slowdown, selection limits, row-group layout and colours are engineering
presentation choices, not biological timing claims.
