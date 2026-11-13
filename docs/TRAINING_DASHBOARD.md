# Training dashboard

The primary training CLI displays gameplay, learning health and hardware status
while training runs. See the [README smoke run](../README.md#safe-local-plastic-brain-smoke-run)
for a small local example and the [GPU runbook](GPU_RUNBOOK.md) for real runs.

Training opens a colourful, responsive terminal dashboard automatically. It
shows each fly's decisions, runs, wins/game overs, current/highest/average round,
Ante, blind chips and training reward. A bar chart compares rounds reached in
recent completed runs, labelling played rounds and skipped blinds separately.
Solid bars show played rounds, patterned bars show skips, and both have exact
counts. Bars start at zero, share the same scale across flies, and run from oldest at the top to
newest at the bottom. The average uses all tracked completed runs; performance
history retains 64 runs while the visible bars adapt to available screen space.
Round 0 means no blind has been selected yet. Round/highest/average statistics
use played rounds; skips add to the separate bar segment without counting as
rounds beaten. Missing legacy skip counts show `?`; fresh runs track them fully.
Real training stays fixed to **Red Deck / White Stake**, with the
current Ante win target and fresh run seeds shown in the header.

Each fly has an action feed with named cards, Jokers and purchases. Game over,
round beaten, Ante beaten and run won appear as separate coloured milestones
and add no decisions. Up/Down or the mouse wheel scroll through up to 256
recent activity entries; PgUp/PgDn jump five entries, Tab or Left/Right select
a fly, Home goes to the oldest entry, and End or L returns to the live feed.
Browsing holds the selected history steady while training and counters continue.
Press H for metric definitions and controls. Ctrl+C still checkpoints and exits.
The feed describes submitted choices; simulator card/consumable legalisation
can adjust what takes effect. Hidden card identities remain hidden.

Training reward is the accumulated learning signal; highest blind chips is
the largest observed blind total. Neither is a final game score. Throughput,
elapsed time, ETA, plasticity guards and GPU utilisation/memory/temperature/power
are shown separately. GPU readings cover the whole device; Torch allocation
is shown separately for this process.

The display follows [Nielsen's usability guidance](https://www.nngroup.com/articles/ten-usability-heuristics/)
for visible status, familiar labels, consistency and accessible help, and
[ONS chart guidance](https://service-manual.ons.gov.uk/data-visualisation/guidance/axes-and-gridlines)
for zero baselines and consistent scales. Gameplay, hardware and learning
limits are grouped separately; status and milestone text remains readable
without colour. Performance line charts and speed sparklines are omitted.
Use `--dashboard-refresh 2` to refresh every two seconds, `--no-dashboard` for
quiet terminal output, or `--dashboard` to force periodic plain snapshots in a
redirected log. Final JSON stays on stdout; the display uses stderr. Set
`NO_COLOR=1` to disable colours; non-Unicode terminals use ASCII.
Final scores unavailable from the pinned simulator are shown as `--`;
observed chip peaks are labelled separately. Monitoring counters survive new
checkpoint resumes and are recorded in `training-dashboard.json` and
`training-episodes.jsonl`. Older checkpoints retain available aggregate
history and label missing per-fly/score/round history. Round averages and graphs
begin with newly tracked runs when resuming a checkpoint without round history.

