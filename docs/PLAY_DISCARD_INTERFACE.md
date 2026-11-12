# Separate PLAY and DISCARD card preferences

Implemented and tested 27 October 2026. The new
`action-conditioned-card-routing-v2` interface gives DISCARD its own card-slot
ranking and exact-count head. PLAY retains the `card` / `card_count` heads;
DISCARD uses `discard_card` / `discard_card_count`. The decoder first chooses
the action type, then reads the corresponding membership/count heads.

Choice credit dispatches through the same action-specific routes. Selected
DISCARD cards no longer add fresh eligibility to PLAY card/count outputs, or
vice versa. The mapping rejects shared neurons between the two layouts and
between their core action/count/card groups. Both card populations are also
excluded from every noncard target pool. Historical eligibility still decays
normally and can receive later reward; the change concerns fresh choice credit.
Recurrent neural interactions remain possible; this is not two isolated brains.

USE_CONSUMABLE retains the existing PLAY card/count interpretation and its
contextual target head. This change does not redesign consumable targeting.
Shop/pack/joker/consumable target pools retain their contextual reuse of
noncard neurons. No hand-strength, pair target, recommended action or new
trainable policy parameter enters the runtime encoder or decoder.

## Population budget and calibration

The first real calibration attempted width-two, fully exclusive core pools.
It could not allocate enough responsive neurons: only 59 distinct candidates
passed the current context-specific response criteria across the core groups.
The fully exclusive layout would need 90 core neurons, with 86 memberships
active in the sampled legality contexts. This failed attempt is preserved in
`artefacts/play-discard-motor-20261027-v1/`.

The qualified exploratory layout therefore explicitly uses **one MBON per
option**, requiring 45 exclusive core neurons. It has 51 pool memberships
including the six contextual targets. Pool width is recorded in the mapping,
configuration, declaration and calibration thresholds. All response, signal
diversity, dynamic-range, silence and context-coverage limits remain unchanged;
the minimum-width criterion explicitly matches this new architectural width.
This is not a width-two qualification or a primary-protocol promotion.

Calibration uses the hash-verified cached unit-weight responses to the existing
96-state reward-free corpus (four neural repeats), with blind skipping disabled
in the legality masks. No reward or pair label chooses neuron membership or
normalization. Native tests use fresh unit efficacies, never trained legacy
weights. Neural anatomy, dynamics and the candidate universe stay fixed.

Configuration validation permits width one only with an explicitly versioned
action-conditioned artifact and checks its width. Legacy scientific mappings
retain their width-two minimum. Generic preflight also retains its primary
minimum of two; the qualification writes an explicit
`preflight-thresholds.json` override for this exploratory width-one layout.

## Verification

The full regression suite passed **691 tests**, including native CUDA tests.
New tests cover:

* Identical activity and hand masks producing PLAY of slots 1/2 and DISCARD of
  other slots, with independent counts; opposing membership also works when
  both counts are held at two.
* NumPy/Torch choice masks and real plastic updates leaving the inactive
  action's card/count efficacies unchanged from zero eligibility.
* Rejection of neuron aliases and routing aliases; independent legality
  contexts; serialization and unchanged legacy artifact identities.
* Explicit width-one configuration, preserved synthetic candidate universe,
  rejection of a legacy checkpoint under the new interpretation and exact
  training/checkpoint resume for the new layout.
* Reward-free calibration at declared widths one and two, preserving all
  other quality thresholds.

The native qualifier uses 13 training hands, one per pair rank, with four
matched neural repeats. The same observation and spike seed are processed in
two rows; only the action legality differs (PLAY versus DISCARD). It checks
native simulator acceptance, same-count top-two rankings, and immediate plastic
updates on both rows. Each diagnostic positive pulse is explicitly synthetic,
not native discard reward; all plastic state is restored exactly afterward.

The final native qualification passed all calibration gates and **104**
inactive-action zero-update checks. PLAY/DISCARD selected different card sets
on 51/52 trials. Their top-two card sets differed on **44/52** trials when
count was held at two in the readout diagnostic, so the distinction is not
only a count change. All 13 artificial opposing-choice/native-action fixtures
passed, and plastic state was restored exactly. This does not require the
two actions to always select different cards in every possible state.
An additional real FlyWire/native training audit passed exact checkpoint resume
after eight environment decisions and across the next eight decisions. Its
test-only checkpoint/report are `qualification-resume.pkl` and
`native-resume-report.json`; they are not a candidate trained policy.

A separate **labelled artificial-readout fixture** drives the PLAY pair and
five non-pair DISCARD cards through the decoder and native simulator. An
eight-card hand has six non-pair cards, so the fixture selects five of them,
respecting the simulator's discard limit. These injected readout values are
tests, not neural responses, learned behavior or teacher signals in training.

## New mapping and reproduction

Final qualification outputs are under
`artefacts/play-discard-motor-20261027-v4/`:

* `action-conditioned-motor.json`: the new versioned mapping.
* `config.toml`: a fresh-learning configuration using that mapping.
* `calibration-report.json`, `declaration.json`, `report.json`: calibration,
  method/source identities and native interface/credit results.
* `preflight-thresholds.json`: explicit exploratory pool-width override.

Reproduce qualification using a new directory:

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/qualify_action_conditioned_motor.py \
  --output-dir artefacts/play-discard-motor-repeat \
  --pool-width 1 --cases 13 --repeats 4 --heavy
```

## Checkpoint and next-experiment boundary

Existing mapping/checkpoint files remain unchanged and retain their original
shared PLAY/DISCARD semantics. A new routing hash prevents interpreting those
weights as this new interface. Source/configuration files from before the
change were preserved with hashes under
`artefacts/source-snapshots/pre-action-conditioned-20261027-v1/`. Strict legacy
source guards intentionally reject the changed current implementation; do not
disable them or edit historical manifests to resume an old experiment.

The project virtual environment was repaired by pinning its interpreter and
base configuration to Python 3.14.8, replacing the moving `latest` link that
had started Python 3.15 against 3.14 packages.

This fixes interface expressivity and direct choice-credit isolation. Learning
good play/discard behavior requires a **fresh** experiment under the new
mapping, evaluation against its own initial policy and frozen controls, and
tasks containing both actions. A PLAY-only pair test cannot establish discard
learning. Keep reward/reference/exploration changes separate from that test.

The first full-game pilot has now completed with one verified Ante-1 win in
the subset-plastic arm. See [learning results and the prepared fresh-validation
command](PLAY_DISCARD_LEARNING_RESULTS.md). This is not yet a reliable policy
gain or evidence of learned discard quality.
