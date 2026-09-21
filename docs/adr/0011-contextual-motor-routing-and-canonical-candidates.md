# ADR 0011: contextual motor routing, canonical candidates and fixed normalization

Status: accepted and locally validated; real calibration is a dedicated-machine gate.

Supersedes the motor-allocation part of ADR 0007.

## Problem

The first motor calibration required globally disjoint neural pools across every
structured action head: `sum(all head widths) x pool_width` distinct outputs.
With the pinned contract that is 61 pools, or 122 distinct neurons at width two
— more MBONs than the annotated v783 MBON population plausibly provides, and far
more than the plastic-reachable subset. It also spent biological outputs on
reserved action-type slots that can never become legal in V1.

## Decision

**Contextual routing (`contextual-motor-routing-v1`).** Pools must be distinct
only where options compete inside one comparison. Four competition groups:

| group | pools | why distinct |
|---|---:|---|
| `action_type` | 13 | one argmax over the supported types |
| `card_count` | 6 | one argmax, read in the same decision as the type |
| `card_slot` | 10 | ten slots ranked simultaneously |
| `context_slot` | 6 | one target argmax |

`joker`, `consumable`, `shop` and `pack` share `context_slot`: the already
selected action type makes them mutually exclusive, so "slot 2" means shop slot
2, pack slot 2, joker slot 2 or consumable slot 2 depending on context. That is
a fixed interpretation rule, not a learned one. `card_slot` stays separate from
`context_slot` because `USE_CONSUMABLE` reads both at once.

35 pools, 70 neurons at width two. The routing table is versioned, hashed and
non-trainable.

**Supported action set.** `SUPPORTED_ACTION_TYPES` is 0-12. `MOVE_JOKER` (13)
needs a source and a destination slot the pinned `ACTION_SPEC` cannot carry, and
14-24 are index-stability placeholders with no upstream semantics. Reserved
slots route to `-1`, score `-inf`, own no neural population, and are counted
whenever the contract reports one as legal. The serialized artifact still spans
the full 25-wide mask/action contract and records
`full_width` / `scientifically_represented` / `reserved_unrepresented`.

**Canonical candidate universe.** For `mbon_direct` the motor candidates are
`unique(real_unshuffled_plastic_topology.post_root_ids)` at the canonical
`minimum_synapse_count = 1`; for `whole_brain` they are the annotated descending
population. The universe is computed from the real unshuffled artifact and
cannot move because topology was shuffled, a weak-edge sensitivity experiment
changed the threshold, or the control condition changed. Its SHA-256 is
persisted and checked when a configuration loads a motor artifact. Each
candidate records root ID, KC plastic input edge count, total KC->MBON
anatomical synapse weight, MBON type and compartment label where known.

**Fixed reward-free normalization (`reward-free-median-iqr-v1`).**
`normalized = (pool_activity - baseline) / scale` with `baseline` the corpus
median and `scale = max(IQR / 1.349, minimum_scale_hz)`. Both vectors are frozen
at calibration, hashed, persisted and applied by `FixedMotorInterface.decode`
before any comparison. The floor prevents an inert pool from being amplified
into a dominant score. Selecting responsive neurons is not enough: without this,
a pool with a high absolute baseline wins every comparison regardless of how it
responds to state.

**Selection quality.** Candidates are ranked by robust scale then dynamic range
then root ID, and assigned greedily with a within-group correlation ceiling, so
near-identical populations are not handed to competing alternatives when a
better fixed choice exists. Calibration reports candidate/selected counts, pool
width, pool reuse, baseline distribution, dynamic range, variance, normalized
dynamic range, silent and high-rate fractions, competing-pool correlation and
the effective number of distinct signals per group, and fails loudly when a
group cannot obtain enough usable diversity.

## Evidence and assumptions

- Biological evidence: annotated MBON, ALPN and descending populations, and the
  KC->MBON edge set, are connectome-derived.
- Modelling assumption: Balatro action heads have no natural fly semantics.
  Which MBON drives "shop slot 2" is a synthetic interface choice; reusing it
  for "pack slot 2" is an additional, explicitly declared interface choice.
- Engineering choice: competition groups, the correlation ceiling, the
  median/IQR estimator and the scale floor are engineering gates with
  configurable, documented thresholds.

## Consequences

The motor artifact schema changed: `pool_indices` + `routing` + `normalization`
replace the flat per-head pool table. Artifacts produced before this ADR are not
loadable and their reports are not valid evidence.
