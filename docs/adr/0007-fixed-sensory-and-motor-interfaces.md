# ADR 0007: fixed sensory and motor interfaces

Status: accepted and locally validated; biological input/output calibration remains heavy validation.

## Decision

Observable simulator features map to annotated input/projection neurons via a
fixed deterministic random projection. `sensory_mapping_seed` selects a
reproducible assignment for replicate experiments; hashes include feature
layout, scaling, indices and root IDs. The encoder contains no strategy-derived
features.

The motor interface is a pure decoder over named population activity. It has
fixed pools for action types, card slots/counts and contextual targets. Legal
masks enforce game rules after neural scoring. Stable argmax is the default;
optional seeded sampling is explicit non-learned exploration. The mapping and
exploration configuration are hashed and checkpointed. MBON-direct pools are
restricted to MBONs that actually receive an identified plastic KC edge, so
the selected outputs are reachable by the learned state.

## Consequences

There are zero trainable motor parameters and no external policy state. The
synthetic mappings are necessary interfaces, not biological claims. Mapping
replicates measure dependence on arbitrary feature/population assignments.
