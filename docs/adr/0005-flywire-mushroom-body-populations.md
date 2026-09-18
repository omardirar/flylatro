# ADR 0005: FlyWire mushroom-body population identification

Status: accepted schema; needs full v783 census validation.

## Decision

Population membership comes from the pinned v783 structured `classification`
and consolidated cell-type tables. KC, MBON and DAN use exact `class` values
`Kenyon_Cell`, `MBON` and `DAN`. PAM and PPL1 refine DAN membership using
anchored resolved `primary_type` families. Antennal-lobe projection inputs use
exact `class == ALPN`; descending outputs retain exact
`super_class == descending`. APL and DPM are exact resolved-type families.

The artifact persists indices and root IDs for every population, resolved type
labels for KC/MBON/DAN cells, population-rule strings, hashes, and the exact
edge positions for all KC->MBON neuron-pair connections. Ambiguous or missing
annotations stay unknown; community free-text labels are not guessed.

## Rationale and limits

The v783 annotation paper describes KCs, MBONs, DANs, DPM and APL as the main
mushroom-body classes and reports 2,597 right plus 2,580 left KCs. The builder
uses the downloadable structured tables that accompany that release. Resolved
type strings can preserve subtype and coarse compartment names, but a
neuron-pair edge does not identify each synapse's neuropil compartment.

## Alternatives

Name substring matching over every label was rejected as irreproducible.
Importing a newer materialization silently was rejected because root IDs and
annotations may drift. Future releases require a new artifact version and
explicit migration.
