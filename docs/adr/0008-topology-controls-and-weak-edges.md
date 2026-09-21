# ADR 0008: topology controls, weak edges, and v783 data quality

Status: accepted and locally tested; full-v783 reports need the supplied data.

## Decision

Topology has three explicit values:

- `real` keeps every retained pair unchanged;
- `kc_mbon_shuffled` permutes MBON postsynaptic labels only at identified
  KC->MBON entries. All other edges, edge entries, weights, KC out-degree and
  the MBON in-degree distribution remain unchanged;
- `whole_brain_shuffled` performs the wider population-preserving
  postsynaptic-label permutation.

These answer different causal questions and cannot share an ambiguous name.
Both controls retain canonical output roots and the same motor mapping.

The primary artifact retains all positive nonzero KC->MBON pairs. Diagnostics
report thresholds 1, 2, 5 and 10 with pair/synapse counts and fractions. An
optional `minimum_synapse_count` filters plastic topology, enters its hash and
manifest, and defines a separate sensitivity experiment; default one is never
silently changed.

The builder hard-gates only the supported exact 5,177 KC census. It reports
and hashes MBON, DAN, PAM, PPL1, ALPN, APL, DPM, descending and KC->MBON
populations with rules, primary types and root IDs. It records unresolved
neurotransmitter neurons, invalid endpoints, dropped connections/synapses,
source-synapse fraction and missing coordinates instead of hiding them.

## Classification

The 5,177 KC reference and structured v783 annotations are biological/data
evidence. Shuffle preservation rules and weak-edge thresholds are modelling
controls. Hashes and fail-fast manifests are engineering safeguards.
