# ADR 0002: FlyWire populations and fixed Balatro encoding

Status: implemented in code; needs full-data and GPU validation.

## Dataset and model

V1 targets the FlyWire FAFB v783 adult female whole-brain connectome:
139,255 neurons, 3,732,460 unique neuron-pair connections and 50,666,648
synapses. The external source files are CC BY-NC-SA 4.0 and are not distributed
with Flylatro. The dynamics implement the fixed whole-brain LIF model described
by Shiu et al. (2024), following its published constants and the inspected
PyTorch batch execution pattern.

## Input population

The encoder uses FlyWire neurons annotated `super_class == sensory`. V783 has
16,938 such neurons. Features are assigned in a stable order to disjoint blocks
of three sensory neurons, with neurons ordered by ascending FlyWire root ID.
The exact simulation indices and root IDs are persisted in the artifact and
included in the encoder hash.

Balatro has no biological sensory analogue. The assignment therefore makes no
claim that a suit is naturally visual, olfactory, or gustatory. The biological
constraint is narrower and reproducible: signals enter through real sensory
neurons, remain fixed for the whole experiment, and must traverse the real
connectome before reaching the readout.

## Encoding

The pinned simulator observation contract is encoded in full:

- existing card and state one-hot values remain bounded one-hot drive;
- Joker, consumable and shop IDs use separate per-slot categorical
  populations;
- signed scalar groups use positive and negative rails;
- raw deck/draw counts are clipped at eight copies;
- upstream log-scaled scalar groups are clipped to documented fixed scales;
- all stimulation lies in 0–150 Hz.

The encoder is immutable and non-trainable. Its version and SHA-256 cover the
feature layout, stimulation range and exact sensory root IDs.

## Readout population

V1 reads all neurons annotated `super_class == descending` (1,305 in v783).
Descending neurons are a biologically motivated brain-output population. Their
spike rates and terminal voltages form the fixed neural representation. The
downstream policy is a direct structured linear readout plus critic, with no
hidden network by default; its trainable parameter count is logged.

## Control topology

The shuffled-connectome control globally permutes postsynaptic endpoints while
retaining every presynaptic edge entry and weight. It exactly preserves the
edge-entry count, presynaptic out-degree sequence, postsynaptic in-degree
sequence, signed-weight distribution and presynaptic neurotransmitter identity.
It destroys biological pairings; parallel edges/self-loops are allowed and
coalesced by the sparse backend. The seed and procedure string are recorded.

