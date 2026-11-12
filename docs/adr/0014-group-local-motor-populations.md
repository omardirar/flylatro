# ADR 0014: group-local motor allocation and selected-output reachability

Status: implemented; all real initial calibration gates pass at 500 ms.

## Problem

The first real 50 ms experiment found 18 globally variable MBONs among 84
canonical candidates. The calibrator required 70 globally distinct neurons for
35 pools of width two. That constraint exceeded the fixed motor mapping's
actual validator, which requires distinct populations only inside a comparison
group. It also rejected candidates using a whole-corpus IQR before considering
variation in the contexts where each head reads them.

Reachability treated the entire canonical candidate universe as selected motor
outputs, even before a motor mapping existed. Silent candidates that calibration
would never select then failed the selected-motor coverage gate.

## Decision

`reward-free-neural-motor-calibration-v4` selects candidates separately in each
group's interpretation states. Competing pools in a group remain disjoint and
retain at least two neurons per pool. Different comparison groups may reuse
neurons. All 35 routed pool IDs remain separate; the four mutually exclusive
target heads continue to share their contextual route. At width two, the
largest comparison requires 26 distinct candidates, while the artifact records
70 total memberships plus its actual unique selected count.

The same physical activity may therefore participate in different independently
normalized comparisons. This is an explicit synthetic decoder convention, not
evidence of natural fly action semantics. Cross-head statistical dependence is
possible and remains an experimental limitation. Within-context correlation,
effective signal diversity, dynamic range and legality gates remain unchanged.
No reward, action quality, gradient or trainable decoder enters calibration.

Selecting individually diverse neurons alone can still yield redundant pool
averages. Width-two assignments below the existing diversity threshold receive
a bounded deterministic pair-membership search: 32 fixed greedy starts and at
most 200 strictly improving single-member exchanges per start maximize pooled
correlation participation ratio. The search uses only frozen reward-free neural
activity, preserves disjoint membership, fits no weights and retains the
original assignment unless improved. Candidate enumeration is bounded to at
least 64 ranked candidates including the initial members. Flat/redundant input
still fails. Selection is recorded as
`reward-free-context-conditioned-disjoint-pool-diversity-v1`.

Optional `.npz` activity caches are SHA-256 verified and bound to anatomy,
dynamics, sensory mapping, candidate universe, corpus, device/backend, initial
unit efficacy and exact sampling settings. They allow deterministic membership
experiments on one fixed measurement, not reuse across different neural models.

`kc-subtype-reachability-v2` makes no motor-coverage claim in the pre-motor
stage. After calibration it measures the unique neurons actually read by the
persisted pools and binds their mapping and candidate-set hashes in preflight.
Full plastic-MBON coverage and KC/plastic-edge coverage are still measured
separately against their original thresholds. Unused silent candidates cannot
fail selected-output coverage or disappear from the full population report.

## Neural duration

A 500 ms ALPN-only pilot raised the MBON state-separation ratio from about 1.05
at 50 ms to 1.39, with the original 1.10 threshold. The full 96-state corpus
with three independent repeats passes pre/post representation at 1.321.
Motor calibration finds 47 eligible candidates; the final assignment uses 31
unique neurons and passes every context, including action-type diversity 0.519
against the unchanged 0.50 requirement. All selected motor neurons are active;
full plastic-MBON coverage is measured separately. A 20-decision integrated
plasticity calibration passes the original safety gates. These are recorded
changes to the fixed neural observation
window, not a retuning against game reward. Fresh pre/post representation,
reachability, motor calibration and plasticity reports must match that duration
before the tiny real gate can start.

## Consequences

The motor allocation change alters the mapping and calibration hashes. Old
motor and reachability reports cannot authorize the new configuration. Matched
conditions still reuse the same exact frozen motor artifact. The anatomical
graph, canonical candidate universe, sensory mapping, plastic edges and
three-factor learning rule are unchanged.
