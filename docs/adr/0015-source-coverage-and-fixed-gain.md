# ADR 0015: explicit source coverage and fixed-dynamics calibration

Status: source repair implemented and built; corrected-graph calibration passes
at gain 0.225 / 100 ms; independent stability/control/resume validation passes;
the three paired primary runs are in progress.

PLAN.MD requires minimum-count-1 anatomy. The historical artifact used
`connections_princeton.csv.gz`, whose KC->MBON pairs are already filtered at
five aggregate synapses. A configuration threshold of one cannot recover the
missing pairs. Direct scans of both pinned local exports establish:

| Source | KC->MBON pairs | Anatomical synapses |
|---|---:|---:|
| Historical threshold-5 export | 26,937 | 226,932 |
| Unthresholded export | 67,504 | 314,975 |

The builder defaults to `connections_princeton_no_threshold.csv.gz`, pins its
SHA-256, records the source threshold, and refuses nonempty output directories.
The explicit `--connection-source threshold5` route preserves historical
reproduction. Historical artifacts/checkpoints/reports remain intact. Preflight
requires unthresholded coverage for V1; earlier PASS reports did not check this
requirement and cannot establish it retrospectively.

The new artifact retains the same 139,255 neurons, 5,177 KCs and annotation
rules, with 19,773,733 whole-brain pairs and 76,944,499 synapses. Its canonical
plastic-reachable motor universe includes all 96 annotated MBONs; historical
calibration used 84. Source-dependent mappings, representation, reachability,
plasticity, specificity, benchmark and control evidence must be regenerated.

At the existing fixed LIF synapse gain 0.275 mV and 500 ms window, full-corpus
representation fails: MBON high-rate fraction is 0.30212 against 0.25, and
separation is 29.09636 / 26.83191 = 1.0844 against 1.10. Reachability and
reward-free motor calibration pass. No threshold is relaxed and no training is
authorized over the representation failure.

The already existing `ShiuLIFParameters.synapse_scale_mv` is exposed as a fixed
configuration parameter and enters dynamics provenance. Its default remains
0.275, preserving historical dynamics identities. It is never learned or
reward-fitted. The named sensitivity predeclares gain 0.18, approximately
0.275 * 50,666,648 / 76,944,499, and windows 100/250/500 ms in ascending order.
The ratio is an engineering comparison of aggregate drive, not biological
equivalence. Selection requires all unchanged readiness gates, without using
game performance or fitting diagnostic classifiers. Anatomy, primary
three-factor equation, eligibility references, rewards and fixed seeded sensory
mapping remain as declared. Fresh motor calibration uses observable states and
neural activity only.

The declared 0.18 variants pass pre-representation/reachability but fail motor
action-type diversity at every window. A regression against Pearson correlation
found inconsistent covariance/std denominators in motor and representation
audits. Motor v5 and representation v4 repair the estimator; fresh reports
preserve all original thresholds and all older evidence. Cached 0.18 activity
still fails after correction. The prospectively ordered fixed-gain follow-up
tests 0.20/0.225/0.25/0.275 at 100 ms. Gain 0.20 fails diversity; 0.225 is the
first passing configuration. It uses 55 eligible candidates and 34 unique
neurons in 70 memberships, with minimum effective fraction 0.511395 > 0.50.
Full pre/post representation, reachability and integrated plasticity pass.
This selection uses no reward, game performance or fitted diagnostic probe.
Independent corrected-graph 250-decision training completes safely with 106
update events, zero bounds and zero training wins. Full 96-state, three-repeat
learned-checkpoint health and exact no-plasticity control pass. The real GPU
full-trainer resume probe passes 13 future steps, including positive progress
on its first resumed step. Strict preflight passes for all six materialized
1000-decision arms; staged execution is in progress.

The resume probe exposed two implementation defects. The pinned simulator's
RNG-state HashMap has arbitrary binary serialization order. Audit identity now
canonicalizes only that verified map and preserves every remaining byte; raw
checkpoint/file hashes remain unchanged. The adapter also lost its prior
observable state on restore, dropping the first resumed transition's reward
decomposition. Restore now initializes that state from the simulator. These
repairs change neither reward formula nor learning equation. Preserved failed
and successful probes document the distinction.

Unit-efficacy activity caches can be reused across pre/motor/post reports only
when anatomy, dynamics, sampling and corpus identities match exactly. Learned
checkpoint diagnostics explicitly refuse them.

Evidence and resource accounting are in `docs/PROGRESS_LOG.md` and
`artefacts/lr-sensitivity-20261009/`. A calibration PASS, lower LR or safe
exposure remains distinct from demonstrated held-out Ante-1 learning.
