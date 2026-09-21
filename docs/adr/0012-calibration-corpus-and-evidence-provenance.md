# ADR 0012: frozen calibration corpus, evidence provenance and preflight profiles

Status: accepted and locally validated; real corpus generation is a dedicated-machine gate.

## Problem

Three defects shared one root cause — calibration evidence that could not prove
what it was measured on.

1. Motor and representation calibration walked the environment with whatever
   bootstrap decoder happened to exist, so the states they measured depended on
   an artifact the experiment was about to replace.
2. `mbon_direct` motor diagnostics measured descending activity while the motor
   interface consumed MBON activity, and nothing in the report could reveal it.
3. `flylatro-preflight` trusted `report["gates"]["status"] == "PASS"`, so a
   report measured at a different duration, sensory mapping or topology passed
   silently.

## Decision

**Frozen reward-free calibration corpus
(`reward-free-calibration-state-corpus-v1`).** A first-class, versioned,
SHA-256-hashed artifact of observable Balatro states and legal masks only. It
carries corpus version, simulator version, per-state hashes, seeds, phase
labels, generation method and coverage, and optionally exact environment
snapshots. It contains no reward, best action, heuristic recommendation,
expected value or strategy annotation. Deterministic scripted legal navigation
reaches diverse states; those actions are discarded and are never training
labels or policy targets. Sensory health, representation diagnostics, reward-free
motor calibration, mapping replicates and neural-duration comparisons all read
the same corpus.

**Explicit motor population.** `representation_diagnostics` takes
`motor_activity` and `motor_activity_population` as arguments and records which
population it measured: `mbon` in `mbon_direct`, `descending` in `whole_brain`.
The CLI supplies `output.neural.output_activity` — the exact array the motor
interface consumes.

**Pre-motor and post-motor stages.** `--stage pre` measures neural
representation before any motor artifact exists and makes no motor claim.
`--stage post` re-runs with the persisted final mapping and reports normalized
pool activity, per-head option ranges, action-option coverage, competition
behaviour and score distributions. Only the post-motor report can satisfy a
motor readiness gate.

**Evidence identity (`flylatro-evidence-identity-v1`).** Every generated report
carries git commit and dirty state, config SHA, simulator version, FlyWire
artifact/population/connectivity SHAs, plastic topology SHA, minimum synapse
threshold, sensory feature-contract and mapping SHAs, calibration corpus SHA,
motor candidate-set and mapping SHAs, reinforcement and plasticity rule SHAs,
fly dynamics SHA, simulation duration, output mode, topology condition, backend
and device. A field that is `None` is explicitly *not claimed* rather than
assumed to match.

**Provenance-verifying preflight.** Each gate declares which identity fields its
report must bind, compares them to the current experiment, and fails with an
exact mismatch message. A required field the report does not claim is a
mismatch: stale evidence must not pass by omission. Sensory readiness moves from
structural assignment counts to the state-conditioned report; structural counts
remain informational, because "how many dormant channels share an ALPN" and "how
many channels are simultaneously active on one ALPN" are different quantities
and only the second can destroy state information.

**Two profiles.** `initial` authorises the first tiny real experiments and
downgrades the benchmark, matched-control, tiny-run, protocol and specificity
gates to WARN. `ante1` requires all of them. Every threshold is configurable and
documented as an engineering gate, not a biological fact.

## Evidence and assumptions

- Biological evidence: none of this changes the model. It changes what can be
  claimed about a measurement.
- Modelling assumption: a corpus reached by scripted legal navigation is
  representative enough of the states a learning fly will meet. Phase coverage
  is reported, including phases the corpus failed to reach.
- Engineering choice: hashing, identity binding and profile separation.

## Consequences

Reports produced before this ADR carry no evidence identity and are refused.
Representation reports produced before the motor-population fix are invalid
evidence regardless of their recorded status.
