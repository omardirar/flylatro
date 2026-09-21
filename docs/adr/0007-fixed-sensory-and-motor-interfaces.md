# ADR 0007: field-aware sensory coding and calibrated canonical motor pools

Status: accepted and locally validated; real activity calibration is pending.

## Decision

`plastic-balatro-field-aware-v2` classifies every upstream observation position.
One-hot and boolean features remain 0/1; bounded ratios remain bounded;
already-log nonnegative values use a saturating transform; signed-log values
use dual rails; raw counts use an explicit log bound; reserved slots must be
zero. The manifest records source index, semantic class and transform. No
strategy, hand-quality, expected-value or recommended-action feature exists.

The seeded ALPN projection stays synthetic and non-trainable. Collisions use
fixed clipped accumulation, not assignment-count averaging. This preserves a
single active categorical channel at full configured rate. Audits report ALPN
use, assignment quantiles, class collisions and isolated effective rates.

The scientific motor default is a persisted reward-free calibration artifact,
not round-robin single neurons. It selects non-silent, non-high-rate-only,
variable outputs into pools wider than one where population size permits. The
artifact records root IDs, pool indices, criteria, state seeds/hashes,
duration, calibration seed and hash. It consumes no reward, action correctness
or win information and has zero trainable parameters.

Canonical output root IDs come from the real unshuffled artifact. Real,
no-plasticity, KC->MBON shuffle, whole-brain shuffle, shuffled reinforcement,
and sensory replicates reuse exactly the same motor artifact. Exploration uses
derived `(motor_seed, learner_id, decision_id)` RNG, independent of row order.

## Evidence and assumptions

- Biological evidence: annotated ALPN, MBON and descending populations provide
  anatomically named input/output candidates.
- Modelling assumption: Balatro features and action heads have no natural fly
  semantics; their assignment is a synthetic interface.
- Engineering choice: field-aware amplitudes, clipped collisions, reward-free
  calibration and canonical hashes reduce avoidable attenuation/confounding.
