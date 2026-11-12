# KC-to-MBON pathway and choice-credit audit

The completed audit identifies substantial cross-head credit aliasing and
reference-rate clipping. Identity remains decodable in the gated fresh
eligibility increments, while card MBON and pooled-rate probes fail the
corrected comparison. The next candidate experiment is a short matched pilot
with separate MBONs for card, action-type, and card-count pools, followed by
reward-free normalization at the existing 100 ms window. These measurements
do not establish the cause of poor gameplay.

This follows the [offline identity probe](CARD_IDENTITY_DECODING.md), which found
strong linear rank/suit decoding in KCs and weak decoding in MBON rates. It asks
whether the fixed card readout has anatomical access to those KCs, whether
fixed-reference clipping removes the signal from fresh eligibility increments,
and whether shared motor neurons cause choices in one head to admit credit to
unchosen card outputs.

This is an exploratory analysis of the same cached observations and neural
responses. The original 192 fitting runs, 64 tuning runs, and 64 evaluation runs
remain separated. Evaluation contexts have already been used in the preceding
probe and are not a new independent confirmation. No neural simulation,
environment stepping, reward application, checkpoint writing, or agent changes
occur in this audit. A normal strict checkpoint load verifies the cached state
and model components before using its anatomy or efficacies.

## Measurements

The source is `artefacts/card-decoding-20261024-v1`; outputs are saved in
`artefacts/card-credit-20261024-v2`. Source files, corpus identities, per-run
selection records, cached labels, and response hashes are verified. The new
declaration records their hashes, the current implementation hash, and the
analysis before fitting follow-up probes. All nine plastic state fields must
match the source cache and remain unchanged afterward.

Anatomical summaries count KC-to-MBON neuron-pair edges, unique reachable KCs,
per-card-pool connections, shared postsynaptic MBONs across structured motor
heads, and actual checkpoint weight changes. Counts include all ten fixed card
pools, including the two unavailable slots in the selected eight-card contexts.

The following eight feature sets use the same offline ridge method as the
source identity probe:

| Feature set | Construction |
| --- | --- |
| Connected KCs | Raw rates of KCs with at least one edge into any card pool |
| Clipped connected KCs | Clip each repeat's KC rate/reference to [0, 1], then average repeats |
| All afferent averages | Anatomical-synapse-count × efficacy weighted KC-rate average per MBON |
| Card afferent averages | The same static averages restricted to card MBONs |
| Card MBONs | Observed rates of card-pool MBONs |
| Card pool rates | Observed mean rate per fixed card pool |
| Fresh card-edge eligibility | Edge-aligned pre/post clipped coactivity, bounded by the configured eligibility maximum, from zero previous traces |
| Gated card-edge eligibility | The same increment after the existing union-of-selected-heads credit gate |

Nonlinear clipping, multiplication, and gating happen per neural repeat before
averaging by game run. Fresh eligibility is validated against the actual
three-factor rule in tests. These increments do not include previous decaying
traces, accumulated dopamine, rewards, or subsequent efficacy changes.

The static afferent average is a linear rate proxy. It omits temporal integration,
spike timing, other synaptic inputs, inhibition, and network feedback. It must
not be interpreted as a prediction of the simulated MBON rate or a causal
intervention on neural dynamics.

## Credit audit

Reconstruct deterministic choices from each cached repeat using the original
fixed motor interface and original legal masks. Apply the real selected-pool
rule through an equivalent CPU `SelectedMotorCredit` assignment, whose manifest
must match the loaded agent. These are current frozen diagnostic choices, not
the executed training history or the exploratory training policy.

For every trial, distinguish posts in explicitly selected competing card pools
from all posts admitted by the union of selected action, count, card, and context
pools. Count admitted card-targeting edges outside the explicitly selected card
posts. This is the defined cross-head credit alias measure. An admitted edge can
still have zero coactivity; admission does not establish an actual update.
Also measure the alias fraction among nonzero new card-edge increments and the
fraction of new eligibility mass on those edges. These quantify the immediate
update opportunity under a hypothetical pulse, without applying one or assuming
an observed reward. The final v2 artifact adds these activity-weighted quantities
to the initial v1 admission audit; feature values and decoding methods are unchanged.

Also report per-slot selection and edge-admission fractions, action/count
histograms, the fraction of admitted edges with nonzero coactivity, and pre/post
reference-rate saturation. Repeats remain repeated measurements within a run.

## Statistical scope

Fit on original fitting runs, choose regularization on tuning runs, and evaluate
on the original held-out runs. Use 499 whole-run training-label shuffles, with
the same shuffle schedule across stages and learners. Average statistics across
three trained learners and apply Holm correction across all 16 stage/task
comparisons. Initial weights provide a secondary control. Report class-balanced
accuracy, ordinary accuracy, training-only slot-majority controls, and whole-run
bootstrap intervals. All conclusions remain exploratory because these contexts
were already examined in the preceding diagnostic.

## Reproduce

```bash
CC="$PWD/artefacts/toolchains/gcc14/usr/bin/gcc-14" \
CXX="$PWD/artefacts/toolchains/gcc14/usr/bin/g++-14" \
TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m flylatro.analysis.card_credit_cli \
  --output-dir artefacts/card-credit-20261024-v3 --heavy
```

Use a new output directory. `--heavy` authorizes constructing the original neural
stack for strict checkpoint verification; the audit reuses the saved responses.
Each learner's derived features and run-level credit diagnostics are saved in
an NPZ file. Individual probe files retain predictions and shuffled-label null
scores. The report binds these files to hashes and retains plastic-state audits.
`assessment.json` contains the corrected exploratory comparisons.

## Results, October 24

The full anatomy contains 67,504 plastic KC-to-MBON neuron-pair edges. The ten
card pools consume 20 MBONs with 14,323 incoming edges from 4,933 of 5,177 KCs
(95.3%). Thus access to the KC population is broad, although individual card
pools have very different incoming connectivity. Thirteen of those 20 MBONs
also occur in action-type pools, and eleven also occur in card-count pools.
These overlaps count shared neurons, not shared pool IDs.

Mean held-out class-balanced accuracy across three trained learners:

| Stage | Rank | Suit |
| --- | ---: | ---: |
| Uniform chance | 7.69% | 25.00% |
| Training-only slot majority | 9.16% | 25.31% |
| Connected KCs | 33.07% | 66.17% |
| Clipped connected KCs | 18.32% | 51.72% |
| All static afferent averages | 11.91% | 41.93% |
| Card static afferent averages | 11.52% | 35.70% |
| Observed card MBON rates | 8.70% | 27.55% |
| Observed card pool rates | 9.15% | 27.33% |
| Fresh card-edge eligibility before choice gating | 17.50% | 50.88% |
| Fresh card-edge eligibility after choice gating | 13.45% | 43.54% |

Both tasks pass the exploratory corrected screen at all stages except observed
card MBON and card pool rates. Passing comparisons have Holm-adjusted p = 0.032;
the four failed comparisons have adjusted p = 0.168. These comparisons concern
this follow-up family and card-specific populations. They do not replace the
preceding study's all-MBON comparison or constitute independent confirmation.

Clipping weakens decoding but does not erase identity. Among admitted card edges,
59.2% of presynaptic rates and 87.4% of postsynaptic rates reach the 20 Hz reference
ceiling. The static projection also retains measurable identity. Its lower
decoding performance is consistent with aggregation losing recoverable detail,
but a static rate average cannot isolate the effects of neural dynamics.

Cross-head credit aliasing is substantial on the diagnostic choices:

| Quantity, trained-learner mean | Value |
| --- | ---: |
| Trials admitting any credit to an unchosen card output | 97.27% |
| Admitted card-targeting edges outside explicitly selected card posts | 20.25% |
| Nonzero fresh card-edge increments outside those posts | 16.68% |
| Fresh card-edge eligibility mass outside those posts | 16.24% |

The last measure ranges from 15.47% to 17.43% across trained learners. It shows
nonzero coactivity, not just an open gate. Even the tenth card output, which is
unavailable in every selected eight-card observation, has a mean incoming-edge
admission fraction of 61.73% because its MBONs are shared with other heads. This
follows directly from the implemented union rule and the shared assignment of
neurons to different motor roles.

An edge described as unchosen here may legitimately serve a chosen action or
count role. The finding is conflicting role assignment under the current
interface, not proof that the gate implementation violates its declared rule.
Actual historical reward pulses and decaying traces were not reconstructed, so
these percentages must not be described as the fraction of training updates or
reward wasted. They measure immediate eligibility opportunities on the cached
frozen choices.

The checkpoint has changed about 58.2% of card-targeting edges from unit
efficacy, but changes are small: mean absolute card-edge deltas are 0.00307,
0.00192, and 0.00474 for the three learners. These facts do not identify an
appropriate learning-rate or punishment change.

All 609 tests pass. The final audit verifies 18 source files, four derived-feature
files, 32 probe files, and all 64 reported task metrics from saved predictions.
All nine plastic state fields remain unchanged. The v2 features, predictions,
and shuffled-label scores reproduce v1 exactly; v2 adds the activity-weighted
alias metrics. Source checkpoints, mappings, and defaults remain unchanged.

Keep 100 ms, failure pulse 0.01, and disabled blind skips. A candidate with
separate card/action/count neurons needs its own reward-free calibration and
fresh matched pilot, since changing motor and credit identities invalidates
ordinary checkpoint compatibility. No long training run is justified by this
diagnostic alone.

The completed [matched separated-pool pilot](SEPARATED_MOTOR_PILOT.md) eliminates
the measured card-admission aliasing but does not pass its blind-learning screen
after 10,000 decisions per learner. Longer-budget learning remains untested.
