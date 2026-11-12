# No-skip / weak-failure outcome, 2026-10-16

Source run:
`runs/plastic-exploratory-event-no-skip-weak-failure-v1-20261016T091308Z`.
It completed the full 100,000-decision budget in 3,410.38 seconds at 29.32
decisions/s, with 12,644 completed episodes and no safety stop. The anatomy,
100 ms neural decision, mappings and learning rule are unchanged; skips are
disabled and the failure pulse is 0.01. No new long training run was launched.

## Stability and training outcomes

Mean efficacy is 1.019508, mean absolute drift 0.034065 and maximum drift
0.136217. Lower and upper bound fractions are both zero. All state-health
checks remain enabled. Allocation is 323.94 MiB GPU with 700 MiB reserved,
matching the prior native single-fly run rather than growing with exposure.

The durable logs have zero skip actions. They record 48 blind-clear events,
12,644 failures and zero Ante-1 wins. Episode records independently show 48
episodes reaching round 2 and 12,596 ending in round 1. Per 10,000 decisions,
blind clears are 5 / 7 / 4 / 4 / 3 / 4 / 5 / 3 / 6 / 7; there is no convincing
upward trend. The 48 clears are 0.380% of completed episodes.

Appetitive magnitude totals 158.512; aversive magnitude totals 126.440. The
previous approximately 75-fold negative imbalance is resolved. Stability alone
does not establish learning success, and these totals do not suggest another
blanket reduction in punishment.

During training, 70.4% of played hands use two or three cards; five-card plays
are 10.2%. This is descriptive, not a claim that playing fewer cards is always
wrong. The scoring quality of the selected cards needs direct testing.

## Corrected paired frozen validation

Both conditions use the same final checkpoint as provenance. The control loads
its verified initial efficacy, while the trained condition retains its final
efficacy. Learning is disabled. Both choose their own actions with deterministic
motor decoding on the same 256 validation seeds, Ante-1 target, and four frozen
lanes. Final-test seeds are untouched.

| Metric | Initial weights | Trained weights |
|---|---:|---:|
| Blind clears | 1 | 0 |
| Ante-1 wins | 0 | 0 |
| Mean episode return | 0.195888 | 0.202005 |
| Mean decisions per episode | 7.285 | 6.469 |
| Blind skips | 0 | 0 |

Mean paired return difference is +0.006118, with an exploratory paired bootstrap
95% interval of [-0.013780, +0.023555] (20,000 resamples, fixed seed 20261016).
This does not demonstrate improved return. One versus zero blind clears is too
sparse to establish a difference. There is no demonstrated win improvement.
Training uses stochastic exploration; these are greedy held-out measurements,
so the raw training clear rate is not a directly matched evaluation control.

Authoritative results are in `artefacts/no-skip-20261016-evaluation-v2`, including
`trained/summary.json`, `initial/summary.json`, complete transitions,
`comparison.json` and `assessment.json`.

The initial v1 evaluation exposed a bug: `_frozen_environment` constructed a new
adapter without forwarding the no-skip option. Training itself respected the
option. Frozen environment replicas now use `build_plastic_environment`, carrying
the whole configured environment policy. Paired comparisons also reject
different action constraints. The v1 outputs remain preserved and explicitly
invalidated in `evaluation-invalidated.json`; they are not used for outcomes.
The corrected transition logs contain zero blind skips in either condition.

## Does learned state affect motor choices?

A controlled reward-free probe compares initial and trained efficacy over the
same 96 corpus states, two repetitions and identical neural RNG. Of 192 samples,
90 complete actions change; action type changes on 62. Relative action scores
change on 120 of 138 samples with competing legal actions. Mean absolute output
change is 2.799 Hz. The learning-to-motor path is active; the failure is not
simply unchanged weights or a disconnected readout.

The probe's masks explicitly disable skips, independently of the former frozen
environment bug. Its data remain valid in the v1 directory's `motor-probe.json`
and `motor-probe-activity.npz`.

An additional diagnostic restores real simulator snapshots for 16 corpus states
and enumerates all legal PLAY_HAND subsets. Among the 15 sampled trained PLAY
choices, mean immediate environment reward is 0.0274; the mean best legal
one-step alternative in those same states is 0.2754. These are curated corpus
samples and an immediate-reward upper bound, not a long-term optimal policy or
a generalization result. This scoring oracle is diagnostic only and is never
used by the fly or its reward mapper. Results are in `hand-diagnostic.json`.

## Next experiment

Do not extend the unchanged run or reduce punishment further on this evidence.
Keep failure pulse 0.01 and learning rate 0.002 as the stable baseline. Next
investigate hand selection and choice-specific credit: the current global
reinforcement pulse changes every eligible pathway, including pathways that
did not determine the selected action. A separately versioned eligibility or
modulation experiment should test whether outcome updates favor the actual
selected motor choices. It must retain the current baseline, start fresh and
be judged by matched held-out blind clears/wins rather than only healthy
weights or total shaped reward. No learning-rule change is implemented here.

Lower motor temperature is not established as a remedy: it would increase
commitment to existing preferences and has no effect on the current greedy
evaluation. A temperature-only training comparison can be a secondary control,
but it should not replace testing whether the rule learns better hand choices.

New regression coverage checks real simulator replicas at widths 1 and 4 with
both skip settings, and paired comparison rejection of mismatched restrictions.
Targeted evaluation tests pass. Full post-fix regression: **534 passed, 10 CUDA
tests skipped**, in 31.25 seconds. Output is preserved in the v2 directory's
`regression.stdout`.
