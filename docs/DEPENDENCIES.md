# Dependencies, licences, and external data

Flylatro keeps proprietary Balatro assets and FlyWire data outside this
repository. Dependencies are separated into small core, training, simulator,
connectome-build, and visualisation extras so routine tests do not import heavy
stacks.

| Component | Role | Licence/data terms | Pinning policy |
|---|---|---|---|
| NumPy | Core arrays and lightweight test doubles | BSD-3-Clause; installed metadata also lists permissive bundled-component licences | Bounded compatible release in `pyproject.toml` |
| pytest | Development tests | MIT | Development extra |
| PyTorch | Sparse CPU/GPU fly simulation; autodiff only in the legacy PPO baseline | BSD-3-Clause | Training/fly extra; checkpoint records exact version |
| PyArrow/pandas | FlyWire artifact construction and neural Parquet export | Apache-2.0 / BSD-3-Clause | Data/recording extras |
| `jahankazimi078/balatroagent` | Seed-faithful headless Balatro PyO3 simulator | MIT | Exact Git commit; external source dependency |
| FlyWire FAFB v783 | Adult female fly connectome and annotations | CC BY-NC-SA 4.0 | User-supplied external files with SHA-256 verification; never vendored |
| Balatro | Live visual replay only | Proprietary | User-owned installation; no assets distributed |
| balatrobot v1.5.2 | Live local JSON-RPC game bridge | MIT | External mod installation only; Flylatro uses the documented HTTP protocol and distributes no game assets |

The selected Balatro simulator is pinned to commit
`38ae214317009952db4d22a98dc0765cef79370a`. Reference fly-dynamics code was
inspected at `vaibhavkedarisetti/fruit-fly-lab` commit
`26672e06427c12c61536ce1bd93dae7442944681` and
`lixiang1076/fly-brain` commit
`f6d525ab10c7a3ed4853d3d8b4b5e73f24cfe52b`; no source from either repository
is vendored. The FlyWire data terms, rather than the code licences, govern the
generated full-connectome artifact.

The upstream Balatro heuristic is not imported into training and may only be
invoked by an explicit external-baseline command.

## Biological and modelling sources for plastic-brain V1

The implementation distinguishes evidence from modelling choices:

| Source | What Flylatro uses | What it does not establish |
|---|---|---|
| Schlegel et al. 2024, DOI `10.1038/s41586-024-07686-5` | Structured v783 classifications, the five principal mushroom-body cell classes, and the reported 5,177-cell bilateral KC census | That every consolidated Codex type string is a synapse-level compartment label |
| Dorkenwald et al. 2024, DOI `10.1038/s41586-024-07558-y` | FAFB v783 as the whole-adult-brain anatomical graph | Biophysical synaptic efficacy or in-vivo dynamics |
| Aso et al. 2014, DOI `10.7554/eLife.04577` and `10.7554/eLife.04580` | Compartmental DAN/KC/MBON organization and MBON contributions to action selection | A natural mapping from Balatro actions to fly outputs |
| Handler et al. 2019, DOI `10.1016/j.cell.2019.05.040` | Timing-dependent dopamine-gated KC->MBON plasticity as biological motivation | The exact discrete Flylatro update equation or its game-scale timing |
| Bennett et al. 2021, DOI `10.1038/s41467-021-22592-4` | A computational precedent for KC->MBON plasticity and reinforcement signals | A requirement for an external learned critic; V1 begins with absolute outcome pulses |
| Gerstner et al. 2018, DOI `10.3389/fncir.2018.00053` | Eligibility traces as a transparent three-factor modelling convention | Fly-specific eligibility constants |

The fixed sensory random projection, structured motor pools, dopamine pulse
magnitudes, trace decay, efficacy bounds, simplified LIF model, and decision-
scale timing are Flylatro-specific engineering choices. They are versioned in
configuration and checkpoint hashes and must be calibrated empirically rather
than described as measured biological constants.
