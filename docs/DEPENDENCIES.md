# Dependencies, licences, and external data

Flylatro keeps proprietary Balatro assets and FlyWire data outside this
repository. Dependencies are separated into small core, training, simulator,
connectome-build, and visualisation extras so routine tests do not import heavy
stacks.

| Component | Role | Licence/data terms | Pinning policy |
|---|---|---|---|
| NumPy | Core arrays and lightweight test doubles | BSD-3-Clause; installed metadata also lists permissive bundled-component licences | Bounded compatible release in `pyproject.toml` |
| pytest | Development tests | MIT | Development extra |
| PyTorch | Autodiff PPO and sparse CPU/GPU fly simulation | BSD-3-Clause | Training/fly extra; checkpoint records exact version |
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
