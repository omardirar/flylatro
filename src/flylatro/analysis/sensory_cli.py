"""Create and audit a deterministic reward-free ALPN sensory mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.interface.sensory import SensoryMapping


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mapping-seed", type=int, required=True)
    parser.add_argument("--population-width", type=int, default=3)
    parser.add_argument("--max-rate-hz", type=float, default=150.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if not args.full:
        parser.error("real FlyWire sensory mapping requires explicit --full")
    artifact = FlyWireArtifact.load(args.artifact)
    mapping = SensoryMapping.from_artifact(
        artifact,
        mapping_seed=args.mapping_seed,
        population_width=args.population_width,
        max_rate_hz=args.max_rate_hz,
    )
    mapping.save(args.output)
    print(json.dumps({"output": str(args.output), "sha256": mapping.sha256, "collision_audit": mapping.collision_audit()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
