"""Materialize every protocol arm into an exact runnable configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol import ExperimentProtocol
from flylatro.learning.protocol_materialize import materialize_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument(
        "--budget-basis",
        required=True,
        help="measured benchmark or gate justifying the exposure budget",
    )
    parser.add_argument("--checkpoint-every-decisions", type=int, required=True)
    parser.add_argument(
        "--no-heavy",
        action="store_true",
        help="omit --heavy from the generated commands (development only)",
    )
    args = parser.parse_args(argv)
    protocol = ExperimentProtocol.load(args.protocol)
    base = PlasticExperimentConfig.load(args.base_config)
    plan = materialize_protocol(
        protocol,
        base,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        run_root=args.run_root,
        budget_basis=args.budget_basis,
        checkpoint_every_decisions=args.checkpoint_every_decisions,
        heavy=not args.no_heavy,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "protocol_sha256": plan["protocol_sha256"],
                "arms": len(plan["arms"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
