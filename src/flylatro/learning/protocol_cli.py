"""Create a reproducible replicate/control protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.learning.protocol import CONTROL_CONDITIONS, ExperimentProtocol
from flylatro.learning.reinforcement import SENSITIVITY_CONDITIONS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--conditions", default=",".join(CONTROL_CONDITIONS))
    parser.add_argument("--exposure-budget-decisions", type=int, required=True)
    parser.add_argument("--curriculum-ladder", default="1,2,3,5,8")
    parser.add_argument("--motor-mapping-id", required=True)
    parser.add_argument(
        "--reinforcement-condition",
        choices=tuple(SENSITIVITY_CONDITIONS),
        default="primary-progress",
        help="predeclared reinforcement-shaping sensitivity condition",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = ExperimentProtocol.create(
        name=args.name,
        base_seed=args.base_seed,
        replicate_count=args.replicates,
        conditions=tuple(value.strip() for value in args.conditions.split(",") if value.strip()),
        exposure_budget_decisions=args.exposure_budget_decisions,
        curriculum_ladder=tuple(int(value) for value in args.curriculum_ladder.split(",")),
        motor_mapping_id=args.motor_mapping_id,
        reinforcement_condition=args.reinforcement_condition,
    )
    protocol.save(args.output)
    print(json.dumps({"output": str(args.output), "sha256": protocol.sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
