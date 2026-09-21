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
        "--sensory-mapping-seed",
        type=int,
        default=0,
        help=(
            "fly.sensory_mapping_seed of the primary mapping block; it is fixed "
            "across the ordinary stochastic replicates"
        ),
    )
    parser.add_argument(
        "--mapping-sensitivity",
        action="append",
        default=[],
        metavar="SENSORY_SEED:MOTOR_STRUCTURE_SHA256",
        help=(
            "add an explicit sensory-mapping replicate block; each needs its own "
            "motor mapping calibrated through that mapping. Repeatable"
        ),
    )
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
        sensory_mapping_seed=args.sensory_mapping_seed,
        mapping_sensitivity_variants=_mapping_variants(args.mapping_sensitivity),
        reinforcement_condition=args.reinforcement_condition,
    )
    protocol.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": protocol.sha256,
                "sensory_mapping_blocks": protocol.mapping_replicate_count,
                "stochastic_replicates_per_block": protocol.replicate_count,
                "arms": len(protocol.arms),
            },
            sort_keys=True,
        )
    )
    return 0


def _mapping_variants(values: Sequence[str]) -> tuple[dict[str, object], ...]:
    """Parse ``SEED:MOTOR_SHA`` sensory-mapping sensitivity blocks."""

    variants: list[dict[str, object]] = []
    for position, raw in enumerate(values, start=1):
        seed, _, motor = raw.partition(":")
        if not seed.strip() or not motor.strip():
            raise SystemExit(
                f"--mapping-sensitivity expects SENSORY_SEED:MOTOR_STRUCTURE_SHA256, got {raw!r}"
            )
        variants.append(
            {
                "mapping_id": f"mapping-{position:03d}",
                "sensory_mapping_seed": int(seed),
                "motor_mapping_id": motor.strip(),
            }
        )
    return tuple(variants)


if __name__ == "__main__":
    raise SystemExit(main())
