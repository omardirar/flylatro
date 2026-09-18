"""Create a deterministic shuffled-dopamine schedule from a matched real run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.reinforcement import DopaminePulse
from flylatro.learning.reward_schedule import DopamineSchedule


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pulses = []
    with args.events.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pulses.extend(
                DopaminePulse(
                    appetitive=float(item["appetitive"]),
                    aversive=float(item["aversive"]),
                    events=tuple(item.get("events", ())),
                )
                for item in row["pulses"]
            )
    manifest = load_plastic_manifest(args.source_checkpoint)
    schedule = DopamineSchedule.shuffled(
        pulses,
        source_weight_hash=manifest["plastic_weight_sha256"],
        seed=args.seed,
    )
    schedule.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pulses": len(schedule.pulses),
                "schedule_sha256": schedule.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
