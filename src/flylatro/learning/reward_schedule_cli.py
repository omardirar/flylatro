"""Create a deterministic shuffled synthetic-reinforcement schedule.

The schedule replays the *same marginal* synthetic appetitive/aversive stream
as its source ``plastic_real`` run with the temporal order deterministically
broken.  It is therefore only a valid control for that exact run, and the
artifact records enough identity for the consuming arm to prove it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.reinforcement import ReinforcementPulse
from flylatro.learning.reward_schedule import (
    ReinforcementSchedule,
    ReinforcementScheduleSource,
)


#: The training logger writes ``synthetic_reinforcement_channels``.  ``pulses``
#: is the pre-rename key and stays readable so an archived log can still be
#: shuffled; both carry the same two synthetic channels.
EVENT_LIST_KEYS: tuple[str, ...] = ("synthetic_reinforcement_channels", "pulses")


def read_reinforcement_events(path: Path) -> Iterator[ReinforcementPulse]:
    """Parse one reinforcement-event log in either supported layout."""

    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            items: Sequence[Any] | None = None
            for key in EVENT_LIST_KEYS:
                if key in row:
                    items = row[key]
                    break
            if items is None:
                raise ValueError(
                    f"{path}:{number} has none of {list(EVENT_LIST_KEYS)}; this is "
                    "not a Flylatro synthetic-reinforcement event log"
                )
            for item in items:
                yield ReinforcementPulse(
                    appetitive=float(
                        item.get("synthetic_appetitive", item.get("appetitive", 0.0))
                    ),
                    aversive=float(
                        item.get("synthetic_aversive", item.get("aversive", 0.0))
                    ),
                    events=tuple(item.get("events", ())),
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-run-manifest",
        type=Path,
        help="run-manifest.json of the source arm (default: next to --events)",
    )
    parser.add_argument(
        "--source-arm-id",
        default="",
        help="protocol arm ID of the source plastic_real run",
    )
    parser.add_argument(
        "--target-arm-id",
        default="",
        help="protocol arm ID of the shuffled_reward arm that will consume this",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    pulses = list(read_reinforcement_events(args.events))
    manifest = load_plastic_manifest(args.source_checkpoint)
    run_manifest_path = args.source_run_manifest or (
        args.events.resolve().parent / "run-manifest.json"
    )
    run_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    components = run_manifest.get("components", {})
    event_log_sha256 = sha256_file(args.events)
    recorded = components.get("synthetic_reinforcement_event_log_sha256")
    if recorded is not None and recorded != event_log_sha256:
        parser.error(
            "the supplied --events file is not the one the source run recorded: "
            f"file={event_log_sha256} run-manifest={recorded}"
        )
    source = ReinforcementScheduleSource(
        arm_id=str(args.source_arm_id or components.get("protocol_arm_id") or ""),
        replicate_id=str(components.get("protocol_replicate_id") or ""),
        condition=str(components.get("condition") or ""),
        protocol_sha256=str(components.get("protocol_sha256") or ""),
        run_manifest_sha256=(
            sha256_file(run_manifest_path) if run_manifest_path.exists() else ""
        ),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        plastic_weight_sha256=str(manifest["plastic_weight_sha256"]),
        event_log_sha256=event_log_sha256,
        action_schedule_sha256=str(
            components.get("executed_action_schedule_sha256")
            or components.get("action_schedule_sha256")
            or ""
        ),
    )
    schedule = ReinforcementSchedule.shuffled(
        pulses,
        source_weight_hash=manifest["plastic_weight_sha256"],
        seed=args.seed,
        source=source,
        target_arm_id=str(args.target_arm_id or ""),
    )
    schedule.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pulses": len(schedule.pulses),
                "schedule_sha256": schedule.sha256,
                "shuffle_seed": schedule.shuffle_seed,
                "source_arm_id": schedule.source.arm_id,
                "target_arm_id": schedule.target_arm_id,
                "source_event_log_sha256": schedule.source.event_log_sha256,
                "source_run_manifest": (
                    str(run_manifest_path) if run_manifest else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
