"""Render activity-driven FlyWire SVG frames from a recorded replay bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.replay.bundle import ReplayBundle
from flylatro.visualization.brain import coordinate_lookup, render_activity_svg
from flylatro.visualization.timeline import TimelineConfig, build_timeline
from flylatro.visualization.video import ffmpeg_concat_command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--playback-seconds", type=float, default=1.0)
    parser.add_argument("--neural-duration-ms", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-neurons-per-frame", type=int, default=5000)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if args.max_neurons_per_frame > 5000 and not args.full:
        raise ValueError("more than 5,000 neurons per frame requires --full")
    if args.fps < 1:
        raise ValueError("fps must be positive")
    bundle = ReplayBundle.open(args.bundle)
    if len(bundle.decisions) > 10 and not args.full:
        raise ValueError("rendering more than 10 decisions requires --full")
    if bundle.neural_path is None:
        raise ValueError("replay bundle has no neural Parquet recording")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("visualization requires Flylatro's recording extra") from error
    artifact = FlyWireArtifact.load(args.artifact)
    table = pq.read_table(bundle.neural_path)
    events = table.to_pydict()
    timeline = build_timeline(
        bundle.decisions,
        TimelineConfig(
            neural_duration_ms=args.neural_duration_ms,
            playback_seconds_per_decision=args.playback_seconds,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "timeline.json").write_text(
        json.dumps(timeline.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decisions = np.asarray(events["decision_id"], dtype=np.int64)
    roots = np.asarray(events["flywire_neuron_id"], dtype=np.int64)
    activity = np.asarray(events["activity"], dtype=np.float32)
    roles = np.asarray(events["role"], dtype=object)
    times_ms = np.asarray(events["simulated_time_ms"], dtype=np.float32)
    kinds = np.asarray(
        events.get("event_kind", ["spike"] * len(decisions)), dtype=object
    )
    lines = ["ffconcat version 1.0"]
    frame_index = 0
    frame_duration = 1.0 / args.fps
    for entry in timeline.entries:
        neural_frames = max(1, int(round(args.playback_seconds * args.fps)))
        for local_frame in range(neural_frames):
            start_ms = local_frame / neural_frames * args.neural_duration_ms
            stop_ms = (local_frame + 1) / neural_frames * args.neural_duration_ms
            in_decision = decisions == entry.decision_id
            stimulated = in_decision & (kinds == "stimulation")
            spiking = in_decision & (kinds != "stimulation")
            spiking &= times_ms >= start_ms
            spiking &= (
                times_ms <= stop_ms
                if local_frame == neural_frames - 1 else times_ms < stop_ms
            )
            selected = stimulated | spiking
            visual_activity = activity[selected].copy()
            selected_kinds = kinds[selected]
            visual_activity[selected_kinds == "stimulation"] /= 150.0
            ids, values, selected_roles = _aggregate(
                roots[selected], visual_activity, roles[selected],
                args.max_neurons_per_frame,
            )
            coordinates = coordinate_lookup(
                artifact.root_ids, artifact.coordinates_nm, ids
            )
            name = f"frame-{frame_index:07d}.svg"
            render_activity_svg(
                args.output_dir / name,
                neuron_ids=ids,
                coordinates_nm=coordinates,
                activity=values,
                roles=selected_roles,
                title=(
                    f"Decision {entry.decision_id} | {entry.action_type} | "
                    f"p={entry.selected_action_probability if entry.selected_action_probability is not None else 'n/a'} | "
                    f"{start_ms:.1f}-{stop_ms:.1f} ms | "
                    f"neural time slowed {timeline.config.slowdown:g}x"
                ),
            )
            lines.extend((f"file '{name}'", f"duration {frame_duration}"))
            frame_index += 1
        if entry.end_seconds > entry.neural_end_seconds and frame_index:
            lines.extend(
                (
                    f"file 'frame-{frame_index - 1:07d}.svg'",
                    f"duration {entry.end_seconds - entry.neural_end_seconds}",
                )
            )
    concat_path = args.output_dir / "frames.ffconcat"
    if frame_index:
        lines.append(f"file 'frame-{frame_index - 1:07d}.svg'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = ffmpeg_concat_command(concat_path, args.output_dir / "brain.mp4")
    print(json.dumps({"frames": frame_index, "video_command": command}))
    return 0


def _aggregate(
    roots: np.ndarray, activity: np.ndarray, roles: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(roots):
        return roots, activity, roles
    unique, inverse = np.unique(roots, return_inverse=True)
    values = np.bincount(inverse, weights=activity)
    priority = {"internal": 0, "input": 1, "readout": 2}
    chosen_roles = np.full(len(unique), "internal", dtype=object)
    chosen_priority = np.zeros(len(unique), dtype=np.int8)
    for index, role in zip(inverse, roles):
        candidate = priority[str(role)]
        if candidate > chosen_priority[index]:
            chosen_roles[index] = role
            chosen_priority[index] = candidate
    order = np.argsort(values)[::-1][:limit]
    return unique[order], values[order].astype(np.float32), chosen_roles[order]


if __name__ == "__main__":
    raise SystemExit(main())
