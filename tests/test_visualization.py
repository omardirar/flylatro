from __future__ import annotations

import numpy as np

from flylatro.visualization.brain import (
    AnatomicalTransform,
    coordinate_lookup,
    render_activity_svg,
)
from flylatro.visualization.timeline import TimelineConfig, build_timeline
from flylatro.visualization.video import (
    ffmpeg_composite_command,
    ffmpeg_x11_capture_command,
)


def test_timeline_labels_neural_slowdown_and_cursor_cues() -> None:
    timeline = build_timeline(
        [
            {
                "decision_id": 0,
                "action": {"type": "play_hand"},
                "reward": 0.5,
                "dopamine_appetitive": 0.4,
            },
            {"decision_id": 1, "action": {"type": "buy"}},
        ],
        TimelineConfig(
            neural_duration_ms=50,
            playback_seconds_per_decision=1,
            inter_decision_pause_seconds=0.2,
        ),
    )

    assert timeline.config.slowdown == 20
    assert timeline.neural_time_to_video(1, 25) == 1.7
    assert timeline.entries[0].cursor_cue == "hand"
    assert timeline.entries[1].cursor_cue == "shop"
    assert timeline.entries[0].selected_action_probability is None
    assert timeline.entries[0].reward == 0.5
    assert timeline.entries[0].dopamine_appetitive == 0.4


def test_activity_renderer_uses_only_selected_neurons(tmp_path) -> None:
    roots = np.array([30, 10, 20], dtype=np.int64)
    coordinates = np.array([[3, 4, 0], [1, 2, 0], [2, 3, 0]], dtype=np.float32)
    selected = coordinate_lookup(roots, coordinates, [10, 30])

    path = render_activity_svg(
        tmp_path / "frame.svg",
        neuron_ids=[10, 30],
        coordinates_nm=selected,
        activity=[1.0, 2.0],
        roles=["input", "readout"],
        title="Decision 0 - neural time slowed 20x",
    )

    content = path.read_text(encoding="utf-8")
    assert content.count("<circle") == 2
    assert "slowed 20x" in content


def test_anatomical_transform_keeps_neuron_position_fixed_across_frames(
    tmp_path,
) -> None:
    whole_brain = np.asarray(
        [[0.0, 0.0, 0.0], [100.0, 100.0, 0.0], [25.0, 75.0, 0.0]]
    )
    transform = AnatomicalTransform.from_coordinates(whole_brain)
    first = transform.project(np.asarray([[25.0, 75.0, 0.0]]))
    second = transform.project(
        np.asarray([[25.0, 75.0, 0.0], [100.0, 100.0, 0.0]])
    )
    np.testing.assert_array_equal(first[0], second[0])
    path = transform.save(tmp_path / "transform.json")
    assert transform.sha256 in path.read_text()


def test_video_composition_is_declarative() -> None:
    command = ffmpeg_composite_command(
        game_video="game.mp4", brain_video="brain.mp4", output_video="final.mp4"
    )

    assert command[0] == "ffmpeg"
    assert command[-1] == "final.mp4"
    capture = ffmpeg_x11_capture_command("game.mp4", offset_x=100, offset_y=50)
    assert ":0.0+100,50" in capture
