"""Declarative capture/composition commands; execution is always explicit."""

from __future__ import annotations

from pathlib import Path


def ffmpeg_x11_capture_command(
    output_video: Path,
    *,
    display: str = ":0.0",
    offset_x: int = 0,
    offset_y: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 60,
) -> tuple[str, ...]:
    """Build (but never execute) a Linux/X11 Balatro-window capture command."""

    if min(width, height, fps) < 1 or min(offset_x, offset_y) < 0:
        raise ValueError("capture dimensions/fps must be positive and offsets nonnegative")
    return (
        "ffmpeg", "-y", "-f", "x11grab", "-framerate", str(fps),
        "-video_size", f"{width}x{height}", "-i",
        f"{display}+{offset_x},{offset_y}", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", str(output_video),
    )


def ffmpeg_composite_command(
    game_video: Path,
    brain_video: Path,
    output_video: Path,
    *,
    width: int = 1920,
    height: int = 1080,
) -> tuple[str, ...]:
    if width < 640 or height < 360:
        raise ValueError("composition canvas is too small")
    filter_graph = (
        f"[0:v]scale={width * 2 // 3}:{height}[game];"
        f"[1:v]scale={width // 3}:{height}[brain];"
        "[game][brain]hstack=inputs=2[v]"
    )
    return (
        "ffmpeg", "-y", "-i", str(game_video), "-i", str(brain_video),
        "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_video),
    )


def ffmpeg_svg_sequence_command(
    frame_pattern: Path, output_video: Path, *, fps: int = 30
) -> tuple[str, ...]:
    if fps < 1:
        raise ValueError("fps must be positive")
    return (
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_pattern),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_video),
    )


def ffmpeg_concat_command(
    concat_manifest: Path, output_video: Path
) -> tuple[str, ...]:
    return (
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
        str(concat_manifest), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_video),
    )
