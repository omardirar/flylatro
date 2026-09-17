"""Print reproducible ffmpeg capture and composition commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Sequence

from flylatro.visualization.video import (
    ffmpeg_composite_command,
    ffmpeg_x11_capture_command,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--display", default=":0.0")
    capture.add_argument("--offset-x", type=int, default=0)
    capture.add_argument("--offset-y", type=int, default=0)
    capture.add_argument("--width", type=int, default=1280)
    capture.add_argument("--height", type=int, default=720)
    capture.add_argument("--fps", type=int, default=60)
    compose = subparsers.add_parser("compose")
    compose.add_argument("--game", type=Path, required=True)
    compose.add_argument("--brain", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--width", type=int, default=1920)
    compose.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)
    if args.command == "capture":
        command = ffmpeg_x11_capture_command(
            args.output,
            display=args.display,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    else:
        command = ffmpeg_composite_command(
            args.game, args.brain, args.output,
            width=args.width, height=args.height,
        )
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
