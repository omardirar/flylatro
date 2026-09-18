"""Replay recording, bundles, and verification."""

from typing import Any

__all__ = [
    "JsonlTransitionRecorder",
    "TransitionRecord",
    "ReplayBundle",
    "ReplayBundleWriter",
    "ReplayIdentity",
    "ReplayDivergedError",
    "verify_replay",
]


def __getattr__(name: str) -> Any:
    """Avoid loading legacy policy evaluation when only bundle I/O is needed."""

    if name in {"JsonlTransitionRecorder", "TransitionRecord"}:
        from flylatro.replay.recorder import JsonlTransitionRecorder, TransitionRecord

        return {
            "JsonlTransitionRecorder": JsonlTransitionRecorder,
            "TransitionRecord": TransitionRecord,
        }[name]
    if name in {"ReplayBundle", "ReplayBundleWriter", "ReplayIdentity"}:
        from flylatro.replay.bundle import ReplayBundle, ReplayBundleWriter, ReplayIdentity

        return {
            "ReplayBundle": ReplayBundle,
            "ReplayBundleWriter": ReplayBundleWriter,
            "ReplayIdentity": ReplayIdentity,
        }[name]
    if name in {"ReplayDivergedError", "verify_replay"}:
        from flylatro.replay.verify import ReplayDivergedError, verify_replay

        return {
            "ReplayDivergedError": ReplayDivergedError,
            "verify_replay": verify_replay,
        }[name]
    raise AttributeError(name)
