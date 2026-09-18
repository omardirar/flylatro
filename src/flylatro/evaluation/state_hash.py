"""Policy-neutral deterministic hashes for simulator observations and terminals."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from flylatro.env.upstream_contract import ObsDict


def hash_observation_row(observations: ObsDict, index: int) -> str:
    digest = hashlib.sha256()
    for key in sorted(observations):
        row = np.ascontiguousarray(observations[key][index])
        digest.update(key.encode())
        digest.update(str(row.dtype).encode())
        digest.update(np.asarray(row.shape, dtype="<i8").tobytes())
        digest.update(row.tobytes())
    return digest.hexdigest()


def terminal_hash(info: dict[str, Any]) -> str:
    terminal = {
        "episode": info.get("episode", {}),
        "seed": info.get("seed"),
        "terminal": True,
    }
    return hashlib.sha256(
        json.dumps(terminal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
