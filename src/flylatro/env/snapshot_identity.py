"""Conservative content identity for the pinned simulator's binary snapshot."""
from __future__ import annotations

import struct

from flylatro.env.upstream_contract import SIMULATOR_VERSION


def canonical_snapshot(snapshot: bytes, *, simulator_version: str) -> bytes:
    """Sort the keyed-RNG HashMap prefix, preserving every remaining byte.

    At pinned commit 38ae21431700, Snapshot begins with Run.rng: String seed,
    f64 hashed_seed and HashMap<String,f64> states, serialized by bincode 1.
    Rust HashMap iteration order changes when deserialized. Only that map's
    order is normalized; keys, float bit patterns and all other snapshot bytes
    remain exact. Any later unsorted map reordering conservatively fails the
    audit instead of being ignored. Unknown simulator formats are refused.
    Source: sim/py/src/env.rs, sim/core/src/run.rs and sim/core/src/rng.rs.
    """
    if simulator_version != SIMULATOR_VERSION:
        raise ValueError("unknown simulator snapshot format")
    offset = 0

    def consume(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(snapshot):
            raise ValueError("truncated simulator snapshot")
        value = snapshot[offset:offset + size]; offset += size
        return value

    def string() -> bytes:
        length = struct.unpack("<Q", consume(8))[0]
        value = consume(length)
        value.decode("utf-8")
        return value

    string()  # seed
    consume(8)  # hashed_seed, retained bit for bit
    count = struct.unpack("<Q", consume(8))[0]
    if count > (len(snapshot) - offset) // 16:
        raise ValueError("invalid simulator RNG map length")
    prefix_end = offset
    entries = {}
    for _ in range(count):
        start = offset
        key = string(); consume(8)
        if key in entries:
            raise ValueError("duplicate simulator RNG key")
        entries[key] = snapshot[start:offset]
    return snapshot[:prefix_end] + b"".join(entries[key] for key in sorted(entries)) + snapshot[offset:]
