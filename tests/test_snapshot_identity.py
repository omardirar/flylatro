import struct

import pytest

from flylatro.env.snapshot_identity import canonical_snapshot
from flylatro.env.upstream_contract import SIMULATOR_VERSION


def snapshot(entries, suffix=b"all-other-state"):
    seed = b"EXAMPLE"
    prefix = struct.pack("<Q", len(seed)) + seed + struct.pack("<dQ", 0.25, len(entries))
    return prefix + b"".join(struct.pack("<Q", len(key)) + key + struct.pack("<d", value)
        for key, value in entries) + suffix


def test_keyed_rng_order_is_ignored_but_every_state_value_is_preserved():
    original = snapshot([(b"boss", 0.25), (b"shop", 0.75)])
    reordered = snapshot([(b"shop", 0.75), (b"boss", 0.25)])
    normalize = lambda value: canonical_snapshot(value, simulator_version=SIMULATOR_VERSION)
    assert original != reordered
    assert normalize(original) == normalize(reordered)
    assert normalize(original) != normalize(snapshot([(b"boss", 0.25), (b"shop", 0.50)]))
    assert normalize(original) != normalize(snapshot([(b"boss", 0.25), (b"shop", 0.75)], b"changed-other-state"))


def test_snapshot_identity_refuses_unknown_or_malformed_formats():
    with pytest.raises(ValueError, match="unknown"):
        canonical_snapshot(snapshot([]), simulator_version="other")
    for payload in (b"", snapshot([(b"boss", 0.25)])[:17], snapshot([(b"boss", 0.25), (b"boss", 0.50)])):
        with pytest.raises(ValueError):
            canonical_snapshot(payload, simulator_version=SIMULATOR_VERSION)
