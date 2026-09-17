"""Disjoint deterministic seed streams for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal

import numpy as np


SeedStream = Literal[
    "training", "validation", "curriculum", "final_test", "showcase"
]


@dataclass(frozen=True, slots=True)
class SeedRange:
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size

    def values(self, count: int, *, offset: int = 0) -> tuple[int, ...]:
        if count < 0 or offset < 0 or offset + count > self.size:
            raise ValueError("requested seeds exceed the reserved range")
        return tuple(range(self.start + offset, self.start + offset + count))


@dataclass(frozen=True, slots=True)
class SeedPlan:
    training: SeedRange = SeedRange(0, 10_000_000)
    validation: SeedRange = SeedRange(20_000_000, 1_000_000)
    curriculum: SeedRange = SeedRange(30_000_000, 1_000_000)
    final_test: SeedRange = SeedRange(40_000_000, 1_000_000)
    showcase: SeedRange = SeedRange(50_000_000, 1_000_000)

    def __post_init__(self) -> None:
        ranges = [getattr(self, name) for name in self.__dataclass_fields__]
        ordered = sorted(ranges, key=lambda item: item.start)
        if any(left.stop > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("seed streams overlap")

    def seeds(
        self, stream: SeedStream, count: int, *, offset: int = 0
    ) -> tuple[int, ...]:
        return getattr(self, stream).values(count, offset=offset)

    @property
    def sha256(self) -> str:
        payload = {
            name: {
                "start": getattr(self, name).start,
                "size": getattr(self, name).size,
            }
            for name in self.__dataclass_fields__
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)


def derive_seed(*parts: object, modulus: int = 2**63 - 1) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % modulus

