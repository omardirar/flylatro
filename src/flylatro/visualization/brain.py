"""Small activity-driven SVG renderer using FlyWire neuron coordinates."""

from __future__ import annotations

import html
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROLE_COLOURS = {
    "input": "#35c6ff",
    "internal": "#f2d34f",
    "readout": "#ff5da2",
    "kc": "#8be28b",
    "mbon": "#ff9f43",
    "dan": "#b980ff",
    "descending": "#ff5da2",
    "plasticity": "#ffffff",
}


@dataclass(frozen=True, slots=True)
class AnatomicalTransform:
    """Persisted projection bounds shared by every frame in a rendering."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int = 960
    height: int = 720
    margin: float = 50.0

    @classmethod
    def from_coordinates(
        cls, coordinates_nm: np.ndarray, *, width: int = 960, height: int = 720
    ) -> "AnatomicalTransform":
        points = np.asarray(coordinates_nm, dtype=np.float64)
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            raise ValueError("whole-brain coordinates contain no finite points")
        low = finite[:, :2].min(axis=0)
        high = finite[:, :2].max(axis=0)
        return cls(
            float(low[0]),
            float(high[0]),
            float(low[1]),
            float(high[1]),
            width,
            height,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def project(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        span_x = max(self.x_max - self.x_min, 1.0)
        span_y = max(self.y_max - self.y_min, 1.0)
        result = np.empty((len(values), 2), dtype=np.float64)
        result[:, 0] = self.margin + (values[:, 0] - self.x_min) / span_x * (
            self.width - 2 * self.margin
        )
        result[:, 1] = self.margin + (
            1.0 - (values[:, 1] - self.y_min) / span_y
        ) * (self.height - 2 * self.margin)
        return result

    def save(self, path: Path) -> Path:
        path.write_text(
            json.dumps({**asdict(self), "sha256": self.sha256}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path


def render_activity_svg(
    output_path: Path,
    *,
    neuron_ids: Sequence[int],
    coordinates_nm: np.ndarray,
    activity: Sequence[float],
    roles: Sequence[str],
    title: str,
    width: int = 960,
    height: int = 720,
    transform: AnatomicalTransform | None = None,
    background_coordinates_nm: np.ndarray | None = None,
) -> Path:
    """Render activity using a fixed whole-brain anatomical transform."""

    ids = np.asarray(neuron_ids, dtype=np.int64)
    points = np.asarray(coordinates_nm, dtype=np.float64)
    values = np.asarray(activity, dtype=np.float64)
    role_values = np.asarray(roles, dtype=object)
    if points.shape != (len(ids), 3):
        raise ValueError("coordinates must have shape [active neurons, 3]")
    if values.shape != (len(ids),) or role_values.shape != (len(ids),):
        raise ValueError("activity and roles must identify every active neuron")
    if any(str(role) not in ROLE_COLOURS for role in role_values):
        raise ValueError("unknown neural role")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(values)
    ids = ids[finite]
    points = points[finite]
    values = values[finite]
    role_values = role_values[finite]
    if len(ids):
        xy = (
            transform.project(points)
            if transform is not None
            else _normalise(points[:, :2], width, height)
        )
        peak = max(float(np.max(np.abs(values))), 1e-6)
    else:
        xy = np.empty((0, 2))
        peak = 1.0
    circles = []
    background = []
    if background_coordinates_nm is not None:
        background_points = np.asarray(background_coordinates_nm, dtype=np.float64)
        background_points = background_points[np.isfinite(background_points).all(axis=1)]
        background_xy = (
            transform.project(background_points)
            if transform is not None
            else _normalise(background_points[:, :2], width, height)
        )
        background = [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.75" '
            'fill="#8a91a8" fill-opacity="0.12"/>'
            for x, y in background_xy
        ]
    for index, (x, y) in enumerate(xy):
        radius = 2.0 + 7.0 * min(abs(float(values[index])) / peak, 1.0)
        role = str(role_values[index])
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{ROLE_COLOURS[role]}" fill-opacity="0.82">'
            f'<title>{int(ids[index])} {html.escape(role)}</title></circle>'
        )
    legend = " ".join(
        f'<text x="{20 + i * 170}" y="{height - 18}" fill="{colour}">{role}</text>'
        for i, (role, colour) in enumerate(ROLE_COLOURS.items())
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" '
        f'fill="#080a12"/><text x="20" y="34" fill="white" font-size="22">'
        f'{html.escape(title)}</text>{"".join(background)}{"".join(circles)}{legend}</svg>\n'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def coordinate_lookup(
    root_ids: np.ndarray, coordinates_nm: np.ndarray, selected_ids: Sequence[int]
) -> np.ndarray:
    order = np.argsort(root_ids)
    sorted_ids = root_ids[order]
    selected = np.asarray(selected_ids, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, selected)
    valid = positions < len(sorted_ids)
    valid &= sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == selected
    if not bool(valid.all()):
        missing = selected[~valid].tolist()
        raise KeyError(f"FlyWire coordinates missing for neuron IDs: {missing[:5]}")
    return coordinates_nm[order[positions]]


def _normalise(points: np.ndarray, width: int, height: int) -> np.ndarray:
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1.0)
    scaled = (points - low) / span
    margin = 50.0
    scaled[:, 0] = margin + scaled[:, 0] * (width - 2 * margin)
    scaled[:, 1] = margin + (1.0 - scaled[:, 1]) * (height - 2 * margin)
    return scaled
