"""Small activity-driven SVG renderer using FlyWire neuron coordinates."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROLE_COLOURS = {
    "input": "#35c6ff",
    "internal": "#f2d34f",
    "readout": "#ff5da2",
}


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
) -> Path:
    """Render active neurons only; connectivity and inactive brain are omitted."""

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
        xy = _normalise(points[:, :2], width, height)
        peak = max(float(np.max(np.abs(values))), 1e-6)
    else:
        xy = np.empty((0, 2))
        peak = 1.0
    circles = []
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
        f'{html.escape(title)}</text>{"".join(circles)}{legend}</svg>\n'
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
