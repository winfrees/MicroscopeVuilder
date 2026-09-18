"""Projection of the 3D bench onto the 2D side elevation.

Kept separate from any Qt import so the projection is testable headlessly and so a
future 3D renderer can ignore it entirely (docs/PLAN.md decision 3). The screen
plane is x-z: the bench runs left to right and folds go up and down, which is how
a side elevation of a real stand reads.
"""

from __future__ import annotations

import numpy as np

from ..bench.bench import Bench


def project(point3d: np.ndarray) -> np.ndarray:
    """3D bench coordinates to 2D scene coordinates (x right, z up)."""
    return np.array([point3d[0], point3d[2]], dtype=float)


def axis_frame(bench: Bench, s: float, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point, unit tangent and unit normal of the axis at path position ``s``.

    The normal is what ray heights and element glyphs are drawn along, so it has to
    follow the fold: after a 90-degree turn an element must be drawn across the new
    axis, not the old one.
    """
    here = project(bench.position_of(s))
    ahead = project(bench.position_of(s + eps))
    behind = project(bench.position_of(max(s - eps, 0.0)))

    tangent = ahead - here
    if np.linalg.norm(tangent) < eps * 1e-3:
        tangent = here - behind
    norm = np.linalg.norm(tangent)
    tangent = tangent / norm if norm else np.array([1.0, 0.0])
    normal = np.array([-tangent[1], tangent[0]])
    return here, tangent, normal


def point_at(bench: Bench, s: float, height_mm: float) -> np.ndarray:
    """Scene position of a ray at path ``s`` and transverse height ``height_mm``."""
    here, _, normal = axis_frame(bench, s)
    return here + normal * height_mm
