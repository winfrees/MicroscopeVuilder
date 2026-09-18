"""Read a contrast technique off the bench and turn it into imaging instructions.

Rounds 13-15 do not change how imaging works; they change the *pupil* and the
*source*. Keeping that translation in one place means phase contrast and DIC are
expressed as components the player places, not as modes the renderer switches
between -- so a misaligned phase ring produces a bad image for the same reason it
would on a bench, rather than being caught by a special case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..bench.bench import Bench
from ..optics.coherence import Source, make_annular_source, make_source
from ..optics.contrast import shear_field
from ..optics.psf import PupilGrid, phase_ring, rayleigh_resolution_um


def apply_polarization(field, bench: Bench) -> tuple[np.ndarray, str, list[str]]:
    """Turn a birefringent specimen into an amplitude field using the bench's polars.

    Without a polarizer and analyzer the specimen is genuinely invisible: it alters
    polarization, and nothing on the bench is sensitive to polarization. That is
    not a modelling shortcut, it is why the round needs the components.
    """
    from ..imaging.specimens import BirefringentField
    from ..optics.jones import field_through

    if not isinstance(field, BirefringentField):
        return field, "", []

    polarizer = _first(bench, "polarizer")
    analyzer = None
    polars = [e for e in bench.elements if e.kind == "polarizer" and e.enabled]
    if len(polars) >= 2:
        polarizer, analyzer = polars[0], polars[-1]

    if polarizer is None or analyzer is None:
        return (
            np.ones(field.shape, dtype=complex),
            "unpolarized",
            [
                "this specimen is birefringent, not absorbing: without a polarizer "
                "and an analyzer there is nothing for the optics to see"
            ],
        )

    transmitted = field_through(
        field.retardance_waves,
        field.azimuth_rad,
        math.radians(float(polarizer.metadata.get("angle_deg", 0.0))),
        math.radians(float(analyzer.metadata.get("angle_deg", 90.0))),
    )
    return np.sqrt(np.clip(transmitted, 0.0, None)).astype(complex), "polarized", []


@dataclass
class TechniqueSetup:
    """What the placed components mean for the source, pupil and specimen field."""

    source: Source | None = None
    pupil_mask: np.ndarray | None = None
    field_transform: Callable[[np.ndarray, float], np.ndarray] | None = None
    name: str = "brightfield"
    notes: list[str] = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def resolve_technique(
    bench: Bench, grid: PupilGrid, coherence_parameter: float, na: float
) -> TechniqueSetup:
    """Translate whatever contrast components are on the bench into imaging terms."""
    notes: list[str] = []

    annulus = _first(bench, "annulus")
    ring = _first(bench, "phase_ring")
    wollaston = _first(bench, "wollaston")

    source: Source | None = None
    pupil_mask: np.ndarray | None = None
    field_transform = None
    name = "brightfield"

    if annulus is not None:
        inner = float(annulus.metadata.get("inner", 0.55))
        outer = float(annulus.metadata.get("outer", 0.75))
        source = make_annular_source(grid, inner, outer)
        name = "annular illumination"

    if ring is not None:
        inner = float(ring.metadata.get("inner", 0.55))
        outer = float(ring.metadata.get("outer", 0.75))
        pupil_mask = phase_ring(
            grid,
            inner,
            outer,
            float(ring.metadata.get("phase_shift_waves", 0.25)),
            float(ring.metadata.get("transmission", 0.15)),
        )
        name = "phase contrast"
        if annulus is None:
            notes.append(
                "a phase ring with no condenser annulus does nothing useful: "
                "undiffracted light fills the whole pupil, so the ring cannot "
                "single it out"
            )

    if wollaston is not None:
        fraction = float(wollaston.metadata.get("shear_fraction", 0.6))
        bias = float(wollaston.metadata.get("bias_waves", 0.15))
        axis = int(wollaston.metadata.get("axis", 1))
        shear_px = fraction * rayleigh_resolution_um(grid.wavelength_um, max(na, 1e-3)) / grid.image_sample_um

        def field_transform(field, _sample_um, shear_px=shear_px, bias=bias, axis=axis):
            return shear_field(field, max(shear_px, 1e-6), bias, axis)

        name = "DIC"

    return TechniqueSetup(source, pupil_mask, field_transform, name, notes)


def _first(bench: Bench, kind: str):
    for element in bench.elements:
        if element.kind == kind and element.enabled:
            return element
    return None
