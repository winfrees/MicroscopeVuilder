"""Everything the workspace needs to draw, computed without touching Qt.

This is the live model: it runs the paraxial pass and the rule report, which the
plan budgets at under 5 ms so they can run on every drag. Image synthesis is
deliberately NOT here -- a 384^2 Abbe integration is ~0.3 s and belongs behind the
Run button on a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bench.bench import Bench
from ..rules.base import RuleReport


@dataclass
class TracedRay:
    """A ray sampled along the bench, as (s, height) pairs.

    ``arm`` names which path the samples are measured along, so the renderer knows
    whether to place them on the main axis or on a branch.
    """

    label: str
    samples: list[tuple[float, float]] = field(default_factory=list)
    arm: str = "main"


@dataclass
class ConjugatePlane:
    """One plane in a conjugate set, for the ribbon under the axis."""

    s: float
    label: str
    kind: str  # "field" or "aperture"


def _illumination_rays(bench: Bench, system, s_object: float) -> list[TracedRay]:
    """Rays from the lamp, if this bench has one.

    On a branched bench the lamp sits on an arm, so the trace runs along that arm's
    own system and coordinates.
    """
    lamps = [e for e in bench.elements if e.kind == "lamp"]
    if not lamps:
        return []
    lamp = lamps[0]
    arm = lamp.arm
    if arm != "main":
        system = bench.to_paraxial(arm)
        elements = [e for e in bench.elements if e.arm == arm]
        junction = bench.arms[arm].length_mm
        downstream = sorted({lamp.s, *(e.s for e in elements), junction})
        return _rays_from(system, lamp, downstream, arm)

    downstream = sorted({lamp.s, *(e.s for e in bench.elements if e.s >= lamp.s)})
    tail = (max(downstream) - min(downstream)) * 0.08 or 10.0
    downstream.append(max(downstream) + tail)
    return _rays_from(system, lamp, downstream, "main")


def _rays_from(system, lamp, stops: list[float], arm: str) -> list[TracedRay]:
    out: list[TracedRay] = []
    axial = system.marginal_ray(lamp.s)
    if axial is not None:
        out.append(
            TracedRay(
                "illumination_axial",
                [(s, system.trace(axial, lamp.s, s).y) for s in stops],
                arm,
            )
        )
    edge = system.chief_ray(lamp.s, lamp.semi_diameter_mm)
    if edge is not None:
        out.append(
            TracedRay(
                "illumination_edge",
                [(s, system.trace(edge, lamp.s, s).y) for s in stops],
                arm,
            )
        )
    return out


@dataclass
class TraceModel:
    bench: Bench
    s_object: float
    rays: list[TracedRay]
    conjugates: list[ConjugatePlane]
    image_s: float | None
    magnification: float | None
    aperture_stop: str | None
    report: RuleReport | None = None


def build_trace_model(
    bench: Bench,
    s_object: float,
    field_height_mm: float = 0.5,
    report: RuleReport | None = None,
) -> TraceModel:
    """Run the live pass: marginal and chief rays, image plane, conjugate sets."""
    system = bench.to_paraxial()
    stop = system.aperture_stop(s_object)
    marginal = system.marginal_ray(s_object)

    # Only sample forward of the object: the bench is traced in one direction, and
    # on an illumination stand there are elements (lamp, collector, diaphragms)
    # upstream of the specimen that the imaging trace must not look back at.
    stops = sorted({s for s in (s_object, *(e.s for e in bench.elements)) if s >= s_object})
    tail = (max(stops) - min(stops)) * 0.08 or 10.0
    stops.append(max(stops) + tail)

    rays: list[TracedRay] = []
    if marginal is not None:
        rays.append(
            TracedRay(
                "marginal",
                [(s, system.trace(marginal, s_object, s).y) for s in stops],
            )
        )
    chief = system.chief_ray(s_object, field_height_mm)
    if chief is not None:
        rays.append(
            TracedRay("chief", [(s, system.trace(chief, s_object, s).y) for s in stops])
        )

    # The illumination path is its own trace, from the lamp forward. Drawing both
    # is what makes an illumination round legible: the two bundles cross at every
    # conjugate plane, and where one focuses the other is spread wide.
    rays.extend(_illumination_rays(bench, system, s_object))

    image_s = system.image_plane(s_object)
    if image_s is not None and image_s < s_object:
        # A virtual image, or a build the trace cannot make sense of. Either way
        # there is nothing to draw downstream, and asking for a magnification
        # across a backwards interval would raise.
        image_s = None
    magnification = (
        system.magnification(s_object, image_s) if image_s is not None else None
    )

    conjugates = [ConjugatePlane(s_object, "specimen", "field")]
    if image_s is not None:
        conjugates.append(ConjugatePlane(image_s, "image", "field"))
    for element in bench.elements:
        if element.kind in ("field_stop", "detector"):
            conjugates.append(ConjugatePlane(element.s, element.name, "field"))
        elif element.kind in ("diaphragm", "aperture_stop"):
            conjugates.append(ConjugatePlane(element.s, element.name, "aperture"))
    if stop is not None:
        conjugates.append(ConjugatePlane(stop.s, f"{stop.name} (stop)", "aperture"))

    return TraceModel(
        bench=bench,
        s_object=s_object,
        rays=rays,
        conjugates=conjugates,
        image_s=image_s,
        magnification=magnification,
        aperture_stop=stop.name if stop else None,
        report=report,
    )
