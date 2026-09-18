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
from ..optics.paraxial import Ray
from ..rules.base import RuleReport


@dataclass
class TracedRay:
    """A ray sampled along the bench, as (s, height) pairs."""

    label: str
    samples: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class ConjugatePlane:
    """One plane in a conjugate set, for the ribbon under the axis."""

    s: float
    label: str
    kind: str  # "field" or "aperture"


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

    stops = sorted({s_object, *(e.s for e in bench.elements)})
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
    chief = Ray(field_height_mm, 0.0)
    rays.append(
        TracedRay("chief", [(s, system.trace(chief, s_object, s).y) for s in stops])
    )

    image_s = system.image_plane(s_object)
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
