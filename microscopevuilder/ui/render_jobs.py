"""Deciding what actually needs recomputing, and where to render it.

Two costs, very different sizes. The paraxial trace is microseconds; forming one
image is a few tenths of a second, and with white cards on the bench there is one
image per card plus the image plane. So the expensive work has to be skipped
whenever its inputs have not changed.

Three signatures, each covering exactly what its consumer depends on:

* **optics** — everything that changes the ray trace. White cards are excluded,
  because a probe must not perturb what it measures, so moving a card never
  invalidates the trace.
* **planes** — where images are wanted: the detector, plus every card.
* **render** — the specimen and illumination settings.

A pane is re-rendered only when its own key changes, so moving one card leaves the
other panes alone, and changing the specimen re-renders all of them.

Qt-free on purpose: this is the logic that decides whether work happens, and it
should be testable without a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bench.bench import Bench
from ..bench.probe import CARD_KIND

# Metadata that changes the optics or the image, as opposed to bookkeeping.
OPTICAL_METADATA = (
    "na", "magnification", "thickness_mm", "index", "pixel_um", "grade",
    "inner", "outer", "phase_shift_waves", "transmission",
    "shear_fraction", "bias_waves", "axis", "angle_deg", "cube", "fluorophore",
    "in_turret", "parfocal_mm",
)


def optics_signature(bench: Bench) -> tuple:
    """Everything that changes the ray trace or the wavefront.

    Cards are excluded: they are probes, so holding one up must not invalidate the
    trace any more than it perturbs it.
    """
    rows = []
    for element in sorted(bench.elements, key=lambda e: (e.arm, e.s, e.name)):
        if element.kind == CARD_KIND:
            continue
        rows.append((
            element.name, element.kind, element.arm, element.enabled,
            round(element.s, 6),
            None if element.focal_length_mm is None else round(element.focal_length_mm, 6),
            round(element.semi_diameter_mm, 6),
            element.catalog_key,
            tuple((k, element.metadata.get(k)) for k in OPTICAL_METADATA
                  if k in element.metadata),
        ))
    folds = tuple((round(f.s, 6), f.direction) for f in bench.folds)
    arms = tuple(
        (a.name, round(a.junction_s, 6), round(a.length_mm, 6), a.direction, a.continues)
        for a in sorted(bench.arms.values(), key=lambda a: a.name)
    )
    return (tuple(rows), folds, arms)


@dataclass(frozen=True)
class RenderPlane:
    """One pane: a plane to form an image at, and what to call it."""

    name: str          # the bench element to render at
    label: str         # what the pane is titled
    s: float
    is_card: bool = False
    ordinal: int | None = None   # cards are numbered 1..n, matching the bench glyphs


def render_planes(bench: Bench, detector_name: str | None) -> list[RenderPlane]:
    """The image plane, then every card in bench order, numbered 1..n.

    The numbering is the contract with the bench view: card *n* in the scene is
    pane *n* here, so a player can look from one to the other without counting.
    """
    planes: list[RenderPlane] = []
    if detector_name and bench.has(detector_name):
        detector = bench.get(detector_name)
        planes.append(
            RenderPlane(detector.name, detector.label or detector.name, detector.s)
        )
    for ordinal, card in enumerate(sorted(bench.cards(), key=lambda c: c.s), start=1):
        planes.append(
            RenderPlane(
                card.name, f"Card {ordinal}", card.s, is_card=True, ordinal=ordinal
            )
        )
    return planes


def card_ordinals(bench: Bench) -> dict[str, int]:
    """Card name to its 1-based number, ordered along the axis."""
    return {
        card.name: ordinal
        for ordinal, card in enumerate(sorted(bench.cards(), key=lambda c: c.s), start=1)
    }


@dataclass(frozen=True)
class RenderSettings:
    """What the player chose, other than the bench itself."""

    specimen: str
    wavelength_um: float
    coherence: float
    field_height: float
    samples: int = 256
    exposure_photons: float = 2000.0

    def key(self) -> tuple:
        return (
            self.specimen, round(self.wavelength_um, 6), round(self.coherence, 4),
            round(self.field_height, 4), self.samples, self.exposure_photons,
        )


def plane_key(optics: tuple, plane: RenderPlane, settings: RenderSettings) -> tuple:
    """The cache key for one pane."""
    return (optics, plane.name, round(plane.s, 6), settings.key())


@dataclass
class RenderPlan:
    """What this Run has to do, and what it can reuse."""

    planes: list[RenderPlane] = field(default_factory=list)
    stale: list[RenderPlane] = field(default_factory=list)
    retrace: bool = True

    @property
    def has_work(self) -> bool:
        return bool(self.stale)


def plan_render(
    bench: Bench,
    detector_name: str | None,
    settings: RenderSettings,
    cache: dict,
    previous_optics: tuple | None,
) -> tuple[RenderPlan, tuple]:
    """Work out which panes are stale and whether the trace needs redoing.

    Returns the plan and the current optics signature, so the caller can store it
    for next time.
    """
    optics = optics_signature(bench)
    planes = render_planes(bench, detector_name)
    stale = [p for p in planes if plane_key(optics, p, settings) not in cache]
    return (
        RenderPlan(planes=planes, stale=stale, retrace=optics != previous_optics),
        optics,
    )
