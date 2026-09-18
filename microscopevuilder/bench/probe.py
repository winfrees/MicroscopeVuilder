"""The white card: a diagnostic surface the player can put anywhere on the axis.

At a real bench you hold a card in the beam and look at what lands on it. That one
habit answers most questions a beginner has -- is there an image here? is this a
pupil? how big is the illuminated patch? -- and it is the cheapest way to make
conjugate planes tangible before Koehler alignment asks the player to reason about
four of them at once.

The classic diagnostic, which this module implements directly:

* where the **marginal** ray crosses the axis is an **image** (field) plane;
* where the **chief** ray crosses the axis is a **pupil** (aperture) plane;
* anywhere else is neither, and the card shows a blur.

The card is a *probe*, not an optical element: it is excluded from the trace, from
stop-finding and from the rules. A real card of course intercepts the beam and
everything downstream goes dark, which is exactly why you take it out again. The
model keeps the card out of the train so that inserting one never changes the
answer it is being used to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bench import Bench

CARD_KIND = "white_card"


@dataclass(frozen=True)
class CardReading:
    """What a card held at ``s`` would show."""

    s: float
    blur_diameter_mm: float
    patch_radius_mm: float
    marginal_height_mm: float
    chief_height_mm: float
    plane_type: str  # "image", "pupil", "both", or "neither"
    magnification: float | None
    nearest_image_s: float | None
    defocus_mm: float | None

    @property
    def is_image_plane(self) -> bool:
        return self.plane_type in ("image", "both")

    @property
    def is_pupil_plane(self) -> bool:
        return self.plane_type in ("pupil", "both")

    def describe(self) -> str:
        if self.plane_type == "image":
            what = (
                f"a sharp image, {2 * self.patch_radius_mm:.2f} mm across"
                + (f", at {abs(self.magnification):.2f}x" if self.magnification else "")
            )
        elif self.plane_type == "pupil":
            what = (
                f"an evenly filled disc {self.blur_diameter_mm:.2f} mm across with no "
                "image structure -- this is a pupil plane"
            )
        elif self.plane_type == "both":
            what = "both an image and a pupil, which means the system is degenerate here"
        else:
            what = f"a blurred patch: every object point spreads {self.blur_diameter_mm:.3f} mm"
            if self.nearest_image_s is not None:
                what += f"; the nearest image plane is {self.defocus_mm:+.2f} mm away"
        return f"card at s = {self.s:.2f} mm shows {what}"


def _trace_without_cards(bench: Bench):
    """The optical train. ``Bench.to_paraxial`` already excludes probe cards."""
    return bench.to_paraxial()


def read_card(
    bench: Bench,
    s_object: float,
    s_card: float,
    field_height_mm: float = 0.5,
    image_tolerance_mm: float = 0.01,
) -> CardReading:
    """What a white card placed at ``s_card`` would show.

    ``image_tolerance_mm`` is the blur below which the eye calls it sharp; 10 um is
    a fair stand-in for what a person can see on a piece of card.
    """
    system = _trace_without_cards(bench)
    marginal = system.marginal_ray(s_object)
    chief = system.chief_ray(s_object, field_height_mm)

    marginal_height = abs(system.trace(marginal, s_object, s_card).y) if marginal else 0.0
    chief_height = abs(system.trace(chief, s_object, s_card).y) if chief else 0.0

    at_image = marginal_height <= image_tolerance_mm
    at_pupil = chief_height <= image_tolerance_mm
    plane_type = (
        "both" if at_image and at_pupil
        else "image" if at_image
        else "pupil" if at_pupil
        else "neither"
    )

    nearest = system.image_plane(s_object, search_to=s_card)
    magnification = (
        system.magnification(s_object, s_card) if at_image and field_height_mm else None
    )

    return CardReading(
        s=s_card,
        # A point object spreads over the full marginal cone at a defocused plane.
        blur_diameter_mm=2 * marginal_height,
        patch_radius_mm=chief_height,
        marginal_height_mm=marginal_height,
        chief_height_mm=chief_height,
        plane_type=plane_type,
        magnification=magnification,
        nearest_image_s=nearest,
        defocus_mm=None if nearest is None else nearest - s_card,
    )


def scan_axis(
    bench: Bench,
    s_object: float,
    s_start: float,
    s_end: float,
    samples: int = 400,
    field_height_mm: float = 0.5,
) -> list[CardReading]:
    """Sweep a card along the axis. Feeds the 'where are the planes?' overlay."""
    if samples < 2:
        raise ValueError("need at least two samples")
    step = (s_end - s_start) / (samples - 1)
    return [
        read_card(bench, s_object, s_start + i * step, field_height_mm)
        for i in range(samples)
    ]


def find_planes(
    bench: Bench,
    s_object: float,
    s_start: float,
    s_end: float,
    samples: int = 400,
    field_height_mm: float = 0.5,
) -> dict[str, list[float]]:
    """Locate image and pupil planes by where the two rays change sign.

    Sign changes rather than minima: a ray crossing the axis is unambiguous, while
    a minimum can be a near-miss that never actually forms a plane.
    """
    system = _trace_without_cards(bench)
    marginal = system.marginal_ray(s_object)
    chief = system.chief_ray(s_object, field_height_mm)

    step = (s_end - s_start) / (samples - 1)
    positions = [s_start + i * step for i in range(samples)]

    found: dict[str, list[float]] = {"image": [], "pupil": []}
    rays = {"image": marginal, "pupil": chief}
    for kind, ray in rays.items():
        if ray is None:
            continue
        heights = [system.trace(ray, s_object, s).y for s in positions]
        for i in range(len(positions) - 1):
            a, b = heights[i], heights[i + 1]
            if a == 0.0:
                found[kind].append(positions[i])
            elif a * b < 0:
                # Linear interpolation to the crossing.
                t = a / (a - b)
                found[kind].append(positions[i] + t * step)
    return found
