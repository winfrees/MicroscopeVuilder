"""A ruler along the optical axis, and the snapping that makes placement possible.

This exists because of a measurement, not a preference. On round 11 the depth of
focus is 0.19 mm while the bench spans 380 mm: fitted to a 760-pixel view, one
pixel of drag moves a component 0.50 mm, so a single pixel overshoots the tolerance
by two and a half times. No amount of care with a mouse can place a component
correctly at that scale. Zoom fixes the resolution; snapping makes the common case
exact rather than merely close.

Kept free of Qt so the tick arithmetic and the snapping rules are testable headless.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4

# Snap steps offered in the workspace, in millimetres. 0.1 mm matters: most round
# targets land on a tenth (17.6, 193.6, 172.6), and a 1 mm step would step straight
# over them.
METRIC_STEPS_MM = (10.0, 5.0, 1.0, 0.5, 0.1)
IMPERIAL_STEPS_MM = (MM_PER_INCH, MM_PER_INCH / 2, MM_PER_INCH / 10, MM_PER_INCH / 20)


@dataclass(frozen=True)
class Tick:
    position_mm: float
    is_major: bool
    label: str


def snap(value_mm: float, step_mm: float) -> float:
    """Round a position to the nearest multiple of the step."""
    if step_mm <= 0:
        return value_mm
    return round(value_mm / step_mm) * step_mm


def snap_to_planes(
    value_mm: float, planes: list[float], capture_mm: float
) -> tuple[float, bool]:
    """Snap to a significant optical plane if one is within the capture distance.

    This is the snap that does the real work. Grid snapping gets a component onto a
    round number; plane snapping gets it onto *the plane where the image actually
    forms*, which is what the round is asking for. The player still has to know that
    the sensor belongs at the image plane -- the snap only removes the pixel-hunting.

    Returns the position and whether a plane captured it, so the caller can say so.
    """
    if not planes or capture_mm <= 0:
        return value_mm, False
    nearest = min(planes, key=lambda p: abs(p - value_mm))
    if abs(nearest - value_mm) <= capture_mm:
        return nearest, True
    return value_mm, False


def choose_step(span_mm: float, target_ticks: int = 20) -> float:
    """A tick spacing that gives roughly ``target_ticks`` across the span.

    Uses a 1-2-5 progression, the usual choice for rulers and axes, so the labels
    stay readable as the view zooms.
    """
    if span_mm <= 0:
        return 1.0
    raw = span_mm / max(target_ticks, 1)
    magnitude = 10.0 ** _floor_log10(raw)
    for multiple in (1.0, 2.0, 5.0, 10.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def _floor_log10(value: float) -> int:
    import math

    return int(math.floor(math.log10(value))) if value > 0 else 0


def ticks(
    start_mm: float,
    end_mm: float,
    step_mm: float,
    unit: str = "mm",
    major_every: int = 5,
) -> list[Tick]:
    """Ticks across a span, with every ``major_every``-th one labelled."""
    if step_mm <= 0 or end_mm <= start_mm:
        return []
    first = int(start_mm // step_mm)
    last = int(end_mm // step_mm) + 1
    if last - first > 5000:  # refuse to generate an unusable number of ticks
        return []

    out: list[Tick] = []
    for index in range(first, last + 1):
        position = index * step_mm
        if position < start_mm or position > end_mm:
            continue
        major = index % major_every == 0
        out.append(Tick(position, major, _label(position, unit) if major else ""))
    return out


def _label(position_mm: float, unit: str) -> str:
    if unit == "in":
        inches = position_mm / MM_PER_INCH
        return f'{inches:.3g}"'
    if abs(position_mm) < 1e-9:
        return "0"
    return f"{position_mm:g}"


def to_display(position_mm: float, unit: str) -> str:
    """Format a position for readout in the chosen unit."""
    if unit == "in":
        return f'{position_mm / MM_PER_INCH:.4f}"'
    return f"{position_mm:.3f} mm"
