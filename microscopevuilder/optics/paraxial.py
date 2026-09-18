"""Paraxial (ABCD) optics on a folded 3D bench.

The engine is deliberately thin-lens only (see docs/PLAN.md decision 2): aberrations
are carried as wavefront budgets in ``optics.wavefront``, not derived from surfaces.

Sign and unit conventions, held to everywhere:

* All lengths are millimetres. Light travels in the direction of increasing path
  length ``s``, measured along the folded axis.
* A paraxial ray is ``(y, u)``: height above the axis and angle in radians
  (small-angle, so ``u == tan(u) == sin(u)``).
* A positive lens has ``f > 0``. Distances to the right of a plane are positive.
* Transfer over distance ``d`` is ``[[1, d], [0, 1]]``; a thin lens is
  ``[[1, 0], [-1/f, 1]]``. System matrices compose right-to-left in the order light
  meets them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- ray ---------------------------------------------------------------------


@dataclass(frozen=True)
class Ray:
    """A paraxial ray: height ``y`` (mm) and angle ``u`` (radians)."""

    y: float
    u: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.y, self.u], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Ray":
        return cls(float(v[0]), float(v[1]))


# --- elements ----------------------------------------------------------------


@dataclass
class Element:
    """One component on the bench.

    ``s`` is the position along the folded axis. ``semi_diameter`` is the clear
    aperture radius, which is what decides stops, pupils and vignetting.
    ``matrix`` is the element's own ABCD (identity for a plane like a diaphragm,
    whose optical job is purely to clip).
    """

    name: str
    s: float
    semi_diameter: float
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    kind: str = "generic"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(2, 2)
        if self.semi_diameter <= 0:
            raise ValueError(f"{self.name}: semi_diameter must be positive")

    @property
    def focal_length(self) -> float | None:
        """Focal length implied by the element's ABCD, or None for a plane."""
        c = self.matrix[1, 0]
        return None if c == 0 else -1.0 / c


def thin_lens(name: str, s: float, f: float, semi_diameter: float, kind: str = "lens") -> Element:
    """An ideal thin lens of focal length ``f`` mm."""
    if f == 0:
        raise ValueError(f"{name}: focal length must be non-zero")
    return Element(name, s, semi_diameter, np.array([[1.0, 0.0], [-1.0 / f, 1.0]]), kind)


def aperture(name: str, s: float, semi_diameter: float, kind: str = "diaphragm") -> Element:
    """A plane that only clips: an iris, a field stop, a sensor edge."""
    return Element(name, s, semi_diameter, np.eye(2), kind)


def transfer(d: float) -> np.ndarray:
    """Free-space propagation over ``d`` mm."""
    return np.array([[1.0, d], [0.0, 1.0]])


def glass_plate(
    name: str, s: float, thickness_mm: float, index: float, semi_diameter: float,
    kind: str = "filter",
) -> Element:
    """A plane-parallel plate: a filter, a dichroic substrate, a coverslip.

    A plate of thickness ``t`` and index ``n`` replaces ``t`` of air with ``t`` of
    glass, and light inside it advances by the *reduced* thickness ``t/n``. Relative
    to the air it displaced, the plate therefore contributes ``transfer(t/n - t)``,
    a negative distance, and the image moves downstream by ``t(1 - 1/n)``.

    This is the element that makes infinity correction demonstrable rather than
    asserted. In converging light a plate shifts focus; in collimated light it does
    nothing at all, because a ray ``(y, 0)`` is unchanged by any transfer. That
    difference is the entire practical argument for an infinity space, and round 11
    turns on it.
    """
    if index <= 0:
        raise ValueError(f"{name}: refractive index must be positive")
    if thickness_mm < 0:
        raise ValueError(f"{name}: thickness cannot be negative")
    reduced = thickness_mm / index - thickness_mm
    return Element(name, s, semi_diameter, transfer(reduced), kind)


def focus_shift_from_plate(thickness_mm: float, index: float) -> float:
    """How far a plate displaces focus in converging light: ``t (1 - 1/n)``."""
    return thickness_mm * (1.0 - 1.0 / index)


# --- system ------------------------------------------------------------------


class ParaxialSystem:
    """An ordered set of elements along one folded axis."""

    def __init__(self, elements: list[Element]):
        self.elements = sorted(elements, key=lambda e: e.s)
        seen: set[str] = set()
        for e in self.elements:
            if e.name in seen:
                raise ValueError(f"duplicate element name: {e.name}")
            seen.add(e.name)

    def __getitem__(self, name: str) -> Element:
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(name)

    def between(self, s0: float, s1: float) -> np.ndarray:
        """System matrix from plane ``s0`` to plane ``s1``.

        Elements exactly at ``s0`` are included (light meets them on the way out);
        elements exactly at ``s1`` are not (the plane is just before them), which is
        what makes "the image plane at the sensor" behave as expected.
        """
        if s1 < s0:
            raise ValueError("s1 must be >= s0; the bench is traced in one direction")
        m = np.eye(2)
        cursor = s0
        for e in self.elements:
            if e.s < s0 or e.s >= s1:
                continue
            m = e.matrix @ transfer(e.s - cursor) @ m
            cursor = e.s
        return transfer(s1 - cursor) @ m

    def trace(self, ray: Ray, s0: float, s1: float) -> Ray:
        """Propagate one ray from ``s0`` to ``s1``."""
        return Ray.from_vector(self.between(s0, s1) @ ray.as_vector())

    def trace_profile(self, ray: Ray, s0: float) -> list[tuple[Element, Ray]]:
        """Ray height/angle as it arrives at each element downstream of ``s0``.

        This is what the ray overlay draws, and what stop-finding consumes: the
        height is taken *before* the element acts, so comparing it to the element's
        semi-diameter answers "does this clip?".
        """
        out: list[tuple[Element, Ray]] = []
        for e in self.elements:
            if e.s < s0:
                continue
            out.append((e, self.trace(ray, s0, e.s)))
        return out

    # --- imaging -------------------------------------------------------------

    def image_plane(self, s_object: float, search_to: float | None = None) -> float | None:
        """Where the system images the plane at ``s_object``.

        The image is where the B term of the object-to-image matrix vanishes: all
        rays leaving one object point arrive at one height regardless of angle.
        Returns ``None`` when the outgoing light is collimated (image at infinity),
        which is the normal, correct answer in an infinity-corrected build.

        ``search_to`` bounds the trace, not just the search: asking where the
        *intermediate* image lands must not trace through the eyepiece that follows
        it. Elements beyond ``search_to`` are excluded entirely.
        """
        last = self.elements[-1].s if self.elements else s_object
        end = search_to if search_to is not None else last + 1.0
        end = max(end, s_object + 1e-9)
        m = self.between(s_object, end)
        # Propagating a further distance t past `end` gives B' = B + t*D.
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        if abs(d) < 1e-12:
            return None  # afocal in this sense: no finite conjugate
        t = -b / d
        s_img = end + t
        return float(s_img)

    def magnification(self, s_object: float, s_image: float) -> float:
        """Transverse magnification between two planes (negative = inverted)."""
        return float(self.between(s_object, s_image)[0, 0])

    # --- stops and pupils ----------------------------------------------------

    def aperture_stop(self, s_object: float) -> Element | None:
        """The element that limits the axial cone from the on-axis object point.

        Found by sending a probe ray from the axial object point and asking which
        element runs out of clear aperture first, i.e. has the smallest ratio of
        semi-diameter to ray height. That ratio is exactly the factor the probe
        angle must be scaled by, so the winner is scale-free.
        """
        probe = Ray(0.0, 1e-3)
        best: tuple[float, Element] | None = None
        for e, r in self.trace_profile(probe, s_object):
            if abs(r.y) < 1e-15:
                continue
            ratio = e.semi_diameter / abs(r.y)
            if best is None or ratio < best[0]:
                best = (ratio, e)
        return best[1] if best else None

    def marginal_ray(self, s_object: float) -> Ray | None:
        """The ray from the axial object point that just fills the aperture stop."""
        stop = self.aperture_stop(s_object)
        if stop is None:
            return None
        probe = Ray(0.0, 1e-3)
        at_stop = self.trace(probe, s_object, stop.s)
        if abs(at_stop.y) < 1e-15:
            return None
        return Ray(0.0, probe.u * stop.semi_diameter / abs(at_stop.y))

    def chief_ray(self, s_object: float, field_height: float) -> Ray | None:
        """The ray from an off-axis object point through the centre of the stop.

        This is the definition -- not "a ray parallel to the axis at field height",
        which only coincides with the chief ray when the stop happens to sit at the
        front focal plane. Getting it wrong moves the pupil planes and understates
        vignetting, since the chief ray is what marks the edge of the field.

        Solving ``A h + B u = 0`` at the stop gives the launch angle.
        """
        stop = self.aperture_stop(s_object)
        if stop is None:
            return None
        a, b = self.between(s_object, stop.s)[0]
        if abs(b) < 1e-15:
            # The object plane is imaged onto the stop: every ray from this point
            # arrives at the same height, so no launch angle can centre it.
            return None if abs(a * field_height) > 1e-12 else Ray(field_height, 0.0)
        return Ray(field_height, -a * field_height / b)

    def object_space_na(self, s_object: float, n: float = 1.0) -> float:
        """Numerical aperture accepted at the object, ``n sin(u)``.

        Paraxially ``sin u == u``; the ``n`` argument carries the immersion medium,
        which is why an oil objective can exceed NA 1.0.
        """
        m = self.marginal_ray(s_object)
        return 0.0 if m is None else abs(n * m.u)
