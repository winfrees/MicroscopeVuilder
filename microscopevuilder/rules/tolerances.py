"""Tolerances derived from optics, not chosen by feel.

Every number a rule grades against should come from somewhere a student can check.
The anchor is the Rayleigh quarter-wave criterion, already validated against the
diffraction engine in ``tests/test_psf.py``: a peak-to-valley wavefront error of
``lambda/4`` leaves a Strehl ratio of about 0.8, which is the conventional edge of
"diffraction limited".

Applying it to defocus gives every focus tolerance in the game. Peak-to-valley
defocus for a longitudinal shift ``dz`` is ``W020 = dz * NA^2 / 2``, so

    dz <= lambda / (2 NA^2)

is the distance you may be out of focus before the image stops being
diffraction limited. That single relation, referred to whichever side of the
system is being graded, replaces what were arbitrary millimetre figures.
"""

from __future__ import annotations

from enum import Enum

RAYLEIGH_QUARTER_WAVE = 0.25  # waves peak-to-valley


class TolerancePolicy(str, Enum):
    """How strictly a component's *position* is graded.

    The physics does not change between these -- the depth of focus is what it is.
    What changes is how much slack the game allows before it calls a placement
    wrong, and the scorecard always reports both numbers so a student is never
    left thinking a 5% placement is what an optical bench would accept.
    """

    STRICT = "strict"       # the derived physical tolerance, nothing added
    FORGIVING = "forgiving"  # the larger of the physical tolerance and 5% of position


FORGIVING_FRACTION = 0.05

# Forgiving by default. The derived tolerances are correct and were unusable by
# hand: on round 11 the depth of focus is 0.19 mm while one pixel of drag at
# fit-to-window is 0.50 mm, so a single pixel of mouse movement overshot the
# tolerance by two and a half times. Zoom, the ruler and numeric entry fix the
# input resolution; this setting exists so the grading is not the thing standing
# between a player and a round they have understood.
_policy = TolerancePolicy.FORGIVING


def tolerance_policy() -> TolerancePolicy:
    return _policy


def set_tolerance_policy(policy: TolerancePolicy) -> TolerancePolicy:
    """Set the active policy, returning the previous one so callers can restore it."""
    global _policy
    previous = _policy
    _policy = TolerancePolicy(policy)
    return previous


def position_tolerance_mm(
    target_mm: float, derived_mm: float, policy: TolerancePolicy | None = None
) -> float:
    """How far a component may sit from where it belongs.

    Under FORGIVING this is the *larger* of the physical tolerance and 5% of the
    position -- never smaller. Relaxing a tolerance must not accidentally tighten
    one, which it would for a component sitting close to the origin where 5% is a
    fraction of a millimetre.
    """
    policy = policy or tolerance_policy()
    if policy is TolerancePolicy.STRICT:
        return derived_mm
    return max(derived_mm, abs(target_mm) * FORGIVING_FRACTION)


def describe_policy(
    target_mm: float, derived_mm: float, policy: TolerancePolicy | None = None
) -> str:
    """A phrase for the scorecard saying which tolerance is being applied, and why.

    Always names the physical figure, even when grading is looser, so the number a
    student takes away is the one an optical bench would hold them to.
    """
    policy = policy or tolerance_policy()
    applied = position_tolerance_mm(target_mm, derived_mm, policy)
    if policy is TolerancePolicy.STRICT or applied <= derived_mm:
        return f"graded at the physical tolerance of {derived_mm:.3f} mm"
    return (
        f"graded at a {FORGIVING_FRACTION:.0%} practice tolerance of {applied:.3f} mm; "
        f"the physical tolerance is {derived_mm:.3f} mm"
    )


def depth_of_focus_mm(wavelength_um: float, na: float) -> float:
    """Rayleigh depth of focus, in mm, for a bundle of numerical aperture ``na``.

    This is the *one-sided* tolerance: being this far out of focus puts the build
    exactly at the quarter-wave limit. It is a physical constant of the bundle, not
    a difficulty setting.
    """
    if na <= 0:
        raise ValueError("NA must be positive")
    return (wavelength_um / (2.0 * na**2)) / 1000.0


def image_side_depth_of_focus_mm(wavelength_um: float, na: float, magnification: float) -> float:
    """Depth of focus at the image, where a screen or sensor actually sits.

    The image-space aperture is ``NA / M``, so depth of focus grows as ``M^2``.
    This is why a 5x loupe tolerates a screen a fifth of a millimetre out while a
    100x objective does not tolerate a tenth of that at the specimen -- and why
    focusing a high-power objective feels so much touchier.
    """
    if magnification == 0:
        raise ValueError("magnification must be non-zero")
    return depth_of_focus_mm(wavelength_um, na / abs(magnification))


def field_conjugate_tolerance_mm(wavelength_um: float, na: float, minimum_mm: float = 0.5) -> float:
    """How far a FIELD conjugate may miss its target plane.

    Field conjugates -- field diaphragm onto specimen, specimen onto the
    intermediate image -- are graded by depth of focus, because what matters is
    sharpness: a field diaphragm imaged out of focus has a soft edge instead of a
    crisp one.

    The floor exists because illumination NA is low, so the depth of focus can run
    to many millimetres. Grading purely on it would accept almost anything, which
    teaches nothing; the floor keeps the check meaningful while the docstring stays
    honest about why the underlying physics is loose here.
    """
    return max(depth_of_focus_mm(wavelength_um, na), minimum_mm)


def pupil_conjugate_tolerance_mm(stop_radius_mm: float, converging_na: float) -> float:
    """How far a PUPIL conjugate may miss its target plane.

    Depth of focus is the wrong criterion here. Defocusing a pupil blurs nothing --
    there is no image at a pupil to blur. What goes wrong is that the diaphragm
    stops acting on the aperture and starts cutting the field: close it and the
    edges of the field darken instead of the illumination cone narrowing.

    The criterion that captures this is geometric. A cone converging at
    ``converging_na`` has radius ``delta * NA`` at a distance ``delta`` from focus,
    so requiring the lamp image to still fit inside the stop gives

        delta <= r_stop / NA

    A larger diaphragm is correspondingly more forgiving, which matches the bench:
    aligning a wide-open condenser is easy and aligning a stopped-down one is not.
    """
    if converging_na <= 0:
        raise ValueError("converging NA must be positive")
    return stop_radius_mm / converging_na


def tube_length_tolerance(magnification_tolerance: float) -> float:
    """Relative tube-length tolerance implied by a magnification specification.

    For a finite-tube objective ``M = L / f``, so ``dM/M = dL/L`` exactly: a 1%
    magnification specification *is* a 1% tube-length specification. Stating it
    this way rather than picking a number keeps the two rules from contradicting
    each other.

    Note the thin-lens model cannot capture the other consequence -- a real
    objective used away from its design tube length also picks up spherical
    aberration, which is why the DIN convention exists at all. That effect lives
    in the assigned aberration budgets (docs/PLAN.md decision 2), not here.
    """
    return magnification_tolerance


# Conventional, not derived: the 70-100% condenser-to-objective NA ratio is a
# working practice (matched gives full resolution; slightly under trades a little
# resolution for contrast), and it is reported as a convention in the rule text
# rather than dressed up as a derivation.
CONDENSER_NA_RATIO_LOW = 0.70
CONDENSER_NA_RATIO_HIGH = 1.00

# A magnification specification tight enough to be worth checking and loose enough
# to be reachable by eye on a bench.
MAGNIFICATION_TOLERANCE = 0.02
