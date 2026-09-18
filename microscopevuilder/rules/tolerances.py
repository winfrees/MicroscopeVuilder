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

RAYLEIGH_QUARTER_WAVE = 0.25  # waves peak-to-valley


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
