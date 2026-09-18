"""Rules graded by measuring the rendered image, not by inspecting the geometry.

Rounds 6-8 ask for things that are only knowable from a picture: "chromatic error
under tolerance", "corner MTF within spec", "sampled, not empty-magnified". None of
them can be answered from the ray trace, so they live here and run on **Run**
rather than on every drag -- each one renders, which costs tenths of a second
against the paraxial pass's microseconds. That split is the same two-clock design
the workspace already uses (see docs/PLAN.md, performance).

Every threshold is still derived. The chromatic criterion is "the whole waveband
lands within the depth of focus", which is what apochromatic correction *means*
rather than a tolerance someone picked.
"""

from __future__ import annotations

import math

from ..bench.bench import Bench
from ..imaging.build_optics import resolve_build_optics
from ..imaging.metrics import modulation_at_period
from ..imaging.specimens import commensurate_period, sinusoidal_amplitude_grating
from ..imaging.synthesis import RenderSpec, render_build
from ..optics.psf import rayleigh_resolution_um
from ..rules.base import RuleResult, Status
from ..rules.tolerances import depth_of_focus_mm


def _focus_shift_um(wavefront, na: float) -> float:
    """Invert the defocus coefficient back to a longitudinal shift in microns."""
    coefficient = wavefront.coefficients.get("defocus", 0.0)
    if na <= 0:
        return 0.0
    return coefficient * 2.0 * math.sqrt(3.0) * 2.0 / (na**2)


def check_chromatic_focus(
    bench: Bench,
    s_object: float,
    detector_name: str,
    objective_name: str = "objective",
    wavelengths_um: tuple[float, ...] = (0.4861, 0.5461, 0.6563),  # F, e, C lines
    field_height: float = 0.0,
) -> RuleResult:
    """Does the whole waveband come to focus together?

    The criterion is not a chosen tolerance: an objective is *colour corrected* to
    the extent that its focus for every wavelength in the band lands inside the
    depth of focus, so that one focus setting serves all of them. Secondary
    spectrum is what breaks this, and it is why an achromat at the blue end looks
    soft however carefully you focus.

    Graded on the F, e and C lines -- the same three an objective is designed
    against.
    """
    optics = [
        resolve_build_optics(
            bench, s_object, detector_name, w, objective_name, field_height
        )
        for w in wavelengths_um
    ]
    na = optics[0].na
    shifts = [_focus_shift_um(o.wavefront, na) for o in optics]
    spread = max(shifts) - min(shifts)
    allowed = depth_of_focus_mm(wavelengths_um[1], na) * 1000.0

    status = (
        Status.PASS if spread <= allowed
        else Status.WARN if spread <= 2 * allowed
        else Status.FAIL
    )
    worst = wavelengths_um[shifts.index(max(shifts, key=abs))] * 1000.0
    return RuleResult(
        name="Chromatic focus",
        status=status,
        summary=(
            f"the F-to-C band focuses over {spread:.2f} um, against a depth of focus "
            f"of {allowed:.2f} um"
            + ("" if status is Status.PASS else f" -- {worst:.0f} nm is the worst offender")
        ),
        equation=(
            f"focus shift across F/e/C = {spread:.2f} um; depth of focus "
            f"= lambda / (2 NA^2) = {allowed:.2f} um at NA {na:.2f}"
        ),
        measured=spread,
        target=allowed,
        units="um",
        culprit=objective_name,
        remedy=(
            "an apochromat brings three wavelengths to a common focus; an achromat "
            "leaves a secondary spectrum of about f/2000, which at this focal length "
            "is more than the depth of focus can absorb"
        ),
    )


def check_field_flatness(
    bench: Bench,
    s_object: float,
    detector_name: str,
    test_period_um: float,
    objective_name: str = "objective",
    minimum_ratio: float = 0.70,
    n: int = 192,
) -> RuleResult:
    """Is the corner as sharp as the centre?

    Measured, not inferred: the same grating is imaged on axis and at full field,
    and the surviving modulation is compared. Field curvature puts the corner focus
    somewhere the centre focus is not, so a non-plan objective loses corner contrast
    that no amount of refocusing recovers -- refocus for the corners and the centre
    goes soft instead.
    """
    def specimen(count, sample_um):
        return sinusoidal_amplitude_grating(
            count, sample_um, commensurate_period(count, sample_um, test_period_um)
        )

    def modulation(field_height: float) -> tuple[float, float]:
        rendered = render_build(
            bench,
            s_object,
            RenderSpec(
                specimen=specimen,
                n=n,
                field_height=field_height,
                detector_name=detector_name,
                objective_name=objective_name,
            ),
        )
        period = commensurate_period(n, rendered.sample_um, test_period_um)
        return modulation_at_period(rendered.intensity, rendered.sample_um, period), period

    centre, period = modulation(0.0)
    corner, _ = modulation(1.0)

    # A flatness ratio measured near the diffraction cutoff is meaningless: both
    # numbers are already close to zero, so their ratio is dominated by whatever
    # survives, and it can invert. Measured directly -- at a 1.0 um period against a
    # 0.85 um cutoff, a flat plan objective read 0.57 while a badly curved one read
    # 0.91. Refuse to answer rather than answer wrongly.
    if centre < 0.05:
        return RuleResult(
            name="Field flatness",
            status=Status.NOT_APPLICABLE,
            summary=(
                f"a {period:.2f} um period leaves only {centre * 100:.1f}% modulation "
                "even at the centre: too close to the cutoff to compare corners with"
            ),
            equation=(
                f"MTF(centre) = {centre:.3f}; a flatness ratio needs a mid-band "
                "frequency, not one the optics have already nearly extinguished"
            ),
            remedy="measure flatness at a coarser period, well inside the cutoff",
        )

    ratio = corner / centre

    status = (
        Status.PASS if ratio >= minimum_ratio
        else Status.WARN if ratio >= minimum_ratio * 0.7
        else Status.FAIL
    )
    return RuleResult(
        name="Field flatness",
        status=status,
        summary=(
            f"corner modulation is {ratio * 100:.0f}% of centre at a {period:.2f} um period"
            + ("" if status is Status.PASS else " -- the corners are visibly soft")
        ),
        equation=(
            f"MTF(corner) / MTF(centre) = {corner:.3f} / {centre:.3f} = {ratio:.2f}, "
            f"want >= {minimum_ratio:.2f}"
        ),
        measured=ratio,
        target=minimum_ratio,
        culprit=objective_name,
        remedy=(
            "a plan objective flattens the field; a plain achromat leaves a Petzval "
            "sag of tens of microns, far more than the depth of focus, so refocusing "
            "trades the centre for the corners rather than fixing either"
        ),
    )


def check_sensor_matches_the_optics(
    bench: Bench,
    s_object: float,
    detector_name: str,
    pixel_um: float,
    objective_name: str = "objective",
    wavelength_um: float = 0.5461,
) -> RuleResult:
    """Is the detector keeping what the optics resolved -- and no more?

    Two failures, in opposite directions, and a good build sits between them.
    Undersampling throws away resolution that was paid for in NA and working
    distance. Over-magnifying spreads the same detail across more pixels, which
    costs field of view and light (irradiance goes as 1/M^2) and buys nothing --
    the classic empty magnification.
    """
    optics = resolve_build_optics(
        bench, s_object, detector_name, wavelength_um, objective_name
    )
    resolution_um = rayleigh_resolution_um(wavelength_um, optics.na)
    at_sensor_um = resolution_um * optics.magnification
    nyquist_pixel_um = at_sensor_um / 2.0
    samples_per_resolution = at_sensor_um / pixel_um if pixel_um > 0 else 0.0

    if samples_per_resolution < 2.0:
        status, note = Status.FAIL, "undersampled: the sensor is discarding resolved detail"
    elif samples_per_resolution > 6.0:
        status, note = (
            Status.WARN,
            "empty magnification: more pixels, no more detail, less field and less light",
        )
    else:
        status, note = Status.PASS, "matched"

    return RuleResult(
        name="Sensor match",
        status=status,
        summary=(
            f"{samples_per_resolution:.1f} pixels across a resolved distance -- {note}"
        ),
        equation=(
            f"0.61 lambda / NA = {resolution_um:.3f} um at the specimen, x{optics.magnification:.0f} "
            f"= {at_sensor_um:.2f} um at the sensor; Nyquist wants pixels <= "
            f"{nyquist_pixel_um:.2f} um, you have {pixel_um:.2f} um"
        ),
        measured=samples_per_resolution,
        target=2.0,
        units="px",
        culprit="camera",
        remedy=(
            "add magnification before the sensor, or use smaller pixels"
            if status is Status.FAIL
            else "drop the extra magnification: it costs field and light for no detail"
        ),
    )


def check_specimen_is_visible(
    bench: Bench,
    s_object: float,
    detector_name: str,
    specimen_factory,
    minimum_contrast: float,
    label: str = "Specimen visibility",
    objective_name: str = "objective",
    n: int = 256,
) -> RuleResult:
    """Can you actually see the thing?

    The only honest win condition for a contrast technique. A phase object is
    genuinely invisible in brightfield -- that is not a modelling artifact but the
    reason the technique exists -- so the round is won when the measured contrast
    crosses a threshold a person could work with.
    """
    from ..imaging.metrics import michelson_contrast
    from ..imaging.synthesis import RenderSpec, render_build

    rendered = render_build(
        bench,
        s_object,
        RenderSpec(
            specimen=specimen_factory, n=n, detector_name=detector_name,
            objective_name=objective_name, coherence_parameter=0.7,
        ),
    )
    measured = michelson_contrast(rendered.intensity)
    status = Status.PASS if measured >= minimum_contrast else Status.FAIL
    return RuleResult(
        name=label,
        status=status,
        summary=(
            f"{rendered.technique} gives {measured:.3f} contrast"
            + ("" if status is Status.PASS else " -- not enough to see the specimen")
        ),
        equation=(
            f"Michelson contrast = (max - min) / (max + min) = {measured:.3f}, "
            f"want >= {minimum_contrast:.2f}"
        ),
        measured=measured,
        target=minimum_contrast,
        remedy=(
            "a pure phase object changes no amplitude at all, so brightfield has "
            "nothing to show; the technique has to convert phase into intensity"
        ),
    )
