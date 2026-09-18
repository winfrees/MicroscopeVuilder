"""The optical invariants themselves.

Each check takes a bench (plus whatever context it needs) and returns one
:class:`RuleResult`. They are deliberately independent so a round can compose the
subset it cares about, and so a failure names one idea rather than a build.
"""

from __future__ import annotations

from ..bench.bench import Bench
from ..optics.paraxial import Ray
from ..optics.psf import abbe_resolution_um, nyquist_pixel_size_um, rayleigh_resolution_um
from .base import RuleResult, Status, tolerance_result


def check_image_lands_on_detector(
    bench: Bench, s_object: float, detector_name: str, tolerance_mm: float = 0.05
) -> RuleResult:
    """Is the image actually in focus where the player put the sensor or eyepoint?"""
    system = bench.to_paraxial()
    detector = bench.get(detector_name)
    s_img = system.image_plane(s_object, search_to=detector.s + 1.0)

    if s_img is None:
        return RuleResult(
            name="Focus",
            status=Status.FAIL,
            summary="the light leaving this system is collimated, so no image forms",
            equation="B = 0 has no solution: the outgoing bundle is parallel",
            culprit=detector_name,
            remedy=(
                "an infinity space needs a tube lens to bring it to an image; "
                "add one, or move the specimen off the objective's front focal plane"
            ),
        )

    return tolerance_result(
        name="Focus",
        measured=s_img,
        target=detector.s,
        tolerance=tolerance_mm,
        units="mm",
        equation=f"image forms at s = {s_img:.3f} mm; {detector_name} sits at {detector.s:.3f} mm",
        culprit=detector_name,
        remedy=(
            f"move {detector_name} to s = {s_img:.3f} mm, or refocus by moving the "
            f"specimen by roughly {(detector.s - s_img) / 100:.4f} mm"
        ),
        relative=False,
    )


def check_magnification(
    bench: Bench,
    s_object: float,
    detector_name: str,
    target: float,
    tolerance: float = 0.05,
) -> RuleResult:
    """Transverse magnification at the detector, signed convention ignored."""
    system = bench.to_paraxial()
    detector = bench.get(detector_name)
    m = abs(system.magnification(s_object, detector.s))
    return tolerance_result(
        name="Magnification",
        measured=m,
        target=target,
        tolerance=tolerance,
        units="x",
        equation=f"M = A of the object-to-image matrix = {m:.4g}",
        culprit=detector_name,
        remedy="adjust the lens focal length or the conjugate distances",
    )


def check_no_unintended_clipping(
    bench: Bench, s_object: float, field_height_mm: float = 0.0
) -> RuleResult:
    """Does any element other than the intended stop cut into the chief-ray bundle?

    Vignetting that the player did not ask for is the classic invisible failure:
    the image is dim at the edges and nobody can say why.
    """
    system = bench.to_paraxial()
    stop = system.aperture_stop(s_object)
    marginal = system.marginal_ray(s_object)
    if stop is None or marginal is None:
        return RuleResult(
            name="Clear aperture",
            status=Status.NOT_APPLICABLE,
            summary="no aperture-limited bundle to check",
        )

    chief = Ray(field_height_mm, 0.0)
    worst_name, worst_fill = None, 0.0
    for element, ray in system.trace_profile(marginal, s_object):
        if element.name == stop.name:
            continue
        _, chief_ray = next(
            (e, r) for e, r in system.trace_profile(chief, s_object) if e.name == element.name
        )
        reach = abs(ray.y) + abs(chief_ray.y)
        fill = reach / element.semi_diameter
        if fill > worst_fill:
            worst_name, worst_fill = element.name, fill

    if worst_name is None:
        return RuleResult(
            name="Clear aperture",
            status=Status.PASS,
            summary="only the aperture stop limits the bundle",
        )

    status = Status.PASS if worst_fill <= 1.0 else Status.FAIL
    return RuleResult(
        name="Clear aperture",
        status=status,
        summary=(
            f"the bundle fills {worst_fill * 100:.0f}% of {worst_name}"
            + ("" if status is Status.PASS else ", so it is vignetting the field")
        ),
        equation=f"|y_marginal| + |y_chief| = {worst_fill:.2f} x semi-diameter at {worst_name}",
        measured=worst_fill,
        target=1.0,
        culprit=None if status is Status.PASS else worst_name,
        remedy=f"use a larger clear aperture at {worst_name}, or reduce the field height",
    )


def check_resolution(
    na_objective: float,
    na_condenser: float,
    wavelength_um: float,
    required_um: float,
) -> RuleResult:
    """Abbe limit against what the round asks the player to resolve."""
    d = abbe_resolution_um(wavelength_um, na_objective, na_condenser)
    status = Status.PASS if d <= required_um else Status.FAIL
    return RuleResult(
        name="Resolution",
        status=status,
        summary=f"resolves {d:.3f} um; the target needs {required_um:.3f} um",
        equation=(
            f"d = lambda / (NA_obj + NA_cond) = {wavelength_um:.4f} / "
            f"({na_objective:.2f} + {na_condenser:.2f}) = {d:.3f} um"
        ),
        measured=d,
        target=required_um,
        units="um",
        remedy=(
            "raise the objective NA, or open the condenser: closing it down to "
            "NA_cond = 0 costs a factor of two in resolution"
        ),
    )


def check_condenser_na_match(
    na_objective: float, na_condenser: float, low: float = 0.7, high: float = 1.0
) -> RuleResult:
    """Is the condenser matched to the objective?

    Conventionally NA_cond is set to 70-100% of NA_obj: matched gives full
    resolution, slightly under trades a little of it for contrast, and far under
    throws away half the resolution the objective was paid for.
    """
    if na_objective <= 0:
        return RuleResult("Condenser match", Status.NOT_APPLICABLE, "no objective")
    ratio = na_condenser / na_objective
    if low <= ratio <= high:
        status = Status.PASS
    elif ratio > high or ratio >= low * 0.75:
        status = Status.WARN
    else:
        status = Status.FAIL
    note = {
        Status.PASS: "matched",
        Status.WARN: (
            "over-open: the extra illumination NA adds glare without resolution"
            if ratio > high
            else "a little closed: some resolution traded for contrast"
        ),
        Status.FAIL: "closed far down: you are throwing away resolution for contrast",
    }[status]
    return RuleResult(
        name="Condenser match",
        status=status,
        summary=f"NA_cond / NA_obj = {ratio:.2f} -- {note}",
        equation=f"{na_condenser:.2f} / {na_objective:.2f} = {ratio:.2f}, want {low:.2f}-{high:.2f}",
        measured=ratio,
        target=(low + high) / 2,
        culprit="condenser",
        remedy="open the condenser aperture diaphragm until it matches the objective",
    )


def check_sampling(
    wavelength_um: float, na: float, magnification: float, pixel_um: float
) -> RuleResult:
    """Nyquist at the sensor: did the camera keep what the optics resolved?"""
    limit = nyquist_pixel_size_um(wavelength_um, na, magnification)
    status = Status.PASS if pixel_um <= limit else Status.FAIL
    return RuleResult(
        name="Sampling",
        status=status,
        summary=(
            f"{pixel_um:.2f} um pixels against a {limit:.2f} um Nyquist limit"
            + ("" if status is Status.PASS else " -- undersampled, detail is being discarded")
        ),
        equation=(
            f"pixel <= 0.61 lambda M / (2 NA) = 0.61 x {wavelength_um:.4f} x "
            f"{magnification:.0f} / (2 x {na:.2f}) = {limit:.2f} um"
        ),
        measured=pixel_um,
        target=limit,
        units="um",
        culprit="camera",
        remedy=(
            "use smaller pixels or add magnification before the sensor; note that "
            "adding magnification costs field of view and light"
        ),
    )


def check_optical_tube_length(
    bench: Bench, objective_name: str, image_name: str, target_mm: float = 160.0
) -> RuleResult:
    """Finite-tube convention: the objective is corrected for one tube length.

    The quantity that sets the objective's magnification is the **optical** tube
    length: back focal plane to intermediate image, ``M = L / f``. It is not the
    objective-to-image distance (that is ``L + f``), and it is not the mechanical
    tube length either -- DIN's 160 mm is a shoulder-to-eyepiece-seat dimension, and
    the optical tube length inside it is shorter. Getting this wrong is worth a
    full magnification step at 10x, which is why the rule measures it explicitly.
    """
    objective = bench.get(objective_name)
    image = bench.get(image_name)
    f = objective.focal_length_mm or 0.0
    length = image.s - (objective.s + f)
    return tolerance_result(
        name="Optical tube length",
        measured=length,
        target=target_mm,
        tolerance=0.01,
        units="mm",
        equation=(
            f"L = {image_name} - (back focal plane of {objective_name}) = "
            f"{image.s:.2f} - ({objective.s:.2f} + {f:.2f}) = {length:.2f} mm; "
            f"M = L / f = {length / f if f else float('nan'):.2f}x"
        ),
        culprit=objective_name,
        remedy=(
            "a finite-tube objective is only corrected at its design tube length; "
            "move the intermediate image, or switch to an infinity system"
        ),
    )


def check_infinity_space(bench: Bench, s_object: float, objective_name: str) -> RuleResult:
    """Is the space behind the objective genuinely collimated?

    This is what lets filters and dichroics be inserted without shifting focus,
    and it is the defining property the player has to establish in round 11.
    """
    system = bench.to_paraxial()
    objective = bench.get(objective_name)
    out = system.trace(Ray(0.0, 1e-3), s_object, objective.s + 1e-6)
    residual = abs(out.u) / 1e-3

    status = Status.PASS if residual < 0.02 else Status.FAIL
    return RuleResult(
        name="Infinity space",
        status=status,
        summary=(
            "the space behind the objective is collimated"
            if status is Status.PASS
            else f"the bundle behind the objective still converges ({residual:.1%} of input angle)"
        ),
        equation=(
            f"u_out / u_in = {residual:.4f}; collimated means 0 "
            f"(specimen exactly at the front focal plane)"
        ),
        measured=residual,
        target=0.0,
        culprit=objective_name,
        remedy=(
            f"place the specimen at the objective's front focal plane, "
            f"s = {objective.s - (objective.focal_length_mm or 0):.3f} mm"
        ),
    )
