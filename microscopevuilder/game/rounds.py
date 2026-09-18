"""Round definitions: the specification a build is graded against.

A round is data plus one function. It states the brief, the parts budget, the test
object, and how to evaluate a bench -- but it never states *the answer*. There is a
reference solution, used only to prove the round is solvable and to power the
"show me a working build" diff after a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..bench.bench import Bench, BenchElement
from ..rules.base import RuleReport
from ..rules.tolerances import (
    MAGNIFICATION_TOLERANCE,
    field_conjugate_tolerance_mm,
    image_side_depth_of_focus_mm,
    pupil_conjugate_tolerance_mm,
    tube_length_tolerance,
)
from ..rules.checks import (
    check_condenser_na_match,
    check_conjugate,
    check_conjugate_to_plane,
    check_illumination_throughput,
    check_illumination_uniformity,
    check_image_lands_on_detector,
    check_magnification,
    check_no_unintended_clipping,
    check_not_conjugate,
    check_optical_tube_length,
    check_relaxed_eye,
    check_resolution,
    condenser_na,
)

VISUAL_REFERENCE_DISTANCE_MM = 250.0  # the conventional near point for visual M


@dataclass(frozen=True)
class Round:
    number: int
    title: str
    brief: str
    teaches: str
    s_object: float
    wavelength_um: float
    evaluate: Callable[[Bench], RuleReport]
    reference_build: Callable[[], Bench]
    parts_budget: int = 99
    available_kinds: tuple[str, ...] = ()
    notes: str = ""

    def grade(self, bench: Bench) -> RuleReport:
        return self.evaluate(bench)


# --- round 1: the loupe ------------------------------------------------------


def _round1_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    s_obj = -60.0
    # The focus tolerance is the image-side depth of focus, not a chosen number.
    na = bench.to_paraxial().object_space_na(s_obj)
    focus_tol = image_side_depth_of_focus_mm(0.5461, max(na, 1e-3), 5.0)
    report.add(check_image_lands_on_detector(bench, s_obj, "screen", tolerance_mm=focus_tol))
    report.add(
        check_magnification(bench, s_obj, "screen", target=5.0, tolerance=MAGNIFICATION_TOLERANCE)
    )
    report.add(check_no_unintended_clipping(bench, s_obj, field_height_mm=1.0))
    return report


def _round1_reference() -> Bench:
    # M = 5 at a 60 mm object distance: s_i = 5 s_o, so 1/f = 1/s_o + 1/(5 s_o)
    # = 1.2 / s_o, giving f = s_o / 1.2 = 50 mm and the image 300 mm out.
    return Bench(
        [
            BenchElement("lens", 0.0, "lens", 12.5, 50.0, label="f = 50 mm singlet"),
            BenchElement("screen", 300.0, "detector", 15.0, label="screen"),
        ]
    )


ROUND_1 = Round(
    number=1,
    title="Single lens loupe",
    brief="Form a 5x real image of a ruling 60 mm in front of a single lens.",
    teaches="conjugates and the imaging equation 1/f = 1/s_o + 1/s_i",
    s_object=-60.0,
    wavelength_um=0.5461,
    evaluate=_round1_evaluate,
    reference_build=_round1_reference,
    parts_budget=2,
    available_kinds=("lens", "detector"),
    notes=(
        "The magnification and the focus are not independent: choosing f fixes where "
        "the image lands, and moving the screen cannot change M without refocusing."
    ),
)


# --- round 2: the compound microscope ---------------------------------------


def _round2_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    s_obj = objective.s - _round2_object_distance(objective.focal_length_mm or 1.0)

    na = float(objective.metadata.get("na", 0.25))
    focus_tol = image_side_depth_of_focus_mm(0.5461, na, 10.0)
    report.add(
        check_optical_tube_length(
            bench, "objective", "intermediate_image", OPTICAL_TUBE_LENGTH_MM,
            tolerance=tube_length_tolerance(MAGNIFICATION_TOLERANCE),
        )
    )
    report.add(check_image_lands_on_detector(bench, s_obj, "intermediate_image", focus_tol))
    report.add(
        check_magnification(
            bench, s_obj, "intermediate_image", target=10.0, tolerance=MAGNIFICATION_TOLERANCE
        )
    )
    report.add(check_no_unintended_clipping(bench, s_obj, field_height_mm=0.5))
    report.add(check_relaxed_eye(bench, "intermediate_image", "eyepiece"))
    report.add(_check_visual_magnification(bench, target=100.0))
    report.add(check_resolution(0.25, 0.0, 0.5461, required_um=2.5))
    return report


OPTICAL_TUBE_LENGTH_MM = 160.0


def _round2_image_distance(f: float) -> float:
    """Objective-to-intermediate-image distance for the design tube length.

    The optical tube length L is measured from the back focal plane, so the image
    sits at ``f + L`` from the lens itself.
    """
    return f + OPTICAL_TUBE_LENGTH_MM


def _round2_object_distance(f: float) -> float:
    """Object distance that lands the image at the design intermediate plane."""
    s_i = _round2_image_distance(f)
    return 1.0 / (1.0 / f - 1.0 / s_i)


def _check_visual_magnification(bench: Bench, target: float):
    """Total visual magnification: objective transverse times eyepiece angular.

    The eyepiece is a loupe on the intermediate image, so its angular magnification
    is the conventional 250 mm near point over its focal length.
    """
    from ..rules.base import tolerance_result

    objective = bench.get("objective")
    eyepiece = bench.get("eyepiece")
    f_obj = objective.focal_length_mm or 1.0
    f_eye = eyepiece.focal_length_mm or 1.0
    m_obj = OPTICAL_TUBE_LENGTH_MM / f_obj
    m_eye = VISUAL_REFERENCE_DISTANCE_MM / f_eye
    total = m_obj * m_eye

    return tolerance_result(
        name="Visual magnification",
        measured=total,
        target=target,
        tolerance=0.05,
        units="x",
        equation=(
            f"M = (L / f_obj) x (250 / f_eye) = ({OPTICAL_TUBE_LENGTH_MM:.0f} / {f_obj:.3g}) x "
            f"(250 / {f_eye:.3g}) = {m_obj:.1f} x {m_eye:.1f} = {total:.1f}x"
        ),
        culprit="objective",
        remedy="change the objective focal length or the eyepiece power",
    )


def _round2_reference() -> Bench:
    # 10x objective on a DIN 160 mm tube, 10x eyepiece: f_obj = 160/10 = 16 mm,
    # f_eye = 250/10 = 25 mm. The eyepiece sits one focal length past the
    # intermediate image so it sends a collimated bundle to a relaxed eye.
    f_obj = 16.0
    s_obj_distance = _round2_object_distance(f_obj)
    s_image = s_obj_distance + _round2_image_distance(f_obj)
    return Bench(
        [
            BenchElement(
                "objective", s_obj_distance, "objective", 4.0, f_obj,
                label="10x/0.25 DIN 160", metadata={"na": 0.25},
            ),
            BenchElement(
                "intermediate_image", s_image, "field_stop", 11.0,
                label="intermediate image / field stop",
            ),
            BenchElement(
                "eyepiece", s_image + 25.0, "eyepiece", 12.5, 25.0,
                label="10x eyepiece",
            ),
        ]
    )


ROUND_2 = Round(
    number=2,
    title="Compound microscope",
    brief=(
        "Build a 100x compound microscope on a DIN 160 mm tube: a 10x objective "
        "forming an intermediate image, viewed through a 10x eyepiece."
    ),
    teaches="the two-stage magnification of a compound scope and the 160 mm tube convention",
    s_object=0.0,
    wavelength_um=0.5461,
    evaluate=_round2_evaluate,
    reference_build=_round2_reference,
    parts_budget=3,
    available_kinds=("objective", "eyepiece", "field_stop"),
    notes=(
        "The objective's 10x is only true at its design tube length. This is the "
        "round where that convention stops being trivia."
    ),
)


# --- rounds 3-5: illumination ------------------------------------------------
#
# All three share one stand, so the player carries a build forward rather than
# starting over. The geometry is derived rather than chosen: the collector images
# the lamp onto the aperture diaphragm, the diaphragm sits at the condenser's front
# focal plane, and the condenser images the field diaphragm onto the specimen.

LAMP_S = 0.0
COLLECTOR_F = 18.0
# A collector of focal length f can only span a lamp-to-image distance of 4f or
# more. At f = 30 mm that is 120 mm, so such a collector CANNOT reach the 75 mm
# needed for critical illumination -- the quadratic has no real root. f = 18 mm
# reaches both targets, which makes collector position the single knob that
# switches critical illumination into Koehler.
COLLECTOR_S = 30.0          # critical: lamp imaged to 75 mm (the condenser's object)
COLLECTOR_KOHLER_S = 22.0526  # Koehler: lamp imaged to 120 mm (aperture diaphragm)
FIELD_DIAPHRAGM_S = 75.0
APERTURE_DIAPHRAGM_S = 120.0  # = lamp imaged by the collector
CONDENSER_F = 15.0
CONDENSER_S = APERTURE_DIAPHRAGM_S + CONDENSER_F  # diaphragm at the front focal plane
SPECIMEN_S = 155.0  # = field diaphragm imaged by the condenser


def _illumination_stand(
    objective_f: float = 16.0,
    objective_na: float = 0.25,
    aperture_semi_mm: float = 3.0,
    collector_s: float = COLLECTOR_S,
) -> Bench:
    """The shared dia-illumination stand for rounds 3-5."""
    objective_s = SPECIMEN_S + 1.0 / (1.0 / objective_f - 1.0 / (objective_f + OPTICAL_TUBE_LENGTH_MM))
    image_s = objective_s + objective_f + OPTICAL_TUBE_LENGTH_MM
    return Bench(
        [
            BenchElement("lamp", LAMP_S, "lamp", 2.0, None, label="tungsten filament"),
            BenchElement("collector", collector_s, "lens", 15.0, COLLECTOR_F, label="collector"),
            BenchElement("field_diaphragm", FIELD_DIAPHRAGM_S, "diaphragm", 8.0, None,
                         label="field diaphragm"),
            BenchElement("aperture_diaphragm", APERTURE_DIAPHRAGM_S, "diaphragm",
                         aperture_semi_mm, None, label="aperture diaphragm"),
            BenchElement("condenser", CONDENSER_S, "condenser", 10.0, CONDENSER_F,
                         label="condenser"),
            BenchElement("specimen", SPECIMEN_S, "field_stop", 12.0, None, label="specimen"),
            BenchElement("objective", objective_s, "objective", 4.0, objective_f,
                         label=f"objective NA {objective_na}", metadata={"na": objective_na}),
            BenchElement("intermediate_image", image_s, "field_stop", 11.0,
                         label="intermediate image"),
        ]
    )


def _collected_na(bench: Bench) -> float:
    """NA the collector gathers from the lamp: r / d."""
    collector = bench.get("collector")
    lamp = bench.get("lamp")
    return collector.semi_diameter_mm / max(collector.s - lamp.s, 1e-6)


def _round3_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    # Critical illumination puts the lamp on a FIELD plane, so depth of focus is
    # the right criterion -- you are meant to see the filament sharply.
    na_illum = condenser_na(bench, "aperture_diaphragm", "condenser")
    report.add(
        check_conjugate(
            bench, bench.get("lamp").s, "specimen", "Lamp on specimen",
            tolerance_mm=field_conjugate_tolerance_mm(0.5461, na_illum),
            from_label="the filament",
        )
    )
    report.add(check_illumination_throughput(bench, "lamp", "collector", required_na=0.4))
    report.add(check_no_unintended_clipping(bench, SPECIMEN_S, field_height_mm=0.3))
    # Deliberately advisory: this WARN is the observation round 4 exists to fix.
    report.add(
        check_illumination_uniformity(bench, "lamp", "specimen", "condenser", advisory=True)
    )
    return report


def collector_position_for(lamp_image_s: float, f: float = COLLECTOR_F) -> float:
    """Where to put the collector so the lamp images at ``lamp_image_s``.

    With the lamp at the origin, ``1/c + 1/(D - c) = 1/f`` gives
    ``c^2 - D c + f D = 0``. The two roots are the near and far conjugate pair;
    the near one is taken. A negative discriminant means the collector is too
    long-focus to span ``D`` at all -- no position works, which is a real
    constraint and not a solver failure.
    """
    discriminant = lamp_image_s**2 - 4.0 * f * lamp_image_s
    if discriminant < 0:
        raise ValueError(
            f"a collector of f = {f:.1f} mm cannot image across {lamp_image_s:.1f} mm; "
            f"it needs at least 4f = {4 * f:.1f} mm"
        )
    return (lamp_image_s - discriminant**0.5) / 2.0


def _round3_reference() -> Bench:
    # Critical illumination: the collector images the lamp to 75 mm, which the
    # condenser then relays onto the specimen -- so the filament lands in the field.
    return _illumination_stand(collector_s=collector_position_for(75.0))


ROUND_3 = Round(
    number=3,
    title="Make it bright",
    brief=(
        "Light the specimen. Focus the collector so the lamp filament is imaged "
        "onto the specimen, and get it close enough to gather the light you need. "
        "This is critical illumination, and it works."
    ),
    teaches=(
        "that illumination is its own imaging problem, with its own conjugates -- "
        "and that collected NA, not focus alone, decides how bright the field is"
    ),
    s_object=SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round3_evaluate,
    reference_build=_round3_reference,
    parts_budget=8,
    available_kinds=("lamp", "lens", "diaphragm", "condenser", "objective", "field_stop"),
    notes=(
        "Two collector positions focus the lamp onto the specimen -- the near and "
        "far conjugates -- but only the near one gathers enough light, because "
        "irradiance goes as the square of the collected NA. Critical illumination "
        "is bright and simple, and you can see the filament across the field. "
        "Round 4 is about getting rid of it without losing the light."
    ),
)


def _round4_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    bfp_s = objective.s + (objective.focal_length_mm or 0.0)

    # The four-plane conjugacy, stated as four questions.
    # The lamp lands on a PUPIL, so the criterion is whether its image still fits
    # inside the diaphragm -- not depth of focus, which would blur nothing here.
    pupil_tol = pupil_conjugate_tolerance_mm(
        bench.get("aperture_diaphragm").semi_diameter_mm, _collected_na(bench)
    )
    field_tol = field_conjugate_tolerance_mm(
        0.5461, condenser_na(bench, "aperture_diaphragm", "condenser")
    )
    report.add(
        check_conjugate(bench, bench.get("lamp").s, "aperture_diaphragm",
                        "Lamp on aperture diaphragm", pupil_tol, "the filament")
    )
    report.add(
        check_conjugate(bench, bench.get("field_diaphragm").s, "specimen",
                        "Field diaphragm on specimen", field_tol, "the field diaphragm")
    )
    report.add(
        check_conjugate_to_plane(
            bench, bench.get("aperture_diaphragm").s, bfp_s,
            "Aperture diaphragm on back focal plane",
            f"the objective back focal plane (s = {bfp_s:.2f} mm)",
            tolerance_mm=pupil_conjugate_tolerance_mm(
                bench.get("objective").semi_diameter_mm,
                float(objective.metadata.get("na", 0.25)),
            ),
            from_label="the aperture diaphragm",
        )
    )
    # ...and the half that is easy to forget.
    report.add(
        check_not_conjugate(bench, bench.get("lamp").s, "specimen",
                            "Filament off the specimen", 5.0, "the filament")
    )
    report.add(check_illumination_uniformity(bench, "lamp", "specimen", "condenser"))
    report.add(check_no_unintended_clipping(bench, SPECIMEN_S, field_height_mm=0.3))
    return report


def _round4_reference() -> Bench:
    # Koehler: the same collector, moved so the lamp images onto the aperture
    # diaphragm instead. That single move is the whole round.
    return _illumination_stand(collector_s=COLLECTOR_KOHLER_S)


ROUND_4 = Round(
    number=4,
    title="Köhler illumination",
    brief=(
        "Re-aim the illumination so the filament lands on the aperture diaphragm "
        "instead of the specimen, and the field diaphragm lands on the specimen. "
        "Even field, no filament, both diaphragms doing a job."
    ),
    teaches="the four-plane conjugacy that defines Köhler illumination",
    s_object=SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round4_evaluate,
    reference_build=_round4_reference,
    parts_budget=8,
    available_kinds=("lamp", "lens", "diaphragm", "condenser", "objective", "field_stop"),
    notes=(
        "Two interleaved sets: the FIELD set (field diaphragm, specimen, "
        "intermediate image) and the APERTURE set (lamp, aperture diaphragm, "
        "objective back focal plane). Put a white card at each to see the difference."
    ),
)


def _round5_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    na_obj = float(bench.get("objective").metadata.get("na", 0.0))
    na_cond = condenser_na(bench, "aperture_diaphragm", "condenser")

    report.add(check_resolution(na_obj, na_cond, 0.5461, required_um=0.50))
    report.add(check_condenser_na_match(na_obj, na_cond))
    report.add(
        check_conjugate(
            bench, bench.get("field_diaphragm").s, "specimen",
            "Field diaphragm on specimen",
            field_conjugate_tolerance_mm(0.5461, na_cond), "the field diaphragm",
        )
    )
    report.add(check_illumination_uniformity(bench, "lamp", "specimen", "condenser"))
    return report


def _round5_reference() -> Bench:
    # A 0.65 NA objective with the condenser opened to NA 0.55 (r = 0.55 * 15 mm).
    return _illumination_stand(
        objective_f=160.0 / 40.0,
        objective_na=0.65,
        aperture_semi_mm=0.55 * CONDENSER_F,
        collector_s=COLLECTOR_KOHLER_S,
    )


ROUND_5 = Round(
    number=5,
    title="Resolution",
    brief=(
        "Resolve 0.50 um. You have a 0.65 NA objective; the condenser aperture "
        "diaphragm is the other half of the equation."
    ),
    teaches="Abbe's d = lambda / (NA_obj + NA_cond), and why closing the condenser costs you",
    s_object=SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round5_evaluate,
    reference_build=_round5_reference,
    parts_budget=8,
    available_kinds=("lamp", "lens", "diaphragm", "condenser", "objective", "field_stop"),
    notes=(
        "Closing the aperture diaphragm makes the image look crisper and is the "
        "commonest way to throw resolution away. The condenser contributes to d "
        "exactly as much as the objective does."
    ),
)


ROUNDS: dict[int, Round] = {
    r.number: r for r in (ROUND_1, ROUND_2, ROUND_3, ROUND_4, ROUND_5)
}


def get_round(number: int) -> Round:
    if number not in ROUNDS:
        raise KeyError(f"round {number} is not implemented yet; have {sorted(ROUNDS)}")
    return ROUNDS[number]
