"""Round definitions: the specification a build is graded against.

A round is data plus one function. It states the brief, the parts budget, the test
object, and how to evaluate a bench -- but it never states *the answer*. There is a
reference solution, used only to prove the round is solvable and to power the
"show me a working build" diff after a failure.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable

from ..bench.bench import Arm, Bench, BenchElement
from ..rules.base import RuleReport
from ..rules.measured import (
    check_chromatic_focus,
    check_field_flatness,
    check_sensor_matches_the_optics,
)
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
    check_epi_separation,
    check_filter_is_focus_neutral,
    check_filter_set,
    check_infinity_space,
    check_parfocality,
    check_stokes_shift,
    check_illumination_throughput,
    check_illumination_uniformity,
    check_image_lands_on_detector,
    check_magnification,
    check_no_unintended_clipping,
    check_not_conjugate,
    check_optical_tube_length,
    check_relaxed_eye,
    check_resolution,
    check_sampling,
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
    # Measured rules render the image, which costs tenths of a second, so they run
    # on Run rather than on every drag. Rounds that only need ray geometry leave
    # this None and stay entirely on the live clock.
    measure: Callable[[Bench], RuleReport] | None = None
    parts_budget: int = 99
    available_kinds: tuple[str, ...] = ()
    notes: str = ""

    def grade(self, bench: Bench) -> RuleReport:
        """The live pass: ray geometry only, fast enough for every drag."""
        return self.evaluate(bench)

    def measure_image(self, bench: Bench) -> RuleReport:
        """The measured pass: renders, so it runs on Run."""
        return self.measure(bench) if self.measure else RuleReport()

    @property
    def has_measured_rules(self) -> bool:
        return self.measure is not None


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



# --- rounds 9-10: episcopic illumination and fluorescence --------------------
#
# The first branched bench. Epi illumination is not a folded path but a second
# path: it enters at the beamsplitter, runs BACK down through the objective to the
# specimen, and the returning light comes up the same way. The objective is its own
# condenser, which is the whole idea.
#
# Geometry solved against the engine, not by hand:
#   objective at the DIN working distance from the specimen
#   epi lens images the lamp onto the objective back focal plane   (aperture set)
#   field diaphragm images onto the specimen                        (field set)

EPI_F_OBJ = 16.0
EPI_SPECIMEN_S = 0.0
EPI_OBJECTIVE_S = 1.0 / (1.0 / EPI_F_OBJ - 1.0 / (EPI_F_OBJ + OPTICAL_TUBE_LENGTH_MM))
EPI_JUNCTION_S = 60.0
EPI_IMAGE_S = EPI_OBJECTIVE_S + EPI_F_OBJ + OPTICAL_TUBE_LENGTH_MM
EPI_ARM_LENGTH = 120.0
EPI_LENS_F = 30.0
EPI_LENS_S = 42.116          # images the lamp onto the back focal plane
EPI_FIELD_DIAPHRAGM_S = 22.615  # images onto the specimen through the objective


def _epi_arm_coordinate(main_s: float) -> float:
    return EPI_ARM_LENGTH + (EPI_JUNCTION_S - main_s)


def _epi_stand(
    epi_lens_s: float = EPI_LENS_S,
    field_diaphragm_s: float = EPI_FIELD_DIAPHRAGM_S,
    objective_na: float = 0.25,
    aperture_semi_mm: float = 4.0,
) -> Bench:
    return Bench(
        [
            BenchElement("specimen", EPI_SPECIMEN_S, "field_stop", 12.0, label="specimen"),
            BenchElement("objective", EPI_OBJECTIVE_S, "objective", 4.0, EPI_F_OBJ,
                         label=f"objective NA {objective_na}", metadata={"na": objective_na}),
            BenchElement("beamsplitter", EPI_JUNCTION_S, "beamsplitter", 12.0,
                         label="vertical illuminator"),
            BenchElement("intermediate_image", EPI_IMAGE_S, "field_stop", 11.0,
                         label="intermediate image"),
            BenchElement("lamp", 0.0, "lamp", 2.0, None, label="epi lamp", arm="epi"),
            BenchElement("field_diaphragm", field_diaphragm_s, "diaphragm", 6.0, None,
                         label="field diaphragm", arm="epi"),
            BenchElement("epi_lens", epi_lens_s, "lens", 12.0, EPI_LENS_F,
                         label="epi collector", arm="epi"),
            BenchElement("aperture_diaphragm", epi_lens_s + 1.0, "diaphragm",
                         aperture_semi_mm, None, label="aperture diaphragm", arm="epi"),
        ],
        arms=[Arm("epi", EPI_JUNCTION_S, EPI_ARM_LENGTH, (1.0, 0.0, 0.0), continues="reverse")],
    )


def _round9_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    bfp_arm_s = _epi_arm_coordinate(objective.s) - (objective.focal_length_mm or 0.0)
    specimen_arm_s = _epi_arm_coordinate(bench.get("specimen").s)

    report.add(check_epi_separation(bench, "beamsplitter", "objective"))
    report.add(
        _epi_conjugate(bench, bench.get("lamp").s, bfp_arm_s,
                       "Lamp on back focal plane", "the objective back focal plane",
                       pupil_conjugate_tolerance_mm(objective.semi_diameter_mm, 0.25),
                       "the epi lamp")
    )
    report.add(
        _epi_conjugate(bench, bench.get("field_diaphragm").s, specimen_arm_s,
                       "Field diaphragm on specimen", "the specimen",
                       field_conjugate_tolerance_mm(0.5461, 0.25), "the field diaphragm")
    )
    report.add(
        check_image_lands_on_detector(
            bench, EPI_SPECIMEN_S, "intermediate_image",
            image_side_depth_of_focus_mm(0.5461, 0.25, 10.0),
        )
    )
    return report


def _epi_conjugate(bench, s_from, s_target, label, target_label, tolerance, from_label):
    """check_conjugate_to_plane against the epi arm rather than the main axis."""
    from ..rules.base import Status, tolerance_result

    system = bench.to_paraxial("epi")
    s_img = system.image_plane(s_from, search_to=s_target + 1e-6)
    if s_img is None:
        from ..rules.base import RuleResult

        return RuleResult(
            name=label, status=Status.FAIL,
            summary=f"{from_label} is not imaged onto {target_label}",
            equation=f"no finite conjugate before {target_label} (arm s = {s_target:.2f} mm)",
            remedy="refocus the epi collector",
        )
    return tolerance_result(
        name=label, measured=s_img, target=s_target, tolerance=tolerance, units="mm",
        equation=(
            f"along the epi arm, the conjugate of {from_label} lands at "
            f"s = {s_img:.2f} mm; {target_label} is at s = {s_target:.2f} mm"
        ),
        remedy=f"move the epi collector so the conjugate falls on {target_label}",
        relative=False,
    )


ROUND_9 = Round(
    number=9,
    title="Episcopic illumination",
    brief=(
        "Light an opaque specimen from above. The objective is its own condenser: "
        "send the illumination in at the beamsplitter, back down through the "
        "objective, and image the field diaphragm onto the specimen."
    ),
    teaches="that an epi path is a second light path sharing the objective, not a folded one",
    s_object=EPI_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round9_evaluate,
    reference_build=_epi_stand,
    parts_budget=8,
    available_kinds=("lamp", "lens", "diaphragm", "beamsplitter", "objective", "field_stop"),
    notes=(
        "Köhler still applies, with the objective back focal plane standing in for "
        "the condenser's: the lamp goes to the aperture set, the field diaphragm to "
        "the field set. Only now both sets live on a branch of the bench."
    ),
)


def _round10_evaluate(bench: Bench) -> RuleReport:
    from ..optics.spectra import CUBES, FLUOROPHORES

    report = RuleReport()
    cube = CUBES[bench.get("beamsplitter").metadata.get("cube", "fitc")]
    dye = FLUOROPHORES[bench.get("specimen").metadata.get("fluorophore", "fitc")]

    report.add(check_stokes_shift(dye, cube))
    report.add(check_filter_set(cube, dye))
    report.add(check_epi_separation(bench, "beamsplitter", "objective"))
    report.add(
        _epi_conjugate(
            bench, bench.get("field_diaphragm").s,
            _epi_arm_coordinate(bench.get("specimen").s),
            "Field diaphragm on specimen", "the specimen",
            field_conjugate_tolerance_mm(0.5461, 0.25), "the field diaphragm",
        )
    )
    return report


def _round10_reference() -> Bench:
    bench = _epi_stand(objective_na=0.75)
    bench.elements = [
        dataclasses.replace(e, metadata={**e.metadata, "cube": "fitc"})
        if e.name == "beamsplitter"
        else dataclasses.replace(e, metadata={**e.metadata, "fluorophore": "fitc"})
        if e.name == "specimen"
        else e
        for e in bench.elements
    ]
    return bench


ROUND_10 = Round(
    number=10,
    title="Epi-fluorescence",
    brief=(
        "Swap the beamsplitter for a dichroic filter cube and image a FITC-labelled "
        "specimen. Excitation must reach the dye; emission must reach the eye; "
        "excitation must not."
    ),
    teaches="that the Stokes shift is what makes fluorescence separable from its own illumination",
    s_object=EPI_SPECIMEN_S,
    wavelength_um=0.519,  # FITC emission
    evaluate=_round10_evaluate,
    reference_build=_round10_reference,
    parts_budget=8,
    available_kinds=("lamp", "lens", "diaphragm", "dichroic", "objective", "field_stop"),
    notes=(
        "Excitation is orders of magnitude brighter than emission, so a few percent "
        "of bleedthrough is the difference between a black background and a grey haze."
    ),
)

# --- rounds 11-12: infinity correction and the full stand --------------------
#
# CFI60 conventions throughout: 200 mm tube lens, 60 mm parfocal distance,
# f_objective = 200 / M. The specimen sits at the objective's front focal plane,
# so the space behind the objective is collimated and its length does not matter.

CFI_TUBE_F = 200.0
CFI_PARFOCAL = 60.0
INF_SPECIMEN_S = 0.0
INF_TUBE_LENS_S = 180.0
INF_FILTER_S = 90.0          # in the infinity space, where a plate is harmless
FILTER_THICKNESS_MM = 5.0
FILTER_INDEX = 1.52


def _infinity_stand(
    magnification: float = 20.0,
    na: float = 0.75,
    filter_s: float = INF_FILTER_S,
    tube_lens_s: float = INF_TUBE_LENS_S,
) -> Bench:
    """A CFI60 infinity stand: objective, infinity space, filter, tube lens, sensor."""
    f_obj = CFI_TUBE_F / magnification
    return Bench(
        [
            BenchElement("specimen", INF_SPECIMEN_S, "field_stop", 12.0, label="specimen"),
            BenchElement(
                "objective", INF_SPECIMEN_S + f_obj, "objective", 6.0, f_obj,
                label=f"CFI {magnification:.0f}x/{na}", catalog_key="cfi_plan_apo_20x",
                metadata={"na": na, "magnification": magnification, "parfocal_mm": CFI_PARFOCAL},
            ),
            BenchElement(
                "filter", filter_s, "filter", 12.5, None, label="5 mm filter",
                metadata={"thickness_mm": FILTER_THICKNESS_MM, "index": FILTER_INDEX},
            ),
            BenchElement("tube_lens", tube_lens_s, "tube_lens", 13.0, CFI_TUBE_F,
                         label="CFI tube lens f = 200 mm"),
            BenchElement("sensor", tube_lens_s + CFI_TUBE_F, "detector", 11.0, label="sensor"),
        ]
    )


def _round11_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    na = float(objective.metadata.get("na", 0.75))
    target_m = float(objective.metadata.get("magnification", 20.0))

    report.add(check_infinity_space(bench, INF_SPECIMEN_S, "objective"))
    report.add(
        check_image_lands_on_detector(
            bench, INF_SPECIMEN_S, "sensor",
            image_side_depth_of_focus_mm(0.5461, na, target_m),
        )
    )
    report.add(
        check_magnification(bench, INF_SPECIMEN_S, "sensor", target_m, MAGNIFICATION_TOLERANCE)
    )
    # The round's real point: the filter must cost nothing.
    report.add(check_filter_is_focus_neutral(bench, INF_SPECIMEN_S, "sensor", "filter"))
    return report


ROUND_11 = Round(
    number=11,
    title="Infinity correction",
    brief=(
        "Rebuild the scope with no tube length at all. Put the specimen at the "
        "objective's front focal plane, add a 200 mm tube lens, and drop a 5 mm "
        "filter into the space between them without refocusing."
    ),
    teaches=(
        "that an infinity space is collimated, so its length is free and a plate "
        "inserted in it does not shift focus"
    ),
    s_object=INF_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round11_evaluate,
    reference_build=_infinity_stand,
    parts_budget=5,
    available_kinds=("objective", "tube_lens", "filter", "detector", "field_stop"),
    notes=(
        "M = f_tube / f_objective, and it does not depend on the separation. A plate "
        "of thickness t and index n shifts focus by t(1 - 1/n) in converging light "
        "and by nothing at all in collimated light -- that difference is why the "
        "whole industry moved to infinity optics."
    ),
)


TURRET = (
    ("objective_10x", 10.0, 0.45),
    ("objective_20x", 20.0, 0.75),
    ("objective_40x", 40.0, 0.95),
)


def _full_stand() -> Bench:
    """A CFI60 stand with a three-objective parfocal turret and both light paths."""
    elements = [
        BenchElement("specimen", INF_SPECIMEN_S, "field_stop", 12.0, label="specimen"),
        BenchElement("filter", INF_FILTER_S, "filter", 12.5, None, label="5 mm filter",
                     metadata={"thickness_mm": FILTER_THICKNESS_MM, "index": FILTER_INDEX}),
        BenchElement("tube_lens", INF_TUBE_LENS_S, "tube_lens", 13.0, CFI_TUBE_F,
                     label="CFI tube lens"),
        BenchElement("sensor", INF_TUBE_LENS_S + CFI_TUBE_F, "detector", 11.0, label="sensor"),
    ]
    for name, magnification, na in TURRET:
        f = CFI_TUBE_F / magnification
        elements.append(
            BenchElement(
                name, INF_SPECIMEN_S + f, "objective", 6.0, f,
                label=f"CFI {magnification:.0f}x/{na}",
                # Only the first position is in the light path; the others are
                # fitted but swung out, exactly as on a real turret.
                enabled=(name == TURRET[0][0]),
                metadata={"na": na, "magnification": magnification,
                          "parfocal_mm": CFI_PARFOCAL, "in_turret": True},
            )
        )
    return Bench(elements)


def _stand_with_objective(bench: Bench, keep: str) -> Bench:
    """The stand as it is with one objective rotated into the light path."""
    return Bench(
        [
            dataclasses.replace(e, enabled=(e.name == keep))
            if e.metadata.get("in_turret") else e
            for e in bench.elements
        ],
        bench.folds, tuple(bench.origin), tuple(bench.initial_direction),
        list(bench.arms.values()),
    )


def _round12_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    turret = [e.name for e in bench.elements if e.metadata.get("in_turret")]
    report.add(check_parfocality(bench, turret, CFI_PARFOCAL))

    # Every objective in the turret must work without moving anything else.
    for name in turret:
        single = _stand_with_objective(bench, name)
        objective = bench.get(name)
        na = float(objective.metadata.get("na", 0.5))
        magnification = float(objective.metadata.get("magnification", 1.0))

        focus = check_image_lands_on_detector(
            single, INF_SPECIMEN_S, "sensor",
            image_side_depth_of_focus_mm(0.5461, na, magnification),
        )
        report.add(dataclasses.replace(focus, name=f"Focus [{magnification:.0f}x]"))

        mag = check_magnification(
            single, INF_SPECIMEN_S, "sensor", magnification, MAGNIFICATION_TOLERANCE
        )
        report.add(dataclasses.replace(mag, name=f"Magnification [{magnification:.0f}x]"))

        infinity = check_infinity_space(single, INF_SPECIMEN_S, name)
        report.add(dataclasses.replace(infinity, name=f"Infinity space [{magnification:.0f}x]"))

    report.add(
        check_filter_is_focus_neutral(
            _stand_with_objective(bench, turret[0]), INF_SPECIMEN_S, "sensor", "filter"
        )
    )
    return report


ROUND_12 = Round(
    number=12,
    title="The full stand",
    brief=(
        "Fit a three-objective parfocal turret to the infinity stand. Rotating "
        "between 10x, 20x and 40x must keep focus, keep the sensor where it is, "
        "and give the magnification on the label."
    ),
    teaches="that parfocality is a mechanical promise the optics have to honour",
    s_object=INF_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round12_evaluate,
    reference_build=_full_stand,
    parts_budget=8,
    available_kinds=("objective", "tube_lens", "filter", "detector", "field_stop"),
    notes=(
        "Every objective in a matched set has the same shoulder-to-specimen "
        "distance -- 60 mm for CFI60 -- so a 10x sitting 20 mm above the specimen "
        "and a 40x sitting 5 mm above it still focus together. Mixing series breaks it."
    ),
)


# --- rounds 6-8: what only the image can tell you ----------------------------
#
# These three are graded on measurements taken from the rendered picture, not on
# ray geometry, so each carries a `measure` pass alongside its live `evaluate`.
# They share the infinity stand from round 11's family, because by this point the
# player has a working scope and is choosing components for it rather than
# rebuilding the light path.

ROUND6_TEST_PERIOD_UM = 1.2
# Mid-band, well inside the cutoff. Measured near the cutoff the ratio is noise:
# see check_field_flatness for the numbers that forced this choice.
ROUND7_TEST_PERIOD_UM = 2.5
CAMERA_PIXEL_UM = 6.5  # a very common scientific CMOS pixel


def _component_stand(
    catalog_key: str,
    na: float,
    magnification: float = 20.0,
    grade: str = "plan_apochromat",
    pixel_um: float = CAMERA_PIXEL_UM,
    extra_magnification: float = 1.0,
) -> Bench:
    """An infinity stand whose objective is a named catalog part."""
    f_obj = CFI_TUBE_F / magnification
    tube_f = CFI_TUBE_F * extra_magnification
    tube_s = INF_TUBE_LENS_S
    return Bench(
        [
            BenchElement("specimen", INF_SPECIMEN_S, "field_stop", 12.0, label="specimen"),
            BenchElement(
                "objective", INF_SPECIMEN_S + f_obj, "objective", 6.0, f_obj,
                label=catalog_key, catalog_key=catalog_key,
                metadata={"na": na, "magnification": magnification, "grade": grade,
                          "parfocal_mm": CFI_PARFOCAL},
            ),
            BenchElement("tube_lens", tube_s, "tube_lens", 13.0, tube_f,
                         label=f"tube lens f = {tube_f:.0f} mm"),
            BenchElement("sensor", tube_s + tube_f, "detector", 11.0, label="camera",
                         metadata={"pixel_um": pixel_um}),
        ]
    )


def _round6_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    na = float(objective.metadata.get("na", 0.4))
    magnification = float(objective.metadata.get("magnification", 20.0))
    report.add(check_infinity_space(bench, INF_SPECIMEN_S, "objective"))
    report.add(
        check_image_lands_on_detector(
            bench, INF_SPECIMEN_S, "sensor",
            image_side_depth_of_focus_mm(0.5461, na, magnification),
        )
    )
    return report


def _round6_measure(bench: Bench) -> RuleReport:
    report = RuleReport()
    report.add(check_chromatic_focus(bench, INF_SPECIMEN_S, "sensor"))
    return report


def _round6_reference() -> Bench:
    # A plan apochromat: three wavelengths to a common focus, so the F-to-C band
    # fits inside the depth of focus.
    return _component_stand("cfi_plan_apo_20x", na=0.75, grade="plan_apochromat")


ROUND_6 = Round(
    number=6,
    title="Colour",
    brief=(
        "Image in white light. Every wavelength from F to C must come to focus "
        "together -- one focus setting that serves the whole band."
    ),
    teaches="secondary spectrum, and what separates an achromat from an apochromat",
    s_object=INF_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round6_evaluate,
    measure=_round6_measure,
    reference_build=_round6_reference,
    parts_budget=4,
    available_kinds=("objective", "tube_lens", "detector", "field_stop"),
    notes=(
        "An achromat's residual secondary spectrum is about f/2000 across the "
        "visible band. At a 10 mm focal length that is 5 um of focus shift, against "
        "a depth of focus under 2 um at NA 0.4 -- so blue is soft however you focus."
    ),
)


def _round7_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    na = float(objective.metadata.get("na", 0.4))
    magnification = float(objective.metadata.get("magnification", 20.0))
    report.add(
        check_image_lands_on_detector(
            bench, INF_SPECIMEN_S, "sensor",
            image_side_depth_of_focus_mm(0.5461, na, magnification),
        )
    )
    report.add(check_infinity_space(bench, INF_SPECIMEN_S, "objective"))
    return report


def _round7_measure(bench: Bench) -> RuleReport:
    report = RuleReport()
    report.add(
        check_field_flatness(bench, INF_SPECIMEN_S, "sensor", ROUND7_TEST_PERIOD_UM)
    )
    return report


def _round7_reference() -> Bench:
    return _component_stand("cfi_plan_achro_20x", na=0.40, grade="plan_achromat")


ROUND_7 = Round(
    number=7,
    title="Flat field",
    brief=(
        "Fill the frame. The corners have to be as sharp as the centre -- not "
        "sharp once you refocus for them, sharp at the same time."
    ),
    teaches="field curvature, and what the 'plan' in a plan objective is paying for",
    s_object=INF_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round7_evaluate,
    measure=_round7_measure,
    reference_build=_round7_reference,
    parts_budget=4,
    available_kinds=("objective", "tube_lens", "detector", "field_stop"),
    notes=(
        "A plain achromat leaves a Petzval sag of tens of microns. The depth of "
        "focus at NA 0.4 is 1.7 um, so refocusing trades the centre for the corners "
        "instead of fixing either. A plan design flattens the field optically."
    ),
)


def _round8_evaluate(bench: Bench) -> RuleReport:
    report = RuleReport()
    objective = bench.get("objective")
    na = float(objective.metadata.get("na", 0.75))
    magnification = float(objective.metadata.get("magnification", 20.0))
    pixel_um = float(bench.get("sensor").metadata.get("pixel_um", CAMERA_PIXEL_UM))
    tube_ratio = (bench.get("tube_lens").focal_length_mm or CFI_TUBE_F) / CFI_TUBE_F

    report.add(
        check_image_lands_on_detector(
            bench, INF_SPECIMEN_S, "sensor",
            image_side_depth_of_focus_mm(0.5461, na, magnification * tube_ratio),
        )
    )
    report.add(check_sampling(0.5461, na, magnification * tube_ratio, pixel_um))
    return report


def _round8_measure(bench: Bench) -> RuleReport:
    report = RuleReport()
    pixel_um = float(bench.get("sensor").metadata.get("pixel_um", CAMERA_PIXEL_UM))
    report.add(
        check_sensor_matches_the_optics(bench, INF_SPECIMEN_S, "sensor", pixel_um)
    )
    return report


def _round8_reference() -> Bench:
    # A 20x/0.75 resolves 0.44 um, which is 8.9 um at the sensor: a 6.5 um pixel
    # undersamples it. A 1.5x tube-lens changer fixes that without tipping over
    # into empty magnification.
    return _component_stand(
        "cfi_plan_apo_20x", na=0.75, grade="plan_apochromat", extra_magnification=1.5
    )


ROUND_8 = Round(
    number=8,
    title="Camera port",
    brief=(
        "Put a camera on it. The sensor has to keep what the optics resolved -- "
        "and no more than that, because extra magnification costs field and light."
    ),
    teaches="Nyquist sampling at the sensor, and why empty magnification is a trap",
    s_object=INF_SPECIMEN_S,
    wavelength_um=0.5461,
    evaluate=_round8_evaluate,
    measure=_round8_measure,
    reference_build=_round8_reference,
    parts_budget=4,
    available_kinds=("objective", "tube_lens", "detector", "field_stop"),
    notes=(
        "A 20x/0.75 resolves 0.444 um; at the sensor that is 8.9 um, so Nyquist "
        "demands pixels no larger than 4.44 um. The 6.5 um pixel on a very common "
        "scientific CMOS undersamples it, and a 1.5x changer is the usual fix."
    ),
)


# --- sandbox -----------------------------------------------------------------


def _sandbox_evaluate(bench: Bench) -> RuleReport:
    """Diagnostics, not a verdict.

    The sandbox reports what the build is doing without grading it: where the image
    lands, what the stop is, what NA you have, what it can resolve. Nothing fails,
    because there is no brief to fail against -- which is the point of a sandbox.
    """
    from ..rules.base import RuleResult, Status

    report = RuleReport()
    system = bench.to_paraxial()
    s_object = min([e.s for e in bench.elements if e.arm == "main"] or [0.0])
    image_s = system.image_plane(s_object)
    stop = system.aperture_stop(s_object)
    na = system.object_space_na(s_object)

    report.add(
        RuleResult(
            "Image plane", Status.NOT_APPLICABLE,
            "output is collimated (afocal)" if image_s is None
            else f"image forms at s = {image_s:.2f} mm",
            equation="B = 0 of the object-to-image matrix",
            measured=image_s,
        )
    )
    if image_s is not None:
        report.add(
            RuleResult(
                "Magnification", Status.NOT_APPLICABLE,
                f"{abs(system.magnification(s_object, image_s)):.3f}x",
                equation="M = A of the object-to-image matrix",
            )
        )
    report.add(
        RuleResult(
            "Aperture stop", Status.NOT_APPLICABLE,
            stop.name if stop else "none: nothing limits the bundle",
            equation="smallest ratio of clear aperture to ray height",
        )
    )
    if na > 0:
        report.add(
            RuleResult(
                "Numerical aperture", Status.NOT_APPLICABLE,
                f"NA {na:.3f}, resolving {abs(0.5461 / na):.3f} um at best",
                equation=f"NA = n sin(u) = {na:.3f}; d = lambda / NA",
                measured=na,
            )
        )
    return report


SANDBOX = Round(
    number=0,
    title="Sandbox",
    brief="No brief. Build whatever you like; the panel reports what it does.",
    teaches="whatever you decide to try",
    s_object=0.0,
    wavelength_um=0.5461,
    evaluate=_sandbox_evaluate,
    reference_build=_infinity_stand,
    parts_budget=99,
    notes="Unlocks alongside round 5. Nothing here passes or fails.",
)


def get_round(number: int) -> Round:
    if number not in ROUNDS:
        raise KeyError(f"round {number} is not implemented yet; have {sorted(ROUNDS)}")
    return ROUNDS[number]


ROUNDS: dict[int, Round] = {
    r.number: r
    for r in (
        ROUND_1, ROUND_2, ROUND_3, ROUND_4, ROUND_5, ROUND_6, ROUND_7, ROUND_8,
        ROUND_9, ROUND_10, ROUND_11, ROUND_12, SANDBOX,
    )
}
