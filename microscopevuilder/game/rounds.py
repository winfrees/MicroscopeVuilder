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
from ..rules.checks import (
    check_image_lands_on_detector,
    check_magnification,
    check_no_unintended_clipping,
    check_resolution,
    check_optical_tube_length,
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
    report.add(check_image_lands_on_detector(bench, s_obj, "screen", tolerance_mm=0.5))
    report.add(check_magnification(bench, s_obj, "screen", target=5.0, tolerance=0.05))
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

    report.add(
        check_optical_tube_length(bench, "objective", "intermediate_image", OPTICAL_TUBE_LENGTH_MM)
    )
    report.add(check_image_lands_on_detector(bench, s_obj, "intermediate_image", 0.2))
    report.add(check_magnification(bench, s_obj, "intermediate_image", target=10.0, tolerance=0.05))
    report.add(check_no_unintended_clipping(bench, s_obj, field_height_mm=0.5))
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


ROUNDS: dict[int, Round] = {r.number: r for r in (ROUND_1, ROUND_2)}


def get_round(number: int) -> Round:
    if number not in ROUNDS:
        raise KeyError(f"round {number} is not implemented yet; have {sorted(ROUNDS)}")
    return ROUNDS[number]
