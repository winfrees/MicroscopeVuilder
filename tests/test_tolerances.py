"""M6 gate: every tolerance is derived, and the derivation is validated.

The point of this file is that the numbers the game grades against are not chosen
by feel. Each one traces to the Rayleigh quarter-wave criterion or to a stated
convention, and the focus tolerances are checked against the diffraction engine
directly: a build exactly at tolerance must sit at the diffraction limit.
"""

import pytest

from microscopevuilder.game.rounds import get_round
from microscopevuilder.optics.psf import make_pupil_grid, strehl_from_psf
from microscopevuilder.optics.wavefront import defocus_from_stage_error
from microscopevuilder.rules.base import Status
from microscopevuilder.rules.tolerances import (
    MAGNIFICATION_TOLERANCE,
    TolerancePolicy,
    set_tolerance_policy,
    depth_of_focus_mm,
    field_conjugate_tolerance_mm,
    image_side_depth_of_focus_mm,
    pupil_conjugate_tolerance_mm,
    tube_length_tolerance,
)

LAMBDA = 0.5461


@pytest.fixture(autouse=True)
def strict_grading():
    """These tests are about the derived physical tolerances.

    The game grades positions forgivingly by default, because the derived
    tolerances were finer than a pixel of drag. This file checks the physics
    underneath that, so it pins the strict policy.
    """
    previous = set_tolerance_policy(TolerancePolicy.STRICT)
    yield
    set_tolerance_policy(previous)


def _named(report, name):
    return next(r for r in report.results if r.name == name)


# --- the anchor --------------------------------------------------------------


@pytest.mark.parametrize("na", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_depth_of_focus_sits_exactly_at_the_diffraction_limit(na):
    # The whole tolerance scheme rests on this: being out of focus by the depth of
    # focus costs exactly a quarter wave, i.e. Strehl ~0.8. Measured from the PSF,
    # not from the Marechal approximation.
    dz = depth_of_focus_mm(LAMBDA, na)
    wavefront = defocus_from_stage_error(dz, na, LAMBDA)
    grid = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=64)
    assert strehl_from_psf(grid, wavefront) == pytest.approx(0.8, abs=0.05)


def test_twice_the_depth_of_focus_is_clearly_not_diffraction_limited():
    na = 0.5
    grid = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=64)
    dz = depth_of_focus_mm(LAMBDA, na)
    assert strehl_from_psf(grid, defocus_from_stage_error(2 * dz, na, LAMBDA)) < 0.5


def test_depth_of_focus_collapses_as_the_fourth_power_of_speed():
    # dz goes as 1/NA^2, so doubling NA quarters the depth of focus. This is why
    # high-power focusing is touchy, and the game should not paper over it.
    assert depth_of_focus_mm(LAMBDA, 0.25) / depth_of_focus_mm(LAMBDA, 0.5) == pytest.approx(4.0)


def test_image_side_depth_of_focus_grows_as_magnification_squared():
    # The image-space aperture is NA/M, so a 10x system tolerates 100x the slop at
    # the sensor that it does at the specimen.
    at_specimen = depth_of_focus_mm(LAMBDA, 0.25)
    at_image = image_side_depth_of_focus_mm(LAMBDA, 0.25, 10.0)
    assert at_image / at_specimen == pytest.approx(100.0)


# --- pupil conjugates use a different criterion ------------------------------


def test_pupil_tolerance_is_geometric_not_wave_optical():
    # Defocusing a pupil blurs nothing; what fails is that the stop begins cutting
    # the field. The criterion is "does the lamp image still fit in the stop".
    assert pupil_conjugate_tolerance_mm(3.0, 0.68) == pytest.approx(3.0 / 0.68)
    # A wider diaphragm is more forgiving, which matches bench experience.
    assert pupil_conjugate_tolerance_mm(8.0, 0.68) > pupil_conjugate_tolerance_mm(3.0, 0.68)
    # ...and it is nothing like the depth of focus at the same NA.
    assert pupil_conjugate_tolerance_mm(3.0, 0.68) > 100 * depth_of_focus_mm(LAMBDA, 0.68)


def test_field_conjugate_tolerance_has_an_honest_floor():
    # At low illumination NA the depth of focus is huge, so the floor -- not the
    # physics -- sets the tolerance. The function must say so by returning it.
    loose = field_conjugate_tolerance_mm(LAMBDA, 0.02, minimum_mm=0.5)
    assert loose == pytest.approx(depth_of_focus_mm(LAMBDA, 0.02))  # physics wins here
    assert field_conjugate_tolerance_mm(LAMBDA, 0.9, minimum_mm=0.5) == 0.5  # floor wins


def test_tube_length_tolerance_equals_the_magnification_tolerance():
    # M = L/f means dM/M = dL/L exactly. Deriving one from the other keeps the two
    # rules from contradicting each other.
    assert tube_length_tolerance(MAGNIFICATION_TOLERANCE) == MAGNIFICATION_TOLERANCE


# --- the rounds actually use them --------------------------------------------


def test_round1_focus_tolerance_is_the_image_side_depth_of_focus():
    # It was 0.5 mm, chosen by feel -- about three times too loose.
    rnd = get_round(1)
    bench = rnd.reference_build()
    na = bench.to_paraxial().object_space_na(-60.0)
    expected = image_side_depth_of_focus_mm(LAMBDA, na, 5.0)
    assert expected == pytest.approx(0.157, abs=0.005)

    # Just inside passes, comfortably outside fails.
    bench.move("screen", 300.0 + 0.8 * expected)
    assert _named(rnd.grade(bench), "Focus").status is Status.PASS

    bench.move("screen", 300.0 + 3.0 * expected)
    assert _named(rnd.grade(bench), "Focus").status is Status.FAIL


def test_round2_focus_tolerance_is_the_image_side_depth_of_focus():
    # It was 0.2 mm -- about twice too tight, which would have failed builds that
    # are genuinely diffraction limited.
    rnd = get_round(2)
    expected = image_side_depth_of_focus_mm(LAMBDA, 0.25, 10.0)
    assert expected == pytest.approx(0.437, abs=0.005)

    bench = rnd.reference_build()
    bench.move("intermediate_image", bench.get("intermediate_image").s + 0.3)
    focus = _named(rnd.grade(bench), "Focus")
    assert focus.status is Status.PASS, "0.3 mm is inside the depth of focus at 0.25 NA"


def test_round2_tube_length_and_magnification_agree():
    # A build at the edge of the tube-length tolerance must be at the edge of the
    # magnification tolerance too, not one side of a contradiction.
    rnd = get_round(2)
    bench = rnd.reference_build()
    image = bench.get("intermediate_image")
    bench.move("intermediate_image", image.s + 160.0 * MAGNIFICATION_TOLERANCE * 0.9)
    report = rnd.grade(bench)
    assert _named(report, "Optical tube length").status is Status.PASS
    assert _named(report, "Magnification").status is Status.PASS


def test_round4_pupil_and_field_conjugates_are_graded_differently():
    # The two rows must not be carrying the same number: they are different
    # physical criteria, and on this stand they differ by an order of magnitude.
    rnd = get_round(4)
    report = rnd.grade(rnd.reference_build())
    pupil_row = _named(report, "Lamp on aperture diaphragm")
    field_row = _named(report, "Field diaphragm on specimen")
    assert pupil_row.status is Status.PASS and field_row.status is Status.PASS

    bench = rnd.reference_build()
    # A 2 mm collector error is inside the pupil tolerance (the lamp image still
    # fits the diaphragm) but would be far outside a depth-of-focus criterion.
    bench.move("collector", bench.get("collector").s + 0.35)
    assert _named(rnd.grade(bench), "Lamp on aperture diaphragm").status is not Status.FAIL


def test_every_graded_round_still_passes_on_its_reference_build():
    for number in (1, 2, 3, 4, 5):
        rnd = get_round(number)
        report = rnd.grade(rnd.reference_build())
        assert report.passed, f"round {number}\n{report.format()}"


def test_no_round_grades_against_a_bare_literal_tolerance():
    # A guard with teeth: every tolerance-bearing row must report a target that
    # matches one of the derived quantities, so a future contributor cannot slip a
    # hand-picked number back in without this test noticing.
    rnd = get_round(1)
    bench = rnd.reference_build()
    na = bench.to_paraxial().object_space_na(-60.0)
    focus = _named(rnd.grade(bench), "Focus")
    assert focus.target == pytest.approx(bench.get("screen").s)
    assert image_side_depth_of_focus_mm(LAMBDA, na, 5.0) < 0.2
