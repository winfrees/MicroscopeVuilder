"""M3 gate: the engine composes into a solvable, correctly-failing puzzle.

A round is only a puzzle if the reference build passes AND plausible wrong builds
fail for the *right stated reason*. A rule that fails everything is not teaching.
"""

import dataclasses

import pytest

from microscopevuilder.bench.bench import Bench, BenchElement, Fold
from microscopevuilder.game.rounds import ROUNDS, get_round
from microscopevuilder.rules.base import Status


@pytest.mark.parametrize("number", sorted(ROUNDS))
def test_reference_build_passes_its_own_round(number):
    rnd = get_round(number)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()


@pytest.mark.parametrize("number", sorted(ROUNDS))
def test_every_result_explains_itself(number):
    # The explanation is the product for this audience; a bare boolean is a bug.
    rnd = get_round(number)
    for result in rnd.grade(rnd.reference_build()).results:
        assert result.summary
        if result.status is not Status.NOT_APPLICABLE:
            assert result.equation, f"{result.name} states no governing equation"


def _named(report, name):
    return next(r for r in report.results if r.name == name)


def test_round1_wrong_focal_length_fails_on_magnification_not_just_focus():
    rnd = get_round(1)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, focal_length_mm=40.0) if e.name == "lens" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    assert not report.passed
    assert _named(report, "Magnification").status is Status.FAIL


def test_round1_moving_the_screen_alone_cannot_fix_magnification():
    # The teaching point in round 1's notes: M and focus are not independent knobs.
    rnd = get_round(1)
    bench = rnd.reference_build()
    bench.move("screen", 200.0)
    report = rnd.grade(bench)
    assert _named(report, "Focus").status is Status.FAIL
    assert _named(report, "Magnification").status is Status.FAIL


def test_round1_remedy_names_the_plane_the_image_actually_forms_at():
    rnd = get_round(1)
    bench = rnd.reference_build()
    bench.move("screen", 250.0)
    focus = _named(rnd.grade(bench), "Focus")
    assert focus.status is Status.FAIL
    assert "300" in focus.remedy  # where the image really is


def test_round1_undersized_screen_is_caught_as_vignetting():
    # The object is 1 mm off axis and M = 5, so the image is 5 mm off axis. A
    # screen with a 3 mm semi-diameter cannot receive it.
    rnd = get_round(1)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, semi_diameter_mm=3.0) if e.name == "screen" else e
        for e in bench.elements
    ]
    clipping = _named(rnd.grade(bench), "Clear aperture")
    assert clipping.status is Status.FAIL
    assert clipping.culprit == "screen"


def test_a_small_lens_is_a_slow_system_not_a_vignetting_one():
    # Worth pinning, because it is a natural thing to get wrong: shrinking the
    # lens makes it the aperture stop, which lowers NA and dims the image. It does
    # not clip the field. Vignetting is what happens when something OTHER than the
    # stop cuts into the off-axis bundle, so this must stay a PASS here.
    rnd = get_round(1)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, semi_diameter_mm=0.4) if e.name == "lens" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    assert _named(report, "Clear aperture").status is Status.PASS
    assert bench.to_paraxial().aperture_stop(-60.0).name == "lens"
    # ...and the system really is slower: NA has dropped by the diameter ratio.
    assert bench.to_paraxial().object_space_na(-60.0) == pytest.approx(0.4 / 60.0, rel=1e-6)


def test_round2_wrong_tube_length_is_reported_as_tube_length():
    # Move the intermediate image 20 mm and the objective is no longer 10x. The
    # round must say "tube length", not merely "magnification".
    rnd = get_round(2)
    bench = rnd.reference_build()
    bench.move("intermediate_image", bench.get("intermediate_image").s + 20.0)
    report = rnd.grade(bench)
    assert _named(report, "Optical tube length").status is Status.FAIL
    assert "back focal plane" in _named(report, "Optical tube length").equation


def test_round2_optical_tube_length_is_measured_from_the_back_focal_plane():
    # The distinction is worth a whole magnification step at 10x: measuring from
    # the objective itself gives 9.0x, not 10x.
    rnd = get_round(2)
    bench = rnd.reference_build()
    objective = bench.get("objective")
    image = bench.get("intermediate_image")
    from_lens = image.s - objective.s
    from_bfp = image.s - (objective.s + objective.focal_length_mm)

    assert from_bfp == pytest.approx(160.0)
    assert from_lens == pytest.approx(176.0)
    assert from_lens / objective.focal_length_mm == pytest.approx(11.0)
    assert from_bfp / objective.focal_length_mm == pytest.approx(10.0)


def test_round2_eyepiece_swap_changes_visual_magnification_only():
    # Swapping the eyepiece must not disturb the objective-side checks.
    rnd = get_round(2)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, focal_length_mm=12.5) if e.name == "eyepiece" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    assert _named(report, "Visual magnification").status is Status.FAIL
    assert _named(report, "Optical tube length").status is Status.PASS
    assert _named(report, "Magnification").status is Status.PASS


def test_image_plane_search_bound_excludes_downstream_elements():
    # The bug M3 surfaced: asking where the intermediate image lands traced
    # straight through the eyepiece behind it and reported "collimated".
    rnd = get_round(2)
    bench = rnd.reference_build()
    system = bench.to_paraxial()
    s_obj = bench.get("objective").s - 17.6
    s_image = bench.get("intermediate_image").s

    bounded = system.image_plane(s_obj, search_to=s_image + 1.0)
    assert bounded == pytest.approx(s_image, abs=0.01)
    # Unbounded, the eyepiece collimates the bundle and there is no finite image.
    assert system.image_plane(s_obj) is None
