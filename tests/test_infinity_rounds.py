"""M8 gate: infinity correction, the parfocal turret, and the sandbox."""

import pytest

from microscopevuilder.bench.bench import Bench, BenchElement
from microscopevuilder.game.rounds import (
    CFI_PARFOCAL,
    CFI_TUBE_F,
    FILTER_INDEX,
    FILTER_THICKNESS_MM,
    INF_SPECIMEN_S,
    TURRET,
    get_round,
)
from microscopevuilder.optics.paraxial import focus_shift_from_plate
from microscopevuilder.rules.base import Status


def _named(report, name):
    return next(r for r in report.results if r.name == name)


# --- the plate, which is what makes round 11 real ----------------------------


def test_a_plate_shifts_focus_in_converging_light_by_the_textbook_amount():
    from microscopevuilder.optics.paraxial import ParaxialSystem, glass_plate, thin_lens

    bare = ParaxialSystem([thin_lens("L", 0.0, 50.0, 12.5)])
    plated = ParaxialSystem(
        [thin_lens("L", 0.0, 50.0, 12.5), glass_plate("F", 30.0, 5.0, 1.52, 12.5)]
    )
    shift = plated.image_plane(-150.0) - bare.image_plane(-150.0)
    assert shift == pytest.approx(focus_shift_from_plate(5.0, 1.52))
    assert shift == pytest.approx(5.0 * (1 - 1 / 1.52))


def test_a_plate_does_nothing_at_all_in_collimated_light():
    # A ray (y, 0) is unchanged by any transfer, so the plate cannot move the image.
    # This is the entire argument for infinity optics, and it must be exact.
    rnd = get_round(11)
    bench = rnd.reference_build()
    without = Bench([e for e in bench.elements if e.name != "filter"])
    assert bench.to_paraxial().image_plane(INF_SPECIMEN_S, search_to=400.0) == pytest.approx(
        without.to_paraxial().image_plane(INF_SPECIMEN_S, search_to=400.0), abs=1e-12
    )


def test_the_same_filter_in_converging_light_would_break_the_round():
    # Moving the filter past the tube lens puts it in converging light, where it
    # shifts focus by exactly t(1-1/n) -- and the round must catch that.
    rnd = get_round(11)
    bench = rnd.reference_build()
    bench.move("filter", bench.get("tube_lens").s + 40.0)
    row = _named(rnd.grade(bench), "Filter neutrality")
    assert row.status is Status.FAIL
    assert row.measured == pytest.approx(focus_shift_from_plate(FILTER_THICKNESS_MM, FILTER_INDEX), rel=0.01)
    assert "infinity space" in row.remedy


def test_a_plate_with_an_impossible_index_is_rejected():
    from microscopevuilder.optics.paraxial import glass_plate

    with pytest.raises(ValueError):
        glass_plate("F", 0.0, 5.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        glass_plate("F", 0.0, -1.0, 1.5, 10.0)


# --- round 11 ----------------------------------------------------------------


def test_round11_reference_passes():
    rnd = get_round(11)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()


def test_round11_magnification_is_independent_of_the_infinity_space_length():
    # The defining property: M = f_tube / f_obj, whatever the separation.
    rnd = get_round(11)
    for tube_lens_s in (120.0, 180.0, 260.0):
        bench = rnd.reference_build()
        bench.move("tube_lens", tube_lens_s)
        bench.move("sensor", tube_lens_s + CFI_TUBE_F)
        report = rnd.grade(bench)
        assert _named(report, "Magnification").status is Status.PASS, tube_lens_s
        assert _named(report, "Focus").status is Status.PASS, tube_lens_s


def test_round11_specimen_off_the_front_focal_plane_breaks_the_infinity_space():
    rnd = get_round(11)
    bench = rnd.reference_build()
    bench.move("objective", bench.get("objective").s + 2.0)
    row = _named(rnd.grade(bench), "Infinity space")
    assert row.status is Status.FAIL
    assert "front focal plane" in row.remedy


def test_round11_magnification_follows_the_cfi60_relation():
    from microscopevuilder.game.rounds import _infinity_stand

    for magnification in (10.0, 20.0, 40.0):
        bench = _infinity_stand(magnification=magnification)
        f_obj = bench.get("objective").focal_length_mm
        assert f_obj == pytest.approx(CFI_TUBE_F / magnification)
        measured = abs(
            bench.to_paraxial().magnification(INF_SPECIMEN_S, bench.get("sensor").s)
        )
        assert measured == pytest.approx(magnification, rel=1e-6)


# --- round 12: the turret ----------------------------------------------------


def test_round12_reference_passes():
    rnd = get_round(12)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()


def test_round12_grades_every_objective_in_the_turret():
    rnd = get_round(12)
    report = rnd.grade(rnd.reference_build())
    for _, magnification, _ in TURRET:
        assert _named(report, f"Focus [{magnification:.0f}x]").status is Status.PASS
        assert _named(report, f"Magnification [{magnification:.0f}x]").status is Status.PASS


def test_parfocality_means_a_common_shoulder_height_not_a_common_lens_height():
    # The objectives sit at visibly different heights above the specimen; what they
    # share is where the mount shoulder lands.
    rnd = get_round(12)
    bench = rnd.reference_build()
    heights = {n: bench.get(n).s for n, _, _ in TURRET}
    assert len(set(heights.values())) == len(TURRET)  # all different

    row = _named(rnd.grade(bench), "Parfocality")
    assert row.status is Status.PASS
    assert row.measured == pytest.approx(0.0, abs=0.01)
    assert f"{CFI_PARFOCAL:.0f}" in row.equation


def test_round12_an_objective_from_another_series_breaks_parfocality():
    # A 45 mm parfocal objective in a 60 mm turret: it focuses, but not with the
    # others, which is exactly the failure a mixed turret produces on a bench.
    rnd = get_round(12)
    bench = rnd.reference_build()
    bench.move("objective_40x", bench.get("objective_40x").s + 15.0)
    report = rnd.grade(bench)
    assert _named(report, "Parfocality").status is Status.FAIL
    assert "loses focus" in _named(report, "Parfocality").summary


def test_round12_sensor_stays_put_across_the_whole_turret():
    # The practical promise: you do not move the camera when you change objective.
    from microscopevuilder.game.rounds import _stand_with_objective

    rnd = get_round(12)
    bench = rnd.reference_build()
    sensor_s = bench.get("sensor").s
    for name, _, _ in TURRET:
        single = _stand_with_objective(bench, name)
        landed = single.to_paraxial().image_plane(INF_SPECIMEN_S, search_to=sensor_s + 1)
        assert landed == pytest.approx(sensor_s, abs=0.01), name


# --- sandbox -----------------------------------------------------------------


def test_sandbox_reports_without_grading():
    rnd = get_round(0)
    report = rnd.grade(rnd.reference_build())
    assert report.results
    assert report.passed  # nothing can fail
    assert all(r.status is Status.NOT_APPLICABLE for r in report.results)


def test_sandbox_describes_an_afocal_build_rather_than_erroring():
    rnd = get_round(0)
    bench = Bench([BenchElement("lens", 10.0, "lens", 12.0, 25.0)])
    report = rnd.grade(bench)
    assert _named(report, "Image plane").summary


def test_a_turret_traces_only_the_selected_objective():
    # Without this a three-objective turret traces as three objectives in series,
    # which is not a microscope -- the live model raised on the nonsense result.
    rnd = get_round(12)
    bench = rnd.reference_build()
    assert len([e for e in bench.elements if e.metadata.get("in_turret")]) == 3
    assert len([e.name for e in bench.to_paraxial().elements if "objective" in e.name]) == 1

    bench.select_turret("objective_40x")
    traced = [e.name for e in bench.to_paraxial().elements if "objective" in e.name]
    assert traced == ["objective_40x"]


def test_disabled_elements_survive_a_round_trip(tmp_path):
    rnd = get_round(12)
    original = rnd.reference_build()
    path = tmp_path / "turret.json"
    original.save(path)
    loaded = Bench.load(path)
    assert loaded.to_dict() == original.to_dict()
    assert sum(1 for e in loaded.elements if e.metadata.get("in_turret") and e.enabled) == 1
