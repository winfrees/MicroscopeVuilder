"""M5 gate: rounds 3-5, the illumination arc.

The arc has to work as a sequence, not just as three isolated puzzles: the same
stand carries forward, one knob moves, and each round's failure mode is the next
round's subject.
"""

import dataclasses

import pytest

from microscopevuilder.bench.probe import find_planes, read_card
from microscopevuilder.game.rounds import (
    APERTURE_DIAPHRAGM_S,
    COLLECTOR_F,
    COLLECTOR_KOHLER_S,
    CONDENSER_F,
    SPECIMEN_S,
    collector_position_for,
    get_round,
)
from microscopevuilder.rules.base import Status
from microscopevuilder.rules.checks import condenser_na


def _named(report, name):
    return next(r for r in report.results if r.name == name)


# --- the collector constraint ------------------------------------------------


def test_a_collector_cannot_image_across_less_than_four_focal_lengths():
    # Not a solver limitation: 4f is the minimum lamp-to-image distance for any
    # single lens. A 30 mm collector genuinely cannot reach the 75 mm that
    # critical illumination needs on this stand.
    with pytest.raises(ValueError, match="at least 4f"):
        collector_position_for(75.0, f=30.0)

    assert collector_position_for(4 * 18.0, f=18.0) == pytest.approx(36.0)


def test_one_collector_reaches_both_illumination_targets():
    # The design constraint that makes rounds 3 and 4 a single-knob puzzle.
    critical = collector_position_for(75.0, COLLECTOR_F)
    kohler = collector_position_for(APERTURE_DIAPHRAGM_S, COLLECTOR_F)
    assert critical == pytest.approx(30.0, abs=0.01)
    assert kohler == pytest.approx(COLLECTOR_KOHLER_S, abs=0.01)


# --- round 3: critical illumination ------------------------------------------


def test_round3_passes_but_warns_that_the_filament_is_visible():
    # Critical illumination is correct, not a mistake. The round passes; the
    # warning is the observation round 4 removes.
    rnd = get_round(3)
    report = rnd.grade(rnd.reference_build())
    assert report.passed
    uniformity = _named(report, "Field uniformity")
    assert uniformity.status is Status.WARN
    assert "critical illumination" in uniformity.summary


def test_round3_puts_the_lamp_image_exactly_on_the_specimen():
    rnd = get_round(3)
    bench = rnd.reference_build()
    system = bench.to_paraxial()
    assert system.image_plane(0.0, search_to=SPECIMEN_S + 1) == pytest.approx(SPECIMEN_S, abs=0.01)


def test_round3_has_two_focusing_positions_but_only_one_gathers_enough_light():
    # c^2 - Dc + fD = 0 has two roots, the near and far conjugate pair, and BOTH
    # focus the lamp onto the specimen. Throughput is what separates them:
    # irradiance goes as the square of the collected NA, so the far position is
    # less than half as bright. This is what makes round 3 well-posed.
    rnd = get_round(3)
    assert 30.0 * 45.0 == pytest.approx(COLLECTOR_F * 75.0)

    for position in (30.0, 45.0):
        bench = rnd.reference_build()
        bench.move("collector", position)
        assert _named(rnd.grade(bench), "Lamp on specimen").status is Status.PASS

    near = rnd.reference_build()
    near.move("collector", 30.0)
    far = rnd.reference_build()
    far.move("collector", 45.0)
    assert rnd.grade(near).passed
    assert not rnd.grade(far).passed
    assert _named(rnd.grade(far), "Throughput").status is Status.FAIL


def test_round3_focus_alone_is_a_weak_constraint():
    # Documenting the physics that forced the throughput rule: the condenser
    # demagnifies the lamp image hard, so the lamp-to-specimen conjugate barely
    # moves across a wide range of collector positions.
    rnd = get_round(3)
    for position in (28.0, 34.0, 38.0, 42.0, 50.0):
        bench = rnd.reference_build()
        bench.move("collector", position)
        landed = bench.to_paraxial().image_plane(0.0, search_to=SPECIMEN_S + 1)
        assert abs(landed - SPECIMEN_S) < 1.0


# --- round 4: Köhler ---------------------------------------------------------


def test_round4_reference_satisfies_all_four_conjugacies():
    rnd = get_round(4)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()
    for name in (
        "Lamp on aperture diaphragm",
        "Field diaphragm on specimen",
        "Aperture diaphragm on back focal plane",
        "Filament off the specimen",
    ):
        assert _named(report, name).status is Status.PASS


def test_round4_back_focal_plane_check_is_not_satisfied_by_the_objective_position():
    # The BFP sits one focal length behind the objective. A check that compared
    # against the objective itself would pass a build a whole focal length out --
    # which is the entire quantity in question.
    rnd = get_round(4)
    bench = rnd.reference_build()
    objective = bench.get("objective")
    row = _named(rnd.grade(bench), "Aperture diaphragm on back focal plane")
    assert row.target == pytest.approx(objective.s + objective.focal_length_mm)
    assert row.target != pytest.approx(objective.s)


def test_round4_fails_if_the_collector_is_left_at_the_critical_position():
    # The single-knob story: round 3's correct answer is round 4's failure.
    rnd4 = get_round(4)
    bench = rnd4.reference_build()
    bench.move("collector", collector_position_for(75.0, COLLECTOR_F))
    report = rnd4.grade(bench)
    assert not report.passed
    assert _named(report, "Lamp on aperture diaphragm").status is Status.FAIL
    assert _named(report, "Filament off the specimen").status is Status.FAIL


def test_round4_collimated_bundle_is_described_as_collimated_not_as_a_huge_number():
    # An almost-collimated bundle solves to a conjugate hundreds of metres away.
    # Reporting that number reads as a bug; it must say "collimated".
    rnd = get_round(4)
    row = _named(rnd.grade(rnd.reference_build()), "Filament off the specimen")
    assert "collimated" in row.summary
    assert "176" not in row.summary


def test_round4_moving_the_field_diaphragm_breaks_only_the_field_set():
    # The two conjugate sets are independent, which is the structural point of
    # Köhler and the reason the ribbon has two rows.
    rnd = get_round(4)
    bench = rnd.reference_build()
    # The condenser demagnifies this conjugate hard -- moving the field diaphragm
    # 10 mm shifts its image by under 1 mm -- so the field diaphragm plane is
    # genuinely insensitive and it takes a large move to break it.
    bench.move("field_diaphragm", 40.0)
    report = rnd.grade(bench)
    assert _named(report, "Field diaphragm on specimen").status is Status.FAIL
    assert _named(report, "Lamp on aperture diaphragm").status is Status.PASS
    assert _named(report, "Aperture diaphragm on back focal plane").status is Status.PASS


def test_kohler_field_and_aperture_planes_interleave_on_the_card():
    # What the conjugate ribbon is supposed to make obvious, checked numerically:
    # the aperture diaphragm is a pupil, the specimen is a field plane, and a card
    # at each shows something different.
    rnd = get_round(4)
    bench = rnd.reference_build()

    at_specimen = read_card(bench, SPECIMEN_S, SPECIMEN_S + 0.0)
    at_image = read_card(bench, SPECIMEN_S, bench.get("intermediate_image").s)
    assert at_image.is_image_plane

    found = find_planes(bench, SPECIMEN_S, SPECIMEN_S, bench.extent() + 10.0)
    for image_s in found["image"]:
        for pupil_s in found["pupil"]:
            assert abs(image_s - pupil_s) > 1.0


# --- round 5: resolution -----------------------------------------------------


def test_round5_reference_resolves_the_target():
    rnd = get_round(5)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()
    assert _named(report, "Resolution").measured == pytest.approx(0.455, abs=0.005)


def test_round5_closing_the_condenser_fails_resolution_and_says_why():
    # The trap, as a test: a closed condenser still forms a perfectly focused,
    # higher-contrast image -- and cannot resolve the target.
    rnd = get_round(5)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, semi_diameter_mm=0.5) if e.name == "aperture_diaphragm" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    assert not report.passed

    resolution = _named(report, "Resolution")
    assert resolution.status is Status.FAIL
    assert "NA_cond" in resolution.equation
    assert "factor of two" in resolution.remedy
    assert _named(report, "Condenser match").status is Status.FAIL


def test_round5_condenser_na_follows_the_diaphragm_radius():
    rnd = get_round(5)
    bench = rnd.reference_build()
    assert condenser_na(bench, "aperture_diaphragm", "condenser") == pytest.approx(0.55)
    # NA = r / f, so halving the radius halves the illumination NA.
    bench.elements = [
        dataclasses.replace(e, semi_diameter_mm=0.275 * CONDENSER_F)
        if e.name == "aperture_diaphragm" else e
        for e in bench.elements
    ]
    assert condenser_na(bench, "aperture_diaphragm", "condenser") == pytest.approx(0.275)


def test_round5_over_opening_the_condenser_warns_rather_than_failing():
    # Over-open costs contrast and adds glare, but it does not cost resolution,
    # so it must not be graded as a failure.
    rnd = get_round(5)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, semi_diameter_mm=0.9 * CONDENSER_F)
        if e.name == "aperture_diaphragm" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    match = _named(report, "Condenser match")
    assert match.status is Status.WARN
    assert report.passed


def test_the_illumination_rounds_share_one_stand():
    # Continuity: a player carries the build forward rather than starting over.
    builds = {n: get_round(n).reference_build() for n in (3, 4, 5)}
    for name in ("lamp", "collector", "field_diaphragm", "aperture_diaphragm", "condenser", "specimen"):
        assert all(b.has(name) for b in builds.values())
    # Only the collector moves between rounds 3 and 4.
    moved = [
        e.name for e in builds[3].elements
        if builds[4].has(e.name) and builds[4].get(e.name).s != e.s
    ]
    assert moved == ["collector"]
