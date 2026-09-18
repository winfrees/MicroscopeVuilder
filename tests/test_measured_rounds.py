"""M11 gate: rounds 6-8, graded on measurements taken from the rendered image.

Each round has to fail for its own reason when the wrong component is fitted --
a colour round that fails on field curvature is not teaching colour.
"""

import pytest

from microscopevuilder.game.rounds import (
    ROUND7_TEST_PERIOD_UM,
    _component_stand,
    get_round,
)
from microscopevuilder.rules.base import Status
from microscopevuilder.rules.measured import (
    check_chromatic_focus,
    check_field_flatness,
    check_sensor_matches_the_optics,
)
from microscopevuilder.rules.tolerances import depth_of_focus_mm

LAMBDA = 0.5461


def _named(report, name):
    return next(r for r in report.results if r.name == name)


# --- the two-clock split -----------------------------------------------------


@pytest.mark.parametrize("number", [6, 7, 8])
def test_measured_rounds_declare_a_measured_pass(number):
    assert get_round(number).has_measured_rules


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 11, 12])
def test_geometry_only_rounds_stay_on_the_live_clock(number):
    # Rendering costs tenths of a second. A round that does not need it must not
    # pay for it on every drag.
    rnd = get_round(number)
    assert not rnd.has_measured_rules
    assert rnd.measure_image(rnd.reference_build()).results == []


@pytest.mark.parametrize("number", [6, 7, 8])
def test_reference_builds_pass_both_passes(number):
    rnd = get_round(number)
    bench = rnd.reference_build()
    assert rnd.grade(bench).passed, "\n" + rnd.grade(bench).format()
    measured = rnd.measure_image(bench)
    assert measured.passed, "\n" + measured.format()


# --- round 6: colour ---------------------------------------------------------


def test_round6_an_achromat_fails_on_chromatic_focus():
    # An achromat's secondary spectrum is f/2000, which at a 10 mm focal length is
    # 5 um -- more than the depth of focus can absorb at any useful NA.
    achromat = _component_stand("cfi_plan_achro_20x", na=0.40, grade="plan_achromat")
    row = check_chromatic_focus(achromat, 0.0, "sensor")
    assert row.status is Status.FAIL
    assert "f/2000" in row.remedy
    assert row.measured > row.target


def test_round6_an_apochromat_keeps_the_band_inside_the_depth_of_focus():
    apo = _component_stand("cfi_plan_apo_20x", na=0.75, grade="plan_apochromat")
    row = check_chromatic_focus(apo, 0.0, "sensor")
    assert row.status is Status.PASS
    assert row.measured <= row.target


def test_round6_criterion_is_the_depth_of_focus_not_a_chosen_number():
    # "Colour corrected" means one focus setting serves the whole band, so the
    # target must be the depth of focus at the build's own NA.
    apo = _component_stand("cfi_plan_apo_20x", na=0.75, grade="plan_apochromat")
    row = check_chromatic_focus(apo, 0.0, "sensor")
    assert row.target == pytest.approx(depth_of_focus_mm(LAMBDA, 0.75) * 1000.0, rel=1e-6)


def test_round6_fails_the_round_with_the_wrong_objective():
    rnd = get_round(6)
    bench = _component_stand("cfi_plan_achro_20x", na=0.40, grade="plan_achromat")
    assert not rnd.measure_image(bench).passed


def test_chromatic_shift_scales_with_focal_length():
    # f/2000 means a 4x suffers ten times the shift of a 40x. Two different
    # objectives of the same grade must therefore differ by that ratio.
    long_focus = check_chromatic_focus(
        _component_stand("cfi_plan_achro_4x", na=0.10, magnification=4.0, grade="plan_achromat"),
        0.0, "sensor",
    )
    short_focus = check_chromatic_focus(
        _component_stand("cfi_plan_achro_40x", na=0.10, magnification=40.0, grade="plan_achromat"),
        0.0, "sensor",
    )
    assert long_focus.measured / short_focus.measured == pytest.approx(10.0, rel=0.02)


# --- round 7: flat field -----------------------------------------------------


def test_round7_a_non_plan_objective_loses_the_corners():
    non_plan = _component_stand("cfi_achro_20x", na=0.40, grade="achromat")
    row = check_field_flatness(non_plan, 0.0, "sensor", ROUND7_TEST_PERIOD_UM)
    assert row.status is Status.FAIL
    assert row.measured < 0.1
    assert "Petzval" in row.remedy


def test_round7_a_plan_objective_holds_them():
    plan = _component_stand("cfi_plan_achro_20x", na=0.40, grade="plan_achromat")
    row = check_field_flatness(plan, 0.0, "sensor", ROUND7_TEST_PERIOD_UM)
    assert row.status is Status.PASS
    assert row.measured > 0.9


def test_round7_refuses_to_measure_flatness_near_the_cutoff():
    # Measured directly: at a 1.0 um period against a 0.85 um cutoff, a flat plan
    # objective read 0.57 and a badly curved one read 0.91. The ratio is noise
    # there, so the rule declines rather than answering wrongly.
    plan = _component_stand("cfi_plan_achro_20x", na=0.40, grade="plan_achromat")
    row = check_field_flatness(plan, 0.0, "sensor", 0.6)
    assert row.status is Status.NOT_APPLICABLE
    assert "cutoff" in row.summary
    assert "coarser period" in row.remedy


def test_round7_test_period_sits_well_inside_the_cutoff():
    from microscopevuilder.optics.psf import mtf_cutoff_cycles_per_um

    cutoff_period = 1.0 / mtf_cutoff_cycles_per_um(LAMBDA, 0.40)
    assert ROUND7_TEST_PERIOD_UM > 3 * cutoff_period


def test_round7_fails_the_round_with_a_non_plan_objective():
    rnd = get_round(7)
    bench = _component_stand("cfi_achro_20x", na=0.40, grade="achromat")
    assert not rnd.measure_image(bench).passed


# --- round 8: camera port ----------------------------------------------------


def test_round8_a_bare_20x_undersamples_a_common_scmos_pixel():
    # The trap: a 20x/0.75 resolves 0.444 um, 8.9 um at the sensor, so Nyquist
    # wants 4.44 um pixels. The very common 6.5 um pixel does not sample it.
    bare = _component_stand("cfi_plan_apo_20x", na=0.75, grade="plan_apochromat")
    row = check_sensor_matches_the_optics(bare, 0.0, "sensor", pixel_um=6.5)
    assert row.status is Status.FAIL
    assert "undersampled" in row.summary


def test_round8_a_tube_lens_changer_fixes_it():
    fixed = _component_stand(
        "cfi_plan_apo_20x", na=0.75, grade="plan_apochromat", extra_magnification=1.5
    )
    row = check_sensor_matches_the_optics(fixed, 0.0, "sensor", pixel_um=6.5)
    assert row.status is Status.PASS


def test_round8_too_much_magnification_is_flagged_as_empty():
    # The opposite failure, and the reason the round is not simply "add more
    # magnification": extra magnification costs field and light for no detail.
    over = _component_stand(
        "cfi_plan_apo_20x", na=0.75, grade="plan_apochromat", extra_magnification=5.0
    )
    row = check_sensor_matches_the_optics(over, 0.0, "sensor", pixel_um=6.5)
    assert row.status is Status.WARN
    assert "empty magnification" in row.summary
    assert "costs field and light" in row.remedy


def test_round8_empty_magnification_really_does_cost_light():
    # The claim in the remedy, checked rather than asserted: irradiance goes as
    # 1/M^2, so a 5x changer is 25 times dimmer for the same detail.
    from microscopevuilder.imaging.build_optics import resolve_build_optics

    matched = resolve_build_optics(
        _component_stand("cfi_plan_apo_20x", na=0.75, extra_magnification=1.5),
        0.0, "sensor", LAMBDA,
    )
    over = resolve_build_optics(
        _component_stand("cfi_plan_apo_20x", na=0.75, extra_magnification=5.0),
        0.0, "sensor", LAMBDA,
    )
    assert matched.relative_irradiance / over.relative_irradiance == pytest.approx(
        (5.0 / 1.5) ** 2, rel=0.01
    )


def test_round8_geometry_and_measured_passes_agree():
    # The live Nyquist check and the measured sensor check must not contradict
    # each other: one is the prediction, the other the confirmation.
    rnd = get_round(8)
    bench = rnd.reference_build()
    assert _named(rnd.grade(bench), "Sampling").status is Status.PASS
    assert _named(rnd.measure_image(bench), "Sensor match").status is Status.PASS


# --- the catalog gained non-plan entries for this -----------------------------


def test_non_plan_objectives_exist_and_are_far_worse_than_plan_ones():
    from microscopevuilder.bench.catalog import load_catalog

    catalog = load_catalog()
    plan = [o for o in catalog.objectives.values() if o.grade.startswith("plan_")]
    non_plan = [o for o in catalog.objectives.values() if not o.grade.startswith("plan_")]
    assert non_plan, "round 7 needs something to contrast a plan objective against"

    worst_plan = max(o.aberrations.field_curvature_sag_um for o in plan)
    best_non_plan = min(o.aberrations.field_curvature_sag_um for o in non_plan)
    assert best_non_plan > 10 * worst_plan


def test_non_plan_sag_exceeds_the_depth_of_focus_by_a_wide_margin():
    # What makes the corners unrecoverable: you cannot focus both ends at once.
    from microscopevuilder.bench.catalog import load_catalog

    for objective in load_catalog().objectives.values():
        if objective.grade.startswith("plan_"):
            continue
        dof_um = depth_of_focus_mm(LAMBDA, objective.na) * 1000.0
        assert objective.aberrations.field_curvature_sag_um > 5 * dof_um, objective.key
