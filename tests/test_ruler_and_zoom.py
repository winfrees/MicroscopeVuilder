"""Placement aids: the ruler, snapping, zoom, and the tolerance policy.

These exist because of a measurement. On round 11 the depth of focus is 0.19 mm
while the bench spans 380 mm, so fitted to a 760-pixel view one pixel of drag moves
a component 0.50 mm -- two and a half times the tolerance. Placement was not hard,
it was impossible.
"""

import pytest

from microscopevuilder.game.rounds import get_round
from microscopevuilder.rules.tolerances import (
    FORGIVING_FRACTION,
    TolerancePolicy,
    describe_policy,
    image_side_depth_of_focus_mm,
    position_tolerance_mm,
    set_tolerance_policy,
    tolerance_policy,
)
from microscopevuilder.ui.ruler import (
    MM_PER_INCH,
    choose_step,
    snap,
    snap_to_planes,
    ticks,
    to_display,
)


@pytest.fixture
def strict():
    previous = set_tolerance_policy(TolerancePolicy.STRICT)
    yield
    set_tolerance_policy(previous)


# --- the problem being solved ------------------------------------------------


def test_one_pixel_of_drag_used_to_overshoot_the_tolerance():
    # The measurement that motivated all of this, kept as a test so nobody
    # "simplifies" the zoom away later.
    tolerance_mm = image_side_depth_of_focus_mm(0.5461, 0.75, 20.0)
    bench_span_mm = get_round(11).reference_build().extent()
    mm_per_pixel_fitted = bench_span_mm / 760

    assert tolerance_mm == pytest.approx(0.194, abs=0.005)
    assert mm_per_pixel_fitted > 2 * tolerance_mm  # unreachable by dragging

    # At 10x zoom the same pixel is well inside the tolerance.
    assert mm_per_pixel_fitted / 10 < tolerance_mm


# --- snapping ----------------------------------------------------------------


def test_snapping_to_a_grid_step():
    assert snap(193.43, 0.1) == pytest.approx(193.4)
    assert snap(193.43, 1.0) == pytest.approx(193.0)
    assert snap(7.0, 0.0) == 7.0  # a zero step is a no-op, not a crash


def test_a_one_millimetre_grid_would_step_over_most_round_targets():
    # Round targets land on tenths (17.6, 193.6, 172.6), so the default snap has
    # to be finer than a millimetre or snapping would prevent passing.
    for target in (17.6, 193.6, 172.6):
        assert snap(target, 1.0) != pytest.approx(target)
        assert snap(target, 0.1) == pytest.approx(target)


def test_plane_snapping_captures_within_range_and_leaves_alone_outside():
    planes = [380.0, 193.6, 20.0]
    assert snap_to_planes(378.6, planes, capture_mm=2.0) == (380.0, True)
    assert snap_to_planes(370.0, planes, capture_mm=2.0) == (370.0, False)
    assert snap_to_planes(5.0, [], capture_mm=2.0) == (5.0, False)


def test_plane_snapping_picks_the_nearest_plane():
    assert snap_to_planes(196.0, [193.6, 200.0], capture_mm=5.0)[0] == pytest.approx(193.6)
    assert snap_to_planes(198.0, [193.6, 200.0], capture_mm=5.0)[0] == pytest.approx(200.0)


# --- ruler -------------------------------------------------------------------


def test_tick_spacing_follows_a_one_two_five_progression():
    assert choose_step(380.0) == 20.0
    assert choose_step(20.0) == 1.0
    assert choose_step(2.0) == 0.1
    assert choose_step(0.0) == 1.0  # degenerate span does not divide by zero


def test_zooming_in_reveals_finer_divisions():
    # The ruler should get more useful as you zoom, not just bigger.
    assert choose_step(380.0) > choose_step(38.0) > choose_step(3.8)


def test_ticks_label_only_the_major_ones():
    out = ticks(0.0, 50.0, 10.0, "mm", major_every=5)
    labelled = [t for t in out if t.label]
    assert [t.position_mm for t in labelled] == [0.0, 50.0]
    assert all(t.is_major == bool(t.label) for t in out)


def test_ticks_refuse_to_generate_an_unusable_number():
    # A tiny step across a long bench would otherwise make tens of thousands of
    # scene items and lock the UI.
    assert ticks(0.0, 1000.0, 0.001, "mm") == []


def test_ticks_handle_a_reversed_or_empty_span():
    assert ticks(50.0, 10.0, 1.0) == []
    assert ticks(0.0, 10.0, 0.0) == []


def test_imperial_display_converts_and_marks_the_unit():
    assert to_display(MM_PER_INCH, "in") == '1.0000"'
    assert to_display(380.0, "mm") == "380.000 mm"


# --- tolerance policy --------------------------------------------------------


def test_forgiving_is_the_default_because_strict_was_unusable_by_hand():
    assert tolerance_policy() is TolerancePolicy.FORGIVING


def test_the_practice_tolerance_is_five_percent_of_the_position(strict):
    set_tolerance_policy(TolerancePolicy.FORGIVING)
    assert position_tolerance_mm(380.0, 0.194) == pytest.approx(380.0 * FORGIVING_FRACTION)


def test_relaxing_a_tolerance_never_tightens_one():
    # Near the origin 5% is a fraction of a millimetre, which would be stricter
    # than the physics. The policy takes the larger of the two.
    assert position_tolerance_mm(2.0, 0.5, TolerancePolicy.FORGIVING) == 0.5
    assert position_tolerance_mm(0.0, 0.3, TolerancePolicy.FORGIVING) == 0.3


def test_strict_returns_the_derived_tolerance_untouched(strict):
    assert position_tolerance_mm(380.0, 0.194, TolerancePolicy.STRICT) == 0.194


def test_the_scorecard_always_names_the_physical_tolerance():
    # Grading may be forgiving; the number a student takes away must not be.
    forgiving = describe_policy(380.0, 0.194, TolerancePolicy.FORGIVING)
    assert "5%" in forgiving
    assert "physical tolerance is 0.194 mm" in forgiving

    strict_text = describe_policy(380.0, 0.194, TolerancePolicy.STRICT)
    assert "physical tolerance of 0.194 mm" in strict_text
    assert "5%" not in strict_text


def test_a_build_five_millimetres_out_passes_practice_and_fails_strict():
    rnd = get_round(11)
    bench = rnd.reference_build()
    bench.move("sensor", bench.get("sensor").s - 5.0)

    set_tolerance_policy(TolerancePolicy.FORGIVING)
    assert rnd.grade(bench).passed

    previous = set_tolerance_policy(TolerancePolicy.STRICT)
    try:
        report = rnd.grade(bench)
        assert not report.passed
        focus = next(r for r in report.results if r.name == "Focus")
        assert "physical tolerance of 0.194 mm" in focus.equation
    finally:
        set_tolerance_policy(previous)


def test_every_round_still_passes_its_reference_build_under_both_policies():
    from microscopevuilder.game.rounds import ROUNDS

    for policy in (TolerancePolicy.STRICT, TolerancePolicy.FORGIVING):
        previous = set_tolerance_policy(policy)
        try:
            for number in sorted(n for n in ROUNDS if n > 0):
                rnd = get_round(number)
                bench = rnd.reference_build()
                assert rnd.grade(bench).passed, f"round {number} under {policy}"
        finally:
            set_tolerance_policy(previous)
