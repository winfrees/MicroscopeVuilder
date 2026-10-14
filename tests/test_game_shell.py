"""M12 gate: budget, scoring, the diff, progress and the specimen library."""

import dataclasses
import json

import pytest

from microscopevuilder.game.progress import Progress, RoundRecord
from microscopevuilder.game.rounds import PREREQUISITES, ROUNDS, get_round
from microscopevuilder.game.scoring import (
    billable_parts,
    check_parts_budget,
    diff_against_reference,
    score_round,
)
from microscopevuilder.imaging.specimens import SPECIMEN_LIBRARY
from microscopevuilder.rules.base import RuleReport, RuleResult, Status


def _report(*statuses) -> RuleReport:
    report = RuleReport()
    for index, status in enumerate(statuses):
        report.add(RuleResult(f"check {index}", status, "summary"))
    return report


# --- the parts budget --------------------------------------------------------


def test_probes_and_free_components_do_not_count_against_the_budget():
    # A white card is a measuring instrument, not a part of the microscope; making
    # it cost a slot would penalise the player for looking.
    rnd = get_round(11)
    bench = rnd.reference_build()
    before = len(billable_parts(bench))
    bench.add_card(100.0)
    assert len(billable_parts(bench)) == before


def test_going_over_budget_fails_and_names_what_was_counted():
    rnd = get_round(11)
    bench = rnd.reference_build()
    row = check_parts_budget(bench, budget=2)
    assert row.status is Status.FAIL
    assert "over" in row.summary
    assert "objective" in row.equation


@pytest.mark.parametrize("number", sorted(n for n in ROUNDS if n > 0))
def test_every_reference_build_fits_its_own_budget(number):
    # A round whose own solution breaks the budget is unwinnable.
    rnd = get_round(number)
    assert check_parts_budget(rnd.reference_build(), rnd.parts_budget).status is Status.PASS


# --- scoring -----------------------------------------------------------------


def test_a_failure_scores_nothing_and_says_why():
    rnd = get_round(11)
    score = score_round(rnd.reference_build(), 5, _report(Status.PASS, Status.FAIL))
    assert score.stars == 0
    assert not score.passed
    assert score.reasons


def test_a_clean_solve_inside_budget_scores_three():
    rnd = get_round(11)
    score = score_round(rnd.reference_build(), 5, _report(Status.PASS, Status.PASS))
    assert score.stars == 3
    assert "***" in score.describe()


def test_a_warning_costs_a_star_but_still_passes():
    # The distinction the second star marks: it worked, and a demonstrator would
    # still raise an eyebrow.
    rnd = get_round(11)
    score = score_round(rnd.reference_build(), 5, _report(Status.PASS, Status.WARN))
    assert score.stars == 2
    assert score.passed
    assert "warning" in score.describe()


def test_going_over_budget_costs_a_star():
    rnd = get_round(11)
    score = score_round(rnd.reference_build(), budget=1, live=_report(Status.PASS))
    assert score.stars == 2
    assert "budget" in score.describe()


def test_measured_failures_count_toward_the_score():
    rnd = get_round(13)
    bench = rnd.reference_build()
    score = score_round(bench, rnd.parts_budget, _report(Status.PASS), _report(Status.FAIL))
    assert score.stars == 0


# --- the diff ----------------------------------------------------------------


def test_the_diff_of_an_identical_build_is_empty():
    rnd = get_round(11)
    diff = diff_against_reference(rnd.reference_build(), rnd.reference_build())
    assert diff.is_empty
    assert "identical" in diff.format()


def test_the_diff_reports_moved_missing_extra_and_changed():
    from microscopevuilder.bench.bench import BenchElement

    rnd = get_round(11)
    reference = rnd.reference_build()
    bench = rnd.reference_build()
    bench.move("sensor", bench.get("sensor").s + 2.0)
    bench.remove("filter")
    bench.add(BenchElement("spare", 250.0, "lens", 10.0, 80.0))

    diff = diff_against_reference(bench, reference)
    assert diff.missing == ["filter"]
    assert diff.extra == ["spare"]
    assert diff.moved and diff.moved[0][0] == "sensor"
    text = diff.format()
    assert "+2.00" in text


def test_the_diff_ignores_probe_cards():
    rnd = get_round(11)
    bench = rnd.reference_build()
    bench.add_card(120.0)
    assert diff_against_reference(bench, rnd.reference_build()).is_empty


def test_the_diff_notices_a_swapped_objective():
    import dataclasses

    rnd = get_round(11)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, metadata={**e.metadata, "na": 0.25})
        if e.name == "objective" else e
        for e in bench.elements
    ]
    diff = diff_against_reference(bench, rnd.reference_build())
    assert any(field == "na" for _, field, _, _ in diff.changed)


# --- progress ----------------------------------------------------------------


def test_progress_round_trips_through_a_file(tmp_path):
    path = tmp_path / "progress.json"
    progress = Progress()
    progress.complete(1, 3, 2)
    progress.complete(2, 2, 3)
    progress.save(path)

    loaded = Progress.load(path)
    assert loaded.record(1).stars == 3
    assert loaded.record(2).stars == 2
    assert loaded.total_stars() == 5


def test_progress_never_goes_backwards():
    # Replaying a round badly must not erase a better earlier result.
    progress = Progress()
    progress.complete(1, 3, 2)
    progress.complete(1, 1, 5)
    assert progress.record(1).stars == 3
    assert progress.record(1).best_parts == 2
    assert progress.record(1).attempts == 2


def test_a_missing_save_starts_a_fresh_game(tmp_path):
    assert Progress.load(tmp_path / "absent.json").total_stars() == 0


def test_a_corrupt_save_does_not_stop_the_game_opening(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert Progress.load(path).total_stars() == 0


def test_a_save_from_a_newer_version_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"version": 99, "rounds": {}}), encoding="utf-8")
    # load() degrades to a fresh game; from_dict is explicit about why.
    assert Progress.load(path).total_stars() == 0
    with pytest.raises(ValueError, match="newer version"):
        Progress.from_dict({"version": 99, "rounds": {}})


def test_saving_is_atomic(tmp_path):
    # An interrupted save must not leave a truncated file where progress was.
    path = tmp_path / "progress.json"
    progress = Progress()
    progress.complete(1, 3, 2)
    progress.save(path)
    assert not (tmp_path / "progress.tmp").exists()
    assert json.loads(path.read_text())["rounds"]["1"]["stars"] == 3


# --- unlocking ---------------------------------------------------------------


def test_round_one_and_the_sandbox_are_always_open():
    progress = Progress()
    assert progress.is_unlocked(1, PREREQUISITES)
    assert progress.is_unlocked(0, PREREQUISITES)


def test_later_rounds_wait_on_their_prerequisite():
    progress = Progress()
    assert not progress.is_unlocked(2, PREREQUISITES)
    progress.complete(1, 3, 2)
    assert progress.is_unlocked(2, PREREQUISITES)


def test_contrast_techniques_are_gated_on_kohler_not_on_the_round_before():
    # A phase ring is meaningless to a player who does not yet own the back focal
    # plane, and round 4 is where that concept is earned.
    assert PREREQUISITES[13] == 4


def test_every_prerequisite_is_a_real_round():
    for number, required in PREREQUISITES.items():
        assert number in ROUNDS
        assert required in ROUNDS


def test_no_round_depends_on_itself_or_on_a_later_round():
    for number, required in PREREQUISITES.items():
        assert required < number


# --- specimen library --------------------------------------------------------


def test_the_library_covers_what_the_plan_promised():
    for name in (
        "USAF target", "Siemens star", "beads", "stained section",
        "unstained cell", "birefringent fibres", "polished metal", "Ronchi ruling",
    ):
        assert name in SPECIMEN_LIBRARY


@pytest.mark.parametrize("name", sorted(SPECIMEN_LIBRARY))
def test_every_specimen_renders_at_the_expected_size(name):
    import numpy as np

    from microscopevuilder.imaging.specimens import BirefringentField

    out = SPECIMEN_LIBRARY[name](128, 0.1)
    assert out.shape == (128, 128)
    if isinstance(out, BirefringentField):
        assert np.isfinite(out.retardance_waves).all()
    else:
        assert np.iscomplexobj(out)
        assert np.isfinite(out).all()


def test_the_usaf_target_carries_both_orientations():
    # Astigmatism and DIC shear resolve one direction and not the other, so a
    # single-orientation target would hide exactly what these rounds are about.
    import numpy as np

    from microscopevuilder.imaging.specimens import usaf_bars

    target = np.abs(usaf_bars(256, 0.05, period_um=1.0))
    rows = target.std(axis=1).max()
    columns = target.std(axis=0).max()
    assert rows > 0.05 and columns > 0.05


# --- M13: verification tooling -----------------------------------------------


def test_the_catalog_reports_what_still_needs_checking():
    from microscopevuilder.bench.catalog import load_catalog

    report = load_catalog().verification_report()
    assert "UNVERIFIED" in report
    assert "await a datasheet check" in report
    # The pedagogical budgets must never be presented as checkable facts.
    assert "never verified" in report


MINIMAL_CATALOG = """
[system]
name = "Test system"
tube_lens_focal_length_mm = 200.0
parfocal_distance_mm = 60.0
reference_wavelength_nm = 546.1
verified = false

[objectives.test_20x]
label = "Test 20x/0.75"
magnification = 20.0
na = 0.75
working_distance_mm = 1.0
immersion = "air"
grade = "plan_apochromat"
verified = {verified}
{citation}
[objectives.test_20x.aberrations]
source = "pedagogical"
spherical_rms_um = 0.010
astigmatism_rms_um = 0.009
field_curvature_sag_um = 0.40
chromatic_focus_fraction = 4.0e-5

[media.air]
index = 1.0
"""


def _write_catalog(tmp_path, verified: bool, citation: str = ""):
    path = tmp_path / "components.toml"
    path.write_text(
        MINIMAL_CATALOG.format(verified=str(verified).lower(), citation=citation),
        encoding="utf-8",
    )
    return path


def test_marking_an_entry_verified_requires_a_citation(tmp_path):
    # Without this guard, `verified = true` is just an assertion and the whole
    # provenance scheme means nothing.
    from microscopevuilder.bench.catalog import load_catalog

    path = _write_catalog(tmp_path, verified=True)
    load_catalog.cache_clear()
    try:
        with pytest.raises(ValueError, match="cite no source"):
            load_catalog(path)
    finally:
        load_catalog.cache_clear()


def test_a_verified_entry_with_a_citation_is_accepted(tmp_path):
    from microscopevuilder.bench.catalog import load_catalog

    path = _write_catalog(
        tmp_path,
        verified=True,
        citation='source = "example datasheet"\nchecked_on = "2026-01-01"',
    )
    load_catalog.cache_clear()
    try:
        catalog = load_catalog(path)
        assert catalog.objective("test_20x").verified
        assert "example datasheet" in catalog.verification_report()
    finally:
        load_catalog.cache_clear()


def test_an_unverified_entry_needs_no_citation(tmp_path):
    from microscopevuilder.bench.catalog import load_catalog

    path = _write_catalog(tmp_path, verified=False)
    load_catalog.cache_clear()
    try:
        assert load_catalog(path).unverified_entries()
    finally:
        load_catalog.cache_clear()


def test_the_shipped_catalog_is_honest_about_being_unverified():
    # If this ever starts failing because entries were verified, that is good news
    # -- but it must happen with citations attached, which the loader enforces.
    from microscopevuilder.bench.catalog import load_catalog

    assert load_catalog().unverified_entries()


# --- progression: carrying a build forward -----------------------------------


def test_no_round_opens_already_solved():
    """The defect this whole mechanism exists to fix.

    Every round used to load its own reference build and be graded on load, so
    opening the app awarded three stars on 14 of 15 rounds and saved them.
    """
    from microscopevuilder.game.scoring import check_parts_budget, score_round

    for number in sorted(n for n in ROUNDS if n):
        round_ = get_round(number)
        required = PREREQUISITES.get(number)
        carried = get_round(required).reference_build() if required else None
        bench = round_.starting_build(carried)

        report = round_.grade(bench)
        report.add(check_parts_budget(bench, round_.parts_budget))
        score = score_round(
            bench, round_.parts_budget, report, None,
            needs_measurement=round_.has_measured_rules,
        )
        assert not score.passed, f"round {number} opens already solved"


def test_a_measured_round_is_provisional_until_it_has_been_run():
    from microscopevuilder.game.scoring import check_parts_budget, score_round

    round_ = get_round(7)
    assert round_.has_measured_rules
    bench = round_.reference_build()
    report = round_.grade(bench)
    report.add(check_parts_budget(bench, round_.parts_budget))

    # Geometry alone: no verdict, because the rules that settle this round need an
    # image. A missing report used to read as "nothing objected", so it passed.
    held = score_round(bench, round_.parts_budget, report, None, needs_measurement=True)
    assert held.provisional and not held.passed and held.stars == 0

    settled = score_round(
        bench, round_.parts_budget, report, round_.measure_image(bench),
        needs_measurement=True,
    )
    assert settled.passed and not settled.provisional


def test_a_carried_build_keeps_the_players_positions():
    previous = get_round(1).reference_build()
    previous.move("lens", 123.0)          # solved, but not where the reference has it
    carried = get_round(2).starting_build(previous)
    assert carried.get("lens").s == 123.0


def test_what_a_round_does_not_use_is_parked_rather_than_removed():
    from microscopevuilder.game.scoring import billable_parts

    # Round 6 is about the camera path, so the illumination train it inherits from
    # round 5 is not in play -- but it is still the player's, and still on the bench.
    carried = get_round(5).reference_build()
    bench = get_round(6).starting_build(carried)

    assert bench.has("condenser"), "a carried component must not be deleted"
    assert bench.get("condenser").enabled is False, "it should be parked"
    # A part on the shelf costs no light and nothing to align, so it is not charged
    # for -- otherwise carrying a build forward would blow every later budget.
    assert "condenser" not in [e.name for e in billable_parts(bench)]
    assert len(billable_parts(bench)) <= get_round(6).parts_budget


def test_new_components_arrive_unplaced():
    from microscopevuilder.game.carry import is_in_tray

    bench = get_round(11).starting_build(get_round(8).reference_build())
    # Round 11 is additive: it introduces the filter and removes nothing.
    assert [e.name for e in bench.elements if is_in_tray(e)] == ["filter"]
    assert not any(not e.enabled for e in bench.elements), "nothing retired at round 11"
    assert bench.get("filter").s != get_round(11).reference_build().get("filter").s


def test_the_round_supplies_its_own_specimen():
    # Round 10 needs a FITC specimen and the cube to match. Carrying the previous
    # round's specimen forward is how it came to open already solved.
    bench = get_round(10).starting_build(get_round(9).reference_build())
    assert bench.get("specimen").metadata.get("fluorophore") == "fitc"


def test_an_unfitted_filter_cube_fails_rather_than_defaulting_to_the_answer():
    from microscopevuilder.rules.base import Status

    bench = get_round(10).reference_build()
    bench.elements = [
        dataclasses.replace(e, metadata={}) if e.name == "beamsplitter" else e
        for e in bench.elements
    ]
    report = get_round(10).grade(bench)
    cube_rows = [r for r in report.results if r.name == "Filter cube"]
    assert cube_rows and cube_rows[0].status is Status.FAIL


def test_replaying_a_round_keeps_the_better_result():
    progress = Progress()
    progress.complete(1, 3, 4, {"marker": "good"})
    progress.complete(1, 1, 9, {"marker": "worse"})

    record = progress.record(1)
    assert record.stars == 3
    assert record.best_parts == 4
    assert record.best_bench == {"marker": "good"}, "a worse replay must not overwrite"
    assert record.attempts == 2, "but it still counts as an attempt"

    progress.complete(1, 3, 2, {"marker": "better"})
    assert progress.record(1).best_bench == {"marker": "better"}
    assert progress.record(1).best_parts == 2


def test_progress_remembers_the_round_and_survives_a_round_trip(tmp_path):
    path = tmp_path / "progress.json"
    progress = Progress()
    progress.complete(1, 2, 3, {"elements": []})
    progress.visit(5)
    progress.save(path)

    reloaded = Progress.load(path)
    assert reloaded.last_round == 5
    assert reloaded.solved_bench(1) == {"elements": []}
    assert json.loads(path.read_text())["last_round"] == 5


def test_a_version_one_save_still_loads():
    old = {"version": 1, "rounds": {"3": {"stars": 2, "attempts": 4, "best_parts": 5}}}
    progress = Progress.from_dict(old)
    assert progress.record(3).stars == 2
    assert progress.record(3).best_bench is None
    assert progress.last_round is None
