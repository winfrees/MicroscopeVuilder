"""M7 gate: episcopic illumination and fluorescence on a branched bench."""

import dataclasses

import pytest

from microscopevuilder.bench.bench import Arm, Bench, BenchElement
from microscopevuilder.game.rounds import (
    EPI_ARM_LENGTH,
    EPI_F_OBJ,
    EPI_JUNCTION_S,
    EPI_LENS_S,
    EPI_OBJECTIVE_S,
    EPI_SPECIMEN_S,
    get_round,
)
from microscopevuilder.optics.spectra import (
    CUBES,
    FLUOROPHORES,
    Band,
    Dichroic,
    FilterCube,
)
from microscopevuilder.rules.base import Status
from microscopevuilder.rules.checks import check_filter_set, check_stokes_shift


def _named(report, name):
    return next(r for r in report.results if r.name == name)


# --- the branched bench ------------------------------------------------------


def test_an_arm_reaches_the_main_axis_at_its_junction():
    bench = Bench(
        [BenchElement("objective", 200.0, "objective", 4.0, 16.0)],
        arms=[Arm("epi", junction_s=150.0, length_mm=100.0, direction=(1.0, 0.0, 0.0))],
    )
    import numpy as np

    np.testing.assert_allclose(
        bench.position_of_arm("epi", 100.0), bench.position_of(150.0)
    )


def test_a_reverse_arm_traverses_the_main_axis_backwards():
    # Epi illumination enters at the beamsplitter and runs back down through the
    # objective. A forward arm would walk away from the specimen instead.
    rnd = get_round(9)
    bench = rnd.reference_build()
    arm = bench.to_paraxial("epi")
    names = [e.name for e in arm.elements]

    assert names.index("epi_lens") < names.index("objective")
    # The objective sits at junction-to-objective distance past the arm length.
    objective = next(e for e in arm.elements if e.name == "objective")
    assert objective.s == pytest.approx(EPI_ARM_LENGTH + (EPI_JUNCTION_S - EPI_OBJECTIVE_S))


def test_the_objective_is_shared_by_both_paths_at_the_right_distance():
    # The physical point of an epi stand: one objective, two paths through it.
    rnd = get_round(9)
    bench = rnd.reference_build()

    imaging = bench.to_paraxial()
    illumination = bench.to_paraxial("epi")
    obj_imaging = next(e for e in imaging.elements if e.name == "objective")
    obj_illum = next(e for e in illumination.elements if e.name == "objective")

    assert obj_imaging.focal_length == obj_illum.focal_length == pytest.approx(EPI_F_OBJ)
    # Objective-to-specimen is the same working distance measured along either path.
    assert EPI_OBJECTIVE_S - EPI_SPECIMEN_S == pytest.approx(
        (EPI_ARM_LENGTH + EPI_JUNCTION_S - EPI_SPECIMEN_S)
        - (EPI_ARM_LENGTH + EPI_JUNCTION_S - EPI_OBJECTIVE_S)
    )


def test_main_axis_trace_ignores_arm_elements():
    rnd = get_round(9)
    bench = rnd.reference_build()
    names = {e.name for e in bench.to_paraxial().elements}
    assert "lamp" not in names and "epi_lens" not in names
    assert "objective" in names


def test_a_branched_bench_round_trips_through_json(tmp_path):
    rnd = get_round(9)
    original = rnd.reference_build()
    path = tmp_path / "epi.json"
    original.save(path)
    loaded = Bench.load(path)

    assert loaded.to_dict() == original.to_dict()
    assert loaded.arms["epi"].continues == "reverse"
    assert [e.name for e in loaded.to_paraxial("epi").elements] == [
        e.name for e in original.to_paraxial("epi").elements
    ]


# --- round 9 -----------------------------------------------------------------


def test_round9_reference_passes():
    rnd = get_round(9)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()


def test_round9_epi_kohler_puts_the_lamp_on_the_back_focal_plane():
    rnd = get_round(9)
    bench = rnd.reference_build()
    bfp_arm_s = EPI_ARM_LENGTH + (EPI_JUNCTION_S - EPI_OBJECTIVE_S) - EPI_F_OBJ
    landed = bench.to_paraxial("epi").image_plane(0.0, search_to=bfp_arm_s + 1)
    assert landed == pytest.approx(bfp_arm_s, abs=0.05)


def test_round9_beamsplitter_below_the_objective_fails_with_the_reason():
    rnd = get_round(9)
    bench = rnd.reference_build()
    bench.move("beamsplitter", 5.0)  # below the objective
    row = _named(rnd.grade(bench), "Epi geometry")
    assert row.status is Status.FAIL
    assert "never reaches the specimen" in row.summary


def test_round9_misfocused_epi_collector_breaks_the_aperture_conjugate():
    rnd = get_round(9)
    bench = rnd.reference_build()
    bench.move("epi_lens", EPI_LENS_S + 12.0)
    assert not rnd.grade(bench).passed


# --- round 10: fluorescence --------------------------------------------------


def test_round10_reference_passes():
    rnd = get_round(10)
    report = rnd.grade(rnd.reference_build())
    assert report.passed, "\n" + report.format()


def test_the_wrong_cube_delivers_almost_no_signal():
    # A TRITC cube on a FITC specimen excites nothing: four orders of magnitude down.
    fitc = FLUOROPHORES["fitc"]
    right = CUBES["fitc"].signal(fitc)
    wrong = CUBES["tritc"].signal(fitc)
    assert wrong < right / 1000

    row = check_filter_set(CUBES["tritc"], fitc)
    assert row.status is Status.FAIL
    assert "barely excites" in row.summary
    assert "495" in row.remedy  # points at the dye's actual absorption


def test_overlapping_bands_leak_excitation_to_the_detector():
    # The other way to fail: plenty of signal, but the background is not black.
    leaky = FilterCube(Band(482.0, 35.0), Dichroic(506.0), Band(500.0, 60.0), "leaky")
    assert leaky.bleedthrough() > 10 * CUBES["fitc"].bleedthrough()

    row = check_filter_set(leaky, FLUOROPHORES["fitc"])
    assert row.status is Status.FAIL
    assert "excitation is reaching the detector" in row.summary


def test_a_dichroic_outside_the_stokes_shift_cannot_work():
    fitc = FLUOROPHORES["fitc"]
    bad = FilterCube(Band(482.0, 35.0), Dichroic(600.0), Band(536.0, 40.0), "bad edge")
    row = check_stokes_shift(fitc, bad)
    assert row.status is Status.FAIL
    assert "outside" in row.summary
    assert check_stokes_shift(fitc, CUBES["fitc"]).status is Status.PASS


@pytest.mark.parametrize("key", sorted(FLUOROPHORES))
def test_every_catalogued_dye_has_a_positive_stokes_shift(key):
    # Emission is always redshifted from excitation; a negative shift would be a
    # data-entry error, and the whole technique depends on the gap.
    dye = FLUOROPHORES[key]
    assert dye.stokes_shift_nm > 0
    assert dye.emission_peak_nm > dye.excitation_peak_nm


@pytest.mark.parametrize("key", sorted(CUBES))
def test_each_standard_cube_has_its_dichroic_between_its_bands(key):
    cube = CUBES[key]
    assert cube.excitation.high_nm <= cube.dichroic.edge_nm + 5.0
    assert cube.emission.low_nm >= cube.dichroic.edge_nm - 5.0
    assert not cube.excitation.overlaps(cube.emission)


def test_round10_swapping_in_a_mismatched_cube_fails_the_round():
    rnd = get_round(10)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, metadata={**e.metadata, "cube": "dapi"})
        if e.name == "beamsplitter" else e
        for e in bench.elements
    ]
    report = rnd.grade(bench)
    assert not report.passed
    assert _named(report, "Filter set").status is Status.FAIL
