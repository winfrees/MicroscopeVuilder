"""Bench model: 3D-native geometry, editing, and round-tripping."""

import numpy as np
import pytest

from microscopevuilder.bench.bench import Bench, BenchElement, Fold


def test_unfolded_bench_runs_along_its_initial_direction():
    b = Bench([BenchElement("lens", 50.0, "lens", 10.0, 25.0)])
    np.testing.assert_allclose(b.position_of(50.0), [50.0, 0.0, 0.0])


def test_a_fold_turns_the_axis_without_changing_the_optics():
    # An episcopic path is the same 1D trace, drawn bent. The paraxial result must
    # be identical with and without the fold.
    elements = [
        BenchElement("obj", 20.0, "objective", 3.0, 10.0),
        BenchElement("sensor", 120.0, "detector", 8.0),
    ]
    straight = Bench(list(elements))
    folded = Bench(list(elements), [Fold(60.0, (0.0, 0.0, 1.0), "dichroic")])

    assert straight.to_paraxial().image_plane(0.0) == pytest.approx(
        folded.to_paraxial().image_plane(0.0)
    )
    # ...but the geometry differs, which is what the renderer draws.
    np.testing.assert_allclose(folded.position_of(100.0), [60.0, 0.0, 40.0])
    np.testing.assert_allclose(straight.position_of(100.0), [100.0, 0.0, 0.0])


def test_multiple_folds_compose():
    b = Bench(
        [BenchElement("end", 300.0, "detector", 5.0)],
        [Fold(100.0, (0.0, 0.0, 1.0)), Fold(200.0, (-1.0, 0.0, 0.0))],
    )
    np.testing.assert_allclose(b.position_of(250.0), [50.0, 0.0, 100.0])


def test_polyline_gives_the_renderer_the_axis_vertices():
    b = Bench(
        [BenchElement("end", 200.0, "detector", 5.0)],
        [Fold(100.0, (0.0, 1.0, 0.0))],
    )
    verts = b.polyline()
    assert len(verts) == 3
    np.testing.assert_allclose(verts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(verts[-1], [100.0, 100.0, 0.0])


def test_duplicate_names_are_rejected():
    b = Bench([BenchElement("lens", 0.0, "lens", 10.0, 25.0)])
    with pytest.raises(ValueError):
        b.add(BenchElement("lens", 50.0, "lens", 10.0, 25.0))


def test_zero_length_fold_direction_is_rejected():
    with pytest.raises(ValueError):
        Bench([], [Fold(10.0, (0.0, 0.0, 0.0))]).position_of(20.0)


def test_bench_round_trips_through_json(tmp_path):
    original = Bench(
        [
            BenchElement("obj", 17.6, "objective", 4.0, 16.0, label="10x", metadata={"na": 0.25}),
            BenchElement("stop", 193.6, "field_stop", 11.0),
        ],
        [Fold(120.0, (0.0, 0.0, 1.0), "mirror")],
    )
    path = tmp_path / "bench.json"
    original.save(path)
    loaded = Bench.load(path)

    assert loaded.to_dict() == original.to_dict()
    assert loaded.get("obj").metadata["na"] == 0.25
    np.testing.assert_allclose(loaded.position_of(200.0), original.position_of(200.0))
