"""The white card: a probe that must never disturb what it measures."""

import pytest

from microscopevuilder.bench.bench import BenchElement
from microscopevuilder.game.rounds import get_round
from microscopevuilder.bench.probe import find_planes, read_card, scan_axis


@pytest.fixture
def scope():
    rnd = get_round(2)
    return rnd.reference_build(), rnd.s_object


def test_a_card_does_not_change_the_optics_it_is_measuring(scope):
    # The whole point: inserting a probe must not move the image plane, the
    # aperture stop or the NA.
    bench, s_obj = scope
    system = bench.to_paraxial()
    before = (
        system.image_plane(s_obj, search_to=200.0),
        system.aperture_stop(s_obj).name,
        system.object_space_na(s_obj),
    )
    for s in (50.0, 120.0, 193.6):
        bench.add_card(s)
    after_system = bench.to_paraxial()
    after = (
        after_system.image_plane(s_obj, search_to=200.0),
        after_system.aperture_stop(s_obj).name,
        after_system.object_space_na(s_obj),
    )
    assert before == after


def test_card_at_the_image_plane_reads_sharp_with_the_right_magnification(scope):
    bench, s_obj = scope
    r = read_card(bench, s_obj, 193.6)
    assert r.is_image_plane
    assert not r.is_pupil_plane
    assert abs(r.magnification) == pytest.approx(10.0, rel=1e-3)
    assert r.blur_diameter_mm == pytest.approx(0.0, abs=1e-3)


def test_card_at_the_aperture_stop_reads_as_a_pupil(scope):
    # By definition the chief ray crosses the axis at the stop, so a card there
    # shows an evenly filled disc with no image structure.
    bench, s_obj = scope
    r = read_card(bench, s_obj, bench.get("objective").s)
    assert r.is_pupil_plane
    assert not r.is_image_plane
    assert r.blur_diameter_mm > 1.0


def test_card_between_planes_reports_the_blur_and_how_far_to_focus(scope):
    bench, s_obj = scope
    r = read_card(bench, s_obj, 150.0)
    assert r.plane_type == "neither"
    assert r.blur_diameter_mm > 0.1
    assert r.defocus_mm == pytest.approx(193.6 - 150.0, abs=0.01)
    assert "nearest image plane" in r.describe()


def test_blur_shrinks_to_zero_as_the_card_approaches_the_image(scope):
    bench, s_obj = scope
    blurs = [read_card(bench, s_obj, s).blur_diameter_mm for s in (150.0, 170.0, 190.0, 193.6)]
    assert blurs == sorted(blurs, reverse=True)
    assert blurs[-1] == pytest.approx(0.0, abs=1e-3)


def test_find_planes_locates_both_sets(scope):
    bench, s_obj = scope
    found = find_planes(bench, s_obj, 0.0, 240.0)
    assert any(s == pytest.approx(193.6, abs=0.5) for s in found["image"])
    assert any(s == pytest.approx(17.6, abs=0.5) for s in found["pupil"])


def test_image_and_pupil_planes_interleave(scope):
    # The structural fact the card is meant to teach: field and aperture planes
    # alternate along the axis, never coincide in a well-formed system.
    bench, s_obj = scope
    found = find_planes(bench, s_obj, 0.0, 240.0)
    for image_s in found["image"]:
        for pupil_s in found["pupil"]:
            assert abs(image_s - pupil_s) > 1.0


def test_scan_axis_returns_a_reading_per_sample(scope):
    bench, s_obj = scope
    readings = scan_axis(bench, s_obj, 0.0, 200.0, samples=50)
    assert len(readings) == 50
    assert readings[0].s == 0.0
    assert readings[-1].s == pytest.approx(200.0)


def test_scan_axis_rejects_a_degenerate_sweep(scope):
    bench, s_obj = scope
    with pytest.raises(ValueError):
        scan_axis(bench, s_obj, 0.0, 200.0, samples=1)


def test_cards_round_trip_through_json(tmp_path, scope):
    bench, _ = scope
    bench.add_card(100.0)
    path = tmp_path / "with_card.json"
    bench.save(path)
    from microscopevuilder.bench.bench import Bench

    loaded = Bench.load(path)
    assert len(loaded.cards()) == 1
    assert loaded.cards()[0].s == 100.0


def test_chief_ray_passes_through_the_centre_of_the_aperture_stop(scope):
    # The definition. Using a ray parallel to the axis instead puts the pupil in
    # the wrong place and understates vignetting.
    bench, s_obj = scope
    system = bench.to_paraxial()
    stop = system.aperture_stop(s_obj)
    chief = system.chief_ray(s_obj, 0.5)
    assert system.trace(chief, s_obj, stop.s).y == pytest.approx(0.0, abs=1e-9)
