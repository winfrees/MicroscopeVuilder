"""M8b gate: rounds 13-15, the contrast techniques.

All three are pupil and polarization engineering. The tests check that the
technique does what it claims *and* that removing the component takes the effect
away -- a contrast technique that works regardless of what is on the bench is not
teaching anything.
"""

import dataclasses
import math

import numpy as np
import pytest

from microscopevuilder.bench.bench import Bench, BenchElement
from microscopevuilder.game.rounds import (
    ANNULUS_INNER,
    ANNULUS_OUTER,
    _birefringent_specimen,
    _unstained_cell,
    get_round,
)
from microscopevuilder.imaging.synthesis import RenderSpec, render_build
from microscopevuilder.optics.coherence import contrast
from microscopevuilder.optics.jones import (
    CIRCULAR_RIGHT,
    HORIZONTAL,
    intensity,
    linear_polarizer,
    retarder,
    transmit,
)
from microscopevuilder.rules.base import Status

LAMBDA = 0.5461


def _named(report, name):
    return next(r for r in report.results if r.name == name)


def _render(bench, specimen, n=256):
    return render_build(
        bench, 0.0,
        RenderSpec(specimen=specimen, n=n, detector_name="sensor", coherence_parameter=0.7),
    )


def _without(bench: Bench, *names: str) -> Bench:
    return Bench([e for e in bench.elements if e.name not in names])


# --- Jones calculus ----------------------------------------------------------


def test_crossed_polarizers_extinguish_to_the_stated_ratio():
    crossed = transmit(HORIZONTAL, linear_polarizer(0.0), linear_polarizer(math.pi / 2))
    assert intensity(crossed) == pytest.approx(1e-4, rel=0.1)


def test_a_half_wave_plate_at_45_degrees_opens_crossed_polars_completely():
    through = transmit(
        HORIZONTAL,
        linear_polarizer(0.0),
        retarder(0.5, math.pi / 4),
        linear_polarizer(math.pi / 2),
    )
    assert intensity(through) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("degrees", [0, 15, 30, 45, 60, 75, 90])
def test_a_retarder_between_crossed_polars_follows_the_sin_squared_law(degrees):
    # I = sin^2(2a) sin^2(pi * retardance): dark when the axis lines up with either
    # polar, brightest at 45 degrees. This is what makes a birefringent object wink
    # four times per stage rotation.
    angle = math.radians(degrees)
    through = transmit(
        HORIZONTAL, linear_polarizer(0.0), retarder(0.25, angle), linear_polarizer(math.pi / 2)
    )
    expected = math.sin(2 * angle) ** 2 * math.sin(math.pi * 0.25) ** 2
    assert intensity(through) == pytest.approx(expected, abs=0.01)


def test_a_quarter_wave_plate_makes_circular_light():
    circular = transmit(HORIZONTAL, retarder(0.25, math.pi / 4))
    assert abs(abs(circular[0]) - abs(circular[1])) < 1e-9
    assert abs(abs(np.vdot(CIRCULAR_RIGHT, circular)) - 1.0) < 1e-6


def test_retardance_follows_thickness_times_birefringence_over_wavelength():
    from microscopevuilder.optics.jones import birefringence_retardance

    assert birefringence_retardance(3.0, 0.02, 0.5461) == pytest.approx(3.0 * 0.02 / 0.5461)
    with pytest.raises(ValueError):
        birefringence_retardance(3.0, 0.02, 0.0)


# --- round 13: phase contrast ------------------------------------------------


def test_round13_reference_passes_both_passes():
    rnd = get_round(13)
    bench = rnd.reference_build()
    assert rnd.grade(bench).passed, "\n" + rnd.grade(bench).format()
    assert rnd.measure_image(bench).passed, "\n" + rnd.measure_image(bench).format()


def test_a_phase_object_is_invisible_until_the_technique_is_fitted():
    # The premise. Brightfield has nothing to show, because the specimen changes no
    # amplitude at all -- so the round is not won by focusing harder.
    rnd = get_round(13)
    full = rnd.reference_build()
    bare = _without(full, "annulus", "phase_ring")

    assert contrast(_render(bare, _unstained_cell).intensity) < 0.10
    assert contrast(_render(full, _unstained_cell).intensity) > 0.40


def test_the_phase_ring_alone_does_nothing_and_says_why():
    rnd = get_round(13)
    ring_only = _without(rnd.reference_build(), "annulus")
    rendered = _render(ring_only, _unstained_cell)
    assert contrast(rendered.intensity) < 0.10
    assert any("no condenser annulus" in n for n in rendered.notes)


def test_a_mismatched_ring_is_caught_before_the_image_is_even_formed():
    rnd = get_round(13)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, metadata={**e.metadata, "inner": 0.2, "outer": 0.4})
        if e.name == "phase_ring" else e
        for e in bench.elements
    ]
    row = _named(rnd.grade(bench), "Ring and annulus match")
    assert row.status is Status.FAIL
    assert "conjugate planes" in row.equation


def test_the_ring_must_sit_at_the_back_focal_plane():
    rnd = get_round(13)
    bench = rnd.reference_build()
    bench.move("phase_ring", bench.get("phase_ring").s + 8.0)
    row = _named(rnd.grade(bench), "Ring at the back focal plane")
    assert row.status is Status.FAIL


def test_both_jobs_of_the_phase_ring_are_necessary():
    # A quarter-wave shift with no attenuation, and attenuation with no shift,
    # must each fail to produce useful contrast on their own.
    rnd = get_round(13)

    def variant(**changes):
        bench = rnd.reference_build()
        bench.elements = [
            dataclasses.replace(e, metadata={**e.metadata, **changes})
            if e.name == "phase_ring" else e
            for e in bench.elements
        ]
        return contrast(_render(bench, _unstained_cell).intensity)

    both = variant()
    shift_only = variant(transmission=1.0)
    attenuation_only = variant(phase_shift_waves=0.0)
    assert both > 2 * shift_only
    assert both > 2 * attenuation_only


# --- round 14: polarization --------------------------------------------------


def test_round14_reference_passes_both_passes():
    rnd = get_round(14)
    bench = rnd.reference_build()
    assert rnd.grade(bench).passed
    assert rnd.measure_image(bench).passed, "\n" + rnd.measure_image(bench).format()


def test_a_birefringent_specimen_is_invisible_without_polars():
    rnd = get_round(14)
    bare = _without(rnd.reference_build(), "polarizer", "analyzer")
    rendered = _render(bare, _birefringent_specimen)
    assert contrast(rendered.intensity) == pytest.approx(0.0, abs=1e-6)
    assert any("birefringent, not absorbing" in n for n in rendered.notes)


def test_rotating_the_analyzer_lifts_the_background_and_kills_the_contrast():
    # The image must follow the bench, not a baked-in answer.
    rnd = get_round(14)
    base = rnd.reference_build()

    def at(angle_deg):
        bench = Bench([
            dataclasses.replace(e, metadata={**e.metadata, "angle_deg": angle_deg})
            if e.name == "analyzer" else e
            for e in base.elements
        ])
        rendered = _render(bench, _birefringent_specimen)
        return contrast(rendered.intensity), float(rendered.intensity.mean())

    crossed_contrast, crossed_background = at(90.0)
    off_contrast, off_background = at(70.0)

    assert crossed_contrast > 0.9
    assert crossed_background < 0.05  # a genuinely dark field
    assert off_contrast < crossed_contrast
    assert off_background > crossed_background


def test_a_few_degrees_off_crossed_is_flagged_by_malus():
    rnd = get_round(14)
    bench = rnd.reference_build()
    bench.elements = [
        dataclasses.replace(e, metadata={**e.metadata, "angle_deg": 80.0})
        if e.name == "analyzer" else e
        for e in bench.elements
    ]
    row = _named(rnd.grade(bench), "Crossed polars")
    assert row.status is Status.FAIL
    assert "Malus" in row.equation


# --- round 15: DIC -----------------------------------------------------------


def test_round15_reference_passes_both_passes():
    rnd = get_round(15)
    bench = rnd.reference_build()
    assert rnd.grade(bench).passed, "\n" + rnd.grade(bench).format()
    assert rnd.measure_image(bench).passed


def test_dic_renders_relief_and_zero_bias_removes_its_direction():
    # At zero bias a gradient and its opposite give the same intensity, so the
    # image is symmetric: no relief, no sense of which way anything slopes.
    from microscopevuilder.optics.contrast import shear_field

    n = 256
    rnd = get_round(15)
    bench = rnd.reference_build()
    rendered = _render(bench, _unstained_cell, n=n)
    sample = rendered.sample_um
    radius_px = int(round((n * sample / 6) / sample))
    centre = n // 2

    def relief(bias):
        variant = Bench([
            dataclasses.replace(e, metadata={**e.metadata, "bias_waves": bias})
            if e.name == "wollaston" else e
            for e in bench.elements
        ])
        image = _render(variant, _unstained_cell, n=n).intensity
        leading = image[centre, centre - radius_px - 3 : centre - radius_px + 4].mean()
        trailing = image[centre, centre + radius_px - 3 : centre + radius_px + 4].mean()
        return float(trailing - leading)

    assert abs(relief(0.0)) < 0.01
    assert relief(0.15) > 0.05


def test_dic_relief_follows_the_shear_axis():
    # A structure parallel to the shear disappears, which is why you rotate the
    # specimen rather than the prism.
    from microscopevuilder.optics.contrast import shear_field
    from microscopevuilder.imaging.specimens import phase_disc

    n, sample = 256, 0.09
    disc = phase_disc(n, sample, radius_um=n * sample / 6, phase_rad=0.6)
    centre, edge = n // 2, int(round((n * sample / 6) / sample))

    for axis in (0, 1):
        sheared = np.abs(shear_field(disc, 3.0, 0.15, axis=axis)) ** 2
        along_x = sheared[centre, centre - edge] - sheared[centre, centre + edge]
        along_y = sheared[centre - edge, centre] - sheared[centre + edge, centre]
        if axis == 1:
            assert abs(along_x) > 0.05 and abs(along_y) < 1e-6
        else:
            assert abs(along_y) > 0.05 and abs(along_x) < 1e-6


def test_a_wrong_prism_is_rejected_with_the_reason():
    rnd = get_round(15)

    def with_prism(**changes):
        bench = rnd.reference_build()
        bench.elements = [
            dataclasses.replace(e, metadata={**e.metadata, **changes})
            if e.name == "wollaston" else e
            for e in bench.elements
        ]
        return _named(rnd.grade(bench), "DIC prism")

    assert "double image" in with_prism(shear_fraction=2.0).summary
    assert "too small" in with_prism(shear_fraction=0.05).summary
    assert "no direction" in with_prism(bias_waves=0.0).summary


def test_dic_shear_is_expressed_against_the_resolution_limit():
    from microscopevuilder.optics.contrast import shear_for_resolution

    assert shear_for_resolution(0.44, 0.09, fraction=0.6) == pytest.approx(0.6 * 0.44 / 0.09)
    with pytest.raises(ValueError):
        shear_for_resolution(0.44, 0.0)


# --- the arc -----------------------------------------------------------------


def test_every_round_from_one_to_fifteen_exists():
    from microscopevuilder.game.rounds import ROUNDS

    assert sorted(n for n in ROUNDS if n > 0) == list(range(1, 16))


@pytest.mark.parametrize("number", [13, 14, 15])
def test_contrast_rounds_are_graded_on_measurements(number):
    assert get_round(number).has_measured_rules
