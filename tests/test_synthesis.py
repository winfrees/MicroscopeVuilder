"""M10 gate: the generated image is an image OF the player's build.

Before M10 synthesis took four loose scalars and rendered a generic picture, so a
plan objective and a non-plan objective produced identical images. These tests pin
that each property of the bench actually reaches the pixels.
"""

import dataclasses

import numpy as np
import pytest

from microscopevuilder.bench.bench import Bench
from microscopevuilder.game.rounds import _infinity_stand, get_round
from microscopevuilder.imaging import metrics
from microscopevuilder.imaging.build_optics import (
    REFERENCE_MAGNIFICATION,
    REFERENCE_NA,
    resolve_build_optics,
)
from microscopevuilder.imaging.specimens import (
    commensurate_period,
    sinusoidal_amplitude_grating,
)
from microscopevuilder.imaging.synthesis import RenderSpec, measure_mtf, render_build

LAMBDA = 0.5461


def _stand(catalog_key: str, na: float = 0.40, magnification: float = 20.0) -> Bench:
    bench = _infinity_stand(magnification=magnification, na=na)
    return Bench(
        [
            dataclasses.replace(e, catalog_key=catalog_key) if e.name == "objective" else e
            for e in bench.elements
        ]
    )


def _grating_spec(period_um: float, **kw) -> RenderSpec:
    def specimen(n, sample_um):
        return sinusoidal_amplitude_grating(
            n, sample_um, commensurate_period(n, sample_um, period_um)
        )

    return RenderSpec(specimen=specimen, **kw)


# --- the resolver ------------------------------------------------------------


def test_aberrations_come_from_the_objective_catalog_entry():
    optics = resolve_build_optics(_stand("cfi_plan_achro_20x"), 0.0, "sensor", LAMBDA)
    assert optics.aberration_source == "catalog:cfi_plan_achro_20x"
    assert optics.wavefront.rms_um > 0


def test_an_objective_with_no_catalog_match_is_declared_ideal_not_silently_perfect():
    bench = _infinity_stand()
    bench.elements = [
        dataclasses.replace(e, catalog_key=None, metadata={"na": 0.4})
        if e.name == "objective" else e
        for e in bench.elements
    ]
    optics = resolve_build_optics(bench, 0.0, "sensor", LAMBDA)
    assert optics.aberration_source == "ideal"
    assert any("aberration-free" in n for n in optics.notes)


def test_an_objective_declaring_a_grade_borrows_that_grade_only():
    # Matching on aperture alone would hand an uncharacterised objective an
    # achromat's secondary spectrum, and it would then image badly in blue for
    # reasons nothing on the bench explains.
    bench = _infinity_stand()
    bench.elements = [
        dataclasses.replace(e, catalog_key=None, metadata={"na": 0.4, "grade": "plan_apochromat"})
        if e.name == "objective" else e
        for e in bench.elements
    ]
    optics = resolve_build_optics(bench, 0.0, "sensor", LAMBDA)
    assert optics.aberration_source.startswith("nearest:cfi_plan_apo")


def test_defocus_is_computed_from_where_the_image_actually_lands():
    bench = _stand("cfi_plan_apo_20x", na=0.75)
    focused = resolve_build_optics(bench, 0.0, "sensor", LAMBDA)
    assert focused.image_defocus_mm == pytest.approx(0.0, abs=1e-6)

    bench.move("sensor", bench.get("sensor").s + 2.0)
    defocused = resolve_build_optics(bench, 0.0, "sensor", LAMBDA)
    assert defocused.image_defocus_mm == pytest.approx(2.0, abs=1e-6)
    # Longitudinal magnification is M^2, so 2 mm at the sensor is 5 um at the object.
    assert defocused.object_defocus_mm == pytest.approx(2.0 / 400.0)
    assert defocused.wavefront.strehl(LAMBDA) < focused.wavefront.strehl(LAMBDA)


def test_irradiance_is_reported_against_a_familiar_reference_build():
    # NA^2/M^2 in absolute units is a tiny number for any real objective and tells
    # the player nothing; against a 10x/0.25 it says something useful.
    reference = _stand("cfi_plan_achro_10x", na=REFERENCE_NA, magnification=REFERENCE_MAGNIFICATION)
    assert resolve_build_optics(reference, 0.0, "sensor", LAMBDA).relative_irradiance == pytest.approx(1.0)

    bright = resolve_build_optics(_stand("cfi_plan_apo_20x", na=0.75), 0.0, "sensor", LAMBDA)
    assert bright.relative_irradiance == pytest.approx((0.75 / 20) ** 2 / (0.25 / 10) ** 2)
    assert bright.relative_irradiance > 1.0


# --- the catalog distinction now reaches the pixels --------------------------


def test_an_achromat_and_an_apochromat_no_longer_render_identically():
    # The failure M10 exists to fix. Same NA, same magnification, same focal
    # length: only the correction grade differs.
    achromat = resolve_build_optics(_stand("cfi_plan_achro_20x"), 0.0, "sensor", LAMBDA)
    apochromat = resolve_build_optics(_stand("cfi_plan_apo_20x"), 0.0, "sensor", LAMBDA)
    assert achromat.wavefront.rms_um > 2 * apochromat.wavefront.rms_um


def test_the_achromat_collapses_in_blue_light_and_the_apochromat_does_not():
    # Round 6 in one assertion. Secondary spectrum is stored as f/2000 for an
    # achromat, so at 436 nm the defocus it induces swamps everything else.
    blue = 0.436
    achromat = resolve_build_optics(_stand("cfi_plan_achro_20x"), 0.0, "sensor", blue)
    apochromat = resolve_build_optics(_stand("cfi_plan_apo_20x"), 0.0, "sensor", blue)

    assert achromat.dominant_aberration[0] == "defocus"
    assert achromat.wavefront.strehl(blue) < 0.2
    assert apochromat.wavefront.strehl(blue) > 0.9


def test_secondary_spectrum_follows_the_focal_length_not_a_stored_constant():
    # f/2000 means a 4x (f = 50 mm) suffers ten times the focus shift of a 40x
    # (f = 5 mm) for the same colour error. A stored wavefront constant could not
    # express that, which is why the catalog stores a fraction of focal length.
    blue = 0.436
    long_focus = resolve_build_optics(
        _stand("cfi_plan_achro_4x", na=0.10, magnification=4.0), 0.0, "sensor", blue
    )
    short_focus = resolve_build_optics(
        _stand("cfi_plan_achro_40x", na=0.10, magnification=40.0), 0.0, "sensor", blue
    )
    ratio = long_focus.wavefront.coefficients["defocus"] / short_focus.wavefront.coefficients["defocus"]
    assert ratio == pytest.approx(10.0, rel=0.01)


def test_field_curvature_only_bites_off_axis():
    bench = _stand("cfi_plan_achro_20x")
    on_axis = resolve_build_optics(bench, 0.0, "sensor", LAMBDA, field_height=0.0)
    corner = resolve_build_optics(bench, 0.0, "sensor", LAMBDA, field_height=1.0)
    assert corner.wavefront.rms_um > on_axis.wavefront.rms_um
    # Spherical is field-independent, so it must be identical in both.
    assert on_axis.wavefront.coefficients["spherical"] == pytest.approx(
        corner.wavefront.coefficients["spherical"]
    )


def test_plan_objectives_hold_their_sag_inside_the_depth_of_focus():
    # What "plan" means. At NA 0.75 the depth of focus is 0.49 um, so a plan design
    # with a sag of several microns would not be a plan design.
    from microscopevuilder.bench.catalog import load_catalog
    from microscopevuilder.rules.tolerances import depth_of_focus_mm

    catalog = load_catalog()
    for objective in catalog.objectives.values():
        dof_um = depth_of_focus_mm(LAMBDA, objective.na) * 1000.0
        assert objective.aberrations.field_curvature_sag_um < 4 * dof_um, objective.key


# --- rendering ---------------------------------------------------------------


def test_a_rendered_image_degrades_when_the_build_does():
    bench = _stand("cfi_plan_apo_20x", na=0.75)
    spec = _grating_spec(0.8)
    sharp = render_build(bench, 0.0, spec)

    bench.move("sensor", bench.get("sensor").s + 3.0)
    soft = render_build(bench, 0.0, spec)

    assert metrics.michelson_contrast(soft.intensity) < metrics.michelson_contrast(sharp.intensity)


def test_measured_mtf_falls_monotonically_and_dies_past_the_cutoff():
    from microscopevuilder.optics.psf import mtf_cutoff_cycles_per_um

    bench = _stand("cfi_plan_apo_20x", na=0.75)
    spec = _grating_spec(1.0, coherence_parameter=0.6)
    periods = [2.0, 1.2, 0.9, 0.7, 0.55, 0.4]
    measured = measure_mtf(bench, 0.0, spec, periods)

    values = [measured[p] for p in sorted(measured, reverse=True)]
    assert values == sorted(values, reverse=True)

    cutoff_period = 1.0 / ((1.0 + 0.6) * 0.75 / LAMBDA)
    finest = min(measured)
    assert finest < cutoff_period
    assert measured[finest] < 0.02


def test_resolved_period_reports_the_finest_grating_that_survives():
    bench = _stand("cfi_plan_apo_20x", na=0.75)
    measured = measure_mtf(bench, 0.0, _grating_spec(1.0), [2.0, 1.0, 0.7, 0.5, 0.4])
    resolved = metrics.resolved_period_um(measured, threshold=0.10)
    assert resolved is not None
    assert 0.4 < resolved < 1.2


def test_a_dim_build_is_rendered_noisy_rather_than_merely_darker():
    # "Technically correct but too dim to use" has to be visible, not just stated.
    bench = _stand("cfi_plan_apo_100x_oil", na=1.45, magnification=100.0)
    spec = _grating_spec(0.6, exposure_photons=2000.0)
    dim = render_build(bench, 0.0, spec)

    bright = render_build(
        _stand("cfi_plan_apo_20x", na=0.75), 0.0, _grating_spec(0.6, exposure_photons=2000.0)
    )
    assert dim.optics.relative_irradiance < bright.optics.relative_irradiance
    # The same exposure yields a measurably worse image, not merely a darker one.
    dim_snr = metrics.signal_to_noise(dim.intensity, sample_um=dim.sample_um, cutoff_cycles_per_um=2.0)
    bright_snr = metrics.signal_to_noise(
        bright.intensity, sample_um=bright.sample_um, cutoff_cycles_per_um=2.0
    )
    assert dim_snr < bright_snr


def test_a_starved_exposure_says_so_in_the_notes():
    bench = _stand("cfi_plan_apo_100x_oil", na=1.45, magnification=100.0)
    starved = render_build(bench, 0.0, _grating_spec(0.6, exposure_photons=20.0))
    assert any("noise-limited" in n for n in starved.notes)


def test_photometry_is_deterministic_for_a_given_seed():
    bench = _stand("cfi_plan_apo_20x", na=0.75)
    spec = _grating_spec(0.8, exposure_photons=500.0, seed=7)
    a = render_build(bench, 0.0, spec)
    b = render_build(bench, 0.0, spec)
    np.testing.assert_allclose(a.intensity, b.intensity)


@pytest.mark.parametrize("number", [1, 2, 11, 12])
def test_every_finite_round_renders_through_its_own_reference_build(number):
    rnd = get_round(number)
    bench = rnd.reference_build()
    detector = next(
        name for name in ("sensor", "screen", "intermediate_image") if bench.has(name)
    )
    rendered = render_build(bench, rnd.s_object, _grating_spec(1.0, detector_name=detector))
    assert np.isfinite(rendered.intensity).all()
    assert rendered.intensity.min() >= 0.0
    assert rendered.optics.na > 0


# --- metrics -----------------------------------------------------------------


def test_modulation_measures_a_known_grating_correctly():
    # An unaberrated, well-sampled grating of known depth must read back its depth.
    n, dx = 256, 0.05
    grating = sinusoidal_amplitude_grating(n, dx, commensurate_period(n, dx, 2.0), modulation=0.4)
    intensity = np.abs(grating) ** 2
    measured = metrics.modulation_at_period(intensity, dx, commensurate_period(n, dx, 2.0))
    assert measured == pytest.approx(0.73, abs=0.05)  # |1+0.4cos|^2 -> 0.8/1.08


def test_modulation_rejects_a_period_it_cannot_represent():
    with pytest.raises(ValueError):
        metrics.modulation_at_period(np.ones((64, 64)), 0.1, 0.0001)


def test_noise_estimate_is_unbiased_and_zero_on_a_clean_image():
    # Two cruder estimators were rejected; this pins why the current one is used.
    rng = np.random.default_rng(0)
    n, dx = 256, 0.05
    axis = (np.arange(n) - n // 2) * dx
    clean = 1.0 + 0.5 * np.cos(2 * np.pi * axis / 1.0)[None, :] * np.ones((n, 1))
    cutoff = 2 * 0.5 / 0.55

    assert metrics.estimate_noise(clean, dx, cutoff) == pytest.approx(0.0, abs=1e-3)
    for true_sd in (0.005, 0.01, 0.05):
        noisy = clean + rng.normal(0, true_sd, (n, n))
        assert metrics.estimate_noise(noisy, dx, cutoff) == pytest.approx(true_sd, rel=0.1)


def test_snr_does_not_report_infinity_on_a_photon_starved_image():
    # The failure mode of a dark-region estimator: quantized zeros give zero spread.
    rng = np.random.default_rng(1)
    starved = rng.poisson(np.full((128, 128), 0.4)).astype(float)
    assert np.isfinite(metrics.signal_to_noise(starved, sample_um=0.05, cutoff_cycles_per_um=2.0))


def test_field_uniformity_is_one_for_an_even_field_and_falls_with_vignetting():
    even = np.ones((128, 128))
    assert metrics.field_uniformity(even) == pytest.approx(1.0)

    n = 128
    axis = np.linspace(-1, 1, n)
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    vignetted = np.clip(1.0 - 0.6 * radius**2, 0, None)
    assert metrics.field_uniformity(vignetted) < 0.8


def test_lateral_shift_finds_a_known_displacement():
    base = np.zeros((64, 64))
    base[30:34, 30:34] = 1.0
    shifted = np.roll(base, 5, axis=1)
    assert metrics.lateral_shift_px(base, shifted) == pytest.approx(5.0, abs=1.0)
