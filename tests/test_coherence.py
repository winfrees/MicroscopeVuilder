"""M2b gate: partially coherent imaging against its analytic limits."""

import numpy as np
import pytest

from microscopevuilder.imaging.specimens import (
    commensurate_period,
    sinusoidal_amplitude_grating,
    sinusoidal_phase_grating,
    two_points,
)
from microscopevuilder.optics.coherence import (
    contrast,
    image_coherent,
    image_partially_coherent,
    make_source,
)
from microscopevuilder.optics.psf import make_pupil_grid
from microscopevuilder.optics.wavefront import Wavefront

LAMBDA = 0.5461
NA = 0.5
N = 384
R_PX = 36


@pytest.fixture(scope="module")
def grid():
    return make_pupil_grid(NA, LAMBDA, n=N, pupil_radius_px=R_PX)


def _grating_contrast(grid, target_period_um, S, phase=False, **kw):
    period = commensurate_period(grid.n, grid.image_sample_um, target_period_um)
    maker = sinusoidal_phase_grating if phase else sinusoidal_amplitude_grating
    specimen = maker(grid.n, grid.image_sample_um, period, **kw)
    return contrast(image_partially_coherent(specimen, grid, S))


def test_zero_coherence_parameter_reproduces_the_coherent_formula_exactly(grid):
    # S -> 0 is a single on-axis source point, which is coherent imaging by
    # definition. Any discrepancy means the source integration is misnormalized.
    specimen = sinusoidal_amplitude_grating(grid.n, grid.image_sample_um, 2.0)
    np.testing.assert_allclose(
        image_partially_coherent(specimen, grid, 0.0),
        image_coherent(specimen, grid),
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(make_source(grid, 0.0)) == 1


@pytest.mark.parametrize("S", [0.0, 0.5, 1.0])
def test_partially_coherent_cutoff_is_one_plus_s_times_na_over_lambda(grid, S):
    # The central result of the module. Just inside the predicted cutoff the
    # grating must be visible; just outside it must vanish completely.
    cutoff_period = LAMBDA / ((1.0 + S) * NA)
    assert _grating_contrast(grid, cutoff_period * 1.25, S) > 0.01
    assert _grating_contrast(grid, cutoff_period * 0.85, S) == pytest.approx(0.0, abs=1e-9)


def test_matched_illumination_doubles_the_resolved_frequency(grid):
    # Round 5, as one assertion: a grating at 0.7 um is invisible with the condenser
    # closed down and plainly visible with it matched to the objective.
    assert _grating_contrast(grid, 0.7, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert _grating_contrast(grid, 0.7, 1.0) > 0.05


def test_closing_the_condenser_raises_contrast_on_coarse_detail(grid):
    # The trap: stopping down genuinely *does* increase contrast on what remains
    # resolvable, which is why the habit persists. The cost is the resolution above.
    coarse = 1.5
    assert _grating_contrast(grid, coarse, 0.0) > _grating_contrast(grid, coarse, 1.0)


def test_a_pure_phase_object_is_essentially_invisible_in_brightfield(grid):
    # The premise of the phase contrast round. A phase grating of 0.2 rad has the
    # same optical thickness modulation as a clearly visible amplitude grating, yet
    # produces almost no intensity contrast -- and less as illumination broadens.
    amplitude = _grating_contrast(grid, 2.0, 0.5, modulation=0.2)
    phase_s0 = _grating_contrast(grid, 2.0, 0.0, phase=True, phase_rad=0.2)
    phase_s1 = _grating_contrast(grid, 2.0, 1.0, phase=True, phase_rad=0.2)

    assert phase_s0 < amplitude / 10
    assert phase_s1 < phase_s0 / 10
    assert phase_s1 < 1e-3


def test_phase_object_visibility_would_be_exactly_zero_without_second_order_terms(grid):
    # What residual contrast there is scales as phase^2: it is the second-order
    # term, not leakage. Halving the phase must quarter the contrast.
    strong = _grating_contrast(grid, 2.0, 0.0, phase=True, phase_rad=0.2)
    weak = _grating_contrast(grid, 2.0, 0.0, phase=True, phase_rad=0.1)
    assert strong / weak == pytest.approx(4.0, rel=0.1)


def test_image_is_real_and_non_negative(grid):
    # Intensities. A negative pixel means the source integration is summing
    # amplitudes somewhere it should be summing intensities.
    specimen = two_points(grid.n, grid.image_sample_um, 2.0)
    img = image_partially_coherent(specimen, grid, 0.7)
    assert img.dtype == np.float64
    assert img.min() >= 0.0


def test_uniform_specimen_gives_a_uniform_image_at_matched_illumination(grid):
    # Every source point inside the objective pupil passes its undiffracted beam,
    # so a blank slide must come out flat. Non-uniformity here would be an artifact
    # of the source sampling leaking into the field.
    blank = np.ones((grid.n, grid.n), dtype=complex)
    img = image_partially_coherent(blank, grid, 1.0)
    assert contrast(img) == pytest.approx(0.0, abs=1e-9)
    assert img.mean() == pytest.approx(1.0, rel=1e-9)


def test_aberrations_reduce_contrast_under_partial_coherence(grid):
    sharp = _grating_contrast(grid, 1.2, 0.5)
    period = commensurate_period(grid.n, grid.image_sample_um, 1.2)
    specimen = sinusoidal_amplitude_grating(grid.n, grid.image_sample_um, period)
    blurred = contrast(
        image_partially_coherent(specimen, grid, 0.5, Wavefront({"spherical": 0.15}))
    )
    assert blurred < sharp


def test_source_sampling_is_refined_enough_to_have_converged(grid):
    # The default source step must not be a visible approximation: refining it
    # should not move the answer much.
    period = commensurate_period(grid.n, grid.image_sample_um, 0.8)
    specimen = sinusoidal_amplitude_grating(grid.n, grid.image_sample_um, period)
    default = contrast(image_partially_coherent(specimen, grid, 1.0))
    fine = contrast(
        image_partially_coherent(
            specimen, grid, 1.0, source=make_source(grid, 1.0, step_px=2)
        )
    )
    assert default == pytest.approx(fine, rel=0.05)


def test_oversized_source_is_rejected_rather_than_silently_aliasing(grid):
    # np.roll wraps, so an out-of-array source point would fold high frequencies
    # back in and fabricate detail. This must fail loudly.
    with pytest.raises(ValueError, match="alias"):
        make_source(grid, 8.0)
    # The largest S this grid supports is (n/2 - r) / r.
    assert make_source(grid, (N // 2 - R_PX) / R_PX)


def test_commensurate_period_prevents_leakage_across_the_cutoff(grid):
    # A grating beyond the coherent cutoff must be exactly invisible at S=0. With a
    # period that does not divide the array, wrap leakage makes it look resolved --
    # this test pins the difference so the helper is not quietly dropped.
    # Deliberately half a cycle off fitting the array: the worst case for wrap
    # discontinuity. (Picking a round number in microns is not safe here -- it can
    # land on a commensurate period by luck and hide the effect.)
    length_um = grid.n * grid.image_sample_um
    beyond = length_um / (round(length_um / (LAMBDA / NA * 0.8)) + 0.5)
    leaky = sinusoidal_amplitude_grating(grid.n, grid.image_sample_um, beyond)
    snapped = sinusoidal_amplitude_grating(
        grid.n, grid.image_sample_um, commensurate_period(grid.n, grid.image_sample_um, beyond)
    )
    assert contrast(image_coherent(leaky, grid)) > 0.1
    assert contrast(image_coherent(snapped, grid)) == pytest.approx(0.0, abs=1e-9)
