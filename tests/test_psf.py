"""M2 gate: the diffraction engine must reproduce closed-form optics."""

import math

import numpy as np
import pytest

from microscopevuilder.optics.psf import (
    abbe_resolution_um,
    amplitude_psf,
    intensity_psf,
    make_pupil_grid,
    mtf,
    mtf_cutoff_cycles_per_um,
    nyquist_pixel_size_um,
    rayleigh_resolution_um,
    strehl_from_psf,
)
from microscopevuilder.optics.wavefront import (
    Wavefront,
    defocus_from_stage_error,
)

LAMBDA = 0.5461


def _radial_profile(img):
    n = img.shape[0]
    c = n // 2
    return img[c, c:]


def _first_minimum_px(profile):
    """Index of the first genuine local minimum, ignoring numerical fuzz."""
    floor = profile.max() * 1e-9
    for i in range(1, len(profile) - 1):
        if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]:
            if profile[i] < floor or profile[i + 1] > profile[i]:
                return i
    raise AssertionError("no minimum found in profile")


def test_psf_matches_the_airy_pattern():
    # The unaberrated PSF must be the Airy function, first zero at 0.61 lambda / NA.
    na = 0.5
    # Oversampled deliberately: locating a zero needs far finer sampling than
    # forming an image does.
    g = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=16)
    prof = _radial_profile(intensity_psf(g))
    first_zero_px = _first_minimum_px(prof)

    expected_px = rayleigh_resolution_um(LAMBDA, na) / g.image_sample_um
    # Tolerance is one sample: the true zero at 19.52 px cannot land on an integer
    # index, so anything tighter would be testing the rounding, not the optics.
    assert first_zero_px == pytest.approx(expected_px, abs=1.0)


def test_airy_zero_lands_at_the_same_pixel_for_any_na_or_wavelength():
    # Consequence of the sampling scheme: 0.61 * n / pupil_radius_px, nothing else.
    expected = 0.61 * 512 / 128
    for na, lam in ((0.1, 0.4), (0.75, 0.55), (1.4, 0.68)):
        g = make_pupil_grid(na, lam, n=512, pupil_radius_px=128)
        assert rayleigh_resolution_um(lam, na) / g.image_sample_um == pytest.approx(expected)


def test_encircled_energy_in_the_airy_disc_is_about_84_percent():
    # The classic number. A PSF that fails this is not an Airy pattern.
    na = 0.5
    g = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=16)
    h = intensity_psf(g)
    n = g.n
    ax = np.arange(n) - n // 2
    rr = np.hypot(*np.meshgrid(ax, ax, indexing="xy"))
    r_airy = rayleigh_resolution_um(LAMBDA, na) / g.image_sample_um
    assert h[rr <= r_airy].sum() == pytest.approx(0.838, abs=0.02)


def test_mtf_cutoff_is_twice_the_coherent_limit():
    na = 0.4
    g = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=96)
    m = mtf(g)
    prof = _radial_profile(m)

    cutoff_px = mtf_cutoff_cycles_per_um(LAMBDA, na) / g.frequency_sample_cycles_per_um
    assert cutoff_px == pytest.approx(2 * g.pupil_radius_px)
    # Non-negligible well inside the cutoff, gone beyond it.
    assert prof[int(0.5 * cutoff_px)] > 0.05
    assert prof[int(1.1 * cutoff_px)] < 1e-6


def test_diffraction_limited_mtf_matches_the_analytic_form():
    # MTF(s) = (2/pi)(acos(s) - s sqrt(1 - s^2)) for normalized frequency s.
    na = 0.4
    g = make_pupil_grid(na, LAMBDA, n=512, pupil_radius_px=96)
    prof = _radial_profile(mtf(g))
    cutoff_px = 2 * g.pupil_radius_px

    for frac in (0.2, 0.4, 0.6, 0.8):
        s = frac
        analytic = (2 / math.pi) * (math.acos(s) - s * math.sqrt(1 - s * s))
        assert prof[int(round(frac * cutoff_px))] == pytest.approx(analytic, abs=0.02)


def test_abbe_limit_beats_the_coherent_limit_by_two():
    # Matched condenser NA doubles the resolvable frequency. Round 5 in one assert.
    matched = abbe_resolution_um(LAMBDA, 0.8, 0.8)
    closed = abbe_resolution_um(LAMBDA, 0.8, 0.0)
    assert closed / matched == pytest.approx(2.0)


def test_strehl_measured_from_psf_agrees_with_marechal_when_small():
    # The approximation is only claimed for near-diffraction-limited builds, so the
    # two estimates must agree there and are allowed to diverge beyond.
    g = make_pupil_grid(0.6, LAMBDA, n=512, pupil_radius_px=128)
    small = Wavefront({"spherical": 0.02})
    assert strehl_from_psf(g, small) == pytest.approx(small.strehl(LAMBDA), abs=0.02)


def test_aberrations_only_ever_reduce_the_peak():
    g = make_pupil_grid(0.6, LAMBDA, n=256, pupil_radius_px=64)
    for w in (
        Wavefront({"spherical": 0.05}),
        Wavefront({"coma_x": 0.05}),
        Wavefront({"astigmatism_0": 0.05}),
        Wavefront({"defocus": 0.05}),
    ):
        assert strehl_from_psf(g, w) < 1.0


def test_defocus_from_stage_error_is_symmetric_and_grows_with_na():
    # Focus error hurts a high-NA objective far more: W020 goes as NA^2.
    low = defocus_from_stage_error(0.001, 0.25, LAMBDA).rms_um
    high = defocus_from_stage_error(0.001, 0.75, LAMBDA).rms_um
    assert high / low == pytest.approx((0.75 / 0.25) ** 2)
    # Defocusing the other way is equally bad.
    assert defocus_from_stage_error(-0.001, 0.5, LAMBDA).rms_um == pytest.approx(
        defocus_from_stage_error(0.001, 0.5, LAMBDA).rms_um
    )


def test_quarter_wave_defocus_gives_the_textbook_strehl():
    # Rayleigh's quarter-wave rule: PV defocus of lambda/4 leaves Strehl ~0.8.
    na, lam = 0.5, LAMBDA
    dz_mm = (lam / 4.0) * 2.0 / na**2 / 1000.0
    w = defocus_from_stage_error(dz_mm, na, lam)
    g = make_pupil_grid(na, lam, n=512, pupil_radius_px=128)
    assert strehl_from_psf(g, w) == pytest.approx(0.8, abs=0.05)


def test_psf_is_energy_normalized():
    g = make_pupil_grid(0.4, LAMBDA, n=256, pupil_radius_px=64)
    assert intensity_psf(g).sum() == pytest.approx(1.0)
    assert intensity_psf(g, Wavefront({"coma_x": 0.1})).sum() == pytest.approx(1.0)


def test_nyquist_pixel_size_flags_empty_magnification():
    # A CFI60 20x/0.75 resolves 0.444 um, which is 8.9 um at the sensor; Nyquist
    # therefore demands pixels no larger than 4.44 um. Worth stating plainly: the
    # very common 6.5 um scientific CMOS pixel UNDERSAMPLES this objective, and
    # round 8 is supposed to make the player discover exactly that.
    limit = nyquist_pixel_size_um(LAMBDA, 0.75, 20.0)
    assert limit == pytest.approx(4.44, abs=0.01)
    assert limit < 6.5  # the sCMOS trap

    # Adding a 1.5x tube-lens changer rescues it, at the cost of field of view.
    assert nyquist_pixel_size_um(LAMBDA, 0.75, 30.0) > 6.5


def test_pupil_grid_rejects_sampling_that_would_alias_the_otf():
    with pytest.raises(ValueError):
        make_pupil_grid(0.5, LAMBDA, n=256, pupil_radius_px=100)
    with pytest.raises(ValueError):
        make_pupil_grid(0.0, LAMBDA)
