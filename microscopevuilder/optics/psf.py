"""Pupil -> PSF -> OTF, and the diffraction limits that anchor the whole engine.

Sampling scheme, stated once because everything downstream depends on it:

The pupil is sampled on an ``n x n`` grid in which the pupil edge (rho = 1) falls at
radius ``pupil_radius_px``. The pupil edge corresponds to spatial frequency NA/lambda,
so one pupil pixel is ``df = NA / (lambda * pupil_radius_px)``. The FFT of an ``n``
grid then has image-plane sample spacing::

    dx = 1 / (n * df) = lambda * pupil_radius_px / (n * NA)

A useful consequence: the Airy first zero at ``0.61 lambda / NA`` lands at
``0.61 * n / pupil_radius_px`` samples regardless of wavelength or NA, which makes
the sampling easy to reason about and easy to test.

All lengths are microns in this module (not mm) -- it is the natural scale for
diffraction, and the conversion happens at the module boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .wavefront import Wavefront


def abbe_resolution_um(wavelength_um: float, na_objective: float, na_condenser: float = 0.0) -> float:
    """Abbe limit ``d = lambda / (NA_obj + NA_cond)``.

    With the condenser closed down (NA_cond -> 0) this degrades to the coherent
    limit ``lambda / NA``, which is *worse* by a factor of two than matched
    illumination -- the lesson of round 5, and the reason stopping down the
    condenser to get contrast is a trap.
    """
    denom = na_objective + na_condenser
    if denom <= 0:
        raise ValueError("total NA must be positive")
    return wavelength_um / denom


def rayleigh_resolution_um(wavelength_um: float, na: float) -> float:
    """Rayleigh criterion, ``0.61 lambda / NA``: the Airy first zero."""
    if na <= 0:
        raise ValueError("NA must be positive")
    return 0.61 * wavelength_um / na


def mtf_cutoff_cycles_per_um(wavelength_um: float, na: float) -> float:
    """Incoherent diffraction cutoff ``2 NA / lambda``.

    Twice the coherent cutoff, because the OTF is the autocorrelation of the pupil
    and so has twice its radius. Nothing beyond this frequency survives, which is
    what makes empty magnification detectable in round 8.
    """
    return 2.0 * na / wavelength_um


def nyquist_pixel_size_um(wavelength_um: float, na: float, magnification: float) -> float:
    """Largest camera pixel that still samples the optical resolution.

    Two samples per resolved distance, referred to the sensor via the magnification.
    Round 8 fails a build whose pixels are larger than this: the optics resolved it,
    the sensor threw it away.
    """
    return rayleigh_resolution_um(wavelength_um, na) * magnification / 2.0


@dataclass(frozen=True)
class PupilGrid:
    """Sampled pupil coordinates plus the sampling metadata the FFT needs."""

    rho: np.ndarray
    theta: np.ndarray
    mask: np.ndarray
    n: int
    pupil_radius_px: int
    na: float
    wavelength_um: float

    @property
    def image_sample_um(self) -> float:
        """Image-plane sample spacing ``dx``, in microns, per the module docstring."""
        return self.wavelength_um * self.pupil_radius_px / (self.n * self.na)

    @property
    def frequency_sample_cycles_per_um(self) -> float:
        return self.na / (self.wavelength_um * self.pupil_radius_px)


def make_pupil_grid(
    na: float,
    wavelength_um: float,
    n: int = 256,
    pupil_radius_px: int | None = None,
) -> PupilGrid:
    """Build a pupil sampling grid.

    The Airy radius always lands at ``0.61 * n / pupil_radius_px`` samples, so
    ``pupil_radius_px`` is the single knob trading PSF sampling against field of
    view: smaller pupil radius means a finer-sampled PSF over a smaller field.

    The default ``n // 8`` gives ~4.9 samples per Airy radius, which resolves the
    first dark ring properly while keeping the OTF (radius ``2 * pupil_radius_px``)
    well inside the array. ``n // 4`` was tried first and rejected: at 2.4 samples
    per Airy radius the first zero is barely representable, so measured Strehl and
    encircled energy both come out wrong.
    """
    if na <= 0:
        raise ValueError("NA must be positive")
    if n % 2:
        raise ValueError("grid size must be even")
    r_px = pupil_radius_px if pupil_radius_px is not None else n // 8
    if 2 * r_px > n // 2:
        raise ValueError("pupil radius too large: the OTF would alias")

    ax = (np.arange(n) - n // 2) / r_px
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    rho = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    return PupilGrid(rho, theta, rho <= 1.0, n, r_px, na, wavelength_um)


def pupil_function(grid: PupilGrid, wavefront: Wavefront | None = None) -> np.ndarray:
    """Complex pupil ``P = A exp(i 2 pi W / lambda)``, zero outside the aperture."""
    p = grid.mask.astype(complex)
    if wavefront is not None:
        w = wavefront.opd(np.clip(grid.rho, 0.0, 1.0), grid.theta)
        p *= np.exp(2j * math.pi * w / grid.wavelength_um)
        p *= grid.mask
    return p


def amplitude_psf(grid: PupilGrid, wavefront: Wavefront | None = None) -> np.ndarray:
    """Coherent (amplitude) PSF: the Fourier transform of the pupil."""
    p = pupil_function(grid, wavefront)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))


def intensity_psf(grid: PupilGrid, wavefront: Wavefront | None = None) -> np.ndarray:
    """Incoherent PSF, normalized to unit sum so it is a conserving kernel."""
    h = np.abs(amplitude_psf(grid, wavefront)) ** 2
    total = h.sum()
    return h / total if total else h


def otf(grid: PupilGrid, wavefront: Wavefront | None = None) -> np.ndarray:
    """Optical transfer function: the normalized FT of the incoherent PSF.

    Equivalently the autocorrelation of the pupil, which is why its support radius
    is twice the pupil's and the cutoff is ``2 NA / lambda``.
    """
    h = intensity_psf(grid, wavefront)
    o = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h)))
    center = o[grid.n // 2, grid.n // 2]
    return o / center if center != 0 else o


def mtf(grid: PupilGrid, wavefront: Wavefront | None = None) -> np.ndarray:
    """Modulation transfer function: |OTF|."""
    return np.abs(otf(grid, wavefront))


def strehl_from_psf(grid: PupilGrid, wavefront: Wavefront) -> float:
    """Strehl ratio measured from the PSFs rather than approximated.

    The scorecard prefers this over the Marechal formula once the build is far from
    diffraction-limited, where the approximation stops being trustworthy.
    """
    aberrated = intensity_psf(grid, wavefront)
    perfect = intensity_psf(grid, None)
    return float(aberrated.max() / perfect.max())
