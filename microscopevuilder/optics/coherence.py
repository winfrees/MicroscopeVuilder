"""Partially coherent image formation, by Abbe source integration.

**Why Abbe and not Hopkins.** The plan originally named the Hopkins TCC
formulation. Abbe's method was chosen instead on implementation: it integrates
intensity over source points, each contributing one coherent image, so it costs
``N_source`` FFTs and ``O(n^2)`` memory. The TCC is a four-dimensional kernel costing
``O(n^4)`` memory and only pays for itself when many objects are imaged through one
fixed system -- the opposite of this game, where the player changes the system
constantly and looks at one specimen at a time. Abbe is also exact for the spatially
incoherent Koehler source we actually have, rather than an approximation of it.

The model, in one line per step, for a Koehler-illuminated system:

1. The condenser pupil is an incoherent source. Each source point ``s`` illuminates
   the specimen with a tilted plane wave ``exp(i 2 pi s . x)``.
2. That tilt shifts the specimen spectrum: ``T(f) -> T(f - s)``.
3. The objective pupil filters it, giving a coherent image amplitude
   ``E_s(x) = F^-1{ P(f) T(f - s) }``.
4. Source points are mutually incoherent, so **intensities** add:
   ``I(x) = sum_s |E_s(x)|^2``, weighted by the source distribution.

The coherence parameter is ``S = NA_condenser / NA_objective``. ``S -> 0`` is coherent
illumination (one source point on axis), and the passband widens with ``S``: the
partially coherent cutoff is ``(1 + S) NA_obj / lambda``, so matched illumination
(``S = 1``) reaches ``2 NA / lambda`` -- the full incoherent cutoff, and twice what
coherent illumination delivers. That factor of two is the entire argument against
stopping the condenser down for contrast.

Note ``S > 1`` is not "more incoherent": source points outside the objective pupil
contribute no undiffracted background, so the build slides toward darkfield. The
image does not converge to :func:`image_incoherent` as ``S`` grows.

Specimens are **complex** transmittance ``t = A exp(i phi)``, which is what makes a
pure phase object correctly invisible in brightfield -- the premise of the phase
contrast round.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .psf import PupilGrid, intensity_psf, pupil_function
from .wavefront import Wavefront


@dataclass(frozen=True)
class Source:
    """Sampled condenser pupil: integer pixel shifts plus weights summing to 1."""

    shifts: np.ndarray  # (N, 2) integer (row, col) offsets in frequency samples
    weights: np.ndarray  # (N,) normalized

    def __len__(self) -> int:
        return len(self.weights)


def make_source(
    grid: PupilGrid,
    coherence_parameter: float,
    step_px: int | None = None,
) -> Source:
    """Sample the condenser pupil for a coherence parameter ``S``.

    ``S = 0`` collapses to a single on-axis point (fully coherent). Otherwise the
    source disc of radius ``S * pupil_radius_px`` is sampled on a square lattice and
    clipped to the disc; the lattice step defaults to roughly 8 rings across the
    radius, which is enough for the image to be converged at the couple-of-percent
    level without making the trace sluggish.
    """
    if coherence_parameter < 0:
        raise ValueError("coherence parameter must be non-negative")
    if coherence_parameter == 0:
        return Source(np.zeros((1, 2), dtype=int), np.ones(1))

    radius = coherence_parameter * grid.pupil_radius_px

    # Source points shift the object spectrum by np.roll, which WRAPS. A source
    # point further from the axis than the array can hold folds high frequencies
    # back into the passband and silently fabricates detail.
    if radius + grid.pupil_radius_px > grid.n // 2:
        raise ValueError(
            f"S={coherence_parameter} needs a source radius of {radius:.0f} px, which "
            f"with a {grid.pupil_radius_px} px pupil exceeds the {grid.n // 2} px "
            "half-array and would alias; use a larger n or a smaller pupil_radius_px"
        )

    step = step_px if step_px is not None else max(1, int(round(radius / 8)))

    reach = int(np.ceil(radius / step))
    offsets = np.arange(-reach, reach + 1) * step
    rr, cc = np.meshgrid(offsets, offsets, indexing="ij")
    inside = (rr**2 + cc**2) <= radius**2
    shifts = np.stack([rr[inside], cc[inside]], axis=1).astype(int)
    if len(shifts) == 0:  # radius smaller than one sample: treat as a point source
        shifts = np.zeros((1, 2), dtype=int)
    weights = np.ones(len(shifts)) / len(shifts)
    return Source(shifts, weights)


def make_annular_source(
    grid: PupilGrid,
    inner: float,
    outer: float,
    step_px: int | None = None,
) -> Source:
    """An annular condenser aperture, as phase contrast requires.

    The annulus is what makes the technique work: illumination arrives only at a
    narrow range of angles, so the undiffracted light lands in a matching narrow
    ring at the objective back focal plane, where a phase ring can act on it alone.
    A full disc would spread undiffracted light across the whole pupil and there
    would be nothing to single out.
    """
    outer_radius = outer * grid.pupil_radius_px
    inner_radius = inner * grid.pupil_radius_px
    if outer_radius + grid.pupil_radius_px > grid.n // 2:
        raise ValueError(
            f"an annulus out to S={outer} would alias on a {grid.n} grid; "
            "use a larger n or a smaller pupil_radius_px"
        )

    step = step_px if step_px is not None else max(1, int(round(outer_radius / 12)))
    reach = int(np.ceil(outer_radius / step))
    offsets = np.arange(-reach, reach + 1) * step
    rr, cc = np.meshgrid(offsets, offsets, indexing="ij")
    radius = np.hypot(rr, cc)
    inside = (radius >= inner_radius) & (radius <= outer_radius)
    shifts = np.stack([rr[inside], cc[inside]], axis=1).astype(int)
    if len(shifts) == 0:
        raise ValueError("the annulus is too thin to sample; widen it or reduce step_px")
    return Source(shifts, np.ones(len(shifts)) / len(shifts))


def image_partially_coherent(
    specimen: np.ndarray,
    grid: PupilGrid,
    coherence_parameter: float,
    wavefront: Wavefront | None = None,
    source: Source | None = None,
    pupil_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Form the image of a complex specimen under partially coherent illumination.

    ``specimen`` is complex transmittance sampled on ``grid``'s image-plane spacing
    and referred to image space (magnification is applied by the caller, which keeps
    this function purely optical).
    """
    if specimen.shape != (grid.n, grid.n):
        raise ValueError(f"specimen must be {grid.n}x{grid.n}, got {specimen.shape}")

    src = source if source is not None else make_source(grid, coherence_parameter)
    pupil = pupil_function(grid, wavefront, pupil_mask)
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(specimen)))

    image = np.zeros((grid.n, grid.n), dtype=float)
    for (dr, dc), w in zip(src.shifts, src.weights):
        shifted = np.roll(spectrum, (int(dr), int(dc)), axis=(0, 1))
        amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil * shifted)))
        image += w * np.abs(amplitude) ** 2
    return image


def image_coherent(
    specimen: np.ndarray, grid: PupilGrid, wavefront: Wavefront | None = None
) -> np.ndarray:
    """Fully coherent image: ``I = |F^-1{P T}|^2``. The ``S -> 0`` reference case."""
    pupil = pupil_function(grid, wavefront)
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(specimen)))
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil * spectrum)))
    return np.abs(amplitude) ** 2


def image_incoherent(
    specimen: np.ndarray, grid: PupilGrid, wavefront: Wavefront | None = None
) -> np.ndarray:
    """Fully incoherent image: the intensity object convolved with the intensity PSF.

    The large-``S`` reference case. Note it depends only on ``|t|^2``, which is
    precisely why a phase object cannot be seen this way either: incoherent imaging
    discards the phase before the optics ever act on it.
    """
    psf = intensity_psf(grid, wavefront)
    obj = np.abs(specimen) ** 2
    conv = np.fft.ifft2(np.fft.fft2(obj) * np.fft.fft2(np.fft.ifftshift(psf)))
    return np.real(conv)


def contrast(image: np.ndarray) -> float:
    """Michelson contrast, the number the scorecard grades visibility with."""
    lo, hi = float(image.min()), float(image.max())
    return 0.0 if hi + lo <= 0 else (hi - lo) / (hi + lo)
