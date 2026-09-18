"""Synthetic test objects as complex transmittance ``t = A exp(i phi)``.

Everything is generated rather than loaded, so specimens are exact at any sampling
and the tests have ground truth. All take the image-plane sample spacing in microns
and a size in samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BirefringentField:
    """A specimen described by its birefringence, not by its transmittance.

    A birefringent object has no amplitude structure of its own -- it is invisible
    without polars, which is the whole point of round 14. What it has is a
    retardance and an axis azimuth per pixel, and how much light gets through is
    decided by the polarizer and analyzer *on the bench*. So the specimen carries
    these maps and the renderer applies the Jones calculation with whatever angles
    the player has set, rather than the specimen baking in an answer.
    """

    retardance_waves: np.ndarray
    azimuth_rad: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.retardance_waves.shape


def _coords(n: int, sample_um: float) -> tuple[np.ndarray, np.ndarray]:
    ax = (np.arange(n) - n // 2) * sample_um
    return np.meshgrid(ax, ax, indexing="ij")


def amplitude_bars(
    n: int, sample_um: float, period_um: float, contrast: float = 1.0
) -> np.ndarray:
    """A square-wave amplitude grating: the USAF-target stand-in.

    Visual target only. For resolution measurements use
    :func:`sinusoidal_amplitude_grating` -- see the note there on sampling beats.
    """
    _, x = _coords(n, sample_um)
    bars = (np.sign(np.sin(2 * np.pi * x / period_um)) + 1) / 2
    return (1.0 - contrast + contrast * bars).astype(complex)


def phase_bars(
    n: int, sample_um: float, period_um: float, phase_rad: float = 0.5
) -> np.ndarray:
    """A pure phase grating: unit amplitude everywhere, so brightfield sees nothing.

    Visual target only; see :func:`sinusoidal_phase_grating` for measurements.
    """
    _, x = _coords(n, sample_um)
    bars = (np.sign(np.sin(2 * np.pi * x / period_um)) + 1) / 2
    return np.exp(1j * phase_rad * bars)


def phase_disc(
    n: int, sample_um: float, radius_um: float, phase_rad: float = 0.5
) -> np.ndarray:
    """An unstained cell: a disc of optical path difference, no absorption at all."""
    y, x = _coords(n, sample_um)
    inside = np.hypot(y, x) <= radius_um
    return np.exp(1j * phase_rad * inside)


def two_points(n: int, sample_um: float, separation_um: float) -> np.ndarray:
    """Two pinholes, for measuring resolution directly."""
    t = np.zeros((n, n), dtype=complex)
    half = separation_um / 2 / sample_um
    c = n // 2
    t[c, int(round(c - half))] = 1.0
    t[c, int(round(c + half))] = 1.0
    return t


def sinusoidal_amplitude_grating(
    n: int, sample_um: float, period_um: float, modulation: float = 0.5
) -> np.ndarray:
    """A single-frequency amplitude grating.

    Preferred over :func:`amplitude_bars` for anything measuring resolution. A
    square wave carries harmonics, and -- worse on a sampled grid -- hard-clipping
    ``sign(sin(...))`` at a period that is not an integer number of samples beats
    against the sampling and injects spurious low-frequency content that sails
    through a pupil which should have blocked the bars entirely.
    """
    _, x = _coords(n, sample_um)
    return (1.0 + modulation * np.cos(2 * np.pi * x / period_um)).astype(complex)


def sinusoidal_phase_grating(
    n: int, sample_um: float, period_um: float, phase_rad: float = 0.2
) -> np.ndarray:
    """A single-frequency pure phase grating: constant modulus, modulated phase."""
    _, x = _coords(n, sample_um)
    return np.exp(1j * phase_rad * np.cos(2 * np.pi * x / period_um))


def commensurate_period(n: int, sample_um: float, target_period_um: float) -> float:
    """Snap a period to one that fits a whole number of times across the array.

    The FFT treats the specimen as periodic. A grating whose period does not divide
    the array length has a discontinuity at the wrap, and the resulting spectral
    leakage is broadband -- it puts energy *inside* the pupil even when the grating
    itself is far beyond the cutoff, which looks exactly like resolving something
    that should be invisible. Any quantitative use of a grating should snap first.
    """
    length_um = n * sample_um
    cycles = max(1, round(length_um / target_period_um))
    return length_um / cycles


def birefringent_fibres(
    n: int,
    sample_um: float,
    thickness_um: float = 3.0,
    birefringence: float = 0.02,
    wavelength_um: float = 0.5461,
) -> BirefringentField:
    """Two crossed bundles of birefringent fibres, at different axis angles.

    Collagen, starch, muscle and cellulose all behave this way: an ordered
    molecular axis delays one polarization against the other. The two bundles sit
    at different azimuths so that rotating the stage extinguishes them at
    different angles -- the observation that identifies birefringence as such.
    """
    from ..optics.jones import birefringence_retardance

    axis = (np.arange(n) - n // 2) * sample_um
    y, x = np.meshgrid(axis, axis, indexing="ij")
    retardance = np.zeros((n, n))
    azimuth = np.zeros((n, n))

    delay = birefringence_retardance(thickness_um, birefringence, wavelength_um)
    width = n * sample_um / 12
    for offset, angle in ((-n * sample_um / 7, 0.0), (n * sample_um / 7, np.pi / 4)):
        band = np.abs(y * np.cos(angle) - x * np.sin(angle) - offset) < width / 2
        retardance[band] = delay
        azimuth[band] = angle + np.pi / 4
    return BirefringentField(retardance, azimuth)
