"""Synthetic test objects as complex transmittance ``t = A exp(i phi)``.

Everything is generated rather than loaded, so specimens are exact at any sampling
and the tests have ground truth. All take the image-plane sample spacing in microns
and a size in samples.
"""

from __future__ import annotations

import numpy as np


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
