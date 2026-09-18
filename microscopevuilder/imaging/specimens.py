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


def usaf_bars(
    n: int, sample_um: float, period_um: float, groups: int = 3
) -> np.ndarray:
    """A USAF-1951-style element: three bars, in both orientations.

    The standard resolution target. Having both orientations in one field is the
    point -- astigmatism and DIC shear both resolve one direction and not the
    other, and a single-orientation grating hides that.
    """
    y, x = _coords(n, sample_um)
    bar = period_um / 2.0
    span = groups * period_um

    horizontal = (np.abs(y) < span / 2) & (np.abs(x - span) < span / 2)
    vertical = (np.abs(x + span) < span / 2) & (np.abs(y) < span / 2)
    pattern = np.ones((n, n))
    pattern[horizontal & (np.mod(y + span, period_um) < bar)] = 0.0
    pattern[vertical & (np.mod(x + span, period_um) < bar)] = 0.0
    return pattern.astype(complex)


def siemens_star(n: int, sample_um: float, spokes: int = 36) -> np.ndarray:
    """A radial star: every spatial frequency at once, in every direction.

    Resolution falls off toward the centre, so the radius at which the spokes blur
    together reads the limit straight off the image.
    """
    y, x = _coords(n, sample_um)
    theta = np.arctan2(y, x)
    radius = np.hypot(y, x)
    pattern = (np.cos(spokes * theta / 2.0) > 0).astype(float)
    pattern[radius > (n * sample_um / 2.4)] = 1.0
    return pattern.astype(complex)


def bead_field(
    n: int, sample_um: float, diameter_um: float = 0.5, count: int = 40, seed: int = 0
) -> np.ndarray:
    """Scattered sub-resolution beads: point sources, for reading the PSF directly."""
    rng = np.random.default_rng(seed)
    y, x = _coords(n, sample_um)
    pattern = np.zeros((n, n))
    extent = n * sample_um * 0.4
    for _ in range(count):
        cy, cx = rng.uniform(-extent, extent, size=2)
        pattern[np.hypot(y - cy, x - cx) <= diameter_um / 2] = 1.0
    return pattern.astype(complex)


def stained_section(n: int, sample_um: float, seed: int = 1) -> np.ndarray:
    """A stained histology section: irregular absorbing structure, no phase."""
    rng = np.random.default_rng(seed)
    y, x = _coords(n, sample_um)
    field = np.ones((n, n))
    extent = n * sample_um * 0.45
    for _ in range(26):
        cy, cx = rng.uniform(-extent, extent, size=2)
        radius = rng.uniform(1.0, 3.5)
        depth = rng.uniform(0.25, 0.75)
        field -= depth * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * radius**2))
    return np.clip(field, 0.05, 1.0).astype(complex)


def polished_metal(n: int, sample_um: float, seed: int = 2) -> np.ndarray:
    """An opaque, reflective surface with scratches: the epi-brightfield specimen."""
    rng = np.random.default_rng(seed)
    y, x = _coords(n, sample_um)
    field = np.full((n, n), 0.85)
    for _ in range(14):
        angle = rng.uniform(0, np.pi)
        offset = rng.uniform(-n * sample_um / 3, n * sample_um / 3)
        width = rng.uniform(0.2, 0.8)
        line = np.abs(y * np.cos(angle) - x * np.sin(angle) - offset)
        field -= 0.6 * np.exp(-(line**2) / (2 * width**2))
    return np.clip(field, 0.02, 1.0).astype(complex)


def ronchi_ruling(n: int, sample_um: float, period_um: float = 2.0) -> np.ndarray:
    """A square-wave ruling for checking magnification and distortion."""
    return amplitude_bars(n, sample_um, period_um)


SPECIMEN_LIBRARY = {
    "USAF target": lambda n, dx: usaf_bars(n, dx, period_um=max(dx * 6, 0.6)),
    "Siemens star": siemens_star,
    "Ronchi ruling": lambda n, dx: ronchi_ruling(n, dx, period_um=max(dx * 10, 1.0)),
    "sine grating": lambda n, dx: sinusoidal_amplitude_grating(
        n, dx, commensurate_period(n, dx, max(dx * 10, 1.0))
    ),
    "beads": lambda n, dx: bead_field(n, dx, diameter_um=max(dx, 0.3)),
    "stained section": stained_section,
    "unstained cell": lambda n, dx: phase_disc(n, dx, radius_um=n * dx / 6, phase_rad=0.6),
    "birefringent fibres": birefringent_fibres,
    "polished metal": polished_metal,
}
