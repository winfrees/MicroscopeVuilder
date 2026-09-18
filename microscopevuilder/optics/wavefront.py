"""Wavefront aberrations as Zernike coefficients over the exit pupil.

Per docs/PLAN.md decision 2 there is no surface-by-surface ray trace: each element
contributes an aberration budget, and the budgets sum in the pupil. What matters
pedagogically is that *defocus and misalignment are computed, not assigned* -- a
specimen at the wrong height produces a real defocus term, which is why the image
goes soft when the player racks the stage.

Conventions: normalized pupil radius ``rho`` in [0, 1], azimuth ``theta`` in radians,
normalized field height ``h`` in [0, 1]. Coefficients are in microns of wavefront
error, RMS-normalized (Noll), so a total RMS is the quadrature sum of the terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Zernike basis (Noll-indexed, RMS-normalized over the unit disc) ----------
#
# Only the terms a microscope build actually exercises are implemented. Piston and
# tilt are omitted deliberately: they shift the image without degrading it, and
# including them would let a player "fix" a build by moving the field of view.

def _defocus(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(3.0) * (2.0 * rho**2 - 1.0)


def _astig_0(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(6.0) * rho**2 * np.cos(2.0 * theta)


def _astig_45(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(6.0) * rho**2 * np.sin(2.0 * theta)


def _coma_x(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)


def _coma_y(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)


def _spherical(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return math.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)


ZERNIKE = {
    "defocus": _defocus,
    "astigmatism_0": _astig_0,
    "astigmatism_45": _astig_45,
    "coma_x": _coma_x,
    "coma_y": _coma_y,
    "spherical": _spherical,
}


@dataclass
class Wavefront:
    """A set of Zernike coefficients in microns RMS."""

    coefficients: dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.coefficients) - set(ZERNIKE)
        if unknown:
            raise ValueError(f"unknown Zernike terms: {sorted(unknown)}")

    def __add__(self, other: "Wavefront") -> "Wavefront":
        """Budgets from separate elements add coefficient-wise in the shared pupil."""
        merged = dict(self.coefficients)
        for k, v in other.coefficients.items():
            merged[k] = merged.get(k, 0.0) + v
        return Wavefront(merged)

    @property
    def rms_um(self) -> float:
        """Total RMS wavefront error: quadrature sum, because Zernikes are orthogonal."""
        return float(np.sqrt(sum(v * v for v in self.coefficients.values())))

    def dominant_term(self) -> tuple[str, float] | None:
        """The term to blame, for the pupil inspector."""
        if not self.coefficients:
            return None
        k = max(self.coefficients, key=lambda k: abs(self.coefficients[k]))
        return (k, self.coefficients[k]) if self.coefficients[k] else None

    def opd(self, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Optical path difference in microns over the sampled pupil."""
        w = np.zeros_like(rho, dtype=float)
        for name, c in self.coefficients.items():
            if c:
                w += c * ZERNIKE[name](rho, theta)
        return w

    def strehl(self, wavelength_um: float) -> float:
        """Marechal approximation: S = exp(-(2*pi*sigma/lambda)^2).

        Valid while the build is near diffraction-limited (RMS below ~lambda/10);
        beyond that it underestimates, and the scorecard should show the PSF instead
        of quoting a number.
        """
        sigma = self.rms_um
        return float(math.exp(-((2.0 * math.pi * sigma / wavelength_um) ** 2)))

    @property
    def is_diffraction_limited(self) -> bool:
        """Marechal's criterion: RMS <= lambda/14 gives Strehl >= 0.8.

        Expressed against the Rayleigh quarter-wave rule's RMS equivalent, which is
        the form that survives contact with real aberration mixes.
        """
        return self.rms_um <= self._reference_wavelength_um / 14.0

    _reference_wavelength_um: float = 0.5461


def defocus_from_stage_error(dz_mm: float, na: float, wavelength_um: float) -> Wavefront:
    """Turn a focus error into a real defocus coefficient.

    Peak-to-valley defocus for a longitudinal shift ``dz`` is ``W020 = dz * NA^2 / 2``
    in the small-angle limit. The Noll-normalized RMS coefficient is that divided by
    ``2*sqrt(3)``. This is the function that makes racking the stage cost something.
    """
    w020_um = (dz_mm * 1000.0) * na**2 / 2.0
    return Wavefront({"defocus": w020_um / (2.0 * math.sqrt(3.0))})


def defocus_rms_from_shift(shift_um: float, na: float, wavelength_um: float) -> float:
    """RMS defocus coefficient for a longitudinal focus shift given in microns.

    The same relation as :func:`defocus_from_stage_error`, expressed for terms that
    are naturally quoted as a *distance* rather than a wavefront: field curvature
    sag, and the secondary spectrum of an achromat. Keeping them as distances is
    what makes them checkable -- a student can look up "sag" or "longitudinal
    chromatic aberration" and find millimetres, not Zernike coefficients.
    """
    w020_pv = shift_um * na**2 / 2.0
    return w020_pv / (2.0 * math.sqrt(3.0))


def budget_at_field(
    aberrations,
    field_height: float,
    na: float,
    reference_na: float,
    wavelength_um: float,
    reference_wavelength_um: float = 0.5461,
    focal_length_mm: float = 0.0,
) -> Wavefront:
    """Scale a catalog aberration budget to the actual field height, NA and colour.

    Seidel dependences, which is why the budgets are storable as single numbers:

    * **spherical** goes as ``NA^4`` and does not depend on field height at all;
    * **astigmatism** goes as ``h^2 NA^2``;
    * **field curvature** is stored as a longitudinal *sag* in microns at full
      field, going as ``h^2``, and converted to defocus here;
    * **secondary spectrum** is stored as a fraction of the focal length -- the
      textbook figure for an achromat is about ``f/2000`` across the visible band --
      and likewise converted to defocus.

    Storing the last two as distances rather than wavefront coefficients matters:
    both scale as ``NA^2`` when converted, so a high-NA objective is punished for
    the same physical focus error far more than a low-NA one. Folding that into a
    stored constant would have hidden it.

    ``field_height`` is normalized to the design field, ``na`` to ``reference_na``.
    """
    a = max(na / reference_na, 0.0) if reference_na else 0.0
    h = max(field_height, 0.0)
    dlam = abs(wavelength_um - reference_wavelength_um) / reference_wavelength_um

    curvature_shift = aberrations.field_curvature_sag_um * h**2
    # The quoted fraction spans roughly the visible band, dlam ~ 0.2 either side of
    # the correction wavelength, so it is referred to that span.
    chromatic_shift = (
        aberrations.chromatic_focus_fraction * focal_length_mm * 1000.0 * (dlam / 0.2)
    )

    return Wavefront(
        {
            "spherical": aberrations.spherical_rms_um * a**4,
            "astigmatism_0": aberrations.astigmatism_rms_um * h**2 * a**2,
            "defocus": (
                defocus_rms_from_shift(curvature_shift, na, wavelength_um)
                + defocus_rms_from_shift(chromatic_shift, na, wavelength_um)
            ),
        }
    )
