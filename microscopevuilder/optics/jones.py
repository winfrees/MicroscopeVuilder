"""Jones calculus: polarization as a two-component complex vector.

Rounds 14 and 15 need light to carry a polarization state, not just an amplitude.
The Jones formalism is the right level for this game: it handles fully polarized
light exactly, composes as matrix products, and stays cheap enough to evaluate per
pixel. (It cannot express partial polarization or depolarization -- that needs
Mueller calculus -- which is a limit worth stating rather than discovering.)

Convention: ``[Ex, Ey]`` with x horizontal. Retarders are defined by their
retardance in waves and the azimuth of their fast axis.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

HORIZONTAL = np.array([1.0, 0.0], dtype=complex)
VERTICAL = np.array([0.0, 1.0], dtype=complex)
DIAGONAL = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
CIRCULAR_RIGHT = np.array([1.0, -1j], dtype=complex) / math.sqrt(2.0)


def rotation(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, s], [-s, c]], dtype=complex)


def rotated(matrix: np.ndarray, angle_rad: float) -> np.ndarray:
    """Express a component rotated to a given azimuth: ``R(-a) M R(a)``."""
    return rotation(-angle_rad) @ matrix @ rotation(angle_rad)


def linear_polarizer(angle_rad: float = 0.0, extinction: float = 1e-4) -> np.ndarray:
    """A linear polarizer at ``angle_rad``.

    ``extinction`` is the intensity leak through crossed polarizers; a real sheet
    polarizer manages 1e-4 or so, and a calcite prism far better. It is not zero,
    which matters: round 14 is partly about how good "extinct" really gets.
    """
    amplitude_leak = math.sqrt(extinction)
    base = np.array([[1.0, 0.0], [0.0, amplitude_leak]], dtype=complex)
    return rotated(base, angle_rad)


def retarder(retardance_waves: float, fast_axis_rad: float = 0.0) -> np.ndarray:
    """A linear retarder: a wave plate, or a birefringent specimen.

    Retardance is in waves, so 0.25 is a quarter-wave plate and 0.5 a half-wave.
    """
    phase = cmath.exp(1j * math.pi * retardance_waves)
    base = np.array([[1.0 / phase, 0.0], [0.0, phase]], dtype=complex)
    return rotated(base, fast_axis_rad)


def intensity(state: np.ndarray) -> float:
    """``|Ex|^2 + |Ey|^2``."""
    return float(np.sum(np.abs(np.asarray(state)) ** 2))


def transmit(state: np.ndarray, *components: np.ndarray) -> np.ndarray:
    """Send a state through components in the order light meets them."""
    out = np.asarray(state, dtype=complex)
    for component in components:
        out = component @ out
    return out


def birefringence_retardance(
    thickness_um: float, birefringence: float, wavelength_um: float
) -> float:
    """Retardance in waves for a birefringent sample: ``t * dn / lambda``.

    This is how a specimen's structure becomes visible between crossed polars: a
    collagen fibre or a starch grain has an ordered molecular axis, so it delays one
    polarization relative to the other, and the analyzer turns that delay into
    brightness.
    """
    if wavelength_um <= 0:
        raise ValueError("wavelength must be positive")
    return thickness_um * birefringence / wavelength_um


def field_through(
    retardance: np.ndarray,
    azimuth: np.ndarray,
    polarizer_rad: float,
    analyzer_rad: float,
    extinction: float = 1e-4,
) -> np.ndarray:
    """Intensity image of a birefringent specimen between polarizer and analyzer.

    ``retardance`` and ``azimuth`` are per-pixel maps: how much the specimen delays
    one axis against the other, and which way that axis points. The classic result
    falls out -- a birefringent object goes dark four times per full rotation,
    whenever its axis lines up with either polarizer.
    """
    retardance = np.asarray(retardance, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    out = np.zeros(retardance.shape, dtype=float)

    polarizer = linear_polarizer(polarizer_rad, extinction)
    analyzer = linear_polarizer(analyzer_rad, extinction)
    incident = polarizer @ HORIZONTAL

    # Vectorized over the two distinct parameters rather than per pixel: the Jones
    # matrix depends only on (retardance, azimuth), so identical pairs share work.
    flat_r = retardance.ravel()
    flat_a = azimuth.ravel()
    result = np.empty(flat_r.size, dtype=float)
    for index in range(flat_r.size):
        sample = retarder(float(flat_r[index]), float(flat_a[index]))
        result[index] = intensity(analyzer @ sample @ incident)
    return result.reshape(retardance.shape) if out.size else out
