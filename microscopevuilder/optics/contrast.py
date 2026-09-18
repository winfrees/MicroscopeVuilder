"""Differential interference contrast: shear, bias, and the gradient image.

DIC splits the illumination into two orthogonally polarized beams, displaces them
laterally by a small **shear** (a fraction of the resolution limit), passes both
through the specimen, and recombines them. What survives at the analyzer is the
*difference* between what the two beams saw -- so the image renders the gradient of
optical path, which is why a DIC image looks like raked lighting across a relief.

Where the shear is applied, and why it is legitimate here: physically the two beams
are sheared before the specimen and recombined after the objective. Imaging is
linear in complex amplitude, so imaging the difference equals the difference of the
images, and applying the shear at the specimen gives the same answer for coherent
imaging exactly, and a good approximation under partial coherence. Stating it this
way keeps one FFT instead of two.
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def shear_field(
    specimen: np.ndarray,
    shear_px: float,
    bias_waves: float = 0.15,
    axis: int = 1,
) -> np.ndarray:
    """Recombine two sheared, bias-retarded copies of a complex specimen.

    ``bias_waves`` is the compensator setting. It matters more than it looks: at
    zero bias the image is symmetric and dark-field-like, because equal and opposite
    gradients give the same intensity. A bias of roughly an eighth of a wave puts
    the working point on the steep part of the sine, which is what produces the
    familiar shadowed relief with a mid-grey background and tells you which way a
    gradient runs.
    """
    if shear_px <= 0:
        raise ValueError("shear must be positive")

    half = shear_px / 2.0
    forward = _subpixel_shift(specimen, +half, axis)
    backward = _subpixel_shift(specimen, -half, axis)

    phase = cmath.exp(1j * math.pi * bias_waves)
    return (forward * phase - backward / phase) / 2.0


def _subpixel_shift(field: np.ndarray, shift_px: float, axis: int) -> np.ndarray:
    """Shift by a possibly fractional number of pixels, via the Fourier shift theorem."""
    n = field.shape[axis]
    frequencies = np.fft.fftfreq(n)
    ramp = np.exp(-2j * np.pi * frequencies * shift_px)
    shape = [1] * field.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(field, axis=axis) * ramp.reshape(shape), axis=axis)


def shear_for_resolution(resolution_um: float, sample_um: float, fraction: float = 0.6) -> float:
    """A shear expressed as a fraction of the resolution limit, in pixels.

    Real DIC prisms shear by somewhat less than the resolution limit: enough to
    generate a difference, little enough not to produce a visible double image. A
    shear larger than the resolution limit turns the relief into two overlapping
    outlines, which is the classic sign of the wrong prism for the objective.
    """
    if sample_um <= 0:
        raise ValueError("sample spacing must be positive")
    return fraction * resolution_um / sample_um
