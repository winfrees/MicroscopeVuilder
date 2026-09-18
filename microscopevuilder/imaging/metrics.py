"""Measurements taken from a rendered image.

These are what turn the generated image from decoration into evidence. A round that
asks for "corner MTF within spec" or "chromatic error under tolerance" is asking for
a number measured off the picture, not a property of the ray geometry, so the
scorecard needs these before rounds 6-8 can be graded honestly.

Everything here is a pure function of an intensity array plus its sampling, so the
measurements are testable against analytic cases without a bench in sight.
"""

from __future__ import annotations

import numpy as np


def michelson_contrast(image: np.ndarray) -> float:
    """``(max - min) / (max + min)``: the visibility of whatever is in the frame."""
    lo, hi = float(image.min()), float(image.max())
    return 0.0 if hi + lo <= 0 else (hi - lo) / (hi + lo)


def modulation_at_period(
    image: np.ndarray, sample_um: float, period_um: float, axis: int = 1
) -> float:
    """Modulation depth at one spatial frequency, read straight off the FFT.

    For a sinusoidal test grating this *is* the MTF at that frequency: the ratio of
    the modulation that survived to the modulation that went in, once normalized by
    the input. Measuring one known frequency rather than fitting the whole curve
    keeps it robust against the leakage that bedevils sampled gratings.
    """
    if period_um <= 0:
        raise ValueError("period must be positive")
    profile = image.mean(axis=1 - axis)
    n = profile.size
    spectrum = np.fft.rfft(profile - profile.mean())
    dc = float(np.abs(profile.mean())) * n

    cycles = n * sample_um / period_um
    bin_index = int(round(cycles))
    if bin_index < 1 or bin_index >= spectrum.size:
        raise ValueError(
            f"a {period_um:.3f} um period is not representable in a {n}-sample "
            f"profile at {sample_um:.4f} um/sample"
        )
    # Take the strongest of the neighbouring bins: a period that does not divide the
    # array exactly spreads its energy across two.
    lo = max(bin_index - 1, 1)
    amplitude = float(np.abs(spectrum[lo : bin_index + 2]).max())
    return 0.0 if dc == 0 else 2.0 * amplitude / dc


def field_uniformity(image: np.ndarray, centre_fraction: float = 0.25) -> float:
    """Corner brightness as a fraction of centre brightness.

    1.0 is an evenly lit field; lower means the edges are falling off, whether from
    vignetting or from illumination that is not Köhler. This is the number round 4
    is really about, measured rather than inferred.
    """
    n = image.shape[0]
    half = max(int(n * centre_fraction / 2), 1)
    c = n // 2
    centre = float(image[c - half : c + half, c - half : c + half].mean())
    corner = float(
        np.mean([
            image[:half * 2, :half * 2].mean(),
            image[:half * 2, -half * 2:].mean(),
            image[-half * 2:, :half * 2].mean(),
            image[-half * 2:, -half * 2:].mean(),
        ])
    )
    return 0.0 if centre <= 0 else corner / centre


def estimate_noise(
    image: np.ndarray, sample_um: float | None = None, cutoff_cycles_per_um: float | None = None
) -> float:
    """Noise standard deviation, separated from the picture by spatial frequency.

    The optics band-limit the image: nothing real survives past ``2 NA / lambda``.
    Any power beyond that cutoff is therefore noise, and measuring it there
    separates grain from detail cleanly.

    Two cruder estimators were tried and rejected. The spread within a dark region
    fails on a photon-starved image, where dark pixels quantize to exactly zero and
    the SNR comes back infinite on the noisiest picture. The spread of adjacent-pixel
    differences fails on any structured specimen, because a fine grating's own
    modulation is counted as noise -- it reported SNR 6 on an image with 4500
    photons per pixel.
    """
    data = np.asarray(image, dtype=float)
    if data.size < 16:
        return 0.0

    if sample_um and cutoff_cycles_per_um and cutoff_cycles_per_um > 0:
        n = data.shape[0]
        frequencies = np.fft.fftfreq(n, d=sample_um)
        fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
        beyond = np.hypot(fy, fx) > cutoff_cycles_per_um
        if beyond.any():
            # Window first. Without it the array edges leak broadband energy past
            # the cutoff and the estimator reports a noise floor on a noiseless
            # image -- measured at 0.011 against a true zero.
            window = np.hanning(n)
            windowed = (data - data.mean()) * np.outer(window, window)
            # A Hann window removes power, so normalize by its RMS to keep the
            # variance estimate unbiased.
            windowed /= np.sqrt((window**2).mean() ** 2)
            spectrum = np.fft.fft2(windowed)
            # Parseval: the variance carried by the out-of-band bins is the noise
            # variance in those bins, and white noise spreads evenly across all of
            # them, so scale up by the fraction of the spectrum sampled.
            out_of_band_power = float((np.abs(spectrum[beyond]) ** 2).sum())
            total_bins = data.size
            fraction = beyond.sum() / total_bins
            if fraction > 0:
                return float(np.sqrt(out_of_band_power / (total_bins**2) / fraction))

    differences = np.diff(data, axis=1)
    return float(differences.std()) / np.sqrt(2.0)


def signal_to_noise(
    image: np.ndarray,
    background_fraction: float = 0.1,
    sample_um: float | None = None,
    cutoff_cycles_per_um: float | None = None,
) -> float:
    """Signal range over the noise, making "too dim to see" measurable."""
    noise = estimate_noise(image, sample_um, cutoff_cycles_per_um)
    flat = np.sort(np.asarray(image, dtype=float).ravel())
    cut = max(int(flat.size * background_fraction), 2)
    signal = float(flat[-cut:].mean() - flat[:cut].mean())
    if noise <= 0:
        return float("inf") if signal > 0 else 0.0
    return signal / noise


def resolved_period_um(
    modulations: dict[float, float], threshold: float = 0.10
) -> float | None:
    """The finest period whose modulation clears the threshold.

    The 10% default is the usual working definition of "resolved" for an MTF
    measurement -- below it the modulation is indistinguishable from noise on a real
    detector. Returns None when nothing on the list was resolved.
    """
    resolved = [p for p, m in modulations.items() if m >= threshold]
    return min(resolved) if resolved else None


def lateral_shift_px(a: np.ndarray, b: np.ndarray) -> float:
    """Radial shift between two images, by cross-correlation peak.

    Lateral colour is exactly this: the blue image and the red image are at
    different scales, so off-axis detail lands in different places and edges grow
    coloured fringes.
    """
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    correlation = np.fft.fftshift(np.real(np.fft.ifft2(fa * np.conj(fb))))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    centre = (correlation.shape[0] // 2, correlation.shape[1] // 2)
    return float(np.hypot(peak[0] - centre[0], peak[1] - centre[1]))
