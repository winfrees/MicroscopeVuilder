"""Spectral bands for epi-fluorescence: filters, dichroics and fluorophores.

Modelled as transmission over wavelength with a simple band description rather than
measured curves. What matters pedagogically is the *logic* of a filter set -- that
excitation and emission must not overlap where the dichroic cannot separate them,
and that a few percent of bleedthrough against a weak signal is the difference
between an image and a haze -- not the exact shape of a coating's edge.

Wavelengths are nanometres throughout this module (the convention everyone uses
for filter sets), unlike the micron convention in the diffraction modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EDGE_STEEPNESS_NM = 5.0  # a real interference filter turns over in a few nm


def _edge(wavelength_nm: float, cut_nm: float, rising: bool) -> float:
    """A smooth filter edge: 0 to 1 over a few nanometres."""
    x = (wavelength_nm - cut_nm) / EDGE_STEEPNESS_NM
    s = 1.0 / (1.0 + math.exp(-x))
    return s if rising else 1.0 - s


@dataclass(frozen=True)
class Band:
    """A bandpass region, named by its centre and width as filters are sold."""

    centre_nm: float
    width_nm: float
    peak_transmission: float = 0.95
    blocking: float = 1e-5  # out-of-band leakage: OD5 is a typical real figure

    @property
    def low_nm(self) -> float:
        return self.centre_nm - self.width_nm / 2

    @property
    def high_nm(self) -> float:
        return self.centre_nm + self.width_nm / 2

    def transmission(self, wavelength_nm: float) -> float:
        inside = _edge(wavelength_nm, self.low_nm, True) * _edge(wavelength_nm, self.high_nm, False)
        return self.blocking + (self.peak_transmission - self.blocking) * inside

    def overlaps(self, other: "Band") -> bool:
        return self.low_nm < other.high_nm and other.low_nm < self.high_nm

    def label(self) -> str:
        return f"{self.centre_nm:.0f}/{self.width_nm:.0f}"


@dataclass(frozen=True)
class Dichroic:
    """A long-pass beamsplitter: reflects below the edge, transmits above.

    In an epi-fluorescence cube this is the component doing the real work -- it
    sends excitation down to the specimen and lets emission back up to the eye.
    """

    edge_nm: float
    reflection: float = 0.95
    transmission_peak: float = 0.93

    def reflectance(self, wavelength_nm: float) -> float:
        return self.reflection * _edge(wavelength_nm, self.edge_nm, False)

    def transmittance(self, wavelength_nm: float) -> float:
        return self.transmission_peak * _edge(wavelength_nm, self.edge_nm, True)


@dataclass(frozen=True)
class Fluorophore:
    """Excitation and emission bands, plus how efficiently it converts."""

    name: str
    excitation_peak_nm: float
    emission_peak_nm: float
    excitation_width_nm: float = 40.0
    emission_width_nm: float = 50.0
    brightness: float = 1.0

    @property
    def stokes_shift_nm(self) -> float:
        """The gap the whole technique depends on.

        Without it excitation and emission would be the same colour and no filter
        set could separate them.
        """
        return self.emission_peak_nm - self.excitation_peak_nm

    def excitation_efficiency(self, wavelength_nm: float) -> float:
        return math.exp(
            -0.5 * ((wavelength_nm - self.excitation_peak_nm) / (self.excitation_width_nm / 2.355)) ** 2
        )

    def emission_efficiency(self, wavelength_nm: float) -> float:
        return math.exp(
            -0.5 * ((wavelength_nm - self.emission_peak_nm) / (self.emission_width_nm / 2.355)) ** 2
        )


# A few standard fluorophores, by their commonly quoted peaks.
FLUOROPHORES = {
    "dapi": Fluorophore("DAPI", 358.0, 461.0, 40.0, 60.0, brightness=0.9),
    "fitc": Fluorophore("FITC", 495.0, 519.0, 35.0, 45.0, brightness=1.0),
    "gfp": Fluorophore("EGFP", 488.0, 507.0, 35.0, 45.0, brightness=1.0),
    "tritc": Fluorophore("TRITC", 557.0, 576.0, 35.0, 45.0, brightness=0.8),
    "cy5": Fluorophore("Cy5", 649.0, 670.0, 35.0, 45.0, brightness=0.7),
}


@dataclass(frozen=True)
class FilterCube:
    """An excitation filter, a dichroic and an emission filter, as sold together."""

    excitation: Band
    dichroic: Dichroic
    emission: Band
    name: str = "custom"

    def signal(self, fluorophore: Fluorophore, samples: int = 200) -> float:
        """Relative signal: excitation delivered, times emission collected."""
        delivered = _integrate(
            lambda w: self.excitation.transmission(w)
            * self.dichroic.reflectance(w)
            * fluorophore.excitation_efficiency(w),
            300.0, 800.0, samples,
        )
        collected = _integrate(
            lambda w: fluorophore.emission_efficiency(w)
            * self.dichroic.transmittance(w)
            * self.emission.transmission(w),
            300.0, 800.0, samples,
        )
        return delivered * collected * fluorophore.brightness

    def bleedthrough(self, samples: int = 200) -> float:
        """Excitation light that reaches the detector anyway.

        The number that decides whether a fluorescence image has black background
        or grey haze. Excitation is orders of magnitude brighter than emission, so
        even a small leak dominates a weak signal.
        """
        return _integrate(
            lambda w: self.excitation.transmission(w)
            * self.dichroic.reflectance(w)
            * self.dichroic.transmittance(w)
            * self.emission.transmission(w),
            300.0, 800.0, samples,
        )


def _integrate(f, low: float, high: float, samples: int) -> float:
    step = (high - low) / samples
    return sum(f(low + (i + 0.5) * step) for i in range(samples)) * step


# Standard cubes, named for the fluorophore they serve.
CUBES = {
    "fitc": FilterCube(Band(482.0, 35.0), Dichroic(506.0), Band(536.0, 40.0), "FITC/GFP"),
    "dapi": FilterCube(Band(360.0, 40.0), Dichroic(400.0), Band(460.0, 50.0), "DAPI"),
    "tritc": FilterCube(Band(545.0, 30.0), Dichroic(565.0), Band(605.0, 50.0), "TRITC"),
}
