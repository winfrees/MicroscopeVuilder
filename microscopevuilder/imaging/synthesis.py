"""Render the image a specific bench actually forms.

The entry point M10 exists to build. Given a bench and a specimen, this resolves the
build's optical state (`build_optics`), forms the partially coherent image through
the *resolved* pupil -- aberrations, defocus and all -- and applies photometry, so
that a dim build is visibly noisy and an aberrated build is visibly soft.

Synthesis happens in **object space**: the grid is built at the objective's NA and
the specimen is in specimen coordinates. Magnification does not belong in the
picture; it belongs in the sampling check, where it decides whether the detector
kept what the optics resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..bench.bench import Bench
from ..optics.coherence import image_partially_coherent
from ..optics.psf import make_pupil_grid
from .build_optics import BuildOptics, resolve_build_optics

SpecimenFactory = Callable[[int, float], np.ndarray]


@dataclass
class RenderSpec:
    """How to render: what to look at, at what field position, in what light."""

    specimen: SpecimenFactory
    n: int = 256
    field_height: float = 0.0
    wavelength_um: float = 0.5461
    coherence_parameter: float = 0.6
    detector_name: str = "sensor"
    objective_name: str = "objective"
    exposure_photons: float = 0.0  # 0 disables noise
    seed: int = 0


@dataclass
class RenderedImage:
    intensity: np.ndarray
    sample_um: float
    optics: BuildOptics
    notes: list[str] = field(default_factory=list)

    @property
    def extent_um(self) -> float:
        return self.intensity.shape[0] * self.sample_um


def render_build(bench: Bench, s_object: float, spec: RenderSpec) -> RenderedImage:
    """Form the image this bench makes of this specimen."""
    optics = resolve_build_optics(
        bench,
        s_object,
        spec.detector_name,
        spec.wavelength_um,
        spec.objective_name,
        spec.field_height,
        spec.coherence_parameter,
    )

    grid = make_pupil_grid(max(optics.na, 0.02), spec.wavelength_um, n=spec.n)
    specimen = spec.specimen(spec.n, grid.image_sample_um)
    intensity = image_partially_coherent(
        specimen, grid, optics.coherence_parameter, optics.wavefront
    )

    notes = list(optics.notes)
    if spec.exposure_photons > 0:
        intensity, note = _apply_photometry(
            intensity, optics.relative_irradiance, spec.exposure_photons, spec.seed
        )
        if note:
            notes.append(note)

    return RenderedImage(intensity, grid.image_sample_um, optics, notes)


def _apply_photometry(
    intensity: np.ndarray, relative_irradiance: float, exposure_photons: float, seed: int
) -> tuple[np.ndarray, str]:
    """Turn relative irradiance into shot noise.

    Photon count scales with irradiance and SNR with its square root, so a build
    that is optically correct but dim comes out grainy rather than merely darker.
    That is the honest rendering of "technically right, too dim to use".
    """
    photons = intensity * exposure_photons * max(relative_irradiance, 0.0)
    if photons.max() <= 0:
        return intensity * 0.0, "no light reaches the detector"

    rng = np.random.default_rng(seed)
    detected = rng.poisson(np.clip(photons, 0.0, None)).astype(float)
    peak = photons.max()
    note = ""
    if peak < 25:
        note = f"only {peak:.0f} photons in the brightest pixel: this build is noise-limited"
    return detected / max(exposure_photons, 1e-12), note


def measure_mtf(
    bench: Bench,
    s_object: float,
    spec: RenderSpec,
    periods_um: list[float],
    grating: Callable[[int, float, float], np.ndarray] | None = None,
) -> dict[float, float]:
    """Modulation transfer measured by imaging gratings through the real build.

    Not read off an analytic curve: each period is rendered through this bench's
    pupil and the surviving modulation is measured, so aberrations, defocus and
    partial coherence all show up in the answer.
    """
    from .metrics import modulation_at_period
    from .specimens import commensurate_period, sinusoidal_amplitude_grating

    maker = grating or sinusoidal_amplitude_grating
    out: dict[float, float] = {}
    for target in periods_um:
        def specimen(n: int, sample_um: float, target=target, maker=maker):
            return maker(n, sample_um, commensurate_period(n, sample_um, target))

        rendered = render_build(
            bench, s_object, dataclasses_replace(spec, specimen=specimen, exposure_photons=0.0)
        )
        period = commensurate_period(spec.n, rendered.sample_um, target)
        out[period] = modulation_at_period(rendered.intensity, rendered.sample_um, period)
    return out


def dataclasses_replace(spec: RenderSpec, **changes) -> RenderSpec:
    import dataclasses

    return dataclasses.replace(spec, **changes)
