"""Resolve a bench into the optical state that image synthesis needs.

This module is the missing link the plan calls out in M10. Everything downstream --
the PSF, the partially coherent image, the measured metrics -- has existed since M2,
but nothing connected them to the *player's build*. Synthesis took four loose scalars
and rendered a generic image; a plan objective and a non-plan objective produced
identical pictures, which made the catalog's central teaching distinction invisible.

What gets resolved here, all of it from the bench rather than from arguments:

* numerical aperture and magnification, from the paraxial trace;
* the aberration budget, from the objective's catalog entry, scaled to the field
  height and aperture actually in use;
* defocus, *computed* from where the image really lands versus where the detector
  really is -- so racking the stage costs something, as decision 2 requires;
* photometry: relative irradiance, which falls as ``NA^2 / M^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bench.bench import Bench
from ..bench.catalog import Aberrations, load_catalog
from ..optics.wavefront import Wavefront, budget_at_field, defocus_from_stage_error

REFERENCE_WAVELENGTH_UM = 0.5461

# Irradiance is reported relative to a plain 10x/0.25 brightfield build rather than
# in absolute units. NA^2/M^2 on its own is a tiny number for any real objective,
# which would make every build look starved and tell the player nothing; against a
# familiar reference it says the useful thing -- this build is twice as bright as a
# 10x, or a third as bright.
REFERENCE_NA = 0.25
REFERENCE_MAGNIFICATION = 10.0
_REFERENCE_THROUGHPUT = (REFERENCE_NA / REFERENCE_MAGNIFICATION) ** 2


@dataclass
class BuildOptics:
    """Everything about a build that determines the image it forms."""

    na: float
    magnification: float
    wavelength_um: float
    coherence_parameter: float
    wavefront: Wavefront
    relative_irradiance: float  # 1.0 == a 10x/0.25 brightfield build
    image_defocus_mm: float
    object_defocus_mm: float
    field_height: float
    aberration_source: str
    notes: list[str] = field(default_factory=list)

    @property
    def dominant_aberration(self) -> tuple[str, float] | None:
        return self.wavefront.dominant_term()

    def describe(self) -> str:
        bits = [f"NA {self.na:.3f}", f"{abs(self.magnification):.1f}x"]
        dominant = self.dominant_aberration
        if dominant:
            bits.append(f"dominated by {dominant[0]} ({dominant[1] * 1000:.0f} nm RMS)")
        bits.append(f"Strehl {self.wavefront.strehl(self.wavelength_um):.2f}")
        return ", ".join(bits)


def _catalog_aberrations(bench: Bench, objective_name: str) -> tuple[Aberrations, float, str]:
    """Find the objective's aberration budget, and say where it came from.

    Preference order: an explicit catalog key, then the nearest entry of the same
    correction grade and aperture, then a perfect-lens fallback. The source is
    returned so the UI can be honest about which of the three it is looking at --
    a perfect lens is a useful default but must never be mistaken for a real one.

    The grade is required for the nearest-match path. Matching on aperture alone
    would silently attribute a correction grade the player never chose: an objective
    declaring only "NA 0.4" would pick up an achromat's secondary spectrum and image
    badly in blue for reasons nothing on the bench explains.
    """
    objective = bench.get(objective_name)
    catalog = load_catalog()

    if objective.catalog_key and objective.catalog_key in catalog.objectives:
        entry = catalog.objective(objective.catalog_key)
        return entry.aberrations, entry.na, f"catalog:{entry.key}"

    grade = objective.metadata.get("grade")
    na = float(objective.metadata.get("na", 0.0))
    candidates = [o for o in catalog.objectives.values() if o.grade == grade] if grade else []
    if candidates and na > 0:
        nearest = min(candidates, key=lambda o: abs(o.na - na))
        return nearest.aberrations, nearest.na, f"nearest:{nearest.key}"

    return Aberrations(), max(na, 1e-6), "ideal"


def resolve_build_optics(
    bench: Bench,
    s_object: float,
    detector_name: str,
    wavelength_um: float = REFERENCE_WAVELENGTH_UM,
    objective_name: str = "objective",
    field_height: float = 0.0,
    coherence_parameter: float = 0.6,
) -> BuildOptics:
    """Read a bench and return the optical state its image is formed with."""
    system = bench.to_paraxial()
    notes: list[str] = []

    na = 0.0
    aberrations, reference_na, source = Aberrations(), 1e-6, "ideal"
    if bench.has(objective_name):
        na = float(bench.get(objective_name).metadata.get("na", 0.0))
        aberrations, reference_na, source = _catalog_aberrations(bench, objective_name)
    if na <= 0:
        na = system.object_space_na(s_object)
        if na > 0:
            notes.append("NA taken from the ray trace: no objective declares one")
    if na <= 0:
        na = 0.1
        notes.append("no aperture-limited bundle; assuming NA 0.1 so an image can be formed")

    detector_s = bench.get(detector_name).s if bench.has(detector_name) else None
    image_s = system.image_plane(s_object, search_to=detector_s)

    magnification = 1.0
    image_defocus = 0.0
    if image_s is not None and detector_s is not None and image_s >= s_object:
        magnification = abs(system.magnification(s_object, image_s))
        image_defocus = detector_s - image_s
    elif image_s is None:
        notes.append("output is collimated: showing the image the optics would form at focus")
        nominal = bench.get(objective_name).metadata.get("magnification") if bench.has(objective_name) else None
        magnification = float(nominal or 1.0)

    # Defocus is computed, never assigned. An image-side error refers to object
    # space as dz / M^2, because the longitudinal magnification is M^2.
    object_defocus = image_defocus / (magnification**2) if magnification else 0.0

    # The secondary-spectrum term is a fraction of the focal length, so the
    # objective's actual focal length has to reach the budget: without it the
    # chromatic term silently evaluates to zero and an achromat images like an apo.
    focal_length_mm = 0.0
    if bench.has(objective_name):
        focal_length_mm = bench.get(objective_name).focal_length_mm or 0.0

    wavefront = budget_at_field(
        aberrations, field_height, na, reference_na, wavelength_um,
        REFERENCE_WAVELENGTH_UM, focal_length_mm,
    )
    if object_defocus:
        wavefront = wavefront + defocus_from_stage_error(object_defocus, na, wavelength_um)

    # Brightness goes as NA^2 / M^2, which is why high power is dim and why empty
    # magnification costs light as well as resolution.
    irradiance = (
        (na / magnification) ** 2 / _REFERENCE_THROUGHPUT if magnification else 0.0
    )

    if source == "ideal":
        notes.append(
            "no catalog key or correction grade on this objective: treating it as "
            "aberration-free rather than guessing a grade"
        )

    return BuildOptics(
        na=na,
        magnification=magnification,
        wavelength_um=wavelength_um,
        coherence_parameter=coherence_parameter,
        wavefront=wavefront,
        relative_irradiance=irradiance,
        image_defocus_mm=image_defocus,
        object_defocus_mm=object_defocus,
        field_height=field_height,
        aberration_source=source,
        notes=notes,
    )
