"""Loader for the component catalog (``assets/components.toml``).

Keeps the distinction between vendor specification and teaching fiction explicit,
because the audience is people who will check these numbers against their own bench.
Anything with ``verified = False`` or ``source = "pedagogical"`` must be surfaced as
such in the UI rather than presented as a datasheet value.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

def _catalog_path() -> Path:
    """Locate the catalog, whether running from source or from a frozen build.

    PyInstaller unpacks bundled data to a temporary directory and points
    ``sys._MEIPASS`` at it, so a path derived from ``__file__`` alone breaks in the
    packaged executable.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidate = Path(bundled) / "microscopevuilder" / "assets" / "components.toml"
        if candidate.exists():
            return candidate
        candidate = Path(bundled) / "assets" / "components.toml"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent / "assets" / "components.toml"


CATALOG_PATH = _catalog_path()


@dataclass(frozen=True)
class Aberrations:
    """An objective's residual aberrations, in the units each is naturally quoted in.

    Two are wavefront errors in microns RMS at the reference aperture and full
    field. Two are longitudinal *distances*, because that is how they are quoted and
    measured: field curvature as a sag, and secondary spectrum as a fraction of the
    focal length. See :func:`microscopevuilder.optics.wavefront.budget_at_field`
    for how each is scaled and converted.
    """

    spherical_rms_um: float = 0.0
    astigmatism_rms_um: float = 0.0
    field_curvature_sag_um: float = 0.0
    chromatic_focus_fraction: float = 0.0
    source: str = "pedagogical"

    @property
    def is_vendor_data(self) -> bool:
        return self.source != "pedagogical"


@dataclass(frozen=True)
class Objective:
    key: str
    label: str
    magnification: float
    na: float
    working_distance_mm: float
    immersion: str
    grade: str
    aberrations: Aberrations
    verified: bool
    source: str = ""
    checked_on: str = ""

    def focal_length_mm(self, tube_lens_focal_length_mm: float) -> float:
        """f = f_tube / M -- the defining relation of an infinity system.

        For CFI60 the tube lens is 200 mm, so a 20x objective is 10 mm. Note this
        is a *system* property: the same objective on a 180 mm stand is no longer
        20x, which is exactly the mistake round 11 is designed to provoke.
        """
        return tube_lens_focal_length_mm / self.magnification

    @property
    def medium_index(self) -> float:
        return MEDIA[self.immersion]

    @property
    def half_angle_rad(self) -> float:
        """Real (non-paraxial) acceptance half-angle, from NA = n sin(u).

        Used for honest reporting in the inspector; the paraxial engine itself
        works in the small-angle limit.
        """
        import math

        return math.asin(min(self.na / self.medium_index, 1.0))


@dataclass(frozen=True)
class SystemSpec:
    name: str
    tube_lens_focal_length_mm: float
    parfocal_distance_mm: float
    reference_wavelength_nm: float
    verified: bool


@dataclass(frozen=True)
class Catalog:
    system: SystemSpec
    objectives: dict[str, Objective]
    media: dict[str, float]

    def objective(self, key: str) -> Objective:
        return self.objectives[key]

    def unverified_entries(self) -> list[str]:
        """Everything still awaiting a datasheet check, for the UI disclaimer."""
        out = [] if self.system.verified else [f"system:{self.system.name}"]
        out += [f"objective:{k}" for k, o in self.objectives.items() if not o.verified]
        return out

    def verification_report(self) -> str:
        """A checklist for whoever does the datasheet pass.

        Lists exactly which fields need confirming against which product, so the
        work is mechanical rather than a research exercise. Marking an entry
        verified requires filling in `source` and `checked_on`; the loader refuses
        the catalog otherwise.
        """
        lines = [
            f"Catalog verification -- {self.system.name}",
            "",
            "Specification fields below are recorded from secondary knowledge and "
            "have NOT been checked against a datasheet.",
            "Aberration budgets are pedagogical by design and are never verified; "
            "manufacturers do not publish Zernike budgets.",
            "",
            f"System: tube lens {self.system.tube_lens_focal_length_mm:.0f} mm, "
            f"parfocal {self.system.parfocal_distance_mm:.0f} mm "
            f"[{'verified' if self.system.verified else 'UNVERIFIED'}]",
            "",
        ]
        for key, o in sorted(self.objectives.items()):
            mark = "verified" if o.verified else "UNVERIFIED"
            lines.append(f"[{mark}] {key}")
            lines.append(f"    label              {o.label}")
            lines.append(f"    magnification      {o.magnification:g}x")
            lines.append(f"    numerical aperture {o.na:g}")
            lines.append(f"    working distance   {o.working_distance_mm:g} mm")
            lines.append(f"    immersion          {o.immersion}")
            if o.verified:
                lines.append(f"    source             {o.source} ({o.checked_on})")
            else:
                lines.append("    source             -- add `source` and `checked_on` to mark verified")
            lines.append("")
        pending = len([o for o in self.objectives.values() if not o.verified])
        lines.append(f"{pending} of {len(self.objectives)} objectives await a datasheet check.")
        return "\n".join(lines)


MEDIA: dict[str, float] = {}


@lru_cache(maxsize=1)
def load_catalog(path: Path | None = None) -> Catalog:
    raw = tomllib.loads((path or CATALOG_PATH).read_text(encoding="utf-8"))

    media = {k: float(v["index"]) for k, v in raw["media"].items()}
    MEDIA.clear()
    MEDIA.update(media)

    s = raw["system"]
    system = SystemSpec(
        name=s["name"],
        tube_lens_focal_length_mm=float(s["tube_lens_focal_length_mm"]),
        parfocal_distance_mm=float(s["parfocal_distance_mm"]),
        reference_wavelength_nm=float(s["reference_wavelength_nm"]),
        verified=bool(s.get("verified", False)),
    )

    objectives: dict[str, Objective] = {}
    for key, o in raw["objectives"].items():
        ab = dict(o.get("aberrations", {}))
        objectives[key] = Objective(
            key=key,
            label=o["label"],
            magnification=float(o["magnification"]),
            na=float(o["na"]),
            working_distance_mm=float(o["working_distance_mm"]),
            immersion=o["immersion"],
            grade=o["grade"],
            aberrations=Aberrations(
                spherical_rms_um=float(ab.get("spherical_rms_um", 0.0)),
                astigmatism_rms_um=float(ab.get("astigmatism_rms_um", 0.0)),
                field_curvature_sag_um=float(ab.get("field_curvature_sag_um", 0.0)),
                chromatic_focus_fraction=float(ab.get("chromatic_focus_fraction", 0.0)),
                source=str(ab.get("source", "pedagogical")),
            ),
            verified=bool(o.get("verified", False)),
            source=str(o.get("source", "")),
            checked_on=str(o.get("checked_on", "")),
        )

    if any(o.immersion not in media for o in objectives.values()):
        missing = {o.immersion for o in objectives.values()} - set(media)
        raise ValueError(f"catalog references undefined media: {sorted(missing)}")

    # A verified entry must say where it was verified from. Without this guard
    # "verified = true" is just an assertion, and the whole provenance scheme rests
    # on it meaning something a reader can follow up.
    unsourced = [
        key for key, o in objectives.items()
        if o.verified and not (o.source and o.checked_on)
    ]
    if unsourced:
        raise ValueError(
            "these entries are marked verified but cite no source and date: "
            f"{sorted(unsourced)}. Verification means a citation, not a flag."
        )

    return Catalog(system=system, objectives=objectives, media=media)
