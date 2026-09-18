"""The optical bench: a 3D-native model that the paraxial engine traces in 1D.

Per docs/PLAN.md decision 3, geometry is stored in 3D from the start even though
only a 2D renderer exists. The trick that makes both cheap is to keep the *optical*
coordinate one-dimensional -- path length ``s`` along the folded axis -- while
deriving 3D positions from the fold geometry on demand. Folding a path never
changes its optics, so the trace stays a simple ordered walk, and a future 3D
renderer reads ``position_of()`` without anything being unflattened first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..optics.paraxial import Element, ParaxialSystem, aperture, thin_lens


@dataclass(frozen=True)
class Fold:
    """A mirror or dichroic at path position ``s`` turning the axis.

    ``direction`` is the outgoing unit vector. The incoming direction is whatever
    the axis was doing before, so a fold is fully described by where it is and
    where it sends the light.
    """

    s: float
    direction: tuple[float, float, float]
    name: str = "fold"

    def unit(self) -> np.ndarray:
        v = np.array(self.direction, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError(f"{self.name}: fold direction cannot be zero")
        return v / n


@dataclass
class BenchElement:
    """A placed component. ``s`` is optical path position; 3D follows from folds."""

    name: str
    s: float
    kind: str
    semi_diameter_mm: float
    focal_length_mm: float | None = None
    label: str = ""
    catalog_key: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_optical(self) -> Element:
        if self.focal_length_mm is None:
            return aperture(self.name, self.s, self.semi_diameter_mm, self.kind)
        return thin_lens(
            self.name, self.s, self.focal_length_mm, self.semi_diameter_mm, self.kind
        )


class Bench:
    """An ordered set of components plus the folds of the axis they sit on."""

    def __init__(
        self,
        elements: list[BenchElement] | None = None,
        folds: list[Fold] | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ):
        self.elements = list(elements or [])
        self.folds = sorted(folds or [], key=lambda f: f.s)
        self.origin = np.array(origin, dtype=float)
        d = np.array(initial_direction, dtype=float)
        self.initial_direction = d / np.linalg.norm(d)

    # --- editing -------------------------------------------------------------

    def add(self, element: BenchElement) -> "Bench":
        if any(e.name == element.name for e in self.elements):
            raise ValueError(f"duplicate element name: {element.name}")
        self.elements.append(element)
        return self

    def remove(self, name: str) -> "Bench":
        self.elements = [e for e in self.elements if e.name != name]
        return self

    def move(self, name: str, s: float) -> "Bench":
        self.elements = [replace(e, s=s) if e.name == name else e for e in self.elements]
        return self

    def get(self, name: str) -> BenchElement:
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(e.name == name for e in self.elements)

    def of_kind(self, kind: str) -> list[BenchElement]:
        return [e for e in self.elements if e.kind == kind]

    # --- geometry ------------------------------------------------------------

    def position_of(self, s: float) -> np.ndarray:
        """3D position at path coordinate ``s``, walking the folds.

        This is the only place the third dimension appears, and it is why an
        episcopic path costs nothing extra: it is the same 1D trace, drawn bent.
        """
        point = self.origin.copy()
        direction = self.initial_direction.copy()
        cursor = 0.0
        for fold in self.folds:
            if fold.s >= s:
                break
            point = point + direction * (fold.s - cursor)
            direction = fold.unit()
            cursor = fold.s
        return point + direction * (s - cursor)

    def polyline(self) -> list[np.ndarray]:
        """Vertices of the axis, for the renderer."""
        stops = [0.0] + [f.s for f in self.folds] + [self.extent()]
        return [self.position_of(s) for s in stops]

    def extent(self) -> float:
        return max([e.s for e in self.elements] + [f.s for f in self.folds] + [0.0])

    # --- optics --------------------------------------------------------------

    def to_paraxial(self) -> ParaxialSystem:
        """Hand the bench to the trace engine.

        Folds are optically transparent, and white cards are excluded entirely:
        they are diagnostic probes, so holding one up must never change the answer
        it is being used to measure. See :mod:`microscopevuilder.bench.probe`.
        """
        return ParaxialSystem(
            [e.to_optical() for e in self.elements if e.kind != "white_card"]
        )

    def cards(self) -> list["BenchElement"]:
        return [e for e in self.elements if e.kind == "white_card"]

    def add_card(self, s: float, name: str | None = None) -> "BenchElement":
        """Drop a white card on the axis at ``s``."""
        existing = {e.name for e in self.elements}
        if name is None:
            i = 1
            while f"card_{i}" in existing:
                i += 1
            name = f"card_{i}"
        card = BenchElement(name, s, "white_card", 25.0, None, label="white card")
        self.add(card)
        return card

    # --- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "initial_direction": self.initial_direction.tolist(),
            "folds": [
                {"s": f.s, "direction": list(f.direction), "name": f.name}
                for f in self.folds
            ],
            "elements": [
                {
                    "name": e.name,
                    "s": e.s,
                    "kind": e.kind,
                    "semi_diameter_mm": e.semi_diameter_mm,
                    "focal_length_mm": e.focal_length_mm,
                    "label": e.label,
                    "catalog_key": e.catalog_key,
                    "metadata": e.metadata,
                }
                for e in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bench":
        return cls(
            elements=[BenchElement(**e) for e in data.get("elements", [])],
            folds=[
                Fold(s=f["s"], direction=tuple(f["direction"]), name=f.get("name", "fold"))
                for f in data.get("folds", [])
            ],
            origin=tuple(data.get("origin", (0.0, 0.0, 0.0))),
            initial_direction=tuple(data.get("initial_direction", (1.0, 0.0, 0.0))),
        )

    def save(self, path: Path) -> None:
        """Benches are plain JSON so players can share builds."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Bench":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
