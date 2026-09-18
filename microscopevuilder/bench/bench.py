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

from ..optics.paraxial import (
    Element,
    ParaxialSystem,
    aperture,
    glass_plate,
    thin_lens,
)


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


@dataclass(frozen=True)
class Arm:
    """A side branch that joins the main axis at a beamsplitter or dichroic.

    Episcopic illumination is not a folded path -- it is a *second* path meeting the
    imaging path. The epi illuminator comes in from the side, turns down through the
    objective, and the returning light carries on up the main axis; two bundles share
    the objective while travelling in opposite directions through it.

    An arm has its own path coordinate running from 0 at its source to ``length_mm``
    at the junction. Tracing the arm walks its own elements and then continues into
    the main axis from ``junction_s``, so the objective is traced once per path and
    the two agree about it by construction.
    """

    name: str
    junction_s: float
    length_mm: float
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    continues: str = "forward"  # "forward" or "reverse" along the main axis

    def __post_init__(self) -> None:
        if self.continues not in ("forward", "reverse"):
            raise ValueError(
                f"arm {self.name}: continues must be 'forward' or 'reverse', "
                f"got {self.continues!r}"
            )

    def unit(self) -> np.ndarray:
        v = np.array(self.direction, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError(f"arm {self.name}: direction cannot be zero")
        return v / n


@dataclass
class BenchElement:
    """A placed component. ``s`` is optical path position; 3D follows from folds.

    ``arm`` names which path the element sits on. Elements on the main axis use
    ``"main"``; elements on a side branch use that arm's name, and their ``s`` is
    measured along the arm rather than along the main axis.
    """

    name: str
    s: float
    kind: str
    semi_diameter_mm: float
    focal_length_mm: float | None = None
    label: str = ""
    catalog_key: str | None = None
    arm: str = "main"
    enabled: bool = True
    metadata: dict = field(default_factory=dict)

    def to_optical(self) -> Element:
        thickness = self.metadata.get("thickness_mm")
        if thickness:
            # A plate carries real glass, so it is not a bare aperture: it displaces
            # focus in converging light and does nothing in collimated light.
            return glass_plate(
                self.name, self.s, float(thickness),
                float(self.metadata.get("index", 1.52)),
                self.semi_diameter_mm, self.kind,
            )
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
        arms: list[Arm] | None = None,
    ):
        self.elements = list(elements or [])
        self.folds = sorted(folds or [], key=lambda f: f.s)
        self.arms = {a.name: a for a in (arms or [])}
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

    def select_turret(self, name: str) -> "Bench":
        """Rotate one turret objective into the light path and the rest out."""
        self.elements = [
            replace(e, enabled=(e.name == name)) if e.metadata.get("in_turret") else e
            for e in self.elements
        ]
        return self

    def of_kind(self, kind: str) -> list[BenchElement]:
        return [e for e in self.elements if e.kind == kind]

    def on_arm(self, arm: str) -> list[BenchElement]:
        return [e for e in self.elements if e.arm == arm]

    # --- geometry ------------------------------------------------------------

    def position_of_arm(self, arm_name: str, s: float) -> np.ndarray:
        """3D position at path coordinate ``s`` along a side arm.

        The arm runs toward the junction, so ``s = length_mm`` lands exactly on the
        main axis at ``junction_s`` and the two paths meet where the beamsplitter is.
        """
        arm = self.arms[arm_name]
        junction = self.position_of(arm.junction_s)
        return junction - arm.unit() * (arm.length_mm - s)

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
        return max(
            [e.s for e in self.elements if e.arm == "main"]
            + [f.s for f in self.folds]
            + [a.junction_s for a in self.arms.values()]
            + [0.0]
        )

    # --- optics --------------------------------------------------------------

    def to_paraxial(self, arm: str = "main") -> ParaxialSystem:
        """Hand one path to the trace engine.

        Folds are optically transparent, and white cards are excluded entirely:
        they are diagnostic probes, so holding one up must never change the answer
        it is being used to measure. See :mod:`microscopevuilder.bench.probe`.

        For a side arm the returned system contains the arm's own elements at their
        arm coordinates, followed by the main-axis elements from the junction
        onward, shifted so the path coordinate runs continuously. The objective
        therefore appears in both systems at the correct distance along each path,
        which is exactly the physical situation in an epi stand.
        """
        if arm == "main":
            return ParaxialSystem(
                [e.to_optical() for e in self.elements
                 if e.arm == "main" and e.kind != "white_card" and e.enabled]
            )

        spec = self.arms[arm]
        elements = [
            e.to_optical() for e in self.elements
            if e.arm == arm and e.kind != "white_card" and e.enabled
        ]
        for e in self.elements:
            if e.arm != "main" or e.kind == "white_card" or not e.enabled:
                continue
            if spec.continues == "reverse":
                # Episcopic illumination runs the other way down the main axis: it
                # enters at the beamsplitter and travels back through the objective
                # to the specimen, while the returning light goes the other way.
                # A thin lens is symmetric, so the same element serves both.
                if e.s > spec.junction_s:
                    continue
                offset = spec.junction_s - e.s
            else:
                if e.s < spec.junction_s:
                    continue
                offset = e.s - spec.junction_s
            optical = e.to_optical()
            optical.s = spec.length_mm + offset
            elements.append(optical)
        return ParaxialSystem(elements)

    def arm_coordinate(self, arm: str, main_s: float) -> float:
        """Convert a main-axis position into the coordinate of a joining arm."""
        spec = self.arms[arm]
        offset = (
            spec.junction_s - main_s if spec.continues == "reverse" else main_s - spec.junction_s
        )
        return spec.length_mm + offset

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
            "arms": [
                {
                    "name": a.name,
                    "junction_s": a.junction_s,
                    "length_mm": a.length_mm,
                    "direction": list(a.direction),
                    "continues": a.continues,
                }
                for a in self.arms.values()
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
                    "arm": e.arm,
                    "enabled": e.enabled,
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
            arms=[
                Arm(
                    name=a["name"],
                    junction_s=a["junction_s"],
                    length_mm=a["length_mm"],
                    direction=tuple(a.get("direction", (0.0, 0.0, -1.0))),
                    continues=a.get("continues", "forward"),
                )
                for a in data.get("arms", [])
            ],
        )

    def save(self, path: Path) -> None:
        """Benches are plain JSON so players can share builds."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Bench":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
