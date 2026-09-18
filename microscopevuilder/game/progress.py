"""Player progress, saved where the operating system says user data belongs.

`platformdirs` was a declared dependency that nothing imported: benches
serialised, but which rounds a player had solved did not survive closing the app.

Saves go to the per-user config directory rather than next to the executable,
which matters for the intended audience: a one-file binary on a shared lab machine
may sit somewhere read-only, and two people using the same machine should not
overwrite each other's progress.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_path

APP_NAME = "MicroscopeVuilder"
SAVE_VERSION = 1


def default_save_path() -> Path:
    return user_config_path(APP_NAME, appauthor=False) / "progress.json"


@dataclass
class RoundRecord:
    stars: int = 0
    attempts: int = 0
    best_parts: int | None = None

    def merge(self, stars: int, parts_used: int) -> "RoundRecord":
        """Keep the best result, not the latest: progress should never go backwards."""
        return RoundRecord(
            stars=max(self.stars, stars),
            attempts=self.attempts + 1,
            best_parts=(
                parts_used if self.best_parts is None else min(self.best_parts, parts_used)
            ) if stars > 0 else self.best_parts,
        )


@dataclass
class Progress:
    rounds: dict[int, RoundRecord] = field(default_factory=dict)
    version: int = SAVE_VERSION

    # --- queries ------------------------------------------------------------

    def record(self, number: int) -> RoundRecord:
        return self.rounds.get(number, RoundRecord())

    def is_solved(self, number: int) -> bool:
        return self.record(number).stars > 0

    def total_stars(self) -> int:
        return sum(r.stars for r in self.rounds.values())

    def is_unlocked(self, number: int, prerequisites: dict[int, int]) -> bool:
        """A round is available once its prerequisite has been solved.

        The sandbox and round 1 are always open. Everything else waits on the round
        before it, except the contrast techniques, which wait on Köhler -- a phase
        ring is meaningless to someone who does not yet own the back focal plane.
        """
        required = prerequisites.get(number)
        return required is None or self.is_solved(required)

    # --- mutation -----------------------------------------------------------

    def complete(self, number: int, stars: int, parts_used: int) -> None:
        self.rounds[number] = self.record(number).merge(stars, parts_used)

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "rounds": {str(k): asdict(v) for k, v in sorted(self.rounds.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Progress":
        version = int(data.get("version", SAVE_VERSION))
        if version > SAVE_VERSION:
            # Forward compatibility: a newer save is not readable, but losing
            # somebody's progress silently is worse than starting fresh loudly.
            raise ValueError(
                f"this save was written by a newer version (save format {version}, "
                f"this build understands {SAVE_VERSION})"
            )
        rounds = {}
        for key, value in data.get("rounds", {}).items():
            rounds[int(key)] = RoundRecord(
                stars=int(value.get("stars", 0)),
                attempts=int(value.get("attempts", 0)),
                best_parts=value.get("best_parts"),
            )
        return cls(rounds=rounds, version=version)

    def save(self, path: Path | None = None) -> Path:
        target = Path(path or default_save_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling and rename, so an interrupted save cannot leave a
        # truncated file where the player's progress used to be.
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        scratch.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Progress":
        target = Path(path or default_save_path())
        if not target.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError, KeyError):
            # A corrupt save must not stop the game opening.
            return cls()
