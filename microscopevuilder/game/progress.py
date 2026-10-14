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
SAVE_VERSION = 2


def default_save_path() -> Path:
    return user_config_path(APP_NAME, appauthor=False) / "progress.json"


@dataclass
class RoundRecord:
    stars: int = 0
    attempts: int = 0
    best_parts: int | None = None
    # The serialized bench behind the best result. The next round starts from this,
    # so it has to be the build that actually earned the stars -- not the latest
    # attempt, which may be a half-dismantled experiment.
    best_bench: dict | None = None

    def is_better_than(self, stars: int, parts_used: int) -> bool:
        """Does an existing record beat this new result?

        More stars wins. On equal stars, fewer parts wins -- the same order the
        rating itself uses, so replaying can improve a record but never spoil one.
        """
        if self.stars != stars:
            return self.stars > stars
        return self.best_parts is not None and self.best_parts <= parts_used

    def merge(self, stars: int, parts_used: int, bench: dict | None = None) -> "RoundRecord":
        """Keep the best result, not the latest: progress should never go backwards."""
        keep_old = self.is_better_than(stars, parts_used)
        return RoundRecord(
            stars=max(self.stars, stars),
            attempts=self.attempts + 1,
            best_parts=(
                parts_used if self.best_parts is None else min(self.best_parts, parts_used)
            ) if stars > 0 else self.best_parts,
            # The saved build and the score it earned travel together: a record
            # that keeps the old stars keeps the old bench. With one exception --
            # a record with no bench at all (a failed attempt, or a save written
            # before builds were kept) takes this one, since a build in hand beats
            # none for the round that starts from it.
            best_bench=(
                self.best_bench
                if stars == 0 or (keep_old and self.best_bench is not None)
                else (bench or self.best_bench)
            ),
        )


@dataclass
class Progress:
    rounds: dict[int, RoundRecord] = field(default_factory=dict)
    version: int = SAVE_VERSION
    # Where the player was, so the game reopens where they left off instead of on
    # whichever round happened to be the code's default.
    last_round: int | None = None

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

    def complete(
        self, number: int, stars: int, parts_used: int, bench: dict | None = None
    ) -> None:
        self.rounds[number] = self.record(number).merge(stars, parts_used, bench)

    def solved_bench(self, number: int) -> dict | None:
        """The build that solved a round, for the next round to start from."""
        return self.record(number).best_bench

    def visit(self, number: int) -> None:
        self.last_round = number

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "rounds": {str(k): asdict(v) for k, v in sorted(self.rounds.items())},
            "last_round": self.last_round,
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
                best_bench=value.get("best_bench"),
            )
        # A version-1 save has no last_round and no saved benches; both read as
        # absent rather than as an error, so older progress still loads.
        return cls(
            rounds=rounds,
            version=SAVE_VERSION,
            last_round=data.get("last_round"),
        )

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
