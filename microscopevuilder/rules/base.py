"""Rule results that teach.

A rule never answers only "pass". It reports the measured value, the target, the
governing equation with the player's own numbers substituted in, and -- on failure --
which element is responsible and what to do about it. For the intended audience
(graduate students who will check this against their own bench) the explanation is
the product; the boolean is incidental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True)
class RuleResult:
    name: str
    status: Status
    summary: str
    equation: str = ""
    measured: float | None = None
    target: float | None = None
    units: str = ""
    culprit: str | None = None
    remedy: str = ""

    @property
    def ok(self) -> bool:
        """A warning is advisory, not blocking.

        Only a FAIL stops a round. A WARN says "this works but a demonstrator
        would raise an eyebrow" -- a condenser slightly closed, a tolerance in the
        second band -- and rounds use it deliberately: round 3 passes *with* a
        warning that the filament is visible, which is exactly the observation
        round 4 then asks the player to fix.
        """
        return self.status is not Status.FAIL

    def format(self) -> str:
        mark = {
            Status.PASS: "PASS",
            Status.WARN: "WARN",
            Status.FAIL: "FAIL",
            Status.NOT_APPLICABLE: " -- ",
        }[self.status]
        line = f"[{mark}] {self.name}: {self.summary}"
        if self.equation:
            line += f"\n         {self.equation}"
        if self.culprit and not self.ok:
            line += f"\n         culprit: {self.culprit}"
        if self.remedy and not self.ok:
            line += f"\n         remedy:  {self.remedy}"
        return line


@dataclass
class RuleReport:
    results: list[RuleResult] = field(default_factory=list)

    def add(self, result: RuleResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)

    def failures(self) -> list[RuleResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    def format(self) -> str:
        return "\n".join(r.format() for r in self.results)


def tolerance_result(
    name: str,
    measured: float,
    target: float,
    tolerance: float,
    units: str,
    equation: str,
    culprit: str | None = None,
    remedy: str = "",
    relative: bool = True,
) -> RuleResult:
    """Compare a measurement to a target, with a warn band at twice the tolerance."""
    error = abs(measured - target)
    allowed = abs(target) * tolerance if relative else tolerance
    if error <= allowed:
        status = Status.PASS
    elif error <= 2 * allowed:
        status = Status.WARN
    else:
        status = Status.FAIL
    direction = "high" if measured > target else "low"
    summary = (
        f"{measured:.4g} {units} against a target of {target:.4g} {units}"
        + ("" if status is Status.PASS else f" -- {direction} by {error:.3g} {units}")
    )
    return RuleResult(
        name=name,
        status=status,
        summary=summary,
        equation=equation,
        measured=measured,
        target=target,
        units=units,
        culprit=culprit,
        remedy=remedy,
    )
