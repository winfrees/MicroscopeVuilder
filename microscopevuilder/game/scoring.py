"""Scoring, the parts budget, and the diff against a working build.

The plan's pillar about a cheap correct solution beating a brute-force one had no
teeth: every round carried a `parts_budget` and nothing enforced it. This module
makes the budget real, turns the measured numbers into a rating, and produces the
diff a player sees after a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bench.bench import Bench, BenchElement
from ..rules.base import RuleReport, RuleResult, Status

# Probes and the components a round hands you for free do not count against the
# budget: a white card is a measuring instrument, not a part of the microscope.
FREE_KINDS = frozenset({"white_card", "field_stop", "lamp"})


def billable_parts(bench: Bench) -> list[BenchElement]:
    """The components that count against a round's budget.

    A parked component -- carried over from an earlier round but not in the path --
    costs nothing: no light, no scatter, nothing to align. `to_paraxial` already
    ignores it, so the budget must too, or carrying a build forward would be
    punished for parts the player is not using.
    """
    return [e for e in bench.elements if e.kind not in FREE_KINDS and e.enabled]


def check_parts_budget(bench: Bench, budget: int) -> RuleResult:
    """Did the build stay inside its parts allowance?

    A budget is what stops "add another lens" being the answer to everything. It is
    also the closest this game gets to the real constraint on a bench, which is that
    every extra surface costs light, scatter and one more thing to align.
    """
    used = len(billable_parts(bench))
    status = Status.PASS if used <= budget else Status.FAIL
    return RuleResult(
        name="Parts budget",
        status=status,
        summary=(
            f"{used} of {budget} parts used"
            + ("" if status is Status.PASS else f" -- {used - budget} over")
        ),
        equation=(
            "counting everything in the path but cards, stops and the lamp: "
            + ", ".join(e.name for e in billable_parts(bench))
        ),
        measured=float(used),
        target=float(budget),
        remedy=(
            "every extra surface costs light, scatter and another alignment; "
            "solve it with fewer components"
        ),
    )


@dataclass
class Score:
    """A round's outcome: stars, and the reasons for them."""

    stars: int
    reasons: list[str] = field(default_factory=list)
    parts_used: int = 0
    parts_budget: int = 0
    # Geometry is green but the rules that need a rendered image have not been run
    # against this build yet, so there is no verdict to give. Not a pass, and not a
    # failure either.
    provisional: bool = False

    @property
    def passed(self) -> bool:
        """A pass is every applicable rule satisfied -- not merely none objecting."""
        return self.stars > 0 and not self.provisional

    def describe(self) -> str:
        pips = "*" * self.stars + "." * (3 - self.stars)
        return f"[{pips}] " + ("; ".join(self.reasons) if self.reasons else "solved")


def score_round(
    bench: Bench,
    budget: int,
    live: RuleReport,
    measured: RuleReport | None = None,
    needs_measurement: bool = False,
) -> Score:
    """Rate a build out of three.

    One star for solving it at all. The second and third are for *how* it was
    solved: no warnings outstanding, and inside the parts budget. A build that
    passes with a warning has worked, and a demonstrator would still raise an
    eyebrow -- which is exactly the distinction the second star marks.

    `needs_measurement` says this round has rules that only a rendered image can
    settle. Until that render has happened the score is *provisional*: it is not a
    pass, however green the geometry looks.
    """
    results = list(live.results) + list(measured.results if measured else [])
    failures = [r for r in results if r.status is Status.FAIL]
    used_now = len(billable_parts(bench))

    if needs_measurement and measured is None and not failures:
        # Absence of evidence was reading as success: a round graded on resolution
        # or sampling would pass on ray geometry alone, skipping the half it exists
        # to teach. Withhold the verdict instead of inventing one.
        return Score(
            0,
            ["geometry is green -- press Run to grade the rules that need an image"],
            used_now,
            budget,
            provisional=True,
        )
    warnings = [r for r in results if r.status is Status.WARN]
    used = len(billable_parts(bench))

    if failures:
        return Score(0, [f"{r.name}: {r.summary}" for r in failures[:3]], used, budget)

    stars = 1
    reasons = []
    if warnings:
        reasons.append(f"{len(warnings)} warning(s): " + ", ".join(r.name for r in warnings))
    else:
        stars += 1
    if used <= budget:
        stars += 1
    else:
        reasons.append(f"{used} parts against a budget of {budget}")
    return Score(stars, reasons, used, budget)


@dataclass
class BenchDiff:
    """How a build differs from one that works."""

    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    moved: list[tuple[str, float, float]] = field(default_factory=list)
    changed: list[tuple[str, str, str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.missing or self.extra or self.moved or self.changed)

    def format(self) -> str:
        if self.is_empty:
            return "identical to the reference build"
        lines = []
        for name in self.missing:
            lines.append(f"  missing:  {name}")
        for name in self.extra:
            lines.append(f"  extra:    {name}")
        for name, yours, reference in self.moved:
            lines.append(
                f"  moved:    {name} at s = {yours:.2f} mm, reference has {reference:.2f} mm"
                f" ({yours - reference:+.2f})"
            )
        for name, field_name, yours, reference in self.changed:
            lines.append(f"  differs:  {name}.{field_name} = {yours}, reference has {reference}")
        return "\n".join(lines)


def diff_against_reference(
    bench: Bench, reference: Bench, position_tolerance_mm: float = 0.01
) -> BenchDiff:
    """Compare a build to a working one.

    Offered only after a failure, and deliberately structural rather than a
    solution: it says what differs, not what to do about it, so the player still
    has to work out why the difference matters.
    """
    yours = {e.name: e for e in bench.elements if e.kind not in FREE_KINDS}
    theirs = {e.name: e for e in reference.elements if e.kind not in FREE_KINDS}

    diff = BenchDiff(
        missing=sorted(set(theirs) - set(yours)),
        extra=sorted(set(yours) - set(theirs)),
    )
    for name in sorted(set(yours) & set(theirs)):
        mine, ref = yours[name], theirs[name]
        if abs(mine.s - ref.s) > position_tolerance_mm:
            diff.moved.append((name, mine.s, ref.s))
        if mine.focal_length_mm != ref.focal_length_mm:
            diff.changed.append(
                (name, "focal length", _show(mine.focal_length_mm), _show(ref.focal_length_mm))
            )
        if abs(mine.semi_diameter_mm - ref.semi_diameter_mm) > 1e-9:
            diff.changed.append(
                (name, "semi-diameter", f"{mine.semi_diameter_mm:g}", f"{ref.semi_diameter_mm:g}")
            )
        for key in sorted(set(mine.metadata) | set(ref.metadata)):
            if mine.metadata.get(key) != ref.metadata.get(key):
                diff.changed.append(
                    (name, key, _show(mine.metadata.get(key)), _show(ref.metadata.get(key)))
                )
    return diff


def _show(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
