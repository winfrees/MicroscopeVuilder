"""Headless round runner: ``python -m microscopevuilder --round 1``.

The milestone M3 deliverable. It proves the engine composes into an actual puzzle
without any UI, and it doubles as the harness the round tests use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bench.bench import Bench
from .game.progress import Progress
from .game.rounds import PREREQUISITES, ROUNDS, get_round
from .game.scoring import check_parts_budget, diff_against_reference, score_round


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="microscopevuilder")
    parser.add_argument("--round", type=int, default=1, help="round number to grade")
    parser.add_argument("--bench", type=Path, help="bench JSON to grade (default: the reference build)")
    parser.add_argument("--list", action="store_true", help="list implemented rounds")
    parser.add_argument("--ui", action="store_true", help="open the graphical workspace")
    parser.add_argument(
        "--no-measure", action="store_true",
        help="skip the measured (rendering) rules and grade on ray geometry only",
    )
    parser.add_argument("--progress", type=Path, help="progress file (default: user config dir)")
    parser.add_argument("--diff", action="store_true", help="on failure, diff against a working build")
    parser.add_argument(
        "--strict", action="store_true",
        help="grade positions at the physical tolerance instead of the 5% practice one",
    )
    parser.add_argument(
        "--verify-catalog", action="store_true",
        help="print which catalog entries still need a datasheet check",
    )
    parser.add_argument("--save-reference", type=Path, help="write the reference build to JSON")
    args = parser.parse_args(argv)

    if args.verify_catalog:
        from .bench.catalog import load_catalog

        print(load_catalog().verification_report())
        return 0

    if args.list:
        progress = Progress.load(args.progress)
        for n, r in sorted(ROUNDS.items()):
            record = progress.record(n)
            pips = "*" * record.stars + "." * (3 - record.stars)
            locked = "" if progress.is_unlocked(n, PREREQUISITES) else "  (locked)"
            print(f"[{pips}] {n}. {r.title} -- {r.teaches}{locked}")
        from .game.progress import default_save_path

        print(f"\n{progress.total_stars()} stars -- progress in {args.progress or default_save_path()}")
        return 0

    if args.ui:
        try:
            from .ui.workspace import launch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            print(f"the workspace needs the 'ui' extra: pip install 'microscopevuilder[ui]' ({exc})")
            return 2
        return launch(args.round)

    if args.strict:
        from .rules.tolerances import TolerancePolicy, set_tolerance_policy

        set_tolerance_policy(TolerancePolicy.STRICT)

    rnd = get_round(args.round)
    bench = Bench.load(args.bench) if args.bench else rnd.reference_build()

    if args.save_reference:
        rnd.reference_build().save(args.save_reference)
        print(f"wrote reference build to {args.save_reference}")
        return 0

    print(f"Round {rnd.number}: {rnd.title}")
    print(f"  {rnd.brief}")
    print(f"  Teaches: {rnd.teaches}\n")

    report = rnd.grade(bench)
    report.add(check_parts_budget(bench, rnd.parts_budget))
    print(report.format())

    # Measured rules render the image, so they are a separate pass. Headless there
    # is no reason to skip them; in the workspace they run behind Run.
    if rnd.has_measured_rules and not args.no_measure:
        print()
        print("-- measured from the rendered image --")
        print(rnd.measure_image(bench).format())

    measured_report = None
    if rnd.has_measured_rules and not args.no_measure:
        measured_report = rnd.measure_image(bench)

    score = score_round(bench, rnd.parts_budget, report, measured_report)
    print()
    print(score.describe())

    progress = Progress.load(args.progress)
    progress.complete(rnd.number, score.stars, score.parts_used)
    progress.save(args.progress)

    if score.passed:
        print("ROUND PASSED")
        return 0

    if args.diff:
        print()
        print("-- how your build differs from a working one --")
        print(diff_against_reference(bench, rnd.reference_build()).format())
    else:
        print("(run again with --diff to compare against a working build)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
