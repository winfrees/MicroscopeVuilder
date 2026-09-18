"""Headless round runner: ``python -m microscopevuilder --round 1``.

The milestone M3 deliverable. It proves the engine composes into an actual puzzle
without any UI, and it doubles as the harness the round tests use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bench.bench import Bench
from .game.rounds import ROUNDS, get_round


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
    parser.add_argument("--save-reference", type=Path, help="write the reference build to JSON")
    args = parser.parse_args(argv)

    if args.list:
        for n, r in sorted(ROUNDS.items()):
            print(f"{n}. {r.title} -- {r.teaches}")
        return 0

    if args.ui:
        try:
            from .ui.workspace import launch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            print(f"the workspace needs the 'ui' extra: pip install 'microscopevuilder[ui]' ({exc})")
            return 2
        return launch(args.round)

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
    print(report.format())

    # Measured rules render the image, so they are a separate pass. Headless there
    # is no reason to skip them; in the workspace they run behind Run.
    if rnd.has_measured_rules and not args.no_measure:
        print()
        print("-- measured from the rendered image --")
        measured = rnd.measure_image(bench)
        print(measured.format())
        report.results.extend(measured.results)

    print()
    if report.passed:
        print("ROUND PASSED")
        return 0
    print(f"ROUND FAILED ({len(report.failures())} failing checks)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
