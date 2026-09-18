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
    parser.add_argument("--save-reference", type=Path, help="write the reference build to JSON")
    args = parser.parse_args(argv)

    if args.list:
        for n, r in sorted(ROUNDS.items()):
            print(f"{n}. {r.title} -- {r.teaches}")
        return 0

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
    print()
    if report.passed:
        print("ROUND PASSED")
        return 0
    print(f"ROUND FAILED ({len(report.failures())} failing checks)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
