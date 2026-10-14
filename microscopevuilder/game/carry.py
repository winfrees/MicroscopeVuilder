"""Carrying a build from one round into the next.

A round used to open on its own reference build, which is the answer already
assembled. It now opens on the bench the player *solved the previous round with*,
plus whatever this round introduces, unplaced. That is pillar 5 made load-bearing:
one instrument grows across fifteen rounds, and a build that passed while sloppy
stays sloppy until its owner fixes it.

Nothing is ever deleted on the player's behalf. A component this round has no use
for is **parked** -- `enabled = False` -- which `Bench.to_paraxial` already ignores
and which the parts budget does not charge for, since a part on the shelf costs no
light, no scatter and nothing to align. Parking is reversible; removal would not be.

What each round introduces is derived by comparing its reference build with the
previous round's, rather than authored fifteen times over. Three cases fall out:

  added    a component this round needs and the last one did not -> arrives in the
           tray, at the near end of the axis, for the player to position
  changed  the same slot, where this round's reference holds a different part --
           a higher-NA objective, say -> the player keeps the part they have. The
           round's own rules then tell them it is not good enough, which is the
           puzzle; installing the right one for them would be handing over the
           answer, and taking theirs away would be removal
  gone     a component outside this round's frame -> parked, not removed
"""

from __future__ import annotations

from dataclasses import replace

from ..bench.bench import Bench, BenchElement

# The frame of the experiment rather than things you align: these keep the
# reference position when they are new, because a round is not a puzzle about where
# the lamp sits or where the image is wanted. `field_stop` is the kind the specimen
# and the intermediate-image plane share -- both are planes the build has to satisfy,
# not components to position.
FIXTURE_KINDS = frozenset({"lamp", "detector", "field_stop"})

# The specimen is what the round hands you to look at, not a component you own, so
# it always comes from the round -- dye and all. Carrying the previous round's
# specimen forward is how round 10 came to open already solved: the old cube and
# the old fluorophore agreed with each other, and the round's rule had nothing to
# object to.
# Keyed by name, not kind: the specimen shares the `field_stop` kind with the
# intermediate-image plane, which the player does have to work for.
ROUND_SUPPLIES_NAMES = frozenset({"specimen"})

# Where an unplaced component waits: along the near end of the axis, spaced so
# several are distinguishable, and nowhere near any solution.
TRAY_START_MM = 3.0
TRAY_SPACING_MM = 4.0
TRAY_FLAG = "in_tray"


def _spec_of(element: BenchElement) -> tuple:
    """Everything about a component except where it sits."""
    return (
        element.kind,
        element.focal_length_mm,
        element.semi_diameter_mm,
        element.catalog_key,
        element.label,
        tuple(sorted((k, repr(v)) for k, v in element.metadata.items())),
    )


def is_in_tray(element: BenchElement) -> bool:
    return bool(element.metadata.get(TRAY_FLAG))


def tray_positions(count: int) -> list[float]:
    return [TRAY_START_MM + i * TRAY_SPACING_MM for i in range(count)]


def derive_starting_build(reference: Bench, carried: Bench | None) -> Bench:
    """The bench a round opens on, given what the player arrives carrying.

    With no carried build -- round 1, or a round reached before its predecessor was
    solved -- every component this round needs is new, so the whole thing arrives in
    the tray and the player assembles it. That is the same rule, not a special case.
    """
    if carried is None:
        carried = Bench(
            elements=[],
            folds=list(reference.folds),
            arms=list(reference.arms.values()),
            origin=tuple(reference.origin),
            initial_direction=tuple(reference.initial_direction),
        )

    wanted = {e.name: e for e in reference.elements}
    held = {e.name: e for e in carried.elements}
    elements: list[BenchElement] = []
    new_names: list[str] = []

    # Everything the player is carrying, in the order they had it.
    for name, mine in held.items():
        target = wanted.get(name)
        if target is None:
            # Outside this round's frame: on the shelf, still theirs.
            elements.append(replace(mine, enabled=False))
            continue
        if target.name in ROUND_SUPPLIES_NAMES:
            elements.append(replace(target, enabled=True))
            continue
        # Their part, their position, untouched -- including when this round wants
        # a better one in that slot. Finding that out is the round.
        elements.append(replace(mine, enabled=True))

    # Whatever this round adds.
    additions = [wanted[n] for n in wanted if n not in held]
    trayed = [a for a in additions if a.kind not in FIXTURE_KINDS]
    for element, s in zip(trayed, tray_positions(len(trayed))):
        elements.append(
            replace(element, s=s, enabled=True, metadata={**element.metadata, TRAY_FLAG: True})
        )
        new_names.append(element.name)
    for element in additions:
        if element.kind in FIXTURE_KINDS:
            elements.append(replace(element, enabled=True))
            new_names.append(element.name)

    # The arms and folds belong to the round's frame, not to the carried build: an
    # epi round needs its vertical illuminator branch to exist before anything can
    # be put on it.
    bench = Bench(
        elements=elements,
        folds=list(reference.folds),
        arms=list(reference.arms.values()),
        origin=tuple(reference.origin),
        initial_direction=tuple(reference.initial_direction),
    )
    bench.introduced = tuple(new_names)
    return bench
