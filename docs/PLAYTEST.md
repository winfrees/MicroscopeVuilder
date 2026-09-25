# Playtest script — the conjugate-plane ribbon

The ribbon is the one design risk here that code cannot settle. Tests confirm
*numerically* that field and aperture planes interleave and land where they should.
Whether it is **legible to a person** is a different question.

This is the script for answering it: one person who has not seen the workspace, about
twenty minutes.

## Who

A graduate student in the biomedical sciences who has used a microscope but has not
aligned Köhler illumination from scratch. Someone who already knows Köhler will
pattern-match and tell you nothing.

## Setup

```sh
python -m microscopevuilder --ui --round 4
```

Do not explain the ribbon. Do not mention conjugate planes. Say only: "This is
round 4. Read the brief and try to solve it. Think out loud."

## What to watch for

Record the answer, and the time to reach it.

1. **Do they notice the ribbon at all?** If they solve the round without ever
   looking at it, it is decoration and should be cut or redesigned.
2. **Do they work out that the two rows are different kinds of plane?** Without
   being told. If not, labelling is not enough and the rows need separating
   visually — different shapes, not just different colours.
3. **Do they connect a tick moving to a component they dragged?** This is the
   core interaction. If the connection is not obvious, the ribbon needs to
   highlight the tick belonging to the selected element.
4. **Do they use the white card?** And if so, does the card teach them what the
   ribbon was already showing? If the card does all the work, the ribbon is
   redundant.
5. **At the end, ask: "what are the two rows?"** A player who has solved round 4
   should be able to answer roughly. If they cannot, they solved it by moving
   things until the scorecard went green, which is the failure mode this whole
   design is meant to avoid.

## Decision rule

- **Keep as is** if 3 of 5 questions land without prompting.
- **Redesign** if they solve the round without looking at the ribbon, or cannot say
  what the rows are afterwards.
- **Cut it** if the white card turns out to do the same job better. Two overlapping
  explanations of the same idea is worse than one good one.

## What has already been done

- Rows are labelled ("field / image planes", "aperture / pupil planes").
- Each tick is named with its plane.
- Hovering a tick gives its position in millimetres.
- The illumination and imaging ray bundles can be toggled independently, so the two
  can be compared one at a time.

None of this substitutes for watching someone use it.
