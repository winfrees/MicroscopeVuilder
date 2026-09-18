"""CFI60 catalog invariants."""

import pytest

from microscopevuilder.bench.catalog import load_catalog


@pytest.fixture(scope="module")
def cat():
    return load_catalog()


def test_cfi60_system_constants(cat):
    assert cat.system.tube_lens_focal_length_mm == 200.0
    assert cat.system.parfocal_distance_mm == 60.0


def test_every_objective_focal_length_follows_f_equals_ftube_over_m(cat):
    ft = cat.system.tube_lens_focal_length_mm
    for o in cat.objectives.values():
        assert o.focal_length_mm(ft) == pytest.approx(ft / o.magnification)
    assert cat.objective("cfi_plan_apo_20x").focal_length_mm(ft) == 10.0


def test_the_same_objective_is_not_the_same_magnification_on_a_180mm_stand(cat):
    # The trap round 11 sets: a CFI60 20x on an Olympus 180 mm tube lens is 18x.
    o = cat.objective("cfi_plan_apo_20x")
    f = o.focal_length_mm(200.0)
    assert 180.0 / f == pytest.approx(18.0)


def test_na_never_exceeds_the_immersion_index(cat):
    # NA = n sin(u) <= n. A catalog entry violating this is a typo, not a product.
    for o in cat.objectives.values():
        assert o.na < o.medium_index, o.key


def test_dry_objectives_stay_under_na_1(cat):
    for o in cat.objectives.values():
        if o.immersion == "air":
            assert o.na < 1.0, o.key


def test_working_distance_falls_as_na_rises(cat):
    # Within a correction grade, higher NA means shorter working distance. This is
    # the tradeoff the player has to feel in round 5.
    apos = sorted(
        (o for o in cat.objectives.values() if o.grade == "plan_apochromat"),
        key=lambda o: o.na,
    )
    wds = [o.working_distance_mm for o in apos]
    assert wds == sorted(wds, reverse=True)


def test_apochromats_carry_less_secondary_spectrum_than_achromats(cat):
    # The defining difference between the grades, and the point of round 6.
    # Stored as a fraction of focal length: the textbook figure for an achromat is
    # about f/2000, and an apochromat is roughly an order of magnitude better.
    achro = max(
        o.aberrations.chromatic_focus_fraction
        for o in cat.objectives.values()
        if o.grade == "plan_achromat"
    )
    apo = max(
        o.aberrations.chromatic_focus_fraction
        for o in cat.objectives.values()
        if o.grade == "plan_apochromat"
    )
    assert achro == pytest.approx(5e-4)  # f/2000
    assert apo < achro / 5


def test_aberration_budgets_are_labelled_as_teaching_values(cat):
    # Guard against a future contributor quietly presenting these as vendor data.
    for o in cat.objectives.values():
        assert not o.aberrations.is_vendor_data, o.key


def test_unverified_entries_are_reported_for_the_ui_disclaimer(cat):
    # Everything is unverified today; the point is that the list is non-empty and
    # reachable, so the UI can say so instead of implying datasheet authority.
    assert cat.unverified_entries()
