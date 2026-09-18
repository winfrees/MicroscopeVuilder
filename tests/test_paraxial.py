"""Golden tests: textbook cases with known closed-form answers.

These are the anchors. If one of these breaks, the engine is wrong, not the test.
"""

import numpy as np
import pytest

from microscopevuilder.optics.paraxial import (
    ParaxialSystem,
    Ray,
    aperture,
    thin_lens,
)


def test_thin_lens_2f_gives_unit_inverted_magnification():
    # Object at 2f images at 2f with M = -1. The oldest check there is.
    f = 50.0
    sys = ParaxialSystem([thin_lens("L", 100.0, f, 12.5)])
    s_obj = 100.0 - 2 * f
    s_img = sys.image_plane(s_obj)
    assert s_img == pytest.approx(100.0 + 2 * f)
    assert sys.magnification(s_obj, s_img) == pytest.approx(-1.0)


@pytest.mark.parametrize("s_o", [75.0, 150.0, 300.0])
def test_thin_lens_obeys_the_imaging_equation(s_o):
    # 1/s' = 1/f - 1/s_o, with s_o the object distance in front of the lens.
    f = 50.0
    sys = ParaxialSystem([thin_lens("L", 500.0, f, 25.0)])
    s_img = sys.image_plane(500.0 - s_o)
    expected = 1.0 / (1.0 / f - 1.0 / s_o)
    assert s_img - 500.0 == pytest.approx(expected)
    assert sys.magnification(500.0 - s_o, s_img) == pytest.approx(-expected / s_o)


def test_object_at_front_focus_images_to_infinity():
    # This is the defining property of infinity-corrected optics: the specimen sits
    # at the objective's front focal plane and the space behind it is collimated.
    f = 4.0
    sys = ParaxialSystem([thin_lens("obj", 100.0, f, 3.0)])
    assert sys.image_plane(100.0 - f) is None

    out = sys.trace(Ray(0.0, 0.05), 100.0 - f, 200.0)
    assert out.u == pytest.approx(0.0, abs=1e-12)  # collimated


def test_infinity_corrected_magnification_is_the_tube_lens_ratio():
    # M = f_tube / f_obj, and -- the point of the whole design -- it does not depend
    # on the separation between objective and tube lens.
    f_obj, f_tube = 2.0, 200.0  # Nikon CFI60 convention
    for gap in (50.0, 120.0, 250.0):
        sys = ParaxialSystem(
            [thin_lens("obj", 0.0, f_obj, 3.0), thin_lens("tube", gap, f_tube, 12.0)]
        )
        s_obj = -f_obj
        s_img = sys.image_plane(s_obj, search_to=gap + 2 * f_tube)
        assert s_img == pytest.approx(gap + f_tube)
        assert abs(sys.magnification(s_obj, s_img)) == pytest.approx(f_tube / f_obj)


def test_finite_tube_compound_microscope_matches_din_convention():
    # A DIN 160 mm objective is labelled with M = 160 / f_obj, referred to the
    # intermediate image at the tube's far end.
    f_obj = 1.6  # nominally a 100x
    sys = ParaxialSystem([thin_lens("obj", 0.0, f_obj, 1.5)])
    s_img = 160.0
    # Solve for the object distance that actually lands the image at 160 mm.
    s_obj = -1.0 / (1.0 / f_obj - 1.0 / s_img)
    assert sys.image_plane(s_obj, search_to=200.0) == pytest.approx(s_img)
    assert abs(sys.magnification(s_obj, s_img)) == pytest.approx(100.0, rel=0.02)


def test_aperture_stop_is_the_element_that_actually_limits_the_cone():
    # The stop is not simply the physically smallest element: it is the one whose
    # clear aperture runs out first for the ray height that reaches it. Here the
    # lens sits five times further from the object, so its ray height is five times
    # larger, and it wins the competition until the iris is stopped well down.
    def stop_for(iris_semi):
        sys = ParaxialSystem(
            [aperture("iris", 10.0, iris_semi), thin_lens("L", 50.0, 40.0, 4.0)]
        )
        return sys.aperture_stop(0.0).name

    assert stop_for(5.0) == "L"  # wide-open iris: the lens limits the cone
    assert stop_for(0.5) == "iris"  # stopped down: the iris takes over
    # Crossover is at semi_diameter = 0.8 mm, where the two ratios are equal.
    assert stop_for(0.79) == "iris"
    assert stop_for(0.81) == "L"


def test_marginal_ray_exactly_fills_the_stop():
    sys = ParaxialSystem([aperture("iris", 20.0, 2.0), thin_lens("L", 60.0, 40.0, 10.0)])
    s_obj = 0.0
    m = sys.marginal_ray(s_obj)
    at_stop = sys.trace(m, s_obj, sys.aperture_stop(s_obj).s)
    assert abs(at_stop.y) == pytest.approx(2.0)


def test_numerical_aperture_scales_with_immersion_index():
    sys = ParaxialSystem([aperture("iris", 10.0, 3.0)])
    dry = sys.object_space_na(0.0, n=1.0)
    oil = sys.object_space_na(0.0, n=1.515)
    assert dry == pytest.approx(0.3, rel=1e-6)
    assert oil == pytest.approx(0.3 * 1.515, rel=1e-6)


def test_system_matrix_is_unimodular():
    # det(ABCD) = n0/n1 = 1 in air. A drifting determinant means a composition bug.
    sys = ParaxialSystem(
        [
            thin_lens("a", 10.0, 25.0, 10.0),
            thin_lens("b", 80.0, -40.0, 10.0),
            thin_lens("c", 150.0, 60.0, 10.0),
        ]
    )
    assert np.linalg.det(sys.between(0.0, 300.0)) == pytest.approx(1.0)


def test_between_excludes_the_element_sitting_on_the_end_plane():
    # Tracing "to the sensor" must not apply the sensor, and tracing "from the lens"
    # must apply the lens. This boundary rule is load-bearing for image_plane().
    sys = ParaxialSystem([thin_lens("L", 50.0, 50.0, 10.0)])
    assert sys.between(50.0, 50.0)[1, 0] == pytest.approx(0.0)  # end-exclusive
    assert sys.between(50.0, 60.0)[1, 0] == pytest.approx(-1 / 50.0)  # start-inclusive


def test_reversed_plane_order_is_rejected():
    sys = ParaxialSystem([thin_lens("L", 50.0, 50.0, 10.0)])
    with pytest.raises(ValueError):
        sys.between(100.0, 0.0)
