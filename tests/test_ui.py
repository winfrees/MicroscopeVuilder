"""M4 gate: the workspace builds, draws, and stays in step with the engine.

Runs headless via the offscreen Qt platform, so CI exercises the real widgets
rather than a mock.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="the workspace needs the 'ui' extra")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from microscopevuilder.bench.bench import Bench, BenchElement, Fold  # noqa: E402
from microscopevuilder.game.rounds import get_round  # noqa: E402
from microscopevuilder.ui.geometry import axis_frame, point_at  # noqa: E402
from microscopevuilder.ui.scene import BenchScene, ElementItem  # noqa: E402
from microscopevuilder.ui.trace_model import build_trace_model  # noqa: E402
from microscopevuilder.ui.workspace import Workspace  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --- projection, no Qt needed ------------------------------------------------


def test_axis_frame_normal_follows_a_fold():
    # Before the fold the axis runs along +x, so heights are drawn along z. After a
    # 90-degree turn into +z, heights must be drawn along x instead -- otherwise
    # every element past a dichroic is drawn lying along the beam.
    bench = Bench([BenchElement("end", 200.0, "detector", 5.0)], [Fold(100.0, (0, 0, 1))])

    _, tangent_before, normal_before = axis_frame(bench, 50.0)
    _, tangent_after, normal_after = axis_frame(bench, 150.0)

    np.testing.assert_allclose(tangent_before, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.abs(normal_before), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(tangent_after, [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.abs(normal_after), [1.0, 0.0], atol=1e-6)


def test_ray_height_is_drawn_across_the_axis_after_a_fold():
    bench = Bench([BenchElement("end", 200.0, "detector", 5.0)], [Fold(100.0, (0, 0, 1))])
    np.testing.assert_allclose(point_at(bench, 50.0, 2.0), [50.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(point_at(bench, 150.0, 2.0), [98.0, 50.0], atol=1e-6)


# --- the live model ----------------------------------------------------------


def test_trace_model_reports_the_marginal_ray_crossing_at_the_image():
    rnd = get_round(1)
    bench = rnd.reference_build()
    model = build_trace_model(bench, rnd.s_object)
    marginal = next(r for r in model.rays if r.label == "marginal")

    assert model.image_s == pytest.approx(300.0)
    at_image = min(marginal.samples, key=lambda p: abs(p[0] - 300.0))
    assert at_image[1] == pytest.approx(0.0, abs=1e-6)


def test_trace_model_lists_both_conjugate_sets():
    rnd = get_round(2)
    model = build_trace_model(rnd.reference_build(), rnd.s_object)
    kinds = {c.kind for c in model.conjugates}
    assert "field" in kinds and "aperture" in kinds
    assert any(c.label == "specimen" for c in model.conjugates)


# --- the scene ---------------------------------------------------------------


def test_scene_creates_one_draggable_item_per_element(app):
    rnd = get_round(2)
    bench = rnd.reference_build()
    scene = BenchScene()
    scene.set_bench(bench, rnd.s_object)

    items = [i for i in scene.items() if isinstance(i, ElementItem)]
    assert len(items) == len(bench.elements)
    for item in items:
        assert item.flags() & QtWidgets.QGraphicsItem.ItemIsMovable


def test_dragging_an_element_updates_the_bench_and_retraces(app):
    rnd = get_round(1)
    bench = rnd.reference_build()
    scene = BenchScene()
    scene.set_bench(bench, rnd.s_object)
    assert scene.model.image_s == pytest.approx(300.0)

    scene.commit_element_move("lens", 40.0)
    assert bench.get("lens").s == 40.0
    # The lens moved 40 mm downstream, so its object distance shrank from 60 to
    # 100 mm and the image must land somewhere new.
    assert scene.model.image_s != pytest.approx(300.0)


def test_a_drag_cannot_push_an_element_behind_the_origin(app):
    rnd = get_round(1)
    bench = rnd.reference_build()
    scene = BenchScene()
    scene.set_bench(bench, rnd.s_object)
    scene.commit_element_move("lens", -500.0)
    assert bench.get("lens").s == 0.0


def test_scene_draws_rays_and_the_conjugate_ribbon(app):
    rnd = get_round(2)
    scene = BenchScene()
    scene.set_bench(rnd.reference_build(), rnd.s_object)
    with_everything = len(scene.items())

    scene.show_rays = False
    scene.refresh()
    without_rays = len(scene.items())

    scene.show_ribbon = False
    scene.refresh()
    without_either = len(scene.items())

    assert without_rays < with_everything
    assert without_either < without_rays


def test_scene_renders_to_an_image_without_error(app, tmp_path):
    # The cheapest guard against a paint() that throws: actually paint it.
    rnd = get_round(2)
    scene = BenchScene()
    scene.set_bench(rnd.reference_build(), rnd.s_object)
    rect = scene.itemsBoundingRect()
    assert rect.width() > 0

    image = QtGui.QImage(600, 300, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("#12151a"))
    painter = QtGui.QPainter(image)
    scene.render(painter)
    painter.end()

    assert image.save(str(tmp_path / "bench.png"))


# --- the window --------------------------------------------------------------


def test_workspace_opens_on_a_passing_reference_build(app):
    w = Workspace(2)
    assert w.report_panel.headline.text() == "ROUND PASSED"
    assert w.report_panel.list.topLevelItemCount() == len(w.round.grade(w.bench).results)


def test_workspace_scorecard_tracks_an_edit(app):
    w = Workspace(2)
    w.bench.move("eyepiece", 300.0)
    w.refresh_live()
    assert "failing" in w.report_panel.headline.text()

    w.load_reference()
    assert w.report_panel.headline.text() == "ROUND PASSED"


def test_failing_rows_expand_themselves_with_a_remedy(app):
    w = Workspace(2)
    w.bench.move("eyepiece", 300.0)
    w.refresh_live()

    failing = [
        w.report_panel.list.topLevelItem(i)
        for i in range(w.report_panel.list.topLevelItemCount())
        if w.report_panel.list.topLevelItem(i).isExpanded()
    ]
    assert failing
    texts = [
        failing[0].child(j).text(1) for j in range(failing[0].childCount())
    ]
    assert any(t.startswith("remedy:") for t in texts)


def test_status_line_explains_an_afocal_system_rather_than_going_quiet(app):
    # A visual scope sends collimated light to the eye, so there IS no finite
    # image plane. Reporting nothing would read as a bug.
    w = Workspace(2)
    assert "afocal" in w.statusBar().currentMessage()


def test_status_line_reports_the_image_plane_when_one_exists(app):
    w = Workspace(1)
    message = w.statusBar().currentMessage()
    assert "image at s = 300.00 mm" in message
    assert "M = 5.00x" in message


def test_scope_view_accepts_a_synthesized_image(app):
    from microscopevuilder.imaging.build_optics import resolve_build_optics
    from microscopevuilder.imaging.synthesis import RenderedImage

    w = Workspace(1)
    bench = w.bench
    optics = resolve_build_optics(bench, w.round.s_object, "screen", 0.5461)
    rendered = RenderedImage(
        np.random.default_rng(0).random((64, 64)), 0.1, optics, []
    )
    w._on_image(rendered)
    assert w.scope.pixmap() is not None
    assert w.scope.pixmap().width() == 64
    assert w.metrics_panel.table.topLevelItemCount() > 0


def test_image_worker_renders_the_actual_bench_off_thread(app):
    from microscopevuilder.imaging.synthesis import RenderSpec
    from microscopevuilder.ui.workspace import SPECIMENS, ImageWorker

    rnd = get_round(11)
    bench = rnd.reference_build()
    spec = RenderSpec(specimen=SPECIMENS["sine grating"], n=128, detector_name="sensor")

    received = []
    worker = ImageWorker(bench, rnd.s_object, spec)
    worker.finished_image.connect(received.append)
    worker.start()
    assert worker.wait(60_000), "image synthesis timed out"
    app.processEvents()

    assert received
    rendered = received[0]
    assert rendered.intensity.shape == (128, 128)
    assert rendered.intensity.min() >= 0.0
    # The point of M10: the render knows which objective it went through.
    assert rendered.optics.aberration_source.startswith("catalog:")
    assert rendered.optics.na == pytest.approx(0.75)


def test_running_a_test_populates_the_measurements_panel(app):
    w = Workspace(11)
    w.run_test()
    assert w.worker.wait(60_000)
    app.processEvents()

    labels = [
        w.metrics_panel.table.topLevelItem(i).text(0)
        for i in range(w.metrics_panel.table.topLevelItemCount())
    ]
    assert "Strehl ratio" in labels
    assert "field uniformity (corner/centre)" in labels
    assert "pedagogical" in w.metrics_panel.provenance.text()


def test_the_measurements_panel_names_its_aberration_provenance(app):
    # A student must never mistake a teaching budget for a datasheet value.
    w = Workspace(11)
    w.run_test()
    assert w.worker.wait(60_000)
    app.processEvents()
    assert "not vendor data" in w.metrics_panel.provenance.text()


# --- the white card ----------------------------------------------------------


def test_workspace_can_place_and_clear_cards(app):
    w = Workspace(2)
    assert "place a card" in w.card_panel.reading.text()

    w.add_card(193.6)
    assert "sharp image" in w.card_panel.reading.text()
    assert len(w.bench.cards()) == 1

    w.clear_cards()
    assert not w.bench.cards()
    assert "place a card" in w.card_panel.reading.text()


def test_placing_a_card_does_not_change_the_scorecard(app):
    # The probe must be invisible to the rules, or players will learn to game it.
    w = Workspace(2)
    before = w.report_panel.headline.text()
    rows_before = w.report_panel.list.topLevelItemCount()
    w.add_card(120.0)
    assert w.report_panel.headline.text() == before
    assert w.report_panel.list.topLevelItemCount() == rows_before


def test_card_panel_lists_the_planes_it_found(app):
    w = Workspace(2)
    w.add_card(100.0)
    labels = [
        w.card_panel.planes.topLevelItem(i).text(0)
        for i in range(w.card_panel.planes.topLevelItemCount())
    ]
    assert any("image" in x for x in labels)
    assert any("pupil" in x for x in labels)


def test_a_card_is_drawn_in_the_scene(app):
    rnd = get_round(2)
    bench = rnd.reference_build()
    scene = BenchScene()
    scene.set_bench(bench, rnd.s_object)
    before = len([i for i in scene.items() if isinstance(i, ElementItem)])

    bench.add_card(120.0)
    scene.set_bench(bench, rnd.s_object)
    after = [i for i in scene.items() if isinstance(i, ElementItem)]
    assert len(after) == before + 1

    image = QtGui.QImage(400, 200, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    scene.render(painter)
    painter.end()


# --- illumination stands -----------------------------------------------------


def test_illumination_rounds_open_without_tracing_backwards(app):
    # The lamp, collector and diaphragms all sit upstream of the specimen, which
    # is the object plane for the imaging trace. Sampling the whole bench from the
    # specimen tried to trace backwards and threw.
    for number in (3, 4, 5):
        w = Workspace(number)
        assert w.report_panel.list.topLevelItemCount() > 0


def test_the_illumination_path_is_traced_as_its_own_ray_set(app):
    w = Workspace(4)
    labels = {r.label for r in w.scene.model.rays}
    assert "illumination_axial" in labels
    assert "marginal" in labels


def test_illumination_and_imaging_paths_toggle_independently(app):
    w = Workspace(4)
    both = len(w.scene.items())
    w._toggle_illumination(False)
    imaging_only = len(w.scene.items())
    w._toggle_imaging(False)
    neither = len(w.scene.items())
    assert neither < imaging_only < both


def test_a_card_upstream_of_the_specimen_explains_itself(app):
    # Rather than raising out of the panel, say what is wrong.
    w = Workspace(4)
    w.add_card(50.0)
    assert "upstream of the specimen" in w.card_panel.reading.text()


def test_reading_a_card_upstream_of_the_object_raises_clearly():
    from microscopevuilder.bench.probe import read_card as read

    rnd = get_round(4)
    bench = rnd.reference_build()
    with pytest.raises(ValueError, match="upstream of the object plane"):
        read(bench, rnd.s_object, 50.0)


# --- branched (episcopic) benches --------------------------------------------


@pytest.mark.parametrize("number", [9, 10])
def test_epi_rounds_open_and_draw(app, number):
    w = Workspace(number)
    assert w.report_panel.headline.text() == "ROUND PASSED"
    labels = {r.label for r in w.scene.model.rays}
    assert "illumination_axial" in labels and "marginal" in labels

    image = QtGui.QImage(700, 350, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    w.scene.render(painter)
    painter.end()


def test_illumination_rays_on_an_arm_are_tagged_with_that_arm(app):
    w = Workspace(9)
    illumination = [r for r in w.scene.model.rays if r.label.startswith("illumination")]
    assert illumination
    assert all(r.arm == "epi" for r in illumination)
    assert all(r.arm == "main" for r in w.scene.model.rays if not r.label.startswith("illumination"))


def test_arm_elements_are_placed_off_the_main_axis(app):
    # If the arm were drawn on the main axis the epi stand would be unreadable.
    rnd = get_round(9)
    bench = rnd.reference_build()
    scene = BenchScene()
    scene.set_bench(bench, rnd.s_object)

    lamp_item = next(i for i in scene.items() if isinstance(i, ElementItem) and i.name == "lamp")
    objective_item = next(
        i for i in scene.items() if isinstance(i, ElementItem) and i.name == "objective"
    )
    assert lamp_item.pos() != objective_item.pos()


@pytest.mark.parametrize("number", [0, 11, 12])
def test_infinity_and_sandbox_rounds_open(app, number):
    w = Workspace(number)
    assert w.report_panel.list.topLevelItemCount() > 0
    image = QtGui.QImage(700, 350, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    w.scene.render(painter)
    painter.end()


def test_a_turret_draws_every_objective_even_the_ones_out_of_the_path(app):
    # They are fitted; the renderer should show them, and only the trace ignores them.
    w = Workspace(12)
    items = [i for i in w.scene.items() if isinstance(i, ElementItem)]
    assert sum(1 for i in items if "objective" in i.name) == 3
