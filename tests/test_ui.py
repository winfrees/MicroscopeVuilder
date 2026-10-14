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


@pytest.fixture
def workspace(app):
    """A workspace that is kept alive and shut down cleanly.

    Letting the window be collected while a render is in flight destroys a running
    QThread, which aborts the process rather than failing a test.
    """
    windows = []

    def make(number: int) -> Workspace:
        window = Workspace(number)
        windows.append(window)
        return window

    yield make

    for window in windows:
        window.close()
        window.deleteLater()
    QtWidgets.QApplication.processEvents()


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


def test_workspace_opens_on_a_passing_reference_build(workspace, app):
    w = workspace(2)
    assert w.report_panel.headline.text() == "ROUND PASSED"
    # The live rules, plus the parts-budget row the workspace appends.
    assert w.report_panel.list.topLevelItemCount() == len(w.round.grade(w.bench).results) + 1


def test_workspace_scorecard_tracks_an_edit(workspace, app):
    w = workspace(2)
    w.bench.move("eyepiece", 300.0)
    w.refresh_live()
    assert "failing" in w.report_panel.headline.text()

    w.load_reference()
    assert w.report_panel.headline.text() == "ROUND PASSED"


def test_failing_rows_expand_themselves_with_a_remedy(workspace, app):
    w = workspace(2)
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


def test_status_line_explains_an_afocal_system_rather_than_going_quiet(workspace, app):
    # A visual scope sends collimated light to the eye, so there IS no finite
    # image plane. Reporting nothing would read as a bug.
    w = workspace(2)
    assert "afocal" in w.statusBar().currentMessage()


def test_status_line_reports_the_image_plane_when_one_exists(workspace, app):
    w = workspace(1)
    message = w.statusBar().currentMessage()
    assert "image at s = 300.00 mm" in message
    assert "M = 5.00x" in message


def test_scope_view_accepts_a_synthesized_image(workspace, app):
    from microscopevuilder.imaging.build_optics import resolve_build_optics
    from microscopevuilder.imaging.synthesis import RenderedImage

    w = workspace(1)
    bench = w.bench
    optics = resolve_build_optics(bench, w.round.s_object, "screen", 0.5461)
    rendered = RenderedImage(
        np.random.default_rng(0).random((64, 64)), 0.1, optics, []
    )
    w.scope.set_planes(w.scope_planes())
    w._on_plane("screen", rendered)
    pane = w.scope.panes["screen"]
    assert pane.pixmap() is not None
    assert pane.pixmap().width() == 64
    assert w.metrics_panel.table.topLevelItemCount() > 0


def test_image_worker_renders_the_actual_bench_off_thread(app):
    from microscopevuilder.imaging.synthesis import RenderSpec
    from microscopevuilder.ui.workspace import SPECIMENS, ImageWorker

    rnd = get_round(11)
    bench = rnd.reference_build()
    spec = RenderSpec(specimen=SPECIMENS["sine grating"], n=128, detector_name="sensor")

    from microscopevuilder.ui.render_jobs import render_planes

    received = []
    done = []
    planes = render_planes(bench, "sensor")
    worker = ImageWorker(bench, rnd.s_object, spec, planes)
    worker.plane_ready.connect(lambda name, r: received.append((name, r)))
    worker.finished_all.connect(done.append)
    worker.start()
    assert worker.wait(60_000), "image synthesis timed out"
    app.processEvents()

    assert received
    assert done == [None]  # no round passed to the worker, so nothing measured
    name, rendered = received[0]
    assert name == "sensor"
    assert rendered.intensity.shape == (128, 128)
    assert rendered.intensity.min() >= 0.0
    # The point of M10: the render knows which objective it went through.
    assert rendered.optics.aberration_source.startswith("catalog:")
    assert rendered.optics.na == pytest.approx(0.75)


def test_running_a_test_populates_the_measurements_panel(workspace, app):
    w = workspace(11)
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


def test_the_measurements_panel_names_its_aberration_provenance(workspace, app):
    # A student must never mistake a teaching budget for a datasheet value.
    w = workspace(11)
    w.run_test()
    assert w.worker.wait(60_000)
    app.processEvents()
    assert "not vendor data" in w.metrics_panel.provenance.text()


# --- the white card ----------------------------------------------------------


def test_workspace_can_place_and_clear_cards(workspace, app):
    w = workspace(2)
    assert "place a card" in w.card_panel.reading.text()

    w.add_card(193.6)
    assert "sharp image" in w.card_panel.reading.text()
    assert len(w.bench.cards()) == 1

    w.clear_cards()
    assert not w.bench.cards()
    assert "place a card" in w.card_panel.reading.text()


def test_placing_a_card_does_not_change_the_scorecard(workspace, app):
    # The probe must be invisible to the rules, or players will learn to game it.
    w = workspace(2)
    before = w.report_panel.headline.text()
    rows_before = w.report_panel.list.topLevelItemCount()
    w.add_card(120.0)
    assert w.report_panel.headline.text() == before
    assert w.report_panel.list.topLevelItemCount() == rows_before


def test_card_panel_lists_the_planes_it_found(workspace, app):
    w = workspace(2)
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


def test_illumination_rounds_open_without_tracing_backwards(workspace, app):
    # The lamp, collector and diaphragms all sit upstream of the specimen, which
    # is the object plane for the imaging trace. Sampling the whole bench from the
    # specimen tried to trace backwards and threw.
    for number in (3, 4, 5):
        w = workspace(number)
        assert w.report_panel.list.topLevelItemCount() > 0


def test_the_illumination_path_is_traced_as_its_own_ray_set(workspace, app):
    w = workspace(4)
    labels = {r.label for r in w.scene.model.rays}
    assert "illumination_axial" in labels
    assert "marginal" in labels


def test_illumination_and_imaging_paths_toggle_independently(workspace, app):
    w = workspace(4)
    both = len(w.scene.items())
    w._toggle_illumination(False)
    imaging_only = len(w.scene.items())
    w._toggle_imaging(False)
    neither = len(w.scene.items())
    assert neither < imaging_only < both


def test_a_card_upstream_of_the_specimen_explains_itself(workspace, app):
    # Rather than raising out of the panel, say what is wrong.
    w = workspace(4)
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
def test_epi_rounds_open_and_draw(workspace, app, number):
    w = workspace(number)
    assert w.report_panel.headline.text() == "ROUND PASSED"
    labels = {r.label for r in w.scene.model.rays}
    assert "illumination_axial" in labels and "marginal" in labels

    image = QtGui.QImage(700, 350, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    w.scene.render(painter)
    painter.end()


def test_illumination_rays_on_an_arm_are_tagged_with_that_arm(workspace, app):
    w = workspace(9)
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
def test_infinity_and_sandbox_rounds_open(workspace, app, number):
    w = workspace(number)
    assert w.report_panel.list.topLevelItemCount() > 0
    image = QtGui.QImage(700, 350, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    w.scene.render(painter)
    painter.end()


def test_a_turret_draws_every_objective_even_the_ones_out_of_the_path(workspace, app):
    # They are fitted; the renderer should show them, and only the trace ignores them.
    w = workspace(12)
    items = [i for i in w.scene.items() if isinstance(i, ElementItem)]
    assert sum(1 for i in items if "objective" in i.name) == 3


# --- measured rounds ---------------------------------------------------------


@pytest.mark.parametrize("number", [6, 7, 8])
def test_measured_rounds_open_and_run(workspace, app, number):
    w = workspace(number)
    assert w.report_panel.list.topLevelItemCount() > 0
    w.run_test()
    assert w.worker.wait(120_000)
    app.processEvents()
    assert "measured rules" in w.statusBar().currentMessage()


def test_measured_rule_rows_appear_above_the_raw_numbers(workspace, app):
    # The rules are the verdict; the raw measurements are the evidence for it.
    w = workspace(7)
    w.run_test()
    assert w.worker.wait(120_000)
    app.processEvents()
    assert w.metrics_panel.table.topLevelItem(0).text(0) == "Field flatness"


def test_a_geometry_only_round_shows_no_measured_verdict(workspace, app):
    w = workspace(11)
    w.run_test()
    assert w.worker.wait(120_000)
    app.processEvents()
    assert "measured rules" not in w.statusBar().currentMessage()
    assert w.metrics_panel.table.topLevelItem(0).text(0) == "Michelson contrast"


# --- the game shell ----------------------------------------------------------


def test_the_scorecard_shows_a_star_rating(workspace, app):
    w = workspace(11)
    assert "★★★" in w.report_panel.score_line.text()
    assert "parts" in w.report_panel.score_line.text()


def test_breaking_the_build_drops_the_rating_and_offers_the_diff(workspace, app):
    w = workspace(11)
    assert not w.diff_action.isEnabled()

    # Far enough out to fail even the 5% practice tolerance, which is what the
    # game grades with by default.
    w.bench.move("sensor", w.bench.get("sensor").s + 40.0)
    w.refresh_live()
    assert "☆☆☆" in w.report_panel.score_line.text()
    assert w.diff_action.isEnabled()


def test_the_workspace_offers_the_whole_specimen_library(workspace, app):
    from microscopevuilder.imaging.specimens import SPECIMEN_LIBRARY

    w = workspace(11)
    offered = {w.specimen.itemText(i) for i in range(w.specimen.count())}
    assert offered == set(SPECIMEN_LIBRARY)


def test_the_budget_row_appears_on_the_scorecard(workspace, app):
    w = workspace(11)
    names = [
        w.report_panel.list.topLevelItem(i).text(0)
        for i in range(w.report_panel.list.topLevelItemCount())
    ]
    assert "Parts budget" in names


# --- zoom, ruler and placement aids ------------------------------------------


def test_the_view_zooms_and_reports_its_scale(workspace, app):
    w = workspace(11)
    w.view.fit_bench()
    fitted = w.view.scale_factor

    w.view.zoom_in()
    assert w.view.scale_factor > fitted
    assert "x" in w.zoom_label.text()
    # The scene is told, so ruler density can follow the zoom.
    assert w.scene.view_scale == pytest.approx(w.view.scale_factor)

    w.view.zoom_out()
    w.view.zoom_out()
    assert w.view.scale_factor < fitted


def test_zoom_is_clamped_at_both_ends(workspace, app):
    from microscopevuilder.ui.view import MAX_SCALE, MIN_SCALE

    w = workspace(11)
    for _ in range(80):
        w.view.zoom_in()
    assert w.view.scale_factor == pytest.approx(MAX_SCALE)

    for _ in range(200):
        w.view.zoom_out()
    assert w.view.scale_factor == pytest.approx(MIN_SCALE)


def test_zooming_in_makes_a_pixel_finer_than_the_tolerance(workspace, app):
    # The whole point of adding zoom: at fit-to-window one pixel overshoots the
    # depth of focus, and zoomed in it does not.
    from microscopevuilder.rules.tolerances import image_side_depth_of_focus_mm

    w = workspace(11)
    w.view.resize(760, 400)
    w.view.fit_bench()
    tolerance = image_side_depth_of_focus_mm(0.5461, 0.75, 20.0)

    mm_per_px_fitted = 1.0 / w.view.scale_factor
    assert mm_per_px_fitted > tolerance

    for _ in range(12):
        w.view.zoom_in()
    assert 1.0 / w.view.scale_factor < tolerance


def test_the_ruler_draws_and_can_be_switched_off(workspace, app):
    w = workspace(11)
    with_ruler = len(w.scene.items())
    w.ruler_toggle.setChecked(False)
    assert len(w.scene.items()) < with_ruler
    w.ruler_toggle.setChecked(True)
    assert len(w.scene.items()) == with_ruler


def test_dragging_snaps_to_the_grid_and_to_optical_planes(workspace, app):
    w = workspace(11)
    # A position close to the image plane is captured by it.
    snapped, captured = w.scene.snap_position(379.2)
    assert captured and snapped == pytest.approx(380.0)

    # Away from any plane it falls back to the grid step.
    w.scene.snap_step_mm = 0.5
    snapped, captured = w.scene.snap_position(300.3)
    assert not captured and snapped == pytest.approx(300.5)


def test_alt_suspends_snapping(workspace, app):
    w = workspace(11)
    w.view._set_snapping(False)
    assert w.scene.snap_position(379.2) == (379.2, False)
    w.view._set_snapping(True)
    assert w.scene.snap_position(379.2)[1] is True


def test_plane_snapping_can_be_turned_off_independently(workspace, app):
    w = workspace(11)
    w.plane_snap.setChecked(False)
    snapped, captured = w.scene.snap_position(379.2)
    assert not captured
    assert snapped == pytest.approx(379.2, abs=0.05)  # grid only


def test_switching_to_inches_changes_the_snap_steps_offered(workspace, app):
    w = workspace(11)
    metric = [w.snap_box.itemText(i) for i in range(w.snap_box.count())]
    w.unit_box.setCurrentText("in")
    imperial = [w.snap_box.itemText(i) for i in range(w.snap_box.count())]

    assert any("mm" in text for text in metric)
    assert all('"' in text for text in imperial)
    assert w.scene.unit == "in"


def test_typing_a_position_places_a_component_exactly(workspace, app):
    # The reliable route: no pixels involved.
    w = workspace(11)
    w._on_element_selected("sensor")
    w._set_element_position("sensor", 372.5)
    assert w.bench.get("sensor").s == pytest.approx(372.5)
    w._set_element_position("sensor", 380.0)
    assert w.bench.get("sensor").s == pytest.approx(380.0)


def test_the_inspector_shows_and_edits_position_in_the_chosen_unit(workspace, app):
    from microscopevuilder.ui.ruler import MM_PER_INCH

    w = workspace(11)
    w._on_element_selected("sensor")
    assert w.inspector.position_spin.value() == pytest.approx(380.0)

    w.unit_box.setCurrentText("in")
    w._on_element_selected("sensor")
    assert w.inspector.position_spin.value() == pytest.approx(380.0 / MM_PER_INCH, abs=1e-3)


def test_the_grading_selector_switches_tolerance_policy(workspace, app):
    from microscopevuilder.rules.tolerances import TolerancePolicy, tolerance_policy

    w = workspace(11)
    w._on_element_selected("sensor")
    w._set_element_position("sensor", 375.0)
    assert w.report_panel.headline.text() == "ROUND PASSED"

    w.tolerance_box.setCurrentIndex(1)  # strict
    try:
        assert tolerance_policy() is TolerancePolicy.STRICT
        assert "failing" in w.report_panel.headline.text()
    finally:
        w.tolerance_box.setCurrentIndex(0)
    assert tolerance_policy() is TolerancePolicy.FORGIVING


# --- multi-pane imaging, auto-run and the toolbar signal path ----------------


def test_toolbar_add_card_lands_midway_not_at_the_origin(workspace, app):
    # QAction.triggered carries a `checked` bool. Connected naively it arrives as
    # the slot's first argument, so every card from the toolbar used to be built
    # at s = False == 0.0. Drive the ACTION, not the method, or the bug hides.
    w = workspace(1)
    action = next(a for a in w.findChildren(QtGui.QAction) if a.text() == "Add card")
    action.trigger()

    cards = w.bench.cards()
    assert len(cards) == 1
    assert cards[0].s == pytest.approx(w.bench.extent() * 0.5)
    assert cards[0].s > 0.0


def test_each_card_gets_its_own_numbered_pane(workspace, app):
    w = workspace(1)
    w.add_card(120.0)
    w.add_card(60.0)

    planes = w.scope_planes()
    w.scope.set_planes(planes)

    # The detector, plus one pane per card, numbered along the axis.
    assert len(w.scope.panes) == 3
    cards = [p for p in planes if p.is_card]
    assert [p.ordinal for p in cards] == [1, 2]
    assert [p.s for p in cards] == [60.0, 120.0]
    labels = " ".join(w.scope.pane_title(p.name) for p in planes)
    assert "Card 1" in labels and "Card 2" in labels


def test_scene_numbers_cards_to_match_the_panes(workspace, app):
    w = workspace(1)
    w.add_card(120.0)
    w.add_card(60.0)
    ordinals = {
        name: w.scene.card_ordinal(name) for name in (c.name for c in w.bench.cards())
    }
    by_s = {c.name: c.s for c in w.bench.cards()}
    assert ordinals[min(by_s, key=by_s.get)] == 1
    assert ordinals[max(by_s, key=by_s.get)] == 2


def test_a_second_run_with_nothing_changed_does_no_work(workspace, app):
    w = workspace(11)
    assert w.run_test() is True
    assert w.worker.wait(60_000)
    app.processEvents()

    # Everything is cached and nothing moved, so there is nothing to recompute.
    assert w.run_test() is False
    assert "nothing" in w.statusBar().currentMessage().lower()


def test_moving_a_card_re_images_only_that_pane_and_skips_the_ray_trace(workspace, app):
    from microscopevuilder.ui.render_jobs import plan_render

    w = workspace(11)
    w.add_card(120.0)
    assert w.run_test() is True
    assert w.worker.wait(60_000)
    app.processEvents()

    card = w.bench.cards()[0]
    w.bench.move(card.name, 130.0)
    plan, _ = plan_render(
        w.bench, w._detector_name(), w.current_settings(), w._image_cache, w._last_optics
    )
    # The optics did not move, so the paraxial solution still stands.
    assert plan.retrace is False
    assert [p.name for p in plan.stale] == [card.name]


def test_changing_the_specimen_invalidates_every_pane(workspace, app):
    from microscopevuilder.ui.render_jobs import plan_render

    w = workspace(11)
    w.add_card(120.0)
    assert w.run_test() is True
    assert w.worker.wait(60_000)
    app.processEvents()

    w.specimen.setCurrentText("beads")
    plan, _ = plan_render(
        w.bench, w._detector_name(), w.current_settings(), w._image_cache, w._last_optics
    )
    assert set(plan.stale) == set(plan.planes)


def test_auto_run_is_off_until_asked_and_then_fires_on_a_change(workspace, app):
    w = workspace(1)
    assert w.auto_run.isChecked() is False

    # Off: a change must not queue any work.
    w.request_auto_run()
    assert w._auto_timer.isActive() is False

    w.auto_run.setChecked(True)
    w.refresh_live()
    assert w._auto_timer.isActive(), "a bench change should queue an auto re-image"


def test_the_progress_bar_appears_only_while_imaging(workspace, app):
    w = workspace(1)
    assert w.scope.progress.isVisibleTo(w.scope) is False
    w.scope.start_progress(3)
    assert w.scope.progress.isVisibleTo(w.scope) is True
    assert w.scope.progress.maximum() == 3
    w.scope.set_progress(2, 3, "card 1")
    assert w.scope.progress.value() == 2
    w.scope.end_progress("done")
    assert w.scope.progress.isVisibleTo(w.scope) is False


def test_a_pane_that_fails_to_render_says_so_instead_of_going_blank(workspace, app):
    w = workspace(1)
    w.scope.set_planes(w.scope_planes())
    name = w._detector_name()
    w._on_plane(name, RuntimeError("no image plane here"))
    assert "no image plane here" in w.scope.panes[name].text()


def test_changing_an_imaging_setting_queues_an_auto_re_image(workspace, app):
    w = workspace(1)
    w.auto_run.setChecked(True)
    w._auto_timer.stop()
    w.specimen.setCurrentText("beads")
    assert w._auto_timer.isActive()


def test_closing_the_window_disarms_a_pending_auto_run(workspace, app):
    # A debounce left armed past close would fire into a half-torn-down window
    # and start a render against it.
    w = workspace(1)
    w.auto_run.setChecked(True)
    w.refresh_live()
    assert w._auto_timer.isActive()

    w.close()
    assert w._auto_timer.isActive() is False
    w.request_auto_run()
    assert w._auto_timer.isActive() is False, "a closed workspace must not re-arm"
