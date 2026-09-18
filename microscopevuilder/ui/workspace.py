"""The workspace window: bench view, inspector, live rule report, testing loop.

Two clocks, deliberately separated (see the plan's performance note):

* The **live** clock runs the paraxial pass and the rule report on every drag.
  Both are microseconds-to-milliseconds, so the ray overlay and the scorecard
  track the mouse.
* The **Run** clock does partially coherent image synthesis, which is ~0.3 s at
  384^2. It runs on a worker thread so the workspace never freezes.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..bench.bench import Bench
from ..bench.probe import find_planes, read_card
from ..game.rounds import Round, get_round
from ..imaging import metrics
from ..game.progress import Progress
from ..game.rounds import PREREQUISITES
from ..game.scoring import check_parts_budget, diff_against_reference, score_round
from ..imaging.specimens import SPECIMEN_LIBRARY
from ..imaging.synthesis import RenderSpec, render_build
from ..rules.base import RuleReport, Status

STATUS_COLORS = {
    Status.PASS: "#4fbf78",
    Status.WARN: "#d9a441",
    Status.FAIL: "#d95f5f",
    Status.NOT_APPLICABLE: "#7a828e",
}


SPECIMENS = SPECIMEN_LIBRARY


class ImageWorker(QtCore.QThread):
    """Off-thread synthesis of the image THIS bench forms.

    Before M10 this took four loose scalars and rendered a generic picture. It now
    renders through the resolved build: the objective's catalog aberrations at the
    chosen field height and wavelength, defocus computed from where the image
    actually lands, and photometry that makes a dim build noisy rather than merely
    darker.
    """

    finished_image = QtCore.Signal(object)

    def __init__(self, bench, s_object: float, spec: RenderSpec, round_=None, parent=None):
        # Parented to the workspace so Qt owns the thread's lifetime. Destroying a
        # running QThread aborts the process, and a render outliving its window is
        # not hypothetical -- it is what happens when a player closes the workspace
        # while an image is being formed.
        super().__init__(parent)
        self.bench = bench
        self.s_object = s_object
        self.spec = spec
        self.round = round_

    def run(self) -> None:
        rendered = render_build(self.bench, self.s_object, self.spec)
        measured = self.round.measure_image(self.bench) if self.round else None
        self.finished_image.emit((rendered, measured))


class CardPanel(QtWidgets.QWidget):
    """What the white card shows, and where the planes actually are.

    The point of the card is that the player finds the planes by looking, so this
    panel reports the reading in the same words a demonstrator would use, and lists
    the image and pupil planes it has found rather than making the player hunt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.reading = QtWidgets.QLabel("place a card on the axis (Add card)")
        self.reading.setWordWrap(True)
        self.reading.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.reading)

        self.detail = QtWidgets.QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #9aa3b0;")
        layout.addWidget(self.detail)

        layout.addWidget(QtWidgets.QLabel("planes found on this bench:"))
        self.planes = QtWidgets.QTreeWidget()
        self.planes.setHeaderLabels(["plane", "s (mm)"])
        layout.addWidget(self.planes, 1)

    def show_reading(self, bench: Bench, s_object: float) -> None:
        cards = bench.cards()
        if not cards:
            self.reading.setText("place a card on the axis (Add card)")
            self.detail.setText("")
        else:
            card = cards[-1]
            if card.s < s_object:
                self.reading.setText(
                    f"the card at s = {card.s:.1f} mm is upstream of the specimen "
                    f"(s = {s_object:.1f} mm) -- move it downstream to read it"
                )
                self.detail.setText("")
                self._list_planes(bench, s_object)
                return
            r = read_card(bench, s_object, card.s)
            self.reading.setText(r.describe())
            self.detail.setText(
                f"marginal height {r.marginal_height_mm:.3f} mm   |   "
                f"chief height {r.chief_height_mm:.3f} mm\n"
                "the marginal ray crossing the axis marks an image plane; "
                "the chief ray crossing marks a pupil plane"
            )

        self._list_planes(bench, s_object)

    def _list_planes(self, bench: Bench, s_object: float) -> None:
        self.planes.clear()
        found = find_planes(bench, s_object, s_object, bench.extent() * 1.1 + 10.0)
        for kind, label in (("image", "image (field)"), ("pupil", "pupil (aperture)")):
            for s in found[kind]:
                self.planes.addTopLevelItem(QtWidgets.QTreeWidgetItem([label, f"{s:.2f}"]))


class InspectorPanel(QtWidgets.QWidget):
    """Properties of the selected element."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)
        self.title = QtWidgets.QLabel("nothing selected")
        self.title.setStyleSheet("font-weight: 600;")
        layout.addRow(self.title)
        self.fields: dict[str, QtWidgets.QLabel] = {}
        for key in ("kind", "position", "focal length", "semi-diameter", "power", "catalog"):
            label = QtWidgets.QLabel("--")
            self.fields[key] = label
            layout.addRow(key, label)

    def show_element(self, bench: Bench, name: str) -> None:
        e = bench.get(name)
        self.title.setText(e.label or e.name)
        f = e.focal_length_mm
        self.fields["kind"].setText(e.kind)
        self.fields["position"].setText(f"s = {e.s:.2f} mm")
        self.fields["focal length"].setText("--" if f is None else f"{f:.3g} mm")
        self.fields["semi-diameter"].setText(f"{e.semi_diameter_mm:.3g} mm")
        self.fields["power"].setText("--" if not f else f"{1000.0 / f:.3g} dioptres")
        self.fields["catalog"].setText(e.catalog_key or "--")


class ReportPanel(QtWidgets.QWidget):
    """The live scorecard. Every row shows the equation with the player's numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.headline = QtWidgets.QLabel("--")
        self.headline.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(self.headline)
        self.score_line = QtWidgets.QLabel("")
        self.score_line.setWordWrap(True)
        self.score_line.setStyleSheet("color: #9aa3b0;")
        layout.addWidget(self.score_line)
        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderLabels(["check", "result"])
        self.list.setRootIsDecorated(True)
        self.list.setColumnWidth(0, 150)
        layout.addWidget(self.list)

    def show_score(self, score) -> None:
        stars = "\u2605" * score.stars + "\u2606" * (3 - score.stars)
        detail = "; ".join(score.reasons) if score.reasons else "clean solve"
        self.score_line.setText(
            f"{stars}   {score.parts_used}/{score.parts_budget} parts   --   {detail}"
        )

    def show_report(self, report: RuleReport) -> None:
        self.list.clear()
        for r in report.results:
            node = QtWidgets.QTreeWidgetItem([r.name, r.summary])
            node.setForeground(0, QtGui.QColor(STATUS_COLORS[r.status]))
            if r.equation:
                node.addChild(QtWidgets.QTreeWidgetItem(["", r.equation]))
            if not r.ok and r.remedy:
                remedy = QtWidgets.QTreeWidgetItem(["", f"remedy: {r.remedy}"])
                remedy.setForeground(1, QtGui.QColor(STATUS_COLORS[Status.WARN]))
                node.addChild(remedy)
            self.list.addTopLevelItem(node)
            node.setExpanded(not r.ok)
        failures = len(report.failures())
        self.headline.setText("ROUND PASSED" if report.passed else f"{failures} failing check(s)")
        self.headline.setStyleSheet(
            "font-weight: 600; font-size: 14px; color: "
            + (STATUS_COLORS[Status.PASS] if report.passed else STATUS_COLORS[Status.FAIL])
        )


class MetricsPanel(QtWidgets.QWidget):
    """What was measured off the rendered image.

    The plan's pillar 2 says a build wins because the image it forms satisfies the
    spec. That requires the image to be measured, not admired, so these numbers are
    read off the picture the player just generated.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel("press Run to measure the image")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.summary)

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["measurement", "value"])
        self.table.setColumnWidth(0, 210)
        layout.addWidget(self.table, 1)

        self.provenance = QtWidgets.QLabel("")
        self.provenance.setWordWrap(True)
        self.provenance.setStyleSheet("color: #9aa3b0; font-size: 11px;")
        layout.addWidget(self.provenance)

    @staticmethod
    def _snr(rendered) -> float:
        from ..optics.psf import mtf_cutoff_cycles_per_um

        cutoff = mtf_cutoff_cycles_per_um(
            rendered.optics.wavelength_um, max(rendered.optics.na, 1e-3)
        )
        return metrics.signal_to_noise(
            rendered.intensity, sample_um=rendered.sample_um, cutoff_cycles_per_um=cutoff
        )

    def show_measurements(self, rendered, measured=None) -> None:
        optics = rendered.optics
        image = rendered.intensity
        self.summary.setText(optics.describe())

        rows = [
            ("Michelson contrast", f"{metrics.michelson_contrast(image):.3f}"),
            ("field uniformity (corner/centre)", f"{metrics.field_uniformity(image):.3f}"),
            ("signal to noise", f"{self._snr(rendered):.1f}"),
            ("Strehl ratio", f"{optics.wavefront.strehl(optics.wavelength_um):.3f}"),
            ("total wavefront error", f"{optics.wavefront.rms_um * 1000:.0f} nm RMS"),
            ("relative irradiance (NA^2/M^2)", f"{optics.relative_irradiance:.3e}"),
            ("defocus at the detector", f"{optics.image_defocus_mm:+.4f} mm"),
            ("sampled field", f"{rendered.extent_um:.1f} um at {rendered.sample_um:.4f} um/px"),
        ]
        dominant = optics.dominant_aberration
        if dominant:
            rows.insert(4, ("dominant aberration", f"{dominant[0]} ({dominant[1] * 1000:+.0f} nm)"))

        self.table.clear()

        # Measured rules first: they are the round's actual verdict on the image,
        # and the raw numbers below are the evidence for it.
        if measured is not None and measured.results:
            for result in measured.results:
                node = QtWidgets.QTreeWidgetItem([result.name, result.summary])
                node.setForeground(0, QtGui.QColor(STATUS_COLORS[result.status]))
                if result.equation:
                    node.addChild(QtWidgets.QTreeWidgetItem(["", result.equation]))
                if not result.ok and result.remedy:
                    node.addChild(QtWidgets.QTreeWidgetItem(["", f"remedy: {result.remedy}"]))
                self.table.addTopLevelItem(node)
                node.setExpanded(not result.ok)

        for label, value in rows:
            self.table.addTopLevelItem(QtWidgets.QTreeWidgetItem([label, value]))

        source = {
            "ideal": "aberrations: none — no catalog match, treated as a perfect lens",
        }.get(
            optics.aberration_source,
            f"aberrations: {optics.aberration_source} (pedagogical budget, not vendor data)",
        )
        self.provenance.setText("\n".join([source, *rendered.notes]))


class ScopeView(QtWidgets.QLabel):
    """The generated image: what the build actually produces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("press Run to image the specimen")
        self.setStyleSheet("background: #0b0d10; color: #7a828e;")

    def show_image(self, image: np.ndarray) -> None:
        arr = np.asarray(image, dtype=float)
        span = arr.max() - arr.min()
        norm = (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)
        buf = np.ascontiguousarray((norm * 255).astype(np.uint8))
        h, w = buf.shape
        qimage = QtGui.QImage(buf.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        self.setPixmap(QtGui.QPixmap.fromImage(qimage))


class Workspace(QtWidgets.QMainWindow):
    def __init__(self, round_number: int = 2, parent=None):
        super().__init__(parent)
        from .scene import BenchScene  # imported late so geometry stays Qt-free

        self.round: Round = get_round(round_number)
        self.bench: Bench = self.round.reference_build()
        self.worker: ImageWorker | None = None
        self.progress = Progress.load()
        self.last_measured = None

        self.setWindowTitle(f"MicroscopeVuilder -- round {self.round.number}: {self.round.title}")
        self.scene = BenchScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)

        self.inspector = InspectorPanel()
        self.report_panel = ReportPanel()
        self.scope = ScopeView()
        self.metrics_panel = MetricsPanel()

        self.card_panel = CardPanel()

        right = QtWidgets.QTabWidget()
        right.addTab(self.report_panel, "scorecard")
        right.addTab(self.card_panel, "white card")
        right.addTab(self.inspector, "inspector")
        right.addTab(self.scope, "image")
        right.addTab(self.metrics_panel, "measurements")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        brief = QtWidgets.QLabel(f"<b>{self.round.title}</b><br>{self.round.brief}")
        brief.setWordWrap(True)
        left_layout.addWidget(brief)
        left_layout.addWidget(self.view, 1)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([760, 420])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self.scene.set_bench(self.bench, self.round.s_object)
        self.scene.benchChanged.connect(self.refresh_live)
        self.scene.elementSelected.connect(self._on_element_selected)
        self.refresh_live()
        self.resize(1200, 720)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("main")
        bar.setMovable(False)

        run = QtGui.QAction("Run", self)
        run.setShortcut("Ctrl+R")
        run.triggered.connect(self.run_test)
        bar.addAction(run)

        add_card = QtGui.QAction("Add card", self)
        add_card.setShortcut("Ctrl+K")
        add_card.setToolTip("drop a white card on the axis to see what lands there")
        add_card.triggered.connect(self.add_card)
        bar.addAction(add_card)

        clear_cards = QtGui.QAction("Clear cards", self)
        clear_cards.triggered.connect(self.clear_cards)
        bar.addAction(clear_cards)

        self.diff_action = QtGui.QAction("Compare with a working build", self)
        self.diff_action.setEnabled(False)
        self.diff_action.triggered.connect(self.show_diff)
        bar.addAction(self.diff_action)

        reset = QtGui.QAction("Reference build", self)
        reset.triggered.connect(self.load_reference)
        bar.addAction(reset)

        bar.addSeparator()
        self.rays_toggle = QtGui.QAction("Rays", self, checkable=True, checked=True)
        self.rays_toggle.toggled.connect(self._toggle_rays)
        bar.addAction(self.rays_toggle)

        self.ribbon_toggle = QtGui.QAction("Conjugates", self, checkable=True, checked=True)
        self.ribbon_toggle.toggled.connect(self._toggle_ribbon)
        bar.addAction(self.ribbon_toggle)

        self.imaging_toggle = QtGui.QAction("Imaging path", self, checkable=True, checked=True)
        self.imaging_toggle.toggled.connect(self._toggle_imaging)
        bar.addAction(self.imaging_toggle)

        self.illumination_toggle = QtGui.QAction("Illumination path", self, checkable=True, checked=True)
        self.illumination_toggle.toggled.connect(self._toggle_illumination)
        bar.addAction(self.illumination_toggle)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" condenser NA/obj NA (S): "))
        self.coherence = QtWidgets.QDoubleSpinBox()
        self.coherence.setRange(0.0, 1.5)
        self.coherence.setSingleStep(0.1)
        self.coherence.setValue(0.8)
        bar.addWidget(self.coherence)

        self.specimen = QtWidgets.QComboBox()
        self.specimen.addItems(list(SPECIMENS))
        bar.addWidget(self.specimen)

        bar.addWidget(QtWidgets.QLabel("  field: "))
        self.field_height = QtWidgets.QDoubleSpinBox()
        self.field_height.setRange(0.0, 1.0)
        self.field_height.setSingleStep(0.25)
        self.field_height.setToolTip("0 = on axis, 1 = corner of the field")
        bar.addWidget(self.field_height)

        bar.addWidget(QtWidgets.QLabel("  lambda (nm): "))
        self.wavelength = QtWidgets.QDoubleSpinBox()
        self.wavelength.setRange(380.0, 750.0)
        self.wavelength.setSingleStep(10.0)
        self.wavelength.setValue(546.1)
        bar.addWidget(self.wavelength)

        self.statusBar().showMessage("drag a component along the axis to rebuild")

    # --- live loop -----------------------------------------------------------

    def refresh_live(self) -> None:
        """Paraxial pass plus rule report. Budgeted for every drag."""
        report = self.round.grade(self.bench)
        report.add(check_parts_budget(self.bench, self.round.parts_budget))
        self.report_panel.show_report(report)
        self._update_score(report)
        self.card_panel.show_reading(self.bench, self.round.s_object)
        self.scene.refresh()
        model = self.scene.model
        bits = []
        if model.image_s is not None:
            bits.append(f"image at s = {model.image_s:.2f} mm")
        if model.magnification is not None:
            bits.append(f"M = {abs(model.magnification):.2f}x")
        else:
            # Collimated output is the correct answer for a visual scope or an
            # infinity space, so say which rather than reporting nothing.
            bits.append("output collimated (afocal) -- correct for a relaxed eye or an infinity space")
        if model.aperture_stop:
            bits.append(f"stop: {model.aperture_stop}")
        self.statusBar().showMessage("   |   ".join(bits))

    def _update_score(self, report) -> None:
        """Rate the build and remember it. Progress should survive closing the app."""
        score = score_round(self.bench, self.round.parts_budget, report, self.last_measured)
        self.report_panel.show_score(score)
        self.diff_action.setEnabled(not score.passed)
        if score.passed:
            self.progress.complete(self.round.number, score.stars, score.parts_used)
            try:
                self.progress.save()
            except OSError:
                # A read-only or full disk must not take the game down with it.
                self.statusBar().showMessage("could not save progress (read-only location?)")

    def show_diff(self) -> None:
        """Offer the structural diff -- what differs, not what to do about it."""
        diff = diff_against_reference(self.bench, self.round.reference_build())
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Compared with a working build")
        box.setText("How your build differs from one that solves this round:")
        box.setDetailedText(diff.format())
        box.setInformativeText(
            "This says what is different, not what to do about it -- working out "
            "why the difference matters is the round."
        )
        box.exec()

    def _toggle_rays(self, on: bool) -> None:
        self.scene.show_rays = on
        self.scene.refresh()

    def _toggle_ribbon(self, on: bool) -> None:
        self.scene.show_ribbon = on
        self.scene.refresh()

    def _toggle_imaging(self, on: bool) -> None:
        self.scene.show_imaging = on
        self.scene.refresh()

    def _toggle_illumination(self, on: bool) -> None:
        self.scene.show_illumination = on
        self.scene.refresh()

    def _on_element_selected(self, name: str) -> None:
        self.inspector.show_element(self.bench, name)

    def add_card(self, s: float | None = None) -> None:
        """Drop a card midway along the bench, or at a given position."""
        if s is None:
            s = self.bench.extent() * 0.5
        self.bench.add_card(s)
        self.scene.set_bench(self.bench, self.round.s_object)
        self.refresh_live()

    def clear_cards(self) -> None:
        for card in self.bench.cards():
            self.bench.remove(card.name)
        self.scene.set_bench(self.bench, self.round.s_object)
        self.refresh_live()

    def load_reference(self) -> None:
        self.bench = self.round.reference_build()
        self.scene.set_bench(self.bench, self.round.s_object)
        self.refresh_live()

    # --- testing loop --------------------------------------------------------

    def run_test(self) -> None:
        """Synthesize the image this build actually produces, off the UI thread."""
        if self.worker is not None and self.worker.isRunning():
            return

        self.statusBar().showMessage("imaging...")
        spec = RenderSpec(
            specimen=SPECIMENS[self.specimen.currentText()],
            n=256,
            field_height=self.field_height.value(),
            wavelength_um=self.wavelength.value() / 1000.0,
            coherence_parameter=self.coherence.value(),
            detector_name=self._detector_name(),
            exposure_photons=2000.0,
        )
        self.worker = ImageWorker(self.bench, self.round.s_object, spec, self.round, self)
        self.worker.finished_image.connect(self._on_image)
        self.worker.start()

    def closeEvent(self, event) -> None:
        """Wait for any render in flight before the window goes away.

        A QThread whose owning widget has been destroyed is a crash: the worker
        finishes, emits into a deleted receiver, and takes the process with it.
        Closing the workspace mid-render is an ordinary thing for a player to do.
        """
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5_000)
        super().closeEvent(event)

    def _detector_name(self) -> str:
        for candidate in ("sensor", "screen", "intermediate_image"):
            if self.bench.has(candidate):
                return candidate
        return self.bench.elements[-1].name

    def _on_image(self, result) -> None:
        rendered, measured = result
        self.last_measured = measured
        self.refresh_live()
        self.scope.show_image(rendered.intensity)
        self.metrics_panel.show_measurements(rendered, measured)
        message = f"image formed at {rendered.optics.describe()}"
        if measured is not None and measured.results:
            failures = len(measured.failures())
            message += (
                "   |   measured rules: all pass" if not failures
                else f"   |   {failures} measured check(s) failing"
            )
        self.statusBar().showMessage(message)


def launch(round_number: int = 2) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Workspace(round_number)
    window.show()
    return app.exec()
