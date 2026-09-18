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
from ..imaging.specimens import amplitude_bars, phase_disc
from ..optics.coherence import image_partially_coherent
from ..optics.psf import make_pupil_grid
from ..rules.base import RuleReport, Status

STATUS_COLORS = {
    Status.PASS: "#4fbf78",
    Status.WARN: "#d9a441",
    Status.FAIL: "#d95f5f",
    Status.NOT_APPLICABLE: "#7a828e",
}


class ImageWorker(QtCore.QThread):
    """Off-thread image synthesis, so a 0.3 s trace never blocks a drag."""

    finished_image = QtCore.Signal(object)

    def __init__(self, na: float, wavelength_um: float, coherence: float, specimen_kind: str):
        super().__init__()
        self.na = na
        self.wavelength_um = wavelength_um
        self.coherence = coherence
        self.specimen_kind = specimen_kind

    def run(self) -> None:
        n = 256
        grid = make_pupil_grid(max(self.na, 0.05), self.wavelength_um, n=n)
        dx = grid.image_sample_um
        if self.specimen_kind == "phase":
            specimen = phase_disc(n, dx, radius_um=n * dx / 6, phase_rad=0.6)
        else:
            specimen = amplitude_bars(n, dx, period_um=max(dx * 8, 1.0))
        self.finished_image.emit(image_partially_coherent(specimen, grid, self.coherence))


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
        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderLabels(["check", "result"])
        self.list.setRootIsDecorated(True)
        self.list.setColumnWidth(0, 150)
        layout.addWidget(self.list)

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

        self.setWindowTitle(f"MicroscopeVuilder -- round {self.round.number}: {self.round.title}")
        self.scene = BenchScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)

        self.inspector = InspectorPanel()
        self.report_panel = ReportPanel()
        self.scope = ScopeView()

        self.card_panel = CardPanel()

        right = QtWidgets.QTabWidget()
        right.addTab(self.report_panel, "scorecard")
        right.addTab(self.card_panel, "white card")
        right.addTab(self.inspector, "inspector")
        right.addTab(self.scope, "image")

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
        self.specimen.addItems(["amplitude", "phase"])
        bar.addWidget(self.specimen)

        self.statusBar().showMessage("drag a component along the axis to rebuild")

    # --- live loop -----------------------------------------------------------

    def refresh_live(self) -> None:
        """Paraxial pass plus rule report. Budgeted for every drag."""
        report = self.round.grade(self.bench)
        self.report_panel.show_report(report)
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
        na = 0.0
        for element in self.bench.elements:
            na = max(na, float(element.metadata.get("na", 0.0)))
        if na <= 0:
            system = self.bench.to_paraxial()
            na = system.object_space_na(self.round.s_object) or 0.1

        self.statusBar().showMessage("imaging...")
        self.worker = ImageWorker(
            na, self.round.wavelength_um, self.coherence.value(), self.specimen.currentText()
        )
        self.worker.finished_image.connect(self._on_image)
        self.worker.start()

    def _on_image(self, image) -> None:
        self.scope.show_image(image)
        self.statusBar().showMessage("image ready")


def launch(round_number: int = 2) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Workspace(round_number)
    window.show()
    return app.exec()
