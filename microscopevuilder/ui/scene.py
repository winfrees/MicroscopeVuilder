"""The optical bench as a Qt scene graph.

A ``QGraphicsScene`` is the reason PySide6 was chosen over pygame: elements are
real items with drag, snap and hit-testing, and the ray overlay is redrawn from
the paraxial pass on every move.

Scene units are millimetres, with y flipped so that +height draws upward. All
pixel scaling is left to the view's transform, so nothing here has to know about
zoom.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..bench.bench import Bench
from .geometry import axis_frame, point_at, project
from .trace_model import TraceModel, build_trace_model

# Marginal and chief rays are the classic teaching pair, so they get the two
# strongest colors; everything structural stays muted.
COLOR_AXIS = QtGui.QColor("#5a6472")
COLOR_MARGINAL = QtGui.QColor("#e8833a")
COLOR_CHIEF = QtGui.QColor("#3a8fe8")
COLOR_LENS = QtGui.QColor("#7fc7d9")
COLOR_STOP = QtGui.QColor("#c05a5a")
COLOR_DETECTOR = QtGui.QColor("#9b8ac4")
COLOR_CARD = QtGui.QColor("#eceff4")
COLOR_ILLUMINATION = QtGui.QColor("#d9c45a")
COLOR_FIELD_SET = QtGui.QColor("#3a8fe8")
COLOR_APERTURE_SET = QtGui.QColor("#e8833a")

KIND_COLORS = {
    "lens": COLOR_LENS,
    "objective": COLOR_LENS,
    "eyepiece": COLOR_LENS,
    "condenser": COLOR_LENS,
    "tube_lens": COLOR_LENS,
    "diaphragm": COLOR_STOP,
    "aperture_stop": COLOR_STOP,
    "field_stop": COLOR_STOP,
    "detector": COLOR_DETECTOR,
    "white_card": COLOR_CARD,
}


class ElementItem(QtWidgets.QGraphicsItem):
    """One component, drawn across the axis and draggable along it."""

    def __init__(self, scene: "BenchScene", name: str):
        super().__init__()
        self._scene = scene
        self.name = name
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(QtCore.Qt.SizeHorCursor)
        self.setZValue(10)

    @property
    def element(self):
        return self._scene.bench.get(self.name)

    def boundingRect(self) -> QtCore.QRectF:
        half = self.element.semi_diameter_mm
        return QtCore.QRectF(-4, -half - 4, 8, 2 * half + 8)

    def paint(self, painter, option, widget=None) -> None:
        element = self.element
        half = element.semi_diameter_mm
        color = KIND_COLORS.get(element.kind, COLOR_AXIS)
        pen = QtGui.QPen(color, 1.2)
        if self.isSelected():
            pen = QtGui.QPen(QtGui.QColor("#f2f2f2"), 2.0)
        painter.setPen(pen)

        if element.focal_length_mm is not None:
            # A lens: a lens-shaped outline, with arrowheads showing its sign.
            path = QtGui.QPainterPath()
            bulge = 2.0 if element.focal_length_mm > 0 else -2.0
            path.moveTo(0, -half)
            path.quadTo(bulge, 0, 0, half)
            path.quadTo(-bulge, 0, 0, -half)
            painter.drawPath(path)
        elif element.kind == "white_card":
            # A card is a solid surface across the beam: draw it filled, because
            # that is what it does -- it intercepts.
            painter.setBrush(QtGui.QBrush(COLOR_CARD))
            painter.setOpacity(0.85)
            painter.drawRect(QtCore.QRectF(-0.8, -half, 1.6, 2 * half))
        else:
            # A stop: two jaws leaving the clear aperture open between them.
            jaw = max(half * 0.45, 1.0)
            painter.drawLine(QtCore.QPointF(0, -half), QtCore.QPointF(0, -half - jaw))
            painter.drawLine(QtCore.QPointF(0, half), QtCore.QPointF(0, half + jaw))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self._scene.bench:
            # Constrain the drag to the axis: only s may change.
            new_s = self._scene.s_from_scene_x(value.x())
            return QtCore.QPointF(value.x(), self._scene.y_for_element(self.name))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._scene.commit_element_move(self.name, self.pos().x())


class BenchScene(QtWidgets.QGraphicsScene):
    """Draws the axis, the components, the ray overlay and the conjugate ribbon."""

    benchChanged = QtCore.Signal()
    elementSelected = QtCore.Signal(str)

    RIBBON_OFFSET_MM = 14.0
    RIBBON_ROW_MM = 4.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bench: Bench | None = None
        self.s_object = 0.0
        self.field_height_mm = 0.5
        self.model: TraceModel | None = None
        self._element_items: dict[str, ElementItem] = {}
        self._overlay: list[QtWidgets.QGraphicsItem] = []
        self.show_rays = True
        self.show_ribbon = True
        self.show_imaging = True
        self.show_illumination = True
        self.setBackgroundBrush(QtGui.QColor("#12151a"))
        self.selectionChanged.connect(self._on_selection)

    # --- state ---------------------------------------------------------------

    def set_bench(self, bench: Bench, s_object: float, field_height_mm: float = 0.5) -> None:
        self.bench = bench
        self.s_object = s_object
        self.field_height_mm = field_height_mm
        self.rebuild()

    def rebuild(self) -> None:
        self.clear()
        self._element_items.clear()
        self._overlay.clear()
        if self.bench is None:
            return
        for element in self.bench.elements:
            item = ElementItem(self, element.name)
            item.setPos(self.scene_pos_for(element.s))
            self.addItem(item)
            self._element_items[element.name] = item
        self.refresh()

    def refresh(self) -> None:
        """Re-run the live pass and redraw the overlay. Must stay cheap."""
        if self.bench is None:
            return
        self.model = build_trace_model(self.bench, self.s_object, self.field_height_mm)
        for item in self._overlay:
            if item.scene() is self:
                self.removeItem(item)
        self._overlay.clear()
        self._draw_axis()
        if self.show_rays:
            self._draw_rays()
        if self.show_ribbon:
            self._draw_ribbon()
        for name, item in self._element_items.items():
            item.setPos(self.scene_pos_for(self.bench.get(name).s))
            item.update()

    # --- coordinate helpers --------------------------------------------------

    def scene_pos_for(self, s: float) -> QtCore.QPointF:
        p = project(self.bench.position_of(s))
        return QtCore.QPointF(p[0], -p[1])

    def s_from_scene_x(self, x: float) -> float:
        return float(x)

    def y_for_element(self, name: str) -> float:
        return self.scene_pos_for(self.bench.get(name).s).y()

    def commit_element_move(self, name: str, scene_x: float) -> None:
        self.bench.move(name, max(0.0, self.s_from_scene_x(scene_x)))
        self.refresh()
        self.benchChanged.emit()

    # --- drawing -------------------------------------------------------------

    def _add(self, item: QtWidgets.QGraphicsItem) -> None:
        self.addItem(item)
        self._overlay.append(item)

    def _polyline(self, points, color: QtGui.QColor, width: float = 0.4, dashed=False):
        path = QtGui.QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
        pen = QtGui.QPen(color, width)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        item = QtWidgets.QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(1)
        self._add(item)
        return item

    def _draw_axis(self) -> None:
        start = min(self.s_object, 0.0)
        end = self.bench.extent() * 1.1 + 10.0
        stops = sorted({start, end, *(f.s for f in self.bench.folds)})
        pts = [self.scene_pos_for(s) for s in stops]
        self._polyline(pts, COLOR_AXIS, 0.3, dashed=True)

    def _ray_points(self, samples):
        pts = []
        for s, height in samples:
            p = point_at(self.bench, s, height)
            pts.append(QtCore.QPointF(p[0], -p[1]))
        return pts

    def _draw_rays(self) -> None:
        colors = {
            "marginal": COLOR_MARGINAL,
            "chief": COLOR_CHIEF,
            "illumination_axial": COLOR_ILLUMINATION,
            "illumination_edge": COLOR_ILLUMINATION,
        }
        for ray in self.model.rays:
            illumination = ray.label.startswith("illumination")
            if illumination and not self.show_illumination:
                continue
            if not illumination and not self.show_imaging:
                continue
            color = colors.get(ray.label, COLOR_AXIS)
            pts = self._ray_points(ray.samples)
            self._polyline(pts, color, 0.7 if illumination else 1.0)
            if illumination:
                self._polyline(
                    self._ray_points([(s, -y) for s, y in ray.samples]), color, 0.7
                )
            # The marginal ray is symmetric about the axis; draw its mirror so the
            # cone reads as a cone.
            if ray.label == "marginal":
                mirrored = [(s, -y) for s, y in ray.samples]
                self._polyline(self._ray_points(mirrored), color, 1.0)

    def _draw_ribbon(self) -> None:
        """Two rows of ticks: the field set and the aperture set.

        The prototype the plan calls for. Köhler alignment is the puzzle of getting
        the ticks in a row to line up, so they have to be visible before any round
        depends on them.
        """
        rows = {"field": (self.RIBBON_OFFSET_MM, COLOR_FIELD_SET),
                "aperture": (self.RIBBON_OFFSET_MM + self.RIBBON_ROW_MM, COLOR_APERTURE_SET)}
        for kind, (offset, color) in rows.items():
            planes = [c for c in self.model.conjugates if c.kind == kind]
            if not planes:
                continue
            xs = [self.scene_pos_for(c.s).x() for c in planes]
            y = offset
            self._polyline(
                [QtCore.QPointF(min(xs) - 5, y), QtCore.QPointF(max(xs) + 5, y)],
                color, 0.3, dashed=True,
            )
            for c, x in zip(planes, xs):
                tick = QtWidgets.QGraphicsLineItem(x, y - 1.6, x, y + 1.6)
                pen = QtGui.QPen(color, 1.4)
                pen.setCosmetic(True)
                tick.setPen(pen)
                tick.setToolTip(f"{c.label} ({kind} conjugate, s = {c.s:.2f} mm)")
                self._add(tick)

    # --- selection -----------------------------------------------------------

    def _on_selection(self) -> None:
        items = [i for i in self.selectedItems() if isinstance(i, ElementItem)]
        if items:
            self.elementSelected.emit(items[0].name)
