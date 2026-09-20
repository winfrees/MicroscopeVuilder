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
from .geometry import arm_axis_frame, axis_frame, point_at, project
from .ruler import choose_step, snap, snap_to_planes, ticks, to_display
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
COLOR_RULER = QtGui.QColor("#7a828e")
COLOR_SNAP = QtGui.QColor("#4fbf78")
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
    "beamsplitter": COLOR_ILLUMINATION,
    "dichroic": COLOR_ILLUMINATION,
    "lamp": COLOR_ILLUMINATION,
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
        painter.rotate(self._scene.glyph_rotation(element))
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
            # Constrain the drag to the axis, then snap: only s may change.
            snapped, captured = self._scene.snap_position(self._scene.s_from_scene_x(value.x()))
            self._scene.report_snap(self.name, snapped, captured)
            return QtCore.QPointF(snapped, self._scene.y_for_element(self.name))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._scene.commit_element_move(self.name, self.pos().x())
        self._scene.report_snap(None, 0.0, False)


class BenchScene(QtWidgets.QGraphicsScene):
    """Draws the axis, the components, the ray overlay and the conjugate ribbon."""

    benchChanged = QtCore.Signal()
    elementSelected = QtCore.Signal(str)
    snapReadout = QtCore.Signal(str)

    RIBBON_OFFSET_MM = 14.0
    RIBBON_ROW_MM = 4.0
    RULER_OFFSET_MM = -13.0

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
        self.show_ruler = True
        self.snap_step_mm = 0.1
        self.snap_to_planes = True
        self.snapping_enabled = True
        self.unit = "mm"
        self.view_scale = 1.0
        self._snap_readout = ""
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
            item.setPos(self.scene_pos_for(element.s, element.arm))
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
        if self.show_ruler:
            self._draw_ruler()
        if self.show_rays:
            self._draw_rays()
        if self.show_ribbon:
            self._draw_ribbon()
        for name, item in self._element_items.items():
            element = self.bench.get(name)
            item.setPos(self.scene_pos_for(element.s, element.arm))
            item.update()

    # --- coordinate helpers --------------------------------------------------

    def scene_pos_for(self, s: float, arm: str = "main") -> QtCore.QPointF:
        p = (
            project(self.bench.position_of(s))
            if arm == "main"
            else project(self.bench.position_of_arm(arm, s))
        )
        return QtCore.QPointF(p[0], -p[1])

    def s_from_scene_x(self, x: float) -> float:
        return float(x)

    def snap_position(self, s: float) -> tuple[float, bool]:
        """Apply the active snapping to a dragged position.

        Plane snapping is tried first and wins: getting a sensor onto the plane
        where the image actually forms is the point of the round, while landing on
        a round number is only a convenience.
        """
        if not self.snapping_enabled:
            return s, False
        if self.snap_to_planes:
            # Capture within a few pixels on screen, so the pull feels the same at
            # any zoom rather than growing into a shove when zoomed out.
            capture = 6.0 / max(self.view_scale, 1e-6)
            snapped, captured = snap_to_planes(s, self.significant_planes(), capture)
            if captured:
                return snapped, True
        return snap(s, self.snap_step_mm), False

    def significant_planes(self) -> list[float]:
        """Positions worth snapping to: image and pupil planes, and focal planes."""
        planes: list[float] = []
        if self.model is not None:
            planes += [c.s for c in self.model.conjugates]
            if self.model.image_s is not None:
                planes.append(self.model.image_s)
        for element in self.bench.elements if self.bench else []:
            if element.focal_length_mm:
                planes.append(element.s + element.focal_length_mm)
                planes.append(element.s - element.focal_length_mm)
        return [p for p in planes if p == p]  # drop any NaN

    def report_snap(self, name: str | None, s: float, captured: bool) -> None:
        text = ""
        if name is not None:
            text = f"{name} at {to_display(s, self.unit)}"
            if captured:
                text += "  -- snapped to an optical plane"
            elif self.snapping_enabled:
                text += f"  (grid {to_display(self.snap_step_mm, self.unit)})"
        if text != self._snap_readout:
            self._snap_readout = text
            self.snapReadout.emit(text)

    def set_view_scale(self, scale: float) -> None:
        """Told by the view how zoomed in we are, so ruler density can follow."""
        if abs(scale - self.view_scale) > 1e-9:
            self.view_scale = scale
            self.refresh()

    def glyph_rotation(self, element) -> float:
        """Degrees to rotate a component glyph so it lies across its own axis."""
        import math

        if element.arm == "main":
            _, tangent, _ = axis_frame(self.bench, element.s)
        else:
            _, tangent, _ = arm_axis_frame(self.bench, element.arm, element.s)
        return math.degrees(math.atan2(-tangent[1], tangent[0]))

    def y_for_element(self, name: str) -> float:
        element = self.bench.get(name)
        return self.scene_pos_for(element.s, element.arm).y()

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

        # Each side arm gets its own axis, drawn in to the junction. This is what
        # makes an epi stand read as a branch rather than a confusing overlay.
        for arm in self.bench.arms.values():
            self._polyline(
                [self.scene_pos_for(0.0, arm.name), self.scene_pos_for(arm.length_mm, arm.name)],
                COLOR_ILLUMINATION, 0.3, dashed=True,
            )

    def _ray_points(self, samples, arm: str = "main"):
        pts = []
        for s, height in samples:
            if arm == "main":
                p = point_at(self.bench, s, height)
            else:
                here, _, normal = arm_axis_frame(self.bench, arm, s)
                p = here + normal * height
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
            arm = getattr(ray, "arm", "main")
            pts = self._ray_points(ray.samples, arm)
            self._polyline(pts, color, 0.7 if illumination else 1.0)
            if illumination:
                self._polyline(
                    self._ray_points([(s, -y) for s, y in ray.samples], arm), color, 0.7
                )
            # The marginal ray is symmetric about the axis; draw its mirror so the
            # cone reads as a cone.
            if ray.label == "marginal":
                mirrored = [(s, -y) for s, y in ray.samples]
                self._polyline(self._ray_points(mirrored), color, 1.0)

    def _draw_ruler(self) -> None:
        """A ruler under the bench, with tick density following the zoom.

        Without this the player has no way to see where a component is, let alone
        put it somewhere specific.
        """
        start = min(self.s_object, 0.0)
        end = self.bench.extent() * 1.05 + 5.0
        if end <= start:
            return

        # Tick spacing is chosen from what is actually visible, so zooming in
        # reveals finer divisions instead of the same coarse ones stretched out.
        visible_span = (end - start) / max(self.view_scale, 1e-6)
        step = choose_step(visible_span)
        y = self.RULER_OFFSET_MM

        self._polyline(
            [QtCore.QPointF(start, -y), QtCore.QPointF(end, -y)], COLOR_RULER, 0.4
        )
        for tick in ticks(start, end, step, self.unit):
            height = 2.2 if tick.is_major else 1.1
            line = QtWidgets.QGraphicsLineItem(
                tick.position_mm, -y, tick.position_mm, -y + height
            )
            pen = QtGui.QPen(COLOR_RULER, 1.2 if tick.is_major else 0.8)
            pen.setCosmetic(True)
            line.setPen(pen)
            self._add(line)

            if tick.label:
                text = QtWidgets.QGraphicsSimpleTextItem(tick.label)
                text.setBrush(QtGui.QBrush(COLOR_RULER))
                text.setPos(tick.position_mm + 0.3, -y + height)
                text.setScale(0.08)
                self._add(text)

    def _draw_ribbon(self) -> None:
        """Two rows of ticks: the field set and the aperture set.

        The prototype the plan calls for. Köhler alignment is the puzzle of getting
        the ticks in a row to line up, so they have to be visible before any round
        depends on them.
        """
        rows = {"field": (self.RIBBON_OFFSET_MM, COLOR_FIELD_SET, "field / image planes"),
                "aperture": (self.RIBBON_OFFSET_MM + self.RIBBON_ROW_MM, COLOR_APERTURE_SET,
                             "aperture / pupil planes")}
        for kind, (offset, color, row_label) in rows.items():
            planes = [c for c in self.model.conjugates if c.kind == kind]
            if not planes:
                continue
            xs = [self.scene_pos_for(c.s).x() for c in planes]
            y = offset
            self._polyline(
                [QtCore.QPointF(min(xs) - 5, y), QtCore.QPointF(max(xs) + 5, y)],
                color, 0.3, dashed=True,
            )
            caption = QtWidgets.QGraphicsSimpleTextItem(row_label)
            caption.setBrush(QtGui.QBrush(color))
            caption.setPos(min(xs) - 5, y - 2.6)
            caption.setScale(0.09)
            caption.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, False)
            self._add(caption)

            for c, x in zip(planes, xs):
                tick = QtWidgets.QGraphicsLineItem(x, y - 1.6, x, y + 1.6)
                pen = QtGui.QPen(color, 1.4)
                pen.setCosmetic(True)
                tick.setPen(pen)
                tick.setToolTip(f"{c.label} ({kind} conjugate, s = {c.s:.2f} mm)")
                self._add(tick)

                # Name each plane. The ribbon's whole job is to make conjugacy
                # legible, and an unlabelled tick asks the player to remember what
                # is at 120 mm instead of showing them.
                text = QtWidgets.QGraphicsSimpleTextItem(c.label)
                text.setBrush(QtGui.QBrush(color))
                text.setPos(x + 0.4, y + 0.4)
                text.setScale(0.08)
                self._add(text)

    # --- selection -----------------------------------------------------------

    def _on_selection(self) -> None:
        items = [i for i in self.selectedItems() if isinstance(i, ElementItem)]
        if items:
            self.elementSelected.emit(items[0].name)
