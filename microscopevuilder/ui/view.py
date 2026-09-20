"""The bench view: a QGraphicsView that zooms.

Zoom is not a convenience here, it is what makes the game playable. Fitted to a
760-pixel window, round 11's 380 mm bench gives 0.50 mm per pixel against a depth of
focus of 0.19 mm, so one pixel of drag overshoots the tolerance by two and a half
times. At 10x zoom the same pixel is 0.05 mm and the placement becomes ordinary.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

MIN_SCALE = 0.05
MAX_SCALE = 400.0
WHEEL_STEP = 1.15


class BenchView(QtWidgets.QGraphicsView):
    """A graphics view with wheel zoom, keyboard zoom and fit-to-bench."""

    scaleChanged = QtCore.Signal(float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        # Zoom toward the pointer: the thing under the cursor is what the player is
        # aiming at, so it should stay put.
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._scale = 1.0

    # --- zoom ---------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        return self._scale

    def zoom_by(self, factor: float) -> float:
        """Multiply the zoom, clamped. Returns the resulting scale."""
        target = self._scale * factor
        clamped = max(MIN_SCALE, min(MAX_SCALE, target))
        applied = clamped / self._scale
        if abs(applied - 1.0) < 1e-12:
            return self._scale
        self.scale(applied, applied)
        self._scale = clamped
        self._announce()
        return self._scale

    def zoom_in(self) -> float:
        return self.zoom_by(WHEEL_STEP**2)

    def zoom_out(self) -> float:
        return self.zoom_by(1.0 / WHEEL_STEP**2)

    def fit_bench(self, margin_mm: float = 12.0) -> float:
        """Frame the whole bench."""
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return self._scale
        self.fitInView(
            rect.adjusted(-margin_mm, -margin_mm, margin_mm, margin_mm),
            QtCore.Qt.KeepAspectRatio,
        )
        self._scale = self.transform().m11()
        self._announce()
        return self._scale

    def zoom_to(self, scale: float) -> float:
        if scale <= 0:
            return self._scale
        return self.zoom_by(scale / self._scale)

    def _announce(self) -> None:
        self.scaleChanged.emit(self._scale)
        scene = self.scene()
        if hasattr(scene, "set_view_scale"):
            scene.set_view_scale(self._scale)

    # --- input --------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        """Wheel zooms. Shift+wheel scrolls, for anyone who expects that instead."""
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(WHEEL_STEP if delta > 0 else 1.0 / WHEEL_STEP)
        event.accept()

    def keyPressEvent(self, event) -> None:
        # Alt suspends snapping for as long as it is held, which is the usual way
        # out of a snap that is fighting you.
        if event.key() == QtCore.Qt.Key_Alt:
            self._set_snapping(False)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Alt:
            self._set_snapping(True)
        super().keyReleaseEvent(event)

    def _set_snapping(self, enabled: bool) -> None:
        scene = self.scene()
        if hasattr(scene, "snapping_enabled"):
            scene.snapping_enabled = enabled
