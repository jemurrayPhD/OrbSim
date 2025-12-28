from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    atomic_number: int


PERIODIC_TABLE = [
    Element("H", "Hydrogen", 1),
    Element("He", "Helium", 2),
    Element("Li", "Lithium", 3),
    Element("Be", "Beryllium", 4),
    Element("B", "Boron", 5),
    Element("C", "Carbon", 6),
    Element("N", "Nitrogen", 7),
    Element("O", "Oxygen", 8),
    Element("F", "Fluorine", 9),
    Element("Ne", "Neon", 10),
    Element("Na", "Sodium", 11),
    Element("Mg", "Magnesium", 12),
    Element("Al", "Aluminum", 13),
    Element("Si", "Silicon", 14),
    Element("P", "Phosphorus", 15),
    Element("S", "Sulfur", 16),
    Element("Cl", "Chlorine", 17),
    Element("Ar", "Argon", 18),
]

_PERIODIC_POSITIONS = {
    "H": (0, 0),
    "He": (0, 17),
    "Li": (1, 0),
    "Be": (1, 1),
    "B": (1, 12),
    "C": (1, 13),
    "N": (1, 14),
    "O": (1, 15),
    "F": (1, 16),
    "Ne": (1, 17),
    "Na": (2, 0),
    "Mg": (2, 1),
    "Al": (2, 12),
    "Si": (2, 13),
    "P": (2, 14),
    "S": (2, 15),
    "Cl": (2, 16),
    "Ar": (2, 17),
}


class PeriodicTableWidget(QtWidgets.QTableWidget):
    element_dragged = QtCore.Signal(Element)
    element_selected = QtCore.Signal(Element)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRowCount(4)
        self.setColumnCount(18)
        self.setShowGrid(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setDragEnabled(True)
        self._configure_headers()
        self._populate()
        self.cellClicked.connect(self._on_cell_clicked)

    def _configure_headers(self) -> None:
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        for col in range(self.columnCount()):
            self.setColumnWidth(col, 38)
        for row in range(self.rowCount()):
            self.setRowHeight(row, 38)

    def _populate(self) -> None:
        for element in PERIODIC_TABLE:
            position = _PERIODIC_POSITIONS.get(element.symbol)
            if position is None:
                continue
            row, col = position
            item = QtWidgets.QTableWidgetItem(f"{element.symbol}\n{element.atomic_number}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, element)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            self.setItem(row, col, item)

    def _element_from_item(self, item: QtWidgets.QTableWidgetItem | None) -> Element | None:
        if not item:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        item = self.item(row, col)
        element = self._element_from_item(item)
        if element:
            self.element_selected.emit(element)

    def startDrag(self, supported_actions: QtCore.Qt.DropActions) -> None:
        item = self.currentItem()
        element = self._element_from_item(item)
        if not element:
            return
        mime = QtCore.QMimeData()
        mime.setText(element.symbol)
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.exec(supported_actions)
        self.element_dragged.emit(element)


class DropPlotter(QtWidgets.QFrame):
    element_dropped = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Sunken)
        self._measurement_mode: str | None = None
        self._picked_points: list[np.ndarray] = []
        self._measurement_actors: list[int] = []
        self._measurement_labels: list[int] = []
        self._plotter_closed = False
        self.show_bounds_axes: bool = True
        self._bounds_actor = None
        self.plotter = QtInteractor(self)
        self.colorbar = QtWidgets.QLabel()
        self.colorbar.setMinimumWidth(48)
        self.colorbar.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.colorbar.setStyleSheet(
            "background-color: #111827; color: #e5e7eb; border: 1px solid #94a3b8; padding: 4px;"
        )
        self._apply_bounds()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter, 1)
        layout.addWidget(self.colorbar, 0)

    def _apply_bounds(self) -> None:
        if not self.show_bounds_axes:
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass
            self._bounds_actor = None
            return
        try:
            self._bounds_actor = self.plotter.show_bounds(
                grid="front",
                ticks="both",
                location="outer",
                xtitle="x (angstrom)",
                ytitle="y (angstrom)",
                ztitle="z (angstrom)",
                color="#f8fafc",
                bold=True,
                font_size=12,
                show_xlabels=True,
                show_ylabels=True,
                show_zlabels=True,
                minor_ticks=True,
            )
            if self._bounds_actor and hasattr(self._bounds_actor, "GetCubeProperty"):
                try:
                    self._bounds_actor.GetCubeProperty().SetLineWidth(2.0)
                except Exception:
                    pass
        except Exception:
            pass

    def set_show_bounds(self, enabled: bool) -> None:
        self.show_bounds_axes = bool(enabled)
        self._apply_bounds()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasText():
            symbol = event.mimeData().text().strip()
            if symbol:
                self.element_dropped.emit(symbol)
            event.acceptProposedAction()

    def reset_scene(self) -> None:
        """Clear the scene but keep measurement tools configured."""
        self.plotter.clear()
        self._clear_measurements(keep_mode=True)
        self._apply_bounds()
        if self._measurement_mode:
            self.plotter.enable_point_picking(
                callback=self._on_pick,
                left_clicking=True,
                show_message=False,
                use_mesh=True,
            )
            hint = (
                "Measure distance: pick two points"
                if self._measurement_mode == "distance"
                else "Measure angle: pick three points"
            )
            self.plotter.add_text(hint, name="measure_hint", font_size=10)
        else:
            self.plotter.disable_picking()

    def _on_pick(self, point: tuple[float, float, float]) -> None:
        if self._measurement_mode is None:
            return
        self._picked_points.append(np.array(point))
        if self._measurement_mode == "distance" and len(self._picked_points) >= 2:
            self._render_distance()
        elif self._measurement_mode == "angle" and len(self._picked_points) >= 3:
            self._render_angle()

    def _render_distance(self) -> None:
        p1, p2 = self._picked_points[:2]
        distance = float(np.linalg.norm(p2 - p1))
        line_actor = self.plotter.add_mesh(pv.Line(p1, p2), color="#93c5fd", line_width=4)
        mid = (p1 + p2) / 2
        label_actor = self.plotter.add_point_labels(
            [mid],
            [f"{distance:.2f} angstrom"],
            point_size=0,
            font_size=12,
            text_color="white",
        )
        self._measurement_actors.append(line_actor)
        self._measurement_labels.append(label_actor)
        self._picked_points.clear()

    def _render_angle(self) -> None:
        p1, p2, p3 = self._picked_points[:3]
        v1 = p1 - p2
        v2 = p3 - p2
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        angle_rad = np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        polyline = self.plotter.add_mesh(
            pv.Spline(np.vstack([p1, p2, p3]), 50),
            color="#f97316",
            line_width=4,
            opacity=0.85,
        )
        label_actor = self.plotter.add_point_labels(
            [p2],
            [f"{angle_deg:.1f} deg"],
            point_size=0,
            font_size=12,
            text_color="white",
        )
        self._measurement_actors.append(polyline)
        self._measurement_labels.append(label_actor)
        self._picked_points.clear()

    def _clear_measurements(self, keep_mode: bool = False) -> None:
        for actor in self._measurement_actors:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
        for label in self._measurement_labels:
            self.plotter.remove_actor(label, reset_camera=False, render=False)
        self.plotter.remove_actor("measure_hint", reset_camera=False, render=False)
        self._measurement_actors.clear()
        self._measurement_labels.clear()
        self._picked_points.clear()
        if not keep_mode:
            self._measurement_mode = None
            self.plotter.disable_picking()

    def start_distance_measurement(self) -> None:
        self._clear_measurements(keep_mode=True)
        self._measurement_mode = "distance"
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_message=False,
            use_mesh=True,
        )
        self.plotter.add_text("Measure distance: pick two points", name="measure_hint", font_size=10)

    def start_angle_measurement(self) -> None:
        self._clear_measurements(keep_mode=True)
        self._measurement_mode = "angle"
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_message=False,
            use_mesh=True,
        )
        self.plotter.add_text("Measure angle: pick three points", name="measure_hint", font_size=10)

    def stop_measurements(self) -> None:
        self._clear_measurements()

    def update_colorbar(self, cmap: str, clim: tuple[float, float], label: str) -> None:
        """Render a small standalone colorbar next to the plotter."""
        try:
            from matplotlib import cm
        except Exception:
            self.colorbar.setText(f"{label}\n{clim[0]:.2g}–{clim[1]:.2g}")
            return
        height = 160
        width = 26
        gradient = np.linspace(0.0, 1.0, height)
        rgba = cm.get_cmap(cmap)(gradient)
        rgb = (rgba[:, :3] * 255).astype(np.uint8)
        bar = np.repeat(rgb[:, None, :], max(width // 2, 6), axis=1)
        image = QtGui.QImage(bar.data, bar.shape[1], bar.shape[0], QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        vmin, vmax = clim
        ticks = [
            (0.0, vmax),
            (0.5, (vmin + vmax) / 2),
            (1.0, vmin),
        ]
        for frac, value in ticks:
            y = int((1 - frac) * (height - 1))
            painter.drawLine(bar.shape[1], y, bar.shape[1] + 6, y)
            painter.drawText(bar.shape[1] + 8, y + 4, f"{value:.2g}")
        painter.drawText(2, 12, label)
        painter.end()
        self.colorbar.setPixmap(pixmap)
        self.colorbar.setMinimumWidth(width + 30)
        self.colorbar.setToolTip(f"{label}: {clim[0]:.3g} to {clim[1]:.3g}")
        self.colorbar.setStyleSheet(
            "background-color: #111827; color: #e5e7eb; border: 1px solid #94a3b8; padding: 4px;"
        )

    def cleanup(self) -> None:
        """Close the underlying VTK render window before Qt tears down."""
        if self._plotter_closed:
            return
        self.plotter.close()
        self._plotter_closed = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)


class CollapsibleGroup(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._update_visibility)
        self._update_visibility(self.isChecked())

    def _update_visibility(self, checked: bool) -> None:
        if self.layout():
            self.layout().setEnabled(checked)
        for child in self.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindDirectChildrenOnly):
            child.setVisible(checked)
        self.setMaximumHeight(16777215 if checked else 28)
