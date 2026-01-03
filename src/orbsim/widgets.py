from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from orbsim.chem.elements import ATOMIC_NUMBER_TO_SYMBOL, ELEMENTS_H_KR

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
        self._theme_tokens: dict | None = None
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

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            "QTableWidget {"
            f"background: {colors['surface']};"
            f"color: {colors['text']};"
            f"gridline-color: {colors['border']};"
            "}"
            "QTableWidget::item:selected {"
            f"background: {colors['accent']};"
            f"color: {colors['bg']};"
            "}"
        )

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
        self._theme_tokens: dict | None = None
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
        self._apply_colorbar_style()
        self._apply_bounds()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter, 1)
        layout.addWidget(self.colorbar, 0)


class ElementTileWidget(QtWidgets.QFrame):
    def __init__(self, element: Element, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.element = element
        self._theme_tokens: dict | None = None
        self.setObjectName("inventoryElementTile")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._drag_start_pos: QtCore.QPoint | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.symbol_label = QtWidgets.QLabel(element.symbol)
        self.symbol_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.number_label = QtWidgets.QLabel(str(element.atomic_number))
        self.number_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.symbol_label)
        layout.addWidget(self.number_label)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        self.setStyleSheet(
            "QFrame#inventoryElementTile {"
            f"background: {colors['surfaceAlt']};"
            f"color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QFrame#inventoryElementTile:focus {"
            f"outline: 2px solid {colors['focusRing']};"
            "}"
        )
        font = self.symbol_label.font()
        font.setBold(True)
        self.symbol_label.setFont(font)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.buttons() & QtCore.Qt.MouseButton.LeftButton
            and self._drag_start_pos is not None
            and (event.pos() - self._drag_start_pos).manhattanLength()
            > QtWidgets.QApplication.startDragDistance()
        ):
            self._start_drag()
            self._drag_start_pos = None
        super().mouseMoveEvent(event)

    def _start_drag(self) -> None:
        payload = json.dumps({"Z": self.element.atomic_number, "symbol": self.element.symbol}).encode("utf-8")
        mime = QtCore.QMimeData()
        mime.setData("application/x-orbsim-element", payload)
        mime.setText(self.element.symbol)
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.exec(QtCore.Qt.DropAction.CopyAction)


class InventoryPeriodicTableWidget(QtWidgets.QScrollArea):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self._content = QtWidgets.QWidget()
        self._layout = QtWidgets.QGridLayout(self._content)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._tiles: list[ElementTileWidget] = []
        self._populate()
        self.setWidget(self._content)

    def _populate(self) -> None:
        columns = 6
        for index, element in enumerate(ELEMENTS_H_KR):
            tile = ElementTileWidget(Element(element.symbol, element.name, element.atomic_number), self._content)
            row = index // columns
            col = index % columns
            self._layout.addWidget(tile, row, col)
            self._tiles.append(tile)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            "QScrollArea {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            "}"
        )
        for tile in self._tiles:
            tile.apply_theme(tokens)


class CraftingSlotWidget(QtWidgets.QFrame):
    element_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self.setAcceptDrops(True)
        self.setObjectName("craftingSlot")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.label = QtWidgets.QLabel("")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self._element: Element | None = None

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        self.setStyleSheet(
            "QFrame#craftingSlot {"
            f"background: {colors['surfaceAlt']};"
            f"color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QFrame#craftingSlot:focus {"
            f"outline: 2px solid {colors['focusRing']};"
            "}"
        )
        font = self.label.font()
        font.setBold(True)
        self.label.setFont(font)

    def element(self) -> Element | None:
        return self._element

    def set_element(self, element: Element | None) -> None:
        self._element = element
        if element:
            self.label.setText(element.symbol)
        else:
            self.label.setText("")
        self.element_changed.emit()

    def clear_element(self) -> None:
        self.set_element(None)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-orbsim-element"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        element = _element_from_mime(event.mimeData())
        if element:
            self.set_element(element)
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.clear_element()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.clear_element()
        else:
            super().keyPressEvent(event)


class CraftingGridWidget(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self.setAcceptDrops(True)
        self.setObjectName("craftingGrid")
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)
        self._slots: list[CraftingSlotWidget] = []
        for row in range(3):
            for col in range(3):
                slot = CraftingSlotWidget(self)
                slot.element_changed.connect(self.changed)
                layout.addWidget(slot, row, col)
                self._slots.append(slot)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            "QWidget#craftingGrid {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            "}"
        )
        for slot in self._slots:
            slot.apply_theme(tokens)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-orbsim-element"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        element = _element_from_mime(event.mimeData())
        if not element:
            event.ignore()
            return
        for slot in self._slots:
            if slot.element() is None:
                slot.set_element(element)
                event.acceptProposedAction()
                return
        if self._slots:
            self._slots[0].set_element(element)
        event.acceptProposedAction()

    def get_elements(self) -> list[int]:
        elements: list[int] = []
        for slot in self._slots:
            element = slot.element()
            if element:
                elements.append(element.atomic_number)
        return elements

    def set_elements(self, atomic_numbers: list[int]) -> None:
        for slot in self._slots:
            slot.clear_element()
        for element_number, slot in zip(atomic_numbers, self._slots, strict=False):
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(element_number)
            if not symbol:
                continue
            slot.set_element(Element(symbol, symbol, element_number))

    def clear(self) -> None:
        for slot in self._slots:
            slot.clear_element()


def _element_from_mime(mime: QtCore.QMimeData) -> Element | None:
    if not mime.hasFormat("application/x-orbsim-element"):
        return None
    try:
        payload = json.loads(bytes(mime.data("application/x-orbsim-element")).decode("utf-8"))
        symbol = payload.get("symbol")
        atomic_number = int(payload.get("Z"))
    except Exception:
        return None
    if not symbol:
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, "")
    if not symbol:
        return None
    return Element(symbol, symbol, atomic_number)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        self._apply_colorbar_style()
        colors = tokens["colors"]
        try:
            self.plotter.set_background(colors["surfaceAlt"])
        except Exception:
            pass
        self._apply_bounds()

    def _apply_colorbar_style(self) -> None:
        colors = (self._theme_tokens or {}).get("colors", {})
        bg = colors.get("surface", "#111827")
        text = colors.get("text", "#e5e7eb")
        border = colors.get("border", "#94a3b8")
        self.colorbar.setStyleSheet(
            f"background-color: {bg}; color: {text}; border: 1px solid {border}; padding: 4px;"
        )

    def _apply_bounds(self) -> None:
        colors = (self._theme_tokens or {}).get("colors", {})
        axis_color = colors.get("text", "#f8fafc")
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
                color=axis_color,
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
            colors = (self._theme_tokens or {}).get("colors", {})
            text_color = colors.get("text", "white")
            shadow = (self._theme_tokens or {}).get("meta", {}).get("mode") in ("dark", "high_contrast")
            self.plotter.add_text(
                hint,
                name="measure_hint",
                font_size=10,
                color=text_color,
                shadow=shadow,
            )
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
        colors = (self._theme_tokens or {}).get("colors", {})
        accent = colors.get("accent", "#93c5fd")
        text_color = colors.get("text", "white")
        p1, p2 = self._picked_points[:2]
        distance = float(np.linalg.norm(p2 - p1))
        line_actor = self.plotter.add_mesh(pv.Line(p1, p2), color=accent, line_width=4)
        mid = (p1 + p2) / 2
        label_actor = self.plotter.add_point_labels(
            [mid],
            [f"{distance:.2f} angstrom"],
            point_size=0,
            font_size=12,
            text_color=text_color,
        )
        self._measurement_actors.append(line_actor)
        self._measurement_labels.append(label_actor)
        self._picked_points.clear()

    def _render_angle(self) -> None:
        colors = (self._theme_tokens or {}).get("colors", {})
        accent = colors.get("accentHover", "#f97316")
        text_color = colors.get("text", "white")
        p1, p2, p3 = self._picked_points[:3]
        v1 = p1 - p2
        v2 = p3 - p2
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        angle_rad = np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        polyline = self.plotter.add_mesh(
            pv.Spline(np.vstack([p1, p2, p3]), 50),
            color=accent,
            line_width=4,
            opacity=0.85,
        )
        label_actor = self.plotter.add_point_labels(
            [p2],
            [f"{angle_deg:.1f} deg"],
            point_size=0,
            font_size=12,
            text_color=text_color,
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
        colors = (self._theme_tokens or {}).get("colors", {})
        text_color = colors.get("text", "white")
        shadow = (self._theme_tokens or {}).get("meta", {}).get("mode") in ("dark", "high_contrast")
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_message=False,
            use_mesh=True,
        )
        self.plotter.add_text(
            "Measure distance: pick two points",
            name="measure_hint",
            font_size=10,
            color=text_color,
            shadow=shadow,
        )

    def start_angle_measurement(self) -> None:
        self._clear_measurements(keep_mode=True)
        self._measurement_mode = "angle"
        colors = (self._theme_tokens or {}).get("colors", {})
        text_color = colors.get("text", "white")
        shadow = (self._theme_tokens or {}).get("meta", {}).get("mode") in ("dark", "high_contrast")
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_message=False,
            use_mesh=True,
        )
        self.plotter.add_text(
            "Measure angle: pick three points",
            name="measure_hint",
            font_size=10,
            color=text_color,
            shadow=shadow,
        )

    def stop_measurements(self) -> None:
        self._clear_measurements()

    def update_colorbar(self, cmap: str, clim: tuple[float, float], label: str) -> None:
        """Render a small standalone colorbar next to the plotter."""
        colors = (self._theme_tokens or {}).get("colors", {})
        text = colors.get("text", "#e5e7eb")
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
        painter.setPen(QtGui.QPen(QtGui.QColor(text)))
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
        self._apply_colorbar_style()

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
