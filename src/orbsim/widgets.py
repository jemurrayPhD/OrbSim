from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets
from pyvistaqt import QtInteractor

from orbsim.chem.elements import (
    ATOMIC_NUMBER_TO_SYMBOL,
    ELEMENTS_H_KR,
    electronegativity,
    is_metal,
    is_nonmetal,
)

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
                callback=self._on_point_picked,
                show_message=False,
                left_clicking=True,
                use_mesh=True,
                show_point=False,
                tolerance=0.025,
            )

    def set_measurement_mode(self, mode: str | None) -> None:
        self._measurement_mode = mode
        self._clear_measurements()
        if mode:
            self.plotter.enable_point_picking(
                callback=self._on_point_picked,
                show_message=True,
                left_clicking=True,
                use_mesh=True,
                show_point=True,
                tolerance=0.02,
            )
        else:
            self.plotter.disable_picking()

    def _on_point_picked(self, point: np.ndarray) -> None:
        if self._measurement_mode is None:
            return
        self._picked_points.append(point)
        if self._measurement_mode == "distance" and len(self._picked_points) == 2:
            self._measure_distance()
        elif self._measurement_mode == "angle" and len(self._picked_points) == 3:
            self._measure_angle()
        elif self._measurement_mode == "dihedral" and len(self._picked_points) == 4:
            self._measure_dihedral()

    def _clear_measurements(self, keep_mode: bool = False) -> None:
        for actor in self._measurement_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        for label in self._measurement_labels:
            try:
                self.plotter.remove_actor(label)
            except Exception:
                pass
        self._measurement_actors = []
        self._measurement_labels = []
        self._picked_points = []
        if not keep_mode:
            self._measurement_mode = None

    def _measure_distance(self) -> None:
        p1, p2 = self._picked_points
        dist = np.linalg.norm(p1 - p2)
        line = pv.Line(p1, p2)
        actor = self.plotter.add_mesh(line, color="#f97316", line_width=4)
        self._measurement_actors.append(actor)
        midpoint = (p1 + p2) / 2
        label = self.plotter.add_point_labels(
            [midpoint],
            [f"{dist:.2f} Å"],
            point_color="#f97316",
            shape=None,
            font_size=14,
            text_color="#f97316",
        )
        self._measurement_labels.append(label)
        self._picked_points = []

    def _measure_angle(self) -> None:
        p1, p2, p3 = self._picked_points
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        arc = pv.CircularArc(p1, p3, p2, angle=angle)
        actor = self.plotter.add_mesh(arc, color="#3b82f6", line_width=4)
        self._measurement_actors.append(actor)
        label = self.plotter.add_point_labels(
            [p2],
            [f"{angle:.1f}°"],
            point_color="#3b82f6",
            shape=None,
            font_size=14,
            text_color="#3b82f6",
        )
        self._measurement_labels.append(label)
        self._picked_points = []

    def _measure_dihedral(self) -> None:
        p1, p2, p3, p4 = self._picked_points
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
        label = self.plotter.add_point_labels(
            [p2],
            [f"{angle:.1f}°"],
            point_color="#14b8a6",
            shape=None,
            font_size=14,
            text_color="#14b8a6",
        )
        self._measurement_labels.append(label)
        self._picked_points = []

    def update_colorbar(self, label: str, clim: tuple[float, float], cmap: str) -> None:
        colors = (self._theme_tokens or {}).get("colors", {})
        text = colors.get("text", "#f8fafc")
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


class ElementTileWidget(QtWidgets.QFrame):
    element_clicked = QtCore.Signal(int)

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
        self._dragging = False

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)
        self.symbol_label = QtWidgets.QLabel(element.symbol)
        self.symbol_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.number_label = QtWidgets.QLabel(str(element.atomic_number))
        self.number_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.symbol_label, 0, 0, 1, 2)
        layout.addWidget(self.number_label, 1, 0, 1, 2)

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
        font.setPointSize(font.pointSize() + 4)
        self.symbol_label.setFont(font)
        number_font = self.number_label.font()
        number_font.setPointSize(max(number_font.pointSize() - 2, 7))
        number_font.setBold(False)
        self.number_label.setFont(number_font)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.buttons() & QtCore.Qt.MouseButton.LeftButton
            and self._drag_start_pos is not None
            and (event.pos() - self._drag_start_pos).manhattanLength()
            > QtWidgets.QApplication.startDragDistance()
        ):
            self._dragging = True
            self._start_drag()
            self._drag_start_pos = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and not self._dragging:
            self.element_clicked.emit(self.element.atomic_number)
        self._dragging = False
        super().mouseReleaseEvent(event)

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




class CraftingTableSlotWidget(QtWidgets.QFrame):
    slot_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("craftingTableSlot")
        self._theme_tokens: dict | None = None
        self.atomic_number: int | None = None
        self.count = 0

        self.symbol_label = QtWidgets.QLabel("")
        self.symbol_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.symbol_label.setObjectName("craftingSlotSymbol")
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)
        self.count_label.setObjectName("craftingSlotCount")

        self.add_button = QtWidgets.QToolButton()
        self.add_button.setText("+")
        self.sub_button = QtWidgets.QToolButton()
        self.sub_button.setText("−")
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("🗑")
        self.add_button.clicked.connect(self._increment)
        self.sub_button.clicked.connect(self._decrement)
        self.clear_button.clicked.connect(self.clear_slot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.symbol_label, 1)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.add_button)
        controls.addWidget(self.sub_button)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        self.setStyleSheet(
            "QFrame#craftingTableSlot {"
            f"background: {colors['surfaceAlt']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QFrame#craftingTableSlot:focus {"
            f"outline: 2px solid {colors['focusRing']};"
            "}"
            "QToolButton {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QLabel#craftingSlotCount {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            f"padding: 1px 4px;"
            "}"
        )
        font = self.symbol_label.font()
        font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        self.symbol_label.setFont(font)
        number_font = self.number_label.font()
        number_font.setPointSize(max(number_font.pointSize() - 2, 7))
        number_font.setBold(False)
        self.number_label.setFont(number_font)

    def set_element(self, atomic_number: int, symbol: str, count: int = 1) -> None:
        self.atomic_number = atomic_number
        self.symbol_label.setText(symbol)
        self.count = max(count, 1)
        self._update_count()
        self.slot_changed.emit()

    def clear_slot(self) -> None:
        self.atomic_number = None
        self.symbol_label.setText("")
        self.count = 0
        self._update_count()
        self.slot_changed.emit()

    def _increment(self) -> None:
        if self.atomic_number is None:
            return
        self.count += 1
        self._update_count()
        self.slot_changed.emit()

    def _decrement(self) -> None:
        if self.atomic_number is None:
            return
        self.count -= 1
        if self.count <= 0:
            self.clear_slot()
            return
        self._update_count()
        self.slot_changed.emit()

    def _update_count(self) -> None:
        self.count_label.setText(f"×{self.count}" if self.count > 0 else "")
        self.count_label.adjustSize()
        self.count_label.move(self.width() - self.count_label.width() - 6, self.height() - self.count_label.height() - 6)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_count()

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
        if self.atomic_number is None:
            self.set_element(element.atomic_number, element.symbol, 1)
        elif self.atomic_number == element.atomic_number:
            self._increment()
        else:
            self.set_element(element.atomic_number, element.symbol, 1)
        event.acceptProposedAction()


class CraftingTableWidget(QtWidgets.QWidget):
    crafting_changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self.setObjectName("craftingTable")
        self.setAcceptDrops(True)
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)
        self._slots: list[CraftingTableSlotWidget] = []
        for row in range(3):
            for col in range(3):
                slot = CraftingTableSlotWidget(self)
                slot.slot_changed.connect(self._emit_change)
                layout.addWidget(slot, row, col)
                self._slots.append(slot)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            "QWidget#craftingTable {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            "}"
        )
        for slot in self._slots:
            slot.apply_theme(tokens)

    def add_element_to_first_empty(self, atomic_number: int, symbol: str) -> None:
        for slot in self._slots:
            if slot.atomic_number is None:
                slot.set_element(atomic_number, symbol, 1)
                return
        QtWidgets.QMessageBox.information(self, "Crafting full", "No empty slots available in the 3×3 grid.")

    def set_counts(self, counts: dict[int, int]) -> None:
        for slot in self._slots:
            slot.clear_slot()
        for idx, (atomic_number, count) in enumerate(counts.items()):
            if idx >= len(self._slots):
                break
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, str(atomic_number))
            self._slots[idx].set_element(atomic_number, symbol, count)
        self._emit_change()

    def counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for slot in self._slots:
            if slot.atomic_number is None:
                continue
            counts[slot.atomic_number] = counts.get(slot.atomic_number, 0) + slot.count
        return counts

    def slots_state(self) -> list[dict]:
        state = []
        for slot in self._slots:
            if slot.atomic_number is None:
                state.append({"atomic_number": None, "count": 0})
            else:
                state.append({"atomic_number": slot.atomic_number, "count": slot.count})
        return state

    def clear(self) -> None:
        for slot in self._slots:
            slot.clear_slot()
        self._emit_change()

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
        self.add_element_to_first_empty(element.atomic_number, element.symbol)
        event.acceptProposedAction()

    def _emit_change(self) -> None:
        payload = {"counts": self.counts(), "slots": self.slots_state()}
        self.crafting_changed.emit(payload)


class AggregatedCraftingSlotWidget(QtWidgets.QFrame):
    add_one = QtCore.Signal(int)
    remove_one = QtCore.Signal(int)
    remove_all = QtCore.Signal(int)

    def __init__(self, atomic_number: int, symbol: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.atomic_number = atomic_number
        self.symbol = symbol
        self._count = 1
        self._theme_tokens: dict | None = None
        self.setObjectName("aggregatedCraftingSlot")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.number_label = QtWidgets.QLabel(str(atomic_number), self)
        self.number_label.setObjectName("craftingAtomicNumber")
        self.symbol_label = QtWidgets.QLabel(symbol, self)
        self.symbol_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.symbol_label.setObjectName("craftingSymbol")
        self.count_label = QtWidgets.QLabel("×1", self)
        self.count_label.setObjectName("craftingCount")
        self.count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.symbol_label, 1)

    def increment(self, amount: int = 1) -> None:
        self._count += amount
        self._update_count_label()

    def decrement(self, amount: int = 1) -> None:
        self._count = max(self._count - amount, 0)
        self._update_count_label()

    def count(self) -> int:
        return self._count

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        self.setStyleSheet(
            "QFrame#aggregatedCraftingSlot {"
            f"background: {colors['surfaceAlt']};"
            f"color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QFrame#aggregatedCraftingSlot:focus {"
            f"outline: 2px solid {colors['focusRing']};"
            "}"
            "QLabel#craftingCount {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            f"padding: 1px 4px;"
            "}"
        )
        symbol_font = self.symbol_label.font()
        symbol_font.setPointSize(symbol_font.pointSize() + 6)
        symbol_font.setBold(True)
        self.symbol_label.setFont(symbol_font)
        number_font = self.number_label.font()
        number_font.setPointSize(number_font.pointSize() - 1)
        self.number_label.setFont(number_font)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.number_label.move(6, 4)
        self.count_label.adjustSize()
        self.count_label.move(self.width() - self.count_label.width() - 6, self.height() - self.count_label.height() - 6)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.add_one.emit(self.atomic_number)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        remove_one = menu.addAction("Remove one")
        remove_all = menu.addAction("Remove all")
        action = menu.exec(event.globalPos())
        if action == remove_one:
            self.remove_one.emit(self.atomic_number)
        elif action == remove_all:
            self.remove_all.emit(self.atomic_number)

    def _update_count_label(self) -> None:
        self.count_label.setText(f"×{self._count}")
        self.count_label.adjustSize()
        self.update()


class AggregatedCraftingGridWidget(QtWidgets.QWidget):
    crafting_changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self.setAcceptDrops(True)
        self.setObjectName("aggregatedCraftingGrid")
        self._counts: dict[int, int] = {}
        self._slots: dict[int, AggregatedCraftingSlotWidget] = {}

        self._layout = QtWidgets.QGridLayout(self)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(4, 4, 4, 4)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            "QWidget#aggregatedCraftingGrid {"
            f"background: {colors['surface']};"
            f"border: 1px solid {colors['border']};"
            "}"
        )
        for slot in self._slots.values():
            slot.apply_theme(tokens)

    def counts(self) -> dict[int, int]:
        return dict(self._counts)

    def set_counts(self, counts: dict[int, int]) -> None:
        self.clear()
        for atomic_number, count in counts.items():
            for _ in range(count):
                self.add_element(atomic_number)

    def add_element(self, atomic_number: int) -> None:
        if atomic_number in self._counts:
            self._counts[atomic_number] += 1
            slot = self._slots[atomic_number]
            slot.increment()
            self._emit_change()
            return
        if len(self._counts) >= 9:
            QtWidgets.QMessageBox.information(self, "Crafting full", "Maximum of 9 distinct elements in the grid.")
            return
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, str(atomic_number))
        slot = AggregatedCraftingSlotWidget(atomic_number, symbol, self)
        slot.add_one.connect(self._handle_add_one)
        slot.remove_one.connect(self._handle_remove_one)
        slot.remove_all.connect(self._handle_remove_all)
        if self._theme_tokens:
            slot.apply_theme(self._theme_tokens)
        self._counts[atomic_number] = 1
        self._slots[atomic_number] = slot
        self._rebuild_layout()
        self._emit_change()

    def clear(self) -> None:
        for slot in list(self._slots.values()):
            slot.setParent(None)
            slot.deleteLater()
        self._counts.clear()
        self._slots.clear()
        self._emit_change()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-orbsim-element"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        element = _element_from_mime(event.mimeData())
        if element:
            self.add_element(element.atomic_number)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _handle_remove_one(self, atomic_number: int) -> None:
        if atomic_number not in self._counts:
            return
        self._counts[atomic_number] -= 1
        if self._counts[atomic_number] <= 0:
            self._remove_slot(atomic_number)
        else:
            self._slots[atomic_number].decrement()
        self._emit_change()

    def _handle_add_one(self, atomic_number: int) -> None:
        self.add_element(atomic_number)

    def _handle_remove_all(self, atomic_number: int) -> None:
        if atomic_number in self._counts:
            self._remove_slot(atomic_number)
            self._emit_change()

    def _remove_slot(self, atomic_number: int) -> None:
        self._counts.pop(atomic_number, None)
        slot = self._slots.pop(atomic_number, None)
        if slot:
            slot.setParent(None)
            slot.deleteLater()
        self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for idx, slot in enumerate(self._slots.values()):
            row = idx // 3
            col = idx % 3
            self._layout.addWidget(slot, row, col)

    def _emit_change(self) -> None:
        self.crafting_changed.emit(dict(self._counts))


_PERIODIC_POSITIONS_FULL = {
    1: (0, 0),
    2: (0, 17),
    3: (1, 0),
    4: (1, 1),
    5: (1, 12),
    6: (1, 13),
    7: (1, 14),
    8: (1, 15),
    9: (1, 16),
    10: (1, 17),
    11: (2, 0),
    12: (2, 1),
    13: (2, 12),
    14: (2, 13),
    15: (2, 14),
    16: (2, 15),
    17: (2, 16),
    18: (2, 17),
    19: (3, 0),
    20: (3, 1),
    21: (3, 2),
    22: (3, 3),
    23: (3, 4),
    24: (3, 5),
    25: (3, 6),
    26: (3, 7),
    27: (3, 8),
    28: (3, 9),
    29: (3, 10),
    30: (3, 11),
    31: (3, 12),
    32: (3, 13),
    33: (3, 14),
    34: (3, 15),
    35: (3, 16),
    36: (3, 17),
}


class PeriodicTableInventoryWidget(QtWidgets.QScrollArea):
    element_clicked = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self._tiles: list[ElementTileWidget] = []
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self._content = QtWidgets.QWidget()
        self._layout = QtWidgets.QGridLayout(self._content)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._populate()
        self.setWidget(self._content)

    def _elements_full(self) -> list[dict]:
        try:
            from periodic_table_cli.cli import load_data
        except Exception:
            return [
                {"symbol": element.symbol, "name": element.name, "atomic_number": element.atomic_number}
                for element in ELEMENTS_H_KR
            ]
        data = load_data()
        return data.get("elements", [])

    def _populate(self) -> None:
        elements = self._elements_full()
        for element in elements:
            atomic_number = int(element.get("atomicNumber") or element.get("atomic_number") or 0)
            symbol = element.get("symbol") or element.get("symbol")
            name = element.get("name") or element.get("name") or symbol
            if not atomic_number or not symbol:
                continue
            period = element.get("period")
            group = element.get("group")
            if atomic_number in range(57, 72):
                row = 7
                col = atomic_number - 57 + 3
            elif atomic_number in range(89, 104):
                row = 8
                col = atomic_number - 89 + 3
            elif period and group:
                row = int(period) - 1
                col = int(group) - 1
            elif atomic_number in _PERIODIC_POSITIONS_FULL:
                row, col = _PERIODIC_POSITIONS_FULL[atomic_number]
            else:
                continue
            tile = ElementTileWidget(Element(symbol, name, atomic_number), self._content)
            tile.setMinimumSize(48, 52)
            tile.element_clicked.connect(self.element_clicked)
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
        self._content.setStyleSheet(
            f"background: {colors['surface']};"
        )
        for tile in self._tiles:
            tile.apply_theme(tokens)


class StructureDiagramWidget(QtWidgets.QLabel):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        self._network = QtNetwork.QNetworkAccessManager(self)
        self._network.finished.connect(self._on_image_reply)
        self._pending_reply: QtNetwork.QNetworkReply | None = None
        self._cid: int | None = None
        self.setFixedHeight(240)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            f"background: {colors['surfaceAlt']}; border: 1px solid {colors['border']};"
        )

    def set_cid(self, cid: int | None) -> None:
        self._cid = cid
        if cid is None:
            self.setText("No structure selected.")
            return
        cache_path = self._cache_path(cid)
        if cache_path.exists():
            pixmap = QtGui.QPixmap(str(cache_path))
            if not pixmap.isNull():
                self.setPixmap(pixmap.scaledToHeight(260, QtCore.Qt.SmoothTransformation))
                self.setToolTip("2D structure depiction (PubChem)")
                return
        url = QtCore.QUrl(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG")
        request = QtNetwork.QNetworkRequest(url)
        self.setText("Loading 2D structure depiction…")
        self._pending_reply = self._network.get(request)

    def _cache_path(self, cid: int) -> Path:
        base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
        path = Path(base) / "compound_images"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{cid}.png"

    def _on_image_reply(self, reply: QtNetwork.QNetworkReply) -> None:
        if reply != self._pending_reply:
            return
        if reply.error() != QtNetwork.QNetworkReply.NetworkError.NoError:
            self.setText("2D structure depiction unavailable offline.")
            reply.deleteLater()
            return
        data = reply.readAll()
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(data)
        if not pixmap.isNull():
            self.setPixmap(pixmap.scaledToHeight(260, QtCore.Qt.SmoothTransformation))
            self.setToolTip("2D structure depiction (PubChem)")
            try:
                self._cache_path(self._cid or 0).write_bytes(bytes(data))
            except OSError:
                pass
        else:
            self.setText("2D structure depiction unavailable.")
        reply.deleteLater()


class BondingAndPolarityWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        layout = QtWidgets.QVBoxLayout(self)
        self.bonding_label = QtWidgets.QLabel("Bonding: —")
        self.bonding_label.setWordWrap(True)
        self.polarity_label = QtWidgets.QLabel("Polarity: —")
        self.polarity_label.setWordWrap(True)
        layout.addWidget(self.bonding_label)
        layout.addWidget(self.polarity_label)

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(f"color: {colors['text']};")

    def set_bonding(self, text: str, polarity_text: str) -> None:
        self.bonding_label.setText(text)
        self.polarity_label.setText(polarity_text)


class OxidationStateTableWidget(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Element", "Count", "Oxidation state"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens["colors"]
        self.setStyleSheet(
            f"background: {colors['surface']}; color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
        )


class CompoundPropertiesTableWidget(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Property", "Value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens["colors"]
        self.setStyleSheet(
            f"background: {colors['surface']}; color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
        )


class CompoundPreviewPane(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_tokens: dict | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        header_layout = QtWidgets.QHBoxLayout()
        self.pubchem_button = QtWidgets.QPushButton("Open PubChem")
        self.pubchem_button.clicked.connect(self._open_pubchem)
        header_layout.addWidget(self.pubchem_button)
        layout.addLayout(header_layout)

        self.source_label = QtWidgets.QLabel("Source: PubChem")
        self.source_label.setOpenExternalLinks(True)
        self.source_label.setText('<a href="https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines">Source: PubChem</a>')
        header_layout.addWidget(self.source_label)

        self.structure_widget = StructureDiagramWidget(self)
        layout.addWidget(self.structure_widget)

        self.bonding_widget = BondingAndPolarityWidget(self)
        layout.addWidget(self.bonding_widget)

        self.oxidation_label = QtWidgets.QLabel("Oxidation states")
        layout.addWidget(self.oxidation_label)
        self.oxidation_table = OxidationStateTableWidget(self)
        layout.addWidget(self.oxidation_table)

        self.properties_label = QtWidgets.QLabel("Compound properties")
        layout.addWidget(self.properties_label)
        self.properties_table = CompoundPropertiesTableWidget(self)
        layout.addWidget(self.properties_table)

        self._pubchem_url: str | None = None

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(f"color: {colors['text']};")
        self.structure_widget.apply_theme(tokens)
        self.bonding_widget.apply_theme(tokens)
        self.oxidation_table.apply_theme(tokens)
        self.properties_table.apply_theme(tokens)

    def set_compound(self, compound: dict | None) -> None:
        if not compound:
            self._pubchem_url = None
            self.structure_widget.set_cid(None)
            self.bonding_widget.set_bonding("Bonding: —", "Polarity: —")
            self._populate_table(self.oxidation_table, [])
            self._populate_table(self.properties_table, [])
            return
        raw_name = compound.get("name") or ""
        name = raw_name.split(";")[0].strip()
        self._pubchem_url = compound.get("pubchem_url")
        self.structure_widget.set_cid(compound.get("cid"))
        oxidation_states, heuristic = estimate_oxidation_states(compound)
        if heuristic:
            self.oxidation_label.setText("Estimated oxidation states (heuristic)")
        else:
            self.oxidation_label.setText("Oxidation states")
        self._populate_oxidation_table(compound.get("elements", {}), oxidation_states)
        bonding_text, polarity_text = classify_bonding_and_polarity(compound)
        self.bonding_widget.set_bonding(bonding_text, polarity_text)
        self._populate_properties(compound)

    def _populate_oxidation_table(self, elements: dict, oxidation_states: dict[str, int | None]) -> None:
        rows = []
        for atomic_number, count in sorted(elements.items()):
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atomic_number), str(atomic_number))
            state = oxidation_states.get(symbol)
            rows.append((symbol, str(count), str(state) if state is not None else "Unknown"))
        self._populate_table(self.oxidation_table, rows)

    def _populate_properties(self, compound: dict) -> None:
        rows = []
        if compound.get("mol_weight"):
            rows.append(("Molecular weight", str(compound["mol_weight"])))
        if compound.get("iupac_name"):
            iupac = str(compound["iupac_name"]).split(";")[0].strip()
            rows.append(("IUPAC name", iupac))
        if compound.get("smiles"):
            rows.append(("SMILES", compound["smiles"]))
        self._populate_table(self.properties_table, rows)

    def _populate_table(self, table: QtWidgets.QTableWidget, rows: list[tuple[str, str]]) -> None:
        table.setRowCount(0)
        for row_index, row in enumerate(rows):
            table.insertRow(row_index)
            for col, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, col, item)

    def _open_pubchem(self) -> None:
        if self._pubchem_url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._pubchem_url))


def estimate_oxidation_states(compound: dict) -> tuple[dict[str, int | None], bool]:
    elements = compound.get("elements", {})
    symbol_counts = {}
    for atomic_number, count in elements.items():
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atomic_number))
        if symbol:
            symbol_counts[symbol] = int(count)

    fixed_states = {"F": -1, "Cl": -1, "Br": -1, "I": -1, "O": -2, "H": 1}
    oxidation_states: dict[str, int | None] = {symbol: None for symbol in symbol_counts}
    total_charge = 0
    unknown_symbols = []
    for symbol, count in symbol_counts.items():
        if symbol in fixed_states:
            oxidation_states[symbol] = fixed_states[symbol]
            total_charge += fixed_states[symbol] * count
        else:
            unknown_symbols.append(symbol)

    heuristic = True
    if len(unknown_symbols) == 1:
        symbol = unknown_symbols[0]
        count = symbol_counts[symbol]
        if count > 0:
            oxidation_states[symbol] = int(round(-total_charge / count))
            heuristic = True
    else:
        for symbol in unknown_symbols:
            oxidation_states[symbol] = None
    return oxidation_states, heuristic


def classify_bonding_and_polarity(compound: dict) -> tuple[str, str]:
    elements = compound.get("elements", {})
    symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(int(z)) for z in elements.keys()]
    symbols = [s for s in symbols if s]
    if not symbols:
        return "Bonding: Unknown", "Polarity: Unknown"
    has_metal = any(is_metal(symbol) for symbol in symbols)
    has_nonmetal = any(is_nonmetal(symbol) for symbol in symbols)

    if has_metal and has_nonmetal:
        bonding = "Bonding: Ionic (heuristic). This compound pairs metal and nonmetal species."
    elif has_metal and not has_nonmetal:
        bonding = "Bonding: Metallic (heuristic). The compound is dominated by metallic elements."
    else:
        bonding = "Bonding: Covalent (heuristic). The compound is primarily nonmetals sharing electrons."

    en_values = [electronegativity(symbol) for symbol in symbols if electronegativity(symbol) is not None]
    if not en_values:
        return bonding, "Polarity: Unknown"
    max_en = max(en_values)
    min_en = min(en_values)
    delta_en = max_en - min_en
    asymmetry = 0.0
    if len(symbols) == 1:
        asymmetry = 0.0
    elif len(symbols) == 2 and elements.get(list(elements.keys())[0], 0) == 1 and elements.get(list(elements.keys())[1], 0) == 1:
        asymmetry = 1.0
    else:
        asymmetry = 0.5
    nea = delta_en * asymmetry
    if nea < 0.4:
        category = "Nonpolar"
    elif nea < 1.0:
        category = "Moderately polar"
    else:
        category = "Strongly polar"
    polarity = (
        f"Polarity estimate: NEA = {nea:.2f} (Δχ max {delta_en:.2f}, asymmetry {asymmetry:.2f}). "
        f"Category: {category}. This is an estimated metric based on electronegativity and symmetry heuristics."
    )
    return bonding, polarity


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
