from __future__ import annotations

from dataclasses import dataclass

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


class PeriodicTableList(QtWidgets.QListWidget):
    element_dragged = QtCore.Signal(Element)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)
        self._populate()

    def _populate(self) -> None:
        for element in PERIODIC_TABLE:
            item = QtWidgets.QListWidgetItem(f"{element.symbol} — {element.name}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, element)
            self.addItem(item)

    def startDrag(self, supported_actions: QtCore.Qt.DropActions) -> None:
        item = self.currentItem()
        if not item:
            return
        element = item.data(QtCore.Qt.ItemDataRole.UserRole)
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
        self.plotter = QtInteractor(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter)

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
