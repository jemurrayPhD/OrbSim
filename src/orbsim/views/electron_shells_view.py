from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.views.periodic_table_view import BohrViewer, OrbitalBoxView, SubshellMiniWidget


class BohrLegendWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._dot_size = 10
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)

        self._empty_dot = QtWidgets.QLabel()
        self._empty_label = QtWidgets.QLabel("Empty")
        self._occupied_dot = QtWidgets.QLabel()
        self._occupied_label = QtWidgets.QLabel("Occupied")
        self._filled_dot = QtWidgets.QLabel()
        self._filled_label = QtWidgets.QLabel("Filled subshell")

        for dot in (self._empty_dot, self._occupied_dot, self._filled_dot):
            dot.setFixedSize(self._dot_size, self._dot_size)

        layout.addWidget(self._empty_dot)
        layout.addWidget(self._empty_label)
        layout.addSpacing(8)
        layout.addWidget(self._occupied_dot)
        layout.addWidget(self._occupied_label)
        layout.addSpacing(8)
        layout.addWidget(self._filled_dot)
        layout.addWidget(self._filled_label)
        layout.addStretch()

    def apply_theme(self, tokens: dict, bohr_view: BohrViewer) -> None:
        colors = bohr_view.legend_colors()
        theme_colors = tokens.get("colors", {})
        text = theme_colors.get("text", colors["text"].name())
        border = theme_colors.get("border", "#303030")
        self._set_dot_style(self._empty_dot, colors["empty"].name(), border)
        self._set_dot_style(self._occupied_dot, colors["occupied"].name(), border)
        self._set_dot_style(self._filled_dot, colors["filled"].name(), border)
        for label in (self._empty_label, self._occupied_label, self._filled_label):
            label.setStyleSheet(f"color: {text};")

    def _set_dot_style(self, widget: QtWidgets.QLabel, color: str, border: str) -> None:
        radius = self._dot_size // 2
        widget.setStyleSheet(
            f"background: {color}; border: 1px solid {border}; border-radius: {radius}px;"
        )


class SubshellGridView(QtWidgets.QWidget):
    def __init__(self, bohr_view: BohrViewer, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.bohr_view = bohr_view
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setAlignment(QtCore.Qt.AlignTop)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(8)
        self.setMinimumSize(220, 220)
        self._theme_colors = {
            "text": "#0f172a",
            "textMuted": "#4b5563",
        }

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        colors = {**colors, "mode": tokens.get("meta", {}).get("mode")}
        self._theme_colors.update(colors)
        for child in self.findChildren(SubshellMiniWidget):
            child.apply_theme(colors)
        self.update()

    def update_view(self, elem: dict, oxidation: int) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        shells, subshells = self.bohr_view._shell_counts(elem)
        l_labels = ["s", "p", "d", "f"]
        for col, label in enumerate(l_labels):
            lbl = QtWidgets.QLabel(label)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {self._theme_colors['textMuted']}; font-weight: 600;")
            self.grid.addWidget(lbl, 0, col + 1)
        for n in range(1, len(shells) + 1):
            row_label = QtWidgets.QLabel(f"n={n}")
            row_label.setAlignment(QtCore.Qt.AlignCenter)
            row_label.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
            self.grid.addWidget(row_label, n, 0)
            lmax = min(3, n - 1)
            for l in range(0, lmax + 1):
                cap = self.bohr_view._subshell_capacity(l)
                filled = min(cap, subshells.get((n, l), 0))
                angles = list(self.bohr_view._angle_sets().get(l, []))
                view = SubshellMiniWidget(
                    f"{n}{l_labels[l]}",
                    filled,
                    cap,
                    angles,
                    filled >= cap,
                    self,
                )
                view.apply_theme(self._theme_colors)
                font = view.font()
                font.setPointSize(font.pointSize() + 1)
                view.setFont(font)
                self.grid.addWidget(view, n, l + 1)


class OrbitalBoxContainer(QtWidgets.QWidget):
    def __init__(self, bohr_view: BohrViewer, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.bohr_view = bohr_view
        layout = QtWidgets.QVBoxLayout(self)
        self.view = OrbitalBoxView({}, 0, self.bohr_view._shell_counts, self.bohr_view._subshell_capacity, self)
        layout.addWidget(self.view)

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        colors = {**colors, "mode": tokens.get("meta", {}).get("mode")}
        self.view.apply_theme(colors)

    def update_view(self, elem: dict, oxidation: int) -> None:
        self.view.elem = elem
        self.view.oxidation = oxidation
        self.view.update()
