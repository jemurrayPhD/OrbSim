from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.views.periodic_table_view import BohrViewer, OrbitalBoxView, SubshellMiniWidget


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
