from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.tabs.periodic_table_tab import PeriodicTableTab
from orbsim.ui.generated.ui_electron_shells import Ui_ElectronShellsTab
from periodic_table_cli.cli import load_data


class ElectronShellsTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_ElectronShellsTab()
        self.ui.setupUi(self)
        self.data = load_data()
        self.current_element: dict | None = None
        self.current_oxidation: int = 0

        layout = self.ui.contentLayout
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Element:"))
        self.element_combo = QtWidgets.QComboBox()
        self._populate_elements()
        self.element_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.element_combo.setMaximumWidth(260)
        controls.addWidget(self.element_combo)
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
        self.oxidation_combo = QtWidgets.QComboBox()
        controls.addWidget(self.oxidation_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.bohr_view = PeriodicTableTab.BohrViewer()
        self.subshell_view = self.SubshellGridView(self.bohr_view)
        self.box_container = self.OrbitalBoxContainer(self.bohr_view)

        self.config_label = QtWidgets.QLabel()
        self.config_label.setWordWrap(True)
        layout.addWidget(self.config_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setOpaqueResize(True)
        splitter.setChildrenCollapsible(False)
        self._wrap_containers: list[QtWidgets.QWidget] = []
        splitter.addWidget(self._wrap_with_label("Bohr model", self.bohr_view))
        splitter.addWidget(self._wrap_with_label("Expanded subshells", self.subshell_view))
        splitter.addWidget(self._wrap_with_label("Orbital box view", self.box_container))
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([600, 400, 400])
        layout.addWidget(splitter, 1)

        self.element_combo.currentIndexChanged.connect(self._on_element_change)
        self.oxidation_combo.currentIndexChanged.connect(self._on_oxidation_change)
        self.prev_btn.clicked.connect(lambda: self._step_element(-1))
        self.next_btn.clicked.connect(lambda: self._step_element(1))
        if self.element_combo.count() > 0:
            self.element_combo.setCurrentIndex(0)
            self._on_element_change()
        self.apply_theme({"colors": {"surface": "#f8fafc", "border": "#cbd5e1", "text": "#0f172a", "textMuted": "#475569"}})

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        bg = colors.get("surface", "#f8fafc")
        border = colors.get("border", "#cbd5e1")
        text = colors.get("text", "#0f172a")
        muted = colors.get("textMuted", "#475569")
        self.config_label.setStyleSheet(f"color: {muted};")
        self._wrap_style = (
            f"QWidget {{ background: {bg}; border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel {{ color: {text}; }}"
        )
        for container in getattr(self, "_wrap_containers", []):
            container.setStyleSheet(self._wrap_style)

    def _wrap_with_label(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        container.setStyleSheet(getattr(self, "_wrap_style", ""))
        self._wrap_containers.append(container)
        return container

    def _populate_elements(self) -> None:
        self.element_combo.clear()
        elements = sorted(self.data["elements"], key=lambda e: e.get("atomicNumber", 0))
        for elem in elements:
            an = elem.get("atomicNumber", 0)
            sym = elem.get("symbol", "")
            name = elem.get("name", "")
            self.element_combo.addItem(f"{an}: {sym} — {name}", userData=elem)

    def _oxidation_items(self, elem: dict) -> list[tuple[str, int]]:
        states = self.bohr_view._parse_oxidation_states_local(elem)
        items = [("Neutral (0)", 0)]
        for s in states:
            label = f"Ion +{s}" if s > 0 else f"Ion {s}"
            items.append((label, s))
        return items

    def _on_element_change(self) -> None:
        data = self.element_combo.currentData()
        if not data:
            return
        self.current_element = data
        items = self._oxidation_items(data)
        self.oxidation_combo.blockSignals(True)
        self.oxidation_combo.clear()
        for label, val in items:
            self.oxidation_combo.addItem(label, userData=val)
        self.oxidation_combo.blockSignals(False)
        neutral_idx = 0
        for idx in range(self.oxidation_combo.count()):
            if self.oxidation_combo.itemData(idx) == 0:
                neutral_idx = idx
                break
        self.oxidation_combo.setCurrentIndex(neutral_idx)
        self.current_oxidation = self.oxidation_combo.currentData() or 0
        self._refresh_views()

    def _on_oxidation_change(self) -> None:
        self.current_oxidation = self.oxidation_combo.currentData() or 0
        self._refresh_views()

    def _step_element(self, delta: int) -> None:
        if self.element_combo.count() == 0:
            return
        idx = self.element_combo.currentIndex()
        new_idx = max(0, min(self.element_combo.count() - 1, idx + delta))
        if new_idx != idx:
            self.element_combo.setCurrentIndex(new_idx)

    def _refresh_views(self) -> None:
        if not self.current_element:
            return
        self.bohr_view.set_element(self.current_element)
        self.bohr_view.set_oxidation_state(int(self.current_oxidation))
        self.subshell_view.update_view(self.current_element, int(self.current_oxidation))
        self.box_container.update_view(self.current_element, int(self.current_oxidation))
        # update configs
        shells, subshells = self.bohr_view._shell_counts(self.current_element)
        full_cfg = self._config_string(subshells)
        abbrev = self._config_abbrev(self.current_element, subshells)
        self.config_label.setText(f"Full: {full_cfg}\nAbbrev: {abbrev}")

    def _config_string(self, subshells: dict[tuple[int, int], int]) -> str:
        subshell_order = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
            (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
            (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
            (7, 0), (5, 3), (6, 2), (7, 1),
        ]
        parts = []
        label_map = {0: "s", 1: "p", 2: "d", 3: "f"}
        for n, l in subshell_order:
            if (n, l) in subshells:
                cnt = subshells[(n, l)]
                parts.append(f"{n}{label_map.get(l, '?')}{cnt}")
        return " ".join(parts)

    def _config_abbrev(self, elem: dict, subshells: dict[tuple[int, int], int]) -> str:
        noble_gases = [
            (2, "He"), (10, "Ne"), (18, "Ar"), (36, "Kr"),
            (54, "Xe"), (86, "Rn"), (118, "Og")
        ]
        an = int(elem.get("atomicNumber", 0))
        ng = max((n for n in noble_gases if n[0] < an), default=None, key=lambda x: x[0])
        if not ng:
            return self._config_string(subshells)
        ng_electrons, ng_symbol = ng
        _, ng_subshells = self.bohr_view._shell_counts({"atomicNumber": ng_electrons, "period": elem.get("period", 1)})
        remainder: dict[tuple[int, int], int] = {}
        for key, cnt in subshells.items():
            remainder[key] = max(0, cnt - ng_subshells.get(key, 0))
        remainder = {k: v for k, v in remainder.items() if v > 0}
        rest = self._config_string(remainder)
        return f"[{ng_symbol}] {rest}" if rest else f"[{ng_symbol}]"

    class SubshellGridView(QtWidgets.QWidget):
        def __init__(self, bohr_view: "PeriodicTableTab.BohrViewer", parent=None):
            super().__init__(parent)
            self.bohr_view = bohr_view
            self.grid = QtWidgets.QGridLayout(self)
            self.grid.setAlignment(QtCore.Qt.AlignTop)
            self.grid.setHorizontalSpacing(10)
            self.grid.setVerticalSpacing(8)
            self.setMinimumSize(220, 220)

        def update_view(self, elem: dict, oxidation: int) -> None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            if not elem:
                return
            self.bohr_view.oxidation_state = oxidation
            shells, subshells = self.bohr_view._shell_counts(elem)
            l_labels = ["s", "p", "d", "f"]
            for col, label in enumerate(l_labels):
                hdr = QtWidgets.QLabel(label)
                hdr.setAlignment(QtCore.Qt.AlignCenter)
                hdr.setStyleSheet("font-weight: bold;")
                self.grid.addWidget(hdr, 0, col + 1)
            for n in range(1, len(shells) + 1):
                row_lbl = QtWidgets.QLabel(f"n={n}")
                row_lbl.setAlignment(QtCore.Qt.AlignCenter)
                row_lbl.setStyleSheet("")
                self.grid.addWidget(row_lbl, n, 0)
                lmax = min(3, n - 1)
                for l in range(0, lmax + 1):
                    cap = self.bohr_view._subshell_capacity(l)
                    filled = min(cap, subshells.get((n, l), 0))
                    view = self.bohr_view.SubshellMiniWidget(
                        f"{n}{l_labels[l]}",
                        filled,
                        cap,
                        self.bohr_view._angle_sets().get(l, []),
                        filled >= cap,
                        self,
                    )
                    self.grid.addWidget(view, n, l + 1)
            self.updateGeometry()

    class OrbitalBoxContainer(QtWidgets.QWidget):
        def __init__(self, bohr_view: "PeriodicTableTab.BohrViewer", parent=None):
            super().__init__(parent)
            self.bohr_view = bohr_view
            self.view = PeriodicTableTab.OrbitalBoxView(
                {}, 0, self.bohr_view._shell_counts, self.bohr_view._subshell_capacity, self
            )
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.view)
            self.setMinimumSize(240, 240)

        def update_view(self, elem: dict, oxidation: int) -> None:
            if not elem:
                return
            self.view.elem = elem
            self.view.oxidation = oxidation
            self.view.update()
