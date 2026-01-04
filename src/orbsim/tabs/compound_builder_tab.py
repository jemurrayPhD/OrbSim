from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtCore, QtWidgets

from orbsim.chem.compound_db import db_exists, get_compound_details, get_db_path, query_compounds_by_elements
from orbsim.chem.elements import get_atomic_number, get_symbol
from orbsim.chem.formula_parser import parse_formula
from orbsim.ui.generated.ui_compound_builder import Ui_CompoundBuilderTab
from orbsim.widgets import CraftingTableWidget, CompoundPreviewPane, PeriodicTableInventoryWidget


class CompoundBuilderTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_CompoundBuilderTab()
        self.ui.setupUi(self)
        self.setMinimumSize(1100, 720)

        self._theme_tokens: dict | None = None
        self._db_ready = False
        self._builder_process: QtCore.QProcess | None = None
        self._build_stdout: list[str] = []
        self._build_stderr: list[str] = []
        self._active_formula: str | None = None

        self.inventory_widget = PeriodicTableInventoryWidget(self)
        self.crafting_grid = CraftingTableWidget(self)
        self.preview_pane = CompoundPreviewPane(self)
        self.preview_scroll = QtWidgets.QScrollArea(self)
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_pane)

        self.ui.inventoryLayout.replaceWidget(self.ui.inventoryPlaceholder, self.inventory_widget)
        self.ui.inventoryPlaceholder.deleteLater()
        self.ui.craftingLayout.replaceWidget(self.ui.craftingPlaceholder, self.crafting_grid)
        self.ui.craftingPlaceholder.deleteLater()
        self.ui.dataLayout.replaceWidget(self.ui.dataPlaceholder, self.preview_scroll)
        self.ui.dataPlaceholder.deleteLater()

        self.ui.craftingSection.setMinimumSize(320, 320)
        self.ui.recipeSection.setMinimumSize(320, 320)
        self.ui.dataSection.setMinimumSize(360, 320)
        self.ui.inventorySection.setMinimumHeight(260)
        self.ui.craftingLayout.setSpacing(10)
        self.ui.currentElementsLabel.setContentsMargins(0, 6, 0, 0)
        self.upper_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        for section in (self.ui.craftingSection, self.ui.recipeSection, self.ui.dataSection):
            self.ui.upperLayout.removeWidget(section)
            self.upper_splitter.addWidget(section)
        self.upper_splitter.setSizes([360, 360, 520])
        self.ui.upperLayout.addWidget(self.upper_splitter)

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._refresh_recipe_list)

        self.ui.addFormulaButton.clicked.connect(self._apply_formula)
        self.ui.clearButton.clicked.connect(self._clear_crafting)
        self.ui.formulaLineEdit.returnPressed.connect(self._apply_formula)
        self.ui.recipeSearchLineEdit.textChanged.connect(self._queue_refresh)
        self.ui.compatibleRadio.toggled.connect(self._queue_refresh)
        self.ui.onlyElementsRadio.toggled.connect(self._queue_refresh)
        self.crafting_grid.crafting_changed.connect(self._on_crafting_changed)
        self.ui.recipeListWidget.itemSelectionChanged.connect(self._select_compound)
        self.ui.buildDbButton.clicked.connect(self._start_build)
        self.inventory_widget.element_clicked.connect(self._add_element_from_inventory)

        self.ui.buildProgressBar.setRange(0, 1)
        self._update_db_state()

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        font = tokens["font"]
        self.setStyleSheet(
            "QLabel#craftingHeaderLabel, QLabel#recipeHeaderLabel, "
            "QLabel#dataHeaderLabel, QLabel#inventoryHeaderLabel {"
            f"color: {colors['textMuted']};"
            "font-weight: 600;"
            f"font-size: {font['titleSize']}px;"
            f"padding-bottom: {tokens['spacing']['xs']}px;"
            "}"
            "QLabel#currentElementsLabel {"
            f"color: {colors['text']};"
            f"padding-top: {tokens['spacing']['sm']}px;"
            "}"
            "QLineEdit, QListWidget {"
            f"background: {colors['surface']};"
            f"color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QPushButton, QCheckBox, QRadioButton {"
            f"color: {colors['text']};"
            "}"
            "QPushButton {"
            f"background: {colors['surfaceAlt']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            f"padding: {tokens['spacing']['xs']}px {tokens['spacing']['sm']}px;"
            "}"
            "QPushButton:hover {"
            f"background: {colors['accentHover']};"
            f"color: {colors['bg']};"
            "}"
        )
        self.inventory_widget.apply_theme(tokens)
        self.crafting_grid.apply_theme(tokens)
        self.preview_pane.apply_theme(tokens)
        self.preview_scroll.setStyleSheet(
            f"background: {colors['surface']}; border: 1px solid {colors['border']};"
        )

    def _queue_refresh(self) -> None:
        if self._db_ready:
            self._update_timer.start()

    def _on_crafting_changed(self, payload: dict) -> None:
        counts = payload.get("counts", {})
        self._update_current_elements_label(counts)
        self._queue_refresh()

    def _add_element_from_inventory(self, atomic_number: int) -> None:
        self.crafting_grid.add_element_to_first_empty(atomic_number)

    def _update_current_elements_label(self, counts: dict[int, int]) -> None:
        if not counts:
            self.ui.currentElementsLabel.setText("Current elements: (none)")
            return
        parts = []
        for atomic_number in sorted(counts):
            symbol = get_symbol(atomic_number)
            if not symbol:
                continue
            count = counts[atomic_number]
            parts.append(f"{symbol}{count if count > 1 else ''}")
        self.ui.currentElementsLabel.setText("Current elements: " + " ".join(parts))

    def _apply_formula(self) -> None:
        text = self.ui.formulaLineEdit.text().strip()
        if not text:
            return
        try:
            counts = parse_formula(text)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Formula error", str(exc))
            return
        if len(counts) > 9:
            QtWidgets.QMessageBox.information(
                self,
                "Formula too large",
                "Formula exceeds the 9 distinct element limit; truncating to the first 9 element types.",
            )
            counts = dict(list(counts.items())[:9])
        atomic_counts: dict[int, int] = {}
        for symbol, count in counts.items():
            atomic_number = get_atomic_number(symbol)
            if atomic_number:
                atomic_counts[atomic_number] = count
        self.crafting_grid.set_counts(atomic_counts)
        self._active_formula = text
        self._on_crafting_changed({"counts": self.crafting_grid.counts()})

    def _clear_crafting(self) -> None:
        self.crafting_grid.clear()
        self._active_formula = None
        self.preview_pane.set_compound(None)
        self._on_crafting_changed({"counts": self.crafting_grid.counts()})

    def _refresh_recipe_list(self) -> None:
        if not self._db_ready:
            return
        required_counts = self.crafting_grid.counts()
        only_elements = self.ui.onlyElementsRadio.isChecked()
        results = query_compounds_by_elements(required_counts, only_elements=only_elements)
        search = self.ui.recipeSearchLineEdit.text().strip().lower()
        if search:
            results = [
                item
                for item in results
                if search in (item["name"] or "").lower() or search in (item["formula"] or "").lower()
            ]
        self.ui.recipeListWidget.clear()
        if not results:
            item = QtWidgets.QListWidgetItem("No matching compounds found.")
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.ui.recipeListWidget.addItem(item)
            return
        for result in results:
            label = f"{result['name']} — {result['formula']}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result["cid"])
            self.ui.recipeListWidget.addItem(item)

    def _select_compound(self) -> None:
        selected = self.ui.recipeListWidget.selectedItems()
        if not selected:
            self.preview_pane.set_compound(None)
            return
        cid = selected[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if not cid:
            return
        compound = get_compound_details(int(cid))
        self.preview_pane.set_compound(compound)

    def _update_db_state(self) -> None:
        self._db_ready = db_exists()
        self.ui.recipeSearchLineEdit.setEnabled(self._db_ready)
        self.ui.recipeListWidget.setEnabled(self._db_ready)
        self.ui.compatibleRadio.setEnabled(self._db_ready)
        self.ui.onlyElementsRadio.setEnabled(self._db_ready)
        self.ui.buildDbButton.setVisible(not self._db_ready)
        self.ui.buildProgressBar.setVisible(False)
        if self._db_ready:
            self.ui.dbStatusLabel.setText("DB status: ready")
            self._refresh_recipe_list()
        else:
            self.ui.dbStatusLabel.setText("DB status: not built")
            self.ui.recipeListWidget.clear()

    def _start_build(self) -> None:
        if self._builder_process and self._builder_process.state() != QtCore.QProcess.NotRunning:
            return
        self._build_stdout.clear()
        self._build_stderr.clear()
        seed_path = Path(__file__).resolve().parents[3] / "tools" / "seed_compounds.csv"
        db_path = get_db_path()
        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "tools.build_compound_db",
                "--seed",
                str(seed_path),
                "--out",
                str(db_path),
            ]
        )
        process.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(lambda: self._read_process_output(process))
        process.readyReadStandardError.connect(lambda: self._read_process_error(process))
        process.finished.connect(self._on_build_finished)
        process.start()
        self._builder_process = process
        self.ui.buildDbButton.setEnabled(False)
        self.ui.buildProgressBar.setVisible(True)
        self.ui.buildProgressBar.setRange(0, 0)
        self.ui.dbStatusLabel.setText("DB status: building…")

    def _read_process_output(self, process: QtCore.QProcess) -> None:
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if text:
            self._build_stdout.append(text)

    def _read_process_error(self, process: QtCore.QProcess) -> None:
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="ignore")
        if text:
            self._build_stderr.append(text)

    def _on_build_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self.ui.buildProgressBar.setVisible(False)
        self.ui.buildDbButton.setEnabled(True)
        success = exit_code == 0 and status == QtCore.QProcess.NormalExit and db_exists()
        if success:
            self.ui.dbStatusLabel.setText("DB status: ready")
            self._update_db_state()
            return
        stderr = "".join(self._build_stderr).strip()
        stdout = "".join(self._build_stdout).strip()
        details = stderr or stdout or "Unknown error while building the database."
        QtWidgets.QMessageBox.critical(self, "Database build failed", details)
        self.ui.dbStatusLabel.setText("DB status: not built")
