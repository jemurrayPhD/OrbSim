from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.chem.compound_db import db_exists, get_compound_details, get_db_path, query_compounds_by_elements
from orbsim.chem.elements import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ELEMENT
from orbsim.chem.formula_parser import parse_formula
from orbsim.ui.generated.ui_compound_builder import Ui_CompoundBuilderTab
from orbsim.widgets import AggregatedCraftingGridWidget, CompoundPreviewPane, PeriodicTableInventoryWidget


PUBCHEM_CITATION_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines"


class CompoundBuilderTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_CompoundBuilderTab()
        self.ui.setupUi(self)

        self._theme_tokens: dict | None = None
        self._db_ready = False
        self._builder_process: QtCore.QProcess | None = None
        self._build_stdout: list[str] = []
        self._build_stderr: list[str] = []
        self._active_formula: str | None = None

        self.inventory_widget = PeriodicTableInventoryWidget(self)
        self.crafting_grid = AggregatedCraftingGridWidget(self)
        self.preview_pane = CompoundPreviewPane(self)

        self.ui.inventoryLayout.replaceWidget(self.ui.inventoryPlaceholder, self.inventory_widget)
        self.ui.inventoryPlaceholder.deleteLater()
        self.ui.craftingLayout.replaceWidget(self.ui.craftingPlaceholder, self.crafting_grid)
        self.ui.craftingPlaceholder.deleteLater()
        self.ui.previewLayout.replaceWidget(self.ui.previewPlaceholder, self.preview_pane)
        self.ui.previewPlaceholder.deleteLater()

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._refresh_recipe_list)

        self.ui.addFormulaButton.clicked.connect(self._apply_formula)
        self.ui.clearButton.clicked.connect(self._clear_crafting)
        self.ui.formulaLineEdit.returnPressed.connect(self._apply_formula)
        self.ui.recipeSearchLineEdit.textChanged.connect(self._queue_refresh)
        self.ui.compatibleRadio.toggled.connect(self._queue_refresh)
        self.ui.exactRadio.toggled.connect(self._queue_refresh)
        self.crafting_grid.crafting_changed.connect(self._on_crafting_changed)
        self.ui.recipeListWidget.itemSelectionChanged.connect(self._select_compound)
        self.ui.openCitationButton.clicked.connect(self._open_citation_page)
        self.ui.buildDbButton.clicked.connect(self._start_build)
        self.inventory_widget.element_added.connect(self._add_element_from_inventory)

        self.ui.citationLabel.setOpenExternalLinks(True)
        self.ui.citationLabel.setText(
            "Compound data from PubChem (NIH/NLM). "
            f"Cite: <a href=\"{PUBCHEM_CITATION_URL}\">PubChem Citation Guidelines</a>."
        )
        self.ui.buildProgressBar.setRange(0, 1)
        self._update_db_state()

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        radii = tokens["radii"]
        font = tokens["font"]
        self.setStyleSheet(
            "QGroupBox {"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            f"margin-top: {radii['sm']}px;"
            "}"
            "QGroupBox::title {"
            f"color: {colors['text']};"
            f"padding: 0 {radii['sm']}px;"
            "}"
            "QLineEdit, QListWidget {"
            f"background: {colors['surface']};"
            f"color: {colors['text']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
            "}"
            "QPushButton {"
            f"background: {colors['surfaceAlt']};"
            f"color: {colors['text']};"
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
        self.ui.citationFrame.setStyleSheet(
            f"background: {colors['surfaceAlt']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
        )
        citation_font = self.ui.citationLabel.font()
        citation_font.setPointSize(font["baseSize"])
        self.ui.citationLabel.setFont(citation_font)

    def _queue_refresh(self) -> None:
        if self._db_ready:
            self._update_timer.start()

    def _on_crafting_changed(self, counts: dict[int, int]) -> None:
        self._update_current_elements_label(counts)
        self._queue_refresh()

    def _add_element_from_inventory(self, atomic_number: int) -> None:
        self.crafting_grid.add_element(atomic_number)

    def _update_current_elements_label(self, counts: dict[int, int]) -> None:
        if not counts:
            self.ui.currentElementsLabel.setText("Current elements: (none)")
            return
        parts = []
        for atomic_number in sorted(counts):
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, str(atomic_number))
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
            element = SYMBOL_TO_ELEMENT.get(symbol)
            if element:
                atomic_counts[element.atomic_number] = count
        self.crafting_grid.set_counts(atomic_counts)
        self._active_formula = text
        self._on_crafting_changed(self.crafting_grid.counts())

    def _clear_crafting(self) -> None:
        self.crafting_grid.clear()
        self._active_formula = None
        self.preview_pane.set_compound(None)
        self._on_crafting_changed(self.crafting_grid.counts())

    def _refresh_recipe_list(self) -> None:
        if not self._db_ready:
            return
        required_counts = self.crafting_grid.counts()
        exact = self.ui.exactRadio.isChecked()
        results = query_compounds_by_elements(required_counts, exact=exact)
        search = self.ui.recipeSearchLineEdit.text().strip().lower()
        if search:
            results = [
                item
                for item in results
                if search in (item["name"] or "").lower() or search in (item["formula"] or "").lower()
            ]
        formula = None
        if self._active_formula:
            try:
                parse_formula(self._active_formula)
                formula = self._active_formula
            except ValueError:
                formula = None
        if formula:
            results.sort(key=lambda item: (0 if (item["formula"] or "").lower() == formula.lower() else 1))
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

    def _open_citation_page(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(PUBCHEM_CITATION_URL))

    def _update_db_state(self) -> None:
        self._db_ready = db_exists()
        self.ui.recipeSearchLineEdit.setEnabled(self._db_ready)
        self.ui.recipeListWidget.setEnabled(self._db_ready)
        self.ui.compatibleRadio.setEnabled(self._db_ready)
        self.ui.exactRadio.setEnabled(self._db_ready)
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
