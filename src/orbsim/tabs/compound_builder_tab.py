from __future__ import annotations

from collections import Counter

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.chem.compound_db import get_compound_details, query_compounds_by_elements
from orbsim.chem.elements import ATOMIC_NUMBER_TO_SYMBOL
from orbsim.chem.formula_parser import expand_formula_to_atomic_numbers, parse_formula
from orbsim.dialogs.compound_details_dialog import CompoundDetailsDialog
from orbsim.ui.generated.ui_compound_builder import Ui_CompoundBuilderTab
from orbsim.widgets import CraftingGridWidget, InventoryPeriodicTableWidget


PUBCHEM_CITATION_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines"


class CompoundBuilderTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_CompoundBuilderTab()
        self.ui.setupUi(self)

        self._theme_tokens: dict | None = None
        self.inventory_widget = InventoryPeriodicTableWidget(self)
        self.crafting_grid = CraftingGridWidget(self)

        self.ui.inventoryLayout.replaceWidget(self.ui.inventoryPlaceholder, self.inventory_widget)
        self.ui.inventoryPlaceholder.deleteLater()
        self.ui.craftingLayout.replaceWidget(self.ui.craftingPlaceholder, self.crafting_grid)
        self.ui.craftingPlaceholder.deleteLater()

        self._active_formula: str | None = None
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._refresh_recipe_list)

        self.ui.addFormulaButton.clicked.connect(self._apply_formula)
        self.ui.clearButton.clicked.connect(self._clear_crafting)
        self.ui.formulaLineEdit.returnPressed.connect(self._apply_formula)
        self.ui.recipeSearchLineEdit.textChanged.connect(self._queue_refresh)
        self.crafting_grid.changed.connect(self._on_crafting_changed)
        self.ui.recipeListWidget.itemActivated.connect(self._open_compound_dialog)
        self.ui.openCitationButton.clicked.connect(self._open_citation_page)

        self.ui.citationLabel.setOpenExternalLinks(True)
        self.ui.citationLabel.setText(
            "Compound data from PubChem (NIH/NLM). "
            f"Cite: <a href=\"{PUBCHEM_CITATION_URL}\">PubChem Citation Guidelines</a>."
        )

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
        self.ui.citationFrame.setStyleSheet(
            f"background: {colors['surfaceAlt']};"
            f"border: 1px solid {colors['border']};"
            f"border-radius: {radii['sm']}px;"
        )
        citation_font = self.ui.citationLabel.font()
        citation_font.setPointSize(font["baseSize"])
        self.ui.citationLabel.setFont(citation_font)

    def _queue_refresh(self) -> None:
        self._update_timer.start()

    def _on_crafting_changed(self) -> None:
        self._update_current_elements_label()
        self._queue_refresh()

    def _update_current_elements_label(self) -> None:
        elements = self.crafting_grid.get_elements()
        if not elements:
            self.ui.currentElementsLabel.setText("Current elements: (none)")
            return
        counts = Counter(elements)
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
            elements = expand_formula_to_atomic_numbers(text)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Formula error", str(exc))
            return
        if len(elements) > 9:
            QtWidgets.QMessageBox.information(
                self,
                "Formula too large",
                "Formula exceeds the 3×3 grid; truncating to the first 9 atoms.",
            )
            elements = elements[:9]
        self.crafting_grid.set_elements(elements)
        self._active_formula = text
        self._on_crafting_changed()

    def _clear_crafting(self) -> None:
        self.crafting_grid.clear()
        self._active_formula = None
        self._on_crafting_changed()

    def _refresh_recipe_list(self) -> None:
        elements = self.crafting_grid.get_elements()
        search = self.ui.recipeSearchLineEdit.text().strip() or None
        formula = None
        if self._active_formula:
            try:
                parse_formula(self._active_formula)
                formula = self._active_formula
            except ValueError:
                formula = None
        results = query_compounds_by_elements(elements, formula=formula, search=search)
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
            if result["exact_formula"] and self._theme_tokens:
                item.setForeground(QtGui.QColor(self._theme_tokens["colors"]["accent"]))
            self.ui.recipeListWidget.addItem(item)

    def _open_compound_dialog(self, item: QtWidgets.QListWidgetItem) -> None:
        cid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not cid:
            return
        compound = get_compound_details(int(cid))
        if not compound:
            QtWidgets.QMessageBox.warning(self, "Missing data", "Compound details are unavailable.")
            return
        dialog = CompoundDetailsDialog(compound, self)
        dialog.exec()

    def _open_citation_page(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(PUBCHEM_CITATION_URL))
