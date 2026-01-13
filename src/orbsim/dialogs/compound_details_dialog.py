from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets

from orbsim.chem.elements import get_atomic_number, get_element, get_symbol
from orbsim.chem.formula_format import format_formula_from_string
from orbsim.content.compound_properties import describe_bonding_and_polarity
from orbsim import pedagogy
from orbsim.chem import compound_db


def _element_family(symbol: str) -> str:
    atomic_number = get_atomic_number(symbol)
    if atomic_number <= 0:
        return ""
    element = get_element(atomic_number)
    return str(element.get("family") or element.get("category") or element.get("categoryName") or "")


def _is_metal_symbol(symbol: str) -> bool:
    family = _element_family(symbol).lower()
    return "metal" in family and "nonmetal" not in family and "metalloid" not in family


def _is_nonmetal_symbol(symbol: str) -> bool:
    family = _element_family(symbol).lower()
    return "nonmetal" in family or "non-metal" in family


def _electronegativity_symbol(symbol: str) -> float | None:
    atomic_number = get_atomic_number(symbol)
    if atomic_number <= 0:
        return None
    element = get_element(atomic_number)
    value = element.get("electronegativity")
    try:
        return None if value in (None, "", "-") else float(value)
    except Exception:
        return None


PUBCHEM_IMAGE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
class CompoundDetailsDialog(QtWidgets.QDialog):
    def __init__(self, compound: dict, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.compound = compound
        self._network = QtNetwork.QNetworkAccessManager(self)
        self._network.finished.connect(self._on_image_reply)
        self._pending_image_reply: QtNetwork.QNetworkReply | None = None
        self._image_cache_path = self._image_cache_dir() / f"{compound['cid']}.png"

        formula_plain = format_formula_from_string(str(compound.get("formula") or "")).plain
        title_formula = formula_plain or str(compound.get("formula") or "")
        self.setWindowTitle(f"{compound['name']} — {title_formula}")
        self.setMinimumSize(640, 520)
        self._build_ui()
        self._populate()
        self._load_image()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setWordWrap(True)
        self.title_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header_layout.addWidget(self.title_label, 1)

        self.pubchem_button = QtWidgets.QPushButton("Open PubChem record")
        self.pubchem_button.clicked.connect(self._open_pubchem)
        header_layout.addWidget(self.pubchem_button, 0)
        layout.addLayout(header_layout)

        self.image_label = QtWidgets.QLabel("Loading 2D structure depiction…")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(220)
        layout.addWidget(self.image_label)

        self.oxidation_label = QtWidgets.QLabel("")
        self.oxidation_label.setWordWrap(True)
        layout.addWidget(self.oxidation_label)

        self.oxidation_table = QtWidgets.QTableWidget(0, 3)
        self.oxidation_table.setHorizontalHeaderLabels(["Element", "Count", "Oxidation state"])
        self.oxidation_table.horizontalHeader().setStretchLastSection(True)
        self.oxidation_table.verticalHeader().setVisible(False)
        self.oxidation_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.oxidation_table)

        self.bonding_label = QtWidgets.QLabel("")
        self.bonding_label.setWordWrap(True)
        layout.addWidget(self.bonding_label)

        self.properties_label = QtWidgets.QLabel("")
        self.properties_label.setWordWrap(True)
        layout.addWidget(self.properties_label)

        self.pedagogy_browser = QtWidgets.QTextBrowser()
        self.pedagogy_browser.setOpenExternalLinks(True)
        self.pedagogy_browser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.pedagogy_browser.setMinimumHeight(140)
        layout.addWidget(self.pedagogy_browser)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def _populate(self) -> None:
        display = compound_db.format_compound_display(self.compound)
        title = display["primary_name"] or "Compound"
        formula_display = display["formula_display"] or self.compound.get("formula") or ""
        if formula_display:
            formatted = format_formula_from_string(formula_display)
            self.title_label.setText(f"{title} ({formatted.rich})")
        else:
            self.title_label.setText(title)

        oxidation_states, heuristic = estimate_oxidation_states(self.compound)
        if heuristic:
            self.oxidation_label.setText("Estimated oxidation states (heuristic)")
        else:
            self.oxidation_label.setText("Oxidation states")
        self._populate_oxidation_table(oxidation_states)

        summary = describe_bonding_and_polarity(self.compound)
        self.bonding_label.setText(f"{summary.bonding_sentence} {summary.polarity_sentence}")

        properties = [("Molecular weight", self.compound.get("mol_weight"))]
        if display["iupac_name"]:
            properties.append(("IUPAC", display["iupac_name"]))
        synonyms = self._clean_synonyms(display["synonyms"])
        if synonyms:
            properties.append(("Also known as", ", ".join(synonyms[:6])))
        prop_lines = [f"{label}: {value}" for label, value in properties if value]
        self.properties_label.setText("Known data: " + ("; ".join(prop_lines) if prop_lines else "Unavailable"))
        self.properties_label.setToolTip("Data source: PubChem (NIH/NLM).")
        self.pedagogy_browser.setHtml(pedagogy.compound_notes_html(self.compound.get("formula") or ""))

    def _populate_oxidation_table(self, oxidation_states: dict[str, int | None]) -> None:
        elements = self.compound.get("elements", {})
        self.oxidation_table.setRowCount(0)
        for row_index, (atomic_number, count) in enumerate(sorted(elements.items())):
            symbol = get_symbol(int(atomic_number))
            state = oxidation_states.get(symbol)
            state_text = str(state) if state is not None else "Unknown"
            self.oxidation_table.insertRow(row_index)
            for col, value in enumerate((symbol, str(count), state_text)):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.oxidation_table.setItem(row_index, col, item)

    def _open_pubchem(self) -> None:
        url = self.compound.get("pubchem_url")
        if url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def _image_cache_dir(self) -> Path:
        base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
        path = Path(base) / "compound_images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_image(self) -> None:
        if self._image_cache_path.exists():
            pixmap = QtGui.QPixmap(str(self._image_cache_path))
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaledToHeight(220, QtCore.Qt.SmoothTransformation))
                self.image_label.setToolTip("2D structure depiction (PubChem)")
                return
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(PUBCHEM_IMAGE_URL.format(cid=self.compound["cid"])))
        self._pending_image_reply = self._network.get(request)

    def _on_image_reply(self, reply: QtNetwork.QNetworkReply) -> None:
        if reply != self._pending_image_reply:
            return
        if reply.error() != QtNetwork.QNetworkReply.NetworkError.NoError:
            self.image_label.setText("2D structure depiction unavailable offline.")
            reply.deleteLater()
            return
        data = reply.readAll()
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(data)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaledToHeight(220, QtCore.Qt.SmoothTransformation))
            self.image_label.setToolTip("2D structure depiction (PubChem)")
            try:
                self._image_cache_path.write_bytes(bytes(data))
            except OSError:
                pass
        else:
            self.image_label.setText("2D structure depiction unavailable.")
        reply.deleteLater()

    @staticmethod
    def _clean_synonyms(values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text or len(text) > 80:
                continue
            lower = text.lower()
            if "inchikey" in lower or lower.startswith("inchi=") or lower.startswith("inchi"):
                continue
            if lower.startswith("cas ") or lower.startswith("cas-"):
                continue
            if not any(ch.isalpha() for ch in text):
                continue
            digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
            if digit_ratio > 0.6:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            cleaned.append(text)
        return cleaned


def estimate_oxidation_states(compound: dict) -> tuple[dict[str, int | None], bool]:
    elements = compound.get("elements", {})
    symbol_counts = {}
    for atomic_number, count in elements.items():
        symbol = get_symbol(int(atomic_number))
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
    summary = describe_bonding_and_polarity(compound)
    return summary.bonding_sentence, summary.polarity_sentence
