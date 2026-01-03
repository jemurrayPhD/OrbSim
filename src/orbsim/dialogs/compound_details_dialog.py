from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets

from orbsim.chem.elements import get_atomic_number, get_element, get_symbol


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
PUBCHEM_CITATION_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines"


class CompoundDetailsDialog(QtWidgets.QDialog):
    def __init__(self, compound: dict, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.compound = compound
        self._network = QtNetwork.QNetworkAccessManager(self)
        self._network.finished.connect(self._on_image_reply)
        self._pending_image_reply: QtNetwork.QNetworkReply | None = None
        self._image_cache_path = self._image_cache_dir() / f"{compound['cid']}.png"

        self.setWindowTitle(f"{compound['name']} — {compound['formula']}")
        self.setMinimumSize(640, 520)
        self._build_ui()
        self._populate()
        self._load_image()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setWordWrap(True)
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

        self.source_label = QtWidgets.QLabel("")
        self.source_label.setOpenExternalLinks(True)
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def _populate(self) -> None:
        self.title_label.setText(f"{self.compound['name']} ({self.compound['formula']})")

        oxidation_states, heuristic = estimate_oxidation_states(self.compound)
        if heuristic:
            self.oxidation_label.setText("Estimated oxidation states (heuristic)")
        else:
            self.oxidation_label.setText("Oxidation states")
        self._populate_oxidation_table(oxidation_states)

        bonding_text, polarity_text = classify_bonding_and_polarity(self.compound)
        self.bonding_label.setText(f"Bonding: {bonding_text}. Polarity: {polarity_text}.")

        properties = [
            ("Molecular weight", self.compound.get("mol_weight")),
            ("IUPAC name", self.compound.get("iupac_name")),
        ]
        synonyms = self.compound.get("synonyms") or []
        if isinstance(synonyms, list) and synonyms:
            properties.append(("Synonyms", ", ".join(synonyms[:5])))
        prop_lines = [f"{label}: {value}" for label, value in properties if value]
        self.properties_label.setText("Known data: " + ("; ".join(prop_lines) if prop_lines else "Unavailable"))

        self.source_label.setText(
            "Data source: PubChem (NIH/NLM). "
            f"<a href=\"{PUBCHEM_CITATION_URL}\">Citation guidelines</a>."
        )

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
    elements = compound.get("elements", {})
    symbols = [get_symbol(int(z)) for z in elements.keys()]
    symbols = [s for s in symbols if s]
    if not symbols:
        return "Unknown", "Unknown"
    has_metal = any(_is_metal_symbol(symbol) for symbol in symbols)
    has_nonmetal = any(_is_nonmetal_symbol(symbol) for symbol in symbols)

    if has_metal and has_nonmetal:
        bonding = "Ionic (heuristic)"
    elif has_metal and not has_nonmetal:
        bonding = "Metallic (heuristic)"
    else:
        bonding = "Covalent (heuristic)"

    en_values = [value for symbol in symbols if (value := _electronegativity_symbol(symbol)) is not None]
    if not en_values:
        polarity = "Unknown"
    else:
        max_en = max(en_values)
        min_en = min(en_values)
        diff = max_en - min_en
        if diff >= 1.7:
            polarity = "Polar (high ΔEN)"
        elif diff >= 0.4:
            polarity = "Polar (moderate ΔEN)"
        else:
            polarity = "Nonpolar (low ΔEN)"
    return bonding, polarity
