from __future__ import annotations

import math
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pint import UnitRegistry

from orbsim.colorbar_widget import HorizontalColorbarWidget
from orbsim.tabs.shared import resolve_cmap
from orbsim.ui.generated.ui_periodic_table import Ui_PeriodicTableTab
from orbsim.views.periodic_table_view import (
    BohrViewer,
    ElementTileButton,
    RotatedLabel,
    TableCanvas,
)
from periodic_table_cli.cli import load_data

ureg = UnitRegistry()
Q_ = ureg.Quantity


class PeriodicTableTab(QtWidgets.QWidget):
    """Interactive periodic table rendered in Qt (values cross-referenced with PubChem)."""

    FAMILY_COLORS = {
        "Noble Gas": "#3b82f6",
        "Alkali Metal": "#f97316",
        "Alkaline Earth Metal": "#fbbf24",
        "Transition Metal": "#10b981",
        "Post-Transition Metal": "#a3e635",
        "Metalloid": "#8b5cf6",
        "Nonmetal": "#e2e8f0",
        "Halogen": "#f43f5e",
        "Lanthanide": "#22d3ee",
        "Actinide": "#f59e0b",
    }
    _FAMILY_ALIASES = {
        "noble gas": "Noble Gas",
        "alkali metal": "Alkali Metal",
        "alkaline earth metal": "Alkaline Earth Metal",
        "transition metal": "Transition Metal",
        "post-transition metal": "Post-Transition Metal",
        "post transition metal": "Post-Transition Metal",
        "metalloid": "Metalloid",
        "nonmetal": "Nonmetal",
        "non-metal": "Nonmetal",
        "halogen": "Halogen",
        "lanthanide": "Lanthanide",
        "actinide": "Actinide",
    }
    FAMILY_SUMMARY = {
        "Noble Gas": "Closed-shell, monatomic gases; extremely low reactivity, colorless/odorless with very low boiling points.",
        "Alkali Metal": "Soft, highly reactive metals; form +1 cations and vigorous reactions with water; low ionization energy.",
        "Alkaline Earth Metal": "Reactive metals forming +2 cations; higher melting points than alkali metals and common oxides/carbonates.",
        "Transition Metal": "Variable oxidation states with partially filled d-subshells; often colored compounds and useful catalysts.",
        "Post-Transition Metal": "Softer, lower-melting metals with mixed metallic/covalent character; form diverse alloys and compounds.",
        "Metalloid": "Intermediate metallic/nonmetallic properties; semiconducting behavior common (e.g., Si, Ge).",
        "Nonmetal": "Generally poor electrical/thermal conductors; form covalent bonds and molecular compounds; wide range of states.",
        "Halogen": "Very reactive nonmetals forming -1 anions; strong oxidizers; exist as diatomic molecules at STP.",
        "Lanthanide": "Rare-earth metals with filling 4f subshell; typically +3 oxidation, magnetic/optical specialty uses.",
        "Actinide": "5f-block metals; radioactive, multiple oxidation states; many are synthetic with complex chemistry.",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PeriodicTableTab()
        self.ui.setupUi(self)
        self.font_point_size: int = 11
        self._theme_colors = {
            "bg": "#f5f7fb",
            "surface": "#f8fafc",
            "surfaceAlt": "#e2e8f0",
            "text": "#0f172a",
            "textMuted": "#475569",
            "border": "#cbd5e1",
            "focusRing": "#0ea5e9",
        }
        layout = self.ui.contentLayout
        credit = QtWidgets.QLabel(
            "Periodic table UI adapted from the Interactive Periodic Table project (codingwithnsh, GitHub).\n"
            "Element properties cross-referenced with PubChem (https://pubchem.ncbi.nlm.nih.gov/)."
        )
        credit.setWordWrap(True)
        layout.addWidget(credit)

        body = QtWidgets.QHBoxLayout()
        layout.addLayout(body, 1)

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left_splitter.setOpaqueResize(True)
        left_splitter.setChildrenCollapsible(False)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("Color by:"))
        self.scheme_combo = QtWidgets.QComboBox()
        self._scheme_options = [
            {"key": "family", "label": "Family", "type": "categorical", "field": None},
            {"key": "state", "label": "State at STP", "type": "categorical", "field": "standardState"},
            {"key": "density", "label": "Density", "type": "numeric", "field": "density"},
            {"key": "atomicMass", "label": "Atomic Mass", "type": "numeric", "field": "atomicMass"},
            {"key": "electronegativity", "label": "Electronegativity", "type": "numeric", "field": "electronegativity"},
            {"key": "atomicRadius", "label": "Atomic Radius", "type": "numeric", "field": "atomicRadius"},
            {"key": "meltingPoint", "label": "Melting Point", "type": "numeric", "field": "meltingPoint"},
            {"key": "boilingPoint", "label": "Boiling Point", "type": "numeric", "field": "boilingPoint"},
        ]
        for opt in self._scheme_options:
            self.scheme_combo.addItem(opt["label"], userData=opt)
        controls_row.addWidget(self.scheme_combo)

        controls_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.table_cmap_combo = QtWidgets.QComboBox()
        for cmap in self._available_colormaps():
            self.table_cmap_combo.addItem(cmap)
        self.table_cmap_combo.setCurrentText(self._scheme_default_cmap("density"))
        controls_row.addWidget(self.table_cmap_combo)
        controls_row.addWidget(QtWidgets.QLabel("Scale:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        self.scale_combo.setCurrentText("Linear")
        controls_row.addWidget(self.scale_combo)
        controls_row.addWidget(QtWidgets.QLabel("Temp unit:"))
        self.temp_unit_combo = QtWidgets.QComboBox()
        self.temp_unit_combo.addItems(["K", "°C", "°F"])
        self.temp_unit_combo.setCurrentText("K")
        controls_row.addWidget(self.temp_unit_combo)
        controls_row.addStretch()
        # container to hold legend, colorbar, table
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(controls_row)

        self.legend_row = QtWidgets.QHBoxLayout()
        self.legend_container = QtWidgets.QWidget()
        self.legend_container.setLayout(self.legend_row)
        self.legend_row.addWidget(QtWidgets.QLabel("Legend:"))
        self.legend_swatches: list[QtWidgets.QLabel] = []
        for family, color in self.FAMILY_COLORS.items():
            swatch = QtWidgets.QLabel("  ")
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid {self._theme_colors['border']};")
            self.legend_swatches.append(swatch)
            self.legend_row.addWidget(swatch)
            self.legend_row.addWidget(QtWidgets.QLabel(family))
        self.legend_row.addStretch()
        self.legend_scroll = QtWidgets.QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.legend_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.legend_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.legend_scroll.setWidget(self.legend_container)
        top_layout.addWidget(self.legend_scroll)

        self.colorbar_widget = HorizontalColorbarWidget()
        self.colorbar_widget.setVisible(False)
        top_layout.addWidget(self.colorbar_widget)

        self.grid_widget = TableCanvas(QtWidgets.QGridLayout(), self)
        self.grid_layout = self.grid_widget.layout()
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        # period label + grid in horizontal layout inside scroll area
        self.grid_row_widget = QtWidgets.QWidget()
        grid_row = QtWidgets.QHBoxLayout(self.grid_row_widget)
        grid_row.setContentsMargins(0, 0, 0, 0)
        self.period_label = RotatedLabel("Period", angle=-90)
        self.period_label.setAlignment(QtCore.Qt.AlignCenter)
        self.period_label.setContentsMargins(0, 12, 0, 12)
        grid_row.addWidget(self.period_label, 0)
        grid_row.addWidget(self.grid_widget, 1)
        self.grid_scroll_area = QtWidgets.QScrollArea()
        self.grid_scroll_area.setWidgetResizable(True)
        self.grid_scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.grid_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.grid_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.grid_scroll_area.setWidget(self.grid_row_widget)
        top_layout.addWidget(self.grid_scroll_area, 6)

        self.bohr_view = BohrViewer()
        self.bohr_view.setVisible(False)
        left_splitter.addWidget(top_container)
        left_splitter.setStretchFactor(0, 1)

        right_widget = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_widget)
        font_row = QtWidgets.QHBoxLayout()
        font_row.addWidget(QtWidgets.QLabel("Font size:"))
        self.font_size_combo = QtWidgets.QComboBox()
        for size in ["9", "10", "11", "12", "13"]:
            self.font_size_combo.addItem(size)
        self.font_size_combo.setCurrentText(str(self.font_point_size))
        self.font_size_combo.currentTextChanged.connect(self._on_font_change)
        font_row.addWidget(self.font_size_combo)
        font_row.addStretch()
        right.addLayout(font_row)
        self.info = QtWidgets.QTextBrowser()
        self.info.setOpenExternalLinks(True)
        self.info.setReadOnly(True)
        self.info.setMinimumWidth(360)
        self.info.anchorClicked.connect(self._handle_info_link)
        right.addWidget(self.info, 2)
        self.family_overview_box = QtWidgets.QLabel()
        self.family_overview_box.setWordWrap(True)
        self.family_overview_box.setObjectName("familyOverviewBox")
        right.addWidget(self.family_overview_box)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setOpaqueResize(True)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        body.addWidget(main_splitter)

        self.data = load_data()
        self.buttons: dict[int, QtWidgets.QPushButton] = {}
        self.current_element: dict | None = None
        self.current_oxidation_state: int = 0
        self.current_scheme: str = "family"
        self.current_table_cmap: str = self._scheme_default_cmap("density")
        self.scale_mode: str = "linear"
        self.field_units: dict[str, str | None] = {
            "density": "g/cm^3",
            "atomicMass": "u",
            "electronegativity": None,
            "atomicRadius": "pm",
            "meltingPoint": "K",
            "boilingPoint": "K",
        }
        self.temp_unit: str = "K"
        self.show_abbrev_config: bool = False
        self._build_grid()
        self._select_atomic_number(14)
        self.scheme_combo.currentIndexChanged.connect(self._on_scheme_change)
        self.table_cmap_combo.currentTextChanged.connect(self._on_table_cmap_change)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_change)
        self.temp_unit_combo.currentIndexChanged.connect(self._on_temp_unit_change)
        self._on_scheme_change()
        self.apply_theme({"colors": self._theme_colors, "meta": {"mode": "light"}})
        self._update_table_min_size()

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        self._theme_colors.update(
            {
                "bg": colors.get("bg", self._theme_colors["bg"]),
                "surface": colors.get("surface", self._theme_colors["surface"]),
                "surfaceAlt": colors.get("surfaceAlt", self._theme_colors["surfaceAlt"]),
                "text": colors.get("text", self._theme_colors["text"]),
                "textMuted": colors.get("textMuted", self._theme_colors["textMuted"]),
                "border": colors.get("border", self._theme_colors["border"]),
                "focusRing": colors.get("focusRing", self._theme_colors["focusRing"]),
            }
        )
        self.grid_widget.setStyleSheet(f"background: {self._theme_colors['surface']};")
        self.info.setStyleSheet(
            "font-size: 11pt; background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 8px;".format(
                bg=self._theme_colors["surface"],
                border=self._theme_colors["border"],
            )
        )
        self.family_overview_box.setStyleSheet(
            "font-size: 10pt; background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 8px; color: {text};".format(
                bg=self._theme_colors["surfaceAlt"],
                border=self._theme_colors["border"],
                text=self._theme_colors["text"],
            )
        )
        self.colorbar_widget.apply_theme(tokens)
        self.legend_container.setStyleSheet(f"color: {self._theme_colors['text']};")
        for swatch in self.legend_swatches:
            style = swatch.styleSheet()
            base = style.split("border:")[0]
            swatch.setStyleSheet(base + f"border: 1px solid {self._theme_colors['border']};")
        for btn in self.buttons.values():
            btn.set_theme(self._theme_colors, self.font_point_size)
        self.bohr_view.apply_theme(tokens)
        self.grid_widget.update()
        self._apply_coloring()
        self._update_table_min_size()

    def _elem_position(self, elem: dict) -> tuple[int, int]:
        an = elem["atomicNumber"]
        period = elem.get("period", 1)
        group = elem.get("group", None)
        if 57 <= an <= 71:
            return (9, an - 57 + 1)
        if 89 <= an <= 103:
            return (10, an - 89 + 1)
        if group is None:
            group = 18
        return (period, group)

    def _parse_oxidation_states(self, elem: dict) -> list[int]:
        raw = elem.get("oxidationStates") or elem.get("oxidationstates") or elem.get("oxidation_states")
        if raw is None:
            return []
        states: list[int] = []
        if isinstance(raw, (list, tuple)):
            for v in raw:
                try:
                    states.append(int(v))
                except Exception:
                    continue
        elif isinstance(raw, str):
            for part in raw.replace("−", "-").replace("+", " ").replace(",", " ").split():
                try:
                    states.append(int(part))
                except Exception:
                    continue
        return sorted(set(states))

    def _format_oxidation_states(self, states: list[int]) -> str:
        if not states:
            return "N/A"
        def fmt(v: int) -> str:
            if v > 0:
                return f"+{v}"
            return str(v)
        return ", ".join(fmt(s) for s in states)

    def _build_grid(self) -> None:
        # headers
        # group labels row
        group_label = QtWidgets.QLabel("Group")
        group_label.setAlignment(QtCore.Qt.AlignCenter)
        group_label.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
        self.grid_layout.addWidget(group_label, 0, 1, 1, 18)
        for col in range(1, 19):
            lbl = QtWidgets.QLabel(str(col))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
            self.grid_layout.addWidget(lbl, 1, col)
        # period labels column
        for row in range(1, 8):
            lbl = QtWidgets.QLabel(str(row))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
            self.grid_layout.addWidget(lbl, row + 1, 0)
        la_lbl = QtWidgets.QLabel("La")
        la_lbl.setAlignment(QtCore.Qt.AlignCenter)
        ac_lbl = QtWidgets.QLabel("Ac")
        ac_lbl.setAlignment(QtCore.Qt.AlignCenter)
        la_lbl.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
        ac_lbl.setStyleSheet(f"color: {self._theme_colors['textMuted']};")
        self.grid_layout.addWidget(la_lbl, 10, 0)
        self.grid_layout.addWidget(ac_lbl, 11, 0)

        for elem in self.data["elements"]:
            row, col = self._elem_position(elem)
            row += 1  # shift for header row
            btn = ElementTileButton()
            btn.set_theme(self._theme_colors, self.font_point_size)
            btn.set_tile_color(self._family_color(elem.get("family", "")), colors=self._theme_colors)
            btn.set_element(elem["symbol"], elem["atomicNumber"])
            btn.clicked.connect(lambda _, a=elem["atomicNumber"]: self._select_atomic_number(a))
            btn.setMinimumHeight(64)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            self.grid_layout.addWidget(btn, row, col)
            self.buttons[elem["atomicNumber"]] = btn
        # keep headers pinned
        self.grid_layout.setRowStretch(0, 0)
        for r in range(1, 8):
            self.grid_layout.setRowStretch(r, 1)
        # reduce gap before lanthanides/actinides
        self.grid_layout.setRowStretch(8, 0)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setRowStretch(10, 1)
        for c in range(0, 19):
            self.grid_layout.setColumnStretch(c, 0)
        self._apply_coloring()

    def _family_color(self, family: str) -> str:
        key = self._FAMILY_ALIASES.get(family.lower(), family) if isinstance(family, str) else family
        color = self.FAMILY_COLORS.get(key, "#94a3b8")
        return color

    def _update_table_min_size(self) -> None:
        font = QtGui.QFont(self.font())
        font.setPointSize(self.font_point_size)
        metrics = QtGui.QFontMetrics(font)
        text_width = max(metrics.horizontalAdvance("Mg"), metrics.horizontalAdvance("118"))
        cell_width = max(60, text_width + 16)
        cell_height = max(56, metrics.height() * 2 + 16)
        cols = max(self.grid_layout.columnCount(), 19)
        rows = max(self.grid_layout.rowCount(), 12)
        spacing = self.grid_layout.spacing()
        margins = self.grid_layout.contentsMargins()
        period_width = max(self.period_label.sizeHint().width(), 24)
        min_width = (
            cell_width * cols
            + spacing * max(cols - 1, 0)
            + margins.left()
            + margins.right()
            + period_width
        )
        min_height = (
            cell_height * rows
            + spacing * max(rows - 1, 0)
            + margins.top()
            + margins.bottom()
        )
        self.grid_row_widget.setMinimumSize(min_width, min_height)
        for btn in self.buttons.values():
            btn.setMinimumSize(cell_width, cell_height)

    def _format_numeric(self, val: float) -> str:
        try:
            v = float(val)
        except Exception:
            return "-"
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
            return f"{v:.3e}"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def _unit_html(self, unit: str | None) -> str:
        if not unit:
            return ""
        repl = {
            "g/cm^3": "g/cm<sup>-3</sup>",
            "degC": "&deg;C",
            "degF": "&deg;F",
        }
        return repl.get(unit, unit.replace("^", "<sup>") + "</sup>" if "^" in unit else unit)

    def _unit_display(self, unit: str | None) -> str:
        if not unit:
            return ""
        mapping = {"degC": "°C", "degF": "°F"}
        return mapping.get(unit, unit)

    def _button_label(self, elem: dict, extra: str | None = None) -> str:
        base_top = f"{elem['symbol']}"
        extra = "-" if extra in (None, "") else str(extra)
        return f"{base_top}\n{extra}"

    def _text_contrast(self, hex_color: str) -> str:
        try:
            hc = hex_color.lstrip("#")
            r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            dark = self._theme_colors.get("text", "#0f172a")
            light = self._theme_colors.get("bg", self._theme_colors.get("surface", "#f8fafc"))
            return dark if luminance > 0.65 else light
        except Exception:
            return self._theme_colors.get("text", "#0f172a")

    def _parse_quantity(self, raw, field: str) -> tuple[float, str | None]:
        """Return magnitude in preferred units (if available) and the unit label."""
        preferred = self.field_units.get(field)
        if field in ("meltingPoint", "boilingPoint"):
            preferred = {"K": "K", "°C": "degC", "°F": "degF"}.get(getattr(self, "temp_unit", "K"), preferred)
        if raw in (None, "", " ", "-"):
            return float("nan"), preferred
        try:
            if isinstance(raw, (int, float)):
                q = Q_(raw)
            else:
                q = Q_(str(raw))
            if preferred:
                try:
                    mag = q.to(preferred).magnitude
                    return float(mag), preferred
                except Exception:
                    pass
            return float(q.magnitude), str(q.units) if q.units != ureg.dimensionless else None
        except Exception:
            try:
                return float(raw), preferred
            except Exception:
                return float("nan"), preferred

    def _available_colormaps(self) -> list[str]:
        base = ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "coolwarm"]
        cram = ["batlow", "bamako", "devon", "oslo", "lajolla", "hawaii", "davos", "vik", "broc", "cork", "roma", "tokyo"]
        return base + cram

    def _get_cmap(self, name: str):
        return resolve_cmap(name)

    def _scheme_default_cmap(self, key: str) -> str:
        defaults = {
            "density": "batlow",
            "atomicMass": "lajolla",
            "electronegativity": "oslo",
            "atomicRadius": "vik",
            "meltingPoint": "davos",
            "boilingPoint": "davos",
        }
        return defaults.get(key, "viridis")

    def _on_scheme_change(self) -> None:
        meta = self.scheme_combo.currentData()
        if not meta:
            return
        self.current_scheme = meta["key"]
        if meta["type"] == "categorical":
            self.table_cmap_combo.setEnabled(False)
        else:
            self.table_cmap_combo.setEnabled(True)
            default = self._scheme_default_cmap(self.current_scheme)
            if default in [self.table_cmap_combo.itemText(i) for i in range(self.table_cmap_combo.count())]:
                self.table_cmap_combo.blockSignals(True)
                self.table_cmap_combo.setCurrentText(default)
                self.table_cmap_combo.blockSignals(False)
                self.current_table_cmap = default
        self._apply_coloring()

    def _on_table_cmap_change(self, cmap: str) -> None:
        self.current_table_cmap = cmap
        self._apply_coloring()

    def _on_temp_unit_change(self) -> None:
        self.temp_unit = self.temp_unit_combo.currentText()
        self._apply_coloring()
        self._refresh_info()

    def _on_scale_change(self) -> None:
        self.scale_mode = "log" if self.scale_combo.currentText().lower().startswith("log") else "linear"
        self._apply_coloring()

    def _on_font_change(self) -> None:
        try:
            self.font_point_size = int(self.font_size_combo.currentText())
        except Exception:
            self.font_point_size = 11
        self._apply_coloring()
        self._refresh_info()
        self._update_table_min_size()

    def _handle_info_link(self, url: QtCore.QUrl) -> None:
        if url.toString() == "toggle-config":
            self.show_abbrev_config = not self.show_abbrev_config
            self._refresh_info()
        else:
            QtGui.QDesktopServices.openUrl(url)

    def _apply_coloring(self) -> None:
        meta = self.scheme_combo.currentData()
        if not meta:
            return
        scheme = meta["key"]
        field = meta.get("field")

        # Toggle legend/colorbar visibility
        if scheme in ("family", "state"):
            self.legend_container.setVisible(True)
            self.colorbar_widget.setVisible(False)
        else:
            self.legend_container.setVisible(False)
            try:
                self._render_table_colorbar(field, scheme, meta["label"])
            except Exception as exc:
                print(f"Table colorbar render error: {exc}", file=sys.stderr)
                self.colorbar_widget.setVisible(False)

        # Precompute numeric range if needed
        vmin = vmax = None
        cmap = None
        to_hex = lambda rgba: f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        log_scale = self.scale_mode == "log"
        if scheme not in ("family", "state") and field:
            vals: list[float] = []
            for elem in self.data["elements"]:
                if field in elem and elem[field] not in ("", None):
                    mag, _ = self._parse_quantity(elem.get(field), field)
                    if not math.isnan(mag):
                        if log_scale and mag <= 0:
                            continue
                        vals.append(mag)
            if vals:
                vmin, vmax = min(vals), max(vals)
                if log_scale:
                    vmin = max(min(vals), sys.float_info.min)
                if math.isclose(vmin, vmax):
                    vmax = vmin * 1.1 if log_scale else vmin + 1.0
                cmap_name = self.current_table_cmap or self._scheme_default_cmap(scheme)
                cmap = self._get_cmap(cmap_name)

        state_colors = {
            "solid": "#f97316",
            "liquid": "#3b82f6",
            "gas": "#22c55e",
            "unknown": "#cbd5e1",
        }

        for an, btn in self.buttons.items():
            elem = next((e for e in self.data["elements"] if e["atomicNumber"] == an), None)
            if not elem:
                continue
            if scheme == "family":
                family = elem.get("family", "")
                bg = self.FAMILY_COLORS.get(self._FAMILY_ALIASES.get(family.lower(), family) if isinstance(family, str) else family, "#94a3b8")
                btn.set_tile_color(bg, colors=self._theme_colors)
                btn.set_element(elem["symbol"], elem["atomicNumber"])
                btn.set_extra_text("")
            elif scheme == "state":
                state = str(elem.get("standardState", "unknown")).lower()
                bg = state_colors.get(state, "#cbd5e1")
                btn.set_tile_color(bg, colors=self._theme_colors)
                btn.set_element(elem["symbol"], elem["atomicNumber"])
                btn.set_extra_text("")
            else:
                if cmap is None or vmin is None or vmax is None:
                    bg = "#cbd5e1"
                    extra = "-"
                else:
                    val, unit_used = self._parse_quantity(elem.get(field), field)
                    if math.isnan(val):
                        bg = "#cbd5e1"
                        extra = "-"
                    elif log_scale and val <= 0:
                        bg = "#cbd5e1"
                        extra = "-"
                    else:
                        if log_scale:
                            t = (math.log10(val) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
                        else:
                            t = (val - vmin) / (vmax - vmin)
                        rgba = cmap(t)
                        bg = to_hex(rgba)
                        extra = self._format_numeric(val)
                btn.set_tile_color(bg, colors=self._theme_colors)
                btn.set_element(elem["symbol"], elem["atomicNumber"])
                btn.set_extra_text(extra)

    def _render_table_colorbar(self, field: str, scheme_key: str, label: str) -> None:
        vals: list[float] = []
        unit_used: str | None = None
        log_scale = getattr(self, "scale_mode", "linear") == "log"
        for elem in self.data["elements"]:
            if field in elem and elem[field] not in ("", None):
                mag, u = self._parse_quantity(elem.get(field), field)
                if math.isnan(mag) or (log_scale and mag <= 0):
                    continue
                vals.append(mag)
                unit_used = unit_used or u
        if not vals:
            self.colorbar_widget.setVisible(False)
            return
        vmin, vmax = min(vals), max(vals)
        if log_scale:
            vmin = max(vmin, sys.float_info.min)
            if math.isclose(vmin, vmax):
                vmax = vmin * 1.1
        else:
            if math.isclose(vmin, vmax):
                vmax = vmin + 1.0
        unit_display = self._unit_display(unit_used)
        lbl = label if not unit_display else f"{label} [{unit_display}]"
        if log_scale:
            lbl = f"{lbl} (log)"
        cmap_name = self.current_table_cmap or self._scheme_default_cmap(scheme_key)
        self.colorbar_widget.set_data(cmap_name, vmin, vmax, lbl, "log" if log_scale else "linear")
        self.colorbar_widget.setVisible(True)
    def _select_atomic_number(self, atomic_number: int) -> None:
        elem = next((e for e in self.data["elements"] if e["atomicNumber"] == atomic_number), None)
        if not elem:
            return
        self.current_element = elem
        self.current_oxidation_state = 0
        for btn in self.buttons.values():
            btn.setChecked(False)
        if atomic_number in self.buttons:
            self.buttons[atomic_number].setChecked(True)
        self.bohr_view.set_element(elem)
        self.bohr_view.set_oxidation_state(self.current_oxidation_state)
        self._refresh_info()

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
                label = label_map.get(l, "?")
                parts.append(f"{n}{label}{cnt}")
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
        # subtract noble gas subshells
        _, ng_subshells = self.bohr_view._shell_counts({"atomicNumber": ng_electrons, "period": elem.get("period", 1)})
        remainder: dict[tuple[int, int], int] = {}
        for key, cnt in subshells.items():
            remainder[key] = max(0, cnt - ng_subshells.get(key, 0))
        remainder = {k: v for k, v in remainder.items() if v > 0}
        rest = self._config_string(remainder)
        return f"[{ng_symbol}] {rest}" if rest else f"[{ng_symbol}]"

    def _refresh_info(self) -> None:
        if not self.current_element:
            return
        elem = self.current_element
        shells, subshells = self.bohr_view._shell_counts(elem)
        if getattr(self, "show_abbrev_config", False):
            config = self._config_abbrev(elem, subshells)
        else:
            config = self._config_string(subshells)
        states = self._parse_oxidation_states(elem)
        formatted_states = self._format_oxidation_states(states)
        stp_note = "Values at STP (0 °C, 1 atm)."
        rows = []
        numeric_fields = {
            "atomicMass",
            "electronegativity",
            "atomicRadius",
            "ionizationEnergy",
            "electronAffinity",
            "meltingPoint",
            "boilingPoint",
            "density",
            "specificHeat",
        }
        for key, title in (
            ("symbol", "Symbol"),
            ("name", "Name"),
            ("atomicNumber", "Atomic Number"),
            ("family", "Family"),
            ("standardState", "State"),
            ("atomicMass", "Atomic Mass"),
            ("electronegativity", "Electronegativity"),
            ("atomicRadius", "Atomic Radius"),
            ("ionizationEnergy", "Ionization Energy"),
            ("electronAffinity", "Electron Affinity"),
            ("meltingPoint", "Melting Point"),
            ("boilingPoint", "Boiling Point"),
            ("density", "Density (at STP)"),
            ("specificHeat", "Specific Heat"),
            ("radioactive", "Radioactive"),
            ("occurrence", "Occurrence"),
            ("yearDiscovered", "Year Discovered"),
        ):
            if key in elem:
                val = elem[key]
                if key in numeric_fields:
                    mag, u = self._parse_quantity(val, key)
                    txt = "-" if math.isnan(mag) else self._format_numeric(mag)
                    if u:
                        txt = f"{txt} {self._unit_html(u)}"
                    rows.append(f"<tr><td><b>{title}</b></td><td>{txt}</td></tr>")
                else:
                    rows.append(f"<tr><td><b>{title}</b></td><td>{val}</td></tr>")
        rows.append(f"<tr><td><b>Oxidation States</b></td><td>{formatted_states}</td></tr>")
        rows.append(
            "<tr><td><b>Electron Config</b></td>"
            f"<td>{config} "
            "<a href='toggle-config'>(toggle)</a></td></tr>"
        )
        rows.append(f"<tr><td><b>STP</b></td><td>{stp_note}</td></tr>")
        if "name" in elem:
            pubchem_link = f"https://pubchem.ncbi.nlm.nih.gov/element/{elem['name']}"
            rows.append(f"<tr><td><b>PubChem</b></td><td><a href='{pubchem_link}'>{pubchem_link}</a></td></tr>")
        rows.append("<tr><td><b>Data Source</b></td><td><a href='https://pubchem.ncbi.nlm.nih.gov/'>PubChem</a></td></tr>")
        family_raw = elem.get("family", "")
        family_norm = self._FAMILY_ALIASES.get(str(family_raw).lower(), family_raw) if family_raw else ""
        fam_summary = self.FAMILY_SUMMARY.get(family_norm, "")
        table_html = (
            "<html><body>"
            "<table style='border-collapse: collapse; width: 100%;'>"
            + "".join(rows) +
            "</table>"
        )
        table_html += "</body></html>"
        self.info.setHtml(table_html)
        if fam_summary:
            self.family_overview_box.setText(f"<b>{family_norm} overview:</b> {fam_summary}")
            self.family_overview_box.setVisible(True)
        else:
            self.family_overview_box.clear()
            self.family_overview_box.setVisible(False)
