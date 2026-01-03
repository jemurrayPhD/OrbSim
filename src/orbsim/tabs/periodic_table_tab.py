from __future__ import annotations

import math
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pint import UnitRegistry

from orbsim.colorbar_widget import HorizontalColorbarWidget
from orbsim.tabs.shared import resolve_cmap
from orbsim.ui.generated.ui_periodic_table import Ui_PeriodicTableTab
from periodic_table_cli.cli import load_data

ureg = UnitRegistry()
Q_ = ureg.Quantity


class PeriodicTableTab(QtWidgets.QWidget):
    """Interactive periodic table rendered in Qt (values cross-referenced with PubChem)."""

    class TableCanvas(QtWidgets.QWidget):
        def __init__(self, layout: QtWidgets.QGridLayout, parent=None):
            super().__init__(parent)
            self.setLayout(layout)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
            self.line_color = "#9ca3af"

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            pen = QtGui.QPen(QtGui.QColor(self.line_color))
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            pen.setWidth(2)
            painter.setPen(pen)
            layout: QtWidgets.QGridLayout = self.layout()
            if not layout:
                return
            # Vertical dashed line after group 2 (between columns 2 and 3)
            def center_of_item(row, col):
                item = layout.itemAtPosition(row, col)
                if not item:
                    return None
                w = item.widget()
                if not w:
                    return None
                geo = w.geometry()
                return QtCore.QPointF(geo.center().x(), geo.center().y())

            # Use representative cells for positioning
            p_top = center_of_item(1, 2)
            p_bottom = center_of_item(7, 2)
            if p_top and p_bottom:
                x = (p_top.x() + center_of_item(1, 3).x()) / 2 if center_of_item(1, 3) else p_top.x() + 20
                painter.drawLine(QtCore.QPointF(x, p_top.y() - 20), QtCore.QPointF(x, p_bottom.y() + 20))
                # Connect to lanthanide/actinide rows
                p_la = center_of_item(9, 1)
                p_ac = center_of_item(10, 1)
                if p_la:
                    painter.drawLine(QtCore.QPointF(x, p_bottom.y() + 20), QtCore.QPointF(x, p_la.y()))
                if p_ac:
                    painter.drawLine(QtCore.QPointF(x, p_bottom.y() + 20), QtCore.QPointF(x, p_ac.y()))
            painter.end()

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

    class BohrViewer(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.elem: dict | None = None
            self.oxidation_state: int = 0
            self.setMinimumHeight(220)

        def set_element(self, elem: dict) -> None:
            self.elem = elem
            self.update()

        def set_oxidation_state(self, oxidation: int) -> None:
            self.oxidation_state = oxidation
            self.update()

        def _parse_oxidation_states_local(self, elem: dict) -> list[int]:
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
                cleaned = raw.replace("−", "-")
                for part in cleaned.replace("+", " ").replace(",", " ").split():
                    try:
                        states.append(int(part))
                    except Exception:
                        continue
            return sorted(set(states))

        def _oxidation_items_local(self, states: list[int]) -> list[tuple[str, int]]:
            items = [("Neutral (0)", 0)]
            for s in states:
                label = f"Ion +{s}" if s > 0 else f"Ion {s}"
                items.append((label, s))
            return items

        def _shell_counts(self, elem: dict) -> tuple[list[int], dict[tuple[int, int], int]]:
            base_e = elem.get("numberOfElectrons") or elem.get("atomicNumber") or 0
            total_e = max(0, int(base_e) - int(self.oxidation_state))
            period = max(int(elem.get("period", 1)), 1)
            subshell_order = [
                (1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6),
                (4, 0, 2), (3, 2, 10), (4, 1, 6), (5, 0, 2), (4, 2, 10),
                (5, 1, 6), (6, 0, 2), (4, 3, 14), (5, 2, 10), (6, 1, 6),
                (7, 0, 2), (5, 3, 14), (6, 2, 10), (7, 1, 6)
            ]
            subshells: dict[tuple[int, int], int] = {}
            remaining = int(total_e)
            for n, l, cap in subshell_order:
                if remaining <= 0:
                    break
                fill = min(cap, remaining)
                subshells[(n, l)] = fill
                remaining -= fill

            # Apply common stability exceptions
            adjustments: dict[int, dict[tuple[int, int], int]] = {
                24: {(4, 0): -1, (3, 2): 1},
                29: {(4, 0): -1, (3, 2): 1},
                42: {(5, 0): -1, (4, 2): 1},
                46: {(5, 0): -2, (4, 2): 2},
                47: {(5, 0): -1, (4, 2): 1},
                79: {(6, 0): -1, (5, 2): 1},
            }
            adj = adjustments.get(int(total_e))
            if adj:
                for (n, l), delta in adj.items():
                    subshells[(n, l)] = max(0, subshells.get((n, l), 0) + delta)

            shells = [0] * max(period, 1)
            for (n, _l), cnt in subshells.items():
                if 1 <= n <= len(shells):
                    shells[n - 1] += cnt
            # If no electrons assigned (shouldn't happen), fall back evenly
            if sum(shells) == 0 and total_e > 0:
                capacity = [2, 8, 18, 32, 32, 18, 8]
                remaining = total_e
                for idx in range(len(shells)):
                    cap = capacity[idx] if idx < len(capacity) else 2 * (idx + 1) * (idx + 1)
                    take = min(remaining, cap)
                    shells[idx] = take
                    remaining -= take
                    if remaining <= 0:
                        break
            return shells, subshells

        def _capacity(self, shell_index: int) -> int:
            n = shell_index + 1
            return 2 * n * n

        def _subshell_capacity(self, l: int) -> int:
            return {0: 2, 1: 6, 2: 10, 3: 14}.get(l, 0)

        def _angle_diff(self, a: float, b: float) -> float:
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

        def _generate_between_gaps(self, base: list[float], count: int) -> list[float]:
            """Place `count` angles into the largest gaps between existing angles."""
            if not base:
                return []
            existing = list(base)
            new_angles: list[float] = []
            max_iters = count * 8
            while len(new_angles) < count and max_iters > 0:
                max_iters -= 1
                sorted_base = sorted(existing)
                gaps: list[tuple[float, float]] = []
                for i, ang in enumerate(sorted_base):
                    nxt = sorted_base[(i + 1) % len(sorted_base)]
                    diff = (nxt - ang) % (2 * math.pi)
                    gaps.append((diff, ang))
                gaps.sort(reverse=True, key=lambda x: x[0])
                diff, start = gaps[0]
                mid = (start + diff / 2) % (2 * math.pi)
                if all(self._angle_diff(mid, a) > 1e-3 for a in existing):
                    new_angles.append(mid)
                    existing.append(mid)
                else:
                    # nudge slightly if midpoint is too close
                    mid = (mid + 1e-3) % (2 * math.pi)
                    new_angles.append(mid)
                    existing.append(mid)
            return new_angles[:count]

        def _angle_sets(self) -> dict[int, list[float]]:
            s_angles = [0.0, math.pi]
            p_angles = [
                math.radians(45), math.radians(90), math.radians(135),
                math.radians(-45) % (2 * math.pi), math.radians(-90) % (2 * math.pi), math.radians(-135) % (2 * math.pi),
            ]
            base_for_d = s_angles + p_angles
            d_angles = self._generate_between_gaps(base_for_d, self._subshell_capacity(2))
            base_for_f = base_for_d + d_angles
            f_angles = self._generate_between_gaps(base_for_f, self._subshell_capacity(3))
            return {0: s_angles, 1: p_angles, 2: d_angles, 3: f_angles}

        class SubshellMiniWidget(QtWidgets.QWidget):
            def __init__(self, label: str, filled: int, capacity: int, angles: list[float], is_full: bool, parent=None):
                super().__init__(parent)
                self.label = label
                self.filled = filled
                self.capacity = capacity
                self.is_full = is_full
                self.setMinimumSize(80, 80)

            def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                if not painter.isActive():
                    return
                try:
                    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                    w, h = self.width(), self.height()
                    center = QtCore.QPointF(w / 2, h / 2 + 6)
                    radius = min(w, h) * 0.32
                    dot_r = radius * 0.18

                    pen = QtGui.QPen(QtGui.QColor("#94a3b8"))
                    pen.setWidth(2 if not self.is_full else 3)
                    if self.is_full:
                        pen.setColor(QtGui.QColor("#22c55e"))
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(center, radius, radius)

                    cap = max(self.capacity, 1)
                    angles = [2 * math.pi * i / cap for i in range(cap)]
                    for idx, ang in enumerate(angles):
                        pos = QtCore.QPointF(
                            center.x() + radius * math.cos(ang),
                            center.y() + radius * math.sin(ang),
                        )
                        if idx < self.filled:
                            pen = QtGui.QPen(QtGui.QColor("#22c55e" if self.is_full else "#0f172a"))
                            pen.setWidthF(2 if self.is_full else 1.5)
                            painter.setPen(pen)
                            painter.setBrush(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                            painter.drawEllipse(pos, dot_r, dot_r)
                        else:
                            painter.setBrush(QtCore.Qt.NoBrush)
                            pen = QtGui.QPen(QtGui.QColor("#ef4444"))
                            pen.setWidthF(1.5)
                            painter.setPen(pen)
                            painter.drawEllipse(pos, dot_r, dot_r)

                    painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                    painter.drawText(QtCore.QPointF(6, 14), self.label)
                finally:
                    painter.end()
        def paintEvent(self, event):
            super().paintEvent(event)
            if not self.elem:
                return
            painter = QtGui.QPainter(self)
            if not painter.isActive():
                return
            try:
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                w, h = self.width(), self.height()
                center = QtCore.QPointF(w / 2, h / 2)
                shells, subshells = self._shell_counts(self.elem)
                max_shell = max(len(shells), 1)
                base_radius = min(w, h) * 0.4
                step = base_radius / max_shell
                dot_r = step * 0.06

                pen = QtGui.QPen(QtGui.QColor("#94a3b8"))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#1f2937")))
                painter.drawEllipse(center, step * 0.3, step * 0.3)

                filled_annotations: list[str] = []

                for shell_idx in range(len(shells)):
                    n = shell_idx + 1
                    l_values = [l for l in range(0, min(3, n - 1) + 1)]
                    lmax = len(l_values) - 1
                    radius_base = step * n
                    offset = step * 0.2
                    start_radius = radius_base - (lmax * offset) / 2 if lmax >= 0 else radius_base

                    for l in l_values:
                        radius = start_radius + l * offset
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 2))
                        painter.drawEllipse(center, radius, radius)
                        if l == lmax:
                            painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a")))
                            # place label just outside outermost subshell circle
                            painter.drawText(center + QtCore.QPointF(radius + 18, -radius - 6), f"n={n}")

                        cap = self._subshell_capacity(l)
                        cap = max(cap, 1)
                        angles = [2 * math.pi * k / cap for k in range(cap)]
                        filled = min(cap, subshells.get((n, l), 0))
                        full = filled >= cap and cap > 0
                        for idx, ang in enumerate(angles):
                            pos = QtCore.QPointF(
                                center.x() + radius * math.cos(ang),
                                center.y() + radius * math.sin(ang),
                            )
                            if idx < filled:
                                pen_color = "#22c55e" if full else "#0f172a"
                                pen = QtGui.QPen(QtGui.QColor(pen_color))
                                pen.setWidthF(2 if full else 1.5)
                                painter.setPen(pen)
                                painter.setBrush(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                                painter.drawEllipse(pos, dot_r, dot_r)
                            else:
                                painter.setBrush(QtCore.Qt.NoBrush)
                                pen = QtGui.QPen(QtGui.QColor("#ef4444"))
                                pen.setWidthF(1.5)
                                painter.setPen(pen)
                                painter.drawEllipse(pos, dot_r, dot_r)
                        if full:
                            filled_annotations.append(f"{n}{'spdf'[l]}")

                if filled_annotations:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#22c55e")))
                    painter.drawText(
                        QtCore.QPointF(10, h - 10),
                        "Filled subshells: " + ", ".join(filled_annotations),
                    )
            finally:
                painter.end()

        def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
            if not self.elem:
                return
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else self
            menu = QtWidgets.QMenu(parent)
            expand_action = menu.addAction("Expand subshells")
            expand_action.triggered.connect(self._show_subshell_dialog)
            box_action = menu.addAction("Orbital box view")
            box_action.triggered.connect(self._show_orbital_box_dialog)
            menu.exec(event.globalPos())

        def _show_subshell_dialog(self) -> None:
            if not self.elem:
                return
            shells, subshells = self._shell_counts(self.elem)
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else None
            dlg = QtWidgets.QDialog(parent)
            name = self.elem.get("name", "")
            sym = self.elem.get("symbol", "")
            dlg.setWindowTitle(f"Subshells for {name} ({sym})")
            layout = QtWidgets.QVBoxLayout(dlg)
            header = QtWidgets.QLabel("Electrons and vacancies by subshell")
            layout.addWidget(header)

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
            ox_combo = QtWidgets.QComboBox()
            states = self._parse_oxidation_states_local(self.elem)
            items = self._oxidation_items_local(states)
            for label, val in items:
                ox_combo.addItem(label, userData=val)
            # set current selection
            try:
                idx = [v for _, v in items].index(self.oxidation_state)
            except ValueError:
                idx = 0
            ox_combo.setCurrentIndex(idx)
            controls.addWidget(ox_combo)
            controls.addStretch()
            layout.addLayout(controls)

            grid = QtWidgets.QGridLayout()
            layout.addLayout(grid)
            l_labels = ["s", "p", "d", "f"]
            def refresh_grid():
                # clear old
                while grid.count():
                    item = grid.takeAt(0)
                    w = item.widget()
                    if w:
                        w.deleteLater()
                for col, label in enumerate(l_labels):
                    lbl = QtWidgets.QLabel(label)
                    lbl.setAlignment(QtCore.Qt.AlignCenter)
                    grid.addWidget(lbl, 0, col + 1)
                shells, subshells = self._shell_counts(self.elem)
                for n in range(1, len(shells) + 1):
                    row = n
                    row_label = QtWidgets.QLabel(f"n={n}")
                    row_label.setAlignment(QtCore.Qt.AlignCenter)
                    grid.addWidget(row_label, row, 0)
                    lmax = min(3, n - 1)
                    for l in range(0, lmax + 1):
                        cap = self._subshell_capacity(l)
                        filled = min(cap, subshells.get((n, l), 0))
                        angles = list(self._angle_sets().get(l, []))
                        view = self.SubshellMiniWidget(
                            f"{n}{l_labels[l]}",
                            filled,
                            cap,
                            angles,
                            filled >= cap,
                            dlg,
                        )
                        grid.addWidget(view, row, l + 1)

            def on_change():
                val = ox_combo.currentData()
                try:
                    self.set_oxidation_state(int(val))
                except Exception:
                    self.set_oxidation_state(0)
                refresh_grid()

            ox_combo.currentIndexChanged.connect(on_change)
            refresh_grid()

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
            dlg.resize(480, 360)
            dlg.setModal(True)
            dlg.exec()

        def _show_orbital_box_dialog(self) -> None:
            if not self.elem:
                return
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else None
            dlg = QtWidgets.QDialog(parent)
            name = self.elem.get("name", "")
            sym = self.elem.get("symbol", "")
            dlg.setWindowTitle(f"Orbital Box View – {name} ({sym})")
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.addWidget(QtWidgets.QLabel("Hund's rule filling; arrows show spin."))

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
            ox_combo = QtWidgets.QComboBox()
            states = self._parse_oxidation_states_local(self.elem)
            items = self._oxidation_items_local(states)
            for label, val in items:
                ox_combo.addItem(label, userData=val)
            try:
                idx = [v for _, v in items].index(self.oxidation_state)
            except ValueError:
                idx = 0
            ox_combo.setCurrentIndex(idx)
            controls.addWidget(ox_combo)
            controls.addStretch()
            layout.addLayout(controls)

            parent_tab = self.parent()
            view_cls = getattr(parent_tab, "OrbitalBoxView", PeriodicTableTab.OrbitalBoxView if "PeriodicTableTab" in globals() else None)
            if view_cls is None:
                return
            view = view_cls(self.elem, self.oxidation_state, self._shell_counts, self._subshell_capacity, dlg)
            self._orbital_box_dialog = dlg
            view.setMinimumSize(520, 420)
            layout.addWidget(view, 1)

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
            dlg.resize(600, 500)
            dlg.setModal(True)
            dlg.exec()

    class OrbitalBoxView(QtWidgets.QWidget):
        def __init__(self, elem: dict, oxidation: int, shell_fn, cap_fn, parent=None):
            super().__init__(parent)
            self.elem = elem
            self.oxidation = oxidation
            self.shell_fn = shell_fn
            self.cap_fn = cap_fn
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            self.setMinimumHeight(320)
            self._order = [
                (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
                (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
                (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
                (7, 0), (5, 3), (6, 2), (7, 1)
            ]

        def _degeneracy(self, l: int) -> int:
            return {0: 1, 1: 3, 2: 5, 3: 7}.get(l, 1)

        def _fill_orbitals(self, electrons: int, deg: int) -> list[list[str]]:
            # Hund: first fill singly (up), then pair with down
            orbitals = [["", ""] for _ in range(deg)]
            cap = 2 * deg
            electrons = max(0, min(electrons, cap))
            # first pass: one up in each orbital
            idx = 0
            while electrons > 0 and idx < deg:
                orbitals[idx][0] = "↑"
                electrons -= 1
                idx += 1
            # second pass: pair with down
            idx = 0
            while electrons > 0 and idx < deg:
                orbitals[idx][1] = "↓"
                electrons -= 1
                idx += 1
            return orbitals

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            if not painter.isActive():
                return
            try:
                try:
                    owner = self.shell_fn.__self__
                    prev = getattr(owner, "oxidation_state", 0)
                    owner.oxidation_state = self.oxidation
                    _, subshells = self.shell_fn(self.elem)
                    owner.oxidation_state = prev
                except Exception:
                    _, subshells = self.shell_fn(self.elem)
                painter.fillRect(self.rect(), QtGui.QColor("#1f2937"))
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

                # Base geometry
                base_left = 70.0
                base_top = 24.0
                base_col_w = 120.0
                base_box_w = 30.0
                base_box_h = 22.0
                base_spacing = 6.0
                label_map = {0: "s", 1: "p", 2: "d", 3: "f"}

                rows = [(n, l, subshells.get((n, l), 0)) for (n, l) in self._order if (n, l) in subshells]
                if not rows:
                    return

                # Horizontal scaling so widest subshell fits
                max_width_needed = 0.0
                for n, l, _e in rows:
                    deg = self._degeneracy(l)
                    width = base_left + l * base_col_w + deg * base_box_w + max(0, deg - 1) * base_spacing
                    max_width_needed = max(max_width_needed, width)
                avail_w = max(200.0, self.width() - 24.0)
                scale_x = min(1.0, avail_w / max_width_needed) if max_width_needed > 0 else 1.0

                left_margin = max(40.0, base_left * scale_x)
                top_margin = max(16.0, base_top * scale_x)
                margin = top_margin
                col_w = max(50.0, base_col_w * scale_x)
                box_w = max(12.0, base_box_w * scale_x)
                box_h = max(12.0, base_box_h * scale_x)
                spacing = max(3.0, base_spacing * scale_x)

                max_row = len(rows)
                avail_h = max(140.0, self.height() - 2 * margin)
                base_row_h = 70.0 * min(1.1, scale_x + 0.2)
                row_h = max(34.0, min(110.0, min(avail_h / max_row, base_row_h)))
                base_y = self.height() - margin - row_h

                arrow_x = left_margin / 2
                arrow_top = top_margin
                arrow_bottom = self.height() - margin
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 3))
                painter.drawLine(arrow_x, arrow_bottom, arrow_x, arrow_top + 14)
                head = QtGui.QPolygonF([
                    QtCore.QPointF(arrow_x, arrow_top),
                    QtCore.QPointF(arrow_x - 8, arrow_top + 14),
                    QtCore.QPointF(arrow_x + 8, arrow_top + 14),
                ])
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#e5e7eb")))
                painter.drawPolygon(head)
                painter.save()
                painter.translate(arrow_x - 16, (arrow_top + arrow_bottom) / 2)
                painter.rotate(-90)
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                painter.drawText(QtCore.QPointF(-30, 0), "Energy")
                painter.restore()

                for idx, (n, l, electrons) in enumerate(rows):
                    y = base_y - idx * row_h
                    x = left_margin + col_w * l
                    deg = self._degeneracy(l)
                    cap = self.cap_fn(l)
                    electrons = max(0, min(cap, electrons))
                    boxes = self._fill_orbitals(electrons, deg)

                    painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    label = f"{n}{label_map.get(l, '?')}"
                    painter.drawText(x, y - 6, label)

                    for j, (up, down) in enumerate(boxes):
                        bx = x + j * (box_w + spacing)
                        by = y
                        rect = QtCore.QRectF(bx, by, box_w, box_h)
                        painter.drawRect(rect)
                        painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                        if up:
                            left_half = QtCore.QRectF(rect.left() + 2, rect.top() + 2, rect.width() / 2 - 4, rect.height() - 4)
                            painter.drawText(left_half, QtCore.Qt.AlignCenter, up)
                        if down:
                            right_half = QtCore.QRectF(rect.left() + rect.width() / 2 + 2, rect.top() + 2, rect.width() / 2 - 4, rect.height() - 4)
                            painter.drawText(right_half, QtCore.Qt.AlignCenter, down)
            finally:
                painter.end()

    class RotatedLabel(QtWidgets.QLabel):
        def __init__(self, text: str, angle: float = -90.0, parent=None):
            super().__init__(text, parent)
            self.angle = angle

        def minimumSizeHint(self):
            s = super().minimumSizeHint()
            return QtCore.QSize(s.height(), s.width())

        def sizeHint(self):
            return self.minimumSizeHint()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.translate(self.width() / 2, self.height() / 2)
            painter.rotate(self.angle)
            painter.translate(-self.height() / 2, -self.width() / 2)
            painter.drawText(QtCore.QRectF(0, 0, self.height(), self.width()), QtCore.Qt.AlignCenter, self.text())
            painter.end()
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PeriodicTableTab()
        self.ui.setupUi(self)
        self.font_point_size: int = 11
        self._theme_colors = {
            "surface": "#f8fafc",
            "surfaceAlt": "#e2e8f0",
            "text": "#0f172a",
            "textMuted": "#475569",
            "border": "#cbd5e1",
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
        self.legend_row.addWidget(QtWidgets.QLabel("Legend (family colors shown when applicable):"))
        self.legend_swatches: list[QtWidgets.QLabel] = []
        for family, color in self.FAMILY_COLORS.items():
            swatch = QtWidgets.QLabel("  ")
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid {self._theme_colors['border']};")
            self.legend_swatches.append(swatch)
            self.legend_row.addWidget(swatch)
            self.legend_row.addWidget(QtWidgets.QLabel(family))
        self.legend_row.addStretch()
        top_layout.addWidget(self.legend_container)

        self.colorbar_widget = HorizontalColorbarWidget()
        self.colorbar_widget.setVisible(False)
        top_layout.addWidget(self.colorbar_widget)

        self.grid_widget = self.TableCanvas(QtWidgets.QGridLayout(), self)
        self.grid_layout = self.grid_widget.layout()
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        # period label + grid in horizontal layout inside scroll area
        self.grid_row_widget = QtWidgets.QWidget()
        grid_row = QtWidgets.QHBoxLayout(self.grid_row_widget)
        grid_row.setContentsMargins(0, 0, 0, 0)
        self.period_label = self.RotatedLabel("Period", angle=-90)
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

        self.bohr_view = self.BohrViewer()
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
                "surface": colors.get("surface", self._theme_colors["surface"]),
                "surfaceAlt": colors.get("surfaceAlt", self._theme_colors["surfaceAlt"]),
                "text": colors.get("text", self._theme_colors["text"]),
                "textMuted": colors.get("textMuted", self._theme_colors["textMuted"]),
                "border": colors.get("border", self._theme_colors["border"]),
            }
        )
        self.grid_widget.line_color = self._theme_colors["border"]
        self.info.setStyleSheet(
            "font-size: 11pt; background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 8px;".format(
                bg=self._theme_colors["surface"],
                border=self._theme_colors["border"],
            )
        )
        self.colorbar_widget.apply_theme(tokens)
        for swatch in self.legend_swatches:
            style = swatch.styleSheet()
            base = style.split("border:")[0]
            swatch.setStyleSheet(base + f"border: 1px solid {self._theme_colors['border']};")
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
        self.grid_layout.addWidget(group_label, 0, 1, 1, 18)
        for col in range(1, 19):
            lbl = QtWidgets.QLabel(str(col))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, 1, col)
        # period labels column
        for row in range(1, 8):
            lbl = QtWidgets.QLabel(str(row))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, row + 1, 0)
        la_lbl = QtWidgets.QLabel("La")
        la_lbl.setAlignment(QtCore.Qt.AlignCenter)
        ac_lbl = QtWidgets.QLabel("Ac")
        ac_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(la_lbl, 10, 0)
        self.grid_layout.addWidget(ac_lbl, 11, 0)

        for elem in self.data["elements"]:
            row, col = self._elem_position(elem)
            row += 1  # shift for header row
            btn = QtWidgets.QPushButton(f"{elem['symbol']}\n{elem['atomicNumber']}")
            btn.setCheckable(True)
            btn.setStyleSheet(self._style_for_family(elem.get("family", "")))
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

    def _style_for_family(self, family: str) -> str:
        key = self._FAMILY_ALIASES.get(family.lower(), family) if isinstance(family, str) else family
        color = self.FAMILY_COLORS.get(key, "#94a3b8")
        return self._button_style(color, self._text_contrast(color))

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
            return "#0f172a" if luminance > 0.65 else "#f8fafc"
        except Exception:
            return "#0f172a"

    def _button_style(self, bg: str, text: str) -> str:
        return (
            f"background-color: {bg}; color: {text}; font-weight: bold; font-size: {self.font_point_size}pt; "
            f"border: 1px solid {self._theme_colors.get('border', '#0f172a')}; padding: 6px; border-radius: 4px;"
        )

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
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(f"{elem['symbol']}\n{elem['atomicNumber']}")
            elif scheme == "state":
                state = str(elem.get("standardState", "unknown")).lower()
                bg = state_colors.get(state, "#cbd5e1")
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(self._button_label(elem, elem.get("standardState", "unknown")))
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
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(self._button_label(elem, extra))

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
        if fam_summary:
            table_html += (
                "<div style='margin-top:10px; padding:8px; border:1px solid #cbd5e1; border-radius:6px; background:#f8fafc;'>"
                f"<b>{family_norm} overview:</b> {fam_summary}"
                "</div>"
            )
        table_html += "</body></html>"
        self.info.setHtml(table_html)
