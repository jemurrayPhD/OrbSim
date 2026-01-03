from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets


def _contrast_text(base: QtGui.QColor, light: QtGui.QColor, dark: QtGui.QColor) -> QtGui.QColor:
    luminance = (0.299 * base.red() + 0.587 * base.green() + 0.114 * base.blue()) / 255
    return dark if luminance > 0.65 else light


class TableCanvas(QtWidgets.QWidget):
    def __init__(self, layout: QtWidgets.QGridLayout, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLayout(layout)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)


class RotatedLabel(QtWidgets.QLabel):
    def __init__(self, text: str, angle: int = 0, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._angle = angle

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        painter.translate(-self.width() / 2, -self.height() / 2)
        painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class ElementTileButton(QtWidgets.QAbstractButton):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self._symbol = ""
        self._atomic_number = 0
        self.setCheckable(True)
        self._base_color = QtGui.QColor("#94a3b8")
        self._text_color = QtGui.QColor("#0f172a")
        self._border_color = QtGui.QColor("#cbd5e1")
        self._focus_color = QtGui.QColor("#0ea5e9")
        self._font_point_size = 10
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_theme(self, colors: dict, font_point_size: int) -> None:
        self._border_color = QtGui.QColor(colors.get("border", "#cbd5e1"))
        self._focus_color = QtGui.QColor(colors.get("focusRing", colors.get("accent", "#0ea5e9")))
        self._font_point_size = font_point_size
        self.update()

    def set_tile_color(self, base_hex: str, text_hex: str | None = None, colors: dict | None = None) -> None:
        self._base_color = QtGui.QColor(base_hex)
        if text_hex is not None:
            self._text_color = QtGui.QColor(text_hex)
        elif colors:
            light = QtGui.QColor(colors.get("bg", "#f8fafc"))
            dark = QtGui.QColor(colors.get("text", "#0f172a"))
            self._text_color = _contrast_text(self._base_color, light, dark)
        self.update()

    def set_element(self, symbol: str, atomic_number: int) -> None:
        self._symbol = symbol
        self._atomic_number = atomic_number
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 5

        base = QtGui.QColor(self._base_color)
        highlight = QtGui.QColor(base).lighter(112)
        shadow = QtGui.QColor(base).darker(112)

        painter.setPen(QtGui.QPen(self._border_color, 1))
        painter.setBrush(QtGui.QBrush(base))
        painter.drawRoundedRect(rect, radius, radius)

        # chamfer highlight (top-left)
        chamfer_rect = rect.adjusted(1, 1, -1, -1)
        painter.setPen(QtGui.QPen(highlight, 1))
        painter.drawLine(chamfer_rect.topLeft(), chamfer_rect.topRight())
        painter.drawLine(chamfer_rect.topLeft(), chamfer_rect.bottomLeft())
        # chamfer shadow (bottom-right)
        painter.setPen(QtGui.QPen(shadow, 1))
        painter.drawLine(chamfer_rect.bottomLeft(), chamfer_rect.bottomRight())
        painter.drawLine(chamfer_rect.topRight(), chamfer_rect.bottomRight())

        if self.hasFocus() or self.isChecked():
            painter.setPen(QtGui.QPen(self._focus_color, 2))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(self._font_point_size + 2)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(self._text_color))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._symbol or self.text())

        if self._atomic_number:
            small_font = painter.font()
            small_font.setBold(False)
            small_font.setPointSize(max(self._font_point_size - 2, 7))
            painter.setFont(small_font)
            painter.setPen(QtGui.QPen(self._text_color))
            painter.drawText(
                rect.adjusted(6, 4, -6, -4),
                QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft,
                str(self._atomic_number),
            )
        painter.end()


class SubshellMiniWidget(QtWidgets.QWidget):
    def __init__(self, label: str, filled: int, capacity: int, angles: list[float], is_full: bool, parent=None):
        super().__init__(parent)
        self.label = label
        self.filled = filled
        self.capacity = capacity
        self.is_full = is_full
        self._theme_colors = {
            "accent": "#2563eb",
            "border": "#cbd5e1",
            "surface": "#ffffff",
            "text": "#0f172a",
            "textMuted": "#4b5563",
        }
        self._semantic_colors = {
            "electron": QtGui.QColor("#3b82f6"),
            "filled": QtGui.QColor("#22c55e"),
            "vacancy": QtGui.QColor("#ef4444"),
        }
        self.setMinimumSize(80, 80)

    def apply_theme(self, colors: dict) -> None:
        self._theme_colors.update(colors)
        mode = colors.get("mode") if isinstance(colors, dict) else None
        if mode == "dark" or mode == "high_contrast":
            self._semantic_colors["electron"] = QtGui.QColor("#60a5fa")
            self._semantic_colors["filled"] = QtGui.QColor("#4ade80")
            self._semantic_colors["vacancy"] = QtGui.QColor("#f87171")
        self.update()

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

            border = QtGui.QColor(self._theme_colors["border"])
            electron = self._semantic_colors["electron"]
            filled = self._semantic_colors["filled"]
            vacancy = self._semantic_colors["vacancy"]
            text = QtGui.QColor(self._theme_colors["text"])
            muted = QtGui.QColor(self._theme_colors["textMuted"])

            pen = QtGui.QPen(border)
            pen.setWidth(2 if not self.is_full else 3)
            if self.is_full:
                pen.setColor(filled)
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
                    pen = QtGui.QPen(filled if self.is_full else electron)
                    pen.setWidthF(2 if self.is_full else 1.5)
                    painter.setPen(pen)
                    painter.setBrush(QtGui.QBrush(electron))
                    painter.drawEllipse(pos, dot_r, dot_r)
                else:
                    painter.setBrush(QtCore.Qt.NoBrush)
                    pen = QtGui.QPen(vacancy)
                    pen.setWidthF(1.5)
                    painter.setPen(pen)
                    painter.drawEllipse(pos, dot_r, dot_r)

            painter.setPen(QtGui.QPen(text))
            painter.drawText(QtCore.QPointF(6, 14), self.label)
        finally:
            painter.end()


class BohrViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.elem: dict | None = None
        self.oxidation_state: int = 0
        self._theme_colors = {
            "accent": "#2563eb",
            "border": "#cbd5e1",
            "surface": "#ffffff",
            "surfaceAlt": "#eef2f7",
            "text": "#0f172a",
            "textMuted": "#4b5563",
        }
        self._semantic_colors = {
            "electron": QtGui.QColor("#3b82f6"),
            "filled": QtGui.QColor("#22c55e"),
            "vacancy": QtGui.QColor("#ef4444"),
        }
        self._semantic_colors = {
            "electron": QtGui.QColor("#3b82f6"),
            "filled": QtGui.QColor("#22c55e"),
            "vacancy": QtGui.QColor("#ef4444"),
        }
        self.setMinimumHeight(220)

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        self._theme_colors.update(colors)
        mode = tokens.get("meta", {}).get("mode")
        if mode in ("dark", "high_contrast"):
            self._semantic_colors["electron"] = QtGui.QColor("#60a5fa")
            self._semantic_colors["filled"] = QtGui.QColor("#4ade80")
            self._semantic_colors["vacancy"] = QtGui.QColor("#f87171")
        self.update()

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

            border = QtGui.QColor(self._theme_colors["border"])
            electron = self._semantic_colors["electron"]
            filled = self._semantic_colors["filled"]
            vacancy = self._semantic_colors["vacancy"]
            text = QtGui.QColor(self._theme_colors["text"])
            muted = QtGui.QColor(self._theme_colors["textMuted"])
            surface_alt = QtGui.QColor(self._theme_colors["surfaceAlt"])

            pen = QtGui.QPen(border)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(surface_alt))
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
                    painter.setPen(QtGui.QPen(border, 2))
                    painter.drawEllipse(center, radius, radius)
                    if l == lmax:
                        painter.setPen(QtGui.QPen(text))
                        label_offset = radius * 0.12
                        painter.drawText(center + QtCore.QPointF(radius + label_offset, -label_offset), f"n={n}")

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
                            pen_color = filled if full else electron
                            pen = QtGui.QPen(pen_color)
                            pen.setWidthF(2 if full else 1.5)
                            painter.setPen(pen)
                            painter.setBrush(QtGui.QBrush(electron))
                            painter.drawEllipse(pos, dot_r, dot_r)
                        else:
                            painter.setBrush(QtCore.Qt.NoBrush)
                            pen = QtGui.QPen(vacancy)
                            pen.setWidthF(1.5)
                            painter.setPen(pen)
                            painter.drawEllipse(pos, dot_r, dot_r)
                    if full:
                        filled_annotations.append(f"{n}{'spdf'[l]}")

            if filled_annotations:
                painter.setPen(QtGui.QPen(filled))
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
                    view = SubshellMiniWidget(
                        f"{n}{l_labels[l]}",
                        filled,
                        cap,
                        angles,
                        filled >= cap,
                        dlg,
                    )
                    view.apply_theme(self._theme_colors)
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

        view = OrbitalBoxView(self.elem, self.oxidation_state, self._shell_counts, self._subshell_capacity, dlg)
        view.apply_theme(self._theme_colors)
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
        self._theme_colors = {
            "accent": "#2563eb",
            "border": "#cbd5e1",
            "surface": "#ffffff",
            "surfaceAlt": "#eef2f7",
            "text": "#0f172a",
            "textMuted": "#4b5563",
        }
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(320)
        self._order = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
            (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
            (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
            (7, 0), (5, 3), (6, 2), (7, 1)
        ]

    def apply_theme(self, colors: dict) -> None:
        self._theme_colors.update(colors)
        mode = colors.get("mode") if isinstance(colors, dict) else None
        if mode == "dark" or mode == "high_contrast":
            self._semantic_colors["electron"] = QtGui.QColor("#60a5fa")
            self._semantic_colors["filled"] = QtGui.QColor("#4ade80")
            self._semantic_colors["vacancy"] = QtGui.QColor("#f87171")
        self.update()

    def _degeneracy(self, l: int) -> int:
        return {0: 1, 1: 3, 2: 5, 3: 7}.get(l, 1)

    def _fill_orbitals(self, electrons: int, deg: int) -> list[list[str]]:
        orbitals = [["", ""] for _ in range(deg)]
        cap = 2 * deg
        electrons = max(0, min(electrons, cap))
        idx = 0
        while electrons > 0 and idx < deg:
            orbitals[idx][0] = "↑"
            electrons -= 1
            idx += 1
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
            painter.fillRect(self.rect(), QtGui.QColor(self._theme_colors["surface"]))
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            base_left = 70.0
            base_top = 24.0
            base_col_w = 120.0
            base_box_w = 34.0
            base_box_h = 24.0
            base_spacing = 7.0
            label_map = {0: "s", 1: "p", 2: "d", 3: "f"}

            rows = [(n, l, subshells.get((n, l), 0)) for (n, l) in self._order if (n, l) in subshells]
            if not rows:
                return

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
            box_w = max(14.0, base_box_w * scale_x)
            box_h = max(14.0, base_box_h * scale_x)
            spacing = max(4.0, base_spacing * scale_x)

            max_row = len(rows)
            avail_h = max(140.0, self.height() - 2 * margin)
            base_row_h = 70.0 * min(1.1, scale_x + 0.2)
            row_h = max(36.0, min(120.0, min(avail_h / max_row, base_row_h)))
            base_y = self.height() - margin - row_h

            electron = self._semantic_colors["electron"]
            filled = self._semantic_colors["filled"]
            vacancy = self._semantic_colors["vacancy"]
            text = QtGui.QColor(self._theme_colors["text"])
            border = QtGui.QColor(self._theme_colors["border"])

            arrow_x = left_margin / 2
            arrow_top = top_margin
            arrow_bottom = self.height() - margin
            painter.setPen(QtGui.QPen(electron, 3.5))
            painter.drawLine(arrow_x, arrow_bottom, arrow_x, arrow_top + 16)
            head = QtGui.QPolygonF([
                QtCore.QPointF(arrow_x, arrow_top),
                QtCore.QPointF(arrow_x - 9, arrow_top + 16),
                QtCore.QPointF(arrow_x + 9, arrow_top + 16),
            ])
            painter.setBrush(QtGui.QBrush(electron))
            painter.drawPolygon(head)
            painter.save()
            painter.translate(arrow_x - 16, (arrow_top + arrow_bottom) / 2)
            painter.rotate(-90)
            painter.setPen(QtGui.QPen(electron))
            painter.drawText(QtCore.QPointF(-30, 0), "Energy")
            painter.restore()

            for idx, (n, l, electrons) in enumerate(rows):
                y = base_y - idx * row_h
                x = left_margin + col_w * l
                deg = self._degeneracy(l)
                cap = self.cap_fn(l)
                electrons = max(0, min(cap, electrons))
                boxes = self._fill_orbitals(electrons, deg)

                    painter.setPen(QtGui.QPen(text, 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    label = f"{n}{label_map.get(l, '?')}"
                    painter.drawText(x, y - 6, label)

                for j, (up, down) in enumerate(boxes):
                    bx = x + j * (box_w + spacing)
                    by = y
                    rect = QtCore.QRectF(bx, by, box_w, box_h)
                    if electrons == 0:
                        painter.setPen(QtGui.QPen(vacancy, 2))
                    elif electrons >= cap:
                        painter.setPen(QtGui.QPen(filled, 2))
                    else:
                        painter.setPen(QtGui.QPen(border, 2))
                    painter.drawRect(rect)
                    if up:
                        painter.setPen(QtGui.QPen(electron, 2))
                        painter.drawText(rect.center() + QtCore.QPointF(-4, 6), up)
                    if down:
                        painter.setPen(QtGui.QPen(electron, 2))
                        painter.drawText(rect.center() + QtCore.QPointF(-4, 6), down)
        finally:
            painter.end()
