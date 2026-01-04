from __future__ import annotations

import json

import qtawesome as qta
from PySide6 import QtCore, QtGui, QtPrintSupport, QtWidgets

from orbsim.theming.apply_theme import apply_theme as apply_theme_tokens
from orbsim.theming.theme_manager import apply_skin, get_theme_manager
from orbsim.ui.generated.ui_main_window import Ui_MainWindow
from orbsim.tabs.atomic_orbitals_tab import AtomicOrbitalsTab
from orbsim.tabs.bonding_orbitals_tab import BondingOrbitalsTab
from orbsim.tabs.compound_builder_tab import CompoundBuilderTab
from orbsim.tabs.electron_shells_tab import ElectronShellsTab
from orbsim.tabs.periodic_table_tab import PeriodicTableTab
from orbsim.views.annotation_editor import AnnotationEditorWindow
from orbsim.dialogs.compound_database_dialog import CompoundDatabaseDialog
from orbsim.widgets import DropPlotter, PeriodicTableWidget


class OrbSimMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OrbSim")
        self.setMinimumSize(1280, 800)
        self._settings = QtCore.QSettings("OrbSim", "OrbSim")
        self._theme_name = self._settings.value("theme", "Fluent Light")
        self._theme_manager = get_theme_manager()
        self._theme_manager.theme_changed.connect(self._on_theme_changed)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._build_toolbar()
        self._build_menus()

        self.tabs = self.ui.tabWidget
        # Order: Periodic Table, Electron Shells, Atomic Orbitals, Bonding Orbitals, Compound Builder
        self.tabs.addTab(PeriodicTableTab(), "Periodic Table")
        self.tabs.addTab(ElectronShellsTab(), "Electron Shells")
        self.tabs.addTab(AtomicOrbitalsTab(), "Atomic Orbitals")
        self.tabs.addTab(BondingOrbitalsTab(), "Bonding Orbitals")
        self.tabs.addTab(CompoundBuilderTab(), "Compound Builder")
        self._annotation_editors: list[AnnotationEditorWindow] = []
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.statusBar().showMessage("Drag elements into the visualization pane to begin.")
        self.apply_theme(self._theme_name)

    def closeEvent(self, event) -> None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            cleanup = getattr(tab, "cleanup", None)
            if callable(cleanup):
                cleanup()
        super().closeEvent(event)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Export/Copy")
        toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        print_act = QtGui.QAction("Print", self)
        print_act.setIcon(qta.icon("fa5s.print"))
        print_act.triggered.connect(self._print_current_view)
        export_act = QtGui.QAction("Export...", self)
        export_act.setIcon(qta.icon("fa5s.file-export"))
        export_act.triggered.connect(self._export_current_view)
        copy_act = QtGui.QAction("Copy to clipboard", self)
        copy_act.setIcon(qta.icon("fa5s.copy"))
        copy_act.triggered.connect(self._copy_current_view)

        export_btn = QtWidgets.QToolButton()
        export_btn.setText("Export/Copy")
        export_menu = QtWidgets.QMenu(export_btn)
        export_menu.addAction(print_act)
        export_menu.addAction(export_act)
        export_menu.addAction(copy_act)
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(export_btn)


    def _build_menus(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QtGui.QActionGroup(self)
        theme_group.setExclusive(True)
        for name in self._theme_manager.available_themes():
            action = QtGui.QAction(name, self)
            action.setCheckable(True)
            action.setChecked(name == self._theme_name)
            action.triggered.connect(lambda checked, n=name: self.apply_theme(n))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        settings_menu = self.menuBar().addMenu("Settings")
        compound_action = QtGui.QAction("Compound Database…", self)
        compound_action.triggered.connect(self._open_compound_db_dialog)
        settings_menu.addAction(compound_action)

    def _open_compound_db_dialog(self) -> None:
        dialog = CompoundDatabaseDialog(self)
        dialog.exec()

    def apply_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        self._settings.setValue("theme", theme_name)
        self._theme_manager.set_theme(theme_name)

    def _on_theme_changed(self, tokens: dict) -> None:
        app = QtWidgets.QApplication.instance()
        if app:
            apply_theme_tokens(app, tokens)
            apply_skin(app, tokens)
        for tab_index in range(self.tabs.count()):
            tab = self.tabs.widget(tab_index)
            apply = getattr(tab, "apply_theme", None)
            if callable(apply):
                apply(tokens)
            else:
                for widget in tab.findChildren(DropPlotter):
                    widget.apply_theme(tokens)
                for widget in tab.findChildren(PeriodicTableWidget):
                    widget.apply_theme(tokens)

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def _grab_current_view(self) -> QtGui.QPixmap:
        target: QtWidgets.QWidget = self.tabs.currentWidget() if getattr(self, "tabs", None) else self
        return target.grab()

    def _copy_current_view(self) -> None:
        decision = self._prompt_annotation_action()
        if decision == "annotate":
            self._open_annotation_editor()
            return
        if decision == "cancel":
            return
        pixmap = self._grab_current_view()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)

    def _print_current_view(self) -> None:
        pixmap = self._grab_current_view()
        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
        dialog = QtPrintSupport.QPrintDialog(printer, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            painter = QtGui.QPainter(printer)
            rect = painter.viewport()
            scaled = pixmap.scaled(rect.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.setViewport(rect.x(), rect.y(), scaled.width(), scaled.height())
            painter.setWindow(pixmap.rect())
            painter.drawPixmap(0, 0, scaled)
            painter.end()

    def _export_current_view(self) -> None:
        decision = self._prompt_annotation_action()
        if decision == "annotate":
            self._open_annotation_editor()
            return
        if decision == "cancel":
            return
        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf)"
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Export current view", "", filters)
        if not path:
            return
        fmt = "png"
        if selected.startswith("JPEG"):
            fmt = "jpg"
        elif selected.startswith("PDF") or path.lower().endswith(".pdf"):
            fmt = "pdf"
        elif path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
            fmt = "jpg"
        elif path.lower().endswith(".png"):
            fmt = "png"
        if fmt == "pdf" and not path.lower().endswith(".pdf"):
            path += ".pdf"
        if fmt == "png" and not path.lower().endswith(".png"):
            path += ".png"
        if fmt == "jpg" and not (path.lower().endswith(".jpg") or path.lower().endswith(".jpeg")):
            path += ".jpg"
        pixmap = self._grab_current_view()
        if fmt == "pdf":
            printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QtPrintSupport.QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(path)
            painter = QtGui.QPainter(printer)
            rect = printer.pageRect()
            scaled = pixmap.scaled(rect.size().toSize(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled)
            painter.end()
        else:
            pixmap.save(path, fmt.upper())

    def _on_tab_changed(self, index: int) -> None:
        return

    def _prompt_annotation_action(self) -> str:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Annotate current view before exporting?")
        dialog.setText("Annotate current view before exporting?")
        annotate_btn = dialog.addButton("Annotate", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        export_btn = dialog.addButton("Export without annotation", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = dialog.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == annotate_btn:
            return "annotate"
        if clicked == export_btn:
            return "export"
        if clicked == cancel_btn:
            return "cancel"
        return "cancel"

    def _open_annotation_editor(self) -> None:
        pixmap = self._grab_current_view()
        editor = AnnotationEditorWindow(pixmap, self)
        editor.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        tokens = self._theme_manager.tokens()
        editor.apply_theme(tokens)
        editor.show()
        self._annotation_editors.append(editor)

    def _annotation_layer_for_tab(self, tab: QtWidgets.QWidget, create: bool = True):
        if tab in self._annotation_layers:
            return self._annotation_layers[tab]
        if not create:
            return None
        layer = self.AnnotationLayer(tab)
        layer.setGeometry(tab.rect())
        layer.setStyleSheet("background: transparent;")
        layer.hide()
        layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layer.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        layer.raise_()
        self._annotation_layers[tab] = layer
        tab.installEventFilter(self)
        return layer

    def _refresh_annotation_layers(self) -> None:
        show_flag = getattr(self, "_show_annotations_act", None).isChecked() if hasattr(self, "_show_annotations_act") else False
        annotate_flag = getattr(self, "_annotate_act", None).isChecked() if hasattr(self, "_annotate_act") else False
        current_tab = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        toolbox = self._get_annotation_toolbox()
        for tab, layer in self._annotation_layers.items():
            if not isinstance(tab, QtWidgets.QWidget):
                continue
            layer.setGeometry(tab.rect())
            if tab is current_tab:
                visible = show_flag or annotate_flag
                layer.setVisible(visible)
                layer.set_edit_mode(annotate_flag)
                layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not annotate_flag)
                layer.raise_()
                self._set_tab_interactive(tab, not annotate_flag)
                if toolbox:
                    toolbox.set_layer(layer)
                    if annotate_flag:
                        toolbox.show()
                    else:
                        toolbox.hide()
            else:
                layer.setVisible(False)
                layer.set_edit_mode(False)
                layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
                self._set_tab_interactive(tab, True)
        if hasattr(self, "_show_annotations_act"):
            self._show_annotations_act.setText("Hide annotations" if show_flag or annotate_flag else "Show annotations")

    def _set_tab_interactive(self, tab: QtWidgets.QWidget, enabled: bool) -> None:
        # Keep viewers enabled; the overlay intercepts events when annotating.
        return

    def _get_annotation_toolbox(self):
        if self._annotation_toolbox is None:
            self._annotation_toolbox = self.AnnotationToolbox(self)
            self._annotation_toolbox.hide()
        return self._annotation_toolbox

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Resize and isinstance(obj, QtWidgets.QWidget):
            layer = self._annotation_layers.get(obj)
            if layer:
                layer.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def _toggle_annotation_mode(self, enabled: bool) -> None:
        self._refresh_annotation_layers()

    def _toggle_annotation_visibility(self, visible: bool) -> None:
        if not visible and hasattr(self, "_annotate_act"):
            self._annotate_act.setChecked(False)
        self._refresh_annotation_layers()

    def _undo_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.undo()

    def _redo_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.redo()

    def _copy_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.copy_selected()

    def _paste_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.paste_copied()

    def _clear_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.clear_annotations()

    def _save_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        if not layer.has_annotations():
            QtWidgets.QMessageBox.information(self, "Save annotations", "No annotations to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        data = {"annotations": layer.to_data()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations", "", "YAML Files (*.yaml *.yml *.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            try:
                import yaml  # type: ignore
                loaded = yaml.safe_load(raw)
            except Exception:
                loaded = json.loads(raw)
            shapes = (loaded or {}).get("annotations", [])
            layer.from_data(shapes)
            layer.setVisible(True)
            self._show_annotations_act.setChecked(True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load annotations", f"Failed to load annotations: {exc}")

    class AnnotationLayer(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.shapes: list[dict] = []
            self.temp_shape: dict | None = None
            self.edit_mode = False
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            # style defaults
            self.mode = "arrow"
            self.line_color = "#0f172a"
            self.fill_color = "#0f172a"
            self.fill_mode = "none"
            self.line_style = "solid"
            self.line_thickness = 2
            self.line_opacity = 1.0
            self.fill_opacity = 0.4
            self.text_font = "Segoe UI"
            self.text_size = 12
            self.start_pos: QtCore.QPointF | None = None
            self.selected_idx: int | None = None
            self.drag_mode: str | None = None
            self.drag_offset: QtCore.QPointF | None = None
            self._resize_anchor: QtCore.QPointF | None = None
            self._snapshot_before_drag: list[dict] | None = None
            self.undo_stack: list[list[dict]] = []
            self.redo_stack: list[list[dict]] = []
            self.clipboard_shape: dict | None = None

        def has_annotations(self) -> bool:
            return bool(self.shapes)

        def clear_annotations(self) -> None:
            if self.shapes:
                self._push_undo()
            self.shapes.clear()
            self.temp_shape = None
            self.selected_idx = None
            self.update()

        def set_edit_mode(self, enabled: bool) -> None:
            self.edit_mode = enabled
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not enabled)
            self.update()

        def set_mode(self, mode: str) -> None:
            self.mode = mode
            self.temp_shape = None
            self.update()
            if mode == "eraser":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        def set_style(self, line_color: str, fill_color: str, fill: str, line: str, line_thickness: int, line_opacity: float, fill_opacity: float, font: str, font_size: int) -> None:
            self.line_color = line_color
            self.fill_color = fill_color
            self.fill_mode = fill
            self.line_style = line
            self.line_thickness = line_thickness
            self.line_opacity = line_opacity
            self.fill_opacity = fill_opacity
            self.text_font = font
            self.text_size = font_size
            self.update()

        def to_data(self) -> list[dict]:
            return [dict(s) for s in self.shapes]

        def from_data(self, data: list[dict]) -> None:
            self.shapes = []
            for item in data or []:
                if "type" in item:
                    self.shapes.append(dict(item))
            self.temp_shape = None
            self.selected_idx = None
            self.update()

        def _make_pen(self, color: str = None, thickness: int = None, line: str = None, opacity: float | None = None) -> QtGui.QPen:
            c = QtGui.QColor(color or self.line_color)
            alpha = int(255 * (opacity if opacity is not None else self.line_opacity))
            c.setAlpha(alpha)
            pen = QtGui.QPen(c, thickness or self.line_thickness)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine if (line or self.line_style) == "dashed" else QtCore.Qt.PenStyle.SolidLine)
            return pen

        def _make_brush(self, color: str = None, fill: str = None, opacity: float | None = None) -> QtGui.QBrush:
            fill_mode = (fill or self.fill_mode).lower()
            alpha = int(255 * (opacity if opacity is not None else self.fill_opacity))
            if fill_mode == "solid":
                col = QtGui.QColor(color or self.fill_color)
                col.setAlpha(alpha)
                return QtGui.QBrush(col)
            return QtCore.Qt.NoBrush

        def _shape_to_path(self, painter: QtGui.QPainter, shape: dict, temp: bool = False):
            stype = shape.get("type")
            start = QtCore.QPointF(*shape.get("start", (0, 0)))
            end = QtCore.QPointF(*shape.get("end", (0, 0)))
            pen = self._make_pen(
                shape.get("line_color", shape.get("color")),
                shape.get("thickness", shape.get("line_thickness")),
                shape.get("line", self.line_style),
                shape.get("opacity", self.line_opacity),
            )
            brush = self._make_brush(
                shape.get("fill_color", shape.get("color")),
                shape.get("fill", self.fill_mode),
                shape.get("fill_opacity", self.fill_opacity),
            )
            painter.setPen(pen)
            if stype == "rect":
                painter.setBrush(brush)
                painter.drawRect(QtCore.QRectF(start, end).normalized())
            elif stype == "ellipse":
                painter.setBrush(brush)
                painter.drawEllipse(QtCore.QRectF(start, end).normalized())
            elif stype == "arrow":
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawLine(start, end)
                self._draw_arrow_head(painter, start, end, pen.color(), pen.widthF(), pen.style())
            elif stype == "text":
                font = QtGui.QFont(shape.get("font", self.text_font), int(shape.get("font_size", self.text_size)))
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(shape.get("color", "#0f172a")), pen.width()))
                painter.drawText(end, shape.get("text", ""))

        def _draw_arrow_head(self, painter: QtGui.QPainter, start: QtCore.QPointF, end: QtCore.QPointF, color: QtGui.QColor, width: float, style: QtCore.Qt.PenStyle):
            line_vec = end - start
            angle = math.atan2(line_vec.y(), line_vec.x())
            length = max(10.0, 4.0 * width)
            theta = math.radians(25)
            p1 = end - QtCore.QPointF(length * math.cos(angle - theta), length * math.sin(angle - theta))
            p2 = end - QtCore.QPointF(length * math.cos(angle + theta), length * math.sin(angle + theta))
            poly = QtGui.QPolygonF([end, p1, p2])
            pen = QtGui.QPen(color, width)
            pen.setStyle(style)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawPolygon(poly)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            for shape in self.shapes:
                self._shape_to_path(painter, shape)
            if self.temp_shape:
                self._shape_to_path(painter, self.temp_shape, temp=True)
            # selection highlight
            if self.selected_idx is not None and 0 <= self.selected_idx < len(self.shapes):
                shape = self.shapes[self.selected_idx]
                rect = QtCore.QRectF(QtCore.QPointF(*shape.get("start", (0, 0))), QtCore.QPointF(*shape.get("end", (0, 0)))).normalized()
                pen = QtGui.QPen(QtGui.QColor("#38bdf8"), 1, QtCore.Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(rect.adjusted(-4, -4, 4, 4))
            painter.end()

        def _push_undo(self):
            self.undo_stack.append([dict(s) for s in self.shapes])
            self.redo_stack.clear()

        def undo(self):
            if not self.undo_stack:
                return
            self.redo_stack.append([dict(s) for s in self.shapes])
            self.shapes = self.undo_stack.pop()
            self.selected_idx = None
            self.temp_shape = None
            self.update()

        def redo(self):
            if not self.redo_stack:
                return
            self.undo_stack.append([dict(s) for s in self.shapes])
            self.shapes = self.redo_stack.pop()
            self.selected_idx = None
            self.temp_shape = None
            self.update()

        def copy_selected(self):
            if self.selected_idx is None or self.selected_idx >= len(self.shapes):
                return
            self.clipboard_shape = dict(self.shapes[self.selected_idx])

        def paste_copied(self):
            if not self.clipboard_shape:
                return
            shape = dict(self.clipboard_shape)
            # offset a bit
            sx, sy = shape.get("start", (0, 0))
            ex, ey = shape.get("end", (0, 0))
            shape["start"] = (sx + 10, sy + 10)
            shape["end"] = (ex + 10, ey + 10)
            self._push_undo()
            self.shapes.append(shape)
            self.selected_idx = len(self.shapes) - 1
            self.update()

        def _hit_test(self, pos: QtCore.QPointF):
            tolerance = 8.0
            for idx in range(len(self.shapes) - 1, -1, -1):
                shape = self.shapes[idx]
                stype = shape.get("type")
                start = QtCore.QPointF(*shape.get("start", (0, 0)))
                end = QtCore.QPointF(*shape.get("end", (0, 0)))
                rect = QtCore.QRectF(start, end).normalized()
                if stype in ("rect", "ellipse", "highlight"):
                    if rect.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(pos):
                        # corner detection
                        corners = [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]
                        for c in corners:
                            if QtCore.QLineF(pos, c).length() <= tolerance:
                                # anchor is opposite corner
                                opp = QtCore.QPointF(rect.center().x() * 2 - c.x(), rect.center().y() * 2 - c.y())
                                return idx, "resize_corner", opp
                        return idx, "move", pos - start
                elif stype == "arrow":
                    # distance to line
                    dist = _point_to_line_distance(pos, QtCore.QLineF(start, end))
                    if dist <= tolerance:
                        if QtCore.QLineF(end, pos).length() <= tolerance:
                            return idx, "resize_end", None
                        if QtCore.QLineF(start, pos).length() <= tolerance:
                            return idx, "resize_start", None
                        return idx, "move", pos - start
                elif stype == "text":
                    if QtCore.QLineF(pos, end).length() <= tolerance:
                        return idx, "move", pos - end
            return None, None, None

        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            pos = event.position()
            if self.mode == "eraser":
                hit_idx, mode, data = self._hit_test(pos)
                if hit_idx is not None:
                    self._push_undo()
                    self.shapes.pop(hit_idx)
                    self.selected_idx = None
                    self.update()
                return
            hit_idx, mode, data = self._hit_test(pos)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                self.drag_mode = mode
                self.drag_offset = data
                self._snapshot_before_drag = [dict(s) for s in self.shapes]
                if mode == "resize_corner":
                    self._resize_anchor = data
                return
            self.selected_idx = None
            self.start_pos = pos
            if self.mode == "text":
                text, ok = QtWidgets.QInputDialog.getText(self, "Add text", "Text:")
                if ok and text:
                    self._push_undo()
                    self.shapes.append({
                        "type": "text",
                        "text": text,
                        "start": (pos.x(), pos.y()),
                        "end": (pos.x(), pos.y()),
                        "line_color": self.line_color,
                        "fill_color": self.fill_color,
                        "line_thickness": self.line_thickness,
                        "line": self.line_style,
                        "fill": self.fill_mode,
                        "opacity": self.line_opacity,
                        "fill_opacity": self.fill_opacity,
                        "font": self.text_font,
                        "font_size": self.text_size,
                    })
                    self.selected_idx = len(self.shapes) - 1
                    self.update()
            else:
                self.temp_shape = {
                    "type": self.mode,
                    "start": (pos.x(), pos.y()),
                    "end": (pos.x(), pos.y()),
                    "line_color": self.line_color,
                    "fill_color": self.fill_color,
                    "line_thickness": self.line_thickness,
                    "line": self.line_style,
                    "fill": self.fill_mode,
                    "opacity": self.line_opacity,
                    "fill_opacity": self.fill_opacity,
                    "font": self.text_font,
                    "font_size": self.text_size,
                }
                self.update()

        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            pos = event.position()
            if self.temp_shape:
                self.temp_shape["end"] = (pos.x(), pos.y())
                self.update()
                return
            if self.selected_idx is not None and self.drag_mode:
                shape = self.shapes[self.selected_idx]
                if self.drag_mode == "move":
                    offset = self.drag_offset or QtCore.QPointF(0, 0)
                    start_pt = pos - offset
                    sx, sy = start_pt.x(), start_pt.y()
                    ex, ey = shape.get("end", (0, 0))
                    end_vec = QtCore.QPointF(ex, ey) - QtCore.QPointF(*shape.get("start", (0, 0)))
                    new_end = start_pt + end_vec
                    shape["start"] = (sx, sy)
                    shape["end"] = (new_end.x(), new_end.y())
                elif self.drag_mode in ("resize_corner", "resize_start", "resize_end"):
                    if self.drag_mode == "resize_corner" and self._resize_anchor is not None:
                        shape["start"] = (self._resize_anchor.x(), self._resize_anchor.y())
                        shape["end"] = (pos.x(), pos.y())
                    elif self.drag_mode == "resize_start":
                        shape["start"] = (pos.x(), pos.y())
                    elif self.drag_mode == "resize_end":
                        shape["end"] = (pos.x(), pos.y())
                self.update()

        def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            if self.temp_shape:
                pos = event.position()
                self.temp_shape["end"] = (pos.x(), pos.y())
                self._push_undo()
                self.shapes.append(self.temp_shape)
                self.selected_idx = len(self.shapes) - 1
                self.temp_shape = None
                self.update()
                return
            if self.selected_idx is not None and getattr(self, "_snapshot_before_drag", None) is not None:
                self.undo_stack.append(self._snapshot_before_drag)
                self.redo_stack.clear()
                self._snapshot_before_drag = None
                self.update()
            self.drag_mode = None
            self.drag_offset = None
            self._resize_anchor = None
            self.start_pos = None

    class AnnotationToolbox(QtWidgets.QDialog):
        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Annotation tools")
            self.setWindowFlag(QtCore.Qt.WindowType.Tool)
            self.layer: OrbSimMainWindow.AnnotationLayer | None = None
            layout = QtWidgets.QVBoxLayout(self)
            # mode buttons
            mode_row = QtWidgets.QHBoxLayout()
            self.mode_buttons: dict[str, QtWidgets.QToolButton] = {}
            modes = [
                ("Arrow", "arrow", "fa5s.arrow-right"),
                ("Rect", "rect", "fa5s.square"),
                ("Ellipse", "ellipse", "fa5s.circle"),
                ("Text", "text", "fa5s.font"),
                ("Eraser", "eraser", "fa5s.eraser"),
            ]
            for label, key, icon_name in modes:
                btn = QtWidgets.QToolButton()
                btn.setText(label)
                btn.setIcon(qta.icon(icon_name))
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, k=key: self._set_mode(k))
                mode_row.addWidget(btn)
                self.mode_buttons[key] = btn
            layout.addLayout(mode_row)

            form = QtWidgets.QFormLayout()
            self.color_combo = QtWidgets.QComboBox()
            colors = {
                "Black": "#0f172a",
                "Blue": "#2563eb",
                "Red": "#ef4444",
                "Green": "#22c55e",
                "Orange": "#f97316",
                "Purple": "#8b5cf6",
            }
            for name, val in colors.items():
                self.color_combo.addItem(name, userData=val)
            form.addRow("Color", self.color_combo)

            self.fill_color_combo = QtWidgets.QComboBox()
            for name, val in colors.items():
                self.fill_color_combo.addItem(name, userData=val)
            form.addRow("Fill color", self.fill_color_combo)

            self.fill_combo = QtWidgets.QComboBox()
            self.fill_combo.addItems(["None", "Solid"])
            form.addRow("Fill", self.fill_combo)

            self.line_combo = QtWidgets.QComboBox()
            self.line_combo.addItems(["Solid", "Dashed"])
            form.addRow("Line style", self.line_combo)

            self.line_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.line_opacity_slider.setRange(10, 100)
            self.line_opacity_slider.setValue(100)
            form.addRow("Line opacity (%)", self.line_opacity_slider)

            self.fill_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.fill_opacity_slider.setRange(10, 100)
            self.fill_opacity_slider.setValue(40)
            form.addRow("Fill opacity (%)", self.fill_opacity_slider)

            self.thickness_spin = QtWidgets.QSpinBox()
            self.thickness_spin.setRange(1, 12)
            self.thickness_spin.setValue(2)
            form.addRow("Thickness", self.thickness_spin)

            self.font_combo = QtWidgets.QFontComboBox()
            form.addRow("Font", self.font_combo)

            self.font_size_spin = QtWidgets.QSpinBox()
            self.font_size_spin.setRange(8, 48)
            self.font_size_spin.setValue(12)
            form.addRow("Font size", self.font_size_spin)

            layout.addLayout(form)

            for widget in (
                self.color_combo,
                self.fill_color_combo,
                self.fill_combo,
                self.line_combo,
                self.line_opacity_slider,
                self.fill_opacity_slider,
                self.thickness_spin,
                self.font_combo,
                self.font_size_spin,
            ):
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.currentIndexChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QSlider):
                    widget.valueChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.valueChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QFontComboBox):
                    widget.currentFontChanged.connect(lambda _: self._apply_style())

            if self.mode_buttons:
                list(self.mode_buttons.values())[0].setChecked(True)
                self.current_mode = modes[0][1]
            else:
                self.current_mode = "arrow"

        def set_layer(self, layer: OrbSimMainWindow.AnnotationLayer) -> None:
            self.layer = layer
            self._apply_style()
            self.layer.set_mode(self.current_mode)

        def _set_mode(self, mode: str) -> None:
            self.current_mode = mode
            for key, btn in self.mode_buttons.items():
                btn.setChecked(key == mode)
            if self.layer:
                self.layer.set_mode(mode)
            if mode == "eraser":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            else:
                self.unsetCursor()

        def _apply_style(self) -> None:
            if not self.layer:
                return
            line_color = self.color_combo.currentData()
            fill_color = self.fill_color_combo.currentData()
            fill = self.fill_combo.currentText().lower()
            line = "dashed" if "dash" in self.line_combo.currentText().lower() else "solid"
            line_opacity = self.line_opacity_slider.value() / 100.0
            fill_opacity = self.fill_opacity_slider.value() / 100.0
            thickness = self.thickness_spin.value()
            font = self.font_combo.currentFont().family()
            font_size = self.font_size_spin.value()
            self.layer.set_style(line_color, fill_color, fill, line, thickness, line_opacity, fill_opacity, font, font_size)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            parent = self.parent()
            if isinstance(parent, OrbSimMainWindow) and hasattr(parent, "_annotate_act"):
                parent._annotate_act.setChecked(False)
                # keep annotations shown by default
                if hasattr(parent, "_show_annotations_act"):
                    parent._show_annotations_act.setChecked(True)
                parent._refresh_annotation_layers()
            super().closeEvent(event)


def _point_to_line_distance(point: QtCore.QPointF, line: QtCore.QLineF) -> float:
    """Perpendicular distance from point to a line segment."""
    x0, y0 = point.x(), point.y()
    x1, y1 = line.p1().x(), line.p1().y()
    x2, y2 = line.p2().x(), line.p2().y()
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den else 0.0


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
