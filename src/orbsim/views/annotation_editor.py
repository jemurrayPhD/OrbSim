from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class ToolState:
    name: str


class AnnotationCanvas(QtWidgets.QGraphicsView):
    def __init__(self, pixmap: QtGui.QPixmap, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.addPixmap(pixmap)
        self._current_tool = ToolState("arrow")
        self._current_color = QtGui.QColor("#f97316")
        self._font_size = 14
        self._start_pos: QtCore.QPointF | None = None
        self._active_item: QtWidgets.QGraphicsItem | None = None
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def set_tool(self, name: str) -> None:
        self._current_tool = ToolState(name)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    def set_color(self, color: QtGui.QColor) -> None:
        self._current_color = color

    def set_font_size(self, size: int) -> None:
        self._font_size = size

    def insert_text(self, text: str) -> None:
        item = QtWidgets.QGraphicsTextItem(text)
        font = item.font()
        font.setPointSize(self._font_size)
        item.setFont(font)
        item.setDefaultTextColor(self._current_color)
        item.setPos(self.mapToScene(self.viewport().rect().center()))
        item.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        self._scene.addItem(item)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._start_pos = self.mapToScene(event.position().toPoint())
            tool = self._current_tool.name
            if tool == "text":
                self.insert_text("Text")
                return
            if tool == "line":
                pen = QtGui.QPen(self._current_color, 2)
                line = QtWidgets.QGraphicsLineItem(QtCore.QLineF(self._start_pos, self._start_pos))
                line.setPen(pen)
                self._scene.addItem(line)
                self._active_item = line
                return
            if tool == "arrow":
                pen = QtGui.QPen(self._current_color, 2)
                line = QtWidgets.QGraphicsLineItem(QtCore.QLineF(self._start_pos, self._start_pos))
                line.setPen(pen)
                self._scene.addItem(line)
                self._active_item = line
                return
            if tool == "highlight":
                brush = QtGui.QBrush(QtGui.QColor(self._current_color).lighter(150))
                brush.setStyle(QtCore.Qt.BrushStyle.Dense4Pattern)
                rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(self._start_pos, self._start_pos))
                rect.setPen(QtGui.QPen(self._current_color, 2))
                rect.setBrush(brush)
                self._scene.addItem(rect)
                self._active_item = rect
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active_item and self._start_pos is not None:
            current = self.mapToScene(event.position().toPoint())
            if isinstance(self._active_item, QtWidgets.QGraphicsLineItem):
                self._active_item.setLine(QtCore.QLineF(self._start_pos, current))
            elif isinstance(self._active_item, QtWidgets.QGraphicsRectItem):
                rect = QtCore.QRectF(self._start_pos, current).normalized()
                self._active_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active_item and self._current_tool.name == "arrow":
            self._add_arrow_head(self._active_item)
        self._active_item = None
        self._start_pos = None
        super().mouseReleaseEvent(event)

    def _add_arrow_head(self, line_item: QtWidgets.QGraphicsLineItem) -> None:
        line = line_item.line()
        angle = QtCore.QLineF(line.p2(), line.p1()).angle()
        arrow_size = 10
        p1 = line.p2()
        p2 = QtCore.QPointF(
            p1.x() + arrow_size * math.cos(math.radians(angle - 20)),
            p1.y() + arrow_size * math.sin(math.radians(angle - 20)),
        )
        p3 = QtCore.QPointF(
            p1.x() + arrow_size * math.cos(math.radians(angle + 20)),
            p1.y() + arrow_size * math.sin(math.radians(angle + 20)),
        )
        polygon = QtGui.QPolygonF([p1, p2, p3])
        arrow = QtWidgets.QGraphicsPolygonItem(polygon)
        arrow.setBrush(QtGui.QBrush(self._current_color))
        arrow.setPen(QtGui.QPen(self._current_color, 1))
        self._scene.addItem(arrow)

    def render_to_image(self) -> QtGui.QImage:
        rect = self._scene.itemsBoundingRect()
        image = QtGui.QImage(rect.size().toSize(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        self._scene.render(painter, QtCore.QRectF(image.rect()), rect)
        painter.end()
        return image

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens["colors"]
        self.setStyleSheet(
            f"background: {colors['surface']}; border: 1px solid {colors['border']};"
        )


class AnnotationEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, pixmap: QtGui.QPixmap, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation Editor")
        self.setMinimumSize(900, 600)
        self.canvas = AnnotationCanvas(pixmap, self)
        self.setCentralWidget(self.canvas)
        self._build_toolbar()
        self._theme_tokens: dict | None = None

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        self.setStyleSheet(
            f"background: {colors['bg']}; color: {colors['text']};"
        )
        self.canvas.apply_theme(tokens)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Annotation Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        def add_tool(name: str, label: str) -> None:
            action = QtGui.QAction(label, self)
            action.triggered.connect(lambda: self.canvas.set_tool(name))
            toolbar.addAction(action)

        add_tool("text", "Text")
        add_tool("arrow", "Arrow")
        add_tool("line", "Line")
        add_tool("highlight", "Highlight rectangle")

        color_action = QtGui.QAction("Color", self)
        color_action.triggered.connect(self._pick_color)
        toolbar.addAction(color_action)

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Font size"))
        font_size = QtWidgets.QSpinBox()
        font_size.setRange(8, 48)
        font_size.setValue(14)
        font_size.valueChanged.connect(self.canvas.set_font_size)
        toolbar.addWidget(font_size)

        date_action = QtGui.QAction("Insert date/time", self)
        date_action.triggered.connect(self._insert_datetime)
        toolbar.addAction(date_action)

        toolbar.addSeparator()
        export_action = QtGui.QAction("Export image", self)
        export_action.triggered.connect(self._export_image)
        toolbar.addAction(export_action)

        copy_action = QtGui.QAction("Copy to clipboard", self)
        copy_action.triggered.connect(self._copy_image)
        toolbar.addAction(copy_action)

    def _pick_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.canvas._current_color, self, "Select annotation color")
        if color.isValid():
            self.canvas.set_color(color)

    def _insert_datetime(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.canvas.insert_text(now)

    def _export_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export annotated image", "", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        image = self.canvas.render_to_image()
        image.save(path)

    def _copy_image(self) -> None:
        image = self.canvas.render_to_image()
        QtWidgets.QApplication.clipboard().setImage(image)
