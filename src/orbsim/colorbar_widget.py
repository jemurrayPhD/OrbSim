from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.tabs.shared import resolve_cmap


@dataclass
class ColorbarData:
    cmap_name: str
    vmin: float
    vmax: float
    label: str
    mode: str = "linear"


class HorizontalColorbarWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: ColorbarData | None = None
        self._colors = {
            "text": "#0f172a",
            "border": "#cbd5e1",
            "surface": "#ffffff",
        }
        self._gradient_cache: QtGui.QImage | None = None
        self.setMinimumHeight(54)

    def apply_theme(self, tokens: dict) -> None:
        colors = tokens.get("colors", {})
        self._colors.update(
            {
                "text": colors.get("text", self._colors["text"]),
                "border": colors.get("border", self._colors["border"]),
                "surface": colors.get("surface", self._colors["surface"]),
            }
        )
        self._gradient_cache = None
        self.update()

    def set_data(self, cmap_name: str, vmin: float, vmax: float, label: str, mode: str = "linear") -> None:
        self._data = ColorbarData(cmap_name=cmap_name, vmin=vmin, vmax=vmax, label=label, mode=mode)
        self._gradient_cache = None
        self.update()

    def clear(self) -> None:
        self._data = None
        self._gradient_cache = None
        self.update()

    def _build_gradient(self, width: int, height: int) -> QtGui.QImage:
        data = self._data
        if not data:
            image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
            image.fill(QtGui.QColor(self._colors["surface"]))
            return image
        gradient = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
        cmap = resolve_cmap(data.cmap_name)
        for x in range(width):
            frac = x / max(width - 1, 1)
            rgba = cmap(frac)
            color = QtGui.QColor(
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
            )
            for y in range(height):
                gradient.setPixelColor(x, y, color)
        return gradient

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor(self._colors["surface"]))
            if not self._data:
                return
            padding = 8
            label_height = 16
            bar_height = 12
            bar_rect = QtCore.QRect(
                rect.left() + padding,
                rect.top() + padding + label_height,
                rect.width() - padding * 2,
                bar_height,
            )
            if self._gradient_cache is None or self._gradient_cache.width() != bar_rect.width():
                self._gradient_cache = self._build_gradient(bar_rect.width(), bar_rect.height())
            painter.drawImage(bar_rect, self._gradient_cache)
            painter.setPen(QtGui.QPen(QtGui.QColor(self._colors["border"])))
            painter.drawRect(bar_rect.adjusted(0, 0, -1, -1))
            painter.setPen(QtGui.QPen(QtGui.QColor(self._colors["text"])))
            painter.setFont(self.font())
            label_text = self._data.label
            painter.drawText(
                rect.left() + padding,
                rect.top() + padding + 12,
                label_text,
            )
            vmin = self._data.vmin
            vmax = self._data.vmax
            text = painter.fontMetrics()
            min_text = f"{vmin:.3g}"
            max_text = f"{vmax:.3g}"
            painter.drawText(bar_rect.left(), bar_rect.bottom() + text.height() + 2, min_text)
            painter.drawText(
                bar_rect.right() - text.horizontalAdvance(max_text),
                bar_rect.bottom() + text.height() + 2,
                max_text,
            )
        finally:
            painter.end()
