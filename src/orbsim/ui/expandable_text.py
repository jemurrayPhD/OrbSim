from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets


class ExpandableTextPopup(QtCore.QObject):
    def __init__(
        self,
        source: QtWidgets.QTextBrowser,
        *,
        anchor: QtWidgets.QWidget | None = None,
        frame_style_source: QtWidgets.QWidget | None = None,
        trigger_widgets: tuple[QtWidgets.QWidget, ...] | None = None,
        link_handler: Callable[[QtCore.QUrl], None] | None = None,
        max_height_ratio: float = 0.65,
        min_height: int = 140,
    ) -> None:
        super().__init__(source)
        self._source = source
        self._source_viewport = source.viewport() if hasattr(source, "viewport") else source
        self._anchor = anchor or source
        self._frame_style_source = frame_style_source or self._anchor
        self._link_handler = link_handler
        self._max_height_ratio = max(0.2, min(max_height_ratio, 0.9))
        self._min_height = max(80, int(min_height))
        self._triggers = trigger_widgets or (self._source_viewport,)

        self._popup = QtWidgets.QFrame(None, QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)
        self._popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._popup.setObjectName(self._frame_style_source.objectName() or "expandableTextPopup")
        layout = QtWidgets.QVBoxLayout(self._popup)
        layout.setContentsMargins(0, 0, 0, 0)

        self._browser = QtWidgets.QTextBrowser(self._popup)
        self._browser.setReadOnly(True)
        self._browser.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._browser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._browser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._browser.setOpenExternalLinks(False)
        self._browser.anchorClicked.connect(self._on_anchor_clicked)
        layout.addWidget(self._browser)

        for widget in self._triggers:
            widget.installEventFilter(self)
        self._browser.viewport().installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if obj in self._triggers:
                if obj == self._source_viewport and self._is_link_click(self._source, event):
                    return False
                self.toggle()
                return True
            if obj == self._browser.viewport():
                if self._is_link_click(self._browser, event):
                    return False
                self._popup.hide()
                return True
        return super().eventFilter(obj, event)

    def toggle(self) -> None:
        if self._popup.isVisible():
            self._popup.hide()
        else:
            self._sync_content()
            self._sync_styles()
            self._position_popup()
            self._popup.show()
            self._popup.raise_()

    def _sync_content(self) -> None:
        html = ""
        if hasattr(self._source, "toHtml"):
            html = self._source.toHtml()
        elif hasattr(self._source, "text"):
            html = self._source.text()
        self._browser.setHtml(html)
        self._browser.setFont(self._source.font())

    def _sync_styles(self) -> None:
        if self._frame_style_source and self._frame_style_source is not self._source:
            frame_style = self._frame_style_source.styleSheet()
            if frame_style:
                self._popup.setStyleSheet(frame_style)
        source_style = self._source.styleSheet() if self._source else ""
        if source_style:
            self._browser.setStyleSheet(source_style)

    def _position_popup(self) -> None:
        anchor_rect = self._anchor.rect()
        if anchor_rect.isNull():
            return
        global_pos = self._anchor.mapToGlobal(QtCore.QPoint(0, 0))
        window = self._anchor.window()
        window_rect = window.frameGeometry() if window else QtWidgets.QApplication.primaryScreen().availableGeometry()
        width = max(anchor_rect.width(), 240)
        max_height = max(self._min_height, int(window_rect.height() * self._max_height_ratio))
        self._browser.document().setTextWidth(max(width - 16, 120))
        doc_height = int(self._browser.document().size().height()) + 16
        height = max(self._min_height, min(doc_height, max_height))
        x = global_pos.x()
        y = global_pos.y()
        if y + height > window_rect.bottom():
            y = max(window_rect.y(), window_rect.bottom() - height)
        self._popup.resize(width, height)
        self._popup.move(x, y)

    def _on_anchor_clicked(self, url: QtCore.QUrl) -> None:
        if self._link_handler:
            self._link_handler(url)
            self._sync_content()
        else:
            QtGui.QDesktopServices.openUrl(url)

    def _is_link_click(self, browser: QtWidgets.QTextBrowser, event: QtCore.QEvent) -> bool:
        if not isinstance(event, QtGui.QMouseEvent):
            return False
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        pos = self._event_pos(event)
        try:
            return bool(browser.anchorAt(pos))
        except Exception:
            return False

    @staticmethod
    def _event_pos(event: QtGui.QMouseEvent) -> QtCore.QPoint:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()
