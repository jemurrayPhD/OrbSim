from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def build_palette(tokens: dict) -> QtGui.QPalette:
    colors = tokens["colors"]
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(colors["bg"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(colors["surface"]))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(colors["surfaceAlt"]))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(colors["surface"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(colors["accent"]))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(colors["accent"]))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(colors["bg"]))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(colors["surface"]))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(colors["text"]))
    return palette


def build_stylesheet(tokens: dict) -> str:
    colors = tokens["colors"]
    radii = tokens["radii"]
    spacing = tokens["spacing"]
    font = tokens["font"]
    focus = colors["focusRing"]
    return f"""
    * {{
        font-family: "{font["family"]}";
        font-size: {font["baseSize"]}pt;
    }}
    QMainWindow {{
        background-color: {colors["bg"]};
    }}
    QWidget {{
        color: {colors["text"]};
    }}
    QTabWidget::pane {{
        border: 1px solid {colors["border"]};
        border-radius: {radii["md"]}px;
        background: {colors["surface"]};
        padding: {spacing["sm"]}px;
    }}
    QTabBar::tab {{
        background: {colors["surfaceAlt"]};
        border: 1px solid {colors["border"]};
        border-bottom: none;
        padding: {spacing["sm"]}px {spacing["md"]}px;
        margin-right: {spacing["xs"]}px;
        border-top-left-radius: {radii["sm"]}px;
        border-top-right-radius: {radii["sm"]}px;
    }}
    QTabBar::tab:selected {{
        background: {colors["surface"]};
        border-color: {colors["accent"]};
        color: {colors["text"]};
    }}
    QTabBar::tab:!selected {{
        color: {colors["textMuted"]};
    }}
    QPushButton, QToolButton {{
        background: {colors["surfaceAlt"]};
        border: 1px solid {colors["border"]};
        border-radius: {radii["sm"]}px;
        padding: {spacing["xs"]}px {spacing["md"]}px;
    }}
    QPushButton:hover, QToolButton:hover {{
        background: {colors["surface"]};
        border-color: {colors["accent"]};
    }}
    QPushButton:pressed, QToolButton:pressed {{
        background: {colors["surfaceAlt"]};
        border-color: {colors["accentHover"]};
    }}
    QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{
        outline: none;
        border: 2px solid {focus};
    }}
    QGroupBox {{
        border: 1px solid {colors["border"]};
        border-radius: {radii["md"]}px;
        margin-top: {spacing["sm"]}px;
        padding: {spacing["sm"]}px;
        background: {colors["surface"]};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {spacing["sm"]}px;
        padding: 0 {spacing["xs"]}px;
        color: {colors["textMuted"]};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: {radii["sm"]}px;
        padding: {spacing["xs"]}px {spacing["sm"]}px;
        min-height: 28px;
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background: {colors["surfaceAlt"]};
        color: {colors["textMuted"]};
    }}
    QToolTip {{
        background: {colors["surface"]};
        color: {colors["text"]};
        border: 1px solid {colors["border"]};
        padding: {spacing["xs"]}px;
    }}
    """


def apply_theme(app: QtWidgets.QApplication, tokens: dict) -> None:
    app.setPalette(build_palette(tokens))
    app.setStyleSheet(build_stylesheet(tokens))
