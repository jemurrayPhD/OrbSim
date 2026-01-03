from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def _relative_luminance(color: QtGui.QColor) -> float:
    def channel(value: float) -> float:
        value /= 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.red())
        + 0.7152 * channel(color.green())
        + 0.0722 * channel(color.blue())
    )


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
    highlight = QtGui.QColor(colors["accent"])
    palette.setColor(QtGui.QPalette.Highlight, highlight)
    highlight_text = QtGui.QColor("#0f172a") if _relative_luminance(highlight) > 0.5 else QtGui.QColor("#f8fafc")
    palette.setColor(QtGui.QPalette.HighlightedText, highlight_text)
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(colors["surface"]))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(colors["text"]))
    return palette


def build_stylesheet(tokens: dict) -> str:
    colors = tokens["colors"]
    radii = tokens["radii"]
    spacing = tokens["spacing"]
    font = tokens["font"]
    focus = colors["focusRing"]
    meta = tokens.get("meta", {})
    mode = meta.get("mode", "light")
    is_high_contrast = mode == "high_contrast"
    is_dark = mode == "dark"

    button_border = colors["surfaceAlt"] if is_high_contrast else colors["surfaceAlt"]
    button_bg = colors["surfaceAlt"] if is_high_contrast else colors["surfaceAlt"]
    button_hover = colors["surface"] if not is_high_contrast else colors["surfaceAlt"]
    input_focus_bg = colors["surfaceAlt"] if not is_high_contrast else colors["surfaceAlt"]
    tab_unselected_bg = colors["surfaceAlt"] if not is_high_contrast else colors["surface"]
    tab_selected_bg = colors["surface"] if not is_high_contrast else colors["surfaceAlt"]
    tab_border = colors["surfaceAlt"] if is_high_contrast else colors["border"]
    tab_selected_border = colors["accent"]
    input_border = colors["border"]
    card_border = colors["border"]
    focus_border_width = 1
    focus_padding_adjust = 0
    if is_high_contrast:
        focus_border_width = 1
        focus_padding_adjust = 0
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
        border: 1px solid {tab_border};
        border-radius: {radii["md"]}px;
        background: {colors["surface"]};
        padding: {spacing["xs"]}px;
    }}
    QTabBar::tab {{
        background: {tab_unselected_bg};
        border: 1px solid {tab_border};
        border-bottom: none;
        padding: {spacing["xs"]}px {spacing["md"]}px;
        margin-right: {spacing["xs"]}px;
        border-top-left-radius: {radii["sm"]}px;
        border-top-right-radius: {radii["sm"]}px;
        color: {colors["textMuted"]};
    }}
    QTabBar::tab:selected {{
        background: {tab_selected_bg};
        border-color: {tab_selected_border};
        color: {colors["text"]};
        padding-bottom: {spacing["xs"]}px;
    }}
    QPushButton, QToolButton {{
        background: {button_bg};
        border: 1px solid {button_border};
        border-radius: {radii["sm"]}px;
        padding: {spacing["xs"]}px {spacing["md"]}px;
        min-height: 28px;
    }}
    QPushButton:hover, QToolButton:hover {{
        background: {button_hover};
        border-color: {colors["accent"] if not is_high_contrast else colors["surfaceAlt"]};
    }}
    QPushButton:pressed, QToolButton:pressed {{
        background: {colors["surfaceAlt"]};
        border-color: {colors["accentHover"] if not is_high_contrast else colors["focusRing"]};
    }}
    QPushButton:focus, QToolButton:focus {{
        outline: none;
        border: {focus_border_width}px solid {focus};
        padding: {spacing["xs"] - focus_padding_adjust}px {spacing["md"] - focus_padding_adjust}px;
        background: {input_focus_bg if not is_high_contrast else colors["surfaceAlt"]};
    }}
    QGroupBox {{
        border: 1px solid {card_border};
        border-radius: {radii["md"]}px;
        margin-top: {spacing["md"]}px;
        padding: {spacing["md"]}px;
        background: {colors["surface"]};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {spacing["md"]}px;
        top: {spacing["xs"]}px;
        padding: 0 {spacing["xs"]}px;
        color: {colors["textMuted"]};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {colors["surface"]};
        border: 1px solid {input_border};
        border-radius: {radii["sm"]}px;
        padding: {spacing["xs"]}px {spacing["sm"]}px;
        min-height: 28px;
    }}
    QComboBox::drop-down {{
        width: 26px;
        border-left: 1px solid {input_border};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background: {colors["surfaceAlt"]};
        color: {colors["textMuted"]};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: {focus_border_width}px solid {focus};
        background: {input_focus_bg if not is_high_contrast else colors["surfaceAlt"]};
        padding: {spacing["xs"] - focus_padding_adjust}px {spacing["sm"] - focus_padding_adjust}px;
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
        border-color: {colors["accent"] if not is_high_contrast else colors["border"]};
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
