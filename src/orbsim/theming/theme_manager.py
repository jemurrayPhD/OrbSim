from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.theming.theme_tokens import THEME_TOKENS


class ThemeManager(QtCore.QObject):
    theme_changed = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._theme_name = "Fluent Light"
        self._tokens: dict = THEME_TOKENS["Fluent Light"]
        self._qss: str = ""
        self._theme_root: Path | None = None

    def available_themes(self) -> list[str]:
        names = list(THEME_TOKENS.keys())
        names.extend(self._skin_names())
        return names

    def _skin_names(self) -> list[str]:
        base = Path(__file__).resolve().parents[3] / "themes"
        if not base.exists():
            return []
        return [p.name.replace("_", " ").title() for p in base.iterdir() if (p / "theme.json").exists()]

    def set_theme(self, name: str) -> dict:
        if name in THEME_TOKENS:
            self._theme_name = name
            self._tokens = THEME_TOKENS[name]
            self._theme_root = None
            self._qss = ""
        else:
            self._load_skin(name)
        self.theme_changed.emit(self._tokens)
        return self._tokens

    def _load_skin(self, name: str) -> None:
        base = Path(__file__).resolve().parents[3] / "themes"
        skin_dir = base / name.lower().replace(" ", "_")
        theme_path = skin_dir / "theme.json"
        if not theme_path.exists():
            self._tokens = THEME_TOKENS["Fluent Light"]
            self._theme_root = None
            self._qss = ""
            return
        data = json.loads(theme_path.read_text())
        self._theme_name = data.get("name", name)
        self._tokens = {
            "meta": {"name": self._theme_name, "mode": data.get("mode", "light")},
            "colors": data.get("colors", {}),
            "radii": data.get("radii", {}),
            "spacing": data.get("spacing", {}),
            "font": data.get("font", {}),
            "assets": data.get("assets", {}),
        }
        self._theme_root = skin_dir
        qss_path = skin_dir / "widgets.qss"
        self._qss = qss_path.read_text() if qss_path.exists() else ""

    def tokens(self) -> dict:
        return self._tokens

    def qss(self) -> str:
        if not self._theme_root:
            return ""
        qss = self._qss
        if self._theme_root:
            qss = qss.replace("textures/", f"{self._theme_root.as_posix()}/textures/")
            qss = qss.replace("icons/", f"{self._theme_root.as_posix()}/icons/")
        return qss

    def resolve_asset(self, relative: str) -> str | None:
        if not self._theme_root or not relative:
            return None
        path = self._theme_root / relative
        return str(path) if path.exists() else None

    def icon_path(self, name: str) -> str | None:
        assets = self._tokens.get("assets", {})
        icon_rel = (assets.get("icons") or {}).get(name)
        return self.resolve_asset(icon_rel) if icon_rel else None

    def texture_path(self, key: str) -> str | None:
        assets = self._tokens.get("assets", {})
        rel = assets.get(key)
        return self.resolve_asset(rel) if rel else None


_theme_manager: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def apply_skin(app: QtWidgets.QApplication, tokens: dict) -> None:
    manager = get_theme_manager()
    qss = manager.qss()
    if qss:
        # ensure relative paths resolve from theme directory
        app.setStyleSheet(qss)
    else:
        app.setStyleSheet("")
    palette = QtGui.QPalette()
    colors = tokens.get("colors", {})
    if colors:
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(colors.get("bg", "#ffffff")))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colors.get("text", "#0f172a")))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(colors.get("surface", "#ffffff")))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(colors.get("text", "#0f172a")))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(colors.get("surfaceAlt", "#f0f0f0")))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(colors.get("text", "#0f172a")))
    app.setPalette(palette)
