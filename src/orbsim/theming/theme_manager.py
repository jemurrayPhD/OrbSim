from __future__ import annotations

import contextlib
import json
from importlib import resources
from importlib.resources.abc import Traversable

from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.theming.theme_tokens import THEME_TOKENS


class ThemeManager(QtCore.QObject):
    theme_changed = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._theme_name = "Fluent Light"
        self._tokens: dict = THEME_TOKENS["Fluent Light"]
        self._qss: str = ""
        self._theme_root: Traversable | None = None
        self._asset_stack = contextlib.ExitStack()

    def available_themes(self) -> list[str]:
        names = list(THEME_TOKENS.keys())
        names.extend(self._skin_names())
        return names

    def _skin_names(self) -> list[str]:
        base = resources.files("orbsim").joinpath("themes")
        if not base.is_dir():
            return []
        return [
            p.name.replace("_", " ").title()
            for p in base.iterdir()
            if p.is_dir() and p.joinpath("theme.json").is_file()
        ]

    def set_theme(self, name: str) -> dict:
        self._asset_stack.close()
        self._asset_stack = contextlib.ExitStack()
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
        base = resources.files("orbsim").joinpath("themes")
        skin_dir = base.joinpath(name.lower().replace(" ", "_"))
        theme_path = skin_dir.joinpath("theme.json")
        if not theme_path.is_file():
            self._tokens = THEME_TOKENS["Fluent Light"]
            self._theme_root = None
            self._qss = ""
            return
        data = json.loads(theme_path.read_text(encoding="utf-8"))
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
        qss_path = skin_dir.joinpath("widgets.qss")
        self._qss = qss_path.read_text(encoding="utf-8") if qss_path.is_file() else ""

    def tokens(self) -> dict:
        return self._tokens

    def qss(self) -> str:
        return self._qss if self._theme_root else ""

    def resolve_asset(self, relative: str) -> str | None:
        if not self._theme_root or not relative:
            return None
        path = self._theme_root.joinpath(relative)
        if not path.is_file():
            return None
        if hasattr(path, "__fspath__"):
            return str(path)
        resolved = self._asset_stack.enter_context(resources.as_file(path))
        return str(resolved)

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
        base = app.styleSheet()
        if base:
            app.setStyleSheet(f"{base}\n\n{qss}")
        else:
            app.setStyleSheet(qss)
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
