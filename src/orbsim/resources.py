from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui

from orbsim.theming.theme_manager import get_theme_manager

try:
    from . import resources_rc  # noqa: F401
except Exception:
    resources_rc = None


def load_icon(name: str) -> QtGui.QIcon:
    manager = get_theme_manager()
    themed_path = manager.icon_path(name.split(".")[0]) if name else None
    if themed_path:
        icon = QtGui.QIcon(themed_path)
        if not icon.isNull():
            return icon
    resource_path = f":/icons/{name}"
    if QtCore.QFile.exists(resource_path):
        icon = QtGui.QIcon(resource_path)
        if not icon.isNull():
            return icon
    asset_path = Path(__file__).resolve().parent / "assets" / "icons" / name
    return QtGui.QIcon(str(asset_path))
