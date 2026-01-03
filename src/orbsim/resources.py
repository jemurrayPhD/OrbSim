from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui


try:
    from . import resources_rc  # noqa: F401
except Exception:
    resources_rc = None


def load_icon(name: str) -> QtGui.QIcon:
    icon = QtGui.QIcon(f":/icons/{name}")
    if not icon.isNull():
        return icon
    asset_path = Path(__file__).resolve().parent / "assets" / "icons" / name
    return QtGui.QIcon(str(asset_path))
