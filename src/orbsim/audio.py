from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

try:
    from PySide6 import QtMultimedia
except Exception:  # pragma: no cover - optional QtMultimedia backend
    QtMultimedia = None


_CLICK_SOUND: object | None = None


def _sound_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "sounds" / "click.wav"


def play_click_sound() -> None:
    """Play a short click sound if available, otherwise fail silently."""
    global _CLICK_SOUND
    if QtMultimedia is None:
        try:
            QtWidgets.QApplication.beep()
        except Exception:
            pass
        return
    if _CLICK_SOUND is None:
        effect = QtMultimedia.QSoundEffect()
        effect.setLoopCount(1)
        effect.setVolume(0.35)
        effect.setSource(QtCore.QUrl.fromLocalFile(str(_sound_path())))
        _CLICK_SOUND = effect
    try:
        if hasattr(_CLICK_SOUND, "play"):
            _CLICK_SOUND.play()
    except Exception:
        try:
            QtWidgets.QApplication.beep()
        except Exception:
            pass
