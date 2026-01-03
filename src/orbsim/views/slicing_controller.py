from __future__ import annotations

from PySide6 import QtCore


class SlicingController(QtCore.QObject):
    slice_changed = QtCore.Signal(dict)
    mode_changed = QtCore.Signal(str)

    def __init__(self, plotter) -> None:
        super().__init__()
        self._plotter = plotter
        self._mode = "none"
        self._plane_widget = None
        self._box_widget = None
        self._plane_state = {"normal": None, "origin": None}
        self._box_state = {"bounds": None}

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode == self._mode:
            return
        self._disable_widgets()
        self._mode = mode
        if mode == "plane":
            self._enable_plane_widget()
        elif mode == "box":
            self._enable_box_widget()
        self.mode_changed.emit(self._mode)
        self._emit_state()

    def reset(self) -> None:
        self._disable_widgets()
        self._mode = "none"
        self._plane_state = {"normal": None, "origin": None}
        self._box_state = {"bounds": None}
        self.mode_changed.emit(self._mode)
        self._emit_state()

    def _disable_widgets(self) -> None:
        for widget in (self._plane_widget, self._box_widget):
            if widget is None:
                continue
            try:
                widget.SetEnabled(False)
            except Exception:
                pass
        self._plane_widget = None
        self._box_widget = None

    def _enable_plane_widget(self) -> None:
        if not self._plotter:
            return

        def _callback(*args):
            normal = None
            origin = None
            if len(args) >= 2:
                normal, origin = args[0], args[1]
            elif len(args) == 1 and hasattr(args[0], "GetNormal"):
                try:
                    normal = args[0].GetNormal()
                    origin = args[0].GetOrigin()
                except Exception:
                    normal = None
            try:
                self._plane_state = {
                    "normal": tuple(normal) if normal is not None else None,
                    "origin": tuple(origin) if origin is not None else None,
                }
            except Exception:
                self._plane_state = {"normal": None, "origin": None}
            self._emit_state()

        try:
            self._plane_widget = self._plotter.add_plane_widget(
                _callback,
                normal=(0.0, 0.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                implicit=True,
            )
        except Exception:
            self._plane_widget = None

    def _enable_box_widget(self) -> None:
        if not self._plotter:
            return

        def _callback(*args):
            bounds = args[0] if args else None
            try:
                self._box_state = {"bounds": tuple(bounds) if bounds is not None else None}
            except Exception:
                self._box_state = {"bounds": None}
            self._emit_state()

        try:
            self._box_widget = self._plotter.add_box_widget(_callback)
        except Exception:
            self._box_widget = None

    def _emit_state(self) -> None:
        payload = {
            "mode": self._mode,
            "plane": self._plane_state,
            "box": self._box_state,
        }
        self.slice_changed.emit(payload)
