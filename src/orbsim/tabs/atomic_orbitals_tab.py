from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from pint import UnitRegistry
from pyvistaqt import QtInteractor

from orbsim.orbitals import (
    default_quantum_numbers,
    evaluate_orbital_values,
    field_from_grid,
    make_orbital_mesh,
    normalize_quantum_numbers,
    occupied_orbitals,
    _ATOMIC_NUMBER,
)
from orbsim.tabs.shared import resolve_cmap
from orbsim.ui.generated.ui_atomic_orbitals import Ui_AtomicOrbitalsTab
from orbsim.views.slicing_controller import SlicingController
from orbsim.widgets import DropPlotter

ureg = UnitRegistry()
Q_ = ureg.Quantity

NUCLEUS_COLORS = {
    "H": "#fbbf24",
    "He": "#a5f3fc",
    "C": "#38bdf8",
    "N": "#6366f1",
    "O": "#ef4444",
    "F": "#22c55e",
    "Ne": "#c084fc",
    "Na": "#38bdf8",
    "Mg": "#22d3ee",
    "Al": "#fde68a",
    "Si": "#60a5fa",
    "P": "#f59e0b",
    "S": "#eab308",
    "Cl": "#22c55e",
    "Ar": "#c084fc",
}


@dataclass
class PositionedOrbital:
    symbol: str
    n: int
    l: int
    m: int
    position: np.ndarray
    visible: bool = True


EXAMPLE_MOLECULES: dict[str, list[PositionedOrbital]] = {
    "H2 (covalent)": [
        PositionedOrbital("H", 1, 0, 0, np.array([-0.37, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([0.37, 0.0, 0.0])),
    ],
    "H2O (covalent)": [
        PositionedOrbital("O", 2, 1, 0, np.array([0.0, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([0.756, 0.587, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-0.756, 0.587, 0.0])),
    ],
    "CH4 (covalent)": [
        PositionedOrbital("C", 2, 1, 0, np.array([0.0, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([0.629, 0.629, 0.629])),
        PositionedOrbital("H", 1, 0, 0, np.array([0.629, -0.629, -0.629])),
        PositionedOrbital("H", 1, 0, 0, np.array([-0.629, 0.629, -0.629])),
        PositionedOrbital("H", 1, 0, 0, np.array([-0.629, -0.629, 0.629])),
    ],
    "CO2 (covalent)": [
        PositionedOrbital("O", 2, 1, 0, np.array([0.0, 0.0, -1.16])),
        PositionedOrbital("C", 2, 1, 0, np.array([0.0, 0.0, 0.0])),
        PositionedOrbital("O", 2, 1, 0, np.array([0.0, 0.0, 1.16])),
    ],
    "NaCl (ionic)": [
        PositionedOrbital("Na", 3, 0, 0, np.array([-1.18, 0.0, 0.0])),
        PositionedOrbital("Cl", 3, 1, 0, np.array([1.18, 0.0, 0.0])),
    ],
    "HF (polar covalent)": [
        PositionedOrbital("H", 1, 0, 0, np.array([-0.462, 0.0, 0.0])),
        PositionedOrbital("F", 2, 1, 0, np.array([0.0, 0.0, 0.0])),
    ],
    "Ethene C2H4 (pi bond)": [
        PositionedOrbital("C", 2, 1, 0, np.array([-0.67, 0.0, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([0.67, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-1.23, 0.93, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-1.23, -0.93, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([1.23, 0.93, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([1.23, -0.93, 0.0])),
    ],
    "Acetylene C2H2 (pi bond)": [
        PositionedOrbital("C", 2, 1, 0, np.array([-0.6, 0.0, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([0.6, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-1.68, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([1.68, 0.0, 0.0])),
    ],
    "Benzene C6H6 (delocalized pi)": [
        PositionedOrbital("C", 2, 1, 0, np.array([1.40, 0.0, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([0.70, 1.21, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([-0.70, 1.21, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([-1.40, 0.0, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([-0.70, -1.21, 0.0])),
        PositionedOrbital("C", 2, 1, 0, np.array([0.70, -1.21, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([2.48, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([1.24, 2.15, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-1.24, 2.15, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-2.48, 0.0, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([-1.24, -2.15, 0.0])),
        PositionedOrbital("H", 1, 0, 0, np.array([1.24, -2.15, 0.0])),
    ],
    "CO (polar sigma)": [
        PositionedOrbital("C", 2, 1, 0, np.array([-0.55, 0.0, 0.0])),
        PositionedOrbital("O", 2, 1, 0, np.array([0.55, 0.0, 0.0])),
    ],
}


class AddOrbitalDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Orbital")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)

        self.symbol_combo = QtWidgets.QComboBox(self)
        for sym in sorted(_ATOMIC_NUMBER.keys(), key=lambda s: _ATOMIC_NUMBER[s]):
            self.symbol_combo.addItem(sym)
        layout.addRow("Element", self.symbol_combo)

        self.n_spin = QtWidgets.QSpinBox(self)
        self.n_spin.setRange(1, 6)
        self.l_spin = QtWidgets.QSpinBox(self)
        self.l_spin.setRange(0, 5)
        self.m_spin = QtWidgets.QSpinBox(self)
        self.m_spin.setRange(-5, 5)
        self.n_spin.valueChanged.connect(self._sync_quantum_ranges)
        self.l_spin.valueChanged.connect(self._sync_quantum_ranges)
        self.symbol_combo.currentTextChanged.connect(self._set_defaults_for_symbol)

        quantum_widget = QtWidgets.QWidget(self)
        quantum_layout = QtWidgets.QHBoxLayout(quantum_widget)
        quantum_layout.setContentsMargins(0, 0, 0, 0)
        quantum_layout.addWidget(QtWidgets.QLabel("n"))
        quantum_layout.addWidget(self.n_spin)
        quantum_layout.addWidget(QtWidgets.QLabel("l"))
        quantum_layout.addWidget(self.l_spin)
        quantum_layout.addWidget(QtWidgets.QLabel("m"))
        quantum_layout.addWidget(self.m_spin)
        layout.addRow("Quantum numbers", quantum_widget)

        self.x_spin = QtWidgets.QDoubleSpinBox(self)
        self.y_spin = QtWidgets.QDoubleSpinBox(self)
        self.z_spin = QtWidgets.QDoubleSpinBox(self)
        for spin in (self.x_spin, self.y_spin, self.z_spin):
            spin.setRange(-20.0, 20.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.25)
        position_widget = QtWidgets.QWidget(self)
        pos_layout = QtWidgets.QHBoxLayout(position_widget)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.addWidget(QtWidgets.QLabel("x"))
        pos_layout.addWidget(self.x_spin)
        pos_layout.addWidget(QtWidgets.QLabel("y"))
        pos_layout.addWidget(self.y_spin)
        pos_layout.addWidget(QtWidgets.QLabel("z"))
        pos_layout.addWidget(self.z_spin)
        layout.addRow("Position (A)", position_widget)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._set_defaults_for_symbol(self.symbol_combo.currentText())

    def _set_defaults_for_symbol(self, symbol: str) -> None:
        default_n, default_l = default_quantum_numbers(symbol)
        valence_orbs = occupied_orbitals(symbol)
        m_val = 0
        if valence_orbs:
            vn, vl, vm = valence_orbs[-1]
            default_n, default_l, m_val = vn, vl, vm
        self.n_spin.blockSignals(True)
        self.l_spin.blockSignals(True)
        self.m_spin.blockSignals(True)
        self.n_spin.setValue(default_n)
        self.l_spin.setValue(default_l)
        self.m_spin.setRange(-default_l, default_l)
        if abs(m_val) <= default_l:
            self.m_spin.setValue(m_val)
        else:
            self.m_spin.setValue(0)
        self.n_spin.blockSignals(False)
        self.l_spin.blockSignals(False)
        self.m_spin.blockSignals(False)
        self._sync_quantum_ranges()

    def _sync_quantum_ranges(self) -> None:
        n_val = max(int(self.n_spin.value()), 1)
        l_val = min(max(int(self.l_spin.value()), 0), max(n_val - 1, 0))
        self.l_spin.setRange(0, max(n_val - 1, 0))
        self.l_spin.setValue(l_val)
        self.m_spin.setRange(-l_val, l_val)
        if abs(self.m_spin.value()) > l_val:
            self.m_spin.setValue(0)

    def populate_from(self, orbital: PositionedOrbital) -> None:
        idx = self.symbol_combo.findText(orbital.symbol)
        if idx >= 0:
            self.symbol_combo.setCurrentIndex(idx)
        self.n_spin.setValue(int(orbital.n))
        self.l_spin.setValue(int(orbital.l))
        self.m_spin.setRange(-orbital.l, orbital.l)
        self.m_spin.setValue(int(orbital.m))
        self.x_spin.setValue(float(orbital.position[0]))
        self.y_spin.setValue(float(orbital.position[1]))
        self.z_spin.setValue(float(orbital.position[2]))

    def result_orbital(self) -> PositionedOrbital:
        symbol = self.symbol_combo.currentText()
        n_val, l_val, m_val = normalize_quantum_numbers(symbol, self.n_spin.value(), self.l_spin.value(), self.m_spin.value())
        position = np.array([self.x_spin.value(), self.y_spin.value(), self.z_spin.value()], dtype=float)
        return PositionedOrbital(symbol=symbol, n=n_val, l=l_val, m=m_val, position=position)


class AtomicOrbitalsTab(QtWidgets.QWidget):
    """Main tab that wires together 3D/2D orbital rendering, controls, and preferences."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        ui_class = getattr(self, "_ui_class", Ui_AtomicOrbitalsTab)
        self.ui = ui_class()
        self.ui.setupUi(self)
        self.current_symbol = "H"
        self.current_n, self.current_l = default_quantum_numbers(self.current_symbol)
        self.current_m = 0
        self.current_mode = "amplitude"
        self.current_cmap = "viridis"
        self.current_representation = "surface"
        self.camera_initialized = False
        self.iso_fraction = 0.85
        self.iso_surfaces_count = 5
        self.slice_contours_count = 10
        self.slice_plane_resolution = 20
        self.show_occupied = False
        self._field_cache: OrderedDict[tuple, object] = OrderedDict()
        self.slice_normal = np.array([0.0, 0.0, 1.0])
        self.slice_offset = 0.0
        self.slice_origin: np.ndarray | None = None
        self.slice_mode: str = "none"
        self.slice_box_bounds: tuple[float, float, float, float, float, float] | None = None
        self.slice_plane_actor = None
        self.offset_spin: QtWidgets.QDoubleSpinBox | None = None
        self.offset_slider: QtWidgets.QSlider | None = None
        self.theta_spin: QtWidgets.QDoubleSpinBox | None = None
        self.phi_spin: QtWidgets.QDoubleSpinBox | None = None
        self.normal_spins: list[QtWidgets.QDoubleSpinBox] = []

        self._theme_tokens: dict | None = None
        self._text_color = "#ffffff"
        self._text_shadow = True
        self._slice_plane_color = "#cbd5e1"
        self._slice_plane_edge_color = "#94a3b8"
        self.plotter_frame = self.ui.plotterFrame
        try:
            self.plotter_frame.colorbar.hide()
            self.plotter_frame.colorbar.setFixedWidth(0)
        except Exception:
            pass
        self.plotter = self.plotter_frame.plotter
        self.slicing_controller = SlicingController(self.plotter)
        self.slicing_controller.slice_changed.connect(self._on_slice_widget_changed)
        try:
            self.plotter.enable_anti_aliasing()
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass
        self._setup_lights()
        self.slice_view = QtInteractor(self)
        self.slice_view.enable_anti_aliasing()
        try:
            self.slice_view.enable_parallel_projection()
            self.slice_view.enable_image_style()
        except Exception:
            pass
        self.slice_container = self.ui.sliceContainer
        slice_layout = QtWidgets.QVBoxLayout(self.slice_container)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        slice_layout.addWidget(self.slice_view)
        self.slice_colorbar = QtWidgets.QLabel()
        self.slice_colorbar.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop)
        self.slice_colorbar.setMinimumHeight(80)
        slice_layout.addWidget(self.slice_colorbar)
        self.slice_colorbar_range = QtWidgets.QLabel("Range: auto")
        self.slice_colorbar_range.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        slice_layout.addWidget(self.slice_colorbar_range)
        self.slice_container.setMinimumSize(320, 320)
        self.slice_container.installEventFilter(self)
        self.slice_colorbar.installEventFilter(self)
        self.slice_view.installEventFilter(self)
        self._colorbar_drag_start: float | None = None
        self._slice_data_range: tuple[float, float] | None = None
        self._slice_autoscale_done: bool = False
        self._scale_bar_actor_2d = None
        self._scale_bar_points = None
        self._scale_bar_poly = None
        self._scale_bar_text = None
        self._scale_bar_length_px = 120
        self._scale_bar_margin = (24, 24)
        self._last_slice_bounds: tuple[float, float, float, float, float, float] | None = None

        self.controls = self._build_controls()
        self._set_slice_normal(self.slice_normal, update_controls=True, trigger_render=False)
        self.slice_offset = 0.0
        self.offset_spin.setValue(self.slice_offset)
        self.slice_cmap = self.current_cmap
        self.slice_vmin = None
        self.slice_vmax = None
        self.slice_scalar_mode = "probability"
        self.iso_opacity_overrides: list[float] = []
        # preferences
        self.pref_show_grid = True
        self.pref_show_slice_plane = True
        self.pref_slice_plane_opacity = 0.35
        self._update_visibility_controls()

        self.controls_scroll = self.ui.controlsScroll
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(self.controls)
        self.controls_scroll.setMinimumWidth(520)
        self.controls_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        self.plotter_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.slice_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self._render_orbital()

    def apply_theme(self, tokens: dict) -> None:
        self._theme_tokens = tokens
        colors = tokens["colors"]
        mode = tokens.get("meta", {}).get("mode", "light")
        self._text_color = colors["text"]
        self._text_shadow = mode in ("dark", "high_contrast")
        self._slice_plane_color = colors["surfaceAlt"]
        self._slice_plane_edge_color = colors["surfaceAlt"] if mode == "high_contrast" else colors["border"]
        self.plotter_frame.apply_theme(tokens)
        try:
            self.plotter.set_background(colors["surfaceAlt"])
            self.slice_view.set_background(colors["surface"])
        except Exception:
            pass
        self.slice_colorbar.setStyleSheet(
            "background-color: {bg}; border: 1px solid {border}; padding: 4px; color: {text};".format(
                bg=colors["surfaceAlt"],
                border=colors["border"],
                text=colors["text"],
            )
        )
        self.slice_colorbar_range.setStyleSheet(
            "color: {text}; padding: 2px; font-size: 10px;".format(text=colors["textMuted"])
        )
        self._update_slice_scalebar()

    def _get_cmap(self, name: str):
        return resolve_cmap(name)

    def _build_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setMinimumWidth(520)
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        layout = QtWidgets.QVBoxLayout(container)

        # 3D controls (non-collapsible)
        three_d_group = QtWidgets.QGroupBox("3D View")
        three_d_layout = QtWidgets.QFormLayout(three_d_group)

        self.representation_combo = QtWidgets.QComboBox()
        self.representation_combo.addItem("Surface (iso-probability)", "surface")
        self.representation_combo.addItem("Volume (semi-transparent)", "volume")
        self.representation_combo.currentIndexChanged.connect(self._set_representation)
        three_d_layout.addRow("Type", self.representation_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["amplitude", "wavefunction"])
        self.mode_combo.currentTextChanged.connect(self._set_mode)
        three_d_layout.addRow("Mode", self.mode_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(
            ["viridis", "plasma", "inferno", "twilight", "coolwarm", "magma", "cividis", "batlow", "bamako", "devon", "oslo", "lajolla", "hawaii", "davos", "vik", "broc", "cork", "roma", "tokyo"]
        )
        self.cmap_combo.currentTextChanged.connect(self._set_cmap)
        three_d_layout.addRow("Colormap", self.cmap_combo)

        self.iso_slider_label = QtWidgets.QLabel("Contained probability (%)")
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.iso_slider.setRange(5, 99)
        self.iso_slider.setValue(int(self.iso_fraction * 100))
        self.iso_slider.valueChanged.connect(self._set_iso_fraction)
        three_d_layout.addRow(self.iso_slider_label, self.iso_slider)

        self.iso_spin_label = QtWidgets.QLabel("Iso surfaces (3D)")
        self.iso_surface_spin = QtWidgets.QSpinBox()
        self.iso_surface_spin.setRange(2, 12)
        self.iso_surface_spin.setValue(self.iso_surfaces_count)
        self.iso_surface_spin.valueChanged.connect(self._set_iso_surfaces_count)
        three_d_layout.addRow(self.iso_spin_label, self.iso_surface_spin)

        self.volume_tf_btn = QtWidgets.QPushButton("Edit transfer function (3D)")
        self.volume_tf_btn.clicked.connect(self._open_volume_tf_dialog)
        three_d_layout.addRow(self.volume_tf_btn)

        self.autoscale_btn = QtWidgets.QPushButton("Autoscale view")
        self.autoscale_btn.clicked.connect(self._autoscale)
        three_d_layout.addRow(self.autoscale_btn)

        prefs_btn = QtWidgets.QPushButton("Preferences")
        prefs_btn.clicked.connect(self._open_preferences)
        three_d_layout.addRow(prefs_btn)

        slicing_group = QtWidgets.QGroupBox("Slicing")
        slicing_layout = QtWidgets.QFormLayout(slicing_group)
        self.slice_mode_combo = QtWidgets.QComboBox()
        self.slice_mode_combo.addItem("None", "none")
        self.slice_mode_combo.addItem("Plane", "plane")
        self.slice_mode_combo.addItem("Box clip", "box")
        self.slice_mode_combo.currentIndexChanged.connect(self._on_slice_mode_change)
        slicing_layout.addRow("Mode", self.slice_mode_combo)
        reset_btn = QtWidgets.QPushButton("Reset slicing")
        reset_btn.clicked.connect(self._reset_slicing)
        slicing_layout.addRow(reset_btn)

        quantum_group = QtWidgets.QGroupBox("Quantum numbers")
        quantum_layout = QtWidgets.QFormLayout(quantum_group)
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(1, 6)
        self.n_spin.setValue(self.current_n)
        self.n_spin.valueChanged.connect(self._set_n)
        quantum_layout.addRow("n (principal)", self.n_spin)

        self.l_spin = QtWidgets.QSpinBox()
        self.l_spin.setRange(0, max(self.current_n - 1, 0))
        self.l_spin.setValue(self.current_l)
        self.l_spin.valueChanged.connect(self._set_l)
        quantum_layout.addRow("l (angular)", self.l_spin)

        self.m_spin = QtWidgets.QSpinBox()
        self.m_spin.setRange(-self.current_l, self.current_l)
        self.m_spin.setValue(self.current_m)
        self.m_spin.valueChanged.connect(self._set_m)
        quantum_layout.addRow("m (magnetic)", self.m_spin)

        layout.addWidget(three_d_group)
        layout.addWidget(slicing_group)
        layout.addWidget(quantum_group)

        # 2D controls (non-collapsible)
        two_d_group = QtWidgets.QGroupBox("2D Slice")
        two_d_layout = QtWidgets.QFormLayout(two_d_group)

        self.theta_spin = QtWidgets.QDoubleSpinBox()
        self.theta_spin.setRange(0.0, 180.0)
        self.theta_spin.setDecimals(1)
        self.theta_spin.valueChanged.connect(self._set_theta_phi)
        self.phi_spin = QtWidgets.QDoubleSpinBox()
        self.phi_spin.setRange(-180.0, 180.0)
        self.phi_spin.setDecimals(1)
        self.phi_spin.valueChanged.connect(self._set_theta_phi)
        angle_widget = QtWidgets.QWidget()
        angle_layout = QtWidgets.QHBoxLayout(angle_widget)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.addWidget(QtWidgets.QLabel("θ"))
        angle_layout.addWidget(self.theta_spin)
        angle_layout.addWidget(QtWidgets.QLabel("φ"))
        angle_layout.addWidget(self.phi_spin)
        two_d_layout.addRow("Angles (deg)", angle_widget)

        self.normal_spins = []
        normal_widget = QtWidgets.QWidget()
        normal_layout = QtWidgets.QHBoxLayout(normal_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        for label in ("nx", "ny", "nz"):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.valueChanged.connect(self._set_normal_components)
            self.normal_spins.append(spin)
            normal_layout.addWidget(QtWidgets.QLabel(label))
            normal_layout.addWidget(spin)
        two_d_layout.addRow("Normal", normal_widget)

        preset_widget = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        for text, vec in (("XY", (0, 0, 1)), ("YZ", (1, 0, 0)), ("XZ", (0, 1, 0))):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, v=vec: self._set_slice_normal(np.array(v, dtype=float)))
            preset_layout.addWidget(btn)
        two_d_layout.addRow("Presets", preset_widget)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-10.0, 10.0)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.valueChanged.connect(self._set_offset)
        two_d_layout.addRow("Offset (Å)", self.offset_spin)

        self.offset_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.offset_slider.setRange(-1000, 1000)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self._offset_slider_changed)
        two_d_layout.addRow(self.offset_slider)

        self.slice_scalar_combo = QtWidgets.QComboBox()
        self.slice_scalar_combo.addItem("Probability density", "probability")
        self.slice_scalar_combo.addItem("Cumulative probability", "cumulative")
        self.slice_scalar_combo.addItem("Probability amplitude", "amplitude")
        self.slice_scalar_combo.addItem("Phase (cyclic)", "phase")
        self.slice_scalar_combo.currentIndexChanged.connect(self._set_slice_scalar_mode)
        two_d_layout.addRow("Slice scalars", self.slice_scalar_combo)

        self.slice_cmap_combo = QtWidgets.QComboBox()
        self.slice_cmap_combo.addItems(
            ["viridis", "plasma", "inferno", "twilight", "coolwarm", "twilight_shifted", "magma", "cividis", "batlow", "bamako", "devon", "oslo", "lajolla", "hawaii", "davos", "vik", "broc", "cork", "roma", "tokyo"]
        )
        self.slice_cmap_combo.setCurrentText(self.current_cmap)
        self.slice_cmap_combo.currentTextChanged.connect(self._set_slice_cmap)
        two_d_layout.addRow("Colormap (2D)", self.slice_cmap_combo)

        self.slice_vmin_spin = QtWidgets.QDoubleSpinBox()
        self.slice_vmin_spin.setDecimals(4)
        self.slice_vmin_spin.setRange(-1e6, 1e6)
        self.slice_vmin_spin.setSpecialValueText("auto")
        self.slice_vmin_spin.setValue(0.0)
        self.slice_vmin_spin.valueChanged.connect(self._set_slice_vmin)
        self.slice_vmax_spin = QtWidgets.QDoubleSpinBox()
        self.slice_vmax_spin.setDecimals(4)
        self.slice_vmax_spin.setRange(-1e6, 1e6)
        self.slice_vmax_spin.setSpecialValueText("auto")
        self.slice_vmax_spin.setValue(0.0)
        self.slice_vmax_spin.valueChanged.connect(self._set_slice_vmax)
        self.slice_autoscale_btn = QtWidgets.QPushButton("Auto scale (2D)")
        self.slice_autoscale_btn.clicked.connect(self._slice_autoscale)
        minmax_widget = QtWidgets.QWidget()
        minmax_layout = QtWidgets.QHBoxLayout(minmax_widget)
        minmax_layout.setContentsMargins(0, 0, 0, 0)
        minmax_layout.addWidget(QtWidgets.QLabel("vmin"))
        minmax_layout.addWidget(self.slice_vmin_spin)
        minmax_layout.addWidget(QtWidgets.QLabel("vmax"))
        minmax_layout.addWidget(self.slice_vmax_spin)
        minmax_layout.addWidget(self.slice_autoscale_btn)
        two_d_layout.addRow("Scale (2D)", minmax_widget)

        self.contour_spin = QtWidgets.QSpinBox()
        self.contour_spin.setRange(2, 24)
        self.contour_spin.setValue(self.slice_contours_count)
        self.contour_spin.valueChanged.connect(self._set_contour_count)
        two_d_layout.addRow("Contours (2D)", self.contour_spin)

        self.slice_tf_btn = QtWidgets.QPushButton("Edit transfer function (2D)")
        self.slice_tf_btn.clicked.connect(self._open_slice_tf_dialog)
        two_d_layout.addRow(self.slice_tf_btn)

        layout.addWidget(two_d_group)

        layout.addStretch()
        return container

    def _set_quantum_numbers(self, n: int, l: int, m: int, trigger_render: bool = True) -> None:
        n_val = max(int(n), 1)
        l_val = min(max(int(l), 0), max(n_val - 1, 0))
        m_val = int(np.clip(int(m), -l_val, l_val))
        self.current_n, self.current_l, self.current_m = n_val, l_val, m_val
        self.n_spin.blockSignals(True)
        self.l_spin.blockSignals(True)
        self.m_spin.blockSignals(True)
        self.n_spin.setValue(n_val)
        self.l_spin.setRange(0, max(n_val - 1, 0))
        self.l_spin.setValue(l_val)
        self.m_spin.setRange(-l_val, l_val)
        self.m_spin.setValue(m_val)
        self.n_spin.blockSignals(False)
        self.l_spin.blockSignals(False)
        self.m_spin.blockSignals(False)
        if trigger_render:
            self._render_orbital()

    def _set_n(self, n: int) -> None:
        self._set_quantum_numbers(n, self.current_l, self.current_m)

    def _set_l(self, l: int) -> None:
        self._set_quantum_numbers(self.current_n, l, self.current_m)

    def _set_m(self, m: int) -> None:
        self._set_quantum_numbers(self.current_n, self.current_l, m)

    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        if mode == "wavefunction" and self.current_cmap == "viridis":
            self.current_cmap = "twilight"
            self.cmap_combo.setCurrentText("twilight")
        self._render_orbital()

    def _on_slice_mode_change(self) -> None:
        mode = self.slice_mode_combo.currentData() if hasattr(self, "slice_mode_combo") else "none"
        if self.slicing_controller:
            self.slicing_controller.set_mode(mode)

    def _reset_slicing(self) -> None:
        if self.slicing_controller:
            self.slicing_controller.reset()

    def _set_cmap(self, cmap: str) -> None:
        self.current_cmap = cmap
        self._render_orbital()

    def _set_iso_fraction(self, value: int) -> None:
        self.iso_fraction = value / 100.0
        self._render_orbital()

    def _set_representation(self, index: int | None = None) -> None:
        self.current_representation = self.representation_combo.currentData()
        self._update_visibility_controls()
        self._render_orbital()

    def _set_iso_surfaces_count(self, value: int) -> None:
        self.iso_surfaces_count = max(2, int(value))
        if self.iso_opacity_overrides:
            self.iso_opacity_overrides = self.iso_opacity_overrides[: self.iso_surfaces_count]
        self._render_orbital()

    def _set_contour_count(self, value: int) -> None:
        self.slice_contours_count = max(2, int(value))
        self._render_orbital()

    def _set_slice_cmap(self, cmap: str) -> None:
        self.slice_cmap = cmap
        self._render_orbital()

    def _set_slice_vmin(self, value: float) -> None:
        self.slice_vmin = None if value == 0 and self.slice_vmin_spin.specialValueText() else value
        self._render_orbital()

    def _set_slice_vmax(self, value: float) -> None:
        self.slice_vmax = None if value == 0 and self.slice_vmax_spin.specialValueText() else value
        self._render_orbital()

    def _slice_autoscale(self) -> None:
        if not self._slice_data_range:
            return
        vmin, vmax = self._slice_data_range
        if vmax <= vmin:
            return
        self._set_slice_range(vmin, vmax, render=True)

    def _set_slice_scalar_mode(self, index: int) -> None:
        self.slice_scalar_mode = self.slice_scalar_combo.currentData()
        self._render_orbital()

    def _on_slice_widget_changed(self, payload: dict) -> None:
        mode = payload.get("mode", "none")
        self.slice_mode = mode
        plane = payload.get("plane", {}) if isinstance(payload.get("plane"), dict) else {}
        box = payload.get("box", {}) if isinstance(payload.get("box"), dict) else {}
        if mode == "plane":
            origin = plane.get("origin")
            normal = plane.get("normal")
            if origin is not None and normal is not None:
                try:
                    self.slice_origin = np.array(origin, dtype=float)
                    self.slice_normal = np.array(normal, dtype=float)
                except Exception:
                    pass
        elif mode == "box":
            bounds = box.get("bounds")
            if bounds is not None:
                self.slice_box_bounds = tuple(bounds)
        else:
            self.slice_origin = None
            self.slice_box_bounds = None
        self._render_orbital()

    def _update_visibility_controls(self) -> None:
        is_volume = self.current_representation == "volume"
        for widget in (
            getattr(self, "iso_slider_label", None),
            getattr(self, "iso_slider", None),
            getattr(self, "iso_spin_label", None),
            getattr(self, "iso_surface_spin", None),
            getattr(self, "volume_tf_btn", None),
        ):
            if widget:
                widget.setVisible(is_volume)

    def _open_volume_tf_dialog(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("3D Transfer Function")
        layout = QtWidgets.QFormLayout(dialog)
        cmap_combo = QtWidgets.QComboBox(dialog)
        cmap_options = [self.cmap_combo.itemText(i) for i in range(self.cmap_combo.count())]
        cmap_combo.addItems(cmap_options)
        cmap_combo.setCurrentText(self.current_cmap)
        layout.addRow("Colormap", cmap_combo)

        sliders: list[QtWidgets.QSlider] = []
        count = max(self.iso_surface_spin.value(), 2)
        base_opacities = np.linspace(1.0, 0.2, count)
        if self.iso_opacity_overrides:
            for idx, val in enumerate(self.iso_opacity_overrides):
                if idx < count:
                    base_opacities[idx] = float(np.clip(val, 0.05, 1.0))
        for i in range(count):
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, dialog)
            slider.setRange(5, 100)
            slider.setValue(int(base_opacities[i] * 100))
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            sliders.append(slider)
            layout.addRow(f"Iso surface {i + 1}", slider)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.iso_opacity_overrides = [slider.value() / 100.0 for slider in sliders]
            new_cmap = cmap_combo.currentText()
            if new_cmap != self.current_cmap:
                self.current_cmap = new_cmap
                self.cmap_combo.setCurrentText(new_cmap)
            self._render_orbital()

    def _open_slice_tf_dialog(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("2D Transfer Function")
        layout = QtWidgets.QFormLayout(dialog)

        cmap_combo = QtWidgets.QComboBox(dialog)
        cmap_options = [self.slice_cmap_combo.itemText(i) for i in range(self.slice_cmap_combo.count())]
        cmap_combo.addItems(cmap_options)
        cmap_combo.setCurrentText(self.slice_cmap)
        layout.addRow("Colormap", cmap_combo)

        vmin_spin = QtWidgets.QDoubleSpinBox(dialog)
        vmin_spin.setDecimals(4)
        vmin_spin.setRange(-1e6, 1e6)
        vmin_spin.setSpecialValueText("auto")
        vmin_spin.setValue(self.slice_vmin if self.slice_vmin is not None else 0.0)
        vmax_spin = QtWidgets.QDoubleSpinBox(dialog)
        vmax_spin.setDecimals(4)
        vmax_spin.setRange(-1e6, 1e6)
        vmax_spin.setSpecialValueText("auto")
        vmax_spin.setValue(self.slice_vmax if self.slice_vmax is not None else 0.0)
        minmax_widget = QtWidgets.QWidget(dialog)
        minmax_layout = QtWidgets.QHBoxLayout(minmax_widget)
        minmax_layout.setContentsMargins(0, 0, 0, 0)
        minmax_layout.addWidget(QtWidgets.QLabel("vmin"))
        minmax_layout.addWidget(vmin_spin)
        minmax_layout.addWidget(QtWidgets.QLabel("vmax"))
        minmax_layout.addWidget(vmax_spin)
        layout.addRow("Scale", minmax_widget)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.slice_cmap = cmap_combo.currentText()
            self.slice_cmap_combo.setCurrentText(self.slice_cmap)
            vmin = None if vmin_spin.value() == 0.0 and vmin_spin.specialValueText() else vmin_spin.value()
            vmax = None if vmax_spin.value() == 0.0 and vmax_spin.specialValueText() else vmax_spin.value()
            self.slice_vmin = vmin
            self.slice_vmax = vmax
            self._render_orbital()

    def _set_theta_phi(self) -> None:
        if not self.theta_spin or not self.phi_spin:
            return
        theta = np.deg2rad(self.theta_spin.value())
        phi = np.deg2rad(self.phi_spin.value())
        normal = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=float,
        )
        self._set_slice_normal(normal, update_controls=False)

    def _set_normal_components(self) -> None:
        normal = np.array([spin.value() for spin in self.normal_spins], dtype=float)
        self._set_slice_normal(normal, update_controls=False)

    def _set_slice_normal(self, normal: np.ndarray, update_controls: bool = True, trigger_render: bool = True) -> None:
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([0.0, 0.0, 1.0])
        normed = normal / np.linalg.norm(normal)
        self.slice_normal = normed
        theta = float(np.rad2deg(np.arccos(np.clip(normed[2], -1.0, 1.0))))
        phi = float(np.rad2deg(np.arctan2(normed[1], normed[0])))
        if update_controls:
            if self.theta_spin and self.phi_spin:
                self.theta_spin.blockSignals(True)
                self.phi_spin.blockSignals(True)
                self.theta_spin.setValue(theta)
                self.phi_spin.setValue(phi)
                self.theta_spin.blockSignals(False)
                self.phi_spin.blockSignals(False)
            for spin, val in zip(self.normal_spins, normed, strict=False):
                spin.blockSignals(True)
                spin.setValue(float(val))
                spin.blockSignals(False)
        if trigger_render:
            self._render_orbital()

    def _set_offset(self, value: float) -> None:
        self.slice_offset = float(value)
        if self.offset_slider:
            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(int(self.slice_offset * 100))
            self.offset_slider.blockSignals(False)
        self._render_orbital()

    def _offset_slider_changed(self, value: int) -> None:
        if not self.offset_spin:
            return
        self.slice_offset = value / 100.0
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(value / 100.0)
        self.offset_spin.blockSignals(False)
        self._render_orbital()

    def _setup_lights(self) -> None:
        try:
            self.plotter.remove_all_lights()
            key = pv.Light(position=(3, 3, 6), focal_point=(0, 0, 0), color="white", intensity=0.9)
            fill = pv.Light(position=(-4, -2, 2), focal_point=(0, 0, 0), color="#94a3b8", intensity=0.6)
            rim = pv.Light(position=(0, -6, -3), focal_point=(0, 0, 0), color="#cbd5e1", intensity=0.35)
            for light in (key, fill, rim):
                self.plotter.add_light(light)
        except Exception:
            pass

    def _cache_key(self, n: int, l: int, m: int) -> tuple:
        return (
            self.current_symbol,
            self.current_mode,
            self.current_representation,
            n,
            l,
            m,
            round(self.iso_fraction, 3),
        )

    def _get_field(self, n_val: int, l_val: int, m_val: int):
        key = self._cache_key(n_val, l_val, m_val)
        if key in self._field_cache:
            field = self._field_cache.pop(key)
            self._field_cache[key] = field
            return field
        field = make_orbital_mesh(
            self.current_symbol,
            self.current_mode,
            n=n_val,
            l=l_val,
            m=m_val,
            representation=self.current_representation,
            iso_fraction=self.iso_fraction,
        )
        self._field_cache[key] = field
        if len(self._field_cache) > 24:
            self._field_cache.popitem(last=False)
        return field

    def _get_volume_field(self, n_val: int, l_val: int, m_val: int):
        key = (
            self.current_symbol,
            self.current_mode,
            "volume",
            n_val,
            l_val,
            m_val,
            round(self.iso_fraction, 3),
            "volume_scalar",
        )
        if key in self._field_cache:
            field = self._field_cache.pop(key)
            self._field_cache[key] = field
            return field
        field = make_orbital_mesh(
            self.current_symbol,
            self.current_mode,
            n=n_val,
            l=l_val,
            m=m_val,
            representation="volume",
            iso_fraction=self.iso_fraction,
        )
        self._field_cache[key] = field
        if len(self._field_cache) > 24:
            self._field_cache.popitem(last=False)
        return field

    def _render_orbital(self) -> None:
        camera_state = self.plotter.camera_position
        orbitals: list[tuple[int, int, int]] = []
        orbitals = [(self.current_n, self.current_l, self.current_m)]

        fields = []
        for n_val, l_val, m_val in orbitals:
            fields.append(self._get_field(n_val, l_val, m_val))
        # Dedicated volume data for slicing
        vol_field = self._get_volume_field(orbitals[0][0], orbitals[0][1], orbitals[0][2])
        self.plotter_frame.reset_scene()
        self.slice_view.clear()
        self.slice_plane_actor = None
        if self.camera_initialized and camera_state:
            try:
                self.plotter.camera_position = camera_state
            except Exception:
                pass
        if not fields:
            return
        clip_bounds = self.slice_box_bounds if self.slice_mode == "box" else None
        if clip_bounds and vol_field:
            try:
                vol_field = vol_field.copy()
                vol_field.dataset = vol_field.dataset.clip_box(clip_bounds, invert=False)
            except Exception:
                pass
        if self.current_representation == "volume":
            prob_arr = np.asarray(vol_field.dataset.get_array("probability"))
            vmax = float(np.nanmax(prob_arr)) if prob_arr.size else 1.0
            if vmax <= 0:
                vmax = 1.0
            clim = (0.0, vmax)
            label = "Probability"
        else:
            if fields[0].scalar_name == "phase":
                clim = (-np.pi, np.pi)
                label = "Phase (rad)"
            else:
                vmax = 1.0
                for f in fields:
                    arr = np.asarray(f.dataset[f.scalar_name]) if f.dataset.get_array(f.scalar_name) is not None else np.array([])
                    if arr.size:
                        vmax = max(vmax, float(np.nanmax(arr)))
                if vmax == 0:
                    vmax = 1.0
                clim = (0.0, vmax)
                label = "Amplitude"

        iso_text = ""
        if self.current_representation == "surface":
            last = fields[0]
            if last.cumulative_probability is not None:
                iso_text = f"Contains {last.cumulative_probability*100:.1f}% of probability"

        if self.current_representation == "volume":
            self._add_iso_surfaces(vol_field, clim)
        else:
            for field in fields:
                dataset = field.dataset
                if clip_bounds:
                    try:
                        dataset = dataset.clip_box(clip_bounds, invert=False)
                    except Exception:
                        dataset = field.dataset
                self.plotter.add_mesh(
                    dataset,
                    scalars=field.scalar_name,
                    cmap=self.current_cmap,
                    opacity=field.opacity,
                    specular=0.55,
                    specular_power=25.0,
                    diffuse=0.8,
                    ambient=0.25,
                    smooth_shading=True,
                    clim=clim,
                    show_scalar_bar=False,
                )
        main_title = f"{self.current_symbol} n={self.current_n}, l={self.current_l}, m={self.current_m}"
        if self.show_occupied:
            main_title += " (occupied)"
        self.plotter.add_text(
            main_title,
            font_size=12,
            color=self._text_color,
            shadow=self._text_shadow,
            name="main_title",
            position="upper_left",
        )
        view_label = "3D View (volume)" if self.current_representation == "volume" else "3D View (surface)"
        self.plotter.add_text(
            view_label,
            font_size=10,
            color=self._text_color,
            shadow=self._text_shadow,
            name="view_label",
            position="upper_right",
        )
        if iso_text:
            self.plotter.add_text(
                iso_text,
                font_size=10,
                color=self._text_color,
                shadow=self._text_shadow,
                name="iso_text",
                position="lower_left",
            )
        # 3D colorbar suppressed; handled in 2D slice
        self._apply_slice(fields, vol_field, clim)
        if not self.camera_initialized:
            self._autoscale()
            self.camera_initialized = True
        else:
            try:
                self.plotter.render()
            except Exception as exc:
                print(f"Render error: {exc}", file=sys.stderr)
        try:
            self.slice_view.render()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if obj is self.slice_container and event.type() == QtCore.QEvent.Resize:
            side = min(self.slice_container.width(), self.slice_container.height())
            if side > 0:
                self.slice_view.setFixedSize(side, side)
                QtCore.QTimer.singleShot(0, self._update_slice_scalebar)
        if obj is self.slice_colorbar:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._colorbar_drag_start = event.position().x()
                self.slice_colorbar.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                return True
            if event.type() == QtCore.QEvent.MouseButtonRelease and self._colorbar_drag_start is not None:
                end_x = event.position().x()
                self._apply_colorbar_drag(self._colorbar_drag_start, end_x)
                self._colorbar_drag_start = None
                self.slice_colorbar.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                return True
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                self._slice_autoscale()
                return True
        if obj is self.slice_view and event.type() == QtCore.QEvent.Wheel:
            QtCore.QTimer.singleShot(0, self._update_slice_scalebar)
            return False
        if obj is self.slice_view and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self._update_slice_scalebar)
            return False
        return super().eventFilter(obj, event)

    def _compute_iso_values(self, probability: np.ndarray, spacing: tuple[float, float, float], count: int) -> list[tuple[float, float]]:
        if count <= 0:
            return []
        probs_flat = probability.ravel(order="F")
        voxel_volume = float(np.prod(spacing))
        total_prob = float(probs_flat.sum() * voxel_volume)
        if total_prob <= 0:
            return []
        order = np.argsort(probs_flat)[::-1]
        sorted_probs = probs_flat[order]
        masses = sorted_probs * voxel_volume
        cumulative_mass = np.cumsum(masses)
        # Uniformly spaced cumulative probability targets in (0, 1)
        max_frac = float(np.clip(self.iso_fraction, 0.05, 0.99))
        fractions = np.linspace(max_frac / (count + 1), max_frac, count)
        iso_values: list[tuple[float, float]] = []
        for frac in fractions:
            target = frac * total_prob
            idx = int(np.searchsorted(cumulative_mass, target, side="left"))
            idx = min(idx, len(sorted_probs) - 1)
            iso_values.append((float(sorted_probs[idx]), float(frac)))
        iso_values.sort(key=lambda pair: pair[0], reverse=True)
        return iso_values

    def _iso_color_for_fraction(self, frac: float) -> str:
        # Map higher fraction (inner surfaces) to brighter hues using current 3D colormap
        frac_clamped = float(np.clip(frac, 0.0, 1.0))
        try:
            cmap_obj = self._get_cmap(self.current_cmap)
        except Exception:
            val = int(80 + 120 * frac_clamped)
            val = max(0, min(255, val))
            return f"#{val:02x}{val:02x}{val:02x}"
        r, g, b, _ = cmap_obj(frac_clamped)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def _add_iso_surfaces(self, volume_field, clim: tuple[float, float]) -> None:
        prob = np.asarray(volume_field.dataset.get_array("probability"))
        if prob.size == 0:
            print("No probability data for iso-surfaces", file=sys.stderr)
            return
        iso_levels = self._compute_iso_values(prob, volume_field.dataset.spacing, max(self.iso_surfaces_count, 2))
        if not iso_levels:
            return
        total_levels = len(iso_levels)
        base_opacities = np.linspace(1.0, 0.2, total_levels)
        if self.iso_opacity_overrides:
            for idx, val in enumerate(self.iso_opacity_overrides):
                if idx < total_levels:
                    base_opacities[idx] = float(np.clip(val, 0.05, 1.0))
        for idx, (iso_val, frac) in enumerate(iso_levels):
            alpha = float(np.clip(base_opacities[idx], 0.05, 1.0))
            try:
                surface = volume_field.dataset.contour(isosurfaces=[iso_val], scalars="probability")
                if surface.n_points == 0:
                    continue
                surface["cum_prob"] = np.full(surface.n_points, frac, dtype=float)
                surface.set_active_scalars("cum_prob")
                self.plotter.add_mesh(
                    surface,
                    scalars="cum_prob",
                    cmap=self.current_cmap,
                    clim=(0.0, 1.0),
                    opacity=alpha,
                    specular=0.4,
                    smooth_shading=True,
                    show_scalar_bar=False,
                    render_lines_as_tubes=False,
                    lighting=True,
                )
            except Exception as exc:
                print(f"Iso-surface error: {exc}", file=sys.stderr)

    def _render_slice_colorbar(self, cmap: str, clim: tuple[float, float], label: str) -> None:
        if not getattr(self, "slice_colorbar", None):
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm, colors, ticker
        except Exception as exc:
            self.slice_colorbar.setText(f"{label}: {clim[0]:.3g} - {clim[1]:.3g}")
            print(f"Colorbar render fallback: {exc}", file=sys.stderr)
            return
        try:
            norm = colors.Normalize(vmin=clim[0], vmax=clim[1])
            cmap_obj = self._get_cmap(cmap) if isinstance(cmap, str) else cmap
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            fig, ax = plt.subplots(figsize=(4.2, 0.5))
            colors_token = (self._theme_tokens or {}).get("colors", {})
            fig.patch.set_facecolor(colors_token.get("surfaceAlt", "#0f172a"))
            cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
            label_color = colors_token.get("text", "#e5e7eb")
            cbar.set_label(f"$\\mathrm{{{label}}}$", color=label_color, fontsize=8)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            cbar.formatter = formatter
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=7, colors=label_color)
            cbar.outline.set_edgecolor(label_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(label_color)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(buf.getvalue())
            if pixmap.height() > 100:
                pixmap = pixmap.scaledToHeight(100, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.slice_colorbar.setPixmap(pixmap)
            self.slice_colorbar.setMinimumHeight(pixmap.height() + 6)
            border_color = colors_token.get("border", "#1f2937")
            self.slice_colorbar.setStyleSheet(
                "background-color: {bg}; color: {text}; border: 1px solid {border}; padding: 6px;".format(
                    bg=colors_token.get("surfaceAlt", "#0f172a"),
                    text=label_color,
                    border=border_color,
                )
            )
            self.slice_colorbar.setToolTip("Drag on the colorbar to set vmin/vmax. Double-click to autoscale.")
            self._update_colorbar_range_label(clim[0], clim[1])
        except Exception as exc:
            self.slice_colorbar.setText(f"{label}: {clim[0]:.3g} - {clim[1]:.3g}")
            print(f"Colorbar render error: {exc}", file=sys.stderr)

    def _apply_colorbar_drag(self, start_x: float, end_x: float) -> None:
        if not self._slice_data_range:
            return
        width = max(self.slice_colorbar.width(), 1)
        data_min, data_max = self._slice_data_range
        if data_max <= data_min:
            return
        f0 = max(0.0, min(1.0, start_x / width))
        f1 = max(0.0, min(1.0, end_x / width))
        vmin = data_min + min(f0, f1) * (data_max - data_min)
        vmax = data_min + max(f0, f1) * (data_max - data_min)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._set_slice_range(vmin, vmax, render=True)

    def _update_colorbar_range_label(self, vmin: float, vmax: float) -> None:
        if getattr(self, "slice_colorbar_range", None):
            self.slice_colorbar_range.setText(f"Range: {vmin:.4g} – {vmax:.4g}")

    def _set_slice_range(self, vmin: float, vmax: float, render: bool = False) -> None:
        """Update slice range spinners/label and optionally re-render."""
        self.slice_vmin = vmin
        self.slice_vmax = vmax
        self.slice_vmin_spin.blockSignals(True)
        self.slice_vmax_spin.blockSignals(True)
        self.slice_vmin_spin.setValue(vmin)
        self.slice_vmax_spin.setValue(vmax)
        self.slice_vmin_spin.blockSignals(False)
        self.slice_vmax_spin.blockSignals(False)
        self._update_colorbar_range_label(vmin, vmax)
        if render:
            self._render_orbital()

    def _apply_slice(self, fields, volume_field, clim: tuple[float, float]) -> None:
        normal = self.slice_normal
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return
        normal = normal / norm
        volume_ds = volume_field.dataset
        if self.slice_mode == "box" and self.slice_box_bounds:
            try:
                volume_ds = volume_ds.clip_box(self.slice_box_bounds, invert=False)
            except Exception:
                pass
        try:
            bounds = volume_ds.bounds
        except Exception:
            bounds = None
        extent = None
        if bounds:
            extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            if extent > 0:
                if self.offset_spin:
                    self.offset_spin.blockSignals(True)
                    self.offset_spin.setRange(-extent * 0.75, extent * 0.75)
                    self.offset_spin.blockSignals(False)
                if self.offset_slider:
                    slider_range = max(int(extent * 100 * 0.75), 50)
                    self.offset_slider.blockSignals(True)
                    self.offset_slider.setRange(-slider_range, slider_range)
                    self.offset_slider.blockSignals(False)

        if self.slice_mode == "plane" and self.slice_origin is not None:
            origin = self.slice_origin
        else:
            origin = np.array(volume_ds.center) + normal * self.slice_offset
        plane_size = extent * 1.2 if extent and extent > 0 else 5.0

        try:
            if self.pref_show_slice_plane and self.slice_mode == "plane":
                plane_geom = pv.Plane(
                    center=origin,
                    direction=normal,
                    i_size=plane_size,
                    j_size=plane_size,
                    i_resolution=self.slice_plane_resolution,
                    j_resolution=self.slice_plane_resolution,
                )
                opacity_val = max(0.0, min(1.0, float(getattr(self, "pref_slice_plane_opacity", 0.35))))
                if self.slice_plane_actor:
                    try:
                        self.plotter.remove_actor(self.slice_plane_actor, reset_camera=False, render=False)
                    except Exception:
                        pass
                self.slice_plane_actor = self.plotter.add_mesh(
                    plane_geom,
                    color=self._slice_plane_color,
                    opacity=opacity_val,
                    lighting=False,
                    show_edges=True,
                    edge_color=self._slice_plane_edge_color,
                    smooth_shading=False,
                    show_scalar_bar=False,
                )
            elif self.slice_plane_actor:
                try:
                    self.plotter.remove_actor(self.slice_plane_actor, reset_camera=False, render=False)
                except Exception:
                    pass
                self.slice_plane_actor = None
        except Exception as exc:
            print(f"Slice plane render error: {exc}", file=sys.stderr)

        try:
            slice_res = 360
            sample_plane = pv.Plane(
                center=origin,
                direction=normal,
                i_size=plane_size,
                j_size=plane_size,
                i_resolution=slice_res,
                j_resolution=slice_res,
            )
            plane = sample_plane.sample(volume_ds)
            if plane.n_points == 0:
                return
            # Drop any extra arrays that may mismatch the plane resolution (normals/texture coords, etc.)
            for arr_name in ("Normals", "normals", "TextureCoordinates", "tcoords", "TCoords"):
                for pdata in (plane.point_data, plane.cell_data):
                    if arr_name in pdata:
                        try:
                            del pdata[arr_name]
                        except Exception:
                            pass

            def _ensure_array(name: str) -> np.ndarray:
                arr = plane.point_data.get(name)
                if arr is not None and len(arr) == plane.n_points:
                    return np.asarray(arr)
                arr = plane.get_array(name, preference="point")
                if arr is not None and len(arr) == plane.n_points:
                    plane.point_data[name] = arr
                    return np.asarray(arr)
                zeros = np.zeros(plane.n_points)
                plane.point_data[name] = zeros
                return zeros

            prob_vals = _ensure_array("probability")
            amp_vals = _ensure_array("amplitude")
            phase_vals = _ensure_array("phase")
            prob_vals = np.nan_to_num(prob_vals, nan=0.0, posinf=0.0, neginf=0.0)
            amp_vals = np.nan_to_num(amp_vals, nan=0.0, posinf=0.0, neginf=0.0)
            phase_vals = np.nan_to_num(phase_vals, nan=0.0, posinf=0.0, neginf=0.0)
            cum_vals = np.zeros_like(prob_vals, dtype=float)
            weights = np.clip(prob_vals, 0.0, None)
            total = float(weights.sum())
            if total > 0:
                sorted_idx = np.argsort(weights)
                csum = np.cumsum(weights[sorted_idx])
                cum_vals[sorted_idx] = np.clip(csum / total, 0.0, 1.0)
            plane["cum_prob"] = cum_vals

            self.slice_view.clear()
            slice_cmap = getattr(self, "slice_cmap", self.current_cmap)
            slice_clim = clim
            scalar_name = "probability"
            scalar_data = prob_vals
            label = "Probability"

            if self.slice_scalar_mode == "cumulative":
                scalar_name = "slice_scalar"
                scalar_data = cum_vals
                slice_clim = (0.0, 1.0)
                label = "Cumulative probability"
            elif self.slice_scalar_mode == "amplitude":
                scalar_name = "slice_scalar"
                scalar_data = amp_vals
                vmin = self.slice_vmin if self.slice_vmin is not None else (float(np.nanmin(amp_vals)) if amp_vals.size else 0.0)
                vmax = self.slice_vmax if self.slice_vmax is not None else (float(np.nanmax(amp_vals)) if amp_vals.size else 1.0)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                slice_clim = (vmin, vmax)
                label = "Amplitude"
            elif self.slice_scalar_mode == "phase":
                scalar_name = "slice_scalar"
                scalar_data = phase_vals
                slice_cmap = "twilight_shifted"
                slice_clim = (-np.pi, np.pi)
                label = "Phase"
            else:
                if self.slice_vmin is not None or self.slice_vmax is not None:
                    vmin = self.slice_vmin if self.slice_vmin is not None else float(np.nanmin(prob_vals))
                    vmax = self.slice_vmax if self.slice_vmax is not None else float(np.nanmax(prob_vals))
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    slice_clim = (vmin, vmax)

            plane["slice_scalar"] = scalar_data
            opacity_arr = 1.0 - np.clip(np.clip(cum_vals, 0.0, 1.0)**0.55, 0.0, 0.95)
            opacity_arr = np.clip(opacity_arr, 0.2, 1.0)

            self.slice_view.add_mesh(
                plane,
                scalars=scalar_name,
                cmap=slice_cmap,
                clim=slice_clim,
                show_scalar_bar=False,
                opacity=opacity_arr,
                lighting=False,
            )

            try:
                levels = np.linspace(
                    1.0 / (self.slice_contours_count + 1),
                    self.slice_contours_count / (self.slice_contours_count + 1),
                    self.slice_contours_count,
                )
                contours = plane.contour(isosurfaces=list(levels), scalars="cum_prob")
                if contours.n_points > 0:
                    self.slice_view.add_mesh(
                        contours,
                        color="#e5e7eb",
                        line_width=1.0,
                        show_scalar_bar=False,
                        lighting=False,
                        render_lines_as_tubes=False,
                    )
            except Exception as exc:
                print(f"Contour render error: {exc}", file=sys.stderr)

            try:
                bounds = plane.bounds
                self._last_slice_bounds = bounds
                self._draw_slice_scalebar(bounds)
            except Exception:
                pass

            self.slice_view.add_text(
                "2D Slice",
                position="upper_right",
                font_size=10,
                color=self._text_color,
                shadow=self._text_shadow,
                name="slice_label",
            )

            pos = origin + normal * (extent if extent and extent > 0 else 5.0)
            up = np.array([0, 0, 1])
            if abs(np.dot(up, normal)) > 0.9:
                up = np.array([0, 1, 0])
            self.slice_view.camera_position = [pos.tolist(), origin.tolist(), up.tolist()]
            if extent and extent > 0:
                try:
                    self.slice_view.camera.parallel_scale = extent * 0.6
                except Exception:
                    pass
            try:
                self.slice_view.enable_parallel_projection()
            except Exception:
                pass
            try:
                self.slice_view.renderer.interactive = False
            except Exception:
                pass

            # cache data range for colorbar interactions
            try:
                data_min = float(np.nanmin(scalar_data))
                data_max = float(np.nanmax(scalar_data))
                self._slice_data_range = (data_min, data_max)
            except Exception:
                self._slice_data_range = slice_clim
                data_min, data_max = slice_clim
            self._render_slice_colorbar(slice_cmap, slice_clim, label)
            if not self._slice_autoscale_done and self.slice_vmin is None and self.slice_vmax is None:
                self._set_slice_range(data_min, data_max, render=False)
                self._slice_autoscale_done = True
        except Exception as exc:
            print(f"Slice plane error: {exc}", file=sys.stderr)

    def _ensure_scale_bar_actor(self) -> None:
        if self._scale_bar_actor_2d is not None:
            return
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(2)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        mapper.SetTransformCoordinate(coord)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2.0)
        self.slice_view.renderer.AddActor2D(actor)
        self._scale_bar_actor_2d = actor
        self._scale_bar_points = points
        self._scale_bar_poly = poly

    def _draw_slice_scalebar(self, bounds: tuple[float, float, float, float, float, float] | None = None) -> None:
        if bounds is None:
            bounds = self._last_slice_bounds
        if bounds is None:
            return
        try:
            self._ensure_scale_bar_actor()
            render_window = getattr(self.slice_view, "render_window", None)
            if render_window:
                width_px, height_px = render_window.GetSize()
            else:
                width_px, height_px = self.slice_view.width(), self.slice_view.height()
            if height_px <= 0:
                return
            try:
                visible_scale = float(self.slice_view.camera.parallel_scale)
            except Exception:
                visible_scale = None
            if not visible_scale or visible_scale <= 0:
                return
            world_height = 2 * visible_scale
            world_per_px = world_height / height_px
            scalebar_length = world_per_px * self._scale_bar_length_px
            start_x, start_y = self._scale_bar_margin
            end_x = start_x + self._scale_bar_length_px
            end_y = start_y
            self._scale_bar_points.SetPoint(0, start_x, start_y, 0)
            self._scale_bar_points.SetPoint(1, end_x, end_y, 0)
            self._scale_bar_points.Modified()
            self._scale_bar_poly.Modified()
            color = QtGui.QColor(self._text_color)
            self._scale_bar_actor_2d.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())
            if self._scale_bar_text:
                try:
                    self.slice_view.remove_actor(self._scale_bar_text)
                except Exception:
                    pass
            self._scale_bar_text = self.slice_view.add_text(
                f"{scalebar_length:.2f} A",
                position="lower_left",
                font_size=10,
                color=self._text_color,
                shadow=self._text_shadow,
                name="scale_bar",
            )
        except Exception as exc:
            print(f"Scalebar render error: {exc}", file=sys.stderr)

    def _update_slice_scalebar(self) -> None:
        self._draw_slice_scalebar(self._last_slice_bounds)

    def _autoscale(self) -> None:
        self.plotter.reset_camera()
        self.camera_initialized = True

    def _open_preferences(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Preferences")
        layout = QtWidgets.QFormLayout(dialog)

        grid_check = QtWidgets.QCheckBox("Show 3D grid")
        grid_check.setChecked(self.pref_show_grid)
        layout.addRow(grid_check)

        slice_plane_check = QtWidgets.QCheckBox("Show slice plane")
        slice_plane_check.setChecked(self.pref_show_slice_plane)
        layout.addRow(slice_plane_check)

        opacity_spin = QtWidgets.QDoubleSpinBox()
        opacity_spin.setRange(0.0, 1.0)
        opacity_spin.setSingleStep(0.05)
        opacity_spin.setValue(float(self.pref_slice_plane_opacity))
        layout.addRow("Slice plane opacity", opacity_spin)

        save_btn = QtWidgets.QPushButton("Export profile (yaml)")
        load_btn = QtWidgets.QPushButton("Load profile (yaml)")
        save_btn.clicked.connect(lambda: self._export_profile())
        load_btn.clicked.connect(lambda: self._import_profile())
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(save_btn)
        hl.addWidget(load_btn)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(hl)
        layout.addRow(wrapper)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.pref_show_grid = grid_check.isChecked()
            self.pref_show_slice_plane = slice_plane_check.isChecked()
            self.pref_slice_plane_opacity = opacity_spin.value()
            try:
                self.plotter_frame.set_show_bounds(self.pref_show_grid)
            except Exception:
                pass
            self._render_orbital()

    def _profile_dict(self) -> dict:
        return {
            "show_grid": self.pref_show_grid,
            "show_slice_plane": self.pref_show_slice_plane,
            "slice_plane_opacity": float(self.pref_slice_plane_opacity),
        }

    def _export_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export profile", "profile.yaml", "YAML Files (*.yaml)")
        if not path:
            return
        profile = self._profile_dict()
        try:
            try:
                import yaml  # type: ignore

                data = yaml.safe_dump(profile)
            except Exception:
                lines = []
                for k, v in profile.items():
                    lines.append(f"{k}: {v}")
                data = "\n".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception as exc:
            print(f"Profile export failed: {exc}", file=sys.stderr)

    def _import_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load profile", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        profile = None
        try:
            try:
                import yaml  # type: ignore

                with open(path, "r", encoding="utf-8") as f:
                    profile = yaml.safe_load(f)
            except Exception:
                profile = {}
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            profile[k.strip()] = v.strip()
        except Exception as exc:
            print(f"Profile load failed: {exc}", file=sys.stderr)
            return
        if not isinstance(profile, dict):
            return
        self.pref_show_grid = bool(profile.get("show_grid", self.pref_show_grid))
        self.pref_show_slice_plane = bool(profile.get("show_slice_plane", self.pref_show_slice_plane))
        self.pref_slice_plane_opacity = float(profile.get("slice_plane_opacity", self.pref_slice_plane_opacity))
        try:
            self.plotter_frame.set_show_bounds(self.pref_show_grid)
        except Exception:
            pass
        self._render_orbital()

    def cleanup(self) -> None:
        self.plotter_frame.cleanup()
        try:
            self.slice_view.close()
        except Exception:
            pass
