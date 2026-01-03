from __future__ import annotations

import io
import sys
import os
import shutil
import subprocess
import threading
import math
import json
import qtawesome as qta

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from PySide6 import QtCore, QtGui, QtWidgets, QtPrintSupport
import pyvista as pv
import cmcrameri.cm as cmc
import matplotlib.cm as mpl_cm
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
from periodic_table_cli.cli import ChartConfig, DataConfig, load_data
from periodic_table_cli.chart_processor import ChartProcessor
from periodic_table_cli.data_processor import DataProcessor
from orbsim.widgets import CollapsibleGroup, DropPlotter, PeriodicTableWidget
from orbsim.theming.apply_theme import apply_theme
from orbsim.theming.theme_tokens import THEME_TOKENS, get_theme_tokens

ureg = UnitRegistry()
Q_ = ureg.Quantity

def _resolve_cmap(name: str):
    try:
        return mpl_cm.get_cmap(name)
    except Exception:
        pass
    try:
        return getattr(cmc, name)
    except Exception:
        pass
    try:
        return mpl_cm.get_cmap(f"cmc.{name}")
    except Exception:
        pass
    return mpl_cm.get_cmap("viridis")


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


class AtomicOrbitalTab(QtWidgets.QWidget):
    """Main tab that wires together 3D/2D orbital rendering, controls, and preferences."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
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
        self.show_occupied = False
        self._field_cache: OrderedDict[tuple, object] = OrderedDict()
        self.slice_normal = np.array([0.0, 0.0, 1.0])
        self.slice_offset = 0.0
        self.slice_plane_actor = None
        self.offset_spin: QtWidgets.QDoubleSpinBox | None = None
        self.offset_slider: QtWidgets.QSlider | None = None
        self.theta_spin: QtWidgets.QDoubleSpinBox | None = None
        self.phi_spin: QtWidgets.QDoubleSpinBox | None = None
        self.normal_spins: list[QtWidgets.QDoubleSpinBox] = []

        self.plotter_frame = DropPlotter()
        try:
            self.plotter_frame.colorbar.hide()
            self.plotter_frame.colorbar.setFixedWidth(0)
        except Exception:
            pass
        self.plotter = self.plotter_frame.plotter
        self.plotter.set_background("#111827")
        try:
            self.plotter.enable_anti_aliasing()
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass
        self._setup_lights()
        self.slice_view = QtInteractor(self)
        self.slice_view.set_background("#0f172a")
        self.slice_view.enable_anti_aliasing()
        try:
            self.slice_view.enable_parallel_projection()
            self.slice_view.enable_image_style()
        except Exception:
            pass
        self.slice_container = QtWidgets.QWidget()
        slice_layout = QtWidgets.QVBoxLayout(self.slice_container)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        slice_layout.addWidget(self.slice_view)
        self.slice_colorbar = QtWidgets.QLabel()
        self.slice_colorbar.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop)
        self.slice_colorbar.setMinimumHeight(80)
        self.slice_colorbar.setStyleSheet("background-color: #0f172a; border: 1px solid #1f2937; padding: 4px;")
        slice_layout.addWidget(self.slice_colorbar)
        self.slice_colorbar_range = QtWidgets.QLabel("Range: auto")
        self.slice_colorbar_range.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.slice_colorbar_range.setStyleSheet("color: #e5e7eb; padding: 2px; font-size: 10px;")
        slice_layout.addWidget(self.slice_colorbar_range)
        self.slice_container.setMinimumSize(320, 320)
        self.slice_container.installEventFilter(self)
        self.slice_colorbar.installEventFilter(self)
        self.slice_view.installEventFilter(self)
        self._colorbar_drag_start: float | None = None
        self._slice_data_range: tuple[float, float] | None = None
        self._slice_autoscale_done: bool = False
        self._scale_bar_actor = None
        self._scale_bar_text = None
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

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.plotter_frame, 3)
        layout.addWidget(self.slice_container, 3)
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_scroll.setWidget(self.controls)
        controls_scroll.setMinimumWidth(520)
        controls_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        layout.addWidget(controls_scroll, 2)
        self.plotter_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.slice_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self._render_orbital()

    def _get_cmap(self, name: str):
        return _resolve_cmap(name)

    def _build_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setMinimumWidth(520)
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        layout = QtWidgets.QVBoxLayout(container)

        # 3D controls (non-collapsible)
        three_d_group = QtWidgets.QGroupBox("3D View")
        three_d_layout = QtWidgets.QFormLayout(three_d_group)

        self.symbol_combo = QtWidgets.QComboBox()
        for sym in sorted(_ATOMIC_NUMBER.keys(), key=lambda s: _ATOMIC_NUMBER[s]):
            self.symbol_combo.addItem(sym)
        self.symbol_combo.setCurrentText(self.current_symbol)
        self.symbol_combo.currentTextChanged.connect(self._update_symbol)
        three_d_layout.addRow("Element", self.symbol_combo)

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

        measurement_group = QtWidgets.QGroupBox("Measurements")
        measure_layout = QtWidgets.QVBoxLayout(measurement_group)
        self.distance_btn = QtWidgets.QPushButton("Measure distance")
        self.distance_btn.clicked.connect(self._start_distance_measurement)
        self.angle_btn = QtWidgets.QPushButton("Measure angle")
        self.angle_btn.clicked.connect(self._start_angle_measurement)
        self.clear_measure_btn = QtWidgets.QPushButton("Clear measurements")
        self.clear_measure_btn.clicked.connect(self._clear_measurements)
        measure_layout.addWidget(self.distance_btn)
        measure_layout.addWidget(self.angle_btn)
        measure_layout.addWidget(self.clear_measure_btn)
        layout.addWidget(measurement_group)
        layout.addStretch()
        return container

    def _update_symbol_from_element(self, element) -> None:
        self._update_symbol(element.symbol)

    def _update_symbol(self, symbol: str) -> None:
        self.current_symbol = symbol
        default_n, default_l = default_quantum_numbers(symbol)
        self._set_quantum_numbers(default_n, default_l, 0, trigger_render=False)
        self._render_orbital()

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

    def _start_distance_measurement(self) -> None:
        self.plotter_frame.start_distance_measurement()

    def _start_angle_measurement(self) -> None:
        self.plotter_frame.start_angle_measurement()

    def _clear_measurements(self) -> None:
        self.plotter_frame.stop_measurements()

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
                self.plotter.add_mesh(
                    field.dataset,
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
        self.plotter.add_text(main_title, font_size=12, color="white", name="main_title", position="upper_left")
        view_label = "3D View (volume)" if self.current_representation == "volume" else "3D View (surface)"
        self.plotter.add_text(view_label, font_size=10, color="white", name="view_label", position="upper_right")
        if iso_text:
            self.plotter.add_text(iso_text, font_size=10, color="white", name="iso_text", position="lower_left")
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
            fig.patch.set_facecolor("#0f172a")
            cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
            cbar.set_label(f"$\\mathrm{{{label}}}$", color="#e5e7eb", fontsize=8)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            cbar.formatter = formatter
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=7, colors="#e5e7eb")
            cbar.outline.set_edgecolor("#e5e7eb")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e5e7eb")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(buf.getvalue())
            if pixmap.height() > 100:
                pixmap = pixmap.scaledToHeight(100, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.slice_colorbar.setPixmap(pixmap)
            self.slice_colorbar.setMinimumHeight(pixmap.height() + 6)
            self.slice_colorbar.setStyleSheet(
                "background-color: #0f172a; color: #e5e7eb; border: 1px solid #1f2937; padding: 6px;"
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

        origin = np.array(volume_ds.center) + normal * self.slice_offset
        plane_size = extent * 1.2 if extent and extent > 0 else 5.0

        try:
            if self.pref_show_slice_plane:
                plane_geom = pv.Plane(
                    center=origin, direction=normal, i_size=plane_size, j_size=plane_size, i_resolution=2, j_resolution=2
                )
                opacity_val = max(0.0, min(1.0, float(getattr(self, "pref_slice_plane_opacity", 0.35))))
                if self.slice_plane_actor:
                    try:
                        self.plotter.remove_actor(self.slice_plane_actor, reset_camera=False, render=False)
                    except Exception:
                        pass
                self.slice_plane_actor = self.plotter.add_mesh(
                    plane_geom,
                    color="#cbd5e1",
                    opacity=opacity_val,
                    lighting=False,
                    show_edges=True,
                    edge_color="#94a3b8",
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

            self.slice_view.add_text("2D Slice", position="upper_right", font_size=10, color="white", name="slice_label")

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

    def _draw_slice_scalebar(self, bounds: tuple[float, float, float, float, float, float] | None = None) -> None:
        if bounds is None:
            bounds = self._last_slice_bounds
        if bounds is None:
            return
        try:
            width = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            visible_scale = None
            try:
                visible_scale = float(self.slice_view.camera.parallel_scale)
            except Exception:
                visible_scale = None
            scalebar_length = visible_scale * 0.5 if visible_scale and visible_scale > 0 else (width / 4 if width > 0 else 1.0)
            start = (bounds[0] + scalebar_length * 0.15, bounds[2] + scalebar_length * 0.15, bounds[4])
            end = (start[0] + scalebar_length, start[1], start[2])
            if self._scale_bar_actor:
                try:
                    self.slice_view.remove_actor(self._scale_bar_actor)
                except Exception:
                    pass
            if self._scale_bar_text:
                try:
                    self.slice_view.remove_actor(self._scale_bar_text)
                except Exception:
                    pass
            self._scale_bar_actor = self.slice_view.add_lines(np.array([start, end]), color="white", width=3)
            self._scale_bar_text = self.slice_view.add_text(
                f"{scalebar_length:.2f} A",
                position="lower_left",
                font_size=10,
                color="white",
                name="scale_bar",
            )
        except Exception as exc:
            print(f"Scalebar render error: {exc}", file=sys.stderr)

    def _update_slice_scalebar(self) -> None:
        try:
            if self._scale_bar_actor:
                self.slice_view.remove_actor(self._scale_bar_actor)
                self._scale_bar_actor = None
            if self._scale_bar_text:
                self.slice_view.remove_actor(self._scale_bar_text)
                self._scale_bar_text = None
        except Exception:
            pass
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


class BondingOrbitalTab(AtomicOrbitalTab):
    """Tab for positioning atomic orbitals and deriving bonding/antibonding hybrids."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.orbitals: list[PositionedOrbital] = [
            PositionedOrbital("H", 1, 0, 0, np.array([-1.2, 0.0, 0.0])),
            PositionedOrbital("H", 1, 0, 0, np.array([1.2, 0.0, 0.0])),
        ]
        self.hybrid_grids: dict[str, pv.ImageData] = {}
        self._hybrid_meta: dict[str, object] | None = None
        self._orbitals_version = len(self.orbitals)
        self.grid_resolution = 120
        self.selected_hybrid = "bonding"
        self.base_grid: pv.ImageData | None = None
        self.mix_fraction = 1.0
        self.show_nuclei = True
        self.nucleus_scale = 1.0
        self.animation_timer: QtCore.QTimer | None = None
        self.animation_running = False
        self.animation_step = 4
        # fallback defaults in case parent init short-circuits
        self.pref_show_slice_plane = True
        self.pref_slice_plane_opacity = 0.08
        self.slice_scalar_mode = "probability"
        self.slice_vmin = None
        self.slice_vmax = None
        super().__init__(parent)
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self._advance_animation)
        self.hybrid_status.setText("Computed default H2-like hybrids. Adjust orbitals and recompute as needed.")
        self._closed = False

    def _build_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        orbitals_group = QtWidgets.QGroupBox("Atomic orbitals")
        orbitals_layout = QtWidgets.QVBoxLayout(orbitals_group)
        self.orbitals_table = QtWidgets.QTableWidget(0, 6)
        self.orbitals_table.setHorizontalHeaderLabels(["Show", "Element", "n", "l", "m", "Position (x, y, z)"])
        self.orbitals_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.orbitals_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.orbitals_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.orbitals_table.verticalHeader().setVisible(False)
        self.orbitals_table.horizontalHeader().setStretchLastSection(True)
        self.orbitals_table.doubleClicked.connect(lambda _: self._edit_selected_orbital())
        self.orbitals_table.itemChanged.connect(self._handle_orbital_checkbox)
        orbitals_layout.addWidget(self.orbitals_table)

        orb_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add orbital")
        add_btn.clicked.connect(self._add_orbital)
        edit_btn = QtWidgets.QPushButton("Edit selected")
        edit_btn.clicked.connect(self._edit_selected_orbital)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_orbital)
        clear_btn = QtWidgets.QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_orbitals)
        orb_buttons.addWidget(add_btn)
        orb_buttons.addWidget(edit_btn)
        orb_buttons.addWidget(remove_btn)
        orb_buttons.addWidget(clear_btn)
        orbitals_layout.addLayout(orb_buttons)

        example_layout = QtWidgets.QHBoxLayout()
        self.examples_combo = QtWidgets.QComboBox()
        for name in EXAMPLE_MOLECULES:
            self.examples_combo.addItem(name)
        load_example_btn = QtWidgets.QPushButton("Load example")
        load_example_btn.clicked.connect(self._load_example)
        example_layout.addWidget(self.examples_combo)
        example_layout.addWidget(load_example_btn)
        orbitals_layout.addLayout(example_layout)

        self.nuclei_check = QtWidgets.QCheckBox("Show nuclei (opaque spheres)")
        self.nuclei_check.setChecked(self.show_nuclei)
        self.nuclei_check.stateChanged.connect(lambda state: self._toggle_nuclei(bool(state)))
        orbitals_layout.addWidget(self.nuclei_check)

        nucleus_size_layout = QtWidgets.QHBoxLayout()
        nucleus_size_layout.addWidget(QtWidgets.QLabel("Nucleus radius scale"))
        self.nucleus_scale_spin = QtWidgets.QDoubleSpinBox()
        self.nucleus_scale_spin.setRange(0.2, 3.0)
        self.nucleus_scale_spin.setSingleStep(0.05)
        self.nucleus_scale_spin.setValue(self.nucleus_scale)
        self.nucleus_scale_spin.valueChanged.connect(self._set_nucleus_scale)
        nucleus_size_layout.addWidget(self.nucleus_scale_spin)
        orbitals_layout.addLayout(nucleus_size_layout)

        compute_form = QtWidgets.QFormLayout()
        self.resolution_spin = QtWidgets.QSpinBox()
        self.resolution_spin.setRange(50, 200)
        self.resolution_spin.setValue(self.grid_resolution)
        self.resolution_spin.valueChanged.connect(self._set_grid_resolution)
        compute_form.addRow("Grid resolution", self.resolution_spin)

        self.hybrid_combo = QtWidgets.QComboBox()
        self.hybrid_combo.addItem("Bonding (in-phase)", "bonding")
        self.hybrid_combo.addItem("Antibonding (alternating)", "antibonding")
        self.hybrid_combo.currentIndexChanged.connect(lambda _: self._render_orbital())
        compute_form.addRow("Hybrid selection", self.hybrid_combo)

        self.mix_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(int(self.mix_fraction * 100))
        self.mix_slider.valueChanged.connect(self._set_mix_fraction)
        self.mix_label = QtWidgets.QLabel(self._mix_label_text())
        mix_widget = QtWidgets.QWidget()
        mix_layout = QtWidgets.QHBoxLayout(mix_widget)
        mix_layout.setContentsMargins(0, 0, 0, 0)
        mix_layout.addWidget(self.mix_slider)
        mix_layout.addWidget(self.mix_label)
        compute_form.addRow("Hybrid mix", mix_widget)

        orbitals_layout.addLayout(compute_form)

        anim_layout = QtWidgets.QHBoxLayout()
        self.play_anim_btn = QtWidgets.QPushButton("Play mix animation")
        self.play_anim_btn.clicked.connect(self._toggle_animation)
        export_anim_btn = QtWidgets.QPushButton("Export animation (GIF)")
        export_anim_btn.clicked.connect(self._export_animation)
        anim_layout.addWidget(self.play_anim_btn)
        anim_layout.addWidget(export_anim_btn)
        orbitals_layout.addLayout(anim_layout)

        self.hybrid_status = QtWidgets.QLabel("Place orbitals, then compute hybrids.")
        self.hybrid_status.setWordWrap(True)
        orbitals_layout.addWidget(self.hybrid_status)
        layout.addWidget(orbitals_group)

        three_d_group = QtWidgets.QGroupBox("3D View")
        three_d_layout = QtWidgets.QFormLayout(three_d_group)

        self.representation_combo = QtWidgets.QComboBox()
        self.representation_combo.addItem("Surface (iso-probability)", "surface")
        self.representation_combo.addItem("Volume (semi-transparent)", "volume")
        self.representation_combo.currentIndexChanged.connect(self._set_representation)
        three_d_layout.addRow("Type", self.representation_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["amplitude", "wavefunction"])
        self.mode_combo.setCurrentText(self.current_mode)
        self.mode_combo.currentTextChanged.connect(self._set_mode)
        three_d_layout.addRow("Mode", self.mode_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(
            ["viridis", "plasma", "inferno", "twilight", "coolwarm", "magma", "cividis", "batlow", "bamako", "devon", "oslo", "lajolla", "hawaii", "davos", "vik", "broc", "cork", "roma", "tokyo"]
        )
        self.cmap_combo.setCurrentText(self.current_cmap)
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

        layout.addWidget(three_d_group)

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
        angle_layout.addWidget(QtWidgets.QLabel("I,"))
        angle_layout.addWidget(self.theta_spin)
        angle_layout.addWidget(QtWidgets.QLabel("I+"))
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
        two_d_layout.addRow("Offset (A.)", self.offset_spin)

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

        measurement_group = QtWidgets.QGroupBox("Measurements")
        measure_layout = QtWidgets.QVBoxLayout(measurement_group)
        self.distance_btn = QtWidgets.QPushButton("Measure distance")
        self.distance_btn.clicked.connect(self._start_distance_measurement)
        self.angle_btn = QtWidgets.QPushButton("Measure angle")
        self.angle_btn.clicked.connect(self._start_angle_measurement)
        self.clear_measure_btn = QtWidgets.QPushButton("Clear measurements")
        self.clear_measure_btn.clicked.connect(self._clear_measurements)
        measure_layout.addWidget(self.distance_btn)
        measure_layout.addWidget(self.angle_btn)
        measure_layout.addWidget(self.clear_measure_btn)
        layout.addWidget(measurement_group)
        layout.addStretch()
        self._refresh_orbital_table()
        return container

    def _refresh_orbital_table(self) -> None:
        self.orbitals_table.setRowCount(len(self.orbitals))
        for row, orb in enumerate(self.orbitals):
            position_str = f"{orb.position[0]:.2f}, {orb.position[1]:.2f}, {orb.position[2]:.2f}"
            show_item = QtWidgets.QTableWidgetItem()
            show_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            show_item.setCheckState(QtCore.Qt.CheckState.Checked if orb.visible else QtCore.Qt.CheckState.Unchecked)
            self.orbitals_table.setItem(row, 0, show_item)
            values = [orb.symbol, str(orb.n), str(orb.l), str(orb.m), position_str]
            for col, value in enumerate(values, start=1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
                )
                self.orbitals_table.setItem(row, col, item)
        self.orbitals_table.resizeColumnsToContents()

    def _selected_hybrid_key(self) -> str:
        key = self.hybrid_combo.currentData()
        return key if key else "bonding"

    def _set_grid_resolution(self, value: int) -> None:
        self.grid_resolution = max(int(value), 20)
        self._hybrid_meta = None
        self.hybrid_grids = {}
        self.base_grid = None
        self.camera_initialized = False
        self.hybrid_status.setText("Grid updated. Recompute hybrids to refresh.")

    def _mark_hybrids_stale(self) -> None:
        self.hybrid_grids = {}
        self._hybrid_meta = None
        self.camera_initialized = False
        self.hybrid_status.setText("Orbital list changed. Recompute hybrids.")
        self.base_grid = None

    def _mix_label_text(self) -> str:
        pct = int(round(self.mix_fraction * 100))
        return f"{pct}% hybrid (0% = atomic)"

    def _set_mix_fraction(self, value: int) -> None:
        self.mix_fraction = float(np.clip(value, 0, 100)) / 100.0
        if getattr(self, "mix_label", None):
            self.mix_label.setText(self._mix_label_text())
        self._render_orbital()

    def _toggle_nuclei(self, enabled: bool) -> None:
        self.show_nuclei = bool(enabled)
        self._render_orbital()

    def _set_nucleus_scale(self, value: float) -> None:
        self.nucleus_scale = float(max(value, 0.05))
        self._render_orbital()

    def _add_orbital(self) -> None:
        dialog = AddOrbitalDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.orbitals.append(dialog.result_orbital())
            self._orbitals_version += 1
            self._refresh_orbital_table()
            self._mark_hybrids_stale()
            self._compute_hybrid_grids()
            self._render_orbital(reuse_camera=True, reuse_slice=True)

    def _remove_selected_orbital(self) -> None:
        selected_rows = self.orbitals_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.orbitals):
            del self.orbitals[row]
            self._orbitals_version += 1
            self._refresh_orbital_table()
            self._mark_hybrids_stale()
            self._compute_and_render()

    def _clear_orbitals(self) -> None:
        if not self.orbitals:
            return
        self.orbitals.clear()
        self._orbitals_version += 1
        self._refresh_orbital_table()
        self._mark_hybrids_stale()
        self.plotter_frame.reset_scene()
        self.slice_view.clear()

    def _handle_orbital_checkbox(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self.orbitals)):
            return
        self.orbitals[row].visible = item.checkState() == QtCore.Qt.CheckState.Checked
        self._orbitals_version += 1
        self._mark_hybrids_stale()
        self._compute_hybrid_grids()
        self._render_orbital(reuse_camera=True, reuse_slice=True)

    def _edit_selected_orbital(self) -> None:
        selected_rows = self.orbitals_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if not (0 <= row < len(self.orbitals)):
            return
        dialog = AddOrbitalDialog(self)
        dialog.populate_from(self.orbitals[row])
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.orbitals[row] = dialog.result_orbital()
            self._orbitals_version += 1
            self._refresh_orbital_table()
            self._mark_hybrids_stale()
            self._compute_hybrid_grids()
            self._render_orbital(reuse_camera=True, reuse_slice=True)

    def _load_example(self) -> None:
        name = self.examples_combo.currentText() if getattr(self, "examples_combo", None) else ""
        orbs = EXAMPLE_MOLECULES.get(name)
        if not orbs:
            return
        self.orbitals = [
            PositionedOrbital(o.symbol, o.n, o.l, o.m, np.array(o.position, dtype=float)) for o in orbs
        ]
        self._orbitals_version += 1
        self._refresh_orbital_table()
        self._mark_hybrids_stale()
        self.mix_slider.setValue(100)
        self._compute_hybrid_grids()
        self._render_orbital(reuse_camera=True, reuse_slice=True)

    def _sampling_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.orbitals:
            span = 4.0
            axis = np.linspace(-span, span, self.grid_resolution)
            return axis, axis, axis
        mins = []
        maxs = []
        for orb in self.orbitals:
            span = 3.0 * max(orb.n, 1)
            mins.append(np.asarray(orb.position, dtype=float) - span)
            maxs.append(np.asarray(orb.position, dtype=float) + span)
        min_bounds = np.min(np.vstack(mins), axis=0) - 1.5
        max_bounds = np.max(np.vstack(maxs), axis=0) + 1.5
        axes = []
        for lo, hi in zip(min_bounds, max_bounds, strict=False):
            if np.isclose(lo, hi):
                lo -= 1.0
                hi += 1.0
            axes.append(np.linspace(lo, hi, self.grid_resolution))
        return axes[0], axes[1], axes[2]

    def _coordinates_from_axes(self, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        x_axis, y_axis, z_axis = axes
        X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
        return np.column_stack((X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")))

    def _grid_from_psi(self, psi: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> pv.ImageData:
        x_axis, y_axis, z_axis = axes
        amplitude = np.abs(psi)
        probability = amplitude**2
        phase = np.angle(psi.astype(complex))
        spacing = (
            x_axis[1] - x_axis[0] if len(x_axis) > 1 else 1.0,
            y_axis[1] - y_axis[0] if len(y_axis) > 1 else 1.0,
            z_axis[1] - z_axis[0] if len(z_axis) > 1 else 1.0,
        )
        origin = (x_axis[0], y_axis[0], z_axis[0])
        grid = pv.ImageData(dimensions=(len(x_axis), len(y_axis), len(z_axis)), spacing=spacing, origin=origin)
        grid["psi"] = psi.ravel(order="F")
        grid["amplitude"] = amplitude.ravel(order="F")
        grid["phase"] = phase.ravel(order="F")
        grid["probability"] = probability.ravel(order="F")
        return grid

    def _ensure_hybrid_grids(self) -> bool:
        if not self.hybrid_grids or not self._hybrid_meta or self.base_grid is None:
            self._compute_hybrid_grids()
        elif self._hybrid_meta.get("resolution") != self.grid_resolution or self._hybrid_meta.get("orbitals_version") != self._orbitals_version:
            self._compute_hybrid_grids()
        return bool(self.hybrid_grids)

    def _compute_hybrid_grids(self) -> None:
        if not self.orbitals:
            self.hybrid_grids = {}
            self.base_grid = None
            self.hybrid_status.setText("Add at least one orbital to compute hybrids.")
            return
        axes = self._sampling_axes()
        coords = self._coordinates_from_axes(axes)
        components: list[np.ndarray] = []
        for orb in self.orbitals:
            if not orb.visible:
                continue
            shifted = coords - np.asarray(orb.position, dtype=float).reshape(1, 3)
            psi, _, _, _ = evaluate_orbital_values(orb.symbol, orb.n, orb.l, orb.m, shifted)
            components.append(psi)
        if not components:
            self.hybrid_grids = {}
            self.base_grid = None
            self.hybrid_status.setText("Unable to build hybrid fields.")
            return
        contrib = np.vstack(components)
        atomic_amplitude = np.sqrt(np.sum(np.abs(contrib) ** 2, axis=0))
        atomic_psi = atomic_amplitude
        bonding_psi = contrib.sum(axis=0)
        alternating = np.array([1 if idx % 2 == 0 else -1 for idx in range(len(components))], dtype=float)
        antibonding_psi = (contrib * alternating[:, None]).sum(axis=0)
        self.base_grid = self._grid_from_psi(atomic_psi, axes)
        self.hybrid_grids = {
            "bonding": self._grid_from_psi(bonding_psi, axes),
            "antibonding": self._grid_from_psi(antibonding_psi, axes),
        }
        self._hybrid_meta = {"resolution": self.grid_resolution, "orbitals_version": self._orbitals_version}
        self.hybrid_status.setText(
            f"Computed bonding/antibonding from {len(self.orbitals)} orbital(s) at {self.grid_resolution}^3 resolution."
        )
        self.camera_initialized = False

    def _blended_grid(self, key: str) -> pv.ImageData | None:
        grid = self.hybrid_grids.get(key)
        base = self.base_grid
        if grid is None:
            return None
        frac = float(np.clip(self.mix_fraction, 0.0, 1.0))
        if base is None or frac >= 0.999:
            return grid
        if frac <= 0.001:
            return base
        try:
            base_psi = np.asarray(base.get_array("psi"))
            hybrid_psi = np.asarray(grid.get_array("psi"))
            if base_psi.shape != hybrid_psi.shape:
                return grid
            psi = (1.0 - frac) * base_psi + frac * hybrid_psi
            blended = grid.copy(deep=True)
            amplitude = np.abs(psi)
            blended["psi"] = psi
            blended["amplitude"] = amplitude
            blended["probability"] = amplitude**2
            blended["phase"] = np.angle(psi.astype(complex))
            return blended
        except Exception as exc:
            print(f"Blend error: {exc}", file=sys.stderr)
            return grid

    def _fields_for_key(self, key: str) -> tuple | None:
        grid = self._blended_grid(key)
        if grid is None:
            return None
        surface_field = field_from_grid(grid.copy(deep=False), self.current_mode, representation="surface", iso_fraction=self.iso_fraction)
        volume_field = field_from_grid(grid.copy(deep=False), self.current_mode, representation="volume", iso_fraction=self.iso_fraction)
        return surface_field, volume_field

    def _nucleus_color(self, symbol: str) -> str:
        return NUCLEUS_COLORS.get(symbol, "#f8fafc")

    def _render_nuclei(self) -> None:
        if not self.show_nuclei or not self.orbitals:
            return
        for orb in self.orbitals:
            if not orb.visible:
                continue
            base_radius = 0.35 + 0.02 * float(np.cbrt(_ATOMIC_NUMBER.get(orb.symbol, 1)))
            base_radius *= float(max(self.nucleus_scale, 0.05))
            color = self._nucleus_color(orb.symbol)
            offsets = [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.2, 0.2, 0.0]) * base_radius,
                np.array([-0.2, 0.2, 0.0]) * base_radius,
                np.array([0.2, -0.2, 0.0]) * base_radius,
                np.array([-0.2, -0.2, 0.0]) * base_radius,
            ]
            for scale, center_offset in zip([1.0, 0.5, 0.5, 0.5, 0.5], offsets, strict=False):
                radius = base_radius * scale
                center = orb.position + center_offset
                try:
                    self.plotter.add_mesh(
                        pv.Sphere(radius=radius, center=center),
                        color=color,
                        opacity=1.0,
                        show_scalar_bar=False,
                        smooth_shading=True,
                        specular=0.1,
                    )
                except Exception:
                    pass
                try:
                    self.slice_view.add_mesh(
                        pv.Sphere(radius=radius * 0.7, center=center),
                        color=color,
                        opacity=1.0,
                        show_scalar_bar=False,
                        smooth_shading=True,
                        specular=0.0,
                    )
                except Exception:
                    pass

    def _render_orbital(self, reuse_camera: bool = False, reuse_slice: bool = False) -> None:
        camera_state = self.plotter.camera_position
        slice_camera = getattr(self.slice_view, "camera_position", None)
        if not self._ensure_hybrid_grids():
            self.plotter_frame.reset_scene()
            self.slice_view.clear()
            return
        key = self._selected_hybrid_key()
        fields_pair = self._fields_for_key(key)
        if not fields_pair:
            return
        surface_field, volume_field = fields_pair
        fields = [surface_field]

        if reuse_camera:
            try:
                self.plotter.clear()
            except Exception:
                pass
        else:
            self.plotter_frame.reset_scene()
        if not reuse_slice:
            self.slice_view.clear()
            self.slice_plane_actor = None
        if self.camera_initialized and camera_state and reuse_camera:
            try:
                self.plotter.camera_position = camera_state
            except Exception:
                pass
        if not fields:
            return

        if self.current_representation == "volume":
            prob_arr = np.asarray(volume_field.dataset.get_array("probability"))
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
                arr = np.asarray(fields[0].dataset[fields[0].scalar_name]) if fields[0].dataset.get_array(fields[0].scalar_name) is not None else np.array([])
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
            self._add_iso_surfaces(volume_field, clim)
        else:
            for field in fields:
                self.plotter.add_mesh(
                    field.dataset,
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
        main_title = f"{key.title()} hybrid ({len(self.orbitals)} orbitals)"
        self.plotter.add_text(main_title, font_size=12, color="white", name="main_title", position="upper_left")
        view_label = f"3D View ({self.current_representation})"
        self.plotter.add_text(view_label, font_size=10, color="white", name="view_label", position="upper_right")
        if iso_text:
            self.plotter.add_text(iso_text, font_size=10, color="white", name="iso_text", position="lower_left")
        self._apply_slice(fields, volume_field, clim)
        self._render_nuclei()
        if not self.camera_initialized:
            self._autoscale()
            self.camera_initialized = True
        else:
            if reuse_camera:
                try:
                    self.plotter.camera_position = camera_state
                except Exception:
                    pass
            try:
                self.plotter.render()
            except Exception as exc:
                print(f"Render error: {exc}", file=sys.stderr)
        if reuse_slice and slice_camera:
            try:
                self.slice_view.camera_position = slice_camera
            except Exception:
                pass
        try:
            self.slice_view.render()
        except Exception:
            pass

    def _advance_animation(self) -> None:
        value = self.mix_slider.value()
        if value >= 100:
            self._stop_animation()
            return
        self.mix_slider.setValue(min(100, value + self.animation_step))

    def _toggle_animation(self) -> None:
        if self.animation_running:
            self._stop_animation()
            return
        self.mix_slider.setValue(0)
        self.animation_running = True
        self.play_anim_btn.setText("Stop animation")
        self.animation_timer.start(70)

    def _stop_animation(self) -> None:
        self.animation_running = False
        self.animation_timer.stop()
        if getattr(self, "play_anim_btn", None):
            self.play_anim_btn.setText("Play mix animation")

    def _capture_frame(self):
        try:
            frame = self.plotter.screenshot(return_img=True)
            if frame is None:
                return None
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        except Exception as exc:
            print(f"Screenshot failed: {exc}", file=sys.stderr)
            return None

    def _export_animation(self) -> None:
        if not self._ensure_hybrid_grids():
            self.hybrid_status.setText("Compute hybrids before exporting.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save hybrid animation", "hybrid.gif", "GIF Files (*.gif)")
        if not path:
            return
        steps = 30
        frames = []
        prev_mix = self.mix_slider.value()
        for i in range(steps + 1):
            frac = i / steps
            self.mix_slider.setValue(int(frac * 100))
            QtWidgets.QApplication.processEvents()
            self._render_orbital()
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)
        self.mix_slider.setValue(prev_mix)
        self._render_orbital()
        if not frames:
            self.hybrid_status.setText("Failed to capture frames for animation.")
            return
        try:
            try:
                import imageio.v2 as imageio  # type: ignore
            except Exception:
                import imageio  # type: ignore
            imageio.mimsave(path, frames, duration=0.08, loop=0)
            self.hybrid_status.setText(f"Saved hybrid animation to {path}")
        except Exception as exc:
            self.hybrid_status.setText(f"Animation export failed: {exc}")

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.animation_running = False
        if self.animation_timer:
            try:
                self.animation_timer.stop()
                self.animation_timer.deleteLater()
            except Exception:
                pass
        try:
            self.slice_view.close()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        super().cleanup()


class PeriodicTableTab(QtWidgets.QWidget):
    """Interactive periodic table rendered in Qt (values cross-referenced with PubChem)."""

    class TableCanvas(QtWidgets.QWidget):
        def __init__(self, layout: QtWidgets.QGridLayout, parent=None):
            super().__init__(parent)
            self.setLayout(layout)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            pen = QtGui.QPen(QtGui.QColor("#9ca3af"))
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            pen.setWidth(2)
            painter.setPen(pen)
            layout: QtWidgets.QGridLayout = self.layout()
            if not layout:
                return
            # Vertical dashed line after group 2 (between columns 2 and 3)
            def center_of_item(row, col):
                item = layout.itemAtPosition(row, col)
                if not item:
                    return None
                w = item.widget()
                if not w:
                    return None
                geo = w.geometry()
                return QtCore.QPointF(geo.center().x(), geo.center().y())

            # Use representative cells for positioning
            p_top = center_of_item(1, 2)
            p_bottom = center_of_item(7, 2)
            if p_top and p_bottom:
                x = (p_top.x() + center_of_item(1, 3).x()) / 2 if center_of_item(1, 3) else p_top.x() + 20
                painter.drawLine(QtCore.QPointF(x, p_top.y() - 20), QtCore.QPointF(x, p_bottom.y() + 20))
                # Connect to lanthanide/actinide rows
                p_la = center_of_item(9, 1)
                p_ac = center_of_item(10, 1)
                if p_la:
                    painter.drawLine(QtCore.QPointF(x, p_bottom.y() + 20), QtCore.QPointF(x, p_la.y()))
                if p_ac:
                    painter.drawLine(QtCore.QPointF(x, p_bottom.y() + 20), QtCore.QPointF(x, p_ac.y()))
            painter.end()

    FAMILY_COLORS = {
        "Noble Gas": "#3b82f6",
        "Alkali Metal": "#f97316",
        "Alkaline Earth Metal": "#fbbf24",
        "Transition Metal": "#10b981",
        "Post-Transition Metal": "#a3e635",
        "Metalloid": "#8b5cf6",
        "Nonmetal": "#e2e8f0",
        "Halogen": "#f43f5e",
        "Lanthanide": "#22d3ee",
        "Actinide": "#f59e0b",
    }
    _FAMILY_ALIASES = {
        "noble gas": "Noble Gas",
        "alkali metal": "Alkali Metal",
        "alkaline earth metal": "Alkaline Earth Metal",
        "transition metal": "Transition Metal",
        "post-transition metal": "Post-Transition Metal",
        "post transition metal": "Post-Transition Metal",
        "metalloid": "Metalloid",
        "nonmetal": "Nonmetal",
        "non-metal": "Nonmetal",
        "halogen": "Halogen",
        "lanthanide": "Lanthanide",
        "actinide": "Actinide",
    }
    FAMILY_SUMMARY = {
        "Noble Gas": "Closed-shell, monatomic gases; extremely low reactivity, colorless/odorless with very low boiling points.",
        "Alkali Metal": "Soft, highly reactive metals; form +1 cations and vigorous reactions with water; low ionization energy.",
        "Alkaline Earth Metal": "Reactive metals forming +2 cations; higher melting points than alkali metals and common oxides/carbonates.",
        "Transition Metal": "Variable oxidation states with partially filled d-subshells; often colored compounds and useful catalysts.",
        "Post-Transition Metal": "Softer, lower-melting metals with mixed metallic/covalent character; form diverse alloys and compounds.",
        "Metalloid": "Intermediate metallic/nonmetallic properties; semiconducting behavior common (e.g., Si, Ge).",
        "Nonmetal": "Generally poor electrical/thermal conductors; form covalent bonds and molecular compounds; wide range of states.",
        "Halogen": "Very reactive nonmetals forming -1 anions; strong oxidizers; exist as diatomic molecules at STP.",
        "Lanthanide": "Rare-earth metals with filling 4f subshell; typically +3 oxidation, magnetic/optical specialty uses.",
        "Actinide": "5f-block metals; radioactive, multiple oxidation states; many are synthetic with complex chemistry.",
    }

    class BohrViewer(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.elem: dict | None = None
            self.oxidation_state: int = 0
            self.setMinimumHeight(220)

        def set_element(self, elem: dict) -> None:
            self.elem = elem
            self.update()

        def set_oxidation_state(self, oxidation: int) -> None:
            self.oxidation_state = oxidation
            self.update()

        def _parse_oxidation_states_local(self, elem: dict) -> list[int]:
            raw = elem.get("oxidationStates") or elem.get("oxidationstates") or elem.get("oxidation_states")
            if raw is None:
                return []
            states: list[int] = []
            if isinstance(raw, (list, tuple)):
                for v in raw:
                    try:
                        states.append(int(v))
                    except Exception:
                        continue
            elif isinstance(raw, str):
                cleaned = raw.replace("−", "-")
                for part in cleaned.replace("+", " ").replace(",", " ").split():
                    try:
                        states.append(int(part))
                    except Exception:
                        continue
            return sorted(set(states))

        def _oxidation_items_local(self, states: list[int]) -> list[tuple[str, int]]:
            items = [("Neutral (0)", 0)]
            for s in states:
                label = f"Ion +{s}" if s > 0 else f"Ion {s}"
                items.append((label, s))
            return items

        def _shell_counts(self, elem: dict) -> tuple[list[int], dict[tuple[int, int], int]]:
            base_e = elem.get("numberOfElectrons") or elem.get("atomicNumber") or 0
            total_e = max(0, int(base_e) - int(self.oxidation_state))
            period = max(int(elem.get("period", 1)), 1)
            subshell_order = [
                (1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6),
                (4, 0, 2), (3, 2, 10), (4, 1, 6), (5, 0, 2), (4, 2, 10),
                (5, 1, 6), (6, 0, 2), (4, 3, 14), (5, 2, 10), (6, 1, 6),
                (7, 0, 2), (5, 3, 14), (6, 2, 10), (7, 1, 6)
            ]
            subshells: dict[tuple[int, int], int] = {}
            remaining = int(total_e)
            for n, l, cap in subshell_order:
                if remaining <= 0:
                    break
                fill = min(cap, remaining)
                subshells[(n, l)] = fill
                remaining -= fill

            # Apply common stability exceptions
            adjustments: dict[int, dict[tuple[int, int], int]] = {
                24: {(4, 0): -1, (3, 2): 1},
                29: {(4, 0): -1, (3, 2): 1},
                42: {(5, 0): -1, (4, 2): 1},
                46: {(5, 0): -2, (4, 2): 2},
                47: {(5, 0): -1, (4, 2): 1},
                79: {(6, 0): -1, (5, 2): 1},
            }
            adj = adjustments.get(int(total_e))
            if adj:
                for (n, l), delta in adj.items():
                    subshells[(n, l)] = max(0, subshells.get((n, l), 0) + delta)

            shells = [0] * max(period, 1)
            for (n, _l), cnt in subshells.items():
                if 1 <= n <= len(shells):
                    shells[n - 1] += cnt
            # If no electrons assigned (shouldn't happen), fall back evenly
            if sum(shells) == 0 and total_e > 0:
                capacity = [2, 8, 18, 32, 32, 18, 8]
                remaining = total_e
                for idx in range(len(shells)):
                    cap = capacity[idx] if idx < len(capacity) else 2 * (idx + 1) * (idx + 1)
                    take = min(remaining, cap)
                    shells[idx] = take
                    remaining -= take
                    if remaining <= 0:
                        break
            return shells, subshells

        def _capacity(self, shell_index: int) -> int:
            n = shell_index + 1
            return 2 * n * n

        def _subshell_capacity(self, l: int) -> int:
            return {0: 2, 1: 6, 2: 10, 3: 14}.get(l, 0)

        def _angle_diff(self, a: float, b: float) -> float:
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

        def _generate_between_gaps(self, base: list[float], count: int) -> list[float]:
            """Place `count` angles into the largest gaps between existing angles."""
            if not base:
                return []
            existing = list(base)
            new_angles: list[float] = []
            max_iters = count * 8
            while len(new_angles) < count and max_iters > 0:
                max_iters -= 1
                sorted_base = sorted(existing)
                gaps: list[tuple[float, float]] = []
                for i, ang in enumerate(sorted_base):
                    nxt = sorted_base[(i + 1) % len(sorted_base)]
                    diff = (nxt - ang) % (2 * math.pi)
                    gaps.append((diff, ang))
                gaps.sort(reverse=True, key=lambda x: x[0])
                diff, start = gaps[0]
                mid = (start + diff / 2) % (2 * math.pi)
                if all(self._angle_diff(mid, a) > 1e-3 for a in existing):
                    new_angles.append(mid)
                    existing.append(mid)
                else:
                    # nudge slightly if midpoint is too close
                    mid = (mid + 1e-3) % (2 * math.pi)
                    new_angles.append(mid)
                    existing.append(mid)
            return new_angles[:count]

        def _angle_sets(self) -> dict[int, list[float]]:
            s_angles = [0.0, math.pi]
            p_angles = [
                math.radians(45), math.radians(90), math.radians(135),
                math.radians(-45) % (2 * math.pi), math.radians(-90) % (2 * math.pi), math.radians(-135) % (2 * math.pi),
            ]
            base_for_d = s_angles + p_angles
            d_angles = self._generate_between_gaps(base_for_d, self._subshell_capacity(2))
            base_for_f = base_for_d + d_angles
            f_angles = self._generate_between_gaps(base_for_f, self._subshell_capacity(3))
            return {0: s_angles, 1: p_angles, 2: d_angles, 3: f_angles}

        class SubshellMiniWidget(QtWidgets.QWidget):
            def __init__(self, label: str, filled: int, capacity: int, angles: list[float], is_full: bool, parent=None):
                super().__init__(parent)
                self.label = label
                self.filled = filled
                self.capacity = capacity
                self.is_full = is_full
                self.setMinimumSize(80, 80)

            def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                if not painter.isActive():
                    return
                try:
                    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                    w, h = self.width(), self.height()
                    center = QtCore.QPointF(w / 2, h / 2 + 6)
                    radius = min(w, h) * 0.32
                    dot_r = radius * 0.18

                    pen = QtGui.QPen(QtGui.QColor("#94a3b8"))
                    pen.setWidth(2 if not self.is_full else 3)
                    if self.is_full:
                        pen.setColor(QtGui.QColor("#22c55e"))
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(center, radius, radius)

                    cap = max(self.capacity, 1)
                    angles = [2 * math.pi * i / cap for i in range(cap)]
                    for idx, ang in enumerate(angles):
                        pos = QtCore.QPointF(
                            center.x() + radius * math.cos(ang),
                            center.y() + radius * math.sin(ang),
                        )
                        if idx < self.filled:
                            pen = QtGui.QPen(QtGui.QColor("#22c55e" if self.is_full else "#0f172a"))
                            pen.setWidthF(2 if self.is_full else 1.5)
                            painter.setPen(pen)
                            painter.setBrush(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                            painter.drawEllipse(pos, dot_r, dot_r)
                        else:
                            painter.setBrush(QtCore.Qt.NoBrush)
                            pen = QtGui.QPen(QtGui.QColor("#ef4444"))
                            pen.setWidthF(1.5)
                            painter.setPen(pen)
                            painter.drawEllipse(pos, dot_r, dot_r)

                    painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                    painter.drawText(QtCore.QPointF(6, 14), self.label)
                finally:
                    painter.end()
        def paintEvent(self, event):
            super().paintEvent(event)
            if not self.elem:
                return
            painter = QtGui.QPainter(self)
            if not painter.isActive():
                return
            try:
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                w, h = self.width(), self.height()
                center = QtCore.QPointF(w / 2, h / 2)
                shells, subshells = self._shell_counts(self.elem)
                max_shell = max(len(shells), 1)
                base_radius = min(w, h) * 0.4
                step = base_radius / max_shell
                dot_r = step * 0.06

                pen = QtGui.QPen(QtGui.QColor("#94a3b8"))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#1f2937")))
                painter.drawEllipse(center, step * 0.3, step * 0.3)

                filled_annotations: list[str] = []

                for shell_idx in range(len(shells)):
                    n = shell_idx + 1
                    l_values = [l for l in range(0, min(3, n - 1) + 1)]
                    lmax = len(l_values) - 1
                    radius_base = step * n
                    offset = step * 0.2
                    start_radius = radius_base - (lmax * offset) / 2 if lmax >= 0 else radius_base

                    for l in l_values:
                        radius = start_radius + l * offset
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 2))
                        painter.drawEllipse(center, radius, radius)
                        if l == lmax:
                            painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a")))
                            # place label just outside outermost subshell circle
                            painter.drawText(center + QtCore.QPointF(radius + 18, -radius - 6), f"n={n}")

                        cap = self._subshell_capacity(l)
                        cap = max(cap, 1)
                        angles = [2 * math.pi * k / cap for k in range(cap)]
                        filled = min(cap, subshells.get((n, l), 0))
                        full = filled >= cap and cap > 0
                        for idx, ang in enumerate(angles):
                            pos = QtCore.QPointF(
                                center.x() + radius * math.cos(ang),
                                center.y() + radius * math.sin(ang),
                            )
                            if idx < filled:
                                pen_color = "#22c55e" if full else "#0f172a"
                                pen = QtGui.QPen(QtGui.QColor(pen_color))
                                pen.setWidthF(2 if full else 1.5)
                                painter.setPen(pen)
                                painter.setBrush(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                                painter.drawEllipse(pos, dot_r, dot_r)
                            else:
                                painter.setBrush(QtCore.Qt.NoBrush)
                                pen = QtGui.QPen(QtGui.QColor("#ef4444"))
                                pen.setWidthF(1.5)
                                painter.setPen(pen)
                                painter.drawEllipse(pos, dot_r, dot_r)
                        if full:
                            filled_annotations.append(f"{n}{'spdf'[l]}")

                if filled_annotations:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#22c55e")))
                    painter.drawText(
                        QtCore.QPointF(10, h - 10),
                        "Filled subshells: " + ", ".join(filled_annotations),
                    )
            finally:
                painter.end()

        def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
            if not self.elem:
                return
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else self
            menu = QtWidgets.QMenu(parent)
            expand_action = menu.addAction("Expand subshells")
            expand_action.triggered.connect(self._show_subshell_dialog)
            box_action = menu.addAction("Orbital box view")
            box_action.triggered.connect(self._show_orbital_box_dialog)
            menu.exec(event.globalPos())

        def _show_subshell_dialog(self) -> None:
            if not self.elem:
                return
            shells, subshells = self._shell_counts(self.elem)
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else None
            dlg = QtWidgets.QDialog(parent)
            name = self.elem.get("name", "")
            sym = self.elem.get("symbol", "")
            dlg.setWindowTitle(f"Subshells for {name} ({sym})")
            layout = QtWidgets.QVBoxLayout(dlg)
            header = QtWidgets.QLabel("Electrons and vacancies by subshell")
            layout.addWidget(header)

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
            ox_combo = QtWidgets.QComboBox()
            states = self._parse_oxidation_states_local(self.elem)
            items = self._oxidation_items_local(states)
            for label, val in items:
                ox_combo.addItem(label, userData=val)
            # set current selection
            try:
                idx = [v for _, v in items].index(self.oxidation_state)
            except ValueError:
                idx = 0
            ox_combo.setCurrentIndex(idx)
            controls.addWidget(ox_combo)
            controls.addStretch()
            layout.addLayout(controls)

            grid = QtWidgets.QGridLayout()
            layout.addLayout(grid)
            l_labels = ["s", "p", "d", "f"]
            def refresh_grid():
                # clear old
                while grid.count():
                    item = grid.takeAt(0)
                    w = item.widget()
                    if w:
                        w.deleteLater()
                for col, label in enumerate(l_labels):
                    lbl = QtWidgets.QLabel(label)
                    lbl.setAlignment(QtCore.Qt.AlignCenter)
                    grid.addWidget(lbl, 0, col + 1)
                shells, subshells = self._shell_counts(self.elem)
                for n in range(1, len(shells) + 1):
                    row = n
                    row_label = QtWidgets.QLabel(f"n={n}")
                    row_label.setAlignment(QtCore.Qt.AlignCenter)
                    grid.addWidget(row_label, row, 0)
                    lmax = min(3, n - 1)
                    for l in range(0, lmax + 1):
                        cap = self._subshell_capacity(l)
                        filled = min(cap, subshells.get((n, l), 0))
                        angles = list(self._angle_sets().get(l, []))
                        view = self.SubshellMiniWidget(
                            f"{n}{l_labels[l]}",
                            filled,
                            cap,
                            angles,
                            filled >= cap,
                            dlg,
                        )
                        grid.addWidget(view, row, l + 1)

            def on_change():
                val = ox_combo.currentData()
                try:
                    self.set_oxidation_state(int(val))
                except Exception:
                    self.set_oxidation_state(0)
                refresh_grid()

            ox_combo.currentIndexChanged.connect(on_change)
            refresh_grid()

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
            dlg.resize(480, 360)
            dlg.setModal(True)
            dlg.exec()

        def _show_orbital_box_dialog(self) -> None:
            if not self.elem:
                return
            parent = self.window() if isinstance(self.window(), QtWidgets.QWidget) else None
            dlg = QtWidgets.QDialog(parent)
            name = self.elem.get("name", "")
            sym = self.elem.get("symbol", "")
            dlg.setWindowTitle(f"Orbital Box View – {name} ({sym})")
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.addWidget(QtWidgets.QLabel("Hund's rule filling; arrows show spin."))

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
            ox_combo = QtWidgets.QComboBox()
            states = self._parse_oxidation_states_local(self.elem)
            items = self._oxidation_items_local(states)
            for label, val in items:
                ox_combo.addItem(label, userData=val)
            try:
                idx = [v for _, v in items].index(self.oxidation_state)
            except ValueError:
                idx = 0
            ox_combo.setCurrentIndex(idx)
            controls.addWidget(ox_combo)
            controls.addStretch()
            layout.addLayout(controls)

            parent_tab = self.parent()
            view_cls = getattr(parent_tab, "OrbitalBoxView", PeriodicTableTab.OrbitalBoxView if "PeriodicTableTab" in globals() else None)
            if view_cls is None:
                return
            view = view_cls(self.elem, self.oxidation_state, self._shell_counts, self._subshell_capacity, dlg)
            self._orbital_box_dialog = dlg
            view.setMinimumSize(520, 420)
            layout.addWidget(view, 1)

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
            dlg.resize(600, 500)
            dlg.setModal(True)
            dlg.exec()

    class OrbitalBoxView(QtWidgets.QWidget):
        def __init__(self, elem: dict, oxidation: int, shell_fn, cap_fn, parent=None):
            super().__init__(parent)
            self.elem = elem
            self.oxidation = oxidation
            self.shell_fn = shell_fn
            self.cap_fn = cap_fn
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            self.setMinimumHeight(320)
            self._order = [
                (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
                (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
                (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
                (7, 0), (5, 3), (6, 2), (7, 1)
            ]

        def _degeneracy(self, l: int) -> int:
            return {0: 1, 1: 3, 2: 5, 3: 7}.get(l, 1)

        def _fill_orbitals(self, electrons: int, deg: int) -> list[list[str]]:
            # Hund: first fill singly (up), then pair with down
            orbitals = [["", ""] for _ in range(deg)]
            cap = 2 * deg
            electrons = max(0, min(electrons, cap))
            # first pass: one up in each orbital
            idx = 0
            while electrons > 0 and idx < deg:
                orbitals[idx][0] = "↑"
                electrons -= 1
                idx += 1
            # second pass: pair with down
            idx = 0
            while electrons > 0 and idx < deg:
                orbitals[idx][1] = "↓"
                electrons -= 1
                idx += 1
            return orbitals

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            if not painter.isActive():
                return
            try:
                try:
                    owner = self.shell_fn.__self__
                    prev = getattr(owner, "oxidation_state", 0)
                    owner.oxidation_state = self.oxidation
                    _, subshells = self.shell_fn(self.elem)
                    owner.oxidation_state = prev
                except Exception:
                    _, subshells = self.shell_fn(self.elem)
                painter.fillRect(self.rect(), QtGui.QColor("#1f2937"))
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

                # Base geometry
                base_left = 70.0
                base_top = 24.0
                base_col_w = 120.0
                base_box_w = 30.0
                base_box_h = 22.0
                base_spacing = 6.0
                label_map = {0: "s", 1: "p", 2: "d", 3: "f"}

                rows = [(n, l, subshells.get((n, l), 0)) for (n, l) in self._order if (n, l) in subshells]
                if not rows:
                    return

                # Horizontal scaling so widest subshell fits
                max_width_needed = 0.0
                for n, l, _e in rows:
                    deg = self._degeneracy(l)
                    width = base_left + l * base_col_w + deg * base_box_w + max(0, deg - 1) * base_spacing
                    max_width_needed = max(max_width_needed, width)
                avail_w = max(200.0, self.width() - 24.0)
                scale_x = min(1.0, avail_w / max_width_needed) if max_width_needed > 0 else 1.0

                left_margin = max(40.0, base_left * scale_x)
                top_margin = max(16.0, base_top * scale_x)
                margin = top_margin
                col_w = max(50.0, base_col_w * scale_x)
                box_w = max(12.0, base_box_w * scale_x)
                box_h = max(12.0, base_box_h * scale_x)
                spacing = max(3.0, base_spacing * scale_x)

                max_row = len(rows)
                avail_h = max(140.0, self.height() - 2 * margin)
                base_row_h = 70.0 * min(1.1, scale_x + 0.2)
                row_h = max(34.0, min(110.0, min(avail_h / max_row, base_row_h)))
                base_y = self.height() - margin - row_h

                arrow_x = left_margin / 2
                arrow_top = top_margin
                arrow_bottom = self.height() - margin
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 3))
                painter.drawLine(arrow_x, arrow_bottom, arrow_x, arrow_top + 14)
                head = QtGui.QPolygonF([
                    QtCore.QPointF(arrow_x, arrow_top),
                    QtCore.QPointF(arrow_x - 8, arrow_top + 14),
                    QtCore.QPointF(arrow_x + 8, arrow_top + 14),
                ])
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#e5e7eb")))
                painter.drawPolygon(head)
                painter.save()
                painter.translate(arrow_x - 16, (arrow_top + arrow_bottom) / 2)
                painter.rotate(-90)
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                painter.drawText(QtCore.QPointF(-30, 0), "Energy")
                painter.restore()

                for idx, (n, l, electrons) in enumerate(rows):
                    y = base_y - idx * row_h
                    x = left_margin + col_w * l
                    deg = self._degeneracy(l)
                    cap = self.cap_fn(l)
                    electrons = max(0, min(cap, electrons))
                    boxes = self._fill_orbitals(electrons, deg)

                    painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    label = f"{n}{label_map.get(l, '?')}"
                    painter.drawText(x, y - 6, label)

                    for j, (up, down) in enumerate(boxes):
                        bx = x + j * (box_w + spacing)
                        by = y
                        rect = QtCore.QRectF(bx, by, box_w, box_h)
                        painter.drawRect(rect)
                        painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
                        if up:
                            left_half = QtCore.QRectF(rect.left() + 2, rect.top() + 2, rect.width() / 2 - 4, rect.height() - 4)
                            painter.drawText(left_half, QtCore.Qt.AlignCenter, up)
                        if down:
                            right_half = QtCore.QRectF(rect.left() + rect.width() / 2 + 2, rect.top() + 2, rect.width() / 2 - 4, rect.height() - 4)
                            painter.drawText(right_half, QtCore.Qt.AlignCenter, down)
            finally:
                painter.end()

    class RotatedLabel(QtWidgets.QLabel):
        def __init__(self, text: str, angle: float = -90.0, parent=None):
            super().__init__(text, parent)
            self.angle = angle

        def minimumSizeHint(self):
            s = super().minimumSizeHint()
            return QtCore.QSize(s.height(), s.width())

        def sizeHint(self):
            return self.minimumSizeHint()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.translate(self.width() / 2, self.height() / 2)
            painter.rotate(self.angle)
            painter.translate(-self.height() / 2, -self.width() / 2)
            painter.drawText(QtCore.QRectF(0, 0, self.height(), self.width()), QtCore.Qt.AlignCenter, self.text())
            painter.end()
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.font_point_size: int = 11
        layout = QtWidgets.QVBoxLayout(self)
        credit = QtWidgets.QLabel(
            "Periodic table UI adapted from the Interactive Periodic Table project (codingwithnsh, GitHub).\n"
            "Element properties cross-referenced with PubChem (https://pubchem.ncbi.nlm.nih.gov/)."
        )
        credit.setWordWrap(True)
        layout.addWidget(credit)

        body = QtWidgets.QHBoxLayout()
        layout.addLayout(body, 1)

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left_splitter.setOpaqueResize(True)
        left_splitter.setChildrenCollapsible(False)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("Color by:"))
        self.scheme_combo = QtWidgets.QComboBox()
        self._scheme_options = [
            {"key": "family", "label": "Family", "type": "categorical", "field": None},
            {"key": "state", "label": "State at STP", "type": "categorical", "field": "standardState"},
            {"key": "density", "label": "Density", "type": "numeric", "field": "density"},
            {"key": "atomicMass", "label": "Atomic Mass", "type": "numeric", "field": "atomicMass"},
            {"key": "electronegativity", "label": "Electronegativity", "type": "numeric", "field": "electronegativity"},
            {"key": "atomicRadius", "label": "Atomic Radius", "type": "numeric", "field": "atomicRadius"},
            {"key": "meltingPoint", "label": "Melting Point", "type": "numeric", "field": "meltingPoint"},
            {"key": "boilingPoint", "label": "Boiling Point", "type": "numeric", "field": "boilingPoint"},
        ]
        for opt in self._scheme_options:
            self.scheme_combo.addItem(opt["label"], userData=opt)
        controls_row.addWidget(self.scheme_combo)

        controls_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.table_cmap_combo = QtWidgets.QComboBox()
        for cmap in self._available_colormaps():
            self.table_cmap_combo.addItem(cmap)
        self.table_cmap_combo.setCurrentText(self._scheme_default_cmap("density"))
        controls_row.addWidget(self.table_cmap_combo)
        controls_row.addWidget(QtWidgets.QLabel("Scale:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        self.scale_combo.setCurrentText("Linear")
        controls_row.addWidget(self.scale_combo)
        controls_row.addWidget(QtWidgets.QLabel("Temp unit:"))
        self.temp_unit_combo = QtWidgets.QComboBox()
        self.temp_unit_combo.addItems(["K", "°C", "°F"])
        self.temp_unit_combo.setCurrentText("K")
        controls_row.addWidget(self.temp_unit_combo)
        controls_row.addStretch()
        # container to hold legend, colorbar, table
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(controls_row)

        self.legend_row = QtWidgets.QHBoxLayout()
        self.legend_container = QtWidgets.QWidget()
        self.legend_container.setLayout(self.legend_row)
        self.legend_row.addWidget(QtWidgets.QLabel("Legend (family colors shown when applicable):"))
        for family, color in self.FAMILY_COLORS.items():
            swatch = QtWidgets.QLabel("  ")
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #0f172a;")
            self.legend_row.addWidget(swatch)
            self.legend_row.addWidget(QtWidgets.QLabel(family))
        self.legend_row.addStretch()
        top_layout.addWidget(self.legend_container)

        self.colorbar_label = QtWidgets.QLabel()
        self.colorbar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.colorbar_label.setVisible(False)
        self.colorbar_label.setMaximumHeight(120)
        top_layout.addWidget(self.colorbar_label)

        self.grid_widget = self.TableCanvas(QtWidgets.QGridLayout(), self)
        self.grid_layout = self.grid_widget.layout()
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        # period label + grid in horizontal layout
        grid_row = QtWidgets.QHBoxLayout()
        self.period_label = self.RotatedLabel("Period", angle=-90)
        self.period_label.setAlignment(QtCore.Qt.AlignCenter)
        self.period_label.setContentsMargins(0, 12, 0, 12)
        grid_row.addWidget(self.period_label, 0)
        grid_row.addWidget(self.grid_widget, 1)
        top_layout.addLayout(grid_row, 6)

        self.bohr_view = self.BohrViewer()
        self.bohr_view.setVisible(False)
        left_splitter.addWidget(top_container)
        left_splitter.setStretchFactor(0, 1)

        right_widget = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_widget)
        font_row = QtWidgets.QHBoxLayout()
        font_row.addWidget(QtWidgets.QLabel("Font size:"))
        self.font_size_combo = QtWidgets.QComboBox()
        for size in ["9", "10", "11", "12", "13"]:
            self.font_size_combo.addItem(size)
        self.font_size_combo.setCurrentText(str(self.font_point_size))
        self.font_size_combo.currentTextChanged.connect(self._on_font_change)
        font_row.addWidget(self.font_size_combo)
        font_row.addStretch()
        right.addLayout(font_row)
        self.info = QtWidgets.QTextBrowser()
        self.info.setOpenExternalLinks(True)
        self.info.setReadOnly(True)
        self.info.setMinimumWidth(360)
        self.info.setStyleSheet(
            "font-family: 'Segoe UI', sans-serif; font-size: 11pt; "
            "background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px;"
        )
        self.info.anchorClicked.connect(self._handle_info_link)
        right.addWidget(self.info, 2)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Electron configuration for oxidation state:"))
        self.oxidation_combo = QtWidgets.QComboBox()
        self.oxidation_combo.currentIndexChanged.connect(self._on_oxidation_change)
        controls_layout.addWidget(self.oxidation_combo, 1)
        right.addLayout(controls_layout)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setOpaqueResize(True)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        body.addWidget(main_splitter)

        self.data = load_data()
        self.buttons: dict[int, QtWidgets.QPushButton] = {}
        self.current_element: dict | None = None
        self.current_oxidation_state: int = 0
        self.current_scheme: str = "family"
        self.current_table_cmap: str = self._scheme_default_cmap("density")
        self.scale_mode: str = "linear"
        self.field_units: dict[str, str | None] = {
            "density": "g/cm^3",
            "atomicMass": "u",
            "electronegativity": None,
            "atomicRadius": "pm",
            "meltingPoint": "K",
            "boilingPoint": "K",
        }
        self.temp_unit: str = "K"
        self.show_abbrev_config: bool = False
        self._build_grid()
        self._select_atomic_number(14)
        self.scheme_combo.currentIndexChanged.connect(self._on_scheme_change)
        self.table_cmap_combo.currentTextChanged.connect(self._on_table_cmap_change)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_change)
        self.temp_unit_combo.currentIndexChanged.connect(self._on_temp_unit_change)
        self._on_scheme_change()

    def _elem_position(self, elem: dict) -> tuple[int, int]:
        an = elem["atomicNumber"]
        period = elem.get("period", 1)
        group = elem.get("group", None)
        if 57 <= an <= 71:
            return (9, an - 57 + 1)
        if 89 <= an <= 103:
            return (10, an - 89 + 1)
        if group is None:
            group = 18
        return (period, group)

    def _parse_oxidation_states(self, elem: dict) -> list[int]:
        raw = elem.get("oxidationStates") or elem.get("oxidationstates") or elem.get("oxidation_states")
        if raw is None:
            return []
        states: list[int] = []
        if isinstance(raw, (list, tuple)):
            for v in raw:
                try:
                    states.append(int(v))
                except Exception:
                    continue
        elif isinstance(raw, str):
            for part in raw.replace("−", "-").replace("+", " ").replace(",", " ").split():
                try:
                    states.append(int(part))
                except Exception:
                    continue
        return sorted(set(states))

    def _format_oxidation_states(self, states: list[int]) -> str:
        if not states:
            return "N/A"
        def fmt(v: int) -> str:
            if v > 0:
                return f"+{v}"
            return str(v)
        return ", ".join(fmt(s) for s in states)

    def _oxidation_items(self, states: list[int]) -> list[tuple[str, int]]:
        items = [("Neutral (0)", 0)]
        for s in states:
            label = f"Ion {self._format_oxidation_states([s])}"
            items.append((label, s))
        return items

    def _update_oxidation_combo(self, elem: dict) -> None:
        states = self._parse_oxidation_states(elem)
        self.oxidation_combo.blockSignals(True)
        self.oxidation_combo.clear()
        for label, val in self._oxidation_items(states):
            self.oxidation_combo.addItem(label, userData=val)
        self.oxidation_combo.setCurrentIndex(0)
        self.oxidation_combo.blockSignals(False)
        self.current_oxidation_state = 0

    def _on_oxidation_change(self) -> None:
        if self.current_element is None:
            return
        val = self.oxidation_combo.currentData()
        try:
            self.current_oxidation_state = int(val)
        except Exception:
            self.current_oxidation_state = 0
        self.bohr_view.set_oxidation_state(self.current_oxidation_state)
        self._refresh_info()

    def _build_grid(self) -> None:
        # headers
        # group labels row
        group_label = QtWidgets.QLabel("Group")
        group_label.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(group_label, 0, 1, 1, 18)
        for col in range(1, 19):
            lbl = QtWidgets.QLabel(str(col))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, 1, col)
        # period labels column
        for row in range(1, 8):
            lbl = QtWidgets.QLabel(str(row))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, row + 1, 0)
        la_lbl = QtWidgets.QLabel("La")
        la_lbl.setAlignment(QtCore.Qt.AlignCenter)
        ac_lbl = QtWidgets.QLabel("Ac")
        ac_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(la_lbl, 10, 0)
        self.grid_layout.addWidget(ac_lbl, 11, 0)

        for elem in self.data["elements"]:
            row, col = self._elem_position(elem)
            row += 1  # shift for header row
            btn = QtWidgets.QPushButton(f"{elem['symbol']}\n{elem['atomicNumber']}")
            btn.setCheckable(True)
            btn.setStyleSheet(self._style_for_family(elem.get("family", "")))
            btn.clicked.connect(lambda _, a=elem["atomicNumber"]: self._select_atomic_number(a))
            btn.setMinimumHeight(64)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            self.grid_layout.addWidget(btn, row, col)
            self.buttons[elem["atomicNumber"]] = btn
        # keep headers pinned
        self.grid_layout.setRowStretch(0, 0)
        for r in range(1, 8):
            self.grid_layout.setRowStretch(r, 1)
        # reduce gap before lanthanides/actinides
        self.grid_layout.setRowStretch(8, 0)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setRowStretch(10, 1)
        for c in range(0, 19):
            self.grid_layout.setColumnStretch(c, 0)
        self._apply_coloring()

    def _style_for_family(self, family: str) -> str:
        key = self._FAMILY_ALIASES.get(family.lower(), family) if isinstance(family, str) else family
        color = self.FAMILY_COLORS.get(key, "#94a3b8")
        return self._button_style(color, self._text_contrast(color))

    def _format_numeric(self, val: float) -> str:
        try:
            v = float(val)
        except Exception:
            return "-"
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
            return f"{v:.3e}"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def _unit_latex(self, unit: str | None) -> str | None:
        if not unit:
            return None
        mapping = {
            "g/cm^3": r"g\,cm^{-3}",
            "u": r"u",
            "pm": r"pm",
            "K": r"K",
            "degC": r"^{\circ}C",
            "degF": r"^{\circ}F",
        }
        return mapping.get(unit, unit.replace("^", "^{") + "}" if "^" in unit else unit)

    def _unit_html(self, unit: str | None) -> str:
        if not unit:
            return ""
        repl = {
            "g/cm^3": "g/cm<sup>-3</sup>",
            "degC": "&deg;C",
            "degF": "&deg;F",
        }
        return repl.get(unit, unit.replace("^", "<sup>") + "</sup>" if "^" in unit else unit)

    def _button_label(self, elem: dict, extra: str | None = None) -> str:
        base_top = f"{elem['symbol']}"
        extra = "-" if extra in (None, "") else str(extra)
        return f"{base_top}\n{extra}"

    def _text_contrast(self, hex_color: str) -> str:
        try:
            hc = hex_color.lstrip("#")
            r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return "#0f172a" if luminance > 0.65 else "#f8fafc"
        except Exception:
            return "#0f172a"

    def _button_style(self, bg: str, text: str) -> str:
        return (
            f"background-color: {bg}; color: {text}; font-weight: bold; font-size: {self.font_point_size}pt; "
            "border: 1px solid #0f172a; padding: 6px; border-radius: 4px;"
        )

    def _parse_quantity(self, raw, field: str) -> tuple[float, str | None]:
        """Return magnitude in preferred units (if available) and the unit label."""
        preferred = self.field_units.get(field)
        if field in ("meltingPoint", "boilingPoint"):
            preferred = {"K": "K", "°C": "degC", "°F": "degF"}.get(getattr(self, "temp_unit", "K"), preferred)
        if raw in (None, "", " ", "-"):
            return float("nan"), preferred
        try:
            if isinstance(raw, (int, float)):
                q = Q_(raw)
            else:
                q = Q_(str(raw))
            if preferred:
                try:
                    mag = q.to(preferred).magnitude
                    return float(mag), preferred
                except Exception:
                    pass
            return float(q.magnitude), str(q.units) if q.units != ureg.dimensionless else None
        except Exception:
            try:
                return float(raw), preferred
            except Exception:
                return float("nan"), preferred

    def _available_colormaps(self) -> list[str]:
        base = ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "coolwarm"]
        cram = ["batlow", "bamako", "devon", "oslo", "lajolla", "hawaii", "davos", "vik", "broc", "cork", "roma", "tokyo"]
        return base + cram

    def _get_cmap(self, name: str):
        return _resolve_cmap(name)

    def _scheme_default_cmap(self, key: str) -> str:
        defaults = {
            "density": "batlow",
            "atomicMass": "lajolla",
            "electronegativity": "oslo",
            "atomicRadius": "vik",
            "meltingPoint": "davos",
            "boilingPoint": "davos",
        }
        return defaults.get(key, "viridis")

    def _on_scheme_change(self) -> None:
        meta = self.scheme_combo.currentData()
        if not meta:
            return
        self.current_scheme = meta["key"]
        if meta["type"] == "categorical":
            self.table_cmap_combo.setEnabled(False)
        else:
            self.table_cmap_combo.setEnabled(True)
            default = self._scheme_default_cmap(self.current_scheme)
            if default in [self.table_cmap_combo.itemText(i) for i in range(self.table_cmap_combo.count())]:
                self.table_cmap_combo.blockSignals(True)
                self.table_cmap_combo.setCurrentText(default)
                self.table_cmap_combo.blockSignals(False)
                self.current_table_cmap = default
        self._apply_coloring()

    def _on_table_cmap_change(self, cmap: str) -> None:
        self.current_table_cmap = cmap
        self._apply_coloring()

    def _on_temp_unit_change(self) -> None:
        self.temp_unit = self.temp_unit_combo.currentText()
        self._apply_coloring()
        self._refresh_info()

    def _on_scale_change(self) -> None:
        self.scale_mode = "log" if self.scale_combo.currentText().lower().startswith("log") else "linear"
        self._apply_coloring()

    def _on_font_change(self) -> None:
        try:
            self.font_point_size = int(self.font_size_combo.currentText())
        except Exception:
            self.font_point_size = 11
        self._apply_coloring()
        self._refresh_info()

    def _handle_info_link(self, url: QtCore.QUrl) -> None:
        if url.toString() == "toggle-config":
            self.show_abbrev_config = not self.show_abbrev_config
            self._refresh_info()
        else:
            QtGui.QDesktopServices.openUrl(url)

    def _on_oxidation_change(self) -> None:
        if self.current_element is None:
            return
        val = self.oxidation_combo.currentData()
        try:
            self.current_oxidation_state = int(val)
        except Exception:
            self.current_oxidation_state = 0
        self.bohr_view.set_oxidation_state(self.current_oxidation_state)
        self._refresh_info()
        # keep orbital box dialog in sync if open
        dlg = getattr(self, "_orbital_box_dialog", None)
        if dlg and dlg.isVisible():
            dlg.close()
            self._show_orbital_box_dialog()

    def _apply_coloring(self) -> None:
        meta = self.scheme_combo.currentData()
        if not meta:
            return
        scheme = meta["key"]
        field = meta.get("field")

        # Toggle legend/colorbar visibility
        if scheme in ("family", "state"):
            self.legend_container.setVisible(True)
            self.colorbar_label.setVisible(False)
        else:
            self.legend_container.setVisible(False)
            try:
                self._render_table_colorbar(field, scheme, meta["label"])
            except Exception as exc:
                print(f"Table colorbar render error: {exc}", file=sys.stderr)
                self.colorbar_label.setVisible(False)

        # Precompute numeric range if needed
        vmin = vmax = None
        cmap = None
        to_hex = lambda rgba: f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        log_scale = self.scale_mode == "log"
        if scheme not in ("family", "state") and field:
            vals: list[float] = []
            for elem in self.data["elements"]:
                if field in elem and elem[field] not in ("", None):
                    mag, _ = self._parse_quantity(elem.get(field), field)
                    if not math.isnan(mag):
                        if log_scale and mag <= 0:
                            continue
                        vals.append(mag)
            if vals:
                vmin, vmax = min(vals), max(vals)
                if log_scale:
                    vmin = max(min(vals), sys.float_info.min)
                if math.isclose(vmin, vmax):
                    vmax = vmin * 1.1 if log_scale else vmin + 1.0
                cmap_name = self.current_table_cmap or self._scheme_default_cmap(scheme)
                cmap = self._get_cmap(cmap_name)

        state_colors = {
            "solid": "#f97316",
            "liquid": "#3b82f6",
            "gas": "#22c55e",
            "unknown": "#cbd5e1",
        }

        for an, btn in self.buttons.items():
            elem = next((e for e in self.data["elements"] if e["atomicNumber"] == an), None)
            if not elem:
                continue
            if scheme == "family":
                family = elem.get("family", "")
                bg = self.FAMILY_COLORS.get(self._FAMILY_ALIASES.get(family.lower(), family) if isinstance(family, str) else family, "#94a3b8")
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(f"{elem['symbol']}\n{elem['atomicNumber']}")
            elif scheme == "state":
                state = str(elem.get("standardState", "unknown")).lower()
                bg = state_colors.get(state, "#cbd5e1")
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(self._button_label(elem, elem.get("standardState", "unknown")))
            else:
                if cmap is None or vmin is None or vmax is None:
                    bg = "#cbd5e1"
                    extra = "-"
                else:
                    val, unit_used = self._parse_quantity(elem.get(field), field)
                    if math.isnan(val):
                        bg = "#cbd5e1"
                        extra = "-"
                    elif log_scale and val <= 0:
                        bg = "#cbd5e1"
                        extra = "-"
                    else:
                        if log_scale:
                            t = (math.log10(val) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
                        else:
                            t = (val - vmin) / (vmax - vmin)
                        rgba = cmap(t)
                        bg = to_hex(rgba)
                        extra = self._format_numeric(val)
                btn.setStyleSheet(self._button_style(bg, self._text_contrast(bg)))
                btn.setText(self._button_label(elem, extra))

    def _render_table_colorbar(self, field: str, scheme_key: str, label: str) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors, ticker
        except Exception as exc:
            self.colorbar_label.setText(label)
            print(f"Table colorbar render fallback: {exc}", file=sys.stderr)
            return
        vals: list[float] = []
        unit_used: str | None = None
        log_scale = getattr(self, "scale_mode", "linear") == "log"
        for elem in self.data["elements"]:
            if field in elem and elem[field] not in ("", None):
                mag, u = self._parse_quantity(elem.get(field), field)
                if math.isnan(mag) or (log_scale and mag <= 0):
                    continue
                vals.append(mag)
                unit_used = unit_used or u
        if not vals:
            self.colorbar_label.setVisible(False)
            return
        vmin, vmax = min(vals), max(vals)
        if log_scale:
            vmin = max(vmin, sys.float_info.min)
            if math.isclose(vmin, vmax):
                vmax = vmin * 1.1
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if math.isclose(vmin, vmax):
                vmax = vmin + 1.0
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = self._get_cmap(self.current_table_cmap or self._scheme_default_cmap(scheme_key))
        fig, ax = plt.subplots(figsize=(4.2, 0.42))
        fig.patch.set_facecolor("#111827")
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
        unit_latex = self._unit_latex(unit_used)
        lbl = label if not unit_latex else f"{label}\\;[{unit_latex}]"
        cbar.set_label(f"$\\mathrm{{{lbl}}}$", color="#e5e7eb", fontsize=8)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=7, colors="#e5e7eb")
        cbar.outline.set_edgecolor("#e5e7eb")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e5e7eb")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(buf.getvalue())
        if pixmap.height() > 100:
            pixmap = pixmap.scaledToHeight(100, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.colorbar_label.setPixmap(pixmap)
        self.colorbar_label.setMinimumHeight(pixmap.height() + 6)
        self.colorbar_label.setStyleSheet("background-color: #111827; border: 1px solid #1f2937; padding: 4px;")
        self.colorbar_label.setVisible(True)
    def _select_atomic_number(self, atomic_number: int) -> None:
        elem = next((e for e in self.data["elements"] if e["atomicNumber"] == atomic_number), None)
        if not elem:
            return
        self.current_element = elem
        self._update_oxidation_combo(elem)
        for btn in self.buttons.values():
            btn.setChecked(False)
        if atomic_number in self.buttons:
            self.buttons[atomic_number].setChecked(True)
        self.bohr_view.set_element(elem)
        self.bohr_view.set_oxidation_state(self.current_oxidation_state)
        self._refresh_info()

    def _config_string(self, subshells: dict[tuple[int, int], int]) -> str:
        subshell_order = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
            (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
            (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
            (7, 0), (5, 3), (6, 2), (7, 1),
        ]
        parts = []
        label_map = {0: "s", 1: "p", 2: "d", 3: "f"}
        for n, l in subshell_order:
            if (n, l) in subshells:
                cnt = subshells[(n, l)]
                label = label_map.get(l, "?")
                parts.append(f"{n}{label}{cnt}")
        return " ".join(parts)

    def _config_abbrev(self, elem: dict, subshells: dict[tuple[int, int], int]) -> str:
        noble_gases = [
            (2, "He"), (10, "Ne"), (18, "Ar"), (36, "Kr"),
            (54, "Xe"), (86, "Rn"), (118, "Og")
        ]
        an = int(elem.get("atomicNumber", 0))
        ng = max((n for n in noble_gases if n[0] < an), default=None, key=lambda x: x[0])
        if not ng:
            return self._config_string(subshells)
        ng_electrons, ng_symbol = ng
        # subtract noble gas subshells
        _, ng_subshells = self.bohr_view._shell_counts({"atomicNumber": ng_electrons, "period": elem.get("period", 1)})
        remainder: dict[tuple[int, int], int] = {}
        for key, cnt in subshells.items():
            remainder[key] = max(0, cnt - ng_subshells.get(key, 0))
        remainder = {k: v for k, v in remainder.items() if v > 0}
        rest = self._config_string(remainder)
        return f"[{ng_symbol}] {rest}" if rest else f"[{ng_symbol}]"

    def _refresh_info(self) -> None:
        if not self.current_element:
            return
        elem = self.current_element
        shells, subshells = self.bohr_view._shell_counts(elem)
        if getattr(self, "show_abbrev_config", False):
            config = self._config_abbrev(elem, subshells)
        else:
            config = self._config_string(subshells)
        states = self._parse_oxidation_states(elem)
        formatted_states = self._format_oxidation_states(states)
        stp_note = "Values at STP (0 °C, 1 atm)."
        rows = []
        numeric_fields = {
            "atomicMass",
            "electronegativity",
            "atomicRadius",
            "ionizationEnergy",
            "electronAffinity",
            "meltingPoint",
            "boilingPoint",
            "density",
            "specificHeat",
        }
        for key, title in (
            ("symbol", "Symbol"),
            ("name", "Name"),
            ("atomicNumber", "Atomic Number"),
            ("family", "Family"),
            ("standardState", "State"),
            ("atomicMass", "Atomic Mass"),
            ("electronegativity", "Electronegativity"),
            ("atomicRadius", "Atomic Radius"),
            ("ionizationEnergy", "Ionization Energy"),
            ("electronAffinity", "Electron Affinity"),
            ("meltingPoint", "Melting Point"),
            ("boilingPoint", "Boiling Point"),
            ("density", "Density (at STP)"),
            ("specificHeat", "Specific Heat"),
            ("radioactive", "Radioactive"),
            ("occurrence", "Occurrence"),
            ("yearDiscovered", "Year Discovered"),
        ):
            if key in elem:
                val = elem[key]
                if key in numeric_fields:
                    mag, u = self._parse_quantity(val, key)
                    txt = "-" if math.isnan(mag) else self._format_numeric(mag)
                    if u:
                        txt = f"{txt} {self._unit_html(u)}"
                    rows.append(f"<tr><td><b>{title}</b></td><td>{txt}</td></tr>")
                else:
                    rows.append(f"<tr><td><b>{title}</b></td><td>{val}</td></tr>")
        rows.append(f"<tr><td><b>Oxidation States</b></td><td>{formatted_states}</td></tr>")
        rows.append(
            "<tr><td><b>Electron Config</b></td>"
            f"<td>{config} "
            "<a href='toggle-config'>(toggle)</a></td></tr>"
        )
        rows.append(f"<tr><td><b>STP</b></td><td>{stp_note}</td></tr>")
        if "name" in elem:
            pubchem_link = f"https://pubchem.ncbi.nlm.nih.gov/element/{elem['name']}"
            rows.append(f"<tr><td><b>PubChem</b></td><td><a href='{pubchem_link}'>{pubchem_link}</a></td></tr>")
        rows.append("<tr><td><b>Data Source</b></td><td><a href='https://pubchem.ncbi.nlm.nih.gov/'>PubChem</a></td></tr>")
        family_raw = elem.get("family", "")
        family_norm = self._FAMILY_ALIASES.get(str(family_raw).lower(), family_raw) if family_raw else ""
        fam_summary = self.FAMILY_SUMMARY.get(family_norm, "")
        table_html = (
            "<html><body>"
            "<table style='border-collapse: collapse; width: 100%;'>"
            + "".join(rows) +
            "</table>"
        )
        if fam_summary:
            table_html += (
                "<div style='margin-top:10px; padding:8px; border:1px solid #cbd5e1; border-radius:6px; background:#f8fafc;'>"
                f"<b>{family_norm} overview:</b> {fam_summary}"
                "</div>"
            )
        table_html += "</body></html>"
        self.info.setHtml(table_html)

class ElectronShellsTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = load_data()
        self.current_element: dict | None = None
        self.current_oxidation: int = 0

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Element:"))
        self.element_combo = QtWidgets.QComboBox()
        self._populate_elements()
        self.element_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.element_combo.setMaximumWidth(260)
        controls.addWidget(self.element_combo)
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(QtWidgets.QLabel("Oxidation state:"))
        self.oxidation_combo = QtWidgets.QComboBox()
        controls.addWidget(self.oxidation_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.bohr_view = PeriodicTableTab.BohrViewer()
        self.subshell_view = self.SubshellGridView(self.bohr_view)
        self.box_container = self.OrbitalBoxContainer(self.bohr_view)

        self.config_label = QtWidgets.QLabel()
        self.config_label.setWordWrap(True)
        layout.addWidget(self.config_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setOpaqueResize(True)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._wrap_with_label("Bohr model", self.bohr_view))
        splitter.addWidget(self._wrap_with_label("Expanded subshells", self.subshell_view))
        splitter.addWidget(self._wrap_with_label("Orbital box view", self.box_container))
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([600, 400, 400])
        layout.addWidget(splitter, 1)

        self.element_combo.currentIndexChanged.connect(self._on_element_change)
        self.oxidation_combo.currentIndexChanged.connect(self._on_oxidation_change)
        self.prev_btn.clicked.connect(lambda: self._step_element(-1))
        self.next_btn.clicked.connect(lambda: self._step_element(1))
        if self.element_combo.count() > 0:
            self.element_combo.setCurrentIndex(0)
            self._on_element_change()

    def _wrap_with_label(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        container.setStyleSheet(
            "QWidget { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; }"
            "QLabel { color: #0f172a; }"
        )
        return container

    def _populate_elements(self) -> None:
        self.element_combo.clear()
        elements = sorted(self.data["elements"], key=lambda e: e.get("atomicNumber", 0))
        for elem in elements:
            an = elem.get("atomicNumber", 0)
            sym = elem.get("symbol", "")
            name = elem.get("name", "")
            self.element_combo.addItem(f"{an}: {sym} — {name}", userData=elem)

    def _oxidation_items(self, elem: dict) -> list[tuple[str, int]]:
        states = self.bohr_view._parse_oxidation_states_local(elem)
        items = [("Neutral (0)", 0)]
        for s in states:
            label = f"Ion +{s}" if s > 0 else f"Ion {s}"
            items.append((label, s))
        return items

    def _on_element_change(self) -> None:
        data = self.element_combo.currentData()
        if not data:
            return
        self.current_element = data
        items = self._oxidation_items(data)
        self.oxidation_combo.blockSignals(True)
        self.oxidation_combo.clear()
        for label, val in items:
            self.oxidation_combo.addItem(label, userData=val)
        self.oxidation_combo.blockSignals(False)
        neutral_idx = 0
        for idx in range(self.oxidation_combo.count()):
            if self.oxidation_combo.itemData(idx) == 0:
                neutral_idx = idx
                break
        self.oxidation_combo.setCurrentIndex(neutral_idx)
        self.current_oxidation = self.oxidation_combo.currentData() or 0
        self._refresh_views()

    def _on_oxidation_change(self) -> None:
        self.current_oxidation = self.oxidation_combo.currentData() or 0
        self._refresh_views()

    def _step_element(self, delta: int) -> None:
        if self.element_combo.count() == 0:
            return
        idx = self.element_combo.currentIndex()
        new_idx = max(0, min(self.element_combo.count() - 1, idx + delta))
        if new_idx != idx:
            self.element_combo.setCurrentIndex(new_idx)

    def _refresh_views(self) -> None:
        if not self.current_element:
            return
        self.bohr_view.set_element(self.current_element)
        self.bohr_view.set_oxidation_state(int(self.current_oxidation))
        self.subshell_view.update_view(self.current_element, int(self.current_oxidation))
        self.box_container.update_view(self.current_element, int(self.current_oxidation))
        # update configs
        shells, subshells = self.bohr_view._shell_counts(self.current_element)
        full_cfg = self._config_string(subshells)
        abbrev = self._config_abbrev(self.current_element, subshells)
        self.config_label.setText(f"Full: {full_cfg}\nAbbrev: {abbrev}")

    def _config_string(self, subshells: dict[tuple[int, int], int]) -> str:
        subshell_order = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
            (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
            (5, 1), (6, 0), (4, 3), (5, 2), (6, 1),
            (7, 0), (5, 3), (6, 2), (7, 1),
        ]
        parts = []
        label_map = {0: "s", 1: "p", 2: "d", 3: "f"}
        for n, l in subshell_order:
            if (n, l) in subshells:
                cnt = subshells[(n, l)]
                parts.append(f"{n}{label_map.get(l, '?')}{cnt}")
        return " ".join(parts)

    def _config_abbrev(self, elem: dict, subshells: dict[tuple[int, int], int]) -> str:
        noble_gases = [
            (2, "He"), (10, "Ne"), (18, "Ar"), (36, "Kr"),
            (54, "Xe"), (86, "Rn"), (118, "Og")
        ]
        an = int(elem.get("atomicNumber", 0))
        ng = max((n for n in noble_gases if n[0] < an), default=None, key=lambda x: x[0])
        if not ng:
            return self._config_string(subshells)
        ng_electrons, ng_symbol = ng
        _, ng_subshells = self.bohr_view._shell_counts({"atomicNumber": ng_electrons, "period": elem.get("period", 1)})
        remainder: dict[tuple[int, int], int] = {}
        for key, cnt in subshells.items():
            remainder[key] = max(0, cnt - ng_subshells.get(key, 0))
        remainder = {k: v for k, v in remainder.items() if v > 0}
        rest = self._config_string(remainder)
        return f"[{ng_symbol}] {rest}" if rest else f"[{ng_symbol}]"

    class SubshellGridView(QtWidgets.QWidget):
        def __init__(self, bohr_view: "PeriodicTableTab.BohrViewer", parent=None):
            super().__init__(parent)
            self.bohr_view = bohr_view
            self.grid = QtWidgets.QGridLayout(self)
            self.grid.setAlignment(QtCore.Qt.AlignTop)
            self.grid.setHorizontalSpacing(10)
            self.grid.setVerticalSpacing(8)
            self.setMinimumSize(220, 220)

        def update_view(self, elem: dict, oxidation: int) -> None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            if not elem:
                return
            self.bohr_view.oxidation_state = oxidation
            shells, subshells = self.bohr_view._shell_counts(elem)
            l_labels = ["s", "p", "d", "f"]
            for col, label in enumerate(l_labels):
                hdr = QtWidgets.QLabel(label)
                hdr.setAlignment(QtCore.Qt.AlignCenter)
                hdr.setStyleSheet("font-weight: bold; color: #0f172a;")
                self.grid.addWidget(hdr, 0, col + 1)
            for n in range(1, len(shells) + 1):
                row_lbl = QtWidgets.QLabel(f"n={n}")
                row_lbl.setAlignment(QtCore.Qt.AlignCenter)
                row_lbl.setStyleSheet("color: #0f172a;")
                self.grid.addWidget(row_lbl, n, 0)
                lmax = min(3, n - 1)
                for l in range(0, lmax + 1):
                    cap = self.bohr_view._subshell_capacity(l)
                    filled = min(cap, subshells.get((n, l), 0))
                    view = self.bohr_view.SubshellMiniWidget(
                        f"{n}{l_labels[l]}",
                        filled,
                        cap,
                        self.bohr_view._angle_sets().get(l, []),
                        filled >= cap,
                        self,
                    )
                    self.grid.addWidget(view, n, l + 1)
            self.updateGeometry()

    class OrbitalBoxContainer(QtWidgets.QWidget):
        def __init__(self, bohr_view: "PeriodicTableTab.BohrViewer", parent=None):
            super().__init__(parent)
            self.bohr_view = bohr_view
            self.view = PeriodicTableTab.OrbitalBoxView(
                {}, 0, self.bohr_view._shell_counts, self.bohr_view._subshell_capacity, self
            )
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.view)
            self.setMinimumSize(240, 240)

        def update_view(self, elem: dict, oxidation: int) -> None:
            if not elem:
                return
            self.view.elem = elem
            self.view.oxidation = oxidation
            self.view.update()

class OrbSimWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OrbSim")
        self.setMinimumSize(1200, 720)
        self._settings = QtCore.QSettings("OrbSim", "OrbSim")
        self._theme_name = self._settings.value("theme", "Fluent Light")

        self._build_toolbar()
        self._build_menus()

        self.tabs = QtWidgets.QTabWidget()
        # Order: Periodic Table, Electron Shells, Atomic Orbitals, Bonding Orbitals
        self.tabs.addTab(PeriodicTableTab(), "Periodic Table")
        self.tabs.addTab(ElectronShellsTab(), "Electron Shells")
        self.tabs.addTab(AtomicOrbitalTab(), "Atomic Orbitals")
        self.tabs.addTab(BondingOrbitalTab(), "Bonding Orbitals")
        self._annotation_layers: dict[QtWidgets.QWidget, OrbSimWindow.AnnotationLayer] = {}
        self._annotation_toolbox: OrbSimWindow.AnnotationToolbox | None = None
        for i in range(self.tabs.count()):
            self._annotation_layer_for_tab(self.tabs.widget(i))
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # initialize show text/state
        self._show_annotations_act.setChecked(True)
        self._show_annotations_act.setText("Hide annotations")
        self._refresh_annotation_layers()

        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage("Drag elements into the visualization pane to begin.")
        self.apply_theme(self._theme_name)

    def closeEvent(self, event) -> None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            cleanup = getattr(tab, "cleanup", None)
            if callable(cleanup):
                cleanup()
        super().closeEvent(event)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Export/Annotations")
        toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        print_act = QtGui.QAction("Print", self)
        print_act.setIcon(qta.icon("fa5s.print"))
        print_act.triggered.connect(self._print_current_view)
        export_act = QtGui.QAction("Export...", self)
        export_act.setIcon(qta.icon("fa5s.file-export"))
        export_act.triggered.connect(self._export_current_view)
        copy_act = QtGui.QAction("Copy to clipboard", self)
        copy_act.setIcon(qta.icon("fa5s.copy"))
        copy_act.triggered.connect(self._copy_current_view)

        export_btn = QtWidgets.QToolButton()
        export_btn.setText("Export/Copy")
        export_menu = QtWidgets.QMenu(export_btn)
        export_menu.addAction(print_act)
        export_menu.addAction(export_act)
        export_menu.addAction(copy_act)
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(export_btn)

        annotate_act = QtGui.QAction("Annotation mode", self)
        annotate_act.setIcon(qta.icon("fa5s.edit"))
        annotate_act.setCheckable(True)
        annotate_act.toggled.connect(self._toggle_annotation_mode)
        show_act = QtGui.QAction("Show annotations", self)
        show_act.setIcon(qta.icon("fa5s.eye"))
        show_act.setCheckable(True)
        show_act.setChecked(False)
        show_act.toggled.connect(self._toggle_annotation_visibility)
        clear_act = QtGui.QAction("Clear annotations", self)
        clear_act.setIcon(qta.icon("fa5s.trash"))
        clear_act.triggered.connect(self._clear_annotations)
        save_act = QtGui.QAction("Save annotations", self)
        save_act.setIcon(qta.icon("fa5s.save"))
        save_act.triggered.connect(self._save_annotations)
        load_act = QtGui.QAction("Load annotations", self)
        load_act.setIcon(qta.icon("fa5s.folder-open"))
        load_act.triggered.connect(self._load_annotations)
        undo_act = QtGui.QAction("Undo annotation", self)
        undo_act.setIcon(qta.icon("fa5s.undo"))
        undo_act.triggered.connect(self._undo_annotation)
        redo_act = QtGui.QAction("Redo annotation", self)
        redo_act.setIcon(qta.icon("fa5s.redo"))
        redo_act.triggered.connect(self._redo_annotation)
        copy_ann_act = QtGui.QAction("Copy annotation", self)
        copy_ann_act.setIcon(qta.icon("fa5s.copy"))
        copy_ann_act.triggered.connect(self._copy_annotation)
        paste_ann_act = QtGui.QAction("Paste annotation", self)
        paste_ann_act.setIcon(qta.icon("fa5s.paste"))
        paste_ann_act.triggered.connect(self._paste_annotation)

        anno_btn = QtWidgets.QToolButton()
        anno_btn.setText("Annotations")
        anno_menu = QtWidgets.QMenu(anno_btn)
        anno_menu.addAction(annotate_act)
        anno_menu.addAction(show_act)
        anno_menu.addSeparator()
        anno_menu.addAction(undo_act)
        anno_menu.addAction(redo_act)
        anno_menu.addAction(copy_ann_act)
        anno_menu.addAction(paste_ann_act)
        anno_menu.addSeparator()
        anno_menu.addAction(clear_act)
        anno_menu.addAction(save_act)
        anno_menu.addAction(load_act)
        anno_btn.setMenu(anno_menu)
        anno_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(anno_btn)

        self._annotate_act = annotate_act
        self._show_annotations_act = show_act

    def _build_menus(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QtGui.QActionGroup(self)
        theme_group.setExclusive(True)
        for name in THEME_TOKENS.keys():
            action = QtGui.QAction(name, self)
            action.setCheckable(True)
            action.setChecked(name == self._theme_name)
            action.triggered.connect(lambda checked, n=name: self.apply_theme(n))
            theme_group.addAction(action)
            theme_menu.addAction(action)

    def apply_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        self._settings.setValue("theme", theme_name)
        tokens = get_theme_tokens(theme_name)
        app = QtWidgets.QApplication.instance()
        if app:
            apply_theme(app, tokens)
        for tab_index in range(self.tabs.count()):
            tab = self.tabs.widget(tab_index)
            for widget in tab.findChildren(DropPlotter):
                widget.apply_theme(tokens)
            for widget in tab.findChildren(PeriodicTableWidget):
                widget.apply_theme(tokens)
        self._refresh_annotation_layers()

    def _grab_current_view(self) -> QtGui.QPixmap:
        target: QtWidgets.QWidget = self.tabs.currentWidget() if getattr(self, "tabs", None) else self
        base = target.grab()
        layer = self._annotation_layer_for_tab(target, create=False)
        if layer and layer.has_annotations() and layer.isVisible():
            composed = QtGui.QPixmap(base.size())
            composed.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(composed)
            painter.drawPixmap(0, 0, base)
            layer.render(painter, QtCore.QPoint(), QtGui.QRegion())
            painter.end()
            return composed
        return base

    def _copy_current_view(self) -> None:
        pixmap = self._grab_current_view()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)

    def _print_current_view(self) -> None:
        pixmap = self._grab_current_view()
        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
        dialog = QtPrintSupport.QPrintDialog(printer, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            painter = QtGui.QPainter(printer)
            rect = painter.viewport()
            scaled = pixmap.scaled(rect.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.setViewport(rect.x(), rect.y(), scaled.width(), scaled.height())
            painter.setWindow(pixmap.rect())
            painter.drawPixmap(0, 0, scaled)
            painter.end()

    def _export_current_view(self) -> None:
        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf)"
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Export current view", "", filters)
        if not path:
            return
        fmt = "png"
        if selected.startswith("JPEG"):
            fmt = "jpg"
        elif selected.startswith("PDF") or path.lower().endswith(".pdf"):
            fmt = "pdf"
        elif path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
            fmt = "jpg"
        elif path.lower().endswith(".png"):
            fmt = "png"
        if fmt == "pdf" and not path.lower().endswith(".pdf"):
            path += ".pdf"
        if fmt == "png" and not path.lower().endswith(".png"):
            path += ".png"
        if fmt == "jpg" and not (path.lower().endswith(".jpg") or path.lower().endswith(".jpeg")):
            path += ".jpg"
        pixmap = self._grab_current_view()
        if fmt == "pdf":
            printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QtPrintSupport.QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(path)
            painter = QtGui.QPainter(printer)
            rect = printer.pageRect()
            scaled = pixmap.scaled(rect.size().toSize(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled)
            painter.end()
        else:
            pixmap.save(path, fmt.upper())

    def _on_tab_changed(self, index: int) -> None:
        self._refresh_annotation_layers()

    def _annotation_layer_for_tab(self, tab: QtWidgets.QWidget, create: bool = True):
        if tab in self._annotation_layers:
            return self._annotation_layers[tab]
        if not create:
            return None
        layer = self.AnnotationLayer(tab)
        layer.setGeometry(tab.rect())
        layer.setStyleSheet("background: transparent;")
        layer.hide()
        layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layer.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        layer.raise_()
        self._annotation_layers[tab] = layer
        tab.installEventFilter(self)
        return layer

    def _refresh_annotation_layers(self) -> None:
        show_flag = getattr(self, "_show_annotations_act", None).isChecked() if hasattr(self, "_show_annotations_act") else False
        annotate_flag = getattr(self, "_annotate_act", None).isChecked() if hasattr(self, "_annotate_act") else False
        current_tab = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        toolbox = self._get_annotation_toolbox()
        for tab, layer in self._annotation_layers.items():
            if not isinstance(tab, QtWidgets.QWidget):
                continue
            layer.setGeometry(tab.rect())
            if tab is current_tab:
                visible = show_flag or annotate_flag
                layer.setVisible(visible)
                layer.set_edit_mode(annotate_flag)
                layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not annotate_flag)
                layer.raise_()
                self._set_tab_interactive(tab, not annotate_flag)
                if toolbox:
                    toolbox.set_layer(layer)
                    if annotate_flag:
                        toolbox.show()
                    else:
                        toolbox.hide()
            else:
                layer.setVisible(False)
                layer.set_edit_mode(False)
                layer.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
                self._set_tab_interactive(tab, True)
        if hasattr(self, "_show_annotations_act"):
            self._show_annotations_act.setText("Hide annotations" if show_flag or annotate_flag else "Show annotations")

    def _set_tab_interactive(self, tab: QtWidgets.QWidget, enabled: bool) -> None:
        # Keep viewers enabled; the overlay intercepts events when annotating.
        return

    def _get_annotation_toolbox(self):
        if self._annotation_toolbox is None:
            self._annotation_toolbox = self.AnnotationToolbox(self)
            self._annotation_toolbox.hide()
        return self._annotation_toolbox

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Resize and isinstance(obj, QtWidgets.QWidget):
            layer = self._annotation_layers.get(obj)
            if layer:
                layer.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def _toggle_annotation_mode(self, enabled: bool) -> None:
        self._refresh_annotation_layers()

    def _toggle_annotation_visibility(self, visible: bool) -> None:
        if not visible and hasattr(self, "_annotate_act"):
            self._annotate_act.setChecked(False)
        self._refresh_annotation_layers()

    def _undo_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.undo()

    def _redo_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.redo()

    def _copy_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.copy_selected()

    def _paste_annotation(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.paste_copied()

    def _clear_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        layer.clear_annotations()

    def _save_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        if not layer.has_annotations():
            QtWidgets.QMessageBox.information(self, "Save annotations", "No annotations to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        data = {"annotations": layer.to_data()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_annotations(self) -> None:
        tab = self.tabs.currentWidget()
        layer = self._annotation_layer_for_tab(tab)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations", "", "YAML Files (*.yaml *.yml *.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            try:
                import yaml  # type: ignore
                loaded = yaml.safe_load(raw)
            except Exception:
                loaded = json.loads(raw)
            shapes = (loaded or {}).get("annotations", [])
            layer.from_data(shapes)
            layer.setVisible(True)
            self._show_annotations_act.setChecked(True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load annotations", f"Failed to load annotations: {exc}")

    class AnnotationLayer(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.shapes: list[dict] = []
            self.temp_shape: dict | None = None
            self.edit_mode = False
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            # style defaults
            self.mode = "arrow"
            self.line_color = "#0f172a"
            self.fill_color = "#0f172a"
            self.fill_mode = "none"
            self.line_style = "solid"
            self.line_thickness = 2
            self.line_opacity = 1.0
            self.fill_opacity = 0.4
            self.text_font = "Segoe UI"
            self.text_size = 12
            self.start_pos: QtCore.QPointF | None = None
            self.selected_idx: int | None = None
            self.drag_mode: str | None = None
            self.drag_offset: QtCore.QPointF | None = None
            self._resize_anchor: QtCore.QPointF | None = None
            self._snapshot_before_drag: list[dict] | None = None
            self.undo_stack: list[list[dict]] = []
            self.redo_stack: list[list[dict]] = []
            self.clipboard_shape: dict | None = None

        def has_annotations(self) -> bool:
            return bool(self.shapes)

        def clear_annotations(self) -> None:
            if self.shapes:
                self._push_undo()
            self.shapes.clear()
            self.temp_shape = None
            self.selected_idx = None
            self.update()

        def set_edit_mode(self, enabled: bool) -> None:
            self.edit_mode = enabled
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not enabled)
            self.update()

        def set_mode(self, mode: str) -> None:
            self.mode = mode
            self.temp_shape = None
            self.update()
            if mode == "eraser":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        def set_style(self, line_color: str, fill_color: str, fill: str, line: str, line_thickness: int, line_opacity: float, fill_opacity: float, font: str, font_size: int) -> None:
            self.line_color = line_color
            self.fill_color = fill_color
            self.fill_mode = fill
            self.line_style = line
            self.line_thickness = line_thickness
            self.line_opacity = line_opacity
            self.fill_opacity = fill_opacity
            self.text_font = font
            self.text_size = font_size
            self.update()

        def to_data(self) -> list[dict]:
            return [dict(s) for s in self.shapes]

        def from_data(self, data: list[dict]) -> None:
            self.shapes = []
            for item in data or []:
                if "type" in item:
                    self.shapes.append(dict(item))
            self.temp_shape = None
            self.selected_idx = None
            self.update()

        def _make_pen(self, color: str = None, thickness: int = None, line: str = None, opacity: float | None = None) -> QtGui.QPen:
            c = QtGui.QColor(color or self.line_color)
            alpha = int(255 * (opacity if opacity is not None else self.line_opacity))
            c.setAlpha(alpha)
            pen = QtGui.QPen(c, thickness or self.line_thickness)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine if (line or self.line_style) == "dashed" else QtCore.Qt.PenStyle.SolidLine)
            return pen

        def _make_brush(self, color: str = None, fill: str = None, opacity: float | None = None) -> QtGui.QBrush:
            fill_mode = (fill or self.fill_mode).lower()
            alpha = int(255 * (opacity if opacity is not None else self.fill_opacity))
            if fill_mode == "solid":
                col = QtGui.QColor(color or self.fill_color)
                col.setAlpha(alpha)
                return QtGui.QBrush(col)
            return QtCore.Qt.NoBrush

        def _shape_to_path(self, painter: QtGui.QPainter, shape: dict, temp: bool = False):
            stype = shape.get("type")
            start = QtCore.QPointF(*shape.get("start", (0, 0)))
            end = QtCore.QPointF(*shape.get("end", (0, 0)))
            pen = self._make_pen(
                shape.get("line_color", shape.get("color")),
                shape.get("thickness", shape.get("line_thickness")),
                shape.get("line", self.line_style),
                shape.get("opacity", self.line_opacity),
            )
            brush = self._make_brush(
                shape.get("fill_color", shape.get("color")),
                shape.get("fill", self.fill_mode),
                shape.get("fill_opacity", self.fill_opacity),
            )
            painter.setPen(pen)
            if stype == "rect":
                painter.setBrush(brush)
                painter.drawRect(QtCore.QRectF(start, end).normalized())
            elif stype == "ellipse":
                painter.setBrush(brush)
                painter.drawEllipse(QtCore.QRectF(start, end).normalized())
            elif stype == "arrow":
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawLine(start, end)
                self._draw_arrow_head(painter, start, end, pen.color(), pen.widthF(), pen.style())
            elif stype == "text":
                font = QtGui.QFont(shape.get("font", self.text_font), int(shape.get("font_size", self.text_size)))
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(shape.get("color", "#0f172a")), pen.width()))
                painter.drawText(end, shape.get("text", ""))

        def _draw_arrow_head(self, painter: QtGui.QPainter, start: QtCore.QPointF, end: QtCore.QPointF, color: QtGui.QColor, width: float, style: QtCore.Qt.PenStyle):
            line_vec = end - start
            angle = math.atan2(line_vec.y(), line_vec.x())
            length = max(10.0, 4.0 * width)
            theta = math.radians(25)
            p1 = end - QtCore.QPointF(length * math.cos(angle - theta), length * math.sin(angle - theta))
            p2 = end - QtCore.QPointF(length * math.cos(angle + theta), length * math.sin(angle + theta))
            poly = QtGui.QPolygonF([end, p1, p2])
            pen = QtGui.QPen(color, width)
            pen.setStyle(style)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawPolygon(poly)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            for shape in self.shapes:
                self._shape_to_path(painter, shape)
            if self.temp_shape:
                self._shape_to_path(painter, self.temp_shape, temp=True)
            # selection highlight
            if self.selected_idx is not None and 0 <= self.selected_idx < len(self.shapes):
                shape = self.shapes[self.selected_idx]
                rect = QtCore.QRectF(QtCore.QPointF(*shape.get("start", (0, 0))), QtCore.QPointF(*shape.get("end", (0, 0)))).normalized()
                pen = QtGui.QPen(QtGui.QColor("#38bdf8"), 1, QtCore.Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(rect.adjusted(-4, -4, 4, 4))
            painter.end()

        def _push_undo(self):
            self.undo_stack.append([dict(s) for s in self.shapes])
            self.redo_stack.clear()

        def undo(self):
            if not self.undo_stack:
                return
            self.redo_stack.append([dict(s) for s in self.shapes])
            self.shapes = self.undo_stack.pop()
            self.selected_idx = None
            self.temp_shape = None
            self.update()

        def redo(self):
            if not self.redo_stack:
                return
            self.undo_stack.append([dict(s) for s in self.shapes])
            self.shapes = self.redo_stack.pop()
            self.selected_idx = None
            self.temp_shape = None
            self.update()

        def copy_selected(self):
            if self.selected_idx is None or self.selected_idx >= len(self.shapes):
                return
            self.clipboard_shape = dict(self.shapes[self.selected_idx])

        def paste_copied(self):
            if not self.clipboard_shape:
                return
            shape = dict(self.clipboard_shape)
            # offset a bit
            sx, sy = shape.get("start", (0, 0))
            ex, ey = shape.get("end", (0, 0))
            shape["start"] = (sx + 10, sy + 10)
            shape["end"] = (ex + 10, ey + 10)
            self._push_undo()
            self.shapes.append(shape)
            self.selected_idx = len(self.shapes) - 1
            self.update()

        def _hit_test(self, pos: QtCore.QPointF):
            tolerance = 8.0
            for idx in range(len(self.shapes) - 1, -1, -1):
                shape = self.shapes[idx]
                stype = shape.get("type")
                start = QtCore.QPointF(*shape.get("start", (0, 0)))
                end = QtCore.QPointF(*shape.get("end", (0, 0)))
                rect = QtCore.QRectF(start, end).normalized()
                if stype in ("rect", "ellipse", "highlight"):
                    if rect.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(pos):
                        # corner detection
                        corners = [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]
                        for c in corners:
                            if QtCore.QLineF(pos, c).length() <= tolerance:
                                # anchor is opposite corner
                                opp = QtCore.QPointF(rect.center().x() * 2 - c.x(), rect.center().y() * 2 - c.y())
                                return idx, "resize_corner", opp
                        return idx, "move", pos - start
                elif stype == "arrow":
                    # distance to line
                    dist = _point_to_line_distance(pos, QtCore.QLineF(start, end))
                    if dist <= tolerance:
                        if QtCore.QLineF(end, pos).length() <= tolerance:
                            return idx, "resize_end", None
                        if QtCore.QLineF(start, pos).length() <= tolerance:
                            return idx, "resize_start", None
                        return idx, "move", pos - start
                elif stype == "text":
                    if QtCore.QLineF(pos, end).length() <= tolerance:
                        return idx, "move", pos - end
            return None, None, None

        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            pos = event.position()
            if self.mode == "eraser":
                hit_idx, mode, data = self._hit_test(pos)
                if hit_idx is not None:
                    self._push_undo()
                    self.shapes.pop(hit_idx)
                    self.selected_idx = None
                    self.update()
                return
            hit_idx, mode, data = self._hit_test(pos)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                self.drag_mode = mode
                self.drag_offset = data
                self._snapshot_before_drag = [dict(s) for s in self.shapes]
                if mode == "resize_corner":
                    self._resize_anchor = data
                return
            self.selected_idx = None
            self.start_pos = pos
            if self.mode == "text":
                text, ok = QtWidgets.QInputDialog.getText(self, "Add text", "Text:")
                if ok and text:
                    self._push_undo()
                    self.shapes.append({
                        "type": "text",
                        "text": text,
                        "start": (pos.x(), pos.y()),
                        "end": (pos.x(), pos.y()),
                        "line_color": self.line_color,
                        "fill_color": self.fill_color,
                        "line_thickness": self.line_thickness,
                        "line": self.line_style,
                        "fill": self.fill_mode,
                        "opacity": self.line_opacity,
                        "fill_opacity": self.fill_opacity,
                        "font": self.text_font,
                        "font_size": self.text_size,
                    })
                    self.selected_idx = len(self.shapes) - 1
                    self.update()
            else:
                self.temp_shape = {
                    "type": self.mode,
                    "start": (pos.x(), pos.y()),
                    "end": (pos.x(), pos.y()),
                    "line_color": self.line_color,
                    "fill_color": self.fill_color,
                    "line_thickness": self.line_thickness,
                    "line": self.line_style,
                    "fill": self.fill_mode,
                    "opacity": self.line_opacity,
                    "fill_opacity": self.fill_opacity,
                    "font": self.text_font,
                    "font_size": self.text_size,
                }
                self.update()

        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            pos = event.position()
            if self.temp_shape:
                self.temp_shape["end"] = (pos.x(), pos.y())
                self.update()
                return
            if self.selected_idx is not None and self.drag_mode:
                shape = self.shapes[self.selected_idx]
                if self.drag_mode == "move":
                    offset = self.drag_offset or QtCore.QPointF(0, 0)
                    start_pt = pos - offset
                    sx, sy = start_pt.x(), start_pt.y()
                    ex, ey = shape.get("end", (0, 0))
                    end_vec = QtCore.QPointF(ex, ey) - QtCore.QPointF(*shape.get("start", (0, 0)))
                    new_end = start_pt + end_vec
                    shape["start"] = (sx, sy)
                    shape["end"] = (new_end.x(), new_end.y())
                elif self.drag_mode in ("resize_corner", "resize_start", "resize_end"):
                    if self.drag_mode == "resize_corner" and self._resize_anchor is not None:
                        shape["start"] = (self._resize_anchor.x(), self._resize_anchor.y())
                        shape["end"] = (pos.x(), pos.y())
                    elif self.drag_mode == "resize_start":
                        shape["start"] = (pos.x(), pos.y())
                    elif self.drag_mode == "resize_end":
                        shape["end"] = (pos.x(), pos.y())
                self.update()

        def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
            if not self.edit_mode:
                return
            if self.temp_shape:
                pos = event.position()
                self.temp_shape["end"] = (pos.x(), pos.y())
                self._push_undo()
                self.shapes.append(self.temp_shape)
                self.selected_idx = len(self.shapes) - 1
                self.temp_shape = None
                self.update()
                return
            if self.selected_idx is not None and getattr(self, "_snapshot_before_drag", None) is not None:
                self.undo_stack.append(self._snapshot_before_drag)
                self.redo_stack.clear()
                self._snapshot_before_drag = None
                self.update()
            self.drag_mode = None
            self.drag_offset = None
            self._resize_anchor = None
            self.start_pos = None

    class AnnotationToolbox(QtWidgets.QDialog):
        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Annotation tools")
            self.setWindowFlag(QtCore.Qt.WindowType.Tool)
            self.layer: OrbSimWindow.AnnotationLayer | None = None
            layout = QtWidgets.QVBoxLayout(self)
            # mode buttons
            mode_row = QtWidgets.QHBoxLayout()
            self.mode_buttons: dict[str, QtWidgets.QToolButton] = {}
            modes = [
                ("Arrow", "arrow", "fa5s.arrow-right"),
                ("Rect", "rect", "fa5s.square"),
                ("Ellipse", "ellipse", "fa5s.circle"),
                ("Text", "text", "fa5s.font"),
                ("Eraser", "eraser", "fa5s.eraser"),
            ]
            for label, key, icon_name in modes:
                btn = QtWidgets.QToolButton()
                btn.setText(label)
                btn.setIcon(qta.icon(icon_name))
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, k=key: self._set_mode(k))
                mode_row.addWidget(btn)
                self.mode_buttons[key] = btn
            layout.addLayout(mode_row)

            form = QtWidgets.QFormLayout()
            self.color_combo = QtWidgets.QComboBox()
            colors = {
                "Black": "#0f172a",
                "Blue": "#2563eb",
                "Red": "#ef4444",
                "Green": "#22c55e",
                "Orange": "#f97316",
                "Purple": "#8b5cf6",
            }
            for name, val in colors.items():
                self.color_combo.addItem(name, userData=val)
            form.addRow("Color", self.color_combo)

            self.fill_color_combo = QtWidgets.QComboBox()
            for name, val in colors.items():
                self.fill_color_combo.addItem(name, userData=val)
            form.addRow("Fill color", self.fill_color_combo)

            self.fill_combo = QtWidgets.QComboBox()
            self.fill_combo.addItems(["None", "Solid"])
            form.addRow("Fill", self.fill_combo)

            self.line_combo = QtWidgets.QComboBox()
            self.line_combo.addItems(["Solid", "Dashed"])
            form.addRow("Line style", self.line_combo)

            self.line_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.line_opacity_slider.setRange(10, 100)
            self.line_opacity_slider.setValue(100)
            form.addRow("Line opacity (%)", self.line_opacity_slider)

            self.fill_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.fill_opacity_slider.setRange(10, 100)
            self.fill_opacity_slider.setValue(40)
            form.addRow("Fill opacity (%)", self.fill_opacity_slider)

            self.thickness_spin = QtWidgets.QSpinBox()
            self.thickness_spin.setRange(1, 12)
            self.thickness_spin.setValue(2)
            form.addRow("Thickness", self.thickness_spin)

            self.font_combo = QtWidgets.QFontComboBox()
            form.addRow("Font", self.font_combo)

            self.font_size_spin = QtWidgets.QSpinBox()
            self.font_size_spin.setRange(8, 48)
            self.font_size_spin.setValue(12)
            form.addRow("Font size", self.font_size_spin)

            layout.addLayout(form)

            for widget in (
                self.color_combo,
                self.fill_color_combo,
                self.fill_combo,
                self.line_combo,
                self.line_opacity_slider,
                self.fill_opacity_slider,
                self.thickness_spin,
                self.font_combo,
                self.font_size_spin,
            ):
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.currentIndexChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QSlider):
                    widget.valueChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.valueChanged.connect(self._apply_style)
                elif isinstance(widget, QtWidgets.QFontComboBox):
                    widget.currentFontChanged.connect(lambda _: self._apply_style())

            if self.mode_buttons:
                list(self.mode_buttons.values())[0].setChecked(True)
                self.current_mode = modes[0][1]
            else:
                self.current_mode = "arrow"

        def set_layer(self, layer: OrbSimWindow.AnnotationLayer) -> None:
            self.layer = layer
            self._apply_style()
            self.layer.set_mode(self.current_mode)

        def _set_mode(self, mode: str) -> None:
            self.current_mode = mode
            for key, btn in self.mode_buttons.items():
                btn.setChecked(key == mode)
            if self.layer:
                self.layer.set_mode(mode)
            if mode == "eraser":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            else:
                self.unsetCursor()

        def _apply_style(self) -> None:
            if not self.layer:
                return
            line_color = self.color_combo.currentData()
            fill_color = self.fill_color_combo.currentData()
            fill = self.fill_combo.currentText().lower()
            line = "dashed" if "dash" in self.line_combo.currentText().lower() else "solid"
            line_opacity = self.line_opacity_slider.value() / 100.0
            fill_opacity = self.fill_opacity_slider.value() / 100.0
            thickness = self.thickness_spin.value()
            font = self.font_combo.currentFont().family()
            font_size = self.font_size_spin.value()
            self.layer.set_style(line_color, fill_color, fill, line, thickness, line_opacity, fill_opacity, font, font_size)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            parent = self.parent()
            if isinstance(parent, OrbSimWindow) and hasattr(parent, "_annotate_act"):
                parent._annotate_act.setChecked(False)
                # keep annotations shown by default
                if hasattr(parent, "_show_annotations_act"):
                    parent._show_annotations_act.setChecked(True)
                parent._refresh_annotation_layers()
            super().closeEvent(event)


def _point_to_line_distance(point: QtCore.QPointF, line: QtCore.QLineF) -> float:
    """Perpendicular distance from point to a line segment."""
    x0, y0 = point.x(), point.y()
    x1, y1 = line.p1().x(), line.p1().y()
    x2, y2 = line.p2().x(), line.p2().y()
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den else 0.0


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OrbSimWindow()
    window.setMinimumSize(1400, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
