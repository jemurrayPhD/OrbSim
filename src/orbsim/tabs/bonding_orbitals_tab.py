from __future__ import annotations

import math
import sys

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets

from orbsim.orbitals import _ATOMIC_NUMBER, evaluate_orbital_values, field_from_grid
from orbsim.tabs.atomic_orbitals_tab import (
    AddOrbitalDialog,
    AtomicOrbitalsTab,
    EXAMPLE_MOLECULES,
    NUCLEUS_COLORS,
    PositionedOrbital,
)
from orbsim.ui.generated.ui_bonding_orbitals import Ui_BondingOrbitalsTab


class BondingOrbitalsTab(AtomicOrbitalsTab):
    """Tab for positioning atomic orbitals and deriving bonding/antibonding hybrids."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._ui_class = Ui_BondingOrbitalsTab
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
        self.slice_plane_resolution = 40
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
        self.nucleus_scale_spin.setRange(0.1, 2.0)
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

        layout.addWidget(three_d_group)
        layout.addWidget(slicing_group)

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

    def _nucleus_base_radius(self) -> float:
        bounds = None
        if self.base_grid is not None:
            bounds = self.base_grid.bounds
        if bounds is None and self._hybrid_meta:
            bounds = self._hybrid_meta.get("bounds")
        if bounds is None:
            return 0.12
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        return max(extent * 0.02, 0.08)

    def _render_nuclei(self) -> None:
        if not self.show_nuclei or not self.orbitals:
            return
        base_radius = self._nucleus_base_radius()
        for orb in self.orbitals:
            if not orb.visible:
                continue
            radius = base_radius * float(max(self.nucleus_scale, 0.05))
            color = self._nucleus_color(orb.symbol)
            offsets = [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.2, 0.2, 0.0]) * radius,
                np.array([-0.2, 0.2, 0.0]) * radius,
                np.array([0.2, -0.2, 0.0]) * radius,
                np.array([-0.2, -0.2, 0.0]) * radius,
            ]
            for scale, center_offset in zip([1.0, 0.5, 0.5, 0.5, 0.5], offsets, strict=False):
                orb_radius = radius * scale
                center = orb.position + center_offset
                try:
                    self.plotter.add_mesh(
                        pv.Sphere(radius=orb_radius, center=center),
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
                        pv.Sphere(radius=orb_radius * 0.7, center=center),
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
        clip_bounds = self.slice_box_bounds if self.slice_mode == "box" else None

        if self.current_representation == "volume":
            if clip_bounds:
                try:
                    volume_field = volume_field.copy()
                    volume_field.dataset = volume_field.dataset.clip_box(clip_bounds, invert=False)
                except Exception:
                    pass
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
        main_title = f"{key.title()} hybrid ({len(self.orbitals)} orbitals)"
        self.plotter.add_text(
            main_title,
            font_size=12,
            color=self._text_color,
            shadow=self._text_shadow,
            name="main_title",
            position="upper_left",
        )
        view_label = f"3D View ({self.current_representation})"
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
