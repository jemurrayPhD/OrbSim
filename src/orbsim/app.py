from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyvista as pv

from orbsim.molecule import MoleculeModel
from orbsim.orbitals import make_orbital_mesh
from orbsim.widgets import CollapsibleGroup, DropPlotter, PeriodicTableList


class AtomicOrbitalTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_symbol = "H"
        self.current_mode = "probability"
        self.current_cmap = "viridis"
        self.use_amplitude_opacity = False
        self.mesh_actor = None

        self.table = PeriodicTableList()
        self.table.element_dragged.connect(self._update_symbol_from_element)

        self.plotter_frame = DropPlotter()
        self.plotter_frame.element_dropped.connect(self._update_symbol)
        self.plotter = self.plotter_frame.plotter
        self.plotter.set_background("#111827")

        self.controls = self._build_controls()

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.plotter_frame, 4)
        layout.addWidget(self.controls, 2)

        self._render_orbital()

    def _build_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        mode_group = CollapsibleGroup("Visualization")
        mode_layout = QtWidgets.QFormLayout(mode_group)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["probability", "wavefunction"])
        self.mode_combo.currentTextChanged.connect(self._set_mode)
        mode_layout.addRow("Mode", self.mode_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "twilight", "coolwarm"])
        self.cmap_combo.currentTextChanged.connect(self._set_cmap)
        mode_layout.addRow("Colormap", self.cmap_combo)

        self.opacity_combo = QtWidgets.QComboBox()
        self.opacity_combo.addItems(["solid", "amplitude"])
        self.opacity_combo.currentTextChanged.connect(self._set_opacity)
        mode_layout.addRow("Opacity", self.opacity_combo)

        layout.addWidget(mode_group)
        layout.addStretch()
        return container

    def _update_symbol_from_element(self, element) -> None:
        self._update_symbol(element.symbol)

    def _update_symbol(self, symbol: str) -> None:
        self.current_symbol = symbol
        self._render_orbital()

    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        if mode == "wavefunction" and self.current_cmap == "viridis":
            self.current_cmap = "twilight"
            self.cmap_combo.setCurrentText("twilight")
        self._render_orbital()

    def _set_cmap(self, cmap: str) -> None:
        self.current_cmap = cmap
        self._render_orbital()

    def _set_opacity(self, opacity: str) -> None:
        self.use_amplitude_opacity = opacity == "amplitude"
        self._render_orbital()

    def _render_orbital(self) -> None:
        field = make_orbital_mesh(self.current_mode, use_amplitude_opacity=self.use_amplitude_opacity)
        self.plotter.clear()
        self.plotter.add_text(
            f"{self.current_symbol} orbital",
            font_size=12,
            color="white",
        )
        self.plotter.add_mesh(
            field.mesh,
            scalars=field.scalars,
            cmap=self.current_cmap,
            opacity=field.opacity if field.opacity is not None else 0.85,
            specular=0.4,
        )
        self.plotter.reset_camera()


class MoleculeBuilderTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = MoleculeModel()

        self.table = PeriodicTableList()
        self.table.element_dragged.connect(self._add_atom_from_element)

        self.plotter_frame = DropPlotter()
        self.plotter_frame.element_dropped.connect(self._add_atom_from_symbol)
        self.plotter = self.plotter_frame.plotter
        self.plotter.set_background("#0b1320")

        self.controls = self._build_controls()

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.plotter_frame, 4)
        layout.addWidget(self.controls, 2)

        self._refresh_scene()

    def _build_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        config_group = CollapsibleGroup("Molecule controls")
        config_layout = QtWidgets.QFormLayout(config_group)

        self.charge_spin = QtWidgets.QSpinBox()
        self.charge_spin.setRange(-3, 3)
        config_layout.addRow("Ionization", self.charge_spin)

        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setRange(0.5, 6.0)
        self.distance_spin.setSingleStep(0.1)
        self.distance_spin.setValue(1.4)
        config_layout.addRow("Distance", self.distance_spin)

        self.interactions_toggle = QtWidgets.QCheckBox("Show interactions")
        self.interactions_toggle.toggled.connect(self._toggle_interactions)
        config_layout.addRow(self.interactions_toggle)

        self.minimize_button = QtWidgets.QPushButton("Minimize energy")
        self.minimize_button.clicked.connect(self._minimize_energy)
        config_layout.addRow(self.minimize_button)

        self.clear_button = QtWidgets.QPushButton("Clear molecule")
        self.clear_button.clicked.connect(self._clear_molecule)
        config_layout.addRow(self.clear_button)

        layout.addWidget(config_group)
        layout.addStretch()
        return container

    def _add_atom_from_element(self, element) -> None:
        self._add_atom_from_symbol(element.symbol)

    def _add_atom_from_symbol(self, symbol: str) -> None:
        distance = self.distance_spin.value()
        if not self.model.atoms:
            position = np.zeros(3)
        else:
            last_pos = self.model.atoms[-1].position
            position = last_pos + np.array([distance, 0.0, 0.0])
        self.model.add_atom(symbol, self.charge_spin.value(), position)
        self._refresh_scene()

    def _toggle_interactions(self, enabled: bool) -> None:
        self.model.toggle_interactions(enabled)
        self._refresh_scene()

    def _minimize_energy(self) -> None:
        self.model.minimize_energy()
        self._refresh_scene()

    def _clear_molecule(self) -> None:
        self.model.clear()
        self._refresh_scene()

    def _refresh_scene(self) -> None:
        self.plotter.clear()
        if not self.model.atoms:
            self.plotter.add_text("Drag atoms here to build a molecule", color="white")
            self.plotter.reset_camera()
            return

        for atom in self.model.atoms:
            color = "#f97316" if atom.charge > 0 else "#38bdf8" if atom.charge < 0 else "#a7f3d0"
            sphere = pv.Sphere(radius=0.4, center=atom.position)
            label = f"{atom.symbol}{'+' if atom.charge > 0 else '-' if atom.charge < 0 else ''}"
            self.plotter.add_mesh(sphere, color=color)
            self.plotter.add_point_labels([atom.position], [label], font_size=14, text_color="white")

        if self.model.interactions_enabled and len(self.model.atoms) > 1:
            for first, second in zip(self.model.atoms[:-1], self.model.atoms[1:], strict=False):
                line = pv.Line(first.position, second.position)
                tube = line.tube(radius=0.07)
                self.plotter.add_mesh(tube, color="#fbbf24", opacity=0.7)
                midpoint = (first.position + second.position) / 2
                cloud = pv.Sphere(radius=0.35, center=midpoint)
                self.plotter.add_mesh(cloud, color="#e879f9", opacity=0.5)

        self.plotter.reset_camera()


class OrbSimWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OrbSim")
        self.setMinimumSize(1200, 720)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(AtomicOrbitalTab(), "Atomic Orbitals")
        self.tabs.addTab(MoleculeBuilderTab(), "Molecule Builder")

        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage("Drag elements into the visualization pane to begin.")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OrbSimWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
