# OrbSim

Python-based atomic and molecular orbital simulation and visualization.

## Features
- Qt6 + PyVista GUI for atomic orbital exploration.
- Drag-and-drop from a periodic table pane into a 3D visualization pane.
- Switch between probability density and wavefunction visualization.
- Cyclic colormap to show phase for wavefunctions.
- Configurable opacity and colormap settings.
- Molecule builder with adjustable ionization, distances, and a simple energy minimizer.

## Getting started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
orbsim
```

## Notes
This prototype focuses on interaction and visualization scaffolding. Orbital physics and
energy minimization are simplified to keep the UI responsive and extensible.
