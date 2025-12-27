from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Atom:
    symbol: str
    charge: int
    position: np.ndarray


class MoleculeModel:
    def __init__(self) -> None:
        self.atoms: list[Atom] = []
        self.interactions_enabled = False

    def add_atom(self, symbol: str, charge: int, position: np.ndarray) -> Atom:
        atom = Atom(symbol=symbol, charge=charge, position=position)
        self.atoms.append(atom)
        return atom

    def clear(self) -> None:
        self.atoms.clear()

    def toggle_interactions(self, enabled: bool) -> None:
        self.interactions_enabled = enabled

    def minimize_energy(self) -> None:
        if not self.atoms:
            return

        count = len(self.atoms)
        angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
        radius = 2.0 + 0.4 * count
        positions = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
        positions *= radius
        center = positions.mean(axis=0)
        positions -= center

        for atom, position in zip(self.atoms, positions, strict=False):
            atom.position = position
