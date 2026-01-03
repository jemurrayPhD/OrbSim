from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementInfo:
    symbol: str
    name: str
    atomic_number: int
    electronegativity: float | None
    category: str


ELEMENTS_H_KR: list[ElementInfo] = [
    ElementInfo("H", "Hydrogen", 1, 2.20, "nonmetal"),
    ElementInfo("He", "Helium", 2, None, "noble_gas"),
    ElementInfo("Li", "Lithium", 3, 0.98, "metal"),
    ElementInfo("Be", "Beryllium", 4, 1.57, "metal"),
    ElementInfo("B", "Boron", 5, 2.04, "metalloid"),
    ElementInfo("C", "Carbon", 6, 2.55, "nonmetal"),
    ElementInfo("N", "Nitrogen", 7, 3.04, "nonmetal"),
    ElementInfo("O", "Oxygen", 8, 3.44, "nonmetal"),
    ElementInfo("F", "Fluorine", 9, 3.98, "nonmetal"),
    ElementInfo("Ne", "Neon", 10, None, "noble_gas"),
    ElementInfo("Na", "Sodium", 11, 0.93, "metal"),
    ElementInfo("Mg", "Magnesium", 12, 1.31, "metal"),
    ElementInfo("Al", "Aluminum", 13, 1.61, "metal"),
    ElementInfo("Si", "Silicon", 14, 1.90, "metalloid"),
    ElementInfo("P", "Phosphorus", 15, 2.19, "nonmetal"),
    ElementInfo("S", "Sulfur", 16, 2.58, "nonmetal"),
    ElementInfo("Cl", "Chlorine", 17, 3.16, "nonmetal"),
    ElementInfo("Ar", "Argon", 18, None, "noble_gas"),
    ElementInfo("K", "Potassium", 19, 0.82, "metal"),
    ElementInfo("Ca", "Calcium", 20, 1.00, "metal"),
    ElementInfo("Sc", "Scandium", 21, 1.36, "metal"),
    ElementInfo("Ti", "Titanium", 22, 1.54, "metal"),
    ElementInfo("V", "Vanadium", 23, 1.63, "metal"),
    ElementInfo("Cr", "Chromium", 24, 1.66, "metal"),
    ElementInfo("Mn", "Manganese", 25, 1.55, "metal"),
    ElementInfo("Fe", "Iron", 26, 1.83, "metal"),
    ElementInfo("Co", "Cobalt", 27, 1.88, "metal"),
    ElementInfo("Ni", "Nickel", 28, 1.91, "metal"),
    ElementInfo("Cu", "Copper", 29, 1.90, "metal"),
    ElementInfo("Zn", "Zinc", 30, 1.65, "metal"),
    ElementInfo("Ga", "Gallium", 31, 1.81, "metal"),
    ElementInfo("Ge", "Germanium", 32, 2.01, "metalloid"),
    ElementInfo("As", "Arsenic", 33, 2.18, "metalloid"),
    ElementInfo("Se", "Selenium", 34, 2.55, "nonmetal"),
    ElementInfo("Br", "Bromine", 35, 2.96, "nonmetal"),
    ElementInfo("Kr", "Krypton", 36, 3.00, "noble_gas"),
]

SYMBOL_TO_ELEMENT: dict[str, ElementInfo] = {elem.symbol: elem for elem in ELEMENTS_H_KR}
ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {elem.atomic_number: elem.symbol for elem in ELEMENTS_H_KR}

METAL_CATEGORIES = {"metal"}
NONMETAL_CATEGORIES = {"nonmetal"}
METALLOID_CATEGORIES = {"metalloid"}
NOBLE_GAS_CATEGORIES = {"noble_gas"}


def is_metal(symbol: str) -> bool:
    element = SYMBOL_TO_ELEMENT.get(symbol)
    return element is not None and element.category in METAL_CATEGORIES


def is_nonmetal(symbol: str) -> bool:
    element = SYMBOL_TO_ELEMENT.get(symbol)
    return element is not None and element.category in NONMETAL_CATEGORIES


def electronegativity(symbol: str) -> float | None:
    element = SYMBOL_TO_ELEMENT.get(symbol)
    return element.electronegativity if element else None
