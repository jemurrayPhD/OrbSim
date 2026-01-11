from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


def _load_elements() -> list[dict]:
    try:
        from periodic_table_cli.cli import load_data
    except Exception:
        load_data = None
    if load_data:
        try:
            data = load_data()
            elements = data.get("elements", [])
            return elements
        except Exception:
            pass
    return [
        {"symbol": "H", "name": "Hydrogen", "atomicNumber": 1, "electronegativity": 2.20, "family": "Nonmetal"},
        {"symbol": "He", "name": "Helium", "atomicNumber": 2, "electronegativity": None, "family": "Noble Gas"},
        {"symbol": "Li", "name": "Lithium", "atomicNumber": 3, "electronegativity": 0.98, "family": "Alkali Metal"},
        {"symbol": "Be", "name": "Beryllium", "atomicNumber": 4, "electronegativity": 1.57, "family": "Alkaline Earth Metal"},
        {"symbol": "B", "name": "Boron", "atomicNumber": 5, "electronegativity": 2.04, "family": "Metalloid"},
        {"symbol": "C", "name": "Carbon", "atomicNumber": 6, "electronegativity": 2.55, "family": "Nonmetal"},
        {"symbol": "N", "name": "Nitrogen", "atomicNumber": 7, "electronegativity": 3.04, "family": "Nonmetal"},
        {"symbol": "O", "name": "Oxygen", "atomicNumber": 8, "electronegativity": 3.44, "family": "Nonmetal"},
        {"symbol": "F", "name": "Fluorine", "atomicNumber": 9, "electronegativity": 3.98, "family": "Halogen"},
        {"symbol": "Ne", "name": "Neon", "atomicNumber": 10, "electronegativity": None, "family": "Noble Gas"},
        {"symbol": "Na", "name": "Sodium", "atomicNumber": 11, "electronegativity": 0.93, "family": "Alkali Metal"},
        {"symbol": "Mg", "name": "Magnesium", "atomicNumber": 12, "electronegativity": 1.31, "family": "Alkaline Earth Metal"},
        {"symbol": "Al", "name": "Aluminum", "atomicNumber": 13, "electronegativity": 1.61, "family": "Post-Transition Metal"},
        {"symbol": "Si", "name": "Silicon", "atomicNumber": 14, "electronegativity": 1.90, "family": "Metalloid"},
        {"symbol": "P", "name": "Phosphorus", "atomicNumber": 15, "electronegativity": 2.19, "family": "Nonmetal"},
        {"symbol": "S", "name": "Sulfur", "atomicNumber": 16, "electronegativity": 2.58, "family": "Nonmetal"},
        {"symbol": "Cl", "name": "Chlorine", "atomicNumber": 17, "electronegativity": 3.16, "family": "Halogen"},
        {"symbol": "Ar", "name": "Argon", "atomicNumber": 18, "electronegativity": None, "family": "Noble Gas"},
        {"symbol": "K", "name": "Potassium", "atomicNumber": 19, "electronegativity": 0.82, "family": "Alkali Metal"},
        {"symbol": "Ca", "name": "Calcium", "atomicNumber": 20, "electronegativity": 1.00, "family": "Alkaline Earth Metal"},
        {"symbol": "Sc", "name": "Scandium", "atomicNumber": 21, "electronegativity": 1.36, "family": "Transition Metal"},
        {"symbol": "Ti", "name": "Titanium", "atomicNumber": 22, "electronegativity": 1.54, "family": "Transition Metal"},
        {"symbol": "V", "name": "Vanadium", "atomicNumber": 23, "electronegativity": 1.63, "family": "Transition Metal"},
        {"symbol": "Cr", "name": "Chromium", "atomicNumber": 24, "electronegativity": 1.66, "family": "Transition Metal"},
        {"symbol": "Mn", "name": "Manganese", "atomicNumber": 25, "electronegativity": 1.55, "family": "Transition Metal"},
        {"symbol": "Fe", "name": "Iron", "atomicNumber": 26, "electronegativity": 1.83, "family": "Transition Metal"},
        {"symbol": "Co", "name": "Cobalt", "atomicNumber": 27, "electronegativity": 1.88, "family": "Transition Metal"},
        {"symbol": "Ni", "name": "Nickel", "atomicNumber": 28, "electronegativity": 1.91, "family": "Transition Metal"},
        {"symbol": "Cu", "name": "Copper", "atomicNumber": 29, "electronegativity": 1.90, "family": "Transition Metal"},
        {"symbol": "Zn", "name": "Zinc", "atomicNumber": 30, "electronegativity": 1.65, "family": "Transition Metal"},
        {"symbol": "Ga", "name": "Gallium", "atomicNumber": 31, "electronegativity": 1.81, "family": "Post-Transition Metal"},
        {"symbol": "Ge", "name": "Germanium", "atomicNumber": 32, "electronegativity": 2.01, "family": "Metalloid"},
        {"symbol": "As", "name": "Arsenic", "atomicNumber": 33, "electronegativity": 2.18, "family": "Metalloid"},
        {"symbol": "Se", "name": "Selenium", "atomicNumber": 34, "electronegativity": 2.55, "family": "Nonmetal"},
        {"symbol": "Br", "name": "Bromine", "atomicNumber": 35, "electronegativity": 2.96, "family": "Halogen"},
        {"symbol": "Kr", "name": "Krypton", "atomicNumber": 36, "electronegativity": 3.00, "family": "Noble Gas"},
    ]


@lru_cache(maxsize=1)
def _element_index() -> tuple[dict[int, dict], dict[str, dict]]:
    elements = _load_elements()
    by_number: dict[int, dict] = {}
    by_symbol: dict[str, dict] = {}
    for element in elements:
        z = int(element.get("atomicNumber") or element.get("atomic_number") or 0)
        symbol = str(element.get("symbol") or "").strip()
        if not z or not symbol:
            continue
        by_number[z] = element
        by_symbol[symbol.lower()] = element
    return by_number, by_symbol


def get_element(z: int) -> dict:
    by_number, _ = _element_index()
    return by_number.get(int(z), {})


def get_symbol(z: int) -> str:
    element = get_element(z)
    return str(element.get("symbol") or "")


def get_name(z: int) -> str:
    element = get_element(z)
    return str(element.get("name") or "")


def get_atomic_number(symbol: str) -> int:
    if not symbol:
        return 0
    _, by_symbol = _element_index()
    element = by_symbol.get(symbol.strip().lower())
    return int(element.get("atomicNumber") or element.get("atomic_number") or 0) if element else 0
