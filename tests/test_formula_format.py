from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orbsim.chem.formula_format import format_formula, format_formula_from_string


class FormulaFormatTests(unittest.TestCase):
    def test_ionic_ordering(self) -> None:
        display = format_formula({"Na": 1, "Cl": 1})
        self.assertEqual(display.plain, "NaCl")
        self.assertEqual(display.rich, "NaCl")

    def test_organic_ordering_and_subscripts(self) -> None:
        display = format_formula({"C": 6, "H": 12, "O": 6})
        self.assertEqual(display.plain, "C\u2086H\u2081\u2082O\u2086")
        self.assertEqual(display.rich, "C<sub>6</sub>H<sub>12</sub>O<sub>6</sub>")

    def test_covalent_electronegativity_ordering(self) -> None:
        display = format_formula({"O": 1, "H": 2})
        self.assertEqual(display.plain, "H\u2082O")
        self.assertEqual(display.rich, "H<sub>2</sub>O")

    def test_polyatomic_parentheses(self) -> None:
        display = format_formula({"Mg": 1, "O": 2, "H": 2})
        self.assertEqual(display.plain, "Mg(OH)\u2082")
        self.assertEqual(display.rich, "Mg(OH)<sub>2</sub>")

    def test_phase_suffix(self) -> None:
        display = format_formula_from_string("HCl(aq)")
        self.assertEqual(display.plain, "HCl(aq)")
        self.assertEqual(display.rich, "HCl(aq)")


if __name__ == "__main__":
    unittest.main()
