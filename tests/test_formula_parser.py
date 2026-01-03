from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orbsim.chem.formula_parser import parse_formula


class FormulaParserTests(unittest.TestCase):
    def test_simple_formula(self) -> None:
        self.assertEqual(parse_formula("H2O"), {"H": 2, "O": 1})

    def test_multi_digit_counts(self) -> None:
        self.assertEqual(parse_formula("C6H12O6"), {"C": 6, "H": 12, "O": 6})

    def test_parentheses(self) -> None:
        self.assertEqual(parse_formula("Ca(OH)2"), {"Ca": 1, "O": 2, "H": 2})

    def test_invalid_formula_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_formula("2H")

    def test_unmatched_parentheses(self) -> None:
        with self.assertRaises(ValueError):
            parse_formula("NaCl)")


if __name__ == "__main__":
    unittest.main()
