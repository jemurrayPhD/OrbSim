from __future__ import annotations

from dataclasses import dataclass
import html
import re

from orbsim.chem.elements import get_atomic_number, get_element, get_symbol
from orbsim.chem.formula_parser import parse_formula


@dataclass(frozen=True)
class FormulaDisplay:
    plain: str
    rich: str


_SUBSCRIPT_MAP = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
}
_SUPERSCRIPT_MAP = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207a",
    "-": "\u207b",
}


@dataclass(frozen=True)
class PolyatomicIon:
    formula: str
    composition: dict[str, int]
    role: str  # "cation" or "anion"


# Common polyatomic ions for grouping and parentheses in ionic formulas.
# Sources: standard general chemistry nomenclature references (IUPAC, LibreTexts).
POLYATOMIC_IONS: tuple[PolyatomicIon, ...] = (
    PolyatomicIon("NH4", {"N": 1, "H": 4}, "cation"),
    PolyatomicIon("OH", {"O": 1, "H": 1}, "anion"),
    PolyatomicIon("NO3", {"N": 1, "O": 3}, "anion"),
    PolyatomicIon("NO2", {"N": 1, "O": 2}, "anion"),
    PolyatomicIon("SO4", {"S": 1, "O": 4}, "anion"),
    PolyatomicIon("SO3", {"S": 1, "O": 3}, "anion"),
    PolyatomicIon("CO3", {"C": 1, "O": 3}, "anion"),
    PolyatomicIon("HCO3", {"H": 1, "C": 1, "O": 3}, "anion"),
    PolyatomicIon("PO4", {"P": 1, "O": 4}, "anion"),
    PolyatomicIon("HPO4", {"H": 1, "P": 1, "O": 4}, "anion"),
    PolyatomicIon("H2PO4", {"H": 2, "P": 1, "O": 4}, "anion"),
    PolyatomicIon("ClO4", {"Cl": 1, "O": 4}, "anion"),
    PolyatomicIon("ClO3", {"Cl": 1, "O": 3}, "anion"),
    PolyatomicIon("ClO2", {"Cl": 1, "O": 2}, "anion"),
    PolyatomicIon("ClO", {"Cl": 1, "O": 1}, "anion"),
    PolyatomicIon("CrO4", {"Cr": 1, "O": 4}, "anion"),
    PolyatomicIon("Cr2O7", {"Cr": 2, "O": 7}, "anion"),
    PolyatomicIon("MnO4", {"Mn": 1, "O": 4}, "anion"),
    PolyatomicIon("CN", {"C": 1, "N": 1}, "anion"),
    PolyatomicIon("CH3COO", {"C": 2, "H": 3, "O": 2}, "anion"),
)


def _to_symbol_counts(composition: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in composition.items():
        try:
            count = int(value)
        except Exception:
            continue
        if count <= 0:
            continue
        if isinstance(key, int):
            symbol = get_symbol(int(key))
        else:
            symbol = str(key).strip()
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + count
    return counts


def _family(symbol: str) -> str:
    if not symbol:
        return ""
    element = get_element(get_atomic_number(symbol))
    return str(element.get("family") or element.get("category") or element.get("categoryName") or "").lower()


def _is_metal(symbol: str) -> bool:
    family = _family(symbol)
    return "metal" in family and "nonmetal" not in family and "metalloid" not in family


def _is_nonmetal(symbol: str) -> bool:
    family = _family(symbol)
    return "nonmetal" in family or "non-metal" in family


def _electronegativity(symbol: str) -> float | None:
    element = get_element(get_atomic_number(symbol))
    value = element.get("electronegativity")
    try:
        return None if value in (None, "", "-") else float(value)
    except Exception:
        return None


def _detect_context(symbol_counts: dict[str, int]) -> str:
    symbols = set(symbol_counts)
    has_metal = any(_is_metal(sym) for sym in symbols)
    has_nonmetal = any(_is_nonmetal(sym) for sym in symbols)
    if has_metal and has_nonmetal:
        return "ionic"
    if "C" in symbols and "H" in symbols and not has_metal:
        return "organic"
    return "covalent"


def _subscript_number(value: int) -> str:
    return "".join(_SUBSCRIPT_MAP.get(ch, ch) for ch in str(value))


def _superscript_text(value: str) -> str:
    return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in value)


def _format_token(symbol: str, count: int, html_mode: bool) -> str:
    if html_mode:
        escaped = html.escape(symbol)
        return escaped if count <= 1 else f"{escaped}<sub>{count}</sub>"
    base = symbol
    return base if count <= 1 else f"{base}{_subscript_number(count)}"


def _format_group(formula: str, html_mode: bool) -> str:
    if not formula:
        return ""
    if html_mode:
        parts = []
        idx = 0
        while idx < len(formula):
            ch = formula[idx]
            if ch.isdigit():
                num = ch
                idx += 1
                while idx < len(formula) and formula[idx].isdigit():
                    num += formula[idx]
                    idx += 1
                parts.append(f"<sub>{num}</sub>")
            else:
                parts.append(html.escape(ch))
                idx += 1
        return "".join(parts)
    output = []
    idx = 0
    while idx < len(formula):
        ch = formula[idx]
        if ch.isdigit():
            num = ch
            idx += 1
            while idx < len(formula) and formula[idx].isdigit():
                num += formula[idx]
                idx += 1
            output.append(_subscript_number(int(num)))
        else:
            output.append(ch)
            idx += 1
    return "".join(output)


def _format_charge(charge: int | None, html_mode: bool) -> str:
    if charge is None or charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(int(charge))
    text = f"{magnitude if magnitude != 1 else ''}{sign}"
    if html_mode:
        return f"<sup>{text}</sup>"
    return _superscript_text(text)


def _ion_match(counts: dict[str, int], ion: PolyatomicIon) -> int | None:
    if not counts:
        return None
    multiplier = None
    for symbol, base_count in ion.composition.items():
        count = counts.get(symbol, 0)
        if count <= 0 or count % base_count != 0:
            return None
        factor = count // base_count
        multiplier = factor if multiplier is None else min(multiplier, factor)
    if multiplier is None or multiplier <= 0:
        return None
    for symbol, count in counts.items():
        base = ion.composition.get(symbol)
        if base is None:
            return None
        if count != base * multiplier:
            return None
    return multiplier


def _match_polyatomic_group(counts: dict[str, int], role: str) -> tuple[PolyatomicIon | None, int]:
    for ion in POLYATOMIC_IONS:
        if ion.role != role:
            continue
        multiplier = _ion_match(counts, ion)
        if multiplier:
            return ion, multiplier
    return None, 0


def _ordered_symbols(symbol_counts: dict[str, int], context: str) -> list[str]:
    symbols = list(symbol_counts)
    if context == "organic":
        ordered: list[str] = []
        if "C" in symbol_counts:
            ordered.append("C")
        if "H" in symbol_counts:
            ordered.append("H")
        remaining = sorted([s for s in symbols if s not in ordered], key=str.upper)
        return ordered + remaining
    if context == "ionic":
        metals = [s for s in symbols if _is_metal(s)]
        nonmetals = [s for s in symbols if s not in metals]
        metals.sort(key=lambda s: (get_atomic_number(s) or 999, s))
        nonmetals.sort(key=lambda s: (s.upper()))
        return metals + nonmetals
    def sort_key(sym: str) -> tuple[int, float, str]:
        en = _electronegativity(sym)
        if en is None:
            return (1, 999.0, sym.upper())
        return (0, en, sym.upper())
    return sorted(symbols, key=sort_key)


def format_formula(
    composition: dict,
    *,
    context: str | None = None,
    charge: int | None = None,
) -> FormulaDisplay:
    symbol_counts = _to_symbol_counts(composition)
    if not symbol_counts:
        return FormulaDisplay("", "")
    resolved_context = context or _detect_context(symbol_counts)

    if resolved_context == "ionic":
        metals = {s: c for s, c in symbol_counts.items() if _is_metal(s)}
        nonmetals = {s: c for s, c in symbol_counts.items() if s not in metals}
        poly_cation: PolyatomicIon | None = None
        cation_multiplier = 0
        if not metals:
            poly_cation, cation_multiplier = _match_polyatomic_group(symbol_counts, "cation")
            if poly_cation:
                remaining = dict(symbol_counts)
                for sym, base_count in poly_cation.composition.items():
                    remaining[sym] = remaining.get(sym, 0) - base_count * cation_multiplier
                    if remaining[sym] <= 0:
                        remaining.pop(sym, None)
                metals = {}
                nonmetals = remaining
        poly_anion, anion_multiplier = _match_polyatomic_group(nonmetals, "anion")

        parts_plain: list[str] = []
        parts_html: list[str] = []

        if poly_cation:
            group_plain = _format_group(poly_cation.formula, False)
            group_html = _format_group(poly_cation.formula, True)
            if cation_multiplier > 1:
                parts_plain.append(f"({group_plain}){_subscript_number(cation_multiplier)}")
                parts_html.append(f"({group_html})<sub>{cation_multiplier}</sub>")
            else:
                parts_plain.append(group_plain)
                parts_html.append(group_html)
        else:
            for sym in _ordered_symbols(metals, "ionic"):
                parts_plain.append(_format_token(sym, metals[sym], False))
                parts_html.append(_format_token(sym, metals[sym], True))

        if poly_anion:
            group_plain = _format_group(poly_anion.formula, False)
            group_html = _format_group(poly_anion.formula, True)
            if anion_multiplier > 1:
                parts_plain.append(f"({group_plain}){_subscript_number(anion_multiplier)}")
                parts_html.append(f"({group_html})<sub>{anion_multiplier}</sub>")
            else:
                parts_plain.append(group_plain)
                parts_html.append(group_html)
        else:
            for sym in _ordered_symbols(nonmetals, "ionic"):
                parts_plain.append(_format_token(sym, nonmetals[sym], False))
                parts_html.append(_format_token(sym, nonmetals[sym], True))

        plain = "".join(parts_plain) + _format_charge(charge, False)
        rich = "".join(parts_html) + _format_charge(charge, True)
        return FormulaDisplay(plain=plain, rich=rich)

    ordered = _ordered_symbols(symbol_counts, resolved_context)
    plain = "".join(_format_token(sym, symbol_counts[sym], False) for sym in ordered) + _format_charge(charge, False)
    rich = "".join(_format_token(sym, symbol_counts[sym], True) for sym in ordered) + _format_charge(charge, True)
    return FormulaDisplay(plain=plain, rich=rich)


_PHASE_RE = re.compile(r"^(.*?)(\((aq|g|l|s)\))$", re.IGNORECASE)
_HYDRATE_SPLIT_RE = re.compile(r"[.\u00b7]")


def _format_literal(formula: str, html_mode: bool) -> str:
    output: list[str] = []
    idx = 0
    while idx < len(formula):
        ch = formula[idx]
        if ch == "^":
            idx += 1
            charge = ""
            while idx < len(formula) and (formula[idx].isdigit() or formula[idx] in "+-"):
                charge += formula[idx]
                idx += 1
            if charge:
                output.append(f"<sup>{html.escape(charge)}</sup>" if html_mode else _superscript_text(charge))
            continue
        if ch.isdigit():
            num = ch
            idx += 1
            while idx < len(formula) and formula[idx].isdigit():
                num += formula[idx]
                idx += 1
            output.append(f"<sub>{num}</sub>" if html_mode else _subscript_number(int(num)))
            continue
        output.append(html.escape(ch) if html_mode else ch)
        idx += 1
    return "".join(output)


def format_formula_from_string(formula: str, *, context: str | None = None) -> FormulaDisplay:
    if not formula:
        return FormulaDisplay("", "")
    phase = ""
    core = formula.strip()
    match = _PHASE_RE.match(core)
    if match:
        core = match.group(1)
        phase = match.group(2)
    parts = [part for part in _HYDRATE_SPLIT_RE.split(core) if part]
    rich_parts: list[str] = []
    plain_parts: list[str] = []
    for part in parts:
        try:
            counts = parse_formula(part)
            display = format_formula(counts, context=context)
            rich_parts.append(display.rich)
            plain_parts.append(display.plain)
        except Exception:
            rich_parts.append(_format_literal(part, True))
            plain_parts.append(_format_literal(part, False))
    joiner_rich = "&middot;" if len(parts) > 1 else ""
    joiner_plain = "\u00b7" if len(parts) > 1 else ""
    rich = joiner_rich.join(rich_parts)
    plain = joiner_plain.join(plain_parts)
    if phase:
        rich = f"{rich}{html.escape(phase)}"
        plain = f"{plain}{phase}"
    return FormulaDisplay(plain=plain, rich=rich)
