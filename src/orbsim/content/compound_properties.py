from __future__ import annotations

from dataclasses import dataclass

from orbsim.chem.elements import get_atomic_number, get_element, get_symbol


@dataclass(frozen=True)
class BondingSummary:
    bonding_sentence: str
    polarity_sentence: str


def _element_family(symbol: str) -> str:
    atomic_number = get_atomic_number(symbol)
    if atomic_number <= 0:
        return ""
    element = get_element(atomic_number)
    return str(element.get("family") or element.get("category") or element.get("categoryName") or "")


def _is_metal(symbol: str) -> bool:
    family = _element_family(symbol).lower()
    return "metal" in family and "nonmetal" not in family and "metalloid" not in family


def _is_nonmetal(symbol: str) -> bool:
    family = _element_family(symbol).lower()
    return "nonmetal" in family or "non-metal" in family


def _electronegativity(symbol: str) -> float | None:
    element = get_element(get_atomic_number(symbol))
    value = element.get("electronegativity")
    try:
        return None if value in (None, "", "-") else float(value)
    except Exception:
        return None


def describe_bonding_and_polarity(compound: dict) -> BondingSummary:
    elements = compound.get("elements", {})
    symbols = [get_symbol(int(z)) for z in elements.keys()]
    symbols = [s for s in symbols if s]
    if not symbols:
        return BondingSummary(
            bonding_sentence="Bonding is unknown because no element data is available.",
            polarity_sentence="Polarity is unknown because no element data is available.",
        )

    has_metal = any(_is_metal(symbol) for symbol in symbols)
    has_nonmetal = any(_is_nonmetal(symbol) for symbol in symbols)

    if has_metal and has_nonmetal:
        bonding_sentence = (
            "Bonding is likely ionic because the composition mixes metals and nonmetals, "
            "so electrons tend to transfer from cations to anions."
        )
    elif has_metal and not has_nonmetal:
        bonding_sentence = (
            "Bonding is likely metallic because the composition is dominated by metals, "
            "which share delocalized electrons."
        )
    else:
        bonding_sentence = (
            "Bonding is likely covalent because the elements are primarily nonmetals "
            "that share electrons."
        )

    en_values = [value for symbol in symbols if (value := _electronegativity(symbol)) is not None]
    if not en_values:
        polarity_sentence = (
            "Polarity cannot be estimated because electronegativity data is missing for one or more elements."
        )
        return BondingSummary(bonding_sentence=bonding_sentence, polarity_sentence=polarity_sentence)

    max_en = max(en_values)
    min_en = min(en_values)
    delta_en = max_en - min_en
    if delta_en < 0.4:
        category = "mostly nonpolar"
    elif delta_en < 1.0:
        category = "moderately polar"
    else:
        category = "strongly polar"
    polarity_sentence = (
        f"Polarity estimate: the electronegativity spread is about {delta_en:.2f}, "
        f"suggesting a {category} bond network. Molecular geometry can still shift the overall polarity."
    )
    return BondingSummary(bonding_sentence=bonding_sentence, polarity_sentence=polarity_sentence)
