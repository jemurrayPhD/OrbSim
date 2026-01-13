from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigSummary:
    name: str
    symbol: str
    atomic_number: int
    oxidation_state: int
    total_electrons: int
    valence_shell: int
    valence_electrons: int
    d_electrons: int
    f_electrons: int
    unpaired_electrons: int
    electronegativity: float | None
    family: str
    oxidation_states: list[int]


def parse_oxidation_states(elem: dict) -> list[int]:
    raw = elem.get("oxidationStates") or elem.get("oxidationstates") or elem.get("oxidation_states")
    if raw is None:
        return []
    states: list[int] = []
    if isinstance(raw, (list, tuple)):
        for v in raw:
            try:
                states.append(int(v))
            except Exception:
                continue
    elif isinstance(raw, str):
        cleaned = raw.replace("'", "-").replace("+", " ").replace(",", " ")
        for part in cleaned.split():
            try:
                states.append(int(part))
            except Exception:
                continue
    return sorted(set(states))


def _subshell_degeneracy(l: int) -> int:
    return {0: 1, 1: 3, 2: 5, 3: 7}.get(l, 1)


def _count_unpaired(subshells: dict[tuple[int, int], int]) -> int:
    total = 0
    for (_n, l), electrons in subshells.items():
        deg = _subshell_degeneracy(l)
        cap = 2 * deg
        electrons = max(0, min(cap, int(electrons)))
        if electrons <= deg:
            total += electrons
        else:
            total += cap - electrons
    return total


def summarize_configuration(elem: dict, subshells: dict[tuple[int, int], int], oxidation: int) -> ConfigSummary:
    name = str(elem.get("name") or "")
    symbol = str(elem.get("symbol") or "")
    atomic_number = int(elem.get("atomicNumber") or elem.get("atomic_number") or 0)
    total_electrons = sum(int(v) for v in subshells.values())
    valence_shell = max((n for (n, _l), count in subshells.items() if count > 0), default=0)
    valence_electrons = sum(
        int(count) for (n, _l), count in subshells.items() if n == valence_shell
    )
    d_electrons = subshells.get((valence_shell - 1, 2), 0) if valence_shell > 1 else 0
    f_electrons = subshells.get((valence_shell - 2, 3), 0) if valence_shell > 2 else 0
    unpaired = _count_unpaired(subshells)
    try:
        en = elem.get("electronegativity")
        electronegativity = None if en in (None, "", "-") else float(en)
    except Exception:
        electronegativity = None
    family = str(elem.get("family") or elem.get("category") or elem.get("categoryName") or "")
    oxidation_states = parse_oxidation_states(elem)
    return ConfigSummary(
        name=name,
        symbol=symbol,
        atomic_number=atomic_number,
        oxidation_state=int(oxidation),
        total_electrons=int(total_electrons),
        valence_shell=int(valence_shell),
        valence_electrons=int(valence_electrons),
        d_electrons=int(d_electrons),
        f_electrons=int(f_electrons),
        unpaired_electrons=int(unpaired),
        electronegativity=electronegativity,
        family=family,
        oxidation_states=oxidation_states,
    )
