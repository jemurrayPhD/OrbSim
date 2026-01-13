from __future__ import annotations

from dataclasses import dataclass


# Aufbau filling order with capacities for s, p, d, f subshells.
SUBSHELL_ORDER: tuple[tuple[int, int, int], ...] = (
    (1, 0, 2),
    (2, 0, 2),
    (2, 1, 6),
    (3, 0, 2),
    (3, 1, 6),
    (4, 0, 2),
    (3, 2, 10),
    (4, 1, 6),
    (5, 0, 2),
    (4, 2, 10),
    (5, 1, 6),
    (6, 0, 2),
    (4, 3, 14),
    (5, 2, 10),
    (6, 1, 6),
    (7, 0, 2),
    (5, 3, 14),
    (6, 2, 10),
    (7, 1, 6),
)

SUBSHELL_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}

# Known ground-state aufbau exceptions for neutral atoms.
# Notes: commonly cited in general chemistry and inorganic chemistry texts and
# standard periodic table references (RSC, Britannica).
AUFBAU_EXCEPTION_ADJUSTMENTS: dict[int, dict[tuple[int, int], int]] = {
    24: {(4, 0): -1, (3, 2): 1},  # Cr
    29: {(4, 0): -1, (3, 2): 1},  # Cu
    42: {(5, 0): -1, (4, 2): 1},  # Mo
    46: {(5, 0): -2, (4, 2): 2},  # Pd
    47: {(5, 0): -1, (4, 2): 1},  # Ag
    79: {(6, 0): -1, (5, 2): 1},  # Au
}


@dataclass(frozen=True)
class AufbauExceptionNote:
    electron_count: int
    expected_config: str
    actual_config: str
    explanation: str
    impact: str


AUFBAU_EXCEPTION_NOTES: dict[int, tuple[str, str]] = {
    24: (
        "The 3d and 4s subshells are close in energy. Promoting one 4s electron "
        "creates a half-filled 3d subshell, which is stabilized by exchange energy.",
        "This often enhances paramagnetism (more unpaired electrons) and supports "
        "multiple oxidation states such as +2, +3, and +6.",
    ),
    29: (
        "A filled 3d10 subshell is especially stable, so one 4s electron is promoted "
        "to complete 3d10.",
        "Cu+ (d10) is commonly diamagnetic, while Cu2+ is also prevalent and often "
        "gives colored complexes.",
    ),
    42: (
        "The 4d and 5s subshells are near-degenerate. Promoting one 5s electron "
        "yields a half-filled 4d5 subshell.",
        "Multiple oxidation states are common, and unpaired d electrons contribute "
        "to magnetic behavior and colored complexes.",
    ),
    46: (
        "The 4d subshell can be stabilized when fully filled, so both 5s electrons "
        "shift into 4d to reach 4d10.",
        "Many Pd(0) compounds are diamagnetic (d10), and Pd(II) chemistry is "
        "especially prominent in catalysis.",
    ),
    47: (
        "Completing the 4d10 subshell lowers the energy, so one 5s electron is promoted.",
        "Ag+ (d10) is common and typically diamagnetic; Ag(0) and Ag(I) compounds "
        "often have characteristic coordination chemistry.",
    ),
    79: (
        "Relativistic effects and near-degenerate 5d/6s energies favor a filled 5d10 "
        "subshell with a single 6s electron.",
        "Gold shows stable +1 and +3 oxidation states; d10 configurations can reduce "
        "magnetic moments in many Au(I) complexes.",
    ),
}


def fill_subshells(electron_count: int, apply_exceptions: bool = True) -> dict[tuple[int, int], int]:
    subshells: dict[tuple[int, int], int] = {}
    remaining = max(0, int(electron_count))
    for n, l, cap in SUBSHELL_ORDER:
        if remaining <= 0:
            break
        fill = min(cap, remaining)
        subshells[(n, l)] = fill
        remaining -= fill
    if apply_exceptions:
        adjustments = AUFBAU_EXCEPTION_ADJUSTMENTS.get(int(electron_count))
        if adjustments:
            for (n, l), delta in adjustments.items():
                subshells[(n, l)] = max(0, subshells.get((n, l), 0) + delta)
    return subshells


def subshells_to_config(subshells: dict[tuple[int, int], int]) -> str:
    parts: list[str] = []
    for n, l, _cap in SUBSHELL_ORDER:
        cnt = subshells.get((n, l))
        if cnt:
            parts.append(f"{n}{SUBSHELL_LABELS.get(l, '?')}{cnt}")
    return " ".join(parts)


def expected_aufbau_subshells(electron_count: int) -> dict[tuple[int, int], int]:
    return fill_subshells(electron_count, apply_exceptions=False)


def actual_aufbau_subshells(electron_count: int) -> dict[tuple[int, int], int]:
    return fill_subshells(electron_count, apply_exceptions=True)


def build_aufbau_exception_note(
    electron_count: int,
    expected_subshells: dict[tuple[int, int], int],
    actual_subshells: dict[tuple[int, int], int],
) -> AufbauExceptionNote | None:
    if expected_subshells == actual_subshells:
        return None
    expected = subshells_to_config(expected_subshells)
    actual = subshells_to_config(actual_subshells)
    explanation, impact = AUFBAU_EXCEPTION_NOTES.get(
        int(electron_count),
        (
            "Subshell energies are close, so a small electron rearrangement can lower the total energy.",
            "This can influence bonding tendencies and magnetic behavior, but details depend on the compound.",
        ),
    )
    return AufbauExceptionNote(
        electron_count=int(electron_count),
        expected_config=expected,
        actual_config=actual,
        explanation=explanation,
        impact=impact,
    )
