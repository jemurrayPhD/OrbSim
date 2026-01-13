from __future__ import annotations

from orbsim.chem.aufbau import AufbauExceptionNote, SUBSHELL_LABELS, SUBSHELL_ORDER
from orbsim.chem.electron_configuration import ConfigSummary


PROPERTIES_TITLE = "Configuration properties"

_LABEL_TO_L = {label: l for l, label in SUBSHELL_LABELS.items()}


def _electronegativity_text(summary: ConfigSummary) -> str:
    if summary.electronegativity is None:
        return "Electronegativity data is unavailable for this element."
    value = float(summary.electronegativity)
    if value < 1.0:
        trend = (
            "Low electronegativity suggests metallic bonding and a tendency to lose electrons to form cations."
        )
    elif value < 2.0:
        trend = (
            "Moderate electronegativity suggests electron sharing; bonds with more electronegative elements are polar."
        )
    elif value < 3.0:
        trend = (
            "High electronegativity indicates a strong pull on shared electrons, so polar covalent bonding is common."
        )
    else:
        trend = (
            "Very high electronegativity indicates strong electron attraction and a tendency to form anions."
        )
    return f"Electronegativity is about {value:.2f}. {trend}"


def properties_html(summary: ConfigSummary) -> str:
    core_electrons = max(summary.total_electrons - summary.valence_electrons, 0)
    family = summary.family.lower()
    bonding_hint = ""
    if "noble gas" in family:
        bonding_hint = "A full valence shell makes the atom very unreactive."
    elif "alkali metal" in family:
        bonding_hint = "A single valence electron is easily lost, so +1 ions are common."
    elif "alkaline earth" in family:
        bonding_hint = "Two valence electrons are often lost, so +2 ions are common."
    elif "halogen" in family:
        bonding_hint = "One electron short of a full shell, halogens often gain one electron or share in covalent bonds."
    elif "transition" in family or "lanthanide" in family or "actinide" in family:
        bonding_hint = "Partially filled d or f subshells enable multiple oxidation states and coordination chemistry."
    elif "nonmetal" in family:
        bonding_hint = "Valence electrons are held more tightly, so electron sharing is common."

    ox_states = summary.oxidation_states
    ox_text = ""
    if ox_states:
        formatted = ", ".join(f"+{s}" if s > 0 else str(s) for s in ox_states)
        ox_text = f"Common oxidation states include {formatted}."

    magnetism = (
        f"{summary.unpaired_electrons} unpaired electron(s) suggest paramagnetism."
        if summary.unpaired_electrons > 0
        else "All electrons are paired, so diamagnetism is expected."
    )
    d_block = ""
    if summary.d_electrons > 0:
        d_block = f"The (n-1)d subshell has {summary.d_electrons} electron(s), which often drives transition-metal behavior."
    elif summary.f_electrons > 0:
        d_block = f"The (n-2)f subshell has {summary.f_electrons} electron(s), typical for f-block chemistry."

    en_text = _electronegativity_text(summary)
    detail_parts = [part for part in (bonding_hint, ox_text, en_text) if part]
    detail_html = "".join(f"<p>{part}</p>" for part in detail_parts)

    return (
        "<div style=\"line-height:1.4;\">"
        f"<p><b>Configuration context</b>: {summary.name} ({summary.symbol}) has {summary.total_electrons} electrons "
        f"at oxidation state {summary.oxidation_state:+d}.</p>"
        "<ul>"
        f"<li>Valence shell n={summary.valence_shell} with {summary.valence_electrons} valence electron(s).</li>"
        f"<li>{magnetism}</li>"
        f"{f'<li>{d_block}</li>' if d_block else ''}"
        "</ul>"
        f"<p>Core electrons ({core_electrons}) shield the nucleus, so valence electrons feel a reduced effective charge.</p>"
        f"{detail_html}"
        "</div>"
    )


def _parse_config_map(config: str) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for token in config.split():
        idx = 0
        n_str = ""
        while idx < len(token) and token[idx].isdigit():
            n_str += token[idx]
            idx += 1
        if not n_str or idx >= len(token):
            continue
        l_char = token[idx]
        idx += 1
        count_str = token[idx:]
        if not count_str.isdigit():
            continue
        l_val = _LABEL_TO_L.get(l_char)
        if l_val is None:
            continue
        result[(int(n_str), int(l_val))] = int(count_str)
    return result


def _format_config_line(
    subshells: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
    diff_keys: set[tuple[int, int]],
    diff_color: str | None,
) -> str:
    tokens: list[str] = []
    for n_val, l_val in subshells:
        label = SUBSHELL_LABELS.get(l_val, "?")
        count = counts.get((n_val, l_val), 0)
        text = f"{n_val}{label}{count}"
        if (n_val, l_val) in diff_keys and diff_color:
            text = f"<span style=\"color:{diff_color}; font-weight:600;\">{text}</span>"
        tokens.append(text)
    return " ".join(tokens)


def aufbau_note_html(note: AufbauExceptionNote, diff_color: str | None = None) -> str:
    if not diff_color:
        diff_color = "#b91c1c"
    expected_map = _parse_config_map(note.expected_config)
    actual_map = _parse_config_map(note.actual_config)
    subshells = [
        (n_val, l_val)
        for n_val, l_val, _cap in SUBSHELL_ORDER
        if (n_val, l_val) in expected_map or (n_val, l_val) in actual_map
    ]
    diff_keys = {
        (n_val, l_val)
        for (n_val, l_val) in subshells
        if expected_map.get((n_val, l_val), 0) != actual_map.get((n_val, l_val), 0)
    }
    expected_line = _format_config_line(subshells, expected_map, diff_keys, diff_color)
    actual_line = _format_config_line(subshells, actual_map, diff_keys, diff_color)
    if not expected_line or not actual_line:
        expected_line = note.expected_config
        actual_line = note.actual_config
    return (
        "<div style=\"line-height:1.4;\">"
        "<p><b>Aufbau exception detected</b></p>"
        "<table style=\"border-collapse:collapse;\">"
        "<tr>"
        "<td valign=\"top\" style=\"padding-right:8px;\"><b>Expected (aufbau):</b></td>"
        f"<td>{expected_line}</td>"
        "</tr>"
        "<tr>"
        "<td valign=\"top\" style=\"padding-right:8px;\"><b>Observed:</b></td>"
        f"<td>{actual_line}</td>"
        "</tr>"
        "</table>"
        f"<p><b>Why:</b> {note.explanation}</p>"
        f"<p><b>Impact:</b> {note.impact}</p>"
        "</div>"
    )
