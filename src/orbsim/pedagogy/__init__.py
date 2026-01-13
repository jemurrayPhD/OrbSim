from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

_PHASE_RE = re.compile(r"\((aq|g|l|s)\)$", re.IGNORECASE)
_DOT_TRANSLATION = str.maketrans(
    {
        "\u00b7": ".",
        "\u2219": ".",
        "\u22c5": ".",
        "\u2022": ".",
        "\u30fb": ".",
    }
)

SECTION_LABELS = {
    "element_title": "Curated element notes",
    "compound_title": "Curated compound notes",
    "properties": "Key properties",
    "applications": "Real-life applications",
    "safety": "Safety and handling",
    "references": "References",
}

FALLBACK_ELEMENT = "No curated notes yet. Add one in src/orbsim/pedagogy/elements.json."
FALLBACK_COMPOUND = "No curated notes yet. Add one in src/orbsim/pedagogy/compounds.json."


def _read_resource_text(filename: str) -> str:
    try:
        return resources.files(__name__).joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        path = Path(__file__).resolve().parent / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
    return "{}"


@lru_cache(maxsize=1)
def load_elements() -> dict:
    return json.loads(_read_resource_text("elements.json"))


@lru_cache(maxsize=1)
def load_compounds() -> dict:
    return json.loads(_read_resource_text("compounds.json"))


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:].lower()


def normalize_formula(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return ""
    text = text.translate(_DOT_TRANSLATION)
    text = _PHASE_RE.sub("", text)
    return text.replace(" ", "")


def element_entry(symbol: str) -> dict | None:
    entries = load_elements()
    key = _normalize_symbol(symbol)
    return entries.get(key)


def compound_entry(formula: str) -> dict | None:
    entries = load_compounds()
    if not formula:
        return None
    direct = entries.get(formula)
    if direct:
        return direct
    normalized = normalize_formula(formula)
    if normalized and normalized in entries:
        return entries[normalized]
    if "." in normalized:
        base = normalized.split(".", 1)[0]
        return entries.get(base)
    return None


def _escape(text: str) -> str:
    return html.escape(str(text or ""))


def _format_list(items: list[str]) -> str:
    if not items:
        return ""
    lines = "".join(f"<li>{_escape(item)}</li>" for item in items if str(item).strip())
    if not lines:
        return ""
    return f"<ul>{lines}</ul>"


def _format_links(links: list[dict]) -> str:
    chunks: list[str] = []
    for entry in links:
        if not isinstance(entry, dict):
            continue
        label = _escape(entry.get("label", ""))
        url = str(entry.get("url", "")).strip()
        if not label or not url.startswith(("http://", "https://")):
            continue
        chunks.append(f"<a href=\"{_escape(url)}\">{label}</a>")
    return " | ".join(chunks)


def _format_entry_html(entry: dict, title_key: str, include_title: bool = True) -> str:
    title = _escape(entry.get("title") or SECTION_LABELS[title_key])
    summary = _escape(entry.get("summary") or "")
    properties = entry.get("properties") or []
    applications = entry.get("applications") or []
    safety = _escape(entry.get("safety") or "")
    links = entry.get("links") or []
    parts = [f"<p><b>{title}</b></p>"] if include_title else []
    if summary:
        parts.append(f"<p>{summary}</p>")
    if properties:
        parts.append(f"<p><b>{SECTION_LABELS['properties']}</b></p>")
        parts.append(_format_list(list(properties)))
    if applications:
        parts.append(f"<p><b>{SECTION_LABELS['applications']}</b></p>")
        parts.append(_format_list(list(applications)))
    if safety:
        parts.append(f"<p><b>{SECTION_LABELS['safety']}</b>: {safety}</p>")
    link_html = _format_links(list(links))
    if link_html:
        parts.append(f"<p><b>{SECTION_LABELS['references']}</b>: {link_html}</p>")
    return f"<div style=\"line-height:1.4;\">{''.join(parts)}</div>"


def element_notes_html(symbol: str, include_title: bool = True) -> str:
    entry = element_entry(symbol)
    if not entry:
        return f"<div style=\"line-height:1.4;\"><p><i>{_escape(FALLBACK_ELEMENT)}</i></p></div>"
    return _format_entry_html(entry, "element_title", include_title=include_title)


def compound_notes_html(formula: str, include_title: bool = True) -> str:
    entry = compound_entry(formula)
    if not entry:
        return f"<div style=\"line-height:1.4;\"><p><i>{_escape(FALLBACK_COMPOUND)}</i></p></div>"
    return _format_entry_html(entry, "compound_title", include_title=include_title)
