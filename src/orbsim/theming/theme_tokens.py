from __future__ import annotations

import platform


def _default_font_family() -> str:
    if platform.system().lower().startswith("win"):
        return "Segoe UI"
    if platform.system().lower().startswith("darwin"):
        return "San Francisco"
    return "Inter"


THEME_TOKENS: dict[str, dict] = {
    "Fluent Light": {
        "meta": {"name": "Fluent Light", "mode": "light"},
        "colors": {
            "bg": "#f5f7fb",
            "surface": "#ffffff",
            "surfaceAlt": "#eef2f7",
            "text": "#0f172a",
            "textMuted": "#4b5563",
            "border": "#cbd5e1",
            "accent": "#2563eb",
            "accentHover": "#1d4ed8",
            "focusRing": "#0ea5e9",
        },
        "radii": {"sm": 6, "md": 10, "lg": 16},
        "spacing": {"xs": 4, "sm": 8, "md": 12, "lg": 16},
        "font": {"family": _default_font_family(), "baseSize": 10, "titleSize": 12, "monoFamily": "Consolas"},
    },
    "Fluent Dark": {
        "meta": {"name": "Fluent Dark", "mode": "dark"},
        "colors": {
            "bg": "#0b1220",
            "surface": "#111827",
            "surfaceAlt": "#1f2937",
            "text": "#f8fafc",
            "textMuted": "#cbd5f5",
            "border": "#334155",
            "accent": "#60a5fa",
            "accentHover": "#3b82f6",
            "focusRing": "#fbbf24",
        },
        "radii": {"sm": 6, "md": 10, "lg": 16},
        "spacing": {"xs": 4, "sm": 8, "md": 12, "lg": 16},
        "font": {"family": _default_font_family(), "baseSize": 10, "titleSize": 12, "monoFamily": "Consolas"},
    },
    "High Contrast": {
        "meta": {"name": "High Contrast", "mode": "high_contrast"},
        "colors": {
            "bg": "#000000",
            "surface": "#000000",
            "surfaceAlt": "#111111",
            "text": "#ffffff",
            "textMuted": "#e5e7eb",
            "border": "#ffffff",
            "accent": "#ffff00",
            "accentHover": "#ffd600",
            "focusRing": "#00ffff",
        },
        "radii": {"sm": 0, "md": 0, "lg": 0},
        "spacing": {"xs": 4, "sm": 8, "md": 12, "lg": 16},
        "font": {"family": _default_font_family(), "baseSize": 11, "titleSize": 13, "monoFamily": "Consolas"},
    },
}


def get_theme_tokens(name: str) -> dict:
    return THEME_TOKENS.get(name, THEME_TOKENS["Fluent Light"])
