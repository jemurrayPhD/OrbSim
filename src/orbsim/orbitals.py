from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt, pi

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class OrbitalField:
    dataset: pv.DataSet
    scalar_name: str
    opacity: np.ndarray | float | str | None
    representation: str
    iso_fraction: float
    cumulative_probability: float | None


# Approximate valence orbital selection for the first three periods.
_ORBITAL_BY_SYMBOL: dict[str, tuple[int, int]] = {
    "H": (1, 0),
    "He": (1, 0),
    "Li": (2, 0),
    "Be": (2, 0),
    "B": (2, 1),
    "C": (2, 1),
    "N": (2, 1),
    "O": (2, 1),
    "F": (2, 1),
    "Ne": (2, 1),
    "Na": (3, 0),
    "Mg": (3, 0),
    "Al": (3, 1),
    "Si": (3, 1),
    "P": (3, 1),
    "S": (3, 1),
    "Cl": (3, 1),
    "Ar": (3, 1),
}

_BOHR_TO_ANGSTROM = 0.529177
_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


def default_quantum_numbers(symbol: str) -> tuple[int, int]:
    """Provide a reasonable (n, l) starting point for a given element symbol."""
    return _ORBITAL_BY_SYMBOL.get(symbol, (2, 0))


def occupied_orbitals(symbol: str) -> list[tuple[int, int, int]]:
    """Return (n,l,m) orbitals occupied for a neutral atom up to Ar (3p)."""
    electrons = _ATOMIC_NUMBER.get(symbol, 1)
    sequence: list[tuple[int, int, int]] = []
    filling_order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]
    for n, l in filling_order:
        for m in range(-l, l + 1):
            if electrons <= 0:
                break
            sequence.append((n, l, m))
            electrons -= min(2, electrons)
        if electrons <= 0:
            break
    return sequence


def _associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Associated Legendre P_l^m(x) with a small explicit fallback."""
    m_abs = abs(m)
    try:
        from scipy.special import lpmv  # type: ignore

        return lpmv(m_abs, l, x)
    except Exception:
        if l == 0 and m_abs == 0:
            return np.ones_like(x)
        if l == 1 and m_abs == 0:
            return x
        if l == 1 and m_abs == 1:
            return -np.sqrt(1 - x**2)
        if l == 2 and m_abs == 0:
            return 0.5 * (3 * x**2 - 1)
        if l == 2 and m_abs == 1:
            return -3 * x * np.sqrt(1 - x**2)
        if l == 2 and m_abs == 2:
            return 3 * (1 - x**2)
        if l == 3 and m_abs == 0:
            return 0.5 * (5 * x**3 - 3 * x)
        if l == 3 and m_abs == 1:
            return -0.5 * (15 * x**2 - 3) * np.sqrt(1 - x**2)
        if l == 3 and m_abs == 2:
            return 15 * x * (1 - x**2)
        if l == 3 and m_abs == 3:
            return -15 * (1 - x**2) ** 1.5
    return np.zeros_like(x)


def _real_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    m = int(m)
    if l < 0 or abs(m) > l:
        return np.zeros_like(theta)
    x = np.cos(theta)
    p_lm = _associated_legendre(l, m, x)
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    if m == 0:
        return norm * p_lm
    factor = sqrt(2) * norm * p_lm
    if m > 0:
        return factor * np.cos(m * phi)
    return factor * np.sin(abs(m) * phi)


def _radial_component(n: int, l: int, r_bohr: np.ndarray) -> np.ndarray:
    """Loose approximation of hydrogenic radial parts (atomic units)."""
    if n == 1 and l == 0:
        return 2.0 * np.exp(-r_bohr)
    if n == 2 and l == 0:
        return (2 - r_bohr) * np.exp(-r_bohr / 2)
    if n == 2 and l == 1:
        return r_bohr * np.exp(-r_bohr / 2)
    if n == 3 and l == 0:
        return (27 - 18 * r_bohr + 2 * r_bohr**2) * np.exp(-r_bohr / 3)
    if n == 3 and l == 1:
        return (6 - r_bohr) * r_bohr * np.exp(-r_bohr / 3)
    if n == 3 and l == 2:
        return r_bohr**2 * np.exp(-r_bohr / 3)
    scale = max(float(n), 1.0)
    return (r_bohr**l) * np.exp(-r_bohr / scale)


def make_orbital_mesh(
    symbol: str,
    mode: str,
    n: int | None = None,
    l: int | None = None,
    m: int = 0,
    resolution: int = 120,
    representation: str = "surface",
    iso_fraction: float = 0.85,
) -> OrbitalField:
    """Sample a hydrogenic orbital and extract an iso-surface or volume."""
    default_n, default_l = default_quantum_numbers(symbol)
    n_val = max(int(n or default_n), 1)
    l_val = min(max(int(l or default_l), 0), max(n_val - 1, 0))
    m_val = int(np.clip(m, -l_val, l_val))

    span = 3.0 * n_val  # angstroms
    axis = np.linspace(-span, span, resolution)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r_cart = np.sqrt(X**2 + Y**2 + Z**2)
    cos_theta = np.clip(Z / (r_cart + 1e-9), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(Y, X)
    r_bohr = r_cart / _BOHR_TO_ANGSTROM

    radial = _radial_component(n_val, l_val, r_bohr)
    angular = _real_spherical_harmonic(l_val, m_val, theta, phi)
    psi = radial * angular
    amplitude = np.abs(psi)
    probability = amplitude**2
    phase = np.angle(psi.astype(complex))

    spacing = (axis[1] - axis[0],) * 3
    origin = (-span, -span, -span)
    grid = pv.ImageData(dimensions=(resolution, resolution, resolution), spacing=spacing, origin=origin)
    grid["psi"] = psi.ravel(order="F")
    grid["amplitude"] = amplitude.ravel(order="F")
    grid["phase"] = phase.ravel(order="F")
    grid["probability"] = probability.ravel(order="F")

    if mode == "wavefunction":
        scalar_name = "phase"
    else:
        scalar_name = "amplitude"

    if representation == "volume":
        opacity_values = np.linspace(0.05, 0.65, 256)
        grid.set_active_scalars(scalar_name)
        return OrbitalField(
            dataset=grid,
            scalar_name=scalar_name,
            opacity=opacity_values,
            representation="volume",
            iso_fraction=iso_fraction,
            cumulative_probability=None,
        )

    iso_fraction_clamped = float(np.clip(iso_fraction, 0.01, 0.99))
    voxel_volume = float(np.prod(spacing))
    total_prob = float(probability.sum() * voxel_volume)
    if total_prob <= 0:
        iso_value = float(probability.max() * 0.2 + 1e-6)
    else:
        probs_flat = probability.ravel(order="F")
        masses = probs_flat * voxel_volume
        order = np.argsort(probs_flat)[::-1]
        sorted_probs = probs_flat[order]
        sorted_mass = masses[order]
        cumulative_mass = np.cumsum(sorted_mass)
        target_mass = iso_fraction_clamped * total_prob
        idx = int(np.searchsorted(cumulative_mass, target_mass, side="left"))
        idx = min(idx, len(sorted_probs) - 1)
        iso_value = float(sorted_probs[idx])

    if iso_value <= 0:
        iso_value = float(probability.max() * 0.2 + 1e-6)

    surface = grid.contour(isosurfaces=[iso_value], scalars="probability")
    surface = surface.sample(grid)
    if surface.n_points == 0:
        surface = pv.Sphere(radius=span * 0.25)
        surface[scalar_name] = np.zeros(surface.n_points)
    surface.set_active_scalars(scalar_name)

    mask_prob = probability >= iso_value
    cumulative = 0.0
    if total_prob > 0:
        cumulative = float(probability[mask_prob].sum() * voxel_volume / total_prob)
        cumulative = float(np.clip(cumulative, 0.0, 1.0))

    return OrbitalField(
        dataset=surface,
        scalar_name=scalar_name,
        opacity=1.0,
        representation="surface",
        iso_fraction=iso_fraction_clamped,
        cumulative_probability=cumulative,
    )
