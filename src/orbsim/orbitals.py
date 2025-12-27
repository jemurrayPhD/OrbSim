from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class OrbitalField:
    mesh: pv.PolyData
    scalars: np.ndarray
    opacity: np.ndarray | None


def _orbital_scalar(mesh: pv.PolyData) -> np.ndarray:
    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (radius + 1e-6), -1.0, 1.0))
    phi = np.arctan2(y, x)
    radial = np.exp(-radius)
    angular = np.cos(2 * theta) * np.sin(3 * phi)
    return radial * angular


def make_orbital_mesh(
    mode: str,
    resolution: int = 64,
    use_amplitude_opacity: bool = False,
) -> OrbitalField:
    sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution, radius=1.5)
    scalar = _orbital_scalar(sphere)

    if mode == "probability":
        values = np.abs(scalar) ** 2
    else:
        values = scalar

    opacity = None
    if use_amplitude_opacity:
        amplitude = np.abs(scalar)
        opacity = np.clip(amplitude / (amplitude.max() + 1e-6), 0.1, 1.0)

    return OrbitalField(mesh=sphere, scalars=values, opacity=opacity)
