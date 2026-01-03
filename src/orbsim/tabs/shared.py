from __future__ import annotations

import cmcrameri.cm as cmc
import matplotlib.cm as mpl_cm


def resolve_cmap(name: str):
    try:
        return mpl_cm.get_cmap(name)
    except Exception:
        pass
    try:
        return getattr(cmc, name)
    except Exception:
        pass
    try:
        return mpl_cm.get_cmap(f"cmc.{name}")
    except Exception:
        pass
    return mpl_cm.get_cmap("viridis")
