from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_phase_names() -> dict[str, dict]:
    data = resources.files(__name__).joinpath("phase_names.json").read_text(encoding="utf-8")
    return json.loads(data)


@lru_cache(maxsize=1)
def load_practice_pool() -> list[dict]:
    data = resources.files(__name__).joinpath("practice_pool.json").read_text(encoding="utf-8")
    return json.loads(data)


@lru_cache(maxsize=1)
def load_tutorial_content() -> dict[str, dict]:
    data = resources.files(__name__).joinpath("tutorial_content.json").read_text(encoding="utf-8")
    return json.loads(data)
