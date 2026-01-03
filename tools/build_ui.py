from __future__ import annotations

import subprocess
from pathlib import Path


def build_ui() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = repo_root / "src" / "orbsim" / "ui"
    output_root = ui_root / "generated"
    output_root.mkdir(parents=True, exist_ok=True)

    ui_files = list(ui_root.glob("*.ui")) + list((ui_root / "tabs").glob("*.ui"))
    for ui_file in ui_files:
        name = ui_file.stem
        target = output_root / f"ui_{name}.py"
        subprocess.run(
            ["pyside6-uic", str(ui_file), "-o", str(target)],
            check=True,
        )


if __name__ == "__main__":
    build_ui()
