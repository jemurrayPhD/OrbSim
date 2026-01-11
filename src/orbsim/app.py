from __future__ import annotations

import sys

from PySide6 import QtWidgets

from orbsim import resources  # noqa: F401
from orbsim.views.main_window import OrbSimMainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OrbSimMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
