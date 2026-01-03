from __future__ import annotations

from PySide6 import QtWidgets


class Ui_BondingOrbitalsTab:
    def setupUi(self, widget: QtWidgets.QWidget) -> None:
        widget.setObjectName("BondingOrbitalsTab")
        self.rootLayout = QtWidgets.QVBoxLayout(widget)
        self.rootLayout.setObjectName("rootLayout")
        self.contentWidget = QtWidgets.QWidget(widget)
        self.contentWidget.setObjectName("contentWidget")
        self.contentLayout = QtWidgets.QVBoxLayout(self.contentWidget)
        self.contentLayout.setObjectName("contentLayout")
        self.rootLayout.addWidget(self.contentWidget)
