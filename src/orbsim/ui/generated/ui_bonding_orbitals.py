from __future__ import annotations

from PySide6 import QtWidgets
from orbsim.widgets import DropPlotter


class Ui_BondingOrbitalsTab:
    def setupUi(self, widget: QtWidgets.QWidget) -> None:
        widget.setObjectName("BondingOrbitalsTab")
        self.rootLayout = QtWidgets.QHBoxLayout(widget)
        self.rootLayout.setObjectName("rootLayout")
        self.plotterFrame = DropPlotter(widget)
        self.plotterFrame.setObjectName("plotterFrame")
        self.rootLayout.addWidget(self.plotterFrame)
        self.sliceContainer = QtWidgets.QWidget(widget)
        self.sliceContainer.setObjectName("sliceContainer")
        self.rootLayout.addWidget(self.sliceContainer)
        self.controlsScroll = QtWidgets.QScrollArea(widget)
        self.controlsScroll.setObjectName("controlsScroll")
        self.controlsScroll.setWidgetResizable(True)
        self.controlsScrollContents = QtWidgets.QWidget()
        self.controlsScrollContents.setObjectName("controlsScrollContents")
        self.controlsScroll.setWidget(self.controlsScrollContents)
        self.rootLayout.addWidget(self.controlsScroll)
