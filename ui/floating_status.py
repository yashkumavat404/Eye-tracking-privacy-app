from __future__ import annotations

from PyQt6 import QtCore, QtWidgets


class FloatingStatusWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.label = QtWidgets.QLabel("Privacy Spotlight: OFF", self)
        self.label.setStyleSheet(
            "background: rgba(13, 18, 24, 210); color: white; border: 1px solid #243545; "
            "border-radius: 8px; padding: 9px 12px; font-weight: 650;"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.resize(230, 44)

    def update_status(self, data: dict) -> None:
        mode = "ON" if data.get("privacy_enabled") else "OFF"
        self.label.setText(f"Privacy Spotlight: {mode} | {data.get('confidence', 0)}%")

