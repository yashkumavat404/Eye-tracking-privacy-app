from __future__ import annotations

from collections import deque
from typing import Callable

import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


class Card(QtWidgets.QFrame):
    def __init__(self, title: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 16)
        self.layout.setSpacing(10)
        if title:
            label = QtWidgets.QLabel(title)
            label.setObjectName("SectionTitle")
            self.layout.addWidget(label)


class ToggleSwitch(QtWidgets.QAbstractButton):
    toggledAnimated = QtCore.pyqtSignal(bool)

    def __init__(self, checked: bool = False) -> None:
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(58, 30)
        self._offset = 1.0 if checked else 0.0
        self._animation = QtCore.QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(160)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = QtCore.pyqtProperty(float, get_offset, set_offset)

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()
        self.toggledAnimated.emit(checked)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        bg = QtGui.QColor("#18a957") if self.isChecked() else QtGui.QColor("#b54848")
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), 15, 15)
        x = 4 + self._offset * 28
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(QtCore.QRectF(x, 4, 22, 22))


class MetricRow(QtWidgets.QWidget):
    def __init__(self, label: str, value: str = "--") -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        name = QtWidgets.QLabel(label)
        name.setObjectName("Muted")
        self.value = QtWidgets.QLabel(value)
        self.value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(name)
        layout.addWidget(self.value)


class LabeledSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int, suffix: str = "") -> None:
        super().__init__()
        self.suffix = suffix
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(label)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setObjectName("Muted")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.label)
        header.addWidget(self.value_label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._changed)
        root.addLayout(header)
        root.addWidget(self.slider)
        self._changed(value)

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def _changed(self, value: int) -> None:
        self.value_label.setText(f"{value}{self.suffix}")
        self.valueChanged.emit(value)


class StatusPill(QtWidgets.QLabel):
    def __init__(self, text: str = "OFF", active: bool = False) -> None:
        super().__init__(text)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(26)
        self.setMinimumWidth(82)
        self.set_state(active, text)

    def set_state(self, active: bool, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        color = "#1db954" if active else "#d34b4b"
        self.setStyleSheet(f"background: {color}; color: white; border-radius: 8px; font-weight: 700;")


class LiveGraph(Card):
    def __init__(self, title: str, color: str) -> None:
        super().__init__(title)
        pg.setConfigOptions(antialias=True)
        self.samples: deque[float] = deque([0.0] * 90, maxlen=90)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#121922")
        self.plot.showGrid(x=False, y=True, alpha=0.16)
        self.plot.setYRange(0, 100)
        self.plot.hideAxis("bottom")
        self.plot.getAxis("left").setTextPen("#8493a3")
        self.curve = self.plot.plot(list(self.samples), pen=pg.mkPen(color, width=2))
        self.layout.addWidget(self.plot)

    def add_sample(self, value: float) -> None:
        self.samples.append(max(0.0, min(100.0, float(value))))
        self.curve.setData(list(self.samples))


class PreviewWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.radius = 42
        self.opacity = 90
        self.softness = 55
        self.setMinimumHeight(180)

    def set_values(self, radius: int | None = None, opacity: int | None = None, softness: int | None = None) -> None:
        if radius is not None:
            self.radius = max(22, int(radius / 5))
        if opacity is not None:
            self.opacity = opacity
        if softness is not None:
            self.softness = softness
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, QtGui.QColor("#0d1218"))
        for index in range(8):
            x = rect.left() + 18 + index * max(24, rect.width() // 8)
            painter.fillRect(QtCore.QRect(x, rect.top() + 24, 12, rect.height() - 48), QtGui.QColor(35, 48, 60))
        center = rect.center()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, int(190 * self.opacity / 100)))
        painter.drawRoundedRect(rect, 8, 8)
        gradient = QtGui.QRadialGradient(QtCore.QPointF(center), float(self.radius + self.softness))
        gradient.setColorAt(0.0, QtGui.QColor(255, 255, 255, 0))
        gradient.setColorAt(0.72, QtGui.QColor(255, 255, 255, 0))
        gradient.setColorAt(1.0, QtGui.QColor(0, 0, 0, int(210 * self.opacity / 100)))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Clear)
        painter.setBrush(gradient)
        painter.drawEllipse(center, self.radius + self.softness, self.radius + self.softness)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e7f7ff"), 2))
        painter.drawEllipse(center, self.radius, self.radius)


def bind_switch(switch: ToggleSwitch, callback: Callable[[bool], None]) -> None:
    switch.toggledAnimated.connect(callback)

