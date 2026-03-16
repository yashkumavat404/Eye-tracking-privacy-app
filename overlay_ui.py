import ctypes
from ctypes import wintypes
from typing import Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
WM_HOTKEY = 0x0312
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class PrivacyOverlay(QtWidgets.QWidget):
    toggle_requested = QtCore.pyqtSignal()
    radius_increase_requested = QtCore.pyqtSignal()
    radius_decrease_requested = QtCore.pyqtSignal()
    calibration_requested = QtCore.pyqtSignal()
    calibration_click_requested = QtCore.pyqtSignal()
    exit_requested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVIDIA Privacy Spotlight")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)

        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(self.rect())
        self.image_label.setScaledContents(True)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setGeometry(20, 20, min(980, self.width() - 40), 78)
        self.status_label.setStyleSheet(
            "color: white; background-color: rgba(0, 0, 0, 140);"
            "padding: 10px; border-radius: 8px; font-size: 13px;"
        )
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.calibration_label = QtWidgets.QLabel(self)
        self.calibration_label.setGeometry(20, self.height() - 124, min(980, self.width() - 40), 100)
        self.calibration_label.setStyleSheet(
            "color: white; background-color: rgba(12, 12, 12, 170);"
            "padding: 12px; border-radius: 10px; font-size: 15px;"
        )
        self.calibration_label.hide()

        self._hidden_for_capture = False
        self._excluded_from_capture = False
        self._capture_exclusion_attempted = False
        self._calibration_mode = False
        self._calibration_target: Optional[Tuple[int, int]] = None
        self._dot_radius = 20
        self._guide_points: list[Tuple[int, int]] = []
        self._register_hotkeys()

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def show_overlay(self) -> None:
        self.showFullScreen()
        self.raise_()
        self._ensure_excluded_from_capture()

    def hide_overlay(self) -> None:
        if self._calibration_mode:
            self.end_calibration_mode()
        self.hide()

    def begin_calibration_mode(self, position: Tuple[int, int], message: str, guide_points: list[Tuple[int, int]]) -> None:
        self._calibration_mode = True
        self._calibration_target = position
        self._guide_points = guide_points
        self.calibration_label.setText(message)
        self.calibration_label.show()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.show_overlay()
        self.update()

    def update_calibration_prompt(self, position: Tuple[int, int], message: str, guide_points: list[Tuple[int, int]]) -> None:
        self._calibration_target = position
        self._guide_points = guide_points
        self.calibration_label.setText(message)
        self.calibration_label.show()
        self.update()

    def end_calibration_mode(self) -> None:
        self._calibration_mode = False
        self._calibration_target = None
        self._guide_points = []
        self.calibration_label.hide()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.update()

    def prepare_for_capture(self) -> None:
        if self._calibration_mode or self._excluded_from_capture:
            return

        if self.isVisible():
            self._hidden_for_capture = True
            self.hide()

    def restore_after_capture(self) -> None:
        if self._calibration_mode or self._excluded_from_capture:
            return

        if self._hidden_for_capture:
            self.show_overlay()
            self._hidden_for_capture = False

    def update_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel_count = rgb_frame.shape
        bytes_per_line = channel_count * width
        image = QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)
        self.status_label.raise_()
        if self._calibration_mode:
            self.calibration_label.raise_()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._calibration_mode and event.button() == QtCore.Qt.LeftButton:
            self.calibration_click_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._calibration_mode or self._calibration_target is None:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        for point in self._guide_points:
            if point == self._calibration_target:
                continue
            self._draw_dot(painter, point, active=False)

        self._draw_dot(painter, self._calibration_target, active=True)

    def _draw_dot(self, painter: QtGui.QPainter, point: Tuple[int, int], active: bool) -> None:
        center = QtCore.QPoint(*point)
        if active:
            ring_pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 4)
            fill_brush = QtGui.QBrush(QtGui.QColor(65, 221, 255, 230))
            glow_brush = QtGui.QBrush(QtGui.QColor(65, 221, 255, 90))
            radius = self._dot_radius
        else:
            ring_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2)
            fill_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 80))
            glow_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 28))
            radius = max(10, self._dot_radius - 6)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(glow_brush)
        painter.drawEllipse(center, radius + 16, radius + 16)
        painter.setBrush(fill_brush)
        painter.drawEllipse(center, radius, radius)
        painter.setPen(ring_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius + 8, radius + 8)
        if active:
            painter.drawLine(center.x() - 36, center.y(), center.x() + 36, center.y())
            painter.drawLine(center.x(), center.y() - 36, center.x(), center.y() + 36)

    def nativeEvent(self, event_type, message):
        msg = wintypes.MSG.from_address(message.__int__())
        if msg.message == WM_HOTKEY:
            hotkey_id = int(msg.wParam)
            if hotkey_id == 1:
                self.toggle_requested.emit()
            elif hotkey_id == 2:
                self.radius_increase_requested.emit()
            elif hotkey_id == 3:
                self.radius_decrease_requested.emit()
            elif hotkey_id == 4:
                self.exit_requested.emit()
            elif hotkey_id == 5:
                self.calibration_requested.emit()
            return True, 0
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._excluded_from_capture:
            ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), WDA_NONE)
        self._unregister_hotkeys()
        super().closeEvent(event)

    def _ensure_excluded_from_capture(self) -> None:
        if self._capture_exclusion_attempted:
            return

        self._capture_exclusion_attempted = True
        user32 = ctypes.windll.user32
        hwnd = int(self.winId())
        result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        self._excluded_from_capture = bool(result)

    def _register_hotkeys(self) -> None:
        self.winId()
        user32 = ctypes.windll.user32
        hwnd = int(self.winId())
        user32.RegisterHotKey(hwnd, 1, MOD_CONTROL | MOD_ALT, ord("P"))
        user32.RegisterHotKey(hwnd, 2, MOD_CONTROL | MOD_ALT, 0xBB)
        user32.RegisterHotKey(hwnd, 3, MOD_CONTROL | MOD_ALT, 0xBD)
        user32.RegisterHotKey(hwnd, 4, MOD_CONTROL | MOD_ALT, ord("Q"))
        user32.RegisterHotKey(hwnd, 5, MOD_CONTROL | MOD_ALT, ord("C"))

    def _unregister_hotkeys(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = int(self.winId())
        for hotkey_id in (1, 2, 3, 4, 5):
            user32.UnregisterHotKey(hwnd, hotkey_id)
