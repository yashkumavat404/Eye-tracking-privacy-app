from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from calibration import CalibrationSession
from face_tracker import FaceTracker
from gaze_estimator import GazeEstimator
from settings import SettingsManager


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
WM_HOTKEY = 0x0312
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
GWL_EXSTYLE = -20


def _user32():
    user32 = ctypes.windll.user32
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    return user32


class SpotlightRenderer:
    """Low-latency screen capture and spotlight composition."""

    def __init__(self) -> None:
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.screen_size = (self.monitor["width"], self.monitor["height"])
        self.acceleration_label = self._detect_acceleration_label()

        pixels = self.screen_size[0] * self.screen_size[1]
        self.render_scale = 0.34 if pixels >= 2560 * 1440 else 0.40 if pixels >= 1920 * 1080 else 0.52

        self._mask_small: Optional[np.ndarray] = None
        self._mask_key: Optional[tuple[int, int, int, int, int, int]] = None
        self._last_shot = None
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_background: Optional[np.ndarray] = None
        self._cached_background_key: Optional[tuple[int, int, int]] = None

    def capture_screen(self) -> np.ndarray:
        self._last_shot = self.sct.grab(self.monitor)
        # Keep the screenshot object alive because the numpy array is a view.
        return np.asarray(self._last_shot, dtype=np.uint8)[:, :, :3]

    def prepare_background(
        self,
        frame: np.ndarray,
        brightness_reduction: int,
        softness: int,
    ) -> None:
        """Capture/blur the desktop once; spotlight movement can then update independently."""
        height, width = frame.shape[:2]
        scale = self.render_scale
        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))

        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        blur_size = 9 + int(np.clip(softness, 0, 100) * 0.18)
        blur_size |= 1
        blurred_small = cv2.GaussianBlur(
            small,
            (blur_size, blur_size),
            0,
            borderType=cv2.BORDER_REPLICATE,
        )
        blurred_native = cv2.resize(
            blurred_small,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        reduction = np.clip(brightness_reduction / 100.0, 0.0, 1.0)
        background_factor = 1.0 - (0.72 * reduction)

        self._cached_frame = frame
        self._cached_background = blurred_native.astype(np.float32) * background_factor
        self._cached_background_key = (int(brightness_reduction), int(softness), int(width))

    def compose_spotlight(
        self,
        gaze_point: Tuple[int, int],
        radius: int,
        softness: int,
        opacity: int,
    ) -> Optional[np.ndarray]:
        """Compose the spotlight from the latest cached desktop frame."""
        frame = self._cached_frame
        background = self._cached_background
        if frame is None or background is None:
            return None

        height, width = frame.shape[:2]
        scale = self.render_scale
        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))

        point = (
            int(np.clip(gaze_point[0] * scale, 0, small_w - 1)),
            int(np.clip(gaze_point[1] * scale, 0, small_h - 1)),
        )
        scaled_radius = max(8, int(radius * scale))
        key = (small_w, small_h, point[0], point[1], scaled_radius, int(softness))

        if key != self._mask_key or self._mask_small is None:
            mask = np.zeros((small_h, small_w), dtype=np.uint8)
            cv2.circle(mask, point, scaled_radius, 255, -1, lineType=cv2.LINE_AA)
            feather = max(7, int(scaled_radius * (0.10 + np.clip(softness, 0, 100) / 300)))
            feather |= 1
            self._mask_small = (
                cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
            )
            self._mask_key = key

        alpha = cv2.resize(
            self._mask_small,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )[..., None]
        alpha *= np.clip(opacity / 100.0, 0.1, 1.0)

        composed = (
            frame.astype(np.float32) * alpha
            + background * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)
        return composed

    def render_spotlight(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
        brightness_reduction: int,
        softness: int,
        opacity: int,
    ) -> np.ndarray:
        self.prepare_background(frame, brightness_reduction, softness)
        composed = self.compose_spotlight(gaze_point, radius, softness, opacity)
        return frame if composed is None else composed

    @staticmethod
    def _detect_acceleration_label() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
        except (ImportError, OSError):
            pass
        return "CPU"


class PrivacyOverlay(QtWidgets.QWidget):
    calibration_click_requested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Privacy Spotlight Overlay")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._image = QtWidgets.QLabel(self)
        self._image.setScaledContents(True)
        # The image covers the whole overlay. Keep it out of the mouse-event
        # chain so calibration clicks reach PrivacyOverlay.mousePressEvent().
        self._image.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._status = QtWidgets.QLabel(self)
        self._status.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._status.setStyleSheet("color: white; background: rgba(0,0,0,132); border-radius: 10px; padding: 10px;")
        self._status.hide()
        self._target_button = QtWidgets.QPushButton(self)
        self._target_button.setFlat(True)
        self._target_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._target_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._target_button.setStyleSheet(
            "QPushButton { background: transparent; border: none; padding: 0; }"
        )
        self._target_button.clicked.connect(lambda _checked=False: self.calibration_click_requested.emit())
        self._target_button.hide()
        self._target: Optional[Tuple[int, int]] = None
        self._guide_points: list[Tuple[int, int]] = []
        self._excluded = False
        self._hidden_for_capture = False
        self._calibrating = False
        self._fit_to_virtual_desktop()

    def _fit_to_virtual_desktop(self) -> None:
        geometry = QtCore.QRect()
        for screen in QtWidgets.QApplication.screens():
            geometry = geometry.united(screen.geometry()) if geometry.isValid() else screen.geometry()
        self.setGeometry(geometry)
        self._image.setGeometry(self.rect())
        self._status.setGeometry(24, 24, min(1060, self.width() - 48), 74)

    def set_status_text(self, text: str) -> None:
        self._status.setText(text)
        self._status.setVisible(bool(text))

    def show_privacy(self) -> None:
        self._fit_to_virtual_desktop()
        self.showFullScreen()
        self.raise_()
        self._exclude_from_capture()
        self._set_click_through(True)

    def hide_privacy(self) -> None:
        self.hide()

    def is_capture_safe_to_hide(self) -> bool:
        return self.isVisible() and not self._calibrating and not self._excluded

    def prepare_for_capture(self) -> None:
        if self._calibrating or self._excluded:
            return
        if self.isVisible():
            self._hidden_for_capture = True
            self.hide()

    def restore_after_capture(self) -> None:
        if self._hidden_for_capture:
            self.show_privacy()
            self._hidden_for_capture = False

    def update_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QtGui.QImage(rgb.data, width, height, channels * width, QtGui.QImage.Format.Format_RGB888)
        self._image.setPixmap(QtGui.QPixmap.fromImage(image))
        self._status.raise_()

    def _position_target_button(self) -> None:
        if self._target is None:
            self._target_button.hide()
            return
        size = 112
        x = max(0, min(int(self._target[0] - size / 2), max(0, self.width() - size)))
        y = max(0, min(int(self._target[1] - size / 2), max(0, self.height() - size)))
        self._target_button.setGeometry(x, y, size, size)
        self._target_button.show()
        self._target_button.raise_()

    def begin_calibration(self, target: Tuple[int, int], guide_points: list[Tuple[int, int]], text: str) -> None:
        self._calibrating = True
        self._target = target
        self._guide_points = guide_points
        self.set_status_text(text)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._image.clear()
        self.show()
        self.raise_()
        self._set_click_through(False)
        self._position_target_button()
        self.update()

    def update_calibration(self, target: Tuple[int, int], guide_points: list[Tuple[int, int]], text: str) -> None:
        self._target = target
        self._guide_points = guide_points
        self.set_status_text(text)
        self._position_target_button()
        self.update()

    def end_calibration(self) -> None:
        self._calibrating = False
        self._target = None
        self._guide_points = []
        self._target_button.hide()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._set_click_through(True)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._calibrating and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.calibration_click_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._calibrating:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(2, 5, 9, 138))
        for point in self._guide_points:
            self._draw_target(painter, point, point == self._target)

    def _draw_target(self, painter: QtGui.QPainter, point: Tuple[int, int], active: bool) -> None:
        center = QtCore.QPoint(*point)
        radius = 24 if active else 12
        glow = QtGui.QColor(41, 196, 255, 92 if active else 24)
        fill = QtGui.QColor(41, 196, 255, 236 if active else 88)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, radius + 22, radius + 22)
        painter.setBrush(fill)
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 3 if active else 1))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius + 9, radius + 9)
        if active:
            painter.drawLine(center.x() - 44, center.y(), center.x() + 44, center.y())
            painter.drawLine(center.x(), center.y() - 44, center.x(), center.y() + 44)

    def _exclude_from_capture(self) -> None:
        if self._excluded:
            return
        try:
            self._excluded = bool(_user32().SetWindowDisplayAffinity(wintypes.HWND(int(self.winId())), WDA_EXCLUDEFROMCAPTURE))
        except AttributeError:
            self._excluded = False

    def _set_click_through(self, enabled: bool) -> None:
        try:
            hwnd = wintypes.HWND(int(self.winId()))
            user32 = _user32()
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED
            style = style | WS_EX_TRANSPARENT if enabled else style & ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except AttributeError:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._excluded:
            _user32().SetWindowDisplayAffinity(wintypes.HWND(int(self.winId())), WDA_NONE)
        super().closeEvent(event)


class HotkeyListener(QtWidgets.QWidget):
    toggle_privacy = QtCore.pyqtSignal()
    start_calibration = QtCore.pyqtSignal()
    radius_increase = QtCore.pyqtSignal()
    radius_decrease = QtCore.pyqtSignal()
    quit_requested = QtCore.pyqtSignal()

    DEFAULT_HOTKEYS = {
        "toggle_privacy": "Ctrl+Alt+P",
        "radius_increase": "Ctrl+Alt+=",
        "radius_decrease": "Ctrl+Alt+-",
        "quit": "Ctrl+Alt+Q",
        "calibration": "Ctrl+Alt+C",
    }

    HOTKEY_IDS = {
        "toggle_privacy": 1,
        "radius_increase": 2,
        "radius_decrease": 3,
        "quit": 4,
        "calibration": 5,
    }

    def __init__(self, hotkeys: dict[str, str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Privacy Spotlight Hotkeys")
        self._registered_ids: set[int] = set()
        self._hotkeys = {**self.DEFAULT_HOTKEYS, **(hotkeys or {})}
        self._register_hotkeys()

    def update_hotkeys(self, hotkeys: dict[str, str]) -> None:
        self._unregister_hotkeys()
        self._hotkeys = {**self.DEFAULT_HOTKEYS, **hotkeys}
        self._register_hotkeys()

    def nativeEvent(self, event_type, message):
        try:
            msg = wintypes.MSG.from_address(int(message))
        except (TypeError, ValueError):
            return False, 0
        if msg.message != WM_HOTKEY:
            return False, 0
        hotkey_id = int(msg.wParam)
        signals = {
            1: self.toggle_privacy,
            2: self.radius_increase,
            3: self.radius_decrease,
            4: self.quit_requested,
            5: self.start_calibration,
        }
        if hotkey_id in signals:
            signals[hotkey_id].emit()
        return True, 0

    def _register_hotkeys(self) -> None:
        try:
            hwnd = wintypes.HWND(int(self.winId()))
            user32 = _user32()
            for action, hotkey in self._hotkeys.items():
                parsed = self._parse_hotkey(hotkey)
                hotkey_id = self.HOTKEY_IDS.get(action)
                if parsed is None or hotkey_id is None:
                    continue
                modifiers, key_code = parsed
                if user32.RegisterHotKey(hwnd, hotkey_id, modifiers, key_code):
                    self._registered_ids.add(hotkey_id)
        except AttributeError:
            pass

    def _unregister_hotkeys(self) -> None:
        try:
            hwnd = wintypes.HWND(int(self.winId()))
            user32 = _user32()
            for hotkey_id in tuple(self._registered_ids):
                user32.UnregisterHotKey(hwnd, hotkey_id)
            self._registered_ids.clear()
        except AttributeError:
            pass

    @staticmethod
    def _parse_hotkey(text: str) -> Optional[tuple[int, int]]:
        parts = [part.strip() for part in text.split("+") if part.strip()]
        if not parts:
            return None
        modifiers = 0
        key_token = parts[-1].upper()
        for part in parts[:-1]:
            token = part.upper()
            if token == "CTRL" or token == "CONTROL":
                modifiers |= MOD_CONTROL
            elif token == "ALT":
                modifiers |= MOD_ALT
        key_map = {
            "=": 0xBB,
            "+": 0xBB,
            "-": 0xBD,
            "MINUS": 0xBD,
            "PLUS": 0xBB,
        }
        if key_token in key_map:
            key_code = key_map[key_token]
        elif len(key_token) == 1 and key_token.isalnum():
            key_code = ord(key_token)
        else:
            return None
        return modifiers, key_code

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._unregister_hotkeys()
        super().closeEvent(event)


class PrivacyController(QtCore.QObject):
    status_changed = QtCore.pyqtSignal(dict)
    calibration_changed = QtCore.pyqtSignal(dict)
    notification_requested = QtCore.pyqtSignal(str, str)

    def __init__(self, app: QtWidgets.QApplication, settings: SettingsManager) -> None:
        super().__init__()
        self.app = app
        self.settings = settings
        self.renderer = SpotlightRenderer()
        self.overlay = PrivacyOverlay()
        self.face_tracker = FaceTracker()
        self.gaze_estimator = GazeEstimator(self.renderer.screen_size)
        self.gaze_estimator.load_calibration()
        self.calibration = CalibrationSession(self.renderer.screen_size)
        self.hotkeys = HotkeyListener(settings.get("hotkeys", {}))
        self.hotkeys.hide()
        self.privacy_enabled = bool(settings.get("privacy_on_startup", settings.get("privacy_enabled")))
        self.eye_tracking_enabled = bool(settings.get("eye_tracking_enabled"))
        self.head_pose_enabled = bool(settings.get("head_pose_enabled"))
        self.radius = int(settings.get("radius"))
        self.last_point = self._screen_center()
        self.fps = 0.0
        self._frame_count = 0
        self._fps_started = time.perf_counter()
        self._frame_busy = False
        self._shutting_down = False
        self._last_observation_timestamp = -1.0
        self._last_gaze_point = self.last_point
        self._capture_failures = 0
        self._calibration_collecting = False
        self._calibration_samples: list[tuple[Tuple[float, float], Tuple[float, ...]]] = []
        self._calibration_deadline = 0.0
        self._calibration_sample_started = 0.0
        self._calibration_click_count = 0
        self._calibration_last_timestamp = -1.0

        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        # Spotlight position updates at ~60 FPS so gaze movement is not gated by
        # the more expensive desktop capture/blur operation.
        self.frame_timer.setInterval(16)
        self.frame_timer.timeout.connect(self._update_overlay_frame)

        self.capture_timer = QtCore.QTimer(self)
        self.capture_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        # Refresh the desktop background at ~30 FPS; gaze composition runs independently.
        self.capture_timer.setInterval(33)
        self.capture_timer.timeout.connect(self._capture_background)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self.emit_status)
        self.status_timer.start()
        self.calibration_timer = QtCore.QTimer(self)
        self.calibration_timer.setInterval(10)
        self.calibration_timer.timeout.connect(self._collect_calibration_frames)

        self.overlay.calibration_click_requested.connect(self.capture_calibration_point)
        self.hotkeys.toggle_privacy.connect(self.toggle_privacy_mode)
        self.hotkeys.start_calibration.connect(self.start_calibration)
        self.hotkeys.radius_increase.connect(lambda: self.set_radius(self.radius + 24))
        self.hotkeys.radius_decrease.connect(lambda: self.set_radius(self.radius - 24))
        self.hotkeys.quit_requested.connect(self.shutdown)

        if self.privacy_enabled:
            self.enable_privacy_mode(True)
        elif self.eye_tracking_enabled:
            # Keep the tracker available for diagnostics and calibration even
            # when privacy rendering is currently disabled.
            self.face_tracker.start()

    def _screen_center(self) -> Tuple[int, int]:
        return self.renderer.screen_size[0] // 2, self.renderer.screen_size[1] // 2

    def enable_privacy_mode(self, enabled: bool) -> None:
        if self.calibration.active:
            return
        self.privacy_enabled = enabled
        self.settings.set("privacy_enabled", enabled)
        if enabled:
            if self.eye_tracking_enabled:
                self.face_tracker.start()
            self.gaze_estimator.reset()
            self.overlay.show_privacy()
            self._capture_background()
            self.frame_timer.start()
            self.capture_timer.start()
            self.notification_requested.emit("Privacy mode enabled", "Spotlight protection is running.")
        else:
            self.frame_timer.stop()
            self.capture_timer.stop()
            if not self.eye_tracking_enabled:
                self.face_tracker.stop()
            self.overlay.hide_privacy()
            self.notification_requested.emit("Privacy mode disabled", "Screen dimming is off.")
        self.emit_status()

    def toggle_privacy_mode(self) -> None:
        self.enable_privacy_mode(not self.privacy_enabled)

    def set_eye_tracking_enabled(self, enabled: bool) -> None:
        self.eye_tracking_enabled = enabled
        self.settings.set("eye_tracking_enabled", enabled)
        if enabled:
            # Tracking is independent from privacy rendering so the diagnostics
            # and calibration can verify the camera/gaze pipeline on their own.
            self.face_tracker.start()
        elif not self.calibration.active:
            self.face_tracker.stop()
        self.emit_status()

    def set_head_pose_enabled(self, enabled: bool) -> None:
        # Hook point: forward this to a production tracker if head-pose can be independently disabled.
        self.head_pose_enabled = enabled
        self.settings.set("head_pose_enabled", enabled)
        self.emit_status()

    def set_radius(self, radius: int) -> None:
        self.radius = max(120, min(560, int(radius)))
        self.settings.set("radius", self.radius)
        self.emit_status()

    def update_setting(self, key: str, value) -> None:
        self.settings.set(key, value)
        self.emit_status()

    def start_calibration(self) -> None:
        if self.calibration.active:
            return
        self.calibration.start()
        self.gaze_estimator.clear_calibration()
        self.face_tracker.start()
        self.frame_timer.stop()
        self.calibration_timer.start()
        self._calibration_collecting = False
        self._calibration_samples.clear()
        self._show_calibration_target()
        self.notification_requested.emit("Calibration started", "Look at each target and click to sample.")

    def stop_calibration(self) -> None:
        if not self.calibration.active:
            return
        self.calibration.stop()
        self._calibration_collecting = False
        self.calibration_timer.stop()
        self.overlay.end_calibration()
        if self.privacy_enabled:
            self._capture_background()
            self.frame_timer.start()
            self.capture_timer.start()
        else:
            if not self.eye_tracking_enabled:
                self.face_tracker.stop()
            self.overlay.hide()

    def reset_calibration(self) -> None:
        self.gaze_estimator.clear_calibration()
        self.calibration_changed.emit({"state": "Waiting", "progress": 0, "accuracy": 0})
        self.notification_requested.emit("Calibration reset", "Stored gaze samples were cleared.")

    def _show_calibration_target(self) -> None:
        current = self.calibration.current_target()
        if current is None:
            return
        label, position, message = current
        guide_points = [target[1] for target in self.calibration.targets]
        progress = int((self.calibration.index / len(self.calibration.targets)) * 100)
        text = f"{self.calibration.progress_text()}\n{message}\nLook at the dot for a moment, then click it."
        if self.calibration.index == 0:
            self.overlay.begin_calibration(position, guide_points, text)
        else:
            self.overlay.update_calibration(position, guide_points, text)
        self.calibration_changed.emit({"state": "Running", "progress": progress, "accuracy": self._accuracy_percent()})

    def _collect_calibration_frames(self) -> None:
        if not self.calibration.active or not self._calibration_collecting:
            return
        observation = self.face_tracker.get_latest_observation()
        if (
            observation
            and observation.gaze_vector
            and observation.gaze_features
            and not observation.blink
            and observation.confidence >= 0.62
            and observation.timestamp != self._calibration_last_timestamp
        ):
            self._calibration_samples.append((observation.gaze_vector, observation.gaze_features))
            self._calibration_last_timestamp = observation.timestamp

        if time.perf_counter() < self._calibration_deadline:
            return

        self._calibration_collecting = False
        if len(self._calibration_samples) < 3:
            observation = self.face_tracker.get_latest_observation()
            if observation and observation.gaze_vector and observation.gaze_features:
                self._calibration_samples = [(observation.gaze_vector, observation.gaze_features)]
            else:
                self.notification_requested.emit(
                    "Waiting for eye detection",
                    "Keep your face centered and click the same target again.",
                )
                self.calibration_changed.emit({
                    "state": "Ready — click target again",
                    "progress": int((self.calibration.index / len(self.calibration.targets)) * 100),
                    "accuracy": self._accuracy_percent(),
                })
                return

        vectors = np.asarray([item[0] for item in self._calibration_samples], dtype=np.float32)
        features = np.asarray([item[1] for item in self._calibration_samples], dtype=np.float32)
        center = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - center, axis=1)
        cutoff = max(
            float(np.percentile(distances, 82)),
            float(np.median(distances) + 2.5 * np.std(distances)),
        )
        keep = distances <= cutoff
        if int(keep.sum()) < 6:
            keep = np.ones(len(vectors), dtype=bool)

        gaze_vector = np.median(vectors[keep], axis=0)
        gaze_features = np.median(features[keep], axis=0)
        current = self.calibration.current_target()
        if current is None:
            return

        label, position, _ = current
        self.gaze_estimator.add_calibration_sample(
            label,
            position,
            (float(gaze_vector[0]), float(gaze_vector[1])),
            tuple(float(value) for value in gaze_features),
        )
        if not self.calibration.advance():
            self.finish_calibration()
            return
        self._show_calibration_target()

    def capture_calibration_point(self) -> None:
        current = self.calibration.current_target()
        if current is None or self._calibration_collecting:
            return

        self._calibration_collecting = True
        self._calibration_samples.clear()
        self._calibration_last_timestamp = -1.0
        self._calibration_sample_started = time.perf_counter()
        self._calibration_deadline = self._calibration_sample_started + 0.28
        self._calibration_click_count += 1
        self.calibration_changed.emit({
            "state": f"Sampling point {self.calibration.index + 1}/25...",
            "progress": int((self.calibration.index / len(self.calibration.targets)) * 100),
            "accuracy": self._accuracy_percent(),
        })

    def finish_calibration(self) -> None:
        self.calibration.stop()
        self._calibration_collecting = False
        self.calibration_timer.stop()
        self.overlay.end_calibration()
        if self.privacy_enabled:
            self._capture_background()
            self.frame_timer.start()
            self.capture_timer.start()
        else:
            if not self.eye_tracking_enabled:
                self.face_tracker.stop()
            self.overlay.hide()
        self.calibration_changed.emit({"state": "Completed", "progress": 100, "accuracy": self._accuracy_percent()})
        self.notification_requested.emit("Calibration completed", "Gaze mapping has been saved.")

    def _accuracy_percent(self) -> int:
        observation = self.face_tracker.get_latest_observation()
        confidence = observation.confidence if observation else 0.0
        return int(max(0.0, min(1.0, confidence)) * 100)

    def _capture_background(self) -> None:
        if not self.privacy_enabled or self.calibration.active:
            return

        overlay_hidden = False
        try:
            if self.overlay.is_capture_safe_to_hide():
                self.overlay.prepare_for_capture()
                overlay_hidden = True

            frame = self.renderer.capture_screen()
            self.renderer.prepare_background(
                frame,
                int(self.settings.get("brightness_reduction")),
                int(self.settings.get("spotlight_softness")),
            )
        finally:
            if overlay_hidden:
                self.overlay.restore_after_capture()

    def _update_overlay_frame(self) -> None:
        if not self.privacy_enabled or self._frame_busy or self.calibration.active:
            return

        self._frame_busy = True
        try:
            # Consume the newest camera result without waiting for inference.
            observation = self.face_tracker.get_latest_observation() if self.eye_tracking_enabled else None
            if observation is not None and observation.timestamp != self._last_observation_timestamp:
                self._last_observation_timestamp = observation.timestamp
                estimate = self.gaze_estimator.estimate(observation)
                if estimate is not None:
                    self.last_point = estimate.screen_point
                    self._last_gaze_point = estimate.screen_point

            processed = self.renderer.compose_spotlight(
                self._last_gaze_point,
                self.radius,
                int(self.settings.get("spotlight_softness")),
                int(self.settings.get("spotlight_opacity")),
            )
            if processed is None:
                return

            self.overlay.update_frame(processed)
            self._frame_count += 1
            elapsed = time.perf_counter() - self._fps_started
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_started = time.perf_counter()
        finally:
            self._frame_busy = False

    def emit_status(self) -> None:
        diagnostics = self.face_tracker.get_diagnostics()
        confidence = int(diagnostics["confidence"] * 100)
        face_detected = bool(diagnostics["face_detected"])
        gaze = diagnostics.get("gaze_vector")
        gaze_text = "—" if gaze is None else f"{gaze[0]:.3f}, {gaze[1]:.3f}"
        self.overlay.set_status_text(
            f"Privacy: {'ON' if self.privacy_enabled else 'OFF'} | "
            f"Face: {'YES' if face_detected else 'NO'} | "
            f"Iris: {'YES' if diagnostics['iris_detected'] else 'NO'} | "
            f"Gaze: {gaze_text} | Confidence: {confidence}% | "
            f"Track FPS: {diagnostics['processing_fps']:.1f}"
        )
        self.status_changed.emit(
            {
                "privacy_enabled": self.privacy_enabled,
                "eye_tracking_enabled": self.eye_tracking_enabled,
                "head_pose_enabled": self.head_pose_enabled,
                "face_detected": face_detected,
                "iris_detected": diagnostics["iris_detected"],
                "camera_open": diagnostics["camera_open"],
                "confidence": confidence,
                "gaze_vector": gaze,
                "left_pupil": diagnostics["left_pupil"],
                "right_pupil": diagnostics["right_pupil"],
                "yaw": diagnostics["yaw"],
                "pitch": diagnostics["pitch"],
                "roll": diagnostics["roll"],
                "ear": diagnostics["ear"],
                "blink": diagnostics["blink"],
                "tracking_fps": diagnostics["processing_fps"],
                "gaze_motion": diagnostics["gaze_motion"],
                "neutral_ready": diagnostics.get("neutral_ready", False),
                "radius": self.radius,
                "fps": self.fps,
                "acceleration": self.renderer.acceleration_label,
                "screen_size": self.renderer.screen_size,
            }
        )

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.frame_timer.stop()
        self.capture_timer.stop()
        self.calibration_timer.stop()
        self.status_timer.stop()
        self.face_tracker.stop()
        self.overlay.close()
        self.hotkeys.close()
        self.app.quit()
