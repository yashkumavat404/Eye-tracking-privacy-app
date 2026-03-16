import ctypes
from ctypes import wintypes
import os
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import torch
    import torch.nn.functional as F
    TORCH_LOAD_ERROR = None
except (ImportError, OSError) as exc:
    torch = None
    F = None
    TORCH_LOAD_ERROR = exc


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
WM_HOTKEY = 0x0312
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
GWL_EXSTYLE = -20


class SpotlightRenderer:
    def __init__(self) -> None:
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.screen_size = (self.monitor["width"], self.monitor["height"])
        self.render_scale = self._choose_render_scale()
        self._kernel_cache = {}
        self._cuda_filter = None
        self._cuda_filter_shape = None
        self.preferred_cuda_index = self._preferred_cuda_index()
        self.torch_device = None
        self.use_torch_cuda = bool(torch is not None and torch.cuda.is_available())
        self.use_cv2_cuda = False

        if self.use_torch_cuda:
            self.preferred_cuda_index = min(self.preferred_cuda_index, torch.cuda.device_count() - 1)
            torch.cuda.set_device(self.preferred_cuda_index)
            self.torch_device = torch.device(f"cuda:{self.preferred_cuda_index}")
            self._warm_up_torch()

        if not self.use_torch_cuda:
            self.use_cv2_cuda = self._init_cv2_cuda()

        if self.use_torch_cuda:
            device_name = torch.cuda.get_device_name(self.preferred_cuda_index)
            self.acceleration_label = f"CUDA GPU{self.preferred_cuda_index + 1} ({device_name})"
        elif self.use_cv2_cuda:
            self.acceleration_label = f"OpenCV CUDA GPU{self.preferred_cuda_index + 1}"
        elif TORCH_LOAD_ERROR is not None:
            self.acceleration_label = "CPU fallback"
        else:
            self.acceleration_label = "CPU"

    def _preferred_cuda_index(self) -> int:
        requested_index = os.environ.get("PRIVACY_APP_GPU_INDEX")
        if requested_index is not None:
            try:
                return max(0, int(requested_index))
            except ValueError:
                pass

        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()
            for device_index in range(count):
                device_name = torch.cuda.get_device_name(device_index).lower()
                if "rtx 3050" in device_name:
                    return device_index
            if count > 1:
                return 1
        return 0

    def _warm_up_torch(self) -> None:
        if self.torch_device is None or torch is None or F is None:
            return
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device=self.torch_device)
            kernel = self._gaussian_kernel(size=15, sigma=4.2, device=self.torch_device, channels=3)
            padded = F.pad(dummy, (7, 7, 7, 7), mode="reflect")
            F.conv2d(padded, kernel, groups=3)
        torch.cuda.synchronize(self.torch_device)

    def _init_cv2_cuda(self) -> bool:
        try:
            if not hasattr(cv2, "cuda"):
                return False
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count <= 0:
                return False
            cv2.cuda.setDevice(min(self.preferred_cuda_index, count - 1))
            return True
        except cv2.error:
            return False

    def _choose_render_scale(self) -> float:
        width, height = self.screen_size
        pixel_count = width * height
        if pixel_count >= 2560 * 1440:
            return 0.42
        if pixel_count >= 1920 * 1080:
            return 0.46
        return 0.62

    def capture_screen(self) -> np.ndarray:
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def render_spotlight(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        mask = self._create_mask(frame.shape[:2], gaze_point, radius)
        background = self._build_background(frame, gaze_point, radius)
        clear_region = frame.astype(np.float32) * mask[..., None]
        background_region = background.astype(np.float32) * (1.0 - mask[..., None])
        return np.clip(clear_region + background_region, 0, 255).astype(np.uint8)

    def _build_background(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        if self.render_scale < 0.999:
            small_frame = self._resize_frame(frame, self.render_scale)
            scaled_point = (int(gaze_point[0] * self.render_scale), int(gaze_point[1] * self.render_scale))
            scaled_radius = max(20, int(radius * self.render_scale))
            processed_small = self._build_background_native(small_frame, scaled_point, scaled_radius)
            return cv2.resize(processed_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return self._build_background_native(frame, gaze_point, radius)

    def _build_background_native(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        blurred = self._blur_frame(frame)
        mask = self._create_mask(frame.shape[:2], gaze_point, radius)
        darkening = (1.0 - mask[..., None]) * 0.26
        return np.clip(blurred.astype(np.float32) * (1.0 - darkening), 0, 255).astype(np.uint8)

    def _resize_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        height, width = frame.shape[:2]
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _blur_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.use_torch_cuda:
            return self._torch_gaussian_blur(frame)
        if self.use_cv2_cuda:
            blurred = self._cv2_cuda_blur(frame)
            if blurred is not None:
                return blurred
            self.use_cv2_cuda = False
            self.acceleration_label = "CPU fallback"
        return cv2.GaussianBlur(frame, (17, 17), 0, borderType=cv2.BORDER_REPLICATE)

    def _cv2_cuda_blur(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            if self._cuda_filter is None or self._cuda_filter_shape != frame.shape[:2]:
                self._cuda_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (17, 17), 0)
                self._cuda_filter_shape = frame.shape[:2]
            return self._cuda_filter.apply(gpu_frame).download()
        except cv2.error:
            return None

    def _torch_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        assert torch is not None and F is not None
        assert self.torch_device is not None
        with torch.inference_mode():
            tensor = (
                torch.from_numpy(frame)
                .to(device=self.torch_device, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            kernel = self._gaussian_kernel(size=15, sigma=4.2, device=tensor.device, channels=3)
            padded = F.pad(tensor, (7, 7, 7, 7), mode="reflect")
            blurred = F.conv2d(padded, kernel, groups=3)
            output = blurred.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return output

    def _gaussian_kernel(self, size: int, sigma: float, device, channels: int):
        cache_key = (size, sigma, str(device), channels)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel_2d = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2 * sigma * sigma))
        kernel_2d /= kernel_2d.sum()
        kernel = kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)
        self._kernel_cache[cache_key] = kernel
        return kernel

    @staticmethod
    def _create_mask(image_shape: Tuple[int, int], gaze_point: Tuple[int, int], radius: int) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(mask, gaze_point, radius, 1.0, -1, lineType=cv2.LINE_AA)
        feather = max(radius // 3, 16)
        blur_amount = feather * 2 + 1
        return cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)


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
            | QtCore.Qt.WindowDoesNotAcceptFocus
        )
        self.setFocusPolicy(QtCore.Qt.NoFocus)
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
            "color: white; background-color: rgba(0, 0, 0, 140);padding: 10px; border-radius: 8px; font-size: 13px;"
        )
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.calibration_label = QtWidgets.QLabel(self)
        self.calibration_label.setGeometry(20, self.height() - 124, min(980, self.width() - 40), 100)
        self.calibration_label.setStyleSheet(
            "color: white; background-color: rgba(12, 12, 12, 170);padding: 12px; border-radius: 10px; font-size: 15px;"
        )
        self.calibration_label.hide()

        self._hidden_for_capture = False
        self._excluded_from_capture = False
        self._capture_exclusion_attempted = False
        self._calibration_mode = False
        self._calibration_target: Optional[Tuple[int, int]] = None
        self._guide_points: list[Tuple[int, int]] = []
        self._dot_radius = 20
        self._register_hotkeys()

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def show_overlay(self) -> None:
        self.showFullScreen()
        self.raise_()
        self._ensure_excluded_from_capture()
        self._set_click_through(True)

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
        self._set_click_through(False)
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
        self._set_click_through(True)
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

    def update_frame(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QtGui.QImage(rgb_frame.data, width, height, channels * width, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
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
            self._draw_dot(painter, point, active=(point == self._calibration_target))

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

    def _set_click_through(self, enabled: bool) -> None:
        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED
        if enabled:
            style |= WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

    def _ensure_excluded_from_capture(self) -> None:
        if self._capture_exclusion_attempted:
            return
        self._capture_exclusion_attempted = True
        hwnd = int(self.winId())
        result = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        self._excluded_from_capture = bool(result)

    def _register_hotkeys(self) -> None:
        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        user32.RegisterHotKey(hwnd, 1, MOD_CONTROL | MOD_ALT, ord("P"))
        user32.RegisterHotKey(hwnd, 2, MOD_CONTROL | MOD_ALT, 0xBB)
        user32.RegisterHotKey(hwnd, 3, MOD_CONTROL | MOD_ALT, 0xBD)
        user32.RegisterHotKey(hwnd, 4, MOD_CONTROL | MOD_ALT, ord("Q"))
        user32.RegisterHotKey(hwnd, 5, MOD_CONTROL | MOD_ALT, ord("C"))

    def _unregister_hotkeys(self) -> None:
        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        for hotkey_id in (1, 2, 3, 4, 5):
            user32.UnregisterHotKey(hwnd, hotkey_id)
