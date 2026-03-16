from typing import Tuple

import cv2
import mss
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_LOAD_ERROR = None
except (ImportError, OSError) as exc:
    torch = None
    F = None
    TORCH_LOAD_ERROR = exc


class ScreenBlurProcessor:
    def __init__(self) -> None:
        self.sct = mss.mss()
        monitor = self.sct.monitors[1]
        self.monitor = monitor
        self.screen_size = (monitor["width"], monitor["height"])
        self.render_scale = self._choose_render_scale()
        self._kernel_cache = {}
        self._cuda_filter = None
        self._cuda_filter_shape = None

        self.use_torch_cuda = bool(torch is not None and torch.cuda.is_available())
        self.use_cv2_cuda = False
        self.cv2_cuda_error = None

        if not self.use_torch_cuda:
            self.use_cv2_cuda = self._init_cv2_cuda()

        if self.use_torch_cuda:
            self.acceleration_label = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif self.use_cv2_cuda:
            self.acceleration_label = "OpenCV CUDA"
        elif TORCH_LOAD_ERROR is not None:
            self.acceleration_label = "CPU fallback"
        else:
            self.acceleration_label = "CPU"

    def _init_cv2_cuda(self) -> bool:
        try:
            if not hasattr(cv2, "cuda"):
                return False
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count <= 0:
                return False
            cv2.cuda.setDevice(0)
            return True
        except cv2.error as exc:
            self.cv2_cuda_error = str(exc)
            return False

    def _choose_render_scale(self) -> float:
        width, height = self.screen_size
        pixel_count = width * height

        if pixel_count >= 2560 * 1440:
            return 0.50
        if pixel_count >= 1920 * 1080:
            return 0.60
        return 0.75

    def capture_screen(self) -> np.ndarray:
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def apply_privacy_effect(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        mask = self._create_spotlight_mask(frame.shape[:2], gaze_point, radius)
        background = self._build_background(frame, gaze_point, radius)

        clear_region = frame.astype(np.float32) * mask[..., None]
        background_region = background.astype(np.float32) * (1.0 - mask[..., None])
        blended = clear_region + background_region
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _build_background(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        if self.render_scale < 0.999:
            small_frame = self._resize_frame(frame, self.render_scale)
            scaled_point = (
                int(gaze_point[0] * self.render_scale),
                int(gaze_point[1] * self.render_scale),
            )
            scaled_radius = max(24, int(radius * self.render_scale))
            processed_small = self._build_background_native(
                small_frame, scaled_point, scaled_radius
            )
            return cv2.resize(
                processed_small,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return self._build_background_native(frame, gaze_point, radius)

    def _build_background_native(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        blurred = self._blur_frame(frame)
        mask = self._create_spotlight_mask(frame.shape[:2], gaze_point, radius)
        darkening = (1.0 - mask[..., None]) * 0.28
        output = blurred.astype(np.float32) * (1.0 - darkening)
        return np.clip(output, 0, 255).astype(np.uint8)

    def _resize_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        height, width = frame.shape[:2]
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _blur_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.use_torch_cuda:
            return self._torch_gaussian_blur(frame)
        if self.use_cv2_cuda:
            blurred = self._cv2_cuda_blur(frame)
            if blurred is not None:
                return blurred
            self.use_cv2_cuda = False
            self.acceleration_label = "CPU fallback"
        return cv2.GaussianBlur(frame, (31, 31), 0, borderType=cv2.BORDER_REPLICATE)

    def _cv2_cuda_blur(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            if self._cuda_filter is None or self._cuda_filter_shape != frame.shape[:2]:
                self._cuda_filter = cv2.cuda.createGaussianFilter(
                    cv2.CV_8UC3,
                    cv2.CV_8UC3,
                    (31, 31),
                    0,
                )
                self._cuda_filter_shape = frame.shape[:2]
            blurred_gpu = self._cuda_filter.apply(gpu_frame)
            return blurred_gpu.download()
        except cv2.error as exc:
            self.cv2_cuda_error = str(exc)
            return None

    def _torch_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        assert torch is not None and F is not None

        tensor = torch.from_numpy(frame).to(torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
        kernel = self._gaussian_kernel(size=25, sigma=6.5, device=tensor.device, channels=3)
        padded = F.pad(tensor, (12, 12, 12, 12), mode="reflect")
        blurred = F.conv2d(padded, kernel, groups=3)
        output = blurred.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
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

    def _create_spotlight_mask(
        self,
        image_shape: Tuple[int, int],
        gaze_point: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(mask, gaze_point, radius, 1.0, -1, lineType=cv2.LINE_AA)
        feather = max(radius // 3, 18)
        blur_amount = feather * 2 + 1
        return cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
