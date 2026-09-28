from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from backend.system_info import SystemInfoProvider
from settings import SettingsManager
from ui.widgets import Card, LabeledSlider, LiveGraph, MetricRow, PreviewWidget, StatusPill, ToggleSwitch


class DashboardPage(QtWidgets.QWidget):
    def __init__(self, settings: SettingsManager, system: SystemInfoProvider) -> None:
        super().__init__()
        self.system = system
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(18)
        title = QtWidgets.QLabel("Dashboard")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(14)
        self.privacy_card = Card("Privacy Mode")
        row = QtWidgets.QHBoxLayout()
        self.privacy_pill = StatusPill()
        self.privacy_switch = ToggleSwitch(settings.get("privacy_enabled"))
        row.addWidget(self.privacy_pill)
        row.addStretch()
        row.addWidget(self.privacy_switch)
        self.privacy_card.layout.addLayout(row)
        self.eye_card = Card("Eye Tracking")
        self.eye_status = MetricRow("Live status", "Waiting")
        self.eye_confidence = MetricRow("Confidence", "0%")
        self.eye_card.layout.addWidget(self.eye_status)
        self.eye_card.layout.addWidget(self.eye_confidence)
        self.head_card = Card("Head Pose")
        self.head_status = MetricRow("State", "Enabled")
        self.head_card.layout.addWidget(self.head_status)
        self.radius_card = Card("Current Radius")
        self.radius_slider = LabeledSlider("Spotlight radius", 120, 560, settings.get("radius"), " px")
        self.radius_card.layout.addWidget(self.radius_slider)
        grid.addWidget(self.privacy_card, 0, 0)
        grid.addWidget(self.eye_card, 0, 1)
        grid.addWidget(self.head_card, 1, 0)
        grid.addWidget(self.radius_card, 1, 1)
        layout.addLayout(grid)

        lower = QtWidgets.QHBoxLayout()
        self.gpu_card = Card("GPU Information")
        self.gpu_name = MetricRow("GPU", system.gpu_name())
        self.gpu_util = MetricRow("GPU utilization", "0%")
        self.processing = MetricRow("Processing device", "Detecting")
        self.gpu_card.layout.addWidget(self.gpu_name)
        self.gpu_card.layout.addWidget(self.gpu_util)
        self.gpu_card.layout.addWidget(self.processing)
        self.display_card = Card("Display Information")
        display = system.display_info()
        self.resolution = MetricRow("Resolution", display.resolution)
        self.refresh = MetricRow("Refresh Rate", f"{display.refresh_rate_hz} Hz")
        self.screens = MetricRow("Displays", str(display.screen_count))
        self.fps = MetricRow("FPS", "0.0")
        self.display_card.layout.addWidget(self.resolution)
        self.display_card.layout.addWidget(self.refresh)
        self.display_card.layout.addWidget(self.screens)
        self.display_card.layout.addWidget(self.fps)
        lower.addWidget(self.gpu_card)
        lower.addWidget(self.display_card)
        layout.addLayout(lower)
        layout.addStretch()

    def update_status(self, data: dict) -> None:
        enabled = data.get("privacy_enabled", False)
        self.privacy_pill.set_state(enabled, "ON" if enabled else "OFF")
        self.privacy_switch.blockSignals(True)
        self.privacy_switch.setChecked(enabled)
        self.privacy_switch.blockSignals(False)
        face = data.get("face_detected", False)
        self.eye_status.value.setText("Tracking" if face else "Waiting")
        self.eye_confidence.value.setText(f"{data.get('confidence', 0)}%")
        self.head_status.value.setText("Enabled" if data.get("head_pose_enabled") else "Disabled")
        self.radius_slider.slider.blockSignals(True)
        self.radius_slider.setValue(data.get("radius", 260))
        self.radius_slider.slider.blockSignals(False)
        self.processing.value.setText(data.get("acceleration", "CPU"))
        self.fps.value.setText(f"{data.get('fps', 0):.1f}")


class TrackingPage(QtWidgets.QWidget):
    def __init__(self, settings: SettingsManager) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Tracking")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        controls = Card("Tracking Controls")
        self.eye_switch = ToggleSwitch(settings.get("eye_tracking_enabled"))
        self.head_switch = ToggleSwitch(settings.get("head_pose_enabled"))
        controls.layout.addLayout(self._switch_row("Eye Tracking", "Enable or disable gaze tracking input.", self.eye_switch))
        controls.layout.addLayout(self._switch_row("Head Pose Tracking", "Enable or disable head pose contribution.", self.head_switch))
        self.sensitivity = LabeledSlider("Tracking Sensitivity", 0, 100, settings.get("tracking_sensitivity"))
        self.smoothing = LabeledSlider("Smoothing", 0, 100, settings.get("smoothing"))
        self.threshold = LabeledSlider("Confidence Threshold", 0, 100, settings.get("confidence_threshold"))
        controls.layout.addWidget(self.sensitivity)
        controls.layout.addWidget(self.smoothing)
        controls.layout.addWidget(self.threshold)
        layout.addWidget(controls)

        diagnostics = Card("Live Tracking Diagnostics")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        self.camera_value = QtWidgets.QLabel("Waiting")
        self.face_value = QtWidgets.QLabel("Waiting")
        self.iris_value = QtWidgets.QLabel("Waiting")
        self.gaze_value = QtWidgets.QLabel("—")
        self.pupil_value = QtWidgets.QLabel("L: —   R: —")
        self.pose_value = QtWidgets.QLabel("Y: 0.000   P: 0.000   R: 0.000")
        self.motion_value = QtWidgets.QLabel("0.000")
        self.track_fps_value = QtWidgets.QLabel("0.0")
        self.blink_value = QtWidgets.QLabel("No")
        rows = [
            ("Camera", self.camera_value),
            ("Face", self.face_value),
            ("Iris landmarks", self.iris_value),
            ("Gaze vector", self.gaze_value),
            ("Iris centers", self.pupil_value),
            ("Head pose", self.pose_value),
            ("Gaze motion", self.motion_value),
            ("Tracker FPS", self.track_fps_value),
            ("Blink", self.blink_value),
        ]
        for index, (label, value) in enumerate(rows):
            row = index % 3
            col = (index // 3) * 2
            grid.addWidget(QtWidgets.QLabel(label), row, col)
            grid.addWidget(value, row, col + 1)
        diagnostics.layout.addLayout(grid)
        layout.addWidget(diagnostics)
        layout.addStretch()

    def update_diagnostics(self, data: dict) -> None:
        self.camera_value.setText("Open" if data.get("camera_open") else "Not available")
        self.face_value.setText("Detected" if data.get("face_detected") else "Not detected")
        self.iris_value.setText("Detected" if data.get("iris_detected") else "Not detected")
        gaze = data.get("gaze_vector")
        self.gaze_value.setText("—" if gaze is None else f"X {gaze[0]:.3f}   Y {gaze[1]:.3f}")
        left = data.get("left_pupil")
        right = data.get("right_pupil")
        left_text = "—" if left is None else f"{left[0]:.0f},{left[1]:.0f}"
        right_text = "—" if right is None else f"{right[0]:.0f},{right[1]:.0f}"
        self.pupil_value.setText(f"L: {left_text}   R: {right_text}")
        self.pose_value.setText(
            f"Y: {data.get('yaw', 0.0):.3f}   "
            f"P: {data.get('pitch', 0.0):.3f}   "
            f"R: {data.get('roll', 0.0):.3f}"
        )
        self.motion_value.setText(f"{data.get('gaze_motion', 0.0):.4f}")
        self.track_fps_value.setText(f"{data.get('tracking_fps', 0.0):.1f}")
        self.blink_value.setText("Yes" if data.get("blink") else "No")

    @staticmethod
    def _switch_row(title: str, subtitle: str, switch: ToggleSwitch) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        labels = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(title)
        muted = QtWidgets.QLabel(subtitle)
        muted.setObjectName("Muted")
        labels.addWidget(label)
        labels.addWidget(muted)
        row.addLayout(labels)
        row.addStretch()
        row.addWidget(switch)
        return row


class CalibrationPage(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Calibration")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        card = Card("Calibration Wizard")
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Calibration")
        self.start_button.setObjectName("PrimaryButton")
        self.stop_button = QtWidgets.QPushButton("Stop Calibration")
        self.reset_button = QtWidgets.QPushButton("Reset Calibration")
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.status = MetricRow("Calibration Status", "Waiting")
        self.accuracy = MetricRow("Accuracy", "0%")
        card.layout.addLayout(buttons)
        card.layout.addWidget(self.progress)
        card.layout.addWidget(self.status)
        card.layout.addWidget(self.accuracy)
        layout.addWidget(card)
        layout.addStretch()

    def update_calibration(self, data: dict) -> None:
        self.status.value.setText(data.get("state", "Waiting"))
        self.progress.setValue(data.get("progress", 0))
        self.accuracy.value.setText(f"{data.get('accuracy', 0)}%")


class DisplayPage(QtWidgets.QWidget):
    def __init__(self, settings: SettingsManager) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Display")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        card = Card("Privacy Mode Preview")
        self.radius = LabeledSlider("Spotlight Radius", 120, 560, settings.get("radius"), " px")
        self.brightness = LabeledSlider("Brightness Reduction", 0, 100, settings.get("brightness_reduction"), "%")
        self.softness = LabeledSlider("Spotlight Softness", 0, 100, settings.get("spotlight_softness"), "%")
        self.opacity = LabeledSlider("Spotlight Opacity", 10, 100, settings.get("spotlight_opacity"), "%")
        self.preview = PreviewWidget()
        for slider in (self.radius, self.brightness, self.softness, self.opacity):
            card.layout.addWidget(slider)
        card.layout.addWidget(self.preview)
        layout.addWidget(card)
        layout.addStretch()


class PerformancePage(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Performance")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(14)
        self.cpu = LiveGraph("CPU Usage", "#24c6dc")
        self.ram = LiveGraph("RAM Usage", "#7bd88f")
        self.gpu = LiveGraph("GPU Usage", "#f7b955")
        self.fps = LiveGraph("FPS Graph", "#ff6f91")
        grid.addWidget(self.cpu, 0, 0)
        grid.addWidget(self.ram, 0, 1)
        grid.addWidget(self.gpu, 1, 0)
        grid.addWidget(self.fps, 1, 1)
        layout.addLayout(grid)


class SettingsPage(QtWidgets.QWidget):
    def __init__(self, settings: SettingsManager) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Settings")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        hotkeys = Card("Hotkey Configuration")
        self.hotkey_fields: dict[str, QtWidgets.QLineEdit] = {}
        labels = {
            "toggle_privacy": "Toggle Privacy Mode",
            "calibration": "Calibration",
            "radius_increase": "Radius Increase",
            "radius_decrease": "Radius Decrease",
            "quit": "Quit",
        }
        for key, label in labels.items():
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            field = QtWidgets.QLineEdit(settings.get("hotkeys", {}).get(key, ""))
            self.hotkey_fields[key] = field
            row.addWidget(field)
            hotkeys.layout.addLayout(row)
        startup = Card("Startup Options")
        self.launch = QtWidgets.QCheckBox("Launch on Startup")
        self.minimized = QtWidgets.QCheckBox("Start Minimized")
        self.start_privacy = QtWidgets.QCheckBox("Enable Privacy Mode at Startup")
        self.launch.setChecked(settings.get("launch_on_startup"))
        self.minimized.setChecked(settings.get("start_minimized"))
        self.start_privacy.setChecked(settings.get("privacy_on_startup"))
        startup.layout.addWidget(self.launch)
        startup.layout.addWidget(self.minimized)
        startup.layout.addWidget(self.start_privacy)
        theme = Card("Theme")
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(["Dark", "Light"])
        self.theme.setCurrentText(settings.get("theme"))
        self.logs = QtWidgets.QCheckBox("Enable logs")
        self.logs.setChecked(settings.get("logs_enabled"))
        theme.layout.addWidget(self.theme)
        theme.layout.addWidget(self.logs)
        layout.addWidget(hotkeys)
        layout.addWidget(startup)
        layout.addWidget(theme)
        layout.addStretch()


class AboutPage(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        title = QtWidgets.QLabel("About")
        title.setObjectName("PageTitle")
        card = Card("Privacy Spotlight")
        body = QtWidgets.QLabel(
            "Commercial desktop dashboard for real-time gaze-aware screen privacy, calibration, "
            "performance monitoring, tray control, and enterprise-ready settings persistence."
        )
        body.setWordWrap(True)
        body.setObjectName("Muted")
        card.layout.addWidget(body)
        layout.addWidget(title)
        layout.addWidget(card)
        layout.addStretch()

