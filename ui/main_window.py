from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from backend import PrivacyController, SystemInfoProvider
from settings import SettingsManager
from ui.floating_status import FloatingStatusWidget
from ui.pages import (
    AboutPage,
    CalibrationPage,
    DashboardPage,
    DisplayPage,
    PerformancePage,
    SettingsPage,
    TrackingPage,
)
from ui.theme import APP_STYLE


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app: QtWidgets.QApplication, controller: PrivacyController, settings: SettingsManager) -> None:
        super().__init__()
        self.app = app
        self.controller = controller
        self.settings = settings
        self.system = SystemInfoProvider(app)
        self.setWindowTitle("Privacy Spotlight")
        self.resize(1240, 780)
        self.setMinimumSize(1040, 680)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._build_tray()
        self._connect()
        self.performance_timer = QtCore.QTimer(self)
        self.performance_timer.setInterval(500)
        self.performance_timer.timeout.connect(self._update_performance)
        self.performance_timer.start()
        self.floating_status = FloatingStatusWidget()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        shell = QtWidgets.QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(218)
        nav = QtWidgets.QVBoxLayout(self.sidebar)
        nav.setContentsMargins(16, 18, 16, 18)
        brand = QtWidgets.QLabel("Privacy\nSpotlight")
        brand.setStyleSheet("font-size: 23px; font-weight: 800; color: white;")
        nav.addWidget(brand)
        nav.addSpacing(16)
        self.stack = QtWidgets.QStackedWidget()
        self.dashboard = DashboardPage(self.settings, self.system)
        self.tracking = TrackingPage(self.settings)
        self.calibration = CalibrationPage()
        self.display = DisplayPage(self.settings)
        self.performance = PerformancePage()
        self.settings_page = SettingsPage(self.settings)
        self.about = AboutPage()
        pages = [
            ("Dashboard", self.dashboard),
            ("Tracking", self.tracking),
            ("Calibration", self.calibration),
            ("Display", self.display),
            ("Performance", self.performance),
            ("Settings", self.settings_page),
            ("About", self.about),
        ]
        self.nav_buttons: list[QtWidgets.QPushButton] = []
        for index, (name, page) in enumerate(pages):
            self.stack.addWidget(page)
            button = QtWidgets.QPushButton(name)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=index: self._set_page(i))
            self.nav_buttons.append(button)
            nav.addWidget(button)
        nav.addStretch()
        version = QtWidgets.QLabel("PyQt6 dashboard")
        version.setObjectName("Muted")
        nav.addWidget(version)
        shell.addWidget(self.sidebar)
        shell.addWidget(self.stack)
        self._set_page(0)

    def _set_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)

    def _build_tray(self) -> None:
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(icon)
        self.tray = QtWidgets.QSystemTrayIcon(icon, self)
        menu = QtWidgets.QMenu()
        self.open_action = menu.addAction("Open Dashboard")
        self.toggle_action = menu.addAction("Toggle Privacy Mode")
        self.calibrate_action = menu.addAction("Start Calibration")
        menu.addSeparator()
        self.quit_action = menu.addAction("Quit")
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Privacy Spotlight")
        self.tray.show()

    def _connect(self) -> None:
        self.dashboard.privacy_switch.toggledAnimated.connect(self.controller.enable_privacy_mode)
        self.dashboard.radius_slider.valueChanged.connect(self.controller.set_radius)
        self.tracking.eye_switch.toggledAnimated.connect(self.controller.set_eye_tracking_enabled)
        self.tracking.head_switch.toggledAnimated.connect(self.controller.set_head_pose_enabled)
        self.tracking.sensitivity.valueChanged.connect(lambda value: self.controller.update_setting("tracking_sensitivity", value))
        self.tracking.smoothing.valueChanged.connect(lambda value: self.controller.update_setting("smoothing", value))
        self.tracking.threshold.valueChanged.connect(lambda value: self.controller.update_setting("confidence_threshold", value))
        self.calibration.start_button.clicked.connect(self.controller.start_calibration)
        self.calibration.stop_button.clicked.connect(self.controller.stop_calibration)
        self.calibration.reset_button.clicked.connect(self.controller.reset_calibration)
        self.display.radius.valueChanged.connect(self.controller.set_radius)
        self.display.radius.valueChanged.connect(lambda value: self.display.preview.set_values(radius=value))
        self.display.brightness.valueChanged.connect(lambda value: self.controller.update_setting("brightness_reduction", value))
        self.display.softness.valueChanged.connect(lambda value: self.controller.update_setting("spotlight_softness", value))
        self.display.softness.valueChanged.connect(lambda value: self.display.preview.set_values(softness=value))
        self.display.opacity.valueChanged.connect(lambda value: self.controller.update_setting("spotlight_opacity", value))
        self.display.opacity.valueChanged.connect(lambda value: self.display.preview.set_values(opacity=value))
        self.settings_page.launch.toggled.connect(lambda value: self.controller.update_setting("launch_on_startup", value))
        self.settings_page.minimized.toggled.connect(lambda value: self.controller.update_setting("start_minimized", value))
        self.settings_page.start_privacy.toggled.connect(lambda value: self.controller.update_setting("privacy_on_startup", value))
        self.settings_page.theme.currentTextChanged.connect(lambda value: self.controller.update_setting("theme", value))
        self.settings_page.logs.toggled.connect(lambda value: self.controller.update_setting("logs_enabled", value))
        for key, field in self.settings_page.hotkey_fields.items():
            field.editingFinished.connect(lambda key=key, field=field: self._save_hotkey(key, field.text()))
        self.controller.status_changed.connect(self._update_status)
        self.controller.calibration_changed.connect(self.calibration.update_calibration)
        self.controller.notification_requested.connect(self._notify)
        self.open_action.triggered.connect(self.show_dashboard)
        self.toggle_action.triggered.connect(self.controller.toggle_privacy_mode)
        self.calibrate_action.triggered.connect(self.controller.start_calibration)
        self.quit_action.triggered.connect(self.controller.shutdown)
        self.tray.activated.connect(self._tray_activated)

    def _save_hotkey(self, key: str, text: str) -> None:
        hotkeys = self.settings.get("hotkeys", {}).copy()
        hotkeys[key] = text
        self.settings.set("hotkeys", hotkeys)
        self.controller.hotkeys.update_hotkeys(hotkeys)
        self._notify("Hotkey saved", "Global hotkeys were re-registered.")

    def _update_status(self, data: dict) -> None:
        self.dashboard.update_status(data)
        self.display.radius.slider.blockSignals(True)
        self.display.radius.setValue(data.get("radius", 260))
        self.display.radius.slider.blockSignals(False)
        self.display.preview.set_values(radius=data.get("radius", 260))
        self.floating_status.update_status(data)

    def _update_performance(self) -> None:
        self.performance.cpu.add_sample(self.system.cpu_usage())
        self.performance.ram.add_sample(self.system.ram_usage())
        self.performance.gpu.add_sample(self.system.gpu_utilization())
        self.performance.fps.add_sample(min(100.0, self.controller.fps))
        self.dashboard.gpu_util.value.setText(f"{self.system.gpu_utilization()}%")

    def _notify(self, title: str, message: str) -> None:
        self.tray.showMessage(title, message, self.windowIcon(), 2500)

    def _tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_dashboard()

    def show_dashboard(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.tray.isVisible():
            event.ignore()
            self.hide()
            self._notify("Privacy Spotlight is still running", "Use the tray menu to quit.")
            return
        super().closeEvent(event)
