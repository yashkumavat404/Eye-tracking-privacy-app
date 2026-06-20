import sys
from typing import Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from calibration import CalibrationSession
from face_tracker import FaceTracker
from gaze_estimator import GazeEstimator
from overlay_renderer import PrivacyOverlay, SpotlightRenderer

HOTKEY_TEXT = (
    "Ctrl+Alt+P toggle, Ctrl+Alt+C calibrate, "
    "Ctrl+Alt+= larger, Ctrl+Alt+- smaller, Ctrl+Alt+Q quit"
)


class PrivacyViewApp(QtCore.QObject):
    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__()
        self.app = app
        self.overlay = PrivacyOverlay()
        self.renderer = SpotlightRenderer()
        self.face_tracker = FaceTracker()
        self.gaze_estimator = GazeEstimator(self.renderer.screen_size)
        self.calibration = CalibrationSession(self.renderer.screen_size)
        self.gaze_estimator.load_calibration()

        self.privacy_enabled = False
        self.radius = 260
        self.last_point: Optional[Tuple[int, int]] = self._screen_center()
        self._shutting_down = False
        self._frame_in_progress = False
        self.display_refresh_hz = self._detect_refresh_rate()
        self.calibration_started_tracker = False
        self._last_status_update_ms = 0

        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.frame_timer.setInterval(self._frame_interval_ms())
        self.frame_timer.timeout.connect(self._update_overlay_frame)

        self.overlay.toggle_requested.connect(self.toggle_privacy_mode)
        self.overlay.radius_increase_requested.connect(self.increase_radius)
        self.overlay.radius_decrease_requested.connect(self.decrease_radius)
        self.overlay.calibration_requested.connect(self.start_calibration)
        self.overlay.calibration_click_requested.connect(self.capture_calibration_point)
        self.overlay.exit_requested.connect(self.shutdown)

        self._refresh_status(None)
        self.overlay.show()
        self.overlay.hide()

    def _detect_refresh_rate(self) -> int:
        screen = self.app.primaryScreen()
        if screen is None:
            return 60
        refresh_rate = int(round(screen.refreshRate() or 60))
        return max(60, min(refresh_rate, 144))

    def _frame_interval_ms(self) -> int:
        return max(7, int(round(1000 / self.display_refresh_hz)))

    def _screen_center(self) -> Tuple[int, int]:
        width, height = self.renderer.screen_size
        return width // 2, height // 2

    def _calibration_status(self) -> str:
        return "Calibrated" if self.gaze_estimator.has_calibration() else "Not calibrated"

    def _status_text(self, observation) -> str:
        mode = "ON" if self.privacy_enabled else "OFF"
        blink_text = "Blink" if observation and observation.blink else "Eyes open"
        pose_text = "yaw 0.00 pitch 0.00"
        if observation and observation.face_detected:
            pose_text = f"yaw {observation.yaw:.2f} pitch {observation.pitch:.2f}"

        line_one = (
            f"Privacy mode: {mode} | Radius: {self.radius}px | "
            f"Display: {self.display_refresh_hz}Hz | "
            f"Calibration: {self._calibration_status()} | "
            f"Acceleration: {self.renderer.acceleration_label}"
        )
        line_two = (
            f"Tracking: {blink_text} | Head pose: {pose_text} | "
            f"Hotkeys: {HOTKEY_TEXT}"
        )
        return (
            f"{line_one}\n"
            f"{line_two}"
        )

    def _refresh_status(self, observation) -> None:
        self.overlay.set_status_text(self._status_text(observation))

    def _refresh_status_if_due(self, observation) -> None:
        now_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now_ms - self._last_status_update_ms < 250:
            return
        self._last_status_update_ms = now_ms
        self._refresh_status(observation)

    def toggle_privacy_mode(self) -> None:
        if self.calibration.active:
            return

        self.privacy_enabled = not self.privacy_enabled
        if self.privacy_enabled:
            self.face_tracker.start()
            self.gaze_estimator.reset()
            self.frame_timer.start()
            self.overlay.show_overlay()
        else:
            self.frame_timer.stop()
            self.face_tracker.stop()
            self.overlay.hide_overlay()

        self._refresh_status(self.face_tracker.get_latest_observation())

    def start_calibration(self) -> None:
        if self.calibration.active:
            return

        self.calibration.start()
        self.gaze_estimator.clear_calibration()
        self.calibration_started_tracker = not self.privacy_enabled and not self.face_tracker.running
        self.face_tracker.start()

        self._show_current_calibration_target()
        self._refresh_status(self.face_tracker.get_latest_observation())

    def _show_current_calibration_target(self) -> None:
        current = self.calibration.current_target()
        if current is None:
            return

        label, position, message = current
        prompt = (
            f"{self.calibration.progress_text()}\n"
            f"{message}\nKeep your head still, keep both eyes open, and click while looking at the highlighted dot."
        )
        guide_points = [target[1] for target in self.calibration.targets]
        if self.calibration.index == 0:
            self.overlay.begin_calibration_mode(position, prompt, guide_points)
        else:
            self.overlay.update_calibration_prompt(position, prompt, guide_points)

    def capture_calibration_point(self) -> None:
        current = self.calibration.current_target()
        if current is None:
            return

        label, position, _ = current
        guide_points = [target[1] for target in self.calibration.targets]
        self.overlay.update_calibration_prompt(
            position,
            f"{self.calibration.progress_text()}\nSampling {label.replace('_', ' ')}... keep looking at the highlighted dot.",
            guide_points,
        )
        self.app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        gaze_vector = self.face_tracker.collect_gaze_vector_sample()
        if gaze_vector is None:
            self.overlay.update_calibration_prompt(
                position,
                f"{self.calibration.progress_text()}\nTracking was unstable. Keep still and click again.",
                guide_points,
            )
            return

        self.gaze_estimator.add_calibration_sample(label, position, gaze_vector)
        if not self.calibration.advance():
            self.finish_calibration()
            return

        self._show_current_calibration_target()

    def finish_calibration(self) -> None:
        self.calibration.stop()
        self.overlay.end_calibration_mode()
        if not self.privacy_enabled and self.calibration_started_tracker:
            self.face_tracker.stop()
        self.calibration_started_tracker = False
        if not self.privacy_enabled:
            self.overlay.hide()
        self._refresh_status(self.face_tracker.get_latest_observation())

    def increase_radius(self) -> None:
        self.radius = min(self.radius + 24, 560)
        self._refresh_status(self.face_tracker.get_latest_observation())

    def decrease_radius(self) -> None:
        self.radius = max(self.radius - 24, 120)
        self._refresh_status(self.face_tracker.get_latest_observation())

    def _update_overlay_frame(self) -> None:
        if not self.privacy_enabled or self._frame_in_progress or self.calibration.active:
            return

        self._frame_in_progress = True
        try:
            observation = self.face_tracker.get_latest_observation()
            estimate = self.gaze_estimator.estimate(observation)
            if estimate is not None:
                self.last_point = estimate.screen_point
            elif self.last_point is None:
                self.last_point = self._screen_center()

            self.overlay.prepare_for_capture()
            self.app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            frame = self.renderer.capture_screen()
            processed = self.renderer.render_spotlight(frame, self.last_point, self.radius)
            self.overlay.restore_after_capture()
            self.overlay.update_frame(processed)
            self._refresh_status_if_due(observation)
        finally:
            self._frame_in_progress = False

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.frame_timer.stop()
        self.face_tracker.stop()
        self.overlay.close()
        self.app.quit()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("NVIDIA Privacy Spotlight")
    app.setQuitOnLastWindowClosed(False)

    controller = PrivacyViewApp(app)
    app.aboutToQuit.connect(controller.shutdown)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
