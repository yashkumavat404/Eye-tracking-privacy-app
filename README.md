---
title: Eye Tracking Privacy Screen
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
---

# Eye Tracking Privacy Screen

Browser-based privacy demo for Hugging Face Spaces. The app reads webcam frames in the browser, estimates gaze direction with MediaPipe Face Mesh, and applies a privacy effect when the user appears to look away or when the face is not tracked reliably.

## Features

- Browser webcam input
- CPU-friendly gaze estimation
- Clear center window when attention is on screen
- Full privacy blur when attention is lost
- Live status panel with detection confidence

## PyQt6 Desktop Dashboard

The commercial Windows dashboard entry point is:

```bash
pip install -r requirements.txt
python privacy_dashboard.py
```

It adds a modern dark desktop UI with:

- Sidebar navigation for Dashboard, Tracking, Calibration, Display, Performance, Settings, and About.
- Toggle cards, live confidence, radius controls, GPU/display status, and FPS.
- PyQt6 calibration overlay with animated fullscreen targets.
- pyqtgraph performance graphs updated every 500 ms.
- JSON settings persistence in `settings/privacy_spotlight_settings.json`.
- System tray menu with Open Dashboard, Toggle Privacy Mode, Start Calibration, and Quit.
- Global hotkeys matching the existing controls.

Dashboard modules:

- `privacy_dashboard.py`: PyQt6 application entry point.
- `backend/privacy_controller.py`: UI-to-backend adapter that reuses `FaceTracker`, `GazeEstimator`, and `CalibrationSession`.
- `backend/system_info.py`: GPU, CPU, RAM, display, and refresh-rate helpers.
- `ui/main_window.py`: main shell, sidebar, stacked pages, tray integration.
- `ui/pages.py`: dashboard, tracking, calibration, display, performance, settings, and about pages.
- `ui/widgets.py`: reusable cards, animated toggle switches, sliders, status pills, preview widget, and live graphs.
- `ui/floating_status.py`: compact always-on-top status widget.
- `settings/settings_manager.py`: JSON settings manager and defaults.

Production integration hooks are marked in `backend/privacy_controller.py`. The dashboard intentionally connects to the existing tracker, estimator, and calibration classes instead of replacing them.

## Files

- `app.py`: Hugging Face entry point
- `face_tracker.py`: gaze and head-pose extraction reused from the original project
- `gaze_estimator.py`: smoothing and mapping helpers from the original project
- `calibration.py`: lightweight calibration utilities
- `smoothing_filter.py`: temporal smoothing
- `requirements.txt`: deployment dependencies

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy To Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose `Gradio` as the SDK.
3. Upload all project files or push the folder with Git.
4. Wait for the Space build to finish.
5. Open the generated public Space URL.

## Notes

- The app is designed for CPU Spaces and does not require CUDA.
- Browser permission for webcam access must be allowed.
- For best results, keep your face centered and screen directly ahead.

## dashboard
Run the new dashboard like this:
cd "D:\CODEX\Privacy app"
.\.venv\Scripts\python.exe privacy_dashboard.py
Then test these in order:
Open the dashboard and check that all pages load.
Toggle Privacy Mode from the Dashboard page.
Try the hotkey: Ctrl + Alt + P.
Go to Calibration and click Start Calibration.
Adjust radius from Dashboard or Display and confirm the spotlight changes.
Right-click the tray icon and test Open Dashboard, Toggle Privacy Mode, and Quit.
If it fails to open, run this once:
cd "D:\CODEX\Privacy app"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe privacy_dashboard.py