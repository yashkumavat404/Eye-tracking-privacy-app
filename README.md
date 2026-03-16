# NVIDIA Privacy Spotlight

A desktop prototype that tracks your gaze with a webcam and keeps a circular area of the screen clear while blurring and dimming the background.

## Features

- Windows 10 and Windows 11 compatible
- Webcam only runs while privacy mode is enabled
- MediaPipe Face Mesh gaze estimation with head-pose compensation
- Full-screen privacy overlay with adjustable spotlight radius
- CUDA acceleration through PyTorch when an NVIDIA GPU is available
- Automatic CPU fallback when CUDA is unavailable
- Global hotkeys for toggle, resize, and exit

## Project Files

- `main.py`: app controller, timers, and lifecycle management
- `gaze_tracker.py`: webcam access, face landmarks, gaze estimation, smoothing
- `screen_blur.py`: screen capture, Gaussian blur, spotlight masking, GPU fallback logic
- `overlay_ui.py`: PyQt5 full-screen overlay and Windows global hotkeys
- `requirements.txt`: Python dependencies

## Installation

1. Install Python 3.10 or 3.11 on Windows.
2. Open PowerShell in the project folder.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

4. Install the dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Optional CUDA Acceleration

The blur path uses CUDA automatically when PyTorch can see a supported NVIDIA GPU.

If you want GPU acceleration:

1. Install the latest NVIDIA driver.
2. Install a CUDA-enabled PyTorch build if your default `pip install torch` did not include CUDA support.
3. Verify CUDA is visible to PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is not available, the app falls back to the CPU path.

## Run

```powershell
python main.py
```

## Hotkeys

- `Ctrl+Alt+P`: toggle privacy mode on or off
- `Ctrl+Alt+=`: increase spotlight radius
- `Ctrl+Alt+-`: decrease spotlight radius
- `Ctrl+Alt+Q`: exit the application

## Notes

- Accuracy is best when the screen is directly in front of the user.
- The overlay briefly hides before each screen capture so it does not capture itself.
- CPU mode works best at 1080p or below.
- Performance depends on CPU/GPU and Camera
- Requires minimum 2GB Ram 