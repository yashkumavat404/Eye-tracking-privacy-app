from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import psutil
from PyQt6 import QtWidgets


@dataclass
class DisplayInfo:
    resolution: str
    refresh_rate_hz: int
    screen_count: int


class SystemInfoProvider:
    def __init__(self, app: QtWidgets.QApplication) -> None:
        self.app = app

    def display_info(self) -> DisplayInfo:
        screen = self.app.primaryScreen()
        if screen is None:
            return DisplayInfo("Unknown", 60, 0)
        size = screen.size()
        refresh = int(round(screen.refreshRate() or 60))
        return DisplayInfo(f"{size.width()} x {size.height()}", refresh, len(self.app.screens()))

    def gpu_name(self) -> str:
        try:
            completed = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
            names = [line.strip() for line in completed.stdout.splitlines() if line.strip() and line.strip() != "Name"]
            if names:
                return names[0]
        except (OSError, subprocess.SubprocessError):
            pass
        return f"{platform.processor() or 'CPU'} fallback"

    def gpu_utilization(self) -> int:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=False,
                text=True,
                timeout=1,
            )
            value = completed.stdout.splitlines()[0].strip()
            return max(0, min(100, int(value)))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return 0

    @staticmethod
    def cpu_usage() -> float:
        return psutil.cpu_percent(interval=None)

    @staticmethod
    def ram_usage() -> float:
        return psutil.virtual_memory().percent

