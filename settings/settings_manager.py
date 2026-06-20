from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "privacy_enabled": False,
    "eye_tracking_enabled": True,
    "head_pose_enabled": True,
    "radius": 260,
    "tracking_sensitivity": 70,
    "smoothing": 55,
    "confidence_threshold": 55,
    "brightness_reduction": 70,
    "spotlight_softness": 55,
    "spotlight_opacity": 90,
    "launch_on_startup": False,
    "start_minimized": False,
    "privacy_on_startup": False,
    "theme": "Dark",
    "logs_enabled": True,
    "hotkeys": {
        "toggle_privacy": "Ctrl+Alt+P",
        "calibration": "Ctrl+Alt+C",
        "radius_increase": "Ctrl+Alt+=",
        "radius_decrease": "Ctrl+Alt+-",
        "quit": "Ctrl+Alt+Q",
    },
}


class SettingsManager:
    """JSON-backed settings store used by the dashboard and backend adapter."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).resolve().parent / "privacy_spotlight_settings.json"
        self._data = deepcopy(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.save()
            return self._data

        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}

        self._data = self._merge(deepcopy(DEFAULT_SETTINGS), loaded)
        return self._data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any, autosave: bool = True) -> None:
        self._data[key] = value
        if autosave:
            self.save()

    def update(self, values: dict[str, Any], autosave: bool = True) -> None:
        self._data.update(values)
        if autosave:
            self.save()

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    @staticmethod
    def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = SettingsManager._merge(base[key], value)
            else:
                base[key] = value
        return base

