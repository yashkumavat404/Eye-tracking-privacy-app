from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from backend import PrivacyController
from settings import SettingsManager
from ui import MainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Privacy Spotlight")
    app.setOrganizationName("Privacy Spotlight")
    app.setQuitOnLastWindowClosed(False)

    settings = SettingsManager()
    controller = PrivacyController(app, settings)
    window = MainWindow(app, controller, settings)
    app.aboutToQuit.connect(controller.shutdown)

    if settings.get("start_minimized"):
        window.hide()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
