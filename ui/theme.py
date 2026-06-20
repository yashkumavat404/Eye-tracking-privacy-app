APP_STYLE = """
* {
    font-family: "Segoe UI", "Inter", Arial;
    color: #ecf2f8;
}
QMainWindow, QWidget#Root {
    background: #0b0f14;
}
QFrame#Sidebar {
    background: #10161d;
    border-right: 1px solid #1f2b36;
}
QPushButton#NavButton {
    background: transparent;
    border: 0;
    border-radius: 8px;
    color: #9eacba;
    font-size: 14px;
    padding: 12px 14px;
    text-align: left;
}
QPushButton#NavButton:checked, QPushButton#NavButton:hover {
    background: #1b2733;
    color: #ffffff;
}
QFrame#Card {
    background: #121922;
    border: 1px solid #22303c;
    border-radius: 8px;
}
QLabel#PageTitle {
    font-size: 25px;
    font-weight: 700;
}
QLabel#SectionTitle {
    font-size: 16px;
    font-weight: 650;
}
QLabel#Muted {
    color: #8493a3;
}
QPushButton {
    background: #1d2a36;
    border: 1px solid #2e4050;
    border-radius: 8px;
    padding: 9px 13px;
}
QPushButton:hover {
    background: #263747;
}
QPushButton#PrimaryButton {
    background: #1f8fff;
    border: 1px solid #4aa5ff;
    color: white;
    font-weight: 650;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #263340;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #24c6dc;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #f5fbff;
    border: 2px solid #24c6dc;
    width: 17px;
    margin: -7px 0;
    border-radius: 9px;
}
QProgressBar {
    background: #263340;
    border: 0;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk {
    background: #24c6dc;
    border-radius: 5px;
}
QLineEdit, QComboBox {
    background: #0f151c;
    border: 1px solid #2c3b49;
    border-radius: 7px;
    padding: 8px;
}
QCheckBox {
    spacing: 9px;
}
"""

