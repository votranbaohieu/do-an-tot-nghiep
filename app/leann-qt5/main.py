from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200, 200, 1000, 1000)
    win.setWindowTitle('Hello World Long')

    win.show()
    sys.exit(app.exec_())

window()