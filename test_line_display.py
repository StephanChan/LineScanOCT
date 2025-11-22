# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 07:28:17 2025

@author: shuai
"""
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor


def fastLinePlot(data, width=800, height=300, ymin=None, ymax=None):
    if ymin is None: ymin = np.min(data)
    if ymax is None: ymax = np.max(data)

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(0, 0, 200), 2)
    painter.setPen(pen)

    n = len(data)
    x_scale = width / (n - 1)
    y_scale = height / (ymax - ymin)

    points = [
        QPointF(i * x_scale, height - ((data[i] - ymin) * y_scale))
        for i in range(n)
    ]

    painter.drawPolyline(*points)
    painter.end()
    return pixmap


class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast Line Plot Test (QPainter)")

        # UI
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Timer for 30+ FPS updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_waveform)
        self.timer.start(30)  # ~33 FPS

    def update_waveform(self):
        # Generate test waveform (sine + noise)
        t = np.linspace(0, 2*np.pi, 800)
        waveform = np.sin(5*t) + 0.2*np.random.randn(800)

        pixmap = fastLinePlot(waveform, width=800, height=300, ymin=-2, ymax=2)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TestWindow()
    win.show()
    sys.exit(app.exec_())
