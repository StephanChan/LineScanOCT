# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:46:51 2026

@author: shuai
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen, QPainter
from PyQt5.QtCore import Qt, QPoint, QEvent, pyqtSignal
import numpy as np

class InteractiveMosaicWidget(QWidget):
    # Signal to send new regions back to the Main GUI
    regions_updated = pyqtSignal(list) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None  # This will hold self.SampleMosaic
        self.display_scale = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        
        self.polygons = []
        self.current_polygon = []
        self.m_min, self.m_max = 0, 255 # Contrast limits

    def set_image(self, numpy_array, m, M):
        """Receives the stitched mosaic from DnSThread."""
        self.polygons = []
        self.current_polygon = []
        self.m_min, self.m_max = m, M
        # Normalize and convert to QImage
        self.adj = np.clip((numpy_array - m) / (M - m) * 255, 0, 255).astype(np.uint8)
        h, w = self.adj.shape
        self.image = QImage(self.adj.data, w, h, w, QImage.Format_Grayscale8)
        self.update()

    def paintEvent(self, event):
        if self.image is None: return
        painter = QPainter(self)
        
        # Use Antialiasing for the polygon lines
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw Background (Black)
        painter.fillRect(self.rect(), Qt.black)

        # 2. Calculate Transform for Zoom/Pan
        w, h = self.image.width(), self.image.height()
        sw, sh = w * self.display_scale, h * self.display_scale
        dx = (self.width() - sw) / 2 + (self.pan_x * self.display_scale)
        dy = (self.height() - sh) / 2 + (self.pan_y * self.display_scale)

        # 3. Draw Mosaic with Smooth Transformation
        # We pass Qt.SmoothTransformation here instead of to the painter hint
        scaled_img = self.image.scaled(int(sw), int(sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(int(dx), int(dy), scaled_img)

        # 4. Draw Polygons
        painter.setPen(QPen(Qt.green, 2))
        for poly in self.polygons:
            self._draw_poly(painter, poly, dx, dy, True)
        
        painter.setPen(QPen(Qt.red, 2))
        self._draw_poly(painter, self.current_polygon, dx, dy, False)
        
    def _draw_poly(self, painter, pts, dx, dy, closed):
        if not pts: return
        mapped = [QPoint(int(p[0]*self.display_scale + dx), int(p[1]*self.display_scale + dy)) for p in pts]
        for i in range(len(mapped)-1):
            painter.drawLine(mapped[i], mapped[i+1])
        if closed and len(mapped) > 2:
            painter.drawLine(mapped[-1], mapped[0])

    def mousePressEvent(self, event):
        # Calculate image coordinates
        w, h = self.image.width(), self.image.height()
        sw, sh = w * self.display_scale, h * self.display_scale
        dx = (self.width() - sw) / 2 + (self.pan_x * self.display_scale)
        dy = (self.height() - sh) / 2 + (self.pan_y * self.display_scale)
        
        img_x = (event.pos().x() - dx) / self.display_scale
        img_y = (event.pos().y() - dy) / self.display_scale

        if event.button() == Qt.LeftButton:
            self.current_polygon.append((img_x, img_y))
        elif event.button() == Qt.RightButton:
            if self.current_polygon: self.current_polygon.pop()
            elif self.polygons: self.polygons.pop()
        elif event.button() == Qt.MidButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        self.display_scale *= (1.15 if event.angleDelta().y() > 0 else 0.85)
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x() / self.display_scale
            self.pan_y += delta.y() / self.display_scale
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MidButton: self.is_dragging = False

    def finish_region(self):
        if len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            self.regions_updated.emit(self.polygons)
            self.update()