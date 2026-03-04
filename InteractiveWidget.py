# -*- coding: utf-8 -*-
"""
Modified on Wed Mar 4 2026
@author: admin
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen, QPainter
from PyQt5.QtCore import Qt, QPoint, QEvent, pyqtSignal, QRectF, QPointF
import numpy as np

class InteractiveMosaicWidget(QWidget):
    # Signal to send new regions back to the Main GUI
    regions_updated = pyqtSignal(list) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None  # This will hold the raw QPixmap
        self.display_scale = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        
        self.polygons = []
        self.current_polygon = []
        self.m_min, self.m_max = 0, 255 # Contrast limits
        
        # New: Ratio of Vertical pixel size to Horizontal pixel size
        self.pixel_aspect_ratio = 1.0

    def set_image(self, numpy_array, m, M, pixel_size_x=1.0, pixel_size_y=1.0):
        """
        Receives the stitched mosaic and the physical resolution.
        pixel_size_x: step size in mm or um for the horizontal axis
        pixel_size_y: step size in mm or um for the vertical axis
        """
        self.polygons = []
        self.current_polygon = []
        self.m_min, self.m_max = m, M
        
        # 1. Calculate aspect ratio for physical pixel stretching
        if pixel_size_x != 0:
            self.pixel_aspect_ratio = pixel_size_y / pixel_size_x
        else:
            self.pixel_aspect_ratio = 1.0

        # 2. Normalize intensity (Contrast Stretching)
        adj = ((numpy_array - m) / (M - m + 1e-5) * 255)
        adj = np.clip(adj, 0, 255).astype(np.uint8)
        self.adj = adj # Keep reference for internal logic if needed
        
        # 3. Determine if image is Grayscale (2D) or RGB (3D)
        if adj.ndim == 2:
            # Grayscale processing
            h, w = adj.shape
            # QImage format for 8-bit grayscale
            qt_image = QImage(adj.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif adj.ndim == 3 and adj.shape[2] == 3:
            # RGB processing
            h, w, ch = adj.shape
            bytes_per_line = ch * w
            # QImage format for 24-bit RGB
            # Note: Ensure adj is in RGB order. If it's BGR (from OpenCV), 
            # use Format_BGR888 or swap channels first.
            qt_image = QImage(adj.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            raise ValueError(f"Unsupported image dimensions: {adj.shape}")

        # 4. Handle Physical Pixel Size Stretching (Aspect Ratio)
        # We stretch the image physically so that pixels appear square in the UI
        if self.pixel_aspect_ratio != 1.0:
            new_h = int(h * self.pixel_aspect_ratio)
            self.image = QPixmap.fromImage(qt_image).scaled(w, new_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        else:
            self.image = QPixmap.fromImage(qt_image)

        self.update()

    def get_view_params(self):
        """Helper to calculate layout parameters used in both painting and mouse events."""
        if self.image is None:
            return 0, 0, 0, 0
            
        base_w = self.image.width()
        # Apply the vertical stretch to the height
        base_h = self.image.height() * self.pixel_aspect_ratio
        
        sw = base_w * self.display_scale
        sh = base_h * self.display_scale
        
        dx = (self.width() - sw) / 2 + (self.pan_x * self.display_scale)
        dy = (self.height() - sh) / 2 + (self.pan_y * self.display_scale)
        
        return dx, dy, sw, sh

    def paintEvent(self, event):
        if self.image is None:
            return
            
        painter = QPainter(self)
        dx, dy, sw, sh = self.get_view_params()
        
        # Draw the stretched image
        target_rect = QRectF(dx, dy, sw, sh)
        painter.drawPixmap(target_rect, self.image, QRectF(self.image.rect()))

        # Draw Polygons
        pen_poly = QPen(Qt.green, 2)
        painter.setPen(pen_poly)
        
        def transform_point(pt):
            """Applies pan, scale, AND vertical stretch to a data point (x, y)."""
            tx = dx + pt[0] * self.display_scale
            # Data Y must be stretched by the ratio for display
            ty = dy + (pt[1] * self.pixel_aspect_ratio) * self.display_scale
            return QPointF(tx, ty)

        # Finished polygons
        for poly in self.polygons:
            if len(poly) > 1:
                pts = [transform_point(p) for p in poly]
                for i in range(len(pts)):
                    painter.drawLine(pts[i], pts[(i+1)%len(pts)])

        # Current drawing polygon
        pen_current = QPen(Qt.red, 2)
        painter.setPen(pen_current)
        if len(self.current_polygon) > 0:
            pts = [transform_point(p) for p in self.current_polygon]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i+1])
            # Draw circles at vertices
            for p in pts:
                painter.drawEllipse(p, 3, 3)

    def mousePressEvent(self, event):
        if self.image is None:
            return
            
        dx, dy, sw, sh = self.get_view_params()
        
        # Map mouse click back to raw data indices
        # We divide Y by the aspect ratio to 'un-stretch' the click back to the matrix index
        img_x = (event.pos().x() - dx) / self.display_scale
        img_y = ((event.pos().y() - dy) / self.display_scale) / self.pixel_aspect_ratio

        if event.button() == Qt.LeftButton:
            self.current_polygon.append((img_x, img_y))
        elif event.button() == Qt.RightButton:
            if self.current_polygon: 
                self.current_polygon.pop()
            elif self.polygons: 
                self.polygons.pop()
                self.regions_updated.emit(self.polygons)
        elif event.button() == Qt.MidButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MidButton:
            self.is_dragging = False
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x() / self.display_scale
            self.pan_y += delta.y() / self.display_scale
            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        # Zoom centered on mouse or simple zoom
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.display_scale *= factor
        self.update()

    def finalize_polygon(self):
        """Call this from a button in the GUI to close the current shape."""
        if len(self.current_polygon) > 2:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            self.regions_updated.emit(self.polygons)
            self.update()