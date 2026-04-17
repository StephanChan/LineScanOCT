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
        self.debug_coordinates = False
        self.display_orientation = "usb_top_view"
        self.raw_shape = None

    def set_image(self, numpy_array, m, M, pixel_size_x=1.0, pixel_size_y=1.0):
        """
        Receives the stitched mosaic and the physical resolution.
        pixel_size_x: step size in mm or um for the horizontal axis
        pixel_size_y: step size in mm or um for the vertical axis
        """
        self.m_min, self.m_max = m, M

        # 1. Normalize intensity (Contrast Stretching)
        adj = ((numpy_array - m) / (M - m + 1e-5) * 255)
        adj = np.ascontiguousarray(np.clip(adj, 0, 255).astype(np.uint8))
        self.adj = adj # Keep raw mosaic coordinates for correction/stitching logic.
        self.raw_shape = adj.shape[:2]

        # Display-only convention for matching the USB top-view image:
        # raw mosaic: rows = stage Y, columns = stage X
        # USB view: vertical = stage X, horizontal = stage Y, right = smaller stage Y
        # display = fliplr(raw.T). For future saved-data stitching, apply the same
        # raw/display conversion explicitly rather than changing the acquisition data.
        display_adj = self.raw_to_display_array(adj)

        # 2. Calculate aspect ratio for physical pixel stretching.
        # In USB top-view display, horizontal pixels are raw Y pixels and vertical
        # pixels are raw X pixels, so the visual vertical/horizontal ratio is X/Y.
        if self.display_orientation == "usb_top_view":
            if pixel_size_y != 0:
                self.pixel_aspect_ratio = pixel_size_x / pixel_size_y
            else:
                self.pixel_aspect_ratio = 1.0
        elif pixel_size_x != 0:
            self.pixel_aspect_ratio = pixel_size_y / pixel_size_x
        else:
            self.pixel_aspect_ratio = 1.0
        
        # 3. Determine if image is Grayscale (2D) or RGB (3D)
        if display_adj.ndim == 2:
            # Grayscale processing
            h, w = display_adj.shape
            # QImage format for 8-bit grayscale
            qt_image = QImage(display_adj.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
        elif display_adj.ndim == 3 and display_adj.shape[2] == 3:
            # RGB processing
            h, w, ch = display_adj.shape
            bytes_per_line = ch * w
            # QImage format for 24-bit RGB
            # Note: Ensure adj is in RGB order. If it's BGR (from OpenCV), 
            # use Format_BGR888 or swap channels first.
            qt_image = QImage(display_adj.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            raise ValueError(f"Unsupported image dimensions: {display_adj.shape}")

        # 4. Handle Physical Pixel Size Stretching (Aspect Ratio)
        # We stretch the image physically so that pixels appear square in the UI
        if self.debug_coordinates:
            print(
                "Mosaic viewer set_image: "
                f"raw_shape={adj.shape}, pixel_size_x={pixel_size_x}, "
                f"pixel_size_y={pixel_size_y}, aspect={self.pixel_aspect_ratio:.6g}, "
                f"orientation={self.display_orientation}, display_shape={display_adj.shape}, "
                f"polygons_preserved={len(self.polygons)}, current_vertices={len(self.current_polygon)}"
            )
        if self.pixel_aspect_ratio != 1.0:
            new_h = int(h * self.pixel_aspect_ratio)
            if self.debug_coordinates:
                print(f"Mosaic viewer visual stretch: display_w={w}, display_h={h}, display_h_if_scaled={new_h}")
            self.image = QPixmap.fromImage(qt_image).scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        else:
            self.image = QPixmap.fromImage(qt_image)

        self.update()

    def raw_to_display_array(self, arr):
        if self.display_orientation != "usb_top_view":
            return arr
        if arr.ndim == 2:
            return np.ascontiguousarray(np.fliplr(arr.T))
        if arr.ndim == 3:
            return np.ascontiguousarray(np.fliplr(np.transpose(arr, (1, 0, 2))))
        return arr

    def raw_to_display_point(self, pt):
        if self.display_orientation != "usb_top_view" or self.raw_shape is None:
            return pt
        raw_h, _ = self.raw_shape
        raw_x, raw_y = pt
        display_x = (raw_h - 1) - raw_y
        display_y = raw_x
        return display_x, display_y

    def display_to_raw_point(self, pt):
        if self.display_orientation != "usb_top_view" or self.raw_shape is None:
            return pt
        raw_h, _ = self.raw_shape
        display_x, display_y = pt
        raw_x = display_y
        raw_y = (raw_h - 1) - display_x
        return raw_x, raw_y

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
            display_pt = self.raw_to_display_point(pt)
            tx = dx + display_pt[0] * self.display_scale
            # Data Y must be stretched by the ratio for display
            ty = dy + (display_pt[1] * self.pixel_aspect_ratio) * self.display_scale
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
        display_x = (event.pos().x() - dx) / self.display_scale
        display_y = ((event.pos().y() - dy) / self.display_scale) / self.pixel_aspect_ratio
        img_x, img_y = self.display_to_raw_point((display_x, display_y))

        if event.button() == Qt.LeftButton:
            self.current_polygon.append((img_x, img_y))
            if self.debug_coordinates:
                print(
                    "Mosaic viewer click: "
                    f"widget=({event.pos().x():.1f}, {event.pos().y():.1f}), "
                    f"view_origin=({dx:.2f}, {dy:.2f}), "
                    f"view_size=({sw:.2f}, {sh:.2f}), "
                    f"display_scale={self.display_scale:.6g}, "
                    f"aspect={self.pixel_aspect_ratio:.6g}, "
                    f"display=({display_x:.2f}, {display_y:.2f}), "
                    f"raw=({img_x:.2f}, {img_y:.2f})"
                )
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
        if len(self.current_polygon) > 3:
            self.polygons.append(list(self.current_polygon))
            if self.debug_coordinates:
                pts = np.array(self.current_polygon, dtype=float)
                xmin, ymin = np.min(pts, axis=0)
                xmax, ymax = np.max(pts, axis=0)
                print(
                    "Mosaic viewer finalized polygon: "
                    f"vertices={len(self.current_polygon)}, "
                    f"raw_bounds=(x:{xmin:.2f}-{xmax:.2f}, y:{ymin:.2f}-{ymax:.2f}), "
                    f"raw_size=({xmax - xmin:.2f}, {ymax - ymin:.2f}), "
                    f"aspect={self.pixel_aspect_ratio:.6g}"
                )
            self.current_polygon = []
            self.regions_updated.emit(self.polygons)
            self.update()

    def clear_polygons(self):
        self.polygons = []
        self.current_polygon = []
        self.regions_updated.emit(self.polygons)
        self.update()

    def clear_image(self, clear_polygons=True):
        self.image = None
        self.adj = None
        self.raw_shape = None
        if clear_polygons:
            self.polygons = []
            self.current_polygon = []
            self.regions_updated.emit(self.polygons)
        self.update()
