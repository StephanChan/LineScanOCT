import os
import cv2
import numpy as np
import tifffile as tiff
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint, QEvent, QRectF
from shapely.geometry import Polygon

# Ensure SampleLocatorUI.py is in the same directory
from SampleLocatorUI import Ui_Form

class UnifiedSampleScanner(QDialog):
    def __init__(self, save_dir=r'D:\LineScanOCT', fov_w_mm=2.0, fov_h_mm=1.0, current_zpos = 0):
        super().__init__()
        self.save_dir = save_dir
        self.current_zpos = current_zpos
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.pic_window.setScaledContents(False)
        self.ui.pic_window.setAlignment(Qt.AlignCenter)

        # Microscope Parameters
        self.pixel_size_mm = 0.0474
        self.X_displacement = 60.5
        self.Y_displacement = 16.5
        self.fov_w_mm = fov_w_mm
        self.fov_h_mm = fov_h_mm

        # State Variables
        self.img_bgr = None
        self.qpixmap_raw = None 
        self.polygons = []         
        self.current_polygon = []   
        self.display_scale = 1.0    
        self.generated_locations = [] 
        self.sample_centers = []      
        self.is_finalized = False
        
        # Data to be returned to Main GUI
        self.final_raw_img = None
        self.final_polygons = []

        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.pan_x = 0 
        self.pan_y = 0
        
        self.img_bgr = self._capture_and_save_raw()
        self._prepare_pixmap()
        
        self.ui.pic_window.setMouseTracking(True)
        self.ui.pic_window.installEventFilter(self) 
        
        self.ui.finishwell.clicked.connect(self.complete_polygon)
        self.ui.finishplate.clicked.connect(self.process_generate_mosaic)

        QTimer.singleShot(100, self.reset_view)

    def _prepare_pixmap(self):
        """Converts CV2 image to QPixmap once to maintain high resolution and memory safety."""
        if self.img_bgr is None: return
        h, w = self.img_bgr.shape[:2]
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        # Using .copy() ensures Qt owns the memory, preventing kernel crashes
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.qpixmap_raw = QPixmap.fromImage(qimg)

    def _get_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _capture_and_save_raw(self):
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        frame_to_use = None
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                ret, frame = cap.read()
                if ret:
                    # Match hardware orientation
                    # frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)
                    frame_to_use = frame#[630:3670, 180:2160]
            
            if frame_to_use is None:
                frame_to_use = np.full((3840, 2160, 3), 40, dtype=np.uint8)

            ts = self._get_timestamp()
            tiff.imwrite(os.path.join(self.save_dir, f"RawCapture_{ts}.tif"), frame_to_use)
            return frame_to_use
        finally:
            cap.release()

    def reset_view(self):
        if self.img_bgr is None: return
        h, w = self.img_bgr.shape[:2]
        self.display_scale = min(self.ui.pic_window.width() / w, self.ui.pic_window.height() / h)
        self.pan_x, self.pan_y = 0, 0
        self.update_display()

    def update_display(self):
        """High-quality rendering using QPainter and Antialiasing."""
        if self.qpixmap_raw is None: return
        
        w, h = self.qpixmap_raw.width(), self.qpixmap_raw.height()
        sw, sh = int(w * self.display_scale), int(h * self.display_scale)
        
        final_buffer = QPixmap(self.ui.pic_window.size())
        final_buffer.fill(Qt.black)
        
        painter = QPainter(final_buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        dx = (self.ui.pic_window.width() - sw) / 2 + (self.pan_x * self.display_scale)
        dy = (self.ui.pic_window.height() - sh) / 2 + (self.pan_y * self.display_scale)
        
        # Draw Background
        painter.drawPixmap(int(dx), int(dy), sw, sh, self.qpixmap_raw)

        def to_ui(pt):
            return int(dx + pt[0] * self.display_scale), int(dy + pt[1] * self.display_scale)

        # Draw Polygons (Green)
        pen_poly = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen_poly)
        for poly in self.polygons:
            if len(poly) > 1:
                for i in range(len(poly)):
                    p1 = to_ui(poly[i])
                    p2 = to_ui(poly[(i+1)%len(poly)])
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # Draw Current drawing (Red)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for i in range(len(self.current_polygon)):
            p = to_ui(self.current_polygon[i])
            painter.drawEllipse(QPoint(p[0], p[1]), 3, 3)
            if i > 0:
                p_prev = to_ui(self.current_polygon[i-1])
                painter.drawLine(p_prev[0], p_prev[1], p[0], p[1])

        # Draw FOV Grid if generated (Yellow)
        if self.is_finalized:
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            for loc in self.generated_locations:
                print(loc)
                cx, cy = (loc['x']-self.X_displacement) / self.pixel_size_mm, (loc['y']-self.Y_displacement) / self.pixel_size_mm
                hw, hh = (self.fov_w_mm / 2) / self.pixel_size_mm, (self.fov_h_mm / 2) / self.pixel_size_mm
                tl = to_ui((cx - hw, cy - hh))
                br = to_ui((cx + hw, cy + hh))
                painter.drawRect(QRectF(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

        painter.end()
        self.ui.pic_window.setPixmap(final_buffer)

    def eventFilter(self, source, event):
        if source is self.ui.pic_window and self.img_bgr is not None:
            h, w = self.img_bgr.shape[:2]
            sw, sh = w * self.display_scale, h * self.display_scale
            dx = (self.ui.pic_window.width() - sw) / 2 + (self.pan_x * self.display_scale)
            dy = (self.ui.pic_window.height() - sh) / 2 + (self.pan_y * self.display_scale)

            if event.type() == QEvent.MouseButtonPress:
                img_x = (event.pos().x() - dx) / self.display_scale
                img_y = (event.pos().y() - dy) / self.display_scale

                if event.button() == Qt.LeftButton:
                    self.current_polygon.append((img_x, img_y))
                    self.update_display()
                    return True
                elif event.button() == Qt.RightButton:
                    if self.current_polygon: self.current_polygon.pop() 
                    elif self.polygons: self.polygons.pop()        
                    self.update_display()
                    return True
                elif event.button() == Qt.MidButton:
                    self.is_dragging = True
                    self.last_mouse_pos = event.pos()
                    return True
            elif event.type() == QEvent.MouseMove and self.is_dragging:
                delta = event.pos() - self.last_mouse_pos
                self.pan_x += delta.x() / self.display_scale
                self.pan_y += delta.y() / self.display_scale
                self.last_mouse_pos = event.pos()
                self.update_display()
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MidButton:
                self.is_dragging = False
                return True
            elif event.type() == QEvent.Wheel:
                self.display_scale *= (1.15 if event.angleDelta().y() > 0 else 0.85)
                self.update_display()
                return True
        return super().eventFilter(source, event)

    def complete_polygon(self):
        if len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            self.update_display()

    def process_generate_mosaic(self):
        if not self.polygons:
            QMessageBox.warning(self, "Error", "No regions drawn.")
            return

        # Explicitly copy data to return to Main GUI
        self.final_raw_img = np.copy(self.img_bgr)
        self.final_polygons = list(self.polygons)
        
        self.generated_locations = []
        self.sample_centers = []

        for idx, poly_pts in enumerate(self.polygons):
            sample_id = idx + 1 
            mm_poly_pts = [(p[0] * self.pixel_size_mm, p[1] * self.pixel_size_mm) for p in poly_pts]
            roi_poly = Polygon(mm_poly_pts)
            
            centroid = roi_poly.centroid
            self.sample_centers.append({'sample_id': sample_id, 'x': centroid.x + self.X_displacement, 'y': centroid.y + self.Y_displacement, 'z': self.current_zpos})

            min_x, min_y, max_x, max_y = roi_poly.bounds
            # x_centers = np.arange(min_x + self.fov_w_mm/2 - self.fov_w_mm, max_x + self.fov_w_mm, self.fov_w_mm)
            # y_centers = np.arange(min_y + self.fov_h_mm/2 - self.fov_h_mm, max_y + self.fov_h_mm, self.fov_h_mm)
            x_centers = np.arange(min_x, max_x + self.fov_w_mm/2, self.fov_w_mm)
            y_centers = np.arange(min_y, max_y + self.fov_w_mm/2, self.fov_h_mm)

            for cx in x_centers:
                for cy in y_centers:
                    tile_coords = [(cx-self.fov_w_mm/2, cy-self.fov_h_mm/2), (cx+self.fov_w_mm/2, cy-self.fov_h_mm/2),
                                   (cx+self.fov_w_mm/2, cy+self.fov_h_mm/2), (cx-self.fov_w_mm/2, cy+self.fov_h_mm/2)]
                    if Polygon(tile_coords).intersects(roi_poly):
                        self.generated_locations.append({'sample_id': sample_id, 'x': round(cx + self.X_displacement, 3), 'y': round(cy + self.Y_displacement, 3), 'z': self.current_zpos})

        self.is_finalized = True
        self.update_display()
        
        QMessageBox.information(self, "Success", f"Mosaic Generated.\n{len(self.sample_centers)} samples found.")
        self.accept()

if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    scanner = UnifiedSampleScanner()
    if scanner.exec_() == QDialog.Accepted:
        # Access results from scanner.generated_locations, scanner.final_raw_img, etc.
        print(f"Captured {len(scanner.generated_locations)} FOVs.")