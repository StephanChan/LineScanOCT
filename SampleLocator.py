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
from mosaic_scan_planner import (
    CENTER_MODE,
    FOV_OVERLAP,
    MAX_Y_FOV_MM,
    ROI_OCCUPANCY_TARGET,
    plan_mosaic_scan,
)

USB_PIXEL_SIZE_MM = 0.0474
USB_X_DISPLACEMENT_MM = 16.5
USB_Y_DISPLACEMENT_MM = 60.5
USB_CAMERA_INDEX = 0
USB_FRAME_WIDTH = 3840
USB_FRAME_HEIGHT = 2160
USB_AUTO_EXPOSURE = 0.25
USB_EXPOSURE = -5.0

def usb_image_to_stage(px, py, image_width):
    """
    Convert displayed USB-camera image coordinates to stage coordinates.

    The displayed image uses a top-right origin for the stage frame:
    image vertical -> stage X, image horizontal -> stage Y.
    """
    stage_x = py * USB_PIXEL_SIZE_MM + USB_X_DISPLACEMENT_MM
    stage_y = (image_width - px) * USB_PIXEL_SIZE_MM + USB_Y_DISPLACEMENT_MM
    return stage_x, stage_y

def stage_to_usb_image(stage_x, stage_y, image_width):
    """Inverse transform of usb_image_to_stage()."""
    px = image_width - ((stage_y - USB_Y_DISPLACEMENT_MM) / USB_PIXEL_SIZE_MM)
    py = (stage_x - USB_X_DISPLACEMENT_MM) / USB_PIXEL_SIZE_MM
    return px, py

def orient_usb_frame(frame):
    """Apply the display orientation used by the sample locator."""
    return cv2.flip(frame, 1)

def open_usb_camera(configure_exposure=False):
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_FRAME_HEIGHT)
    if configure_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, USB_AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, USB_EXPOSURE)
    return cap

def capture_usb_frame(configure_exposure=False):
    cap = open_usb_camera(configure_exposure=configure_exposure)
    try:
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        if not ret:
            return None
        return orient_usb_frame(frame)
    finally:
        cap.release()

def blank_usb_frame():
    return np.full((USB_FRAME_HEIGHT, USB_FRAME_WIDTH, 3), 40, dtype=np.uint8)

class UnifiedSampleScanner(QDialog):
    def __init__(
        self,
        save_dir=r'D:\LineScanOCT',
        fov_w_mm=2.0,
        fov_h_mm=1.0,
        current_zpos=0,
        y_step_um=10.0,
        stage_bounds=(0, 200, 0, 120),
    ):
        super().__init__()
        self.save_dir = save_dir
        self.current_zpos = current_zpos
        self.y_step_um = y_step_um
        self.stage_bounds = stage_bounds
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.pic_window.setScaledContents(False)
        self.ui.pic_window.setAlignment(Qt.AlignCenter)

        # Microscope Parameters
        self.pixel_size_mm = USB_PIXEL_SIZE_MM
        self.X_displacement = USB_X_DISPLACEMENT_MM
        self.Y_displacement = USB_Y_DISPLACEMENT_MM
        self.fov_w_mm = fov_w_mm
        self.fov_h_mm = fov_h_mm
        self.roi_occupancy = ROI_OCCUPANCY_TARGET
        self.fov_overlap = FOV_OVERLAP
        self.max_y_fov_mm = MAX_Y_FOV_MM
        self.center_mode = CENTER_MODE
        self.debug_scan_planner = False

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

    def image_to_stage(self, px, py):
        """
        Convert displayed USB-camera image coordinates to stage coordinates.

        The displayed image uses a top-right origin for the stage frame:
        image vertical -> stage X, image horizontal -> stage Y.
        """
        if self.img_bgr is None:
            return 0.0, 0.0
        _, image_w = self.img_bgr.shape[:2]
        return usb_image_to_stage(px, py, image_w)

    def stage_to_image(self, stage_x, stage_y):
        """Inverse transform of image_to_stage()."""
        if self.img_bgr is None:
            return 0.0, 0.0
        _, image_w = self.img_bgr.shape[:2]
        return stage_to_usb_image(stage_x, stage_y, image_w)

    def _prepare_pixmap(self):
        """Converts CV2 image to QPixmap once to maintain high resolution and memory safety."""
        if self.img_bgr is None: return
        h, w = self.img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        # Using .copy() ensures Qt owns the memory, preventing kernel crashes
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        self.qpixmap_raw = QPixmap.fromImage(qimg)

    def _get_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _capture_and_save_raw(self):
        frame_to_use = capture_usb_frame()
        if frame_to_use is None:
            frame_to_use = blank_usb_frame()

        ts = self._get_timestamp()
        tiff.imwrite(os.path.join(self.save_dir, f"RawCapture_{ts}.tif"), frame_to_use)
        return frame_to_use

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
                cx, cy = self.stage_to_image(loc['x'], loc['y'])
                loc_y_fov = loc.get('y_length_mm', self.fov_h_mm)
                h_half = (self.fov_w_mm / 2) / self.pixel_size_mm
                w_half = (loc_y_fov / 2) / self.pixel_size_mm
                tl = to_ui((cx - w_half, cy - h_half))
                br = to_ui((cx + w_half, cy + h_half))
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
            stage_poly_pts = [self.image_to_stage(p[0], p[1]) for p in poly_pts]
            scan_plan = plan_mosaic_scan(
                sample_id=sample_id,
                mm_polygons=[stage_poly_pts],
                x_fov_mm=self.fov_w_mm,
                y_step_um=self.y_step_um,
                stage_bounds=self.stage_bounds,
                occupancy=self.roi_occupancy,
                overlap=self.fov_overlap,
                max_y_fov_mm=self.max_y_fov_mm,
                center_mode=self.center_mode,
            )
            self.sample_centers.append(
                {
                    'sample_id': sample_id,
                    'x': scan_plan.center_x,
                    'y': scan_plan.center_y,
                    'z': self.current_zpos,
                }
            )
            for loc in scan_plan.fov_locations:
                self.generated_locations.append(
                    {
                        'sample_id': sample_id,
                        'x': loc['x'],
                        'y': loc['y'],
                        'z': self.current_zpos,
                        'y_length_mm': scan_plan.y_length_mm,
                        'y_pixels': scan_plan.y_pixels,
                    }
                )
            if self.debug_scan_planner:
                print(
                    "USB sample scan plan: "
                    f"sample_id={sample_id}, center=({scan_plan.center_x:.3f}, {scan_plan.center_y:.3f}), "
                    f"roi_size=({scan_plan.roi_size[0]:.3f}, {scan_plan.roi_size[1]:.3f}), "
                    f"tile_count={scan_plan.tile_count}, accepted={len(scan_plan.fov_locations)}, "
                    f"planned_YLength={scan_plan.y_length_mm:.3f}, planned_Ypixels={scan_plan.y_pixels}"
                )

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
