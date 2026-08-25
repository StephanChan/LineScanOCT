import os
import cv2
import json
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets as QW
from PyQt5.QtWidgets import QMessageBox, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint, QEvent, QRectF

# Ensure SampleLocatorUI.py is in the same directory
from SampleLocatorUI import Ui_Form
from mosaic_scan_planner import (
    CENTER_MODE,
    FOV_OVERLAP,
    ROI_OCCUPANCY_TARGET,
    plan_mosaic_scan,
)
from ScanModels import FOVLocation, SampleCenter

USB_CAMERA_INDEX = 0
USB_FRAME_WIDTH = 3840
USB_FRAME_HEIGHT = 2160
USB_AUTO_EXPOSURE = 0.25
USB_EXPOSURE = -5.0
USB_MOSAIC_X_MIN_MM = 90.0
USB_MOSAIC_X_MAX_MM = 135.0
USB_MOSAIC_Y_MIN_MM = 50.0
USB_MOSAIC_Y_MAX_MM = 130.0
USB_MOSAIC_GRID_X = 3
USB_MOSAIC_GRID_Y = 4
USB_VALID_CENTER_REGION_PX = 1500
SAMPLE_STAGE_COLUMN_Y_TOLERANCE_MM = 10.0

USB_CAMERA_TO_STAGE_AFFINE = [
    [-9.61990609461207e-05, 0.046074609344122719, -5.3744508746854729],
    [-0.046323119248083452, 2.7853843648006915e-05, 196.4359168459153],
    [0.0, 0.0, 1.0],
]

USB_CAMERA_STAGE_CONTROL_POINTS = [
    {"camera": [672.0, 240.0], "stage": [5.92, 165.1]},
    {"camera": [658.0, 1408.0], "stage": [59.33, 165.9]},
    {"camera": [725.0, 1708.0], "stage": [73.17, 162.72]},
    {"camera": [977.0, 279.0], "stage": [7.31, 151.16]},
    {"camera": [1010.0, 604.0], "stage": [22.37, 149.6]},
    {"camera": [1020.0, 1078.0], "stage": [44.27, 149.27]},
    {"camera": [1526.0, 887.0], "stage": [35.1, 125.9]},
    {"camera": [1291.0, 888.0], "stage": [35.38, 136.8]},
    {"camera": [1546.0, 1116.0], "stage": [45.9, 124.9]},
    {"camera": [1357.0, 1435.0], "stage": [60.6, 133.84]},
    {"camera": [1558.0, 1691.0], "stage": [72.63, 124.55]},
    {"camera": [1789.0, 258.0], "stage": [6.33, 113.7]},
    {"camera": [1782.0, 1157.0], "stage": [47.7, 114.0]},
    {"camera": [2099.0, 840.0], "stage": [32.75, 99.24]},
    {"camera": [2161.0, 1669.0], "stage": [71.63, 96.32]},
    {"camera": [2500.0, 580.0], "stage": [21, 80.54]},
    {"camera": [2659.0, 838.0], "stage": [32.64, 73.17]},
    {"camera": [2381.0, 1100.0], "stage": [45.1, 86]},
    {"camera": [2615.0, 1660.0], "stage": [70.92, 75.19]},
    {"camera": [2763.0, 215.0], "stage": [4.69, 68.58]},
]


def default_usb_mosaic_calibration():
    return {
        "calibration_method": "affine_matrix",
        "camera_to_stage_affine": USB_CAMERA_TO_STAGE_AFFINE,
        "camera_stage_control_points": USB_CAMERA_STAGE_CONTROL_POINTS,
        "affine_pixel_space": "fiji_horizontal_flip",
    }

def calibration_uses_affine(calibration):
    return (
        calibration.get("calibration_method") == "affine_matrix"
        and calibration.get("camera_to_stage_affine") is not None
    )

def display_pixel_to_affine_pixel(calibration, px, py):
    if calibration.get("affine_pixel_space") == "raw_camera_fiji":
        return float(USB_FRAME_WIDTH - 1 - float(py)), float(USB_FRAME_HEIGHT - 1 - float(px))
    return float(px), float(py)

def affine_pixel_to_display_pixel(calibration, px, py):
    if calibration.get("affine_pixel_space") == "raw_camera_fiji":
        return float(USB_FRAME_HEIGHT - 1 - float(py)), float(USB_FRAME_WIDTH - 1 - float(px))
    return float(px), float(py)

def image_to_stage_from_calibration(calibration, px, py):
    matrix = np.asarray(calibration["camera_to_stage_affine"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_to_stage_affine must be 3x3, got {matrix.shape}")
    affine_px, affine_py = display_pixel_to_affine_pixel(calibration, px, py)
    point = matrix @ np.asarray([affine_px, affine_py, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-12:
        raise ValueError(f"Invalid affine transform output for pixel ({affine_px}, {affine_py}): {point}")
    return float(point[0] / point[2]), float(point[1] / point[2])

def stage_to_image_from_calibration(calibration, stage_x, stage_y):
    matrix = np.asarray(calibration["camera_to_stage_affine"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_to_stage_affine must be 3x3, got {matrix.shape}")
    inv_matrix = np.linalg.inv(matrix)
    point = inv_matrix @ np.asarray([float(stage_x), float(stage_y), 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-12:
        raise ValueError(f"Invalid inverse affine output for stage ({stage_x}, {stage_y}): {point}")
    affine_px = float(point[0] / point[2])
    affine_py = float(point[1] / point[2])
    return affine_pixel_to_display_pixel(calibration, affine_px, affine_py)

def affine_fov_half_size_pixels(calibration, x_length_mm, y_length_mm):
    matrix = np.asarray(calibration["camera_to_stage_affine"], dtype=np.float64)
    inv_linear = np.linalg.inv(matrix[:2, :2])
    x_half = float(x_length_mm) / 2.0
    y_half = float(y_length_mm) / 2.0
    corners = np.asarray(
        [
            [x_half, y_half],
            [x_half, -y_half],
            [-x_half, y_half],
            [-x_half, -y_half],
        ],
        dtype=np.float64,
    )
    pixel_offsets = corners @ inv_linear.T
    if calibration.get("affine_pixel_space") == "raw_camera_fiji":
        pixel_offsets = np.column_stack((-pixel_offsets[:, 1], -pixel_offsets[:, 0]))
    return float(np.max(np.abs(pixel_offsets[:, 0]))), float(np.max(np.abs(pixel_offsets[:, 1])))

def sort_sample_entries_by_stage_columns(entries, y_tolerance_mm=SAMPLE_STAGE_COLUMN_Y_TOLERANCE_MM):
    columns = []
    for entry in sorted(entries, key=lambda item: (-float(item["center_y"]), -float(item["center_x"]))):
        best_column = None
        best_distance = None
        for column in columns:
            distance = abs(float(entry["center_y"]) - float(column["mean_y"]))
            if distance <= float(y_tolerance_mm) and (
                best_distance is None or distance < best_distance
            ):
                best_column = column
                best_distance = distance
        if best_column is None:
            columns.append({"mean_y": float(entry["center_y"]), "entries": [entry]})
        else:
            best_column["entries"].append(entry)
            best_column["mean_y"] = float(
                np.mean([float(item["center_y"]) for item in best_column["entries"]])
            )

    ordered_entries = []
    for column in sorted(columns, key=lambda item: -float(item["mean_y"])):
        ordered_entries.extend(
            sorted(
                column["entries"],
                key=lambda item: (float(item["center_x"]), -float(item["center_y"])),
            )
        )
    return ordered_entries


def orient_usb_frame(frame):
    """Apply the horizontal flip used for Fiji calibration."""
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
    return orient_usb_frame(np.full((USB_FRAME_HEIGHT, USB_FRAME_WIDTH, 3), 40, dtype=np.uint8))

class _SampleLocatorDrawingBase(QDialog):
    def _prepare_pixmap(self):
        """Converts CV2 image to QPixmap once to maintain high resolution and memory safety."""
        if self.img_bgr is None: return
        h, w = self.img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        # Using .copy() ensures Qt owns the memory, preventing kernel crashes
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        self.qpixmap_raw = QPixmap.fromImage(qimg)

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

        valid_region = self.valid_roi_rect()
        if valid_region is not None:
            painter.setPen(QPen(QColor(0, 180, 255), 3))
            x1, y1, x2, y2 = valid_region
            tl = to_ui((x1, y1))
            br = to_ui((x2, y2))
            painter.drawRect(QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]))

        # Draw FOV Grid if generated (Yellow)
        if self.is_finalized:
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            for loc in self.fov_locations_for_current_view():
                cx, cy = self.stage_to_image(loc.x, loc.y)
                w_half, h_half = self.fov_half_size_pixels(loc)
                tl = to_ui((cx - w_half, cy - h_half))
                br = to_ui((cx + w_half, cy + h_half))
                painter.drawRect(QRectF(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

        painter.end()
        self.ui.pic_window.setPixmap(final_buffer)

    def valid_roi_rect(self):
        if not isinstance(self, MosaicUSBSampleScanner):
            return None
        if self.current_tile().get("single_frame", False):
            return None
        if self.img_bgr is None:
            return None
        h, w = self.img_bgr.shape[:2]
        size = min(int(USB_VALID_CENTER_REGION_PX), int(w), int(h))
        x1 = (w - size) / 2.0
        y1 = (h - size) / 2.0
        return x1, y1, x1 + size, y1 + size

    def polygon_inside_valid_region(self, polygon):
        valid_region = self.valid_roi_rect()
        if valid_region is None:
            return True
        x1, y1, x2, y2 = valid_region
        for px, py in polygon:
            if px < x1 or px > x2 or py < y1 or py > y2:
                return False
        return True

    def fov_half_size_pixels(self, loc):
        raise NotImplementedError("Sample locator FOV pixel size requires an affine calibration.")

    def fov_locations_for_current_view(self):
        return self.generated_locations

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
            if not self.polygon_inside_valid_region(self.current_polygon):
                QMessageBox.warning(
                    self,
                    "ROI outside center region",
                    "This ROI has points outside the blue 1500x1500 center region. "
                    "Please redraw it inside the guide rectangle.",
                )
                return
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            self.update_display()

class MosaicUSBSampleScanner(_SampleLocatorDrawingBase):
    def __init__(
        self,
        tile_records,
        save_dir=r"D:\LineScanOCT",
        fov_w_mm=2.0,
        fov_h_mm=1.0,
        current_zpos=0,
        y_step_um=10.0,
        stage_bounds=(0, 200, 0, 120),
        max_y_fov_mm=0.25,
        sample_id_start=1,
        allow_empty=False,
        initial_tile_index=0,
        initial_roi_records=None,
        initial_calibration=None,
    ):
        QDialog.__init__(self)
        if not tile_records:
            raise ValueError("MosaicUSBSampleScanner requires at least one tile image.")
        self.tile_records = tile_records
        self.save_dir = save_dir
        self.current_zpos = current_zpos
        self.y_step_um = y_step_um
        self.stage_bounds = stage_bounds
        self.fov_w_mm = fov_w_mm
        self.fov_h_mm = fov_h_mm
        self.roi_occupancy = ROI_OCCUPANCY_TARGET
        self.fov_overlap = FOV_OVERLAP
        self.max_y_fov_mm = float(max_y_fov_mm)
        self.center_mode = CENTER_MODE
        self.debug_scan_planner = False
        self.sample_id_start = int(sample_id_start)
        self.allow_empty = bool(allow_empty)
        self.calibration = default_usb_mosaic_calibration()
        if initial_calibration is not None:
            for key in self.calibration:
                if key in initial_calibration:
                    value = initial_calibration[key]
                    if isinstance(value, list):
                        self.calibration[key] = [
                            dict(item) if isinstance(item, dict) else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        self.calibration[key] = dict(value)
                    elif isinstance(value, bool):
                        self.calibration[key] = bool(value)
                    elif isinstance(value, str):
                        self.calibration[key] = value
                    else:
                        self.calibration[key] = float(value)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Mosaic USB Sample Locator")
        self.ui.pic_window.setScaledContents(False)
        self.ui.pic_window.setAlignment(Qt.AlignCenter)
        self.ui.label.setText("Left click draw ROI. Right click undo. Middle drag pan. Wheel zoom.")
        self.ui.label_2.setText("Draw ROIs near each region center, then move to next region.")
        self.ui.finishwell.setText("finish ROI in this region")
        self.ui.finishplate.setText("finish this region")

        self._add_mosaic_controls()

        self.current_tile_index = 0
        self.tile_polygons = [[] for _ in self.tile_records]
        self.tile_current_polygons = [[] for _ in self.tile_records]
        self.initial_roi_sample_ids = {}
        self.tile_loaded = False
        self.img_bgr = None
        self.qpixmap_raw = None
        self.polygons = []
        self.current_polygon = []
        self.display_scale = 1.0
        self.generated_locations = []
        self.sample_centers = []
        self.final_raw_img = None
        self.final_polygons = []
        self.final_tile_roi_records = []
        self.preview_ready = False
        self.is_finalized = False
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.pan_x = 0
        self.pan_y = 0

        self._load_initial_roi_records(initial_roi_records)
        if self._rebuild_mosaic_outputs():
            print(
                "USB locator restored previous selections: "
                f"{len(self.final_tile_roi_records)} ROI(s), "
                f"{len(self.generated_locations)} FOV(s)."
            )

        self.ui.pic_window.setMouseTracking(True)
        self.ui.pic_window.installEventFilter(self)
        self.ui.finishwell.clicked.connect(self.complete_polygon)
        self.ui.finishplate.clicked.connect(self.process_generate_mosaic)

        self.load_tile(initial_tile_index)
        QTimer.singleShot(100, self.reset_view)

    def _add_mosaic_controls(self):
        self.tile_label = QW.QLabel()
        self.ui.verticalLayout.addWidget(self.tile_label)

        nav_layout = QW.QHBoxLayout()
        self.prev_tile_button = QW.QPushButton("previous region")
        self.next_tile_button = QW.QPushButton("next region")
        nav_layout.addWidget(self.prev_tile_button)
        nav_layout.addWidget(self.next_tile_button)
        self.ui.verticalLayout.addLayout(nav_layout)
        self.prev_tile_button.clicked.connect(self.previous_tile)
        self.next_tile_button.clicked.connect(self.next_tile)

        self.ui.verticalLayout.addWidget(
            QW.QLabel(
                "USB calibration from code: "
                "Fiji horizontal-flip affine matrix"
            )
        )

    def _clear_fov_preview(self):
        self.generated_locations = []
        self.sample_centers = []
        self.final_polygons = []
        self.final_tile_roi_records = []
        self.is_finalized = False
        self.preview_ready = False
        if hasattr(self, "ui"):
            self.ui.finishplate.setText("finish this region")

    def _load_initial_roi_records(self, initial_roi_records):
        if not initial_roi_records:
            return
        loaded_count = 0
        for record in initial_roi_records:
            tile_index = int(record.get("tile_index", 0)) - 1
            if 0 <= tile_index < len(self.tile_polygons):
                polygon = record.get("pixel_polygon", [])
                if len(polygon) >= 3:
                    polygon_pts = [(float(x), float(y)) for x, y in polygon]
                    self.tile_polygons[tile_index].append(polygon_pts)
                    self.initial_roi_sample_ids[
                        self._polygon_key(tile_index, polygon_pts)
                    ] = int(record["sample_id"])
                    loaded_count += 1
        print(f"USB locator loaded previous ROI records: {loaded_count}/{len(initial_roi_records)}")

    def _polygon_key(self, tile_index, polygon):
        rounded_polygon = tuple(
            (round(float(px), 3), round(float(py), 3))
            for px, py in polygon
        )
        return int(tile_index), rounded_polygon

    def _completed_tile_polygons(self):
        completed = []
        for tile_index, polygons in enumerate(self.tile_polygons):
            for poly in polygons:
                if len(poly) >= 3:
                    completed.append((tile_index, poly))
        return completed

    def _rebuild_mosaic_outputs(self):
        all_polygons = self._completed_tile_polygons()
        self.generated_locations = []
        self.sample_centers = []
        self.final_polygons = []
        self.final_tile_roi_records = []
        self.final_raw_img = None
        if not all_polygons:
            return False

        active_tile_index = self.current_tile_index
        sample_entries = []
        for tile_index, poly_pts in all_polygons:
            self.current_tile_index = tile_index
            fiji_poly_pts = [
                display_pixel_to_affine_pixel(self.calibration, p[0], p[1])
                for p in poly_pts
            ]
            stage_poly_pts = [self.image_to_stage(p[0], p[1]) for p in poly_pts]
            scan_plan = plan_mosaic_scan(
                sample_id=0,
                mm_polygons=[stage_poly_pts],
                x_fov_mm=self.fov_w_mm,
                y_step_um=self.y_step_um,
                stage_bounds=self.stage_bounds,
                occupancy=self.roi_occupancy,
                overlap=self.fov_overlap,
                max_y_fov_mm=self.max_y_fov_mm,
                center_mode=self.center_mode,
            )
            sample_entries.append(
                {
                    "tile_index": tile_index,
                    "poly_pts": poly_pts,
                    "fiji_poly_pts": fiji_poly_pts,
                    "stage_poly_pts": stage_poly_pts,
                    "center_x": float(scan_plan.center_x),
                    "center_y": float(scan_plan.center_y),
                }
            )

        sample_entries = sort_sample_entries_by_stage_columns(sample_entries)
        self.initial_roi_sample_ids = {}
        tile_roi_records = []
        for idx, entry in enumerate(sample_entries):
            sample_id = self.sample_id_start + idx
            tile_index = int(entry["tile_index"])
            poly_pts = entry["poly_pts"]
            fiji_poly_pts = entry["fiji_poly_pts"]
            stage_poly_pts = entry["stage_poly_pts"]
            polygon_key = self._polygon_key(tile_index, poly_pts)
            self.initial_roi_sample_ids[polygon_key] = sample_id

            self.current_tile_index = tile_index
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
                SampleCenter(
                    sample_id=sample_id,
                    x=scan_plan.center_x,
                    y=scan_plan.center_y,
                    z=self.current_zpos,
                )
            )
            for loc in scan_plan.fov_locations:
                self.generated_locations.append(
                    FOVLocation(
                        sample_id=sample_id,
                        x=loc.x,
                        y=loc.y,
                        z=self.current_zpos,
                        y_length_mm=scan_plan.y_length_mm,
                    )
                )
            self.final_polygons.append(list(poly_pts))
            tile = self.tile_records[tile_index]
            tile_roi_records.append(
                {
                    "sample_id": sample_id,
                    "tile_index": tile_index + 1,
                    "tile_stage_x": float(tile["stage_x"]),
                    "tile_stage_y": float(tile["stage_y"]),
                    "tile_stage_z": float(tile["stage_z"]),
                    "pixel_polygon": [[float(x), float(y)] for x, y in poly_pts],
                    "fiji_pixel_polygon": [[float(x), float(y)] for x, y in fiji_poly_pts],
                    "stage_polygon": [[float(x), float(y)] for x, y in stage_poly_pts],
                    "center_x": float(scan_plan.center_x),
                    "center_y": float(scan_plan.center_y),
                }
            )

        self.current_tile_index = active_tile_index
        self.final_tile_roi_records = tile_roi_records
        print(
            "Sample IDs sorted by motor stage columns: "
            f"Y tolerance={SAMPLE_STAGE_COLUMN_Y_TOLERANCE_MM:.1f} mm, "
            "columns high-to-low Y, samples low-to-high X within each column."
        )
        return True

    def current_tile(self):
        return self.tile_records[self.current_tile_index]

    def _save_current_tile_state(self):
        self.tile_polygons[self.current_tile_index] = [list(poly) for poly in self.polygons]
        self.tile_current_polygons[self.current_tile_index] = list(self.current_polygon)

    def load_tile(self, index):
        if self.tile_loaded:
            self._save_current_tile_state()
        self.current_tile_index = max(0, min(int(index), len(self.tile_records) - 1))
        tile = self.current_tile()
        self.img_bgr = tile["image"]
        self.polygons = [list(poly) for poly in self.tile_polygons[self.current_tile_index]]
        self.current_polygon = list(self.tile_current_polygons[self.current_tile_index])
        self.tile_loaded = True
        self._prepare_pixmap()
        self.reset_view()
        self.tile_label.setText(
            f"Region {self.current_tile_index + 1}/{len(self.tile_records)} "
            f"stage X={tile['stage_x']:.3f}, Y={tile['stage_y']:.3f}, Z={tile['stage_z']:.3f}"
        )

    def previous_tile(self):
        if self.current_tile_index > 0:
            self.load_tile(self.current_tile_index - 1)

    def next_tile(self):
        if self.current_tile_index < len(self.tile_records) - 1:
            self.load_tile(self.current_tile_index + 1)

    def image_to_stage(self, px, py):
        if not calibration_uses_affine(self.calibration):
            raise ValueError(f"USB locator requires affine calibration: {self.calibration}")
        return image_to_stage_from_calibration(self.calibration, px, py)

    def stage_to_image_for_tile(self, stage_x, stage_y, tile):
        if not calibration_uses_affine(self.calibration):
            raise ValueError(f"USB locator requires affine calibration: {self.calibration}")
        return stage_to_image_from_calibration(self.calibration, stage_x, stage_y)

    def stage_to_image(self, stage_x, stage_y):
        return self.stage_to_image_for_tile(stage_x, stage_y, self.current_tile())

    def fov_half_size_pixels(self, loc):
        loc_y_fov = loc.y_length_mm if loc.y_length_mm is not None else self.fov_h_mm
        if not calibration_uses_affine(self.calibration):
            raise ValueError(f"USB locator requires affine calibration: {self.calibration}")
        return affine_fov_half_size_pixels(self.calibration, self.fov_w_mm, loc_y_fov)

    def fov_locations_for_current_view(self):
        current_tile_number = self.current_tile_index + 1
        current_sample_ids = {
            int(record["sample_id"])
            for record in self.final_tile_roi_records
            if int(record["tile_index"]) == current_tile_number
        }
        if not current_sample_ids:
            return []
        return [
            loc
            for loc in self.generated_locations
            if int(loc.sample_id) in current_sample_ids
        ]

    def _prepare_pixmap(self):
        return _SampleLocatorDrawingBase._prepare_pixmap(self)

    def reset_view(self):
        return _SampleLocatorDrawingBase.reset_view(self)

    def update_display(self):
        return _SampleLocatorDrawingBase.update_display(self)

    def eventFilter(self, source, event):
        return _SampleLocatorDrawingBase.eventFilter(self, source, event)

    def complete_polygon(self):
        if getattr(self, "preview_ready", False):
            self._clear_fov_preview()
        _SampleLocatorDrawingBase.complete_polygon(self)
        self._save_current_tile_state()

    def process_generate_mosaic(self):
        if getattr(self, "preview_ready", False):
            self._save_locator_records()
            self.accept()
            return

        if len(self.current_polygon) >= 3:
            if not self.polygon_inside_valid_region(self.current_polygon):
                QMessageBox.warning(
                    self,
                    "ROI outside center region",
                    "This ROI has points outside the blue 1500x1500 center region. "
                    "Please redraw it inside the guide rectangle.",
                )
                return
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
        elif len(self.current_polygon) > 0:
            QMessageBox.warning(
                self,
                "Incomplete ROI",
                "The current ROI has fewer than 3 points. Finish or clear it before finishing this region.",
            )
            return

        self._save_current_tile_state()
        if not self._completed_tile_polygons():
            if not self.allow_empty:
                QMessageBox.warning(self, "Error", "No regions drawn.")
                return
            self.generated_locations = []
            self.sample_centers = []
            self.final_polygons = []
            self.final_tile_roi_records = []
            self.final_raw_img = None
            self.accept()
            return

        self._rebuild_mosaic_outputs()
        self.is_finalized = True
        self.preview_ready = True
        current_tile_number = self.current_tile_index + 1
        current_region_records = [
            record
            for record in self.final_tile_roi_records
            if int(record["tile_index"]) == current_tile_number
        ]
        tile = self.current_tile()
        print(
            "USB locator region finished: "
            f"region={current_tile_number}/{len(self.tile_records)}, "
            f"row={int(tile['row']) + 1}, col={int(tile['col']) + 1}, "
            f"stage X={float(tile['stage_x']):.4f}, Y={float(tile['stage_y']):.4f}, "
            "calibration=Fiji horizontal-flip affine matrix"
        )
        if not current_region_records:
            print("USB locator region finished with no sample centers.")
        else:
            print("USB locator sample centers from this region:")
            for record in current_region_records:
                print(
                    f"sampleID-{int(record['sample_id'])}: "
                    f"X={float(record['center_x']):.4f}, "
                    f"Y={float(record['center_y']):.4f}, "
                    f"Z={float(self.current_zpos):.4f}"
                )
                print("  ROI coordinate mapping: GUI display px == Fiji horizontal-flip px -> stage mm")
                display_polygon = record.get("pixel_polygon", [])
                fiji_polygon = record.get("fiji_pixel_polygon", [])
                stage_polygon = record.get("stage_polygon", [])
                for idx, (display_pt, fiji_pt, stage_pt) in enumerate(
                    zip(display_polygon, fiji_polygon, stage_polygon),
                    start=1,
                ):
                    print(
                        "    "
                        f"P{idx}: "
                        f"GUI=({float(display_pt[0]):.2f}, {float(display_pt[1]):.2f}) "
                        f"Fiji=({float(fiji_pt[0]):.2f}, {float(fiji_pt[1]):.2f}) "
                        f"Stage=({float(stage_pt[0]):.4f}, {float(stage_pt[1]):.4f})"
                    )
        self.update_display()
        if tile.get("single_frame", False):
            self.ui.finishplate.setText("finish sample locator")
        else:
            self.ui.finishplate.setText("continue to next region")
        next_step = "finish" if tile.get("single_frame", False) else "continue"
        self.tile_label.setText(
            self.tile_label.text()
            + f"\nGenerated {len(self.generated_locations)} FOVs. Review the yellow boxes, then click {next_step}."
        )

    def _save_locator_records(self):
        folder = os.path.join(self.save_dir, "Mosaic")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "usb_mosaic_locator_rois.json")
        records = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "calibration": self.calibration,
            "tile_records": [
                {
                    "tile_index": idx + 1,
                    "stage_x": float(tile["stage_x"]),
                    "stage_y": float(tile["stage_y"]),
                    "stage_z": float(tile["stage_z"]),
                    "image_path": tile.get("image_path", ""),
                    "single_frame": bool(tile.get("single_frame", False)),
                }
                for idx, tile in enumerate(self.tile_records)
            ],
            "rois": self.final_tile_roi_records,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
        print(f"USB mosaic locator ROI records saved: {path}")
