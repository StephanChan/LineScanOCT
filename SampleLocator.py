import os
import cv2
import numpy as np
import tifffile as tiff
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QTimer, QPoint, QEvent

# Ensure SampleLocatorUI.py is in the same directory
from SampleLocatorUI import Ui_Form

class UnifiedSampleScanner(QDialog):
    def __init__(self, save_dir=r'D:\LineScanOCT',fov_w_mm = 2.0,fov_h_mm = 1.0):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.pic_window.setScaledContents(False)
        self.ui.pic_window.setAlignment(Qt.AlignCenter)

        # Microscope Parameters
        self.pixel_size_mm = 0.02
        self.fov_w_mm = fov_w_mm
        self.fov_h_mm = fov_h_mm

        # State Variables
        self.img_bgr = None
        self.polygons = []         
        self.current_polygon = []   
        self.display_scale = 1.0    
        self.generated_locations = [] # List of FOVs: {'x', 'y', 'sample_id'}
        self.sample_centers = []      # List of Sample Centers: {'x', 'y', 'sample_id'}
        self.result_overlay = None  
        
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.pan_x = 0 
        self.pan_y = 0
        
        self.img_bgr = self._capture_and_save_raw()
        
        self.ui.pic_window.setMouseTracking(True)
        self.ui.pic_window.installEventFilter(self) 
        
        self.ui.finishwell.clicked.connect(self.complete_polygon)
        self.ui.finishplate.clicked.connect(self.process_generate_mosaic)

        QTimer.singleShot(100, self.reset_view)

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
                    frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)
                    frame_to_use = frame[630:3670, 180:2160]
            
            if frame_to_use is None:
                frame_to_use = np.full((3040, 1980, 3), 40, dtype=np.uint8)
                for i in range(3):
                    cv2.circle(frame_to_use, (500 + i*500, 1500), 120, (100, 100, 100), -1)

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

    def eventFilter(self, source, event):
        if source is self.ui.pic_window and self.img_bgr is not None:
            h, w = self.img_bgr.shape[:2]
            if event.type() == QEvent.MouseButtonPress:
                win_w, win_h = self.ui.pic_window.width(), self.ui.pic_window.height()
                img_display_w = w * self.display_scale
                img_display_h = h * self.display_scale
                base_x = (win_w - img_display_w) / 2
                base_y = (win_h - img_display_h) / 2
                img_x = (event.pos().x() - base_x) / self.display_scale - self.pan_x
                img_y = (event.pos().y() - base_y) / self.display_scale - self.pan_y

                if event.button() == Qt.LeftButton and self.result_overlay is None:
                    if 0 <= img_x < w and 0 <= img_y < h:
                        self.current_polygon.append((img_x, img_y))
                        self.update_display()
                    return True
                elif event.button() == Qt.RightButton and self.result_overlay is None:
                    if self.current_polygon: self.current_polygon.pop() 
                    elif self.polygons: self.polygons.pop()        
                    self.update_display()
                    return True
                elif event.button() == Qt.MidButton:
                    self.is_dragging = True
                    self.last_mouse_pos = event.pos()
                    return True
            elif event.type() == QEvent.MouseMove:
                if self.is_dragging:
                    delta = event.pos() - self.last_mouse_pos
                    self.pan_x += delta.x() / self.display_scale
                    self.pan_y += delta.y() / self.display_scale
                    self.last_mouse_pos = event.pos()
                    self.update_display()
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.MidButton:
                    self.is_dragging = False
                    return True
            elif event.type() == QEvent.Wheel:
                self.display_scale *= (1.15 if event.angleDelta().y() > 0 else 0.85)
                self.display_scale = max(0.01, min(15.0, self.display_scale))
                self.update_display()
                return True
        return super().eventFilter(source, event)

    def update_display(self):
        if self.img_bgr is None: return
        h, w = self.img_bgr.shape[:2]
        if self.result_overlay is not None:
            canvas = cv2.cvtColor(self.result_overlay, cv2.COLOR_BGR2RGB)
        else:
            canvas = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            for poly in self.polygons:
                cv2.polylines(canvas, [np.array(poly, np.int32)], True, (0, 255, 0), 12)
            for pt in self.current_polygon:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)
            if len(self.current_polygon) > 1:
                cv2.polylines(canvas, [np.array(self.current_polygon, np.int32)], False, (255, 0, 0), 8)

        qimg = QImage(canvas.data, w, h, 3 * w, QImage.Format_RGB888)
        full_pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = full_pixmap.scaled(int(w * self.display_scale), int(h * self.display_scale), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        final_buffer = QPixmap(self.ui.pic_window.size())
        final_buffer.fill(Qt.black)
        painter = QPainter(final_buffer)
        dx = (self.ui.pic_window.width() - scaled_pixmap.width()) / 2 + (self.pan_x * self.display_scale)
        dy = (self.ui.pic_window.height() - scaled_pixmap.height()) / 2 + (self.pan_y * self.display_scale)
        painter.drawPixmap(int(dx), int(dy), scaled_pixmap)
        painter.end()
        self.ui.pic_window.setPixmap(final_buffer)

    def complete_polygon(self):
        if self.result_overlay is not None: return 
        if len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            self.update_display()

    def process_generate_mosaic(self):
        if not self.polygons:
            QMessageBox.warning(self, "Error", "No regions drawn.")
            return

        ts = self._get_timestamp()
        h, w = self.img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self.polygons:
            cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
        
        fov_img = self.img_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.generated_locations = []
        self.sample_centers = []
        
        fov_w_px = self.fov_w_mm / self.pixel_size_mm
        fov_h_px = self.fov_h_mm / self.pixel_size_mm

        for idx, cnt in enumerate(contours):
            sample_id = idx + 1 
            
            # 1. Compute Mass Center (Centroid)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX_px = int(M["m10"] / M["m00"])
                cY_px = int(M["m01"] / M["m00"])
            else:
                x, y, cw, ch = cv2.boundingRect(cnt)
                cX_px, cY_px = int(x + cw/2), int(y + ch/2)
            
            self.sample_centers.append({
                'sample_id': sample_id,
                'x': cX_px * self.pixel_size_mm,
                'y': cY_px * self.pixel_size_mm
            })

            # 2. Draw Mass Center as a Red Point on the overlay
            cv2.circle(fov_img, (cX_px, cY_px), 25, (0, 0, 255), -1)

            # 3. Calculate FOVs for this region
            x, y, cw, ch = cv2.boundingRect(cnt)
            cols = int(np.ceil((cw * self.pixel_size_mm) / self.fov_w_mm))
            rows = int(np.ceil((ch * self.pixel_size_mm) / self.fov_h_mm))
            
            # Center the grid of FOVs around the sample mass center
            start_x_px = cX_px - (cols * fov_w_px) / 2
            start_y_px = cY_px - (rows * fov_h_px) / 2
            
            for r in range(rows):
                for c in range(cols):
                    px1, py1 = int(start_x_px + c * fov_w_px), int(start_y_px + r * fov_h_px)
                    px2, py2 = int(px1 + fov_w_px), int(py1 + fov_h_px)
                    
                    # Only add FOVs that overlap with the drawn mask
                    if np.any(mask[max(0,py1):min(h,py2), max(0,px1):min(w,px2)] > 0):
                        cv2.rectangle(fov_img, (px1, py1), (px2, py2), (255, 255, 0), 5)
                        
                        self.generated_locations.append({
                            'sample_id': sample_id,
                            'x': (px1 + fov_w_px/2) * self.pixel_size_mm, 
                            'y': (py1 + fov_h_px/2) * self.pixel_size_mm
                        })

        self.result_overlay = fov_img
        cv2.imwrite(os.path.join(self.save_dir, f"FOVoverlayed_{ts}.png"), fov_img)
        
        # Save CSVs
        with open(os.path.join(self.save_dir, f"Locations_{ts}.csv"), "w") as f:
            f.write("sample_id,center_x_mm,center_y_mm\n")
            for p in self.generated_locations:
                f.write(f"{p['sample_id']},{p['x']:.3f},{p['y']:.3f}\n")
        
        with open(os.path.join(self.save_dir, f"SampleCenters_{ts}.csv"), "w") as f:
            f.write("sample_id,mass_center_x_mm,mass_center_y_mm\n")
            for c in self.sample_centers:
                f.write(f"{c['sample_id']},{c['x']:.3f},{c['y']:.3f}\n")

        self.update_display()
        # Message box and processing logic now occur only once
        QMessageBox.information(self, "Success", f"Mosaic Generated.\n{len(self.sample_centers)} samples found.")
        # This tells the main GUI that the user finished successfully
        self.accept()

if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    scanner = UnifiedSampleScanner()
    scanner.show()
    app.exec_()
    fov_locs, centers = scanner.generated_locations, scanner.sample_centers
    
    print("\n--- Sample Mass Centers ---")
    for c in centers:
        print(f"Sample {c['sample_id']} Center -> x: {c['x']:.3f} mm, y: {c['y']:.3f} mm")
        
    print("\n--- FOV Locations ---")
    for loc in fov_locs:
        print(f"Sample {loc['sample_id']} FOV -> x: {loc['x']:.3f} mm, y: {loc['y']:.3f} mm")