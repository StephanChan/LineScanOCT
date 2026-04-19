# -*- coding: utf-8 -*-
"""Display and overlay rendering helpers for the OCT control UI."""

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

from Generaic_functions import RGBImagePlot, RGBOverlayArray, RGBOverlayPlot, fastLinePlot, LinePlot
from SampleLocator import USB_PIXEL_SIZE_MM, stage_to_usb_image


def mosaic_label_render_size(label):
    QApplication.processEvents()
    label_w = label.width()
    label_h = label.height()
    if label_w < 100 or label_h < 100:
        blank = QPixmap(300, 300)
        blank.fill(Qt.black)
        label.setPixmap(blank)
        QApplication.processEvents()
        label_w = label.width()
        label_h = label.height()
    if label_w < 100 or label_h < 100:
        return 300, 300
    upscale = max(1, min(2, int(np.ceil(300 / max(label_w, label_h)))))
    return int(label_w * upscale), int(label_h * upscale)


def display_sample_overlay(ui, overlay_images, sample_id, fov_locations_getter):
    source = overlay_images.get(sample_id)
    if source is None:
        ui.MosaicLabel.clear()
        return
    if isinstance(source, QPixmap):
        ui.MosaicLabel.setPixmap(source)
        return
    if source.get('type') == 'usb_roi':
        render_usb_roi_overlay(
            ui,
            sample_id,
            source['raw_img'],
            source['pixel_polygons'],
            fov_locations_getter,
        )
    elif source.get('type') == 'mosaic_correction':
        render_mosaic_correction_overlay(ui, source)


def render_usb_roi_overlay(ui, sample_id, raw_img, pixel_polygons, fov_locations_getter):
    poly_pts = pixel_polygons[sample_id - 1]
    poly_np = np.array(poly_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(poly_np)

    pad = 150
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(raw_img.shape[1], x + w + pad), min(raw_img.shape[0], y + h + pad)
    crop_img = raw_img[y1:y2, x1:x2].copy()

    rgb_img = np.ascontiguousarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    h_v, w_v, ch = rgb_img.shape
    qt_img = QImage(rgb_img.tobytes(), w_v, h_v, ch * w_v, QImage.Format_RGB888).copy()
    base_pixmap = QPixmap.fromImage(qt_img)

    label_w, label_h = mosaic_label_render_size(ui.MosaicLabel)
    final_buffer = QPixmap(label_w, label_h)
    final_buffer.fill(Qt.black)

    painter = QPainter(final_buffer)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)

    scale = min(label_w / w_v, label_h / h_v)
    sw, sh = int(w_v * scale), int(h_v * scale)
    dx, dy = (label_w - sw) // 2, (label_h - sh) // 2
    painter.drawPixmap(dx, dy, sw, sh, base_pixmap)

    def to_ui(px, py):
        return dx + (px - x1) * scale, dy + (py - y1) * scale

    painter.setPen(QPen(QColor(0, 120, 255), 3))
    for i in range(len(poly_pts)):
        p1 = to_ui(*poly_pts[i])
        p2 = to_ui(*poly_pts[(i + 1) % len(poly_pts)])
        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    usb_pixel_size = USB_PIXEL_SIZE_MM
    fov_x_px = ui.XLength.value() / usb_pixel_size

    painter.setPen(QPen(QColor(0, 255, 0), 2))
    for fov in fov_locations_getter(sample_id):
        cx_px, cy_px = stage_to_usb_image(fov['x'], fov['y'], raw_img.shape[1])
        loc_y_fov = fov.get('y_length_mm', ui.YLength.value())
        fov_y_px = loc_y_fov / usb_pixel_size
        tl = to_ui(cx_px - fov_y_px / 2, cy_px - fov_x_px / 2)
        br = to_ui(cx_px + fov_y_px / 2, cy_px + fov_x_px / 2)
        painter.drawRect(QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]))

    painter.end()
    ui.MosaicLabel.setPixmap(final_buffer)


def render_mosaic_correction_overlay(ui, source):
    mos_img = np.ascontiguousarray(source['mos_img'])
    orig_h, orig_w = mos_img.shape
    px_w_mm = source['px_w_mm']
    px_h_mm = source['px_h_mm']
    xfov = source['XFOV']
    yfov = source['YFOV']
    mm_polygons = source['mm_polygons']
    new_fov_locations = source['fov_locations']
    mos_min_x, mos_min_y, _, _ = source['mosaic_bounds']
    global_min_x, global_min_y, _, _ = source['global_bounds']
    canvas_w_px, canvas_h_px = source['canvas_size_px']

    label_w, label_h = mosaic_label_render_size(ui.MosaicLabel)
    final_buffer = QPixmap(label_w, label_h)
    final_buffer.fill(Qt.black)

    painter = QPainter(final_buffer)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)

    scale_w = label_w / canvas_w_px
    scale_h = label_h / canvas_h_px
    sw, sh = int(canvas_w_px * scale_w), int(canvas_h_px * scale_h)
    dx, dy = (label_w - sw) // 2, (label_h - sh) // 2

    qt_mos = QImage(mos_img.tobytes(), orig_w, orig_h, orig_w, QImage.Format_Grayscale8).copy()
    mos_pixmap = QPixmap.fromImage(qt_mos)

    mos_offset_x = (mos_min_x - global_min_x) / px_w_mm
    mos_offset_y = (mos_min_y - global_min_y) / px_h_mm
    painter.drawPixmap(
        int(dx + mos_offset_x * scale_w),
        int(dy + mos_offset_y * scale_h),
        int(orig_w * scale_w),
        int(orig_h * scale_h),
        mos_pixmap,
    )

    painter.setPen(QPen(QColor(0, 255, 0), 1))
    for fov in new_fov_locations:
        loc_y_fov = fov.get('y_length_mm', yfov)
        tl_x = (fov['x'] - xfov / 2 - global_min_x) / px_w_mm
        tl_y = (fov['y'] - loc_y_fov / 2 - global_min_y) / px_h_mm
        br_x = (fov['x'] + xfov / 2 - global_min_x) / px_w_mm
        br_y = (fov['y'] + loc_y_fov / 2 - global_min_y) / px_h_mm
        painter.drawRect(QRectF(dx + tl_x * scale_w, dy + tl_y * scale_h, (br_x - tl_x) * scale_w, (br_y - tl_y) * scale_h))

    painter.setPen(QPen(QColor(255, 0, 0), 2))
    for mm_poly in mm_polygons:
        for i in range(len(mm_poly)):
            p1_mm, p2_mm = mm_poly[i], mm_poly[(i + 1) % len(mm_poly)]
            x1_ui = dx + ((p1_mm[0] - global_min_x) / px_w_mm) * scale_w
            y1_ui = dy + ((p1_mm[1] - global_min_y) / px_h_mm) * scale_h
            x2_ui = dx + ((p2_mm[0] - global_min_x) / px_w_mm) * scale_w
            y2_ui = dy + ((p2_mm[1] - global_min_y) / px_h_mm) * scale_h
            painter.drawLine(int(x1_ui), int(y1_ui), int(x2_ui), int(y2_ui))

    painter.end()
    ui.MosaicLabel.setPixmap(final_buffer)


def render_aodo_waveform_ready(ui, payload):
    ao_waveform = payload.get("ao_waveform", None)
    do_waveform = payload.get("do_waveform", None)
    if ao_waveform is None or do_waveform is None:
        return
    pixmap = LinePlot(
        ao_waveform,
        do_waveform,
        np.min([np.min(ao_waveform), 0]),
        np.max([np.max(ao_waveform), 1]),
    )
    ui.XwaveformLabel.setPixmap(pixmap)


def render_aline_ready(ui, payload):
    aline = payload.get("aline", None)
    if aline is None:
        return
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    pixmap = fastLinePlot(aline, width=ui.XZplane.width(), height=ui.XZplane.height(), m=ym, M=yM)
    ui.XZplane.setPixmap(pixmap)


def render_bline_ready(ui, payload):
    bline = payload.get("bline", None)
    if bline is None:
        return
    dyn = payload.get("dyn", None)
    use_dynamic = ui.DynCheckBox.isChecked()
    dynamic_alpha = ui.DynContrast.value() / 100.0 if hasattr(ui, "DynContrast") else 0.5
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    if use_dynamic and dyn is not None and np.size(dyn) > 0:
        pixmap = RGBOverlayPlot(bline, dyn, ym, yM, alpha=dynamic_alpha)
    else:
        pixmap = RGBImagePlot(matrix1=bline, m=ym, M=yM)
    ui.XZplane.setPixmap(pixmap)


def render_cscan_ready(ui, payload):
    bline = payload.get("bline", None)
    dynb = payload.get("dynb", None)
    aip = payload.get("aip", None)
    dyn = payload.get("dyn", None)
    use_dynamic = ui.DynCheckBox.isChecked()
    dynamic_alpha = ui.DynContrast.value() / 100.0 if hasattr(ui, "DynContrast") else 0.5

    if bline is not None:
        ym = ui.XZmin.value()
        yM = ui.XZmax.value()
        if use_dynamic and dynb is not None and np.size(dynb) > 0:
            pixmap = RGBOverlayPlot(bline, dynb, ym, yM, alpha=dynamic_alpha)
        else:
            pixmap = RGBImagePlot(matrix1=bline, m=ym, M=yM)
        ui.XZplane.setPixmap(pixmap)

    if aip is not None and getattr(ui, "mosaic_viewer", None) is not None:
        x_step_size = ui.XStepSize.value()
        y_step_size = ui.YStepSize.value()
        if use_dynamic and dyn is not None and np.size(dyn) > 0:
            tmp = RGBOverlayArray(aip, dyn, ui.Intmin.value(), ui.Intmax.value(), alpha=dynamic_alpha)
            ui.mosaic_viewer.set_image(tmp, ui.Intmin.value(), ui.Intmax.value(), x_step_size, y_step_size)
        else:
            ui.mosaic_viewer.set_image(aip, ui.Intmin.value(), ui.Intmax.value(), x_step_size, y_step_size)


def render_mosaic_ready(ui, payload):
    mosaic = payload.get("mosaic", None)
    bline = payload.get("bline", None)
    if bline is not None:
        ym = ui.XZmin.value()
        yM = ui.XZmax.value()
        pixmap = RGBImagePlot(matrix1=bline, m=ym, M=yM)
        ui.XZplane.setPixmap(pixmap)
    if mosaic is not None and getattr(ui, "mosaic_viewer", None) is not None:
        x_step_size = ui.XStepSize.value()
        y_step_size = ui.YStepSize.value()
        ui.mosaic_viewer.set_image(mosaic, ui.Intmin.value(), ui.Intmax.value(), x_step_size, y_step_size)
