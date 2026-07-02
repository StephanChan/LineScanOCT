# -*- coding: utf-8 -*-
"""Display and overlay rendering helpers for the OCT control UI."""

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

from Generaic_functions import RGBImagePlot, fastLinePlot, LinePlot
from SampleLocator import USB_PIXEL_SIZE_MM, stage_to_usb_image


RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT = 15.0 / 1000.0
RGB_DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ = (0.0, 8.0)
RGB_DYNAMIC_VALUE_DYNAMIC_RANGE = (0.0, 500.0)
RGB_DYNAMIC_VALUE_GAMMA = 1.0


def display_array(array):
    if isinstance(array, np.ndarray) and array.dtype.kind == 'c':
        return np.abs(array)
    return array


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


def rgb_pixmap(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8, copy=False))
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def rgb_display_limits(min_widget, max_widget):
    control_max = max(
        1,
        int(min_widget.maximum()) if hasattr(min_widget, "maximum") else 255,
        int(max_widget.maximum()) if hasattr(max_widget, "maximum") else 255,
    )
    scale = 255.0 / float(control_max)
    return float(min_widget.value()) * scale, float(max_widget.value()) * scale


def dynamic_rgb_display_array(ui, rgb, min_widget, max_widget):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    m, M = rgb_display_limits(min_widget, max_widget)
    adjusted = (rgb - m) / (M - m + 1e-5) * 255.0
    if hasattr(ui, "DynContrast"):
        adjusted *= float(ui.DynContrast.value()) / 50.0
    return np.ascontiguousarray(np.clip(adjusted, 0, 255).astype(np.uint8))


def hue_frequency_range_from_controls(min_widget, max_widget):
    return (
        float(min_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
        float(max_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
    )


def normalize_to_unit_interval(image, value_range, gamma=1.0):
    low_value, high_value = float(value_range[0]), float(value_range[1])
    if high_value <= low_value:
        raise ValueError(f"Invalid display normalization range: {value_range}")
    normalized = (np.asarray(image, dtype=np.float32) - low_value) / (high_value - low_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    gamma = float(gamma)
    if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        normalized = normalized ** (1.0 / gamma)
    return normalized


def hsv_to_rgb_array(hue, saturation, value):
    hue = np.mod(np.asarray(hue, dtype=np.float32), 1.0)
    saturation = np.clip(np.asarray(saturation, dtype=np.float32), 0.0, 1.0)
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i.astype(np.float32)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    i_mod = np.mod(i, 6)

    rgb = np.empty(hue.shape + (3,), dtype=np.float32)
    masks = [
        (i_mod == 0, value, t, p),
        (i_mod == 1, q, value, p),
        (i_mod == 2, p, value, t),
        (i_mod == 3, p, q, value),
        (i_mod == 4, t, p, value),
        (i_mod == 5, value, p, q),
    ]
    for mask, red, green, blue in masks:
        rgb[..., 0][mask] = red[mask]
        rgb[..., 1][mask] = green[mask]
        rgb[..., 2][mask] = blue[mask]
    return np.ascontiguousarray(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8))


def dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, min_widget, max_widget):
    hue_range = hue_frequency_range_from_controls(min_widget, max_widget)
    hue = normalize_to_unit_interval(frequency_hz, hue_range)
    saturation = normalize_to_unit_interval(
        bandwidth_hz,
        RGB_DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ,
    )
    value = normalize_to_unit_interval(
        value,
        RGB_DYNAMIC_VALUE_DYNAMIC_RANGE,
        gamma=RGB_DYNAMIC_VALUE_GAMMA,
    )
    if hasattr(ui, "DynContrast"):
        value = np.clip(value * (float(ui.DynContrast.value()) / 50.0), 0.0, 1.0)
    return hsv_to_rgb_array(hue, saturation, value)


def z_depth_index(ui, z_pixels):
    if z_pixels <= 0:
        return 0
    if not hasattr(ui, "ZDepthBar"):
        return 0
    return max(0, min(int(ui.ZDepthBar.value()), int(z_pixels) - 1))


def z_plane_from_volume(ui, volume):
    if volume is None or np.size(volume) == 0:
        return None
    volume = np.asarray(volume)
    if volume.ndim == 3:
        return volume[:, :, z_depth_index(ui, volume.shape[2])]
    if volume.ndim == 4 and volume.shape[-1] == 3:
        return volume[:, :, z_depth_index(ui, volume.shape[2]), :]
    raise ValueError(f"XY volume must have shape (Y, X, Z) or (Y, X, Z, 3), got {volume.shape}")


def render_xz_pixmap(ui, intensity, rgb=None, hsv=None, frequency_hz=None, bandwidth_hz=None, value=None):
    if hsv is not None and np.size(hsv) > 0:
        hsv = np.asarray(hsv, dtype=np.float32)
        return rgb_pixmap(dynamic_metric_rgb_display_array(ui, hsv[..., 0], hsv[..., 1], hsv[..., 2], ui.XZmin, ui.XZmax))
    if rgb is not None and np.size(rgb) > 0:
        if frequency_hz is not None and bandwidth_hz is not None and value is not None:
            return rgb_pixmap(dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, ui.XZmin, ui.XZmax))
        return rgb_pixmap(dynamic_rgb_display_array(ui, rgb, ui.XZmin, ui.XZmax))
    intensity = display_array(intensity)
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    return RGBImagePlot(matrix1=intensity, m=ym, M=yM)


def set_xy_projection(
    ui,
    intensity,
    rgb=None,
    hsv=None,
    frequency_hz=None,
    bandwidth_hz=None,
    value=None,
    volume=None,
    hsv_volume=None,
):
    if getattr(ui, "mosaic_viewer", None) is None:
        return
    if intensity is None and volume is None:
        return
    x_step_size = ui.XStepSize.value()
    y_step_size = ui.YStepSize.value()
    volume_plane = z_plane_from_volume(ui, volume)
    if volume_plane is not None:
        intensity = volume_plane
    hsv_volume_plane = z_plane_from_volume(ui, hsv_volume)
    if hsv_volume_plane is not None:
        hsv = hsv_volume_plane
    if hsv is not None and np.size(hsv) > 0:
        hsv = np.asarray(hsv, dtype=np.float32)
        rgb = dynamic_metric_rgb_display_array(ui, hsv[..., 0], hsv[..., 1], hsv[..., 2], ui.XZmin, ui.XZmax)
    elif rgb is not None and np.size(rgb) > 0:
        if frequency_hz is not None and bandwidth_hz is not None and value is not None:
            rgb = dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, ui.XZmin, ui.XZmax)
        else:
            rgb = dynamic_rgb_display_array(ui, rgb, ui.XZmin, ui.XZmax)
    else:
        intensity = display_array(intensity)
        ui.mosaic_viewer.set_image(intensity, ui.XZmin.value(), ui.XZmax.value(), x_step_size, y_step_size)
        return
    if rgb is not None and np.size(rgb) > 0:
        ui.mosaic_viewer.set_image(
            rgb,
            0,
            255,
            x_step_size,
            y_step_size,
        )
        return


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
        cx_px, cy_px = stage_to_usb_image(fov.x, fov.y, raw_img.shape[1])
        loc_y_fov = fov.y_length_mm if fov.y_length_mm is not None else ui.YLength.value()
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
        loc_y_fov = fov.y_length_mm if fov.y_length_mm is not None else yfov
        tl_x = (fov.x - xfov / 2 - global_min_x) / px_w_mm
        tl_y = (fov.y - loc_y_fov / 2 - global_min_y) / px_h_mm
        br_x = (fov.x + xfov / 2 - global_min_x) / px_w_mm
        br_y = (fov.y + loc_y_fov / 2 - global_min_y) / px_h_mm
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
    ao_waveform = np.asarray(ao_waveform, dtype=np.float32)
    do_waveform = np.asarray(do_waveform, dtype=np.float32)
    wave_min = float(min(np.min(ao_waveform), np.min(do_waveform)))
    wave_max = float(max(np.max(ao_waveform), np.max(do_waveform)))
    if wave_max <= wave_min:
        margin = 1.0
    else:
        margin = 0.05 * (wave_max - wave_min)
    pixmap = LinePlot(
        ao_waveform,
        do_waveform,
        wave_min - margin,
        wave_max + margin,
    )
    ui.XwaveformLabel.setPixmap(pixmap)


def render_aline_ready(ui, payload):
    aline = payload.get("aline", None)
    if aline is None:
        return
    aline = display_array(aline)
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    pixmap = fastLinePlot(aline, width=ui.XZplane.width(), height=ui.XZplane.height(), m=ym, M=yM)
    ui.XZplane.setPixmap(pixmap)


def render_bline_ready(ui, payload):
    bline = payload.get("bline", None)
    if bline is None:
        return
    rgb = payload.get("rgb", None)
    ui.XZplane.setPixmap(
        render_xz_pixmap(
            ui,
            bline,
            rgb,
            payload.get("hsv", None),
            payload.get("freq", None),
            payload.get("bandwidth", None),
            payload.get("value", None),
        )
    )


def render_cscan_ready(ui, payload):
    bline = payload.get("bline", None)
    rgbb = payload.get("rgbb", None)
    aip = payload.get("aip", None)
    rgb = payload.get("rgb", None)

    if bline is not None:
        ui.XZplane.setPixmap(
            render_xz_pixmap(
                ui,
                bline,
                rgbb,
                payload.get("hsvb", None),
                payload.get("freqb", None),
                payload.get("bandwidthb", None),
                payload.get("valueb", None),
            )
        )

    set_xy_projection(
        ui,
        aip,
        rgb,
        payload.get("hsv", None),
        payload.get("freq", None),
        payload.get("bandwidth", None),
        payload.get("value", None),
        payload.get("volume", None),
        payload.get("hsv_volume", None),
    )


def render_mosaic_ready(ui, payload):
    mosaic = payload.get("mosaic", None)
    bline = payload.get("bline", None)
    bline_rgb = payload.get("bline_rgb", None)
    mosaic_rgb = payload.get("mosaic_rgb", None)
    if bline is not None:
        ui.XZplane.setPixmap(
            render_xz_pixmap(
                ui,
                bline,
                bline_rgb,
                payload.get("bline_hsv", None),
                payload.get("bline_freq", None),
                payload.get("bline_bandwidth", None),
                payload.get("bline_value", None),
            )
        )
    set_xy_projection(
        ui,
        mosaic,
        mosaic_rgb,
        payload.get("mosaic_hsv", None),
        payload.get("mosaic_freq", None),
        payload.get("mosaic_bandwidth", None),
        payload.get("mosaic_value", None),
        payload.get("mosaic_volume", None),
        payload.get("mosaic_hsv_volume", None),
    )
