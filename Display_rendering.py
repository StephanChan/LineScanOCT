# -*- coding: utf-8 -*-
"""Display and overlay rendering helpers for the OCT control UI."""

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

from Generaic_functions import RGBImagePlot, fastLinePlot, LinePlot
import Rulers
from SampleLocator import (
    affine_fov_half_size_pixels,
    calibration_uses_affine,
    stage_to_image_from_calibration,
)


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


def set_label_pixmap_fit(label, pixmap):
    """Display ``pixmap`` on a QLabel preserving its native aspect ratio.

    The label keeps the black background and the image is letterboxed (centered)
    inside a black canvas sized to the label. This avoids the distortion caused
    by QLabel ``setScaledContents(True)`` stretching the pixmap to the widget.
    """
    if label is None or pixmap is None or pixmap.isNull():
        return
    try:
        label_w, label_h = label.width(), label.height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 300, 300
        image_w, image_h = pixmap.width(), pixmap.height()
        if image_w <= 0 or image_h <= 0:
            label.setPixmap(pixmap)
            return
        fit = min(label_w / image_w, label_h / image_h)
        draw_w = max(1, int(round(image_w * fit)))
        draw_h = max(1, int(round(image_h * fit)))
        dx = (label_w - draw_w) // 2
        dy = (label_h - draw_h) // 2

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(
            QRectF(dx, dy, draw_w, draw_h),
            pixmap,
            QRectF(pixmap.rect()),
        )
        painter.end()
        label.setPixmap(canvas)
    except Exception as error:
        print(f"Label aspect-fit failed: {error}")
        try:
            label.setPixmap(pixmap)
        except Exception:
            pass


def rgb_display_limits(min_widget, max_widget):
    control_max = max(
        1,
        int(min_widget.maximum()) if hasattr(min_widget, "maximum") else 255,
        int(max_widget.maximum()) if hasattr(max_widget, "maximum") else 255,
    )
    scale = 255.0 / float(control_max)
    return float(min_widget.value()) * scale, float(max_widget.value()) * scale


def dynamic_brightness_contrast(ui):
    """Return (brightness_offset, contrast_gain) from the UI sliders.

    Contrast is DynContrast (default 50 => gain 1.0). Brightness is DynBrightness
    centered at 50 (range 0..100) mapped to an additive offset in [-0.5, 0.5].
    """
    contrast = 1.0
    brightness = 0.0
    if hasattr(ui, "DynContrast"):
        try:
            contrast = float(ui.DynContrast.value()) / 50.0
        except Exception:
            contrast = 1.0
    if hasattr(ui, "DynBrightness"):
        try:
            brightness = (float(ui.DynBrightness.value()) - 50.0) / 100.0
        except Exception:
            brightness = 0.0
    return brightness, contrast


def dynamic_rgb_display_array(ui, rgb, min_widget, max_widget):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    m, M = rgb_display_limits(min_widget, max_widget)
    adjusted = (rgb - m) / (M - m + 1e-5) * 255.0
    brightness, contrast = dynamic_brightness_contrast(ui)
    adjusted = adjusted * contrast + brightness * 255.0
    return np.ascontiguousarray(np.clip(adjusted, 0, 255).astype(np.uint8))


def hue_frequency_range_from_controls(min_widget, max_widget):
    return (
        float(min_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
        float(max_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
    )


def normalize_to_unit_interval(image, value_range, gamma=1.0):
    low_value, high_value = float(value_range[0]), float(value_range[1])
    if not (np.isfinite(low_value) and np.isfinite(high_value)):
        raise ValueError(f"Invalid display normalization range: {value_range}")
    if high_value < low_value:
        low_value, high_value = high_value, low_value
    if high_value <= low_value:
        # Degenerate window (e.g. XZmin == XZmax, reached by sliding XZmax to 0
        # while XZmin is 0): there is no scale to normalize against, so return a
        # flat zero map instead of crashing the GUI.
        return np.zeros(np.shape(image), dtype=np.float32)
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
    brightness, contrast = dynamic_brightness_contrast(ui)
    value = np.clip(value * contrast + brightness, 0.0, 1.0)
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


def cscan_display_y_index(ui, y_pixels):
    """Clamp YBar to a valid C-scan Y index (mid-plane fallback)."""
    y_pixels = int(y_pixels)
    if y_pixels <= 0:
        return 0
    bar = getattr(ui, "YBar", None)
    if bar is None:
        return y_pixels // 2
    try:
        y_value = int(bar.value())
    except Exception:
        y_value = y_pixels // 2
    return max(0, min(y_value, y_pixels - 1))


def downsample_rgb(rgb, factor):
    """Block/cubic downsample an RGB array in X/Y by ``factor``."""
    factor = max(1, int(factor))
    if factor <= 1:
        return rgb
    rgb = np.asarray(rgb)
    height, width = rgb.shape[:2]
    new_width = max(1, width // factor)
    new_height = max(1, height // factor)
    return cv2.resize(
        np.ascontiguousarray(rgb),
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def set_xyplane_dyn_pixmap(ui, rgb, pixel_size_x=1.0, pixel_size_y=1.0):
    """Draw an HSV/RGB en-face plane on XYplaneDyn.

    Applies exactly the same geometry as XYplaneInt's interactive mosaic widget:
    ``usb_top_view`` orientation (transpose + one vertical flip, matching the
    XYplaneInt mosaic viewer exactly) and a physical
    pixel-size aspect stretch. The image is drawn bottom-left anchored inside
    the ruler margins and physical axis rulers are painted in the margins.
    ``pixel_size_x`` / ``pixel_size_y`` must be the effective pitch (µm per
    displayed pixel) of the supplied array, i.e. the raw pitch already scaled
    by any X/Y downsampling applied before this call. Bottom ruler = stage Y,
    left ruler = stage X.
    """
    label = getattr(ui, "XYplaneDyn", None)
    if label is None or rgb is None or np.size(rgb) == 0:
        return
    try:
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            display = np.ascontiguousarray(np.flip(np.transpose(rgb, (1, 0, 2)), axis=0))
        elif rgb.ndim == 2:
            display = np.ascontiguousarray(np.flipud(rgb.T))
        else:
            return
        height, width = display.shape[:2]
        try:
            sx = float(pixel_size_x)
            sy = float(pixel_size_y)
            aspect = (sx / sy) if sy != 0 else 1.0
        except (TypeError, ValueError):
            sx = sy = 1.0
            aspect = 1.0

        base_pixmap = rgb_pixmap(display)
        label_w, label_h = mosaic_label_render_size(label)

        phys_w = float(width)
        phys_h = float(height) * aspect
        if phys_w <= 0 or phys_h <= 0:
            return
        inner = Rulers.inner_rect(label_w, label_h)
        if inner.width() <= 0 or inner.height() <= 0:
            return
        fit = min(inner.width() / phys_w, inner.height() / phys_h)
        draw_w = max(1, int(round(phys_w * fit)))
        draw_h = max(1, int(round(phys_h * fit)))
        # Anchor the image to the bottom-left corner of the inner area so the
        # bottom and left rulers hug the image edges.
        draw_x = int(inner.left())
        draw_y = int(inner.bottom() - draw_h)

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(
            QRectF(draw_x, draw_y, draw_w, draw_h),
            base_pixmap,
            QRectF(base_pixmap.rect()),
        )
        image_rect = QRectF(draw_x, draw_y, draw_w, draw_h)
        stage_y_total = float(width) * sy
        stage_x_total = float(height) * sx
        y_unit = Rulers.unit_suffix(stage_y_total)
        x_unit = Rulers.unit_suffix(stage_x_total)
        Rulers.draw_bottom_ruler(
            painter,
            image_rect,
            stage_y_total,
            caption=f"stage Y ({y_unit})",
            canvas_right=label_w,
            origin_at_right=True,
        )
        Rulers.draw_left_ruler(
            painter,
            image_rect,
            stage_x_total,
            caption=f"stage X ({x_unit})",
        )
        painter.end()
        label.setPixmap(canvas)
    except Exception as error:
        print(f"XYplaneDyn update failed: {error}")


def clear_xyplane_dyn(ui):
    """Non-dynamic frames should leave the dynamic view blank/black."""
    label = getattr(ui, "XYplaneDyn", None)
    if label is None:
        return
    try:
        blank = QPixmap(300, 300)
        blank.fill(Qt.black)
        label.setPixmap(blank)
    except Exception:
        pass


def render_cscan_xz_from_volume(ui, payload):
    """Render the XZ view of a C-scan from the full [Y, X, Z] volume at the
    Y-plane selected by YBar. Returns True when it updated the view."""
    volume = payload.get("volume", None)
    if volume is None or np.size(volume) == 0:
        return False
    volume = np.asarray(volume)
    if volume.ndim != 3:
        return False
    y_index = cscan_display_y_index(ui, volume.shape[0])
    # volume[y] is (X, Z); the XZ display wants (Z, X).
    xz_intensity = np.transpose(volume[y_index]).copy()

    hsv_volume = payload.get("hsv_volume", None)
    xz_hsv = None
    if hsv_volume is not None and np.size(hsv_volume) > 0:
        hsv_volume = np.asarray(hsv_volume)
        if hsv_volume.ndim == 4 and hsv_volume.shape[0] == volume.shape[0]:
            xz_hsv = np.transpose(hsv_volume[y_index], (1, 0, 2)).copy()

    if xz_hsv is not None and np.size(xz_hsv) > 0:
        pixmap = render_xz_pixmap(ui, xz_intensity, None, xz_hsv)
    else:
        pixmap = render_xz_pixmap(ui, xz_intensity)
    set_xzplane_pixmap_with_aspect(ui, pixmap)
    return True


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


def xz_physical_pixel_sizes_um(ui):
    """Return (x_um_per_pixel, z_um_per_pixel) for the XZ-plane display.

    x: lateral pitch = XStepSize * AlineAVG (each displayed X pixel spans
       AlineAVG galvo steps after averaging).
    z: axial depth pitch in µm, defined in OCT_MT (default 4.0 µm).
    """
    aline_avg = 1
    avg_ctrl = getattr(ui, "AlineAVG", None)
    if avg_ctrl is not None:
        try:
            aline_avg = max(1, int(avg_ctrl.value()))
        except Exception:
            aline_avg = 1
    try:
        x_um = float(ui.XStepSize.value()) * aline_avg
    except Exception:
        x_um = 1.0
    try:
        z_um = float(getattr(ui, "axial_pixel_size_um", 4.0))
    except Exception:
        z_um = 4.0
    return x_um, z_um


def set_xzplane_pixmap_with_aspect(ui, pixmap):
    """Display a pixmap on XZplane preserving its true physical aspect.

    XZ images are stored with rows = depth (Z) and columns = lateral (X), so
    each column spans x_um and each row spans axial z_um. The image is drawn
    bottom-left anchored inside the ruler margins and physical axis rulers are
    painted in the margins (bottom = lateral X, left = depth Z from 0 at the
    top of the displayed depth window).
    """
    label = getattr(ui, "XZplane", None)
    if label is None or pixmap is None or pixmap.isNull():
        return
    try:
        x_count = float(pixmap.width())
        z_count = float(pixmap.height())
        if x_count <= 0 or z_count <= 0:
            label.setPixmap(pixmap)
            return
        x_um, z_um = xz_physical_pixel_sizes_um(ui)
        phys_w = x_count * x_um
        phys_h = z_count * z_um
        if phys_w <= 0 or phys_h <= 0:
            label.setPixmap(pixmap)
            return
        label_w, label_h = mosaic_label_render_size(label)
        inner = Rulers.inner_rect(label_w, label_h)
        if inner.width() <= 0 or inner.height() <= 0:
            label.setPixmap(pixmap)
            return
        fit = min(inner.width() / phys_w, inner.height() / phys_h)
        draw_w = max(1, int(round(phys_w * fit)))
        draw_h = max(1, int(round(phys_h * fit)))
        # Anchor the image to the bottom-left corner of the inner area so the
        # bottom and left rulers hug the image edges.
        draw_x = int(inner.left())
        draw_y = int(inner.bottom() - draw_h)

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(
            QRectF(draw_x, draw_y, draw_w, draw_h),
            pixmap,
            QRectF(pixmap.rect()),
        )
        image_rect = QRectF(draw_x, draw_y, draw_w, draw_h)
        x_unit = Rulers.unit_suffix(x_count * x_um)
        z_unit = Rulers.unit_suffix(z_count * z_um)
        Rulers.draw_bottom_ruler(
            painter,
            image_rect,
            x_count * x_um,
            caption=f"lateral X ({x_unit})",
            canvas_right=label_w,
        )
        Rulers.draw_left_ruler(
            painter,
            image_rect,
            z_count * z_um,
            caption=f"depth ({z_unit})",
        )
        painter.end()
        label.setPixmap(canvas)
    except Exception as error:
        print(f"XZplane aspect-fit failed: {error}")
        try:
            label.setPixmap(pixmap)
        except Exception:
            pass


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
    volume_downsample=1.0,
):
    """Route en-face projections to the structure view (XYplaneInt/mosaic_viewer)
    and the dynamic view (XYplaneDyn) at the shared ZDepthBar plane.

    Intensity goes to XYplaneInt with XZmin/XZmax contrast; dynamic (HSV or
    frequency/bandwidth/value RGB) goes to XYplaneDyn with the DynBrightness /
    DynContrast sliders. Both honour the ``scale`` downsample control.

    ``volume_downsample`` is the X/Y downsample factor already applied to
    ``volume`` / ``hsv_volume`` (mosaic stitched volumes). The raw µm-per-pixel
    pitch is multiplied by this factor so physical extents (and the rulers
    derived from them) stay correct for the downsampled volume.
    """
    if getattr(ui, "mosaic_viewer", None) is None:
        return
    if intensity is None and volume is None:
        return
    x_step_size = ui.XStepSize.value()
    y_step_size = ui.YStepSize.value()
    # AlineAVG reduces the X pixel count (AlinesPerBline // AlineAVG), so each
    # displayed X pixel spans AlineAVG galvo steps; keep the physical aspect
    # ratio correct.
    aline_avg = max(1, int(getattr(ui, "AlineAVG", 1).value() if hasattr(ui, "AlineAVG") else 1))
    x_step_size = float(x_step_size) * aline_avg
    # XY-plane display downsample from the UI spinbox (X/Y only).
    scale_control = getattr(ui, "scale", None)
    downsample = max(1, int(scale_control.value())) if scale_control is not None else 1
    volume_downsample = max(1.0, float(volume_downsample or 1.0))

    # --- Structure (intensity) en-face plane -> XYplaneInt ---
    # In-RAM stitched volumes are already downsampled by the UI scale for
    # display; do not downsample again on this path. Live non-volume planes
    # (e.g. per-tile AIP) still use the scale downsample.
    intensity_plane = None
    volume_plane = z_plane_from_volume(ui, volume)
    if volume_plane is not None:
        intensity_plane = display_array(volume_plane)
        int_downsample = 1
        struct_step_x = x_step_size * volume_downsample
        struct_step_y = y_step_size * volume_downsample
    elif intensity is not None:
        intensity_plane = display_array(intensity)
        int_downsample = downsample
        struct_step_x = x_step_size
        struct_step_y = y_step_size
    if intensity_plane is not None:
        ui.mosaic_viewer.set_image(
            intensity_plane,
            ui.XZmin.value(),
            ui.XZmax.value(),
            struct_step_x,
            struct_step_y,
            downsample=int_downsample,
        )

    # --- Dynamic (HSV / RGB) en-face plane -> XYplaneDyn ---
    # Same rule as intensity: downsampled in-RAM HSV volumes are used as-is for
    # display; live 2-D HSV planes get the scale downsample.
    hsv_volume_plane = z_plane_from_volume(ui, hsv_volume)
    if hsv_volume_plane is not None:
        hsv = hsv_volume_plane
        dyn_downsample = 1
    else:
        dyn_downsample = downsample
    dyn_rgb = None
    if hsv is not None and np.size(hsv) > 0:
        hsv = np.asarray(hsv, dtype=np.float32)
        dyn_rgb = dynamic_metric_rgb_display_array(ui, hsv[..., 0], hsv[..., 1], hsv[..., 2], ui.XZmin, ui.XZmax)
    elif rgb is not None and np.size(rgb) > 0:
        if frequency_hz is not None and bandwidth_hz is not None and value is not None:
            dyn_rgb = dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, ui.XZmin, ui.XZmax)
        else:
            dyn_rgb = dynamic_rgb_display_array(ui, rgb, ui.XZmin, ui.XZmax)
    if dyn_rgb is not None:
        if hsv_volume_plane is not None:
            eff_step_x = x_step_size * volume_downsample
            eff_step_y = y_step_size * volume_downsample
        else:
            # downsample_rgb() below reduces the plane by dyn_downsample, so the
            # per-pixel pitch seen by the drawing helper must be scaled up by it.
            eff_step_x = x_step_size * dyn_downsample
            eff_step_y = y_step_size * dyn_downsample
        set_xyplane_dyn_pixmap(
            ui,
            downsample_rgb(dyn_rgb, dyn_downsample),
            eff_step_x,
            eff_step_y,
        )
    else:
        clear_xyplane_dyn(ui)


def display_sample_overlay(ui, overlay_images, sample_id, fov_locations_getter):
    source = overlay_images.get(sample_id)
    if source is None:
        ui.MosaicLabel.clear()
        return
    if isinstance(source, QPixmap):
        ui.MosaicLabel.setPixmap(source)
        return
    if source.get('type') == 'usb_region':
        render_usb_region_overlay(ui, source, fov_locations_getter)
    elif source.get('type') == 'mosaic_correction':
        render_mosaic_correction_overlay(ui, source)


def render_usb_region_overlay(ui, source, fov_locations_getter):
    image_path = source.get("image_path", "")
    raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if raw_img is None:
        print(f"USB region overlay image missing or unreadable: {image_path}")
        ui.MosaicLabel.clear()
        return

    poly_pts = source.get("pixel_polygon", [])
    if len(poly_pts) < 3:
        ui.MosaicLabel.clear()
        return
    poly_np = np.asarray(poly_pts, dtype=np.float32)
    calibration = source.get("calibration")
    if calibration is None or not calibration_uses_affine(calibration):
        raise ValueError(f"USB region overlay requires affine calibration: {calibration}")

    # Crop tightly to the union of the drawn ROI and all of its generated FOVs
    # (both in full USB-frame pixels) plus a small margin, so the sample area and
    # FOV layout fill the MosaicLabel instead of being lost in a large fixed
    # 200 px border.
    min_x = float(np.min(poly_np[:, 0]))
    max_x = float(np.max(poly_np[:, 0]))
    min_y = float(np.min(poly_np[:, 1]))
    max_y = float(np.max(poly_np[:, 1]))
    for fov in fov_locations_getter(int(source["sample_id"])):
        cx_px, cy_px = stage_to_image_from_calibration(calibration, fov.x, fov.y)
        loc_y_fov = fov.y_length_mm if fov.y_length_mm is not None else ui.YLength.value()
        fov_half_x, fov_half_y = affine_fov_half_size_pixels(
            calibration, ui.XLength.value(), loc_y_fov
        )
        min_x = min(min_x, float(cx_px) - fov_half_x)
        max_x = max(max_x, float(cx_px) + fov_half_x)
        min_y = min(min_y, float(cy_px) - fov_half_y)
        max_y = max(max_y, float(cy_px) + fov_half_y)
    crop_margin = 30.0
    x_min = int(max(0, np.floor(min_x - crop_margin)))
    y_min = int(max(0, np.floor(min_y - crop_margin)))
    x_max = int(min(raw_img.shape[1], np.ceil(max_x + crop_margin)))
    y_max = int(min(raw_img.shape[0], np.ceil(max_y + crop_margin)))
    crop_img = raw_img[y_min:y_max, x_min:x_max].copy()
    if crop_img.size == 0:
        print(f"USB region overlay crop is empty for sampleID-{source.get('sample_id')}")
        ui.MosaicLabel.clear()
        return

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
        return dx + (float(px) - x_min) * scale, dy + (float(py) - y_min) * scale

    calibration = source["calibration"]
    if not calibration_uses_affine(calibration):
        raise ValueError(f"USB region overlay requires affine calibration: {calibration}")

    def stage_to_image(stage_x, stage_y):
        return stage_to_image_from_calibration(calibration, stage_x, stage_y)

    painter.setPen(QPen(QColor(0, 120, 255), 3))
    for i in range(len(poly_pts)):
        p1 = to_ui(*poly_pts[i])
        p2 = to_ui(*poly_pts[(i + 1) % len(poly_pts)])
        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    painter.setPen(QPen(QColor(0, 255, 0), 2))
    for fov in fov_locations_getter(int(source["sample_id"])):
        cx_px, cy_px = stage_to_image(fov.x, fov.y)
        loc_y_fov = fov.y_length_mm if fov.y_length_mm is not None else ui.YLength.value()
        fov_y_px, fov_x_px = affine_fov_half_size_pixels(
            calibration,
            ui.XLength.value(),
            loc_y_fov,
        )
        fov_x_px *= 2.0
        fov_y_px *= 2.0
        tl = to_ui(cx_px - fov_y_px / 2.0, cy_px - fov_x_px / 2.0)
        br = to_ui(cx_px + fov_y_px / 2.0, cy_px + fov_x_px / 2.0)
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
    set_xzplane_pixmap_with_aspect(
        ui,
        render_xz_pixmap(
            ui,
            bline,
            rgb,
            payload.get("hsv", None),
            payload.get("freq", None),
            payload.get("bandwidth", None),
            payload.get("value", None),
        ),
    )


def render_cscan_ready(ui, payload):
    bline = payload.get("bline", None)
    rgbb = payload.get("rgbb", None)
    aip = payload.get("aip", None)
    rgb = payload.get("rgb", None)

    if not render_cscan_xz_from_volume(ui, payload):
        if bline is not None:
            set_xzplane_pixmap_with_aspect(
                ui,
                render_xz_pixmap(
                    ui,
                    bline,
                    rgbb,
                    payload.get("hsvb", None),
                    payload.get("freqb", None),
                    payload.get("bandwidthb", None),
                    payload.get("valueb", None),
                ),
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
        set_xzplane_pixmap_with_aspect(
            ui,
            render_xz_pixmap(
                ui,
                bline,
                bline_rgb,
                payload.get("bline_hsv", None),
                payload.get("bline_freq", None),
                payload.get("bline_bandwidth", None),
                payload.get("bline_value", None),
            ),
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
        volume_downsample=float(getattr(ui, "mosaic_display_downsample", 1) or 1),
    )
