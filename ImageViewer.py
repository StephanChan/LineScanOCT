# -*- coding: utf-8 -*-
"""Unified interactive OCT image viewer with viewport-locked physical rulers.

PanZoomImageView is used by every OCT data display window (XZplane,
XZplaneInt, XZplaneDyn, XYplaneDyn and the XYplaneInt structure view). It
paints an image-only pixmap inside fixed ruler gutters, lets the user zoom
with the wheel, pan with a middle-button drag and reset with a double-click,
and draws the physical axis rulers ALONG THE WIDGET BORDERS. The ruler ticks
are placed at the current screen position of each physical value, so the
numbers renumber when zooming and slide when panning.

ROI / polygon drawing is intentionally not part of this widget.
"""

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QSizePolicy

import Rulers


def downsample_display_array(array, scale):
    """Block-mean downsample a (H, W) or (H, W, C) array by an integer scale."""
    scale = max(1, int(scale))
    if scale == 1:
        return array
    if array.ndim not in (2, 3):
        return array
    h, w = array.shape[:2]
    h_crop, w_crop = h - h % scale, w - w % scale
    if h_crop == 0 or w_crop == 0:
        return array[::scale, ::scale]
    if array.ndim == 2:
        view = array[:h_crop, :w_crop]
        return view.reshape(h_crop // scale, scale, w_crop // scale, scale).mean(axis=(1, 3))
    view = array[:h_crop, :w_crop, :]
    out = view.reshape(h_crop // scale, scale, w_crop // scale, scale, array.shape[2]).mean(axis=(1, 3))
    return out.astype(array.dtype, copy=False)


def array_to_pixmap(display_adj):
    """Convert a grayscale (H, W) or RGB (H, W, 3) uint8 array to a QPixmap."""
    display_adj = np.ascontiguousarray(display_adj)
    if display_adj.ndim == 2:
        h, w = display_adj.shape
        image = QImage(display_adj.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
    elif display_adj.ndim == 3 and display_adj.shape[2] == 3:
        h, w, _ = display_adj.shape
        image = QImage(display_adj.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
    else:
        raise ValueError(f"Unsupported display array shape: {display_adj.shape}")
    return QPixmap.fromImage(image)


def usb_top_view_orientation(adj):
    """Return the ``usb_top_view`` display orientation of a raw (Y, X[, 3]) array.

    raw mosaic: rows = stage Y, columns = stage X
    USB top view: horizontal = stage Y, vertical = stage X.
    display = transpose + vertical flip, matching the physical/locator view.
    """
    if adj.ndim == 2:
        return np.ascontiguousarray(np.flipud(adj.T))
    if adj.ndim == 3:
        return np.ascontiguousarray(np.flip(np.transpose(adj, (1, 0, 2)), axis=0))
    return adj


class PanZoomImageView(QWidget):
    """Image viewer with viewport-locked rulers shared by all OCT windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(size_policy)

        # Content pixmap (image only - no baked rulers / letterbox margins).
        self._content = None
        # Geometry dict; None disables rulers (legacy pixmap content).
        self._geo = None
        # Screen pixels per source column.
        self._scale_x = 1.0
        # Vertical screen stretch relative to horizontal (preserves physical
        # aspect when the two axes have different µm-per-pixel values).
        self._vertical_factor = 1.0
        # Pan offsets in screen pixels relative to the fit position.
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._fit_applied = False
        self._last_w = 0
        self._last_h = 0
        self._dragging = False
        self._last_mouse = QPointF()

    # ------------------------------------------------------------- geometry
    def inner_rect(self):
        return Rulers.inner_rect(max(1, self.width()), max(1, self.height()))

    def _viewport_rect(self):
        """Rect the content is drawn in: the ruler-gutter inner rect for
        geometry-bearing images, the full widget for legacy pixmaps."""
        if self._geo is None:
            return QRectF(0, 0, max(1, self.width()), max(1, self.height()))
        return self.inner_rect()

    def content_size(self):
        if self._content is None or self._content.isNull():
            return 0, 0
        return self._content.width(), self._content.height()

    def geometry_signature(self):
        pw, ph = self.content_size()
        g = self._geo or {}
        return (
            pw,
            ph,
            float(g.get("um_per_col", 1.0) or 1.0),
            float(g.get("um_per_row", 1.0) or 1.0),
        )

    def _set_source(self, pixmap, geometry):
        """Store new content pixmap; keep the view when nothing changed."""
        if pixmap is not None and pixmap.isNull():
            pixmap = None
        old_key = self.geometry_signature()
        self._content = pixmap
        self._geo = geometry if geometry is not None else None
        if pixmap is None:
            self._fit_applied = False
            self.update()
            return
        g = self._geo or {}
        self._vertical_factor = (
            float(g.get("um_per_row", 1.0) or 1.0)
            / float(g.get("um_per_col", 1.0) or 1.0)
            if self._geo is not None
            else 1.0
        )
        new_key = self.geometry_signature()
        if not self._fit_applied or old_key != new_key:
            if self._geo is None:
                self.reset_legacy_view()
            else:
                self.reset_view()
        else:
            self.update()
    # ------------------------------------------------------------- public API
    def setPixmap(self, pixmap):
        """Legacy content path (no physical axis metadata -> no rulers)."""
        self._set_source(pixmap, None)

    def pixmap(self):
        return self._content

    def clear(self):
        self._set_source(None, None)

    def set_data(self, pixmap, geometry):
        """Set image-only content plus axis metadata (µm per pixel etc.)."""
        if pixmap is None or pixmap.isNull():
            self.clear()
            return
        self._set_source(pixmap, geometry)

    def set_image(self, numpy_array, m, M, pixel_size_x=1.0, pixel_size_y=1.0, downsample=1):
        """NumPy structure/en-face input (XYplaneInt path).

        ``numpy_array`` is a raw (stage-Y, stage-X[, RGB]) array. It is
        contrast-stretched, oriented with the ``usb_top_view`` convention and
        optionally block-downsampled for display. ``pixel_size_x`` / ``_y`` are
        the physical pitch (µm) per RAW pixel. The bottom ruler shows stage Y
        (origin at the right edge), the left ruler shows stage X.
        """
        numpy_array = np.asarray(numpy_array)
        if numpy_array.size == 0:
            return
        try:
            m = float(m)
            M = float(M)
        except (TypeError, ValueError):
            m, M = float(np.nanmin(numpy_array)), float(np.nanmax(numpy_array))
        denom = float(M - m)
        if not np.isfinite(denom) or abs(denom) < 1e-9:
            denom = 1.0
        adj = np.ascontiguousarray(
            np.clip(((numpy_array - m) / denom * 255.0), 0, 255).astype(np.uint8)
        )
        downsample = max(1, int(downsample))
        display_adj = downsample_display_array(usb_top_view_orientation(adj), downsample)
        try:
            pixel_size_x = float(pixel_size_x) if pixel_size_x not in (None, 0) else 1.0
            pixel_size_y = float(pixel_size_y) if pixel_size_y not in (None, 0) else 1.0
        except (TypeError, ValueError):
            pixel_size_x = pixel_size_y = 1.0
        if pixel_size_x <= 0:
            pixel_size_x = 1.0
        if pixel_size_y <= 0:
            pixel_size_y = 1.0

        pixmap = array_to_pixmap(display_adj)
        ph, pw = display_adj.shape[:2]
        # Display columns are raw stage-Y pixels; display rows are raw stage-X
        # pixels. µm per displayed pixmap pixel is the pitch * downsample.
        um_per_col = pixel_size_y * downsample
        um_per_row = pixel_size_x * downsample
        geometry = {
            "um_per_col": um_per_col,
            "um_per_row": um_per_row,
            "bottom_axis": {"caption": "stage Y", "origin": "right"},
            "left_axis": {"caption": "stage X", "origin": "top"},
            "bottom_total_um": pw * um_per_col,
            "left_total_um": ph * um_per_row,
        }
        self._set_source(pixmap, geometry)

    def reset_view(self):
        """Fit the whole content pixmap (physical aspect) bottom-left anchored
        inside the viewport between the ruler gutters."""
        pw, ph = self.content_size()
        inner = self.inner_rect()
        if pw <= 0 or ph <= 0 or inner.width() <= 0 or inner.height() <= 0:
            return
        g = self._geo or {}
        um_col = float(g.get("um_per_col", 1.0) or 1.0)
        um_row = float(g.get("um_per_row", 1.0) or 1.0)
        phys_w = pw * um_col
        phys_h = ph * um_row
        # Same screen px per µm on both axes preserves the physical aspect.
        fit = min(inner.width() / phys_w, inner.height() / phys_h)
        if not np.isfinite(fit) or fit <= 0:
            fit = min(inner.width() / pw, inner.height() / ph)
        self._scale_x = fit * um_col
        self._vertical_factor = um_row / um_col
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._fit_applied = True
        self.update()

    def reset_legacy_view(self):
        """Fit legacy (no-geometry) pixmaps centred like a QLabel would."""
        pw, ph = self.content_size()
        inner = self._viewport_rect()
        if pw <= 0 or ph <= 0 or inner.width() <= 0 or inner.height() <= 0:
            return
        fit = min(inner.width() / pw, inner.height() / ph)
        self._scale_x = fit
        self._vertical_factor = 1.0
        self._pan_x = (inner.width() - pw * fit) / 2.0
        self._pan_y = -(inner.height() - ph * fit) / 2.0
        self._fit_applied = True
        self.update()

    # -------------------------------------------------------------- view maths
    def _view_geometry(self):
        """Return (x0, y0, s_col, s_row, pw, ph).

        x0/y0 is the screen top-left of the content. At fit the content bottom
        and left edges sit on the inner viewport bottom/left; pan offsets move
        the whole content in screen pixels.
        """
        pw, ph = self.content_size()
        inner = self._viewport_rect()
        s_col = self._scale_x
        s_row = s_col * self._vertical_factor
        x0 = inner.left() + self._pan_x
        y0 = inner.bottom() + self._pan_y - ph * s_row
        return x0, y0, s_col, s_row, pw, ph

    def _fit_scale_limits(self):
        """(min_scale_x, max_scale_x) around the current fit geometry."""
        pw, ph = self.content_size()
        inner = self._viewport_rect()
        if pw <= 0 or ph <= 0 or inner.width() <= 0 or inner.height() <= 0:
            return self._scale_x, self._scale_x * 40.0
        g = self._geo or {}
        um_col = float(g.get("um_per_col", 1.0) or 1.0)
        um_row = float(g.get("um_per_row", 1.0) or 1.0)
        fit = min(inner.width() / (pw * um_col), inner.height() / (ph * um_row))
        if not np.isfinite(fit) or fit <= 0:
            fit = min(inner.width() / pw, inner.height() / ph)
        base = fit * um_col
        return base * 0.05, base * 40.0

    # ------------------------------------------------------------ ruler helpers
    def _axis_meta(self):
        g = self._geo or {}
        return (
            g.get("bottom_axis", {}),
            g.get("left_axis", {}),
            float(g.get("um_per_col", 1.0) or 1.0),
            float(g.get("um_per_row", 1.0) or 1.0),
            float(g.get("bottom_total_um", 0.0) or 0.0),
            float(g.get("left_total_um", 0.0) or 0.0),
        )

    def _draw_rulers(self, painter):
        if self._content is None or self._geo is None:
            return
        bottom_axis, left_axis, um_col, um_row, bottom_total, left_total = self._axis_meta()
        if um_col <= 0 or um_row <= 0 or bottom_total <= 0 or left_total <= 0:
            return
        x0, y0, s_col, s_row, pw, ph = self._view_geometry()
        inner = self.inner_rect()

        # ---- bottom ruler (horizontal axis) -------------------------------
        # Ticks are drawn wherever the data's horizontal footprint projects onto
        # the bottom edge - the ruler does not depend on the vertical pan/zoom.
        p_lo = max(0.0, (inner.left() - x0) / s_col)
        p_hi = min(float(pw), (inner.right() - x0) / s_col)
        if p_hi > p_lo and s_col > 0:
            origin = bottom_axis.get("origin", "left")

            def pixel_um(p):
                if origin == "right":
                    return (pw - p) * um_col
                return p * um_col

            v0 = pixel_um(p_lo)
            v1 = pixel_um(p_hi)
            lo, hi = min(v0, v1), max(v0, v1)
            if hi > lo:

                def value_to_x(v):
                    if origin == "right":
                        p = pw - v / um_col
                    else:
                        p = v / um_col
                    return x0 + p * s_col

                caption = f"{bottom_axis.get('caption', 'X')} ({Rulers.unit_suffix(bottom_total)})"
                Rulers.draw_viewport_bottom_ruler(
                    painter,
                    inner,
                    lo,
                    hi,
                    value_to_x,
                    bottom_total,
                    caption=caption,
                    outer_width=self.width(),
                )

        # ---- left ruler (vertical axis) -----------------------------------
        # Ticks are drawn wherever the data's vertical footprint projects onto
        # the left edge - the ruler does not depend on the horizontal pan/zoom.
        r_lo = max(0.0, (inner.top() - y0) / s_row)
        r_hi = min(float(ph), (inner.bottom() - y0) / s_row)
        if r_hi > r_lo and s_row > 0:
            lo = r_lo * um_row
            hi = r_hi * um_row
            if hi > lo:

                def value_to_y(v):
                    return y0 + (v / um_row) * s_row

                caption = f"{left_axis.get('caption', 'Y')} ({Rulers.unit_suffix(left_total)})"
                Rulers.draw_viewport_left_ruler(
                    painter,
                    inner,
                    lo,
                    hi,
                    value_to_y,
                    left_total,
                    caption=caption,
                )
    # -------------------------------------------------------------- painting
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._content is None or self._content.isNull():
            painter.end()
            return
        if not self._fit_applied:
            if self._geo is None:
                self.reset_legacy_view()
            else:
                self.reset_view()
            if not self._fit_applied:
                painter.end()
                return
        pw, ph = self.content_size()
        x0, y0, s_col, s_row, _, _ = self._view_geometry()
        viewport = self._viewport_rect()
        if pw > 0 and ph > 0 and s_col > 0 and s_row > 0:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.save()
            painter.setClipRect(viewport)
            painter.translate(x0, y0)
            painter.scale(s_col, s_row)
            painter.drawPixmap(0, 0, self._content)
            painter.restore()
        painter.setClipping(False)
        self._draw_rulers(painter)
        painter.end()

    def resizeEvent(self, event):
        old_w, old_h = self._last_w, self._last_h
        new_w, new_h = self.width(), self.height()
        if not self._fit_applied and self._content is not None:
            if self._geo is None:
                self.reset_legacy_view()
            else:
                self.reset_view()
        elif old_w > 0 and old_h > 0:
            self._pan_x += (new_w - old_w) / 2.0
            self._pan_y += (new_h - old_h) / 2.0
        self._last_w = new_w
        self._last_h = new_h
        super().resizeEvent(event)

    # ------------------------------------------------------------- interaction
    def wheelEvent(self, event):
        if self._content is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        lo, hi = self._fit_scale_limits()
        new_scale = min(max(self._scale_x * factor, lo), hi)
        if new_scale <= 0:
            return
        x0, y0, s_col, s_row, pw, ph = self._view_geometry()
        cursor = event.pos()
        # Source pixel fraction under the cursor stays fixed while zooming.
        src_col = (cursor.x() - x0) / s_col
        src_row = (cursor.y() - y0) / s_row
        self._scale_x = new_scale
        new_s_row = new_scale * self._vertical_factor
        inner = self.inner_rect()
        x0_new = cursor.x() - src_col * new_scale
        self._pan_x = x0_new - inner.left()
        y0_new = cursor.y() - src_row * new_s_row
        self._pan_y = y0_new + ph * new_s_row - inner.bottom()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:
            self._dragging = True
            self._last_mouse = QPointF(event.pos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            pos = QPointF(event.pos())
            delta = pos - self._last_mouse
            self._last_mouse = pos
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MidButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._geo is None:
                self.reset_legacy_view()
            else:
                self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
