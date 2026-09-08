# -*- coding: utf-8 -*-
"""Physical axis rulers drawn in the letterbox margins of OCT display panels.

Each display panel already letterboxes its image with a known aspect ratio, so a
ruler can be drawn along the bottom edge (horizontal axis) and the left edge
(vertical axis) using the physical extent of that axis in micrometres. Tick
spacing is chosen automatically from "nice" 1/2/5 x 10^n steps so labels are
always round numbers. The unit switches from micrometres to millimetres once an
axis total reaches 1 mm.
"""

import math

from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRectF

# Reserved margins (widget/canvas pixels) used for the rulers.
RULER_LEFT = 66
RULER_BOTTOM = 30
RULER_TOP = 8
RULER_RIGHT = 54

_RULER_PEN = QColor(255, 255, 255, 230)
_FONT = None


def _ruler_font():
    global _FONT
    if _FONT is None:
        _FONT = QFont("Arial")
        _FONT.setPixelSize(14)
    return _FONT


def inner_rect(widget_w, widget_h):
    """Return the available QRectF for the image after reserving ruler margins."""
    w = max(0.0, float(widget_w) - RULER_LEFT - RULER_RIGHT)
    h = max(0.0, float(widget_h) - RULER_TOP - RULER_BOTTOM)
    return QRectF(RULER_LEFT, RULER_TOP, w, h)


def _nice_step_um(total_um, span_px, min_spacing_px=60.0):
    if span_px <= 0 or total_um <= 0:
        return None
    raw = float(total_um) * min_spacing_px / float(span_px)
    if raw <= 0 or not math.isfinite(raw):
        return None
    exponent = math.floor(math.log10(raw))
    base = 10.0 ** exponent
    mantissa = raw / base
    if mantissa <= 1.0:
        factor = 1.0
    elif mantissa <= 2.0:
        factor = 2.0
    elif mantissa <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * base


def use_mm(total_um):
    return float(total_um) >= 999.5


def _format_tick(value_um, total_um):
    if use_mm(total_um):
        value = float(value_um) / 1000.0
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    value = float(value_um)
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    return f"{value:.1f}"


def unit_suffix(total_um):
    return "mm" if use_mm(total_um) else "um"


def draw_bottom_ruler(painter, rect, total_um, caption="", canvas_right=None, origin_at_right=False):
    """Horizontal ruler along the bottom edge of ``rect``.

    ``total_um`` is the physical extent that the full rect width represents.
    By default zero is placed at the left edge and values increase to the right.
    With ``origin_at_right=True`` zero sits at the right edge and values increase
    toward the left (mirrored horizontal axis).
    """
    span_px = rect.width()
    step_um = _nice_step_um(total_um, span_px)
    if step_um is None:
        return
    x_left = rect.left()
    x_right = rect.right()
    y0 = rect.bottom()
    per_um_px = span_px / float(total_um)

    painter.save()
    painter.setPen(QPen(_RULER_PEN, 1))
    painter.setFont(_ruler_font())

    # Baseline along the image bottom edge.
    painter.drawLine(int(x_left), int(y0), int(x_right), int(y0))

    value = 0.0
    index = 0
    while value <= float(total_um) * (1.0 + 1e-9):
        if origin_at_right:
            px = x_right - value * per_um_px
        else:
            px = x_left + value * per_um_px
        painter.drawLine(int(px), int(y0), int(px), int(y0 + 5))
        painter.drawText(
            QRectF(px - 45, y0 + 5, 90, 18),
            Qt.AlignHCenter | Qt.AlignTop,
            _format_tick(value, total_um),
        )
        index += 1
        if index > 500:
            break
        value += step_um

    if caption:
        left = rect.right() + 4
        if canvas_right is not None:
            width = max(10.0, float(canvas_right) - left)
            painter.drawText(
                QRectF(left, y0 + 5, width, 18),
                Qt.AlignRight | Qt.AlignTop,
                caption,
            )
        else:
            painter.drawText(
                QRectF(left, y0 + 5, 120, 18),
                Qt.AlignLeft | Qt.AlignTop,
                caption,
            )
    painter.restore()


def draw_left_ruler(painter, rect, total_um, caption=""):
    """Vertical ruler along the left edge of ``rect``.

    ``total_um`` is the physical extent that the full rect height represents.
    Zero is placed at the top edge of the image (e.g. depth origin for XZ).
    """
    span_px = rect.height()
    step_um = _nice_step_um(total_um, span_px)
    if step_um is None:
        return
    x0 = rect.left()
    y0 = rect.top()
    per_um_px = span_px / float(total_um)

    painter.save()
    painter.setPen(QPen(_RULER_PEN, 1))
    painter.setFont(_ruler_font())

    # Baseline along the image left edge.
    painter.drawLine(int(x0), int(y0), int(x0), int(y0 + span_px))

    value = 0.0
    index = 0
    while value <= float(total_um) * (1.0 + 1e-9):
        ty = y0 + value * per_um_px
        painter.drawLine(int(x0), int(ty), int(x0 - 5), int(ty))
        painter.drawText(
            QRectF(x0 - 64, ty - 10, 56, 20),
            Qt.AlignRight | Qt.AlignVCenter,
            _format_tick(value, total_um),
        )
        index += 1
        if index > 500:
            break
        value += step_um

    if caption:
        # Rotated caption reading bottom-to-top along the left margin.
        painter.save()
        painter.translate(4.0, rect.bottom())
        painter.rotate(-90.0)
        painter.drawText(
            QRectF(0, -12, 600, 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            caption,
        )
        painter.restore()
    painter.restore()


# =============================================================================
# Viewport-locked rulers
#
# These are used by PanZoomImageView when the content image is zoomed / panned.
# Ticks are drawn in the fixed gutter strips along the widget edges, at the
# CURRENT screen position of each physical value, so the ruler always stays
# glued to the window border and the numbers slide / renumber as the view
# changes.
# =============================================================================

def draw_viewport_bottom_ruler(
    painter,
    inner_rect,
    lo_um,
    hi_um,
    value_to_x,
    total_um_for_units,
    caption="",
    outer_width=None,
):
    """Horizontal ticks along the bottom edge of the content viewport.

    ``value_to_x(v)`` maps a physical value (µm) to the screen x of that value
    (takes pan/zoom into account). Only ticks in [lo_um, hi_um] are drawn.
    """
    span_um = abs(float(hi_um) - float(lo_um))
    if span_um <= 0:
        return
    try:
        span_px = abs(float(value_to_x(hi_um)) - float(value_to_x(lo_um)))
    except Exception:
        return
    step_um = _nice_step_um(span_um, span_px)
    if step_um is None:
        return
    edge_y = inner_rect.bottom()
    total_um = float(total_um_for_units)

    painter.save()
    painter.setPen(QPen(_RULER_PEN, 1))
    painter.setFont(_ruler_font())

    lo = min(float(lo_um), float(hi_um))
    hi = max(float(lo_um), float(hi_um))
    index = 0
    value = math.ceil(lo / step_um - 1e-9) * step_um
    while value <= hi * (1.0 + 1e-9):
        try:
            px = float(value_to_x(value))
        except Exception:
            break
        if outer_width is not None:
            if px < -5 or px > float(outer_width) + 5:
                value += step_um
                index += 1
                if index > 2000:
                    break
                continue
        painter.drawLine(int(px), int(edge_y), int(px), int(edge_y + 5))
        painter.drawText(
            QRectF(px - 45, edge_y + 5, 90, 18),
            Qt.AlignHCenter | Qt.AlignTop,
            _format_tick(value, total_um),
        )
        value += step_um
        index += 1
        if index > 2000:
            break

    if caption:
        left = inner_rect.right() + 4
        if outer_width is not None:
            width = max(10.0, float(outer_width) - left)
        else:
            width = 140.0
        painter.drawText(
            QRectF(left, edge_y + 5, width, 18),
            Qt.AlignRight | Qt.AlignTop,
            caption,
        )
    painter.restore()


def draw_viewport_left_ruler(
    painter,
    inner_rect,
    lo_um,
    hi_um,
    value_to_y,
    total_um_for_units,
    caption="",
):
    """Vertical ticks along the left edge of the content viewport.

    ``value_to_y(v)`` maps a physical value (µm) to the screen y of that value.
    Only ticks in [lo_um, hi_um] are drawn.
    """
    span_um = abs(float(hi_um) - float(lo_um))
    if span_um <= 0:
        return
    try:
        span_px = abs(float(value_to_y(hi_um)) - float(value_to_y(lo_um)))
    except Exception:
        return
    step_um = _nice_step_um(span_um, span_px)
    if step_um is None:
        return
    edge_x = inner_rect.left()
    total_um = float(total_um_for_units)

    painter.save()
    painter.setPen(QPen(_RULER_PEN, 1))
    painter.setFont(_ruler_font())

    lo = min(float(lo_um), float(hi_um))
    hi = max(float(lo_um), float(hi_um))
    index = 0
    value = math.ceil(lo / step_um - 1e-9) * step_um
    while value <= hi * (1.0 + 1e-9):
        try:
            py = float(value_to_y(value))
        except Exception:
            break
        painter.drawLine(int(edge_x), int(py), int(edge_x - 5), int(py))
        painter.drawText(
            QRectF(edge_x - 64, py - 10, 56, 20),
            Qt.AlignRight | Qt.AlignVCenter,
            _format_tick(value, total_um),
        )
        value += step_um
        index += 1
        if index > 2000:
            break

    if caption:
        painter.save()
        painter.translate(inner_rect.left() - 6.0, inner_rect.bottom())
        painter.rotate(-90.0)
        painter.drawText(
            QRectF(0, -12, 600, 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            caption,
        )
        painter.restore()
    painter.restore()
