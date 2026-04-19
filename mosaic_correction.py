# -*- coding: utf-8 -*-
"""Geometry helpers for refining sample ROIs from the OCT mosaic viewer."""

import numpy as np


def mosaic_polygons_to_stage_mm(
    raw_polygons,
    current_fov_locations,
    x_fov_mm,
    source_y_length_mm,
    x_step_um,
    y_step_um,
):
    """
    Convert mosaic-viewer raw pixel polygons into stage-mm polygons.

    The mosaic viewer stores polygons in raw image coordinates. The physical
    anchor is the upper-left physical edge of the stitched mosaic region.
    """
    px_w_mm = x_step_um / 1000.0
    px_h_mm = y_step_um / 1000.0

    xs_orig = [p["x"] for p in current_fov_locations]
    ys_orig = [p["y"] for p in current_fov_locations]
    anchor_x = min(xs_orig) - (x_fov_mm / 2.0)
    anchor_y = min(ys_orig) - (source_y_length_mm / 2.0)

    mm_polygons = []
    polygon_debug = []
    for poly in raw_polygons:
        raw_poly = np.array(poly, dtype=float)
        mm_poly = [(p[0] * px_w_mm + anchor_x, p[1] * px_h_mm + anchor_y) for p in poly]
        mm_poly_array = np.array(mm_poly, dtype=float)
        mm_polygons.append(mm_poly)

        raw_min = np.min(raw_poly, axis=0)
        raw_max = np.max(raw_poly, axis=0)
        mm_min = np.min(mm_poly_array, axis=0)
        mm_max = np.max(mm_poly_array, axis=0)
        polygon_debug.append(
            {
                "vertices": len(poly),
                "raw_bounds": (raw_min[0], raw_min[1], raw_max[0], raw_max[1]),
                "raw_size": (raw_max[0] - raw_min[0], raw_max[1] - raw_min[1]),
                "mm_bounds": (mm_min[0], mm_min[1], mm_max[0], mm_max[1]),
                "mm_size": (mm_max[0] - mm_min[0], mm_max[1] - mm_min[1]),
            }
        )

    return {
        "mm_polygons": mm_polygons,
        "px_w_mm": px_w_mm,
        "px_h_mm": px_h_mm,
        "anchor": (anchor_x, anchor_y),
        "polygon_debug": polygon_debug,
    }


def build_mosaic_correction_overlay_source(
    mosaic_image,
    current_fov_locations,
    mm_polygons,
    new_fov_locations,
    x_fov_mm,
    y_fov_mm,
    source_y_length_mm,
    x_step_um,
    y_step_um,
    margin_mm=1.0,
):
    """Build the source dict consumed by Display_rendering for correction overlays."""
    mos_img = np.ascontiguousarray(mosaic_image)
    orig_h, orig_w = mos_img.shape
    px_w_mm = x_step_um / 1000.0
    px_h_mm = y_step_um / 1000.0

    xs_orig = [p["x"] for p in current_fov_locations]
    ys_orig = [p["y"] for p in current_fov_locations]
    mos_min_x = min(xs_orig) - (x_fov_mm / 2.0)
    mos_min_y = min(ys_orig) - (source_y_length_mm / 2.0)
    mos_max_x = mos_min_x + (orig_w * px_w_mm)
    mos_max_y = mos_min_y + (orig_h * px_h_mm)

    all_poly_pts = [pt for poly in mm_polygons for pt in poly]
    poly_min_x = min(p[0] for p in all_poly_pts)
    poly_min_y = min(p[1] for p in all_poly_pts)
    poly_max_x = max(p[0] for p in all_poly_pts)
    poly_max_y = max(p[1] for p in all_poly_pts)

    global_min_x = min(mos_min_x, poly_min_x) - margin_mm
    global_min_y = min(mos_min_y, poly_min_y) - margin_mm
    global_max_x = max(mos_max_x, poly_max_x) + margin_mm
    global_max_y = max(mos_max_y, poly_max_y) + margin_mm

    canvas_w_px = int((global_max_x - global_min_x) / px_w_mm)
    canvas_h_px = int((global_max_y - global_min_y) / px_h_mm)

    return {
        "type": "mosaic_correction",
        "mos_img": mos_img,
        "mm_polygons": mm_polygons,
        "fov_locations": [dict(fov) for fov in new_fov_locations],
        "px_w_mm": px_w_mm,
        "px_h_mm": px_h_mm,
        "XFOV": x_fov_mm,
        "YFOV": y_fov_mm,
        "mosaic_bounds": (mos_min_x, mos_min_y, mos_max_x, mos_max_y),
        "global_bounds": (global_min_x, global_min_y, global_max_x, global_max_y),
        "canvas_size_px": (canvas_w_px, canvas_h_px),
    }
