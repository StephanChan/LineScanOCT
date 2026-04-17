import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


ROI_OCCUPANCY_TARGET = 0.80
FOV_OVERLAP = 0.10
MAX_Y_FOV_MM = 5.0
CENTER_MODE = "bounds"  # "bounds" or "centroid"
DEBUG_SCAN_PLANNER = False


@dataclass
class MosaicScanPlan:
    fov_locations: list
    y_length_mm: float
    y_pixels: int
    center_x: float
    center_y: float
    roi_bounds: tuple
    roi_size: tuple
    required_span: tuple
    tile_count: tuple
    candidate_count: int


def _coverage(tile_size, count, overlap):
    if count <= 1:
        return tile_size
    return tile_size + (count - 1) * tile_size * (1.0 - overlap)


def _fixed_tile_count(required_span, tile_size, overlap):
    if required_span <= tile_size:
        return 1
    step = tile_size * (1.0 - overlap)
    return int(math.ceil((required_span - tile_size) / step) + 1)


def _variable_y_geometry(required_span, y_step_um, max_y_fov_mm, overlap):
    if required_span <= 0:
        y_pixels = 1
        return y_pixels * y_step_um / 1000.0, y_pixels, 1

    if required_span <= max_y_fov_mm:
        count = 1
        target_y_length = required_span
    else:
        count = _fixed_tile_count(required_span, max_y_fov_mm, overlap)
        denom = 1.0 + (count - 1) * (1.0 - overlap)
        target_y_length = required_span / denom

    max_y_pixels = max(1, int(math.floor(max_y_fov_mm * 1000.0 / y_step_um)))
    y_pixels = max(1, int(math.ceil(target_y_length * 1000.0 / y_step_um)))
    y_pixels = min(y_pixels, max_y_pixels)
    y_length_mm = y_pixels * y_step_um / 1000.0
    return y_length_mm, y_pixels, count


def _centered_centers(center, count, tile_size, overlap):
    step = tile_size * (1.0 - overlap)
    offsets = (np.arange(count, dtype=float) - (count - 1) / 2.0) * step
    return center + offsets


def _clamp(value, low, high):
    return min(max(value, low), high)


def _dedupe_locations(locations):
    unique = {}
    for loc in locations:
        key = (loc["sample_id"], round(loc["x"], 3), round(loc["y"], 3))
        unique[key] = loc
    return list(unique.values())


def plan_mosaic_scan(
    sample_id,
    mm_polygons,
    x_fov_mm,
    y_step_um,
    stage_bounds,
    occupancy=ROI_OCCUPANCY_TARGET,
    overlap=FOV_OVERLAP,
    max_y_fov_mm=MAX_Y_FOV_MM,
    center_mode=CENTER_MODE,
):
    polygons = [Polygon(poly) for poly in mm_polygons if len(poly) >= 3]
    if not polygons:
        raise ValueError("No valid ROI polygons for mosaic scan planning.")

    roi = unary_union(polygons)
    min_x, min_y, max_x, max_y = roi.bounds
    roi_size_x = max_x - min_x
    roi_size_y = max_y - min_y

    if center_mode == "centroid":
        center = roi.centroid
        center_x, center_y = center.x, center.y
    elif center_mode == "bounds":
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
    else:
        raise ValueError(f"Unsupported center_mode: {center_mode}")

    required_x = roi_size_x / occupancy if occupancy > 0 else roi_size_x
    required_y = roi_size_y / occupancy if occupancy > 0 else roi_size_y

    nx = _fixed_tile_count(required_x, x_fov_mm, overlap)
    y_fov_mm, y_pixels, ny = _variable_y_geometry(required_y, y_step_um, max_y_fov_mm, overlap)

    x_centers = _centered_centers(center_x, nx, x_fov_mm, overlap)
    y_centers = _centered_centers(center_y, ny, y_fov_mm, overlap)

    x_min, x_max, y_min, y_max = stage_bounds
    if DEBUG_SCAN_PLANNER:
        print(
            "Scan planner ROI: "
            f"sample_id={sample_id}, bounds=(x:{min_x:.3f}-{max_x:.3f}, y:{min_y:.3f}-{max_y:.3f}), "
            f"size=({roi_size_x:.3f}, {roi_size_y:.3f}), "
            f"center_mode={center_mode}, center=({center_x:.3f}, {center_y:.3f})"
        )
        print(
            "Scan planner policy: "
            f"occupancy={occupancy:.3f}, overlap={overlap:.3f}, "
            f"x_fov={x_fov_mm:.3f}, max_y_fov={max_y_fov_mm:.3f}, y_step_um={y_step_um:.3f}, "
            f"stage_bounds=(x:{x_min:.3f}-{x_max:.3f}, y:{y_min:.3f}-{y_max:.3f})"
        )
        print(
            "Scan planner spans: "
            f"required=({required_x:.3f}, {required_y:.3f}), "
            f"tile_count=({nx}, {ny}), "
            f"coverage=({_coverage(x_fov_mm, nx, overlap):.3f}, {_coverage(y_fov_mm, ny, overlap):.3f}), "
            f"planned_y_fov={y_fov_mm:.3f}, y_pixels={y_pixels}"
        )
        print(
            "Scan planner raw centers: "
            f"x={np.round(x_centers, 3).tolist()}, "
            f"y={np.round(y_centers, 3).tolist()}"
        )

    new_locations = []
    candidate_count = 0
    for cx in x_centers:
        for cy in y_centers:
            candidate_count += 1
            safe_x = _clamp(cx, x_min, x_max)
            safe_y = _clamp(cy, y_min, y_max)
            clamped = (abs(safe_x - cx) > 1e-9) or (abs(safe_y - cy) > 1e-9)
            tile_poly = Polygon(
                [
                    (safe_x - x_fov_mm / 2.0, safe_y - y_fov_mm / 2.0),
                    (safe_x + x_fov_mm / 2.0, safe_y - y_fov_mm / 2.0),
                    (safe_x + x_fov_mm / 2.0, safe_y + y_fov_mm / 2.0),
                    (safe_x - x_fov_mm / 2.0, safe_y + y_fov_mm / 2.0),
                ]
            )
            intersects = tile_poly.intersects(roi)
            if DEBUG_SCAN_PLANNER:
                tx0, ty0, tx1, ty1 = tile_poly.bounds
                print(
                    "Scan planner candidate: "
                    f"#{candidate_count}, raw_center=({cx:.3f}, {cy:.3f}), "
                    f"safe_center=({safe_x:.3f}, {safe_y:.3f}), clamped={clamped}, "
                    f"tile_bounds=(x:{tx0:.3f}-{tx1:.3f}, y:{ty0:.3f}-{ty1:.3f}), "
                    f"intersects={intersects}"
                )
            if intersects:
                new_locations.append(
                    {"sample_id": sample_id, "x": round(safe_x, 3), "y": round(safe_y, 3)}
                )

    new_locations = _dedupe_locations(new_locations)
    if not new_locations:
        new_locations = [
            {
                "sample_id": sample_id,
                "x": round(_clamp(center_x, x_min, x_max), 3),
                "y": round(_clamp(center_y, y_min, y_max), 3),
            }
        ]
        if DEBUG_SCAN_PLANNER:
            print("Scan planner fallback: no intersecting FOV after clamping; using clamped ROI center.")
    if DEBUG_SCAN_PLANNER:
        print(
            "Scan planner accepted: "
            f"count={len(new_locations)}, locations={new_locations}"
        )
    return MosaicScanPlan(
        fov_locations=new_locations,
        y_length_mm=y_fov_mm,
        y_pixels=y_pixels,
        center_x=center_x,
        center_y=center_y,
        roi_bounds=(min_x, min_y, max_x, max_y),
        roi_size=(roi_size_x, roi_size_y),
        required_span=(required_x, required_y),
        tile_count=(nx, ny),
        candidate_count=candidate_count,
    )
