# -*- coding: utf-8 -*-
"""Export SampleLocator USB images + user-drawn ROIs for future AI training.

On every successful locator action (scanner accepted, at least one sample
center generated) the main window calls :func:`export_sample_locator_dataset`,
which stores one flat, train-ready copy under ``<repo>/SampleLocatorData``:

    SampleLocatorData/
        images/       <run_id>__usb_region-XX-rowRR-colCC.png   (exact copy)
        labels/       <run_id>__usb_region-XX-rowRR-colCC.json   (all ROIs on image)
        previews/     <run_id>__usb_region-XX-rowRR-colCC.preview.png  (QC overlay)
        dataset_index.jsonl

Coordinates in the label files are the same pixel coordinates as the saved PNG
(GUI display px == Fiji horizontal-flip px; vertical = stage X,
horizontal = stage Y), so polygons can be overlaid directly onto the image.
"""

import datetime
import json
import os
import shutil

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - preview only
    cv2 = None

# <repo root>/SampleLocatorData
DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SampleLocatorData",
)

_JSON_SAFE = (str, int, float, bool, type(None))


def _json_safe(value):
    """Recursively convert numpy arrays/numeric types to plain JSON values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _bbox(polygon):
    try:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        if not xs or not ys:
            return None
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    except Exception:
        return None


def _tile_for_roi(roi, tile_records):
    """Match an ROI record to its tile record (1-based tile_index)."""
    try:
        roi_tile = int(roi.get("tile_index") or 0)
    except (TypeError, ValueError):
        roi_tile = 0
    for tile in tile_records:
        try:
            if int(tile.get("tile_index") or -1) == roi_tile:
                return tile
        except (TypeError, ValueError):
            continue
    # Fallback: ROI tile_index is 1-based index into the tile record list.
    if 1 <= roi_tile <= len(tile_records):
        return tile_records[roi_tile - 1]
    if len(tile_records) == 1:
        return tile_records[0]
    return None


def _source_image(tile):
    """Return BGR image array from path or in-memory numpy frame."""
    path = tile.get("image_path") or ""
    if path and os.path.isfile(path) and cv2 is not None:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    frame = tile.get("image")
    if frame is not None:
        return np.asarray(frame)
    return None


def _write_image(image, dst):
    if cv2 is None:
        return False
    return bool(cv2.imwrite(dst, image))


def _draw_preview(image, roi_records):
    """Return a copy of ``image`` with ROI polygons drawn on it (QC only)."""
    if cv2 is None or image is None:
        return None
    preview = image.copy()
    for roi in roi_records:
        polygon = roi.get("pixel_polygon") or []
        if len(polygon) < 3:
            continue
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        for x, y in polygon:
            cv2.circle(preview, (int(round(float(x))), int(round(float(y)))), 3, (0, 0, 255), -1)
    return preview
def export_sample_locator_dataset(
    tile_records,
    roi_records,
    calibration=None,
    root=None,
    created_at=None,
):
    """Copy every captured USB frame + its ROIs into ``SampleLocatorData``.

    tile_records: list of capture dicts (tile_index, row, col, stage_*,
                  image/image_path).
    roi_records : list of ROI dicts (sample_id, tile_index, pixel_polygon, ...).

    Returns a summary dict or None when there is nothing to export.
    """
    tile_records = list(tile_records or [])
    roi_records = list(roi_records or [])
    if not tile_records:
        return None
    root = os.path.abspath(root) if root else DATASET_ROOT
    if created_at is None:
        created_at = datetime.datetime.now()
    run_id = created_at.strftime("%Y%m%d_%H%M%S_%f")
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    previews_dir = os.path.join(root, "previews")
    index_path = os.path.join(root, "dataset_index.jsonl")
    for folder in (images_dir, labels_dir, previews_dir):
        os.makedirs(folder, exist_ok=True)

    rois_by_tile = {}
    for roi in roi_records:
        tile = _tile_for_roi(roi, tile_records)
        if tile is None:
            continue
        rois_by_tile.setdefault(id(tile), []).append(roi)

    written_images = 0
    written_rois = 0
    index_lines = []
    for tile in tile_records:
        source_name = os.path.basename(tile.get("image_path") or "")
        if source_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            stem = source_name[:-4]
        else:
            stem = "usb_region-{0:02d}-row{1:02d}-col{2:02d}".format(
                int(tile.get("tile_index") or 0),
                int(tile.get("row") or 0),
                int(tile.get("col") or 0),
            )
        image = _source_image(tile)
        if image is None:
            print("USB training export: no image available for", source_name or tile)
            continue
        height, width = image.shape[:2]
        base = "{0}__{1}".format(run_id, stem or "usb_frame")
        image_dst = os.path.join(images_dir, base + ".png")
        if not _write_image(image, image_dst):
            path = tile.get("image_path") or ""
            if path and os.path.isfile(path):
                shutil.copyfile(path, image_dst)
            else:
                print("USB training export: could not write image", image_dst)
                continue

        tile_rois = rois_by_tile.get(id(tile), [])
        cleaned_rois = []
        for roi in tile_rois:
            polygon = _json_safe(roi.get("pixel_polygon") or [])
            cleaned_rois.append(
                {
                    "sample_id": int(roi.get("sample_id") or 0),
                    "class_name": "sample",
                    "polygon": polygon,
                    "bbox": _bbox(polygon),
                    "fiji_pixel_polygon": _json_safe(roi.get("fiji_pixel_polygon") or []),
                    "stage_polygon": _json_safe(roi.get("stage_polygon") or []),
                    "center_x_mm": (
                        float(roi["center_x"]) if roi.get("center_x") is not None else None
                    ),
                    "center_y_mm": (
                        float(roi["center_y"]) if roi.get("center_y") is not None else None
                    ),
                }
            )

        label_dst = os.path.join(labels_dir, base + ".json")

        label = {
            "run_id": run_id,
            "created_at": created_at.isoformat(timespec="seconds"),
            "image_file": base + ".png",
            "image_width": int(width),
            "image_height": int(height),
            "coordinate_system": {
                "roi_vertices": "USB displayed image pixel coordinates",
                "image_vertical_axis": "stage X",
                "image_horizontal_axis": "stage Y",
                "image_right_direction": "smaller stage Y",
                "stage_units": "mm",
                "calibration": _json_safe(calibration),
            },
            "tile": {
                "tile_index": int(tile.get("tile_index") or 0),
                "row": int(tile.get("row") or 0),
                "col": int(tile.get("col") or 0),
                "stage_x_mm": (
                    float(tile["stage_x"]) if tile.get("stage_x") is not None else None
                ),
                "stage_y_mm": (
                    float(tile["stage_y"]) if tile.get("stage_y") is not None else None
                ),
                "stage_z_mm": (
                    float(tile["stage_z"]) if tile.get("stage_z") is not None else None
                ),
                "source_image_path": tile.get("image_path") or "",
            },
            "rois": cleaned_rois,
        }
        with open(label_dst, "w", encoding="utf-8") as file:
            json.dump(label, file, indent=2)

        preview_dst = os.path.join(previews_dir, base + ".preview.png")
        preview = _draw_preview(image, tile_rois)
        if preview is not None:
            _write_image(preview, preview_dst)

        index_lines.append(
            {
                "run_id": run_id,
                "created_at": created_at.isoformat(timespec="seconds"),
                "image": os.path.relpath(image_dst, root).replace("\\", "/"),
                "label": os.path.relpath(label_dst, root).replace("\\", "/"),
                "preview": os.path.relpath(preview_dst, root).replace("\\", "/"),
                "image_width": int(width),
                "image_height": int(height),
                "roi_count": len(cleaned_rois),
                "source_image_path": tile.get("image_path") or "",
            }
        )
        written_images += 1
        written_rois += len(cleaned_rois)

    if not written_images:
        return None
    with open(index_path, "a", encoding="utf-8") as file:
        for line in index_lines:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "images": written_images,
        "rois": written_rois,
        "root": root,
    }
    print(
        "USB training data exported: run {run_id} -> {root} "
        "({images} image(s), {rois} ROI(s)).".format(**summary)
    )
    return summary


if __name__ == "__main__":
    print("This module is imported by the OCT control software; no CLI.")

