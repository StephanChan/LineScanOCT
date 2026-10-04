# -*- coding: utf-8 -*-
"""Affine stage / USB-camera coordinate calibration.

The user draws ROIs on the USB camera image, the stage is moved to each ROI's
predicted center, and the operator manually re-centers the sample under the live
OCT C-scan before pressing Stop. For every ROI we therefore know:

    camera pixel centroid (u, v)  ->  actual stage position (X, Y)

A least-squares 3x3 affine transform is fitted from those correspondences and
stored in config.ini so future locator sessions are automatically corrected.

Saving a model also *publishes* it for the running software (``set_active_affine``
/ ``current_affine``) and ``ThreadWeaver.calibrate_coordinates`` re-maps the plan
of the current folder onto the positions the operator aligned, so a fresh
calibration is applied in-session: no restart and no second locator run.
"""

import json
import os
import time

import numpy as np
from PyQt5.QtCore import QSettings

CONFIG_KEYS = (
    "UsbCamToStageA",
    "UsbCamToStageB",
    "UsbCamToStageC",
    "UsbCamToStageD",
    "UsbCamToStageE",
    "UsbCamToStageF",
)

MIN_CALIBRATION_POINTS = 5

# The affine the *running* software must use right now.  ``config.ini`` stays the
# persistent model (every process reads it when it starts), but a model fitted by
# a CoordinateCalibration run has to reach the current session immediately, so
# ``save_affine_to_config`` publishes it here and every reader goes through
# ``current_affine`` instead of keeping its own cached copy.  Without this the
# operator had to restart the software before the new calibration was used.
_ACTIVE_AFFINE = None


def set_active_affine(matrix):
    """Publish ``matrix`` as the calibration the running software uses now.

    Called whenever a model is fitted / saved so the current session (locator
    runs, USB region overlays, plan re-mapping) applies it right away instead of
    waiting for a restart.  ``None`` clears the in-process copy.
    """
    global _ACTIVE_AFFINE
    _ACTIVE_AFFINE = None if matrix is None else np.asarray(matrix, dtype=np.float64).copy()
    return _ACTIVE_AFFINE


def get_active_affine():
    """Return the in-process calibration, or None when none was published yet."""
    return None if _ACTIVE_AFFINE is None else _ACTIVE_AFFINE.copy()


def fit_pixel_to_stage_affine(points):
    """Fit an affine transform from USB-camera display pixels to actual stage XY.

    ``points``: list of dicts with keys pixel_x, pixel_y, actual_x, actual_y.
    Returns (3x3 matrix, stats dict). Raises ValueError below MIN_CALIBRATION_POINTS.
    """
    valid = [
        point
        for point in points
        if point.get("pixel_x") is not None and point.get("pixel_y") is not None
    ]
    if len(valid) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            f"At least {MIN_CALIBRATION_POINTS} calibration points are required "
            f"(got {len(valid)})."
        )

    design = np.asarray(
        [
            [float(point["pixel_x"]), float(point["pixel_y"]), 1.0]
            for point in valid
        ],
        dtype=np.float64,
    )
    target_x = np.asarray([float(point["actual_x"]) for point in valid], dtype=np.float64)
    target_y = np.asarray([float(point["actual_y"]) for point in valid], dtype=np.float64)

    coeff_x, _, _, _ = np.linalg.lstsq(design, target_x, rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(design, target_y, rcond=None)

    matrix = np.asarray(
        [
            [coeff_x[0], coeff_x[1], coeff_x[2]],
            [coeff_y[0], coeff_y[1], coeff_y[2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    predicted = design @ np.asarray(
        [
            [coeff_x[0], coeff_y[0]],
            [coeff_x[1], coeff_y[1]],
            [coeff_x[2], coeff_y[2]],
        ],
        dtype=np.float64,
    )
    errors_mm = np.hypot(
        predicted[:, 0] - target_x,
        predicted[:, 1] - target_y,
    )
    rms_mm = float(np.sqrt(np.mean(errors_mm ** 2)))
    max_mm = float(np.max(errors_mm))

    per_point = []
    for point, error_mm in zip(valid, errors_mm):
        per_point.append(
            {
                "sample_id": int(point.get("sample_id", 0)),
                "pixel_x": float(point["pixel_x"]),
                "pixel_y": float(point["pixel_y"]),
                "predicted_x": float(point.get("predicted_x", np.nan)),
                "predicted_y": float(point.get("predicted_y", np.nan)),
                "actual_x": float(point["actual_x"]),
                "actual_y": float(point["actual_y"]),
                "residual_mm": float(error_mm),
            }
        )

    stats = {
        "count": len(valid),
        "rms_mm": rms_mm,
        "max_mm": max_mm,
        "per_point": per_point,
    }
    return matrix, stats


def save_affine_to_config(matrix, config_path="config.ini"):
    """Persist the 6 affine coefficients of the 3x3 matrix to config.ini."""
    settings = QSettings(config_path, QSettings.IniFormat)
    settings.setValue("UsbCamCalibMethod", "affine")
    settings.setValue("UsbCamToStageA", float(matrix[0, 0]))
    settings.setValue("UsbCamToStageB", float(matrix[0, 1]))
    settings.setValue("UsbCamToStageC", float(matrix[0, 2]))
    settings.setValue("UsbCamToStageD", float(matrix[1, 0]))
    settings.setValue("UsbCamToStageE", float(matrix[1, 1]))
    settings.setValue("UsbCamToStageF", float(matrix[1, 2]))
    settings.sync()
    # Saving == applying: publish the model in-process too, so the current
    # session (next locator run, USB overlays, plan re-mapping) uses it without a
    # software restart.
    set_active_affine(matrix)


def load_affine_from_config(config_path="config.ini"):
    """Return the fitted 3x3 affine matrix from config.ini, or None if absent."""
    settings = QSettings(config_path, QSettings.IniFormat)
    if settings.value("UsbCamCalibMethod", "") != "affine":
        return None
    values = []
    for key in CONFIG_KEYS:
        raw = settings.value(key, None)
        if raw is None:
            return None
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None
    return np.asarray(
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def current_affine(config_path="config.ini"):
    """Return the affine the running software must use, without a restart.

    ``config.ini`` wins: it is the persistent model and is re-read on every call,
    so a freshly fitted (and saved) model - or even a hand-edited file - is picked
    up immediately.  The in-process copy published by :func:`set_active_affine`
    is used as the fallback when the file holds no fitted model.  Every reader
    (locator sessions, USB region overlays, the plan correction applied by
    ``ThreadWeaver.calibrate_coordinates``) must go through this function so the
    whole running session shares exactly one calibration.
    """
    matrix = load_affine_from_config(config_path)
    if matrix is not None:
        set_active_affine(matrix)
        return matrix
    return get_active_affine()


def save_calibration_points_report(points, matrix, stats, folder_path):
    """Write a machine-readable calibration report to folder_path.

    Returns the full path of the written JSON file.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = os.path.join(
        folder_path, f"coordinate_calibration_{timestamp}.json"
    )
    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "affine_pixel_to_stage",
        "matrix": matrix.tolist(),
        "stats": {
            "count": int(stats["count"]),
            "rms_mm": float(stats["rms_mm"]),
            "max_mm": float(stats["max_mm"]),
            "per_point": stats["per_point"],
        },
        "points": [
            {
                key: (
                    float(value)
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                    else value
                )
                for key, value in point.items()
            }
            for point in points
        ],
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"Coordinate calibration report saved: {report_path}")
    return report_path

