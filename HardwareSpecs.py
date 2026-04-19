# -*- coding: utf-8 -*-
"""
Central hardware specifications for the LineScanOCT control software.

Keep physical hardware constants here instead of scattering them through GUI,
waveform, and thread-control code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    angle_to_mm_ratio: float
    camera_step_divisor: float


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pixel_size_um: float
    max_height_px: int
    frame_rate_multiplier: float = 2.0


@dataclass(frozen=True)
class StageAxisSpec:
    axis_index: int
    init_speed_mm_s: float


@dataclass(frozen=True)
class LaserSpec:
    aline_frequency_hz: int


SYSTEM_MAGNIFICATION_4X = 2.5


OBJECTIVE_SPECS = {
    "4X": ObjectiveSpec(
        name="4X",
        angle_to_mm_ratio=2.094 / 1.19 * 1.25,
        camera_step_divisor=1.0,
    ),
    "5X": ObjectiveSpec(
        name="5X",
        angle_to_mm_ratio=2.094 / 1.19,
        camera_step_divisor=1.25,
    ),
    "10X": ObjectiveSpec(
        name="10X",
        angle_to_mm_ratio=2.094 / 2 / 1.19,
        camera_step_divisor=2.5,
    ),
    "20X": ObjectiveSpec(
        name="20X",
        angle_to_mm_ratio=2.094 / 1.19 / 4,
        camera_step_divisor=5.0,
    ),
}


CAMERA_SPECS = {
    "PhotonFocus": CameraSpec(name="PhotonFocus", pixel_size_um=9.0, max_height_px=1100),
    "XingTu": CameraSpec(name="XingTu", pixel_size_um=6.5, max_height_px=1024),
    "Daheng": CameraSpec(name="Daheng", pixel_size_um=9.0, max_height_px=1600),
}


STAGE_AXIS_SPECS = {
    "X": StageAxisSpec(axis_index=0, init_speed_mm_s=1.0),
    "Y": StageAxisSpec(axis_index=1, init_speed_mm_s=1.0),
    "Z": StageAxisSpec(axis_index=2, init_speed_mm_s=0.1),
}


LASER_SPECS = {
    "Axsun100k": LaserSpec(aline_frequency_hz=100000),
    "Thorlabs200k": LaserSpec(aline_frequency_hz=200000),
}


AODO_TRIGGER_OUT_PFI = "PFI3"
AODO_TRIGGER_IN_PFI = "PFI7"
AODO_DEFAULT_FRAME_RATE = 400
AODO_AO_VOLTAGE_MIN = -10.0
AODO_AO_VOLTAGE_MAX = 10.0


def get_objective_spec(name):
    return OBJECTIVE_SPECS.get(str(name))


def get_camera_spec(name):
    return CAMERA_SPECS.get(str(name))


def get_stage_axis_spec(axis):
    return STAGE_AXIS_SPECS[str(axis)]


def get_laser_spec(name):
    return LASER_SPECS.get(str(name))


def camera_step_size_um(camera_name, objective_name):
    camera = get_camera_spec(camera_name)
    objective = get_objective_spec(objective_name)
    if camera is None:
        raise KeyError(f"Unknown camera: {camera_name}")
    if objective is None:
        raise KeyError(f"Unknown objective: {objective_name}")
    return camera.pixel_size_um / SYSTEM_MAGNIFICATION_4X / objective.camera_step_divisor


def digital_line_mask(line_name):
    try:
        line_number = int(str(line_name).split("line")[-1])
    except (TypeError, ValueError):
        raise KeyError(f"Unsupported digital line: {line_name}")
    if line_number < 0 or line_number > 31:
        raise KeyError(f"Unsupported digital line: {line_name}")
    return 1 << line_number
