from dataclasses import dataclass

import numpy as np

from ActionTypes import AcqTypes
from CameraUi import effective_camera_sample_count


CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
    AcqTypes.FAST_VOLUME_CSCAN,
)

MOSAIC_DISPLAY_MODES = (
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)


@dataclass(frozen=True)
class DataShapeInfo:
    frame_count: int
    y_pixels: int
    repeat_count: int
    x_pixels: int
    z_pixels: int


def depth_pixels(ui, data=None, raw=False):
    if data is not None and getattr(data, "ndim", 0) >= 3:
        return int(data.shape[2])
    if raw:
        return effective_camera_sample_count(ui)
    return int(ui.DepthRange.value())


def x_pixels(ui, raw=False):
    """Number of X (A-line) pixels after applying AlineAVG.

    The GPU averages consecutive A-lines before the FFT, so the FFT output and
    all downstream display/save sizes use AlinesPerBline // AlineAVG. Raw
    spectral data (FFTDevice='None') is not averaged and keeps the full count.
    """
    if raw:
        return int(ui.AlinesPerBline.value())
    aline_avg = max(1, int(ui.AlineAVG.value()))
    return max(1, int(ui.AlinesPerBline.value()) // aline_avg)


def repeat_count(ui, data=None, raw=False, acq_mode=None, gpu_avg_count=1):
    configured_bline_avg = max(1, int(ui.BlineAVG.value()))
    if data is None or getattr(data, "ndim", 0) < 1:
        if raw:
            return configured_bline_avg
        return max(1, configured_bline_avg // max(1, int(gpu_avg_count)))

    frame_count = int(data.shape[0])
    if not raw:
        return frame_count

    if acq_mode in CSCAN_MODES + MOSAIC_DISPLAY_MODES:
        if frame_count > configured_bline_avg and frame_count % configured_bline_avg == 0:
            return configured_bline_avg
    return frame_count


def cscan_y_count(ui, data, raw=False):
    frame_count = int(data.shape[0])
    if not raw:
        return frame_count

    configured_bline_avg = max(1, int(ui.BlineAVG.value()))
    if frame_count > configured_bline_avg and frame_count % configured_bline_avg == 0:
        return frame_count // configured_bline_avg
    return frame_count


def data_shape(ui, data=None, raw=False, acq_mode=None, gpu_avg_count=1):
    frame_count = 0
    if data is not None and getattr(data, "ndim", 0) >= 1:
        frame_count = int(data.shape[0])
    return DataShapeInfo(
        frame_count=frame_count,
        y_pixels=cscan_y_count(ui, data, raw) if data is not None else max(1, int(ui.Ypixels.value())),
        repeat_count=repeat_count(ui, data, raw, acq_mode, gpu_avg_count),
        x_pixels=x_pixels(ui, raw),
        z_pixels=depth_pixels(ui, data, raw),
    )


def fast_volume_group_indices(y_pixels, micro_steps, bline_avg):
    """
    Source-frame index map for FastVolumeCscan acquisition order.

    Frames are captured in (block, repetition, step) order:
      - block b covers Y positions [b*MicroSteps, b*MicroSteps + steps_b), where
        steps_b = min(MicroSteps, remaining) so a partial last block is allowed;
      - inside each block, repetition r sweeps the block's steps, one camera
        frame per step (short pixel time).

    Returns an int64 array of shape [y_pixels, bline_avg] where element [y, r]
    is the linear frame index of Y position y at repetition r.
    """
    y_pixels = int(y_pixels)
    micro_steps = int(micro_steps)
    bline_avg = int(bline_avg)
    src = np.zeros((y_pixels, bline_avg), dtype=np.int64)
    linear = 0
    y0 = 0
    while y0 < y_pixels:
        steps = min(micro_steps, y_pixels - y0)
        for r in range(bline_avg):
            for s in range(steps):
                src[y0 + s, r] = linear + r * steps + s
        linear += steps * bline_avg
        y0 += steps
    return src


def fast_volume_ring_count(micro_steps):
    """
    Number of per-Y shared-memory slots needed for safe FastVolumeCscan dynamic
    acquisition.

    Each per-Y slot is written across one micro-block and is reused about
    `2*MicroSteps` Y-positions later when the ring has `2*MicroSteps + 2` slots,
    which gives the GPU roughly one full micro-block
    (``MicroSteps * BlineAVG`` frames) to drain the slot before it is
    overwritten.
    """
    micro_steps = max(1, int(micro_steps))
    return max(6, 2 * micro_steps + 2)


def fast_volume_frame_map(y_pixels, micro_steps, bline_avg):
    """
    Map each FastVolumeCscan acquisition frame index to its destination
    (Y position, repetition).

    Frames arrive in (block, repetition, step) order:
      - block b covers Y positions [b*MicroSteps, b*MicroSteps + steps_b), where
        steps_b = min(MicroSteps, remaining) so a partial last block is allowed;
      - inside each block, repetition r sweeps the block's steps, one camera
        frame per step.

    Returns two int64 arrays ``dest_y``, ``dest_r`` of length
    ``y_pixels * bline_avg``, where acquisition frame ``i`` belongs to
    Y position ``dest_y[i]`` at repetition ``dest_r[i]``.
    """
    y_pixels = int(y_pixels)
    micro_steps = int(micro_steps)
    bline_avg = int(bline_avg)
    total = y_pixels * bline_avg
    dest_y = np.zeros(total, dtype=np.int64)
    dest_r = np.zeros(total, dtype=np.int64)
    i = 0
    y0 = 0
    while y0 < y_pixels:
        steps = min(micro_steps, y_pixels - y0)
        for r in range(bline_avg):
            for s in range(steps):
                dest_y[i] = y0 + s
                dest_r[i] = r
                i += 1
        y0 += steps
    return dest_y, dest_r


def fast_volume_regroup(frames, y_pixels, micro_steps, bline_avg):
    """
    Regroup FastVolumeCscan frames [total_frames, X, Z] captured in
    (block, repetition, step) order into per-Y time series and average the
    repetitions.

    Returns [y_pixels, X, Z] after taking the mean over the bline_avg axis.
    """
    frames = np.asarray(frames)
    y_pixels = int(y_pixels)
    micro_steps = int(micro_steps)
    bline_avg = int(bline_avg)
    expected = y_pixels * bline_avg
    if frames.shape[0] != expected:
        raise ValueError(
            "FastVolumeCscan regroup expected "
            f"{expected} frames, got {frames.shape[0]}."
        )
    src = fast_volume_group_indices(y_pixels, micro_steps, bline_avg)
    grouped = frames[src.reshape(-1)].reshape(
        (y_pixels, bline_avg) + tuple(frames.shape[1:])
    )
    return grouped.mean(axis=1)
