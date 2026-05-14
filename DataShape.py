from dataclasses import dataclass

from ActionTypes import AcqTypes


CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
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
        return int(ui.NSamples_DH.value())
    return int(ui.DepthRange.value())


def x_pixels(ui):
    return int(ui.AlinesPerBline.value())


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
        x_pixels=x_pixels(ui),
        z_pixels=depth_pixels(ui, data, raw),
    )
