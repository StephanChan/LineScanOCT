"""Camera UI helpers shared by acquisition, processing, and display code."""


CAMERA_DAHENG = "Daheng"
CAMERA_PHOTONFOCUS = "PhotonFocus"
CAMERA_HIK = "HiK"


def widget_value(ui, name, default=None):
    widget = getattr(ui, name, None)
    if widget is None:
        return default
    if hasattr(widget, "value"):
        return widget.value()
    if hasattr(widget, "currentText"):
        return widget.currentText()
    if hasattr(widget, "text"):
        return widget.text()
    return default


def current_camera_name(ui):
    return str(widget_value(ui, "Camera", ""))


def spectral_downsample(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return max(1, int(widget_value(ui, "SpectralDS_PF", 1)))
    if camera == CAMERA_HIK and hasattr(ui, "SpectralDS_HK"):
        return max(1, int(widget_value(ui, "SpectralDS_HK", 1)))
    return max(1, int(widget_value(ui, "SpectralDS_DH", 1)))


def raw_camera_sample_count(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return int(widget_value(ui, "NSamples_PF", 1024))
    if camera == CAMERA_HIK and hasattr(ui, "NSamples_HK"):
        return int(widget_value(ui, "NSamples_HK", 1024))
    return int(widget_value(ui, "NSamples_DH", 1024))


def effective_camera_sample_count(ui):
    raw_samples = raw_camera_sample_count(ui)
    ds = spectral_downsample(ui)
    if raw_samples % ds != 0:
        raise ValueError(
            "SpectralDS must divide the selected camera sample count: "
            f"raw_samples={raw_samples}, SpectralDS={ds}"
        )
    return raw_samples // ds


def camera_pixel_format(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return str(widget_value(ui, "PixelFormat_display_PF", "Mono12"))
    if camera == CAMERA_HIK and hasattr(ui, "PixelFormat_display_HK"):
        return str(widget_value(ui, "PixelFormat_display_HK", "Mono12"))
    return str(widget_value(ui, "PixelFormat_display_DH", "Mono12"))


def downsample_spectral_axis(data, ratio, axis):
    ratio = max(1, int(ratio))
    if ratio == 1:
        return data
    samples = int(data.shape[axis])
    if samples % ratio != 0:
        raise ValueError(
            f"SpectralDS={ratio} must divide spectral samples={samples}"
        )
    out_samples = samples // ratio
    moved = data.swapaxes(axis, -1)
    out_shape = moved.shape[:-1] + (out_samples, ratio)
    downsampled = moved.reshape(out_shape).mean(axis=-1)
    downsampled = downsampled.swapaxes(axis, -1)
    return downsampled.astype(data.dtype, copy=False)
