import gc
import os
import re
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\AMP Phase"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_TIMEPOINT_COUNTS = list(range(20, 401, 20))
DEFAULT_NOTCH_BAND_HZ = (46.0, 48.0)
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 10
DEFAULT_DYNAMIC_CHUNK_X = 96
DEFAULT_OUTPUT_DIR = None  # None saves figures beside each TIFF stack.
DEFAULT_SHOW_FIGURES = True
DEFAULT_SAVE_FIGURES = True

BLINE_NAME_RE = re.compile(
    r"^Bline-(?P<index>\d+)-Yrpt(?P<yrpt>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)


def iter_tiff_stacks(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_path}")

    paths = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}
    ]
    paths.sort(key=natural_bline_sort_key)
    if not paths:
        raise ValueError(f"No TIFF stacks found in: {input_path}")
    return paths


def natural_bline_sort_key(path):
    match = BLINE_NAME_RE.match(path.name)
    if match is None:
        return (1, path.name.lower())
    return (0, int(match.group("index")), int(match.group("yrpt")))


def read_amp_phase_tiff_stack(path):
    with TIFF.TiffFile(path) as tif:
        stack = np.stack([page.asarray() for page in tif.pages], axis=0)

    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2D/3D TIFF stack, got shape {stack.shape}")
    if stack.shape[-1] % 2 != 0:
        raise ValueError(
            "AMP+PHASE TIFF depth dimension must be even. "
            f"Got shape {stack.shape} from {path}"
        )

    z_pixels = stack.shape[-1] // 2
    amplitude = np.ascontiguousarray(stack[..., :z_pixels], dtype=np.float32)
    phase = np.ascontiguousarray(stack[..., z_pixels:], dtype=np.float32)
    del stack
    return amplitude, phase


def fft_bandstop_stack_axis0(stack, frame_rate_hz, stop_band_hz=None):
    stack = np.asarray(stack, dtype=np.float32)
    if stack.shape[0] < 2 or stop_band_hz is None:
        return stack.astype(np.float32, copy=True)

    low_hz, high_hz = sorted([float(stop_band_hz[0]), float(stop_band_hz[1])])
    if high_hz <= 0 or high_hz <= low_hz:
        return stack.astype(np.float32, copy=True)

    frequencies = np.fft.rfftfreq(stack.shape[0], d=1.0 / float(frame_rate_hz))
    spectrum = np.fft.rfft(stack, axis=0)
    spectrum[(frequencies >= low_hz) & (frequencies <= high_hz), :, :] = 0
    filtered = np.fft.irfft(spectrum, n=stack.shape[0], axis=0)
    return filtered.astype(np.float32, copy=False)


def gpu_style_dynamic_from_real_stack(stack, uniform_filter_size):
    filtered = uniform_filter1d(
        np.asarray(stack, dtype=np.float32),
        size=max(1, int(uniform_filter_size)),
        axis=0,
        mode="nearest",
    )
    return np.var(filtered, axis=0, dtype=np.float32).astype(np.float32, copy=False)


def complex_dynamic_from_filtered_amp_phase(amplitude_stack, phase_stack, uniform_filter_size):
    real_stack = amplitude_stack * np.cos(phase_stack)
    imag_stack = amplitude_stack * np.sin(phase_stack)
    filter_size = max(1, int(uniform_filter_size))
    real_filtered = uniform_filter1d(real_stack, size=filter_size, axis=0, mode="nearest")
    imag_filtered = uniform_filter1d(imag_stack, size=filter_size, axis=0, mode="nearest")
    mean_power = np.mean(
        real_filtered * real_filtered + imag_filtered * imag_filtered,
        axis=0,
        dtype=np.float32,
    )
    mean_real = np.mean(real_filtered, axis=0, dtype=np.float32)
    mean_imag = np.mean(imag_filtered, axis=0, dtype=np.float32)
    dynamic = mean_power - (mean_real * mean_real + mean_imag * mean_imag)
    return np.maximum(dynamic, np.float32(0.0)).astype(np.float32, copy=False)


def compute_dynamic_images_for_prefix(
    amplitude_stack,
    phase_stack,
    frame_count,
    frame_rate_hz,
    notch_band_hz,
    uniform_filter_size,
    chunk_x,
):
    frame_count = int(min(frame_count, amplitude_stack.shape[0]))
    if frame_count < 2:
        raise ValueError(f"Need at least 2 frames for dynamic processing, got {frame_count}")

    _, x_pixels, z_pixels = amplitude_stack.shape
    amplitude_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    complex_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        amplitude = amplitude_stack[:frame_count, x0:x1, :]
        phase = np.unwrap(phase_stack[:frame_count, x0:x1, :], axis=0).astype(
            np.float32,
            copy=False,
        )
        amplitude = fft_bandstop_stack_axis0(
            amplitude,
            frame_rate_hz=frame_rate_hz,
            stop_band_hz=notch_band_hz,
        )
        phase = fft_bandstop_stack_axis0(
            phase,
            frame_rate_hz=frame_rate_hz,
            stop_band_hz=notch_band_hz,
        )
        amplitude_dynamic[x0:x1, :] = gpu_style_dynamic_from_real_stack(
            amplitude,
            uniform_filter_size=uniform_filter_size,
        )
        complex_dynamic[x0:x1, :] = complex_dynamic_from_filtered_amp_phase(
            amplitude,
            phase,
            uniform_filter_size=uniform_filter_size,
        )

    return amplitude_dynamic, complex_dynamic


def initial_image_clim(images):
    values = np.concatenate(
        [image[np.isfinite(image)].reshape(-1) for image in images if np.size(image) > 0]
    )
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1.0, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def plot_dynamic_montage(images, frame_counts, title, output_path=None, show=True):
    columns = 5
    rows = int(np.ceil(len(images) / columns))
    vmin, vmax = initial_image_clim(images)
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows))
    axes = np.asarray(axes).reshape(rows, columns)

    last_im = None
    for idx, axis in enumerate(axes.flat):
        if idx >= len(images):
            axis.axis("off")
            continue
        last_im = axis.imshow(
            images[idx].T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"N={frame_counts[idx]}")
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Depth")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, label="Dynamic signal")
    fig.suptitle(title)
    if output_path is not None:
        fig.savefig(output_path, dpi=180)
        print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def output_directory_for_stack(stack_path, configured_output_dir):
    if configured_output_dir is not None:
        output_dir = Path(configured_output_dir)
    else:
        output_dir = stack_path.parent / "dynamic_timepoint_sufficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_stack(stack_path):
    print(f"Loading AMP+PHASE stack: {stack_path}")
    amplitude_stack, phase_stack = read_amp_phase_tiff_stack(stack_path)
    frame_total = amplitude_stack.shape[0]
    frame_counts = [count for count in DEFAULT_TIMEPOINT_COUNTS if count <= frame_total]
    if not frame_counts:
        frame_counts = [frame_total]

    print(
        f"Stack shape: frames={frame_total}, X={amplitude_stack.shape[1]}, "
        f"Z={amplitude_stack.shape[2]}"
    )
    amplitude_images = []
    complex_images = []

    for frame_count in frame_counts:
        print(f"Computing dynamic images from first {frame_count} frame(s)...")
        amplitude_dynamic, complex_dynamic = compute_dynamic_images_for_prefix(
            amplitude_stack=amplitude_stack,
            phase_stack=phase_stack,
            frame_count=frame_count,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
            uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )
        amplitude_images.append(amplitude_dynamic)
        complex_images.append(complex_dynamic)
        gc.collect()

    output_dir = output_directory_for_stack(stack_path, DEFAULT_OUTPUT_DIR)
    base_name = stack_path.stem
    amp_output = output_dir / f"{base_name}_amplitude_dynamic_timepoints.png"
    complex_output = output_dir / f"{base_name}_complex_dynamic_timepoints.png"

    plot_dynamic_montage(
        amplitude_images,
        frame_counts,
        title=f"{stack_path.name} amplitude dynamic vs timepoint count",
        output_path=amp_output if DEFAULT_SAVE_FIGURES else None,
        show=DEFAULT_SHOW_FIGURES,
    )
    plot_dynamic_montage(
        complex_images,
        frame_counts,
        title=f"{stack_path.name} complex dynamic vs timepoint count",
        output_path=complex_output if DEFAULT_SAVE_FIGURES else None,
        show=DEFAULT_SHOW_FIGURES,
    )

    del amplitude_stack
    del phase_stack
    del amplitude_images
    del complex_images
    gc.collect()


def main():
    stack_paths = iter_tiff_stacks(DEFAULT_INPUT_PATH)
    print(f"Found {len(stack_paths)} TIFF stack(s).")
    for stack_path in stack_paths:
        process_stack(stack_path)


if __name__ == "__main__":
    main()
