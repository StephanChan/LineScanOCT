import gc
import os
import re
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as TIFF


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\AMP Phase"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_NOTCH_BAND_HZ = (46.0, 48.0)
DEFAULT_BAND_WIDTH_HZ = 10.0
DEFAULT_AMPLITUDE_FREQ_RANGE_HZ = (0.0, 100.0)
DEFAULT_COMPLEX_FREQ_RANGE_HZ = (-100.0, 100.0)
DEFAULT_CHUNK_X = 96
DEFAULT_OUTPUT_DIR = None  # None saves into "frequency_band_power" beside each TIFF.
DEFAULT_SAVE_FIGURES = True
DEFAULT_SHOW_FIGURES = False
DEFAULT_FIGURE_DPI = 400

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


def make_bands(start_hz, stop_hz, band_width_hz):
    edges = np.arange(
        float(start_hz),
        float(stop_hz) + float(band_width_hz) * 0.5,
        float(band_width_hz),
        dtype=np.float32,
    )
    return [(float(edges[idx]), float(edges[idx + 1])) for idx in range(edges.size - 1)]


def fft_bandstop_stack_axis0(stack, frame_rate_hz, stop_band_hz=None):
    stack = np.asarray(stack, dtype=np.float32)
    if stack.shape[0] < 2 or stop_band_hz is None:
        return stack.astype(np.float32, copy=True)

    low_hz, high_hz = sorted([float(stop_band_hz[0]), float(stop_band_hz[1])])
    if high_hz <= 0 or high_hz <= low_hz:
        return stack.astype(np.float32, copy=True)

    frequencies = np.fft.fftfreq(stack.shape[0], d=1.0 / float(frame_rate_hz))
    spectrum = np.fft.fft(stack, axis=0)
    stop_mask = (np.abs(frequencies) >= low_hz) & (np.abs(frequencies) <= high_hz)
    spectrum[stop_mask, :, :] = 0
    filtered = np.fft.ifft(spectrum, axis=0).real
    return filtered.astype(np.float32, copy=False)


def normalized_power_spectrum_axis0(stack):
    stack = np.asarray(stack)
    centered = stack - np.mean(stack, axis=0, keepdims=True)
    spectrum = np.fft.fft(centered, axis=0) / stack.shape[0]
    power = np.abs(spectrum) ** 2
    return power.astype(np.float32, copy=False)


def band_power_from_power_spectrum(power, frequencies, bands):
    _, x_count, z_count = power.shape
    band_images = np.empty((len(bands), x_count, z_count), dtype=np.float32)
    for band_idx, (low_hz, high_hz) in enumerate(bands):
        if band_idx == len(bands) - 1:
            mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        else:
            mask = (frequencies >= low_hz) & (frequencies < high_hz)
        if not np.any(mask):
            band_images[band_idx, :, :] = 0.0
            continue
        band_images[band_idx, :, :] = np.sum(power[mask, :, :], axis=0, dtype=np.float32)
    return band_images


def compute_band_power_images(
    amplitude_stack,
    phase_stack,
    frame_rate_hz,
    notch_band_hz,
    amplitude_bands,
    complex_bands,
    chunk_x,
):
    frames, x_pixels, z_pixels = amplitude_stack.shape
    amplitude_band_images = np.zeros((len(amplitude_bands), x_pixels, z_pixels), dtype=np.float32)
    complex_band_images = np.zeros((len(complex_bands), x_pixels, z_pixels), dtype=np.float32)
    full_frequencies = np.fft.fftfreq(frames, d=1.0 / float(frame_rate_hz)).astype(np.float32)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        amplitude = amplitude_stack[:, x0:x1, :]
        phase = np.unwrap(phase_stack[:, x0:x1, :], axis=0).astype(np.float32, copy=False)

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
        complex_stack = (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)

        amplitude_power = normalized_power_spectrum_axis0(amplitude)
        complex_power = normalized_power_spectrum_axis0(complex_stack)
        amplitude_band_images[:, x0:x1, :] = band_power_from_power_spectrum(
            amplitude_power,
            full_frequencies,
            amplitude_bands,
        )
        complex_band_images[:, x0:x1, :] = band_power_from_power_spectrum(
            complex_power,
            full_frequencies,
            complex_bands,
        )
        print(f"Band power: processed X {x0}-{x1} / {x_pixels}")

    return amplitude_band_images, complex_band_images


def output_directory_for_stack(stack_path, configured_output_dir):
    if configured_output_dir is not None:
        output_dir = Path(configured_output_dir)
    else:
        output_dir = stack_path.parent / "frequency_band_power"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def montage_clim(images):
    values = images[np.isfinite(images)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1.0, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def plot_band_montage(
    band_images,
    bands,
    title,
    output_path,
    columns=5,
    cmap="magma",
    show=False,
):
    rows = int(np.ceil(len(bands) / columns))
    vmin, vmax = montage_clim(band_images)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.2 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(rows, columns)
    last_im = None

    for idx, axis in enumerate(axes.flat):
        if idx >= len(bands):
            axis.axis("off")
            continue
        low_hz, high_hz = bands[idx]
        last_im = axis.imshow(
            band_images[idx].T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"{low_hz:g} to {high_hz:g} Hz")
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Depth")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.86, label="Summed normalized power")
    fig.suptitle(title)
    if output_path is not None:
        fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI)
        print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def process_stack(stack_path):
    print(f"Loading AMP+PHASE stack: {stack_path}")
    amplitude_stack, phase_stack = read_amp_phase_tiff_stack(stack_path)
    print(
        f"Stack shape: frames={amplitude_stack.shape[0]}, X={amplitude_stack.shape[1]}, "
        f"Z={amplitude_stack.shape[2]}"
    )

    amplitude_bands = make_bands(
        DEFAULT_AMPLITUDE_FREQ_RANGE_HZ[0],
        DEFAULT_AMPLITUDE_FREQ_RANGE_HZ[1],
        DEFAULT_BAND_WIDTH_HZ,
    )
    complex_bands = make_bands(
        DEFAULT_COMPLEX_FREQ_RANGE_HZ[0],
        DEFAULT_COMPLEX_FREQ_RANGE_HZ[1],
        DEFAULT_BAND_WIDTH_HZ,
    )
    amplitude_band_images, complex_band_images = compute_band_power_images(
        amplitude_stack=amplitude_stack,
        phase_stack=phase_stack,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        amplitude_bands=amplitude_bands,
        complex_bands=complex_bands,
        chunk_x=DEFAULT_CHUNK_X,
    )

    output_dir = output_directory_for_stack(stack_path, DEFAULT_OUTPUT_DIR)
    base_name = stack_path.stem
    amp_output = output_dir / f"{base_name}_amplitude_frequency_band_power.png"
    complex_output = output_dir / f"{base_name}_complex_frequency_band_power.png"

    plot_band_montage(
        amplitude_band_images,
        amplitude_bands,
        title=f"{stack_path.name} amplitude band power",
        output_path=amp_output if DEFAULT_SAVE_FIGURES else None,
        columns=5,
        show=DEFAULT_SHOW_FIGURES,
    )
    plot_band_montage(
        complex_band_images,
        complex_bands,
        title=f"{stack_path.name} complex band power",
        output_path=complex_output if DEFAULT_SAVE_FIGURES else None,
        columns=5,
        show=DEFAULT_SHOW_FIGURES,
    )

    del amplitude_stack
    del phase_stack
    del amplitude_band_images
    del complex_band_images
    gc.collect()


def main():
    stack_paths = iter_tiff_stacks(DEFAULT_INPUT_PATH)
    print(f"Found {len(stack_paths)} TIFF stack(s).")
    for stack_path in stack_paths:
        process_stack(stack_path)


if __name__ == "__main__":
    main()
