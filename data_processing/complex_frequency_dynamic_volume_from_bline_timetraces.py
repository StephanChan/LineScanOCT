import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import tifffile as TIFF


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_DIR = r"E:\IOCTData\BreastCancerMice\0630\Z21_29"
DEFAULT_FILENAME_GLOB = r"Cscan-2-Bline-*-Yrpt100-X1104-Z73.tif"
DEFAULT_FRAME_RATE_HZ = 50.0
DEFAULT_DURATION_SECONDS = 2.0
DEFAULT_EXPECTED_FRAME_COUNT = 100  # 2 s x 50 Hz

# Symmetric complex-frequency bands. Each band uses both +f and -f.
# Edit this list directly to test different ranges.
DEFAULT_SYMMETRIC_BANDS_HZ = [

    (0.0, 25.0)
]

# Fixed normalization/display constants. Change these manually when you want
# a different visualization scaling without touching the processing logic.
DEFAULT_OUTPUT_NORMALIZATION_CONSTANT = 1.0
DEFAULT_MEAN_BLINE_CLIM = (0.0, 3000.0)
DEFAULT_ENFACE_CLIM = (0.0, 1500.0)

# Paper-style HSV rendering:
# H = mean frequency, S = spectral bandwidth, V = total dynamic magnitude.
# For complex data we fold the spectrum onto |f| first, so both positive and
# negative frequencies contribute to the same nonnegative frequency axis.
DEFAULT_HUE_FREQUENCY_RANGE_HZ = (0.0, 15.0)
DEFAULT_SATURATION_BANDWIDTH_RANGE_HZ = (0.0, 8.0)
DEFAULT_VALUE_DYNAMIC_RANGE = (0.0, 500.0)
DEFAULT_VALUE_GAMMA = 1.0

DEFAULT_OUTPUT_DIR = None  # None saves into input_dir / "complex_frequency_dynamic_volume".
DEFAULT_SAVE_DPI = 300
DEFAULT_SHOW_FIGURES = False
DEFAULT_REUSE_SAVED_HSV_METRICS = True

# Keep False for Spyder/IPython. Set True only when running from a terminal.
USE_COMMAND_LINE_ARGS = False

BLINE_INDEX_RE = re.compile(r"Bline-(?P<index>\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load matched AMP+PHASE B-line time-trace TIFF stacks, reconstruct the "
            "complex field, compute symmetric frequency-band complex dynamic "
            "(std-equivalent) volumes, and save the results."
        )
    )
    parser.add_argument("input_dir", help="Directory that contains the AMP+PHASE TIFF stacks.")
    parser.add_argument(
        "--glob",
        default=DEFAULT_FILENAME_GLOB,
        help="Filename glob used to select only the related B-line time-trace TIFF stacks.",
    )
    parser.add_argument(
        "--frame-rate-hz",
        type=float,
        default=DEFAULT_FRAME_RATE_HZ,
        help="Acquisition frame rate in Hz.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Duration to use from each time-trace stack.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Optional output directory. Defaults to input_dir / complex_frequency_dynamic_volume.",
    )
    return parser.parse_args()


def natural_bline_sort_key(path):
    match = BLINE_INDEX_RE.search(path.name)
    if match is None:
        return (1, path.name.lower())
    return (0, int(match.group("index")), path.name.lower())


def iter_matching_stacks(input_dir, filename_glob):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    paths = [path for path in input_dir.glob(filename_glob) if path.is_file()]
    paths.sort(key=natural_bline_sort_key)
    if not paths:
        raise ValueError(
            f"No TIFF stacks matched glob '{filename_glob}' in directory: {input_dir}"
        )
    return paths


def output_directory(input_dir, configured_output_dir):
    if configured_output_dir is not None and str(configured_output_dir).strip() != "":
        out_dir = Path(configured_output_dir)
    else:
        out_dir = Path(input_dir) / "complex_frequency_dynamic_volume"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_amp_phase_stack(path):
    with TIFF.TiffFile(path) as tif:
        stack = np.stack([page.asarray() for page in tif.pages], axis=0)

    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2D/3D TIFF stack, got shape {stack.shape} from {path}")
    if stack.shape[-1] % 2 != 0:
        raise ValueError(
            "AMP+PHASE TIFF depth dimension must be even. "
            f"Got shape {stack.shape} from {path}"
        )

    z_pixels = stack.shape[-1] // 2
    amplitude = np.ascontiguousarray(stack[..., :z_pixels], dtype=np.float32)
    phase = np.ascontiguousarray(stack[..., z_pixels:], dtype=np.float32)
    return amplitude, phase


def reconstruct_complex_stack(amplitude_stack, phase_stack):
    return (
        np.asarray(amplitude_stack, dtype=np.float32)
        * np.exp(1j * np.asarray(phase_stack, dtype=np.float32))
    ).astype(np.complex64, copy=False)


def limit_stack_duration(complex_stack, frame_rate_hz, duration_seconds):
    if duration_seconds is None:
        return np.ascontiguousarray(complex_stack, dtype=np.complex64)

    max_frames = int(round(float(frame_rate_hz) * float(duration_seconds)))
    if max_frames < 2:
        raise ValueError("Duration is too short. Need at least 2 frames.")
    if complex_stack.shape[0] <= max_frames:
        return np.ascontiguousarray(complex_stack, dtype=np.complex64)
    return np.ascontiguousarray(complex_stack[:max_frames], dtype=np.complex64)


def validate_frequency_bands(frame_rate_hz, bands_hz):
    nyquist_hz = 0.5 * float(frame_rate_hz)
    validated = []
    for low_hz, high_hz in bands_hz:
        low_hz = float(low_hz)
        high_hz = float(high_hz)
        if low_hz < 0 or high_hz <= low_hz or high_hz > nyquist_hz + 1e-6:
            raise ValueError(
                f"Invalid symmetric band ({low_hz}, {high_hz}) Hz for frame rate "
                f"{frame_rate_hz:g} Hz (Nyquist {nyquist_hz:g} Hz)."
            )
        validated.append((low_hz, high_hz))
    return validated


def symmetric_band_mask(frequencies_hz, band_hz, is_last_band=False):
    low_hz, high_hz = band_hz
    abs_frequencies_hz = np.abs(frequencies_hz)
    if is_last_band:
        return (abs_frequencies_hz >= low_hz) & (abs_frequencies_hz <= high_hz)
    return (abs_frequencies_hz >= low_hz) & (abs_frequencies_hz < high_hz)


def complex_std_image_from_band(complex_stack, frequencies_hz, band_hz, is_last_band=False):
    centered = complex_stack - np.mean(complex_stack, axis=0, keepdims=True, dtype=np.complex64)
    spectrum = np.fft.fft(centered, axis=0)
    mask = symmetric_band_mask(frequencies_hz, band_hz, is_last_band=is_last_band)
    masked_spectrum = np.zeros_like(spectrum)
    masked_spectrum[mask, :, :] = spectrum[mask, :, :]
    band_limited = np.fft.ifft(masked_spectrum, axis=0).astype(np.complex64, copy=False)

    mean_power = np.mean(np.abs(band_limited) ** 2, axis=0, dtype=np.float32)
    mean_complex = np.mean(band_limited, axis=0, dtype=np.complex64)
    variance = mean_power - np.abs(mean_complex) ** 2
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32, copy=False)


def normalized_complex_power_spectrum(complex_stack):
    centered = complex_stack - np.mean(complex_stack, axis=0, keepdims=True, dtype=np.complex64)
    spectrum = np.fft.fft(centered, axis=0) / centered.shape[0]
    power = np.abs(spectrum) ** 2
    return power.astype(np.float32, copy=False)


def average_complex_power_spectrum(complex_stack):
    power = normalized_complex_power_spectrum(complex_stack)
    return np.mean(power, axis=(1, 2), dtype=np.float64).astype(np.float32)


def fold_complex_power_spectrum_to_abs_frequency(power_spectrum, frequencies_hz):
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float32)
    abs_frequencies_hz = np.abs(frequencies_hz)
    unique_abs_frequencies_hz = np.unique(abs_frequencies_hz)
    unique_abs_frequencies_hz.sort()

    folded_power = np.zeros(
        (unique_abs_frequencies_hz.size,) + power_spectrum.shape[1:],
        dtype=np.float32,
    )
    for idx, abs_frequency_hz in enumerate(unique_abs_frequencies_hz):
        mask = np.isclose(abs_frequencies_hz, abs_frequency_hz)
        folded_power[idx, :, :] = np.sum(power_spectrum[mask, :, :], axis=0, dtype=np.float32)
    return unique_abs_frequencies_hz.astype(np.float32), folded_power


def compute_hsv_metric_images(complex_stack, frame_rate_hz):
    power_spectrum = normalized_complex_power_spectrum(complex_stack)
    frequencies_hz = np.fft.fftfreq(
        complex_stack.shape[0],
        d=1.0 / float(frame_rate_hz),
    ).astype(np.float32)
    abs_frequencies_hz, folded_power = fold_complex_power_spectrum_to_abs_frequency(
        power_spectrum,
        frequencies_hz,
    )

    total_power = np.sum(folded_power, axis=0, dtype=np.float32)
    total_power_safe = np.maximum(total_power, np.float32(1e-12))

    frequency_axis = abs_frequencies_hz[:, np.newaxis, np.newaxis]
    mean_frequency_hz = (
        np.sum(folded_power * frequency_axis, axis=0, dtype=np.float32) / total_power_safe
    ).astype(np.float32, copy=False)

    centered_frequency2 = (frequency_axis - mean_frequency_hz[np.newaxis, :, :]) ** 2
    bandwidth_hz = np.sqrt(
        np.maximum(
            np.sum(folded_power * centered_frequency2, axis=0, dtype=np.float32) / total_power_safe,
            0.0,
        )
    ).astype(np.float32, copy=False)

    dynamic_std = np.sqrt(np.maximum(total_power, 0.0)).astype(np.float32, copy=False)
    return mean_frequency_hz, bandwidth_hz, dynamic_std


def normalize_to_unit_interval(image, value_range, gamma=1.0):
    low_value, high_value = float(value_range[0]), float(value_range[1])
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        raise ValueError(f"Invalid normalization range: {value_range}")

    normalized = (np.asarray(image, dtype=np.float32) - low_value) / (high_value - low_value)
    normalized = np.clip(normalized, 0.0, 1.0)

    gamma = float(gamma)
    if np.isfinite(gamma) and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        normalized = normalized ** gamma
    return normalized.astype(np.float32, copy=False)


def hsv_volume_from_metric_volumes(mean_frequency_volume_hz, bandwidth_volume_hz, dynamic_std_volume):
    hue = normalize_to_unit_interval(mean_frequency_volume_hz, DEFAULT_HUE_FREQUENCY_RANGE_HZ)
    saturation = normalize_to_unit_interval(
        bandwidth_volume_hz,
        DEFAULT_SATURATION_BANDWIDTH_RANGE_HZ,
    )
    value = normalize_to_unit_interval(
        dynamic_std_volume,
        DEFAULT_VALUE_DYNAMIC_RANGE,
        gamma=DEFAULT_VALUE_GAMMA,
    )
    hsv_volume = np.stack([hue, saturation, value], axis=-1).astype(np.float32, copy=False)
    rgb_volume = hsv_to_rgb(hsv_volume).astype(np.float32, copy=False)
    rgb_uint8_volume = np.clip(np.round(rgb_volume * 255.0), 0, 255).astype(np.uint8)
    return hsv_volume, rgb_volume, rgb_uint8_volume


def save_volume_tiff(volume, output_path):
    TIFF.imwrite(output_path, np.asarray(volume, dtype=np.float32))
    print(f"Saved TIFF volume: {output_path}")


def save_rgb_volume_tiff(rgb_uint8_volume, output_path):
    TIFF.imwrite(output_path, np.asarray(rgb_uint8_volume, dtype=np.uint8), photometric="rgb")
    print(f"Saved RGB TIFF volume: {output_path}")


def load_volume_tiff(path, dtype=np.float32):
    volume = TIFF.imread(path)
    return np.asarray(volume, dtype=dtype)


def safe_normalize(image_or_volume, normalization_constant):
    normalization_constant = float(normalization_constant)
    if not np.isfinite(normalization_constant) or normalization_constant == 0:
        raise ValueError(
            "DEFAULT_OUTPUT_NORMALIZATION_CONSTANT must be finite and non-zero."
        )
    return np.asarray(image_or_volume, dtype=np.float32) / normalization_constant


def save_quicklook_figure(mean_bline_image, enface_image, output_path, title):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
    )

    mean_im = axes[0].imshow(
        mean_bline_image.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=DEFAULT_MEAN_BLINE_CLIM[0],
        vmax=DEFAULT_MEAN_BLINE_CLIM[1],
    )
    axes[0].set_title("Mean B-line")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Depth")
    fig.colorbar(mean_im, ax=axes[0], shrink=0.92)

    enface_im = axes[1].imshow(
        enface_image,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=DEFAULT_ENFACE_CLIM[0],
        vmax=DEFAULT_ENFACE_CLIM[1],
    )
    axes[1].set_title("Mean over depth")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("B-line index")
    fig.colorbar(enface_im, ax=axes[1], shrink=0.92)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if DEFAULT_SHOW_FIGURES:
        plt.show(block=False)
    plt.close(fig)


def save_rgb_quicklook_figure(mean_bline_rgb, enface_rgb, output_path, title):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
    )

    axes[0].imshow(mean_bline_rgb.transpose(1, 0, 2), aspect="auto", origin="lower")
    axes[0].set_title("RGB mean B-line")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Depth")

    axes[1].imshow(enface_rgb, aspect="auto", origin="lower")
    axes[1].set_title("RGB mean over depth")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("B-line index")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if DEFAULT_SHOW_FIGURES:
        plt.show(block=False)
    plt.close(fig)


def save_global_spectrum_figure(frequencies_hz, mean_spectrum, output_path, title):
    shifted_frequencies_hz = np.fft.fftshift(frequencies_hz)
    shifted_spectrum = np.fft.fftshift(mean_spectrum)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(shifted_frequencies_hz, shifted_spectrum, color="black", linewidth=1.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean complex power")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if DEFAULT_SHOW_FIGURES:
        plt.show(block=False)
    plt.close(fig)


def save_summary_csv(records, output_path):
    fieldnames = [
        "bline_index",
        "stack_name",
        "band_label",
        "band_low_hz",
        "band_high_hz",
        "raw_volume_mean",
        "raw_volume_std",
        "raw_volume_min",
        "raw_volume_max",
        "normalized_volume_mean",
        "normalized_volume_std",
        "normalized_volume_min",
        "normalized_volume_max",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV summary: {output_path}")


def save_hsv_metric_summary_csv(records, output_path):
    fieldnames = [
        "bline_index",
        "stack_name",
        "mean_frequency_mean_hz",
        "mean_frequency_std_hz",
        "bandwidth_mean_hz",
        "bandwidth_std_hz",
        "dynamic_std_mean",
        "dynamic_std_std",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV summary: {output_path}")


def hsv_metric_cache_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "mean_frequency": output_dir / "hsv_mean_frequency_hz.tif",
        "bandwidth": output_dir / "hsv_bandwidth_hz.tif",
        "dynamic_std": output_dir / "hsv_dynamic_std.tif",
        "summary_csv": output_dir / "complex_hsv_metric_summary.csv",
        "metadata_json": output_dir / "hsv_metric_cache_metadata.json",
    }


def build_hsv_cache_metadata(input_dir, filename_glob, frame_rate_hz, duration_seconds):
    return {
        "input_dir": str(Path(input_dir)),
        "filename_glob": str(filename_glob),
        "frame_rate_hz": float(frame_rate_hz),
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "expected_frame_count": int(DEFAULT_EXPECTED_FRAME_COUNT),
    }


def save_hsv_cache_metadata(output_dir, metadata):
    metadata_path = hsv_metric_cache_paths(output_dir)["metadata_json"]
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved cache metadata: {metadata_path}")


def load_hsv_cache_metadata(output_dir):
    metadata_path = hsv_metric_cache_paths(output_dir)["metadata_json"]
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def saved_hsv_metric_cache_exists(output_dir, expected_metadata=None):
    paths = hsv_metric_cache_paths(output_dir)
    files_exist = (
        paths["mean_frequency"].exists()
        and paths["bandwidth"].exists()
        and paths["dynamic_std"].exists()
    )
    if not files_exist:
        return False
    if expected_metadata is None:
        return True

    cached_metadata = load_hsv_cache_metadata(output_dir)
    if cached_metadata is None:
        print("HSV metric cache metadata is missing; recomputing from TIFF stacks.")
        return False
    if cached_metadata != expected_metadata:
        print("HSV metric cache metadata does not match current input settings; recomputing.")
        return False
    return True


def load_saved_hsv_metric_cache(output_dir):
    paths = hsv_metric_cache_paths(output_dir)
    mean_frequency_volume_hz = load_volume_tiff(paths["mean_frequency"], dtype=np.float32)
    bandwidth_volume_hz = load_volume_tiff(paths["bandwidth"], dtype=np.float32)
    dynamic_std_volume = load_volume_tiff(paths["dynamic_std"], dtype=np.float32)

    if (
        mean_frequency_volume_hz.shape != bandwidth_volume_hz.shape
        or mean_frequency_volume_hz.shape != dynamic_std_volume.shape
    ):
        raise ValueError(
            "Saved HSV metric volumes do not share the same shape. "
            "Please delete the cached HSV files and rerun."
        )

    print("Loaded saved HSV metric volumes; skipping TIFF-stack reprocessing.")
    return mean_frequency_volume_hz, bandwidth_volume_hz, dynamic_std_volume


def extract_bline_index(path):
    match = BLINE_INDEX_RE.search(path.name)
    if match is None:
        return np.nan
    return int(match.group("index"))


def band_label(low_hz, high_hz):
    return f"{low_hz:g}_to_{high_hz:g}_Hz"


def process_single_stack(path, frame_rate_hz, duration_seconds, bands_hz):
    amplitude_stack, phase_stack = load_amp_phase_stack(path)
    complex_stack = reconstruct_complex_stack(amplitude_stack, phase_stack)
    complex_stack = limit_stack_duration(complex_stack, frame_rate_hz, duration_seconds)

    if complex_stack.shape[0] != DEFAULT_EXPECTED_FRAME_COUNT:
        print(
            f"Warning: {path.name} uses {complex_stack.shape[0]} frames after duration limit; "
            f"expected {DEFAULT_EXPECTED_FRAME_COUNT}."
        )

    frequencies_hz = np.fft.fftfreq(complex_stack.shape[0], d=1.0 / float(frame_rate_hz)).astype(np.float32)
    mean_spectrum = average_complex_power_spectrum(complex_stack)
    mean_frequency_image_hz, bandwidth_image_hz, dynamic_std_image = compute_hsv_metric_images(
        complex_stack,
        frame_rate_hz=frame_rate_hz,
    )

    dynamic_images = []
    for band_idx, band_hz in enumerate(bands_hz):
        dynamic_images.append(
            complex_std_image_from_band(
                complex_stack,
                frequencies_hz,
                band_hz,
                is_last_band=(band_idx == len(bands_hz) - 1),
            )
        )

    return {
        "complex_stack": complex_stack,
        "frequencies_hz": frequencies_hz,
        "mean_spectrum": mean_spectrum,
        "mean_frequency_image_hz": mean_frequency_image_hz,
        "bandwidth_image_hz": bandwidth_image_hz,
        "dynamic_std_image": dynamic_std_image,
        "dynamic_images": dynamic_images,
    }


def main():
    if USE_COMMAND_LINE_ARGS:
        args = parse_args()
        input_dir = args.input_dir
        filename_glob = args.glob
        frame_rate_hz = args.frame_rate_hz
        duration_seconds = args.duration_seconds
        configured_output_dir = args.output_dir
    else:
        input_dir = DEFAULT_INPUT_DIR
        filename_glob = DEFAULT_FILENAME_GLOB
        frame_rate_hz = DEFAULT_FRAME_RATE_HZ
        duration_seconds = DEFAULT_DURATION_SECONDS
        configured_output_dir = DEFAULT_OUTPUT_DIR

    out_dir = output_directory(input_dir, configured_output_dir)
    expected_cache_metadata = build_hsv_cache_metadata(
        input_dir,
        filename_glob,
        frame_rate_hz,
        duration_seconds,
    )
    cache_paths = hsv_metric_cache_paths(out_dir)

    if DEFAULT_REUSE_SAVED_HSV_METRICS and saved_hsv_metric_cache_exists(
        out_dir,
        expected_metadata=expected_cache_metadata,
    ):
        mean_frequency_volume_hz, bandwidth_volume_hz, dynamic_std_volume = load_saved_hsv_metric_cache(
            out_dir
        )
        hsv_volume, rgb_volume, rgb_uint8_volume = hsv_volume_from_metric_volumes(
            mean_frequency_volume_hz,
            bandwidth_volume_hz,
            dynamic_std_volume,
        )
        save_volume_tiff(hsv_volume, out_dir / "hsv_metric_volume_float.tif")
        save_rgb_volume_tiff(rgb_uint8_volume, out_dir / "hsv_rendered_rgb_volume.tif")

        rgb_mean_bline = np.mean(rgb_volume, axis=0, dtype=np.float32)
        rgb_enface = np.mean(rgb_volume, axis=2, dtype=np.float32)
        save_rgb_quicklook_figure(
            rgb_mean_bline,
            rgb_enface,
            out_dir / "hsv_rendered_rgb_quicklook.png",
            title="HSV-rendered complex dynamic volume",
        )
        print(
            "Finished HSV rerender from saved metric volumes. "
            f"Using cached files: {cache_paths['mean_frequency'].name}, "
            f"{cache_paths['bandwidth'].name}, {cache_paths['dynamic_std'].name}"
        )
        return

    bands_hz = validate_frequency_bands(frame_rate_hz, DEFAULT_SYMMETRIC_BANDS_HZ)
    stack_paths = iter_matching_stacks(input_dir, filename_glob)

    print(f"Matched {len(stack_paths)} stacks with glob: {filename_glob}")

    first_result = process_single_stack(
        stack_paths[0],
        frame_rate_hz=frame_rate_hz,
        duration_seconds=duration_seconds,
        bands_hz=bands_hz,
    )

    bline_count = len(stack_paths)
    x_pixels = first_result["dynamic_images"][0].shape[0]
    z_pixels = first_result["dynamic_images"][0].shape[1]
    band_count = len(bands_hz)

    band_volumes = np.zeros((band_count, bline_count, x_pixels, z_pixels), dtype=np.float32)
    global_mean_spectrum = np.zeros_like(first_result["mean_spectrum"], dtype=np.float64)
    summary_records = []
    hsv_summary_records = []
    mean_frequency_volume_hz = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)
    bandwidth_volume_hz = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)
    dynamic_std_volume = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)

    for stack_idx, stack_path in enumerate(stack_paths):
        if stack_idx == 0:
            result = first_result
        else:
            result = process_single_stack(
                stack_path,
                frame_rate_hz=frame_rate_hz,
                duration_seconds=duration_seconds,
                bands_hz=bands_hz,
            )

        global_mean_spectrum += result["mean_spectrum"].astype(np.float64)
        mean_frequency_volume_hz[stack_idx, :, :] = result["mean_frequency_image_hz"]
        bandwidth_volume_hz[stack_idx, :, :] = result["bandwidth_image_hz"]
        dynamic_std_volume[stack_idx, :, :] = result["dynamic_std_image"]

        hsv_summary_records.append(
            {
                "bline_index": extract_bline_index(stack_path),
                "stack_name": stack_path.name,
                "mean_frequency_mean_hz": float(np.mean(result["mean_frequency_image_hz"])),
                "mean_frequency_std_hz": float(np.std(result["mean_frequency_image_hz"], ddof=1))
                if result["mean_frequency_image_hz"].size > 1
                else 0.0,
                "bandwidth_mean_hz": float(np.mean(result["bandwidth_image_hz"])),
                "bandwidth_std_hz": float(np.std(result["bandwidth_image_hz"], ddof=1))
                if result["bandwidth_image_hz"].size > 1
                else 0.0,
                "dynamic_std_mean": float(np.mean(result["dynamic_std_image"])),
                "dynamic_std_std": float(np.std(result["dynamic_std_image"], ddof=1))
                if result["dynamic_std_image"].size > 1
                else 0.0,
            }
        )

        for band_idx, (low_hz, high_hz) in enumerate(bands_hz):
            image = result["dynamic_images"][band_idx]
            band_volumes[band_idx, stack_idx, :, :] = image

            normalized_image = safe_normalize(image, DEFAULT_OUTPUT_NORMALIZATION_CONSTANT)
            summary_records.append(
                {
                    "bline_index": extract_bline_index(stack_path),
                    "stack_name": stack_path.name,
                    "band_label": band_label(low_hz, high_hz),
                    "band_low_hz": float(low_hz),
                    "band_high_hz": float(high_hz),
                    "raw_volume_mean": float(np.mean(image)),
                    "raw_volume_std": float(np.std(image, ddof=1)) if image.size > 1 else 0.0,
                    "raw_volume_min": float(np.min(image)),
                    "raw_volume_max": float(np.max(image)),
                    "normalized_volume_mean": float(np.mean(normalized_image)),
                    "normalized_volume_std": float(np.std(normalized_image, ddof=1))
                    if normalized_image.size > 1
                    else 0.0,
                    "normalized_volume_min": float(np.min(normalized_image)),
                    "normalized_volume_max": float(np.max(normalized_image)),
                }
            )

        print(f"Processed {stack_idx + 1}/{bline_count}: {stack_path.name}")

    global_mean_spectrum = (global_mean_spectrum / float(bline_count)).astype(np.float32)

    save_summary_csv(summary_records, out_dir / "complex_frequency_dynamic_summary.csv")
    save_hsv_metric_summary_csv(hsv_summary_records, out_dir / "complex_hsv_metric_summary.csv")
    save_hsv_cache_metadata(out_dir, expected_cache_metadata)
    save_global_spectrum_figure(
        first_result["frequencies_hz"],
        global_mean_spectrum,
        out_dir / "global_mean_complex_power_spectrum.png",
        title="Global mean complex temporal power spectrum",
    )

    for band_idx, (low_hz, high_hz) in enumerate(bands_hz):
        label = band_label(low_hz, high_hz)
        raw_volume = band_volumes[band_idx]
        normalized_volume = safe_normalize(raw_volume, DEFAULT_OUTPUT_NORMALIZATION_CONSTANT)
        mean_bline = np.mean(normalized_volume, axis=0, dtype=np.float32)
        enface = np.mean(normalized_volume, axis=2, dtype=np.float32)

        save_volume_tiff(raw_volume, out_dir / f"complex_dynamic_raw_{label}.tif")
        save_volume_tiff(normalized_volume, out_dir / f"complex_dynamic_normalized_{label}.tif")
        save_quicklook_figure(
            mean_bline,
            enface,
            out_dir / f"complex_dynamic_quicklook_{label}.png",
            title=(
                f"Complex dynamic std, symmetric band +/-[{low_hz:g}, {high_hz:g}] Hz"
            ),
        )

    hsv_volume, rgb_volume, rgb_uint8_volume = hsv_volume_from_metric_volumes(
        mean_frequency_volume_hz,
        bandwidth_volume_hz,
        dynamic_std_volume,
    )
    save_volume_tiff(mean_frequency_volume_hz, out_dir / "hsv_mean_frequency_hz.tif")
    save_volume_tiff(bandwidth_volume_hz, out_dir / "hsv_bandwidth_hz.tif")
    save_volume_tiff(dynamic_std_volume, out_dir / "hsv_dynamic_std.tif")
    save_volume_tiff(hsv_volume, out_dir / "hsv_metric_volume_float.tif")
    save_rgb_volume_tiff(rgb_uint8_volume, out_dir / "hsv_rendered_rgb_volume.tif")

    rgb_mean_bline = np.mean(rgb_volume, axis=0, dtype=np.float32)
    rgb_enface = np.mean(rgb_volume, axis=2, dtype=np.float32)
    save_rgb_quicklook_figure(
        rgb_mean_bline,
        rgb_enface,
        out_dir / "hsv_rendered_rgb_quicklook.png",
        title="HSV-rendered complex dynamic volume",
    )

    print("Finished complex frequency dynamic volume analysis.")


if __name__ == "__main__":
    main()
