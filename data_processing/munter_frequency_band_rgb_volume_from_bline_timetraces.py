import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import tifffile as TIFF

try:
    from skimage.exposure import equalize_adapthist
except ImportError:
    equalize_adapthist = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_DIR = r"E:\IOCTData\BreastCancerMice\0630\Z21_33"
DEFAULT_FILENAME_GLOB = r"Cscan-4-Bline-*-Yrpt100-X1104-Z73.tif"
DEFAULT_FRAME_RATE_HZ = 50.0
DEFAULT_DURATION_SECONDS = 2.0
DEFAULT_EXPECTED_FRAME_COUNT = 100  # 2 s x 50 Hz

# Munter et al. band assignment:
#   Blue  = slow frequency band
#   Green = intermediate frequency band
#   Red   = fast frequency band
# Edit these if you want to test nearby variants.
DEFAULT_BLUE_BAND_HZ = (0.0, 0.5)
DEFAULT_GREEN_BAND_HZ = (0.5, 5.0)
DEFAULT_RED_BAND_HZ = (5.0, 25.0)

# Intermediate raw channel volumes are cached. These settings only affect the
# rerendered final RGB output.
DEFAULT_LOG_OFFSET = 1e-3
DEFAULT_APPLY_LOG_SCALE = True
DEFAULT_APPLY_CLAHE = True
DEFAULT_CLAHE_KERNEL_FRACTION = 0.125
DEFAULT_CLAHE_CLIP_LIMIT = 0.01
DEFAULT_APPLY_FOCUS_BRIGHTNESS_NORMALIZATION = True
DEFAULT_FOCUS_DEPTH_RANGE = None  # Example: (20, 120). None means use all depths.
DEFAULT_MEDIAN_FILTER_SIZE = 3
DEFAULT_RGB_SATURATION_SCALE = 0.6  # 1.0 keeps full color, 0.0 gives grayscale.

DEFAULT_OUTPUT_DIR = None  # None saves into input_dir / "munter_frequency_band_rgb_volume".
DEFAULT_SAVE_DPI = 300
DEFAULT_SHOW_FIGURES = False
DEFAULT_REUSE_SAVED_CHANNEL_CACHE = True

# Keep False for Spyder/IPython. Set True only when running from a terminal.
USE_COMMAND_LINE_ARGS = False

BLINE_INDEX_RE = re.compile(r"Bline-(?P<index>\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load matched AMP+PHASE B-line time-trace TIFF stacks, reconstruct the "
            "complex signal, compute Munter-style RGB band volumes from temporal "
            "frequency bands using symmetric complex spectral power, cache intermediate "
            "channel volumes, and save the final RGB volume."
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
        help="Optional output directory. Defaults to input_dir / munter_frequency_band_rgb_volume.",
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
        out_dir = Path(input_dir) / "munter_frequency_band_rgb_volume"
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


def validate_band(frame_rate_hz, band_hz, label):
    nyquist_hz = 0.5 * float(frame_rate_hz)
    low_hz = float(band_hz[0])
    high_hz = float(band_hz[1])
    if low_hz < 0 or high_hz <= low_hz or high_hz > nyquist_hz + 1e-6:
        raise ValueError(
            f"Invalid {label} band ({low_hz}, {high_hz}) Hz for frame rate "
            f"{frame_rate_hz:g} Hz (Nyquist {nyquist_hz:g} Hz)."
        )
    return low_hz, high_hz


def validate_rgb_bands(frame_rate_hz):
    return {
        "blue": validate_band(frame_rate_hz, DEFAULT_BLUE_BAND_HZ, "blue"),
        "green": validate_band(frame_rate_hz, DEFAULT_GREEN_BAND_HZ, "green"),
        "red": validate_band(frame_rate_hz, DEFAULT_RED_BAND_HZ, "red"),
    }


def band_mask(frequencies_hz, band_hz, is_last=False):
    low_hz, high_hz = band_hz
    abs_frequencies_hz = np.abs(frequencies_hz)
    if is_last:
        return (abs_frequencies_hz >= low_hz) & (abs_frequencies_hz <= high_hz)
    return (abs_frequencies_hz >= low_hz) & (abs_frequencies_hz < high_hz)


def compute_rgb_band_images_from_complex_stack(complex_stack, frame_rate_hz, rgb_bands_hz):
    centered_complex = complex_stack - np.mean(
        complex_stack,
        axis=0,
        keepdims=True,
        dtype=np.complex64,
    )
    spectrum = np.fft.fft(centered_complex, axis=0) / centered_complex.shape[0]
    complex_power_spectrum = (np.abs(spectrum) ** 2).astype(np.float32, copy=False)
    frequencies_hz = np.fft.fftfreq(
        centered_complex.shape[0],
        d=1.0 / float(frame_rate_hz),
    ).astype(np.float32)

    rgb_images = {}
    band_names = ["blue", "green", "red"]
    for band_idx, band_name in enumerate(band_names):
        mask = band_mask(
            frequencies_hz,
            rgb_bands_hz[band_name],
            is_last=(band_idx == len(band_names) - 1),
        )
        rgb_images[band_name] = np.sum(complex_power_spectrum[mask, :, :], axis=0, dtype=np.float32)
    return frequencies_hz, rgb_images


def save_volume_tiff(volume, output_path, dtype=np.float32, photometric=None):
    kwargs = {}
    if photometric is not None:
        kwargs["photometric"] = photometric
    TIFF.imwrite(output_path, np.asarray(volume, dtype=dtype), **kwargs)
    print(f"Saved TIFF volume: {output_path}")


def load_volume_tiff(path, dtype=np.float32):
    return np.asarray(TIFF.imread(path), dtype=dtype)


def cache_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "blue": output_dir / "band_blue_raw_volume.tif",
        "green": output_dir / "band_green_raw_volume.tif",
        "red": output_dir / "band_red_raw_volume.tif",
        "metadata_json": output_dir / "band_cache_metadata.json",
        "raw_summary_csv": output_dir / "band_raw_summary.csv",
    }


def build_cache_metadata(input_dir, filename_glob, frame_rate_hz, duration_seconds):
    return {
        "analysis_method": "complex_fft_symmetric_band_power",
        "input_dir": str(Path(input_dir)),
        "filename_glob": str(filename_glob),
        "frame_rate_hz": float(frame_rate_hz),
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "expected_frame_count": int(DEFAULT_EXPECTED_FRAME_COUNT),
        "blue_band_hz": list(DEFAULT_BLUE_BAND_HZ),
        "green_band_hz": list(DEFAULT_GREEN_BAND_HZ),
        "red_band_hz": list(DEFAULT_RED_BAND_HZ),
    }


def save_cache_metadata(output_dir, metadata):
    metadata_path = cache_paths(output_dir)["metadata_json"]
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved cache metadata: {metadata_path}")


def load_cache_metadata(output_dir):
    metadata_path = cache_paths(output_dir)["metadata_json"]
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def saved_channel_cache_exists(output_dir, expected_metadata=None):
    paths = cache_paths(output_dir)
    files_exist = paths["blue"].exists() and paths["green"].exists() and paths["red"].exists()
    if not files_exist:
        return False
    if expected_metadata is None:
        return True
    cached_metadata = load_cache_metadata(output_dir)
    if cached_metadata is None:
        print("Band cache metadata is missing; recomputing from TIFF stacks.")
        return False
    if cached_metadata != expected_metadata:
        print("Band cache metadata does not match current input settings; recomputing.")
        return False
    return True


def load_saved_channel_cache(output_dir):
    paths = cache_paths(output_dir)
    blue_volume = load_volume_tiff(paths["blue"], dtype=np.float32)
    green_volume = load_volume_tiff(paths["green"], dtype=np.float32)
    red_volume = load_volume_tiff(paths["red"], dtype=np.float32)
    if blue_volume.shape != green_volume.shape or blue_volume.shape != red_volume.shape:
        raise ValueError("Saved band cache volumes do not share the same shape.")
    print("Loaded saved RGB band volumes; skipping TIFF-stack reprocessing.")
    return blue_volume, green_volume, red_volume


def safe_log_transform(volume):
    return np.log10(np.maximum(np.asarray(volume, dtype=np.float32), 0.0) + float(DEFAULT_LOG_OFFSET))


def normalize_channel_to_unit(channel):
    channel = np.asarray(channel, dtype=np.float32)
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)
    low_value = float(np.percentile(finite, 1.0))
    high_value = float(np.percentile(finite, 99.5))
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        low_value = float(np.min(finite))
        high_value = float(np.max(finite))
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        return np.zeros_like(channel, dtype=np.float32)
    normalized = (channel - low_value) / (high_value - low_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def clahe_kernel_size_2d(shape_2d):
    x_pixels, z_pixels = shape_2d
    kernel_x = max(8, int(round(x_pixels * float(DEFAULT_CLAHE_KERNEL_FRACTION))))
    kernel_z = max(8, int(round(z_pixels * float(DEFAULT_CLAHE_KERNEL_FRACTION))))
    return (kernel_x, kernel_z)


def apply_clahe_volume(channel_volume):
    if equalize_adapthist is None or not DEFAULT_APPLY_CLAHE:
        return np.asarray(channel_volume, dtype=np.float32)

    processed = np.zeros_like(channel_volume, dtype=np.float32)
    kernel_size = clahe_kernel_size_2d(channel_volume.shape[1:])
    for bline_index in range(channel_volume.shape[0]):
        processed[bline_index] = equalize_adapthist(
            np.asarray(channel_volume[bline_index], dtype=np.float32),
            kernel_size=kernel_size,
            clip_limit=float(DEFAULT_CLAHE_CLIP_LIMIT),
        ).astype(np.float32, copy=False)
    return processed


def apply_focus_brightness_normalization(rgb_volume):
    if not DEFAULT_APPLY_FOCUS_BRIGHTNESS_NORMALIZATION:
        return np.asarray(rgb_volume, dtype=np.float32)

    rgb_volume = np.asarray(rgb_volume, dtype=np.float32)
    x_pixels = rgb_volume.shape[1]
    z_pixels = rgb_volume.shape[2]
    del x_pixels

    if DEFAULT_FOCUS_DEPTH_RANGE is None:
        focus_start = 0
        focus_stop = z_pixels
    else:
        focus_start = max(0, int(DEFAULT_FOCUS_DEPTH_RANGE[0]))
        focus_stop = min(z_pixels, int(DEFAULT_FOCUS_DEPTH_RANGE[1]))
        if focus_stop <= focus_start:
            focus_start = 0
            focus_stop = z_pixels

    brightness_profile = np.mean(rgb_volume, axis=(0, 1, 3), dtype=np.float32)
    focus_peak = float(np.max(brightness_profile[focus_start:focus_stop]))
    if not np.isfinite(focus_peak) or focus_peak <= 0:
        return rgb_volume

    scale_profile = np.ones(z_pixels, dtype=np.float32)
    valid = brightness_profile > 0
    scale_profile[valid] = np.minimum(focus_peak / brightness_profile[valid], 1.0)
    scale_profile = scale_profile[np.newaxis, np.newaxis, :, np.newaxis]
    normalized = rgb_volume * scale_profile
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def apply_median_filter_rgb_volume(rgb_volume):
    filter_size = int(DEFAULT_MEDIAN_FILTER_SIZE)
    if filter_size <= 1:
        return np.asarray(rgb_volume, dtype=np.float32)
    filtered = np.zeros_like(rgb_volume, dtype=np.float32)
    for channel_index in range(3):
        filtered[..., channel_index] = median_filter(
            rgb_volume[..., channel_index],
            size=(1, filter_size, filter_size),
            mode="nearest",
        ).astype(np.float32, copy=False)
    return filtered


def adjust_rgb_saturation(rgb_volume, saturation_scale):
    saturation_scale = float(saturation_scale)
    rgb_volume = np.asarray(rgb_volume, dtype=np.float32)
    if not np.isfinite(saturation_scale):
        saturation_scale = 1.0
    saturation_scale = float(np.clip(saturation_scale, 0.0, 1.0))
    if abs(saturation_scale - 1.0) < 1e-6:
        return rgb_volume

    grayscale = np.mean(rgb_volume, axis=-1, keepdims=True, dtype=np.float32)
    adjusted = grayscale + saturation_scale * (rgb_volume - grayscale)
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32, copy=False)


def render_rgb_volume_from_band_volumes(blue_volume, green_volume, red_volume):
    blue = np.asarray(blue_volume, dtype=np.float32)
    green = np.asarray(green_volume, dtype=np.float32)
    red = np.asarray(red_volume, dtype=np.float32)

    if DEFAULT_APPLY_LOG_SCALE:
        blue = safe_log_transform(blue)
        green = safe_log_transform(green)
        red = safe_log_transform(red)

    blue = normalize_channel_to_unit(blue)
    green = normalize_channel_to_unit(green)
    red = normalize_channel_to_unit(red)

    if DEFAULT_APPLY_CLAHE:
        blue = apply_clahe_volume(blue)
        green = apply_clahe_volume(green)
        red = apply_clahe_volume(red)

    rgb_volume = np.stack([red, green, blue], axis=-1).astype(np.float32, copy=False)
    rgb_volume = apply_focus_brightness_normalization(rgb_volume)
    rgb_volume = apply_median_filter_rgb_volume(rgb_volume)
    rgb_volume = adjust_rgb_saturation(rgb_volume, DEFAULT_RGB_SATURATION_SCALE)
    rgb_uint8_volume = np.clip(np.round(rgb_volume * 255.0), 0, 255).astype(np.uint8)
    return rgb_volume, rgb_uint8_volume


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


def save_raw_summary_csv(records, output_path):
    fieldnames = [
        "bline_index",
        "stack_name",
        "blue_mean",
        "green_mean",
        "red_mean",
        "blue_std",
        "green_std",
        "red_std",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV summary: {output_path}")


def extract_bline_index(path):
    match = BLINE_INDEX_RE.search(path.name)
    if match is None:
        return np.nan
    return int(match.group("index"))


def process_single_stack(path, frame_rate_hz, duration_seconds, rgb_bands_hz):
    amplitude_stack, phase_stack = load_amp_phase_stack(path)
    complex_stack = reconstruct_complex_stack(amplitude_stack, phase_stack)
    complex_stack = limit_stack_duration(complex_stack, frame_rate_hz, duration_seconds)

    if complex_stack.shape[0] != DEFAULT_EXPECTED_FRAME_COUNT:
        print(
            f"Warning: {path.name} uses {complex_stack.shape[0]} frames after duration limit; "
            f"expected {DEFAULT_EXPECTED_FRAME_COUNT}."
        )

    _, rgb_images = compute_rgb_band_images_from_complex_stack(
        complex_stack,
        frame_rate_hz=frame_rate_hz,
        rgb_bands_hz=rgb_bands_hz,
    )
    return rgb_images


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
    expected_cache_metadata = build_cache_metadata(
        input_dir,
        filename_glob,
        frame_rate_hz,
        duration_seconds,
    )

    if DEFAULT_REUSE_SAVED_CHANNEL_CACHE and saved_channel_cache_exists(
        out_dir,
        expected_metadata=expected_cache_metadata,
    ):
        blue_volume, green_volume, red_volume = load_saved_channel_cache(out_dir)
        rgb_volume, rgb_uint8_volume = render_rgb_volume_from_band_volumes(
            blue_volume,
            green_volume,
            red_volume,
        )
        save_volume_tiff(rgb_volume, out_dir / "munter_rgb_volume_float.tif", dtype=np.float32)
        save_volume_tiff(
            rgb_uint8_volume,
            out_dir / "munter_rgb_volume_uint8.tif",
            dtype=np.uint8,
            photometric="rgb",
        )
        rgb_mean_bline = np.mean(rgb_volume, axis=0, dtype=np.float32)
        rgb_enface = np.mean(rgb_volume, axis=2, dtype=np.float32)
        save_rgb_quicklook_figure(
            rgb_mean_bline,
            rgb_enface,
            out_dir / "munter_rgb_quicklook.png",
            title="Munter-style frequency-band RGB volume",
        )
        print("Finished RGB rerender from saved band cache.")
        return

    rgb_bands_hz = validate_rgb_bands(frame_rate_hz)
    stack_paths = iter_matching_stacks(input_dir, filename_glob)
    print(f"Matched {len(stack_paths)} stacks with glob: {filename_glob}")

    first_images = process_single_stack(
        stack_paths[0],
        frame_rate_hz=frame_rate_hz,
        duration_seconds=duration_seconds,
        rgb_bands_hz=rgb_bands_hz,
    )

    bline_count = len(stack_paths)
    x_pixels, z_pixels = first_images["blue"].shape
    blue_volume = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)
    green_volume = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)
    red_volume = np.zeros((bline_count, x_pixels, z_pixels), dtype=np.float32)
    summary_records = []

    for stack_idx, stack_path in enumerate(stack_paths):
        if stack_idx == 0:
            rgb_images = first_images
        else:
            rgb_images = process_single_stack(
                stack_path,
                frame_rate_hz=frame_rate_hz,
                duration_seconds=duration_seconds,
                rgb_bands_hz=rgb_bands_hz,
            )

        blue_volume[stack_idx, :, :] = rgb_images["blue"]
        green_volume[stack_idx, :, :] = rgb_images["green"]
        red_volume[stack_idx, :, :] = rgb_images["red"]
        summary_records.append(
            {
                "bline_index": extract_bline_index(stack_path),
                "stack_name": stack_path.name,
                "blue_mean": float(np.mean(rgb_images["blue"])),
                "green_mean": float(np.mean(rgb_images["green"])),
                "red_mean": float(np.mean(rgb_images["red"])),
                "blue_std": float(np.std(rgb_images["blue"], ddof=1)) if rgb_images["blue"].size > 1 else 0.0,
                "green_std": float(np.std(rgb_images["green"], ddof=1)) if rgb_images["green"].size > 1 else 0.0,
                "red_std": float(np.std(rgb_images["red"], ddof=1)) if rgb_images["red"].size > 1 else 0.0,
            }
        )
        print(f"Processed {stack_idx + 1}/{bline_count}: {stack_path.name}")

    save_volume_tiff(blue_volume, cache_paths(out_dir)["blue"], dtype=np.float32)
    save_volume_tiff(green_volume, cache_paths(out_dir)["green"], dtype=np.float32)
    save_volume_tiff(red_volume, cache_paths(out_dir)["red"], dtype=np.float32)
    save_raw_summary_csv(summary_records, cache_paths(out_dir)["raw_summary_csv"])
    save_cache_metadata(out_dir, expected_cache_metadata)

    rgb_volume, rgb_uint8_volume = render_rgb_volume_from_band_volumes(
        blue_volume,
        green_volume,
        red_volume,
    )
    save_volume_tiff(rgb_volume, out_dir / "munter_rgb_volume_float.tif", dtype=np.float32)
    save_volume_tiff(
        rgb_uint8_volume,
        out_dir / "munter_rgb_volume_uint8.tif",
        dtype=np.uint8,
        photometric="rgb",
    )

    rgb_mean_bline = np.mean(rgb_volume, axis=0, dtype=np.float32)
    rgb_enface = np.mean(rgb_volume, axis=2, dtype=np.float32)
    save_rgb_quicklook_figure(
        rgb_mean_bline,
        rgb_enface,
        out_dir / "munter_rgb_quicklook.png",
        title="Munter-style frequency-band RGB volume",
    )

    print("Finished Munter-style frequency-band RGB analysis.")


if __name__ == "__main__":
    main()
