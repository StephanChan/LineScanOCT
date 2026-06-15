import argparse
import csv
import re
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, Slider
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_DIR = r"E:\IOCTData\Lung Cancer mice 260601\260608\200Hz 2seconds Blines"
DEFAULT_BACKGROUND_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"  # Optional separate noise/background AMP+PHASE TIFF stack or directory.
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_TISSUE_DURATION_SECONDS = 2.0
DEFAULT_BACKGROUND_DURATION_SECONDS = 2.0
DEFAULT_NOTCH_BAND_HZ = None
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 1
DEFAULT_DYNAMIC_CHUNK_X = 200
DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT = 0.6284  # Manual upper-bound C for sigma_phase^2 = C/SNR + floor.
DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2 = 0.0  # Manual residual phase-variance floor in rad^2.
DEFAULT_INITIAL_SNR_THRESHOLD_DB = 15.0
DEFAULT_BIN_COUNT = 20
DEFAULT_MIN_POINTS_PER_BIN = 25
DEFAULT_SCATTER_ALPHA = 0.18
DEFAULT_SCATTER_SIZE = 5.0
DEFAULT_SAVE_ROI = True

# Keep False for Spyder/IPython. Set True only when running from a terminal and
# passing command-line arguments.
USE_COMMAND_LINE_ARGS = False

BLINE_NAME_RE = re.compile(
    r"^Bline-(?P<index>\d+)-Yrpt(?P<yrpt>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)


def iter_tiff_stacks(input_dir):
    input_path = Path(input_dir)
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_dir}")

    paths = [
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    paths.sort(key=natural_bline_sort_key)
    if not paths:
        raise ValueError(f"No TIFF stacks found in: {input_dir}")
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
    return reconstruct_complex_data(amplitude, phase)


def reconstruct_complex_data(amplitude, phase):
    amplitude = np.asarray(amplitude, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)


def limit_stack_duration(complex_stack, frame_rate_hz, duration_seconds):
    complex_stack = np.asarray(complex_stack, dtype=np.complex64)
    if duration_seconds is None:
        return complex_stack

    duration_seconds = float(duration_seconds)
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return complex_stack

    max_frames = int(np.floor(duration_seconds * float(frame_rate_hz)))
    if max_frames < 1:
        return np.ascontiguousarray(complex_stack[:1], dtype=np.complex64)
    if complex_stack.shape[0] <= max_frames:
        return complex_stack
    return np.ascontiguousarray(complex_stack[:max_frames], dtype=np.complex64)


def fft_bandstop_stack_axis0(stack, frame_rate_hz, stop_band_hz=None):
    stack = np.asarray(stack)
    if stack.shape[0] < 2 or stop_band_hz is None:
        return stack.copy()

    low_hz, high_hz = sorted([float(stop_band_hz[0]), float(stop_band_hz[1])])
    if high_hz <= 0 or high_hz <= low_hz:
        return stack.copy()

    frequencies = np.fft.fftfreq(stack.shape[0], d=1.0 / float(frame_rate_hz))
    spectrum = np.fft.fft(stack, axis=0)
    stop_mask = (np.abs(frequencies) >= low_hz) & (np.abs(frequencies) <= high_hz)
    spectrum[stop_mask, ...] = 0
    filtered = np.fft.ifft(spectrum, axis=0)
    if np.iscomplexobj(stack):
        return filtered.astype(np.complex64, copy=False)
    return filtered.real.astype(np.float32, copy=False)


def notch_filter_complex_stack_amp_phase(complex_stack, frame_rate_hz, notch_band_hz, chunk_x):
    frames, x_pixels, z_pixels = complex_stack.shape
    filtered_stack = np.empty((frames, x_pixels, z_pixels), dtype=np.complex64)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        complex_chunk = complex_stack[:, x0:x1, :]
        amplitude = np.abs(complex_chunk).astype(np.float32, copy=False)
        phase = np.unwrap(np.angle(complex_chunk).astype(np.float32, copy=False), axis=0)
        amplitude = fft_bandstop_stack_axis0(amplitude, frame_rate_hz, notch_band_hz)
        phase = fft_bandstop_stack_axis0(phase, frame_rate_hz, notch_band_hz)
        filtered_stack[:, x0:x1, :] = (
            amplitude * np.exp(1j * phase)
        ).astype(np.complex64, copy=False)
        print(f"Notch filter: processed X {x0}-{x1} / {x_pixels}")

    return filtered_stack


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


def compute_dynamic_images_from_complex_stack(complex_stack, uniform_filter_size, chunk_x):
    frames, x_pixels, z_pixels = complex_stack.shape
    amplitude_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    complex_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        complex_chunk = complex_stack[:, x0:x1, :]
        amplitude = np.abs(complex_chunk).astype(np.float32, copy=False)
        phase = np.unwrap(np.angle(complex_chunk).astype(np.float32, copy=False), axis=0)
        amplitude_dynamic[x0:x1, :] = gpu_style_dynamic_from_real_stack(
            amplitude,
            uniform_filter_size=uniform_filter_size,
        )
        complex_dynamic[x0:x1, :] = complex_dynamic_from_filtered_amp_phase(
            amplitude,
            phase,
            uniform_filter_size=uniform_filter_size,
        )
        print(f"Dynamic maps: processed X {x0}-{x1} / {x_pixels}")

    return amplitude_dynamic, complex_dynamic


def estimate_sigma_q_from_complex_samples(complex_samples):
    complex_samples = np.asarray(complex_samples, dtype=np.complex64)
    if complex_samples.size == 0:
        return np.nan

    real_part = np.real(complex_samples).astype(np.float32, copy=False)
    imag_part = np.imag(complex_samples).astype(np.float32, copy=False)
    real_centered = real_part - np.mean(real_part, axis=0, keepdims=True, dtype=np.float32)
    imag_centered = imag_part - np.mean(imag_part, axis=0, keepdims=True, dtype=np.float32)
    sigma_q2 = 0.5 * (
        np.var(real_centered, axis=0, dtype=np.float32)
        + np.var(imag_centered, axis=0, dtype=np.float32)
    )
    sigma_q = float(np.sqrt(np.mean(sigma_q2, dtype=np.float32)))
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        return np.nan
    return sigma_q


def expected_phase_variance_from_snr_db(
    snr_db,
    coefficient=DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT,
    variance_floor_rad2=DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2,
):
    snr_db = np.asarray(snr_db, dtype=np.float32)
    return (
        float(coefficient) * np.power(10.0, -snr_db / 10.0)
        + float(variance_floor_rad2)
    ).astype(np.float32, copy=False)


def snr_db_map_from_amplitude_mean(amplitude_mean, sigma_q):
    amplitude_mean = np.asarray(amplitude_mean, dtype=np.float32)
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        return np.full(amplitude_mean.shape, np.nan, dtype=np.float32)
    snr_linear = (amplitude_mean * amplitude_mean) / np.float32(2.0 * sigma_q * sigma_q)
    snr_linear = np.maximum(snr_linear, np.float32(1e-12))
    return (10.0 * np.log10(snr_linear)).astype(np.float32, copy=False)


def polygon_mask_for_image_shape(vertices, image_shape):
    if vertices is None or len(vertices) < 3:
        raise ValueError("ROI needs at least three points.")

    x_pixels, z_pixels = image_shape
    x_grid, z_grid = np.meshgrid(np.arange(x_pixels), np.arange(z_pixels), indexing="ij")
    points = np.column_stack([x_grid.reshape(-1), z_grid.reshape(-1)])
    mask = MplPath(vertices).contains_points(points).reshape(image_shape)
    if not np.any(mask):
        raise ValueError("ROI mask is empty. Please draw a larger region.")
    return mask


def select_polygon_roi(image, title):
    vertices = []
    accepted = {"done": False}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image.T, aspect="auto", origin="upper", cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth index")
    instruction = ax.text(
        0.01,
        0.99,
        "Click ROI vertices. Press Enter to accept, Esc to reset.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )

    def on_select(selected_vertices):
        vertices[:] = selected_vertices

    try:
        selector = PolygonSelector(
            ax,
            on_select,
            useblit=True,
            props={"color": "yellow", "linewidth": 1.5, "alpha": 0.9},
            handle_props={"markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 5},
        )
    except TypeError:
        selector = PolygonSelector(
            ax,
            on_select,
            useblit=True,
            lineprops={"color": "yellow", "linewidth": 1.5, "alpha": 0.9},
            markerprops={"markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 5},
        )

    def on_key(event):
        if event.key == "enter":
            accepted["done"] = True
            plt.close(fig)
        elif event.key == "escape":
            vertices.clear()
            try:
                selector.verts = []
            except Exception:
                pass
            instruction.set_text("ROI reset. Click ROI vertices. Press Enter to accept.")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    selector.disconnect_events()

    if not accepted["done"]:
        raise RuntimeError("ROI selection window was closed before pressing Enter.")
    return np.asarray(vertices, dtype=np.float32)


def normalize_roi_key(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def load_saved_roi_vertices(stack_path, output_dir):
    stack_path = Path(stack_path)
    stack_stem = stack_path.stem
    workbook_candidates = [
        Path(output_dir) / f"{stack_stem}_dynamic_correlation_roi.xlsx",
        stack_path.parent / "dynamic_timepoint_sufficiency" / f"{stack_stem}_dynamic_timepoint_metrics.xlsx",
        stack_path.parent / "frequency_band_power_analysis" / f"{stack_stem}_frequency_domain_metrics.xlsx",
    ]
    csv_candidates = [
        Path(output_dir) / f"{stack_stem}_dynamic_correlation_roi.csv",
    ]

    if pd is not None:
        for workbook_path in workbook_candidates:
            if not workbook_path.exists():
                continue
            try:
                dataframe = pd.read_excel(workbook_path, sheet_name="roi_vertices")
                records = dataframe.to_dict("records")
            except Exception as error:
                print(f"Could not read ROI workbook {workbook_path} ({error}); trying next fallback.")
                continue

            vertices = vertices_map_from_records_for_stack(records, stack_path)
            if vertices:
                print(f"Loaded tissue ROI from {workbook_path}")
                return vertices

    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", newline="") as file:
                records = list(csv.DictReader(file))
            vertices = vertices_map_from_records_for_stack(records, stack_path)
            if vertices:
                print(f"Loaded tissue ROI from {csv_path}")
                return vertices
        except Exception as error:
            print(f"Could not read ROI CSV {csv_path} ({error}); trying next fallback.")

    return {}


def vertices_map_from_records_for_stack(records, stack_path):
    if not records:
        return {}

    if "source_stack" in records[0]:
        source_keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_path.stem)}
        matching_records = [
            record
            for record in records
            if normalize_roi_key(record.get("source_stack", "")) in source_keys
        ]
    else:
        label_keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_path.stem)}
        matching_records = [
            record
            for record in records
            if normalize_roi_key(record.get("label", "")) in label_keys
        ]

    vertices_map = {}
    subset = [
        record
        for record in matching_records
        if str(record.get("roi", "")).strip().lower() == "tissue"
    ]
    if subset:
        subset.sort(key=lambda record: int(record["vertex_index"]))
        vertices = np.asarray(
            [[float(record["x_pixel"]), float(record["depth_pixel"])] for record in subset],
            dtype=np.float32,
        )
        if vertices.shape[0] >= 3:
            vertices_map["tissue"] = vertices
    return vertices_map


def save_roi_set(output_dir, stack_path, roi_vertices_map):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for roi_name, vertices in roi_vertices_map.items():
        vertices = np.asarray(vertices, dtype=np.float32)
        for index, (x_value, z_value) in enumerate(vertices):
            records.append(
                {
                    "roi": roi_name,
                    "source_stack": stack_path.name,
                    "vertex_index": int(index),
                    "x_pixel": float(x_value),
                    "depth_pixel": float(z_value),
                }
            )
    if not records:
        return

    workbook_path = output_dir / f"{stack_path.stem}_dynamic_correlation_roi.xlsx"
    if pd is not None:
        try:
            with pd.ExcelWriter(workbook_path) as writer:
                pd.DataFrame.from_records(records).to_excel(
                    writer,
                    sheet_name="roi_vertices",
                    index=False,
                )
            print(f"Saved ROI workbook: {workbook_path}")
            return
        except Exception as error:
            print(f"Could not save ROI workbook ({error}); saving CSV fallback instead.")

    csv_path = output_dir / f"{stack_path.stem}_dynamic_correlation_roi.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved ROI CSV: {csv_path}")


def output_directory_for_stack(stack_path):
    output_dir = Path(stack_path).parent / "dynamic_correlation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_missing_rois(mean_abs_bline, stack_path, existing_vertices_map=None):
    roi_vertices_map = {}
    if existing_vertices_map is not None:
        roi_vertices_map.update(existing_vertices_map)

    if "tissue" not in roi_vertices_map:
        vertices = select_polygon_roi(
            mean_abs_bline,
            f"{stack_path.name}: draw TISSUE ROI, then press Enter",
        )
        roi_vertices_map["tissue"] = vertices
    return roi_vertices_map


def load_or_select_roi_set(mean_abs_bline, stack_path, output_dir):
    saved_vertices_map = load_saved_roi_vertices(stack_path, output_dir)
    if "tissue" in saved_vertices_map:
        return saved_vertices_map

    roi_vertices_map = select_missing_rois(
        mean_abs_bline,
        Path(stack_path),
        existing_vertices_map=saved_vertices_map,
    )
    if DEFAULT_SAVE_ROI:
        save_roi_set(output_dir, Path(stack_path), roi_vertices_map)
    return roi_vertices_map


def load_external_noise_sigma_q(
    background_input_path,
    frame_rate_hz,
    background_duration_seconds,
    notch_band_hz,
    chunk_x,
):
    if background_input_path is None or str(background_input_path).strip() == "":
        return np.nan, None

    try:
        background_paths = iter_tiff_stacks(background_input_path)
        background_path = background_paths[0]
        print(f"Loading background AMP+PHASE stack for sigma_q: {background_path}")
        background_complex_stack = read_amp_phase_tiff_stack(background_path)
        background_complex_stack = limit_stack_duration(
            background_complex_stack,
            frame_rate_hz=frame_rate_hz,
            duration_seconds=background_duration_seconds,
        )
        if notch_band_hz is not None:
            background_complex_stack = notch_filter_complex_stack_amp_phase(
                background_complex_stack,
                frame_rate_hz=frame_rate_hz,
                notch_band_hz=notch_band_hz,
                chunk_x=chunk_x,
            )
        top_half_depth = max(1, background_complex_stack.shape[2] // 2)
        sigma_q = estimate_sigma_q_from_complex_samples(
            np.asarray(background_complex_stack[:, :, :top_half_depth], dtype=np.complex64).reshape(
                background_complex_stack.shape[0],
                -1,
            )
        )
        print(
            "Measured external sigma_q from top-half background stack XZ range "
            f"(depth < {top_half_depth}): {sigma_q:.6g}"
        )
        return sigma_q, str(background_path)
    except Exception as error:
        print(
            f"Could not load separate background stack from {background_input_path} "
            f"({error}). Falling back to top-half tissue region."
        )
        return np.nan, None


def estimate_tissue_fallback_sigma_q(filtered_complex_stack):
    top_half_depth = max(1, filtered_complex_stack.shape[2] // 2)
    sigma_q = estimate_sigma_q_from_complex_samples(
        np.asarray(filtered_complex_stack[:, :, :top_half_depth], dtype=np.complex64).reshape(
            filtered_complex_stack.shape[0],
            -1,
        )
    )
    print(
        "Measured fallback sigma_q from top-half tissue stack XZ range "
        f"(depth < {top_half_depth}): {sigma_q:.6g}"
    )
    return sigma_q, "Top-half tissue region"


def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def finite_axis_limits(values, percentile_low=1.0, percentile_high=99.0, pad_fraction=0.06):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(values, [percentile_low, percentile_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.mean(values)) if values.size else 0.0
        return center - 1.0, center + 1.0
    pad = max((high - low) * float(pad_fraction), 1e-6)
    return float(low - pad), float(high + pad)


def compute_binned_curve(x_values, y_values, bin_count, min_points_per_bin):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < max(3, int(min_points_per_bin)):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    x_values = x_values[valid]
    y_values = y_values[valid]
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    bin_edges = np.linspace(x_min, x_max, int(bin_count) + 1, dtype=np.float32)
    x_centers = []
    y_medians = []
    for start, stop in zip(bin_edges[:-1], bin_edges[1:]):
        if stop == bin_edges[-1]:
            mask = (x_values >= start) & (x_values <= stop)
        else:
            mask = (x_values >= start) & (x_values < stop)
        if np.count_nonzero(mask) < int(min_points_per_bin):
            continue
        x_centers.append(0.5 * (start + stop))
        y_medians.append(np.median(y_values[mask]))
    return np.asarray(x_centers, dtype=np.float32), np.asarray(y_medians, dtype=np.float32)


def compute_correlation_text(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return "Too few points after threshold"

    x_values = x_values[valid]
    y_values = y_values[valid]
    pearson = np.nan
    if np.std(x_values) > 0 and np.std(y_values) > 0:
        pearson = float(np.corrcoef(x_values, y_values)[0, 1])

    if spearmanr is not None:
        spearman_result = spearmanr(x_values, y_values, nan_policy="omit")
        if hasattr(spearman_result, "statistic"):
            spearman = float(spearman_result.statistic)
        elif hasattr(spearman_result, "correlation"):
            spearman = float(spearman_result.correlation)
        else:
            spearman = float(spearman_result[0])
        return (
            f"N = {count}\n"
            f"Pearson r = {pearson:.3f}\n"
            f"Spearman rho = {spearman:.3f}"
        )

    return (
        f"N = {count}\n"
        f"Pearson r = {pearson:.3f}"
    )


def compute_linear_fit(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), None

    x_values = x_values[valid]
    y_values = y_values[valid]
    if np.std(x_values) <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), None

    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(np.min(x_values), np.max(x_values), 200, dtype=np.float32)
    y_line = slope * x_line + intercept
    return x_line, y_line.astype(np.float32, copy=False), (float(slope), float(intercept))


class DynamicOCTCorrelationExplorer:
    def __init__(
        self,
        complex_stack,
        stack_path,
        frame_rate_hz,
        sigma_q,
        sigma_q_source,
        tissue_mask,
        roi_vertices_map,
        phase_noise_variance_coefficient,
        phase_noise_variance_floor_rad2,
        notch_band_hz,
        dynamic_uniform_filter_size,
        dynamic_chunk_x,
    ):
        self.stack_path = Path(stack_path)
        self.frame_rate_hz = float(frame_rate_hz)
        self.sigma_q = float(sigma_q)
        self.sigma_q_source = sigma_q_source
        self.tissue_mask = np.asarray(tissue_mask, dtype=bool)
        self.roi_vertices_map = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in roi_vertices_map.items()
        }
        self.phase_noise_variance_coefficient = float(phase_noise_variance_coefficient)
        self.phase_noise_variance_floor_rad2 = float(phase_noise_variance_floor_rad2)
        self.notch_band_hz = notch_band_hz
        self.dynamic_uniform_filter_size = int(dynamic_uniform_filter_size)
        self.dynamic_chunk_x = int(dynamic_chunk_x)

        self.raw_complex_stack = np.asarray(complex_stack, dtype=np.complex64)
        if self.notch_band_hz is None:
            self.filtered_complex_stack = self.raw_complex_stack
        else:
            self.filtered_complex_stack = notch_filter_complex_stack_amp_phase(
                self.raw_complex_stack,
                frame_rate_hz=self.frame_rate_hz,
                notch_band_hz=self.notch_band_hz,
                chunk_x=self.dynamic_chunk_x,
            )

        self.mean_abs_bline = np.mean(np.abs(self.raw_complex_stack), axis=0, dtype=np.float32)
        self.amplitude_dynamic, self.complex_dynamic = compute_dynamic_images_from_complex_stack(
            self.filtered_complex_stack,
            uniform_filter_size=self.dynamic_uniform_filter_size,
            chunk_x=self.dynamic_chunk_x,
        )
        self.phase_std_map = np.std(
            np.angle(self.filtered_complex_stack).astype(np.float32, copy=False),
            axis=0,
            ddof=1,
        ).astype(np.float32, copy=False)
        self.snr_db_map = snr_db_map_from_amplitude_mean(self.mean_abs_bline, self.sigma_q)
        self.snr_limited_phase_std_map = np.sqrt(
            np.maximum(
                expected_phase_variance_from_snr_db(
                    self.snr_db_map,
                    coefficient=self.phase_noise_variance_coefficient,
                    variance_floor_rad2=self.phase_noise_variance_floor_rad2,
                ),
                0.0,
            )
        ).astype(np.float32, copy=False)

        tissue = self.tissue_mask
        self.intensity_values = self.mean_abs_bline[tissue].astype(np.float32, copy=False)
        self.snr_values_db = self.snr_db_map[tissue].astype(np.float32, copy=False)
        self.amp_dynamic_values = self.amplitude_dynamic[tissue].astype(np.float32, copy=False)
        self.complex_dynamic_values = self.complex_dynamic[tissue].astype(np.float32, copy=False)
        self.phase_std_values = self.phase_std_map[tissue].astype(np.float32, copy=False)
        self.snr_limited_phase_std_values = self.snr_limited_phase_std_map[tissue].astype(np.float32, copy=False)
        self.amp_over_complex = safe_ratio(self.amp_dynamic_values, self.complex_dynamic_values)
        self.complex_over_amp = safe_ratio(self.complex_dynamic_values, self.amp_dynamic_values)
        self.phase_std_over_limit = safe_ratio(
            self.phase_std_values,
            self.snr_limited_phase_std_values,
        )

    def show(self):
        plt.rcParams.update({"font.size": 11})
        fig = plt.figure(figsize=(18.0, 10.0))
        grid = GridSpec(
            3,
            3,
            figure=fig,
            height_ratios=[16.0, 11.0, 1.3],
            hspace=0.30,
            wspace=0.24,
        )

        ax_ratio_intensity = fig.add_subplot(grid[0, 0])
        ax_phase_ratio = fig.add_subplot(grid[0, 1])
        ax_intensity_phase = fig.add_subplot(grid[0, 2])
        ax_map_left = fig.add_subplot(grid[1, 0])
        ax_map_middle = fig.add_subplot(grid[1, 1])
        ax_map_right = fig.add_subplot(grid[1, 2])
        ax_slider_left = fig.add_subplot(grid[2, 0])
        ax_slider_middle = fig.add_subplot(grid[2, 1])
        ax_slider_right = fig.add_subplot(grid[2, 2])

        slider_min, slider_max = self._snr_slider_limits()
        left_slider = Slider(
            ax=ax_slider_left,
            label="Max SNR (dB)",
            valmin=slider_min,
            valmax=slider_max,
            valinit=np.clip(DEFAULT_INITIAL_SNR_THRESHOLD_DB, slider_min, slider_max),
            valstep=0.5,
        )
        middle_slider = Slider(
            ax=ax_slider_middle,
            label="Min SNR (dB)",
            valmin=slider_min,
            valmax=slider_max,
            valinit=np.clip(DEFAULT_INITIAL_SNR_THRESHOLD_DB, slider_min, slider_max),
            valstep=0.5,
        )
        right_slider = Slider(
            ax=ax_slider_right,
            label="Max SNR (dB)",
            valmin=slider_min,
            valmax=slider_max,
            valinit=np.clip(DEFAULT_INITIAL_SNR_THRESHOLD_DB, slider_min, slider_max),
            valstep=0.5,
        )

        left_scatters = self._make_region_scatters(ax_ratio_intensity)
        right_scatters = self._make_region_scatters(ax_phase_ratio)
        bottom_scatters = self._make_region_scatters(ax_intensity_phase)
        left_fit, = ax_ratio_intensity.plot([], [], color="tab:blue", linewidth=1.8, linestyle="--", label="Linear fit")
        right_fit, = ax_phase_ratio.plot([], [], color="tab:blue", linewidth=1.8, linestyle="--", label="Linear fit")
        bottom_fit, = ax_intensity_phase.plot([], [], color="tab:blue", linewidth=1.8, linestyle="--", label="Linear fit")
        right_reference, = ax_phase_ratio.plot([], [], color="0.45", linewidth=1.4, linestyle=":", label="Slope 1")

        left_text = ax_ratio_intensity.text(
            0.02,
            0.98,
            "",
            transform=ax_ratio_intensity.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        right_text = ax_phase_ratio.text(
            0.02,
            0.98,
            "",
            transform=ax_phase_ratio.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        bottom_text = ax_intensity_phase.text(
            0.02,
            0.98,
            "",
            transform=ax_intensity_phase.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

        ax_ratio_intensity.set_xlabel("SNR (dB)")
        ax_ratio_intensity.set_ylabel("Amp dynamic / Complex dynamic")
        ax_ratio_intensity.set_title("Low-SNR behavior of dynamic-signal ratio")
        ax_ratio_intensity.grid(True, alpha=0.25)

        ax_phase_ratio.set_xlabel("Phase std / SNR-limited phase std")
        ax_phase_ratio.set_ylabel("Complex dynamic / Amp dynamic")
        ax_phase_ratio.set_title("Excess phase fluctuation vs complex gain")
        ax_phase_ratio.grid(True, alpha=0.25)

        ax_intensity_phase.set_xlabel("OCT intensity (mean amplitude)")
        ax_intensity_phase.set_ylabel("Phase std / SNR-limited phase std")
        ax_intensity_phase.set_title("Intensity vs excess phase fluctuation")
        ax_intensity_phase.grid(True, alpha=0.25)

        left_map_image = self._initialize_selected_pixel_map(ax_map_left, "Selected pixels: panel 1")
        middle_map_image = self._initialize_selected_pixel_map(ax_map_middle, "Selected pixels: panel 2")
        right_map_image = self._initialize_selected_pixel_map(ax_map_right, "Selected pixels: panel 3")

        snr_xlim = finite_axis_limits(self.snr_values_db, percentile_low=1.0, percentile_high=99.5)
        intensity_xlim = finite_axis_limits(self.intensity_values, percentile_low=1.0, percentile_high=99.5)
        left_ylim = finite_axis_limits(self.amp_over_complex, percentile_low=1.0, percentile_high=99.0)
        bottom_ylim = finite_axis_limits(self.phase_std_over_limit, percentile_low=1.0, percentile_high=99.0)
        right_ylim = finite_axis_limits(self.complex_over_amp, percentile_low=1.0, percentile_high=99.0)
        ax_ratio_intensity.set_xlim(*snr_xlim)
        ax_ratio_intensity.set_ylim(*left_ylim)
        ax_phase_ratio.set_xlim(1.0, 3.0)
        ax_phase_ratio.set_ylim(*right_ylim)
        ax_intensity_phase.set_xlim(*intensity_xlim)
        ax_intensity_phase.set_ylim(*bottom_ylim)
        reference_x = np.linspace(1.0, 3.0, 200, dtype=np.float32)
        right_reference.set_data(reference_x, reference_x)

        summary_title = (
            f"{self.stack_path.name}\n"
            f"sigma_q = {self.sigma_q:.4g}, C = {self.phase_noise_variance_coefficient:.4g}, "
            f"floor = {self.phase_noise_variance_floor_rad2:.4g} rad^2"
        )
        fig.suptitle(summary_title, fontsize=13)

        def update_left(_value=None):
            threshold = float(left_slider.val)
            mask = (
                np.isfinite(self.snr_values_db)
                & np.isfinite(self.amp_over_complex)
                & (self.snr_values_db <= threshold)
            )
            x_values = self.snr_values_db[mask]
            y_values = self.amp_over_complex[mask]
            self._update_region_scatters(
                left_scatters,
                mask,
                self.snr_values_db,
                self.amp_over_complex,
            )
            fit_x, fit_y, fit_params = compute_linear_fit(x_values, y_values)
            left_fit.set_data(fit_x, fit_y)
            left_text.set_text(
                self._compose_region_text(mask, x_values, y_values, fit_params)
                + f"\nMax SNR = {threshold:.1f} dB"
            )
            self._update_selected_pixel_map(left_map_image, mask)
            fig.canvas.draw_idle()

        def update_middle(_value=None):
            threshold = float(middle_slider.val)
            mask = (
                np.isfinite(self.phase_std_over_limit)
                & np.isfinite(self.complex_over_amp)
                & np.isfinite(self.snr_values_db)
                & (self.snr_values_db >= threshold)
            )
            x_values = self.phase_std_over_limit[mask]
            y_values = self.complex_over_amp[mask]
            self._update_region_scatters(
                right_scatters,
                mask,
                self.phase_std_over_limit,
                self.complex_over_amp,
            )
            fit_x, fit_y, fit_params = compute_linear_fit(x_values, y_values)
            right_fit.set_data(fit_x, fit_y)
            right_text.set_text(
                self._compose_region_text(mask, x_values, y_values, fit_params)
                + f"\nThreshold = {threshold:.1f} dB"
            )
            self._update_selected_pixel_map(middle_map_image, mask)
            fig.canvas.draw_idle()

        def update_right(_value=None):
            threshold = float(right_slider.val)
            mask = (
                np.isfinite(self.intensity_values)
                & np.isfinite(self.phase_std_over_limit)
                & np.isfinite(self.snr_values_db)
                & (self.snr_values_db <= threshold)
            )
            x_values = self.intensity_values[mask]
            y_values = self.phase_std_over_limit[mask]
            self._update_region_scatters(
                bottom_scatters,
                mask,
                self.intensity_values,
                self.phase_std_over_limit,
            )
            fit_x, fit_y, fit_params = compute_linear_fit(x_values, y_values)
            bottom_fit.set_data(fit_x, fit_y)
            bottom_text.set_text(
                self._compose_region_text(mask, x_values, y_values, fit_params)
                + f"\nMax SNR = {threshold:.1f} dB"
            )
            self._update_selected_pixel_map(right_map_image, mask)
            fig.canvas.draw_idle()

        left_slider.on_changed(update_left)
        middle_slider.on_changed(update_middle)
        right_slider.on_changed(update_right)
        update_left()
        update_middle()
        update_right()
        plt.show(block=True)

    def _snr_slider_limits(self):
        valid = self.snr_values_db[np.isfinite(self.snr_values_db)]
        if valid.size == 0:
            return 0.0, 40.0
        lower = float(np.floor(np.percentile(valid, 1.0)))
        upper = float(np.ceil(np.percentile(valid, 99.5)))
        if upper <= lower:
            lower = float(np.floor(np.min(valid)))
            upper = float(np.ceil(np.max(valid)))
        if upper <= lower:
            upper = lower + 1.0
        return lower, upper

    def _make_region_scatters(self, axis):
        scatter = axis.scatter(
            [],
            [],
            s=DEFAULT_SCATTER_SIZE,
            c="black",
            alpha=DEFAULT_SCATTER_ALPHA,
            edgecolors="none",
            rasterized=True,
        )
        return {"tissue": scatter}

    def _update_region_scatters(self, scatters, global_mask, x_all, y_all):
        scatter = scatters["tissue"]
        x_values = x_all[global_mask]
        y_values = y_all[global_mask]
        if x_values.size:
            scatter.set_offsets(np.column_stack([x_values, y_values]))
        else:
            scatter.set_offsets(np.empty((0, 2)))

    def _compose_region_text(self, global_mask, x_values, y_values, fit_params=None):
        del global_mask
        text = compute_correlation_text(x_values, y_values)
        if fit_params is None:
            return text
        slope, intercept = fit_params
        return f"{text}\nSlope = {slope:.4g}\nIntercept = {intercept:.4g}"

    def _initialize_selected_pixel_map(self, axis, title):
        axis.imshow(self.mean_abs_bline.T, aspect="auto", origin="upper", cmap="gray")
        overlay = np.full(self.mean_abs_bline.shape, np.nan, dtype=np.float32)
        image_handle = axis.imshow(
            overlay.T,
            aspect="auto",
            origin="upper",
            cmap="autumn",
            vmin=0.0,
            vmax=1.0,
            alpha=0.75,
        )
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Depth index")
        return image_handle

    def _update_selected_pixel_map(self, image_handle, global_mask):
        selection_map = np.full(self.mean_abs_bline.shape, np.nan, dtype=np.float32)
        selected_tissue_mask = np.zeros(self.mean_abs_bline.shape, dtype=bool)
        selected_tissue_mask[self.tissue_mask] = global_mask
        selection_map[selected_tissue_mask] = 1.0
        image_handle.set_data(selection_map.T)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Interactive correlation explorer for dynamic OCT amplitude/complex signals.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="AMP+PHASE TIFF stack or directory of TIFF stacks.",
    )
    parser.add_argument(
        "--background-input-path",
        default=DEFAULT_BACKGROUND_INPUT_PATH,
        help="Optional separate background/noise AMP+PHASE TIFF stack or directory.",
    )
    parser.add_argument(
        "--frame-rate-hz",
        type=float,
        default=DEFAULT_FRAME_RATE_HZ,
        help="Acquisition rate in Hz.",
    )
    parser.add_argument(
        "--tissue-duration-seconds",
        type=float,
        default=DEFAULT_TISSUE_DURATION_SECONDS,
        help="Only use the first N seconds of the tissue stack.",
    )
    parser.add_argument(
        "--background-duration-seconds",
        type=float,
        default=DEFAULT_BACKGROUND_DURATION_SECONDS,
        help="Only use the first N seconds of the background stack.",
    )
    return parser


def load_settings_from_defaults():
    return argparse.Namespace(
        input_dir=DEFAULT_INPUT_DIR,
        background_input_path=DEFAULT_BACKGROUND_INPUT_PATH,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        tissue_duration_seconds=DEFAULT_TISSUE_DURATION_SECONDS,
        background_duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
    )


def main():
    if USE_COMMAND_LINE_ARGS:
        args = build_argument_parser().parse_args()
    else:
        args = load_settings_from_defaults()

    stack_paths = iter_tiff_stacks(args.input_dir)
    stack_path = stack_paths[0]
    output_dir = output_directory_for_stack(stack_path)

    print(f"Loading tissue AMP+PHASE stack: {stack_path}")
    complex_stack = read_amp_phase_tiff_stack(stack_path)
    complex_stack = limit_stack_duration(
        complex_stack,
        frame_rate_hz=args.frame_rate_hz,
        duration_seconds=args.tissue_duration_seconds,
    )

    sigma_q, sigma_q_source = load_external_noise_sigma_q(
        args.background_input_path,
        frame_rate_hz=args.frame_rate_hz,
        background_duration_seconds=args.background_duration_seconds,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    )

    if DEFAULT_NOTCH_BAND_HZ is None:
        roi_stack = complex_stack
    else:
        roi_stack = notch_filter_complex_stack_amp_phase(
            complex_stack,
            frame_rate_hz=args.frame_rate_hz,
            notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )

    mean_abs_bline = np.mean(np.abs(roi_stack), axis=0, dtype=np.float32)
    roi_vertices_map = load_or_select_roi_set(mean_abs_bline, stack_path, output_dir)
    tissue_mask = polygon_mask_for_image_shape(roi_vertices_map["tissue"], mean_abs_bline.shape)

    if not np.isfinite(sigma_q) or sigma_q <= 0:
        sigma_q, sigma_q_source = estimate_tissue_fallback_sigma_q(roi_stack)

    print(f"Using sigma_q = {sigma_q:.6g}")
    if sigma_q_source:
        print(f"sigma_q source: {sigma_q_source}")

    explorer = DynamicOCTCorrelationExplorer(
        complex_stack=complex_stack,
        stack_path=stack_path,
        frame_rate_hz=args.frame_rate_hz,
        sigma_q=sigma_q,
        sigma_q_source=sigma_q_source,
        tissue_mask=tissue_mask,
        roi_vertices_map=roi_vertices_map,
        phase_noise_variance_coefficient=DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT,
        phase_noise_variance_floor_rad2=DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    )
    explorer.show()


if __name__ == "__main__":
    main()
