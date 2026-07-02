import argparse
import csv
import re
from pathlib import Path
from statistics import NormalDist

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_TISSUE_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\200Hz 2seconds Blines\Bline-6-Yrpt200-X1264-Z132.tif"
DEFAULT_BACKGROUND_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"
DEFAULT_OUTPUT_DIR = None  # None saves into tissue_stack.parent / "phase_excess_noise_analysis".
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_TISSUE_DURATION_SECONDS = 2.0
DEFAULT_BACKGROUND_DURATION_SECONDS = 2.0
DEFAULT_PHASE_NOISE_BOUND_COEFFICIENT = 0.53  # Replace with your current upper-bound fit C.
DEFAULT_PHASE_NOISE_BOUND_FLOOR_RAD2 = 0.0  # Replace with your current upper-bound phase-variance floor.
DEFAULT_MAIN_FIGURE_SNR_LOWER_LIMIT_DB = 15.0
DEFAULT_PHASE_NOISE_CENTRAL_PERCENTILE = 99.0
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 1
DEFAULT_SNR_SCATTER_MAX_POINTS = 30000
DEFAULT_FIGURE_DPI = 400
DEFAULT_SHOW_FIGURES = False
DEFAULT_REUSE_SAVED_ROI = True

FONT_SIZES = {
    "suptitle": 26,
    "title": 24,
    "label": 22,
    "tick": 22,
    "legend": 22,
    "info": 22,
}

# Keep False for Spyder/IPython. Set True only when running from a terminal.
USE_COMMAND_LINE_ARGS = False


def load_stack(path):
    with TIFF.TiffFile(path) as tif:
        stack = np.stack([page.asarray() for page in tif.pages], axis=0)
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got shape {stack.shape}")
    return stack


def read_saved_amp_phase_tiff_stack(path):
    stack = load_stack(path)
    if stack.shape[-1] % 2 != 0:
        raise ValueError(
            "Saved AMP+PHASE TIFF depth dimension must be even. "
            f"Got shape {stack.shape} from {path}"
        )
    z_pixels = stack.shape[-1] // 2
    amplitude_stack = np.ascontiguousarray(stack[..., :z_pixels], dtype=np.float32)
    phase_stack = np.ascontiguousarray(stack[..., z_pixels:], dtype=np.float32)
    return amplitude_stack, phase_stack


def reconstruct_complex_stack(amplitude_stack, phase_stack):
    amplitude_stack = np.asarray(amplitude_stack, dtype=np.float32)
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    return (amplitude_stack * np.exp(1j * phase_stack)).astype(np.complex64, copy=False)


def limit_stack_duration(amplitude_stack, phase_stack, frame_rate_hz, duration_seconds):
    if duration_seconds is None:
        return amplitude_stack, phase_stack
    max_frames = int(np.floor(float(frame_rate_hz) * float(duration_seconds)))
    if max_frames < 1:
        return amplitude_stack[:1].copy(), phase_stack[:1].copy()
    if amplitude_stack.shape[0] <= max_frames:
        return amplitude_stack, phase_stack
    return (
        np.ascontiguousarray(amplitude_stack[:max_frames], dtype=np.float32),
        np.ascontiguousarray(phase_stack[:max_frames], dtype=np.float32),
    )


def estimate_sigma_q_from_complex_samples(complex_samples):
    complex_samples = np.asarray(complex_samples, dtype=np.complex64)
    if complex_samples.size == 0:
        raise ValueError("Noise-only complex sample array is empty.")
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
        raise ValueError(f"Invalid sigma_q estimated from noise stack: {sigma_q}")
    return sigma_q


def circular_phase_std_axis0(phase_stack):
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    unit_phasor = np.exp(1j * phase_stack).astype(np.complex64, copy=False)
    resultant = np.mean(unit_phasor, axis=0, dtype=np.complex64)
    resultant_magnitude = np.abs(resultant).astype(np.float32, copy=False)
    resultant_magnitude = np.clip(resultant_magnitude, 1e-8, 1.0)
    circular_std = np.sqrt(np.maximum(-2.0 * np.log(resultant_magnitude), 0.0)).astype(
        np.float32,
        copy=False,
    )
    return circular_std


def gpu_style_dynamic_from_real_stack(stack, uniform_filter_size):
    filtered = uniform_filter1d(
        np.asarray(stack, dtype=np.float32),
        size=max(1, int(uniform_filter_size)),
        axis=0,
        mode="nearest",
    )
    return np.var(filtered, axis=0, dtype=np.float32).astype(np.float32, copy=False)


def complex_dynamic_from_amp_phase(amplitude_stack, phase_stack, uniform_filter_size):
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


def normalize_roi_key(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


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

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
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


def output_directory_for_stack(stack_path, configured_output_dir):
    if configured_output_dir is not None:
        output_dir = Path(configured_output_dir)
    else:
        output_dir = Path(stack_path).parent / "phase_excess_noise_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_saved_tissue_vertices(stack_path, local_output_dir):
    stack_path = Path(stack_path)
    stack_stem = stack_path.stem
    workbook_candidates = [
        Path(local_output_dir) / f"{stack_stem}_phase_excess_noise_metrics.xlsx",
        stack_path.parent / "dynamic_timepoint_sufficiency" / f"{stack_stem}_dynamic_timepoint_metrics.xlsx",
        stack_path.parent / "frequency_band_power_analysis" / f"{stack_stem}_frequency_domain_metrics.xlsx",
    ]
    csv_candidates = [
        Path(local_output_dir) / f"{stack_stem}_phase_excess_noise_roi.csv",
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
            vertices = vertices_from_records(records, stack_path)
            if vertices is not None:
                print(f"Loaded tissue ROI from {workbook_path}")
                return vertices

    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", newline="") as file:
                records = list(csv.DictReader(file))
            vertices = vertices_from_records(records, stack_path)
            if vertices is not None:
                print(f"Loaded tissue ROI from {csv_path}")
                return vertices
        except Exception as error:
            print(f"Could not read ROI CSV {csv_path} ({error}); trying next fallback.")

    return None


def vertices_from_records(records, stack_path):
    if not records:
        return None

    if "source_stack" in records[0]:
        keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_path.stem)}
        subset = [
            record
            for record in records
            if str(record.get("roi", "")).strip().lower() == "tissue"
            and normalize_roi_key(record.get("source_stack", "")) in keys
        ]
    else:
        keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_path.stem)}
        subset = [
            record
            for record in records
            if str(record.get("roi", "")).strip().lower() == "tissue"
            and normalize_roi_key(record.get("label", "")) in keys
        ]

    if not subset:
        return None

    subset.sort(key=lambda record: int(record["vertex_index"]))
    vertices = np.asarray(
        [[float(record["x_pixel"]), float(record["depth_pixel"])] for record in subset],
        dtype=np.float32,
    )
    if vertices.shape[0] < 3:
        return None
    return vertices


def save_tissue_roi(local_output_dir, stack_path, vertices):
    records = [
        {
            "roi": "tissue",
            "source_stack": stack_path.name,
            "vertex_index": int(index),
            "x_pixel": float(vertex[0]),
            "depth_pixel": float(vertex[1]),
        }
        for index, vertex in enumerate(np.asarray(vertices, dtype=np.float32))
    ]
    workbook_path = Path(local_output_dir) / f"{stack_path.stem}_phase_excess_noise_metrics.xlsx"
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

    csv_path = Path(local_output_dir) / f"{stack_path.stem}_phase_excess_noise_roi.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved ROI CSV: {csv_path}")


def load_or_select_tissue_roi(mean_amplitude_image, stack_path, local_output_dir):
    if DEFAULT_REUSE_SAVED_ROI:
        saved_vertices = load_saved_tissue_vertices(stack_path, local_output_dir)
        if saved_vertices is not None:
            return polygon_mask_for_image_shape(saved_vertices, mean_amplitude_image.shape), saved_vertices

    vertices = select_polygon_roi(
        mean_amplitude_image,
        f"{Path(stack_path).name}: draw TISSUE ROI, then press Enter",
    )
    save_tissue_roi(local_output_dir, Path(stack_path), vertices)
    return polygon_mask_for_image_shape(vertices, mean_amplitude_image.shape), vertices


def expected_phase_variance_from_snr_db(snr_db, coefficient, variance_floor_rad2):
    snr_db = np.asarray(snr_db, dtype=np.float32)
    return (
        float(coefficient) * np.power(10.0, -snr_db / 10.0)
        + float(variance_floor_rad2)
    ).astype(np.float32, copy=False)


def percentile_band_sigma_multiplier(central_percentile):
    central_percentile = float(central_percentile)
    central_percentile = min(max(central_percentile, 0.0), 99.999)
    tail_probability = 0.5 + 0.5 * (central_percentile / 100.0)
    return float(NormalDist().inv_cdf(tail_probability))


def snr_db_map_from_mean_amplitude(mean_amplitude, sigma_q):
    mean_amplitude = np.asarray(mean_amplitude, dtype=np.float32)
    snr_linear = (mean_amplitude * mean_amplitude) / np.float32(2.0 * sigma_q * sigma_q)
    snr_linear = np.maximum(snr_linear, np.float32(1e-12))
    return (10.0 * np.log10(snr_linear)).astype(np.float32, copy=False)


def downsample_for_scatter(x_values, y_values, max_points):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size <= int(max_points):
        return x_values, y_values
    indices = np.linspace(0, x_values.size - 1, int(max_points), dtype=np.int32)
    return x_values[indices], y_values[indices]


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
    fit_x = np.linspace(np.min(x_values), np.max(x_values), 200, dtype=np.float32)
    fit_y = slope * fit_x + intercept
    return fit_x, fit_y.astype(np.float32, copy=False), (float(slope), float(intercept))


def compute_fit_statistics(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return None

    x_values = x_values[valid]
    y_values = y_values[valid]
    if np.std(x_values) <= 0 or np.std(y_values) <= 0:
        return None

    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residual = y_values - predicted
    ss_res = float(np.sum(residual * residual, dtype=np.float64))
    ss_tot = float(np.sum((y_values - np.mean(y_values, dtype=np.float32)) ** 2, dtype=np.float64))
    r_squared = np.nan
    if ss_tot > 0:
        r_squared = 1.0 - (ss_res / ss_tot)

    pearson_r = float(np.corrcoef(x_values, y_values)[0, 1])
    pearson_p = np.nan
    if pearsonr is not None:
        pearson_result = pearsonr(x_values, y_values)
        if hasattr(pearson_result, "pvalue"):
            pearson_p = float(pearson_result.pvalue)
        elif isinstance(pearson_result, tuple) and len(pearson_result) > 1:
            pearson_p = float(pearson_result[1])

    return {
        "n": count,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
        "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
    }


def finite_clim(image, percentile_low=1.0, percentile_high=99.7):
    values = np.asarray(image, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [percentile_low, percentile_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0, 1.0
    return float(low), float(high)


def mask_image_outside_roi_to_zero(image, roi_mask):
    masked = np.zeros(np.asarray(image).shape, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    masked[np.asarray(roi_mask, dtype=bool)] = image[np.asarray(roi_mask, dtype=bool)]
    return masked


def select_default_tissue_pixel(tissue_mask, snr_db_map):
    valid_mask = np.asarray(tissue_mask, dtype=bool) & np.isfinite(np.asarray(snr_db_map, dtype=np.float32))
    if not np.any(valid_mask):
        indices = np.argwhere(np.asarray(tissue_mask, dtype=bool))
        if indices.size == 0:
            return 0, 0
        x_index, z_index = indices[0]
        return int(x_index), int(z_index)
    candidate_indices = np.argwhere(valid_mask)
    candidate_snr = snr_db_map[valid_mask]
    best_idx = int(np.nanargmax(candidate_snr))
    x_index, z_index = candidate_indices[best_idx]
    return int(x_index), int(z_index)


def save_roi_overlay(mean_amplitude_image, vertices, output_path, title):
    plt.rcParams.update({"font.size": FONT_SIZES["label"]})
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.imshow(mean_amplitude_image.T, aspect="auto", origin="upper", cmap="gray")
    closed_vertices = np.vstack([vertices, vertices[0]])
    ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], color="cyan", linewidth=1.8)
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel("X pixel", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Depth index", fontsize=FONT_SIZES["label"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def save_summary_figure(
    snr_db_map,
    complex_dynamic_image,
    amplitude_dynamic_image,
    phase_std_map,
    normalized_phase_map,
    wrapped_phase_stack,
    time_axis_s,
    noise_std_map,
    tissue_vertices,
    default_pixel,
    output_path,
    stack_name,
):
    plt.rcParams.update({"font.size": FONT_SIZES["label"]})
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.5), constrained_layout=True)
    axes = np.asarray(axes)

    snr_vmin, snr_vmax = finite_clim(snr_db_map)
    complex_vmin, complex_vmax = finite_clim(complex_dynamic_image)
    amplitude_vmin, amplitude_vmax = finite_clim(amplitude_dynamic_image)
    phase_vmin, phase_vmax = finite_clim(phase_std_map)
    ratio_values = normalized_phase_map[np.isfinite(normalized_phase_map)]
    ratio_vmin, ratio_vmax = 0.0, max(2.0, float(np.nanpercentile(ratio_values, 99.0))) if ratio_values.size else 2.0

    panels = [
        (axes[0, 0], snr_db_map, "SNR map (dB)", "viridis", snr_vmin, snr_vmax),
        (axes[0, 1], complex_dynamic_image, "Complex dynamic signal", "magma", complex_vmin, complex_vmax),
        (axes[0, 2], amplitude_dynamic_image, "Amplitude dynamic signal", "magma", amplitude_vmin, amplitude_vmax),
        (axes[1, 0], phase_std_map, "Observed circular phase std (rad)", "magma", phase_vmin, phase_vmax),
        (
            axes[1, 1],
            normalized_phase_map,
            f"Observed / SNR-limited phase std\n(SNR >= {DEFAULT_MAIN_FIGURE_SNR_LOWER_LIMIT_DB:g} dB)",
            "plasma",
            ratio_vmin,
            ratio_vmax,
        ),
    ]

    closed_vertices = np.vstack([tissue_vertices, tissue_vertices[0]])
    for axis, image, title, cmap, vmin, vmax in panels:
        im = axis.imshow(image.T, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.plot(closed_vertices[:, 0], closed_vertices[:, 1], color="cyan", linewidth=1.4)
        axis.set_title(title, fontsize=FONT_SIZES["title"])
        axis.set_xlabel("X pixel", fontsize=FONT_SIZES["label"])
        axis.set_ylabel("Depth index", fontsize=FONT_SIZES["label"])
        axis.tick_params(labelsize=FONT_SIZES["tick"])
        colorbar = fig.colorbar(im, ax=axis, shrink=0.84)
        colorbar.ax.tick_params(labelsize=FONT_SIZES["tick"])

    x_index, z_index = default_pixel
    plot_phase_trace_panel(
        axes[1, 2],
        wrapped_phase_stack[:, x_index, z_index],
        time_axis_s,
        noise_std_map[x_index, z_index],
        pixel_title=f"Phase trace at X={x_index}, depth={z_index}",
    )

    fig.suptitle(f"{stack_name}: phase-noise excess summary", fontsize=FONT_SIZES["suptitle"])
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def plot_phase_trace_panel(axis, phase_trace, time_axis_s, noise_std_value, pixel_title):
    axis.clear()
    phase_trace = np.asarray(phase_trace, dtype=np.float32)
    time_axis_s = np.asarray(time_axis_s, dtype=np.float32)
    axis.plot(time_axis_s, phase_trace, color="black", linewidth=1.3)
    phase_mean = float(np.mean(phase_trace, dtype=np.float32)) if phase_trace.size else 0.0
    sigma_multiplier = percentile_band_sigma_multiplier(DEFAULT_PHASE_NOISE_CENTRAL_PERCENTILE)
    if np.isfinite(noise_std_value) and noise_std_value > 0:
        upper = phase_mean + sigma_multiplier * float(noise_std_value)
        lower = phase_mean - sigma_multiplier * float(noise_std_value)
        axis.axhline(upper, color="red", linestyle="--", linewidth=1.6)
        axis.axhline(lower, color="red", linestyle="--", linewidth=1.6)
    axis.set_title(pixel_title, fontsize=FONT_SIZES["title"])
    axis.set_xlabel("Time (s)", fontsize=FONT_SIZES["label"])
    axis.set_ylabel("Wrapped phase (rad)", fontsize=FONT_SIZES["label"])
    axis.tick_params(labelsize=FONT_SIZES["tick"])
    axis.grid(True, alpha=0.28)


def show_interactive_summary_figure(
    snr_db_map,
    complex_dynamic_image,
    amplitude_dynamic_image,
    phase_std_map,
    normalized_phase_map,
    wrapped_phase_stack,
    time_axis_s,
    noise_std_map,
    tissue_mask,
    tissue_vertices,
    default_pixel,
    stack_name,
):
    plt.rcParams.update({"font.size": FONT_SIZES["label"]})
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.5), constrained_layout=True)
    axes = np.asarray(axes)

    snr_vmin, snr_vmax = finite_clim(snr_db_map)
    complex_vmin, complex_vmax = finite_clim(complex_dynamic_image)
    amplitude_vmin, amplitude_vmax = finite_clim(amplitude_dynamic_image)
    phase_vmin, phase_vmax = finite_clim(phase_std_map)
    ratio_values = normalized_phase_map[np.isfinite(normalized_phase_map)]
    ratio_vmin, ratio_vmax = 0.0, max(2.0, float(np.nanpercentile(ratio_values, 99.0))) if ratio_values.size else 2.0

    closed_vertices = np.vstack([tissue_vertices, tissue_vertices[0]])
    panels = [
        (axes[0, 0], snr_db_map, "SNR map (dB)", "viridis", snr_vmin, snr_vmax),
        (axes[0, 1], complex_dynamic_image, "Complex dynamic signal", "magma", complex_vmin, complex_vmax),
        (axes[0, 2], amplitude_dynamic_image, "Amplitude dynamic signal", "magma", amplitude_vmin, amplitude_vmax),
        (axes[1, 0], phase_std_map, "Observed circular phase std (rad)", "magma", phase_vmin, phase_vmax),
        (
            axes[1, 1],
            normalized_phase_map,
            f"Observed / SNR-limited phase std\n(SNR >= {DEFAULT_MAIN_FIGURE_SNR_LOWER_LIMIT_DB:g} dB)",
            "plasma",
            ratio_vmin,
            ratio_vmax,
        ),
    ]

    for axis, image, title, cmap, vmin, vmax in panels:
        im = axis.imshow(image.T, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.plot(closed_vertices[:, 0], closed_vertices[:, 1], color="cyan", linewidth=1.4)
        axis.set_title(title, fontsize=FONT_SIZES["title"])
        axis.set_xlabel("X pixel", fontsize=FONT_SIZES["label"])
        axis.set_ylabel("Depth index", fontsize=FONT_SIZES["label"])
        axis.tick_params(labelsize=FONT_SIZES["tick"])
        colorbar = fig.colorbar(im, ax=axis, shrink=0.84)
        colorbar.ax.tick_params(labelsize=FONT_SIZES["tick"])

    phase_axis = axes[1, 2]
    marker_artists = []
    for axis in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        marker, = axis.plot([], [], marker="+", color="white", markersize=12, markeredgewidth=1.8)
        marker_artists.append(marker)

    info_text = fig.text(
        0.5,
        0.01,
        "Click a tissue pixel in any panel to inspect SNR and normalized phase ratio.",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["info"],
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    def update_pixel_views(x_index, z_index):
        for marker in marker_artists:
            marker.set_data([x_index], [z_index])
        snr_value = float(snr_db_map[x_index, z_index])
        ratio_value = float(normalized_phase_map[x_index, z_index])
        plot_phase_trace_panel(
            phase_axis,
            wrapped_phase_stack[:, x_index, z_index],
            time_axis_s,
            noise_std_map[x_index, z_index],
            pixel_title=f"Phase trace at X={x_index}, depth={z_index}",
        )
        info_text.set_text(
            f"X={x_index}, depth={z_index}, SNR={snr_value:.3f} dB, "
            f"observed/std-limit={ratio_value:.4f}"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in {axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]}:
            return
        x_index = int(round(event.xdata))
        z_index = int(round(event.ydata))
        if x_index < 0 or x_index >= tissue_mask.shape[0] or z_index < 0 or z_index >= tissue_mask.shape[1]:
            return
        if not tissue_mask[x_index, z_index]:
            info_text.set_text("Clicked pixel is outside the tissue ROI.")
            fig.canvas.draw_idle()
            return

        update_pixel_views(x_index, z_index)

    fig.canvas.mpl_connect("button_press_event", on_click)
    update_pixel_views(int(default_pixel[0]), int(default_pixel[1]))
    fig.suptitle(f"{stack_name}: interactive phase-noise excess summary", fontsize=FONT_SIZES["suptitle"])
    plt.show(block=True)
    plt.close(fig)


def save_scatter_figure(snr_values_db, normalized_phase_values, output_path, stack_name):
    x_values, y_values = downsample_for_scatter(
        snr_values_db,
        normalized_phase_values,
        DEFAULT_SNR_SCATTER_MAX_POINTS,
    )
    plt.rcParams.update({"font.size": FONT_SIZES["label"]})
    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    ax.scatter(
        x_values,
        y_values,
        s=5.0,
        c="black",
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.8, label="Noise-limit ratio = 1")
    ax.set_xlabel("SNR (dB)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Observed phase std / SNR-limited phase std", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{stack_name}: SNR vs normalized phase fluctuation", fontsize=FONT_SIZES["suptitle"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=FONT_SIZES["legend"])
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def save_phase_ratio_vs_dynamic_scatter(
    phase_ratio_values,
    dynamic_values,
    output_path,
    stack_name,
    dynamic_label,
):
    x_values, y_values = downsample_for_scatter(
        phase_ratio_values,
        dynamic_values,
        DEFAULT_SNR_SCATTER_MAX_POINTS,
    )
    plt.rcParams.update({"font.size": FONT_SIZES["label"]})
    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    ax.scatter(
        x_values,
        y_values,
        s=5.0,
        c="black",
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
    )
    fit_x, fit_y, fit_params = compute_linear_fit(x_values, y_values)
    fit_stats = compute_fit_statistics(x_values, y_values)
    if fit_params is not None and fit_stats is not None:
        ax.plot(fit_x, fit_y, color="tab:blue", linestyle="--", linewidth=2.0, label="Linear fit")
        pearson_p_text = (
            f"{fit_stats['pearson_p']:.3g}"
            if np.isfinite(fit_stats["pearson_p"])
            else "n/a"
        )
        ax.text(
            0.02,
            0.98,
            (
                f"N = {fit_stats['n']}\n"
                f"Slope = {fit_stats['slope']:.4g}\n"
                f"Intercept = {fit_stats['intercept']:.4g}\n"
                f"R^2 = {fit_stats['r_squared']:.4g}\n"
                f"Pearson r = {fit_stats['pearson_r']:.4g}\n"
                f"Pearson p = {pearson_p_text}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FONT_SIZES["info"],
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.set_xlabel("Observed circular phase std / SNR-limited phase std", fontsize=FONT_SIZES["label"])
    ax.set_ylabel(dynamic_label, fontsize=FONT_SIZES["label"])
    ax.set_title(f"{stack_name}: phase ratio vs {dynamic_label.lower()}", fontsize=FONT_SIZES["suptitle"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    ax.grid(True, alpha=0.28)
    if fit_params is not None and fit_stats is not None:
        ax.legend(frameon=False, fontsize=FONT_SIZES["legend"])
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def save_metrics_workbook(output_path, roi_vertices, scatter_snr_db, scatter_ratio, sigma_q, stack_name):
    roi_records = [
        {
            "roi": "tissue",
            "source_stack": stack_name,
            "vertex_index": int(index),
            "x_pixel": float(vertex[0]),
            "depth_pixel": float(vertex[1]),
        }
        for index, vertex in enumerate(np.asarray(roi_vertices, dtype=np.float32))
    ]
    scatter_records = [
        {
            "snr_db": float(snr_db),
            "observed_over_noise_limit": float(ratio),
        }
        for snr_db, ratio in zip(scatter_snr_db, scatter_ratio)
    ]
    summary_records = [
        {
            "stack_name": stack_name,
            "sigma_q": float(sigma_q),
            "phase_noise_bound_coefficient": float(DEFAULT_PHASE_NOISE_BOUND_COEFFICIENT),
            "phase_noise_bound_floor_rad2": float(DEFAULT_PHASE_NOISE_BOUND_FLOOR_RAD2),
            "frame_rate_hz": float(DEFAULT_FRAME_RATE_HZ),
            "tissue_duration_seconds": float(DEFAULT_TISSUE_DURATION_SECONDS),
            "background_duration_seconds": float(DEFAULT_BACKGROUND_DURATION_SECONDS),
        }
    ]

    if pd is None:
        write_csv(output_path.with_suffix(".roi_vertices.csv"), roi_records)
        write_csv(output_path.with_suffix(".scatter.csv"), scatter_records)
        write_csv(output_path.with_suffix(".summary.csv"), summary_records)
        print("pandas is not installed; saved CSV files instead of Excel workbook.")
        return

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame.from_records(summary_records).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame.from_records(roi_records).to_excel(writer, sheet_name="roi_vertices", index=False)
        pd.DataFrame.from_records(scatter_records).to_excel(writer, sheet_name="scatter_data", index=False)
    print(f"Saved metrics workbook: {output_path}")


def write_csv(output_path, records):
    if not records:
        return
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV: {output_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate SNR/phase-noise excess maps and scatter from one AMP+PHASE OCT stack.",
    )
    parser.add_argument("tissue_input_path", nargs="?", default=DEFAULT_TISSUE_INPUT_PATH)
    parser.add_argument("--background-input-path", default=DEFAULT_BACKGROUND_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-rate-hz", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--tissue-duration-seconds", type=float, default=DEFAULT_TISSUE_DURATION_SECONDS)
    parser.add_argument("--background-duration-seconds", type=float, default=DEFAULT_BACKGROUND_DURATION_SECONDS)
    parser.add_argument("--main-figure-snr-lower-limit-db", type=float, default=DEFAULT_MAIN_FIGURE_SNR_LOWER_LIMIT_DB)
    return parser


def load_settings_from_defaults():
    return argparse.Namespace(
        tissue_input_path=DEFAULT_TISSUE_INPUT_PATH,
        background_input_path=DEFAULT_BACKGROUND_INPUT_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        tissue_duration_seconds=DEFAULT_TISSUE_DURATION_SECONDS,
        background_duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
        main_figure_snr_lower_limit_db=DEFAULT_MAIN_FIGURE_SNR_LOWER_LIMIT_DB,
    )


def main():
    if USE_COMMAND_LINE_ARGS:
        args = build_argument_parser().parse_args()
    else:
        args = load_settings_from_defaults()

    if args.background_input_path is None or str(args.background_input_path).strip() == "":
        raise ValueError("A separate background noise stack is required for this analysis.")

    tissue_path = Path(args.tissue_input_path)
    output_dir = output_directory_for_stack(tissue_path, args.output_dir)

    print(f"Loading tissue AMP+PHASE stack: {tissue_path}")
    tissue_amplitude_stack, tissue_phase_stack = read_saved_amp_phase_tiff_stack(tissue_path)
    tissue_amplitude_stack, tissue_phase_stack = limit_stack_duration(
        tissue_amplitude_stack,
        tissue_phase_stack,
        frame_rate_hz=args.frame_rate_hz,
        duration_seconds=args.tissue_duration_seconds,
    )
    mean_amplitude_image = np.mean(tissue_amplitude_stack, axis=0, dtype=np.float32)
    tissue_mask, tissue_vertices = load_or_select_tissue_roi(mean_amplitude_image, tissue_path, output_dir)

    background_path = Path(args.background_input_path)
    print(f"Loading background AMP+PHASE stack for sigma_q: {background_path}")
    background_amplitude_stack, background_phase_stack = read_saved_amp_phase_tiff_stack(background_path)
    background_amplitude_stack, background_phase_stack = limit_stack_duration(
        background_amplitude_stack,
        background_phase_stack,
        frame_rate_hz=args.frame_rate_hz,
        duration_seconds=args.background_duration_seconds,
    )
    background_complex_stack = reconstruct_complex_stack(background_amplitude_stack, background_phase_stack)
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

    tissue_complex_stack = reconstruct_complex_stack(tissue_amplitude_stack, tissue_phase_stack)
    wrapped_phase_stack = np.angle(tissue_complex_stack).astype(np.float32, copy=False)
    observed_phase_std_map = circular_phase_std_axis0(wrapped_phase_stack)
    amplitude_dynamic_image = gpu_style_dynamic_from_real_stack(
        tissue_amplitude_stack,
        uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    )
    complex_dynamic_image = complex_dynamic_from_amp_phase(
        tissue_amplitude_stack,
        tissue_phase_stack,
        uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    )
    snr_db_map = snr_db_map_from_mean_amplitude(mean_amplitude_image, sigma_q)
    noise_variance_map = expected_phase_variance_from_snr_db(
        snr_db_map,
        coefficient=DEFAULT_PHASE_NOISE_BOUND_COEFFICIENT,
        variance_floor_rad2=DEFAULT_PHASE_NOISE_BOUND_FLOOR_RAD2,
    )
    noise_std_map = np.sqrt(np.maximum(noise_variance_map, 0.0)).astype(np.float32, copy=False)
    normalized_phase_map = np.full(noise_std_map.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(observed_phase_std_map) & np.isfinite(noise_std_map) & (noise_std_map > 0)
    normalized_phase_map[valid] = observed_phase_std_map[valid] / noise_std_map[valid]
    time_axis_s = np.arange(tissue_amplitude_stack.shape[0], dtype=np.float32) / np.float32(args.frame_rate_hz)
    default_pixel = select_default_tissue_pixel(tissue_mask, snr_db_map)

    main_figure_mask = tissue_mask & np.isfinite(snr_db_map) & (snr_db_map >= float(args.main_figure_snr_lower_limit_db))
    snr_db_map_display = mask_image_outside_roi_to_zero(snr_db_map, tissue_mask)
    complex_dynamic_image_display = mask_image_outside_roi_to_zero(complex_dynamic_image, tissue_mask)
    amplitude_dynamic_image_display = mask_image_outside_roi_to_zero(amplitude_dynamic_image, tissue_mask)
    observed_phase_std_map_display = mask_image_outside_roi_to_zero(observed_phase_std_map, tissue_mask)
    normalized_phase_map_display = np.zeros(normalized_phase_map.shape, dtype=np.float32)
    normalized_phase_map_display[main_figure_mask] = normalized_phase_map[main_figure_mask]

    scatter_snr_db = snr_db_map[tissue_mask].astype(np.float32, copy=False)
    scatter_ratio = normalized_phase_map[tissue_mask].astype(np.float32, copy=False)

    save_roi_overlay(
        mean_amplitude_image,
        tissue_vertices,
        output_dir / f"{tissue_path.stem}_tissue_roi_overlay.png",
        f"{tissue_path.name} mean OCT intensity with tissue ROI",
    )
    save_summary_figure(
        snr_db_map_display,
        complex_dynamic_image_display,
        amplitude_dynamic_image_display,
        observed_phase_std_map_display,
        normalized_phase_map_display,
        wrapped_phase_stack,
        time_axis_s,
        noise_std_map,
        tissue_vertices,
        default_pixel,
        output_dir / f"{tissue_path.stem}_phase_excess_summary.png",
        tissue_path.name,
    )
    show_interactive_summary_figure(
        snr_db_map_display,
        complex_dynamic_image_display,
        amplitude_dynamic_image_display,
        observed_phase_std_map_display,
        normalized_phase_map_display,
        wrapped_phase_stack,
        time_axis_s,
        noise_std_map,
        tissue_mask,
        tissue_vertices,
        default_pixel,
        tissue_path.name,
    )
    save_scatter_figure(
        scatter_snr_db,
        scatter_ratio,
        output_dir / f"{tissue_path.stem}_snr_vs_normalized_phase_scatter.png",
        tissue_path.name,
    )
    high_snr_scatter_mask = tissue_mask & np.isfinite(snr_db_map) & (snr_db_map >= float(args.main_figure_snr_lower_limit_db))
    save_phase_ratio_vs_dynamic_scatter(
        normalized_phase_map[high_snr_scatter_mask].astype(np.float32, copy=False),
        amplitude_dynamic_image[high_snr_scatter_mask].astype(np.float32, copy=False),
        output_dir / f"{tissue_path.stem}_phase_ratio_vs_amplitude_dynamic_scatter.png",
        tissue_path.name,
        "Amplitude dynamic signal",
    )
    save_phase_ratio_vs_dynamic_scatter(
        normalized_phase_map[high_snr_scatter_mask].astype(np.float32, copy=False),
        complex_dynamic_image[high_snr_scatter_mask].astype(np.float32, copy=False),
        output_dir / f"{tissue_path.stem}_phase_ratio_vs_complex_dynamic_scatter.png",
        tissue_path.name,
        "Complex dynamic signal",
    )
    save_metrics_workbook(
        output_dir / f"{tissue_path.stem}_phase_excess_noise_metrics.xlsx",
        tissue_vertices,
        scatter_snr_db,
        scatter_ratio,
        sigma_q,
        tissue_path.name,
    )
    print("Finished phase-noise excess analysis.")


if __name__ == "__main__":
    main()
