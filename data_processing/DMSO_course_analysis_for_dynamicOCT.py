import gc
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
import numpy as np
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF

try:
    import pandas as pd
except ImportError:
    pd = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_ROOT_DIR = r"E:\IOCTData\Lung Cancer mice 260601\260608\sampleID-1"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_DMSO_TIME_LABEL = "Time-8 +DMSO"
DEFAULT_TIMEPOINT_INTERVAL_MINUTES = 1.8
DEFAULT_ANALYSIS_DURATION_SECONDS = 1.0
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 10
DEFAULT_DYNAMIC_CHUNK_X = 96
DEFAULT_OUTPUT_DIR = None  # None saves into root_dir / "dmso_time_course_analysis".
DEFAULT_SAVE_DPI = 360
DEFAULT_METRICS_FONT_SIZE = 10
DEFAULT_MONTAGE_COLUMNS = 4
DEFAULT_HIGH_FREQUENCY_FRACTION_CUTOFF = 0.25
DEFAULT_SAVE_ROI_OVERLAYS = True
DEFAULT_ROI_SHIFT_SEARCH_PIXELS = 20
DEFAULT_ROI_SHIFT_PADDING_PIXELS = 10
DEFAULT_REUSE_SAVED_ROIS = True
DEFAULT_LAST_TIMEPOINT_WITH_SHIFT = None
# Example: set to 12 to estimate ROI vertical shift only through "Time-12".
# For later folders, the script will reuse the original time-point-1 ROIs with dz = 0.

TIME_FOLDER_RE = re.compile(r"Time-(?P<index>\d+)", re.IGNORECASE)


def iter_timepoint_folders(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    folders = []
    for item in root.iterdir():
        if item.is_dir():
            time_index = parse_time_label_to_index(item.name)
            if time_index is not None:
                folders.append((item, time_index))
    folders.sort(key=lambda entry: entry[1])
    if not folders:
        raise ValueError(f"No time-point folders found in: {root_dir}")
    return folders


def parse_time_label_to_index(label):
    match = TIME_FOLDER_RE.search(label)
    if match is None:
        return None
    return int(match.group("index"))


def relative_minutes_from_dmso(folder_name, dmso_label):
    folder_index = parse_time_label_to_index(folder_name)
    dmso_index = parse_time_label_to_index(dmso_label)
    if folder_index is None or dmso_index is None:
        return np.nan
    return float(folder_index - dmso_index) * float(DEFAULT_TIMEPOINT_INTERVAL_MINUTES)


def find_tiff_stacks(folder):
    paths = [
        item
        for item in Path(folder).iterdir()
        if (
            item.is_file()
            and item.suffix.lower() in {".tif", ".tiff"}
            and "bline" in item.name.lower()
        )
    ]
    paths.sort(key=lambda path: path.name.lower())
    if not paths:
        raise ValueError(f"No Bline TIFF stacks found in folder: {folder}")
    return paths


def position_key_from_path(path):
    return Path(path).stem


def normalize_roi_key(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


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


def compute_dynamic_images_for_duration(
    amplitude_stack,
    phase_stack,
    frame_count,
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

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.imshow(image.T, aspect="auto", origin="lower", cmap="gray")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth")
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
    return polygon_mask_for_image_shape(vertices, image.shape), np.asarray(vertices, dtype=np.float32)


def save_roi_overlay(image, vertices, title, output_path):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.imshow(image.T, aspect="auto", origin="lower", cmap="gray")
    polygon = np.asarray(vertices, dtype=np.float32)
    closed_polygon = np.vstack([polygon, polygon[0]])
    ax.plot(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        color="cyan",
        linewidth=1.8,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth")
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved ROI overlay: {output_path}")
    plt.close(fig)


def shift_vertices_in_depth(vertices, dz_pixels):
    shifted = np.asarray(vertices, dtype=np.float32).copy()
    shifted[:, 1] = shifted[:, 1] + float(dz_pixels)
    return shifted


def estimate_vertical_shift_from_roi(
    reference_image,
    current_image,
    reference_vertices,
    search_pixels=20,
    padding_pixels=10,
):
    reference_image = np.asarray(reference_image, dtype=np.float32)
    current_image = np.asarray(current_image, dtype=np.float32)
    vertices = np.asarray(reference_vertices, dtype=np.float32)
    if reference_image.ndim != 2 or current_image.ndim != 2:
        return 0

    x_min = max(0, int(np.floor(np.min(vertices[:, 0]))))
    x_max = min(reference_image.shape[0], int(np.ceil(np.max(vertices[:, 0]))) + 1)
    z_min = max(0, int(np.floor(np.min(vertices[:, 1]))) - int(padding_pixels))
    z_max = min(reference_image.shape[1], int(np.ceil(np.max(vertices[:, 1]))) + 1 + int(padding_pixels))
    if x_max <= x_min or z_max <= z_min:
        return 0

    current_z_max = min(current_image.shape[1], z_max)
    if current_z_max <= z_min:
        return 0

    ref_crop = reference_image[x_min:x_max, z_min:z_max]
    cur_crop = current_image[x_min:min(current_image.shape[0], x_max), z_min:current_z_max]
    if ref_crop.size == 0 or cur_crop.size == 0:
        return 0

    ref_profile = np.mean(ref_crop, axis=0, dtype=np.float32)
    cur_profile = np.mean(cur_crop, axis=0, dtype=np.float32)
    common_length = min(ref_profile.size, cur_profile.size)
    if common_length < 4:
        return 0

    ref_profile = ref_profile[:common_length] - np.mean(ref_profile[:common_length], dtype=np.float32)
    cur_profile = cur_profile[:common_length] - np.mean(cur_profile[:common_length], dtype=np.float32)
    if not np.any(np.isfinite(ref_profile)) or not np.any(np.isfinite(cur_profile)):
        return 0

    correlation = np.correlate(cur_profile, ref_profile, mode="full")
    lags = np.arange(-(common_length - 1), common_length, dtype=np.int32)
    valid = np.abs(lags) <= int(search_pixels)
    if not np.any(valid):
        return 0
    best_lag = int(lags[valid][np.argmax(correlation[valid])])
    return best_lag


def correlation_profile_for_shift_search(
    reference_image,
    current_image,
    reference_vertices,
    search_pixels=20,
    padding_pixels=10,
):
    reference_image = np.asarray(reference_image, dtype=np.float32)
    current_image = np.asarray(current_image, dtype=np.float32)
    vertices = np.asarray(reference_vertices, dtype=np.float32)
    if reference_image.ndim != 2 or current_image.ndim != 2:
        return None, None

    try:
        roi_mask = polygon_mask_for_image_shape(vertices, reference_image.shape)
    except ValueError:
        return None, None

    x_min = max(0, int(np.floor(np.min(vertices[:, 0]))))
    x_max = min(reference_image.shape[0], int(np.ceil(np.max(vertices[:, 0]))) + 1)
    z_min = max(0, int(np.floor(np.min(vertices[:, 1]))) - int(padding_pixels))
    z_max = min(reference_image.shape[1], int(np.ceil(np.max(vertices[:, 1]))) + 1 + int(padding_pixels))
    if x_max <= x_min or z_max <= z_min:
        return None, None

    current_x_max = min(current_image.shape[0], x_max)
    current_z_max = min(current_image.shape[1], z_max)
    if current_x_max <= x_min or current_z_max <= z_min:
        return None, None

    ref_crop = reference_image[x_min:x_max, z_min:z_max]
    cur_crop = current_image[x_min:current_x_max, z_min:current_z_max]
    if ref_crop.size == 0 or cur_crop.size == 0:
        return None, None

    roi_mask_crop = roi_mask[x_min:x_max, z_min:z_max]
    if roi_mask_crop.shape != ref_crop.shape:
        return None, None

    if not np.any(roi_mask_crop):
        return None, None

    # Use the axial gradient so boundary structure drives the match more than
    # slowly varying intensity or stationary broad background.
    ref_feature = np.gradient(ref_crop, axis=1).astype(np.float32, copy=False)
    cur_feature = np.gradient(cur_crop, axis=1).astype(np.float32, copy=False)

    lags = np.arange(-int(search_pixels), int(search_pixels) + 1, dtype=np.int32)
    scores = np.full(lags.shape, np.nan, dtype=np.float64)

    for lag_index, lag in enumerate(lags):
        if lag >= 0:
            ref_z0 = 0
            ref_z1 = min(ref_feature.shape[1], cur_feature.shape[1] - lag)
            cur_z0 = lag
            cur_z1 = cur_z0 + (ref_z1 - ref_z0)
        else:
            cur_z0 = 0
            cur_z1 = min(cur_feature.shape[1], ref_feature.shape[1] + lag)
            ref_z0 = -lag
            ref_z1 = ref_z0 + (cur_z1 - cur_z0)

        if ref_z1 <= ref_z0 or cur_z1 <= cur_z0:
            continue

        ref_slice = ref_feature[:, ref_z0:ref_z1]
        cur_slice = cur_feature[:, cur_z0:cur_z1]
        mask_slice = roi_mask_crop[:, ref_z0:ref_z1]
        if ref_slice.shape != cur_slice.shape or ref_slice.shape != mask_slice.shape:
            continue
        if np.count_nonzero(mask_slice) < 8:
            continue

        ref_values = ref_slice[mask_slice]
        cur_values = cur_slice[mask_slice]
        valid = np.isfinite(ref_values) & np.isfinite(cur_values)
        if np.count_nonzero(valid) < 8:
            continue

        ref_values = ref_values[valid].astype(np.float64, copy=False)
        cur_values = cur_values[valid].astype(np.float64, copy=False)
        ref_values = ref_values - np.mean(ref_values)
        cur_values = cur_values - np.mean(cur_values)

        ref_norm = np.sqrt(np.sum(ref_values * ref_values))
        cur_norm = np.sqrt(np.sum(cur_values * cur_values))
        if ref_norm <= 0 or cur_norm <= 0:
            continue

        scores[lag_index] = np.sum(ref_values * cur_values) / (ref_norm * cur_norm)

    valid_scores = np.isfinite(scores)
    if not np.any(valid_scores):
        return None, None
    return lags[valid_scores], scores[valid_scores]


def estimate_shared_vertical_shift(
    position_roi_map,
    current_mean_amp_map,
    search_pixels=20,
    padding_pixels=10,
):
    accumulated_scores = {}
    used_positions = 0

    for position_key, current_mean_amp in current_mean_amp_map.items():
        if position_key not in position_roi_map:
            continue
        roi_info = position_roi_map[position_key]
        lags, scores = correlation_profile_for_shift_search(
            roi_info["reference_mean_amp"],
            current_mean_amp,
            roi_info["tissue_vertices"],
            search_pixels=search_pixels,
            padding_pixels=padding_pixels,
        )
        if lags is None or scores is None:
            continue
        used_positions += 1
        for lag, score in zip(lags, scores):
            accumulated_scores[int(lag)] = accumulated_scores.get(int(lag), 0.0) + float(score)

    if used_positions == 0 or not accumulated_scores:
        return 0, 0

    best_lag = max(accumulated_scores.items(), key=lambda item: item[1])[0]
    return int(best_lag), int(used_positions)


def initial_image_clim(images):
    finite_arrays = [image[np.isfinite(image)].reshape(-1) for image in images if np.size(image) > 0]
    if not finite_arrays:
        return 0.0, 1.0
    values = np.concatenate(finite_arrays)
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1.0, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def safe_mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def safe_percentile(values, percentile):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def high_frequency_fraction(image, mask, cutoff_fraction):
    rows, cols = np.where(mask)
    if rows.size < 4 or cols.size < 4:
        return np.nan
    x0, x1 = int(rows.min()), int(rows.max()) + 1
    z0, z1 = int(cols.min()), int(cols.max()) + 1
    crop = np.asarray(image[x0:x1, z0:z1], dtype=np.float64)
    crop_mask = mask[x0:x1, z0:z1]
    if crop.size == 0 or np.count_nonzero(crop_mask) < 4:
        return np.nan
    crop = np.where(crop_mask, crop, np.nan)
    mean_value = np.nanmean(crop)
    if not np.isfinite(mean_value):
        return np.nan
    crop = np.where(crop_mask, crop - mean_value, 0.0)
    spectrum_power = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    total_power = np.sum(spectrum_power)
    if total_power <= 0 or not np.isfinite(total_power):
        return np.nan

    fx = np.fft.fftshift(np.fft.fftfreq(crop.shape[0]))
    fz = np.fft.fftshift(np.fft.fftfreq(crop.shape[1]))
    fx_grid, fz_grid = np.meshgrid(fx, fz, indexing="ij")
    radius = np.sqrt(fx_grid * fx_grid + fz_grid * fz_grid)
    high_frequency_mask = radius >= float(cutoff_fraction) * np.max(radius)
    return float(np.sum(spectrum_power[high_frequency_mask]) / total_power)


def compute_metrics(image, tissue_mask, background_mask):
    tissue_values = image[tissue_mask]
    background_values = image[background_mask]
    tissue_mean = safe_mean(tissue_values)
    tissue_std = safe_std(tissue_values)
    background_mean = safe_mean(background_values)
    background_std = safe_std(background_values)
    cnr = np.nan
    if np.isfinite(background_std) and background_std > 0:
        cnr = (tissue_mean - background_mean) / background_std

    return {
        "tissue_mean": tissue_mean,
        "tissue_std": tissue_std,
        "background_mean": background_mean,
        "background_std": background_std,
        "background_p95": safe_percentile(background_values, 95),
        "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
        "tissue_cv": tissue_std / tissue_mean if np.isfinite(tissue_mean) and tissue_mean != 0 else np.nan,
        "high_frequency_fraction": high_frequency_fraction(
            image,
            tissue_mask,
            DEFAULT_HIGH_FREQUENCY_FRACTION_CUTOFF,
        ),
    }


def output_directory(root_dir):
    if DEFAULT_OUTPUT_DIR is None:
        out_dir = Path(root_dir) / "dmso_time_course_analysis"
    else:
        out_dir = Path(DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def overlay_output_path(base_dir, folder_name, position_key):
    overlay_dir = Path(base_dir) / "roi_overlays" / folder_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return overlay_dir / f"{position_key}_mean_bline_roi_overlay.png"


def plot_time_course_montage(records, signal_type, position_key, output_path):
    subset = [
        record
        for record in records
        if record["signal_type"] == signal_type and record["position_key"] == position_key
    ]
    subset.sort(key=lambda record: record["relative_minutes"])
    if not subset:
        return

    images = [record["image"] for record in subset]
    vmin, vmax = initial_image_clim(images)
    columns = DEFAULT_MONTAGE_COLUMNS
    rows = int(np.ceil(len(subset) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.8 * rows), squeeze=False)
    last_im = None
    for idx, axis in enumerate(axes.flat):
        if idx >= len(subset):
            axis.axis("off")
            continue
        record = subset[idx]
        last_im = axis.imshow(
            record["image"].T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"{record['folder_name']}\n{record['relative_minutes']:.0f} min")
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Depth")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, label="Dynamic signal")
    fig.suptitle(f"{signal_type.capitalize()} dynamic, {position_key}", fontsize=12)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved montage: {output_path}")
    plt.close(fig)


def plot_metric_pair(axis, metric_records, position_key, metric, ylabel, title):
    for signal_type, marker in [("amplitude", "o-"), ("complex", "s-")]:
        subset = [
            record
            for record in metric_records
            if record["signal_type"] == signal_type and record["position_key"] == position_key
        ]
        subset.sort(key=lambda record: record["relative_minutes"])
        axis.plot(
            [record["relative_minutes"] for record in subset],
            [record[metric] for record in subset],
            marker,
            linewidth=1.8,
            markersize=4.5,
            label=signal_type.capitalize(),
        )
    axis.axvline(0, color="0.25", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Minutes after DMSO")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.28)


def plot_metrics_summary(metric_records, position_key, output_path):
    plt.rcParams.update({"font.size": DEFAULT_METRICS_FONT_SIZE})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.0), constrained_layout=True)
    axes = axes.reshape(-1)
    plot_metric_pair(axes[0], metric_records, position_key, "cnr", "CNR", "Tissue-background CNR")
    plot_metric_pair(axes[1], metric_records, position_key, "tissue_mean", "Tissue mean", "Tissue dynamic signal")
    plot_metric_pair(axes[2], metric_records, position_key, "background_std", "Background std", "Background noise")
    plot_metric_pair(axes[3], metric_records, position_key, "tissue_cv", "CV", "Tissue spatial CV")
    plot_metric_pair(
        axes[4],
        metric_records,
        position_key,
        "high_frequency_fraction",
        "Fraction",
        "High-frequency spatial power",
    )
    plot_metric_pair(
        axes[5],
        metric_records,
        position_key,
        "background_p95",
        "Background P95",
        "Background high-percentile signal",
    )
    axes[3].set_ylim(bottom=0)
    axes[4].set_ylim(0, 1)
    axes[5].set_ylim(bottom=0)
    for axis in axes:
        axis.legend(frameon=False, fontsize=DEFAULT_METRICS_FONT_SIZE - 1)
        axis.tick_params(labelsize=DEFAULT_METRICS_FONT_SIZE - 1)
    fig.suptitle(f"DMSO time-course metrics, {position_key}", fontsize=DEFAULT_METRICS_FONT_SIZE + 3)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved metrics figure: {output_path}")
    plt.close(fig)


def save_metrics_workbook(output_path, metric_records, roi_records):
    export_records = [
        {key: value for key, value in record.items() if key != "image"}
        for record in metric_records
    ]
    if pd is None:
        write_csv(output_path.with_suffix(".metrics.csv"), export_records)
        write_csv(output_path.with_suffix(".rois.csv"), roi_records)
        print("pandas is not installed; saved CSV files instead of Excel workbook.")
        return

    try:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame.from_records(export_records).to_excel(writer, sheet_name="metrics", index=False)
            pd.DataFrame.from_records(roi_records).to_excel(writer, sheet_name="roi_vertices", index=False)
        print(f"Saved metrics workbook: {output_path}")
    except Exception as error:
        print(f"Could not save Excel workbook ({error}); saving CSV files instead.")
        write_csv(output_path.with_suffix(".metrics.csv"), export_records)
        write_csv(output_path.with_suffix(".rois.csv"), roi_records)


def save_progress(output_dir, metric_records, roi_records):
    save_metrics_workbook(
        Path(output_dir) / "dmso_dynamic_time_course_metrics.xlsx",
        metric_records,
        roi_records,
    )


def write_csv(output_path, records):
    if not records:
        return
    import csv

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV: {output_path}")


def vertices_to_records(vertices, folder_name, roi_name):
    return [
        {
            "folder_name": folder_name,
            "roi": roi_name,
            "vertex_index": int(index),
            "x_pixel": float(vertex[0]),
            "depth_pixel": float(vertex[1]),
        }
        for index, vertex in enumerate(vertices)
    ]


def load_saved_roi_records(output_dir):
    workbook_path = Path(output_dir) / "dmso_dynamic_time_course_metrics.xlsx"
    csv_path = Path(output_dir) / "dmso_dynamic_time_course_metrics.rois.csv"

    if pd is not None and workbook_path.exists():
        try:
            dataframe = pd.read_excel(workbook_path, sheet_name="roi_vertices")
            return dataframe.to_dict("records")
        except Exception as error:
            print(f"Could not read saved ROI workbook ({error}); trying CSV fallback.")

    if csv_path.exists():
        import csv

        with open(csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            return list(reader)

    return []


def vertices_from_records(records, folder_name, roi_name, aliases=None):
    alias_set = {normalize_roi_key(folder_name)}
    if aliases is not None:
        for alias in aliases:
            if alias is None:
                continue
            alias_set.add(normalize_roi_key(alias))

    subset = [
        record for record in records
        if normalize_roi_key(record.get("folder_name", "")) in alias_set
        and str(record.get("roi", "")).strip().lower() == str(roi_name).strip().lower()
    ]
    if not subset:
        return None
    subset.sort(key=lambda record: int(record["vertex_index"]))
    vertices = np.asarray(
        [
            [float(record["x_pixel"]), float(record["depth_pixel"])]
            for record in subset
        ],
        dtype=np.float32,
    )
    if vertices.shape[0] < 3:
        return None
    return vertices


def process_time_course():
    plt.ioff()
    folders = iter_timepoint_folders(DEFAULT_ROOT_DIR)
    out_dir = output_directory(DEFAULT_ROOT_DIR)
    print(f"Found {len(folders)} time-point folder(s).")
    saved_roi_records = load_saved_roi_records(out_dir) if DEFAULT_REUSE_SAVED_ROIS else []
    if DEFAULT_REUSE_SAVED_ROIS:
        print(f"Loaded {len(saved_roi_records)} saved ROI vertex record(s) from {out_dir}")

    first_folder = folders[0][0]
    first_stack_paths = find_tiff_stacks(first_folder)
    if not first_stack_paths:
        raise ValueError(f"No TIFF stacks found in first time-point folder: {first_folder}")

    first_amp_for_background, _ = read_amp_phase_tiff_stack(first_stack_paths[0])
    first_mean_amp_for_background = np.mean(first_amp_for_background, axis=0, dtype=np.float32)
    del first_amp_for_background
    gc.collect()

    background_vertices = vertices_from_records(
        saved_roi_records,
        "shared",
        "background",
        aliases=["background", "shared_background"],
    )
    if background_vertices is None:
        background_mask, background_vertices = select_polygon_roi(
            first_mean_amp_for_background,
            f"{first_folder.name}: draw shared BACKGROUND ROI, then press Enter",
        )
        roi_records = vertices_to_records(background_vertices, "shared", "background")
    else:
        background_mask = polygon_mask_for_image_shape(background_vertices, first_mean_amp_for_background.shape)
        roi_records = vertices_to_records(background_vertices, "shared", "background")
        print("Loaded saved shared background ROI.")
    position_roi_map = {}

    for stack_path in first_stack_paths:
        position_key = position_key_from_path(stack_path)
        amplitude_stack, _ = read_amp_phase_tiff_stack(stack_path)
        mean_amp = np.mean(amplitude_stack, axis=0, dtype=np.float32)
        tissue_vertices = vertices_from_records(
            saved_roi_records,
            position_key,
            "tissue",
            aliases=[stack_path.name, stack_path.stem],
        )
        if tissue_vertices is None:
            tissue_mask, tissue_vertices = select_polygon_roi(
                mean_amp,
                f"{first_folder.name} / {position_key}: draw TISSUE ROI, then press Enter",
            )
            roi_records.extend(vertices_to_records(tissue_vertices, position_key, "tissue"))
            print(f"Saved new tissue ROI for {position_key}.")
        else:
            tissue_mask = polygon_mask_for_image_shape(tissue_vertices, mean_amp.shape)
            print(f"Loaded saved tissue ROI for {position_key}.")
        position_roi_map[position_key] = {
            "reference_mean_amp": mean_amp.copy(),
            "tissue_vertices": tissue_vertices,
        }
        if DEFAULT_SAVE_ROI_OVERLAYS:
            save_roi_overlay(
                mean_amp,
                tissue_vertices,
                f"{first_folder.name} / {position_key} mean B-line with tissue ROI",
                overlay_output_path(out_dir, first_folder.name, position_key),
            )
        del amplitude_stack
        del mean_amp
        gc.collect()

    metric_records = []
    image_records = []
    save_progress(out_dir, metric_records, roi_records)
    frame_count = max(2, int(round(DEFAULT_ANALYSIS_DURATION_SECONDS * DEFAULT_FRAME_RATE_HZ)))

    for folder, folder_index in folders:
        stack_paths = find_tiff_stacks(folder)
        relative_minutes = relative_minutes_from_dmso(folder.name, DEFAULT_DMSO_TIME_LABEL)
        print(
            f"Processing {folder.name} ({relative_minutes:.1f} min after DMSO): "
            f"{len(stack_paths)} stack(s)"
        )

        current_mean_amp_map = {}
        current_stack_map = {}
        for stack_path in stack_paths:
            position_key = position_key_from_path(stack_path)
            if position_key not in position_roi_map:
                print(f"  Skipping {stack_path.name}: no ROI defined for position {position_key}")
                continue
            amplitude_stack, phase_stack = read_amp_phase_tiff_stack(stack_path)
            mean_amp = np.mean(amplitude_stack, axis=0, dtype=np.float32)
            current_mean_amp_map[position_key] = mean_amp
            current_stack_map[position_key] = {
                "stack_path": stack_path,
                "amplitude_stack": amplitude_stack,
                "phase_stack": phase_stack,
            }

        apply_shift_estimation = (
            DEFAULT_LAST_TIMEPOINT_WITH_SHIFT is None
            or int(folder_index) <= int(DEFAULT_LAST_TIMEPOINT_WITH_SHIFT)
        )
        if apply_shift_estimation:
            shared_dz_pixels, used_positions = estimate_shared_vertical_shift(
                position_roi_map,
                current_mean_amp_map,
                search_pixels=DEFAULT_ROI_SHIFT_SEARCH_PIXELS,
                padding_pixels=DEFAULT_ROI_SHIFT_PADDING_PIXELS,
            )
            print(
                f"  Shared ROI dz relative to time point 1 = {shared_dz_pixels} pixel(s) "
                f"using {used_positions} position(s)"
            )
        else:
            shared_dz_pixels, used_positions = 0, 0
            print(
                f"  ROI shift disabled after Time-{DEFAULT_LAST_TIMEPOINT_WITH_SHIFT}; "
                "reusing time-point-1 ROIs with dz = 0"
            )

        for position_key, stack_info in current_stack_map.items():
            stack_path = stack_info["stack_path"]
            print(f"  {stack_path.name}")
            amplitude_stack = stack_info["amplitude_stack"]
            phase_stack = stack_info["phase_stack"]
            mean_amp = current_mean_amp_map[position_key]
            current_frame_count = min(frame_count, amplitude_stack.shape[0])
            amp_dyn, complex_dyn = compute_dynamic_images_for_duration(
                amplitude_stack=amplitude_stack,
                phase_stack=phase_stack,
                frame_count=current_frame_count,
                uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
                chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
            )

            roi_info = position_roi_map[position_key]
            shifted_tissue_vertices = shift_vertices_in_depth(roi_info["tissue_vertices"], shared_dz_pixels)
            current_tissue_mask = polygon_mask_for_image_shape(shifted_tissue_vertices, mean_amp.shape)
            current_background_mask = polygon_mask_for_image_shape(background_vertices, mean_amp.shape)
            if DEFAULT_SAVE_ROI_OVERLAYS:
                save_roi_overlay(
                    mean_amp,
                    shifted_tissue_vertices,
                    f"{folder.name} / {position_key} mean B-line with tissue ROI (dz={shared_dz_pixels})",
                    overlay_output_path(out_dir, folder.name, position_key),
                )

            for signal_type, image in [("amplitude", amp_dyn), ("complex", complex_dyn)]:
                metrics = compute_metrics(image, current_tissue_mask, current_background_mask)
                record = {
                    "folder_name": folder.name,
                    "stack_name": stack_path.name,
                    "position_key": position_key,
                    "relative_minutes": relative_minutes,
                    "duration_s": float(DEFAULT_ANALYSIS_DURATION_SECONDS),
                    "frame_count": int(current_frame_count),
                    "roi_shift_z_pixels": int(shared_dz_pixels),
                    "roi_shift_positions_used": int(used_positions),
                    "signal_type": signal_type,
                    **metrics,
                }
                metric_records.append(record)
                image_records.append({**record, "image": image.copy()})

            # Keep the first time point as the fixed registration reference so
            # a single bad estimate does not propagate forward through the run.

            del amplitude_stack
            del phase_stack
            del mean_amp
            del amp_dyn
            del complex_dyn
            gc.collect()

        save_progress(out_dir, metric_records, roi_records)

    for position_key in sorted(position_roi_map.keys()):
        plot_time_course_montage(
            image_records,
            "amplitude",
            position_key,
            out_dir / f"{position_key}_dmso_amplitude_dynamic_montage.png",
        )
        plot_time_course_montage(
            image_records,
            "complex",
            position_key,
            out_dir / f"{position_key}_dmso_complex_dynamic_montage.png",
        )
        plot_metrics_summary(
            metric_records,
            position_key,
            out_dir / f"{position_key}_dmso_dynamic_metrics.png",
        )

    save_progress(out_dir, metric_records, roi_records)
    print("Finished DMSO time-course analysis.")


def main():
    process_time_course()


if __name__ == "__main__":
    main()
