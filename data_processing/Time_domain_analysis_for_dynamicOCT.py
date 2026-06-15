import gc
import os
import re
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
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
DEFAULT_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines"
DEFAULT_NOISE_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"
DEFAULT_FRAME_RATE_HZ = 100.0
DEFAULT_TIMEPOINT_COUNTS = list([20,40,60,80,100,120,140,160,180,200,250,300,350,400,450,500,600,700,800,900,1000])
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 1
DEFAULT_DYNAMIC_CHUNK_X = 96
DEFAULT_OUTPUT_DIR = None  # None saves figures beside each TIFF stack.
DEFAULT_SAVE_FIGURES = True
DEFAULT_SAVE_EXCEL = True
DEFAULT_SAVE_DPI = 360
DEFAULT_SHOW_FIGURES = False
DEFAULT_METRICS_FIGSIZE = (9.5, 7.5)
DEFAULT_METRICS_FONT_SIZE = 10
DEFAULT_HIGH_FREQUENCY_FRACTION_CUTOFF = 0.25

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
    """
    Read complex OCT data saved by ThreadDnS.save_data().

    Each saved TIFF frame stores amplitude in the first half of the depth axis
    and phase in radians in the second half:
        saved[..., :Z] = abs(E)
        saved[..., Z:] = angle(E)
    """
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


def select_tissue_roi(mean_amplitude_image, stack_name):
    tissue_mask, tissue_vertices = select_polygon_roi(
        mean_amplitude_image,
        f"{stack_name}: draw TISSUE ROI, then press Enter",
    )
    return {
        "tissue_mask": tissue_mask,
        "tissue_vertices": tissue_vertices,
    }


def select_background_roi(mean_amplitude_image, stack_name):
    background_mask, background_vertices = select_polygon_roi(
        mean_amplitude_image,
        f"{stack_name}: draw BACKGROUND ROI on NOISE stack, then press Enter",
    )
    return {
        "background_mask": background_mask,
        "background_vertices": background_vertices,
    }


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


def correlation_to_reference(image, reference, mask):
    values = np.asarray(image[mask], dtype=np.float64)
    ref_values = np.asarray(reference[mask], dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(ref_values)
    if np.count_nonzero(valid) < 3:
        return np.nan
    values = values[valid]
    ref_values = ref_values[valid]
    if np.std(values) <= 0 or np.std(ref_values) <= 0:
        return np.nan
    return float(np.corrcoef(values, ref_values)[0, 1])


def relative_error_to_reference(image, reference, mask):
    values = np.asarray(image[mask], dtype=np.float64)
    ref_values = np.asarray(reference[mask], dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(ref_values)
    if np.count_nonzero(valid) == 0:
        return np.nan
    denominator = np.mean(np.abs(ref_values[valid]))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.mean(np.abs(values[valid] - ref_values[valid])) / denominator)


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


def compute_metrics_for_images(
    tissue_images,
    background_images,
    frame_counts,
    frame_rate_hz,
    tissue_mask,
    background_mask,
    signal_name,
    background_source_name,
):
    records = []

    for tissue_image, background_image, frame_count in zip(
        tissue_images,
        background_images,
        frame_counts,
    ):
        tissue_values = tissue_image[tissue_mask]
        background_values = background_image[background_mask]
        tissue_mean = safe_mean(tissue_values)
        tissue_std = safe_std(tissue_values)
        background_mean = safe_mean(background_values)
        background_std = safe_std(background_values)
        background_p95 = safe_percentile(background_values, 95)
        cnr = np.nan
        if np.isfinite(background_std) and background_std > 0:
            cnr = (tissue_mean - background_mean) / background_std

        records.append(
            {
                "signal_type": signal_name,
                "frame_count": int(frame_count),
                "duration_s": float(frame_count) / float(frame_rate_hz),
                "background_source": background_source_name,
                "tissue_mean": tissue_mean,
                "tissue_std": tissue_std,
                "background_mean": background_mean,
                "background_std": background_std,
                "background_p95": background_p95,
                "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
                "tissue_cv": tissue_std / tissue_mean if np.isfinite(tissue_mean) and tissue_mean != 0 else np.nan,
                "high_frequency_fraction": high_frequency_fraction(
                    tissue_image,
                    tissue_mask,
                    DEFAULT_HIGH_FREQUENCY_FRACTION_CUTOFF,
                ),
            }
        )

    return records


def compute_dynamic_images_for_prefix(
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


def plot_dynamic_montage(images, frame_counts, title, output_path=None, show=False):
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
        fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def save_roi_overlay(image, vertices, title, output_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.imshow(image.T, aspect="auto", origin="lower", cmap="gray")
    polygon = np.asarray(vertices, dtype=np.float32)
    closed_polygon = np.vstack([polygon, polygon[0]])
    ax.plot(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        color="cyan",
        linewidth=1.8,
    )
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth")
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved ROI overlay: {output_path}")
    plt.close(fig)


def write_metrics_excel(output_path, amplitude_records, complex_records, roi_info, metadata=None):
    if metadata is None:
        metadata = []
    if pd is None:
        csv_base = Path(output_path).with_suffix("")
        write_metrics_csv(csv_base.with_name(csv_base.name + "_amplitude_metrics.csv"), amplitude_records)
        write_metrics_csv(csv_base.with_name(csv_base.name + "_complex_metrics.csv"), complex_records)
        write_metrics_csv(
            csv_base.with_name(csv_base.name + "_roi_vertices.csv"),
            roi_vertices_records(roi_info),
        )
        write_metrics_csv(
            csv_base.with_name(csv_base.name + "_metadata.csv"),
            metadata,
        )
        print("pandas is not installed; saved CSV files instead of Excel workbook.")
        return

    try:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame.from_records(amplitude_records).to_excel(
                writer,
                sheet_name="amplitude_metrics",
                index=False,
            )
            pd.DataFrame.from_records(complex_records).to_excel(
                writer,
                sheet_name="complex_metrics",
                index=False,
            )
            pd.DataFrame.from_records(roi_vertices_records(roi_info)).to_excel(
                writer,
                sheet_name="roi_vertices",
                index=False,
            )
            pd.DataFrame.from_records(metadata).to_excel(
                writer,
                sheet_name="metadata",
                index=False,
            )
        print(f"Saved metrics workbook: {output_path}")
    except Exception as error:
        csv_base = Path(output_path).with_suffix("")
        print(f"Could not save Excel workbook ({error}); saving CSV files instead.")
        write_metrics_csv(csv_base.with_name(csv_base.name + "_amplitude_metrics.csv"), amplitude_records)
        write_metrics_csv(csv_base.with_name(csv_base.name + "_complex_metrics.csv"), complex_records)
        write_metrics_csv(
            csv_base.with_name(csv_base.name + "_roi_vertices.csv"),
            roi_vertices_records(roi_info),
        )
        write_metrics_csv(
            csv_base.with_name(csv_base.name + "_metadata.csv"),
            metadata,
        )


def write_metrics_csv(output_path, records):
    if not records:
        return
    import csv

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved metrics CSV: {output_path}")


def normalize_roi_key(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def convert_frequency_domain_roi_records(records):
    converted = []
    for record in records:
        converted.append(
            {
                "roi": record.get("roi", ""),
                "source_stack": record.get("label", ""),
                "vertex_index": record.get("vertex_index", 0),
                "x_pixel": record.get("x_pixel", np.nan),
                "depth_pixel": record.get("depth_pixel", np.nan),
            }
        )
    return converted


def load_saved_roi_records(output_dir, stack_stem):
    output_dir = Path(output_dir)
    workbook_candidates = [
        output_dir / f"{stack_stem}_dynamic_timepoint_metrics.xlsx",
        output_dir.parent / "frequency_band_power_analysis" / f"{stack_stem}_frequency_domain_metrics.xlsx",
    ]
    csv_candidates = [
        output_dir / f"{stack_stem}_dynamic_timepoint_metrics_roi_vertices.csv",
        output_dir.parent / "frequency_band_power_analysis" / f"{stack_stem}_frequency_domain_metrics.rois.csv",
    ]

    if pd is not None:
        for workbook_path in workbook_candidates:
            if workbook_path.exists():
                try:
                    dataframe = pd.read_excel(workbook_path, sheet_name="roi_vertices")
                    records = dataframe.to_dict("records")
                    if records and "source_stack" not in records[0] and "label" in records[0]:
                        return convert_frequency_domain_roi_records(records)
                    return records
                except Exception as error:
                    print(f"Could not read saved ROI workbook {workbook_path} ({error}); trying next fallback.")

    for csv_path in csv_candidates:
        if csv_path.exists():
            import csv

            with open(csv_path, "r", newline="") as file:
                reader = csv.DictReader(file)
                records = list(reader)
                if records and "source_stack" not in records[0] and "label" in records[0]:
                    return convert_frequency_domain_roi_records(records)
                return records

    return []


def vertices_from_records(records, roi_name, source_name):
    target_roi = str(roi_name).strip().lower()
    target_source = normalize_roi_key(source_name)
    subset = [
        record
        for record in records
        if str(record.get("roi", "")).strip().lower() == target_roi
        and normalize_roi_key(record.get("source_stack", "")) == target_source
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


def save_roi_only_workbook(output_path, roi_info):
    write_metrics_excel(output_path, [], [], roi_info, metadata=[])


def roi_vertices_records(roi_info):
    records = []
    roi_specs = [
        ("tissue", "tissue_vertices", roi_info.get("tissue_source", "")),
        ("background", "background_vertices", roi_info.get("background_source", "")),
    ]
    for roi_name, key, source_name in roi_specs:
        for vertex_index, (x_value, z_value) in enumerate(roi_info[key]):
            records.append(
                {
                    "roi": roi_name,
                    "source_stack": source_name,
                    "vertex_index": int(vertex_index),
                    "x_pixel": float(x_value),
                    "depth_pixel": float(z_value),
                }
            )
    return records


def plot_metric_pair(axis, amplitude_records, complex_records, metric, ylabel, title):
    amp_x = [record["duration_s"] for record in amplitude_records]
    complex_x = [record["duration_s"] for record in complex_records]
    amp_y = [record[metric] for record in amplitude_records]
    complex_y = [record[metric] for record in complex_records]
    axis.plot(amp_x, amp_y, "o-", linewidth=1.8, markersize=4.5, label="Amplitude")
    axis.plot(complex_x, complex_y, "s-", linewidth=1.8, markersize=4.5, label="Complex")
    axis.set_xlabel("Image duration (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.28)


def plot_metrics_summary(amplitude_records, complex_records, stack_name, output_path):
    plt.rcParams.update({"font.size": DEFAULT_METRICS_FONT_SIZE})
    fig, axes = plt.subplots(2, 2, figsize=DEFAULT_METRICS_FIGSIZE, constrained_layout=True)
    axes = axes.reshape(-1)

    plot_metric_pair(axes[0], amplitude_records, complex_records, "cnr", "CNR", "Tissue-background CNR")
    plot_metric_pair(
        axes[1],
        amplitude_records,
        complex_records,
        "background_std",
        "Background std",
        "Background noise floor",
    )
    plot_metric_pair(axes[2], amplitude_records, complex_records, "tissue_cv", "CV", "Tissue spatial CV")
    plot_metric_pair(
        axes[3],
        amplitude_records,
        complex_records,
        "high_frequency_fraction",
        "Fraction",
        "Tissue high-frequency spatial power",
    )

    axes[1].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    axes[3].set_ylim(0.0, 1.0)

    for axis in axes:
        axis.set_xlim(left=0.0)
        axis.legend(frameon=False, fontsize=DEFAULT_METRICS_FONT_SIZE - 1)
        axis.tick_params(labelsize=DEFAULT_METRICS_FONT_SIZE - 1)

    fig.suptitle(f"{stack_name}: dynamic signal duration metrics", fontsize=DEFAULT_METRICS_FONT_SIZE + 3)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved metrics figure: {output_path}")
    plt.close(fig)


def output_directory_for_stack(stack_path, configured_output_dir):
    if configured_output_dir is not None:
        output_dir = Path(configured_output_dir)
    else:
        output_dir = stack_path.parent / "dynamic_timepoint_sufficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_noise_stack_and_roi(noise_input_path):
    if noise_input_path is None or str(noise_input_path).strip() == "":
        raise ValueError("DEFAULT_NOISE_INPUT_PATH must point to a separate noise TIFF stack.")
    noise_path = Path(noise_input_path)
    print(f"Loading noise AMP+PHASE stack: {noise_path}")
    noise_amplitude_stack, noise_phase_stack = read_amp_phase_tiff_stack(noise_path)
    noise_mean_amplitude_image = np.mean(noise_amplitude_stack, axis=0, dtype=np.float32)
    noise_output_dir = output_directory_for_stack(noise_path, DEFAULT_OUTPUT_DIR)
    saved_noise_roi_records = load_saved_roi_records(noise_output_dir, noise_path.stem)
    saved_background_vertices = vertices_from_records(
        saved_noise_roi_records,
        "background",
        noise_path.name,
    )
    if saved_background_vertices is not None:
        noise_roi_info = {
            "background_mask": polygon_mask_for_image_shape(
                saved_background_vertices,
                noise_mean_amplitude_image.shape,
            ),
            "background_vertices": saved_background_vertices,
        }
        print(f"Loaded saved background ROI for noise stack {noise_path.name}.")
    else:
        noise_roi_info = select_background_roi(noise_mean_amplitude_image, noise_path.name)
        noise_roi_export = {
            "tissue_vertices": np.empty((0, 2), dtype=np.float32),
            "background_vertices": noise_roi_info["background_vertices"],
            "tissue_source": "",
            "background_source": noise_path.name,
        }
        save_roi_only_workbook(
            noise_output_dir / f"{noise_path.stem}_dynamic_timepoint_metrics.xlsx",
            noise_roi_export,
        )

    noise_overlay_output = noise_output_dir / f"{noise_path.stem}_background_roi_overlay.png"
    if DEFAULT_SAVE_FIGURES:
        save_roi_overlay(
            noise_mean_amplitude_image,
            noise_roi_info["background_vertices"],
            f"{noise_path.name} mean B-line with background ROI",
            noise_overlay_output,
        )
    return {
        "path": noise_path,
        "amplitude_stack": noise_amplitude_stack,
        "phase_stack": noise_phase_stack,
        "mean_amplitude_image": noise_mean_amplitude_image,
        "roi_info": noise_roi_info,
    }


def process_stack(stack_path, noise_context):
    print(f"Loading AMP+PHASE stack: {stack_path}")
    amplitude_stack, phase_stack = read_amp_phase_tiff_stack(stack_path)
    mean_amplitude_image = np.mean(amplitude_stack, axis=0, dtype=np.float32)
    output_dir = output_directory_for_stack(stack_path, DEFAULT_OUTPUT_DIR)
    saved_roi_records = load_saved_roi_records(output_dir, stack_path.stem)
    saved_tissue_vertices = vertices_from_records(
        saved_roi_records,
        "tissue",
        stack_path.name,
    )
    if saved_tissue_vertices is not None:
        tissue_roi_info = {
            "tissue_mask": polygon_mask_for_image_shape(saved_tissue_vertices, mean_amplitude_image.shape),
            "tissue_vertices": saved_tissue_vertices,
        }
        print(f"Loaded saved tissue ROI for {stack_path.name}.")
    else:
        tissue_roi_info = select_tissue_roi(mean_amplitude_image, stack_path.name)
        immediate_roi_export = {
            "tissue_vertices": tissue_roi_info["tissue_vertices"],
            "background_vertices": noise_context["roi_info"]["background_vertices"],
            "tissue_source": stack_path.name,
            "background_source": noise_context["path"].name,
        }
        save_roi_only_workbook(
            output_dir / f"{stack_path.stem}_dynamic_timepoint_metrics.xlsx",
            immediate_roi_export,
        )

    noise_frame_total = noise_context["amplitude_stack"].shape[0]
    frame_total = amplitude_stack.shape[0]
    max_shared_frames = min(frame_total, noise_frame_total)
    frame_counts = [count for count in DEFAULT_TIMEPOINT_COUNTS if count <= max_shared_frames]
    if not frame_counts:
        frame_counts = [max_shared_frames]

    print(
        f"Stack shape: frames={frame_total}, X={amplitude_stack.shape[1]}, "
        f"Z={amplitude_stack.shape[2]}; noise frames={noise_frame_total}"
    )
    amplitude_images = []
    complex_images = []
    noise_amplitude_images = []
    noise_complex_images = []

    for frame_count in frame_counts:
        print(f"Computing dynamic images from first {frame_count} frame(s)...")
        amplitude_dynamic, complex_dynamic = compute_dynamic_images_for_prefix(
            amplitude_stack=amplitude_stack,
            phase_stack=phase_stack,
            frame_count=frame_count,
            uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )
        noise_amplitude_dynamic, noise_complex_dynamic = compute_dynamic_images_for_prefix(
            amplitude_stack=noise_context["amplitude_stack"],
            phase_stack=noise_context["phase_stack"],
            frame_count=frame_count,
            uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )
        amplitude_images.append(amplitude_dynamic)
        complex_images.append(complex_dynamic)
        noise_amplitude_images.append(noise_amplitude_dynamic)
        noise_complex_images.append(noise_complex_dynamic)
        gc.collect()

    base_name = stack_path.stem
    amp_output = output_dir / f"{base_name}_amplitude_dynamic_timepoints.png"
    complex_output = output_dir / f"{base_name}_complex_dynamic_timepoints.png"
    metrics_output = output_dir / f"{base_name}_dynamic_timepoint_metrics.png"
    excel_output = output_dir / f"{base_name}_dynamic_timepoint_metrics.xlsx"
    tissue_overlay_output = output_dir / f"{base_name}_tissue_roi_overlay.png"
    noise_overlay_output = output_dir / f"{base_name}_noise_background_roi_overlay.png"

    if DEFAULT_SAVE_FIGURES:
        save_roi_overlay(
            mean_amplitude_image,
            tissue_roi_info["tissue_vertices"],
            f"{stack_path.name} mean B-line with tissue ROI",
            tissue_overlay_output,
        )
        save_roi_overlay(
            noise_context["mean_amplitude_image"],
            noise_context["roi_info"]["background_vertices"],
            f"{noise_context['path'].name} mean B-line with background ROI",
            noise_overlay_output,
        )

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

    amplitude_metrics = compute_metrics_for_images(
        tissue_images=amplitude_images,
        background_images=noise_amplitude_images,
        frame_counts=frame_counts,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        tissue_mask=tissue_roi_info["tissue_mask"],
        background_mask=noise_context["roi_info"]["background_mask"],
        signal_name="amplitude",
        background_source_name=noise_context["path"].name,
    )
    complex_metrics = compute_metrics_for_images(
        tissue_images=complex_images,
        background_images=noise_complex_images,
        frame_counts=frame_counts,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        tissue_mask=tissue_roi_info["tissue_mask"],
        background_mask=noise_context["roi_info"]["background_mask"],
        signal_name="complex",
        background_source_name=noise_context["path"].name,
    )

    roi_info = {
        "tissue_vertices": tissue_roi_info["tissue_vertices"],
        "background_vertices": noise_context["roi_info"]["background_vertices"],
        "tissue_source": stack_path.name,
        "background_source": noise_context["path"].name,
    }
    metadata = [
        {"key": "tissue_stack", "value": str(stack_path)},
        {"key": "noise_stack", "value": str(noise_context["path"])},
        {"key": "frame_rate_hz", "value": DEFAULT_FRAME_RATE_HZ},
        {"key": "timepoint_counts", "value": ",".join(str(count) for count in frame_counts)},
        {"key": "dynamic_uniform_filter_size", "value": DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE},
        {"key": "dynamic_chunk_x", "value": DEFAULT_DYNAMIC_CHUNK_X},
    ]

    if DEFAULT_SAVE_FIGURES:
        plot_metrics_summary(
            amplitude_metrics,
            complex_metrics,
            stack_name=stack_path.name,
            output_path=metrics_output,
        )
    if DEFAULT_SAVE_EXCEL:
        write_metrics_excel(excel_output, amplitude_metrics, complex_metrics, roi_info, metadata=metadata)
    else:
        save_roi_only_workbook(excel_output, roi_info)

    del amplitude_stack
    del phase_stack
    del mean_amplitude_image
    del amplitude_images
    del complex_images
    del noise_amplitude_images
    del noise_complex_images
    del amplitude_metrics
    del complex_metrics
    gc.collect()


def main():
    plt.ioff()
    stack_paths = iter_tiff_stacks(DEFAULT_INPUT_PATH)
    noise_context = load_noise_stack_and_roi(DEFAULT_NOISE_INPUT_PATH)
    print(f"Found {len(stack_paths)} TIFF stack(s).")
    failed_paths = []
    for stack_path in stack_paths:
        try:
            process_stack(stack_path, noise_context)
        except Exception as error:
            failed_paths.append((stack_path, error))
            print(f"Failed to process {stack_path}: {error}")

    if failed_paths:
        print(f"Finished with {len(failed_paths)} failed stack(s):")
        for stack_path, error in failed_paths:
            print(f"  {stack_path}: {error}")
    else:
        print("Finished processing all TIFF stacks.")

    del noise_context
    gc.collect()


if __name__ == "__main__":
    main()
