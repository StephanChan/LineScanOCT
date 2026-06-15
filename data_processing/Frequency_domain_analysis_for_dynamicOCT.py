import gc
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
import numpy as np
import tifffile as TIFF

try:
    import pandas as pd
except ImportError:
    pd = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_TISSUE_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\200Hz 2seconds Blines"
DEFAULT_BACKGROUND_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_TISSUE_DURATION_SECONDS = 2.0
DEFAULT_BACKGROUND_DURATION_SECONDS = 2.0
# Use None when matching the full-band time-domain pipeline. The time-domain
# script subtracts only the temporal mean when filter size = 1; it does not
# remove an additional 0-0.5 Hz band.
DEFAULT_NOTCH_BAND_HZ = None
DEFAULT_BAND_WIDTH_HZ = 2
DEFAULT_AMPLITUDE_FREQ_RANGE_HZ = (0.0, DEFAULT_FRAME_RATE_HZ/2)
DEFAULT_COMPLEX_FREQ_RANGE_HZ = (0.0, DEFAULT_FRAME_RATE_HZ/2)
DEFAULT_CHUNK_X = 200
DEFAULT_OUTPUT_DIR = None  # None saves into tissue_dir / "frequency_band_power_analysis".
DEFAULT_SAVE_FIGURES = True
DEFAULT_SHOW_FIGURES = False
DEFAULT_FIGURE_DPI = 400
DEFAULT_MONTAGE_COLUMNS = 5
DEFAULT_REUSE_SAVED_ROIS = True
DEFAULT_FONT_SIZE = 11
DEFAULT_LOG_SPECTRUM_EPS = 1e-0
DEFAULT_CUMULATIVE_BAND_STEP_HZ = DEFAULT_BAND_WIDTH_HZ
DEFAULT_ENABLE_DOWNSAMPLE_COMPARISON = True
DEFAULT_DOWNSAMPLE_TARGET_FRAME_RATES_HZ = [50.0]
DEFAULT_DOWNSAMPLE_METHODS = ("original", "skip", "integrate")

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
        if (
            item.is_file()
            and item.suffix.lower() in {".tif", ".tiff"}
            and "bline" in item.name.lower()
        )
    ]
    paths.sort(key=natural_bline_sort_key)
    if not paths:
        raise ValueError(f"No Bline TIFF stacks found in: {input_path}")
    return paths


def try_iter_tiff_stacks(input_path):
    if input_path is None:
        return []
    text = str(input_path).strip()
    if text == "":
        return []
    try:
        return iter_tiff_stacks(text)
    except Exception as error:
        print(f"Background stack input unavailable ({error}). Falling back to tissue-stack background ROI.")
        return []

def natural_bline_sort_key(path):
    match = BLINE_NAME_RE.match(path.name)
    if match is None:
        return (1, path.name.lower())
    return (0, int(match.group("index")), int(match.group("yrpt")))


def output_directory(base_input_path, configured_output_dir):
    if configured_output_dir is not None:
        out_dir = Path(configured_output_dir)
    else:
        base_path = Path(base_input_path)
        out_dir = base_path / "frequency_band_power_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def limit_stack_duration(amplitude_stack, phase_stack, frame_rate_hz, duration_seconds):
    if duration_seconds is None:
        return amplitude_stack, phase_stack
    max_frames = int(round(float(frame_rate_hz) * float(duration_seconds)))
    if max_frames <= 0 or amplitude_stack.shape[0] <= max_frames:
        return amplitude_stack, phase_stack
    return amplitude_stack[:max_frames], phase_stack[:max_frames]


def make_bands(start_hz, stop_hz, band_width_hz):
    edges = np.arange(
        float(start_hz),
        float(stop_hz) + float(band_width_hz) * 0.5,
        float(band_width_hz),
        dtype=np.float32,
    )
    return [(float(edges[idx]), float(edges[idx + 1])) for idx in range(edges.size - 1)]


def fft_bandstop_stack_axis0(stack, frame_rate_hz, stop_band_hz=None):
    stack = np.asarray(stack)
    if stack.shape[0] < 2 or stop_band_hz is None:
        return stack.copy()

    low_hz, high_hz = sorted([float(stop_band_hz[0]), float(stop_band_hz[1])])
    if high_hz <= low_hz:
        return stack.copy()

    frequencies = np.fft.fftfreq(stack.shape[0], d=1.0 / float(frame_rate_hz))
    spectrum = np.fft.fft(stack, axis=0)
    stop_mask = (np.abs(frequencies) >= low_hz) & (np.abs(frequencies) <= high_hz)
    spectrum[stop_mask, :, :] = 0
    filtered = np.fft.ifft(spectrum, axis=0)
    if np.isrealobj(stack):
        return filtered.real.astype(np.float32, copy=False)
    return filtered.astype(np.complex64, copy=False)


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
    ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], color="cyan", linewidth=1.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth")
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved ROI overlay: {output_path}")
    plt.close(fig)


def vertices_to_records(vertices, label, roi_name):
    return [
        {
            "label": label,
            "roi": roi_name,
            "vertex_index": int(index),
            "x_pixel": float(vertex[0]),
            "depth_pixel": float(vertex[1]),
        }
        for index, vertex in enumerate(vertices)
    ]


def replace_roi_records(roi_records, label, roi_name, vertices):
    label_key = normalize_roi_key(label)
    roi_key = str(roi_name).strip().lower()
    filtered = [
        record
        for record in roi_records
        if not (
            normalize_roi_key(record.get("label", "")) == label_key
            and str(record.get("roi", "")).strip().lower() == roi_key
        )
    ]
    filtered.extend(vertices_to_records(vertices, label, roi_name))
    return filtered


def frequency_metrics_workbook_path(output_dir, stack_stem):
    return Path(output_dir) / f"{stack_stem}_frequency_domain_metrics.xlsx"


def time_metrics_workbook_path(stack_path):
    return Path(stack_path).parent / "dynamic_timepoint_sufficiency" / f"{Path(stack_path).stem}_dynamic_timepoint_metrics.xlsx"


def frequency_metrics_csv_roi_path(output_dir, stack_stem):
    return Path(output_dir) / f"{stack_stem}_frequency_domain_metrics.rois.csv"


def time_metrics_csv_roi_path(stack_path):
    return (
        Path(stack_path).parent
        / "dynamic_timepoint_sufficiency"
        / f"{Path(stack_path).stem}_dynamic_timepoint_metrics_roi_vertices.csv"
    )


def convert_time_domain_roi_records(records):
    converted = []
    for record in records:
        converted.append(
            {
                "label": record.get("source_stack", ""),
                "roi": record.get("roi", ""),
                "vertex_index": record.get("vertex_index", 0),
                "x_pixel": record.get("x_pixel", np.nan),
                "depth_pixel": record.get("depth_pixel", np.nan),
            }
        )
    return converted


def roi_records_for_stack(roi_records, stack_path, include_shared_background=True):
    stack_path = Path(stack_path)
    stack_keys = {
        normalize_roi_key(stack_path.stem),
        normalize_roi_key(stack_path.name),
    }
    shared_keys = set()
    if include_shared_background:
        shared_keys.update(
            {
                normalize_roi_key("shared_background"),
                normalize_roi_key("background"),
            }
        )

    filtered_records = []
    for record in roi_records:
        label_key = normalize_roi_key(record.get("label", ""))
        if label_key in stack_keys or label_key in shared_keys:
            filtered_records.append(record)
    return filtered_records


def load_saved_roi_records(output_dir, stack_path):
    stack_path = Path(stack_path)
    stack_stem = stack_path.stem
    workbook_candidates = [
        frequency_metrics_workbook_path(output_dir, stack_stem),
        time_metrics_workbook_path(stack_path),
        Path(output_dir) / "frequency_band_power_metrics.xlsx",
    ]
    csv_candidates = [
        frequency_metrics_csv_roi_path(output_dir, stack_stem),
        time_metrics_csv_roi_path(stack_path),
        Path(output_dir) / "frequency_band_power_metrics.rois.csv",
    ]

    if pd is not None:
        for workbook_path in workbook_candidates:
            if workbook_path.exists():
                try:
                    dataframe = pd.read_excel(workbook_path, sheet_name="roi_vertices")
                    print(f"Loaded ROI records from {workbook_path}")
                    records = dataframe.to_dict("records")
                    if records and "label" not in records[0] and "source_stack" in records[0]:
                        return convert_time_domain_roi_records(records)
                    return records
                except Exception as error:
                    print(f"Could not read saved ROI workbook {workbook_path} ({error}); trying next fallback.")

    for csv_path in csv_candidates:
        if csv_path.exists():
            with open(csv_path, "r", newline="") as file:
                print(f"Loaded ROI records from {csv_path}")
                records = list(csv.DictReader(file))
                if records and "label" not in records[0] and "source_stack" in records[0]:
                    return convert_time_domain_roi_records(records)
                return records

    return []


def vertices_from_records(records, label, roi_name, aliases=None):
    alias_set = {normalize_roi_key(label)}
    if aliases is not None:
        for alias in aliases:
            if alias is not None:
                alias_set.add(normalize_roi_key(alias))

    subset = [
        record
        for record in records
        if normalize_roi_key(record.get("label", "")) in alias_set
        and str(record.get("roi", "")).strip().lower() == str(roi_name).strip().lower()
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


def normalized_power_spectrum_axis0(stack):
    stack = np.asarray(stack)
    centered = stack - np.mean(stack, axis=0, keepdims=True)
    spectrum = np.fft.fft(centered, axis=0) / stack.shape[0]
    power = np.abs(spectrum) ** 2
    return power.astype(np.float32, copy=False)


def reconstruct_complex_stack(amplitude_stack, phase_stack):
    amplitude_stack = np.asarray(amplitude_stack, dtype=np.float32)
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    return (amplitude_stack * np.exp(1j * phase_stack)).astype(np.complex64, copy=False)


def split_complex_stack_to_amp_phase(complex_stack):
    complex_stack = np.asarray(complex_stack, dtype=np.complex64)
    amplitude_stack = np.abs(complex_stack).astype(np.float32, copy=False)
    phase_stack = np.angle(complex_stack).astype(np.float32, copy=False)
    return amplitude_stack, phase_stack


def downsample_amp_phase_skip(amplitude_stack, phase_stack, factor):
    factor = max(1, int(factor))
    if factor <= 1:
        return (
            np.ascontiguousarray(amplitude_stack, dtype=np.float32),
            np.ascontiguousarray(phase_stack, dtype=np.float32),
        )
    return (
        np.ascontiguousarray(amplitude_stack[::factor], dtype=np.float32),
        np.ascontiguousarray(phase_stack[::factor], dtype=np.float32),
    )


def downsample_amp_phase_integrate(amplitude_stack, phase_stack, factor):
    factor = max(1, int(factor))
    if factor <= 1:
        return (
            np.ascontiguousarray(amplitude_stack, dtype=np.float32),
            np.ascontiguousarray(phase_stack, dtype=np.float32),
        )

    complex_stack = reconstruct_complex_stack(amplitude_stack, phase_stack)
    usable_frames = (complex_stack.shape[0] // factor) * factor
    if usable_frames < factor:
        raise ValueError(
            f"Not enough frames ({complex_stack.shape[0]}) for integrate downsampling by factor {factor}."
        )
    complex_stack = np.ascontiguousarray(complex_stack[:usable_frames], dtype=np.complex64)
    downsampled = np.mean(
        complex_stack.reshape(usable_frames // factor, factor, *complex_stack.shape[1:]),
        axis=1,
        dtype=np.complex64,
    )
    return split_complex_stack_to_amp_phase(downsampled)


def variance_equivalent_real_power_spectrum_axis0(stack):
    """
    Return a one-sided, variance-equivalent power spectrum for a real signal.

    With the DC component already removed by temporal mean subtraction, the
    total sum of this one-sided spectrum matches the time-domain variance.
    """
    power = normalized_power_spectrum_axis0(stack)
    corrected = np.zeros_like(power)
    frame_count = power.shape[0]
    corrected[0, :, :] = power[0, :, :]
    if frame_count % 2 == 0:
        corrected[1 : frame_count // 2, :, :] = 2.0 * power[1 : frame_count // 2, :, :]
        corrected[frame_count // 2, :, :] = power[frame_count // 2, :, :]
    else:
        corrected[1 : (frame_count + 1) // 2, :, :] = 2.0 * power[1 : (frame_count + 1) // 2, :, :]
    return corrected.astype(np.float32, copy=False)


def band_power_from_power_spectrum(power, frequencies, bands, symmetric=False):
    _, x_count, z_count = power.shape
    band_images = np.empty((len(bands), x_count, z_count), dtype=np.float32)
    for band_idx, (low_hz, high_hz) in enumerate(bands):
        if symmetric:
            abs_frequencies = np.abs(frequencies)
            if band_idx == len(bands) - 1:
                mask = (abs_frequencies >= low_hz) & (abs_frequencies <= high_hz)
            else:
                mask = (abs_frequencies >= low_hz) & (abs_frequencies < high_hz)
        else:
            if band_idx == len(bands) - 1:
                mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            else:
                mask = (frequencies >= low_hz) & (frequencies < high_hz)
        if not np.any(mask):
            band_images[band_idx, :, :] = 0.0
            continue
        band_images[band_idx, :, :] = np.sum(power[mask, :, :], axis=0, dtype=np.float32)
    return band_images


def roi_spectrum_sum_and_count(power, mask):
    if not np.any(mask):
        return np.zeros(power.shape[0], dtype=np.float64), 0
    values = power[:, mask]
    if values.size == 0:
        return np.zeros(power.shape[0], dtype=np.float64), 0
    return np.sum(values, axis=1, dtype=np.float64), int(values.shape[1])


def compute_band_power_images_and_spectra(
    amplitude_stack,
    phase_stack,
    frame_rate_hz,
    notch_band_hz,
    amplitude_bands,
    complex_bands,
    cumulative_upper_limits=None,
    tissue_mask=None,
    chunk_x=96,
):
    frames, x_pixels, z_pixels = amplitude_stack.shape
    amplitude_band_images = np.zeros((len(amplitude_bands), x_pixels, z_pixels), dtype=np.float32)
    complex_band_images = np.zeros((len(complex_bands), x_pixels, z_pixels), dtype=np.float32)
    if cumulative_upper_limits is None:
        amplitude_cumulative_images = None
        complex_cumulative_images = None
    else:
        amplitude_cumulative_images = np.zeros((len(cumulative_upper_limits), x_pixels, z_pixels), dtype=np.float32)
        complex_cumulative_images = np.zeros((len(cumulative_upper_limits), x_pixels, z_pixels), dtype=np.float32)
    full_frequencies = np.fft.fftfreq(frames, d=1.0 / float(frame_rate_hz)).astype(np.float32)
    amplitude_roi_spectrum_sum = np.zeros(frames, dtype=np.float64) if tissue_mask is not None else None
    complex_roi_spectrum_sum = np.zeros(frames, dtype=np.float64) if tissue_mask is not None else None
    amplitude_roi_pixel_count = 0
    complex_roi_pixel_count = 0
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
        complex_stack = (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)
        complex_stack = fft_bandstop_stack_axis0(
            complex_stack,
            frame_rate_hz=frame_rate_hz,
            stop_band_hz=notch_band_hz,
        )

        amplitude_power = variance_equivalent_real_power_spectrum_axis0(amplitude)
        complex_power = normalized_power_spectrum_axis0(complex_stack)
        amplitude_band_images[:, x0:x1, :] = band_power_from_power_spectrum(
            amplitude_power,
            full_frequencies,
            amplitude_bands,
            symmetric=False,
        )
        complex_band_images[:, x0:x1, :] = band_power_from_power_spectrum(
            complex_power,
            full_frequencies,
            complex_bands,
            symmetric=True,
        )
        if cumulative_upper_limits is not None:
            amplitude_cumulative_images[:, x0:x1, :] = cumulative_images_from_power_spectrum(
                amplitude_power,
                full_frequencies,
                cumulative_upper_limits,
                signal_type="amplitude",
            )
            complex_cumulative_images[:, x0:x1, :] = cumulative_images_from_power_spectrum(
                complex_power,
                full_frequencies,
                cumulative_upper_limits,
                signal_type="complex",
            )

        if tissue_mask is not None:
            chunk_mask = tissue_mask[x0:x1, :]
            amp_sum, amp_count = roi_spectrum_sum_and_count(amplitude_power, chunk_mask)
            complex_sum, complex_count = roi_spectrum_sum_and_count(complex_power, chunk_mask)
            amplitude_roi_spectrum_sum += amp_sum
            complex_roi_spectrum_sum += complex_sum
            amplitude_roi_pixel_count += amp_count
            complex_roi_pixel_count += complex_count

        print(f"Frequency analysis: processed X {x0}-{x1} / {x_pixels}")

    if tissue_mask is not None:
        if amplitude_roi_pixel_count > 0:
            amplitude_roi_spectrum = (
                amplitude_roi_spectrum_sum / float(amplitude_roi_pixel_count)
            ).astype(np.float32)
        else:
            amplitude_roi_spectrum = np.full(frames, np.nan, dtype=np.float32)
        if complex_roi_pixel_count > 0:
            complex_roi_spectrum = (
                complex_roi_spectrum_sum / float(complex_roi_pixel_count)
            ).astype(np.float32)
        else:
            complex_roi_spectrum = np.full(frames, np.nan, dtype=np.float32)
    else:
        amplitude_roi_spectrum = None
        complex_roi_spectrum = None

    return (
        full_frequencies,
        amplitude_band_images,
        complex_band_images,
        amplitude_roi_spectrum,
        complex_roi_spectrum,
        amplitude_cumulative_images,
        complex_cumulative_images,
    )


def cumulative_images_from_power_spectrum(power, frequencies, upper_limits, signal_type):
    _, x_count, z_count = power.shape
    cumulative = np.zeros((len(upper_limits), x_count, z_count), dtype=np.float32)
    if signal_type == "amplitude":
        working_frequencies = frequencies
    else:
        working_frequencies = np.abs(frequencies)

    for idx, upper_hz in enumerate(upper_limits):
        if signal_type == "amplitude":
            mask = working_frequencies <= float(upper_hz) + 1e-6
        else:
            mask = working_frequencies <= float(upper_hz) + 1e-6
        if not np.any(mask):
            continue
        cumulative[idx, :, :] = np.sum(power[mask, :, :], axis=0, dtype=np.float32)
    return cumulative


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


def plot_band_montage(band_images, bands, title, output_path, columns=5, cmap="magma", show=False):
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
    fig.suptitle(title, fontsize=DEFAULT_FONT_SIZE + 2)
    if output_path is not None:
        fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_roi_spectra(
    frequencies,
    tissue_amp_spectrum,
    background_amp_spectrum,
    tissue_complex_spectrum,
    background_complex_spectrum,
    output_path,
    title,
    show=False,
):
    common_length = min(
        len(frequencies),
        len(tissue_amp_spectrum),
        len(background_amp_spectrum),
        len(tissue_complex_spectrum),
        len(background_complex_spectrum),
    )
    frequencies = np.asarray(frequencies[:common_length], dtype=np.float32)
    tissue_amp_spectrum = np.asarray(tissue_amp_spectrum[:common_length], dtype=np.float32)
    background_amp_spectrum = np.asarray(background_amp_spectrum[:common_length], dtype=np.float32)
    tissue_complex_spectrum = np.asarray(tissue_complex_spectrum[:common_length], dtype=np.float32)
    background_complex_spectrum = np.asarray(background_complex_spectrum[:common_length], dtype=np.float32)

    positive = frequencies >= 0
    plt.rcParams.update({"font.size": DEFAULT_FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    amp_freq = frequencies[positive]
    amp_tissue = np.asarray(tissue_amp_spectrum[positive], dtype=np.float32)
    amp_background = np.asarray(background_amp_spectrum[positive], dtype=np.float32)
    complex_sort_index = np.argsort(frequencies)
    complex_freq = np.asarray(frequencies[complex_sort_index], dtype=np.float32)
    complex_tissue = np.asarray(tissue_complex_spectrum[complex_sort_index], dtype=np.float32)
    complex_background = np.asarray(background_complex_spectrum[complex_sort_index], dtype=np.float32)

    axes[0].plot(amp_freq, amp_tissue, color="tab:red", linewidth=1.8, label="Tissue")
    axes[0].plot(amp_freq, amp_background, color="black", linewidth=1.4, label="Background")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("ROI-mean power")
    axes[0].set_title("Amplitude tissue vs background")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False)

    axes[1].plot(complex_freq, complex_tissue, color="tab:blue", linewidth=1.8, label="Tissue")
    axes[1].plot(complex_freq, complex_background, color="black", linewidth=1.4, label="Background")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("ROI-mean power")
    axes[1].set_title("Complex tissue vs background")
    axes[1].set_xlim(float(np.min(complex_freq)), float(np.max(complex_freq)))
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(frameon=False)

    fig.suptitle(title, fontsize=DEFAULT_FONT_SIZE + 3)
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def save_metrics_workbook(output_path, metric_records, roi_records, spectrum_records, cumulative_records=None):
    if pd is None:
        write_csv(output_path.with_suffix(".metrics.csv"), metric_records)
        write_csv(output_path.with_suffix(".rois.csv"), roi_records)
        write_csv(output_path.with_suffix(".spectra.csv"), spectrum_records)
        if cumulative_records is not None:
            write_csv(output_path.with_suffix(".cumulative.csv"), cumulative_records)
        print("pandas is not installed; saved CSV files instead of Excel workbook.")
        return

    try:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame.from_records(metric_records).to_excel(writer, sheet_name="band_metrics", index=False)
            pd.DataFrame.from_records(roi_records).to_excel(writer, sheet_name="roi_vertices", index=False)
            pd.DataFrame.from_records(spectrum_records).to_excel(writer, sheet_name="roi_mean_spectra", index=False)
            pd.DataFrame.from_records(cumulative_records or []).to_excel(
                writer,
                sheet_name="cumulative_metrics",
                index=False,
            )
        print(f"Saved metrics workbook: {output_path}")
    except Exception as error:
        print(f"Could not save Excel workbook ({error}); saving CSV files instead.")
        write_csv(output_path.with_suffix(".metrics.csv"), metric_records)
        write_csv(output_path.with_suffix(".rois.csv"), roi_records)
        write_csv(output_path.with_suffix(".spectra.csv"), spectrum_records)
        if cumulative_records is not None:
            write_csv(output_path.with_suffix(".cumulative.csv"), cumulative_records)


def save_roi_progress(output_path, roi_records):
    save_metrics_workbook(
        Path(output_path),
        [],
        roi_records,
        [],
        [],
    )


def write_csv(output_path, records):
    if not records:
        return
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV: {output_path}")


def process_background_reference(background_stack_paths, out_dir, roi_records, reference_label="background"):
    background_stack_path = background_stack_paths[0]
    background_workbook_path = frequency_metrics_workbook_path(out_dir, background_stack_path.stem)
    print(f"Loading {reference_label} reference stack: {background_stack_path}")
    amplitude_stack, phase_stack = read_amp_phase_tiff_stack(background_stack_path)
    amplitude_stack, phase_stack = limit_stack_duration(
        amplitude_stack,
        phase_stack,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
    )
    mean_amp = np.mean(amplitude_stack, axis=0, dtype=np.float32)

    background_vertices = vertices_from_records(
        roi_records,
        f"shared_{reference_label}",
        "background",
        aliases=[background_stack_path.name, background_stack_path.stem, reference_label, "background"],
    )
    if background_vertices is None:
        background_mask, background_vertices = select_polygon_roi(
            mean_amp,
            f"{background_stack_path.name}: draw {reference_label.upper()} ROI, then press Enter",
        )
        print(f"Saved new {reference_label} ROI.")
        roi_records = replace_roi_records(
            roi_records,
            f"shared_{reference_label}",
            "background",
            background_vertices,
        )
        save_roi_progress(
            background_workbook_path,
            roi_records_for_stack(roi_records, background_stack_path),
        )
    else:
        background_mask = polygon_mask_for_image_shape(background_vertices, mean_amp.shape)
        print(f"Loaded saved {reference_label} ROI.")

    overlay_path = out_dir / f"{background_stack_path.stem}_{reference_label}_roi_overlay.png"
    save_roi_overlay(mean_amp, background_vertices, f"{reference_label.capitalize()} mean B-line with ROI", overlay_path)

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
    cumulative_limits = cumulative_upper_limits(
        DEFAULT_AMPLITUDE_FREQ_RANGE_HZ[1],
        DEFAULT_CUMULATIVE_BAND_STEP_HZ,
    )
    (
        frequencies,
        amplitude_band_images,
        complex_band_images,
        amplitude_roi_spectrum,
        complex_roi_spectrum,
        amplitude_cumulative_images,
        complex_cumulative_images,
    ) = compute_band_power_images_and_spectra(
        amplitude_stack=amplitude_stack,
        phase_stack=phase_stack,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        amplitude_bands=amplitude_bands,
        complex_bands=complex_bands,
        cumulative_upper_limits=cumulative_limits,
        tissue_mask=background_mask,
        chunk_x=DEFAULT_CHUNK_X,
    )

    results = {
        "stack_path": background_stack_path,
        "vertices": background_vertices,
        "mask": background_mask,
        "frequencies": frequencies,
        "amplitude_bands": amplitude_bands,
        "complex_bands": complex_bands,
        "amplitude_band_images": amplitude_band_images,
        "complex_band_images": complex_band_images,
        "amplitude_roi_spectrum": amplitude_roi_spectrum,
        "complex_roi_spectrum": complex_roi_spectrum,
        "cumulative_limits": cumulative_limits,
        "amplitude_cumulative_images": amplitude_cumulative_images,
        "complex_cumulative_images": complex_cumulative_images,
        "mean_amp": mean_amp,
        "roi_records": roi_records,
    }
    del amplitude_stack
    del phase_stack
    gc.collect()
    return results


def process_tissue_stack(stack_path, background_results, out_dir, roi_records):
    stack_workbook_path = frequency_metrics_workbook_path(out_dir, stack_path.stem)
    stack_saved_roi_records = load_saved_roi_records(out_dir, stack_path) if DEFAULT_REUSE_SAVED_ROIS else []
    saved_tissue_vertices = vertices_from_records(
        stack_saved_roi_records,
        stack_path.stem,
        "tissue",
        aliases=[stack_path.name, "shared_tissue"],
    )
    if saved_tissue_vertices is not None:
        roi_records = replace_roi_records(roi_records, stack_path.stem, "tissue", saved_tissue_vertices)
    saved_local_background_vertices = vertices_from_records(
        stack_saved_roi_records,
        stack_path.stem,
        "background",
        aliases=[stack_path.name, "shared_background_local"],
    )
    if saved_local_background_vertices is not None:
        roi_records = replace_roi_records(roi_records, stack_path.stem, "background", saved_local_background_vertices)
    saved_shared_background_vertices = vertices_from_records(
        stack_saved_roi_records,
        "shared_background",
        "background",
        aliases=["background"],
    )
    if saved_shared_background_vertices is not None:
        roi_records = replace_roi_records(roi_records, "shared_background", "background", saved_shared_background_vertices)

    print(f"Loading tissue stack: {stack_path}")
    amplitude_stack, phase_stack = read_amp_phase_tiff_stack(stack_path)
    amplitude_stack, phase_stack = limit_stack_duration(
        amplitude_stack,
        phase_stack,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        duration_seconds=DEFAULT_TISSUE_DURATION_SECONDS,
    )
    mean_amp = np.mean(amplitude_stack, axis=0, dtype=np.float32)

    tissue_vertices = vertices_from_records(
        roi_records,
        stack_path.stem,
        "tissue",
        aliases=[stack_path.name, "shared_tissue"],
    )
    if tissue_vertices is None:
        tissue_mask, tissue_vertices = select_polygon_roi(
            mean_amp,
            f"{stack_path.name}: draw TISSUE ROI, then press Enter",
        )
        print(f"Saved new tissue ROI for {stack_path.stem}.")
        roi_records = replace_roi_records(roi_records, stack_path.stem, "tissue", tissue_vertices)
        save_roi_progress(
            stack_workbook_path,
            roi_records_for_stack(roi_records, stack_path),
        )
    else:
        tissue_mask = polygon_mask_for_image_shape(tissue_vertices, mean_amp.shape)
        print(f"Loaded saved tissue ROI for {stack_path.stem}.")

    overlay_path = out_dir / f"{stack_path.stem}_tissue_roi_overlay.png"
    save_roi_overlay(mean_amp, tissue_vertices, "Tissue mean B-line with ROI", overlay_path)

    if background_results is None:
        background_vertices = vertices_from_records(
            roi_records,
            stack_path.stem,
            "background",
            aliases=[stack_path.name, "shared_background_local"],
        )
        if background_vertices is None:
            background_mask, background_vertices = select_polygon_roi(
                mean_amp,
                f"{stack_path.name}: draw BACKGROUND ROI, then press Enter",
            )
            print(f"Saved new local background ROI for {stack_path.stem}.")
            roi_records = replace_roi_records(roi_records, stack_path.stem, "background", background_vertices)
            save_roi_progress(
                stack_workbook_path,
                roi_records_for_stack(roi_records, stack_path, include_shared_background=False),
            )
        else:
            background_mask = polygon_mask_for_image_shape(background_vertices, mean_amp.shape)
            print(f"Loaded saved local background ROI for {stack_path.stem}.")

        background_overlay_path = out_dir / f"{stack_path.stem}_background_roi_overlay.png"
        save_roi_overlay(
            mean_amp,
            background_vertices,
            "Tissue-stack background ROI",
            background_overlay_path,
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
    else:
        background_mask = background_results["mask"]
        background_vertices = background_results["vertices"]
        amplitude_bands = background_results["amplitude_bands"]
        complex_bands = background_results["complex_bands"]

    cumulative_limits = cumulative_upper_limits(
        DEFAULT_AMPLITUDE_FREQ_RANGE_HZ[1],
        DEFAULT_CUMULATIVE_BAND_STEP_HZ,
    )

    (
        frequencies,
        amplitude_band_images,
        complex_band_images,
        amplitude_roi_spectrum,
        complex_roi_spectrum,
        amplitude_cumulative_images,
        complex_cumulative_images,
    ) = compute_band_power_images_and_spectra(
        amplitude_stack=amplitude_stack,
        phase_stack=phase_stack,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        amplitude_bands=amplitude_bands,
        complex_bands=complex_bands,
        cumulative_upper_limits=cumulative_limits,
        tissue_mask=tissue_mask,
        chunk_x=DEFAULT_CHUNK_X,
    )

    if background_results is None:
        (
            _background_frequencies,
            background_amplitude_band_images,
            background_complex_band_images,
            background_amplitude_roi_spectrum,
            background_complex_roi_spectrum,
            amplitude_background_cumulative_images,
            complex_background_cumulative_images,
        ) = compute_band_power_images_and_spectra(
            amplitude_stack=amplitude_stack,
            phase_stack=phase_stack,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
            amplitude_bands=amplitude_bands,
            complex_bands=complex_bands,
            cumulative_upper_limits=cumulative_limits,
            tissue_mask=background_mask,
            chunk_x=DEFAULT_CHUNK_X,
        )
    else:
        background_amplitude_stack, background_phase_stack = read_amp_phase_tiff_stack(
            background_results["stack_path"]
        )
        background_amplitude_stack, background_phase_stack = limit_stack_duration(
            background_amplitude_stack,
            background_phase_stack,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
        )
        matched_frame_count = min(amplitude_stack.shape[0], background_amplitude_stack.shape[0])
        background_amplitude_stack = background_amplitude_stack[:matched_frame_count]
        background_phase_stack = background_phase_stack[:matched_frame_count]
        (
            _background_frequencies,
            background_amplitude_band_images,
            background_complex_band_images,
            background_amplitude_roi_spectrum,
            background_complex_roi_spectrum,
            amplitude_background_cumulative_images,
            complex_background_cumulative_images,
        ) = compute_band_power_images_and_spectra(
            amplitude_stack=background_amplitude_stack,
            phase_stack=background_phase_stack,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
            amplitude_bands=amplitude_bands,
            complex_bands=complex_bands,
            cumulative_upper_limits=cumulative_limits,
            tissue_mask=background_results["mask"],
            chunk_x=DEFAULT_CHUNK_X,
        )
        del background_amplitude_stack
        del background_phase_stack

    base_name = stack_path.stem
    plot_band_montage(
        amplitude_band_images,
        amplitude_bands,
        title=f"{stack_path.name} amplitude band power",
        output_path=(out_dir / f"{base_name}_amplitude_frequency_band_power.png") if DEFAULT_SAVE_FIGURES else None,
        columns=DEFAULT_MONTAGE_COLUMNS,
        show=DEFAULT_SHOW_FIGURES,
    )
    plot_band_montage(
        complex_band_images,
        complex_bands,
        title=f"{stack_path.name} complex band power",
        output_path=(out_dir / f"{base_name}_complex_frequency_band_power.png") if DEFAULT_SAVE_FIGURES else None,
        columns=DEFAULT_MONTAGE_COLUMNS,
        show=DEFAULT_SHOW_FIGURES,
    )
    plot_roi_spectra(
        frequencies,
        amplitude_roi_spectrum,
        background_amplitude_roi_spectrum,
        complex_roi_spectrum,
        background_complex_roi_spectrum,
        output_path=out_dir / f"{base_name}_tissue_vs_background_power_spectra.png",
        title=f"{stack_path.name} tissue vs background ROI-mean temporal power spectra",
        show=DEFAULT_SHOW_FIGURES,
    )

    metric_records = []
    metric_records.extend(
        compute_band_metrics(
            amplitude_band_images[:, tissue_mask].reshape(len(amplitude_bands), -1),
            background_amplitude_band_images[:, background_mask].reshape(len(amplitude_bands), -1),
            amplitude_bands,
            "amplitude",
            stack_path.name,
        )
    )
    metric_records.extend(
        compute_band_metrics(
            complex_band_images[:, tissue_mask].reshape(len(complex_bands), -1),
            background_complex_band_images[:, background_mask].reshape(len(complex_bands), -1),
            complex_bands,
            "complex",
            stack_path.name,
        )
    )

    cumulative_records = []
    cumulative_records.extend(
        compute_cumulative_metrics(
            amplitude_cumulative_images,
            tissue_mask,
            amplitude_background_cumulative_images,
            background_mask,
            cumulative_limits,
            "amplitude",
            stack_path.name,
        )
    )
    cumulative_records.extend(
        compute_cumulative_metrics(
            complex_cumulative_images,
            tissue_mask,
            complex_background_cumulative_images,
            background_mask,
            cumulative_limits,
            "complex",
            stack_path.name,
        )
    )
    plot_cumulative_metrics(
        cumulative_records,
        out_dir / f"{base_name}_cumulative_frequency_metrics.png",
        title=f"{stack_path.name} cumulative frequency-band metrics",
        show=DEFAULT_SHOW_FIGURES,
    )

    spectrum_records = []
    common_spectrum_length = min(
        len(frequencies),
        len(amplitude_roi_spectrum),
        len(background_amplitude_roi_spectrum),
        len(complex_roi_spectrum),
        len(background_complex_roi_spectrum),
    )
    for index, frequency in enumerate(frequencies[:common_spectrum_length]):
        spectrum_records.append(
            {
                "stack_name": stack_path.name,
                "frequency_hz": float(frequency),
                "amplitude_tissue_power": float(amplitude_roi_spectrum[index]),
                "amplitude_background_power": float(background_amplitude_roi_spectrum[index]),
                "complex_tissue_power": float(complex_roi_spectrum[index]),
                "complex_background_power": float(background_complex_roi_spectrum[index]),
            }
        )

    save_metrics_workbook(
        stack_workbook_path,
        metric_records,
        roi_records_for_stack(roi_records, stack_path, include_shared_background=background_results is not None),
        spectrum_records,
        cumulative_records,
    )

    if background_results is None:
        comparison_background_amplitude_stack = amplitude_stack
        comparison_background_phase_stack = phase_stack
        comparison_background_mask = background_mask
    else:
        comparison_background_amplitude_stack, comparison_background_phase_stack = read_amp_phase_tiff_stack(
            background_results["stack_path"]
        )
        comparison_background_amplitude_stack, comparison_background_phase_stack = limit_stack_duration(
            comparison_background_amplitude_stack,
            comparison_background_phase_stack,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
        )
        comparison_background_mask = background_results["mask"]

    run_downsample_comparison_session(
        stack_path=stack_path,
        tissue_mask=tissue_mask,
        background_mask=comparison_background_mask,
        amplitude_stack=amplitude_stack,
        phase_stack=phase_stack,
        background_amplitude_stack=comparison_background_amplitude_stack,
        background_phase_stack=comparison_background_phase_stack,
        out_dir=out_dir,
    )

    if background_results is not None:
        del comparison_background_amplitude_stack
        del comparison_background_phase_stack

    del amplitude_stack
    del phase_stack
    del mean_amp
    del amplitude_band_images
    del complex_band_images
    del amplitude_cumulative_images
    del complex_cumulative_images
    gc.collect()
    return metric_records, spectrum_records, cumulative_records, roi_records


def compute_band_metrics(tissue_band_values, background_band_values, bands, signal_type, stack_name):
    records = []
    for band_idx, (low_hz, high_hz) in enumerate(bands):
        tissue_values = np.asarray(tissue_band_values[band_idx], dtype=np.float32).reshape(-1)
        background_values = np.asarray(background_band_values[band_idx], dtype=np.float32).reshape(-1)
        tissue_values = tissue_values[np.isfinite(tissue_values)]
        background_values = background_values[np.isfinite(background_values)]
        tissue_mean = float(np.mean(tissue_values)) if tissue_values.size else np.nan
        tissue_std = float(np.std(tissue_values, ddof=1)) if tissue_values.size > 1 else np.nan
        background_mean = float(np.mean(background_values)) if background_values.size else np.nan
        background_std = float(np.std(background_values, ddof=1)) if background_values.size > 1 else np.nan
        cnr = np.nan
        if np.isfinite(background_std) and background_std > 0:
            cnr = (tissue_mean - background_mean) / background_std
        records.append(
            {
                "stack_name": stack_name,
                "signal_type": signal_type,
                "band_index": int(band_idx),
                "band_low_hz": float(low_hz),
                "band_high_hz": float(high_hz),
                "tissue_mean_band_power": tissue_mean,
                "tissue_std_band_power": tissue_std,
                "background_mean_band_power": background_mean,
                "background_std_band_power": background_std,
                "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
            }
        )
    return records


def cumulative_upper_limits(stop_hz, step_hz):
    upper_limits = np.arange(float(step_hz), float(stop_hz) + 0.5 * float(step_hz), float(step_hz))
    return [float(value) for value in upper_limits]


def cumulative_band_images(band_images, bands, upper_limits, signal_type):
    cumulative = np.zeros((len(upper_limits),) + band_images.shape[1:], dtype=np.float32)
    for idx, upper_hz in enumerate(upper_limits):
        if signal_type == "amplitude":
            include = [band_idx for band_idx, (_low, high) in enumerate(bands) if high <= upper_hz + 1e-6]
        else:
            include = [
                band_idx
                for band_idx, (low, high) in enumerate(bands)
                if (low < upper_hz + 1e-6) and (high > -upper_hz - 1e-6)
            ]
        if include:
            cumulative[idx] = np.sum(band_images[include], axis=0, dtype=np.float32)
    return cumulative


def compute_cumulative_metrics(cumulative_images, tissue_mask, background_images, background_mask, upper_limits, signal_type, stack_name):
    records = []
    for idx, upper_hz in enumerate(upper_limits):
        tissue_values = np.asarray(cumulative_images[idx][tissue_mask], dtype=np.float32).reshape(-1)
        background_values = np.asarray(background_images[idx][background_mask], dtype=np.float32).reshape(-1)
        tissue_values = tissue_values[np.isfinite(tissue_values)]
        background_values = background_values[np.isfinite(background_values)]
        tissue_mean = float(np.mean(tissue_values)) if tissue_values.size else np.nan
        background_mean = float(np.mean(background_values)) if background_values.size else np.nan
        background_std = float(np.std(background_values, ddof=1)) if background_values.size > 1 else np.nan
        cnr = np.nan
        if np.isfinite(background_std) and background_std > 0:
            cnr = (tissue_mean - background_mean) / background_std
        tbr = np.nan
        if np.isfinite(background_mean) and background_mean > 0:
            tbr = tissue_mean / background_mean
        records.append(
            {
                "stack_name": stack_name,
                "signal_type": signal_type,
                "upper_frequency_hz": float(upper_hz),
                "frequency_band_label": (
                    f"0 to {upper_hz:g} Hz"
                    if signal_type == "amplitude"
                    else f"-{upper_hz:g} to {upper_hz:g} Hz"
                ),
                "tissue_mean_cumulative_power": tissue_mean,
                "background_mean_cumulative_power": background_mean,
                "background_std_cumulative_power": background_std,
                "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
                "tbr": float(tbr) if np.isfinite(tbr) else np.nan,
            }
        )
    return records


def compute_single_image_metrics(image, tissue_mask, background_image, background_mask):
    tissue_values = np.asarray(image[tissue_mask], dtype=np.float32).reshape(-1)
    background_values = np.asarray(background_image[background_mask], dtype=np.float32).reshape(-1)
    tissue_values = tissue_values[np.isfinite(tissue_values)]
    background_values = background_values[np.isfinite(background_values)]
    tissue_mean = float(np.mean(tissue_values)) if tissue_values.size else np.nan
    tissue_std = float(np.std(tissue_values, ddof=1)) if tissue_values.size > 1 else np.nan
    background_mean = float(np.mean(background_values)) if background_values.size else np.nan
    background_std = float(np.std(background_values, ddof=1)) if background_values.size > 1 else np.nan
    cnr = np.nan
    if np.isfinite(background_std) and background_std > 0:
        cnr = (tissue_mean - background_mean) / background_std
    return {
        "tissue_mean": tissue_mean,
        "tissue_std": tissue_std,
        "background_mean": background_mean,
        "background_std": background_std,
        "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
    }


def compute_fullband_power_images(amplitude_stack, phase_stack, frame_rate_hz, notch_band_hz, upper_limit_hz=None):
    amplitude_stack = np.asarray(amplitude_stack, dtype=np.float32)
    phase_stack = np.unwrap(np.asarray(phase_stack, dtype=np.float32), axis=0)
    amplitude_stack = fft_bandstop_stack_axis0(
        amplitude_stack,
        frame_rate_hz=frame_rate_hz,
        stop_band_hz=notch_band_hz,
    )
    complex_stack = (amplitude_stack * np.exp(1j * phase_stack)).astype(np.complex64, copy=False)
    complex_stack = fft_bandstop_stack_axis0(
        complex_stack,
        frame_rate_hz=frame_rate_hz,
        stop_band_hz=notch_band_hz,
    )

    amplitude_power = variance_equivalent_real_power_spectrum_axis0(amplitude_stack)
    complex_power = normalized_power_spectrum_axis0(complex_stack)
    frequencies = np.fft.fftfreq(amplitude_stack.shape[0], d=1.0 / float(frame_rate_hz)).astype(np.float32)
    if upper_limit_hz is None:
        upper_limit_hz = float(frame_rate_hz) / 2.0
    amplitude_full = cumulative_images_from_power_spectrum(
        amplitude_power,
        frequencies,
        [float(upper_limit_hz)],
        signal_type="amplitude",
    )[0]
    complex_full = cumulative_images_from_power_spectrum(
        complex_power,
        frequencies,
        [float(upper_limit_hz)],
        signal_type="complex",
    )[0]
    return (
        amplitude_full.astype(np.float32, copy=False),
        complex_full.astype(np.float32, copy=False),
    )


def comparison_label(method_name, frame_rate_hz):
    return f"{method_name.capitalize()} {frame_rate_hz:g} Hz"


def plot_downsample_comparison_images(records, output_path, stack_name):
    plt.rcParams.update({"font.size": DEFAULT_FONT_SIZE})
    fig, axes = plt.subplots(2, len(records), figsize=(4.2 * len(records), 7.2), constrained_layout=True)
    if len(records) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    amplitude_images = np.stack([record["amplitude_image"] for record in records], axis=0)
    complex_images = np.stack([record["complex_image"] for record in records], axis=0)
    amp_vmin, amp_vmax = montage_clim(amplitude_images)
    complex_vmin, complex_vmax = montage_clim(complex_images)

    for col, record in enumerate(records):
        axes[0, col].imshow(
            record["amplitude_image"].T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=amp_vmin,
            vmax=amp_vmax,
        )
        axes[0, col].set_title(comparison_label(record["method"], record["frame_rate_hz"]))
        axes[0, col].set_xlabel("X pixel")
        axes[0, col].set_ylabel("Depth")

        axes[1, col].imshow(
            record["complex_image"].T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=complex_vmin,
            vmax=complex_vmax,
        )
        axes[1, col].set_xlabel("X pixel")
        axes[1, col].set_ylabel("Depth")

    axes[0, 0].set_ylabel("Depth\nAmplitude")
    axes[1, 0].set_ylabel("Depth\nComplex")
    upper_limit_hz = records[0]["upper_frequency_limit_hz"]
    fig.suptitle(
        f"{stack_name}: downsampling comparison (power integrated to {upper_limit_hz:g} Hz)",
        fontsize=DEFAULT_FONT_SIZE + 3,
    )
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def plot_downsample_comparison_metrics(records, output_path, stack_name):
    plt.rcParams.update({"font.size": DEFAULT_FONT_SIZE})
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 10.4), constrained_layout=True)
    labels = [comparison_label(record["method"], record["frame_rate_hz"]) for record in records]
    x = np.arange(len(records), dtype=np.int32)
    width = 0.36

    amp_cnr = [record["amplitude_cnr"] for record in records]
    complex_cnr = [record["complex_cnr"] for record in records]
    amp_tissue = [record["amplitude_tissue_mean"] for record in records]
    complex_tissue = [record["complex_tissue_mean"] for record in records]
    amp_background = [record["amplitude_background_std"] for record in records]
    complex_background = [record["complex_background_std"] for record in records]

    axes[0].bar(x - width / 2, amp_cnr, width=width, color="tab:red", label="Amplitude")
    axes[0].bar(x + width / 2, complex_cnr, width=width, color="tab:blue", label="Complex")
    axes[0].set_ylabel("CNR")
    axes[0].set_title("Full-band dynamic-image CNR")
    axes[0].grid(True, axis="y", alpha=0.28)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, amp_tissue, width=width, color="tab:red", label="Amplitude")
    axes[1].bar(x + width / 2, complex_tissue, width=width, color="tab:blue", label="Complex")
    axes[1].set_ylabel("Tissue mean")
    axes[1].set_title("Tissue mean signal")
    axes[1].grid(True, axis="y", alpha=0.28)

    axes[2].bar(x - width / 2, amp_background, width=width, color="tab:red", label="Amplitude")
    axes[2].bar(x + width / 2, complex_background, width=width, color="tab:blue", label="Complex")
    axes[2].set_ylabel("Background std")
    axes[2].set_title("Background noise floor")
    axes[2].grid(True, axis="y", alpha=0.28)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=0)

    upper_limit_hz = records[0]["upper_frequency_limit_hz"]
    fig.suptitle(
        f"{stack_name}: skip vs integrate downsampling (to {upper_limit_hz:g} Hz)",
        fontsize=DEFAULT_FONT_SIZE + 3,
    )
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if DEFAULT_SHOW_FIGURES:
        plt.show(block=True)
    plt.close(fig)


def save_downsample_comparison_workbook(output_path, records):
    rows = []
    for record in records:
        rows.append(
            {
                "method": record["method"],
                "frame_rate_hz": float(record["frame_rate_hz"]),
                "comparison_target_frame_rate_hz": float(record["comparison_target_frame_rate_hz"]),
                "upper_frequency_limit_hz": float(record["upper_frequency_limit_hz"]),
                "frame_count": int(record["frame_count"]),
                "downsample_factor": int(record["downsample_factor"]),
                "amplitude_tissue_mean": record["amplitude_tissue_mean"],
                "amplitude_tissue_std": record["amplitude_tissue_std"],
                "amplitude_background_mean": record["amplitude_background_mean"],
                "amplitude_background_std": record["amplitude_background_std"],
                "amplitude_cnr": record["amplitude_cnr"],
                "complex_tissue_mean": record["complex_tissue_mean"],
                "complex_tissue_std": record["complex_tissue_std"],
                "complex_background_mean": record["complex_background_mean"],
                "complex_background_std": record["complex_background_std"],
                "complex_cnr": record["complex_cnr"],
            }
        )

    if pd is None:
        write_csv(output_path.with_suffix(".csv"), rows)
        return

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame.from_records(rows).to_excel(writer, sheet_name="downsample_compare", index=False)
    print(f"Saved metrics workbook: {output_path}")


def run_downsample_comparison_session(
    stack_path,
    tissue_mask,
    background_mask,
    amplitude_stack,
    phase_stack,
    background_amplitude_stack,
    background_phase_stack,
    out_dir,
):
    if not DEFAULT_ENABLE_DOWNSAMPLE_COMPARISON:
        return

    source_frame_rate_hz = float(DEFAULT_FRAME_RATE_HZ)
    source_frames = amplitude_stack.shape[0]
    duration_seconds = source_frames / source_frame_rate_hz
    max_target_hz = source_frame_rate_hz

    for target_frame_rate_hz in DEFAULT_DOWNSAMPLE_TARGET_FRAME_RATES_HZ:
        target_frame_rate_hz = float(target_frame_rate_hz)
        if target_frame_rate_hz <= 0 or target_frame_rate_hz > max_target_hz:
            print(f"Skipping invalid comparison target frame rate: {target_frame_rate_hz:g} Hz")
            continue
        factor = source_frame_rate_hz / target_frame_rate_hz
        rounded_factor = int(round(factor))
        if abs(factor - rounded_factor) > 1e-6 or rounded_factor < 1:
            print(
                f"Skipping target {target_frame_rate_hz:g} Hz because source {source_frame_rate_hz:g} Hz "
                "is not an integer multiple."
            )
            continue

        comparison_records = []
        for method in DEFAULT_DOWNSAMPLE_METHODS:
            if method == "original":
                method_amplitude_stack = amplitude_stack
                method_phase_stack = phase_stack
                method_background_amplitude_stack = background_amplitude_stack
                method_background_phase_stack = background_phase_stack
                method_frame_rate_hz = source_frame_rate_hz
                method_factor = 1
            elif method == "skip":
                method_amplitude_stack, method_phase_stack = downsample_amp_phase_skip(
                    amplitude_stack,
                    phase_stack,
                    rounded_factor,
                )
                method_background_amplitude_stack, method_background_phase_stack = downsample_amp_phase_skip(
                    background_amplitude_stack,
                    background_phase_stack,
                    rounded_factor,
                )
                method_frame_rate_hz = target_frame_rate_hz
                method_factor = rounded_factor
            elif method == "integrate":
                method_amplitude_stack, method_phase_stack = downsample_amp_phase_integrate(
                    amplitude_stack,
                    phase_stack,
                    rounded_factor,
                )
                method_background_amplitude_stack, method_background_phase_stack = downsample_amp_phase_integrate(
                    background_amplitude_stack,
                    background_phase_stack,
                    rounded_factor,
                )
                method_frame_rate_hz = target_frame_rate_hz
                method_factor = rounded_factor
            else:
                continue

            matched_frame_count = min(
                method_amplitude_stack.shape[0],
                method_background_amplitude_stack.shape[0],
            )
            method_amplitude_stack = np.ascontiguousarray(
                method_amplitude_stack[:matched_frame_count],
                dtype=np.float32,
            )
            method_phase_stack = np.ascontiguousarray(
                method_phase_stack[:matched_frame_count],
                dtype=np.float32,
            )
            method_background_amplitude_stack = np.ascontiguousarray(
                method_background_amplitude_stack[:matched_frame_count],
                dtype=np.float32,
            )
            method_background_phase_stack = np.ascontiguousarray(
                method_background_phase_stack[:matched_frame_count],
                dtype=np.float32,
            )

            amplitude_image, complex_image = compute_fullband_power_images(
                method_amplitude_stack,
                method_phase_stack,
                frame_rate_hz=method_frame_rate_hz,
                notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
                upper_limit_hz=float(target_frame_rate_hz) / 2.0,
            )
            background_amplitude_image, background_complex_image = compute_fullband_power_images(
                method_background_amplitude_stack,
                method_background_phase_stack,
                frame_rate_hz=method_frame_rate_hz,
                notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
                upper_limit_hz=float(target_frame_rate_hz) / 2.0,
            )
            amp_metrics = compute_single_image_metrics(
                amplitude_image,
                tissue_mask,
                background_amplitude_image,
                background_mask,
            )
            complex_metrics = compute_single_image_metrics(
                complex_image,
                tissue_mask,
                background_complex_image,
                background_mask,
            )
            comparison_records.append(
                {
                    "method": method,
                    "frame_rate_hz": method_frame_rate_hz,
                    "comparison_target_frame_rate_hz": target_frame_rate_hz,
                    "upper_frequency_limit_hz": float(target_frame_rate_hz) / 2.0,
                    "frame_count": matched_frame_count,
                    "downsample_factor": method_factor,
                    "amplitude_image": amplitude_image,
                    "complex_image": complex_image,
                    "amplitude_tissue_mean": amp_metrics["tissue_mean"],
                    "amplitude_tissue_std": amp_metrics["tissue_std"],
                    "amplitude_background_mean": amp_metrics["background_mean"],
                    "amplitude_background_std": amp_metrics["background_std"],
                    "amplitude_cnr": amp_metrics["cnr"],
                    "complex_tissue_mean": complex_metrics["tissue_mean"],
                    "complex_tissue_std": complex_metrics["tissue_std"],
                    "complex_background_mean": complex_metrics["background_mean"],
                    "complex_background_std": complex_metrics["background_std"],
                    "complex_cnr": complex_metrics["cnr"],
                }
            )

        if not comparison_records:
            continue

        target_label = f"{int(round(target_frame_rate_hz))}Hz"
        image_output_path = out_dir / f"{stack_path.stem}_downsample_compare_{target_label}_images.png"
        metrics_output_path = out_dir / f"{stack_path.stem}_downsample_compare_{target_label}_metrics.png"
        workbook_output_path = out_dir / f"{stack_path.stem}_downsample_compare_{target_label}.xlsx"
        plot_downsample_comparison_images(comparison_records, image_output_path, stack_path.name)
        plot_downsample_comparison_metrics(comparison_records, metrics_output_path, stack_path.name)
        save_downsample_comparison_workbook(workbook_output_path, comparison_records)


def plot_cumulative_metrics(cumulative_records, output_path, title, show=False):
    plt.rcParams.update({"font.size": DEFAULT_FONT_SIZE})
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), constrained_layout=True)
    for signal_type, color in [("amplitude", "tab:red"), ("complex", "tab:blue")]:
        subset = [record for record in cumulative_records if record["signal_type"] == signal_type]
        subset.sort(key=lambda record: record["upper_frequency_hz"])
        x = [record["upper_frequency_hz"] for record in subset]
        cnr = [record["cnr"] for record in subset]
        tissue_mean = [record["tissue_mean_cumulative_power"] for record in subset]
        background_mean = [record["background_mean_cumulative_power"] for record in subset]
        axes[0].plot(x, cnr, "o-", color=color, linewidth=1.8, markersize=4.5, label=signal_type.capitalize())
        axes[1].plot(x, tissue_mean, "o-", color=color, linewidth=1.8, markersize=4.5, label=signal_type.capitalize())
        axes[2].plot(x, background_mean, "o-", color=color, linewidth=1.8, markersize=4.5, label=signal_type.capitalize())

    axes[0].set_xlabel("Band limit n (Hz)")
    axes[0].set_ylabel("CNR")
    axes[0].set_title("Cumulative band CNR: amplitude 0 to n, complex -n to n")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Band limit n (Hz)")
    axes[1].set_ylabel("Tissue mean power")
    axes[1].set_title("Cumulative tissue mean power")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(frameon=False)

    axes[2].set_xlabel("Band limit n (Hz)")
    axes[2].set_ylabel("Background mean power")
    axes[2].set_title("Cumulative background mean power")
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(frameon=False)

    fig.suptitle(title, fontsize=DEFAULT_FONT_SIZE + 3)
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def main():
    plt.ioff()
    tissue_stack_paths = iter_tiff_stacks(DEFAULT_TISSUE_INPUT_PATH)
    background_stack_paths = try_iter_tiff_stacks(DEFAULT_BACKGROUND_INPUT_PATH)
    out_dir = output_directory(DEFAULT_TISSUE_INPUT_PATH, DEFAULT_OUTPUT_DIR)
    roi_records = []
    background_results = None
    if background_stack_paths:
        background_results = process_background_reference(
            background_stack_paths,
            out_dir,
            roi_records,
            reference_label="background",
        )
        roi_records = background_results["roi_records"]
    else:
        print("No separate background/noise stack provided. Using background ROI from each tissue stack.")

    all_metric_records = []
    all_spectrum_records = []
    all_cumulative_records = []

    for stack_path in tissue_stack_paths:
        metric_records, spectrum_records, cumulative_records, roi_records = process_tissue_stack(
            stack_path,
            background_results,
            out_dir,
            roi_records,
        )
        all_metric_records.extend(metric_records)
        all_spectrum_records.extend(spectrum_records)
        all_cumulative_records.extend(cumulative_records)
    print("Finished frequency-band power analysis.")


if __name__ == "__main__":
    main()
