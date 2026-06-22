import argparse
import gc
import re
from statistics import NormalDist
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import Slider
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF

try:
    import pandas as pd
except ImportError:
    pd = None


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_DIR = r"E:\IOCTData\Lung Cancer mice 260601\260608\200Hz 2seconds Blines"
DEFAULT_BACKGROUND_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"  # Optional separate noise/background AMP+PHASE TIFF stack or directory.
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_TISSUE_DURATION_SECONDS = 2.0
DEFAULT_BACKGROUND_DURATION_SECONDS = 2.0
DEFAULT_CENTER_WAVELENGTH_NM = 840.0
DEFAULT_PROFILE_START_DEPTH = 0
DEFAULT_MAX_AUTOCORR_LAG = 200
DEFAULT_NOTCH_BAND_HZ = None
DEFAULT_AMPLITUDE_SCALE_LIMIT = 1000.0
DEFAULT_PHASE_NOISE_REFERENCE_START_DEPTH = None  # None uses the deepest quarter of the B-line.
DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT = 0.55  # Fit coefficient C in sigma_phase^2 = C*10^(-SNR/10) + floor^2.
DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2 = 0.0  # Residual phase-variance floor (rad^2).
DEFAULT_PHASE_NOISE_CENTRAL_PERCENTILE = 99.0  # Draw the central percentile band of the expected Gaussian phase noise.

DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 1
DEFAULT_DYNAMIC_CHUNK_X = 200

DEFAULT_VIEWER_TITLE_FONT_SIZE = 10
DEFAULT_VIEWER_LABEL_FONT_SIZE = 9
DEFAULT_VIEWER_TICK_FONT_SIZE = 8
DEFAULT_VIEWER_SUMMARY_FONT_SIZE = 9
DEFAULT_VIEWER_SUPTITLE_FONT_SIZE = 11

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

    complex_stack = reconstruct_complex_data(amplitude, phase)
    return complex_stack


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
        return complex_stack[:1].copy()
    if complex_stack.shape[0] <= max_frames:
        return complex_stack
    return np.ascontiguousarray(complex_stack[:max_frames], dtype=np.complex64)


def format_notch_band(notch_band_hz):
    if notch_band_hz is None:
        return "None"
    return f"{float(notch_band_hz[0]):g}-{float(notch_band_hz[1]):g} Hz"


def resolve_noise_start_depth(z_pixels, configured_start_depth):
    if configured_start_depth is None:
        return int(np.clip(np.floor(0.75 * z_pixels), 0, max(0, z_pixels - 1)))
    return int(np.clip(configured_start_depth, 0, max(0, z_pixels - 1)))


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
    return polygon_mask_for_image_shape(vertices, image.shape)


def normalize_roi_key(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def load_saved_tissue_vertices(stack_path):
    stack_path = Path(stack_path)
    stack_stem = stack_path.stem
    workbook_candidates = [
        stack_path.parent / "dynamic_timepoint_sufficiency" / f"{stack_stem}_dynamic_timepoint_metrics.xlsx",
        stack_path.parent / "frequency_band_power_analysis" / f"{stack_stem}_frequency_domain_metrics.xlsx",
    ]

    if pd is None:
        return None

    for workbook_path in workbook_candidates:
        if not workbook_path.exists():
            continue
        try:
            dataframe = pd.read_excel(workbook_path, sheet_name="roi_vertices")
            records = dataframe.to_dict("records")
        except Exception as error:
            print(f"Could not read ROI workbook {workbook_path} ({error}); trying next fallback.")
            continue

        if not records:
            continue

        if "source_stack" in records[0]:
            source_keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_stem)}
            subset = [
                record
                for record in records
                if str(record.get("roi", "")).strip().lower() == "tissue"
                and normalize_roi_key(record.get("source_stack", "")) in source_keys
            ]
        else:
            label_keys = {normalize_roi_key(stack_path.name), normalize_roi_key(stack_stem)}
            subset = [
                record
                for record in records
                if str(record.get("roi", "")).strip().lower() == "tissue"
                and normalize_roi_key(record.get("label", "")) in label_keys
            ]

        if not subset:
            continue

        subset.sort(key=lambda record: int(record["vertex_index"]))
        vertices = np.asarray(
            [[float(record["x_pixel"]), float(record["depth_pixel"])] for record in subset],
            dtype=np.float32,
        )
        if vertices.shape[0] >= 3:
            print(f"Loaded tissue ROI from {workbook_path}")
            return vertices

    return None


def expected_phase_variance_from_snr_db(
    snr_db,
    coefficient=DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT,
    variance_floor_rad2=DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2,
):
    snr_db = float(snr_db)
    coefficient = float(coefficient)
    variance_floor_rad2 = float(variance_floor_rad2)
    return coefficient * np.power(10.0, -snr_db / 10.0) + variance_floor_rad2


def circular_phase_std(phase_trace):
    phase_trace = np.asarray(phase_trace, dtype=np.float32)
    if phase_trace.size == 0:
        return np.nan
    resultant_length = float(np.abs(np.mean(np.exp(1j * phase_trace))))
    resultant_length = float(np.clip(resultant_length, 1e-8, 1.0))
    return float(np.sqrt(max(0.0, -2.0 * np.log(resultant_length))))


def percentile_band_sigma_multiplier(central_percentile):
    central_percentile = float(central_percentile)
    central_percentile = min(max(central_percentile, 0.0), 99.999)
    tail_probability = 0.5 + 0.5 * (central_percentile / 100.0)
    return float(NormalDist().inv_cdf(tail_probability))


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
            f"({error}). Falling back to deep tissue region."
        )
        return np.nan, None


def phase_to_delta_z_nm(phase_trace, center_wavelength_nm):
    unwrapped = np.unwrap(np.asarray(phase_trace, dtype=np.float32))
    relative_phase = unwrapped - unwrapped[0]
    delta_z_nm = relative_phase * float(center_wavelength_nm) / (2.0 * np.pi)
    return delta_z_nm.astype(np.float32, copy=False), unwrapped.astype(np.float32, copy=False)


def amplitude_autocorrelation(amplitude_trace, max_lag):
    amplitude_trace = np.asarray(amplitude_trace, dtype=np.float32)
    max_lag = min(int(max_lag), amplitude_trace.size - 1)
    if max_lag < 1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    centered = amplitude_trace - np.mean(amplitude_trace, dtype=np.float32)
    zero_lag = np.mean(centered * centered, dtype=np.float32)
    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    g_amp = np.full(lags.shape, np.nan, dtype=np.float32)
    if not np.isfinite(zero_lag) or zero_lag <= 0:
        return lags, g_amp

    for idx, lag in enumerate(lags):
        g_amp[idx] = np.mean(centered[:-lag] * centered[lag:], dtype=np.float32) / zero_lag
    return lags, g_amp


def amplitude_autocorrelation_uncentered(amplitude_trace, max_lag):
    amplitude_trace = np.asarray(amplitude_trace, dtype=np.float32)
    max_lag = min(int(max_lag), amplitude_trace.size - 1)
    if max_lag < 1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    denominator = np.mean(amplitude_trace * amplitude_trace, dtype=np.float32)
    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    g_amp = np.full(lags.shape, np.nan, dtype=np.float32)
    if not np.isfinite(denominator) or denominator <= 0:
        return lags, g_amp

    for idx, lag in enumerate(lags):
        g_amp[idx] = np.mean(
            amplitude_trace[:-lag] * amplitude_trace[lag:],
            dtype=np.float32,
        ) / denominator
    return lags, g_amp


def complex_field_autocorrelation(complex_trace, max_lag):
    complex_trace = np.asarray(complex_trace, dtype=np.complex64)
    max_lag = min(int(max_lag), complex_trace.size - 1)
    if max_lag < 1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.complex64)

    denominator = np.mean(np.abs(complex_trace) ** 2, dtype=np.float32)
    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    g1 = np.full(lags.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    if not np.isfinite(denominator) or denominator <= 0:
        return lags, g1

    for idx, lag in enumerate(lags):
        g1[idx] = np.mean(complex_trace[:-lag] * np.conj(complex_trace[lag:])) / denominator
    return lags, g1


def fft_bandstop_trace(trace, frame_rate_hz, stop_band_hz=None):
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size < 2 or stop_band_hz is None:
        return trace.astype(np.float32, copy=True)

    low_hz, high_hz = sorted([float(stop_band_hz[0]), float(stop_band_hz[1])])
    if high_hz <= 0 or high_hz <= low_hz:
        return trace.astype(np.float32, copy=True)

    frequencies = np.fft.fftfreq(trace.size, d=1.0 / float(frame_rate_hz))
    spectrum = np.fft.fft(trace)
    stop_mask = (np.abs(frequencies) >= low_hz) & (np.abs(frequencies) <= high_hz)
    spectrum[stop_mask] = 0
    filtered = np.fft.ifft(spectrum).real
    return filtered.astype(np.float32, copy=False)


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


def compute_dynamic_images(
    complex_stack,
    frame_rate_hz,
    notch_band_hz,
    uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
):
    frames, x_pixels, z_pixels = complex_stack.shape
    amplitude_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    complex_dynamic = np.empty((x_pixels, z_pixels), dtype=np.float32)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        complex_chunk = complex_stack[:, x0:x1, :]
        amplitude = np.abs(complex_chunk).astype(np.float32, copy=False)
        phase = np.unwrap(np.angle(complex_chunk).astype(np.float32, copy=False), axis=0)
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
        print(f"Dynamic maps: processed X {x0}-{x1} / {x_pixels}")

    return amplitude_dynamic, complex_dynamic


def notch_filter_complex_stack_amp_phase(
    complex_stack,
    frame_rate_hz,
    notch_band_hz,
    chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
):
    frames, x_pixels, z_pixels = complex_stack.shape
    filtered_stack = np.empty((frames, x_pixels, z_pixels), dtype=np.complex64)
    chunk_x = max(1, int(chunk_x))

    for x0 in range(0, x_pixels, chunk_x):
        x1 = min(x_pixels, x0 + chunk_x)
        complex_chunk = complex_stack[:, x0:x1, :]
        amplitude = np.abs(complex_chunk).astype(np.float32, copy=False)
        phase = np.unwrap(np.angle(complex_chunk).astype(np.float32, copy=False), axis=0)
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
        filtered_stack[:, x0:x1, :] = (
            amplitude * np.exp(1j * phase)
        ).astype(np.complex64, copy=False)
        print(f"Notch filter: processed X {x0}-{x1} / {x_pixels}")

    return filtered_stack


class AmpPhaseBlineTraceViewer:
    def __init__(
        self,
        complex_stack,
        stack_path,
        external_noise_sigma_q=np.nan,
        external_noise_source=None,
        phase_noise_variance_coefficient=DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT,
        phase_noise_variance_floor_rad2=DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
        profile_start_depth=0,
        max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    ):
        self.complex_stack = np.asarray(complex_stack, dtype=np.complex64)
        if self.complex_stack.ndim != 3:
            raise ValueError(f"Expected complex stack with shape (T, X, Z), got {self.complex_stack.shape}")

        self.stack_path = Path(stack_path)
        self.external_noise_sigma_q = float(external_noise_sigma_q)
        self.external_noise_source = external_noise_source
        self.phase_noise_variance_coefficient = float(phase_noise_variance_coefficient)
        self.phase_noise_variance_floor_rad2 = float(phase_noise_variance_floor_rad2)
        self.frame_rate_hz = float(frame_rate_hz)
        self.center_wavelength_nm = float(center_wavelength_nm)
        self.max_autocorr_lag = int(max_autocorr_lag)
        self.notch_band_hz = notch_band_hz
        self.dynamic_uniform_filter_size = int(dynamic_uniform_filter_size)
        self.dynamic_chunk_x = int(dynamic_chunk_x)
        self.frames, self.x_pixels, self.z_pixels = self.complex_stack.shape
        self.profile_start_depth = int(np.clip(profile_start_depth, 0, self.z_pixels - 1))
        self.phase_noise_reference_start_depth = resolve_noise_start_depth(
            self.z_pixels,
            DEFAULT_PHASE_NOISE_REFERENCE_START_DEPTH,
        )
        self.time_axis_s = np.arange(self.frames, dtype=np.float32) / np.float32(self.frame_rate_hz)
        if self.notch_band_hz is None:
            self.filtered_complex_stack = self.complex_stack
        else:
            self.filtered_complex_stack = notch_filter_complex_stack_amp_phase(
                self.complex_stack,
                frame_rate_hz=self.frame_rate_hz,
                notch_band_hz=self.notch_band_hz,
                chunk_x=self.dynamic_chunk_x,
            )
        self.mean_abs_bline = np.mean(np.abs(self.filtered_complex_stack), axis=0, dtype=np.float32)
        self.amplitude_dynamic, self.complex_dynamic = compute_dynamic_images(
            self.filtered_complex_stack,
            frame_rate_hz=self.frame_rate_hz,
            notch_band_hz=None,
            uniform_filter_size=self.dynamic_uniform_filter_size,
            chunk_x=self.dynamic_chunk_x,
        )
        self.dynamic_ratio = np.divide(
            self.amplitude_dynamic,
            self.complex_dynamic,
            out=np.zeros_like(self.amplitude_dynamic, dtype=np.float32),
            where=self.complex_dynamic > np.float32(0.0),
        ).astype(np.float32, copy=False)
        saved_tissue_vertices = load_saved_tissue_vertices(self.stack_path)
        if saved_tissue_vertices is not None:
            self.tissue_vertices = saved_tissue_vertices
            self.tissue_mask = polygon_mask_for_image_shape(saved_tissue_vertices, self.mean_abs_bline.shape)
        else:
            self.tissue_mask = select_polygon_roi(
                self.mean_abs_bline,
                f"{self.stack_path.name}: draw TISSUE ROI, then press Enter",
            )
            self.tissue_vertices = None
        if np.isfinite(self.external_noise_sigma_q) and self.external_noise_sigma_q > 0:
            self.noise_sigma_q = self.external_noise_sigma_q
            self.noise_sigma_source = "separate_background_stack"
        else:
            noise_region = self.filtered_complex_stack[:, :, self.phase_noise_reference_start_depth:]
            self.noise_sigma_q = estimate_sigma_q_from_complex_samples(
                noise_region.reshape(noise_region.shape[0], -1)
            )
            self.noise_sigma_source = "deep_tissue_region"

        self.selected_x = self.x_pixels // 2
        profile = self.mean_abs_bline[self.selected_x, self.profile_start_depth:]
        self.selected_depth = (
            int(np.argmax(profile) + self.profile_start_depth)
            if profile.size
            else self.profile_start_depth
        )

        self.fig = None
        self.ax_bline = None
        self.ax_amp_dynamic = None
        self.ax_complex_dynamic = None
        self.ax_amp_trace_raw = None
        self.ax_complex_scatter_raw = None
        self.ax_phase_trace = None
        self.ax_g_amp_raw = None
        self.ax_g_amp = None
        self.ax_g1_raw = None
        self.ax_g1 = None
        self.ax_summary = None
        self.ax_ratio = None
        self.im_bline = None
        self.im_amp_dynamic = None
        self.im_complex_dynamic = None
        self.cursor_bline = None
        self.cursor_amp_dynamic = None
        self.cursor_complex_dynamic = None
        self.amp_trace_raw_line = None
        self.complex_scatter_raw = None
        self.phase_trace_line = None
        self.phase_noise_upper_line = None
        self.phase_noise_lower_line = None
        self.g_amp_raw_line = None
        self.g_amp_line = None
        self.g1_raw_abs_line = None
        self.g1_abs_line = None
        self.summary_text = None
        self.ratio_scatter = None
        self.current_lags = None
        self.current_g_amp_raw = None
        self.current_g_amp = None
        self.current_g1_raw = None
        self.current_g1 = None
        self.current_amp_trace = None
        self.current_phase_trace = None
        self.current_unfiltered_complex_trace = None
        self.current_filtered_complex_trace = None
        self.current_complex_trace_raw_display = None
        self.current_pixel_snr_db = None
        self.current_expected_phase_std_rad = None
        self.current_amplitude_dynamic = self.amplitude_dynamic
        self.current_complex_dynamic = self.complex_dynamic
        self.brightness_slider = None
        self.bline_vmin = None
        self.bline_vmax = None
        self.amp_dynamic_vmin = None
        self.amp_dynamic_vmax = None
        self.complex_dynamic_vmin = None
        self.complex_dynamic_vmax = None
        self.pan_start = None

    def show(self):
        self.build_figure()
        plt.show(block=True)
        plt.close(self.fig)

    def style_axis(self, ax, title=None, xlabel=None, ylabel=None, title_font_size=None):
        if title is not None:
            ax.set_title(
                title,
                fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE if title_font_size is None else title_font_size,
            )
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=DEFAULT_VIEWER_LABEL_FONT_SIZE)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=DEFAULT_VIEWER_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=DEFAULT_VIEWER_TICK_FONT_SIZE)

    def build_figure(self):
        self.fig = plt.figure(figsize=(19.0, 14.0))
        grid = self.fig.add_gridspec(
            3,
            4,
            height_ratios=[1.0, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0, 0.68],
            hspace=0.42,
            wspace=0.34,
        )
        self.ax_bline = self.fig.add_subplot(grid[0, 0])
        self.ax_amp_dynamic = self.fig.add_subplot(
            grid[1, 0],
            sharex=self.ax_bline,
            sharey=self.ax_bline,
        )
        self.ax_complex_dynamic = self.fig.add_subplot(
            grid[2, 0],
            sharex=self.ax_bline,
            sharey=self.ax_bline,
        )
        self.ax_amp_trace_raw = self.fig.add_subplot(grid[0, 1])
        self.ax_g_amp_raw = self.fig.add_subplot(grid[1, 1])
        self.ax_g1_raw = self.fig.add_subplot(grid[2, 1])
        self.ax_phase_trace = self.fig.add_subplot(grid[0, 2])
        self.ax_g_amp = self.fig.add_subplot(grid[1, 2])
        self.ax_g1 = self.fig.add_subplot(grid[2, 2])
        self.ax_complex_scatter_raw = self.fig.add_subplot(grid[0, 3], projection="polar")
        self.ax_summary = self.fig.add_subplot(grid[1, 3])
        self.ax_ratio = self.fig.add_subplot(grid[2, 3])

        self.bline_vmin, self.bline_vmax = self.initial_bline_clim()
        self.im_bline = self.ax_bline.imshow(
            self.mean_abs_bline.T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            vmin=self.bline_vmin,
            vmax=self.bline_vmax,
        )
        self.style_axis(
            self.ax_bline,
            title="Mean abs B-line (click a pixel)",
            xlabel="X pixel",
            ylabel="Depth index",
        )

        self.amp_dynamic_vmin, self.amp_dynamic_vmax = self.initial_image_clim(
            self.amplitude_dynamic
        )
        self.im_amp_dynamic = self.ax_amp_dynamic.imshow(
            self.amplitude_dynamic.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=self.amp_dynamic_vmin,
            vmax=self.amp_dynamic_vmax,
        )
        self.style_axis(
            self.ax_amp_dynamic,
            title="Amplitude dynamic signal",
            xlabel="X pixel",
            ylabel="Depth index",
        )

        self.complex_dynamic_vmin, self.complex_dynamic_vmax = self.initial_image_clim(
            self.complex_dynamic
        )
        self.im_complex_dynamic = self.ax_complex_dynamic.imshow(
            self.complex_dynamic.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=self.complex_dynamic_vmin,
            vmax=self.complex_dynamic_vmax,
        )
        self.style_axis(
            self.ax_complex_dynamic,
            title="Complex dynamic signal, abs result",
            xlabel="X pixel",
            ylabel="Depth index",
        )

        tissue_intensity = np.asarray(self.mean_abs_bline[self.tissue_mask], dtype=np.float32)
        tissue_ratio = np.asarray(self.dynamic_ratio[self.tissue_mask], dtype=np.float32)
        self.ratio_scatter = self.ax_ratio.scatter(
            tissue_intensity,
            tissue_ratio,
            s=6,
            c="black",
            alpha=0.35,
            edgecolors="none",
        )
        self.style_axis(
            self.ax_ratio,
            title="Tissue ROI: amp/complex dynamic vs intensity",
            xlabel="OCT intensity",
            ylabel="Amp/complex dynamic",
        )
        self.ax_ratio.grid(True, alpha=0.25)

        self.cursor_bline = self.ax_bline.plot(
            [self.selected_x],
            [self.selected_depth],
            marker="+",
            markersize=14,
            color="red",
            linestyle="None",
        )[0]
        self.cursor_amp_dynamic = self.ax_amp_dynamic.plot(
            [self.selected_x],
            [self.selected_depth],
            marker="+",
            markersize=14,
            color="cyan",
            linestyle="None",
        )[0]
        self.cursor_complex_dynamic = self.ax_complex_dynamic.plot(
            [self.selected_x],
            [self.selected_depth],
            marker="+",
            markersize=14,
            color="cyan",
            linestyle="None",
        )[0]
        self.amp_trace_raw_line, = self.ax_amp_trace_raw.plot([], [], lw=1.2)
        self.style_axis(
            self.ax_amp_trace_raw,
            xlabel="Time (s)",
            ylabel="Amplitude",
        )
        self.ax_amp_trace_raw.grid(True, alpha=0.25)

        self.complex_scatter_raw = self.ax_complex_scatter_raw.scatter(
            [],
            [],
            s=5,
            c="black",
            alpha=0.55,
            marker="o",
            edgecolors="none",
        )
        self.ax_complex_scatter_raw.grid(True, alpha=0.25)
        self.ax_complex_scatter_raw.tick_params(axis="both", labelsize=DEFAULT_VIEWER_TICK_FONT_SIZE)

        self.ax_summary.axis("off")
        self.summary_text = self.ax_summary.text(
            0.02,
            0.98,
            "",
            transform=self.ax_summary.transAxes,
            va="top",
            ha="left",
            fontsize=DEFAULT_VIEWER_SUMMARY_FONT_SIZE,
            family="monospace",
            wrap=True,
        )

        self.phase_trace_line, = self.ax_phase_trace.plot([], [], lw=1.1)
        self.phase_noise_upper_line = self.ax_phase_trace.axhline(
            0.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        self.phase_noise_lower_line = self.ax_phase_trace.axhline(
            0.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        self.style_axis(
            self.ax_phase_trace,
            xlabel="Time (s)",
            ylabel="Unwrapped phase (rad)",
        )
        self.ax_phase_trace.grid(True, alpha=0.25)

        self.g_amp_raw_line, = self.ax_g_amp_raw.plot([], [], lw=1.3, color="red")
        self.style_axis(
            self.ax_g_amp_raw,
            xlabel="Lag time (s)",
            ylabel="g_amp raw",
        )
        self.ax_g_amp_raw.grid(True, alpha=0.25)

        self.g_amp_line, = self.ax_g_amp.plot([], [], lw=1.5, color="red")
        self.style_axis(
            self.ax_g_amp,
            xlabel="Lag time (s)",
            ylabel="g_amp(k)",
        )
        self.ax_g_amp.grid(True, alpha=0.25)

        self.g1_raw_abs_line, = self.ax_g1_raw.plot([], [], lw=1.3, color="red")
        self.style_axis(
            self.ax_g1_raw,
            xlabel="Lag time (s)",
            ylabel="|g1 raw|",
        )
        self.ax_g1_raw.grid(True, alpha=0.25)

        self.g1_abs_line, = self.ax_g1.plot([], [], lw=1.5, color="red")
        self.style_axis(
            self.ax_g1,
            xlabel="Lag time (s)",
            ylabel="|g1(k)|",
        )
        self.ax_g1.grid(True, alpha=0.25)

        colorbar_bline = self.fig.colorbar(self.im_bline, ax=self.ax_bline, label="Mean amplitude")
        colorbar_amp = self.fig.colorbar(self.im_amp_dynamic, ax=self.ax_amp_dynamic, label="Variance")
        colorbar_complex = self.fig.colorbar(self.im_complex_dynamic, ax=self.ax_complex_dynamic, label="Abs variance")
        for colorbar in (colorbar_bline, colorbar_amp, colorbar_complex):
            colorbar.ax.tick_params(labelsize=DEFAULT_VIEWER_TICK_FONT_SIZE)
            colorbar.ax.yaxis.label.set_size(DEFAULT_VIEWER_LABEL_FONT_SIZE)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.add_brightness_slider()

        self.fig.suptitle(
            f"{self.stack_path.name}    "
            f"T={self.frames}, X={self.x_pixels}, Z={self.z_pixels}, "
            f"frame rate={self.frame_rate_hz:g} Hz, "
            f"notch={format_notch_band(self.notch_band_hz)}",
            fontsize=DEFAULT_VIEWER_SUPTITLE_FONT_SIZE,
        )
        self.update_views()
        self.fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.96], h_pad=1.5, w_pad=1.0)

    def initial_bline_clim(self):
        return self.initial_image_clim(self.mean_abs_bline)

    @staticmethod
    def initial_image_clim(image):
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            return 0.0, 1.0
        vmin, vmax = np.percentile(finite_values, [1.0, 99.7])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return 0.0, 1.0
        return float(vmin), float(vmax)

    def add_brightness_slider(self):
        slider_ax = self.fig.add_axes([0.13, 0.02, 0.36, 0.025])
        self.brightness_slider = Slider(
            ax=slider_ax,
            label="Brightness",
            valmin=0.2,
            valmax=5.0,
            valinit=1.0,
            valstep=0.05,
        )
        self.brightness_slider.on_changed(self.on_brightness_changed)

    def on_brightness_changed(self, brightness):
        display_vmax = self.bline_vmin + (self.bline_vmax - self.bline_vmin) / brightness
        self.im_bline.set_clim(self.bline_vmin, display_vmax)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes in (self.ax_bline, self.ax_amp_dynamic, self.ax_complex_dynamic):
            if event.xdata is None or event.ydata is None:
                return
            if event.button == 2 or event.button == 3:
                self.pan_start = {
                    "xdata": float(event.xdata),
                    "ydata": float(event.ydata),
                    "xlim": self.ax_bline.get_xlim(),
                    "ylim": self.ax_bline.get_ylim(),
                }
                return
            self.selected_x = int(np.clip(round(event.xdata), 0, self.x_pixels - 1))
            self.selected_depth = int(np.clip(round(event.ydata), 0, self.z_pixels - 1))
            self.update_views()

    def on_release(self, event):
        self.pan_start = None

    def on_motion(self, event):
        if self.pan_start is None:
            return
        if event.inaxes not in (self.ax_bline, self.ax_amp_dynamic, self.ax_complex_dynamic):
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = float(event.xdata) - self.pan_start["xdata"]
        dy = float(event.ydata) - self.pan_start["ydata"]
        x0, x1 = self.pan_start["xlim"]
        y0, y1 = self.pan_start["ylim"]
        self.set_image_limits((x0 - dx, x1 - dx), (y0 - dy, y1 - dy))

    def on_scroll(self, event):
        if event.inaxes not in (self.ax_bline, self.ax_amp_dynamic, self.ax_complex_dynamic):
            return
        if event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        xlim = self.ax_bline.get_xlim()
        ylim = self.ax_bline.get_ylim()
        new_xlim = self.zoom_limits(xlim, float(event.xdata), scale, 0, self.x_pixels - 1)
        new_ylim = self.zoom_limits(ylim, float(event.ydata), scale, 0, self.z_pixels - 1)
        self.set_image_limits(new_xlim, new_ylim)

    @staticmethod
    def zoom_limits(limits, center, scale, lower_bound, upper_bound):
        lo, hi = limits
        new_lo = center - (center - lo) * scale
        new_hi = center + (hi - center) * scale
        span = new_hi - new_lo
        max_span = upper_bound - lower_bound
        if span >= max_span:
            return lower_bound, upper_bound
        if new_lo < lower_bound:
            new_hi += lower_bound - new_lo
            new_lo = lower_bound
        if new_hi > upper_bound:
            new_lo -= new_hi - upper_bound
            new_hi = upper_bound
        return new_lo, new_hi

    def set_image_limits(self, xlim, ylim):
        for axis in (self.ax_bline, self.ax_amp_dynamic, self.ax_complex_dynamic):
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    def update_views(self):
        self.cursor_bline.set_data([self.selected_x], [self.selected_depth])
        self.cursor_amp_dynamic.set_data([self.selected_x], [self.selected_depth])
        self.cursor_complex_dynamic.set_data([self.selected_x], [self.selected_depth])
        unfiltered_complex_trace = self.complex_stack[:, self.selected_x, self.selected_depth]
        filtered_complex_trace = self.filtered_complex_stack[:, self.selected_x, self.selected_depth]
        amp_trace = np.abs(filtered_complex_trace).astype(np.float32, copy=False)
        wrapped_phase_trace = np.angle(filtered_complex_trace).astype(np.float32, copy=False)
        _, unwrapped_phase = phase_to_delta_z_nm(
            wrapped_phase_trace,
            center_wavelength_nm=self.center_wavelength_nm,
        )
        delta_z_nm = (
            (unwrapped_phase - unwrapped_phase[0])
            * np.float32(self.center_wavelength_nm)
            / np.float32(2.0 * np.pi)
        ).astype(np.float32, copy=False)
        amp_trace_centered = amp_trace - np.mean(amp_trace, dtype=np.float32)
        complex_trace_centered = filtered_complex_trace - np.mean(filtered_complex_trace)
        lags, g_amp_raw = amplitude_autocorrelation_uncentered(
            amp_trace,
            self.max_autocorr_lag,
        )
        lags, g_amp = amplitude_autocorrelation(amp_trace_centered, self.max_autocorr_lag)
        _, g1_raw = complex_field_autocorrelation(filtered_complex_trace, self.max_autocorr_lag)
        _, g1 = complex_field_autocorrelation(complex_trace_centered, self.max_autocorr_lag)
        lag_times_s = lags.astype(np.float32) / np.float32(self.frame_rate_hz)
        self.current_lags = lags
        self.current_g_amp_raw = g_amp_raw
        self.current_g_amp = g_amp
        self.current_g1_raw = g1_raw
        self.current_g1 = g1
        self.current_amp_trace = amp_trace
        self.current_phase_trace = unwrapped_phase
        self.current_unfiltered_complex_trace = unfiltered_complex_trace
        self.current_filtered_complex_trace = filtered_complex_trace
        self.current_complex_trace_raw_display = filtered_complex_trace
        amp_mean = float(np.nanmean(amp_trace))
        pixel_snr_db = np.nan
        expected_phase_std_rad = np.nan
        if np.isfinite(self.noise_sigma_q) and self.noise_sigma_q > 0 and np.isfinite(amp_mean):
            snr_linear_power = (amp_mean * amp_mean) / (2.0 * self.noise_sigma_q * self.noise_sigma_q)
            if np.isfinite(snr_linear_power) and snr_linear_power > 0:
                pixel_snr_db = float(10.0 * np.log10(max(snr_linear_power, 1e-12)))
                expected_phase_std_rad = float(
                    np.sqrt(
                        max(
                            0.0,
                            expected_phase_variance_from_snr_db(
                                pixel_snr_db,
                                coefficient=self.phase_noise_variance_coefficient,
                                variance_floor_rad2=self.phase_noise_variance_floor_rad2,
                            ),
                        )
                    )
                )
        self.current_pixel_snr_db = pixel_snr_db
        self.current_expected_phase_std_rad = expected_phase_std_rad

        self.amp_trace_raw_line.set_data(self.time_axis_s, amp_trace)
        self.ax_amp_trace_raw.set_xlim(self.time_axis_s[0], self.time_axis_s[-1])
        self.ax_amp_trace_raw.set_ylim(0, DEFAULT_AMPLITUDE_SCALE_LIMIT)
        self.ax_amp_trace_raw.set_title(
            f"Amplitude trace at X={self.selected_x}, depth={self.selected_depth}",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE,
        )

        polar_points = np.column_stack(
            [np.angle(filtered_complex_trace), np.abs(filtered_complex_trace)]
        )
        self.complex_scatter_raw.set_offsets(polar_points)
        self.ax_complex_scatter_raw.set_ylim(0, DEFAULT_AMPLITUDE_SCALE_LIMIT)
        self.ax_complex_scatter_raw.set_title(
            f"Complex trace polar plot at X={self.selected_x}, depth={self.selected_depth}",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE,
        )

        self.phase_trace_line.set_data(self.time_axis_s, unwrapped_phase)
        phase_center = float(np.nanmean(unwrapped_phase))
        if np.isfinite(expected_phase_std_rad):
            band_half_width = (
                percentile_band_sigma_multiplier(DEFAULT_PHASE_NOISE_CENTRAL_PERCENTILE)
                * expected_phase_std_rad
            )
            self.phase_noise_upper_line.set_ydata(
                [phase_center + band_half_width, phase_center + band_half_width]
            )
            self.phase_noise_lower_line.set_ydata(
                [phase_center - band_half_width, phase_center - band_half_width]
            )
            self.phase_noise_upper_line.set_visible(True)
            self.phase_noise_lower_line.set_visible(True)
        else:
            self.phase_noise_upper_line.set_visible(False)
            self.phase_noise_lower_line.set_visible(False)
        self.ax_phase_trace.set_xlim(self.time_axis_s[0], self.time_axis_s[-1])
        self.ax_phase_trace.relim()
        self.ax_phase_trace.autoscale_view(scalex=False, scaley=True)
        self.ax_phase_trace.set_title(
            f"Unwrapped phase at X={self.selected_x}, depth={self.selected_depth}",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE,
        )

        self.g_amp_raw_line.set_data(lag_times_s, g_amp_raw)
        self.ax_g_amp_raw.set_xlim(
            float(lag_times_s[0]) if lag_times_s.size else 0.0,
            float(lag_times_s[-1]) if lag_times_s.size else max(1, self.max_autocorr_lag) / self.frame_rate_hz,
        )
        self.ax_g_amp_raw.set_ylim(-1, 1.1)
        self.ax_g_amp_raw.set_title(
            "Amp autocorr, raw",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE - 1,
        )

        self.g_amp_line.set_data(lag_times_s, g_amp)
        self.ax_g_amp.set_xlim(
            float(lag_times_s[0]) if lag_times_s.size else 0.0,
            float(lag_times_s[-1]) if lag_times_s.size else max(1, self.max_autocorr_lag) / self.frame_rate_hz,
        )
        self.ax_g_amp.set_ylim(-1, 1.1)
        self.ax_g_amp.set_title(
            "Amp autocorr, centered",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE - 1,
        )

        self.g1_raw_abs_line.set_data(lag_times_s, np.abs(g1_raw))
        self.ax_g1_raw.set_xlim(
            float(lag_times_s[0]) if lag_times_s.size else 0.0,
            float(lag_times_s[-1]) if lag_times_s.size else max(1, self.max_autocorr_lag) / self.frame_rate_hz,
        )
        self.ax_g1_raw.set_ylim(0, 1.1)
        self.ax_g1_raw.set_title(
            "Complex autocorr, raw",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE - 1,
        )

        self.g1_abs_line.set_data(lag_times_s, np.abs(g1))
        self.ax_g1.set_xlim(
            float(lag_times_s[0]) if lag_times_s.size else 0.0,
            float(lag_times_s[-1]) if lag_times_s.size else max(1, self.max_autocorr_lag) / self.frame_rate_hz,
        )
        self.ax_g1.set_ylim(0, 1.1)
        self.ax_g1.set_title(
            "Complex autocorr, centered",
            fontsize=DEFAULT_VIEWER_TITLE_FONT_SIZE - 1,
        )

        dynamic_ratio_value = float(self.dynamic_ratio[self.selected_x, self.selected_depth])
        phase_std = circular_phase_std(wrapped_phase_trace)
        dz_std = float(np.nanstd(delta_z_nm))
        summary = (
            f"{self.stack_path.name}: X={self.selected_x}, depth={self.selected_depth}, "
            f"amp mean={amp_mean:.4g}, amp/complex dynamic={dynamic_ratio_value:.4g}, "
            f"circular phase std={phase_std:.4g} rad, "
            f"SNR-limited phase std={expected_phase_std_rad:.4g} rad, "
            f"delta-z std={dz_std:.4g} nm, "
            f"SNR={pixel_snr_db:.3g} dB"
        )
        print(summary)
        if self.summary_text is not None:
            self.summary_text.set_text(summary.replace(", ", "\n"))

        self.fig.canvas.draw_idle()


def process_one_stack(
    path,
    external_noise_sigma_q,
    external_noise_source,
    phase_noise_variance_coefficient,
    phase_noise_variance_floor_rad2,
    frame_rate_hz,
    tissue_duration_seconds,
    center_wavelength_nm,
    profile_start_depth,
    max_autocorr_lag,
    notch_band_hz,
    dynamic_uniform_filter_size,
    dynamic_chunk_x,
):
    print(f"Loading AMP+PHASE stack: {path}")
    complex_stack = read_amp_phase_tiff_stack(path)
    complex_stack = limit_stack_duration(
        complex_stack,
        frame_rate_hz=frame_rate_hz,
        duration_seconds=tissue_duration_seconds,
    )
    print(f"Reconstructed complex stack shape: {complex_stack.shape}, dtype={complex_stack.dtype}")
    viewer = AmpPhaseBlineTraceViewer(
        complex_stack=complex_stack,
        stack_path=path,
        external_noise_sigma_q=external_noise_sigma_q,
        external_noise_source=external_noise_source,
        phase_noise_variance_coefficient=phase_noise_variance_coefficient,
        phase_noise_variance_floor_rad2=phase_noise_variance_floor_rad2,
        frame_rate_hz=frame_rate_hz,
        center_wavelength_nm=center_wavelength_nm,
        profile_start_depth=profile_start_depth,
        max_autocorr_lag=max_autocorr_lag,
        notch_band_hz=notch_band_hz,
        dynamic_uniform_filter_size=dynamic_uniform_filter_size,
        dynamic_chunk_x=dynamic_chunk_x,
    )
    viewer.show()
    del viewer
    del complex_stack
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect saved AMP+PHASE OCT B-line time series. The TIFF format is "
            "assumed to be (time, X, 2Z), with amplitude in the first Z samples "
            "and phase in radians in the second Z samples."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Input TIFF file or directory containing TIFF stacks.",
    )
    parser.add_argument(
        "--background-input",
        default=DEFAULT_BACKGROUND_INPUT_PATH,
        help="Optional separate background/noise TIFF stack or directory.",
    )
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--tissue-duration-seconds", type=float, default=DEFAULT_TISSUE_DURATION_SECONDS)
    parser.add_argument("--background-duration-seconds", type=float, default=DEFAULT_BACKGROUND_DURATION_SECONDS)
    parser.add_argument("--lambda0-nm", type=float, default=DEFAULT_CENTER_WAVELENGTH_NM)
    parser.add_argument("--profile-start-depth", type=int, default=DEFAULT_PROFILE_START_DEPTH)
    parser.add_argument("--max-autocorr-lag", type=int, default=DEFAULT_MAX_AUTOCORR_LAG)
    default_notch_low = None if DEFAULT_NOTCH_BAND_HZ is None else DEFAULT_NOTCH_BAND_HZ[0]
    default_notch_high = None if DEFAULT_NOTCH_BAND_HZ is None else DEFAULT_NOTCH_BAND_HZ[1]
    parser.add_argument("--notch-low-hz", type=float, default=default_notch_low)
    parser.add_argument("--notch-high-hz", type=float, default=default_notch_high)
    parser.add_argument(
        "--dynamic-uniform-filter-size",
        type=int,
        default=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    )
    parser.add_argument("--dynamic-chunk-x", type=int, default=DEFAULT_DYNAMIC_CHUNK_X)
    return parser.parse_known_args()[0]


def run_viewer(
    input_path=DEFAULT_INPUT_DIR,
    background_input_path=DEFAULT_BACKGROUND_INPUT_PATH,
    frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
    tissue_duration_seconds=DEFAULT_TISSUE_DURATION_SECONDS,
    background_duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
    center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
    profile_start_depth=DEFAULT_PROFILE_START_DEPTH,
    max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
    notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
    dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
):
    paths = iter_tiff_stacks(input_path)
    external_noise_sigma_q, external_noise_source = load_external_noise_sigma_q(
        background_input_path,
        frame_rate_hz=frame_rate_hz,
        background_duration_seconds=background_duration_seconds,
        notch_band_hz=notch_band_hz,
        chunk_x=dynamic_chunk_x,
    )
    if np.isfinite(external_noise_sigma_q) and external_noise_sigma_q > 0:
        print(
            f"Using separate background stack for sigma_q: "
            f"{external_noise_source}, sigma_q={external_noise_sigma_q:.6g}"
        )
    else:
        print("No separate background stack provided. Using deep region of each tissue stack for sigma_q.")
    print(f"Found {len(paths)} TIFF stack(s). Close each figure to load the next stack.")
    for path in paths:
        process_one_stack(
            path=path,
            external_noise_sigma_q=external_noise_sigma_q,
            external_noise_source=external_noise_source,
            phase_noise_variance_coefficient=DEFAULT_PHASE_NOISE_VARIANCE_COEFFICIENT,
            phase_noise_variance_floor_rad2=DEFAULT_PHASE_NOISE_VARIANCE_FLOOR_RAD2,
            frame_rate_hz=frame_rate_hz,
            tissue_duration_seconds=tissue_duration_seconds,
            center_wavelength_nm=center_wavelength_nm,
            profile_start_depth=profile_start_depth,
            max_autocorr_lag=max_autocorr_lag,
            notch_band_hz=notch_band_hz,
            dynamic_uniform_filter_size=dynamic_uniform_filter_size,
            dynamic_chunk_x=dynamic_chunk_x,
        )


def main():
    if USE_COMMAND_LINE_ARGS:
        args = parse_args()
        notch_band_hz = None
        if args.notch_low_hz is not None and args.notch_high_hz is not None:
            notch_band_hz = (args.notch_low_hz, args.notch_high_hz)
        run_viewer(
            input_path=args.input,
            background_input_path=args.background_input,
            frame_rate_hz=args.frame_rate,
            tissue_duration_seconds=args.tissue_duration_seconds,
            background_duration_seconds=args.background_duration_seconds,
            center_wavelength_nm=args.lambda0_nm,
            profile_start_depth=args.profile_start_depth,
            max_autocorr_lag=args.max_autocorr_lag,
            notch_band_hz=notch_band_hz,
            dynamic_uniform_filter_size=args.dynamic_uniform_filter_size,
            dynamic_chunk_x=args.dynamic_chunk_x,
        )
        return

    run_viewer(
        input_path=DEFAULT_INPUT_DIR,
        background_input_path=DEFAULT_BACKGROUND_INPUT_PATH,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        tissue_duration_seconds=DEFAULT_TISSUE_DURATION_SECONDS,
        background_duration_seconds=DEFAULT_BACKGROUND_DURATION_SECONDS,
        center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
        profile_start_depth=DEFAULT_PROFILE_START_DEPTH,
        max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    )


if __name__ == "__main__":
    main()
