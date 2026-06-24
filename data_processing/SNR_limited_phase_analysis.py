import os
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
DEFAULT_INPUT_PATH = (
    r"E:\IOCTData\HighResData\50Hz_2s\paper\With_sub_background\incubatorOff\Bline-1-Yrpt800-X1104-Z187.tif"
)
DEFAULT_OUTPUT_DIR = None  # None saves results beside the input stack.
DEFAULT_INPUT_IS_SAVED_AMP_PHASE = True
DEFAULT_NOISE_INPUT_PATH =(
    r"E:\IOCTData\HighResData\50Hz_2s\noise\Wout_sub_background\AllOn\Bline-1-Yrpt800-X1104-Z187.tif"
)  # Set a noise-only TIFF here. None or "" falls back to signal-stack bottom-region noise.
DEFAULT_FRAME_RATE_HZ = 50.0
DEFAULT_CENTER_WAVELENGTH_NM = 840.0
DEFAULT_REFRACTIVE_INDEX = 1.0
DEFAULT_USE_HANNING = False
DEFAULT_SPECTRAL_HIGHPASS_SIZE = 21
DEFAULT_FFT_AMPLIFICATION = 100.0
DEFAULT_ANALYSIS_START_DEPTH = 10
DEFAULT_SNR_BIN_WIDTH_DB = 2.0
DEFAULT_MAX_G1_LAG = 200
DEFAULT_G1_MAX_PIXELS = 20000
DEFAULT_TARGET_TRACE_SNR_DB = (20.0, 30.0, 40.0, 50.0)
DEFAULT_THEORY_VARIANCE_COEFFICIENT = 0.5
DEFAULT_ENABLE_PHASE_VARIANCE_FIT = True
DEFAULT_FIT_SNR_RANGE_DB = (5.0, 45.0)
DEFAULT_UPPER_BOUND_PERCENTILE = 99.9
DEFAULT_FIT_MIN_BIN_PIXELS = 100
DEFAULT_FIT_GRID_POINTS = 500
DEFAULT_SAVE_DPI = 360
DEFAULT_SHOW_FIGURES = False

FONT_SIZES = {
    "title": 26,
    "label": 21,
    "tick": 18,
    "legend": 17,
    "annotation": 16,
    "message": 19,
}


def format_percentile_label(value):
    value = float(value)
    if np.isfinite(value) and value.is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def load_stack(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        stack = np.load(path)
    elif ext in {".tif", ".tiff"}:
        with TIFF.TiffFile(path) as tif:
            stack = np.stack([page.asarray() for page in tif.pages], axis=0)
        print(f"Loaded TIFF stack shape: {stack.shape}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got {stack.shape}")
    return np.asarray(stack, dtype=np.float32)


def load_saved_amp_phase_stack(path):
    stack = load_stack(path)
    if stack.shape[-1] % 2 != 0:
        raise ValueError(
            "Saved AMP+PHASE TIFF depth dimension must be even. "
            f"Got shape {stack.shape} from {path}"
        )
    return stack


def reconstruct_complex_from_amp_phase_stack(stack):
    """
    Regenerate complex OCT data saved by ThreadDnS.save_data().

    Saved AMP+PHASE TIFF frames store amplitude in the first half of the depth
    axis and phase in radians in the second half:
        saved[..., :Z] = abs(E)
        saved[..., Z:] = angle(E)
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(f"Expected 2D/3D AMP+PHASE stack, got {stack.shape}")
    if stack.shape[-1] % 2 != 0:
        raise ValueError(f"AMP+PHASE depth dimension must be even, got {stack.shape}")

    z_pixels = stack.shape[-1] // 2
    amplitude = stack[..., :z_pixels]
    phase = stack[..., z_pixels:]
    complex_stack = amplitude * np.exp(1j * phase)
    return complex_stack.astype(np.complex64, copy=False)


def prepare_depth_stack(
    raw_stack,
    use_hanning=False,
    spectral_highpass_size=51,
    fft_amplification=100.0,
):
    spectra = np.asarray(raw_stack, dtype=np.float32)
    frames, x_pixels, samples = spectra.shape
    half_depth = samples // 2
    window = np.hanning(samples).astype(np.float32) if use_hanning else None

    depth_complex = np.empty((frames, x_pixels, half_depth), dtype=np.complex64)
    for frame_idx in range(frames):
        frame_spectra = spectra[frame_idx]
        if spectral_highpass_size and spectral_highpass_size > 1:
            baseline = uniform_filter1d(
                frame_spectra,
                size=int(spectral_highpass_size),
                axis=1,
                mode="nearest",
            )
            frame_spectra = frame_spectra - baseline

        if window is not None:
            frame_spectra = frame_spectra * window[np.newaxis, :]

        depth_complex[frame_idx] = (
            np.fft.fft(frame_spectra, axis=1)[:, :half_depth] / samples
        ).astype(np.complex64, copy=False)

    depth_complex *= np.float32(fft_amplification)
    depth_magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    return depth_complex, depth_magnitude


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


def estimate_sigma_q_from_complex_samples(complex_samples):
    complex_samples = np.asarray(complex_samples, dtype=np.complex64)
    if complex_samples.size == 0:
        raise ValueError("Complex noise sample array is empty.")

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
        raise ValueError(f"Invalid sigma_q estimated from complex samples: {sigma_q}")
    return sigma_q


def circular_phase_mean_axis0(phase_stack):
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    unit_phasor = np.exp(1j * phase_stack).astype(np.complex64, copy=False)
    mean_phasor = np.mean(unit_phasor, axis=0, dtype=np.complex64)
    return np.angle(mean_phasor).astype(np.float32, copy=False)


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


def wrap_phase_residual(phase_stack, phase_mean):
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    phase_mean = np.asarray(phase_mean, dtype=np.float32)
    return np.angle(np.exp(1j * (phase_stack - phase_mean[np.newaxis, :, :]))).astype(
        np.float32,
        copy=False,
    )


def top_half_depth_stop(depth_pixels, analysis_start_depth=0):
    analysis_start_depth = int(max(0, analysis_start_depth))
    half_depth = int(max(1, depth_pixels // 2))
    return int(max(analysis_start_depth + 1, min(depth_pixels - 1, half_depth)))


def summarize_noise_distribution(noise_complex, max_points=20000):
    noise_complex = np.asarray(noise_complex, dtype=np.complex64)
    values = noise_complex.reshape(-1)
    if values.size == 0:
        return {
            "real": np.array([], dtype=np.float32),
            "imag": np.array([], dtype=np.float32),
            "sigma_q": np.nan,
        }

    if values.size > int(max_points):
        indices = np.linspace(0, values.size - 1, int(max_points), dtype=np.int64)
        values = values[indices]

    return {
        "real": np.real(values).astype(np.float32, copy=False),
        "imag": np.imag(values).astype(np.float32, copy=False),
        "sigma_q": estimate_sigma_q_from_complex_samples(noise_complex.reshape(noise_complex.shape[0], -1)),
    }


def measure_noise_sigma_q_from_roi(noise_complex, noise_roi_mask):
    roi_values = np.asarray(noise_complex[:, noise_roi_mask], dtype=np.complex64)
    if roi_values.size == 0:
        raise ValueError("Noise ROI is empty.")
    return estimate_sigma_q_from_complex_samples(roi_values)


class NoiseStartDepthPicker:
    def __init__(self, mean_depth_image):
        self.mean_depth_image = np.asarray(mean_depth_image, dtype=np.float32)
        self.x_pixels, self.depth_pixels = self.mean_depth_image.shape
        self.selected_depth = max(1, int(self.depth_pixels * 0.75))
        self.fig = None
        self.ax_image = None
        self.im_depth = None
        self.cursor_depth = None
        self.brightness_slider = None
        self.depth_image_vmin = None
        self.depth_image_vmax = None

    def show(self):
        self.fig, self.ax_image = plt.subplots(figsize=(9, 6))
        self.fig.subplots_adjust(bottom=0.14)

        self.depth_image_vmin, self.depth_image_vmax = self._initial_depth_image_clim()
        self.im_depth = self.ax_image.imshow(
            self.mean_depth_image.T,
            aspect="auto",
            origin="upper",
            cmap="gray",
            vmin=self.depth_image_vmin,
            vmax=self.depth_image_vmax,
        )
        self.cursor_depth = self.ax_image.axhline(self.selected_depth, color="red", lw=1.2)
        self.ax_image.set_title("Click the signal-analysis end depth, then close the window")
        self.ax_image.set_xlabel("X pixel")
        self.ax_image.set_ylabel("One-sided FFT depth index")
        self.fig.colorbar(self.im_depth, ax=self.ax_image, label="Mean FFT amplitude")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.add_brightness_slider()
        plt.show(block=True)
        depth_image_clim = self.im_depth.get_clim()
        plt.close(self.fig)
        return self.selected_depth, depth_image_clim

    def _initial_depth_image_clim(self):
        finite_values = self.mean_depth_image[np.isfinite(self.mean_depth_image)]
        if finite_values.size == 0:
            return 0.0, 1.0

        vmin, vmax = np.percentile(finite_values, [1.0, 99.5])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return 0.0, 1.0
        return float(vmin), float(vmax)

    def add_brightness_slider(self):
        slider_ax = self.fig.add_axes([0.14, 0.035, 0.42, 0.03])
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
        display_vmax = self.depth_image_vmin + (
            self.depth_image_vmax - self.depth_image_vmin
        ) / brightness
        self.im_depth.set_clim(self.depth_image_vmin, display_vmax)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax_image or event.ydata is None:
            return
        self.selected_depth = int(np.clip(round(event.ydata), 0, self.depth_pixels - 1))
        self.cursor_depth.set_ydata([self.selected_depth, self.selected_depth])
        self.fig.canvas.draw_idle()


def calculate_phase_noise_metrics(
    depth_complex,
    noise_start_depth,
    analysis_start_depth=10,
    center_wavelength_nm=840.0,
    refractive_index=1.0,
    snr_bin_width_db=2.0,
    external_sigma_q=None,
):
    magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    frames, x_pixels, depth_pixels = magnitude.shape
    analysis_start_depth = int(np.clip(analysis_start_depth, 0, depth_pixels - 1))
    noise_start_depth = int(np.clip(noise_start_depth, 0, depth_pixels - 1))
    if noise_start_depth <= analysis_start_depth:
        raise ValueError(
            "Signal analysis region is empty. Select an analysis end depth deeper than "
            f"{analysis_start_depth}."
        )

    if external_sigma_q is None:
        noise_region = np.asarray(depth_complex[:, :, :noise_start_depth], dtype=np.complex64)
        if noise_region.shape[2] < 2:
            raise ValueError("Top-half noise region is too shallow.")
        sigma_q = estimate_sigma_q_from_complex_samples(
            noise_region.reshape(noise_region.shape[0], -1)
        )
        noise_source = "signal_stack_top_half_region"
    else:
        sigma_q = float(external_sigma_q)
        noise_source = "separate_noise_stack_top_half"
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        raise ValueError(f"Invalid sigma_q calculated from selected source: {sigma_q}")

    mean_amplitude = np.mean(magnitude, axis=0, dtype=np.float32)
    snr_linear_power = (mean_amplitude * mean_amplitude) / np.float32(2.0 * sigma_q * sigma_q)
    snr_db_map = 10.0 * np.log10(np.maximum(snr_linear_power, np.float32(1e-12)))

    phase = np.angle(depth_complex).astype(np.float32, copy=False)
    phase_mean = circular_phase_mean_axis0(phase)
    phase_centered = wrap_phase_residual(phase, phase_mean)
    phase_std_map = circular_phase_std_axis0(phase)
    phase_variance_map = phase_std_map * phase_std_map

    wavelength_nm = float(center_wavelength_nm)
    opd_std_nm_map = phase_std_map * wavelength_nm / (2.0 * np.pi)
    displacement_std_nm_map = phase_std_map * wavelength_nm / (
        4.0 * np.pi * float(refractive_index)
    )

    signal_slice = (slice(None), slice(analysis_start_depth, noise_start_depth))
    snr_flat = snr_db_map[signal_slice].reshape(-1)
    phase_std_flat = phase_std_map[signal_slice].reshape(-1)
    phase_var_flat = phase_variance_map[signal_slice].reshape(-1)
    opd_flat = opd_std_nm_map[signal_slice].reshape(-1)
    displacement_flat = displacement_std_nm_map[signal_slice].reshape(-1)

    x_grid, z_grid = np.meshgrid(
        np.arange(x_pixels),
        np.arange(analysis_start_depth, noise_start_depth),
        indexing="ij",
    )
    x_flat = x_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    valid = (
        np.isfinite(snr_flat)
        & np.isfinite(phase_std_flat)
        & np.isfinite(opd_flat)
        & (phase_std_flat > 0)
    )
    if not np.any(valid):
        raise ValueError("No valid SNR/phase-noise pixels were found.")

    pixel_metrics = {
        "x": x_flat[valid],
        "z": z_flat[valid],
        "snr_db": snr_flat[valid],
        "phase_std_rad": phase_std_flat[valid],
        "phase_variance_rad2": phase_var_flat[valid],
        "opd_std_nm": opd_flat[valid],
        "displacement_std_nm": displacement_flat[valid],
    }

    binned = bin_snr_statistics(pixel_metrics, snr_bin_width_db)
    phase_variance_fit = None
    phase_variance_fit_upper = None
    if DEFAULT_ENABLE_PHASE_VARIANCE_FIT:
        phase_variance_fit = fit_phase_variance_model_from_binned(
            binned,
            snr_range_db=DEFAULT_FIT_SNR_RANGE_DB,
            value_key="phase_variance_rad2_median",
            min_bin_pixels=DEFAULT_FIT_MIN_BIN_PIXELS,
            grid_points=DEFAULT_FIT_GRID_POINTS,
        )
        phase_variance_fit_upper = fit_phase_variance_model_from_binned(
            binned,
            snr_range_db=DEFAULT_FIT_SNR_RANGE_DB,
            value_key="phase_variance_rad2_q95",
            min_bin_pixels=DEFAULT_FIT_MIN_BIN_PIXELS,
            grid_points=DEFAULT_FIT_GRID_POINTS,
        )
    g1 = calculate_g1_static_stability(
        depth_complex,
        analysis_start_depth,
        noise_start_depth,
        max_lag=DEFAULT_MAX_G1_LAG,
        max_pixels=DEFAULT_G1_MAX_PIXELS,
    )
    representatives = select_representative_traces(
        pixel_metrics,
        phase_centered,
        target_snr_db=DEFAULT_TARGET_TRACE_SNR_DB,
    )

    summary = {
        "frames": frames,
        "x_pixels": x_pixels,
        "depth_pixels": depth_pixels,
        "analysis_start_depth": analysis_start_depth,
        "noise_start_depth": noise_start_depth,
        "noise_sigma_q": sigma_q,
        "noise_source": noise_source,
        "phase_std_floor_rad_p1": float(np.nanpercentile(pixel_metrics["phase_std_rad"], 0.1)),
        "phase_std_median_rad": float(np.nanmedian(pixel_metrics["phase_std_rad"])),
        "opd_std_floor_nm_p1": float(np.nanpercentile(pixel_metrics["opd_std_nm"], 0.1)),
        "opd_std_median_nm": float(np.nanmedian(pixel_metrics["opd_std_nm"])),
        "displacement_std_floor_nm_p1": float(
            np.nanpercentile(pixel_metrics["displacement_std_nm"], 0.1)
        ),
        "displacement_std_median_nm": float(np.nanmedian(pixel_metrics["displacement_std_nm"])),
        "theory_variance_coefficient": float(DEFAULT_THEORY_VARIANCE_COEFFICIENT),
    }
    if phase_variance_fit is not None:
        summary["fit_coefficient"] = float(phase_variance_fit["coefficient"])
        summary["fit_sigma_floor_rad"] = float(phase_variance_fit["sigma_floor_rad"])
        summary["fit_sigma_floor_rad2"] = float(phase_variance_fit["sigma_floor2_rad2"])
        summary["fit_sigma_floor_opd_nm"] = float(
            phase_variance_fit["sigma_floor_rad"] * wavelength_nm / (2.0 * np.pi)
        )
        summary["fit_r_squared"] = float(phase_variance_fit["r_squared"])
        summary["fit_snr_min_db"] = float(phase_variance_fit["fit_snr_min_db"])
        summary["fit_snr_max_db"] = float(phase_variance_fit["fit_snr_max_db"])
        summary["fit_bin_count"] = int(phase_variance_fit["bin_count"])
    if phase_variance_fit_upper is not None:
        summary["fit_upper_percentile"] = float(DEFAULT_UPPER_BOUND_PERCENTILE)
        summary["fit_upper_coefficient"] = float(phase_variance_fit_upper["coefficient"])
        summary["fit_upper_sigma_floor_rad"] = float(phase_variance_fit_upper["sigma_floor_rad"])
        summary["fit_upper_sigma_floor_rad2"] = float(phase_variance_fit_upper["sigma_floor2_rad2"])
        summary["fit_upper_sigma_floor_opd_nm"] = float(
            phase_variance_fit_upper["sigma_floor_rad"] * wavelength_nm / (2.0 * np.pi)
        )
        summary["fit_upper_r_squared"] = float(phase_variance_fit_upper["r_squared"])
        summary["fit_upper_snr_min_db"] = float(phase_variance_fit_upper["fit_snr_min_db"])
        summary["fit_upper_snr_max_db"] = float(phase_variance_fit_upper["fit_snr_max_db"])
        summary["fit_upper_bin_count"] = int(phase_variance_fit_upper["bin_count"])

    return {
        "mean_amplitude": mean_amplitude,
        "snr_db_map": snr_db_map,
        "phase_centered": phase_centered,
        "phase_std_map": phase_std_map,
        "phase_variance_map": phase_variance_map,
        "opd_std_nm_map": opd_std_nm_map,
        "displacement_std_nm_map": displacement_std_nm_map,
        "pixel_metrics": pixel_metrics,
        "binned": binned,
        "phase_variance_fit": phase_variance_fit,
        "phase_variance_fit_upper": phase_variance_fit_upper,
        "g1": g1,
        "representatives": representatives,
        "summary": summary,
    }


def bin_snr_statistics(pixel_metrics, bin_width_db):
    snr = np.asarray(pixel_metrics["snr_db"], dtype=np.float32)
    if snr.size == 0:
        return {}

    bin_width_db = float(bin_width_db)
    snr_min = np.floor(np.nanmin(snr) / bin_width_db) * bin_width_db
    snr_max = np.ceil(np.nanmax(snr) / bin_width_db) * bin_width_db
    edges = np.arange(snr_min, snr_max + bin_width_db, bin_width_db)
    centers = edges[:-1] + bin_width_db / 2.0

    rows = []
    for idx, center in enumerate(centers):
        mask = (snr >= edges[idx]) & (snr < edges[idx + 1])
        if np.count_nonzero(mask) < 10:
            continue
        row = {
            "snr_bin_center_db": float(center),
            "snr_bin_start_db": float(edges[idx]),
            "snr_bin_stop_db": float(edges[idx + 1]),
            "pixel_count": int(np.count_nonzero(mask)),
        }
        for key in (
            "phase_std_rad",
            "phase_variance_rad2",
            "opd_std_nm",
            "displacement_std_nm",
        ):
            values = np.asarray(pixel_metrics[key], dtype=np.float32)[mask]
            row[f"{key}_median"] = float(np.nanmedian(values))
            row[f"{key}_q25"] = float(np.nanpercentile(values, 25.0))
            row[f"{key}_q75"] = float(np.nanpercentile(values, 75.0))
            row[f"{key}_q95"] = float(np.nanpercentile(values, 95.0))
        rows.append(row)
    return rows


def phase_variance_from_snr_db(snr_db, coefficient):
    snr_db = np.asarray(snr_db, dtype=np.float64)
    return float(coefficient) * np.power(10.0, -snr_db / 10.0)


def fit_phase_variance_model_from_binned(
    binned,
    snr_range_db,
    value_key="phase_variance_rad2_median",
    min_bin_pixels=100,
    grid_points=500,
):
    if not binned:
        return None

    snr_min, snr_max = sorted([float(snr_range_db[0]), float(snr_range_db[1])])
    fit_rows = [
        row for row in binned
        if float(row["snr_bin_center_db"]) >= snr_min
        and float(row["snr_bin_center_db"]) <= snr_max
        and int(row["pixel_count"]) >= int(min_bin_pixels)
        and np.isfinite(float(row[value_key]))
        and float(row[value_key]) > 0.0
    ]
    if len(fit_rows) < 3:
        return None

    snr_db = np.asarray([row["snr_bin_center_db"] for row in fit_rows], dtype=np.float64)
    phase_variance = np.asarray(
        [row[value_key] for row in fit_rows],
        dtype=np.float64,
    )
    snr_term = np.power(10.0, -snr_db / 10.0)

    min_variance = float(np.min(phase_variance))
    sigma_floor2_candidates = np.linspace(
        0.0,
        max(0.0, 0.98 * min_variance),
        max(10, int(grid_points)),
        dtype=np.float64,
    )

    best = None
    for sigma_floor2 in sigma_floor2_candidates:
        target = phase_variance - sigma_floor2
        coefficient = float(np.dot(snr_term, target) / np.dot(snr_term, snr_term))
        coefficient = max(0.0, coefficient)
        predicted = coefficient * snr_term + sigma_floor2
        residual = phase_variance - predicted
        rss = float(np.sum(residual * residual))
        if best is None or rss < best["rss"]:
            best = {
                "coefficient": coefficient,
                "sigma_floor2_rad2": float(sigma_floor2),
                "sigma_floor_rad": float(np.sqrt(max(0.0, sigma_floor2))),
                "rss": rss,
                "bin_count": len(fit_rows),
                "fit_snr_min_db": snr_min,
                "fit_snr_max_db": snr_max,
                "value_key": value_key,
            }

    if best is None:
        return None

    total = phase_variance - np.mean(phase_variance)
    tss = float(np.sum(total * total))
    best["r_squared"] = 1.0 - best["rss"] / tss if tss > 0 else np.nan
    return best


def select_representative_traces(pixel_metrics, phase_centered, target_snr_db):
    snr = np.asarray(pixel_metrics["snr_db"], dtype=np.float32)
    x_values = np.asarray(pixel_metrics["x"], dtype=np.int32)
    z_values = np.asarray(pixel_metrics["z"], dtype=np.int32)
    selected = []
    used = set()

    for target in target_snr_db:
        order = np.argsort(np.abs(snr - float(target)))
        chosen = None
        for flat_idx in order:
            coord = (int(x_values[flat_idx]), int(z_values[flat_idx]))
            if coord not in used:
                chosen = flat_idx
                used.add(coord)
                break
        if chosen is None:
            continue
        x = int(x_values[chosen])
        z = int(z_values[chosen])
        selected.append(
            {
                "target_snr_db": float(target),
                "x": x,
                "z": z,
                "actual_snr_db": float(snr[chosen]),
                "phase_trace_rad": phase_centered[:, x, z].astype(np.float32, copy=True),
            }
        )
    return selected


def calculate_g1_static_stability(
    depth_complex,
    analysis_start_depth,
    noise_start_depth,
    max_lag=200,
    max_pixels=20000,
):
    analysis = depth_complex[:, :, analysis_start_depth:noise_start_depth]
    frames = analysis.shape[0]
    reshaped = analysis.reshape(frames, -1)
    if reshaped.shape[1] > int(max_pixels):
        indices = np.linspace(0, reshaped.shape[1] - 1, int(max_pixels), dtype=np.int64)
        reshaped = reshaped[:, indices]

    max_lag = int(min(max_lag, frames - 1))
    lags = np.arange(max_lag + 1, dtype=np.int32)
    g1_abs = np.empty(max_lag + 1, dtype=np.float32)
    denom = float(np.mean(np.abs(reshaped) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        g1_abs[:] = np.nan
        return {"lag_frames": lags, "g1_abs": g1_abs}

    for lag in lags:
        if lag == 0:
            g1_abs[lag] = 1.0
            continue
        numerator = np.mean(np.conj(reshaped[:-lag]) * reshaped[lag:])
        g1_abs[lag] = float(np.abs(numerator / denom))
    return {"lag_frames": lags, "g1_abs": g1_abs}


def phase_power_spectrum(trace, frame_rate_hz):
    trace = np.asarray(trace, dtype=np.float32)
    trace = trace - np.mean(trace, dtype=np.float32)
    n_samples = trace.size
    if n_samples < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    spectrum = np.fft.rfft(trace) / n_samples
    frequency = np.fft.rfftfreq(n_samples, d=1.0 / float(frame_rate_hz))
    power = (np.abs(spectrum) ** 2).astype(np.float32, copy=False)
    return frequency.astype(np.float32), power


def plot_noise_floor_summary(
    metrics,
    noise_distribution,
    center_wavelength_nm,
    output_path,
    show_figures=False,
):
    pixel_metrics = metrics["pixel_metrics"]
    binned = metrics["binned"]
    phase_variance_fit = metrics.get("phase_variance_fit")
    phase_variance_fit_upper = metrics.get("phase_variance_fit_upper")
    summary = metrics["summary"]
    upper_percentile_label = format_percentile_label(DEFAULT_UPPER_BOUND_PERCENTILE)

    fig, axes = plt.subplots(2, 2, figsize=(26.0, 18.0), constrained_layout=True)
    ax_var, ax_noise, ax_std, ax_text = axes.ravel()

    scatter_kwargs = {"s": 5, "alpha": 0.12, "edgecolors": "none", "color": "0.35"}
    ax_var.scatter(
        pixel_metrics["snr_db"],
        pixel_metrics["phase_variance_rad2"],
        **scatter_kwargs,
    )
    snr_curve_db = np.linspace(
        max(0.0, float(np.nanpercentile(pixel_metrics["snr_db"], 0.5))),
        max(60.0, float(np.nanpercentile(pixel_metrics["snr_db"], 99.5))),
        300,
    )
    theory_variance = phase_variance_from_snr_db(
        snr_curve_db,
        DEFAULT_THEORY_VARIANCE_COEFFICIENT,
    )
    theory_std = np.sqrt(theory_variance)
    theory_opd_nm = theory_std * float(center_wavelength_nm) / (2.0 * np.pi)
    ax_var.plot(
        snr_curve_db,
        theory_variance,
        color="red",
        lw=4.4,
        label="Theory for power SNR",
    )
    if phase_variance_fit is not None:
        fit_variance = phase_variance_from_snr_db(
            snr_curve_db,
            phase_variance_fit["coefficient"],
        ) + phase_variance_fit["sigma_floor2_rad2"]
        ax_var.plot(
            snr_curve_db,
            fit_variance,
            color="dodgerblue",
            lw=4.0,
            ls="--",
            label="Fit: C*10^(-SNR/10)+floor",
        )
    if phase_variance_fit_upper is not None:
        fit_variance_upper = phase_variance_from_snr_db(
            snr_curve_db,
            phase_variance_fit_upper["coefficient"],
        ) + phase_variance_fit_upper["sigma_floor2_rad2"]
        ax_var.plot(
            snr_curve_db,
            fit_variance_upper,
            color="darkorange",
            lw=4.0,
            ls=":",
            label=f"{upper_percentile_label}th percentile fit",
        )
    ax_var.set_yscale("log")
    ax_var.set_xlabel("Pixel SNR (dB)", fontsize=FONT_SIZES["label"])
    ax_var.set_ylabel("Phase variance (rad^2)", fontsize=FONT_SIZES["label"])
    ax_var.set_title("Phase Variance vs SNR", fontsize=FONT_SIZES["title"])
    ax_var.legend(fontsize=FONT_SIZES["legend"])
    ax_var.grid(True, which="both", alpha=0.25)

    ax_std.scatter(pixel_metrics["snr_db"], pixel_metrics["phase_std_rad"], **scatter_kwargs)
    ax_std.plot(
        snr_curve_db,
        theory_std,
        color="red",
        lw=4.4,
        label="Theory for power SNR",
    )
    if phase_variance_fit is not None:
        fit_std = np.sqrt(
            phase_variance_from_snr_db(
                snr_curve_db,
                phase_variance_fit["coefficient"],
            ) + phase_variance_fit["sigma_floor2_rad2"]
        )
        ax_std.plot(
            snr_curve_db,
            fit_std,
            color="dodgerblue",
            lw=4.0,
            ls="--",
            label="Two-term fit",
        )
    if phase_variance_fit_upper is not None:
        fit_std_upper = np.sqrt(
            phase_variance_from_snr_db(
                snr_curve_db,
                phase_variance_fit_upper["coefficient"],
            ) + phase_variance_fit_upper["sigma_floor2_rad2"]
        )
        ax_std.plot(
            snr_curve_db,
            fit_std_upper,
            color="darkorange",
            lw=4.0,
            ls=":",
            label=f"{upper_percentile_label}th percentile fit",
        )
    ax_std.set_yscale("log")
    ax_std.set_xlabel("Pixel SNR (dB)", fontsize=FONT_SIZES["label"])
    ax_std.set_ylabel("Circular phase standard deviation (rad)", fontsize=FONT_SIZES["label"])
    ax_std.set_title("Circular Phase Noise vs SNR", fontsize=FONT_SIZES["title"])
    ax_std.grid(True, which="both", alpha=0.25)
    ax_std.legend(fontsize=FONT_SIZES["legend"])

    noise_real = np.asarray(noise_distribution["real"], dtype=np.float32)
    noise_imag = np.asarray(noise_distribution["imag"], dtype=np.float32)
    sigma_q = float(noise_distribution["sigma_q"])
    mean_real = float(np.nanmean(noise_real)) if noise_real.size else 0.0
    mean_imag = float(np.nanmean(noise_imag)) if noise_imag.size else 0.0
    ax_noise.scatter(
        noise_real,
        noise_imag,
        s=4,
        alpha=0.10,
        edgecolors="none",
        color="0.25",
    )
    if np.isfinite(sigma_q) and sigma_q > 0:
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        for multiplier, color, label in (
            (1.0, "red", r"$1\sigma_q$"),
            (2.0, "dodgerblue", r"$2\sigma_q$"),
        ):
            radius = multiplier * sigma_q
            ax_noise.plot(
                mean_real + radius * np.cos(theta),
                mean_imag + radius * np.sin(theta),
                color=color,
                lw=3.0,
                label=label,
            )
    ax_noise.set_xlim(-500.0, 500.0)
    ax_noise.set_ylim(-500.0, 500.0)
    ax_noise.set_aspect("equal", adjustable="box")
    ax_noise.set_xlabel("Real", fontsize=FONT_SIZES["label"])
    ax_noise.set_ylabel("Imaginary", fontsize=FONT_SIZES["label"])
    ax_noise.set_title("Noise-Stack IQ Distribution", fontsize=FONT_SIZES["title"])
    ax_noise.grid(True, alpha=0.25)
    if np.isfinite(sigma_q) and sigma_q > 0:
        ax_noise.legend(fontsize=FONT_SIZES["legend"], loc="upper right")
        ax_noise.text(
            0.03,
            0.03,
            f"$\\sigma_q$ = {sigma_q:.4g}",
            transform=ax_noise.transAxes,
            fontsize=FONT_SIZES["annotation"],
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    info = (
        f"sigma_q = {summary['noise_sigma_q']:.4g}\n"
        f"theory C = {summary['theory_variance_coefficient']:.4g}"
    )
    if phase_variance_fit is not None:
        info += (
            f"\n\nmedian-fit C = {summary['fit_coefficient']:.4g}"
            f"\nmedian-fit phase floor = {summary['fit_sigma_floor_rad']:.4g} rad"
            f"\nmedian-fit OPD floor = {summary['fit_sigma_floor_opd_nm']:.4g} nm"
            f"\nmedian-fit range = {summary['fit_snr_min_db']:.0f}-{summary['fit_snr_max_db']:.0f} dB"
        )
    if phase_variance_fit_upper is not None:
        info += (
            f"\n\n{upper_percentile_label}th-percentile-fit C = {summary['fit_upper_coefficient']:.4g}"
            f"\n{upper_percentile_label}th-percentile-fit phase floor = "
            f"{summary['fit_upper_sigma_floor_rad']:.4g} rad"
            f"\n{upper_percentile_label}th-percentile-fit OPD floor = "
            f"{summary['fit_upper_sigma_floor_opd_nm']:.4g} nm"
            f"\n{upper_percentile_label}th-percentile-fit range = "
            f"{summary['fit_upper_snr_min_db']:.0f}-{summary['fit_upper_snr_max_db']:.0f} dB"
        )
    ax_text.axis("off")
    ax_text.text(
        0.03,
        0.97,
        info,
        transform=ax_text.transAxes,
        fontsize=FONT_SIZES["message"],
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    style_all_axes(fig)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    if show_figures:
        plt.show(block=True)
    plt.close(fig)


def plot_binned_curve(ax, binned, metric_name, color, label, suffix="_median", fill_iqr=True):
    if not binned:
        return
    x = np.asarray([row["snr_bin_center_db"] for row in binned], dtype=np.float32)
    y = np.asarray([row[f"{metric_name}{suffix}"] for row in binned], dtype=np.float32)
    ax.plot(x, y, color=color, lw=4.0, label=label)
    if fill_iqr:
        q25 = np.asarray([row[f"{metric_name}_q25"] for row in binned], dtype=np.float32)
        q75 = np.asarray([row[f"{metric_name}_q75"] for row in binned], dtype=np.float32)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)


def plot_temporal_stability(metrics, frame_rate_hz, output_path, show_figures=False):
    representatives = metrics["representatives"]
    g1 = metrics["g1"]

    fig, axes = plt.subplots(3, 1, figsize=(18.0, 20.0), constrained_layout=True)
    ax_trace, ax_psd, ax_g1 = axes

    for trace_info in representatives:
        trace = trace_info["phase_trace_rad"]
        time_axis = np.arange(trace.size, dtype=np.float32) / float(frame_rate_hz)
        label = (
            f"SNR {trace_info['actual_snr_db']:.1f} dB "
            f"(X={trace_info['x']}, Z={trace_info['z']})"
        )
        ax_trace.plot(time_axis, trace, lw=2.0, label=label)

        frequency, power = phase_power_spectrum(trace, frame_rate_hz)
        if frequency.size > 0:
            ax_psd.plot(frequency, power, lw=2.0, label=f"{trace_info['actual_snr_db']:.1f} dB")

    ax_trace.set_xlabel("Time (s)", fontsize=FONT_SIZES["label"])
    ax_trace.set_ylabel("Mean-subtracted phase (rad)", fontsize=FONT_SIZES["label"])
    ax_trace.set_title("Representative Phase Traces", fontsize=FONT_SIZES["title"])
    ax_trace.legend(fontsize=FONT_SIZES["legend"], loc="best")
    ax_trace.grid(True, alpha=0.25)

    ax_psd.set_xlabel("Frequency (Hz)", fontsize=FONT_SIZES["label"])
    ax_psd.set_ylabel("Phase power (rad^2)", fontsize=FONT_SIZES["label"])
    ax_psd.set_title("Representative Phase Power Spectra", fontsize=FONT_SIZES["title"])
    ax_psd.set_yscale("log")
    ax_psd.grid(True, which="both", alpha=0.25)

    lags = np.asarray(g1["lag_frames"], dtype=np.float32)
    ax_g1.plot(lags / float(frame_rate_hz), g1["g1_abs"], color="red", lw=4.0)
    ax_g1.set_xlabel("Lag time (s)", fontsize=FONT_SIZES["label"])
    ax_g1.set_ylabel("|g1(tau)|", fontsize=FONT_SIZES["label"])
    ax_g1.set_ylim(0.0, 1.05)
    ax_g1.set_title("Static Complex-Field Stability", fontsize=FONT_SIZES["title"])
    ax_g1.grid(True, alpha=0.25)

    style_all_axes(fig)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    if show_figures:
        plt.show(block=True)
    plt.close(fig)


def style_all_axes(fig):
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])


def save_results_tables(metrics, output_base):
    pixel_metrics = metrics["pixel_metrics"]
    binned = metrics["binned"]
    summary = metrics["summary"]
    g1 = metrics["g1"]
    representatives = metrics["representatives"]

    pixel_table = {
        key: np.asarray(value)
        for key, value in pixel_metrics.items()
    }
    binned_table = binned
    summary_table = [{"metric": key, "value": value} for key, value in summary.items()]
    g1_table = {
        "lag_frames": np.asarray(g1["lag_frames"]),
        "g1_abs": np.asarray(g1["g1_abs"]),
    }

    trace_rows = []
    for trace_info in representatives:
        trace = trace_info["phase_trace_rad"]
        for frame_idx, phase_value in enumerate(trace):
            trace_rows.append(
                {
                    "target_snr_db": trace_info["target_snr_db"],
                    "actual_snr_db": trace_info["actual_snr_db"],
                    "x": trace_info["x"],
                    "z": trace_info["z"],
                    "frame": frame_idx,
                    "phase_rad": float(phase_value),
                }
            )

    if pd is not None:
        excel_path = f"{output_base}_phase_noise_metrics.xlsx"
        try:
            with pd.ExcelWriter(excel_path) as writer:
                pd.DataFrame(pixel_table).to_excel(writer, sheet_name="pixel_metrics", index=False)
                pd.DataFrame(binned_table).to_excel(writer, sheet_name="snr_bins", index=False)
                pd.DataFrame(summary_table).to_excel(writer, sheet_name="summary", index=False)
                pd.DataFrame(g1_table).to_excel(writer, sheet_name="g1", index=False)
                pd.DataFrame(trace_rows).to_excel(writer, sheet_name="phase_traces", index=False)
            print(f"Saved Excel metrics: {excel_path}")
            return
        except Exception as error:
            print(f"Could not save Excel workbook ({error}); saving CSV files instead.")

    csv_paths = {
        "pixel_metrics": (pixel_table, f"{output_base}_pixel_metrics.csv"),
        "snr_bins": (binned_table, f"{output_base}_snr_bins.csv"),
        "summary": (summary_table, f"{output_base}_summary.csv"),
        "g1": (g1_table, f"{output_base}_g1.csv"),
        "phase_traces": (trace_rows, f"{output_base}_phase_traces.csv"),
    }
    for _, (table, path) in csv_paths.items():
        save_csv_table(table, path)
        print(f"Saved CSV metrics: {path}")


def save_csv_table(table, path):
    if isinstance(table, dict):
        keys = list(table.keys())
        rows = zip(*[np.asarray(table[key]).reshape(-1) for key in keys])
    else:
        rows = table
        keys = sorted({key for row in rows for key in row.keys()}) if rows else []

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            if isinstance(row, dict):
                values = [row.get(key, "") for key in keys]
            else:
                values = row
            handle.write(",".join(str(value) for value in values) + "\n")


def output_base_for_input(input_path, output_dir):
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    return str(Path(output_dir) / input_path.stem)


def main():
    input_path = DEFAULT_INPUT_PATH
    output_base = output_base_for_input(input_path, DEFAULT_OUTPUT_DIR)
    external_sigma_q = None
    noise_distribution = None

    if DEFAULT_NOISE_INPUT_PATH:
        noise_path = DEFAULT_NOISE_INPUT_PATH
        if DEFAULT_INPUT_IS_SAVED_AMP_PHASE:
            noise_saved_stack = load_saved_amp_phase_stack(noise_path)
            noise_complex = reconstruct_complex_from_amp_phase_stack(noise_saved_stack)
            del noise_saved_stack
        else:
            noise_raw_stack = load_stack(noise_path)
            noise_complex, _ = prepare_depth_stack(
                noise_raw_stack,
                use_hanning=DEFAULT_USE_HANNING,
                spectral_highpass_size=DEFAULT_SPECTRAL_HIGHPASS_SIZE,
                fft_amplification=DEFAULT_FFT_AMPLIFICATION,
            )
            del noise_raw_stack

        noise_top_half_depth = top_half_depth_stop(
            noise_complex.shape[2],
            analysis_start_depth=0,
        )
        noise_distribution = summarize_noise_distribution(
            noise_complex[:, :, :noise_top_half_depth]
        )
        external_sigma_q = estimate_sigma_q_from_complex_samples(
            noise_complex[:, :, :noise_top_half_depth].reshape(noise_complex.shape[0], -1)
        )
        print(
            "Measured external sigma_q from top-half noise stack XZ range "
            f"(depth < {noise_top_half_depth}): {external_sigma_q:.6g}"
        )
        del noise_complex
    else:
        raise ValueError(
            "A separate noise TIFF stack is required for SNR_limited_phase_analysis. "
            "Please set DEFAULT_NOISE_INPUT_PATH to a valid noise-only stack acquired "
            "under the same system settings."
        )

    if DEFAULT_INPUT_IS_SAVED_AMP_PHASE:
        saved_stack = load_saved_amp_phase_stack(input_path)
        depth_complex = reconstruct_complex_from_amp_phase_stack(saved_stack)
        depth_magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
        del saved_stack
        print("Input treated as saved AMP+PHASE FFT-domain B-line stack.")
    else:
        raw_stack = load_stack(input_path)
        depth_complex, depth_magnitude = prepare_depth_stack(
            raw_stack,
            use_hanning=DEFAULT_USE_HANNING,
            spectral_highpass_size=DEFAULT_SPECTRAL_HIGHPASS_SIZE,
            fft_amplification=DEFAULT_FFT_AMPLIFICATION,
        )
        del raw_stack
        print("Input treated as raw spectral stack and reconstructed with FFT.")

    noise_start_depth = top_half_depth_stop(
        depth_complex.shape[2],
        analysis_start_depth=DEFAULT_ANALYSIS_START_DEPTH,
    )
    print(
        "Using top-half depth rule; "
        f"signal analysis end depth set to {noise_start_depth}."
    )

    metrics = calculate_phase_noise_metrics(
        depth_complex,
        noise_start_depth,
        analysis_start_depth=DEFAULT_ANALYSIS_START_DEPTH,
        center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
        refractive_index=DEFAULT_REFRACTIVE_INDEX,
        snr_bin_width_db=DEFAULT_SNR_BIN_WIDTH_DB,
        external_sigma_q=external_sigma_q,
    )

    summary_path = f"{output_base}_phase_noise_summary.png"
    temporal_path = f"{output_base}_phase_temporal_stability.png"
    plot_noise_floor_summary(
        metrics,
        noise_distribution,
        DEFAULT_CENTER_WAVELENGTH_NM,
        summary_path,
        show_figures=DEFAULT_SHOW_FIGURES,
    )
    print(f"Saved figure: {summary_path}")

    plot_temporal_stability(
        metrics,
        DEFAULT_FRAME_RATE_HZ,
        temporal_path,
        show_figures=DEFAULT_SHOW_FIGURES,
    )
    print(f"Saved figure: {temporal_path}")

    save_results_tables(metrics, output_base)

    summary = metrics["summary"]
    print(
        "Phase floor summary: "
        f"{summary['phase_std_floor_rad_p1']:.4g} rad, "
        f"{summary['opd_std_floor_nm_p1']:.4g} nm OPD, "
        f"{summary['displacement_std_floor_nm_p1']:.4g} nm displacement"
    )


if __name__ == "__main__":
    main()
