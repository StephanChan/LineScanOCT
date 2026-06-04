import argparse
import gc
import os
import re
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_DIR = r"E:\IOCTData\Lung Cancer mice 260601\AMP Phase"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_CENTER_WAVELENGTH_NM = 840.0
DEFAULT_PROFILE_START_DEPTH = 0
DEFAULT_MAX_AUTOCORR_LAG = 200
DEFAULT_SPECTRUM_MIN_HZ = 2.0
DEFAULT_NOTCH_BAND_HZ = (46.0, 48.0)
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 10
DEFAULT_DYNAMIC_CHUNK_X = 96

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

    complex_stack = reconstruct_complex_data(amplitude, phase)
    return complex_stack


def reconstruct_complex_data(amplitude, phase):
    amplitude = np.asarray(amplitude, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)


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


def single_sided_power_spectrum(trace, frame_rate_hz, min_frequency_hz=0.0):
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    centered = trace - np.mean(trace, dtype=np.float32)
    spectrum = np.fft.rfft(centered) / trace.size
    power = (np.abs(spectrum) ** 2).astype(np.float32, copy=False)
    frequencies = np.fft.rfftfreq(trace.size, d=1.0 / float(frame_rate_hz)).astype(
        np.float32,
        copy=False,
    )
    if min_frequency_hz > 0:
        keep = frequencies > np.float32(min_frequency_hz)
        frequencies = frequencies[keep]
        power = power[keep]
    return frequencies, power


def complex_power_spectrum(complex_trace, frame_rate_hz, min_abs_frequency_hz=0.0):
    complex_trace = np.asarray(complex_trace, dtype=np.complex64)
    if complex_trace.size < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    centered = complex_trace - np.mean(complex_trace)
    spectrum = np.fft.fftshift(np.fft.fft(centered) / complex_trace.size)
    power = (np.abs(spectrum) ** 2).astype(np.float32, copy=False)
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(complex_trace.size, d=1.0 / float(frame_rate_hz))
    ).astype(np.float32, copy=False)
    if min_abs_frequency_hz > 0:
        keep = np.abs(frequencies) > np.float32(min_abs_frequency_hz)
        frequencies = frequencies[keep]
        power = power[keep]
    return frequencies, power


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
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
        profile_start_depth=0,
        max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
        spectrum_min_hz=DEFAULT_SPECTRUM_MIN_HZ,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    ):
        self.complex_stack = np.asarray(complex_stack, dtype=np.complex64)
        if self.complex_stack.ndim != 3:
            raise ValueError(f"Expected complex stack with shape (T, X, Z), got {self.complex_stack.shape}")

        self.stack_path = Path(stack_path)
        self.frame_rate_hz = float(frame_rate_hz)
        self.center_wavelength_nm = float(center_wavelength_nm)
        self.max_autocorr_lag = int(max_autocorr_lag)
        self.spectrum_min_hz = float(spectrum_min_hz)
        self.notch_band_hz = notch_band_hz
        self.dynamic_uniform_filter_size = int(dynamic_uniform_filter_size)
        self.dynamic_chunk_x = int(dynamic_chunk_x)
        self.frames, self.x_pixels, self.z_pixels = self.complex_stack.shape
        self.profile_start_depth = int(np.clip(profile_start_depth, 0, self.z_pixels - 1))
        self.time_axis_s = np.arange(self.frames, dtype=np.float32) / np.float32(self.frame_rate_hz)
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
        self.ax_amp_spectrum = None
        self.ax_phase_spectrum = None
        self.im_bline = None
        self.im_amp_dynamic = None
        self.im_complex_dynamic = None
        self.cursor_bline = None
        self.cursor_amp_dynamic = None
        self.cursor_complex_dynamic = None
        self.amp_trace_raw_line = None
        self.complex_scatter_raw = None
        self.phase_trace_line = None
        self.g_amp_raw_line = None
        self.g_amp_line = None
        self.g1_raw_abs_line = None
        self.g1_abs_line = None
        self.summary_text = None
        self.amp_spectrum_line = None
        self.phase_spectrum_line = None
        self.current_lags = None
        self.current_g_amp_raw = None
        self.current_g_amp = None
        self.current_g1_raw = None
        self.current_g1 = None
        self.current_frequency_hz = None
        self.current_amp_power = None
        self.current_complex_frequency_hz = None
        self.current_complex_power = None
        self.current_amp_trace = None
        self.current_phase_trace = None
        self.current_unfiltered_complex_trace = None
        self.current_filtered_complex_trace = None
        self.current_complex_trace_raw_display = None
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

    def build_figure(self):
        self.fig = plt.figure(figsize=(19.0, 14.0))
        grid = self.fig.add_gridspec(
            3,
            4,
            height_ratios=[1.0, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0, 0.68],
        )
        self.ax_bline = self.fig.add_subplot(grid[0, 0])
        self.ax_amp_dynamic = self.fig.add_subplot(grid[1, 0])
        self.ax_complex_dynamic = self.fig.add_subplot(grid[2, 0])
        self.ax_amp_trace_raw = self.fig.add_subplot(grid[0, 1])
        amp_corr_grid = grid[1, 1].subgridspec(2, 1, hspace=0.28)
        complex_corr_grid = grid[1, 2].subgridspec(2, 1, hspace=0.28)
        self.ax_g_amp_raw = self.fig.add_subplot(amp_corr_grid[0, 0])
        self.ax_g_amp = self.fig.add_subplot(amp_corr_grid[1, 0])
        self.ax_amp_spectrum = self.fig.add_subplot(grid[2, 1])
        self.ax_phase_trace = self.fig.add_subplot(grid[0, 2])
        self.ax_g1_raw = self.fig.add_subplot(complex_corr_grid[0, 0])
        self.ax_g1 = self.fig.add_subplot(complex_corr_grid[1, 0])
        self.ax_phase_spectrum = self.fig.add_subplot(grid[2, 2])
        self.ax_complex_scatter_raw = self.fig.add_subplot(grid[0, 3], projection="polar")
        self.ax_summary = self.fig.add_subplot(grid[1, 3])

        self.bline_vmin, self.bline_vmax = self.initial_bline_clim()
        self.im_bline = self.ax_bline.imshow(
            self.mean_abs_bline.T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            vmin=self.bline_vmin,
            vmax=self.bline_vmax,
        )
        self.ax_bline.set_title("Mean abs B-line (click a pixel)")
        self.ax_bline.set_xlabel("X pixel")
        self.ax_bline.set_ylabel("Depth index")

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
        self.ax_amp_dynamic.set_title("Amplitude dynamic signal")
        self.ax_amp_dynamic.set_xlabel("X pixel")
        self.ax_amp_dynamic.set_ylabel("Depth index")

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
        self.ax_complex_dynamic.set_title("Complex dynamic signal, abs result")
        self.ax_complex_dynamic.set_xlabel("X pixel")
        self.ax_complex_dynamic.set_ylabel("Depth index")

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
        self.ax_amp_trace_raw.set_xlabel("Time (s)")
        self.ax_amp_trace_raw.set_ylabel("Amplitude")
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

        self.ax_summary.axis("off")
        self.summary_text = self.ax_summary.text(
            0.02,
            0.98,
            "",
            transform=self.ax_summary.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            wrap=True,
        )

        self.phase_trace_line, = self.ax_phase_trace.plot([], [], lw=1.1)
        self.ax_phase_trace.set_xlabel("Time (s)")
        self.ax_phase_trace.set_ylabel("Phase (rad)")
        self.ax_phase_trace.grid(True, alpha=0.25)

        self.g_amp_raw_line, = self.ax_g_amp_raw.plot([], [], lw=1.3, color="red")
        self.ax_g_amp_raw.set_xlabel("Lag k")
        self.ax_g_amp_raw.set_ylabel("g_amp raw")
        self.ax_g_amp_raw.grid(True, alpha=0.25)

        self.g_amp_line, = self.ax_g_amp.plot([], [], lw=1.5, color="red")
        self.ax_g_amp.set_xlabel("Lag k")
        self.ax_g_amp.set_ylabel("g_amp(k)")
        self.ax_g_amp.grid(True, alpha=0.25)

        self.g1_raw_abs_line, = self.ax_g1_raw.plot([], [], lw=1.3, color="red")
        self.ax_g1_raw.set_xlabel("Lag k")
        self.ax_g1_raw.set_ylabel("|g1 raw|")
        self.ax_g1_raw.grid(True, alpha=0.25)

        self.g1_abs_line, = self.ax_g1.plot([], [], lw=1.5, color="red")
        self.ax_g1.set_xlabel("Lag k")
        self.ax_g1.set_ylabel("|g1(k)|")
        self.ax_g1.grid(True, alpha=0.25)

        self.amp_spectrum_line, = self.ax_amp_spectrum.plot([], [], lw=1.4)
        self.ax_amp_spectrum.set_xlabel("Frequency (Hz)")
        self.ax_amp_spectrum.set_ylabel("Power")
        self.ax_amp_spectrum.grid(True, alpha=0.25)

        self.phase_spectrum_line, = self.ax_phase_spectrum.plot([], [], lw=1.4)
        self.ax_phase_spectrum.set_xlabel("Frequency (Hz)")
        self.ax_phase_spectrum.set_ylabel("Complex power")
        self.ax_phase_spectrum.grid(True, alpha=0.25)

        self.fig.colorbar(self.im_bline, ax=self.ax_bline, label="Mean amplitude")
        self.fig.colorbar(self.im_amp_dynamic, ax=self.ax_amp_dynamic, label="Variance")
        self.fig.colorbar(self.im_complex_dynamic, ax=self.ax_complex_dynamic, label="Abs variance")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.add_brightness_slider()

        self.fig.suptitle(
            f"{self.stack_path.name}    "
            f"T={self.frames}, X={self.x_pixels}, Z={self.z_pixels}, "
            f"frame rate={self.frame_rate_hz:g} Hz, "
            f"notch={self.notch_band_hz[0]:g}-{self.notch_band_hz[1]:g} Hz"
        )
        self.update_views()
        self.fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])

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
        phase_trace = np.angle(filtered_complex_trace).astype(np.float32, copy=False)
        _, unwrapped_phase = phase_to_delta_z_nm(
            phase_trace,
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
        frequency_hz, amp_power = single_sided_power_spectrum(
            amp_trace,
            self.frame_rate_hz,
            min_frequency_hz=self.spectrum_min_hz,
        )
        complex_frequency_hz, complex_power = complex_power_spectrum(
            filtered_complex_trace,
            self.frame_rate_hz,
            min_abs_frequency_hz=self.spectrum_min_hz,
        )
        self.current_lags = lags
        self.current_g_amp_raw = g_amp_raw
        self.current_g_amp = g_amp
        self.current_g1_raw = g1_raw
        self.current_g1 = g1
        self.current_frequency_hz = frequency_hz
        self.current_amp_power = amp_power
        self.current_complex_frequency_hz = complex_frequency_hz
        self.current_complex_power = complex_power
        self.current_amp_trace = amp_trace
        self.current_phase_trace = unwrapped_phase
        self.current_unfiltered_complex_trace = unfiltered_complex_trace
        self.current_filtered_complex_trace = filtered_complex_trace
        self.current_complex_trace_raw_display = filtered_complex_trace

        self.amp_trace_raw_line.set_data(self.time_axis_s, amp_trace)
        self.ax_amp_trace_raw.set_xlim(self.time_axis_s[0], self.time_axis_s[-1])
        self.ax_amp_trace_raw.set_ylim(0, 500)
        self.ax_amp_trace_raw.set_title(
            f"Amplitude trace at X={self.selected_x}, depth={self.selected_depth}"
        )

        polar_points = np.column_stack(
            [np.angle(filtered_complex_trace), np.abs(filtered_complex_trace)]
        )
        self.complex_scatter_raw.set_offsets(polar_points)
        self.ax_complex_scatter_raw.set_ylim(0, 500)
        self.ax_complex_scatter_raw.set_title(
            f"Complex trace polar plot at X={self.selected_x}, depth={self.selected_depth}"
        )

        self.phase_trace_line.set_data(self.time_axis_s, unwrapped_phase)
        self.ax_phase_trace.set_xlim(self.time_axis_s[0], self.time_axis_s[-1])
        self.ax_phase_trace.relim()
        self.ax_phase_trace.autoscale_view(scalex=False, scaley=True)
        self.ax_phase_trace.set_title(
            f"Phase trace at X={self.selected_x}, depth={self.selected_depth}"
        )

        self.g_amp_raw_line.set_data(lags, g_amp_raw)
        self.ax_g_amp_raw.set_xlim(1, max(1, self.max_autocorr_lag))
        self.ax_g_amp_raw.set_ylim(-1, 1.1)
        self.ax_g_amp_raw.set_title(
            "Amplitude autocorrelation without mean subtraction"
        )

        self.g_amp_line.set_data(lags, g_amp)
        self.ax_g_amp.set_xlim(1, max(1, self.max_autocorr_lag))
        self.ax_g_amp.set_ylim(-1, 1.1)
        self.ax_g_amp.set_title(
            "Amplitude autocorrelation with mean subtraction"
        )

        self.g1_raw_abs_line.set_data(lags, np.abs(g1_raw))
        self.ax_g1_raw.set_xlim(1, max(1, self.max_autocorr_lag))
        self.ax_g1_raw.set_ylim(0, 1.1)
        self.ax_g1_raw.set_title(
            "Complex field autocorrelation without mean subtraction"
        )

        self.g1_abs_line.set_data(lags, np.abs(g1))
        self.ax_g1.set_xlim(1, max(1, self.max_autocorr_lag))
        self.ax_g1.set_ylim(0, 1.1)
        self.ax_g1.set_title(
            "Complex field autocorrelation with mean subtraction"
        )

        self.amp_spectrum_line.set_data(frequency_hz, amp_power)
        if frequency_hz.size:
            self.ax_amp_spectrum.set_xlim(frequency_hz[0], frequency_hz[-1])
        power_ymax = np.nanmax(
            [
                np.nanmax(amp_power) if amp_power.size else 0.0,
                np.nanmax(complex_power) if complex_power.size else 0.0,
            ]
        )
        if not np.isfinite(power_ymax) or power_ymax <= 0:
            power_ymax = 1.0
        power_ymax *= 1.05
        self.ax_amp_spectrum.set_ylim(0, power_ymax)
        self.ax_amp_spectrum.set_title(
            f"Amplitude power spectrum, >{self.spectrum_min_hz:g} Hz"
        )

        self.phase_spectrum_line.set_data(complex_frequency_hz, complex_power)
        if complex_frequency_hz.size:
            self.ax_phase_spectrum.set_xlim(complex_frequency_hz[0], complex_frequency_hz[-1])
        self.ax_phase_spectrum.set_ylim(0, power_ymax)
        self.ax_phase_spectrum.set_title(
            f"Complex power spectrum, |f|>{self.spectrum_min_hz:g} Hz"
        )

        amp_mean = float(np.nanmean(amp_trace))
        amp_std = float(np.nanstd(amp_trace))
        phase_std = float(np.nanstd(unwrapped_phase - unwrapped_phase[0]))
        dz_std = float(np.nanstd(delta_z_nm))
        summary = (
            f"{self.stack_path.name}: X={self.selected_x}, depth={self.selected_depth}, "
            f"amp mean={amp_mean:.4g}, amp std={amp_std:.4g}, "
            f"phase std={phase_std:.4g} rad, delta-z std={dz_std:.4g} nm"
        )
        print(summary)
        if self.summary_text is not None:
            self.summary_text.set_text(summary.replace(", ", "\n"))

        self.fig.canvas.draw_idle()


def process_one_stack(
    path,
    frame_rate_hz,
    center_wavelength_nm,
    profile_start_depth,
    max_autocorr_lag,
    spectrum_min_hz,
    notch_band_hz,
    dynamic_uniform_filter_size,
    dynamic_chunk_x,
):
    print(f"Loading AMP+PHASE stack: {path}")
    complex_stack = read_amp_phase_tiff_stack(path)
    print(f"Reconstructed complex stack shape: {complex_stack.shape}, dtype={complex_stack.dtype}")
    viewer = AmpPhaseBlineTraceViewer(
        complex_stack=complex_stack,
        stack_path=path,
        frame_rate_hz=frame_rate_hz,
        center_wavelength_nm=center_wavelength_nm,
        profile_start_depth=profile_start_depth,
        max_autocorr_lag=max_autocorr_lag,
        spectrum_min_hz=spectrum_min_hz,
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
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--lambda0-nm", type=float, default=DEFAULT_CENTER_WAVELENGTH_NM)
    parser.add_argument("--profile-start-depth", type=int, default=DEFAULT_PROFILE_START_DEPTH)
    parser.add_argument("--max-autocorr-lag", type=int, default=DEFAULT_MAX_AUTOCORR_LAG)
    parser.add_argument("--spectrum-min-hz", type=float, default=DEFAULT_SPECTRUM_MIN_HZ)
    parser.add_argument("--notch-low-hz", type=float, default=DEFAULT_NOTCH_BAND_HZ[0])
    parser.add_argument("--notch-high-hz", type=float, default=DEFAULT_NOTCH_BAND_HZ[1])
    parser.add_argument(
        "--dynamic-uniform-filter-size",
        type=int,
        default=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    )
    parser.add_argument("--dynamic-chunk-x", type=int, default=DEFAULT_DYNAMIC_CHUNK_X)
    return parser.parse_known_args()[0]


def run_viewer(
    input_path=DEFAULT_INPUT_DIR,
    frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
    center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
    profile_start_depth=DEFAULT_PROFILE_START_DEPTH,
    max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
    spectrum_min_hz=DEFAULT_SPECTRUM_MIN_HZ,
    notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
    dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
    dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
):
    paths = iter_tiff_stacks(input_path)
    print(f"Found {len(paths)} TIFF stack(s). Close each figure to load the next stack.")
    for path in paths:
        process_one_stack(
            path=path,
            frame_rate_hz=frame_rate_hz,
            center_wavelength_nm=center_wavelength_nm,
            profile_start_depth=profile_start_depth,
            max_autocorr_lag=max_autocorr_lag,
            spectrum_min_hz=spectrum_min_hz,
            notch_band_hz=notch_band_hz,
            dynamic_uniform_filter_size=dynamic_uniform_filter_size,
            dynamic_chunk_x=dynamic_chunk_x,
        )


def main():
    if USE_COMMAND_LINE_ARGS:
        args = parse_args()
        run_viewer(
            input_path=args.input,
            frame_rate_hz=args.frame_rate,
            center_wavelength_nm=args.lambda0_nm,
            profile_start_depth=args.profile_start_depth,
            max_autocorr_lag=args.max_autocorr_lag,
            spectrum_min_hz=args.spectrum_min_hz,
            notch_band_hz=(args.notch_low_hz, args.notch_high_hz),
            dynamic_uniform_filter_size=args.dynamic_uniform_filter_size,
            dynamic_chunk_x=args.dynamic_chunk_x,
        )
        return

    run_viewer(
        input_path=DEFAULT_INPUT_DIR,
        frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
        center_wavelength_nm=DEFAULT_CENTER_WAVELENGTH_NM,
        profile_start_depth=DEFAULT_PROFILE_START_DEPTH,
        max_autocorr_lag=DEFAULT_MAX_AUTOCORR_LAG,
        spectrum_min_hz=DEFAULT_SPECTRUM_MIN_HZ,
        notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
        dynamic_uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        dynamic_chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    )


if __name__ == "__main__":
    main()
