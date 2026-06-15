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
DEFAULT_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\AMP Phase"
DEFAULT_FRAME_RATE_HZ = 200.0
DEFAULT_NOTCH_BAND_HZ = (46.0, 48.0)
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 10
DEFAULT_DYNAMIC_CHUNK_X = 96
DEFAULT_POLAR_RADIUS_LIMIT = 500.0

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
    return (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)


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


def compute_dynamic_images(complex_stack, uniform_filter_size, chunk_x):
    _, x_pixels, z_pixels = complex_stack.shape
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
            uniform_filter_size,
        )
        complex_dynamic[x0:x1, :] = complex_dynamic_from_filtered_amp_phase(
            amplitude,
            phase,
            uniform_filter_size,
        )
        print(f"Dynamic maps: processed X {x0}-{x1} / {x_pixels}")

    return amplitude_dynamic, complex_dynamic


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


class CompactAmpPhaseDynamicViewer:
    def __init__(self, complex_stack, stack_path):
        self.stack_path = Path(stack_path)
        self.filtered_complex_stack = notch_filter_complex_stack_amp_phase(
            complex_stack,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            notch_band_hz=DEFAULT_NOTCH_BAND_HZ,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )
        self.mean_abs_bline = np.mean(
            np.abs(self.filtered_complex_stack),
            axis=0,
            dtype=np.float32,
        )
        self.amplitude_dynamic, self.complex_dynamic = compute_dynamic_images(
            self.filtered_complex_stack,
            uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
            chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
        )
        self.frames, self.x_pixels, self.z_pixels = self.filtered_complex_stack.shape
        self.selected_x = self.x_pixels // 2
        self.selected_depth = int(np.argmax(self.mean_abs_bline[self.selected_x, :]))

        self.fig = None
        self.ax_mean = None
        self.ax_amp_dyn = None
        self.ax_complex_dyn = None
        self.ax_polar = None
        self.cursor_mean = None
        self.cursor_amp = None
        self.cursor_complex = None
        self.polar_scatter = None
        self.mean_vmin = None
        self.mean_vmax = None
        self.brightness_slider = None

    def show(self):
        self.build_figure()
        plt.show(block=True)
        plt.close(self.fig)

    def build_figure(self):
        self.fig = plt.figure(figsize=(12, 8.5))
        grid = self.fig.add_gridspec(2, 2)
        self.ax_mean = self.fig.add_subplot(grid[0, 0])
        self.ax_amp_dyn = self.fig.add_subplot(grid[1, 0])
        self.ax_complex_dyn = self.fig.add_subplot(grid[0, 1])
        self.ax_polar = self.fig.add_subplot(grid[1, 1], projection="polar")

        self.mean_vmin, self.mean_vmax = initial_image_clim(self.mean_abs_bline)
        im_mean = self.ax_mean.imshow(
            self.mean_abs_bline.T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            vmin=self.mean_vmin,
            vmax=self.mean_vmax,
        )
        self.ax_mean.set_title("Mean abs B-line")
        self.ax_mean.set_xlabel("X pixel")
        self.ax_mean.set_ylabel("Depth index")
        self.fig.colorbar(im_mean, ax=self.ax_mean, label="Mean amplitude")

        amp_vmin, amp_vmax = initial_image_clim(self.amplitude_dynamic)
        im_amp = self.ax_amp_dyn.imshow(
            self.amplitude_dynamic.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=amp_vmin,
            vmax=amp_vmax,
        )
        self.ax_amp_dyn.set_title("Amplitude dynamic signal")
        self.ax_amp_dyn.set_xlabel("X pixel")
        self.ax_amp_dyn.set_ylabel("Depth index")
        self.fig.colorbar(im_amp, ax=self.ax_amp_dyn, label="Variance")

        complex_vmin, complex_vmax = initial_image_clim(self.complex_dynamic)
        im_complex = self.ax_complex_dyn.imshow(
            self.complex_dynamic.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=complex_vmin,
            vmax=complex_vmax,
        )
        self.ax_complex_dyn.set_title("Complex dynamic signal")
        self.ax_complex_dyn.set_xlabel("X pixel")
        self.ax_complex_dyn.set_ylabel("Depth index")
        self.fig.colorbar(im_complex, ax=self.ax_complex_dyn, label="Abs variance")

        self.cursor_mean = self.ax_mean.plot([], [], marker="+", color="red", markersize=14, linestyle="None")[0]
        self.cursor_amp = self.ax_amp_dyn.plot([], [], marker="+", color="cyan", markersize=14, linestyle="None")[0]
        self.cursor_complex = self.ax_complex_dyn.plot([], [], marker="+", color="cyan", markersize=14, linestyle="None")[0]
        self.polar_scatter = self.ax_polar.scatter([], [], s=5, c="black", alpha=0.55, edgecolors="none")
        self.ax_polar.set_ylim(0, DEFAULT_POLAR_RADIUS_LIMIT)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.add_brightness_slider()
        self.fig.suptitle(self.stack_path.name)
        self.update_selection()
        self.fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    def add_brightness_slider(self):
        slider_ax = self.fig.add_axes([0.12, 0.02, 0.34, 0.025])
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
        display_vmax = self.mean_vmin + (self.mean_vmax - self.mean_vmin) / brightness
        self.ax_mean.images[0].set_clim(self.mean_vmin, display_vmax)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in (self.ax_mean, self.ax_amp_dyn, self.ax_complex_dyn):
            return
        if event.xdata is None or event.ydata is None:
            return
        self.selected_x = int(np.clip(round(event.xdata), 0, self.x_pixels - 1))
        self.selected_depth = int(np.clip(round(event.ydata), 0, self.z_pixels - 1))
        self.update_selection()

    def update_selection(self):
        self.cursor_mean.set_data([self.selected_x], [self.selected_depth])
        self.cursor_amp.set_data([self.selected_x], [self.selected_depth])
        self.cursor_complex.set_data([self.selected_x], [self.selected_depth])

        trace = self.filtered_complex_stack[:, self.selected_x, self.selected_depth]
        polar_points = np.column_stack([np.angle(trace), np.abs(trace)])
        self.polar_scatter.set_offsets(polar_points)
        self.ax_polar.set_title(f"Complex polar trace: X={self.selected_x}, depth={self.selected_depth}")
        print(f"{self.stack_path.name}: X={self.selected_x}, depth={self.selected_depth}")
        self.fig.canvas.draw_idle()


def process_stack(stack_path):
    print(f"Loading AMP+PHASE stack: {stack_path}")
    complex_stack = read_amp_phase_tiff_stack(stack_path)
    viewer = CompactAmpPhaseDynamicViewer(complex_stack, stack_path)
    viewer.show()
    del viewer
    del complex_stack
    gc.collect()


def main():
    stack_paths = iter_tiff_stacks(DEFAULT_INPUT_PATH)
    print(f"Found {len(stack_paths)} TIFF stack(s). Close each figure to load the next stack.")
    for stack_path in stack_paths:
        process_stack(stack_path)


if __name__ == "__main__":
    main()
