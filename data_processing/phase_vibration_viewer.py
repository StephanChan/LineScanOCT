import os
import csv

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from matplotlib.widgets import Slider
import tifffile as TIFF


SUPPORTED_STACK_EXTENSIONS = {".npy", ".tif", ".tiff"}


def iter_stack_files(input_dir):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path must be a directory, got: {input_dir}")

    stack_paths = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_STACK_EXTENSIONS
    ]
    stack_paths.sort()
    if not stack_paths:
        raise ValueError(f"No .npy, .tif, or .tiff files found in: {input_dir}")
    return stack_paths


def load_stack(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        stack = np.load(path)
    elif ext in {".tif", ".tiff"}:
        with TIFF.TiffFile(path) as tif:
            stack = np.stack([page.asarray() for page in tif.pages], axis=0)
            print(f"Stacked shape: {stack.shape}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got {stack.shape}")
    return np.float32(stack)


def prepare_depth_stack(
    raw_stack,
    use_hanning=False,
    spectral_highpass_size=51,
    fft_amplification=100.0,
):
    spectra = np.asarray(raw_stack, dtype=np.float32)
    frames, x_pixels, samples = spectra.shape
    window = None
    if use_hanning:
        window = np.hanning(samples).astype(np.float32)

    depth_complex = np.empty((frames, x_pixels, samples), dtype=np.complex64)
    for frame_idx in range(frames):
        frame_spectra = spectra[frame_idx]
        if spectral_highpass_size and spectral_highpass_size > 1:
            baseline = uniform_filter1d(
                frame_spectra,
                size=int(spectral_highpass_size),
                axis=1,
            )
            frame_spectra = frame_spectra - baseline

        if window is not None:
            frame_spectra = frame_spectra * window[np.newaxis, :]

        depth_complex[frame_idx] = (
            np.fft.fft(frame_spectra, axis=1) / samples
        ).astype(np.complex64, copy=False)

    depth_complex *= np.float32(fft_amplification)
    depth_magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    return depth_complex, depth_magnitude


def phase_to_delta_z(phase_trace, center_wavelength_m):
    unwrapped = np.unwrap(phase_trace)
    relative_phase = unwrapped - unwrapped[0]
    delta_z_m = relative_phase * center_wavelength_m / (2.0 * np.pi)
    return delta_z_m.astype(np.float32, copy=False), unwrapped.astype(np.float32, copy=False)


def compute_depth_displacement_map(depth_complex, depth_index, center_wavelength_m):
    complex_plane = depth_complex[:, :, depth_index]
    phase_plane = np.unwrap(np.angle(complex_plane), axis=0)
    relative_phase = phase_plane - phase_plane[0:1, :]
    delta_z_m = relative_phase * center_wavelength_m / (2.0 * np.pi)
    return np.asarray(delta_z_m, dtype=np.float32)


def highpass_filter(data, frame_rate_hz, cutoff_hz, order=3, axis=0):
    data = np.asarray(data, dtype=np.float32)
    cutoff_hz = float(cutoff_hz)
    frame_rate_hz = float(frame_rate_hz)

    if cutoff_hz <= 0:
        return data - np.mean(data, axis=axis, keepdims=True)

    nyquist_hz = frame_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"High-pass cutoff must be below Nyquist frequency "
            f"({nyquist_hz:g} Hz), got {cutoff_hz:g} Hz"
        )

    samples = data.shape[axis]
    padlen = 3 * (2 * order + 1)
    if samples <= padlen:
        print(
            f"Only {samples} samples available; subtracting the mean instead "
            f"of applying a {order}th-order high-pass filter."
        )
        return data - np.mean(data, axis=axis, keepdims=True)

    sos = butter(order, cutoff_hz / nyquist_hz, btype="highpass", output="sos")
    filtered = sosfiltfilt(sos, data, axis=axis)
    return np.asarray(filtered, dtype=np.float32)


def compute_filtered_outputs(
    depth_complex,
    selected_x,
    selected_depth,
    center_wavelength_m,
    frame_rate_hz,
    highpass_cutoff_hz,
    highpass_order,
):
    complex_trace = depth_complex[:, selected_x, selected_depth]
    phase_trace = np.angle(complex_trace)
    delta_z_trace_m, _ = phase_to_delta_z(phase_trace, center_wavelength_m)
    delta_z_trace_nm = delta_z_trace_m * 1e9
    filtered_trace_nm = highpass_filter(
        delta_z_trace_nm,
        frame_rate_hz=frame_rate_hz,
        cutoff_hz=highpass_cutoff_hz,
        order=highpass_order,
        axis=0,
    )

    delta_z_map_m = compute_depth_displacement_map(
        depth_complex,
        selected_depth,
        center_wavelength_m,
    )
    delta_z_map_nm = delta_z_map_m * 1e9
    filtered_map_nm = highpass_filter(
        delta_z_map_nm,
        frame_rate_hz=frame_rate_hz,
        cutoff_hz=highpass_cutoff_hz,
        order=highpass_order,
        axis=0,
    )
    return filtered_trace_nm, filtered_map_nm


def save_filtered_vibration_figure(
    stack_path,
    output_dir,
    time_axis_s,
    filtered_trace_nm,
    filtered_map_nm,
    selected_x,
    selected_depth,
    highpass_cutoff_hz,
    vibration_limit_nm,
):
    base_name = os.path.splitext(os.path.basename(stack_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_filtered_phase_vibration.png")
    trace_std_nm = float(np.nanstd(filtered_trace_nm))

    fig, (ax_trace, ax_map) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax_trace.plot(time_axis_s, filtered_trace_nm, lw=1.2)
    ax_trace.set_title(
        f"Filtered vibration at X={selected_x}, depth={selected_depth}\n"
        f"STD = {trace_std_nm:.3f} nm"
    )
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Delta z (nm)")
    ax_trace.set_ylim(-vibration_limit_nm, vibration_limit_nm)
    ax_trace.grid(True, alpha=0.25)

    im = ax_map.imshow(
        filtered_map_nm,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vibration_limit_nm,
        vmax=vibration_limit_nm,
        extent=[0, filtered_map_nm.shape[1] - 1, time_axis_s[0], time_axis_s[-1]],
    )
    ax_map.set_title(
        f"Filtered vibration map at depth={selected_depth}\n"
        f"High-pass cutoff = {highpass_cutoff_hz:g} Hz"
    )
    ax_map.set_xlabel("X pixel")
    ax_map.set_ylabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("Delta z (nm)")

    fig.suptitle(os.path.basename(stack_path))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, trace_std_nm


def process_stack_file(
    stack_path,
    output_dir,
    selected_x,
    selected_depth,
    center_wavelength_m,
    frame_rate_hz,
    use_hanning,
    spectral_highpass_size,
    fft_amplification,
    highpass_cutoff_hz,
    highpass_order,
    vibration_limit_nm,
):
    raw_stack = load_stack(stack_path)
    depth_complex, depth_magnitude = prepare_depth_stack(
        raw_stack,
        use_hanning=use_hanning,
        spectral_highpass_size=spectral_highpass_size,
        fft_amplification=fft_amplification,
    )
    frames, x_pixels, depth_pixels = depth_magnitude.shape
    if not (0 <= selected_x < x_pixels):
        raise ValueError(
            f"Selected X index {selected_x} is outside {stack_path} X range 0-{x_pixels - 1}"
        )
    if not (0 <= selected_depth < depth_pixels):
        raise ValueError(
            f"Selected depth index {selected_depth} is outside {stack_path} "
            f"depth range 0-{depth_pixels - 1}"
        )

    filtered_trace_nm, filtered_map_nm = compute_filtered_outputs(
        depth_complex=depth_complex,
        selected_x=selected_x,
        selected_depth=selected_depth,
        center_wavelength_m=center_wavelength_m,
        frame_rate_hz=frame_rate_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
        highpass_order=highpass_order,
    )
    time_axis_s = np.arange(frames, dtype=np.float32) / frame_rate_hz
    return save_filtered_vibration_figure(
        stack_path=stack_path,
        output_dir=output_dir,
        time_axis_s=time_axis_s,
        filtered_trace_nm=filtered_trace_nm,
        filtered_map_nm=filtered_map_nm,
        selected_x=selected_x,
        selected_depth=selected_depth,
        highpass_cutoff_hz=highpass_cutoff_hz,
        vibration_limit_nm=vibration_limit_nm,
    )


class PhaseVibrationViewer:
    def __init__(
        self,
        raw_stack,
        center_wavelength_m,
        frame_rate_hz,
        use_hanning=False,
        spectral_highpass_size=51,
        fft_amplification=100.0,
        vibration_limit_nm=None,
        profile_start_depth=10,
    ):
        self.raw_stack = raw_stack
        self.center_wavelength_m = float(center_wavelength_m)
        self.frame_rate_hz = float(frame_rate_hz)
        self.vibration_limit_nm = vibration_limit_nm
        self.profile_start_depth = int(profile_start_depth)
        self.depth_complex, self.depth_magnitude = prepare_depth_stack(
            raw_stack,
            use_hanning=use_hanning,
            spectral_highpass_size=spectral_highpass_size,
            fft_amplification=fft_amplification,
        )

        self.frames, self.x_pixels, self.depth_pixels = self.depth_magnitude.shape
        self.profile_start_depth = int(np.clip(self.profile_start_depth, 0, self.depth_pixels - 1))
        self.time_axis_s = np.arange(self.frames, dtype=np.float32) / self.frame_rate_hz
        self.mean_depth_image = np.mean(self.depth_magnitude, axis=0)
        self.mean_depth_profile_by_x = self.mean_depth_image

        self.selected_x = self.x_pixels // 2
        profile = self.mean_depth_profile_by_x[self.selected_x, self.profile_start_depth:]
        self.selected_depth = (
            int(np.argmax(profile) + self.profile_start_depth)
            if profile.size
            else self.profile_start_depth
        )

        self.fig = None
        self.ax_depth_image = None
        self.ax_profile = None
        self.ax_trace = None
        self.ax_map = None
        self.im_depth = None
        self.profile_line = None
        self.trace_line = None
        self.im_map = None
        self.cursor_image = None
        self.cursor_profile = None
        self.brightness_slider = None
        self.depth_image_vmin = None
        self.depth_image_vmax = None

    def build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        grid = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.2, 1.0])

        self.ax_depth_image = self.fig.add_subplot(grid[0, 0])
        self.ax_profile = self.fig.add_subplot(grid[0, 1])
        self.ax_trace = self.fig.add_subplot(grid[1, 0])
        self.ax_map = self.fig.add_subplot(grid[1, 1])

        self.depth_image_vmin, self.depth_image_vmax = self._initial_depth_image_clim()
        self.im_depth = self.ax_depth_image.imshow(
            self.mean_depth_image.T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            vmin=self.depth_image_vmin,
            vmax=self.depth_image_vmax,
        )
        self.ax_depth_image.set_title("Mean depth magnitude image (click to choose X and depth)")
        self.ax_depth_image.set_xlabel("X pixel")
        self.ax_depth_image.set_ylabel("Depth index")

        profile_depths = np.arange(self.profile_start_depth, self.depth_pixels)
        self.profile_line, = self.ax_profile.plot(
            self.mean_depth_profile_by_x[self.selected_x, self.profile_start_depth:],
            profile_depths,
            lw=1.5,
        )
        self.ax_profile.set_title(
            f"Depth profile at selected X, from depth {self.profile_start_depth}"
        )
        self.ax_profile.set_xlabel("Magnitude")
        self.ax_profile.set_ylabel("Depth index")

        self.cursor_image = self.ax_depth_image.plot(
            [self.selected_x],
            [self.selected_depth],
            marker="+",
            markersize=14,
            color="red",
            linestyle="None",
        )[0]
        self.cursor_profile = self.ax_profile.axhline(self.selected_depth, color="red", lw=1.0)

        self.trace_line, = self.ax_trace.plot([], [], lw=1.5)
        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.set_ylabel("Delta z")
        self.ax_trace.grid(True, alpha=0.25)

        self.im_map = self.ax_map.imshow(
            np.zeros((self.frames, self.x_pixels), dtype=np.float32),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )
        self.ax_map.set_xlabel("X pixel")
        self.ax_map.set_ylabel("Frame index")
        self.ax_map.set_title("Delta z map at selected depth")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.add_brightness_slider()
        self.update_views()
        self.fig.tight_layout(rect=[0, 0.06, 1, 1])

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
        slider_ax = self.fig.add_axes([0.12, 0.02, 0.34, 0.025])
        self.brightness_slider = Slider(
            ax=slider_ax,
            label="Depth image brightness",
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
        if event.inaxes == self.ax_depth_image:
            if event.xdata is None or event.ydata is None:
                return
            self.selected_x = int(np.clip(round(event.xdata), 0, self.x_pixels - 1))
            self.selected_depth = int(np.clip(round(event.ydata), 0, self.depth_pixels - 1))
            self.update_views()
        elif event.inaxes == self.ax_profile:
            if event.ydata is None:
                return
            self.selected_depth = int(np.clip(round(event.ydata), 0, self.depth_pixels - 1))
            self.update_views()

    def update_views(self):
        profile = self.mean_depth_profile_by_x[self.selected_x, self.profile_start_depth:]
        profile_depths = np.arange(self.profile_start_depth, self.depth_pixels)
        self.profile_line.set_xdata(profile)
        self.profile_line.set_ydata(profile_depths)
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()

        self.cursor_image.set_data([self.selected_x], [self.selected_depth])
        self.cursor_profile.set_ydata([self.selected_depth, self.selected_depth])

        complex_trace = self.depth_complex[:, self.selected_x, self.selected_depth]
        phase_trace = np.angle(complex_trace)
        delta_z_m, _ = phase_to_delta_z(phase_trace, self.center_wavelength_m)
        delta_z_nm = delta_z_m * 1e9

        self.trace_line.set_data(self.time_axis_s, delta_z_nm)
        self.ax_trace.relim()
        self.ax_trace.autoscale_view()
        self.ax_trace.set_title(
            f"Delta z(t) at X={self.selected_x}, depth={self.selected_depth}"
        )
        self.ax_trace.set_ylabel("Delta z (nm)")

        delta_z_map_m = compute_depth_displacement_map(
            self.depth_complex,
            self.selected_depth,
            self.center_wavelength_m,
        )
        delta_z_map_nm = delta_z_map_m * 1e9
        self.im_map.set_data(delta_z_map_nm)

        if self.vibration_limit_nm is None:
            vmax = float(np.nanmax(np.abs(delta_z_map_nm))) if delta_z_map_nm.size else 1.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        else:
            vmax = float(self.vibration_limit_nm)
        self.im_map.set_clim(-vmax, vmax)
        self.ax_map.set_title(
            f"Delta z(x, t) at depth={self.selected_depth} (nm)"
        )

        self.fig.canvas.draw_idle()

    def show(self):
        self.build_figure()
        plt.show(block=True)
        plt.close(self.fig)


def main():
    input_dir = (
        r"E:\IOCTData\vibration charac\5mm glass base\vibration measurement"
        r"\Table float pump on closure open\closure closed"
    )
    output_dir = None  # None creates "phase_vibration_results" inside input_dir.
    lambda0_nm = 840.0
    frame_rate_hz = 50.0
    use_hanning = False
    spectral_highpass_size = 51
    fft_amplification = 100.0
    highpass_cutoff_hz = 1.0
    highpass_order = 3
    vibration_limit_nm = 15.0

    stack_paths = iter_stack_files(input_dir)
    if output_dir is None:
        output_dir = os.path.join(input_dir, "phase_vibration_results")
    os.makedirs(output_dir, exist_ok=True)

    first_stack_path = stack_paths[0]
    print(f"Pick reflector from first file, then close the figure: {first_stack_path}")
    raw_stack = load_stack(first_stack_path)
    viewer = PhaseVibrationViewer(
        raw_stack=raw_stack,
        center_wavelength_m=lambda0_nm * 1e-9,
        frame_rate_hz=frame_rate_hz,
        use_hanning=use_hanning,
        spectral_highpass_size=spectral_highpass_size,
        fft_amplification=fft_amplification,
        vibration_limit_nm=vibration_limit_nm,
        profile_start_depth=10,
    )
    viewer.show()

    selected_x = viewer.selected_x
    selected_depth = viewer.selected_depth
    print(f"Using selected X={selected_x}, depth={selected_depth} for {len(stack_paths)} files.")

    summary_rows = []
    for stack_path in stack_paths:
        print(f"Processing: {stack_path}")
        output_path, trace_std_nm = process_stack_file(
            stack_path=stack_path,
            output_dir=output_dir,
            selected_x=selected_x,
            selected_depth=selected_depth,
            center_wavelength_m=lambda0_nm * 1e-9,
            frame_rate_hz=frame_rate_hz,
            use_hanning=use_hanning,
            spectral_highpass_size=spectral_highpass_size,
            fft_amplification=fft_amplification,
            highpass_cutoff_hz=highpass_cutoff_hz,
            highpass_order=highpass_order,
            vibration_limit_nm=vibration_limit_nm,
        )
        print(f"Saved: {output_path} | filtered STD = {trace_std_nm:.3f} nm")
        summary_rows.append(
            {
                "file": os.path.basename(stack_path),
                "selected_x": selected_x,
                "selected_depth": selected_depth,
                "highpass_cutoff_hz": highpass_cutoff_hz,
                "filtered_trace_std_nm": trace_std_nm,
                "figure": os.path.basename(output_path),
            }
        )

    summary_path = os.path.join(output_dir, "filtered_phase_vibration_summary.csv")
    with open(summary_path, "w", newline="") as summary_file:
        fieldnames = [
            "file",
            "selected_x",
            "selected_depth",
            "highpass_cutoff_hz",
            "filtered_trace_std_nm",
            "figure",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
