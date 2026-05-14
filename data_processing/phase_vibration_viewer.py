import os
# import matplotlib
# matplotlib.use('Qt5Agg')   # 或 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import tifffile as TIFF


def load_stack(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        stack = np.load(path)
    elif ext in {".tif", ".tiff"}:
        with TIFF.TiffFile(path) as tif:
            pages = tif.pages
            # 堆叠所有页 -> shape (500, 1104, 1104)
            stack = np.stack([page.asarray() for page in pages], axis=0)
            print(f"Stacked shape: {stack.shape}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    stack = np.asarray(stack)
    # 后续处理保持不变（2D 单帧或 3D 多帧）
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]   # 单帧
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got {stack.shape}")
    return np.float32(stack)


def prepare_depth_stack(raw_stack, background=None, use_hanning=True):
    spectra = np.asarray(raw_stack, dtype=np.float32)

    if background is not None:
        background = np.asarray(background, dtype=np.float32)
        if background.shape != (spectra.shape[2],):
            raise ValueError(
                "Background must have shape (spectral_pixels,), "
                f"got {background.shape} for spectral size {spectra.shape[2]}"
            )
        spectra = spectra - background[np.newaxis, np.newaxis, :]
    else:
        spectra = spectra - np.mean(spectra, axis=2, keepdims=True)

    if use_hanning:
        window = np.hanning(spectra.shape[2]).astype(np.float32)
        spectra = spectra * window[np.newaxis, np.newaxis, :]

    depth_complex = np.empty_like(spectra, dtype=np.complex64)
    for i in range(spectra.shape[0]):
        depth_complex[i] = np.fft.fft(spectra[i], axis=1)
    half = depth_complex.shape[2] // 2
    depth_complex = depth_complex[:, :, :half]
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


class PhaseVibrationViewer:
    def __init__(self, raw_stack, center_wavelength_m, frame_rate_hz, background=None, use_hanning=True):
        self.raw_stack = raw_stack
        self.center_wavelength_m = float(center_wavelength_m)
        self.frame_rate_hz = float(frame_rate_hz)
        self.depth_complex, self.depth_magnitude = prepare_depth_stack(
            raw_stack,
            background=background,
            use_hanning=use_hanning,
        )

        self.frames, self.x_pixels, self.depth_pixels = self.depth_magnitude.shape
        self.time_axis_s = np.arange(self.frames, dtype=np.float32) / self.frame_rate_hz
        self.mean_depth_image = np.mean(self.depth_magnitude, axis=0)
        self.mean_depth_profile_by_x = self.mean_depth_image

        self.selected_x = self.x_pixels // 2
        profile = self.mean_depth_profile_by_x[self.selected_x]
        self.selected_depth = int(np.argmax(profile[1:]) + 1) if profile.size > 1 else 0

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

    def build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        grid = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.2, 1.0])

        self.ax_depth_image = self.fig.add_subplot(grid[0, 0])
        self.ax_profile = self.fig.add_subplot(grid[0, 1])
        self.ax_trace = self.fig.add_subplot(grid[1, 0])
        self.ax_map = self.fig.add_subplot(grid[1, 1])

        self.im_depth = self.ax_depth_image.imshow(
            self.mean_depth_image.T,
            aspect="auto",
            origin="lower",
            cmap="gray",
        )
        self.ax_depth_image.set_title("Mean depth magnitude image (click to choose X and depth)")
        self.ax_depth_image.set_xlabel("X pixel")
        self.ax_depth_image.set_ylabel("Depth index")

        self.profile_line, = self.ax_profile.plot(
            self.mean_depth_profile_by_x[self.selected_x],
            np.arange(self.depth_pixels),
            lw=1.5,
        )
        self.ax_profile.set_title("Depth profile at selected X (click to choose depth)")
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
        self.update_views()
        self.fig.tight_layout()

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
        profile = self.mean_depth_profile_by_x[self.selected_x]
        self.profile_line.set_xdata(profile)
        self.profile_line.set_ydata(np.arange(self.depth_pixels))
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

        vmax = float(np.nanmax(np.abs(delta_z_map_nm))) if delta_z_map_nm.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        self.im_map.set_clim(-vmax, vmax)
        self.ax_map.set_title(
            f"Delta z(x, t) at depth={self.selected_depth} (nm)"
        )

        self.fig.canvas.draw_idle()

    def show(self):
        self.build_figure()
        plt.show()


def main():
    # Set input parameters as variables
    input_path = r"E:\IOCTData\vibration charac\5mm glass base\vibration measurement\Table float pump on closure open\closure closed\Bline-10-Yrpt500-X1264-Z1104.tif"  # Path to a .npy or .tif/.tiff stack shaped (frames, x_pixels, spectral_pixels)
    lambda0_nm = 840.0  # Center wavelength in nm
    frame_rate_hz = 50.0  # Frame rate in Hz
    background_path = None  # Optional .npy background spectrum shaped (spectral_pixels,), set to None if not used
    use_hanning = True  # Set to False to disable Hanning window before FFT

    raw_stack = load_stack(input_path)
    background = None
    if background_path:
        background = np.load(background_path)

    viewer = PhaseVibrationViewer(
        raw_stack=raw_stack,
        center_wavelength_m=lambda0_nm * 1e-9,
        frame_rate_hz=frame_rate_hz,
        background=background,
        use_hanning=use_hanning,
    )
    # ========== 手动指定 X 和 Depth 索引 ==========
    viewer.selected_x = 700          # 例如第 300 个 X 像素（0‑based）
    viewer.selected_depth = 58      # 例如第 150 个深度层
    # ===========================================
    viewer.show()
    
    # ========== 手动指定 X 和 Depth 索引 ==========
    viewer.selected_x = 700          # 例如第 300 个 X 像素（0‑based）
    viewer.selected_depth = 124      # 例如第 150 个深度层
    # ===========================================
    viewer.show()


if __name__ == "__main__":
    main()
