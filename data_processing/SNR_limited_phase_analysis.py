import os

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.ndimage import uniform_filter1d
import tifffile as TIFF


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
    half_depth = samples // 2
    window = None
    if use_hanning:
        window = np.hanning(samples).astype(np.float32)

    depth_complex = np.empty((frames, x_pixels, half_depth), dtype=np.complex64)
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
            np.fft.fft(frame_spectra, axis=1)[:, :half_depth] / samples
        ).astype(np.complex64, copy=False)

    depth_complex *= np.float32(fft_amplification)
    depth_magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    return depth_complex, depth_magnitude


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
        self.ax_image.set_title("Click noise-region starting depth, then close the window")
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


def calculate_phase_variance_and_snr(depth_complex, noise_start_depth, analysis_start_depth=10):
    magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    frames, x_pixels, depth_pixels = magnitude.shape
    analysis_start_depth = int(np.clip(analysis_start_depth, 0, depth_pixels - 1))
    noise_start_depth = int(np.clip(noise_start_depth, 0, depth_pixels - 1))
    if noise_start_depth < analysis_start_depth:
        noise_start_depth = analysis_start_depth
    analysis_stop_depth = noise_start_depth
    if analysis_stop_depth <= analysis_start_depth:
        raise ValueError(
            "Signal analysis region is empty. Select a noise start depth deeper than "
            f"{analysis_start_depth}."
        )
    noise_region = magnitude[:, :, noise_start_depth:]
    if noise_region.shape[2] < 2:
        raise ValueError("Noise region is too shallow. Select a smaller starting depth.")

    noise_level = float(np.mean(np.std(noise_region, axis=0)))
    if not np.isfinite(noise_level) or noise_level <= 0:
        raise ValueError(f"Invalid noise level calculated from selected region: {noise_level}")

    mean_intensity = np.mean(magnitude, axis=0, dtype=np.float32)
    pixel_snr_linear = mean_intensity / np.float32(noise_level)
    pixel_snr_db = 20.0 * np.log10(pixel_snr_linear)

    phase = np.angle(depth_complex)
    phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
    phase_variance = np.var(phase, axis=0, dtype=np.float32)

    signal_phase_variance = phase_variance[:, analysis_start_depth:analysis_stop_depth]

    snr_flat = pixel_snr_db[:, analysis_start_depth:analysis_stop_depth].reshape(-1)
    phase_variance_flat = signal_phase_variance.reshape(-1)
    valid = (
        np.isfinite(snr_flat)
        & np.isfinite(phase_variance_flat)
        & (phase_variance_flat > 0)
    )
    if not np.any(valid):
        raise ValueError("No valid positive SNR/phase-variance pixels were found.")
    phase_variance_valid = phase_variance_flat[valid]
    noise_floor_percentile = 0.1
    noise_floor_phase_variance = float(np.nanpercentile(phase_variance_valid, noise_floor_percentile))
    return snr_flat[valid], phase_variance_valid, noise_level, noise_floor_phase_variance


def plot_phase_variance_vs_snr(
    snr,
    phase_variance,
    mean_depth_image,
    noise_level,
    noise_floor_phase_variance,
    center_wavelength_m,
    noise_start_depth,
    analysis_start_depth,
    bline_clim,
    stack_path,
    output_path,
):
    font_sizes = {
        "title": 18,
        "label": 16,
        "tick": 13,
        "legend": 13,
        "annotation": 13,
        "colorbar": 14,
    }
    fig, (ax, ax_bline) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    ax.scatter(snr, phase_variance, s=8, alpha=0.2, edgecolors="none", label="Pixels")

    snr_min = float(np.nanmin(snr))
    snr_max = float(np.nanmax(snr))
    if not np.isfinite(snr_min) or not np.isfinite(snr_max) or snr_max <= snr_min:
        raise ValueError("SNR values do not span a valid range for plotting.")
    snr_plot_max = max(snr_max, 70.0)
    snr_curve_min = max(0.0, snr_min)
    snr_curve_db = np.linspace(snr_curve_min, snr_plot_max, 300)
    snr_amplitude_ratio = 10.0 ** (snr_curve_db / 20.0)
    snr_power_linear = snr_amplitude_ratio ** 2
    ax.plot(
        snr_curve_db,
        1.0 / (2.0 * snr_power_linear),
        color="red",
        lw=3.0,
        label="1 / (2 SNR)",
    )
    if np.isfinite(noise_floor_phase_variance) and noise_floor_phase_variance > 0:
        ax.axhline(
            noise_floor_phase_variance,
            color="black",
            lw=2.2,
            ls="--",
            label="Phase noise floor",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Pixel SNR (dB, 20 log10(signal/noise))", fontsize=font_sizes["label"])
    ax.set_ylabel("Phase variance (rad^2)", fontsize=font_sizes["label"])
    ax.set_title("Phase Variance vs Pixel SNR", fontsize=font_sizes["title"])
    ax.set_xlim(snr_min, snr_plot_max)
    ax.tick_params(axis="both", labelsize=font_sizes["tick"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=font_sizes["legend"])

    phase_floor_rad = np.sqrt(noise_floor_phase_variance)
    axial_floor_nm = phase_floor_rad * center_wavelength_m / (2.0 * np.pi) * 1e9
    info_text = (
        f"noise level = {noise_level:.4g}\n"
        f"phase floor = {phase_floor_rad:.4g} rad\n"
        f"corresponding axial displacement precision = {axial_floor_nm:.4g} nm"
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=font_sizes["annotation"],
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    im = ax_bline.imshow(
        mean_depth_image.T,
        aspect="auto",
        origin="upper",
        cmap="gray",
        vmin=bline_clim[0],
        vmax=bline_clim[1],
    )
    ax_bline.axhline(analysis_start_depth, color="cyan", lw=1.8, label="Analysis start")
    ax_bline.axhline(noise_start_depth, color="red", lw=2.0, label="Noise start")
    ax_bline.set_title("Average One-Sided B-line", fontsize=font_sizes["title"])
    ax_bline.set_xlabel("X pixel", fontsize=font_sizes["label"])
    ax_bline.set_ylabel("Depth index", fontsize=font_sizes["label"])
    ax_bline.tick_params(axis="both", labelsize=font_sizes["tick"])
    ax_bline.legend(loc="upper right", fontsize=font_sizes["legend"])
    cbar = fig.colorbar(im, ax=ax_bline)
    cbar.set_label("Mean FFT amplitude", fontsize=font_sizes["colorbar"])
    cbar.ax.tick_params(labelsize=font_sizes["tick"])

    fig.savefig(output_path, dpi=200)
    plt.show(block=True)
    plt.close(fig)


def main():
    input_path = (
        r"E:\IOCTData\vibration charac\5mm glass base\vibration measurement"
        r"\Table float pump on closure open\closure closed"
        r"\Bline-10-Yrpt500-X1264-Z1104.tif"
    )
    output_path = None
    lambda0_nm = 840.0
    use_hanning = False
    spectral_highpass_size = 51
    fft_amplification = 100.0
    analysis_start_depth = 10

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            os.path.dirname(input_path),
            f"{base_name}_phase_variance_vs_snr.png",
        )

    raw_stack = load_stack(input_path)
    depth_complex, depth_magnitude = prepare_depth_stack(
        raw_stack,
        use_hanning=use_hanning,
        spectral_highpass_size=spectral_highpass_size,
        fft_amplification=fft_amplification,
    )

    mean_depth_image = np.mean(depth_magnitude, axis=0)
    picker = NoiseStartDepthPicker(mean_depth_image)
    noise_start_depth, bline_clim = picker.show()
    print(f"Using noise start depth: {noise_start_depth}")

    snr, phase_variance, noise_level, noise_floor_phase_variance = calculate_phase_variance_and_snr(
        depth_complex,
        noise_start_depth,
        analysis_start_depth=analysis_start_depth,
    )
    plot_phase_variance_vs_snr(
        snr=snr,
        phase_variance=phase_variance,
        mean_depth_image=mean_depth_image,
        noise_level=noise_level,
        noise_floor_phase_variance=noise_floor_phase_variance,
        center_wavelength_m=lambda0_nm * 1e-9,
        noise_start_depth=noise_start_depth,
        analysis_start_depth=analysis_start_depth,
        bline_clim=bline_clim,
        stack_path=input_path,
        output_path=output_path,
    )
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
