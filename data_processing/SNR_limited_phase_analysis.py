import os
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as TIFF


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_INPUT_PATH = (
    r"E:\IOCTData\vibration_test260824\100Cscans\TranditionCscan\Cscan-1-Bline-20-Yrpt100-X1104-Z182.tif"
)
DEFAULT_OUTPUT_DIR = None  # None saves results beside the input stack.
DEFAULT_NOISE_INPUT_PATH =(
    r"E:\IOCTData\HighResData\50Hz_2s\noise\Wout_sub_background\AllOn\070726\Bline-9-Yrpt200-X1104-Z109.tif"
)  # Set a noise-only saved AMP+PHASE TIFF here.
DEFAULT_FRAME_RATE_HZ = 40.0
DEFAULT_CENTER_WAVELENGTH_NM = 840.0
DEFAULT_REFRACTIVE_INDEX = 1.0
DEFAULT_ANALYSIS_START_DEPTH = 15
DEFAULT_NOISE_ANALYSIS_START_DEPTH = 50
# Number of leading frames to discard from the SIGNAL stack before the
# phase-noise calculation (e.g. settling/warm-up frames). The noise stack is
# always used in full. Set to 0 to keep every signal frame.
DEFAULT_DISCARD_FIRST_FRAMES = 5
# --- Complex-STD / ROI CNR analysis ---
# Master switch: compute the temporal complex-signal STD map and let the user
# draw two rectangle ROIs (tissue on the signal stack, background on the noise
# stack) to compute the per-ROI CNR.
DEFAULT_ENABLE_COMPLEX_STD_ROI_ANALYSIS = True
# Image shown for ROI drawing and on the saved ROI figure: "complex_std"
# (default - the log-scaled temporal complex-STD map, i.e. the exact map used
# for the CNR metrics) or "mean_amplitude" (log-scaled B-scan).
DEFAULT_ROI_DISPLAY_IMAGE = "complex_std"
# Optional preset rectangle ROIs as (x0, x1, z0, z1) pixel tuples. If both are
# set, the interactive selection window is skipped (useful for repeat/headless
# runs). Set both to None to draw interactively.
DEFAULT_PRESET_TISSUE_ROI = None
DEFAULT_PRESET_BACKGROUND_ROI = None
# TEMPORARY ROLLBACK: draw/measure the background ROI on the SAME tissue
# B-line image as the tissue ROI. Set back to True to use the noise stack for
# the background ROI again.
DEFAULT_ROI_BACKGROUND_ON_NOISE_STACK = False
# Fractional-dynamics CNR (power-normalized): temporal complex-STD /
# amplitude-STD maps divided by the per-pixel mean amplitude. The metric is
# invariant to the overall signal level (e.g. laser power), so it stays
# meaningful for dynamic samples while reproducing the no-contrast result for
# static/dead tissue. Reported in addition to the complex/amplitude-STD CNR.
DEFAULT_ENABLE_FRACTIONAL_DYNAMICS = True
DEFAULT_SNR_BIN_WIDTH_DB = 2.0
DEFAULT_MAX_G1_LAG = 200
DEFAULT_G1_MAX_PIXELS = 20000
DEFAULT_TARGET_TRACE_SNR_DB = (20.0, 30.0, 40.0, 50.0)
DEFAULT_THEORY_VARIANCE_COEFFICIENT = 0.5
DEFAULT_ENABLE_PHASE_VARIANCE_FIT = True
DEFAULT_FIT_SNR_RANGE_DB = (10.0, 60.0)
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


def load_saved_amp_phase_stack(path):
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
    stack = np.asarray(stack, dtype=np.float32)
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


def full_depth_stop(depth_pixels, analysis_start_depth=0):
    analysis_start_depth = int(max(0, analysis_start_depth))
    return int(max(analysis_start_depth + 1, depth_pixels))


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


def calculate_phase_noise_metrics(
    depth_complex,
    analysis_stop_depth,
    analysis_start_depth=10,
    center_wavelength_nm=840.0,
    refractive_index=1.0,
    snr_bin_width_db=2.0,
    external_sigma_q=None,
):
    magnitude = np.abs(depth_complex).astype(np.float32, copy=False)
    frames, x_pixels, depth_pixels = magnitude.shape
    analysis_start_depth = int(np.clip(analysis_start_depth, 0, depth_pixels - 1))
    analysis_stop_depth = int(np.clip(analysis_stop_depth, analysis_start_depth + 1, depth_pixels))
    if analysis_stop_depth <= analysis_start_depth:
        raise ValueError(
            "Signal analysis region is empty. Select an analysis end depth deeper than "
            f"{analysis_start_depth}."
        )

    if external_sigma_q is None:
        noise_region = np.asarray(depth_complex[:, :, :analysis_stop_depth], dtype=np.complex64)
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

    signal_slice = (slice(None), slice(analysis_start_depth, analysis_stop_depth))
    snr_flat = snr_db_map[signal_slice].reshape(-1)
    phase_std_flat = phase_std_map[signal_slice].reshape(-1)
    phase_var_flat = phase_variance_map[signal_slice].reshape(-1)
    opd_flat = opd_std_nm_map[signal_slice].reshape(-1)
    displacement_flat = displacement_std_nm_map[signal_slice].reshape(-1)

    x_grid, z_grid = np.meshgrid(
        np.arange(x_pixels),
        np.arange(analysis_start_depth, analysis_stop_depth),
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
        analysis_stop_depth,
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
        "analysis_stop_depth": analysis_stop_depth,
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
    analysis_stop_depth,
    max_lag=200,
    max_pixels=20000,
):
    analysis = depth_complex[:, :, analysis_start_depth:analysis_stop_depth]
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


def output_base_for_input(input_path, output_dir):
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    return str(Path(output_dir) / input_path.stem)


# ---------------------------------------------------------------------------
# Complex-signal STD map + ROI CNR analysis
# ---------------------------------------------------------------------------
def compute_complex_std_map(depth_complex):
    """Temporal complex-field fluctuation (std) map plus mean amplitude.

    For every pixel: complex_std = sqrt( mean_t |E(t) - <E>_t|^2 ).
    Input depth_complex: [frames, x, z] complex64.
    """
    depth_complex = np.asarray(depth_complex, dtype=np.complex64)
    mean_complex = np.mean(depth_complex, axis=0, dtype=np.complex64)
    centered = depth_complex - mean_complex[np.newaxis, :, :]
    complex_std_map = np.sqrt(
        np.mean(np.abs(centered) ** 2, axis=0, dtype=np.float32)
    ).astype(np.float32, copy=False)
    amplitude = np.abs(depth_complex).astype(np.float32, copy=False)
    mean_amplitude_map = np.mean(amplitude, axis=0, dtype=np.float32)
    amplitude_std_map = np.std(amplitude, axis=0, dtype=np.float32)
    # Fractional dynamics: temporal fluctuation normalized by the per-pixel
    # mean amplitude (power / signal-level invariant contrast).
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_complex_std_map = (complex_std_map / mean_amplitude_map).astype(
            np.float32, copy=False
        )
        normalized_amplitude_std_map = (amplitude_std_map / mean_amplitude_map).astype(
            np.float32, copy=False
        )
    normalized_complex_std_map[~np.isfinite(normalized_complex_std_map)] = np.nan
    normalized_amplitude_std_map[~np.isfinite(normalized_amplitude_std_map)] = np.nan
    return {
        "mean_complex": mean_complex,
        "complex_std_map": complex_std_map,
        "mean_amplitude_map": mean_amplitude_map,
        "amplitude_std_map": amplitude_std_map,
        "normalized_complex_std_map": normalized_complex_std_map,
        "normalized_amplitude_std_map": normalized_amplitude_std_map,
    }


def rect_roi_mask(roi, image_shape):
    """Boolean mask [x, z] for a rectangle ROI given as (x0, x1, z0, z1)."""
    x0, x1, z0, z1 = [int(round(float(v))) for v in roi]
    x0, x1 = sorted((max(0, x0), min(image_shape[0], x1)))
    z0, z1 = sorted((max(0, z0), min(image_shape[1], z1)))
    if x1 <= x0 or z1 <= z0:
        raise ValueError(f"ROI rectangle has zero area: {roi}")
    mask = np.zeros(image_shape, dtype=bool)
    mask[x0:x1, z0:z1] = True
    return mask


def select_one_rect_roi(display_image, title, roi_label="ROI", color="green"):
    """Interactive matplotlib window: draw a single rectangle ROI.

    Press-drag-release draws the rectangle; the window closes automatically.
    Press Esc to redo. Returns (x0, x1, z0, z1) as a pixel tuple.
    """
    try:
        from matplotlib.widgets import RectangleSelector
    except ImportError as error:
        raise RuntimeError(
            "matplotlib.widgets.RectangleSelector is not available; "
            "set DEFAULT_PRESET_TISSUE_ROI / DEFAULT_PRESET_BACKGROUND_ROI "
            "to run without the interactive window."
        ) from error

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.imshow(np.asarray(display_image).T, aspect="auto", origin="lower", cmap="gray")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Depth pixel")

    state = {"roi": None}

    instruction = ax.text(
        0.01,
        0.99,
        f"Drag to draw the {roi_label} ROI ({color}). Esc to redo.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )

    def onselect(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        if (x1 - x0) < 2.0 or (y1 - y0) < 2.0:
            return
        state["roi"] = (x0, x1, y0, y1)
        for patch in list(ax.patches):
            patch.remove()
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
        )
        instruction.set_text(f"{roi_label} ROI selected. Closing...")
        fig.canvas.draw_idle()
        plt.close(fig)

    def onkey(event):
        if event.key == "escape":
            state["roi"] = None
            for patch in list(ax.patches):
                patch.remove()
            instruction.set_text(f"Redo: drag to draw the {roi_label} ROI.")
            fig.canvas.draw_idle()

    rect_props = {
        "facecolor": color,
        "edgecolor": color,
        "alpha": 0.2,
        "linewidth": 1.8,
    }
    try:
        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            interactive=True,
            props=rect_props,
        )
    except TypeError:
        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            interactive=True,
            rectprops=rect_props,
        )

    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show(block=True)
    selector.disconnect_events()

    if state["roi"] is None:
        raise RuntimeError(
            "ROI selection window was closed before a rectangle was drawn."
        )
    return tuple(int(round(float(v))) for v in state["roi"])


def compute_roi_cnr_metrics(
    signal_complex_std_map,
    signal_amplitude_std_map,
    noise_complex_std_map,
    noise_amplitude_std_map,
    signal_normalized_complex_std_map,
    signal_normalized_amplitude_std_map,
    noise_normalized_complex_std_map,
    noise_normalized_amplitude_std_map,
    tissue_roi,
    background_roi,
):
    """Per-ROI CNR computed on the complex-STD / amplitude-STD maps and on the
    power-normalized (fractional-dynamics) maps.

    Tissue statistics are taken from the SIGNAL stack maps; background
    statistics from the NOISE stack maps (the background ROI is selected on the
    noise stack, so it measures the true noise floor). The fractional-dynamics
    maps divide the temporal STDs by the per-pixel mean amplitude, making the
    CNR invariant to the overall signal level (e.g. laser power).

    Returns cnr_complex_std, cnr_amplitude_std, cnr_fractional_complex,
    cnr_fractional_amplitude and the per-ROI std of the underlying values.
    """

    def roi_mean_std(mask, image):
        values = np.asarray(image[mask], dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan, np.nan
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
        return mean, std

    tissue_mask = rect_roi_mask(tissue_roi, signal_complex_std_map.shape)
    background_mask = rect_roi_mask(background_roi, noise_complex_std_map.shape)

    cstd_t_mean, cstd_t_std = roi_mean_std(tissue_mask, signal_complex_std_map)
    cstd_b_mean, cstd_b_std = roi_mean_std(background_mask, noise_complex_std_map)
    astd_t_mean, astd_t_std = roi_mean_std(tissue_mask, signal_amplitude_std_map)
    astd_b_mean, astd_b_std = roi_mean_std(background_mask, noise_amplitude_std_map)

    def ratio(numerator, denominator):
        if np.isfinite(denominator) and denominator > 0:
            return float(numerator / denominator)
        return np.nan

    cnr_complex_std = ratio(cstd_t_mean - cstd_b_mean, cstd_b_std)
    cnr_amplitude_std = ratio(astd_t_mean - astd_b_mean, astd_b_std)

    # Fractional (power-normalized) dynamics metrics.
    fcstd_t_mean, fcstd_t_std = roi_mean_std(
        tissue_mask, signal_normalized_complex_std_map
    )
    fcstd_b_mean, fcstd_b_std = roi_mean_std(
        background_mask, noise_normalized_complex_std_map
    )
    fastd_t_mean, fastd_t_std = roi_mean_std(
        tissue_mask, signal_normalized_amplitude_std_map
    )
    fastd_b_mean, fastd_b_std = roi_mean_std(
        background_mask, noise_normalized_amplitude_std_map
    )

    cnr_fractional_complex = ratio(fcstd_t_mean - fcstd_b_mean, fcstd_b_std)
    cnr_fractional_amplitude = ratio(fastd_t_mean - fastd_b_mean, fastd_b_std)

    return {
        "tissue_roi": tuple(int(v) for v in tissue_roi),
        "background_roi": tuple(int(v) for v in background_roi),
        "cnr_complex_std": cnr_complex_std,
        "cnr_amplitude_std": cnr_amplitude_std,
        "complex_std_tissue_mean": cstd_t_mean,
        "complex_std_background_mean": cstd_b_mean,
        "complex_std_tissue_std": cstd_t_std,
        "complex_std_background_std": cstd_b_std,
        "amplitude_std_tissue_std": astd_t_std,
        "amplitude_std_background_std": astd_b_std,
        "cnr_fractional_complex": cnr_fractional_complex,
        "cnr_fractional_amplitude": cnr_fractional_amplitude,
        "fractional_complex_tissue_mean": fcstd_t_mean,
        "fractional_complex_background_mean": fcstd_b_mean,
        "fractional_complex_background_std": fcstd_b_std,
        "fractional_amplitude_tissue_std": fastd_t_std,
        "fractional_amplitude_background_std": fastd_b_std,
    }


def roi_display_image(complex_std_map, mean_amplitude_map):
    """Log-scaled image shown for ROI drawing / on the saved ROI figure,
    selected by DEFAULT_ROI_DISPLAY_IMAGE ("complex_std" or "mean_amplitude")."""
    if DEFAULT_ROI_DISPLAY_IMAGE == "complex_std":
        return np.log1p(np.asarray(complex_std_map, dtype=np.float32))
    return np.log1p(np.asarray(mean_amplitude_map, dtype=np.float32))


def plot_roi_cnr_figure(
    signal_display_image,
    signal_display_label,
    noise_display_image,
    noise_display_label,
    result,
    output_path,
    show_figures=False,
    background_panel_title="Noise stack (background ROI)",
):
    """Two-panel ROI figure: signal image with the tissue ROI (left) and the
    background image (noise stack, or tissue stack during the temporary
    rollback) with the background ROI (right), annotated with the CNR
    metrics."""
    tissue_roi = result["tissue_roi"]
    background_roi = result["background_roi"]

    fig, axes = plt.subplots(1, 2, figsize=(20.0, 6.5), constrained_layout=True)

    def draw_panel(ax, image, label, roi, roi_color, roi_label):
        im = ax.imshow(np.asarray(image).T, aspect="auto", origin="lower", cmap="gray")
        fig.colorbar(im, ax=ax, label=label)
        x0, x1, z0, z1 = roi
        ax.add_patch(
            plt.Rectangle(
                (x0, z0),
                x1 - x0,
                z1 - z0,
                fill=False,
                edgecolor=roi_color,
                linewidth=2.5,
                label=roi_label,
            )
        )
        ax.set_xlabel("X pixel", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Depth pixel", fontsize=FONT_SIZES["label"])
        ax.legend(fontsize=FONT_SIZES["legend"], loc="best")

    draw_panel(
        axes[0],
        signal_display_image,
        signal_display_label,
        tissue_roi,
        "lime",
        "Tissue ROI",
    )
    draw_panel(
        axes[1],
        noise_display_image,
        noise_display_label,
        background_roi,
        "red",
        "Background ROI",
    )
    axes[0].set_title("Signal stack (tissue ROI)", fontsize=FONT_SIZES["title"])
    axes[1].set_title(background_panel_title, fontsize=FONT_SIZES["title"])

    info = (
        f"CNR (complex std) = {result['cnr_complex_std']:.3f}\n"
        f"CNR (amplitude std) = {result['cnr_amplitude_std']:.3f}"
    )
    if DEFAULT_ENABLE_FRACTIONAL_DYNAMICS:
        info += (
            f"\nCNR (fractional dyn., complex) = {result['cnr_fractional_complex']:.3f}\n"
            f"CNR (fractional dyn., amplitude) = {result['cnr_fractional_amplitude']:.3f}"
        )
    fig.text(
        0.5,
        0.02,
        info,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["annotation"],
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    style_all_axes(fig)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    if show_figures:
        plt.show(block=True)
    plt.close(fig)


def plot_fractional_dynamics_figure(
    normalized_complex_map,
    normalized_amplitude_map,
    result,
    output_path,
    show_figures=False,
):
    """Two-panel figure of the power-normalized (fractional-dynamics) maps with
    the tissue and background ROIs overlaid, annotated with the fractional CNR.

    Fractional dynamics = temporal complex-STD / mean amplitude (and the
    amplitude-STD equivalent). The metric is invariant to the overall signal
    level (e.g. laser power), so it stays meaningful for dynamic samples.
    """
    tissue_roi = result["tissue_roi"]
    background_roi = result["background_roi"]

    fig, axes = plt.subplots(1, 2, figsize=(20.0, 6.5), constrained_layout=True)

    def draw_panel(ax, image, title):
        im = ax.imshow(np.asarray(image).T, aspect="auto", origin="lower", cmap="gray")
        fig.colorbar(im, ax=ax)
        for roi, roi_color, roi_label in (
            (tissue_roi, "lime", "Tissue ROI"),
            (background_roi, "red", "Background ROI"),
        ):
            x0, x1, z0, z1 = roi
            ax.add_patch(
                plt.Rectangle(
                    (x0, z0),
                    x1 - x0,
                    z1 - z0,
                    fill=False,
                    edgecolor=roi_color,
                    linewidth=2.5,
                    label=roi_label,
                )
            )
        ax.set_title(title, fontsize=FONT_SIZES["title"])
        ax.set_xlabel("X pixel", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Depth pixel", fontsize=FONT_SIZES["label"])
        ax.legend(fontsize=FONT_SIZES["legend"], loc="best")

    draw_panel(
        axes[0],
        normalized_complex_map,
        "Normalized dynamics (complex std / mean amplitude)",
    )
    draw_panel(
        axes[1],
        normalized_amplitude_map,
        "Normalized dynamics (amplitude std / mean amplitude)",
    )

    info = (
        f"CNR (fractional dyn., complex) = {result['cnr_fractional_complex']:.3f}\n"
        f"CNR (fractional dyn., amplitude) = {result['cnr_fractional_amplitude']:.3f}\n"
        f"mean fractional complex dynamics: tissue = "
        f"{result['fractional_complex_tissue_mean']:.4g}, "
        f"bg = {result['fractional_complex_background_mean']:.4g}\n"
        f"spatial std of fractional complex dynamics in bg = "
        f"{result['fractional_complex_background_std']:.4g}"
    )
    fig.text(
        0.5,
        0.02,
        info,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["annotation"],
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    style_all_axes(fig)
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    if show_figures:
        plt.show(block=True)
    plt.close(fig)


def run_complex_std_cnr_analysis(
    signal_complex,
    noise_complex,
    output_base,
    show_figures=False,
):
    """Compute complex-STD / amplitude-STD maps, collect the tissue ROI and the
    background ROI, and report the per-ROI CNR.

    By default (DEFAULT_ROI_BACKGROUND_ON_NOISE_STACK = False, temporary
    rollback) both ROIs are drawn and measured on the tissue B-line image. Set
    that flag to True to draw/measure the background ROI on the noise stack.
    """
    signal_maps = compute_complex_std_map(signal_complex)
    signal_complex_std_map = signal_maps["complex_std_map"]
    signal_amplitude_std_map = signal_maps["amplitude_std_map"]
    signal_normalized_complex_std_map = signal_maps["normalized_complex_std_map"]
    signal_normalized_amplitude_std_map = signal_maps["normalized_amplitude_std_map"]
    signal_display = roi_display_image(signal_complex_std_map, signal_maps["mean_amplitude_map"])
    display_label = f"log(1 + {DEFAULT_ROI_DISPLAY_IMAGE})"

    background_on_noise = bool(DEFAULT_ROI_BACKGROUND_ON_NOISE_STACK)
    if background_on_noise:
        noise_maps = compute_complex_std_map(noise_complex)
        noise_complex_std_map = noise_maps["complex_std_map"]
        noise_amplitude_std_map = noise_maps["amplitude_std_map"]
        noise_normalized_complex_std_map = noise_maps["normalized_complex_std_map"]
        noise_normalized_amplitude_std_map = noise_maps["normalized_amplitude_std_map"]
        bg_display = roi_display_image(noise_complex_std_map, noise_maps["mean_amplitude_map"])
        bg_title = "Noise stack (background ROI)"
    else:
        noise_complex_std_map = signal_complex_std_map
        noise_amplitude_std_map = signal_amplitude_std_map
        noise_normalized_complex_std_map = signal_normalized_complex_std_map
        noise_normalized_amplitude_std_map = signal_normalized_amplitude_std_map
        bg_display = signal_display
        bg_title = "Signal stack (background ROI)"

    tissue_roi = DEFAULT_PRESET_TISSUE_ROI
    background_roi = DEFAULT_PRESET_BACKGROUND_ROI
    if tissue_roi is None or background_roi is None:
        tissue_roi = select_one_rect_roi(
            signal_display,
            "Draw TISSUE ROI on the SIGNAL stack (green)",
            roi_label="TISSUE",
            color="green",
        )
        print(f"Tissue ROI (signal stack): {tuple(tissue_roi)}")
        background_roi = select_one_rect_roi(
            bg_display,
            f"Draw BACKGROUND ROI on the {'NOISE' if background_on_noise else 'SIGNAL'} "
            "stack (red)",
            roi_label="BACKGROUND",
            color="red",
        )
        print(f"Background ROI ({'noise' if background_on_noise else 'signal'} stack): "
              f"{tuple(background_roi)}")

    result = compute_roi_cnr_metrics(
        signal_complex_std_map,
        signal_amplitude_std_map,
        noise_complex_std_map,
        noise_amplitude_std_map,
        signal_normalized_complex_std_map,
        signal_normalized_amplitude_std_map,
        noise_normalized_complex_std_map,
        noise_normalized_amplitude_std_map,
        tissue_roi,
        background_roi,
    )

    png_path = f"{output_base}_roi_cnr.png"
    plot_roi_cnr_figure(
        signal_display,
        display_label,
        bg_display,
        display_label,
        result,
        png_path,
        show_figures=show_figures,
        background_panel_title=bg_title,
    )
    print(f"Saved figure: {png_path}")

    if DEFAULT_ENABLE_FRACTIONAL_DYNAMICS:
        fractional_png_path = f"{output_base}_fractional_dynamics.png"
        plot_fractional_dynamics_figure(
            signal_normalized_complex_std_map,
            signal_normalized_amplitude_std_map,
            result,
            fractional_png_path,
            show_figures=show_figures,
        )
        print(f"Saved figure: {fractional_png_path}")

    print(
        "Complex-STD / ROI CNR summary:\n"
        f"  Tissue ROI     (x,z): {tuple(result['tissue_roi'])}\n"
        f"  Background ROI (x,z): {tuple(result['background_roi'])}\n"
        f"  CNR (complex std) = {result['cnr_complex_std']:.4g}\n"
        f"  CNR (amplitude std) = {result['cnr_amplitude_std']:.4g}\n"
        f"  mean complex std: tissue = {result['complex_std_tissue_mean']:.4g}, "
        f"bg = {result['complex_std_background_mean']:.4g}\n"
        f"  spatial std of complex std: tissue = "
        f"{result['complex_std_tissue_std']:.4g}, "
        f"bg = {result['complex_std_background_std']:.4g}\n"
        f"  per-ROI std of amplitude std: tissue = "
        f"{result['amplitude_std_tissue_std']:.4g}, "
        f"bg = {result['amplitude_std_background_std']:.4g}"
    )
    if DEFAULT_ENABLE_FRACTIONAL_DYNAMICS:
        print(
            "  Fractional dynamics (power-normalized, complex std / mean amp):\n"
            f"  CNR (fractional dyn., complex) = {result['cnr_fractional_complex']:.4g}\n"
            f"  CNR (fractional dyn., amplitude) = {result['cnr_fractional_amplitude']:.4g}\n"
            f"  mean fractional complex dynamics: tissue = "
            f"{result['fractional_complex_tissue_mean']:.4g}, "
            f"bg = {result['fractional_complex_background_mean']:.4g}\n"
            f"  spatial std of fractional complex dynamics in bg = "
            f"{result['fractional_complex_background_std']:.4g}"
        )
    return result


def plot_phase_variance_fit_figure(metrics, output_path, show_figures=False):
    """Standalone phase-variance-vs-SNR fitting figure with the large
    publication fonts used in the scan-mode bar-plot style."""
    pixel_metrics = metrics["pixel_metrics"]
    phase_variance_fit = metrics.get("phase_variance_fit")
    phase_variance_fit_upper = metrics.get("phase_variance_fit_upper")

    fig, ax = plt.subplots(figsize=(20.0, 18.0), constrained_layout=True)
    ax.scatter(
        pixel_metrics["snr_db"],
        pixel_metrics["phase_variance_rad2"],
        s=12,
        alpha=0.15,
        edgecolors="none",
        color="0.35",
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
    ax.plot(snr_curve_db, theory_variance, color="red", lw=6.0, label="Theory C=0.5")
    if phase_variance_fit is not None:
        fit_variance = (
            phase_variance_from_snr_db(
                snr_curve_db,
                phase_variance_fit["coefficient"],
            )
            + phase_variance_fit["sigma_floor2_rad2"]
        )
        ax.plot(
            snr_curve_db,
            fit_variance,
            color="dodgerblue",
            lw=5.0,
            ls="--",
            label="Median fit",
        )
    if phase_variance_fit_upper is not None:
        fit_variance_upper = (
            phase_variance_from_snr_db(
                snr_curve_db,
                phase_variance_fit_upper["coefficient"],
            )
            + phase_variance_fit_upper["sigma_floor2_rad2"]
        )
        ax.plot(
            snr_curve_db,
            fit_variance_upper,
            color="darkorange",
            lw=5.0,
            ls=":",
            label="Upper-percentile fit",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Pixel SNR (dB)", fontsize=42)
    ax.set_ylabel("Phase variance (rad$^2$)", fontsize=42)
    ax.tick_params(labelsize=36)
    ax.legend(fontsize=33, frameon=False)
    ax.set_xlim(-10, 50)
    ax.set_ylim(1e-4, 10)
    ax.grid(True, which="both", alpha=0.25)

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    if show_figures:
        plt.show(block=True)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def main():
    input_path = DEFAULT_INPUT_PATH
    output_base = output_base_for_input(input_path, DEFAULT_OUTPUT_DIR)
    external_sigma_q = None
    noise_distribution = None

    if DEFAULT_NOISE_INPUT_PATH:
        noise_path = DEFAULT_NOISE_INPUT_PATH
        noise_saved_stack = load_saved_amp_phase_stack(noise_path)
        noise_complex = reconstruct_complex_from_amp_phase_stack(noise_saved_stack)
        del noise_saved_stack

        noise_analysis_stop_depth = full_depth_stop(
            noise_complex.shape[2],
            analysis_start_depth=DEFAULT_NOISE_ANALYSIS_START_DEPTH,
        )
        noise_start_depth = int(np.clip(DEFAULT_NOISE_ANALYSIS_START_DEPTH, 0, max(0, noise_complex.shape[2] - 1)))
        if noise_analysis_stop_depth <= noise_start_depth:
            raise ValueError(
                "Noise analysis region is empty. "
                f"DEFAULT_NOISE_ANALYSIS_START_DEPTH={DEFAULT_NOISE_ANALYSIS_START_DEPTH} "
                f"is too deep for noise stack depth {noise_complex.shape[2]}."
            )
        noise_distribution = summarize_noise_distribution(
            noise_complex[:, :, noise_start_depth:noise_analysis_stop_depth]
        )
        external_sigma_q = estimate_sigma_q_from_complex_samples(
            noise_complex[:, :, noise_start_depth:noise_analysis_stop_depth].reshape(noise_complex.shape[0], -1)
        )
        print(
            "Measured external sigma_q from noise stack XZ range "
            f"({noise_start_depth} <= depth < {noise_analysis_stop_depth}): {external_sigma_q:.6g}"
        )
    else:
        raise ValueError(
            "A separate noise TIFF stack is required for SNR_limited_phase_analysis. "
            "Please set DEFAULT_NOISE_INPUT_PATH to a valid noise-only stack acquired "
            "under the same system settings."
        )

    saved_stack = load_saved_amp_phase_stack(input_path)
    depth_complex = reconstruct_complex_from_amp_phase_stack(saved_stack)
    del saved_stack
    print("Input treated as saved AMP+PHASE FFT-domain B-line stack.")

    # Discard the first frames of the SIGNAL stack only (warm-up/settling
    # frames). The noise stack above is intentionally kept in full.
    discard_frames = int(DEFAULT_DISCARD_FIRST_FRAMES)
    if discard_frames > 0:
        if depth_complex.shape[0] <= discard_frames:
            raise ValueError(
                f"Cannot discard {discard_frames} leading frame(s): "
                f"signal stack has only {depth_complex.shape[0]} frames."
            )
        depth_complex = depth_complex[discard_frames:, :, :]
        print(
            f"Discarded first {discard_frames} signal frame(s); analysis uses "
            f"frames {discard_frames}..{discard_frames + depth_complex.shape[0] - 1} "
            f"({depth_complex.shape[0]} frames)."
        )

    analysis_stop_depth = full_depth_stop(
        depth_complex.shape[2],
        analysis_start_depth=DEFAULT_ANALYSIS_START_DEPTH,
    )
    print(
        "Using full saved-depth range; "
        f"signal analysis end depth set to {analysis_stop_depth}."
    )

    metrics = calculate_phase_noise_metrics(
        depth_complex,
        analysis_stop_depth,
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

    phase_fit_path = f"{output_base}_phase_variance_fit.png"
    plot_phase_variance_fit_figure(
        metrics,
        phase_fit_path,
        show_figures=DEFAULT_SHOW_FIGURES,
    )

    if DEFAULT_ENABLE_COMPLEX_STD_ROI_ANALYSIS:
        run_complex_std_cnr_analysis(
            depth_complex,
            noise_complex,
            output_base=output_base,
            show_figures=DEFAULT_SHOW_FIGURES,
        )
    del noise_complex

    summary = metrics["summary"]
    print(
        "Phase floor summary: "
        f"{summary['phase_std_floor_rad_p1']:.4g} rad, "
        f"{summary['opd_std_floor_nm_p1']:.4g} nm OPD, "
        f"{summary['displacement_std_floor_nm_p1']:.4g} nm displacement"
    )


if __name__ == "__main__":
    main()
