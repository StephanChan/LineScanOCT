# -*- coding: utf-8 -*-
"""
Port of CalibInterpolationDispersion.m.

This calibration uses two B-line measurements of a single reflector at
different depths to generate the interpolation and dispersion compensation
files consumed by ThreadGPU.update_Dispersion().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import hilbert
from tifffile import imread

try:
    from matplotlib import pyplot as plt
except Exception:
    plt = None


@dataclass(frozen=True)
class CalibrationConfig:
    data_path: Path = Path(r"E:\IOCTData\disperison_compensation")
    test_path: Path = Path(r"E:\IOCTData\disperison_compensation\test")
    nk: int = 1104
    nx: int = 1104
    ny: int = 1
    highpass_smooth_span: int = 51
    phase_smooth_span: int = 21
    dphase_smooth_span: int = 11
    phase_column_start: int = 251  # MATLAB columns 701:800
    phase_column_stop: int = 351
    file1_left_guard: int = 15
    file1_right_guard: int = 15
    file2_left_guard: int = 25
    file2_right_guard: int = 25
    interpolation_epsilon: float = 1.0e-5
    plot: bool = True
    plot_every_n_columns: int = 50
    plot_column_start: int = 1  # MATLAB column 2
    phase_plot_column_start: int = 251  # MATLAB column 702
    phase_plot_column_stop: int = 351
    run_calibration: bool = True
    run_test: bool = True


def smooth_moving_mean(values: np.ndarray, span: int = 5, axis: int = 0) -> np.ndarray:
    """Approximate MATLAB smooth(..., span) with a centered moving average."""
    values = np.asarray(values, dtype=np.float64)
    span = max(1, int(span))
    if span == 1:
        return values.copy()

    values = np.moveaxis(values, axis, 0)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty_like(flat, dtype=np.float64)
    half = span // 2

    for idx in range(flat.shape[0]):
        start = max(0, idx - half)
        stop = min(flat.shape[0], idx + half + 1)
        out[idx, :] = flat[start:stop, :].mean(axis=0)

    out = out.reshape(values.shape)
    return np.moveaxis(out, 0, axis)


def read_tiff_spectrum(path: Path, config: CalibrationConfig) -> np.ndarray:
    """Read one TIFF and return data as [samples, alines], matching MATLAB's transpose."""
    image = imread(path)
    if image.ndim > 2:
        image = np.mean(image[: config.ny, :, :], axis=0)

    data = np.asarray(image, dtype=np.float64).T
    if data.shape[0] != config.nk:
        raise ValueError(f"{path.name}: expected {config.nk} samples, got {data.shape[0]}.")
    if data.shape[1] != config.nx:
        raise ValueError(f"{path.name}: expected {config.nx} A-lines, got {data.shape[1]}.")
    plt.figure()
    plt.imshow(data)
    plt.show()
    return data


def highpass_spectra(data: np.ndarray, span: int) -> np.ndarray:
    return data - smooth_moving_mean(data, span=span, axis=0)


def analytic_phase(data: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(hilbert(data, axis=0)), axis=0)


def plot_columns(
    data: np.ndarray,
    title: str,
    config: CalibrationConfig,
    *,
    xlim: tuple[int, int] | None = None,
    linewidth: float = 1.0,
) -> None:
    if not config.plot or plt is None:
        return

    plt.figure()
    columns = range(config.plot_column_start, data.shape[1], config.plot_every_n_columns)
    for col in columns:
        plt.plot(data[:, col], linewidth=linewidth)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)


def plot_column_range(
    data: np.ndarray,
    title: str,
    config: CalibrationConfig,
    *,
    column_start: int | None = None,
    column_stop: int | None = None,
    xlim: tuple[int, int] | None = None,
    ylabel: str | None = None,
    linewidth: float = 1.0,
) -> None:
    if not config.plot or plt is None:
        return

    plt.figure()
    start = config.phase_plot_column_start if column_start is None else column_start
    stop = config.phase_plot_column_stop if column_stop is None else column_stop
    for col in range(start, min(stop, data.shape[1])):
        plt.plot(data[:, col], linewidth=linewidth)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)


def plot_line(
    data: np.ndarray,
    title: str,
    config: CalibrationConfig,
    *,
    xlim: tuple[int, int] | None = None,
    linewidth: float = 2.0,
) -> None:
    if not config.plot or plt is None:
        return

    plt.figure()
    plt.plot(data, linewidth=linewidth)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)


def plot_two_lines(
    first: np.ndarray,
    second: np.ndarray,
    title: str,
    config: CalibrationConfig,
    *,
    first_style: str | None = None,
    second_style: str | None = None,
    linewidth: float = 2.0,
) -> None:
    if not config.plot or plt is None:
        return

    plt.figure()
    if first_style is None:
        plt.plot(first, linewidth=linewidth)
    else:
        plt.plot(first, first_style, linewidth=linewidth)
    if second_style is None:
        plt.plot(second, linewidth=linewidth)
    else:
        plt.plot(second, second_style, linewidth=linewidth)
    plt.title(title)


def plot_alines_from_spectrum(
    spectra: np.ndarray,
    title: str,
    config: CalibrationConfig,
    *,
    z_range: int,
    linewidth: float = 2.0,
) -> None:
    rr0 = np.fft.ifft(spectra, axis=0)
    rr = np.abs(rr0[:z_range, :])
    plot_columns(rr, title, config, xlim=(0, z_range), linewidth=linewidth)


def clean_reflector_peak_complex(
    spectra: np.ndarray,
    z: int,
    left_guard: int,
    right_guard: int,
) -> np.ndarray:
    """Keep the reflector and conjugate regions in Fourier space."""
    nk = spectra.shape[0]
    rr0 = np.fft.ifft(spectra, axis=0)

    start = max(0, z - left_guard)
    middle_start = min(nk, z + right_guard)
    middle_stop = max(0, nk - z - right_guard)
    end_start = min(nk, nk - z + left_guard)

    rr0[:start, :] = 0
    if middle_start < middle_stop:
        rr0[middle_start:middle_stop, :] = 0
    rr0[end_start:, :] = 0
    return rr0


def clean_reflector_peak(
    spectra: np.ndarray,
    z: int,
    left_guard: int,
    right_guard: int,
) -> np.ndarray:
    rr0 = clean_reflector_peak_complex(spectra, z, left_guard, right_guard)
    return np.real(np.fft.fft(rr0, axis=0))


def phase_for_file1(data: np.ndarray, config: CalibrationConfig) -> tuple[np.ndarray, np.ndarray]:
    nk = data.shape[0]
    z_range = round(nk / 2)
    plot_columns(data, "file1 raw spectrum", config, xlim=(9, nk))

    data_hp = highpass_spectra(data, config.highpass_smooth_span)
    plot_columns(data_hp, "file1 DC removed raw spectrum", config, xlim=(0, nk))

    raw_phase = analytic_phase(data_hp)
    plot_columns(raw_phase, "file1 phase after HT", config, xlim=(0, nk))

    plot_columns(data_hp, "file1 trimmed spectrum", config, xlim=(0, nk))
    rr0 = np.fft.ifft(data_hp, axis=0)
    rra = np.abs(rr0[:z_range, :])
    plot_columns(rra, "file1 raw Alines", config, xlim=(0, z_range), linewidth=2.0)

    z = int(np.argmax(np.mean(rra, axis=1)))
    rr0_clean = clean_reflector_peak_complex(
        data_hp,
        z=z,
        left_guard=config.file1_left_guard,
        right_guard=config.file1_right_guard,
    )
    plot_columns(np.abs(rr0_clean), "file1 cleaned Alines", config, xlim=(0, z_range), linewidth=2.0)

    cleaned = np.real(np.fft.fft(rr0_clean, axis=0))
    plot_columns(cleaned, "file1 cleaned spectrum", config, xlim=(0, nk))

    phase = analytic_phase(cleaned)
    plot_column_range(phase, "file1 phase after spectrum trim", config, xlim=(0, nk), ylabel="rad")

    phase_mean = np.mean(phase[:, config.phase_column_start : config.phase_column_stop], axis=1)
    phase_mean = smooth_moving_mean(phase_mean, config.phase_smooth_span)
    plot_line(phase_mean, "phaseA", config)
    return cleaned, phase_mean


def phase_for_file2(data: np.ndarray, config: CalibrationConfig) -> tuple[np.ndarray, np.ndarray]:
    nk = data.shape[0]
    z_range = round(nk / 2)
    plot_columns(data, "file2 raw spectrum", config, xlim=(0, nk))

    data_hp = highpass_spectra(data, config.highpass_smooth_span)
    plot_columns(data_hp, "file2 trimmed spectrum", config, xlim=(0, nk))

    rr0 = np.fft.ifft(data_hp, axis=0)
    rrb = np.abs(rr0[:z_range, :])
    plot_columns(rrb, "file2 raw Alines", config, xlim=(0, z_range), linewidth=2.0)

    search_start = 10
    z = int(np.argmax(np.mean(rrb[search_start:, :], axis=1)) + search_start)
    rr0_clean = clean_reflector_peak_complex(
        data_hp,
        z=z,
        left_guard=config.file2_left_guard,
        right_guard=config.file2_right_guard,
    )
    plot_columns(np.abs(rr0_clean), "file2 cleaned Alines", config, xlim=(0, z_range), linewidth=2.0)

    cleaned = np.real(np.fft.fft(rr0_clean, axis=0))
    plot_columns(cleaned, "file2 cleaned spectrum", config, xlim=(0, nk))

    phase = analytic_phase(cleaned)
    plot_column_range(phase, "file2 phase after HT", config, xlim=(0, nk), ylabel="rad")

    phase_mean = np.mean(phase[:, config.phase_column_start : config.phase_column_stop], axis=1)
    phase_mean = smooth_moving_mean(phase_mean, config.phase_smooth_span)
    plot_line(phase_mean, "phaseB", config)
    return cleaned, phase_mean


def find_interp_indices(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Return zero-based bracketing indices for linear interpolation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    xp = np.asarray(xp, dtype=np.float64).ravel()

    if x.size != xp.size:
        raise ValueError("x and xp must have the same length.")

    order = np.argsort(x)
    x_sorted = x[order]

    positions = np.searchsorted(x_sorted, xp, side="left")
    positions = np.clip(positions, 1, x_sorted.size - 1)
    i0_sorted = positions - 1
    i1_sorted = positions

    idx0 = order[i0_sorted]
    idx1 = order[i1_sorted]

    same = np.abs(x[idx1] - x[idx0]) < np.finfo(np.float64).eps
    if np.any(same):
        idx1[same] = np.clip(idx0[same] + 1, 0, x.size - 1)

    return np.vstack([idx0, idx1]).astype(np.uint16)


def interpolate_spectra(
    x: np.ndarray,
    xp: np.ndarray,
    spectra: np.ndarray,
    indices: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    idx0 = indices[0, :].astype(np.intp, copy=False)
    idx1 = indices[1, :].astype(np.intp, copy=False)
    x0 = x[idx0, np.newaxis]
    x1 = x[idx1, np.newaxis]
    xt = xp[:, np.newaxis]
    y0 = spectra[idx0, :]
    y1 = spectra[idx1, :]
    return y0 + (xt - x0) * (y1 - y0) / (x1 - x0 + epsilon)


def residual_dispersion_phase(spectra: np.ndarray) -> np.ndarray:
    phase = analytic_phase(spectra)
    phase_mean = smooth_moving_mean(np.mean(phase, axis=1))
    line = np.linspace(phase_mean[0], phase_mean[-1], phase_mean.size)
    return phase_mean - line


def load_generated_outputs(config: CalibrationConfig) -> dict[str, np.ndarray]:
    """Load generated interpolation/dispersion files without recalibrating."""
    intp_x = np.fromfile(config.data_path / "intpX.bin", dtype=np.float32)
    intp_xp = np.fromfile(config.data_path / "intpXp.bin", dtype=np.float32)
    raw_indices = np.fromfile(config.data_path / "intpIndice.bin", dtype=np.uint16)
    dsp_phase = np.fromfile(config.data_path / "dspPhase.bin", dtype=np.float32)

    expected_indices = 2 * config.nk
    if intp_x.size != config.nk:
        raise ValueError(f"intpX.bin has {intp_x.size} values, expected {config.nk}.")
    if intp_xp.size != config.nk:
        raise ValueError(f"intpXp.bin has {intp_xp.size} values, expected {config.nk}.")
    if dsp_phase.size != config.nk:
        raise ValueError(f"dspPhase.bin has {dsp_phase.size} values, expected {config.nk}.")
    if raw_indices.size != expected_indices:
        raise ValueError(
            f"intpIndice.bin has {raw_indices.size} values, expected {expected_indices}."
        )

    index_options = [
        raw_indices.reshape(2, config.nk),
        raw_indices.reshape(config.nk, 2).T,
    ]
    for indices in index_options:
        if int(indices.min()) >= 0 and int(indices.max()) < config.nk:
            return {
                "intpX": intp_x,
                "intpXp": intp_xp,
                "intpIndice": indices,
                "dspPhase": dsp_phase,
            }

    raise ValueError("intpIndice.bin contains indices outside the valid sample range.")


def apply_generated_compensation(
    data: np.ndarray,
    calibration: dict[str, np.ndarray],
    config: CalibrationConfig,
    title_prefix: str,
) -> dict[str, np.ndarray]:
    """Apply existing calibration files to one measured spectrum after DC removal."""
    nk = data.shape[0]
    z_range = round(nk / 2)
    data_dc_removed = highpass_spectra(data, config.highpass_smooth_span)
    plot_columns(data_dc_removed, f"{title_prefix} DC removed spectrum", config, xlim=(0, nk))

    data_interp = interpolate_spectra(
        calibration["intpX"],
        calibration["intpXp"],
        data_dc_removed,
        calibration["intpIndice"],
        config.interpolation_epsilon,
    )
    data_compensated = data_interp * np.exp(1j * calibration["dspPhase"][:, np.newaxis])
    final_alines = np.abs(np.fft.ifft(data_compensated, axis=0)[:z_range, :])
    final_profile = np.mean(final_alines, axis=1)
    peak_index = int(np.argmax(final_profile))
    peak_value = float(final_profile[peak_index])

    plot_columns(final_alines, f"{title_prefix} final Alines", config, xlim=(0, z_range), linewidth=2.0)
    plot_line(final_profile, f"{title_prefix} final mean profile, peak={peak_index}", config)

    print(f"{title_prefix}: final peak index={peak_index}, value={peak_value:.6g}")

    return {
        "dc_removed": data_dc_removed,
        "interpolated": data_interp,
        "compensated": data_compensated,
        "final_alines": final_alines,
        "final_profile": final_profile,
        "peak_index": peak_index,
        "peak_value": peak_value,
    }


def test_generated_files(config: CalibrationConfig = CalibrationConfig()) -> dict[str, dict[str, np.ndarray]]:
    """Test generated calibration files on DC-removed TIFF spectra in config.test_path."""
    if config.plot and plt is None:
        print("matplotlib is unavailable; skipping diagnostic plots.")

    calibration = load_generated_outputs(config)
    files = sorted(config.test_path.glob("*.tif"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least two TIFF files in {config.test_path}.")

    results = {}
    for index, file_path in enumerate(files[:2], start=1):
        data = read_tiff_spectrum(file_path, config)
        title_prefix = f"test file{index} {file_path.stem}"
        results[file_path.name] = apply_generated_compensation(
            data,
            calibration,
            config,
            title_prefix,
        )

    if config.plot and plt is not None:
        plt.show()

    return results


def write_outputs(
    output_path: Path,
    dphase: np.ndarray,
    lin_phase: np.ndarray,
    indices: np.ndarray,
    dispersion_phase: np.ndarray,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    np.asarray(dphase, dtype=np.float32).tofile(output_path / "intpX.bin")
    np.asarray(lin_phase, dtype=np.float32).tofile(output_path / "intpXp.bin")
    np.asarray(indices, dtype=np.uint16).tofile(output_path / "intpIndice.bin")
    np.asarray(dispersion_phase, dtype=np.float32).tofile(output_path / "dspPhase.bin")

    savemat(output_path / "dphase.mat", {"dphase": np.asarray(dphase, dtype=np.float32)})
    savemat(output_path / "linPhase.mat", {"linPhase": np.asarray(lin_phase, dtype=np.float32)})
    savemat(output_path / "indice.mat", {"indice": np.asarray(indices, dtype=np.uint16)})


def calibrate(config: CalibrationConfig = CalibrationConfig()) -> dict[str, np.ndarray]:
    if config.plot and plt is None:
        print("matplotlib is unavailable; skipping diagnostic plots.")

    files = sorted(config.data_path.glob("*.tif"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least two TIFF files in {config.data_path}.")

    dat_a = read_tiff_spectrum(files[0], config)
    dat_b = read_tiff_spectrum(files[1], config)

    dat_a_clean, phase_a = phase_for_file1(dat_a, config)
    dat_b_clean, phase_b = phase_for_file2(dat_b, config)

    dphase = smooth_moving_mean(phase_b - phase_a, config.dphase_smooth_span)
    lin_phase = np.linspace(dphase[0], dphase[-1], dphase.size)
    plot_two_lines(
        phase_a,
        phase_b,
        "phase of two Alines at different depths",
        config,
        first_style="r",
        second_style="b",
    )
    plot_two_lines(dphase, lin_phase, "phaseB", config)

    indices = find_interp_indices(dphase, lin_phase)

    dat_a_interp = interpolate_spectra(
        dphase,
        lin_phase,
        dat_a_clean,
        indices,
        config.interpolation_epsilon,
    )
    dat_b_interp = interpolate_spectra(
        dphase,
        lin_phase,
        dat_b_clean,
        indices,
        config.interpolation_epsilon,
    )

    nk = dat_a.shape[0]
    z_range = round(nk / 2)
    plot_columns(dat_a_interp, "file1 spectrum after interpolation", config, xlim=(0, nk), linewidth=2.0)
    plot_columns(dat_b_interp, "file2 spectrum after interpolation", config, xlim=(0, nk), linewidth=2.0)
    plot_alines_from_spectrum(dat_a_interp, "file1 Alines after interpolation", config, z_range=z_range)
    plot_alines_from_spectrum(dat_b_interp, "file2 Alines after interpolation", config, z_range=z_range)

    dphase1 = residual_dispersion_phase(dat_a_interp)
    dphase2 = residual_dispersion_phase(dat_b_interp)
    plot_two_lines(dphase1, dphase2, "residual dispersion", config)

    dat_a_dispersion = dat_a_interp * np.exp(1j * dphase1[:, np.newaxis])
    dat_b_dispersion = dat_b_interp * np.exp(1j * dphase1[:, np.newaxis])
    plot_alines_from_spectrum(dat_a_dispersion, "file1 Alines", config, z_range=z_range)
    plot_alines_from_spectrum(dat_b_dispersion, "file2 Alines", config, z_range=z_range)

    write_outputs(config.data_path, dphase, lin_phase, indices, dphase1)

    if config.plot and plt is not None:
        plt.show()

    return {
        "intpX": dphase,
        "intpXp": lin_phase,
        "intpIndice": indices,
        "dspPhase": dphase1,
        "file2_residual_dispersion": dphase2,
    }


if __name__ == "__main__":
    cfg = CalibrationConfig()

    if cfg.run_calibration:
        result = calibrate(cfg)
        print("Wrote calibration files:")
        print("  intpX.bin")
        print("  intpXp.bin")
        print("  intpIndice.bin")
        print("  dspPhase.bin")
        print(f"Samples: {result['dspPhase'].size}")

    if cfg.run_test:
        test_results = test_generated_files(cfg)
        print("Tested generated calibration files after DC removal:")
        for filename, result in test_results.items():
            print(
                f"  {filename}: peak index={result['peak_index']}, "
                f"value={result['peak_value']:.6g}"
            )
