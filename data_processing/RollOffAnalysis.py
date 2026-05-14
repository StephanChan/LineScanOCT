# -*- coding: utf-8 -*-
"""
Roll-off analysis for processed OCT intensity B-line files.

This script:
1. reads all TIFF B-line intensity files from a directory,
2. reduces each file to one mean depth profile,
3. finds fixed artifact peak depths shared across many files,
4. rejects those artifact depths when selecting the signal peak in each file,
5. saves a CSV and scatter plot of signal peak value versus depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from tifffile import imread

try:
    from matplotlib import pyplot as plt
except Exception:
    plt = None

try:
    from scipy.signal import find_peaks
except Exception as exc:
    raise ImportError("scipy is required for RollOffAnalysis.py") from exc


@dataclass(frozen=True)
class RollOffConfig:
    data_path: Path = Path(r"E:\IOCTData\roll-off")
    output_dir_name: str = "roll_off_analysis"
    depth_pixel_size_um: float = 6.3
    envelope_bin_count: int = 30
    envelope_percentile: float = 90.0
    envelope_smoothing_span: int = 5
    peak_prominence: float = 5.0
    peak_distance: int = 8
    artifact_count_threshold: int = 40
    artifact_fraction_threshold: float = 0.20
    artifact_exclusion_half_width: int = 6
    smoothing_span: int = 5
    min_depth_index: int = 0
    max_depth_index: int | None = None
    plot: bool = True


def smooth_1d(values: np.ndarray, span: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    span = max(1, int(span))
    if span <= 1:
        return values.copy()
    kernel = np.ones(span, dtype=np.float64) / span
    return np.convolve(values, kernel, mode="same")


def load_depth_profile(path: Path) -> np.ndarray:
    data = np.asarray(imread(path), dtype=np.float64)

    if data.ndim == 3:
        # [frames, X, Z] -> mean over frames and X
        profile = np.mean(data, axis=(0, 1))
    elif data.ndim == 2:
        profile = np.mean(data, axis=0)
    elif data.ndim == 1:
        profile = data
    else:
        raise ValueError(f"{path.name}: unsupported shape {data.shape}")

    return np.asarray(profile, dtype=np.float64).ravel()


def collect_profiles(files: list[Path], config: RollOffConfig) -> tuple[np.ndarray, int, int]:
    profiles = [load_depth_profile(path) for path in files]
    min_len = min(profile.size for profile in profiles)
    start = max(0, int(config.min_depth_index))
    stop = min_len if config.max_depth_index is None else min(min_len, int(config.max_depth_index))
    if stop <= start:
        raise ValueError("Invalid depth range after applying min/max depth limits.")

    cropped = []
    for profile in profiles:
        values = smooth_1d(profile[:min_len], config.smoothing_span)
        cropped.append(values[start:stop])
    return np.vstack(cropped), start, stop


def artifact_depth_mask(profiles: np.ndarray, config: RollOffConfig) -> tuple[np.ndarray, np.ndarray]:
    n_files, depth_len = profiles.shape
    peak_counts = np.zeros(depth_len, dtype=np.int32)

    for profile in profiles:
        peaks, _ = find_peaks(
            profile,
            prominence=config.peak_prominence,
            distance=config.peak_distance,
        )
        peak_counts[peaks] += 1

    count_threshold = max(
        int(config.artifact_count_threshold),
        int(np.ceil(config.artifact_fraction_threshold * n_files)),
    )
    artifact_centers = np.flatnonzero(peak_counts >= count_threshold)
    mask = np.zeros(depth_len, dtype=bool)

    for center in artifact_centers:
        left = max(0, center - config.artifact_exclusion_half_width)
        right = min(depth_len, center + config.artifact_exclusion_half_width + 1)
        mask[left:right] = True

    return mask, artifact_centers


def choose_signal_peak(profile: np.ndarray, artifact_mask: np.ndarray, config: RollOffConfig) -> tuple[int | None, float | None]:
    peaks, properties = find_peaks(
        profile,
        prominence=config.peak_prominence,
        distance=config.peak_distance,
    )

    if peaks.size == 0:
        return None, None

    valid = ~artifact_mask[peaks]
    valid_peaks = peaks[valid]
    if valid_peaks.size == 0:
        return None, None

    peak_values = profile[valid_peaks]
    best_idx = int(np.argmax(peak_values))
    return int(valid_peaks[best_idx]), float(peak_values[best_idx])


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "filename",
        "peak_depth_index",
        "peak_depth_um",
        "peak_value",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upper_envelope_curve(
    depths_um: np.ndarray,
    values: np.ndarray,
    config: RollOffConfig,
) -> tuple[np.ndarray, np.ndarray]:
    depths_um = np.asarray(depths_um, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if depths_um.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if depths_um.size == 1:
        return depths_um.copy(), values.copy()

    order = np.argsort(depths_um)
    depths_um = depths_um[order]
    values = values[order]

    bin_count = max(3, min(int(config.envelope_bin_count), depths_um.size))
    edges = np.linspace(depths_um[0], depths_um[-1], bin_count + 1)
    centers = []
    envelope = []

    for i in range(bin_count):
        if i == bin_count - 1:
            in_bin = (depths_um >= edges[i]) & (depths_um <= edges[i + 1])
        else:
            in_bin = (depths_um >= edges[i]) & (depths_um < edges[i + 1])
        if not np.any(in_bin):
            continue
        centers.append(np.mean(depths_um[in_bin]))
        envelope.append(np.percentile(values[in_bin], config.envelope_percentile))

    if not centers:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    centers = np.asarray(centers, dtype=np.float64)
    envelope = smooth_1d(np.asarray(envelope, dtype=np.float64), config.envelope_smoothing_span)
    return centers, envelope


def plot_results(
    rows: list[dict[str, object]],
    artifact_centers: np.ndarray,
    peak_counts: np.ndarray,
    profiles: np.ndarray,
    depth_offset: int,
    output_dir: Path,
    config: RollOffConfig,
) -> None:
    if plt is None or not config.plot:
        return

    depths = [row["peak_depth_index"] for row in rows if row["peak_depth_index"] is not None]
    depths_um = [row["peak_depth_um"] for row in rows if row["peak_depth_um"] is not None]
    values = [row["peak_value"] for row in rows if row["peak_value"] is not None]

    plt.figure(figsize=(8, 5))
    plt.scatter(depths_um, values, s=18)
    if depths_um:
        envelope_depths_um, envelope_values = upper_envelope_curve(depths_um, values, config)
        if envelope_depths_um.size:
            plt.plot(envelope_depths_um, envelope_values, linewidth=2.0, alpha=0.9)
    for center in artifact_centers:
        plt.axvline((center + depth_offset) * config.depth_pixel_size_um, color="r", alpha=0.15, linewidth=1.0)
    plt.xlabel("Depth (um)")
    plt.ylabel("Mean peak value")
    plt.title("Roll-off: signal peak value over depth")
    plt.tight_layout()
    plt.savefig(output_dir / "roll_off_scatter.png", dpi=200)

    plt.figure(figsize=(10, 6))
    depth_axis = (np.arange(profiles.shape[1]) + depth_offset) * config.depth_pixel_size_um
    for profile in profiles:
        plt.plot(depth_axis, profile, linewidth=0.8, alpha=0.35)
    for center in artifact_centers:
        plt.axvline((center + depth_offset) * config.depth_pixel_size_um, color="r", alpha=0.15, linewidth=1.0)
    plt.xlabel("Depth (um)")
    plt.ylabel("Mean intensity")
    plt.title("All files mean depth profiles")
    plt.tight_layout()
    plt.savefig(output_dir / "all_mean_profiles.png", dpi=200)

    plt.figure(figsize=(8, 4))
    plt.plot((np.arange(peak_counts.size) + depth_offset) * config.depth_pixel_size_um, peak_counts, linewidth=1.5)
    for center in artifact_centers:
        plt.axvline((center + depth_offset) * config.depth_pixel_size_um, color="r", alpha=0.2, linewidth=1.0)
    plt.xlabel("Depth (um)")
    plt.ylabel("Peak count across files")
    plt.title("Fixed artifact peak frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "artifact_peak_counts.png", dpi=200)
    plt.show()


def analyze_roll_off(config: RollOffConfig = RollOffConfig()) -> dict[str, object]:
    files = sorted(config.data_path.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {config.data_path}")

    output_dir = config.data_path / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles, start, _ = collect_profiles(files, config)
    artifact_mask, artifact_centers = artifact_depth_mask(profiles, config)

    peak_counts = np.zeros(profiles.shape[1], dtype=np.int32)
    for profile in profiles:
        peaks, _ = find_peaks(
            profile,
            prominence=config.peak_prominence,
            distance=config.peak_distance,
        )
        peak_counts[peaks] += 1

    rows: list[dict[str, object]] = []
    misses = 0
    for file_path, profile in zip(files, profiles):
        peak_depth, peak_value = choose_signal_peak(profile, artifact_mask, config)
        if peak_depth is None:
            misses += 1
        rows.append(
            {
                "filename": file_path.name,
                "peak_depth_index": None if peak_depth is None else int(peak_depth + start),
                "peak_depth_um": None if peak_depth is None else float((peak_depth + start) * config.depth_pixel_size_um),
                "peak_value": peak_value,
            }
        )

    save_csv(rows, output_dir / "roll_off_peaks.csv")
    plot_results(rows, artifact_centers, peak_counts, profiles, start, output_dir, config)

    print(f"Processed {len(files)} files from {config.data_path}")
    print(f"Artifact peak centers: {(artifact_centers + start).tolist()}")
    print(f"Files without accepted signal peak: {misses}")
    print(f"Saved CSV: {output_dir / 'roll_off_peaks.csv'}")
    print(f"Saved scatter plot: {output_dir / 'roll_off_scatter.png'}")
    print(f"Saved all-profile plot: {output_dir / 'all_mean_profiles.png'}")

    return {
        "rows": rows,
        "artifact_centers": artifact_centers + start,
        "artifact_mask": artifact_mask,
        "peak_counts": peak_counts,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    analyze_roll_off()
