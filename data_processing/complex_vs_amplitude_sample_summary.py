import gc
from pathlib import Path

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

import Time_domain_analysis_for_dynamicOCT as td


# Spyder/default run settings. Edit these values, then press Run.
DEFAULT_TISSUE_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\200Hz 2seconds Blines"
DEFAULT_NOISE_INPUT_PATH = r"E:\IOCTData\Lung Cancer mice 260601\260608\100Hz 10seconds Blines\noise\Noise-Yrpt1001-X1264-Z276.tif"
DEFAULT_SOURCE_FRAME_RATE_HZ = 200.0
DEFAULT_TARGET_FRAME_RATE_HZ = 50.0
DEFAULT_DURATION_SECONDS = 2.0
DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD = "skip"  # "skip" approximates lower-rate acquisition; "integrate" averages complex samples within each block.
DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE = 1
DEFAULT_DYNAMIC_CHUNK_X = 96
DEFAULT_OUTPUT_DIR = None  # None saves into "complex_vs_amplitude_sample_summary" beside the tissue stacks.
DEFAULT_SAVE_DPI = 360
DEFAULT_SUMMARY_FIGSIZE = (10.5, 5.2)
DEFAULT_SUMMARY_FONT_SIZE = 11
DEFAULT_SAVE_EXCEL = True


def iter_bline_tiff_stacks(input_path):
    paths = td.iter_tiff_stacks(input_path)
    filtered = [path for path in paths if "bline" in path.name.lower()]
    if not filtered:
        raise ValueError(f"No B-line TIFF stacks found in: {input_path}")
    return filtered


def output_directory_for_summary(tissue_input_path, configured_output_dir):
    if configured_output_dir is not None:
        output_dir = Path(configured_output_dir)
    else:
        base = Path(tissue_input_path)
        parent = base if base.is_dir() else base.parent
        output_dir = parent / "complex_vs_amplitude_sample_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def round_if_close_to_integer(value, tolerance=1e-6):
    rounded = round(float(value))
    if abs(float(value) - rounded) > tolerance:
        raise ValueError(f"Expected an integer ratio, got {value}")
    return int(rounded)


def reconstruct_complex_stack(amplitude_stack, phase_stack):
    return (
        np.asarray(amplitude_stack, dtype=np.float32)
        * np.exp(1j * np.asarray(phase_stack, dtype=np.float32))
    ).astype(np.complex64, copy=False)


def split_complex_stack_to_amp_phase(complex_stack):
    amplitude = np.abs(complex_stack).astype(np.float32, copy=False)
    phase = np.angle(complex_stack).astype(np.float32, copy=False)
    return amplitude, phase


def downsample_amp_phase(amplitude_stack, phase_stack, source_frame_rate_hz, target_frame_rate_hz, method):
    source_frame_rate_hz = float(source_frame_rate_hz)
    target_frame_rate_hz = float(target_frame_rate_hz)
    method = str(method).strip().lower()

    if target_frame_rate_hz <= 0:
        raise ValueError("DEFAULT_TARGET_FRAME_RATE_HZ must be positive.")
    if source_frame_rate_hz < target_frame_rate_hz:
        raise ValueError(
            f"Target frame rate {target_frame_rate_hz:g} Hz exceeds source frame rate "
            f"{source_frame_rate_hz:g} Hz."
        )
    if abs(source_frame_rate_hz - target_frame_rate_hz) < 1e-6:
        return (
            np.ascontiguousarray(amplitude_stack, dtype=np.float32),
            np.ascontiguousarray(phase_stack, dtype=np.float32),
        )

    factor = round_if_close_to_integer(source_frame_rate_hz / target_frame_rate_hz)
    usable_frames = (amplitude_stack.shape[0] // factor) * factor
    if usable_frames < factor:
        raise ValueError("Not enough frames for the requested temporal downsampling.")

    amplitude_stack = np.ascontiguousarray(amplitude_stack[:usable_frames], dtype=np.float32)
    phase_stack = np.ascontiguousarray(phase_stack[:usable_frames], dtype=np.float32)

    if method == "skip":
        return amplitude_stack[::factor].copy(), phase_stack[::factor].copy()

    if method == "integrate":
        complex_stack = reconstruct_complex_stack(amplitude_stack, phase_stack)
        new_shape = (usable_frames // factor, factor, amplitude_stack.shape[1], amplitude_stack.shape[2])
        complex_stack = complex_stack.reshape(new_shape)
        averaged_complex = np.mean(complex_stack, axis=1, dtype=np.complex64)
        return split_complex_stack_to_amp_phase(averaged_complex)

    raise ValueError(f"Unknown DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD: {method}")


def limit_frame_count(amplitude_stack, phase_stack, frame_count):
    frame_count = int(min(frame_count, amplitude_stack.shape[0], phase_stack.shape[0]))
    if frame_count < 2:
        raise ValueError(f"Need at least 2 frames for dynamic processing, got {frame_count}")
    return (
        np.ascontiguousarray(amplitude_stack[:frame_count], dtype=np.float32),
        np.ascontiguousarray(phase_stack[:frame_count], dtype=np.float32),
    )


def load_noise_context(noise_input_path, source_frame_rate_hz, target_frame_rate_hz, duration_seconds, method, output_dir):
    if noise_input_path is None or str(noise_input_path).strip() == "":
        raise ValueError("DEFAULT_NOISE_INPUT_PATH must point to a separate noise TIFF stack.")

    noise_path = Path(noise_input_path)
    print(f"Loading noise AMP+PHASE stack: {noise_path}")
    noise_amplitude_stack, noise_phase_stack = td.read_amp_phase_tiff_stack(noise_path)
    noise_amplitude_stack, noise_phase_stack = downsample_amp_phase(
        noise_amplitude_stack,
        noise_phase_stack,
        source_frame_rate_hz=source_frame_rate_hz,
        target_frame_rate_hz=target_frame_rate_hz,
        method=method,
    )
    target_frame_count = int(round(float(target_frame_rate_hz) * float(duration_seconds)))
    noise_amplitude_stack, noise_phase_stack = limit_frame_count(
        noise_amplitude_stack,
        noise_phase_stack,
        target_frame_count,
    )
    noise_mean_amplitude = np.mean(noise_amplitude_stack, axis=0, dtype=np.float32)

    saved_noise_roi_records = td.load_saved_roi_records(output_dir, noise_path.stem)
    saved_background_vertices = td.vertices_from_records(
        saved_noise_roi_records,
        "background",
        noise_path.name,
    )
    if saved_background_vertices is not None:
        background_mask = td.polygon_mask_for_image_shape(saved_background_vertices, noise_mean_amplitude.shape)
        background_vertices = saved_background_vertices
        print(f"Loaded saved background ROI for noise stack {noise_path.name}.")
    else:
        roi_info = td.select_background_roi(noise_mean_amplitude, noise_path.name)
        background_mask = roi_info["background_mask"]
        background_vertices = roi_info["background_vertices"]
        td.save_roi_only_workbook(
            output_dir / f"{noise_path.stem}_dynamic_timepoint_metrics.xlsx",
            {
                "tissue_vertices": np.empty((0, 2), dtype=np.float32),
                "background_vertices": background_vertices,
                "tissue_source": "",
                "background_source": noise_path.name,
            },
        )

    td.save_roi_overlay(
        noise_mean_amplitude,
        background_vertices,
        f"{noise_path.name} mean B-line with background ROI",
        output_dir / f"{noise_path.stem}_background_roi_overlay.png",
    )

    return {
        "path": noise_path,
        "amplitude_stack": noise_amplitude_stack,
        "phase_stack": noise_phase_stack,
        "mean_amplitude_image": noise_mean_amplitude,
        "background_mask": background_mask,
        "background_vertices": background_vertices,
        "frame_count": noise_amplitude_stack.shape[0],
    }


def load_tissue_roi(mean_amplitude_image, stack_path, output_dir, noise_context):
    saved_roi_records = td.load_saved_roi_records(output_dir, stack_path.stem)
    saved_tissue_vertices = td.vertices_from_records(saved_roi_records, "tissue", stack_path.name)
    if saved_tissue_vertices is not None:
        tissue_mask = td.polygon_mask_for_image_shape(saved_tissue_vertices, mean_amplitude_image.shape)
        tissue_vertices = saved_tissue_vertices
        print(f"Loaded saved tissue ROI for {stack_path.name}.")
    else:
        roi_info = td.select_tissue_roi(mean_amplitude_image, stack_path.name)
        tissue_mask = roi_info["tissue_mask"]
        tissue_vertices = roi_info["tissue_vertices"]
        td.save_roi_only_workbook(
            output_dir / f"{stack_path.stem}_dynamic_timepoint_metrics.xlsx",
            {
                "tissue_vertices": tissue_vertices,
                "background_vertices": noise_context["background_vertices"],
                "tissue_source": stack_path.name,
                "background_source": noise_context["path"].name,
            },
        )

    td.save_roi_overlay(
        mean_amplitude_image,
        tissue_vertices,
        f"{stack_path.name} mean B-line with tissue ROI",
        output_dir / f"{stack_path.stem}_tissue_roi_overlay.png",
    )
    return tissue_mask, tissue_vertices


def compute_dynamic_images(amplitude_stack, phase_stack):
    amplitude_dynamic, complex_dynamic = td.compute_dynamic_images_for_prefix(
        amplitude_stack=amplitude_stack,
        phase_stack=phase_stack,
        frame_count=amplitude_stack.shape[0],
        uniform_filter_size=DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE,
        chunk_x=DEFAULT_DYNAMIC_CHUNK_X,
    )
    return amplitude_dynamic, complex_dynamic


def compute_single_metric_record(image, tissue_mask, background_image, background_mask, signal_type, stack_name):
    tissue_values = image[tissue_mask]
    background_values = background_image[background_mask]
    tissue_mean = td.safe_mean(tissue_values)
    tissue_std = td.safe_std(tissue_values)
    background_mean = td.safe_mean(background_values)
    background_std = td.safe_std(background_values)
    cnr = np.nan
    if np.isfinite(background_std) and background_std > 0:
        cnr = (tissue_mean - background_mean) / background_std

    return {
        "stack_name": stack_name,
        "signal_type": signal_type,
        "tissue_mean": tissue_mean,
        "tissue_std": tissue_std,
        "background_mean": background_mean,
        "background_std": background_std,
        "cnr": float(cnr) if np.isfinite(cnr) else np.nan,
        "tissue_cv": tissue_std / tissue_mean if np.isfinite(tissue_mean) and tissue_mean != 0 else np.nan,
    }


def process_stack(stack_path, noise_context, output_dir):
    print(f"Loading tissue AMP+PHASE stack: {stack_path}")
    amplitude_stack, phase_stack = td.read_amp_phase_tiff_stack(stack_path)
    amplitude_stack, phase_stack = downsample_amp_phase(
        amplitude_stack,
        phase_stack,
        source_frame_rate_hz=DEFAULT_SOURCE_FRAME_RATE_HZ,
        target_frame_rate_hz=DEFAULT_TARGET_FRAME_RATE_HZ,
        method=DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD,
    )
    target_frame_count = int(round(float(DEFAULT_TARGET_FRAME_RATE_HZ) * float(DEFAULT_DURATION_SECONDS)))
    amplitude_stack, phase_stack = limit_frame_count(amplitude_stack, phase_stack, target_frame_count)
    mean_amplitude = np.mean(amplitude_stack, axis=0, dtype=np.float32)
    tissue_mask, tissue_vertices = load_tissue_roi(mean_amplitude, stack_path, output_dir, noise_context)

    amplitude_dynamic, complex_dynamic = compute_dynamic_images(amplitude_stack, phase_stack)
    noise_amplitude_dynamic, noise_complex_dynamic = compute_dynamic_images(
        noise_context["amplitude_stack"],
        noise_context["phase_stack"],
    )

    amplitude_record = compute_single_metric_record(
        image=amplitude_dynamic,
        tissue_mask=tissue_mask,
        background_image=noise_amplitude_dynamic,
        background_mask=noise_context["background_mask"],
        signal_type="amplitude",
        stack_name=stack_path.name,
    )
    complex_record = compute_single_metric_record(
        image=complex_dynamic,
        tissue_mask=tissue_mask,
        background_image=noise_complex_dynamic,
        background_mask=noise_context["background_mask"],
        signal_type="complex",
        stack_name=stack_path.name,
    )

    roi_info = {
        "tissue_vertices": tissue_vertices,
        "background_vertices": noise_context["background_vertices"],
        "tissue_source": stack_path.name,
        "background_source": noise_context["path"].name,
    }
    metadata = [
        {"key": "tissue_stack", "value": str(stack_path)},
        {"key": "noise_stack", "value": str(noise_context["path"])},
        {"key": "source_frame_rate_hz", "value": DEFAULT_SOURCE_FRAME_RATE_HZ},
        {"key": "target_frame_rate_hz", "value": DEFAULT_TARGET_FRAME_RATE_HZ},
        {"key": "duration_seconds", "value": DEFAULT_DURATION_SECONDS},
        {"key": "target_frame_count", "value": int(amplitude_stack.shape[0])},
        {"key": "temporal_downsample_method", "value": DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD},
        {"key": "dynamic_uniform_filter_size", "value": DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE},
        {"key": "dynamic_chunk_x", "value": DEFAULT_DYNAMIC_CHUNK_X},
    ]
    if DEFAULT_SAVE_EXCEL:
        td.write_metrics_excel(
            output_dir / f"{stack_path.stem}_complex_vs_amplitude_metrics.xlsx",
            [amplitude_record],
            [complex_record],
            roi_info,
            metadata=metadata,
        )

    del amplitude_stack
    del phase_stack
    del amplitude_dynamic
    del complex_dynamic
    del noise_amplitude_dynamic
    del noise_complex_dynamic
    gc.collect()
    return amplitude_record, complex_record


def paired_metric_arrays(amplitude_records, complex_records, metric_key):
    amplitude_values = np.asarray([record[metric_key] for record in amplitude_records], dtype=np.float64)
    complex_values = np.asarray([record[metric_key] for record in complex_records], dtype=np.float64)
    return amplitude_values, complex_values


def plot_summary_barplots(amplitude_records, complex_records, output_path):
    plt.rcParams.update({"font.size": DEFAULT_SUMMARY_FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=DEFAULT_SUMMARY_FIGSIZE, constrained_layout=True)

    plot_specs = [
        ("cnr", "CNR", "Tissue-background CNR"),
        ("tissue_cv", "CV", "Tissue spatial CV"),
    ]
    labels = ["Amplitude", "Complex"]
    colors = ["#4C78A8", "#F58518"]
    x_positions = np.arange(2, dtype=np.float64)
    rng = np.random.default_rng(0)

    for axis, (metric_key, y_label, title) in zip(axes, plot_specs):
        amp_values, complex_values = paired_metric_arrays(amplitude_records, complex_records, metric_key)
        means = [np.nanmean(amp_values), np.nanmean(complex_values)]
        stds = [np.nanstd(amp_values, ddof=1) if np.count_nonzero(np.isfinite(amp_values)) > 1 else 0.0,
                np.nanstd(complex_values, ddof=1) if np.count_nonzero(np.isfinite(complex_values)) > 1 else 0.0]
        axis.bar(
            x_positions,
            means,
            yerr=stds,
            width=0.62,
            color=colors,
            alpha=0.82,
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
        )

        for idx, (amp_value, complex_value) in enumerate(zip(amp_values, complex_values)):
            if not np.isfinite(amp_value) or not np.isfinite(complex_value):
                continue
            jitter = rng.normal(scale=0.025, size=2)
            xs = x_positions + jitter
            axis.plot(xs, [amp_value, complex_value], color="black", alpha=0.28, linewidth=0.8, zorder=3)
            axis.scatter(xs[0], amp_value, s=20, color=colors[0], edgecolors="black", linewidths=0.35, zorder=4)
            axis.scatter(xs[1], complex_value, s=20, color=colors[1], edgecolors="black", linewidths=0.35, zorder=4)

        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        if metric_key == "tissue_cv":
            axis.set_ylim(bottom=0.0)

    fig.suptitle(
        f"Amplitude vs complex dynamic summary ({DEFAULT_DURATION_SECONDS:g} s, "
        f"{DEFAULT_TARGET_FRAME_RATE_HZ:g} Hz, {DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD})",
        fontsize=DEFAULT_SUMMARY_FONT_SIZE + 2,
    )
    fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    print(f"Saved summary figure: {output_path}")
    plt.close(fig)


def write_summary_workbook(output_path, amplitude_records, complex_records):
    metadata = [
        {"key": "tissue_input_path", "value": DEFAULT_TISSUE_INPUT_PATH},
        {"key": "noise_input_path", "value": DEFAULT_NOISE_INPUT_PATH},
        {"key": "source_frame_rate_hz", "value": DEFAULT_SOURCE_FRAME_RATE_HZ},
        {"key": "target_frame_rate_hz", "value": DEFAULT_TARGET_FRAME_RATE_HZ},
        {"key": "duration_seconds", "value": DEFAULT_DURATION_SECONDS},
        {"key": "temporal_downsample_method", "value": DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD},
        {"key": "dynamic_uniform_filter_size", "value": DEFAULT_DYNAMIC_UNIFORM_FILTER_SIZE},
        {"key": "dynamic_chunk_x", "value": DEFAULT_DYNAMIC_CHUNK_X},
        {"key": "sample_count", "value": len(amplitude_records)},
    ]

    if pd is None or not DEFAULT_SAVE_EXCEL:
        td.write_metrics_csv(output_path.with_name(output_path.stem + "_amplitude.csv"), amplitude_records)
        td.write_metrics_csv(output_path.with_name(output_path.stem + "_complex.csv"), complex_records)
        td.write_metrics_csv(output_path.with_name(output_path.stem + "_metadata.csv"), metadata)
        return

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame.from_records(amplitude_records).to_excel(writer, sheet_name="amplitude", index=False)
        pd.DataFrame.from_records(complex_records).to_excel(writer, sheet_name="complex", index=False)
        pd.DataFrame.from_records(metadata).to_excel(writer, sheet_name="metadata", index=False)
    print(f"Saved summary workbook: {output_path}")


def main():
    tissue_paths = iter_bline_tiff_stacks(DEFAULT_TISSUE_INPUT_PATH)
    output_dir = output_directory_for_summary(DEFAULT_TISSUE_INPUT_PATH, DEFAULT_OUTPUT_DIR)
    noise_context = load_noise_context(
        noise_input_path=DEFAULT_NOISE_INPUT_PATH,
        source_frame_rate_hz=DEFAULT_SOURCE_FRAME_RATE_HZ,
        target_frame_rate_hz=DEFAULT_TARGET_FRAME_RATE_HZ,
        duration_seconds=DEFAULT_DURATION_SECONDS,
        method=DEFAULT_TEMPORAL_DOWNSAMPLE_METHOD,
        output_dir=output_dir,
    )

    amplitude_records = []
    complex_records = []
    for stack_path in tissue_paths:
        amp_record, complex_record = process_stack(stack_path, noise_context, output_dir)
        amplitude_records.append(amp_record)
        complex_records.append(complex_record)

    summary_figure_path = output_dir / "complex_vs_amplitude_summary_barplots.png"
    summary_workbook_path = output_dir / "complex_vs_amplitude_summary_metrics.xlsx"
    plot_summary_barplots(amplitude_records, complex_records, summary_figure_path)
    write_summary_workbook(summary_workbook_path, amplitude_records, complex_records)
    print("Finished complex vs amplitude sample-level summary.")


if __name__ == "__main__":
    main()
