import os
import re
import time

import numpy as np
import tifffile as TIFF


TILE_BLINE_RE = re.compile(
    r"^tile-(?P<tile>\d+)-Bline-(?P<bline>\d+)-Yrpt(?P<yrpt>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
TILE_DYN_RE = re.compile(
    r"^tile-(?P<tile>\d+)-Dyn-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
TILE_MEAN_RE = re.compile(
    r"^tile-(?P<tile>\d+)-Mean-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
STITCHED_DYN_RE = re.compile(
    r"^stitched-Dyn-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
STITCHED_MEAN_RE = re.compile(
    r"^stitched-Mean-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)


def list_sample_time_dirs(root_dir):
    sample_time_dirs = []
    if not os.path.isdir(root_dir):
        return sample_time_dirs

    for sample_name in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample_name)
        sample_match = re.match(r"sampleID-(\d+)$", sample_name)
        if not sample_match or not os.path.isdir(sample_path):
            continue
        sample_id = int(sample_match.group(1))
        for time_name in os.listdir(sample_path):
            time_path = os.path.join(sample_path, time_name)
            time_match = re.match(r"Time-(\d+)$", time_name)
            if not time_match or not os.path.isdir(time_path):
                continue
            time_id = int(time_match.group(1))
            sample_time_dirs.append((sample_id, time_id, time_path))

    sample_time_dirs.sort(key=lambda item: (item[1], item[0]))
    return sample_time_dirs


def collect_tile_bline_files(folder_path):
    tile_groups = {}
    if not os.path.isdir(folder_path):
        return tile_groups

    for filename in os.listdir(folder_path):
        match = TILE_BLINE_RE.match(filename)
        if match is None:
            continue
        tile_id = int(match.group("tile"))
        tile_groups.setdefault(tile_id, []).append(
            {
                "tile_id": tile_id,
                "bline_id": int(match.group("bline")),
                "yrpt": int(match.group("yrpt")),
                "x": int(match.group("x")),
                "z": int(match.group("z")),
                "path": os.path.join(folder_path, filename),
            }
        )

    for entries in tile_groups.values():
        entries.sort(key=lambda item: item["bline_id"])
    return tile_groups


def dynamic_output_path(folder_path, tile_id, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"tile-{tile_id}-Dyn-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def mean_output_path(folder_path, tile_id, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"tile-{tile_id}-Mean-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def tile_outputs_exist(folder_path, tile_id):
    dyn_prefix = f"tile-{tile_id}-Dyn-"
    mean_prefix = f"tile-{tile_id}-Mean-"
    dyn_exists = False
    mean_exists = False
    for filename in os.listdir(folder_path):
        if not dyn_exists and filename.startswith(dyn_prefix) and TILE_DYN_RE.match(filename):
            dyn_exists = True
        if not mean_exists and filename.startswith(mean_prefix) and TILE_MEAN_RE.match(filename):
            mean_exists = True
        if dyn_exists and mean_exists:
            return True
    return False


def stitched_dynamic_output_path(folder_path, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"stitched-Dyn-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def stitched_mean_output_path(folder_path, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"stitched-Mean-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def stitched_outputs_exist(folder_path):
    dyn_exists = False
    mean_exists = False
    if not os.path.isdir(folder_path):
        return False
    for filename in os.listdir(folder_path):
        if not dyn_exists and STITCHED_DYN_RE.match(filename):
            dyn_exists = True
        if not mean_exists and STITCHED_MEAN_RE.match(filename):
            mean_exists = True
        if dyn_exists and mean_exists:
            return True
    return False


def update_timer_readout(ui, deadline):
    if deadline is None:
        ui.TimerRead.setValue(0.0)
        return 0.0
    remaining_hours = max(0.0, (deadline - time.time()) / 3600.0)
    ui.TimerRead.setValue(remaining_hours)
    return remaining_hours


def process_idle_dynamic_until_deadline(weaver, deadline, current_message):
    while weaver.ui.RunButton.isChecked() and time.time() < deadline:
        processed = process_next_idle_dynamic_folder(weaver, deadline)
        update_timer_readout(weaver.ui, deadline)
        if not processed:
            return "Timed plate scan idle processing finished: no pending dynamic folders."
        current_message = "Timed plate scan idle processing completed one sample/time folder."
    return current_message


def process_next_idle_dynamic_folder(weaver, deadline):
    root_dir = weaver.ui.DIR.toPlainText()
    gpu_thread = getattr(weaver, "gpu_thread", None)
    # prefer_gpu = gpu_thread is not None and not getattr(gpu_thread, "SIM", True)
    if gpu_thread is None:
        return False

    for sample_id, time_id, folder_path in list_sample_time_dirs(root_dir):
        tile_groups = collect_tile_bline_files(folder_path)
        if not tile_groups:
            continue

        processed_any = False
        expected_tile_count = len(weaver.sample_fov_locations(sample_id))
        tile_count = len(tile_groups)
        for tile_id in sorted(tile_groups):
            if tile_outputs_exist(folder_path, tile_id):
                continue
            if time.time() >= deadline or not weaver.ui.RunButton.isChecked():
                return processed_any

            dynamic_slices = []
            mean_slices = []
            for entry in tile_groups[tile_id]:
                if time.time() >= deadline or not weaver.ui.RunButton.isChecked():
                    return processed_any
                stack = TIFF.imread(entry["path"])
                if stack.ndim == 2:
                    stack = stack[np.newaxis, :, :]
                for log_entry in gpu_thread.dynamic_deviation_entries(
                    np.mean(stack, axis=(1, 2)),
                    "offline_dynamic_processing_input",
                ):
                    weaver.log.dynamic_write(
                        f"{log_entry['stage']}: stack mean intensity={log_entry['reference_mean']:.3f}, "
                        f"outlier frame number={log_entry['frame_index']}, "
                        f"outlier intensity={log_entry['mean_intensity']:.3f}, "
                        f"percentage difference={log_entry['deviation_pct']:.2f}%, "
                        f"file={entry['path']}"
                    )
                dynamic_2d, mean_2d = gpu_thread.compute_dynamic_and_mean_from_stack(
                    stack
                )
                dynamic_slices.append(dynamic_2d)
                mean_slices.append(mean_2d)

            dynamic_volume = np.stack(dynamic_slices, axis=0).astype(np.float32, copy=False)
            mean_volume = np.stack(mean_slices, axis=0).astype(np.float32, copy=False)
            TIFF.imwrite(dynamic_output_path(folder_path, tile_id, dynamic_volume.shape), dynamic_volume, append=False)
            TIFF.imwrite(mean_output_path(folder_path, tile_id, mean_volume.shape), mean_volume, append=False)
            processed_any = True

        stitched_created = False
        if (
            expected_tile_count > 1
            and tile_count == expected_tile_count
            and not stitched_outputs_exist(folder_path)
        ):
            stitched_created = write_stitched_idle_outputs(weaver, sample_id, folder_path, expected_tile_count)

        if processed_any or stitched_created:
            remaining = update_timer_readout(weaver.ui, deadline)
            message = (
                f"Offline dynamic processing saved sampleID-{sample_id}/Time-{time_id}. "
                f"Remaining time: {remaining:.1f} h."
            )
            weaver.emit_status(message)
            print(message)
            return True

    return False


def write_stitched_idle_outputs(weaver, sample_id, folder_path, tile_count):
    sample_locations = weaver.sample_fov_locations(sample_id)
    if not sample_locations:
        return False

    tile_dynamic_volumes = {}
    tile_mean_volumes = {}
    for tile_id in range(1, tile_count + 1):
        dyn_path = None
        mean_path = None
        for filename in os.listdir(folder_path):
            candidate_path = os.path.join(folder_path, filename)
            if dyn_path is None and filename.startswith(f"tile-{tile_id}-Dyn-"):
                dyn_path = candidate_path
            if mean_path is None and filename.startswith(f"tile-{tile_id}-Mean-"):
                mean_path = candidate_path
            if dyn_path is not None and mean_path is not None:
                break
        if dyn_path is None or mean_path is None:
            return False
        tile_dynamic_volumes[tile_id] = TIFF.imread(dyn_path)
        tile_mean_volumes[tile_id] = TIFF.imread(mean_path)

    first_tile = tile_dynamic_volumes[1]
    fh_px, fw_px, z_px = first_tile.shape
    fw_mm = float(weaver.ui.XLength.value())
    first_y_length = sample_locations[0].y_length_mm
    fh_mm = float(first_y_length if first_y_length is not None else weaver.ui.YLength.value())

    xs = [loc.x for loc in sample_locations]
    ys = [loc.y for loc in sample_locations]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    num_cols = int(round((max_x - min_x) / fw_mm)) + 1
    num_rows = int(round((max_y - min_y) / fh_mm)) + 1

    stitched_shape = (num_rows * fh_px, num_cols * fw_px, z_px)
    stitched_dyn = np.zeros(stitched_shape, dtype=np.float32)
    stitched_mean = np.zeros(stitched_shape, dtype=np.float32)

    for tile_id, loc in enumerate(sample_locations, start=1):
        if tile_id not in tile_dynamic_volumes or tile_id not in tile_mean_volumes:
            continue
        col_idx = int(round((loc.x - min_x) / fw_mm))
        row_idx = int(round((loc.y - min_y) / fh_mm))
        y1 = row_idx * fh_px
        y2 = y1 + fh_px
        x1 = col_idx * fw_px
        x2 = x1 + fw_px
        stitched_dyn[y1:y2, x1:x2, :] = tile_dynamic_volumes[tile_id]
        stitched_mean[y1:y2, x1:x2, :] = tile_mean_volumes[tile_id]

    TIFF.imwrite(
        stitched_dynamic_output_path(folder_path, stitched_dyn.shape),
        stitched_dyn,
        append=False,
    )
    TIFF.imwrite(
        stitched_mean_output_path(folder_path, stitched_mean.shape),
        stitched_mean,
        append=False,
    )
    return True
