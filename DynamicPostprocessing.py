import json
import os
import re
import time

import numpy as np
import tifffile as TIFF


OFFLINE_DYNAMIC_PROCESSING_ENABLED = True


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
# Non-dynamic (static) per-tile Cscan volumes: tile-<id>-Y...-X...-Z....tif
TILE_STATIC_RE = re.compile(
    r"^tile-(?P<tile>\d+)-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
STITCHED_STATIC_RE = re.compile(
    r"^stitched-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)

# Fallback XY downsample for the stitched output (applied along both spatial
# axes Y and X, keeping the depth Z unchanged, for both dynamic Dyn/Mean and
# static full-volume stitching). The active factor is read from the UI
# "downsample scale" spinbox (ui.scale) at stitch time; this constant is only
# the default when that widget is not present (e.g. offline test scripts).
STITCHED_XY_DOWNSAMPLE = 2


def stitch_xy_downsample(weaver):
    """Return the XY downsample factor for offline stitched volumes.

    Read from the UI "downsample scale" spinbox (``ui.scale``) - the same
    widget ThreadDnS uses for the realtime stitched mosaic volumes. Falls back
    to ``STITCHED_XY_DOWNSAMPLE`` when the widget is not available (e.g.
    offline test/processing scripts).
    """
    scale_control = getattr(weaver.ui, "scale", None)
    if scale_control is not None:
        try:
            value = int(scale_control.value())
        except (TypeError, ValueError):
            value = 0
        if value >= 1:
            return value
    return STITCHED_XY_DOWNSAMPLE


def read_volume_stack(path):
    """Read a multi-page TIFF volume as one 3D array.

    Tile files written frame-by-frame with ``TIFF.imwrite(..., append=True)``
    store every page as its own series in tifffile, so a plain ``imread``
    returns only the first page. Stacking the pages explicitly recovers the
    full ``[Y, X, Z]`` volume (also works for normally-written single-series
    stacks and single-page images).
    """
    with TIFF.TiffFile(path) as tif:
        if len(tif.pages) <= 1:
            return tif.pages[0].asarray()
        if len(tif.series) == len(tif.pages):
            return np.stack([page.asarray() for page in tif.pages])
        return tif.series[0].asarray()


def block_mean_xy(array, factor):
    """Average Y (axis 0) and X (axis 1) in ``factor``-sized blocks.

    All remaining axes (e.g. Z) are kept unchanged, so the stitched output
    can be downsampled in-plane without touching the depth axis.
    """
    factor = max(1, int(factor))
    if factor <= 1 or array.ndim < 2:
        return array
    y_len, x_len = array.shape[0], array.shape[1]
    y_trim = y_len - y_len % factor
    x_trim = x_len - x_len % factor
    if y_trim == 0 or x_trim == 0:
        return array[::factor, ::factor]
    view = array[:y_trim, :x_trim]
    reshaped = view.reshape(
        (y_trim // factor, factor, x_trim // factor, factor) + tuple(view.shape[2:])
    )
    return reshaped.mean(axis=(1, 3))


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


def collect_tile_volume_files(folder_path):
    """Return the set of tile ids that already have a per-tile dynamic volume
    file (realtime dynamic path: tile-<id>-Dyn-...)."""
    tile_ids = set()
    if not os.path.isdir(folder_path):
        return tile_ids
    for filename in os.listdir(folder_path):
        match = TILE_DYN_RE.match(filename)
        if match is not None:
            tile_ids.add(int(match.group("tile")))
    return tile_ids


def collect_tile_static_files(folder_path):
    """Return the set of tile ids that have a static per-tile Cscan volume
    file (non-dynamic path: tile-<id>-Y...-X...-Z....tif)."""
    tile_ids = set()
    if not os.path.isdir(folder_path):
        return tile_ids
    for filename in os.listdir(folder_path):
        match = TILE_STATIC_RE.match(filename)
        if match is not None:
            tile_ids.add(int(match.group("tile")))
    return tile_ids


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


def stitched_static_output_path(folder_path, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"stitched-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def stitched_outputs_exist(folder_path):
    dyn_exists = False
    mean_exists = False
    static_exists = False
    if not os.path.isdir(folder_path):
        return False
    for filename in os.listdir(folder_path):
        if not dyn_exists and STITCHED_DYN_RE.match(filename):
            dyn_exists = True
        if not mean_exists and STITCHED_MEAN_RE.match(filename):
            mean_exists = True
        if not static_exists and (
            STITCHED_STATIC_RE.match(filename)
            or (filename.startswith("stitched-offline-") and filename.endswith(".tif"))
        ):
            static_exists = True
        if (dyn_exists and mean_exists) or static_exists:
            return True
    return False


def load_tile_positions_manifest(folder_path):
    """Return the tile records from ``tile_positions.json``, or ``None``.

    The manifest is the source of truth for which tile files belong to the
    current scan: ``tile_index`` is the tile file number and ``tile_filename``
    the exact saved filename. ``None`` is returned when the file is missing or
    malformed.
    """
    path = os.path.join(folder_path, "tile_positions.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, ValueError):
        return None
    records = manifest.get("tiles")
    if not isinstance(records, list) or not records:
        return None
    return records


def update_timer_readout(ui, deadline):
    if deadline is None:
        ui.TimerRead.setValue(0.0)
        return 0.0
    remaining_hours = max(0.0, (deadline - time.time()) / 3600.0)
    ui.TimerRead.setValue(remaining_hours)
    return remaining_hours


def process_next_idle_dynamic_folder(weaver, deadline):
    if not OFFLINE_DYNAMIC_PROCESSING_ENABLED:
        return False
    root_dir = weaver.ui.DIR.toPlainText()
    gpu_thread = getattr(weaver, "gpu_thread", None)
    # prefer_gpu = gpu_thread is not None and not getattr(gpu_thread, "SIM", True)
    if gpu_thread is None:
        return False

    for sample_id, time_id, folder_path in list_sample_time_dirs(root_dir):
        tile_groups = collect_tile_bline_files(folder_path)
        volume_tile_ids = collect_tile_volume_files(folder_path)
        static_tile_ids = collect_tile_static_files(folder_path)
        if not tile_groups and not volume_tile_ids and not static_tile_ids:
            continue

        processed_any = False
        expected_tile_count = len(weaver.sample_fov_locations(sample_id))
        # Tiles can come from the non-realtime path (per-Y Bline time-trace
        # stacks, from which Dyn/Mean are computed below), already exist as
        # per-tile Dyn/Mean volumes from the realtime path, or be static
        # per-tile Cscan volumes. All feed the same stitched output.
        tile_count = max(len(tile_groups), len(volume_tile_ids), len(static_tile_ids))
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
                stack = read_volume_stack(entry["path"])
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
            (tile_groups or volume_tile_ids)
            and expected_tile_count > 1
            and tile_count == expected_tile_count
            and not stitched_outputs_exist(folder_path)
        ):
            stitched_created = write_stitched_idle_outputs(weaver, sample_id, folder_path, expected_tile_count)

        # Static (non-dynamic) per-tile Cscan volumes are stitched into one
        # full-resolution stack as well. When tile_positions.json is present
        # and complete it is the source of truth for the expected tiles, so
        # stale/offset files left behind by an interrupted repeat scan do not
        # break the completeness check.
        manifest_records = load_tile_positions_manifest(folder_path)
        manifest_complete = False
        if manifest_records is not None and len(manifest_records) > 1:
            manifest_complete = True
            for record in manifest_records:
                filename = record.get("tile_filename")
                if not filename or not os.path.isfile(
                    os.path.join(folder_path, filename)
                ):
                    manifest_complete = False
                    break
        static_ready = manifest_complete or (
            bool(static_tile_ids)
            and expected_tile_count > 1
            and len(static_tile_ids) == expected_tile_count
        )
        if static_ready and not stitched_outputs_exist(folder_path):
            stitched_created = (
                write_stitched_static_outputs(
                    weaver,
                    sample_id,
                    folder_path,
                    expected_tile_count,
                    manifest_records=manifest_records if manifest_complete else None,
                )
                or stitched_created
            )

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
    """Stitch offline Dyn/Mean per-tile volumes into memory-mapped BigTIFFs.

    The Dyn and Mean stitched outputs are created directly on disk with
    ``tifffile.memmap`` (same RAM-cheap strategy as the static stitcher); only
    one tile's Dyn + Mean volumes are held in memory at a time.
    """
    if not OFFLINE_DYNAMIC_PROCESSING_ENABLED:
        return False
    sample_locations = weaver.sample_fov_locations(sample_id)
    if not sample_locations:
        return False
    # Stitched mosaic is saved at ORIGINAL resolution (no in-plane downsampling).
    downsample = 1

    # Locate every Dyn/Mean tile file first so we never create partial outputs
    # when a tile is missing.
    tile_paths = {}
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
        tile_paths[tile_id] = (dyn_path, mean_path)

    # Volume geometry from the first tile (Dyn and Mean must agree).
    first_dyn_path, first_mean_path = tile_paths[1]
    first_dyn = read_volume_stack(first_dyn_path)
    if first_dyn.ndim < 3:
        first_dyn = first_dyn[np.newaxis, ...]
    if downsample > 1:
        first_dyn = block_mean_xy(first_dyn, downsample)
    first_mean = read_volume_stack(first_mean_path)
    if first_mean.ndim < 3:
        first_mean = first_mean[np.newaxis, ...]
    if downsample > 1:
        first_mean = block_mean_xy(first_mean, downsample)
    if first_dyn.shape != first_mean.shape:
        print(
            "Dynamic stitch: first Dyn/Mean tile shapes differ "
            f"({first_dyn.shape} vs {first_mean.shape}); aborting."
        )
        return False
    fh_px, fw_px, z_px = first_dyn.shape
    dyn_dtype = first_dyn.dtype
    mean_dtype = first_mean.dtype
    del first_dyn, first_mean

    fw_mm = float(weaver.ui.XLength.value())
    first_y_length = sample_locations[0].y_length_mm
    fh_mm = float(
        first_y_length if first_y_length is not None else weaver.ui.YLength.value()
    )

    xs = [loc.x for loc in sample_locations]
    ys = [loc.y for loc in sample_locations]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    num_cols = int(round((max_x - min_x) / fw_mm)) + 1
    num_rows = int(round((max_y - min_y) / fh_mm)) + 1

    stitched_shape = (num_rows * fh_px, num_cols * fw_px, z_px)
    dyn_out = stitched_dynamic_output_path(folder_path, stitched_shape)
    mean_out = stitched_mean_output_path(folder_path, stitched_shape)
    print(
        f"Dynamic mosaic stitch: {len(tile_paths)} tile(s) -> "
        f"Dyn/Mean shape={stitched_shape[0]}x{stitched_shape[1]}x{stitched_shape[2]}."
    )

    stitched_dyn = TIFF.memmap(
        dyn_out, shape=stitched_shape, dtype=dyn_dtype, bigtiff=True
    )
    stitched_mean = TIFF.memmap(
        mean_out, shape=stitched_shape, dtype=mean_dtype, bigtiff=True
    )
    try:
        for tile_id, loc in enumerate(sample_locations, start=1):
            if tile_id not in tile_paths:
                continue
            dyn_path, mean_path = tile_paths[tile_id]

            dyn_volume = read_volume_stack(dyn_path)
            if dyn_volume.ndim < 3:
                dyn_volume = dyn_volume[np.newaxis, ...]
            if downsample > 1:
                dyn_volume = block_mean_xy(dyn_volume, downsample)

            mean_volume = read_volume_stack(mean_path)
            if mean_volume.ndim < 3:
                mean_volume = mean_volume[np.newaxis, ...]
            if downsample > 1:
                mean_volume = block_mean_xy(mean_volume, downsample)

            if dyn_volume.shape != (fh_px, fw_px, z_px) or mean_volume.shape != (
                fh_px,
                fw_px,
                z_px,
            ):
                print(
                    f"Dynamic stitch: skipping tile-{tile_id} "
                    f"(Dyn {dyn_volume.shape}, Mean {mean_volume.shape})."
                )
                del dyn_volume, mean_volume
                continue

            col_idx = int(round((loc.x - min_x) / fw_mm))
            row_idx = int(round((loc.y - min_y) / fh_mm))
            # Match the live mosaic stitch order: reverse both axes (right-to-left,
            # top-to-bottom) without rotating the tile pixels.
            col_idx = num_cols - 1 - col_idx
            row_idx = num_rows - 1 - row_idx
            y1 = row_idx * fh_px
            y2 = y1 + fh_px
            x1 = col_idx * fw_px
            x2 = x1 + fw_px
            stitched_dyn[y1:y2, x1:x2, :] = dyn_volume
            stitched_mean[y1:y2, x1:x2, :] = mean_volume
            del dyn_volume, mean_volume
            if tile_id % 5 == 0 or tile_id == len(sample_locations):
                print(f"  placed tile {tile_id}/{len(sample_locations)}")
        stitched_dyn.flush()
        stitched_mean.flush()
    finally:
        for mapped in (stitched_dyn, stitched_mean):
            try:
                mapped.flush()
            except Exception:
                pass
            try:
                del mapped
            except Exception:
                pass
    return True
def write_stitched_static_outputs(
    weaver, sample_id, folder_path, tile_count, manifest_records=None
):
    """Stitch static (non-dynamic) per-tile Cscan volumes into one stack.

    Each static tile is a full ``[Y, X, Z]`` volume saved as
    ``tile-<id>-Y...-X...-Z....tif``. Tiles are placed on the FOV grid and the
    result is written as ``stitched-Y...-X...-Z....tif`` at ORIGINAL resolution
    (depth unchanged).

    Positions come from ``weaver.sample_fov_locations(sample_id)`` unless
    ``manifest_records`` is given, in which case each record's
    ``stage_x_mm``/``stage_y_mm`` and ``tile_filename`` (from
    ``tile_positions.json``) are used as the source of truth.

    The output file is created as a memory-mapped BigTIFF so the whole mosaic
    volume is never held in RAM; peak memory is roughly one tile volume plus a
    single Y-X page of the output.
    """
    if not OFFLINE_DYNAMIC_PROCESSING_ENABLED:
        return False

    # Stitched mosaic is saved at ORIGINAL resolution (no in-plane downsampling).
    downsample = 1

    if manifest_records is not None:
        entries = []
        for record in manifest_records:
            filename = record.get("tile_filename")
            if not filename:
                return False
            path = os.path.join(folder_path, filename)
            if not os.path.isfile(path):
                return False
            y_length = record.get("y_length_mm")
            entries.append(
                (
                    float(record["stage_x_mm"]),
                    float(record["stage_y_mm"]),
                    float(y_length) if y_length is not None else None,
                    path,
                )
            )
        fw_mm = float(
            manifest_records[0].get("x_length_mm", weaver.ui.XLength.value())
        )
        first_y_length = entries[0][2]
        fh_mm = float(
            first_y_length if first_y_length is not None else weaver.ui.YLength.value()
        )
    else:
        sample_locations = weaver.sample_fov_locations(sample_id)
        if not sample_locations:
            return False
        entries = [(loc.x, loc.y, loc.y_length_mm, None) for loc in sample_locations]
        fw_mm = float(weaver.ui.XLength.value())
        first_y_length = sample_locations[0].y_length_mm
        fh_mm = float(
            first_y_length if first_y_length is not None else weaver.ui.YLength.value()
        )

    def _tile_path(tile_id):
        for filename in os.listdir(folder_path):
            if filename.startswith(f"tile-{tile_id}-Y"):
                return os.path.join(folder_path, filename)
        return None

    # Resolve every tile's file and skip missing ones (like the in-memory
    # version did for non-manifest runs).
    resolved = []
    for tile_id, (x, y, _y_len, path) in enumerate(entries, start=1):
        if path is None:
            path = _tile_path(tile_id)
        if path is None or not os.path.isfile(path):
            continue
        resolved.append((float(x), float(y), path))
    if not resolved:
        return False

    # Tile volume geometry (first available tile). read_volume_stack returns
    # the full [Y, X, Z] tile; only one tile is kept in memory at a time.
    first_volume = read_volume_stack(resolved[0][2])
    if first_volume.ndim < 3:
        first_volume = first_volume[np.newaxis, ...]
    if downsample > 1:
        first_volume = block_mean_xy(first_volume, downsample)
    fh_px, fw_px, z_px = first_volume.shape
    first_dtype = first_volume.dtype
    del first_volume

    xs = [entry[0] for entry in resolved]
    ys = [entry[1] for entry in resolved]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    num_cols = int(round((max_x - min_x) / fw_mm)) + 1
    num_rows = int(round((max_y - min_y) / fh_mm)) + 1

    stitched_shape = (num_rows * fh_px, num_cols * fw_px, z_px)
    out_path = stitched_static_output_path(folder_path, stitched_shape)
    print(
        f"Static mosaic stitch: {len(resolved)} tile(s) -> "
        f"shape={stitched_shape[0]}x{stitched_shape[1]}x{stitched_shape[2]} "
        f"({stitched_shape[0]*stitched_shape[1]*stitched_shape[2]*np.dtype(first_dtype).itemsize/1e9:.2f} GB)."
    )

    # Create the empty mosaic directly on disk as a memory-mapped BigTIFF.
    stitched = TIFF.memmap(
        out_path,
        shape=stitched_shape,
        dtype=first_dtype,
        bigtiff=True,
    )
    try:
        for idx, (x, y, path) in enumerate(resolved):
            volume = read_volume_stack(path)
            if volume.ndim < 3:
                volume = volume[np.newaxis, ...]
            if downsample > 1:
                volume = block_mean_xy(volume, downsample)
            if volume.shape != (fh_px, fw_px, z_px):
                print(
                    f"Static mosaic stitch: skipping tile {os.path.basename(path)} "
                    f"(shape {volume.shape}, expected {(fh_px, fw_px, z_px)})."
                )
                continue
            col_idx = int(round((x - min_x) / fw_mm))
            row_idx = int(round((y - min_y) / fh_mm))
            # Match the live mosaic stitch order: reverse both axes
            # (right-to-left, top-to-bottom) without rotating the tile pixels.
            col_idx = num_cols - 1 - col_idx
            row_idx = num_rows - 1 - row_idx
            y1 = row_idx * fh_px
            y2 = y1 + fh_px
            x1 = col_idx * fw_px
            x2 = x1 + fw_px
            stitched[y1:y2, x1:x2, :] = volume
            del volume
            if (idx + 1) % 5 == 0 or idx + 1 == len(resolved):
                print(f"  placed tile {idx + 1}/{len(resolved)}")
        stitched.flush()
    finally:
        try:
            stitched.flush()
        except Exception:
            pass
        try:
            del stitched
        except Exception:
            pass
    return True
def process_pending_dynamic_folders(weaver, label="post-scan dynamic stitching", deadline=None):
    """Run the offline tile stitching once for every pending sample/time folder.

    Used by PlateScan / WellScan after a scan completes, and by TimedPlateScan
    through its embedded PlateScan call (which passes the time-point deadline so
    the stitching stops at the interval boundary and resumes on the next slice).
    Requires saved per-tile volumes (or per-Y Bline stacks) and an idle GPU
    thread. With deadline=None the stitching runs to completion.
    """
    if not OFFLINE_DYNAMIC_PROCESSING_ENABLED:
        message = f"{label}: offline dynamic processing is disabled."
        print(message)
        weaver.emit_status(message)
        return message
    if deadline is None:
        deadline = time.time() + 3600.0
    processed_any = False
    while weaver.ui.RunButton.isChecked() and time.time() < deadline:
        processed = process_next_idle_dynamic_folder(weaver, deadline)
        if not processed:
            break
        processed_any = True
    if processed_any:
        message = f"{label}: stitched one or more sample/time folders."
    elif not weaver.ui.RunButton.isChecked():
        message = f"{label}: stopped by user."
    elif time.time() >= deadline:
        message = f"{label}: reached the time deadline with folders still pending."
    else:
        message = f"{label}: no pending dynamic folders."
    print(message)
    weaver.emit_status(message)
    return message
