import json
import os
import re
import time

import numpy as np
import tifffile as TIFF

from mosaic_geometry import blend_paste, layout_for_ui, new_weight_map
import shading_correction


def _mosaic_layout(weaver, positions, fw_px, fh_px, fw_mm, fh_mm, downsample=1,
                   missing=()):
    """Shared mosaic layout for the offline stitchers (see mosaic_geometry)."""
    return layout_for_ui(
        weaver.ui,
        positions,
        fw_px,
        fh_px,
        fw_mm,
        fh_mm,
        downsample=downsample,
        missing=missing,
    )


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
# Dynamic colour tiles are stored as three per-channel float16 volumes (H, S, V)
# instead of a rendered uint8 RGB (see shading_correction / FileNaming).
TILE_DYN_CHANNEL_RE = re.compile(
    r"^tile-(?P<tile>\d+)-Dyn(?P<channel>[HSV])-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
)
DYNAMIC_CHANNELS = ("H", "S", "V")
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


def stitched_channel_output_path(folder_path, channel, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"stitched-Dyn{channel}-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def stitched_rgb_output_path(folder_path, volume_shape):
    ypix, xpix, zpix = volume_shape
    filename = f"stitched-DynRGB-Y{ypix}-X{xpix}-Z{zpix}.tif"
    return os.path.join(folder_path, filename)


def collect_tile_dynamic_channels(folder_path, tile_count):
    """``{tile_id: {"H": path, "S": path, "V": path}}`` for the colour tiles.

    Returns ``{}`` when the folder predates the H/S/V layout (or is incomplete), so
    the colour mosaic is simply skipped instead of failing the whole stitching.
    """
    found = {}
    if not os.path.isdir(folder_path):
        return found
    for filename in os.listdir(folder_path):
        match = TILE_DYN_CHANNEL_RE.match(filename)
        if match is None:
            continue
        tile_id = int(match.group("tile"))
        if tile_id < 1 or tile_id > int(tile_count):
            continue
        found.setdefault(tile_id, {})[match.group("channel")] = os.path.join(
            folder_path, filename
        )
    complete = {}
    for tile_id, channel_paths in found.items():
        if all(channel in channel_paths for channel in DYNAMIC_CHANNELS):
            complete[tile_id] = channel_paths
    return complete


def render_stitched_dynamic_rgb(folder_path, stitched_shape, channel_paths,
                                block_rows=32, ranges=None):
    """Render one uint8 RGB mosaic from the stitched H/S/V volumes.

    The colour is rendered once, after stitching, from the already corrected H/S/V
    mosaics: reading them back in Y blocks keeps the RAM cost flat, and both the
    three float16 channel mosaics and the rendered uint8 RGB stay on disk.

    ``ranges`` are the normalisation windows of the stored physical channels
    (``dynamic_hsv_ranges`` in ``tile_positions.json``); without them the fallback
    constants are used, which may not match what the operator saw while scanning.
    """
    height, width, depth = stitched_shape
    rgb_out = stitched_rgb_output_path(folder_path, stitched_shape)
    rgb = TIFF.memmap(
        rgb_out, shape=(height, width, depth, 3), dtype=np.uint8, bigtiff=True,
        photometric="rgb",
    )
    try:
        for start in range(0, height, block_rows):
            stop = min(height, start + block_rows)
            hsv = np.empty((stop - start, width, depth, 3), dtype=np.float32)
            for index, channel in enumerate(DYNAMIC_CHANNELS):
                block = TIFF.imread(
                    channel_paths[channel], key=range(start, stop)
                )
                hsv[..., index] = np.asarray(block, dtype=np.float32)
            rgb[start:stop] = shading_correction.hsv_to_rgb(hsv, ranges=ranges)
        rgb.flush()
    finally:
        try:
            del rgb
        except Exception:
            pass
    print("Dynamic mosaic colour: rendered " + os.path.basename(rgb_out))
    return rgb_out


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

    # Optional colour product: H / S / V per tile (new layout only).
    channel_paths = collect_tile_dynamic_channels(folder_path, tile_count)
    if channel_paths and len(channel_paths) != len(tile_paths):
        print(
            "Dynamic mosaic colour: incomplete H/S/V tiles "
            f"({len(channel_paths)}/{len(tile_paths)}); the colour mosaic is skipped."
        )
        channel_paths = {}

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

    # One shared placement rule (mosaic_geometry): every tile goes to its physical
    # stage offset, the canvas is the physical extent of the scan and overlapping
    # strips are cross-faded.  The old round((pos - min) / fov) grid assumed zero
    # overlap and silently folded tiles together once the FOVs overlapped.
    missing_tiles = [
        tile_id - 1
        for tile_id in range(1, len(sample_locations) + 1)
        if tile_id not in tile_paths
    ]
    layout = _mosaic_layout(
        weaver,
        [(loc.x, loc.y) for loc in sample_locations],
        fw_px,
        fh_px,
        fw_mm,
        fh_mm,
        downsample=downsample,
        missing=missing_tiles,
    )
    layout_problems = layout.problems()
    if layout_problems:
        print("Dynamic stitch: mosaic layout: " + "; ".join(layout_problems))

    stitched_shape = (layout.height_px, layout.width_px, z_px)
    dyn_out = stitched_dynamic_output_path(folder_path, stitched_shape)
    mean_out = stitched_mean_output_path(folder_path, stitched_shape)
    print(
        f"Dynamic mosaic stitch: {len(tile_paths)} tile(s) -> "
        f"Dyn/Mean shape={stitched_shape[0]}x{stitched_shape[1]}x{stitched_shape[2]}."
    )
    print("Dynamic mosaic geometry: " + layout.describe())

    stitched_dyn = TIFF.memmap(
        dyn_out, shape=stitched_shape, dtype=dyn_dtype, bigtiff=True
    )
    stitched_mean = TIFF.memmap(
        mean_out, shape=stitched_shape, dtype=mean_dtype, bigtiff=True
    )
    # Colour mosaics: three float16 channel volumes, cross-faded like Dyn/Mean.
    stitched_channels = {}
    if channel_paths:
        for channel in DYNAMIC_CHANNELS:
            channel_out = stitched_channel_output_path(folder_path, channel, stitched_shape)
            stitched_channels[channel] = TIFF.memmap(
                channel_out, shape=stitched_shape, dtype=np.float16, bigtiff=True
            )
    # One normalised cross-fade weight map per stitched product (they are all
    # blended in the same tile order, but each product needs its own map: sharing
    # one would advance it more than once per tile).
    weight_maps = {
        "dyn": new_weight_map(stitched_shape[:2]),
        "mean": new_weight_map(stitched_shape[:2]),
    }
    for channel in stitched_channels:
        weight_maps[channel] = new_weight_map(stitched_shape[:2])
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

            paste = layout.placements[tile_id - 1]
            blend_paste(stitched_dyn, dyn_volume, paste, weights=weight_maps["dyn"])
            blend_paste(stitched_mean, mean_volume, paste, weights=weight_maps["mean"])
            del dyn_volume, mean_volume
            if stitched_channels and tile_id in channel_paths:
                for channel, mapped in stitched_channels.items():
                    volume = read_volume_stack(channel_paths[tile_id][channel])
                    if volume.ndim < 3:
                        volume = volume[np.newaxis, ...]
                    if volume.shape == (fh_px, fw_px, z_px):
                        blend_paste(mapped, np.asarray(volume, np.float32), paste,
                                    weights=weight_maps[channel])
                    else:
                        print(
                            f"Dynamic mosaic colour: skipping {channel} of tile-{tile_id} "
                            f"({volume.shape})."
                        )
                    del volume
            if tile_id % 5 == 0 or tile_id == len(sample_locations):
                print(f"  placed tile {tile_id}/{len(sample_locations)}")
        stitched_dyn.flush()
        stitched_mean.flush()
        for mapped in stitched_channels.values():
            mapped.flush()
    finally:
        for mapped in [stitched_dyn, stitched_mean] + list(stitched_channels.values()):
            try:
                mapped.flush()
            except Exception:
                pass
            try:
                del mapped
            except Exception:
                pass
    if stitched_channels:
        # Render the colour once, after stitching, from the corrected channels,
        # using the normalisation windows the scan was displayed with.
        try:
            ranges = None
            manifest_path = os.path.join(folder_path, "tile_positions.json")
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    ranges = json.load(handle).get("dynamic_hsv_ranges")
            print(
                "Dynamic mosaic colour: ranges "
                + (str(ranges) if ranges else "(fallback constants)")
            )
            render_stitched_dynamic_rgb(
                folder_path, stitched_shape,
                {channel: stitched_channel_output_path(folder_path, channel, stitched_shape)
                 for channel in DYNAMIC_CHANNELS},
                ranges=ranges,
            )
        except Exception as error:
            print("Dynamic mosaic colour: RGB rendering failed: {0}".format(error))
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

    # One shared placement rule (mosaic_geometry): physical offsets + physical
    # canvas + cross-faded overlap strips (see write_stitched_idle_outputs).
    layout = _mosaic_layout(
        weaver,
        [(entry[0], entry[1]) for entry in resolved],
        fw_px,
        fh_px,
        fw_mm,
        fh_mm,
        downsample=downsample,
    )
    layout_problems = layout.problems()
    if layout_problems:
        print("Static stitch: mosaic layout: " + "; ".join(layout_problems))

    stitched_shape = (layout.height_px, layout.width_px, z_px)
    out_path = stitched_static_output_path(folder_path, stitched_shape)
    print(
        f"Static mosaic stitch: {len(resolved)} tile(s) -> "
        f"shape={stitched_shape[0]}x{stitched_shape[1]}x{stitched_shape[2]} "
        f"({stitched_shape[0]*stitched_shape[1]*stitched_shape[2]*np.dtype(first_dtype).itemsize/1e9:.2f} GB)."
    )
    print("Static mosaic geometry: " + layout.describe())

    # Create the empty mosaic directly on disk as a memory-mapped BigTIFF.
    stitched = TIFF.memmap(
        out_path,
        shape=stitched_shape,
        dtype=first_dtype,
        bigtiff=True,
    )
    # Normalised cross-fade weight map of the static mosaic (see blend_paste).
    static_weights = new_weight_map(stitched_shape[:2])
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
            paste = layout.placements[idx]
            blend_paste(stitched, volume, paste, weights=static_weights)
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
