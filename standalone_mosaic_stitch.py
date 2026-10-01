# -*- coding: utf-8 -*-
"""Standalone original-resolution static mosaic stitcher.

Tile files saved by the OCT software store one TIFF page per Y row: page
``k`` has shape ``[X, Z]`` (X lines by Z depths), and the number of pages
equals Y pixels. ``read_volume_stack`` reconstructs each tile as
``[Y, X, Z]``.

This script builds the same layout the app's in-memory/offline stitcher
produces (through the shared ``mosaic_geometry`` module: physical stage offsets,
mirrored axes, overlap strips cross-faded, no tile rotation) WITHOUT allocating
the whole mosaic volume in RAM. It creates the output directly on disk as a
memory-mapped BigTIFF of shape ``[Y_total, X_total, Z]`` and copies one tile
volume into it at a time, so peak RAM is roughly one tile volume (~160 MB),
independent of mosaic size.

Usage:
    python standalone_mosaic_stitch.py "E:\\IOCTData\\BJRcellcluster\\20XNomiror\\sampleID-1\\Time-2"
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import tifffile as TIFF

from mosaic_geometry import blend_paste, build_layout, new_weight_map, pixel_size_mm


def load_manifest(folder):
    """Return the parsed tile_positions.json manifest (exit with an error)."""
    path = os.path.join(folder, "tile_positions.json")
    if not os.path.isfile(path):
        sys.exit("tile_positions.json not found in: " + str(folder))
    try:
        with open(path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, ValueError) as error:
        sys.exit("Could not read tile_positions.json: " + str(error))
    records = manifest.get("tiles")
    if not isinstance(records, list) or not records:
        sys.exit("tile_positions.json contains no tile records.")
    return manifest


def stitched_output_path(folder, y_px, x_px, z_px):
    filename = "stitched-Y{0}-X{1}-Z{2}.tif".format(y_px, x_px, z_px)
    return os.path.join(folder, filename)


def read_tile_volume(path):
    """Read a tile TIFF back as one [Y, X, Z] volume (same logic as the app)."""
    with TIFF.TiffFile(path) as tif:
        if len(tif.pages) <= 1:
            return tif.pages[0].asarray()
        if len(tif.series) == len(tif.pages):
            return np.stack([page.asarray() for page in tif.pages])
        return tif.series[0].asarray()
def main():
    parser = argparse.ArgumentParser(
        description="Offline original-resolution static mosaic stitching."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=r"E:\IOCTData\BJRcellcluster\20XNomiror\sampleID-1\Time-2",
        help="Sample/time folder containing tile-*.tif and tile_positions.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing stitched-Y...-X...-Z....tif file.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Optional output folder (defaults to the input folder).",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    out_folder = os.path.abspath(args.output_folder) if args.output_folder else folder
    manifest = load_manifest(folder)
    records = manifest["tiles"]

    # ---- field-of-view geometry from the manifest -------------------------
    fw_mm = float(records[0].get("x_length_mm") or 0.0)
    fh_mm = float(records[0].get("y_length_mm") or 0.0)
    fw_px = int(records[0].get("x_pixels") or 0)
    fh_px = int(records[0].get("y_pixels") or 0)
    z_px = int(records[0].get("z_pixels") or 0)
    if not (fw_mm and fh_mm and fw_px and fh_px and z_px):
        sys.exit(
            "Manifest record is missing x_length_mm / y_length_mm / x_pixels / "
            "y_pixels / z_pixels."
        )

    entries = []
    for record in records:
        filename = record.get("tile_filename")
        if not filename:
            sys.exit("Tile record missing tile_filename: " + str(record))
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            sys.exit("Tile file not found: " + str(path))
        try:
            x = float(record["stage_x_mm"])
            y = float(record["stage_y_mm"])
        except (KeyError, TypeError, ValueError):
            sys.exit("Tile record missing stage_x_mm/stage_y_mm: " + str(record))
        entries.append((x, y, path))

    # ---- one shared placement rule (mosaic_geometry) ----------------------
    # Tiles go to their physical stage offset and the canvas is the physical
    # extent of the scan, so the FOV overlap never folds two tiles into one cell
    # (the old round((pos - min) / fov) did exactly that at 10% overlap).
    # The pixel size comes from the scan step (x_step_um / y_step_um); the FOV
    # size divided by the pixel count is only the fallback.
    x_step_um = manifest.get("x_step_um")
    y_step_um = manifest.get("y_step_um")
    mm_per_px_x = pixel_size_mm(x_step_um, fw_mm, fw_px)
    mm_per_px_y = pixel_size_mm(y_step_um, fh_mm, fh_px)
    layout = build_layout(
        [(entry[0], entry[1]) for entry in entries],
        fw_px,
        fh_px,
        mm_per_px_x,
        mm_per_px_y,
    )
    problems = layout.problems()
    if problems:
        sys.exit("Mosaic layout is inconsistent: " + "; ".join(problems))
    placements = [(entry[2], paste) for entry, paste in zip(entries, layout.placements)]

    mosaic_y = layout.height_px
    mosaic_x = layout.width_px
    out_path = stitched_output_path(out_folder, mosaic_y, mosaic_x, z_px)

    print(
        "Input        : {0}\n".format(folder)
        + "Tiles        : {0}\n".format(len(entries))
        + "Grid         : {0} cols x {1} rows\n".format(layout.cols, layout.rows)
        + "Geometry     : {0}\n".format(layout.describe())
        + "Pixel size   : {0:.4f}/{1:.4f} um/px ({2})\n".format(
            mm_per_px_x * 1000.0,
            mm_per_px_y * 1000.0,
            "scan step" if (x_step_um and y_step_um) else "FOV size / pixels",
        )
        + "Tile size    : {0} (Y) x {1} (X) x {2} (Z)\n".format(fh_px, fw_px, z_px)
        + "Mosaic size  : {0} (Y) x {1} (X) x {2} (Z)\n".format(mosaic_y, mosaic_x, z_px)
        + "Output       : {0}".format(out_path)
    )
    # ---- disk-space estimate from the first tile's dtype -------------------
    with TIFF.TiffFile(placements[0][0]) as probe:
        actual_pages = len(probe.pages)
        print("First tile pages (Y rows): {0} (manifest y_pixels={1}).".format(
            actual_pages, fh_px))
        if actual_pages < fh_px:
            print("WARNING: tile has fewer Y pages than the manifest expects.")
            fh_px = actual_pages
            mosaic_y = num_rows * fh_px
        if actual_pages <= 0:
            sys.exit("No TIFF pages found in the first tile.")
        page0 = probe.pages[0].asarray()
        page_shape = tuple(int(s) for s in page0.shape)
        print("First tile page shape: {0} (expected {1}).".format(
            page_shape, (fw_px, z_px)))
        if page_shape != (fw_px, z_px):
            sys.exit(
                "Tile page shape {0} does not match expected [X={1}, Z={2}]. "
                "Aborting to avoid producing a misaligned mosaic.".format(
                    page_shape, fw_px, z_px)
            )
        dtype = page0.dtype
        del page0

    plane_bytes = mosaic_x * z_px * int(dtype.itemsize)
    total_bytes = plane_bytes * mosaic_y
    os.makedirs(out_folder, exist_ok=True)
    free_bytes = shutil.disk_usage(out_folder).free
    print(
        "Estimated output size : {0:.2f} GB\n".format(total_bytes / 1e9)
        + "Free space on output drive : {0:.2f} GB".format(free_bytes / 1e9)
    )
    if total_bytes > free_bytes:
        sys.exit("Not enough free disk space for the stitched output.")
    if os.path.exists(out_path):
        if not args.force:
            sys.exit("Output already exists: {0} (use --force to overwrite).".format(out_path))
        print("Overwriting existing output (--force).")
        try:
            os.remove(out_path)
        except PermissionError:
            sys.exit(
                "Cannot overwrite {0}: the file is open/locked by another program "
                "(e.g. an image viewer or a previous stitcher run still running). "
                "Close it and re-run.".format(out_path)
            )
    os.makedirs(out_folder, exist_ok=True)

    print("Stitching tiles (memory-mapped BigTIFF output)...")
    stitched = TIFF.memmap(
        out_path,
        shape=(mosaic_y, mosaic_x, z_px),
        dtype=dtype,
        bigtiff=True,
    )
    # Normalised cross-fade weight map (raised cosine ramps, see blend_paste).
    weights = new_weight_map((mosaic_y, mosaic_x))
    try:
        for idx, (path, paste) in enumerate(placements):
            volume = read_tile_volume(path)
            if volume.ndim < 3:
                volume = volume[np.newaxis, ...]
            if volume.shape != (fh_px, fw_px, z_px):
                print(
                    "Skipping tile {0}: shape {1}, expected {(fh_px, fw_px, z_px)}.".format(
                        os.path.basename(path), tuple(volume.shape)
                    )
                )
                continue
            blend_paste(stitched, volume, paste, weights=weights)
            del volume
            if (idx + 1) % 5 == 0 or idx + 1 == len(placements):
                print("  placed tile {0}/{1}".format(idx + 1, len(placements)))
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

    print("Done. Stitched mosaic written to:")
    print("  {0}".format(out_path))


if __name__ == "__main__":
    main()
