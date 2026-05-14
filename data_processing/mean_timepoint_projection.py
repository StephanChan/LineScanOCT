import argparse
import os
import re

import numpy as np
import tifffile as TIFF


DEFAULT_SAMPLE_DIR = r"E:\IOCTData\vibration charac\5mm glass base\tissuePlateScanTest\sampleID-1"


TIME_DIR_RE = re.compile(r"^Time-(?P<time>\d+)$")
STITCHED_MEAN_RE = re.compile(r"^stitched-Mean-Y\d+-X\d+-Z\d+\.tif$")
TILE_MEAN_RE = re.compile(r"^tile-(?P<tile>\d+)-Mean-Y\d+-X\d+-Z\d+\.tif$")


def find_time_dirs(sample_dir):
    time_dirs = []
    for name in os.listdir(sample_dir):
        path = os.path.join(sample_dir, name)
        match = TIME_DIR_RE.match(name)
        if match and os.path.isdir(path):
            time_dirs.append((int(match.group("time")), path))
    time_dirs.sort(key=lambda item: item[0])
    return time_dirs


def mean_over_depth(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    return np.mean(volume, axis=2, dtype=np.float32)


def process_time_folder(time_id, time_dir, sample_dir):
    filenames = os.listdir(time_dir)

    stitched_mean_files = [
        name for name in filenames
        if STITCHED_MEAN_RE.match(name)
    ]
    if stitched_mean_files:
        stitched_mean_files.sort()
        source_name = stitched_mean_files[0]
        source_path = os.path.join(time_dir, source_name)
        volume = TIFF.imread(source_path)
        projection = mean_over_depth(volume)
        out_name = f"Time-{time_id}-stitched-Mean2D.tif"
        out_path = os.path.join(sample_dir, out_name)
        TIFF.imwrite(out_path, projection.astype(np.float32), append=False)
        print(f"Saved {out_path} from {source_path}")
        return

    tile_mean_files = []
    for name in filenames:
        match = TILE_MEAN_RE.match(name)
        if match:
            tile_mean_files.append((int(match.group("tile")), name))
    tile_mean_files.sort(key=lambda item: item[0])

    if not tile_mean_files:
        print(f"Skipped Time-{time_id}: no stitched or tile mean volumes found.")
        return

    if len(tile_mean_files) == 1:
        tile_id, source_name = tile_mean_files[0]
        source_path = os.path.join(time_dir, source_name)
        volume = TIFF.imread(source_path)
        projection = mean_over_depth(volume)
        out_name = f"Time-{time_id}-tile-{tile_id}-Mean2D.tif"
        out_path = os.path.join(sample_dir, out_name)
        TIFF.imwrite(out_path, projection.astype(np.float32), append=False)
        print(f"Saved {out_path} from {source_path}")
        return

    for tile_id, source_name in tile_mean_files:
        source_path = os.path.join(time_dir, source_name)
        volume = TIFF.imread(source_path)
        projection = mean_over_depth(volume)
        out_name = f"Time-{time_id}-tile-{tile_id}-Mean2D.tif"
        out_path = os.path.join(sample_dir, out_name)
        TIFF.imwrite(out_path, projection.astype(np.float32), append=False)
        print(f"Saved {out_path} from {source_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read mean intensity volumes for one sample across time points, "
            "average each volume over depth, and save the 2D results into the sample directory."
        )
    )
    parser.add_argument(
        "sample_dir",
        nargs="?",
        help="Path to one sample directory, e.g. .../sampleID-3",
    )
    parser.add_argument(
        "--sample-dir",
        dest="sample_dir_flag",
        default=None,
        help="Path to one sample directory, e.g. .../sampleID-3",
    )
    args = parser.parse_args()

    sample_dir_input = args.sample_dir_flag or args.sample_dir or DEFAULT_SAMPLE_DIR
    if not sample_dir_input:
        raise RuntimeError(
            "Missing sample directory. Example:\n"
            "python mean_timepoint_projection.py D:\\sampleID-3"
        )

    sample_dir = os.path.abspath(sample_dir_input)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    time_dirs = find_time_dirs(sample_dir)
    if not time_dirs:
        raise RuntimeError(f"No Time-* folders found under {sample_dir}")

    for time_id, time_dir in time_dirs:
        process_time_folder(time_id, time_dir, sample_dir)


if __name__ == "__main__":
    main()
