# -*- coding: utf-8 -*-
"""
cellpose_organoid_segmentation.py  (standalone, Spyder-ready)

Segment organoids in a 3D OCT volume (TIFF stack saved by the LineScanOCT
software) with Cellpose.

OCT conventions used here (same as standalone_mosaic_stitch.py /
mosaic_correction.py):

* A saved volume is ``[Y, X, Z]`` - one TIFF page per Y row and page ``k``
  has shape ``[X, Z]`` (X lines by Z depths). File names follow
  ``tile-{n}-Y{y}-X{x}-Z{z}.tif`` (see FileNaming.py).
* Cellpose 4 wants ``[Z, Y, X]`` for 3D and returns ``masks`` as
  ``[Z, Y, X]`` (see https://cellpose.readthedocs.io - "3D segmentation").

So the script transposes to ``[Z, Y, X]`` before segmentation and transposes
the labels back to ``[Y, X, Z]`` on write, which means the mask TIFF overlays
directly on the input TIFF in the app, ImageJ, or any of the other
``data_processing`` viewers.

Pipeline
--------
1. read the volume (memory-mapped when possible so a big tile stays cheap),
2. optional Y/X crop (``--crop``) and depth crop (``--depth-range``),
3. OCT display compression: ``20*log10`` + dynamic-range clip, then a
   percentile stretch. This is the step that matters most for OCT - raw
   amplitude data is dominated by the surface reflection and by speckle, and
   Cellpose's own 1-99 percentile normalisation cannot rescue that,
4. optional de-speckle smoothing (``--smooth-xy``, ``--smooth-z``),
5. optional integer downsampling (``--downsample``) for very large volumes,
6. Cellpose inference (``--mode 3d`` | ``2d-stitch`` | ``projection``),
7. labels written back in the input orientation + per-organoid statistics
   CSV + QC figure + JSON summary, written into
   ``<input folder>/segmentation_results`` (``--output-dir``, see
   :func:`output_paths`).

Modes
-----
``3d``          ``do_3D=True``: 3D flows in/out of the volume. Needs
                ``--xy-pixel-um``/``--z-step-um`` so ``anisotropy`` (the
                Z-to-XY sampling ratio) is right; this is the default.
``2d-stitch``   Cellpose runs per Z plane and stitches planes into 3D labels
                (``stitch_threshold``). More forgiving when Z is coarse or
                the volume is noisy.
``projection``  2D segmentation of the en-face (XY) maximum/mean projection
                through the depth band - the fast way to count organoids and
                measure their XY cross-section.

Requires cellpose (4.x recommended) + torch in the SAME environment:
    "<env>\\python.exe" -m pip install "cellpose[gui]"

On Windows, pip-installing Cellpose into a conda-forge environment leaves two
sets of OpenMP/MKL DLLs in play (torch\lib versus <env>\Library\bin), which can
make `import torch` fail with
    OSError: [WinError 127] ... "torch\lib\shm.dll" or one of its dependencies.
This module fixes the DLL search order at import time, before it imports numpy
(see DLL_BOOTSTRAP_ENABLED and torch_dll_report further down), and if the current
process still cannot load torch - a long-lived Spyder kernel that already loaded
the conda-forge OpenMP/MKL builds cannot be repaired from the inside - it simply
re-runs the pipeline in a fresh interpreter automatically (see
DEFAULT_USE_SUBPROCESS_FALLBACK). So pressing Run in an old kernel still works;
there is no need to restart anything first.

Run in Spyder: open the file and press Run (F5). Edit DEFAULT_* below, or
pass arguments on the command line, e.g.
    python cellpose_organoid_segmentation.py "...\\tile-1-Y575-X1016-Z76.tif"
    python cellpose_organoid_segmentation.py "...\\tile-1-...tif" --mode 2d-stitch --downsample 2
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows DLL bootstrap (must run before numpy/scipy/torch are imported)
# ---------------------------------------------------------------------------
# ``pip install cellpose`` pulls in torch, and torch ships its own Intel OpenMP
# (torch\lib\libiomp5md.dll, ~1.5 MB), MKL and CUDA DLLs. A conda-forge
# environment ships *different* builds of the same DLL names in
# <env>\Library\bin (libiomp5md.dll ~0.2 MB plus mkl_*.dll). Windows resolves a
# dependency by file name and reuses whichever copy was loaded first, so if
# numpy/scipy/MKL load before torch - which is what happens under Spyder, where
# the console has usually imported them already - torch's DLLs bind to the
# conda-forge OpenMP build and fail with
#     OSError: [WinError 127] ... "torch\lib\shm.dll" or one of its dependencies
# WinError 127 means "the specified procedure could not be found", i.e. Windows
# found *a* DLL of that name but it is the wrong build - not that a file is
# missing (that would be WinError 126).
#
# The fix has two halves: put torch's own DLL directories in front of the conda
# ones, and import torch here - before numpy/scipy/matplotlib can load a
# conflicting OpenMP/MKL build into this process. Set DLL_BOOTSTRAP_ENABLED =
# False to skip all of it; call torch_dll_report() in the console to inspect the
# search order when something still refuses to load.
DLL_BOOTSTRAP_ENABLED = True
_DLL_DIRECTORY_HANDLES = []  # os.add_dll_directory handles must stay referenced

# Names that exist both in torch\lib and in <env>\Library\bin: the DLL
# collisions this bootstrap is about.
DLL_CONFLICT_NAMES = (
    "libiomp5md.dll",
    "mkl_rt.2.dll",
    "torch_cpu.dll",
    "c10.dll",
    "vcruntime140.dll",
    "msvcp140.dll",
)


def torch_dll_directories():
    """Return the DLL directories torch needs, most important first.

    Empty list when torch is not installed. ``importlib.util.find_spec`` locates
    torch without importing it, so this is safe to call before anything else.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return []

    torch_root = Path(spec.origin).parent
    directories = [torch_root / "lib"]

    # CUDA/other dependencies are shipped as separate "nvidia-*" wheels, either
    # as nvidia/<package>/bin or, for CUDA 13, nvidia/<package>/lib.
    search_roots = {str(torch_root.parent)}
    try:
        import site as site_module

        search_roots.update(site_module.getsitepackages())
    except Exception:
        pass
    for root in sorted(search_roots):
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for package in sorted(nvidia_root.iterdir()):
            for sub in ("bin", "lib"):
                candidate = package / sub
                if candidate.is_dir():
                    directories.append(candidate)
    return [path for path in directories if path.is_dir()]


def bootstrap_torch_dlls(enabled=DLL_BOOTSTRAP_ENABLED):
    """Put torch's DLL directories ahead of the conda environment (Windows).

    Returns a dict describing what happened, kept in ``TORCH_DLL_BOOTSTRAP`` and
    stored in the run summary JSON. Safe to call repeatedly and a no-op when
    disabled, on other platforms, or when torch is not installed.
    """
    if not enabled:
        return {"applied": False, "reason": "disabled (DLL_BOOTSTRAP_ENABLED = False)"}
    if os.name != "nt":
        return {"applied": False, "reason": "not Windows"}

    directories = torch_dll_directories()
    if not directories:
        return {"applied": False, "reason": "torch not installed"}

    # 1. move the directories to the front of PATH so newly loaded DLLs win
    entries = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    lowered = {str(path).lower() for path in directories}
    os.environ["PATH"] = os.pathsep.join(
        [str(path) for path in directories]
        + [part for part in entries if part.lower() not in lowered]
    )

    # 2. register them with the loader for DLLs opened through ctypes/torch
    registered = []
    if hasattr(os, "add_dll_directory"):
        for path in directories:
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
                registered.append(str(path))
            except OSError:
                pass

    # 3. two OpenMP runtimes in one process would otherwise abort with
    #    "OMP: Error #15"; a symbol mismatch (WinError 127) is what the
    #    ordering above fixes
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    result = {
        "applied": True,
        "directories": registered or [str(path) for path in directories],
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
    }

    # 4. load torch now, while no other OpenMP/MKL build is in the process
    try:
        import torch  # noqa: F401

        result["torch_import"] = "ok"
    except Exception as error:
        result["torch_import"] = f"{type(error).__name__}: {error}"
        # Measured behaviour on this machine: a fresh interpreter can import torch,
        # but any process that imported numpy/scipy/matplotlib first cannot -
        # those load the conda-forge OpenMP/MKL DLLs from Library\bin, and torch's
        # DLLs then bind to an incompatible copy (a loaded DLL cannot be swapped).
        # Spyder's kernel imports numpy at startup, so restarting the kernel does
        # NOT help; the pipeline therefore re-runs itself in a fresh process.
        print(
            "note: torch cannot be imported in this process because numpy/scipy "
            f"were imported first ({type(error).__name__}: {error}).\n"
            "      This is expected inside a Spyder kernel and needs no action: a "
            "run will continue automatically in a fresh process, where torch loads "
            "first. torch_dll_report() shows the DLL search order."
        )
    return result


def torch_dll_error_message(error):
    """Explain a Windows torch DLL failure and how to resolve it."""
    return (
        "Cellpose/torch could not be loaded because of a Windows DLL conflict:\n"
        f"    {type(error).__name__}: {error}\n\n"
        "WinError 127 ('the specified procedure could not be found') means "
        "Windows found a DLL of the right name but of the wrong build. On this "
        "machine it is triggered by import order: a fresh interpreter loads torch "
        "fine, but a process that imported numpy/scipy/matplotlib first cannot, "
        "because those load the conda-forge OpenMP/MKL copies from "
        "<env>\\Library\\bin and torch's DLLs then bind to an incompatible copy "
        "(a loaded DLL can never be swapped).\n"
        "What to do:\n"
        "  1. Nothing, in Spyder: the kernel imports numpy at startup, so in-process "
        "torch is impossible there and restarting it does not help - the pipeline "
        "re-runs itself in a fresh process automatically (DEFAULT_USE_SUBPROCESS_"
        "FALLBACK).\n"
        "  2. Run the script with run_cellpose_organoid_segmentation.bat (or any "
        "plain interpreter) - that process starts fresh, so torch loads first.\n"
        "  3. For a cleaner setup use a dedicated environment that has no "
        "conda-forge OpenMP/MKL next to torch:\n"
        "        conda create -n cellpose_env python=3.11 -y\n"
        "        conda activate cellpose_env\n"
        "        pip install \"cellpose[gui]\"\n"
        "  4. torch_dll_report() prints the DLL search order and the conflicting "
        "copies."
    )


def torch_dll_report(limit=12):
    """Print the Windows DLL search order and the conflicting copies found.

    Call it in the Spyder console when torch refuses to import:

        torch_dll_report()
    """
    if os.name != "nt":
        print("torch_dll_report: only meaningful on Windows")
        return {}
    print("torch DLL bootstrap:", TORCH_DLL_BOOTSTRAP)
    print("torch DLL directories (priority order):")
    for path in torch_dll_directories() or ["  (torch not found)"]:
        print(f"    {path}")

    directories = [("torch", path) for path in torch_dll_directories()]
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isdir(entry):
            continue
        if any(os.path.isfile(os.path.join(entry, name))
               for name in DLL_CONFLICT_NAMES):
            directories.append(("PATH", Path(entry)))

    found = {}
    print("\nCopies of the conflicting DLL names, in search order "
          "(the copy loaded first wins for the whole process):")
    for source, directory in directories[: int(limit)]:
        present = [name for name in DLL_CONFLICT_NAMES
                   if (directory / name).is_file()]
        if not present:
            continue
        found[str(directory)] = present
        print(f"  [{source}] {directory}")
        for name in present:
            megabytes = (directory / name).stat().st_size / 1e6
            print(f"      {name:20s} {megabytes:8.1f} MB")
    if not found:
        print("  (none found)")
    return found


def silence_sparse_invariant_warning(torch_module):
    """Opt out of torch's sparse-invariant checks, silencing their warning.

    Cellpose builds a ``torch.sparse_coo_tensor`` for its 3D neighbour graph
    (``cellpose/dynamics.py``), and torch warns once per process that those
    checks are "implicitly disabled ... explicitly opt in or out". They are off
    by default anyway, so the message is noise in a long run; ``disable()``
    states it explicitly - exactly what the warning asks for - and changes
    nothing else. Enabling them instead would add overhead to every sparse
    operation and could turn a large mosaic into a hard error.

    Returns the resulting state, or ``None`` when this torch has no such switch.
    """
    try:
        invariants = torch_module.sparse.check_sparse_tensor_invariants
        invariants.disable()
        return bool(invariants.is_enabled())
    except Exception:
        return None


TORCH_DLL_BOOTSTRAP = bootstrap_torch_dlls()

import argparse  # noqa: E402  (these imports follow the DLL bootstrap by design)
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tifffile as TIFF  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402



# ---------------------------------------------------------------------------
# Settings (edit these, then press Run / F5 in Spyder)
# ---------------------------------------------------------------------------
# A single TIFF, or a folder that is scanned for *DEFAULT_PATTERN (mosaics are
# named stitched-*.tif by the app; tile files are tile-*.tif).
DEFAULT_INPUT_PATH = (
    r"E:\IOCTData\BJRcellcluster\0922\sampleID-1\Time-2\stitched-bgcorr-Y2272-X2515-Z76.tif"
)
DEFAULT_PATTERN = "stitched-*.tif"
DEFAULT_RECURSIVE = False  # True -> also scan sub-folders (rglob)
# Where the outputs go. A relative name is created under each input file's own
# folder (the default, so results stay next to the data they came from but out of
# the way), an absolute path is used as-is, and "" writes next to the input file.
# The folder is created when missing and reused when it already exists: the file
# names are derived from the input stem, so a re-run overwrites its own outputs.
DEFAULT_OUTPUT_DIR = "segmentation_results"
DEFAULT_SHOW_ORGANOID_TABLE = True  # print the per-organoid table at the end

# Window applied to every run (also settable per console call / on the command
# line). None = use the whole volume.
DEFAULT_CROP = None  # (y0, y1, x0, x1) px, e.g. (100, 400, 900, 1500)
DEFAULT_DEPTH_RANGE = (1,63)  # (z0, z1) depth planes, e.g. (5, 70) to skip the
                            # surface reflection and the noise floor

DEFAULT_MODE = "3d"  # "3d" | "2d-stitch" | "projection"
DEFAULT_MODEL = "cpsam_v2"  # Cellpose 4 default (Cellpose-SAM); v3: "cyto3"
# Where the pretrained weights live. Deliberately OUTSIDE the git repositories
# (D:\LineScanOCT is not a repo, D:\LineScanOCT\LineScanOCT is), so the ~1.2 GB
# cpsam_v2 file can never end up in a commit. huggingface.co is not reachable
# from this machine, so the downloader next to the weights fetches them through
# https://hf-mirror.com instead:  download_cpsam_v2.ps1
DEFAULT_MODEL_DIR = r"D:\LineScanOCT\cellpose_models"
DEFAULT_GPU = True
DEFAULT_DIAMETER = 0.0  # 0 = let Cellpose decide (cpsam is scale-invariant)
DEFAULT_MIN_SIZE = 5000  # Cellpose min ROI size, in voxels (3D) / pixels (2D)
DEFAULT_FLOW_THRESHOLD = 0.4
DEFAULT_CELLPROB_THRESHOLD = 0.5
DEFAULT_NITER = 0  # 0 = Cellpose default (scales with ROI size)
DEFAULT_BATCH_SIZE = 32
DEFAULT_STITCH_THRESHOLD = 0.5  # "2d-stitch" mode only
DEFAULT_PROJECTION = "max"  # "max" | "mean" ("projection" mode only)

# Sampling. 20X objective on the 9.0 um (Daheng / PhotonFocus) camera gives
# 9.0 / 10.8 = 0.833 um in XY; HardwareSpecs.DEFAULT_AXIAL_PIXEL_SIZE_UM = 4.4.
# Set either of these to 0 to disable the corresponding unit conversion and
# fall back to pixel units in the statistics CSV.
DEFAULT_XY_PIXEL_UM = 1.666
DEFAULT_Z_STEP_UM = 4.4
DEFAULT_ANISOTROPY = 0.0  # >0 overrides z_step/xy_pixel

# OCT display compression / preprocessing.
DEFAULT_COMPRESSION = "db"  # "db" | "log" | "none"
DEFAULT_DYNAMIC_RANGE_DB = 35.0
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.9
DEFAULT_SMOOTH_XY = 1.0  # Gaussian sigma in XY pixels; 0 = off
DEFAULT_SMOOTH_Z = 0.5  # Gaussian sigma in Z pixels; 0 = off
DEFAULT_DOWNSAMPLE = 4  # integer factor; 1 = off

# Where the four preprocessing steps (compression, percentile stretch,
# de-speckle, projection) run: "auto" uses the GPU when torch sees a CUDA
# device, "cpu" always uses numpy/scipy, "gpu" demands the GPU and prints why it
# fell back when it cannot. The GPU path is the same arithmetic in torch; see
# the "GPU preprocessing" section and preprocess_device_report().
DEFAULT_PREPROCESS_DEVICE = "auto"

# Stitched mosaics. A 2272 x 2515 x 76 mosaic is 434 M voxels and one Cellpose
# 3D call needs roughly 30 bytes per voxel of working set (~14 GB), which does
# not fit for the larger mosaics. With TILE_Y = 0 the script estimates the
# working set and, when it does not fit, automatically processes the mosaic in
# Y strips (each strip is read page-by-page from the TIFF), then merges the
# labels that a strip boundary cut in two.
DEFAULT_TILE_Y = 256  # 0 = auto (whole volume if it fits, else Y strips)
DEFAULT_TILE_OVERLAP_Y = 64  # extra Y context per strip; keep it >= the organoid
                              # diameter, otherwise a large organoid that lies
                              # across a strip boundary could be split in two
DEFAULT_MAX_RAM_GIGABYTES = 24.0  # working-set budget for one Cellpose call

# Post-filter: drop labels smaller than this from the mask and the CSV.
DEFAULT_MIN_VOLUME_VOXELS = 50000
DEFAULT_MAX_GIGABYTES = 20.0  # refuse to load bigger inputs than this
DEFAULT_QC_RESOLUTION = 140  # dpi of the QC PNG
DEFAULT_SHOW_FIGURES = False  # True -> also display the QC figure (Spyder Plots)
DEFAULT_UPDATE_QC = False  # True -> only rebuild the QC figure(s) from the
                           # outputs already on disk (no segmentation at all)

# Keep False for Spyder/IPython: the DEFAULT_* block above is then used and no
# command-line arguments are required. run_cellpose_organoid_segmentation.bat
# passes --use-command-line-args, which switches parsing on for one run.
USE_COMMAND_LINE_ARGS = False

# A long-lived Spyder kernel that has already loaded numpy/Qt/MKL cannot import
# torch any more (see the DLL bootstrap at the top of this file). Since a fresh
# interpreter has no such state, the script then re-runs itself in a new process
# automatically. Set False (or pass --no-subprocess-fallback) to keep everything
# in the current process instead.
DEFAULT_USE_SUBPROCESS_FALLBACK = True
SUBPROCESS_GUARD_VAR = "CP_ORGANOID_IN_SUBPROCESS"

# Figures are always written as PNG files. With DEFAULT_SHOW_FIGURES = True the
# QC figure is also displayed, so the backend is left untouched (use this in
# Spyder if you want the Plots pane / a Qt window); otherwise Agg is forced so
# nothing tries to open a window.
if not DEFAULT_SHOW_FIGURES:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (imported after the backend choice)



# ---------------------------------------------------------------------------
# Input / output helpers
# ---------------------------------------------------------------------------
def natural_sort_key(text):
    """Sort key that orders tile-2 before tile-10 (used for folder input)."""
    import re

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


# Folder name of the default results folder, used by is_own_output() so that a
# recursive folder scan never treats this script's own outputs as input volumes.
_OUTPUT_FOLDER_NAME = Path(DEFAULT_OUTPUT_DIR).name.lower() if str(DEFAULT_OUTPUT_DIR).strip() else ""


def is_own_output(file_path):
    """True for a file written by this script, so a folder scan skips it.

    Every output carries the ``cellpose_`` prefix (``cellpose_mask_*``,
    ``cellpose_organoids_*``, ...) and/or lives in the results folder, so a
    recursive scan with a broad ``--pattern`` can never feed a mask or a CSV
    back in as if it were a raw volume.
    """
    file_path = Path(file_path)
    if file_path.name.startswith("cellpose_"):
        return True
    if not _OUTPUT_FOLDER_NAME:
        return False
    return file_path.parent.name.lower() == _OUTPUT_FOLDER_NAME


def iter_input_paths(input_path, pattern=DEFAULT_PATTERN, recursive=DEFAULT_RECURSIVE):
    """Return the list of TIFF volumes to process.

    Accepts either one file or a folder. A folder is scanned for ``pattern``
    (recursively when ``recursive``) and sorted naturally, so a mosaic/tile set
    is processed in order. This script's own outputs are skipped (see
    :func:`is_own_output`), so pointing a recursive scan at a folder that already
    holds results (``<input folder>/segmentation_results``) is safe.
    """
    path = Path(input_path)
    if path.is_dir():
        walker = path.rglob if recursive else path.glob
        paths = sorted(
            (item for item in walker(pattern)
             if item.is_file() and not is_own_output(item)),
            key=lambda item: natural_sort_key(str(item)),
        )
        if not paths:
            raise RuntimeError(f"No files matching '{pattern}' in: {path}")
        return paths
    if not path.is_file():
        raise FileNotFoundError(f"Input volume not found: {path}")
    return [path]


def volume_page_info(path):
    """Return ``(page_count, page_shape, dtype)`` of a volume TIFF.

    The app writes one page per Y row, so ``page_count`` is Y and
    ``page_shape`` is ``(X, Z)``.
    """
    with TIFF.TiffFile(path) as tif:
        page_count = len(tif.pages)
        if page_count <= 0:
            raise ValueError(f"No TIFF pages in: {path}")
        return page_count, tuple(int(v) for v in tif.pages[0].shape), tif.pages[0].dtype


def read_volume(path, y_slice=None, max_gigabytes=DEFAULT_MAX_GIGABYTES):
    """Read an OCT volume TIFF as a ``[Y, X, Z]`` array.

    The app saves one TIFF page per Y row (page shape ``[X, Z]``, float32
    amplitude), so a multi-page read reconstructs ``[Y, X, Z]``.

    ``y_slice`` limits the read to ``[y0, y1)`` Y rows *without* touching the
    other pages, which is what keeps a 20 GB stitched mosaic out of RAM when
    the volume is processed in strips.

    A memory map is used only when it provably covers the whole file - for some
    multi-page files ``tifffile.memmap`` returns a single page, which would
    silently drop every other Y row.
    """
    page_count, page_shape, page_dtype = volume_page_info(path)
    if y_slice is None:
        y0, y1 = 0, page_count
    else:
        y0, y1 = sorted((max(0, int(y_slice[0])), min(page_count, int(y_slice[1]))))
    if y1 <= y0:
        raise ValueError(f"Empty Y range [{y0}:{y1}] for {path}")

    load_gb = (y1 - y0) * int(np.prod(page_shape)) * page_dtype.itemsize / 1e9
    if load_gb > float(max_gigabytes):
        raise RuntimeError(
            f"Loading {y1 - y0} of {page_count} Y rows from "
            f"{os.path.basename(path)} needs {load_gb:.2f} GB, above the "
            f"--max-gigabytes={max_gigabytes:.2f} guard. Use --tile-y (strips), "
            "--downsample, --crop or --depth-range."
        )

    with TIFF.TiffFile(path) as tif:
        rows = y1 - y0
        if rows == 1:
            volume = tif.pages[y0].asarray()  # [X, Z]
        elif rows == page_count:
            volume = None
            expected_size = page_count * int(np.prod(page_shape))
            try:
                mapped = TIFF.memmap(path, mode="r")
                if mapped.size == expected_size and mapped.dtype == page_dtype:
                    volume = mapped
                else:
                    print(
                        f"  memmap shape {tuple(mapped.shape)} does not match "
                        f"{page_count} pages of {page_shape}; reading pages explicitly"
                    )
            except Exception:
                volume = None  # not contiguous - fall back to page stacking
            if volume is None:
                volume = np.stack([page.asarray() for page in tif.pages])
        else:
            volume = np.stack([page.asarray() for page in tif.pages[y0:y1]])

    volume = np.asarray(volume)
    if volume.ndim == 2:
        # A single en-face frame (or a one-row read): add the Y or Z axis back.
        volume = volume[np.newaxis, :, :] if rows == 1 else volume[:, :, np.newaxis]
    elif volume.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a [Y, X, Z] volume, got shape {volume.shape}"
        )
    return volume




def output_paths(input_path, output_dir=DEFAULT_OUTPUT_DIR, suffix="cellpose"):
    """Build the (mask, csv, qc, json, npy) output paths for one volume.

    ``output_dir`` - a relative name (the default) is created *under the input
    file's own folder*, e.g. ``...\\Time-2\\segmentation_results\\``, so every
    volume keeps its results next to it without polluting the data folder; an
    absolute path is used as-is; ``""``/``None`` writes next to the input file.
    The folder is created when missing and reused when it already exists - the
    file names are derived from the input stem, so a re-run simply overwrites the
    previous outputs of that volume.
    """
    source = Path(input_path)
    configured = str(output_dir or "").strip()
    if not configured:
        folder = source.parent
    else:
        folder = Path(configured)
        if not folder.is_absolute():
            folder = source.parent / folder
    folder.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    return {
        "folder": folder,
        "mask": folder / f"{suffix}_mask_{stem}.tif",
        "csv": folder / f"{suffix}_organoids_{stem}.csv",
        "qc": folder / f"{suffix}_qc_{stem}.png",
        "json": folder / f"{suffix}_summary_{stem}.json",
        "seg": folder / f"{suffix}_seg_{stem}.npy",
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def compress_volume(volume, mode="db", dynamic_range_db=DEFAULT_DYNAMIC_RANGE_DB,
                    device="cpu"):
    """Map raw OCT amplitude to a 0-1 display-like volume.

    ``db``   ``20*log10``, shift so the brightest voxel is 0 dB, clip to
             ``dynamic_range_db`` and rescale to 0-1. This is the OCT
             B-scan/C-scan look Cellpose handles best.
    ``log``  ``log1p`` compression, then a plain max scaling.
    ``none`` only clean up non-finite values (the percentile stretch still
             runs).

    ``device`` "gpu"/"auto" runs the arithmetic in torch on the GPU (see
    :func:`torch_preprocess_device`); anything else - and every fallback - uses
    the numpy code below.
    """
    torch_device, _reason = torch_preprocess_device(device)
    if torch_device is not None:
        tensor = _torch_from_numpy(volume, torch_device)
        return _compress_torch(tensor, mode, dynamic_range_db).cpu().numpy()
    data = np.array(volume, dtype=np.float32, copy=True)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(data, 0.0, out=data)

    if mode == "db":
        np.log10(np.maximum(data, 1e-6), out=data)
        data *= 20.0
        data -= float(data.max())
        data += float(dynamic_range_db)
        np.clip(data, 0.0, float(dynamic_range_db), out=data)
        data /= float(dynamic_range_db)
    elif mode == "log":
        np.log1p(data, out=data)
        peak = float(data.max())
        if peak > 0:
            data /= peak
    elif mode != "none":
        raise ValueError(f"Unknown compression mode: {mode}")
    return data


def normalise_percentiles(data, low=DEFAULT_PERCENTILE_LOW,
                          high=DEFAULT_PERCENTILE_HIGH, device="cpu"):
    """Clip to the [low, high] percentile range and rescale to 0-1.

    On the GPU the two percentiles are estimated from a uniform stride
    subsample when the input has more than ``_PERCENTILE_SAMPLE_LIMIT`` voxels;
    the exact numpy partition is used otherwise - see ``_percentiles_torch``.
    """
    if high <= low:
        return np.asarray(data, dtype=np.float32)
    torch_device, _reason = torch_preprocess_device(device)
    if torch_device is not None:
        tensor = _torch_from_numpy(data, torch_device)
        out = _normalise_percentiles_torch(tensor, low, high)
        return out.cpu().numpy().astype(np.float32, copy=False)
    lo = float(np.percentile(data, float(low)))
    hi = float(np.percentile(data, float(high)))
    if hi <= lo:
        return np.asarray(data, dtype=np.float32)
    out = np.clip(np.asarray(data, dtype=np.float32), lo, hi)
    out = (out - lo) / (hi - lo)
    return out.astype(np.float32, copy=False)


def smooth_volume(data, sigma_xy=DEFAULT_SMOOTH_XY, sigma_z=DEFAULT_SMOOTH_Z,
                  device="cpu"):
    """Gaussian de-speckle on a ``[Z, Y, X]`` volume (0 sigma = skip).

    The GPU version runs three 1-D convolutions with scipy's own kernel
    (``_smooth_torch``); a 2D input always goes through scipy.
    """
    sigma_xy = max(0.0, float(sigma_xy))
    sigma_z = max(0.0, float(sigma_z))
    if sigma_xy <= 0.0 and sigma_z <= 0.0:
        return data
    torch_device, _reason = torch_preprocess_device(device)
    if torch_device is not None and np.asarray(data).ndim == 3:
        tensor = _torch_from_numpy(data, torch_device)
        return _smooth_torch(tensor, sigma_z, sigma_xy).cpu().numpy()
    return ndi.gaussian_filter(
        data, sigma=(sigma_z, sigma_xy, sigma_xy), mode="nearest"
    ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# GPU preprocessing (optional)
# ---------------------------------------------------------------------------
# compress_volume / normalise_percentiles / smooth_volume / project_volume are
# the CPU-bound half of a run: they touch every voxel in numpy/scipy and stay
# the only CPU work once Cellpose itself runs on the GPU (measured on the
# 2272 x 2515 x 76 mosaic: the GPU drops to ~5 % and ~2 CPU cores stay busy).
# torch is loaded for Cellpose anyway and the VRAM is mostly idle, so the same
# steps are also implemented with torch and used whenever a CUDA device is
# there. The torch path matches numpy up to float32 rounding - same Gaussian
# kernel and truncation as scipy (see gaussian_kernel1d) and "replicate"
# padding for scipy's mode="nearest". Only the percentile pair is estimated
# above _PERCENTILE_SAMPLE_LIMIT voxels, because torch.quantile sorts its
# input; preprocess_device_report() measures every difference.
_PERCENTILE_SAMPLE_LIMIT = 16_000_000  # voxels (~64 MB of float32)
_PREPROCESS_DEVICES = {}  # requested -> (device, reason); resolved once


def torch_preprocess_device(requested=DEFAULT_PREPROCESS_DEVICE):
    """Resolve ``requested`` to ``(torch device, reason)``; device None = numpy.

    Anything unexpected - torch missing, a CPU-only torch build, no CUDA device,
    the Windows DLL conflict - returns ``None`` so the caller quietly uses
    numpy/scipy: preprocessing must never be the reason a run fails.
    """
    requested = str(requested if requested is not None else "auto").strip().lower()
    if requested in ("cpu", "numpy", "none", "off"):
        return None, "requested cpu"
    if requested not in _PREPROCESS_DEVICES:
        try:
            module = sys.modules.get("torch") or __import__("torch")
        except Exception as error:  # ImportError, OSError (DLL conflict)
            return None, f"torch unavailable ({type(error).__name__}: {error})"
        try:
            if not module.cuda.is_available():
                return None, "torch has no CUDA device here"
            device = module.device("cuda")
            version = getattr(module, "__version__", "?")
            _PREPROCESS_DEVICES[requested] = (
                device, f"torch {version}, {module.cuda.get_device_name(device)}"
            )
        except Exception as error:
            return None, f"torch.cuda failed ({type(error).__name__}: {error})"
    return _PREPROCESS_DEVICES[requested]


def preprocess_device_summary(requested=DEFAULT_PREPROCESS_DEVICE):
    """``{"device": "gpu"|"cpu", "reason": str}`` for the log and summary JSON."""
    device, reason = torch_preprocess_device(requested)
    return {"device": "cpu" if device is None else "gpu", "reason": reason}


def gaussian_kernel1d(sigma, truncate=4.0):
    """Normalised 1-D Gaussian kernel + radius, built exactly like scipy's.

    ``ndi.gaussian_filter`` uses ``radius = int(truncate * sigma + 0.5)`` and a
    kernel normalised over that window; the torch convolution in _smooth_torch
    has to use the same numbers, otherwise the GPU and the CPU paths would
    disagree at the borders and around sharp edges.
    """
    sigma = float(sigma)
    radius = max(1, int(float(truncate) * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32), radius


def _torch_from_numpy(array, device):
    """A contiguous float32 copy of ``array`` as a tensor on ``device``."""
    torch = sys.modules["torch"]
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


def _compress_torch(tensor, mode, dynamic_range_db):
    """``compress_volume`` for a float tensor, in place where it can."""
    torch = sys.modules["torch"]
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    tensor.clamp_(min=0.0)
    if mode == "db":
        torch.log10(tensor.clamp_min(1e-6), out=tensor)
        tensor.mul_(20.0)
        tensor.sub_(float(tensor.max()))  # brightest voxel -> 0 dB
        tensor.add_(float(dynamic_range_db))
        tensor.clamp_(0.0, float(dynamic_range_db))
        tensor.div_(float(dynamic_range_db))
    elif mode == "log":
        torch.log1p(tensor, out=tensor)
        peak = float(tensor.max())
        if peak > 0:
            tensor.div_(peak)
    elif mode != "none":
        raise ValueError(f"Unknown compression mode: {mode}")
    return tensor


def _percentiles_torch(tensor, low, high):
    """The ``(low, high)`` percentiles of a tensor, as floats.

    ``torch.quantile`` sorts its input, so above ``_PERCENTILE_SAMPLE_LIMIT``
    voxels the value comes from a uniform stride subsample: deterministic, and
    spread over the whole volume instead of one corner of it.
    """
    torch = sys.modules["torch"]
    flat = tensor.reshape(-1)
    if int(flat.numel()) > _PERCENTILE_SAMPLE_LIMIT:
        stride = max(1, int(flat.numel()) // _PERCENTILE_SAMPLE_LIMIT)
        flat = flat[::stride].contiguous()
    if int(flat.numel()) < 2:
        return 0.0, 0.0
    quantiles = torch.tensor([float(low) / 100.0, float(high) / 100.0],
                             dtype=torch.float32, device=flat.device)
    values = torch.quantile(flat, quantiles)
    return float(values[0]), float(values[1])


def _normalise_percentiles_torch(tensor, low, high):
    """``normalise_percentiles`` for a float tensor."""
    low_value, high_value = _percentiles_torch(tensor, low, high)
    if high_value <= low_value:
        return tensor
    out = tensor.clamp(min=low_value, max=high_value)
    out.sub_(low_value).div_(high_value - low_value)
    return out


def _smooth_torch(tensor_zyx, sigma_z, sigma_xy):
    """``smooth_volume`` for a ``[Z, Y, X]`` tensor: three 1-D convolutions.

    Convolving in Z, then Y, then X with the scipy kernels reproduces the same
    separable Gaussian scipy applies, and ``mode="replicate"`` matches scipy's
    ``mode="nearest"`` border handling.
    """
    torch = sys.modules["torch"]
    functional = torch.nn.functional
    sigma_z = max(0.0, float(sigma_z))
    sigma_xy = max(0.0, float(sigma_xy))
    if sigma_z <= 0.0 and sigma_xy <= 0.0:
        return tensor_zyx
    work = tensor_zyx.unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]
    for axis, sigma in ((2, sigma_z), (3, sigma_xy), (4, sigma_xy)):
        if sigma <= 0.0:
            continue
        kernel, radius = gaussian_kernel1d(sigma)
        shape = [1, 1, 1, 1, 1]
        shape[axis] = int(kernel.size)
        weights = torch.from_numpy(kernel).to(work.device).reshape(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - axis) * 2] = radius
        pad[(4 - axis) * 2 + 1] = radius
        work = functional.conv3d(functional.pad(work, pad, mode="replicate"), weights)
    return work[0, 0]


def _project_torch(tensor_zyx, method):
    """``project_volume`` for a ``[Z, Y, X]`` tensor."""
    if str(method).lower() == "mean":
        return tensor_zyx.mean(dim=0)
    return tensor_zyx.amax(dim=0)


def oct_preprocess_volume(subvolume_yxz, args, device=None, projection=None):
    """compress -> percentile stretch -> de-speckle, on one device.

    ``subvolume_yxz`` is a raw, already cropped ``[Y, X, Z]`` amplitude slice.
    Returns ``(volume_zyx, projection_yx)``: the 0-1 ``[Z, Y, X]`` volume
    Cellpose is given, plus - when ``projection`` is ``"max"``/``"mean"`` - its
    en-face projection, which costs nothing here because the volume is still on
    the device (and a projection is monotone, so the QC panel and the mask
    always agree).

    The single-pass, strip and streamed paths all go through this, so it is the
    one place that decides between the numpy and the torch implementation.
    """
    device = getattr(args, "preprocess_device", None) if device is None else device
    torch_device, _reason = torch_preprocess_device(device)
    method = None if projection is None else str(projection).lower()

    if torch_device is None:
        data = compress_volume(subvolume_yxz, args.compression, args.dynamic_range_db)
        data = normalise_percentiles(data, args.percentile_low, args.percentile_high)
        data = oct_to_cellpose(data)  # -> [Z, Y, X]
        data = smooth_volume(data, args.smooth_xy, args.smooth_z)
        return data, (None if method is None else project_volume(data, method))

    tensor = _torch_from_numpy(subvolume_yxz, torch_device)
    tensor = _compress_torch(tensor, args.compression, args.dynamic_range_db)
    tensor = _normalise_percentiles_torch(tensor, args.percentile_low,
                                          args.percentile_high)
    tensor = tensor.permute(2, 0, 1).contiguous()  # [Y, X, Z] -> [Z, Y, X]
    tensor = _smooth_torch(tensor, args.smooth_z, args.smooth_xy)
    projected = None
    if method is not None:
        projected = _project_torch(tensor, method).cpu().numpy()
    return tensor.cpu().numpy().astype(np.float32, copy=False), projected


def preprocess_device_report(verbose=True):
    """Check the GPU preprocessing: which device, and how exact it is.

    Prints the resolved device and the largest difference between the numpy and
    the torch implementation of the whole chain - on a small random volume and
    on one large enough to trigger the percentile subsampling. Returns the
    numbers, so it doubles as a smoke test from the Spyder console:

        cellpose_organoid_segmentation.preprocess_device_report()
    """
    device, reason = torch_preprocess_device(DEFAULT_PREPROCESS_DEVICE)
    report = {
        "requested": str(DEFAULT_PREPROCESS_DEVICE),
        "device": "cpu" if device is None else "gpu",
        "reason": reason,
        "max_abs_difference": {},
    }
    if device is None:
        if verbose:
            print(f"Preprocessing device: cpu (numpy/scipy) - {reason}")
        return report

    sample = argparse.Namespace(
        compression="db", dynamic_range_db=35.0, percentile_low=1.0,
        percentile_high=99.9, smooth_xy=1.5, smooth_z=0.5,
        preprocess_device="cpu",
    )
    rng = np.random.default_rng(20260926)
    raw = (rng.random((10, 24, 40)).astype(np.float32) ** 3) * 5e3
    raw[2, 3, 4] = 0.0  # exercises the log10 floor
    reference, reference_projection = oct_preprocess_volume(
        raw, sample, device="cpu", projection="max"
    )
    sample.preprocess_device = "gpu"
    gpu_volume, gpu_projection = oct_preprocess_volume(
        raw, sample, device="gpu", projection="max"
    )
    differences = report["max_abs_difference"]
    differences["volume"] = float(np.max(np.abs(reference - gpu_volume)))
    differences["projection"] = float(
        np.max(np.abs(reference_projection - gpu_projection))
    )

    # Large enough that _percentiles_torch switches to its stride subsample.
    rows = _PERCENTILE_SAMPLE_LIMIT // 4000 + 1
    large = (rng.random((rows, 100, 40)).astype(np.float32) ** 3) * 5e3
    large_reference, _ = oct_preprocess_volume(large, sample, device="cpu")
    large_gpu, _ = oct_preprocess_volume(large, sample, device="gpu")
    differences["volume_subsampled_percentiles"] = float(
        np.max(np.abs(large_reference - large_gpu))
    )

    if verbose:
        print(f"Preprocessing device: gpu ({reason})")
        for step, difference in differences.items():
            print(f"  max |numpy - torch| {step:30s}: {difference:.3e}")
    return report


# ---------------------------------------------------------------------------
# Sampling / placement
# ---------------------------------------------------------------------------



def resolve_placement(shape_yxz, crop=None, depth_range=None):
    """Normalise the crop/depth arguments to ``(y0, y1, x0, x1, z0, z1)``.

    Works from a shape alone, so a mosaic can be planned before any page is
    read.
    """
    y_total, x_total, z_total = [int(v) for v in shape_yxz[:3]]
    if crop is not None:
        y0, y1, x0, x1 = [int(v) for v in crop]
        y0, y1 = sorted((max(0, y0), min(y_total, y1)))
        x0, x1 = sorted((max(0, x0), min(x_total, x1)))
    else:
        y0, y1, x0, x1 = 0, y_total, 0, x_total
    if depth_range is not None:
        z0, z1 = [int(v) for v in depth_range]
        z0, z1 = sorted((max(0, z0), min(z_total, z1)))
    else:
        z0, z1 = 0, z_total

    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        raise ValueError(
            f"Empty sub-volume selection: y[{y0}:{y1}] x[{x0}:{x1}] z[{z0}:{z1}]"
        )
    return (y0, y1, x0, x1, z0, z1)


# ---------------------------------------------------------------------------
# Orientation / sampling helpers
# ---------------------------------------------------------------------------
def oct_to_cellpose(subvolume_yxz):
    """``[Y, X, Z]`` (file layout) -> ``[Z, Y, X]`` (Cellpose 3D layout)."""
    return np.transpose(np.asarray(subvolume_yxz), (2, 0, 1))


def cellpose_to_oct(masks_zyx):
    """``[Z, Y, X]`` (Cellpose 3D layout) -> ``[Y, X, Z]`` (file layout)."""
    return np.transpose(np.asarray(masks_zyx), (1, 2, 0))


def project_volume(volume_zyx, method="max", device="cpu"):
    """En-face projection of a ``[Z, Y, X]`` volume -> ``[Y, X]``."""
    torch_device, _reason = torch_preprocess_device(device)
    if torch_device is not None and np.asarray(volume_zyx).ndim == 3:
        tensor = _torch_from_numpy(volume_zyx, torch_device)
        return _project_torch(tensor, method).cpu().numpy().astype(
            np.float32, copy=False
        )
    if str(method).lower() == "mean":
        return np.mean(volume_zyx, axis=0, dtype=np.float32)
    return np.max(volume_zyx, axis=0)


def downsample_volume(volume, factor):
    """Integer-stride downsample; returns ``(volume, factor)``.

    Works for both the ``[Z, Y, X]`` volume and the ``[Y, X]`` projection.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return volume, 1
    slices = tuple(slice(None, None, factor) for _ in range(np.ndim(volume)))
    return volume[slices], factor


def upsample_labels(labels, full_shape, factor):
    """Nearest-neighbour restore of downsampled labels to ``full_shape``.

    Voxel replication never merges or drops labels, so the label count and
    the organoid identities survive the round trip. Works for 2D and 3D.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return labels
    out = np.asarray(labels)
    for axis in range(out.ndim):
        out = np.repeat(out, factor, axis=axis)
    slices = tuple(slice(0, int(size)) for size in full_shape[: out.ndim])
    return out[slices]


def resolve_anisotropy(args):
    """Return ``(anisotropy, source)`` for the Cellpose ``anisotropy`` argument.

    Cellpose defines anisotropy as the ratio between the Z sampling and the
    XY sampling ("set to 2.0 if Z is sampled half as dense as X or Y"), i.e.
    ``z_step_um / xy_pixel_um``.
    """
    if float(args.anisotropy) > 0:
        return float(args.anisotropy), "explicit"
    if float(args.z_step_um) > 0 and float(args.xy_pixel_um) > 0:
        return float(args.z_step_um) / float(args.xy_pixel_um), "z_step/xy_pixel"
    return 0.0, "unset"


def restore_full_mask(masks, placement, full_shape, factor, is_volume):
    """Map Cellpose labels back into the full-volume ``[Y, X, Z]`` (or ``[Y, X]``).

    Cellpose returns ``[Z, Y, X]`` for 3D and ``[Y, X]`` for 2D; both are
    upsampled back to the sub-volume grid, transposed to the file layout and
    pasted into a full-size zero label volume, so the output mask always
    matches the input TIFF dimensions.
    """
    y0, y1, x0, x1, z0, z1 = placement
    if not is_volume:
        if masks.ndim == 3:
            masks = np.max(masks, axis=0)
        labels = upsample_labels(masks, (y1 - y0, x1 - x0), factor)
        full = np.zeros((full_shape[0], full_shape[1]), dtype=np.int32)
        full[y0:y1, x0:x1] = labels
        return full

    if masks.ndim == 2:  # a single Z plane came back as 2D
        masks = masks[np.newaxis, ...]
    labels = upsample_labels(masks, (z1 - z0, y1 - y0, x1 - x0), factor)
    labels_yxz = cellpose_to_oct(labels)
    full = np.zeros((full_shape[0], full_shape[1], full_shape[2]), dtype=np.int32)
    full[y0:y1, x0:x1, z0:z1] = labels_yxz
    return full


# ---------------------------------------------------------------------------
# Memory budgeting and mosaic tiling
# ---------------------------------------------------------------------------
# Rough bytes needed per voxel of the analysed volume: the loaded strip, its
# compressed copy, Cellpose's four flow/cellprob planes, Cellpose's mask and
# our own label volume. Used only to decide whether a mosaic must be split.
BYTES_PER_VOXEL_WORKING_SET = 32.0
BYTES_PER_VOXEL_PROJECTION = 8.0


def estimate_working_set_gigabytes(shape, mode="3d"):
    """Estimate the working set (GB) of one Cellpose pass over ``shape``."""
    dims = [int(v) for v in shape[:3] if int(v) > 0]
    voxels = float(np.prod(dims)) if dims else 0.0
    per_voxel = (
        BYTES_PER_VOXEL_PROJECTION if str(mode) == "projection"
        else BYTES_PER_VOXEL_WORKING_SET
    )
    return voxels * per_voxel / 1e9


def plan_y_strips(y0, y1, tile_y, overlap):
    """Split the Y range ``[y0, y1)`` into strips.

    Returns ``[(core_y0, core_y1, read_y0, read_y1), ...]``. The core defines
    which strip owns an object (by centroid); the read range adds ``overlap``
    rows of context on both sides so Cellpose still sees whole objects.
    """
    y0, y1 = sorted((int(y0), int(y1)))
    tile_y = max(1, int(tile_y))
    overlap = max(0, int(overlap))
    strips = []
    for core_y0 in range(y0, y1, tile_y):
        core_y1 = min(core_y0 + tile_y, y1)
        strips.append((
            core_y0,
            core_y1,
            max(y0, core_y0 - overlap),
            min(y1, core_y1 + overlap),
        ))
    return strips


def resolve_tile_y(shape_zyx, mode, requested_tile_y, max_ram_gigabytes, overlap):
    """Decide the Y strip height for a volume. ``0`` means one Cellpose call.

    Returns ``(tile_y, reason)``.
    """
    if str(mode) == "projection":
        if int(requested_tile_y) > 0:
            return int(requested_tile_y), "explicit strip height for the projection"
        return 0, "projection mode is a single 2D image, no tiling needed"
    if int(requested_tile_y) > 0:
        return int(requested_tile_y), "explicit TILE_Y / --tile-y"

    budget = float(max_ram_gigabytes)
    whole = estimate_working_set_gigabytes(shape_zyx, mode)
    if whole <= budget:
        return 0, (
            f"whole volume working set ~{whole:.1f} GB <= {budget:.1f} GB budget"
        )
    for candidate in (1024, 512, 256, 128, 64):
        strip_shape = (int(shape_zyx[0]), min(candidate, int(shape_zyx[1])),
                       int(shape_zyx[2]))
        estimate = estimate_working_set_gigabytes(strip_shape, mode)
        if estimate <= budget:
            return int(candidate), (
                f"auto: whole volume needs ~{whole:.1f} GB > {budget:.1f} GB, so "
                f"{candidate}-row Y strips (~{estimate:.1f} GB each) are used"
            )
    raise RuntimeError(
        f"Even 64-row Y strips need more than the {budget:.1f} GB budget for "
        f"{shape_zyx}. Use --downsample, --crop, --depth-range, or --mode "
        "projection, or raise --max-ram-gigabytes deliberately."
    )


# ---------------------------------------------------------------------------
# Cellpose
# ---------------------------------------------------------------------------
def configure_model_cache(model_dir=DEFAULT_MODEL_DIR):
    """Point Cellpose at the local weights folder and report what is in it.

    ``cellpose.models`` reads ``CELLPOSE_LOCAL_MODELS_PATH`` when it is imported
    and then only downloads a model if the file is missing, so setting the
    variable before the import is enough to keep every run offline. A variable
    the user set themselves is never overridden. Returns a small dict that is
    printed before the model is created.
    """
    folder = Path(model_dir) if str(model_dir).strip() else None
    if folder is None:
        return {}
    summary = {"folder": str(folder), "exists": folder.is_dir(), "models": [],
               "notes": []}
    if folder.is_dir():
        # Only report weight-sized files, so the downloader/README next to them
        # are not mistaken for models.
        summary["models"] = sorted(
            f"{item.name} ({item.stat().st_size / 1e9:.2f} GB)"
            for item in folder.iterdir()
            if item.is_file() and item.stat().st_size >= 10_000_000
        )
        current = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", "")
        if not current.strip():
            os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(folder)
            summary["env_set"] = True

        # Cellpose's own cache folder (``~/.cellpose/models``) can still hold an
        # incomplete download from an earlier failed attempt, which is then
        # loaded as a "corrupt checkpoint" - easy to mistake for a broken
        # install, so call it out.
        default_folder = Path.home() / ".cellpose" / "models"
        if default_folder.is_dir() and default_folder != folder:
            for item in sorted(default_folder.iterdir()):
                if not item.is_file():
                    continue
                megabytes = item.stat().st_size / 1e6
                if megabytes < 100:
                    summary["notes"].append(
                        f"{item} is only {megabytes:.0f} MB - an incomplete "
                        "download from an earlier attempt; delete it so nothing "
                        "loads it by accident."
                    )
    return summary


def model_load_error_message(model_name, error):
    """Explain a Cellpose model load/download failure for this machine."""
    text = f"{type(error).__name__}: {error}"
    network = any(
        marker in text
        for marker in ("URLError", "WinError 10060", "timed out", "Timeout",
                       "Connection", "SSL", "getaddrinfo", "Temporary failure")
    )
    lines = [f"Cellpose could not load the model '{model_name}':", f"    {text}", ""]
    if network:
        lines += [
            "This is a download failure: the built-in models are fetched from",
            "huggingface.co, which is not reachable from this machine.",
            "Fetch the weights once with the resumable downloader that sits next",
            "to them (uses https://hf-mirror.com), then run again:",
            "    powershell -NoProfile -ExecutionPolicy Bypass -File "
            f'"{DEFAULT_MODEL_DIR}\\download_cpsam_v2.ps1"',
            "or copy the file from any machine with internet access.",
            "",
            "You can also point at a weight file directly:",
            f'    segment_organoids(model=r"{DEFAULT_MODEL_DIR}\\{model_name}")',
            f'    python cellpose_organoid_segmentation.py --model "{DEFAULT_MODEL_DIR}\\{model_name}"',
        ]
    else:
        lines += [
            f"Check that {DEFAULT_MODEL_DIR} contains the file '{model_name}', or",
            "pass an explicit path with model=... / --model.",
        ]
    return "\n".join(lines)


def load_cellpose_model(model_name, gpu):
    """Create a Cellpose model, tolerating the Cellpose 3 -> 4 API change.

    Cellpose 4 removed ``models.Cellpose`` and the classic model zoo: the only
    entry point is ``models.CellposeModel(pretrained_model='cpsam_v2')``.
    Cellpose 3 keeps ``models.CellposeModel(gpu=..., model_type='cyto3')``.

    The weights folder has to be configured *before* ``cellpose.models`` is
    imported, because Cellpose reads ``CELLPOSE_LOCAL_MODELS_PATH`` into its
    ``MODEL_DIR`` at import time.
    """
    cache = configure_model_cache()
    if cache.get("models"):
        print(f"Weights: {cache['folder']} -> {', '.join(cache['models'])}")
    elif cache:
        print(
            f"Weights: {cache['folder']} is empty, so Cellpose would download "
            f"'{model_name}' from huggingface.co (unreachable here); see "
            "download_cpsam_v2.ps1 in that folder."
        )
    for note in cache.get("notes", []):
        print(f"Note   : {note}")

    try:
        from cellpose import models
    except ImportError as error:
        raise SystemExit(
            "Cellpose is not installed in this Python environment.\n"
            "Install it into the SAME environment used to run this script, e.g.\n"
            '    "C:\\Users\\shuaibin\\.conda\\envs\\python311_env\\python.exe" '
            '-m pip install "cellpose[gui]"\n'
            "or create a dedicated environment (recommended, so the OCT "
            "acquisition app environment stays untouched):\n"
            "    conda create -n cellpose_env python=3.11 -y\n"
            "    conda activate cellpose_env\n"
            '    pip install "cellpose[gui]"\n'
            f"(original import error: {error})"
        )
    except OSError as error:
        # WinError 126/127 while importing torch: a DLL conflict, not a missing
        # Cellpose. Explain it instead of showing a raw traceback.
        raise SystemExit(torch_dll_error_message(error))

    cp_version = "unknown"
    major = 4
    try:
        from importlib.metadata import version as package_version

        cp_version = str(package_version("cellpose"))
        major = int(cp_version.split(".")[0])
    except Exception:
        pass

    # Point Cellpose at the local weights folder before it is asked for a model:
    # Cellpose only downloads a model whose file is missing from that folder.
    # CPU performance: Cellpose-SAM defaults to bfloat16, which is *emulated* on
    # most x86 CPUs (and then very slow), and torch's default thread count can be
    # low. On the CPU run float32 with all cores; on a GPU keep the bf16 default.
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        # Cellpose's 3D neighbour graph is a sparse tensor: say explicitly that
        # the invariant checks stay off, which is what silences torch's warning.
        silence_sparse_invariant_warning(torch_module)
    use_bfloat16 = True
    if gpu:
        if torch_module is not None and not torch_module.cuda.is_available():
            print(
                "          torch was built without CUDA support (CPU wheel), so "
                "Cellpose runs on the CPU; pass --no-gpu to skip this note."
            )
            gpu = False
    if not gpu and torch_module is not None:
        use_bfloat16 = False
        try:
            threads = int(os.cpu_count() or 1)
            torch_module.set_num_threads(threads)
            print(f"CPU    : torch threads = {threads}, bfloat16 off (CPU "
                  "emulation is slow)")
        except Exception:
            pass

    def build_model(**extra):
        if major >= 4:
            return models.CellposeModel(gpu=bool(gpu), pretrained_model=str(model_name),
                                        **extra)
        return models.CellposeModel(gpu=bool(gpu), model_type=str(model_name), **extra)

    try:
        model = build_model(use_bfloat16=use_bfloat16)
    except TypeError:
        # Cellpose 3 has no use_bfloat16 argument
        model = build_model()
    except Exception as error:
        raise SystemExit(model_load_error_message(model_name, error))
    print(f"Model  : {getattr(model, 'pretrained_model', model_name)}")
    return model, major, cp_version


def build_eval_kwargs(args, major, anisotropy):
    """Assemble ``CellposeModel.eval`` keyword arguments for the chosen mode."""
    kwargs = {
        "do_3D": str(args.mode) == "3d",
        "normalize": True,  # Cellpose's own percentile normalisation
        "flow_threshold": float(args.flow_threshold),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "batch_size": int(args.batch_size),
    }
    if str(args.mode) == "projection":
        # A 2D [Y, X] image: Cellpose rejects a z_axis here ("2D image processing
        # selected, but z_axis is not None").
        kwargs["z_axis"] = None
    else:
        kwargs["z_axis"] = 0  # our [Z, Y, X] volume has Z first
    if major < 4:
        kwargs["channels"] = [0, 0]  # Cellpose 3 only; ignored in Cellpose 4
    if float(args.diameter) > 0:
        kwargs["diameter"] = float(args.diameter)
    if int(args.niter) > 0:
        kwargs["niter"] = int(args.niter)
    if str(args.mode) == "2d-stitch":
        kwargs["do_3D"] = False
        kwargs["stitch_threshold"] = float(args.stitch_threshold)
    if str(args.mode) == "3d" and anisotropy and abs(float(anisotropy) - 1.0) > 1e-6:
        kwargs["anisotropy"] = float(anisotropy)
    return kwargs


def run_cellpose(model, args, major, image, anisotropy):
    """Run Cellpose and return ``(masks, flows, styles)``.

    Cellpose 4 returns ``masks, flows, styles`` from ``eval`` (Cellpose 3
    returned a fourth ``diams`` value), so the tuple is unpacked defensively.
    """
    kwargs = build_eval_kwargs(args, major, anisotropy)
    print("  eval kwargs: " + json.dumps(kwargs, sort_keys=True))
    try:
        output = model.eval(image, **kwargs)
    except Exception as error:
        # Cellpose-SAM is trained on three channels. If the plain grayscale
        # call is rejected, retry with the single channel replicated - the
        # pattern used in cellpose's own paper/cpsam/eval_3D.py.
        print(f"  first eval() call failed: {type(error).__name__}: {error}")
        print("  retrying with the grayscale image replicated to 3 channels ...")
        stacked = np.repeat(np.asarray(image)[..., np.newaxis], 3, axis=-1)
        output = model.eval(stacked, channel_axis=-1, **kwargs)

    if isinstance(output, (tuple, list)):
        masks = output[0]
        flows = output[1] if len(output) > 1 else None
        styles = output[2] if len(output) > 2 else None
    else:
        masks, flows, styles = output, None, None
    return np.asarray(masks), flows, styles


def dry_run_masks(image, min_size):
    """Segmentation stand-in used by ``--dry-run`` to test the plumbing.

    Thresholds the normalised image, labels connected components and drops
    objects below ``min_size``. It is NOT a segmentation - it only exercises
    reading, statistics, writing and the QC figure without Cellpose/GPU.
    """
    threshold = 0.4
    labels, count = ndi.label(np.asarray(image) >= threshold)
    if count == 0:
        return labels.astype(np.int32)
    sizes = np.bincount(labels.reshape(-1))
    keep = np.nonzero(sizes >= max(1, int(min_size)))[0]
    keep = keep[keep > 0]
    lut = np.zeros(sizes.size, dtype=np.int32)
    lut[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return lut[labels]


# ---------------------------------------------------------------------------
# Label statistics
# ---------------------------------------------------------------------------
# Column order of the statistics CSV. In projection (2D) mode "volume_um3"
# carries the XY area in um^2 and the z columns stay empty.
CSV_FIELDS = (
    "organoid_id",
    "voxel_count",
    "volume_um3",
    "equivalent_diameter_um",
    "surface_area_um2",
    "roundness",
    "y_centroid_px",
    "x_centroid_px",
    "z_centroid_px",
    "y_extent_px",
    "x_extent_px",
    "z_extent_px",
    "y0_px",
    "y1_px",
    "x0_px",
    "x1_px",
    "z0_px",
    "z1_px",
)


def filter_small_labels(masks, min_size):
    """Relabel ``masks`` keeping only objects with >= ``min_size`` voxels.

    Returns ``(relabelled_masks, dropped_count)``. Labels come back
    consecutively numbered from 1, so the CSV ids match the mask values.
    """
    masks = np.asarray(masks)
    labels, counts = np.unique(masks, return_counts=True)
    positive = labels > 0
    labels, counts = labels[positive], counts[positive]
    if labels.size == 0:  # nothing but background
        return np.zeros(masks.shape, dtype=np.int32), 0

    keep = labels[counts >= max(1, int(min_size))]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    if keep.size:
        lut[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    dropped = int(labels.size - keep.size)
    return lut[masks], dropped


# A voxel-face surface sum overestimates a smooth surface by 1.5 - the mean of
# |nx| + |ny| + |nz| over a surface, i.e. the staircase area - so the area is
# divided by this before the sphericity ratio. Calibrated on digital shapes:
# sphere 1.00, ellipsoid 2:1 0.96, rod 6:1 0.83, two touching lobes 0.81; flat
# faced objects overshoot (cube 1.21) because the correction assumes smoothness.
VOXEL_FACE_AREA_CORRECTION = 1.5


def exposed_face_area(block, xy_um, z_um):
    """Surface area of one binary ``[Y, X, Z]`` block = its exposed voxel faces.

    A face whose normal is Z lies in an X-Y plane and measures ``xy_um^2``; X-
    and Y-normal faces measure ``xy_um * z_um``, so anisotropic sampling is
    handled. With pixel sizes 0 the return value is a plain face count, which
    keeps roundness (a ratio) meaningful and makes the column "faces".
    """
    filled = np.asarray(block)
    if filled.ndim != 3:
        raise ValueError(f"Expected a [Y, X, Z] block, got shape {filled.shape}")
    filled = filled.astype(bool, copy=False)
    face_z = float(xy_um) ** 2 if float(xy_um) > 0 else 1.0
    face_y = (float(xy_um) * float(z_um)
              if float(xy_um) > 0 and float(z_um) > 0 else 1.0)

    padded = np.zeros(
        (filled.shape[0] + 2, filled.shape[1] + 2, filled.shape[2] + 2), dtype=bool
    )
    padded[1:-1, 1:-1, 1:-1] = filled
    total = 0.0
    for axis, weight in ((0, face_y), (1, face_y), (2, face_z)):
        for offset in (-1, 1):
            lower = [1, 1, 1]
            upper = [-1, -1, -1]
            lower[axis] = 0 if offset < 0 else 2
            upper[axis] = -2 if offset < 0 else None
            neighbour = padded[tuple(slice(a, b) for a, b in zip(lower, upper))]
            total += float(np.count_nonzero(filled & ~neighbour)) * weight
    return total


def sphericity(volume, surface_area):
    """Wadell sphericity of an object, corrected so a sphere reads ~1.00.

    ``pi**(1/3) * (6V)**(2/3) / A`` with ``A`` divided by
    ``VOXEL_FACE_AREA_CORRECTION`` first. ``volume`` and ``surface_area`` must be
    in matching units (um^3 with um^2, or voxels with faces). Returns ``None``
    when there is nothing to measure.
    """
    volume = float(volume)
    surface_area = float(surface_area)
    if volume <= 0.0 or surface_area <= 0.0:
        return None
    corrected = surface_area / VOXEL_FACE_AREA_CORRECTION
    return float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / corrected)


def roundness_summary(records):
    """Mean/median/min/max roundness for the summary JSON, or ``None``.

    2D modes have no roundness, and so has a mask whose objects were all
    dropped, so the summary simply omits the block in those cases.
    """
    values = sorted(
        float(record["roundness"]) for record in records
        if record.get("roundness") is not None
    )
    if not values:
        return None
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(values[len(values) // 2], 4),
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
    }


def _size_fields(count, ndim, xy_um, z_um, surface_area=None):
    """Volume/area, equivalent diameter and shape fields for one object.

    With sampled pixel sizes the equivalent diameter is the diameter of a
    sphere (3D) or circle (2D) with the same volume/area; with pixel sizes 0
    the same formula is applied to the voxel/pixel count, so the value is in px.
    ``surface_area`` (from :func:`exposed_face_area`, in um^2 or in faces) adds
    ``surface_area_um2`` and ``roundness`` for 3D objects.
    """
    xy_um = float(xy_um)
    z_um = float(z_um) if ndim == 3 else 0.0
    sampled = ndim == 3 and xy_um > 0 and z_um > 0
    if sampled:
        volume = count * xy_um * xy_um * z_um
        fields = {
            "volume_um3": round(volume, 3),
            "equivalent_diameter_um": round(
                2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0), 3
            ),
        }
    elif ndim == 2 and xy_um > 0:
        area = count * xy_um * xy_um
        fields = {
            # 2D mode: the "volume" column carries the XY area in um^2.
            "volume_um3": round(area, 3),
            "equivalent_diameter_um": round(2.0 * np.sqrt(area / np.pi), 3),
        }
    elif ndim == 3:
        fields = {
            "equivalent_diameter_um": round(
                2.0 * (3.0 * count / (4.0 * np.pi)) ** (1.0 / 3.0), 3
            )
        }
    else:
        fields = {"equivalent_diameter_um": round(2.0 * np.sqrt(count / np.pi), 3)}

    if ndim == 3 and surface_area is not None:
        fields["surface_area_um2"] = round(float(surface_area), 3)
        roundness = sphericity(
            count * xy_um * xy_um * z_um if sampled else float(count),
            float(surface_area),
        )
        if roundness is not None:
            fields["roundness"] = round(roundness, 4)
    return fields


def _position_fields(label_id, count, centroid, bbox, ndim, origin, xy_um, z_um,
                     surface_area=None):
    """Build one CSV record from count, centroid, bbox and an axis origin.

    ``origin`` shifts the reported positions into whole-volume coordinates
    (used when a strip of a mosaic is measured on its own).
    """
    record = {"organoid_id": int(label_id), "voxel_count": int(count)}
    names = ("y", "x", "z") if ndim == 3 else ("y", "x")
    for axis, name in enumerate(names):
        shift = float(origin[axis]) if axis < len(origin) else 0.0
        record[f"{name}_centroid_px"] = round(float(centroid[axis]) + shift, 2)
        record[f"{name}_extent_px"] = int(bbox[axis][1] - bbox[axis][0])
        record[f"{name}0_px"] = int(bbox[axis][0] + shift)
        record[f"{name}1_px"] = int(bbox[axis][1] + shift)
    record.update(_size_fields(count, ndim, xy_um, z_um, surface_area))
    return record


def normalise_origin(origin, ndim):
    """Return an ``origin`` tuple with ``ndim`` entries (missing ones are 0)."""
    if origin is None:
        return (0,) * ndim
    values = tuple(int(v) for v in origin)
    if len(values) == ndim:
        return values
    if len(values) < ndim:
        return values + (0,) * (ndim - len(values))
    return values[:ndim]


def label_records(masks, xy_um, z_um, origin=None, with_shape=True):
    """Per-organoid statistics for a 2D ``[Y, X]`` or 3D ``[Y, X, Z]`` mask.

    ``with_shape=False`` skips the surface-area / roundness measurement (one
    extra pass over every object's bounding box). The per-strip filtering uses
    that, because its records never reach the CSV.

    Volumes/areas use ``xy_um`` and ``z_um`` (pass 0 to keep pixel units).
    ``origin`` shifts reported positions into whole-volume coordinates.
    """
    masks = np.asarray(masks)
    ndim = masks.ndim
    origin = normalise_origin(origin, ndim)

    labels, counts = np.unique(masks, return_counts=True)
    positive = labels > 0
    labels, counts = labels[positive], counts[positive]
    if labels.size == 0:
        return []

    boxes = ndi.find_objects(masks)
    records = []
    for position, label_id in enumerate(labels):
        label_id = int(label_id)
        box = boxes[label_id - 1]
        count = int(counts[position])
        if box is None:  # label present in the counts but without voxels
            continue
        block = masks[box] == label_id
        index = np.nonzero(block)
        surface_area = (
            exposed_face_area(block, xy_um, z_um) if with_shape and ndim == 3 else None
        )
        centroid = [
            float(box[axis].start) + float(index[axis].mean())
            for axis in range(ndim)
        ]
        bbox = [(int(box[axis].start), int(box[axis].stop)) for axis in range(ndim)]
        records.append(
            # "organoid_id" is the value in the mask, so the CSV row always maps
            # back to the label volume (labels are numbered from 1).
            _position_fields(label_id, count, centroid, bbox, ndim, origin, xy_um, z_um,
                             surface_area=surface_area)
        )
    return records


# ---------------------------------------------------------------------------
# Strip-level label handling (stitched mosaics)
# ---------------------------------------------------------------------------
def _accumulate_face_area(accumulator, block, face_y, face_z,
                          halo_above=None, halo_below=None):
    """Add the exposed-face area of every label in one ``[Y, X, Z]`` chunk.

    A face with a Z normal measures ``face_z``, the others ``face_y`` (see
    :func:`exposed_face_area`). The two halo rows are used only as the Y
    neighbour of the chunk's first and last row, so a chunk seam where an object
    continues is not mistaken for a surface; voxels outside the chunk are never
    counted, so every face is attributed exactly once.
    """
    if block.size == 0:
        return
    maximum = int(block.max())
    if maximum <= 0:
        return
    zero_row = np.zeros((1,) + tuple(block.shape[1:]), dtype=block.dtype)
    above = zero_row if halo_above is None else halo_above
    below = zero_row if halo_below is None else halo_below
    neighbours = []
    if face_y:
        neighbours.append(np.concatenate((above, block[:-1]), axis=0))
        neighbours.append(np.concatenate((block[1:], below), axis=0))
        for source, target in ((block[:, :-1, :], (slice(None), slice(1, None), slice(None))),
                               (block[:, 1:, :], (slice(None), slice(None, -1), slice(None)))):
            shifted = np.zeros_like(block)
            shifted[target] = source
            neighbours.append(shifted)
    if face_z:
        for source, target in ((block[:, :, :-1], (slice(None), slice(None), slice(1, None))),
                               (block[:, :, 1:], (slice(None), slice(None), slice(None, -1)))):
            shifted = np.zeros_like(block)
            shifted[target] = source
            neighbours.append(shifted)

    weights = []
    for index in range(len(neighbours)):
        weights.append(face_y if (face_y and index < 4) else face_z)
    for neighbour, weight in zip(neighbours, weights):
        filled = block > 0
        exposed = filled & (block != neighbour)
        if not exposed.any():
            continue
        counts = np.bincount(block[exposed], minlength=maximum + 1)
        for label_id in np.nonzero(counts)[0]:
            accumulator[int(label_id)] = (
                accumulator.get(int(label_id), 0.0) + float(counts[label_id]) * weight
            )


def measure_mask_chunked(mask_source, y_chunk, xy_um, z_um, placement):
    """Measure a label volume in Y chunks, so a mosaic mask need not be loaded.

    ``mask_source`` is any array-like with ``[Y, X, Z]`` indexing (a memory map
    of the written BigTIFF is the intended use). Objects that reach across a
    chunk boundary are accumulated (count, coordinate sums, bounding box), and
    the reported positions are whole-volume coordinates from ``placement``.

    The final mask is the source of truth for the statistics: a strip boundary
    can hand a few voxels to the neighbouring object (writes never clobber), so
    measuring the per-strip labels instead would give slightly different counts
    than the mask that is written to disk.
    """
    y0, y1, x0, x1, z0, z1 = [int(v) for v in placement]
    counts = {}
    sums = {}
    boxes = {}
    faces = {}
    face_y = (float(xy_um) * float(z_um)
              if float(xy_um) > 0 and float(z_um) > 0 else 1.0)
    face_z = float(xy_um) ** 2 if float(xy_um) > 0 else 1.0
    y_chunk = max(1, int(y_chunk))
    for chunk_y0 in range(y0, y1, y_chunk):
        chunk_y1 = min(chunk_y0 + y_chunk, y1)
        block = np.asarray(mask_source[chunk_y0:chunk_y1, x0:x1, z0:z1])
        _accumulate_face_area(
            faces, block, face_y, face_z,
            halo_above=(np.asarray(mask_source[chunk_y0 - 1:chunk_y0, x0:x1, z0:z1])
                        if chunk_y0 > y0 else None),
            halo_below=(np.asarray(mask_source[chunk_y1:chunk_y1 + 1, x0:x1, z0:z1])
                        if chunk_y1 < y1 else None),
        )
        for label_id in np.unique(block):
            if label_id == 0:
                continue
            label_id = int(label_id)
            index = np.nonzero(block == label_id)
            number = int(index[0].size)
            counts[label_id] = counts.get(label_id, 0) + number
            y_values = index[0] + chunk_y0
            accumulated = sums.setdefault(label_id, [0.0, 0.0, 0.0])
            accumulated[0] += float(y_values.sum())
            accumulated[1] += float(index[1].sum()) + float(x0 * number)
            accumulated[2] += float(index[2].sum()) + float(z0 * number)

            y_min, y_max = int(y_values.min()), int(y_values.max()) + 1
            x_min = int(index[1].min()) + x0
            x_max = int(index[1].max()) + 1 + x0
            z_min = int(index[2].min()) + z0
            z_max = int(index[2].max()) + 1 + z0
            entry = boxes.get(label_id)
            if entry is None:
                boxes[label_id] = [y_min, y_max, x_min, x_max, z_min, z_max]
            else:
                entry[0] = min(entry[0], y_min)
                entry[1] = max(entry[1], y_max)
                entry[2] = min(entry[2], x_min)
                entry[3] = max(entry[3], x_max)
                entry[4] = min(entry[4], z_min)
                entry[5] = max(entry[5], z_max)

    records = []
    for label_id in sorted(counts):
        number = counts[label_id]
        accumulated = sums[label_id]
        centroid = (
            accumulated[0] / number,
            accumulated[1] / number,
            accumulated[2] / number,
        )
        box = boxes[label_id]
        bbox = [(box[0], box[1]), (box[2], box[3]), (box[4], box[5])]
        records.append(
            _position_fields(label_id, number, centroid, bbox, 3, (0, 0, 0),
                             xy_um, z_um, surface_area=faces.get(label_id))
        )
    return records


def keep_owned_labels(labels_yxz, core_y0, core_y1, min_voxels, origin,
                      xy_um, z_um):
    """Select the objects that one Y strip owns, and measure them.

    A strip owns an object when the object's centroid Y falls inside the
    strip's *core* Y range ``[core_y0, core_y1)``. Because strips are run with
    extra context rows (the overlap), the owner sees the whole object and is
    the only strip that writes it - so a mosaic can be segmented in strips
    without splitting objects, and without a merge pass.

    Returns ``(labels, records, dropped)``: an int32 ``[Y, X, Z]`` volume where
    the kept objects are renumbered from 1 (0 elsewhere), their statistics in
    whole-volume coordinates (``origin``), and the number of objects that were
    dropped (unowned or below ``min_voxels``).
    """
    labels_yxz = np.asarray(labels_yxz)
    measured = label_records(labels_yxz, xy_um, z_um, origin=origin,
                             with_shape=False)
    owned = [
        record for record in measured
        if float(core_y0) <= float(record.get("y_centroid_px", origin[0])) < float(core_y1)
    ]
    kept = [record for record in owned if int(record["voxel_count"]) >= max(1, int(min_voxels))]
    dropped = len(measured) - len(kept)

    if not kept:
        return np.zeros(labels_yxz.shape, dtype=np.int32), [], dropped

    keep_ids = np.asarray([record["organoid_id"] for record in kept], dtype=np.int64)
    lut = np.zeros(int(labels_yxz.max()) + 1, dtype=np.int32)
    lut[keep_ids] = np.arange(1, keep_ids.size + 1, dtype=np.int32)
    relabelled = lut[labels_yxz]

    for new_id, record in enumerate(kept, start=1):
        record["organoid_id"] = new_id
    return relabelled, kept, dropped


def write_labels_into(mask_region, labels_yxz):
    """Copy a strip's labels into ``mask_region`` without clobbering it.

    A strip fills only voxels that are still background, so the parts of an
    object that belong to the neighbouring strip stay available for that
    strip. Returns the number of voxels written.
    """
    source = np.asarray(labels_yxz)
    region = np.asarray(mask_region)
    if source.shape != region.shape:
        raise ValueError(
            f"Strip labels {source.shape} do not match the target region {region.shape}"
        )
    fill = source > 0
    if not fill.any():
        return 0
    fill &= region == 0
    region[fill] = source[fill].astype(region.dtype, copy=False)
    return int(fill.sum())


def accumulate_max_projection(target, strip_projection, y0, y1, x0, x1):
    """Accumulate one strip's en-face projection into ``target[y0:y1, x0:x1]``.

    Used for maximum projections (and for the QC panel): neighbouring strips
    read overlapping rows, and a maximum is idempotent, so writing the same
    region twice cannot distort the result. Mean projections must instead be
    accumulated as a sum with a per-pixel plane count (see
    :func:`process_projection_streamed`).
    """
    region = target[y0:y1, x0:x1]
    np.maximum(region, strip_projection, out=region)
    return target



# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_volume_stack(path, volume):
    """Write a ``[Y, X, Z]`` (or ``[Y, X]``) array as one TIFF page per Y row.

    Same layout the app and the other data_processing scripts write, so the
    mask overlays the input volume in ImageJ, the GUI, and numpy.
    """
    TIFF.imwrite(path, np.asarray(volume), photometric="minisblack", append=False)
    return path


def write_mask_tiff(path, mask_yxz):
    """Write a label volume as uint16 and return the number of labels."""
    mask = np.asarray(mask_yxz)
    peak = int(mask.max()) if mask.size else 0
    limit = int(np.iinfo(np.uint16).max)
    if peak > limit:
        raise ValueError(
            f"{peak} labels exceeds the uint16 limit ({limit}); enable "
            "--downsample or raise --min-volume-voxels to merge fragments."
        )
    write_volume_stack(path, mask.astype(np.uint16, copy=False))
    return peak


def write_csv(path, records):
    """Write the per-organoid records with a fixed column order."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS),
                                extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
    return path


def write_summary_json(path, summary):
    """Write the run summary (settings + counts) for reproducibility.

    The Windows DLL bootstrap result is added automatically, so a summary always
    records which torch/DLL directories a result was produced with.
    """
    def _plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {str(key): _plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(item) for item in value]
        return value

    with open(path, "w", encoding="utf-8") as handle:
        payload = dict(summary)
        payload.setdefault("torch_dll_bootstrap", TORCH_DLL_BOOTSTRAP)
        json.dump(_plain(payload), handle, indent=2, sort_keys=True)
    return path


def save_seg_npy(path, mask_yxz, image_shape, outlines=None, flows=None,
                 styles=None, extras=None):
    """Best-effort ``*_seg.npy`` in the Cellpose GUI format.

    The Cellpose GUI can open this file to inspect and hand-correct the labels
    (File > Load segmentation). Only the fields the GUI needs are written; the
    image itself is not stored (matching Cellpose's own format) - keep the
    input TIFF next to this file so the masks can be reloaded.
    ``image_shape`` is either an array or an already-known shape tuple.
    """
    if isinstance(image_shape, (tuple, list)):
        shape_values = [int(v) for v in image_shape]
    else:
        shape_values = [int(v) for v in np.asarray(image_shape).shape]
    payload = {
        "masks": np.asarray(mask_yxz).astype(np.uint16),
        "outlines": outlines if outlines is not None else np.zeros((0, 0)),
        "filename": "",
        "flows": flows if flows is not None else [],
        "styles": styles if styles is not None else [],
        "chan_choose": [0, 0],
        "ismanual": np.zeros(0, dtype=bool),
        "zdraw": [],
        "image_shape": shape_values,
    }
    if extras:
        payload.update(extras)
    np.save(path, payload, allow_pickle=True)
    return path


# ---------------------------------------------------------------------------
# QC figure
# ---------------------------------------------------------------------------
def make_qc_figure(path, enface, mask_2d, section, records, title,
                   dpi=DEFAULT_QC_RESOLUTION, diameter_unit="um",
                   show_figures=DEFAULT_SHOW_FIGURES):
    """Write the 4-panel QC PNG (image, labels, roundness, diameter histogram).

    The top row is bare image - no ticks, tick labels or axis labels - and every
    text element in the figure is twice the default size (panel titles, axis
    labels, ticks). ``title`` and ``section`` are still accepted so the call
    sites stay simple, but neither is drawn any more: the run's identity is in
    the file name and the JSON summary.

    The PNG is always written. With ``show_figures`` the figure is also
    displayed (Spyder Plots pane / interactive window), which needs a display
    backend - the module only forces Agg when DEFAULT_SHOW_FIGURES is False.
    """
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=int(dpi))
    # Every text element in the figure is twice the default size (panel titles,
    # axis labels and ticks), in both rows.
    panel_font = 2.0 * float(plt.rcParams.get("font.size", 10.0))

    ax = axes[0, 0]
    ax.imshow(enface, cmap="gray")
    ax.set_title("en-face projection (XY)", fontsize=panel_font)
    # Pure image on the top row: no ticks, no tick labels, no axis labels.
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(enface, cmap="gray")
    if mask_2d is not None and np.any(np.asarray(mask_2d) > 0):
        labels = np.asarray(mask_2d)
        ax.imshow(np.ma.masked_equal(labels, 0), cmap="nipy_spectral", alpha=0.5,
                  interpolation="nearest")
        ax.contour(labels > 0, levels=[0.5], colors="lime", linewidths=0.6)
    ax.set_title(f"Organoid labels ({len(records)} objects)", fontsize=panel_font)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    roundness = sorted(
        float(record["roundness"]) for record in records
        if record.get("roundness") is not None
    )
    if roundness:
        bins = int(min(20, max(5, len(roundness) // 2)))
        ax.hist(roundness, bins=bins, color="#55A868", edgecolor="white")
        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_title(
            f"roundness (n={len(roundness)}), "
            f"median {roundness[len(roundness) // 2]:.2f}",
            fontsize=panel_font,
        )
        ax.set_xlabel("roundness (1.00 = sphere)", fontsize=panel_font)
        ax.set_ylabel("count", fontsize=panel_font)
        ax.tick_params(axis="both", labelsize=panel_font)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "roundness needs a 3D mask", ha="center", va="center",
                fontsize=panel_font)

    ax = axes[1, 1]
    diameters = [
        float(record["equivalent_diameter_um"])
        for record in records
        if "equivalent_diameter_um" in record
    ]
    if diameters:
        bins = int(min(20, max(5, len(diameters) // 2)))
        ax.hist(diameters, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_title(f"equivalent diameter (n={len(diameters)})",
                     fontsize=panel_font)
        ax.set_xlabel(f"diameter ({diameter_unit})", fontsize=panel_font)
        ax.set_ylabel("count", fontsize=panel_font)
        ax.tick_params(axis="both", labelsize=panel_font)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no objects", ha="center", va="center",
                fontsize=panel_font)

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    if show_figures:
        try:
            figure.show()
        except Exception:
            pass
    else:
        plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Stitched mosaics: strip processing
# ---------------------------------------------------------------------------
def update_qc_figure(input_path, output_dir=DEFAULT_OUTPUT_DIR, show=False, y_chunk=512):
    """Rebuild one QC PNG from the outputs already on disk - no segmentation.

    The summary JSON in the output folder (``<input folder>/segmentation_results``
    by default, or ``output_dir``) carries the mask/CSV paths and every
    preprocessing setting of the run that produced them, so the rebuilt figure
    matches that run instead of the current ``DEFAULT_*`` values. The mask is
    memory-mapped and reduced in Y chunks, and the en-face image and the X-Z
    section are recomputed from the input volume page by page, so refreshing is
    seconds of work even for a 20 GB mosaic - useful when the figure predates a
    setting change, or after a mask was re-measured by hand.

    Returns the QC path.
    """
    source = Path(input_path)
    paths = output_paths(source, output_dir)
    summary = {}
    if paths["json"].is_file():
        try:
            with open(paths["json"], "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            summary = {}

    # The summary records the mask/CSV of the run that produced them. If those
    # files have moved since - e.g. they were written next to the input before
    # the results folder existed - fall back to the paths computed here.
    mask_path = Path(summary.get("mask") or paths["mask"])
    if not mask_path.is_file():  # summary path stale -> ask the current output_dir
        mask_path = paths["mask"]
    if not mask_path.is_file():  # projection mode writes cellpose_maskXY_*
        alternative = paths["mask"].with_name(
            paths["mask"].name.replace("cellpose_mask", "cellpose_maskXY")
        )
        if alternative.is_file():
            mask_path = alternative
    if not mask_path.is_file():
        raise SystemExit(f"No mask to build a QC figure from: {mask_path}")
    csv_path = Path(summary.get("csv") or paths["csv"])
    if not csv_path.is_file():
        csv_path = paths["csv"]

    page_count, page_shape, _dtype = volume_page_info(source)
    shape = summary.get("volume_shape_yxz") or [
        page_count, page_shape[0], page_shape[1],
    ]
    y_total, x_total, z_total = (int(value) for value in shape)
    subvolume = summary.get("subvolume") or {}
    y0 = int(subvolume.get("y0", 0))
    y1 = int(subvolume.get("y1", y_total))
    x0 = int(subvolume.get("x0", 0))
    x1 = int(subvolume.get("x1", x_total))
    z0 = int(subvolume.get("z0", 0))
    z1 = int(subvolume.get("z1", z_total))
    placement = (y0, y1, x0, x1, z0, z1)
    xy_um = float(summary.get("xy_pixel_um", DEFAULT_XY_PIXEL_UM) or 0.0)
    z_um = float(summary.get("z_step_um", DEFAULT_Z_STEP_UM) or 0.0)

    # Strip runs always show the maximum projection; single-pass runs follow
    # --projection, so honour what the summary recorded.
    projection = str(summary.get("projection", DEFAULT_PROJECTION))
    if str(summary.get("processing", "")) == "y-strips":
        projection = "max"
    settings = argparse.Namespace(
        compression=summary.get("compression", DEFAULT_COMPRESSION),
        dynamic_range_db=float(summary.get("dynamic_range_db", DEFAULT_DYNAMIC_RANGE_DB)),
        percentile_low=float(summary.get("percentile_low", DEFAULT_PERCENTILE_LOW)),
        percentile_high=float(summary.get("percentile_high", DEFAULT_PERCENTILE_HIGH)),
        smooth_xy=float(summary.get("smooth_xy", DEFAULT_SMOOTH_XY)),
        smooth_z=float(summary.get("smooth_z", DEFAULT_SMOOTH_Z)),
        preprocess_device=summary.get("preprocess_device", DEFAULT_PREPROCESS_DEVICE),
        max_gigabytes=float(summary.get("max_gigabytes", DEFAULT_MAX_GIGABYTES)),
        projection=projection,
    )
    y_chunk = max(1, int(y_chunk))

    records = read_records_csv(csv_path) if csv_path.is_file() else []
    measured_here = not records
    if records and "roundness" not in records[0]:
        # A CSV written before the shape columns existed: re-measure so the
        # roundness panel has data instead of an empty axis.
        print("Note   : the CSV has no roundness column, re-measuring the mask")
        records = []
        measured_here = True
    mask_2d = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    mask_source = TIFF.memmap(mask_path, mode="r")
    try:
        if not records:
            records = measure_mask_chunked(mask_source, 64, xy_um, z_um, placement)
            print(f"Measured: {len(records)} organoid(s) from the mask (no CSV)")
        for chunk_y0 in range(y0, y1, y_chunk):
            chunk_y1 = min(chunk_y0 + y_chunk, y1)
            block = np.asarray(mask_source[chunk_y0:chunk_y1, x0:x1, z0:z1])
            mask_2d[chunk_y0 - y0:chunk_y1 - y0] = block.max(axis=2)
            del block
    finally:
        del mask_source

    enface = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for chunk_y0 in range(y0, y1, y_chunk):
        chunk_y1 = min(chunk_y0 + y_chunk, y1)
        pages = read_volume(source, y_slice=(chunk_y0, chunk_y1),
                            max_gigabytes=settings.max_gigabytes)
        _, projected = oct_preprocess_volume(
            pages[:, x0:x1, z0:z1], settings, projection=projection
        )
        del pages
        enface[chunk_y0 - y0:chunk_y1 - y0] = projected

    section = None
    if records:
        biggest = max(records, key=lambda item: item["voxel_count"])
        y_ref = int(round(float(biggest.get("y_centroid_px", 0))))
        y_ref = min(max(y_ref, 0), y_total - 1)
        pages = read_volume(source, y_slice=(y_ref, y_ref + 1),
                            max_gigabytes=settings.max_gigabytes)
        data, _projection = oct_preprocess_volume(pages[:, x0:x1, z0:z1], settings)
        del pages
        section = np.asarray(data[:, 0, :])  # [Z, X] through that Y row

    make_qc_figure(
        paths["qc"],
        enface,
        mask_2d,
        section,
        records,
        title=(f"{source.name} | QC refreshed | mode={summary.get('mode', '?')} | "
               f"{len(records)} objects"),
        diameter_unit="um" if xy_um > 0 else "px",
        show_figures=bool(show),
    )
    print(f"QC     : {paths['qc']} (rebuilt from the mask and the summary JSON)")
    if measured_here and records:
        # The mask was the source of truth for the measurement above, so rewrite
        # the table as well: that is how an old CSV gets the shape columns
        # without re-running the segmentation.
        write_csv(csv_path, records)
        print(f"CSV    : {csv_path} (rewritten with the shape columns)")
    return paths["qc"]


# ---------------------------------------------------------------------------
# Stitched mosaics: strip processing
# ---------------------------------------------------------------------------
def format_duration(seconds):
    """Readable duration for the progress lines: ``12.3 s`` or ``2 min 03.4 s``."""
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)} min {rest:04.1f} s"


def prepare_strip(path, read_y0, read_y1, x0, x1, z0, z1, args, projection="max"):
    """Read and preprocess one Y strip of a mosaic.

    Returns ``(volume_zyx, projection_yx)``: the ``[Z, Ys, Xs]`` 0-1 volume for
    Cellpose and the strip's en-face projection for the QC panel - free here,
    because it is taken while the strip is still on the preprocessing device.
    """
    subvolume = read_volume(
        path, y_slice=(read_y0, read_y1), max_gigabytes=args.max_gigabytes
    )
    return oct_preprocess_volume(
        subvolume[:, x0:x1, z0:z1], args, projection=projection
    )


def process_volume_tiled(path, args, model, major, cp_version, tile_y, anisotropy,
                         placement):
    """Segment a stitched mosaic in Y strips into a memory-mapped mask.

    Every strip is read page-by-page from the TIFF and only the objects it owns
    are written (see ``keep_owned_labels``), so peak RAM is one strip plus that
    strip's label volume - not the whole mosaic. Returns ``(summary, records)``.
    """
    started = time.time()
    page_count, page_shape, _dtype = volume_page_info(path)
    y_total, x_total, z_total = int(page_count), int(page_shape[0]), int(page_shape[1])
    y0, y1, x0, x1, z0, z1 = placement
    strips = plan_y_strips(y0, y1, tile_y, args.tile_overlap_y)

    print("=" * 78)
    print(f"Input  : {path}")
    print(f"Volume : [Y, X, Z] = ({y_total}, {x_total}, {z_total})")
    print(f"Sub-vol: Y[{y0}:{y1}] X[{x0}:{x1}] Z[{z0}:{z1}]")
    print(
        f"Method : {len(strips)} Y strip(s) of {tile_y} rows, "
        f"+/-{args.tile_overlap_y} rows of context"
    )
    strip_estimate = estimate_working_set_gigabytes(
        (z1 - z0, min(tile_y, y1 - y0) + 2 * int(args.tile_overlap_y), x1 - x0),
        args.mode,
    )
    print(f"RAM    : ~{strip_estimate:.1f} GB per strip (budget "
          f"{float(args.max_ram_gigabytes):.1f} GB)")
    preprocess = preprocess_device_summary(getattr(args, "preprocess_device", None))
    print(f"Preproc: device={preprocess['device']} ({preprocess['reason']}), "
          f"{args.compression} {args.dynamic_range_db:g} dB, "
          f"smooth xy={args.smooth_xy:g} z={args.smooth_z:g} px")
    if int(args.save_seg_npy):
        print("Note   : --save-seg-npy is skipped in strip mode (the mask is "
              "written as a memory-mapped BigTIFF instead)")

    paths = output_paths(path, args.output_dir)
    if paths["mask"].exists():
        print(f"Replacing existing {paths['mask'].name}")
        paths["mask"].unlink()
    mask_volume = TIFF.memmap(
        paths["mask"], shape=(y_total, x_total, z_total), dtype=np.uint16, bigtiff=True
    )
    print(f"Mask   : {paths['mask'].name} (uint16, BigTIFF memory map)")

    enface = np.zeros((y_total, x_total), dtype=np.float32)
    mask_enface = np.zeros((y_total, x_total), dtype=np.int32)
    offset = 0
    dropped = 0
    # Per-strip timings. perf_counter is monotonic, so a clock adjustment during
    # a long mosaic run cannot produce a negative or inflated strip time.
    strip_seconds = []

    def report_strip_time(started, prepared, segmented, finished):
        """Print one strip's duration with its phases, mean and time left."""
        elapsed = finished - started
        strip_seconds.append(elapsed)
        mean = sum(strip_seconds) / len(strip_seconds)
        remaining = mean * (len(strips) - len(strip_seconds))
        print(
            f"    time   : {format_duration(elapsed)}"
            f" (read+preproc {prepared - started:.1f} s,"
            f" cellpose {segmented - prepared:.1f} s,"
            f" labels {finished - segmented:.1f} s)"
            f" | mean {mean:.1f} s/strip"
            + (f", ~{format_duration(remaining)} left" if remaining > 1.0 else "")
        )

    try:
        for index, (core_y0, core_y1, read_y0, read_y1) in enumerate(strips, start=1):
            strip_started = time.perf_counter()
            print(
                f"\n  strip {index}/{len(strips)}: core Y[{core_y0}:{core_y1}], "
                f"read Y[{read_y0}:{read_y1}] ({read_y1 - read_y0} rows)"
            )
            data, strip_projection = prepare_strip(
                path, read_y0, read_y1, x0, x1, z0, z1, args, projection="max"
            )
            prepared = time.perf_counter()
            accumulate_max_projection(
                enface, strip_projection, read_y0, read_y1, x0, x1
            )
            image, factor = downsample_volume(data, args.downsample)
            del data
            if args.dry_run:
                masks = dry_run_masks(image, args.min_size)
            else:
                masks, _flows, _styles = run_cellpose(model, args, major, image, anisotropy)
            segmented = time.perf_counter()
            del image

            strip_shape = (read_y1 - read_y0, x1 - x0, z1 - z0)
            labels_strip = restore_full_mask(
                masks,
                (0, strip_shape[0], 0, strip_shape[1], 0, strip_shape[2]),
                strip_shape,
                factor,
                True,
            )
            del masks
            labels_strip, strip_records, strip_dropped = keep_owned_labels(
                labels_strip, core_y0, core_y1, args.min_volume_voxels,
                origin=(read_y0, x0, z0),
                xy_um=args.xy_pixel_um, z_um=args.z_step_um,
            )
            dropped += strip_dropped

            if not strip_records:
                print("    no owned object in this strip")
                del labels_strip
                report_strip_time(strip_started, prepared, segmented,
                                  time.perf_counter())
                continue
            if offset + len(strip_records) > int(np.iinfo(np.uint16).max):
                raise ValueError(
                    "More than 65535 organoids in this mosaic, which does not fit "
                    "a uint16 mask. Raise --min-volume-voxels or use --downsample."
                )
            if index == 1:
                longest = max(int(r["y_extent_px"]) for r in strip_records)
                if longest > 2 * int(args.tile_overlap_y):
                    print(
                        f"    WARNING: the largest object spans {longest} Y rows, more "
                        f"than 2 x tile_overlap_y ({2 * int(args.tile_overlap_y)}). An "
                        "object that long can be split at a strip boundary - raise "
                        "--tile-overlap-y (and --tile-y to keep the RAM in check)."
                    )

            labels_strip[labels_strip > 0] += np.int32(offset)
            written = write_labels_into(
                mask_volume[read_y0:read_y1, x0:x1, z0:z1], labels_strip
            )
            strip_max = labels_strip.max(axis=2)
            np.copyto(
                mask_enface[read_y0:read_y1, x0:x1], strip_max,
                where=(mask_enface[read_y0:read_y1, x0:x1] == 0) & (strip_max > 0),
            )
            offset += len(strip_records)
            print(
                f"    kept {len(strip_records)} object(s), {written} voxels written, "
                f"label ids up to {offset}"
            )
            del labels_strip
            try:
                mask_volume.flush()
            except Exception:
                pass
            report_strip_time(strip_started, prepared, segmented, time.perf_counter())
    finally:
        try:
            mask_volume.flush()
        except Exception:
            pass
        del mask_volume

    if strip_seconds:
        total_seconds = sum(strip_seconds)
        print(
            f"Strips : {len(strip_seconds)} strip(s) in {format_duration(total_seconds)}"
            f" (mean {total_seconds / len(strip_seconds):.1f} s,"
            f" min {min(strip_seconds):.1f} s, max {max(strip_seconds):.1f} s)"
        )
    print(f"\nOrganoids: {offset} label(s) written (dropped {dropped} unowned/small "
          "objects from the strips)")
    # The written mask is the source of truth: re-measure it in chunks so the CSV
    # matches the mask exactly (a strip seam can hand a few voxels to the
    # neighbouring object).
    mask_source = TIFF.memmap(paths["mask"], mode="r")
    try:
        records = measure_mask_chunked(
            mask_source, 64, args.xy_pixel_um, args.z_step_um, placement
        )
    finally:
        del mask_source
    print(f"Measured: {len(records)} organoid(s) in the mask")
    diameters = sorted(float(r["equivalent_diameter_um"]) for r in records)
    if diameters:
        print(
            f"Sizes (equivalent diameter): min {diameters[0]:g}, median "
            f"{diameters[len(diameters) // 2]:g}, max {diameters[-1]:g} "
            f"{'um' if args.xy_pixel_um > 0 else 'px'}"
        )
    else:
        print("No objects passed --min-volume-voxels; relax it or lower "
              "--cellprob-threshold.")

    # X-Z section for the QC figure: read the single page of the biggest object.
    section = None
    if records:
        biggest = max(records, key=lambda item: item["voxel_count"])
        y_ref = int(round(float(biggest.get("y_centroid_px", 0))))
        y_ref = min(max(y_ref, 0), y_total - 1)
        page = read_volume(path, y_slice=(y_ref, y_ref + 1),
                           max_gigabytes=args.max_gigabytes)
        page = np.asarray(page[:, x0:x1, z0:z1], dtype=np.float32)
        page = compress_volume(page, args.compression, args.dynamic_range_db)
        page = normalise_percentiles(page, args.percentile_low, args.percentile_high)
        page = smooth_volume(oct_to_cellpose(page), args.smooth_xy, args.smooth_z)
        # The read is one Y row, so ``page`` is [Z, 1, X]: drop the Y axis, not
        # the Z axis (``page[0]`` used to leave a 1-row image in the panel).
        section = np.asarray(page[:, 0, :])  # [Z, X]

    write_csv(paths["csv"], records)
    make_qc_figure(
        paths["qc"],
        enface[y0:y1, x0:x1],
        mask_enface[y0:y1, x0:x1],
        section,
        records,
        title=(f"{Path(path).name} | mode={args.mode} | model={args.model} | "
               f"{len(strips)} Y strips of {tile_y}"),
        diameter_unit="um" if args.xy_pixel_um > 0 else "px",
        show_figures=args.show_figures,
    )

    summary = {
        "input": path,
        "mask": paths["mask"],
        "csv": paths["csv"],
        "qc": paths["qc"],
        "volume_shape_yxz": [y_total, x_total, z_total],
        "subvolume": {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "z0": z0, "z1": z1},
        "processing": "y-strips",
        "strip_count": len(strips),
        "strip_seconds": [round(value, 1) for value in strip_seconds],
        "strip_seconds_mean": (
            round(sum(strip_seconds) / len(strip_seconds), 1) if strip_seconds else None
        ),
        "tile_y": int(tile_y),
        "tile_overlap_y": int(args.tile_overlap_y),
        "preprocess_device": preprocess["device"],
        "preprocess_device_reason": preprocess["reason"],
        "estimated_strip_working_set_gb": round(strip_estimate, 2),
        "mode": args.mode,
        "model": args.model,
        "cellpose_version": cp_version,
        "cellpose_api_major": major,
        "gpu": bool(args.gpu),
        "dry_run": bool(args.dry_run),
        "anisotropy": anisotropy,
        "anisotropy_source": resolve_anisotropy(args)[1],
        "xy_pixel_um": float(args.xy_pixel_um),
        "z_step_um": float(args.z_step_um),
        "compression": args.compression,
        "dynamic_range_db": float(args.dynamic_range_db),
        "percentile_low": float(args.percentile_low),
        "percentile_high": float(args.percentile_high),
        "smooth_xy": float(args.smooth_xy),
        "smooth_z": float(args.smooth_z),
        "downsample": int(args.downsample),
        "projection": args.projection,
        "flow_threshold": float(args.flow_threshold),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "diameter": float(args.diameter),
        "stitch_threshold": float(args.stitch_threshold),
        "niter": int(args.niter),
        "min_volume_voxels": int(args.min_volume_voxels),
        "organoid_count": len(records),
        "dropped_small_or_unowned_objects": int(dropped),
        "roundness": roundness_summary(records),
        "elapsed_s": round(time.time() - started, 1),
    }
    write_summary_json(paths["json"], summary)
    print(f"Wrote  : {paths['mask'].name}\n         {paths['csv'].name}"
          f"\n         {paths['qc'].name}\n         {paths['json'].name}")
    return summary, records


# ---------------------------------------------------------------------------
# Stitched mosaics: streamed 2D projection
# ---------------------------------------------------------------------------
def process_projection_streamed(path, args, model, major, cp_version, tile_y,
                                placement):
    """2D projection mode for a mosaic too large to load: project strip by strip.

    The en-face projection is accumulated in raw amplitude, page by page. A
    max/sum over Z is monotone, so projecting first and compressing afterwards
    gives the same image as compressing first - which means no strip seams and
    no need for global statistics. Cellpose then runs once on the 2D projection.
    Returns ``(summary, records)``.
    """
    started = time.time()
    page_count, page_shape, _dtype = volume_page_info(path)
    y_total, x_total, z_total = int(page_count), int(page_shape[0]), int(page_shape[1])
    y0, y1, x0, x1, z0, z1 = placement
    tile_y = int(tile_y) if int(tile_y) > 0 else 512
    strips = plan_y_strips(y0, y1, tile_y, args.tile_overlap_y)

    print("=" * 78)
    print(f"Input  : {path}")
    print(f"Volume : [Y, X, Z] = ({y_total}, {x_total}, {z_total})")
    print(f"Sub-vol: Y[{y0}:{y1}] X[{x0}:{x1}] Z[{z0}:{z1}]")
    print(
        f"Method : {args.projection} projection accumulated from {len(strips)} "
        f"streamed strip(s) of {tile_y} rows (2D projection mode)"
    )

    use_mean = str(args.projection).lower() == "mean"
    preprocess = preprocess_device_summary(getattr(args, "preprocess_device", None))
    print(f"Preproc: device={preprocess['device']} ({preprocess['reason']})")
    projection = np.zeros((y_total, x_total), dtype=np.float32)
    plane_hits = np.zeros((y_total, x_total), dtype=np.uint16)
    for index, (core_y0, core_y1, read_y0, read_y1) in enumerate(strips, start=1):
        print(f"  strip {index}/{len(strips)}: Y[{read_y0}:{read_y1}]")
        page = read_volume(path, y_slice=(read_y0, read_y1),
                           max_gigabytes=args.max_gigabytes)
        patch = np.asarray(page[:, x0:x1, z0:z1], dtype=np.float32)
        del page
        patch = smooth_volume(oct_to_cellpose(patch), args.smooth_xy, args.smooth_z)
        if use_mean:
            # Sum the raw values and count how many planes hit each pixel, so
            # overlapping strip reads cannot be counted twice.
            projection[read_y0:read_y1, x0:x1] += patch.sum(axis=0, dtype=np.float32)
            plane_hits[read_y0:read_y1, x0:x1] += np.uint16(patch.shape[0])
        else:
            accumulate_max_projection(
                projection, project_volume(patch, "max"), read_y0, read_y1, x0, x1
            )
        del patch

    if use_mean:
        np.divide(projection, np.maximum(plane_hits, 1), out=projection)
        print("  note: the mean is accumulated in linear amplitude, then compressed")
    del plane_hits

    # Compress the full projection (the dB shift uses the global maximum, which is
    # the same maximum the crop sees), then keep only the selected window - the
    # same [Y, X] image the single-pass path hands to Cellpose.
    compressed = compress_volume(projection, args.compression, args.dynamic_range_db,
                                 device=getattr(args, "preprocess_device", None))
    del projection
    image = np.ascontiguousarray(compressed[y0:y1, x0:x1])
    del compressed
    image = normalise_percentiles(image, args.percentile_low, args.percentile_high,
                                  device=getattr(args, "preprocess_device", None))
    enface = image  # [Y, X] window, for the QC panel
    image, factor = downsample_volume(image, args.downsample)
    print(f"Input to Cellpose: projection [Y, X] = {tuple(int(v) for v in np.shape(image))}")

    if args.dry_run:
        masks = dry_run_masks(image, args.min_size)
    else:
        masks, _flows, _styles = run_cellpose(model, args, major, image, 0.0)
    del image

    mask_full = restore_full_mask(
        masks, placement, (y_total, x_total, z_total), factor, False
    )
    del masks
    mask_full, dropped = filter_small_labels(mask_full, args.min_volume_voxels)
    print(f"Organoids: {int(mask_full.max()) if mask_full.size else 0}")
    records = label_records(mask_full, args.xy_pixel_um, args.z_step_um)

    paths = output_paths(path, args.output_dir)
    paths["mask"] = paths["mask"].with_name(
        paths["mask"].name.replace("cellpose_mask", "cellpose_maskXY")
    )
    write_mask_tiff(paths["mask"], mask_full)
    write_csv(paths["csv"], records)
    make_qc_figure(
        paths["qc"],
        enface,
        np.asarray(mask_full)[y0:y1, x0:x1],
        None,
        records,
        title=(f"{Path(path).name} | mode=projection ({args.projection}) | "
               f"streamed | model={args.model}"),
        diameter_unit="um" if args.xy_pixel_um > 0 else "px",
        show_figures=args.show_figures,
    )

    summary = {
        "input": path,
        "mask": paths["mask"],
        "csv": paths["csv"],
        "qc": paths["qc"],
        "volume_shape_yxz": [y_total, x_total, z_total],
        "subvolume": {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "z0": z0, "z1": z1},
        "processing": "streamed-projection",
        "preprocess_device": preprocess["device"],
        "preprocess_device_reason": preprocess["reason"],
        "strip_count": len(strips),
        "tile_y": int(tile_y),
        "mode": "projection",
        "projection": args.projection,
        "model": args.model,
        "cellpose_version": cp_version,
        "cellpose_api_major": major,
        "gpu": bool(args.gpu),
        "dry_run": bool(args.dry_run),
        "xy_pixel_um": float(args.xy_pixel_um),
        "z_step_um": float(args.z_step_um),
        "compression": args.compression,
        "dynamic_range_db": float(args.dynamic_range_db),
        "percentile_low": float(args.percentile_low),
        "percentile_high": float(args.percentile_high),
        "smooth_xy": float(args.smooth_xy),
        "smooth_z": float(args.smooth_z),
        "downsample": int(factor),
        "flow_threshold": float(args.flow_threshold),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "diameter": float(args.diameter),
        "min_volume_voxels": int(args.min_volume_voxels),
        "organoid_count": int(mask_full.max()) if mask_full.size else 0,
        "dropped_small_objects": int(dropped),
        "elapsed_s": round(time.time() - started, 1),
    }
    write_summary_json(paths["json"], summary)
    print(f"Wrote  : {paths['mask'].name}\n         {paths['csv'].name}"
          f"\n         {paths['qc'].name}\n         {paths['json'].name}")
    return summary, records


# ---------------------------------------------------------------------------
# Per-volume driver
# ---------------------------------------------------------------------------
def process_volume(path, args, model, major, cp_version):
    """Run the whole pipeline for one volume, choosing the memory strategy.

    Volumes that fit the RAM budget are read and segmented with one Cellpose
    call. A stitched mosaic that does not fit is processed in Y strips
    (``process_volume_tiled``), or streamed page-by-page for 2D projection mode
    (``process_projection_streamed``). Returns ``(summary_dict, records)``.
    """
    anisotropy, anisotropy_source = resolve_anisotropy(args)
    started = time.time()
    page_count, page_shape, page_dtype = volume_page_info(path)
    full_shape = (int(page_count), int(page_shape[0]), int(page_shape[1]))
    placement = resolve_placement(full_shape, args.crop, args.depth_range)
    y0, y1, x0, x1, z0, z1 = placement
    shape_zyx = (z1 - z0, y1 - y0, x1 - x0)

    tile_y, reason = resolve_tile_y(
        shape_zyx, args.mode, args.tile_y, args.max_ram_gigabytes, args.tile_overlap_y
    )
    print("=" * 78)
    print(f"Input  : {path}")
    print(
        f"Volume : [Y, X, Z] = {full_shape}, {page_dtype}, "
        f"{os.path.getsize(path) / 1e9:.3f} GB"
    )
    print(f"Sub-vol: Y[{y0}:{y1}] X[{x0}:{x1}] Z[{z0}:{z1}] -> {shape_zyx} as [Z, Y, X]")
    print(f"Tiling : {reason}")

    if tile_y and str(args.mode) != "projection":
        return process_volume_tiled(
            path, args, model, major, cp_version, tile_y, anisotropy, placement
        )
    if str(args.mode) == "projection" and (
        estimate_working_set_gigabytes(shape_zyx, "projection")
        > float(args.max_ram_gigabytes)
    ):
        return process_projection_streamed(
            path, args, model, major, cp_version, tile_y, placement
        )

    # Read only the rows we need, then crop X and Z in memory.
    subvolume = read_volume(path, y_slice=(y0, y1), max_gigabytes=args.max_gigabytes)
    print(f"Loaded : {tuple(int(v) for v in np.shape(subvolume[:, x0:x1, z0:z1]))} "
          "(Y, X, Z) into RAM")
    preprocess = preprocess_device_summary(getattr(args, "preprocess_device", None))
    data, enface = oct_preprocess_volume(
        subvolume[:, x0:x1, z0:z1], args, projection=args.projection
    )
    del subvolume
    print(
        f"Preproc: {args.compression} compression, {args.dynamic_range_db:g} dB range, "
        f"percentiles [{args.percentile_low:g}, {args.percentile_high:g}], "
        f"smoothing xy={args.smooth_xy:g} z={args.smooth_z:g} px, "
        f"device={preprocess['device']} ({preprocess['reason']})"
    )

    is_volume = str(args.mode) in ("3d", "2d-stitch")
    if is_volume:
        image, factor = downsample_volume(data, args.downsample)
    else:
        # projection mode: Cellpose gets the 2D en-face projection, [Y, X]
        image, factor = downsample_volume(enface, args.downsample)
    print(
        f"Input to Cellpose: {'volume [Z, Y, X]' if is_volume else 'projection [Y, X]'}"
        f" = {tuple(int(v) for v in np.shape(image))}"
    )
    if factor > 1:
        print(f"Downsampled by {factor}")

    anisotropy, anisotropy_source = resolve_anisotropy(args)
    if str(args.mode) == "3d":
        if anisotropy_source == "unset":
            print(
                "WARNING: anisotropy is unset (no usable --xy-pixel-um/--z-step-um), "
                "so Cellpose assumes Z is sampled like XY - which is never true for "
                "OCT. Set both pixel sizes or pass --anisotropy explicitly."
            )
        else:
            print(
                f"anisotropy = {anisotropy:.3f} ({anisotropy_source}) "
                f"[Z {args.z_step_um:g} um vs XY {args.xy_pixel_um:g} um]"
            )

    if args.dry_run:
        print("DRY RUN: skipping Cellpose, thresholding the normalised image instead")
        masks = dry_run_masks(image, args.min_size)
        flows, styles = None, None
    else:
        masks, flows, styles = run_cellpose(model, args, major, image, anisotropy)
    print(f"Raw masks: shape {tuple(int(v) for v in np.shape(masks))}")

    mask_full = restore_full_mask(masks, placement, full_shape, factor, is_volume)
    del masks
    mask_full, dropped = filter_small_labels(mask_full, args.min_volume_voxels)
    if dropped:
        print(
            f"Dropped {dropped} object(s) smaller than "
            f"--min-volume-voxels={args.min_volume_voxels}"
        )
    object_count = int(mask_full.max()) if mask_full.size else 0
    print(f"Organoids: {object_count}")

    records = label_records(mask_full, args.xy_pixel_um, args.z_step_um)
    diameters = sorted(float(r["equivalent_diameter_um"]) for r in records)
    if diameters:
        print(
            f"Sizes (equivalent diameter): min {diameters[0]:g}, median "
            f"{diameters[len(diameters) // 2]:g}, max {diameters[-1]:g} "
            f"{'um' if args.xy_pixel_um > 0 else 'px'}"
        )
    else:
        print("No objects passed --min-volume-voxels; relax it or lower "
              "--cellprob-threshold.")

    # X-Z section for the QC figure: through the biggest organoid's Y centroid.
    section = None
    if is_volume and data.ndim == 3 and data.shape[0] > 1:
        if records:
            biggest = max(records, key=lambda item: item["voxel_count"])
            y_ref = int(round(float(biggest.get("y_centroid_px", 0)) - y0))
        else:
            y_ref = data.shape[1] // 2
        y_ref = min(max(y_ref, 0), data.shape[1] - 1)
        section = np.asarray(data[:, y_ref, :])  # [Z, X]

    paths = output_paths(path, args.output_dir)
    if not is_volume:
        paths["mask"] = paths["mask"].with_name(
            paths["mask"].name.replace("cellpose_mask", "cellpose_maskXY")
        )
    write_mask_tiff(paths["mask"], mask_full)
    write_csv(paths["csv"], records)
    mask_2d = mask_full if mask_full.ndim == 2 else np.max(mask_full, axis=2)
    make_qc_figure(
        paths["qc"],
        enface,
        np.asarray(mask_2d)[y0:y1, x0:x1],
        section,
        records,
        title=f"{Path(path).name} | mode={args.mode} | model={args.model}",
        diameter_unit="um" if args.xy_pixel_um > 0 else "px",
        show_figures=args.show_figures,
    )
    if args.save_seg_npy:
        save_seg_npy(paths["seg"], mask_full, full_shape, flows=flows, styles=styles)

    summary = {
        "input": path,
        "mask": paths["mask"],
        "csv": paths["csv"],
        "qc": paths["qc"],
        "volume_shape_yxz": [int(v) for v in full_shape],
        "subvolume": {
            "y0": y0, "y1": y1, "x0": x0, "x1": x1, "z0": z0, "z1": z1,
        },
        "processing": "single-pass",
        "preprocess_device": preprocess["device"],
        "preprocess_device_reason": preprocess["reason"],
        "mode": args.mode,
        "model": args.model,
        "cellpose_version": cp_version,
        "cellpose_api_major": major,
        "gpu": bool(args.gpu),
        "dry_run": bool(args.dry_run),
        "anisotropy": anisotropy,
        "anisotropy_source": anisotropy_source,
        "xy_pixel_um": float(args.xy_pixel_um),
        "z_step_um": float(args.z_step_um),
        "compression": args.compression,
        "dynamic_range_db": float(args.dynamic_range_db),
        "percentile_low": float(args.percentile_low),
        "percentile_high": float(args.percentile_high),
        "smooth_xy": float(args.smooth_xy),
        "smooth_z": float(args.smooth_z),
        "downsample": int(factor),
        "projection": args.projection,
        "flow_threshold": float(args.flow_threshold),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "diameter": float(args.diameter),
        "stitch_threshold": float(args.stitch_threshold),
        "niter": int(args.niter),
        "min_volume_voxels": int(args.min_volume_voxels),
        "organoid_count": object_count,
        "dropped_small_objects": int(dropped),
        "roundness": roundness_summary(records),
        "elapsed_s": round(time.time() - started, 1),
    }
    write_summary_json(paths["json"], summary)
    print(f"Wrote  : {paths['mask'].name}\n         {paths['csv'].name}"
          f"\n         {paths['qc'].name}\n         {paths['json'].name}")
    return summary, records


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Segment organoids in an OCT [Y, X, Z] TIFF volume with Cellpose. "
            "The volume is log/dB compressed, transposed to Cellpose's [Z, Y, X] "
            "layout, segmented, and the labels are written back in the input "
            "[Y, X, Z] orientation together with a statistics CSV and a QC figure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", nargs="?", default=DEFAULT_INPUT_PATH,
                        help="One volume TIFF, or a folder scanned with --pattern")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="Glob used when input_path is a folder")
    parser.add_argument("--recursive", action="store_true", default=DEFAULT_RECURSIVE,
                        help="Scan sub-folders as well when input_path is a folder")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the outputs. A relative name is "
                             "created under each input file's folder "
                             "(default 'segmentation_results'), an absolute path "
                             "is used as-is, '' writes next to the input file")
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        choices=("3d", "2d-stitch", "projection"),
                        help="Cellpose 3D flows, per-plane labels stitched in Z, "
                             "or 2D on the en-face projection")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Cellpose 4: cpsam_v2/cpsam/cpdino/cpdino-vitb | "
                             "Cellpose 3: cyto3/nuclei")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Force CPU inference")
    parser.add_argument("--diameter", type=float, default=DEFAULT_DIAMETER,
                        help="Expected organoid diameter in px (0 = let Cellpose decide)")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE,
                        help="Cellpose minimum object size (voxels for 3D, px for 2D)")
    parser.add_argument("--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD,
                        help="Raise to keep more (worse shaped) objects")
    parser.add_argument("--cellprob-threshold", type=float, default=DEFAULT_CELLPROB_THRESHOLD,
                        help="Lower to accept dimmer objects")
    parser.add_argument("--niter", type=int, default=DEFAULT_NITER,
                        help="Flow dynamics iterations (0 = Cellpose default)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Tiles per GPU batch (lower if you run out of VRAM)")
    parser.add_argument("--preprocess-device", default=DEFAULT_PREPROCESS_DEVICE,
                        choices=("auto", "cpu", "gpu"),
                        help="Run compression/percentiles/smoothing on the GPU "
                             "('auto' = when torch has a CUDA device)")
    parser.add_argument("--stitch-threshold", type=float, default=DEFAULT_STITCH_THRESHOLD,
                        help="IoU used to stitch Z planes in --mode 2d-stitch")
    parser.add_argument("--projection", default=DEFAULT_PROJECTION, choices=("max", "mean"),
                        help="En-face projection type (used for --mode projection "
                             "and for the QC panel)")
    parser.add_argument("--compression", default=DEFAULT_COMPRESSION,
                        choices=("db", "log", "none"),
                        help="OCT intensity compression before Cellpose")
    parser.add_argument("--dynamic-range-db", type=float, default=DEFAULT_DYNAMIC_RANGE_DB,
                        help="dB window kept above the brightest voxel (db compression)")
    parser.add_argument("--percentile-low", type=float, default=DEFAULT_PERCENTILE_LOW)
    parser.add_argument("--percentile-high", type=float, default=DEFAULT_PERCENTILE_HIGH)
    parser.add_argument("--smooth-xy", type=float, default=DEFAULT_SMOOTH_XY,
                        help="Gaussian sigma in XY px to reduce speckle (0 = off)")
    parser.add_argument("--smooth-z", type=float, default=DEFAULT_SMOOTH_Z,
                        help="Gaussian sigma in Z px (0 = off)")
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE,
                        help="Integer stride for very large volumes (1 = off)")
    parser.add_argument("--crop", nargs=4, type=int, default=DEFAULT_CROP,
                        metavar=("Y0", "Y1", "X0", "X1"),
                        help="Process only this Y/X window (mask is written full size)")
    parser.add_argument("--depth-range", nargs=2, type=int, default=DEFAULT_DEPTH_RANGE,
                        metavar=("Z0", "Z1"),
                        help="Process only this depth range (mask is written full size)")
    parser.add_argument("--xy-pixel-um", type=float, default=DEFAULT_XY_PIXEL_UM,
                        help="Lateral pixel size (0 = report pixel units)")
    parser.add_argument("--z-step-um", type=float, default=DEFAULT_Z_STEP_UM,
                        help="Axial pixel size, HardwareSpecs.DEFAULT_AXIAL_PIXEL_SIZE_UM")
    parser.add_argument("--anisotropy", type=float, default=DEFAULT_ANISOTROPY,
                        help="Override z_step/xy_pixel (Cellpose: 2.0 if Z is sampled "
                             "half as densely as XY)")
    parser.add_argument("--min-volume-voxels", type=int, default=DEFAULT_MIN_VOLUME_VOXELS,
                        help="Drop objects smaller than this from mask and CSV")
    parser.add_argument("--max-gigabytes", type=float, default=DEFAULT_MAX_GIGABYTES,
                        help="Refuse to load inputs larger than this")
    parser.add_argument("--tile-y", type=int, default=DEFAULT_TILE_Y,
                        help="Y strip height for a large mosaic (0 = auto: use one "
                             "Cellpose call when it fits --max-ram-gigabytes)")
    parser.add_argument("--tile-overlap-y", type=int, default=DEFAULT_TILE_OVERLAP_Y,
                        help="Context rows added around each strip; must exceed the "
                             "largest organoid radius in Y, otherwise it can be split")
    parser.add_argument("--max-ram-gigabytes", type=float, default=DEFAULT_MAX_RAM_GIGABYTES,
                        help="Working-set budget for one Cellpose call (~32 bytes per "
                             "voxel); drives the automatic tiling decision")
    parser.add_argument("--show-figures", action="store_true", default=DEFAULT_SHOW_FIGURES,
                        help="Also display the QC figure (Spyder Plots pane; the PNG "
                             "is always written)")
    parser.add_argument("--save-seg-npy", action="store_true",
                        help="Also write a *_seg.npy loadable by the Cellpose GUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test reading/preprocessing/writing with a threshold "
                             "stand-in instead of Cellpose (no GPU needed)")
    parser.add_argument("--update-qc", action="store_true", default=DEFAULT_UPDATE_QC,
                        help="Only rebuild the QC figure(s) from the mask, CSV and "
                             "summary already written for the input volume(s)")
    parser.add_argument("--use-command-line-args", dest="use_command_line_args",
                        action="store_true",
                        help="Ignore the DEFAULT_* block and use these arguments "
                             "(the .bat launcher passes this)")
    parser.add_argument("--no-subprocess-fallback", dest="use_subprocess_fallback",
                        action="store_false",
                        default=DEFAULT_USE_SUBPROCESS_FALLBACK,
                        help="Do not re-run in a fresh process when torch cannot be "
                             "imported in this one (see the DLL bootstrap)")
    return parser


def default_settings():
    """All run settings as an ``argparse.Namespace`` built from the ``DEFAULT_*``.

    This is what makes the script Spyder-friendly: pressing Run uses this
    namespace, and ``segment_organoids(...)`` overrides individual entries
    without any command-line parsing.
    """
    return argparse.Namespace(
        input_path=DEFAULT_INPUT_PATH,
        pattern=DEFAULT_PATTERN,
        recursive=DEFAULT_RECURSIVE,
        output_dir=DEFAULT_OUTPUT_DIR,
        mode=DEFAULT_MODE,
        model=DEFAULT_MODEL,
        gpu=DEFAULT_GPU,
        diameter=DEFAULT_DIAMETER,
        min_size=DEFAULT_MIN_SIZE,
        flow_threshold=DEFAULT_FLOW_THRESHOLD,
        cellprob_threshold=DEFAULT_CELLPROB_THRESHOLD,
        niter=DEFAULT_NITER,
        batch_size=DEFAULT_BATCH_SIZE,
        preprocess_device=DEFAULT_PREPROCESS_DEVICE,
        stitch_threshold=DEFAULT_STITCH_THRESHOLD,
        projection=DEFAULT_PROJECTION,
        compression=DEFAULT_COMPRESSION,
        dynamic_range_db=DEFAULT_DYNAMIC_RANGE_DB,
        percentile_low=DEFAULT_PERCENTILE_LOW,
        percentile_high=DEFAULT_PERCENTILE_HIGH,
        smooth_xy=DEFAULT_SMOOTH_XY,
        smooth_z=DEFAULT_SMOOTH_Z,
        downsample=DEFAULT_DOWNSAMPLE,
        crop=DEFAULT_CROP,
        depth_range=DEFAULT_DEPTH_RANGE,
        xy_pixel_um=DEFAULT_XY_PIXEL_UM,
        z_step_um=DEFAULT_Z_STEP_UM,
        anisotropy=DEFAULT_ANISOTROPY,
        min_volume_voxels=DEFAULT_MIN_VOLUME_VOXELS,
        max_gigabytes=DEFAULT_MAX_GIGABYTES,
        tile_y=DEFAULT_TILE_Y,
        tile_overlap_y=DEFAULT_TILE_OVERLAP_Y,
        max_ram_gigabytes=DEFAULT_MAX_RAM_GIGABYTES,
        show_figures=DEFAULT_SHOW_FIGURES,
        update_qc=DEFAULT_UPDATE_QC,
        use_subprocess_fallback=DEFAULT_USE_SUBPROCESS_FALLBACK,
        save_seg_npy=False,
        dry_run=False,
    )



def print_organoid_table(records, limit=20):
    """Print the largest organoids of the current volume."""
    if not records:
        print("  (no organoids to list)")
        return
    ordered = sorted(records, key=lambda item: item["voxel_count"], reverse=True)
    print(f"  {'id':>4} {'voxels':>11} {'volume_um3':>12} {'diam_um':>9} "
          f"{'y':>8} {'x':>8} {'z':>8}")
    for record in ordered[:limit]:
        print(
            f"  {record['organoid_id']:>4} {record['voxel_count']:>11} "
            f"{record.get('volume_um3', ''):>12} "
            f"{record.get('equivalent_diameter_um', ''):>9} "
            f"{record.get('y_centroid_px', ''):>8} "
            f"{record.get('x_centroid_px', ''):>8} "
            f"{record.get('z_centroid_px', ''):>8}"
        )
    if len(ordered) > limit:
        print(f"  ... {len(ordered) - limit} more (see the CSV)")


def settings_to_cli_arguments(settings):
    """Render a settings namespace as the arguments a fresh process needs.

    Only values that differ from ``default_settings()`` are passed, so the child
    command stays readable. Boolean flags are emitted only when they are set and
    list options (``--crop``, ``--depth-range``) are expanded.
    """
    parser = build_argument_parser()
    defaults = default_settings()
    arguments = [str(getattr(settings, "input_path", ""))]
    for option in parser._actions:  # noqa: SLF001 - argparse has no public API
        flag = next(
            (name for name in option.option_strings if name.startswith("--")), None
        )
        if flag is None or option.dest in ("help", "use_command_line_args",
                                           "input_path"):
            continue
        value = getattr(settings, option.dest, None)
        if option.dest == "gpu":
            if not value:
                arguments.append("--no-gpu")
            continue
        if getattr(option, "nargs", None) == 0:  # store_true / store_false
            if bool(value) == bool(getattr(option, "const", True)):
                arguments.append(flag)
            continue
        if value is None or value == getattr(defaults, option.dest, None):
            continue
        arguments.append(flag)
        if isinstance(value, (list, tuple)):
            arguments.extend(str(item) for item in value)
        else:
            arguments.append(str(value))
    return arguments


def read_records_csv(path):
    """Read a per-organoid statistics CSV back into records."""
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = {}
            for key, value in row.items():
                if value is None or value == "":
                    record[key] = None
                    continue
                try:
                    record[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = value
            records.append(record)
    return records


def collect_results_from_outputs(settings):
    """Load the summary and organoid table of every input volume of a run."""
    results = []
    for path in iter_input_paths(
        settings.input_path, settings.pattern, settings.recursive
    ):
        paths = output_paths(path, settings.output_dir)
        summary = {}
        if paths["json"].is_file():
            try:
                with open(paths["json"], "r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            except (OSError, ValueError):
                summary = {}
        records = read_records_csv(paths["csv"])
        if summary or records:
            results.append({"summary": summary, "records": records})
    return results


def should_run_in_subprocess(settings):
    """True when this process cannot import torch but a fresh one can.

    A fresh interpreter starts with an empty DLL table and therefore loads
    torch's own OpenMP/MKL copies; a long-lived Spyder kernel has usually already
    loaded the conda-forge builds and cannot be repaired from the inside. Only
    cases where re-running elsewhere actually helps return True.
    """
    if os.environ.get(SUBPROCESS_GUARD_VAR) == "1":
        return False  # already the fresh child process
    if not getattr(settings, "use_subprocess_fallback",
                   DEFAULT_USE_SUBPROCESS_FALLBACK):
        return False
    if settings.dry_run:
        return False  # a dry run never touches Cellpose/torch
    if TORCH_DLL_BOOTSTRAP.get("torch_import") == "ok":
        return False  # torch imports fine here, there is nothing to work around
    if not torch_dll_directories():
        return False  # torch is not installed: say so instead of re-running
    return True


def run_in_subprocess(settings):
    """Re-run this script in a fresh interpreter and return its results.

    The child prints its progress here, writes the usual outputs into the output
    folder, and its summary + organoid table are loaded back, so the return value
    is the same as for an in-process run.
    """
    try:
        script_path = Path(__file__).resolve()
    except NameError:  # only when the module was exec'd without __file__
        script_path = Path(sys.argv[0]).resolve()
    arguments = [
        sys.executable, "-u", str(script_path), "--use-command-line-args",
    ] + settings_to_cli_arguments(settings)

    child_environment = dict(os.environ)
    child_environment[SUBPROCESS_GUARD_VAR] = "1"
    child_environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print("=" * 78)
    print(
        "torch cannot be imported in this Python process (Windows DLL conflict\n"
        "with the conda-forge OpenMP/MKL build loaded by this kernel), but a\n"
        "fresh interpreter can - so this run continues in a new process:"
    )
    print("  " + " ".join(arguments))
    print("=" * 78)

    with subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_environment,
    ) as process:
        for line in process.stdout:
            print(line.rstrip())
        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(
            f"The fresh-process run failed with exit code {return_code}; see the "
            "output above."
        )
    results = collect_results_from_outputs(settings)
    if not results:
        raise SystemExit(
            "The fresh-process run finished but no outputs were found in the "
            "output folder."
        )
    return results


def run(settings):
    """Segment every input volume described by ``settings``.

    Returns a list of ``{"summary": ..., "records": ...}`` dictionaries, so the
    results can be inspected further in the Spyder/IPython console.
    """
    if getattr(settings, "update_qc", False):
        # A QC refresh needs no Cellpose and no torch, so it stays in this
        # process (no subprocess fallback) and costs seconds per volume.
        input_paths = iter_input_paths(
            settings.input_path, settings.pattern, settings.recursive
        )
        for index, path in enumerate(input_paths, start=1):
            print(f"\n[{index}/{len(input_paths)}] {path.name}")
            try:
                update_qc_figure(path, settings.output_dir,
                                 show=settings.show_figures)
            except Exception as error:
                print(f"FAILED to refresh {path.name}: "
                      f"{type(error).__name__}: {error}")
        return collect_results_from_outputs(settings)

    if should_run_in_subprocess(settings):
        return run_in_subprocess(settings)

    input_paths = iter_input_paths(
        settings.input_path, settings.pattern, settings.recursive
    )
    print(f"Volumes to process: {len(input_paths)}")
    if TORCH_DLL_BOOTSTRAP.get("applied"):
        print(
            "Windows DLL search: torch's own directory/directories placed first "
            f"({len(TORCH_DLL_BOOTSTRAP.get('directories', []))} path(s)); "
            f"torch import: {TORCH_DLL_BOOTSTRAP.get('torch_import', 'n/a')}"
        )

    model, major, cp_version = None, 0, "n/a"
    if settings.dry_run:
        print("dry-run: Cellpose will not be loaded or run")
    else:
        model, major, cp_version = load_cellpose_model(settings.model, settings.gpu)
        print(f"Cellpose {cp_version} (API v{major}), gpu={bool(settings.gpu)}, "
              f"model={settings.model}")

    results = []
    for index, path in enumerate(input_paths, start=1):
        print(f"\n[{index}/{len(input_paths)}]")
        destination = output_paths(path, settings.output_dir)
        print(
            f"Output : {destination['folder']}"
            + ("" if str(settings.output_dir or "").strip()
               else "   (same folder as the input; set --output-dir / output_dir= "
                    "to change)")
        )
        try:
            summary, records = process_volume(
                path, settings, model, major, cp_version
            )
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"FAILED on {path.name}: {type(error).__name__}: {error}")
            continue
        results.append({"summary": summary, "records": records})
        if DEFAULT_SHOW_ORGANOID_TABLE:
            print_organoid_table(records)

    if not results:
        raise SystemExit("No volume was processed successfully.")
    total = sum(int(item["summary"]["organoid_count"]) for item in results)
    print(f"\nDone: {len(results)}/{len(input_paths)} volume(s) processed, "
          f"{total} organoid(s) in total.")
    return results


def segment_organoids(**overrides):
    """Spyder/console entry point: segment with the ``DEFAULT_*`` settings.

    Any setting can be overridden by keyword, e.g.

        segment_organoids()                                  # DEFAULT_INPUT_PATH
        segment_organoids(mode="projection", smooth_xy=1)
        segment_organoids(input_path=r"E:\\...\\stitched-Y5436-X6096-Z66.tif",
                          tile_y=512, depth_range=(5, 70))
        segment_organoids(input_path=r"E:\\...\\BJRcellcluster", recursive=True)

    Allowed keywords are exactly the ``DEFAULT_*`` names and the command-line
    option names with dashes replaced by underscores (see ``default_settings``).
    Returns the list of results produced by :func:`run`.
    """
    settings = default_settings()
    unknown = sorted(key for key in overrides if not hasattr(settings, key))
    if unknown:
        raise TypeError(
            f"Unknown setting(s) {unknown}. Valid names: "
            f"{sorted(vars(settings))}"
        )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return run(settings)


def main(settings=None):
    """Script entry point.

    With no argument (the Spyder F5 case, ``USE_COMMAND_LINE_ARGS = False``) the
    ``DEFAULT_*`` block at the top of the file is used. Command-line arguments
    are parsed only when ``USE_COMMAND_LINE_ARGS`` is True or when
    ``--use-command-line-args`` is passed - which is what the .bat launcher does.
    """
    if settings is None:
        argv = sys.argv[1:]
        wants_csv = USE_COMMAND_LINE_ARGS or "--use-command-line-args" in argv
        if wants_csv:
            argv = [item for item in argv if item != "--use-command-line-args"]
            settings = build_argument_parser().parse_args(argv)
        else:
            settings = default_settings()
            if argv:
                print(
                    "Note: ignoring command-line arguments "
                    f"{argv} because USE_COMMAND_LINE_ARGS = False "
                    "(use segment_organoids(...) or --use-command-line-args)."
                )
    return run(settings)



if __name__ == "__main__":
    main()

