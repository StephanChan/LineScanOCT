# -*- coding: utf-8 -*-
"""One depth plane of a TimedPlateScan dynamic run, stacked over the time points.

Per sample of ONE acquisition root::

    <root>/sampleID-N/Time-1/stitched-<CHANNEL>-Y1159-X1599-Z6.tif   (multi tile)
    <root>/sampleID-N/Time-2/stitched-<CHANNEL>-Y1159-X1599-Z6.tif
    ...
  -> <root>/sampleID-N/<CHANNEL>TimeStack-Z3-Yd2-Xd2-Zd1-Y579-X799-T108.tif
       (T, Y/dy, X/dx) BigTIFF, one page = one time point = one en-face plane

How the volume is read (validated against a real TimedPlateScan folder):

* Multi-tile time points -> the stitched mosaic ``stitched-<CHANNEL>-Y..-X..-Z...tif``.
* Single-tile time points -> that one tile ``tile-1-<CHANNEL>-Y..-X..-Z...tif``
  (the stitcher does not write a stitched file for a single-FOV acquisition).
* The saved volumes are ``(Y, X, Z)`` with the TIFF *pages* being Y rows of
  ``(X, Z)``, so one depth plane costs ``page[:, z0:z1]`` per page and nothing
  else is loaded: the RAM peak is a single ``(Y, X)`` plane, never a volume.
* ``DOWNSAMPLE_X/Y/Z`` (2 = 2x) is block averaging (``"sample"`` switches to
  decimation); the depth selection is resolved on the ORIGINAL depth axis, so
  ``DEPTH_PIXEL = 3`` with ``DOWNSAMPLE_Z = 2`` keeps the mean of depth rows 2..3
  and with ``DOWNSAMPLE_Z = 1`` keeps depth row 3 alone.  Block averaging drops an
  odd last row/column (1159/1599 -> 579/799, matching ``-Y579-X799-``).
* A sidecar JSON stores every parameter plus the T-page -> Time-N -> source file
  mapping, so the time axis stays interpretable when time points were skipped.

Time label (``STAMP_TIME``, for the movie): every frame can get "``N 小时 n 分钟``"
in the corner, the elapsed time since the first time point. The timestamps come
from ``tile_positions.json`` -> ``created_at`` of each ``Time-*`` folder (verified
to advance per time point; the file mtime is the fallback), which matters because
the two acquisitions have very different cadences. The label is sized once per
sample so its size never jumps between frames, and it is written as an extra
8-bit display stack ``<stack>-timeStamp.tif`` (``STAMP_OUTPUT = "burn"`` puts it
into the stack itself instead). Pillow is only needed for this label; without it
the run simply writes the plain stacks.

Scale bar (``SCALE_BAR``): the same display frames also carry a physical scale bar
in one corner - a round 1/2/5 x 10^n length picked from the lateral pixel pitch of
the stack (``x_step_um`` of ``tile_positions.json``, times ``DOWNSAMPLE_X``) and
sized to span about ``SCALE_BAR_WIDTH_FRACTION`` (1/10) of the frame width.  The
label font adapts to the frame: as large as ``SCALE_BAR_FONT_PX`` while its ink
covers at most ``SCALE_BAR_TEXT_AREA`` of the frame area, shrunk towards
``SCALE_BAR_FONT_MIN_PX`` when it would not.  ``SCALE_BAR_AREA``,
``SCALE_BAR_MAX_WIDTH`` and ``SCALE_BAR_MIN_PX`` cap the bar block itself.
Because it is drawn with the time label, the bar lands in
``<stack>-timeStamp.tif`` (or in the burned stack) and therefore in the movie; a
stack whose scan metadata holds no lateral step is simply drawn without a bar.


Movie (``WRITE_MOVIE``): the labelled 8-bit frames are also encoded into a movie
next to the stack, ``<stack>-timeStamp.mp4`` by default (OpenCV ``mp4v``; no
system ffmpeg needed, the OpenCV wheel ships its own).  ``MOVIE_FORMAT = "avi"``
switches to the ``MJPG`` encoder instead.  The encoder needs even frame sizes, so
a stack with an odd width/height is padded with a duplicated last row/column
instead of losing that line.  ``MOVIE_FPS`` (the first line of the CONFIG block)
is the only timing knob - the frames are already labelled, so the file plays the
real acquisition order.  The pages are streamed one at a time (RAM = one frame)
and the movie needs OpenCV, so without it the run simply writes the TIFF stacks.

Spyder: set ``ROOT_DIR`` in the CONFIG block to ONE acquisition folder and run
the file (F5); run it again with the other acquisition folder. Nothing in the
source folders is modified: only the new stack(s) + JSON are written next to the
``Time-*`` folders, plus the movie(s) and a run report in the root.
"""

import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import tifffile as TIFF


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# --- MP4 frame rate (playback speed of the movie) --------------------------
MOVIE_FPS = 10.0             # frames per second written into the MP4/AVI, e.g. 10
                             # (= 10 fps playback: the 108 time points of the Dyn
                             # run play in ~10.8 s).  The frames are streamed in
                             # the real acquisition order, so this is the only
                             # timing knob of the movie (see WRITE_MOVIE below).

ROOT_DIR = r"E:\IOCTData\Tcell093026\96wellplate\dynamic\1FOV"   # ONE acquisition root
RECURSIVE = False            # True: also search for sampleID-* below ROOT_DIR
SAMPLE_IDS = None            # None = every sampleID-*; e.g. [1, 3]

CHANNEL = "Dyn"              # Dyn / Mean / DynH / DynS / DynV
DEPTH_PIXEL = 2              # depth row to keep (0-based, on the original Z axis)
DEPTH_RANGE = None           # e.g. (2, 4) -> mean of depth rows 2..3 instead

DOWNSAMPLE_X = 1             # 2 = 2x fewer columns
DOWNSAMPLE_Y = 1             # 2 = 2x fewer rows
DOWNSAMPLE_Z = 1             # depth rows averaged around DEPTH_PIXEL
DOWNSAMPLE_MODE = "mean"     # "mean" (block average) or "sample" (decimate)

TIME_START = None            # None = from the first Time-*; e.g. 5 = start at Time-5
TIME_STOP = None             # None = through the last Time-* (inclusive)
TIME_STRIDE = 1              # e.g. 2 = every second time point

OUTPUT_DTYPE = "float32"     # "float32" / "float16" / "same" (as the source)
OUTPUT_NAME = None           # None -> <CHANNEL>TimeStack-Z..-Yd..-Xd..-Zd..-Y..-X..-T...tif
OVERWRITE = True             # False: keep an existing stack
WRITE_ROOT_REPORT = True     # write <CHANNEL>TimeStack-report.json into ROOT_DIR
DRY_RUN = False              # True: only list what would be read/written

# --- time label in the corner of every frame (movie overlay) ----------------
STAMP_TIME = True            # False: plain stacks only
STAMP_OUTPUT = "sidecar"     # "sidecar": keep the float stack, add <stack>-timeStamp.tif
                             # "burn": the stack itself becomes the 8-bit stamped stack
STAMP_TIME_MODE = "elapsed"  # "elapsed": "2 小时 15 分钟" since the first time point
                             # "absolute": "2026-10-01 15:11" wall clock of that frame
STAMP_TIME_ORIGIN = None     # None = the first usable time point; or "2026-10-01 15:11:43"
STAMP_AREA = 0.05            # the label covers ~5 % of the frame area
STAMP_POSITION = "top-left"  # top-left / top-right / bottom-left / bottom-right
STAMP_MARGIN = 12            # px between the label and the frame edge
STAMP_COLOR = "white"        # PIL colour name or (r, g, b)
STAMP_OUTLINE = "black"      # outline behind the glyphs (None = no outline)
STAMP_SCALE = "frame"        # "frame": display window per frame; "stack": first frame
STAMP_PERCENTILES = (2.0, 99.8)   # display window as percentiles of the plane
STAMP_GAMMA = 1.0            # 1.0 = linear; >1 brighter mid-tones
STAMP_FIXED_RANGE = None     # e.g. (0.0, 1.0) to override the percentiles
STAMP_FONT = None            # a .ttf/.ttc path, or None to search the list below
STAMP_FONT_CANDIDATES = (    # searched in order (CJK font needed for 小时/分钟)
    r"C:\Windows\Fonts\msyh.ttc",     # Microsoft YaHei
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",   # SimHei
    r"C:\Windows\Fonts\Deng.ttf",     # DengXian
    r"C:\Windows\Fonts\simsun.ttc",   # SimSun
    r"C:\Windows\Fonts\arial.ttf",    # last resort: no CJK glyphs
)

# --- scale bar burned into the stamped frames (stack + movie) ---------------
# The frames that feed the -timeStamp stack and the movie also get a physical
# scale bar in one corner, so the movie is self-describing without a caption.
# The pitch of the stored plane comes from the scan metadata (x_step_um of
# tile_positions.json, times DOWNSAMPLE_X), i.e. from the same file as the time
# labels, and the bar is a round 1/2/5 x 10^n length chosen so that it spans
# about SCALE_BAR_WIDTH_FRACTION of the frame width (1/10), while the whole block
# (bar length x bar+label strip) stays inside SCALE_BAR_AREA of the frame area.
# The label is capped on its own as well: its ink may cover at most
# SCALE_BAR_TEXT_AREA of the frame area, so the font is shrunk from
# SCALE_BAR_FONT_PX (the size used when it fits) down to SCALE_BAR_FONT_MIN_PX
# until it does.  SCALE_BAR_MAX_WIDTH and the frame margin cap the bar too, and a
# bar shorter than SCALE_BAR_MIN_PX drops it.  Needs Pillow.
SCALE_BAR = True             # False: time label only
SCALE_BAR_UM_PER_PX = None   # None -> x_step_um * DOWNSAMPLE_X from the metadata;
                             # a number overrides it (um per STORED column)
SCALE_BAR_WIDTH_FRACTION = 0.10   # wanted: the bar spans ~1/10 of the frame width
SCALE_BAR_AREA = 0.02        # hard cap: the bar block <= 2 % of the frame area
SCALE_BAR_TEXT_AREA = 0.005  # hard cap: the label ink <= 0.5 % of the frame area
SCALE_BAR_MAX_WIDTH = 0.45   # hard cap: the bar <= 45 % of the frame width
SCALE_BAR_MIN_PX = 32        # no bar when the round length would be shorter
SCALE_BAR_THICKNESS = 10     # px of the bar line
SCALE_BAR_FONT_PX = 84       # px of the label - the size used when it fits
SCALE_BAR_FONT_MIN_PX = 12   # ... shrunk to fit SCALE_BAR_TEXT_AREA down to this
SCALE_BAR_GAP = 4            # px between the bar and its label
SCALE_BAR_POSITION = "bottom-right"   # bottom-right / bottom-left / top-right
SCALE_BAR_MARGIN = 16        # px between the bar block and the frame edge
SCALE_BAR_COLOR = "white"    # PIL colour (or int grey) of bar and glyphs
SCALE_BAR_OUTLINE = "black"  # halo behind them (None = no halo)


# --- movie next to the stack (needs the labelled frames, so STAMP_TIME) -----
# MOVIE_FPS (playback speed, in frames per second) is the FIRST setting of this
# CONFIG block, at the very top of the file.
WRITE_MOVIE = True           # True: encode the labelled frames into a movie file
MOVIE_FORMAT = "mp4"         # "mp4" (OpenCV 'mp4v') or "avi" (OpenCV 'MJPG')
MOVIE_FOURCC = None          # None -> 'mp4v' for mp4 / 'MJPG' for avi
MOVIE_NAME = None            # None -> <stack stem>-timeStamp.mp4 / .avi

SOURCE_TIME_META = "tile_positions.json"   # per-time-point metadata (created_at)

try:                                       # only needed when STAMP_TIME is on
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:                          # noqa: BLE001 - optional dependency
    Image = ImageDraw = ImageFont = None
    PIL_AVAILABLE = False

try:                                       # only needed when WRITE_MOVIE is on
    import cv2
    CV2_AVAILABLE = True
except Exception:                          # noqa: BLE001 - optional dependency
    cv2 = None
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
SAMPLE_DIR_RE = re.compile(r"^sampleID-(?P<sample>\d+)$")
TIME_DIR_RE = re.compile(r"^Time-(?P<time>\d+)$")


def _tile_re(channel):
    """``tile-<n>-<CHANNEL>-Y<y>-X<x>-Z<z>.tif`` of one channel."""
    return re.compile(
        r"^tile-(?P<tile>\d+)-" + re.escape(channel)
        + r"-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
    )


def _stitched_re(channel):
    """``stitched-<CHANNEL>-Y<y>-X<x>-Z<z>.tif`` of one channel."""
    return re.compile(
        r"^stitched-" + re.escape(channel)
        + r"-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tif$"
    )


def parse_shape(filename):
    """``(Y, X, Z)`` parsed from a ``...-Y<y>-X<x>-Z<z>.tif`` name, or None."""
    stem = os.path.basename(filename)
    if stem.lower().endswith(".tif"):
        stem = stem[:-4]
    try:
        tail = stem.split("-Y")[-1]
        y_text, rest = tail.split("-X", 1)
        x_text, z_text = rest.split("-Z", 1)
        return int(y_text), int(x_text), int(z_text)
    except (IndexError, ValueError):
        return None


def find_sample_dirs(root, recursive=RECURSIVE):
    """``[(sample_id, path)]`` of the ``sampleID-*`` folders of one root."""
    found = []
    if not os.path.isdir(root):
        return found
    if not recursive:
        for name in os.listdir(root):
            match = SAMPLE_DIR_RE.match(name)
            path = os.path.join(root, name)
            if match and os.path.isdir(path):
                found.append((int(match.group("sample")), path))
    else:
        for current, dirs, _files in os.walk(root):
            matches = [name for name in dirs if SAMPLE_DIR_RE.match(name)]
            for name in matches:
                found.append(
                    (int(SAMPLE_DIR_RE.match(name).group("sample")),
                     os.path.join(current, name))
                )
            # never descend into a sample folder (its Time-* folders are leaves)
            dirs[:] = [name for name in dirs if name not in matches]
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def find_time_dirs(sample_dir, time_start=None, time_stop=None, time_stride=1):
    """``[(time_id, path)]`` of the ``Time-*`` folders, sorted numerically."""
    time_dirs = []
    try:
        names = os.listdir(sample_dir)
    except OSError:
        return time_dirs
    for name in names:
        match = TIME_DIR_RE.match(name)
        path = os.path.join(sample_dir, name)
        if not match or not os.path.isdir(path):
            continue
        time_id = int(match.group("time"))
        if time_start is not None and time_id < int(time_start):
            continue
        if time_stop is not None and time_id > int(time_stop):
            continue
        time_dirs.append((time_id, path))
    time_dirs.sort(key=lambda item: item[0])
    stride = max(1, int(time_stride))
    if stride > 1:
        time_dirs = time_dirs[::stride]
    return time_dirs


def select_source(folder, channel=CHANNEL):
    """Volume to read in one ``Time-*`` folder.

    Returns ``{"kind", "name", "shape", "note"}`` on success (``kind`` is
    ``"stitched"`` for a multi-tile time point and ``"tile"`` for a single-tile
    one), or ``{"reason"}`` when the time point cannot be used.
    """
    stitched = []
    tiles = []
    try:
        names = sorted(os.listdir(folder))
    except OSError as error:
        return {"reason": "unreadable folder ({0})".format(error)}
    stitch_match = _stitched_re(channel)
    tile_match = _tile_re(channel)
    for name in names:
        if stitch_match.match(name):
            stitched.append(name)
        elif tile_match.match(name):
            tiles.append(name)
    if stitched:
        name = stitched[0]
        note = "" if len(stitched) == 1 else "first of {0}".format(len(stitched))
        return {"kind": "stitched", "name": name, "shape": parse_shape(name),
                "note": note}
    if len(tiles) == 1:
        return {"kind": "tile", "name": tiles[0], "shape": parse_shape(tiles[0]),
                "note": ""}
    if len(tiles) > 1:
        return {"reason": "{0} tiles but no stitched-{1} mosaic".format(
            len(tiles), channel)}
    return {"reason": "no {0} volume".format(channel)}


def resolve_depth_window(depth_pixel, depth_range, downsample_z):
    """``(z0, z1)`` depth rows to read (``z1`` exclusive) on the original axis."""
    if depth_range is not None:
        z0 = int(depth_range[0])
        z1 = int(depth_range[1])
        if z1 <= z0:
            raise ValueError("DEPTH_RANGE must be (first, last+1), got {0}".format(
                tuple(depth_range)))
        return z0, z1
    depth_pixel = max(0, int(depth_pixel))
    factor = max(1, int(downsample_z))
    z0 = (depth_pixel // factor) * factor
    return z0, z0 + factor


def downsample_size(length, factor, mode=DOWNSAMPLE_MODE):
    """Output length of one axis for :func:`downsample_plane`."""
    factor = max(1, int(factor))
    if factor == 1:
        return int(length)
    if mode == "sample":
        return -(-int(length) // factor)
    trimmed = int(length) - int(length) % factor
    if trimmed == 0:
        return -(-int(length) // factor)
    return trimmed // factor


def downsample_plane(plane, factor_y, factor_x, mode=DOWNSAMPLE_MODE):
    """Block-average (or decimate) a ``(Y, X)`` plane in Y and X."""
    factor_y = max(1, int(factor_y))
    factor_x = max(1, int(factor_x))
    if factor_y == 1 and factor_x == 1:
        return plane
    if mode == "sample":
        return plane[::factor_y, ::factor_x]
    y_len, x_len = plane.shape[0], plane.shape[1]
    y_trim = y_len - y_len % factor_y
    x_trim = x_len - x_len % factor_x
    if y_trim == 0 or x_trim == 0:
        return plane[::factor_y, ::factor_x]
    view = plane[:y_trim, :x_trim]
    reshaped = view.reshape(y_trim // factor_y, factor_y,
                            x_trim // factor_x, factor_x)
    return reshaped.mean(axis=(1, 3))


def read_depth_plane(path, z0, z1, mode=DOWNSAMPLE_MODE):
    """Read one depth plane of a volume TIFF as ``(Y, X)`` float32.

    The pages of the saved volumes are Y rows holding ``(X, Z)``, so only the
    requested depth rows of every page are touched; the whole volume is never
    materialised.  A volume stored as a single ``(Y, X, Z)`` page (what tifffile
    does for very small arrays) is handled too.
    """
    rows = []
    with TIFF.TiffFile(path) as tif:
        page_count = len(tif.pages)
        for page in tif.pages:
            block = np.asarray(page.asarray(), dtype=np.float32)
            if block.ndim == 3:
                if page_count == 1:
                    return _plane_from_volume(block, z0, z1, mode, path)
                continue  # not a Y-row page layout -> not a depth axis
            if block.ndim == 1:
                block = block[:, np.newaxis]
            if block.ndim != 2 or block.shape[1] == 0:
                continue
            if z0 >= block.shape[1]:
                raise ValueError("depth {0} is outside {1} (Z={2})".format(
                    z0, os.path.basename(path), block.shape[1]))
            slab = block[:, z0:min(z1, block.shape[1])]
            if mode == "sample":
                rows.append(slab[:, slab.shape[1] // 2])
            else:
                rows.append(slab.mean(axis=1))
    if not rows:
        raise ValueError("no page in {0}".format(path))
    return np.stack(rows)


def _plane_from_volume(volume, z0, z1, mode, path):
    """One ``(Y, X)`` depth plane out of a single-page ``(Y, X, Z)`` array."""
    if z0 >= volume.shape[2]:
        raise ValueError("depth {0} is outside {1} (Z={2})".format(
            z0, os.path.basename(path), volume.shape[2]))
    slab = volume[:, :, z0:min(z1, volume.shape[2])]
    if mode == "sample":
        return slab[:, :, slab.shape[2] // 2]
    return slab.mean(axis=2)


def source_dtype(path):
    """dtype of the pages of a TIFF (used by ``OUTPUT_DTYPE = "same"``)."""
    with TIFF.TiffFile(path) as tif:
        return np.dtype(tif.pages[0].dtype)


def resolve_output_dtype(name, fallback_dtype):
    """``numpy.dtype`` of the output stack."""
    key = str(name or "float32").strip().lower()
    if key in ("same", "source", "keep"):
        return np.dtype(fallback_dtype)
    return np.dtype(key)


def output_filename(t_count, y_out, x_out, z0, z1, channel=CHANNEL):
    """Name of the stack; the trailing ``-Y..-X..-T..`` is the OUTPUT shape."""
    depth = "Z{0}".format(z0) if z1 - z0 == 1 else "Z{0}-{1}".format(z0, z1 - 1)
    return (
        "{0}TimeStack-{1}-Yd{2}-Xd{3}-Zd{4}-Y{5}-X{6}-T{7}.tif".format(
            channel, depth, int(DOWNSAMPLE_Y), int(DOWNSAMPLE_X),
            max(1, int(DOWNSAMPLE_Z)), int(y_out), int(x_out), int(t_count),
        )
    )


# ---------------------------------------------------------------------------
# time metadata + frame label
# ---------------------------------------------------------------------------
_MANIFEST_CACHE = {}


def parse_datetime(text):
    """``datetime`` from an ISO-ish string (``2026-10-01T15:11:43``), or None."""
    cleaned = str(text).strip().replace("T", " ").replace("Z", "")
    cleaned = cleaned.split("+")[0].split(".")[0].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def read_time_manifest(folder, meta_name=SOURCE_TIME_META):
    """Parsed ``tile_positions.json`` of one ``Time-*`` folder (cached), or None."""
    path = os.path.join(folder, meta_name)
    if path not in _MANIFEST_CACHE:
        data = None
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                data = None
        _MANIFEST_CACHE[path] = data
    return _MANIFEST_CACHE[path]


def time_point_stamp(folder, volume_path=None, meta_name=SOURCE_TIME_META):
    """Timestamp of one time point.

    ``tile_positions.json`` -> ``created_at`` is written when that time point is
    acquired (verified to advance per time point), the volume file mtime is the
    fallback, then the folder mtime.  Returns ``datetime`` or ``None``.
    """
    manifest = read_time_manifest(folder, meta_name)
    if isinstance(manifest, dict):
        parsed = parse_datetime(manifest.get("created_at") or "")
        if parsed is not None:
            return parsed
    for candidate in (volume_path, os.path.join(folder, meta_name), folder):
        if candidate and os.path.isfile(candidate):
            try:
                return datetime.fromtimestamp(os.path.getmtime(candidate))
            except OSError:
                continue
    return None


def format_elapsed(seconds):
    """``"N 小时 n 分钟"`` for an elapsed duration (``N`` hours, ``n`` minutes)."""
    minutes = int(round(max(0.0, float(seconds)) / 60.0))
    return "{0} 小时 {1} 分钟".format(minutes // 60, minutes % 60)


def _font_path():
    """Path of the label font: ``STAMP_FONT`` or the first existing candidate."""
    if STAMP_FONT:
        if os.path.isfile(STAMP_FONT):
            return STAMP_FONT
        print("  STAMP_FONT {0} not found -> searching candidates".format(STAMP_FONT))
    for candidate in STAMP_FONT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


class FrameLabeler(object):
    """Same-size time label drawn into one corner of every frame.

    The font size is fitted once (from the longest label of the sample) so it
    never jumps between frames, and the label covers roughly ``area`` of the
    frame.  :meth:`render` returns the 8-bit display frame including the label.
    """

    def __init__(self, longest_text, shape, position=None, area=None, margin=None,
                 color=None, outline=None, gamma=None):
        self.height, self.width = int(shape[0]), int(shape[1])
        self.position = str(STAMP_POSITION if position is None else position).lower()
        self.margin = max(0, int(STAMP_MARGIN if margin is None else margin))
        self.color = STAMP_COLOR if color is None else color
        self.outline = STAMP_OUTLINE if outline is None else outline
        self.gamma = float(STAMP_GAMMA if gamma is None else gamma)
        self.area = float(STAMP_AREA if area is None else area)
        self.font_path = _font_path()
        self._fonts = {}
        self._probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        self.text = longest_text
        self.size = 0
        self.stroke = 0
        self.width_px = 0
        self.height_px = 0
        self.area_ratio = 0.0
        self.text_at = (self.margin, self.margin)
        self._place(longest_text, self.area)

    # -- font fitting -------------------------------------------------------
    def _font_for(self, size):
        size = max(4, int(size))
        font = self._fonts.get(size)
        if font is None:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                font = ImageFont.load_default(size=size)
            self._fonts[size] = font
        return font

    def _stroke_for(self, size):
        if self.outline is None:
            return 0
        return max(0, int(round(int(size) * 0.08)))

    def _measure(self, text, font, stroke):
        box = self._probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
        return (int(box[2] - box[0]), int(box[3] - box[1]), (int(box[0]), int(box[1])))

    def _place(self, text, area):
        """Largest font whose label fills ``area`` of the frame while fitting in it."""
        target = max(16.0, float(area) * self.width * self.height)
        room_w = max(1, self.width - 2 * self.margin)
        room_h = max(1, self.height - 2 * self.margin)
        size = max(4, int(round(self.height * 0.10)))
        under = None      # biggest label at or under the target area
        over = None       # smallest label above the target area
        for _ in range(12):
            stroke = self._stroke_for(size)
            font = self._font_for(size)
            width, height, origin = self._measure(text, font, stroke)
            found = (font, stroke, width, height, origin)
            fits = width <= room_w and height <= room_h
            if fits:
                if width * height <= target:
                    if under is None or width * height > under[2] * under[3]:
                        under = found
                elif over is None or width * height < over[2] * over[3]:
                    over = found
            wanted = 1.0
            if width > 0 and height > 0:
                wanted = (target / float(width * height)) ** 0.5   # aim at the target area
            if not fits:
                wanted = min(wanted, room_w / float(max(1, width)),
                             room_h / float(max(1, height)))
            next_size = max(4, int(round(size * wanted)))
            if next_size == size or next_size <= 4 or next_size > 4096:
                break
            size = next_size
        font, stroke, width, height, origin = under or over or found
        self.font, self.stroke = font, stroke
        self.size = int(getattr(font, "size", size))
        self.width_px, self.height_px = width, height
        self.area_ratio = (width * height) / float(self.width * self.height)
        offset_x = self.margin - origin[0]
        offset_y = self.margin - origin[1]
        if "right" in self.position:
            offset_x = self.width - self.margin - width - origin[0]
        if "bottom" in self.position:
            offset_y = self.height - self.margin - height - origin[1]
        # Pillow lays the text out from the em box, so keep the ink inside the frame
        self.text_at = (int(min(max(offset_x, 0), max(0, self.width - width))),
                        int(min(max(offset_y, 0), max(0, self.height - height))))

    def describe(self):
        """One line about the fitted label, for the console/report."""
        font_name = os.path.basename(self.font_path) if self.font_path else "PIL default"
        return ("{0} '{1}', font {2} size {3} stroke {4}, {5:.1f} % of the frame, "
                "drawn at {6}".format(self.position, self.text, font_name, self.size,
                                      self.stroke, 100.0 * self.area_ratio,
                                      self.text_at))

    # -- rendering ----------------------------------------------------------
    def window(self, plane, percentiles=None, fixed=None):
        """``(low, high)`` display window of one plane."""
        percentiles = STAMP_PERCENTILES if percentiles is None else percentiles
        fixed = STAMP_FIXED_RANGE if fixed is None else fixed
        if fixed is not None:
            return float(fixed[0]), float(fixed[1])
        data = np.asarray(plane, dtype=np.float32)
        low, high = (float(value) for value in np.percentile(data, percentiles))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(data)), float(np.max(data))
            if high <= low:
                high = low + 1.0
        return low, high

    def render(self, plane, text, window=None):
        """8-bit display frame of ``plane`` with ``text`` drawn in the corner."""
        low, high = self.window(plane) if window is None else window
        span = max(1e-12, float(high) - float(low))
        scaled = np.clip((np.asarray(plane, dtype=np.float32) - low) / span, 0.0, 1.0)
        if self.gamma != 1.0:
            scaled = np.power(scaled, self.gamma)
        frame = Image.fromarray((scaled * 255.0 + 0.5).astype(np.uint8), mode="L")
        ImageDraw.Draw(frame).text(self.text_at, text, font=self.font, fill=self.color,
                                   stroke_width=self.stroke, stroke_fill=self.outline)
        return np.asarray(frame)


# ---------------------------------------------------------------------------
# scale bar
# ---------------------------------------------------------------------------
def _axis_step_um(meta, axis):
    """Sampling step (um) of one lateral axis in a ``tile_positions.json``."""
    for key in (axis + "_step_um", axis + "_pitch_um"):
        try:
            step = float(meta[key])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if 0.0 < step < 1e6:
            return step
    try:                                # older manifests: length over pixels
        length = float(meta[axis + "_length_mm"])
        pixels = int(meta[axis + "_pixels"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return length * 1000.0 / pixels if length > 0.0 and pixels > 0 else None


def sample_um_per_px(sources, meta_name=SOURCE_TIME_META):
    """``(um_x, um_y, time_point)`` lateral pixel pitch of the sample, or Nones.

    A TimedPlateScan writes the sampling step of its tiles into the per-time-point
    ``tile_positions.json`` (``x_step_um`` / ``y_step_um``, e.g. 0.96 and 1.0 um),
    and a stitched plane is a plain pixel mosaic of those tiles, so the stored
    plane keeps exactly that pitch.  ``sources`` are the time points chosen by
    :func:`plan_planes`; the first metadata file that can be read wins and the
    returned name says which time point it came from (console/report only).
    """
    for entry in sources:
        path = os.path.join(entry["folder"], meta_name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        x_step = _axis_step_um(meta, "x")
        y_step = _axis_step_um(meta, "y")
        if x_step or y_step:
            return (x_step or y_step, y_step or x_step,
                    os.path.basename(entry["folder"]))
    return (None, None, None)


def _nearest_length_um(target_um, max_um=None):
    """Round 1/2/5 x 10^n length (um) closest to ``target_um``, or None.

    The ladder is the usual 1/2/5 x 10^n series (0.1 um ... 50 m) and the winner
    is the candidate with the smallest length *ratio* to ``target_um``, so the
    bar gets as close to the wanted pixel span as the round series allows.
    ``max_um`` (a hard cap in um) removes every candidate that would not fit.
    """
    try:
        target = float(target_um)
        limit = float(max_um) if max_um else float("inf")
    except (TypeError, ValueError):
        return None
    if not target > 0.0:
        return None
    best, best_error = None, None
    for exponent in range(-1, 8):
        base = 10.0 ** exponent
        for factor in (1.0, 2.0, 5.0):
            candidate = factor * base
            if candidate > limit * (1.0 + 1e-9):
                continue
            error = max(candidate / target, target / candidate)
            if best_error is None or error < best_error:
                best, best_error = candidate, error
    return best


def scale_bar_label(value_um, total_um=None):
    """Label of the scale bar, e.g. ``"200 um"`` / ``"0.5 mm"``.

    ``total_um`` is the physical width the frame spans and only picks the unit
    (mm from 1 mm up), i.e. the same wording as the axis rulers of the viewer.
    """
    total = float(total_um) if total_um else float(value_um)
    if not total > 0.0:
        total = float(value_um)
    value = float(value_um)
    if total >= 999.5:
        text = "{0:.3f}".format(value / 1000.0).rstrip("0").rstrip(".")
        return "{0} mm".format(text if text not in ("", "-0") else "0")
    if abs(value - round(value)) < 0.01:
        return "{0} um".format(int(round(value)))
    return "{0:.1f} um".format(value)



class ScaleBar(object):
    """Adaptive scale bar burned into one corner of every stamped frame.

    The bar is a round 1/2/5 x 10^n physical length chosen to span about
    ``SCALE_BAR_WIDTH_FRACTION`` (1/10) of the frame width, while its footprint -
    the bar length times the height of the bar+label strip - stays inside
    ``SCALE_BAR_AREA`` of the frame area and the bar inside ``SCALE_BAR_MAX_WIDTH``
    of the width; a round length below ``SCALE_BAR_MIN_PX`` drops the bar.  The
    label font is ``SCALE_BAR_FONT_PX`` as long as its ink covers at most
    ``SCALE_BAR_TEXT_AREA`` of the frame area and is shrunk down to
    ``SCALE_BAR_FONT_MIN_PX`` when it would not.
    Like the time label the geometry is fixed once per sample, so it never jumps
    between frames.  ``value_um`` is None when no readable bar fits the frame
    (:meth:`draw` then returns the frame untouched).  ``um_per_px`` is the pitch
    of the stored plane (um per column, ``DOWNSAMPLE_X`` already included) and
    ``unit_total_um`` the physical width of the frame, used only to pick the
    um/mm label unit.
    """

    def __init__(self, shape, um_per_px=None, unit_total_um=None, position=None,
                 area=None, margin=None, thickness=None, font_px=None, color=None,
                 outline=None):
        self.height, self.width = int(shape[0]), int(shape[1])
        self.position = str(SCALE_BAR_POSITION if position is None else position).lower()
        self.area = float(SCALE_BAR_AREA if area is None else area)
        self.margin = max(0, int(SCALE_BAR_MARGIN if margin is None else margin))
        self.thickness = max(1, int(SCALE_BAR_THICKNESS if thickness is None
                                    else thickness))
        self.font_px = max(6, int(SCALE_BAR_FONT_PX if font_px is None else font_px))
        self.color = SCALE_BAR_COLOR if color is None else color
        self.outline = SCALE_BAR_OUTLINE if outline is None else outline
        self.unit_total_um = unit_total_um
        try:
            self.um_per_px = float(um_per_px)
        except (TypeError, ValueError):
            self.um_per_px = None
        self.value_um = None            # None -> no readable bar in this frame
        self.length_px = 0.0
        self.label = ""
        self.bar_rect = (0, 0, 0, 0)    # inclusive PIL rectangle of the bar
        self.text_at = (0, 0)
        self.font = None
        self.stroke = 0
        self.text_px2 = 0.0             # drawn area (px^2) of the current label
        self._probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        if self.um_per_px is not None and self.um_per_px > 0.0:
            self._place()

    # -- geometry -----------------------------------------------------------
    def block_height(self):
        """Height (px) of the whole bar block: bar + gap + label."""
        return self.thickness + SCALE_BAR_GAP + self.font_px + 4



    def _place(self):
        """Pick the round length, the label font, then the corner geometry.

        The wish is a bar spanning ``SCALE_BAR_WIDTH_FRACTION`` of the frame
        width labelled at ``SCALE_BAR_FONT_PX`` px.  Both are then fitted to the
        ``SCALE_BAR_AREA`` budget: the length only has to leave room for a
        minimum-size label, and the font then takes whatever the budget still
        pays for (never above the wanted size), so a small frame shrinks the
        label instead of cutting the bar short.  Fixed once per sample.
        """
        prefix = self.thickness + SCALE_BAR_GAP + 4      # block without the label
        budget = self.area * self.width * self.height    # e.g. 2 % of the area
        min_block = prefix + float(SCALE_BAR_FONT_MIN_PX)
        fits_px = min(
            budget / min_block,
            float(SCALE_BAR_MAX_WIDTH) * self.width,
            self.width - 2.0 * self.margin,
        )
        target_px = min(float(SCALE_BAR_WIDTH_FRACTION) * self.width, fits_px)
        value = None
        if target_px > 0.0:
            value = _nearest_length_um(target_px * self.um_per_px,
                                       fits_px * self.um_per_px)
            if value is not None and (value / self.um_per_px
                                      < float(SCALE_BAR_MIN_PX)):
                value = None
        if value is None:
            return
        self.value_um = float(value)
        self.length_px = self.value_um / self.um_per_px
        self.label = scale_bar_label(self.value_um, self.unit_total_um)
        # Font: the wanted size as long as the label ink fits its own budget.
        self._fit_font()
        # The bar is aligned to its corner and its ink spans exactly
        # round(length_px) whole pixels, so the quoted length is exact to 1 px.
        span = max(1, int(round(self.length_px)))
        if "right" in self.position:
            right = self.width - self.margin
            left = right - span + 1
        else:
            left = self.margin
            right = left + span - 1
        if "bottom" in self.position:
            bottom = self.height - self.margin
            top = bottom - self.thickness + 1
        else:
            top = self.margin
            bottom = top + self.thickness - 1
        self.bar_rect = (left, top, right, bottom)
        # Label centred on the bar and kept inside the frame.
        box = self._probe.textbbox((0, 0), self.label, font=self.font,
                                   stroke_width=self.stroke)
        text_w, text_h = box[2] - box[0], box[3] - box[1]
        x = (left + right) / 2.0 - text_w / 2.0 - box[0]
        if "top" in self.position:
            y = bottom + SCALE_BAR_GAP - box[1]
        else:
            y = top - SCALE_BAR_GAP - text_h - box[1]
        self.text_at = (int(max(0, min(x, self.width - text_w))),
                        int(max(0, min(y, self.height - text_h))))

    def _stroke_width(self):
        """Halo width (px) that goes with the current font size."""
        return 0 if self.outline is None else max(
            1, int(round(self.font_px * 0.08)))

    def _fit_font(self):
        """Size the label font so its ink fits ``SCALE_BAR_TEXT_AREA``.

        Starts at the wanted ``SCALE_BAR_FONT_PX`` and shrinks (never below
        ``SCALE_BAR_FONT_MIN_PX``) until the drawn box of the label covers at
        most the allowed share of the frame area; the ink area of the label
        grows with the square of the font size, so one rescaling per step is
        enough.  Also refreshes ``font``, ``stroke`` and ``text_px2``.
        """
        floor = int(SCALE_BAR_FONT_MIN_PX)
        budget = float(SCALE_BAR_TEXT_AREA) * self.width * self.height
        while True:
            self.font = self._font()
            self.stroke = self._stroke_width()
            box = self._probe.textbbox((0, 0), self.label, font=self.font,
                                       stroke_width=self.stroke)
            ink = float((box[2] - box[0]) * (box[3] - box[1]))
            self.text_px2 = ink
            if ink <= budget or self.font_px <= floor:
                return
            smaller = int(self.font_px * (budget / ink) ** 0.5)
            self.font_px = max(floor, min(self.font_px - 1, smaller))

    def _font(self):
        path = _font_path()
        if path:
            return ImageFont.truetype(path, self.font_px)
        try:
            return ImageFont.load_default(size=self.font_px)
        except TypeError:               # Pillow < 9.2: no sized default font
            return ImageFont.load_default()

    def describe(self):
        """One line about the bar, for the console and the report."""
        if self.value_um is None:
            reason = ("no lateral pixel pitch" if self.um_per_px is None else
                      "no room for {0:.0f} px of bar".format(SCALE_BAR_MIN_PX))
            return "no scale bar ({0}x{1} frame: {2})".format(self.width,
                                                              self.height, reason)
        footprint = self.length_px * self.block_height()
        area = float(self.width * self.height)
        return ("{0} '{1}' = {2:.0f} px = {6:.1f} % of the width at {3:.4g} um/px, "
                "{4:.2f} % of the frame area, label {7} px ({8:.2f} % ink), "
                "bar at {5}".format(
                    self.position, self.label, self.length_px, self.um_per_px,
                    100.0 * footprint / area, self.bar_rect,
                    100.0 * self.length_px / float(self.width), self.font_px,
                    100.0 * self.text_px2 / area))

    # -- rendering ----------------------------------------------------------
    def draw(self, frame):
        """``frame`` (8-bit array or PIL image) with the bar burned into it."""
        image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        if self.value_um is None:
            return np.asarray(image)
        if image.mode != "L":
            image = image.convert("L")
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = self.bar_rect
        if self.outline is not None:    # a 1 px halo keeps the bar readable
            draw.rectangle((left - 1, top - 1, right + 1, bottom + 1),
                           fill=self.outline)
        draw.rectangle((left, top, right, bottom), fill=self.color)
        draw.text(self.text_at, self.label, font=self.font, fill=self.color,
                  stroke_width=self.stroke, stroke_fill=self.outline)
        return np.asarray(image)



def plan_planes(sample_dir, channel=CHANNEL):
    """First pass: pick the source file of every usable time point.

    Returns ``(sources, skipped, time_dirs)``; ``sources`` (in ``Time-*`` order)
    becomes the T axis and ``skipped`` lists the dropped time points with their
    reason.  No pixel is read, so a dry run stops here.
    """
    time_dirs = find_time_dirs(sample_dir, TIME_START, TIME_STOP, TIME_STRIDE)
    sources = []
    skipped = []
    for time_id, time_dir in time_dirs:
        source = select_source(time_dir, channel)
        if "reason" in source:
            skipped.append({"time_id": int(time_id), "folder": time_dir,
                            "reason": source["reason"]})
            continue
        if source["shape"] is None:
            skipped.append({"time_id": int(time_id), "folder": time_dir,
                            "reason": "unparsable volume name {0}".format(
                                source["name"])})
            continue
        volume_path = os.path.join(time_dir, source["name"])
        stamp = time_point_stamp(time_dir, volume_path)
        sources.append({
            "time_id": int(time_id),
            "folder": time_dir,
            "path": volume_path,
            "kind": source["kind"],
            "name": source["name"],
            "shape": [int(value) for value in source["shape"]],
            "note": source["note"],
            "stamp": stamp,
            "timestamp": stamp.isoformat(sep=" ", timespec="seconds") if stamp else None,
        })
    return sources, skipped, time_dirs


def time_labels(sources, mode=None, origin=None):
    """Label of every time point plus the origin they are measured from.

    ``mode="elapsed"`` gives ``"N 小时 n 分钟"`` since ``origin`` (the first time
    point when ``origin`` is None), ``mode="absolute"`` gives the wall clock of
    that time point.  Time points without timestamps fall back to ``Time-N``.
    """
    mode = str(STAMP_TIME_MODE if mode is None else mode).lower()
    origin = STAMP_TIME_ORIGIN if origin is None else origin
    if isinstance(origin, str):
        origin = parse_datetime(origin)
    else:
        origin = next((entry["stamp"] for entry in sources if entry.get("stamp")), None) \
            if origin is None else origin
    labels = []
    missing = 0
    for entry in sources:
        stamp = entry.get("stamp")
        if stamp is None:
            missing += 1
            labels.append("Time-{0}".format(entry["time_id"]))
        elif mode == "absolute":
            labels.append(stamp.strftime("%Y-%m-%d %H:%M"))
        else:
            labels.append(format_elapsed((stamp - origin).total_seconds()))
    return labels, origin, missing


MOVIE_FORMATS = {              # MOVIE_FORMAT name -> (fourcc, file extension)
    "mp4": ("mp4v", "mp4"),
    "avi": ("MJPG", "avi"),
}


def movie_codec(format_name=None, fourcc=None):
    """``(fourcc, extension)`` of the configured movie container."""
    name = str(MOVIE_FORMAT if format_name is None else format_name).lower().lstrip(".")
    if name not in MOVIE_FORMATS:
        raise ValueError("MOVIE_FORMAT must be 'mp4' or 'avi', got {0}".format(
            MOVIE_FORMAT if format_name is None else format_name))
    default, extension = MOVIE_FORMATS[name]
    chosen = MOVIE_FOURCC if fourcc is None else fourcc
    return (str(chosen) if chosen else default), extension


def movie_name_for(stack_name, extension, custom=None):
    """Movie file name of a stack: ``<stack stem>-timeStamp.mp4`` by default."""
    name = MOVIE_NAME if custom is None else custom
    if name:
        return name
    return "{0}-timeStamp.{1}".format(os.path.splitext(stack_name)[0], extension)


def even_frame(frame):
    """``frame`` padded to an even width/height (the encoders require that).

    The extra row/column repeats the last one, so no content is lost; a plain
    odd frame would be silently cropped by the video encoder instead.
    """
    height, width = frame.shape[0], frame.shape[1]
    pad_y, pad_x = height % 2, width % 2
    if not pad_y and not pad_x:
        return frame
    padding = [(0, pad_y), (0, pad_x)] + [(0, 0)] * (frame.ndim - 2)
    return np.pad(frame, padding, mode="edge")


def frames_of_page(page):
    """The 2D greyscale planes of one TIFF page.

    A stack written plane by plane (what this script does) has one 2D page per
    plane, but ``tifffile.imwrite(path, stack)`` stores a whole ``(T, Y, X)``
    array as ONE page holding all planes, so a page must be expanded instead of
    being handed to the encoder as a 3D array (which FFmpeg silently drops).
    """
    plane = np.asarray(page.asarray())
    if plane.ndim == 2:
        return [plane]
    if plane.ndim == 3 and plane.shape[-1] == 1:            # (Y, X, 1)
        return [plane[:, :, 0]]
    if plane.ndim == 3 and plane.shape[-1] not in (3, 4):   # (Z, Y, X) volume page
        return [plane[index] for index in range(plane.shape[0])]
    raise ValueError("a {0}-shaped page is not a greyscale frame".format(
        tuple(plane.shape)))


def write_movie(source_path, movie_path, fps=None, fourcc=None):
    """Encode the pages of a TIFF stack into ``movie_path`` (mp4/avi).

    The pages are streamed one at a time, so the RAM cost stays a single frame
    however many time points the stack holds.  A ``<name>.part.<ext>`` file is
    used while writing and renamed into place, exactly like the stacks.  Returns
    ``(frame_count, (width, height))`` of the encoded frames.

    ``fps`` defaults to the ``MOVIE_FPS`` setting at the top of the CONFIG block,
    which is the playback speed of the file.
    """
    fps = float(MOVIE_FPS if fps is None else fps)
    code = movie_codec(fourcc=fourcc)[0]
    if fps <= 0:
        raise ValueError("MOVIE_FPS must be positive, got {0}".format(fps))
    stem, extension = os.path.splitext(movie_path)
    part_path = stem + ".part" + extension           # keep .mp4/.avi for the muxer
    writer = None
    frames = 0
    size = (0, 0)
    try:
        with TIFF.TiffFile(source_path) as stack:
            for page in stack.pages:
                for plane in frames_of_page(page):
                    frame = even_frame(plane)
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise RuntimeError(
                            "cannot encode a frame of shape {0} from {1}: the "
                            "encoder needs 2D greyscale frames".format(
                                tuple(frame.shape), os.path.basename(source_path)))
                    if writer is None:
                        height, width = frame.shape[0], frame.shape[1]
                        writer = cv2.VideoWriter(
                            part_path, cv2.VideoWriter_fourcc(*code), fps,
                            (width, height), True)
                        if not writer.isOpened():
                            raise RuntimeError(
                                "OpenCV cannot write '{0}' ({1} @ {2:.3g} fps, "
                                "{3}x{4}): try MOVIE_FORMAT='avi' (fourcc 'MJPG')".format(
                                    os.path.basename(movie_path), code, fps, width, height))
                        size = (width, height)
                    writer.write(frame)
                    frames += 1
    finally:
        if writer is not None:
            writer.release()
    if frames == 0:
        raise RuntimeError("no frame could be read from {0}".format(
            os.path.basename(source_path)))
    os.replace(part_path, movie_path)
    return frames, size


def movie_wanted():
    """True when this run should encode a movie; says why not when it cannot."""
    if not WRITE_MOVIE:
        return False
    if not CV2_AVAILABLE:
        print("  WRITE_MOVIE is on but OpenCV is unavailable -> no movie "
              "(install it with: pip install opencv-python-headless)")
        return False
    if not STAMP_TIME:
        print("  WRITE_MOVIE is on but STAMP_TIME is off -> no movie (the movie "
              "is encoded from the labelled 8-bit frames)")
        return False
    return True


def export_movie(source_path, sample_dir, stack_name, report):
    """Encode an already written frame stack into the configured movie file."""
    try:
        fourcc, extension = movie_codec()
    except ValueError as error:
        print("  movie FAILED: {0}".format(error))
        return None
    name = movie_name_for(stack_name, extension)
    path = os.path.join(sample_dir, name)
    info = {"path": path, "name": name, "fourcc": fourcc, "fps": float(MOVIE_FPS)}
    report["output"]["movie"] = info
    if not source_path or not os.path.isfile(source_path):
        info["status"] = "skipped: no labelled frames"
        print("  movie skipped: {0} is missing (run once with OVERWRITE=True)".format(
            os.path.basename(source_path) if source_path else "the labelled stack"))
        return None
    if os.path.isfile(path) and not OVERWRITE:
        info["status"] = "kept"
        print("  keeping existing " + name)
        return path
    try:
        frames, size = write_movie(source_path, path)
    except Exception as error:       # a failed movie must never lose the stack
        info["status"] = "failed: {0}".format(error)
        print("  movie FAILED: {0}".format(error))
        return None
    size_mb = os.path.getsize(path) / 1e6
    info.update({"status": "written", "frames": int(frames),
                 "frame_width": int(size[0]), "frame_height": int(size[1]),
                 "size_mb": round(size_mb, 3)})
    print("  movie: {0} ({1} frame(s) @ {2:.3g} fps, {3}, {4:.1f} MB)".format(
        name, frames, float(MOVIE_FPS), fourcc, size_mb))
    return path


def config_snapshot():
    """CONFIG block of the run, stored in the reports."""
    return {
        "root_dir": ROOT_DIR,
        "recursive": bool(RECURSIVE),
        "sample_ids": SAMPLE_IDS,
        "channel": CHANNEL,
        "depth_pixel": DEPTH_PIXEL,
        "depth_range": DEPTH_RANGE,
        "downsample_x": DOWNSAMPLE_X,
        "downsample_y": DOWNSAMPLE_Y,
        "downsample_z": DOWNSAMPLE_Z,
        "downsample_mode": DOWNSAMPLE_MODE,
        "time_start": TIME_START,
        "time_stop": TIME_STOP,
        "time_stride": TIME_STRIDE,
        "output_dtype": OUTPUT_DTYPE,
        "output_name": OUTPUT_NAME,
        "overwrite": bool(OVERWRITE),
        "dry_run": bool(DRY_RUN),
        "stamp_time": bool(STAMP_TIME),
        "stamp_output": STAMP_OUTPUT,
        "stamp_time_mode": STAMP_TIME_MODE,
        "stamp_time_origin": STAMP_TIME_ORIGIN,
        "stamp_area": STAMP_AREA,
        "stamp_position": STAMP_POSITION,
        "stamp_scale": STAMP_SCALE,
        "stamp_font": STAMP_FONT or _font_path(),
        "scale_bar": bool(SCALE_BAR),
        "scale_bar_um_per_px": SCALE_BAR_UM_PER_PX,
        "scale_bar_width_fraction": SCALE_BAR_WIDTH_FRACTION,
        "scale_bar_area": SCALE_BAR_AREA,
        "scale_bar_position": SCALE_BAR_POSITION,
        "scale_bar_text_area": SCALE_BAR_TEXT_AREA,
        "scale_bar_font_px": SCALE_BAR_FONT_PX,
        "scale_bar_font_min_px": SCALE_BAR_FONT_MIN_PX,

        "write_movie": bool(WRITE_MOVIE),
        "movie_format": MOVIE_FORMAT,
        "movie_fps": MOVIE_FPS,
        "movie_fourcc": MOVIE_FOURCC,
        "movie_name": MOVIE_NAME,
    }


def process_sample(sample_id, sample_dir, root_dir=None, channel=CHANNEL):
    """Build the ``(T, Y, X)`` depth time stack of one sample.

    The stack is written page by page straight from the source TIFFs (one page =
    one time point), so the RAM cost stays one plane regardless of the volume or
    time-point count, and the output is renamed into place only when complete.
    """
    report = {
        "sample_id": int(sample_id),
        "sample_dir": sample_dir,
        "channel": channel,
        "status": "pending",
        "time_points": 0,
        "planes": [],
        "skipped": [],
    }
    sources, skipped, time_dirs = plan_planes(sample_dir, channel)
    report["skipped"] = skipped
    report["time_points"] = len(sources)
    if not time_dirs:
        report["status"] = "skipped: no Time-* folder"
        print("  no Time-* folder -> skipped")
        return report
    if not sources:
        report["status"] = "skipped: no usable {0} volume in {1} time point(s)".format(
            channel, len(time_dirs))
        print("  " + report["status"])
        return report

    z0, z1 = resolve_depth_window(DEPTH_PIXEL, DEPTH_RANGE, DOWNSAMPLE_Z)
    y_src = min(entry["shape"][0] for entry in sources)
    x_src = min(entry["shape"][1] for entry in sources)
    z_src = min(entry["shape"][2] for entry in sources)
    if z0 >= z_src:
        report["status"] = (
            "skipped: the depth window starts at {0} but the volumes only have "
            "Z={1}".format(z0, z_src))
        print("  " + report["status"])
        return report
    if z1 > z_src:
        print("  depth rows {0}..{1} exceed Z={2} -> clipped".format(
            z0, z1 - 1, z_src))
    y_out = downsample_size(y_src, DOWNSAMPLE_Y)
    x_out = downsample_size(x_src, DOWNSAMPLE_X)
    t_count = len(sources)
    out_name = OUTPUT_NAME or output_filename(t_count, y_out, x_out, z0, z1, channel)
    out_path = os.path.join(sample_dir, out_name)
    dtype = resolve_output_dtype(OUTPUT_DTYPE, source_dtype(sources[0]["path"]))
    labels, origin, missing_stamps = time_labels(sources)
    labeler = None
    burn = False
    if STAMP_TIME:
        burn = str(STAMP_OUTPUT or "sidecar").lower() == "burn"
        if not PIL_AVAILABLE:
            print("  STAMP_TIME is on but Pillow is unavailable -> no time label "
                  "(install it with: pip install pillow)")
        else:
            labeler = FrameLabeler(max(labels, key=len) if labels else "", (y_out, x_out))
            print("  label: " + labeler.describe())
            if burn:
                print("  STAMP_OUTPUT='burn': the stack itself holds 8-bit labelled frames")
    if missing_stamps:
        print("  warning: {0} time point(s) have no {1} timestamp -> labelled "
              "'Time-N'".format(missing_stamps, SOURCE_TIME_META))
    if labeler is None:      # Pillow missing or STAMP_TIME off: never "burn" nothing
        burn = False
    scale_bar = None
    if SCALE_BAR:
        if not PIL_AVAILABLE:
            print("  SCALE_BAR is on but Pillow is unavailable -> no scale bar "
                  "(install it with: pip install pillow)")
        else:
            if SCALE_BAR_UM_PER_PX:
                step_um = float(SCALE_BAR_UM_PER_PX)
                pitch_from = "SCALE_BAR_UM_PER_PX"
            else:
                step_x, step_y, pitch_from = sample_um_per_px(sources)
                step_um = step_x if step_x else step_y
                if step_um:
                    step_um *= float(max(1, int(DOWNSAMPLE_X)))
            if not step_um:
                print("  SCALE_BAR is on but {0} holds no lateral step -> no scale "
                      "bar".format(SOURCE_TIME_META))
            else:
                scale_bar = ScaleBar((y_out, x_out), step_um,
                                     unit_total_um=x_out * step_um)
                print("  scale bar: {0}  (pitch from {1})".format(
                    scale_bar.describe(), pitch_from))

    stamp_name = None
    if labeler is not None and not burn:
        stamp_name = out_name[:-4] + "-timeStamp.tif"
    block_size = 1 if burn else dtype.itemsize
    dtype_name = "uint8 (display)" if burn else dtype.name
    size_mb = t_count * y_out * x_out * block_size / 1e6

    report.update({
        "root_dir": root_dir,
        "depth_window": {"first": z0, "last": min(z1, z_src) - 1,
                         "rows": min(z1, z_src) - z0,
                         "depth_pixel": DEPTH_PIXEL, "depth_range": DEPTH_RANGE},
        "downsample": {"x": int(DOWNSAMPLE_X), "y": int(DOWNSAMPLE_Y),
                       "z": max(1, int(DOWNSAMPLE_Z)), "mode": DOWNSAMPLE_MODE},
        "source_shape": [y_src, x_src, z_src],
        "output": {"path": out_path, "name": out_name,
                   "dtype": "uint8" if burn else str(dtype),
                   "shape": [t_count, y_out, x_out], "movie": None},
        "time_axis": {"mode": str(STAMP_TIME_MODE).lower(),
                      "origin": origin.isoformat(sep=" ", timespec="seconds")
                      if origin else None,
                      "first": sources[0]["timestamp"],
                      "last": sources[-1]["timestamp"],
                      "missing_timestamps": missing_stamps},
        "labels": {"style": labeler.describe() if labeler is not None else None,
                   "stamped_stack": stamp_name,
                   "scale_bar": scale_bar.describe() if scale_bar is not None else None},

    })
    print("  T={0}  depth {1}..{2}  Y {3}/Yd{4}={5}  X {6}/Xd{7}={8}  {9}  ~{10:.0f} MB".format(
        t_count, z0, min(z1, z_src) - 1, y_src, int(DOWNSAMPLE_Y), y_out,
        x_src, int(DOWNSAMPLE_X), x_out, dtype_name, size_mb))

    if DRY_RUN:
        for index, entry in enumerate(sources):
            print("    T{0:>3}  Time-{1:<4} {2:<8} {3:<9} {4:<20} {5}".format(
                index, entry["time_id"], entry["kind"], labels[index],
                entry["timestamp"] or "-", entry["name"]))
        if WRITE_MOVIE:
            try:
                fourcc, extension = movie_codec()
            except ValueError as error:
                print("    movie: invalid config - {0}".format(error))
            else:
                print("    -> movie {0} at {1:.3g} fps ({2})".format(
                    movie_name_for(out_name, extension), float(MOVIE_FPS), fourcc))
        report["status"] = "dry run: nothing written"
        return report

    if os.path.isfile(out_path) and not OVERWRITE:
        report["status"] = "kept existing {0}".format(out_name)
        print("  keeping existing " + out_name)
        if movie_wanted() and labeler is not None:      # the movie may still be new
            labelled = out_path if burn else (
                os.path.join(sample_dir, stamp_name) if stamp_name else None)
            export_movie(labelled, sample_dir, out_name, report)
            if scale_bar is not None and scale_bar.value_um is not None:
                print("  note: the kept stamped stack has no scale bar -> set "
                      "OVERWRITE=True to burn one in")

        return report

    part_path = out_path + ".part"
    stamp_path = os.path.join(sample_dir, stamp_name) if stamp_name else None
    stamp_part = stamp_path + ".part" if stamp_path else None
    use_stack_window = str(STAMP_SCALE or "frame").lower() == "stack"
    stack_window = None
    written = 0
    writer = None
    stamp_writer = None
    try:
        writer = TIFF.TiffWriter(part_path, bigtiff=True)
        if stamp_path:
            stamp_writer = TIFF.TiffWriter(stamp_part, bigtiff=True)
        for index, entry in enumerate(sources):
            plane = read_depth_plane(entry["path"], z0, z1, DOWNSAMPLE_MODE)
            plane = downsample_plane(plane, DOWNSAMPLE_Y, DOWNSAMPLE_X,
                                     DOWNSAMPLE_MODE)
            plane = plane[:y_out, :x_out]
            if plane.shape != (y_out, x_out):
                raise ValueError("time point {0} gave a {1} plane, expected {2}".format(
                    entry["time_id"], plane.shape, (y_out, x_out)))
            frame = None
            if labeler is not None:
                window = stack_window if (use_stack_window and stack_window) else None
                frame = labeler.render(plane, labels[index], window=window)
                if scale_bar is not None:
                    frame = scale_bar.draw(frame)

                if stack_window is None:
                    stack_window = labeler.window(plane)
            if burn:
                writer.write(frame, contiguous=True)     # 8-bit labelled frames
            else:
                writer.write(np.ascontiguousarray(plane, dtype=dtype), contiguous=True)
                if stamp_writer is not None:
                    stamp_writer.write(frame, contiguous=True)
            written += 1
            report["planes"].append({
                "t": index,
                "time_id": entry["time_id"],
                "kind": entry["kind"],
                "source": entry["name"],
                "source_shape": entry["shape"],
                "folder": entry["folder"],
                "timestamp": entry["timestamp"],
                "label": labels[index],
            })
            print("    T{0:>3}  Time-{1:<4} {2:<8} {3}".format(
                index, entry["time_id"], entry["kind"], labels[index]))
    except Exception as error:  # keep going: one bad time point must not stop the run
        report["status"] = "failed: {0}".format(error)
        print("  FAILED: {0}".format(error))
        for stale in (part_path, stamp_part):
            if stale and os.path.isfile(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        return report
    finally:
        for handle in (writer, stamp_writer):
            if handle is not None:
                handle.close()

    os.replace(part_path, out_path)
    if stamp_path:
        os.replace(stamp_part, stamp_path)
    report["output"]["planes_written"] = written
    report["status"] = "written {0} plane(s) -> {1}".format(written, out_name)
    print("  wrote {0} ({1:.0f} MB)".format(out_name, size_mb))
    if stamp_path:
        print("  labelled frames: {0}".format(os.path.basename(stamp_path)))
    if movie_wanted() and labeler is not None:
        export_movie(out_path if burn else stamp_path, sample_dir, out_name, report)
    with open(out_path[:-4] + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def process_root(root, channel=CHANNEL):
    """Every ``sampleID-*`` of one acquisition root -> one stack each."""
    root = os.path.abspath(root)
    print("\n=== {0} ===".format(root))
    if not os.path.isdir(root):
        print("  not a folder -> nothing to do")
        return {"root_dir": root, "samples": [], "status": "not a folder"}
    samples = find_sample_dirs(root, RECURSIVE)
    if SAMPLE_IDS is not None:
        wanted = {int(value) for value in SAMPLE_IDS}
        samples = [item for item in samples if item[0] in wanted]
    names = ", ".join("sampleID-{0}".format(sample_id) for sample_id, _ in samples)
    print("  {0} sample folder(s): {1}".format(len(samples), names or "-"))

    reports = []
    for sample_id, sample_dir in samples:
        print("sampleID-{0}".format(sample_id))
        reports.append(process_sample(sample_id, sample_dir, root, channel))

    run_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": root,
        "config": config_snapshot(),
        "samples": reports,
    }
    if WRITE_ROOT_REPORT and not DRY_RUN:
        report_path = os.path.join(root, "{0}TimeStack-report.json".format(channel))
        try:
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(run_report, handle, indent=2)
            print("run report: {0}".format(report_path))
        except OSError as error:
            print("could not write the run report: {0}".format(error))
    return run_report


def summarise(reports):
    """One line per sample plus the run counters."""
    print("\n=== summary ===")
    counts = {"written": 0, "kept": 0, "skipped": 0, "failed": 0, "dry": 0}
    for report in reports:
        status = str(report.get("status", ""))
        key = status.split(":", 1)[0].split(" ", 1)[0]
        if key not in counts:
            key = "skipped"
        counts[key] += 1
        output = report.get("output") or {}
        planes = output.get("shape", [len(report.get("planes", []))])[0]
        print("  sampleID-{0}: T={1}  {2}".format(
            report.get("sample_id", "?"), planes, status or "-"))
    print("  {0} written, {1} kept, {2} skipped, {3} failed, {4} dry run".format(
        counts["written"], counts["kept"], counts["skipped"], counts["failed"],
        counts["dry"]))


def main(argv):
    """Entry point; ``argv`` may hold acquisition roots instead of ROOT_DIR."""
    roots = [arg for arg in argv if not str(arg).startswith("-")]
    if not roots:
        roots = [ROOT_DIR] if ROOT_DIR else []
    if not roots:
        print("Nothing to do: set ROOT_DIR in the CONFIG block to ONE acquisition "
              "folder (or pass one on the command line).")
        return 1
    if DOWNSAMPLE_MODE not in ("mean", "sample"):
        print("DOWNSAMPLE_MODE must be 'mean' or 'sample', got {0}".format(
            DOWNSAMPLE_MODE))
        return 1
    try:
        movie_fourcc, movie_extension = movie_codec()
    except ValueError as error:
        print(str(error))
        return 1
    if WRITE_MOVIE and float(MOVIE_FPS) <= 0:
        print("MOVIE_FPS must be positive, got {0}".format(MOVIE_FPS))
        return 1

    print("dynamic depth time stack: channel={0}  depth_pixel={1}  depth_range={2}  "
          "downsample X/Y/Z={3}/{4}/{5} ({6})".format(
              CHANNEL, DEPTH_PIXEL, DEPTH_RANGE, DOWNSAMPLE_X, DOWNSAMPLE_Y,
              DOWNSAMPLE_Z, DOWNSAMPLE_MODE))
    if not STAMP_TIME:
        print("time label: off")
    elif not PIL_AVAILABLE:
        print("time label: unavailable - Pillow is missing (pip install pillow)")
    else:
        print("time label: {0} in '{1}' mode, {2:.0%} of the frame, output '{3}'".format(
            STAMP_POSITION, str(STAMP_TIME_MODE).lower(), float(STAMP_AREA),
            STAMP_OUTPUT))
    if WRITE_MOVIE and STAMP_TIME:
        print("movie: {0} ({1}) at {2:.3g} fps, encoded from the labelled frames".format(
            movie_extension, movie_fourcc, float(MOVIE_FPS)))
    elif WRITE_MOVIE:
        print("movie: off (WRITE_MOVIE encodes the labelled frames, so it needs "
              "STAMP_TIME)")
    else:
        print("movie: off")
    reports = []
    for root in roots:
        reports.extend(process_root(root, CHANNEL).get("samples", []))
    summarise(reports)
    failed = any(str(report.get("status", "")).startswith("failed") for report in reports)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
