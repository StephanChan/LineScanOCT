# -*- coding: utf-8 -*-
"""In-run shading (flat / dark) correction for mosaic and plate scans.

Design decisions (agreed with the operator):

* One flat/dark pair is **fitted from the first sample of a scan run** and reused
  for every later sample of the same session, as long as the acquisition geometry
  is unchanged.  There is **no on-disk field library, no phantom acquisition and
  no drift tracking**.
* Data is corrected **before it is written**, so only corrected data reaches the
  disk.  The reference sample itself cannot be corrected while it is being
  acquired (the field does not exist yet), so it is corrected afterwards by
  re-reading and overwriting its tiles **once per run** (``correct_sample_folder``).
* The structure volume uses ``(I - dark) / flat``; the dynamic products use the
  *gain only* (``I / flat``) because a temporal standard deviation has no additive
  camera offset.  For the HSV dynamic volume only the V channel is scaled.
* Per-tile volumes are stored as **float16**; stitched mosaics stay float32
  (Fiji / segmentation compatibility).

Pure numpy + tifffile, no Qt: usable from the DnS thread, the weaver thread and
offline scripts.
"""

import datetime
import json
import os
import threading

import numpy as np
import tifffile as TIFF

from mosaic_correction import (
    BASIC_DARK_PERCENTILE,
    BASIC_ITER,
    BASIC_PER_IMAGE,
    BASIC_TERMS,
    basic_fields,
    expand_field_planes,
    percentile_background,
    read_volume_stack,
    split_amplitude_phase,
    write_volume_stack,
)

# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------
SHADING_CORRECTION_ENABLED_DEFAULT = True   # "本 run 实时阴影校正" 的默认值
SHADING_DEPTH_STEP = 8                      # depth planes between two BaSiC fits
SHADING_MAX_DEPTH_PLANES = 24               # cap on the fitted depth planes
SHADING_MAX_REFERENCE_TILES = 24            # tiles that contribute depth planes
SHADING_MIN_TILES = 3                       # below this no field is fitted
SHADING_STORE_DTYPE = np.float16            # per-tile storage dtype
SHADING_FIELDS_FILE = "bgcorr-fields.npz"
SHADING_STATE_FILE = "bgcorr_state.json"
SHADING_REPORT_FILE = "bgcorr-report.json"

# ---------------------------------------------------------------------------
# dynamic colour volume (H / S / V are PHYSICAL values, not 0..1)
# ---------------------------------------------------------------------------
# The stored colour volume is ``[mean_frequency_hz, bandwidth_hz, dynamic_std]``
# (ThreadGPU.frequency_hsv_metrics_*), so rendering it needs the same three
# normalisation windows the live display uses (Display_rendering /
# ThreadDnS.dynamic_hsv_to_saved_rgb).  The hue window follows the contrast
# sliders (``XZmin`` / ``XZmax`` x 15/1000 Hz per unit) and is recorded per sample
# in ``tile_positions.json`` as ``dynamic_hsv_ranges``; these constants are only
# the fallback for folders recorded before that field existed.
DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT = 15.0 / 1000.0
DYNAMIC_HUE_RANGE_HZ = (0.0, 15.0)
DYNAMIC_SATURATION_RANGE_HZ = (0.0, 8.0)
DYNAMIC_VALUE_RANGE = (0.0, 500.0)
DYNAMIC_VALUE_GAMMA = 1.0

_SESSION_LOCK = threading.Lock()
_SESSION_FIELD = None
_SESSION_SIGNATURE = None


# ---------------------------------------------------------------------------
# field signature / session cache
# ---------------------------------------------------------------------------
def field_signature(x_pixels=0, y_pixels=0, z_pixels=0, x_step_um=0.0,
                    y_step_um=0.0, z_start=0, z_range=0, extra=None):
    """Everything the flat/dark geometry depends on, as a JSON friendly dict."""
    signature = {
        "x_pixels": int(x_pixels or 0),
        "y_pixels": int(y_pixels or 0),
        "z_pixels": int(z_pixels or 0),
        "x_step_um": round(float(x_step_um or 0.0), 6),
        "y_step_um": round(float(y_step_um or 0.0), 6),
        "z_start": int(z_start or 0),
        "z_range": int(z_range or 0),
    }
    if extra:
        for key in sorted(extra):
            signature[key] = extra[key]
    return signature


def signature_matches(a, b):
    """True when two signatures describe the same field geometry."""
    if not a or not b:
        return False
    keys = ("x_pixels", "y_pixels", "z_pixels", "x_step_um", "y_step_um",
            "z_start", "z_range")
    return all(a.get(key) == b.get(key) for key in keys)


def get_session_field(signature=None, require_match=True):
    """The field fitted in this session, or None (optionally geometry checked)."""
    with _SESSION_LOCK:
        field = _SESSION_FIELD
    if field is None:
        return None
    if signature is not None and require_match and not signature_matches(
        field.signature, signature
    ):
        return None
    return field


def set_session_field(field, signature=None):
    """Remember the field for the rest of the session."""
    global _SESSION_FIELD, _SESSION_SIGNATURE
    with _SESSION_LOCK:
        _SESSION_FIELD = field
        _SESSION_SIGNATURE = signature if signature is not None else (
            field.signature if field is not None else None
        )
    return field


def clear_session_field():
    """Forget the field (next scan re-fits it)."""
    global _SESSION_FIELD, _SESSION_SIGNATURE
    with _SESSION_LOCK:
        _SESSION_FIELD = None
        _SESSION_SIGNATURE = None


# ---------------------------------------------------------------------------
# the field itself
# ---------------------------------------------------------------------------
class ShadingField:
    """Per-depth flat/dark fields plus the per-tile levels of the fit."""

    def __init__(self, flat, dark, z_planes, levels=None, signature=None,
                 tiles=0, terms=BASIC_TERMS, info=None):
        self.flat = np.asarray(flat, dtype=np.float32)
        self.dark = np.asarray(dark, dtype=np.float32)
        self.z_planes = [int(plane) for plane in z_planes]
        self.levels = None if levels is None else np.asarray(levels, np.float32)
        self.signature = dict(signature or {})
        self.tiles = int(tiles)
        self.terms = tuple(int(value) for value in terms)
        self.info = dict(info or {})
        self.created = datetime.datetime.now().isoformat(timespec="seconds")

    @property
    def per_depth(self):
        return self.flat.ndim == 3

    def describe(self):
        return (
            "{0} tile(s), {1} depth planes ({2}), flat {3:.3f}..{4:.3f}, "
            "dark {5:.3f}..{6:.3f}".format(
                self.tiles, self.flat.shape[0] if self.per_depth else 1,
                self.z_planes[0] if self.z_planes else 0,
                float(self.flat.min()), float(self.flat.max()),
                float(self.dark.min()), float(self.dark.max()),
            )
        )

    # -- geometry helpers ---------------------------------------------------
    def depth_fields(self, z_start, z_count):
        """Fields over ``z_count`` depths starting at ``z_start``: ``(Z, Y, X)``."""
        z_count = int(max(1, z_count))
        if not self.per_depth:
            return (np.repeat(self.flat[None, ...], z_count, axis=0),
                    np.repeat(self.dark[None, ...], z_count, axis=0))
        planes = [plane + int(z_start) for plane in self.z_planes]
        return (expand_field_planes(self.flat, planes, z_count),
                expand_field_planes(self.dark, planes, z_count))

    def gain_at(self, z_depth):
        """``1 / flat`` at one **depth** as a ``[Y, X]`` map (2-D dynamic maps).

        The index is a depth along the tile's Z axis (not a plane index): the two
        neighbouring fitted planes are interpolated, so the gain always matches the
        depth slice the dynamic map was taken at.
        """
        if not self.per_depth:
            return (1.0 / np.maximum(self.flat, 1e-3)).astype(np.float32)
        field_flat, _dark = self.depth_fields(int(z_depth), 1)
        return (1.0 / np.maximum(field_flat[0], 1e-3)).astype(np.float32)

    def line_fields(self, y_row, z_start, z_count):
        """``(flat, dark)`` of one Y line as ``(X, Z)`` (per-line correction).

        Only that row is interpolated (``expand_field_planes`` would build the whole
        ``[Z, Y, X]`` block, which is what made a per-line correction expensive);
        the interpolation itself is identical, so a line-wise correction equals the
        whole-volume one.
        """
        y_row = int(y_row)
        z_count = int(max(1, z_count))
        if not self.per_depth:
            flat = np.repeat(self.flat[y_row][None, :], z_count, axis=0)
            dark = np.repeat(self.dark[y_row][None, :], z_count, axis=0)
            return flat.T, dark.T
        planes = np.asarray(self.z_planes, dtype=np.float64).ravel() + int(z_start)
        flat_planes = self.flat[:, y_row, :]
        dark_planes = self.dark[:, y_row, :]
        flat = np.empty((z_count, flat_planes.shape[-1]), dtype=np.float32)
        dark = np.empty_like(flat)
        for z in range(z_count):
            index = int(np.clip(np.searchsorted(planes, z, side="right") - 1,
                                0, planes.size - 1))
            if index >= planes.size - 1:
                flat[z] = flat_planes[-1]
                dark[z] = dark_planes[-1]
                continue
            span = planes[index + 1] - planes[index]
            weight = 0.0 if span <= 0 else float((z - planes[index]) / span)
            flat[z] = (1.0 - weight) * flat_planes[index] + weight * flat_planes[index + 1]
            dark[z] = (1.0 - weight) * dark_planes[index] + weight * dark_planes[index + 1]
        return flat.T, dark.T

    # -- application --------------------------------------------------------
    def apply_structure_line(self, line, y_row, z_start, z_count, z_logical=None):
        """Correct one line of data, ``[X, Z]`` or ``[X, 2Z]`` (amp+phase).

        ``z_logical`` is the logical depth count: when the line is stored as
        interleaved amplitude+phase (2Z) only the amplitude half is corrected and
        the phase is passed through, so the stored layout is untouched.  The fields
        are looked up over the *logical* depth window, never over the interleaved
        length, which would interpolate the field over twice the Z range.
        """
        depth = int(z_logical) if z_logical else int(z_count)
        flat_map, dark_map = self.line_fields(y_row, z_start, depth)
        flat_map = np.maximum(flat_map, 1e-3)
        data = np.asarray(line)
        if z_logical and data.ndim >= 2 and data.shape[-1] == 2 * int(z_logical):
            out = np.array(data, dtype=np.float32, copy=True)
            out[..., :depth] = (data[..., :depth] - dark_map) / flat_map
            return out
        return (data - dark_map) / flat_map

    def apply_structure_volume(self, volume, z_start=0, clip=False, inplace=False):
        """Correct a whole ``[Y, X, Z]`` amplitude volume (``(I - dark)/flat``)."""
        out = np.asarray(volume, np.float32) if inplace else np.array(volume, np.float32, copy=True)
        z_count = int(out.shape[2])
        flat, dark = self.depth_fields(z_start, z_count)
        for z in range(z_count):
            out[:, :, z] = (out[:, :, z] - dark[z]) / np.maximum(flat[z], 1e-3)
        if clip:
            np.maximum(out, 0.0, out=out)
        return out

    def apply_gain_volume(self, volume, z_start=0, clip=False, inplace=False):
        """Scale a whole volume by the flat gain only (dynamic products)."""
        out = np.asarray(volume, np.float32) if inplace else np.array(volume, np.float32, copy=True)
        z_count = int(out.shape[2])
        flat, _dark = self.depth_fields(z_start, z_count)
        for z in range(z_count):
            out[:, :, z] /= np.maximum(flat[z], 1e-3)
        if clip:
            np.maximum(out, 0.0, out=out)
        return out

    def apply_gain_channel(self, channel, z_start=0, clip=False, inplace=False):
        """Scale one ``[Y, X, Z]`` dynamic channel (the V of the colour volume)."""
        return self.apply_gain_volume(channel, z_start=z_start, clip=clip, inplace=inplace)

    def apply_gain_map(self, image, z_depth):
        """Scale a ``[Y, X]`` (or ``[Y, X, C]``) dynamic map by the flat gain."""
        gain = self.gain_at(z_depth)
        if np.asarray(image).ndim == 3:
            gain = gain[..., None]
        return np.asarray(image, dtype=np.float32) * gain



# ---------------------------------------------------------------------------
# file naming (single source of truth for the tile products)
# ---------------------------------------------------------------------------
def tile_filenames(record):
    """The files of one tile that shading correction touches.

    ``structure`` is the mean-intensity volume of the FOV (``mean_filename`` for a
    dynamic acquisition, ``tile_filename`` for a static one); the HSV names are the
    three per-channel files of the dynamic colour volume.
    """
    names = {
        "structure": record.get("mean_filename") or record.get("tile_filename"),
        "raw": record.get("tile_filename"),
        "dynamic_std": record.get("dynamic_std_filename") or record.get("dynamic_filename"),
        "dynamic_h": record.get("dynamic_h_filename"),
        "dynamic_s": record.get("dynamic_s_filename"),
        "dynamic_v": record.get("dynamic_v_filename"),
        "dynamic_rgb": record.get("dynamic_rgb_filename"),      # legacy layout
    }
    return {key: value for key, value in names.items() if value}


def dynamic_channel_names(tile_index, y_pixels, x_pixels, z_pixels):
    """Filenames of the H / S / V dynamic volumes of one tile."""
    base = "tile-{0}-Dyn{{0}}-Y{1}-X{2}-Z{3}.tif".format(
        int(tile_index), int(y_pixels), int(x_pixels), int(z_pixels)
    )
    return (base.format("H"), base.format("S"), base.format("V"))


# ---------------------------------------------------------------------------
# HSV -> RGB (same maths and same normalisation windows as the live display)
# ---------------------------------------------------------------------------
def normalize_dynamic_channel(image, value_range, gamma=1.0):
    """Map a physical dynamic channel into 0..1 over its display window."""
    low_value, high_value = float(value_range[0]), float(value_range[1])
    if high_value <= low_value:
        raise ValueError(
            "Invalid dynamic HSV normalization range: {0}".format(value_range)
        )
    normalized = (np.asarray(image, dtype=np.float32) - low_value) / (
        high_value - low_value
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    gamma = float(gamma)
    if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        normalized = normalized ** (1.0 / gamma)
    return normalized


def hsv_to_rgb_array(hue, saturation, value):
    """``[..., 3]`` HSV in 0..1 to uint8 RGB (identical to the live display)."""
    hue = np.mod(np.asarray(hue, dtype=np.float32), 1.0)
    saturation = np.clip(np.asarray(saturation, dtype=np.float32), 0.0, 1.0)
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    h6 = hue * 6.0
    index = np.floor(h6).astype(np.int32)
    fraction = h6 - index.astype(np.float32)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    index_mod = np.mod(index, 6)
    rgb = np.empty(hue.shape + (3,), dtype=np.float32)
    for mask, red, green, blue in (
        (index_mod == 0, value, t, p),
        (index_mod == 1, q, value, p),
        (index_mod == 2, p, value, t),
        (index_mod == 3, p, q, value),
        (index_mod == 4, t, p, value),
        (index_mod == 5, value, p, q),
    ):
        rgb[..., 0][mask] = red[mask]
        rgb[..., 1][mask] = green[mask]
        rgb[..., 2][mask] = blue[mask]
    return np.ascontiguousarray(
        np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    )


def _safe_range(value_range, fallback, label):
    """A usable ``(low, high)`` window: swapped is fixed, degenerate falls back."""
    if value_range is None:
        return fallback
    try:
        low, high = float(value_range[0]), float(value_range[1])
    except (TypeError, ValueError, IndexError):
        print("Dynamic colour: unusable {0} range {1}; using {2}".format(
            label, value_range, fallback))
        return fallback
    if not (np.isfinite(low) and np.isfinite(high)):
        print("Dynamic colour: non-finite {0} range {1}; using {2}".format(
            label, value_range, fallback))
        return fallback
    if high < low:
        low, high = high, low
    if high <= low:
        print("Dynamic colour: degenerate {0} range {1}; using {2}".format(
            label, value_range, fallback))
        return fallback
    return (low, high)


def dynamic_hsv_ranges(hue_hz=None, saturation_hz=None, value=None, value_gamma=None):
    """The normalisation windows of the stored colour volume, for the manifest."""
    return {
        "hue_hz": [float(v) for v in (hue_hz or DYNAMIC_HUE_RANGE_HZ)],
        "saturation_hz": [float(v) for v in (saturation_hz or DYNAMIC_SATURATION_RANGE_HZ)],
        "value": [float(v) for v in (value or DYNAMIC_VALUE_RANGE)],
        "value_gamma": float(DYNAMIC_VALUE_GAMMA if value_gamma is None else value_gamma),
    }


def hsv_to_rgb(hsv, ranges=None, hue_range=None, saturation_range=None,
               value_range=None, value_gamma=None):
    """Render a physical ``[..., 3]`` (H Hz, S Hz, V) colour volume to uint8 RGB.

    The channels *must* be normalised over their display windows first: feeding a
    frequency in Hz straight into the HSV maths (treating it as a 0..1 hue) turns
    the hue index into noise, which is exactly what a "noisy" dynamic RGB mosaic
    looks like.  ``ranges`` is the ``dynamic_hsv_ranges`` block of the manifest.
    """
    if ranges:
        hue_range = hue_range if hue_range is not None else ranges.get("hue_hz")
        saturation_range = (saturation_range if saturation_range is not None
                            else ranges.get("saturation_hz"))
        value_range = value_range if value_range is not None else ranges.get("value")
        value_gamma = (value_gamma if value_gamma is not None
                       else ranges.get("value_gamma"))
    hsv = np.asarray(hsv, dtype=np.float32)
    if hsv.ndim < 3 or hsv.shape[-1] != 3:
        raise ValueError(
            "Dynamic HSV source must have last dimension 3, got {0}".format(hsv.shape)
        )
    hue = normalize_dynamic_channel(hsv[..., 0], _safe_range(
        hue_range, DYNAMIC_HUE_RANGE_HZ, "hue"))
    saturation = normalize_dynamic_channel(hsv[..., 1], _safe_range(
        saturation_range, DYNAMIC_SATURATION_RANGE_HZ, "saturation"))
    value = normalize_dynamic_channel(
        hsv[..., 2], _safe_range(value_range, DYNAMIC_VALUE_RANGE, "value"),
        gamma=DYNAMIC_VALUE_GAMMA if value_gamma is None else value_gamma,
    )
    return hsv_to_rgb_array(hue, saturation, value)



# ---------------------------------------------------------------------------
# accumulating the fit inputs while the reference sample is acquired
# ---------------------------------------------------------------------------
class FieldAccumulator:
    """Collects what the flat/dark fit needs while the reference sample is scanned.

    RAM only: the AIP of every tile plus the amplitude planes of the fit depths for
    up to ``max_tiles`` tiles, spread over the sample so the order statistics see
    the whole field.  Nothing is written and nothing is read back from disk.
    """

    def __init__(self, signature=None, depth_planes=None, max_tiles=None,
                 tile_total=None):
        self.signature = dict(signature or {})
        self.depth_planes = [int(plane) for plane in (depth_planes or [])]
        self.max_tiles = int(max_tiles or SHADING_MAX_REFERENCE_TILES)
        self.tile_total = None if not tile_total else int(tile_total)
        self.aips = []
        self.plane_stacks = {plane: [] for plane in self.depth_planes}
        self.tile_count = 0
        self.plane_tiles = 0

    def _plane_wanted(self, index):
        """Spread the depth-plane tiles evenly over the whole sample."""
        if self.plane_tiles >= self.max_tiles:
            return False
        total = self.tile_total or 0
        if total <= self.max_tiles:
            return True
        stride = max(1, int(round(total / float(self.max_tiles))))
        return (int(index) % stride) == 0

    def add_tile(self, structure_volume, index=None):
        """Add one tile's amplitude volume ``[Y, X, Z]`` (float32)."""
        volume = np.asarray(structure_volume)
        if volume.ndim != 3 or not np.size(volume):
            return False
        self.aips.append(np.asarray(volume.mean(axis=2), np.float32))
        position = self.tile_count if index is None else int(index)
        if self.depth_planes and self._plane_wanted(position):
            for plane in self.depth_planes:
                if plane < volume.shape[2]:
                    self.plane_stacks[plane].append(
                        np.asarray(volume[:, :, plane], np.float32)
                    )
            self.plane_tiles += 1
        self.tile_count += 1
        return True

    @property
    def ready(self):
        return self.tile_count >= SHADING_MIN_TILES

    def reset(self):
        self.aips = []
        self.plane_stacks = {plane: [] for plane in self.depth_planes}
        self.tile_count = 0
        self.plane_tiles = 0

    def fit(self, terms=BASIC_TERMS, n_iter=BASIC_ITER,
            dark_percentile=BASIC_DARK_PERCENTILE, per_image=BASIC_PER_IMAGE,
            verbose=False):
        """Fit the field from what was accumulated (reuses ``basic_fields``)."""
        if not self.ready or not self.aips:
            return None
        aip_stack = np.stack(self.aips)
        flat, dark, levels = basic_fields(
            aip_stack, terms=terms, n_iter=n_iter,
            dark_percentile=dark_percentile, per_image=per_image, verbose=verbose,
        )
        reference_energy = 0.0
        for plane in self.depth_planes:
            stack = self.plane_stacks.get(plane) or []
            if stack:
                reference_energy = max(
                    reference_energy, float(np.mean(np.abs(np.stack(stack[:2]))))
                )
        flats, darks, scales, fitted_planes = [], [], [], []
        for plane in self.depth_planes:
            stack = self.plane_stacks.get(plane) or []
            energy = float(np.mean(np.abs(np.stack(stack)))) if stack else 0.0
            if not stack or energy <= 1e-9 * max(reference_energy, 1e-9):
                # blank depth plane (above the surface): keep the previous fields
                flats.append(flats[-1] if flats else flat)
                darks.append(darks[-1] if darks else dark)
                scales.append(scales[-1] if scales else levels)
                continue
            plane_flat, plane_dark, plane_scale = basic_fields(
                np.stack(stack), terms=terms, n_iter=n_iter,
                dark_percentile=dark_percentile, per_image=per_image, verbose=False,
            )
            flats.append(plane_flat)
            darks.append(plane_dark)
            scales.append(plane_scale)
            fitted_planes.append(int(plane))
        if fitted_planes:
            field_flat = np.stack(flats)
            field_dark = np.stack(darks)
            field_levels = np.stack(scales)
        else:
            field_flat, field_dark, field_levels = flat, dark, levels
        return ShadingField(
            field_flat, field_dark, fitted_planes, levels=field_levels,
            signature=self.signature, tiles=self.tile_count, terms=terms,
            info={
                "fit_tiles": int(self.tile_count),
                "plane_tiles": int(self.plane_tiles),
                "depth_planes": [int(plane) for plane in fitted_planes],
                "aip_only": not bool(fitted_planes),
            },
        )



# ---------------------------------------------------------------------------
# persistence (record only: never used as a field library)
# ---------------------------------------------------------------------------
def save_field_files(folder, field, extra=None):
    """Write the fitted field (npz) and a small JSON report into the sample folder."""
    os.makedirs(folder, exist_ok=True)
    npz_path = os.path.join(folder, SHADING_FIELDS_FILE)
    np.savez_compressed(
        npz_path,
        flat=np.asarray(field.flat, np.float32),
        dark=np.asarray(field.dark, np.float32),
        z_planes=np.asarray(field.z_planes, np.float32),
        levels=np.asarray(field.levels if field.levels is not None else np.zeros(0),
                          np.float32),
        signature=np.asarray(json.dumps(field.signature)),
        info=np.asarray(json.dumps(field.info)),
        created=np.asarray(field.created),
    )
    report = {
        "field": {
            "created": field.created,
            "tiles": field.tiles,
            "per_depth": bool(field.per_depth),
            "depth_planes": list(field.z_planes),
            "terms": list(field.terms),
            "flat_range": [float(field.flat.min()), float(field.flat.max())],
            "dark_range": [float(field.dark.min()), float(field.dark.max())],
            "level_range": None if field.levels is None or not np.size(field.levels)
            else [float(field.levels.min()), float(field.levels.max())],
            "signature": field.signature,
            "info": field.info,
        },
        "npz": npz_path,
    }
    if extra:
        report.update(extra)
    report_path = os.path.join(folder, SHADING_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return npz_path, report_path


def load_field_file(npz_path):
    """Load a field written by :func:`save_field_files`."""
    with np.load(npz_path, allow_pickle=False) as data:
        signature = json.loads(str(data["signature"])) if "signature" in data else {}
        info = json.loads(str(data["info"])) if "info" in data else {}
        levels = data["levels"] if "levels" in data and data["levels"].size else None
        z_planes = data["z_planes"] if "z_planes" in data else np.zeros(0)
        field = ShadingField(
            data["flat"], data["dark"],
            [int(plane) for plane in np.asarray(z_planes).ravel()],
            levels=levels, signature=signature, tiles=int(info.get("fit_tiles", 0)),
            info=info,
        )
    return field



# ---------------------------------------------------------------------------
# correcting the reference sample after the fact (once per run)
# ---------------------------------------------------------------------------
def _correct_volume_file(path, field, z_logical, dtype, mode, verbose=True):
    """Read one tile volume, correct it and write it back in ``dtype``."""
    volume = read_volume_stack(path)
    if volume.ndim < 3:
        return False
    amplitude, phase = split_amplitude_phase(volume, z_logical)
    amplitude = np.asarray(amplitude, np.float32)
    if mode == "structure":
        corrected = field.apply_structure_volume(amplitude)
    else:
        corrected = field.apply_gain_volume(amplitude)
    stack = combine_amplitude_phase(corrected, phase) if phase is not None else corrected
    write_volume_stack(path, np.asarray(stack, dtype))
    if verbose:
        print("    corrected {0} ({1})".format(os.path.basename(path), mode))
    return True


def correct_sample_folder(folder, records, field, z_logical=None, dtype=None,
                          state_name=None, verbose=True, structure_only=False):
    """Correct the reference sample's files on disk (read -> correct -> overwrite).

    Runs once per scan run, for the sample the field was fitted from: every later
    sample is corrected while it is written, so its data never needs a second pass.
    The pass is resumable through a state file, so an interrupted run does not leave
    part of the sample corrected without knowing which part.
    """
    dtype = np.dtype(dtype if dtype is not None else SHADING_STORE_DTYPE)
    state_path = os.path.join(folder, state_name or SHADING_STATE_FILE)
    state = {
        "signature": field.signature,
        "done": [],
        "mode": "shading-corrected",
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("signature") == field.signature:
                state["done"] = list(previous.get("done") or [])
        except (OSError, ValueError):
            pass
    done = set(int(value) for value in state["done"])
    summary = {"tiles": 0, "files": 0, "skipped": 0, "resumed": len(done)}
    for record in records:
        index = int(record.get("tile_index") or 0)
        if index in done:
            continue
        names = tile_filenames(record)
        for key, mode in (("structure", "structure"), ("raw", "structure"),
                          ("dynamic_std", "gain"), ("dynamic_v", "gain")):
            if structure_only and mode == "gain":
                continue
            name = names.get(key)
            if not name:
                continue
            if key == "raw" and names.get("structure") == name:
                continue
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                summary["skipped"] += 1
                continue
            try:
                if _correct_volume_file(path, field, z_logical, dtype, mode, verbose):
                    summary["files"] += 1
            except Exception as error:
                print("    correction failed for {0}: {1}".format(name, error))
                summary["skipped"] += 1
        summary["tiles"] += 1
        done.add(index)
        state["done"] = sorted(done)
        try:
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
        except OSError as error:
            print("    could not update the correction state: {0}".format(error))
    return summary

