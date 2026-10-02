# -*- coding: utf-8 -*-
"""Mosaic tile placement: one source of truth for stage mm -> mosaic pixels.

Everything that assembles a mosaic (the live display in ``ThreadDnS``, the
offline stitchers in ``DynamicPostprocessing`` / ``standalone_mosaic_stitch``,
the tile manifest in ``ThreadWeaver``) must place tiles with the *same* rule,
otherwise the FOV boxes the operator sees, the manifest grid and the stitched
file disagree.

The rule here (and the reason this module exists):

* a tile is placed at its **physical** stage offset, ``(x - x_min) / mm_per_px``,
  so the mosaic pixel size is the real one no matter how much the FOVs overlap;
* the canvas is the **physical extent** of the scan (``span / mm_per_px + tile``),
  so overlap never changes the scale and can never push two tiles into one cell;
* the mirror that the app always had is kept (larger stage X -> left, larger
  stage Y -> top), so existing outputs do not flip;
* the overlap strips are **cross-faded** (``feather=True``): a tile that covers
  ground already written blends with it instead of overwriting it, which removes
  the hard seam that a plain paste leaves.  The ramps are raised cosines (zero
  slope at both ends, so no band is visible) and, when the caller keeps a weight
  map, the blend is normalised so a faded tile is never darkened.

Old behaviour ``round((x - x_min) / fov_mm)`` (i.e. "the step equals the FOV
size") is only correct at 0 % overlap: at 10 % overlap seven rows drift by 0.6
cells and two of them land in the same cell, so the last one silently replaced
the first.  No such round() is used here.

Pure numpy, no Qt / hardware imports, usable from every thread and script.
"""

from dataclasses import dataclass, field

import numpy as np

POSITION_TOL_MM = 1e-6      # two tiles closer than this count as the same cell
BLEND_BLOCK_ROWS = 16       # rows per float block of a cross-fade (bounded RAM)


def cos_ramp(length, fade_start=0, fade_end=0):
    """1-D cross-fade ramp of ``length`` samples: raised cosine, ends ~0 and ~1.

    ``0.5 - 0.5*cos(pi*t)`` with ``t`` half-pixel centred, so the ramp leaves and
    reaches the plateau with **zero slope** (a linear ramp does not, which is what
    makes its ends visible as a band).  ``fade_start`` / ``fade_end`` are clamped to
    ``length``; 0 disables that side.
    """
    length = max(1, int(length))
    ramp = np.ones(length, dtype=np.float32)
    for fade, at_start in ((int(fade_start), True), (int(fade_end), False)):
        fade = min(max(0, fade), length)
        if fade <= 0:
            continue
        index = (np.arange(fade, dtype=np.float32) + 0.5) / float(fade)
        values = (0.5 - 0.5 * np.cos(np.pi * index)).astype(np.float32)
        if at_start:
            ramp[:fade] = values
        else:
            ramp[length - fade:] = values[::-1]
    return ramp


def new_weight_map(shape):
    """Sum of the tile fade weights of one mosaic plane, for :func:`blend_paste`.

    Read it as ``S(x, y) = sum(w_i)`` over the tiles written so far: the blend is
    the weighted average ``sum(w_i * I_i) / sum(w_i)``, so the effective weight of
    every tile (``w_i / sum(w_j)``) adds up to 1 whatever the overlap is.
    """
    return np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)


def unique_positions(values, tol=POSITION_TOL_MM):
    """Sorted unique positions of one axis (stage mm, tolerance in mm)."""
    unique = []
    for value in sorted(float(item) for item in values):
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def cluster_positions(values, relative_tolerance=0.3):
    """Cluster the positions of one axis and return ``(centres, pitch_mm)``.

    Stage positions carry micrometre noise, so two tiles of the same grid line
    are rarely bit-identical.  The cluster width is taken relative to the largest
    gap (which is the grid pitch itself), so a jittered grid still collapses to
    the right number of lines instead of inventing a line per tile.  ``pitch_mm``
    is the median gap between the cluster centres (None for a single line).
    """
    ordered = sorted(float(item) for item in values)
    if len(ordered) < 2:
        return ordered, None
    gaps = [b - a for a, b in zip(ordered[:-1], ordered[1:])]
    largest = max(gaps)
    if largest <= POSITION_TOL_MM:
        return [float(np.mean(ordered))], None
    tolerance = max(POSITION_TOL_MM, float(relative_tolerance) * largest)
    clusters = []
    for value in ordered:
        if not clusters or value - clusters[-1][-1] > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    centres = [float(np.mean(cluster)) for cluster in clusters]
    steps = [b - a for a, b in zip(centres[:-1], centres[1:]) if b > a]
    return centres, (float(np.median(steps)) if steps else None)



@dataclass
class TilePaste:
    """Where one tile goes and how it blends with the tiles already there."""

    order: int
    row: int
    col: int
    x1: int
    y1: int
    w: int
    h: int
    fade_left: int = 0
    fade_right: int = 0
    fade_top: int = 0
    fade_bottom: int = 0
    _weight: object = field(default=None, repr=False, compare=False)

    @property
    def x2(self):
        return self.x1 + self.w

    @property
    def y2(self):
        return self.y1 + self.h

    @property
    def blends(self):
        return bool(self.fade_left or self.fade_right or self.fade_top or self.fade_bottom)

    def weight(self):
        """Cross-fade weights ``[h, w]`` float32, or None when the tile overwrites.

        The ramps are **raised cosines** (``0.5 - 0.5 cos(pi t)``, half-pixel
        centred): they reach 0 and 1 with zero slope, so no Mach band is visible at
        the ends of a fade, which a linear ramp always leaves behind.  The two axes
        are multiplied, so a corner that is inside a fade on both axes gets the
        product of the two ramps (``blend_paste`` normalises the sum of those
        weights, so the ramps of neighbouring tiles do not have to add up to 1).
        """
        if not self.blends:
            return None
        if self._weight is None:
            wx = cos_ramp(self.w, self.fade_left, self.fade_right)
            wy = cos_ramp(self.h, self.fade_top, self.fade_bottom)
            self._weight = wy[:, None] * wx[None, :]
        return self._weight
@dataclass
class MosaicLayout:
    """Tile placements of one mosaic (one sample, one pixel grid)."""

    fw_px: int
    fh_px: int
    width_px: int
    height_px: int
    mm_per_px_x: float
    mm_per_px_y: float
    origin_x_mm: float
    origin_y_mm: float
    pitch_x_mm: float
    pitch_y_mm: float
    pitch_x_px: int
    pitch_y_px: int
    mirror: bool
    unique_x: list
    unique_y: list
    placements: list
    duplicates: list = field(default_factory=list)
    missing: set = field(default_factory=set)

    @property
    def cols(self):
        return len(self.unique_x)

    @property
    def rows(self):
        return len(self.unique_y)

    @property
    def overlap_x_px(self):
        return max(0, self.fw_px - self.pitch_x_px)

    @property
    def overlap_y_px(self):
        return max(0, self.fh_px - self.pitch_y_px)

    def grid_index(self, x_mm, y_mm):
        """Exact (row, col) of a stage position: index in the unique coordinates.

        This replaces the old ``round((x - x_min) / fov_mm)`` and can never fold
        two tiles into one cell, whatever the overlap is.
        """
        return self._index_of(self.unique_y, y_mm), self._index_of(self.unique_x, x_mm)

    @staticmethod
    def _index_of(unique, value):
        best = 0
        best_distance = None
        for index, candidate in enumerate(unique):
            distance = abs(float(value) - float(candidate))
            if best_distance is None or distance < best_distance:
                best, best_distance = index, distance
        return int(best)

    def problems(self):
        """List of consistency problems (empty when the layout is sound)."""
        issues = []
        if self.duplicates:
            issues.append("duplicate stage positions: " + str(self.duplicates))
        for paste in self.placements:
            if paste.x1 < 0 or paste.y1 < 0 or paste.x2 > self.width_px or paste.y2 > self.height_px:
                issues.append(
                    "tile {0} (row {1}, col {2}) is outside the canvas [{3}x{4}]".format(
                        paste.order, paste.row, paste.col, self.width_px, self.height_px
                    )
                )
        cells = {}
        for paste in self.placements:
            if paste.order in self.missing:
                continue
            key = (paste.row, paste.col)
            if key in cells:
                issues.append(
                    "tiles {0} and {1} share grid cell {2}".format(cells[key], paste.order, key)
                )
            cells[key] = paste.order
        return issues

    def describe(self):
        return (
            "{0} cols x {1} rows, tile {2}x{3} px, pitch {4:.4f}/{5:.4f} mm "
            "({6}/{7} px), overlap {8}/{9} px, canvas {10}x{11} px".format(
                self.cols, self.rows, self.fw_px, self.fh_px,
                self.pitch_x_mm, self.pitch_y_mm, self.pitch_x_px, self.pitch_y_px,
                self.overlap_x_px, self.overlap_y_px, self.height_px, self.width_px,
            )
        )



def pixel_size_mm(step_um=None, length_mm=None, pixels=None, downsample=1):
    """Millimetres per pixel of the target grid.

    The scan sampling step (``x_step_um`` / ``y_step_um``) is authoritative; the
    FOV size divided by the pixel count is only a fallback.  Tiny differences
    between the two are ignored on purpose.  For a downsampled grid pass the
    ``downsample`` factor (and the downsampled pixel count in ``pixels``).
    """
    if step_um:
        return float(step_um) / 1000.0 * max(1, int(downsample))
    if length_mm and pixels:
        return float(length_mm) / float(pixels)
    return 1e-3


def build_layout(positions, fw_px, fh_px, mm_per_px_x, mm_per_px_y,
                 mirror=True, feather=True, missing=()):
    """Place every tile of one mosaic.

    ``positions`` are ``(x_mm, y_mm)`` pairs **in paste order** (the cross-fade
    assumes the tiles before the current one are already written).
    ``fw_px`` / ``fh_px`` is the tile size in the target pixel grid; the tile is
    never cropped (the overlap is blended, not trimmed).  ``missing`` lists the
    paste orders that will not be written (a tile that was not acquired): their
    neighbours must not fade against them, otherwise the blend would mix the
    unwritten canvas fill into the strip.  Every overlap is ramped on both of its
    tiles, so the fade weights of the two tiles add up to 1 across the overlap band
    (see :func:`blend_paste`); the tile pasted first keeps its full level because
    its own weight sum there is still 0.
    """
    points = [(float(item[0]), float(item[1])) for item in positions]
    if not points:
        raise ValueError("build_layout needs at least one tile position")
    missing_orders = {int(order) for order in missing}
    fw_px = max(1, int(fw_px))
    fh_px = max(1, int(fh_px))
    mm_per_px_x = float(mm_per_px_x) or 1e-3
    mm_per_px_y = float(mm_per_px_y) or 1e-3
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    origin_x, origin_y = min(xs), min(ys)
    unique_x, measured_pitch_x = cluster_positions(xs)
    unique_y, measured_pitch_y = cluster_positions(ys)
    pitch_x_mm = measured_pitch_x or float(fw_px) * mm_per_px_x
    pitch_y_mm = measured_pitch_y or float(fh_px) * mm_per_px_y
    pitch_x_px = max(1, int(round(pitch_x_mm / mm_per_px_x)))
    pitch_y_px = max(1, int(round(pitch_y_mm / mm_per_px_y)))
    width_px = int(round((max(xs) - origin_x) / mm_per_px_x)) + fw_px
    height_px = int(round((max(ys) - origin_y) / mm_per_px_y)) + fh_px

    layout = MosaicLayout(
        fw_px=fw_px, fh_px=fh_px, width_px=width_px, height_px=height_px,
        mm_per_px_x=mm_per_px_x, mm_per_px_y=mm_per_px_y,
        origin_x_mm=origin_x, origin_y_mm=origin_y,
        pitch_x_mm=float(pitch_x_mm), pitch_y_mm=float(pitch_y_mm),
        pitch_x_px=pitch_x_px, pitch_y_px=pitch_y_px,
        mirror=bool(mirror), unique_x=unique_x, unique_y=unique_y, placements=[],
        missing=set(missing_orders),
    )

    seen = {}
    for order, (x_mm, y_mm) in enumerate(points):
        off_x = int(round((x_mm - origin_x) / mm_per_px_x))
        off_y = int(round((y_mm - origin_y) / mm_per_px_y))
        # Mirror: larger stage X goes left, larger stage Y goes top (unchanged
        # from the original app layout, so existing mosaics keep their facing).
        x1 = width_px - fw_px - off_x if mirror else off_x
        y1 = height_px - fh_px - off_y if mirror else off_y
        row = layout._index_of(unique_y, y_mm)
        col = layout._index_of(unique_x, x_mm)
        if (row, col) in seen:
            if order not in missing_orders and seen[(row, col)] not in missing_orders:
                layout.duplicates.append((seen[(row, col)], order))
        seen[(row, col)] = order
        layout.placements.append(
            TilePaste(order=order, row=row, col=col, x1=x1, y1=y1, w=fw_px, h=fh_px)
        )

    if feather:
        by_cell = {
            (paste.row, paste.col): paste
            for paste in layout.placements
            if paste.order not in missing_orders
        }
        # Which way "left"/"above" is depends on the mirror: with the mirror the
        # larger stage coordinate lands at the smaller pixel coordinate.
        left_col = 1 if mirror else -1
        top_row = 1 if mirror else -1
        sides = (
            ("fade_left", 0, left_col), ("fade_right", 0, -left_col),
            ("fade_top", top_row, 0), ("fade_bottom", -top_row, 0),
        )
        # An overlap is ramped on *both* of its tiles: the two ramps add up to 1
        # across the band (same raised cosine over the same pixel range), so the
        # weight sum of a pixel in the band is exactly 1 and the pair reduces to
        # its plain w / 1-w cross-fade.  The tile written first has nothing under it
        # yet -- its weight sum there is still 0 -- so its own ramp cannot darken
        # it, and a neighbour that is never acquired (or is marked missing) simply
        # leaves its band at full level.
        for paste in layout.placements:
            for attribute, delta_row, delta_col in sides:
                neighbour = by_cell.get((paste.row + delta_row, paste.col + delta_col))
                if neighbour is None:
                    continue    # nothing there: no ramp on that side
                if attribute == "fade_left":
                    overlap = neighbour.x2 - paste.x1
                    limit = paste.w
                elif attribute == "fade_right":
                    overlap = paste.x2 - neighbour.x1
                    limit = paste.w
                elif attribute == "fade_top":
                    overlap = neighbour.y2 - paste.y1
                    limit = paste.h
                else:
                    overlap = paste.y2 - neighbour.y1
                    limit = paste.h
                overlap = int(min(overlap, limit))
                if overlap > 0:
                    setattr(paste, attribute, overlap)
    return layout


def blend_paste(destination, source, paste, block_rows=BLEND_BLOCK_ROWS, weights=None,
                advance=True):
    """Write ``source`` into ``destination`` at ``paste``, blending the overlap.

    ``destination`` is ``[Y, X]`` or ``[Y, X, ...]`` (Z slices / RGB channels keep
    their layout); ``source`` has exactly the tile size.  Where the tile covers
    ground that an earlier tile already wrote, the two are cross-faded (raised
    cosine ramps, see :meth:`TilePaste.weight`) instead of the later one replacing
    the earlier one; the fade is computed in blocks of rows so even a
    full-resolution tile volume stays light in RAM.

    ``weights`` (optional) is :func:`new_weight_map` for this mosaic: the map holds
    the sum of the fade weights written so far, so the blend stays a **weighted
    average** ``dest = sum(w_i * I_i) / sum(w_i)``, updated incrementally as
    ``dest = dest*(1-g) + source*g`` with ``g = w / (S_old + w)``.  The effective
    weights ``w_i / sum(w_j)`` therefore always add up to 1, and where the ramps of
    two neighbours add up to 1 (a plain two-tile overlap, ``S_old + w == 1``) the
    result is exactly the ``w`` / ``1-w`` cross-fade of those two tiles.  Without
    the map, a pixel that is only touched by the faded part of a single tile (its
    neighbour was never written, a tile is missing or was aborted early) keeps only
    ``w`` of its brightness -- a visible dark strip along the overlap; the map
    avoids it because a pixel with nothing else in it is taken at full level
    (``S_old == 0`` gives ``g == 1``).

    A tile may be counted in ``weights`` only once, so a tile that is pasted again
    and again while it is still being acquired passes ``advance=False``: the map is
    then read but not updated, and only the closing paste of that tile advances it.
    """
    region = destination[paste.y1:paste.y2, paste.x1:paste.x2]
    height = int(region.shape[0])
    width = int(region.shape[1])
    weight = paste.weight()
    if weight is not None:
        # Stay inside the canvas even if the paste box runs off it (the layout
        # normally prevents that, but a hand-built paste must not corrupt data).
        weight = weight[:height, :width]
        source = source[:height, :width]
    if weight is None:
        region[...] = source
        if weights is not None and advance:
            weights[paste.y1:paste.y2, paste.x1:paste.x2] = 1.0
        return
    block_rows = max(1, int(block_rows))
    weight_map = None
    if weights is not None:
        weight_map = weights[paste.y1:paste.y2, paste.x1:paste.x2]
        if weight_map.shape[:2] != region.shape[:2]:
            weight_map = None       # geometry mismatch: fall back to the plain mix
    for start in range(0, region.shape[0], block_rows):
        stop = min(region.shape[0], start + block_rows)
        fade = weight[start:stop]
        if weight_map is None:
            gain = fade
        else:
            total = weight_map[start:stop] + fade
            gain = np.divide(
                fade, total, out=np.ones_like(total), where=total > 1e-6,
            )
            if advance:
                weight_map[start:stop] = total
        if region.ndim > 2:
            gain = gain.reshape(gain.shape + (1,) * (region.ndim - 2))
        block = region[start:stop].astype(np.float32)
        block *= (1.0 - gain)
        block += gain * source[start:stop]
        if np.issubdtype(region.dtype, np.integer):
            limits = np.iinfo(region.dtype)
            block = np.clip(np.rint(block), limits.min, limits.max)
        region[start:stop] = block.astype(region.dtype, copy=False)



def ui_steps_um(ui):
    """``(x_step_um, y_step_um)`` read from a Qt UI object, else ``(None, None)``.

    The scan sampling steps are the authoritative pixel size; a UI without the
    spinboxes (offline scripts, test stubs) yields None so the caller falls back
    to the FOV size divided by the pixel count.
    """
    try:
        x_step = float(ui.XStepSize.value()) if hasattr(ui, "XStepSize") else None
        y_step = float(ui.YStepSize.value()) if hasattr(ui, "YStepSize") else None
    except Exception:
        return None, None
    return (x_step or None), (y_step or None)


def layout_for_ui(ui, positions, fw_px, fh_px, fw_mm, fh_mm, downsample=1,
                  mirror=True, feather=True, missing=()):
    """``build_layout`` with the pixel size taken from the UI scan steps."""
    x_step_um, y_step_um = ui_steps_um(ui)
    return build_layout(
        positions,
        fw_px,
        fh_px,
        pixel_size_mm(x_step_um, fw_mm, fw_px, downsample),
        pixel_size_mm(y_step_um, fh_mm, fh_px, downsample),
        mirror=mirror,
        feather=feather,
        missing=missing,
    )

