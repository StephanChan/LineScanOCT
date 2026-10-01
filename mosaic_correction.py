# -*- coding: utf-8 -*-
"""Offline mosaic background correction helpers (percentile floor + BaSiC fields).

numpy + tifffile + scipy.fft (the DCT smooth basis of the BaSiC fields), no Qt or
hardware imports, so the same code can later be
reused by the offline stitchers (``standalone_mosaic_stitch.py`` and
``DynamicPostprocessing.write_stitched_static_outputs``).

Conventions
-----------
* A tile is a ``[Y, X, Z]`` volume; one TIFF page per Y row.
* AMP+PHASE tiles are saved by ``ThreadDnS.save_data()`` as
  ``[..., 2 * Z]`` with the amplitude in the first half and the phase in the
  second half. Only the amplitude half is background corrected.
"""

import numpy as np
import tifffile as TIFF
from scipy.fft import dctn, idctn

BACKGROUND_PERCENTILE = 5.0
BACKGROUND_Z_CHUNK = 64


def read_volume_stack(path):
    """Read a tile TIFF as one ``[Y, X, Z]`` volume."""
    with TIFF.TiffFile(path) as tif:
        if len(tif.pages) <= 1:
            return tif.pages[0].asarray()
        if len(tif.series) == len(tif.pages):
            return np.stack([page.asarray() for page in tif.pages])
        return tif.series[0].asarray()


def write_volume_stack(path, volume):
    """Write a ``[Y, X, Z]`` volume as one TIFF page per Y row."""
    TIFF.imwrite(path, np.asarray(volume), photometric="minisblack", append=False)


def split_amplitude_phase(volume, logical_z=None):
    """Return ``(amplitude, phase)`` for an AMP+PHASE tile (phase None if plain)."""
    if logical_z is None:
        return volume, None
    if int(volume.shape[-1]) != 2 * int(logical_z):
        return volume, None
    z_pixels = int(logical_z)
    return volume[..., :z_pixels], volume[..., z_pixels:]


def combine_amplitude_phase(amplitude, phase):
    """Inverse of :func:`split_amplitude_phase` (keeps the phase half untouched)."""
    if phase is None:
        return amplitude
    out = np.empty(
        amplitude.shape[:-1] + (amplitude.shape[-1] * 2,), dtype=np.float32
    )
    out[..., : amplitude.shape[-1]] = amplitude
    out[..., amplitude.shape[-1]:] = phase
    return out


def percentile_background(volume, percentile=BACKGROUND_PERCENTILE, z_chunk=BACKGROUND_Z_CHUNK):
    """Background map ``[X, Z]`` = Y-direction low percentile of ``volume``.

    Evaluated in Z slabs so the working copy stays small for big tiles.
    """
    z_pixels = int(volume.shape[-1])
    z_chunk = max(1, int(z_chunk))
    background = np.empty(volume.shape[1:], dtype=np.float32)
    for start in range(0, z_pixels, z_chunk):
        stop = min(z_pixels, start + z_chunk)
        background[:, start:stop] = np.percentile(
            volume[..., start:stop], float(percentile), axis=0
        )
    return background


def apply_background(volume, background, clip=True):
    """Subtract a background map from every Y row of ``volume``.

    ``background`` may be a ``[X, Z]`` map (shared by all Y rows) or a full
    ``[Y, X, Z]`` map (per-Y scaled floor).
    """
    background = np.asarray(background, dtype=np.float32)
    if background.ndim == 2:
        background = background[np.newaxis, :, :]
    corrected = volume - background
    if clip:
        np.maximum(corrected, 0.0, out=corrected)
    return corrected


# ---------------------------------------------------------------------------
# step 2: BaSiC fields (dark + flat) estimated from the tile AIPs
# ---------------------------------------------------------------------------
# The AIP (= mean over Z) of every tile is one picture of the fixed pattern: the
# illumination / scan profile and an obstruction in the beam path do not move in
# scanner (tile-local) coordinates, while the sample lies somewhere else in every
# tile.  BaSiC (Peng et al. 2017) describes that stack as
#
#     I_j = F * s_j + B + S_j      F = flat, B = dark, s_j = level of tile j
#
# with a smooth F and B (low-order DCT basis) and a sparse sample S_j, and solves
# it by alternating minimisation.  Only F and B are used here; they are applied
# to every Z slice of every tile as ``(I - B) / F``.
BASIC_TERMS = (64, 64)     # (ky, kx) DCT terms of flat/dark.  25 terms cannot
                           #  follow the ~8-row settling ramp at the tile end;
                           #  64 keeps the fields smooth but resolves that edge
BASIC_ITER = 20            # refinement rounds (stop early when not improving)
BASIC_DARK_PERCENTILE = 10.0   # order statistic over the tiles for the dark field
BASIC_LEVEL_PERCENTILE = 20.0  # order statistic over the pixels for a tile level
BASIC_PER_IMAGE = True     # fit one level scalar per tile (the BaSiC s_j)
CORRECTION_CHUNK = 16      # B-lines per chunk of the per-A-line passes

INVALID_ROW_FRACTION = 0.35      # leading Y rows below this fraction of the median
INVALID_ROW_MAX = 8


def smooth_axis(array, axis, width):
    """Edge-padded box smoothing along one axis (no scipy dependency)."""
    array = np.asarray(array, dtype=np.float32)
    width = max(1, int(width))
    if width == 1:
        return array.copy()
    pad_before = (width - 1) // 2
    pad_after = width - 1 - pad_before
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (pad_before, pad_after)
    padded = np.pad(array, pad_width, mode="edge")
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.apply_along_axis(
        lambda vector: np.convolve(vector, kernel, mode="valid"), axis, padded
    ).astype(np.float32)


def smooth2d(array, width_z, width_x):
    """Box smoothing over the two axes of an ``[X, Z]`` map."""
    return smooth_axis(smooth_axis(array, 1, width_x), 0, width_z)


def normalise_map(array):
    """Scale a map so its median is 1 (keeps the tile-to-tile level differences)."""
    array = np.asarray(array, dtype=np.float32)
    level = float(np.median(array))
    if level > 0:
        return array / level
    return np.ones_like(array)


def row_levels(volume):
    """Mean over (X, Z) of every Y row: blank/dummy rows show up as ~0."""
    return np.asarray(volume, dtype=np.float32).mean(axis=(1, 2))


def invalid_leading_rows(volume, fraction=INVALID_ROW_FRACTION, max_rows=INVALID_ROW_MAX):
    """Number of leading Y rows that are blank (dummy frame) instead of data."""
    levels = row_levels(volume)
    if levels.size == 0:
        return 0
    reference = float(np.median(levels))
    if reference <= 0:
        return 0
    count = 0
    while count < min(int(max_rows), levels.size) and levels[count] < fraction * reference:
        count += 1
    return count


def invalid_edge_rows(volume, floor_level, fraction=INVALID_ROW_FRACTION,
                      max_rows=INVALID_ROW_MAX, floor_fraction=0.8):
    """``(leading, trailing)`` Y rows that carry no data.

    A row is treated as invalid when its level is clearly below the tile's own
    row levels *and* below the tile floor, which is how the dummy/settling rows
    at the start and end of a tile show up (they sit under the noise floor, not
    at it).  Rows that are merely empty still sit at the floor and are kept.
    """
    levels = row_levels(volume)
    count = levels.size
    if count == 0:
        return 0, 0
    reference = float(np.median(levels))
    threshold = fraction * reference if reference > 0 else 0.0
    if floor_level and float(floor_level) > 0.0:
        floor_threshold = float(floor_fraction) * float(floor_level)
        threshold = min(threshold, floor_threshold) if threshold > 0 else floor_threshold
    limit = min(int(max_rows), max(0, count // 2))
    leading = 0
    while leading < limit and levels[leading] < threshold:
        leading += 1
    trailing = 0
    while trailing < limit and levels[count - 1 - trailing] < threshold:
        trailing += 1
    return leading, trailing


def fill_invalid_blines(volume, leading=0, trailing=0, mode="edge"):
    """Blank/dummy edge rows: ``edge`` copies the nearest valid row, ``drop`` cuts."""
    leading = max(0, int(leading))
    trailing = max(0, int(trailing))
    if leading == 0 and trailing == 0:
        return volume
    if mode == "drop":
        stop = volume.shape[0] - trailing
        return volume[leading:stop] if stop > leading else volume
    if mode == "edge":
        volume = np.asarray(volume, dtype=np.float32).copy()
        if leading and leading < volume.shape[0]:
            volume[:leading] = volume[leading]
        if trailing and trailing < volume.shape[0]:
            volume[volume.shape[0] - trailing:] = volume[volume.shape[0] - trailing - 1]
    return volume


# ---------------------------------------------------------------------------
# BaSiC fields from the tile AIPs (dark + flat)
# ---------------------------------------------------------------------------


def tile_aip(volume, percentile=None):
    """``[Y, X, Z]`` -> ``[Y, X]`` AIP (mean over Z, or one Z percentile)."""
    volume = np.asarray(volume, dtype=np.float32)
    if percentile is None:
        return volume.mean(axis=2)
    return np.percentile(volume, float(percentile), axis=2).astype(np.float32)


def _basis_filter(array, terms):
    """Keep only the lowest ``terms`` DCT coefficients of a 2-D map.

    This is the "shading is smooth" constraint of BaSiC: flat and dark live in a
    low-order basis, so the sample structure cannot leak into them.
    """
    ky = max(1, min(int(terms[0]), array.shape[0]))
    kx = max(1, min(int(terms[1]), array.shape[1]))
    coefficients = dctn(np.asarray(array, dtype=np.float64), type=2, norm="ortho")
    keep = np.zeros(coefficients.shape, dtype=bool)
    keep[:ky, :kx] = True
    smooth = idctn(np.where(keep, coefficients, 0.0), type=2, norm="ortho")
    return smooth.astype(np.float32)


def basic_fields(stack, terms=BASIC_TERMS, n_iter=BASIC_ITER,
                 dark_percentile=BASIC_DARK_PERCENTILE,
                 level_percentile=BASIC_LEVEL_PERCENTILE,
                 per_image=BASIC_PER_IMAGE, verbose=False):
    """BaSiC flat field and dark field of the AIP stack ``[n, Y, X]``.

    Returns ``(flat, dark, scale)``:

    * ``flat[Y, X]``  the fixed bright/dark pattern, median 1,
    * ``dark[Y, X]``  the fixed additive background,
    * ``scale[n]``    the level BaSiC fitted for every tile.

    The model is BaSiC's ``I_j = s_j * F + B + S_j``: one level per tile, one
    flat and one dark field shared by all tiles.  The sample ``S_j`` is sparse in
    tile-local coordinates (a cell sits somewhere else in every tile), which is
    what makes F and B separable; here that prior is applied with order
    statistics instead of an L1 weight -- the median over tiles for F, a low
    percentile over tiles for B -- so there is no threshold to tune:

    * ``F``: median over the tiles of ``(I - B) / s_j`` (a bright cell is a
      minority at any tile-local pixel, so the median is background),
    * ``B``: low percentile over the tiles of ``I - s_j * F`` (the sample only
      *adds* light, so the darkest tiles at a pixel show the background),
    * ``s_j``: low percentile over the pixels of ``(I - B) / F``, i.e. the
      background light of that tile (the pixel *median* would follow the
      sample and inflate the level of the sample-rich tiles).

    Flat and dark are projected onto the low-order DCT basis every round, which
    is the "the shading is smooth" constraint of BaSiC.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError("basic_fields expects an [n, Y, X] stack")
    count, height, width = stack.shape
    if count < 3:
        raise ValueError("basic_fields needs at least three tiles")

    def _score(flat_map, dark_map, level):
        """Median absolute residual of the model (robust, the sample is a minority)."""
        model = level[:, None, None] * flat_map[None, :, :] + dark_map[None, :, :]
        return float(np.median(np.abs(stack - model)))

    def _renormalise(array, level):
        """Smooth a field and move its median into the tile levels (F * s fixed)."""
        smooth = _basis_filter(array, terms)
        factor = max(float(np.median(smooth)), 1e-6)
        return np.maximum(smooth / factor, 1e-3), level * factor

    # --- robust pass: always stable, no threshold to tune -------------------
    # a tile level is the *background* light of that tile: a low percentile over
    # its pixels, because the pixel median follows the sample instead
    scale = np.maximum(
        np.percentile(stack.reshape(count, -1), float(level_percentile), axis=1), 1e-6
    ).astype(np.float32)
    flat, scale = _renormalise(np.median(stack / scale[:, None, None], axis=0), scale)
    flat = flat.astype(np.float32)
    # the sample only *adds* light, so the darkest tiles at a pixel show B
    dark = _basis_filter(
        np.percentile(stack - scale[:, None, None] * flat[None, :, :],
                      float(dark_percentile), axis=0),
        terms,
    )
    best = (flat, dark, scale, _score(flat, dark, scale))

    # --- refinement rounds, accepted only while the model keeps improving ---
    first_range = float(flat.max() / max(float(flat.min()), 1e-6))
    first_level = float(np.median(scale))
    for step in range(max(0, int(n_iter) - 1)):
        flat_fit, scale_fit = _renormalise(
            np.median((stack - best[1][None, :, :]) / best[2][:, None, None], axis=0), best[2]
        )
        if per_image:
            scale_fit = np.maximum(
                np.percentile(
                    (stack - best[1][None, :, :]) / flat_fit[None, :, :],
                    float(level_percentile), axis=(1, 2),
                ),
                1e-6,
            ).astype(np.float32)
        dark_fit = _basis_filter(
            np.percentile(
                stack - scale_fit[:, None, None] * flat_fit[None, :, :],
                float(dark_percentile), axis=0,
            ),
            terms,
        )
        score = _score(flat_fit, dark_fit, scale_fit)
        if not np.isfinite(score) or score > best[3]:
            break                       # keep the last stable fields, never drift
        # a smaller residual alone is not enough: a wider flat field can also
        # "explain" the sample, so reject a round that runs away from the robust
        # pass (this is what keeps the fields physical on sample-rich scans)
        if float(flat_fit.max() / max(float(flat_fit.min()), 1e-6)) > 2.0 * first_range:
            break
        if float(np.median(scale_fit)) < 0.5 * first_level:
            break
        best = (
            flat_fit.astype(np.float32), dark_fit.astype(np.float32),
            scale_fit.astype(np.float32), score,
        )
        if verbose:
            print(
                "    BaSiC {0:>2}/{1}: flat {2:.3f}..{3:.3f}   dark {4:.3f}..{5:.3f}"
                "   level {6:.3f}..{7:.3f}   residual {8:.3f}".format(
                    step + 1, int(n_iter), float(best[0].min()), float(best[0].max()),
                    float(best[1].min()), float(best[1].max()),
                    float(best[2].min()), float(best[2].max()), score,
                )
            )

    return best[0], best[1], best[2]


def apply_basic_fields(volume, flat, dark, clip=False, y_chunk=CORRECTION_CHUNK):
    """Apply one pair of BaSiC fields to a whole volume: ``(volume - dark) / flat``.

    ``flat`` / ``dark`` are ``[Y, X]`` 2-D fields (one field for every depth, as
    estimated from the tile AIPs).  For per-depth fields use
    :func:`apply_basic_fields_per_depth`.
    """
    flat = np.asarray(flat, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    if flat.ndim == 3 or dark.ndim == 3:
        raise ValueError("per-depth fields need apply_basic_fields_per_depth()")
    out = np.array(volume, dtype=np.float32, copy=True)
    if flat.shape != out.shape[:2] or dark.shape != out.shape[:2]:
        raise ValueError(
            "field shape {0}/{1} does not match the volume's [Y, X] {2}".format(
                tuple(dark.shape), tuple(flat.shape), tuple(out.shape[:2])
            )
        )
    chunk = max(1, int(y_chunk))
    for start in range(0, out.shape[0], chunk):
        stop = min(out.shape[0], start + chunk)
        block = out[start:stop]
        block -= dark[start:stop, :, np.newaxis]
        block /= np.maximum(flat[start:stop, :, np.newaxis], 1e-3)
    if clip:
        np.maximum(out, 0.0, out=out)
    return out


def expand_field_planes(field, z_planes, z_count):
    """Expand ``[n_planes, Y, X]`` fields to ``[Z, Y, X]`` by linear interpolation.

    ``z_planes`` are the depths the fields were fitted at; every depth in between
    uses a linear blend of its two neighbours, so the correction stays smooth in Z
    and no depth is corrected with a field from a far-away plane.
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("expand_field_planes expects [Z, Y, X] fields")
    planes = np.asarray(z_planes, dtype=np.float64).ravel()
    if planes.size != field.shape[0]:
        raise ValueError("z_planes does not match the number of field planes")
    expanded = np.empty((int(z_count),) + field.shape[1:], dtype=np.float32)
    for z in range(int(z_count)):
        index = int(np.clip(np.searchsorted(planes, z, side="right") - 1, 0, planes.size - 1))
        if index >= planes.size - 1:
            expanded[z] = field[-1]
            continue
        span = planes[index + 1] - planes[index]
        weight = 0.0 if span <= 0 else float((z - planes[index]) / span)
        expanded[z] = (1.0 - weight) * field[index] + weight * field[index + 1]
    return expanded


def apply_basic_fields_per_depth(volume, flat, dark, z_planes=None, clip=False,
                                 z_chunk=16):
    """``(volume[z] - dark[z]) / flat[z]``, one field pair per depth.

    ``flat`` / ``dark`` are ``[n_planes, Y, X]`` fitted on ``z_planes`` (for example
    every 8th depth plane); the fields of all other depths are interpolated, so the
    whole volume gets a depth-dependent correction instead of one AIP field.
    Depths are processed in small blocks, so the temporary RAM stays a slice.
    """
    flat = np.asarray(flat, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    if flat.ndim == 2 and dark.ndim == 2:
        return apply_basic_fields(volume, flat, dark, clip=clip)
    if flat.ndim != 3 or dark.ndim != 3:
        raise ValueError("per-depth fields must both be [Z, Y, X]")
    z_count = int(volume.shape[2])
    if z_planes is None or len(z_planes) != flat.shape[0]:
        z_planes = np.linspace(0, max(0, z_count - 1), flat.shape[0])
    out = np.array(volume, dtype=np.float32, copy=True)
    block = max(1, int(z_chunk))
    for start in range(0, z_count, block):
        stop = min(z_count, start + block)
        # (nz, Y, X) fields -> (Y, X, nz) so they broadcast against out[:, :, z]
        flat_block = expand_field_planes(flat, z_planes, stop)[start:stop].transpose(1, 2, 0)
        dark_block = expand_field_planes(dark, z_planes, stop)[start:stop].transpose(1, 2, 0)
        chunk = out[:, :, start:stop]
        chunk -= dark_block
        chunk /= flat_block
    if clip:
        np.maximum(out, 0.0, out=out)
    return out


def apply_gain_per_depth(volume, flat, z_planes=None, clip=False, z_chunk=16):
    """Divide a volume by the flat *gain* only: ``volume / flat`` (no dark).

    Used for the dynamic (std / fluctuation) products: a temporal standard
    deviation has no additive camera offset, so only the multiplicative
    illumination / collection non-uniformity applies.  ``flat`` may be ``[Y, X]``
    or ``[n_planes, Y, X]`` (then interpolated along Z, see
    :func:`expand_field_planes`).
    """
    flat = np.asarray(flat, dtype=np.float32)
    if flat.ndim == 2:
        out = np.array(volume, dtype=np.float32, copy=True)
        chunk = max(1, int(CORRECTION_CHUNK))
        for start in range(0, out.shape[0], chunk):
            stop = min(out.shape[0], start + chunk)
            out[start:stop] /= np.maximum(flat[start:stop, :, np.newaxis], 1e-3)
        if clip:
            np.maximum(out, 0.0, out=out)
        return out
    if flat.ndim != 3:
        raise ValueError("the flat gain must be [Y, X] or [Z, Y, X]")
    z_count = int(volume.shape[2])
    if z_planes is None or len(z_planes) != flat.shape[0]:
        z_planes = np.linspace(0, max(0, z_count - 1), flat.shape[0])
    out = np.array(volume, dtype=np.float32, copy=True)
    block = max(1, int(z_chunk))
    for start in range(0, z_count, block):
        stop = min(z_count, start + block)
        flat_block = expand_field_planes(flat, z_planes, stop)[start:stop].transpose(1, 2, 0)
        out[:, :, start:stop] /= np.maximum(flat_block, 1e-3)
    if clip:
        np.maximum(out, 0.0, out=out)
    return out


def apply_calibration_per_depth(volume, dark_volume, flat_volume, clip=False):
    """Apply acquired 3-D calibration volumes: ``(I - dark) / (flat - dark)``.

    ``dark_volume`` (beam blocked) and ``flat_volume`` (uniform reflector) must have
    the same ``[Y, X, Z]`` shape as ``volume``; every depth is corrected with its own
    plane, which is the depth-resolved form of the flat/dark correction.
    """
    out = np.array(volume, dtype=np.float32, copy=True)
    dark = np.asarray(dark_volume, dtype=np.float32)
    flat = np.asarray(flat_volume, dtype=np.float32)
    if dark.shape != out.shape or flat.shape != out.shape:
        raise ValueError(
            "calibration volume shape {0}/{1} does not match the tile {2}".format(
                tuple(dark.shape), tuple(flat.shape), tuple(out.shape)
            )
        )
    out -= dark
    out /= np.maximum(flat - dark, 1e-6)
    if clip:
        np.maximum(out, 0.0, out=out)
    return out

