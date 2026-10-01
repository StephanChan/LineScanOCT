
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import findchangept
import dataclasses
import numpy as np
import traceback
global SCALE
SCALE =1000
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
from scipy import ndimage
from scipy import fft as scipy_fft
# from libtiff import TIFF
import tifffile as TIFF
import time
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION
from HardwareSpecs import TIFF_APPEND_WRITES_DEFAULT
from DataShape import data_shape
from mosaic_geometry import blend_paste, build_layout, new_weight_map, pixel_size_mm, ui_steps_um
import shading_correction

ALINE_MODES = (
    AcqTypes.FINITE_ALINE,
    AcqTypes.CONTINUOUS_ALINE,
)

BLINE_MODES = (
    AcqTypes.FINITE_BLINE,
    AcqTypes.CONTINUOUS_BLINE,
)

CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
    AcqTypes.FAST_VOLUME_CSCAN,
    AcqTypes.TD_ENFACE,
)

SAVE_SAMPLE_TIME_MODES = (
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)
STITCH_MOSAIC_VOLUMES_IN_MEMORY = True
# Mosaic live 2-D HSV / RGB / freq / bandwidth buffers are built and emitted for
# real-time display (all dynamic-capable modes show HSV, never bare std).
STITCH_MOSAIC_DYNAMIC_UI = True

# Dynamic-mode structure-only Y notch -----------------------------------------
# In dynamic C-scan / mosaic acquisitions only one Y line arrives per action, so
# the volume Y notch of ThreadGPU (it needs the whole spatial-Y volume, and only
# runs on non-dynamic paths) cannot be applied there.  Instead the DnS thread
# keeps a per-FOV field volume (complex when the acquisition runs in AMP+PHASE,
# otherwise the mean intensity) and runs the same 1-D ky band-stop once, when the
# FOV is complete, on the *structure* data only: mean-intensity volume, AIP and
# the stitched structure volume that is pasted/saved from it.
# The dynamic maps (std / HSV / frequency / bandwidth) are deliberately left
# untouched - they are computed per line in the GPU thread, before any
# spatial-Y operation is possible.
DYNAMIC_STRUCTURE_Y_NOTCH_ENABLED = True
# Half-band of the notch as a fraction of the Y Nyquist; the UI "HoloBandwidth"
# value is used when present, this is only the fallback (same default as
# ThreadGPU.HOLO_CENTER_BAND_HALF_WIDTH_FRACTION).
DYNAMIC_STRUCTURE_Y_NOTCH_HALF_BAND = 0.25
# Depth planes filtered per FFT block (bounds the temporary RAM of the notch).
DYNAMIC_STRUCTURE_Y_NOTCH_BLOCK_Z = 64

MOSAIC_DISPLAY_MODES = (
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)

DYNAMIC_HUE_FREQUENCY_RANGE_HZ = (0.0, 15.0)
DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ = (0.0, 8.0)
DYNAMIC_VALUE_DYNAMIC_RANGE = (0.0, 500.0)
DYNAMIC_VALUE_GAMMA = 1.0
DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT = 15.0 / 1000.0


def downsample_mosaic_volume(array, scale):
    """Block-mean downsample in the first two axes (Y, X) only.

    Trailing axes (Z depth, and channel dimension for HSV) are kept unchanged,
    so the downsampled stitched volume still has the same depth planes. Used to
    shrink the RAM held by the stitched mosaic volumes using the UI "downsample
    scale" control (scale=1 is a no-op).
    """
    scale = max(1, int(scale))
    if scale == 1 or array.ndim < 2:
        return array
    y, x = array.shape[:2]
    y_c, x_c = y - y % scale, x - x % scale
    if y_c == 0 or x_c == 0:
        return array[::scale, ::scale]
    view = array[:y_c, :x_c]
    new_shape = (y_c // scale, scale, x_c // scale, scale) + tuple(array.shape[2:])
    return view.reshape(new_shape).mean(axis=(1, 3))


class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        # self.SampleDynamic= []
        self.SampleMosaic= []
        self.SampleMosaicVolume = []
        self.AIP = []
        self.XYVolume = []
        self.Dyn = []
        self.DynHSV = []
        self.DynHSVBline = []
        self.DynRGB = []
        self.DynRGBBline = []
        self.DynFreq = []
        self.DynFreqBline = []
        self.DynBandwidth = []
        self.DynBandwidthBline = []
        # self.totalTiles = 0
        self.display_actions = 0
        self.active_tasks = 0
        self.tiff_append_writes = TIFF_APPEND_WRITES_DEFAULT
        self._tiff_initialized_files = set()
        # Open TIFF writers, kept between frames so a stack is not re-opened (and
        # its whole IFD chain re-read) once per written page.  See
        # write_tiff_frame / close_tiff_writers.
        self._tiff_writers = {}
        self.MeanVolume = []
        self.DynamicVolume = []
        self.DynamicHSVVolume = []
        self.DynamicRGBVolume = []
        self.SampleMosaicRGB = []
        self.SampleMosaicHSV = []
        self.SampleMosaicDynamicVolume = []
        self.SampleMosaicHSVVolume = []
        self.SampleMosaicDyn = []
        self.SampleMosaicFreq = []
        self.SampleMosaicBandwidth = []
        self.mosaic_y_pixels = None
        # Stitched-volume X/Y downsample (from the UI "downsample scale" spinbox).
        # The 2D AIP mosaic stays full-resolution; only the 3D stitched volumes
        # (intensity / dynamic / HSV) are downsampled to save RAM.
        self.mosaic_downsample = 1
        self.fw_px_ds = 0
        self.fh_px_ds = 0
        self.mosaic_volume_shape = (0, 0)
        # Shading correction state (see shading_correction): the field lives for the
        # whole session, the accumulator only while the reference sample is scanned.
        self.shading_field = None
        self.shading_accumulator = None
        self.shading_signature = {}
        self.mosaic_structure_volume = None
        # Per-FOV field volume used by the dynamic-mode structure Y notch (see
        # DYNAMIC_STRUCTURE_Y_NOTCH_ENABLED): complex64 when the acquisition runs
        # in AMP+PHASE, plain float32 (mean intensity) otherwise.
        self.MeanVolumeComplex = None
        self._y_notch_info_printed = False

    def reset_dynamic_accumulators(self):
        # Any file still open here is finished (new tile / new scan): close it
        # before the buffers below are dropped.
        self.close_tiff_writers("reset")
        self.AIP = []
        self.XYVolume = []
        self.Dyn = []
        self.DynHSV = []
        self.DynHSVBline = []
        self.DynRGB = []
        self.DynBline = []
        self.DynRGBBline = []
        self.DynFreq = []
        self.DynFreqBline = []
        self.DynBandwidth = []
        self.DynBandwidthBline = []
        self.MeanVolume = []
        self.DynamicVolume = []
        self.DynamicHSVVolume = []
        self.DynamicRGBVolume = []
        self.SampleMosaicRGB = []
        self.SampleMosaicHSV = []
        self.SampleMosaicVolume = []
        self.SampleMosaicDynamicVolume = []
        self.SampleMosaicHSVVolume = []
        self.SampleMosaicDyn = []
        self.SampleMosaicFreq = []
        self.SampleMosaicBandwidth = []
        self.mosaic_y_pixels = None
        # Keep the stitched-volume downsample config from the last Init_Mosaic.
        self.MeanVolumeComplex = None
        # The field stays in the session; the pending fit does not survive a sample.
        self.shading_accumulator = None

    # ------------------------------------------------------------------ Y notch
    def _y_notch_half_band_fraction(self):
        """Half-band of the dynamic structure Y notch (fraction of the Y Nyquist)."""
        widget = getattr(self.ui, "HoloBandwidth", None)
        if widget is not None:
            try:
                return max(0.0, min(0.99, float(widget.value())))
            except Exception:
                pass
        return DYNAMIC_STRUCTURE_Y_NOTCH_HALF_BAND

    def _accumulate_structure_field(self, data, index, shape):
        """Keep the field of one acquired Y line for the end-of-FOV Y notch.

        ``data`` is the per-Y-line stack the GPU thread produced (leading frame
        axis); averaging over those frames mirrors how the intensity ``Bline`` is
        built, so the filtered structure stays aligned with the other accumulators.
        ``shape`` is the per-FOV (Y, X, Z) volume shape.
        """
        if not DYNAMIC_STRUCTURE_Y_NOTCH_ENABLED or index is None:
            return
        if not isinstance(data, np.ndarray) or data.ndim != 3 or data.shape[0] < 1:
            return
        line = np.mean(data, axis=0)
        if tuple(line.shape) != (int(shape[1]), int(shape[2])):
            return
        dtype = np.complex64 if line.dtype.kind == "c" else np.float32
        field = self.MeanVolumeComplex
        if (
            index == 0
            or not isinstance(field, np.ndarray)
            or tuple(field.shape) != tuple(shape)
            or field.dtype != dtype
        ):
            field = np.zeros(shape, dtype=dtype)
            self.MeanVolumeComplex = field
        field[int(index), :, :] = line

    def _apply_y_notch_to_structure(self):
        """1-D ky band-stop (same recipe as ThreadGPU) on the FOV structure volume.

        Called once when the last Y line of a FOV has arrived.  The filtered
        magnitude replaces the mean-intensity volume and the AIP, so the live
        mosaic and the saved mean/AIP/structure data carry the notch; the dynamic
        maps are left exactly as computed per line.
        """
        if not DYNAMIC_STRUCTURE_Y_NOTCH_ENABLED:
            return False
        field = self.MeanVolumeComplex
        if not isinstance(field, np.ndarray) or np.size(field) == 0:
            return False
        rows = int(field.shape[0])
        fraction = self._y_notch_half_band_fraction()
        if rows < 2 or fraction <= 0.0:
            return False
        # |ky| cutoff expressed in FFT bins: ky = 2*pi*f/(N*d) and the Nyquist is
        # pi/d, so |f| > fraction/2 (cycles per sample) is the suppressed band.
        mask = (np.abs(np.fft.fftfreq(rows)) > fraction / 2.0).astype(np.float32)
        filtered = np.empty(field.shape, dtype=np.float32)
        block = max(1, int(DYNAMIC_STRUCTURE_Y_NOTCH_BLOCK_Z))
        for start in range(0, int(field.shape[2]), block):
            stop = min(int(field.shape[2]), start + block)
            # scipy.fft with all cores: same recipe as ThreadGPU's volume Y notch.
            spectrum = scipy_fft.fft(field[:, :, start:stop], axis=0, workers=-1)
            spectrum *= mask[:, None, None]
            filtered[:, :, start:stop] = np.abs(
                scipy_fft.ifft(spectrum, axis=0, workers=-1)
            )
        shape2d = filtered.shape[:2]
        if isinstance(self.MeanVolume, np.ndarray) and self.MeanVolume.shape == filtered.shape:
            self.MeanVolume[...] = filtered
        if isinstance(self.XYVolume, np.ndarray) and self.XYVolume.shape == filtered.shape:
            self.XYVolume[...] = filtered
        if isinstance(self.AIP, np.ndarray) and np.size(self.AIP) > 0 and tuple(self.AIP.shape[:2]) == shape2d:
            z_idx = self.current_z_depth_index(int(filtered.shape[2]))
            self.AIP[...] = filtered[:, :, z_idx]
        self.MeanVolumeComplex = None       # release the field until the next FOV
        if not self._y_notch_info_printed:
            self._y_notch_info_printed = True
            message = (
                "Dynamic structure Y notch: {0} Y lines, |ky| <= {1:.2f} x Nyquist "
                "suppressed (hard band-stop). Structure/AIP/mean only; dynamic maps "
                "are not filtered."
            ).format(rows, fraction)
            print(message)
        return True

    def run(self):
        # self.Dynmax = self.ui.Dynmax.value()
        # self.Dynmin = self.ui.Dynmin.value()
        
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            self.active_tasks += 1
            start=time.time()
            self.current_acq_mode = self.item.acq_mode
            try:
                if self.item.action in (
                    AcqTypes.FINITE_ALINE,
                    AcqTypes.CONTINUOUS_ALINE,
                ):
                    self.display_actions += 1
                    self.Process_aline(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="aline")
                elif self.item.action in (
                    AcqTypes.FINITE_BLINE,
                    AcqTypes.CONTINUOUS_BLINE,
                ):
                    self.Process_bline(self.item.data, self.item.raw, self.item.dynamic, self.current_acq_mode, self.item.gpu_avg_count)
                    self.display_actions += 1
                    self._emit_display(kind="bline")
                elif self.item.action in (
                    AcqTypes.FINITE_CSCAN,
                    AcqTypes.CONTINUOUS_CSCAN,
                    AcqTypes.FAST_VOLUME_CSCAN,
                    AcqTypes.TD_ENFACE,
                ):
                    self.display_actions += 1
                    # Dynamic results (HSV; the std is only the HSV value
                    # channel) are produced exclusively by the realtime path,
                    # which requires BOTH the DOCT checkbox (DynCheckBox) and the
                    # realtime-dynamic checkbox (RealtimeDynCheckBox) to be on.
                    # If realtime is off, no dynamic result is computed, saved,
                    # or displayed at all - the frame is treated as static.
                    if self.realtime_cscan_dynamic_enabled(self.current_acq_mode, self.item.dynamic):
                        self.Process_Cscan_RealtimeDynamic(
                            self.item.data,
                            self.item.dynamic,
                            self.current_acq_mode,
                            self.item.gpu_avg_count,
                        )
                    else:
                        self.Process_Cscan(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="cscan")
                    
                elif self.item.action == DnSActions.PROCESS_MOSAIC:
                    self.Process_Mosaic(self.item.data, self.item.raw, self.item.context, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="mosaic")
                elif self.item.action == DnSActions.RETURN_MOSAIC:
                    self.Return_mosaic()
                elif self.item.action == DnSActions.CLEAR:
                    self.reset_dynamic_accumulators()
                elif self.item.action == DnSActions.DISPLAY_COUNTS:
                    self.print_display_counts(self.item.context)

                elif self.item.action == DnSActions.AGAR_TILE:
                    self.SurfFilename()
                elif self.item.action == DnSActions.WRITE_AGAR:
                    self.WriteAgar(self.item.data, self.item.context)
                elif self.item.action == DnSActions.INIT_MOSAIC:
                    self.reset_dynamic_accumulators()
                    self.Init_Mosaic(self.item.context)
                elif self.item.action == DnSActions.SHADING_FIT:
                    # Fit (or report) the session flat/dark field; the result is
                    # handed back through the action's context dictionary.
                    try:
                        if isinstance(self.item.context, dict):
                            self.item.context.update(self.shading_fit())
                    except Exception as error:
                        print("Shading correction fit failed: {0}".format(error))
                        self.emit_status("Shading correction fit failed.")
                elif self.item.action == DnSActions.SAVE_MOSAIC:
                    self.Save_mosaic()
                else:
                    message = f"Unknown display/save command: {self.item.action}"
                    print(message)
                    self.emit_status(message)
                    # self.ui.PrintOut.append(message)
                if time.time()-start>2.0:
                    print('time for DnS:',round(time.time()-start,3))
            except Exception as error:
                message = "Display/save processing failed. This item was skipped."
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                print(traceback.format_exc())
            finally:
                self.active_tasks = max(0, self.active_tasks - 1)
            self.item = self.queue.get()
            
        self.emit_status("Display/save thread exited.")

    def is_idle(self):
        return self.queue.qsize() == 0 and self.active_tasks == 0

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def current_dynamic_enabled(self):
        return self.ui.DynCheckBox.isChecked()

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_aline_avg(self):
        return max(1, int(self.ui.AlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_z_depth_index(self, z_pixels):
        if z_pixels <= 0:
            return 0
        if not hasattr(self.ui, "ZDepthBar"):
            return 0
        return max(0, min(int(self.ui.ZDepthBar.value()), int(z_pixels) - 1))

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def realtime_mosaic_dynamic_enabled(self, acq_mode, dynamic):
        return (
            self.current_dynamic_enabled()
            and acq_mode in SAVE_SAMPLE_TIME_MODES
            and self.has_dynamic_data(dynamic)
        )

    def realtime_cscan_dynamic_enabled(self, acq_mode, dynamic):
        return (
            self.current_dynamic_enabled()
            and acq_mode in CSCAN_MODES
            and self.ui.RealtimeDynCheckBox.isChecked()
            and self.has_dynamic_data(dynamic)
        )

    def write_stack_tiff(self, filename, stack, count=None, close_after=False):
        """Write ``stack`` as pages of ``filename``.

        The writer stays open between calls (``close_after=False``) so a stack
        written line by line (one Y line at a time in dynamic mode) does not have
        to re-open the file and re-read its page chain for every single frame;
        pass ``close_after=True`` for acquisitions whose file is complete after
        this call, and see :meth:`close_tiff_writers` for the tile-level closes.
        """
        frame_count = int(stack.shape[0]) if count is None else int(count)
        for ii in range(frame_count):
            self.write_tiff_frame(filename, stack[ii], append_if_exists=True)
        if close_after:
            self._close_tiff_writer(str(Path(filename).resolve()))

    @staticmethod
    def display_data(data):
        if isinstance(data, np.ndarray) and data.dtype.kind == 'c':
            return np.abs(data)
        return data

    @staticmethod
    def dynamic_std_data(dynamic):
        if isinstance(dynamic, dict):
            if "hsv" in dynamic:
                return dynamic["hsv"][..., 2]
            return dynamic.get("dynamic_std", [])
        return dynamic

    @staticmethod
    def dynamic_hsv_data(dynamic):
        if isinstance(dynamic, dict):
            return dynamic.get("hsv", [])
        return []

    @staticmethod
    def dynamic_frequency_data(dynamic):
        if isinstance(dynamic, dict):
            if "hsv" in dynamic:
                return dynamic["hsv"][..., 0]
            return dynamic.get("mean_frequency_hz", [])
        return []

    @staticmethod
    def dynamic_bandwidth_data(dynamic):
        if isinstance(dynamic, dict):
            if "hsv" in dynamic:
                return dynamic["hsv"][..., 1]
            return dynamic.get("bandwidth_hz", [])
        return []

    @staticmethod
    def has_dynamic_data(dynamic):
        if isinstance(dynamic, dict):
            if "hsv" in dynamic:
                return np.size(dynamic.get("hsv", [])) > 0
            return np.size(dynamic.get("dynamic_std", [])) > 0
        return np.size(dynamic) > 0

    @staticmethod
    def normalize_dynamic_channel(image, value_range, gamma=1.0):
        low_value, high_value = float(value_range[0]), float(value_range[1])
        if high_value <= low_value:
            raise ValueError(f"Invalid dynamic HSV normalization range: {value_range}")
        normalized = (np.asarray(image, dtype=np.float32) - low_value) / (high_value - low_value)
        normalized = np.clip(normalized, 0.0, 1.0)
        gamma = float(gamma)
        if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
            normalized = normalized ** (1.0 / gamma)
        return normalized

    @staticmethod
    def hsv_to_rgb_array(hue, saturation, value):
        hue = np.mod(np.asarray(hue, dtype=np.float32), 1.0)
        saturation = np.clip(np.asarray(saturation, dtype=np.float32), 0.0, 1.0)
        value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

        h6 = hue * 6.0
        i = np.floor(h6).astype(np.int32)
        f = h6 - i.astype(np.float32)
        p = value * (1.0 - saturation)
        q = value * (1.0 - saturation * f)
        t = value * (1.0 - saturation * (1.0 - f))
        i_mod = np.mod(i, 6)

        rgb = np.empty(hue.shape + (3,), dtype=np.float32)
        masks = [
            (i_mod == 0, value, t, p),
            (i_mod == 1, q, value, p),
            (i_mod == 2, p, value, t),
            (i_mod == 3, p, q, value),
            (i_mod == 4, t, p, value),
            (i_mod == 5, value, p, q),
        ]
        for mask, red, green, blue in masks:
            rgb[..., 0][mask] = red[mask]
            rgb[..., 1][mask] = green[mask]
            rgb[..., 2][mask] = blue[mask]
        return np.ascontiguousarray(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8))

    @classmethod
    def dynamic_hsv_to_rgb(cls, hsv, hue_range=None):
        hsv = np.asarray(hsv, dtype=np.float32)
        if hsv.ndim < 3 or hsv.shape[-1] != 3:
            raise ValueError(f"Dynamic HSV source must have last dimension 3, got {hsv.shape}")
        if hue_range is None:
            hue_range = DYNAMIC_HUE_FREQUENCY_RANGE_HZ
        hue = cls.normalize_dynamic_channel(hsv[..., 0], hue_range)
        saturation = cls.normalize_dynamic_channel(hsv[..., 1], DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ)
        value = cls.normalize_dynamic_channel(
            hsv[..., 2],
            DYNAMIC_VALUE_DYNAMIC_RANGE,
            gamma=DYNAMIC_VALUE_GAMMA,
        )
        return cls.hsv_to_rgb_array(hue, saturation, value)

    def current_save_hue_frequency_range_hz(self):
        return (
            float(self.ui.XZmin.value()) * DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
            float(self.ui.XZmax.value()) * DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
        )

    def dynamic_hsv_to_saved_rgb(self, hsv):
        return self.dynamic_hsv_to_rgb(hsv, hue_range=self.current_save_hue_frequency_range_hz())

    @staticmethod
    def save_data(data):
        if not (isinstance(data, np.ndarray) and data.dtype.kind == 'c'):
            return data
        z_pixels = data.shape[-1]
        interleaved = np.empty(data.shape[:-1] + (z_pixels * 2,), dtype=np.float32)
        interleaved[..., :z_pixels] = np.abs(data).astype(np.float32, copy=False)
        interleaved[..., z_pixels:] = np.angle(data).astype(np.float32, copy=False)
        return interleaved

    def reset_tiff_output(self, filename):
        path = Path(filename)
        self._close_tiff_writer(str(path.resolve()))
        try:
            if path.exists():
                path.unlink()
        except OSError as error:
            message = f"Failed to reset TIFF output {filename}: {error}"
            print(message)
            self.emit_status(message)
        self._tiff_initialized_files.discard(str(path.resolve()))

    def _close_tiff_writer(self, resolved):
        """Close (and forget) the open writer of one output file."""
        writer = self._tiff_writers.pop(resolved, None)
        if writer is None:
            return
        try:
            writer.close()
        except Exception as error:
            print(f"Warning: closing TIFF output {resolved} failed: {error}")

    def close_tiff_writers(self, reason=""):
        """Close every output file left open (tile/scan finished).

        Frames are written through a writer that stays open, so this must be
        called when a file is complete: at the end of a tile, before a new scan
        resets the buffers, and when a file is re-created.
        """
        for resolved in list(self._tiff_writers):
            self._close_tiff_writer(resolved)

    def write_tiff_frame(self, filename, image, append_if_exists=True):
        path = Path(filename)
        resolved = str(path.resolve())
        if self.tiff_append_writes:
            TIFF.imwrite(filename, image, append=append_if_exists)
            if append_if_exists:
                self._tiff_initialized_files.add(resolved)
            return

        if not append_if_exists:
            # One-shot overwrite (e.g. the B-line dynamic frame): a fresh file
            # with a single page, exactly like the old imwrite(append=False).
            self.reset_tiff_output(filename)
            with TIFF.TiffWriter(filename, append=False) as writer:
                writer.write(np.asarray(image))
            return

        writer = self._tiff_writers.get(resolved)
        if writer is None:
            if resolved in self._tiff_initialized_files:
                # The file was started earlier (e.g. stacked B-line frames) and
                # its writer was closed in between: continue appending to it.
                writer = TIFF.TiffWriter(filename, append=True)
            else:
                # First frame of this file: drop a stale file, then keep the
                # writer open for the following frames instead of re-opening
                # (and re-reading the page chain) once per page.
                self.reset_tiff_output(filename)
                writer = TIFF.TiffWriter(filename, append=False)
            self._tiff_writers[resolved] = writer
        writer.write(np.asarray(image))
        self._tiff_initialized_files.add(resolved)

    def _emit_display(self, kind: str):
        """
        Emit display payloads to GUI thread via ui_bridge.
        This thread must NOT create QPixmap or touch widgets for rendering.
        """
        bridge = getattr(self, "ui_bridge", None)
        if bridge is None:
            return

        acq_mode = self.current_acq_mode
        use_realtime_dynamic = self.current_dynamic_enabled() and self.ui.RealtimeDynCheckBox.isChecked()

        if kind == "aline" and hasattr(self, "Aline") and np.size(self.Aline) > 0:
            bridge.aline_ready.emit({"mode": acq_mode, "aline": np.array(self.Aline, copy=True)})

        if kind == "bline" and hasattr(self, "Bline") and np.size(self.Bline) > 0:
            rgb = None
            hsv = None
            freq = None
            bandwidth = None
            value = None
            if use_realtime_dynamic and hasattr(self, "DynRGBBline") and np.size(self.DynRGBBline) > 0:
                rgb = np.array(self.DynRGBBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynHSVBline") and np.size(self.DynHSVBline) > 0:
                hsv = np.array(self.DynHSVBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynFreqBline") and np.size(self.DynFreqBline) > 0:
                freq = np.array(self.DynFreqBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBandwidthBline") and np.size(self.DynBandwidthBline) > 0:
                bandwidth = np.array(self.DynBandwidthBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                value = np.array(self.DynBline, copy=True)
            bridge.bline_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "rgb": rgb,
                    "hsv": hsv,
                    "freq": freq,
                    "bandwidth": bandwidth,
                    "value": value,
                }
            )

        if (
            kind == "cscan"
            and hasattr(self, "Bline")
            and hasattr(self, "AIP")
            and np.size(self.Bline) > 0
            and np.size(self.AIP) > 0
        ):
            rgbb = None
            rgb = None
            hsvb = None
            hsv = None
            freqb = None
            bandwidthb = None
            valueb = None
            freq = None
            bandwidth = None
            value = None
            if use_realtime_dynamic and hasattr(self, "DynRGBBline") and np.size(self.DynRGBBline) > 0:
                rgbb = np.array(self.DynRGBBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynRGB") and np.size(self.DynRGB) > 0:
                rgb = np.array(self.DynRGB, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynHSVBline") and np.size(self.DynHSVBline) > 0:
                hsvb = np.array(self.DynHSVBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynHSV") and np.size(self.DynHSV) > 0:
                hsv = np.array(self.DynHSV, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynFreqBline") and np.size(self.DynFreqBline) > 0:
                freqb = np.array(self.DynFreqBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBandwidthBline") and np.size(self.DynBandwidthBline) > 0:
                bandwidthb = np.array(self.DynBandwidthBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                valueb = np.array(self.DynBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynFreq") and np.size(self.DynFreq) > 0:
                freq = np.array(self.DynFreq, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBandwidth") and np.size(self.DynBandwidth) > 0:
                bandwidth = np.array(self.DynBandwidth, copy=True)
            if use_realtime_dynamic and hasattr(self, "Dyn") and np.size(self.Dyn) > 0:
                value = np.array(self.Dyn, copy=True)
            bridge.cscan_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "rgbb": rgbb,
                    "hsvb": hsvb,
                    "freqb": freqb,
                    "bandwidthb": bandwidthb,
                    "valueb": valueb,
                    "aip": np.array(self.AIP, copy=True),
                    "volume": np.array(self.XYVolume, copy=True) if hasattr(self, "XYVolume") and np.size(self.XYVolume) > 0 else None,
                    "hsv_volume": np.array(self.DynamicHSVVolume, copy=True)
                    if hasattr(self, "DynamicHSVVolume") and np.size(self.DynamicHSVVolume) > 0
                    else None,
                    "rgb": rgb,
                    "hsv": hsv,
                    "freq": freq,
                    "bandwidth": bandwidth,
                    "value": value,
                }
            )

        if kind == "mosaic" and hasattr(self, "SampleMosaic") and np.size(self.SampleMosaic) > 0:
            bline = None
            bline_rgb = None
            mosaic_rgb = None
            bline_hsv = None
            mosaic_hsv = None
            bline_freq = None
            bline_bandwidth = None
            bline_value = None
            mosaic_freq = None
            mosaic_bandwidth = None
            mosaic_value = None
            if hasattr(self, "Bline") and np.size(self.Bline) > 0:
                bline = np.array(self.Bline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynRGBBline") and np.size(self.DynRGBBline) > 0:
                bline_rgb = np.array(self.DynRGBBline, copy=True)
            if STITCH_MOSAIC_DYNAMIC_UI and use_realtime_dynamic and hasattr(self, "SampleMosaicRGB") and np.size(self.SampleMosaicRGB) > 0:
                mosaic_rgb = np.array(self.SampleMosaicRGB, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynHSVBline") and np.size(self.DynHSVBline) > 0:
                bline_hsv = np.array(self.DynHSVBline, copy=True)
            if STITCH_MOSAIC_DYNAMIC_UI and use_realtime_dynamic and hasattr(self, "SampleMosaicHSV") and np.size(self.SampleMosaicHSV) > 0:
                mosaic_hsv = np.array(self.SampleMosaicHSV, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynFreqBline") and np.size(self.DynFreqBline) > 0:
                bline_freq = np.array(self.DynFreqBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBandwidthBline") and np.size(self.DynBandwidthBline) > 0:
                bline_bandwidth = np.array(self.DynBandwidthBline, copy=True)
            if use_realtime_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                bline_value = np.array(self.DynBline, copy=True)
            if STITCH_MOSAIC_DYNAMIC_UI and use_realtime_dynamic and hasattr(self, "SampleMosaicFreq") and np.size(self.SampleMosaicFreq) > 0:
                mosaic_freq = np.array(self.SampleMosaicFreq, copy=True)
            if STITCH_MOSAIC_DYNAMIC_UI and use_realtime_dynamic and hasattr(self, "SampleMosaicBandwidth") and np.size(self.SampleMosaicBandwidth) > 0:
                mosaic_bandwidth = np.array(self.SampleMosaicBandwidth, copy=True)
            if STITCH_MOSAIC_DYNAMIC_UI and use_realtime_dynamic and hasattr(self, "SampleMosaicDyn") and np.size(self.SampleMosaicDyn) > 0:
                mosaic_value = np.array(self.SampleMosaicDyn, copy=True)
            bridge.mosaic_ready.emit(
                {
                    "mode": acq_mode,
                    "mosaic": np.array(self.SampleMosaic, copy=True),
                    "mosaic_volume": (
                        np.array(self.SampleMosaicVolume, copy=False)
                        if hasattr(self, "SampleMosaicVolume") and np.size(self.SampleMosaicVolume) > 0
                        else None
                    ),
                    "mosaic_rgb": mosaic_rgb,
                    "mosaic_hsv": mosaic_hsv,
                    "mosaic_hsv_volume": (
                        np.array(self.SampleMosaicHSVVolume, copy=False)
                        if hasattr(self, "SampleMosaicHSVVolume") and np.size(self.SampleMosaicHSVVolume) > 0
                        else None
                    ),
                    "tile_volume": (
                        np.array(self.XYVolume, copy=True)
                        if hasattr(self, "XYVolume") and np.size(self.XYVolume) > 0
                        else None
                    ),
                    "tile_hsv_volume": (
                        np.array(self.DynamicHSVVolume, copy=True)
                        if hasattr(self, "DynamicHSVVolume") and np.size(self.DynamicHSVVolume) > 0
                        else None
                    ),
                    "mosaic_freq": mosaic_freq,
                    "mosaic_bandwidth": mosaic_bandwidth,
                    "mosaic_value": mosaic_value,
                    "bline": bline,
                    "bline_rgb": bline_rgb,
                    "bline_hsv": bline_hsv,
                    "bline_freq": bline_freq,
                    "bline_bandwidth": bline_bandwidth,
                    "bline_value": bline_value,
                }
            )
            
    def print_display_counts(self, display_name = ''):
        message = f"{self.display_actions} {display_name} display update(s) completed."
        print(message)
        # self.ui.PrintOut.append(message)
        self.display_actions = 0
        
    def Process_aline(self, data, raw = False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        # Bline averaging
        if display_data.shape[0] > 1:
            Ascan = np.mean(display_data,0)
        else:
            Ascan = display_data[0]

        self.Aline = Ascan[Xpixels//2]
        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
            
    
    def Process_bline(self, data, raw = False, dynamic = [], acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        if self.current_dynamic_enabled() or raw:
            if display_data.shape[0] > 1:
                Bline=np.mean(display_data,0)
            else:
                Bline = display_data[0]
        else:
            Bline = display_data[0]
        self.Bline = np.transpose(Bline)
        dyn_data = self.dynamic_std_data(dynamic)
        hsv_data = self.dynamic_hsv_data(dynamic)
        freq_data = self.dynamic_frequency_data(dynamic)
        bandwidth_data = self.dynamic_bandwidth_data(dynamic)
        if self.current_dynamic_enabled() and np.size(dyn_data)>0:
            self.DynBline = np.transpose(dyn_data)
        else:
            self.DynBline = []
            self.Dyn = []
        if self.current_dynamic_enabled() and np.size(hsv_data)>0:
            hsv_data = np.asarray(hsv_data, dtype=np.float32)
            self.DynHSVBline = np.transpose(hsv_data, (1, 0, 2))
            self.DynRGBBline = np.transpose(self.dynamic_hsv_to_rgb(hsv_data), (1, 0, 2))
        else:
            self.DynHSVBline = []
            self.DynRGBBline = []
        if self.current_dynamic_enabled() and np.size(freq_data)>0:
            self.DynFreqBline = np.transpose(np.asarray(freq_data, dtype=np.float32))
        else:
            self.DynFreqBline = []
        if self.current_dynamic_enabled() and np.size(bandwidth_data)>0:
            self.DynBandwidthBline = np.transpose(np.asarray(bandwidth_data, dtype=np.float32))
        else:
            self.DynBandwidthBline = []

        
        if self.current_save_enabled():
            self.Save(data=data, dynamic=dynamic, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

            
    def Process_Cscan_Dynamic(self, data, dynamic=[], acq_mode=None, gpu_avg_count=1):
        # print(dynamic.shape)
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)
        # Shading correction: correct the acquired line before it is used/stored.
        data = self._shading_correct_chunk(data, dynamic_bline_idx, z_logical=Zpixels)
        display_data = self.display_data(data)
        # Bline averaging
        if display_data.shape[0] > 1:
            Bline=np.mean(display_data,0)
        else:
            Bline = display_data[0]

        self.Bline = np.transpose(Bline)
        
        # print('Bline:', self.Bline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        dyn_data = self._shading_gain_line(
            self.dynamic_std_data(dynamic), dynamic_bline_idx, Zpixels
        )
        if np.size(dyn_data)>0:
            self.DynBline = np.transpose(dyn_data)
            # print('DynBline:', self.DynBline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        else:
            self.DynBline = []
        self.DynRGBBline = []
        self.DynHSVBline = []
        self.DynRGB = []
        self.DynHSV = []
        self.DynFreqBline = []
        self.DynBandwidthBline = []
        self.DynFreq = []
        self.DynBandwidth = []
        
        if dynamic_bline_idx == 0:
            self.AIP = np.zeros([Ypixels, Xpixels])
            self.XYVolume = np.zeros((Ypixels, Xpixels, Zpixels), dtype=np.float32)
        # The structure volume of the tile being scanned (shading correction fits
        # the field from the first sample, so it needs this one deterministically).
        self.mosaic_structure_volume = self.XYVolume
        if np.size(dyn_data)>0:
            if dynamic_bline_idx == 0:
                self.Dyn = np.zeros([Ypixels, Xpixels])
                
        # print(Bline.shape, self.AIP.shape)
        print('Ypixel: ', dynamic_bline_idx + 1, ' / ', Ypixels)
        z_idx = self.current_z_depth_index(Zpixels)
        self.XYVolume[dynamic_bline_idx, :, :] = Bline
        self.AIP[dynamic_bline_idx, :] = Bline[:, z_idx]
        if np.size(dyn_data)>0:
            self.Dyn[dynamic_bline_idx, :] = dyn_data[:, z_idx]
        # Keep the field of this line for the end-of-FOV structure Y notch.
        self._accumulate_structure_field(
            data, dynamic_bline_idx, (Ypixels, Xpixels, Zpixels)
        )
        if self.current_save_enabled():
            self.Save(data=data, dynamic=dynamic, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        if dynamic_bline_idx + 1 >= Ypixels:
            # Last Y line of this tile: apply the structure Y notch, then close the
            # files written line by line.
            self._apply_y_notch_to_structure()
            self.close_tiff_writers("cscan dynamic line sweep done")
        
        
    def Process_Cscan(self, data, raw = False, acq_mode=None, gpu_avg_count=1):
        # Shading correction: the whole volume arrives at once here.
        data = self._shading_correct_chunk(
            data, None,
            z_logical=(int(self.ui.DepthRange.value())
                       if hasattr(self.ui, "DepthRange") else None),
        )
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = shape.y_pixels
        # Raw data still needs repeat-frame grouping. Processed data should already be averaged in GPU.
        bline_avg = self.current_bline_avg()
        if raw and bline_avg > 1:
            # reshape into Ypixels x Xpixels x Zpixels
            Cscan = display_data.reshape([Ypixels, bline_avg, Xpixels,Zpixels])
            Cscan=np.mean(Cscan,1)
        else:
            Cscan = display_data.copy()
        # print(data[10,100,50:60])
        self.XYVolume = Cscan
        self.mosaic_structure_volume = self.XYVolume
        z_idx = self.current_z_depth_index(Zpixels)
        self.Bline = np.transpose(Cscan[Ypixels//2,:,:]).copy()# has to be first index, otherwise the memory space is not continuous
        self.AIP = Cscan[:, :, z_idx]
        self.DynBline = []
        self.Dyn = []
        self.DynRGBBline = []
        self.DynHSVBline = []
        self.DynRGB = []
        self.DynHSV = []
        self.DynFreqBline = []
        self.DynBandwidthBline = []
        self.DynFreq = []
        self.DynBandwidth = []
        
        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        # The whole tile was written in this one call, so the file is complete.
        self.close_tiff_writers("cscan tile done")

    def Process_Cscan_RealtimeDynamic(self, data, dynamic=[], acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        zpixels = shape.z_pixels
        xpixels = shape.x_pixels
        ypixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)
        if dynamic_bline_idx == 0:
            # New tile: the whole-tile shading pass has not run for it yet.
            self.shading_tile_corrected = False
        # Shading correction is applied to the whole tile volume at the end of the
        # FOV (see _shading_correct_tile_volumes), not line by line.

        if display_data.shape[0] > 1:
            bline = np.mean(display_data, 0)
        else:
            bline = display_data[0]

        dyn_slice = np.asarray(self.dynamic_std_data(dynamic), dtype=np.float32)
        hsv_slice = self.dynamic_hsv_data(dynamic)
        freq_slice = self.dynamic_frequency_data(dynamic)
        bandwidth_slice = self.dynamic_bandwidth_data(dynamic)
        if np.size(hsv_slice) > 0:
            hsv_slice = np.asarray(hsv_slice, dtype=np.float32)
            rgb_slice = self.dynamic_hsv_to_rgb(hsv_slice)
        else:
            rgb_slice = []
        if np.size(freq_slice) > 0:
            freq_slice = np.asarray(freq_slice, dtype=np.float32)
        if np.size(bandwidth_slice) > 0:
            bandwidth_slice = np.asarray(bandwidth_slice, dtype=np.float32)

        if (
            not isinstance(self.MeanVolume, np.ndarray)
            or self.MeanVolume.shape != (ypixels, xpixels, zpixels)
            or dynamic_bline_idx == 0
        ):
            self.MeanVolume = np.zeros((ypixels, xpixels, zpixels), dtype=np.float32)
            self.DynamicVolume = np.zeros((ypixels, xpixels, zpixels), dtype=np.float32)
            if np.size(hsv_slice) > 0:
                self.DynamicHSVVolume = np.zeros((ypixels, xpixels, zpixels, 3), dtype=np.float32)
                self.DynamicRGBVolume = np.zeros((ypixels, xpixels, zpixels, 3), dtype=np.uint8)
            else:
                self.DynamicHSVVolume = []
                self.DynamicRGBVolume = []
            self.AIP = np.zeros((ypixels, xpixels), dtype=np.float32)
            self.Dyn = np.zeros((ypixels, xpixels), dtype=np.float32)
            self.DynHSV = np.zeros((ypixels, xpixels, 3), dtype=np.float32)
            self.DynFreq = np.zeros((ypixels, xpixels), dtype=np.float32)
            self.DynBandwidth = np.zeros((ypixels, xpixels), dtype=np.float32)

        self.Bline = np.transpose(bline)
        self.DynBline = np.transpose(dyn_slice)
        if np.size(hsv_slice) > 0:
            self.DynHSVBline = np.transpose(hsv_slice, (1, 0, 2))
        else:
            self.DynHSVBline = []
        if np.size(rgb_slice) > 0:
            self.DynRGBBline = np.transpose(rgb_slice, (1, 0, 2))
        else:
            self.DynRGBBline = []
        if np.size(freq_slice) > 0:
            self.DynFreqBline = np.transpose(freq_slice)
        else:
            self.DynFreqBline = []
        if np.size(bandwidth_slice) > 0:
            self.DynBandwidthBline = np.transpose(bandwidth_slice)
        else:
            self.DynBandwidthBline = []
        self.MeanVolume[dynamic_bline_idx, :, :] = bline
        self.XYVolume = self.MeanVolume
        self.DynamicVolume[dynamic_bline_idx, :, :] = dyn_slice
        if np.size(hsv_slice) > 0:
            self.DynamicHSVVolume[dynamic_bline_idx, :, :, :] = hsv_slice
        if np.size(rgb_slice) > 0:
            self.DynamicRGBVolume[dynamic_bline_idx, :, :, :] = rgb_slice
        z_idx = self.current_z_depth_index(zpixels)
        self.AIP[dynamic_bline_idx, :] = bline[:, z_idx]
        self.Dyn[dynamic_bline_idx, :] = dyn_slice[:, z_idx]
        if np.size(freq_slice) > 0:
            self.DynFreq[dynamic_bline_idx, :] = freq_slice[:, z_idx]
        if np.size(bandwidth_slice) > 0:
            self.DynBandwidth[dynamic_bline_idx, :] = bandwidth_slice[:, z_idx]
        if np.size(hsv_slice) > 0:
            self.DynHSV[dynamic_bline_idx, :, :] = hsv_slice[:, z_idx, :]
        if np.size(rgb_slice) > 0:
            if not isinstance(self.DynRGB, np.ndarray) or self.DynRGB.shape != (ypixels, xpixels, 3):
                self.DynRGB = np.zeros((ypixels, xpixels, 3), dtype=np.uint8)
            self.DynRGB[dynamic_bline_idx, :, :] = rgb_slice[:, z_idx, :].astype(np.uint8)
        else:
            self.DynRGB = []
            self.DynHSV = []
            self.DynFreq = []
            self.DynBandwidth = []
        # Keep the field of this line for the end-of-FOV structure Y notch.
        self._accumulate_structure_field(
            data, dynamic_bline_idx, (ypixels, xpixels, zpixels)
        )
        print('Ypixel: ', dynamic_bline_idx + 1, ' / ', ypixels)
        if dynamic_bline_idx + 1 == ypixels:
            # Whole-tile shading correction, once, before display and save.
            self._shading_correct_tile_volumes(zpixels)
            # Filter the structure volume before the per-FOV volumes are saved.
            self._apply_y_notch_to_structure()
            if self.current_save_enabled():
                self.SaveRealtimeCscanDynamicVolumes(acq_mode)
            self.close_tiff_writers("cscan dynamic live sweep done")
            
   
    # --------------------------------------------------------------- shading
    def _shading_line_cache_entry(self, index, z_count):
        """``(flat, dark)`` of one line as ``(X, Z)``, computed once per line.

        The per-line path is only used where the data must be corrected before it is
        written (the non-realtime dynamic stack): caching the interpolated row means
        the 50 repeat frames of a line share one interpolation instead of rebuilding
        the whole ``[Z, Y, X]`` field for every frame.
        """
        cache = getattr(self, "shading_line_cache", None)
        if cache is None:
            cache = {}
            self.shading_line_cache = cache
        key = (int(index), int(z_count))
        entry = cache.get(key)
        if entry is None:
            flat, dark = self.shading_field.line_fields(int(index), 0, int(z_count))
            entry = (np.maximum(flat, 1e-3), dark)
            cache[key] = entry
        return entry

    def _shading_correct_tile_volumes(self, z_pixels=None):
        """Correct the whole tile once, before it is displayed and written.

        Whole-tile pass (the per-line version interpolated the field for every Y line
        and every repeat frame): the structure volume gets ``(I - dark)/flat``, the
        dynamic std volume and the V channel of the colour volume get the flat gain
        only (H and S are shape quantities and stay untouched), and the depth maps
        that feed the live mosaic are refreshed from the corrected volumes.
        """
        field = self.shading_field
        if field is None or getattr(self, "shading_tile_corrected", False):
            return False
        corrected = False
        structure = self.MeanVolume if isinstance(self.MeanVolume, np.ndarray) else self.XYVolume
        if isinstance(structure, np.ndarray) and structure.ndim == 3 and np.size(structure):
            field.apply_structure_volume(structure, inplace=True)
            corrected = True
        dynamic = getattr(self, "DynamicVolume", None)
        if isinstance(dynamic, np.ndarray) and dynamic.ndim == 3 and np.size(dynamic):
            field.apply_gain_volume(np.asarray(dynamic, np.float32), inplace=True)
        hsv = getattr(self, "DynamicHSVVolume", None)
        if isinstance(hsv, np.ndarray) and hsv.ndim == 4 and np.size(hsv):
            field.apply_gain_channel(np.asarray(hsv[..., 2], np.float32), inplace=True)
        if not corrected:
            return False
        self._refresh_mosaic_maps_from_volumes(z_pixels)
        self.shading_tile_corrected = True
        return True

    def _refresh_mosaic_maps_from_volumes(self, z_pixels=None):
        """Rebuild the depth maps of the live mosaic from the corrected volumes."""
        structure = self.MeanVolume if isinstance(self.MeanVolume, np.ndarray) else None
        if not isinstance(structure, np.ndarray) or structure.ndim != 3 or not np.size(structure):
            return
        z_count = int(structure.shape[2])
        z_idx = self.current_z_depth_index(z_pixels or z_count)
        z_idx = int(np.clip(z_idx, 0, z_count - 1))
        if isinstance(getattr(self, "AIP", None), np.ndarray) and self.AIP.shape[:2] == structure.shape[:2]:
            self.AIP[:] = structure[:, :, z_idx]
        dynamic = getattr(self, "DynamicVolume", None)
        if (isinstance(dynamic, np.ndarray) and dynamic.ndim == 3
                and isinstance(getattr(self, "Dyn", None), np.ndarray)
                and self.Dyn.shape[:2] == dynamic.shape[:2]):
            self.Dyn[:] = dynamic[:, :, min(z_idx, dynamic.shape[2] - 1)]
            hsv = getattr(self, "DynamicHSVVolume", None)
            if (isinstance(hsv, np.ndarray) and hsv.ndim == 4
                    and isinstance(getattr(self, "DynHSV", None), np.ndarray)
                    and self.DynHSV.shape[:2] == hsv.shape[:2]):
                self.DynHSV[:] = hsv[:, :, min(z_idx, hsv.shape[2] - 1), :]

    def shading_enabled(self):
        """The run-level "shading correction" switch (default on)."""
        return bool(getattr(self.ui, "shading_correction",
                            shading_correction.SHADING_CORRECTION_ENABLED_DEFAULT))

    def _shading_signature(self, fw_px, fh_px, x_step_um, y_step_um):
        """The acquisition geometry the flat/dark field is valid for."""
        z_pixels = int(self.ui.DepthRange.value()) if hasattr(self.ui, "DepthRange") else 0
        z_start = int(self.ui.DepthStart.value()) if hasattr(self.ui, "DepthStart") else 0
        return shading_correction.field_signature(
            x_pixels=fw_px, y_pixels=fh_px, z_pixels=z_pixels,
            x_step_um=x_step_um or 0.0, y_step_um=y_step_um or 0.0,
            z_start=z_start, z_range=z_pixels,
        )

    def _init_shading_correction(self, fov_locs, fw_px, fh_px, x_step_um, y_step_um):
        """Reuse the field fitted earlier in this session, or arm a new fit.

        Everything happens in RAM: the reference sample accumulates the fit inputs
        while it is scanned and is corrected afterwards, every later sample of the
        session is corrected while it is written.
        """
        self.shading_field = None
        self.shading_accumulator = None
        self.shading_signature = {}
        # Per-line field cache and the "this tile is already corrected" flag.
        self.shading_line_cache = {}
        self.shading_tile_corrected = False
        if not self.shading_enabled():
            print("Shading correction: disabled for this run.")
            return
        signature = self._shading_signature(fw_px, fh_px, x_step_um, y_step_um)
        self.shading_signature = signature
        field = shading_correction.get_session_field(signature)
        if field is not None:
            self.shading_field = field
            print("Shading correction: reusing the session field (" + field.describe() + ")")
            return
        z_pixels = max(1, int(signature.get("z_pixels") or 1))
        planes = list(range(0, z_pixels, shading_correction.SHADING_DEPTH_STEP))
        if not planes or planes[-1] != z_pixels - 1:
            planes.append(z_pixels - 1)
        max_planes = max(2, int(shading_correction.SHADING_MAX_DEPTH_PLANES))
        if len(planes) > max_planes:
            step = max(1, int(np.ceil((z_pixels - 1) / float(max_planes - 1))))
            planes = list(range(0, z_pixels, step))
            if planes[-1] != z_pixels - 1:
                planes.append(z_pixels - 1)
        self.shading_accumulator = shading_correction.FieldAccumulator(
            signature=signature, depth_planes=planes, tile_total=len(fov_locs),
        )
        print(
            "Shading correction: this sample is the reference "
            "({0} tiles, {1} depth planes); the field is fitted when it finishes "
            "and this sample is corrected afterwards.".format(len(fov_locs), len(planes))
        )

    def _shading_accumulate_tile(self, order):
        """Feed one finished tile's structure volume into the pending fit."""
        if self.shading_accumulator is None:
            return
        volume = getattr(self, "mosaic_structure_volume", None)
        if not isinstance(volume, np.ndarray) or volume.ndim != 3 or not np.size(volume):
            return
        if self.shading_accumulator.add_tile(volume, index=order):
            if self.shading_accumulator.tile_count % 10 == 0:
                print("Shading correction: collected {0} tile(s) for the fit".format(
                    self.shading_accumulator.tile_count))

    def _shading_correct_chunk(self, data, index=None, z_logical=None):
        """Correct acquired data before it is used, displayed or stored.

        ``index`` is the Y line of the tile (None corrects a whole volume at once).
        With no field active (the reference sample) the data is returned untouched.
        The whole frame stack of a line is corrected in one vectorised step; the
        field rows are cached per line (:meth:`_shading_line_cache_entry`).
        """
        field = self.shading_field
        if field is None or not isinstance(data, np.ndarray) or not np.size(data):
            return data
        if index is None and data.ndim == 3:
            amplitude, phase = split_amplitude_phase(data, z_logical)
            corrected = field.apply_structure_volume(np.asarray(amplitude, np.float32))
            return combine_amplitude_phase(corrected, phase) if phase is not None else corrected
        if data.ndim != 3:
            return data
        depth = int(z_logical) if z_logical else int(data.shape[-1])
        flat, dark = self._shading_line_cache_entry(int(index), depth)
        interleaved = bool(z_logical) and data.shape[-1] == 2 * int(z_logical)
        if interleaved:
            # amplitude + phase interleaved: correct the amplitude half only
            out = np.array(data, np.float32, copy=True)
            out[..., :depth] = (out[..., :depth] - dark) / flat
            return out
        if data.dtype.kind == "c":
            amplitude = np.abs(data)
            unit_phase = np.divide(data, amplitude, out=np.zeros_like(data),
                                   where=amplitude > 1e-12)
            return ((amplitude - dark) / flat) * unit_phase
        return (np.asarray(data, np.float32) - dark) / flat

    def _shading_gain_line(self, line, index, z_count):
        """Apply the flat gain to one line of a dynamic product (``[X, Z]``)."""
        field = self.shading_field
        data = np.asarray(line)
        if field is None or data.ndim < 2 or not np.size(data):
            return line
        flat, _dark = self._shading_line_cache_entry(int(index), int(z_count))
        if data.ndim == 3:
            out = np.array(data, np.float32, copy=True)
            out[..., 2] = out[..., 2] / flat          # HSV: only the V channel
            return out
        return (data / flat).astype(np.float32)

    def shading_fit(self):
        """Fit the field from the reference sample (called through an action).

        Returns ``{"field", "was_reference", "signature"}``: the weaver thread puts
        the field into the session cache and, when this sample was the reference
        one, corrects it on disk afterwards.
        """
        result = {
            "field": self.shading_field,
            "was_reference": self.shading_accumulator is not None,
            "signature": dict(self.shading_signature),
        }
        accumulator = self.shading_accumulator
        self.shading_accumulator = None
        if accumulator is None or not accumulator.ready:
            if accumulator is not None:
                print("Shading correction: only {0} tile(s) collected -> no field".format(
                    accumulator.tile_count))
                accumulator.reset()
            return result
        started = time.time()
        field = accumulator.fit(verbose=False)
        accumulator.reset()
        if field is None:
            return result
        shading_correction.set_session_field(field)
        self.shading_field = field
        result["field"] = field
        print("Shading correction: field fitted in {0:.1f} s -> {1}".format(
            time.time() - started, field.describe()))
        return result

    def Init_Mosaic(self, context):
        """
        Initializes the mosaic buffer based on the physical span of all FOVs.
        context: [fov_locs, fov_size_px, fov_size_mm]
        fov_locs: list of FOVLocation in mm
        fov_size_px: (width_px, height_px) e.g., (1000, 2000)
        fov_size_mm: (width_mm, height_mm) e.g., (2.0, 3.0)
        """
        fov_locs, fov_size_px, fov_size_mm = context
        fw_px, fh_px = fov_size_px
        fw_mm, fh_mm = fov_size_mm

        # 1. One shared placement rule (mosaic_geometry): a tile lands at its
        # physical stage offset and the canvas is the physical extent of the scan,
        # so the FOV overlap chosen by the planner (FOV_OVERLAP) can never fold two
        # tiles into one cell.  The previous round((pos - min) / fov) grid assumed
        # zero overlap and silently overwrote a tile as soon as the FOVs overlapped
        # by more than half a tile step.
        x_step_um, y_step_um = ui_steps_um(self.ui)
        mm_per_px_x = pixel_size_mm(x_step_um, fw_mm, fw_px)
        mm_per_px_y = pixel_size_mm(y_step_um, fh_mm, fh_px)
        # Shading correction: reuse the session field or arm a fit for this sample.
        self._init_shading_correction(fov_locs, fw_px, fh_px, x_step_um, y_step_um)
        # Fresh mosaic: the normalised cross-fade weight maps start empty.
        self.mosaic_weight_maps = {}
        self.mosaic_positions = [
            (float(location.x), float(location.y)) for location in fov_locs
        ]
        self.mosaic_mm_per_px = (mm_per_px_x, mm_per_px_y)
        self.mosaic_missing_orders = set()
        self.mosaic_layout = build_layout(
            self.mosaic_positions, fw_px, fh_px, mm_per_px_x, mm_per_px_y
        )
        layout_problems = self.mosaic_layout.problems()
        if layout_problems:
            print("WARNING: mosaic layout: " + "; ".join(layout_problems))

        # 2. Canvas = the physical extent of the scan
        num_cols = self.mosaic_layout.cols
        num_rows = self.mosaic_layout.rows
        mw_px = self.mosaic_layout.width_px
        mh_px = self.mosaic_layout.height_px
        self.SampleMosaic = np.ones((mh_px, mw_px), dtype=np.float32)*10
        self.SampleMosaicVolume = []
        self.SampleMosaicRGB = np.zeros((mh_px, mw_px, 3), dtype=np.uint8)
        self.SampleMosaicHSV = np.zeros((mh_px, mw_px, 3), dtype=np.float32)
        self.SampleMosaicDynamicVolume = []
        self.SampleMosaicHSVVolume = []
        self.SampleMosaicDyn = np.zeros((mh_px, mw_px), dtype=np.float32)
        self.SampleMosaicFreq = np.zeros((mh_px, mw_px), dtype=np.float32)
        self.SampleMosaicBandwidth = np.zeros((mh_px, mw_px), dtype=np.float32)
        # print(self.SampleMosaic.shape)
        # Store these for use in Process_Mosaic
        self.fw_mm, self.fh_mm = fw_mm, fh_mm
        self.fw_px, self.fh_px = fw_px, fh_px
        self.mosaic_y_pixels = int(fh_px)

        # Individual tiles are saved at FULL resolution during acquisition. The
        # in-RAM stitched volumes used for LIVE display are downsampled in X/Y
        # by the UI "downsample scale" spinbox so the display stays light; the
        # FULL-resolution stitched mosaic is produced later (offline) from the
        # saved full-res tiles by DynamicPostprocessing.
        scale_control = getattr(self.ui, "scale", None)
        self.mosaic_downsample = max(1, int(scale_control.value())) if scale_control is not None else 1
        self.fw_px_ds = max(1, int(fw_px) // self.mosaic_downsample)
        self.fh_px_ds = max(1, int(fh_px) // self.mosaic_downsample)
        # The in-RAM volumes live on the downsampled grid: same physical placement,
        # coarser pixel size, so the two canvases stay aligned.
        self.mosaic_mm_per_px_ds = (
            pixel_size_mm(x_step_um, fw_mm, self.fw_px_ds, self.mosaic_downsample),
            pixel_size_mm(y_step_um, fh_mm, self.fh_px_ds, self.mosaic_downsample),
        )
        self.mosaic_layout_ds = build_layout(
            self.mosaic_positions,
            self.fw_px_ds,
            self.fh_px_ds,
            self.mosaic_mm_per_px_ds[0],
            self.mosaic_mm_per_px_ds[1],
        )
        self.mosaic_volume_shape = (
            self.mosaic_layout_ds.height_px,
            self.mosaic_layout_ds.width_px,
        )
        # Expose the display downsample so mosaic-correction geometry and the
        # display path stay consistent with the downsampled in-RAM volumes.
        try:
            self.ui.mosaic_display_downsample = self.mosaic_downsample
        except Exception:
            pass
        if self.mosaic_downsample > 1:
            print(
                f"In-RAM stitched volumes downsampled by {self.mosaic_downsample} "
                f"in X/Y (display only); tiles stay full resolution: volume size "
                f"{self.mosaic_volume_shape[0]}x{self.mosaic_volume_shape[1]} px."
            )
        print(f"Mosaic Initialized: {num_cols}x{num_rows} tiles ({mw_px}x{mh_px} px)")
        print(f"Mosaic geometry: {self.mosaic_layout.describe()}")

    def _mosaic_paste_order(self, fov_location):
        """Paste order (index in the layout) of the FOV currently being stitched."""
        if not getattr(self, "mosaic_positions", None):
            return None
        best_index = None
        best_distance = None
        for index, (x_mm, y_mm) in enumerate(self.mosaic_positions):
            distance = (x_mm - float(fov_location.x)) ** 2 + (y_mm - float(fov_location.y)) ** 2
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        return best_index

    def _mark_mosaic_tile_missing(self, order):
        """Remember an unwritten tile so its neighbours never blend over the gap."""
        if order is None or order in self.mosaic_missing_orders:
            return
        self.mosaic_missing_orders.add(order)
        mm_per_px_x, mm_per_px_y = self.mosaic_mm_per_px
        self.mosaic_layout = build_layout(
            self.mosaic_positions, self.fw_px, self.fh_px, mm_per_px_x, mm_per_px_y,
            missing=self.mosaic_missing_orders,
        )
        mm_per_px_x_ds, mm_per_px_y_ds = self.mosaic_mm_per_px_ds
        self.mosaic_layout_ds = build_layout(
            self.mosaic_positions, self.fw_px_ds, self.fh_px_ds,
            mm_per_px_x_ds, mm_per_px_y_ds, missing=self.mosaic_missing_orders,
        )
        print(f"Note: tile {order} was not stitched; its neighbours will not fade over it.")

    def _blend_mosaic_tile(self, buffer, source, paste, weight_key=None):
        """Cross-fade one acquired tile into a full-resolution live mosaic buffer.

        ``weight_key`` names the normalised weight map of that buffer (one per
        mosaic plane, because every plane is blended in the same tile order): the
        map keeps the fade normalised, so the part of a tile that has no neighbour
        yet is not darkened by its own ramp.  Preview pastes of a partly acquired
        tile pass ``None`` instead: they are repeated for the same box, and only the
        final full paste of a tile may advance the weight map.
        """
        if (
            paste is None
            or buffer is None
            or not isinstance(buffer, np.ndarray)
            or source is None
            or not isinstance(source, np.ndarray)
            or np.size(source) == 0
        ):
            return False
        if tuple(source.shape[:2]) != (paste.h, paste.w):
            # A shorter tile (aborted scan) still contributes what it has; the rest
            # of its box stays as it was instead of shifting the whole mosaic.
            padded = np.zeros((paste.h, paste.w) + tuple(source.shape[2:]), dtype=source.dtype)
            height = min(paste.h, int(source.shape[0]))
            width = min(paste.w, int(source.shape[1]))
            padded[:height, :width] = source[:height, :width]
            source = padded
        weights = self._mosaic_weight_map(weight_key, buffer.shape[:2])
        blend_paste(buffer, source, paste, weights=weights)
        return True

    def _mosaic_weight_map(self, key, shape):
        """Normalised fade weight map of one live mosaic plane (lazy, per key)."""
        if not key:
            return None
        maps = getattr(self, "mosaic_weight_maps", None)
        if maps is None:
            maps = {}
            self.mosaic_weight_maps = maps
        wanted = (int(shape[0]), int(shape[1]))
        existing = maps.get(key)
        if existing is None or tuple(existing.shape) != wanted:
            existing = new_weight_map(wanted)
            maps[key] = existing
        return existing
        
    def _paste_stitched_volume(self, storage_attr, source, order, final=True):
        """Paste a per-FOV 3D volume into the stitched mosaic volume.

        The source (e.g. XYVolume / DynamicVolume / DynamicHSVVolume) is
        block-mean downsampled in X and Y only (Z depth and channel axes
        unchanged) using the UI "downsample scale", and cross-faded into the
        correspondingly downsampled stitched volume buffer using the downsampled
        layout (same physical placement as the full-resolution mosaic).
        """
        if (
            source is None
            or not isinstance(source, np.ndarray)
            or np.size(source) == 0
            or self.mosaic_downsample < 1
            or order is None
            or getattr(self, "mosaic_layout_ds", None) is None
        ):
            return
        paste = self.mosaic_layout_ds.placements[order]
        down = downsample_mosaic_volume(source, self.mosaic_downsample)
        if tuple(down.shape[:2]) != (paste.h, paste.w):
            padded = np.zeros((paste.h, paste.w) + tuple(down.shape[2:]), dtype=down.dtype)
            height = min(paste.h, int(down.shape[0]))
            width = min(paste.w, int(down.shape[1]))
            padded[:height, :width] = down[:height, :width]
            down = padded
        tensor = getattr(self, storage_attr, None)
        if not isinstance(tensor, np.ndarray) or tensor.shape[:2] != self.mosaic_volume_shape:
            tensor = np.zeros(
                self.mosaic_volume_shape + tuple(down.shape[2:]), dtype=np.float32
            )
            setattr(self, storage_attr, tensor)
        weights = self._mosaic_weight_map(storage_attr, self.mosaic_volume_shape) if final else None
        blend_paste(tensor, down, paste, weights=weights)

    def Focusing(self, cscan):
         print(cscan.shape)

         bscan = cscan.mean(0)

         ascan = bscan.mean(0)
         print(ascan.shape)
         surfHeight = findchangept(ascan,1)

         ##########################################################
         self.ui.SurfHeight.setValue(surfHeight)
         message = 'Detected tile surface height: '+str(surfHeight)
         print(message)
 
    def _blend_mosaic_maps(self, paste, aip_rows=None):
        """Blend the 2-D mosaic maps of the current tile (AIP + dynamic maps).

        ``aip_rows`` limits every source to the Y rows acquired so far, which is
        what the per-line live preview uses; with ``None`` the complete tile is
        pasted.  The 3-D stitched volumes are not touched here (they are the
        expensive part: see ``_paste_stitched_volume``).
        """
        def acquired(source):
            if aip_rows is None or not isinstance(source, np.ndarray) or source.ndim < 2:
                return source
            return source[: min(int(aip_rows), int(source.shape[0]))]

        wrote = self._blend_mosaic_tile(
            self.SampleMosaic, acquired(self.AIP), paste,
            None if aip_rows is not None else "xy",
        )
        if STITCH_MOSAIC_DYNAMIC_UI and (
            hasattr(self, "DynRGB")
            and isinstance(self.DynRGB, np.ndarray)
            and np.size(self.DynRGB) > 0
            and isinstance(self.SampleMosaicRGB, np.ndarray)
        ):
            self._blend_mosaic_tile(
                self.SampleMosaicRGB, acquired(self.DynRGB), paste,
                None if aip_rows is not None else "rgb",
            )
        if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicHSV, np.ndarray) and isinstance(self.DynHSV, np.ndarray) and np.size(self.DynHSV) > 0:
            self._blend_mosaic_tile(
                self.SampleMosaicHSV, acquired(self.DynHSV), paste,
                None if aip_rows is not None else "hsv",
            )
        if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicDyn, np.ndarray) and isinstance(self.Dyn, np.ndarray) and np.size(self.Dyn) > 0:
            self._blend_mosaic_tile(
                self.SampleMosaicDyn, acquired(self.Dyn), paste,
                None if aip_rows is not None else "dyn",
            )
        if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicFreq, np.ndarray) and isinstance(self.DynFreq, np.ndarray) and np.size(self.DynFreq) > 0:
            self._blend_mosaic_tile(
                self.SampleMosaicFreq, acquired(self.DynFreq), paste,
                None if aip_rows is not None else "freq",
            )
        if STITCH_MOSAIC_DYNAMIC_UI and (
            isinstance(self.SampleMosaicBandwidth, np.ndarray)
            and isinstance(self.DynBandwidth, np.ndarray)
            and np.size(self.DynBandwidth) > 0
        ):
            self._blend_mosaic_tile(
                self.SampleMosaicBandwidth, acquired(self.DynBandwidth), paste,
                None if aip_rows is not None else "bandwidth",
            )
        return wrote

    def _mosaic_filled_rows(self):
        """``(rows acquired so far, rows per FOV)`` or ``(None, 0)``.

        ``(None, 0)`` means the acquisition hands the complete tile over in a
        single action (all non-dynamic modes and the offline stitchers), so the
        caller pastes the whole tile in one pass.
        """
        if not self.current_dynamic_enabled():
            return None, 0
        index = getattr(self.item, "dynamic_bline_idx", None)
        if index is None:
            return None, 0
        source = self.AIP
        if isinstance(source, np.ndarray) and source.ndim == 2 and np.size(source) > 0:
            tile_rows = int(source.shape[0])
        else:
            tile_rows = int(self.mosaic_y_pixels or self.current_y_pixels())
        if tile_rows <= 0:
            return None, 0
        return max(1, min(int(index) + 1, tile_rows)), tile_rows

    @staticmethod
    def _row_limited_paste(paste, rows):
        """Copy of ``paste`` covering only the first ``rows`` rows of the tile."""
        if paste is None or rows >= paste.h:
            return paste
        return dataclasses.replace(
            paste,
            h=int(rows),
            fade_top=min(int(paste.fade_top), int(rows)),
            fade_bottom=0,
            _weight=None,
        )

    def _mosaic_volume_refresh_interval(self):
        """Refresh the stitched volumes every this many acquired Y lines.

        The value is hard-coded in ``OCT_MT.MOSAIC_VOLUME_REFRESH_LINES``; the
        import is deferred because OCT_MT imports this module, and the fallback
        keeps headless scripts (no Qt entry point) working.
        """
        try:
            from OCT_MT import MOSAIC_VOLUME_REFRESH_LINES
        except Exception:
            return 16
        try:
            return max(1, int(MOSAIC_VOLUME_REFRESH_LINES))
        except Exception:
            return 16

    def _paste_mosaic_volumes(self, order, final=True):
        """Paste the per-FOV stitched volumes (XY / dynamic / HSV) into the mosaic.

        The block-mean downsample of a whole per-FOV volume is the expensive part
        of a mosaic update (~0.1/0.1/0.3 s for XY/Dyn/HSV on a 247x840x507 tile),
        so in dynamic mode this runs every few acquired lines and once more when
        the FOV is complete.  The closing call recomputes the complete volume and
        overwrites the same box, so the final content does not depend on how often
        this ran before.  Only that closing call (``final=True``) advances the
        normalised cross-fade weight maps: the repeated preview pastes must not,
        otherwise the same tile would be counted again and again.
        """
        if order is None:
            return
        if STITCH_MOSAIC_VOLUMES_IN_MEMORY and hasattr(self, "XYVolume") and isinstance(self.XYVolume, np.ndarray) and np.size(self.XYVolume) > 0:
            self._paste_stitched_volume("SampleMosaicVolume", self.XYVolume, order, final=final)
        if STITCH_MOSAIC_VOLUMES_IN_MEMORY and (
            hasattr(self, "DynamicVolume")
            and isinstance(self.DynamicVolume, np.ndarray)
            and np.size(self.DynamicVolume) > 0
        ):
            self._paste_stitched_volume(
                "SampleMosaicDynamicVolume", self.DynamicVolume, order, final=final
            )
        if STITCH_MOSAIC_VOLUMES_IN_MEMORY and (
            hasattr(self, "DynamicHSVVolume")
            and isinstance(self.DynamicHSVVolume, np.ndarray)
            and np.size(self.DynamicHSVVolume) > 0
        ):
            self._paste_stitched_volume(
                "SampleMosaicHSVVolume", self.DynamicHSVVolume, order, final=final
            )

    def Process_Mosaic(self, data, raw=False, context=None, acq_mode=None, gpu_avg_count=1):
        """
        Stitches the FOV into the mosaic by calculating its grid index from the anchor.
        context: [fov_locs, fov_location]
        """
        fov_locs, fov_location = context
        # 1. Generate AIP projection from raw data (Y, X, Z)
        if self.realtime_mosaic_dynamic_enabled(acq_mode, getattr(self.item, "dynamic", [])):
            self.Process_Mosaic_RealtimeDynamic(
                data,
                self.item.dynamic,
                acq_mode=acq_mode,
                gpu_avg_count=gpu_avg_count,
            )
        elif self.current_dynamic_enabled():
            self.Process_Cscan_Dynamic(data, self.item.dynamic, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        else:
            self.Process_Cscan(data, raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        # 2. Placement comes from the shared layout built in Init_Mosaic: the tile
        # goes to its physical stage offset (mirrored like the display always was)
        # and the strips shared with already-pasted tiles are cross-faded.  No
        # round((pos - min) / fov) index, so overlapping FOVs can never collide.
        order = self._mosaic_paste_order(fov_location)
        paste = self.mosaic_layout.placements[order] if order is not None else None

        # 3. Dynamic acquisitions call this once per acquired Y line while the
        # per-FOV accumulators are still filling up.  Only the rows that exist yet
        # are blended then (cheap live preview, keeps the colour views updating).
        # The stitched volumes are the expensive part (full-volume block-mean
        # downsample, ~0.1/0.1/0.3 s for XY/Dyn/HSV), so they are refreshed every
        # MOSAIC_VOLUME_REFRESH_LINES acquired lines - the live XYplaneInt /
        # XYplaneDyn views read them - and once more when the FOV is complete.
        filled_rows, tile_rows = self._mosaic_filled_rows()
        if filled_rows is not None and filled_rows < tile_rows:
            self._blend_mosaic_maps(
                self._row_limited_paste(paste, filled_rows), filled_rows
            )
            if filled_rows % self._mosaic_volume_refresh_interval() == 0:
                self._paste_mosaic_volumes(order, final=False)
            return

        wrote = self._blend_mosaic_maps(paste)
        if not wrote:
            self._mark_mosaic_tile_missing(order)
        self._paste_mosaic_volumes(order)
        # Shading correction: collect this tile for the reference-sample fit.
        self._shading_accumulate_tile(order)
        # A full paste means this FOV is complete: close the files it wrote.
        self.close_tiff_writers("mosaic tile done")

        # 5. Update UI

    def Process_Mosaic_RealtimeDynamic(self, data, dynamic, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = int(self.mosaic_y_pixels or self.current_y_pixels())
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)
        if dynamic_bline_idx == 0:
            # New tile: the whole-tile shading pass has not run for it yet.
            self.shading_tile_corrected = False
        # Shading correction is applied to the whole tile volume at the end of the
        # FOV (see _shading_correct_tile_volumes), not line by line: the per-line
        # version interpolated the field for every Y line and every repeat frame.

        if display_data.shape[0] > 1:
            Bline = np.mean(display_data, 0)
        else:
            Bline = display_data[0]

        DynSlice = np.asarray(self.dynamic_std_data(dynamic), dtype=np.float32)
        HSVSlice = self.dynamic_hsv_data(dynamic)
        FreqSlice = self.dynamic_frequency_data(dynamic)
        BandwidthSlice = self.dynamic_bandwidth_data(dynamic)
        if np.size(HSVSlice) > 0:
            HSVSlice = np.asarray(HSVSlice, dtype=np.float32)
            RGBSlice = self.dynamic_hsv_to_rgb(HSVSlice)
        else:
            RGBSlice = []
        if np.size(FreqSlice) > 0:
            FreqSlice = np.asarray(FreqSlice, dtype=np.float32)
        if np.size(BandwidthSlice) > 0:
            BandwidthSlice = np.asarray(BandwidthSlice, dtype=np.float32)

        if (
            not isinstance(self.MeanVolume, np.ndarray)
            or self.MeanVolume.shape != (Ypixels, Xpixels, Zpixels)
            or dynamic_bline_idx == 0
        ):
            self.MeanVolume = np.zeros((Ypixels, Xpixels, Zpixels), dtype=np.float32)
            self.DynamicVolume = np.zeros((Ypixels, Xpixels, Zpixels), dtype=np.float32)
            if np.size(HSVSlice) > 0:
                self.DynamicHSVVolume = np.zeros((Ypixels, Xpixels, Zpixels, 3), dtype=np.float32)
                self.DynamicRGBVolume = np.zeros((Ypixels, Xpixels, Zpixels, 3), dtype=np.uint8)
            else:
                self.DynamicHSVVolume = []
                self.DynamicRGBVolume = []
            self.AIP = np.zeros((Ypixels, Xpixels), dtype=np.float32)
            self.Dyn = np.zeros((Ypixels, Xpixels), dtype=np.float32)
            self.DynHSV = np.zeros((Ypixels, Xpixels, 3), dtype=np.float32)
            self.DynFreq = np.zeros((Ypixels, Xpixels), dtype=np.float32)
            self.DynBandwidth = np.zeros((Ypixels, Xpixels), dtype=np.float32)

        self.Bline = np.transpose(Bline)
        self.DynBline = np.transpose(DynSlice)
        if np.size(HSVSlice) > 0:
            self.DynHSVBline = np.transpose(HSVSlice, (1, 0, 2))
        else:
            self.DynHSVBline = []
        if np.size(RGBSlice) > 0:
            self.DynRGBBline = np.transpose(RGBSlice, (1, 0, 2))
        else:
            self.DynRGBBline = []
        if np.size(FreqSlice) > 0:
            self.DynFreqBline = np.transpose(FreqSlice)
        else:
            self.DynFreqBline = []
        if np.size(BandwidthSlice) > 0:
            self.DynBandwidthBline = np.transpose(BandwidthSlice)
        else:
            self.DynBandwidthBline = []

        self.MeanVolume[dynamic_bline_idx, :, :] = Bline
        self.XYVolume = self.MeanVolume
        self.mosaic_structure_volume = self.MeanVolume
        self.DynamicVolume[dynamic_bline_idx, :, :] = DynSlice
        if np.size(HSVSlice) > 0:
            self.DynamicHSVVolume[dynamic_bline_idx, :, :, :] = HSVSlice
        if np.size(RGBSlice) > 0:
            self.DynamicRGBVolume[dynamic_bline_idx, :, :, :] = RGBSlice
        z_idx = self.current_z_depth_index(Zpixels)
        self.AIP[dynamic_bline_idx, :] = Bline[:, z_idx]
        self.Dyn[dynamic_bline_idx, :] = DynSlice[:, z_idx]
        if np.size(FreqSlice) > 0:
            self.DynFreq[dynamic_bline_idx, :] = FreqSlice[:, z_idx]
        if np.size(BandwidthSlice) > 0:
            self.DynBandwidth[dynamic_bline_idx, :] = BandwidthSlice[:, z_idx]
        if np.size(HSVSlice) > 0:
            self.DynHSV[dynamic_bline_idx, :, :] = HSVSlice[:, z_idx, :]
        if np.size(RGBSlice) > 0:
            if not isinstance(self.DynRGB, np.ndarray) or self.DynRGB.shape != (Ypixels, Xpixels, 3):
                self.DynRGB = np.zeros((Ypixels, Xpixels, 3), dtype=np.uint8)
            self.DynRGB[dynamic_bline_idx, :, :] = RGBSlice[:, z_idx, :].astype(np.uint8)
        else:
            self.DynRGB = []
            self.DynHSV = []
            self.DynFreq = []
            self.DynBandwidth = []

        # Keep the field of this line for the end-of-FOV structure Y notch.
        self._accumulate_structure_field(
            data, dynamic_bline_idx, (Ypixels, Xpixels, Zpixels)
        )

        tile_complete = dynamic_bline_idx + 1 == Ypixels
        if tile_complete:
            # Whole-tile shading correction, once, before display and save.
            self._shading_correct_tile_volumes(Zpixels)
            # Filter the structure volume before the per-FOV volumes are saved.
            self._apply_y_notch_to_structure()
            if self.current_save_enabled():
                self.SaveRealtimeMosaicDynamicVolumes(acq_mode)
            self.close_tiff_writers("mosaic dynamic line sweep done")

    def SaveRealtimeMosaicDynamicVolumes(self, acq_mode):
        bundle = self.item.filename_bundle
        dynamic_filename = bundle.get("dynamic_filename")
        h_filename = bundle.get("dynamic_h_filename")
        s_filename = bundle.get("dynamic_s_filename")
        v_filename = bundle.get("dynamic_v_filename")
        mean_filename = bundle.get("mean_filename")
        if not dynamic_filename or not mean_filename:
            raise RuntimeError("Missing realtime mosaic dynamic filename bundle.")
        store = shading_correction.SHADING_STORE_DTYPE
        try:
            # Per-tile volumes are float16 (the stitched mosaics stay float32).
            TIFF.imwrite(dynamic_filename,
                         np.asarray(self.DynamicVolume, store), append=False)
            if h_filename and s_filename and v_filename:
                if not (isinstance(self.DynamicHSVVolume, np.ndarray)
                        and np.size(self.DynamicHSVVolume) > 0):
                    raise RuntimeError("Missing realtime mosaic HSV volume for dynamic save.")
                hsv = np.asarray(self.DynamicHSVVolume, dtype=store)
                for channel, name in enumerate((h_filename, s_filename, v_filename)):
                    TIFF.imwrite(name, hsv[..., channel], append=False)
            TIFF.imwrite(mean_filename,
                         np.asarray(self.MeanVolume, store), append=False)
        finally:
            pass

    def SaveRealtimeCscanDynamicVolumes(self, acq_mode):
        bundle = self.item.filename_bundle
        dynamic_filename = bundle.get("dynamic_filename")
        dynamic_rgb_filename = bundle.get("dynamic_rgb_filename")
        mean_filename = bundle.get("mean_filename")
        if not dynamic_filename or not mean_filename:
            raise RuntimeError("Missing realtime cscan dynamic filename bundle.")
        store = shading_correction.SHADING_STORE_DTYPE
        try:
            TIFF.imwrite(dynamic_filename,
                         np.asarray(self.DynamicVolume, store), append=False)
            if dynamic_rgb_filename:
                if not (isinstance(self.DynamicHSVVolume, np.ndarray) and np.size(self.DynamicHSVVolume) > 0):
                    raise RuntimeError("Missing realtime cscan HSV volume for RGB dynamic save.")
                TIFF.imwrite(
                    dynamic_rgb_filename,
                    self.dynamic_hsv_to_saved_rgb(self.DynamicHSVVolume),
                    photometric="rgb",
                    append=False,
                )
            TIFF.imwrite(mean_filename,
                         np.asarray(self.MeanVolume, store), append=False)
        finally:
            pass


    def writeTiff(self,filename, image, overlap):
        tif = TIFF.open(filename, mode=overlap)
        tif.write_image(image)
        tif.close()
        
    def Return_mosaic(self):
        self.MosaicQueue.put(self.SampleMosaic)
        # self.Focusing(self.cscan_sum)
        

        
    def Save_mosaic(self):
        message = "Stitched mosaic volume saving is disabled; use tile_positions.json for offline stitching."
        print(message)
        self.emit_status(message)
            
        
    def Save(self, data=[], dynamic=[], raw=False, acq_mode=None, gpu_avg_count=1):
        if getattr(self.item, "skip_save", False):
            return
        shape = data_shape(self.ui, data, raw, acq_mode, gpu_avg_count)
        # Per-tile volumes are written as float16: half the disk traffic, and the
        # stitched mosaics are rebuilt in float32 anyway (see shading_correction).
        data_to_save = np.asarray(
            self.save_data(data), shading_correction.SHADING_STORE_DTYPE
        )
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Yrpt = shape.repeat_count
        Ypixels = shape.y_pixels
        bundle = self.item.filename_bundle or {}
        if acq_mode in ALINE_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing aline filename bundle.")
            self.write_stack_tiff(filename, data_to_save, Yrpt, close_after=True)
                
        elif acq_mode in BLINE_MODES:
            filename = bundle.get("filename")
            dyn_filename = bundle.get("dynamic_filename")
            dyn_rgb_filename = bundle.get("dynamic_rgb_filename")
            if not filename:
                raise RuntimeError("Missing bline filename bundle.")
            dyn_data = self.dynamic_std_data(dynamic)
            hsv_data = self.dynamic_hsv_data(dynamic)
            if dyn_filename is not None and np.size(dyn_data) > 0:
                self.write_tiff_frame(dyn_filename, dyn_data, append_if_exists=False)
            if dyn_rgb_filename is not None and np.size(hsv_data) > 0:
                TIFF.imwrite(
                    dyn_rgb_filename,
                    self.dynamic_hsv_to_saved_rgb(hsv_data),
                    photometric="rgb",
                    append=False,
                )
            self.write_stack_tiff(filename, data_to_save, Yrpt, close_after=True)
        elif acq_mode in CSCAN_MODES:
            if self.current_dynamic_enabled():
                if self.ui.RealtimeDynCheckBox.isChecked():
                    return
                bline_filename = bundle.get("filename")
                if not bline_filename:
                    raise RuntimeError("Missing cscan dynamic stack filename bundle.")
                self.write_stack_tiff(bline_filename, data_to_save, Yrpt)
            else:
                filename = bundle.get("filename")
                if not filename:
                    raise RuntimeError("Missing cscan filename bundle.")
                self.write_stack_tiff(filename, data_to_save, Ypixels)
        elif acq_mode in MOSAIC_DISPLAY_MODES:
            if self.current_dynamic_enabled():
                filename = bundle.get("filename")
                if not filename:
                    raise RuntimeError("Missing mosaic dynamic filename bundle.")
                self.write_stack_tiff(filename, data_to_save, Yrpt)
            else:
                filename = bundle.get("filename")
                if not filename:
                    raise RuntimeError("Missing mosaic filename bundle.")
                self.write_stack_tiff(filename, data_to_save, Ypixels)

    def WriteData(self, data, filename):
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        import time
        start = time.time()
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        if time.time()-start > 1:
            message = 'Saving took '+str(round(time.time()-start,3))+' s.'
            print(message)
            # self.ui.PrintOut.append(message)
        
    def WriteAgar(self, data, context):
        [Ystep, Xstep] = context
        slice_num = self.ui.SliceN.value()
        filename = 'slice-'+str(slice_num)+'-agarTiles X-'+str(Xstep)+'-by Y-'+str(Ystep)+'-.bin'
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        
