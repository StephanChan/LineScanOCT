
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import findchangept
import numpy as np
import traceback
global SCALE
SCALE =1000
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
from scipy import ndimage
# from libtiff import TIFF
import tifffile as TIFF
import time
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION
from HardwareSpecs import TIFF_APPEND_WRITES_DEFAULT
from DataShape import data_shape

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
)

SAVE_SAMPLE_TIME_MODES = (
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)
STITCH_MOSAIC_VOLUMES_IN_MEMORY = False
STITCH_MOSAIC_DYNAMIC_UI = False

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

    def reset_dynamic_accumulators(self):
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
                ):
                    self.display_actions += 1
                    if self.realtime_cscan_dynamic_enabled(self.current_acq_mode, self.item.dynamic):
                        self.Process_Cscan_RealtimeDynamic(
                            self.item.data,
                            self.item.dynamic,
                            self.current_acq_mode,
                            self.item.gpu_avg_count,
                        )
                    elif self.current_dynamic_enabled():
                        self.Process_Cscan_Dynamic(self.item.data, self.item.dynamic, self.current_acq_mode, self.item.gpu_avg_count)
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

    def write_stack_tiff(self, filename, stack, count=None):
        frame_count = int(stack.shape[0]) if count is None else int(count)
        for ii in range(frame_count):
            self.write_tiff_frame(filename, stack[ii], append_if_exists=True)

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
        try:
            if path.exists():
                path.unlink()
        except OSError as error:
            message = f"Failed to reset TIFF output {filename}: {error}"
            print(message)
            self.emit_status(message)
        self._tiff_initialized_files.discard(str(path.resolve()))

    def write_tiff_frame(self, filename, image, append_if_exists=True):
        path = Path(filename)
        resolved = str(path.resolve())
        if self.tiff_append_writes:
            TIFF.imwrite(filename, image, append=append_if_exists)
            if append_if_exists:
                self._tiff_initialized_files.add(resolved)
            return

        first_write = resolved not in self._tiff_initialized_files
        if first_write:
            self.reset_tiff_output(filename)
        TIFF.imwrite(filename, image, append=(append_if_exists and not first_write))
        if append_if_exists:
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
                    "mosaic_volume": None,
                    "mosaic_rgb": mosaic_rgb,
                    "mosaic_hsv": mosaic_hsv,
                    "mosaic_hsv_volume": None,
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
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Ascan = Ascan.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Ascan = np.mean(Ascan,1)
            Xpixels = Xpixels//aline_avg
            
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
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//aline_avg
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
        # Bline averaging
        if display_data.shape[0] > 1:
            Bline=np.mean(display_data,0)
        else:
            Bline = display_data[0]
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//aline_avg

        self.Bline = np.transpose(Bline)
        
        # print('Bline:', self.Bline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        dyn_data = self.dynamic_std_data(dynamic)
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
        if self.current_save_enabled():
            self.Save(data=data, dynamic=dynamic, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        
        
    def Process_Cscan(self, data, raw = False, acq_mode=None, gpu_avg_count=1):
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
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Cscan = Cscan.reshape([Ypixels, Xpixels//aline_avg, aline_avg, Zpixels])
            Cscan = np.mean(Cscan,2)
            Xpixels = Xpixels//aline_avg
        # print(data[10,100,50:60])
        self.XYVolume = Cscan
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

    def Process_Cscan_RealtimeDynamic(self, data, dynamic=[], acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        zpixels = shape.z_pixels
        xpixels = shape.x_pixels
        ypixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)

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
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            bline = bline.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            dyn_slice = dyn_slice.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            if np.size(hsv_slice) > 0:
                hsv_slice = hsv_slice.reshape([xpixels // aline_avg, aline_avg, zpixels, 3]).mean(axis=1)
                rgb_slice = self.dynamic_hsv_to_rgb(hsv_slice)
            if np.size(freq_slice) > 0:
                freq_slice = freq_slice.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            if np.size(bandwidth_slice) > 0:
                bandwidth_slice = bandwidth_slice.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            xpixels = xpixels // aline_avg

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
        print('Ypixel: ', dynamic_bline_idx + 1, ' / ', ypixels)
        if dynamic_bline_idx + 1 == ypixels:
            if self.current_save_enabled():
                self.SaveRealtimeCscanDynamicVolumes(acq_mode)
            
   
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
        
        # 1. Find the bounding box of the FOV centers
        # print(fov_locs)
        xs = [location.x for location in fov_locs]
        ys = [location.y for location in fov_locs]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # 2. Calculate how many tiles are needed in each dimension
        # We use round() to handle tiny stage step errors
        num_cols = int(round((max_x - min_x) / fw_mm)) + 1
        num_rows = int(round((max_y - min_y) / fh_mm)) + 1
        
        # 3. Initialize/Overwrite the active mosaic buffer
        mw_px = num_cols * fw_px
        mh_px = num_rows * fh_px
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
        
        print(f"Mosaic Initialized: {num_cols}x{num_rows} tiles ({mw_px}x{mh_px} px)")
        
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
 
    def Process_Mosaic(self, data, raw=False, context=None, acq_mode=None, gpu_avg_count=1):
        """
        Stitches the FOV into the mosaic by calculating its grid index from the anchor.
        context: [fov_locs, fov_location]
        """
        fov_locs, fov_location = context
        fov_x, fov_y = fov_location.x, fov_location.y
        xs = [location.x for location in fov_locs]
        ys = [location.y for location in fov_locs]
        min_x = min(xs)
        min_y = min(ys)
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
        # 2. Calculate the Grid Index (Column and Row)
        # We determine how many FOV-widths away from the minimum X/Y we are
        col_idx = int(round((fov_x - min_x) / self.fw_mm))
        row_idx = int(round((fov_y - min_y) / self.fh_mm))
        # 3. Calculate pixel offsets
        off_x = col_idx * self.fw_px
        off_y = row_idx * self.fh_px

        # 4. Paste into the mosaic buffer
        # Bounds checking added just in case of unexpected stage travel
        # print('sampleMosaic: ', self.SampleMosaic)
        mh, mw = self.SampleMosaic.shape
        y1, y2 = off_y, off_y + self.fh_px
        x1, x2 = off_x, off_x + self.fw_px
        # print(x1,x2,y1,y2)
        
        if y2 <= mh and x2 <= mw:
            # print(y1,y2,x1,x2, off_y, off_x, mh, mw)
            self.SampleMosaic[y1:y2, x1:x2] = self.AIP
            if STITCH_MOSAIC_VOLUMES_IN_MEMORY and hasattr(self, "XYVolume") and isinstance(self.XYVolume, np.ndarray) and np.size(self.XYVolume) > 0:
                z_pixels = self.XYVolume.shape[2]
                if not isinstance(self.SampleMosaicVolume, np.ndarray) or self.SampleMosaicVolume.shape != (mh, mw, z_pixels):
                    self.SampleMosaicVolume = np.zeros((mh, mw, z_pixels), dtype=np.float32)
                self.SampleMosaicVolume[y1:y2, x1:x2, :] = self.XYVolume
            if STITCH_MOSAIC_DYNAMIC_UI and (
                hasattr(self, "DynRGB")
                and isinstance(self.DynRGB, np.ndarray)
                and np.size(self.DynRGB) > 0
                and isinstance(self.SampleMosaicRGB, np.ndarray)
            ):
                self.SampleMosaicRGB[y1:y2, x1:x2, :] = self.DynRGB
            if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicHSV, np.ndarray) and isinstance(self.DynHSV, np.ndarray) and np.size(self.DynHSV) > 0:
                self.SampleMosaicHSV[y1:y2, x1:x2, :] = self.DynHSV
            if STITCH_MOSAIC_VOLUMES_IN_MEMORY and (
                hasattr(self, "DynamicVolume")
                and isinstance(self.DynamicVolume, np.ndarray)
                and np.size(self.DynamicVolume) > 0
            ):
                z_pixels = self.DynamicVolume.shape[2]
                if (
                    not isinstance(self.SampleMosaicDynamicVolume, np.ndarray)
                    or self.SampleMosaicDynamicVolume.shape != (mh, mw, z_pixels)
                ):
                    self.SampleMosaicDynamicVolume = np.zeros((mh, mw, z_pixels), dtype=np.float32)
                self.SampleMosaicDynamicVolume[y1:y2, x1:x2, :] = self.DynamicVolume
            if STITCH_MOSAIC_VOLUMES_IN_MEMORY and (
                hasattr(self, "DynamicHSVVolume")
                and isinstance(self.DynamicHSVVolume, np.ndarray)
                and np.size(self.DynamicHSVVolume) > 0
            ):
                z_pixels = self.DynamicHSVVolume.shape[2]
                if (
                    not isinstance(self.SampleMosaicHSVVolume, np.ndarray)
                    or self.SampleMosaicHSVVolume.shape != (mh, mw, z_pixels, 3)
                ):
                    self.SampleMosaicHSVVolume = np.zeros((mh, mw, z_pixels, 3), dtype=np.float32)
                self.SampleMosaicHSVVolume[y1:y2, x1:x2, :, :] = self.DynamicHSVVolume
            if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicDyn, np.ndarray) and isinstance(self.Dyn, np.ndarray) and np.size(self.Dyn) > 0:
                self.SampleMosaicDyn[y1:y2, x1:x2] = self.Dyn
            if STITCH_MOSAIC_DYNAMIC_UI and isinstance(self.SampleMosaicFreq, np.ndarray) and isinstance(self.DynFreq, np.ndarray) and np.size(self.DynFreq) > 0:
                self.SampleMosaicFreq[y1:y2, x1:x2] = self.DynFreq
            if STITCH_MOSAIC_DYNAMIC_UI and (
                isinstance(self.SampleMosaicBandwidth, np.ndarray)
                and isinstance(self.DynBandwidth, np.ndarray)
                and np.size(self.DynBandwidth) > 0
            ):
                self.SampleMosaicBandwidth[y1:y2, x1:x2] = self.DynBandwidth
        else:
            print(f"Warning: Tile at ({col_idx}, {row_idx}) is out of mosaic bounds.")

        # 5. Update UI

    def Process_Mosaic_RealtimeDynamic(self, data, dynamic, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = int(self.mosaic_y_pixels or self.current_y_pixels())
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)

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
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels // aline_avg, aline_avg, Zpixels]).mean(axis=1)
            DynSlice = DynSlice.reshape([Xpixels // aline_avg, aline_avg, Zpixels]).mean(axis=1)
            if np.size(HSVSlice) > 0:
                HSVSlice = HSVSlice.reshape([Xpixels // aline_avg, aline_avg, Zpixels, 3]).mean(axis=1)
                RGBSlice = self.dynamic_hsv_to_rgb(HSVSlice)
            if np.size(FreqSlice) > 0:
                FreqSlice = FreqSlice.reshape([Xpixels // aline_avg, aline_avg, Zpixels]).mean(axis=1)
            if np.size(BandwidthSlice) > 0:
                BandwidthSlice = BandwidthSlice.reshape([Xpixels // aline_avg, aline_avg, Zpixels]).mean(axis=1)
            Xpixels = Xpixels // aline_avg

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

        tile_complete = dynamic_bline_idx + 1 == Ypixels
        if tile_complete:
            if self.current_save_enabled():
                self.SaveRealtimeMosaicDynamicVolumes(acq_mode)

    def SaveRealtimeMosaicDynamicVolumes(self, acq_mode):
        bundle = self.item.filename_bundle
        dynamic_filename = bundle.get("dynamic_filename")
        dynamic_rgb_filename = bundle.get("dynamic_rgb_filename")
        mean_filename = bundle.get("mean_filename")
        if not dynamic_filename or not mean_filename:
            raise RuntimeError("Missing realtime mosaic dynamic filename bundle.")
        try:
            TIFF.imwrite(dynamic_filename, np.asarray(self.DynamicVolume, dtype=np.float32), append=False)
            if dynamic_rgb_filename:
                if not (isinstance(self.DynamicHSVVolume, np.ndarray) and np.size(self.DynamicHSVVolume) > 0):
                    raise RuntimeError("Missing realtime mosaic HSV volume for RGB dynamic save.")
                TIFF.imwrite(
                    dynamic_rgb_filename,
                    self.dynamic_hsv_to_saved_rgb(self.DynamicHSVVolume),
                    photometric="rgb",
                    append=False,
                )
            TIFF.imwrite(mean_filename, np.asarray(self.MeanVolume, dtype=np.float32), append=False)
        finally:
            pass

    def SaveRealtimeCscanDynamicVolumes(self, acq_mode):
        bundle = self.item.filename_bundle
        dynamic_filename = bundle.get("dynamic_filename")
        dynamic_rgb_filename = bundle.get("dynamic_rgb_filename")
        mean_filename = bundle.get("mean_filename")
        if not dynamic_filename or not mean_filename:
            raise RuntimeError("Missing realtime cscan dynamic filename bundle.")
        try:
            TIFF.imwrite(dynamic_filename, np.asarray(self.DynamicVolume, dtype=np.float32), append=False)
            if dynamic_rgb_filename:
                if not (isinstance(self.DynamicHSVVolume, np.ndarray) and np.size(self.DynamicHSVVolume) > 0):
                    raise RuntimeError("Missing realtime cscan HSV volume for RGB dynamic save.")
                TIFF.imwrite(
                    dynamic_rgb_filename,
                    self.dynamic_hsv_to_saved_rgb(self.DynamicHSVVolume),
                    photometric="rgb",
                    append=False,
                )
            TIFF.imwrite(mean_filename, np.asarray(self.MeanVolume, dtype=np.float32), append=False)
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
        data_to_save = self.save_data(data)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Yrpt = shape.repeat_count
        Ypixels = shape.y_pixels
        bundle = self.item.filename_bundle or {}
        if acq_mode in ALINE_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing aline filename bundle.")
            self.write_stack_tiff(filename, data_to_save, Yrpt)
                
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
            self.write_stack_tiff(filename, data_to_save, Yrpt)
                
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
        
