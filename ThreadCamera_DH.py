# -*- coding: utf-8 -*-
"""
Main camera control thread using Amcam SDK with PyQt GUI integration.
Includes functionality for live preview, snap image, exposure control,
and mosaic image stitching.
"""

import time
import threading
import queue
import os
import sys
from PyQt5.QtCore import QThread
import numpy as np
import traceback
from Generaic_functions import *  # Shared plotting and waveform helpers used by the camera thread.
import matplotlib.pyplot as plt
from ActionFields import DbackActionField, DActionField
from CameraUi import (
    downsample_spectral_axis,
    effective_camera_sample_count,
    raw_camera_sample_count,
    spectral_downsample,
)
import matplotlib.pyplot as plt
global SIM
# Fall back to simulation when the Daheng SDK cannot be imported.
try:
    GALAXY_SDK_ROOT = r"D:\GalaxySDK"

    GALAXY_PYTHON_DIR = os.path.join(
        GALAXY_SDK_ROOT,
        "Development",
        "Samples",
        "Python",
    )

    GALAXY_DLL_DIR = os.path.join(
        GALAXY_SDK_ROOT,
        "APIDll",
        "Win64",
    )

    sys.path.append(GALAXY_PYTHON_DIR)
    os.add_dll_directory(GALAXY_DLL_DIR)
    import gxipy as gx 
    import DahengCamera_init
    from ctypes import *
    from gxipy.gxidef import *
    from gxipy.ImageFormatConvert import *
    SIM = False
except Exception as error:
    print(
        "Daheng camera SDK import failed. The configured Galaxy SDK directory may be wrong: "
        f"{locals().get('GALAXY_PYTHON_DIR', '<unresolved>')}. "
        f"Import error: {error}. Using simulation."
    )
    SIM = True

CONTINUOUS = 0x7FFFFFFF
DAHENG_PRINT_CAMERA_CONFIG = False
DAHENG_SPECTRAL_DIMENSION = "horizontal"
# Set DAHENG_SPECTRAL_DIMENSION to:
#   "vertical"   -> camera Height = NSamples_DH, camera Width = AlinesPerBline
#   "horizontal" -> camera Width = NSamples_DH, camera Height = AlinesPerBline
# Packed conversion and transpose-write are the long pole for long dynamic scans.
# Increase this cautiously: block completion is still emitted in frame order below.
DAHENG_CONSUMER_WORKERS = 4
DAHENG_BUFFER_TIMEOUT_MS = 200
DAHENG_MAX_CONSECUTIVE_TIMEOUTS = 20


def daheng_spectral_axis():
    dimension = DAHENG_SPECTRAL_DIMENSION.lower()
    if dimension == "vertical":
        return 0
    if dimension == "horizontal":
        return 1
    raise ValueError(
        "DAHENG_SPECTRAL_DIMENSION must be 'vertical' or 'horizontal', "
        f"got {DAHENG_SPECTRAL_DIMENSION!r}"
    )


def daheng_spectral_is_horizontal():
    return daheng_spectral_axis() == 1

def get_best_valid_bits(pixel_format):
    valid_bits = DxValidBit.BIT0_7
    if pixel_format in (GxPixelFormatEntry.MONO8,
                        GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8,
                        GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8,
                        GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8,
                        GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        valid_bits = DxValidBit.BIT0_7
    elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.MONO10_P,
                          GxPixelFormatEntry.BAYER_GR10, GxPixelFormatEntry.BAYER_RG10,
                          GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10,
                          GxPixelFormatEntry.BAYER_GR10_P, GxPixelFormatEntry.BAYER_RG10_P,
                          GxPixelFormatEntry.BAYER_GB10_P, GxPixelFormatEntry.BAYER_BG10_P,
                          GxPixelFormatEntry.BAYER_GR10_PACKED, GxPixelFormatEntry.BAYER_RG10_PACKED,
                          GxPixelFormatEntry.BAYER_GB10_PACKED, GxPixelFormatEntry.BAYER_BG10_PACKED):
        valid_bits = DxValidBit.BIT2_9
    elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.MONO12_P,
                          GxPixelFormatEntry.BAYER_GR12, GxPixelFormatEntry.BAYER_RG12,
                          GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12,
                          GxPixelFormatEntry.BAYER_GR12_P, GxPixelFormatEntry.BAYER_RG12_P,
                          GxPixelFormatEntry.BAYER_GB12_P, GxPixelFormatEntry.BAYER_BG12_P,
                          GxPixelFormatEntry.BAYER_GR12_PACKED, GxPixelFormatEntry.BAYER_RG12_PACKED,
                          GxPixelFormatEntry.BAYER_GB12_PACKED, GxPixelFormatEntry.BAYER_BG12_PACKED):
        valid_bits = DxValidBit.BIT4_11
    elif pixel_format in (GxPixelFormatEntry.MONO14, GxPixelFormatEntry.MONO14_P,
                          GxPixelFormatEntry.BAYER_GR14, GxPixelFormatEntry.BAYER_RG14,
                          GxPixelFormatEntry.BAYER_GB14, GxPixelFormatEntry.BAYER_BG14,
                          GxPixelFormatEntry.BAYER_GR14_P, GxPixelFormatEntry.BAYER_RG14_P,
                          GxPixelFormatEntry.BAYER_GB14_P, GxPixelFormatEntry.BAYER_BG14_P,
                          ):
        valid_bits = DxValidBit.BIT6_13
    elif pixel_format in (GxPixelFormatEntry.MONO16,
                          GxPixelFormatEntry.BAYER_GR16, GxPixelFormatEntry.BAYER_RG16,
                          GxPixelFormatEntry.BAYER_GB16, GxPixelFormatEntry.BAYER_BG16):
        valid_bits = DxValidBit.BIT8_15
    return valid_bits


class PackedPixelFormatConverter(object):
    """
    Reuses ImageFormatConvert destination/valid-bits settings and a single output buffer
    while width, height, source pixel format, and destination format stay the same.
    This avoids per-frame allocation and redundant SDK configuration.
    """
    def __init__(self, image_convert_obj):
        self._image_convert = image_convert_obj
        self._cache_key = None
        self._dest_pf = None
        self._valid_bits = None
        self._out_buf = None
        self._buf_size = 0

    @staticmethod
    def _geometry_key(raw_image):
        fd = raw_image.frame_data
        return (fd.width, fd.height, fd.pixel_format)

    def convert(self, raw_image, dest_pixel_format):
        key = self._geometry_key(raw_image)
        src_pf = raw_image.get_pixel_format()
        valid_bits = get_best_valid_bits(src_pf)
        if (
            key != self._cache_key
            or dest_pixel_format != self._dest_pf
            or valid_bits != self._valid_bits
        ):
            self._image_convert.set_dest_format(dest_pixel_format)
            self._image_convert.set_valid_bits(valid_bits)
            buf_size = self._image_convert.get_buffer_size_for_conversion(raw_image)
            if self._out_buf is None or buf_size != self._buf_size:
                self._out_buf = (c_ubyte * buf_size)()
                self._buf_size = buf_size
            self._cache_key = key
            self._dest_pf = dest_pixel_format
            self._valid_bits = valid_bits

        self._image_convert.convert(raw_image, addressof(self._out_buf), self._buf_size, False)
        return self._out_buf, self._buf_size


# Daheng camera worker thread.
class Camera(QThread):
    def __init__(self):
        # Initialize thread state and camera handles.
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Camera thread exited.'
        self.hcam = None       # Daheng camera handle
        self.hcam_fr = None    # Remote feature-control handle
        self.device_manager = None
        self._memory_write_mode = None

    def run(self):
        if not (SIM or self.SIM):
            print('initializing camera...')
            self.initCamera()
            self.GetExposure()
            self.GetGain()
            self.GetPixelDepth()
        self.QueueOut()

    # Main queue dispatcher. GUI updates are sent through the UI bridge.
    def QueueOut(self):
        self.item = self.queue.get()  # Wait for the next camera command.
        while self.item.action != 'exit':
            try:
                if self.item.action == 'ConfigureBoard':
                    self.ConfigureBoard()
                # elif self.item.action == 'SetExposure':
                #     self.SetExposure()
                # elif self.item.action == 'GetExposure':
                #     self.GetExposure()
                # elif self.item.action == 'AutoExposure':
                #     self.AutoExposure()
                # elif self.item.action == 'SetGain':
                #     self.SetGain()
                # elif self.item.action == 'GetGain':
                #     self.GetGain()
                # elif self.item.action == 'AutoGain':
                #     self.AutoGain()
                elif self.item.action == 'Acquire':
                    if self.hcam is not None:
                        self.Stream_on()
                        self.Acquire()
                        self.Stream_off()
                    else:
                        self.simData()
                
                else:
                    message = f"Unknown camera command: {self.item.action}"
                    self.emit_status(message)
                    print(message)
            except Exception as error:
                message = "Camera command failed. This action was skipped: " + str(error)
                self.emit_status(message)
                print(message)
                print(traceback.format_exc())
            self.item = self.queue.get()  # Wait for the next camera command.
        self.Close()
        print(self.exit_message)
        self.emit_status(self.exit_message)

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))
        
    # Open the camera and apply persistent stream settings.
    def initCamera(self):
        # Open the device manager, then open the first camera if one is present.
        # If no hardware is available, leave self.hcam as None and let simulation handle acquisition.
        if not (SIM or self.SIM):
            self.device_manager = gx.DeviceManager()  # Open device manager
            if self.device_manager.update_all_device_list()[0] == 0:
                print("No camera found")
                self.hcam = None
            else:
                self.hcam = self.device_manager.open_device_by_index(1)
                try:
                    self.hcam_fr = self.hcam.get_remote_device_feature_control()  # Remote device feature control
                    self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                    self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                    # self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
                    # pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
                    # self.ui.PixelFormat_display_DH.setText(pixelformat[1])
                    # self.hcam_fr.feature_save("export_config_file.txt")
                    # self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
                    
                    self.hcam_s = self.hcam.get_stream(1).get_feature_control()  # Stream feature control
                    self.hcam_s.get_enum_feature("StreamBufferHandlingMode").set("OldestFirst")
                    self.hcam.data_stream[0].set_acquisition_buffer_number(1000)
                except Exception as ex:
                    # Keep startup running even if optional stream tuning fails.
                    print(ex)
    
    def ConfigureBoard(self):
        self.AlinesPerBline = self.ui.AlinesPerBline.value()
        self.NSamples_DH = raw_camera_sample_count(self.ui)
        self.SpectralDS = spectral_downsample(self.ui)
        self.ProcessedSamples = effective_camera_sample_count(self.ui)
        if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline']:
            self.BlinesPerAcq = self.ui.BlineAVG.value() 
        elif self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','ContinuousCscan']:
            self.BlinesPerAcq = CONTINUOUS
        elif self.ui.ACQMode.currentText() in ['FiniteCscan','PlateScan','PlatePreScan', 'WellScan','TimedPlateScan']:
            self.BlinesPerAcq = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
            
        if self.hcam is not None:
            self.SetExposure()
            self.SetGain()
            self.SetPixelDepth()
            self.hcam_fr.get_enum_feature("TriggerMode").set(self.ui.TriggerON_DH.currentText())
            self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
            self.hcam_fr.get_enum_feature("TriggerActivation").set(self.ui.TriggerActivation_DH.currentText())
            # self.hcam_fr.get_enum_feature("TriggerDelay").set(int(self.ui.TriggerDelay_DH.value()*1000.0))

            if daheng_spectral_is_horizontal():
                camera_width = self.NSamples_DH
                camera_height = self.AlinesPerBline
                offset_x = self.ui.offsetW_DH.value()
                offset_y = self.ui.offsetH.value()
            else:
                camera_width = self.AlinesPerBline
                camera_height = self.NSamples_DH
                offset_x = self.ui.offsetH.value()
                offset_y = self.ui.offsetW_DH.value()

            self.hcam_fr.get_int_feature("Width").set(camera_width)
            self.hcam_fr.get_int_feature("Height").set(camera_height)
            self.hcam_fr.get_int_feature("OffsetX").set(offset_x)
            self.hcam_fr.get_int_feature("OffsetY").set(offset_y)
        if DAHENG_PRINT_CAMERA_CONFIG:
            self.print_configuration_readback()
        # self.DbackQueue.put(0)

    def print_configuration_readback(self):
        print("Daheng camera configuration:")
        print("  Camera selection: %s" % self.ui.Camera.currentText())
        print("  Spectral dimension: %s" % DAHENG_SPECTRAL_DIMENSION)
        print("  Spectral array axis: %d" % daheng_spectral_axis())
        if daheng_spectral_is_horizontal():
            print("  ROI raw camera Width/NSamples_DH: %d" % int(self.NSamples_DH))
            print("  ROI raw camera Height/AlinesPerBline: %d" % int(self.AlinesPerBline))
            print("  OffsetX/offsetW_DH: %d" % int(self.ui.offsetW_DH.value()))
            print("  OffsetY/offsetH: %d" % int(self.ui.offsetH.value()))
        else:
            print("  ROI raw camera Height/NSamples_DH: %d" % int(self.NSamples_DH))
            print("  ROI raw camera Width/AlinesPerBline: %d" % int(self.AlinesPerBline))
            print("  OffsetY/offsetW_DH: %d" % int(self.ui.offsetW_DH.value()))
            print("  OffsetX/offsetH: %d" % int(self.ui.offsetH.value()))
        print("  SpectralDS: %d" % int(self.SpectralDS))
        print("  Processed samples in memory: %d" % int(self.ProcessedSamples))
        print("  PixelFormat_DH: %s" % self.ui.PixelFormat_DH.currentText())
        print("  PixelFormat_display_DH: %s" % self.ui.PixelFormat_display_DH.text())
        print("  TriggerMode UI: %s" % self.ui.TriggerON_DH.currentText())
        print("  TriggerSource UI: %s" % self.ui.TriggerSource_DH.currentText())
        print("  TriggerActivation UI: %s" % self.ui.TriggerActivation_DH.currentText())
        print("  TriggerDelay_DH UI ms: %.6g" % float(self.ui.TriggerDelay_DH.value()))
        print("  ACQMode: %s" % self.ui.ACQMode.currentText())
        print("  BlinesPerAcq: %s" % str(self.BlinesPerAcq))
        try:
            print("  Memory[0].shape: %s dtype=%s" % (str(self.Memory[0].shape), self.Memory[0].dtype))
        except Exception as error:
            print("  Memory[0].shape: unavailable (%s)" % error)
        if self.hcam is not None:
            for name in ("Width", "Height", "OffsetX", "OffsetY"):
                try:
                    feature = self.hcam_fr.get_int_feature(name).get()
                    print("  Camera %s readback: %s" % (name, str(feature)))
                except Exception as error:
                    print("  Camera %s readback unavailable: %s" % (name, error))
            for name in ("TriggerMode", "TriggerSource", "TriggerActivation"):
                try:
                    feature = self.hcam_fr.get_enum_feature(name).get()
                    print("  Camera %s readback: %s" % (name, str(feature)))
                except Exception as error:
                    print("  Camera %s readback unavailable: %s" % (name, error))
            try:
                feature = self.hcam_fr.get_float_feature("TriggerDelay").get()
                print("  Camera TriggerDelay readback: %s" % str(feature))
            except Exception as error:
                print("  Camera TriggerDelay readback unavailable: %s" % error)
            
    def Acquire(self):
        NBlines = self.Memory[0].shape[0]
        self._memory_write_mode = None
        grab_q = queue.Queue(maxsize=128)
        grab_stop = object()
        use_packed = self.ui.PixelFormat_DH.currentText() in ["Mono12Packed"]
        consumer_error = []
        ds = self.hcam.data_stream[0]
        start_memory_slot = self.MemoryLoc
        worker_count = DAHENG_CONSUMER_WORKERS if use_packed else 1
        profile = {
            "dq_buf": 0.0,
            "queue_put": 0.0,
            "queue_get": 0.0,
            "packed_convert": 0.0,
            "numpy_view": 0.0,
            "memory_write": 0.0,
            "q_buf": 0.0,
            "grabbed": 0,
            "processed": 0,
            "timeouts": 0,
            "max_processing_queue": 0,
            "max_databack_queue": 0,
        }
        profile_lock = threading.Lock()
        completion_lock = threading.Lock()
        completed_blocks = {}
        completed_block_ids = set()
        next_block_to_emit = [0]
        total_t0 = time.perf_counter()

        def add_profile(key, value):
            with profile_lock:
                profile[key] += value

        def increment_profile(key, value=1):
            with profile_lock:
                profile[key] += value

        def mark_frame_complete(frame_number):
            block_id = frame_number // NBlines
            with completion_lock:
                completed = completed_blocks.get(block_id, 0) + 1
                completed_blocks[block_id] = completed
                if completed == NBlines:
                    completed_block_ids.add(block_id)
                    del completed_blocks[block_id]
                    while next_block_to_emit[0] in completed_block_ids:
                        emit_block_id = next_block_to_emit[0]
                        memory_slot = (start_memory_slot + emit_block_id) % self.memoryCount
                        self.DatabackQueue.put(DbackActionField(memory_slot))
                        with profile_lock:
                            profile["max_databack_queue"] = max(
                                profile["max_databack_queue"],
                                self.DatabackQueue.qsize(),
                            )
                        completed_block_ids.remove(emit_block_id)
                        next_block_to_emit[0] += 1

        def consumer(worker_id):
            converter = None
            if use_packed:
                converter = PackedPixelFormatConverter(
                    self.device_manager.create_image_format_convert()
                )
            try:
                while True:
                    t_get = time.perf_counter()
                    item = grab_q.get()
                    add_profile("queue_get", time.perf_counter() - t_get)
                    if item is grab_stop:
                        break
                    buf, frame_number = item
                    block_id = frame_number // NBlines
                    memory_slot = (start_memory_slot + block_id) % self.memoryCount
                    frame_index = frame_number % NBlines
                    try:
                        if buf is None:
                            print("camera time out...")
                            increment_profile("timeouts")
                            if daheng_spectral_is_horizontal():
                                fallback_shape = [self.AlinesPerBline, self.ProcessedSamples]
                            else:
                                fallback_shape = [self.ProcessedSamples, self.AlinesPerBline]
                            Bline = np.zeros(fallback_shape, dtype=self.Memory[memory_slot].dtype)
                        else:
                            if use_packed:
                                t_convert = time.perf_counter()
                                mono_image_array, _ = converter.convert(
                                    buf, GxPixelFormatEntry.MONO12)
                                add_profile("packed_convert", time.perf_counter() - t_convert)
                                t_view = time.perf_counter()
                                if daheng_spectral_is_horizontal():
                                    raw_shape = (self.AlinesPerBline, self.NSamples_DH)
                                else:
                                    raw_shape = (self.NSamples_DH, self.AlinesPerBline)
                                Bline = np.frombuffer(
                                    mono_image_array, dtype=np.uint16).reshape(raw_shape)
                                add_profile("numpy_view", time.perf_counter() - t_view)
                            else:
                                t_view = time.perf_counter()
                                Bline = buf.get_numpy_array()
                                add_profile("numpy_view", time.perf_counter() - t_view)
                            Bline = downsample_spectral_axis(
                                Bline,
                                self.SpectralDS,
                                axis=daheng_spectral_axis(),
                            )

                        t_write = time.perf_counter()
                        self.write_bline_to_memory(Bline, memory_slot, frame_index)
                        add_profile("memory_write", time.perf_counter() - t_write)
                        increment_profile("processed")
                        mark_frame_complete(frame_number)
                    finally:
                        if buf is not None:
                            t_q = time.perf_counter()
                            ds.q_buf(buf)
                            add_profile("q_buf", time.perf_counter() - t_q)
            except Exception:
                consumer_error.append(f"Worker {worker_id} failed:\n" + traceback.format_exc())
                print(traceback.format_exc())

        workers = [
            threading.Thread(target=consumer, args=(worker_id,), name=f"DHGrabConvert{worker_id}", daemon=True)
            for worker_id in range(worker_count)
        ]
        for worker in workers:
            worker.start()
        acquisition_error_message = None
        try:
            print(
                "Daheng camera acquisition ready: "
                f"stream_on={getattr(self, '_last_stream_on_elapsed', 0.0):.3f}s, "
                f"workers={worker_count}, expected_frames={self.BlinesPerAcq}."
            )
            self.DbackQueue.put(0)
            BlinesCount = 0
            consecutive_timeouts = 0
            while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
                t_dq = time.perf_counter()
                buf = ds.dq_buf(timeout=DAHENG_BUFFER_TIMEOUT_MS)
                add_profile("dq_buf", time.perf_counter() - t_dq)
                if buf is None:
                    consecutive_timeouts += 1
                    print(
                        "camera time out... "
                        f"consecutive={consecutive_timeouts}, "
                        f"received={BlinesCount}/{self.BlinesPerAcq}, "
                        f"timeout={DAHENG_BUFFER_TIMEOUT_MS}ms"
                    )
                    increment_profile("timeouts")
                    if (
                        self.BlinesPerAcq != CONTINUOUS
                        and consecutive_timeouts >= DAHENG_MAX_CONSECUTIVE_TIMEOUTS
                    ):
                        print(
                            "Daheng camera acquisition aborted after "
                            f"{consecutive_timeouts} consecutive buffer timeouts. "
                            "This usually means the camera missed one or more finite triggers."
                        )
                        acquisition_error_message = (
                            "Daheng camera acquisition aborted after "
                            f"{consecutive_timeouts} consecutive buffer timeouts; "
                            f"received {BlinesCount}/{self.BlinesPerAcq} frame(s)."
                        )
                        break
                    continue
                consecutive_timeouts = 0
                t_put = time.perf_counter()
                grab_q.put((buf, BlinesCount))
                add_profile("queue_put", time.perf_counter() - t_put)
                queue_depth = grab_q.qsize()
                with profile_lock:
                    profile["max_processing_queue"] = max(profile["max_processing_queue"], queue_depth)
                BlinesCount += 1
                increment_profile("grabbed")
                if self.ui.PauseButton.isChecked():
                    self.hcam.stream_off()
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(0.5)
                    self.hcam.stream_on()
        finally:
            for _ in workers:
                grab_q.put(grab_stop)
        for worker in workers:
            worker.join()
        if acquisition_error_message is not None:
            self.DatabackQueue.put(DbackActionField(None, error=acquisition_error_message))
        self.MemoryLoc = (start_memory_slot + next_block_to_emit[0]) % self.memoryCount
        total = time.perf_counter() - total_t0
        queue_put_fraction = profile["queue_put"] / max(total, 1e-9)
        if BlinesCount > 1000:
            print(
                "Daheng acquisition summary: \n"
                f"consumer_workers={worker_count}, \n"
                f"frames_grabbed={profile['grabbed']}, \n"
                f"frames_processed={profile['processed']}, \n"
                f"total_time={total:.3f}s, \n"
                f"camera_wait={profile['dq_buf']:.3f}s (longer better), \n"
                f"max_processing_queue_size={profile['max_processing_queue']}/128\n"
            )
        if profile["max_processing_queue"] > 100:
            print(
                "Daheng acquire warning: processing queue was nearly full \n"
                f"({profile['max_processing_queue']}/128). \n"
                "The conversion/write thread is close to falling behind the camera stream.\n"
            )
        if profile["max_processing_queue"] > 100 and queue_put_fraction > 0.05:
            print(
                "Daheng acquire warning: producer waited \n"
                f"{profile['queue_put']:.3f}s while handing frames to the conversion thread. \n"
                "The camera readout path is close to falling behind; consider reducing trigger rate \n"
                "or optimizing packed conversion / memory transpose-write.\n"
            )
        if consumer_error:
            raise RuntimeError("Acquire consumer failed:\n" + consumer_error[0])

    def write_bline_to_memory(self, bline, memory_slot, frame_index):
        dest = self.Memory[memory_slot][frame_index]
        if bline.shape == dest.shape:
            write_mode = "direct"
            dest[...] = bline
        elif bline.T.shape == dest.shape:
            write_mode = "transpose"
            dest[...] = bline.T
        else:
            raise ValueError(
                "Daheng frame shape does not match destination memory slice: "
                f"frame={bline.shape}, frame_T={bline.T.shape}, dest={dest.shape}"
            )

        if self._memory_write_mode is None:
            self._memory_write_mode = write_mode
            print(
                "Daheng memory write mode selected: "
                f"{write_mode} (frame={bline.shape}, dest={dest.shape})"
            )

    def Stream_on(self):
        if self.hcam is not None:
            t0 = time.perf_counter()
            self.hcam.stream_on()
            self._last_stream_on_elapsed = time.perf_counter() - t0
            print(f"Daheng stream_on completed in {self._last_stream_on_elapsed:.3f}s.")

    def Stream_off(self):
        if self.hcam is not None:
            self.hcam.stream_off() 
    
    def SetPixelDepth(self):
        if self.hcam is not None:
            self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
            pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
            self.ui.PixelFormat_display_DH.setText(pixelformat[1])
    
    def GetPixelDepth(self):
        if self.hcam is not None:
            pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
            self.ui.PixelFormat_display_DH.setText(pixelformat[1])

    # Set exposure time from the UI.
    def SetExposure(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("ExposureTime").set(self.ui.Exposure_DH.value()*1000.0)
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)
        
    # Read exposure time from the camera back into the UI.
    def GetExposure(self):
        if self.hcam is not None:
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)

    # Toggle auto-exposure.
    def AutoExposure(self):
        if self.hcam is not None:
            if self.ui.AutoExpo.isChecked():
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                self.ui.Exposure_DH.setValue(self.ui.Exposure_display_DH.value())
                
    def SetGain(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("Gain").set(self.ui.DGain_DH.value()*1.0)
            self.ui.DGain_display_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)
        
    # Read gain from the camera back into the UI.
    def GetGain(self):
        if self.hcam is not None:
           self.ui.DGain_display_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)

    # Toggle auto-gain.
    def AutoGain(self):
        if self.hcam is not None:
            if self.ui.AutoGain.isChecked():
                self.hcam_fr.get_enum_feature("GainAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                self.ui.DGain_DH.setValue(self.ui.DGain_display_DH.value())

    
    # Close the Daheng device.
    def Close(self):
        if self.hcam is not None:
            self.hcam.close_device()
            self.hcam = None
    
    def simData(self):
        
        # print('D using memory loc: ',self.MemoryLoc)
        # print(self.Memory[self.MemoryLoc].shape)
        NBlines = self.Memory[0].shape[0]
        # print(NBlines)
        # Number of frames written into the current memory slot.
        BlinesCount = 0
        self.DbackQueue.put(0)
        # print('start dbackqueue size:', self.DbackQueue.qsize())
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            # t0=time.time()
            
            if self.ui.PixelFormat_display_DH.text() in ['Mono8']:
                Bline = np.uint8(np.zeros([self.ui.AlinesPerBline.value(), effective_camera_sample_count(self.ui)]))
            else:
                Bline = np.uint16(np.zeros([self.ui.AlinesPerBline.value(), effective_camera_sample_count(self.ui)]))
            # print('camera outputs:', Bline[0,0:20])
            # print(BlinesCount, self.BlinesPerAcq)
            self.Memory[self.MemoryLoc][BlinesCount % NBlines] = Bline

            # print(BlinesCount % NBlines)
            BlinesCount += 1
            if BlinesCount % NBlines == 0:
                an_action = DbackActionField(self.MemoryLoc)
                self.DatabackQueue.put(an_action)
                self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount
                # print('MemoryLoc:', self.MemoryLoc)
                

            # handle pause action
            if self.ui.PauseButton.isChecked():
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
