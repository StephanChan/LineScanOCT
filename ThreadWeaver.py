# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:10:17 2024

@author: admin
"""

#################################################################
from PyQt5.QtCore import  QThread
import time
import numpy as np
from Generaic_functions import *
from ActionFields import DnSActionField, AODOActionField, GPUActionField, DActionField
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION, GPUActions, WeaverActions
import traceback
import os
import matplotlib.pyplot as plt
from queue import Empty
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
# from matplotlib.path import Path
# from scipy.signal import hilbert
import datetime
import cv2
from mosaic_scan_planner import (
    CENTER_MODE,
    FOV_OVERLAP,
    MAX_Y_FOV_MM,
    ROI_OCCUPANCY_TARGET,
    plan_mosaic_scan,
)
from mosaic_correction import (
    build_mosaic_correction_overlay_source,
    mosaic_polygons_to_stage_mm,
)
from SampleLocator import open_usb_camera, orient_usb_frame
from Display_rendering import (
    display_sample_overlay,
    mosaic_label_render_size,
    render_mosaic_correction_overlay,
    render_usb_roi_overlay,
)
from DynamicPostprocessing import (
    process_idle_dynamic_until_deadline,
    process_next_idle_dynamic_folder,
    update_timer_readout,
    write_stitched_idle_outputs,
)
from CameraUi import effective_camera_sample_count, camera_pixel_format
from ScanSession import (
    load_session_data,
    populate_sample_selector,
    save_session_data,
    save_usb_training_data,
)

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

CONTINUOUS_MODES = (
    AcqTypes.CONTINUOUS_ALINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.CONTINUOUS_CSCAN,
)

MOSAIC_DISPLAY_MODES = (
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)

SAVE_SAMPLE_TIME_MODES = (
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)

class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.overlay_images = {}
        self.FOV_locations = {}
        self.mosaic_roi_occupancy = ROI_OCCUPANCY_TARGET
        self.mosaic_fov_overlap = FOV_OVERLAP
        self.mosaic_max_y_fov_mm = MAX_Y_FOV_MM
        self.mosaic_center_mode = CENTER_MODE
        self.debug_mosaic_correction = False
        self._restore_y_geometry = None
        self.exit_message = 'Acquisition thread exited.'
        
    def run(self):
        if getattr(self, "file_naming", None) is None:
            raise RuntimeError("WeaverThread.file_naming must be assigned before starting the thread.")
        self.QueueOut()

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def finish_with_message(self, message):
        if message is None:
            return
        print(str(message))
        self.emit_status(message)
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            try:
                if self.item.action in (
                    AcqTypes.CONTINUOUS_ALINE,
                    AcqTypes.CONTINUOUS_BLINE,
                    AcqTypes.CONTINUOUS_CSCAN,
                ):
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    message = self.RptScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)
                    
                elif self.item.action in (
                    AcqTypes.FINITE_ALINE,
                    AcqTypes.FINITE_BLINE,
                    AcqTypes.FINITE_CSCAN,
                ):
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    message = self.SingleScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)
                elif self.item.action == AcqTypes.LOCATION_CAMERA_LIVE:
                    self.live()

                elif self.item.action == AcqTypes.PLATE_PRESCAN:
                    message = self.prepare_and_run_plate_prescan(acq_mode=self.item.action, context=self.item.context)
                    self.finish_with_message(message)
                elif self.item.action == AcqTypes.PLATE_SCAN:
                    # make directories
                    # if not os.path.exists(self.ui.DIR.toPlainText()+'/aip'):
                    #     os.mkdir(self.ui.DIR.toPlainText()+'/aip')
                    # if not os.path.exists(self.ui.DIR.toPlainText()+'/surf'):
                    #     os.mkdir(self.ui.DIR.toPlainText()+'/surf')
                    self.load_session_data(self.ui.DIR.toPlainText()+'/Mosaic')
                    """Updates the combo box content on the main thread."""
                    self.ui.sampleSelector.clear()
                    if len(self.sample_centers) == 0:
                        self.ui.sampleSelector.addItem("No Samples Found")
                    else:
                        for i in range(len(self.sample_centers)):
                            self.ui.sampleSelector.addItem(f"Sample {i+1}")
                    message = self.PlateScan(acq_mode=self.item.action, context=self.item.context)
                    self.finish_with_message(message)
                    if self.current_save_enabled():
                        self.increment_time_reader()
                elif self.item.action == AcqTypes.TIMED_PLATE_SCAN:
                    if not self.ensure_plate_scan_plan_loaded():
                        message = "Timed plate scan stopped: no saved scan plan was found."
                    else:
                        message = self.TimedPlateScan(acq_mode=self.item.action, context=self.item.context)
                    self.finish_with_message(message)
                elif self.item.action == AcqTypes.WELL_SCAN:
                    if self.has_plate_plan():
                        message = self.WellScan(acq_mode=self.item.action, context=self.item.context)
                    else:
                        message = "Well scan stopped: no in-memory scan plan is available."
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.ZSTAGE_REPEATIBILITY:
                    message = self.ZstageRepeatibility()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GOTO_ZERO:
                    message = self.Gotozero()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_BACKGROUND:
                    message = self.get_background_cscan()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_SURFACE:
                    message = self.get_surfCurve()
                    self.finish_with_message(message)
                    
                else:
                    message = f"Unknown acquisition command: {self.item.action}"
                    self.finish_with_message(message)

            except StopIteration:
                pass
            except Exception as error:
                message = f"Acquisition command failed: {self.item.action}"
                self.finish_with_message(message)
                print(traceback.format_exc())
            # reset RUN button
            self.wait_for_processing_barrier(
                label=f"finishing {self.item.action}",
                stop_if_run_unchecked=False,
            )
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText('Go')
            self.ui.PauseButton.setChecked(False)
            self.ui.PauseButton.setText('Pause')
            self.ui.RunButton.setEnabled(True)
            self.ui.PauseButton.setEnabled(True)
            self.GPUQueue.put(GPUActionField(GPUActions.CLEAR))
            if self.ui_bridge is not None:
                self.ui_bridge.acquisition_controls_locked.emit(False)
            # wait for next command
            self.item = self.queue.get()
        # exit weaver thread
        self.emit_status(self.exit_message)
            
    def drain_queue(self, queue, name, keep=None):
        """Remove queued items, optionally keeping items selected by keep(item)."""
        drained = 0
        kept = []
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            if keep is not None and keep(item):
                kept.append(item)
            else:
                drained += 1
        for item in kept:
            queue.put(item)
        if drained:
            message = f"Cleared {drained} stale item(s) from {name}."
            print(message)
        return drained

    def drain_continuous_backlog(self, reason=""):
        def keep_gpu_item(item):
            is_fft_action = getattr(item, 'action', None) in {GPUActions.GPU, GPUActions.CPU}
            is_continuous_mode = getattr(item, 'DnS_action', None) in CONTINUOUS_MODES
            return not (is_fft_action and is_continuous_mode)

        drained_gpu = self.drain_queue(self.GPUQueue, "GPUQueue", keep=keep_gpu_item)
        drained_camera = self.drain_queue(self.DatabackQueue, "DatabackQueue")
        if drained_gpu or drained_camera:
            suffix = f" ({reason})" if reason else ""
            print(
                "Continuous backlog drain complete"
                f"{suffix}: GPUQueue={drained_gpu}, DatabackQueue={drained_camera}"
            )

    def clear_mosaic_display(self):
        if getattr(self.ui, "mosaic_viewer", None) is not None:
            self.ui.mosaic_viewer.clear_image()

    def processing_backlog(self):
        gpu_pending = self.GPUQueue.qsize()
        gpu_active = getattr(getattr(self, "gpu_thread", None), "active_tasks", 0)
        dns_pending = self.DnSQueue.qsize()
        dns_active = getattr(getattr(self, "dns_thread", None), "active_tasks", 0)
        return gpu_pending, gpu_active, dns_pending, dns_active

    def current_acq_mode(self):
        return self.ui.ACQMode.currentText()

    def current_fft_device(self):
        return self.ui.FFTDevice.currentText()

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_dynamic_enabled(self):
        return self.ui.DynCheckBox.isChecked()

    def current_geometry(self):
        return None

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_alines_per_bline(self):
        return max(1, int(self.ui.AlinesPerBline.value()))

    def current_nsamples(self):
        return effective_camera_sample_count(self.ui)

    def current_depth_range(self):
        return int(self.ui.DepthRange.value())

    def current_realtime_dynamic_enabled(self):
        return self.current_dynamic_enabled() and self.ui.RealtimeDynCheckBox.isChecked()

    def current_pre_avg_factor(self):
        fft_device = self.current_fft_device()
        if fft_device not in ['GPU', 'CPU']:
            return 1
        if self.current_dynamic_enabled():
            gpu_thread = getattr(self, "gpu_thread", None)
            return max(1, int(getattr(gpu_thread, "gpu_pre_avg_factor", 1)))
        return self.current_bline_avg()

    def current_processed_repeat_count(self, raw_frame_count):
        pre_avg_factor = self.current_pre_avg_factor()
        if pre_avg_factor <= 1:
            return int(raw_frame_count)
        complete_frames = (int(raw_frame_count) // pre_avg_factor) * pre_avg_factor
        if complete_frames < pre_avg_factor:
            return int(raw_frame_count)
        return max(1, complete_frames // pre_avg_factor)

    def finalize_partial_dynamic_naming(self, acq_mode):
        if (
            not self.current_save_enabled()
            or not self.current_dynamic_enabled()
            or self.current_realtime_dynamic_enabled()
        ):
            return
        if self.file_naming.dynamic_bline_idx == 1:
            return
        if acq_mode in CSCAN_MODES:
            self.file_naming.increment_cscan()
            self.file_naming.reset_dynamic_bline_idx()
        elif acq_mode in MOSAIC_DISPLAY_MODES:
            self.file_naming.increment_tile()
            self.file_naming.reset_dynamic_bline_idx()

    def build_filename_bundle(self, DnS_action, acq_mode, memory_slot, raw=False):
        if not self.current_save_enabled():
            return {}
        if self.file_naming is None:
            raise RuntimeError("WeaverThread.file_naming is required when saving is enabled.")

        raw_shape = self.Memory[memory_slot].shape
        raw_frames = int(raw_shape[0])
        x_pixels = int(raw_shape[1])
        z_pixels = self.current_nsamples() if raw else self.current_depth_range()
        repeat_count = raw_frames if raw else self.current_processed_repeat_count(raw_frames)
        y_pixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.file_naming.dynamic_bline_idx)
        bundle = {}

        if acq_mode in ALINE_MODES:
            filename = self.file_naming.get_filename("aline", acq_mode, [repeat_count, x_pixels, z_pixels])
            bundle = {"filename": filename, "log_filename": filename}
            self.file_naming.increment_aline()
            return bundle

        if acq_mode in BLINE_MODES:
            filename = self.file_naming.get_filename("bline", acq_mode, [repeat_count, x_pixels, z_pixels])
            bundle = {"filename": filename, "log_filename": filename}
            if self.current_realtime_dynamic_enabled():
                bundle["dynamic_filename"] = self.file_naming.get_filename(
                    "bline_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                )
            self.file_naming.increment_bline()
            return bundle

        if acq_mode in CSCAN_MODES:
            if self.current_dynamic_enabled():
                if self.current_realtime_dynamic_enabled():
                    dynamic_filename = self.file_naming.get_filename(
                        "cscan_dyn",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                        ypixels=y_pixels,
                    )
                    mean_filename = self.file_naming.get_filename(
                        "cscan_mean",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    bundle = {
                        "dynamic_filename": dynamic_filename,
                        "mean_filename": mean_filename,
                        "log_filename": dynamic_filename,
                    }
                    if dynamic_bline_idx == y_pixels:
                        self.file_naming.increment_cscan()
                        self.file_naming.reset_dynamic_bline_idx()
                    else:
                        self.file_naming.increment_dynY()
                    return bundle

                bline_filename = self.file_naming.get_filename(
                    "cscan_bline",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                    ypixels=y_pixels,
                )
                dyn_filename = self.file_naming.get_filename(
                    "cscan_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                    ypixels=y_pixels,
                )
                bundle = {
                    "filename": bline_filename,
                    "dynamic_filename": dyn_filename,
                    "log_filename": bline_filename,
                }
                self.file_naming.advance_cscan_dynamic_bline(y_pixels)
                return bundle

            filename = self.file_naming.get_filename("cscan", acq_mode, [y_pixels, x_pixels, z_pixels])
            self.file_naming.increment_cscan()
            return {"filename": filename, "log_filename": filename}

        if acq_mode in MOSAIC_DISPLAY_MODES:
            if self.current_dynamic_enabled():
                if self.current_realtime_dynamic_enabled():
                    dynamic_filename = self.file_naming.get_filename(
                        "tile_dyn",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    mean_filename = self.file_naming.get_filename(
                        "tile_mean",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    bundle = {
                        "dynamic_filename": dynamic_filename,
                        "mean_filename": mean_filename,
                        "log_filename": dynamic_filename,
                    }
                    if dynamic_bline_idx == y_pixels:
                        self.file_naming.increment_tile()
                        self.file_naming.reset_dynamic_bline_idx()
                    else:
                        self.file_naming.increment_dynY()
                    return bundle

                filename = self.file_naming.get_filename(
                    "sample_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                )
                self.file_naming.advance_tile_dynamic_bline(y_pixels)
                return {"filename": filename, "log_filename": filename}

            filename = self.file_naming.get_filename("sample", acq_mode, [y_pixels, x_pixels, z_pixels])
            self.file_naming.increment_tile()
            return {"filename": filename, "log_filename": filename}

        return {}

    def wait_for_processing_barrier(self, label="", poll_interval=0.2, stop_if_run_unchecked=True):
        label = label or "the next sample"
        while self.ui.RunButton.isChecked() or not stop_if_run_unchecked:
            gpu_pending, gpu_active, dns_pending, dns_active = self.processing_backlog()
            if gpu_pending <= 0 and gpu_active <= 0 and dns_pending <= 0 and dns_active <= 0:
                return True
            self.emit_status(
                "Waiting for processing of the previous sample before "
                f"{label}. GPU queue={gpu_pending}, GPU active={gpu_active}, "
                f"DnS queue={dns_pending}, DnS active={dns_active}."
            )
            time.sleep(poll_interval)
            if stop_if_run_unchecked and not self.ui.RunButton.isChecked():
                return False
        return False
        
    
    def InitMemory(self):
        #################################################################
        # get number samplers per Aline
        samples = self.current_nsamples()
        alines_per_bline = self.current_alines_per_bline()
        configured_bline_avg = self.current_bline_avg()
        configured_y_pixels = self.current_y_pixels()
        configured_dynamic = self.current_dynamic_enabled()
        configured_acq_mode = self.current_acq_mode()
        # print(self.ui.PixelFormat_display.text())
        if camera_pixel_format(self.ui) in ['Mono8']:
            data_type =  np.uint8
        else:
            data_type =  np.uint16
            
        for ii in range(self.memoryCount):
             if configured_acq_mode in (
                 AcqTypes.FINITE_ALINE,
                 AcqTypes.CONTINUOUS_ALINE,
                 AcqTypes.FINITE_BLINE,
                 AcqTypes.CONTINUOUS_BLINE,
             ):
                 self.Memory[ii]=np.zeros([configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                 self.NAcq = 1
             elif configured_acq_mode == AcqTypes.CONTINUOUS_CSCAN:
                 self.Memory[ii]=np.zeros([configured_y_pixels*configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                 self.NAcq = 1
             elif configured_acq_mode in (
                 AcqTypes.FINITE_CSCAN,
                 AcqTypes.PLATE_PRESCAN,
                 AcqTypes.PLATE_SCAN,
                 AcqTypes.WELL_SCAN,
                 AcqTypes.TIMED_PLATE_SCAN,
             ):
                 if configured_dynamic:
                     self.Memory[ii]=np.zeros([configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                     self.NAcq = configured_y_pixels
                 else:
                     self.Memory[ii]=np.zeros([configured_y_pixels*configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                     self.NAcq = 1

        ###########################################################################################
        
    def SingleScan(self, DnS_action, acq_mode, context=None, skip_save=False):
        # an_action = DnSActionField(DnSActions.CLEAR)
        # self.DnSQueue.put(an_action)
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        self.drain_continuous_backlog(reason=f"before {DnS_action}")
        fft_device = self.current_fft_device()
        t0=time.time()
        # print(self.DbackQueue.qsize())
        an_action = DActionField('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        t1=time.time()
        ###########################################################################################
        # start AODO 
        an_action = AODOActionField('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        t2=time.time()
        # start camera

        an_action = DActionField('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
        t3=time.time()

        # print('current dbackqueue size:', self.DbackQueue.qsize())
        an_action = AODOActionField('StartTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        t4=time.time()
        # print('\n')
        # print('Camera config took: ',round(t1-t0,3),'sec')
        print('Galvo board config took: ',round(t2-t1,3),'sec')
        # print('Camera start took: ',round(t3-t2,3),'sec')
        # print('Galvo board start took: ',round(t4-t3,3),'sec')
        # print('current dbackqueue size:', self.DbackQueue.qsize())
        # print('\n')
        message = f"{DnS_action} stopped by user."
        for iAcq in range(self.NAcq):
            start = time.time()
            dynamic_bline_idx = iAcq if (self.current_dynamic_enabled() and acq_mode in (CSCAN_MODES + MOSAIC_DISPLAY_MODES)) else None
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            # print('waiting for camera data...')
            while self.ui.RunButton.isChecked():
                try:
                    an_action = self.DatabackQueue.get(timeout = 2)
                    # print('camera queue size:', self.DatabackQueue.qsize())
                    # print('time to fetch data: '+str(round(time.time()-start,3))+'sec')
                    memory_slot = an_action.memory_slot
                    filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=(fft_device in ['None']))
                    # print(memory_slot)
                    ############################################### display and save data
                    if fft_device in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        self.data = self.Memory[memory_slot].copy()
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            message = "No usable spectral data received."
                            print(message)
                        else:
                            an_action = DnSActionField(
                                DnS_action,
                                acq_mode=acq_mode,
                                data=self.data,
                                raw=True,
                                context=context,
                                dynamic_bline_idx=dynamic_bline_idx,
                                filename_bundle=filename_bundle,
                                skip_save=skip_save,
                            )
                            self.DnSQueue.put(an_action)
                            message = f"{DnS_action} completed."
                    else:
                        # In other modes, do FFT first
                        an_action = GPUActionField(
                            action=fft_device,
                            DnS_action=DnS_action,
                            acq_mode=acq_mode,
                            memory_slot=memory_slot,
                            context=context,
                            dynamic_bline_idx=dynamic_bline_idx,
                            filename_bundle=filename_bundle,
                            skip_save=skip_save,
                        )
                        self.GPUQueue.put(an_action)
                        message = f"{DnS_action} completed."
                    break
                except:
                    print(f"{DnS_action}: waiting for camera data...")
                    
        an_action = AODOActionField('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOActionField('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        self.finalize_partial_dynamic_naming(acq_mode)
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        print(message)
        return message
    
            
    def RptScan(self, DnS_action, acq_mode):
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        fft_device = self.current_fft_device()
        # an_action = DnSActionField(DnSActions.CLEAR)
        # self.DnSQueue.put(an_action)
        frame_rate = self.ui.FrameRate_DH.value()
        if acq_mode in tuple(mode for mode in ALINE_MODES + BLINE_MODES if mode in CONTINUOUS_MODES):
            self.ui.FrameRate_DH.setValue(20)
        an_action = DActionField('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        # config AODO
        an_action = AODOActionField('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        data_backs = 0 # count number of data backs
        skipped_fft_actions = 0

        # start digitizer for one acuquqisition
        an_action = DActionField('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()

        # start AODO 
        an_action = AODOActionField('StartTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()

        ######################################################### repeat acquisition until Stop button is clicked
        while self.ui.RunButton.isChecked():
            ######################################### collect data
            try: # use try-except in cases where Stop button clicked and camera stopped prior to while loop
                start = time.time()
                an_action = self.DatabackQueue.get(timeout=2) # never time out
                # print('time to fetch data: '+str(round(time.time()-start,3)))
                memory_slot = an_action.memory_slot
                # print(memory_slot)
                data_backs += 1
                
                ######################################### display data
                if fft_device in ['None']:
                    filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=True)
                    # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                    self.data = self.Memory[memory_slot].copy()
                    # In None mode, directly do display and save
                    if np.sum(self.data)<10:
                        message = "No usable spectral data received."
                        print(message)
                    else:
                        an_action = DnSActionField(
                            DnS_action,
                            acq_mode=acq_mode,
                            data=self.data,
                            raw=True,
                            filename_bundle=filename_bundle,
                            skip_save=False,
                        )
                        self.DnSQueue.put(an_action)
                        message = f"{DnS_action} completed."
                else:
                    # In other modes, do FFT first
                    if self.GPUQueue.qsize() == 0:
                        filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=False)
                        an_action = GPUActionField(
                            fft_device,
                            DnS_action=DnS_action,
                            acq_mode=acq_mode,
                            memory_slot=memory_slot,
                            filename_bundle=filename_bundle,
                            skip_save=False,
                        )
                        self.GPUQueue.put(an_action)
                    else:
                        skipped_fft_actions += 1
                    message = f"{DnS_action} completed."
                ######################################## check if Pause or Stop button is clicked
            except:
                pass
                # print('camera stopped')
            # handle pause action
            if self.ui.PauseButton.isChecked():
                # camera will wait for trigger, no need to stop
                # stop AODO task, can be restarted
                an_action = AODOActionField('StopTask')
                self.AODOQueue.put(an_action)
                # wait until stop button or pause button is clicked
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                # if resume, restart AODO task
                if not self.ui.PauseButton.isChecked():
                    # start AODO 
                    an_action = AODOActionField('StartTask')
                    self.AODOQueue.put(an_action)
                    self.StagebackQueue.get()
        # Camera will stop once Stop Button is clicked
        # AODO thread will need StopTask command
        an_action = AODOActionField('tryStopTask')
        self.AODOQueue.put(an_action)
        # close AODO
        an_action = AODOActionField('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        # digitizer will close automatically
        self.finalize_partial_dynamic_naming(acq_mode)
        message = f"{DnS_action} stopped. Received {data_backs} camera buffer(s)."
        if skipped_fft_actions:
            message += f" Skipped {skipped_fft_actions} stale continuous FFT request(s)."
        print(message)
        self.drain_continuous_backlog(reason=f"after {DnS_action}")
        an_action = GPUActionField(GPUActions.DISPLAY_FFT_ACTIONS)
        self.GPUQueue.put(an_action)
        an_action = GPUActionField(GPUActions.DISPLAY_COUNTS, context=DnS_action)
        self.GPUQueue.put(an_action)
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        self.ui.FrameRate_DH.setValue(frame_rate)
        return message
  

    def prepare_and_run_plate_prescan(self, acq_mode, context=None):
        mosaic_folder = self.ui.DIR.toPlainText() + '/Mosaic'

        if context:
            if not os.path.exists(mosaic_folder):
                os.mkdir(mosaic_folder)
            return self.PlatePreScan(acq_mode=acq_mode, context=context)

        if not self.has_plate_plan():
            metadata_path = os.path.join(mosaic_folder, 'scan_metadata.pkl')
            if not os.path.exists(metadata_path):
                message = "Plate pre-scan needs sample FOVs. Locate samples first or select a Mosaic folder with scan_metadata.pkl."
                print(message)
                self.ui.RunButton.setChecked(False)
                self.ui.RunButton.setText('Go')
                return message
            self.load_session_data(mosaic_folder)
            self.update_sample_selector_from_plan()

        if not self.has_plate_plan():
            message = "Plate pre-scan stopped: no sample FOVs were found in memory or in the current Mosaic folder."
            print(message)
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText('Go')
            return message

        if not os.path.exists(mosaic_folder):
            os.mkdir(mosaic_folder)
        return self.PlatePreScan(acq_mode=acq_mode)

    def ensure_plate_scan_plan_loaded(self):
        if self.FOV_locations not in ({}, [], None) and self.sample_centers not in ({}, [], None):
            if len(self.sample_centers) > 0:
                self.update_sample_selector_from_plan()
                return True

        mosaic_folder = self.ui.DIR.toPlainText() + '/Mosaic'
        metadata_path = os.path.join(mosaic_folder, 'scan_metadata.pkl')
        if not os.path.exists(metadata_path):
            return False
        self.load_session_data(mosaic_folder)
        self.update_sample_selector_from_plan()
        return len(self.sample_centers) > 0

    def has_plate_plan(self):
        return (
            isinstance(self.FOV_locations, list)
            and isinstance(self.sample_centers, list)
            and len(self.FOV_locations) > 0
            and len(self.sample_centers) > 0
        )

    def update_sample_selector_from_plan(self):
        populate_sample_selector(self.ui, self.sample_centers)

    def PlatePreScan(self, acq_mode, context=None):
        fresh_locator_data = bool(context)
        if fresh_locator_data:
            self.overlay_images = {}
            self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= context
        if self.sample_centers is None or len(self.sample_centers) == 0:
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText('Go')
            return "Plate pre-scan stopped: no sample centers are available."
        self.ui.FFTDevice.setCurrentText('GPU')
        BlineAVG = self.ui.BlineAVG.value()
        self.ui.BlineAVG.setValue(1)
        self.ui.RunButton.setChecked(True)
        for sample_center in self.sample_centers:
            if self.ui.RunButton.isChecked():
                self.ui.NextSampleButton.setText('扫描中，请等待')
                self.ui.RepeatSampleButton.setText('扫描中，请等待')
                barrier_sample_id = max(1, sample_center.sample_id - 1)
                if not self.wait_for_processing_barrier(
                    label=f"sampleID-{barrier_sample_id} pre-scan"
                ):
                    return "Plate pre-scan stopped by user."
                self.emit_status(f"Scanning sampleID-{sample_center.sample_id} pre-scan.")
                self.CurrentSampleLocations = [
                    location
                    for location in self.FOV_locations
                    if location.sample_id == sample_center.sample_id
                ]
                print(f"PlatePreScan start sampleID-{sample_center.sample_id} FOV XYZ:")
                for idx, location in enumerate(self.CurrentSampleLocations, start=1):
                    print(
                        f"  FOV {idx}: X={location.x:.4f}, Y={location.y:.4f}, Z={location.z:.4f}"
                    )
                if fresh_locator_data:
                    self.display_initial_scan_overlay(sample_center.sample_id, self.raw_img, self.pixel_polygons)
                else:
                    self.display_sample_overlay(sample_center.sample_id)
                
                # User stopped continuousBline, then we do Mosaic scan for this sample
                self.AdjustZstage(sample_center.sample_id)
    
                message = self.iterate_FOVs(acq_mode=acq_mode)
                if not self.wait_for_processing_barrier(
                    label=f"finishing sampleID-{sample_center.sample_id} pre-scan",
                    stop_if_run_unchecked=False,
                ):
                    return "Plate pre-scan stopped by user."
                self.ui.NextSampleButton.setText('下一个样品')
                self.ui.RepeatSampleButton.setText('重新扫描')
                while (not self.ui.NextSampleButton.isChecked()) and self.ui.RunButton.isChecked():
                    if self.ui.RepeatSampleButton.isChecked():
                        self.ui.NextSampleButton.setText('扫描中，请等待')
                        self.ui.RepeatSampleButton.setText('扫描中，请等待')
                        correction_applied = self.process_mosaic_correction()
                        if not correction_applied:
                            self.CurrentSampleLocations = [
                                location
                                for location in self.FOV_locations
                                if location.sample_id == sample_center.sample_id
                            ]
                        self.AdjustZstage(sample_center.sample_id)
                        message = self.iterate_FOVs(acq_mode=acq_mode)
                        if not self.wait_for_processing_barrier(
                            label=f"finishing sampleID-{sample_center.sample_id} repeat pre-scan",
                            stop_if_run_unchecked=False,
                        ):
                            return "Plate pre-scan stopped by user."
                        self.ui.NextSampleButton.setText('下一个样品')
                        self.ui.RepeatSampleButton.setText('重新扫描')
                        self.ui.RepeatSampleButton.setChecked(False)
                    time.sleep(1)
                        
                self.ui.NextSampleButton.setChecked(False)
                self.ui.sampleSelector.setCurrentIndex(self.ui.sampleSelector.currentIndex() + 1)
                # 1. Remove all old entries matching this sample_id
                # We keep everything that DOES NOT match the ID we are updating
                lower_id_locations = [
                    location
                    for location in self.FOV_locations
                    if location.sample_id < sample_center.sample_id
                ]
                
                higher_id_locations = [
                    location
                    for location in self.FOV_locations
                    if location.sample_id > sample_center.sample_id
                ]
            
                # 2. Combine them back together
                self.FOV_locations = lower_id_locations + self.CurrentSampleLocations + higher_id_locations
                saved_sample_locations = [
                    location
                    for location in self.FOV_locations
                    if location.sample_id == sample_center.sample_id
                ]
                print(f"PlatePreScan end sampleID-{sample_center.sample_id} FOV XYZ saved to memory:")
                for idx, location in enumerate(saved_sample_locations, start=1):
                    print(
                        f"  FOV {idx}: X={location.x:.4f}, Y={location.y:.4f}, Z={location.z:.4f}"
                    )
                
        
        # save self.FOV_locations, self.sample_centers, self.overlay_images
        self.save_session_data(self.ui.DIR.toPlainText()+'/Mosaic')
        self.ui.NextSampleButton.setText('扫描结束')
        self.ui.RepeatSampleButton.setText('扫描结束')
        self.ui.BlineAVG.setValue(BlineAVG)
        return(message)
            
    def PlateScan(self, acq_mode, context):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= context
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        # print(self.sample_centers)
        # print(self.FOV_locations)
        for sample_center in self.sample_centers:
            if self.ui.RunButton.isChecked():
                barrier_sample_id = max(1, sample_center.sample_id - 1)
                if not self.wait_for_processing_barrier(
                    label=f"sampleID-{barrier_sample_id} plate scan"
                ):
                    return "Plate scan stopped by user."
                self.emit_status(f"Scanning sampleID-{sample_center.sample_id} plate scan.")
                self.ui.sampleSelector.setCurrentIndex(sample_center.sample_id - 1)
                self.CurrentSampleLocations = [
                    location
                    for location in self.FOV_locations
                    if location.sample_id == sample_center.sample_id
                ]
                # print('self.CurrentSampleLocations', self.CurrentSampleLocations)
                self.display_sample_overlay(sample_center.sample_id)
                self.iterate_FOVs(acq_mode=acq_mode)
                if not self.wait_for_processing_barrier(
                    label=f"finishing sampleID-{sample_center.sample_id} plate scan",
                    stop_if_run_unchecked=False,
                ):
                    return "Plate scan stopped by user."
                self.update_timer_readout(getattr(self, "_timed_plate_deadline", None))
                message = "Plate scan completed."
            else:
                message = "Plate scan stopped by user."
        return(message)   

    def TimedPlateScan(self, acq_mode, context):
        interval_hours = float(self.ui.Timer.value())
        interval_hours = max(0.0, interval_hours)
        interval_seconds = interval_hours * 3600.0
        total_slices = int(self.ui.SliceTotal.value())
        current_slice = int(self.ui.CuSlice.value())
        total_slices = max(1, total_slices)
        current_slice = max(1, current_slice)

        if current_slice > total_slices:
            self._timed_plate_deadline = None
            self.update_timer_readout(None)
            return f"Timed plate scan finished: current time index {current_slice} exceeds SliceTotal {total_slices}."

        final_message = "Timed plate scan stopped by user."
        while self.ui.RunButton.isChecked() and current_slice <= total_slices:
            session_start = time.time()
            if interval_seconds > 0:
                self._timed_plate_deadline = session_start + interval_seconds
                self.update_timer_readout(self._timed_plate_deadline)
            else:
                self._timed_plate_deadline = None
                self.update_timer_readout(None)

            message = self.PlateScan(acq_mode=acq_mode, context=context)
            final_message = message
            if not self.ui.RunButton.isChecked():
                break

            if current_slice >= total_slices:
                final_message = f"Timed plate scan finished after {total_slices} time point(s)."
                break

            if self._timed_plate_deadline is not None:
                if not self.wait_for_processing_barrier(label="offline dynamic processing"):
                    break
                final_message = self.process_idle_dynamic_until_deadline(self._timed_plate_deadline, final_message)
                if not self.ui.RunButton.isChecked():
                    break

                while self.ui.RunButton.isChecked() and time.time() < self._timed_plate_deadline:
                    self.update_timer_readout(self._timed_plate_deadline)
                    time.sleep(min(60.0, self._timed_plate_deadline - time.time()))

            current_slice += 1
            self.set_time_reader_value(current_slice)

        self._timed_plate_deadline = None
        self.update_timer_readout(None)
        return final_message
    
    
    def WellScan(self, acq_mode, context):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= context
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        selected_index = self.ui.sampleSelector.currentIndex()
        requested_sample_id = max(1, selected_index + 1)
        self.emit_status(f"Scanning sampleID-{requested_sample_id} well scan.")
        self.CurrentSampleLocations = [
            location for location in self.FOV_locations if location.sample_id == requested_sample_id
        ]
        if not self.CurrentSampleLocations:
            return f"Well scan stopped: no FOV locations were found for sampleID-{requested_sample_id}."
        self.display_sample_overlay(requested_sample_id)
        if not self.wait_for_processing_barrier(label=f"starting sampleID-{requested_sample_id} well scan"):
            return "Well scan stopped by user."
        message = self.iterate_FOVs(acq_mode=acq_mode)
        if not self.wait_for_processing_barrier(
            label=f"finishing sampleID-{requested_sample_id} well scan",
            stop_if_run_unchecked=False,
        ):
            return "Well scan stopped by user."
        return(message) 

    def AdjustZstage(self, sample_id):
        sample_center = self.sample_centers[sample_id-1]
        # move to center position of this sample
        self.move_stage_axis('X', sample_center.x)
        self.move_stage_axis('Y', sample_center.y)
        self.move_stage_axis('Z', sample_center.z)
        # do continuous scan to display Bline
        self.ui.ACQMode.setCurrentText(AcqTypes.CONTINUOUS_BLINE)
        if not self.wait_for_processing_barrier(label=f"starting {AcqTypes.CONTINUOUS_BLINE}"):
            return
        self.InitMemory()
        self.ui.RunButton.setChecked(True)
        self.ui.RunButton.setText('点击开始扫描')
        self.RptScan(DnS_action=AcqTypes.CONTINUOUS_BLINE, acq_mode=AcqTypes.CONTINUOUS_BLINE)
        # User can move Z stage up and down to put sample at focus
        for location in self.CurrentSampleLocations:
            location.z = self.ui.ZPosition.value()
        
        self.ui.RunButton.setText('Stop')
        self.ui.RunButton.setChecked(True)
        
    def iterate_FOVs(self, acq_mode):
        # print(self.CurrentSampleLocations)
        if not self.wait_for_processing_barrier(label=f"starting {acq_mode}"):
            return f"{acq_mode} stopped by user."
        self.drain_continuous_backlog(reason=f"before {acq_mode}")
        self.apply_y_geometry_from_locations()
        # move to position of this FOV
        first_fov_location = self.CurrentSampleLocations[0]
        self.move_stage_axis('X', first_fov_location.x)
        self.move_stage_axis('Y', first_fov_location.y)
        self.move_stage_axis('Z', first_fov_location.z)
        self.clear_mosaic_display()
        
        aline_avg = self.ui.AlineAVG.value()
        x_length = self.ui.XLength.value()
        y_length = self.ui.YLength.value()
        Xpixels = self.current_alines_per_bline()//max(1, int(aline_avg))
        Ypixels = self.current_y_pixels()
        XFOV = x_length
        YFOV = y_length
        an_action = GPUActionField(GPUActions.INIT_MOSAIC, context=[self.CurrentSampleLocations, (Xpixels, Ypixels), (XFOV, YFOV)])
        self.GPUQueue.put(an_action)
        self.ui.ACQMode.setCurrentText(acq_mode)

        for fov_location in self.CurrentSampleLocations:
            if self.ui.RunButton.isChecked():
                # print(fov_location.x, fov_location.y)
                # move to position of this FOV
                self.move_stage_axis('X', fov_location.x)
                self.move_stage_axis('Y', fov_location.y)
                self.move_stage_axis('Z', fov_location.z)
                
                self.get_background_cscan()
                self.InitMemory()
                # do FiniteCscan at this position
                self.SingleScan(DnS_action=DnSActions.PROCESS_MOSAIC, acq_mode=acq_mode, context=[self.CurrentSampleLocations, fov_location])
                if not self.wait_for_processing_barrier(
                    label=f"finishing FOV ({fov_location.x:.3f}, {fov_location.y:.3f})",
                    stop_if_run_unchecked=False,
                ):
                    return "Sample FOV scan stopped by user."
                # handle pause action
                if self.ui.PauseButton.isChecked():
                    # wait until stop button or pause button is clicked
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(1)
                        print('waiting')
                message = "Sample FOV scan completed."
            else:
                message = "Sample FOV scan stopped by user."
        # Home X and Y stages here?
        self.wait_for_processing_barrier(label=f"finishing {acq_mode}", stop_if_run_unchecked=False)
        return(message)

    def update_timer_readout(self, deadline):
        return update_timer_readout(self.ui, deadline)

    def set_time_reader_value(self, value):
        value = int(value)
        if self.ui_bridge is not None:
            self.ui_bridge.time_reader_value.emit(value)
            self.ui_bridge.cu_slice_value.emit(value)

    def increment_time_reader(self):
        if hasattr(self.ui, "timeReader"):
            next_value = int(self.ui.timeReader.value()) + 1
        else:
            next_value = int(self.ui.CuSlice.value()) + 1
        self.set_time_reader_value(next_value)

    def process_idle_dynamic_until_deadline(self, deadline, current_message):
        return process_idle_dynamic_until_deadline(self, deadline, current_message)

    def process_next_idle_dynamic_folder(self, deadline):
        return process_next_idle_dynamic_folder(self, deadline)

    def write_stitched_idle_outputs(self, sample_id, folder_path, tile_count):
        return write_stitched_idle_outputs(self, sample_id, folder_path, tile_count)

    def move_stage_axis(self, axis, target, tolerance=0.005):
        position_widget = getattr(self.ui, f"{axis}Position")
        current_widget = getattr(self.ui, f"{axis}current")
        position_widget.setValue(target)

        current = current_widget.value()
        distance = target - current
        if abs(distance) <= tolerance:
            return

        an_action = AODOActionField(f"{axis}move2")
        self.AODOQueue.put(an_action)
        timeout = self.stage_move_timeout(axis, distance)
        try:
            self.StagebackQueue.get(timeout=timeout)
        except Empty:
            message = (
                f"Stage move timeout during {axis} move: "
                f"target={target:.4f}, current={current:.4f}, "
                f"distance={distance:.4f}, timeout={timeout:.1f}s."
            )
            print(message)
            self.emit_status(message + " Assuming motion completed and continuing.")
            return False
        return True

    def stage_move_timeout(self, axis, distance):
        speed_widget = getattr(self.ui, f"{axis}Speed", None)
        try:
            speed = float(speed_widget.value()) if speed_widget is not None else 1.0
        except Exception:
            speed = 1.0
        speed = max(abs(speed), 0.001)
        return max(20.0, abs(distance) / speed * 10.0 + 10.0)

    def sample_fov_locations(self, sample_id):
        if getattr(self, "CurrentSampleLocations", None):
            current_ids = {location.sample_id for location in self.CurrentSampleLocations}
            if sample_id in current_ids:
                return [
                    location
                    for location in self.CurrentSampleLocations
                    if location.sample_id == sample_id
                ]
        return [location for location in self.FOV_locations if location.sample_id == sample_id]

    def display_initial_scan_overlay(self, sample_id, raw_img, pixel_polygons):
        """Stores USB overlay source data and renders it at the current label size."""
        self.overlay_images[sample_id] = {
            'type': 'usb_roi',
            'raw_img': raw_img,
            'pixel_polygons': pixel_polygons,
        }
        self.display_sample_overlay(sample_id)

    def display_sample_overlay(self, sample_id):
        display_sample_overlay(self.ui, self.overlay_images, sample_id, self.sample_fov_locations)

    def mosaic_label_render_size(self):
        return mosaic_label_render_size(self.ui.MosaicLabel)

    def render_usb_roi_overlay(self, sample_id, raw_img, pixel_polygons):
        render_usb_roi_overlay(self.ui, sample_id, raw_img, pixel_polygons, self.sample_fov_locations)

    def y_pixels_from_length(self, y_length_mm):
        y_step_um = max(float(self.ui.YStepSize.value()), 1e-6)
        return max(1, int(np.round(float(y_length_mm) * 1000.0 / y_step_um)))

    def apply_y_geometry_for_correction(self, y_length_mm, y_pixels=None):
        computed_y_pixels = self.y_pixels_from_length(y_length_mm)
        if self._restore_y_geometry is None:
            self._restore_y_geometry = {
                "YLength": self.ui.YLength.value(),
                "Ypixels": self.ui.Ypixels.value(),
            }
        if self.debug_mosaic_correction:
            print(
                "Mosaic correction Y geometry apply: "
                f"YLength {self.ui.YLength.value():.3f} -> {y_length_mm:.3f}, "
                f"Ypixels {self.ui.Ypixels.value()} -> {computed_y_pixels}"
            )
        self.ui.YLength.setValue(float(y_length_mm))
        self.ui.Ypixels.setValue(int(computed_y_pixels))

    def restore_y_geometry_after_correction(self):
        if self._restore_y_geometry is None:
            return
        y_length = self._restore_y_geometry["YLength"]
        y_pixels = self._restore_y_geometry["Ypixels"]
        if self.debug_mosaic_correction:
            print(
                "Mosaic correction Y geometry restore: "
                f"YLength {self.ui.YLength.value():.3f} -> {y_length:.3f}, "
                f"Ypixels {self.ui.Ypixels.value()} -> {y_pixels}"
            )
        self.ui.YLength.setValue(float(y_length))
        self.ui.Ypixels.setValue(int(y_pixels))
        self._restore_y_geometry = None

    def apply_y_geometry_from_locations(self):
        if not self.CurrentSampleLocations:
            return
        first_fov_location = self.CurrentSampleLocations[0]
        y_length = first_fov_location.y_length_mm
        if y_length is None:
            return
        self.apply_y_geometry_for_correction(float(y_length))

    def current_location_y_length(self):
        if self.CurrentSampleLocations:
            y_length = self.CurrentSampleLocations[0].y_length_mm
            if y_length is not None:
                return float(y_length)
        return self.ui.YLength.value()

    def process_mosaic_correction(self):
        """Called when user finishes drawing in XYPlane/InteractiveWidget."""
        # Assume this is triggered for the currently active sample_id
        current_id = self.ui.sampleSelector.currentIndex() + 1 
        self.ui.mosaic_viewer.finalize_polygon()
        # Get new regions from the interactive widget
        new_polygons = self.ui.mosaic_viewer.polygons
        if not new_polygons:
            print('No regions draw, please re-draw interested region')
            return False

        # Convert the interactive widget polygons back to mm coordinates
        source_y_length = self.current_location_y_length()
        correction_geometry = mosaic_polygons_to_stage_mm(
            raw_polygons=new_polygons,
            current_fov_locations=self.CurrentSampleLocations,
            x_fov_mm=self.ui.XLength.value(),
            source_y_length_mm=source_y_length,
            x_step_um=self.ui.XStepSize.value(),
            y_step_um=self.ui.YStepSize.value(),
        )
        mm_polygons = correction_geometry["mm_polygons"]
        px_w_mm = correction_geometry["px_w_mm"]
        px_h_mm = correction_geometry["px_h_mm"]
        v_anchor_x, v_anchor_y = correction_geometry["anchor"]

        viewer = self.ui.mosaic_viewer
        if self.debug_mosaic_correction and hasattr(viewer, "adj"):
            print(
                "Mosaic correction input: "
                f"sample_id={current_id}, mosaic_shape={viewer.adj.shape}, "
                f"pixel_aspect_ratio={getattr(viewer, 'pixel_aspect_ratio', None)}, "
                f"px_w_mm={px_w_mm:.6g}, px_h_mm={px_h_mm:.6g}, "
                f"anchor=({v_anchor_x:.3f}, {v_anchor_y:.3f}), source_YLength={source_y_length:.3f}, "
                f"current_fovs={len(self.CurrentSampleLocations)}"
            )

        for ii, poly_debug in enumerate(correction_geometry["polygon_debug"], start=1):
            if self.debug_mosaic_correction:
                raw_bounds = poly_debug["raw_bounds"]
                raw_size = poly_debug["raw_size"]
                mm_bounds = poly_debug["mm_bounds"]
                mm_size = poly_debug["mm_size"]
                print(
                    "Mosaic correction polygon: "
                    f"#{ii}, vertices={poly_debug['vertices']}, "
                    f"raw_bounds=(x:{raw_bounds[0]:.2f}-{raw_bounds[2]:.2f}, "
                    f"y:{raw_bounds[1]:.2f}-{raw_bounds[3]:.2f}), "
                    f"raw_size=({raw_size[0]:.2f}, {raw_size[1]:.2f}), "
                    f"mm_bounds=(x:{mm_bounds[0]:.3f}-{mm_bounds[2]:.3f}, "
                    f"y:{mm_bounds[1]:.3f}-{mm_bounds[3]:.3f}), "
                    f"mm_size=({mm_size[0]:.3f}, {mm_size[1]:.3f})"
                )

        scan_plan = plan_mosaic_scan(
            sample_id=current_id,
            mm_polygons=mm_polygons,
            x_fov_mm=self.ui.XLength.value(),
            y_step_um=self.ui.YStepSize.value(),
            stage_bounds=(
                self.ui.Xmin.value(),
                self.ui.Xmax.value(),
                self.ui.Ymin.value(),
                self.ui.Ymax.value(),
            ),
            occupancy=self.mosaic_roi_occupancy,
            overlap=self.mosaic_fov_overlap,
            max_y_fov_mm=self.mosaic_max_y_fov_mm,
            center_mode=self.mosaic_center_mode,
        )
        reference_z = self.sample_centers[current_id - 1].z
        if self.CurrentSampleLocations:
            reference_z = self.CurrentSampleLocations[0].z
        if self.debug_mosaic_correction:
            print(
                "Mosaic correction scan plan: "
                f"center_mode={self.mosaic_center_mode}, "
                f"center=({scan_plan.center_x:.3f}, {scan_plan.center_y:.3f}), "
                f"roi_bounds=(x:{scan_plan.roi_bounds[0]:.3f}-{scan_plan.roi_bounds[2]:.3f}, "
                f"y:{scan_plan.roi_bounds[1]:.3f}-{scan_plan.roi_bounds[3]:.3f}), "
                f"roi_size=({scan_plan.roi_size[0]:.3f}, {scan_plan.roi_size[1]:.3f}), "
                f"required_span=({scan_plan.required_span[0]:.3f}, {scan_plan.required_span[1]:.3f}), "
                f"tile_count={scan_plan.tile_count}, candidates={scan_plan.candidate_count}, "
                f"accepted={len(scan_plan.fov_locations)}, "
                f"planned_YLength={scan_plan.y_length_mm:.3f}, planned_Ypixels={scan_plan.y_pixels}, "
                f"locations={scan_plan.fov_locations}"
            )
        for fov_location in scan_plan.fov_locations:
            fov_location.z = reference_z
            fov_location.y_length_mm = scan_plan.y_length_mm
        print(
            "Mosaic correction new FOV centers: "
            + ", ".join(
                f"(x={fov_location.x:.3f}, y={fov_location.y:.3f}, z={fov_location.z:.3f}, "
                f"YLength={fov_location.y_length_mm if fov_location.y_length_mm is not None else 'None'})"
                for fov_location in scan_plan.fov_locations
            )
        )
        self.apply_y_geometry_for_correction(scan_plan.y_length_mm, scan_plan.y_pixels)

        # Apply the corrected scan plan and update the overlay.
        self.apply_mosaic_correction_plan(current_id, mm_polygons, scan_plan, source_y_length)
        self.ui.mosaic_viewer.clear_polygons()
        return True

    def apply_mosaic_correction_plan(self, sample_id, mm_polygons, scan_plan, source_y_length=None):
        """Apply a corrected scan plan and create the corresponding overlay source."""
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        new_fov_locations = scan_plan.fov_locations
        
        self.sample_centers[sample_id - 1].x = scan_plan.center_x
        self.sample_centers[sample_id - 1].y = scan_plan.center_y
        if self.debug_mosaic_correction:
            print(
                "Mosaic correction FOV grid result: "
                f"sample_id={sample_id}, accepted_tiles={len(new_fov_locations)}, "
                f"YFOV={YFOV:.3f}"
            )

        mosaic_source_y_fov = source_y_length if source_y_length is not None else YFOV
        self.overlay_images[sample_id] = build_mosaic_correction_overlay_source(
            mosaic_image=self.ui.mosaic_viewer.adj,
            current_fov_locations=self.CurrentSampleLocations,
            mm_polygons=mm_polygons,
            new_fov_locations=new_fov_locations,
            x_fov_mm=XFOV,
            y_fov_mm=YFOV,
            source_y_length_mm=mosaic_source_y_fov,
            x_step_um=self.ui.XStepSize.value(),
            y_step_um=self.ui.YStepSize.value(),
        )
        self.render_mosaic_correction_overlay(sample_id, self.overlay_images[sample_id])
        self.CurrentSampleLocations = new_fov_locations        

    def render_mosaic_correction_overlay(self, sample_id, source):
        render_mosaic_correction_overlay(self.ui, source)
        
    def get_background(self):
        an_action = AODOActionField('rotate_servo_out')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        # print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        BAvg = self.ui.BlineAVG.value()
        dyn_checked = self.ui.DynCheckBox.isChecked()
        realtime_dyn_checked = self.ui.RealtimeDynCheckBox.isChecked()
        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_BLINE)
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(100)
        self.ui.DynCheckBox.setChecked(False)
        self.ui.RealtimeDynCheckBox.setChecked(False)
        ############################# measure an Aline
        # print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_BLINE,
            acq_mode=AcqTypes.FINITE_BLINE,
            skip_save=True,
        )
        # print('got Bline')
        # print(self.data.shape)
        #######################################################################
        Xpixels = self.ui.AlinesPerBline.value()
        Yrpt = self.ui.BlineAVG.value()
        BLINE = self.data.reshape([Yrpt, Xpixels, self.current_nsamples()])
        
        background = np.float32(np.mean(BLINE,0))
        # plt.figure()
        # plt.imshow(background)
        # plt.show()
        # background = np.smooth()
        # print(background.shape)
        filePath = self.ui.DIR.toPlainText()
        current_time = datetime.datetime.now()
        filePath = filePath + "/" + 'background_'+\
            str(current_time.year)+'-'+\
            str(current_time.month)+'-'+\
            str(current_time.day)+'-'+\
            str(current_time.hour)+'-'+\
            str(current_time.minute)+'-'+\
            str(current_time.second)+\
            '.bin'
        fp = open(filePath, 'wb')
        background.tofile(fp)
        fp.close()
        
        self.ui.BG_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(BAvg)
        self.ui.DynCheckBox.setChecked(dyn_checked)
        self.ui.RealtimeDynCheckBox.setChecked(realtime_dyn_checked)
        
        an_action = AODOActionField('rotate_servo_back')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        return "Background measurement completed."

    def get_background_cscan(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        bline_avg = self.ui.BlineAVG.value()
        dyn_checked = self.ui.DynCheckBox.isChecked()
        realtime_dyn_checked = self.ui.RealtimeDynCheckBox.isChecked()

        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_CSCAN)
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(1)
        self.ui.DynCheckBox.setChecked(False)
        self.ui.RealtimeDynCheckBox.setChecked(False)

        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_CSCAN,
            acq_mode=AcqTypes.FINITE_CSCAN,
            skip_save=True,
        )

        background = np.float32(np.mean(self.data, axis=0))

        filePath = self.ui.DIR.toPlainText()
        current_time = datetime.datetime.now()
        filePath = filePath + "/" + 'background_'+\
            str(current_time.year)+'-'+\
            str(current_time.month)+'-'+\
            str(current_time.day)+'-'+\
            str(current_time.hour)+'-'+\
            str(current_time.minute)+'-'+\
            str(current_time.second)+\
            '.bin'
        fp = open(filePath, 'wb')
        background.tofile(fp)
        fp.close()

        self.ui.BG_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(bline_avg)
        self.ui.DynCheckBox.setChecked(dyn_checked)
        self.ui.RealtimeDynCheckBox.setChecked(realtime_dyn_checked)
        return "Background measurement completed."
    
    def get_surfCurve(self):
        
        print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_BLINE)
        self.ui.FFTDevice.setCurrentText('GPU')
        self.ui.DSing.setChecked(True)
        ############################# measure an Cscan
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(DnS_action='SingleCscan', acq_mode='SingleCscan')
        # while self.GPU2weaverQueue.qsize()<1:
        #     time.sleep(1)
        cscan =self.GPU2weaverQueue.get()
        
        Zpixels = self.ui.DepthRange.value()
        # get number of X pixels
        Xpixels = self.ui.AlinesPerBline.value()
        # get number of Y pixels
        Ypixels = self.ui.Ypixels.value()* self.ui.BlineAVG.value()
        # reshape into Ypixels x Xpixels x Zpixels
        cscan = cscan.reshape([Ypixels,Xpixels,Zpixels])
        
        Bline = np.float32(np.mean(cscan,0))
        surfCurve = np.zeros([Xpixels])

        plt.figure()
        plt.imshow(Bline)
        plt.title('Bline for finding surface')
        for xx in range(Xpixels):
            surfCurve[xx] = findchangept(Bline[xx,:],1)
        
        surfCurve = surfCurve - min(surfCurve)
        plt.figure()
        plt.plot(surfCurve)
        plt.title('surface')
        plt.figure()

        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + 'surfCurve.bin'
        fp = open(filePath, 'wb')
        np.uint16(surfCurve).tofile(fp)
        fp.close()
        
        self.ui.Surf_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.DSing.setChecked(False)
        return "Surface curve measurement completed."
    
    def live(self):
        """Continuously captures and displays video until RunButton is unchecked."""
        cap = open_usb_camera(configure_exposure=True)
    
        while self.ui.RunButton.isChecked():
            ret, frame = cap.read()
            if not ret:
                break
    
            frame = orient_usb_frame(frame)
            
            # 3. Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # 4. Convert to QImage then QPixmap
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
    
            # 5. Display on Label (scaled to fit the widget)
            # Note: If this is a separate thread, UI updates should ideally 
            # use Signals, but for a simple script, this often works:
            self.ui.XZplane.setPixmap(pixmap.scaled(
                self.ui.XZplane.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
        # 6. Release resources when button is unchecked
        
        
    
    def save_session_data(self, folder_path):
        def render_overlay(sample_id):
            self.display_sample_overlay(sample_id)
            pixmap = self.ui.MosaicLabel.pixmap()
            if pixmap is None:
                return None
            return pixmap

        save_session_data(
            folder_path,
            self.FOV_locations,
            self.sample_centers,
            self.overlay_images,
            raw_img=getattr(self, "raw_img", None),
            pixel_polygons=getattr(self, "pixel_polygons", None),
            render_overlay=render_overlay,
        )
        print(f"Session data saved to {folder_path}")

    def save_usb_training_data(self, folder_path):
        save_usb_training_data(
            folder_path,
            getattr(self, "raw_img", None),
            getattr(self, "pixel_polygons", None),
            self.sample_centers,
            self.FOV_locations,
        )
    
    def load_session_data(self, folder_path):
        self.FOV_locations, self.sample_centers, self.overlay_images = load_session_data(folder_path)
        print(f"Session data loaded from {folder_path}")
