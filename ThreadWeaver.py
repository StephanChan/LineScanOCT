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
from Actions import DnSAction, AODOAction, GPUAction, DAction
import traceback
import os
import json
import matplotlib.pyplot as plt
from queue import Empty
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
# from matplotlib.path import Path
# from scipy.signal import hilbert
import datetime
import pickle
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
# axial pixel size, measure with a microscope glass slide
global ZPIXELSIZE
ZPIXELSIZE = 4.4 # unit: um

class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.mosaic = None
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
        self.QueueOut()

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def finish_with_message(self, message):
        if message is None:
            return
        self.emit_status(message)
        self.log.write(str(message))
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != 'exit':
            try:
                if self.item.action in ['ContinuousAline','ContinuousBline','ContinuousCscan']:
                    self.InitMemory()
                    message = self.RptScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)
                    
                elif self.item.action in ['FiniteBline', 'FiniteAline', 'FiniteCscan']:
                    self.InitMemory()
                    message = self.SingleScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    if self.item.action in ['FiniteCscan'] and self.ui.Save.isChecked():
                        an_action = GPUAction('IncrementCscanNum')
                        self.GPUQueue.put(an_action)
                    self.finish_with_message(message)
                elif self.item.action in ['LocationCameraLive']:
                    self.live()

                elif self.item.action == 'PlatePreScan':
                    message = self.prepare_and_run_plate_prescan(acq_mode=self.item.action, payload=self.item.payload)
                    self.finish_with_message(message)
                elif self.item.action == 'PlateScan':
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
                    message = self.PlateScan(acq_mode=self.item.action, payload=self.item.payload)
                    self.finish_with_message(message)
                    if self.ui.Save.isChecked():
                        an_action = GPUAction('IncrementTime')
                        self.GPUQueue.put(an_action)
                elif self.item.action == 'WellScan':
                    if self.FOV_locations == {}:
                        self.load_session_data(self.ui.DIR.toPlainText()+'/Mosaic')
                        """Updates the combo box content on the main thread."""
                        self.ui.sampleSelector.clear()
                        if len(self.sample_centers) == 0:
                            self.ui.sampleSelector.addItem("No Samples Found")
                        else:
                            for i in range(len(self.sample_centers)):
                                self.ui.sampleSelector.addItem(f"Sample {i+1}")
                        self.ui.sampleSelector.setCurrentIndex(0)
                    message = self.WellScan(acq_mode=self.item.action, payload=self.item.payload)
                    self.finish_with_message(message)
                    # if self.ui.Save.isChecked():
                    #     an_action = GPUAction('IncrementSampleID')
                    #     self.GPUQueue.put(an_action)

                elif self.item.action == 'ZstageRepeatibility':
                    message = self.ZstageRepeatibility()
                    self.finish_with_message(message)
                    
                elif self.item.action == 'get_background':
                    message = self.get_background()
                    self.finish_with_message(message)
                    
                elif self.item.action == 'get_surface':
                    message = self.get_surfCurve()
                    self.finish_with_message(message)
                    
                else:
                    message = f"Unknown acquisition command: {self.item.action}"
                    self.finish_with_message(message)

            except Exception as error:
                message = f"Acquisition command failed: {self.item.action}"
                self.finish_with_message(message)
                print(traceback.format_exc())
            # reset RUN button
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText('Go')
            self.ui.PauseButton.setChecked(False)
            self.ui.PauseButton.setText('Pause')
            self.ui.RunButton.setEnabled(True)
            self.ui.PauseButton.setEnabled(True)
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
            self.log.write(message + "\n")
        return drained

    def drain_continuous_backlog(self, reason=""):
        continuous_modes = {'ContinuousAline', 'ContinuousBline', 'ContinuousCscan'}

        def keep_gpu_item(item):
            is_fft_action = getattr(item, 'action', None) in {'GPU', 'CPU'}
            is_continuous_mode = getattr(item, 'DnS_action', None) in continuous_modes
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
        
    
    def InitMemory(self):
        #################################################################
        # get number samplers per Aline
        samples = self.ui.NSamples_DH.value()
            
        AlinesPerBline = self.ui.AlinesPerBline.value()
        # print(self.ui.PixelFormat_display.text())
        if self.ui.PixelFormat_display_DH.text() in ['Mono8']:
            data_type =  np.uint8
        else:
            data_type =  np.uint16
            
        for ii in range(self.memoryCount):
             
             if self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','FiniteBline', 'FiniteAline']:
                 self.Memory[ii]=np.zeros([self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                 self.NAcq = 1
             elif self.ui.ACQMode.currentText() in ['ContinuousCscan']:
                 self.Memory[ii]=np.zeros([self.ui.Ypixels.value()*self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                 self.NAcq = 1
             elif self.ui.ACQMode.currentText() in ['FiniteCscan', 'PlateScan', 'PlatePreScan','WellScan']:
                 if self.ui.DynCheckBox.isChecked():
                     self.Memory[ii]=np.zeros([self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = self.ui.Ypixels.value()
                 else:
                     self.Memory[ii]=np.zeros([self.ui.Ypixels.value()*self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = 1

        ###########################################################################################
        
    def SingleScan(self, DnS_action, acq_mode, payload=[]):
        # an_action = DnSAction('Clear')
        # self.DnSQueue.put(an_action)
        self.drain_continuous_backlog(reason=f"before {DnS_action}")
        t0=time.time()
        # print(self.DbackQueue.qsize())
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        t1=time.time()
        ###########################################################################################
        # start AODO 
        an_action = AODOAction('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        t2=time.time()
        # start camera

        an_action = DAction('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
        t3=time.time()

        # print('current dbackqueue size:', self.DbackQueue.qsize())
        an_action = AODOAction('StartTask')
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
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            print('waiting for camera data...')
            while self.ui.RunButton.isChecked():
                try:
                    an_action = self.DatabackQueue.get(timeout = 5)
                    # print('camera queue size:', self.DatabackQueue.qsize())
                    print('time to fetch data: '+str(round(time.time()-start,3))+'sec')
                    memory_slot = an_action.memory_slot
                    # print(memory_slot)
                    ############################################### display and save data
                    if self.ui.FFTDevice.currentText() in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        self.data = self.Memory[memory_slot].copy()
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            message = "No usable spectral data received."
                            print(message)
                            self.log.write(message)
                        else:
                            an_action = DnSAction(DnS_action, acq_mode=acq_mode, data=self.data, raw=True, payload=payload) # data in Memory[memory_slot]
                            self.DnSQueue.put(an_action)
                            message = f"{DnS_action} completed."
                    else:
                        # In other modes, do FFT first
                        an_action = GPUAction(action = self.ui.FFTDevice.currentText(), DnS_action = DnS_action, acq_mode=acq_mode, memory_slot = memory_slot, payload=payload)
                        self.GPUQueue.put(an_action)
                        message = f"{DnS_action} completed."
                    break
                except:
                    print(f"{DnS_action}: waiting for camera data...")
                    
        an_action = AODOAction('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        print(message)
        return message
    
            
    def RptScan(self, DnS_action, acq_mode):
        # an_action = DnSAction('Clear')
        # self.DnSQueue.put(an_action)
        frame_rate = self.ui.FrameRate_DH.value()
        if acq_mode in ['ContinuousAline','ContinuousBline']:
            self.ui.FrameRate_DH.setValue(20)
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        # config AODO
        an_action = AODOAction('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        data_backs = 0 # count number of data backs
        skipped_fft_actions = 0

        # start digitizer for one acuquqisition
        an_action = DAction('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()

        # start AODO 
        an_action = AODOAction('StartTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()

        ######################################################### repeat acquisition until Stop button is clicked
        while self.ui.RunButton.isChecked():
            ######################################### collect data
            try: # use try-except in cases where Stop button clicked and camera stopped prior to while loop
                start = time.time()
                an_action = self.DatabackQueue.get(timeout=5) # never time out
                # print('time to fetch data: '+str(round(time.time()-start,3)))
                memory_slot = an_action.memory_slot
                # print(memory_slot)
                data_backs += 1
                if memory_slot < self.ui.DisplayRatio.value():
                    ######################################### display data
                    if self.ui.FFTDevice.currentText() in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        self.data = self.Memory[memory_slot].copy()
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            message = "No usable spectral data received."
                            print(message)
                            self.log.write(message)
                        else:
                            an_action = DnSAction(DnS_action, acq_mode=acq_mode, data=self.data, raw=True) # data in Memory[memory_slot]
                            self.DnSQueue.put(an_action)
                            message = f"{DnS_action} completed."
                    else:
                        # In other modes, do FFT first
                        if self.GPUQueue.qsize() == 0:
                            an_action = GPUAction(self.ui.FFTDevice.currentText(), DnS_action=DnS_action, acq_mode=acq_mode, memory_slot=memory_slot)
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
                an_action = AODOAction('StopTask')
                self.AODOQueue.put(an_action)
                # wait until stop button or pause button is clicked
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                # if resume, restart AODO task
                if not self.ui.PauseButton.isChecked():
                    # start AODO 
                    an_action = AODOAction('StartTask')
                    self.AODOQueue.put(an_action)
                    self.StagebackQueue.get()
        # Camera will stop once Stop Button is clicked
        # AODO thread will need StopTask command
        an_action = AODOAction('tryStopTask')
        self.AODOQueue.put(an_action)
        # close AODO
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        # digitizer will close automatically
        message = f"{DnS_action} stopped. Received {data_backs} camera buffer(s)."
        if skipped_fft_actions:
            message += f" Skipped {skipped_fft_actions} stale continuous FFT request(s)."
        self.log.write(message)
        print(message)
        self.drain_continuous_backlog(reason=f"after {DnS_action}")
        an_action = GPUAction('display_FFT_actions')
        self.GPUQueue.put(an_action)
        an_action = GPUAction('display_counts', payload=DnS_action)
        self.GPUQueue.put(an_action)
        self.ui.FrameRate_DH.setValue(frame_rate)
        return message
  

    def prepare_and_run_plate_prescan(self, acq_mode, payload=None):
        mosaic_folder = self.ui.DIR.toPlainText() + '/Mosaic'

        if payload:
            if not os.path.exists(mosaic_folder):
                os.mkdir(mosaic_folder)
            return self.PlatePreScan(acq_mode=acq_mode, payload=payload)

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

    def has_plate_plan(self):
        return (
            isinstance(self.FOV_locations, list)
            and isinstance(self.sample_centers, list)
            and len(self.FOV_locations) > 0
            and len(self.sample_centers) > 0
        )

    def update_sample_selector_from_plan(self):
        self.ui.sampleSelector.clear()
        if not self.sample_centers:
            self.ui.sampleSelector.addItem("No Samples Found")
            return
        for i in range(len(self.sample_centers)):
            self.ui.sampleSelector.addItem(f"Sample {i+1}")

    def PlatePreScan(self, acq_mode, payload=None):
        fresh_locator_data = bool(payload)
        if fresh_locator_data:
            self.overlay_images = {}
            self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= payload
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
                self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == sample_center['sample_id']]
                if fresh_locator_data:
                    self.display_initial_scan_overlay(sample_center['sample_id'], self.raw_img, self.pixel_polygons)
                else:
                    self.display_sample_overlay(sample_center['sample_id'])
                
                # User stopped continuousBline, then we do Mosaic scan for this sample
                self.AdjustZstage(sample_center['sample_id'])
    
                try:
                    message = self.iterate_FOVs(acq_mode=acq_mode)
                finally:
                    self.restore_y_geometry_after_correction()
                self.ui.NextSampleButton.setText('下一个样品')
                self.ui.RepeatSampleButton.setText('重新扫描')
                while (not self.ui.NextSampleButton.isChecked()) and self.ui.RunButton.isChecked():
                    if self.ui.RepeatSampleButton.isChecked():
                        self.ui.NextSampleButton.setText('扫描中，请等待')
                        self.ui.RepeatSampleButton.setText('扫描中，请等待')
                        self.process_mosaic_correction()
                        self.AdjustZstage(sample_center['sample_id'])
                        try:
                            message = self.iterate_FOVs(acq_mode=acq_mode)
                        finally:
                            self.restore_y_geometry_after_correction()
                        self.ui.NextSampleButton.setText('下一个样品')
                        self.ui.RepeatSampleButton.setText('重新扫描')
                        self.ui.RepeatSampleButton.setChecked(False)
                    time.sleep(1)
                        
                self.ui.NextSampleButton.setChecked(False)
                self.ui.sampleSelector.setCurrentIndex(self.ui.sampleSelector.currentIndex() + 1)
                # 1. Remove all old entries matching this sample_id
                # We keep everything that DOES NOT match the ID we are updating
                lower_id_locations = [location for location in self.FOV_locations
                                       if location.get('sample_id') < sample_center['sample_id']]
                
                higher_id_locations = [location for location in self.FOV_locations
                                       if location.get('sample_id') > sample_center['sample_id']]
            
                # 2. Combine them back together
                self.FOV_locations = lower_id_locations + self.CurrentSampleLocations + higher_id_locations
                
        
        # save self.FOV_locations, self.sample_centers, self.overlay_images
        self.save_session_data(self.ui.DIR.toPlainText()+'/Mosaic')
        self.ui.NextSampleButton.setText('扫描结束')
        self.ui.RepeatSampleButton.setText('扫描结束')
        self.ui.BlineAVG.setValue(BlineAVG)
        return(message)
            
    def PlateScan(self, acq_mode, payload):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= payload
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        # print(self.sample_centers)
        # print(self.FOV_locations)
        for sample_center in self.sample_centers:
            if self.ui.RunButton.isChecked():
                self.ui.sampleSelector.setCurrentIndex(sample_center['sample_id']-1)
                self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == sample_center['sample_id']]
                # print('self.CurrentSampleLocations', self.CurrentSampleLocations)
                self.display_sample_overlay(sample_center['sample_id'])
                try:
                    self.iterate_FOVs(acq_mode=acq_mode)
                finally:
                    self.restore_y_geometry_after_correction()
                if self.ui.Save.isChecked():
                    an_action = GPUAction('IncrementSampleID')
                    self.GPUQueue.put(an_action)
                message = "Plate scan completed."
            else:
                message = "Plate scan stopped by user."
        return(message)   
    
    
    def WellScan(self, acq_mode, payload):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= payload
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        sample_id = self.ui.sampleSelector.currentIndex()+1
        self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == sample_id]
        self.display_sample_overlay(sample_id)
        try:
            message = self.iterate_FOVs(acq_mode=acq_mode)
        finally:
            self.restore_y_geometry_after_correction()
        return(message) 

    def AdjustZstage(self, sample_id):
        self.clear_mosaic_display()
        sample_center = self.sample_centers[sample_id-1]
        # move to center position of this sample
        self.move_stage_axis('X', sample_center['x'])
        self.move_stage_axis('Y', sample_center['y'])
        self.move_stage_axis('Z', sample_center['z'])
        # do continuous scan to display Bline
        self.ui.ACQMode.setCurrentText('ContinuousBline')
        self.InitMemory()
        self.ui.RunButton.setChecked(True)
        self.ui.RunButton.setText('点击开始扫描')
        self.RptScan(DnS_action='ContinuousBline', acq_mode='ContinuousBline')
        # User can move Z stage up and down to put sample at focus
        for ii, item in enumerate(self.CurrentSampleLocations):
            self.CurrentSampleLocations[ii]['z'] = self.ui.ZPosition.value()
        
        self.ui.RunButton.setText('Stop')
        self.ui.RunButton.setChecked(True)
        
    def iterate_FOVs(self, acq_mode):
        # print(self.CurrentSampleLocations)
        self.clear_mosaic_display()
        self.drain_continuous_backlog(reason=f"before {acq_mode}")
        self.apply_y_geometry_from_locations()
        # move to position of this FOV
        first_fov_location = self.CurrentSampleLocations[0]
        self.move_stage_axis('X', first_fov_location['x'])
        self.move_stage_axis('Y', first_fov_location['y'])
        self.move_stage_axis('Z', first_fov_location['z'])
        
        Xpixels = self.ui.AlinesPerBline.value()//self.ui.AlineAVG.value()
        Ypixels = self.ui.Ypixels.value()
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        an_action = GPUAction('Init_Mosaic', payload=[self.CurrentSampleLocations, (Xpixels, Ypixels), (XFOV, YFOV)])
        self.GPUQueue.put(an_action)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.InitMemory()
        for fov_location in self.CurrentSampleLocations:
            if self.ui.RunButton.isChecked():
                # print(fov_location['x'],fov_location['y'])
                # move to position of this FOV
                self.move_stage_axis('X', fov_location['x'])
                self.move_stage_axis('Y', fov_location['y'])
                self.move_stage_axis('Z', fov_location['z'])
                
                # do FiniteCscan at this position
                self.SingleScan(DnS_action='Process_Mosaic', acq_mode=acq_mode, payload=[self.CurrentSampleLocations, fov_location])
                if self.ui.Save.isChecked():
                    an_action = GPUAction('IncrementTileNum')
                    self.GPUQueue.put(an_action)
                # handle pause action
                if self.ui.PauseButton.isChecked():
                    # wait until stop button or pause button is clicked
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(1)
                        print('waiting')
                message = "Sample FOV scan completed."
            else:
                message = "Sample FOV scan stopped by user."
        return(message)

    def move_stage_axis(self, axis, target, tolerance=0.005):
        position_widget = getattr(self.ui, f"{axis}Position")
        current_widget = getattr(self.ui, f"{axis}current")
        position_widget.setValue(target)

        current = current_widget.value()
        distance = target - current
        if abs(distance) <= tolerance:
            return

        an_action = AODOAction(f"{axis}move2")
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
            self.log.write(message + "\n")
            self.ui.RunButton.setChecked(False)
            raise TimeoutError(message)

    def stage_move_timeout(self, axis, distance):
        speed_widget = getattr(self.ui, f"{axis}Speed", None)
        try:
            speed = float(speed_widget.value()) if speed_widget is not None else 1.0
        except Exception:
            speed = 1.0
        speed = max(abs(speed), 0.001)
        return max(20.0, abs(distance) / speed * 10.0 + 100.0)

    def sample_fov_locations(self, sample_id):
        if getattr(self, "CurrentSampleLocations", None):
            current_ids = {location.get('sample_id') for location in self.CurrentSampleLocations}
            if sample_id in current_ids:
                return [location for location in self.CurrentSampleLocations if location.get('sample_id') == sample_id]
        return [location for location in self.FOV_locations if location.get('sample_id') == sample_id]

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

    def apply_y_geometry_for_correction(self, y_length_mm, y_pixels):
        if self._restore_y_geometry is None:
            self._restore_y_geometry = {
                "YLength": self.ui.YLength.value(),
                "Ypixels": self.ui.Ypixels.value(),
            }
        if self.debug_mosaic_correction:
            print(
                "Mosaic correction Y geometry apply: "
                f"YLength {self.ui.YLength.value():.3f} -> {y_length_mm:.3f}, "
                f"Ypixels {self.ui.Ypixels.value()} -> {y_pixels}"
            )
        self.ui.YLength.setValue(float(y_length_mm))
        self.ui.Ypixels.setValue(int(y_pixels))

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
        y_length = first_fov_location.get("y_length_mm", None)
        y_pixels = first_fov_location.get("y_pixels", None)
        if y_length is None or y_pixels is None:
            return
        self.apply_y_geometry_for_correction(float(y_length), int(y_pixels))

    def current_location_y_length(self):
        if self.CurrentSampleLocations:
            y_length = self.CurrentSampleLocations[0].get("y_length_mm", None)
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
            return

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
            fov_location["y_length_mm"] = scan_plan.y_length_mm
            fov_location["y_pixels"] = scan_plan.y_pixels
        self.apply_y_geometry_for_correction(scan_plan.y_length_mm, scan_plan.y_pixels)

        # Apply the corrected scan plan and update the overlay.
        self.apply_mosaic_correction_plan(current_id, mm_polygons, scan_plan, source_y_length)
        self.ui.mosaic_viewer.clear_polygons()

    def apply_mosaic_correction_plan(self, sample_id, mm_polygons, scan_plan, source_y_length=None):
        """Apply a corrected scan plan and create the corresponding overlay source."""
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        new_fov_locations = scan_plan.fov_locations
        
        self.sample_centers[sample_id-1]['x'] = scan_plan.center_x
        self.sample_centers[sample_id-1]['y'] = scan_plan.center_y
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
       
    def ZstageRepeatibility(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        DnS_action = 'SingleAline'
        self.ui.ACQMode.setCurrentText(DnS_action)
        self.ui.FFTDevice.setCurrentText('GPU')
        current_Xposition = self.ui.XPosition.value()
        current_Yposition = self.ui.YPosition.value()
        current_Zposition = self.ui.ZPosition.value()
        iteration = 50
        for i in range(iteration):
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            # measure ALine
            message = self.SingleScan(DnS_action=DnS_action, acq_mode=DnS_action)
            self.log.write(message)
            # self.ui.PrintOut.append(message)
            failed_times = 0
            while message != f"{DnS_action} completed.":
                failed_times+=1
                if failed_times > 10:
                    self.ui.ACQMode.setCurrentText(acq_mode)
                    self.ui.FFTDevice.setCurrentText(fft_device)
                    self.ui.Gotozero.setChecked(False)
                    return message
                message = self.SingleScan(DnS_action=DnS_action, acq_mode=DnS_action)
                self.log.write(message)
                # self.ui.PrintOut.append(message)
                time.sleep(1)
            time.sleep(0.1)
            
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            self.ui.ZPosition.setValue(5)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            # move to clear XY position
            self.ui.XPosition.setValue(45)
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            self.ui.YPosition.setValue(20)
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            # move Z stage 
            self.ui.ZPosition.setValue(40)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
            self.ui.ZPosition.setValue(40.1)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
            self.ui.ZPosition.setValue(40.15)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            self.ui.ZPosition.setValue(5)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            # move to original XY position
            self.ui.XPosition.setValue(current_Xposition)
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            self.ui.YPosition.setValue(current_Yposition)
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = "Stage repeatability test stopped by user."
                break
            # move Z stage up
            self.ui.ZPosition.setValue(current_Zposition)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
        self.ui.ZstageTest.setChecked(False)
        # self.weaverBackQueue.put(0)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        return "Stage repeatability test completed."
        
    def ZstageRepeatibility2(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        self.ui.FFTDevice.setCurrentText('GPU')
        DnS_action = 'SingleAline'
        self.ui.ACQMode.setCurrentText(DnS_action)
        current_position = self.ui.ZPosition.value() # this is the target Z pos in this test
        iteration = 100
        for i in range(iteration):
            if not self.ui.ZstageTest2.isChecked():
                break
            # measure ALine
            message = self.SingleScan(DnS_action=DnS_action, acq_mode=DnS_action)
            self.log.write(message)
            while message != f"{DnS_action} completed.":
                message = self.SingleScan(DnS_action=DnS_action, acq_mode=DnS_action)
                self.log.write(message)
                # self.ui.PrintOut.append(message)
                time.sleep(1)
            time.sleep(0.1) # let GUI update Aline 
            # move Z stage down
            
            self.ui.ZPosition.setValue(3)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
            self.ui.XPosition.setValue(70)
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            self.ui.YPosition.setValue(20)
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            # remeasure background
            self.get_background()
            
            self.ui.ZPosition.setValue(7)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
            # go to defined zero
            self.ui.Gotozero.setChecked(True)
            message = self.Gotozero()
            display_message = "Stage returned to zero." if message == 'gotozero success...' else message
            self.emit_status(display_message)
            if message != 'gotozero success...':
                message = "Go-to-zero failed. Stage test stopped."
                print(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                break
            else:
                # move to target AlinesPerBline
                self.ui.ZPosition.setValue(current_position)
                an_action = AODOAction('Zmove2')
                self.AODOQueue.put(an_action)
                self.StagebackQueue.get()
            self.ui.ACQMode.setCurrentText(DnS_action)
            self.ui.FFTDevice.setCurrentText('GPU')
            
            
        self.ui.ZstageTest2.setChecked(False)
        # self.weaverBackQueue.put(0)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)

            
        
    def get_background(self):
        print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        BAvg = self.ui.BlineAVG.value()
        self.ui.ACQMode.setCurrentText('FiniteBline')
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(100)
        ############################# measure an Aline
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(DnS_action='FiniteBline', acq_mode='FiniteBline')
        print('got Bline')
        print(self.data.shape)
        #######################################################################
        Xpixels = self.ui.AlinesPerBline.value()
        Yrpt = self.ui.BlineAVG.value()
        BLINE = self.data.reshape([Yrpt, Xpixels, self.ui.NSamples_DH.value()])
        
        background = np.float32(np.mean(BLINE,0))
        plt.figure()
        plt.imshow(background)
        plt.show()
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
        return "Background measurement completed."
    
    def get_surfCurve(self):
        
        print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText('FiniteBline')
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
        """
        Saves FOV locations, sample centers, and overlay images to the specified folder.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
        # 1. Save Numerical/Text Data (FOV locations and Sample Centers)
        data_to_save = {
            'FOV_locations': self.FOV_locations, #
            'sample_centers': self.sample_centers
        }
        
        with open(os.path.join(folder_path, 'scan_metadata.pkl'), 'wb') as f:
            pickle.dump(data_to_save, f)

        self.save_usb_training_data(folder_path)
    
        overlay_sources = {
            sample_id: source
            for sample_id, source in self.overlay_images.items()
            if isinstance(source, dict)
        }
        with open(os.path.join(folder_path, 'overlay_sources.pkl'), 'wb') as f:
            pickle.dump(overlay_sources, f)

        # Also save rendered overlay snapshots for quick visual inspection and
        # backward compatibility with older sessions.
        image_dir = os.path.join(folder_path, 'overlays')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    
        for sample_id, source in self.overlay_images.items():
            file_path = os.path.join(image_dir, f'sample_{sample_id}_overlay.png')
            if isinstance(source, QPixmap):
                source.save(file_path, "PNG")
            else:
                self.display_sample_overlay(sample_id)
                pixmap = self.ui.MosaicLabel.pixmap()
                if pixmap is not None:
                    pixmap.save(file_path, "PNG")
        
        print(f"Session data saved to {folder_path}")

    def save_usb_training_data(self, folder_path):
        if not hasattr(self, "raw_img") or self.raw_img is None:
            return
        if not hasattr(self, "pixel_polygons") or self.pixel_polygons is None:
            return
        if isinstance(self.raw_img, list) and len(self.raw_img) == 0:
            return

        image_path = os.path.join(folder_path, 'usb_raw_image.png')
        cv2.imwrite(image_path, self.raw_img)

        def clean_points(points):
            return [[float(x), float(y)] for x, y in points]

        def clean_records(records):
            cleaned = []
            for item in records:
                cleaned.append(
                    {
                        key: (
                            int(value)
                            if isinstance(value, (np.integer,))
                            else float(value)
                            if isinstance(value, (np.floating,))
                            else value
                        )
                        for key, value in item.items()
                    }
                )
            return cleaned

        roi_data = {
            'image_file': os.path.basename(image_path),
            'coordinate_system': {
                'roi_vertices': 'USB displayed image pixel coordinates',
                'image_vertical_axis': 'stage X',
                'image_horizontal_axis': 'stage Y',
                'image_right_direction': 'smaller stage Y',
                'stage_units': 'mm',
            },
            'pixel_polygons': [
                {
                    'sample_id': idx + 1,
                    'points': clean_points(poly),
                }
                for idx, poly in enumerate(self.pixel_polygons)
            ],
            'sample_centers': clean_records(self.sample_centers),
            'fov_locations': clean_records(self.FOV_locations),
        }

        json_path = os.path.join(folder_path, 'usb_rois.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(roi_data, f, indent=2)
    
    def load_session_data(self, folder_path):
        """
        Loads FOV locations, sample centers, and overlay images from the specified folder.
        """
        # 1. Load Numerical/Text Data
        metadata_path = os.path.join(folder_path, 'scan_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.FOV_locations = data.get('FOV_locations', []) #
                self.sample_centers = data.get('sample_centers', [])
        else:
            print("Metadata file not found.")
    
        self.overlay_images = {}
        overlay_sources_path = os.path.join(folder_path, 'overlay_sources.pkl')
        if os.path.exists(overlay_sources_path):
            with open(overlay_sources_path, 'rb') as f:
                self.overlay_images = pickle.load(f)
            print(f"Session data loaded from {folder_path}")
            return

        # 2. Load legacy rendered overlay images
        image_dir = os.path.join(folder_path, 'overlays')
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.startswith('sample_') and filename.endswith('.png'):
                    # Extract sample_id from filename (e.g., 'sample_1_overlay.png' -> 1)
                    try:
                        sample_id = int(filename.split('_')[1])
                        pixmap = QPixmap()
                        pixmap.load(os.path.join(image_dir, filename))
                        self.overlay_images[sample_id] = pixmap
                    except ValueError:
                        continue
        
        print(f"Session data loaded from {folder_path}")
