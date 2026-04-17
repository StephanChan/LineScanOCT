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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF
# from matplotlib.path import Path
from shapely.geometry import Polygon # Shapely is best for intersection math
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
from SampleLocator import USB_PIXEL_SIZE_MM, open_usb_camera, orient_usb_frame, stage_to_usb_image
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
        self.exit_message = 'ACQ thread successfully exited'
        
    def run(self):
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != 'exit':
            # self.ui.statusbar.showMessage('King thread is doing: '+self.item.action)
            try:
                if self.item.action in ['ContinuousAline','ContinuousBline','ContinuousCscan']:
                    self.InitMemory()
                    message = self.RptScan(self.item.action)
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action in ['FiniteBline', 'FiniteAline', 'FiniteCscan']:
                    self.InitMemory()
                    message = self.SingleScan(self.item.action)
                    if self.item.action in ['FiniteCscan'] and self.ui.Save.isChecked():
                        an_action = GPUAction('IncrementCscanNum')
                        self.GPUQueue.put(an_action)
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                elif self.item.action in ['LocationCameraLive']:
                    self.live()

                elif self.item.action == 'PlatePreScan':
                    # make directories
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/Mosaic'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/Mosaic')
                    # if not os.path.exists(self.ui.DIR.toPlainText()+'/surf'):
                    #     os.mkdir(self.ui.DIR.toPlainText()+'/surf')
                    message = self.PlatePreScan(self.item.args)
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    self.log.write(message)
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
                    message = self.PlateScan(self.item.args)
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    self.log.write(message)
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
                    message = self.WellScan(self.item.args)
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    self.log.write(message)
                    # if self.ui.Save.isChecked():
                    #     an_action = GPUAction('IncrementSampleID')
                    #     self.GPUQueue.put(an_action)

                elif self.item.action == 'ZstageRepeatibility':
                    message = self.ZstageRepeatibility()
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action == 'get_background':
                    message = self.get_background()
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action == 'get_surface':
                    message = self.get_surfCurve()
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                else:
                    message = 'Weaver thread is doing something invalid: '+self.item.action
                    if getattr(self, "ui_bridge", None) is not None:
                        try:
                            self.ui_bridge.status_message.emit(message)
                        except Exception:
                            self.ui.statusbar.showMessage(message)
                    else:
                        self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)

            except Exception as error:
                message = "An error occurred in"+  self.item.action + "\n"
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit(message)
                    except Exception:
                        self.ui.statusbar.showMessage(message)
                else:
                    self.ui.statusbar.showMessage(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
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
        if getattr(self, "ui_bridge", None) is not None:
            try:
                self.ui_bridge.status_message.emit(self.exit_message)
            except Exception:
                self.ui.statusbar.showMessage(self.exit_message)
        else:
            self.ui.statusbar.showMessage(self.exit_message)
            
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
            message = f"Drained {drained} stale item(s) from {name}"
            print(message)
            self.log.write(message + "\n")
        return drained

    def drain_continuous_backlog(self, reason=""):
        continuous_modes = {'ContinuousAline', 'ContinuousBline', 'ContinuousCscan'}

        def keep_gpu_item(item):
            is_fft_action = getattr(item, 'action', None) in {'GPU', 'CPU'}
            is_continuous_mode = getattr(item, 'mode', None) in continuous_modes
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
        
    def SingleScan(self, mode, Args=[]):
        # an_action = DnSAction('Clear')
        # self.DnSQueue.put(an_action)
        self.drain_continuous_backlog(reason=f"before {mode}")
        t0=time.time()
        # print(self.DbackQueue.qsize())
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
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
        # # print('current dbackqueue size:', self.DbackQueue.qsize())
        # print('\n')
        message = 'User stopped SingleScan'
        for iAcq in range(self.NAcq):
            start = time.time()
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            print('waiting for camera data...')
            while self.ui.RunButton.isChecked():
                try:
                    an_action = self.DatabackQueue.get(timeout = 5)
                    # print('camera queue size:', self.DatabackQueue.qsize())
                    print('time to fetch data: '+str(round(time.time()-start,3)))
                    memoryLoc = an_action.action
                    # print(memoryLoc)
                    ############################################### display and save data
                    if self.ui.FFTDevice.currentText() in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        self.data = self.Memory[memoryLoc].copy()
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            print('spectral data all zeros!')
                            # self.ui.PrintOut.append('spectral data all zeros!')
                            self.log.write('spectral data all zeros!')
                            message = 'spectral data all zeros!'
                        else:
                            an_action = DnSAction(mode, self.data, raw=True, args = Args) # data in Memory[memoryLoc]
                            self.DnSQueue.put(an_action)
                            message = mode + " successfully finished..."
                    else:
                        # In other modes, do FFT first
                        an_action = GPUAction(action = self.ui.FFTDevice.currentText(), mode = mode, memoryLoc = memoryLoc, args = Args)
                        self.GPUQueue.put(an_action)
                        message = mode + " successfully finished..."
                    break
                except:
                    print('waiting for camera data...')
                    
        an_action = AODOAction('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        print(message)
        return message
    
            
    def RptScan(self, mode):
        # an_action = DnSAction('Clear')
        # self.DnSQueue.put(an_action)
        frame_rate = self.ui.FrameRate_DH.value()
        self.ui.FrameRate_DH.setValue(10)
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
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
                memoryLoc = an_action.action
                # print(memoryLoc)
                data_backs += 1
                if memoryLoc < self.ui.DisplayRatio.value():
                    ######################################### display data
                    if self.ui.FFTDevice.currentText() in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        self.data = self.Memory[memoryLoc].copy()
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            print('spectral data all zeros!')
                            # self.ui.PrintOut.append('spectral data all zeros!')
                            self.log.write('spectral data all zeros!')
                            message = 'spectral data all zeros!'
                        else:
                            an_action = DnSAction(mode, self.data, raw=True) # data in Memory[memoryLoc]
                            self.DnSQueue.put(an_action)
                            message = mode + " successfully finished..."
                    else:
                        # In other modes, do FFT first
                        if self.GPUQueue.qsize() == 0:
                            an_action = GPUAction(self.ui.FFTDevice.currentText(), mode, memoryLoc)
                            self.GPUQueue.put(an_action)
                        else:
                            skipped_fft_actions += 1
                        message = mode + " successfully finished..."
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
        message = message + str(data_backs)+ ' data received by weaver'
        if skipped_fft_actions:
            message = message + f", {skipped_fft_actions} stale continuous FFT action(s) skipped"
        self.log.write(message)
        print(message)
        self.drain_continuous_backlog(reason=f"after {mode}")
        an_action = GPUAction('display_FFT_actions')
        self.GPUQueue.put(an_action)
        an_action = GPUAction('display_counts', args = mode)
        self.GPUQueue.put(an_action)
        self.ui.FrameRate_DH.setValue(frame_rate)
        return message
  

    def PlatePreScan(self, args):
        self.overlay_images = {}
        self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= args
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        BlineAVG = self.ui.BlineAVG.value()
        self.ui.BlineAVG.setValue(1)
        self.ui.RunButton.setChecked(True)
        for isample in self.sample_centers:
            if self.ui.RunButton.isChecked():
                self.ui.NextSampleButton.setText('扫描中，请等待')
                self.ui.RepeatSampleButton.setText('扫描中，请等待')
                self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == isample['sample_id']]
                self.display_initial_scan_overlay(isample['sample_id'], self.raw_img, self.pixel_polygons)
                
                # User stopped continuousBline, then we do Mosaic scan for this sample
                self.AdjustZstage(isample['sample_id'])
    
                try:
                    message = self.iterate_FOVs('PlatePreScan')
                finally:
                    self.restore_y_geometry_after_correction()
                self.ui.NextSampleButton.setText('下一个样品')
                self.ui.RepeatSampleButton.setText('重新扫描')
                while (not self.ui.NextSampleButton.isChecked()) and self.ui.RunButton.isChecked():
                    if self.ui.RepeatSampleButton.isChecked():
                        self.ui.NextSampleButton.setText('扫描中，请等待')
                        self.ui.RepeatSampleButton.setText('扫描中，请等待')
                        self.process_mosaic_correction()
                        self.AdjustZstage(isample['sample_id'])
                        try:
                            message = self.iterate_FOVs('PlatePreScan')
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
                lower_id_locations = [loc for loc in self.FOV_locations 
                                       if loc.get('sample_id') < isample['sample_id']]
                
                higher_id_locations = [loc for loc in self.FOV_locations 
                                       if loc.get('sample_id') > isample['sample_id']]
            
                # 2. Combine them back together
                self.FOV_locations = lower_id_locations + self.CurrentSampleLocations + higher_id_locations
                
        
        # save self.FOV_locations, self.sample_centers, self.overlay_images
        self.save_session_data(self.ui.DIR.toPlainText()+'/Mosaic')
        self.ui.NextSampleButton.setText('扫描结束')
        self.ui.RepeatSampleButton.setText('扫描结束')
        self.ui.BlineAVG.setValue(BlineAVG)
        return(message)
            
    def PlateScan(self, args):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= args
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        # print(self.sample_centers)
        # print(self.FOV_locations)
        for isample in self.sample_centers:
            if self.ui.RunButton.isChecked():
                self.ui.sampleSelector.setCurrentIndex(isample['sample_id']-1)
                self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == isample['sample_id']]
                # print('self.CurrentSampleLocations', self.CurrentSampleLocations)
                self.display_sample_overlay(isample['sample_id'])
                try:
                    self.iterate_FOVs('PlateScan')
                finally:
                    self.restore_y_geometry_after_correction()
                if self.ui.Save.isChecked():
                    an_action = GPUAction('IncrementSampleID')
                    self.GPUQueue.put(an_action)
                message = 'PlateScan successfully finished'
            else:
                message = 'User stopped PlateScan'
        return(message)   
    
    
    def WellScan(self, args):
        self.ui.MosaicLabel.clear()
        # self.FOV_locations, self.sample_centers, self.raw_img, self.pixel_polygons= args
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        sample_id = self.ui.sampleSelector.currentIndex()+1
        self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == sample_id]
        self.display_sample_overlay(sample_id)
        try:
            message = self.iterate_FOVs('WellScan')
        finally:
            self.restore_y_geometry_after_correction()
        return(message) 

    def AdjustZstage(self, sample_id):
        self.clear_mosaic_display()
        isample = self.sample_centers[sample_id-1]
        # move to center position of this sample
        self.ui.XPosition.setValue(isample['x'])
        an_action = AODOAction('Xmove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        self.ui.YPosition.setValue(isample['y'])
        an_action = AODOAction('Ymove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        self.ui.ZPosition.setValue(isample['z'])
        an_action = AODOAction('Zmove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        # do continuous scan to display Bline
        self.ui.ACQMode.setCurrentText('ContinuousBline')
        self.InitMemory()
        self.ui.RunButton.setChecked(True)
        self.ui.RunButton.setText('点击开始扫描')
        self.RptScan(mode = 'ContinuousBline')
        # User can move Z stage up and down to put sample at focus
        for ii, item in enumerate(self.CurrentSampleLocations):
            self.CurrentSampleLocations[ii]['z'] = self.ui.ZPosition.value()
        
        self.ui.RunButton.setText('Stop')
        self.ui.RunButton.setChecked(True)
        
    def iterate_FOVs(self, mode):
        # print(self.CurrentSampleLocations)
        self.clear_mosaic_display()
        self.drain_continuous_backlog(reason=f"before {mode}")
        self.apply_y_geometry_from_locations()
        # move to position of this FOV
        iFOV = self.CurrentSampleLocations[0]
        self.ui.XPosition.setValue(iFOV['x'])
        an_action = AODOAction('Xmove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        self.ui.YPosition.setValue(iFOV['y'])
        an_action = AODOAction('Ymove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        self.ui.ZPosition.setValue(iFOV['z'])
        an_action = AODOAction('Zmove2')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        
        Xpixels = self.ui.AlinesPerBline.value()//self.ui.AlineAVG.value()
        Ypixels = self.ui.Ypixels.value()
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        an_action = GPUAction('Init_Mosaic', args = [self.CurrentSampleLocations, (Xpixels, Ypixels), (XFOV, YFOV)]) 
        self.GPUQueue.put(an_action)
        self.ui.ACQMode.setCurrentText(mode)
        self.InitMemory()
        for iFOV in self.CurrentSampleLocations:
            if self.ui.RunButton.isChecked():
                # print(iFOV['x'],iFOV['y'])
                # move to position of this FOV
                self.ui.XPosition.setValue(iFOV['x'])
                an_action = AODOAction('Xmove2')
                self.AODOQueue.put(an_action)
                self.StagebackQueue.get()
                self.ui.YPosition.setValue(iFOV['y'])
                an_action = AODOAction('Ymove2')
                self.AODOQueue.put(an_action)
                self.StagebackQueue.get()
                self.ui.ZPosition.setValue(iFOV['z'])
                an_action = AODOAction('Zmove2')
                self.AODOQueue.put(an_action)
                self.StagebackQueue.get()
                
                # do FiniteCscan at this position
                self.SingleScan(mode = 'Process_Mosaic', Args = [self.CurrentSampleLocations, iFOV])
                if self.ui.Save.isChecked():
                    an_action = GPUAction('IncrementTileNum')
                    self.GPUQueue.put(an_action)
                # handle pause action
                if self.ui.PauseButton.isChecked():
                    # wait until stop button or pause button is clicked
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(1)
                        print('waiting')
                message = 'iterate FOV successfully finished'
            else:
                message = 'User stopped'
        return(message)

    def sample_fov_locations(self, sample_id):
        if getattr(self, "CurrentSampleLocations", None):
            current_ids = {loc.get('sample_id') for loc in self.CurrentSampleLocations}
            if sample_id in current_ids:
                return [loc for loc in self.CurrentSampleLocations if loc.get('sample_id') == sample_id]
        return [loc for loc in self.FOV_locations if loc.get('sample_id') == sample_id]

    def display_initial_scan_overlay(self, sample_id, raw_img, pixel_polygons):
        """Stores USB overlay source data and renders it at the current label size."""
        self.overlay_images[sample_id] = {
            'type': 'usb_roi',
            'raw_img': raw_img,
            'pixel_polygons': pixel_polygons,
        }
        self.display_sample_overlay(sample_id)

    def display_sample_overlay(self, sample_id):
        source = self.overlay_images.get(sample_id)
        if source is None:
            self.ui.MosaicLabel.clear()
            return
        if isinstance(source, QPixmap):
            self.ui.MosaicLabel.setPixmap(source)
            return
        if source.get('type') == 'usb_roi':
            self.render_usb_roi_overlay(sample_id, source['raw_img'], source['pixel_polygons'])
        elif source.get('type') == 'mosaic_correction':
            self.render_mosaic_correction_overlay(sample_id, source)

    def mosaic_label_render_size(self):
        QApplication.processEvents()
        label_w = self.ui.MosaicLabel.width()
        label_h = self.ui.MosaicLabel.height()
        if label_w < 100 or label_h < 100:
            blank = QPixmap(300, 300)
            blank.fill(Qt.black)
            self.ui.MosaicLabel.setPixmap(blank)
            QApplication.processEvents()
            label_w = self.ui.MosaicLabel.width()
            label_h = self.ui.MosaicLabel.height()
        if label_w < 100 or label_h < 100:
            return 300, 300
        upscale = max(1, min(2, int(np.ceil(300 / max(label_w, label_h)))))
        return int(label_w * upscale), int(label_h * upscale)

    def render_usb_roi_overlay(self, sample_id, raw_img, pixel_polygons):
        poly_pts = pixel_polygons[sample_id - 1]
        poly_np = np.array(poly_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly_np)
        
        pad = 150
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(raw_img.shape[1], x + w + pad), min(raw_img.shape[0], y + h + pad)
        crop_img = raw_img[y1:y2, x1:x2].copy()
        
        rgb_img = np.ascontiguousarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        h_v, w_v, ch = rgb_img.shape
        qt_img = QImage(rgb_img.tobytes(), w_v, h_v, ch * w_v, QImage.Format_RGB888).copy()
        base_pixmap = QPixmap.fromImage(qt_img)

        label_w, label_h = self.mosaic_label_render_size()
        final_buffer = QPixmap(label_w, label_h)
        final_buffer.fill(Qt.black)

        painter = QPainter(final_buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        scale = min(label_w / w_v, label_h / h_v)
        sw, sh = int(w_v * scale), int(h_v * scale)
        dx, dy = (label_w - sw) // 2, (label_h - sh) // 2
        painter.drawPixmap(dx, dy, sw, sh, base_pixmap)

        def to_ui(px, py):
            return dx + (px - x1) * scale, dy + (py - y1) * scale

        # Draw Sample Polygon
        painter.setPen(QPen(QColor(0, 120, 255), 3))
        for i in range(len(poly_pts)):
            p1 = to_ui(*poly_pts[i])
            p2 = to_ui(*poly_pts[(i + 1) % len(poly_pts)])
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # Draw FOV Grid
        usb_pixel_size = USB_PIXEL_SIZE_MM
        fov_x_px = self.ui.XLength.value() / usb_pixel_size
        fov_y_px = self.ui.YLength.value() / usb_pixel_size

        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for fov in self.sample_fov_locations(sample_id):
            cx_px, cy_px = stage_to_usb_image(fov['x'], fov['y'], raw_img.shape[1])
            loc_y_fov = fov.get('y_length_mm', self.ui.YLength.value())
            fov_y_px = loc_y_fov / usb_pixel_size
            tl = to_ui(cx_px - fov_y_px/2, cy_px - fov_x_px/2)
            br = to_ui(cx_px + fov_y_px/2, cy_px + fov_x_px/2)
            painter.drawRect(QRectF(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

        painter.end()
        self.ui.MosaicLabel.setPixmap(final_buffer)

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
        first_loc = self.CurrentSampleLocations[0]
        y_length = first_loc.get("y_length_mm", None)
        y_pixels = first_loc.get("y_pixels", None)
        if y_length is None or y_pixels is None:
            return
        self.apply_y_geometry_for_correction(float(y_length), int(y_pixels))

    def current_location_y_length(self):
        if self.CurrentSampleLocations:
            y_length = self.CurrentSampleLocations[0].get("y_length_mm", None)
            if y_length is not None:
                return float(y_length)
        return self.ui.YLength.value()

    def process_mosaic_correction(self ):
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
        mm_polygons = []
        px_w_mm = self.ui.XStepSize.value() / 1000.0
        px_h_mm = self.ui.YStepSize.value() / 1000.0
        
        # Find the anchor used by the mosaic viewer to convert back to mm
        xs_orig = [p['x'] for p in self.CurrentSampleLocations]
        ys_orig = [p['y'] for p in self.CurrentSampleLocations]
        v_anchor_x = min(xs_orig) - (self.ui.XLength.value() / 2)
        source_y_length = self.current_location_y_length()
        v_anchor_y = min(ys_orig) - (source_y_length / 2)

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

        for ii, poly in enumerate(new_polygons, start=1):
            raw_poly = np.array(poly, dtype=float)
            raw_min = np.min(raw_poly, axis=0)
            raw_max = np.max(raw_poly, axis=0)
            mm_poly = [(p[0] * px_w_mm + v_anchor_x, p[1] * px_h_mm + v_anchor_y) for p in poly]
            mm_polygons.append(mm_poly)
            mm_poly_array = np.array(mm_poly, dtype=float)
            mm_min = np.min(mm_poly_array, axis=0)
            mm_max = np.max(mm_poly_array, axis=0)
            if self.debug_mosaic_correction:
                print(
                    "Mosaic correction polygon: "
                    f"#{ii}, vertices={len(poly)}, "
                    f"raw_bounds=(x:{raw_min[0]:.2f}-{raw_max[0]:.2f}, y:{raw_min[1]:.2f}-{raw_max[1]:.2f}), "
                    f"raw_size=({raw_max[0] - raw_min[0]:.2f}, {raw_max[1] - raw_min[1]:.2f}), "
                    f"mm_bounds=(x:{mm_min[0]:.3f}-{mm_max[0]:.3f}, y:{mm_min[1]:.3f}-{mm_max[1]:.3f}), "
                    f"mm_size=({mm_max[0] - mm_min[0]:.3f}, {mm_max[1] - mm_min[1]:.3f})"
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
        for loc in scan_plan.fov_locations:
            loc["y_length_mm"] = scan_plan.y_length_mm
            loc["y_pixels"] = scan_plan.y_pixels
        self.apply_y_geometry_for_correction(scan_plan.y_length_mm, scan_plan.y_pixels)

        # Re-generate the scan grid and the new overlay
        self.generate_new_scan_grid(current_id, mm_polygons, new_polygons, scan_plan, source_y_length)
        self.ui.mosaic_viewer.clear_polygons()

    def generate_new_scan_grid(self, sample_id, mm_polygons, pixel_polygons, scan_plan=None, source_y_length=None):
        """Generates FOVs and creates a new overlay encompassing both mosaic and new polygon."""
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        px_w_mm = self.ui.XStepSize.value() / 1000.0
        px_h_mm = self.ui.YStepSize.value() / 1000.0
        new_fov_locations = scan_plan.fov_locations if scan_plan is not None else []
        
        if scan_plan is not None:
            self.sample_centers[sample_id-1]['x'] = scan_plan.center_x
            self.sample_centers[sample_id-1]['y'] = scan_plan.center_y
            if self.debug_mosaic_correction:
                print(
                    "Mosaic correction FOV grid result: "
                    f"sample_id={sample_id}, accepted_tiles={len(new_fov_locations)}, "
                    f"YFOV={YFOV:.3f}"
                )

        # 2. Setup Coordinate System and Bounding Box
        mos_img = np.ascontiguousarray(self.ui.mosaic_viewer.adj)
        orig_h, orig_w = mos_img.shape
        
        # Current Mosaic physical boundaries
        xs_orig = [p['x'] for p in self.CurrentSampleLocations]
        ys_orig = [p['y'] for p in self.CurrentSampleLocations]
        mos_min_x = min(xs_orig) - (XFOV / 2)
        mosaic_source_y_fov = source_y_length if source_y_length is not None else YFOV
        mos_min_y = min(ys_orig) - (mosaic_source_y_fov / 2)
        mos_max_x = mos_min_x + (orig_w * px_w_mm)
        mos_max_y = mos_min_y + (orig_h * px_h_mm)

        # Polygon physical boundaries
        all_poly_pts = [pt for poly in mm_polygons for pt in poly]
        poly_min_x, poly_min_y = min(p[0] for p in all_poly_pts), min(p[1] for p in all_poly_pts)
        poly_max_x, poly_max_y = max(p[0] for p in all_poly_pts), max(p[1] for p in all_poly_pts)

        # Global Bounding Box (Encompasses both)
        global_min_x = min(mos_min_x, poly_min_x) - 1.0 # 1mm margin
        global_min_y = min(mos_min_y, poly_min_y) - 1.0
        global_max_x = max(mos_max_x, poly_max_x) + 1.0
        global_max_y = max(mos_max_y, poly_max_y) + 1.0

        # Create the canvas size based on global mm dimensions
        canvas_w_px = int((global_max_x - global_min_x) / px_w_mm)
        canvas_h_px = int((global_max_y - global_min_y) / px_h_mm)

        self.overlay_images[sample_id] = {
            'type': 'mosaic_correction',
            'mos_img': mos_img,
            'mm_polygons': mm_polygons,
            'fov_locations': [dict(fov) for fov in new_fov_locations],
            'px_w_mm': px_w_mm,
            'px_h_mm': px_h_mm,
            'XFOV': XFOV,
            'YFOV': YFOV,
            'mosaic_bounds': (mos_min_x, mos_min_y, mos_max_x, mos_max_y),
            'global_bounds': (global_min_x, global_min_y, global_max_x, global_max_y),
            'canvas_size_px': (canvas_w_px, canvas_h_px),
        }
        self.render_mosaic_correction_overlay(sample_id, self.overlay_images[sample_id])
        self.CurrentSampleLocations = new_fov_locations        

    def render_mosaic_correction_overlay(self, sample_id, source):
        mos_img = np.ascontiguousarray(source['mos_img'])
        orig_h, orig_w = mos_img.shape
        px_w_mm = source['px_w_mm']
        px_h_mm = source['px_h_mm']
        XFOV = source['XFOV']
        YFOV = source['YFOV']
        mm_polygons = source['mm_polygons']
        new_fov_locations = source['fov_locations']
        mos_min_x, mos_min_y, _, _ = source['mosaic_bounds']
        global_min_x, global_min_y, _, _ = source['global_bounds']
        canvas_w_px, canvas_h_px = source['canvas_size_px']

        label_w, label_h = self.mosaic_label_render_size()
        final_buffer = QPixmap(label_w, label_h)
        final_buffer.fill(Qt.black)

        painter = QPainter(final_buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        scale_w = label_w / canvas_w_px
        scale_h = label_h / canvas_h_px
        sw, sh = int(canvas_w_px * scale_w), int(canvas_h_px * scale_h)
        dx, dy = (label_w - sw) // 2, (label_h - sh) // 2

        qt_mos = QImage(mos_img.tobytes(), orig_w, orig_h, orig_w, QImage.Format_Grayscale8).copy()
        mos_pixmap = QPixmap.fromImage(qt_mos)

        mos_offset_x = (mos_min_x - global_min_x) / px_w_mm
        mos_offset_y = (mos_min_y - global_min_y) / px_h_mm
        painter.drawPixmap(
            int(dx + mos_offset_x * scale_w),
            int(dy + mos_offset_y * scale_h),
            int(orig_w * scale_w),
            int(orig_h * scale_h),
            mos_pixmap,
        )

        painter.setPen(QPen(QColor(0, 255, 0), 1))
        for fov in new_fov_locations:
            loc_y_fov = fov.get('y_length_mm', YFOV)
            tl_x = (fov['x'] - XFOV/2 - global_min_x) / px_w_mm
            tl_y = (fov['y'] - loc_y_fov/2 - global_min_y) / px_h_mm
            br_x = (fov['x'] + XFOV/2 - global_min_x) / px_w_mm
            br_y = (fov['y'] + loc_y_fov/2 - global_min_y) / px_h_mm
            painter.drawRect(QRectF(dx + tl_x * scale_w, dy + tl_y * scale_h, (br_x-tl_x)*scale_w, (br_y-tl_y)*scale_h))

        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for mm_poly in mm_polygons:
            for i in range(len(mm_poly)):
                p1_mm, p2_mm = mm_poly[i], mm_poly[(i+1)%len(mm_poly)]
                x1_ui = dx + ((p1_mm[0] - global_min_x) / px_w_mm) * scale_w
                y1_ui = dy + ((p1_mm[1] - global_min_y) / px_h_mm) * scale_h
                x2_ui = dx + ((p2_mm[0] - global_min_x) / px_w_mm) * scale_w
                y2_ui = dy + ((p2_mm[1] - global_min_y) / px_h_mm) * scale_h
                painter.drawLine(int(x1_ui), int(y1_ui), int(x2_ui), int(y2_ui))

        painter.end()
        self.ui.MosaicLabel.setPixmap(final_buffer)
       
    def ZstageRepeatibility(self):
        mode = self.ui.ACQMode.currentText()
        device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText('SingleAline')
        self.ui.FFTDevice.setCurrentText('GPU')
        current_Xposition = self.ui.XPosition.value()
        current_Yposition = self.ui.YPosition.value()
        current_Zposition = self.ui.ZPosition.value()
        iteration = 50
        for i in range(iteration):
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            # measure ALine
            message = self.SingleScan(self.ui.ACQMode.currentText())
            self.log.write(message)
            # self.ui.PrintOut.append(message)
            failed_times = 0
            while message != self.ui.ACQMode.currentText()+" successfully finished...":
                failed_times+=1
                if failed_times > 10:
                    self.ui.ACQMode.setCurrentText(mode)
                    self.ui.FFTDevice.setCurrentText(device)
                    self.ui.Gotozero.setChecked(False)
                    return message
                message = self.SingleScan(self.ui.ACQMode.currentText())
                self.log.write(message)
                # self.ui.PrintOut.append(message)
                time.sleep(1)
            time.sleep(0.1)
            
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            self.ui.ZPosition.setValue(5)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            # move to clear XY position
            self.ui.XPosition.setValue(45)
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            self.ui.YPosition.setValue(20)
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
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
                message = 'Stage test stopped by user...'
                break
            self.ui.ZPosition.setValue(5)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            # move to original XY position
            self.ui.XPosition.setValue(current_Xposition)
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            self.ui.YPosition.setValue(current_Yposition)
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            if not self.ui.ZstageTest.isChecked():
                message = 'Stage test stopped by user...'
                break
            # move Z stage up
            self.ui.ZPosition.setValue(current_Zposition)
            an_action = AODOAction('Zmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
        self.ui.ZstageTest.setChecked(False)
        # self.weaverBackQueue.put(0)
        self.ui.ACQMode.setCurrentText(mode)
        self.ui.FFTDevice.setCurrentText(device)
        return 'Stage test successfully finished...'
        
    def ZstageRepeatibility2(self):
        mode = self.ui.ACQMode.currentText()
        device = self.ui.FFTDevice.currentText()
        self.ui.FFTDevice.setCurrentText('GPU')
        self.ui.ACQMode.setCurrentText('SingleAline')
        current_position = self.ui.ZPosition.value() # this is the target Z pos in this test
        iteration = 100
        for i in range(iteration):
            if not self.ui.ZstageTest2.isChecked():
                break
            # measure ALine
            message = self.SingleScan(self.ui.ACQMode.currentText())
            self.log.write(message)
            while message != self.ui.ACQMode.currentText()+" successfully finished...":
                message = self.SingleScan(self.ui.ACQMode.currentText())
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
            self.ui.statusbar.showMessage(message)
            if message != 'gotozero success...':
                message = 'go to zero failed, abort test...'
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
            self.ui.ACQMode.setCurrentText('SingleAline')
            self.ui.FFTDevice.setCurrentText('GPU')
            
            
        self.ui.ZstageTest2.setChecked(False)
        # self.weaverBackQueue.put(0)
        self.ui.ACQMode.setCurrentText(mode)
        self.ui.FFTDevice.setCurrentText(device)

            
        
    def get_background(self):
        print('start getting background...')
        mode = self.ui.ACQMode.currentText()
        device = self.ui.FFTDevice.currentText()
        BAvg = self.ui.BlineAVG.value()
        self.ui.ACQMode.setCurrentText('FiniteBline')
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(100)
        ############################# measure an Aline
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(self.ui.ACQMode.currentText())
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
        self.ui.ACQMode.setCurrentText(mode)
        self.ui.FFTDevice.setCurrentText(device)
        self.ui.BlineAVG.setValue(BAvg)
        return 'background measruement success...'
    
    def get_surfCurve(self):
        
        print('start getting background...')
        mode = self.ui.ACQMode.currentText()
        device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText('FiniteBline')
        self.ui.FFTDevice.setCurrentText('GPU')
        self.ui.DSing.setChecked(True)
        ############################# measure an Cscan
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan('SingleCscan')
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
        self.ui.ACQMode.setCurrentText(mode)
        self.ui.FFTDevice.setCurrentText(device)
        self.ui.DSing.setChecked(False)
        return 'surface measruement success...'
    
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
