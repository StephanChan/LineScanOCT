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
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF
# from matplotlib.path import Path
from shapely.geometry import Polygon # Shapely is best for intersection math
# from scipy.signal import hilbert
import datetime
import pickle
import cv2
# axial pixel size, measure with a microscope glass slide
global ZPIXELSIZE
ZPIXELSIZE = 4.4 # unit: um

class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.mosaic = None
        self.overlay_images = {}
        self.FOV_locations = {}
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
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action in ['FiniteBline', 'FiniteAline', 'FiniteCscan']:
                    self.InitMemory()
                    message = self.SingleScan(self.item.action)
                    if self.item.action in ['FiniteCscan'] and self.ui.Save.isChecked():
                        an_action = GPUAction('IncrementCscanNum')
                        self.GPUQueue.put(an_action)
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
                    self.ui.statusbar.showMessage(message)
                    self.log.write(message)
                elif self.item.action == 'PlateScan':
                    # make directories
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/aip'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/aip')
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
                    self.ui.statusbar.showMessage(message)
                    self.log.write(message)
                    if self.ui.Save.isChecked():
                        an_action = GPUAction('IncrementSampleID')
                        self.GPUQueue.put(an_action)

                elif self.item.action == 'ZstageRepeatibility':
                    message = self.ZstageRepeatibility()
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action == 'get_background':
                    message = self.get_background()
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                elif self.item.action == 'get_surface':
                    message = self.get_surfCurve()
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                    
                else:
                    message = 'Weaver thread is doing something invalid: '+self.item.action
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)

            except Exception as error:
                message = "An error occurred in"+  self.item.action + "\n"
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
        self.ui.statusbar.showMessage(self.exit_message)
            
    
    def InitMemory(self):
        #################################################################
        # get number samplers per Aline
        samples = self.ui.NSamples.value()
            
        AlinesPerBline = self.ui.AlinesPerBline.value()
        # print(self.ui.PixelFormat_display.text())
        if self.ui.PixelFormat_display.text() in ['Mono8']:
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
        print('\n')
        print('Camera config took: ',round(t1-t0,3),'sec')
        print('Galvo board config took: ',round(t2-t1,3),'sec')
        print('Camera start took: ',round(t3-t2,3),'sec')
        print('Galvo board start took: ',round(t4-t3,3),'sec')
        # print('current dbackqueue size:', self.DbackQueue.qsize())
        print('\n')
        message = 'User stopped SingleScan'
        for iAcq in range(self.NAcq):
            start = time.time()
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            print('waiting for camera data...')
            while self.ui.RunButton.isChecked():
                try:
                    an_action = self.DatabackQueue.get(timeout = 5)
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
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
        # config AODO
        an_action = AODOAction('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        data_backs = 0 # count number of data backs

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
                an_action = self.DatabackQueue.get(timeout=3) # never time out
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
                        an_action = GPUAction(self.ui.FFTDevice.currentText(), mode, memoryLoc)
                        self.GPUQueue.put(an_action)
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
        self.log.write(message)
        print(message)
        an_action = GPUAction('display_FFT_actions')
        self.GPUQueue.put(an_action)
        an_action = DnSAction('display_counts', args = mode)
        self.DnSQueue.put(an_action)
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
    
                message = self.iterate_FOVs('PlatePreScan')
                self.ui.NextSampleButton.setText('下一个样品')
                self.ui.RepeatSampleButton.setText('重新扫描')
                while (not self.ui.NextSampleButton.isChecked()) and self.ui.RunButton.isChecked():
                    if self.ui.RepeatSampleButton.isChecked():
                        self.ui.NextSampleButton.setText('扫描中，请等待')
                        self.ui.RepeatSampleButton.setText('扫描中，请等待')
                        self.process_mosaic_correction()
                        self.AdjustZstage(isample['sample_id'])
                        message = self.iterate_FOVs('PlatePreScan')
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
                self.ui.MosaicLabel.setPixmap(self.overlay_images[isample['sample_id']])
                self.iterate_FOVs('PlateScan')
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
        self.ui.MosaicLabel.setPixmap(self.overlay_images[sample_id])
        message = self.iterate_FOVs('WellScan')
        return(message) 
    
            
    def AdjustZstage(self, sample_id):
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
        an_action = DnSAction('Init_Mosaic', args = [self.CurrentSampleLocations, (Xpixels, Ypixels), (XFOV, YFOV)]) 
        self.DnSQueue.put(an_action)
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

   
    def display_initial_scan_overlay(self, sample_id, raw_img, pixel_polygons):
        """Displays initial USB scan crop and saves to self.overlay_images."""
        poly_pts = pixel_polygons[sample_id - 1]
        poly_np = np.array(poly_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly_np)
        
        pad = 150
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(raw_img.shape[1], x + w + pad), min(raw_img.shape[0], y + h + pad)
        crop_img = raw_img[y1:y2, x1:x2].copy()
        
        rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        h_v, w_v, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w_v, h_v, ch * w_v, QImage.Format_RGB888).copy()
        base_pixmap = QPixmap.fromImage(qt_img)

        label_w = self.ui.MosaicLabel.width()
        label_h = self.ui.MosaicLabel.height()
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
        usb_pixel_size = 0.02 
        fov_w_px = self.ui.XLength.value() / usb_pixel_size
        fov_h_px = self.ui.YLength.value() / usb_pixel_size

        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for fov in self.CurrentSampleLocations:
            if fov['sample_id'] == sample_id:
                cx_px, cy_px = fov['x'] / usb_pixel_size, fov['y'] / usb_pixel_size
                tl = to_ui(cx_px - fov_w_px/2, cy_px - fov_h_px/2)
                br = to_ui(cx_px + fov_w_px/2, cy_px + fov_h_px/2)
                painter.drawRect(QRectF(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

        painter.end()
        
        # Save to global storage and display
        self.overlay_images[sample_id] = final_buffer
        self.ui.MosaicLabel.setPixmap(final_buffer)

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
        v_anchor_y = min(ys_orig) - (self.ui.YLength.value() / 2)

        for poly in new_polygons:
            mm_poly = [(p[0] * px_w_mm + v_anchor_x, p[1] * px_h_mm + v_anchor_y) for p in poly]
            mm_polygons.append(mm_poly)

        # Re-generate the scan grid and the new overlay
        self.generate_new_scan_grid(current_id, mm_polygons, new_polygons)

    def generate_new_scan_grid(self, sample_id, mm_polygons, pixel_polygons):
        """Generates FOVs and creates a new overlay encompassing both mosaic and new polygon."""
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        px_w_mm = self.ui.XStepSize.value() / 1000.0
        px_h_mm = self.ui.YStepSize.value() / 1000.0
        new_fov_locations = []
        
        # 1. Math logic for FOV generation (Shapely)
        for mm_poly_pts in mm_polygons:
            roi_poly = Polygon(mm_poly_pts)
            p_xs, p_ys = zip(*mm_poly_pts)
            x_range = np.arange(min(p_xs), max(p_xs)+XFOV/2, XFOV)
            y_range = np.arange(min(p_ys), max(p_ys)+YFOV/2, YFOV)

            centroid = roi_poly.centroid
            self.sample_centers[sample_id-1]['x'] = centroid.x
            self.sample_centers[sample_id-1]['y'] = centroid.y

            for cx in x_range:
                for cy in y_range:
                    tile_poly = Polygon([
                        (cx-XFOV/2, cy-YFOV/2), (cx+XFOV/2, cy-YFOV/2),
                        (cx+XFOV/2, cy+YFOV/2), (cx-XFOV/2, cy+YFOV/2)
                    ])
                    if tile_poly.intersects(roi_poly):
                        new_fov_locations.append({'sample_id': sample_id, 'x': round(cx, 3), 'y': round(cy, 3)})

        # 2. Setup Coordinate System and Bounding Box
        mos_img = self.ui.mosaic_viewer.adj 
        orig_h, orig_w = mos_img.shape
        
        # Current Mosaic physical boundaries
        xs_orig = [p['x'] for p in self.CurrentSampleLocations]
        ys_orig = [p['y'] for p in self.CurrentSampleLocations]
        mos_min_x = min(xs_orig) - (XFOV / 2)
        mos_min_y = min(ys_orig) - (YFOV / 2)
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

        # 3. Render the Overlay
        label_w, label_h = self.ui.MosaicLabel.width(), self.ui.MosaicLabel.height()
        final_buffer = QPixmap(label_w, label_h)
        final_buffer.fill(Qt.black)

        painter = QPainter(final_buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Scale to fit UI Label
        scale_w = label_w / canvas_w_px
        scale_h = label_h / canvas_h_px
        sw, sh = int(canvas_w_px * scale_w), int(canvas_h_px * scale_h)
        dx, dy = (label_w - sw) // 2, (label_h - sh) // 2

        # Draw stretched Mosaic at its relative position
        
        qt_mos = QImage(mos_img.data, orig_w, orig_h, orig_w, QImage.Format_Grayscale8).copy()
        mos_pixmap = QPixmap.fromImage(qt_mos)

        mos_offset_x = (mos_min_x - global_min_x) / px_w_mm
        mos_offset_y = (mos_min_y - global_min_y) / px_h_mm
        painter.drawPixmap(int(dx + mos_offset_x * scale_w), int(dy + mos_offset_y * scale_h), 
                           int(orig_w * scale_w), int(orig_h * scale_h), mos_pixmap)

        # Draw New FOV Overlays
        painter.setPen(QPen(QColor(0, 255, 0), 1))
        for fov in new_fov_locations:
            tl_x = (fov['x'] - XFOV/2 - global_min_x) / px_w_mm
            tl_y = (fov['y'] - YFOV/2 - global_min_y) / px_h_mm
            br_x = (fov['x'] + XFOV/2 - global_min_x) / px_w_mm
            br_y = (fov['y'] + YFOV/2 - global_min_y) / px_h_mm
            painter.drawRect(QRectF(dx + tl_x * scale_w, dy + tl_y * scale_h, (br_x-tl_x)*scale_w, (br_y-tl_y)*scale_h))

        # Draw New Polygons
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
        
        # Update global storage and UI
        self.overlay_images[sample_id] = final_buffer
        self.ui.MosaicLabel.setPixmap(final_buffer)
        self.CurrentSampleLocations = new_fov_locations        
    
    
    def identify_agar(self, cscan, stripes, cscans):
        value = np.mean(cscan,1)
        # reshape into Ypixels x Xpixels matrix
        value = value.reshape([self.ui.AlinesPerBline.value()*self.ui.BlineAVG.value(),\
                               self.ui.NSamples.value()*self.ui.AlineAVG.value()+ \
                               self.ui.PreClock.value()*2 + self.ui.PostClock.value()])
        # trim galvo fly-back data
        value = value[:,self.ui.PreClock.value():self.ui.PreClock.value()+\
                      self.ui.NSamples.value()*self.ui.AlineAVG.value()]
        # # downsample X dimension
        # value = value.reshape([self.ui.AlinesPerBline.value()*self.ui.BlineAVG.value(),\
        #                        self.ui.NSamples.value()*self.ui.AlineAVG.value()//self.Yds,self.Yds]).mean(-1)
        self.ui.tileMean.setValue(np.mean(value))
        if np.sum(value > self.ui.ThresholdValue.value())>value.shape[1]*value.shape[0]*0.05: 
            self.tile_flag[stripes - 1][cscans] = 1
            self.ui.TissueRadio.setChecked(True)
            self.tmp_cscan = self.tmp_cscan + cscan/100.0
        else:
            self.ui.TissueRadio.setChecked(False)
            
    def Focusing(self, cscan):
        ######################################################### find average slice surface
        cscan = cscan.reshape([self.ui.AlinesPerBline.value()*self.ui.BlineAVG.value(),\
                               self.ui.NSamples.value()*self.ui.AlineAVG.value()+ self.ui.PreClock.value()*2 + self.ui.PostClock.value(),\
                               self.ui.DepthRange.value()])
        bscan = cscan.mean(0)
        # remove galvo flayback data
        bscan = bscan[self.ui.PreClock.value():self.ui.PreClock.value()+self.ui.NSamples.value()*self.ui.AlineAVG.value(),:]
        # flatten surface
        if np.any(self.surfCurve):
            bscan_flatten = np.zeros(bscan.shape, dtype = np.float32)
            for xx in range(bscan_flatten.shape[0]):
                bscan_flatten[xx,0:bscan.shape[1]-self.surfCurve[xx]] = bscan[xx,self.surfCurve[xx]:]
            plt.figure()
            plt.imshow(bscan_flatten)
            plt.savefig('slice'+str(self.ui.CuSlice.value())+'surface.jpg')
            plt.close()
        else:
            bscan_flatten = bscan
        # find tile surface
        ascan = bscan_flatten.mean(0)
        
        surfAlinesPerBline = findchangept(ascan,1)

        ##########################################################
        self.ui.SurfAlinesPerBline.setValue(surfAlinesPerBline)
        message = 'tile surf is:'+str(surfAlinesPerBline)
        print(message)
        self.log.write(message)
        delta_z = (self.ui.SurfAlinesPerBline.value()-self.ui.SurfSet.value())*ZPIXELSIZE/1000.0
        self.ui.ZIncrease.setValue(delta_z)
        
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
        self.ui.BlineAVG.setValue(200)
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
        BLINE = self.data.reshape([Yrpt, Xpixels, self.ui.NSamples.value()])
        
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
        # 1. Initialize Camera
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
        while self.ui.RunButton.isChecked():
            ret, frame = cap.read()
            if not ret:
                break
    
            # 2. Process Image (Rotate/Flip/Crop to match your previous setup)
            frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)
            # Assuming the same crop as your SampleLocator
            frame = frame[630:3670, 180:2160]
            
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
        cap.release()
        
    
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
    
        # 2. Save Overlay Images
        # QPixmap must be saved as individual files (PNG/JPG)
        image_dir = os.path.join(folder_path, 'overlays')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    
        for sample_id, pixmap in self.overlay_images.items():
            file_path = os.path.join(image_dir, f'sample_{sample_id}_overlay.png')
            pixmap.save(file_path, "PNG")
        
        print(f"Session data saved to {folder_path}")
    
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
    
        # 2. Load Overlay Images
        self.overlay_images = {}
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