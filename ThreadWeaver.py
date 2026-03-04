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
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.path import Path
from shapely.geometry import Polygon # Shapely is best for intersection math
# from scipy.signal import hilbert
import datetime
import cv2
# axial pixel size, measure with a microscope glass slide
global ZPIXELSIZE
ZPIXELSIZE = 4.4 # unit: um

class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.mosaic = None
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
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                elif self.item.action in ['LocationCameraLive']:
                    self.live()

                elif self.item.action == 'PlateScan':
                    # make directories
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/aip'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/aip')
                    # if not os.path.exists(self.ui.DIR.toPlainText()+'/surf'):
                    #     os.mkdir(self.ui.DIR.toPlainText()+'/surf')
                    message = self.PlateScan(self.item.args)
                    # TODO: take an image, manually find tissue region, generate scan pattern
                    # for each scan region:
                        # do fast scan to identify tissue area
                        # adjust scan patern in X Y Z dimension
                        # do viability scan
                        
                    # # do fast pre-scan to identify tissue area
                    # self.InitMemory()
                    # message= self.PreMosaic()
                    # if message != 'success' :
                    #     message = "action aborted by user..."
                    # else:
                    #     self.InitMemory()
                    #     message = self.Mosaic()
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)

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
             elif self.ui.ACQMode.currentText() in ['FiniteCscan', 'PlateScan']:
                 if self.ui.DynCheckBox.isChecked():
                     self.Memory[ii]=np.zeros([self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = self.ui.Ypixels.value()
                 else:
                     self.Memory[ii]=np.zeros([self.ui.Ypixels.value()*self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = 1

        ###########################################################################################
        
    def SingleScan(self, mode, Args=[]):
        an_action = DnSAction('Clear')
        self.DnSQueue.put(an_action)
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

        for iAcq in range(self.NAcq):
            start = time.time()
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            print('waiting for camera data...')
            an_action = self.DatabackQueue.get() # never time out
            if an_action != 0:
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
            else:
                message = 'an_action is 0'
                    
        an_action = AODOAction('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        print(message)
        return message
    
            
    def RptScan(self, mode):
        an_action = DnSAction('Clear')
        self.DnSQueue.put(an_action)
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
                an_action = self.DatabackQueue.get(timeout=1) # never time out
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
  

    def PlateScan(self, args):
        self.FOV_locations, self.sample_centers = args
        if self.sample_centers is None:
            return
        self.ui.FFTDevice.setCurrentText('GPU')
        for isample in self.sample_centers:
            # move to center position of this sample
            self.ui.XPosition.setValue(isample['x'])
            an_action = AODOAction('Xmove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            self.ui.YPosition.setValue(isample['y'])
            an_action = AODOAction('Ymove2')
            self.AODOQueue.put(an_action)
            self.StagebackQueue.get()
            
            # do continuous scan to display Bline
            self.ui.ACQMode.setCurrentText('ContinuousBline')
            self.InitMemory()
            self.ui.RunButton.setChecked(True)
            self.RptScan(mode = 'ContinuousBline')
            # User can move Z stage up and down to put sample at focus
            
            # User stopped continuousBline, then we do Mosaic scan for this sample
            self.iterate_FOVs(isample)
            
            while not self.ui.NextSampleButton.isChecked():
                time.sleep(1)
                if self.ui.RepeatSampleButton.isChecked():
                    self.process_mosaic_correction(isample['sample_id'])
                    # 1. Remove all old entries matching this sample_id
                    # We keep everything that DOES NOT match the ID we are updating
                    lower_id_locations = [loc for loc in self.FOV_locations 
                                           if loc.get('sample_id') < isample['sample_id']]
                    
                    higher_id_locations = [loc for loc in self.FOV_locations 
                                           if loc.get('sample_id') > isample['sample_id']]
                
                    # 2. Combine them back together
                    self.FOV_locations = lower_id_locations + self.CurrentSampleLocations + higher_id_locations
                    
                    # print(self.FOV_locations)
                    
                    self.iterate_FOVs(isample)
                    self.ui.RepeatSampleButton.setChecked(False)
            self.ui.NextSampleButton.setChecked(False)
        
        message = 'PlateScan successfully finished'
        return(message)
            # plt.figure()
            # plt.imshow(SampleMosaic)
            # plt.show()
                
            
    def iterate_FOVs(self, isample):
        # User stopped continuousBline, then we do Mosaic scan for this sample
        self.CurrentSampleLocations = [ii for ii in self.FOV_locations if ii['sample_id'] == isample['sample_id']]
        # print(self.CurrentSampleLocations)
        Xpixels = self.ui.AlinesPerBline.value()//self.ui.AlineAVG.value()
        Ypixels = self.ui.Ypixels.value()
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        an_action = DnSAction('Init_Mosaic', args = [self.CurrentSampleLocations, (Xpixels, Ypixels), (XFOV, YFOV)]) 
        self.DnSQueue.put(an_action)
        self.ui.ACQMode.setCurrentText('PlateScan')
        self.InitMemory()
        for iFOV in self.CurrentSampleLocations:
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
            
            # do FiniteCscan at this position

            self.ui.RunButton.setChecked(True)
            self.SingleScan(mode = 'Process_Mosaic', Args = iFOV)
        # # After finishing this sample, analysis the Mosaic image
        # an_action = DnSAction('Return_mosaic') 
        # self.DnSQueue.put(an_action)
        # SampleMosaic = self.MosaicQueue.get()
        # # update locations for this sample if button clicked
   
    def process_mosaic_correction(self, sample_id):
        # 1. Get drawn polygons from the interactive widget (ensure they are finished)
        self.ui.mosaic_viewer.finish_region()
        new_polygons = self.ui.mosaic_viewer.polygons 
        
        if not new_polygons:
            print("No regions drawn on mosaic.")
            return
        
        # 2. Define the anchor (top-left of the first FOV center)
        xs = [p['x'] for p in self.CurrentSampleLocations]
        ys = [p['y'] for p in self.CurrentSampleLocations]
        
        # The mosaic [0,0] pixel corresponds to the center of the min_x/min_y FOV 
        # minus half the FOV physical size.
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        anchor_x = min(xs) - (XFOV / 2)
        anchor_y = min(ys) - (YFOV / 2)
        
        # 3. Calculate separate pixel sizes for X and Y
        # Based on your StepSize (resolution)
        px_w_mm = self.ui.XStepSize.value() / 1000.0
        px_h_mm = self.ui.YStepSize.value() / 1000.0
    
        # 4. Map pixel coordinates to microscope MM coordinates
        corrected_mm_polys = []
        for poly in new_polygons:
            mm_poly = [
                (anchor_x + p[0] * px_w_mm, anchor_y + p[1] * px_h_mm) 
                for p in poly
            ]
            corrected_mm_polys.append(mm_poly)
        
        # 5. Generate and Visualize
        self.generate_new_scan_grid(sample_id, corrected_mm_polys, new_polygons)
        print("New corrected regions ready for scanning.")
    
    def generate_new_scan_grid(self, sample_id, mm_polygons, pixel_polygons):
        
        XFOV = self.ui.XLength.value()
        YFOV = self.ui.YLength.value()
        px_w_mm = self.ui.XStepSize.value() / 1000.0
        px_h_mm = self.ui.YStepSize.value() / 1000.0

        new_fov_locations = []
        
        # --- 1. Generate FOVs (Corner-Check Overlap Logic) ---
        for mm_poly in mm_polygons:
            poly_path = Path(mm_poly)
            p_xs, p_ys = zip(*mm_poly)
            
            min_x, max_x = min(p_xs), max(p_xs)
            min_y, max_y = min(p_ys), max(p_ys)

            # Grid of centers
            x_range = np.arange(min_x + XFOV/2, max_x + XFOV, XFOV)
            y_range = np.arange(min_y + YFOV/2, max_y + YFOV, YFOV)

            for cx in x_range:
                for cy in y_range:
                    # Test Center + 4 Corners to simulate overlap detection
                    test_points = [
                        (cx, cy),
                        (cx - XFOV/2, cy - YFOV/2), (cx + XFOV/2, cy - YFOV/2),
                        (cx - XFOV/2, cy + YFOV/2), (cx + XFOV/2, cy + YFOV/2)
                    ]
                    
                    if any(poly_path.contains_points(test_points)):
                        new_fov_locations.append({
                            'sample_id': sample_id, 
                            'x': round(cx, 3), 
                            'y': round(cy, 3)
                        })

        # --- 2. Setup Visualization Canvas ---
        mos_h, mos_w = self.ui.mosaic_viewer.adj.shape
        all_pts = np.concatenate(pixel_polygons)
        min_p_x, min_p_y = np.min(all_pts, axis=0)
        max_p_x, max_p_y = np.max(all_pts, axis=0)

        canvas_x1, canvas_y1 = int(min(0, min_p_x)), int(min(0, min_p_y))
        canvas_x2, canvas_y2 = int(max(mos_w, max_p_x)), int(max(mos_h, max_p_y))
        canvas_w, canvas_h = canvas_x2 - canvas_x1, canvas_y2 - canvas_y1

        vis_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        offset_x, offset_y = -canvas_x1, -canvas_y1
        
        # Background Mosaic
        vis_img[offset_y:offset_y+mos_h, offset_x:offset_x+mos_w] = \
            cv2.cvtColor(self.ui.mosaic_viewer.adj, cv2.COLOR_GRAY2BGR)

        # Mapping constants
        xs_orig = [p['x'] for p in self.CurrentSampleLocations]
        ys_orig = [p['y'] for p in self.CurrentSampleLocations]
        v_anchor_x = min(xs_orig) - (XFOV / 2)
        v_anchor_y = min(ys_orig) - (YFOV / 2)

        # Draw Green FOVs
        for fov in new_fov_locations:
            tl_x = int((fov['x'] - XFOV/2 - v_anchor_x) / px_w_mm) + offset_x
            tl_y = int((fov['y'] - YFOV/2 - v_anchor_y) / px_h_mm) + offset_y
            br_x = int((fov['x'] + XFOV/2 - v_anchor_x) / px_w_mm) + offset_x
            br_y = int((fov['y'] + YFOV/2 - v_anchor_y) / px_h_mm) + offset_y
            cv2.rectangle(vis_img, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 1)

        # Draw Blue Polygons
        for p_poly in pixel_polygons:
            pts = (np.array(p_poly) + [offset_x, offset_y]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], True, (255, 0, 0), 2)

        # --- 3. DISPLAY ON self.ui.MosaicLabel ---
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the pixmap to fit the Label size while keeping aspect ratio
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.ui.MosaicLabel.width(), 
                                      self.ui.MosaicLabel.height(), 
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
        
        self.ui.MosaicLabel.setPixmap(scaled_pixmap)
        
        # Save locations
        self.CurrentSampleLocations = new_fov_locations
        print(f"Generated {len(new_fov_locations)} FOVs. Display updated.")
    
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