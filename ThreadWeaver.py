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
from scipy.signal import hilbert
import datetime
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

                elif self.item.action == 'Mosaic':
                    # make directories
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/aip'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/aip')
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/surf'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/surf')
                    if not os.path.exists(self.ui.DIR.toPlainText()+'/fitting'):
                        os.mkdir(self.ui.DIR.toPlainText()+'/fitting')
                    # do fast pre-scan to identify tissue area
                    self.InitMemory()
                    message= self.PreMosaic()
                    if message != 'success' :
                        message = "action aborted by user..."
                    else:
                        self.InitMemory()
                        message = self.Mosaic()
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
             elif self.ui.ACQMode.currentText() in ['FiniteCscan']:
                 if self.ui.DynCheckBox.isChecked():
                     self.Memory[ii]=np.zeros([self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = self.ui.Ypixels.value()
                 else:
                     self.Memory[ii]=np.zeros([self.ui.Ypixels.value()*self.ui.BlineAVG.value(), AlinesPerBline, samples], dtype = data_type)
                     self.NAcq = 1

        ###########################################################################################
        
    def SingleScan(self, mode):
        an_action = DnSAction('Clear')
        self.DnSQueue.put(an_action)
        # t0=time.time()
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
        # while self.DbackQueue.qsize()>0:
        #     print('current dbackqueue size:', self.DbackQueue.qsize())
        #     try:
        #         self.DbackQueue.get_nowait()
        #     except:
        #         break
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
        # print('Camera config took: ',round(t1-t0,3),'sec')
        print('Galvo board config took: ',round(t2-t1,3),'sec')
        print('Camera start took: ',round(t3-t2,3),'sec')
        print('Galvo board start took: ',round(t4-t3,3),'sec')
        # print('current dbackqueue size:', self.DbackQueue.qsize())
        print('\n')

        for iAcq in range(self.NAcq):
            start = time.time()
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            # print('current dbackqueue size:', self.DbackQueue.qsize())
            an_action = self.DatabackQueue.get() # never time out
            # print('current dbackqueue size:', self.DbackQueue.qsize())
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
                        an_action = DnSAction(mode, self.data, raw=True) # data in Memory[memoryLoc]
                        self.DnSQueue.put(an_action)
                        message = mode + " successfully finished..."
                else:
                    # In other modes, do FFT first
                    an_action = GPUAction(self.ui.FFTDevice.currentText(), mode, memoryLoc)
                    self.GPUQueue.put(an_action)
                    message = mode + " successfully finished..."
            else:
                message = 'an_action is 0'
                    
                
        an_action = AODOAction('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        # print(message)
        # print('current dbackqueue size:', self.DbackQueue.qsize())
        return message
    
            
    def RptScan(self, mode):
        an_action = DnSAction('Clear')
        self.DnSQueue.put(an_action)
        an_action = DAction('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        # config AODO
        an_action = AODOAction('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        data_backs = 0 # count number of data backs
        
                
        # while self.DbackQueue.qsize()>0:
        #     try:
        #         print('current dbackqueue size:', self.DbackQueue.qsize())
        #         self.DbackQueue.get_nowait()
        #     except:
        #         break
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
            # print('waiting...')
            try: # use try-except in cases where Stop button clicked and camera stopped prior to while loop
                an_action = self.DatabackQueue.get(timeout=1) # never time out
                memoryLoc = an_action.action
                # print(memoryLoc)
                data_backs += 1
                if memoryLoc < self.ui.DisplayRatio.value():
                    ######################################### display data
                    if self.ui.FFTDevice.currentText() in ['None']:
                        # In None mode, directly do display and save
                        data = self.Memory[memoryLoc].copy()
                        an_action = DnSAction(mode, data, raw=True) # data in Memory[memoryLoc]
                        self.DnSQueue.put(an_action)
                    else:
                        # In other modes, do FFT first
                        an_action = GPUAction(self.ui.FFTDevice.currentText(), mode, memoryLoc)
                        self.GPUQueue.put(an_action)
                    ######################################## check if Pause or Stop button is clicked
            except:
                print('camera stopped')
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
        print('AODO stopped')
        # close AODO
        an_action = AODOAction('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        # digitizer will close automatically
        message = str(data_backs)+ ' data received by weaver'
        self.log.write(message)
        an_action = GPUAction('display_FFT_actions')
        self.GPUQueue.put(an_action)
        an_action = DnSAction('display_counts')
        self.DnSQueue.put(an_action)
        return mode + ' successfully finished...'
  

    
        
   
    
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
        if np.sum(value > self.ui.AgarValue.value())>value.shape[1]*value.shape[0]*0.05: 
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
        self.ui.BlineAVG.setValue(50)
        ############################# measure an Aline
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.SingleScan(self.ui.ACQMode.currentText())
        print('got Bline')
        print(self.data.shape)
        #######################################################################
        Xpixels = self.ui.AlinesPerBline.value()
        Yrpt = self.ui.BlineAVG.value()
        BLINE = self.data.reshape([Yrpt, Xpixels, self.ui.NSamples.value()])
        
        background = np.float32(np.mean(BLINE,0))
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
    
