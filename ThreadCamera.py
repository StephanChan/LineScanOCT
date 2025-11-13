# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:32:10 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:51:26 2024

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:45:08 2024

@author: admin
"""
from PyQt5.QtCore import  QThread
import time
import ctypes
import os, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.environ['PF_ROOT'],'PFSDK','bin/Python'))
os.add_dll_directory(os.path.join(os.environ['PF_ROOT'],'PFSDK','bin'))
if sys.version_info >= (3,8):
    os.add_dll_directory(os.path.join(os.environ['PF_ROOT'],'DoubleRateSDK','bin'))

from Actions import DbackAction, DAction
import traceback

import PFPyCameraLib as pf
import colorama

global SIM
SIM = False

CONTINUOUS = 0x7FFFFFFF

class Camera(QThread):
    def __init__(self):
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Digitizer thread successfully exited'

    def run(self):
        if not (SIM or self.SIM):
            self.InitBoard()
            # self.ConfigureBoard()
            self.GetTemp()
        self.QueueOut()
        
    def QueueOut(self):
        try:
            self.item = self.queue.get_nowait()
        except:
            self.item = DAction('GetTemp')
        # start = time.time()
        while self.item.action != 'exit':
            try:
                if self.item.action == 'ConfigureBoard':
                    if not (SIM or self.SIM):
                        self.ConfigureBoard()
                elif self.item.action == 'StartAcquire':
                    if not (SIM or self.SIM):
                        self.StartAcquire()
                    else:
                        self.simData()
                        
                elif self.item.action == 'AcquireOnce':
                    if not (SIM or self.SIM):
                        self.AcquireOnce()
                    else:
                        self.simData()
                
                elif self.item.action == 'UninitBoard':
                    if not (SIM or self.SIM):
                        self.UninitBoard()
                elif self.item.action == 'InitBoard':
                    if not (SIM or self.SIM):
                        self.InitBoard()
                elif self.item.action == 'simData':
                    self.simData()
                elif self.item.action == 'GetTemp':
                    self.GetTemp()
                    time.sleep(0.5)
                    
                else:
                    self.ui.statusbar.showMessage('Digitizer thread is doing something invalid: '+self.item.action)
                    # self.ui.PrintOut.append('Digitizer thread is doing something invalid: '+self.item.action)
            except Exception as error:
                self.ui.statusbar.showMessage("\nAn error occurred:"+" skip the Digitizer action\n")
                # self.ui.PrintOut.append("\nAn error occurred:"+" skip the Digitizer action\n")
                print(traceback.format_exc())
            # message = 'DIGITIZER spent: '+ str(round(time.time()-start,3))+'s'
            # self.ui.PrintOut.append(message)
            # print(message)
            # self.log.write(message)
            try:
                self.item = self.queue.get_nowait()
            except:
                self.item = DAction('GetTemp')
            # start = time.time()
        if not (SIM or self.SIM):
            self.UninitBoard()
        print(self.exit_message)
        
    def ExitWithErrorPrompt(errString, pfResult = None):
        print(errString)
        if pfResult is not None:
            print(pfResult)
        colorama.deinit()
        sys.exit(0)

    def EventErrorCallback(cameraNumber, errorCode, errorMessage):
        print("[Communication error callback] Camera(",cameraNumber,") Error(", errorCode, ", ", errorMessage, ")\n")

        
    def InitBoard(self):
        #Discover cameras in the network or connected to the USB port
        discovery = pf.PFDiscovery()
        pfResult = discovery.DiscoverCameras()

        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Discovery error:", pfResult)

        #Print all available cameras
        num_discovered_cameras = discovery.GetCameraCount()
        camera_info_list = []
        for x in range(num_discovered_cameras):
            [pfResult, camera_info] = discovery.GetCameraInfo(x)
            camera_info_list.append(camera_info) 
            print("[",x,"]")
            print(camera_info_list[x])

        #Prompt user to select a camera
        # user_input = input("Select camera: ")
        try:
            cam_id = 0#int(user_input)
        except:
            self.ExitWithErrorPrompt("Error parsing input, not a number")

        #Check selected camera is within range
        if not 0 <= cam_id < num_discovered_cameras:
            self.ExitWithErrorPrompt("Selected camera out of range")

        selected_cam_info = camera_info_list[cam_id]
        #Call copy constructor
        #The camera info list elements are destroyed with PFDiscover
        if selected_cam_info.GetType() == pf.CameraType.CAMTYPE_GEV:
            self.cam_info = pf.PFCameraInfoGEV(selected_cam_info)
        else:
            self.cam_info = pf.PFCameraInfoU3V(selected_cam_info)

        #Connect camera
        self.pfCam = pf.PFCamera()

        pfResult = self.pfCam.Connect(self.cam_info)
        #pfResult = pfCam.Connect(ip = "192.168.3.158")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt(["Could not connect to the selected camera", pfResult])
        print('camera init success')
        # return copy_cam_info
        # self.log.write(message)
        
    def ConfigureBoard(self):
        self.AlinesPerBline = self.ui.Width.value()
        if self.ui.ACQMode.currentText() in ['SingleBline', 'SingleAline']:
            self.BlinesPerAcq = self.ui.BlineAVG.value() 
        # if self.ui.ACQMode.currentText() in ['SingleCscan']:
        #     self.BlinesPerAcq = self.ui.BlineAVG.value() * self.ui.Height.value() 
        elif self.ui.ACQMode.currentText() in ['RptBline', 'RptAline']:
            self.BlinesPerAcq = CONTINUOUS
        # get all camera features
        [pfResult, featureList] = self.pfCam.GetFeatureList()
        if pfResult != pf.Error.NONE:
           self. ExitWithErrorPrompt(["Could not get feature list from camera", pfResult])
        # for elem in featureList:
        #     print(elem.Name)
        # print('\r')
        
        if self.ui.ACQMode.currentText() in ['SingleBline', 'SingleAline']:
            pfResult = self.pfCam.SetFeatureEnum("AcquisitionMode", "MultiFrame")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
            pfResult = self.pfCam.SetFeatureInt("AcquisitionFrameCount", self.BlinesPerAcq)
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set acquisition Frame Count", pfResult)
        elif self.ui.ACQMode.currentText() in ['RptBline', 'RptAline']:
            pfResult = self.pfCam.SetFeatureEnum("AcquisitionMode", "Continuous")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
        
        
        pfResult = self.pfCam.SetFeatureFloat("ExposureTime", self.ui.Exposure.value()*1000)
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not set exposure time", pfResult)
        pfResult, pfFeatureParam =self.pfCam.GetFeatureFloat("ExposureTime")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not get ExposureTime", pfResult)
        self.ui.Exposure_display.setValue(pfFeatureParam)
        
        pfResult = self.pfCam.SetFeatureEnum("ExposureMode", "Timed")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not set ExposureMode", pfResult)
            
        pfResult = self.pfCam.SetFeatureEnum("DigitalGain", self.ui.DGain.currentText())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not set DigitalGain", pfResult)
        pfResult, pfFeatureParam =self.pfCam.GetFeatureEnum("DigitalGain")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not get Digital Gain", pfResult)
        self.ui.DGain_display.setText(pfFeatureParam)
        
        #Check DoubleRate_Enable feature is present
        if any(elem.Name == "DoubleRate_Enable" for elem in featureList):
            print("DoubleRate_Enable feature found. Disabling feature.")
            pfResult = self.pfCam.SetFeatureBool("DoubleRate_Enable", False)
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Failed to set DoubleRate_Enable", pfResult)

        #Set Mono8 pixel format
        pfResult = self.pfCam.SetFeatureEnum("PixelFormat", self.ui.PixelFormat.currentText())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
        pfResult, pfFeatureParam =self.pfCam.GetFeatureEnum("PixelFormat")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not get pixel format", pfResult)
        self.ui.PixelFormat_display.setText(pfFeatureParam)


        pfResult = self.pfCam.SetFeatureInt("Width", self.ui.Width.value())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error setting width", pfResult)
        
        pfResult = self.pfCam.SetFeatureInt("OffsetX", self.ui.offsetW.value())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error setting X offset", pfResult)

        #Set Height to maximum
        # pfResult, pfFeatureParam = self.pfCam.GetFeatureParams("Height")
        # if pfResult != pf.Error.NONE:
        #     self.ExitWithErrorPrompt("Could not get Height feature parameters", pfResult)

        pfResult = self.pfCam.SetFeatureInt("Height", self.ui.Height.value())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error setting Height", pfResult)
        
        pfResult = self.pfCam.SetFeatureInt("OffsetY", self.ui.offsetH.value())
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error setting Y offset", pfResult)
            
        self.SetupStream()
        
    def GetTemp(self):
        pfResult, pfFeatureParam = self.pfCam.GetFeatureFloat("DeviceTemperature")
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not get Height feature parameters", pfResult)
        self.ui.Temporature.setValue(pfFeatureParam)
        
    def SetupStream(self):
        #Create stream depending on camera type
        if self.cam_info.GetType() == pf.CameraType.CAMTYPE_GEV:
            self.pfStream = pf.PFStreamGEV(False, True, True, True)
        else:
            self.pfStream = pf.PFStreamU3V()
        #Set ring buffer size to 100
        self.pfStream.SetBufferCount(100)
    
        pfResult = self.pfCam.AddStream(self.pfStream)
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error setting stream", pfResult)
    
        # return pfStream
        
    def StartAcquire(self):
        pfResult = self.pfCam.Grab()
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not start grab process", pfResult)

        [_, width] = self.pfCam.GetFeatureInt("Width")
        [_, height] = self.pfCam.GetFeatureInt("Height")
        
        pfBuffer = 0
        pfImage = pf.PFImage()
        pfImageUnpacked = pf.PFImage()

        #Allocate memory 
        if not pfImageUnpacked.IsMemAllocated():
           # print('allocate')
           #Allocate 8 bit image
           pfResult = pfImageUnpacked.ReserveImage(pf.GetPixelType("Mono16"), width, height)
           if pfResult != pf.Error.NONE:
              self.ExitWithErrorPrompt("Error allocating image: ", pfResult)
        #开始采集任务
        BlinesCount = 0
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            # t0=time.time()
            [pfResult, pfBuffer] = self.pfStream.GetNextBuffer()
            if pfResult == pf.Error.NONE:
                #Get image object from buffer
                # t1=time.time()
                pfBuffer.GetImage(pfImage)
                # t2=time.time()
                if self.ui.PixelFormat_display.text() in ['Mono8']:
                    Bline = np.array(pfImage, copy = False)
                else:
                    pfResult = pfImage.ConvertTo(pfImageUnpacked)
                    if pfResult != pf.Error.NONE:
                        self.ExitWithErrorPrompt("Error unpacking image: ", pfResult)
                    Bline = np.array(pfImageUnpacked, copy = False)
                
                # print(Bline[0:10,0:5])

                t3=time.time()
                self.Memory[self.MemoryLoc][BlinesCount % self.ui.BlineAVG.value()] = Bline
                # t3=time.time()
                # fig = plt.figure()
                # plt.imshow(imageData)
                # plt.show()
                # t4=time.time()
                # print('t1-t0: ', round(t1-t0,6))
                # print('t2-t1: ', round(t2-t1,6))
                # print('t3-t2: ', round(t3-t2,6))
                BlinesCount += 1
                
            #Release frame buffer, otherwise ring buffer will get full
            self.pfStream.ReleaseBuffer(pfBuffer)
            
            if BlinesCount % self.ui.BlineAVG.value() == 0:
                an_action = DbackAction(self.MemoryLoc)
                self.DbackQueue.put(an_action)
                self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount
                # print('MemoryLoc:', self.MemoryLoc)
                # 
                
                
            # handle pause action
            if self.ui.PauseButton.isChecked():
                pfResult = self.pfCam.Freeze()
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Error stopping grab process", pfResult)
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                pfResult = self.pfCam.Grab()
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Could not start grab process", pfResult)
            
        #Stop frame grabbing
        pfResult = self.pfCam.Freeze()
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error stopping grab process", pfResult)
            
            
    
    def UninitBoard(self):
        #Disconnect camera
        pfResult = self.pfCam.Disconnect()
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Error disconnecting", pfResult)
        colorama.deinit()
        sys.exit(0)
        
    
    def simData(self):
        self.AlinesPerBline = self.ui.AlineAVG.value()*self.ui.Width.value()
        if self.ui.ACQMode.currentText() in ['SingleBline', 'SingleAline','RptBline', 'RptAline']:
            self.Height = self.ui.BlineAVG.value()
        elif self.ui.ACQMode.currentText() in ['SingleCscan', 'Mosaic','Mosaic+Cut']:
            self.Height = self.ui.BlineAVG.value() * self.ui.Height.value()
        # all samples in all Alines, include galvo fly-backs
        readLength = self.ui.PostSamples_2.value()  * 1 * self.AlinesPerBline # 数据读取长度
        
        # print('D using memory loc: ',self.MemoryLoc)
        # print(self.Memory[self.MemoryLoc].shape)

        self.Memory[self.MemoryLoc] = np.random.random([self.Height, readLength])
        time.sleep(self.Height/100)
        an_action = DbackAction(self.MemoryLoc)
        self.DbackQueue.put(an_action)
        self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount

        # print('finish acquiring')
        # self.ui.PrintOut.append('finish acquiring')
        # self.log.write('finish acquiring')
        
