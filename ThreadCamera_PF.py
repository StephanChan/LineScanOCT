# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:32:10 2025

@author: shuaibin
"""

from PyQt5.QtCore import  QThread
import time
import ctypes
import os, sys
import numpy as np
from matplotlib import pyplot as plt

global SIM
# SIM=True
try:
    PFSDK_PYTHON_DIR = os.path.join(os.environ['PF_ROOT'],'PFSDK','bin/Python')
    PFSDK_DLL_DIR = os.path.join(os.environ['PF_ROOT'],'PFSDK','bin')
    sys.path.append(PFSDK_PYTHON_DIR)
    os.add_dll_directory(PFSDK_DLL_DIR)
    if sys.version_info >= (3,8):
        DOUBLE_RATE_DLL_DIR = os.path.join(os.environ['PF_ROOT'],'DoubleRateSDK','bin')
        os.add_dll_directory(DOUBLE_RATE_DLL_DIR)
    import PFPyCameraLib as pf
    import colorama
    SIM = False
except Exception as error:
    pf_root = os.environ.get('PF_ROOT', '<PF_ROOT not set>')
    print(
        "PhotonFocus SDK import failed. PF_ROOT or the SDK directories may be wrong: "
        f"PF_ROOT={pf_root}, PFSDK Python={locals().get('PFSDK_PYTHON_DIR', '<unresolved>')}, "
        f"PFSDK DLL={locals().get('PFSDK_DLL_DIR', '<unresolved>')}. "
        f"Import error: {error}. Using simulation."
    )
    SIM = True

from ActionFields import DbackActionField, DActionField
from CameraUi import (
    downsample_spectral_axis,
    effective_camera_sample_count,
    raw_camera_sample_count,
    spectral_downsample,
)
import traceback

CONTINUOUS = 0x7FFFFFFF
PHOTONFOCUS_PRINT_CAMERA_CONFIG = False

class Camera(QThread):
    def __init__(self):
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Digitizer thread exited.'

    def run(self):
        if not SIM:
            self.InitBoard()
            # self.ConfigureBoard()
            self.GetTemp()
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get(1)
        # start = time.time()
        while self.item.action != 'exit':
            try:
                if self.item.action == 'ConfigureBoard':
                    self.ConfigureBoard()
                elif self.item.action == 'Acquire':
                    if not (SIM or self.SIM):
                        self.Acquire()
                    else:
                        self.simData()         
                elif self.item.action == 'UninitBoard':
                    self.UninitBoard()
                elif self.item.action == 'InitBoard':
                    self.InitBoard()
                elif self.item.action == 'GetTemp':
                    self.GetTemp()
                else:
                    self.emit_status(f"Unknown digitizer command: {self.item.action}")
            except Exception as error:
                self.emit_status("Digitizer command failed. This action was skipped.")
                print(traceback.format_exc())
            # message = 'DIGITIZER spent: '+ str(round(time.time()-start,3))+'s'
            # print(message)
            # self.log.write(message)
            try:
                self.item = self.queue.get(1)
            except:
                self.item = DActionField('GetTemp')
        if not (SIM or self.SIM):
            self.UninitBoard()
        print(self.exit_message)

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))
        
    def ExitWithErrorPrompt(self, errString, pfResult = None):
        print(errString)
        if pfResult is not None:
            print(pfResult)
        colorama.deinit()
        sys.exit(0)

    def EventErrorCallback(cameraNumber, errorCode, errorMessage):
        print("[Communication error callback] Camera(",cameraNumber,") Error(", errorCode, ", ", errorMessage, ")\n")

        
    def InitBoard(self):
        if not (SIM or self.SIM):
            #Discover cameras in the network or connected to the USB port
            discovery = pf.PFDiscovery()
            pfResult = discovery.DiscoverCameras()
    
            if pfResult != pf.Error.NONE:
                # self.ExitWithErrorPrompt("Discovery error:", pfResult)
                self.SIM = True
                # print(self.SIM)
            else:
    
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
                # if not 0 <= cam_id < num_discovered_cameras:
                #     self.ExitWithErrorPrompt("Selected camera out of range")
        
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
                    print('Camera init failed, using simulation')
                    self.SIM = True
                print('camera init success')
                # return copy_cam_info
                # self.log.write(message)
        
    def ConfigureBoard(self):
        self.AlinesPerBline = self.ui.AlinesPerBline.value()
        self.NSamples = raw_camera_sample_count(self.ui)
        self.SpectralDS = spectral_downsample(self.ui)
        self.ProcessedSamples = effective_camera_sample_count(self.ui)
        if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline']:
            self.BlinesPerAcq = self.ui.BlineAVG.value() 
        elif self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','ContinuousCscan']:
            self.BlinesPerAcq = CONTINUOUS
        elif self.ui.ACQMode.currentText() in ['FiniteCscan']:
            self.BlinesPerAcq = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
        if not (SIM or self.SIM):
            # get all camera features
            [pfResult, featureList] = self.pfCam.GetFeatureList()
            if pfResult != pf.Error.NONE:
               self. ExitWithErrorPrompt(["Could not get feature list from camera", pfResult])
            # for elem in featureList:
            #     print(elem.Name)
            # print('\r')
            
            if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline','FiniteCscan']:
                pfResult = self.pfCam.SetFeatureEnum("AcquisitionMode", "MultiFrame")
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
                pfResult = self.pfCam.SetFeatureInt("AcquisitionFrameCount", self.BlinesPerAcq)
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Could not set acquisition Frame Count", pfResult)
            elif self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','ContinuousCscan']:
                pfResult = self.pfCam.SetFeatureEnum("AcquisitionMode", "Continuous")
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
            
            pfResult = self.pfCam.SetFeatureEnum("AcquisitionStatusSelector", self.ui.AcquisitionStatusSelector_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set AcquisitionStatusSelector feature parameters", pfResult)
                
            pfResult = self.pfCam.SetFeatureEnum("TriggerSelector", self.ui.TriggerSelector_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set TriggerSelector feature parameters", pfResult)
                
            pfResult = self.pfCam.SetFeatureEnum("TriggerMode", self.ui.TriggerON_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set TriggerMode feature parameters", pfResult)
                
            pfResult = self.pfCam.SetFeatureEnum("TriggerSource", self.ui.TriggerSource_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set TriggerSource feature parameters", pfResult)
                
            pfResult = self.pfCam.SetFeatureFloat("ExposureTime", self.ui.Exposure_PF.value()*1000)
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set exposure time", pfResult)
            pfResult, pfFeatureParam =self.pfCam.GetFeatureFloat("ExposureTime")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not get ExposureTime", pfResult)
            self.ui.Exposure_display_PF.setValue(pfFeatureParam/1000)
            
            pfResult = self.pfCam.SetFeatureEnum("ExposureMode", "Timed")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set ExposureMode", pfResult)
                
            pfResult = self.pfCam.SetFeatureEnum("DigitalGain", self.ui.DGain_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set DigitalGain", pfResult)
            pfResult, pfFeatureParam =self.pfCam.GetFeatureEnum("DigitalGain")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not get Digital Gain", pfResult)
            self.ui.DGain_display_PF.setText(pfFeatureParam)
            
            #Check DoubleRate_Enable feature is present
            if any(elem.Name == "DoubleRate_Enable" for elem in featureList):
                print("DoubleRate_Enable feature found. Disabling feature.")
                pfResult = self.pfCam.SetFeatureBool("DoubleRate_Enable", False)
                if pfResult != pf.Error.NONE:
                    self.ExitWithErrorPrompt("Failed to set DoubleRate_Enable", pfResult)
    
            #Set Mono8 pixel format
            pfResult = self.pfCam.SetFeatureEnum("PixelFormat", self.ui.PixelFormat_PF.currentText())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
            pfResult, pfFeatureParam =self.pfCam.GetFeatureEnum("PixelFormat")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not get pixel format", pfResult)
            self.ui.PixelFormat_display_PF.setText(pfFeatureParam)
    
    
            pfResult = self.pfCam.SetFeatureInt("Width", self.NSamples)
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error setting width", pfResult)
            
            pfResult = self.pfCam.SetFeatureInt("OffsetX", self.ui.offsetW_PF.value())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error setting X offset", pfResult)
    
    
            pfResult = self.pfCam.SetFeatureInt("Height", self.ui.AlinesPerBline.value())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error setting Height", pfResult)
            
            pfResult = self.pfCam.SetFeatureInt("OffsetY", self.ui.offsetH.value())
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error setting Y offset", pfResult)
            
            # get frame rate
            pfResult, pfFeatureParam = self.pfCam.GetFeatureFloat("AcquisitionFrameRateMax")
            self.ui.FrameRate_PF.setValue(pfFeatureParam)
            
            self.SetupStream()
            self.pfImageUnpacked = pf.PFImage()
            [_, width] = self.pfCam.GetFeatureInt("Width")
            [_, height] = self.pfCam.GetFeatureInt("Height")
            #Allocate memory 
            if not self.pfImageUnpacked.IsMemAllocated():
               #Allocate 16 bit image
               pfResult = self.pfImageUnpacked.ReserveImage(pf.GetPixelType("Mono16"), width, height)
               if pfResult != pf.Error.NONE:
                  self.ExitWithErrorPrompt("Error allocating image: ", pfResult)
        if PHOTONFOCUS_PRINT_CAMERA_CONFIG:
            self.print_configuration_readback()
        self.DbackQueue.put(0)
        # print('config dbackqueue size:', self.DbackQueue.qsize())

    def print_configuration_readback(self):
        print("PhotonFocus camera configuration:")
        print("  Camera selection: %s" % self.ui.Camera.currentText())
        print("  ROI raw camera Width/NSamples: %d" % int(self.NSamples))
        print("  ROI raw camera Height/AlinesPerBline: %d" % int(self.AlinesPerBline))
        print("  SpectralDS: %d" % int(self.SpectralDS))
        print("  Processed samples in memory: %d" % int(self.ProcessedSamples))
        print("  OffsetX/offsetW_PF: %d" % int(self.ui.offsetW_PF.value()))
        print("  OffsetY/offsetH: %d" % int(self.ui.offsetH.value()))
        print("  PixelFormat_PF: %s" % self.ui.PixelFormat_PF.currentText())
        print("  PixelFormat_display_PF: %s" % self.ui.PixelFormat_display_PF.text())
        print("  AcquisitionStatusSelector_PF UI: %s" % self.ui.AcquisitionStatusSelector_PF.currentText())
        print("  TriggerSelector_PF UI: %s" % self.ui.TriggerSelector_PF.currentText())
        print("  TriggerMode_PF UI: %s" % self.ui.TriggerON_PF.currentText())
        print("  TriggerSource_PF UI: %s" % self.ui.TriggerSource_PF.currentText())
        print("  ACQMode: %s" % self.ui.ACQMode.currentText())
        print("  BlinesPerAcq: %s" % str(self.BlinesPerAcq))
        try:
            print("  Memory[0].shape: %s dtype=%s" % (str(self.Memory[0].shape), self.Memory[0].dtype))
        except Exception as error:
            print("  Memory[0].shape: unavailable (%s)" % error)
        if not (SIM or self.SIM):
            for name in ("Width", "Height", "OffsetX", "OffsetY"):
                try:
                    pfResult, value = self.pfCam.GetFeatureInt(name)
                    if pfResult == pf.Error.NONE:
                        print("  Camera %s readback: %s" % (name, str(value)))
                    else:
                        print("  Camera %s readback failed: %s" % (name, str(pfResult)))
                except Exception as error:
                    print("  Camera %s readback unavailable: %s" % (name, error))
            for name in (
                "AcquisitionStatusSelector",
                "TriggerSelector",
                "TriggerMode",
                "TriggerSource",
            ):
                try:
                    pfResult, value = self.pfCam.GetFeatureEnum(name)
                    if pfResult == pf.Error.NONE:
                        print("  Camera %s readback: %s" % (name, str(value)))
                    else:
                        print("  Camera %s readback failed: %s" % (name, str(pfResult)))
                except Exception as error:
                    print("  Camera %s readback unavailable: %s" % (name, error))
        
    def GetTemp(self):
        if not (SIM or self.SIM):
            pfResult, pfFeatureParam = self.pfCam.GetFeatureFloat("DeviceTemperature")
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Could not get teporature feature parameters", pfResult)
            if hasattr(self.ui, "Temporature_PF"):
                self.ui.Temporature_PF.setValue(pfFeatureParam)
        
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
        
    def Acquire(self):
        pfResult = self.pfCam.Grab()
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not start grab process", pfResult)

        
        
        pfBuffer = 0
        pfImage = pf.PFImage()
        
        #开始采集任务
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        # print('start dbackqueue size:', self.DbackQueue.qsize())
        self.DbackQueue.put(0)
        
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            t0=time.time()
            [pfResult, pfBuffer] = self.pfStream.GetNextBuffer()
            # print(pfResult)
            if pfResult == pf.Error.NONE:
                #Get image object from buffer
                t1=time.time()
                pfBuffer.GetImage(pfImage)
                t2=time.time()
                if self.ui.PixelFormat_display_PF.text() in ['Mono8']:
                    Bline = np.array(pfImage, copy = False)
                else:
                    pfResult = pfImage.ConvertTo(self.pfImageUnpacked)
                    if pfResult != pf.Error.NONE:
                        self.ExitWithErrorPrompt("Error unpacking image: ", pfResult)
                    Bline = np.array(self.pfImageUnpacked, copy = False)
                
                # print(Bline[0:10,0:5])

                t3=time.time()
                Bline = downsample_spectral_axis(Bline, self.SpectralDS, axis=1)
                self.Memory[self.MemoryLoc][BlinesCount % NBlines] = Bline
                t4=time.time()
                # fig = plt.figure()
                # plt.imshow(imageData)
                # plt.show()
                # t4=time.time()
                # print('t1-t0: ', round(t1-t0,6))
                # print('t2-t1: ', round(t2-t1,6))
                # print('t3-t2: ', round(t3-t2,6))
                # print('t4-t3: ', round(t4-t3,6))
                
                
                #Release frame buffer, otherwise ring buffer will get full
                self.pfStream.ReleaseBuffer(pfBuffer)
                BlinesCount += 1
                # print(BlinesCount)
                if BlinesCount % NBlines == 0:
                    an_action = DbackActionField(self.MemoryLoc)
                    self.DatabackQueue.put(an_action)
                    self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount
                    # print('MemoryLoc:', self.MemoryLoc)
                    
    
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
        if not (SIM or self.SIM):
            #Disconnect camera
            pfResult = self.pfCam.Disconnect()
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error disconnecting", pfResult)
            colorama.deinit()
            sys.exit(0)
        
        
                    
    def simData(self):
        
        # print('D using memory loc: ',self.MemoryLoc)
        # print(self.Memory[self.MemoryLoc].shape)
        NBlines = self.Memory[0].shape[0]
        # print(NBlines)
        #开始采集任务
        BlinesCount = 0
        self.DbackQueue.put(0)
        # print('start dbackqueue size:', self.DbackQueue.qsize())
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            # t0=time.time()
            
            if self.ui.PixelFormat_display_PF.text() in ['Mono8']:
                Bline = np.uint8(np.random.rand(self.ui.AlinesPerBline.value(), effective_camera_sample_count(self.ui))*255)
            else:
                Bline = np.uint16(np.random.rand(self.ui.AlinesPerBline.value(), effective_camera_sample_count(self.ui))*65535)
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
                    
