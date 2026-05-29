# -*- coding: utf-8 -*-
"""
PhotonFocus camera control thread.
Keeps the same shared-memory contract as the Daheng camera thread while using
the PhotonFocus SDK-specific grab path.
"""

from PyQt5.QtCore import  QThread
import time
import threading
import queue
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
PHOTONFOCUS_CONSUMER_WORKERS = 4

class Camera(QThread):
    def __init__(self):
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Camera thread exited.'
        self.SIM = False
        self._memory_write_mode = None
        self._spectral_axis_mode = None

    def run(self):
        if not (SIM or self.SIM):
            print('initializing camera...')
            self.InitBoard()
            self.GetTemp()
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get(1)
        while self.item.action != 'exit':
            try:
                if self.item.action == 'ConfigureBoard':
                    self.ConfigureBoard()
                elif self.item.action == 'Acquire':
                    if not (SIM or self.SIM):
                        self.Acquire_producer_consumer()
                    else:
                        self.simData()         
                elif self.item.action == 'UninitBoard':
                    self.UninitBoard()
                elif self.item.action == 'InitBoard':
                    self.InitBoard()
                elif self.item.action == 'GetTemp':
                    self.GetTemp()
                else:
                    message = f"Unknown camera command: {self.item.action}"
                    self.emit_status(message)
                    print(message)
            except Exception as error:
                message = "Camera command failed. This action was skipped: " + str(error)
                self.emit_status(message)
                print(message)
                print(traceback.format_exc())
            try:
                self.item = self.queue.get(1)
            except:
                self.item = DActionField('GetTemp')
        if not (SIM or self.SIM):
            self.UninitBoard()
        print(self.exit_message)
        self.emit_status(self.exit_message)

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
                print('PhotonFocus camera init success')
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
        elif self.ui.ACQMode.currentText() in ['FiniteCscan','PlateScan','PlatePreScan', 'WellScan','TimedPlateScan']:
            self.BlinesPerAcq = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
        if not (SIM or self.SIM):
            # get all camera features
            [pfResult, featureList] = self.pfCam.GetFeatureList()
            if pfResult != pf.Error.NONE:
               self. ExitWithErrorPrompt(["Could not get feature list from camera", pfResult])
            # for elem in featureList:
            #     print(elem.Name)
            # print('\r')
            
            if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline','FiniteCscan','PlateScan','PlatePreScan', 'WellScan','TimedPlateScan']:
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

        self._memory_write_mode = None
        self._spectral_axis_mode = None
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
                Bline = self.prepare_bline(Bline)
                self.write_bline_to_memory(Bline, self.MemoryLoc, BlinesCount % NBlines)
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
            
            
    
    def Acquire_producer_consumer(self):
        pfResult = self.pfCam.Grab()
        if pfResult != pf.Error.NONE:
            self.ExitWithErrorPrompt("Could not start grab process", pfResult)

        self._memory_write_mode = None
        self._spectral_axis_mode = None
        NBlines = self.Memory[0].shape[0]
        start_memory_slot = self.MemoryLoc
        mono8 = self.ui.PixelFormat_display_PF.text() in ['Mono8']
        worker_count = PHOTONFOCUS_CONSUMER_WORKERS
        grab_q = queue.Queue(maxsize=128)
        grab_stop = object()
        consumer_error = []
        profile = {
            "get_next_buffer": 0.0,
            "queue_put": 0.0,
            "queue_get": 0.0,
            "get_image": 0.0,
            "convert": 0.0,
            "prepare": 0.0,
            "memory_write": 0.0,
            "release_buffer": 0.0,
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
            pf_image = pf.PFImage()
            pf_image_unpacked = None
            if not mono8:
                pf_image_unpacked = pf.PFImage()
                if not pf_image_unpacked.IsMemAllocated():
                    reserve_result = pf_image_unpacked.ReserveImage(
                        pf.GetPixelType("Mono16"),
                        self.NSamples,
                        self.AlinesPerBline,
                    )
                    if reserve_result != pf.Error.NONE:
                        raise RuntimeError(
                            f"Worker {worker_id} failed to reserve unpack image: {reserve_result}"
                        )
            try:
                while True:
                    t_get = time.perf_counter()
                    item = grab_q.get()
                    add_profile("queue_get", time.perf_counter() - t_get)
                    if item is grab_stop:
                        break
                    pf_buffer, frame_number = item
                    block_id = frame_number // NBlines
                    memory_slot = (start_memory_slot + block_id) % self.memoryCount
                    frame_index = frame_number % NBlines
                    try:
                        t_image = time.perf_counter()
                        pf_buffer.GetImage(pf_image)
                        add_profile("get_image", time.perf_counter() - t_image)
                        if mono8:
                            bline = np.array(pf_image, copy=False)
                        else:
                            t_convert = time.perf_counter()
                            pf_result = pf_image.ConvertTo(pf_image_unpacked)
                            add_profile("convert", time.perf_counter() - t_convert)
                            if pf_result != pf.Error.NONE:
                                raise RuntimeError(f"Error unpacking image: {pf_result}")
                            bline = np.array(pf_image_unpacked, copy=False)

                        t_prepare = time.perf_counter()
                        bline = self.prepare_bline(bline)
                        add_profile("prepare", time.perf_counter() - t_prepare)

                        t_write = time.perf_counter()
                        self.write_bline_to_memory(bline, memory_slot, frame_index)
                        add_profile("memory_write", time.perf_counter() - t_write)
                        increment_profile("processed")
                        mark_frame_complete(frame_number)
                    finally:
                        t_release = time.perf_counter()
                        self.pfStream.ReleaseBuffer(pf_buffer)
                        add_profile("release_buffer", time.perf_counter() - t_release)
            except Exception:
                consumer_error.append(f"Worker {worker_id} failed:\n" + traceback.format_exc())
                print(traceback.format_exc())

        workers = [
            threading.Thread(
                target=consumer,
                args=(worker_id,),
                name=f"PFGrabConvert{worker_id}",
                daemon=True,
            )
            for worker_id in range(worker_count)
        ]
        for worker in workers:
            worker.start()

        self.DbackQueue.put(0)
        blines_count = 0
        try:
            while blines_count < self.BlinesPerAcq and self.ui.RunButton.isChecked():
                t_buffer = time.perf_counter()
                [pfResult, pfBuffer] = self.pfStream.GetNextBuffer()
                add_profile("get_next_buffer", time.perf_counter() - t_buffer)
                if pfResult != pf.Error.NONE:
                    increment_profile("timeouts")
                    continue

                t_put = time.perf_counter()
                grab_q.put((pfBuffer, blines_count))
                add_profile("queue_put", time.perf_counter() - t_put)
                with profile_lock:
                    profile["max_processing_queue"] = max(
                        profile["max_processing_queue"],
                        grab_q.qsize(),
                    )
                blines_count += 1
                increment_profile("grabbed")

                if blines_count % NBlines == 0 and self.ui.PauseButton.isChecked():
                    pfResult = self.pfCam.Freeze()
                    if pfResult != pf.Error.NONE:
                        self.ExitWithErrorPrompt("Error stopping grab process", pfResult)
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(0.5)
                    pfResult = self.pfCam.Grab()
                    if pfResult != pf.Error.NONE:
                        self.ExitWithErrorPrompt("Could not start grab process", pfResult)
        finally:
            pfResult = self.pfCam.Freeze()
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error stopping grab process", pfResult)
            for _ in workers:
                grab_q.put(grab_stop)

        for worker in workers:
            worker.join()

        self.MemoryLoc = (start_memory_slot + next_block_to_emit[0]) % self.memoryCount
        total = time.perf_counter() - total_t0
        queue_put_fraction = profile["queue_put"] / max(total, 1e-9)
        if blines_count > 1000:
            print(
                "PhotonFocus acquisition summary: \n"
                f"consumer_workers={worker_count}, \n"
                f"frames_grabbed={profile['grabbed']}, \n"
                f"frames_processed={profile['processed']}, \n"
                f"total_time={total:.3f}s, \n"
                f"camera_wait={profile['get_next_buffer']:.3f}s (longer better), \n"
                f"max_processing_queue_size={profile['max_processing_queue']}/128\n"
            )
        if profile["max_processing_queue"] > 100:
            print(
                "PhotonFocus acquire warning: processing queue was nearly full \n"
                f"({profile['max_processing_queue']}/128). \n"
                "The conversion/write thread is close to falling behind the camera stream.\n"
            )
        if profile["max_processing_queue"] > 100 and queue_put_fraction > 0.05:
            print(
                "PhotonFocus acquire warning: producer waited \n"
                f"{profile['queue_put']:.3f}s while handing frames to the conversion thread. \n"
                "The camera readout path is close to falling behind; consider reducing trigger rate \n"
                "or optimizing unpack / memory transpose-write.\n"
            )
        if consumer_error:
            raise RuntimeError("Acquire consumer failed:\n" + consumer_error[0])

    def UninitBoard(self):
        if not (SIM or self.SIM):
            #Disconnect camera
            pfResult = self.pfCam.Disconnect()
            if pfResult != pf.Error.NONE:
                self.ExitWithErrorPrompt("Error disconnecting", pfResult)
            colorama.deinit()
            sys.exit(0)
        
        
                    
    def simData(self):
        self._memory_write_mode = None
        self._spectral_axis_mode = None
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
            self.write_bline_to_memory(Bline, self.MemoryLoc, BlinesCount % NBlines)

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

    def prepare_bline(self, bline):
        bline = np.asarray(bline)
        if bline.ndim != 2:
            raise ValueError(f"PhotonFocus frame must be 2D, got shape {bline.shape}")

        expected_raw_a = (self.NSamples, self.AlinesPerBline)
        expected_raw_b = (self.AlinesPerBline, self.NSamples)

        if bline.shape == expected_raw_a:
            spectral_axis = 0
        elif bline.shape == expected_raw_b:
            spectral_axis = 1
        else:
            raise ValueError(
                "PhotonFocus raw frame shape does not match expected camera ROI: "
                f"frame={bline.shape}, expected={expected_raw_a} or {expected_raw_b}"
            )

        bline = downsample_spectral_axis(bline, self.SpectralDS, axis=spectral_axis)

        if self._spectral_axis_mode is None:
            self._spectral_axis_mode = spectral_axis
            print(
                "PhotonFocus spectral axis selected: "
                f"axis={spectral_axis} (raw frame={expected_raw_a if spectral_axis == 0 else expected_raw_b})"
            )
        return bline

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
                "PhotonFocus frame shape does not match destination memory slice: "
                f"frame={bline.shape}, frame_T={bline.T.shape}, dest={dest.shape}"
            )

        if self._memory_write_mode is None:
            self._memory_write_mode = write_mode
            print(
                "PhotonFocus memory write mode selected: "
                f"{write_mode} (frame={bline.shape}, dest={dest.shape})"
            )
                    
