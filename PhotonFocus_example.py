# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:44:00 2025

@author: admin
"""

#
##############################################################################
# @attention
#
#<h2><center>&copy; COPYRIGHT(c) 2021 Photonfocus AG</center></h2>
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of Photonfocus nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################
#
##Auto generated code snippet##
#Add PFSDK bin folder to Python path so the bindings can be found
import os, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.environ['PF_ROOT'],'PFSDK','bin/Python'))
os.add_dll_directory(os.path.join(os.environ['PF_ROOT'],'PFSDK','bin'))
if sys.version_info >= (3,8):
    os.add_dll_directory(os.path.join(os.environ['PF_ROOT'],'DoubleRateSDK','bin'))


##End of auto generated code##

import os, sys
import colorama
import time

import PFPyCameraLib as pf
# from kbhit import KBHit

def PromptEnterAndExit(code = 0):
    input("\n\n\nPress enter to exit ...")
    colorama.deinit()
    sys.exit(code)

def ExitWithErrorPrompt(errString, pfResult = None):
    print(errString)
    if pfResult is not None:
        print(pfResult)
    PromptEnterAndExit(-1)

def EventErrorCallback(cameraNumber, errorCode, errorMessage):
    print("[Communication error callback] Camera(",cameraNumber,") Error(", errorCode, ", ", errorMessage, ")\n")

def DiscoverAndConnectCamera():
    #Discover cameras in the network or connected to the USB port
    discovery = pf.PFDiscovery()
    pfResult = discovery.DiscoverCameras()

    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Discovery error:", pfResult)

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
        ExitWithErrorPrompt("Error parsing input, not a number")

    #Check selected camera is within range
    if not 0 <= cam_id < num_discovered_cameras:
        ExitWithErrorPrompt("Selected camera out of range")

    selected_cam_info = camera_info_list[cam_id]
    #Call copy constructor
    #The camera info list elements are destroyed with PFDiscover
    if selected_cam_info.GetType() == pf.CameraType.CAMTYPE_GEV:
        copy_cam_info = pf.PFCameraInfoGEV(selected_cam_info)
    else:
        copy_cam_info = pf.PFCameraInfoU3V(selected_cam_info)
    
    #Connect camera
    pfCam = pf.PFCamera()

    pfResult = pfCam.Connect(copy_cam_info)
    #pfResult = pfCam.Connect(ip = "192.168.3.158")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not connect to the selected camera", pfResult)

    return pfCam, copy_cam_info


def ConfigureCamera(pfCam, pfCameraInfo):
    
    #This sample assumes no DR
    #But to ensure it works with all cameras, we are going to disable DR
    [pfResult, featureList] = pfCam.GetFeatureList()

        
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not get feature list from camera", pfResult)
    for elem in featureList:
        print(elem.Name)
    print('\r')
    
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("DeviceTemperature")
    print('current temperature: ', pfFeatureParam)
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("DeviceTemperatureHighLimit")
    print('temperature high limit: ', pfFeatureParam)
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("DeviceTemperatureLowLimit")
    print('temperature low limit: ', pfFeatureParam)
    
    pfResult, pfFeatureParam = pfCam.GetFeatureEnum("AcquisitionMode")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not get acquisition mode", pfResult)
    print('acquisition mode: ', pfFeatureParam)
    pfResult, pfFeatureParam = pfCam.GetFeatureInt("AcquisitionFrameCount")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not get acquisitionframe count", pfResult)
    print('acquisition frameCount: ', pfFeatureParam)
    
    pfResult = pfCam.SetFeatureEnum("AcquisitionMode", "MultiFrame")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
    pfResult = pfCam.SetFeatureInt("AcquisitionFrameCount", 10)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set acquisitionMode", pfResult)
        
    # set exposuretime
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("ExposureTime")
    print('exposure time (us): ', pfFeatureParam)
    pfResult, pfFeatureParam = pfCam.GetFeatureEnum("ExposureMode")
    print('exposure mode: ', pfFeatureParam)
    pfResult = pfCam.SetFeatureFloat("ExposureTime", 3500)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
    
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("ExposureTime")
    print('exposure time (us): ', pfFeatureParam)
    
    pfResult = pfCam.SetFeatureEnum("ExposureMode", "Timed")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
        
    # set Gain
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("Gain")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
    print('Gain: ', pfFeatureParam)
    
    pfResult, pfFeatureParam = pfCam.GetFeatureEnum("DigitalGain")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
    print('Digital Gain: ', pfFeatureParam)
    
    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("AnalogGain")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not read AnalogGain", pfResult)
    print('AnalogGain: ', pfFeatureParam)
    
    pfResult, pfFeatureParam = pfCam.GetFeatureEnum("GainSelector")
    print('GainSelector: ', pfFeatureParam)
    pfResult = pfCam.SetFeatureFloat("Gain", 1.0)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
    
    pfResult = pfCam.SetFeatureEnum("GainSelector", "DigitalAll")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)
    
    #Check DoubleRate_Enable feature is present
    if any(elem.Name == "DoubleRate_Enable" for elem in featureList):
        print("DoubleRate_Enable feature found. Disabling feature.")
        pfResult = pfCam.SetFeatureBool("DoubleRate_Enable", False)
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Failed to set DoubleRate_Enable", pfResult)

    #Set Mono8 pixel format
    pfResult = pfCam.SetFeatureEnum("PixelFormat", "Mono8")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not set PixelFormat", pfResult)

    #Set Width to maximum
    pfResult, pfFeatureParam = pfCam.GetFeatureParams("Width")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not get width feature parameters", pfResult)

    pfResult = pfCam.SetFeatureInt("Width", pfFeatureParam.Max)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Error setting width", pfResult)

    #Set Height to maximum
    pfResult, pfFeatureParam = pfCam.GetFeatureParams("Height")
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not get Height feature parameters", pfResult)

    pfResult = pfCam.SetFeatureInt("Height", pfFeatureParam.Max)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Error setting Height", pfResult)

    pfResult, pfFeatureParam = pfCam.GetFeatureFloat("AcquisitionFrameRate")
    print('frame rate is: ', pfFeatureParam)
    #Set Height to maximum
    # pfResult, pfFeatureParam = pfCam.GetFeatureParams("PowerDown")
    # if pfResult != pf.Error.NONE:
    #     ExitWithErrorPrompt("Could not get powerDown", pfResult)
    # print(pfFeatureParam.PowerDown_Status)
    
    # pfResult = pfCam.SetFeatureBool("PowerDown", False)
    # if pfResult != pf.Error.NONE:
    #     ExitWithErrorPrompt("Error setting PowerDown", pfResult)

def SetupStream(pfCam, pfCamInfo):
    #Create stream depending on camera type
    if pfCamInfo.GetType() == pf.CameraType.CAMTYPE_GEV:
        pfStream = pf.PFStreamGEV(False, True, True, True)
    else:
        pfStream = pf.PFStreamU3V()
    #Set ring buffer size to 100
    pfStream.SetBufferCount(100)

    pfResult = pfCam.AddStream(pfStream)
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Error setting stream", pfResult)

    return pfStream
    
def Grab(pfCam, pfStream):
    #Start grabbing images into stream
    print('Starting grab...')
    # kb = KBHit()

    pfResult = pfCam.Grab()
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Could not start grab process", pfResult)
        
def Acquire(pfCam, pfStream):
    pfBuffer = 0
    pfImage = pf.PFImage()

    print("\n\n\nPress any key to stop capturing")
    print("\033[1A\033[1A\033[1A", end="")
    #Loop over stream frames
    # while not kb.kbhit():
    for ii in range(5):
        #Fetch buffer
        [pfResult, pfBuffer] = pfStream.GetNextBuffer()
        if pfResult == pf.Error.NONE:
            #Get image object from buffer
            t1=time.time()
            pfBuffer.GetImage(pfImage)
            #Do something with pfImage, e.g. convert to numpy array
            t2=time.time()
            imageData = np.transpose(np.array(pfImage, copy = False))
            t3=time.time()
            # fig = plt.figure()
            # plt.imshow(imageData)
            # plt.show()
            print(imageData.shape)
            # print(imageData[0:10,0:10])
            t4=time.time()
            print('t2-t1: ', round(t2-t1,6))
            print('t3-t2: ', round(t3-t2,6))
            print('t4-t3: ', round(t4-t3,6))

        #Release frame buffer, otherwise ring buffer will get full
        pfStream.ReleaseBuffer(pfBuffer)

        #Display stream statistics
        pfStreamStats = pfStream.GetStreamStatistics()
        print(pfStreamStats, end="\r")

    #Stop frame grabbing
    pfResult = pfCam.Freeze()
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Error stopping grab process", pfResult)

def DisconnectCamera(pfCam):
    #Disconnect camera
    pfResult = pfCam.Disconnect()
    if pfResult != pf.Error.NONE:
        ExitWithErrorPrompt("Error disconnecting", pfResult)
    colorama.deinit()
    sys.exit(0)

if __name__ == "__main__":
    colorama.init()

    pf.SetEventErrorReceiver(EventErrorCallback)

    pfCam, pfCamInfo = DiscoverAndConnectCamera()
    ConfigureCamera(pfCam, pfCamInfo)
    pfStream = SetupStream(pfCam, pfCamInfo)
    Grab(pfCam, pfStream)
    Acquire(pfCam, pfStream)
    # ConfigureCamera(pfCam, pfCamInfo)
    # pfStream = SetupStream(pfCam, pfCamInfo)
    Grab(pfCam, pfStream)
    Acquire(pfCam, pfStream)
    DisconnectCamera(pfCam)
    # PromptEnterAndExit()