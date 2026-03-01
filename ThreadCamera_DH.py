# -*- coding: utf-8 -*-
"""
Main camera control thread using Amcam SDK with PyQt GUI integration.
Includes functionality for live preview, snap image, exposure control,
and mosaic image stitching.
"""

import time
from PyQt5.QtCore import QThread
import numpy as np
import traceback
from Generaic_functions import *  # 自定义函数集合，可能包含图像处理或转换方法
import matplotlib.pyplot as plt
from Actions import DbackAction, DAction
global SIM
# 尝试导入 amcam 模块，如果失败则进入仿真模式（模拟环境）
try:
    import sys
    sys.path.append("")
    import gxipy as gx 
    import initAPI
    SIM = False
except:
    print('no camera driver, using simulation')
    SIM = True

CONTINUOUS = 0x7FFFFFFF
# 主相机线程类，继承自 QThread，用于异步相机操作
class Camera(QThread):
    def __init__(self):
        #定义Camera类的初始化函数，以及一些通用变量
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Camera thread successfully exited'
        self.hcam = None       # 相机句柄
        self.hcam_fr = None    # 相机外部特征句柄

    def run(self):
        if not (SIM or self.SIM):
            print('initializing camera...')
            self.initCamera()
            self.GetExposure()
            self.GetGain()
            self.GetPixelDepth()
        self.QueueOut()

    # 异步任务处理主循环（用于执行 UI 下发的命令）
    def QueueOut(self):
        self.item = self.queue.get()  # 获取消息队列中的第一个任务
        while self.item.action != 'exit':
            try:
                if self.item.action == 'ConfigureBoard':
                    self.ConfigureBoard()
                # elif self.item.action == 'SetExposure':
                #     self.SetExposure()
                # elif self.item.action == 'GetExposure':
                #     self.GetExposure()
                # elif self.item.action == 'AutoExposure':
                #     self.AutoExposure()
                # elif self.item.action == 'SetGain':
                #     self.SetGain()
                # elif self.item.action == 'GetGain':
                #     self.GetGain()
                # elif self.item.action == 'AutoGain':
                #     self.AutoGain()
                elif self.item.action == 'Acquire':
                    if self.hcam is not None:
                        self.Stream_on()
                        self.Acquire()
                        self.Stream_off()
                    else:
                        self.simData()
                
                else:
                    message = 'Invalid camera action: ' + self.item.action
                    self.ui.statusbar.showMessage(message)
                    self.log.write(message)
            except Exception as error:
                message = "\nError occurred, skipping: " + str(error)
                self.ui.statusbar.showMessage(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()  # 获取下一个任务
        self.Close()
        print(self.exit_message)
        self.ui.statusbar.showMessage(self.exit_message)
        
    # 初始化并打开真实相机
    def initCamera(self):
        # 已修改完毕
        # 判断是否使用真实相机。如果导入 amcam 成功，camera_sim 为 None，代表使用真实硬件
        if not (SIM or self.SIM):
            device_manager = gx.DeviceManager()  # 打开设备
    
            if device_manager.update_all_device_list()[0] == 0:
                # 如果没有找到任何相机设备
                print("No camera found")
                self.hcam = None  # 清空相机句柄
            else:
                self.hcam = device_manager.open_device_by_index(1)  # 打开设备，返回相机句柄对象
                try:
                    self.hcam_fr = self.hcam.get_remote_device_feature_control() # 返回设备属性对象
                    self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                    self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                    self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
                    self.ui.PixelFormat_display_DH.setText(self.hcam_fr.get_enum_feature("PixelFormat").get())
                    # self.hcam_fr.feature_save("export_config_file.txt")
                    self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
                    
                    self.hcam_s = self.hcam.get_stream(1).get_feature_control()  # 返回流属性对象
                    self.hcam_s.get_enum_feature("StreamBufferHandlingMode").set("NewestOnly")
                except Exception as ex:
                    # 打开失败，打印错误
                    print(ex)
    
    def ConfigureBoard(self):
        self.AlinesPerBline = self.ui.AlinesPerBline.value()
        self.NSamples = self.ui.NSamples_DH.value()
        if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline']:
            self.BlinesPerAcq = self.ui.BlineAVG.value() 
        elif self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','ContinuousCscan']:
            self.BlinesPerAcq = CONTINUOUS
        elif self.ui.ACQMode.currentText() in ['FiniteCscan']:
            self.BlinesPerAcq = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
            
        if self.hcam is not None:
            self.SetExposure()
            self.SetGain()
            self.SetPixelDepth()
            self.hcam_fr.get_enum_feature("TriggerMode").set(self.ui.TriggerON.currentText())
            self.hcam_fr.get_int_feature("Width").set(self.NSamples)
            self.hcam_fr.get_int_feature("Height").set(self.AlinesPerBline)
            self.hcam_fr.get_int_feature("OffsetX").set(self.ui.offsetW_DH.value())
            self.hcam_fr.get_int_feature("OffsetY").set(self.ui.offsetH.value())
        self.DbackQueue.put(0)
            
    def Acquire(self):
        #开始采集任务
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        # print('start dbackqueue size:', self.DbackQueue.qsize())
        self.DbackQueue.put(0)
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            t0=time.time()
            buf = self.hcam.data_stream[0].get_image(timeout=10000)
            t1=time.time()
            Bline = buf.get_numpy_array()
            # self.image = np.clip(np.rint(np.mean(all_images, axis = 0)), 0, 65535).astype(np.uint16)
            # self.img_16bit = Image.fromarray(self.image.astype(np.uint16), mode='I;16')
            # [w,h] = Bline.shape
            # Bline = np.rot90(Bline,1)
                
            # print(Bline[0:10,0:5])

            t2=time.time()
            self.Memory[self.MemoryLoc][BlinesCount % NBlines] = Bline
            t3=time.time()
            # fig = plt.figure()
            # plt.imshow(imageData)
            # plt.show()
            # print('t1-t0: ', round(t1-t0,6))
            # print('t2-t1: ', round(t2-t1,6))
            # print('t3-t2: ', round(t3-t2,6))
            # print('t4-t3: ', round(t4-t3,6))
            
            
            BlinesCount += 1
            # print(BlinesCount)
            if BlinesCount % NBlines == 0:
                an_action = DbackAction(self.MemoryLoc)
                self.DatabackQueue.put(an_action)
                self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount
                # print('MemoryLoc:', self.MemoryLoc)
                
                # handle pause action
                if self.ui.PauseButton.isChecked():
                    self.hcam.stream_off() 
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(0.5)
                    self.hcam.stream_on() 
            


    def Stream_on(self):
        if self.hcam is not None:
            self.hcam.stream_on() 

    def Stream_off(self):
        if self.hcam is not None:
            self.hcam.stream_off() 
    
    def SetPixelDepth(self):
        if self.hcam is not None:
            self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
            self.ui.PixelFormat_display_DH.setText(self.hcam_fr.get_enum_feature("PixelFormat").get())
    
    def GetPixelDepth(self):
        if self.hcam is not None:
            self.ui.PixelFormat_display_DH.setText(self.hcam_fr.get_enum_feature("PixelFormat").get())

    # 设置曝光时间（从界面获取值）
    def SetExposure(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("ExposureTime").set(self.ui.Exposure_DH.value()*1000.0)
            self.ui.CurrentExpo_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)
        
    # 获取曝光时间
    def GetExposure(self):
        if self.hcam is not None:
            self.ui.CurrentExpo_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)

    # 控制自动曝光开关
    def AutoExposure(self):
        if self.hcam is not None:
            if self.ui.AutoExpo.isChecked():
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                self.ui.Exposure_DH.setValue(self.ui.CurrentExpo_DH.value())
                
    def SetGain(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("Gain").set(self.ui.Gain_DH.value()*1.0)
            self.ui.CurrentGain_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)
        
    # 获取曝光时间
    def GetGain(self):
        if self.hcam is not None:
           self.ui.CurrentGain_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)

    # 控制自动曝光开关
    def AutoGain(self):
        if self.hcam is not None:
            if self.ui.AutoGain.isChecked():
                self.hcam_fr.get_enum_feature("GainAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                self.ui.Gain_DH.setValue(self.ui.CurrentGain_DH.value())

    
    # 关闭相机并释放资源
    def Close(self):
        if self.hcam is not None:
            self.hcam.close_device()
            self.hcam = None
    
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
            
            if self.ui.PixelFormat_display_DH.text() in ['Mono8']:
                Bline = np.uint8(np.random.rand(self.ui.AlinesPerBline.value(), self.NSamples)*255)
            else:
                Bline = np.uint16(np.random.rand(self.ui.AlinesPerBline.value(), self.NSamples)*65535)
            # print('camera outputs:', Bline[0,0:20])
            # print(BlinesCount, self.BlinesPerAcq)
            self.Memory[self.MemoryLoc][BlinesCount % NBlines] = Bline

            # print(BlinesCount % NBlines)
            BlinesCount += 1
            if BlinesCount % NBlines == 0:
                an_action = DbackAction(self.MemoryLoc)
                self.DatabackQueue.put(an_action)
                self.MemoryLoc = (self.MemoryLoc+1) % self.memoryCount
                # print('MemoryLoc:', self.MemoryLoc)
                

            # handle pause action
            if self.ui.PauseButton.isChecked():
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
