# -*- coding: utf-8 -*-
"""
Main camera control thread using Amcam SDK with PyQt GUI integration.
Includes functionality for live preview, snap image, exposure control,
and mosaic image stitching.
"""

import time
import threading
import queue
from PyQt5.QtCore import QThread
import numpy as np
import traceback
from Generaic_functions import *  # 自定义函数集合，可能包含图像处理或转换方法
import matplotlib.pyplot as plt
from Actions import DbackAction, DAction
import matplotlib.pyplot as plt
global SIM
# 尝试导入 amcam 模块，如果失败则进入仿真模式（模拟环境）
try:
    import sys
    sys.path.append(r"D:\\GalaxySDK\\Development\\Samples\\Python\\")
    import gxipy as gx 
    import DahengCamera_init
    from ctypes import *
    from gxipy.gxidef import *
    from gxipy.ImageFormatConvert import *
    SIM = False
except:
    print('no camera driver, using simulation')
    SIM = True

CONTINUOUS = 0x7FFFFFFF

def get_best_valid_bits(pixel_format):
    valid_bits = DxValidBit.BIT0_7
    if pixel_format in (GxPixelFormatEntry.MONO8,
                        GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8,
                        GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8,
                        GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8,
                        GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        valid_bits = DxValidBit.BIT0_7
    elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.MONO10_P,
                          GxPixelFormatEntry.BAYER_GR10, GxPixelFormatEntry.BAYER_RG10,
                          GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10,
                          GxPixelFormatEntry.BAYER_GR10_P, GxPixelFormatEntry.BAYER_RG10_P,
                          GxPixelFormatEntry.BAYER_GB10_P, GxPixelFormatEntry.BAYER_BG10_P,
                          GxPixelFormatEntry.BAYER_GR10_PACKED, GxPixelFormatEntry.BAYER_RG10_PACKED,
                          GxPixelFormatEntry.BAYER_GB10_PACKED, GxPixelFormatEntry.BAYER_BG10_PACKED):
        valid_bits = DxValidBit.BIT2_9
    elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.MONO12_P,
                          GxPixelFormatEntry.BAYER_GR12, GxPixelFormatEntry.BAYER_RG12,
                          GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12,
                          GxPixelFormatEntry.BAYER_GR12_P, GxPixelFormatEntry.BAYER_RG12_P,
                          GxPixelFormatEntry.BAYER_GB12_P, GxPixelFormatEntry.BAYER_BG12_P,
                          GxPixelFormatEntry.BAYER_GR12_PACKED, GxPixelFormatEntry.BAYER_RG12_PACKED,
                          GxPixelFormatEntry.BAYER_GB12_PACKED, GxPixelFormatEntry.BAYER_BG12_PACKED):
        valid_bits = DxValidBit.BIT4_11
    elif pixel_format in (GxPixelFormatEntry.MONO14, GxPixelFormatEntry.MONO14_P,
                          GxPixelFormatEntry.BAYER_GR14, GxPixelFormatEntry.BAYER_RG14,
                          GxPixelFormatEntry.BAYER_GB14, GxPixelFormatEntry.BAYER_BG14,
                          GxPixelFormatEntry.BAYER_GR14_P, GxPixelFormatEntry.BAYER_RG14_P,
                          GxPixelFormatEntry.BAYER_GB14_P, GxPixelFormatEntry.BAYER_BG14_P,
                          ):
        valid_bits = DxValidBit.BIT6_13
    elif pixel_format in (GxPixelFormatEntry.MONO16,
                          GxPixelFormatEntry.BAYER_GR16, GxPixelFormatEntry.BAYER_RG16,
                          GxPixelFormatEntry.BAYER_GB16, GxPixelFormatEntry.BAYER_BG16):
        valid_bits = DxValidBit.BIT8_15
    return valid_bits


class PackedPixelFormatConverter(object):
    """
    Reuses ImageFormatConvert destination/valid-bits settings and a single output buffer
    while width/height/source pixel format (and dest) stay the same — avoids per-frame alloc
    and redundant SDK configuration.
    """
    def __init__(self):
        self._cache_key = None
        self._dest_pf = None
        self._valid_bits = None
        self._out_buf = None
        self._buf_size = 0

    @staticmethod
    def _geometry_key(raw_image):
        fd = raw_image.frame_data
        return (fd.width, fd.height, fd.pixel_format)

    def convert(self, raw_image, dest_pixel_format):
        global image_convert
        key = self._geometry_key(raw_image)
        src_pf = raw_image.get_pixel_format()
        valid_bits = get_best_valid_bits(src_pf)
        if (
            key != self._cache_key
            or dest_pixel_format != self._dest_pf
            or valid_bits != self._valid_bits
        ):
            image_convert.set_dest_format(dest_pixel_format)
            image_convert.set_valid_bits(valid_bits)
            buf_size = image_convert.get_buffer_size_for_conversion(raw_image)
            if self._out_buf is None or buf_size != self._buf_size:
                self._out_buf = (c_ubyte * buf_size)()
                self._buf_size = buf_size
            self._cache_key = key
            self._dest_pf = dest_pixel_format
            self._valid_bits = valid_bits

        image_convert.convert(raw_image, addressof(self._out_buf), self._buf_size, False)
        return self._out_buf, self._buf_size


# 主相机线程类，继承自 QThread，用于异步相机操作
class Camera(QThread):
    def __init__(self):
        #定义Camera类的初始化函数，以及一些通用变量
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Camera thread successfully exited'
        self.hcam = None       # 相机句柄
        self.hcam_fr = None    # 相机外部特征句柄
        self._packed_converter = None

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
            global image_convert
            image_convert = device_manager.create_image_format_convert()
            self._packed_converter = PackedPixelFormatConverter()
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
                    # self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
                    # pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
                    # self.ui.PixelFormat_display_DH.setText(pixelformat[1])
                    # self.hcam_fr.feature_save("export_config_file.txt")
                    # self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
                    
                    self.hcam_s = self.hcam.get_stream(1).get_feature_control()  # 返回流属性对象
                    self.hcam_s.get_enum_feature("StreamBufferHandlingMode").set("NewestOnly")
                    self.hcam.data_stream[0].set_acquisition_buffer_number(1000)
                except Exception as ex:
                    # 打开失败，打印错误
                    print(ex)
    
    def ConfigureBoard(self):
        self.AlinesPerBline = self.ui.AlinesPerBline.value()
        self.NSamples_DH = self.ui.NSamples_DH.value()
        if self.ui.ACQMode.currentText() in ['FiniteBline', 'FiniteAline']:
            self.BlinesPerAcq = self.ui.BlineAVG.value() 
        elif self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','ContinuousCscan']:
            self.BlinesPerAcq = CONTINUOUS
        elif self.ui.ACQMode.currentText() in ['FiniteCscan','PlateScan','PlatePreScan', 'WellScan']:
            self.BlinesPerAcq = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
            
        if self.hcam is not None:
            self.SetExposure()
            self.SetGain()
            self.SetPixelDepth()
            self.hcam_fr.get_enum_feature("TriggerMode").set(self.ui.TriggerON_DH.currentText())
            self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
            self.hcam_fr.get_enum_feature("TriggerActivation").set(self.ui.TriggerActivation_DH.currentText())
            # self.hcam_fr.get_enum_feature("TriggerDelay").set(int(self.ui.TriggerDelay_DH.value()*1000.0))
            
            self.hcam_fr.get_int_feature("Height").set(self.NSamples_DH )
            self.hcam_fr.get_int_feature("Width").set(self.AlinesPerBline )
            self.hcam_fr.get_int_feature("OffsetY").set(self.ui.offsetW_DH.value())
            self.hcam_fr.get_int_feature("OffsetX").set(self.ui.offsetH.value())
        self.DbackQueue.put(0)
            
    def Acquire(self):
        # 开始采集任务：producer 使用 dq_buf；consumer 转换并写入 Memory，处理后必须 q_buf 归还缓冲
        NBlines = self.Memory[0].shape[0]
        grab_q = queue.Queue(maxsize=4)
        grab_stop = object()
        use_packed = self.ui.PixelFormat_DH.currentText() in ["Mono12Packed"]
        consumer_error = []
        ds = self.hcam.data_stream[0]

        def consumer():
            try:
                BlinesCount = 0
                while True:
                    item = grab_q.get()
                    if item is grab_stop:
                        break
                    buf, grab_ms = item
                    conv_ms = 0.0
                    try:
                        if buf is None:
                            print("camera time out...")
                            Bline = np.zeros([self.NSamples_DH, self.AlinesPerBline])
                        else:
                            # t_conv = time.perf_counter()
                            if use_packed:
                                mono_image_array, _ = self._packed_converter.convert(
                                    buf, GxPixelFormatEntry.MONO12)
                                Bline = np.frombuffer(
                                    mono_image_array, dtype=np.uint16).reshape(
                                   self.NSamples_DH, self.AlinesPerBline)
                                # print(Bline.shape)
                            else:
                                Bline = buf.get_numpy_array()
                            # conv_ms = (time.perf_counter() - t_conv) * 1000.0

                        self.Memory[self.MemoryLoc][BlinesCount % NBlines] = np.transpose(Bline)
                        BlinesCount += 1
                        # print("grab_ms={:.3f}  conv_ms={:.3f}".format(grab_ms, conv_ms))
                        if BlinesCount % NBlines == 0:
                            an_action = DbackAction(self.MemoryLoc)
                            self.DatabackQueue.put(an_action)
                            self.MemoryLoc = (self.MemoryLoc + 1) % self.memoryCount
                            if self.ui.PauseButton.isChecked():
                                self.hcam.stream_off()
                                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                                    time.sleep(0.5)
                                self.hcam.stream_on()
                    finally:
                        if buf is not None:
                            ds.q_buf(buf)
            except Exception:
                consumer_error.append(traceback.format_exc())
                print(traceback.format_exc())

        worker = threading.Thread(target=consumer, name="DHGrabConvert", daemon=True)
        worker.start()
        try:
            self.DbackQueue.put(0)
            BlinesCount = 0
            while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
                # t_grab = time.perf_counter()
                buf = ds.dq_buf(timeout=200)
                # grab_ms = (time.perf_counter() - t_grab) * 1000.0
                grab_q.put((buf, 0))#grab_ms))
                BlinesCount += 1
                # print(BlinesCount)
        finally:
            grab_q.put(grab_stop)
        worker.join()
        if consumer_error:
            raise RuntimeError("Acquire consumer failed:\n" + consumer_error[0])

    def Stream_on(self):
        if self.hcam is not None:
            self.hcam.stream_on() 

    def Stream_off(self):
        if self.hcam is not None:
            self.hcam.stream_off() 
    
    def SetPixelDepth(self):
        if self.hcam is not None:
            self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
            pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
            self.ui.PixelFormat_display_DH.setText(pixelformat[1])
    
    def GetPixelDepth(self):
        if self.hcam is not None:
            pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
            self.ui.PixelFormat_display_DH.setText(pixelformat[1])

    # 设置曝光时间（从界面获取值）
    def SetExposure(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("ExposureTime").set(self.ui.Exposure_DH.value()*1000.0)
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)
        
    # 获取曝光时间
    def GetExposure(self):
        if self.hcam is not None:
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)

    # 控制自动曝光开关
    def AutoExposure(self):
        if self.hcam is not None:
            if self.ui.AutoExpo.isChecked():
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                self.ui.Exposure_DH.setValue(self.ui.Exposure_display_DH.value())
                
    def SetGain(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("Gain").set(self.ui.DGain_DH.value()*1.0)
            self.ui.DGain_display_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)
        
    # 获取曝光时间
    def GetGain(self):
        if self.hcam is not None:
           self.ui.DGain_display_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)

    # 控制自动曝光开关
    def AutoGain(self):
        if self.hcam is not None:
            if self.ui.AutoGain.isChecked():
                self.hcam_fr.get_enum_feature("GainAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                self.ui.DGain_DH.setValue(self.ui.DGain_display_DH.value())

    
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
                Bline = np.uint8(np.random.rand(self.ui.AlinesPerBline.value(), self.NSamples_DH)*np.random.randint(255))
            else:
                Bline = np.uint16(np.random.rand(self.ui.AlinesPerBline.value(), self.NSamples_DH)*np.random.randint(4096))
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
