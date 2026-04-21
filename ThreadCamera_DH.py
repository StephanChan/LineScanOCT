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
from Generaic_functions import *  # 閼奉亜鐣炬稊澶婂毐閺佷即娉﹂崥鍫礉閸欘垵鍏橀崠鍛儓閸ユ儳鍎氭径鍕倞閹存牞娴嗛幑銏℃煙濞?
import matplotlib.pyplot as plt
from Actions import DbackAction, DAction
import matplotlib.pyplot as plt
global SIM
# 鐏忔繆鐦€电厧鍙?amcam 濡€虫健閿涘苯顩ч弸婊冦亼鐠愩儱鍨潻娑樺弳娴犺法婀″Ο鈥崇础閿涘牊膩閹风喓骞嗘晶鍐跨礆
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
DAHENG_MEMORY_WRITE_METHOD = "assign_transpose"
# Packed conversion and transpose-write are the long pole for long dynamic scans.
# Increase this cautiously: block completion is still emitted in frame order below.
DAHENG_CONSUMER_WORKERS = 4

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
    while width/height/source pixel format (and dest) stay the same 閳?avoids per-frame alloc
    and redundant SDK configuration.
    """
    def __init__(self, image_convert_obj):
        self._image_convert = image_convert_obj
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
        key = self._geometry_key(raw_image)
        src_pf = raw_image.get_pixel_format()
        valid_bits = get_best_valid_bits(src_pf)
        if (
            key != self._cache_key
            or dest_pixel_format != self._dest_pf
            or valid_bits != self._valid_bits
        ):
            self._image_convert.set_dest_format(dest_pixel_format)
            self._image_convert.set_valid_bits(valid_bits)
            buf_size = self._image_convert.get_buffer_size_for_conversion(raw_image)
            if self._out_buf is None or buf_size != self._buf_size:
                self._out_buf = (c_ubyte * buf_size)()
                self._buf_size = buf_size
            self._cache_key = key
            self._dest_pf = dest_pixel_format
            self._valid_bits = valid_bits

        self._image_convert.convert(raw_image, addressof(self._out_buf), self._buf_size, False)
        return self._out_buf, self._buf_size


# 娑撹崵娴夐張铏瑰殠缁嬪琚敍宀€鎴烽幍鑳殰 QThread閿涘瞼鏁ゆ禍搴＄磽濮濄儳娴夐張鐑樻惙娴?
class Camera(QThread):
    def __init__(self):
        #鐎规矮绠烠amera缁崵娈戦崚婵嗩潗閸栨牕鍤遍弫甯礉娴犮儱寮锋稉鈧禍娑⑩偓姘辨暏閸欐﹢鍣?
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = 'Camera thread exited.'
        self.hcam = None       # 閻╁憡婧€閸欍儲鐒?
        self.hcam_fr = None    # 閻╁憡婧€婢舵牠鍎撮悧鐟扮窙閸欍儲鐒?
        self.device_manager = None
        self.memory_write_method = DAHENG_MEMORY_WRITE_METHOD

    def run(self):
        if not (SIM or self.SIM):
            print('initializing camera...')
            self.initCamera()
            self.GetExposure()
            self.GetGain()
            self.GetPixelDepth()
        self.QueueOut()

    # 瀵倹顒炴禒璇插婢跺嫮鎮婃稉璇叉儕閻滎垽绱欓悽銊ょ艾閹笛嗩攽 UI 娑撳褰傞惃鍕嚒娴犮倧绱?
    def QueueOut(self):
        self.item = self.queue.get()  # 閼惧嘲褰囧☉鍫熶紖闂冪喎鍨稉顓犳畱缁楊兛绔存稉顏冩崲閸?
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
                    message = f"Unknown camera command: {self.item.action}"
                    self.emit_status(message)
                    self.log.write(message)
            except Exception as error:
                message = "Camera command failed. This action was skipped: " + str(error)
                self.emit_status(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()  # 閼惧嘲褰囨稉瀣╃娑擃亙鎹㈤崝?
        self.Close()
        print(self.exit_message)
        self.emit_status(self.exit_message)

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))
        
    # 閸掓繂顫愰崠鏍ц嫙閹垫挸绱戦惇鐔风杽閻╁憡婧€
    def initCamera(self):
        # 瀹歌弓鎱ㄩ弨鐟扮暚濮?
        # 閸掋倖鏌囬弰顖氭儊娴ｈ法鏁ら惇鐔风杽閻╁憡婧€閵嗗倸顩ч弸婊冾嚤閸?amcam 閹存劕濮涢敍瀹慳mera_sim 娑?None閿涘奔鍞悰銊ゅ▏閻劎婀＄€圭偟鈥栨禒?
        if not (SIM or self.SIM):
            self.device_manager = gx.DeviceManager()  # Open device manager
            if self.device_manager.update_all_device_list()[0] == 0:
                print("No camera found")
                self.hcam = None
            else:
                self.hcam = self.device_manager.open_device_by_index(1)
                try:
                    self.hcam_fr = self.hcam.get_remote_device_feature_control() # 鏉╂柨娲栫拋鎯ь槵鐏炵偞鈧冾嚠鐠?
                    self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                    self.hcam_fr.get_enum_feature("ExposureAuto").set("Off")
                    # self.hcam_fr.get_enum_feature("PixelFormat").set(self.ui.PixelFormat_DH.currentText())
                    # pixelformat = self.hcam_fr.get_enum_feature("PixelFormat").get()
                    # self.ui.PixelFormat_display_DH.setText(pixelformat[1])
                    # self.hcam_fr.feature_save("export_config_file.txt")
                    # self.hcam_fr.get_enum_feature("TriggerSource").set(self.ui.TriggerSource_DH.currentText())
                    
                    self.hcam_s = self.hcam.get_stream(1).get_feature_control()  # 鏉╂柨娲栧ù浣哥潣閹冾嚠鐠?
                    self.hcam_s.get_enum_feature("StreamBufferHandlingMode").set("OldestFirst")
                    self.hcam.data_stream[0].set_acquisition_buffer_number(1000)
                except Exception as ex:
                    # 閹垫挸绱戞径杈Е閿涘本澧﹂崡浼存晩鐠?
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
        # self.DbackQueue.put(0)
            
    def Acquire(self):
        NBlines = self.Memory[0].shape[0]
        grab_q = queue.Queue(maxsize=128)
        grab_stop = object()
        use_packed = self.ui.PixelFormat_DH.currentText() in ["Mono12Packed"]
        consumer_error = []
        ds = self.hcam.data_stream[0]
        start_memory_slot = self.MemoryLoc
        worker_count = DAHENG_CONSUMER_WORKERS if use_packed else 1
        profile = {
            "dq_buf": 0.0,
            "queue_put": 0.0,
            "queue_get": 0.0,
            "packed_convert": 0.0,
            "numpy_view": 0.0,
            "memory_write": 0.0,
            "q_buf": 0.0,
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
                        self.DatabackQueue.put(DbackAction(memory_slot))
                        with profile_lock:
                            profile["max_databack_queue"] = max(
                                profile["max_databack_queue"],
                                self.DatabackQueue.qsize(),
                            )
                        completed_block_ids.remove(emit_block_id)
                        next_block_to_emit[0] += 1

        def consumer(worker_id):
            converter = None
            if use_packed:
                converter = PackedPixelFormatConverter(
                    self.device_manager.create_image_format_convert()
                )
            try:
                while True:
                    t_get = time.perf_counter()
                    item = grab_q.get()
                    add_profile("queue_get", time.perf_counter() - t_get)
                    if item is grab_stop:
                        break
                    buf, frame_number = item
                    block_id = frame_number // NBlines
                    memory_slot = (start_memory_slot + block_id) % self.memoryCount
                    frame_index = frame_number % NBlines
                    try:
                        if buf is None:
                            print("camera time out...")
                            increment_profile("timeouts")
                            Bline = np.zeros(
                                [self.NSamples_DH, self.AlinesPerBline],
                                dtype=self.Memory[memory_slot].dtype,
                            )
                        else:
                            if use_packed:
                                t_convert = time.perf_counter()
                                mono_image_array, _ = converter.convert(
                                    buf, GxPixelFormatEntry.MONO12)
                                add_profile("packed_convert", time.perf_counter() - t_convert)
                                t_view = time.perf_counter()
                                Bline = np.frombuffer(
                                    mono_image_array, dtype=np.uint16).reshape(
                                   self.NSamples_DH, self.AlinesPerBline)
                                add_profile("numpy_view", time.perf_counter() - t_view)
                            else:
                                t_view = time.perf_counter()
                                Bline = buf.get_numpy_array()
                                add_profile("numpy_view", time.perf_counter() - t_view)

                        t_write = time.perf_counter()
                        self.write_bline_to_memory(Bline, memory_slot, frame_index)
                        add_profile("memory_write", time.perf_counter() - t_write)
                        increment_profile("processed")
                        mark_frame_complete(frame_number)
                    finally:
                        if buf is not None:
                            t_q = time.perf_counter()
                            ds.q_buf(buf)
                            add_profile("q_buf", time.perf_counter() - t_q)
            except Exception:
                consumer_error.append(f"Worker {worker_id} failed:\n" + traceback.format_exc())
                print(traceback.format_exc())

        workers = [
            threading.Thread(target=consumer, args=(worker_id,), name=f"DHGrabConvert{worker_id}", daemon=True)
            for worker_id in range(worker_count)
        ]
        for worker in workers:
            worker.start()
        try:
            self.DbackQueue.put(0)
            BlinesCount = 0
            while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
                t_dq = time.perf_counter()
                buf = ds.dq_buf(timeout=200)
                add_profile("dq_buf", time.perf_counter() - t_dq)
                t_put = time.perf_counter()
                grab_q.put((buf, BlinesCount))
                add_profile("queue_put", time.perf_counter() - t_put)
                queue_depth = grab_q.qsize()
                with profile_lock:
                    profile["max_processing_queue"] = max(profile["max_processing_queue"], queue_depth)
                BlinesCount += 1
                increment_profile("grabbed")
                if self.ui.PauseButton.isChecked():
                    self.hcam.stream_off()
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(0.5)
                    self.hcam.stream_on()
        finally:
            for _ in workers:
                grab_q.put(grab_stop)
        for worker in workers:
            worker.join()
        self.MemoryLoc = (start_memory_slot + next_block_to_emit[0]) % self.memoryCount
        total = time.perf_counter() - total_t0
        queue_put_fraction = profile["queue_put"] / max(total, 1e-9)
        if BlinesCount > 1000:
            print(
                "Daheng acquisition summary: \n"
                f"consumer_workers={worker_count}, \n"
                f"frames_grabbed={profile['grabbed']}, \n"
                f"frames_processed={profile['processed']}, \n"
                f"total_time={total:.3f}s, \n"
                f"camera_wait={profile['dq_buf']:.3f}s (longer better), \n"
                f"max_processing_queue_size={profile['max_processing_queue']}/128\n"
            )
        if profile["max_processing_queue"] > 100:
            print(
                "Daheng acquire warning: processing queue was nearly full \n"
                f"({profile['max_processing_queue']}/128). \n"
                "The conversion/write thread is close to falling behind the camera stream.\n"
            )
        if profile["max_processing_queue"] > 100 and queue_put_fraction > 0.05:
            print(
                "Daheng acquire warning: producer waited \n"
                f"{profile['queue_put']:.3f}s while handing frames to the conversion thread. \n"
                "The camera readout path is close to falling behind; consider reducing trigger rate \n"
                "or optimizing packed conversion / memory transpose-write.\n"
            )
        if consumer_error:
            raise RuntimeError("Acquire consumer failed:\n" + consumer_error[0])

    def write_bline_to_memory(self, bline, memory_slot, frame_index):
        dest = self.Memory[memory_slot][frame_index]
        if self.memory_write_method == "copyto_transpose":
            np.copyto(dest, bline.T)
        elif self.memory_write_method == "swapaxes_copyto":
            np.copyto(dest, np.swapaxes(bline, 0, 1))
        elif self.memory_write_method == "assign_transpose":
            dest[...] = bline.T
        else:
            raise ValueError(f"Unknown Daheng memory write method: {self.memory_write_method}")

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

    # 鐠佸墽鐤嗛弴婵嗗帨閺冨爼妫块敍鍫滅矤閻ｅ矂娼伴懢宄板絿閸婄》绱?
    def SetExposure(self):
        if self.hcam is not None:
            self.hcam_fr.get_float_feature("ExposureTime").set(self.ui.Exposure_DH.value()*1000.0)
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)
        
    # 閼惧嘲褰囬弴婵嗗帨閺冨爼妫?
    def GetExposure(self):
        if self.hcam is not None:
            self.ui.Exposure_display_DH.setValue(self.hcam_fr.get_float_feature("ExposureTime").get()/1000.0)

    # 閹貉冨煑閼奉亜濮╅弴婵嗗帨瀵偓閸?
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
        
    # 閼惧嘲褰囬弴婵嗗帨閺冨爼妫?
    def GetGain(self):
        if self.hcam is not None:
           self.ui.DGain_display_DH.setValue(self.hcam_fr.get_float_feature("Gain").get()/1.0)

    # 閹貉冨煑閼奉亜濮╅弴婵嗗帨瀵偓閸?
    def AutoGain(self):
        if self.hcam is not None:
            if self.ui.AutoGain.isChecked():
                self.hcam_fr.get_enum_feature("GainAuto").set("Continuous")
            else:
                self.hcam_fr.get_enum_feature("GainAuto").set("Off")
                self.ui.DGain_DH.setValue(self.ui.DGain_display_DH.value())

    
    # 閸忔娊妫撮惄鍛婃簚楠炲爼鍣撮弨鎹愮カ濠?
    def Close(self):
        if self.hcam is not None:
            self.hcam.close_device()
            self.hcam = None
    
    def simData(self):
        
        # print('D using memory loc: ',self.MemoryLoc)
        # print(self.Memory[self.MemoryLoc].shape)
        NBlines = self.Memory[0].shape[0]
        # print(NBlines)
        #瀵偓婵鍣伴梿鍡曟崲閸?
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
