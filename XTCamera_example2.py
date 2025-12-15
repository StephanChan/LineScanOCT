import threading

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from industrial_camera.TUCam import *


class XinTu:
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(XinTu, "_instance"):
            with XinTu._instance_lock:
                if not hasattr(XinTu, "_instance"):
                    XinTu._instance = object.__new__(cls)
        return XinTu._instance

    def __init__(self):
        # 相机列表
        self.devList = {}

    def get_all_cams(self):
        cam = XTCamDev()
        cam.Open_device()
        return {cam.get_sn(): cam}


class XTCamDev:
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(XTCamDev, "_instance"):
            with XTCamDev._instance_lock:
                if not hasattr(XTCamDev, "_instance"):
                    XTCamDev._instance = object.__new__(cls)
        return XTCamDev._instance

    def __init__(self):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Api_Init(pointer(self.TUCAMINIT))
        print(self.TUCAMINIT.uiCamCount)
        print(self.TUCAMINIT.pstrConfigPath)
        print('Connect %d camera' % self.TUCAMINIT.uiCamCount)
        # 标记现在相机的打开状态
        self.is_open = False
        # 标记现在相机是否在取流
        self.is_grabbing = False

    def open(self, Idx):
        if not self.is_open:
            if Idx >= self.TUCAMINIT.uiCamCount:
                return

            self.TUCAMOPEN = TUCAM_OPEN(Idx, 0)

            TUCAM_Dev_Open(pointer(self.TUCAMOPEN))

            if 0 == self.TUCAMOPEN.hIdxTUCam:
                print('Open the camera failure!')
                return
            else:
                print('Open the camera success!')
            self.is_open = True
            # 设置分别率
            TUCAM_Capa_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDCAPA.TUIDC_RESOLUTION.value, 0)
            # 设置 HDR增益模式
            # 0:"HDR"
            # 1 :"High gain"
            # 2:"Low gain'
            # 3:"HDR - Raw
            # 4: "High gain - Raw
            # 5: "Low gain - Raw
            TUCAM_Prop_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDPROP.TUIDP_GLOBALGAIN.value, 2)
            # 设置位深为 8bit
            TUCAM_Capa_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDCAPA.TUIDC_BITOFDEPTH.value, 8)

    def Open_device(self):
        self.open(0)

    def set_roi(self, width, height, offset_x, offset_y):
        pass

    def start_grabbing(self):
        if not self.is_grabbing and self.is_open:
            self.m_fs = TUCAM_FILE_SAVE()
            self.m_frame = TUCAM_FRAME()
            self.m_format = TUIMG_FORMATS
            self.m_frformat = TUFRM_FORMATS
            self.m_capmode = TUCAM_CAPTURE_MODES

            self.m_frame.pBuffer = 0
            self.m_frame.ucFormatGet = self.m_frformat.TUFRM_FMT_USUAl.value
            self.m_frame.uiRsdSize = 1

            self.m_fs.nSaveFmt = self.m_format.TUFMT_TIF.value

            TUCAM_Buf_Alloc(c_int64(self.TUCAMOPEN.hIdxTUCam), pointer(self.m_frame))
            TUCAM_Cap_Start(c_int64(self.TUCAMOPEN.hIdxTUCam), self.m_capmode.TUCCM_SEQUENCE.value)

            self.offset = self.m_frame.usHeader
            # 图像数据大小
            self.data_size = self.m_frame.uiImgSize
            self.width = self.m_frame.usWidth
            self.height = self.m_frame.usHeight
            print(self.data_size)
            print(self.width)
            print(self.height)
            self.is_grabbing = True

    def robust_get_frame(self):
        if self.is_grabbing and self.is_open:
            # try:
            result = TUCAM_Buf_WaitForFrame(c_int64(self.TUCAMOPEN.hIdxTUCam), pointer(self.m_frame), 1000)
            # ImgName = './Image_' + str(1)
            self.m_fs.pFrame = pointer(self.m_frame)
            # print(self.m_frame.pBuffer)
            # print(self.m_frame.usHeader)
            # print(self.m_frame.usOffset)
            # print(self.m_frame.usWidth)
            # print(self.m_frame.usHeight)
            # print(self.m_frame.uiImgSize)
            # 每个像素占用字节数
            self.bytes_per_pixel = 1

            # 将整数地址转换为 ctypes 对象
            self.memory_address = ctypes.c_void_p(self.m_frame.pBuffer)

            # 偏移内存地址
            offset_memory_address = ctypes.c_void_p(self.memory_address.value + self.offset)

            # 从内存中读取数据
            self.buffer_type = ctypes.c_uint8 * (self.data_size // self.bytes_per_pixel)
            self.buffer = self.buffer_type.from_address(offset_memory_address.value)
            self.numpy_array = np.frombuffer(self.buffer, dtype=np.uint8, count=self.width * self.height)

            # 将一维数组转换为二维数组
            self.two_d_array = self.numpy_array.reshape((self.height, self.width))
            return self.two_d_array
        else:
            return []
        # return np.resize(self.two_d_array, (1024, 1280))
        # except Exception as e:
        #     print(e)
        #     print('Grab the frame failure, index number is %#d' % 1)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_sn(self):
        # SN
        TUCAM_Reg_Read = TUSDKdll.TUCAM_Reg_Read
        cSN = (c_char * 64)()
        pSN = cast(cSN, c_char_p)
        TUCAMREGRW = TUCAM_REG_RW(1, pSN, 64)
        TUCAM_Reg_Read(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAMREGRW)
        # print(bytes(bytearray(cSN)))
        return string_at(pSN).decode("utf-8")

    def stop_grabbing(self):
        if self.is_grabbing and self.is_open:
            TUCAM_Buf_AbortWait(c_int64(self.TUCAMOPEN.hIdxTUCam))
            TUCAM_Cap_Stop(c_int64(self.TUCAMOPEN.hIdxTUCam))
            TUCAM_Buf_Release(c_int64(self.TUCAMOPEN.hIdxTUCam))
            print("Xintu stop grabbing successfully!")
            self.is_grabbing = False

    def close(self):
        if self.is_open:
            if 0 != self.TUCAMOPEN.hIdxTUCam:
                TUCAM_Dev_Close(c_int64(self.TUCAMOPEN.hIdxTUCam))
        self.is_open = False
        self.is_grabbing = False
        print('Close the camera success')

    def set_frame_rate(self, fr):
        # 设置帧率
        pass

    def get_frame_rate(self):
        # 获取帧率
        return 45

    def get_max_frame_rate(self):
        # 获取最大帧率
        return 45

    def get_exp_min(self):
        # 获取最小曝光
        prop = TUCAM_PROP_ATTR()
        prop.nIdxChn = 0
        prop.idProp = TUCAM_IDPROP.TUIDP_EXPOSURETM.value
        TUCAM_Prop_GetAttr(c_int64(self.TUCAMOPEN.hIdxTUCam), pointer(prop))
        return prop.dbValMin * 1000.0

    def get_exp_max(self):
        # 获取最大曝光
        # return 20000
        prop = TUCAM_PROP_ATTR()
        prop.nIdxChn = 0
        prop.idProp = TUCAM_IDPROP.TUIDP_EXPOSURETM.value
        TUCAM_Prop_GetAttr(c_int64(self.TUCAMOPEN.hIdxTUCam), pointer(prop))
        return prop.dbValMax * 1000.0

    def get_exp(self):
        # 获取曝光时间
        value = c_double(0)
        result = TUCAM_Prop_GetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDPROP.TUIDP_EXPOSURETM.value,
                                     pointer(value), 0)
        # print("PropID=", TUCAM_IDPROP.TUIDP_EXPOSURETM.value, "The current value is=", value)
        return value.value * 1000.0

    def set_exp(self, exp_time):
        # 设置曝光时间
        TUCAM_Prop_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDPROP.TUIDP_EXPOSURETM.value,
                            c_double(exp_time / 1000.0), 0)

    def get_auto_exposure(self):
        # 获取自动曝光状态
        value = c_int32(0)
        TUCAM_Capa_GetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, pointer(value))
        # print("CapaID=", TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, "The current value is=", value)
        if value.value == 0:
            return False
        elif value.value == 1:
            return True

    def set_auto_exposure(self, mode):
        # 关闭或开启自动曝光  0 关闭  1 打开
        if mode == True:
            TUCAM_Capa_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 1)
        elif mode == False:
            TUCAM_Capa_SetValue(c_int64(self.TUCAMOPEN.hIdxTUCam), TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 0)

    def get_auto_gain(self):
        # 获取自动增益状态
        return 0

    def set_auto_gain(self, mode):
        # TUGAIN_HDR 0x00 HDR模式
        # TUGAIN_HIGH 0x01高增益模式
        # TUGAIN_LOW 0x02低增益模
        pass

    def get_gain(self):
        # 获取增益数值
        return 0

    def get_gain_min(self):
        # 获取增益最小值
        return 0

    def get_gain_max(self):
        # 获取增益最大值
        return 20

    def set_gain(self, gain):
        # 设置增益
        pass


if __name__ == '__main__':
    cam_cls = XinTu()
    cams = cam_cls.get_all_cams()
    print(cams)
    # cam = XTCamDev()
    # cam.start()
    # result_image = cam.read()
    # # cam.set_frame_rate(10)
    # cam.set_auto_exposure(False)
    # print(cam.get_auto_exposure())
    # cam.set_exp(10)
    # print(cam.get_exp())
    # print(cam.get_exp_min())
    # print(cam.get_exp_max())
    # # cam.set_auto_gain(False)
    # # cam.set_gain(0)
    # # print(cam.get_frame_rate())
    # # print(cam.get_auto_gain())
    # # print(cam.get_gain())
    # # print(cam.get_gain_min())
    # # print(cam.get_gain_max())
    # print(cam.get_width())
    # print(cam.get_height())
    # print(type(result_image))
    # print(result_image)
    # print(cam.get_sn())
    # # cv2.imwrite("xintuDhyana401D.tif", result_image)
    # plt.imshow(result_image, cmap="gray")
    # plt.show()
    # cam.stop()
    # cam.close()
