# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:58:04 2026

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:17:06 2025
@author: admin
"""
import sys
sys.path.append(r"D:\\GalaxySDK\\Development\\Samples\\Python\\")
import gxipy as gx 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# sys.path.append(r"C:\\Program Files (x86)\\ART Technology\\ART-DAQ\\Samples\\Python\\LIB\\")
# import artdaq
# from artdaq.constants import LineGrouping
from ctypes import *
from gxipy.gxidef import *
from gxipy.ImageFormatConvert import *
import matplotlib.pyplot as plt
import time

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

def convert_to_special_pixel_format(raw_image, pixel_format):
    image_convert.set_dest_format(pixel_format)
    # print('raw_image.get_pixel_format()', raw_image.get_pixel_format())
    valid_bits = get_best_valid_bits(raw_image.get_pixel_format())
    # print('valid_bits', valid_bits)
    image_convert.set_valid_bits(valid_bits)

    # create out put image buffer
    buffer_out_size = image_convert.get_buffer_size_for_conversion(raw_image)
    output_image_array = (c_ubyte * buffer_out_size)()
    output_image = addressof(output_image_array)

    # convert to pixel_format
    image_convert.convert(raw_image, output_image, buffer_out_size, False)
    if output_image is None:
        print('Pixel format conversion failed')
        return

    return output_image_array, buffer_out_size

# 参数设置
num_images = 8
exposure_time = 1000.0  # 若为 -1 则开启自动曝光
gain_value = 1.0         # 若为 -1 则开启自动增益
base_save_path = "brain_5"

width, height = 1600, 1104
pixel_format = "Mono12Packed"

# 创建保存路径
save_dir = os.path.join(base_save_path, 'Ori')
ave_dir = os.path.join(base_save_path, 'ave')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(ave_dir, exist_ok=True)

# 打开设备
device_manager = gx.DeviceManager()
global image_convert
image_convert = device_manager.create_image_format_convert()
if device_manager.update_all_device_list()[0] == 0:
    print("No Device")
    sys.exit(1)

cam = device_manager.open_device_by_index(1)
# print("Success Open Device")

# 设备参数设置
remote = cam.get_remote_device_feature_control()

remote.feature_save("export_config_file.txt")
# trigger_soft_ware_feature =  remote.get_register_feature( "TriggerSoftware")
remote.get_enum_feature("TriggerMode").set('Off')

# remote.get_enum_feature("TriggerMode").set('On')
remote.get_enum_feature("TriggerSource").set('Line2')
remote.get_enum_feature("TriggerActivation").set('RisingEdge')
# === 自动曝光与增益控制 ===
remote.get_enum_feature("ExposureAuto").set("Off")
remote.get_enum_feature("GainAuto").set("Off")
remote.get_int_feature("Width").set(width)
remote.get_int_feature("Height").set(height)
remote.get_int_feature("OffsetX").set(0)
remote.get_int_feature("OffsetY").set(0)
cam.data_stream[0].set_acquisition_buffer_number(1000)
# print("[INFO] ExposureAuto and GainAuto set to Off")

# if exposure_time == -1:
#     remote.get_enum_feature("ExposureAuto").set("On")
#     print("[INFO] ExposureAuto set to On")
# else:
#     remote.get_float_feature("ExposureTime").set(exposure_time)
#     print(f"[INFO] ExposureTime set to {exposure_time} µs")

# if gain_value == -1:
#     remote.get_enum_feature("GainAuto").set("On")
#     print("[INFO] GainAuto set to On")
# else:
#     remote.get_float_feature("Gain").set(gain_value)
#     print(f"[INFO] Gain set to {gain_value} dB")

# 像素格式
remote.get_enum_feature("PixelFormat").set(pixel_format)
# print(f"[INFO] PixelFormat set to {pixel_format}")
# print("Success Set Device")

# # 开灯
# with artdaq.Task() as light_task:
#     light_task.do_channels.add_do_chan("Robot/port1/line6:7", line_grouping=LineGrouping.CHAN_PER_LINE)
#     light_task.write([1, 1])

# 图像采集
cam.stream_on()
all_images = []
# print("StreamAnnouncedBufferCount", cam.data_stream[0].get_feature_control().get_int_feature("StreamAnnouncedBufferCount").__feature_name)
# for i in range(num_images):
#     t0=time.time()
#     raw_image = cam.data_stream[0].get_image(200)
#     # raw_image = cam.data_stream[0].dq_buf()
#     # cam.data_stream[0].q_buf(raw_image)
#     t1=time.time()
#     if raw_image == None:
#         print("camera time out...")
#         # Bline = np.zeros([self.AlinesPerBline, self.NSamples])
#     else:
#         if pixel_format == "Mono12Packed":
#             mono_image_array, buffer_out_size=convert_to_special_pixel_format(raw_image, GxPixelFormatEntry.MONO12)
#             img = np.frombuffer(mono_image_array, dtype=np.uint16) .reshape(height,width)
#         else:
#             img = raw_image.get_numpy_array()
#         t2=time.time()
#     print(img[0,0:4])

# plt.figure()
# plt.imshow(img)
# plt.show()

    # all_images.append(img)

    # 保存每张图
    # Image.fromarray(img.astype(np.uint16)).save(os.path.join(save_dir, f"{i+1}.tiff"), format="TIFF", compression=None)
    # print(f"Success Save Image {i+1}")
    # print('camera fetch data took: ', round((t1-t0)*1000,3), 'msec')
    # print('data conversion took: ', round((t2-t1)*1000,3), 'msec')
# print('data into memory took: ', round((t3-t2)*1000,3), 'msec')
cam.stream_off()

# # 关灯
# with artdaq.Task() as light_task:
#     light_task.do_channels.add_do_chan("Robot/port1/line6:7", line_grouping=LineGrouping.CHAN_PER_LINE)
#     light_task.write([0, 0])

cam.close_device()
# print("Success Close Device")

# 平均图像计算与保存
# average_img = np.mean(all_images, axis=0)
# Image.fromarray(average_img.astype(np.uint16)).save(os.path.join(ave_dir, "average_image.tiff"), format="TIFF", compression=None)

# # 展示
# plt.imshow(all_images[0], cmap='gray')
# plt.title("First Image")
# plt.axis('off')
# plt.show()
# plt.imshow(average_img, cmap='gray')
# plt.title("Average Image")
# plt.axis('off')
# plt.show()