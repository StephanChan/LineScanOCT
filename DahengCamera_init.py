# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:42:51 2026

@author: shuaibin
"""

import sys
sys.path.append(r"D:\\GalaxySDK\\Development\\Samples\\Python\\")
import gxipy as gx 

# 打开设备
device_manager = gx.DeviceManager()
if device_manager.update_all_device_list()[0] == 0:
    print("No Device")
    sys.exit(1)

cam = device_manager.open_device_by_index(1)

cam.stream_on()

cam.stream_off()

cam.close_device()
