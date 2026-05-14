# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:42:51 2026

@author: shuaibin
"""

import sys

GALAXY_SDK_PYTHON_DIR = r"D:\\GalaxySDK\\Development\\Samples\\Python\\"
sys.path.append(GALAXY_SDK_PYTHON_DIR)
try:
    import gxipy as gx
except Exception as error:
    raise ImportError(
        "Daheng camera SDK import failed. The configured Galaxy SDK directory may be wrong: "
        f"{GALAXY_SDK_PYTHON_DIR}. Import error: {error}"
    ) from error

def smoke_test_camera():
    device_manager = gx.DeviceManager()
    if device_manager.update_all_device_list()[0] == 0:
        print("No Device")
        return 1

    cam = device_manager.open_device_by_index(1)
    cam.stream_on()
    cam.stream_off()
    cam.close_device()
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke_test_camera())
