# -*- coding: utf-8 -*-
"""Standalone LED indicator wiring test (ART-DAQ digital outputs on port 1).

Cycles the three LED outputs the software drives, so the wiring can be checked
without starting the GUI:

    port1/line0  illumination LED   high while the USB camera is in use
    port1/line1  acquisition LED    red: an acquisition is running
    port1/line2  idle LED           green: no acquisition is running

The DAQ device is the ``AODOboard`` value of config.ini (normally ``Galvo``).
Run from the project root, e.g.

    C:\\Users\\shuaibin\\.conda\\envs\\python311_env\\python.exe artdaq_led_test.py
"""

import time

from HardwareSpecs import LED_ACQUISITION_LINE, LED_IDLE_LINE, LED_ILLUMINATION_LINE
from LedIndicators import (
    acquisition_finished,
    acquisition_started,
    all_off,
    illumination_off,
    illumination_on,
    led_device_name,
)

HOLD_SECONDS = 1.0


def step(line_name, description, led_call):
    ok = led_call()
    print(
        f"{led_device_name()}/{line_name} {description} -> "
        + ("ok" if ok else "FAILED (see the warning above)")
    )
    time.sleep(HOLD_SECONDS)


def main():
    print(f"LED indicator test on ART-DAQ device '{led_device_name()}' (port 1)")
    step(LED_ILLUMINATION_LINE, "high (illumination before the camera opens)", illumination_on)
    step(LED_ILLUMINATION_LINE, "low  (camera done)", illumination_off)
    step(LED_ACQUISITION_LINE, "high + idle low (acquisition running)", acquisition_started)
    step(LED_IDLE_LINE, "high + acquisition low (idle)", acquisition_finished)
    all_off()
    print("All LED lines low. Test finished.")


if __name__ == "__main__":
    main()
