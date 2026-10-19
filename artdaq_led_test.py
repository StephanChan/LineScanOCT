# -*- coding: utf-8 -*-
"""Standalone LED indicator wiring test (ART-DAQ digital outputs on port 0).

Cycles the LED outputs the software drives, so the wiring can be checked without
starting the GUI:

    port0/line2 + port0/line3  illumination LEDs   high while the USB camera is in use
    port0/line7                warning LED         high while an acquisition runs

The DAQ device is the ``AODOboard`` value of config.ini (normally ``Galvo``).
Run from the project root, e.g.

    C:\\Users\\shuaibin\\.conda\\envs\\python311_env\\python.exe artdaq_led_test.py
"""

import time

from HardwareSpecs import LED_ILLUMINATION_LINES, LED_WARNING_LINE
from LedIndicators import (
    all_off,
    illumination_off,
    illumination_on,
    led_device_name,
    set_acquisition_state,
)

HOLD_SECONDS = 1.0

ILLUMINATION_LABEL = " + ".join(LED_ILLUMINATION_LINES)


def step(line_name, description, led_call):
    ok = led_call()
    print(
        f"{led_device_name()}/{line_name} {description} -> "
        + ("ok" if ok else "FAILED (see the warning above)")
    )
    time.sleep(HOLD_SECONDS)


def main():
    print(f"LED indicator test on ART-DAQ device '{led_device_name()}' (port 0)")
    step(ILLUMINATION_LABEL, "high (illumination before the camera opens)", illumination_on)
    step(ILLUMINATION_LABEL, "low  (camera done)", illumination_off)
    step(LED_WARNING_LINE, "high (acquisition running)", lambda: set_acquisition_state(True))
    step(LED_WARNING_LINE, "low  (no acquisition)", lambda: set_acquisition_state(False))
    all_off()
    print("All LED lines low. Test finished.")


if __name__ == "__main__":
    main()
