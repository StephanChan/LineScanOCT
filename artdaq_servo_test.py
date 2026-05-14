# -*- coding: utf-8 -*-
"""
Simple ART-DAQ servo test.

Sequence:
1. Start at the initial angle.
2. Move 90 degrees clockwise.
3. Wait 2 seconds.
4. Move 90 degrees counterclockwise back to the start angle.

Notes:
- The DAQ digital line is only the control signal.
- Power the servo from an external 5 V supply.
- Connect the servo ground to the DAQ ground.
"""

import sys
import time
import numpy as np

ARTDAQ_PYTHON_LIB_DIR = r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB\\"
sys.path.append(ARTDAQ_PYTHON_LIB_DIR)

try:
    import artdaq as ni
    from artdaq.constants import AcquisitionType as Atype
except Exception as error:
    raise ImportError(
        "ART-DAQ SDK import failed. The configured ART-DAQ Python library directory may be wrong: "
        f"{ARTDAQ_PYTHON_LIB_DIR}. Import error: {error}"
    ) from error

try:
    from artdaq.constants import LineGrouping
except Exception:
    LineGrouping = None


DEVICE_LINE = "Galvo/port0/line0"
SAMPLE_RATE_HZ = 10_000
PWM_FREQUENCY_HZ = 50

START_ANGLE_DEG = 0.0
MOVE_DELTA_DEG = 30.0
MOVE_TIME_S = 0.7
WAIT_TIME_S = 2.0

# Adjust these if your servo uses a different range.
MIN_PULSE_US = 500
MAX_PULSE_US = 2500


def add_do_chan_single_line(task, line_name):
    if LineGrouping is None:
        task.do_channels.add_do_chan(lines=line_name)
    else:
        task.do_channels.add_do_chan(
            lines=line_name,
            line_grouping=LineGrouping.CHAN_PER_LINE,
        )


def angle_to_pulse_width_us(angle_deg):
    angle_deg = float(np.clip(angle_deg, 0.0, 180.0))
    return MIN_PULSE_US + (MAX_PULSE_US - MIN_PULSE_US) * (angle_deg / 180.0)


def build_constant_angle_waveform(angle_deg, duration_s):
    pulse_width_us = angle_to_pulse_width_us(angle_deg)
    samples_per_period = int(round(SAMPLE_RATE_HZ / PWM_FREQUENCY_HZ))
    high_samples = int(round(SAMPLE_RATE_HZ * pulse_width_us / 1_000_000.0))
    high_samples = max(1, min(high_samples, samples_per_period - 1))

    single_period = np.zeros(samples_per_period, dtype=np.bool_)
    single_period[:high_samples] = True

    cycle_count = max(1, int(round(duration_s * PWM_FREQUENCY_HZ)))
    waveform = np.tile(single_period, cycle_count)
    return waveform, pulse_width_us


def play_waveform(task, waveform, duration_s):
    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE_HZ,
        sample_mode=Atype.FINITE,
        samps_per_chan=len(waveform),
    )
    task.write(waveform, auto_start=False)
    task.start()
    task.wait_until_done(timeout=duration_s + 2.0)
    task.stop()


def hold_angle(task, angle_deg, duration_s, label):
    waveform, pulse_width_us = build_constant_angle_waveform(angle_deg, duration_s)
    print(
        f"{label}: angle={angle_deg:.1f} deg, pulse={pulse_width_us:.0f} us, "
        f"duration={duration_s:.1f} s"
    )
    play_waveform(task, waveform, duration_s)


def release_line_low():
    release_task = ni.Task("ServoRelease")
    try:
        add_do_chan_single_line(release_task, DEVICE_LINE)
        release_task.write(False, auto_start=True)
        time.sleep(0.05)
    finally:
        try:
            release_task.stop()
        except Exception:
            pass
        release_task.close()


def main():
    target_angle_deg = float(np.clip(START_ANGLE_DEG + MOVE_DELTA_DEG, 0.0, 180.0))

    task = ni.Task("ServoPWM")
    try:
        add_do_chan_single_line(task, DEVICE_LINE)
        hold_angle(task, target_angle_deg, MOVE_TIME_S, "Move clockwise 90 deg")
        hold_angle(task, target_angle_deg, WAIT_TIME_S, "Wait at destination")
        hold_angle(task, START_ANGLE_DEG, MOVE_TIME_S, "Move back counterclockwise 90 deg")
    finally:
        task.close()
        release_line_low()

    print("Servo sequence complete.")


if __name__ == "__main__":
    main()
