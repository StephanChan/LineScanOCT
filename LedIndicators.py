# -*- coding: utf-8 -*-
"""LED status / illumination indicators on ART-DAQ digital outputs (port 1).

Three LEDs are driven from port 1 of the ART-DAQ device the GUI names
(``AODOboard`` in config.ini, normally ``Galvo``).  The terminals themselves are
defined centrally in ``HardwareSpecs.py``:

``port1/line0``
    Illumination LED - high while the USB camera is in use (live view, sample
    locator, coordinate calibration).  Switched on by
    ``SampleLocator.open_usb_camera`` *before* the sensor starts streaming and
    switched off by ``SampleLocator.close_usb_camera`` when the camera is done.
``port1/line1``
    Acquisition LED (red) - high while an acquisition command is running.
``port1/line2``
    Idle LED (green) - high while no acquisition is running.

The acquisition / idle pair is switched by ``ThreadWeaver.WeaverThread`` around
every queued command (``WeaverThread.set_acquisition_leds``), so the red LED is
on for exactly as long as the software is acquiring - the USB-only live view is
the one command that keeps the green LED on.

Every write is best-effort.  Without the ART-DAQ SDK, without the configured
device, or if a line is unavailable, a warning is printed once and the run
continues: an indicator LED must never abort a scan.

Each line is held by one long-lived ``artdaq`` task (created on first use and kept
open) so the line keeps its level between two writes instead of dropping back
when a per-write task would be closed.  ``all_off()`` releases them (software
exit).
"""

import sys
import threading

from HardwareSpecs import (
    LED_ACQUISITION_LINE,
    LED_DEVICE_CONFIG_KEY,
    LED_DEVICE_DEFAULT,
    LED_IDLE_LINE,
    LED_ILLUMINATION_LINE,
)

CONFIG_PATH = "config.ini"

# Master switch: set False to run the software without touching the LEDs at all.
LED_CONTROL_ENABLED = True

# ART-DAQ device identifier.  ``None`` -> the device the GUI reports
# (``set_led_device``) or, failing that, ``AODOboard`` from config.ini.
LED_DEVICE_NAME = None

ARTDAQ_PYTHON_LIB_DIR = r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB\\"

try:
    if ARTDAQ_PYTHON_LIB_DIR not in sys.path:
        sys.path.append(ARTDAQ_PYTHON_LIB_DIR)
    import artdaq as ni

    try:
        from artdaq.constants import LineGrouping
    except Exception:
        LineGrouping = None
except Exception as error:
    print(
        "ART-DAQ SDK import failed for the LED indicators "
        f"({error}); LED outputs disabled."
    )
    ni = None
    LineGrouping = None

try:
    from PyQt5.QtCore import QSettings
except Exception:
    QSettings = None

# line suffix -> open artdaq task holding that line
_tasks = {}
# lines that could not be configured: never retried, to keep the log quiet
_failed_lines = set()
_warned = set()
# serialises the task table (set_led_device / set_led / all_off)
_lock = threading.RLock()


def led_device_name():
    """Device identifier the LED terminals are built from.

    ``set_led_device`` wins (the GUI knows the live ``AODOboard`` value), then
    ``AODOboard`` from config.ini, then ``LED_DEVICE_DEFAULT``.
    """
    if LED_DEVICE_NAME:
        return str(LED_DEVICE_NAME)
    name = ""
    if QSettings is not None:
        try:
            settings = QSettings(CONFIG_PATH, QSettings.IniFormat)
            name = str(settings.value(LED_DEVICE_CONFIG_KEY, "") or "").strip()
        except Exception as error:
            _warn_once(f"Could not read {LED_DEVICE_CONFIG_KEY} from {CONFIG_PATH}: {error}")
    return name or LED_DEVICE_DEFAULT


def set_led_device(name):
    """Adopt the DAQ device name shown in the GUI.

    When the name changes, the tasks already holding lines on the previous device
    are closed so the next write re-opens them on the new device.  Returns the
    device name now in use.
    """
    global LED_DEVICE_NAME
    name = str(name or "").strip()
    if not name or name == led_device_name():
        return led_device_name()
    with _lock:
        for task in _tasks.values():
            try:
                task.close()
            except Exception:
                pass
        _tasks.clear()
        for line_name in (LED_ILLUMINATION_LINE, LED_ACQUISITION_LINE, LED_IDLE_LINE):
            _failed_lines.discard(line_name)
        LED_DEVICE_NAME = name
    return LED_DEVICE_NAME


def _task_for(line_name):
    """Open (once) and return the artdaq task holding ``line_name``."""
    task = _tasks.get(line_name)
    if task is not None:
        return task
    terminal = f"{led_device_name()}/{line_name}"
    task = ni.Task()
    try:
        if LineGrouping is None:
            task.do_channels.add_do_chan(lines=terminal)
        else:
            task.do_channels.add_do_chan(
                lines=terminal,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
    except Exception:
        try:
            task.close()
        except Exception:
            pass
        raise
    _tasks[line_name] = task
    return task


def set_led(line_name, state):
    """Drive one LED line high (``True``) or low (``False``).  Never raises."""
    if not LED_CONTROL_ENABLED or ni is None or line_name in _failed_lines:
        return False
    with _lock:
        try:
            _task_for(line_name).write(bool(state), auto_start=True)
            return True
        except Exception as error:
            _failed_lines.add(line_name)
            task = _tasks.pop(line_name, None)
            if task is not None:
                try:
                    task.close()
                except Exception:
                    pass
            _warn_once(
                f"LED line {led_device_name()}/{line_name} unavailable ({error}); "
                "that indicator is disabled for this session."
            )
            return False


def _warn_once(message):
    if message in _warned:
        return
    _warned.add(message)
    print(message)


def illumination_on():
    """Sample illumination on (USB camera about to stream)."""
    return set_led(LED_ILLUMINATION_LINE, True)


def illumination_off():
    """Sample illumination off (USB camera done)."""
    return set_led(LED_ILLUMINATION_LINE, False)


def set_acquisition_state(acquiring):
    """Red acquisition LED = ``acquiring``, green idle LED = not ``acquiring``."""
    active = bool(acquiring)
    acquisition_ok = set_led(LED_ACQUISITION_LINE, active)
    idle_ok = set_led(LED_IDLE_LINE, not active)
    return acquisition_ok and idle_ok


def acquisition_started():
    """Acquisition underway: red LED on, green idle LED off."""
    return set_acquisition_state(True)


def acquisition_finished():
    """No acquisition: red LED off, green idle LED on."""
    return set_acquisition_state(False)


def all_off():
    """Switch every LED off and release the DO tasks (software exit)."""
    with _lock:
        for line_name in (LED_ILLUMINATION_LINE, LED_ACQUISITION_LINE, LED_IDLE_LINE):
            if LED_CONTROL_ENABLED and ni is not None and line_name not in _failed_lines:
                try:
                    _task_for(line_name).write(False, auto_start=True)
                except Exception as error:
                    _warn_once(f"Could not switch LED line {line_name} off: {error}")
            task = _tasks.pop(line_name, None)
            if task is not None:
                try:
                    task.close()
                except Exception as error:
                    _warn_once(f"Could not close the LED task for {line_name}: {error}")
