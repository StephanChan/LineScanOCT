# -*- coding: utf-8 -*-
"""LED status / illumination indicators on ART-DAQ digital outputs (port 0).

Two LEDs are driven on port 0 of the ART-DAQ device the GUI names (``AODOboard``
in config.ini, normally ``Galvo``).  The terminals themselves are defined centrally
in ``HardwareSpecs.py``:

``port0/line2`` + ``port0/line3``
    Illumination LEDs (one line per lamp, always switched together) - high while
    the USB camera is in use (live view, sample locator, coordinate calibration).
    Switched on by ``SampleLocator.open_usb_camera`` *before* the sensor starts
    streaming and switched off by ``SampleLocator.close_usb_camera`` when the
    camera is done.
``port0/line7``
    Warning LED - high while an acquisition command is running.

The warning LED is switched by ``ThreadWeaver.WeaverThread`` around every queued
command (``WeaverThread.set_acquisition_leds`` -> ``set_acquisition_state``), so it
is lit for exactly as long as the software is acquiring - the USB-only live view
(``CAMERA_ONLY_ACTIONS``) never lights it.

Port 0 also carries the acquisition outputs, so the LED lines deliberately avoid
them: ``port0/line0`` is the camera trigger (``SyncDO`` in the GUI) and
``port0/line1`` is the servo output (``ThreadAODO_art.SERVO_DEVICE_LINE``).  An
on/off probe on this board (``_tmp_port0_probe_cells.py``) showed that a LED task
which stays open while the hardware-timed acquisition DOtask and its waveform run
on port0/line0 does not conflict with it and is not disturbed by the waveform -
that is why the indicators may share the port with the acquisition.

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
    LED_DEVICE_CONFIG_KEY,
    LED_DEVICE_DEFAULT,
    LED_ILLUMINATION_LINES,
    LED_WARNING_LINE,
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

# every line an indicator drives: device changes and all_off() walk this list
_LED_LINES = tuple(LED_ILLUMINATION_LINES) + (LED_WARNING_LINE,)


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
        for line_name in _LED_LINES:
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


def _set_lines(line_names, state):
    """Drive a group of lines that always switch together.

    Every line is written (no short circuit), so one unavailable lamp can never
    hide a failure of the other; the result reports whether all were driven.
    """
    results = [set_led(line_name, state) for line_name in line_names]
    return all(results)


def illumination_on():
    """Sample illumination on (USB camera about to stream)."""
    return _set_lines(LED_ILLUMINATION_LINES, True)


def illumination_off():
    """Sample illumination off (USB camera done)."""
    return _set_lines(LED_ILLUMINATION_LINES, False)


def warning_on():
    """Warning LED on: an acquisition command is running."""
    return set_led(LED_WARNING_LINE, True)


def warning_off():
    """Warning LED off: no acquisition is running."""
    return set_led(LED_WARNING_LINE, False)


def set_acquisition_state(acquiring):
    """Warning LED = ``acquiring`` (one lamp, there is no idle twin any more)."""
    return warning_on() if bool(acquiring) else warning_off()


def all_off():
    """Switch every LED off and release the DO tasks (software exit)."""
    with _lock:
        for line_name in _LED_LINES:
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
