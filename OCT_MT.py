
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:14:40 2023

@author: Shuaibin Chang
"""

# Queue functions:
# maxsize – Number of items allowed in the queue.
# empty() – Return True if the queue is empty, False otherwise.
# full() – Return True if there are maxsize items in the queue. If the queue was initialized with maxsize=0 (the default), then full() never returns True.
# get() – Remove and return an item from the queue. If queue is empty, wait until an item is available.
# get_nowait() – Return an item if one is immediately available, else raise QueueEmpty.
# put(item) – Put an item into the queue. If the queue is full, wait until a free slot is available before adding the item.
# put_nowait(item) – Put an item into the queue without blocking. If no free slot is immediately available, raise QueueFull.
# qsize() – Return the number of items in the queue.

# TODO: revisit pause handling so it does not depend on thread interruption.
# TODO: add standalone dynamic processing workflow for a single B-line.
# TODO: extend full-volume dynamic workflows and storage layout where offline dynamic processing is still incomplete.
# TODO: add automated region scan workflows for organoid and slice use cases.
# TODO: add standalone mosaic dynamic-processing workflow for a selected region.
# TODO: extend scheduled longitudinal dynamic scans beyond the current timed plate workflow.

import sys
import os
import cv2
import json
import numpy as np
from queue import Queue
from PyQt5.QtWidgets import QApplication
from Dialogs import StageDialog
from PyQt5 import QtWidgets as QW
import PyQt5.QtCore as qc
from PyQt5.QtCore import QObject, pyqtSignal
from mainWindow import MainWindow
from ActionFields import *
from ActionTypes import AcqTypes, DnSActions, GPUActions, WeaverActions
from FileNaming import FileNaming
from Generaic_functions import LOG
from ScanSession import load_session_data, populate_sample_selector, save_session_data
import time
from SampleLocator import (
    MosaicUSBSampleScanner,
    blank_usb_frame,
    capture_usb_frame,
    capture_usb_frame_averaged,
    default_usb_mosaic_calibration,
)
from usb_training_dataset import export_sample_locator_dataset
from Display_rendering import (
    render_aodo_waveform_ready,
    render_aline_ready,
    render_bline_ready,
    render_cscan_ready,
    render_cscan_xz_dual,
    render_cscan_xz_from_volume,
    render_mosaic_ready,
    render_mosaic_xz_slices,
    render_td_enface,
)
from HardwareSpecs import PHOTONFOCUS_STATIC_NORMALIZATION_MEAN, get_objective_spec

CONTINUOUS_ACQ_MODES = (
    AcqTypes.CONTINUOUS_ALINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.CONTINUOUS_CSCAN,
)

SAMPLE_LOCATOR_Z_MM = 1.0

# Axial (depth) pixel size in micrometres, used for the XZ-plane display aspect
# ratio (µm per depth pixel).
AXIAL_PIXEL_SIZE_UM = 3.5 # 4.6 in air, 3.5um in water

FINITE_ACQ_MODES = (
    AcqTypes.FINITE_ALINE,
    AcqTypes.FINITE_BLINE,
    AcqTypes.FINITE_CSCAN,
    AcqTypes.FAST_VOLUME_CSCAN,
    AcqTypes.TD_ENFACE,
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)

LIVE_ONLY_MODES = (
    AcqTypes.LOCATION_CAMERA_LIVE,
    AcqTypes.MOSAIC,
)

MOSAIC_DISPLAY_MODES = (
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)

# How often (in acquired Y lines) the live mosaic *volumes* are refreshed while a
# FOV is still being scanned. XYplaneInt / XYplaneDyn (and the mosaic XZ panes)
# are rendered from the stitched 3-D volumes, so with a refresh only at the end of
# a FOV those views froze on the previous tile (or stayed black) for the whole
# acquisition. Refreshing every N lines keeps them live; the full-volume
# block-mean downsample costs ~0.5 s, so this is a deliberate trade-off between
# view latency and DnS time (N=16 -> ~16 refreshes per 247-line tile, ~8 s).
# Read by DnSThread.Process_Mosaic (see DnSThread._mosaic_volume_refresh_interval).
MOSAIC_VOLUME_REFRESH_LINES = 16

# Shared raw-data ring buffer. More than two slots allows acquisition and processing to overlap safely.
global memoryCount
memoryCount = 6

global Memory
Memory = list(range(memoryCount))


# Simulation switch for running without live hardware.
global SIM
SIM = False
# Optional mayavi-style 3D visualization path.
global use_maya
use_maya = False

# Queue topology for thread-to-thread coordination.
WeaverQueue = Queue()
# Queue for galvo / stage control actions.
AODOQueue = Queue()
# Status/ack queue from the galvo / stage thread back to Weaver.
StagebackQueue = Queue()
# Queue for display and save actions.
DnSQueue = Queue()
# Queue for GPU / CPU FFT processing actions.
GPUQueue = Queue()
# Status/ack queue from GPU thread back to Weaver.
GPU2weaverQueue = Queue()
# Queue for camera-control actions.
DQueue = Queue()
# Status/ack queue from the camera thread back to Weaver.
DbackQueue = Queue()
# Queue carrying completed raw-data memory slots.
DatabackQueue = Queue()
MosaicQueue = Queue()

        
# Thread wrappers inject shared queues, memory, logging, and the UI bridge.

from ThreadCamera_DH import Camera
class Camera_2(Camera):
    def __init__(self, ui, log):
        super().__init__()
        global Memory
        self.memoryCount = memoryCount
        self.Memory = Memory
        self.ui = ui
        self.queue = DQueue
        self.DbackQueue = DbackQueue
        self.DatabackQueue = DatabackQueue
        self.log = log
        self.SIM = SIM    
        self.ui_bridge = None
            

from ThreadWeaver import WeaverThread
class WeaverThread_2(WeaverThread):
    def __init__(self, ui, log):
        super().__init__()
        global Memory
        self.Memory = Memory
        self.memoryCount = memoryCount
        self.ui = ui
        self.queue = WeaverQueue
        self.DnSQueue = DnSQueue
        self.AODOQueue = AODOQueue
        self.StagebackQueue = StagebackQueue
        self.DbackQueue = DbackQueue
        self.DatabackQueue = DatabackQueue
        self.GPUQueue = GPUQueue
        self.DQueue = DQueue
        self.GPU2weaverQueue = GPU2weaverQueue
        self.MosaicQueue = MosaicQueue
        self.log = log
        self.ui_bridge = None

# GPU processing thread wrapper.
from ThreadGPU import GPUThread
class GPUThread_2(GPUThread):
    def __init__(self, ui, log):
            super().__init__()
            global Memory
            self.Memory = Memory
            self.ui = ui
            self.queue = GPUQueue
            self.DnSQueue = DnSQueue
            self.GPU2weaverQueue = GPU2weaverQueue
            self.log = log
            self.SIM = SIM
            self.AMPLIFICATION = 100#AMPLIFICATION
            self.default_static_normalization_mean = PHOTONFOCUS_STATIC_NORMALIZATION_MEAN
            self.static_normalization_mean = self.default_static_normalization_mean
            self.dynamic_use_first_frame_background = False
            self.ui_bridge = None
            
# Galvo / stage control thread wrapper.
from ThreadAODO_art import AODOThread
class AODOThread_2(AODOThread):
    def __init__(self, ui, log):
        super().__init__()
        self.ui = ui
        self.queue = AODOQueue
        self.StagebackQueue = StagebackQueue
        self.log = log
        self.SIM = SIM
        self.ui_bridge = None

# Display and save thread wrapper.
from ThreadDnS import DnSThread
class DnSThread_2(DnSThread):
    def __init__(self, ui, log):
        super().__init__()
        self.ui = ui
        self.queue = DnSQueue
        self.MosaicQueue = MosaicQueue
        self.log = log
        self.use_maya = use_maya
        self.ui_bridge = None
        

# GUI-thread bridge for cross-thread status and display payload delivery.
class UiBridge(QObject):
    status_message = pyqtSignal(str)
    acquisition_controls_locked = pyqtSignal(bool)
    cu_slice_value = pyqtSignal(int)
    time_reader_value = pyqtSignal(int)
    aline_ready = pyqtSignal(object)   # dict payload
    bline_ready = pyqtSignal(object)   # dict payload
    cscan_ready = pyqtSignal(object)   # dict payload
    mosaic_ready = pyqtSignal(object)  # dict payload
    aodo_waveform_ready = pyqtSignal(object)  # dict payload


# Main GUI object with thread wiring and queue orchestration.
class GUI(MainWindow):
    def __init__(self):
        super().__init__()
        # Axial (depth) pixel size used for XZ-plane aspect-ratio display.
        self.ui.axial_pixel_size_um = AXIAL_PIXEL_SIZE_UM
        # if use_maya:
        #     self.addMaya()
        self.log = LOG(self.ui)
        self.log.install_stream_redirects()
        
        self.FOV_locations = []
        self.sample_centers = []
        self.raw_img = []
        self.pixel_polygons = []
        
        self.ui.RunButton.clicked.connect(self.run_task)
        self.ui.PauseButton.clicked.connect(self.Pause_task)
        self.ui.CenterGalvo.clicked.connect(self.CenterGalvo)
        # YGalvoBias live DC update to the second (Y) galvo through the AODO
        # thread. Debounced so continuous spin-box dragging collapses into a
        # single AO write instead of flooding the AODO queue.
        self._ygalvo_timer = qc.QTimer(self)
        self._ygalvo_timer.setSingleShot(True)
        self._ygalvo_timer.setInterval(150)
        self._ygalvo_timer.timeout.connect(self._apply_ygalvo_voltage)
        self.ui.YGalvoBias.valueChanged.connect(self._schedule_ygalvo_voltage)
        self._add_sample_center_controls()
        self.ui.SampleLocateButton.clicked.connect(self.LocateSample)
        # set window length for FFT
        # self.ui.PostSamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.PreSamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.PostSamples_2.valueChanged.connect(self.update_Dispersion)
        # self.ui.DelaySamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.TrimSamples.valueChanged.connect(self.update_Dispersion)
        # set stage boundary
        self.ui.XZmax.valueChanged.connect(self.Update_contrast)
        # self.ui.DepthStart.valueChanged.connect(self.Update_contrast_Bline)
        # self.ui.DepthRange.valueChanged.connect(self.Update_contrast_Bline)
        self.ui.XZmin.valueChanged.connect(self.Update_contrast)
        if hasattr(self.ui, "ZDepthBar"):
            self.ui.ZDepthBar.valueChanged.connect(self.Update_contrast)
        self.ui.DynContrast.valueChanged.connect(self.Update_contrast)
        if hasattr(self.ui, "DynBrightness"):
            self.ui.DynBrightness.valueChanged.connect(self.Update_contrast)
        if hasattr(self.ui, "YBar"):
            self.ui.YBar.valueChanged.connect(self._on_ybar_changed)
        if hasattr(self.ui, "MosaicYbar"):
            self.ui.MosaicYbar.valueChanged.connect(self._on_mosaic_ybar_changed)
        # self.ui.Dynmax.valueChanged.connect(self.Update_contrast_Dyn)
        # self.ui.Dynmin.valueChanged.connect(self.Update_contrast_Dyn)

        self.ui.redoBG.clicked.connect(self.redo_background)
        self.ui.redoSurf.clicked.connect(self.redo_surface)
        self.ui.BG_DIR.textChanged.connect(self.update_background)
        self.ui.AlinesPerBline.valueChanged.connect(self.update_background)
        self.ui.offsetH.valueChanged.connect(self.update_background)
        self.ui.NSamples_PF.valueChanged.connect(self.update_background)
        self.ui.NSamples_DH.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_DH.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_DH.valueChanged.connect(self.update_Dispersion)
        self.ui.SpectralDS_PF.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_PF.valueChanged.connect(self.update_Dispersion)
        self.ui.offsetW_PF.valueChanged.connect(self.update_background)
        self.ui.offsetW_DH.valueChanged.connect(self.update_background)
        self.ui.Camera.currentTextChanged.connect(self.update_background)
        self.ui.Camera.currentTextChanged.connect(self.update_Dispersion)
        self.ui.InD_DIR.textChanged.connect(self.update_Dispersion)
        self.ui.Xmove2.clicked.connect(self.Xmove2)
        self.ui.Ymove2.clicked.connect(self.Ymove2)
        self.ui.Zmove2.clicked.connect(self.Zmove2)
        self.ui.XUP.clicked.connect(self.XUP)
        self.ui.YUP.clicked.connect(self.YUP)
        self.ui.ZUP.clicked.connect(self.ZUP)
        self.ui.XHome.clicked.connect(self.XHome)
        self.ui.YHome.clicked.connect(self.YHome)
        self.ui.ZHome.clicked.connect(self.ZHome)
        self.ui.XDOWN.clicked.connect(self.XDOWN)
        self.ui.YDOWN.clicked.connect(self.YDOWN)
        self.ui.ZDOWN.clicked.connect(self.ZDOWN)
        
        self.ui.XSpeed.valueChanged.connect(self.SetXSpeed)
        self.ui.YSpeed.valueChanged.connect(self.SetYSpeed)
        self.ui.ZSpeed.valueChanged.connect(self.SetZSpeed)
        
        self.ui.XAccelerate.valueChanged.connect(self.SetXAcc)
        self.ui.YAccelerate.valueChanged.connect(self.SetYAcc)
        self.ui.ZAccelerate.valueChanged.connect(self.SetZAcc)
        
        self.ui.InitStageButton.clicked.connect(self.InitStages)
        self.ui.StageUninit.clicked.connect(self.Uninit)
        # self.ui.SliceDir.clicked.connect(self.SliceDirection)
        # self.ui.VibEnabled.clicked.connect(self.Vibratome)
        self.ui.SliceN.valueChanged.connect(self._on_slice_n_changed)
        
        # testing buttons
        self.ui.TestButten1.clicked.connect(self.StartCoordinateCalibration)
        self.ui.TestButten2.clicked.connect(self.TestButton2Func)
        self.ui.TestButten3.clicked.connect(self.TestButton3Func)

        # UI bridge (must live on GUI thread)
        self._ui_bridge = UiBridge()
        self._ui_bridge.status_message.connect(self._on_status_message)
        self._ui_bridge.acquisition_controls_locked.connect(self._on_acquisition_controls_locked)
        self._ui_bridge.cu_slice_value.connect(self._on_cu_slice_value)
        self._ui_bridge.time_reader_value.connect(self._on_time_reader_value)
        self._ui_bridge.aline_ready.connect(self._on_aline_ready)
        self._ui_bridge.bline_ready.connect(self._on_bline_ready)
        self._ui_bridge.cscan_ready.connect(self._on_cscan_ready)
        self._ui_bridge.mosaic_ready.connect(self._on_mosaic_ready)
        self._ui_bridge.aodo_waveform_ready.connect(self._on_aodo_waveform_ready)
        self._on_slice_n_changed(self.ui.SliceN.value())
        self._last_display_payloads = {
            "aline": None,
            "bline": None,
            "cscan": None,
            "mosaic": None,
        }
        # Last C-scan volume kept for live YBar XZ re-slicing.
        self._last_cscan_render_payload = None
        self._acquisition_lock_depth = 0
        self._locked_widget_states = {}
        
        # Init all threads
        self.Init_allThreads()
        # Simple FPS limiter for rendering-heavy slots
        self._render_fps_limit = 30.0
        self._last_render_t = {"aline": 0.0, "bline": 0.0, "cscan": 0.0, "mosaic": 0.0}

    def Init_allThreads(self):
        self.Weaver_thread = WeaverThread_2(self.ui, self.log)
        self.AODO_thread = AODOThread_2(self.ui, self.log)
        self.DnS_thread = DnSThread_2(self.ui, self.log)
        self.GPU_thread = GPUThread_2(self.ui, self.log)
        self.D_thread = Camera_2(self.ui, self.log)
        self.file_naming = FileNaming(self.ui)

        # Inject the GUI-thread bridge into worker threads.
        self.Weaver_thread.ui_bridge = self._ui_bridge
        self.AODO_thread.ui_bridge = self._ui_bridge
        self.DnS_thread.ui_bridge = self._ui_bridge
        self.GPU_thread.ui_bridge = self._ui_bridge
        self.D_thread.ui_bridge = self._ui_bridge
        self.Weaver_thread.file_naming = self.file_naming
        self.Weaver_thread.gpu_thread = self.GPU_thread
        self.Weaver_thread.dns_thread = self.DnS_thread
        
        self.D_thread.start()
        self.GPU_thread.start()
        self.Weaver_thread.start()
        self.AODO_thread.start()
        self.DnS_thread.start()

    def _fps_ok(self, key: str) -> bool:
        now = time.monotonic()
        min_dt = 1.0 / max(self._render_fps_limit, 1.0)
        if now - self._last_render_t.get(key, 0.0) < min_dt:
            return False
        self._last_render_t[key] = now
        return True

    def _on_status_message(self, msg: str):
        self.ui.statusbar.showMessage(msg)

    def _on_slice_n_changed(self, value: int):
        if hasattr(self.ui, "CuSlice"):
            self.ui.CuSlice.setValue(int(value))

    def _on_cu_slice_value(self, value: int):
        if hasattr(self.ui, "CuSlice"):
            self.ui.CuSlice.setValue(int(value))

    def _on_time_reader_value(self, value: int):
        if hasattr(self.ui, "timeReader"):
            self.ui.timeReader.setValue(int(value))

    def _wait_stageback(self, label, timeout=300.0):
        timeout = float(timeout)
        try:
            StagebackQueue.get(timeout=timeout)
        except Exception:
            message = (
                f"Stage timeout while waiting for {label} acknowledgement "
                f"after {timeout:.1f}s. "
                "Assuming motion completed and continuing."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            return False
        return True

    def stage_move_timeout(self, axis):
        return 300.0

    def _add_sample_center_controls(self):
        # GotoSampleCenter is declared in GUI.ui (TestTab). It is always enabled so
        # the user can jump to the center of the sample selected in the main
        # top-bar sampleSelector.
        button = getattr(self.ui, "GotoSampleCenter", None)
        if button is None:
            return
        button.setEnabled(True)
        button.clicked.connect(self.GotoSelectedSampleCenter)

    def refresh_sample_center_selector(self):
        # Legacy call sites only. The Testing-popup SampleCenterSelector no longer
        # exists, so this simply re-asserts that the center-stage button stays
        # enabled.
        button = getattr(self.ui, "GotoSampleCenter", None)
        if button is not None:
            button.setEnabled(True)

    def _sample_center_by_id(self, sample_id):
        for center in getattr(self, "sample_centers", []) or []:
            if int(center.sample_id) == int(sample_id):
                return center
        return None

    def _managed_acquisition_widgets(self):
        widget_types = (
            QW.QAbstractButton,
            QW.QAbstractSpinBox,
            QW.QComboBox,
            QW.QLineEdit,
            QW.QTextEdit,
            QW.QPlainTextEdit,
            QW.QAbstractSlider,
        )
        return [widget for widget in self.findChildren(QW.QWidget) if isinstance(widget, widget_types)]

    def _live_acquisition_widgets(self):
        live_widgets = set()
        live_names = (
            "RunButton",
            "PauseButton",
            "Xmove2",
            "Ymove2",
            "Zmove2",
            "XUP",
            "YUP",
            "ZUP",
            "XDOWN",
            "YDOWN",
            "ZDOWN",
            "XHome",
            "YHome",
            "ZHome",
            "XPosition",
            "YPosition",
            "ZPosition",
            "XStepSize",
            "Xstagestepsize",
            "Ystagestepsize",
            "Zstagestepsize",
            "XSpeed",
            "YSpeed",
            "ZSpeed",
            "XAccelerate",
            "YAccelerate",
            "ZAccelerate",
        )
        for name in live_names:
            widget = getattr(self.ui, name, None)
            if isinstance(widget, QW.QWidget):
                live_widgets.add(widget)
        for widget in self._managed_acquisition_widgets():
            if isinstance(widget, QW.QAbstractSlider):
                live_widgets.add(widget)
        return live_widgets

    @staticmethod
    def _is_descendant_of(widget, ancestors):
        parent = widget.parentWidget()
        while parent is not None:
            if parent in ancestors:
                return True
            parent = parent.parentWidget()
        return False

    @staticmethod
    def _widget_and_descendants(widget):
        yield widget
        for child in widget.findChildren(QW.QWidget):
            yield child

    def _restore_editor_children(self, widget, prior_states):
        if isinstance(widget, QW.QAbstractSpinBox):
            editor = widget.findChild(QW.QLineEdit)
            if editor is not None:
                editor.setEnabled(prior_states.get(editor, True))
        elif isinstance(widget, QW.QComboBox):
            editor = widget.lineEdit()
            if editor is not None:
                editor.setEnabled(prior_states.get(editor, True))

    # ------------------------------------------------------------------ shading
    def set_shading_correction(self, enabled=True):
        """Turn the live flat/dark shading correction on/off for the next run.

        The switch is run level (config.ini ``ShadingCorrection``): it is read when a
        mosaic/plate scan starts.  Tiles already written are never re-corrected.
        """
        self.ui.shading_correction = bool(enabled)
        message = "阴影校正: {0}".format("已开启" if enabled else "已关闭")
        print(message)
        self.log.append(message)
        return self.ui.shading_correction

    def shading_correction_status(self):
        """One-line status of the live shading correction (UI / log helper)."""
        weaver = getattr(self, "Weaver_thread", None)
        if weaver is None or not hasattr(weaver, "shading_correction_status"):
            return "阴影校正: 未初始化"
        return weaver.shading_correction_status()

    def clear_shading_field(self):
        """Drop the fitted field: the next sample of the run becomes the reference."""
        weaver = getattr(self, "Weaver_thread", None)
        if weaver is None or not hasattr(weaver, "clear_shading_field"):
            return
        weaver.clear_shading_field()

    def set_acquisition_controls_locked(self, locked: bool):
        if locked:
            self._acquisition_lock_depth += 1
            if self._acquisition_lock_depth != 1:
                return
            live_widgets = self._live_acquisition_widgets()
            self._locked_widget_states = {
                widget: widget.isEnabled() for widget in self.findChildren(QW.QWidget)
            }
            for widget in self._managed_acquisition_widgets():
                if widget not in live_widgets and not self._is_descendant_of(widget, live_widgets):
                    for controlled_widget in self._widget_and_descendants(widget):
                        controlled_widget.setEnabled(False)
            return

        if self._acquisition_lock_depth == 0:
            return
        self._acquisition_lock_depth -= 1
        if self._acquisition_lock_depth != 0:
            return
        prior_states = self._locked_widget_states
        self._locked_widget_states = {}
        for widget, was_enabled in prior_states.items():
            try:
                widget.setEnabled(was_enabled)
            except RuntimeError:
                continue
        for widget, was_enabled in prior_states.items():
            if not was_enabled:
                continue
            try:
                self._restore_editor_children(widget, prior_states)
            except RuntimeError:
                continue

    def _on_acquisition_controls_locked(self, locked: bool):
        self.set_acquisition_controls_locked(locked)

    def enqueue_weaver_action(self, action):
        self.set_acquisition_controls_locked(True)
        WeaverQueue.put(action)

    def _on_aline_ready(self, payload: dict):
        self._last_display_payloads["aline"] = payload
        if not self._fps_ok("aline"):
            return
        render_aline_ready(self.ui, payload)

    def _on_bline_ready(self, payload: dict):
        self._last_display_payloads["bline"] = payload
        if not self._fps_ok("bline"):
            return
        render_bline_ready(self.ui, payload)

    def _sync_ybar_to_volume(self, volume):
        """Clamp/sync the YBar range to the number of Y rows in a C-scan volume."""
        if volume is None or np.size(volume) == 0:
            return
        if not hasattr(self.ui, "YBar"):
            return
        y_pixels = int(np.asarray(volume).shape[0])
        bar = self.ui.YBar
        if bar.maximum() != y_pixels - 1:
            bar.setMaximum(y_pixels - 1)
            if bar.value() == 0:
                bar.setValue(max(0, (y_pixels - 1) // 2))

    def _on_cscan_ready(self, payload: dict):
        self._last_display_payloads["cscan"] = payload
        mode = payload.get("mode")
        if mode == AcqTypes.TD_ENFACE:
            # TD-enface shows only the spectral-mean en-face on XYplaneInt;
            # all other OCT display windows stay cleared.
            if self._fps_ok("cscan"):
                render_td_enface(self.ui, payload)
            return
        # Keep the last volume so YBar can re-slice the XZ view immediately.
        self._last_cscan_render_payload = {
            "volume": payload.get("volume"),
            "hsv_volume": payload.get("hsv_volume"),
        }
        self._sync_ybar_to_volume(payload.get("volume", None))
        if not self._fps_ok("cscan"):
            return
        render_cscan_ready(self.ui, payload)

    def _on_ybar_changed(self, *_args):
        payload = getattr(self, "_last_cscan_render_payload", None)
        if payload is None:
            return
        try:
            render_cscan_xz_from_volume(self.ui, payload)
            render_cscan_xz_dual(self.ui, payload)
        except Exception as error:
            print(f"YBar re-render failed: {error}")

    def _sync_mosaic_ybar_to_volume(self, volume):
        """Clamp/sync the MosaicYbar range to the rows of the stitched mosaic volume."""
        if volume is None or np.size(volume) == 0:
            return
        if not hasattr(self.ui, "MosaicYbar"):
            return
        y_pixels = int(np.asarray(volume).shape[0])
        bar = self.ui.MosaicYbar
        if bar.maximum() != y_pixels - 1:
            bar.setMaximum(y_pixels - 1)
            if bar.value() == 0:
                bar.setValue(max(0, (y_pixels - 1) // 2))

    def _on_mosaic_ybar_changed(self, *_args):
        payload = self._last_display_payloads.get("mosaic")
        if payload is None:
            return
        try:
            render_mosaic_xz_slices(self.ui, payload)
        except Exception as error:
            print(f"MosaicYbar re-render failed: {error}")

    def _on_mosaic_ready(self, payload: dict):
        self._last_display_payloads["mosaic"] = payload
        # Mosaic/WellScan FOVs arrive through the mosaic path (no cscan_ready
        # volume), so cache the per-FOV tile volume to let YBar re-slice the XZ
        # B-scan plane.
        tile_volume = payload.get("tile_volume", None)
        if tile_volume is not None and np.size(tile_volume) > 0:
            self._last_cscan_render_payload = {
                "volume": tile_volume,
                "hsv_volume": payload.get("tile_hsv_volume", None),
            }
            self._sync_ybar_to_volume(tile_volume)
        self._sync_mosaic_ybar_to_volume(payload.get("mosaic_volume", None))
        if not self._fps_ok("mosaic"):
            return
        render_mosaic_ready(self.ui, payload)

    def _on_aodo_waveform_ready(self, payload: dict):
        render_aodo_waveform_ready(self.ui, payload)
            
    def Stop_allThreads(self):
        self.ui.RunButton.setChecked(False)
        self.ui.RunButton.setText('Go')
        self.ui.PauseButton.setChecked(False)
        self.ui.PauseButton.setText('Pause')

        # Ask hardware tasks to stop before their worker thread receives exit.
        AODOQueue.put(AODOActionField('tryStopTask'))
        AODOQueue.put(AODOActionField('CloseTask'))

        exit_element = EXITField()
        WeaverQueue.put(exit_element)
        AODOQueue.put(exit_element)
        DnSQueue.put(exit_element)
        GPUQueue.put(exit_element)
        DQueue.put(exit_element)

    def _wait_for_threads_to_finish(self, timeout_ms=5000):
        deadline = time.time() + timeout_ms / 1000.0
        threads = [
            ("Weaver", self.Weaver_thread),
            ("AODO", self.AODO_thread),
            ("DnS", self.DnS_thread),
            ("GPU", self.GPU_thread),
            ("Camera", self.D_thread),
        ]
        unfinished = []
        for name, thread in threads:
            remaining_ms = max(0, int((deadline - time.time()) * 1000))
            if not thread.wait(remaining_ms):
                unfinished.append(name)
        return unfinished
        
    def run_task(self):
        acq_mode = self.ui.ACQMode.currentText()
        if (
            acq_mode == AcqTypes.PLATE_PRESCAN
            and getattr(self.ui, "_prescan_awaiting_next", False)
        ):
            # The running PlatePreScan has finished a sample and relabelled the
            # Run button "下一个样品". This click just advances to the next
            # sample - the prescan loop consumes it, so do NOT enqueue another
            # PlatePreScan action here.
            return
        if acq_mode in CONTINUOUS_ACQ_MODES + LIVE_ONLY_MODES:
            if self.ui.RunButton.isChecked():
                self.ui.RunButton.setText('Stop')
                an_action = WeaverActionField(acq_mode, acq_mode=acq_mode)
                self.enqueue_weaver_action(an_action)
            else:
                self.Stop_task()
        elif acq_mode in FINITE_ACQ_MODES:
            if self.ui.RunButton.isChecked():
                self.ui.RunButton.setText('Stop')
                # self.ui.RunButton.setEnabled(False)
                # self.ui.PauseButton.setEnabled(False)
                an_action = WeaverActionField(acq_mode, acq_mode=acq_mode)
                self.enqueue_weaver_action(an_action)
        
    def capture_usb_locator_tile(self, tile_position, stage_z):
        tile_dir = os.path.join(self.ui.DIR.toPlainText(), "Mosaic", "usb_locator_regions")
        os.makedirs(tile_dir, exist_ok=True)
        tile_index = int(tile_position["tile_index"])
        row_idx = int(tile_position["row"])
        col_idx = int(tile_position["col"])
        x_pos = float(tile_position["stage_x"])
        y_pos = float(tile_position["stage_y"])

        print(
            "USB mosaic locator region capture: "
            f"region={tile_index}, row={row_idx}, col={col_idx}, "
            f"X={x_pos:.4f}, Y={y_pos:.4f}"
        )
        self.ui.XPosition.setValue(x_pos)
        self.Xmove2()
        self.ui.YPosition.setValue(y_pos)
        self.Ymove2()

        # Capture and average a few USB frames so the locator image (that the
        # user draws ROIs on) is cleaner.
        frame = capture_usb_frame_averaged()
        if frame is None:
            message = f"USB camera returned no image at locator region {tile_index}; using blank frame."
            print(message)
            self.ui.statusbar.showMessage(message)
            frame = blank_usb_frame()

        image_path = os.path.join(tile_dir, f"usb_region-{tile_index:02d}-row{row_idx}-col{col_idx}.png")
        cv2.imwrite(image_path, frame)
        print(
            "USB locator saved frame: "
            f"path={image_path}, shape={frame.shape}, "
            "coordinate_space=Fiji horizontal-flip calibration"
        )
        return {
            "tile_index": tile_index,
            "row": row_idx,
            "col": col_idx,
            "stage_x": x_pos,
            "stage_y": y_pos,
            "stage_z": float(stage_z),
            "image": frame,
            "image_path": image_path,
            "single_frame": bool(tile_position.get("single_frame", False)),
        }

    def save_usb_locator_run_records(self, tile_records, roi_records):
        folder = os.path.join(self.ui.DIR.toPlainText(), "Mosaic")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "usb_mosaic_locator_run.json")
        data = {
            "mode": "single_frame",
            "grid_x": 1,
            "grid_y": 1,
            "x_range_mm": (
                [float(tile_records[0]["stage_x"]), float(tile_records[0]["stage_x"])]
                if tile_records
                else []
            ),
            "y_range_mm": (
                [float(tile_records[0]["stage_y"]), float(tile_records[0]["stage_y"])]
                if tile_records
                else []
            ),
            "tile_records": [
                {
                    "tile_index": int(tile["tile_index"]),
                    "row": int(tile["row"]),
                    "col": int(tile["col"]),
                    "stage_x": float(tile["stage_x"]),
                    "stage_y": float(tile["stage_y"]),
                    "stage_z": float(tile["stage_z"]),
                    "image_path": tile.get("image_path", ""),
                    "single_frame": bool(tile.get("single_frame", False)),
                }
                for tile in tile_records
            ],
            "rois": roi_records,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        print(f"USB mosaic locator run records saved: {path}")

    def build_usb_region_overlay_sources(self, tile_records, roi_records, calibration):
        tile_lookup = {}
        for local_index, tile in enumerate(tile_records, start=1):
            tile_lookup[local_index] = tile
            tile_lookup[int(tile["tile_index"])] = tile
        overlay_images = {}
        for roi in roi_records:
            sample_id = int(roi["sample_id"])
            tile_index = int(roi["tile_index"])
            if tile_index not in tile_lookup:
                raise ValueError(f"Missing USB locator region record for tile_index={tile_index}")
            tile = tile_lookup[tile_index]
            overlay_images[sample_id] = {
                "type": "usb_region",
                "sample_id": sample_id,
                "image_path": tile.get("image_path", ""),
                "tile_index": tile_index,
                "tile_row": int(tile["row"]),
                "tile_col": int(tile["col"]),
                "tile_stage_x": float(tile["stage_x"]),
                "tile_stage_y": float(tile["stage_y"]),
                "tile_stage_z": float(tile["stage_z"]),
                "pixel_polygon": roi["pixel_polygon"],
                "fiji_pixel_polygon": roi.get("fiji_pixel_polygon", []),
                "stage_polygon": roi["stage_polygon"],
                "calibration": calibration,
            }
        return overlay_images

    def save_usb_roi_overview_image(self, tile_records, roi_records, filename="usb_calibration_overview.png"):
        """Save one combined overlay image of the USB region(s) + user-drawn ROIs.

        For a single-frame locator run this is the full-resolution USB frame
        with every ROI drawn on it. For a multi-region grid run it is a montage
        of all tiles (grid order) with each tile's ROIs drawn on top. This gives
        the operator a persistent map of where every ROI / sample was drawn.
        """
        folder = os.path.join(self.ui.DIR.toPlainText(), "Mosaic")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        if not tile_records or not roi_records:
            return None
        try:
            tile_lookup = {}
            for tile in tile_records:
                tile_lookup[int(tile["tile_index"])] = tile
            rois_by_tile = {}
            for roi in roi_records:
                tile_index = int(roi.get("tile_index", 0))
                if tile_index not in tile_lookup:
                    continue
                rois_by_tile.setdefault(tile_index, []).append(roi)

            def draw_rois(image, tile_index, offset_x=0, offset_y=0, scale=1.0):
                for roi in rois_by_tile.get(int(tile_index), []):
                    polygon = roi.get("pixel_polygon") or []
                    if len(polygon) < 3:
                        continue
                    points = []
                    for pt in polygon:
                        points.append(
                            (
                                offset_x + float(pt[0]) * scale,
                                offset_y + float(pt[1]) * scale,
                            )
                        )
                    points = np.asarray(points, dtype=np.float32)
                    points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
                    thickness = max(2, int(round(6 * scale)))
                    cv2.polylines(image, [points], True, (0, 255, 0), thickness)
                    sample_id = int(roi.get("sample_id", 0))
                    cx = int(np.mean(points[:, 0, 0]))
                    cy = int(np.mean(points[:, 0, 1]))
                    font_scale = max(0.4, 0.8 * scale)
                    cv2.putText(
                        image,
                        str(sample_id),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        max(1, int(round(2 * scale))),
                        cv2.LINE_AA,
                    )

            rows = sorted({int(tile["row"]) for tile in tile_records})
            cols = sorted({int(tile["col"]) for tile in tile_records})
            single_frame = len(tile_records) == 1

            if single_frame:
                tile = tile_records[0]
                image = cv2.imread(tile.get("image_path", ""), cv2.IMREAD_COLOR)
                if image is None:
                    return None
                draw_rois(image, int(tile["tile_index"]))
            else:
                first = tile_records[0]
                base_image = cv2.imread(first.get("image_path", ""), cv2.IMREAD_COLOR)
                if base_image is None:
                    return None
                base_h, base_w = base_image.shape[:2]
                cell_w = 560
                cell_h = max(1, int(round(cell_w * base_h / max(1, base_w))))
                gap = 6
                scale = cell_w / float(base_w)
                canvas = np.full(
                    (len(rows) * (cell_h + gap) - gap, len(cols) * (cell_w + gap) - gap, 3),
                    20,
                    dtype=np.uint8,
                )
                row_to_index = {row: idx for idx, row in enumerate(rows)}
                col_to_index = {col: idx for idx, col in enumerate(cols)}
                for tile in tile_records:
                    image = cv2.imread(tile.get("image_path", ""), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    row_idx = row_to_index[int(tile["row"])]
                    col_idx = col_to_index[int(tile["col"])]
                    y0 = row_idx * (cell_h + gap)
                    x0 = col_idx * (cell_w + gap)
                    canvas[y0:y0 + cell_h, x0:x0 + cell_w] = image
                    draw_rois(canvas, int(tile["tile_index"]), x0, y0, scale)
                image = canvas

            saved = cv2.imwrite(path, image)
            if saved:
                print(
                    f"USB locator overlay overview saved: {path} "
                    f"({image.shape[1]}x{image.shape[0]} px, {len(roi_records)} ROI(s))."
                )
                return path
            return None
        except Exception as error:
            print(f"Could not save USB locator overview image: {error}")
            return None

    def _mosaic_folder(self):
        return os.path.join(self.ui.DIR.toPlainText(), "Mosaic")

    def _apply_plan_scan_geometry(self, fov_locations):
        """Mirror a located/loaded plan's FOV sizes into the scan spin boxes.

        ``ThreadWeaver.iterate_FOVs`` / ``PlatePreScan`` apply the same geometry
        right before scanning.  Applying it here as well keeps ``ui.XLength`` /
        ``ui.YLength`` / ``ui.Ypixels`` consistent with the plan the moment it
        exists, so an operator started live C-scan (or the plate pre-scan Z
        adjustment) can never build an empty galvo waveform from a stale Y
        geometry inherited from config.ini.
        """
        weaver = getattr(self, "Weaver_thread", None)
        if weaver is None:
            return
        try:
            weaver.apply_scan_geometry_from_locations(fov_locations)
        except Exception as error:
            print(f"Could not apply the planned FOV geometry to the scan settings: {error}")

    def _load_sample_locations_from_current_folder(self):
        """Load a previously saved sample-location plan (DIR/Mosaic/scan_metadata.pkl)
        into memory and refresh the main sampleSelector.

        Returns True when a plan with at least one sample center was loaded.
        Returns False (so the caller runs the interactive locator) when no usable
        saved plan exists in the current save folder.
        """
        mosaic_folder = self._mosaic_folder()
        metadata_path = os.path.join(mosaic_folder, "scan_metadata.pkl")
        if not os.path.exists(metadata_path):
            return False
        try:
            fov_locations, sample_centers, overlay_images = load_session_data(mosaic_folder)
        except Exception as error:
            message = f"Could not load saved sample locations: {error}"
            print(message)
            self.ui.statusbar.showMessage(message)
            return False
        sample_centers = sorted(
            list(sample_centers),
            key=lambda center: int(center.sample_id),
        )
        if not sample_centers:
            return False

        self.FOV_locations = fov_locations
        self.sample_centers = sample_centers
        self.overlay_images = overlay_images

        # Keep the Weaver thread's plan in sync so that Go -> PlatePreScan /
        # WellScan / PlateScan immediately uses the freshly loaded locations.
        weaver = getattr(self, "Weaver_thread", None)
        if weaver is not None:
            weaver.FOV_locations = fov_locations
            weaver.sample_centers = sample_centers
            weaver.overlay_images = overlay_images

        # A plan loaded from disk carries the fixed Y FOV (0.5 mm), while
        # config.ini may still hold a degenerate one (YLength=0 -> empty galvo
        # waveform); mirror the plan into the scan spin boxes right away.
        self._apply_plan_scan_geometry(fov_locations)

        populate_sample_selector(self.ui, sample_centers)
        message = (
            f"Loaded {len(sample_centers)} saved sample location(s) "
            f"from {mosaic_folder}."
        )
        print(message)
        self.ui.statusbar.showMessage(message)
        return True

    def _current_usb_calibration(self):
        """Return the USB camera -> stage calibration for the next locator run.

        Uses the last fitted model stored in config.ini (see CoordinateCalibration
        save/load). There is no hard-coded fallback matrix any more; when config.ini
        has no fitted model the calibration matrix stays None so callers fail with a
        clear 'requires affine calibration' message instead of using stale numbers.
        """
        calibration = default_usb_mosaic_calibration()
        try:
            from CoordinateCalibration import load_affine_from_config
            matrix = load_affine_from_config("config.ini")
            if matrix is not None:
                calibration["camera_to_stage_affine"] = matrix
        except Exception as error:
            print(f"Could not load saved USB camera calibration: {error}")
        return calibration

    def StartCoordinateCalibration(self):
        if getattr(self.ui, "_coordinate_calibration_active", False):
            message = "Coordinate calibration is already running."
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        # calibration_mode=True always re-runs the USB locator from scratch and
        # then starts the per-sample alignment loop - it never loads saved plans.
        self.LocateSample(calibration_mode=True)

    def LocateSample(self, calibration_mode=False):
        # Unless this is a coordinate-calibration run, a previously saved sample
        # plan in the current folder is loaded instead of re-running the locator.
        if not calibration_mode and self._load_sample_locations_from_current_folder():
            return

        objective = get_objective_spec(self.ui.Objective.currentText())
        if objective is None:
            message = f"Unknown objective for sample locator: {self.ui.Objective.currentText()}"
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        default_sample_z = self.ui.ZPosition.value()
        locator_z = SAMPLE_LOCATOR_Z_MM
        if locator_z < self.ui.ZPosition.minimum() or locator_z > self.ui.ZPosition.maximum():
            message = (
                f"Sample locator requires Z={locator_z:.4f} mm, but ZPosition range is "
                f"[{self.ui.ZPosition.minimum():.4f}, {self.ui.ZPosition.maximum():.4f}] mm."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        print(
            "Sample locator moving Z to calibration height: "
            f"current={default_sample_z:.4f}, locator_z={locator_z:.4f}"
        )
        if not self.ZeroZForSampleLocator():
            return
        print("Sample locator homing X/Y before single-frame capture.")
        self.XHome()
        self.YHome()

        tile_position = {
            "tile_index": 1,
            "row": 0,
            "col": 0,
            "stage_x": 0.0,
            "stage_y": 0.0,
            "single_frame": True,
        }
        tile_record = self.capture_usb_locator_tile(tile_position, locator_z)
        all_tile_records = [tile_record]
        self.scanner = MosaicUSBSampleScanner(
            all_tile_records,
            self.ui.DIR.toPlainText(),
            fov_w_mm=self.ui.XLength.value(),
            fov_h_mm=self.ui.YLength.value(),
            current_zpos=default_sample_z,
            y_step_um=self.ui.YStepSize.value(),
            max_y_fov_mm=objective.max_y_fov_mm,
            stage_bounds=(
                self.ui.Xmin.value(),
                self.ui.Xmax.value(),
                self.ui.Ymin.value(),
                self.ui.Ymax.value(),
            ),
            sample_id_start=1,
            allow_empty=True,
            initial_tile_index=0,
            initial_calibration=self._current_usb_calibration(),
        )
        if not self.scanner.exec_():
            message = (
                f"Sample locator canceled. Leaving Z at locator height {locator_z:.4f} mm; "
                "move X/Y to a safe position before raising Z."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            self.ui.sampleSelector.clear()
            self.ui.sampleSelector.addItem("No Samples Found")
            self.FOV_locations = []
            self.sample_centers = []
            self.refresh_sample_center_selector()
            return
        all_fov_locations = list(self.scanner.generated_locations)
        all_sample_centers = list(self.scanner.sample_centers)
        all_pixel_polygons = list(self.scanner.final_polygons)
        all_roi_records = list(getattr(self.scanner, "final_tile_roi_records", []))

        self.save_usb_locator_run_records(all_tile_records, all_roi_records)

        if len(all_sample_centers) == 0:
            message = (
                f"No samples selected. Leaving Z at locator height {locator_z:.4f} mm; "
                "move X/Y to a safe position before raising Z."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            self.ui.sampleSelector.clear()
            self.ui.sampleSelector.addItem("No Samples Found")
            self.FOV_locations = []
            self.sample_centers = []
            self.refresh_sample_center_selector()
            return

        if calibration_mode and len(all_sample_centers) < 5:
            message = (
                "Coordinate calibration requires at least 5 drawn samples; "
                f"only {len(all_sample_centers)} were drawn. "
                "Run it again and draw at least 5 ROIs."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            self.ui.sampleSelector.clear()
            self.ui.sampleSelector.addItem("No Samples Found")
            return

        # Successful locator run: export USB frames + drawn ROIs for AI training.
        try:
            export_sample_locator_dataset(
                all_tile_records,
                all_roi_records,
                calibration=self._current_usb_calibration(),
            )
        except Exception as error:
            print(f"Could not export USB training data: {error}")

        FOV_locations = all_fov_locations
        sample_centers = all_sample_centers
        raw_img = None
        pixel_polygons = all_pixel_polygons
        overlay_images = self.build_usb_region_overlay_sources(all_tile_records, all_roi_records, self._current_usb_calibration())
        self.FOV_locations = FOV_locations
        self.sample_centers = sample_centers
        self.raw_img = raw_img
        self.pixel_polygons = pixel_polygons
        self.overlay_images = overlay_images
        # Mirror the fresh plan into the scan spin boxes now: the operator can
        # start a live C-scan (or Go -> PlatePreScan) before the first FOV loop
        # applies the geometry, and a stale YLength=0 in config.ini would then
        # build an empty galvo waveform.
        self._apply_plan_scan_geometry(FOV_locations)
        # Persist the located plan so the same folder can be reloaded later by
        # SampleLocateButton without re-running the interactive locator.
        try:
            save_session_data(
                self._mosaic_folder(),
                FOV_locations,
                sample_centers,
                overlay_images,
                raw_img=None,
                pixel_polygons=pixel_polygons,
            )
        except Exception as error:
            print(f"Could not persist located sample plan: {error}")
        print("Sample locator center positions:")
        for center in sample_centers:
            print(
                f"sampleID-{center.sample_id}: "
                f"X={center.x:.4f}, Y={center.y:.4f}, Z={center.z:.4f}"
            )
        """Updates the combo box content on the main thread."""
        self.ui.sampleSelector.clear()
        if len(sample_centers) == 0:
            self.ui.sampleSelector.addItem("No Samples Found")
            return
        else:
            for i in range(len(sample_centers)):
                self.ui.sampleSelector.addItem(f"Sample {i+1}")
        self.refresh_sample_center_selector()
        # print(self.sample_centers)
        # print(FOV_locations)
        # print(sample_centers)
        first_center = sample_centers[0]
        print(
            "Sample locator complete. Moving X/Y to first sample center before raising Z: "
            f"sampleID-{first_center.sample_id}, X={first_center.x:.4f}, Y={first_center.y:.4f}"
        )
        self.ui.XPosition.setValue(first_center.x)
        self.Xmove2()
        self.ui.YPosition.setValue(first_center.y)
        self.Ymove2()
        print(f"Restoring Z to {default_sample_z:.4f} mm before pre-scan.")
        self.ui.ZPosition.setValue(default_sample_z)
        self.Zmove2()
        if calibration_mode:
            # Coordinate calibration: always uses the freshly drawn ROI pixel
            # polygons. The action is put directly on the WeaverQueue (no
            # acquisition-lock), because the operator must keep using the stage
            # controls and the Stop button during the alignment loop.
            setattr(self.ui, "_coordinate_calibration_active", True)
            overview_path = self.save_usb_roi_overview_image(all_tile_records, all_roi_records)
            an_action = WeaverActionField(
                AcqTypes.COORDINATE_CALIBRATION,
                acq_mode=AcqTypes.COORDINATE_CALIBRATION,
                context=[sample_centers, all_roi_records, FOV_locations, overlay_images],
            )
            WeaverQueue.put(an_action)
            message = (
                f"Coordinate calibration started for {len(sample_centers)} samples. "
                "At each sample, adjust X/Y/Z until it matches your drawing, "
                "then press Stop."
            )
            if overview_path:
                message += f" ROI overlay overview saved: {overview_path}"
            print(message)
            self.ui.statusbar.showMessage(message)
        else:
            an_action = WeaverActionField(
                AcqTypes.PLATE_PRESCAN,
                acq_mode=AcqTypes.PLATE_PRESCAN,
                context=[FOV_locations, sample_centers, raw_img, pixel_polygons, overlay_images],
            )
            self.enqueue_weaver_action(an_action)

    def InitStages(self):
        an_action = AODOActionField('Init')
        AODOQueue.put(an_action)
        self._wait_stageback("stage init")
        
        
    def Uninit(self):
        an_action = AODOActionField('Uninit')
        AODOQueue.put(an_action)
        self._wait_stageback("stage uninit")
        
    def Xmove2(self):
        an_action = AODOActionField('Xmove2')
        AODOQueue.put(an_action)
        self._wait_stageback("X move", timeout=self.stage_move_timeout('X'))
        
    def Ymove2(self):
        an_action = AODOActionField('Ymove2')
        AODOQueue.put(an_action)
        self._wait_stageback("Y move", timeout=self.stage_move_timeout('Y'))
        
    def Zmove2(self):
        an_action = AODOActionField('Zmove2')
        AODOQueue.put(an_action)
        self._wait_stageback("Z move", timeout=self.stage_move_timeout('Z'))

    def ZeroZForSampleLocator(self):
        locator_z = SAMPLE_LOCATOR_Z_MM
        if locator_z < self.ui.ZPosition.minimum() or locator_z > self.ui.ZPosition.maximum():
            message = (
                "Sample locator Z safety move requires "
                f"final_z={locator_z:.4f} mm, "
                f"but ZPosition range is [{self.ui.ZPosition.minimum():.4f}, "
                f"{self.ui.ZPosition.maximum():.4f}] mm."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            return False
        self.ui.ZPosition.setValue(locator_z)
        self.Zmove2()
        return True
        
    def XUP(self):
        an_action = AODOActionField('XUP')
        AODOQueue.put(an_action)
        self._wait_stageback("X step up")
    def YUP(self):
        an_action = AODOActionField('YUP')
        AODOQueue.put(an_action)
        self._wait_stageback("Y step up")
    def ZUP(self):
        an_action = AODOActionField('ZUP')
        AODOQueue.put(an_action)
        self._wait_stageback("Z step up")
        
    def XDOWN(self):
        an_action = AODOActionField('XDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("X step down")
    def YDOWN(self):
        an_action = AODOActionField('YDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("Y step down")
    def ZDOWN(self):
        an_action = AODOActionField('ZDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("Z step down")
        
    def XHome(self):
        an_action = AODOActionField('XHome')
        AODOQueue.put(an_action)
        self._wait_stageback("X home")
    def YHome(self):
        an_action = AODOActionField('YHome')
        AODOQueue.put(an_action)
        self._wait_stageback("Y home")
    def ZHome(self):
        an_action = AODOActionField('ZHome')
        AODOQueue.put(an_action)
        self._wait_stageback("Z home")
        
    def SetXSpeed(self):
        an_action = AODOActionField('XSpeed')
        AODOQueue.put(an_action)
        
    def SetYSpeed(self):
        an_action = AODOActionField('YSpeed')
        AODOQueue.put(an_action)
        
    def SetZSpeed(self):
        an_action = AODOActionField('ZSpeed')
        AODOQueue.put(an_action)
        
    def SetXAcc(self):
        an_action = AODOActionField('XAcc')
        AODOQueue.put(an_action)
        
    def SetYAcc(self):
        an_action = AODOActionField('YAcc')
        AODOQueue.put(an_action)
        
    def SetZAcc(self):
        an_action = AODOActionField('ZAcc')
        AODOQueue.put(an_action)
        
    def Vibratome(self):
        if self.ui.VibEnabled.isChecked():
            self.ui.VibEnabled.setText('Stop Vibratome')
            an_action = AODOActionField('startVibratome')
            AODOQueue.put(an_action)
            self._wait_stageback("servo out")
        else:
            self.ui.VibEnabled.setText('Start Vibratome')
            an_action = AODOActionField('stopVibratome')
            AODOQueue.put(an_action)
            self._wait_stageback("servo back")
        
    def SliceDirection(self):
        if self.ui.SliceDir.isChecked():
            self.ui.SliceDir.setText('Forward')
        else:
            self.ui.SliceDir.setText('Backward')
            
    def RepTest(self):
        if self.ui.ZstageTest.isChecked():
            an_action = WeaverActionField(WeaverActions.ZSTAGE_REPEATIBILITY, acq_mode=self.ui.ACQMode.currentText())
            self.enqueue_weaver_action(an_action)
        # wait until weaver done
        
    def Gotozero(self):
        if self.ui.Gotozero.isChecked():
            an_action = WeaverActionField(WeaverActions.GOTO_ZERO, acq_mode=self.ui.ACQMode.currentText())
            self.enqueue_weaver_action(an_action)

        
    def _schedule_ygalvo_voltage(self, *_args):
        self._ygalvo_timer.start()

    def _apply_ygalvo_voltage(self):
        AODOQueue.put(AODOActionField('setYGalvo'))

    def CenterGalvo(self):
        an_action = AODOActionField('centergalvo')
        AODOQueue.put(an_action)

    def GotoSelectedSampleCenter(self):
        centers = sorted(
            list(getattr(self, "sample_centers", []) or []),
            key=lambda center: int(center.sample_id),
        )
        if not centers:
            message = "No sample locations are available. Use 定位样品 to locate or load samples first."
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        selected_index = self.ui.sampleSelector.currentIndex()
        if selected_index < 0 or self.ui.sampleSelector.currentText() == "No Samples Found":
            message = "No sample is selected. Locate or load sample locations first."
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        sample_id = selected_index + 1
        center = self._sample_center_by_id(sample_id)
        if center is None:
            message = f"sampleID-{int(sample_id)} center was not found in memory."
            print(message)
            self.ui.statusbar.showMessage(message)
            return
        print(
            "Moving stage to selected sample center: "
            f"sampleID-{int(sample_id)}, X={center.x:.4f}, Y={center.y:.4f}"
        )
        self.ui.statusbar.showMessage(f"Moving to sampleID-{int(sample_id)} center.")
        self.ui.XPosition.setValue(center.x)
        self.Xmove2()
        self.ui.YPosition.setValue(center.y)
        self.Ymove2()
        
    def Pause_task(self):
        if self.ui.PauseButton.isChecked():
            self.ui.PauseButton.setText('Resume')
            self.ui.statusbar.showMessage('acquisition paused...')
            print('acquisition paused...')
        else:
            self.ui.PauseButton.setText('Pause')
            self.ui.statusbar.showMessage('acquisition resumed...')
            print('acquisition resumed...')
      
    def Stop_task(self):
        self.ui.statusbar.showMessage('acquisition stopped...')
        print('acquisition stopped...')
        
    def update_Dispersion(self):
        an_action = GPUActionField(GPUActions.UPDATE_DISPERSION)
        GPUQueue.put(an_action)
        # self.update_background()
        
    def update_background(self):
        an_action = GPUActionField(GPUActions.UPDATE_BACKGROUND)
        GPUQueue.put(an_action)
        
    def Update_contrast(self):
        acq_mode = self.ui.ACQMode.currentText()
        if acq_mode == AcqTypes.TD_ENFACE:
            payload = self._last_display_payloads.get("cscan")
            if payload is not None:
                render_td_enface(self.ui, payload)
        elif acq_mode in ["FiniteAline", "ContinuousAline"]:
            payload = self._last_display_payloads.get("aline")
            if payload is not None:
                render_aline_ready(self.ui, payload)
        elif acq_mode in ["FiniteBline", "ContinuousBline"]:
            payload = self._last_display_payloads.get("bline")
            if payload is not None:
                render_bline_ready(self.ui, payload)
        elif acq_mode in ["FiniteCscan", "ContinuousCscan", "FastVolumeCscan"]:
            payload = self._last_display_payloads.get("cscan")
            if payload is not None:
                render_cscan_ready(self.ui, payload)
        elif acq_mode in MOSAIC_DISPLAY_MODES:
            payload = self._last_display_payloads.get("mosaic")
            if payload is not None:
                render_mosaic_ready(self.ui, payload)

    def Update_contrast_Mosaic(self):
        self.Update_contrast()
            
    def Update_contrast_Dyn(self):
        self.Update_contrast()
        
    # def redo_dispersion_compensation(self):
    #     an_action = WeaverActionField('dispersion_compensation')
    #     WeaverQueue.put(an_action)
        
    def redo_background(self):
        an_action = WeaverActionField(
            WeaverActions.GET_BACKGROUND,
            acq_mode=AcqTypes.FINITE_BLINE,
        )
        self.enqueue_weaver_action(an_action)
        
    def redo_surface(self):
        an_action = WeaverActionField(
            WeaverActions.GET_SURFACE,
            acq_mode=AcqTypes.FINITE_BLINE,
        )
        self.enqueue_weaver_action(an_action)
        
    # def update_intDk(self):
    #     self.ui.intDk.setValue(self.ui.intDkSlider.value()/100)
    #     an_action = GPUActionField('update_intDk')
    #     GPUQueue.put(an_action)
        
    # def UninitBoard(self):
    #     an_action = DActionField('UninitBoard')
    #     DQueue.put(an_action)
    
    def TestButton1Func(self):
        self.StartCoordinateCalibration()
        
    def TestButton2Func(self):
        context = [[1, 1], [10, 100]]
        an_action = DnSActionField(DnSActions.MOSAIC, data = np.ones([300*1700,150],dtype=np.float32)*50, context = context)
        DnSQueue.put(an_action)
    
    def TestButton3Func(self):
        an_action = DnSActionField(DnSActions.DISPLAY_MOSAIC)
        DnSQueue.put(an_action)
        
    def closeEvent(self, event):
        print('Exiting all threads')
        self.ui.statusbar.showMessage('Closing: stopping acquisition and worker threads...')
        print('Closing: stopping acquisition and worker threads...')
        self.SaveSettings()
        self.Stop_allThreads()
        unfinished = self._wait_for_threads_to_finish(timeout_ms=5000)
        if unfinished:
            message = "Close delayed: waiting for " + ", ".join(unfinished) + " thread(s)."
            print(message)
            self.ui.statusbar.showMessage(message)
            event.ignore()
            return
        event.accept()

                

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = GUI()
    example.show()
    sys.exit(app.exec_())

    
