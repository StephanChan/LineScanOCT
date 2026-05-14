# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:41:46 2023

@author: admin
"""
# DO configure: port0 line 0 for X stage, port0 line 1 for Y stage, port 0 line 2 for Z stage, port 0 line 3 for Digitizer enable

# Generating Galvo X direction waveforms based on step size, Xsteps, Aline averages and objective
# StepSize in unit of um
# bias in unit of mm

import numpy as np
import os
import sys
import threading
from HardwareSpecs import get_objective_spec
from PyQt5.QtGui import QPixmap, QImage
from matplotlib import pyplot as plt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF

class _LogTeeStream:
    def __init__(self, logger, original_stream):
        self.logger = logger
        self.original_stream = original_stream

    def write(self, text):
        if self.original_stream is not None:
            self.original_stream.write(text)
        self.logger.write_stream(text)
        return len(text)

    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()


class LOG():
    def __init__(self, ui):
        super().__init__()
        import datetime
        current_time = datetime.datetime.now()
        self.ui = ui
        suffix = (
            str(current_time.year)+'-'+
            str(current_time.month)+'-'+
            str(current_time.day)+'-'+
            str(current_time.hour)+'-'+
            str(current_time.minute)+'-'+
            str(current_time.second)
        )
        self._suffix = suffix
        self._lock = threading.Lock()
        self._stdout_installed = False
        self._stdout = None
        self._stderr = None
        self._suppress_next_newline = False
        self._suppressed_stream_fragments = (
            "waiting for camera data...",
            "time to fetch data:",
        )

    def _current_log_dir(self):
        base_dir = os.getcwd()
        try:
            candidate_dir = self.ui.DIR.toPlainText().strip()
            if candidate_dir:
                base_dir = candidate_dir
        except Exception:
            pass
        log_dir = os.path.join(base_dir, 'log_files')
        return log_dir

    def _current_file_path(self):
        return os.path.join(self._current_log_dir(), 'log_' + self._suffix + '.txt')

    def _current_dynamic_file_path(self):
        return os.path.join(self._current_log_dir(), 'dynamic_log_' + self._suffix + '.txt')

    def _append_text(self, path, text):
        if text is None or text == "":
            return
        with self._lock:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fp = open(path, 'a', encoding='utf-8')
                fp.write(text)
                fp.close()
                return
            except OSError:
                fallback_dir = os.path.join(os.getcwd(), 'log_files')
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_path = os.path.join(fallback_dir, os.path.basename(path))
                fp = open(fallback_path, 'a', encoding='utf-8')
                fp.write(text)
                fp.close()

    def write(self, message):
        self._append_text(self._current_file_path(), str(message) + '\n')

    def dynamic_write(self, message):
        self._append_text(self._current_dynamic_file_path(), str(message) + '\n')

    def write_stream(self, text):
        if text is None:
            return
        text_str = str(text)
        if self._suppress_next_newline and text_str in ("\n", "\r\n", "\r"):
            self._suppress_next_newline = False
            return
        self._suppress_next_newline = False
        for fragment in self._suppressed_stream_fragments:
            if fragment in text_str:
                self._suppress_next_newline = True
                return
        self._append_text(self._current_file_path(), text_str)

    def install_stream_redirects(self):
        if self._stdout_installed:
            return
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _LogTeeStream(self, self._stdout)
        sys.stderr = _LogTeeStream(self, self._stderr)
        self._stdout_installed = True


def GenGalvoWave(StepSize = 1, Steps = 1000, AVG = 1, obj = '5X', postclocks = 50, Galvo_bias = 0):
    # total number of steps is the product of steps and aline average number
    # use different angle to mm ratio for different objective
    objective = get_objective_spec(obj)
    if objective is None:
        status = 'objective not calibrated, abort generating Galvo waveform'
        return None, status
    angle2mmratio = objective.angle_to_mm_ratio
    # X range is product of steps and step size
    Xrange = StepSize*Steps/1000
    # max voltage is converted from half of max X range plus bias divided by angle2mm ratio
    # extra division by 2 is because galvo angle change is only half of beam deviation angle
    Vmax = (Xrange/2)/angle2mmratio/2+Galvo_bias
    Vmin = (-Xrange/2)/angle2mmratio/2+Galvo_bias
    # fly-back time in unit of clocks
    steps2=postclocks
    # linear waveform: sweep from positive Y toward negative Y
    waveform=np.linspace(Vmax, Vmin, Steps)
    # Bline average
    waveform = np.tile(waveform,(AVG,1)).transpose().flatten()

    # print(len(waveform))
    # fly-back waveform returns smoothly to the next sweep start
    Postwave = (Vmin-Vmax)/2*np.cos(np.arange(0,np.pi,np.pi/steps2))+(Vmax+Vmin)/2
    # add prewave to avoid galvo big jump at beginning
    prewave = np.ones(50)*waveform[0]
    # append all waveforms together
    waveform = np.append(np.append(prewave, waveform), Postwave)

    status = 'waveform updated'
    return waveform, status


def GenAODO(mode='ContinuousBline',obj = '5X',postclocks = 50, YStepSize = 1, YSteps = 200, BVG = 1, Galvo_bias = 0):
    # BVG: Bline average
    # bias: Galvo bias voltage
    # postclocks: #Aline triggers for Galvo fly-back

    # DO clock is synchronuous with Galvo waveform
    # DO configure: port0 line 0
    if mode in ['ContinuousAline', 'FiniteAline', 'ContinuousBline', 'FiniteBline']:

        AOwaveform = np.ones(BVG*2) * Galvo_bias
        DOwaveform = np.ones([BVG, 2],dtype = np.uint32)
        DOwaveform[:,1] = 0
        DOwaveform = DOwaveform.flatten()
        status = 'waveform updated'
        return np.uint32(DOwaveform), AOwaveform, status

    elif mode in ['FiniteCscan','ContinuousCscan', 'PlateScan','PlatePreScan', 'WellScan','TimedPlateScan']:
        # generate AO waveform for Galvo control for one Bline
        AOwaveform, status = GenGalvoWave(YStepSize, YSteps, BVG*2, obj, postclocks, Galvo_bias)
        DOwaveform = np.ones([YSteps*BVG, 2],dtype = np.uint32)
        DOwaveform[:,1] = 0
        DOwaveform=DOwaveform.flatten()
        postDOwave = np.zeros(postclocks, dtype = np.uint32)
        prewave = np.zeros(50, dtype = np.uint32)
        DOwaveform = np.append(np.append(prewave, DOwaveform), postDOwave)
        status = 'waveform updated'
        return np.uint32(DOwaveform), AOwaveform, status

    else:
        status = 'invalid task type! Abort action'
        return None, None, status


def LinePlot(AOwaveform, DOwaveform = None, m=2, M=4):
    # clear content on plot
    plt.cla()

    if np.any(DOwaveform):
        plt.plot(range(len(DOwaveform)),DOwaveform,linewidth=2)
    # plot the new waveform
    plt.plot(range(len(AOwaveform)),AOwaveform,linewidth=2)
    # plt.ylim(np.min(AOwaveform)-0.2,np.max(AOwaveform)+0.2)
    plt.ylim([m,M])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.rcParams['savefig.dpi']=150
    # save plot as jpeg
    plt.savefig('lineplot.jpg')
    # load waveform image
    pixmap = QPixmap('lineplot.jpg')
    return pixmap

def fastLinePlot(AOwaveform, DOwaveform = None, width=800, height=300, m=2, M=4 ):
    """
    Render a 1D waveform directly to QPixmap (FAST!)
    """
    if m is None: m = np.min(AOwaveform)
    if M is None: M = np.max(AOwaveform)

    # Create empty pixmap
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)

    # Draw using QPainter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(0, 0, 200), 2)
    painter.setPen(pen)

    n = len(AOwaveform)
    x_scale = width / (n - 1)
    y_scale = height / (M - m+1)

    # Precompute transformed points
    points = [
        QPointF(i * x_scale, height - ((AOwaveform[i] - m) * y_scale))
        for i in range(n)
    ]
    # Draw polyline
    painter.drawPolyline(*points)
    painter.end()

    return pixmap

def ScatterPlot(mosaic):
    # clear content on plot
    plt.cla()
    # plot the new waveform
    plt.scatter(mosaic[0],mosaic[1])
    plt.plot(mosaic[0],mosaic[1])
    # plt.ylim(-2,2)
    plt.ylabel('Y stage',fontsize=15)
    plt.xlabel('X stage',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.rcParams['savefig.dpi']=150
    # save plot as jpeg
    plt.savefig('scatter.jpg')
    # load waveform image
    pixmap = QPixmap('scatter.jpg')
    return pixmap



def RGBImagePlot(matrix1 = [], matrix2 = [], m=0, M=1):
    if len(matrix2)>0:
        scale = 1
        matrix2 = np.float32(np.array(matrix2))
        matrix2[matrix2<m] = m
        matrix2[matrix2>M] = M
        # adjust image brightness
        matrix2 = np.uint8((matrix2-m+0.01)/np.abs(M-m+0.1)*127)
    else:
        scale = 2
        matrix2 = np.zeros(matrix1.shape)

    matrix1 = np.float32(np.array(matrix1))
    matrix1[matrix1<m] = m
    matrix1[matrix1>M] = M
    matrix1 = np.uint8((matrix1-m+0.01)/np.abs(M-m+0.1)*127*scale)

    height, width = matrix1.shape

    # Create an empty RGB array
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    # plt.figure()
    # plt.imshow(matrix1)
    # plt.show()
    # Assign each channel
    rgb_array[..., 0] = matrix1 + matrix2   # Red channel
    rgb_array[..., 1] = matrix1 # Green channel
    rgb_array[..., 2] = matrix1  # Blue channel

    # Convert to QImage
    bytes_per_line = 3 * width
    qimage = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

    # Convert to QPixmap and display
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def _normalize_to_uint8(matrix, m=None, M=None):
    matrix = np.float32(np.array(matrix))
    if m is None:
        m = np.nanpercentile(matrix, 1)
    if M is None:
        M = np.nanpercentile(matrix, 99)
    if M <= m:
        M = m + 1e-5
    matrix = np.clip(matrix, m, M)
    return np.uint8((matrix - m) / (M - m + 1e-5) * 255)

def RGBOverlayArray(intensity, dynamic, intensity_min, intensity_max, alpha=0.5, dyn_min=None, dyn_max=None):
    base = _normalize_to_uint8(intensity, intensity_min, intensity_max)
    overlay = _normalize_to_uint8(dynamic, dyn_min, dyn_max)
    alpha = float(np.clip(alpha, 0.0, 0.99))

    rgb_array = np.empty((base.shape[0], base.shape[1], 3), dtype=np.uint8)
    rgb_array[..., 0] = np.clip(base.astype(np.float32) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    rgb_array[..., 1] = np.clip(base.astype(np.float32) * (1.0 - 0.5 * alpha), 0, 255).astype(np.uint8)
    rgb_array[..., 2] = rgb_array[..., 1]
    return rgb_array

def RGBOverlayPlot(intensity, dynamic, intensity_min, intensity_max, alpha=0.5, dyn_min=None, dyn_max=None):
    rgb_array = RGBOverlayArray(intensity, dynamic, intensity_min, intensity_max, alpha, dyn_min, dyn_max)
    height, width, _ = rgb_array.shape
    qimage = QImage(rgb_array.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

def findchangept(signal, step):
    # python implementation of matlab function findchangepts
    L = len(signal)
    z = np.argmax(signal)
    last = np.min([z+30,L-2])
    signal = signal[1:last]
    L = len(signal)
    residual_error = np.ones(L)*9999999
    for ii in range(2,L-2,step):
        residual_error[ii] = (ii-1)*np.var(signal[0:ii])+(L-ii+1)*np.var(signal[ii+1:L])
    pts = np.argmin(residual_error)
    # plt.plot(residual_error[2:-2])
    return pts
