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
from PyQt5.QtGui import QPixmap
import qimage2ndarray as qpy
from matplotlib import pyplot as plt

class LOG():
    def __init__(self, ui):
        super().__init__()
        import datetime
        current_time = datetime.datetime.now()
        self.dir = os.getcwd() + '/log_files'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.filePath = self.dir +  "/" + 'log_'+\
            str(current_time.year)+'-'+\
            str(current_time.month)+'-'+\
            str(current_time.day)+'-'+\
            str(current_time.hour)+'-'+\
            str(current_time.minute)+'-'+\
            str(current_time.second)+'.txt'
    def write(self, message):
        fp = open(self.filePath, 'a')
        fp.write(message+'\n')
        fp.close()
        # return 0


def GenGalvoWave(StepSize = 1, Steps = 1000, AVG = 1, obj = '5X', postclocks = 50, Galvo_bias = 0):
    # total number of steps is the product of steps and aline average number
    # use different angle to mm ratio for different objective
    if obj == '5X':
        angle2mmratio = 2.094/1.19
    elif obj == '10X':
        angle2mmratio = 2.094/2/1.19
    elif obj == '20X':
        angle2mmratio = 2.094/1.19/4
        
    else:
        status = 'objective not calibrated, abort generating Galvo waveform'
        return None, status
    # X range is product of steps and step size
    Xrange = StepSize*Steps/AVG/1000
    # max voltage is converted from half of max X range plus bias divided by angle2mm ratio
    # extra division by 2 is because galvo angle change is only half of beam deviation angle
    Vmax = (Xrange/2)/angle2mmratio/2+Galvo_bias
    Vmin = (-Xrange/2)/angle2mmratio/2+Galvo_bias
    # fly-back time in unit of clocks
    steps2=postclocks
    # linear waveform
    waveform=np.linspace(Vmin, Vmax, Steps)
    # print(len(waveform))
    # fly-back waveform
    Postwave = (Vmax-Vmin)/2*np.cos(np.arange(0,np.pi,np.pi/steps2))+(Vmax+Vmin)/2
    # append all waveforms together
    waveform = np.append(waveform, Postwave)
    
    status = 'waveform updated'
    return waveform, status


def GenAODO(mode='RptBline',obj = '5X',postclocks = 50, YStepSize = 1, YSteps = 200, BVG = 1, Galvo_bias = 0):
    # BVG: Bline average
    # bias: Galvo bias voltage
    # postclocks: #Aline triggers for Galvo fly-back
    
    # DO clock is synchronuous with Galvo waveform
    # DO configure: port0 line 0 
    if mode in ['RptAline', 'SingleAline', 'RptBline', 'SingleBline']:
        
        AOwaveform = np.ones(BVG*2) * Galvo_bias
        DOwaveform = np.ones([BVG, 2],dtype = np.uint32)
        DOwaveform[:,1] = 0
        DOwaveform=DOwaveform.flatten()
        status = 'waveform updated'
        return np.uint32(DOwaveform), AOwaveform, status
    
    elif mode in ['SingleCscan','Mosaic']:
        # generate AO waveform for Galvo control for one Bline
        AOwaveform, status = GenGalvoWave(YStepSize, YSteps*2, BVG*2, obj, postclocks, Galvo_bias)
        DOwaveform = np.ones([YSteps, 2],dtype = np.uint32)
        DOwaveform[:,1] = 0
        DOwaveform=DOwaveform.flatten()
        status = 'waveform updated'
        return np.uint32(DOwaveform), AOwaveform, status
    
    else:
        status = 'invalid task type! Abort action'
        return None, None, status
    

def GenMosaic_XYGalvo(Xmin, Xmax, Ymin, Ymax, FOV, overlap=10):
    # all arguments are with units mm
    # overlap is with unit %
    if Xmin > Xmax:
        status = 'Xmin is larger than Xmax, Mosaic generation failed'
        return None, status
    if Ymin > Ymax:
        status = 'Y min is larger than Ymax, Mosaic generation failed'
        return None, status
    # get FOV step size
    stepsize = FOV*(1-overlap/100)
    # get how many FOVs in X direction
    Xsteps = np.ceil((Xmax-Xmin)/stepsize)
    # get actual X range
    actualX=Xsteps*stepsize
    # generate start and stop position in X direction
    # add or subtract a small number to avoid precision loss
    startX=Xmin-(actualX-(Xmax-Xmin))/2
    stopX = Xmax+(actualX-(Xmax-Xmin))/2+0.01
    # generate X positions
    Xpositions = np.arange(startX, stopX, stepsize)
    #print(Xpositions)
    
    Ysteps = np.ceil((Ymax-Ymin)/stepsize)
    actualY=Ysteps*stepsize
    
    startY=Ymin-(actualY-(Ymax-Ymin))/2
    stopY = Ymax+(actualY-(Ymax-Ymin))/2+0.01
    
    Ypositions = np.arange(startY, stopY, stepsize)
    
    Positions = np.meshgrid(Xpositions, Ypositions)
    status = 'Mosaic Generation success'
    return Positions, status
    
    
def GenHeights(start, depth, Nplanes):
    return np.arange(start, start+Nplanes*depth/1000+0.01, depth/1000)



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


def ImagePlot(matrix, m=0, M=1):
    matrix = np.array(matrix)
    matrix[matrix<m] = m
    matrix[matrix>M] = M
    # adjust image brightness
    data = np.uint8((matrix-m)/np.abs(M-m+0.00001)*255.0)
    try:
        im = qpy.gray2qimage(data)
        pixmap = QPixmap(im)
    except:
        # print(data.shape)
        pixmap = QPixmap(qpy.gray2qimage(np.zeros(1000,1000)))
    return pixmap
    
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
        