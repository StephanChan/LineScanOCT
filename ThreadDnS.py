
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import LinePlot, findchangept, RGBImagePlot, fastLinePlot
import numpy as np
import traceback
global SCALE
SCALE =1000
import matplotlib.pyplot as plt
import datetime
import os
from scipy import ndimage
from libtiff import TIFF
import time

class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        self.SampleDynamic= []
        self.SampleMosaic= []
        self.sliceNum = 0
        self.tileNum = 1
        self.AlineNum = 1
        self.BlineNum = 1
        self.BlineDynNum = 1
        self.DynamicBlineIdx = 0
        self.CscanNum = 1
        self.CscanDynNum = 1
        self.totalTiles = 0
        self.display_actions = 0
        
    def run(self):
        self.sliceNum = self.ui.SliceN.value()-1
        self.XZmax = self.ui.XZmax.value()
        self.XZmin = self.ui.XZmin.value()
        self.Intmax = self.ui.Intmax.value()
        self.Intmin = self.ui.Intmin.value()
        self.Dynmax = self.ui.Dynmax.value()
        self.Dynmin = self.ui.Dynmin.value()
        
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        self.DnSflag = 'busy'
        while self.item.action != 'exit':
            start=time.time()
            #self.ui.statusbar.showMessage('Display thread is doing ' + self.item.action)
            try:
                if self.item.action in ['FiniteAline','ContinuousAline']:
                    self.display_actions += 1
                    self.Display_aline(self.item.data, self.item.raw)
                elif self.item.action in ['FiniteBline','ContinuousBline']:
                    self.Display_bline(self.item.data, self.item.raw, self.item.dynamic)
                    self.display_actions += 1
                elif self.item.action in ['FiniteCscan','ContinuousCscan']:
                    self.display_actions += 1
                    if len(self.item.dynamic)>0:
                        self.display_Cscan_Dynamic(self.item.data, self.item.dynamic)
                    else:
                        self.Display_Cscan(self.item.data, self.item.raw)
                    
                elif self.item.action == 'Mosaic':
                    self.Process_Mosaic(self.item.data, self.item.raw, self.item.args)
                elif self.item.action == 'display_mosaic':
                    self.Display_mosaic()
                elif self.item.action == 'Clear':
                    self.SampleDynamic= []
                    self.SampleMosaic= []
                    self.DynamicBlineIdx = 0
                elif self.item.action == 'UpdateContrastBline':
                    self.Update_contrast_Bline()
                elif self.item.action == 'UpdateContrastMosaic':
                    self.Update_contrast_Mosaic()
                elif self.item.action == 'UpdateContrastDyn':
                    self.Update_contrast_Dyn()
                elif self.item.action == 'display_counts':
                    self.print_display_counts()
                elif self.item.action == 'restart_tilenum':
                    self.restart_tilenum()
                elif self.item.action == 'change_slice_number':
                    self.sliceNum = self.ui.SliceN.value()-1
                    self.ui.CuSlice.setValue(self.sliceNum)
                elif self.item.action == 'agarTile':
                    self.SurfFilename()
                elif self.item.action == 'WriteAgar':
                    self.WriteAgar(self.item.data, self.item.args)
                elif self.item.action == 'Init_Mosaic':
                    self.Init_Mosaic(self.item.data, self.item.args)
                elif self.item.action == 'Save_mosaic':
                    self.Save_mosaic()
                    
                else:
                    message = 'Display and save thread is doing something invalid' + self.item.action
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                if time.time()-start>0.1:
                    print('time for DnS:',round(time.time()-start,3))
            except Exception as error:
                message = "\nAn error occurred:"+" skip the display and save action\n"
                print(message)
                self.ui.statusbar.showMessage(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            # num+=1
            # print(num, 'th display\n')
            self.item = self.queue.get()
            
        self.ui.statusbar.showMessage("Display and save Thread successfully exited...")
            
    def print_display_counts(self):
        message = str(self.display_actions)+ self.ui.ACQMode.currentText() +' displayed\n'
        print(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        self.display_actions = 0
        
    def get_FOV_size(self, raw=False):
        # get number of Z pixels
        if not raw:
            Zpixels = self.ui.DepthRange.value()
        else:
            Zpixels = self.ui.NSamples.value()#-self.ui.DelaySamples.value()-self.ui.TrimSamples.value()
        # get number of X pixels
        Xpixels = self.ui.AlinesPerBline.value()
            
        Yrpt = self.ui.BlineAVG.value()
        

        return Zpixels, Xpixels, Yrpt
            
    def Display_aline(self, data, raw = False):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ascan = np.float32(np.mean(data,0))
        Ascan = Ascan[Xpixels//2]
        # print(Ascan)
        self.Aline = Ascan
        # float32 data type
        if self.ui.FFTDevice.currentText() in ['None']:
            ym=self.ui.XZmin.value()
            yM=self.ui.XZmax.value()
        else:
            ym=self.ui.XZmin.value()
            yM=self.ui.XZmax.value()
        # t0=time.time()
        # pixmap = LinePlot(Ascan, [], ym, yM)
        w = self.ui.XZplane.width()
        h = self.ui.XZplane.height()
        pixmap = fastLinePlot(Ascan, width=w, height=h, m=ym, M=yM )
        # print('time for line plot took ',round(time.time()-t0,3))
        # clear content on the waveformLabel
        # self.ui.XZplane.clear()
        # update iamge on the waveformLabel
        self.ui.XZplane.setPixmap(pixmap)
        
        if self.ui.Save.isChecked():
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+self.AlineFilename([Yrpt,Xpixels,Zpixels]), mode='w')
            for ii in range(Yrpt):
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(data[ii])
            tif.close()
            
    
    def Display_bline(self, data, raw = False, dynamic = []):
        # print(data.shape)
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        # print(data.shape)
        # print(data[0,0,0:5])
        # Bline averaging
        if data.shape[0] > 1:
            Bline=np.mean(data,0)
        else:
            Bline = data[0]
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Bline = Bline.reshape([self.ui.NSamples.value()//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = self.ui.NSamples.value()//self.ui.AlineAVG.value()

        self.Bline = np.transpose(Bline)

        if self.ui.DynCheckBox.isChecked():
            self.DynBline = np.transpose(dynamic)
            pixmap = RGBImagePlot(matrix1 = np.float32(self.Bline), matrix2 = np.float32(self.DynBline*1), m=self.ui.XZmin.value(), M=self.ui.XZmax.value())
        else:
            self.DynBline = []
            pixmap = RGBImagePlot(matrix1 = np.float32(self.Bline), m=self.ui.XZmin.value(), M=self.ui.XZmax.value())
        # clear content on the waveformLabel
        self.ui.XZplane.clear()
        # update iamge on the waveformLabel
        self.ui.XZplane.setPixmap(pixmap)
        
        if self.ui.Save.isChecked():
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+self.BlineFilename([Yrpt,Xpixels,Zpixels]), mode='w')
            for ii in range(Yrpt):
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(data[ii])
            tif.close()
            if self.ui.DynCheckBox.isChecked():
                tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+self.BlineDynFilename([Yrpt,Xpixels,Zpixels]), mode='w')
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(dynamic)
                tif.close()
            
    def display_Cscan_Dynamic(self, data, dynamic):
        # print(data.shape)
        Zpixels, Xpixels, Yrpt = self.get_FOV_size()
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = self.ui.Ypixels.value()
        # print(data.shape)
        # print(data[0,0,0:5])
        # Bline averaging
        if data.shape[0] > 1:
            Bline=np.mean(data,0)
        else:
            Bline = data[0]
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Bline = Bline.reshape([self.ui.NSamples.value()//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = self.ui.NSamples.value()//self.ui.AlineAVG.value()

        self.Bline = np.transpose(Bline)

        self.DynBline = np.transpose(dynamic)
        pixmap = RGBImagePlot(matrix1 = np.float32(self.Bline), matrix2 = np.float32(self.DynBline*1), m=self.ui.XZmin.value(), M=self.ui.XZmax.value())
        # clear content on the waveformLabel
        self.ui.XZplane.clear()
        # update iamge on the waveformLabel
        self.ui.XZplane.setPixmap(pixmap)
        
        if len(self.SampleMosaic)==0:
            self.SampleMosaic = np.zeros([Xpixels, Ypixels])
            self.SampleDynamic = np.zeros([Xpixels, Ypixels])
        # print(Bline.shape, self.SampleMosaic.shape)
        self.SampleMosaic[:,self.DynamicBlineIdx] = np.mean(Bline,1)
        self.SampleDynamic[:,self.DynamicBlineIdx] = np.mean(dynamic,1)
        self.DynamicBlineIdx = self.DynamicBlineIdx + 1
        
        pixmap = RGBImagePlot(matrix1 = np.float32(self.SampleMosaic), m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
        # clear content on the waveformLabel
        self.ui.SampleMosaic.clear()
        # update iamge on the waveformLabel
        self.ui.SampleMosaic.setPixmap(pixmap)
        
        pixmap = RGBImagePlot(matrix2 = np.float32(self.SampleDynamic), m=self.ui.Dynmin.value(), M=self.ui.Dynmax.value())
        # clear content on the waveformLabel
        self.ui.SampleDynamic.clear()
        # update iamge on the waveformLabel
        self.ui.SampleDynamic.setPixmap(pixmap)
        
        if self.ui.Save.isChecked():
            CscanBlineFileName, CscanDynBlineFileName = self.CscanDynFilename([Yrpt,Xpixels,Zpixels])
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+CscanBlineFileName, mode='w')
            for ii in range(Yrpt):
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(data[ii])
            tif.close()
            
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+CscanDynBlineFileName, mode='a')
            # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
            tif.write_image(dynamic)
            tif.close()
        
        
    def Display_Cscan(self, data, raw = False):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = self.ui.Ypixels.value() * self.ui.BlineAVG.value()

        # Bline averaging
        if self.ui.BlineAVG.value() > 1:
            # reshape into Ypixels x Xpixels x Zpixels
            Cscan = data.reshape([self.ui.Ypixels.value(), self.ui.BlineAVG.value(), Xpixels,Zpixels])
            Cscan=np.mean(Cscan,1)
            Ypixels = self.ui.Ypixels.value()
        else:
            Cscan = data.copy()
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Cscan = Cscan.reshape([Ypixels,self.ui.NSamples.value(), self.ui.AlineAVG.value(), Zpixels])
            Cscan = np.mean(Cscan,2)
            Xpixels = self.ui.NSamples.value()
            
        self.Cscan = Cscan
        self.Bline = np.transpose(Cscan[0,:,:]).copy()# has to be first index, otherwise the memory space is not continuous
        # print(plane.shape)
        pixmap = RGBImagePlot(matrix1 = self.Bline, m=self.ui.XZmin.value(), M=self.ui.XZmax.value())
        # clear content on the waveformLabel
        self.ui.XZplane.clear()
        # update image on the waveformLabel
        self.ui.XZplane.setPixmap(pixmap)
        
        self.SampleMosaic = np.mean(Cscan,2)# has to be first index, otherwise the memory space is not continuous
        pixmap = RGBImagePlot(matrix1 = self.SampleMosaic, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
        # clear content on the waveformLabel
        self.ui.SampleMosaic.clear()
        # update image on the waveformLabel
        self.ui.SampleMosaic.setPixmap(pixmap)
        ###################### plot 3D visulaization
        if self.use_maya:
            self.ui.mayavi_widget.visualization.update_data(self.Cscan/500)
        if self.ui.Save.isChecked():
            if raw:
                data = np.uint16(data)
            else:
                data = np.uint16(data/SCALE*65535)
            self.WriteData(data, self.CscanFilename([Ypixels,Xpixels,Zpixels]))
        
   

    # def writeTiff(self,filename, image, overlap):
    #     tif = TIFF.open(filename, mode=overlap)
    #     tif.write_image(image)
    #     tif.close()
        
    # def Display_mosaic(self):
    #     pixmap = RGBImagePlot(self.SampleMosaic, self.ui.Intmin.value(), self.ui.Intmax.value())
    #     self.ui.SampleMosaic.clear()
    #     self.ui.SampleMosaic.setPixmap(pixmap)
        
    # def Save_mosaic(self):
    #     if self.ui.Save.isChecked():
    #         tif = TIFF.open(self.ui.DIR.toPlainText()+'/aip/slice'+str(self.sliceNum)+'coase.tif', mode='w')
    #         tif.write_image(self.SampleMosaic)
    #         tif.close()
            
        
    def Update_contrast_Bline(self):
        if self.ui.XZmin.value() != self.XZmin or self.ui.XZmax.value() != self.XZmax:
            if self.ui.ACQMode.currentText() in ['FiniteAline', 'ContinuousAline']:
                try:
                    # if self.ui.LOG.currentText() == '10log10':
                    #     data=10*np.log10(data+0.000001)
                    if self.ui.FFTDevice.currentText() in ['None']:
                        ym=self.ui.XZmin.value()
                        yM=self.ui.XZmax.value()
                    else:
                        ym=self.ui.XZmin.value()
                        yM=self.ui.XZmax.value()
                    w = self.ui.XZplane.width()
                    h = self.ui.XZplane.height()
                    pixmap = fastLinePlot(self.Aline, width=w, height=h, m=ym, M=yM )
                    # clear content on the waveformLabel
                    self.ui.XZplane.clear()
                    # update iamge on the waveformLabel
                    self.ui.XZplane.setPixmap(pixmap)
                except Exception as error:
                    print(error)
                    
            elif self.ui.ACQMode.currentText() in ['FiniteBline', 'ContinuousBline','Mosaic','ContinuousCscan', 'FiniteCscan']:
                try:
                    # data = np.flip(data, 1).copy()
                    # if self.ui.LOG.currentText() == '10log10':
                    #     data=np.float32(10*np.log10(data+0.000001))
                    
                    if len(self.DynBline)>0:
                        pixmap = RGBImagePlot(np.float32(self.Bline), np.float32(self.DynBline*1), self.ui.XZmin.value(), self.ui.XZmax.value())
                    else:
                        pixmap = RGBImagePlot(matrix1=np.float32(self.Bline), m=self.ui.XZmin.value(), M=self.ui.XZmax.value())
                    # clear content on the waveformLabel
                    # print(self.Bline[0,0:5])
                    self.ui.XZplane.clear()
                    # update iamge on the waveformLabel
                    self.ui.XZplane.setPixmap(pixmap)
                except:
                    pass
            self.XZmax = self.ui.XZmax.value()
            self.XZmin = self.ui.XZmin.value()

    def Update_contrast_Mosaic(self):
        if self.ui.Intmin.value() != self.Intmin or self.ui.Intmax.value() != self.Intmax:
            try:
                pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
                # clear content on the waveformLabel
                self.ui.SampleMosaic.clear()
                # update iamge on the waveformLabel
                self.ui.SampleMosaic.setPixmap(pixmap)
            except:
                pass
            self.Intmax = self.ui.Intmax.value()
            self.Intmin = self.ui.Intmin.value()
            
    def Update_contrast_Dyn(self):
        if self.ui.Dynmin.value() != self.Dynmin or self.ui.Dynmax.value() != self.Dynmax:
            try:
                pixmap = RGBImagePlot(matrix2=self.SampleDynamic, m=self.ui.Dynmin.value(), M=self.ui.Dynmax.value())
                # clear content on the waveformLabel
                self.ui.SampleDynamic.clear()
                # update iamge on the waveformLabel
                self.ui.SampleDynamic.setPixmap(pixmap)
            except:
                pass
            self.Dynmax = self.ui.Dynmax.value()
            self.Dynmin = self.ui.Dynmin.value()
    
    def restart_tilenum(self):
        self.tileNum = 1
        self.sliceNum = self.sliceNum+1
        self.ui.CuSlice.setValue(self.sliceNum)
        if not os.path.exists(self.ui.DIR.toPlainText()+'/aip/vol'+str(self.sliceNum)):
            os.mkdir(self.ui.DIR.toPlainText()+'/aip/vol'+str(self.sliceNum))
        if not os.path.exists(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum)):
            os.mkdir(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum))
        if not os.path.exists(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum)):
            os.mkdir(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum))
        
    def SurfFilename(self, shape = [0,0,0]):
        filename = 'slice-'+str(self.sliceNum)+'-tile-'+str(self.tileNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.bin'
        self.tileNum = self.tileNum + 1
    
        print(filename)
        # self.ui.PrintOut.append(filename)
        self.log.write(filename)
        return filename
    
    def CscanFilename(self, shape):
        filename = 'Cscan-'+str(self.CscanNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.bin'
        self.CscanNum = self.CscanNum + 1
        return filename
    
    def CscanDynFilename(self, shape):
        DynBlinefilename = 'CscanDynBline-'+str(self.CscanDynNum)+'-Y'+str(self.ui.Ypixels.value())+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        BlineFilename = 'CscanBline-'+str(self.CscanDynNum)+str(self.DynamicBlineIdx)+'-Yrpt'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        # self.DynamicBlineIdx = self.DynamicBlineIdx + 1
        if self.DynamicBlineIdx == self.ui.Ypixels.value():
            self.CscanDynNum = self.CscanDynNum + 1
            # self.DynamicBlineIdx = 0
        return BlineFilename, DynBlinefilename
    
    def BlineFilename(self, shape):
        filename = 'Bline-'+str(self.BlineNum)+'-Yrpt'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.BlineNum = self.BlineNum + 1
        return filename
    
    def BlineDynFilename(self, shape):
        filename = 'BlineDyn-'+str(self.BlineNum)+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.BlineDynNum = self.BlineDynNum + 1
        return filename
    
    def AlineFilename(self, shape):
        filename = 'Aline-'+str(self.AlineNum)+'-Yrpt'+str(shape[0])+'-Xrpt'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.AlineNum = self.AlineNum + 1
        return filename

    def WriteData(self, data, filename):
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        import time
        start = time.time()
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        if time.time()-start > 1:
            message = 'time for saving: '+str(round(time.time()-start,3))
            print(message)
            # self.ui.PrintOut.append(message)
            self.log.write(message)
        
    def WriteAgar(self, data, args):
        [Ystep, Xstep] = args
        filename = 'slice-'+str(self.sliceNum)+'-agarTiles X-'+str(Xstep)+'-by Y-'+str(Ystep)+'-.bin'
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        
