
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
# from libtiff import TIFF
import tifffile as TIFF
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
        # self.Dynmax = self.ui.Dynmax.value()
        # self.Dynmin = self.ui.Dynmin.value()
        
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
                    self.Process_aline(self.item.data, self.item.raw)
                    self.Update_contrast()
                elif self.item.action in ['FiniteBline','ContinuousBline']:
                    self.Process_bline(self.item.data, self.item.raw, self.item.dynamic)
                    self.Update_contrast()
                    self.display_actions += 1
                elif self.item.action in ['FiniteCscan','ContinuousCscan']:
                    self.display_actions += 1
                    if self.ui.DynCheckBox.isChecked():
                        self.Process_Cscan_Dynamic(self.item.data, self.item.dynamic)
                    else:
                        self.Process_Cscan(self.item.data, self.item.raw)
                    self.Update_contrast()
                    
                elif self.item.action == 'Process_Mosaic':
                    self.Process_Mosaic(self.item.data, self.item.raw, self.item.args)
                    self.Update_contrast()
                elif self.item.action == 'Return_mosaic':
                    self.Return_mosaic()
                elif self.item.action == 'Clear':
                    # self.SampleDynamic= []
                    # self.SampleMosaic= []
                    self.DynamicBlineIdx = 0
                elif self.item.action == 'UpdateContrast':
                    self.Update_contrast()
                elif self.item.action == 'UpdateContrastMosaic':
                    self.Update_contrast_Mosaic()
                elif self.item.action == 'UpdateContrastDyn':
                    self.Update_contrast_Dyn()
                elif self.item.action == 'display_counts':
                    self.print_display_counts(self.item.args)
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
                    self.Init_Mosaic(self.item.args)
                elif self.item.action == 'Save_mosaic':
                    self.Save_mosaic()
                    
                else:
                    message = 'Display and save thread is doing something invalid' + self.item.action
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                if time.time()-start>1:
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
            
    def print_display_counts(self, mode = ''):
        message = str(self.display_actions) + ' ' + mode +' displayed\n'
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
            
    def Process_aline(self, data, raw = False):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        # Bline averaging
        if data.shape[0] > 1:
            Ascan = np.mean(data,0)
        else:
            Ascan = data[0]
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Ascan = Ascan.reshape([Xpixels//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Ascan = np.mean(Ascan,1)
            Xpixels = Xpixels//self.ui.AlineAVG.value()
            
        self.Aline = Ascan[Xpixels//2]
        if self.ui.Save.isChecked():
            self.Save(Data = data, Raw = raw)
            
    
    def Process_bline(self, data, raw = False, dynamic = []):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        # Bline averaging
        if data.shape[0] > 1:
            Bline=np.mean(data,0)
        else:
            Bline = data[0]
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Bline = Bline.reshape([Xpixels//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//self.ui.AlineAVG.value()

        self.Bline = np.transpose(Bline)
        self.DynBline = np.transpose(dynamic)

        
        if self.ui.Save.isChecked():
            self.Save(Data = data, Raw = raw)

            
    def Process_Cscan_Dynamic(self, data, dynamic):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size()
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = self.ui.Ypixels.value()
        # Bline averaging
        if data.shape[0] > 1:
            Bline=np.mean(data,0)
        else:
            Bline = data[0]
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Bline = Bline.reshape([Xpixels//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//self.ui.AlineAVG.value()

        self.Bline = np.transpose(Bline)
        self.DynBline = np.transpose(dynamic)
        
        if len(self.AIP)==0:
            self.AIP = np.zeros([Ypixels, Xpixels])
        if any(dynamic):
            if len(self.Dyn)==0:
                self.Dyn = np.zeros([Ypixels, Xpixels])
                
        # print(Bline.shape, self.AIP.shape)
        print('Ypixel: ', self.DynamicBlineIdx)
        self.AIP[self.DynamicBlineIdx, :] = np.mean(Bline,1)
        if any(dynamic):
            self.Dyn[self.DynamicBlineIdx, :] = np.mean(dynamic,1)
        self.DynamicBlineIdx = self.DynamicBlineIdx + 1
        
        
        if self.ui.Save.isChecked():
            self.Save(Data = data, Dynamic = dynamic)
        
        
    def Process_Cscan(self, data, raw = False):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = self.ui.Ypixels.value() * Yrpt
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
            Cscan = Cscan.reshape([Ypixels, Xpixels//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Cscan = np.mean(Cscan,2)
            Xpixels = Xpixels//self.ui.AlineAVG.value()
        # print(data[10,100,50:60])
        self.AIP = np.mean(Cscan,2)
        self.Bline = np.transpose(Cscan[0,:,:]).copy()# has to be first index, otherwise the memory space is not continuous

        
        if self.ui.Save.isChecked():
            self.Save(Data = data, Raw = raw)
            
    def Save(self, Data=[], Dynamic=[], Raw=False):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(Raw)
        if Zpixels != Data.shape[2]:
            Zpixels = Data.shape[2]
        Ypixels = self.ui.Ypixels.value() * Yrpt
        
        if self.ui.ACQMode.currentText() in ['FiniteAline', 'ContinuousAline']:
            filename = self.ui.DIR.toPlainText()+'/'+self.AlineFilename([Yrpt,Xpixels,Zpixels])
            for ii in range(Yrpt):
                TIFF.imwrite(filename, Data[ii], append=True)
                
        elif self.ui.ACQMode.currentText() in ['FiniteBline', 'ContinuousBline']:
            filename = self.ui.DIR.toPlainText()+'/'+self.BlineFilename([Yrpt,Xpixels,Zpixels])
            for ii in range(Yrpt):
                TIFF.imwrite(filename, Data[ii], append=True)

            if self.ui.DynCheckBox.isChecked():
                filename = self.ui.DIR.toPlainText()+'/'+self.BlineDynFilename([Yrpt,Xpixels,Zpixels])
                TIFF.imwrite(filename, Data[ii], append=True)
                
        elif self.ui.ACQMode.currentText() in ['ContinuousCscan', 'FiniteCscan']:
            if self.ui.DynCheckBox.isChecked():
                CscanBlineFileName, CscanDynBlineFileName = self.CscanDynFilename([Yrpt,Xpixels,Zpixels])
                filename = self.ui.DIR.toPlainText()+'/'+CscanBlineFileName
                for ii in range(Yrpt):
                    TIFF.imwrite(filename, Data[ii], append=True)
                
                filename = self.ui.DIR.toPlainText()+'/'+CscanDynBlineFileName
                TIFF.imwrite(filename, Dynamic, append=True)
            else:
                filename = self.ui.DIR.toPlainText()+'/'+self.CscanFilename([Ypixels,Xpixels,Zpixels])
                for ii in range(Ypixels):
                    TIFF.imwrite(filename, Data[ii], append=True)
   
    def Init_Mosaic(self, args):
        """
        Initializes the mosaic buffer based on the physical span of all FOVs.
        args: [fov_locs, fov_size_px, fov_size_mm]
        fov_locs: list of {'x': val, 'y': val} in mm
        fov_size_px: (width_px, height_px) e.g., (1000, 2000)
        fov_size_mm: (width_mm, height_mm) e.g., (2.0, 3.0)
        """
        fov_locs, fov_size_px, fov_size_mm = args
        fw_px, fh_px = fov_size_px
        fw_mm, fh_mm = fov_size_mm
        
        # 1. Find the bounding box of the FOV centers
        xs = [p['x'] for p in fov_locs]
        ys = [p['y'] for p in fov_locs]
        self.min_x, max_x = min(xs), max(xs)
        self.min_y, max_y = min(ys), max(ys)
        
        # 2. Calculate how many tiles are needed in each dimension
        # We use round() to handle tiny stage step errors
        num_cols = int(round((max_x - self.min_x) / fw_mm)) + 1
        num_rows = int(round((max_y - self.min_y) / fh_mm)) + 1
        
        # 3. Initialize/Overwrite the active mosaic buffer
        mw_px = num_cols * fw_px
        mh_px = num_rows * fh_px
        self.SampleMosaic = np.zeros((mh_px, mw_px), dtype=np.float32)
        # print(self.SampleMosaic.shape)
        # Store these for use in Process_Mosaic
        self.fw_mm, self.fh_mm = fw_mm, fh_mm
        self.fw_px, self.fh_px = fw_px, fh_px
        
        print(f"Mosaic Initialized: {num_cols}x{num_rows} tiles ({mw_px}x{mh_px} px)")
        
    def Focusing(self, cscan):
         print(cscan.shape)

         bscan = cscan.mean(0)

         ascan = bscan.mean(0)
         print(ascan.shape)
         surfHeight = findchangept(ascan,1)

         ##########################################################
         self.ui.SurfHeight.setValue(surfHeight)
         message = 'tile surf is:'+str(surfHeight)
         print(message)
         self.log.write(message)
 
    def Process_Mosaic(self, data, raw=False, args=[]):
        """
        Stitches the FOV into the mosaic by calculating its grid index from the anchor.
        args: [fov_x, fov_y]
        """
        fov_x, fov_y = args['x'], args['y']
        
        # 1. Generate AIP projection from raw data (Y, X, Z)
        self.Process_Cscan(data, raw)
        # 2. Calculate the Grid Index (Column and Row)
        # We determine how many FOV-widths away from the minimum X/Y we are
        col_idx = int(round((fov_x - self.min_x) / self.fw_mm))
        row_idx = int(round((fov_y - self.min_y) / self.fh_mm))

        # 3. Calculate pixel offsets
        off_x = col_idx * self.fw_px
        off_y = row_idx * self.fh_px

        # 4. Paste into the mosaic buffer
        # Bounds checking added just in case of unexpected stage travel
        # print('sampleMosaic: ', self.SampleMosaic)
        mh, mw = self.SampleMosaic.shape
        y1, y2 = off_y, off_y + self.fh_px
        x1, x2 = off_x, off_x + self.fw_px
        # print(x1,x2,y1,y2)
        
        if y2 <= mh and x2 <= mw:
            self.SampleMosaic[y1:y2, x1:x2] = self.AIP
        else:
            print(f"Warning: Tile at ({col_idx}, {row_idx}) is out of mosaic bounds.")

        # 5. Update UI


    def writeTiff(self,filename, image, overlap):
        tif = TIFF.open(filename, mode=overlap)
        tif.write_image(image)
        tif.close()
        
    def Return_mosaic(self):
        self.MosaicQueue.put(self.SampleMosaic)
        # self.Focusing(self.cscan_sum)
        

        
    def Save_mosaic(self):
        if self.ui.Save.isChecked():
            filename = self.ui.DIR.toPlainText()+'/aip/slice'+str(self.sliceNum)+'coase.tif'
            TIFF.imwrite(filename, self.SampleMosaic, append=False)
            
        
    def Update_contrast(self):
            ym=self.ui.XZmin.value()
            yM=self.ui.XZmax.value()
        # if self.ui.XZmin.value() != self.XZmin or self.ui.XZmax.value() != self.XZmax:
            if self.ui.ACQMode.currentText() in ['FiniteAline', 'ContinuousAline']:
                try:
                    # if self.ui.LOG.currentText() == '10log10':
                    #     data=10*np.log10(data+0.000001)
                    w = self.ui.XZplane.width()
                    h = self.ui.XZplane.height()
                    pixmap = fastLinePlot(self.Aline, width=w, height=h, m=ym, M=yM )
                    self.ui.XZplane.setPixmap(pixmap)
                except Exception as error:
                    print(error)
                    
            elif self.ui.ACQMode.currentText() in ['FiniteBline', 'ContinuousBline']:
                try:
                    # if self.ui.LOG.currentText() == '10log10':
                    #     data=np.float32(10*np.log10(data+0.000001))
                    if len(self.DynBline)>0:
                        pixmap = RGBImagePlot(matrix1=self.Bline, matrix2=self.DynBline, m=ym, M=yM)
                    else:
                        pixmap = RGBImagePlot(matrix1=self.Bline, m=ym, M=yM)
                    self.ui.XZplane.setPixmap(pixmap)
                except Exception as error:
                    print(error)
            elif self.ui.ACQMode.currentText() in ['ContinuousCscan', 'FiniteCscan']:
                try:
                    # if self.ui.LOG.currentText() == '10log10':
                    #     data=np.float32(10*np.log10(data+0.000001))
                    if self.ui.DynCheckBox.isChecked():
                        pixmap = RGBImagePlot(matrix1=self.Bline, matrix2=self.DynBline, m=ym, M=yM)
                        self.ui.XZplane.setPixmap(pixmap)
                        pixmap = RGBImagePlot(matrix1=self.AIP, matrix2=self.Dyn, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
                        self.ui.XYplane.setPixmap(pixmap)
                        
                    else:
                        pixmap = RGBImagePlot(matrix1=self.Bline, m=ym, M=yM)
                        self.ui.XZplane.setPixmap(pixmap)
                        pixmap = RGBImagePlot(matrix1=self.AIP, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
                        self.ui.XYplane.setPixmap(pixmap)
                except Exception as error:
                    print(error)
            elif self.ui.ACQMode.currentText() in ['PlateScan']:
                try:
                    pixmap = RGBImagePlot(matrix1=self.Bline, m=ym, M=yM)
                    self.ui.XZplane.setPixmap(pixmap)
                    # print('Bline intensity:', self.Bline[100,100:105])
                    pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
                    self.ui.XYplane.setPixmap(pixmap)
                except:
                    pass

            # self.XZmax = self.ui.XZmax.value()
            # self.XZmin = self.ui.XZmin.value()

    # def Update_contrast_Mosaic(self):
    #     # if self.ui.Intmin.value() != self.Intmin or self.ui.Intmax.value() != self.Intmax:
    #         try:
    #             ym=self.ui.XZmin.value()
    #             yM=self.ui.XZmax.value()
    #             pixmap = RGBImagePlot(matrix1=self.Bline, m=ym, M=yM)
    #             self.ui.XZplane.setPixmap(pixmap)
    #             pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Intmin.value(), M=self.ui.Intmax.value())
    #             self.ui.XYplane.setPixmap(pixmap)
    #         except:
    #             pass
    #         # self.Intmax = self.ui.Intmax.value()
    #         # self.Intmin = self.ui.Intmin.value()
            
    def Update_contrast_Dyn(self):
        if self.ui.Dynmin.value() != self.Dynmin or self.ui.Dynmax.value() != self.Dynmax:
            try:
                pixmap = RGBImagePlot(matrix2=self.SampleDynamic, m=self.ui.Dynmin.value(), M=self.ui.Dynmax.value())
                # clear content on the waveformLabel
                # self.ui.SampleDynamic.clear()
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
        filename = 'Cscan-'+str(self.CscanNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
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
        
