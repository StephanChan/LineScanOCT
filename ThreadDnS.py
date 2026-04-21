
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import findchangept
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
        # self.SampleDynamic= []
        self.SampleMosaic= []
        self.AIP = []
        self.Dyn = []
        self.sliceNum = 0
        self.sampleID = 1
        self.tileNum = 1
        self.CscanNum = 1
        self.AlineNum = 1
        self.BlineNum = 1
        
        self.DynamicBlineIdx = 0
        # self.totalTiles = 0
        self.display_actions = 0
        
    def run(self):
        self.sliceNum = self.ui.SliceN.value()
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
            self.current_acq_mode = self.item.acq_mode
            try:
                if self.item.action in ['FiniteAline','ContinuousAline']:
                    self.display_actions += 1
                    self.Process_aline(self.item.data, self.item.raw, self.current_acq_mode)
                    self._emit_display(kind="aline")
                elif self.item.action in ['FiniteBline','ContinuousBline']:
                    self.Process_bline(self.item.data, self.item.raw, self.item.dynamic, self.current_acq_mode)
                    self.display_actions += 1
                    self._emit_display(kind="bline")
                elif self.item.action in ['FiniteCscan','ContinuousCscan']:
                    self.display_actions += 1
                    if self.ui.DynCheckBox.isChecked():
                        self.Process_Cscan_Dynamic(self.item.data, self.item.dynamic, self.current_acq_mode)
                    else:
                        self.Process_Cscan(self.item.data, self.item.raw, self.current_acq_mode)
                    self._emit_display(kind="cscan")
                    
                elif self.item.action == 'Process_Mosaic':
                    self.Process_Mosaic(self.item.data, self.item.raw, self.item.payload, self.current_acq_mode)
                    self._emit_display(kind="mosaic")
                elif self.item.action == 'Return_mosaic':
                    self.Return_mosaic()
                elif self.item.action == 'Clear':
                    # self.SampleDynamic= []
                    # self.SampleMosaic= []
                    self.DynamicBlineIdx = 0
                elif self.item.action == 'display_counts':
                    self.print_display_counts(self.item.payload)
                elif self.item.action == 'restart_tilenum':
                    self.restart_tilenum()
                elif self.item.action == 'change_slice_number':
                    self.sliceNum = self.ui.SliceN.value()
                    self.ui.CuSlice.setValue(self.sliceNum)
                elif self.item.action == 'agarTile':
                    self.SurfFilename()
                elif self.item.action == 'WriteAgar':
                    self.WriteAgar(self.item.data, self.item.payload)
                elif self.item.action == 'Init_Mosaic':
                    self.Init_Mosaic(self.item.payload)
                elif self.item.action == 'Save_mosaic':
                    self.Save_mosaic()
                elif self.item.action == 'IncrementSampleID':
                    self.IncrementSampleID()
                elif self.item.action == 'IncrementCscanNum':
                    self.IncrementCscanNum()
                elif self.item.action == 'IncrementTileNum':
                    self.IncrementTileNum()
                elif self.item.action == 'IncrementTime':
                    self.IncrementTime()
                else:
                    message = f"Unknown display/save command: {self.item.action}"
                    self.emit_status(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                if time.time()-start>1:
                    print('time for DnS:',round(time.time()-start,3))
            except Exception as error:
                message = "Display/save processing failed. This item was skipped."
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            # num+=1
            # print(num, 'th display\n')
            self.item = self.queue.get()
            
        self.emit_status("Display/save thread exited.")

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def _emit_display(self, kind: str):
        """
        Emit display payloads to GUI thread via ui_bridge.
        This thread must NOT create QPixmap or touch widgets for rendering.
        """
        bridge = getattr(self, "ui_bridge", None)
        if bridge is None:
            return

        acq_mode = self.current_acq_mode
        use_dynamic = self.ui.DynCheckBox.isChecked()

        if kind == "aline" and hasattr(self, "Aline") and np.size(self.Aline) > 0:
            bridge.aline_ready.emit({"mode": acq_mode, "aline": np.array(self.Aline, copy=True)})

        if kind == "bline" and hasattr(self, "Bline") and np.size(self.Bline) > 0:
            dyn = None
            if use_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                dyn = np.array(self.DynBline, copy=True)
            bridge.bline_ready.emit({"mode": acq_mode, "bline": np.array(self.Bline, copy=True), "dyn": dyn})

        if (
            kind == "cscan"
            and hasattr(self, "Bline")
            and hasattr(self, "AIP")
            and np.size(self.Bline) > 0
            and np.size(self.AIP) > 0
        ):
            dynb = None
            dyn = None
            if use_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                dynb = np.array(self.DynBline, copy=True)
            if use_dynamic and hasattr(self, "Dyn") and np.size(self.Dyn) > 0:
                dyn = np.array(self.Dyn, copy=True)
            bridge.cscan_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "dynb": dynb,
                    "aip": np.array(self.AIP, copy=True),
                    "dyn": dyn,
                }
            )

        if kind == "mosaic" and hasattr(self, "SampleMosaic") and np.size(self.SampleMosaic) > 0:
            bline = None
            if hasattr(self, "Bline") and np.size(self.Bline) > 0:
                bline = np.array(self.Bline, copy=True)
            bridge.mosaic_ready.emit(
                {"mode": acq_mode, "mosaic": np.array(self.SampleMosaic, copy=True), "bline": bline}
            )
            
    def print_display_counts(self, display_name = ''):
        message = f"{self.display_actions} {display_name} display update(s) completed."
        print(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        self.display_actions = 0
        
    def get_FOV_size(self, raw=False):
        # get number of Z pixels
        if not raw:
            Zpixels = self.ui.DepthRange.value()
        else:
            Zpixels = self.ui.NSamples_DH.value()#-self.ui.DelaySamples.value()-self.ui.TrimSamples.value()
        # get number of X pixels
        Xpixels = self.ui.AlinesPerBline.value()
            
        Yrpt = self.ui.BlineAVG.value()
        

        return Zpixels, Xpixels, Yrpt
            
    def Process_aline(self, data, raw = False, acq_mode=None):
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
            self.Save(data=data, raw=raw, acq_mode=acq_mode)
            
    
    def Process_bline(self, data, raw = False, dynamic = [], acq_mode=None):
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
        if self.ui.DynCheckBox.isChecked() and len(dynamic)>0:
            self.DynBline = np.transpose(dynamic)
        else:
            self.DynBline = []
            self.Dyn = []

        
        if self.ui.Save.isChecked():
            self.Save(data=data, dynamic=dynamic, raw=raw, acq_mode=acq_mode)

            
    def Process_Cscan_Dynamic(self, data, dynamic=[], acq_mode=None):
        # print(dynamic.shape)
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
        
        # print('Bline:', self.Bline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        if len(dynamic)>0:
            self.DynBline = np.transpose(dynamic)
            # print('DynBline:', self.DynBline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        else:
            self.DynBline = []
        
        if self.DynamicBlineIdx==0:
            self.AIP = np.zeros([Ypixels, Xpixels])
        if len(dynamic)>0:
            if self.DynamicBlineIdx==0:
                self.Dyn = np.zeros([Ypixels, Xpixels])
                
        # print(Bline.shape, self.AIP.shape)
        print('Ypixel: ', self.DynamicBlineIdx, ' / ', Ypixels)
        self.AIP[self.DynamicBlineIdx, :] = np.mean(Bline,1)
        if len(dynamic)>0:
            self.Dyn[self.DynamicBlineIdx, :] = np.mean(dynamic,1)
        self.DynamicBlineIdx = self.DynamicBlineIdx + 1
        
        if self.DynamicBlineIdx == Ypixels:
            self.DynamicBlineIdx = 0
        if self.ui.Save.isChecked():
            self.Save(data=data, dynamic=dynamic, acq_mode=acq_mode)
        
        
    def Process_Cscan(self, data, raw = False, acq_mode=None):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = data.shape[0]
        # Bline averaging
        if self.ui.BlineAVG.value() > 1:
            # reshape into Ypixels x Xpixels x Zpixels
            Ypixels = data.shape[0] // self.ui.BlineAVG.value()
            Cscan = data.reshape([Ypixels, self.ui.BlineAVG.value(), Xpixels,Zpixels])
            Cscan=np.mean(Cscan,1)
        else:
            Cscan = data.copy()
        # Aline averaging if needed
        if self.ui.AlineAVG.value() > 1:
            Cscan = Cscan.reshape([Ypixels, Xpixels//self.ui.AlineAVG.value(), self.ui.AlineAVG.value(), Zpixels])
            Cscan = np.mean(Cscan,2)
            Xpixels = Xpixels//self.ui.AlineAVG.value()
        # print(data[10,100,50:60])
        self.Bline = np.transpose(Cscan[Ypixels//2,:,:]).copy()# has to be first index, otherwise the memory space is not continuous
        self.AIP = np.mean(Cscan,2)
        self.DynBline = []
        self.Dyn = []
        
        if self.ui.Save.isChecked():
            self.Save(data=data, raw=raw, acq_mode=acq_mode)
            
   
    def Init_Mosaic(self, payload):
        """
        Initializes the mosaic buffer based on the physical span of all FOVs.
        payload: [fov_locs, fov_size_px, fov_size_mm]
        fov_locs: list of {'x': val, 'y': val} in mm
        fov_size_px: (width_px, height_px) e.g., (1000, 2000)
        fov_size_mm: (width_mm, height_mm) e.g., (2.0, 3.0)
        """
        fov_locs, fov_size_px, fov_size_mm = payload
        fw_px, fh_px = fov_size_px
        fw_mm, fh_mm = fov_size_mm
        
        # 1. Find the bounding box of the FOV centers
        # print(fov_locs)
        xs = [p['x'] for p in fov_locs]
        ys = [p['y'] for p in fov_locs]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # 2. Calculate how many tiles are needed in each dimension
        # We use round() to handle tiny stage step errors
        num_cols = int(round((max_x - min_x) / fw_mm)) + 1
        num_rows = int(round((max_y - min_y) / fh_mm)) + 1
        
        # 3. Initialize/Overwrite the active mosaic buffer
        mw_px = num_cols * fw_px
        mh_px = num_rows * fh_px
        self.SampleMosaic = np.ones((mh_px, mw_px), dtype=np.float32)*10
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
         message = 'Detected tile surface height: '+str(surfHeight)
         print(message)
         self.log.write(message)
 
    def Process_Mosaic(self, data, raw=False, payload=[], acq_mode=None):
        """
        Stitches the FOV into the mosaic by calculating its grid index from the anchor.
        payload: [fov_locs, fov_location]
        """
        fov_locs, fov_location = payload
        fov_x, fov_y = fov_location['x'], fov_location['y']
        xs = [p['x'] for p in fov_locs]
        ys = [p['y'] for p in fov_locs]
        min_x = min(xs)
        min_y = min(ys)
        # 1. Generate AIP projection from raw data (Y, X, Z)
        if self.ui.DynCheckBox.isChecked():
            self.Process_Cscan_Dynamic(data, acq_mode=acq_mode)
        else:
            self.Process_Cscan(data, raw, acq_mode=acq_mode)
        # 2. Calculate the Grid Index (Column and Row)
        # We determine how many FOV-widths away from the minimum X/Y we are
        col_idx = int(round((fov_x - min_x) / self.fw_mm))
        row_idx = int(round((fov_y - min_y) / self.fh_mm))
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
            # print(y1,y2,x1,x2, off_y, off_x, mh, mw)
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
            
        
    def Save(self, data=[], dynamic=[], raw=False, acq_mode=None):
        Zpixels, Xpixels, Yrpt = self.get_FOV_size(raw)
        if Zpixels != data.shape[2]:
            Zpixels = data.shape[2]
        Ypixels = data.shape[0]
        
        if acq_mode in ['FiniteAline', 'ContinuousAline']:
            filename = self.ui.DIR.toPlainText()+'/'+self.AlineFilename([Yrpt,Xpixels,Zpixels])
            for ii in range(Yrpt):
                TIFF.imwrite(filename, data[ii], append=True)
                
        elif acq_mode in ['FiniteBline', 'ContinuousBline']:
            if self.ui.DynCheckBox.isChecked() and np.size(dynamic) > 0:
                filename = self.ui.DIR.toPlainText()+'/'+self.BlineDynFilename([Yrpt,Xpixels,Zpixels])
                TIFF.imwrite(filename, dynamic, append=True)
                
            filename = self.ui.DIR.toPlainText()+'/'+self.BlineFilename([Yrpt,Xpixels,Zpixels])
            for ii in range(Yrpt):
                TIFF.imwrite(filename, data[ii], append=True)
                
        elif acq_mode in ['ContinuousCscan', 'FiniteCscan']:
            if self.ui.DynCheckBox.isChecked():
                CscanBlineFileName, CscanDynBlineFileName = self.CscanDynFilename([Yrpt,Xpixels,Zpixels])
                filename = self.ui.DIR.toPlainText()+'/'+CscanBlineFileName
                for ii in range(Yrpt):
                    TIFF.imwrite(filename, data[ii], append=True)
                
                if np.size(dynamic) > 0:
                    filename = self.ui.DIR.toPlainText()+'/'+CscanDynBlineFileName
                    TIFF.imwrite(filename, dynamic, append=True)
            else:
                filename = self.ui.DIR.toPlainText()+'/'+self.CscanFilename([Ypixels,Xpixels,Zpixels])
                for ii in range(Ypixels):
                    TIFF.imwrite(filename, data[ii], append=True)
        elif acq_mode in ['PlatePreScan', 'PlateScan', 'WellScan']:
            if self.ui.DynCheckBox.isChecked():
                BlineFileName = self.SampleDynFilename([Yrpt,Xpixels,Zpixels])
                filename = self.ui.DIR.toPlainText()+'/'+BlineFileName
                for ii in range(Yrpt):
                    TIFF.imwrite(filename, data[ii], append=True)
                
            else:
                filename = self.ui.DIR.toPlainText()+'/'+self.SampleFilename([Ypixels,Xpixels,Zpixels])
                for ii in range(Ypixels):
                    TIFF.imwrite(filename, data[ii], append=True)
        
    def IncrementCscanNum(self):
        self.CscanNum = self.CscanNum + 1
        
    def IncrementTileNum(self):
        self.tileNum = self.tileNum + 1
        
    def IncrementSampleID(self):
        self.sampleID = self.sampleID +1
        self.tileNum = 1
        # if not os.path.exists(self.ui.DIR.toPlainText()+'/aip/vol'+str(self.sliceNum)):
        #     os.mkdir(self.ui.DIR.toPlainText()+'/aip/vol'+str(self.sliceNum))
        # if not os.path.exists(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum)):
        #     os.mkdir(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum))
        # if not os.path.exists(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum)):
        #     os.mkdir(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum))
    def IncrementTime(self):
        self.sliceNum = self.sliceNum+1
        self.ui.CuSlice.setValue(self.sliceNum)
        self.tileNum = 1
        self.sampleID = 1
            
    def SampleFilename(self, shape = [0,0,0]):
        filename = 'Time-'+str(self.sliceNum)+'-sampleID-'+str(self.sampleID)+'-tile-'+str(self.tileNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        print(filename)
        self.log.write(filename)
        return filename
    
    def SampleDynFilename(self, shape):
        BlineFilename = 'Time-'+str(self.sliceNum)+'-sampleID-'+str(self.sampleID)+'-tile-'+str(self.tileNum)+'-Bline-'+str(self.DynamicBlineIdx)+'-Yrpt'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        return BlineFilename
    
    def CscanFilename(self, shape):
        filename = 'Cscan-'+str(self.CscanNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        return filename
    
    def CscanDynFilename(self, shape):
        DynBlinefilename = 'CscanDyn-'+str(self.CscanNum)+'-Y'+str(self.ui.Ypixels.value())+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        BlineFilename = 'Cscan-'+str(self.CscanNum)+'-Bline-'+str(self.DynamicBlineIdx)+'-Yrpt'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        return BlineFilename, DynBlinefilename
    
    def BlineFilename(self, shape):
        filename = 'Bline-'+str(self.BlineNum)+'-Yrpt'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.BlineNum = self.BlineNum + 1
        return filename
    
    def BlineDynFilename(self, shape):
        filename = 'BlineDyn-'+str(self.BlineNum)+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
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
            message = 'Saving took '+str(round(time.time()-start,3))+' s.'
            print(message)
            # self.ui.PrintOut.append(message)
            self.log.write(message)
        
    def WriteAgar(self, data, payload):
        [Ystep, Xstep] = payload
        filename = 'slice-'+str(self.sliceNum)+'-agarTiles X-'+str(Xstep)+'-by Y-'+str(Ystep)+'-.bin'
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        
