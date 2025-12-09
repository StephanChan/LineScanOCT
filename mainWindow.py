# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:35:04 2023

@author: admin
"""

from GUI import Ui_MainWindow
import os
from PyQt5 import QtWidgets as QW
from PyQt5.QtWidgets import  QMainWindow, QFileDialog, QWidget, QVBoxLayout
from Dialogs import  StageDialog
import PyQt5.QtCore as qc
import numpy as np
from Actions import *
from Generaic_functions import *
import traceback
# system magnification with 10X objective

global Magnification10X
Magnification10X = 6.7
# try:
#     from traits.api import HasTraits, Instance, on_trait_change
#     from traitsui.api import View, Item
#     from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
#             SceneEditor
#     from mayavi import mlab 
#     print('using maya for 3D visulizaiton')
#     maya_installed = True
# except:
#     print('maya import failed, no 3D visulization')
#     maya_installed = False


# if maya_installed:
#     class Visualization(HasTraits):
#         scene = Instance(MlabSceneModel, ())
#         def __init__(self, data):
#             HasTraits.__init__(self)
#             self.data = data
            
#         def update_contrast(self, low, high):
#             # pass
#             # calculate data according to low and high
#             # self.plot.mlab_source.scalars = data-low+1000-high
#             M=np.max(self.plot.mlab_source.scalars)
#             self.plot.current_range=(low,M*high/1000)
#             # # print(low, high)
#             # print(self.plot.current_range)
            
#         def update_data(self, data):
#             self.plot.mlab_source.scalars = data
#             # print(data.shape)
#             # print(self.plot.current_range)
#             # print(np.max(self.plot.mlab_source.scalars))
#             # print(data[1,1,:])
#             M=np.max(data)
#             self.plot.current_range=(0, M*0.2)
#             # print(self.plot.current_range)
            
#         @on_trait_change('scene.activated')
#         def inital_plot(self):
#             self.plot = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data))
#             self.scene.background=(0,0,0)
#             # print(self.plot.current_range)
#             # self.plot.lut_manager.lut_mode = 'grey'
#             # a=self.plot.lut_manager
#             # print(self.plot.mlab_source.all_trait_names())
#             # print(self.plot.all_trait_names())

#         # the layout of the dialog screated
#         view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
#                           AlinesPerBline=250, width=300, show_label=False),
#                     resizable=True # We need this to resize with the parent widget
#                     )
        
#     # The QWidget containing the visualization, this is pure PyQt4 code.
#     class MayaviQWidget(QWidget):
#         def __init__(self, data = None):
#             super().__init__()
            
#             data = np.random.random((200,1000,300))
#             self.visualization = Visualization(data)
#             layout = QVBoxLayout(self)
#             # layout.setContentsMargins(0,0,0,0)
#             # layout.setSpacing(0)
#             # The edit_traits call will generate the widget to embed.
#             self.ui = self.visualization.edit_traits(parent=self,
#                                                       kind='subpanel').control
#             layout.addWidget(self.ui)
#             # self.setLayout(layout)
#             # self.ui.setParent(self)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.LoadSettings("config.ini")
        self.setStageMinMax()
        self.Calculate_CameraWidth_settings()
        self.Calculate_Galvo_settings()
        
        #################### load configuration settings
        
        # self.Update_laser()
        # self.update_galvoXwaveform()
        # self.update_Mosaic()
        self.connectActions()
        
    def setStageMinMax(self):
        self.ui.XPosition.setMinimum(self.ui.Xmin.value())
        self.ui.XPosition.setMaximum(self.ui.Xmax.value())
        
        self.ui.YPosition.setMinimum(self.ui.Ymin.value())
        self.ui.YPosition.setMaximum(self.ui.Ymax.value())
        
        self.ui.ZPosition.setMinimum(self.ui.Zmin.value())
        self.ui.ZPosition.setMaximum(self.ui.Zmax.value())
    
    
        
    # def addMaya(self):
    #     if (maya_installed and self.use_maya):
    #         self.ui.mayavi_widget = MayaviQWidget()
    #         self.ui.mayavi_widget.setMinimumSize(qc.QSize(100, 100))
    #         self.ui.mayavi_widget.setMaximumSize(qc.QSize(1000, 1000))
    #         self.ui.mayavi_widget.setObjectName("XYZView")
            
    #         # self.ui.verticalLayout_2.removeWidget(self.ui.tmp_label)
    #         # self.ui.verticalLayout_2.addWidget(self.ui.mayavi_widget)
    #         self.ui.verticalLayout_5.replaceWidget(self.ui.MUS_mosaic, self.ui.mayavi_widget)
    #         self.ui.verticalLayout_5.removeWidget(self.ui.MUS_mosaic)
        
    def LoadSettings(self, config_filepath):
        settings = qc.QSettings(config_filepath, qc.QSettings.IniFormat)
        for ii in dir(self.ui):
            if ii == 'ACQMode':
                pass
            elif type(self.ui.__getattribute__(ii)) == QW.QComboBox:
                try:
                    self.ui.__getattribute__(ii).setCurrentText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif type(self.ui.__getattribute__(ii)) == QW.QDoubleSpinBox:
                try:
                    self.ui.__getattribute__(ii).setValue(np.float32(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif type(self.ui.__getattribute__(ii)) == QW.QSpinBox:
                try:
                    self.ui.__getattribute__(ii).setValue(np.int16(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif type(self.ui.__getattribute__(ii)) == QW.QTextEdit:
                try:
                    self.ui.__getattribute__(ii).setText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif type(self.ui.__getattribute__(ii)) == QW.QLineEdit:
                try:
                    self.ui.__getattribute__(ii).setText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif type(self.ui.__getattribute__(ii)) == QW.QPushButton:
                if settings.value(ii) in ['true', 'True']:
                    status = True
                else:
                    status = False
                try:
                    self.ui.__getattribute__(ii).setChecked(status)
                except:
                    print(ii, ' setting missing, using default...')
                
    def Calculate_CameraWidth_settings(self):
        # select camera brand
        if self.ui.Camera.currentText() == 'PhotonFocus':
            CameraPixelSize = 9 # um
            MaxHeight = 1100
        elif self.ui.Camera.currentText() == 'XingTu':
            CameraPixelSize = 6.5 # um
            MaxHeight = 1024
        else:
            status = 'camera not calibrated, abort FOV calculation'
            self.ui.statusbar.showMessage(status)
            return None, status
        # select objective magnification
        if self.ui.Objective.currentText() == '5X':
            cameraStepSize =  CameraPixelSize/Magnification10X*2 # um
            MaxXLength = cameraStepSize/1000.0*MaxHeight
        elif self.ui.Objective.currentText() == '10X':
            cameraStepSize = CameraPixelSize/Magnification10X # um
            MaxXLength = cameraStepSize/1000.0*MaxHeight
        elif self.ui.Objective.currentText() == '20X':
            cameraStepSize =  CameraPixelSize/Magnification10X/2 # um
            MaxXLength = cameraStepSize/1000.0*MaxHeight
        else:
            status = 'objective not calibrated, abort generating Galvo waveform'
            self.ui.statusbar.showMessage(status)
            return None, status
        self.ui.XStepSize.setValue(cameraStepSize)
        
        # set XLength limit
        self.ui.XLength.setMaximum(MaxXLength)
        
        # Calculate AlinesPerBline pixel numbers based on user set FOV size
        AlinesPerBline = np.uint16(np.round(self.ui.XLength.value()*1000/cameraStepSize))
        self.ui.AlinesPerBline.setValue(AlinesPerBline)
        # set offsetH limit
        # self.ui.offsetH.setMaximum((MaxHeight - Height)//2)
        self.ui.Xoffsetlength.setMaximum((MaxHeight - AlinesPerBline)//2*cameraStepSize/1000)
        self.ui.Xoffsetlength.setMinimum(-(MaxHeight - AlinesPerBline)//2*cameraStepSize/1000)
        # Calculate offsetH pixel numbers based on corrected user set offsetLength
        offsetH = (MaxHeight - AlinesPerBline)//2+np.int16(np.round(self.ui.Xoffsetlength.value()*1000/cameraStepSize))
        self.ui.offsetH.setValue(offsetH)
        
    def Calculate_Galvo_settings(self):
        # select objective magnification
        if self.ui.Objective.currentText() == '5X':
            angle2mmratio = 2.094/1.19
        elif self.ui.Objective.currentText() == '10X':
            angle2mmratio = 2.094/2/1.19
        elif self.ui.Objective.currentText() == '20X':
            angle2mmratio = 2.094/1.19/4
        else:
            status = 'objective not calibrated, abort generating Galvo waveform'
            self.ui.statusbar.showMessage(status)
            return None, status
        
        # Calculate AlinesPerBline pixel numbers based on user set FOV size
        Ypixels = np.uint16(np.round(self.ui.YLength.value()*1000/self.ui.YStepSize.value()))
        self.ui.Ypixels.setValue(Ypixels)
        self.ui.GalvoBias.setMinimum(-1)
        # Calculate offsetH pixel numbers based on corrected user set offsetLength
        self.ui.GalvoBias.setValue(self.ui.Yoffsetlength.value()/angle2mmratio)
        
    def Adjust_Bline_Height(self):
        self.ui.DepthStart.setValue(1024 - self.ui.DepthStartBar.value())
        self.ui.DepthRange.setValue(np.max([self.ui.DepthStartBar.value() - self.ui.DepthEndBar.value(),1]))
        
    def SaveSettings(self):
        settings = qc.QSettings("config.ini", qc.QSettings.IniFormat)
        for ii in dir(self.ui):
            if type(self.ui.__getattribute__(ii)) == QW.QComboBox:
                settings.setValue(ii,self.ui.__getattribute__(ii).currentText())
            elif type(self.ui.__getattribute__(ii)) == QW.QDoubleSpinBox:
                settings.setValue(ii,self.ui.__getattribute__(ii).value())
            elif type(self.ui.__getattribute__(ii)) == QW.QSpinBox:
                settings.setValue(ii,self.ui.__getattribute__(ii).value())
            elif type(self.ui.__getattribute__(ii)) == QW.QTextEdit:
                settings.setValue(ii,self.ui.__getattribute__(ii).toPlainText())
            elif type(self.ui.__getattribute__(ii)) == QW.QLineEdit:
                settings.setValue(ii,self.ui.__getattribute__(ii).text())
            elif type(self.ui.__getattribute__(ii)) == QW.QPushButton:
                settings.setValue(ii,self.ui.__getattribute__(ii).isChecked())
            
    def connectActions(self):
        self.ui.Objective.currentTextChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.Objective.currentTextChanged.connect(self.Calculate_Galvo_settings)
        self.ui.Camera.currentTextChanged.connect(self.Calculate_CameraWidth_settings)
        
        self.ui.XLength.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.Xoffsetlength.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.YLength.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.Yoffsetlength.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.YStepSize.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.BlineAVG.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.DepthStartBar.valueChanged.connect(self.Adjust_Bline_Height)
        self.ui.DepthEndBar.valueChanged.connect(self.Adjust_Bline_Height)
        
        # self.ui.Objective.currentTextChanged.connect(self.update_galvoXwaveform)
        # self.ui.PreClock.valueChanged.connect(self.update_galvoXwaveform)
        # self.ui.PostClock.valueChanged.connect(self.update_galvoXwaveform)
        # self.ui.Ysteps.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.YStepSize.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.BlineAVG.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.YBias.valueChanged.connect(self.update_galvoYwaveform)
        
        
        self.ui.XStart.valueChanged.connect(self.update_Mosaic)
        self.ui.XStop.valueChanged.connect(self.update_Mosaic)
        self.ui.YStart.valueChanged.connect(self.update_Mosaic)
        self.ui.YStop.valueChanged.connect(self.update_Mosaic)
        self.ui.Overlap.valueChanged.connect(self.update_Mosaic)
        
        # self.ui.ImageZDepth.valueChanged.connect(self.Calculate_ImageDepth)
        # self.ui.ImageZnumber.valueChanged.connect(self.Calculate_ImageDepth)
        
        # self.ui.SliceZStart.valueChanged.connect(self.Calculate_SliceDepth)
        # self.ui.SliceZDepth.valueChanged.connect(self.Calculate_SliceDepth)
        # self.ui.SliceZnumber.valueChanged.connect(self.Calculate_SliceDepth)
        # change brigtness and contrast
        # self.ui.ACQMode.currentIndexChanged.connect(self.Adjust_contrast)
        # self.ui.FFTDevice.currentIndexChanged.connect(self.Update_scale)

        # update laser model
        # self.ui.Laser.currentIndexChanged.connect(self.Update_laser)
        self.ui.Save.clicked.connect(self.chooseDir)
        self.ui.LoadInD.clicked.connect(self.chooseInD)
        self.ui.LoadBG.clicked.connect(self.chooseBackground)
        self.ui.ConfigButton.clicked.connect(self.LoadConfig)

        self.ui.Xmax.valueChanged.connect(self.setStageMinMax)
        self.ui.Xmin.valueChanged.connect(self.setStageMinMax)
        self.ui.Ymin.valueChanged.connect(self.setStageMinMax)
        self.ui.Ymax.valueChanged.connect(self.setStageMinMax)
        self.ui.Zmin.valueChanged.connect(self.setStageMinMax)
        self.ui.Zmax.valueChanged.connect(self.setStageMinMax)
        
        self.ui.LoadSurface.clicked.connect(self.chooseSurfaceFile)
        # self.ui.LoadDarkField.clicked.connect(self.chooseDarkFieldFile)
        # self.ui.LoadFlatField.clicked.connect(self.chooseFlatFieldFile)

    def chooseSurfaceFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                   "select surface file",  
                                   os.getcwd(), # 起始路径 
                                   "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
           print("\n use default")
           return
        self.ui.Surf_DIR.setText(fileName_choose)
        
    def chooseDarkFieldFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                   "select dark field file",  
                                   os.getcwd(), # 起始路径 
                                   "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
           print("\n use default")
           return
        self.ui.DarkField_DIR.setText(fileName_choose)
    
    def chooseFlatFieldFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                   "select flat field file",  
                                   os.getcwd(), # 起始路径 
                                   "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
           print("\n use default")
           return
        self.ui.FlatField_DIR.setText(fileName_choose)

    def chooseDir(self):
        if self.ui.Save.isChecked():
            
             dir_choose = QFileDialog.getExistingDirectory(self,  
                                         "select saving directory",  
                                         os.getcwd()) # 起始路径
        
             if dir_choose == "":
                 print("\n use default")
                 return
             self.ui.DIR.setText(dir_choose)
             
    def LoadConfig(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                   "select config file",  
                                   os.getcwd(), # 起始路径 
                                   "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
           print("\n use default")
           return
        
        try:
            self.LoadSettings(fileName_choose)
        except Exception as error:
            print('settings reload failed, using default settings')
            print(traceback.format_exc())
        
    def chooseInD(self):
         dir_choose = QFileDialog.getExistingDirectory(self,  
                                     "select saving directory",  
                                     os.getcwd()) # 起始路径

         if dir_choose == "":
             print("\n use default")
             return
         self.ui.InD_DIR.setText(dir_choose)
        
    def chooseBackground(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                   "select background file",  
                                   os.getcwd(), # 起始路径 
                                   "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
           print("\n use default")
           return

        self.ui.BG_DIR.setText(fileName_choose)
        # self.update_background()
        
    def update_galvoXwaveform(self):
        # calculate the total X range
        Xrange=self.ui.Xsteps.value()*self.ui.XStepSize.value()/1000
        # update X range in lable
        self.ui.XrangeLabel.setText('X(mm): '+str(Xrange))
        self.ui.YrangeLabel.setText('Y(mm): '+str(self.ui.Ysteps.value()*self.ui.YStepSize.value()/1000))
        # generate waveform
        DOwaveform, AOwaveform, status = GenAODO(mode='RptBline', \
                                                 XStepSize = self.ui.XStepSize.value(), \
                                                 XSteps = self.ui.Xsteps.value(), \
                                                 AVG = self.ui.AlineAVG.value(), \
                                                 obj = self.ui.Objective.currentText(),\
                                                 preclocks = self.ui.PreClock.value(),\
                                                 postclocks = self.ui.PostClock.value(), \
                                                 YStepSize = self.ui.YStepSize.value(), \
                                                 YSteps =  self.ui.Ysteps.value(), \
                                                 BVG = self.ui.BlineAVG.value(),\
                                                 Galvo_bias = self.ui.GalvoBias.value(),\
                                                 DISTANCE = self.ui.Xmm.value(), \
                                                 STEPS = self.ui.Xdevides.value())
        # show generating waveform result
        #print(self.Xwaveform)
        # current_message = self.ui.statusbar.currentMessage()
        # self.ui.statusbar.showMessage(current_message+status)
        if np.any(AOwaveform):
            pixmap = LinePlot(AOwaveform, DOwaveform, np.min(AOwaveform),np.max(AOwaveform))
            # clear content on the waveformLabel
            self.ui.XwaveformLabel.clear()
            # update iamge on the waveformLabel
            self.ui.XwaveformLabel.setPixmap(pixmap)
        
    # def update_galvoYwaveform(self):
    #     # calculate the total X range
    #     Yrange=self.ui.Ysteps.value()*self.ui.YStepSize.value()/1000
    #     # update X range in lable
    #     self.ui.YrangeLabel.setText('Y range(mm): '+str(Yrange))
    #     # generate waveform
    #     self.Ywaveform, status = GenGalvoWave(self.ui.YStepSize.value(),\
    #                                     self.ui.Ysteps.value(),\
    #                                     self.ui.BlineAVG.value(),\
    #                                     self.ui.YBias.value(),\
    #                                     self.ui.Objective.currentText())
    #     # show generating waveform result
    #     self.ui.statusbar.showMessage(status)
    #     if self.Ywaveform != None:
    #         pixmap = LinePlot(self.Xwaveform)
    #         # clear content on the waveformLabel
    #         self.ui.YwaveformLabel.clear()
    #         # update iamge on the waveformLabel
    #         self.ui.YwaveformLabel.setPixmap(pixmap)
        
    def update_Mosaic(self):
        self.Mosaic, status = GenMosaic_XGalvo(self.ui.XStart.value(),\
                                        self.ui.XStop.value(),\
                                        self.ui.YStart.value(),\
                                        self.ui.YStop.value(),\
                                        self.ui.Xsteps.value()*self.ui.XStepSize.value()/1000,\
                                        self.ui.Overlap.value())
        self.ui.statusbar.showMessage(status)
        if self.Mosaic is not None:
            mosaic=np.zeros([2,len(self.Mosaic)*2])
            for ii, element in enumerate(self.Mosaic):
                if ii%2 == 0:
                    mosaic[0,ii*2] = element.x
                    mosaic[1,ii*2] = element.ystart
                    mosaic[0,ii*2+1] = element.x
                    mosaic[1,ii*2+1] = element.ystop
                else:
                    mosaic[0,ii*2] = element.x
                    mosaic[1,ii*2] = element.ystop
                    mosaic[0,ii*2+1] = element.x
                    mosaic[1,ii*2+1] = element.ystart
                
            pixmap = ScatterPlot(mosaic)
            # clear content on the waveformLabel
            self.ui.MosaicLabel.clear()
            # update iamge on the waveformLabel
            self.ui.MosaicLabel.setPixmap(pixmap)
        
    def Calculate_ImageDepth(self):
        self.image_depths = GenAlinesPerBlines(self.ui.ImageZStart.value(),\
                                       self.ui.ImageZDepth.value(),\
                                       self.ui.ImageZnumber.value())
        #print(self.image_depths)
        #self.ui.statusbar.showMessage(self.image_depths)
            
    def Calculate_SliceDepth(self):
        self.slice_depths = GenAlinesPerBlines(self.ui.SliceZStart.value(),\
                                       self.ui.SliceZDepth.value(),\
                                       self.ui.SliceZnumber.value())
        #print(self.slice_depths)
        #self.ui.statusbar.showMessage(self.image_depths)
        
    def Update_laser(self):
        if self.ui.Laser.currentText() == 'Axsun100k':
            self.Aline_frq = 100000
        elif self.ui.Laser.currentText() == 'Thorlabs200k':
            self.Aline_frq = 200000
        else:
            self.ui.statusbar.showMessage('Laser invalid!!!')
    
        
        