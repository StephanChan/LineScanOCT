# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:51:20 2023

@author: admin
"""
###########################################

global SIM
SIM = False
###########################################
from PyQt5.QtCore import  QThread

try:
    import nidaqmx as ni
    from nidaqmx.constants import AcquisitionType as Atype
    from nidaqmx.constants import Edge, ProductCategory
    
    def get_terminal_name_with_dev_prefix(task: ni.Task, terminal_name: str) -> str:
        """Gets the terminal name with the device prefix.

        Args:
            task: Specifies the task to get the device name from.
            terminal_name: Specifies the terminal name to get.

        Returns:
            Indicates the terminal name with the device prefix.
        """
        for device in task.devices:
            if device.product_category not in [
                ProductCategory.C_SERIES_MODULE,
                ProductCategory.SCXI_MODULE,
            ]:
                return f"/{device.name}/{terminal_name}"

        raise RuntimeError("Suitable device not found in task.")
except:
    SIM = True
    
from Generaic_functions import GenAODO, LinePlot
import time
import traceback
import numpy as np


    
    
class AODOThread(QThread):
    def __init__(self):
        super().__init__()
        self.AOtask = None
        self.DOtask = None
        
    
    def run(self):
        self.Init_all_termial()
        self.StagebackQueue.get()
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != 'exit':
            try:
                if self.item.action == 'Xmove2':
                    self.DirectMove(axis = 'X')
                elif self.item.action == 'Ymove2':
                    self.DirectMove(axis = 'Y')
                elif self.item.action == 'Zmove2':
                    self.DirectMove(axis = 'Z')
                elif self.item.action == 'XUP':
                    self.StepMove(axis = 'X', Direction = 'UP')
                elif self.item.action == 'YUP':
                    self.StepMove(axis = 'Y', Direction = 'UP')
                elif self.item.action == 'ZUP':
                    self.StepMove(axis = 'Z', Direction = 'UP')
                elif self.item.action == 'XDOWN':
                    self.StepMove(axis = 'X', Direction = 'DOWN')
                elif self.item.action == 'YDOWN':
                    self.StepMove(axis = 'Y', Direction = 'DOWN')
                elif self.item.action == 'ZDOWN':
                    self.StepMove(axis = 'Z', Direction = 'DOWN')
                    
                elif self.item.action == 'Init':
                    self.Init_all_termial()
                elif self.item.action == 'Uninit':
                    self.Uninit()
                elif self.item.action == 'ConfigTask':
                    self.ConfigTask()
                elif self.item.action == 'StartTask':
                    self.StartTask()
                elif self.item.action == 'StopTask':
                    self.StopTask()
                elif self.item.action == 'tryStopTask':
                    self.tryStopTask()
                elif self.item.action == 'CloseTask':
                    self.CloseTask()
                elif self.item.action == 'centergalvo':
                    self.centergalvo()

                else:
                    message = 'AODO thread is doing something undefined: '+self.item.action
                    self.ui.statusbar.showMessage(message)
                    print(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
            except Exception:
                message = "\nAn error occurred,"+" skip the AODO action\n"
                print(message)
                self.ui.statusbar.showMessage(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()
        self.ui.statusbar.showMessage('AODO thread successfully exited')

    def Init_all_termial(self):
        # Galvo terminal
        self.GalvoAO = self.ui.AODOboard.toPlainText()+'/'+self.ui.GalvoAO.currentText()
        
        # synchronized DO terminal
        self.SyncDO = '/'+self.ui.AODOboard.toPlainText()+'/'+self.ui.SyncDO.currentText()

        self.ui.Xcurrent.setValue(self.ui.XPosition.value())
        self.ui.Ycurrent.setValue(self.ui.YPosition.value())
        self.ui.Zcurrent.setValue(self.ui.ZPosition.value())
        message = "Stage position updated..."

        self.ui.statusbar.showMessage(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        print(message)
        self.StagebackQueue.put(0)
    
    def Uninit(self):
        if not (SIM or self.SIM):
            pass
        self.StagebackQueue.put(0)
        
    def ConfigTask(self):
        # Generate waveform
        DOwaveform,AOwaveform,status = GenAODO(mode=self.ui.ACQMode.currentText(), \
                                                 obj = self.ui.Objective.currentText(),\
                                                 postclocks = self.ui.FlyBack.value(), \
                                                 YStepSize = self.ui.YStepSize.value(), \
                                                 YSteps =  self.ui.Ypixels.value(), \
                                                 BVG = self.ui.BlineAVG.value(),\
                                                 Galvo_bias = self.ui.GalvoBias.value())
        pixmap = LinePlot(AOwaveform,DOwaveform, np.min([np.min(AOwaveform),0]), np.max([np.max(AOwaveform),1]))
        # clear content on the waveformLabel
        self.ui.XwaveformLabel.clear()
        # update iamge on the waveformLabel
        self.ui.XwaveformLabel.setPixmap(pixmap)
        if not (SIM or self.SIM): # if not running simulation mode
            
            ######################################################################################
            # init AO task
            self.AOtask = ni.Task('AOtask')
            # Config channel and vertical
            self.AOtask.ao_channels.add_ao_voltage_chan(physical_channel=self.GalvoAO, \
                                                  min_val=- 10.0, max_val=10.0, \
                                                  units=ni.constants.VoltageUnits.VOLTS)
            # depending on whether continuous or finite, config clock and mode
            if self.mode in ['RptAline', 'RptBline']:
                mode =  Atype.CONTINUOUS
            else:
                mode =  Atype.FINITE
            self.AOtask.timing.cfg_samp_clk_timing(rate=np.floor(self.ui.FrameRate.value())*2, \
                                                   # source=self.ClockTerm, \
                                                   # active_edge= Edge.RISING,\
                                                   sample_mode=mode,samps_per_chan=len(AOwaveform))
            # # Config start mode
            # self.AOtask.triggers.start_trigger.cfg_dig_edge_start_trig(self.AODOTrig)
            terminal_name = get_terminal_name_with_dev_prefix(self.AOtask, "ao/StartTrigger")
            # write waveform and start
            self.AOtask.write(AOwaveform, auto_start = False)
            # self.AOtask.start()

            # config DO task
            self.DOtask = ni.Task('DOtask')
            self.DOtask.do_channels.add_do_chan(lines=self.SyncDO)
            self.DOtask.timing.cfg_samp_clk_timing(rate=np.floor(self.ui.FrameRate.value())*2, \
                                                   # source=self.ClockTerm, \
                                                   # active_edge= Edge.RISING,\
                                                   sample_mode=mode,samps_per_chan=len(DOwaveform))
           

            # self.DOtask.triggers.start_trigger.cfg_dig_edge_start_trig(self.AODOTrig)
            self.DOtask.triggers.start_trigger.cfg_dig_edge_start_trig(terminal_name)
            self.DOtask.write(DOwaveform, auto_start = False)
            # self.DOtask.start()
            # print(DOwaveform.shape)
            # steps = np.sum(DOwaveform)/25000.0*2/pow(2,1)
            # message = 'distance per Cscan: '+str(steps)+'mm'
            # # self.ui.PrintOut.append(message)
            # print(message)
            # self.log.write(message)
        return 'AODO configuration success'
        
    def StartTask(self):
        if not (SIM or self.SIM):
            self.DOtask.start()
            self.AOtask.start()
        self.StagebackQueue.put(0)

    def StopTask(self):
        if not (SIM or self.SIM):
            # self.AOtask.wait_until_done(timeout = 60)
            self.AOtask.stop()
            self.DOtask.stop()
        
    def tryStopTask(self):
        if not (SIM or self.SIM):
            try:
                self.AOtask.wait_until_done(timeout = 0.5)
            except:
                self.AOtask.stop()
                self.DOtask.stop()

    
    def CloseTask(self):
        if not (SIM or self.SIM):
            self.AOtask.close()
            self.DOtask.close()
        self.StagebackQueue.put(0)

    
    def centergalvo(self):
        if not (SIM or self.SIM):
            with ni.Task('AO task') as AOtask:
                AOtask.ao_channels.add_ao_voltage_chan(physical_channel=self.GalvoAO, \
                                                      min_val=- 10.0, max_val=10.0, \
                                                      units=ni.constants.VoltageUnits.VOLTS)
                AOtask.write(self.ui.GalvoBias.value(), auto_start = True)
                AOtask.wait_until_done(timeout = 1)
                AOtask.stop()

        
    def Move(self, axis = 'X'):
        
        # if axis == 'X':
        #     self.ui.Xcurrent.setValue(self.ui.Xcurrent.value()+distance)
        #     # self.ui.XPosition.setValue(self.Xpos)
        # elif axis == 'Y':
        #     self.ui.Ycurrent.setValue(self.ui.Ycurrent.value()+distance)
        #     # self.ui.YPosition.setValue(self.Ypos)
        # elif axis == 'Z':
        #     self.ui.Zcurrent.setValue(self.ui.Zcurrent.value()+distance)
        #     # self.ui.ZPosition.setValue(self.Zpos)
        message = 'X :'+str(self.ui.Xcurrent.value())+' Y :'+str(round(self.ui.Ycurrent.value(),2))+' Z :'+str(self.ui.Zcurrent.value())
        print(message)
        self.log.write(message)
        
    def DirectMove(self, axis):
        self.Move(axis)
        self.StagebackQueue.put(0)
        
    def StepMove(self, axis, Direction):
        if axis == 'X':
            distance = self.ui.Xstagestepsize.value() if Direction == 'UP' else -self.ui.Xstagestepsize.value() 
            self.ui.XPosition.setValue(self.ui.Xcurrent.value()+distance)
            self.Move(axis)
            self.StagebackQueue.put(0)
        elif axis == 'Y':
            distance = self.ui.Ystagestepsize.value() if Direction == 'UP' else -self.ui.Ystagestepsize.value() 
            self.ui.YPosition.setValue(self.ui.Ycurrent.value()+distance)
            self.Move(axis)
            self.StagebackQueue.put(0)
        elif axis == 'Z':
            distance = self.ui.Zstagestepsize.value() if Direction == 'UP' else -self.ui.Zstagestepsize.value() 
            self.ui.ZPosition.setValue(self.ui.Zcurrent.value()+distance)
            self.Move(axis)
            self.StagebackQueue.put(0)
            