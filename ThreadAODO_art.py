# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 19:11:24 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 15:10:40 2025

@author: admin
"""

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
    import sys
    sys.path.append(r"C:\\Program Files (x86)\\ART Technology\\ART-DAQ\\Samples\\Python\\LIB\\")

    if "ni" not in sys.modules:
        import artdaq as ni
        from artdaq.constants import AcquisitionType as Atype
        from artdaq.constants import Edge, ProductCategory, RegenerationMode, Signal
    else:
        # Module already loaded; you can safely use it.
        pass
except:
    print('ART digitizer init failed, using simulation')
    SIM = True

try:
    from StageControl import ZC300MotorController
    motors = ZC300MotorController()
except:
    print('stage init failed, using simulation')
    SIM = True

from Generaic_functions import GenAODO
from HardwareSpecs import (
    AODO_AO_VOLTAGE_MAX,
    AODO_AO_VOLTAGE_MIN,
    AODO_DEFAULT_FRAME_RATE,
    AODO_TRIGGER_IN_PFI,
    AODO_TRIGGER_OUT_PFI,
    digital_line_mask,
    get_camera_spec,
    get_stage_axis_spec,
)
import time
import traceback



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
                elif self.item.action == 'XHome':
                    self.Home(axis = 'X')
                elif self.item.action == 'YHome':
                    self.Home(axis = 'Y')
                elif self.item.action == 'ZHome':
                    self.Home(axis = 'Z')
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
                    message = f"Unknown stage/galvo command: {self.item.action}"
                    self.emit_status(message)
                    print(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
            except Exception:
                message = "Stage/galvo command failed. This action was skipped."
                print(Exception)
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()
        self.emit_status("Stage/galvo thread exited.")

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def Init_all_termial(self):
        # Galvo terminal
        self.GalvoAO = self.ui.AODOboard.toPlainText()+'/'+self.ui.GalvoAO.currentText()

        # synchronized DO terminal
        self.SyncDO = self.ui.AODOboard.toPlainText()+'/'+self.ui.SyncDO.currentText()
        self.Trigger_out = '/'+ self.ui.AODOboard.toPlainText()+'/'+AODO_TRIGGER_OUT_PFI
        self.Trigger_in ='/'+ self.ui.AODOboard.toPlainText()+'/'+AODO_TRIGGER_IN_PFI
        # print(self.GalvoAO, self.SyncDO)
        self.ui.Xcurrent.setValue(self.ui.XPosition.value())
        self.ui.Ycurrent.setValue(self.ui.YPosition.value())
        self.ui.Zcurrent.setValue(self.ui.ZPosition.value())
        if not (SIM or self.SIM):
            # initialize stages
            x_axis = get_stage_axis_spec('X')
            y_axis = get_stage_axis_spec('Y')
            z_axis = get_stage_axis_spec('Z')
            motors.configure_axis(x_axis.axis_index)
            motors.configure_axis(y_axis.axis_index)
            motors.configure_axis(z_axis.axis_index)
            motors.set_init_speed(x_axis.axis_index, x_axis.init_speed_mm_s)
            motors.set_move_speed(x_axis.axis_index, self.ui.XSpeed.value())
            motors.set_acceleration(x_axis.axis_index, self.ui.XAccelerate.value())
            motors.set_home_speed(x_axis.axis_index, self.ui.XSpeed.value())
            motors.set_position(x_axis.axis_index,-self.ui.XPosition.value())

            motors.set_init_speed(y_axis.axis_index, y_axis.init_speed_mm_s)
            motors.set_move_speed(y_axis.axis_index, self.ui.YSpeed.value())
            motors.set_acceleration(y_axis.axis_index, self.ui.YAccelerate.value())
            motors.set_home_speed(y_axis.axis_index, self.ui.YSpeed.value())
            motors.set_position(y_axis.axis_index,-self.ui.YPosition.value())

            motors.set_init_speed(z_axis.axis_index, z_axis.init_speed_mm_s)
            motors.set_move_speed(z_axis.axis_index, self.ui.ZSpeed.value())
            motors.set_acceleration(z_axis.axis_index, self.ui.ZAccelerate.value())
            motors.set_home_speed(z_axis.axis_index, self.ui.ZSpeed.value())
            motors.set_position(z_axis.axis_index,-self.ui.ZPosition.value())

        message = "Stage position updated."

        self.emit_status(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        print(message)
        self.StagebackQueue.put(0)

    def Uninit(self):
        if not (SIM or self.SIM):
            motors.set_enable(get_stage_axis_spec('X').axis_index, False)
            motors.set_enable(get_stage_axis_spec('Y').axis_index, False)
            motors.set_enable(get_stage_axis_spec('Z').axis_index, False)

            # pass
        self.StagebackQueue.put(0)

    def ConfigTask(self):
        # Generate waveform
        self.DOwaveform,self.AOwaveform,status = GenAODO(mode=self.ui.ACQMode.currentText(), \
                                                 obj = self.ui.Objective.currentText(),\
                                                 postclocks = self.ui.FlyBack.value(), \
                                                 YStepSize = self.ui.YStepSize.value(), \
                                                 YSteps =  self.ui.Ypixels.value(), \
                                                 BVG = self.ui.BlineAVG.value(),\
                                                 Galvo_bias = self.ui.GalvoBias.value())
        self.DOwaveform = self.DOwaveform * digital_line_mask(self.ui.SyncDO.currentText())
        if not self.ui.DynCheckBox.isChecked():
            self.ui_bridge.aodo_waveform_ready.emit({
                "ao_waveform": self.AOwaveform,
                "do_waveform": self.DOwaveform,
            })
        if not (SIM or self.SIM): # if not running simulation mode
            camera_name = self.ui.Camera.currentText()
            camera = get_camera_spec(camera_name)
            if camera_name == 'Daheng' and camera is not None:
                frameRate = self.ui.FrameRate_DH.value() * camera.frame_rate_multiplier
            elif camera_name == 'PhotonFocus' and camera is not None:
                frameRate = self.ui.FrameRate.value() * camera.frame_rate_multiplier
            else:
                frameRate = AODO_DEFAULT_FRAME_RATE
            ######################################################################################
            # init AO task
            self.AOtask = ni.Task('AOtask')
            # Config channel and vertical
            self.AOtask.ao_channels.add_ao_voltage_chan(physical_channel=self.GalvoAO, \
                                                  min_val=AODO_AO_VOLTAGE_MIN, max_val=AODO_AO_VOLTAGE_MAX, \
                                                  units=ni.constants.VoltageUnits.VOLTS)
            # depending on whether continuous or finite, config clock and mode
            if self.ui.ACQMode.currentText() in ['ContinuousAline', 'ContinuousBline','ContinuousCscan']:
                mode =  Atype.CONTINUOUS
            else:
                mode =  Atype.FINITE
            self.AOtask.timing.cfg_samp_clk_timing(rate=frameRate, \
                                                   # source=self.ClockTerm, \
                                                   # active_edge= Edge.RISING,\
                                                   sample_mode=mode,samps_per_chan=len(self.AOwaveform))
            # # Config start mode
            # self.AOtask.triggers.start_trigger.cfg_dig_edge_start_trig(self.AODOTrig)
            self.AOtask.export_signals.export_signal(signal_id = Signal.START_TRIGGER, output_terminal = self.Trigger_out)
            # write waveform and start

            # self.AOtask.start()
            # actual_sampling_rate = self.AOtask.timing.samp_clk_rate
            # print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")
            # config DO task
            self.DOtask = ni.Task('DOtask')
            self.DOtask.do_channels.add_do_chan(lines=self.SyncDO)
            self.DOtask.timing.cfg_samp_clk_timing(rate=frameRate, \
                                                   # source=self.ClockTerm, \
                                                   # active_edge= Edge.RISING,\
                                                   sample_mode=mode,samps_per_chan=len(self.DOwaveform))

            # self.DOtask.triggers.start_trigger.cfg_dig_edge_start_trig(self.AODOTrig)
            self.DOtask.triggers.start_trigger.cfg_dig_edge_start_trig(self.Trigger_in)
            # self.DOtask.triggers.sync_type.SLAVE = True

            # self.DOtask.start()
            # print(DOwaveform.shape)
            # steps = np.sum(DOwaveform)/25000.0*2/pow(2,1)
            # message = 'distance per Cscan: '+str(steps)+'mm'
            # # self.ui.PrintOut.append(message)
            # print(message)
            # self.log.write(message)
        self.StagebackQueue.put(0)
        return 'AODO configuration success'

    def StartTask(self):
        if not (SIM or self.SIM):
            self.AOtask.write(self.AOwaveform, auto_start = False)
            self.DOtask.write(self.DOwaveform, auto_start = False)
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
                try:
                    self.AOtask.close()
                except:
                    pass
                try:
                    self.DOtask.close()
                except:
                    pass


    def CloseTask(self):
        if not (SIM or self.SIM):
            try:
                self.AOtask.close()
            except:
                pass
            try:
                self.DOtask.close()
            except:
                pass
        self.StagebackQueue.put(0)


    def centergalvo(self):
        if not (SIM or self.SIM):
            with ni.Task('AOtask') as AOtask:
                AOtask.ao_channels.add_ao_voltage_chan(physical_channel=self.GalvoAO, \
                                                      min_val=AODO_AO_VOLTAGE_MIN, max_val=AODO_AO_VOLTAGE_MAX, \
                                                      units=ni.constants.VoltageUnits.VOLTS)
                AOtask.write(self.ui.GalvoBias.value(), auto_start = True)
                AOtask.wait_until_done(timeout = 1)
                AOtask.stop()


    def Move(self, axis = 'X'):


        if axis =='X':
            x_axis = get_stage_axis_spec('X')
            motors.set_enable(x_axis.axis_index,True)
            motors.set_move_speed(x_axis.axis_index, self.ui.XSpeed.value())
            motors.set_acceleration(x_axis.axis_index, self.ui.XAccelerate.value())
            distance = self.ui.XPosition.value() - self.ui.Xcurrent.value()
            motors.move_relative(x_axis.axis_index, distance)
            self.ui.Xcurrent.setValue(self.ui.Xcurrent.value()+distance)

        if axis =='Y':
            y_axis = get_stage_axis_spec('Y')
            motors.set_enable(y_axis.axis_index,True)
            motors.set_move_speed(y_axis.axis_index, self.ui.YSpeed.value())
            motors.set_acceleration(y_axis.axis_index, self.ui.YAccelerate.value())
            distance = self.ui.YPosition.value() - self.ui.Ycurrent.value()
            # print(distance, self.ui.YPosition.value())
            motors.move_relative(y_axis.axis_index, distance)
            # print(self.ui.Ycurrent.value()+distance, self.ui.YPosition.value())
            self.ui.Ycurrent.setValue(self.ui.Ycurrent.value()+distance)

        if axis =='Z':
            z_axis = get_stage_axis_spec('Z')
            motors.set_enable(z_axis.axis_index,True)
            motors.set_move_speed(z_axis.axis_index, self.ui.ZSpeed.value())
            motors.set_acceleration(z_axis.axis_index, self.ui.ZAccelerate.value())
            distance = self.ui.ZPosition.value() - self.ui.Zcurrent.value()
            motors.move_relative(z_axis.axis_index, distance)
            self.ui.Zcurrent.setValue(self.ui.Zcurrent.value()+distance)

        # if axis == 'X':
        #     self.ui.Xcurrent.setValue(self.ui.Xcurrent.value()+distance)
        #     # self.ui.XPosition.setValue(self.Xpos)
        # elif axis == 'Y':
        #     self.ui.Ycurrent.setValue(self.ui.Ycurrent.value()+distance)
        #     # self.ui.YPosition.setValue(self.Ypos)
        # elif axis == 'Z':
        #     self.ui.Zcurrent.setValue(self.ui.Zcurrent.value()+distance)
        #     # self.ui.ZPosition.setValue(self.Zpos)
        # message = 'X :'+str(self.ui.Xcurrent.value())+' Y :'+str(round(self.ui.Ycurrent.value(),2))+' Z :'+str(self.ui.Zcurrent.value())
        # print(message)
        # self.log.write(message)

    def DirectMove(self, axis):
        if not (SIM or self.SIM):
            self.Move(axis)
        else:
            time.sleep(1)
        message = f"Stage position: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        print(message)
        self.log.write(message)
        self.StagebackQueue.put(0)

    def StepMove(self, axis, Direction):
        if not (SIM or self.SIM):
            if axis == 'X':
                distance = self.ui.Xstagestepsize.value() if Direction == 'UP' else -self.ui.Xstagestepsize.value()
                self.ui.XPosition.setValue(self.ui.Xcurrent.value()+distance)
                self.Move(axis)
            elif axis == 'Y':
                distance = self.ui.Ystagestepsize.value() if Direction == 'UP' else -self.ui.Ystagestepsize.value()
                self.ui.YPosition.setValue(self.ui.Ycurrent.value()+distance)
                self.Move(axis)
            elif axis == 'Z':
                distance = self.ui.Zstagestepsize.value() if Direction == 'UP' else -self.ui.Zstagestepsize.value()
                self.ui.ZPosition.setValue(self.ui.Zcurrent.value()+distance)
                self.Move(axis)
        else:
            time.sleep(1)
        message = f"Stage position: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        print(message)
        self.log.write(message)
        self.StagebackQueue.put(0)

    def Home(self, axis):
        if not (SIM or self.SIM):
            if axis == 'X':
                # self.ui.XPosition.setValue(0)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
                x_axis = get_stage_axis_spec('X')
                motors.set_home_speed(x_axis.axis_index, self.ui.XSpeed.value())
                motors.home(x_axis.axis_index)
                self.ui.XPosition.setValue(0)
                self.ui.Xcurrent.setValue(0)
                # self.ui.XPosition.setValue(1)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
            elif axis == 'Y':
                # self.ui.YPosition.setValue(0)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
                y_axis = get_stage_axis_spec('Y')
                motors.set_home_speed(y_axis.axis_index, self.ui.YSpeed.value())
                motors.home(y_axis.axis_index)
                self.ui.YPosition.setValue(0)
                self.ui.Ycurrent.setValue(0)
                # self.ui.YPosition.setValue(1)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
            elif axis == 'Z':
                # self.ui.ZPosition.setValue(0)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
                z_axis = get_stage_axis_spec('Z')
                motors.set_home_speed(z_axis.axis_index, self.ui.ZSpeed.value())
                motors.home(z_axis.axis_index)
                self.ui.ZPosition.setValue(0)
                self.ui.Zcurrent.setValue(0)
                # self.ui.ZPosition.setValue(1)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
        else:
            time.sleep(2)
        message = f"Stage position: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        print(message)
        self.log.write(message)
        self.StagebackQueue.put(0)
