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
ARTDAQ_SIM = False
STAGE_SIM = False
###########################################
from PyQt5.QtCore import QThread

try:
    import sys
    ARTDAQ_PYTHON_LIB_DIR = r"C:\\Program Files (x86)\\ART Technology\\ART-DAQ\\Samples\\Python\\LIB\\"
    sys.path.append(ARTDAQ_PYTHON_LIB_DIR)

    import artdaq as ni
    from artdaq.constants import AcquisitionType as Atype
    from artdaq.constants import Edge, ProductCategory, RegenerationMode, Signal
    try:
        from artdaq.constants import LineGrouping
    except Exception:
        LineGrouping = None
except Exception as error:
    print(
        "ART-DAQ SDK import failed. The configured ART-DAQ Python library directory may be wrong: "
        f"{ARTDAQ_PYTHON_LIB_DIR}. Import error: {error}. Using simulation."
    )
    ARTDAQ_SIM = True
    LineGrouping = None

try:
    from StageControl import ZC300MotorController
    motors = ZC300MotorController()
except Exception as error:
    print(f"stage init failed, using stage simulation. Import/init error: {error}")
    STAGE_SIM = True
    motors = None

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
import numpy as np
import threading

SERVO_DEVICE_LINE = "Galvo/port0/line1"
SERVO_SAMPLE_RATE_HZ = 10_000
SERVO_PWM_FREQUENCY_HZ = 50
SERVO_START_ANGLE_DEG = 0.0
SERVO_MOVE_DELTA_DEG = 30.0
SERVO_MOVE_TIME_S = 0.7
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500



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
                elif self.item.action == 'rotate_servo_out':
                    self.rotate_servo_out()
                elif self.item.action == 'rotate_servo_back':
                    self.rotate_servo_back()
                elif self.item.action == 'XSpeed':
                    self.SetSpeed(axis = 'X')
                elif self.item.action == 'YSpeed':
                    self.SetSpeed(axis = 'Y')
                elif self.item.action == 'ZSpeed':
                    self.SetSpeed(axis = 'Z')
                elif self.item.action == 'XAcc':
                    self.SetAcc(axis = 'X')
                elif self.item.action == 'YAcc':
                    self.SetAcc(axis = 'Y')
                elif self.item.action == 'ZAcc':
                    self.SetAcc(axis = 'Z')

                else:
                    message = f"Unknown stage/galvo command: {self.item.action}"
                    self.emit_status(message)
                    print(message)
                    # self.ui.PrintOut.append(message)
            except Exception:
                message = "Stage/galvo command failed. This action was skipped."
                print(Exception)
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
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
        if not (STAGE_SIM or self.SIM):
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
        print(message)
        self.StagebackQueue.put(0)

    def Uninit(self):
        if not (STAGE_SIM or self.SIM):
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
        if not (ARTDAQ_SIM or self.SIM): # if not running DAQ simulation mode
            camera_name = self.ui.Camera.currentText()
            camera = get_camera_spec(camera_name)
            if camera_name == 'Daheng' and camera is not None:
                frameRate = self.ui.FrameRate_DH.value() * 2
            elif camera_name == 'PhotonFocus' and camera is not None:
                frameRate = self.ui.FrameRate_PF.value() * 2
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
        if not (ARTDAQ_SIM or self.SIM):
            self.AOtask.write(self.AOwaveform, auto_start = False)
            self.DOtask.write(self.DOwaveform, auto_start = False)
            self.DOtask.start()
            self.AOtask.start()
        self.StagebackQueue.put(0)

    def StopTask(self):
        if not (ARTDAQ_SIM or self.SIM):
            # self.AOtask.wait_until_done(timeout = 60)
            self.AOtask.stop()
            self.DOtask.stop()

    def tryStopTask(self):
        if not (ARTDAQ_SIM or self.SIM):
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
        if not (ARTDAQ_SIM or self.SIM):
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
        if not (ARTDAQ_SIM or self.SIM):
            with ni.Task('AOtask') as AOtask:
                AOtask.ao_channels.add_ao_voltage_chan(physical_channel=self.GalvoAO, \
                                                      min_val=AODO_AO_VOLTAGE_MIN, max_val=AODO_AO_VOLTAGE_MAX, \
                                                      units=ni.constants.VoltageUnits.VOLTS)
                AOtask.write(self.ui.GalvoBias.value(), auto_start = True)
                AOtask.wait_until_done(timeout = 1)
                AOtask.stop()

    def add_do_chan_single_line(self, task, line_name):
        if LineGrouping is None:
            task.do_channels.add_do_chan(lines=line_name)
        else:
            task.do_channels.add_do_chan(
                lines=line_name,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )

    def servo_target_angle_deg(self):
        return float(np.clip(SERVO_START_ANGLE_DEG + SERVO_MOVE_DELTA_DEG, 0.0, 180.0))

    def angle_to_pulse_width_us(self, angle_deg):
        angle_deg = float(np.clip(angle_deg, 0.0, 180.0))
        return SERVO_MIN_PULSE_US + (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US) * (angle_deg / 180.0)

    def build_constant_angle_waveform(self, angle_deg, duration_s):
        pulse_width_us = self.angle_to_pulse_width_us(angle_deg)
        samples_per_period = int(round(SERVO_SAMPLE_RATE_HZ / SERVO_PWM_FREQUENCY_HZ))
        high_samples = int(round(SERVO_SAMPLE_RATE_HZ * pulse_width_us / 1_000_000.0))
        high_samples = max(1, min(high_samples, samples_per_period - 1))

        single_period = np.zeros(samples_per_period, dtype=np.bool_)
        single_period[:high_samples] = True

        cycle_count = max(1, int(round(duration_s * SERVO_PWM_FREQUENCY_HZ)))
        waveform = np.tile(single_period, cycle_count)
        return waveform, pulse_width_us

    def play_servo_waveform(self, task, waveform, duration_s):
        task.timing.cfg_samp_clk_timing(
            rate=SERVO_SAMPLE_RATE_HZ,
            sample_mode=Atype.FINITE,
            samps_per_chan=len(waveform),
        )
        task.write(waveform, auto_start=False)
        task.start()
        task.wait_until_done(timeout=duration_s + 2.0)
        task.stop()

    def hold_servo_angle(self, task, angle_deg, duration_s, label):
        waveform, pulse_width_us = self.build_constant_angle_waveform(angle_deg, duration_s)
        message = (
            f"{label}: angle={angle_deg:.1f} deg, pulse={pulse_width_us:.0f} us, "
            f"duration={duration_s:.1f} s"
        )
        print(message)
        self.play_servo_waveform(task, waveform, duration_s)

    def rotate_servo_out(self):
        target_angle_deg = self.servo_target_angle_deg()
        if ARTDAQ_SIM or self.SIM:
            time.sleep(SERVO_MOVE_TIME_S)
        else:
            task = ni.Task("ServoPWM")
            try:
                self.add_do_chan_single_line(task, SERVO_DEVICE_LINE)
                self.hold_servo_angle(task, target_angle_deg, SERVO_MOVE_TIME_S, "Rotate servo out")
            finally:
                task.close()
        self.StagebackQueue.put(0)

    def rotate_servo_back(self):
        if ARTDAQ_SIM or self.SIM:
            time.sleep(SERVO_MOVE_TIME_S)
        else:
            task = ni.Task("ServoPWM")
            try:
                self.add_do_chan_single_line(task, SERVO_DEVICE_LINE)
                self.hold_servo_angle(task, SERVO_START_ANGLE_DEG, SERVO_MOVE_TIME_S, "Rotate servo back")
            finally:
                task.close()
        self.StagebackQueue.put(0)

    def stage_motion_timeout(self, axis, distance):
        speed = float(getattr(self.ui, f"{axis}Speed").value())
        speed = max(speed, 0.01)
        return max(5.0, abs(float(distance)) / speed + 5.0)

    def stage_home_timeout(self, axis):
        speed = float(getattr(self.ui, f"{axis}Speed").value())
        speed = max(speed, 0.01)
        position_widget = getattr(self.ui, f"{axis}Position")
        stage_range = float(position_widget.maximum()) - float(position_widget.minimum())
        return max(30.0, 2.0 * stage_range / speed)

    def run_motor_call(self, callback, timeout, description):
        error_box = []
        finished = threading.Event()

        def runner():
            try:
                callback()
            except Exception as error:
                error_box.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        if not finished.wait(timeout):
            message = (
                f"Warning: {description} timed out after {timeout:.1f}s. "
                "Assuming motion completed and continuing."
            )
            print(message)
            self.emit_status(message)
            return False
        if error_box:
            raise error_box[0]
        return True

    def SetSpeed(self, axis = 'X'):
        if not (STAGE_SIM or self.SIM):
            if axis =='X':
                x_axis = get_stage_axis_spec('X')
                motors.set_move_speed(x_axis.axis_index, self.ui.XSpeed.value())
            elif axis =='Y':
                y_axis = get_stage_axis_spec('Y')
                motors.set_move_speed(y_axis.axis_index, self.ui.YSpeed.value())
            elif axis =='Z':
                z_axis = get_stage_axis_spec('Z')
                motors.set_move_speed(z_axis.axis_index, self.ui.ZSpeed.value())
            # print('done')
            
    def SetAcc(self, axis = 'X'):
        if not (STAGE_SIM or self.SIM):
            if axis =='X':
                x_axis = get_stage_axis_spec('X')
                motors.set_acceleration(x_axis.axis_index, self.ui.XAccelerate.value())
            elif axis =='Y':
                y_axis = get_stage_axis_spec('Y')
                motors.set_acceleration(y_axis.axis_index, self.ui.YAccelerate.value())
            elif axis =='Z':
                z_axis = get_stage_axis_spec('Z')
                motors.set_acceleration(z_axis.axis_index, self.ui.ZAccelerate.value())
            
            
    def Move(self, axis = 'X'):


        if axis =='X':
            x_axis = get_stage_axis_spec('X')
            # motors.set_enable(x_axis.axis_index,True)
            distance = self.ui.XPosition.value() - self.ui.Xcurrent.value()
            message = f"Stage X move start: current={self.ui.Xcurrent.value():.4f}, target={self.ui.XPosition.value():.4f}, distance={distance:.4f}."
            print(message)
            move_completed = self.run_motor_call(
                lambda: motors.move_relative(x_axis.axis_index, distance),
                self.stage_motion_timeout('X', distance),
                "Stage X move",
            )
            self.ui.Xcurrent.setValue(self.ui.Xcurrent.value()+distance)
            if move_completed:
                message = f"Stage X move complete: new_current={self.ui.Xcurrent.value():.4f}."
            else:
                message = f"Stage X move timed out but was assumed complete: new_current={self.ui.Xcurrent.value():.4f}."
                print(message)

        if axis =='Y':
            y_axis = get_stage_axis_spec('Y')
            # motors.set_enable(y_axis.axis_index,True)
            distance = self.ui.YPosition.value() - self.ui.Ycurrent.value()
            message = f"Stage Y move start: current={self.ui.Ycurrent.value():.4f}, target={self.ui.YPosition.value():.4f}, distance={distance:.4f}."
            print(message)
            move_completed = self.run_motor_call(
                lambda: motors.move_relative(y_axis.axis_index, distance),
                self.stage_motion_timeout('Y', distance),
                "Stage Y move",
            )
            self.ui.Ycurrent.setValue(self.ui.Ycurrent.value()+distance)
            if move_completed:
                message = f"Stage Y move complete: new_current={self.ui.Ycurrent.value():.4f}."
            else:
                message = f"Stage Y move timed out but was assumed complete: new_current={self.ui.Ycurrent.value():.4f}."
                print(message)

        if axis =='Z':
            z_axis = get_stage_axis_spec('Z')
            # motors.set_enable(z_axis.axis_index,True)
            distance = self.ui.ZPosition.value() - self.ui.Zcurrent.value()
            message = f"Stage Z move start: current={self.ui.Zcurrent.value():.4f}, target={self.ui.ZPosition.value():.4f}, distance={distance:.4f}."
            print(message)
            move_completed = self.run_motor_call(
                lambda: motors.move_relative(z_axis.axis_index, distance),
                self.stage_motion_timeout('Z', distance),
                "Stage Z move",
            )
            self.ui.Zcurrent.setValue(self.ui.Zcurrent.value()+distance)
            if move_completed:
                message = f"Stage Z move complete: new_current={self.ui.Zcurrent.value():.4f}."
            else:
                message = f"Stage Z move timed out but was assumed complete: new_current={self.ui.Zcurrent.value():.4f}."
                print(message)

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
        if not (STAGE_SIM or self.SIM):
            self.Move(axis)
        else:
            time.sleep(1)
        message = f"Stage position: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        print(message)
        self.StagebackQueue.put(0)

    def StepMove(self, axis, Direction):
        if not (STAGE_SIM or self.SIM):
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
        self.StagebackQueue.put(0)

    def Home(self, axis):
        if not (STAGE_SIM or self.SIM):
            if axis == 'X':
                # self.ui.XPosition.setValue(0)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
                x_axis = get_stage_axis_spec('X')
                motors.set_home_speed(x_axis.axis_index, self.ui.XSpeed.value())
                home_completed = self.run_motor_call(
                    lambda: motors.home(x_axis.axis_index),
                    self.stage_home_timeout('X'),
                    "Stage X home",
                )
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
                home_completed = self.run_motor_call(
                    lambda: motors.home(y_axis.axis_index),
                    self.stage_home_timeout('Y'),
                    "Stage Y home",
                )
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
                home_completed = self.run_motor_call(
                    lambda: motors.home(z_axis.axis_index),
                    self.stage_home_timeout('Z'),
                    "Stage Z home",
                )
                self.ui.ZPosition.setValue(0)
                self.ui.Zcurrent.setValue(0)
                # self.ui.ZPosition.setValue(1)
                # self.DirectMove(axis)
                # self.StagebackQueue.get()
        else:
            time.sleep(2)
        if not (STAGE_SIM or self.SIM):
            if axis == 'X':
                home_completed = locals().get("home_completed", True)
                prefix = "Stage home complete" if home_completed else "Stage home timed out but was assumed complete"
            elif axis == 'Y':
                home_completed = locals().get("home_completed", True)
                prefix = "Stage home complete" if home_completed else "Stage home timed out but was assumed complete"
            else:
                home_completed = locals().get("home_completed", True)
                prefix = "Stage home complete" if home_completed else "Stage home timed out but was assumed complete"
            message = f"{prefix}: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        else:
            message = f"Stage position: X={self.ui.Xcurrent.value()}, Y={round(self.ui.Ycurrent.value(), 2)}, Z={self.ui.Zcurrent.value()}."
        print(message)
        self.StagebackQueue.put(0)
