# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:32:29 2026

@author: shuaibin
"""

"""
基于卓立汉光 ZC300 控制器的电机控制类（修正版）
关键修正：zc300_set_spr 传入的是细分数（如 1600），而非 200*1600
"""

import time
from devices_zc300 import ZC300  # 你提供的 SDK 封装（已按调试结果修改）
# XYZ 电机参数，更换电机需手动修改,顺序：X Y Z
# 电机丝杠导程，单位mm
PITCH = [4,5,10]
# 电机传动比
REDUCTION = [1,1,120]

# 常量定义（与 SDK 头文件保持一致）
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

UNIT_PP = 0      # 脉冲
UNIT_MM = 1      # 毫米
UNIT_DEG = 2     # 度

STAGE_LINEAR = 0
STAGE_ROTARY = 1

HOME_BY_USER = 1
HOME_BY_LIMIT = 2
HOME_BY_ZERO = 3

MOVE_ABSOLUTE = 0
MOVE_RELATIVE = 1
MOVE_CONTINUOUS = 2

STOP_SLOWLY = 0
STOP_IMMEDIATELY = 1



class ZC300MotorController:
    """ZC300 电机控制器高级封装类（修正 spr 含义）"""

    def __init__(self, port_info=None, addr=1, auto_connect=True):
        """
        初始化控制器
        :param port_info: 串口信息，如 'COM3' 或 None（自动枚举第一个）
        :param addr: 控制器地址（1~255）
        :param auto_connect: 是否自动连接
        """
        self.zc = ZC300()
        self.addr = addr
        self.connected = False

        if auto_connect:
            self.connect(port_info)

        # 存储各轴配置参数
        self.axis_config = {
            AXIS_X: {},
            AXIS_Y: {},
            AXIS_Z: {}
        }

    def connect(self, port_info=None):
        """连接控制器"""
        if port_info is None:
            dev_count = self.zc.zc300_enum_count()
            if dev_count == 0:
                raise RuntimeError("未检测到 ZC300 控制器")
            buff_size = 64
            port_info = self.zc.zc300_enum_info(0, buff_size).decode('ascii', errors='ignore')
            print(f"自动使用设备: {port_info}")

        ret = self.zc.zc300_open(port_info.encode(), self.addr)
        if not ret:
            raise RuntimeError(f"连接失败: {port_info}, 地址: {self.addr}")
        self.connected = True
        print("ZC300 连接成功")

    def close(self):
        """断开连接"""
        if self.connected:
            self.zc.zc300_close()
            self.connected = False
            print("ZC300 已断开")

    def __del__(self):
        self.close()

    # ==================== 轴参数配置 ====================
    def configure_axis(self, axis, subdivide=1600,
                       unit=UNIT_MM, stage_type=STAGE_LINEAR,
                       direction_reverse=True):
        """
        配置指定轴的机械参数（必须在使用运动前调用）
        :param axis: 轴号 (AXIS_X/Y/Z)
        :param lead_screw: 丝杆导程 (mm/r)
        :param reduction: 减速比 (电机转数 / 丝杆转数)，默认 1.0
        :param subdivide: 步距角细分数（1.8° 细分为 subdivide 微步），例如 1600
        :param unit: 单位 (UNIT_MM / UNIT_PP / UNIT_DEG)
        :param stage_type: 类型 (STAGE_LINEAR / STAGE_ROTARY)
        :param direction_reverse: 是否反向运动（移动距离自动取反）
        """
        if not self.connected:
            raise RuntimeError("控制器未连接")

        # 有效导程 = 丝杆导程 / 减速比（因为 DLL 未使用传动比，需手动折算）
        effective_pitch = PITCH[axis] / REDUCTION[axis]

        # 关键修正：zc300_set_spr 直接传入细分数（如 1600），而非 200*1600
        # 卓立汉光 SDK 中 spr 参数含义是“每步（1.8°）的微步细分数”
        spr = subdivide

        # 调用 SDK 设置参数
        self.zc.zc300_set_stage_type(axis, stage_type)
        self.zc.zc300_set_unit(axis, unit)
        self.zc.zc300_set_pitch(axis, effective_pitch)
        self.zc.zc300_set_spr(axis, spr)   # 直接传细分数
        self.zc.zc300_set_home_mode(axis, HOME_BY_ZERO)
        # 注意：传动比 ratio 在 DLL 中未使用，不设置

        # 保存配置供后续使用
        self.axis_config[axis] = {
            'effective_pitch': effective_pitch,
            'spr': spr,
            'unit': unit,
            'direction_reverse': direction_reverse,
            'reduction': REDUCTION[axis],
            'lead_screw': PITCH[axis],
            'subdivide': subdivide
        }
        print(f"轴{axis} 配置完成: 有效导程={effective_pitch:.4f} mm/r, 细分数={spr}")

    # ==================== 使能控制 ====================
    def set_enable(self, axis, enable):
        """设置轴使能状态 (True=使能, False=禁用)"""
        return self.zc.zc300_set_enabled(axis, enable)

    def get_enable(self, axis):
        """读取轴使能状态"""
        return self.zc.zc300_get_enabled(axis)

    # ==================== 速度/加速度参数设置 ====================
    def set_init_speed(self, axis, speed_mm_s):
        """设置起始速度 (单位: mm/s)"""
        return self.zc.zc300_set_init_speed(axis, speed_mm_s)

    def set_move_speed(self, axis, speed_mm_s):
        """设置运行速度 (单位: mm/s)"""
        return self.zc.zc300_set_move_speed(axis, speed_mm_s)

    def set_acceleration(self, axis, acc_mm_s2):
        """设置加速度 (单位: mm/s²)"""
        return self.zc.zc300_set_acc_speed(axis, acc_mm_s2)

    def set_home_speed(self, axis, speed_mm_s):
        """设置回原点速度 (单位: mm/s)"""
        return self.zc.zc300_set_home_speed(axis, speed_mm_s)

    def set_home_mode(self, axis, mode):
        """
        设置归零方式
        :param mode: HOME_BY_USER, HOME_BY_LIMIT, HOME_BY_ZERO
        """
        return self.zc.zc300_set_home_mode(axis, mode)

    # ==================== 运动控制 ====================
    def _apply_direction(self, axis, distance):
        """根据方向反转标志调整移动距离符号"""
        if self.axis_config[axis].get('direction_reverse', False):
            return -distance
        return distance

    def move_relative(self, axis, distance_mm):
        """
        相对运动（正负表示方向）
        :param distance_mm: 移动距离 (mm)
        """
        dist = self._apply_direction(axis, distance_mm)
        result = self.zc.zc300_move(axis, MOVE_RELATIVE, dist)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def move_absolute(self, axis, position_mm):
        """
        绝对运动（移动到指定坐标）
        :param position_mm: 目标位置 (mm)
        """
        # 绝对移动不受 direction_reverse 影响，因为用户坐标系已定义方向
        result = self.zc.zc300_move(axis, MOVE_ABSOLUTE, position_mm)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def move_continuous(self, axis, direction_positive=True):
        """
        连续运动
        :param direction_positive: True=正向, False=反向
        """
        distance = 1.0 if direction_positive else -1.0
        dist = self._apply_direction(axis, distance)
        result = self.zc.zc300_move(axis, MOVE_CONTINUOUS, dist)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def stop(self, axis, immediate=False):
        """
        停止运动
        :param immediate: True=立即停止, False=减速停止
        """
        mode = STOP_IMMEDIATELY if immediate else STOP_SLOWLY
        return self.zc.zc300_stop(axis, mode)

    def home(self, axis):
        """
        执行回原点操作
        :param wait: 是否等待回零完成
        :param timeout_s: 等待超时时间（秒）
        """
        # self.move_absolute(axis, -1)
        ret = self.zc.zc300_home(axis)
        if not ret:
            return False
        while not self.is_idle(axis):
            time.sleep(0.2)
        # time.sleep(2)
        # while not self.is_idle(axis):
        #     time.sleep(0.1)
        # print(self.is_idle(axis))
        self.set_position(axis, 0)
        return True

    # ==================== 位置和状态查询 ====================
    def get_position(self, axis):
        """获取当前位置 (单位: mm)"""
        return self.zc.zc300_get_position(axis)

    def set_position(self, axis, pos_mm):
        """设置当前位置为自定义坐标（不会移动电机）"""
        return self.zc.zc300_set_position(axis, pos_mm)

    def is_idle(self, axis):
        """检查轴是否空闲（无运动）"""
        return self.zc.zc300_get_idle(axis)

    def get_status(self):
        """获取控制器全局状态位（包含限位、原点、报警等）"""
        return self.zc.zc300_get_status()

    def get_axis_limit_status(self, axis):
        """
        获取指定轴的限位/原点状态
        返回字典: {'positive_limit': bool, 'negative_limit': bool, 'origin': bool}
        """
        status = self.get_status()
        if axis == AXIS_X:
            pos_limit = bool(status & 0x0001)
            neg_limit = bool(status & 0x0002)
            origin = bool(status & 0x0004)
        elif axis == AXIS_Y:
            pos_limit = bool(status & 0x0008)
            neg_limit = bool(status & 0x0010)
            origin = bool(status & 0x0020)
        elif axis == AXIS_Z:
            pos_limit = bool(status & 0x0040)
            neg_limit = bool(status & 0x0080)
            origin = bool(status & 0x0100)
        else:
            return {}
        return {
            'positive_limit': pos_limit,
            'negative_limit': neg_limit,
            'origin': origin
        }

    def wait_idle(self, axis, timeout_s=10):
        """等待轴运动停止"""
        start = time.time()
        while not self.is_idle(axis):
            if time.time() - start > timeout_s:
                return False
            time.sleep(0.2)
        return True


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建控制器实例（自动连接第一个设备）
    motor = ZC300MotorController()
    AXIS = AXIS_X
    try:
        # 配置 X 轴（丝杆导程4mm，无减速，细分1600，方向反向）
        motor.configure_axis(AXIS)   # 根据你的调试加负号

        # 设置运动参数
        motor.set_init_speed(AXIS, 0.1)      # 起始速度 1 mm/s
        motor.set_move_speed(AXIS, 10)     # 运行速度 20 mm/s
        motor.set_acceleration(AXIS, 10)   # 加速度 50 mm/s²
        motor.set_home_speed(AXIS, 10)      # 回零速度 5 mm/s
        # motor.set_home_mode(AXIS, HOME_BY_ZERO)  # 零点开关归零

        # 使能电机
        motor.set_enable(AXIS, True)

        # 执行回原点
        print("开始回原点...")
        motor.home(AXIS)
        print(f"原点位置: {motor.get_position(AXIS)} mm")

        # 相对移动 10mm
        motor.move_relative(AXIS, 1.0)
        print(f"移动后位置: {motor.get_position(AXIS)} mm")

        # 绝对移动到 5mm 处
        # motor.move_absolute(AXIS_X, 5.0)
        # motor.wait_idle(AXIS_X)
        # print(f"绝对移动后位置: {motor.get_position(AXIS_X)} mm")

        # 禁用电机
        motor.set_enable(AXIS, False)

    finally:
        motor.close()