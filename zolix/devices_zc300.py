import ctypes


class ZC300:
    def __init__(self) -> None:
        # 初始化
        self.zc300 = ctypes.cdll.LoadLibrary(r"D:\zolix\zolix\zolix_zc_300.dll")

    def zc300_enum_count(self):
        """获取设备数量"""
        enum_dev = self.zc300.zc300_enum_count()
        return enum_dev

    def zc300_enum_info(self, index, buff_size):
        """枚举串口设备的串口信息
        index: 设备序列号,0  ~  zc300_enum_count - 1.
                buff: 存放设备串口信息的缓冲区,建议至少 16 个字节.
                buff_size: 缓冲区大小"""
        self.zc300.zc300_enum_info.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.zc300.zc300_enum_info.restype = ctypes.c_void_p

        a = ctypes.c_int(index)
        str_info = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)

        self.zc300.zc300_enum_info(a, str_info, buff_size1)
        return str_info.value

    def zc300_open(self, dev_sn, addr):
        """连接zc300控制器,注意串口信息和zc300设备一定要匹配
        参数：
                info: 设备串口信息.
                addr: 设备通信地址, 有效范围 1 ~ 255."""
        self.zc300.zc300_open.argtypes = [ctypes.c_char_p, ctypes.c_ubyte]
        self.zc300.zc300_open.restype = ctypes.c_bool

        # dev_sn_1 = ctypes.c_char_p(dev_sn)
        addr1 = ctypes.c_ubyte(addr)

        issuccess = self.zc300.zc300_open(dev_sn, addr1)
        # print('设备型号：',dev_sn)
        return issuccess

    def zc300_close(self):
        """与zc300设备断开连接"""
        self.zc300.zc300_close()
        return None

###########################################
    def zc300_get_sn(self):
        """
        获取zc300设备序列号,十进制表示"""
        self.zc300.zc300_get_sn.argtypes = [
            ctypes.POINTER(ctypes.c_long)]
        self.zc300.zc300_get_sn.restype = ctypes.c_bool

        sn1 = ctypes.c_long(999)
        self.zc300.zc300_get_sn(ctypes.byref(sn1))

        return sn1.value

    def zc300_get_model(self, buff_size):
        """
        获取zc300设备型号信息
        参数：
                buff: 存放型号信息的缓冲区,建议至少 16 个字节.
                buff_size: 缓冲区大小.
        """
        self.zc300.zc300_get_model.argtypes = [
            ctypes.c_void_p, ctypes.c_int]
        self.zc300.zc300_get_model.restype = ctypes.c_bool

        buff1 = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)
        self.zc300.zc300_get_model(
            buff1, buff_size1)

        return buff1.value

    def zc300_error_info(self, buff_size):
        """
        其他功能(一般设置功能)执行失败后,读取相应的错误信息
        参数：
                buff: 存放错误信息的缓冲区,建议大小 64 个字节.
                buff_size: 缓冲区长度.
        """
        self.zc300.zc300_error_info.argtypes = [
            ctypes.c_void_p, ctypes.c_int]
        self.zc300.zc300_error_info.restype = ctypes.c_bool

        buff1 = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)
        self.zc300.zc300_error_info(
            buff1, buff_size1)

        return buff1.value

###########################################

    def zc300_set_enabled(self, axis, enabled):
        """
        设置电移台使能状态
        参数：
                axis: 要控制的目标电移台轴号.
                enabled: true - 启用, false - 禁用."""
        self.zc300.zc300_set_enabled.argtypes = [
            ctypes.c_short, ctypes.c_bool]
        self.zc300.zc300_set_enabled.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        enabled1 = ctypes.c_bool(enabled)
        

        issuccess = self.zc300.zc300_set_enabled(axis, enabled1)
        return issuccess

    def zc300_get_enabled(self,  axis):
        """
        读取板使能状态, 默认是使能.
        参数：
                axis: 要控制的目标电移台轴号.
                enabled: true - 启用, false - 禁用."""
        self.zc300.zc300_get_enabled.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_bool)]
        self.zc300.zc300_get_enabled.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        enabled1 = ctypes.c_bool(0)

        self.zc300.zc300_get_enabled(axis1, ctypes.byref(enabled1))
        return enabled1.value
###########################################

    def zc300_set_stage_type(self, axis, type1):
        """
        设置电移台类型,一般分为直线型和旋转型
        参数：
                axis: 要控制的目标电移台轴号
                type: 电移台类型,参考宏定义"""
        self.zc300.zc300_set_stage_type.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_stage_type.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        type11 = ctypes.c_uint(type1)

        issuccess = self.zc300.zc300_set_stage_type(axis1, type11)
        return issuccess

    def zc300_get_stage_type(self, axis):
        """
        读取电移台类型
        参数：
                axis: 要控制的目标电移台轴号
                type: 电移台类型,参考宏定义"""
        self.zc300.zc300_get_stage_type.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_stage_type.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        type1 = ctypes.c_short(5)

        self.zc300.zc300_get_stage_type(axis1, ctypes.byref(type1))
        return type1.value


###########################################

    def zc300_set_io_output(self, output):
        """
        设置电移台IO口输出值,有效范围 0 ~ 8.
        参数：
                output: IO口输出值 """
        self.zc300.zc300_set_io_output.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_set_io_output.restype = ctypes.c_bool

        output1 = ctypes.c_short(output)

        issuccess = self.zc300.zc300_set_io_output(output1)
        return issuccess

    def zc300_get_io_output(self):
        """
        读取电移台IO口输出值.
        参数：
                output: 存放io输出值的变量的地址."""
        self.zc300.zc300_get_io_output.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_io_output.restype = ctypes.c_bool

        output1 = ctypes.c_short(55)

        self.zc300.zc300_get_io_output(ctypes.byref(output1))
        return output1.value

    def zc300_get_io_input(self):
        """
        读取电移台IO口输入值
        参数：
                input: 存放io输入值的变量的地址"""
        self.zc300.zc300_get_io_input.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_io_input.restype = ctypes.c_bool

        input1 = ctypes.c_short(55)

        self.zc300.zc300_get_io_input(ctypes.byref(input1))
        return input1.value

###########################################
    def zc300_set_unit(self, axis, unit):
        """
        设置 使用单位
            axis: 轴号 [0:X, 1:Y, 2:Z]
            unit: 单位 [0:mm, 1:um, 2:pp, 3:deg]"""
        self.zc300.zc300_set_unit.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_unit.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        unit1 = ctypes.c_short(unit)

        issuccess = self.zc300.zc300_set_unit(axis1, unit1)
        return issuccess

    def zc300_get_unit(self, axis):
        """
        读取 使用单位
            axis: 轴号 [0:X, 1:Y, 2:Z]
            unit: 单位 [0:mm, 1:um, 2:pp, 3:deg]"""
        self.zc300.zc300_get_unit.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_unit.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        unit1 = ctypes.c_short(55)

        self.zc300.zc300_get_unit(axis1, ctypes.byref(unit1))
        return unit1.value


###########################################


    def zc300_set_pitch(self, axis, pitch):
        """
        设置电移台运动参数-丝杆导程,单位:mm/r,有效范围:0 <pitch < 1000
        axis: 轴号 [0:X, 1:Y, 2:Z]
        pitch: 导程"""
        self.zc300.zc300_set_pitch.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_pitch.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pitch1 = ctypes.c_float(pitch)

        issuccess = self.zc300.zc300_set_pitch(axis1, pitch1)
        return issuccess

    def zc300_get_pitch(self, axis):
        """
        读取电移台运动参数-丝杆导程,单位:mm/r
        axis: 轴号 [0:X, 1:Y, 2:Z]
        pitch: 导程"""
        self.zc300.zc300_get_pitch.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_pitch.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pitch1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_pitch(axis1, ctypes.byref(pitch1))
        return pitch1.value


###########################################


    def zc300_set_spr(self, axis, spr):
        """
        设置电移台运动参数-每转脉冲数,单位:pp/r,有效范围:0 <spr < 10000000.
        axis: 轴号 [0:X, 1:Y, 2:Z]
        spr: 每转脉冲数,注意数值是否在有效范围内"""
        self.zc300.zc300_set_spr.argtypes = [
            ctypes.c_short, ctypes.c_long]
        self.zc300.zc300_set_spr.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        spr1 = ctypes.c_long(spr)

        issuccess = self.zc300.zc300_set_spr(axis1, spr1)
        return issuccess

    def zc300_get_spr(self, axis):
        """
        读取电移台运动参数-每转脉冲数,单位:pp/r
        axis: 轴号 [0:X, 1:Y, 2:Z]
        spr: 每转脉冲数,注意数值是否在有效范围内"""
        self.zc300.zc300_get_spr.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_long)]
        self.zc300.zc300_get_spr.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        spr1 = ctypes.c_long(99)

        self.zc300.zc300_get_spr(axis1, ctypes.byref(spr1))
        return spr1.value


###########################################


    def zc300_set_ratio(self, axis, ratio):
        """
        设置电移台运动参数-传动比,有效范围:0 <ratio < 10000.
        axis: 轴号 [0:X, 1:Y, 2:Z]
        axis: 要控制的目标电移台轴号.
                ratio: 传动比,注意数值是否在有效范围内"""
        self.zc300.zc300_set_ratio.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_ratio.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        ratio1 = ctypes.c_float(ratio)

        issuccess = self.zc300.zc300_set_ratio(axis1, ratio1)
        return issuccess

    def zc300_get_ratio(self, axis):
        """读取电移台运动参数-传动比
        axis: 轴号 [0:X, 1:Y, 2:Z]
        axis: 要控制的目标电移台轴号.
                ratio: 传动比,注意数值是否在有效范围内"""
        self.zc300.zc300_get_ratio.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_ratio.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        ratio1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_ratio(axis1, ctypes.byref(ratio1))
        return ratio1.value

###########################################
    def zc300_set_init_speed(self, axis, speed):
        """
        设置电移台的运动初速度,单位和当前轴设置的单位相同, 此值要小于常速度.
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 初速度"""
        self.zc300.zc300_set_init_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_init_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_init_speed(axis1, speed1)
        return issuccess

    def zc300_get_init_speed(self, axis):
        """
        读取电移台的运动初速度,单位和当前轴设置的单位相同.
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 初速度"""
        self.zc300.zc300_get_init_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_init_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_init_speed(axis1, ctypes.byref(speed1))
        return speed1.value

###########################################
    def zc300_set_move_speed(self, axis, speed):
        """
        设置电移台的运动常速度,单位和当前轴设置的单位相同
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 常速度"""
        self.zc300.zc300_set_move_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_move_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_move_speed(axis1, speed1)
        return issuccess

    def zc300_get_move_speed(self, axis):
        """
        读取电移台的运动常速度,单位和当前轴设置的单位相同
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 常速度"""
        self.zc300.zc300_get_move_speed.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_move_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_move_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_acc_speed(self, axis, speed):
        """
        设置电移台的运动加速度
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 加速度"""
        self.zc300.zc300_set_acc_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_acc_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_acc_speed(axis1, speed1)
        return issuccess

    def zc300_get_acc_speed(self, axis):
        """
        读取电移台的运动加速度
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 加速度"""
        self.zc300.zc300_get_acc_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_acc_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_acc_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_home_speed(self, axis, speed):
        """
        设置电移台的原点回归(复位)速度,单位和当前轴设置的单位相同
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 复位速度"""
        self.zc300.zc300_set_home_speed.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_home_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_home_speed(axis1, speed1)
        return issuccess

    def zc300_get_home_speed(self, axis):
        """
        读取电移台的原点回归(复位)速度,单位和当前轴设置的单位相同
        axis: 轴号 [0:X, 1:Y, 2:Z]
        speed: 复位速度"""
        self.zc300.zc300_get_home_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_home_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_home_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_home_mode(self, axis, mode):
        """
        设置电移台归零方式
        axis: 要控制的目标电移台轴号.
                mode: 归零方式, 参考宏定义."""
        self.zc300.zc300_set_home_mode.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_home_mode.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = ctypes.c_short(mode)

        issuccess = self.zc300.zc300_set_home_mode(axis1, mode1)
        return issuccess

    def zc300_get_home_mode(self, axis,mode =ctypes.c_short(55) ):
        """
        读取电移台归零模式
        axis: 要控制的目标电移台轴号.
                mode: 归零方式, 参考宏定义."""
        self.zc300.zc300_get_home_mode.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_home_mode.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = mode

        self.zc300.zc300_get_home_mode(axis1, ctypes.byref(mode1))
        return mode1.value

###########################################
    def zc300_move(self, axis, type1, distance):
        """
        控制电移台移动,距离单位和当前轴设置的单位相同.
        axis: 要控制的目标电移台轴号.
        type: 移动类型, 参考宏定义.
        distance: 运动距离,方向由变量的符号决定,正向前负向后"""
        self.zc300.zc300_move.argtypes = [
            ctypes.c_short, ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_move.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        type11 = ctypes.c_short(type1)
        distance1 = ctypes.c_float(distance)

        issuccess = self.zc300.zc300_move(axis1, type11, distance1)
        return issuccess

###########################################

    def zc300_stop(self, axis, mode):
        """
        控制电移台停止移动
        axis: 轴号 [0:X, 1:Y, 2:Z]
        mode: 模式 [0:立刻停止, 1:减速停止]"""
        self.zc300.zc300_stop.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_stop.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = ctypes.c_short(mode)

        issuccess = self.zc300.zc300_stop(axis1, mode1)
        return issuccess


###########################################


    def zc300_home(self, axis):
        """
        控制电移台回到原点"""
        self.zc300.zc300_home.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_home.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)

        issuccess = self.zc300.zc300_home(axis1)
        return issuccess

###########################################
    def zc300_set_position(self, axis, pos):
        """
        设置(自定义)电移台当前位置坐标,单位和当前轴设置的单位相同,例:当前轴位置为坐标5,通过zc300_set_position 设置为0时,可实现自定义用户零点位置. 
        axis: 要控制的目标电移台轴号.
		pos: 自定义的位置坐标."""
        self.zc300.zc300_set_position.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_position.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pos1 = ctypes.c_float(pos)

        issuccess = self.zc300.zc300_set_position(axis1, pos1)
        return issuccess

    def zc300_get_position(self, axis):
        """
        读取电移台当前位置坐标,单位和当前轴设置的单位相同
        axis: 要控制的目标电移台轴号.
		pos: 自定义的位置坐标"""
        self.zc300.zc300_get_position.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_position.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        position1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_position(axis1, ctypes.byref(position1))
        return position1.value

###########################################
    def zc300_get_idle(self, axis):
        """
        获取电移台当前是否为空闲状态
        axis: 要读取的目标电移台轴号.
		idle: 存放状态的变量的地址."""
        self.zc300.zc300_get_idle.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_bool)]
        self.zc300.zc300_get_idle.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        idle1 = ctypes.c_bool(1)

        self.zc300.zc300_get_idle(axis1, ctypes.byref(idle1))
        return idle1.value

###########################################
    def zc300_get_status(self):
        """
        获取控制器监视到的所有电移台当前的物理状态,注意:可能获取到多种状态信息,参考状态宏定义,通过位运算去判别是否存在某种状态
        status: 存放状态的变量的地址, 参考宏定义"""
        self.zc300.zc300_get_status.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_status.restype = ctypes.c_bool

        status1 = ctypes.c_short(99)

        self.zc300.zc300_get_status(ctypes.byref(status1))
        return status1.value

###########################################
    def zc300_set_buzzer(self, status):
        """
        设置蜂鸣器在异常状态时是否鸣叫
        status: 工作状态,参考宏定义"""
        self.zc300.zc300_set_buzzer.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_set_buzzer.restype = ctypes.c_bool

        status1 = ctypes.c_short(status)
        issuccess = self.zc300.zc300_set_buzzer(status1)
        return issuccess

    def zc300_get_buzzer(self):
        """
        读取蜂鸣器在异常状态时是否鸣叫
        status: 存放状态的变量的地址"""
        self.zc300.zc300_get_buzzer.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_buzzer.restype = ctypes.c_bool

        status1 = ctypes.c_short(55)

        self.zc300.zc300_get_buzzer(ctypes.byref(status1))
        return status1.value
