# 在python3.11.x版本测试无问题
# 设备使用demo
import devices_zc300 as de
import time


# 创建设备
zc300_1 = de.ZC300()

try:
    # 打开，并连接硬件
    enum_dev = zc300_1.zc300_enum_count()
    print(f'设备数量为:{enum_dev}')
    # 设备序列号为
    device_sn = zc300_1.zc300_enum_info(enum_dev-1, 16)
    print(f'设备序列号为:{device_sn}')

    com_port=1
    tt= zc300_1.zc300_open(device_sn, com_port)
    print(f'设备打开状态:{tt}')
    
    de_sn=zc300_1.zc300_get_sn()
    print(f'设备序列号为:{de_sn}')

    # 读取台子类型
    mode1 = zc300_1.zc300_get_model(32)
    print(f'台子类型为:{mode1}')

    # 电移台使能状态
    tt = zc300_1.zc300_set_enabled(0,True)
    print(f'电移台使能状态:{tt}')

    # 读取 电移台类型
    type1 = zc300_1.zc300_get_stage_type(0)
    print(f'电移台类型为:{type1}')
    
    # 设置 使用单位
    tt = zc300_1.mc600_set_unit(0, 1)
    print(f'设置单位:{tt}')
    
    # 读取/设置 运行常速度
    tt = zc300_1.zc300_set_move_speed(0, 10.0)
    print(f'设置运行速度:{tt}')

    # 读取归零模式
    home_mode = zc300_1.zc300_get_home_mode(0)
    print(f'归零模式为:{home_mode}')
    
    # 设置位置
    tt= zc300_1.zc300_set_position(0, 40.0)
    print(f'设置位置:{tt}')
    
    # 设置蜂鸣器状态
    tt = zc300_1.zc300_set_buzzer(1)

finally:
    # 关闭设备
    zc300_1.zc300_close()
    print("关闭设备")


