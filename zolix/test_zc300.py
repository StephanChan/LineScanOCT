# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:42:17 2026

@author: shuaibin
"""

"""
ZC300 电机控制器 SDK 全功能测试脚本
使用前请确保：
1. 已安装 Python 3.x
2. devices_zc300.py 和 zolix_zc_300.dll 位于正确路径（或修改 DLL 路径）
3. 控制器已通过 USB/串口连接，并已知设备串口号和地址（通常为 1）
"""

import sys
import time
from devices_zc300 import ZC300

# 定义常量（与头文件 zolix_zc_300.h 保持一致）
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

UNIT_PP = 0      # 脉冲单位
UNIT_MM = 1      # 毫米
UNIT_DEG = 2     # 度

STAGE_LINEAR = 0
STAGE_ROTARY = 1

HOME_BY_USER = 1   # 用户当前位置为原点
HOME_BY_LIMIT = 2  # 限位归零
HOME_BY_ZERO = 3   # 零点开关归零

MOVE_ABSOLUTE = 0
MOVE_RELATIVE = 1
MOVE_CONTINUOUS = 2

STOP_SLOWLY = 0        # 减速停止
STOP_IMMEDIATELY = 1   # 立即停止

BUZZER_SILENT = 0
BUZZER_BEEP = 1        # 注意头文件中 BUZZER_BEEP 定义为 0，实际测试时建议用 1

# 测试参数（根据实际硬件调整）
TEST_AXIS = AXIS_Y
TEST_ADDR = 1          # 控制器地址，通常为 1
TEST_SERIAL_PORT = None  # 将通过 zc300_enum_info 获取

def print_separator(title):
    print("\n" + "=" * 60)
    print(f"【{title}】")
    print("=" * 60)

def test_all_functions():
    # 1. 创建 ZC300 实例
    zc = ZC300()
    print("✅ ZC300 实例创建成功")

    # ==================== 设备枚举与连接 ====================
    print_separator("设备枚举与连接")

    # 2. zc300_enum_count() - 获取设备数量
    dev_count = zc.zc300_enum_count()
    print(f"zc300_enum_count() -> 设备数量: {dev_count}")

    if dev_count == 0:
        print("⚠️ 未检测到任何 ZC300 控制器，后续调用可能失败（但仍会尝试执行）")
    else:
        # 3. zc300_enum_info() - 枚举设备串口信息
        buff_size = 64
        dev_info = zc.zc300_enum_info(0, buff_size).decode('ascii', errors='ignore')
        print(f"zc300_enum_info(0, {buff_size}) -> 设备信息: {dev_info}")
        TEST_SERIAL_PORT = dev_info

        # 4. zc300_open() - 连接控制器
        opened = zc.zc300_open(TEST_SERIAL_PORT.encode(), TEST_ADDR)
        print(f"zc300_open('{TEST_SERIAL_PORT}', {TEST_ADDR}) -> 连接{'成功' if opened else '失败'}")
        if not opened:
            print("⚠️ 连接失败，后续依赖连接的函数将返回 False")
    print()

    # ==================== 基础信息获取 ====================
    print_separator("基础信息获取")

    # 5. zc300_get_sn() - 获取设备序列号
    sn = zc.zc300_get_sn()
    print(f"zc300_get_sn() -> 序列号: {sn}")

    # 6. zc300_get_model() - 获取设备型号
    model = zc.zc300_get_model(32).decode('ascii', errors='ignore')
    print(f"zc300_get_model(32) -> 型号: {model}")

    # 7. zc300_error_info() - 读取错误信息（如果有）
    err_info = zc.zc300_error_info(128).decode('ascii', errors='ignore')
    print(f"zc300_error_info(128) -> 错误信息: {err_info if err_info else '无'}")
    print()

    # ==================== 轴使能控制 ====================
    print_separator("轴使能控制")

    # 8. zc300_set_enabled() - 设置使能状态
    enabled_set = zc.zc300_set_enabled(TEST_AXIS, True)
    print(f"zc300_set_enabled({TEST_AXIS}, True) -> 设置使能: {'成功' if enabled_set else '失败'}")

    # 9. zc300_get_enabled() - 读取使能状态
    enabled_get = zc.zc300_get_enabled(TEST_AXIS)
    print(f"zc300_get_enabled({TEST_AXIS}) -> 当前使能状态: {enabled_get}")
    print()

    # ==================== 电移台类型设置 ====================
    print_separator("电移台类型（直线/旋转）")

    # 10. zc300_set_stage_type() - 设置类型（直线）
    set_type_ok = zc.zc300_set_stage_type(TEST_AXIS, STAGE_LINEAR)
    print(f"zc300_set_stage_type({TEST_AXIS}, STAGE_LINEAR) -> {'成功' if set_type_ok else '失败'}")

    # 11. zc300_get_stage_type() - 读取类型
    stage_type = zc.zc300_get_stage_type(TEST_AXIS)
    print(f"zc300_get_stage_type({TEST_AXIS}) -> 当前类型: {'直线' if stage_type == STAGE_LINEAR else '旋转' if stage_type == STAGE_ROTARY else stage_type}")
    print()

    # ==================== IO 端口读写 ====================
    print_separator("IO 端口读写")

    # 12. zc300_set_io_output() - 设置 IO 输出值（0~8）
    io_output_val = 5
    set_io_ok = zc.zc300_set_io_output(io_output_val)
    print(f"zc300_set_io_output({io_output_val}) -> {'成功' if set_io_ok else '失败'}")

    # 13. zc300_get_io_output() - 读取 IO 输出值
    io_output = zc.zc300_get_io_output()
    print(f"zc300_get_io_output() -> IO 输出值: {io_output}")

    # 14. zc300_get_io_input() - 读取 IO 输入值
    io_input = zc.zc300_get_io_input()
    print(f"zc300_get_io_input() -> IO 输入值: {io_input}")
    print()

    # ==================== 单位设置 ====================
    print_separator("坐标单位设置")

    # 15. zc300_set_unit() - 设置单位（毫米）
    set_unit_ok = zc.zc300_set_unit(TEST_AXIS, UNIT_MM)
    print(f"zc300_set_unit({TEST_AXIS}, UNIT_MM) -> {'成功' if set_unit_ok else '失败'}")

    # 16. zc300_get_unit() - 读取单位
    unit = zc.zc300_get_unit(TEST_AXIS)
    unit_str = {UNIT_PP: "脉冲", UNIT_MM: "毫米", UNIT_DEG: "度"}.get(unit, "未知")
    print(f"zc300_get_unit({TEST_AXIS}) -> 当前单位: {unit_str} ({unit})")
    print()

    # ==================== 机械参数设置（导程、每转脉冲数、传动比） ====================
    print_separator("机械参数")

    # 17. zc300_set_pitch() - 设置丝杠导程（mm/r）
    pitch_mm = 5  # 与你实际滑台一致
    set_pitch_ok = zc.zc300_set_pitch(TEST_AXIS, pitch_mm)
    print(f"zc300_set_pitch({TEST_AXIS}, {pitch_mm}) -> {'成功' if set_pitch_ok else '失败'}")

    # 18. zc300_get_pitch() - 读取导程
    pitch = zc.zc300_get_pitch(TEST_AXIS)
    print(f"zc300_get_pitch({TEST_AXIS}) -> 导程 = {pitch} mm/r")

    # 19. zc300_set_spr() - 设置每转脉冲数（脉冲/转）
    spr_value = 1600  # 你的配置：200步/圈 * 1600细分 = 320000
    set_spr_ok = zc.zc300_set_spr(TEST_AXIS, spr_value)
    print(f"zc300_set_spr({TEST_AXIS}, {spr_value}) -> {'成功' if set_spr_ok else '失败'}")

    # 20. zc300_get_spr() - 读取每转脉冲数
    spr = zc.zc300_get_spr(TEST_AXIS)
    print(f"zc300_get_spr({TEST_AXIS}) -> 每转脉冲数 = {spr} pp/r")

    ## 传动比不适用，导程需要除以实际传动比
    # 21. zc300_set_ratio() - 设置传动比（电机转数/丝杠转数，无减速器时为1）
    ratio_val = 1
    set_ratio_ok = zc.zc300_set_ratio(TEST_AXIS, ratio_val)
    print(f"zc300_set_ratio({TEST_AXIS}, {ratio_val}) -> {'成功' if set_ratio_ok else '失败'}")

    # 22. zc300_get_ratio() - 读取传动比
    ratio = zc.zc300_get_ratio(TEST_AXIS)
    print(f"zc300_get_ratio({TEST_AXIS}) -> 传动比 = {ratio}")
    print()

    # ==================== 运动速度参数（初速、常速、加速度） ====================
    print_separator("运动速度参数")

    # 23. zc300_set_init_speed() - 设置起始速度（mm/s）
    init_speed = 2
    set_init_ok = zc.zc300_set_init_speed(TEST_AXIS, init_speed)
    print(f"zc300_set_init_speed({TEST_AXIS}, {init_speed}) -> {'成功' if set_init_ok else '失败'}")

    # 24. zc300_get_init_speed() - 读取起始速度
    init_speed_get = zc.zc300_get_init_speed(TEST_AXIS)
    print(f"zc300_get_init_speed({TEST_AXIS}) -> 起始速度 = {init_speed_get} mm/s")

    # 25. zc300_set_move_speed() - 设置运行速度
    move_speed = 20
    set_move_ok = zc.zc300_set_move_speed(TEST_AXIS, move_speed)
    print(f"zc300_set_move_speed({TEST_AXIS}, {move_speed}) -> {'成功' if set_move_ok else '失败'}")

    # 26. zc300_get_move_speed() - 读取运行速度
    move_speed_get = zc.zc300_get_move_speed(TEST_AXIS)
    print(f"zc300_get_move_speed({TEST_AXIS}) -> 运行速度 = {move_speed_get} mm/s")

    # 27. zc300_set_acc_speed() - 设置加速度（mm/s²）
    acc_speed = 10
    set_acc_ok = zc.zc300_set_acc_speed(TEST_AXIS, acc_speed)
    print(f"zc300_set_acc_speed({TEST_AXIS}, {acc_speed}) -> {'成功' if set_acc_ok else '失败'}")

    # 28. zc300_get_acc_speed() - 读取加速度
    acc_speed_get = zc.zc300_get_acc_speed(TEST_AXIS)
    print(f"zc300_get_acc_speed({TEST_AXIS}) -> 加速度 = {acc_speed_get} mm/s²")
    print()

    # ==================== 回原点参数 ====================
    print_separator("回原点参数")

    # 29. zc300_set_home_speed() - 设置回原点速度
    home_speed = 20
    set_home_speed_ok = zc.zc300_set_home_speed(TEST_AXIS, home_speed)
    print(f"zc300_set_home_speed({TEST_AXIS}, {home_speed}) -> {'成功' if set_home_speed_ok else '失败'}")

    # 30. zc300_get_home_speed() - 读取回原点速度
    home_speed_get = zc.zc300_get_home_speed(TEST_AXIS)
    print(f"zc300_get_home_speed({TEST_AXIS}) -> 回原点速度 = {home_speed_get} mm/s")

    # 31. zc300_set_home_mode() - 设置归零方式（限位归零）
    set_home_mode_ok = zc.zc300_set_home_mode(TEST_AXIS, HOME_BY_ZERO)
    print(f"zc300_set_home_mode({TEST_AXIS}, HOME_BY_ZERO) -> {'成功' if set_home_mode_ok else '失败'}")

    # 32. zc300_get_home_mode() - 读取归零方式
    home_mode = zc.zc300_get_home_mode(TEST_AXIS)
    mode_str = {HOME_BY_USER: "用户零点", HOME_BY_LIMIT: "限位", HOME_BY_ZERO: "零点开关"}.get(home_mode, "未知")
    print(f"zc300_get_home_mode({TEST_AXIS}) -> 归零方式 = {mode_str} ({home_mode})")
    print()

    # ==================== 运动控制 ====================
    print_separator("运动控制（请确保滑台行程安全）")

    # 33. zc300_move() - 相对运动 5mm
    distance = 1
    move_ok = zc.zc300_move(TEST_AXIS, MOVE_RELATIVE, -distance)
    print(f"zc300_move({TEST_AXIS}, MOVE_RELATIVE, {distance}) -> 相对移动{'成功' if move_ok else '失败'}")
    if move_ok:
        # 等待运动完成
        while not zc.zc300_get_idle(TEST_AXIS):  # 只要电移台没有空闲，表示它在移动
            # print("moving...",axis)
            time.sleep(0.5)  # 每 0.5 秒检查一次
        # 34. zc300_stop() - 停止运动（这里不实际停止，仅演示）
        # stop_ok = zc.zc300_stop(TEST_AXIS, STOP_SLOWLY)
        # print(f"zc300_stop({TEST_AXIS}, STOP_SLOWLY) -> {'成功' if stop_ok else '失败'}")
        print("（运动完成后未调用停止，避免中断移动）")

    # 35. zc300_home() - 回原点（小心执行）
    print("\n⚠️ 即将执行回原点操作，请确保滑台安全！")
    user_input = input("是否执行回原点？(y/n): ")
    if user_input.lower() == 'y':
        home_ok = zc.zc300_home(TEST_AXIS)
        print(f"zc300_home({TEST_AXIS}) -> 回原点{'成功' if home_ok else '失败'}")
        while not zc.zc300_get_idle(TEST_AXIS):  # 只要电移台没有空闲，表示它在移动
            # print("moving...",axis)
            time.sleep(0.5)  # 每 0.5 秒检查一次
    else:
        print("已跳过回原点操作。")
    print()

    # ==================== 位置读写 ====================
    print_separator("位置读写")

    # 36. zc300_set_position() - 设置当前位置为自定义坐标（例如设为 10.0 mm）
    custom_pos = 10.0
    set_pos_ok = zc.zc300_set_position(TEST_AXIS, custom_pos)
    print(f"zc300_set_position({TEST_AXIS}, {custom_pos}) -> 当前位置设为 {custom_pos}: {'成功' if set_pos_ok else '失败'}")

    # 37. zc300_get_position() - 读取当前位置
    current_pos = zc.zc300_get_position(TEST_AXIS)
    print(f"zc300_get_position({TEST_AXIS}) -> 当前位置 = {current_pos} mm")
    print()

    # ==================== 状态查询 ====================
    print_separator("状态查询")

    # 38. zc300_get_idle() - 判断轴是否空闲
    idle = zc.zc300_get_idle(TEST_AXIS)
    print(f"zc300_get_idle({TEST_AXIS}) -> 空闲状态: {idle} (True=空闲)")

    # 39. zc300_get_status() - 获取控制器所有轴的状态位
    status = zc.zc300_get_status()
    print(f"zc300_get_status() -> 状态码: 0x{status:04X}")
    # 解析常用状态位（根据头文件）
    if status & 0x0001: print("  - X轴正限位触发")
    if status & 0x0002: print("  - X轴负限位触发")
    if status & 0x0004: print("  - X轴原点触发")
    if status & 0x0200: print("  - 紧急停止触发")
    if status & 0x0400: print("  - X轴报警")
    # 可按需扩展其他位
    print()

    # ==================== 蜂鸣器控制 ====================
    print_separator("蜂鸣器控制")

    # 40. zc300_set_buzzer() - 设置蜂鸣器异常时是否鸣叫（1=鸣叫）
    set_buzzer_ok = zc.zc300_set_buzzer(1)
    print(f"zc300_set_buzzer(1) -> 异常时鸣叫: {'成功' if set_buzzer_ok else '失败'}")

    # 41. zc300_get_buzzer() - 读取蜂鸣器设置
    buzzer_status = zc.zc300_get_buzzer()
    print(f"zc300_get_buzzer() -> 蜂鸣器设置: {buzzer_status} (1=鸣叫,0=静音)")
    print()

    # ==================== 断开连接 ====================
    print_separator("断开连接")
    # 42. zc300_close() - 关闭连接
    zc.zc300_close()
    print("zc300_close() -> 已断开控制器连接")
    print("✅ 所有函数测试执行完毕（部分操作可能因硬件缺失而失败，属于正常现象）")

if __name__ == "__main__":
    test_all_functions()