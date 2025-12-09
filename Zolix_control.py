# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:28:32 2025

@author: BITU-OCT2
"""
import devices_zc300 as de
import time
import time
zc = de.ZC300()
# 假设 ZC300 类已经定义好了
class Stepper:
    def __init__(self):
        self.idx = 1000/3.1395
        self.unit = 0
        dev_info=b"COM3" 
        # 连接设备
        if not zc.zc300_open(dev_info, 1):
            print("设备连接失败！")


        
    def move(self,axis,distance,speed):

        zc.zc300_set_unit(axis, self.unit) 
        zc.zc300_set_move_speed(axis,speed*self.idx)
        distance = round(distance*self.idx)
        a =  zc.zc300_move(axis, 1, distance)  
        if a == False:
            print('move',axis,'failed')
        while not zc.zc300_get_idle(axis):  # 只要电移台没有空闲，表示它在移动
            # print("moving...",axis)
            time.sleep(0.5)  # 每 0.5 秒检查一次
        return 0
    
    def home(self,axis,speed):

        zc.zc300_set_unit(axis, self.unit) 
        # zc.zc300_set_move_speed(axis,speed*self.idx)
        zc.zc300_set_home_speed(axis,speed*self.idx)  # 1表示相对移动，5mm的移动
        zc.zc300_set_home_mode(axis,2)
        a = zc.zc300_home(axis)
        print(a)
        if a == False:
            print('move',axis,a,'failed')
        while not zc.zc300_get_idle(axis):  # 只要电移台没有空闲，表示它在移动
            # print("moving...",axis)
            time.sleep(0.5)  # 每 0.5 秒检查一次
        return 0
    def enable(self,axis,state = True):
        zc.zc300_set_enabled(axis,state)

if __name__ == "__main__":
    st = Stepper()
    st.enable(0,True)

    st.move(0,10,5)
    st.home(0,200)
    # 断开连接
    zc.zc300_close()
    
    print("X轴操作完成！")
