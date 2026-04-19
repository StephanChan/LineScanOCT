# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:43:51 2023

@author: admin
"""
# here defines all the legitimate actions that can be put into a queue

import numpy as np

class AODOAction():
    def __init__(self, action, direction = 1):
        super().__init__()
        self.action=action
        self.direction = direction

        

class WeaverAction():
    def __init__(self, action, payload=[]):
        super().__init__()
        self.action = action
        self.payload = payload
        
class DnSAction():
    def __init__(self, action, acq_mode=None, data=[], raw = False, dynamic =[], payload = []):
        super().__init__()
        self.action=action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.data = data
        self.raw = raw
        self.dynamic = dynamic
        self.payload = payload

class GPUAction():
    def __init__(self, action, DnS_action='', acq_mode=None, memory_slot=0, payload=[]):
        super().__init__()
        self.action = action
        self.DnS_action = DnS_action
        self.acq_mode = DnS_action if acq_mode is None else acq_mode
        self.memory_slot = memory_slot
        self.payload = payload
        
class DAction():
    def __init__(self, action):
        super().__init__()
        self.action = action 
        
class DbackAction():
    def __init__(self, memory_slot):
        super().__init__()
        self.memory_slot = memory_slot
        
class EXIT():
    def __init__(self):
        super().__init__()
        self.action='exit'
