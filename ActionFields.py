# -*- coding: utf-8 -*-
"""Queue action field classes shared across worker threads."""

from ActionTypes import EXIT_ACTION


class AODOActionField:
    def __init__(self, action, direction=1):
        super().__init__()
        self.action = action
        self.direction = direction


class WeaverActionField:
    def __init__(self, action, acq_mode=None, context=None):
        super().__init__()
        self.action = action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.context = [] if context is None else context


class DnSActionField:
    def __init__(self, action, acq_mode=None, data=None, raw=False, dynamic=None, context=None, gpu_avg_count=1, dynamic_bline_idx=None, filename_bundle=None, skip_save=False):
        super().__init__()
        self.action = action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.data = [] if data is None else data
        self.raw = raw
        self.dynamic = [] if dynamic is None else dynamic
        self.context = [] if context is None else context
        self.gpu_avg_count = gpu_avg_count
        self.dynamic_bline_idx = dynamic_bline_idx
        self.filename_bundle = {} if filename_bundle is None else filename_bundle
        self.skip_save = bool(skip_save)


class GPUActionField:
    def __init__(self, action, DnS_action='', acq_mode=None, memory_slot=0, context=None, dynamic_bline_idx=None, filename_bundle=None, skip_save=False):
        super().__init__()
        self.action = action
        self.DnS_action = DnS_action
        self.acq_mode = DnS_action if acq_mode is None else acq_mode
        self.memory_slot = memory_slot
        self.context = [] if context is None else context
        self.dynamic_bline_idx = dynamic_bline_idx
        self.filename_bundle = {} if filename_bundle is None else filename_bundle
        self.skip_save = bool(skip_save)


class DActionField:
    def __init__(self, action):
        super().__init__()
        self.action = action


class DbackActionField:
    def __init__(self, memory_slot):
        super().__init__()
        self.memory_slot = memory_slot


class EXITField:
    def __init__(self):
        super().__init__()
        self.action = EXIT_ACTION
