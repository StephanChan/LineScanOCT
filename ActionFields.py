# -*- coding: utf-8 -*-
"""Queue action field classes shared across worker threads."""

from ActionTypes import EXIT_ACTION


class AODOActionField:
    def __init__(self, action, direction=1, acq_mode=None):
        super().__init__()
        self.action = action
        self.direction = direction
        # Effective acquisition-mode string for ConfigTask waveform generation
        # (e.g. "FastVolumeCscan" for plate-scan FOVs routed through FastVolume).
        self.acq_mode = acq_mode


class WeaverActionField:
    def __init__(self, action, acq_mode=None, context=None):
        super().__init__()
        self.action = action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.context = [] if context is None else context


class DnSActionField:
    def __init__(self, action, acq_mode=None, data=None, raw=False, dynamic=None, context=None, gpu_avg_count=1, dynamic_bline_idx=None, filename_bundle=None, skip_save=False, fast_volume=False, per_y_dynamic=False):
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
        # Explicit acquisition flags set once by the Weaver at dispatch time.
        self.fast_volume = bool(fast_volume)
        self.per_y_dynamic = bool(per_y_dynamic)


class GPUActionField:
    def __init__(self, action, DnS_action='', acq_mode=None, memory_slot=0, context=None, dynamic_bline_idx=None, filename_bundle=None, skip_save=False, fast_volume=False, per_y_dynamic=False):
        super().__init__()
        self.action = action
        self.DnS_action = DnS_action
        self.acq_mode = DnS_action if acq_mode is None else acq_mode
        self.memory_slot = memory_slot
        self.context = [] if context is None else context
        self.dynamic_bline_idx = dynamic_bline_idx
        self.filename_bundle = {} if filename_bundle is None else filename_bundle
        self.skip_save = bool(skip_save)
        # Explicit acquisition flags set once by the Weaver at dispatch time.
        self.fast_volume = bool(fast_volume)
        self.per_y_dynamic = bool(per_y_dynamic)


class DActionField:
    def __init__(self, action, acq_mode=None, per_y_dynamic=False):
        super().__init__()
        self.action = action
        # Effective acquisition-mode string (e.g. "FastVolumeCscan" for
        # plate-scan FOVs routed through FastVolume) and the explicit per-Y
        # (dynamic) layout flag used by the camera thread.
        self.acq_mode = acq_mode
        self.per_y_dynamic = bool(per_y_dynamic)


class DbackActionField:
    def __init__(self, memory_slot, error=None):
        super().__init__()
        self.memory_slot = memory_slot
        self.error = error


class EXITField:
    def __init__(self):
        super().__init__()
        self.action = EXIT_ACTION
