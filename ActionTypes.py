# -*- coding: utf-8 -*-
"""Central action and acquisition type names shared across worker threads."""

EXIT_ACTION = "exit"


class AcqTypes:
    CONTINUOUS_ALINE = "ContinuousAline"
    FINITE_ALINE = "FiniteAline"
    CONTINUOUS_BLINE = "ContinuousBline"
    FINITE_BLINE = "FiniteBline"
    CONTINUOUS_CSCAN = "ContinuousCscan"
    FINITE_CSCAN = "FiniteCscan"
    LOCATION_CAMERA_LIVE = "LocationCameraLive"
    MOSAIC = "Mosaic"
    PLATE_PRESCAN = "PlatePreScan"
    PLATE_SCAN = "PlateScan"
    TIMED_PLATE_SCAN = "TimedPlateScan"
    WELL_SCAN = "WellScan"


class WeaverActions:
    ZSTAGE_REPEATIBILITY = "ZstageRepeatibility"
    GET_BACKGROUND = "get_background"
    GET_SURFACE = "get_surface"
    GOTO_ZERO = "Gotozero"


class GPUActions:
    GPU = "GPU"
    CPU = "CPU"
    CLEAR = "Clear"
    UPDATE_DISPERSION = "update_Dispersion"
    UPDATE_BACKGROUND = "update_background"
    DISPLAY_FFT_ACTIONS = "display_FFT_actions"
    DISPLAY_COUNTS = "display_counts"
    INIT_MOSAIC = "Init_Mosaic"


class DnSActions:
    PROCESS_MOSAIC = "Process_Mosaic"
    RETURN_MOSAIC = "Return_mosaic"
    CLEAR = "Clear"
    DISPLAY_COUNTS = "display_counts"
    AGAR_TILE = "agarTile"
    WRITE_AGAR = "WriteAgar"
    INIT_MOSAIC = "Init_Mosaic"
    SAVE_MOSAIC = "Save_mosaic"
    MOSAIC = "Mosaic"
    DISPLAY_MOSAIC = "display_mosaic"
