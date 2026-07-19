import os

from ActionTypes import AcqTypes


SAVE_SAMPLE_TIME_MODES = (
    AcqTypes.PLATE_PRESCAN,
    AcqTypes.PLATE_SCAN,
    AcqTypes.WELL_SCAN,
    AcqTypes.TIMED_PLATE_SCAN,
)


def sample_time_save_dir(base_dir, sample_id, time_number):
    folder_path = os.path.join(
        base_dir,
        f"sampleID-{sample_id}",
        f"Time-{time_number}",
    )
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def _shape3(shape):
    return int(shape[0]), int(shape[1]), int(shape[2])


def sample_filename(tile_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"tile-{tile_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def sample_dyn_filename(tile_num, dynamic_bline_idx, shape):
    yrpt, xpix, zpix = _shape3(shape)
    return (
        f"tile-{tile_num}-Bline-{dynamic_bline_idx}-Yrpt{yrpt}-X{xpix}-Z{zpix}.tif"
    )


def tile_dynamic_volume_filename(tile_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"tile-{tile_num}-Dyn-Y{ypix}-X{xpix}-Z{zpix}.tif"


def tile_dynamic_rgb_volume_filename(tile_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"tile-{tile_num}-DynRGB-Y{ypix}-X{xpix}-Z{zpix}.tif"


def tile_mean_volume_filename(tile_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"tile-{tile_num}-Mean-Y{ypix}-X{xpix}-Z{zpix}.tif"


def cscan_filename(cscan_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"Cscan-{cscan_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def cscan_dyn_filenames(cscan_num, dynamic_bline_idx, ypixels, shape):
    yrpt, xpix, zpix = _shape3(shape)
    dyn_filename = f"CscanDyn-{cscan_num}-Y{int(ypixels)}-X{xpix}-Z{zpix}.tif"
    bline_filename = (
        f"Cscan-{cscan_num}-Bline-{dynamic_bline_idx}-Yrpt{yrpt}-X{xpix}-Z{zpix}.tif"
    )
    return bline_filename, dyn_filename


def cscan_mean_volume_filename(cscan_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"CscanMean-{cscan_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def cscan_dynamic_rgb_volume_filename(cscan_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"CscanDynRGB-{cscan_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def bline_filename(bline_num, shape):
    yrpt, xpix, zpix = _shape3(shape)
    return f"Bline-{bline_num}-Yrpt{yrpt}-X{xpix}-Z{zpix}.tif"


def bline_dyn_filename(bline_num, shape):
    _, xpix, zpix = _shape3(shape)
    return f"BlineDyn-{bline_num}-X{xpix}-Z{zpix}.tif"


def bline_dyn_rgb_filename(bline_num, shape):
    _, xpix, zpix = _shape3(shape)
    return f"BlineDynRGB-{bline_num}-X{xpix}-Z{zpix}.tif"


def aline_filename(aline_num, shape):
    yrpt, xrpt, zpix = _shape3(shape)
    return f"Aline-{aline_num}-Yrpt{yrpt}-Xrpt{xrpt}-Z{zpix}.tif"


class FileNaming:
    def __init__(self, ui):
        self.ui = ui
        self.tile_num = 1
        self.aline_num = 1
        self.bline_num = 1
        self.cscan_num = 1
        self.dynamic_bline_idx = 1
        self.last_sample_id = None
        self.last_time_number = None

    def _base_dir(self):
        return self.ui.DIR.toPlainText()

    def _current_sample_id(self):
        selector = getattr(self.ui, "sampleSelector", None)
        if selector is None:
            return 1
        index = int(selector.currentIndex())
        return max(1, index + 1)

    def _current_time_number(self):
        return self._current_time_number_for_acq_mode(None)

    def _current_time_number_for_acq_mode(self, acq_mode):
        if acq_mode == AcqTypes.PLATE_PRESCAN:
            return 0
        if hasattr(self.ui, "timeReader"):
            return int(self.ui.timeReader.value())
        if hasattr(self.ui, "CuSlice"):
            return int(self.ui.CuSlice.value())
        return 1

    def _sync_context(self, acq_mode):
        if acq_mode not in SAVE_SAMPLE_TIME_MODES:
            return
        sample_id = self._current_sample_id()
        time_number = self._current_time_number_for_acq_mode(acq_mode)
        if sample_id != self.last_sample_id or time_number != self.last_time_number:
            self.reset_all_counters()
            self.last_sample_id = sample_id
            self.last_time_number = time_number

    def save_dir(self, acq_mode):
        self._sync_context(acq_mode)
        base_dir = self._base_dir()
        if acq_mode in SAVE_SAMPLE_TIME_MODES:
            return sample_time_save_dir(
                base_dir,
                self._current_sample_id(),
                self._current_time_number_for_acq_mode(acq_mode),
            )
        return base_dir

    def reset_all_counters(self):
        self.tile_num = 1
        self.aline_num = 1
        self.bline_num = 1
        self.cscan_num = 1
        self.dynamic_bline_idx = 1

    def reset_tilenum(self):
        self.tile_num = 1
        
    def reset_dynamic_bline_idx(self):
        self.dynamic_bline_idx = 1

    def increment_tile(self):
        self.tile_num += 1

    def increment_aline(self):
        self.aline_num += 1

    def increment_bline(self):
        self.bline_num += 1

    def increment_cscan(self):
        self.cscan_num += 1
    
    def increment_dynY(self):
        self.dynamic_bline_idx += 1

    def advance_tile_dynamic_bline(self, ypixels):
        self.dynamic_bline_idx += 1
        if self.dynamic_bline_idx > int(ypixels):
            self.dynamic_bline_idx = 1
            self.increment_tile()

    def advance_cscan_dynamic_bline(self, ypixels):
        self.dynamic_bline_idx += 1
        if self.dynamic_bline_idx > int(ypixels):
            self.dynamic_bline_idx = 1
            self.increment_cscan()

    def get_filename(self, kind, acq_mode, shape, dynamic_bline_idx=None, ypixels=None):
        base_dir = self.save_dir(acq_mode)

        if kind == "aline":
            return os.path.join(base_dir, aline_filename(self.aline_num, shape))
        if kind == "bline":
            return os.path.join(base_dir, bline_filename(self.bline_num, shape))
        if kind == "bline_dyn":
            return os.path.join(base_dir, bline_dyn_filename(self.bline_num, shape))
        if kind == "bline_dyn_rgb":
            return os.path.join(base_dir, bline_dyn_rgb_filename(self.bline_num, shape))
        if kind == "cscan":
            return os.path.join(base_dir, cscan_filename(self.cscan_num, shape))
        if kind == "cscan_bline":
            if dynamic_bline_idx is None:
                dynamic_bline_idx = self.dynamic_bline_idx
            if ypixels is None:
                raise ValueError("ypixels is required for cscan_bline filenames")
            bline_name, _ = cscan_dyn_filenames(
                self.cscan_num,
                dynamic_bline_idx,
                ypixels,
                shape,
            )
            return os.path.join(base_dir, bline_name)
        if kind == "cscan_dyn":
            if dynamic_bline_idx is None:
                dynamic_bline_idx = 0
            if ypixels is None:
                raise ValueError("ypixels is required for cscan_dyn filenames")
            _, dyn_name = cscan_dyn_filenames(
                self.cscan_num,
                dynamic_bline_idx,
                ypixels,
                shape,
            )
            return os.path.join(base_dir, dyn_name)
        if kind == "cscan_mean":
            return os.path.join(base_dir, cscan_mean_volume_filename(self.cscan_num, shape))
        if kind == "cscan_dyn_rgb":
            return os.path.join(base_dir, cscan_dynamic_rgb_volume_filename(self.cscan_num, shape))
        if kind == "sample":
            return os.path.join(base_dir, sample_filename(self.tile_num, shape))
        if kind == "sample_dyn":
            if dynamic_bline_idx is None:
                dynamic_bline_idx = self.dynamic_bline_idx
            return os.path.join(
                base_dir,
                sample_dyn_filename(self.tile_num, dynamic_bline_idx, shape),
            )
        if kind == "tile_dyn":
            return os.path.join(base_dir, tile_dynamic_volume_filename(self.tile_num, shape))
        if kind == "tile_dyn_rgb":
            return os.path.join(base_dir, tile_dynamic_rgb_volume_filename(self.tile_num, shape))
        if kind == "tile_mean":
            return os.path.join(base_dir, tile_mean_volume_filename(self.tile_num, shape))

        raise ValueError(f"Unknown filename kind: {kind}")
