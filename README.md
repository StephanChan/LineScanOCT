# LineScanOCT Control Software

Control software for a custom line-scan OCT / microscopy system. The program coordinates camera acquisition, galvo and stage motion, GPU FFT processing, display, saving, sample localization, and plate/well scan workflows.

The main entry point is `OCT_MT.py`.

## System Overview

The system uses PyQt threads plus queues to coordinate hardware and processing:

- `OCT_MT.py`: main GUI application, thread startup, UI signal routing, display bridge.
- `ThreadWeaver.py`: high-level scan orchestration. This is the "weaver" that sequences camera, AODO, GPU, saving, stage motion, plate scan, well scan, pre-scan, and ROI refinement.
- `ThreadCamera_DH.py`: Daheng camera acquisition path.
- `ThreadCamera.py`: PhotonFocus camera acquisition path.
- `ThreadAODO_art.py`: ART-DAQ/AODO galvo trigger and digital-output task control, plus motorized stage commands.
- `ThreadGPU.py`: OCT spectral processing using GPU acceleration, with a CPU fallback path.
- `ThreadDnS.py`: display-and-save worker for processed data and raw data.
- `Actions.py`: queue message/action classes shared between threads.

The raw spectral data path is based on shared global memory slots. Camera threads write acquired frames into `Memory[memory_slot]`; downstream threads pass the memory slot index through queues instead of copying the full data array.

## Hardware Coordinate Model

The OCT scan coordinate system is:

- X: line illumination / camera line direction.
- Y: galvo scan direction.
- Z: axial stage / focus direction.

For plate and well scans, the software combines three coordinate systems:

- OCT galvo/FOV coordinates.
- Motorized stage coordinates.
- USB camera image coordinates for sample localization.

The current USB camera display is transformed so the well plate view matches the physical top view. In the USB image display, the horizontal axis corresponds to stage Y and the vertical axis corresponds to stage X. The mosaic display is adjusted for user-facing consistency, while data stitching conventions are kept explicit for future downstream analysis.

## Key Modules

### Hardware Specs

Hardware constants are centralized in:

- `HardwareSpecs.py`

This includes:

- Objective calibration for galvo angle-to-mm conversion.
- Camera pixel size and sensor height.
- Stage axis indices and initialization speeds.
- AODO trigger terminals and default AODO constants.
- Laser A-line frequency values.

When changing objective, camera, stage-axis, laser, or AODO constants, prefer editing `HardwareSpecs.py` instead of scattering hardcoded values across control scripts.

### Display Rendering

Display logic that should run on the GUI thread is centralized in:

- `Display_rendering.py`

This includes:

- A-line display rendering.
- B-line display rendering.
- C-scan and mosaic display rendering.
- Dynamic-signal RGB overlay rendering.
- USB ROI and FOV overlay rendering.
- Mosaic correction overlay rendering.
- AODO waveform preview rendering.

Worker threads should avoid directly creating or setting `QPixmap` objects. Instead, they emit payloads through `UiBridge` in `OCT_MT.py`, and the GUI thread calls the rendering helpers.

### Sample Localization And Mosaic Planning

Sample localization and FOV generation are split into:

- `SampleLocator.py`: USB camera coordinate conversion, ROI/FOV overlay helpers, and initial sample localization logic.
- `mosaic_scan_planner.py`: FOV planning for ROIs, including coverage margin, overlap, Y-FOV resizing, and tile acceptance.
- `mosaic_correction.py`: ROI correction on acquired mosaic images and conversion back into stage/FOV coordinates.
- `InteractiveWidget.py`: interactive mosaic image display and polygon drawing.

The FOV planner supports the current rule:

- Use one FOV if the ROI fits inside the configured coverage fraction.
- Use multiple FOVs when the ROI exceeds that coverage.
- Allow software-adjusted Y length up to the configured limit.
- Use overlap and polygon intersection to avoid unnecessary tiles for irregular ROIs.

### GPU/CPU OCT Processing

`ThreadGPU.py` handles FFT processing for A-line, B-line, C-scan, plate scan, and well scan data.

Current processing features include:

- Chunked GPU FFT processing to reduce memory pressure.
- Optional overlapped host/device transfer and GPU processing.
- Reused GPU buffers where practical.
- Dynamic-mode background strategy using the first acquired frame when enabled.
- Frame-mean normalization for dynamic processing.
- CUDA-accelerated dynamic signal processing helper.
- CPU FFT fallback intended to mirror the GPU routine.

The processed output is still returned to `ThreadDnS.py` as assembled NumPy arrays, so the display/save layer does not need to know whether GPU chunking was used internally.

## Acquisition Modes

Common modes include:

- `ContinuousAline`
- `FiniteAline`
- `ContinuousBline`
- `FiniteBline`
- `ContinuousCscan`
- `FiniteCscan`
- `PlatePreScan`
- `PlateScan`
- `WellScan`

`acq_mode` is passed explicitly through queue actions so downstream display/save code can distinguish the user-level acquisition mode from lower-level processing actions such as `Process_Mosaic`.

## Plate / Well Scan Workflow

Typical plate scan workflow:

1. Use the USB camera sample locator to draw ROIs on the well plate image.
2. Generate initial FOV locations from the ROI polygons.
3. Run `PlatePreScan` to quickly image each ROI.
4. If needed, redraw/refine ROI on the mosaic display.
5. Use the corrected ROI to generate a new scan grid.
6. Run final plate or well acquisition and save data.

The software can also rerun `PlatePreScan` from existing FOV locations, or load previously saved mosaic/FOV data from the current directory when no in-memory FOV list exists.

## Dynamic Signal Mode

The UI `DynCheckBox` changes acquisition and processing behavior.

For B-line and C-scan display, dynamic signal can be overlaid on intensity images using an RGB overlay. The `DynContrast` slider controls the overlay alpha.

For plate and well scans, dynamic processing is generally delayed until after acquisition because on-the-fly dynamic processing is too expensive for large multi-sample scans. The raw frame stack is still acquired in a way that supports later dynamic analysis.

## Generated Files And Logs

The project creates local runtime files that should generally not be committed:

- `log_files/`: application log files created by the Python `LOG` class.
- `ACTS2210.log`: ART-DAQ driver/DLL diagnostic log. This is generated by the ART hardware library, not by the Python application logger.
- `__pycache__/`: Python bytecode cache.

Data files are saved under the directory selected in the UI. Mosaic folders may include processed data, raw USB images, ROI metadata, and FOV-location files.

## Running

Install the Python environment from:

- `python311_env.yml`

Then run:

```bash
python OCT_MT.py
```

The full hardware setup requires the camera SDKs, ART-DAQ library, GPU/CuPy environment, and stage-control dependencies to be installed on the acquisition computer.

## Notes For Development

- Keep UI-only rendering in `Display_rendering.py`.
- Keep hardware constants in `HardwareSpecs.py`.
- Keep high-level sequencing in `ThreadWeaver.py`.
- Keep raw acquisition in camera threads.
- Keep OCT spectral processing in `ThreadGPU.py`.
- Keep display/save output handling in `ThreadDnS.py`.
- Avoid writing GUI widgets directly from worker threads; use `UiBridge` signals.
- Avoid large data copies between threads; pass memory-slot indices when possible.

This is active research software, so the architecture prioritizes practical control of the custom instrument while gradually moving hardware constants, display rendering, and planning logic into clearer modules.
