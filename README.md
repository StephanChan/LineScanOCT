# LineScanOCT Control Software

Control software for a custom line-scan OCT / microscopy system. The application coordinates:

- camera acquisition
- galvo / stage motion
- GPU or CPU OCT FFT processing
- display rendering
- file saving
- plate / well scan workflows
- USB-camera-based sample localization
- timed offline dynamic processing

Main entry point:

- `OCT_MT.py`

## Current Architecture

The codebase is organized around worker threads plus queue messages.

High-level data flow:

1. Camera thread acquires raw spectra into shared memory slots.
2. `ThreadWeaver.py` orchestrates acquisition and decides what happens next.
3. `ThreadGPU.py` processes raw spectral data into OCT intensity / dynamic outputs.
4. `ThreadDnS.py` interprets processed payloads, prepares save outputs, and emits display payloads.
5. `Display_rendering.py` runs on the GUI thread and renders the final display.

The raw data path uses shared global memory:

- camera threads write into `Memory[memory_slot]`
- downstream threads pass memory-slot indices through queues instead of copying large raw arrays between threads

## Core Modules

### Entry / GUI

- `OCT_MT.py`
  - application entry point
  - thread construction and wiring
  - queue creation
  - UI signal hookups
  - `UiBridge` signals for thread-safe display/status updates

- `mainWindow.py`
  - main GUI composition and widget setup

- `GUI.py`, `GUI.ui`
  - generated / designed Qt UI definition

### Queue Actions And Type Catalog

- `ActionFields.py`
  - queue payload classes:
    - `WeaverActionField`
    - `GPUActionField`
    - `DnSActionField`
    - `AODOActionField`
    - `DActionField`
    - `DbackActionField`
    - `EXITField`

- `ActionTypes.py`
  - single source of truth for named action / acquisition strings
  - contains:
    - `AcqTypes`
    - `WeaverActions`
    - `GPUActions`
    - `DnSActions`
    - `EXIT_ACTION`

This replaced the earlier split between mode strings and action strings.

### Orchestration

- `ThreadWeaver.py`
  - top-level acquisition orchestration
  - routes acquisition commands from `WeaverQueue`
  - manages:
    - single finite scans
    - continuous scans
    - plate pre-scan
    - plate scan
    - timed plate scan
    - well scan
    - mosaic correction loop
    - sample-by-sample stage sequencing
  - owns processing barriers so new acquisition does not overwrite data still in flight

### Acquisition

- `ThreadCamera_DH.py`
  - Daheng camera acquisition path
  - writes raw data into shared memory
  - uses multiple consumer workers for packed conversion and memory write

- `ThreadCamera.py`
  - PhotonFocus camera path

- `ThreadAODO_art.py`
  - ART-DAQ / AODO control
  - galvo waveform output
  - digital trigger output
  - stage move commands

### Processing

- `ThreadGPU.py`
  - OCT spectral-domain processing
  - GPU FFT with CPU fallback
  - background subtraction
  - normalization
  - interpolation / dispersion compensation
  - dynamic processing
  - pre-FFT averaging
  - chunked processing and overlapped transfer/FFT path

- `DataShape.py`
  - canonical interpretation of payload shape
  - computes:
    - frame count
    - repeat count
    - Y count
    - X count
    - Z count


### Display And Save

- `ThreadDnS.py`
  - consumes processed payloads from GPU / raw payloads from Weaver
  - interprets payload semantics
  - manages:
    - A-line / B-line / C-scan output
    - mosaic accumulation
    - file saving
    - display payload emission to GUI thread

- `Display_rendering.py`
  - GUI-thread rendering only
  - owns:
    - A-line rendering
    - B-line rendering
    - C-scan rendering
    - mosaic rendering
    - dynamic overlay rendering
    - USB ROI overlay rendering
    - mosaic correction overlay rendering
    - AODO waveform preview rendering

- `FileNaming.py`
  - file naming and save-folder layout
  - centralizes:
    - sample/time directories
    - tile filenames
    - B-line / C-scan filenames
    - dynamic filenames

### Plate Session / Offline Processing

- `ScanSession.py`
  - session save / load helpers
  - sample selector population
  - USB training-data export

- `DynamicPostprocessing.py`
  - timed idle processing helpers
  - processes saved plate/well data after acquisition
  - writes:
    - tile dynamic volumes
    - tile mean-intensity volumes
    - stitched sample-level dynamic / mean volumes

### Sample Localization And Scan Planning

- `SampleLocator.py`
  - USB camera coordinate model
  - stage/image coordinate conversion
  - ROI drawing and FOV overlay
  - initial sample localization workflow

- `mosaic_scan_planner.py`
  - ROI-to-FOV planning
  - overlap policy
  - occupancy policy
  - Y-FOV resizing
  - candidate acceptance

- `mosaic_correction.py`
  - converts mosaic-drawn polygons back into stage/FOV coordinates

- `InteractiveWidget.py`
  - interactive mosaic image view and polygon editing

### Structured Domain Models

- `ScanModels.py`
  - explicit dataclasses:
    - `SampleCenter`
    - `FOVLocation`

These are now strict dataclasses. Dict-style compatibility was removed.

### Hardware / Calibration / Generic Helpers

- `HardwareSpecs.py`
  - centralized physical and processing defaults
  - includes:
    - objective specs
    - camera specs
    - stage-axis specs
    - laser specs
    - AODO trigger constants
    - OCT processing defaults now shared by GPU / GUI setup

- `Generaic_functions.py`
  - legacy shared numerical / plotting helpers used across the project

- `CalibInterpolationDispersion.py`
  - calibration support for interpolation / dispersion data used by GPU processing

## Acquisition Modes

Defined centrally in `ActionTypes.py` under `AcqTypes`.

Main acquisition modes:

- `ContinuousAline`
- `FiniteAline`
- `ContinuousBline`
- `FiniteBline`
- `ContinuousCscan`
- `FiniteCscan`
- `LocationCameraLive`
- `Mosaic`
- `PlatePreScan`
- `PlateScan`
- `TimedPlateScan`
- `WellScan`

## Control Routine By Acquisition Mode

This section is intended to mirror the current control logic closely enough to drive a control-flow diagram.

### Common thread path

1. `OCT_MT.py` enqueues a `WeaverActionField(...)` into `WeaverQueue`.
2. `ThreadWeaver.py::QueueOut()` selects the acquisition branch, waits on `wait_for_processing_barrier(...)` where required, and starts orchestration.
3. Raw spectra are written by the camera thread into shared `Memory[memory_slot]`, and Weaver dispatches either:
   - `GPUActionField(...)` to `GPUQueue`, or
   - `DnSActionField(..., raw=True)` directly to `DnSQueue` when FFT is disabled.
4. `ThreadGPU.py::QueueOut()` runs `cudaFFT(...)` or `fft_cpu(...)`, then forwards processed payloads to `ThreadDnS.py`.
5. `ThreadDnS.py::QueueOut()` calls the matching processing function, emits display payloads through `UiBridge`, and saves when enabled.

### FiniteAline

1. `ThreadWeaver.py::QueueOut()` matches `FiniteAline`, waits for the processing barrier, calls `InitMemory()`, then calls `SingleScan(DnS_action=FiniteAline, acq_mode=FiniteAline)`.
2. `SingleScan()` waits again at start, drains stale continuous leftovers with `drain_continuous_backlog(...)`, configures camera and AODO, starts acquisition, and dispatches one acquisition block from `DatabackQueue`.
3. If FFT is enabled, GPU runs `cudaFFT(...)` or `fft_cpu(...)`; otherwise Weaver sends raw data directly to DnS. The outgoing payload carries the already-chosen `filename_bundle`.
4. `ThreadDnS.py::QueueOut()` calls `Process_aline(...)`, which prepares `self.Aline`, optionally calls `Save(...)`, and `_emit_display(kind="aline")` sends the display payload to the GUI thread.
5. `SingleScan()` closes AODO, finalizes naming state if needed, waits for end-of-scan processing to drain, returns to Weaver, and Weaver waits again before sending `GPUActions.CLEAR`.

### ContinuousAline

1. `ThreadWeaver.py::QueueOut()` matches `ContinuousAline`, waits for the processing barrier, calls `InitMemory()`, then enters `RptScan(DnS_action=ContinuousAline, acq_mode=ContinuousAline)`.
2. `RptScan()` waits again at start, configures camera and AODO once, then loops while `RunButton` is checked, consuming camera buffers and dispatching either raw or FFT work.
3. When FFT is enabled, continuous FFT requests are rate-limited so GPU is only fed when `GPUQueue.qsize() == 0`; otherwise raw payloads go directly to DnS.
4. `ThreadDnS.py::Process_aline(...)` updates the current A-line display and optional save each time a payload arrives.
5. On stop, `RptScan()` closes AODO, finalizes naming, drains stale continuous backlog, requests `DISPLAY_FFT_ACTIONS` and `DISPLAY_COUNTS`, waits for processing to drain, then returns to Weaver for final `CLEAR`.

### FiniteBline

1. `ThreadWeaver.py::QueueOut()` matches `FiniteBline`, waits for the processing barrier, calls `InitMemory()`, then calls `SingleScan(DnS_action=FiniteBline, acq_mode=FiniteBline)`.
2. `SingleScan()` configures hardware, waits for one camera block, builds the filename bundle in Weaver, and dispatches either raw B-line data or FFT work.
3. `ThreadGPU.py` performs pre-FFT averaging, background correction, interpolation/dispersion, FFT, optional realtime dynamic, and forwards the payload to DnS.
4. `ThreadDnS.py::Process_bline(...)` computes the displayed B-line, optional dynamic overlay when realtime dynamic exists, and `Save(...)` writes:
   - only `Bline-...` when dynamic is off or realtime dynamic is off
   - `Bline-...` plus `BlineDyn-...` only when realtime dynamic output exists.
5. The scan then follows the same end barrier and `GPUActions.CLEAR` path as other finite scans.

### ContinuousBline

1. `ThreadWeaver.py::QueueOut()` matches `ContinuousBline`, waits for the barrier, calls `InitMemory()`, then runs `RptScan(DnS_action=ContinuousBline, acq_mode=ContinuousBline)`.
2. `RptScan()` streams camera buffers continuously and dispatches either raw or FFT work using the same live loop as continuous A-line.
3. `ThreadDnS.py::Process_bline(...)` updates the XZ display continuously and saves when enabled.
4. Pause uses `AODOActionField('StopTask')` / `StartTask` around the pause loop without tearing down the whole run.
5. Stop closes AODO, waits for GPU and DnS to drain, emits the display-count actions, and then Weaver sends `GPUActions.CLEAR`.

### FiniteCscan

1. `ThreadWeaver.py::QueueOut()` matches `FiniteCscan`, waits for the barrier, calls `InitMemory()`, then calls `SingleScan(DnS_action=FiniteCscan, acq_mode=FiniteCscan)`.
2. `InitMemory()` chooses between:
   - one acquisition containing the whole repeated stack when dynamic is off, or
   - `NAcq = Ypixels` repeated acquisitions when dynamic is on.
3. `SingleScan()` dispatches each acquisition with `dynamic_bline_idx = iAcq` when the mode is dynamic C-scan.
4. `ThreadGPU.py` returns:
   - `data` only for non-realtime paths, or
   - `data` plus `dynamic` for realtime dynamic paths.
5. `ThreadDnS.py` chooses one of:
   - `Process_Cscan(...)` for non-dynamic C-scan
   - `Process_Cscan_Dynamic(...)` for dynamic mode with realtime calculation off
   - `Process_Cscan_RealtimeDynamic(...)` for dynamic mode with realtime calculation on.
6. Save behavior is:
   - non-dynamic: `Save(...)` writes one `Cscan-...` stack
   - dynamic + realtime off: `Save(...)` writes one repeated B-line stack per Y slice using `cscan_bline`
   - dynamic + realtime on: `SaveRealtimeCscanDynamicVolumes(...)` writes only `CscanDyn-...` and `CscanMean-...` at the final Y slice.

### ContinuousCscan

1. `ThreadWeaver.py::QueueOut()` matches `ContinuousCscan`, waits for the barrier, calls `InitMemory()`, then runs `RptScan(DnS_action=ContinuousCscan, acq_mode=ContinuousCscan)`.
2. `RptScan()` continuously dispatches one full C-scan acquisition block at a time while the run is active.
3. `ThreadDnS.py` routes the payload through `Process_Cscan(...)` or the dynamic variants depending on the current UI dynamic state and whether realtime dynamic output is present.
4. Display updates go through `_emit_display(kind="cscan")`, which sends `bline`, `aip`, and optional `dynb`/`dyn` to the GUI thread.
5. Stop waits for GPU and DnS to drain before cleanup and final count reporting.

### PlatePreScan

1. `ThreadWeaver.py::QueueOut()` calls `prepare_and_run_plate_prescan(...)`, which ensures the plate plan exists, then enters `PlatePreScan(acq_mode=PlatePreScan, context=...)`.
2. `PlatePreScan()` loops sample by sample; before each sample it waits on `wait_for_processing_barrier(...)`, updates overlays, and calls `AdjustZstage(sample_id)` so the user can focus with a temporary continuous B-line run.
3. After focus, it calls `iterate_FOVs(acq_mode=PlatePreScan)`. `iterate_FOVs()` applies per-sample Y geometry, sends `GPUActions.INIT_MOSAIC`, initializes memory, then scans each FOV by calling `SingleScan(DnS_action=DnSActions.PROCESS_MOSAIC, ...)`.
4. After every FOV, `iterate_FOVs()` waits for GPU and DnS to go idle before moving to the next FOV. After the whole sample, `PlatePreScan()` waits again before enabling `NextSample` / `RepeatSample`.
5. `ThreadDnS.py::Process_Mosaic(...)` routes each FOV through `Process_Cscan(...)`, `Process_Cscan_Dynamic(...)`, or `Process_Mosaic_RealtimeDynamic(...)`, then pastes `self.AIP` into `self.SampleMosaic`.
6. Re-scan after mosaic correction repeats the same `AdjustZstage()` -> `iterate_FOVs()` -> barrier flow. Session metadata and overlays are saved through `save_session_data(...)` at the end.

### PlateScan

1. `ThreadWeaver.py::QueueOut()` loads the saved Mosaic plan, refreshes `sampleSelector`, then enters `PlateScan(acq_mode=PlateScan, context=...)`.
2. `PlateScan()` loops through samples; before each sample it waits for the previous sample’s processing barrier, sets `CurrentSampleLocations`, updates the sample overlay, and calls `iterate_FOVs(acq_mode=PlateScan)`.
3. `iterate_FOVs()` performs the per-FOV loop exactly as in pre-scan: `INIT_MOSAIC`, `SingleScan(PROCESS_MOSAIC, ...)`, then a drain barrier after every FOV.
4. After the last FOV of the sample, `PlateScan()` waits again for the sample-level barrier before declaring the sample complete and moving on.
5. In DnS:
   - non-realtime mosaic paths save per-FOV outputs through `Save(...)`
   - realtime dynamic mosaic paths accumulate `MeanVolume` and `DynamicVolume` in `Process_Mosaic_RealtimeDynamic(...)` and save them with `SaveRealtimeMosaicDynamicVolumes(...)` when the last Y slice of the tile arrives.
6. After the full plate scan returns to Weaver, `QueueOut()` increments `timeReader` when saving is enabled.

### TimedPlateScan

1. `ThreadWeaver.py::QueueOut()` ensures the saved plate plan exists, then calls `TimedPlateScan(acq_mode=TimedPlateScan, context=...)`.
2. `TimedPlateScan()` runs `PlateScan(...)` once per time point, using `CuSlice` / `SliceTotal` / `Timer` to control the loop and the next deadline.
3. After each time point, if there is idle time before the next deadline, Weaver waits for the processing barrier, then calls `process_idle_dynamic_until_deadline(...)` for offline dynamic work.
4. When a session is complete and another one remains, Weaver increments `timeReader` and `CuSlice` through `set_time_reader_value(...)`.
5. The per-sample and per-FOV acquisition logic inside each timed session is the same as `PlateScan()`.

### WellScan

1. `ThreadWeaver.py::QueueOut()` verifies that an in-memory plate plan exists, then calls `WellScan(acq_mode=WellScan, context=...)`.
2. `WellScan()` resolves the selected sample from `sampleSelector`, waits for the processing barrier, sets `CurrentSampleLocations`, updates the overlay, then calls `iterate_FOVs(acq_mode=WellScan)`.
3. `iterate_FOVs()` again performs `INIT_MOSAIC`, one `SingleScan(PROCESS_MOSAIC, ...)` per FOV, and a drain barrier after each FOV.
4. After the sample finishes, `WellScan()` waits once more for GPU/DnS idle before returning completion.
5. DnS processing and save behavior are the same as `PlateScan()` for the same dynamic/realtime settings.

### LocationCameraLive

1. `ThreadWeaver.py::QueueOut()` calls `live()`.
2. This path uses the USB/sample-localization workflow and rendering helpers such as `display_sample_overlay(...)`, `render_usb_roi_overlay(...)`, and `render_mosaic_correction_overlay(...)`.
3. It does not use the OCT FFT pipeline in `ThreadGPU.py` or the OCT display/save path in `ThreadDnS.py`.

## Barrier And Cleanup Policy

1. `wait_for_processing_barrier(...)` checks:
   - `GPUQueue.qsize()`
   - `gpu_thread.active_tasks`
   - `DnSQueue.qsize()`
   - `dns_thread.active_tasks`
2. Weaver now uses this barrier:
   - before starting direct finite/continuous acquisitions
   - at the start of `SingleScan()`, `RptScan()`, `AdjustZstage()`, and `iterate_FOVs()`
   - after every mosaic FOV
   - at sample boundaries
   - before final acquisition completion / cleanup.
3. At command end, Weaver waits for the barrier, then sends `GPUActionField(GPUActions.CLEAR)`. GPU forwards `DnSActions.CLEAR` to DnS after earlier processed payloads, so cleanup happens after the in-flight results.

## Plate / Well Workflow

Typical workflow:

1. Use USB camera localization to draw ROI(s).
2. Generate sample centers and FOV locations.
3. Run `PlatePreScan` to quickly inspect each sample.
4. Optionally correct the FOV plan using the mosaic view.
5. Run `PlateScan`, `TimedPlateScan`, or `WellScan`.
6. If timed acquisition is enabled, use idle time between imaging sessions for offline dynamic processing.

For timed plate scan:

- each sample is saved under:
  - `sampleID-<n>/Time-<t>/`
- offline processing writes:
  - tile dynamic volumes
  - tile mean volumes
  - stitched sample-level volumes when multiple tiles exist

## Dynamic Processing

Dynamic processing currently supports:

- online dynamic overlays for B-line / C-scan display
- delayed dynamic processing for plate / well data
- frame normalization before dynamic calculation
- loaded-background subtraction by default
- optional first-frame background mode in GPU code
- GPU and CPU helper paths for dynamic calculation from saved stacks

## Processing Barrier

The project now uses an explicit processing barrier:

- queue emptiness alone is not enough
- GPU and DnS track active tasks
- Weaver waits for:
  - queue empty
  - worker active count = 0

This is used at:

- sample boundaries
- before timed offline processing starts

## Strict Data Model

Sample/FOV records are no longer loose dicts. The main structured records are:

- `SampleCenter(sample_id, x, y, z)`
- `FOVLocation(sample_id, x, y, z, y_length_mm, y_pixels)`

These are serialized/deserialized through `ScanSession.py`.

## Running

Install the environment from:

- `python311_env.yml`

Then run:

```bash
python OCT_MT.py
```

The full instrument requires the relevant camera SDKs, ART-DAQ driver/library, GPU/CuPy environment, and stage-control dependencies.

## Generated Files

Typical local/generated files:

- `log_files/`
- `ACTS2210.log`
- `__pycache__/`

Acquired data are saved under the directory selected in the GUI.

Typical saved content includes:

- B-line / C-scan TIFF stacks
- plate/well tile TIFF stacks
- dynamic TIFF outputs
- stitched sample-level dynamic/mean volumes
- `Mosaic/scan_metadata.pkl`
- `Mosaic/overlay_sources.pkl`
- USB ROI JSON / raw USB image for training data

## Development Notes

Current design rules:

- keep queue action names in `ActionTypes.py`
- keep hardware and processing constants in `HardwareSpecs.py`
- keep rendering-only code in `Display_rendering.py`
- keep session persistence in `ScanSession.py`
- keep offline dynamic processing in `DynamicPostprocessing.py`
- keep orchestration in `ThreadWeaver.py`
- keep display/save interpretation in `ThreadDnS.py`
- keep OCT spectral processing in `ThreadGPU.py`
- avoid direct GUI widget writes from worker threads
- use live UI state for continuous acquisitions where interactive updates are expected

This remains research instrument software, but the current structure is now substantially more explicit than the earlier mixed thread/UI layout.
