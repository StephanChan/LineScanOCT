# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:50:25 2023

@author: admin
"""
from PyQt5.QtCore import  QThread
import time
global SIM
from scipy.ndimage import uniform_filter1d, gaussian_filter
try:
    import cupy
    SIM = False
except:
    SIM = True
import numpy as np
from Actions import DnSAction
import os
import traceback
from matplotlib import pyplot as plt

class GPUThread(QThread):
    def __init__(self):
        super().__init__()
        # TODO: write windowing and dispersion function
        self.exit_message = 'GPU processing thread exited.'
        self.FFT_actions = 0 # count how many FFT actions have taken place
        self.bg_sub = False
        self.background_gpu = None
        self.background_x_normalization = None
        self.background_x_normalization_gpu = None
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None
        self.release_gpu_memory_each_fft = False
        self.gpu_chunk_frames = 8
        self.gpu_overlap_transfer = True
        self.default_static_normalization_mean = 40000.0
        self.static_normalization_mean = self.default_static_normalization_mean
        self.static_normalization_eps = 1e-3
        self.background_x_normalization_eps = 1e-3
        self.background_x_normalization_root_order = 2.0
        self.dynamic_normalization_eps = 1e-3
        self.dynamic_use_first_frame_background = False
        self.dynamic_uniform_filter_size = 10
        self.dynamic_gaussian_smoothing = False
        self.print_dynamic_frame_mean = False
        self.print_dynamic_frame_mean_full = False
        self.dynamic_frame_mean_values = None
        self.dynamic_normalized_frame_mean_values = None
        self.dynamic_normalized_sample_values = None
        self.dynMagnification = 10
        # Profiling synchronizes each GPU stage, so overlap is temporarily bypassed.
        self.gpu_profile_timing = False
        self.yp_gpu_buffer = None
        self.yp_gpu_stream_buffers = [None, None]
        self.raw_gpu_buffer = None
        self.raw_gpu_stream_buffers = [None, None]
        self.float_gpu_buffer = None
        self.float_gpu_stream_buffers = [None, None]
        self.highpass_gpu_buffer = None
        self.highpass_gpu_stream_buffers = [None, None]
        self.dynamic_filter_gpu_buffer = None
        self.dynamic_var_gpu_buffer = None
        self.gpu_streams = None
        
    def defwin(self):
        if not (SIM or self.SIM):
            self.winfunc = cupy.ElementwiseKernel(
                'float32 x, complex64 y',
                'complex64 z',
                'z=x*y',
                'winfunc')
    
    def definterp(self):
        if not (SIM or self.SIM):
            # define interpolation kernel
            self.interp_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void interp1d(long long NAlines, long long NSamples, float* x, float* xp, float* y, unsigned short* indice1, unsigned short* indice2, float* yp){
                const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
                const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
                const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
                
                long long int i;
                for(i=threadID; i<NAlines * NSamples; i += numThreads){
                        int sampx = i%NSamples;
                        long long int sampy0 = i / NSamples * NSamples;
                        float xt = xp[sampx];
                        float x0 = x[indice1[sampx]];
                        float x1 = x[indice2[sampx]];
                        float y0 = y[sampy0+indice1[sampx]];
                        float y1 = y[sampy0+indice2[sampx]];
                        //if (i==0){
                        //        printf("sampx is %d, sampy0 is %llu\n",sampx, sampy0);
                         //       printf("indice1 is %hu, indice2 is %hu\n", indice1[sampx], indice2[sampx]);    
                        //        printf("xt: %.5f, x0: %.5f, x1: %.5f, y0: %.5f, y1: %.5f\n",xt,x0,x1,y0,y1);
                        //        }
                        yp[i] = (y0+(xt-x0)*(y1-y0)/(x1-x0+0.00001));
                        
                        }
                }
            ''','interp1d')
            self.highpass51_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void highpass51(const float* src, float* dst, int lines, int samples){
                const int radius = 25;
                const float inv_size = 1.0f / 51.0f;
                extern __shared__ float tile[];

                int line = blockIdx.y;
                int sample_start = blockIdx.x * blockDim.x;
                int tx = threadIdx.x;
                int sample = sample_start + tx;
                long long base = (long long)line * samples;

                if(line >= lines){
                    return;
                }

                int center_index = sample_start + tx;
                while(center_index < 0 || center_index >= samples){
                    if(center_index < 0){
                        center_index = -center_index - 1;
                    }
                    if(center_index >= samples){
                        center_index = 2 * samples - center_index - 1;
                    }
                }
                tile[tx + radius] = src[base + center_index];

                if(tx < radius){
                    int left_index = sample_start + tx - radius;
                    int right_index = sample_start + blockDim.x + tx;

                    while(left_index < 0 || left_index >= samples){
                        if(left_index < 0){
                            left_index = -left_index - 1;
                        }
                        if(left_index >= samples){
                            left_index = 2 * samples - left_index - 1;
                        }
                    }
                    while(right_index < 0 || right_index >= samples){
                        if(right_index < 0){
                            right_index = -right_index - 1;
                        }
                        if(right_index >= samples){
                            right_index = 2 * samples - right_index - 1;
                        }
                    }
                    tile[tx] = src[base + left_index];
                    tile[tx + blockDim.x + radius] = src[base + right_index];
                }

                __syncthreads();

                if(sample < samples){
                    float sum = 0.0f;
                    int tile_center = tx + radius;
                    for(int offset = -radius; offset <= radius; ++offset){
                        sum += tile[tile_center + offset];
                    }
                    dst[base + sample] = src[base + sample] - sum * inv_size;
                }
            }
            ''','highpass51')
            self.dynamic_uniform_axis0_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void uniform_axis0_nearest(const float* src, float* dst, int frames, int xpix, int zpix, int window){
                long long total = (long long)frames * xpix * zpix;
                long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                long long stride = (long long)blockDim.x * gridDim.x;
                int left = window / 2;
                int right = window - left - 1;

                for(long long idx = tid; idx < total; idx += stride){
                    int z = idx % zpix;
                    long long tmp = idx / zpix;
                    int x = tmp % xpix;
                    int t = tmp / xpix;
                    float sum = 0.0f;
                    for(int offset = -left; offset <= right; ++offset){
                        int tt = t + offset;
                        if(tt < 0){
                            tt = 0;
                        }
                        if(tt >= frames){
                            tt = frames - 1;
                        }
                        long long src_idx = ((long long)tt * xpix + x) * zpix + z;
                        sum += src[src_idx];
                    }
                    dst[idx] = sum / (float)window;
                }
            }
            ''','uniform_axis0_nearest')
            self.dynamic_variance_axis0_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void variance_axis0(const float* src, float* dst, int frames, int xpix, int zpix){
                long long total = (long long)xpix * zpix;
                long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                long long stride = (long long)blockDim.x * gridDim.x;

                for(long long idx = tid; idx < total; idx += stride){
                    int z = idx % zpix;
                    int x = idx / zpix;
                    float sum = 0.0f;
                    float sumsq = 0.0f;
                    for(int t = 0; t < frames; ++t){
                        long long src_idx = ((long long)t * xpix + x) * zpix + z;
                        float v = src[src_idx];
                        sum += v;
                        sumsq += v * v;
                    }
                    float mean = sum / (float)frames;
                    float var = sumsq / (float)frames - mean * mean;
                    dst[idx] = var > 0.0f ? var : 0.0f;
                }
            }
            ''','variance_axis0')

    def run(self):
        self.defwin()
        self.definterp()
        self.update_Dispersion()
        self.update_background()
        # self.update_FFTlength()
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != 'exit':
            start=time.time()
            try:
                if self.item.action == 'GPU':
                    self.cudaFFT(self.item.DnS_action, self.item.acq_mode, self.item.memory_slot, self.item.payload)
                    self.FFT_actions += 1
                elif self.item.action == 'CPU':
                    self.cpuFFT(self.item.DnS_action, self.item.acq_mode, self.item.memory_slot, self.item.payload)
                    self.FFT_actions += 1
                elif self.item.action == 'update_Dispersion':
                    self.update_Dispersion()
                elif self.item.action == 'update_background':
                    self.update_background()
                elif self.item.action == 'display_FFT_actions':
                    self.display_FFT_actions()
                    
                else:
                    an_action = DnSAction(
                        self.item.action,
                        acq_mode=self.item.acq_mode,
                        payload=self.item.payload,
                    )
                    self.DnSQueue.put(an_action)
                if time.time()-start > 1:
                    message = 'FFT processing took '+str(round(time.time()-start,3))+' s.'
                    print(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
            except Exception as error:
                message = "FFT processing failed. This frame was skipped."
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()
            # print('GPU queue size:', self.queue.qsize())
        self.emit_status(self.exit_message)

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def cudaFFT(self, DnS_action, acq_mode, memory_slot, payload):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()
        # get depth pixels after FFT
        Pixel_start = self.ui.DepthStart.value()
        Pixel_range = self.ui.DepthRange.value()
        shape = self.Memory[memory_slot].shape
        self.reset_dynamic_frame_mean_debug(DnS_action)
        # print('GPU data size: ', shape, ' memory_slot: ', memory_slot)
        # print('data shape', shape)
        # print('GPU receives:',self.data_CPU[0,0,0:10])
        if not (SIM or self.SIM):
            t_gpu_start = time.time()
            chunk_frames = self.gpu_fft_chunk_frames(DnS_action, shape[0])
            dynamic_reference_gpu = self.dynamic_reference_gpu_for_chunks(DnS_action, memory_slot, shape)
            self.data_CPU = np.empty((shape[0], shape[1], Pixel_range), dtype=np.float32)
            profile = self.new_gpu_profile() if self.gpu_profile_timing else None
            if self.gpu_overlap_transfer and not self.gpu_profile_timing and shape[0] > chunk_frames:
                self.cudaFFT_chunked_overlapped(
                    DnS_action,
                    memory_slot,
                    samples,
                    Pixel_start,
                    Pixel_range,
                    chunk_frames,
                    dynamic_reference_gpu,
                )
            else:
                for chunk_start in range(0, shape[0], chunk_frames):
                    chunk_end = min(shape[0], chunk_start + chunk_frames)
                    chunk = self.Memory[memory_slot][chunk_start:chunk_end, :, :]
                    self.data_CPU[chunk_start:chunk_end, :, :] = self.cudaFFT_chunk(
                        chunk,
                        DnS_action,
                        samples,
                        Pixel_start,
                        Pixel_range,
                        dynamic_reference_gpu,
                        profile,
                    )
            del dynamic_reference_gpu
            if self.release_gpu_memory_each_fft:
                self.release_gpu_memory()
            t5=time.time()
            if profile is not None:
                self.print_gpu_profile(profile)
            if round(t5-t_gpu_start,4) >0.8:
                print('time for chunked GPU FFT: ', round(t5-t_gpu_start,3),'sec')
            # print('data_CPU shape', self.data_CPU.shape)
            # print('data_CPU:', self.data_CPU[0,0,0:15])
            if self.ui.DynCheckBox.isChecked() and DnS_action in [ 'FiniteBline', 'FiniteCscan']:
                Dyn = self.Dynamic_Processing()
                # print('dyn:',Dyn[0,0:5])
                # Dyn = []
            else:
                Dyn = []
            self.print_dynamic_frame_mean_debug(DnS_action)
            t6=time.time()
            if round(t6-t5,4) >0.3:
                print('time for dynamic calculation: ', round(t6-t5,3),'sec')
            # display and save data, data type is float32
            an_action = DnSAction(DnS_action, acq_mode=acq_mode, data = self.data_CPU, raw = False, dynamic = Dyn, payload = payload) # data in Memory[memory_slot]
            self.DnSQueue.put(an_action)

            
            # print('send for display')
            if self.ui.DSing.isChecked():
                self.GPU2weaverQueue.put(self.data_CPU)
                # print('GPU data to weaver')
        
        else:
            if self.ui.ACQMode.currentText() in ['ContinuousBline', 'ContinuousAline','FiniteBline', 'FiniteAline']:
                self.AlineCount = self.ui.BlineAVG.value() * self.ui.AlinesPerBline.value()
            elif self.ui.ACQMode.currentText() in ['ContinuousCscan']:
                self.AlineCount = self.ui.AlinesPerBline.value() * self.ui.Ypixels.value() * self.ui.BlineAVG.value()
            elif self.ui.ACQMode.currentText() in ['FiniteCscan']:
                if self.ui.DynCheckBox.isChecked():
                    self.AlineCount = self.ui.BlineAVG.value() * self.ui.AlinesPerBline.value()
                else:
                    self.AlineCount = self.ui.AlinesPerBline.value() * self.ui.Ypixels.value() * self.ui.BlineAVG.value()
            data_CPU = 65535*np.random.random([self.AlineCount, Pixel_range]).reshape(shape[0],shape[1],Pixel_range)
            an_action = DnSAction(DnS_action, acq_mode=acq_mode, data = data_CPU, raw = False, payload = payload) # data in Memory[memory_slot]
            self.DnSQueue.put(an_action)
            # print('send for display')
            # print(self.ui.DSing.isChecked())
            if self.ui.DSing.isChecked():
                self.GPU2weaverQueue.put(data_CPU)
                # print('GPU data to weaver')
            # print('GPU finish')
 
            
    def cpuFFT(self, DnS_action, acq_mode, memory_slot, payload):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # get depth pixels after FFT
        Pixel_start = self.ui.DepthStart.value()
        Pixel_range = self.ui.DepthRange.value()

        t_cpu_start = time.time()
        self.data_CPU = self.Memory[memory_slot].astype(np.float32, copy=True)
        shape = self.data_CPU.shape
        self.reset_dynamic_frame_mean_debug(DnS_action)
        # print('data shape', shape)
        # print('GPU receives:',self.data_CPU[0,0,0:10])

        t0=time.time()
        self.subtract_background_after_conversion_cpu(DnS_action, shape, self.data_CPU)
        baseline = uniform_filter1d(self.data_CPU, size=51, axis=2)
        self.data_CPU -= baseline
        del baseline
        if round(time.time()-t0,4) >0.1:
            print('CPU background/high-pass took ', round(time.time()-t0,3),'sec')
        
        Alines =shape[0]*shape[1]
        self.data_CPU=self.data_CPU.reshape([Alines, samples])
        
        if self.interp:
            self.data_CPU = self.interpolate_cpu(self.data_CPU)
            self.data_CPU = np.fft.fft(self.data_CPU * self.dispersion, axis=1) / samples
        else:
            self.data_CPU = np.fft.fft(self.data_CPU, axis=1) / samples
        
        self.data_CPU = np.abs(self.data_CPU[:, Pixel_start: Pixel_start + Pixel_range]) * self.AMPLIFICATION
        self.data_CPU = np.float32(self.data_CPU)
        self.data_CPU = self.data_CPU.reshape(shape[0],shape[1],Pixel_range)
        self.apply_background_x_normalization_cpu(shape, self.data_CPU)
        # print('data_CPU:', self.data_CPU[0,0,0:5])
        if self.ui.DynCheckBox.isChecked() and DnS_action in [ 'FiniteBline', 'FiniteCscan']:
            Dyn = self.Dynamic_Processing_CPU()
        else:
            Dyn = []
        self.print_dynamic_frame_mean_debug(DnS_action)
        if round(time.time()-t_cpu_start,4) >0.5:
            print('time for CPU FFT: ', round(time.time()-t_cpu_start,3),'sec')
        # display and save data, data type is float32
        an_action = DnSAction(DnS_action, acq_mode=acq_mode, data = self.data_CPU, raw = False, dynamic = Dyn, payload = payload) # data in Memory[memory_slot]
        self.DnSQueue.put(an_action)
        
        if self.ui.DSing.isChecked():
            self.GPU2weaverQueue.put(self.data_CPU)
            # print('GPU data to weaver')

    def subtract_background_after_conversion_cpu(self, DnS_action, chunk_shape, data_cpu):
        if self.ui.DynCheckBox.isChecked() and DnS_action in [
            'ContinuousBline',
            'FiniteBline',
            'ContinuousCscan',
            'FiniteCscan',
            'Process_Mosaic',
            'PlatePreScan',
            'PlateScan',
            'WellScan',
        ]:
            bline_avg = int(self.ui.BlineAVG.value())
            if bline_avg >= 2 and len(chunk_shape) >= 3 and chunk_shape[0] >= 2:
                self.normalize_dynamic_frames_cpu(data_cpu)
                if not self.dynamic_use_first_frame_background:
                    self.subtract_static_background_from_normalized_cpu(chunk_shape, data_cpu)
                    return True
                if chunk_shape[0] == bline_avg:
                    dynamic_background = data_cpu[0:1, :, :].copy()
                    data_cpu -= dynamic_background
                    del dynamic_background
                    return True
                if DnS_action in ['ContinuousCscan', 'FiniteCscan'] and chunk_shape[0] % bline_avg == 0:
                    y_count = chunk_shape[0] // bline_avg
                    data_view = data_cpu.reshape([y_count, bline_avg, chunk_shape[1], chunk_shape[2]])
                    dynamic_background = data_view[:, 0:1, :, :].copy()
                    data_view -= dynamic_background
                    del dynamic_background
                    return True

        if self.bg_sub:
            if self.background.shape == chunk_shape[1:]:
                data_cpu -= self.background[np.newaxis, :, :]
                data_cpu /= np.float32(self.static_normalization_mean)
                return True
            print(
                'Background shape mismatch. Background subtraction skipped: ',
                self.background.shape,
                'data frame shape:',
                chunk_shape[1:],
            )
        data_cpu /= np.float32(self.static_normalization_mean)
        return False

    def normalize_dynamic_frames_cpu(self, data_cpu):
        frame_mean = np.mean(data_cpu, axis=(1, 2), keepdims=True)
        self.record_dynamic_frame_mean_debug(frame_mean.reshape(-1))
        data_cpu /= frame_mean + np.float32(self.dynamic_normalization_eps)
        self.record_dynamic_normalized_debug(
            np.mean(data_cpu, axis=(1, 2)),
            data_cpu.reshape(-1)[:10],
        )
        return data_cpu

    def subtract_static_background_from_normalized_cpu(self, chunk_shape, data_cpu):
        if not self.bg_sub:
            return False
        if self.background.shape != chunk_shape[1:]:
            print(
                'Background shape mismatch. Background subtraction skipped: ',
                self.background.shape,
                'data frame shape:',
                chunk_shape[1:],
            )
            return False
        data_cpu -= (self.background / np.float32(self.static_normalization_mean))[np.newaxis, :, :]
        return True

    def apply_background_x_normalization_cpu(self, chunk_shape, data_cpu):
        if self.background_x_normalization is None:
            return False
        if self.background_x_normalization.size != chunk_shape[1]:
            print(
                'Background X normalization mismatch. Skipped: ',
                self.background_x_normalization.size,
                'data X pixels:',
                chunk_shape[1],
            )
            return False
        data_cpu /= self.background_x_normalization[np.newaxis, :, np.newaxis]
        return True

    def interpolate_cpu(self, data_cpu):
        idx0 = self.indice[0, :].astype(np.intp, copy=False)
        idx1 = self.indice[1, :].astype(np.intp, copy=False)
        x0 = self.intpX[idx0]
        x1 = self.intpX[idx1]
        xt = self.intpXp
        y0 = data_cpu[:, idx0]
        y1 = data_cpu[:, idx1]
        return np.float32(y0 + (xt - x0) * (y1 - y0) / (x1 - x0 + 0.00001))

    def gpu_fft_chunk_frames(self, DnS_action, total_frames):
        chunk_frames = max(1, int(self.gpu_chunk_frames))
        if not self.ui.DynCheckBox.isChecked():
            return min(total_frames, chunk_frames)
        if not self.dynamic_use_first_frame_background:
            return min(total_frames, chunk_frames)

        bline_avg = max(1, int(self.ui.BlineAVG.value()))
        if bline_avg < 2:
            return min(total_frames, chunk_frames)

        if DnS_action in ['ContinuousCscan', 'FiniteCscan'] and total_frames > bline_avg and total_frames % bline_avg == 0:
            chunk_frames = max(chunk_frames, bline_avg)
            chunk_frames = (chunk_frames // bline_avg) * bline_avg
            return min(total_frames, max(bline_avg, chunk_frames))

        return min(total_frames, chunk_frames)

    def cudaFFT_chunk(
        self,
        raw_chunk,
        DnS_action,
        samples,
        Pixel_start,
        Pixel_range,
        dynamic_reference_gpu=None,
        profile=None,
    ):
        chunk_shape = raw_chunk.shape
        self.profile_start_chunk(profile)

        t0 = time.perf_counter()
        raw_gpu = self.load_raw_chunk_to_gpu(raw_chunk)
        self.profile_sync_add(profile, 'h2d_uint16', t0)

        t0 = time.perf_counter()
        y_gpu, _ = self.prepare_float_chunk(
            DnS_action,
            chunk_shape,
            raw_gpu,
            dynamic_reference_gpu,
        )
        self.profile_sync_add(profile, 'convert_bg', t0)

        t0 = time.perf_counter()
        y_gpu = self.apply_highpass_filter(y_gpu)
        self.profile_sync_add(profile, 'highpass', t0)

        t0 = time.perf_counter()
        alines = chunk_shape[0] * chunk_shape[1]
        y_gpu = y_gpu.reshape([alines, samples])
        self.profile_sync_add(profile, 'reshape', t0)

        t0 = time.perf_counter()
        if self.interp:
            self.ensure_dispersion_gpu_cache()
            yp_gpu = self.gpu_interpolation_buffer(y_gpu.shape)
            self.interp_kernel(
                (8, 8),
                (16, 16),
                (
                    alines,
                    samples,
                    self.intpX_gpu,
                    self.intpXp_gpu,
                    y_gpu,
                    self.indice1_gpu,
                    self.indice2_gpu,
                    yp_gpu,
                ),
            )
        else:
            yp_gpu = y_gpu
        self.profile_sync_add(profile, 'interpolation', t0)

        t0 = time.perf_counter()
        if self.interp:
            data_gpu = cupy.fft.fft(yp_gpu * self.dispersion_gpu, axis=1) / samples
        else:
            data_gpu = cupy.fft.fft(yp_gpu, axis=1) / samples
        self.profile_sync_add(profile, 'fft', t0)

        t0 = time.perf_counter()
        data_gpu = cupy.absolute(data_gpu[:, Pixel_start:Pixel_start + Pixel_range]) * self.AMPLIFICATION
        data_gpu = data_gpu.reshape(chunk_shape[0], chunk_shape[1], Pixel_range)
        self.apply_background_x_normalization_gpu(chunk_shape, data_gpu)
        self.profile_sync_add(profile, 'abs_crop', t0)

        t0 = time.perf_counter()
        data_cpu = cupy.asnumpy(data_gpu)
        self.profile_add(profile, 'd2h', time.perf_counter() - t0)

        del data_gpu, y_gpu
        if self.interp:
            del yp_gpu

        return data_cpu

    def load_raw_chunk_to_gpu(self, raw_chunk, slot=None, stream=None):
        raw_gpu = self.gpu_raw_buffer(raw_chunk.shape, raw_chunk.dtype, slot=slot)
        if stream is None:
            raw_gpu.set(raw_chunk)
        else:
            try:
                raw_gpu.set(raw_chunk, stream=stream)
            except TypeError:
                raw_gpu.set(raw_chunk)
        return raw_gpu

    def prepare_float_chunk(self, DnS_action, chunk_shape, raw_gpu, dynamic_reference_gpu=None, slot=None):
        y_gpu = self.gpu_float_buffer(raw_gpu.shape, slot=slot)
        y_gpu[...] = raw_gpu
        background_subtracted = self.subtract_background_after_conversion(
            DnS_action,
            chunk_shape,
            y_gpu,
            dynamic_reference_gpu,
        )
        return y_gpu, background_subtracted

    def subtract_background_after_conversion(self, DnS_action, chunk_shape, y_gpu, dynamic_reference_gpu=None):
        if self.ui.DynCheckBox.isChecked() and DnS_action in [
            'ContinuousBline',
            'FiniteBline',
            'ContinuousCscan',
            'FiniteCscan',
            'Process_Mosaic',
            'PlatePreScan',
            'PlateScan',
            'WellScan',
        ]:
            bline_avg = int(self.ui.BlineAVG.value())
            if bline_avg >= 2 and len(chunk_shape) >= 3 and chunk_shape[0] >= 2:
                self.normalize_dynamic_frames(y_gpu)
                if not self.dynamic_use_first_frame_background:
                    self.subtract_static_background_from_normalized_gpu(chunk_shape, y_gpu)
                    return True
                if dynamic_reference_gpu is not None:
                    y_gpu -= dynamic_reference_gpu
                    return True
                if chunk_shape[0] == bline_avg:
                    dynamic_background = y_gpu[0:1, :, :].copy()
                    y_gpu -= dynamic_background
                    del dynamic_background
                    return True
                if DnS_action in ['ContinuousCscan', 'FiniteCscan'] and chunk_shape[0] % bline_avg == 0:
                    y_count = chunk_shape[0] // bline_avg
                    data_view = y_gpu.reshape([y_count, bline_avg, chunk_shape[1], chunk_shape[2]])
                    dynamic_background = data_view[:, 0:1, :, :].copy()
                    data_view -= dynamic_background
                    del dynamic_background
                    return True

        if self.bg_sub:
            if self.background.shape == chunk_shape[1:]:
                if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
                    self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                y_gpu -= self.background_gpu[cupy.newaxis, :, :]
                y_gpu /= cupy.float32(self.static_normalization_mean)
                return True
            print(
                'Background shape mismatch. Background subtraction skipped: ',
                self.background.shape,
                'data frame shape:',
                chunk_shape[1:],
            )
        y_gpu /= cupy.float32(self.static_normalization_mean)
        return False

    def normalize_dynamic_frames(self, y_gpu):
        """
        Normalize each frame by its own mean before dynamic subtraction.

        Dynamic raw data are arranged as (frame, Y/X pixel, spectral pixel) in
        this processing path, so the per-frame light-source intensity estimate
        is the mean over the second and third dimensions. A small EPS is added
        to the denominator, matching Dynamic_Processing(), to avoid excessive
        gain when the reference intensity is very small.
        """
        frame_mean = cupy.mean(y_gpu, axis=(1, 2), keepdims=True)
        self.record_dynamic_frame_mean_debug(cupy.asnumpy(frame_mean.reshape(-1)))
        y_gpu /= frame_mean + cupy.float32(self.dynamic_normalization_eps)
        normalized_mean = cupy.mean(y_gpu, axis=(1, 2))
        normalized_sample = y_gpu.reshape(-1)[:10]
        self.record_dynamic_normalized_debug(
            cupy.asnumpy(normalized_mean),
            cupy.asnumpy(normalized_sample),
        )
        return y_gpu

    def subtract_static_background_from_normalized_gpu(self, chunk_shape, y_gpu):
        if not self.bg_sub:
            return False
        if self.background.shape != chunk_shape[1:]:
            print(
                'Background shape mismatch. Background subtraction skipped: ',
                self.background.shape,
                'data frame shape:',
                chunk_shape[1:],
            )
            return False
        if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
            self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
        y_gpu -= self.background_gpu[cupy.newaxis, :, :] / cupy.float32(self.static_normalization_mean)
        return True

    def apply_background_x_normalization_gpu(self, chunk_shape, y_gpu):
        if self.background_x_normalization is None:
            return False
        if self.background_x_normalization.size != chunk_shape[1]:
            print(
                'Background X normalization mismatch. Skipped: ',
                self.background_x_normalization.size,
                'data X pixels:',
                chunk_shape[1],
            )
            return False
        if (
            self.background_x_normalization_gpu is None
            or self.background_x_normalization_gpu.shape != self.background_x_normalization.shape
        ):
            self.background_x_normalization_gpu = cupy.asarray(
                self.background_x_normalization,
                dtype=cupy.float32,
            )
        y_gpu /= self.background_x_normalization_gpu[cupy.newaxis, :, cupy.newaxis]
        return True

    def reset_dynamic_frame_mean_debug(self, DnS_action):
        if self.print_dynamic_frame_mean and self.ui.DynCheckBox.isChecked() and DnS_action in ['FiniteBline', 'FiniteCscan']:
            self.dynamic_frame_mean_values = []
            self.dynamic_normalized_frame_mean_values = []
            self.dynamic_normalized_sample_values = []
        else:
            self.dynamic_frame_mean_values = None
            self.dynamic_normalized_frame_mean_values = None
            self.dynamic_normalized_sample_values = None

    def record_dynamic_frame_mean_debug(self, frame_mean):
        if self.dynamic_frame_mean_values is None:
            return
        values = np.asarray(frame_mean, dtype=np.float32).reshape(-1)
        if values.size > 0:
            self.dynamic_frame_mean_values.append(values)

    def record_dynamic_normalized_debug(self, normalized_mean, normalized_sample):
        if self.dynamic_normalized_frame_mean_values is None:
            return
        mean_values = np.asarray(normalized_mean, dtype=np.float32).reshape(-1)
        sample_values = np.asarray(normalized_sample, dtype=np.float32).reshape(-1)
        if mean_values.size > 0:
            self.dynamic_normalized_frame_mean_values.append(mean_values)
        if sample_values.size > 0:
            self.dynamic_normalized_sample_values.append(sample_values)

    def print_dynamic_frame_mean_debug(self, DnS_action):
        if self.dynamic_frame_mean_values is None or len(self.dynamic_frame_mean_values) == 0:
            return
        values = np.concatenate(self.dynamic_frame_mean_values)
        if values.size == 0:
            return
        preview_count = min(5, values.size)
        first_values = ', '.join(f'{v:.1f}' for v in values[:preview_count])
        last_values = ', '.join(f'{v:.1f}' for v in values[-preview_count:])
        print(
            f"Dynamic frame mean summary ({DnS_action}): "
            f"frames={values.size}, "
            f"min={float(np.min(values)):.1f}, "
            f"max={float(np.max(values)):.1f}, "
            f"mean={float(np.mean(values)):.1f}, "
            f"std={float(np.std(values)):.1f}, "
            f"first=[{first_values}], "
            f"last=[{last_values}]"
        )
        if self.print_dynamic_frame_mean_full:
            print(
                "Dynamic raw frame means: "
                + np.array2string(values, precision=1, separator=', ', threshold=values.size + 1)
            )
        if self.dynamic_normalized_frame_mean_values:
            normalized_means = np.concatenate(self.dynamic_normalized_frame_mean_values)
            normalized_samples = np.concatenate(self.dynamic_normalized_sample_values)
            preview_count = min(10, normalized_samples.size)
            sample_preview = ', '.join(f'{v:.4f}' for v in normalized_samples[:preview_count])
            print(
                f"Dynamic normalized summary ({DnS_action}): "
                f"frame_mean_min={float(np.min(normalized_means)):.4f}, "
                f"frame_mean_max={float(np.max(normalized_means)):.4f}, "
                f"frame_mean_avg={float(np.mean(normalized_means)):.4f}, "
                f"sample_values=[{sample_preview}]"
            )
            if self.print_dynamic_frame_mean_full:
                print(
                    "Dynamic normalized frame means: "
                    + np.array2string(
                        normalized_means,
                        precision=6,
                        separator=', ',
                        threshold=normalized_means.size + 1,
                    )
                )

    def new_gpu_profile(self):
        return {
            'chunks': 0,
            'h2d_uint16': 0.0,
            'convert_bg': 0.0,
            'highpass': 0.0,
            'reshape': 0.0,
            'interpolation': 0.0,
            'fft': 0.0,
            'abs_crop': 0.0,
            'd2h': 0.0,
        }

    def profile_start_chunk(self, profile):
        if profile is not None:
            profile['chunks'] += 1

    def profile_sync_add(self, profile, key, start_time):
        if profile is None:
            return
        cupy.cuda.get_current_stream().synchronize()
        profile[key] += time.perf_counter() - start_time

    def profile_add(self, profile, key, elapsed):
        if profile is not None:
            profile[key] += elapsed

    def print_gpu_profile(self, profile):
        total = sum(profile[key] for key in profile if key != 'chunks')
        if total <= 0:
            return
        print(
            'GPU profile: '
            f"chunks={profile['chunks']}, "
            f"h2d_uint16={profile['h2d_uint16']:.3f}s, "
            f"convert/bg={profile['convert_bg']:.3f}s, "
            f"highpass={profile['highpass']:.3f}s, "
            f"reshape={profile['reshape']:.3f}s, "
            f"interp={profile['interpolation']:.3f}s, "
            f"fft={profile['fft']:.3f}s, "
            f"abs/crop={profile['abs_crop']:.3f}s, "
            f"d2h={profile['d2h']:.3f}s, "
            f"sum={total:.3f}s"
        )

    def cudaFFT_chunked_overlapped(
        self,
        DnS_action,
        memory_slot,
        samples,
        Pixel_start,
        Pixel_range,
        chunk_frames,
        dynamic_reference_gpu,
    ):
        streams = self.gpu_overlap_streams()
        slot_refs = [None, None]
        for chunk_index, chunk_start in enumerate(range(0, self.Memory[memory_slot].shape[0], chunk_frames)):
            slot = chunk_index % 2
            if slot_refs[slot] is not None:
                slot_refs[slot]['stream'].synchronize()
                slot_refs[slot] = None

            chunk_end = min(self.Memory[memory_slot].shape[0], chunk_start + chunk_frames)
            chunk = self.Memory[memory_slot][chunk_start:chunk_end, :, :]
            host_out = self.data_CPU[chunk_start:chunk_end, :, :]
            slot_refs[slot] = self.cudaFFT_chunk_async(
                chunk,
                DnS_action,
                samples,
                Pixel_start,
                Pixel_range,
                dynamic_reference_gpu,
                streams[slot],
                slot,
                host_out,
            )

        for slot in range(2):
            if slot_refs[slot] is not None:
                slot_refs[slot]['stream'].synchronize()
                slot_refs[slot] = None

    def cudaFFT_chunk_async(
        self,
        raw_chunk,
        DnS_action,
        samples,
        Pixel_start,
        Pixel_range,
        dynamic_reference_gpu,
        stream,
        slot,
        host_out,
    ):
        chunk_shape = raw_chunk.shape
        keep_alive = []
        with stream:
            raw_gpu = self.load_raw_chunk_to_gpu(raw_chunk, slot=slot, stream=stream)
            keep_alive.append(raw_gpu)
            y_gpu, _ = self.prepare_float_chunk(
                DnS_action,
                chunk_shape,
                raw_gpu,
                dynamic_reference_gpu,
                slot=slot,
            )
            keep_alive.append(y_gpu)

            y_gpu = self.apply_highpass_filter(y_gpu, slot=slot)
            keep_alive.append(y_gpu)

            alines = chunk_shape[0] * chunk_shape[1]
            y_gpu = y_gpu.reshape([alines, samples])
            keep_alive.append(y_gpu)

            if self.interp:
                self.ensure_dispersion_gpu_cache()
                yp_gpu = self.gpu_interpolation_buffer(y_gpu.shape, slot=slot)
                self.interp_kernel(
                    (8, 8),
                    (16, 16),
                    (
                        alines,
                        samples,
                        self.intpX_gpu,
                        self.intpXp_gpu,
                        y_gpu,
                        self.indice1_gpu,
                        self.indice2_gpu,
                        yp_gpu,
                    ),
                )
            else:
                yp_gpu = y_gpu
            keep_alive.append(yp_gpu)

            if self.interp:
                data_gpu = cupy.fft.fft(yp_gpu * self.dispersion_gpu, axis=1) / samples
            else:
                data_gpu = cupy.fft.fft(yp_gpu, axis=1) / samples

            data_gpu = cupy.absolute(data_gpu[:, Pixel_start:Pixel_start + Pixel_range]) * self.AMPLIFICATION
            data_gpu = data_gpu.reshape(chunk_shape[0], chunk_shape[1], Pixel_range)
            self.apply_background_x_normalization_gpu(chunk_shape, data_gpu)
            keep_alive.append(data_gpu)
            self.copy_gpu_to_host_async(data_gpu, host_out, stream)

        return {'stream': stream, 'keep_alive': keep_alive}

    def copy_gpu_to_host_async(self, data_gpu, host_out, stream):
        try:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream, blocking=False)
        except TypeError:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream)

    def gpu_overlap_streams(self):
        if self.gpu_streams is None:
            self.gpu_streams = [cupy.cuda.Stream(non_blocking=True), cupy.cuda.Stream(non_blocking=True)]
        return self.gpu_streams

    def gpu_raw_buffer(self, shape, dtype, slot=None):
        dtype = np.dtype(dtype)
        if slot is None:
            if (
                self.raw_gpu_buffer is None
                or self.raw_gpu_buffer.shape != shape
                or self.raw_gpu_buffer.dtype != dtype
            ):
                self.raw_gpu_buffer = cupy.empty(shape, dtype=dtype)
            return self.raw_gpu_buffer

        if (
            self.raw_gpu_stream_buffers[slot] is None
            or self.raw_gpu_stream_buffers[slot].shape != shape
            or self.raw_gpu_stream_buffers[slot].dtype != dtype
        ):
            self.raw_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=dtype)
        return self.raw_gpu_stream_buffers[slot]

    def gpu_float_buffer(self, shape, slot=None):
        if slot is None:
            if self.float_gpu_buffer is None or self.float_gpu_buffer.shape != shape:
                self.float_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.float_gpu_buffer

        if self.float_gpu_stream_buffers[slot] is None or self.float_gpu_stream_buffers[slot].shape != shape:
            self.float_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.float_gpu_stream_buffers[slot]

    def apply_highpass_filter(self, data_gpu, slot=None):
        out_gpu = self.gpu_highpass_buffer(data_gpu.shape, slot=slot)
        threads = 256
        lines = int(data_gpu.shape[0] * data_gpu.shape[1])
        samples = int(data_gpu.shape[2])
        blocks_x = max(1, (samples + threads - 1) // threads)
        self.highpass51_kernel(
            (blocks_x, lines),
            (threads,),
            (
                data_gpu,
                out_gpu,
                lines,
                samples,
            ),
            shared_mem=(threads + 50) * 4,
        )
        return out_gpu

    def gpu_highpass_buffer(self, shape, slot=None):
        if slot is None:
            if self.highpass_gpu_buffer is None or self.highpass_gpu_buffer.shape != shape:
                self.highpass_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.highpass_gpu_buffer

        if self.highpass_gpu_stream_buffers[slot] is None or self.highpass_gpu_stream_buffers[slot].shape != shape:
            self.highpass_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.highpass_gpu_stream_buffers[slot]

    def gpu_interpolation_buffer(self, shape, slot=None):
        if slot is None:
            if self.yp_gpu_buffer is None or self.yp_gpu_buffer.shape != shape:
                self.yp_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.yp_gpu_buffer

        if self.yp_gpu_stream_buffers[slot] is None or self.yp_gpu_stream_buffers[slot].shape != shape:
            self.yp_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.yp_gpu_stream_buffers[slot]

    def dynamic_reference_gpu_for_chunks(self, DnS_action, memory_slot, shape):
        if not self.dynamic_use_first_frame_background:
            return None
        if not self.ui.DynCheckBox.isChecked():
            return None
        if DnS_action not in [
            'ContinuousBline',
            'FiniteBline',
            'ContinuousCscan',
            'FiniteCscan',
            'Process_Mosaic',
            'PlatePreScan',
            'PlateScan',
            'WellScan',
        ]:
            return None
        if len(shape) < 3 or shape[0] < 2:
            return None
        bline_avg = int(self.ui.BlineAVG.value())
        if bline_avg < 2:
            return None
        if DnS_action in ['ContinuousCscan', 'FiniteCscan'] and shape[0] != bline_avg:
            return None
        reference_gpu = cupy.asarray(self.Memory[memory_slot][0:1, :, :], dtype=cupy.float32)
        return self.normalize_dynamic_frames(reference_gpu)

    def ensure_dispersion_gpu_cache(self):
        if self.intpX_gpu is None or self.intpX_gpu.shape != self.intpX.shape:
            self.intpX_gpu = cupy.asarray(self.intpX, dtype=cupy.float32)
        if self.intpXp_gpu is None or self.intpXp_gpu.shape != self.intpXp.shape:
            self.intpXp_gpu = cupy.asarray(self.intpXp, dtype=cupy.float32)
        if self.indice1_gpu is None or self.indice1_gpu.shape != self.indice[0, :].shape:
            self.indice1_gpu = cupy.asarray(self.indice[0, :], dtype=cupy.uint16)
        if self.indice2_gpu is None or self.indice2_gpu.shape != self.indice[1, :].shape:
            self.indice2_gpu = cupy.asarray(self.indice[1, :], dtype=cupy.uint16)
        if self.dispersion_gpu is None or self.dispersion_gpu.shape != self.dispersion.shape:
            self.dispersion_gpu = cupy.asarray(self.dispersion, dtype=cupy.complex64)

    def clear_dispersion_gpu_cache(self):
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None

    def release_gpu_memory(self):
        cupy.get_default_memory_pool().free_all_blocks()
        pinned_pool = getattr(cupy, "get_default_pinned_memory_pool", None)
        if pinned_pool is not None:
            pinned_pool().free_all_blocks()

    def load_interpolation_indices(self, filename, samples):
        raw = np.fromfile(filename, dtype=np.uint16)
        expected = 2 * samples
        if raw.size != expected:
            raise ValueError(
                f"intpIndice.bin has {raw.size} values, expected {expected} "
                f"for {samples} samples."
            )

        max_valid = self.intpX.size - 1
        candidates = [
            ("2xsamples", raw.reshape([2, samples])),
            ("samplesx2", raw.reshape([samples, 2]).T),
        ]
        diagnostics = []
        for layout_name, candidate in candidates:
            min_index = int(candidate.min())
            max_index = int(candidate.max())
            diagnostics.append(f"{layout_name}: min={min_index}, max={max_index}")
            if min_index >= 0 and max_index <= max_valid:
                if layout_name != "2xsamples":
                    print(f"Interpolation index table loaded as {layout_name}.")
                return candidate

        raise ValueError(
            "Interpolation index table is out of range for intpX. "
            f"intpX length={self.intpX.size}, valid index=0-{max_valid}; "
            + "; ".join(diagnostics)
        )

            
    def update_Dispersion(self):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # print('GPU dispersion samples: ',samples)
            
        # self.window = np.float32(np.hanning(samples))
        # update dispersion and window
        dispersion_path = self.ui.InD_DIR.text()
        # print(dispersion_path+'/dspPhase.bin')
        if os.path.isfile(dispersion_path+'/dspPhase.bin'):
            try:
                self.interp = True
                self.intpX  = np.float32(np.fromfile(dispersion_path+'/intpX.bin', dtype=np.float32))
                self.intpXp  = np.float32(np.fromfile(dispersion_path+'/intpXp.bin', dtype=np.float32))
                if self.intpX.size != samples or self.intpXp.size != samples:
                    raise ValueError(
                        f"Interpolation arrays do not match NSamples_DH={samples}: "
                        f"len(intpX)={self.intpX.size}, len(intpXp)={self.intpXp.size}."
                    )
                self.indice = self.load_interpolation_indices(dispersion_path+'/intpIndice.bin', samples)
                dispersion_raw = np.fromfile(dispersion_path+'/dspPhase.bin', dtype=np.float32)
                if dispersion_raw.size != samples:
                    raise ValueError(
                        f"dspPhase.bin has {dispersion_raw.size} values, expected {samples}."
                    )
                self.dispersion = np.float32(dispersion_raw).reshape([1, samples])
                self.dispersion = np.complex64(np.exp(1j*self.dispersion))
                if not (SIM or self.SIM):
                    self.clear_dispersion_gpu_cache()
                    self.ensure_dispersion_gpu_cache()
                message = "Dispersion compensation loaded."
                self.emit_status(message)
                self.log.write(message)
                print(message)
            except:
                self.interp = False
                self.clear_dispersion_gpu_cache()
                # self.intpX  = np.float32(np.linspace(0,1,samples))
                # self.intpXp  = np.float32(np.linspace(0,1,samples))
                # self.indice = np.uint16(np.linspace(0,samples-1,samples)).reshape([samples,1])
                # self.indice = np.tile(self.indice,[1,2])
                # self.dispersion = np.complex64(np.ones(samples)).reshape([1,samples])
                message = "No dispersion compensation file found. Interpolation is disabled."
                self.emit_status(message)
                self.log.write(message)
                print(message)
        else:
            self.interp = False
            self.clear_dispersion_gpu_cache()
            # self.intpX  = np.float32(np.linspace(0,1,samples))
            # self.intpXp  = np.float32(np.linspace(0,1,samples))
            # self.indice = np.uint16(np.linspace(0,samples-1,samples)).reshape([samples,1])
            # self.indice = np.tile(self.indice,[1,2])
            # self.dispersion = np.complex64(np.ones(samples)).reshape([1,samples])
            message = "No dispersion compensation file found. Interpolation is disabled."
            self.emit_status(message)
            self.log.write(message)
            print(message)
        
    def update_background(self):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # print('GPU dispersion samples: ',samples)
        Xpixels = self.ui.AlinesPerBline.value()
        # self.window = np.float32(np.hanning(samples))
        # update dispersion and window
        background_path = self.ui.BG_DIR.text()
        # print(dispersion_path+'/dspPhase.bin')
        if os.path.isfile(background_path):
            try:
                self.background  = np.float32(np.fromfile(background_path, dtype=np.float32)).reshape([Xpixels,samples])
                background_mean = float(np.mean(self.background))
                if np.isfinite(background_mean) and background_mean > self.static_normalization_eps:
                    self.static_normalization_mean = background_mean
                else:
                    self.static_normalization_mean = self.default_static_normalization_mean
                self.update_background_x_normalization()
                if not (SIM or self.SIM):
                    self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                    if self.background_x_normalization is not None:
                        self.background_x_normalization_gpu = cupy.asarray(
                            self.background_x_normalization,
                            dtype=cupy.float32,
                        )
                # print(self.background.shape)
    
                # plt.figure()
                # plt.imshow(self.background)
                # plt.show()
                message = "Background file loaded."
                self.emit_status(message)
                self.log.write(message)
                print(message)
                self.bg_sub = True
            except:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                self.static_normalization_mean = self.default_static_normalization_mean
                message = "No valid background file found. Using zero background."
                self.emit_status(message)
                self.log.write(message)
                print(message)
        else:
            self.bg_sub = False
            self.background = np.zeros([Xpixels, samples])
            self.background_gpu = None
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            self.static_normalization_mean = self.default_static_normalization_mean
            message = "No background file selected. Using zero background."
            self.emit_status(message)
            self.log.write(message)
            print(message)
        self.background_tile = self.background

    def update_background_x_normalization(self):
        if self.background is None or self.background.size == 0:
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            return False

        x_profile = np.mean(self.background, axis=1, dtype=np.float32)
        profile_mean = float(np.mean(x_profile))
        if (
            not np.isfinite(profile_mean)
            or profile_mean <= self.background_x_normalization_eps
        ):
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            return False

        normalized = x_profile / np.float32(profile_mean)
        bad = (
            ~np.isfinite(normalized)
            | (np.abs(normalized) <= np.float32(self.background_x_normalization_eps))
        )
        if np.any(bad):
            normalized[bad] = np.float32(1.0)
        root_order = float(self.background_x_normalization_root_order)
        if np.isfinite(root_order) and root_order > 1.0:
            normalized = np.power(normalized, np.float32(1.0 / root_order), dtype=np.float32)
        self.background_x_normalization = np.asarray(normalized, dtype=np.float32)
        self.background_x_normalization_gpu = None
        return True
        

    def update_FFTlength(self):
        self.length_FFT = 2
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # print('GPU dispersion samples: ',samples)
        while self.length_FFT < samples:
            self.length_FFT *=2

    def display_FFT_actions(self):
        message = f"{self.FFT_actions} FFT request(s) processed."
        print(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        self.FFT_actions = 0
        
    def Dynamic_Processing(self, EPS=1e-3):
        if not (SIM or self.SIM):
            # only for Bline processing, Cscan processing need to do after all data are saved in disk
            dynamic_GPU = self.dynamic_signal_gpu(cupy.asarray(self.data_CPU, dtype=cupy.float32))
            dynamic = cupy.asnumpy(dynamic_GPU) * self.dynMagnification
            if self.dynamic_gaussian_smoothing:
                dynamic = gaussian_filter(dynamic, sigma=(1, 1))
            return dynamic
        else:
            data_CPU = self.data_CPU.copy()# (T,X,Z)
            data_CPU = uniform_filter1d(data_CPU, size=self.dynamic_uniform_filter_size, axis=0, mode='nearest')
            liv = np.var(data_CPU, axis=0, ddof=0)  # 1/N（与论文一致）
            if self.dynamic_gaussian_smoothing:
                liv = gaussian_filter(liv, sigma=(1, 1))
            return liv

    def Dynamic_Processing_CPU(self):
        data_cpu = uniform_filter1d(
            self.data_CPU,
            size=self.dynamic_uniform_filter_size,
            axis=0,
            mode='nearest',
        )
        dynamic = np.var(data_cpu, axis=0, ddof=0) * self.dynMagnification
        if self.dynamic_gaussian_smoothing:
            dynamic = gaussian_filter(dynamic, sigma=(1, 1))
        return np.float32(dynamic)

    def dynamic_signal_gpu(self, data_gpu):
        frames, xpix, zpix = data_gpu.shape
        filtered_gpu = self.dynamic_filter_buffer(data_gpu.shape)
        dynamic_gpu = self.dynamic_var_buffer((xpix, zpix))
        threads = 256

        total = int(frames * xpix * zpix)
        blocks = max(1, min(65535, (total + threads - 1) // threads))
        self.dynamic_uniform_axis0_kernel(
            (blocks,),
            (threads,),
            (
                data_gpu,
                filtered_gpu,
                int(frames),
                int(xpix),
                int(zpix),
                int(self.dynamic_uniform_filter_size),
            ),
        )

        total = int(xpix * zpix)
        blocks = max(1, min(65535, (total + threads - 1) // threads))
        self.dynamic_variance_axis0_kernel(
            (blocks,),
            (threads,),
            (
                filtered_gpu,
                dynamic_gpu,
                int(frames),
                int(xpix),
                int(zpix),
            ),
        )
        return dynamic_gpu

    def dynamic_filter_buffer(self, shape):
        if self.dynamic_filter_gpu_buffer is None or self.dynamic_filter_gpu_buffer.shape != shape:
            self.dynamic_filter_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
        return self.dynamic_filter_gpu_buffer

    def dynamic_var_buffer(self, shape):
        if self.dynamic_var_gpu_buffer is None or self.dynamic_var_gpu_buffer.shape != shape:
            self.dynamic_var_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
        return self.dynamic_var_gpu_buffer

        
   
