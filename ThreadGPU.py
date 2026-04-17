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
try:
    from cupyx.scipy.ndimage import uniform_filter1d as cupy_uniform_filter1d
except:
    cupy_uniform_filter1d = None
import numpy as np
from Actions import DnSAction
import os
import traceback
from matplotlib import pyplot as plt

class GPUThread(QThread):
    def __init__(self):
        super().__init__()
        # TODO: write windowing and dispersion function
        self.exit_message = 'GPU thread successfully exited\n'
        self.FFT_actions = 0 # count how many FFT actions have taken place
        self.bg_sub = False
        self.background_gpu = None
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None
        self.release_gpu_memory_each_fft = False
        self.gpu_chunk_frames = 8
        self.gpu_overlap_transfer = True
        # Profiling synchronizes each GPU stage, so overlap is temporarily bypassed.
        self.gpu_profile_timing = False
        self.yp_gpu_buffer = None
        self.yp_gpu_stream_buffers = [None, None]
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
                    self.cudaFFT(self.item.mode, self.item.memoryLoc, self.item.args)
                    self.FFT_actions += 1
                elif self.item.action == 'CPU':
                    self.cpuFFT(self.item.mode, self.item.memoryLoc, self.item.args)
                    self.FFT_actions += 1
                elif self.item.action == 'update_Dispersion':
                    self.update_Dispersion()
                elif self.item.action == 'update_background':
                    self.update_background()
                elif self.item.action == 'display_FFT_actions':
                    self.display_FFT_actions()
                    
                else:
                    # self.ui.statusbar.showMessage('GPU thread is doing something invalid '+self.item.action)
                    an_action = DnSAction(self.item.action, args = self.item.args)
                    self.DnSQueue.put(an_action)
                if time.time()-start > 1:
                    message = '\n an FFT action took '+str(round(time.time()-start,3))+' s\n'
                    print(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
            except Exception as error:
                message = "An error occurred, skip the FFT action\n"
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit(message)
                    except Exception:
                        self.ui.statusbar.showMessage(message)
                else:
                    self.ui.statusbar.showMessage(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            self.item = self.queue.get()
            # print('GPU queue size:', self.queue.qsize())
        if getattr(self, "ui_bridge", None) is not None:
            try:
                self.ui_bridge.status_message.emit(self.exit_message)
            except Exception:
                self.ui.statusbar.showMessage(self.exit_message)
        else:
            self.ui.statusbar.showMessage(self.exit_message)

    def cudaFFT(self, mode, memoryLoc, args):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()
        # get depth pixels after FFT
        Pixel_start = self.ui.DepthStart.value()
        Pixel_range = self.ui.DepthRange.value()
        shape = self.Memory[memoryLoc].shape
        # print('GPU data size: ', shape, ' memoryLoc: ', memoryLoc)
        # print('data shape', shape)
        # print('GPU receives:',self.data_CPU[0,0,0:10])
        if not (SIM or self.SIM):
            t_gpu_start = time.time()
            chunk_frames = self.gpu_fft_chunk_frames(mode, shape[0])
            dynamic_reference_gpu = self.dynamic_reference_gpu_for_chunks(mode, memoryLoc, shape)
            self.data_CPU = np.empty((shape[0], shape[1], Pixel_range), dtype=np.float32)
            profile = self.new_gpu_profile() if self.gpu_profile_timing else None
            if self.gpu_overlap_transfer and not self.gpu_profile_timing and shape[0] > chunk_frames:
                self.cudaFFT_chunked_overlapped(
                    mode,
                    memoryLoc,
                    samples,
                    Pixel_start,
                    Pixel_range,
                    chunk_frames,
                    dynamic_reference_gpu,
                )
            else:
                for chunk_start in range(0, shape[0], chunk_frames):
                    chunk_end = min(shape[0], chunk_start + chunk_frames)
                    chunk = self.Memory[memoryLoc][chunk_start:chunk_end, :, :]
                    self.data_CPU[chunk_start:chunk_end, :, :] = self.cudaFFT_chunk(
                        chunk,
                        mode,
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
            if round(t5-t_gpu_start,4) >0.2:
                print('time for chunked GPU FFT: ', round(t5-t_gpu_start,3))
            # print('data_CPU shape', self.data_CPU.shape)
            # print('data_CPU:', self.data_CPU[0,0,0:15])
            if self.ui.DynCheckBox.isChecked() and mode in [ 'FiniteBline', 'FiniteCscan']:
                Dyn = self.Dynamic_Processing()
                # print('dyn:',Dyn[0,0:5])
                # Dyn = []
            else:
                Dyn = []
            t6=time.time()
            if round(t6-t5,4) >0.2:
                print('time for dynamic calculation: ', round(t6-t5,3))
            # display and save data, data type is float32
            an_action = DnSAction(mode, data = self.data_CPU, raw = False, dynamic = Dyn, args = args) # data in Memory[memoryLoc]
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
            an_action = DnSAction(mode, data = data_CPU, raw = False, args = args) # data in Memory[memoryLoc]
            self.DnSQueue.put(an_action)
            # print('send for display')
            # print(self.ui.DSing.isChecked())
            if self.ui.DSing.isChecked():
                self.GPU2weaverQueue.put(data_CPU)
                # print('GPU data to weaver')
            # print('GPU finish')
 
            
    def cpuFFT(self, mode, memoryLoc, args):
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # get depth pixels after FFT
        Pixel_start = self.ui.DepthStart.value()
        Pixel_range = self.ui.DepthRange.value()

        self.data_CPU = self.Memory[memoryLoc].astype(np.float32, copy=True)
        shape = self.data_CPU.shape
        # print('data shape', shape)
        # print('GPU receives:',self.data_CPU[0,0,0:10])
        # subtract background
        t0=time.time()
        self.subtract_dynamic_reference_background(mode, shape)
        # self.data_CPU -= self.background[np.newaxis, :, :]
        baseline = uniform_filter1d(self.data_CPU, size=51, axis=2)
        self.data_CPU -= baseline
        del baseline
        if round(time.time()-t0,4) >1:
            print('background subtraction took ', round(time.time()-t0,3),'s')
        
        Alines =shape[0]*shape[1]
        self.data_CPU=self.data_CPU.reshape([Alines, samples])
        
        fftAxis = 1
        # # zero-padding data before FFT
        # if data_CPU.shape[1] != self.length_FFT:
        #     data_padded = np.zeros([Alines, self.length_FFT], dtype = np.float32)
        #     tmp = np.uint16((self.length_FFT-samples)/2)
        #     data_padded[:,tmp:samples+tmp] = data_CPU
        # else:
        #     data_padded = data_CPU
        
        # self.data_CPU = self.data_CPU*self.dispersion
        # print(self.data_CPU[0:10,0:5])
        self.data_CPU = np.abs(np.fft.fft(self.data_CPU, axis=fftAxis))/samples
        # print(self.data_CPU[0:10,0:5])
        
        self.data_CPU = self.data_CPU[:,Pixel_start: Pixel_start+Pixel_range ]*self.AMPLIFICATION
        # data_CPU = data_CPU.reshape([shape[0],Pixel_range * np.uint32(Alines/shape[0])])
        self.data_CPU = self.data_CPU.reshape(shape[0],shape[1],Pixel_range)
        # print('data_CPU:', self.data_CPU[0,0,0:5])
        if self.ui.DynCheckBox.isChecked() and mode in [ 'FiniteBline', 'FiniteCscan']:
            Dyn = self.Dynamic_Processing()
            print('dyn:',Dyn[0,0:5])
        else:
            Dyn = []
        # display and save data, data type is float32
        an_action = DnSAction(mode, data = self.data_CPU, raw = False, dynamic = Dyn, args = args) # data in Memory[memoryLoc]
        self.DnSQueue.put(an_action)
        
        if self.ui.DSing.isChecked():
            self.GPU2weaverQueue.put(self.data_CPU)
            # print('GPU data to weaver')
        
    def subtract_dynamic_reference_background(self, mode, shape):
        """
        For dynamic acquisitions, use the first repeated frame in the burst as
        the local DC/background reference. This tracks slow DC drift better than
        the pre-measured background when BlineAVG frames are acquired close in time.
        """
        if not self.ui.DynCheckBox.isChecked():
            return False
        if mode not in [
            'ContinuousBline',
            'FiniteBline',
            'ContinuousCscan',
            'FiniteCscan',
            'Process_Mosaic',
            'PlatePreScan',
            'PlateScan',
            'WellScan',
        ]:
            return False
        if len(shape) < 3 or shape[0] < 2:
            return False
        bline_avg = int(self.ui.BlineAVG.value())
        if bline_avg < 2:
            return False

        if shape[0] == bline_avg:
            dynamic_background = self.data_CPU[0:1, :, :].copy()
            self.data_CPU -= dynamic_background
            del dynamic_background
            return True

        if mode in ['ContinuousCscan', 'FiniteCscan'] and shape[0] % bline_avg == 0:
            y_count = shape[0] // bline_avg
            data_view = self.data_CPU.reshape([y_count, bline_avg, shape[1], shape[2]])
            dynamic_background = data_view[:, 0:1, :, :].copy()
            data_view -= dynamic_background
            del dynamic_background
            return True

        return False

    def gpu_fft_chunk_frames(self, mode, total_frames):
        chunk_frames = max(1, int(self.gpu_chunk_frames))
        if not self.ui.DynCheckBox.isChecked():
            return min(total_frames, chunk_frames)

        bline_avg = max(1, int(self.ui.BlineAVG.value()))
        if bline_avg < 2:
            return min(total_frames, chunk_frames)

        if mode in ['ContinuousCscan', 'FiniteCscan'] and total_frames > bline_avg and total_frames % bline_avg == 0:
            chunk_frames = max(chunk_frames, bline_avg)
            chunk_frames = (chunk_frames // bline_avg) * bline_avg
            return min(total_frames, max(bline_avg, chunk_frames))

        return min(total_frames, chunk_frames)

    def cudaFFT_chunk(
        self,
        raw_chunk,
        mode,
        samples,
        Pixel_start,
        Pixel_range,
        dynamic_reference_gpu=None,
        profile=None,
    ):
        chunk_shape = raw_chunk.shape
        self.profile_start_chunk(profile)

        t0 = time.perf_counter()
        y_gpu = cupy.asarray(raw_chunk, dtype=cupy.float32)
        self.profile_sync_add(profile, 'h2d_convert', t0)

        t0 = time.perf_counter()
        dynamic_reference_subtracted = self.subtract_dynamic_reference_background_gpu(
            mode,
            chunk_shape,
            y_gpu,
            dynamic_reference_gpu,
        )
        if self.bg_sub and not dynamic_reference_subtracted:
            if self.background.shape == chunk_shape[1:]:
                if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
                    self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                y_gpu -= self.background_gpu[cupy.newaxis, :, :]
            else:
                print(
                    'background shape mismatch, skip subtraction: ',
                    self.background.shape,
                    'data frame shape:',
                    chunk_shape[1:],
                )
        self.profile_sync_add(profile, 'background', t0)

        t0 = time.perf_counter()
        baseline = self.gpu_uniform_filter1d(y_gpu, size=51, axis=2)
        y_gpu -= baseline
        del baseline
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
        self.profile_sync_add(profile, 'abs_crop', t0)

        t0 = time.perf_counter()
        data_cpu = cupy.asnumpy(data_gpu).reshape(chunk_shape[0], chunk_shape[1], Pixel_range)
        self.profile_add(profile, 'd2h', time.perf_counter() - t0)

        del data_gpu, y_gpu
        if self.interp:
            del yp_gpu

        return data_cpu

    def new_gpu_profile(self):
        return {
            'chunks': 0,
            'h2d_convert': 0.0,
            'background': 0.0,
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
            f"h2d+convert={profile['h2d_convert']:.3f}s, "
            f"background={profile['background']:.3f}s, "
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
        mode,
        memoryLoc,
        samples,
        Pixel_start,
        Pixel_range,
        chunk_frames,
        dynamic_reference_gpu,
    ):
        streams = self.gpu_overlap_streams()
        slot_refs = [None, None]
        for chunk_index, chunk_start in enumerate(range(0, self.Memory[memoryLoc].shape[0], chunk_frames)):
            slot = chunk_index % 2
            if slot_refs[slot] is not None:
                slot_refs[slot]['stream'].synchronize()
                slot_refs[slot] = None

            chunk_end = min(self.Memory[memoryLoc].shape[0], chunk_start + chunk_frames)
            chunk = self.Memory[memoryLoc][chunk_start:chunk_end, :, :]
            host_out = self.data_CPU[chunk_start:chunk_end, :, :]
            slot_refs[slot] = self.cudaFFT_chunk_async(
                chunk,
                mode,
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
        mode,
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
            y_gpu = cupy.asarray(raw_chunk, dtype=cupy.float32)
            keep_alive.append(y_gpu)

            dynamic_reference_subtracted = self.subtract_dynamic_reference_background_gpu(
                mode,
                chunk_shape,
                y_gpu,
                dynamic_reference_gpu,
                keep_alive,
            )
            if self.bg_sub and not dynamic_reference_subtracted:
                if self.background.shape == chunk_shape[1:]:
                    if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
                        self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                    y_gpu -= self.background_gpu[cupy.newaxis, :, :]
                else:
                    print(
                        'background shape mismatch, skip subtraction: ',
                        self.background.shape,
                        'data frame shape:',
                        chunk_shape[1:],
                    )

            baseline = self.gpu_uniform_filter1d(y_gpu, size=51, axis=2)
            keep_alive.append(baseline)
            y_gpu -= baseline

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

    def gpu_interpolation_buffer(self, shape, slot=None):
        if slot is None:
            if self.yp_gpu_buffer is None or self.yp_gpu_buffer.shape != shape:
                self.yp_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.yp_gpu_buffer

        if self.yp_gpu_stream_buffers[slot] is None or self.yp_gpu_stream_buffers[slot].shape != shape:
            self.yp_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.yp_gpu_stream_buffers[slot]

    def dynamic_reference_gpu_for_chunks(self, mode, memoryLoc, shape):
        if not self.ui.DynCheckBox.isChecked():
            return None
        if mode not in [
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
        if mode in ['ContinuousCscan', 'FiniteCscan'] and shape[0] != bline_avg:
            return None
        return cupy.asarray(self.Memory[memoryLoc][0:1, :, :], dtype=cupy.float32)

    def subtract_dynamic_reference_background_gpu(
        self,
        mode,
        shape,
        data_gpu,
        dynamic_reference_gpu=None,
        keep_alive=None,
    ):
        if not self.ui.DynCheckBox.isChecked():
            return False
        if mode not in [
            'ContinuousBline',
            'FiniteBline',
            'ContinuousCscan',
            'FiniteCscan',
            'Process_Mosaic',
            'PlatePreScan',
            'PlateScan',
            'WellScan',
        ]:
            return False
        if len(shape) < 3 or shape[0] < 2:
            return False
        bline_avg = int(self.ui.BlineAVG.value())
        if bline_avg < 2:
            return False

        if dynamic_reference_gpu is not None:
            data_gpu -= dynamic_reference_gpu
            return True

        if shape[0] == bline_avg:
            dynamic_background = data_gpu[0:1, :, :].copy()
            if keep_alive is not None:
                keep_alive.append(dynamic_background)
            data_gpu -= dynamic_background
            return True

        if mode in ['ContinuousCscan', 'FiniteCscan'] and shape[0] % bline_avg == 0:
            y_count = shape[0] // bline_avg
            data_view = data_gpu.reshape([y_count, bline_avg, shape[1], shape[2]])
            dynamic_background = data_view[:, 0:1, :, :].copy()
            if keep_alive is not None:
                keep_alive.append(dynamic_background)
            data_view -= dynamic_background
            return True

        return False

    def gpu_uniform_filter1d(self, data_gpu, size, axis):
        if cupy_uniform_filter1d is not None:
            return cupy_uniform_filter1d(data_gpu, size=size, axis=axis)
        return cupy.mean(data_gpu, axis=axis, keepdims=True)

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
                self.indice = np.uint16(np.fromfile(dispersion_path+'/intpIndice.bin', dtype=np.uint16)).reshape([2,samples])
                self.dispersion = np.float32(np.fromfile(dispersion_path+'/dspPhase.bin', dtype=np.float32)).reshape([1, samples])
                self.dispersion = np.complex64(np.exp(1j*self.dispersion))
                if not (SIM or self.SIM):
                    self.clear_dispersion_gpu_cache()
                    self.ensure_dispersion_gpu_cache()
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit("load disperison compensation success...")
                    except Exception:
                        self.ui.statusbar.showMessage("load disperison compensation success...")
                else:
                    self.ui.statusbar.showMessage("load disperison compensation success...")
                # self.ui.PrintOut.append("load disperison compensation success...")
                self.log.write("load disperison compensation success...")
                print("load disperison compensation success...")
            except:
                self.interp = False
                self.clear_dispersion_gpu_cache()
                # self.intpX  = np.float32(np.linspace(0,1,samples))
                # self.intpXp  = np.float32(np.linspace(0,1,samples))
                # self.indice = np.uint16(np.linspace(0,samples-1,samples)).reshape([samples,1])
                # self.indice = np.tile(self.indice,[1,2])
                # self.dispersion = np.complex64(np.ones(samples)).reshape([1,samples])
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit('no disperison compensation found...skip interpolation')
                    except Exception:
                        self.ui.statusbar.showMessage('no disperison compensation found...skip interpolation')
                else:
                    self.ui.statusbar.showMessage('no disperison compensation found...skip interpolation')
                # self.ui.PrintOut.append("no disperison compensation found...")
                self.log.write("no disperison compensation found...skip interpolation")
                print("no disperison compensation found...skip interpolation")
        else:
            self.interp = False
            self.clear_dispersion_gpu_cache()
            # self.intpX  = np.float32(np.linspace(0,1,samples))
            # self.intpXp  = np.float32(np.linspace(0,1,samples))
            # self.indice = np.uint16(np.linspace(0,samples-1,samples)).reshape([samples,1])
            # self.indice = np.tile(self.indice,[1,2])
            # self.dispersion = np.complex64(np.ones(samples)).reshape([1,samples])
            if getattr(self, "ui_bridge", None) is not None:
                try:
                    self.ui_bridge.status_message.emit('no disperison compensation found...skip interpolation')
                except Exception:
                    self.ui.statusbar.showMessage('no disperison compensation found...skip interpolation')
            else:
                self.ui.statusbar.showMessage('no disperison compensation found...skip interpolation')
            # self.ui.PrintOut.append("no disperison compensation found...")
            self.log.write("no disperison compensation found...skip interpolation")
            print("no disperison compensation found...skip interpolation")
        
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
                if not (SIM or self.SIM):
                    self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                # print(self.background.shape)
    
                # plt.figure()
                # plt.imshow(self.background)
                # plt.show()
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit("load background success...")
                    except Exception:
                        self.ui.statusbar.showMessage("load background success...")
                else:
                    self.ui.statusbar.showMessage("load background success...")
                # self.ui.PrintOut.append("load disperison compensation success...")
                self.log.write("load background success...")
                print("load background success...")
                self.bg_sub = True
            except:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                if getattr(self, "ui_bridge", None) is not None:
                    try:
                        self.ui_bridge.status_message.emit('no background found...using default')
                    except Exception:
                        self.ui.statusbar.showMessage('no background found...using default')
                else:
                    self.ui.statusbar.showMessage('no background found...using default')
                # self.ui.PrintOut.append("no disperison compensation found...")
                self.log.write("no background found...using default")
                print("no background found...using default")
        else:
            self.bg_sub = False
            self.background = np.zeros([Xpixels, samples])
            self.background_gpu = None
            if getattr(self, "ui_bridge", None) is not None:
                try:
                    self.ui_bridge.status_message.emit('no background found...using default')
                except Exception:
                    self.ui.statusbar.showMessage('no background found...using default')
            else:
                self.ui.statusbar.showMessage('no background found...using default')
            # self.ui.PrintOut.append("no disperison compensation found...")
            self.log.write("no background found...using default")
            print("no background found...using default2")
        self.background_tile = self.background
        

    def update_FFTlength(self):
        self.length_FFT = 2
        # get samples per Aline
        samples = self.ui.NSamples_DH.value()# - self.ui.DelaySamples.value()
        # print('GPU dispersion samples: ',samples)
        while self.length_FFT < samples:
            self.length_FFT *=2

    def display_FFT_actions(self):
        message = str(self.FFT_actions)+ ' FFT actions taken place\n'
        print(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        self.FFT_actions = 0
        
    def Dynamic_Processing(self, EPS=1e-3):
        if not (SIM or self.SIM):
            # only for Bline processing, Cscan processing need to do after all data are saved in disk
            data_GPU = cupy.array(self.data_CPU) # (T,X,Z)
            # data_GPU = cupy.moveaxis(data_GPU, 0, -1)  # (Z,X,T)
            time_mean = cupy.mean(data_GPU, axis=(1, 2), keepdims=True)  # (T,1,1)
            data_GPU = data_GPU / (time_mean + EPS)
            data_GPU = uniform_filter1d(data_GPU.get(), size=10, axis=0, mode='nearest')
        
            liv = cupy.var(data_GPU, axis=0, ddof=0)  # 1/N（与论文一致）
            dynamic_GPU = gaussian_filter(liv, sigma=(1, 1))
            dynamic = cupy.asnumpy(dynamic_GPU)*1000
            # print(dynamic[0,0:5])
            return dynamic
        else:
            data_CPU = self.data_CPU.copy()# (T,X,Z)
            time_mean = np.mean(data_CPU, axis=(1, 2), keepdims=True)  # (T,1,1)
            data_CPU = data_CPU / (time_mean + EPS)
            data_CPU = uniform_filter1d(data_CPU, size=10, axis=0, mode='nearest')
            print(data_CPU.shape)
            liv = np.var(data_CPU, axis=0, ddof=0)  # 1/N（与论文一致）
            dynamic = gaussian_filter(liv, sigma=(1, 1))
            # print(dynamic[0,0:5])
            return dynamic

        
   
