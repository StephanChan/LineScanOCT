# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:35:03 2024

@author: admin
"""
import numpy as np
import time
from time import process_time
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter

# define number of samples per FFT and total number of FFTs
nSamp = 1600
nX = 1100
nY = 100
# init raw data

data = np.random.rand(nY, nX, nSamp)

background = np.zeros([nX, nSamp])
# window = np.complex64(np.hanning(nSamp))
# define CUDA kernel that calculate the product of two arrays
# winfunc = cp.ElementwiseKernel(
#     'float32 x, complex64 y',
#     'complex64 z',
#     'z=x*y',
#     'winfunc')

# define interpolation kernel
interp_kernel = cp.RawKernel(r'''
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
        
fftAxis = 1
# from scipy.interpolate import interp1d

# # calculate and time NumPy FFT, i.e., performming FFT using CPU
# # t0 = time.time()
# data_cpu = data#*np.transpose(window)
# t1 = time.time()
# f = interp1d(x,data_cpu,axis = 1)
# data2 = f(xp)
# t2 = time.time()
# data_cpu     = np.fft.fft(data2, axis=fftAxis)
# t3 = time.time()
# # print('\n CPU windowing time is: ',t1-t0)
# print('\n CPU interpolation time is: ',t2-t1)
# print('CPU FFT time is: ',t3-t2,'\n')

################################################################ calculate and time GPU FFT
dispersion_path = "E:/IOCTData/disperison_compensation"
import os
# print(dispersion_path+'/dspPhase.bin')
if os.path.isfile(dispersion_path+'/dspPhase.bin'):
    intpX  = np.float32(np.fromfile(dispersion_path+'/intpX.bin', dtype=np.float32))
    intpXp  = np.float32(np.fromfile(dispersion_path+'/intpXp.bin', dtype=np.float32))
    indice = np.uint16(np.fromfile(dispersion_path+'/intpIndice.bin', dtype=np.uint16)).reshape([2,nSamp])
    dispersion = np.float32(np.fromfile(dispersion_path+'/dspPhase.bin', dtype=np.float32)).reshape([1, nSamp])
    dispersion = np.complex64(np.exp(-1j*dispersion))
##
# data = np.float32(data + noise)
t1 = time.time()

# t1 = process_time()
# transfer input data to Device
# mempool= cp.get_default_memory_pool()
shape = data.shape
data = data - np.tile(background,[shape[0],1,1])
data = data - uniform_filter1d(data, size=51, axis=2)

Alines =shape[0]*shape[1]
data=data.reshape([Alines, nSamp])


x_gpu  = cp.array(intpX)
xp_gpu  = cp.array(intpXp)
y_gpu  = cp.array(data)
indice1 = cp.array(indice[0,:])
indice2 = cp.array(indice[1,:])
yp_gpu = cp.zeros(data.shape, dtype = cp.float32)
dispersion = cp.array(dispersion)
# window_gpu = cp.array(window)
# calculate array product
t2 = time.time()
# tp1 = process_time()
# # data_gpu = winfunc(data_gpu, window_gpu)
# tp2 = process_time()


interp_kernel((8,8),(16,16), (Alines, nSamp, x_gpu, xp_gpu, y_gpu, indice1, indice2, yp_gpu))
t3 = time.time()
data_gpu  = cp.fft.fft(yp_gpu, axis=fftAxis)
# print(data_gpu[0:10,0:5])

t4 = time.time()

# calculate absolute value of the complex FFT results, and only save the first 200 elements
data_gpu = cp.absolute(data_gpu[:,0:200])
# data_o_gpu.astype(cp.float32)

# t4 = process_time()
# print('\n fft time is: ',(t4-t3)/1.0)
# cp.cuda.Device().synchronize()
# t5 = process_time()
# transfer data back from GPU to CPU
results = cp.asnumpy(data_gpu)
t5 = time.time()
# t6 = process_time()
# print('\n device to host time is: ',(t6-t5)/1.0)
# cache.clear()
print('data 2 GPU takes  ',round(t2-t1,3),' sec')
print('interpolation takes ',round(t3-t2,5),' sec')
# print('windowing takes ',round(tp2-tp1,5),' sec')
print('FFT takes ',round(t4-t3,5),' sec')
print('data 2 CPU takes  ',round(t5-t4,3),' sec')
print('\n GPU total time is: ',t5-t1)
# plt.figure()
# plt.plot(results[1][:])
# print(mempool.used_bytes())