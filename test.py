import cupy as cp
from build import demo
from cupyx.profiler import benchmark
import numpy as np
import sys
sys.path.append('/home/pdudenas/lib/pyRSoXS/src/')

from cuda import einsum_kernel, einsum_kernel_1d


N = 1024*1024
array1 = cp.random.rand(N*3*3,dtype=cp.float32)
array2 = cp.random.rand(N*3*3,dtype=cp.float32)
array3 = cp.empty(N*3*3,dtype=cp.float32)

def cupy_einsum(array1_in, array2_in):
    cp.einsum('...ij,...jk->...ik', array1_in, array2_in, optimize='optimal')


threadsperblock = (32, 3, 3)
blockspergrid = (int(np.ceil(N/32)), 1, 1)

def cuda_einsum(array1_in, array2_in, array3_out):
    einsum_kernel[blockspergrid, threadsperblock](array1_in, array2_in, array3_out)

def cuda_einsum_1d(N, array1_in, array2_in, array3_out):
    einsum_kernel_1d[blockspergrid, threadsperblock](N, array1_in, array2_in, array3_out)

print('Pybind einsum kernel call:')
print(benchmark(demo.py_einsum,(N,array1.data.ptr,array2.data.ptr,array3.data.ptr),n_repeat=10))

print('Numba CUDA einsum 1D kernel call')
print(benchmark(cuda_einsum_1d, (N, array1, array2, array3), n_repeat=10))

array1 = cp.reshape(array1,(N,3,3))
array2 = cp.reshape(array2,(N,3,3))
array3 = cp.reshape(array3,(N,3,3))

print('Cupy einsum kernel call:')
print(benchmark(cupy_einsum, (array1, array2), n_repeat=10))

print('Numba CUDA einsum kernel call:')
print(benchmark(cuda_einsum, (array1, array2, array3), n_repeat=10))
# print(cp.reshape(array3[:27],(3,3,3)))
# print('\n')
# print(array3_cp[:3,:,:])
# print('\n')
# diff = array3-array3_cp.ravel()

# for i, val in enumerate(diff):
#     if val > 1e-6:
#         print(i, val)
