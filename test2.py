import cupy as cp
from build import demo
from cupyx.profiler import benchmark
import numpy as np
import sys
# sys.path.append('/home/pdudenas/lib/pyRSoXS/src/')

# from cuda import einsum_kernel, einsum_kernel_1d


N = 1024*1024
array1 = cp.random.rand(3*3,dtype=cp.float32)
array2 = cp.random.rand(N*3*3,dtype=cp.float32)
array3 = cp.empty(N*3*3,dtype=cp.float32)

# print('rotation kernel call:')
# print(benchmark(demo.py_rotate,(N,array1.data.ptr,array2.data.ptr,array3.data.ptr),n_repeat=10))

# print('rotation kernel shared call:')
# print(benchmark(demo.py_rotate_shared,(N,array1.data.ptr,array2.data.ptr,array3.data.ptr),n_repeat=10))

demo.py_rotate(N, array1.data.ptr, array2.data.ptr, array3.data.ptr)

array1 = cp.reshape(array1, (3,3))
array2 = cp.reshape(array2, (N,3,3))
cupy_result = cp.einsum('...ij,...jk->...ik',array2,cp.einsum('...ij,...kj->...ik',array1,array2))
print(array3.shape, cupy_result.ravel().shape)
assert cp.allclose(array3, cupy_result.ravel()), 'cuda kernel and cupy results do not match'