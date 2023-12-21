from numba import cuda, float32, int32, complex64, void


@cuda.jit(void(float32[:,:,:],float32[:,:,:],float32[:,:,:]))
def einsum_kernel(A,B,C):
    '''
    Cuda kernel for matrix multiplication of size [N, 3, 3]. 
    Performs the matrix contraction 'aij,ajk->aik'.
         
    Parameters
    ----------
    
    A : cuda or cupy array
        Input array 1
    B : cuda or cupy array
        Input array 2
    C : cuda or cupy array
        Output array
    
    '''
    x, y, z = cuda.grid(3)

    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    stride, _, _ = cuda.gridsize(3)    # blocks per grid
    for pos in range(x,A.shape[0],stride):
        C[pos, ty, tz] = A[pos, ty, 0] * B[pos, 0, tz] + A[pos, ty, 1] * B[pos, 1, tz] + A[pos, ty, 2] * B[pos, 2, tz]


@cuda.jit((int32, float32[:], float32[:], float32[:]))
def einsum_kernel_1d(n, A, B, C):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idy = cuda.threadIdx.y
    idz = cuda.threadIdx.z

    if ((idx % 9 == 0) & (idx < n)):
        C[idx + idy*3 + idz] = A[idx + idy*3 + 0] * B[idx + 0 + idz] + A[idx + idy*3 + 1] * B[idx + 1*3 + idz] + A[idx + idy*3 + 2] * B[idx + 2*3 + idz]

