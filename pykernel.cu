#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>
// #include <helper_cuda.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel function to perform the matrix contraction '...ij,...jk->...ik'
__global__
void einsum_kernel(int n, float *A, float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockIdx.x * blockDim.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;

    if ((idx % 9 == 0) && (idx < n)){
        C[idx + idy*3 + idz] = A[idx + idy*3 + 0] * B[idx + 0 + idz] + A[idx + idy*3 + 1] * B[idx + 1*3 + idz] + A[idx + idy*3 + 2] * B[idx + 2*3 + idz];
        
    }


}

// A is 3x3 optical tensor, B is Z*Y*X*3*3 set of rotation matrices
__global__
void rotation_kernel(int n, float *A, float*B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;

    // __shared__ float tmp_res[9];

    if ((idx % 9 == 0) && (idx < n)){
        C[idx + idy*3 + idz] = A[idy*3 + 0] * B[idx + idz*3 + 0] + A[idy*3 + 1] * B[idx + idz*3 + 1] + A[idy*3 + 2] * B[idx + idz*3 + 2];

        // __syncthreads();
        // tmp_res[idy*3 + idz]
        // C[idx + idy*3 + idz] = B[idx + idy*3 + 0]*tmp_res[0 + idz] + B[idx + idy*3 + 1] * tmp_res[1*3 + idz] + B[idx + idy*3 + 2] * tmp_res[2*3 + idz];
    }
}

// A is 3x3 optical tensor, B is Z*Y*X*3*3 set of rotation matrices
__global__
void rotation_kernel_shared(int n, float *A, float*B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;

    __shared__ float shared_B[9];
    __shared__ float tmp_res[9];

    if ((idx % 9 == 0) && (idx < n)){
        // copy B into shared memory
        shared_B[idy*3 + idz] = B[idx + idy*3 + idz];
        tmp_res[idy*3 + idz] = A[idy*3 + 0] * shared_B[idz*3 + 0] + A[idy*3 + 1] * shared_B[idz*3 + 1] + A[idy*3 + 2] * shared_B[idz*3 + 2];
        C[idx + idy*3 + idz] = shared_B[idy*3 + 0]*tmp_res[0 + idz] + shared_B[idy*3 + 1] * tmp_res[1*3 + idz] + shared_B[idy*3 + 2] * tmp_res[2*3 + idz];
    }
}

void py_einsum(int N, size_t pA, size_t pB, size_t pC)
{

    float *A = reinterpret_cast<float*> (pA);
    float *B = reinterpret_cast<float*> (pB);
    float *C = reinterpret_cast<float*> (pC);

    dim3 threadsPerBlock(32,3,3);
    dim3 numBlocks((N + 32 - 1) / 32, 1, 1);
    einsum_kernel<<<numBlocks, threadsPerBlock>>>(N, A, B, C);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

void py_rotate(int N, size_t pA, size_t pB, size_t pC)
{
    float *A = reinterpret_cast<float*> (pA);
    float *B = reinterpret_cast<float*> (pB);
    float *C = reinterpret_cast<float*> (pC);

    dim3 threadsPerBlock(32,3,3);
    dim3 numBlocks((N + 32 - 1) / 32, 1, 1);
    rotation_kernel<<<numBlocks, threadsPerBlock>>>(N, A, B, C);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void py_rotate_shared(int N, size_t pA, size_t pB, size_t pC)
{
    float *A = reinterpret_cast<float*> (pA);
    float *B = reinterpret_cast<float*> (pB);
    float *C = reinterpret_cast<float*> (pC);

    dim3 threadsPerBlock(32,3,3);
    dim3 numBlocks((N + 32 - 1) / 32, 1, 1);
    rotation_kernel_shared<<<numBlocks, threadsPerBlock>>>(N, A, B, C);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

PYBIND11_MODULE(demo, m) {
    m.doc() = "pybind11 example kernel";
    m.def("py_einsum", &py_einsum, "A function which performs the matrix contraction ...ij,...jk->...ik");
    m.def("py_rotate", &py_rotate, "A function to rotate optical tensor");
    m.def("py_rotate_shared", &py_rotate_shared, "A function to rotate optical tensor, with A in shared memory");
}