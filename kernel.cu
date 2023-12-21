#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/time.h>
#include <random>


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

    if (idx < n){
        C[idx*9 + idy*3 + idz] = A[idx*9 + idy*3 + 0] * B[idx*9 + 0 + idz] + A[idx*9 + idy*3 + 1] * B[idx*9 + 1*3 + idz] + A[idx*9 + idy*3 + 2] * B[idx*9 + 2*3 + idz];
        
    }


}

// CUDA kernel function to perform the matrix contraction '...ij,...kj->...ik'
__global__
void einsum_kernel_T(int n, float *A, float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;

    if (idx < n)
    {
        C[idx*9 + idy*3 + idz] = A[idx*9 + idy*3 + 0] * B[idx*9 + idz*3 + 0] + A[idx*9 + idy*3 + 1] * B[idx*9 + idz*3 + 1] + A[idx*9 + idy*3 + 2] * B[idx*9 + idz*3 + 2];
    }
}

// fused CUDA kernel to rotate 3x3 optical tensors using a rotation matrix R@A@R.T
__global__
void rotate_kernel(int n, float *A, float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;

    // int thread_1d = idx*9 + idy*3 + idz;
    if (idx < n) {
    __shared__ float As[288];
    __shared__ float Bs[288];
    __shared__ float Cs[288];

    As[threadIdx.x*9 + idy*3 + idz] = A[idx*9 + idy*3 + idz];
    Bs[threadIdx.x*9 + idy*3 + idz] = B[idx*9 + idy*3 + idz];
    __syncthreads();

    Cs[threadIdx.x*9 + idy*3 + idz] = As[threadIdx.x*9 + idy*3 + 0] * Bs[threadIdx.x*9 + idz*3 + 0] + 
                                      As[threadIdx.x*9 + idy*3 + 1] * Bs[threadIdx.x*9 + idz*3 + 1] + 
                                      As[threadIdx.x*9 + idy*3 + 2] * Bs[threadIdx.x*9 + idz*3 + 2];
    __syncthreads();

    C[idx*9 + idy*3 + idz] = Bs[threadIdx.x*9 + idy*3 + 0] * Cs[threadIdx.x*9 + 0*3 + idz] + 
                             Bs[threadIdx.x*9 + idy*3 + 1] * Cs[threadIdx.x*9 + 1*3 + idz] + 
                             Bs[threadIdx.x*9 + idy*3 + 2] * Cs[threadIdx.x*9 + 2*3 + idz];
    }
}
// CUDA kernel function to perform the matrix contraction '...ij,...jk->...ik'
__global__
void einsum_kernel3(int n, float *A, float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockIdx.x * blockDim.x;
    // int idy = threadIdx.y;
    // int idz = threadIdx.z;

    if (idx < n){
        int pos = idx*9;
        C[pos + 0*3 + 0] = A[pos + 0*3 + 0] * B[pos + 0 + 0] + A[pos + 0*3 + 1] * B[pos + 1*3 + 0] + A[pos + 0*3 + 2] * B[pos + 2*3 + 0];
        C[pos + 1*3 + 0] = A[pos + 1*3 + 0] * B[pos + 0 + 0] + A[pos + 1*3 + 1] * B[pos + 1*3 + 0] + A[pos + 1*3 + 2] * B[pos + 2*3 + 0];
        C[pos + 2*3 + 0] = A[pos + 2*3 + 0] * B[pos + 0 + 0] + A[pos + 2*3 + 1] * B[pos + 1*3 + 0] + A[pos + 2*3 + 2] * B[pos + 2*3 + 0];

        C[pos + 0*3 + 1] = A[pos + 0*3 + 0] * B[pos + 0 + 1] + A[pos + 0*3 + 1] * B[pos + 1*3 + 1] + A[pos + 0*3 + 2] * B[pos + 2*3 + 1];
        C[pos + 1*3 + 1] = A[pos + 1*3 + 0] * B[pos + 0 + 1] + A[pos + 1*3 + 1] * B[pos + 1*3 + 1] + A[pos + 1*3 + 2] * B[pos + 2*3 + 1];
        C[pos + 2*3 + 1] = A[pos + 2*3 + 0] * B[pos + 0 + 1] + A[pos + 2*3 + 1] * B[pos + 1*3 + 1] + A[pos + 2*3 + 2] * B[pos + 2*3 + 1];

        C[pos + 0*3 + 2] = A[pos + 0*3 + 0] * B[pos + 0 + 2] + A[pos + 0*3 + 1] * B[pos + 1*3 + 2] + A[pos + 0*3 + 2] * B[pos + 2*3 + 2];
        C[pos + 1*3 + 2] = A[pos + 1*3 + 0] * B[pos + 0 + 2] + A[pos + 1*3 + 1] * B[pos + 1*3 + 2] + A[pos + 1*3 + 2] * B[pos + 2*3 + 2];
        C[pos + 2*3 + 2] = A[pos + 2*3 + 0] * B[pos + 0 + 2] + A[pos + 2*3 + 1] * B[pos + 1*3 + 2] + A[pos + 2*3 + 2] * B[pos + 2*3 + 2];
        
    }


}


int main(void)
{   
    std::mt19937 gen(1);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    struct timeval start_cpu, end_cpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    int N = 512*512;
    float *h_x, *h_y, *h_z, *ans_z, *d_x, *d_y, *d_z, *d_tmp;
    int mem_size = N*3*3*sizeof(float);

    // Allocate Host Memory
    gpuErrchk(cudaMallocHost(&h_x, mem_size));
    gpuErrchk(cudaMallocHost(&h_y, mem_size));
    gpuErrchk(cudaMallocHost(&h_z, mem_size));

    // Allocate Device Memory
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_y), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**> (&d_tmp), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_z), mem_size));

    for (int i = 0; i < N*3*3; i++)
    {
        h_x[i] = dis(gen);
        h_y[i] = dis(gen);
    }

    // copy host memory to device
    gpuErrchk(cudaMemcpy(d_x, h_x, mem_size, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_y, h_y, mem_size, cudaMemcpyHostToDevice))

    // dim3 threadsPerBlock(3, 3, 32);
    // dim3 numBlocks(1, 1, (N + 32 - 1) / 32);

    dim3 threadsPerBlock(32, 3, 3);
    dim3 numBlocks((N + 32 - 1) / 32, 1, 1);
    // int threadsPerBlock = 128;
    // int numBlocks = (N+128-1)/128;
    float total_CPU = 0.0;
    float total_GPU = 0.0;

    for (int i = 0; i < 10; ++i)
    {
    gettimeofday(&start_cpu,NULL);
    cudaEventRecord(start);
    cudaEventSynchronize(start);


    rotate_kernel<<<numBlocks, threadsPerBlock>>>(N,d_x,d_y,d_z);
    gpuErrchk( cudaPeekAtLastError() )
    gpuErrchk( cudaDeviceSynchronize() );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gettimeofday(&end_cpu, NULL);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float cpu_milliseconds = (end_cpu.tv_usec - start_cpu.tv_usec)/1000.0 - milliseconds;
    if (i > 0){
    total_CPU += cpu_milliseconds;
    total_GPU += milliseconds;
    }
    std::cout << "Run " << i << ":\n";
    std::cout << "GPU Time: " << milliseconds*1000.0 << " us" << "\n";
    std::cout << "CPU Time: " << cpu_milliseconds*1000.0 << " us" << "\n";
    }
    std::cout << "Averages: \n";
    std::cout << "GPU Time: " << total_GPU/9.0*1000.0 << " us" << "\n";
    std::cout << "CPU Time: " << total_CPU/9.0*1000.0 << " us" << "\n";

    // einsum_kernel_T<<<numBlocks, threadsPerBlock>>>(N, d_x, d_y, d_tmp);
    // cudaDeviceSynchronize();
    // einsum_kernel<<<numBlocks, threadsPerBlock>>>(N, d_y, d_tmp, d_z);

    // rotate_kernel<<<numBlocks, threadsPerBlock>>>(N, d_x, d_y, d_z);
    gpuErrchk( cudaMemcpy(h_z, d_z, mem_size, cudaMemcpyDeviceToHost) );

    // gpuErrchk( cudaMallocHost(&ans_z, mem_size));
    ans_z = (float*)malloc(mem_size);
    float *tmp_z = (float*)malloc(N*3*3*sizeof(float));
    float tmp = 0;
    for (int n = 0; n < N; ++n){
        for (int j = 0; j < 3; ++j){
            for (int k = 0; k < 3; ++k){
                tmp = 0;
                for (int i = 0; i < 3; ++i){
                    tmp += h_x[n*9 + j*3 + i] * h_y[n*9 + k*3 + i]; 
                }
                tmp_z[n*9 + j*3 + k] = tmp;
            }
        }
    }

    for (int n=0; n < N; ++n){
        for (int j = 0; j < 3; ++j){
            for (int k = 0; k < 3; ++k){
                tmp = 0;
                for (int i = 0; i < 3; ++i){
                    tmp += h_y[n*9 + k*3 + i] * tmp_z[n*9 + i*3 + j];
                }
                ans_z[n*9 + k*3 + j] = tmp;
            }
        }
    }

    bool success = true;
    float diff;

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < 3; ++j){
            for (int k = 0; k < 3; ++k){
                diff = abs(ans_z[i*3*3 + j*3 + k] - h_z[i*3*3 + j*3 + k]);
                if (diff > 1e-6){
                std::cout << "Error greater than 1e-6 at index: " << i*9 + j*3 + k << "\n";
                std::cout << ans_z[i*3*3 + j*3 +k] << std::endl;
                std::cout << h_z[i*3*3 + j*3 + k] << std::endl;
                success = false;
                return 1;
                }
            }
        

        }
    }
    if (success){
        std::cout << "Successful Matrix Multiplication!" << "\n";
    }

    // for (int i = 0; i < N*3*3; ++i){
    //     std::cout << i << " " << ans_z[i] << " " << h_z[i] << std::endl;
    // }
    gpuErrchk( cudaFree(d_x) );
    gpuErrchk( cudaFree(d_y) );
    gpuErrchk( cudaFree(d_z) );

    gpuErrchk( cudaFreeHost(h_x) );
    gpuErrchk( cudaFreeHost(h_y) );
    gpuErrchk( cudaFreeHost(h_z) );
    // gpuErrchk( cudaFreeHost(ans_z) );
    free(ans_z);
    return 0;
}