#pragma once

// Cross-platform GPU programming header
// This header provides unified interface for both CUDA and HIP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GPU_DEVICE __device__
#define GPU_HOST __host__
#define GPU_GLOBAL __global__
#define GPU_SHARED __shared__
#define GPU_CONSTANT __constant__

// CUDA-specific types
using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;

// CUDA-specific functions
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset
#define gpuGetLastError cudaGetLastError
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime

// CUDA constants
#define gpuSuccess cudaSuccess

// Thread indexing
#define gpu_thread_idx_x threadIdx.x
#define gpu_thread_idx_y threadIdx.y
#define gpu_thread_idx_z threadIdx.z
#define gpu_block_idx_x blockIdx.x
#define gpu_block_idx_y blockIdx.y
#define gpu_block_idx_z blockIdx.z
#define gpu_block_dim_x blockDim.x
#define gpu_block_dim_y blockDim.y
#define gpu_block_dim_z blockDim.z
#define gpu_grid_dim_x gridDim.x
#define gpu_grid_dim_y gridDim.y
#define gpu_grid_dim_z gridDim.z

#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_common.h>
// Optional HIP libraries - only include if available
#ifdef HIP_HAS_BLAS
#include <hipblas.h>
#endif
#define GPU_DEVICE __device__
#define GPU_HOST __host__
#define GPU_GLOBAL __global__
#define GPU_SHARED __shared__
#define GPU_CONSTANT __constant__

// HIP-specific types
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;

// HIP-specific functions
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset hipMemset
#define gpuGetLastError hipGetLastError
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime

// HIP constants
#define gpuSuccess hipSuccess

// Thread indexing - HIP uses the same thread indexing as CUDA
#define gpu_thread_idx_x threadIdx.x
#define gpu_thread_idx_y threadIdx.y
#define gpu_thread_idx_z threadIdx.z
#define gpu_block_idx_x blockIdx.x
#define gpu_block_idx_y blockIdx.y
#define gpu_block_idx_z blockIdx.z
#define gpu_block_dim_x blockDim.x
#define gpu_block_dim_y blockDim.y
#define gpu_block_dim_z blockDim.z
#define gpu_grid_dim_x gridDim.x
#define gpu_grid_dim_y gridDim.y
#define gpu_grid_dim_z gridDim.z

#else
// CPU fallback definitions
#define GPU_DEVICE
#define GPU_HOST
#define GPU_GLOBAL
#define GPU_SHARED
#define GPU_CONSTANT

// Stub types for CPU-only builds
using gpuError_t = int;
using gpuStream_t = void *;
using gpuEvent_t = void *;

#endif

namespace stereo_vision
{
    namespace gpu
    {

        /**
         * @brief GPU utility functions
         */
        class GPUManager
        {
        public:
            static bool isGPUAvailable();
            static int getDeviceCount();
            static void setDevice(int device_id);
            static size_t getAvailableMemory();
            static std::string getDeviceName(int device_id = 0);
            static void checkError(gpuError_t error, const char *file, int line);

        private:
            static bool gpu_initialized_;
        };

// Error checking macro
#define GPU_CHECK(call)                                                            \
    do                                                                             \
    {                                                                              \
        gpuError_t error = call;                                                   \
        if (error != gpuSuccess)                                                   \
        {                                                                          \
            stereo_vision::gpu::GPUManager::checkError(error, __FILE__, __LINE__); \
        }                                                                          \
    } while (0)

    } // namespace gpu
} // namespace stereo_vision
