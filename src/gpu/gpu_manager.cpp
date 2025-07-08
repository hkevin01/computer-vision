#include "gpu_common.hpp"
#include <iostream>
#include <stdexcept>

namespace stereo_vision
{
    namespace gpu
    {

        bool GPUManager::gpu_initialized_ = false;

        bool GPUManager::isGPUAvailable()
        {
#if defined(USE_CUDA) || defined(USE_HIP)
            int device_count = 0;
#ifdef USE_CUDA
            cudaError_t error = cudaGetDeviceCount(&device_count);
            return (error == cudaSuccess && device_count > 0);
#elif defined(USE_HIP)
            hipError_t error = hipGetDeviceCount(&device_count);
            return (error == hipSuccess && device_count > 0);
#endif
#else
            return false;
#endif
        }

        int GPUManager::getDeviceCount()
        {
#if defined(USE_CUDA) || defined(USE_HIP)
            int device_count = 0;
#ifdef USE_CUDA
            GPU_CHECK(cudaGetDeviceCount(&device_count));
#elif defined(USE_HIP)
            GPU_CHECK(hipGetDeviceCount(&device_count));
#endif
            return device_count;
#else
            return 0;
#endif
        }

        void GPUManager::setDevice(int device_id)
        {
#if defined(USE_CUDA) || defined(USE_HIP)
#ifdef USE_CUDA
            GPU_CHECK(cudaSetDevice(device_id));
#elif defined(USE_HIP)
            GPU_CHECK(hipSetDevice(device_id));
#endif
            gpu_initialized_ = true;
#endif
        }

        size_t GPUManager::getAvailableMemory()
        {
#if defined(USE_CUDA) || defined(USE_HIP)
            size_t free_mem = 0, total_mem = 0;
#ifdef USE_CUDA
            GPU_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
#elif defined(USE_HIP)
            GPU_CHECK(hipMemGetInfo(&free_mem, &total_mem));
#endif
            return free_mem;
#else
            return 0;
#endif
        }

        std::string GPUManager::getDeviceName(int device_id)
        {
#if defined(USE_CUDA) || defined(USE_HIP)
#ifdef USE_CUDA
            cudaDeviceProp prop;
            GPU_CHECK(cudaGetDeviceProperties(&prop, device_id));
            return std::string(prop.name);
#elif defined(USE_HIP)
            hipDeviceProp_t prop;
            GPU_CHECK(hipGetDeviceProperties(&prop, device_id));
            return std::string(prop.name);
#endif
#else
            return "CPU Only";
#endif
        }

        void GPUManager::checkError(gpuError_t error, const char *file, int line)
        {
#if defined(USE_CUDA) || defined(USE_HIP)
            if (error != gpuSuccess)
            {
                std::string error_msg;
#ifdef USE_CUDA
                error_msg = cudaGetErrorString(error);
#elif defined(USE_HIP)
                error_msg = hipGetErrorString(error);
#endif

                std::cerr << "GPU Error: " << error_msg
                          << " at " << file << ":" << line << std::endl;
                throw std::runtime_error("GPU Error: " + error_msg);
            }
#endif
        }

    } // namespace gpu
} // namespace stereo_vision
