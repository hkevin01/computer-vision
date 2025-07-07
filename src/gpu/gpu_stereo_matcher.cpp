#include "gpu_stereo_matcher.hpp"
#include <algorithm>
#include <climits>

namespace stereo_vision {
namespace gpu {

GPUStereoMatcher::GPUStereoMatcher() 
    : d_left_image_(nullptr), d_right_image_(nullptr), d_disparity_(nullptr),
      width_(0), height_(0), initialized_(false) {}

GPUStereoMatcher::~GPUStereoMatcher() {
    deallocateMemory();
}

bool GPUStereoMatcher::initialize(int width, int height) {
    if (!GPUManager::isGPUAvailable()) {
        return false;
    }
    
    width_ = width;
    height_ = height;
    
    try {
        allocateMemory();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        deallocateMemory();
        return false;
    }
}

void GPUStereoMatcher::allocateMemory() {
#if defined(USE_CUDA) || defined(USE_HIP)
    size_t image_size = width_ * height_ * sizeof(unsigned char);
    size_t disparity_size = width_ * height_ * sizeof(short);
    
    GPU_CHECK(gpuMalloc(&d_left_image_, image_size));
    GPU_CHECK(gpuMalloc(&d_right_image_, image_size));
    GPU_CHECK(gpuMalloc(&d_disparity_, disparity_size));
#endif
}

void GPUStereoMatcher::deallocateMemory() {
#if defined(USE_CUDA) || defined(USE_HIP)
    if (d_left_image_) {
        gpuFree(d_left_image_);
        d_left_image_ = nullptr;
    }
    if (d_right_image_) {
        gpuFree(d_right_image_);
        d_right_image_ = nullptr;
    }
    if (d_disparity_) {
        gpuFree(d_disparity_);
        d_disparity_ = nullptr;
    }
#endif
    initialized_ = false;
}

cv::Mat GPUStereoMatcher::computeDisparity(const cv::Mat& left, const cv::Mat& right,
                                          int min_disp, int max_disp, int block_size) {
    if (!initialized_) {
        throw std::runtime_error("GPUStereoMatcher not initialized");
    }
    
    if (left.size() != right.size() || left.type() != CV_8UC1 || right.type() != CV_8UC1) {
        throw std::runtime_error("Input images must be grayscale and same size");
    }
    
    cv::Mat disparity_result(left.size(), CV_16S);
    
#if defined(USE_CUDA) || defined(USE_HIP)
    // Copy images to GPU
    size_t image_size = width_ * height_ * sizeof(unsigned char);
    GPU_CHECK(gpuMemcpy(d_left_image_, left.ptr(), image_size, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_right_image_, right.ptr(), image_size, gpuMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((width_ + block_dim.x - 1) / block_dim.x,
                  (height_ + block_dim.y - 1) / block_dim.y);
    
    #ifdef USE_CUDA
        stereo_matching_kernel<<<grid_dim, block_dim>>>(
            d_left_image_, d_right_image_, d_disparity_,
            width_, height_, min_disp, max_disp, block_size);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    #elif defined(USE_HIP)
        hipLaunchKernelGGL(stereo_matching_kernel, grid_dim, block_dim, 0, 0,
            d_left_image_, d_right_image_, d_disparity_,
            width_, height_, min_disp, max_disp, block_size);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    #endif
    
    // Copy result back to CPU
    size_t disparity_size = width_ * height_ * sizeof(short);
    GPU_CHECK(gpuMemcpy(disparity_result.ptr(), d_disparity_, disparity_size, gpuMemcpyDeviceToHost));
    
#else
    // CPU fallback implementation
    disparity_result.setTo(0);
    int half_block = block_size / 2;
    
    for (int y = half_block; y < height_ - half_block; y++) {
        for (int x = half_block; x < width_ - half_block; x++) {
            int best_disparity = 0;
            int best_cost = INT_MAX;
            
            for (int d = min_disp; d < max_disp && x - d >= half_block; d++) {
                int cost = 0;
                
                for (int dy = -half_block; dy <= half_block; dy++) {
                    for (int dx = -half_block; dx <= half_block; dx++) {
                        int left_val = left.at<unsigned char>(y + dy, x + dx);
                        int right_val = right.at<unsigned char>(y + dy, x + dx - d);
                        cost += abs(left_val - right_val);
                    }
                }
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_disparity = d;
                }
            }
            
            disparity_result.at<short>(y, x) = best_disparity * 16;
        }
    }
#endif
    
    return disparity_result;
}

} // namespace gpu
} // namespace stereo_vision
