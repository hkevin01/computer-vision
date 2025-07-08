#include "gpu_common.hpp"
#include <opencv2/opencv.hpp>

namespace stereo_vision
{
    namespace gpu
    {

#ifdef USE_CUDA
        /**
         * @brief CUDA-accelerated stereo matching kernel
         */
        GPU_GLOBAL void stereo_matching_kernel(
            const unsigned char *left_image,
            const unsigned char *right_image,
            short *disparity_map,
            int width,
            int height,
            int min_disparity,
            int max_disparity,
            int block_size)
        {
            int x = gpu_block_idx_x * gpu_block_dim_x + gpu_thread_idx_x;
            int y = gpu_block_idx_y * gpu_block_dim_y + gpu_thread_idx_y;

            if (x >= width || y >= height)
                return;

            int half_block = block_size / 2;

            // Bounds checking
            if (x < half_block || x >= width - half_block ||
                y < half_block || y >= height - half_block)
            {
                disparity_map[y * width + x] = 0;
                return;
            }

            int best_disparity = 0;
            int best_cost = INT_MAX;

            // Search for best disparity
            for (int d = min_disparity; d < max_disparity; d++)
            {
                if (x - d < half_block)
                    continue;

                int cost = 0;

                // Sum of Absolute Differences (SAD)
                for (int dy = -half_block; dy <= half_block; dy++)
                {
                    for (int dx = -half_block; dx <= half_block; dx++)
                    {
                        int left_idx = (y + dy) * width + (x + dx);
                        int right_idx = (y + dy) * width + (x + dx - d);

                        cost += abs((int)left_image[left_idx] - (int)right_image[right_idx]);
                    }
                }

                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_disparity = d;
                }
            }

            disparity_map[y * width + x] = (short)(best_disparity * 16); // OpenCV format
        }
#elif defined(USE_HIP)
        /**
         * @brief HIP-accelerated stereo matching kernel
         * Note: HIP kernel implementation will be added once HIP compilation issues are resolved
         */
        GPU_GLOBAL void stereo_matching_kernel(
            const unsigned char *left_image,
            const unsigned char *right_image,
            short *disparity_map,
            int width,
            int height,
            int min_disparity,
            int max_disparity,
            int block_size)
        {
            // TODO: Implement HIP kernel once compilation issues are resolved
            // For now, this is a placeholder that will be replaced with proper HIP implementation
        }
#endif

        /**
         * @brief GPU stereo matching class
         */
        class GPUStereoMatcher
        {
        public:
            GPUStereoMatcher();
            ~GPUStereoMatcher();

            bool initialize(int width, int height);
            cv::Mat computeDisparity(const cv::Mat &left, const cv::Mat &right,
                                     int min_disp = 0, int max_disp = 64, int block_size = 11);

        private:
            unsigned char *d_left_image_;
            unsigned char *d_right_image_;
            short *d_disparity_;
            int width_, height_;
            bool initialized_;

            void allocateMemory();
            void deallocateMemory();
        };

    } // namespace gpu
} // namespace stereo_vision
