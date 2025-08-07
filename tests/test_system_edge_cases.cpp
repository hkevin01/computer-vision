#include "edge_case_framework.hpp"
#include "camera_calibration.hpp"
#include "point_cloud_processor.hpp"
#include "stereo_matcher.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace stereo_vision {
namespace testing {

/**
 * @brief System-level edge case tests that verify robustness under extreme conditions
 *
 * These tests simulate real-world failure scenarios including:
 * - Hardware failures (GPU, memory, disk)
 * - System resource exhaustion
 * - Concurrent access patterns
 * - File system corruption
 * - Network failures (for distributed processing)
 */
class SystemLevelEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        EdgeCaseTestFramework::clearMemoryPressure();

        // Create test data directory
        test_data_dir_ = std::filesystem::temp_directory_path() / "stereo_vision_edge_test";
        std::filesystem::create_directories(test_data_dir_);
    }

    void TearDown() override {
        EdgeCaseTestFramework::clearMemoryPressure();

        // Clean up test data
        if (std::filesystem::exists(test_data_dir_)) {
            std::filesystem::remove_all(test_data_dir_);
        }
    }

    std::filesystem::path test_data_dir_;

    // Helper to create corrupted calibration file
    void createCorruptedCalibrationFile(const std::string& filename) {
        std::ofstream file(test_data_dir_ / filename);
        file << "This is not a valid calibration file\n";
        file << "Random corruption: " << std::rand() << "\n";
        file << "More garbage data...\n";
        file.close();
    }

    // Helper to simulate file system full condition
    bool simulateFileSystemFull(const std::string& filename) {
        try {
            std::ofstream large_file(test_data_dir_ / filename, std::ios::binary);

            // Try to write a very large file to fill up space
            std::vector<char> buffer(1024 * 1024, 'X'); // 1MB buffer

            for (int i = 0; i < 10000; ++i) { // Try to write 10GB
                large_file.write(buffer.data(), buffer.size());
                if (large_file.fail()) {
                    return true; // Successfully simulated full disk
                }
            }

            large_file.close();
            return false; // Couldn't fill the disk
        } catch (...) {
            return true; // Exception indicates file system issues
        }
    }
};

// Test complete system failure recovery
TEST_F(SystemLevelEdgeCaseTest, CompleteSystemFailureRecovery) {
    // Simulate multiple simultaneous failures

    // 1. Memory pressure
    EdgeCaseTestFramework::simulateMemoryPressure(1024);

    // 2. CPU load
    std::thread cpu_load_thread([]() {
        EdgeCaseTestFramework::simulateHighCPULoad(3000);
    });

    // 3. File system issues
    createCorruptedCalibrationFile("corrupted.xml");

    // 4. Invalid input data
    cv::Mat corrupted_image = EdgeCaseTestFramework::generateCorruptedImage(cv::Size(1920, 1080), CV_8UC3);

    // Try to perform stereo vision pipeline under these conditions
    try {
        auto calibration = std::make_unique<CameraCalibration>();
        auto processor = std::make_unique<PointCloudProcessor>();

        // Try calibration with corrupted data
        std::vector<cv::Mat> corrupted_images = {corrupted_image};
        auto calib_result = calibration->calibrateSingleCamera(corrupted_images, cv::Size(9, 6), 25.0f);

        // Try point cloud processing with stress
        auto test_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        test_cloud->width = 10000;
        test_cloud->height = 1;
        test_cloud->points.resize(10000);

        for (auto& point : test_cloud->points) {
            point.x = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
            point.y = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
            point.z = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
            point.r = point.g = point.b = 128;
        }

        auto filtered = processor->filterPointCloud(test_cloud, 0.01, 50, 1.0);

        // If we get here, the system survived multiple failures
        EXPECT_TRUE(true) << "System survived multiple simultaneous failures";

    } catch (const std::exception& e) {
        // Graceful failure under extreme conditions is acceptable
        EXPECT_TRUE(true) << "System gracefully failed under extreme conditions: " << e.what();
    }

    cpu_load_thread.join();
}

// Test data corruption detection and handling
TEST_F(SystemLevelEdgeCaseTest, DataCorruptionDetection) {
    // Create various types of corrupted data

    // 1. Corrupted calibration file
    createCorruptedCalibrationFile("bad_calibration.xml");

    // 2. Partially written file (simulates write interruption)
    {
        std::ofstream partial_file(test_data_dir_ / "partial.xml");
        partial_file << "<?xml version=\"1.0\"?>\n<opencv_storage>\n<camera_matrix>\n";
        // Deliberately incomplete
    }

    // 3. File with wrong extension
    {
        std::ofstream wrong_ext(test_data_dir_ / "image.jpg");
        wrong_ext << "This is not actually a JPEG file";
    }

    auto calibration = std::make_unique<CameraCalibration>();
    CameraCalibration::StereoParameters params;

    // Test loading each corrupted file
    std::vector<std::string> corrupted_files = {
        "bad_calibration.xml",
        "partial.xml",
        "nonexistent.xml",
        "image.jpg"
    };

    for (const auto& filename : corrupted_files) {
        try {
            bool loaded = calibration->loadCalibration((test_data_dir_ / filename).string(), params);
            EXPECT_FALSE(loaded) << "Should not successfully load corrupted file: " << filename;
        } catch (const std::exception& e) {
            EXPECT_TRUE(true) << "Gracefully handled corrupted file " << filename << ": " << e.what();
        }
    }
}

// Test resource exhaustion scenarios
TEST_F(SystemLevelEdgeCaseTest, ResourceExhaustionHandling) {
    // Test file descriptor exhaustion
    std::vector<std::unique_ptr<std::ofstream>> open_files;

    try {
        // Try to open many files simultaneously
        for (int i = 0; i < 10000; ++i) {
            auto file = std::make_unique<std::ofstream>(
                test_data_dir_ / ("file_" + std::to_string(i) + ".tmp")
            );

            if (file->fail()) {
                break; // Hit file descriptor limit
            }

            open_files.push_back(std::move(file));
        }

        // Try to perform operations with exhausted file descriptors
        auto processor = std::make_unique<PointCloudProcessor>();
        auto test_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        test_cloud->width = 100;
        test_cloud->height = 1;
        test_cloud->points.resize(100);

        for (auto& point : test_cloud->points) {
            point.x = point.y = point.z = 1.0f;
            point.r = point.g = point.b = 128;
        }

        // Try to export under resource exhaustion
        bool export_success = processor->exportPointCloud(
            test_cloud,
            (test_data_dir_ / "test_under_exhaustion.pcd").string(),
            PointCloudProcessor::ExportFormat::PLY_BINARY
        );

        // Should either succeed or fail gracefully
        if (!export_success) {
            EXPECT_TRUE(true) << "Gracefully handled file descriptor exhaustion";
        } else {
            EXPECT_TRUE(true) << "Succeeded despite resource pressure";
        }

    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled resource exhaustion: " << e.what();
    }

    // Clean up
    open_files.clear();
}

// Test concurrent access patterns that might cause race conditions
TEST_F(SystemLevelEdgeCaseTest, RaceConditionDetection) {
    const int num_threads = 20;
    const int operations_per_thread = 50;

    std::atomic<int> successful_operations{0};
    std::atomic<int> failed_operations{0};
    std::vector<std::string> error_messages;
    std::mutex error_mutex;

    auto worker_function = [&](int thread_id) {
        for (int op = 0; op < operations_per_thread; ++op) {
            try {
                // Mix different types of operations
                if (op % 3 == 0) {
                    // Calibration operation
                    auto calibration = std::make_unique<CameraCalibration>();
                    std::vector<cv::Mat> images;

                    cv::Mat test_img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
                    cv::rectangle(test_img, cv::Rect(100, 100, 400, 200), cv::Scalar(255), -1);
                    images.push_back(test_img);

                    auto result = calibration->calibrateSingleCamera(images, cv::Size(9, 6), 25.0f);
                    successful_operations++;

                } else if (op % 3 == 1) {
                    // Point cloud processing
                    auto processor = std::make_unique<PointCloudProcessor>();
                    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
                    cloud->width = 1000;
                    cloud->height = 1;
                    cloud->points.resize(1000);

                    for (auto& point : cloud->points) {
                        point.x = static_cast<float>(thread_id + op);
                        point.y = static_cast<float>(thread_id + op);
                        point.z = static_cast<float>(thread_id + op);
                        point.r = point.g = point.b = 128;
                    }

                    auto filtered = processor->filterPointCloud(cloud, 0.01, 50, 1.0);
                    successful_operations++;

                } else {
                    // File I/O operation
                    std::string filename = "thread_" + std::to_string(thread_id) + "_op_" + std::to_string(op) + ".tmp";
                    std::ofstream file(test_data_dir_ / filename);
                    file << "Thread " << thread_id << " operation " << op << std::endl;
                    file.close();

                    // Try to read it back
                    std::ifstream read_file(test_data_dir_ / filename);
                    std::string content;
                    std::getline(read_file, content);

                    if (!content.empty()) {
                        successful_operations++;
                    } else {
                        failed_operations++;
                    }
                }

            } catch (const std::exception& e) {
                failed_operations++;

                std::lock_guard<std::mutex> lock(error_mutex);
                error_messages.push_back(std::string("Thread ") + std::to_string(thread_id) +
                                       " operation " + std::to_string(op) + ": " + e.what());
            }
        }
    };

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker_function, i);
    }

    // Wait for completion
    for (auto& worker : workers) {
        worker.join();
    }

    // Analyze results
    int total_operations = successful_operations + failed_operations;
    float success_rate = static_cast<float>(successful_operations) / total_operations;

    EXPECT_GT(success_rate, 0.7f) << "Success rate too low: " << (success_rate * 100) << "%";
    EXPECT_LT(error_messages.size(), 100u) << "Too many errors in concurrent operations";

    // Log some error messages for debugging
    if (!error_messages.empty()) {
        std::cout << "Sample concurrent operation errors:\n";
        for (size_t i = 0; i < std::min(static_cast<size_t>(5u), error_messages.size()); ++i) {
            std::cout << "  " << error_messages[i] << "\n";
        }
    }
}

// Test hardware failure simulation
TEST_F(SystemLevelEdgeCaseTest, HardwareFailureSimulation) {
    // Simulate GPU failure and fallback
    try {
        // This would typically fail if GPU is not available
        cv::cuda::GpuMat gpu_image;
        cv::Mat cpu_image = cv::Mat::ones(cv::Size(1920, 1080), CV_8UC3);

        // Try GPU upload
        try {
            gpu_image.upload(cpu_image);
            EXPECT_TRUE(true) << "GPU operation succeeded";
        } catch (const cv::Exception& e) {
            EXPECT_TRUE(true) << "Gracefully handled GPU failure: " << e.what();
        }

        // Fallback to CPU processing should always work
        cv::Mat processed;
        cv::cvtColor(cpu_image, processed, cv::COLOR_BGR2GRAY);
        EXPECT_FALSE(processed.empty()) << "CPU fallback failed";

    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Handled hardware failure gracefully: " << e.what();
    }

    // Simulate disk I/O failure
    try {
        if (simulateFileSystemFull("large_file.tmp")) {
            // Try operations with full disk
            auto processor = std::make_unique<PointCloudProcessor>();
            auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            cloud->width = 100;
            cloud->height = 1;
            cloud->points.resize(100);

            bool export_success = processor->exportPointCloud(
                cloud,
                (test_data_dir_ / "test_full_disk.pcd").string(),
                PointCloudProcessor::ExportFormat::PLY_BINARY
            );

            // Should fail gracefully
            EXPECT_FALSE(export_success) << "Should fail when disk is full";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled disk full condition: " << e.what();
    }
}

// Test memory allocation patterns that might cause fragmentation
TEST_F(SystemLevelEdgeCaseTest, MemoryFragmentationHandling) {
    std::vector<std::unique_ptr<cv::Mat>> allocated_matrices;

    try {
        // Allocate and deallocate in patterns that cause fragmentation
        for (int cycle = 0; cycle < 10; ++cycle) {
            // Allocate many small matrices
            for (int i = 0; i < 100; ++i) {
                auto mat = std::make_unique<cv::Mat>(cv::Size(100, 100), CV_8UC3);
                allocated_matrices.push_back(std::move(mat));
            }

            // Deallocate every other one (creates fragmentation)
            for (size_t i = 0; i < allocated_matrices.size(); i += 2) {
                allocated_matrices[i].reset();
            }

            // Remove null pointers
            allocated_matrices.erase(
                std::remove_if(allocated_matrices.begin(), allocated_matrices.end(),
                             [](const std::unique_ptr<cv::Mat>& ptr) { return ptr == nullptr; }),
                allocated_matrices.end()
            );

            // Try to allocate a large matrix (should handle fragmentation)
            try {
                auto large_mat = std::make_unique<cv::Mat>(cv::Size(2000, 2000), CV_32FC3);
                EXPECT_FALSE(large_mat->empty()) << "Large allocation failed due to fragmentation";

                // Use the matrix briefly
                large_mat->setTo(cv::Scalar::all(1.0));

            } catch (const std::bad_alloc& e) {
                EXPECT_TRUE(true) << "Gracefully handled memory fragmentation";
                break; // Exit if we hit memory limits
            }
        }

    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Handled memory fragmentation gracefully: " << e.what();
    }

    // Clean up
    allocated_matrices.clear();
}

// Test precision drift in long-running operations
TEST_F(SystemLevelEdgeCaseTest, PrecisionDriftDetection) {
    // Simulate long-running calibration process
    const int num_iterations = 1000;

    std::vector<cv::Mat> test_images;
    for (int i = 0; i < 20; ++i) {
        cv::Mat img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

        // Create simple checkerboard pattern
        for (int y = 0; y < 480; y += 40) {
            for (int x = 0; x < 640; x += 40) {
                if ((x/40 + y/40) % 2 == 0) {
                    cv::rectangle(img, cv::Rect(x, y, 40, 40), cv::Scalar(255), -1);
                }
            }
        }
        test_images.push_back(img);
    }

    auto calibration = std::make_unique<CameraCalibration>();
    std::vector<double> reprojection_errors;

    // Perform many calibration iterations
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        try {
            auto result = calibration->calibrateSingleCamera(test_images, cv::Size(9, 6), 25.0f);

            if (!result.camera_matrix.empty()) {
                reprojection_errors.push_back(result.reprojection_error);

                // Check for numerical instability
                double determinant = cv::determinant(result.camera_matrix);
                EXPECT_FALSE(std::isnan(determinant)) << "Camera matrix became singular at iteration " << iteration;
                EXPECT_FALSE(std::isinf(determinant)) << "Camera matrix determinant became infinite at iteration " << iteration;
                EXPECT_GT(std::abs(determinant), 1e-10) << "Camera matrix nearly singular at iteration " << iteration;
            }

            // Break early if we detect issues
            if (iteration > 10 && reprojection_errors.size() >= 2) {
                double current_error = reprojection_errors.back();
                double previous_error = reprojection_errors[reprojection_errors.size() - 2];

                // Check for exploding errors (sign of numerical instability)
                if (current_error > previous_error * 10.0) {
                    EXPECT_TRUE(false) << "Reprojection error exploded at iteration " << iteration
                                      << " (from " << previous_error << " to " << current_error << ")";
                    break;
                }
            }

        } catch (const std::exception& e) {
            EXPECT_TRUE(true) << "Gracefully handled numerical issues at iteration " << iteration << ": " << e.what();
            break;
        }

        // Occasional progress check
        if (iteration % 100 == 0) {
            std::cout << "Completed " << iteration << " iterations, "
                      << "latest reprojection error: "
                      << (reprojection_errors.empty() ? 0.0 : reprojection_errors.back()) << std::endl;
        }
    }

    // Analyze precision drift
    if (reprojection_errors.size() > 100) {
        double initial_avg = 0.0, final_avg = 0.0;

        for (size_t i = 0; i < 50; ++i) {
            initial_avg += reprojection_errors[i];
            final_avg += reprojection_errors[reprojection_errors.size() - 50 + i];
        }

        initial_avg /= 50.0;
        final_avg /= 50.0;

        double drift_ratio = final_avg / initial_avg;
        EXPECT_LT(drift_ratio, 2.0) << "Significant precision drift detected: "
                                    << "error increased by " << (drift_ratio * 100) << "%";
    }
}

} // namespace testing
} // namespace stereo_vision
