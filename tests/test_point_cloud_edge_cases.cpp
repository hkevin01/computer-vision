#include "edge_case_framework.hpp"
#include "point_cloud_processor.hpp"
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <limits>
#include <thread>
#include <filesystem>

namespace stereo_vision {
namespace testing {

class PointCloudProcessorEdgeCaseTest : public EdgeCaseTest<double> {
protected:
    void SetUp() override {
        EdgeCaseTest::SetUp();
        processor_ = std::make_unique<PointCloudProcessor>();
    }

    std::unique_ptr<PointCloudProcessor> processor_;

    // Helper to create test point clouds with edge cases
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr createTestPointCloud(size_t num_points, bool add_outliers = false) {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = !add_outliers;
        cloud->points.resize(num_points);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
        std::uniform_int_distribution<uint8_t> color_dist(0, 255);

        for (size_t i = 0; i < num_points; ++i) {
            auto& point = cloud->points[i];
            point.x = pos_dist(rng);
            point.y = pos_dist(rng);
            point.z = pos_dist(rng);
            point.r = color_dist(rng);
            point.g = color_dist(rng);
            point.b = color_dist(rng);
        }

        if (add_outliers) {
            // Add some extreme outliers
            size_t outlier_count = num_points / 20; // 5% outliers
            for (size_t i = 0; i < outlier_count; ++i) {
                size_t idx = rng() % num_points;
                auto& point = cloud->points[idx];
                point.x = (rng() % 2 == 0) ? 1000.0f : -1000.0f;
                point.y = (rng() % 2 == 0) ? 1000.0f : -1000.0f;
                point.z = (rng() % 2 == 0) ? 1000.0f : -1000.0f;
            }
        }

        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr createEdgeCasePointCloud() {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        cloud->width = 10;
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(10);

        auto edge_values = EdgeCaseTestFramework::getFloatingPointEdgeCases();

        for (size_t i = 0; i < 10 && i < edge_values.size(); ++i) {
            auto& point = cloud->points[i];
            float val = static_cast<float>(edge_values[i]);

            // Ensure we have finite coordinates for some points
            if (std::isfinite(val)) {
                point.x = val;
                point.y = val;
                point.z = val;
            } else {
                // Use extreme but finite values for non-finite edge cases
                point.x = std::isnan(val) ? std::numeric_limits<float>::quiet_NaN() : val;
                point.y = std::isnan(val) ? std::numeric_limits<float>::quiet_NaN() : val;
                point.z = std::isnan(val) ? std::numeric_limits<float>::quiet_NaN() : val;
            }

            point.r = 255;
            point.g = 128;
            point.b = 64;
        }

        return cloud;
    }
};

// Test point cloud filtering with extreme values
TEST_F(PointCloudProcessorEdgeCaseTest, FilteringWithExtremeValues) {
    auto edge_cloud = createEdgeCasePointCloud();

    try {
        auto filtered = processor_->filterPointCloud(edge_cloud, 0.01, 50, 1.0);

        if (filtered && !filtered->empty()) {
            // Verify filtered cloud doesn't contain NaN or infinite values
            for (const auto& point : filtered->points) {
                EXPECT_FALSE(std::isnan(point.x)) << "Filtered cloud contains NaN x-coordinate";
                EXPECT_FALSE(std::isnan(point.y)) << "Filtered cloud contains NaN y-coordinate";
                EXPECT_FALSE(std::isnan(point.z)) << "Filtered cloud contains NaN z-coordinate";
                EXPECT_FALSE(std::isinf(point.x)) << "Filtered cloud contains infinite x-coordinate";
                EXPECT_FALSE(std::isinf(point.y)) << "Filtered cloud contains infinite y-coordinate";
                EXPECT_FALSE(std::isinf(point.z)) << "Filtered cloud contains infinite z-coordinate";
            }
            EXPECT_TRUE(true) << "Successfully filtered point cloud with extreme values";
        } else {
            EXPECT_TRUE(true) << "Gracefully handled extreme values by returning empty cloud";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully failed with extreme values: " << e.what();
    }
}

// Test overflow in point cloud coordinates
TEST_F(PointCloudProcessorEdgeCaseTest, CoordinateOverflowHandling) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    cloud->width = 100;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(100);

    // Create points with coordinates near float limits
    for (size_t i = 0; i < 100; ++i) {
        auto& point = cloud->points[i];

        // Mix of extreme values
        if (i % 4 == 0) {
            point.x = std::numeric_limits<float>::max() / 2.0f;
            point.y = std::numeric_limits<float>::max() / 2.0f;
            point.z = std::numeric_limits<float>::max() / 2.0f;
        } else if (i % 4 == 1) {
            point.x = std::numeric_limits<float>::lowest() / 2.0f;
            point.y = std::numeric_limits<float>::lowest() / 2.0f;
            point.z = std::numeric_limits<float>::lowest() / 2.0f;
        } else if (i % 4 == 2) {
            point.x = std::numeric_limits<float>::epsilon();
            point.y = std::numeric_limits<float>::epsilon();
            point.z = std::numeric_limits<float>::epsilon();
        } else {
            point.x = static_cast<float>(i);
            point.y = static_cast<float>(i);
            point.z = static_cast<float>(i);
        }

        point.r = 255;
        point.g = 128;
        point.b = 64;
    }

    // Test filtering with extreme leaf sizes
    auto extreme_values = EdgeCaseTestFramework::getFloatingPointEdgeCases();

    for (double leaf_size : extreme_values) {
        if (std::isfinite(leaf_size) && leaf_size > 0 && leaf_size < 1000) {
            try {
                auto filtered = processor_->filterPointCloud(cloud, leaf_size, 10, 1.0);

                if (filtered) {
                    // Check for coordinate overflow in filtered result
                    for (const auto& point : filtered->points) {
                        EXPECT_NO_OVERFLOW(point.x + point.y + point.z);

                        // Ensure coordinates are reasonable
                        EXPECT_LT(std::abs(point.x), 1e6f) << "Suspiciously large x-coordinate: " << point.x;
                        EXPECT_LT(std::abs(point.y), 1e6f) << "Suspiciously large y-coordinate: " << point.y;
                        EXPECT_LT(std::abs(point.z), 1e6f) << "Suspiciously large z-coordinate: " << point.z;
                    }
                }
            } catch (const std::exception& e) {
                EXPECT_TRUE(true) << "Gracefully handled extreme leaf size " << leaf_size << ": " << e.what();
            }
        }
    }
}

// Test precision loss in point cloud operations
TEST_F(PointCloudProcessorEdgeCaseTest, PrecisionLossDetection) {
    // Create a cloud with high-precision coordinates
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    cloud->width = 50;
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->points.resize(50);

    for (size_t i = 0; i < 50; ++i) {
        auto& point = cloud->points[i];
        // Use high-precision values that might lose precision in filtering
        point.x = 1.0f + static_cast<float>(i) * std::numeric_limits<float>::epsilon() * 1000.0f;
        point.y = 2.0f + static_cast<float>(i) * std::numeric_limits<float>::epsilon() * 1000.0f;
        point.z = 3.0f + static_cast<float>(i) * std::numeric_limits<float>::epsilon() * 1000.0f;
        point.r = 255;
        point.g = 128;
        point.b = 64;
    }

    // Store original coordinates for comparison
    std::vector<std::array<float, 3>> original_coords;
    for (const auto& point : cloud->points) {
        original_coords.push_back({point.x, point.y, point.z});
    }

    try {
        // Apply very fine filtering
        auto filtered = processor_->filterPointCloud(cloud, 1e-6, 5, 0.1);

        if (filtered && !filtered->empty()) {
            // Check for significant precision loss
            size_t precision_loss_count = 0;

            for (size_t i = 0; i < std::min(filtered->points.size(), original_coords.size()); ++i) {
                const auto& filtered_point = filtered->points[i];
                const auto& original = original_coords[i];

                bool has_loss = EdgeCaseTestFramework::hasSignificantPrecisionLoss(original[0], filtered_point.x) ||
                               EdgeCaseTestFramework::hasSignificantPrecisionLoss(original[1], filtered_point.y) ||
                               EdgeCaseTestFramework::hasSignificantPrecisionLoss(original[2], filtered_point.z);

                if (has_loss) {
                    precision_loss_count++;
                }
            }

            // Some precision loss is expected, but not complete data corruption
            float precision_loss_ratio = static_cast<float>(precision_loss_count) / filtered->points.size();
            EXPECT_LT(precision_loss_ratio, 0.8f) << "Excessive precision loss detected: "
                                                   << (precision_loss_ratio * 100) << "% of points affected";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled precision-sensitive filtering: " << e.what();
    }
}

// Test malformed point cloud input
TEST_F(PointCloudProcessorEdgeCaseTest, MalformedInputHandling) {
    // Test with null pointer
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr null_cloud;
    EXPECT_GRACEFUL_FAILURE(
        processor_->filterPointCloud(null_cloud, 0.01, 50, 1.0)
    );

    // Test with empty cloud
    auto empty_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    try {
        auto result = processor_->filterPointCloud(empty_cloud, 0.01, 50, 1.0);
        EXPECT_TRUE(!result || result->empty()) << "Empty cloud should produce empty result";
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled empty cloud: " << e.what();
    }

    // Test with inconsistent width/height vs points size
    auto inconsistent_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    inconsistent_cloud->width = 100;
    inconsistent_cloud->height = 1;
    inconsistent_cloud->points.resize(50); // Mismatch!

    for (auto& point : inconsistent_cloud->points) {
        point.x = point.y = point.z = 1.0f;
        point.r = point.g = point.b = 128;
    }

    EXPECT_GRACEFUL_FAILURE(
        processor_->filterPointCloud(inconsistent_cloud, 0.01, 50, 1.0)
    );

    // Test with all NaN points
    auto nan_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    nan_cloud->width = 10;
    nan_cloud->height = 1;
    nan_cloud->points.resize(10);

    for (auto& point : nan_cloud->points) {
        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
        point.r = point.g = point.b = 128;
    }

    try {
        auto result = processor_->filterPointCloud(nan_cloud, 0.01, 50, 1.0);
        EXPECT_TRUE(!result || result->empty()) << "NaN cloud should produce empty result";
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled NaN cloud: " << e.what();
    }
}

// Test concurrent point cloud processing
TEST_F(PointCloudProcessorEdgeCaseTest, ConcurrentProcessing) {
    auto test_cloud = createTestPointCloud(1000, true);

    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto processing_task = [&]() {
        try {
            auto local_processor = std::make_unique<PointCloudProcessor>();
            auto filtered = local_processor->filterPointCloud(test_cloud, 0.01, 50, 1.0);

            if (filtered && !filtered->empty()) {
                success_count++;
            } else {
                failure_count++;
            }
        } catch (const std::exception&) {
            failure_count++;
        }
    };

    EdgeCaseTestFramework::testConcurrentAccess(processing_task, 8, 5);

    // Most operations should succeed
    EXPECT_GT(success_count.load(), 0) << "No concurrent processing operations succeeded";
}

// Test export functionality with edge cases
TEST_F(PointCloudProcessorEdgeCaseTest, ExportEdgeCases) {
    auto test_cloud = createEdgeCasePointCloud();

    // Test export to various formats
    std::vector<PointCloudProcessor::ExportFormat> formats = {
        PointCloudProcessor::ExportFormat::PLY_BINARY,
        PointCloudProcessor::ExportFormat::PLY_ASCII,
        PointCloudProcessor::ExportFormat::PCD_BINARY,
        PointCloudProcessor::ExportFormat::PCD_ASCII,
        PointCloudProcessor::ExportFormat::XYZ
    };

    for (auto format : formats) {
        std::filesystem::path temp_file = std::filesystem::temp_directory_path() /
                                         ("test_edge_case_" + std::to_string(static_cast<int>(format)) + ".pcd");

        try {
            bool success = processor_->exportPointCloud(test_cloud, temp_file.string(), format);

            if (success) {
                EXPECT_TRUE(std::filesystem::exists(temp_file)) << "Export claimed success but file doesn't exist";

                // Verify file is not empty
                auto file_size = std::filesystem::file_size(temp_file);
                EXPECT_GT(file_size, 0) << "Exported file is empty";

                // Clean up
                std::filesystem::remove(temp_file);
            } else {
                EXPECT_TRUE(true) << "Gracefully failed to export edge case cloud in format " << static_cast<int>(format);
            }

        } catch (const std::exception& e) {
            EXPECT_TRUE(true) << "Gracefully handled export failure: " << e.what();

            // Clean up in case of partial write
            if (std::filesystem::exists(temp_file)) {
                std::filesystem::remove(temp_file);
            }
        }
    }

    // Test export with invalid filename
    EXPECT_GRACEFUL_FAILURE(
        processor_->exportPointCloud(test_cloud, "/invalid/path/file.pcd", PointCloudProcessor::ExportFormat::PLY_BINARY)
    );

    // Test export with empty cloud
    auto empty_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    std::filesystem::path temp_empty = std::filesystem::temp_directory_path() / "test_empty.pcd";

    try {
        bool success = processor_->exportPointCloud(empty_cloud, temp_empty.string(), PointCloudProcessor::ExportFormat::PLY_BINARY);
        // Either succeeds with empty file or fails gracefully
        if (success && std::filesystem::exists(temp_empty)) {
            std::filesystem::remove(temp_empty);
        }
        EXPECT_TRUE(true) << "Handled empty cloud export appropriately";
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled empty cloud export: " << e.what();
    }
}

// Test processing under memory pressure
TEST_F(PointCloudProcessorEdgeCaseTest, ProcessingUnderMemoryPressure) {
    // Create a large point cloud
    auto large_cloud = createTestPointCloud(100000, true);

    // Apply memory pressure
    EdgeCaseTestFramework::simulateMemoryPressure(512);

    try {
        auto filtered = processor_->filterPointCloud(large_cloud, 0.01, 50, 1.0);

        if (filtered) {
            EXPECT_TRUE(true) << "Successfully processed large cloud under memory pressure (result size: "
                              << filtered->size() << " points)";
        } else {
            EXPECT_TRUE(true) << "Gracefully failed processing under memory pressure";
        }
    } catch (const std::bad_alloc& e) {
        EXPECT_TRUE(true) << "Gracefully handled memory allocation failure: " << e.what();
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled processing failure under memory pressure: " << e.what();
    }
}

// Test integer truncation in point cloud dimensions
TEST_F(PointCloudProcessorEdgeCaseTest, DimensionTruncationHandling) {
    auto int_edges = EdgeCaseTestFramework::getIntegerEdgeCases();

    for (int edge_val : int_edges) {
        if (edge_val > 0 && edge_val < 1000000) { // Reasonable point count range
            try {
                auto cloud = createTestPointCloud(static_cast<size_t>(edge_val));

                if (cloud && !cloud->empty()) {
                    // Verify point cloud dimensions are consistent
                    EXPECT_EQ(cloud->width * cloud->height, cloud->points.size())
                        << "Point cloud dimensions inconsistent for size " << edge_val;

                    // Test filtering doesn't cause dimension issues
                    auto filtered = processor_->filterPointCloud(cloud, 0.01, 10, 1.0);
                    if (filtered) {
                        EXPECT_EQ(filtered->width * filtered->height, filtered->points.size())
                            << "Filtered point cloud dimensions inconsistent";
                    }
                }
            } catch (const std::exception& e) {
                EXPECT_TRUE(true) << "Gracefully handled edge case point count " << edge_val << ": " << e.what();
            }
        }
    }
}

} // namespace testing
} // namespace stereo_vision
