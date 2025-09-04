#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ai/onnx_provider_manager.hpp"
#include <opencv2/opencv.hpp>

using namespace cv_stereo;
using ::testing::HasSubstr;
using ::testing::Contains;
using ::testing::Not;
using ::testing::IsEmpty;

class ONNXProviderManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager_ = &ONNXProviderManager::instance();
    }

    void TearDown() override {
        manager_->clear_session();
    }

    ONNXProviderManager* manager_;
};

TEST_F(ONNXProviderManagerTest, SingletonInstance) {
    auto& instance1 = ONNXProviderManager::instance();
    auto& instance2 = ONNXProviderManager::instance();

    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(ONNXProviderManagerTest, ProviderStringConversion) {
    EXPECT_EQ(manager_->provider_to_string(ONNXProvider::CPU), "CPUExecutionProvider");
    EXPECT_EQ(manager_->provider_to_string(ONNXProvider::CUDA), "CUDAExecutionProvider");
    EXPECT_EQ(manager_->provider_to_string(ONNXProvider::TensorRT), "TensorrtExecutionProvider");
    EXPECT_EQ(manager_->provider_to_string(ONNXProvider::DirectML), "DmlExecutionProvider");
    EXPECT_EQ(manager_->provider_to_string(ONNXProvider::CoreML), "CoreMLExecutionProvider");

    EXPECT_EQ(manager_->string_to_provider("CPUExecutionProvider"), ONNXProvider::CPU);
    EXPECT_EQ(manager_->string_to_provider("CUDAExecutionProvider"), ONNXProvider::CUDA);
    EXPECT_EQ(manager_->string_to_provider("TensorrtExecutionProvider"), ONNXProvider::TensorRT);
    EXPECT_EQ(manager_->string_to_provider("DmlExecutionProvider"), ONNXProvider::DirectML);
    EXPECT_EQ(manager_->string_to_provider("CoreMLExecutionProvider"), ONNXProvider::CoreML);

    // Test unknown provider defaults to CPU
    EXPECT_EQ(manager_->string_to_provider("UnknownProvider"), ONNXProvider::CPU);
}

TEST_F(ONNXProviderManagerTest, AvailableProviders) {
    auto providers = manager_->get_available_providers();

    // CPU should always be available
    EXPECT_THAT(providers, Contains(ONNXProvider::CPU));
    EXPECT_FALSE(providers.empty());

    // Check that all returned providers are actually available
    for (auto provider : providers) {
        EXPECT_TRUE(manager_->is_provider_available(provider));
    }
}

TEST_F(ONNXProviderManagerTest, SessionConfigDefaults) {
    ONNXSessionConfig config;

    EXPECT_EQ(config.preferred_provider, ONNXProvider::CPU);
    EXPECT_THAT(config.fallback_providers, Contains(ONNXProvider::CPU));
    EXPECT_EQ(config.optimization_level, "basic");
    EXPECT_TRUE(config.enable_graph_optimization);
    EXPECT_TRUE(config.enable_cpu_mem_arena);
    EXPECT_TRUE(config.enable_memory_pattern);
    EXPECT_EQ(config.num_threads, 0);  // Auto-detect
    EXPECT_EQ(config.gpu_device_id, 0);
    EXPECT_EQ(config.gpu_mem_limit, 0);  // Unlimited
}

TEST_F(ONNXProviderManagerTest, InitialSessionState) {
    EXPECT_FALSE(manager_->has_active_session());
    EXPECT_EQ(manager_->get_active_provider(), ONNXProvider::CPU);
}

TEST_F(ONNXProviderManagerTest, CreateSessionWithoutModel) {
    ONNXSessionConfig config;
    std::string error_msg;

    bool result = manager_->create_session("nonexistent_model.onnx", config, error_msg);

    EXPECT_FALSE(result);
    EXPECT_FALSE(error_msg.empty());
}

TEST_F(ONNXProviderManagerTest, RunInferenceWithoutSession) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> input_shape = {1, 2, 2, 1};
    std::vector<float> output_data;
    std::vector<int64_t> output_shape;
    std::string error_msg;

    bool result = manager_->run_inference(input_data, input_shape,
                                         output_data, output_shape, error_msg);

    EXPECT_FALSE(result);
    EXPECT_THAT(error_msg, HasSubstr("No active session"));
}

TEST_F(ONNXProviderManagerTest, GetModelInfoWithoutModel) {
    ONNXModelInfo info;
    std::string error_msg;

    bool result = manager_->get_model_info("nonexistent_model.onnx", info, error_msg);

    EXPECT_FALSE(result);
    EXPECT_FALSE(error_msg.empty());
}

TEST_F(ONNXProviderManagerTest, ClearSession) {
    // Initially no session
    EXPECT_FALSE(manager_->has_active_session());

    // Clear should not crash
    manager_->clear_session();

    EXPECT_FALSE(manager_->has_active_session());
    EXPECT_EQ(manager_->get_active_provider(), ONNXProvider::CPU);
}

TEST_F(ONNXProviderManagerTest, InferenceStatsInitial) {
    const auto& stats = manager_->get_last_inference_stats();

    EXPECT_EQ(stats.preprocessing_time_ms, 0.0);
    EXPECT_EQ(stats.inference_time_ms, 0.0);
    EXPECT_EQ(stats.postprocessing_time_ms, 0.0);
    EXPECT_EQ(stats.total_time_ms, 0.0);
    EXPECT_EQ(stats.memory_usage_bytes, 0);
    EXPECT_TRUE(stats.provider_used.empty());
}

// Utility function tests
class ONNXUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test image
        test_image_ = cv::Mat::zeros(100, 200, CV_8UC3);
        test_image_.setTo(cv::Scalar(128, 64, 192));  // Some test values
    }

    cv::Mat test_image_;
};

TEST_F(ONNXUtilsTest, MatToONNXInputBasic) {
    std::vector<int64_t> target_shape = {1, 3, 50, 100};  // NCHW format

    auto result = onnx_utils::mat_to_onnx_input(test_image_, target_shape, false);

    // Should resize to 100x50 and convert to CHW format
    EXPECT_EQ(result.size(), 3 * 50 * 100);  // 3 channels * height * width

    // Check that values are in expected range (0-1 after normalization)
    for (float val : result) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

TEST_F(ONNXUtilsTest, MatToONNXInputWithNormalization) {
    std::vector<int64_t> target_shape = {1, 3, 50, 100};
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> std = {0.5f, 0.5f, 0.5f};

    auto result = onnx_utils::mat_to_onnx_input(test_image_, target_shape, true, mean, std);

    EXPECT_EQ(result.size(), 3 * 50 * 100);

    // With normalization, values can be negative
    bool has_negative = false;
    for (float val : result) {
        if (val < 0.0f) {
            has_negative = true;
            break;
        }
    }
    // Should have some normalized values
    EXPECT_TRUE(has_negative || std::any_of(result.begin(), result.end(),
                                           [](float v) { return v > 1.0f; }));
}

TEST_F(ONNXUtilsTest, ONNXOutputToMat) {
    std::vector<float> output_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> output_shape = {2, 3};  // 2x3 matrix

    cv::Mat result = onnx_utils::onnx_output_to_mat(output_data, output_shape, CV_32FC1);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 3);
    EXPECT_EQ(result.type(), CV_32FC1);

    // Check values
    EXPECT_FLOAT_EQ(result.at<float>(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result.at<float>(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result.at<float>(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(result.at<float>(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(result.at<float>(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(result.at<float>(1, 2), 6.0f);
}

TEST_F(ONNXUtilsTest, ONNXOutputToMatInvalidShape) {
    std::vector<float> output_data = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> output_shape = {3};  // 1D shape

    EXPECT_THROW(onnx_utils::onnx_output_to_mat(output_data, output_shape),
                 std::invalid_argument);
}

TEST_F(ONNXUtilsTest, AutoConfigureSession) {
    auto config = onnx_utils::auto_configure_session();

    // Should have reasonable defaults
    EXPECT_TRUE(config.preferred_provider == ONNXProvider::CPU ||
                config.preferred_provider == ONNXProvider::CUDA ||
                config.preferred_provider == ONNXProvider::TensorRT ||
                config.preferred_provider == ONNXProvider::DirectML ||
                config.preferred_provider == ONNXProvider::CoreML);

    EXPECT_GT(config.num_threads, 0);
    EXPECT_FALSE(config.fallback_providers.empty());
}

TEST_F(ONNXUtilsTest, BenchmarkProvidersWithoutModel) {
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    auto results = onnx_utils::benchmark_providers("nonexistent_model.onnx", input_shape, 1);

    EXPECT_FALSE(results.empty());

    // All results should be unsuccessful
    for (const auto& result : results) {
        EXPECT_FALSE(result.successful);
        EXPECT_FALSE(result.error_message.empty());
    }
}

// Test compile-time feature detection
TEST(ONNXFeatureTest, CompileTimeFeatures) {
#ifdef CV_WITH_ONNX
    // If ONNX is enabled, we should be able to create a manager
    auto& manager = ONNXProviderManager::instance();
    EXPECT_TRUE(true);  // Just test that compilation works
#else
    // If ONNX is disabled, test basic functionality still works
    ONNXSessionConfig config;
    EXPECT_EQ(config.preferred_provider, ONNXProvider::CPU);
#endif
}

// Test thread safety basics
TEST(ONNXThreadSafetyTest, ConcurrentSingletonAccess) {
    std::vector<std::thread> threads;
    std::vector<ONNXProviderManager*> managers;
    managers.resize(10);

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([i, &managers] {
            managers[i] = &ONNXProviderManager::instance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All should be the same instance
    for (int i = 1; i < 10; ++i) {
        EXPECT_EQ(managers[0], managers[i]);
    }
}
