#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ai/tensorrt_cache.hpp"
#include <filesystem>

using namespace cv_stereo;

class TensorRTCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up temporary cache directory
        temp_cache_dir_ = std::filesystem::temp_directory_path() / "cv_stereo_trt_test";
        std::filesystem::create_directories(temp_cache_dir_);

        TensorRTEngineCache::instance().set_cache_directory(temp_cache_dir_.string());

        // Create test cache key
        test_key_.model_name = "TestModel";
        test_key_.gpu_arch = "sm_75";
        test_key_.onnx_opset_version = 11;
        test_key_.precision = TensorRTPrecision::FP16;
        test_key_.input_shape = {1, 3, 256, 512};
    }

    void TearDown() override {
        std::filesystem::remove_all(temp_cache_dir_);
    }

    std::filesystem::path temp_cache_dir_;
    TensorRTCacheKey test_key_;
};

TEST_F(TensorRTCacheTest, CacheKeyToStringIsConsistent) {
    std::string key_str1 = test_key_.to_string();
    std::string key_str2 = test_key_.to_string();

    EXPECT_EQ(key_str1, key_str2);
    EXPECT_THAT(key_str1, testing::HasSubstr("TestModel"));
    EXPECT_THAT(key_str1, testing::HasSubstr("sm_75"));
    EXPECT_THAT(key_str1, testing::HasSubstr("fp16"));
    EXPECT_THAT(key_str1, testing::HasSubstr("shape_1_3_256_512"));
}

TEST_F(TensorRTCacheTest, CacheKeyEqualityWorks) {
    TensorRTCacheKey key1 = test_key_;
    TensorRTCacheKey key2 = test_key_;
    TensorRTCacheKey key3 = test_key_;
    key3.precision = TensorRTPrecision::FP32;  // Different precision

    EXPECT_TRUE(key1 == key2);
    EXPECT_FALSE(key1 == key3);
}

TEST_F(TensorRTCacheTest, CacheKeyHashingWorks) {
    TensorRTCacheKeyHash hasher;

    TensorRTCacheKey key1 = test_key_;
    TensorRTCacheKey key2 = test_key_;
    TensorRTCacheKey key3 = test_key_;
    key3.model_name = "DifferentModel";

    EXPECT_EQ(hasher(key1), hasher(key2));
    EXPECT_NE(hasher(key1), hasher(key3));
}

TEST_F(TensorRTCacheTest, HasCachedEngineReturnsFalseForNonexistent) {
    auto& cache = TensorRTEngineCache::instance();

    EXPECT_FALSE(cache.has_cached_engine(test_key_));
}

TEST_F(TensorRTCacheTest, CacheAndLoadEngineWorks) {
    auto& cache = TensorRTEngineCache::instance();

    // Cache some dummy engine data
    std::string engine_data = "dummy_tensorrt_engine_data_12345";
    bool cache_result = cache.cache_engine(test_key_, engine_data);
    EXPECT_TRUE(cache_result);

    // Check if cached
    EXPECT_TRUE(cache.has_cached_engine(test_key_));

    // Load cached data
    auto loaded_data = cache.load_cached_engine(test_key_);
    EXPECT_FALSE(loaded_data.empty());

    std::string loaded_string(loaded_data.begin(), loaded_data.end());
    EXPECT_EQ(loaded_string, engine_data);
}

TEST_F(TensorRTCacheTest, LoadNonexistentEngineReturnsEmpty) {
    auto& cache = TensorRTEngineCache::instance();

    auto loaded_data = cache.load_cached_engine(test_key_);
    EXPECT_TRUE(loaded_data.empty());
}

TEST_F(TensorRTCacheTest, ClearCacheWorks) {
    auto& cache = TensorRTEngineCache::instance();

    // Cache some data
    std::string engine_data = "test_data";
    cache.cache_engine(test_key_, engine_data);
    EXPECT_TRUE(cache.has_cached_engine(test_key_));

    // Clear all cache
    cache.clear_cache();
    EXPECT_FALSE(cache.has_cached_engine(test_key_));
}

TEST_F(TensorRTCacheTest, ClearCacheByModelNameWorks) {
    auto& cache = TensorRTEngineCache::instance();

    // Cache data for two different models
    TensorRTCacheKey key1 = test_key_;
    key1.model_name = "Model1";

    TensorRTCacheKey key2 = test_key_;
    key2.model_name = "Model2";

    cache.cache_engine(key1, "data1");
    cache.cache_engine(key2, "data2");

    EXPECT_TRUE(cache.has_cached_engine(key1));
    EXPECT_TRUE(cache.has_cached_engine(key2));

    // Clear only Model1
    cache.clear_cache("Model1");

    EXPECT_FALSE(cache.has_cached_engine(key1));
    EXPECT_TRUE(cache.has_cached_engine(key2));
}

TEST_F(TensorRTCacheTest, GetCacheStatsWorks) {
    auto& cache = TensorRTEngineCache::instance();

    // Cache some engines
    TensorRTCacheKey key1 = test_key_;
    key1.model_name = "Model1";

    TensorRTCacheKey key2 = test_key_;
    key2.model_name = "Model1";
    key2.precision = TensorRTPrecision::FP32;

    TensorRTCacheKey key3 = test_key_;
    key3.model_name = "Model2";

    cache.cache_engine(key1, "data1");
    cache.cache_engine(key2, "data2");
    cache.cache_engine(key3, "data3");

    auto stats = cache.get_cache_stats();

    EXPECT_EQ(stats.total_engines, 3);
    EXPECT_GT(stats.total_size_bytes, 0);
    EXPECT_EQ(stats.engines_per_model["Model1"], 2);
    EXPECT_EQ(stats.engines_per_model["Model2"], 1);
}

// TensorRTSessionManager tests
class TensorRTSessionManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.model_name = "TestModel";
        config_.onnx_path = "test_model.onnx";
        config_.precision = TensorRTPrecision::FP16;
        config_.input_shape = {1, 3, 256, 512};
        config_.enable_cache = true;
        config_.fallback_to_onnx = true;
    }

    TensorRTSessionManager::SessionConfig config_;
};

TEST_F(TensorRTSessionManagerTest, CreateSessionSucceeds) {
    TensorRTSessionManager manager;
    std::string error_msg;

    bool result = manager.create_session(config_, error_msg);

    // Should succeed even without actual TensorRT (will fallback)
    EXPECT_TRUE(result);
}

TEST_F(TensorRTSessionManagerTest, CreateSessionWithoutFallbackCanFail) {
    config_.fallback_to_onnx = false;

    TensorRTSessionManager manager;
    std::string error_msg;

    bool result = manager.create_session(config_, error_msg);

    // May fail if TensorRT is not available
    if (!result) {
        EXPECT_THAT(error_msg, testing::HasSubstr("TensorRT not available"));
    }
}

TEST_F(TensorRTSessionManagerTest, RunInferenceWithValidSession) {
    TensorRTSessionManager manager;
    std::string error_msg;

    bool create_result = manager.create_session(config_, error_msg);
    ASSERT_TRUE(create_result);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output_data;

    bool inference_result = manager.run_inference(input_data, output_data, error_msg);

    EXPECT_TRUE(inference_result);
    EXPECT_FALSE(output_data.empty());
}

TEST_F(TensorRTSessionManagerTest, IsTensorRTAvailableReturnsBoolean) {
    // This test just verifies the function doesn't crash
    bool available = TensorRTSessionManager::is_tensorrt_available();

    // Result depends on system configuration, just check it's callable
    (void)available;  // Suppress unused variable warning
    SUCCEED();
}
