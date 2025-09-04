#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ai/model_registry.hpp"
#include <fstream>
#include <filesystem>
#include <tempfile>

using namespace cv_stereo;
using ::testing::_;
using ::testing::Return;

class ModelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory
        temp_dir_ = std::filesystem::temp_directory_path() / "cv_stereo_test";
        std::filesystem::create_directories(temp_dir_);

        // Create test YAML file
        test_yaml_path_ = temp_dir_ / "test_models.yaml";
        create_test_yaml();

        // Clear registry for clean test
        ModelRegistry::instance().clear();
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all(temp_dir_);
    }

    void create_test_yaml() {
        std::ofstream yaml_file(test_yaml_path_);
        yaml_file << R"yaml(
models:
  - name: TestModel1
    onnx_path: data/models/test1.onnx
    download_url: "https://example.com/test1.onnx"
    checksum_sha256: "abc123def456"
    input_hw: [256, 512]
    input_channels: 3
    output_format: "disparity"
    provider_preference: ["tensorrt", "cuda", "cpu"]
    precision: "fp16"
    preprocessing:
      normalize: true
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      color_order: "BGR"

  - name: TestModel2
    onnx_path: data/models/test2.onnx
    checksum_sha256: "def456ghi789"
    input_hw: [384, 640]
    input_channels: 6
    provider_preference: ["cuda", "cpu"]
    precision: "fp32"

global:
  cache_dir: "data/models/cache"
  download_timeout: 300
  verify_checksums: true
  fallback_provider: "cpu"
)yaml";
    }

    void create_test_model_file(const std::string& path, const std::string& content = "dummy model") {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream file(path);
        file << content;
    }

    std::filesystem::path temp_dir_;
    std::filesystem::path test_yaml_path_;
};

TEST_F(ModelRegistryTest, LoadValidYamlSucceeds) {
    std::string error_msg;
    bool result = ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    EXPECT_TRUE(result) << "Error: " << error_msg;
    EXPECT_TRUE(error_msg.empty());
}

TEST_F(ModelRegistryTest, LoadNonexistentYamlFails) {
    std::string error_msg;
    bool result = ModelRegistry::instance().load_from_yaml("/nonexistent/path.yaml", error_msg);

    EXPECT_FALSE(result);
    EXPECT_THAT(error_msg, testing::HasSubstr("not found"));
}

TEST_F(ModelRegistryTest, GetExistingModelReturnsSpec) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    auto model_opt = ModelRegistry::instance().get_model("TestModel1");

    ASSERT_TRUE(model_opt.has_value());
    const auto& model = model_opt.value();

    EXPECT_EQ(model.name, "TestModel1");
    EXPECT_EQ(model.onnx_path, "data/models/test1.onnx");
    EXPECT_EQ(model.checksum_sha256, "abc123def456");
    EXPECT_EQ(model.input_hw.size(), 2);
    EXPECT_EQ(model.input_hw[0], 256);
    EXPECT_EQ(model.input_hw[1], 512);
    EXPECT_EQ(model.input_channels, 3);
    EXPECT_EQ(model.precision, "fp16");
    EXPECT_TRUE(model.preprocessing.normalize);
    EXPECT_EQ(model.preprocessing.mean.size(), 3);
    EXPECT_FLOAT_EQ(model.preprocessing.mean[0], 0.485f);
}

TEST_F(ModelRegistryTest, GetNonexistentModelReturnsNullopt) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    auto model_opt = ModelRegistry::instance().get_model("NonexistentModel");

    EXPECT_FALSE(model_opt.has_value());
}

TEST_F(ModelRegistryTest, ListModelsReturnsAllNames) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    auto model_names = ModelRegistry::instance().list_models();

    EXPECT_EQ(model_names.size(), 2);
    EXPECT_THAT(model_names, testing::UnorderedElementsAre("TestModel1", "TestModel2"));
}

TEST_F(ModelRegistryTest, ValidateModelWithExistingFileSucceeds) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    // Create a dummy model file
    create_test_model_file("data/models/test1.onnx");

    bool result = ModelRegistry::instance().validate_model("TestModel1", error_msg);

    // Should succeed since we're not doing checksum verification with real checksums
    EXPECT_TRUE(result) << "Error: " << error_msg;
}

TEST_F(ModelRegistryTest, ValidateModelWithMissingFileFails) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    bool result = ModelRegistry::instance().validate_model("TestModel1", error_msg);

    EXPECT_FALSE(result);
    EXPECT_THAT(error_msg, testing::HasSubstr("not found"));
}

TEST_F(ModelRegistryTest, GlobalConfigurationLoadsCorrectly) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    const auto& global_config = ModelRegistry::instance().get_global_config();

    EXPECT_EQ(global_config.cache_dir, "data/models/cache");
    EXPECT_EQ(global_config.download_timeout, 300);
    EXPECT_TRUE(global_config.verify_checksums);
    EXPECT_EQ(global_config.fallback_provider, "cpu");
}

TEST_F(ModelRegistryTest, ClearResetsRegistry) {
    std::string error_msg;
    ModelRegistry::instance().load_from_yaml(test_yaml_path_.string(), error_msg);

    // Verify models are loaded
    EXPECT_EQ(ModelRegistry::instance().list_models().size(), 2);

    // Clear and verify
    ModelRegistry::instance().clear();
    EXPECT_EQ(ModelRegistry::instance().list_models().size(), 0);
    EXPECT_FALSE(ModelRegistry::instance().get_model("TestModel1").has_value());
}

// Test for malformed YAML
TEST_F(ModelRegistryTest, LoadMalformedYamlFails) {
    std::filesystem::path bad_yaml = temp_dir_ / "bad.yaml";
    std::ofstream bad_file(bad_yaml);
    bad_file << "models:\n  - name: BadModel\n    invalid_yaml: [\n";  // Unclosed bracket
    bad_file.close();

    std::string error_msg;
    bool result = ModelRegistry::instance().load_from_yaml(bad_yaml.string(), error_msg);

    EXPECT_FALSE(result);
    EXPECT_THAT(error_msg, testing::HasSubstr("YAML parsing error"));
}

// Test for missing required fields
TEST_F(ModelRegistryTest, LoadYamlWithMissingRequiredFieldsFails) {
    std::filesystem::path incomplete_yaml = temp_dir_ / "incomplete.yaml";
    std::ofstream incomplete_file(incomplete_yaml);
    incomplete_file << R"yaml(
models:
  - name: IncompleteModel
    # Missing onnx_path
    input_hw: [256, 512]
)yaml";
    incomplete_file.close();

    std::string error_msg;
    bool result = ModelRegistry::instance().load_from_yaml(incomplete_yaml.string(), error_msg);

    EXPECT_FALSE(result);
    EXPECT_THAT(error_msg, testing::HasSubstr("missing required"));
}
