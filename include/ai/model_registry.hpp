#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <memory>

namespace cv_stereo {

struct ModelSpec {
    std::string name;
    std::string onnx_path;
    std::string download_url;
    std::string checksum_sha256;
    std::vector<int> input_hw;  // [height, width]
    int input_channels;
    std::string output_format;
    std::vector<std::string> provider_preference;
    std::string precision;

    struct Preprocessing {
        bool normalize = true;
        std::vector<float> mean;
        std::vector<float> std;
        std::string color_order = "BGR";
    } preprocessing;
};

struct GlobalConfig {
    std::string cache_dir = "data/models/cache";
    int download_timeout = 300;
    bool verify_checksums = true;
    std::string fallback_provider = "cpu";
};

class ModelRegistry {
public:
    static ModelRegistry& instance();

    // Load models from YAML configuration
    bool load_from_yaml(const std::string& path, std::string& error_msg);

    // Get model specification by name
    std::optional<ModelSpec> get_model(const std::string& name) const;

    // List all available models
    std::vector<std::string> list_models() const;

    // Validate model file exists and checksum matches
    bool validate_model(const std::string& name, std::string& error_msg) const;

    // Download model if not present
    bool download_model(const std::string& name, std::string& error_msg) const;

    // Get global configuration
    const GlobalConfig& get_global_config() const { return global_config_; }

    // Clear registry (useful for testing)
    void clear();

private:
    ModelRegistry() = default;

    // Helper methods
    bool verify_file_checksum(const std::string& filepath, const std::string& expected_sha256) const;
    std::string calculate_sha256(const std::string& filepath) const;
    bool download_file(const std::string& url, const std::string& output_path, int timeout_seconds) const;

    std::unordered_map<std::string, ModelSpec> models_;
    GlobalConfig global_config_;
    bool loaded_ = false;
};

} // namespace cv_stereo
