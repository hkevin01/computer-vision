#include "ai/model_registry.hpp"
#include <yaml-cpp/yaml.h>
#include <openssl/sha.h>
#include <curl/curl.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace cv_stereo {

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry instance;
    return instance;
}

bool ModelRegistry::load_from_yaml(const std::string& path, std::string& error_msg) {
    try {
        if (!std::filesystem::exists(path)) {
            error_msg = "YAML file not found: " + path;
            return false;
        }

        YAML::Node config = YAML::LoadFile(path);

        // Load global configuration
        if (config["global"]) {
            auto global = config["global"];
            if (global["cache_dir"]) global_config_.cache_dir = global["cache_dir"].as<std::string>();
            if (global["download_timeout"]) global_config_.download_timeout = global["download_timeout"].as<int>();
            if (global["verify_checksums"]) global_config_.verify_checksums = global["verify_checksums"].as<bool>();
            if (global["fallback_provider"]) global_config_.fallback_provider = global["fallback_provider"].as<std::string>();
        }

        // Load models
        if (!config["models"] || !config["models"].IsSequence()) {
            error_msg = "Invalid YAML: 'models' section must be a sequence";
            return false;
        }

        models_.clear();

        for (const auto& model_node : config["models"]) {
            ModelSpec spec;

            // Required fields
            if (!model_node["name"]) {
                error_msg = "Model missing required 'name' field";
                return false;
            }
            spec.name = model_node["name"].as<std::string>();

            if (!model_node["onnx_path"]) {
                error_msg = "Model '" + spec.name + "' missing required 'onnx_path' field";
                return false;
            }
            spec.onnx_path = model_node["onnx_path"].as<std::string>();

            // Optional fields with defaults
            spec.download_url = model_node["download_url"] ? model_node["download_url"].as<std::string>() : "";
            spec.checksum_sha256 = model_node["checksum_sha256"] ? model_node["checksum_sha256"].as<std::string>() : "";
            spec.input_channels = model_node["input_channels"] ? model_node["input_channels"].as<int>() : 3;
            spec.output_format = model_node["output_format"] ? model_node["output_format"].as<std::string>() : "disparity";
            spec.precision = model_node["precision"] ? model_node["precision"].as<std::string>() : "fp32";

            // Input dimensions
            if (model_node["input_hw"] && model_node["input_hw"].IsSequence()) {
                for (const auto& dim : model_node["input_hw"]) {
                    spec.input_hw.push_back(dim.as<int>());
                }
            }

            // Provider preferences
            if (model_node["provider_preference"] && model_node["provider_preference"].IsSequence()) {
                for (const auto& provider : model_node["provider_preference"]) {
                    spec.provider_preference.push_back(provider.as<std::string>());
                }
            } else {
                spec.provider_preference = {"cpu"};  // default fallback
            }

            // Preprocessing configuration
            if (model_node["preprocessing"]) {
                auto prep = model_node["preprocessing"];
                spec.preprocessing.normalize = prep["normalize"] ? prep["normalize"].as<bool>() : true;
                spec.preprocessing.color_order = prep["color_order"] ? prep["color_order"].as<std::string>() : "BGR";

                if (prep["mean"] && prep["mean"].IsSequence()) {
                    for (const auto& val : prep["mean"]) {
                        spec.preprocessing.mean.push_back(val.as<float>());
                    }
                }

                if (prep["std"] && prep["std"].IsSequence()) {
                    for (const auto& val : prep["std"]) {
                        spec.preprocessing.std.push_back(val.as<float>());
                    }
                }
            }

            models_[spec.name] = spec;
        }

        loaded_ = true;
        return true;

    } catch (const YAML::Exception& e) {
        error_msg = "YAML parsing error: " + std::string(e.what());
        return false;
    } catch (const std::exception& e) {
        error_msg = "Error loading model registry: " + std::string(e.what());
        return false;
    }
}

std::optional<ModelSpec> ModelRegistry::get_model(const std::string& name) const {
    auto it = models_.find(name);
    if (it != models_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& [name, spec] : models_) {
        names.push_back(name);
    }
    return names;
}

bool ModelRegistry::validate_model(const std::string& name, std::string& error_msg) const {
    auto model_opt = get_model(name);
    if (!model_opt) {
        error_msg = "Model not found: " + name;
        return false;
    }

    const auto& model = *model_opt;

    // Check if file exists
    if (!std::filesystem::exists(model.onnx_path)) {
        error_msg = "Model file not found: " + model.onnx_path;
        return false;
    }

    // Verify checksum if enabled and available
    if (global_config_.verify_checksums && !model.checksum_sha256.empty() &&
        model.checksum_sha256 != "TODO_REPLACE_WITH_ACTUAL_SHA256_HASH") {

        if (!verify_file_checksum(model.onnx_path, model.checksum_sha256)) {
            error_msg = "Checksum verification failed for: " + model.onnx_path;
            return false;
        }
    }

    return true;
}

bool ModelRegistry::download_model(const std::string& name, std::string& error_msg) const {
    auto model_opt = get_model(name);
    if (!model_opt) {
        error_msg = "Model not found: " + name;
        return false;
    }

    const auto& model = *model_opt;

    if (model.download_url.empty()) {
        error_msg = "No download URL available for model: " + name;
        return false;
    }

    // Create directory if needed
    std::filesystem::path model_path(model.onnx_path);
    std::filesystem::create_directories(model_path.parent_path());

    // Download file
    if (!download_file(model.download_url, model.onnx_path, global_config_.download_timeout)) {
        error_msg = "Failed to download model from: " + model.download_url;
        return false;
    }

    // Verify checksum after download
    std::string validation_error;
    if (!validate_model(name, validation_error)) {
        error_msg = "Downloaded model validation failed: " + validation_error;
        std::filesystem::remove(model.onnx_path);  // Clean up invalid download
        return false;
    }

    return true;
}

void ModelRegistry::clear() {
    models_.clear();
    loaded_ = false;
    global_config_ = GlobalConfig{};
}

bool ModelRegistry::verify_file_checksum(const std::string& filepath, const std::string& expected_sha256) const {
    std::string actual_sha256 = calculate_sha256(filepath);
    return actual_sha256 == expected_sha256;
}

std::string ModelRegistry::calculate_sha256(const std::string& filepath) const {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return "";
    }

    SHA256_CTX sha256_ctx;
    SHA256_Init(&sha256_ctx);

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        SHA256_Update(&sha256_ctx, buffer, file.gcount());
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256_ctx);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return ss.str();
}

// CURL write callback
static size_t write_data(void* ptr, size_t size, size_t nmemb, std::ofstream* stream) {
    stream->write(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

bool ModelRegistry::download_file(const std::string& url, const std::string& output_path, int timeout_seconds) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    std::ofstream output_file(output_path, std::ios::binary);
    if (!output_file) {
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_file);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    CURLcode res = curl_easy_perform(curl);

    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    curl_easy_cleanup(curl);
    output_file.close();

    if (res != CURLE_OK || response_code != 200) {
        std::filesystem::remove(output_path);  // Clean up partial download
        return false;
    }

    return true;
}

} // namespace cv_stereo
