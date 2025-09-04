#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <optional>
#include <numeric>
#include <cmath>

#include "ai/model_registry.hpp"
#include "ai/onnx_provider_manager.hpp"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct BenchmarkConfig {
    std::string model_name;
    std::vector<std::string> providers;
    std::string input_image_path;
    std::string output_format = "csv";  // csv, json, console
    std::string output_file;
    int iterations = 10;
    int warmup_iterations = 3;
    bool validate_output = false;
    std::string golden_output_dir;
    double tolerance = 1e-5;
    bool verbose = false;

    // Model-specific settings
    std::vector<int64_t> input_shape;
    bool normalize_input = true;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
};

struct BenchmarkResult {
    std::string model_name;
    std::string provider;
    bool successful = false;
    std::string error_message;

    // Timing results
    double avg_inference_time_ms = 0.0;
    double min_inference_time_ms = 0.0;
    double max_inference_time_ms = 0.0;
    double std_dev_ms = 0.0;

    // Memory usage
    size_t peak_memory_mb = 0;

    // Model info
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    size_t input_size_mb = 0;
    size_t output_size_mb = 0;

    // Validation results
    bool output_validation_passed = false;
    double max_output_diff = 0.0;

    // System info
    std::string gpu_name;
    std::string driver_version;
    std::string timestamp;
};

class StereoVisionBenchmark {
public:
    StereoVisionBenchmark() : model_registry_(cv_stereo::ModelRegistry::instance()) {}

    bool run_benchmark(const BenchmarkConfig& config) {
        std::cout << "Starting benchmark for model: " << config.model_name << std::endl;

        // Load and validate model
        if (!load_model(config)) {
            std::cerr << "Failed to load model: " << config.model_name << std::endl;
            return false;
        }

        // Load input image
        cv::Mat input_image;
        if (!load_input_image(config, input_image)) {
            std::cerr << "Failed to load input image: " << config.input_image_path << std::endl;
            return false;
        }

        // Run benchmarks for each provider
        std::vector<BenchmarkResult> results;
        for (const auto& provider_name : config.providers) {
            auto provider = string_to_provider(provider_name);
            if (provider) {
                auto result = benchmark_provider(config, *provider, input_image);
                results.push_back(result);
            } else {
                std::cerr << "Unknown provider: " << provider_name << std::endl;
            }
        }

        // Save results
        if (!save_results(config, results)) {
            std::cerr << "Failed to save results" << std::endl;
            return false;
        }

        // Print summary
        print_summary(results);

        return true;
    }

private:
    cv_stereo::ModelRegistry& model_registry_;
    cv_stereo::ONNXProviderManager& onnx_manager_ = cv_stereo::ONNXProviderManager::instance();

    bool load_model(const BenchmarkConfig& config) {
        // First ensure the registry is loaded
        if (!model_registry_.list_models().empty() || load_model_registry()) {
            std::string error_msg;
            // Validate and potentially download the model
            if (!model_registry_.validate_model(config.model_name, error_msg)) {
                // Try downloading if validation fails
                if (!model_registry_.download_model(config.model_name, error_msg)) {
                    std::cerr << "Failed to download model: " << error_msg << std::endl;
                    return false;
                }
                // Validate again after download
                return model_registry_.validate_model(config.model_name, error_msg);
            }
            return true;
        }
        return false;
    }

    bool load_model_registry() {
        std::string error_msg;
        // Try common config file locations
        std::vector<std::string> config_paths = {
            "config/models.yaml",
            "../config/models.yaml",
            "/home/kevin/Projects/computer-vision/config/models.yaml"
        };

        for (const auto& path : config_paths) {
            if (fs::exists(path)) {
                if (model_registry_.load_from_yaml(path, error_msg)) {
                    return true;
                }
                std::cerr << "Failed to load config from " << path << ": " << error_msg << std::endl;
            }
        }

        std::cerr << "Could not find or load model configuration file" << std::endl;
        return false;
    }

    bool load_input_image(const BenchmarkConfig& config, cv::Mat& image) {
        if (config.input_image_path.empty()) {
            // Create synthetic test image
            image = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            return true;
        }

        image = cv::imread(config.input_image_path);
        return !image.empty();
    }

    std::optional<cv_stereo::ONNXProvider> string_to_provider(const std::string& provider_name) {
        if (provider_name == "cpu") return cv_stereo::ONNXProvider::CPU;
        if (provider_name == "cuda") return cv_stereo::ONNXProvider::CUDA;
        if (provider_name == "tensorrt") return cv_stereo::ONNXProvider::TensorRT;
        if (provider_name == "directml") return cv_stereo::ONNXProvider::DirectML;
        if (provider_name == "coreml") return cv_stereo::ONNXProvider::CoreML;
        return std::nullopt;
    }

    BenchmarkResult benchmark_provider(const BenchmarkConfig& config,
                                      cv_stereo::ONNXProvider provider,
                                      const cv::Mat& input_image) {
        BenchmarkResult result;
        result.model_name = config.model_name;
        result.provider = onnx_manager_.provider_to_string(provider);
        result.timestamp = get_timestamp();

        try {
            // Configure ONNX session
            cv_stereo::ONNXSessionConfig session_config;
            session_config.preferred_provider = provider;
            session_config.fallback_providers.clear();  // No fallbacks for pure provider test

            // Set model-specific parameters
            if (!config.input_shape.empty()) {
                result.input_shape = config.input_shape;
            } else {
                // Use default shape based on model
                result.input_shape = {1, 3, 480, 640};  // Default for stereo models
            }

            // Create session
            auto model_spec = model_registry_.get_model(config.model_name);
            if (!model_spec) {
                result.error_message = "Model not found in registry";
                return result;
            }

            std::string error_msg;
            if (!onnx_manager_.create_session(model_spec->onnx_path, session_config, error_msg)) {
                result.error_message = error_msg;
                return result;
            }

            // Prepare input data
            auto input_data = cv_stereo::onnx_utils::mat_to_onnx_input(
                input_image, result.input_shape, config.normalize_input, config.mean, config.std);

            result.input_size_mb = (input_data.size() * sizeof(float)) / (1024 * 1024);

            // Warmup runs
            if (config.verbose) {
                std::cout << "Running " << config.warmup_iterations << " warmup iterations..." << std::endl;
            }

            for (int i = 0; i < config.warmup_iterations; ++i) {
                std::vector<float> output_data;
                std::vector<int64_t> output_shape;
                onnx_manager_.run_inference(input_data, result.input_shape,
                                           output_data, output_shape, error_msg);
            }

            // Benchmark runs
            if (config.verbose) {
                std::cout << "Running " << config.iterations << " benchmark iterations..." << std::endl;
            }

            std::vector<double> times;
            std::vector<float> final_output;
            std::vector<int64_t> final_output_shape;

            for (int i = 0; i < config.iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();

                std::vector<float> output_data;
                std::vector<int64_t> output_shape;

                if (!onnx_manager_.run_inference(input_data, result.input_shape,
                                                output_data, output_shape, error_msg)) {
                    result.error_message = error_msg;
                    return result;
                }

                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                times.push_back(time_ms);

                // Save output from first iteration for validation
                if (i == 0) {
                    final_output = output_data;
                    final_output_shape = output_shape;
                    result.output_shape = output_shape;
                    result.output_size_mb = (output_data.size() * sizeof(float)) / (1024 * 1024);
                }
            }

            // Calculate statistics
            if (!times.empty()) {
                result.avg_inference_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                result.min_inference_time_ms = *std::min_element(times.begin(), times.end());
                result.max_inference_time_ms = *std::max_element(times.begin(), times.end());

                // Calculate standard deviation
                double sum_sq_diff = 0.0;
                for (double time : times) {
                    double diff = time - result.avg_inference_time_ms;
                    sum_sq_diff += diff * diff;
                }
                result.std_dev_ms = std::sqrt(sum_sq_diff / times.size());

                result.successful = true;

                // Validate output if requested
                if (config.validate_output && !config.golden_output_dir.empty()) {
                    result.output_validation_passed = validate_output(
                        config, final_output, final_output_shape, result.max_output_diff);
                }
            }

        } catch (const std::exception& e) {
            result.error_message = e.what();
        }

        return result;
    }

    bool validate_output(const BenchmarkConfig& config,
                        const std::vector<float>& output,
                        const std::vector<int64_t>& output_shape,
                        double& max_diff) {
        // Load golden output
        std::string golden_file = config.golden_output_dir + "/" +
                                 config.model_name + "_golden_output.bin";

        if (!fs::exists(golden_file)) {
            std::cerr << "Golden output file not found: " << golden_file << std::endl;
            return false;
        }

        std::ifstream file(golden_file, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open golden output file" << std::endl;
            return false;
        }

        // Read golden output
        std::vector<float> golden_output;
        float value;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(float))) {
            golden_output.push_back(value);
        }

        if (golden_output.size() != output.size()) {
            std::cerr << "Output size mismatch: expected " << golden_output.size()
                     << ", got " << output.size() << std::endl;
            return false;
        }

        // Compare outputs
        max_diff = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            double diff = std::abs(output[i] - golden_output[i]);
            max_diff = std::max(max_diff, diff);
        }

        return max_diff <= config.tolerance;
    }

    bool save_results(const BenchmarkConfig& config,
                     const std::vector<BenchmarkResult>& results) {
        if (config.output_format == "json") {
            return save_results_json(config, results);
        } else if (config.output_format == "csv") {
            return save_results_csv(config, results);
        }
        return true;  // Console output doesn't need saving
    }

    bool save_results_csv(const BenchmarkConfig& config,
                         const std::vector<BenchmarkResult>& results) {
        std::string filename = config.output_file;
        if (filename.empty()) {
            filename = "reports/benchmarks/benchmark_" + config.model_name + "_" +
                      get_timestamp() + ".csv";
        }

        // Create directory if it doesn't exist
        fs::create_directories(fs::path(filename).parent_path());

        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to create output file: " << filename << std::endl;
            return false;
        }

        // Write header
        file << "model_name,provider,successful,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,"
             << "input_size_mb,output_size_mb,validation_passed,max_output_diff,error_message,timestamp\n";

        // Write results
        for (const auto& result : results) {
            file << result.model_name << ","
                 << result.provider << ","
                 << (result.successful ? "true" : "false") << ","
                 << result.avg_inference_time_ms << ","
                 << result.min_inference_time_ms << ","
                 << result.max_inference_time_ms << ","
                 << result.std_dev_ms << ","
                 << result.input_size_mb << ","
                 << result.output_size_mb << ","
                 << (result.output_validation_passed ? "true" : "false") << ","
                 << result.max_output_diff << ","
                 << "\"" << result.error_message << "\","
                 << result.timestamp << "\n";
        }

        std::cout << "Results saved to: " << filename << std::endl;
        return true;
    }

    bool save_results_json(const BenchmarkConfig& config,
                          const std::vector<BenchmarkResult>& results) {
        std::string filename = config.output_file;
        if (filename.empty()) {
            filename = "reports/benchmarks/benchmark_" + config.model_name + "_" +
                      get_timestamp() + ".json";
        }

        // Create directory if it doesn't exist
        fs::create_directories(fs::path(filename).parent_path());

        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to create output file: " << filename << std::endl;
            return false;
        }

        file << "{\n";
        file << "  \"benchmark_config\": {\n";
        file << "    \"model_name\": \"" << config.model_name << "\",\n";
        file << "    \"iterations\": " << config.iterations << ",\n";
        file << "    \"warmup_iterations\": " << config.warmup_iterations << "\n";
        file << "  },\n";
        file << "  \"results\": [\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            file << "    {\n";
            file << "      \"model_name\": \"" << result.model_name << "\",\n";
            file << "      \"provider\": \"" << result.provider << "\",\n";
            file << "      \"successful\": " << (result.successful ? "true" : "false") << ",\n";
            file << "      \"timing\": {\n";
            file << "        \"avg_inference_time_ms\": " << result.avg_inference_time_ms << ",\n";
            file << "        \"min_inference_time_ms\": " << result.min_inference_time_ms << ",\n";
            file << "        \"max_inference_time_ms\": " << result.max_inference_time_ms << ",\n";
            file << "        \"std_dev_ms\": " << result.std_dev_ms << "\n";
            file << "      },\n";
            file << "      \"memory\": {\n";
            file << "        \"input_size_mb\": " << result.input_size_mb << ",\n";
            file << "        \"output_size_mb\": " << result.output_size_mb << "\n";
            file << "      },\n";
            file << "      \"validation\": {\n";
            file << "        \"passed\": " << (result.output_validation_passed ? "true" : "false") << ",\n";
            file << "        \"max_output_diff\": " << result.max_output_diff << "\n";
            file << "      },\n";
            file << "      \"error_message\": \"" << result.error_message << "\",\n";
            file << "      \"timestamp\": \"" << result.timestamp << "\"\n";
            file << "    }";
            if (i < results.size() - 1) file << ",";
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        std::cout << "Results saved to: " << filename << std::endl;
        return true;
    }

    void print_summary(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Benchmark Summary ===\n";
        std::cout << std::left << std::setw(15) << "Provider"
                  << std::setw(12) << "Status"
                  << std::setw(15) << "Avg Time (ms)"
                  << std::setw(15) << "Min Time (ms)"
                  << std::setw(15) << "Max Time (ms)"
                  << std::setw(12) << "Std Dev"
                  << "Error\n";
        std::cout << std::string(90, '-') << "\n";

        for (const auto& result : results) {
            std::cout << std::left << std::setw(15) << result.provider;

            if (result.successful) {
                std::cout << std::setw(12) << "SUCCESS"
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.avg_inference_time_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.min_inference_time_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.max_inference_time_ms
                          << std::setw(12) << std::fixed << std::setprecision(2) << result.std_dev_ms;
            } else {
                std::cout << std::setw(12) << "FAILED"
                          << std::setw(15) << "-"
                          << std::setw(15) << "-"
                          << std::setw(15) << "-"
                          << std::setw(12) << "-"
                          << result.error_message;
            }
            std::cout << "\n";
        }

        // Find fastest provider
        auto fastest = std::min_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                if (!a.successful) return false;
                if (!b.successful) return true;
                return a.avg_inference_time_ms < b.avg_inference_time_ms;
            });

        if (fastest != results.end() && fastest->successful) {
            std::cout << "\nFastest provider: " << fastest->provider
                      << " (" << std::fixed << std::setprecision(2)
                      << fastest->avg_inference_time_ms << " ms)\n";
        }
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <name>        Model name to benchmark (required)\n";
    std::cout << "  -p, --providers <list>    Comma-separated list of providers (default: cpu,cuda,tensorrt)\n";
    std::cout << "  -i, --input <path>        Input image path (default: synthetic image)\n";
    std::cout << "  -o, --output <path>       Output file path\n";
    std::cout << "  -f, --format <format>     Output format: csv, json, console (default: csv)\n";
    std::cout << "  -n, --iterations <num>    Number of iterations (default: 10)\n";
    std::cout << "  -w, --warmup <num>        Number of warmup iterations (default: 3)\n";
    std::cout << "  -v, --validate            Enable output validation\n";
    std::cout << "  -g, --golden <dir>        Golden output directory for validation\n";
    std::cout << "  -t, --tolerance <val>     Validation tolerance (default: 1e-5)\n";
    std::cout << "  --verbose                 Verbose output\n";
    std::cout << "  -h, --help                Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -m hitnet -p cpu,cuda\n";
    std::cout << "  " << program_name << " -m raft_stereo -p tensorrt -n 100 -f json\n";
    std::cout << "  " << program_name << " -m crestereo -v -g data/golden_outputs\n";
}

BenchmarkConfig parse_arguments(int argc, char* argv[]) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                config.model_name = argv[++i];
            }
        } else if (arg == "-p" || arg == "--providers") {
            if (i + 1 < argc) {
                std::string providers_str = argv[++i];
                std::stringstream ss(providers_str);
                std::string provider;
                while (std::getline(ss, provider, ',')) {
                    config.providers.push_back(provider);
                }
            }
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.input_image_path = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_file = argv[++i];
            }
        } else if (arg == "-f" || arg == "--format") {
            if (i + 1 < argc) {
                config.output_format = argv[++i];
            }
        } else if (arg == "-n" || arg == "--iterations") {
            if (i + 1 < argc) {
                config.iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) {
                config.warmup_iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--validate") {
            config.validate_output = true;
        } else if (arg == "-g" || arg == "--golden") {
            if (i + 1 < argc) {
                config.golden_output_dir = argv[++i];
            }
        } else if (arg == "-t" || arg == "--tolerance") {
            if (i + 1 < argc) {
                config.tolerance = std::stod(argv[++i]);
            }
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }

    // Set defaults
    if (config.providers.empty()) {
        config.providers = {"cpu", "cuda", "tensorrt"};
    }

    return config;
}

int main(int argc, char* argv[]) {
    auto config = parse_arguments(argc, argv);

    if (config.model_name.empty()) {
        std::cerr << "Error: Model name is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // Validate output format
    if (config.output_format != "csv" && config.output_format != "json" &&
        config.output_format != "console") {
        std::cerr << "Error: Invalid output format. Use csv, json, or console\n";
        return 1;
    }

    StereoVisionBenchmark benchmark;
    if (!benchmark.run_benchmark(config)) {
        std::cerr << "Benchmark failed\n";
        return 1;
    }

    return 0;
}
