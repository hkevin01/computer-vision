#include "ai/onnx_provider_manager.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cstring>

#ifdef CV_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#endif

namespace cv_stereo {

ONNXProviderManager& ONNXProviderManager::instance() {
    static ONNXProviderManager instance;
    return instance;
}

bool ONNXProviderManager::is_provider_available(ONNXProvider provider) const {
    auto available = get_available_providers();
    return std::find(available.begin(), available.end(), provider) != available.end();
}

std::vector<ONNXProvider> ONNXProviderManager::get_available_providers() const {
    if (!providers_cached_) {
        cached_available_providers_ = detect_available_providers();
        providers_cached_ = true;
    }
    return cached_available_providers_;
}

std::string ONNXProviderManager::provider_to_string(ONNXProvider provider) const {
    switch (provider) {
        case ONNXProvider::CPU:
            return "CPUExecutionProvider";
        case ONNXProvider::CUDA:
            return "CUDAExecutionProvider";
        case ONNXProvider::DirectML:
            return "DmlExecutionProvider";
        case ONNXProvider::CoreML:
            return "CoreMLExecutionProvider";
        case ONNXProvider::TensorRT:
            return "TensorrtExecutionProvider";
        default:
            return "CPUExecutionProvider";
    }
}

ONNXProvider ONNXProviderManager::string_to_provider(const std::string& provider_str) const {
    if (provider_str == "CUDAExecutionProvider") return ONNXProvider::CUDA;
    if (provider_str == "DmlExecutionProvider") return ONNXProvider::DirectML;
    if (provider_str == "CoreMLExecutionProvider") return ONNXProvider::CoreML;
    if (provider_str == "TensorrtExecutionProvider") return ONNXProvider::TensorRT;
    return ONNXProvider::CPU;  // Default fallback
}

bool ONNXProviderManager::create_session(const std::string& model_path,
                                         const ONNXSessionConfig& config,
                                         std::string& error_msg) {
#ifndef CV_WITH_ONNX
    error_msg = "ONNX Runtime support not compiled in";
    return false;
#else
    try {
        // Clear any existing session
        clear_session();

        // Initialize environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CVStereoONNX");

        // Configure session options
        session_options_ = Ort::SessionOptions{};

        // Set threading options
        if (config.num_threads > 0) {
            session_options_.SetIntraOpNumThreads(config.num_threads);
        }

        // Set memory options
        session_options_.SetGraphOptimizationLevel(
            config.enable_graph_optimization ?
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED :
            GraphOptimizationLevel::ORT_DISABLE_ALL);

        // Try providers in preference order
        std::vector<ONNXProvider> providers_to_try = {config.preferred_provider};
        providers_to_try.insert(providers_to_try.end(),
                               config.fallback_providers.begin(),
                               config.fallback_providers.end());

        for (auto provider : providers_to_try) {
            if (is_provider_available(provider)) {
                if (create_session_with_provider(model_path, provider, config, error_msg)) {
                    active_provider_ = provider;
                    current_config_ = config;

                    // Get model information
                    if (!get_model_info(model_path, model_info_, error_msg)) {
                        std::cerr << "Warning: Could not get model info: " << error_msg << std::endl;
                    }

                    return true;
                }
            }
        }

        error_msg = "Failed to create session with any available provider";
        return false;

    } catch (const std::exception& e) {
        error_msg = "Exception creating ONNX session: " + std::string(e.what());
        return false;
    }
#endif
}

bool ONNXProviderManager::create_session_with_provider(const std::string& model_path,
                                                      ONNXProvider provider,
                                                      const ONNXSessionConfig& config,
                                                      std::string& error_msg) {
#ifndef CV_WITH_ONNX
    error_msg = "ONNX Runtime support not compiled in";
    return false;
#else
    try {
        std::string provider_name = provider_to_string(provider);

        // Configure provider-specific options
        switch (provider) {
            case ONNXProvider::CUDA: {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = config.gpu_device_id;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.gpu_mem_limit = config.gpu_mem_limit;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;

                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                break;
            }
            case ONNXProvider::TensorRT: {
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = config.gpu_device_id;
                trt_options.trt_max_workspace_size = config.trt_max_workspace_size;
                trt_options.trt_fp16_enable = config.trt_enable_fp16 ? 1 : 0;
                trt_options.trt_int8_enable = config.trt_enable_int8 ? 1 : 0;

                session_options_.AppendExecutionProvider_TensorRT(trt_options);
                break;
            }
            case ONNXProvider::DirectML: {
#ifdef _WIN32
                session_options_.AppendExecutionProvider("DML");
#else
                error_msg = "DirectML provider only available on Windows";
                return false;
#endif
                break;
            }
            case ONNXProvider::CoreML: {
#ifdef __APPLE__
                session_options_.AppendExecutionProvider("CoreML");
#else
                error_msg = "CoreML provider only available on macOS";
                return false;
#endif
                break;
            }
            case ONNXProvider::CPU:
            default:
                // CPU provider is always available and added by default
                break;
        }

        // Create the session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);

        // Initialize memory info
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        return true;

    } catch (const std::exception& e) {
        error_msg = "Failed to create session with " + provider_to_string(provider) +
                   ": " + std::string(e.what());
        return false;
    }
#endif
}

bool ONNXProviderManager::run_inference(const std::vector<float>& input_data,
                                        const std::vector<int64_t>& input_shape,
                                        std::vector<float>& output_data,
                                        std::vector<int64_t>& output_shape,
                                        std::string& error_msg) {
#ifndef CV_WITH_ONNX
    error_msg = "ONNX Runtime support not compiled in";
    return false;
#else
    if (!session_) {
        error_msg = "No active session";
        return false;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

        // Prepare input/output names
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for (const auto& name : model_info_.input_names) {
            input_names.push_back(name.c_str());
        }
        for (const auto& name : model_info_.output_names) {
            output_names.push_back(name.c_str());
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                           input_names.data(),
                                           &input_tensor,
                                           1,
                                           output_names.data(),
                                           output_names.size());

        auto inference_end = std::chrono::high_resolution_clock::now();

        // Extract output data
        if (!output_tensors.empty()) {
            auto& output_tensor = output_tensors[0];
            auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();

            // Get output shape
            auto shape = tensor_info.GetShape();
            output_shape = std::vector<int64_t>(shape.begin(), shape.end());

            // Get output data
            float* output_ptr = output_tensor.GetTensorMutableData<float>();
            size_t output_size = tensor_info.GetElementCount();
            output_data = std::vector<float>(output_ptr, output_ptr + output_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        // Update statistics
        last_stats_.inference_time_ms = std::chrono::duration<double, std::milli>(
            inference_end - inference_start).count();
        last_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        last_stats_.provider_used = provider_to_string(active_provider_);

        return true;

    } catch (const std::exception& e) {
        error_msg = "Inference failed: " + std::string(e.what());
        return false;
    }
#endif
}

bool ONNXProviderManager::get_model_info(const std::string& model_path,
                                         ONNXModelInfo& info,
                                         std::string& error_msg) const {
#ifndef CV_WITH_ONNX
    error_msg = "ONNX Runtime support not compiled in";
    return false;
#else
    try {
        // Create temporary session for introspection
        Ort::Env temp_env(ORT_LOGGING_LEVEL_WARNING, "ModelInfo");
        Ort::SessionOptions temp_options;
        Ort::Session temp_session(temp_env, model_path.c_str(), temp_options);

        // Get input information
        size_t num_inputs = temp_session.GetInputCount();
        info.input_names.clear();
        info.input_shapes.clear();
        info.input_types.clear();

        for (size_t i = 0; i < num_inputs; ++i) {
            // Input name
            auto input_name = temp_session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            info.input_names.push_back(std::string(input_name.get()));

            // Input shape and type
            auto type_info = temp_session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            auto shape = tensor_info.GetShape();
            info.input_shapes.push_back(std::vector<int64_t>(shape.begin(), shape.end()));
            info.input_types.push_back(static_cast<int>(tensor_info.GetElementType()));
        }

        // Get output information
        size_t num_outputs = temp_session.GetOutputCount();
        info.output_names.clear();
        info.output_shapes.clear();
        info.output_types.clear();

        for (size_t i = 0; i < num_outputs; ++i) {
            // Output name
            auto output_name = temp_session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            info.output_names.push_back(std::string(output_name.get()));

            // Output shape and type
            auto type_info = temp_session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            auto shape = tensor_info.GetShape();
            info.output_shapes.push_back(std::vector<int64_t>(shape.begin(), shape.end()));
            info.output_types.push_back(static_cast<int>(tensor_info.GetElementType()));
        }

        return true;

    } catch (const std::exception& e) {
        error_msg = "Failed to get model info: " + std::string(e.what());
        return false;
    }
#endif
}

void ONNXProviderManager::clear_session() {
#ifdef CV_WITH_ONNX
    session_.reset();
    env_.reset();
    model_info_ = ONNXModelInfo{};
#endif
    active_provider_ = ONNXProvider::CPU;
    last_stats_ = InferenceStats{};
}

bool ONNXProviderManager::has_active_session() const {
#ifdef CV_WITH_ONNX
    return session_ != nullptr;
#else
    return false;
#endif
}

std::vector<ONNXProvider> ONNXProviderManager::detect_available_providers() const {
    std::vector<ONNXProvider> available;

#ifdef CV_WITH_ONNX
    try {
        // CPU is always available
        available.push_back(ONNXProvider::CPU);

        // Check for CUDA
        try {
            Ort::Env temp_env(ORT_LOGGING_LEVEL_ERROR, "ProviderCheck");
            Ort::SessionOptions temp_options;
            OrtCUDAProviderOptions cuda_options{};
            temp_options.AppendExecutionProvider_CUDA(cuda_options);
            available.push_back(ONNXProvider::CUDA);
        } catch (...) {
            // CUDA not available
        }

        // Check for TensorRT
        try {
            Ort::Env temp_env(ORT_LOGGING_LEVEL_ERROR, "ProviderCheck");
            Ort::SessionOptions temp_options;
            OrtTensorRTProviderOptions trt_options{};
            temp_options.AppendExecutionProvider_TensorRT(trt_options);
            available.push_back(ONNXProvider::TensorRT);
        } catch (...) {
            // TensorRT not available
        }

#ifdef _WIN32
        // Check for DirectML on Windows
        try {
            Ort::Env temp_env(ORT_LOGGING_LEVEL_ERROR, "ProviderCheck");
            Ort::SessionOptions temp_options;
            temp_options.AppendExecutionProvider("DML");
            available.push_back(ONNXProvider::DirectML);
        } catch (...) {
            // DirectML not available
        }
#endif

#ifdef __APPLE__
        // Check for CoreML on macOS
        try {
            Ort::Env temp_env(ORT_LOGGING_LEVEL_ERROR, "ProviderCheck");
            Ort::SessionOptions temp_options;
            temp_options.AppendExecutionProvider("CoreML");
            available.push_back(ONNXProvider::CoreML);
        } catch (...) {
            // CoreML not available
        }
#endif

    } catch (...) {
        // If any error occurs, just return CPU
        available = {ONNXProvider::CPU};
    }
#endif

    return available;
}

// Utility functions implementation
namespace onnx_utils {

std::vector<float> mat_to_onnx_input(const cv::Mat& image,
                                    const std::vector<int64_t>& target_shape,
                                    bool normalize,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& std) {
    cv::Mat processed = image.clone();

    // Resize if needed
    if (target_shape.size() >= 2) {
        int target_height = static_cast<int>(target_shape[target_shape.size()-2]);
        int target_width = static_cast<int>(target_shape[target_shape.size()-1]);

        if (processed.rows != target_height || processed.cols != target_width) {
            cv::resize(processed, processed, cv::Size(target_width, target_height));
        }
    }

    // Convert to float
    if (processed.type() != CV_32FC3) {
        processed.convertTo(processed, CV_32FC3, 1.0/255.0);
    }

    std::vector<float> result;

    if (normalize && mean.size() == 3 && std.size() == 3) {
        // Normalize with mean and std
        std::vector<cv::Mat> channels;
        cv::split(processed, channels);

        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - mean[c]) / std[c];
        }

        // Convert to CHW format
        for (int c = 0; c < 3; ++c) {
            cv::Mat& channel = channels[c];
            result.insert(result.end(),
                         reinterpret_cast<float*>(channel.data),
                         reinterpret_cast<float*>(channel.data) + channel.total());
        }
    } else {
        // Just convert to CHW format without normalization
        std::vector<cv::Mat> channels;
        cv::split(processed, channels);

        for (int c = 0; c < 3; ++c) {
            cv::Mat& channel = channels[c];
            result.insert(result.end(),
                         reinterpret_cast<float*>(channel.data),
                         reinterpret_cast<float*>(channel.data) + channel.total());
        }
    }

    return result;
}

cv::Mat onnx_output_to_mat(const std::vector<float>& output_data,
                          const std::vector<int64_t>& output_shape,
                          int cv_type) {
    if (output_shape.size() < 2) {
        throw std::invalid_argument("Output shape must have at least 2 dimensions");
    }

    int height = static_cast<int>(output_shape[output_shape.size()-2]);
    int width = static_cast<int>(output_shape[output_shape.size()-1]);

    cv::Mat result(height, width, cv_type);
    std::memcpy(result.data, output_data.data(), output_data.size() * sizeof(float));

    return result;
}

ONNXSessionConfig auto_configure_session() {
    ONNXSessionConfig config;

    auto& manager = ONNXProviderManager::instance();
    auto available = manager.get_available_providers();

    // Set preferred provider based on availability
    if (std::find(available.begin(), available.end(), ONNXProvider::TensorRT) != available.end()) {
        config.preferred_provider = ONNXProvider::TensorRT;
        config.trt_enable_fp16 = true;  // Enable FP16 for better performance
    } else if (std::find(available.begin(), available.end(), ONNXProvider::CUDA) != available.end()) {
        config.preferred_provider = ONNXProvider::CUDA;
    } else if (std::find(available.begin(), available.end(), ONNXProvider::DirectML) != available.end()) {
        config.preferred_provider = ONNXProvider::DirectML;
    } else if (std::find(available.begin(), available.end(), ONNXProvider::CoreML) != available.end()) {
        config.preferred_provider = ONNXProvider::CoreML;
    } else {
        config.preferred_provider = ONNXProvider::CPU;
    }

    // Set up fallback chain
    config.fallback_providers.clear();
    for (auto provider : available) {
        if (provider != config.preferred_provider) {
            config.fallback_providers.push_back(provider);
        }
    }

    // Auto-detect optimal thread count for CPU
    config.num_threads = std::thread::hardware_concurrency();

    return config;
}

std::vector<ProviderBenchmark> benchmark_providers(
    const std::string& model_path,
    const std::vector<int64_t>& input_shape,
    int num_iterations) {

    std::vector<ProviderBenchmark> results;
    auto& manager = ONNXProviderManager::instance();
    auto available = manager.get_available_providers();

    // Create dummy input data
    size_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }
    std::vector<float> input_data(input_size, 0.5f);

    for (auto provider : available) {
        ProviderBenchmark benchmark;
        benchmark.provider = provider;
        benchmark.successful = false;

        try {
            ONNXSessionConfig config;
            config.preferred_provider = provider;
            config.fallback_providers.clear();  // No fallbacks for benchmarking

            std::string error_msg;
            if (!manager.create_session(model_path, config, error_msg)) {
                benchmark.error_message = error_msg;
                results.push_back(benchmark);
                continue;
            }

            // Warmup run
            std::vector<float> output_data;
            std::vector<int64_t> output_shape;
            manager.run_inference(input_data, input_shape, output_data, output_shape, error_msg);

            // Benchmark runs
            std::vector<double> times;
            for (int i = 0; i < num_iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();

                if (manager.run_inference(input_data, input_shape, output_data, output_shape, error_msg)) {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    times.push_back(time_ms);
                } else {
                    benchmark.error_message = error_msg;
                    break;
                }
            }

            if (!times.empty()) {
                benchmark.avg_inference_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                benchmark.successful = true;
            }

        } catch (const std::exception& e) {
            benchmark.error_message = e.what();
        }

        results.push_back(benchmark);
    }

    return results;
}

} // namespace onnx_utils

} // namespace cv_stereo
