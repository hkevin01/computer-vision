#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    std::string model_path = "models/hitnet.onnx";
    if (argc > 1) model_path = argv[1];

    std::cout << "test_onnx_load: starting" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    // This is a minimal stub that tries to load ONNX Runtime if available at runtime.
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);

        Ort::Session session(env, model_path.c_str(), opts);

        auto info = session.GetModelMetadata();
        std::cout << "Provider count: " << session.GetProviderCount() << std::endl;

        // Create dummy input for testing
        std::vector<int64_t> dims = {1, 3, 480, 640};
        size_t tensor_size = 1 * 3 * 480 * 640;
        std::vector<float> input_tensor_values(tensor_size, 0.0f);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), tensor_size, dims.data(), dims.size());
        const char* input_names[] = {session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
        const char* output_names[] = {session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        std::cout << "ONNX Runtime inference completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
#else
    std::cout << "ONNX Runtime not available at compile time — skipping deep test" << std::endl;
    std::cout << "To enable ONNX support, install libonnxruntime-dev and reconfigure with -DWITH_ONNX=ON" << std::endl;
#endif

    std::cout << "test_onnx_load: done" << std::endl;
    return 0;
}
