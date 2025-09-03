#include <iostream>
#include <vector>
#include <chrono>

int main(){
    std::cout<<"test_onnx_load: starting"<<std::endl;
    // This is a minimal stub that tries to load ONNX Runtime if available at runtime.
#if __has_include(<onnxruntime_cxx_api.h>)
    try{
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout<<"ONNX Runtime available"<<std::endl;
    }catch(...){
        std::cout<<"ONNX Runtime present but failed to init"<<std::endl;
        return 1;
    }
#else
    std::cout<<"ONNX Runtime headers not found at compile time — skipping deep test"<<std::endl;
#endif
    std::cout<<"test_onnx_load: done"<<std::endl;
    return 0;
}
#include <iostream>
#include <chrono>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    std::string model_path = "models/hitnet.onnx";
    if (argc > 1) model_path = argv[1];
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    try {
        Ort::Session session(env, model_path.c_str(), opts);
        std::cout << "Loaded model: " << model_path << std::endl;
        auto info = session.GetModelMetadata();
        std::cout << "Provider count: " << session.GetProviderCount() << std::endl;
        // create synthetic input according to common small shape
        std::vector<int64_t> dims = {1,3,32,32};
        size_t tensor_size = 1*3*32*32;
        std::vector<float> input_tensor_values(tensor_size, 0.5f);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), tensor_size, dims.data(), dims.size());
        const char* input_names[] = {session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
        const char* output_names[] = {session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Inference time ms: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << std::endl;
        std::cout << "Output tensor count: " << output_tensors.size() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Failed to load/run ONNX model: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
