#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "camera_calibration.hpp"
#include "stereo_matcher.hpp"
#include "point_cloud_processor.hpp"

using namespace stereo_vision;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --calibrate <left_dir> <right_dir>  Calibrate stereo cameras\n";
    std::cout << "  --process <left_img> <right_img>    Process stereo image pair\n";
    std::cout << "  --gui                               Launch GUI interface\n";
    std::cout << "  --help                              Show this help message\n";
}

int runCalibration(const std::string& left_dir, const std::string& right_dir) {
    try {
        CameraCalibration calibrator;
        
        // Load calibration images
        std::vector<cv::Mat> left_images, right_images;
        
        // Simple image loading (you would expand this for directory scanning)
        std::cout << "Loading calibration images from " << left_dir << " and " << right_dir << std::endl;
        
        // For demonstration, assuming numbered image files
        for (int i = 1; i <= 20; ++i) {
            std::string left_file = left_dir + "/left_" + std::to_string(i) + ".jpg";
            std::string right_file = right_dir + "/right_" + std::to_string(i) + ".jpg";
            
            cv::Mat left_img = cv::imread(left_file);
            cv::Mat right_img = cv::imread(right_file);
            
            if (!left_img.empty() && !right_img.empty()) {
                left_images.push_back(left_img);
                right_images.push_back(right_img);
            }
        }
        
        if (left_images.size() < 10) {
            std::cerr << "Error: Need at least 10 calibration image pairs" << std::endl;
            return -1;
        }
        
        // Perform stereo calibration
        cv::Size board_size(9, 6);  // 9x6 checkerboard
        float square_size = 20.0f;  // 20mm squares
        
        std::cout << "Performing stereo calibration..." << std::endl;
        auto stereo_params = calibrator.calibrateStereoCamera(
            left_images, right_images, board_size, square_size);
        
        // Save calibration results
        std::string calib_file = "stereo_calibration.yml";
        if (calibrator.saveCalibration(calib_file, stereo_params)) {
            std::cout << "Calibration saved to " << calib_file << std::endl;
        } else {
            std::cerr << "Failed to save calibration" << std::endl;
            return -1;
        }
        
        std::cout << "Stereo calibration completed successfully!" << std::endl;
        std::cout << "Reprojection error: " << stereo_params.reprojection_error << " pixels" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Calibration error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

int processStereoPair(const std::string& left_file, const std::string& right_file) {
    try {
        // Load calibration
        CameraCalibration calibrator;
        CameraCalibration::StereoParameters stereo_params;
        
        if (!calibrator.loadCalibration("stereo_calibration.yml", stereo_params)) {
            std::cerr << "Error: Could not load calibration file. Run calibration first." << std::endl;
            return -1;
        }
        
        // Load stereo images
        cv::Mat left_img = cv::imread(left_file);
        cv::Mat right_img = cv::imread(right_file);
        
        if (left_img.empty() || right_img.empty()) {
            std::cerr << "Error: Could not load input images" << std::endl;
            return -1;
        }
        
        // Initialize stereo matcher
        StereoMatcher matcher;
        matcher.initialize(stereo_params);
        
        std::cout << "Computing disparity map..." << std::endl;
        cv::Mat disparity = matcher.computeDisparity(left_img, right_img);
        
        // Generate point cloud
        std::cout << "Generating point cloud..." << std::endl;
        auto point_cloud = matcher.generatePointCloud(disparity, left_img);
        
        // Process and save point cloud
        PointCloudProcessor processor;
        auto filtered_cloud = processor.filterPointCloud(point_cloud);
        
        std::string output_file = "output_pointcloud.ply";
        if (processor.exportPointCloud(filtered_cloud, output_file)) {
            std::cout << "Point cloud saved to " << output_file << std::endl;
        } else {
            std::cerr << "Failed to save point cloud" << std::endl;
            return -1;
        }
        
        // Display statistics
        auto stats = processor.computeStatistics(filtered_cloud);
        std::cout << "Point cloud statistics:" << std::endl;
        std::cout << "  Number of points: " << stats.num_points << std::endl;
        std::cout << "  Bounding box volume: " << stats.bounding_box_volume << " m³" << std::endl;
        
        // Save disparity visualization
        cv::Mat disparity_vis;
        cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(disparity_vis, disparity_vis, cv::COLORMAP_JET);
        cv::imwrite("disparity_map.jpg", disparity_vis);
        std::cout << "Disparity map saved to disparity_map.jpg" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

int launchGUI() {
    std::cout << "GUI mode not yet implemented." << std::endl;
    std::cout << "GUI will provide:" << std::endl;
    std::cout << "  - Interactive parameter adjustment" << std::endl;
    std::cout << "  - Real-time preview" << std::endl;
    std::cout << "  - Point cloud visualization" << std::endl;
    std::cout << "  - Calibration workflow" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "Stereo Vision 3D Point Cloud Generator" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string command = argv[1];
    
    if (command == "--help") {
        printUsage(argv[0]);
        return 0;
    } else if (command == "--calibrate") {
        if (argc != 4) {
            std::cerr << "Error: --calibrate requires two directory paths" << std::endl;
            printUsage(argv[0]);
            return -1;
        }
        return runCalibration(argv[2], argv[3]);
    } else if (command == "--process") {
        if (argc != 4) {
            std::cerr << "Error: --process requires two image file paths" << std::endl;
            printUsage(argv[0]);
            return -1;
        }
        return processStereoPair(argv[2], argv[3]);
    } else if (command == "--gui") {
        return launchGUI();
    } else {
        std::cerr << "Error: Unknown command '" << command << "'" << std::endl;
        printUsage(argv[0]);
        return -1;
    }
    
    return 0;
}
