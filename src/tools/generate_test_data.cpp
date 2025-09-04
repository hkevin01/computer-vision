#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>

namespace fs = std::filesystem;

void create_synthetic_stereo_pair(const std::string& output_dir,
                                 const std::string& basename,
                                 int width = 640,
                                 int height = 480) {
    // Create synthetic stereo pair with known disparity pattern
    cv::Mat left_image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat right_image = cv::Mat::zeros(height, width, CV_8UC3);

    // Create a pattern with known disparities
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a gradient pattern
            int intensity = (x * 255 / width + y * 255 / height) / 2;

            // Add some geometric shapes with known disparities
            if (x > width/4 && x < 3*width/4 && y > height/4 && y < 3*height/4) {
                // Central rectangle - disparity 16 pixels
                intensity = 200;
                int right_x = std::max(0, x - 16);
                if (right_x < width) {
                    right_image.at<cv::Vec3b>(y, right_x) = cv::Vec3b(intensity, intensity, intensity);
                }
            } else if ((x - width/2) * (x - width/2) + (y - height/2) * (y - height/2) < 50*50) {
                // Central circle - disparity 8 pixels
                intensity = 150;
                int right_x = std::max(0, x - 8);
                if (right_x < width) {
                    right_image.at<cv::Vec3b>(y, right_x) = cv::Vec3b(intensity, intensity, intensity);
                }
            } else {
                // Background - disparity 4 pixels
                int right_x = std::max(0, x - 4);
                if (right_x < width) {
                    right_image.at<cv::Vec3b>(y, right_x) = cv::Vec3b(intensity, intensity, intensity);
                }
            }

            left_image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
        }
    }

    // Save images
    cv::imwrite(output_dir + "/" + basename + "_left.png", left_image);
    cv::imwrite(output_dir + "/" + basename + "_right.png", right_image);

    std::cout << "Created synthetic stereo pair: " << basename << std::endl;
}

void create_calibration_data(const std::string& output_dir, const std::string& basename) {
    // Create sample camera calibration parameters
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
        525.0, 0.0, 320.0,
        0.0, 525.0, 240.0,
        0.0, 0.0, 1.0);

    cv::Mat dist_coeffs = (cv::Mat_<double>(1,5) << 0.1, -0.2, 0.0, 0.0, 0.0);

    // Save calibration data in OpenCV XML format
    cv::FileStorage fs(output_dir + "/" + basename + "_camera_params.xml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs << "image_width" << 640;
    fs << "image_height" << 480;
    fs.release();

    // Also save stereo calibration parameters
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);  // Rectified, so rotation is identity
    cv::Mat T = (cv::Mat_<double>(3,1) << -60.0, 0.0, 0.0);  // 60mm baseline

    cv::FileStorage stereo_fs(output_dir + "/" + basename + "_stereo_params.xml", cv::FileStorage::WRITE);
    stereo_fs << "camera_matrix_left" << camera_matrix;
    stereo_fs << "camera_matrix_right" << camera_matrix;
    stereo_fs << "dist_coeffs_left" << dist_coeffs;
    stereo_fs << "dist_coeffs_right" << dist_coeffs;
    stereo_fs << "R" << R;
    stereo_fs << "T" << T;
    stereo_fs << "baseline_mm" << 60.0;
    stereo_fs.release();

    std::cout << "Created calibration data: " << basename << std::endl;
}

void create_expected_disparity_map(const std::string& output_dir, const std::string& basename) {
    // Create expected disparity map that matches our synthetic stereo pair
    cv::Mat disparity = cv::Mat::zeros(480, 640, CV_16S);  // OpenCV disparity format (fixed point)

    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            short disp_value = 4 * 16;  // Background disparity = 4 pixels, scaled by 16

            // Central rectangle - disparity 16 pixels
            if (x > 640/4 && x < 3*640/4 && y > 480/4 && y < 3*480/4) {
                disp_value = 16 * 16;
            }
            // Central circle - disparity 8 pixels
            else if ((x - 640/2) * (x - 640/2) + (y - 480/2) * (y - 480/2) < 50*50) {
                disp_value = 8 * 16;
            }

            disparity.at<short>(y, x) = disp_value;
        }
    }

    // Save as 16-bit signed image (OpenCV standard)
    cv::imwrite(output_dir + "/" + basename + "_disparity.png", disparity);

    // Also save normalized version for visualization
    cv::Mat normalized;
    cv::normalize(disparity, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(output_dir + "/" + basename + "_disparity_vis.png", normalized);

    std::cout << "Created expected disparity map: " << basename << std::endl;
}

int main(int argc, char* argv[]) {
    std::string base_dir = "data";
    if (argc > 1) {
        base_dir = argv[1];
    }

    std::string stereo_dir = base_dir + "/stereo_images/smoke_test";
    std::string calib_dir = base_dir + "/calibration/smoke_test";

    // Create directories
    fs::create_directories(stereo_dir);
    fs::create_directories(calib_dir);

    // Create test datasets
    std::vector<std::string> test_cases = {
        "simple_gradient",
        "geometric_shapes",
        "textured_scene"
    };

    for (const auto& test_case : test_cases) {
        create_synthetic_stereo_pair(stereo_dir, test_case);
        create_calibration_data(calib_dir, test_case);
        create_expected_disparity_map(stereo_dir, test_case);
    }

    // Create README with dataset description
    std::ofstream readme(stereo_dir + "/README.md");
    readme << "# Smoke Test Stereo Datasets\n\n";
    readme << "This directory contains synthetic stereo image pairs for automated testing.\n\n";
    readme << "## Test Cases:\n\n";
    readme << "### simple_gradient\n";
    readme << "- Basic gradient pattern with uniform disparity\n";
    readme << "- Expected disparity: 4 pixels background\n\n";
    readme << "### geometric_shapes\n";
    readme << "- Rectangle and circle shapes with different disparities\n";
    readme << "- Rectangle: 16 pixels disparity\n";
    readme << "- Circle: 8 pixels disparity\n";
    readme << "- Background: 4 pixels disparity\n\n";
    readme << "### textured_scene\n";
    readme << "- More complex textured pattern\n";
    readme << "- Variable disparities across the scene\n\n";
    readme << "## Files per test case:\n";
    readme << "- `{name}_left.png` - Left camera image\n";
    readme << "- `{name}_right.png` - Right camera image\n";
    readme << "- `{name}_disparity.png` - Expected disparity map (16-bit)\n";
    readme << "- `{name}_disparity_vis.png` - Visualization (8-bit)\n\n";
    readme << "## Calibration data:\n";
    readme << "Located in `../calibration/smoke_test/`\n";
    readme << "- `{name}_camera_params.xml` - Camera intrinsics\n";
    readme << "- `{name}_stereo_params.xml` - Stereo rig parameters\n\n";
    readme << "## Usage in tests:\n";
    readme << "```cpp\n";
    readme << "// Load test images\n";
    readme << "cv::Mat left = cv::imread(\"data/stereo_images/smoke_test/simple_gradient_left.png\");\n";
    readme << "cv::Mat right = cv::imread(\"data/stereo_images/smoke_test/simple_gradient_right.png\");\n";
    readme << "cv::Mat expected = cv::imread(\"data/stereo_images/smoke_test/simple_gradient_disparity.png\", cv::IMREAD_UNCHANGED);\n";
    readme << "\n";
    readme << "// Compute disparity\n";
    readme << "cv::Mat computed_disparity = stereo_matcher.compute(left, right);\n";
    readme << "\n";
    readme << "// Compare with tolerance\n";
    readme << "double mse = compare_disparity_maps(computed_disparity, expected, tolerance=2.0);\n";
    readme << "```\n";
    readme.close();

    std::cout << "\nSynthetic test data created successfully!" << std::endl;
    std::cout << "Stereo images: " << stereo_dir << std::endl;
    std::cout << "Calibration data: " << calib_dir << std::endl;

    return 0;
}
