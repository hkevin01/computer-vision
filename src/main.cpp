#include <iostream>
#include <string>
#include <QtWidgets/QApplication>
#include <QtCore/QCommandLineParser>
#include <QtCore/QDir>

#include "camera_calibration.hpp"
#include "stereo_matcher.hpp"
#include "point_cloud_processor.hpp"
#include "gpu_common.hpp"
#include "gui/main_window.hpp"

#ifdef USE_CUDA
#include "gpu_stereo_matcher.hpp"
#endif

void printUsage()
{
    std::cout << "Stereo Vision 3D Point Cloud Generator\n";
    std::cout << "Usage: stereo_vision_app [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --gui              Launch GUI application (default)\n";
    std::cout << "  --console          Run in console mode\n";
    std::cout << "  --left <file>      Left image file\n";
    std::cout << "  --right <file>     Right image file\n";
    std::cout << "  --calibration <file> Camera calibration file\n";
    std::cout << "  --output <file>    Output point cloud file\n";
    std::cout << "  --help             Show this help\n";
}

int runConsoleMode(const QCommandLineParser &parser)
{
    std::cout << "Stereo Vision 3D Point Cloud Generator - Console Mode\n";
    std::cout << "=======================================================\n\n";

    // Check for required arguments
    if (!parser.isSet("left") || !parser.isSet("right"))
    {
        std::cerr << "Error: Both left and right images are required for console mode.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    std::string leftImage = parser.value("left").toStdString();
    std::string rightImage = parser.value("right").toStdString();
    std::string calibrationFile = parser.value("calibration").toStdString();
    std::string outputFile = parser.value("output").toStdString();

    if (outputFile.empty())
    {
        outputFile = "output_pointcloud.ply";
    }

    std::cout << "Configuration:\n";
    std::cout << "  Left image: " << leftImage << "\n";
    std::cout << "  Right image: " << rightImage << "\n";
    std::cout << "  Calibration: " << (calibrationFile.empty() ? "None" : calibrationFile) << "\n";
    std::cout << "  Output: " << outputFile << "\n\n";

    // Initialize GPU
    try
    {
        if (!initializeGPU())
        {
            std::cout << "GPU initialization failed, using CPU processing\n";
        }
        else
        {
            std::cout << "GPU initialized successfully\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU initialization error: " << e.what() << "\n";
        std::cout << "Falling back to CPU processing\n";
    }

    // Initialize components
    CameraCalibration calibration;
    StereoMatcher stereoMatcher;
    PointCloudProcessor pointCloudProcessor;

    try
    {
        // Load calibration if provided
        if (!calibrationFile.empty())
        {
            std::cout << "Loading camera calibration...\n";
            if (!calibration.loadCalibration(calibrationFile))
            {
                std::cerr << "Warning: Failed to load calibration file. Using default parameters.\n";
            }
            else
            {
                std::cout << "Calibration loaded successfully\n";
            }
        }

        // Process stereo images
        std::cout << "Processing stereo images...\n";
        auto pointCloud = stereoMatcher.processStereoPair(leftImage, rightImage);

        if (pointCloud && pointCloud->size() > 0)
        {
            std::cout << "Generated point cloud with " << pointCloud->size() << " points\n";

            // Post-process point cloud
            std::cout << "Post-processing point cloud...\n";
            pointCloudProcessor.filterPointCloud(pointCloud);

            // Save point cloud
            std::cout << "Saving point cloud to " << outputFile << "...\n";
            if (pointCloudProcessor.savePointCloud(pointCloud, outputFile))
            {
                std::cout << "Point cloud saved successfully!\n";
            }
            else
            {
                std::cerr << "Error: Failed to save point cloud\n";
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Failed to generate point cloud\n";
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during processing: " << e.what() << "\n";
        return 1;
    }

    // Cleanup GPU
    cleanupGPU();

    std::cout << "\nProcessing completed successfully!\n";
    return 0;
}

int runGuiMode(QApplication &app)
{
    MainWindow window;
    window.show();
    return app.exec();
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("Stereo Vision 3D Point Cloud Generator");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("Computer Vision Project");

    QCommandLineParser parser;
    parser.setApplicationDescription("Generate 3D point clouds from stereo camera images");
    parser.addHelpOption();
    parser.addVersionOption();

    // Add command line options
    QCommandLineOption guiOption("gui", "Launch GUI application (default)");
    QCommandLineOption consoleOption("console", "Run in console mode");
    QCommandLineOption leftOption("left", "Left image file", "file");
    QCommandLineOption rightOption("right", "Right image file", "file");
    QCommandLineOption calibrationOption("calibration", "Camera calibration file", "file");
    QCommandLineOption outputOption("output", "Output point cloud file", "file");

    parser.addOption(guiOption);
    parser.addOption(consoleOption);
    parser.addOption(leftOption);
    parser.addOption(rightOption);
    parser.addOption(calibrationOption);
    parser.addOption(outputOption);

    parser.process(app);

    // Check if console mode is requested
    if (parser.isSet(consoleOption))
    {
        return runConsoleMode(parser);
    }
    else
    {
        return runGuiMode(app);
    }
}
