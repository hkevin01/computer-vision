#include <QApplication>
#include <QMessageBox>
#include <iostream>

#include "gui/live_stereo_tuning_window.hpp"

/**
 * Test application for Live Stereo Parameter Tuning
 *
 * Usage: ./live_stereo_tuning [left_image] [right_image]
 */
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Set application properties
    QApplication::setApplicationName("Live Stereo Parameter Tuning");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("Computer Vision Suite");

    try {
        // Create and show the main window
        stereo_vision::gui::LiveStereoTuningWindow window;

        // If command line arguments provided, load the images
        if (argc >= 3) {
            QString leftPath = QString::fromLocal8Bit(argv[1]);
            QString rightPath = QString::fromLocal8Bit(argv[2]);

            std::cout << "Loading stereo images:" << std::endl;
            std::cout << "Left:  " << leftPath.toStdString() << std::endl;
            std::cout << "Right: " << rightPath.toStdString() << std::endl;

            if (!window.loadStereoImages(leftPath, rightPath)) {
                QMessageBox::critical(nullptr, "Error", "Failed to load stereo images");
                return 1;
            }

            std::cout << "Images loaded successfully!" << std::endl;
        } else {
            std::cout << "Usage: " << argv[0] << " [left_image] [right_image]" << std::endl;
            std::cout << "Or use File menu to load images interactively." << std::endl;
        }

        window.show();

        return app.exec();

    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Fatal Error",
                              QString("Application error: %1").arg(e.what()));
        return 1;
    } catch (...) {
        QMessageBox::critical(nullptr, "Fatal Error", "Unknown application error occurred");
        return 1;
    }
}
