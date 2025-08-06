#include <QApplication>
#include "gui/batch_processing_window.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Set application properties
    app.setApplicationName("Batch Processing Test");
    app.setApplicationVersion("1.0.0");

    // Create and show the batch processing window
    stereo_vision::batch::BatchProcessingWindow window;
    window.show();

    return app.exec();
}
