#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QTimer>
#include <QStatusBar>
#include <iostream>
#include <memory>

#include "streaming/streaming_optimizer.hpp"
#include "live_stereo_processor.hpp"
#include "camera_manager.hpp"
#include "logging/structured_logger.hpp"

class StreamingDemoWindow : public QMainWindow {
    Q_OBJECT

public:
    StreamingDemoWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setupUI();
        setupProcessor();
        setupTimers();

        auto& logger = StructuredLogger::instance();
        logger.log_info("Streaming demo window initialized");
    }

private slots:
    void startStreaming() {
        if (!streaming_optimizer_) {
            return;
        }

        streaming_optimizer_->start();
        start_button_->setEnabled(false);
        stop_button_->setEnabled(true);

        // Start fake frame generation for demo
        frame_timer_->start(33);  // ~30 FPS input

        auto& logger = StructuredLogger::instance();
        logger.log_info("Streaming started");
    }

    void stopStreaming() {
        if (!streaming_optimizer_) {
            return;
        }

        frame_timer_->stop();
        streaming_optimizer_->stop();
        start_button_->setEnabled(true);
        stop_button_->setEnabled(false);

        auto& logger = StructuredLogger::instance();
        logger.log_info("Streaming stopped");
    }

    void updateConfig() {
        if (!streaming_optimizer_) {
            return;
        }

        cv_stereo::streaming::StreamingOptimizer::StreamingConfig config;
        config.max_buffer_size = buffer_size_spin_->value();
        config.target_fps = target_fps_spin_->value();
        config.enable_frame_dropping = frame_dropping_check_->isChecked();
        config.enable_adaptive_fps = adaptive_fps_check_->isChecked();

        streaming_optimizer_->update_config(config);

        auto& logger = StructuredLogger::instance();
        logger.log_info("Configuration updated", {
            {"buffer_size", std::to_string(config.max_buffer_size)},
            {"target_fps", std::to_string(config.target_fps)}
        });
    }

    void generateFrame() {
        if (!streaming_optimizer_) {
            return;
        }

        // Generate dummy stereo frames for demonstration
        cv::Mat left_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat right_frame = cv::Mat::zeros(480, 640, CV_8UC3);

        // Add some pattern to make it interesting
        cv::rectangle(left_frame, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255, 0, 0), -1);
        cv::rectangle(right_frame, cv::Point(90, 100), cv::Point(190, 200), cv::Scalar(255, 0, 0), -1);

        streaming_optimizer_->push_frame_pair(left_frame, right_frame);
    }

    void updateStats() {
        if (!streaming_optimizer_) {
            return;
        }

        auto stats = streaming_optimizer_->get_stats();

        fps_label_->setText(QString("FPS: %1").arg(stats.current_fps, 0, 'f', 1));
        buffer_label_->setText(QString("Buffer: %1").arg(stats.buffer_size));
        processing_time_label_->setText(QString("Proc Time: %1 ms").arg(stats.average_processing_time_ms, 0, 'f', 1));
        dropped_label_->setText(QString("Dropped: %1").arg(stats.frames_dropped));
        total_label_->setText(QString("Total: %1").arg(stats.total_frames_processed));
    }

private:
    void setupUI() {
        auto* central_widget = new QWidget;
        setCentralWidget(central_widget);

        auto* main_layout = new QVBoxLayout(central_widget);

        // Title
        auto* title_label = new QLabel("Streaming Pipeline Optimizer Demo");
        title_label->setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;");
        main_layout->addWidget(title_label);

        // Control panel
        auto* control_frame = new QWidget;
        auto* control_layout = new QHBoxLayout(control_frame);

        start_button_ = new QPushButton("Start Streaming");
        stop_button_ = new QPushButton("Stop Streaming");
        stop_button_->setEnabled(false);

        connect(start_button_, &QPushButton::clicked, this, &StreamingDemoWindow::startStreaming);
        connect(stop_button_, &QPushButton::clicked, this, &StreamingDemoWindow::stopStreaming);

        control_layout->addWidget(start_button_);
        control_layout->addWidget(stop_button_);
        control_layout->addStretch();

        main_layout->addWidget(control_frame);

        // Configuration panel
        auto* config_frame = new QWidget;
        auto* config_layout = new QHBoxLayout(config_frame);

        config_layout->addWidget(new QLabel("Buffer Size:"));
        buffer_size_spin_ = new QSpinBox;
        buffer_size_spin_->setRange(1, 50);
        buffer_size_spin_->setValue(8);
        connect(buffer_size_spin_, QOverload<int>::of(&QSpinBox::valueChanged), this, &StreamingDemoWindow::updateConfig);
        config_layout->addWidget(buffer_size_spin_);

        config_layout->addWidget(new QLabel("Target FPS:"));
        target_fps_spin_ = new QDoubleSpinBox;
        target_fps_spin_->setRange(1.0, 120.0);
        target_fps_spin_->setValue(30.0);
        connect(target_fps_spin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &StreamingDemoWindow::updateConfig);
        config_layout->addWidget(target_fps_spin_);

        frame_dropping_check_ = new QCheckBox("Frame Dropping");
        frame_dropping_check_->setChecked(true);
        connect(frame_dropping_check_, &QCheckBox::toggled, this, &StreamingDemoWindow::updateConfig);
        config_layout->addWidget(frame_dropping_check_);

        adaptive_fps_check_ = new QCheckBox("Adaptive FPS");
        adaptive_fps_check_->setChecked(true);
        connect(adaptive_fps_check_, &QCheckBox::toggled, this, &StreamingDemoWindow::updateConfig);
        config_layout->addWidget(adaptive_fps_check_);

        config_layout->addStretch();
        main_layout->addWidget(config_frame);

        // Statistics panel
        auto* stats_frame = new QWidget;
        auto* stats_layout = new QHBoxLayout(stats_frame);

        fps_label_ = new QLabel("FPS: 0.0");
        buffer_label_ = new QLabel("Buffer: 0");
        processing_time_label_ = new QLabel("Proc Time: 0.0 ms");
        dropped_label_ = new QLabel("Dropped: 0");
        total_label_ = new QLabel("Total: 0");

        stats_layout->addWidget(fps_label_);
        stats_layout->addWidget(buffer_label_);
        stats_layout->addWidget(processing_time_label_);
        stats_layout->addWidget(dropped_label_);
        stats_layout->addWidget(total_label_);
        stats_layout->addStretch();

        main_layout->addWidget(stats_frame);

        // Description
        auto* desc_label = new QLabel(
            "This demo shows the streaming optimizer in action. "
            "It generates dummy stereo frame pairs and processes them through "
            "the optimized pipeline with buffering, adaptive frame rate control, "
            "and performance monitoring."
        );
        desc_label->setWordWrap(true);
        desc_label->setStyleSheet("margin: 10px; color: #666;");
        main_layout->addWidget(desc_label);

        main_layout->addStretch();

        // Status bar
        statusBar()->showMessage("Ready to start streaming");

        setWindowTitle("Streaming Pipeline Optimizer Demo");
        resize(800, 400);
    }

    void setupProcessor() {
        try {
            // Create the base stereo processor
            processor_ = std::make_unique<stereo_vision::LiveStereoProcessor>();

            // Create streaming optimizer with balanced configuration
            streaming_optimizer_ = cv_stereo::streaming::StreamingOptimizerFactory::create_balanced_optimizer(
                processor_.get());

            auto& logger = StructuredLogger::instance();
            logger.log_info("Streaming processor setup complete");

        } catch (const std::exception& e) {
            auto& logger = StructuredLogger::instance();
            logger.log_error("Failed to setup streaming processor", {{"error", e.what()}});

            // Disable controls if setup failed
            start_button_->setEnabled(false);
            statusBar()->showMessage("Failed to setup streaming processor");
        }
    }

    void setupTimers() {
        // Timer for generating demo frames
        frame_timer_ = new QTimer(this);
        connect(frame_timer_, &QTimer::timeout, this, &StreamingDemoWindow::generateFrame);

        // Timer for updating statistics display
        stats_timer_ = new QTimer(this);
        connect(stats_timer_, &QTimer::timeout, this, &StreamingDemoWindow::updateStats);
        stats_timer_->start(500);  // Update stats every 500ms
    }

private:
    // UI components
    QPushButton* start_button_;
    QPushButton* stop_button_;
    QSpinBox* buffer_size_spin_;
    QDoubleSpinBox* target_fps_spin_;
    QCheckBox* frame_dropping_check_;
    QCheckBox* adaptive_fps_check_;
    QLabel* fps_label_;
    QLabel* buffer_label_;
    QLabel* processing_time_label_;
    QLabel* dropped_label_;
    QLabel* total_label_;

    // Processing components
    std::unique_ptr<stereo_vision::LiveStereoProcessor> processor_;
    std::unique_ptr<cv_stereo::streaming::StreamingOptimizer> streaming_optimizer_;

    // Timers
    QTimer* frame_timer_;
    QTimer* stats_timer_;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Initialize structured logging
    auto& logger = StructuredLogger::instance();
    logger.log_info("Streaming optimizer demo started");

    StreamingDemoWindow window;
    window.show();

    return app.exec();
}

#include "streaming_demo.moc"
