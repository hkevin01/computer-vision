#include "gui/epipolar_checker.hpp"
#include "gui/modern_theme.hpp"
#include <QApplication>
#include <QHeaderView>
#include <QTableWidgetItem>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QStandardPaths>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace stereo_vision::gui {

// EpipolarImageWidget Implementation
EpipolarImageWidget::EpipolarImageWidget(QWidget* parent)
    : EnhancedImageWidget(parent)
    , is_left_image_(true)
    , show_point_labels_(true)
    , color_coding_enabled_(true)
    , line_thickness_(2)
    , error_threshold_(1.0)
    , selected_point_id_(-1) {

    setMinimumSize(400, 300);
    setMouseTracking(true);

    // Enable context menu for additional options
    setContextMenuPolicy(Qt::CustomContextMenu);
}

void EpipolarImageWidget::setImage(const cv::Mat& image) {
    current_image_ = image.clone();
    EnhancedImageWidget::setImage(image);
}

void EpipolarImageWidget::setEpipolarLines(const std::vector<cv::Vec3f>& lines) {
    epipolar_lines_ = lines;
    update();
}

void EpipolarImageWidget::setEpipolarPoints(const std::vector<EpipolarPoint>& points) {
    epipolar_points_ = points;
    update();
}

void EpipolarImageWidget::clearEpipolarData() {
    epipolar_lines_.clear();
    epipolar_points_.clear();
    selected_point_id_ = -1;
    update();
}

void EpipolarImageWidget::paintEvent(QPaintEvent* event) {
    // Call parent implementation to draw the base image
    EnhancedImageWidget::paintEvent(event);

    if (current_image_.empty()) return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw epipolar lines and points
    drawEpipolarLines(painter);
    drawEpipolarPoints(painter);

    if (show_point_labels_) {
        drawPointLabels(painter);
    }
}

void EpipolarImageWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !current_image_.empty()) {
        cv::Point2f image_point = screenToImage(event->pos());

        // Check if point is within image bounds
        if (image_point.x >= 0 && image_point.x < current_image_.cols &&
            image_point.y >= 0 && image_point.y < current_image_.rows) {

            last_clicked_point_ = image_point;
            emit pointClicked(image_point);

            // Check if clicking near an existing point to select it
            for (const auto& ep_point : epipolar_points_) {
                cv::Point2f ref_point = is_left_image_ ? ep_point.left_point : ep_point.right_point;
                double distance = cv::norm(ref_point - image_point);
                if (distance < 10.0) { // 10 pixel tolerance
                    selected_point_id_ = ep_point.point_id;
                    emit pointSelected(ep_point.point_id);
                    update();
                    break;
                }
            }
        }
    }

    EnhancedImageWidget::mousePressEvent(event);
}

void EpipolarImageWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !current_image_.empty()) {
        cv::Point2f image_point = screenToImage(event->pos());

        // Double-click to add a new point pair
        if (image_point.x >= 0 && image_point.x < current_image_.cols &&
            image_point.y >= 0 && image_point.y < current_image_.rows) {

            last_clicked_point_ = image_point;
            emit pointClicked(image_point);
        }
    }

    EnhancedImageWidget::mouseDoubleClickEvent(event);
}

void EpipolarImageWidget::drawEpipolarLines(QPainter& painter) {
    if (epipolar_lines_.empty() || current_image_.empty()) return;

    painter.save();

    QPen line_pen;
    line_pen.setWidth(line_thickness_);
    line_pen.setStyle(Qt::SolidLine);

    for (size_t i = 0; i < epipolar_lines_.size(); ++i) {
        const cv::Vec3f& line = epipolar_lines_[i];

        // Calculate line endpoints within image bounds
        std::vector<cv::Point2f> intersections;

        // Intersection with left edge (x = 0)
        if (std::abs(line[0]) > 1e-6) {
            float y = -line[2] / line[1];
            if (y >= 0 && y < current_image_.rows) {
                intersections.push_back(cv::Point2f(0, y));
            }
        }

        // Intersection with right edge (x = width)
        if (std::abs(line[0]) > 1e-6) {
            float y = -(line[0] * current_image_.cols + line[2]) / line[1];
            if (y >= 0 && y < current_image_.rows) {
                intersections.push_back(cv::Point2f(current_image_.cols - 1, y));
            }
        }

        // Intersection with top edge (y = 0)
        if (std::abs(line[1]) > 1e-6) {
            float x = -line[2] / line[0];
            if (x >= 0 && x < current_image_.cols) {
                intersections.push_back(cv::Point2f(x, 0));
            }
        }

        // Intersection with bottom edge (y = height)
        if (std::abs(line[1]) > 1e-6) {
            float x = -(line[1] * current_image_.rows + line[2]) / line[0];
            if (x >= 0 && x < current_image_.cols) {
                intersections.push_back(cv::Point2f(x, current_image_.rows - 1));
            }
        }

        if (intersections.size() >= 2) {
            // Color coding based on error if enabled
            QColor line_color = Qt::green;
            if (color_coding_enabled_ && i < epipolar_points_.size()) {
                line_color = getErrorColor(epipolar_points_[i].epipolar_error);
            }

            line_pen.setColor(line_color);
            painter.setPen(line_pen);

            QPoint start = imageToScreen(intersections[0]);
            QPoint end = imageToScreen(intersections[1]);
            painter.drawLine(start, end);
        }
    }

    painter.restore();
}

void EpipolarImageWidget::drawEpipolarPoints(QPainter& painter) {
    if (epipolar_points_.empty() || current_image_.empty()) return;

    painter.save();

    for (const auto& ep_point : epipolar_points_) {
        cv::Point2f ref_point = is_left_image_ ? ep_point.left_point : ep_point.right_point;
        QPoint screen_point = imageToScreen(ref_point);

        // Determine point color
        QColor point_color;
        if (ep_point.point_id == selected_point_id_) {
            point_color = Qt::yellow;
        } else if (ep_point.is_ground_truth) {
            point_color = Qt::blue;
        } else if (color_coding_enabled_) {
            point_color = getErrorColor(ep_point.epipolar_error);
        } else {
            point_color = Qt::red;
        }

        // Draw point
        painter.setBrush(QBrush(point_color));
        painter.setPen(QPen(Qt::black, 1));

        int point_size = (ep_point.point_id == selected_point_id_) ? 8 : 6;
        painter.drawEllipse(screen_point, point_size, point_size);

        // Draw predicted point in right image
        if (!is_left_image_ && ep_point.predicted_right_point.x >= 0) {
            QPoint predicted_screen = imageToScreen(ep_point.predicted_right_point);
            painter.setBrush(QBrush(Qt::magenta));
            painter.drawEllipse(predicted_screen, 4, 4);

            // Draw line between actual and predicted
            painter.setPen(QPen(Qt::magenta, 1, Qt::DashLine));
            painter.drawLine(screen_point, predicted_screen);
        }
    }

    painter.restore();
}

void EpipolarImageWidget::drawPointLabels(QPainter& painter) {
    if (epipolar_points_.empty() || current_image_.empty()) return;

    painter.save();

    QFont label_font = painter.font();
    label_font.setPointSize(10);
    label_font.setBold(true);
    painter.setFont(label_font);

    for (const auto& ep_point : epipolar_points_) {
        cv::Point2f ref_point = is_left_image_ ? ep_point.left_point : ep_point.right_point;
        QPoint screen_point = imageToScreen(ref_point);

        QString label = QString("P%1").arg(ep_point.point_id);
        if (!is_left_image_) {
            label += QString("\nE:%.2f").arg(ep_point.epipolar_error);
        }

        // Text background
        QRect text_rect = painter.fontMetrics().boundingRect(label);
        text_rect.moveCenter(screen_point + QPoint(15, -15));

        painter.fillRect(text_rect.adjusted(-2, -2, 2, 2), QColor(255, 255, 255, 200));
        painter.setPen(Qt::black);
        painter.drawText(text_rect, Qt::AlignCenter, label);
    }

    painter.restore();
}

QColor EpipolarImageWidget::getErrorColor(double error) const {
    if (error <= error_threshold_ * 0.5) {
        return Qt::green;
    } else if (error <= error_threshold_) {
        return Qt::yellow;
    } else if (error <= error_threshold_ * 2.0) {
        return QColor(255, 165, 0); // Orange
    } else {
        return Qt::red;
    }
}

cv::Point2f EpipolarImageWidget::screenToImage(const QPoint& screen_point) const {
    if (current_image_.empty()) return cv::Point2f(-1, -1);

    // Account for widget scaling and positioning
    QRect widget_rect = rect();
    QSize image_size(current_image_.cols, current_image_.rows);
    QSize scaled_size = image_size.scaled(widget_rect.size(), Qt::KeepAspectRatio);

    QRect image_rect;
    image_rect.setSize(scaled_size);
    image_rect.moveCenter(widget_rect.center());

    if (!image_rect.contains(screen_point)) {
        return cv::Point2f(-1, -1);
    }

    QPoint relative_point = screen_point - image_rect.topLeft();

    float scale_x = static_cast<float>(current_image_.cols) / scaled_size.width();
    float scale_y = static_cast<float>(current_image_.rows) / scaled_size.height();

    return cv::Point2f(relative_point.x() * scale_x, relative_point.y() * scale_y);
}

QPoint EpipolarImageWidget::imageToScreen(const cv::Point2f& image_point) const {
    if (current_image_.empty()) return QPoint(-1, -1);

    QRect widget_rect = rect();
    QSize image_size(current_image_.cols, current_image_.rows);
    QSize scaled_size = image_size.scaled(widget_rect.size(), Qt::KeepAspectRatio);

    QRect image_rect;
    image_rect.setSize(scaled_size);
    image_rect.moveCenter(widget_rect.center());

    float scale_x = static_cast<float>(scaled_size.width()) / current_image_.cols;
    float scale_y = static_cast<float>(scaled_size.height()) / current_image_.rows;

    QPoint relative_point(static_cast<int>(image_point.x * scale_x),
                         static_cast<int>(image_point.y * scale_y));

    return image_rect.topLeft() + relative_point;
}

// EpipolarChecker Implementation
EpipolarChecker::EpipolarChecker(QWidget* parent)
    : QWidget(parent)
    , has_calibration_(false)
    , has_images_(false)
    , has_valid_fundamental_matrix_(false)
    , live_mode_enabled_(false)
    , next_point_id_(1)
    , selected_point_id_(-1)
    , accuracy_threshold_(1.0)
    , auto_compute_statistics_(true)
    , max_epipolar_points_(50)
    , pending_left_point_(-1, -1) {

    setupUI();
    connectSignals();

    // Initialize statistics
    current_statistics_.clear();

    // Setup live mode timer
    live_update_timer_ = new QTimer(this);
    live_update_timer_->setSingleShot(false);
    live_update_timer_->setInterval(100); // 10 FPS
    connect(live_update_timer_, &QTimer::timeout, this, &EpipolarChecker::updateVisualization);
}

EpipolarChecker::~EpipolarChecker() = default;

void EpipolarChecker::setupUI() {
    setWindowTitle("Epipolar Line Checker - Calibration Quality Assessment");

    // Main layout
    auto* main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(6, 6, 6, 6);
    main_layout->setSpacing(6);

    // Create main splitter
    main_splitter_ = new QSplitter(Qt::Horizontal, this);
    main_layout->addWidget(main_splitter_);

    setupImageDisplays();
    setupControlPanel();
    setupStatisticsPanel();

    // Set initial splitter sizes
    main_splitter_->setSizes({800, 300});
    main_splitter_->setStretchFactor(0, 1);
    main_splitter_->setStretchFactor(1, 0);
}

void EpipolarChecker::setupImageDisplays() {
    // Create image splitter
    image_splitter_ = new QSplitter(Qt::Vertical, main_splitter_);

    // Left image display
    auto* left_container = new QWidget();
    auto* left_layout = new QVBoxLayout(left_container);
    left_layout->setContentsMargins(2, 2, 2, 2);

    auto* left_label = new QLabel("Left Image");
    left_label->setStyleSheet("font-weight: bold; padding: 4px;");
    left_layout->addWidget(left_label);

    left_image_widget_ = new EpipolarImageWidget();
    left_image_widget_->setIsLeftImage(true);
    left_scroll_area_ = new QScrollArea();
    left_scroll_area_->setWidget(left_image_widget_);
    left_scroll_area_->setWidgetResizable(true);
    left_layout->addWidget(left_scroll_area_);

    image_splitter_->addWidget(left_container);

    // Right image display
    auto* right_container = new QWidget();
    auto* right_layout = new QVBoxLayout(right_container);
    right_layout->setContentsMargins(2, 2, 2, 2);

    auto* right_label = new QLabel("Right Image");
    right_label->setStyleSheet("font-weight: bold; padding: 4px;");
    right_layout->addWidget(right_label);

    right_image_widget_ = new EpipolarImageWidget();
    right_image_widget_->setIsLeftImage(false);
    right_scroll_area_ = new QScrollArea();
    right_scroll_area_->setWidget(right_image_widget_);
    right_scroll_area_->setWidgetResizable(true);
    right_layout->addWidget(right_scroll_area_);

    image_splitter_->addWidget(right_container);

    // Set equal sizes for image displays
    image_splitter_->setSizes({400, 400});

    main_splitter_->addWidget(image_splitter_);
}

void EpipolarChecker::setupControlPanel() {
    control_panel_ = new QWidget();
    control_panel_->setMaximumWidth(300);
    control_panel_->setMinimumWidth(280);

    auto* layout = new QVBoxLayout(control_panel_);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(8);

    // Calibration group
    calibration_group_ = new QGroupBox("Calibration");
    auto* calib_layout = new QVBoxLayout(calibration_group_);

    calibration_status_label_ = new QLabel("No calibration loaded");
    calibration_status_label_->setWordWrap(true);
    calibration_status_label_->setStyleSheet("color: red; font-style: italic;");
    calib_layout->addWidget(calibration_status_label_);

    load_calibration_button_ = new QPushButton("Load Calibration");
    load_images_button_ = new QPushButton("Load Stereo Images");
    calib_layout->addWidget(load_calibration_button_);
    calib_layout->addWidget(load_images_button_);

    layout->addWidget(calibration_group_);

    // Point management group
    point_management_group_ = new QGroupBox("Point Management");
    auto* point_layout = new QVBoxLayout(point_management_group_);

    add_point_button_ = new QPushButton("Add Point Pair");
    add_point_button_->setEnabled(false);
    point_layout->addWidget(add_point_button_);

    remove_point_button_ = new QPushButton("Remove Selected");
    remove_point_button_->setEnabled(false);
    point_layout->addWidget(remove_point_button_);

    clear_all_button_ = new QPushButton("Clear All Points");
    clear_all_button_->setEnabled(false);
    point_layout->addWidget(clear_all_button_);

    auto* separator = new QFrame();
    separator->setFrameStyle(QFrame::HLine);
    point_layout->addWidget(separator);

    auto_verify_button_ = new QPushButton("Run Auto Verification");
    auto_verify_button_->setEnabled(false);
    point_layout->addWidget(auto_verify_button_);

    layout->addWidget(point_management_group_);

    // Visualization group
    visualization_group_ = new QGroupBox("Visualization");
    auto* vis_layout = new QGridLayout(visualization_group_);

    show_lines_checkbox_ = new QCheckBox("Show Epipolar Lines");
    show_lines_checkbox_->setChecked(true);
    vis_layout->addWidget(show_lines_checkbox_, 0, 0, 1, 2);

    show_points_checkbox_ = new QCheckBox("Show Points");
    show_points_checkbox_->setChecked(true);
    vis_layout->addWidget(show_points_checkbox_, 1, 0, 1, 2);

    show_labels_checkbox_ = new QCheckBox("Show Labels");
    show_labels_checkbox_->setChecked(true);
    vis_layout->addWidget(show_labels_checkbox_, 2, 0, 1, 2);

    color_coding_checkbox_ = new QCheckBox("Color Coding");
    color_coding_checkbox_->setChecked(true);
    vis_layout->addWidget(color_coding_checkbox_, 3, 0, 1, 2);

    vis_layout->addWidget(new QLabel("Line Thickness:"), 4, 0);
    line_thickness_slider_ = new QSlider(Qt::Horizontal);
    line_thickness_slider_->setRange(1, 5);
    line_thickness_slider_->setValue(2);
    vis_layout->addWidget(line_thickness_slider_, 4, 1);

    vis_layout->addWidget(new QLabel("Error Threshold:"), 5, 0);
    error_threshold_spinbox_ = new QDoubleSpinBox();
    error_threshold_spinbox_->setRange(0.1, 10.0);
    error_threshold_spinbox_->setValue(1.0);
    error_threshold_spinbox_->setSingleStep(0.1);
    error_threshold_spinbox_->setSuffix(" px");
    vis_layout->addWidget(error_threshold_spinbox_, 5, 1);

    layout->addWidget(visualization_group_);

    // Live mode
    live_mode_checkbox_ = new QCheckBox("Live Mode");
    live_mode_checkbox_->setToolTip("Automatically update visualization when parameters change");
    layout->addWidget(live_mode_checkbox_);

    // Export buttons
    auto* export_layout = new QVBoxLayout();
    export_report_button_ = new QPushButton("Export Report");
    export_report_button_->setEnabled(false);
    export_layout->addWidget(export_report_button_);

    save_points_button_ = new QPushButton("Save Points");
    save_points_button_->setEnabled(false);
    export_layout->addWidget(save_points_button_);

    load_points_button_ = new QPushButton("Load Points");
    export_layout->addWidget(load_points_button_);

    layout->addLayout(export_layout);

    layout->addStretch();

    main_splitter_->addWidget(control_panel_);
}

void EpipolarChecker::setupStatisticsPanel() {
    statistics_panel_ = new QWidget();
    auto* layout = new QVBoxLayout(statistics_panel_);
    layout->setContentsMargins(6, 6, 6, 6);

    // Analysis group
    analysis_group_ = new QGroupBox("Analysis Results");
    auto* analysis_layout = new QVBoxLayout(analysis_group_);

    // Points table
    points_table_ = new QTableWidget();
    points_table_->setColumnCount(6);
    QStringList headers = {"ID", "Left X", "Left Y", "Right X", "Right Y", "Error (px)"};
    points_table_->setHorizontalHeaderLabels(headers);
    points_table_->horizontalHeader()->setStretchLastSection(true);
    points_table_->setAlternatingRowColors(true);
    points_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    points_table_->setMaximumHeight(200);
    analysis_layout->addWidget(points_table_);

    // Statistics text
    statistics_text_ = new QTextEdit();
    statistics_text_->setMaximumHeight(200);
    statistics_text_->setReadOnly(true);
    statistics_text_->setFont(QFont("Courier", 9));
    analysis_layout->addWidget(statistics_text_);

    // Verification progress
    verification_progress_ = new QProgressBar();
    verification_progress_->setVisible(false);
    analysis_layout->addWidget(verification_progress_);

    layout->addWidget(analysis_group_);

    main_splitter_->addWidget(statistics_panel_);
}

void EpipolarChecker::connectSignals() {
    // Calibration and image loading
    connect(load_calibration_button_, &QPushButton::clicked, this, &EpipolarChecker::loadCalibrationFile);
    connect(load_images_button_, &QPushButton::clicked, this, &EpipolarChecker::loadStereoImages);

    // Point management
    connect(add_point_button_, &QPushButton::clicked, this, &EpipolarChecker::addPointPair);
    connect(remove_point_button_, &QPushButton::clicked, this, &EpipolarChecker::removeSelectedPoint);
    connect(clear_all_button_, &QPushButton::clicked, this, &EpipolarChecker::clearAllPoints);
    connect(auto_verify_button_, &QPushButton::clicked, this, &EpipolarChecker::runAutomaticVerification);

    // Image widget signals
    connect(left_image_widget_, &EpipolarImageWidget::pointClicked, this, &EpipolarChecker::onLeftImageClicked);
    connect(right_image_widget_, &EpipolarImageWidget::pointClicked, this, &EpipolarChecker::onRightImageClicked);
    connect(left_image_widget_, &EpipolarImageWidget::pointSelected, this, &EpipolarChecker::onPointSelected);
    connect(right_image_widget_, &EpipolarImageWidget::pointSelected, this, &EpipolarChecker::onPointSelected);

    // Visualization settings
    connect(show_lines_checkbox_, &QCheckBox::toggled, this, &EpipolarChecker::updateDisplaySettings);
    connect(show_points_checkbox_, &QCheckBox::toggled, this, &EpipolarChecker::updateDisplaySettings);
    connect(show_labels_checkbox_, &QCheckBox::toggled, this, &EpipolarChecker::updateDisplaySettings);
    connect(color_coding_checkbox_, &QCheckBox::toggled, this, &EpipolarChecker::updateDisplaySettings);
    connect(line_thickness_slider_, &QSlider::valueChanged, this, &EpipolarChecker::updateDisplaySettings);
    connect(error_threshold_spinbox_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &EpipolarChecker::onAccuracyThresholdChanged);

    // Live mode
    connect(live_mode_checkbox_, &QCheckBox::toggled, this, &EpipolarChecker::toggleLiveMode);

    // Export functions
    connect(export_report_button_, &QPushButton::clicked, this, &EpipolarChecker::exportReport);
    connect(save_points_button_, &QPushButton::clicked, [this]() {
        QString filename = QFileDialog::getSaveFileName(this, "Save Points", "", "JSON Files (*.json)");
        if (!filename.isEmpty()) {
            // Implementation for saving points would go here
        }
    });
    connect(load_points_button_, &QPushButton::clicked, [this]() {
        QString filename = QFileDialog::getOpenFileName(this, "Load Points", "", "JSON Files (*.json)");
        if (!filename.isEmpty()) {
            loadGroundTruthPoints(filename);
        }
    });

    // Table selection
    connect(points_table_, &QTableWidget::itemSelectionChanged, [this]() {
        int current_row = points_table_->currentRow();
        if (current_row >= 0 && current_row < epipolar_points_.size()) {
            selected_point_id_ = epipolar_points_[current_row].point_id;
            updateVisualization();
        }
    });
}

void EpipolarChecker::setCalibration(const stereo_vision::CameraCalibration::StereoParameters& stereo_params) {
    stereo_params_ = stereo_params;
    has_calibration_ = true;

    // Compute fundamental matrix from stereo parameters
    cv::Mat K1 = stereo_params.left_camera.camera_matrix;
    cv::Mat K2 = stereo_params.right_camera.camera_matrix;
    cv::Mat R = stereo_params.R;
    cv::Mat t = stereo_params.T;

    // Essential matrix E = [t]_x * R
    cv::Mat t_cross = (cv::Mat_<double>(3, 3) <<
        0, -t.at<double>(2), t.at<double>(1),
        t.at<double>(2), 0, -t.at<double>(0),
        -t.at<double>(1), t.at<double>(0), 0);

    essential_matrix_ = t_cross * R;
    fundamental_matrix_ = stereo_params.F;

    has_valid_fundamental_matrix_ = true;

    // Update UI
    calibration_status_label_->setText(QString("Calibration loaded\nFocal length: %.1f, %.1f\nBaseline: %.1f mm")
        .arg(K1.at<double>(0,0)).arg(K1.at<double>(1,1)).arg(cv::norm(t) * 1000));
    calibration_status_label_->setStyleSheet("color: green; font-style: normal;");

    updateUI();
}

void EpipolarChecker::setStereoImages(const cv::Mat& left_image, const cv::Mat& right_image) {
    left_image_ = left_image.clone();
    right_image_ = right_image.clone();
    has_images_ = true;

    left_image_widget_->setImage(left_image_);
    right_image_widget_->setImage(right_image_);

    updateUI();

    if (live_mode_enabled_) {
        updateVisualization();
    }
}

void EpipolarChecker::updateUI() {
    bool can_add_points = has_calibration_ && has_images_ && has_valid_fundamental_matrix_;

    add_point_button_->setEnabled(can_add_points);
    auto_verify_button_->setEnabled(can_add_points);

    bool has_points = !epipolar_points_.empty();
    remove_point_button_->setEnabled(has_points && selected_point_id_ >= 0);
    clear_all_button_->setEnabled(has_points);
    export_report_button_->setEnabled(has_points);
    save_points_button_->setEnabled(has_points);
}

void EpipolarChecker::loadStereoImages() {
    QString left_file = QFileDialog::getOpenFileName(this, "Select Left Image", "",
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (left_file.isEmpty()) return;

    QString right_file = QFileDialog::getOpenFileName(this, "Select Right Image", "",
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (right_file.isEmpty()) return;

    cv::Mat left = cv::imread(left_file.toStdString());
    cv::Mat right = cv::imread(right_file.toStdString());

    if (left.empty() || right.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load images");
        return;
    }

    setStereoImages(left, right);
}

void EpipolarChecker::loadCalibrationFile() {
    QString filename = QFileDialog::getOpenFileName(this, "Load Calibration", "",
        "YAML Files (*.yml *.yaml);;XML Files (*.xml)");
    if (filename.isEmpty()) return;

    try {
        cv::FileStorage fs(filename.toStdString(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            QMessageBox::warning(this, "Error", "Failed to open calibration file");
            return;
        }

        stereo_vision::CameraCalibration::StereoParameters params;

        fs["left_camera_matrix"] >> params.left_camera.camera_matrix;
        fs["right_camera_matrix"] >> params.right_camera.camera_matrix;
        fs["left_distortion_coeffs"] >> params.left_camera.distortion_coeffs;
        fs["right_distortion_coeffs"] >> params.right_camera.distortion_coeffs;
        fs["rotation_matrix"] >> params.R;
        fs["translation_vector"] >> params.T;
        fs["essential_matrix"] >> params.E;
        fs["fundamental_matrix"] >> params.F;

        fs.release();

        setCalibration(params);

    } catch (const cv::Exception& e) {
        QMessageBox::warning(this, "Error", QString("Failed to load calibration: %1").arg(e.what()));
    }
}

void EpipolarChecker::onLeftImageClicked(const cv::Point2f& point) {
    if (!has_calibration_ || !has_valid_fundamental_matrix_) return;

    // Store pending left point for pair creation
    pending_left_point_ = point;

    // Compute and show epipolar line in right image
    cv::Vec3f epipolar_line;
    computeEpipolarLine(point, epipolar_line);

    std::vector<cv::Vec3f> lines = {epipolar_line};
    right_image_widget_->setEpipolarLines(lines);

    // Enable point pair addition
    add_point_button_->setEnabled(true);
    add_point_button_->setText("Add Point Pair");
    add_point_button_->setStyleSheet("QPushButton { background-color: lightgreen; }");
}

void EpipolarChecker::onRightImageClicked(const cv::Point2f& point) {
    if (!has_calibration_ || !has_valid_fundamental_matrix_) return;

    // If we have a pending left point, create a point pair
    if (pending_left_point_.x >= 0) {
        EpipolarPoint ep_point;
        ep_point.left_point = pending_left_point_;
        ep_point.right_point = point;
        ep_point.point_id = next_point_id_++;
        ep_point.is_ground_truth = false;
        ep_point.is_valid = true;

        // Compute epipolar error
        computeEpipolarError(ep_point.left_point, ep_point.right_point, ep_point.epipolar_error);

        // Compute predicted right point on epipolar line
        cv::Vec3f line;
        computeEpipolarLine(ep_point.left_point, line);
        ep_point.predicted_right_point = findClosestPointOnLine(line, ep_point.right_point);

        epipolar_points_.push_back(ep_point);

        // Reset pending point
        pending_left_point_ = cv::Point2f(-1, -1);
        add_point_button_->setEnabled(has_calibration_ && has_images_);
        add_point_button_->setText("Add Point Pair");
        add_point_button_->setStyleSheet("");

        updateEpipolarLines();
        updateStatistics();
        updateVisualization();
        updateUI();
    }
}

void EpipolarChecker::addPointPair() {
    // This is triggered when user wants to add a point pair manually
    // The actual addition happens in onRightImageClicked after left click
    QMessageBox::information(this, "Add Point Pair",
        "Click on a point in the left image, then click the corresponding point in the right image.");
}

void EpipolarChecker::removeSelectedPoint() {
    if (selected_point_id_ < 0) return;

    auto it = std::find_if(epipolar_points_.begin(), epipolar_points_.end(),
        [this](const EpipolarPoint& p) { return p.point_id == selected_point_id_; });

    if (it != epipolar_points_.end()) {
        epipolar_points_.erase(it);
        selected_point_id_ = -1;

        updateEpipolarLines();
        updateStatistics();
        updateVisualization();
        updateUI();
    }
}

void EpipolarChecker::clearAllPoints() {
    epipolar_points_.clear();
    ground_truth_points_.clear();
    selected_point_id_ = -1;
    next_point_id_ = 1;

    left_image_widget_->clearEpipolarData();
    right_image_widget_->clearEpipolarData();

    current_statistics_.clear();
    updateStatistics();
    updateUI();
}

void EpipolarChecker::computeEpipolarLine(const cv::Point2f& left_point, cv::Vec3f& epipolar_line) {
    if (!has_valid_fundamental_matrix_) return;

    cv::Mat point_homogeneous = (cv::Mat_<double>(3, 1) << left_point.x, left_point.y, 1.0);
    cv::Mat line_coeffs = fundamental_matrix_ * point_homogeneous;

    epipolar_line[0] = line_coeffs.at<double>(0);
    epipolar_line[1] = line_coeffs.at<double>(1);
    epipolar_line[2] = line_coeffs.at<double>(2);
}

void EpipolarChecker::computeEpipolarError(const cv::Point2f& left_point, const cv::Point2f& right_point, double& error) {
    cv::Vec3f line;
    computeEpipolarLine(left_point, line);
    error = distanceToLine(line, right_point);
}

double EpipolarChecker::distanceToLine(const cv::Vec3f& line, const cv::Point2f& point) {
    double a = line[0], b = line[1], c = line[2];
    return std::abs(a * point.x + b * point.y + c) / std::sqrt(a * a + b * b);
}

cv::Point2f EpipolarChecker::findClosestPointOnLine(const cv::Vec3f& line, const cv::Point2f& point) {
    double a = line[0], b = line[1], c = line[2];

    // Parametric form of the perpendicular from point to line
    double denom = a * a + b * b;
    if (denom < 1e-6) return point;

    double x = (b * (b * point.x - a * point.y) - a * c) / denom;
    double y = (a * (-b * point.x + a * point.y) - b * c) / denom;

    return cv::Point2f(x, y);
}

void EpipolarChecker::updateEpipolarLines() {
    if (!has_valid_fundamental_matrix_ || epipolar_points_.empty()) return;

    std::vector<cv::Vec3f> lines;
    for (const auto& point : epipolar_points_) {
        cv::Vec3f line;
        computeEpipolarLine(point.left_point, line);
        lines.push_back(line);
    }

    right_image_widget_->setEpipolarLines(lines);
}

void EpipolarChecker::updateStatistics() {
    if (epipolar_points_.empty()) {
        current_statistics_.clear();
        points_table_->setRowCount(0);
        statistics_text_->clear();
        return;
    }

    // Extract errors
    std::vector<double> errors = extractErrors();

    // Calculate statistics
    current_statistics_.total_points = epipolar_points_.size();
    current_statistics_.valid_points = std::count_if(epipolar_points_.begin(), epipolar_points_.end(),
        [](const EpipolarPoint& p) { return p.is_valid; });

    current_statistics_.accuracy_threshold = accuracy_threshold_;
    current_statistics_.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

    // Calculate median
    std::vector<double> sorted_errors = errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    size_t n = sorted_errors.size();
    current_statistics_.median_error = (n % 2 == 0) ?
        (sorted_errors[n/2-1] + sorted_errors[n/2]) / 2.0 : sorted_errors[n/2];

    current_statistics_.min_error = *std::min_element(errors.begin(), errors.end());
    current_statistics_.max_error = *std::max_element(errors.begin(), errors.end());

    // Calculate standard deviation
    double variance = 0.0;
    for (double error : errors) {
        variance += (error - current_statistics_.mean_error) * (error - current_statistics_.mean_error);
    }
    current_statistics_.std_deviation = std::sqrt(variance / errors.size());

    // Calculate RMS error
    double sum_squared = 0.0;
    for (double error : errors) {
        sum_squared += error * error;
    }
    current_statistics_.rms_error = std::sqrt(sum_squared / errors.size());

    // Calculate accuracy percentage
    int accurate_points = std::count_if(errors.begin(), errors.end(),
        [this](double error) { return error <= accuracy_threshold_; });
    current_statistics_.accuracy_percentage = (100.0 * accurate_points) / errors.size();

    // Update points table
    updatePointsTable();

    // Update statistics text
    updateStatisticsText();
}

void EpipolarChecker::updatePointsTable() {
    points_table_->setRowCount(epipolar_points_.size());

    for (size_t i = 0; i < epipolar_points_.size(); ++i) {
        const auto& point = epipolar_points_[i];

        points_table_->setItem(i, 0, new QTableWidgetItem(QString::number(point.point_id)));
        points_table_->setItem(i, 1, new QTableWidgetItem(QString::number(point.left_point.x, 'f', 1)));
        points_table_->setItem(i, 2, new QTableWidgetItem(QString::number(point.left_point.y, 'f', 1)));
        points_table_->setItem(i, 3, new QTableWidgetItem(QString::number(point.right_point.x, 'f', 1)));
        points_table_->setItem(i, 4, new QTableWidgetItem(QString::number(point.right_point.y, 'f', 1)));

        auto* error_item = new QTableWidgetItem(QString::number(point.epipolar_error, 'f', 3));
        if (point.epipolar_error <= accuracy_threshold_) {
            error_item->setBackground(QColor(200, 255, 200)); // Light green
        } else if (point.epipolar_error <= accuracy_threshold_ * 2) {
            error_item->setBackground(QColor(255, 255, 200)); // Light yellow
        } else {
            error_item->setBackground(QColor(255, 200, 200)); // Light red
        }
        points_table_->setItem(i, 5, error_item);
    }
}

void EpipolarChecker::updateStatisticsText() {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "EPIPOLAR LINE ACCURACY REPORT\n";
    ss << "==============================\n\n";

    ss << "Dataset Summary:\n";
    ss << "  Total Points: " << current_statistics_.total_points << "\n";
    ss << "  Valid Points: " << current_statistics_.valid_points << "\n";
    ss << "  Accuracy Threshold: " << current_statistics_.accuracy_threshold << " pixels\n\n";

    ss << "Error Statistics:\n";
    ss << "  Mean Error: " << current_statistics_.mean_error << " px\n";
    ss << "  Median Error: " << current_statistics_.median_error << " px\n";
    ss << "  RMS Error: " << current_statistics_.rms_error << " px\n";
    ss << "  Std Deviation: " << current_statistics_.std_deviation << " px\n";
    ss << "  Min Error: " << current_statistics_.min_error << " px\n";
    ss << "  Max Error: " << current_statistics_.max_error << " px\n\n";

    ss << "Calibration Quality:\n";
    ss << "  Accuracy: " << std::setprecision(1) << current_statistics_.accuracy_percentage << "%\n";

    if (current_statistics_.accuracy_percentage >= 95.0) {
        ss << "  Assessment: EXCELLENT\n";
    } else if (current_statistics_.accuracy_percentage >= 85.0) {
        ss << "  Assessment: GOOD\n";
    } else if (current_statistics_.accuracy_percentage >= 70.0) {
        ss << "  Assessment: ACCEPTABLE\n";
    } else {
        ss << "  Assessment: POOR - Recalibration recommended\n";
    }

    statistics_text_->setText(QString::fromStdString(ss.str()));
}

std::vector<double> EpipolarChecker::extractErrors() const {
    std::vector<double> errors;
    errors.reserve(epipolar_points_.size());

    for (const auto& point : epipolar_points_) {
        if (point.is_valid) {
            errors.push_back(point.epipolar_error);
        }
    }

    return errors;
}

void EpipolarChecker::updateVisualization() {
    if (!has_images_) return;

    // Update point display
    left_image_widget_->setEpipolarPoints(epipolar_points_);
    right_image_widget_->setEpipolarPoints(epipolar_points_);

    // Update epipolar lines
    updateEpipolarLines();
}

void EpipolarChecker::updateDisplaySettings() {
    left_image_widget_->setShowPointLabels(show_labels_checkbox_->isChecked());
    right_image_widget_->setShowPointLabels(show_labels_checkbox_->isChecked());

    left_image_widget_->setColorCoding(color_coding_checkbox_->isChecked());
    right_image_widget_->setColorCoding(color_coding_checkbox_->isChecked());

    left_image_widget_->setLineThickness(line_thickness_slider_->value());
    right_image_widget_->setLineThickness(line_thickness_slider_->value());

    left_image_widget_->setErrorThreshold(error_threshold_spinbox_->value());
    right_image_widget_->setErrorThreshold(error_threshold_spinbox_->value());
}

void EpipolarChecker::onAccuracyThresholdChanged() {
    accuracy_threshold_ = error_threshold_spinbox_->value();
    updateStatistics();
    updateDisplaySettings();
}

void EpipolarChecker::toggleLiveMode() {
    live_mode_enabled_ = live_mode_checkbox_->isChecked();

    if (live_mode_enabled_) {
        live_update_timer_->start();
    } else {
        live_update_timer_->stop();
    }
}

void EpipolarChecker::onPointSelected(int pointId) {
    selected_point_id_ = pointId;
    updateUI();

    // Highlight row in table
    for (int row = 0; row < points_table_->rowCount(); ++row) {
        if (points_table_->item(row, 0)->text().toInt() == pointId) {
            points_table_->selectRow(row);
            break;
        }
    }
}

void EpipolarChecker::exportReport() {
    QString filename = QFileDialog::getSaveFileName(this, "Export Verification Report",
        "epipolar_verification_report.txt", "Text Files (*.txt)");
    if (filename.isEmpty()) return;

    saveVerificationReport(filename);
}

void EpipolarChecker::saveVerificationReport(const QString& filename) {
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "Failed to save report");
        return;
    }

    QTextStream out(&file);
    out << statistics_text_->toPlainText();

    out << "\n\nDetailed Point Data:\n";
    out << "====================\n";
    out << "ID\tLeft_X\tLeft_Y\tRight_X\tRight_Y\tError\tStatus\n";

    for (const auto& point : epipolar_points_) {
        out << point.point_id << "\t"
            << QString::number(point.left_point.x, 'f', 2) << "\t"
            << QString::number(point.left_point.y, 'f', 2) << "\t"
            << QString::number(point.right_point.x, 'f', 2) << "\t"
            << QString::number(point.right_point.y, 'f', 2) << "\t"
            << QString::number(point.epipolar_error, 'f', 3) << "\t"
            << (point.epipolar_error <= accuracy_threshold_ ? "GOOD" : "POOR") << "\n";
    }

    emit verificationReportGenerated(statistics_text_->toPlainText());
}

void EpipolarChecker::loadGroundTruthPoints(const QString& filename) {
    // Implementation for loading ground truth points from JSON would go here
    QMessageBox::information(this, "Load Ground Truth",
        "Ground truth loading feature will be implemented based on your specific format requirements.");
}

void EpipolarChecker::runAutomaticVerification() {
    QMessageBox::information(this, "Automatic Verification",
        "Automatic verification will detect and verify feature points across stereo pairs.\n"
        "This feature requires additional computer vision algorithms and will be implemented as needed.");
}

void EpipolarChecker::onParameterChanged() {
    // Update the epipolar line visualization when parameters change
    updateEpipolarLines();
    updateVisualization();
}

void EpipolarChecker::calculateStatistics() {
    // Update statistics display and calculations
    updateStatistics();
    updateStatisticsText();
}

void EpipolarChecker::onVisualizationSettingsChanged() {
    // Update visualization when display settings change
    updateDisplaySettings();
    updateVisualization();
}

} // namespace stereo_vision::gui
