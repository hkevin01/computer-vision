#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QTableWidget>
#include <QProgressBar>
#include <QGroupBox>
#include <QSplitter>
#include <QScrollArea>
#include <QTextEdit>
#include <QSlider>
#include <QTimer>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QFileDialog>
#include <QMessageBox>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "camera_calibration.hpp"
#include "enhanced_image_widget.hpp"

namespace stereo_vision::gui {

/**
 * @brief Structure to hold epipolar line verification data
 */
struct EpipolarPoint {
    cv::Point2f left_point;           ///< Point in left image
    cv::Point2f right_point;          ///< Corresponding point in right image
    cv::Point2f predicted_right_point; ///< Predicted point from epipolar geometry
    double epipolar_error;            ///< Distance from epipolar line to actual point
    bool is_ground_truth;             ///< Whether this is a ground truth correspondence
    bool is_valid;                    ///< Whether the correspondence is considered valid
    int point_id;                     ///< Unique identifier for the point pair
};

/**
 * @brief Statistics for epipolar line accuracy assessment
 */
struct EpipolarStatistics {
    double mean_error;                ///< Mean epipolar error in pixels
    double median_error;              ///< Median epipolar error in pixels
    double std_deviation;             ///< Standard deviation of errors
    double max_error;                 ///< Maximum error observed
    double min_error;                 ///< Minimum error observed
    double rms_error;                 ///< Root mean square error
    int total_points;                 ///< Total number of point pairs
    int valid_points;                 ///< Number of valid correspondences
    double accuracy_threshold;        ///< Threshold for considering a point accurate
    double accuracy_percentage;       ///< Percentage of points within threshold

    void clear() {
        mean_error = median_error = std_deviation = 0.0;
        max_error = min_error = rms_error = 0.0;
        total_points = valid_points = 0;
        accuracy_threshold = 1.0;
        accuracy_percentage = 0.0;
    }
};

/**
 * @brief Custom image widget for interactive epipolar line visualization
 */
class EpipolarImageWidget : public EnhancedImageWidget {
    Q_OBJECT

public:
    explicit EpipolarImageWidget(QWidget* parent = nullptr);

    void setImage(const cv::Mat& image);
    void setEpipolarLines(const std::vector<cv::Vec3f>& lines);
    void setEpipolarPoints(const std::vector<EpipolarPoint>& points);
    void setIsLeftImage(bool isLeft) { is_left_image_ = isLeft; }
    void setLineThickness(int thickness) { line_thickness_ = thickness; update(); }
    void setShowPointLabels(bool show) { show_point_labels_ = show; update(); }
    void setColorCoding(bool enable) { color_coding_enabled_ = enable; update(); }
    void setErrorThreshold(double threshold) { error_threshold_ = threshold; update(); }

    void clearEpipolarData();
    cv::Point2f getLastClickedPoint() const { return last_clicked_point_; }

signals:
    void pointClicked(const cv::Point2f& point);
    void pointSelected(int pointId);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void drawEpipolarLines(QPainter& painter);
    void drawEpipolarPoints(QPainter& painter);
    void drawPointLabels(QPainter& painter);
    QColor getErrorColor(double error) const;
    cv::Point2f screenToImage(const QPoint& screen_point) const;
    QPoint imageToScreen(const cv::Point2f& image_point) const;

    cv::Mat current_image_;
    std::vector<cv::Vec3f> epipolar_lines_;
    std::vector<EpipolarPoint> epipolar_points_;
    cv::Point2f last_clicked_point_;

    bool is_left_image_;
    bool show_point_labels_;
    bool color_coding_enabled_;
    int line_thickness_;
    double error_threshold_;
    int selected_point_id_;
};

/**
 * @brief Main epipolar line checker widget
 */
class EpipolarChecker : public QWidget {
    Q_OBJECT

public:
    explicit EpipolarChecker(QWidget* parent = nullptr);
    ~EpipolarChecker();

    void setCalibration(const stereo_vision::CameraCalibration::StereoParameters& stereo_params);
    void setStereoImages(const cv::Mat& left_image, const cv::Mat& right_image);
    void loadGroundTruthPoints(const QString& filename);
    void saveVerificationReport(const QString& filename);

public slots:
    void clearAllPoints();
    void loadStereoImages();
    void loadCalibrationFile();
    void exportReport();
    void runAutomaticVerification();
    void onLeftImageClicked(const cv::Point2f& point);
    void onRightImageClicked(const cv::Point2f& point);
    void onPointSelected(int pointId);
    void updateVisualization();
    void onParameterChanged();
    void toggleLiveMode();

signals:
    void calibrationQualityAssessed(double accuracy_percentage, double mean_error);
    void verificationReportGenerated(const QString& report_text);

private slots:
    void addPointPair();
    void removeSelectedPoint();
    void calculateStatistics();
    void updateDisplaySettings();
    void onAccuracyThresholdChanged();
    void onVisualizationSettingsChanged();

private:
    void setupUI();
    void setupImageDisplays();
    void setupControlPanel();
    void setupStatisticsPanel();
    void setupMenuActions();
    void connectSignals();
    void updateUI();

    void computeEpipolarLine(const cv::Point2f& left_point, cv::Vec3f& epipolar_line);
    void computeEpipolarError(const cv::Point2f& left_point, const cv::Point2f& right_point, double& error);
    void updateEpipolarLines();
    void updateStatistics();
    void updatePointsTable();
    void updateStatisticsText();
    void generateVerificationReport();
    void resetVerification();

    cv::Point2f findClosestPointOnLine(const cv::Vec3f& line, const cv::Point2f& point);
    double distanceToLine(const cv::Vec3f& line, const cv::Point2f& point);
    std::vector<double> extractErrors() const;

    // UI Components
    QSplitter* main_splitter_;
    QSplitter* image_splitter_;
    QWidget* control_panel_;
    QWidget* statistics_panel_;

    // Image display widgets
    EpipolarImageWidget* left_image_widget_;
    EpipolarImageWidget* right_image_widget_;
    QScrollArea* left_scroll_area_;
    QScrollArea* right_scroll_area_;

    // Control widgets
    QGroupBox* calibration_group_;
    QLabel* calibration_status_label_;
    QPushButton* load_calibration_button_;
    QPushButton* load_images_button_;

    QGroupBox* point_management_group_;
    QPushButton* add_point_button_;
    QPushButton* remove_point_button_;
    QPushButton* clear_all_button_;
    QPushButton* auto_verify_button_;

    QGroupBox* visualization_group_;
    QCheckBox* show_lines_checkbox_;
    QCheckBox* show_points_checkbox_;
    QCheckBox* show_labels_checkbox_;
    QCheckBox* color_coding_checkbox_;
    QSlider* line_thickness_slider_;
    QDoubleSpinBox* error_threshold_spinbox_;

    QGroupBox* analysis_group_;
    QTableWidget* points_table_;
    QTextEdit* statistics_text_;
    QProgressBar* verification_progress_;

    // Export and reporting
    QPushButton* export_report_button_;
    QPushButton* save_points_button_;
    QPushButton* load_points_button_;

    // Live mode
    QCheckBox* live_mode_checkbox_;
    QTimer* live_update_timer_;

    // Data members
    cv::Mat left_image_;
    cv::Mat right_image_;
    stereo_vision::CameraCalibration::StereoParameters stereo_params_;
    cv::Mat fundamental_matrix_;
    cv::Mat essential_matrix_;

    std::vector<EpipolarPoint> epipolar_points_;
    EpipolarStatistics current_statistics_;

    bool has_calibration_;
    bool has_images_;
    bool has_valid_fundamental_matrix_;
    bool live_mode_enabled_;

    int next_point_id_;
    int selected_point_id_;

    // Ground truth data
    QString ground_truth_filename_;
    std::vector<EpipolarPoint> ground_truth_points_;

    // Verification settings
    double accuracy_threshold_;
    bool auto_compute_statistics_;
    int max_epipolar_points_;
    cv::Point2f pending_left_point_;
};

} // namespace stereo_vision::gui
