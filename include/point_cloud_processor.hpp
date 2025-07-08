#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <memory>

namespace stereo_vision
{

    /**
     * @brief Point cloud processing and export utilities
     */
    class PointCloudProcessor
    {
    public:
        enum class ExportFormat
        {
            PLY_BINARY,
            PLY_ASCII,
            PCD_BINARY,
            PCD_ASCII,
            XYZ
        };

    public:
        PointCloudProcessor();
        ~PointCloudProcessor();

        /**
         * @brief Filter point cloud to remove noise and outliers
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterPointCloud(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
            double leaf_size = 0.01,
            int mean_k = 50,
            double std_dev_thresh = 1.0);

        /**
         * @brief Export point cloud to file
         */
        bool exportPointCloud(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            const std::string &filename,
            ExportFormat format = ExportFormat::PLY_BINARY);

        /**
         * @brief Load point cloud from file
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string &filename);

        /**
         * @brief Create PCL visualizer for point cloud display
         */
        std::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer(
            const std::string &window_name = "Point Cloud Viewer");

        /**
         * @brief Display point cloud in visualizer
         */
        void displayPointCloud(
            std::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            const std::string &id = "cloud");

        /**
         * @brief Compute point cloud statistics
         */
        struct CloudStatistics
        {
            size_t num_points;
            pcl::PointXYZRGB min_point;
            pcl::PointXYZRGB max_point;
            pcl::PointXYZRGB centroid;
            double bounding_box_volume;
        };

        CloudStatistics computeStatistics(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    private:
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeOutliers(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            int mean_k,
            double std_dev_thresh);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsample(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            double leaf_size);
    };

} // namespace stereo_vision
