#include "point_cloud_processor.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <iostream>

namespace stereo_vision {

PointCloudProcessor::PointCloudProcessor() {}

PointCloudProcessor::~PointCloudProcessor() {}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcessor::filterPointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    double leaf_size,
    int mean_k,
    double std_dev_thresh) {
    
    // Downsample the point cloud
    auto downsampled = downsample(input_cloud, leaf_size);
    
    // Remove outliers
    auto filtered = removeOutliers(downsampled, mean_k, std_dev_thresh);
    
    return filtered;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcessor::downsample(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    double leaf_size) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*filtered);
    
    return filtered;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcessor::removeOutliers(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    int mean_k,
    double std_dev_thresh) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outlier_filter;
    outlier_filter.setInputCloud(cloud);
    outlier_filter.setMeanK(mean_k);
    outlier_filter.setStddevMulThresh(std_dev_thresh);
    outlier_filter.filter(*filtered);
    
    return filtered;
}

bool PointCloudProcessor::exportPointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const std::string& filename,
    ExportFormat format) {
    
    try {
        switch (format) {
            case ExportFormat::PLY_BINARY:
                return pcl::io::savePLYFileBinary(filename, *cloud) == 0;
            case ExportFormat::PLY_ASCII:
                return pcl::io::savePLYFileASCII(filename, *cloud) == 0;
            case ExportFormat::PCD_BINARY:
                return pcl::io::savePCDFileBinary(filename, *cloud) == 0;
            case ExportFormat::PCD_ASCII:
                return pcl::io::savePCDFileASCII(filename, *cloud) == 0;
            default:
                std::cerr << "Unsupported export format" << std::endl;
                return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving point cloud: " << e.what() << std::endl;
        return false;
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcessor::loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Determine file format from extension
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    
    try {
        if (extension == "ply") {
            if (pcl::io::loadPLYFile(filename, *cloud) != 0) {
                return nullptr;
            }
        } else if (extension == "pcd") {
            if (pcl::io::loadPCDFile(filename, *cloud) != 0) {
                return nullptr;
            }
        } else {
            std::cerr << "Unsupported file format: " << extension << std::endl;
            return nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading point cloud: " << e.what() << std::endl;
        return nullptr;
    }
    
    return cloud;
}

PointCloudProcessor::CloudStatistics PointCloudProcessor::computeStatistics(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    
    CloudStatistics stats;
    stats.num_points = cloud->size();
    
    if (stats.num_points == 0) {
        return stats;
    }
    
    // Initialize min/max with first point
    stats.min_point = cloud->points[0];
    stats.max_point = cloud->points[0];
    
    // Compute centroid and bounds
    double sum_x = 0, sum_y = 0, sum_z = 0;
    
    for (const auto& point : cloud->points) {
        sum_x += point.x;
        sum_y += point.y;
        sum_z += point.z;
        
        if (point.x < stats.min_point.x) stats.min_point.x = point.x;
        if (point.y < stats.min_point.y) stats.min_point.y = point.y;
        if (point.z < stats.min_point.z) stats.min_point.z = point.z;
        
        if (point.x > stats.max_point.x) stats.max_point.x = point.x;
        if (point.y > stats.max_point.y) stats.max_point.y = point.y;
        if (point.z > stats.max_point.z) stats.max_point.z = point.z;
    }
    
    stats.centroid.x = sum_x / stats.num_points;
    stats.centroid.y = sum_y / stats.num_points;
    stats.centroid.z = sum_z / stats.num_points;
    
    // Compute bounding box volume
    double width = stats.max_point.x - stats.min_point.x;
    double height = stats.max_point.y - stats.min_point.y;
    double depth = stats.max_point.z - stats.min_point.z;
    stats.bounding_box_volume = width * height * depth;
    
    return stats;
}

} // namespace stereo_vision
