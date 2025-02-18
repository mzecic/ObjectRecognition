/*
    Matej Zecic
    Spring 2025 - CS5330
    This file contains preprocessing techniques used to prepare the image for object recognition.
*/

#include <opencv2/opencv.hpp>
#include "../include/objectRecPreprocessing.h"
#include <queue>
#include <set>
#include <utility>
#include <iostream>
#include <unordered_set>

// -----------------------
// Preprocessing Functions
// -----------------------

// 1. Thresholding: Separate the object (assumed darker) from the white background.
void applyThresholding(const cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    int threshold_value = 140;

     // Process each pixel
     for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* src_row = src.ptr<cv::Vec3b>(i);
        uchar* dst_row = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; j++) {
            // Get BGR values
            int blue = src_row[j][0];
            int green = src_row[j][1];
            int red = src_row[j][2];

            // Simple average for brightness
            int brightness = (red + green + blue) / 3;

            // Apply threshold - if brightness is greater than threshold, it's background
            dst_row[j] = (brightness > threshold_value) ? 0 : 255;
        }
    }
}

// 2. Dilation: Clean up the binary image (fill holes, reduce noise).
void applyDilation(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::dilate(src, dst, element);
}

void applyMorphologicalFiltering(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat opened, closed;

    // Create a structuring element.
    // You can adjust the size (here 5x5) based on your image resolution and noise.
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Apply Opening: removes small noise
    cv::morphologyEx(src, opened, cv::MORPH_OPEN, element);

    // Apply Closing: fills small holes inside objects
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, element);

    dst = closed;
}

// 3. Connected Components: Segment the binary image into regions.
//    This function filters out small regions and those that touch the image edges.
void applyConnectedComponents(const cv::Mat &binary_image, cv::Mat &dst, cv::Mat &labels) {
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_image, labels, stats, centroids);

    // Minimum region size (adjust this value based on your needs)
    int min_size = 500;  // Increased from previous value

    // Create color output
    dst = cv::Mat::zeros(binary_image.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(num_labels);
    colors[0] = cv::Vec3b(0, 0, 0); // Background

    // Keep track of valid regions and their mapping
    std::vector<int> valid_regions;

    for (int i = 1; i < num_labels; i++) {
        colors[i] = cv::Vec3b(rand()%256, rand()%256, rand()%256);

        // Get region size
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Skip small regions
        if (area < min_size) {
            colors[i] = cv::Vec3b(0, 0, 0);  // Make small regions black
            // Also remove them from labels
            for(int y = 0; y < labels.rows; y++) {
                for(int x = 0; x < labels.cols; x++) {
                    if(labels.at<int>(y,x) == i) {
                        labels.at<int>(y,x) = 0;
                    }
                }
            }
        } else {
            valid_regions.push_back(i);
        }
    }

    // Color the regions and draw labels
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            int label = labels.at<int>(y, x);
            dst.at<cv::Vec3b>(y, x) = colors[label];
        }
    }

    // Draw region numbers at centroids
    for(size_t i = 0; i < valid_regions.size(); i++) {
        int region_id = valid_regions[i];
        cv::Point center(centroids.at<double>(region_id, 0), centroids.at<double>(region_id, 1));

        // Get bounding box for this region
        int x = stats.at<int>(region_id, cv::CC_STAT_LEFT);
        int y = stats.at<int>(region_id, cv::CC_STAT_TOP);
        int area = stats.at<int>(region_id, cv::CC_STAT_AREA);

        // Draw the region ID
        std::string label = std::to_string(region_id);
        cv::putText(dst, label, center, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 2);

        std::cout << "Region " << region_id << " at position ("
                  << center.x << "," << center.y << ") with area "
                  << area << " pixels" << std::endl;
    }

    // Print number of regions above minimum size
    std::cout << "Found " << valid_regions.size() << " valid regions" << std::endl;
}

// 4. Compute Features for a Region:
//    Given a label map and a region ID, compute region-based features and overlay them.
std::pair<std::vector<double>, cv::Mat> compute_features(const cv::Mat &labels, int region_id) {
    // Create a binary mask for the region
    cv::Mat region_mask = (labels == region_id);

    // Get contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(region_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return {{0, 0, 0}, cv::Mat()};
    }

    // Get the largest contour
    auto largest_contour = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });

    // Get contour area
    double contour_area = cv::contourArea(*largest_contour);

    // Get rotated rectangle and its properties
    cv::RotatedRect rRect = cv::minAreaRect(*largest_contour);
    double width = rRect.size.width;
    double height = rRect.size.height;
    double bbox_area = width * height;

    // Calculate percent filled using contour area
    double percent_filled = contour_area / bbox_area;

    // Ensure percent_filled doesn't exceed 1.0
    percent_filled = std::min(1.0, percent_filled);

    // Calculate aspect ratio (always between 0 and 1)
    double aspect_ratio = std::min(width, height) / std::max(width, height);

    // Calculate perimeter and circularity (new shape feature)
    double perimeter = cv::arcLength(*largest_contour, true);
    double circularity = 4 * CV_PI * contour_area / (perimeter * perimeter);

    std::cout << "\nFeature Calculations:" << std::endl;
    std::cout << "Area: " << contour_area << " pixels" << std::endl;
    std::cout << "Perimeter: " << perimeter << " pixels" << std::endl;
    std::cout << "Bounding Box: " << width << " x " << height << " pixels" << std::endl;
    std::cout << "Percent Filled: " << percent_filled << std::endl;
    std::cout << "Aspect Ratio: " << aspect_ratio << std::endl;
    std::cout << "Circularity: " << circularity << std::endl;

    // Create visualization
    cv::Mat visualization;
    cv::cvtColor(region_mask, visualization, cv::COLOR_GRAY2BGR);

    // Draw rotated rectangle
    cv::Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(visualization, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
    }

    // Return percent_filled, aspect_ratio, and circularity instead of angle
    std::vector<double> features = {percent_filled, aspect_ratio, circularity};
    return {features, visualization};
}

// Draw features for all valid regions on the given display image.
// 'labels' is the connected components label matrix,
// 'display' is an image (for example, the colorized connected components image) that we will draw on.
// 'minAreaThreshold' filters out very small regions.
// Draw features (oriented bounding box, axis, and centroid) for all valid regions in the label map.
// Regions with an area smaller than minAreaThreshold or whose bounding rectangle touches the image edge (within edgeMargin) are skipped.
void drawAllFeatures(const cv::Mat &labels, cv::Mat &display, double minAreaThreshold = 1000, int edgeMargin = 10) {
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    // Iterate over all region IDs (skip 0, which is the background)
    for (int region_id = 1; region_id <= maxVal; region_id++) {
        // Create a binary mask for the region.
        cv::Mat region_mask = (labels == region_id);
        // Compute moments to determine area and centroid.
        cv::Moments m = cv::moments(region_mask, true);
        double area = m.m00;
        if (area < minAreaThreshold)
            continue; // Skip small regions.

        // Compute centroid.
        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;

        // Gather all points of the region.
        std::vector<cv::Point> points;
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                if (labels.at<int>(y, x) == region_id)
                    points.push_back(cv::Point(x, y));
            }
        }
        if (points.empty()) continue;

        // Compute the axis-aligned bounding box from the points.
        cv::Rect bbox = cv::boundingRect(points);
        // If the bounding box touches the image edge (within the margin), skip this region.
        if (bbox.x <= edgeMargin || bbox.y <= edgeMargin ||
            (bbox.x + bbox.width) >= (labels.cols - edgeMargin) ||
            (bbox.y + bbox.height) >= (labels.rows - edgeMargin))
        {
            continue;
        }

        // Compute the oriented (rotated) bounding box.
        cv::RotatedRect rRect = cv::minAreaRect(points);

        // Compute orientation from normalized central moments.
        double mu20 = m.mu20 / area;
        double mu02 = m.mu02 / area;
        double mu11 = m.mu11 / area;
        double theta = 0.5 * atan2(2 * mu11, mu20 - mu02); // in radians

        // Draw the oriented bounding box (green).
        cv::Point2f boxPoints[4];
        rRect.points(boxPoints);
        for (int i = 0; i < 4; i++) {
            cv::line(display, boxPoints[i], boxPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        // Draw the axis of least central moment (red line) from the centroid.
        int line_length = 50; // You can adjust this length.
        cv::Point axis_start(static_cast<int>(cx), static_cast<int>(cy));
        cv::Point axis_end(static_cast<int>(cx + line_length * cos(theta)),
                           static_cast<int>(cy + line_length * sin(theta)));
        cv::line(display, axis_start, axis_end, cv::Scalar(0, 0, 255), 2);

        // Draw region id number at the top-left corner of the object region box.
        char region_text[50];
        sprintf(region_text, "Region %d", region_id);
        cv::putText(display, region_text, cv::Point(rRect.boundingRect().x, rRect.boundingRect().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        // Draw the centroid (blue).
        cv::circle(display, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 4, cv::Scalar(255, 0, 0), -1);
    }
}
