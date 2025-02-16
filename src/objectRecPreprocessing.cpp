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
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int brightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
            // Invert threshold: if brightness > 100, it's background (0), else object (255).
            dst.at<uchar>(i, j) = (brightness > 100) ? 0 : 255;
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
    std::cout << "Total regions found (including background): " << num_labels << std::endl;

    // Identify regions that touch the image boundaries.
    std::unordered_set<int> edge_regions;
    int rows = binary_image.rows, cols = binary_image.cols;
    // Define a margin (in pixels)
    int edgeMargin = 50;

    for (int i = 1; i < num_labels; i++) { // Skip background (label 0)
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        // If the bounding box comes within edgeMargin of any border, mark it.
        if (x <= edgeMargin || y <= edgeMargin || (x + w) >= (cols - edgeMargin) || (y + h) >= (rows - edgeMargin)) {
            edge_regions.insert(i);
        }
    }

    // Filter out small regions. (Adjust min_size as needed.)
    int min_size = 1000;
    std::vector<std::pair<int, int>> region_sizes;
    for (int i = 1; i < num_labels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_size && edge_regions.find(i) == edge_regions.end()) {
            region_sizes.push_back({area, i});
        }
    }
    std::cout << "Remaining valid regions after filtering: " << region_sizes.size() << std::endl;
    // Sort regions by size (largest first).
    std::sort(region_sizes.rbegin(), region_sizes.rend());

    // Create a color image for visualization.
    cv::Mat output(binary_image.size(), CV_8UC3, cv::Scalar(50, 50, 50)); // Start with a gray background.
    std::vector<cv::Vec3b> colors(num_labels);
    colors[0] = cv::Vec3b(0, 0, 0); // Background: black.
    for (int i = 1; i < num_labels; i++) {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    // Color the regions (skip those touching edges).
    for (int y = 0; y < binary_image.rows; y++) {
        for (int x = 0; x < binary_image.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0 && edge_regions.find(label) == edge_regions.end()) {
                output.at<cv::Vec3b>(y, x) = colors[label];
            }
        }
    }
    dst = output;
}

// 4. Compute Features for a Region:
//    Given a label map and a region ID, compute region-based features and overlay them.
std::pair<std::vector<double>, cv::Mat> compute_features(const cv::Mat &labels, int region_id) {
    // Validate region_id (skip background, which is label 0)
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    if (region_id <= 0 || region_id > maxVal) {
        std::cerr << "Warning: Region ID " << region_id << " is not valid." << std::endl;
        return { {0, 0, 0}, cv::Mat::zeros(labels.size(), CV_8UC3) };
    }

    // Create a binary mask for the region.
    cv::Mat region_mask = (labels == region_id);

    // Compute moments.
    cv::Moments m = cv::moments(region_mask, true);
    double area = m.m00;
    if (area < 1e-6) {
        std::cerr << "Warning: Region " << region_id << " has zero area." << std::endl;
        return { {0, 0, 0}, cv::Mat::zeros(labels.size(), CV_8UC3) };
    }

    // Compute centroid.
    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    // Compute normalized central moments and orientation.
    double mu20 = m.mu20 / area;
    double mu02 = m.mu02 / area;
    double mu11 = m.mu11 / area;
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);  // in radians

    // Gather all points for the region.
    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) == region_id)
                points.push_back(cv::Point(x, y));
        }
    }
    if (points.empty()) {
        std::cerr << "Warning: No points found for region " << region_id << std::endl;
        return { {0, 0, 0}, cv::Mat::zeros(labels.size(), CV_8UC3) };
    }

    // Compute the oriented bounding box.
    cv::RotatedRect rRect = cv::minAreaRect(points);
    double bbox_area = rRect.size.width * rRect.size.height;
    double percent_filled = area / bbox_area;
    double aspect_ratio = rRect.size.height / rRect.size.width;

    // Prepare an overlay image.
    cv::Mat mask_8u;
    region_mask.convertTo(mask_8u, CV_8UC1, 255);
    cv::Mat feature_display;
    cv::cvtColor(mask_8u, feature_display, cv::COLOR_GRAY2BGR);
    feature_display += cv::Scalar(50, 50, 50); // Brighten for visibility.

    // Draw the oriented (rotated) bounding box in green with thicker lines.
    cv::Point2f boxPoints[4];
    rRect.points(boxPoints);
    for (int i = 0; i < 4; i++) {
        cv::line(feature_display, boxPoints[i], boxPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
    }

    // Draw the axis of least central moment as a red line.
    int line_length = 50;  // Adjust as needed.
    cv::Point axis_start(static_cast<int>(cx), static_cast<int>(cy));
    cv::Point axis_end(static_cast<int>(cx + line_length * cos(theta)),
                       static_cast<int>(cy + line_length * sin(theta)));
    cv::line(feature_display, axis_start, axis_end, cv::Scalar(0, 0, 255), 3);

    // Overlay the computed angle (in degrees) for reference.
    char angle_text[50];
    sprintf(angle_text, "Theta: %.2f deg", theta * 180.0 / CV_PI);
    cv::putText(feature_display, angle_text, cv::Point(static_cast<int>(cx) + 15, static_cast<int>(cy) + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

    // Draw the centroid as a blue circle and label it.
    cv::Point centroid(static_cast<int>(cx), static_cast<int>(cy));
    cv::circle(feature_display, centroid, 4, cv::Scalar(255, 0, 0), -1);
    cv::putText(feature_display, "Centroid", centroid + cv::Point(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

    std::vector<double> features = { percent_filled, aspect_ratio, theta };
    return { features, feature_display };
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

        // Draw the centroid (blue).
        cv::circle(display, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 4, cv::Scalar(255, 0, 0), -1);
    }
}
