/*
    Matej Zecic
    Spring 2025 - CS5330
    This is an include file that contains preprocessing techniques used to prepare the image for object recognition.
*/

#ifndef OBJECT_RECOGNITION_PREPROCESSING_H
#define OBJECT_RECOGNITION_PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

// Function declarations must exactly match the definitions:
void applyThresholding(const cv::Mat &src, cv::Mat &dst);
void applyDilation(const cv::Mat &src, cv::Mat &dst);
void applyMorphologicalFiltering(const cv::Mat &src, cv::Mat &dst);
void applyConnectedComponents(const cv::Mat &binary_image, cv::Mat &dst, cv::Mat &labels);
std::pair<std::vector<double>, cv::Mat> compute_features(const cv::Mat &labels, int region_id);
void drawAllFeatures(const cv::Mat &labels, cv::Mat &display, double minAreaThreshold, int edgeMargin);

#endif // OBJECT_RECOGNITION_PREPROCESSING_H
