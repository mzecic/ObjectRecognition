/*
    Matej Zecic
    Spring 2025 - CS5330
    This is an include file that contains preprocessing techniques used to prepare the image for object recognition.
*/

#include <opencv2/opencv.hpp>

void applyThresholding(cv::Mat &src, cv::Mat &dst);

void applyDilation(cv::Mat &src, cv::Mat &dst);
