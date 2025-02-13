/*
    Matej Zecic
    Spring 2025 - CS5330
    This file contains preprocessing techniques used to prepare the image for object recognition.
*/

#include <opencv2/opencv.hpp>
#include "../include/objectRecPreprocessing.h"
#include <queue>
#include <iostream>

void applyThresholding(cv::Mat &src, cv::Mat &dst) {
    // Compute mean depth
    // double sum_depth = 0;
    // int count = 0;

    // for (int i = 0; i < dst.rows; ++i) {
    //     for (int j = 0; j < dst.cols; ++j) {
    //         sum_depth += dst.at<unsigned char>(i, j); // Correct: Read depth values
    //         count++;
    //     }
    // }

    // double mean_depth = sum_depth / count;

    // // Apply thresholding (modify `src` in place)
    // for (int i = 0; i < src.rows; ++i) {
    //     for (int j = 0; j < src.cols; ++j) {
    //         int depth_value = dst.at<unsigned char>(i, j); // Get depth value

    //         if (depth_value > mean_depth) { // Highlight closer objects
    //             src.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0); // Mark in green
    //         }
    //     }
    // }

    // Check if pixels are darker or brighter and apply thresholding
    dst = src.clone();
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            if (pixel[0] > 100 && pixel[1] > 100 && pixel[2] > 100) {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255); // White
            } else {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // Black
            }
        }
    }
}

void applyDilation(cv::Mat &src, cv::Mat &dst) {
    // Apply dilation
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(src, dst, element);
}

void applyCCA(cv::Mat &src, cv::Mat &dst) {
    cv::Mat labels = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    int curr_region = 1

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols, ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j)

            if (pixel[0] =! 0 && pixel[1] =! 0 && pixel[2] =! 0) {

                cv::queue<std::pair<int, int> = queue;
                // push first black pixel onto the queue
                queue.push(i, j)
                labels[i, j] = curr_region
                // get neighbors
                while (!queue.empty()) {
                    std::pair<int, int> curr = queue.pop()
                    cv::Vec3b curr_neighbor = src.
                    // add neighbor to queue if its black and not visited
                    if (src.at<cv::Vec3b>(i - 1, j)[0] != 0 && ) {
                        queue.push(i - 1, j)

                    }
                }
            }
        }
    }
}
