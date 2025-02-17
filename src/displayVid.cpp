/*
    Matej Zecic
    Spring 2025 - CS5330
    Main file for displaying the video stream and processing both live video and a static image.
    This version adds keypresses for:
      - Thresholding (t)
      - Morphological filtering (m)
      - Connected Components segmentation (k)
      - Feature Extraction overlay (f)
      - Continuous overlay mode toggle (c)
      - Training mode (n) to collect feature vectors with labels.
      - Save a frame (s)
      - Quit (q)
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "../include/objectRecPreprocessing.h"
#include <utility>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    // -----------------------------
    // Variable Declarations
    // -----------------------------
    cv::VideoCapture *capdev;
    cv::Mat src;                // Live camera frame
    cv::Mat dst;                // (Optional) Depth output from network
    cv::Mat dst_vis;
    cv::Mat image;              // Static image loaded from file
    cv::Mat image_dst;          // (Optional) Depth output for static image
    cv::Mat image_dilated;      // For morphological filtering (static image)
    cv::Mat image_cc;           // For connected components (static image)

    cv::Mat cam_threshold;      // Thresholded camera frame
    cv::Mat static_threshold;   // Thresholded static image
    cv::Mat cam_dilated;        // Morphologically filtered (dilated) camera frame
    cv::Mat static_dilated;     // Morphologically filtered (dilated) static image
    cv::Mat cam_cc;             // Connected components (camera)
    cv::Mat static_cc;          // Connected components (static)
    cv::Mat region_analysis_cam;    // Feature overlay for camera snapshot
    cv::Mat region_analysis_static; // Feature overlay for static snapshot
    cv::Mat overlay;            // Blended overlay for continuous mode

    // Global reduction factor to speed up processing.
    const float reduction = 0.5;
    char filename[256];         // For saving file names if needed.

    // -----------------------------
    // Load Static Image
    // -----------------------------
    image = cv::imread("../Proj03Examples/img5P3.png");
    if (image.empty()) {
        printf("Unable to read image\n");
        return -1;
    }
    cv::imshow("Static Image", image);

    // -----------------------------
    // Create the DANetwork Object (if using depth processing)
    // -----------------------------
    DA2Network da_net("model_fp16.onnx");

    // -----------------------------
    // Open Camera
    // -----------------------------
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    float scale_factor = 256.0 / (refS.height * reduction);
    printf("Using scale factor %.2f\n", scale_factor);

    // -----------------------------
    // Create Output Windows
    // -----------------------------
    cv::namedWindow("Video", 1);
    cv::namedWindow("Thresholded Camera", 1);
    cv::namedWindow("Thresholded Static", 1);
    cv::namedWindow("Dilated Camera", 1);
    cv::namedWindow("Dilated Static", 1);
    cv::namedWindow("Connected Components (Camera)", 1);
    cv::namedWindow("Connected Components (Static)", 1);
    cv::namedWindow("Region Analysis (Camera)", 1);
    cv::namedWindow("Region Analysis (Static)", 1);
    cv::namedWindow("Live Region Overlay", 1);
    cv::namedWindow("Labeled Image", 1);  // For training mode visualization

    // -----------------------------
    // Overlay Mode Flag for Continuous Processing
    // -----------------------------
    bool overlayMode = false;

    // -----------------------------
    // Main Loop
    // -----------------------------
    for (;;) {
        // Capture frame from camera.
        *capdev >> src;
        if (src.empty()) {
            printf("Frame is empty\n");
            break;
        }
        // Resize frame for speed.
        cv::resize(src, src, cv::Size(), reduction, reduction);
        cv::imshow("Video", src);

        // Check for keypress.
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;  // Quit the program.
        }
        else if (key == 'c') {
            // Toggle continuous overlay mode.
            overlayMode = !overlayMode;
        }
        else if (key == 's') {
            // Save current camera frame.
            cv::imwrite("video_image.png", src);
            printf("Camera frame saved\n");
        }
        else if (key == 't') {
            // --- Static Snapshot: Thresholding ---
            applyThresholding(src, cam_threshold);
            cv::imshow("Thresholded Camera", cam_threshold);
            applyThresholding(image, static_threshold);
            cv::imshow("Thresholded Static", static_threshold);
        }
        else if (key == 'm') {
            // --- Static Snapshot: Morphological Filtering ---
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::imshow("Dilated Camera", cam_dilated);
            applyThresholding(image, static_threshold);
            applyMorphologicalFiltering(static_threshold, static_dilated);
            cv::imshow("Dilated Static", static_dilated);
        }
        else if (key == 'k') {
            // --- Static Snapshot: Connected Components ---
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::Mat cam_labels;
            applyConnectedComponents(cam_dilated, cam_cc, cam_labels);
            cv::imshow("Connected Components (Camera)", cam_cc);

            applyThresholding(image, static_threshold);
            applyMorphologicalFiltering(static_threshold, static_dilated);
            cv::Mat static_labels;
            applyConnectedComponents(static_dilated, static_cc, static_labels);
            cv::imshow("Connected Components (Static)", static_cc);
        }
        else if (key == 'f') {
            // --- Static Snapshot: Feature Extraction ---
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::Mat cam_labels;
            applyConnectedComponents(cam_dilated, cam_cc, cam_labels);
            cv::Mat cam_features = cam_cc.clone();
            drawAllFeatures(cam_labels, cam_features, 1000, 10);
            cv::imshow("Region Analysis (Camera)", cam_features);

            applyThresholding(image, static_threshold);
            applyMorphologicalFiltering(static_threshold, static_dilated);
            cv::Mat static_labels;
            applyConnectedComponents(static_dilated, static_cc, static_labels);
            cv::Mat static_features = static_cc.clone();
            drawAllFeatures(static_labels, static_features, 1000, 10);
            cv::imshow("Region Analysis (Static)", static_features);
        }
        else if (key == 'n') {
            // Freeze the current processing for training mode:
            // Reprocess the current frame to get a valid label map.
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::Mat cam_labels;
            applyConnectedComponents(cam_dilated, cam_cc, cam_labels);
            cv::imshow("Labeled Image", cam_cc);  // Display the colorized label map

            int selectedRegionID;
            std::cout << "Enter the region ID to extract features: ";
            std::cin >> selectedRegionID;

            // Now compute features using the correct label map (cam_labels).
            std::pair<std::vector<double>, cv::Mat> feature_result = compute_features(cam_labels, selectedRegionID);
            std::vector<double> features = feature_result.first;

            // Prompt the user for a label (via console)
            std::cout << "Enter label for the current object: ";
            std::string objectLabel;
            std::cin >> objectLabel;

            // Append the label and feature vector to a CSV file.
            std::ofstream outfile("training_data.csv", std::ios::app);
            if (!outfile.is_open()) {
                std::cerr << "Error: Could not open training_data.csv for writing!" << std::endl;
            } else {
                outfile << objectLabel;
                for (double f : features) {
                    outfile << "," << f;
                }
                outfile << "\n";
                outfile.close();
                std::cout << "Training data saved for object: " << objectLabel << std::endl;
            }
            // Optionally, display the feature overlay for visual feedback.
            cv::imshow("Region Analysis (Camera)", feature_result.second);
        }

        // --- Continuous Overlay Mode ---
        if (overlayMode) {
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::Mat cam_labels;
            applyConnectedComponents(cam_dilated, cam_cc, cam_labels);
            cv::imshow("Connected Components (Camera)", cam_cc);
            cv::Mat cam_features = cam_cc.clone();
            drawAllFeatures(cam_labels, cam_features, 1000, 10);
            // Blend the overlay with the original frame.
            cv::addWeighted(src, 0.7, cam_features, 0.3, 0, overlay);
            cv::imshow("Live Region Overlay", overlay);
        }
    }

    printf("Terminating\n");
    return 0;
}
