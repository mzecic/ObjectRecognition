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
    bool overlayMode = true;

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
        // Training mode ('n')
        else if (key == 'n') {
            // Freeze the current frame
            cv::Mat frozen_frame = src.clone();

            // Process the frozen frame
            cv::Mat frozen_threshold;
            cv::Mat frozen_dilated;
            cv::Mat frozen_cc;
            cv::Mat frozen_labels;
            cv::Mat stats, centroids;

            // Process frozen frame
            applyThresholding(frozen_frame, frozen_threshold);
            applyMorphologicalFiltering(frozen_threshold, frozen_dilated);

            // Do connected components directly here to get stats and centroids
            cv::Mat binary_image = frozen_dilated;
            int num_labels = cv::connectedComponentsWithStats(binary_image, frozen_labels, stats, centroids);

            // Create visualization image
            frozen_cc = cv::Mat::zeros(binary_image.size(), CV_8UC3);
            std::vector<cv::Vec3b> colors(num_labels);
            colors[0] = cv::Vec3b(0, 0, 0); // Background

            // Create random colors and identify valid regions
            std::vector<int> valid_regions;
            int min_size = 500;  // Minimum region size

            for (int i = 1; i < num_labels; i++) {
                colors[i] = cv::Vec3b(rand()%256, rand()%256, rand()%256);
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area >= min_size) {
                    valid_regions.push_back(i);
                }
            }

            // Color the regions
            for(int y = 0; y < frozen_cc.rows; y++) {
                for(int x = 0; x < frozen_cc.cols; x++) {
                    int label = frozen_labels.at<int>(y, x);
                    if(std::find(valid_regions.begin(), valid_regions.end(), label) != valid_regions.end()) {
                        frozen_cc.at<cv::Vec3b>(y, x) = colors[label];
                    }
                }
            }

            // Draw region numbers and print region information
            std::cout << "\nAvailable regions:" << std::endl;
            for(int region_id : valid_regions) {
                // Get region information
                int x = stats.at<int>(region_id, cv::CC_STAT_LEFT);
                int y = stats.at<int>(region_id, cv::CC_STAT_TOP);
                int width = stats.at<int>(region_id, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(region_id, cv::CC_STAT_HEIGHT);
                int area = stats.at<int>(region_id, cv::CC_STAT_AREA);

                // Draw region ID at top-left of bounding box
                cv::Point text_pos(x, y - 10);
                std::string label = std::to_string(region_id);
                cv::putText(frozen_cc, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(255, 255, 255), 2);

                // Print region information
                std::cout << "Region " << region_id << ": Area = " << area << " pixels" << std::endl;
            }

            // Show all images
            cv::imshow("Thresholded", frozen_threshold);
            cv::imshow("Connected Components", frozen_cc);
            cv::imshow("Frozen Frame", frozen_frame);

            // Prompt for region selection
            int selectedRegionID;
            std::cout << "\nEnter the region ID to extract features (or -1 to cancel): ";
            std::cin >> selectedRegionID;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Clear input buffer

            if (selectedRegionID != -1) {
                // Verify the selected region exists
                if(std::find(valid_regions.begin(), valid_regions.end(), selectedRegionID) == valid_regions.end()) {
                    std::cout << "Error: Invalid region ID. Please select from the available regions shown above." << std::endl;
                } else {
                    // Compute features
                    std::pair<std::vector<double>, cv::Mat> feature_result =
                        compute_features(frozen_labels, selectedRegionID);
                    std::vector<double> features = feature_result.first;

                    if (features[0] == 0) {
                        std::cout << "Error: Failed to compute features for region " << selectedRegionID << std::endl;
                    } else {
                        // Show feature visualization
                        cv::imshow("Region Features", feature_result.second);

                        // Get object label
                        std::string objectLabel;
                        std::cout << "Enter label for the current object: ";
                        std::getline(std::cin, objectLabel);

                        // Save to training data
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
                    }
                }
            }

            // Wait for key before continuing
            std::cout << "Press any key to continue..." << std::endl;
            cv::waitKey(0);
        }

        // Classification mode ('d')
        else if (key == 'd') {
            // Freeze the current frame
            cv::Mat frozen_frame = src.clone();

            // Process the frozen frame
            cv::Mat frozen_threshold;
            cv::Mat frozen_dilated;
            cv::Mat frozen_cc;
            cv::Mat frozen_labels;
            cv::Mat stats, centroids;

            // Process frozen frame
            applyThresholding(frozen_frame, frozen_threshold);
            applyMorphologicalFiltering(frozen_threshold, frozen_dilated);

            // Do connected components directly here to get stats and centroids
            cv::Mat binary_image = frozen_dilated;
            int num_labels = cv::connectedComponentsWithStats(binary_image, frozen_labels, stats, centroids);

            // Create visualization image
            frozen_cc = cv::Mat::zeros(binary_image.size(), CV_8UC3);
            std::vector<cv::Vec3b> colors(num_labels);
            colors[0] = cv::Vec3b(0, 0, 0); // Background

            // Create random colors and identify valid regions
            std::vector<int> valid_regions;
            int min_size = 500;  // Minimum region size

            for (int i = 1; i < num_labels; i++) {
                colors[i] = cv::Vec3b(rand()%256, rand()%256, rand()%256);
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area >= min_size) {
                    valid_regions.push_back(i);
                }
            }

            // Color the regions
            for(int y = 0; y < frozen_cc.rows; y++) {
                for(int x = 0; x < frozen_cc.cols; x++) {
                    int label = frozen_labels.at<int>(y, x);
                    if(std::find(valid_regions.begin(), valid_regions.end(), label) != valid_regions.end()) {
                        frozen_cc.at<cv::Vec3b>(y, x) = colors[label];
                    }
                }
            }

            // Draw region numbers and print region information
            std::cout << "\nAvailable regions:" << std::endl;
            for(int region_id : valid_regions) {
                // Get region information
                int x = stats.at<int>(region_id, cv::CC_STAT_LEFT);
                int y = stats.at<int>(region_id, cv::CC_STAT_TOP);
                int area = stats.at<int>(region_id, cv::CC_STAT_AREA);

                // Draw region ID at top-left of bounding box
                cv::Point text_pos(x, y - 10);
                std::string label = std::to_string(region_id);
                cv::putText(frozen_cc, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(255, 255, 255), 2);

                // Print region information
                std::cout << "Region " << region_id << ": Area = " << area << " pixels" << std::endl;
            }

            // Show the processed frozen frame
            cv::imshow("Thresholded", frozen_threshold);
            cv::imshow("Connected Components", frozen_cc);
            cv::imshow("Frozen Frame", frozen_frame);

            // Prompt for region selection
            int selectedRegionID;
            std::cout << "\nEnter the region ID to extract features (or -1 to cancel): ";
            std::cin >> selectedRegionID;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Clear input buffer

            if (selectedRegionID != -1) {
                // Verify the selected region exists
                if(std::find(valid_regions.begin(), valid_regions.end(), selectedRegionID) == valid_regions.end()) {
                    std::cout << "Error: Invalid region ID. Please select from the available regions shown above." << std::endl;
                } else {
                    // Compute features
                    std::pair<std::vector<double>, cv::Mat> feature_result = compute_features(frozen_labels, selectedRegionID);
                    std::vector<double> features = feature_result.first;

                    if (features[0] == 0) {
                        std::cout << "Error: Failed to compute features for region " << selectedRegionID << std::endl;
                    } else {
                        cv::imshow("Region Features", feature_result.second);

                        // Open and read training data
                        std::ifstream infile("training_data.csv");
                        if (!infile.is_open()) {
                            std::cerr << "Error: Could not open training_data.csv for reading!" << std::endl;
                        } else {
                            std::string line;
                            std::vector<std::string> trainLabels;
                            std::vector<std::vector<double>> training_data;

                            // Read training data
                            while (std::getline(infile, line)) {
                                std::istringstream ss(line);
                                std::string label;
                                std::vector<double> data;

                                // Get label
                                std::getline(ss, label, ',');

                                // Skip empty or whitespace-only labels
                                if (label.empty() || std::all_of(label.begin(), label.end(), isspace)) {
                                    std::cout << "Warning: Skipping entry with empty label" << std::endl;
                                    continue;
                                }

                                // Read feature values
                                double value;
                                std::vector<double> row_data;
                                while (ss >> value) {
                                    row_data.push_back(value);
                                    if (ss.peek() == ',') ss.ignore();
                                }

                                // Validate feature vector size
                                if (row_data.size() != 3) {  // Assuming we expect 3 features
                                    std::cout << "Warning: Skipping entry with incorrect number of features" << std::endl;
                                    continue;
                                }

                                // Only add valid entries
                                trainLabels.push_back(label);
                                training_data.push_back(row_data);
                            }
                            infile.close();

                            // Verify we have valid training data
                            if (training_data.empty()) {
                                std::cerr << "Error: No valid training data found!" << std::endl;
                                continue;
                            }

                            // Print validated training data
                            std::cout << "\nValid Training Data (" << training_data.size() << " samples):" << std::endl;
                            for (size_t i = 0; i < training_data.size(); i++) {
                                std::cout << trainLabels[i] << ": ";
                                for (double v : training_data[i]) {
                                    std::cout << v << " ";
                                }
                                std::cout << std::endl;
                            }

                            // Compute mean and stddev
                            std::vector<double> mean(features.size(), 0.0);
                            std::vector<double> stddev(features.size(), 0.0);

                            // Calculate means
                            for (const auto& data : training_data) {
                                for (size_t j = 0; j < data.size(); j++) {
                                    mean[j] += data[j];
                                }
                            }
                            for (size_t j = 0; j < mean.size(); j++) {
                                mean[j] /= training_data.size();
                            }

                            // Calculate standard deviations
                            for (const auto& data : training_data) {
                                for (size_t j = 0; j < data.size(); j++) {
                                    stddev[j] += std::pow(data[j] - mean[j], 2);
                                }
                            }
                            for (size_t j = 0; j < stddev.size(); j++) {
                                stddev[j] = std::sqrt(stddev[j] / training_data.size());
                                if (stddev[j] < 1e-6) stddev[j] = 1e-6;  // Prevent division by zero
                            }

                            // Find nearest neighbor with bounds checking
                            double min_distance = std::numeric_limits<double>::max();
                            std::string min_label;

                            for (size_t i = 0; i < training_data.size(); i++) {
                                if (training_data[i].size() != features.size()) {
                                    std::cerr << "Error: Feature vector size mismatch!" << std::endl;
                                    continue;
                                }

                                double distance = 0;
                                std::cout << "Distance to " << trainLabels[i] << ": ";
                                for (size_t j = 0; j < features.size(); j++) {
                                    if (stddev[j] < 1e-6) {
                                        std::cout << "0 ";  // Skip this feature if stddev is too small
                                        continue;
                                    }
                                    double scaled_diff = (features[j] - training_data[i][j]) / stddev[j];
                                    distance += std::pow(scaled_diff, 2);
                                    std::cout << scaled_diff << " ";
                                }
                                distance = std::sqrt(distance);
                                std::cout << "=> " << distance << std::endl;

                                if (distance < min_distance) {
                                    min_distance = distance;
                                    min_label = trainLabels[i];
                                }
                            }

                            std::cout << "The object is classified as: " << min_label << std::endl;

                            // Save to evaluation data
                            std::cout << "Enter the true label for the object: ";
                            std::string true_label;
                            std::getline(std::cin, true_label);

                            std::ofstream outfile("evaluation_data.csv", std::ios::app);
                            if (!outfile.is_open()) {
                                std::cerr << "Error: Could not open evaluation_data.csv for writing!" << std::endl;
                            } else {
                                outfile << true_label << "," << min_label;
                                for (double f : features) {
                                    outfile << "," << f;
                                }
                                outfile << "\n";
                                outfile.close();
                                std::cout << "Evaluation data saved." << std::endl;
                            }
                        }
                    }
                }
            }

            // Wait for key before continuing
            std::cout << "Press any key to continue..." << std::endl;
            cv::waitKey(0);
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
