/*
    Matej Zecic
    Spring 2025 - CS5330
    Main file for displaying the video stream and processing both live video and a static image
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "../include/objectRecPreprocessing.h"
#include <utility>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    cv::Mat src;            // Live camera frame
    cv::Mat dst;            // Depth output from network (for visualization)
    cv::Mat dst_vis;
    cv::Mat image;          // Static image loaded from file
    cv::Mat image_dst;      // Depth output for static image from network
    cv::Mat image_dilated;  // For morphological filtering result (static image)
    cv::Mat image_cc;       // For connected components (static image)
    cv::Mat cam_threshold;  // For thresholded camera frame
    cv::Mat static_threshold; // For thresholded static image
    cv::Mat cam_dilated;    // For dilated camera frame
    cv::Mat static_dilated; // For dilated static image
    cv::Mat cam_cc;         // For connected components (camera)
    cv::Mat static_cc;      // For connected components (static)
    cv::Mat region_analysis_cam;   // Feature overlay for camera
    cv::Mat region_analysis_static; // Feature overlay for static image

    char filename[256]; // a string for the filename
    const float reduction = 0.5;

    // Load the static image
    image = cv::imread("../Proj03Examples/img5P3.png");
    if( image.empty() ) {
        printf("Unable to read image\n");
        return -1;
    }
    cv::imshow("Static Image", image);

    // Create the DANetwork object (for depth processing, if needed)
    DA2Network da_net("model_fp16.onnx");

    // Open the camera
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT) );
    printf("Expected size: %d %d\n", refS.width, refS.height);

    float scale_factor = 256.0 / (refS.height * reduction);
    printf("Using scale factor %.2f\n", scale_factor);

    // Create output windows
    cv::namedWindow("Video", 1);
    cv::namedWindow("Thresholded Camera", 1);
    cv::namedWindow("Thresholded Static", 1);
    cv::namedWindow("Dilated Camera", 1);
    cv::namedWindow("Dilated Static", 1);
    cv::namedWindow("Connected Components (Camera)", 1);
    cv::namedWindow("Connected Components (Static)", 1);
    cv::namedWindow("Region Analysis (Camera)", 1);
    cv::namedWindow("Region Analysis (Static)", 1);

   // Define a flag for overlay mode
bool overlayMode = false;

    for (;;) {
        // Capture a frame from the camera
        *capdev >> src;
        if (src.empty()) {
            printf("Frame is empty\n");
            break;
        }
        // Resize for speed
        cv::resize(src, src, cv::Size(), reduction, reduction);

        // Display the live video frame
        cv::imshow("Video", src);

        // Check for keypress
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        else if (key == 'c') {
            // Toggle overlay mode when 'c' is pressed.
            overlayMode = !overlayMode;
        }
        else if (key == 's') {
            cv::imwrite("video_image.png", src);
            printf("Camera frame saved\n");
        }
        // Other key commands can be handled similarly (like 't' or 'm' for thresholding/dilation)

        // If overlay mode is enabled, compute and display the overlay continuously.
        if (overlayMode) {
            // Process the current frame for overlay:
            // You might choose a robust morphological filtering function here.
            applyThresholding(src, cam_threshold);
            applyMorphologicalFiltering(cam_threshold, cam_dilated);
            cv::Mat cam_labels;
            applyConnectedComponents(cam_dilated, cam_cc, cam_labels);

            // Optionally, display the segmentation output.
            cv::imshow("Connected Components (Camera)", cam_cc);

            // Clone the connected components image to draw features on.
            cv::Mat cam_features = cam_cc.clone();
            // Draw features (oriented bounding boxes, axes, centroids) for all regions.
            drawAllFeatures(cam_labels, cam_features, 1000, 10);

            // Blend the overlay with the original frame for a nicer effect.
            cv::Mat overlay;
            cv::addWeighted(src, 0.7, cam_features, 0.3, 0, overlay);
            cv::imshow("Live Region Overlay", overlay);
        }
    }

    printf("Terminating\n");
    return 0;
}
