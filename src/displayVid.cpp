/*
    Matej Zecic
    Spring 2025 - CS5330
    Main file for displaying the video stream
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "../include/objectRecPreprocessing.h"

// opens a video stream and runs it through the depth anything network
// displays both the original video stream and the depth stream
int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev;
  cv::Mat src;
  cv::Mat dst;
  cv::Mat dst_vis;
  cv::Mat image;
  cv::Mat image_dst;
  cv::Mat image_dilated;
  cv::Mat image_dst_vis;
  char filename[256]; // a string for the filename
  const float reduction = 0.5;

  image = cv::imread("../Proj03Examples/img5P3.png");
  if( image.empty() ) {
    printf("Unable to read image\n");
    return -1;
  }

  cv::imshow("image", image);

  // make a DANetwork object
  DA2Network da_net( "model_fp16.onnx" );

  // open example video
  capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 256.0 / (refS.height*reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  cv::namedWindow( "Video", 1 );
  cv::namedWindow( "Depth", 2 );

  for(;;) {
    // capture the next frame
    *capdev >> src;
    if( src.empty()) {
      printf("frame is empty\n");
      break;
    }
    // for speed purposes, reduce the size of the input frame by half
    cv::resize( src, src, cv::Size(), reduction, reduction );

    // set the network input
    da_net.set_input( src, scale_factor );

    // run the network
    da_net.run_network( dst, src.size() );

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO );



    // da input for static image
    da_net.set_input( image, scale_factor );

    // run the network
    da_net.run_network( image_dst, src.size() );

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(image_dst, image_dst_vis, cv::COLORMAP_INFERNO );



    // display the images
    cv::imshow("video", src);
    cv::imshow("depth", dst_vis);

    // terminate if the user types 'q'
    char key = cv::waitKey(10);
    if( key == 'q' ) {
      break;
    } else if (key == 's') {
      cv::imwrite("video_image.png", src);
      cv::imwrite("depth_image.png", dst_vis);
      printf("Images saved\n");
    } else if (key == 't') {
      applyThresholding(image, image_dst);
      cv::imshow("Thresholded", image);
    } else if (key == 'm') {
      applyThresholding(image, image_dst);
      applyDilation(image_dst, image_dilated);
      cv::imshow("Dilated", image_dilated);
    }
  }

  printf("Terminating\n");

  return(0);
}
