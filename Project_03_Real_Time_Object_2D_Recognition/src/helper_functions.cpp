/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include "filter.h"



// src: input mask of 8UC1
// dst: output mask of 8UC3 with each channel a copy of the input channel 
int mask_1c_to_3c( cv::Mat &src, cv::Mat &dst ){
  std::vector<cv::Mat> channels;
  channels.push_back(src);
  channels.push_back(src);
  channels.push_back(src);
  cv::merge(channels, dst);
  // std::cout << "################### Reached inside mask function ###################" << std::endl;
  return 0;
}

// src: input mask of 8UC3
// dst: output binary image of 8UC1
int thresholding( cv::Mat &src, cv::Mat &dst ){
  
  // allocate dst image
  dst = cv::Mat::zeros(src.size(), CV_8UC1); //Because output is binary. i.e. 8-bit single channel
  cv::Mat hsv_frame_opencv, temp;
  src.copyTo(temp);
  cv::cvtColor(src, hsv_frame_opencv, cv::COLOR_BGR2HSV);
  // convert the color image to HSV

  // Loop over src rows to access the pixel
  for (int i = 0; i<hsv_frame_opencv.rows; i++){
    // row pointer (rptr) for the src
    cv::Vec3b *rptr = hsv_frame_opencv.ptr<cv::Vec3b>(i);
    // destination pointer (dptr) for the dst/ output
    cv::Vec3b *dptr = temp.ptr<cv::Vec3b>(i);
    //looping through pixels in the row
    for (int j = 0; j<hsv_frame_opencv.cols; j++){
        if (rptr[j][1] > 95){
          for (int c = 0; c < hsv_frame_opencv.channels(); c++){
            dptr[j][c] = dptr[j][c]*0.1;
        }
      }
    }
  }

  // Threshold the changed BGR image to binary by converting into greyscale
  // cv::cvtColor(temp, dst, cv::COLOR_BGR2GRAY);
  greyscale(temp, dst);
  for (int i = 0; i < dst.rows; i++){
    // row pointer (rptr) for the src
    unsigned char *rptr = dst.ptr<uchar>(i);
    for (int j = 0; j<dst.cols; j++){
        if (rptr[j] > 120){
          rptr[j] = 0;
        }else{
          rptr[j] = 255;
        }
      }
    }
  

  std::cout << "################### Reached inside thresholding function ###################" << std::endl;
  return 0;
}


// src: input mask of 8UC3
// dst: output binary image of 8UC1
int find_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec ){
  std::vector<cv::Moments> mu(contours.size());
  std::vector<cv::Point2f> mc(contours.size());
  cv::Point2f vtx[4];

  for (int i=0; i < contours.size(); i++){
    // Get the moments
    mu[i] = moments( contours[i], true); 
    //  Get the mass centers:
    mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    // std::cout << "Centroid x: " << mc[i].x << " | Centroid y: " << mc[i].y << std::en 
    
    // Get the oriented bounding box for the ith contour
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    box.points(vtx);
    vtx_vec.push_back(vtx[0]);
    vtx_vec.push_back(vtx[1]);
    vtx_vec.push_back(vtx[2]);
    vtx_vec.push_back(vtx[3]);
    centroid_vec.push_back(mc[i]);
    }
  return 0;    
}

int plot_obb_on_image(cv::Mat &src, std::vector<cv::Point2f> vtx_vec, std::vector<cv::Point2f> centroid_vec){
  int dist, dx, dy, short_index = 0, short_dist, long_dist;
  // Draw the oriented bounding box
  cv::Point2f vtx[4];
  for(int k=0; k<vtx_vec.size()/4; k++){
    vtx[0] = vtx_vec[4*k];
    vtx[1] = vtx_vec[4*k + 1];
    vtx[2] = vtx_vec[4*k + 2];
    vtx[3] = vtx_vec[4*k + 3];
    cv::Point2f centroid_pt = cv::Point2f(centroid_vec[k]);
    std::cout << "Valid Centroid x: " << centroid_pt.x << " | Valid Centroid y: " << centroid_pt.y << std::endl;
    for( int i = 0; i < 4; i++ ){
      cv::line(src, vtx[i], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      
      cv::circle(src,centroid_pt, 4, cv::Scalar(0,0,255), -1);
      // cv::line(frame_copy, [centroids.at<double>(i,0),centroids.at<double>(i,0)], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      dx = vtx[i].x - vtx[(i+1)%4].x;
      dy = vtx[i].y - vtx[(i+1)%4].y;
      dist = sqrt(dx*dx + dy*dy);
      if (i == 0){
          short_dist = dist;
      }
      if (dist < short_dist){
          short_dist = dist;
          short_index = i;
      }else{
          long_dist = dist;
      }
      std::cout << "Distance between " << i << " and " << i+1 << " is: " << dist << std::endl; 
    }
  }
  return 0;

}