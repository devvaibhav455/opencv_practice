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
int find_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vel){
  cv::Moments m_sc; //Spatial and central moment  | Src: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html#ab8972f76cccd51af351cbda199fb4a0d
  float alpha;
  cv::Point2f mc;
  cv::Point2f vtx[4];
  for (int i=0; i < contours.size(); i++){
    // Get the moments
    m_sc = moments( contours[i], true);
    // Calculate Hu Moments 
    alpha = 0.5*atan(2*m_sc.mu11/double(m_sc.mu20 - m_sc.mu02));

    //  Get the mass centers:
    mc = cv::Point2f( m_sc.m10/m_sc.m00 , m_sc.m01/m_sc.m00 );
    // std::cout << "Centroid x: " << mc[i].x << " | Centroid y: " << mc[i].y << std::en 
    
    // Get the oriented bounding box for the ith contour
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    box.points(vtx);
    vtx_vec.push_back(vtx[0]);
    vtx_vec.push_back(vtx[1]);
    vtx_vec.push_back(vtx[2]);
    vtx_vec.push_back(vtx[3]);
    centroid_vec.push_back(mc);
    alpha_vel.push_back(alpha);
    }
  return 0;    
}


int plot_obb_on_image(cv::Mat &src, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec){
  int dist, dx, dy, short_index = 0, short_dist, long_dist;
  // Draw the oriented bounding box
  cv::Point2f vtx[4];
  for(int k=0; k<vtx_vec.size()/4; k++){
    vtx[0] = vtx_vec[4*k];
    vtx[1] = vtx_vec[4*k + 1];
    vtx[2] = vtx_vec[4*k + 2];
    vtx[3] = vtx_vec[4*k + 3];
    cv::Point2f centroid_pt = cv::Point2f(centroid_vec[k]);

    
    int fontFace = 0;
    double fontScale = 1;
    int thickness = 3;
    int baseline=0;
                              
    cv::circle(src,centroid_pt, 4, cv::Scalar(0,0,255), -1);
    cv::Size textSize = cv::getTextSize(std::to_string(k), fontFace,fontScale, thickness, &baseline);
    baseline += thickness;
    cv::Point textOrg = centroid_pt;
    putText(src, std::to_string(k), textOrg, fontFace, fontScale,cv::Scalar(255,0,0), thickness, 8);

    for( int i = 0; i < 4; i++ ){
      cv::line(src, vtx[i], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
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
      // std::cout << "Distance between " << i << " and " << i+1 << " is: " << dist << std::endl; 
    }
    // Plotting Axis of Least Central Moment (alcm) by finding its end and start points
    cv::Point2f alcm[2];
    alcm[0].x = centroid_pt.x + long_dist*cos(alpha_vec[k]);
    alcm[0].y = centroid_pt.y + long_dist*sin(alpha_vec[k]);       
    alcm[1].x = centroid_pt.x - long_dist*cos(alpha_vec[k]);
    alcm[1].y = centroid_pt.y - long_dist*sin(alpha_vec[k]);
    cv::line(src, alcm[0], alcm[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  }
  return 0;

}



int find_valid_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec, std::vector<double> &feature_vec ,int distance_from_boundary)
{
  cv::Moments m_sc;
  cv::Point2f mc; //mass center
  cv::Point2f vtx[4]; //vertices of 4 corners of bounding box rectangle
  float alpha;

  std::cout << "Valid Centroid rangle x: (" << distance_from_boundary << " , " << src.cols - distance_from_boundary << ")"
  << " | Valid Centroid rangle y: (" << distance_from_boundary << " , " << src.rows - distance_from_boundary << ")" << std::endl;
  
  for (int i=0; i < contours.size(); i++){
    // Get the moments
    m_sc = moments( contours[i], true);

    // Calculate Hu Moments 
    double huMoments[7]; 
    std::cout << "Declared hu moments" << std::endl;
    cv::HuMoments(m_sc, huMoments);
    std::cout << "Calculated hu moments" << std::endl;

    // std::cout << "i in outer loop: " << i << std::endl;
    // Log scale hu moments 
    for(int i = 0; i < 7; i++) {
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])); 
        // std::cout << "i in inner loop: " << i << std::endl;
    }
    // std::cout << "i in outer loop: " << i << std::endl;
    
    int N = sizeof(huMoments) / sizeof(huMoments[0]);
  
    // Initialize a vector by passing the
    // pointer to the first and last element
    // of the range as arguments
    std::vector<double> huMoments_arr_to_vec(huMoments, huMoments + N); //Converting array to vector

    alpha = 0.5*atan(2*m_sc.mu11/double(m_sc.mu20 - m_sc.mu02));
    //  Get the mass centers:
    mc = cv::Point2f( m_sc.m10/m_sc.m00 , m_sc.m01/m_sc.m00 );
    std::cout << "Current Centroid x: " << mc.x << " | Centroid y: " << mc.y << std::endl;
    
    // Get the oriented bounding box for the ith contour
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    std::cout << "Aspect ration is: " << box.size.aspectRatio() << std::endl;
    box.points(vtx);

    int is_rect_valid = 0;

    // Check if centroid is valid or not
    if(mc.x < distance_from_boundary || mc.x > src.cols - distance_from_boundary || mc.y < distance_from_boundary || mc.y > src.rows - distance_from_boundary){
        is_rect_valid = 0;
    }else {
        is_rect_valid = 1;
        for(int j=0; j<4 ; j++){
          // std::cout << "Checking corner x: " << vtx[j].x << " | corner y: " << vtx[j].y << std::endl;
          // If centroid is valid, check if the four corners of rectangle are valid or not
          if (vtx[j].x < distance_from_boundary || vtx[j].x > src.cols - distance_from_boundary || vtx[j].y < distance_from_boundary || vtx[j].y > src.rows - distance_from_boundary){
              is_rect_valid = 0;
              break;
          }
        }
    }

    
    if (is_rect_valid == 1){
      vtx_vec.push_back(vtx[0]);
      vtx_vec.push_back(vtx[1]);
      vtx_vec.push_back(vtx[2]);
      vtx_vec.push_back(vtx[3]);
      centroid_vec.push_back(mc);
      alpha_vec.push_back(alpha);
      for (int i = 0; i <huMoments_arr_to_vec.size(); i++){
        feature_vec.push_back(huMoments_arr_to_vec[i]);
      }
      feature_vec.push_back(cv::contourArea(contours[i])/box.size.area());
      feature_vec.push_back(box.size.aspectRatio());

      
      std::cout << "Pushed Valid Centroid x: " << mc.x << " | Valid Centroid y: " << mc.y << std::endl;
    }
  }
return 0;    
}


