/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"



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

