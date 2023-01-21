/*Dev Vaibhav
Spring 2023 CS 5330
Project 1: Real-time filtering
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"

// src: input image
// dst: output image, allocated by this function, data type is CV_16SC3
// apply a 3x3 filter
//  [ -1 0 1]
//  [ -2 0 2]
//  [ -1 0 1]
//

int greyscale( cv::Mat &src, cv::Mat &dst ){
  dst = cv::Mat::zeros(src.size(), CV_8UC1); //Because output is greyscale. i.e. 8-bit single channel

  for (int i = 1; i<src.rows; i++){
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    unsigned char *dptr = dst.ptr<uchar>(i);

    for (int j = 0; j<src.cols; j++){
      dptr[j] = 0.299*rptr[j][2] + 0.587*rptr[j][1] + 0.114*rptr[j][0];
      // dptr[j] = (rptr[j][2] + rptr[j][1] + rptr[j][0])/10;
    }
  }
  return 0;
}


// int gradX( cv::Mat &src, cv::Mat &dst ) {
//   // allocate dst image
//   dst = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type

//   // loop over src and apply a 3x3 filter
//   for(int i=1;i<src.rows-1;i++) {
//     // src pointer
//     cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
//     cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
//     cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);

//     // destination pointer
//     cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);

//     // for each column
//     for(int j=1;j<src.cols-1;j++) {

//       // for each color channel
//       for(int c=0;c<3;c++) {
// 	dptr[j][c] = (-1 * rptrm1[j-1][c] + 1 * rptrm1[j+1][c] +
// 		      -2*rptr[j-1][c] + 2*rptr[j+1][c] +
// 		      -1 * rptrp1[j-1][c] + 1*rptrp1[j+1][c]) / 4;
//       }
//     }
//   }

//   // return
//   return(0);
// }
