/*Dev Vaibhav
Spring 2023 CS 5330
Project 1: Real-time filtering
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"



int greyscale( cv::Mat &src, cv::Mat &dst ){
  // allocate dst image
  dst = cv::Mat::zeros(src.size(), CV_8UC1); //Because output is greyscale. i.e. 8-bit single channel

  // Loop over src rows to access the pixel
  for (int i = 1; i<src.rows; i++){
    // row pointer (rptr) for the src
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    // destination pointer (dptr) for the dst/ output
    unsigned char *dptr = dst.ptr<uchar>(i);
    
    //looping through pixels in the row
    for (int j = 0; j<src.cols; j++){
      dptr[j] = 0.299*rptr[j][2] + 0.587*rptr[j][1] + 0.114*rptr[j][0]; // applying weighted average method to compute the greyscale image
    }
  }
  return 0;
}

// src: input image
// dst: output image, 
// input is a color image (type vec3b) and the output should is a color image (type vec3b)
// apply a 5x5 Gaussian filter as separable 1x5 filters ([1 2 4 2 1] vertical and horizontal)
int blur5x5( cv::Mat &src, cv::Mat &dst ){
  dst = cv::Mat::zeros(src.size(), CV_8UC3); //Output is color image

  // loop over src and apply the 1x4 horizontal filter
  for(int i=2;i<src.rows-1;i++) {
    // src pointer for the ith row
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    // destination pointer for the ith row
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=2;j<src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptr[j-2][c] + 2*rptr[j-1][c] + 4*rptr[j][c] + 2*rptr[j+1][c] + 1*rptr[j+2][c])/10;
      }
    }
  }

  cv::Mat tmp = cv::Mat::zeros(src.size(), CV_8UC3); //Output is color image
  dst.copyTo(tmp);

  // loop over tmp, apply a 1x4 vertical filter and store the result in dst
  for(int i=2;i<src.rows-1;i++) {
    // src pointer for the i-2 to i+2 rows. Now, the source is
    cv::Vec3b *rptrm2 = tmp.ptr<cv::Vec3b>(i-2);
    cv::Vec3b *rptrm1 = tmp.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr = tmp.ptr<cv::Vec3b>(i);
    cv::Vec3b *rptrp1 = tmp.ptr<cv::Vec3b>(i+1);
    cv::Vec3b *rptrp2 = tmp.ptr<cv::Vec3b>(i+2);
    // destination pointer for the ith row
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=2;j<src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptrm2[j][c] + 2*rptrm1[j][c] + 4*rptr[j][c] + 2*rptrp1[j][c] + 1*rptrp2[j][c])/10;
      }
    }
  }

  // return
  return(0);
}


// src: input image
// dst: output image, 
// input is a color image (type vec3b) and the output should is a color image (type vec3s)
// apply a 3x3 Gaussian filter as separable filters ([1 2 1]^T vertical and [-1 0 1] horizontal)
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
  dst = cv::Mat::zeros(src.size(), CV_16SC3); //Output is color image of shorts
  cv::Mat tmp; //Using tmp matrix for intermediate operation as it is separable filter
  dst.copyTo(tmp);

  // loop over tmp, apply a 3x1  vertical filter: [1 2 1]^T and store the result in dst
  for(int i=1;i<=src.rows-1;i++) {
    // src pointer for the i-1 to i+1 rows. Now, the source is
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr   = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer for the ith row
    cv::Vec3s *dptr = tmp.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptrm1[j][c] + 2*rptr[j][c] + 1*rptrp1[j][c])/4;
      }
    // printf("Accessed all j's : %d\n " , j);
    }
    // printf("Accessed all i's %d\n", i);
  }
  
  
  // loop over src and apply a 1x3 horizontal filter: [-1 0 1]
  for(int i=1 ; i <= src.rows-1 ; i++) {
    // src pointer for the ith row
    cv::Vec3s *rptr = tmp.ptr<cv::Vec3s>(i);
    // destination pointer for the ith row
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-2;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = -1*rptr[j-1][c] + 1*rptr[j+1][c];
      }
      // printf("Accessed all j's : %d\n " , j);
    }
    // printf("Accessed all i's %d\n", i);
  }

  // return
  return(0);
}




// int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// src: input image
// dst: output image, allocated by this function, data type is CV_16SC3
// apply a 3x3 filter
//  [ -1 0 1]
//  [ -2 0 2]
//  [ -1 0 1]
//
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
