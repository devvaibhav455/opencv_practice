/*Dev Vaibhav
Spring 2023 CS 5330
Project 1: Real-time filtering
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"
#include <cmath>


// src: input image
// dst: output image, 
// input is a color image (type vec3b) and the output should is a greyscale image (type uchar)
// apply the weighted average method to compute the greyscale image
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
// input is a color image (type vec3b) and the output is also a color image (type vec3b)
// apply a 5x5 Gaussian filter as separable 1x5 filters ([1 2 4 2 1] vertical and horizontal)
int blur5x5( cv::Mat &src, cv::Mat &dst ){
  dst = cv::Mat::zeros(src.size(), CV_8UC3); //Output is color image

  // loop over src and apply the 1x4 [1 2 4 2 1] horizontal filter
  for(int i=2;i<src.rows-2;i++) {
    // src pointer for the ith row
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    // destination pointer for the ith row
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=2;j<src.cols-2;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptr[j-2][c] + 2*rptr[j-1][c] + 4*rptr[j][c] + 2*rptr[j+1][c] + 1*rptr[j+2][c])/10;
      }
    }
  }

  cv::Mat tmp = cv::Mat::zeros(src.size(), CV_8UC3); //Output is color image
  dst.copyTo(tmp);

  // loop over tmp, apply a 1x4 vertical filter and store the result in dst
  for(int i=2;i<src.rows-2;i++) {
    // src pointer for the i-2 to i+2 rows. Now, the source is
    cv::Vec3b *rptrm2 = tmp.ptr<cv::Vec3b>(i-2);
    cv::Vec3b *rptrm1 = tmp.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr = tmp.ptr<cv::Vec3b>(i);
    cv::Vec3b *rptrp1 = tmp.ptr<cv::Vec3b>(i+1);
    cv::Vec3b *rptrp2 = tmp.ptr<cv::Vec3b>(i+2);
    // destination pointer for the ith row
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=2;j<src.cols-2;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptrm2[j][c] + 2*rptrm1[j][c] + 4*rptr[j][c] + 2*rptrp1[j][c] + 1*rptrp2[j][c])/10;
      }
    }
  }


  // return
  return(0);
}


// src: input image: 8UC3
// dst: output image: 8Uc3
// input is a color image (type vec3b) and the output should is a color image (type vec3b)
// apply a 3x3 Gaussian filter as separable filters ([1 2 1]^T vertical and [-1 0 1] horizontal)
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
  
  dst = cv::Mat::zeros(src.size(), CV_16SC3); //Output is color image of 16S
  cv::Mat tmp_3s = cv::Mat::zeros(src.size(), CV_16SC3); //Using tmp_3s matrix for intermediate operation as it is separable filter
  
  // loop over tmp_3s, apply a 3x1  vertical filter: [1 2 1]^T and store the result in tmp_3s
  for(int i=1;i<=src.rows-1;i++) {
    // src pointer for the i-1 to i+1 rows. Now, the source is
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr   = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer for the ith row
    cv::Vec3s *dptr = tmp_3s.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = (1*rptrm1[j][c] + 2*rptr[j][c] + 1*rptrp1[j][c])/4;
      }
    }
  }

  
  // loop over src and apply a 1x3 horizontal filter: [-1 0 1]
  for(int i=1 ; i <= src.rows-1 ; i++) {
    // src pointer for the ith row
    cv::Vec3s *rptr = tmp_3s.ptr<cv::Vec3s>(i);
    // destination pointer for the ith row
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-2;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = -1*rptr[j-1][c] + 1*rptr[j+1][c];
      }
    }
  }
  // return
  return(0);
}

// src: input image: 8UC3
// dst: output image: 16SC3
// input is a color image (type vec3b) and the output should is a color image (type vec3b)
// apply a 3x3 Gaussian filter as separable filters ([1 2 1]^T vertical and [-1 0 1] horizontal)
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
  
  dst = cv::Mat::zeros(src.size(), CV_16SC3); //Output is color image of uchar
  cv::Mat tmp_3s = cv::Mat::zeros(src.size(), CV_16SC3); //Using tmp_3s matrix for intermediate operation as it is separable filter
  
  // loop over tmp_3s, apply a 3x1  vertical filter: [1 0 -1]^T and store the result in tmp_3s
  for(int i=1;i<=src.rows-1;i++) {
    // src pointer for the i-1 to i+1 rows. Now, the source is
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer for the ith row
    cv::Vec3s *dptr = tmp_3s.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = 1*rptrm1[j][c] - 1*rptrp1[j][c];
      }
    }
  }

  
  // loop over src and apply a 1x3 horizontal filter: [1 2 1]
  for(int i=1 ; i <= src.rows-1 ; i++) {
    // src pointer for the ith row
    cv::Vec3s *rptr = tmp_3s.ptr<cv::Vec3s>(i);
    // destination pointer for the ith row
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<=src.cols-2;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] =  1*rptr[j-1][c] + 2*rptr[j][c] + 1*rptr[j+1][c];
      }
    }
  }

  // return
  return(0);
}

// sx: SobelX input image: 16SC3
// sy: SobelY input image: 16SC3
// dst: output image: 16SC3
// gradient magnitude image using Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
  dst = cv::Mat::zeros(sx.size(), CV_16SC3); //Output is color image of uchar
  
  // loop over rows in the source
  for(int i=0;i<=sx.rows-1;i++) {
    // src pointer for the ith row. Now, the source is
    cv::Vec3s *sx_rptr = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *sy_rptr = sy.ptr<cv::Vec3s>(i);
    // destination pointer for the ith row
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=0;j<=sx.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
	      dptr[j][c] = sqrt(pow(sx_rptr[j][c] , 2) + pow(sy_rptr[j][c] , 2));
      }
    }
  }
  
  // return
  return(0);
}

// blurs and quantizes a color image
// src: input image: 8UC3
// dst: output image: 8UC3
// input is a color image (type vec3b) and the output should is a color image (type vec3b)
// levels: No. of levels used for quantization
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
  
  //Blur the image using Gaussial blur
  cv::Mat blurred_output(src.size(), CV_8UC3);
  blur5x5(src, blurred_output);
  
  // Quantize the image using the no. of levels specified
  int b = 255/levels; //b is bucket size
  dst = cv::Mat::zeros(src.size(), CV_8UC3);

  // Loop for all the rows
  for (int i = 0; i <= src.rows - 1 ; i++){
    cv::Vec3b *rptr = blurred_output.ptr<cv::Vec3b>(i); //Row pointer for the blurred image
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i); // Row pointer for the destination
    // Loop for all the columns
    for (int j = 0; j <= src.cols - 1 ; j++){
      // Instead of using another loop for color channels, statements are written to save time on code execution
      dptr[j][0] = rptr[j][0] / b;
      dptr[j][0] = dptr[j][0] * b;
      dptr[j][1] = rptr[j][1] / b;
      dptr[j][1] = dptr[j][1] * b;
      dptr[j][2] = rptr[j][2] / b;
      dptr[j][2] = dptr[j][2] * b;
    }
  }

  //return
  return(0);
}

// live video cartoonization function using the gradient magnitude and blur/quantize filters
// src: input image: 8UC3
// dst: output image: 8UC3
// input is a color image (type vec3b) and the output should is a color image (type vec3b)
// levels: No. of levels used for quantization
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){
  dst = cv::Mat::zeros(src.size(), CV_8UC3);

  //Gradient Magnitude
  cv::Mat short_c3_output(src.size(),CV_16SC3); //3 channel matrix of shorts for color image
  cv::Mat temp_sx_short16s(src.size(),CV_16SC3); //3 channel matrix of shorts for color image
  cv::Mat temp_sy_short16s(src.size(),CV_16SC3); //3 channel matrix of shorts for color image
  cv::Mat gradient_magnitude_output(src.size(),CV_8UC3); //3 channel matrix for color image
  sobelX3x3(src, temp_sx_short16s);
  sobelY3x3(src, temp_sy_short16s);
  magnitude(temp_sx_short16s, temp_sy_short16s, short_c3_output);
  cv::convertScaleAbs(short_c3_output, gradient_magnitude_output); //Converting 16SC3 to 8UC3

  // blurring and quantizing gradient_magnitude_output using no. of levels given  and storing the result in dst
  blurQuantize(gradient_magnitude_output, dst, levels);

  // // Loop for all the rows
  // for (int i = 0; i <=src.rows - 1; i++){
  //   cv::Vec3b *rptr = gradient_magnitude_output.ptr<cv::Vec3b>(i); // ith Row pointer for gradient_magnitude_output
  //   cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i); // ith row pointer for dst
  //   // Loop for all the columns
  //   for (int j = 0; j <= src.cols - 1; j++){
  //     if (rptr[j][0] > magThreshold){dptr[j][0] = 0;}
  //     if (rptr[j][1] > magThreshold){dptr[j][1] = 0;}
  //     if (rptr[j][2] > magThreshold){dptr[j][2] = 0;}
  //   }
  // }

//return
return(0);
}

// live video negative function 
// src: input image: 8UC3
// dst: output image: 8UC3
// input is a color image (type vec3b) and the output is also a color image (type vec3b)
int negative( cv::Mat &src, cv::Mat&dst){
  dst = cv::Mat::zeros(src.size(), CV_8UC3);

  // Loop for all the rows
  for (int i = 0; i < src.rows; i++){
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i); // ith Row pointer for gradient_magnitude_output
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i); // ith row pointer for dst
    // Loop for all the columns
    for (int j = 0; j < src.cols; j++){
      // Subtract the pixel's color channel value from 255 to get the negative
      dptr[j][0] = 255 - rptr[j][0];
      dptr[j][1] = 255 - rptr[j][1];
      dptr[j][2] = 255 - rptr[j][2];
    }
  }

//return
return(0);
}


