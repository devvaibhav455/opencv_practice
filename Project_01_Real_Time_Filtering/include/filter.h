/*Dev Vaibhav
Spring 2023 CS 5330
Project 1: Real-time filtering
*/

// function computing horizontal gradient
int greyscale( cv::Mat &src, cv::Mat &dst );

// function to compute Gausian blur
int blur5x5( cv::Mat &src, cv::Mat &dst );

// Sobel X and Y as separable filters. X: positive right; : positive up
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// gradient magnitude image using Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

// blurs and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// live video cartoonization function using the gradient magnitude and blur/quantize filters
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

// Function to calcuate the negative of a color image
int negative( cv::Mat &src, cv::Mat&dst);