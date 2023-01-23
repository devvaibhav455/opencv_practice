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
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );