/*Dev Vaibhav
Spring 2023 CS 5330
Project 2: Content-based Image Retrieval
Code adapted from the sample code provided by Professor: Bruce A. Maxwell
*/

/*
  Bruce A. Maxwell
  S21
  
  Sample code to identify image fils in a directory
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <unistd.h>
#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include "filter.h" // To use Sobel Magnitude Image

int num_buckets = 8;
int num_gabor_filters = 4;

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */

// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  // std::vector <float> feature_vector{1,2,3,4,5}; //Sample vector to write in csv file
  std::vector <float> feature_vector;
  int reset_initially = 1;
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 3) {
    printf("Please enter sufficient arguments\n");
    printf("usage:\n %s <directory path> <feature set> <distance_metric>(optional)\n", argv[0]);
    printf("Valid feature sets:\n\tbaseline\n\thm :stands for Histogram Matching\n\tmhm :stands for Multi Histogram Matching\n\ttc: stands for Texture and Color\n\tcd: stands for Custom Design\n\t1: Extension 1: Gabor filters\n\t2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image\n\t3: Extension 3: Histogram matching for different distance metrics. Distance_metric is required for extension 3\nValid distance metric\n\tcorrelation/ chisquare/ intersection/ bhattacharyya\n");    
    exit(-1);
  }

  // copy command line feature set name to a local variable and define the name of csv file
  char feature_set[256];
  strcpy( feature_set, argv[2] ); 
  char* result = argv[2];
  strcat(result, ".csv");; //Ref: https://www.geeksforgeeks.org/char-vs-stdstring-vs-char-c/
  const char* csv_filename = result;

  // Check if the feature_set is entered correctly or not
  if (!((strcmp("baseline", feature_set ) == 0) || (strcmp("hm", feature_set ) == 0) || (strcmp("mhm", feature_set ) == 0) || (strcmp("tc", feature_set ) == 0) || (strcmp("cd", feature_set ) == 0) || (strcmp("1", feature_set ) == 0) || (strcmp("2", feature_set ) == 0))){
    printf("Please enter correct feature_set\n");
    printf("usage:\n %s <directory path> <feature set> <distance_metric>(optional)\n", argv[0]);
    printf("Valid feature sets:\n\tbaseline\n\thm :stands for Histogram Matching\n\tmhm :stands for Multi Histogram Matching\n\ttc: stands for Texture and Color\n\tcd: stands for Custom Design\n\t1: Extension 1: Gabor filters\n\t2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image\n\t3: Extension 3: Histogram matching for different distance metrics. Distance_metric is required for extension 3\nValid distance metric\n\tcorrelation/ chisquare/ intersection/ bhattacharyya\n");    
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }
  int loop_counter = 0;
  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      // strcat(buffer, "/"); //This is where the second / gets added in csv file
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);
      
      // Read the image file
      // cv::Mat image = cv::imread("../src/sample_image.jpeg",cv::ImreadModes::IMREAD_UNCHANGED); //Working
      cv::Mat image = cv::imread(buffer,cv::ImreadModes::IMREAD_UNCHANGED); //Working
      
      // Check for failure
      if (image.empty()) {
          std::cout << "Could not open or find the image. If you are sure that the image is present at that path, please confirm by opening it manually that it is not corrupt." << std::endl;
          // std::cin.get(); //wait for any key press
          return -1;
      }

      // Code to display the image
      // cv::String windowName = "Image display"; //Name of the window
      // cv::namedWindow(windowName); // Create a window
      // cv::imshow(windowName, image); // Show our image inside the created window.

      // while(cv::waitKey(0) != 113){
      //   //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
      // }
      // cv::destroyWindow(windowName); //destroy the created window

      // cv::String windowName_target = "Sobel magnitude image"; //Name of the window
      // cv::namedWindow(windowName_target); // Create a window

      // Calculate the feature vector
      feature_vector.clear();
      // std::cout << "Stringcompare result should be zero: " << strcmp ("baseline", feature_set ) << std::endl;
      if (strcmp ("baseline", feature_set ) == 0){ 
        std::cout << "Calculating 9x9 baseline feature vectors" << std::endl;
        // Find 9x9 matrix at the center and store it in row-major order in a vector
        // Loop over src rows to access the pixel
        int center_row = image.rows/2;
        int center_col = image.cols/2;
        // row pointer (rptr) for the image
        for (int i = center_row - 4; i <= center_row + 4; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i);
          //looping through pixels in the row
          for (int j = center_col - 4; j <= center_col + 4 ; j++){
            feature_vector.push_back(rptr[j][0]);
            feature_vector.push_back(rptr[j][1]);
            feature_vector.push_back(rptr[j][2]);
          }
        }
      }else if(strcmp ("hm", feature_set ) == 0){
        // std::cout << "Calculating Histogram Matching (HM) feature vectors" << std::endl;
        float r,g;
        int bucket_r_idx, bucket_g_idx;
        
        int rg_histogram[num_buckets][num_buckets];
        // Initializing the rg 2d histogram to zero to start the counter
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              rg_histogram[i][j] = 0;
          }
        }

        // Accessing each and every pixel in row-major order and filling up the buckets
        for (int i = 0; i < image.rows; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i);
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Handling edge cases when R,G,B are all zero.
            if(rptr[j][0] == 0 && rptr[j][1] == 0 && rptr[j][2] == 0){
              // std::cout << "Pixel is black" << std::endl;
              r = 0;
              g = 0;
            }else{ // Keep in mind that its stored in BGR order in OpenCV
              //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
              r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
            }
            bucket_r_idx = r*num_buckets; //Its over x-axis
            bucket_g_idx = g*num_buckets; //Its over y-axis
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            rg_histogram[bucket_r_idx][bucket_g_idx]++; //Increment the bucket value by one    
          }
        }

        // rg_histogram is now computed. Need to normalize it store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              feature_vector.push_back(rg_histogram[i][j]/double(image.rows*image.cols));
          }
        }

      }else if(strcmp ("mhm", feature_set ) == 0){
        // std::cout << "Calculating Multi-Histogram Matching (HM) feature vectors using 3d RGB  histogram" << std::endl;
        int bucket_R_idx, bucket_G_idx, bucket_B_idx;
        int RGB_histogram_top[num_buckets][num_buckets][num_buckets];
        int RGB_histogram_bottom[num_buckets][num_buckets][num_buckets];

        // Initializing the RGB 3d histograms to zero to start the counter for both top and bottom 
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              RGB_histogram_top[i][j][k] = 0;
              RGB_histogram_bottom[i][j][k] = 0;
            }
          }
        }

        // Accessing each and every pixel for top histogram in row-major order and filling up the buckets
        for (int i = 0; i < image.rows/2; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i); //Row pointer for ith row
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Keep in mind that its stored in BGR order in OpenCV
            bucket_R_idx = rptr[j][2]*num_buckets/256; // Bucket index for the RED value
            bucket_G_idx = rptr[j][1]*num_buckets/256; // Bucket index for the GREEN value
            bucket_B_idx = rptr[j][0]*num_buckets/256; // Bucket index for the BLUE value
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            RGB_histogram_top[bucket_R_idx][bucket_G_idx][bucket_B_idx]++; //Increment the bucket value by one  
          }
        }

        // RGB_histogram_top is now computed. Need to store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              feature_vector.push_back(RGB_histogram_top[i][j][k]/double(image.rows*image.cols/2)); //Number of pixels in top and bottom half is half the no. of pixels in whole image.
            }
          }
        }

        // BOTTOM HISTOGRAM
        // Accessing each and every pixel for bottom histogram in row-major order and filling up the buckets
        for (int i = image.rows/2; i < image.rows; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i); //Row pointer for ith row
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Keep in mind that its stored in BGR order in OpenCV
            bucket_R_idx = rptr[j][2]*num_buckets/256; // Bucket index for the RED value
            bucket_G_idx = rptr[j][1]*num_buckets/256; // Bucket index for the GREEN value
            bucket_B_idx = rptr[j][0]*num_buckets/256; // Bucket index for the BLUE value
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            RGB_histogram_bottom[bucket_R_idx][bucket_G_idx][bucket_B_idx]++; //Increment the bucket value by one  
          }
        }

        // RGB_histogram_top is now computed. Need to store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              feature_vector.push_back(RGB_histogram_bottom[i][j][k]/double(image.rows*image.cols/2)); //Number of pixels in top and bottom half is half the no. of pixels in whole image.
            }
          }
        }
      }else if(strcmp ("tc", feature_set ) == 0){
        //Calculating the texture metric/ simply finding the texture using Sobel Magnitude Image's histogram.
        cv::Mat temp_sx_short16s(image.rows,image.cols,CV_16SC3); //3 channel matrix of shorts for color image
        cv::Mat temp_sy_short16s(image.rows,image.cols,CV_16SC3); //3 channel matrix of shorts for color image
        cv::Mat short_c3_output(image.rows,image.cols,CV_16SC3); //3 channel matrix of shorts for color image
        cv::Mat grad_mag_color_output(image.rows,image.cols,CV_8UC3); //3 channel matrix for color image
              
        sobelX3x3(image, temp_sx_short16s); // Computing Sobel X
        sobelY3x3(image, temp_sy_short16s); // Computing Sobel Y
        magnitude(temp_sx_short16s, temp_sy_short16s, short_c3_output); // Combining Sobel X and Sobel Y
        cv::convertScaleAbs(short_c3_output, grad_mag_color_output); //Converting 16SC3 to 8UC3 to make it suitable for display

        // cv::String windowName = "Image display"; //Name of the window
        // cv::namedWindow(windowName); // Create a window
        // cv::imshow(windowName, grad_mag_color_output); // Show our image inside the created window.

        // while(cv::waitKey(0) != 113){
        // //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
        // }
        // cv::destroyWindow(windowName); //destroy the created window
        
        // std::cout << "This is cout: " << (int)grad_mag_color_output.at<cv::Vec3s>(0,0)[0] << std::endl;
        // printf("0,0 element is: %d", grad_mag_color_output.at<cv::Vec3b>(0,0)[0]);
        // sleep(10);

        int bucket_R_idx, bucket_G_idx, bucket_B_idx;
        int RGB_histogram_grad_mag_color[num_buckets][num_buckets][num_buckets]; //Texture histogrsm from Sobel Magnitude image
        int RGB_histogram[num_buckets][num_buckets][num_buckets]; // Color histogram from original image

        // Initializing the textture and color RGB 3d histograms to zero to start the counter for each bucket 
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              RGB_histogram_grad_mag_color[i][j][k] = 0;
              RGB_histogram[i][j][k] = 0;
            }
          }
        }

        // Calculating texture histogram (RGB_histogram_grad_mag_color) using Sobel Magnitude color output.
        // Accessing each and every pixel for sobel magnitude color image in row-major order and filling up the buckets
        for (int i = 0; i < grad_mag_color_output.rows; i++){
          cv::Vec3b *rptr = grad_mag_color_output.ptr<cv::Vec3b>(i); //Row pointer for ith row
          //looping through pixels in the row
          for (int j = 0; j < grad_mag_color_output.cols ; j++){
            // Keep in mind that its stored in BGR order in OpenCV
            bucket_R_idx = rptr[j][2]*num_buckets/256; // Bucket index for the RED value
            bucket_G_idx = rptr[j][1]*num_buckets/256; // Bucket index for the GREEN value
            bucket_B_idx = rptr[j][0]*num_buckets/256; // Bucket index for the BLUE value
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            RGB_histogram_grad_mag_color[bucket_R_idx][bucket_G_idx][bucket_B_idx]++; //Increment the bucket value by one  
          }
        }

        // RGB_histogram_grad_mag_color or texture histogram is now computed. Need to store this in feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              feature_vector.push_back(RGB_histogram_grad_mag_color[i][j][k]/double(grad_mag_color_output.rows*grad_mag_color_output.cols)); 
            }
          }
        }

        // Calculating Color histogram from original image i.e. image
        // Accessing each and every pixel from the original image in row-major order and filling up the buckets
        for (int i = 0; i < image.rows; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i); //Row pointer for ith row
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Keep in mind that its stored in BGR order in OpenCV
            bucket_R_idx = rptr[j][2]*num_buckets/256; // Bucket index for the RED value
            bucket_G_idx = rptr[j][1]*num_buckets/256; // Bucket index for the GREEN value
            bucket_B_idx = rptr[j][0]*num_buckets/256; // Bucket index for the BLUE value
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            RGB_histogram[bucket_R_idx][bucket_G_idx][bucket_B_idx]++; //Increment the bucket value by one  
          }
        }

        // RGB_histogram or color histogram is now computed. Need to store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              feature_vector.push_back(RGB_histogram[i][j][k]/double(image.rows*image.cols)); 
            }
          }
        }
      }else if(strcmp ("cd", feature_set ) == 0){
        cv::Mat l5e5_output(image.rows,image.cols,CV_8UC3); //3 channel matrix for color image
        L5E5(image, l5e5_output);

        // Generate 2D rg chromaticity histogram from l5e5_output
        float r,g;
        int bucket_r_idx, bucket_g_idx;
        
        int rg_histogram[num_buckets][num_buckets];
        // Initializing the rg 2d histogram to zero to start the counter
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              rg_histogram[i][j] = 0;
          }
        }

        // Accessing each and every pixel in row-major order and filling up the buckets
        for (int i = 0; i < l5e5_output.rows; i++){
          cv::Vec3b *rptr = l5e5_output.ptr<cv::Vec3b>(i);
          //looping through pixels in the row
          for (int j = 0; j < l5e5_output.cols ; j++){
            // Handling edge cases when R,G,B are all zero.
            if(rptr[j][0] == 0 && rptr[j][1] == 0 && rptr[j][2] == 0){
              // std::cout << "Pixel is black" << std::endl;
              r = 0;
              g = 0;
            }else{ // Keep in mind that its stored in BGR order in OpenCV
              //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
              r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
            }
            bucket_r_idx = r*num_buckets; //Its over x-axis
            bucket_g_idx = g*num_buckets; //Its over y-axis
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            rg_histogram[bucket_r_idx][bucket_g_idx]++; //Increment the bucket value by one    
          }
        }

        // rg_histogram is now computed. Need to normalize it and store this in feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              feature_vector.push_back(rg_histogram[i][j]/double(image.rows*image.cols));
          }
        }

        // #######################################################################//
        // Generating 3D RGB histogram of the original image

        int bucket_R_idx, bucket_G_idx, bucket_B_idx;
        int RGB_histogram[num_buckets][num_buckets][num_buckets];

        // Initializing the RGB 3d histograms to zero to start the counter 
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              RGB_histogram[i][j][k] = 0;
            }
          }
        }

        // Accessing each and every pixel for top histogram in row-major order and filling up the buckets
        for (int i = 0; i < image.rows; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i); //Row pointer for ith row
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Keep in mind that its stored in BGR order in OpenCV
            bucket_R_idx = rptr[j][2]*num_buckets/256; // Bucket index for the RED value
            bucket_G_idx = rptr[j][1]*num_buckets/256; // Bucket index for the GREEN value
            bucket_B_idx = rptr[j][0]*num_buckets/256; // Bucket index for the BLUE value
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            RGB_histogram[bucket_R_idx][bucket_G_idx][bucket_B_idx]++; //Increment the bucket value by one  
          }
        }

        // RGB_histogram is now computed. Need to store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              feature_vector.push_back(RGB_histogram[i][j][k]/double(image.rows*image.cols)); //Normalizing while storing the feature vector
            }
          }
        }


      }else if(strcmp ("1", feature_set ) == 0){
        // Implementing Extension 1: Using Gabor filters
        // Src: https://stackoverflow.com/questions/26948110/gabor-kernel-parameters-in-opencv; https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

        // cv::Mat grey_output(image.rows,image.cols,CV_8UC1); //Single channel matrix for greyscale image
        // cv::cvtColor(image, grey_output, cv::COLOR_BGR2GRAY);
        // image = cv::imread("../gabor_sample.png",cv::ImreadModes::IMREAD_UNCHANGED); //Working
        cv::Mat gabor_input, gabor_output, gabor_kernel_resized; //3 channel matrix for color image
        image.convertTo(gabor_input, CV_32F);

        cv::Mat temp; //3 channel matrix for color image
        
        // Ravina params double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0; double th1 = M_PI/2;
        // cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, (i*th1), lm, gm, ps);
        cv::Size KernelSize(31,31); // sumegha 9,9
        double Sigma = 2; //5 
        double Lambda = 45*CV_PI/180; //4
        // double Lambda = 1.0;
        // double psi = 1*CV_PI/180; // pi/4
        double Gamma = 0.5; // 0.04
        // double Gamma = 0.02; // 0.04
        double psi = 0; // pi/4
        
        for (int i = 0; i < num_gabor_filters ; i++){
          double Theta = i*(360/num_gabor_filters)*CV_PI/180;
          cv::Mat gabor_kernel = cv::getGaborKernel(KernelSize, Sigma, Theta, Lambda,Gamma,psi);
          cv::resize(gabor_kernel, gabor_kernel_resized, cv::Size(500,500), cv::INTER_LINEAR);
          cv::filter2D(gabor_input, temp, CV_32F, gabor_kernel);
          temp.convertTo(gabor_output, CV_8UC3);//,1.0/255.0);

          // Visualize gabor kernel
          // cv::String windowName0 = "Gabor kernel " + cv::String(std::to_string(i+1)) + ": Angle: " + cv::String(std::to_string(i*360/num_gabor_filters)) + " degrees"; //Name of the window
          // cv::namedWindow(windowName0); // Create a window
          // cv::imshow(windowName0, gabor_kernel_resized);

          // while(cv::waitKey(0) != 113){
          // //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
          // }
          // cv::destroyWindow(windowName0); //destroy the created window

          // std::cout << "Gabor output is: " << gabor_output.size << std::endl;
          
          // Calculating the histogram for gabor filter
          float r,g;
          int bucket_r_idx, bucket_g_idx;
          
          int rg_histogram[num_buckets][num_buckets];
          // Initializing the rg 2d histogram to zero to start the counter
          for(int i = 0; i < num_buckets; i++){
            for(int j = 0; j < num_buckets; j++){
                rg_histogram[i][j] = 0;
            }
          }

          // Accessing each and every pixel in row-major order and filling up the buckets
          for (int i = 0; i < gabor_output.rows; i++){
            cv::Vec3b *rptr = gabor_output.ptr<cv::Vec3b>(i);
            //looping through pixels in the row
            for (int j = 0; j < gabor_output.cols ; j++){
              // Handling edge cases when R,G,B are all zero.
              if(rptr[j][0] == 0 && rptr[j][1] == 0 && rptr[j][2] == 0){
                // std::cout << "Pixel is black" << std::endl;
                r = 0;
                g = 0;
              }else{ // Keep in mind that its stored in BGR order in OpenCV
                //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
                r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
                g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              }
              bucket_r_idx = r*num_buckets; //Its over x-axis
              bucket_g_idx = g*num_buckets; //Its over y-axis
              // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
              rg_histogram[bucket_r_idx][bucket_g_idx]++; //Increment the bucket value by one    
            }
          }

          // rg_histogram is now computed. Need to normalize it store this is feature_vector
          for(int i = 0; i < num_buckets; i++){
            for(int j = 0; j < num_buckets; j++){
                feature_vector.push_back(rg_histogram[i][j]/double(image.rows*image.cols));
            }
          }       
          
        }

        // cv::Mat gabor_kernel_resized;

        // std::cout << gabor_kernel << std::endl;

        // max of filter2d output and original image (in 32F)..convert to 8uc3 take this image for 2d histogram calculation
        // cv::convertScaleAbs(temp, gabor_input); //Converting 16SC3 to 8UC3 to make it suitable for display
        // gabor_kernel_resized.convertTo(temp, CV_8UC3,1.0/255.0);

        // Create histogram from gabor_input

        //Showing Original image

        


        // cv::String windowName1 = "Original image"; //Name of the window
        // cv::namedWindow(windowName1); // Create a window
        // cv::imshow(windowName1, image);
        
        // cv::String windowName = "Gabor output"; //Name of the window
        // cv::namedWindow(windowName); // Create a window
        // cv::imshow(windowName, gabor_input);
        


      }else if(strcmp ("2", feature_set ) == 0){
        //Collect all blue trash bins, Laws L5E5 filter with the combination + rg chromaticity histogram for the original. 
        cv::Mat l5e5_output(image.rows,image.cols,CV_8UC3); //3 channel matrix for color image
        L5E5(image, l5e5_output);

        // Generate 2D rg chromaticity histogram from l5e5_output
        float r,g;
        int bucket_r_idx, bucket_g_idx;
        
        int rg_histogram[num_buckets][num_buckets];
        // Initializing the rg 2d histogram to zero to start the counter
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              rg_histogram[i][j] = 0;
          }
        }

        // Accessing each and every pixel in row-major order and filling up the buckets
        for (int i = 0; i < l5e5_output.rows; i++){
          cv::Vec3b *rptr = l5e5_output.ptr<cv::Vec3b>(i);
          //looping through pixels in the row
          for (int j = 0; j < l5e5_output.cols ; j++){
            // Handling edge cases when R,G,B are all zero.
            if(rptr[j][0] == 0 && rptr[j][1] == 0 && rptr[j][2] == 0){
              // std::cout << "Pixel is black" << std::endl;
              r = 0;
              g = 0;
            }else{ // Keep in mind that its stored in BGR order in OpenCV
              //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
              r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
            }
            bucket_r_idx = r*num_buckets; //Its over x-axis
            bucket_g_idx = g*num_buckets; //Its over y-axis
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            rg_histogram[bucket_r_idx][bucket_g_idx]++; //Increment the bucket value by one    
          }
        }

        // rg_histogram is now computed. Need to normalize it store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              feature_vector.push_back(rg_histogram[i][j]/double(image.rows*image.cols));
          }
        }

        // #######################################################################//
        // Generating 2D rg histogram of the original image
        
        // Initializing the rg 2d histogram to zero to start the counter
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              rg_histogram[i][j] = 0;
          }
        }

        // Accessing each and every pixel in row-major order and filling up the buckets
        for (int i = 0; i < image.rows; i++){
          cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i);
          //looping through pixels in the row
          for (int j = 0; j < image.cols ; j++){
            // Handling edge cases when R,G,B are all zero.
            if(rptr[j][0] == 0 && rptr[j][1] == 0 && rptr[j][2] == 0){
              // std::cout << "Pixel is black" << std::endl;
              r = 0;
              g = 0;
            }else{ // Keep in mind that its stored in BGR order in OpenCV
              //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
              r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
            }
            bucket_r_idx = r*num_buckets; //Its over x-axis
            bucket_g_idx = g*num_buckets; //Its over y-axis
            // std::cout << "(r,g): " << r << " " << g << " | " << bucket_r_idx << " | " << bucket_g_idx << std::endl;
            rg_histogram[bucket_r_idx][bucket_g_idx]++; //Increment the bucket value by one    
          }
        }

        // rg_histogram is now computed. Need to normalize it store this is feature_vector
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
              feature_vector.push_back(rg_histogram[i][j]/double(image.rows*image.cols));
          }
        }
      }else if(strcmp ("3", feature_set ) == 0){
        //Implementing distance using various distance metric
        // Nothing to compute here. We will use hm.csv file which is already generated in other binary
      }
      
      // After Calculate the feature vector using 9x9 square in the middle
      append_image_data_csv(csv_filename, buffer, feature_vector, reset_initially );
      // sleep(1);
      std::cout << loop_counter << std::endl;
      loop_counter++;
      reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now. 

    } // Closing bracket for if file is an image in directory condition
  }       
  printf("Terminating\n");
  return(0);
}


