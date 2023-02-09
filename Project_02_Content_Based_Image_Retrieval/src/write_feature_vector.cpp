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

int num_buckets = 8;

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
    printf("usage:\n %s <directory path> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline | hm :stands for Histogram Matching | mhm :stands for Multi Histogram Matching\n");    
    exit(-1);
  }

  // copy command line feature set name to a local variable and define the name of csv file
  char feature_set[256];
  strcpy( feature_set, argv[2] ); 
  char* result = argv[2];
  strcat(result, ".csv");; //Ref: https://www.geeksforgeeks.org/char-vs-stdstring-vs-char-c/
  const char* csv_filename = result;

  // Check if the feature_set is entered correctly or not
  if (!((strcmp("baseline", feature_set ) == 0) || (strcmp("hm", feature_set ) == 0) || (strcmp("mhm", feature_set ) == 0))){
    printf("usage:\n %s <target image> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline | hm :stands for Histogram Matching | mhm :stands for Multi Histogram Matching\n");    
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
        float r,g;
        int bucket_R_idx, bucket_G_idx, bucket_B_idx;
        
        int RGB_histogram[num_buckets][num_buckets][num_buckets];
        // Initializing the RGB 3d histogram to zero to start the counter
        for(int i = 0; i < num_buckets; i++){
          for(int j = 0; j < num_buckets; j++){
            for(int k = 0; k < num_buckets; k++){
              RGB_histogram[i][j][k] = 0;
            }
          }
        }

        // Calculating start and end index of rows for top and bottom half of the image
        int row_start;
        int row_end;
        for (int k = 0; k<2 ; k++){
          if (k ==0){
            row_start = 0;
            row_end = image.rows/2;
          }else{
            row_start = image.rows/2;
            row_end = image.rows;
          }

          // Accessing each and every pixel in row-major order and filling up the buckets
          for (int i = row_start; i < row_end; i++){
            cv::Vec3b *rptr = image.ptr<cv::Vec3b>(i); //Row pointer for ith row
            //looping through pixels in the row
            for (int j = 0; j < image.cols ; j++){
               // Keep in mind that its stored in BGR order in OpenCV
                //Src: https://stackoverflow.com/questions/7571326/why-does-dividing-two-int-not-yield-the-right-value-when-assigned-to-double
                r = rptr[j][2]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
                g = rptr[j][1]/double(rptr[j][0] + rptr[j][1] + rptr[j][2]);
              
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
                feature_vector.push_back(RGB_histogram[i][j][k]/double(image.rows*image.cols));
              }
            }
          }
        }
      }
    }    

      // Calculate the feature vector using 9x9 square in the middle
      append_image_data_csv(csv_filename, buffer, feature_vector, reset_initially );
      // sleep(1);
      std::cout << loop_counter << std::endl;
      loop_counter++;
      reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now.
  }
  printf("Terminating\n");
  return(0);
}


