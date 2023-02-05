/*Dev Vaibhav
Spring 2023 CS 5330
Project 2: Content-based Image Retrieval
Code adapted from the sample code provided by Bruce A. Maxwell
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

#include "csv_util.h"
#include <opencv2/opencv.hpp>

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  std::vector <float> feature_vector{1,2,3,4,5}; //Sample vector to write in csv file
  // std::vector <float> feature_vector;
  int reset_initially = 1;
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 3) {
    printf("usage:\n %s <directory path> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline\n");
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // copy command line feature set name to a local variable and define the name of csv file
  char feature_set[256];
  strcpy( feature_set, argv[2] ); 
  char* result = argv[2];
  strcat(result, ".csv");; //Ref: https://www.geeksforgeeks.org/char-vs-stdstring-vs-char-c/
  const char* csv_filename = result;

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }
  
  

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
      strcat(buffer, "/"); //This is where the second / gets added in csv file
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);

      // Read the image file
      cv::Mat image = cv::imread(buffer,cv::ImreadModes::IMREAD_UNCHANGED);

      if (strcmp ("baseline", feature_set ) == 0)
      {
        std::cout << "Entered in baseline" << std::endl;
      }
      

      // Calculate the feature vector using 9x9 square in the middle
      append_image_data_csv(csv_filename, buffer, feature_vector, reset_initially );
      reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now.




    }
  }  
  printf("Terminating\n");

  return(0);
}


