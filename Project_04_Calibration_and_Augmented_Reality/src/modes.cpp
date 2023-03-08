/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include "filter.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <csv_util.h>



// Basic training mode. Reads images from directory and writes the feature vectors to a csv file
int calibrate_camera(){

  char dirname[256] = "../calibration_images/";
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  // Src: https://stackoverflow.com/questions/2802188/file-count-in-a-directory-using-c
  // get the directory path
  // strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );
  int i = 0;
  // open the directory
  dirp = opendir( dirname );
  if( dirp != NULL) {
    while (dp = readdir (dirp)){
      // check if the file is an image
      if( strstr(dp->d_name, ".jpg") ||
          strstr(dp->d_name, ".jpeg") ||
          strstr(dp->d_name, ".png") ||
          strstr(dp->d_name, ".ppm") ||
          strstr(dp->d_name, ".tif") ){
        i++;
          }
    }
    (void) closedir (dirp);    
  }else{
    perror ("Couldn't open the directory");
    exit(-1);
  }
  printf("There's %d files in the current directory.\n", i);

  if (i == 0){
    //Capture calibration images from the camera
    capture_calibration_images();  
  }

  //READ all the images from a directory and calculate calibration parameters using them
  calc_calib_params(dirname);

 return 0;
}

