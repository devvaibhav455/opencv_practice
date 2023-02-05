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

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 3) {
    printf("usage:\n %s <target image> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline\n");
    exit(-1);
  }

  //Get the target image file name
  char target_image[256];
  strcpy(target_image, argv[1]);
  printf("Processing directory %s\n", dirname );

  // // get the directory path
  // strcpy(dirname, argv[1]);
  // printf("Processing directory %s\n", dirname );

  // copy command line feature set name to a local variable and define the name of csv file
  char feature_set[256];
  strcpy( feature_set, argv[2] );

  // Check if the feature_set is entered correctly or not
  if (strcmp ("baseline", feature_set ) != 0){
    printf("usage:\n %s <target image> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline\n");
    exit(-1);
  }

  char* result = argv[2];
  strcat(result, ".csv");; //Ref: https://www.geeksforgeeks.org/char-vs-stdstring-vs-char-c/
  const char* csv_filename = result;

  // open the directory
  // dirp = opendir( dirname );
  // if( dirp == NULL) {
  //   printf("Cannot open directory %s\n", dirname);
  //   exit(-1);
  // }

  std::vector<const char *> files_in_csv;
  std::vector<std::vector<float>> feature_vectors_from_csv;
  read_image_data_csv(csv_filename, files_in_csv, feature_vectors_from_csv, 1);

  // Printing the name of all files in csv
  for (auto i: files_in_csv){
    std::cout << i << ' ' << std::endl;
  }

  // Printing the vector of all files in csv
  for (int i = 0; i <feature_vectors_from_csv.size(); i++){
    for (int j = 0; j <feature_vectors_from_csv[i].size(); j++){
      std::cout << feature_vectors_from_csv[i][j] << ' ';
    }
    std::cout << std::endl;
  }

      // build the overall filename
      // strcpy(buffer, dirname);
      // strcat(buffer, "/"); //This is where the second / gets added in csv file
      // strcat(buffer, dp->d_name);






  
  
  
  printf("Terminating\n");

  return(0);
}


