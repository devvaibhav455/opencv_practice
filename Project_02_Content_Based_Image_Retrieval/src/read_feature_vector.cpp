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
#include <bits/stdc++.h>
#include "csv_util.h"
#include <opencv2/opencv.hpp>

int num_buckets = 8;


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
    printf("Valid feature sets:\nbaseline | hm :stands for Histogram Matching | mhm :stands for Multi Histogram Matching\n");    
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
  if (!((strcmp("baseline", feature_set ) == 0) || (strcmp("hm", feature_set ) == 0) || (strcmp("mhm", feature_set ) == 0))){
    printf("usage:\n %s <target image> <feature set>\n", argv[0]);
    printf("Valid feature sets:\nbaseline | hm :stands for Histogram Matching | mhm :stands for Multi Histogram Matching\n");    
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

  // Read image data from CSV
  std::vector<const char *> files_in_csv;
  std::vector<std::vector<float>> feature_vectors_from_csv;
  read_image_data_csv(csv_filename, files_in_csv, feature_vectors_from_csv, 0);

  // Printing the name of all files in CSV and finding index of target_image
  const char *search_result;
  int found_in_csv = 0;
  int target_index;
  char target_filename[256];
  for (auto it = files_in_csv.begin(); it != files_in_csv.end(); it++) {
    std::cout << *it << std::endl;
    search_result = strstr (*it,target_image); //Return Value: This function returns a pointer points to the first character of the found s2 in s1 otherwise a null pointer if s2 is not present in s1. If s2 points to an empty string, s1 is returned. Src: https://www.geeksforgeeks.org/strstr-in-ccpp/
    if(search_result) {
        found_in_csv = 1;
        strcpy(target_filename, *it); //Storing ../images/target_image_name in target_filename in order to show it on GUI
        target_index = std::distance(files_in_csv.begin(), it);
        printf("Target image: %s found in CSV file at index %d\n", target_image, target_index)
        ;
    }
  }
  if (!found_in_csv){
    printf("Target image: %s not found in the CSV file. Please check!\n", target_image);
    exit(-1);
  }

  // Distance metric calculation
  std::vector<std::pair<float, int>> distance_metric_vector; //Less distance means that images are more similar

  std::vector<std::pair<int, int> > vp;

  std::vector< std::vector<float> >::iterator feature_vectors_image;
  std::vector<float>::iterator feature_vectors_image_data;
  // std:: cout << "Feature vector of the target image:\n" << feature_vectors_from_csv[target_index][0] << std::endl;
  
  // Calculate distance transform using sum-of-squared-difference as the distance metric if feature_set is baseline
  // Perform calculations between target image and all the images
  if (strcmp ("baseline", feature_set ) == 0){
    // Iterating through the features of all the images one by one
    for (feature_vectors_image = feature_vectors_from_csv.begin() ; feature_vectors_image != feature_vectors_from_csv.end(); feature_vectors_image++) {
      int index_image = feature_vectors_image - feature_vectors_from_csv.begin();
      float distance = 0;
      for (feature_vectors_image_data = feature_vectors_image->begin(); feature_vectors_image_data != feature_vectors_image->end(); feature_vectors_image_data++) {
        int index = feature_vectors_image_data - feature_vectors_image->begin();
        distance += pow(feature_vectors_from_csv[target_index][index] - feature_vectors_from_csv[index_image][index], 2);
      }
      distance_metric_vector.push_back(std::make_pair(distance, index_image));
    }   
  }else if(strcmp ("hm", feature_set ) == 0){
    // Iterating through the features of all the images one by one
    for (feature_vectors_image = feature_vectors_from_csv.begin() ; feature_vectors_image != feature_vectors_from_csv.end(); feature_vectors_image++) {
      int index_image = feature_vectors_image - feature_vectors_from_csv.begin();
      float distance = 0;
      for (feature_vectors_image_data = feature_vectors_image->begin(); feature_vectors_image_data != feature_vectors_image->end(); feature_vectors_image_data++) {
        int index = feature_vectors_image_data - feature_vectors_image->begin();
        distance += std::min(feature_vectors_from_csv[target_index][index], feature_vectors_from_csv[index_image][index]);
      }
      distance_metric_vector.push_back(std::make_pair(1-distance, index_image));
    }   
  }else if(strcmp ("mhm", feature_set ) == 0){
    // Iterating through the features of all the images one by one
    for (feature_vectors_image = feature_vectors_from_csv.begin() ; feature_vectors_image != feature_vectors_from_csv.end(); feature_vectors_image++) {
      int index_image = feature_vectors_image - feature_vectors_from_csv.begin(); //Index of the current image which is being iterated
      //If bucket size is 8 for RGB histogram, there will be 8x8x8 = 512 elements for the top half and 512 for the bottom half. So, a total of 1024 elements
      float distance;
      float weight;
      
      std::vector<float>::iterator vector_start;
      std::vector<float>::iterator vector_end;

      for (int k = 0; k<2 ; k++){
        if (k == 0){
          weight = 0.5;
          // printf("k: %d | Weight: %f", k, weight);
          vector_start = feature_vectors_image->begin();
          vector_end = feature_vectors_image->end()  - float(num_buckets*num_buckets*num_buckets); //This is not coverted in the loop. Infact, the value just before this iterator is used
        }else{
          weight = 1 - weight;
          // printf("k: %d | Weight: %f", k, weight);
          vector_start = feature_vectors_image->end() - float(num_buckets*num_buckets*num_buckets);
          vector_end = feature_vectors_image->end();
        }
        for (feature_vectors_image_data = vector_start; feature_vectors_image_data != vector_end; feature_vectors_image_data++) {
          // std::cout << feature_vectors_image->end() << std::endl;
          int index = feature_vectors_image_data - feature_vectors_image->begin();
          // std::cout << "k: " << k << " Iterated through: " << index << std::endl;
          distance += std::min(feature_vectors_from_csv[target_index][index], feature_vectors_from_csv[index_image][index]);
        }
        distance = weight*(1-distance);
      }
      distance_metric_vector.push_back(std::make_pair(distance, index_image));
    }   
  }

  //Handling boundary case. Removing the distance corresponding to the target_image for matching. Just in case, there is an exactly similar image apart from target_image in the database which has distance 0 and we want to find that.

  distance_metric_vector.erase(distance_metric_vector.begin() + target_index);

  // Sorting pair vector in ascending order: Src: https://www.geeksforgeeks.org/keep-track-of-previous-indexes-after-sorting-a-vector-in-c-stl/
  std::sort(distance_metric_vector.begin(), distance_metric_vector.end());
  
  // Printing the distance_metric_vector for confirmation
  for (int i = 0; i <distance_metric_vector.size(); i++){
      std::cout << distance_metric_vector[i].first << " , " << distance_metric_vector[i].second << std::endl;
  }  

  // Need to find the minimum N (say 3) distances (except 0 for the target index) because they are the closest
  int n = 3;
  printf("The %d closest matches to %s are\n", n, target_image);
  // Code to display the image
  cv::Mat target_image_mat = cv::imread(target_filename,cv::ImreadModes::IMREAD_UNCHANGED); //Working
  cv::String windowName_target = "TARGET_IMAGE: " + cv::String(target_image); //Name of the window
  cv::namedWindow(windowName_target); // Create a window
  cv::imshow(windowName_target, target_image_mat); // Show our image inside the created window
  

  for (int i = 0; i<n ; i++){
    // Show the target image and the closest matches
    printf("%s\n", files_in_csv[distance_metric_vector[i].second]);
    cv::Mat match = cv::imread(files_in_csv[distance_metric_vector[i].second],cv::ImreadModes::IMREAD_UNCHANGED); //Working
    cv::String windowName_target = "MATCH_" + cv::String(std::to_string(i)) + ": " + cv::String(files_in_csv[distance_metric_vector[i].second]); //Name of the window
    cv::namedWindow(windowName_target); // Create a window
    cv::imshow(windowName_target, match); // Show our image inside the created window
  }

  while(cv::waitKey(0) != 113){
    //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
  }
  // cv::destroyWindow(windowName_target); //destroy the created window






  
      
    

  // auto it = std::find_if(std::begin(files_in_csv), std::end(files_in_csv), [&](const char * j){return j == "Hello"; });
  // int index = std::distance(files_in_csv.begin(), it); // Ref: https://www.techiedelight.com/find-index-element-vector-cpp/

  // std::vector<const char *>::iterator itr = std::find(files_in_csv.begin(), files_in_csv.end(), "../images//pic.0014.jpg");  
  // // If element was found
  // if (itr != files_in_csv.cend()) {
  //     std::cout << "Element present at index " << std::distance(files_in_csv.begin(), itr);
  // }
  // else {
  //     std::cout << "Element not found";
  // }

  // Printing the vector of all files in csv
  // for (int i = 0; i <feature_vectors_from_csv.size(); i++){
  //   for (int j = 0; j <feature_vectors_from_csv[i].size(); j++){
  //     std::cout << feature_vectors_from_csv[i][j] << ' ';
  //   }
  //   std::cout << std::endl;
  // }

  // Calculate distance between target_image and all other images in the database

      // build the overall filename
      // strcpy(buffer, dirname);
      // strcat(buffer, "/"); //This is where the second / gets added in csv file
      // strcat(buffer, dp->d_name);






  
  
  
  printf("Terminating\n");

  return(0);
}


