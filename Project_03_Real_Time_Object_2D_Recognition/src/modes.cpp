/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
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



int basic_training_mode_from_directory(){
  char dirname[256] = "../training_set/";
  char buffer[256];
  // std::vector <float> feature_vector{1,2,3,4,5}; //Sample vector to write in csv file
  int reset_initially = 1;
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  // get the directory path
  // strcpy(dirname, argv[1]);
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
        strstr(dp->d_name, ".jpeg") ||
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
      cv::Mat frame = cv::imread(buffer);//,cv::ImreadModes::IMREAD_UNCHANGED); //Working
      
      // Check for failure
      if (frame.empty()) {
          std::cout << "Could not open or find the image. If you are sure that the image is present at that path, please confirm by opening it manually that it is not corrupt." << std::endl;
          // std::cin.get(); //wait for any key press
          return -1;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                  Applying thresholding to the masked greyscale image                                                         //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //Implementing mask for thresholded HSV using. Src: https://stackoverflow.com/questions/14582082/merging-channels-in-opencv
      cv::Mat mask_3c(frame.size(), CV_8UC3, cv::Scalar(0,0,0)), mask_1c(frame.size(), CV_8UC1, cv::Scalar(0));
      cv::Mat thresholded_image;
      thresholding(frame, thresholded_image);
      cv::String window_binary_masked = "4: Binary masked";
      cv::namedWindow(window_binary_masked);
      cv::imshow(window_binary_masked, thresholded_image);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                  Step 2: Clean up the binary image                                                                           //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      
      cv::Mat cleaned_image(frame.size(), CV_8UC1), temp_cleaned(frame.size(), CV_8UC1);
      grassfire_tf(thresholded_image,cleaned_image, 8, 0, 20); // Perform erosion with 8 connectedness for at most 20 times
      cv::namedWindow("5: Cleaned Image");
      cv::imshow("5: Cleaned Image", cleaned_image);
      // grassfire_tf(temp_cleaned,cleaned_image, 8, 1, 4);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                  Step 3: Segment the image into regions                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Src: https://answers.opencv.org/question/120698/drawning-labeling-components-in-a-image-opencv-c/; https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

        // stats: size is 5(cols) x nLabels(rows). i.e. 5 is column,  stats for each label (row). Different stats type: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
        // labelImage : Stores the labelID/ region ID for each pixel. 0 means it is a background
        // CV_32S: output image label type. Currently CV_32S and CV_16U are supported.
        /// centroids:

      cv::Mat stats, centroids, labelImage;
      int nLabels = connectedComponentsWithStats(cleaned_image, labelImage, stats, centroids, 8, CV_32S);
      std::cout << "Number of labels: " << nLabels << std::endl;
      std::cout << "Stats image size: " << stats.size() << std::endl;
      std::cout << "Centroids size is: " << centroids.size() << std::endl;
      mask_1c = cv::Mat::zeros(labelImage.size(), CV_8UC1); //Creates a mat of single channel with each pixel containing zero value
      cv::Mat surfSup = stats.col(4) > 600;// & stats.col(4) < 2000; // It will store 255 in surfSup if the condition is met for that label, otherwise zero
      
          // Que: How to use binary "and" in opencv mat?
          // Ans: https://stackoverflow.com/questions/17961092/how-do-i-do-boolean-operation-on-mat-such-as-mat3-mat1-mat2

      int num_filtered_regions = 0;
      // Ignore the background pixel. That's why, starting the counter from 1 instead of 0.
      for (int i = 1; i < nLabels; i++){
          if (surfSup.at<uchar>(i, 0)){
              //All the pixels with label id i will be set to 255/ 1 with this operation
              // Src: Observation and https://stackoverflow.com/questions/26776045/compare-opencv-mat-with-scalar-elementwise
              mask_1c = mask_1c | (labelImage==i); // (labelImage==i) will return a mat with the pixel having label 'i' as 255 and rest zero. After ORing with mask and doing it repeatedly for all the labels, mask will have 255 for all the clusters/ regions and 0 elsewhere.
              num_filtered_regions++;
          }
      }
      std::cout << "Filtered regions are: " << num_filtered_regions << std::endl;
      
      // if (num_filtered_regions == 1){
      // Src: https://answers.opencv.org/question/4183/what-is-the-best-way-to-find-bounding-box-for-binary-mask/
      cv::Mat Points;
      cv::findNonZero(mask_1c, Points);
      cv::Rect Min_Rect=boundingRect(Points);
      
      mask_1c_to_3c(mask_1c, mask_3c);
      // mask_3c contains the required segmented foreground pixels
      // cv::Moments m = cv::moments(mask_1c, true);
      
    
      
      // Idea: Try to use findContours then pass the findContours result into boundingRect
      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      // cv::Mat contourOutput = frame.clone();
      cv::findContours( mask_1c, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE ); //Input should be single channel binary image
      std::cout << "Number of countours: " << hierarchy.size() << std::endl;
      
      //Draw the contours: Src: https://stackoverflow.com/questions/8449378/finding-contours-in-opencv, https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html, https://learnopencv.com/contour-detection-using-opencv-python-c/
      cv::Mat contourImage(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
      cv::Scalar colors[3];
      colors[0] = cv::Scalar(255, 0, 0);
      colors[1] = cv::Scalar(0, 255, 0);
      colors[2] = cv::Scalar(0, 0, 255);
      for (size_t idx = 0; idx < contours.size(); idx++) {
          cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
      }
      cv::imshow("Contours", contourImage);
      cv::moveWindow("Contours", 200, 0);
      
      // Find the oriented bounding box from the outermost contors | Src: https://docs.opencv.org/3.4/df/dee/samples_2cpp_2minarea_8cpp-example.html
      cv::Point2f vtx[4];
      std::vector<cv::Moments> all_mu(contours.size());
      std::vector<cv::Point2f> all_vtx_vec, valid_vtx_vec, all_centroid_vec, valid_centroid_vec;
      std::vector<cv::Point2f> mc(contours.size() );
      std::vector<float> all_alpha_vec, valid_alpha_vec;
      
      find_obb_vertices_and_centroid(labelImage, contours, all_vtx_vec, all_centroid_vec, all_alpha_vec);
      std::cout << "Num obb are: " << all_alpha_vec.size() << std::endl;
      cv::Mat frame_copy_all_boxes;
      frame.copyTo(frame_copy_all_boxes);
      plot_obb_on_image(frame_copy_all_boxes,all_vtx_vec, all_centroid_vec, all_alpha_vec);
      cv::String windowName_all_obb = "All OBB"; //Name of the window
      cv::namedWindow(windowName_all_obb); // Create a window
      cv::imshow(windowName_all_obb,frame_copy_all_boxes);
      
      cv::Mat frame_copy_valid_boxes;
      frame.copyTo(frame_copy_valid_boxes);

      std::vector<double> feature_vec;
      find_valid_obb_vertices_and_centroid(labelImage, contours, valid_vtx_vec, valid_centroid_vec, valid_alpha_vec, feature_vec, 15);
      std::cout << "Valid obb are: " << valid_alpha_vec.size() << std::endl;

      plot_obb_on_image(frame_copy_valid_boxes,valid_vtx_vec, valid_centroid_vec, valid_alpha_vec);
      cv::String windowName_valid_obb = "Valid OBB"; //Name of the window
      cv::namedWindow(windowName_valid_obb); // Create a window
      cv::imshow(windowName_valid_obb,frame_copy_valid_boxes);

      for(auto it:feature_vec){
        std::cout << it << std::endl;
      }

      // Axis Aligned Bounding Box (AABB) | NOT USED
      // cv::rectangle(frame_copy,Min_Rect.tl(),Min_Rect.br(),cv::Scalar(0,255,0),2); 

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                  Step 4: Compute features for each major region                                                              //
      ///////////////////////////////////////////////////////////////////////////////////////////
      std::cout << "Filename is modes function: " << buffer << std::endl;
      std::vector<float> feature_vec_float(feature_vec.begin(), feature_vec.end());
      if(feature_vec_float.size() != 0 && valid_centroid_vec.size() == 1){
        append_image_data_csv("training_data.csv", buffer, feature_vec_float, reset_initially );
        reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now. 
      }

    }
  }
  return 0;
}