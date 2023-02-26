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

// Resolution required for the window
int res_width = 600; //columns
int res_height = res_width*9/16; //rows

int basic_training_mode_from_directory(){
  std::cout << " ####################### BASIC TRAINING MODE STARTED #######################" << std::endl;
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

      printf("####################### processing image file: %s #######################\n", dp->d_name);

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
      std::cout << "Filename in modes function: " << buffer << std::endl;
      std::vector<float> feature_vec_float(feature_vec.begin(), feature_vec.end());
      if(feature_vec_float.size() != 0 && valid_centroid_vec.size() == 1){
        
        // Remove unwanted file paths from the image names
        std::string buffer_string = char_to_String(buffer);
        size_t last_slash_pos = buffer_string.find_last_of("/");
        buffer_string = buffer_string.substr(last_slash_pos + 1);
        size_t last_underscor_pos = buffer_string.find_last_of("_");
        buffer_string = buffer_string.substr(0, last_underscor_pos);
        char *buffer_star = buffer_string.data();  

        append_image_data_csv("training_data.csv", buffer_star, feature_vec_float, reset_initially );
        reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now. 
      }
    }
  }
  return 0;
}


int object_recognition_mode(char* target_filename){
  
  char *target_filename_char_star = target_filename;
  // Take image name input from user and find the closest match from DB
  char csv_file_name[256] = "training_data.csv";
  const char *csv_filename = csv_file_name;
  std::vector<const char *> files_in_csv;
  std::vector<std::vector<float>> feature_vectors_from_csv;

  //files_in_csv contains the image file names. 49 in length
  //feature_vectors_from_csv contains 49 feature vectors. Each of length 9
  read_image_data_csv(csv_filename, files_in_csv, feature_vectors_from_csv, 0);

  //Variance calculation for each feature in the feature vector
  std::vector<float> standard_deviation_vec;
  calc_standard_deviation_vec(feature_vectors_from_csv, standard_deviation_vec );

  cv::Mat frame_target = cv::imread(target_filename);//,cv::ImreadModes::IMREAD_UNCHANGED);  
  cv::resize(frame_target, frame_target, cv::Size(res_width, res_height));
  cv::String windowName_target_image = "Target Image"; //Name of the window

  // Calculate feature vector for the target image
  std::string target_image_label;
  std::vector<float> target_image_feature_vec;
  cv::Mat src_image_copy_valid_boxes;
  calc_feature_vector(frame_target, target_image_feature_vec, src_image_copy_valid_boxes);

  std::cout << "Accessing the vectors from main: " << target_image_feature_vec.size() << std::endl;
  // std::cout << target_image_feature_vec[0] << " | " << feature_vectors_from_csv[0][0] << " | " << standard_deviation_vec[0] << std::endl;

  // Calculated scaled euclidean distance with all the images in the DB and find the closest match.
  one_object_classifier(target_image_feature_vec, files_in_csv, feature_vectors_from_csv, standard_deviation_vec, target_filename_char_star , target_image_label);

  int fontFace = 0;
  double fontScale = 1;
  int thickness = 3;
  int baseline=0;
  std::cout << "Label is: " << target_image_label  << std::endl;
  cv::Size textSize = cv::getTextSize(target_image_label, fontFace,fontScale, thickness, &baseline);
  baseline += thickness;
  cv::Point textOrg((frame_target.cols - textSize.width)/2 , frame_target.rows - 20);
  putText(src_image_copy_valid_boxes, target_image_label, textOrg, fontFace, fontScale,cv::Scalar(255,0,0), thickness, 8);
  cv::namedWindow(windowName_target_image); // Create a window
  cv::imshow(windowName_target_image, src_image_copy_valid_boxes); // Show our image inside the created window.
  
  while(cv::waitKey(0) != 113){
  //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
  }
  cv::destroyWindow(windowName_target_image); //destroy the created window

  return 0;
}

int knn_classifier(int &k, char * target_filename_char_star){

  std::cout << "Inside KNN function" << std::endl;
  

  char csv_file_name[256] = "training_data.csv";
  const char *csv_filename = csv_file_name;
  std::vector<const char *> files_in_csv;
  std::vector<std::vector<float>> feature_vectors_from_csv;

  cv::Mat frame_target = cv::imread(target_filename_char_star);//,cv::ImreadModes::IMREAD_UNCHANGED);  
  // Check for failure
  if (frame_target.empty()) {
      std::cout << "Could not open or find the image" << std::endl;
      // std::cin.get(); //wait for any key press
      return -1;
  }
  // Resolution required for the window
  cv::resize(frame_target, frame_target, cv::Size(res_width, res_height));
  cv::String windowName_target_image = "Target Image"; //Name of the window

  // Calculate feature vector for the target image
  std::string target_image_label;
  std::vector<float> target_image_feature_vec;
  cv::Mat src_image_copy_valid_boxes;
  calc_feature_vector(frame_target, target_image_feature_vec, src_image_copy_valid_boxes);

  //files_in_csv contains the image file names. 49 in length
  //feature_vectors_from_csv contains 49 feature vectors. Each of length 9
  read_image_data_csv(csv_filename, files_in_csv, feature_vectors_from_csv, 0);

  std::vector<const char *> unique_labels;
  std::vector<std::vector<int>> unique_labels_indices;
  find_unique_and_their_indices(files_in_csv, unique_labels, unique_labels_indices);


  // Perform calculation of distances to each class
  std::vector<std::pair<float, int>> distance_to_class;

  std::vector<float> sdv_all_class;
  calc_standard_deviation_vec(feature_vectors_from_csv,sdv_all_class);

  // Iterate over every label
  for (int i = 0; i < unique_labels.size(); i++){
    std::cout << "\nCurrent class under check: " << unique_labels[i] << std::endl;
    // Vector which stores distance of target image within a class. If class has 10 pictures, 10 distances will be stored in it. 
    std::vector<float> distance_within_class_vec;
    if(unique_labels_indices[i].size() < k){
      std::cout << "Class: " << unique_labels[i] << " contains less than " << k << " images. Skipping this class for feature matching" << std::endl;
      continue;
    }

    // Vector of feature vector (vofv) within a class
    std::vector<std::vector<float>> vofv_within_class;
    for (int j = 0; j < unique_labels_indices[i].size(); j++){
      vofv_within_class.push_back(feature_vectors_from_csv[i]);
    }

    //Calculate standard deviation vector (sdv) within a class. One for each element of the feature vector
    // std::vector<float> sdv_within_class;
    // calc_standard_deviation_vec(vofv_within_class,sdv_within_class);


    // Distance metric vector (dmv) within class. Contains distance of target image with all the images in a class/ label
    std::vector<std::pair<float, int>> dmv_within_class;
    calc_scaled_euclidean_dist_vec(target_image_feature_vec, vofv_within_class, sdv_all_class, dmv_within_class);

    float min_dist;
    find_min_k_distance(k, dmv_within_class, min_dist);
    distance_to_class.push_back((std::make_pair(min_dist, i)));
  }
  float all_class_min_dist;
  int all_class_min_dist_index;

  std::cout << "\nDistance from all classes" << std::endl;
  all_class_min_dist_index = find_min_k_distance(1,distance_to_class,all_class_min_dist );
  target_image_label = unique_labels[all_class_min_dist_index];


  int fontFace = 0;
  double fontScale = 1;
  int thickness = 3;
  int baseline=0;
  std::cout << "Label is: " << target_image_label  << std::endl;
  cv::Size textSize = cv::getTextSize(target_image_label, fontFace,fontScale, thickness, &baseline);
  baseline += thickness;
  cv::Point textOrg((frame_target.cols - textSize.width)/2 , frame_target.rows - 20);
  putText(src_image_copy_valid_boxes, target_image_label, textOrg, fontFace, fontScale,cv::Scalar(255,0,0), thickness, 8);
  cv::namedWindow(windowName_target_image); // Create a window
  cv::imshow(windowName_target_image, src_image_copy_valid_boxes); // Show our image inside the created window.
  
  while(cv::waitKey(0) != 113){
  //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
  }
  cv::destroyWindow(windowName_target_image); //destroy the created window

return 0;
}

int testing_mode(int &source, cv::Mat &frame){
  // Step 1 | Displaying live video
  // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
  // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
  
  cv::VideoCapture *capdev;

  if (source == 1){
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
      throw std::runtime_error("Error");
      return -1;
    }
    std::cout << "Input feed is camera" << std::endl;
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
  }
  // Integer to save images from live feed with _1, _2, .... .jpeg in the name
  int saved_image_id = 0;
  
  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::String window_binary_masked = std::to_string(window_id) + ": Binary masked";
  cv::namedWindow(window_binary_masked);
  window_id++;

  cv::String window_cleaned_image = std::to_string(window_id) + ": Cleaned Image";
  cv::namedWindow(window_cleaned_image);
  window_id++;

  cv::String window_contours = std::to_string(window_id) + ": Contours";
  cv::namedWindow(window_contours);
  window_id++;

  cv::String window_all_obb = std::to_string(window_id) + ": All OBB"; //Name of the window
  cv::namedWindow(window_all_obb); // Create a window
  window_id++;

  cv::String window_valid_obb = std::to_string(window_id) + ": Valid OBB"; //Name of the window
  cv::namedWindow(window_valid_obb); // Create a window
  window_id++;

  cv::String window_segmentation = std::to_string(window_id) + ": Segmentation"; //Name of the window
  cv::namedWindow(window_segmentation);
  window_id++;




  while (true) {
    std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
    // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    if( source == 1){
        std::cout << "Getting data from camera" << std::endl;
        *capdev >> frame; //frame.type() is 16 viz. 8UC3
    }
    cv::resize(frame, frame, cv::Size(res_width, res_height));
    
    // std::cout << "Resized size is:" << frame.size() << std::endl; //Prints columnsxrows
    // std::cout << "Frame after input from camera = " << std::endl << " " << frame.at<cv::Vec3b>(0,0) << std::endl; 
    // std::cout << "Frame after input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    
    
    cv::imshow(window_original_image, frame);
    
    int key_pressed = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
    // std::cout << "Pressed key is: " << key_pressed << std::endl;

    // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
    if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
        //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
        std::cout << "q is pressed. Exiting the program" << std::endl;
        cv::destroyWindow("1: Original_Image"); //destroy the created window
        return 0;
    }else if (key_pressed == 's'){
        //Save image if 's' is pressed. 115 is s's ASCII value 
        std::cout << "Saving image to file saved_image_id.jpeg" << std::endl;
        std::string image_name = "saved_image_" + std::to_string(saved_image_id) + ".jpeg";
        cv::imwrite(image_name, frame);
        saved_image_id++;
        // cv::imwrite("saved_image.jpeg", frame);
    }
    printf("Resizing image to %dx%d\n", res_width, res_height);

    // Need to fix user implementation of bgr to hsv converter not working
    // bgr_to_hsv(frame,hsv_frame);
    // cv::convertScaleAbs(hsv_frame, rgb_output); //Converting 16SC3 to 8UC3
    // rgb_output.copyTo(filtered_output);
    // cv::namedWindow("HSV Image user function");
    // cv::imshow("HSV Image user function", filtered_output);

    //Convert image to BW and calculate histogram
    // cv::cvtColor(frame, grey_output, cv::COLOR_BGR2GRAY);
    
    // cv::cvtColor(frame, hsv_frame_opencv, cv::COLOR_BGR2HSV);
    // cv::namedWindow("2: HSV Image OpenCV function");
    // cv::imshow("2: HSV Image OpenCV function", hsv_frame_opencv);
    // hsv_frame_opencv = hsv_frame_opencv + cv::Scalar(0,-20,0); //Making pixels dark

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  Histogram calculation for the HSV Image                                                                     //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // const int histSize = 256;
    // // A vector for storing individual channels of HSV image
    // std::vector<cv::Mat> hsv_planes;
    // // Split the channels of hsv and store it in the vector
    // cv::split(hsv_frame_opencv, hsv_planes);
    // float range[] = {0, 256};
    // const float *histRange = {range};
    // bool uniform = true;
    // bool accumulate = false;
    // cv::Mat h_hist, s_hist, v_hist;
    // cv::calcHist(&hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &histSize,
    //             &histRange, uniform, accumulate);
    // cv::calcHist(&hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &histSize,
    //             &histRange, uniform, accumulate);
    // cv::calcHist(&hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize,
    //             &histRange, uniform, accumulate);
    // drawHistogram(h_hist,s_hist,v_hist);
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //           Detect the object based on HSV Range Values                                                                        //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        
    // More info: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    // That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the specified 1D, 2D, 3D, ... box and 0 otherwise. It outputs a binary image
    
    // cv::inRange(hsv_frame_opencv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), hsv_thresholded);
    // cv::inRange(hsv_frame_opencv, cv::Scalar(0, 29, 0), cv::Scalar(37, 255, 255), hsv_thresholded); // Values found after changing the slider
    // Show the frames
    // imshow(window_hsv_thresholded, hsv_thresholded);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //           Convert the hsv_thresholded pixels to greyscale image by creating a mask (3 channel mask                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Implementing mask for thresholded HSV using. Src: https://stackoverflow.com/questions/14582082/merging-channels-in-opencv
    cv::Mat mask_3c(frame.size(), CV_8UC3, cv::Scalar(0,0,0)), mask_1c(frame.size(), CV_8UC1, cv::Scalar(0));
    // hsv_thresholded.copyTo(mask_1c);
    // mask_1c_to_3c(mask_1c, mask_3c);
    // cv::Mat color_masked = mask_3c & frame;
    // cv::cvtColor(color_masked, grey_output, cv::COLOR_BGR2GRAY);
    // cv::imshow(window_grey_masked, color_masked);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  Applying thresholding to the masked greyscale image                                                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::Mat thresholded_image;
    // Src: https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/; https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    // cv::threshold(grey_output, thresholded_image, threshold_grey, 255, cv::THRESH_BINARY ); //THRESH_BINARY_INV
    // cv::threshold(grey_output, thresholded_image, 22, 255, cv::THRESH_BINARY);

    thresholding(frame, thresholded_image);
    
    cv::imshow(window_binary_masked, thresholded_image);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  Step 2: Clean up the binary image                                                                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    cv::Mat cleaned_image(frame.size(), CV_8UC1), temp_cleaned(frame.size(), CV_8UC1);
    grassfire_tf(thresholded_image,cleaned_image, 8, 0, 20); // Perform erosion with 8 connectedness for at most 20 times
    // grassfire_tf(temp_cleaned,cleaned_image, 8, 1, 4);
    
    cv::imshow(window_cleaned_image, cleaned_image);
    // std::cout << "################### Reached clean up part ###################" << std::endl;

    // while(cv::waitKey(0) != 113){
    // //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
    // }
    // cv::destroyWindow("Cleaned Image"); //destroy the created window

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
    cv::imshow(window_contours, contourImage);
    
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
    cv::imshow(window_all_obb,frame_copy_all_boxes);
    
    cv::Mat frame_copy_valid_boxes;
    frame.copyTo(frame_copy_valid_boxes);

    std::vector<double> feature_vec;
    find_valid_obb_vertices_and_centroid(labelImage, contours, valid_vtx_vec, valid_centroid_vec, valid_alpha_vec, feature_vec, 15);
    std::cout << "Valid obb are: " << valid_alpha_vec.size() << std::endl;

    plot_obb_on_image(frame_copy_valid_boxes,valid_vtx_vec, valid_centroid_vec, valid_alpha_vec);
    
    
    cv::imshow(window_valid_obb,frame_copy_valid_boxes);

    

    // Axis Aligned Bounding Box (AABB) | NOT USED
    // cv::rectangle(frame_copy,Min_Rect.tl(),Min_Rect.br(),cv::Scalar(0,255,0),2); 

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  Step 4: Compute features for each major region                                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Features of the image are already captured in function call find_valid_obb_vertices_and_centroid and present in feature_vec


    // char *label_from_user=new char[256];
    // fgets(label_from_user, 256, stdin);
    // std::vector<float> feature_vec_float(feature_vec.begin(), feature_vec.end());
    // append_image_data_csv("training_data.csv", label_from_user, feature_vec_float, reset_initially );
    // reset_initially = 0; //Don't reset the file after it has been done once. Instead start appending to it now. 

    
    
    

    // sleep(10);
    cv::Mat color_segmented(frame.size(), CV_8UC3, cv::Scalar(0));
    frame.copyTo(color_segmented,mask_3c);
    
    cv::imshow(window_segmentation, color_segmented);
    // }
    std::cout << "####################### REACHED END OF WHILE LOOP #######################" << std::endl;

  }
  return 0;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                PLEASE IGNORE CODE BELOW THIS LINE. KEP FOR MY PERSONAL REFERENCE                                             //  
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Some parameters to display the trackbar to find the threshold using GUI for HSV image
// Src: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
// const int max_value_H = 360/2;
// const int max_value = 255;
// const cv::String window_hsv_thresholded = "3: HSV Thresholded";
// int low_H = 0, low_S = 0, low_V = 0;
// int high_H = max_value_H, high_S = max_value, high_V = max_value;

// // Trackbar parameters for the grey masked image
// const cv::String window_grey_masked = "Grey Masked";
// const int threshold_slider_max = 255;
// int threshold_grey = 0;


// static void on_low_H_thresh_trackbar(int, void *)
// {
//     low_H = cv::min(high_H-1, low_H);
//     cv::setTrackbarPos("Low H", window_hsv_thresholded, low_H);
// }
// static void on_high_H_thresh_trackbar(int, void *)
// {
//     high_H = cv::max(high_H, low_H+1);
//     cv::setTrackbarPos("High H", window_hsv_thresholded, high_H);
// }
// static void on_low_S_thresh_trackbar(int, void *)
// {
//     low_S = cv::min(high_S-1, low_S);
//     cv::setTrackbarPos("Low S", window_hsv_thresholded, low_S);
// }
// static void on_high_S_thresh_trackbar(int, void *)
// {
//     high_S = cv::max(high_S, low_S+1);
//     cv::setTrackbarPos("High S", window_hsv_thresholded, high_S);
// }
// static void on_low_V_thresh_trackbar(int, void *)
// {
//     low_V = cv::min(high_V-1, low_V);
//     cv::setTrackbarPos("Low V", window_hsv_thresholded, low_V);
// }
// static void on_high_V_thresh_trackbar(int, void *)
// {
//     high_V = cv::max(high_V, low_V+1);
//     cv::setTrackbarPos("High V", window_hsv_thresholded, high_V);
// }
// static void on_threshold_grey_trackbar( int, void* )
// {
//     threshold_grey = threshold_grey;
//     cv::setTrackbarPos("Grey Threshold", window_grey_masked, threshold_grey);
// }