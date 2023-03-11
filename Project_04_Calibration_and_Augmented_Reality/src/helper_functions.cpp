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
#include "csv_util.h"
#include <set>
#include <dirent.h>


cv::Size patternsize(9,6); //interior number of corners
// Resolution required for the window
int res_width = 600; //columns
int res_height = res_width*9/16; //rows


// Converts char[] to std::string
std::string char_to_String(char* a)
{
  //Ref: https://www.geeksforgeeks.org/convert-character-array-to-string-in-c/
    std::string s(a);

    //Usage:
    // char a[] = { 'C', 'O', 'D', 'E' };
    // char b[] = "geeksforgeeks";
 
    // string s_a = convertToString(a);
    // string s_b = convertToString(b);
 
    // cout << s_a << endl;
    // cout << s_b << endl;
 
    // we cannot use this technique again
    // to store something in s
    // because we use constructors
    // which are only called
    // when the string is declared.
 
    // Remove commented portion
    // to see for yourself
 
    /*
    char demo[] = "gfg";
    s(demo); // compilation error
    */
 
    return s;
}

int capture_calibration_images(){
  // Step 1 | Displaying live video
  // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
  // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
  
  cv::Mat frame;
  cv::VideoCapture *capdev;

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

  // Integer to save images from live feed with _1, _2, .... .jpeg in the name
  int saved_image_id = 0;
  
  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::Mat grey_image;
  cv::Mat mriwc; //most recent image with corners
  
  std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners
  bool patternfound;

  while (true) {
    // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
    // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    // std::cout << "Getting data from camera" << std::endl;
    *capdev >> frame; //frame.type() is 16 viz. 8UC3

    cv::resize(frame, frame, cv::Size(res_width, res_height));    
    // printf("Resizing image to %dx%d\n", res_width, res_height);
    
    int key_pressed = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
    // std::cout << "Pressed key is: " << key_pressed << std::endl;

    // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
    if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
        //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
        std::cout << "q is pressed. Exiting the program" << std::endl;
        cv::destroyWindow("1: Original_Image"); //destroy the created window
        return 0;
    }

    // STEP 1: Detect and Extract Chessboard Corners
    // Src: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
    greyscale(frame, grey_image); //source image
    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    patternfound = findChessboardCorners(grey_image, patternsize, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);
    if(patternfound){
      cornerSubPix(grey_image, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      frame.copyTo(mriwc); //Need to store the original image without corners for calibration purposes
      drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);
      std::cout << "Number of corners found: " << corner_set.size() << std::endl;
      std::cout << "First corner: x-> " << corner_set[0].x << " | y-> " << corner_set[0].y << std::endl;
    }

    if(key_pressed == 's'){ //Save the last calibration image          
      std::cout << "Saving image to file calibration_image_.jpeg" << std::endl;
      std::string folderName = "../calibration_images/";
      std::string image_name = "calibration_image_" + std::to_string(saved_image_id) + ".jpeg";
      cv::imwrite(folderName + image_name, mriwc);
      saved_image_id++;
    }
    cv::imshow(window_original_image, frame);

  }
return 0;
}


// Calculate the calibration parameters from the images in the directory
int calc_calib_params(char dirname[]){
  // char dirname[256] = "../calibration_images/";
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

  cv::Mat grey_image;
  bool patternfound;
  std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners

  std::vector<cv::Vec3f> point_set; //Stores 3D world coord of an image
  for(int i=0; i<patternsize.height; i++){
    for(int j=0; j<patternsize.width; j++){
      point_set.push_back(cv::Vec3f(j, -i, 0));
    }
  }

  // std::cout << point_set[0][0] << std::endl ;
  // std::cout << point_set[0][1] << std::endl ;
  // std::cout << point_set[0][2] << std::endl ;
  
  for (auto it : point_set)
    std::cout << it << std::endl;

	
  std::vector<std::vector<cv::Vec3f>> point_list; //Stores 3D world coord of all the images
	std::vector<std::vector<cv::Point2f>> corner_list; //Stores 2D corner coordinates of all the images in the image co-ord frame

  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;



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

      // STEP 1: Detect and Extract Chessboard Corners
      // Src: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
      greyscale(frame, grey_image); //source image
      //CALIB_CB_FAST_CHECK saves a lot of time on images
      //that do not contain any chessboard corners
      patternfound = findChessboardCorners(grey_image, patternsize, corner_set,
              cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
              + cv::CALIB_CB_FAST_CHECK);
      if(patternfound){
        cornerSubPix(grey_image, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);
        corner_list.push_back(corner_set);
        point_list.push_back(point_set);
        std::cout << "Number of corners found: " << corner_set.size() << std::endl;
        std::cout << "First corner: x-> " << corner_set[0].x << " | y-> " << corner_set[0].y << std::endl;
      }
      
      // int key_pressed = cv::waitKey(0); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
      // std::cout << "Pressed key is: " << key_pressed << std::endl;

      // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
      // if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
      //     //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
      //     std::cout << "q is pressed. Exiting the program" << std::endl;
      //     cv::destroyWindow("1: Original_Image"); //destroy the created window
      //     return 0;
      // }
      //   cv::imshow(window_original_image, frame);
    }
  }

  double data_for_initialization[9] = { 1, 0, (double)res_width/2, 0, 1, (double)res_height/2, 0, 0, 1};
  cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64FC1, data_for_initialization);
  // cv::Mat camera_calib_mat(cv::Matx33f::eye());  // intrinsic camera matrix
  cv::Vec<float, 5> distCoeffs(0, 0, 0, 0, 0); // distortion coefficients

  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
  int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
              cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
  cv::Size frameSize(res_width, res_height);

  std::cout << "Calibrating..." << std::endl;
  // 4. Call "float error = cv::calibrateCamera()" with the input coordinates
  // and output parameters as declared above...

  std::cout << "Before calibration" << "\ncameraMatrix =\n"
            << cameraMatrix << "\ndistCoeffs=\n"
            << distCoeffs << std::endl;

  float error = cv::calibrateCamera(point_list, corner_list, frameSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);

  std::cout << "Reprojection error = " << error << "\ncameraMatrix =\n"
            << cameraMatrix << "\ndistCoeffs=\n"
            << distCoeffs << std::endl;

  // Write to CSV: https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
  MyData m(1);
  cv::FileStorage fs("calibparam.yaml", cv::FileStorage::WRITE);
  // or:
  // FileStorage fs;
  // fs.open(filename, FileStorage::WRITE);
  // fs << "iterationNr" << 100;

  // fs << "strings" << "[";                              // text - string sequence
  // fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
  // fs << "]";                                           // close sequence
  
  // fs << "Mapping";                              // text - mapping
  // fs << "{" << "One" << 1;
  // fs <<        "Two" << 2 << "}";

  fs << "Camera_Matrix" << cameraMatrix;                                      // cv::Mat
  fs << "Distortion_Coefficients" << distCoeffs;
  fs << "Point_List" << point_list;
  fs << "Rotations" << rvecs;
  fs << "Translations" << tvecs;


  // fs << "MyData" << m;                                // your own data structures
  fs.release();                                       // explicit close
  std::cout << "Write Done." << std::endl;


  return 0;
}

