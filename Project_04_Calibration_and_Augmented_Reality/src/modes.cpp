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

  //READ all the images from a directory, calculate calibration parameters using them and writes them to a yaml file
  calc_calib_params(dirname);

 return 0;
}


int project(){

  // Read calibration parameters from the yaml file
  std::cout << std::endl << "Reading: " << std::endl;
  cv::FileStorage fs;
  fs.open("calibparam.yaml", cv::FileStorage::READ);
  int itNr;
  //fs["iterationNr"] >> itNr;
  itNr = (int) fs["iterationNr"];
  std::cout << itNr << std::endl;
  if (!fs.isOpened())
  {
      std::cerr << "Failed to open " << "calibparam.yaml" << std::endl;
      return 1;
  }
  
  cv::FileNode n;// = fs["strings"];  
                         // Read string sequence - Get node
  // if (n.type() != cv::FileNode::SEQ)
  // {
  //     std::cerr << "strings is not a sequence! FAIL" << std::endl;
  //     return 1;
  // }
  // cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
  // for (; it != it_end; ++it)
  //     std::cout << (std::string)*it << std::endl;
  
  // n = fs["Mapping"];                                // Read mappings from a sequence
  // std::cout << "Two  " << (int)(n["Two"]) << "; ";
  // std::cout << "One  " << (int)(n["One"]) << std::endl << std::endl;
  // MyData m;
  // fs["MyData"] >> m;                                 // Read your own structure_
  // std::cout << "MyData = " << std::endl << m << std::endl << std::endl;

  cv::Mat cameraMatrix;
  cv::Vec<float, 5> distCoeffs;
  std::vector<std::vector<cv::Vec3f>> point_list;

  fs["Camera_Matrix"] >> cameraMatrix;                                      // Read cv::Mat
  fs["Distortion_Coefficients"] >> distCoeffs;
  fs["Point_List"] >> point_list;


  
  std::cout << std::endl
      << "Camera_Matrix = " << cameraMatrix << std::endl;
  // std::cout << " ############### REACHED HERE ############### " << std::endl;

  std::cout << "Distortion_Coefficients = " << distCoeffs << std::endl << std::endl;
  std::cout << "Point_List = " << point_list[0][0] << std::endl << std::endl;
  //Show default behavior for non existing nodes
  // std::cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
  // fs["NonExisting"] >> m;
  // cout << endl << "NonExisting = " << endl << m << endl;
  
  std::cout << std::endl
  << "Tip: Open up " << "calibparam.yaml" << " with a text editor to see the serialized data." << std::endl;
  fs.release();

  // Start a video loop

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
  cv::Mat rvec, tvec;

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
      // drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);
      std::cout << "Number of corners found: " << corner_set.size() << std::endl;
      
      //Src: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
      cv::solvePnP(point_list[0], corner_set, cameraMatrix, distCoeffs, rvec, tvec );

      // Project a 3D point (0, 0, 1000.0) onto the image plane.
      // We use this to draw a line sticking out of the nose
      draw_shape_on_image("wa_monument", frame, cameraMatrix, distCoeffs, rvec, tvec, corner_set);
      
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