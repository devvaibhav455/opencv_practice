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

// Import the aruco module in OpenCV
#include <opencv2/aruco.hpp>
 




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


int project(int &source, char *target_filename_char_star){

  //Source: 0 :image | 1 : video from file | 2 : live video from webcam
  cv::Mat frame;
  cv::VideoCapture *capdev;
  int fps;
  if(source == 0){
    // Check if the input is image or video file by counting the number of frames in the input
    
    capdev = new cv::VideoCapture(target_filename_char_star); 
    // Print error message if the stream is invalid
    if (!capdev->isOpened()){
      std::cout << "Error opening video stream or image file" << std::endl;
    }else{
      // Obtain fps and frame count by get() method and print
      // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
      fps = capdev->get(5);
      std::cout << "Frames per second in video:" << fps << std::endl;
  
      // Obtain frame_count using opencv built in frame count reading method
      // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
      int frame_count = capdev->get(7);
      std::cout << "  Frame count :" << frame_count << std::endl;

      (frame_count == 1) ? source = 0 : source = 1;
      std::cout << "Source is: " << source << std::endl;
    }

    if (source == 0){
      // Read the image file
      frame = cv::imread(target_filename_char_star);//,cv::ImreadModes::IMREAD_UNCHANGED);
      std::cout << "Reading image from disk successful. Number of channels in image: " << frame.channels() << std::endl;
      
      // Check for failure
      if (frame.empty()) {
          std::cout << "Could not open or find the image" << std::endl;
          // std::cin.get(); //wait for any key press
          return -1;
      }
    }
  }

  if (source == 2){
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
      throw std::runtime_error("Error");
      return -1;
    }
    fps = capdev->get(5);
    std::cout << "Input feed is camera" << std::endl;
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
  }
    

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
  
    
  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::Mat grey_image;
  
  std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners
  bool patternfound;
  cv::Mat rvec, tvec;

  int key_pressed;

  while (true) {
    // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
    // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    // std::cout << "Getting data from camera" << std::endl;
    
    
    if( source == 1){
      bool isSuccess = capdev->read(frame);

      // If frames are not there, close it
      if (isSuccess == false){
        std::cout << "Video file has ended. Running the video in loop" << std::endl;
        capdev->set(1,0);
        capdev->read(frame); //Src: https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
      }
    }else if( source == 2){
        // std::cout << "Getting data from camera" << std::endl;
        *capdev >> frame; //frame.type() is 16 viz. 8UC3
    }

    cv::resize(frame, frame, cv::Size(res_width, res_height));    
    // printf("Resizing image to %dx%d\n", res_width, res_height);
    
    

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
      // drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);
      std::cout << "Number of corners found: " << corner_set.size() << std::endl;
      
      //Src: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
      cv::solvePnP(point_list[0], corner_set, cameraMatrix, distCoeffs, rvec, tvec );

      // Project a 3D point (0, 0, 1000.0) onto the image plane.
      // We use this to draw a line sticking out of the nose
      draw_shape_on_image("wa_monument", frame, cameraMatrix, distCoeffs, rvec, tvec, corner_set);
      
    }

  
    cv::imshow(window_original_image, frame);
    //  = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
    // std::cout << "Pressed key is: " << key_pressed << std::endl;

    // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
    
    if(source == 0){
      key_pressed = cv::waitKey(0);
    }else{
      key_pressed = cv::waitKey(1000/fps); // It will play video at its original fps
    }
    if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
      //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
      std::cout << "q is pressed. Exiting the program" << std::endl;
      cv::destroyWindow("1: Original_Image"); //destroy the created window
      return 0;
    }

  }


  return 0;
}

int arucoTV(int &ar_source, char *target_filename_char_star){

  //ar_source: 0 :image | 1 : video from file | 2 : live video from webcam
  cv::Mat frame, frame_ar;
  cv::VideoCapture *capdev;
  int fps;
  if(ar_source == 0){
    // Check if the input is image or video file by counting the number of frames in the input
    
    capdev = new cv::VideoCapture(target_filename_char_star); 
    // Print error message if the stream is invalid
    if (!capdev->isOpened()){
      std::cout << "Error opening video stream or image file" << std::endl;
    }else{
      // Obtain fps and frame count by get() method and print
      // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
      fps = capdev->get(5);
      std::cout << "Frames per second in video:" << fps << std::endl;
  
      // Obtain frame_count using opencv built in frame count reading method
      // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
      int frame_count = capdev->get(7);
      std::cout << "  Frame count :" << frame_count << std::endl;

      (frame_count == 1) ? ar_source = 0 : ar_source = 1;
      
    }

    if (ar_source == 0){
      // Read the image file
      frame_ar = cv::imread(target_filename_char_star);//,cv::ImreadModes::IMREAD_UNCHANGED);
      std::cout << "Reading image from disk successful. Number of channels in image: " << frame.channels() << std::endl;
      
      // Check for failure
      if (frame_ar.empty()) {
          std::cout << "Could not open or find the image" << std::endl;
          // std::cin.get(); //wait for any key press
          return -1;
      }
    }
  }

  
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened()) {
    throw std::runtime_error("Error");
    return -1;
  }
  fps = capdev->get(5);
  std::cout << "Input feed is camera" << std::endl;
  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
              (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
  
  std::cout << "ar_source is: " << ar_source << std::endl;

  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::String window_artv_image = std::to_string(window_id) + " :AR TV";
  window_id++;

  while (true) {
    // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
    // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    // std::cout << "Getting data from camera" << std::endl;
    
    *capdev >> frame; //frame.type() is 16 viz. 8UC3
    cv::resize(frame, frame, cv::Size(res_width, res_height));    
    


    if( ar_source == 1){
      bool isSuccess = capdev->read(frame_ar);

      // If frames are not there, close it
      if (isSuccess == false){
        std::cout << "Video file has ended. Running the video in loop" << std::endl;
        capdev->set(1,0);
        capdev->read(frame_ar); //Src: https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
      }
    }else if( ar_source == 2){
      // std::cout << "Getting data from camera" << std::endl;
      frame_ar = frame;
    }

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

    // Load the dictionary that was used to generate the markers.
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250); // Try 50, 100, 1000 as well
    
    // Initialize the detector parameters using default values
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
                
    // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    
    // The ids of the detected markers are stored in a vector
    std::vector<int> markerIds;
                
    // Detect the markers in the image. Please not that the markers can be detected in any order. It is not fixed that id 0 will be detected first.
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    
    // draw detected markers: frame, corner id
    cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
    
    for (auto it:markerIds){
      std::cout << "Marker id: " << it << " "; 
    }
    std::cout << std::endl;

    // pts_src contains the 4 corners of the ar_source frame in clockwise order. TL, TR, BR, BL
    std::vector<cv::Point2i> pts_src{cv::Point2i(0,0), cv::Point2i(res_width,0), cv::Point2i(res_width,res_height), cv::Point2i(0,res_height) };
    
    
    std::vector<cv::Point2i> BoxCoordinates;
    if (markerIds.size() == 4){

      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 0) - markerIds.begin()][0]);
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 1) - markerIds.begin()][1]);
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 3) - markerIds.begin()][2]);
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 2) - markerIds.begin()][3]);
      // pts_dst contains the 4 corners where we want to show another video/ image (i.e. using corners of 4 aruco markers). They also need to be stores the  same way as pts_src
      std::vector<cv::Point2i> pts_dst = BoxCoordinates;
      //Resize frame_wr wrt the dimensions of the 


      // std::cout << "pts_src 0th points is: " << pts_src[1] <<  std::endl;

      // Compute homography from ar_source and destination points
      cv::Mat h = cv::findHomography(pts_src, pts_dst);
      
      // Warped image
      cv::Mat warpedImage;
                  
      // Warp ar_source image to destination based on homography
      cv::warpPerspective(frame_ar, warpedImage, h, frame.size(), cv::INTER_CUBIC);
      cv::imshow("warpPerspective output", warpedImage);       
      // Prepare a mask representing region to copy from the warped image into the original frame.
      cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
      cv::imshow("Mask original", mask);
      // Src: https://stackoverflow.com/questions/43629978/opencv-exception-fillconvexppoly
      cv::fillConvexPoly(mask, pts_dst, cv::Scalar(255, 255, 255));
      // Mask now contains 255 inside the bounding box rectangle
      cv::imshow("Mask after filling", mask);
      std::cout << " ########### REACHED HERE ###########" << std::endl;
                  
      // Erode the mask to not copy the boundary effects from the warping
      cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3) );
      cv::erode(mask, mask, element);
      cv::imshow("Mask after eroding", mask); //almost the same as mask before eroding
      
      // // Copy the masked warped image into the original frame in the mask region.
      // cv::Mat imOut = frame_ar.clone();
      // warpedImage.copyTo(imOut, mask);
      // cv::imshow(window_artv_image, warpedImage);
    }
    
    cv::imshow(window_original_image, frame);
  }
  return 0;
}