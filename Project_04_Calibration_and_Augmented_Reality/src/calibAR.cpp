/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // To use sleep functionality
#include <filter.h>
#include <helper_functions.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "csv_util.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <modes.h>


// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

int main(int argc, char** argv)
{   
    std::cout << "Please enter the operating mode" << std::endl;
    std::cout << "\tc: Calculate camera calibration parameters" << std::endl;
    std::cout << "\tp: Projection mode" << std::endl;
    std::cout << "\t1: AR TV using Aruco tags" << std::endl;
    

    int reset_initially = 1;
    char mode[256];
    char source[256];
    char target_filename[256];
    strcpy(mode, argv[1] ); 
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  CAMERA CALIBRATION MODE: Captures pictures of checkerboard if not already taken and calibrates the camera   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (strcmp("c", mode) == 0){
        //Read images from directory and write feature vector to a csv file
        calibrate_camera();
        return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  OBJECT RECOGNITION MODE MODE: Takes image from user and finds the closest match in the DB                   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    else if (strcmp("p", mode) == 0){
        int source = 0; //0 :image | 1 : video from file | 2 : live video from webcam
        cv::Mat frame;

        if( argc == 3) {
            strcpy(target_filename, argv[2] );
            char *target_filename_char_star = target_filename;
            std::cout << "Reading image/ video from directory" << std::endl;
            source = 0; //It could be 0 or 1
        }else{
            source = 2;
        }
        project(source, target_filename);
        return 0;
    }

    else if (strcmp("1", mode) == 0){
        int ar_source = 0; //0 :image | 1 : video from file | 2 : live video from webcam
        cv::Mat frame;

        if( argc == 3) {
            strcpy(target_filename, argv[2] );
            char *target_filename_char_star = target_filename;
            std::cout << "Reading image/ video from directory" << std::endl;
            ar_source = 0; //It could be 0 or 1
        }else{
            ar_source = 2;
        }
        arucoTV(ar_source, target_filename);
        return 0;
    }
    
    return 0;
}
