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
    

    int reset_initially = 1;
    char mode[256];
    char target_filename[256];
    strcpy(mode, argv[1] ); 
    
    // Resolution required for the window
    // int res_width = 600; //columns
    // int res_height = res_width*9/16; //rows
    
    


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
        project();
        return 0;
    }
    
    return 0;
}
