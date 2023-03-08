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
    

    int reset_initially = 1;
    char mode[256];
    char target_filename[256];
    strcpy(mode, argv[1] ); 
    
    // Resolution required for the window
    int res_width = 600; //columns
    int res_height = res_width*9/16; //rows
    
    


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  BASIC TRAINING MODE: Reads images from a directory and writes feature vectors to a csv file                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (strcmp("c", mode) == 0){
        //Read images from directory and write feature vector to a csv file
        calibrate_camera();
        return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  OBJECT RECOGNITION MODE MODE: Takes image from user and finds the closest match in the DB                   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    else if (strcmp("o", mode) == 0){
        int source;
        cv::Mat frame;

        if( argc == 3) {
            strcpy(target_filename, argv[2] );
            std::cout << "Reading image from directory" << std::endl;
            // Read the image file
            frame = cv::imread(target_filename);//,cv::ImreadModes::IMREAD_UNCHANGED);
            std::cout << "Number of channels: " << frame.channels() << std::endl;

            // Check for failure
            if (frame.empty()) {
                std::cout << "Could not open or find the image" << std::endl;
                // std::cin.get(); //wait for any key press
                return -1;
            }
            source = 0;
        }else{
            source = 1;
        }

        return 0;
    }
    
    
    return 0;
}
