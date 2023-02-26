/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
Notes: Some code adapted from Project 1/2
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
    int reset_initially = 1;
    char mode[256];
    char target_filename[256];
    strcpy(mode, argv[1] ); 
    
    // Resolution required for the window
    int res_width = 600; //columns
    int res_height = res_width*9/16; //rows
    
    std::cout << "Please enter the operating mode" << std::endl;
    std::cout << "\tb: Basic training mode. Reads images from a directory and writes feature vectors to a csv file" << std::endl;
    std::cout << "\to: Object recognition mode. Takes image from user and finds the closest match in the DB" << std::endl;
    std::cout << "\tk: KNN classifier mode. Takes image from user and finds the closest match in the DB using KNN search" << std::endl;
    std::cout << "\tl: Testing mode. Takes live feed from the camera to tune and test the system" << std::endl;


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  BASIC TRAINING MODE: Reads images from a directory and writes feature vectors to a csv file                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (strcmp("b", mode) == 0){
        //Read images from directory and write feature vector to a csv file
        basic_training_mode_from_directory();
        return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  OBJECT RECOGNITION MODE MODE: Takes image from user and finds the closest match in the DB                   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    else if (strcmp("o", mode) == 0){
        strcpy(target_filename, argv[2] ); 
        object_recognition_mode(target_filename);
        return 0;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  KNN classifier mode. Takes image from user and finds the closest match in the DB using KNN search           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    else if (strcmp("k", mode) == 0){
        //KNN training mode

        // K value for the KNN classifier
        int k = atoi(argv[2]);
        strcpy(target_filename, argv[3] ); 
        char *target_filename_char_star = target_filename;
        std::cout << "k is: " << k << std::endl; 
        knn_classifier(k, target_filename_char_star);
        return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                  Testing mode. Takes live feed from the camera to tune and test the system                                   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    else if (strcmp("t", mode) == 0){
        // Testing mode

        // Use image as input if it is passed as a CLI, otherwise use video input from webcam.
        int source; //0: means image as input | 1 means camera live feed is input
        cv::Mat frame; //frame.type() is 0 viz. 8UC1 (https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/39780825#39780825)

        if( argc == 3) {
            strcpy(target_filename, argv[2] );
            std::cout << "Reading images from directory" << std::endl;
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
        testing_mode(source, frame);  
        return 0;
    }
    
    return 0;
}
