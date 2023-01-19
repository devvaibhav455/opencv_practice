// Dev Vaibhav
// Spring 2023 CS 5330
// Project 1: Real-time filtering
// Task 1. Read an image from a file and display it 

#include <opencv2/opencv.hpp>
#include <iostream>


// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

// argc is the number of command line arguments
// argv is an array of character arrays (command line arguments)
// argv[0] is the name of the executable function
int main(int argc, char** argv)
{
 
    char filename[256]; //To take user's desired image file as input 
  
    if( argc < 2 ) {
        printf("Usage %s <image filename>\n", argv[0] );
        exit(-1);
    }
    strcpy( filename, argv[1] ); // copy command line filename to a local variable

    // Read the image file
    cv::Mat image = cv::imread(filename,cv::ImreadModes::IMREAD_UNCHANGED);

    // Check for failure
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        // std::cin.get(); //wait for any key press
        return -1;
    }

    cv::String windowName = "Image display"; //Name of the window
    cv::namedWindow(windowName); // Create a window
    cv::imshow(windowName, image); // Show our image inside the created window.

    while(cv::waitKey(0) != 113){
      //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
    }
    cv::destroyWindow(windowName); //destroy the created window
    return 0;
}