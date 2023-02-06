/*Dev Vaibhav
Spring 2023 CS 5330
Project 1: Real-time filtering
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // To use sleep functionality
#include <filter.h>


// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

int main(int argc, char** argv)
{
    // Task 2 | Displaying live video
    // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
    // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
    
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    cv::Mat frame; //frame.type() is 0 viz. 8UC1 (https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/39780825#39780825)
    // Use image as input if it is passed as a CLI, otherwise use video input from webcam.

    if( argc == 2) {
        char filename[256]; //To take user's desired image file as input 
        strcpy( filename, argv[1] ); // copy command line filename to a local variable
        // Read the image file
        frame = cv::imread(filename,cv::ImreadModes::IMREAD_UNCHANGED);

        // Check for failure
        if (frame.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
            // std::cin.get(); //wait for any key press
            return -1;
        }

        // cv::String windowName = "Image display"; //Name of the window
        // cv::namedWindow(windowName); // Create a window
        // cv::imshow(windowName, frame); // Show our image inside the created window.
    }else{
        if (!capdev->isOpened()) {
            throw std::runtime_error("Error");
            return -1;
        }
        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
    }


    // Resolution required for the window
    int res_width = 600; //columns
    int res_height = res_width*9/16; //rows

    printf("Resizing image to %dx%d\n", res_width, res_height);

    // Usage of cv::Mat(int rows, int cols, int type)
    cv::Mat output(res_height,res_width,CV_8UC1); //Single channel matrix for greyscale image
    cv::Mat rgb_output(res_height,res_width,CV_8UC3); //3 channel matrix for color image
    cv::Mat short_c3_output(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
    cv::Mat filtered_output;


    std::string desired_filter;
    cv::namedWindow("Color_Image");
    int change_brightness_level = 0;
    std::string user_caption;
    
    // Define the codec and create VideoWriter object.The output is stored in 'saved_video.avi' file.
    // Define the fps to be equal to 10. Also frame size is passed.
    
    cv::VideoWriter video_original("saved_video_original.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(res_width,res_height));;
    cv::VideoWriter video_color("saved_video_color.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(res_width,res_height));;
    cv::VideoWriter video_grey("saved_video_grey.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(res_width,res_height), 0);;
    int save_video = 0;
    int save_video_grey = 0;
    // cv::namedWindow("Window");
    while (true) {
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        if( argc == 1){
            *capdev >> frame; //frame.type() is 16 viz. 8UC3
        }
        // cv::Mat image = cv::imread(filename,cv::ImreadModes::IMREAD_UNCHANGED);~/Pictures/Screenshots
        cv::resize(frame, frame, cv::Size(res_width, res_height));
        
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame.at<cv::Vec3b>(0,0) << std::endl; 
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        
        cv::imshow("Color_Image", frame);
        
        int key_pressed = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
        // std::cout << "Pressed key is: " << key_pressed << std::endl;

        // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
        if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            std::cout << "q is pressed. Exiting the program" << std::endl;
            video_original.release();
            video_color.release();
            video_grey.release();
            cv::destroyWindow("Color_Image"); //destroy the created window
            return 0;
        }else if (key_pressed == 's'){
            //Save image if 's' is pressed. 115 is s's ASCII value 
            std::cout << "Saving image to file saved_image.jpeg" << std::endl;
            cv::imwrite("saved_image.jpeg", frame);
        }else if (key_pressed == 'g'){
            //Show greyscale image using openCV function if 'g' is pressed. 71 is g's ASCII value 
            desired_filter = "GREYSCALE_OPENCV_FUNC";
            save_video_grey = 1;
            cv::namedWindow(desired_filter);
            std::cout << "Showing greyscale output using OpenCV function" << std::endl;
        }else if (key_pressed == 'h'){
            // Show greyscale image using user defined function if 'h' is pressed. 104 is h's ASCII value
            desired_filter = "GREYSCALE_USER_FUNC";
            save_video_grey = 1;
            cv::namedWindow(desired_filter);
            std::cout << "Showing greyscale output using user defined function" << std::endl;
        }else if (key_pressed == 'b'){
            //Show blurred version of the image using separable Gaussian filter if b (ASCII = 98) is pressed
            desired_filter = "BLURRED_GAUSSIAN_FILTER";
            cv::namedWindow(desired_filter);
            std::cout << "Showing blurred version of the image using Separable Gaussian Filter [1 2 4 2 1] vertical and horizontal" << std::endl;
        }else if (key_pressed == 'x'){
            //Show Sobel X version of the image using separable Sobel X 3x3 filter if x is pressed. 120 is x's ASCII value.
            desired_filter = "SOBEL_X";
            cv::namedWindow(desired_filter);
            std::cout << "Showing Sobel X version of the image using Separable filters [1 2 1]^T vertical and [-1 0 1] horizontal" << std::endl;
        }else if (key_pressed == 'y'){
            //Show Sobel Y version of the image using separable Sobel Y 3x3 filter if y is pressed. 121 is x's ASCII value.
            desired_filter = "SOBEL_Y";
            cv::namedWindow(desired_filter);
            std::cout << "Showing Sobel Y version of the image using Separable filters [1 0 -1]^T vertical and [1 2 1] horizontal" << std::endl;
        }else if (key_pressed == 'm'){
            //Show Gradient Magnitude image if m is pressed. 109 is m's ASCII value.
            desired_filter = "GRADIENT_MAGNITUDE";
            cv::namedWindow(desired_filter);
            std::cout << "Showing Gradient Magnitude version of the image" << std::endl;
        }else if (key_pressed == 'l'){
            //Show blur quantized image if l (ASCII = 108) is pressed.
            desired_filter = "BLUR_QUANTIZE";
            cv::namedWindow(desired_filter);
            std::cout << "Showing blur quantized version of the image" << std::endl;
        }else if (key_pressed == 'c'){
            //Show live video cartoonization function using the gradient magnitude and blur/quantize filters if c (ASCII = 99) is pressed.
            desired_filter = "LIVE_CARTOONIZATION";
            cv::namedWindow(desired_filter);
            std::cout << "live video cartoonization function using the gradient magnitude and blur/quantize filters" << std::endl;
        }else if (key_pressed == 'n'){
            //Show negative image if n (ASCII = 110) is pressed.
            desired_filter = "NEGATIVE";
            cv::namedWindow(desired_filter);
            std::cout << "Showing the image negative" << std::endl;
        }else if (key_pressed == 82 || key_pressed == 84){//Change brightness when up/ down arrow is pressed
            desired_filter = "CHANGE_BRIGHTNESS";
            if (key_pressed == 82){change_brightness_level += 10;} //Increase brightness if up arrow key is pressed
            else {change_brightness_level -= 10;} //Decrease brightness if down arrow key is pressed
            cv::namedWindow(desired_filter);
         }else if (key_pressed == 49){//Sharpen the image when 1 is pressed
            desired_filter = "SHARPEN";
            cv::namedWindow(desired_filter);
            std::cout << "Showing sharpened image" << std::endl;
         }else if (key_pressed == 50){//Sketch the image when 2 is pressed
            desired_filter = "SKETCH";
            cv::namedWindow(desired_filter);
            std::cout << "Showing sketched image" << std::endl;
         }else if (key_pressed == 51){//Add captions to the image when 3 is pressed
            desired_filter = "CAPTION";
            cv::namedWindow(desired_filter);
            std::cout << "Please enter the caption:" << std::endl;
            std::cin >> user_caption;
            std::cout << "Showing image with caption" << std::endl;
         }else if (key_pressed == 52){//Let the user save the video with special effect when 4 is pressed
            save_video = 1;
            std::cout << "Saving video with special effects!" << std::endl;
         }



        if (desired_filter == "GREYSCALE_OPENCV_FUNC"){
            // Explanation for the math part: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
            output.copyTo(filtered_output);
            cv::imshow(desired_filter, output);
        }else if (desired_filter == "GREYSCALE_USER_FUNC"){
            greyscale(frame, output);
            output.copyTo(filtered_output);
            cv::imshow(desired_filter, output);
        }else if (desired_filter == "BLURRED_GAUSSIAN_FILTER"){
            blur5x5(frame, rgb_output);
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter, rgb_output);
        }else if (desired_filter == "SOBEL_X"){
            sobelX3x3(frame, short_c3_output);
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter, rgb_output);
        }else if (desired_filter == "SOBEL_Y"){
            sobelY3x3(frame, short_c3_output);
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "GRADIENT_MAGNITUDE"){
            cv::Mat temp_sx_short16s(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
            cv::Mat temp_sy_short16s(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
            sobelX3x3(frame, temp_sx_short16s); // Computing Sobel X
            sobelY3x3(frame, temp_sy_short16s); // Computing Sobel Y
            magnitude(temp_sx_short16s, temp_sy_short16s, short_c3_output); // Combining Sobel X and Sobel Y
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3 to make it suitable for display
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "BLUR_QUANTIZE"){
            blurQuantize(frame, rgb_output, 15);
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "LIVE_CARTOONIZATION"){
            cartoon(frame, rgb_output, 15, 20);
            cv::imshow(desired_filter , rgb_output);
            rgb_output.copyTo(filtered_output);
        }else if (desired_filter == "NEGATIVE"){
            negative(frame, rgb_output);
            cv::imshow(desired_filter , rgb_output);
            rgb_output.copyTo(filtered_output);
        }else if (desired_filter == "CHANGE_BRIGHTNESS"){
            // Changing the brightness level by change_brightness_level in percentage
            rgb_output = frame + cv::Scalar(change_brightness_level,change_brightness_level,change_brightness_level)*2.55;
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "SHARPEN"){
            cv::Mat temp_sx_short16s(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
            sharpen(frame, temp_sx_short16s);
            cv::convertScaleAbs(temp_sx_short16s, rgb_output); //Converting 16SC3 to 8UC3 to make it suitable for 
            rgb_output.copyTo(filtered_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "SKETCH"){
            sketch(frame, output);
            output.copyTo(filtered_output);
            cv::imshow(desired_filter , output);
        }else if (desired_filter == "CAPTION"){
            caption(frame, output, user_caption);
            output.copyTo(filtered_output);
            cv::imshow(desired_filter , output);
        }

        if (save_video_grey == 1){
            video_original.write(frame);
            video_grey.write(filtered_output);
        }else {
            video_original.write(frame);
            video_color.write(filtered_output);
        }
    }
    return 0;
}