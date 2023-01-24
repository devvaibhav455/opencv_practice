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

// Ref: https://soubhihadri.medium.com/image-processing-best-practices-c-280caadacb82

class ImageOperator{
public:
    ImageOperator() = default;
    ~ImageOperator() = default;
    
    static void to_gray_m1(const cv::Mat& input, cv::Mat& output);
    static void to_gray_m2(const cv::Mat& input, cv::Mat& output);
    static void to_gray_m3(const cv::Mat& input, cv::Mat& output);
    static void to_gray(const unsigned char* input,
                        const int width,
                        const int height,
                        const int channel,
                        unsigned char* output);
    static void rotate(const unsigned char* input,
                           const int width,
                           const int height,
                           const int channel,
                           unsigned char* &output); //Receiving the pointer via reference. It is essentially saying that the pointer is referenced or aliased as `output`. While calling it, we are using `output_rotate.data`. So, we are basically saying that `output_rotate.data` and `output` are one and the same thing and whatever changes are made to output will be made to `output_rotate.data` as well. Src: https://stackoverflow.com/questions/5789806/meaning-of-and-in-c. Check Mahesh (simple code for difference) and Juna Chavarro answer (read from right to left). Practice https://onlinegdb.com/n1d1nYx4S

    
};

    void ImageOperator::to_gray_m1(const cv::Mat &input, cv::Mat &output) {
            unsigned char *data_out = (unsigned char*)(output.data);
            int ind = 0;
            auto end = input.end<cv::Vec3b>();
            cv::MatConstIterator_<cv::Vec3b> it = input.begin<cv::Vec3b>();
            for (; it != end; ++it) {
                const unsigned char &r = (*it)[2];
                const unsigned char &g = (*it)[1];
                const unsigned char &b = (*it)[0];
                data_out[ind] = 0.3*r+0.59*g+0.11*b;
                ind++;
            }
            
        }

    void ImageOperator::to_gray_m2(const cv::Mat &input, cv::Mat &output) {
        unsigned char *data_in = (unsigned char*)(input.data);
        unsigned char *data_out = (unsigned char*)(output.data);

        int index = 0;
        int byte_size = input.channels()*input.rows*input.cols;
        while(index!=byte_size){
            data_out[index/input.channels()] = unsigned(0.11*data_in[index]+0.59*data_in[index+1]+0.3*data_in[index+2]);

            index+=3;
        }
        
    }

    void ImageOperator::to_gray_m3(const cv::Mat &input, cv::Mat &output) {
        unsigned char *data_in = (unsigned char*)(input.data);
        unsigned char *data_out = (unsigned char*)(output.data);

        int index = 0;
        for (int row = 0; row < input.rows; ++row) {
            for (int col = 0; col < input.cols*input.channels(); col+=input.channels()) {
                data_out[index]= 0.11*data_in[row*input.step+col]+
                                0.59*data_in[row*input.step+col+1]+
                                0.3*data_in[row*input.step+col+2];
                index++;
            }
        }
        
    }

    void ImageOperator::to_gray(const unsigned char* bgr_input,
                            const int width,
                            const int height,
                            const int channel,
                            unsigned char* gray_output){
        int index = 0;
        int step = channel*width;
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width*channel; col+=channel) {
                gray_output[index] = 0.11*bgr_input[row*step+col]+
                                0.59*bgr_input[row*step+col+1]+
                                0.3*bgr_input[row*step+col+2];
                index++;
            }
        }
        
    }

    void ImageOperator::rotate(const unsigned char* input,
                           const int width,
                           const int height,
                           const int channel,
                           unsigned char* &output){ //Receiving the pointer via reference
        
        // std::cout << "Rotate fn: width, height, channels are: " << width << " " << height << " " << " " << channel << std::endl;
        
        unsigned char* tmp = new unsigned char[width*height*channel]; //Storing value on the heap; https://stackoverflow.com/questions/5688417/how-to-initialize-unsigned-char-pointer

        int step = channel*width;

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; col+=1) {
                tmp[(col*width) + width - 1 - row] = input[(row*width) + col];
                
            }
        }
        output = tmp;
    }

int main(int argc, char** argv)
{
    // Task 2 | Displaying live video
    // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
    // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        throw std::runtime_error("Error");
        return -1;
    }
    
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);

    // Resolution required for the window
    int res_width = 400; //columns
    int res_height = res_width*9/16; //rows

    printf("Resizing image to %dx%d\n", res_width, res_height);

    // Usage of cv::Mat(int rows, int cols, int type)
    cv::Mat output(res_height,res_width,CV_8UC1); //Single channel matrix for greyscale image
    cv::Mat rgb_output(res_height,res_width,CV_8UC3); //3 channel matrix for color image
    cv::Mat short_c3_output(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
    

    ImageOperator ImageOperator;
    std::string desired_filter;
    cv::namedWindow("Color_Image");
    int change_brightness_level = 0;
    

    // cv::namedWindow("Window");
    while (true) {
        cv::Mat frame; //frame.type() is 0 viz. 8UC1 (https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/39780825#39780825)
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        *capdev >> frame; //frame.type() is 16 viz. 8UC3
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
            cv::destroyWindow("Color_Image"); //destroy the created window
            return 0;
        }else if (key_pressed == 's'){
            //Save image if 's' is pressed. 115 is s's ASCII value 
            std::cout << "Saving image to file saved_image.jpeg" << std::endl;
            cv::imwrite("saved_image.jpeg", frame);
        }else if (key_pressed == 'g'){
            //Show greyscale image using openCV function if 'g' is pressed. 71 is g's ASCII value 
            desired_filter = "GREYSCALE_OPENCV_FUNC";
            cv::namedWindow(desired_filter);
            std::cout << "Showing greyscale output using OpenCV function" << std::endl;
        }else if (key_pressed == 'h'){
            // Show greyscale image using user defined function if 'h' is pressed. 104 is h's ASCII value
            desired_filter = "GREYSCALE_USER_FUNC";
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
            greyscale(frame, output);
            int current_brightness = cv::mean(output).val[0]*100/255;
            if (key_pressed == 82){change_brightness_level += 10;} //Increase brightness if up arrow key is pressed
            else {change_brightness_level -= 10;} //Decrease brightness if down arrow key is pressed
            cv::namedWindow(desired_filter);
         }



        if (desired_filter == "GREYSCALE_OPENCV_FUNC"){
            // Explanation for the math part: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
            cv::imshow(desired_filter, output);
        }else if (desired_filter == "GREYSCALE_USER_FUNC"){
            greyscale(frame, output);
            cv::imshow(desired_filter, output);
        }else if (desired_filter == "BLURRED_GAUSSIAN_FILTER"){
            blur5x5(frame, rgb_output);
            cv::imshow(desired_filter, rgb_output);
        }else if (desired_filter == "SOBEL_X"){
            sobelX3x3(frame, short_c3_output);
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3
            cv::imshow(desired_filter, rgb_output);
        }else if (desired_filter == "SOBEL_Y"){
            sobelY3x3(frame, short_c3_output);
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "GRADIENT_MAGNITUDE"){
            cv::Mat temp_sx_short16s(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
            cv::Mat temp_sy_short16s(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
            sobelX3x3(frame, temp_sx_short16s); // Computing Sobel X
            sobelY3x3(frame, temp_sy_short16s); // Computing Sobel Y
            magnitude(temp_sx_short16s, temp_sy_short16s, short_c3_output); // Combining Sobel X and Sobel Y
            cv::convertScaleAbs(short_c3_output, rgb_output); //Converting 16SC3 to 8UC3 to make it suitable for display
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "BLUR_QUANTIZE"){
            blurQuantize(frame, rgb_output, 15);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "LIVE_CARTOONIZATION"){
            cartoon(frame, rgb_output, 15, 20);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "NEGATIVE"){
            negative(frame, rgb_output);
            cv::imshow(desired_filter , rgb_output);
        }else if (desired_filter == "CHANGE_BRIGHTNESS"){
            // Changing the brightness level by change_brightness_level in percentage
            frame = frame + cv::Scalar(change_brightness_level,change_brightness_level,change_brightness_level)*2.55;
            cv::imshow(desired_filter , frame);
        }

        

        
            


        // ImageOperator::to_gray_m1(frame,output);
        // cv::imshow("to_gray_m1",output);

        // ImageOperator::to_gray_m2(frame,output);
        // cv::imshow("to_gray_m2",output);

        // ImageOperator::to_gray_m3(frame,output);
        // cv::imshow("to_gray_m3",output);

        ///No OpenCV
        // const unsigned char* bgr_input = (unsigned char*)frame.data;
        // unsigned char* gray_output = new unsigned char[frame.rows*frame.cols];
        // ImageOperator::to_gray(bgr_input,frame.cols,frame.rows,frame.channels(),gray_output);
        // cv::Mat output_gray(frame.rows, frame.cols, CV_8UC1, gray_output);
        // cv::imshow("to_gray_no_opencv", output_gray);

        // std::cout << "Data after B/W conversion m3= " << std::endl << " " << output << std::endl << std::endl;

        // cv::Mat output_rotate(output.rows, output.cols, CV_8UC1);
        // ImageOperator::rotate(output.data, output.cols, output.rows, output.channels(), output_rotate.data);
        // cv::imshow("Rotated by 90deg", output_rotate);

        // ImageOperator::rotate(output_rotate.data, output_rotate.cols, output_rotate.rows, output.channels(), output_rotate.data);
        // cv::imshow("Rotated by 180deg", output_rotate);
        
        // std::cout << "M after rotation = " << std::endl << " " << output_rotate << std::endl << std::endl;
        // sleep(20);

        // if(cv::waitKey(30) >= 0) break;
    }
    return 0;
}