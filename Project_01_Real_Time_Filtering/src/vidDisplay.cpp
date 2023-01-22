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
    cv::Mat rgb_output(res_height,res_width,CV_8UC3), tmp_mat; //3 channel matrix for color image
    cv::Mat rgb_short_output(res_height, res_width, CV_16SC3);

    ImageOperator ImageOperator;
    
    int greyscale_mode_opencv = 0;
    int greyscale_mode_user = 0;
    int blur_mode = 0;
    int sobel_x_mode = 0; 
    cv::namedWindow("Color_Image");
    
    

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

        // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
        if(key_pressed == 113){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            std::cout << "q is pressed. Exiting the program" << std::endl;
            cv::destroyWindow("Color_Image"); //destroy the created window
            return 0;
        }else if (key_pressed == 115){
            //Save image if 's' is pressed. 115 is s's ASCII value 
            std::cout << "Saving image to file saved_image.jpeg" << std::endl;
            cv::imwrite("saved_image.jpeg", frame);
        }else if (key_pressed == 103){
            //Show greyscale image using openCV function if 'g' is pressed. 71 is g's ASCII value 
            cv::namedWindow("greyscale_opencv_func");
            greyscale_mode_opencv = 1;
            std::cout << "Showing greyscale output using OpenCV function" << std::endl;
        }else if (key_pressed == 104){
            // Show greyscale image using user defined function if 'h' is pressed. 104 is h's ASCII value
            cv::namedWindow("greyscale_user_func");
            greyscale_mode_user = 1;
            std::cout << "Showing greyscale output using user defined function" << std::endl;
        }else if (key_pressed == 98){
            //Show blurred version of the image using separable Gaussian filter
            cv::namedWindow("Blurred_Gaussian_Filter");
            blur_mode = 1;
            std::cout << "Showing blurred version of the image using Separable Gaussian Filter [1 2 4 2 1] vertical and horizontal" << std::endl;
        }else if (key_pressed == 120){
            //Show Sobel X version of the image using separable Sobel X 3x3 filter if x is pressed. 120 is x's ASCII value.
            cv::namedWindow("Sobel_X");
            sobel_x_mode = 1;
            std::cout << "Showing Sobel X version of the image using Separable filters [1 2 1]^T vertical and [-1 0 1] horizontal" << std::endl;
        }


        if (greyscale_mode_opencv == 1){
            // Explanation for the math part: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
            cv::imshow("greyscale_opencv_func", output);
        }else if (greyscale_mode_user == 1){
            greyscale(frame, output);
            cv::imshow("greyscale_user_func", output);
        }else if (blur_mode == 1){
            blur5x5(frame, rgb_output);
            cv::imshow("Blurred_Gaussian_Filter", rgb_output);
        }else if (sobel_x_mode == 1){
            sobelX3x3(frame, tmp_mat);
            // cv::convertScaleAbs(tmp_mat, rgb_short_output);
            // printf("Size of sobel input: %dx%d | sobel output size: %dx%d", frame.cols, frame.rows, tmp_mat.cols, tmp_mat.rows);
            cv::imshow("Sobel_X", tmp_mat);
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