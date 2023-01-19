// Dev Vaibhav
// Spring 2023 CS 5330
// Project 1: Real-time filtering
// Task 2. Display live video  

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // To use sleep functionality


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

    int res_width = 400; //columns
    int res_height = 400; //rows


    cv::Mat output(res_height,res_width,CV_8UC1); //Single channel matrix for grayscale image
    cv::Mat rgb_output(res_height,res_width,CV_8UC3); //3 channel matrix for color image

    ImageOperator ImageOperator;
    cv::namedWindow("Color_Image");

    // cv::namedWindow("Window");
    while (true) {
        cv::Mat frame; //frame.type() is 0 viz. 8UC1 (https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/39780825#39780825)
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        *capdev >> frame; //frame.type() is 16 viz. 8UC3
        // cv::resize(frame, frame, cv::Size(res_height, res_width));
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame.at<cv::Vec3b>(0,0) << std::endl; 
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        cv::imshow("Color_Image", frame);
        if(cv::waitKey(25) == 113){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            cv::destroyWindow("Color_Image"); //destroy the created window
            return 0;
        }else if (cv::waitKey(0) == 115){
            //Save image if 's' is pressed. 115 is s's ASCII value 
            cv::imwrite("saved_image.jpeg", frame);
        }
            

        // cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
        // cv::imshow("opencv_func", output);

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