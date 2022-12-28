//Uncomment the following line if you are compiling this code in Visual Studio
//#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <iostream>

// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// cmake . && make #The executable gets stored into the bin folder

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

int main(int argc, char** argv)
{
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        throw std::runtime_error("Error");
    }
    
    cv::Mat output(350,350,CV_8UC1);
    cv::Mat rgb_output(350,350,CV_8UC3);

    ImageOperator ImageOperator;

    // cv::namedWindow("Window");
    while (true) {
        cv::Mat frame;
        cam >> frame;
        cv::resize(frame, frame, cv::Size(400, 400));
        cv::imshow("bgr_frame", frame);

        cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
        cv::imshow("opencv_func", output);

        ImageOperator::to_gray_m1(frame,output);
        cv::imshow("to_gray_m1",output);

        ImageOperator::to_gray_m2(frame,output);
        cv::imshow("to_gray_m2",output);

        ImageOperator::to_gray_m3(frame,output);
        cv::imshow("to_gray_m3",output);

        ///No OpenCV
        const unsigned char* bgr_input = (unsigned char*)frame.data;
        unsigned char* gray_output = new unsigned char[frame.rows*frame.cols];
        ImageOperator::to_gray(bgr_input,frame.cols,frame.rows,frame.channels(),gray_output);
        cv::Mat output_gray(frame.rows, frame.cols, CV_8UC1, gray_output);
        cv::imshow("to_gray_no_opencv", output_gray);
        if(cv::waitKey(30) >= 0) break;
    }
    return 0;
}