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


// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

// Some parameters to display the trackbar to find the threshold using GUI for HSV image
// Src: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
const int max_value_H = 360/2;
const int max_value = 255;
const cv::String window_hsv_thresholded = "3: HSV Thresholded";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

// Trackbar parameters for the grey masked image
const cv::String window_grey_masked = "Grey Masked";
const int threshold_slider_max = 255;
int threshold_grey = 0;


// static void on_low_H_thresh_trackbar(int, void *)
// {
//     low_H = cv::min(high_H-1, low_H);
//     cv::setTrackbarPos("Low H", window_hsv_thresholded, low_H);
// }
// static void on_high_H_thresh_trackbar(int, void *)
// {
//     high_H = cv::max(high_H, low_H+1);
//     cv::setTrackbarPos("High H", window_hsv_thresholded, high_H);
// }
// static void on_low_S_thresh_trackbar(int, void *)
// {
//     low_S = cv::min(high_S-1, low_S);
//     cv::setTrackbarPos("Low S", window_hsv_thresholded, low_S);
// }
// static void on_high_S_thresh_trackbar(int, void *)
// {
//     high_S = cv::max(high_S, low_S+1);
//     cv::setTrackbarPos("High S", window_hsv_thresholded, high_S);
// }
// static void on_low_V_thresh_trackbar(int, void *)
// {
//     low_V = cv::min(high_V-1, low_V);
//     cv::setTrackbarPos("Low V", window_hsv_thresholded, low_V);
// }
// static void on_high_V_thresh_trackbar(int, void *)
// {
//     high_V = cv::max(high_V, low_V+1);
//     cv::setTrackbarPos("High V", window_hsv_thresholded, high_V);
// }
// static void on_threshold_grey_trackbar( int, void* )
// {
//     threshold_grey = threshold_grey;
//     cv::setTrackbarPos("Grey Threshold", window_grey_masked, threshold_grey);
// }




int main(int argc, char** argv)
{
    // Step 1 | Displaying live video
    // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
    // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
    
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    cv::Mat frame; //frame.type() is 0 viz. 8UC1 (https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/39780825#39780825)
    // Use image as input if it is passed as a CLI, otherwise use video input from webcam.

    if( argc == 2) {
        std::cout << "Reading image from CLI" << std::endl;
        char filename[256]; //To take user's desired image file as input 
        strcpy( filename, argv[1] ); // copy command line filename to a local variable
        // Read the image file
        frame = cv::imread(filename);//,cv::ImreadModes::IMREAD_UNCHANGED);
        std::cout << "Number of channels: " << frame.channels() << std::endl;

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
        std::cout << "Input feed is camera" << std::endl;
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
    cv::Mat grey_output(res_height,res_width,CV_8UC1); //Single channel matrix for greyscale image
    cv::Mat rgb_output(res_height,res_width,CV_8UC3); //3 channel matrix for color image
    cv::Mat short_c3_output(res_height,res_width,CV_16SC3); //3 channel matrix of shorts for color image
    cv::Mat filtered_output, hsv_frame, hsv_frame_opencv;


    std::string desired_filter;
    cv::namedWindow("1: Original_Image");
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

    cv::namedWindow(window_hsv_thresholded);
    cv::namedWindow(window_grey_masked);
    // Trackbars to set thresholds for HSV values
    // cv::createTrackbar("Low H", window_hsv_thresholded, &low_H, max_value_H, on_low_H_thresh_trackbar);
    // cv::createTrackbar("High H", window_hsv_thresholded, &high_H, max_value_H, on_high_H_thresh_trackbar);
    // cv::createTrackbar("Low S", window_hsv_thresholded, &low_S, max_value, on_low_S_thresh_trackbar);
    // cv::createTrackbar("High S", window_hsv_thresholded, &high_S, max_value, on_high_S_thresh_trackbar);
    // cv::createTrackbar("Low V", window_hsv_thresholded, &low_V, max_value, on_low_V_thresh_trackbar);
    // cv::createTrackbar("High V", window_hsv_thresholded, &high_V, max_value, on_high_V_thresh_trackbar);
    // cv::createTrackbar("Grey Threshold", window_grey_masked, &threshold_grey, threshold_slider_max, on_threshold_grey_trackbar);
    // on_threshold_grey_trackbar(threshold_grey, 0);

    cv::Mat hsv_thresholded;
    // threshold_grey = 0;
    frame = cv::imread("../sample_set/img2P3.png"); //USED FOR TESTING/ DEBUGGING ONLY
    while (true) {
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        if( argc == 1){
            *capdev >> frame; //frame.type() is 16 viz. 8UC3
        }
        
        
        cv::resize(frame, frame, cv::Size(res_width, res_height));
        
        // std::cout << "Resized size is:" << frame.size() << std::endl; //Prints columnsxrows
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame.at<cv::Vec3b>(0,0) << std::endl; 
        // std::cout << "Frame after input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        cv::imshow("1: Original_Image", frame);
        
        int key_pressed = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
        // std::cout << "Pressed key is: " << key_pressed << std::endl;

        // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
        if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            std::cout << "q is pressed. Exiting the program" << std::endl;
            video_original.release();
            video_color.release();
            video_grey.release();
            cv::destroyWindow("1: Original_Image"); //destroy the created window
            return 0;
        }
        
        // Need to fix user implementation of bgr to hsv converter not working
        // bgr_to_hsv(frame,hsv_frame);
        // cv::convertScaleAbs(hsv_frame, rgb_output); //Converting 16SC3 to 8UC3
        // rgb_output.copyTo(filtered_output);
        // cv::namedWindow("HSV Image user function");
        // cv::imshow("HSV Image user function", filtered_output);

        //Convert image to BW and calculate histogram
        // cv::cvtColor(frame, grey_output, cv::COLOR_BGR2GRAY);
        
        // cv::cvtColor(frame, hsv_frame_opencv, cv::COLOR_BGR2HSV);
        // cv::namedWindow("2: HSV Image OpenCV function");
        // cv::imshow("2: HSV Image OpenCV function", hsv_frame_opencv);
        // hsv_frame_opencv = hsv_frame_opencv + cv::Scalar(0,-20,0); //Making pixels dark

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                  Histogram calculation for the HSV Image                                                                     //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // const int histSize = 256;
        // // A vector for storing individual channels of HSV image
        // std::vector<cv::Mat> hsv_planes;
        // // Split the channels of hsv and store it in the vector
        // cv::split(hsv_frame_opencv, hsv_planes);
        // float range[] = {0, 256};
        // const float *histRange = {range};
        // bool uniform = true;
        // bool accumulate = false;
        // cv::Mat h_hist, s_hist, v_hist;
        // cv::calcHist(&hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &histSize,
        //             &histRange, uniform, accumulate);
        // cv::calcHist(&hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &histSize,
        //             &histRange, uniform, accumulate);
        // cv::calcHist(&hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize,
        //             &histRange, uniform, accumulate);
        // drawHistogram(h_hist,s_hist,v_hist);
        

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //           Detect the object based on HSV Range Values                                                                        //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

           
        // More info: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
        // That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the specified 1D, 2D, 3D, ... box and 0 otherwise. It outputs a binary image
        
        // cv::inRange(hsv_frame_opencv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), hsv_thresholded);
        // cv::inRange(hsv_frame_opencv, cv::Scalar(0, 29, 0), cv::Scalar(37, 255, 255), hsv_thresholded); // Values found after changing the slider
        // Show the frames
        // imshow(window_hsv_thresholded, hsv_thresholded);


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //           Convert the hsv_thresholded pixels to greyscale image by creating a mask (3 channel mask                           //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
        //Implementing mask for thresholded HSV using. Src: https://stackoverflow.com/questions/14582082/merging-channels-in-opencv
        cv::Mat mask_3c(frame.size(), CV_8UC3, cv::Scalar(0,0,0)), mask_1c(frame.size(), CV_8UC1, cv::Scalar(0));
        // hsv_thresholded.copyTo(mask_1c);
        // mask_1c_to_3c(mask_1c, mask_3c);
        // cv::Mat color_masked = mask_3c & frame;
        // cv::cvtColor(color_masked, grey_output, cv::COLOR_BGR2GRAY);
        // cv::imshow(window_grey_masked, color_masked);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                  Applying thresholding to the masked greyscale image                                                         //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cv::Mat thresholded_image;
        // Src: https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/; https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
        // cv::threshold(grey_output, thresholded_image, threshold_grey, 255, cv::THRESH_BINARY ); //THRESH_BINARY_INV
        // cv::threshold(grey_output, thresholded_image, 22, 255, cv::THRESH_BINARY);

        thresholding(frame, thresholded_image);
        cv::String window_binary_masked = "4: Binary masked";
        cv::namedWindow(window_binary_masked);
        cv::imshow(window_binary_masked, thresholded_image);


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                  Step 2: Clean up the binary image                                                                           //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        cv::Mat cleaned_image(frame.size(), CV_8UC1), temp_cleaned(frame.size(), CV_8UC1);
        grassfire_tf(thresholded_image,cleaned_image, 8, 0, 20); // Perform erosion with 8 connectedness for at most 20 times
        cv::namedWindow("5: Cleaned Image");
        cv::imshow("5: Cleaned Image", cleaned_image);
        std::cout << "################### Reached here ###################" << std::endl;
        // grassfire_tf(temp_cleaned,cleaned_image, 8, 1, 4);

        // while(cv::waitKey(0) != 113){
        // //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
        // }
        // cv::destroyWindow("Cleaned Image"); //destroy the created window

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                  Step 3: Segment the image into regions                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Src: https://answers.opencv.org/question/120698/drawning-labeling-components-in-a-image-opencv-c/; https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

        // stats: size is 5(cols) x nLabels(rows). i.e. 5 is column,  stats for each label (row). Different stats type: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
        // labelImage : Stores the labelID/ region ID for each pixel. 0 means it is a background
        // CV_32S: output image label type. Currently CV_32S and CV_16U are supported.
        /// centroids:

        cv::Mat stats, centroids, labelImage;
        int nLabels = connectedComponentsWithStats(cleaned_image, labelImage, stats, centroids, 8, CV_32S);
        std::cout << "Number of labels: " << nLabels << std::endl;
        std::cout << "Stats image size: " << stats.size() << std::endl;
        std::cout << "Centroids size is: " << centroids.size() << std::endl;
        mask_1c = cv::Mat::zeros(labelImage.size(), CV_8UC1); //Creates a mat of single channel with each pixel containing zero value
        cv::Mat surfSup = stats.col(4) > 600;// & stats.col(4) < 2000; // It will store 255 in surfSup if the condition is met for that label, otherwise zero
        
            // Que: How to use binary "and" in opencv mat?
            // Ans: https://stackoverflow.com/questions/17961092/how-do-i-do-boolean-operation-on-mat-such-as-mat3-mat1-mat2

        int num_filtered_regions = 0;
        // Ignore the background pixel. That's why, starting the counter from 1 instead of 0.
        for (int i = 1; i < nLabels; i++){
            if (surfSup.at<uchar>(i, 0)){
                //All the pixels with label id i will be set to 255/ 1 with this operation
                // Src: Observation and https://stackoverflow.com/questions/26776045/compare-opencv-mat-with-scalar-elementwise
                mask_1c = mask_1c | (labelImage==i); // (labelImage==i) will return a mat with the pixel having label 'i' as 255 and rest zero. After ORing with mask and doing it repeatedly for all the labels, mask will have 255 for all the clusters/ regions and 0 elsewhere.
                num_filtered_regions++;
            }
        }
        std::cout << "Filtered regions are: " << num_filtered_regions << std::endl;
        
        // if (num_filtered_regions == 1){
        // Src: https://answers.opencv.org/question/4183/what-is-the-best-way-to-find-bounding-box-for-binary-mask/
        cv::Mat Points;
        cv::findNonZero(mask_1c, Points);
        cv::Rect Min_Rect=boundingRect(Points);
        
        mask_1c_to_3c(mask_1c, mask_3c);
        // mask_3c contains the required segmented foreground pixels
        // cv::Moments m = cv::moments(mask_1c, true);
        
        // float alpha = 0.5*atan(2*m.m11/double(m.m20 - m.m02));
        // float alpha = 0.5*atan2(2*m.m11, m.m20 - m.m02);
        // float beta = alpha + M_PI/2;
        
        // Idea: Try to use findContours then pass the findContours result into boundingRect
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        // cv::Mat contourOutput = frame.clone();
        cv::findContours( mask_1c, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE ); //Input should be single channel binary image
        std::cout << "Number of countours: " << hierarchy.size() << std::endl;
        //Draw the contours: Src: https://stackoverflow.com/questions/8449378/finding-contours-in-opencv, https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html, https://learnopencv.com/contour-detection-using-opencv-python-c/
        cv::Mat contourImage(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
        cv::Scalar colors[3];
        colors[0] = cv::Scalar(255, 0, 0);
        colors[1] = cv::Scalar(0, 255, 0);
        colors[2] = cv::Scalar(0, 0, 255);
        for (size_t idx = 0; idx < contours.size(); idx++) {
            cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
        }
        cv::imshow("Contours", contourImage);
        cv::moveWindow("Contours", 200, 0);
        
        // Find the oriented bounding box from the outermost countor | Src: https://docs.opencv.org/3.4/df/dee/samples_2cpp_2minarea_8cpp-example.html
        cv::Point2f vtx[4];
        std::vector<cv::Moments> mu(contours.size() );
        std::vector<cv::Point2f> all_vtx_vec, valid_vtx_vec, all_centroid_vec, valid_centroid_vec;
        std::vector<cv::Point2f> mc( contours.size() );
        find_obb_vertices_and_centroid(labelImage, contours, all_vtx_vec, all_centroid_vec);
        cv::Mat frame_copy;
        frame.copyTo(frame_copy);
        plot_obb_on_image(frame_copy,all_vtx_vec, all_centroid_vec);
        

        // Ignore the bounding box for contors which are close to the edges
        
        // for (int i=0; i < contours.size(); i++){

        //     /// Get the moments
        //     mu[i] = moments( contours[i], true );

        //     //  Get the mass centers:
        //     mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        //     std::cout << "Centroid x: " << mc[i].x << " | Centroid y: " << mc[i].y << std::endl;

        //     // Get the oriented bounding box for the ith contour
        //     cv::RotatedRect box = cv::minAreaRect(contours[i]);
        //     box.points(vtx);
        //     int is_rect_valid = 0;

        //     // Check if centroid is valid or not
        //     if(mc[i].x < 15 || mc[i].x > frame.cols - 15 || mc[i].y < 15 || mc[i].y > frame.rows - 15){
        //         is_rect_valid = 0;
        //     }else {
        //         is_rect_valid = 1;
        //         for(int j=0; j<4 ; j++){
        //             if (vtx[j].x < 15 || vtx[j].x > frame.cols - 15 || vtx[j].y < 15 || vtx[j].y > frame.rows - 15){
        //                 is_rect_valid = 0;
        //                 break;
        //             }
        //         }
        //         std::cout << "Break executed" << std::endl;
        //     }

        //     if (is_rect_valid == 1){
        //         valid_vtx_vec.push_back(vtx[0]);
        //         valid_vtx_vec.push_back(vtx[1]);
        //         valid_vtx_vec.push_back(vtx[2]);
        //         valid_vtx_vec.push_back(vtx[3]);
        //         std::cout << "Pushing Valid Centroid x: " << mc[i].x << " | Pushing Valid Centroid y: " << mc[i].y << std::endl;
        //         valid_centroid_vec.push_back(mc[i]);
        //     }
        // }
        // std::cout << "Valid bounding boxes: " << valid_centroid_vec.size() << std::endl;
        
        
        
        
            // Axis Aligned Bounding Box (AABB) | NOT USED
            // cv::rectangle(frame_copy,Min_Rect.tl(),Min_Rect.br(),cv::Scalar(0,255,0),2); 
        
        // int dist, dx, dy, short_index = 0, short_dist, long_dist;
        // // Draw the oriented bounding box
        // for(int k=0; k<valid_vtx_vec.size()/4; k++){
        //     vtx[0] = valid_vtx_vec[4*k];
        //     vtx[1] = valid_vtx_vec[4*k + 1];
        //     vtx[2] = valid_vtx_vec[4*k + 2];
        //     vtx[3] = valid_vtx_vec[4*k + 3];
        //     cv::Point2f centroid_pt = cv::Point2f(valid_centroid_vec[k]);
        //     std::cout << "Valid Centroid x: " << centroid_pt.x << " | Valid Centroid y: " << centroid_pt.y << std::endl;
        //     for( int i = 0; i < 4; i++ ){
        //         cv::line(frame_copy, vtx[i], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                
        //         cv::circle(frame_copy,centroid_pt, 4, cv::Scalar(0,0,255), -1);
        //         // cv::line(frame_copy, [centroids.at<double>(i,0),centroids.at<double>(i,0)], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        //         dx = vtx[i].x - vtx[(i+1)%4].x;
        //         dy = vtx[i].y - vtx[(i+1)%4].y;
        //         dist = sqrt(dx*dx + dy*dy);
        //         if (i == 0){
        //             short_dist = dist;
        //         }
        //         if (dist < short_dist){
        //             short_dist = dist;
        //             short_index = i;
        //         }else{
        //             long_dist = dist;
        //         }
        //         std::cout << "Distance between " << i << " and " << i+1 << " is: " << dist << std::endl;
        // }
        // }
        
        // std::cout << "Short dist and its index" << short_dist << " and " << short_index << " is: " << " long dist" << long_dist<< std::endl;
        
        // Finding points on the line of least central moment. i.e. line with slope alpha. i.e. midpoint of points at short_index and short_index + 1
        cv::Point2f llcm[2];
        // llcm[0].x = (vtx[short_index].x + vtx[short_index+1].y)/2;
        // llcm[0].y = (vtx[short_index].y + vtx[short_index+1].y)/2;
        // llcm[1].x = (vtx[short_index+2].x + vtx[short_index+3].y)/2;
        // llcm[1].y = (vtx[short_index+2].y + vtx[short_index+3].y)/2;
        
        // llcm[0].x = centroids.at<double>(0,0) + 0.5*long_dist*cos(alpha);
        // llcm[0].y = centroids.at<double>(0,1) + 0.5*long_dist*sin(alpha);       
        // llcm[1].x = centroids.at<double>(1,0) - 0.5*long_dist*cos(alpha);
        // llcm[1].y = centroids.at<double>(1,1) - 0.5*long_dist*sin(alpha);
        // cv::line(frame_copy, llcm[0], llcm[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::imshow("Result",frame_copy);
        
        // mu_22_alpha can be used as a feature vector
        

            // sleep(10);
            cv::Mat color_segmented(frame.size(), CV_8UC3, cv::Scalar(0));
            frame.copyTo(color_segmented,mask_3c);
            cv::namedWindow("6: Segmentation");
            cv::imshow("6: Segmentation", color_segmented);
        // }
        


        
        

    }
    return 0;
}
