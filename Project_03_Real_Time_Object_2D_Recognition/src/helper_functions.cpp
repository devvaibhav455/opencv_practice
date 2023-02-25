/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include "filter.h"



// src: input mask of 8UC1
// dst: output mask of 8UC3 with each channel a copy of the input channel 
int mask_1c_to_3c( cv::Mat &src, cv::Mat &dst ){
  std::vector<cv::Mat> channels;
  channels.push_back(src);
  channels.push_back(src);
  channels.push_back(src);
  cv::merge(channels, dst);
  // std::cout << "################### Reached inside mask function ###################" << std::endl;
  return 0;
}

// src: input mask of 8UC3
// dst: output binary image of 8UC1
int thresholding( cv::Mat &src, cv::Mat &dst ){
  
  // allocate dst image
  dst = cv::Mat::zeros(src.size(), CV_8UC1); //Because output is binary. i.e. 8-bit single channel
  cv::Mat hsv_frame_opencv, temp;
  src.copyTo(temp);
  cv::cvtColor(src, hsv_frame_opencv, cv::COLOR_BGR2HSV);
  // convert the color image to HSV

  // Loop over src rows to access the pixel
  for (int i = 0; i<hsv_frame_opencv.rows; i++){
    // row pointer (rptr) for the src
    cv::Vec3b *rptr = hsv_frame_opencv.ptr<cv::Vec3b>(i);
    // destination pointer (dptr) for the dst/ output
    cv::Vec3b *dptr = temp.ptr<cv::Vec3b>(i);
    //looping through pixels in the row
    for (int j = 0; j<hsv_frame_opencv.cols; j++){
        if (rptr[j][1] > 95){
          for (int c = 0; c < hsv_frame_opencv.channels(); c++){
            dptr[j][c] = dptr[j][c]*0.1;
        }
      }
    }
  }

  // Threshold the changed BGR image to binary by converting into greyscale
  // cv::cvtColor(temp, dst, cv::COLOR_BGR2GRAY);
  greyscale(temp, dst);
  for (int i = 0; i < dst.rows; i++){
    // row pointer (rptr) for the src
    unsigned char *rptr = dst.ptr<uchar>(i);
    for (int j = 0; j<dst.cols; j++){
        if (rptr[j] > 110){
          rptr[j] = 0;
        }else{
          rptr[j] = 255;
        }
      }
    }
  

  // std::cout << "################### Reached inside thresholding function ###################" << std::endl;
  return 0;
}


// src: input mask of 8UC3
// dst: output binary image of 8UC1
int find_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vel){
  cv::Moments m_sc; //Spatial and central moment  | Src: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html#ab8972f76cccd51af351cbda199fb4a0d
  float alpha;
  cv::Point2f mc;
  cv::Point2f vtx[4];
  for (int i=0; i < contours.size(); i++){
    // Get the moments
    m_sc = moments( contours[i], true);
    // Calculate Hu Moments 
    // alpha = 0.5*atan(2*m_sc.mu11/double(m_sc.mu20 - m_sc.mu02));
    alpha = 0.5*atan2(2*m_sc.mu11,double(m_sc.mu20 - m_sc.mu02));

    //  Get the mass centers:
    mc = cv::Point2f( m_sc.m10/m_sc.m00 , m_sc.m01/m_sc.m00 );
    // std::cout << "Centroid x: " << mc[i].x << " | Centroid y: " << mc[i].y << std::en 
    
    // Get the oriented bounding box for the ith contour
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    box.points(vtx);
    vtx_vec.push_back(vtx[0]);
    vtx_vec.push_back(vtx[1]);
    vtx_vec.push_back(vtx[2]);
    vtx_vec.push_back(vtx[3]);
    centroid_vec.push_back(mc);
    alpha_vel.push_back(alpha);
    }
  return 0;    
}


int plot_obb_on_image(cv::Mat &src, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec){
  int dist, dx, dy, short_index = 0, short_dist, long_dist;
  // Draw the oriented bounding box
  cv::Point2f vtx[4];
  for(int k=0; k<vtx_vec.size()/4; k++){
    vtx[0] = vtx_vec[4*k];
    vtx[1] = vtx_vec[4*k + 1];
    vtx[2] = vtx_vec[4*k + 2];
    vtx[3] = vtx_vec[4*k + 3];
    cv::Point2f centroid_pt = cv::Point2f(centroid_vec[k]);

    
    int fontFace = 0;
    double fontScale = 1;
    int thickness = 3;
    int baseline=0;
                              
    cv::circle(src,centroid_pt, 4, cv::Scalar(0,0,255), -1);
    cv::Size textSize = cv::getTextSize(std::to_string(k), fontFace,fontScale, thickness, &baseline);
    baseline += thickness;
    cv::Point textOrg = centroid_pt;
    putText(src, std::to_string(k), textOrg, fontFace, fontScale,cv::Scalar(255,0,0), thickness, 8);

    for( int i = 0; i < 4; i++ ){
      cv::line(src, vtx[i], vtx[(i+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      dx = vtx[i].x - vtx[(i+1)%4].x;
      dy = vtx[i].y - vtx[(i+1)%4].y;
      dist = sqrt(dx*dx + dy*dy);
      if (i == 0){
          short_dist = dist;
      }
      if (dist < short_dist){

          short_dist = dist;
          short_index = i;
      }else{
          long_dist = dist;
      }
      // std::cout << "Distance between " << i << " and " << i+1 << " is: " << dist << std::endl; 
    }
    // Plotting Axis of Least Central Moment (alcm) by finding its end and start points
    cv::Point2f alcm[2];
    alcm[0].x = centroid_pt.x + long_dist*cos(alpha_vec[k]);
    alcm[0].y = centroid_pt.y + long_dist*sin(alpha_vec[k]);       
    alcm[1].x = centroid_pt.x - long_dist*cos(alpha_vec[k]);
    alcm[1].y = centroid_pt.y - long_dist*sin(alpha_vec[k]);
    cv::line(src, alcm[0], alcm[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  }
  return 0;

}



int find_valid_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec, std::vector<double> &feature_vec ,int distance_from_boundary)
{
  cv::Moments m_sc;
  cv::Point2f mc; //mass center
  cv::Point2f vtx[4]; //vertices of 4 corners of bounding box rectangle
  float alpha;

  std::cout << "Valid Centroid rangle x: (" << distance_from_boundary << " , " << src.cols - distance_from_boundary << ")"
  << " | Valid Centroid rangle y: (" << distance_from_boundary << " , " << src.rows - distance_from_boundary << ")" << std::endl;
  
  for (int i=0; i < contours.size(); i++){
    // Get the moments
    m_sc = moments( contours[i], true);

    // Calculate Hu Moments 
    double huMoments[7]; 
    std::cout << "Declared hu moments" << std::endl;
    cv::HuMoments(m_sc, huMoments);
    std::cout << "Calculated hu moments" << std::endl;

    // std::cout << "i in outer loop: " << i << std::endl;
    // Log scale hu moments 
    for(int i = 0; i < 7; i++) {
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])); 
        // std::cout << "i in inner loop: " << i << std::endl;
    }
    // std::cout << "i in outer loop: " << i << std::endl;
    
    int N = sizeof(huMoments) / sizeof(huMoments[0]);
  
    // Initialize a vector by passing the
    // pointer to the first and last element
    // of the range as arguments
    std::vector<double> huMoments_arr_to_vec(huMoments, huMoments + N); //Converting array to vector

    alpha = 0.5*atan2(2*m_sc.mu11,double(m_sc.mu20 - m_sc.mu02));
    //  Get the mass centers:
    mc = cv::Point2f( m_sc.m10/m_sc.m00 , m_sc.m01/m_sc.m00 );
    std::cout << "Current Centroid x: " << mc.x << " | Centroid y: " << mc.y << std::endl;
    
    // Get the oriented bounding box for the ith contour
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    std::cout << "Aspect ratio is: " << box.size.aspectRatio() << std::endl;
    box.points(vtx);

    int is_rect_valid = 0;

    // Check if centroid is valid or not
    if(mc.x < distance_from_boundary || mc.x > src.cols - distance_from_boundary || mc.y < distance_from_boundary || mc.y > src.rows - distance_from_boundary){
        is_rect_valid = 0;
    }else {
        is_rect_valid = 1;
        for(int j=0; j<4 ; j++){
          // std::cout << "Checking corner x: " << vtx[j].x << " | corner y: " << vtx[j].y << std::endl;
          // If centroid is valid, check if the four corners of rectangle are valid or not
          if (vtx[j].x < distance_from_boundary || vtx[j].x > src.cols - distance_from_boundary || vtx[j].y < distance_from_boundary || vtx[j].y > src.rows - distance_from_boundary){
              is_rect_valid = 0;
              break;
          }
        }
    }

    
    if (is_rect_valid == 1){
      vtx_vec.push_back(vtx[0]);
      vtx_vec.push_back(vtx[1]);
      vtx_vec.push_back(vtx[2]);
      vtx_vec.push_back(vtx[3]);
      centroid_vec.push_back(mc);
      alpha_vec.push_back(alpha);
      for (int i = 0; i <huMoments_arr_to_vec.size(); i++){
        feature_vec.push_back(huMoments_arr_to_vec[i]);
      }
      feature_vec.push_back(cv::contourArea(contours[i])/box.size.area());
      feature_vec.push_back(box.size.aspectRatio());
      std::cout << "Pushed Valid Centroid x: " << mc.x << " | Valid Centroid y: " << mc.y << std::endl;
    }
  }
return 0;    
}

int calc_standard_deviation_vec(std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec){
  
  std::cout << "Feature vector size is: " << vector_of_feature_vectors[0].size() << std::endl;
  std::cout << "Total number of images: " << vector_of_feature_vectors.size() << std::endl;
  
    float mean, sum, std, diff;
    for (int i = 0; i < vector_of_feature_vectors[0].size(); i++ ){
      sum = 0;
      std = 0;
      for (int j = 0; j < vector_of_feature_vectors.size(); j++ ){
        sum += vector_of_feature_vectors[j][i];
      }
      mean = sum/vector_of_feature_vectors[0].size();
      
      for (int j = 0; j < vector_of_feature_vectors.size(); j++ ){
        diff = vector_of_feature_vectors[j][i] - mean;
        std += diff*diff;
      }
      std = std/vector_of_feature_vectors.size();
      standard_deviation_vec.push_back(std);
    }
    // std::cout << "INSIDE STD DEV: " << vector_of_feature_vectors[0][0] << std::endl;
  return 0;
}


int calc_feature_vector(cv::Mat &src_image, std::vector<float> &feature_vec_float){
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                  Applying thresholding to the masked greyscale image                                                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Implementing mask for thresholded HSV using. Src: https://stackoverflow.com/questions/14582082/merging-channels-in-opencv
  cv::Mat mask_3c(src_image.size(), CV_8UC3, cv::Scalar(0,0,0)), mask_1c(src_image.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat thresholded_image;
  thresholding(src_image, thresholded_image);
  cv::String window_binary_masked = "4: Binary masked";
  cv::namedWindow(window_binary_masked);
  cv::imshow(window_binary_masked, thresholded_image);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                  Step 2: Clean up the binary image                                                                           //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  cv::Mat cleaned_image(src_image.size(), CV_8UC1), temp_cleaned(src_image.size(), CV_8UC1);
  grassfire_tf(thresholded_image,cleaned_image, 8, 0, 20); // Perform erosion with 8 connectedness for at most 20 times
  cv::namedWindow("5: Cleaned Image");
  cv::imshow("5: Cleaned Image", cleaned_image);
  // grassfire_tf(temp_cleaned,cleaned_image, 8, 1, 4);

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
  

  
  // Idea: Try to use findContours then pass the findContours result into boundingRect
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  // cv::Mat contourOutput = src_image.clone();
  cv::findContours( mask_1c, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE ); //Input should be single channel binary image
  std::cout << "Number of countours: " << hierarchy.size() << std::endl;
  
  //Draw the contours: Src: https://stackoverflow.com/questions/8449378/finding-contours-in-opencv, https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html, https://learnopencv.com/contour-detection-using-opencv-python-c/
  cv::Mat contourImage(src_image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);
  for (size_t idx = 0; idx < contours.size(); idx++) {
      cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
  }
  cv::imshow("Contours", contourImage);
  cv::moveWindow("Contours", 200, 0);
  
  // Find the oriented bounding box from the outermost contors | Src: https://docs.opencv.org/3.4/df/dee/samples_2cpp_2minarea_8cpp-example.html
  cv::Point2f vtx[4];
  std::vector<cv::Moments> all_mu(contours.size());
  std::vector<cv::Point2f> all_vtx_vec, valid_vtx_vec, all_centroid_vec, valid_centroid_vec;
  std::vector<cv::Point2f> mc(contours.size() );
  std::vector<float> all_alpha_vec, valid_alpha_vec;
  
  find_obb_vertices_and_centroid(labelImage, contours, all_vtx_vec, all_centroid_vec, all_alpha_vec);
  std::cout << "Num obb are: " << all_alpha_vec.size() << std::endl;
  cv::Mat src_image_copy_all_boxes;
  src_image.copyTo(src_image_copy_all_boxes);
  plot_obb_on_image(src_image_copy_all_boxes,all_vtx_vec, all_centroid_vec, all_alpha_vec);
  cv::String windowName_all_obb = "All OBB"; //Name of the window
  cv::namedWindow(windowName_all_obb); // Create a window
  cv::imshow(windowName_all_obb,src_image_copy_all_boxes);
  
  cv::Mat src_image_copy_valid_boxes;
  src_image.copyTo(src_image_copy_valid_boxes);

  std::vector<double> feature_vec;
  find_valid_obb_vertices_and_centroid(labelImage, contours, valid_vtx_vec, valid_centroid_vec, valid_alpha_vec, feature_vec, 15);
  std::cout << "Valid obb are: " << valid_alpha_vec.size() << std::endl;

  plot_obb_on_image(src_image_copy_valid_boxes,valid_vtx_vec, valid_centroid_vec, valid_alpha_vec);
  cv::String windowName_valid_obb = "Valid OBB"; //Name of the window
  cv::namedWindow(windowName_valid_obb); // Create a window
  cv::imshow(windowName_valid_obb,src_image_copy_valid_boxes);

  for(auto it:feature_vec){
    std::cout << it << std::endl;
  }

  // Axis Aligned Bounding Box (AABB) | NOT USED
  // cv::rectangle(frame_copy,Min_Rect.tl(),Min_Rect.br(),cv::Scalar(0,255,0),2); 

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                  Step 4: Compute features for each major region                                                              //
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  if (valid_alpha_vec.size() != 1){
    std::cout << "Either could not find any OBB or found multiple OBB" << std::endl;
    exit(-1);
  }
  std::vector<float> feature_vec_new(feature_vec.begin(), feature_vec.end());
  feature_vec_float = feature_vec_new;
  return 0;
}

int one_object_classifier(std::vector<float> &target_image_feature_vector, std::vector<const char *> &image_names_in_db, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::string &target_image_label){
  
  std::vector<std::pair<float, int>> distance_metric_vector; //Less distance means that images are more similar
  float distance;
  for (int i = 0; i < vector_of_feature_vectors.size(); i++ ){
    distance = 0;
    for (int j = 0; j < vector_of_feature_vectors[0].size(); j++ ){
      
      // std::cout << "Accessing the vectors" << std::endl;
      // std::cout << target_image_feature_vector[0] << " | " << vector_of_feature_vectors[0][0] << " | " << standard_deviation_vec[0] << std::endl;

      distance += (target_image_feature_vector[j] - vector_of_feature_vectors[i][j])/standard_deviation_vec[j];
      distance = distance*distance;
    }
    distance_metric_vector.push_back(std::make_pair(distance, i));
  }  
  
  //Find the image corresponding to minimum distance

  std::sort(distance_metric_vector.begin(), distance_metric_vector.end());
  
  // Printing the distance_metric_vector for confirmation
  for (int i = 0; i <distance_metric_vector.size(); i++){
      std::cout << distance_metric_vector[i].first << " , " << distance_metric_vector[i].second << std::endl;
  }  
  std::string closest_match(image_names_in_db[distance_metric_vector[0].second]);
  target_image_label = closest_match;

  // strcpy(target_image_label,image_names_in_db[distance_metric_vector[0].second]);
  // std::cout << "Closest match" << target_image_label  << std::endl;



  return 0;

}


