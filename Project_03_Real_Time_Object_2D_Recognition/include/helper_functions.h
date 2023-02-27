/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

// Converts char[] to std::string
std::string char_to_String(char* a);

// src: input mask of 8UC1
// dst: output mask of 8UC3 with each channel a copy of the input channel 
int mask_1c_to_3c( cv::Mat &src, cv::Mat &dst );


// src: input mask of 8UC3
// dst: output binary image of 8UC1
int thresholding(cv::Mat &src, cv::Mat &dst );

// Finds all the possible bounding boxes corresponding to the contours in the image
// Input:   src           --> cv::Mat, input mask of 8UC1 having 255 at foreground and zero elsewhere
//          contours      --> std::vector<std::vector<cv::Point>> , contour coordinates of all the contours in the image
// Output:  vtx_vec       --> std::vector<cv::Point2f> , vertices of all the bounding boxes (4 each)
//          centroid_vec  --> std::vector<cv::Point2f> , coordinates of the centroids of all the bounding boxes
//          alpha_vel     --> std::vector<float> , Orientation of the axis of least central moment for all the contours
int find_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vel);

// Plots the oriented bounding boxes in the image
// Input:   src           --> cv::Mat, input image on which the obb needs to be drawn. It gets modified
//          vtx_vec       --> std::vector<cv::Point2f> , vertices of all the bounding boxes (4 each)
//          centroid_vec  --> std::vector<cv::Point2f> , coordinates of the centroids of the bounding boxes
//          alpha_vel     --> std::vector<float> , Orientation of the axis of least central moment for the contours
int plot_obb_on_image(cv::Mat &src, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec);

// Finds all the valid bounding boxes corresponding to the contours in the image. Validity checked on certain conditions inside the code
// Input:   src           --> cv::Mat, input image on which the obb needs to be drawn. It gets modified
//          contours      --> std::vector<std::vector<cv::Point>> , contour coordinates of all the contours in the image
// Output:  vtx_vec       --> std::vector<cv::Point2f> , vertices of all the valid bounding boxes (4 each)
//          centroid_vec  --> std::vector<cv::Point2f> , coordinates of the valid centroids of the bounding boxes
//          alpha_vel     --> std::vector<float> , Orientation of the axis of least central moment for the valid contours/ obb
int find_valid_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec, std::vector<std::vector<double>> &vector_of_feature_vec ,int distance_from_boundary);

// Finds a standard deviation vector whose elements correspond to each element of the feature vector
// Input:   vector_of_feature_vectors    --> std::vector<std::vector<float>>, A vector containing all the feature vectors
// Output:  standard_deviation_vec       --> std::vector<float> , A vector containing standard deviations for each elemnt of the feature vector
int calc_standard_deviation_vec(std::vector<std::vector<float>> &feature_vectors, std::vector<float> &standard_deviation_vec);

// Calculates the feature vector of an image
// Input:   src_image                   --> cv::Mat, Image for which the feature vector needs to be computed
// Output:  feature_vec_float           --> std::vector<float> , Feature vector of the image
//          src_image_copy_valid_boxes  --> cv::Mat , An image which is copy of the source image but has valid obb drawn on it
int calc_feature_vector(cv::Mat &src_image, std::vector<std::vector<float>> &vec_of_feature_vec_float, cv::Mat &src_image_copy_valid_boxes, std::vector<cv::Point2f> &valid_centroid_vec);

// Classifies an object
// Input:   target_image_feature_vector --> std::vector<float>, Feature vector of the target image
//          image_names_in_db           --> std::vector<const char *> , All the images names in the DB
//          vector_of_feature_vectors   --> std::vector<std::vector<float>> , A vector containing all the feature vectors of all the images
//          standard_deviation_vec      --> std::vector<float> , A vector containing standard deviations for each elemnt of the feature vector
//          target_image_filename       --> char * , Image name of the target file. Used if it is a new image and add it to the DB
// Output:  target_image_label          --> std::string , Label/ class of the target image
int one_object_classifier(cv::Mat &frame, std::vector<float> &target_image_feature_vector, std::vector<const char *> &image_names_in_db, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::string &target_image_label);

// Finds unique entries and their indices from a vector
// Input:   image_names_in_db           --> std::vector<const char *> , All the images names in the DB 
// Output:  unique_labels               --> std::vector<const char *> , Unique labels/ classes in the DB
//          unique_labels_indices       --> std::vector<std::vector<int>>, Indices for each of the class in the DB
int find_unique_and_their_indices(std::vector<const char *> &image_names_in_db, std::vector<const char *> &unique_labels, std::vector<std::vector<int>> &unique_labels_indices);

// Calculates scaled euclidean distance of a target image with all the images in the DB
// Input:   target_image_feature_vector --> std::vector<float>, Feature vector of the target image
//          vector_of_feature_vectors   --> std::vector<std::vector<float>> , A vector containing all the feature vectors of all the images
//          standard_deviation_vec      --> std::vector<float> , A vector containing standard deviations for each elemnt of the feature vector
// Output:  distance_metric_vector      --> std::vector<std::pair<float, int>> , A vector containing distance and class/ label index pair for each of the class
int calc_scaled_euclidean_dist_vec(std::vector<float> &target_image_feature_vector, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::vector<std::pair<float, int>> &distance_metric_vector);

// Finds the KNN distance to each class/ label
// Input:   k                        --> int, K parameter for the algorithm.. 1/2/3....
//          distance_metric_vector   --> std::vector<std::pair<float, int>>,A vector containing distance and class/ label index pair for each of the class
// Output:  min_dist                 --> float, If k != 1, Distance to the class for which distance_metric_vector is provided
// If k=1, returns the index of the class corresponding to the minimum index
int find_min_k_distance(int k, std::vector<std::pair<float, int>> &distance_metric_vector, float &min_dist);

// Calculates Mnhattan L1 distance of a target image with all the images in the DB
// Input:   target_image_feature_vector --> std::vector<float>, Feature vector of the target image
//          vector_of_feature_vectors   --> std::vector<std::vector<float>> , A vector containing all the feature vectors of all the images
//          standard_deviation_vec      --> std::vector<float> , A vector containing standard deviations for each elemnt of the feature vector
// Output:  distance_metric_vector      --> std::vector<std::pair<float, int>> , A vector containing distance and class/ label index pair for each of the class
int calc_manhattan_l1_dist_vec(std::vector<float> &target_image_feature_vector, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::vector<std::pair<float, int>> &distance_metric_vector);

// Calculates Chi-Square distance of a target image with all the images in the DB
// Input:   target_image_feature_vector --> std::vector<float>, Feature vector of the target image
//          vector_of_feature_vectors   --> std::vector<std::vector<float>> , A vector containing all the feature vectors of all the images
//          standard_deviation_vec      --> std::vector<float> , A vector containing standard deviations for each elemnt of the feature vector
// Output:  distance_metric_vector      --> std::vector<std::pair<float, int>> , A vector containing distance and class/ label index pair for each of the class
int calc_chisquare_dist_vec(std::vector<float> &target_image_feature_vector, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::vector<std::pair<float, int>> &distance_metric_vector);




