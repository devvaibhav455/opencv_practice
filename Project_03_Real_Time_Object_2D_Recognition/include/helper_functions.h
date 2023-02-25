/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

// Function to convert 1C mask to 3C by copying the single channel to all the channels
int mask_1c_to_3c( cv::Mat &src, cv::Mat &dst );


// Function to convert 1C mask to 3C by copying the single channel to all the channels
int thresholding(cv::Mat &src, cv::Mat &dst );

int find_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vel);

int plot_obb_on_image(cv::Mat &src, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec);


int find_valid_obb_vertices_and_centroid(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &vtx_vec, std::vector<cv::Point2f> &centroid_vec, std::vector<float> &alpha_vec, std::vector<double> &feature_vec ,int distance_from_boundary);

int calc_standard_deviation_vec(std::vector<std::vector<float>> &feature_vectors, std::vector<float> &standard_deviation_vec);

int calc_feature_vector(cv::Mat &src_image, std::vector<float> &feature_vec_float);

int one_object_classifier(std::vector<float> &target_image_feature_vector, std::vector<const char *> &image_names_in_db, std::vector<std::vector<float>> &vector_of_feature_vectors, std::vector<float> &standard_deviation_vec, std::string &target_image_label);



