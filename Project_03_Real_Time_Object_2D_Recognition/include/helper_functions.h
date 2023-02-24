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

// int calc_mu_pq_alpha(cv::Mat &src, std::vector<cv::Point> &contour, float &alpha, int p, int q);




