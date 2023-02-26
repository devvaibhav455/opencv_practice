/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

// Function to convert 1C mask to 3C by copying the single channel to all the channels

int basic_training_mode_from_directory();

int object_recognition_mode(char* target_filename);

int knn_classifier(int &k, char * target_filename_char_star);

int testing_mode(int &source, cv::Mat &frame);