/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

// Basic training mode. Reads images from directory and writes the feature vectors to a csv file
int basic_training_mode_from_directory();

// Takes an image as source and finds the closest match in DB. If new image, adds it to the DB
int object_recognition_mode(int &source, cv::Mat &frame);

// KNN classifier mode. Identifies closest match to a target match from DB. Does not add image to DB if new is found
int knn_classifier(int &k, char * target_filename_char_star);

// Testing mode. Takes image/ video input from a file or camera live feed and used to test the image processing pipeline
int testing_mode(int &source, cv::Mat &frame);