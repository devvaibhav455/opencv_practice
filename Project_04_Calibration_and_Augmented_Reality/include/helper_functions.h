/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

// Converts char[] to std::string
std::string char_to_String(char* a);

// Saves the images used for camera calibration to disk
int capture_calibration_images();

int calc_calib_params(char dirname[]);
