/*Dev Vaibhav
Spring 2023 CS 5330
Project 3: Real-time Object 2-D Recognition 
*/

extern cv::Size patternsize; //interior number of corners
// Resolution required for the window
extern int res_width; //columns
extern int res_height; //rows

class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    {}
    void write(cv::FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const cv::FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (std::string)node["id"];
    }
public:   // Data Members
    int A;
    double X;
    std::string id;
};
//These write and read functions must be defined for the serialization in FileStorage to work
static void write(cv::FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const cv::FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}
// This function will print our custom class to the console
static std::ostream& operator<<(std::ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

// Converts char[] to std::string
std::string char_to_String(char* a);

// Capture/ save calibration images from camera when user presses 's' if not already saved in "calibration_images" directory
int capture_calibration_images();

// Calculate the calibration parameters from the images in the directory "calibration_images" which is provided as an argument to the function
int calc_calib_params(char dirname[]);

// Draw either the 3d axes or washington monument like structure on particular predefined checkerboard corners
// Input:               std::string shape                       --> The shape which we want to be projected. Either "axis" or "wa_monument"
//                      cv::Mat &cameraMatrix                   --> Camera calibration matrix
//                      cv::Vec<float, 5> &distCoeffs           --> Distortion co-efficients of the camera
//                      cv::Mat &rvec                           --> Rotation vector of the current frame
//                      cv::Mat &tvec                           --> Translation vector of the current frame
//                      cv::Mat &tvec                           --> Translation vector of the current frame
//                      std::vector<cv::Point2f> &corner_set    --> Vector of detected checkerboard corners (0 through 53) 
//                      int &alter_base                         --> Fill the base with a color if set to 1; otherwise 0
// Input/ Output        cv::Mat &frame                          --> Image on which shape should be drawn
int draw_shape_on_image(std::string shape, cv::Mat &frame, cv::Mat &cameraMatrix, cv::Vec<float, 5> &distCoeffs, cv::Mat &rvec, cv::Mat &tvec, std::vector<cv::Point2f> &corner_set, int &alter_base);
