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

// Saves the images used for camera calibration to disk
int capture_calibration_images();

int calc_calib_params(char dirname[]);
