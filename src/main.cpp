#include <iostream>
#include "cam_calib.hpp"

int main(int argc, char **argv)
{
    std::string pic_path = "../RGB_camera_calib_img/";
    cv::Size pattern_size = cv::Size(11, 8); //points_per_row（width）、points_per_colum（height）
    CamCalib CAM0(pic_path, pattern_size, 0.02f, 0.02f);
    CAM0.CalibrateByZZY();
    CAM0.CalibrateByOpencv();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
