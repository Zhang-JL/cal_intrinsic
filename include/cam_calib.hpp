#ifndef CAL_INSTRINSIC_CAM_CALIB_HPP
#define CAL_INSTRINSIC_CAM_CALIB_HPP
#pragma once
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define DEBUG_MODE false


void ProjectPointFromWordtoPiexel(cv::Mat const *Mat_A, cv::Mat const *Mat_R, cv::Mat const *Mat_t,
                                  cv::Mat const *Mat_k, cv::Point3f const *P_XYZ, cv::Point2f *p_uv);

class CamCalib
{
public:
    explicit CamCalib(std::string &path, cv::Size &pattern_size, float height, float length):
    msPicPath(path), msPatternSize(pattern_size), mfCheeseSquarHgt(height), mfCheeseSquarLen(length){}
    struct RepjErrCostFunctor;
    bool ReadPicFromPath();
    bool FindCorners();
    void NormalizePoints(const std::vector<cv::Point2f> *points_vec,
                         std::vector<cv::Point2f> *norm_points_vec,
                         cv::Mat *T_p2pnorm);
    bool CalVecVij(cv::Mat *vij, int i, int j, const cv::Mat *H);
    bool CalMatH();
    bool CalIntrinsicA();
    bool CalExtrinsicT();
    void CalDistortK();
    double CalRepjErr();
    void OptCaliData();
    bool CalibrateByZZY();
    bool CalibrateByOpencv();

private:
    std::string msPicPath;
    cv::Size msPatternSize;     //棋盘格内部角点的行列数
    float mfCheeseSquarHgt;
    float mfCheeseSquarLen;

    cv::Mat mMatIntrA;
    cv::Mat mMatDistK;
    std::vector<cv::Mat> mvMatInitPics;
    std::vector<cv::Mat> mvMatPostPics;
    std::vector<cv::Mat> mvMatH;
    std::vector<cv::Mat> mvMatExtr_R;
    std::vector<cv::Mat> mvMatExtr_t;


    std::vector<std::vector<cv::Point2f>>   mvvCornersEachImg_uv;
    std::vector<cv::Point2f>                mvCornersInCheese_XYZ; //Z=0，因此只记录X和Y

};

#endif //CAL_INSTRINSIC_CAM_CALIB_HPP
