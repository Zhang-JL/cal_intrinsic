#include "cam_calib.hpp"


bool CamCalib::ReadPicFromPath()
{
    mvMatInitPics.clear();
    mvMatPostPics.clear();
    for (int i = 0; i <= 40; ++i)
    {
        std::string single_picture_path = msPicPath + "/" + std::to_string(i + 100000) + ".png";
        cv::Mat pic = cv::imread(single_picture_path, cv::IMREAD_GRAYSCALE);
        if (pic.empty())
        {
            std::cerr << "read picture failed: " << single_picture_path;
            return false;
        }
        mvMatInitPics.push_back(pic);
        mvMatPostPics.push_back(pic);
        #if DEBUG_MODE == TRUE
        if (i == 1)
        {
            cv::imshow("ori img", pic);
            cv::waitKey(100);
        }
        #endif
    }
    return true;
}

bool CamCalib::FindCorners()
{
    //通过棋盘格的已知信息，求角点在世界坐标系下的坐标XYZ，这里的Z=0。角点按行，从左到右，上到下编号。
    mvCornersInCheese_XYZ.clear();
    cv::Point2f cheese_corner_XYZ;
    for (int row = 0; row < msPatternSize.height; ++row)
    {
        cheese_corner_XYZ.y = (float)row * mfCheeseSquarHgt;
        for (int col = 0; col < msPatternSize.width; ++col)
        {
            cheese_corner_XYZ.x = (float)col * mfCheeseSquarLen;
            mvCornersInCheese_XYZ.push_back(cheese_corner_XYZ);
        }
    }

    //通过opencv的内置函数，寻找棋盘格的角点
    mvvCornersEachImg_uv.clear();
    uint16_t for_cnt = 0;
    for (const auto & mvInitPic : mvMatInitPics)
    {
        for_cnt++;
        cv::Mat temp_mat;
        temp_mat = mvInitPic;
        //ref: https://cloud.tencent.com/developer/article/2080799
        //初步角点检测
        std::vector<cv::Point2f> corners;
        bool found_flag = cv::findChessboardCorners(temp_mat, msPatternSize, corners,
                                  cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!found_flag)
        {
            std::cerr<< "not found corner!";
            return false;
        }
        //角点进一步进行亚像素检测。winize定义了一个 矩形窗口，在该窗口内计算每个角点的精细位置，窗口越大精度越高。
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                                     30, 0.001);
        cv::cornerSubPix(temp_mat, corners, cv::Size(11,11),
                         cv::Size(-1, -1), criteria);

        mvvCornersEachImg_uv.push_back(std::move(corners)); //move: 通过将对象的右值（临时或者即将销毁的对象）引用转发给其他函数。
        #if DEBUG_MODE == TRUE
        //棋盘格角点绘制
        cv::drawChessboardCorners(temp_mat, msPatternSize, corners, found_flag);
        std::string img_name = "corner img" + std::to_string(for_cnt);
        cv::imshow(img_name, temp_mat);
        cv::waitKey(0);
        #endif
    }
    return true;
}

void CamCalib::NormalizePoints(const std::vector<cv::Point2f> *points_vec,
                               std::vector<cv::Point2f> *norm_points_vec,
                               cv::Mat *T_p2pnorm)
{
    double mean_x = 0.0l, mean_y = 0.0l;
    for (const auto &p: *points_vec)
    {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= (double)points_vec->size();
    mean_y /= (double)points_vec->size();
    double mean_dev_x = 0.0l, mean_dev_y = 0.0l;
    for (const auto &p: *points_vec)
    {
        mean_dev_x += fabs(p.x - mean_x);
        mean_dev_y += fabs(p.y - mean_y);
    }
    mean_dev_x /= (double)points_vec->size();
    mean_dev_y /= (double)points_vec->size();
    double s_x = 1.0 / mean_dev_x;
    double s_y = 1.0 / mean_dev_y;
    norm_points_vec->clear();
    cv::Point2f norm_p_uv;
    for (auto &p: *points_vec)
    {
        norm_p_uv.x = (float)(s_x * p.x - mean_x * s_x);
        norm_p_uv.y = (float)(s_y * p.y - mean_y * s_y);
        norm_points_vec->push_back(norm_p_uv);
    }
    T_p2pnorm->at<double>(0, 0) = s_x;
    T_p2pnorm->at<double>(0, 1) = - mean_x * s_x;
    T_p2pnorm->at<double>(1, 0) = s_y;
    T_p2pnorm->at<double>(1, 1) = - mean_y * s_y;
}

//已知像素坐标 u,v 和世界坐标 X,Y,Z 求矩阵 H ：
//p_uv = 1/Z * H * P_XYZ
//p_uv_norm = T1 * p_uv
//P_XYZ_norm = T2 * P_XYZ
// → T1_inv * p_uv_norm = 1/Z * H * T2_inv * P_XYZ_norm
// → p_uv_norm =  1/Z * H_norm * P_XYZ_norm
// → T1 * H * T2_inv = H_norm
bool CamCalib::CalMatH()
{
    //默认全部角点都能拍到
    std::vector<cv::Point2f> vec_norm_P_XYZ;
    cv::Mat T_P2Pnorm;
    NormalizePoints(&mvCornersInCheese_XYZ, &vec_norm_P_XYZ, &T_P2Pnorm);
    cv::Mat H_norm = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
    cv::Mat H = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
    // A * h = 0，填充 A 阵：
    for (const auto &vec_points_uv : mvvCornersEachImg_uv)
    {
        std::vector<cv::Point2f> vec_norm_p_uv;
        cv::Mat T_p2pnorm, T_p2pnorm_inv;
        NormalizePoints(&vec_points_uv, &vec_norm_p_uv, &T_p2pnorm);
        cv::invert(T_P2Pnorm, T_p2pnorm_inv);
        if (vec_points_uv.size() < 4)
        {
            std::cerr << "corners num < 4, can't solve matrix H";
            return false;
        }
        cv::Mat mat_A((int)vec_points_uv.size() * 2, 9, CV_64F, cv::Scalar(0));
        //遍历每张图的角点
        for (int i = 0; i < vec_points_uv.size(); ++i)
        {
            float X = vec_norm_P_XYZ.at(i).x;
            float Y = vec_norm_P_XYZ.at(i).y;
            float u = vec_points_uv.at(i).x;
            float v = vec_points_uv.at(i).y;
            mat_A.at<double>(2 * i, 0) = X;
            mat_A.at<double>(2 * i, 1) = Y;
            mat_A.at<double>(2 * i, 2) = 1;
            mat_A.at<double>(2 * i, 6) = -u * X;
            mat_A.at<double>(2 * i, 7) = -u * Y;
            mat_A.at<double>(2 * i, 8) = -u;

            mat_A.at<double>(2 * i + 1, 3) = X;
            mat_A.at<double>(2 * i + 1, 4) = Y;
            mat_A.at<double>(2 * i + 1, 5) = 1;
            mat_A.at<double>(2 * i + 1, 6) = -v * X;
            mat_A.at<double>(2 * i + 1, 7) = -v * Y;
            mat_A.at<double>(2 * i + 1, 8) = -v;
        }
        //最小二乘法求 Ah=0 ，使用 SVD 分解法求解。
        //找到 (A'*A) 最小特征值对应的 V 中的特征向量即为最小二乘解
        cv::Mat U, W, VT;   // A =UWV^T
        // Eigen 返回的是V,列向量就是特征向量；opencv 返回的是VT，所以行向量是特征向量
        cv::SVD::compute(mat_A, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        H_norm = VT.row(8).reshape(0, 3);
        H = T_p2pnorm_inv * H_norm * T_P2Pnorm;
        mvMatH.push_back(H);
    }
    return true;
}

bool CamCalib::CalVecVij(cv::Mat *vij, int i, int j, const cv::Mat *H)
{
    i -= 1;
    j -= 1;
    if (i < 0 || j < 0)
    {
        return false;
    }
    vij->at<double>(0,0) = H->at<double>(0,i) * H->at<double>(0,j);
    vij->at<double>(0,1) = H->at<double>(0,i) * H->at<double>(1,j) + H->at<double>(1,i) * H->at<double>(0,j),
    vij->at<double>(0,2) = H->at<double>(1,i) * H->at<double>(1,j),
    vij->at<double>(0,3) = H->at<double>(0,i) * H->at<double>(2,j) + H->at<double>(2,i) * H->at<double>(0,j),
    vij->at<double>(0,4) = H->at<double>(1,i) * H->at<double>(2,j) + H->at<double>(2,i) * H->at<double>(1,j),
    vij->at<double>(0,5) = H->at<double>(2,i) * H->at<double>(2,j);
}

//H_i^T * B * H_j = v_ij * b
//求解 vb = 0 ，得到 b 向量后可恢复内参矩阵 A 。
bool CamCalib::CalIntrinsicA()
{
    bool cal_flag = true;
    cv::Mat MatV(2 * (int)mvMatH.size(), 6, CV_64F, cv::Scalar (0));
    cv::Mat v_11(1, 6, CV_64F, cv::Scalar (0));
    cv::Mat v_12(1, 6, CV_64F, cv::Scalar (0));
    cv::Mat v_22(1, 6, CV_64F, cv::Scalar (0));
    cv::Mat v_temp(1, 6, CV_64F, cv::Scalar (0));
    for (int i = 0; i < mvMatH.size(); ++i)
    {
        cal_flag &= CalVecVij(&v_11, 1, 1, &mvMatH.at(i));
        cal_flag &= CalVecVij(&v_12, 1, 2, &mvMatH.at(i));
        cal_flag &= CalVecVij(&v_22, 2, 2, &mvMatH.at(i));
        v_temp = v_11 - v_22;
        v_12.copyTo(MatV.row(2 * i));
        v_temp.copyTo(MatV.row(2 * i + 1));
    }
    cv::Mat U, W, VT;   // A =UWV^T
    // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
    cv::SVD::compute(MatV, W, U, VT);
    cv::Mat B = VT.row(5);
    double B11 = B.at<double>(0, 0);
    double B12 = B.at<double>(0, 1);
    double B22 = B.at<double>(0, 2);
    double B13 = B.at<double>(0, 3);
    double B23 = B.at<double>(0, 4);
    double B33 = B.at<double>(0, 5);

    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = sqrt(lambda / B11);
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    gamma = 0;

    mMatIntrA.setTo(cv::Scalar(0));
    mMatIntrA.at<double>(0, 0) = alpha;
    mMatIntrA.at<double>(0, 1) = gamma;
    mMatIntrA.at<double>(0, 2) = u0;
    mMatIntrA.at<double>(1, 1) = beta;
    mMatIntrA.at<double>(1, 2) = v0;

    return cal_flag;
}

// [r1 r2 t] = inv(A) * H
bool CamCalib::CalExtrinsicT()
{
    cv::Mat mat_temp, matA_inv;
    cv::invert(mMatIntrA, matA_inv);
    for (const auto &matH : mvMatH)
    {
        mat_temp = matA_inv * matH;
        cv::Vec3d r1(mat_temp.at<double>(0, 0), mat_temp.at<double>(1, 0), mat_temp.at<double>(2, 0));
        cv::Vec3d r2(mat_temp.at<double>(0, 1), mat_temp.at<double>(1, 1), mat_temp.at<double>(2, 1));
        cv::Vec3d r3 = r1.cross(r2);
        cv::Mat Q = cv::Mat::zeros(3, 3, CV_64F);
        Q.at<double>(0, 0) = r1(0);
        Q.at<double>(1, 0) = r1(1);
        Q.at<double>(2, 0) = r1(2);
        Q.at<double>(0, 1) = r2(0);
        Q.at<double>(1, 1) = r2(1);
        Q.at<double>(2, 1) = r2(2);
        Q.at<double>(0, 2) = r3(0);
        Q.at<double>(1, 2) = r3(1);
        Q.at<double>(2, 2) = r3(2);
        cv::Mat norm_Q;
        cv::normalize(Q, norm_Q);
        cv::Mat U, W, VT;   // A =UWV^T
        // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        // FULL_UV 表示计算完整的 U 和 V 矩阵。
        // MODIFY_A 表示输入矩阵 A 可以被修改。
        cv::SVD::compute(norm_Q, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat R = U * VT;
        mvMatExtr_R.push_back(R);

        cv::Mat R_T;
        cv::transpose(R, R_T);
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
        mat_temp.col(2).copyTo(t.col(0));
        mvMatExtr_t.push_back(t);
    }
    return true;
}

// use one pic to calc
// u_hat - u = (u - u0) * r^2 * k1 + (u - u0) * r^4 * k2
// v_hat - v = (v - v0) * r^2 * k1 + (v - v0) * r^4 * k2
// r^2 = u^2 + v^2
#define PIC_INDEX_CAL_K 0
void CamCalib::CalDistortK()
{
    std::vector<double> vec_r2;
    std::vector<cv::Point2f> vec_ideal_pts;
    double u0 = mMatIntrA.at<double>(0, 2);
    double v0 = mMatIntrA.at<double>(1, 2);
    cv::Mat mat_R = mvMatExtr_R.at(PIC_INDEX_CAL_K);
    cv::Mat mat_t = mvMatExtr_R.at(PIC_INDEX_CAL_K);
    const std::vector<cv::Point2f> *vec_dist_pts = &mvvCornersEachImg_uv[PIC_INDEX_CAL_K];

    for (const auto &p : mvCornersInCheese_XYZ)
    {
        cv::Mat p_3d = (cv::Mat_<double>(3, 1) << p.x, p.y, 0);
        cv::Mat p_pic = mat_R * p_3d + mat_t;
        p_pic.at<double>(0, 0) = p_pic.at<double>(0, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(1, 0) = p_pic.at<double>(1, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(2, 0) = 1;
        double x = p_pic.at<double>(0, 0);
        double y = p_pic.at<double>(1, 0);
        double r2 = x * x + y * y;
        vec_r2.push_back(r2);

        cv::Mat p_uv = mMatIntrA * p_pic;
        vec_ideal_pts.emplace_back(p_uv.at<double>(0, 0), p_uv.at<double>(1, 0));
    }

    cv::Mat D = cv::Mat::zeros(vec_dist_pts->size() * 2, 2, CV_64F);
    cv::Mat d = cv::Mat::zeros(vec_dist_pts->size() * 2, 1, CV_64F);
    for (int i = 0; i < vec_dist_pts->size(); ++i)
    {
        double r2 = vec_r2[i];
        D.at<double>(2 * i, 0) = (vec_ideal_pts[i].x - u0) * r2;
        D.at<double>(2 * i, 1) = (vec_ideal_pts[i].x - u0) * r2 * r2;
        D.at<double>(2 * i + 1, 0) = (vec_ideal_pts[i].y - v0) * r2;
        D.at<double>(2 * i + 1, 1) = (vec_ideal_pts[i].y - v0) * r2 * r2;
        d.at<double>(0, 0) = vec_dist_pts->at(i).x - vec_ideal_pts[i].x;
        d.at<double>(1, 0) = vec_dist_pts->at(i).y - vec_ideal_pts[i].y;
    }
    cv::Mat DT;
    cv::transpose(D, DT);
    cv::Mat DTD_inverse;
    cv::invert(DT * D, DTD_inverse);
    mMatDistK = DTD_inverse * DT * d;
}

bool CamCalib::CalibrateByZZY()
{
    if (ReadPicFromPath() && FindCorners())
    {
        CalMatH();
        CalIntrinsicA();
        CalExtrinsicT();
        CalDistortK();
    }
    return true;
}