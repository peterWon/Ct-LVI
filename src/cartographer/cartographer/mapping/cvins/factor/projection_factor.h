#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "utility.h"
#include "parameters.h"

// 对于每一个空间共视点P会有一个投影误差方程，将第一次观测到P点的i帧中的归一化球面坐标根据逆深度估计值反向投影到空间中，该点再投影到j帧中的归一化球面坐标，该理想投影坐标与j帧所观测到的像素坐标之差即是对应该点的残差
//残差为2维，i帧的位姿p_i+q_i为7维；j帧的位姿p_j+q_j为7维；相机与IMU之间的相对位姿为7维；该点的逆深度为1维
namespace cvins{
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
}