/*
 * CLINS: Continuous-Time Trajectory Estimation for LiDAR-Inertial System
 * Copyright (C) 2022 Jiajun Lv
 * Copyright (C) 2022 Kewei Hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#define _USE_MATH_DEFINES
#include <cmath>

#include <factor/auto_diff/imu_factor.h>
#include <factor/auto_diff/lidar_feature_factor.h>

#include <glog/logging.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/lidar_data.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <fstream>
#include <memory>
#include <utils/eigen_utils.hpp>
#include "parameters.h"

namespace clins {

class CalibParamManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<CalibParamManager> Ptr;

  CalibParamManager(const cvins::ParameterServer* ps)
      : ps_(ps),
        p_LinI(Eigen::Vector3d(0, 0, 0)),
        q_LtoI(Eigen::Quaterniond::Identity()),
        p_CinI(Eigen::Vector3d(0, 0, 0)),
        q_CtoI(Eigen::Quaterniond::Identity()),
        g_refine(Eigen::Vector2d(0, 0)),
        gyro_bias(Eigen::Vector3d(0, 0, 0)),
        acce_bias(Eigen::Vector3d(0, 0, 0)),
       time_offset_lidar(0), time_offset_camera(0) {
   
    // Lidar-IMU extrinsic Param
    p_LinI = ps->p_LinI;
    q_LtoI = ps->q_LtoI;
    q_LtoI.normalized();
    so3_LtoI = SO3d(q_LtoI);
    se3_LtoI = SE3d(so3_LtoI, p_LinI);
    
    p_CinI = ps->p_CinI;
    q_CtoI = ps->q_CtoI;
    q_CtoI.normalized();
    so3_CtoI = SO3d(q_CtoI);
    se3_CtoI = SE3d(so3_CtoI, p_CinI);

    time_offset_lidar = ps->time_offset_lidar;
    time_offset_camera = ps->time_offset_camera;

    // g_refine = ps->g_refine;
    gravity = Eigen::Vector3d(0, 0, ps_->GRAVITY_NORM),

    /// estimate weight param
    global_opt_gyro_weight = ps->global_opt_gyro_weight;
    global_opt_acce_weight = ps->global_opt_acce_weight;
    
    global_opt_lidar_weight = ps->global_opt_lidar_weight;
    
    global_opt_lidar_rot_weight = ps->global_opt_lidar_rot_weight;
    global_opt_lidar_pos_weight = ps->global_opt_lidar_pos_weight;
    
    global_opt_cam_rot_weight = ps->global_opt_cam_rot_weight;
    global_opt_cam_pos_weight = ps->global_opt_cam_pos_weight;
    
    global_opt_cam_uv_weight = ps->global_opt_cam_uv_weight;
    
    global_opt_imu_pos_weight = ps->global_opt_imu_pos_weight;
    global_opt_imu_vel_weight = ps->global_opt_imu_vel_weight;
    global_opt_imu_rot_weight = ps->global_opt_imu_rot_weight;
    global_opt_imu_ba_weight = ps->global_opt_imu_ba_weight;
    global_opt_imu_bg_weight = ps->global_opt_imu_bg_weight;

    gyro_bias_uppper_bound = ps->gyro_bias_uppper_bound;
    acce_bias_uppper_bound = ps->acce_bias_uppper_bound;
    
    enable_debug = ps->enable_debug;
  }

  void ShowIMUBias() {
    std::cout << BLUE << "Gyro Bias : " << gyro_bias.transpose() << RESET
              << std::endl;
    std::cout << BLUE << "Accel Bias : " << acce_bias.transpose() << RESET
              << std::endl;
  }

  bool CheckIMUBias() {
    if (fabs(gyro_bias(0)) > gyro_bias_uppper_bound ||
        fabs(gyro_bias(1)) > gyro_bias_uppper_bound ||
        fabs(gyro_bias(2)) > gyro_bias_uppper_bound) {
      gyro_bias = Eigen::Vector3d(0, 0, 0);
    }

    if (fabs(acce_bias(0)) > acce_bias_uppper_bound ||
        fabs(acce_bias(1)) > acce_bias_uppper_bound ||
        fabs(acce_bias(2)) > acce_bias_uppper_bound) {
      acce_bias = Eigen::Vector3d(0, 0, 0);
    }

    return true;
  }

  // Update after Ceres optimization
  void UpdateExtrinicParam() {
    q_LtoI = so3_LtoI.unit_quaternion();
    se3_LtoI = SE3d(so3_LtoI, p_LinI);
  }

  void UpdateGravity(Eigen::Vector3d gravity_in, int segment_id = 0) {
    gravity = gravity_in;

    gravity_in = (gravity_in / ps_->GRAVITY_NORM).eval();
    double cr = std::sqrt(gravity_in[0] * gravity_in[0] +
                          gravity_in[2] * gravity_in[2]);
    g_refine[0] = std::acos(cr);
    g_refine[1] = std::acos(-gravity_in[2] / cr);
  }

  void UpdateGravity() {
    // Eigen::Map<const Eigen::Matrix<double, 2, 1>> g_param(g_refine.data());
    // gravity = gravity_factor::refined_gravity<double>(g_param);
  }

  IMUBias GetIMUBias() {
    IMUBias bias;
    bias.gyro_bias = gyro_bias;
    bias.accel_bias = acce_bias;
    return bias;
  }

  Eigen::Vector3d GetGravity() { return gravity; }

  void SetGravity(Eigen::Vector3d g) {
    // Eigen::Vector2d g_param = gravity_factor::recover_gravity_param(g);
    // g_refine = g_param;
    // UpdateGravity();
    gravity = g;
    LOG(INFO) << "Initial gravity: " <<gravity;
  }

  void SetAccelBias(Eigen::Vector3d bias) { acce_bias = bias; }

  void SetGyroBias(Eigen::Vector3d bias) { gyro_bias = bias; }

 public:
  const cvins::ParameterServer* ps_;
  // extrinsics
  Eigen::Vector3d p_LinI;
  SO3d so3_LtoI;
  Eigen::Quaterniond q_LtoI;
  SE3d se3_LtoI;
  
  Eigen::Vector3d p_CinI;
  SO3d so3_CtoI;
  Eigen::Quaterniond q_CtoI;
  SE3d se3_CtoI;

  Eigen::Vector2d g_refine;
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d acce_bias;
  Eigen::Vector3d gravity;

  double time_offset_lidar = 0.;
  double time_offset_camera = 0.;

  /// opt weight
  double global_opt_gyro_weight;

  double global_opt_acce_weight;

  double global_opt_lidar_weight;

  double global_opt_lidar_rot_weight;
  double global_opt_lidar_pos_weight;
  
  double global_opt_imu_rot_weight;
  double global_opt_imu_pos_weight;
  double global_opt_imu_vel_weight;
  double global_opt_imu_ba_weight;
  double global_opt_imu_bg_weight;

  double global_opt_cam_rot_weight;
  double global_opt_cam_pos_weight;
  double global_opt_cam_uv_weight;

  double gyro_bias_uppper_bound;
  double acce_bias_uppper_bound;

  bool enable_debug;
};

}  // namespace clins

#endif
