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

#ifndef TRAJECTORY_ESTIMATOR_HPP
#define TRAJECTORY_ESTIMATOR_HPP

#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <factor/auto_diff/imu_factor.h>
#include <factor/auto_diff/lidar_feature_factor.h>
#include <factor/auto_diff/loop_closure_factor.h>
#include <factor/auto_diff/motion_factor.h>
#include <factor/auto_diff/vicon_factor.h>
#include <factor/auto_diff/camera_reprojection_factor.h>
#include <utils/ceres_callbacks.h>
#include <basalt/spline/ceres_local_param.hpp>
#include <memory>
#include <thread>
#include <trajectory/se3_trajectory.hpp>

#include <feature/lidar_feature.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/image_data.h>
#include <sensor_data/loop_closure_data.h>
#include <sensor_data/calibration.hpp>

// #include <camera_models/Camera.h>
// #include <camera_models/CameraFactory.h>
// #include <camera_models/PinholeCamera.h>
// #include <camera_models/EquidistantCamera.h>
// #include <camera_models/CataCamera.h>

namespace clins {

template <int _N>
class TrajectoryEstimator {
  static ceres::Problem::Options DefaultProblemOptions() {
    ceres::Problem::Options options;
    options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    return options;
  }

 public:
  TrajectoryEstimator(std::shared_ptr<Trajectory<_N>> trajectory,
                      CalibParamManager::Ptr calib_param)
      /* : trajectory_(trajectory), calib_param_(calib_param), traj_locked_(true) */ {
    trajectory_ = trajectory;
    calib_param_ = calib_param;
    traj_locked_ = true;
    problem_ = std::make_shared<ceres::Problem>(DefaultProblemOptions());
    local_parameterization = new LieLocalParameterization<SO3d>();

    // camera_ = camodocal::CameraFactory::instance()
    //     ->generateCameraFromYamlFile(calib_param_->camera_intrincs_file);
  }

  void SetProblem(std::shared_ptr<ceres::Problem> problem_in) {
    problem_ = problem_in;
  }

  void SetTrajectory(std::shared_ptr<Trajectory<_N>> trajectory_in) {
    trajectory_ = trajectory_in;
  }

  void AddControlPoints(const SplineMeta<_N>& spline_meta,
                        std::vector<double*>& vec, bool addPosKont = false);

  void AddLoamMeasurement(const PointCorrespondence& pc, double weight,
                          double huber_loss = 5);
  
  void AddCameraMeasurement(
      const std::shared_ptr<sfm::Observation>& ref,
      const std::shared_ptr<sfm::Observation>& obs,
      double weight,
      double v_depth_upper_bound = 150.,
      double v_depth_lower_bound = 1.);
  
  void AddCameraMeasurement(
      const double& t_ref,
      const double& t_obs,
      const Eigen::Vector3d& pt_ref,
      const Eigen::Vector3d& pt_obs,
      double *p_rho,
      double weight,
      bool lock_depth = false,
      double v_depth_upper_bound = 150.,
      double v_depth_lower_bound = 1.);

  void AddCameraRelativePoseMeasurement(
    const double t_ref, const double t_obs, 
    const Eigen::Matrix3d& relative_rot, const Eigen::Vector3d& relative_pos,
    double pos_weight, double rot_weight, double* scale);
  
  void AddCameraRelativeRotationMeasurement(
    const double t_ref, const double t_obs, 
    const Eigen::Matrix3d& relative_rot,
    double rot_weight);

  void AddLiDARPoseMeasurement(const PoseData& pose_data, double rot_weight,
                               double pos_weight);

  void AddLiDAROrientationMeasurement(const PoseData& pose_data,
                                      double rot_weight);

  void AddIMUGyroMeasurement(const IMUData& imu_data, double gyro_weight);

  void AddIMUMeasurement(const IMUData& imu_data, double gyro_weight,
                         double accel_weight, double huber_loss = 5);

  void AddIMUBiasMeasurement(const IMUData& imu_data, const IMUBias& bias,
                             double gyro_weight, double accel_weight,
                             double ba_weight, double bg_weight);

  void AddPoseMeasurement(const PoseData& pose_data, double rot_weight,
                          double pos_weight);

  void AddPositionMeasurement(const PoseData& pose_data, double pos_weight);

  void AddOrientationMeasurement(const PoseData& pose_data, double rot_weight);

  void AddIMUDeltaOrientationMeasurement(const IMUData& ref_data,
                                         const IMUData& imu_data, double rot);

  void AddGlobalVelocityMeasurement(const IMUState& state, double vel_weight);

  void AddQuadraticIntegralFactor(double min_time, double max_time,
                                  Eigen::Matrix3d weight);

  void AddAngularVelocityConvexHullFactor(double time, double weight);

  void AddLoopClosureEdgeMeasurement(const RelativePoseData& measurement,
                                     const double pos_weight,
                                     const double rot_weight);

  void AddNeighborEdgsMeasurement(const RelativePoseData& measurement,
                                  const double pos_weight,
                                  const double rot_weight);

  void AddVelocityConstraintMeasurement(const VelocityData& measurement,
                                        const double vel_weight,
                                        const double gyro_weight);

  void AddViconPoseMeasurement(const PoseData& vicon_pose,
                               const double pos_weight,
                               const double rot_weight);

  void AddGPSPoseMeasurement(const PoseData& measurement, double pos_weight);

  void SetTrajectorControlPointVariable(double min_time, double max_time);

  void SetKeyScanConstant(double max_time);

  ceres::Solver::Summary Solve(int max_iterations = 50, bool progress = true,
                               int num_threads = -1, double max_time_cost = 0.1);

  bool getCovariance();

  bool IsLocked() const { return traj_locked_; }

  void LockTrajectory(bool lock) { traj_locked_ = lock; }

  void LockIMUState(bool lock_ab, bool lock_wb, bool lock_g);

  void LockExtrinsicParam(bool lock_P, bool lock_R);

 private:
  CalibParamManager::Ptr calib_param_;

  std::shared_ptr<Trajectory<_N>> trajectory_;
  std::shared_ptr<ceres::Problem> problem_;
  ceres::LocalParameterization* local_parameterization;

  bool traj_locked_;

  int fixed_control_point_index_ = -1;

  bool callback_needs_state_;
  std::vector<std::unique_ptr<ceres::IterationCallback>> callbacks_;
  // camodocal::CameraPtr camera_ = nullptr;
};

template <int _N>
void TrajectoryEstimator<_N>::AddControlPoints(
    const SplineMeta<_N>& spline_meta, std::vector<double*>& vec,
    bool addPosKont) {
  for (auto const& seg : spline_meta.segments) {
    size_t start_idx = trajectory_->computeTIndex(seg.t0 + 1e-9).second;
    for (size_t i = start_idx; i < (start_idx + seg.NumParameters()); ++i) {
      if (addPosKont) {
        vec.emplace_back(trajectory_->getKnotPos(i).data());
        problem_->AddParameterBlock(trajectory_->getKnotPos(i).data(), 3);
      } else {
        vec.emplace_back(trajectory_->getKnotSO3(i).data());
        problem_->AddParameterBlock(trajectory_->getKnotSO3(i).data(), 4,
                                    local_parameterization);
      }
      if (IsLocked() || (fixed_control_point_index_ >= 0 &&
                         i <= fixed_control_point_index_)) {
        problem_->SetParameterBlockConstant(vec.back());
      }
    }
  }
}

template <int _N>
void TrajectoryEstimator<_N>::AddLoamMeasurement(const PointCorrespondence& pc,
                                                 double weight,
                                                 double huber_loss) {
  if(!trajectory_->GetTrajQuality(pc.t_map)) return;
  if(!trajectory_->GetTrajQuality(pc.t_point)) return;

  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta(
      {{pc.t_map, pc.t_map}, {pc.t_point, pc.t_point}}, spline_meta);

  using Functor = PointFeatureFactor<_N>;
  Functor* functor = new Functor(pc, spline_meta, weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(4);  // R_LtoI
  cost_function->AddParameterBlock(3);  // p_LinI

  cost_function->SetNumResiduals(1);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  vec.emplace_back(calib_param_->so3_LtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_LtoI.data(), 4,
                              local_parameterization);
  vec.emplace_back(calib_param_->p_LinI.data());

  ceres::HuberLoss loss_function_(huber_loss);
  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddLiDARPoseMeasurement(const PoseData& pose_data,
                                                      double rot_weight,
                                                      double pos_weight) {
  if(!trajectory_->GetTrajQuality(pose_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{pose_data.timestamp, pose_data.timestamp}},
                                  spline_meta);

  using Functor = LiDARPoseFactor<_N>;
  Functor* functor =
      new Functor(pose_data, spline_meta, rot_weight, pos_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(4);  // R_LtoI
  cost_function->AddParameterBlock(3);  // p_LinI
  cost_function->AddParameterBlock(1);  // time_offset

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);
  vec.emplace_back(calib_param_->so3_LtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_LtoI.data(), 4,
                              local_parameterization);
  vec.emplace_back(calib_param_->p_LinI.data());

  vec.emplace_back(&calib_param_->time_offset_lidar);

  problem_->AddParameterBlock(&calib_param_->time_offset_lidar, 1);

  problem_->SetParameterBlockConstant(&calib_param_->time_offset_lidar);

  cost_function->SetNumResiduals(6);
  problem_->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddCameraRelativeRotationMeasurement(
    const double t_ref, const double t_obs, 
    const Eigen::Matrix3d& relative_rot,
    double rot_weight){
  if(!trajectory_->GetTrajQuality(t_ref)) return;
  if(!trajectory_->GetTrajQuality(t_obs)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{t_ref, t_ref},{t_obs, t_obs}}, spline_meta);

  using Functor = RelativeCameraRotationFactor<_N>;
  Functor* functor = new Functor(t_ref, t_obs, relative_rot, 
              spline_meta, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  
  cost_function->AddParameterBlock(4);  // R_CtoI
  cost_function->AddParameterBlock(1);  // time_offset
  
  // add trajectory
  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  // add camera-IMU extrinsics
  vec.emplace_back(calib_param_->so3_CtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_CtoI.data(), 4,
                              local_parameterization);
  problem_->SetParameterBlockConstant(calib_param_->so3_CtoI.data());

  // add time offset
  vec.emplace_back(&calib_param_->time_offset_camera);
  problem_->AddParameterBlock(&calib_param_->time_offset_camera, 1);
  problem_->SetParameterBlockConstant(&calib_param_->time_offset_camera);
  
  cost_function->SetNumResiduals(3);
  problem_->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddCameraRelativePoseMeasurement(
    const double t_ref, const double t_obs, 
    const Eigen::Matrix3d& relative_rot, const Eigen::Vector3d& relative_pos,
    double pos_weight, double rot_weight, double* scale) {
  if(!trajectory_->GetTrajQuality(t_ref)) return;
  if(!trajectory_->GetTrajQuality(t_obs)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{t_ref, t_ref},{t_obs, t_obs}}, spline_meta);

  using Functor = RelativeCameraPoseFactor<_N>;
  Functor* functor = new Functor(t_ref, t_obs, relative_rot, relative_pos, 
              spline_meta, pos_weight, rot_weight, scale != NULL);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(4);  // R_CtoI
  cost_function->AddParameterBlock(3);  // p_CinI
  cost_function->AddParameterBlock(1);  // time_offset
  if(scale){
    cost_function->AddParameterBlock(1);  // scale
  }
  
  // add trajectory
  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  // add camera-IMU extrinsics
  vec.emplace_back(calib_param_->so3_CtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_CtoI.data(), 4,
                              local_parameterization);
  vec.emplace_back(calib_param_->p_CinI.data());
  
  // add time offset
  vec.emplace_back(&calib_param_->time_offset_camera);
  problem_->AddParameterBlock(&calib_param_->time_offset_camera, 1);
  problem_->SetParameterBlockConstant(&calib_param_->time_offset_camera);
  
  // add scale
  if(scale){
    vec.emplace_back(scale);
    problem_->AddParameterBlock(scale, 1);
    // problem_->SetParameterLowerBound(scale, 0, 0.); 
  }
  
  cost_function->SetNumResiduals(6);
  problem_->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), vec);
}


template <int _N>
void TrajectoryEstimator<_N>::AddCameraMeasurement(
    const double& t_ref,
    const double& t_obs,
    const Eigen::Vector3d& pt_ref,
    const Eigen::Vector3d& pt_obs,
    double *p_rho,
    double weight,
    bool lock_depth,
    double v_depth_upper_bound,
    double v_depth_lower_bound) {
  if(!trajectory_->GetTrajQuality(t_ref)) return;
  if(!trajectory_->GetTrajQuality(t_obs)) return;
  
  // The list of spans must be ordered
  double t0_ref = t_ref;
  double t0_obs = t_obs;
  double t1, t2;
  if (t0_ref <= t0_obs) {
    t1 = t0_ref;
    t2 = t0_obs;
  } else {
    t1 = t0_obs;
    t2 = t0_ref;
  }
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{t1, t1}, {t2, t2}}, spline_meta);

  using Functor = ReprojectionFactorV1<_N>;
  Functor* functor =
      new Functor(t_ref, t_obs, pt_ref, pt_obs, spline_meta, weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(4);  // R_CtoI
  cost_function->AddParameterBlock(3);  // p_CinI
  cost_function->AddParameterBlock(1);  // time_offset
  cost_function->AddParameterBlock(1);  // inverse_depth
  
  // add trajectory
  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  // add camera-IMU extrinsics
  vec.emplace_back(calib_param_->so3_CtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_CtoI.data(), 4,
                              local_parameterization);
  vec.emplace_back(calib_param_->p_CinI.data());
  problem_->AddParameterBlock(calib_param_->p_CinI.data(), 3);
  
  
  // add time offset
  vec.emplace_back(&calib_param_->time_offset_camera);
  problem_->AddParameterBlock(&calib_param_->time_offset_camera, 1);
  problem_->SetParameterBlockConstant(&calib_param_->time_offset_camera);
  
  // add inverse depth
  vec.emplace_back(p_rho);
  problem_->AddParameterBlock(p_rho, 1);
  problem_->SetParameterLowerBound(p_rho, 0, 1. / v_depth_upper_bound);//50m 
  problem_->SetParameterUpperBound(p_rho, 0, 1. / v_depth_lower_bound);//0.5m 
  if(lock_depth){
    problem_->SetParameterBlockConstant(p_rho);
  } 
  
  cost_function->SetNumResiduals(2);
  problem_->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), vec);
}


template <int _N>
void TrajectoryEstimator<_N>::AddCameraMeasurement(
    const std::shared_ptr<sfm::Observation>& ref,
    const std::shared_ptr<sfm::Observation>& obs,
    double weight,
    double v_depth_upper_bound,
    double v_depth_lower_bound) {
  if(!trajectory_->GetTrajQuality(ref->view()->t0())) return;
  if(!trajectory_->GetTrajQuality(obs->view()->t0())) return;
  
  // The list of spans must be ordered
  double t0_ref = ref->view()->t0();
  double t0_obs = obs->view()->t0();
  double t1, t2;
  if (t0_ref <= t0_obs) {
    t1 = t0_ref;
    t2 = t0_obs;
  } else {
    t1 = t0_obs;
    t2 = t0_ref;
  }
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{t1, t1}, {t2, t2}}, spline_meta);

  using Functor = ReprojectionFactor<_N>;
  Functor* functor =
      new Functor(ref, obs, spline_meta, /* camera_, */ weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(4);  // R_CtoI
  cost_function->AddParameterBlock(3);  // p_CinI
  cost_function->AddParameterBlock(1);  // time_offset
  cost_function->AddParameterBlock(1);  // inverse_depth
  
  // add trajectory
  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  // add camera-IMU extrinsics
  vec.emplace_back(calib_param_->so3_CtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_CtoI.data(), 4,
                              local_parameterization);
  vec.emplace_back(calib_param_->p_CinI.data());
  
  // add time offset
  vec.emplace_back(&calib_param_->time_offset_camera);
  problem_->AddParameterBlock(&calib_param_->time_offset_camera, 1);
  problem_->SetParameterBlockConstant(&calib_param_->time_offset_camera);
  
  // add inverse depth
  double *p_rho = ref->landmark()->inverse_depth_ptr();
  vec.emplace_back(p_rho);
  problem_->AddParameterBlock(p_rho, 1);
  problem_->SetParameterLowerBound(p_rho, 0, 1. / v_depth_upper_bound);//50m 
  problem_->SetParameterUpperBound(p_rho, 0, 1. / v_depth_lower_bound);//0.5m 
  if(ref->landmark()->IsLocked()){
    problem_->SetParameterBlockConstant(p_rho);
  } 
  
  cost_function->SetNumResiduals(2);
  problem_->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddLiDAROrientationMeasurement(
    const PoseData& pose_data, double rot_weight) {
  if(!trajectory_->GetTrajQuality(pose_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{pose_data.timestamp, pose_data.timestamp}},
                                  spline_meta);

  using Functor = LiDAROrientationFactor<_N>;
  Functor* functor = new Functor(pose_data, spline_meta, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }

  cost_function->AddParameterBlock(4);  // R_LtoI

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  vec.emplace_back(calib_param_->so3_LtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_LtoI.data(), 4,
                              local_parameterization);

  cost_function->SetNumResiduals(3);
  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddIMUMeasurement(const IMUData& imu_data,
                                                double gyro_weight,
                                                double accel_weight,
                                                double huber_loss) {
  if(!trajectory_->GetTrajQuality(imu_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{imu_data.timestamp, imu_data.timestamp}},
                                  spline_meta);
  using Functor = GyroAcceWithConstantBiasFactor<_N>;
  Functor* functor =
      new Functor(imu_data, spline_meta, calib_param_->ps_->GRAVITY_NORM,
                  gyro_weight, accel_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(3);  // gyro bias
  cost_function->AddParameterBlock(3);  // acce bias
  cost_function->AddParameterBlock(2);  // g_refine

  cost_function->SetNumResiduals(6);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  vec.emplace_back(calib_param_->gyro_bias.data());
  vec.emplace_back(calib_param_->acce_bias.data());
  vec.emplace_back(calib_param_->g_refine.data());

  problem_->AddParameterBlock(calib_param_->g_refine.data(), 2);
  for (int i = 0; i < 2; i++) {
    problem_->SetParameterLowerBound(calib_param_->g_refine.data(), i,
                                     -M_PI / 2.0);
    problem_->SetParameterUpperBound(calib_param_->g_refine.data(), i,
                                     M_PI / 2.0);
  }
  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddIMUBiasMeasurement(const IMUData& imu_data,
                                                    const IMUBias& bias,
                                                    double gyro_weight,
                                                    double accel_weight,
                                                    double ba_weight,
                                                    double bg_weight) {
  if(!trajectory_->GetTrajQuality(imu_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{imu_data.timestamp, imu_data.timestamp}},
                                  spline_meta);
  using Functor = GyroAcceBiasFactor<_N>;
  Functor* functor = new Functor(imu_data, bias, spline_meta, 
                                 calib_param_->ps_->GRAVITY_NORM,
                                 gyro_weight,
                                 accel_weight, ba_weight, bg_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }
  cost_function->AddParameterBlock(3);  // gyro bias
  cost_function->AddParameterBlock(3);  // acce bias
  cost_function->AddParameterBlock(2);  // g_refine

  cost_function->SetNumResiduals(12);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  vec.emplace_back(calib_param_->gyro_bias.data());
  vec.emplace_back(calib_param_->acce_bias.data());
  vec.emplace_back(calib_param_->g_refine.data());

  problem_->AddParameterBlock(calib_param_->g_refine.data(), 2);
  for (int i = 0; i < 2; i++) {
    problem_->SetParameterLowerBound(calib_param_->g_refine.data(), i,
                                     -M_PI / 2.0);
    problem_->SetParameterUpperBound(calib_param_->g_refine.data(), i,
                                     M_PI / 2.0);
  }
  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddIMUGyroMeasurement(const IMUData& imu_data,
                                                    double gyro_weight) {
  if(!trajectory_->GetTrajQuality(imu_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{imu_data.timestamp, imu_data.timestamp}},
                                  spline_meta);
  using Functor = GyroscopeWithConstantBiasFactor<_N>;
  Functor* functor = new Functor(imu_data, spline_meta, gyro_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  cost_function->AddParameterBlock(3);  // gyro bias

  cost_function->SetNumResiduals(3);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  vec.emplace_back(calib_param_->gyro_bias.data());
  problem_->AddParameterBlock(calib_param_->gyro_bias.data(), 3);
  problem_->SetParameterBlockConstant(calib_param_->gyro_bias.data());

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddPoseMeasurement(const PoseData& pose_data,
                                                 double rot_weight,
                                                 double pos_weight) {
  if(!trajectory_->GetTrajQuality(pose_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{pose_data.timestamp, pose_data.timestamp}},
                                  spline_meta);

  using Functor = IMUPoseFactor<_N>;
  Functor* functor =
      new Functor(pose_data, spline_meta, rot_weight, pos_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(6);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddPositionMeasurement(const PoseData& pose_data,
                                                     double pos_weight) {
  if(!trajectory_->GetTrajQuality(pose_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{pose_data.timestamp, pose_data.timestamp}},
                                  spline_meta);

  using Functor = IMUPositionFactor<_N>;
  Functor* functor = new Functor(pose_data, spline_meta, pos_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(3);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddOrientationMeasurement(
    const PoseData& pose_data, double rot_weight) {
  if(!trajectory_->GetTrajQuality(pose_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{pose_data.timestamp, pose_data.timestamp}},
                                  spline_meta);

  using Functor = IMUOrientationFactor<_N>;
  Functor* functor = new Functor(pose_data, spline_meta, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add SO3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }

  cost_function->SetNumResiduals(3);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddIMUDeltaOrientationMeasurement(
    const IMUData& ref_data, const IMUData& imu_data, double rot_weight) {
  if(!trajectory_->GetTrajQuality(ref_data.timestamp)) return;
  if(!trajectory_->GetTrajQuality(imu_data.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{ref_data.timestamp, imu_data.timestamp}},
                                  spline_meta);

  using Functor = IMUDeltaOrientationFactor<_N>;
  Functor* functor = new Functor(ref_data, imu_data, spline_meta, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add SO3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }

  cost_function->SetNumResiduals(3);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddGlobalVelocityMeasurement(
    const IMUState& state, double vel_weight) {
  if(!trajectory_->GetTrajQuality(state.timestamp)) return;
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{state.timestamp, state.timestamp}},
                                  spline_meta);

  using Functor = IMUGlobalVelocityFactor<_N>;
  Functor* functor =
      new Functor(state.v, state.timestamp, spline_meta, vel_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(3);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddQuadraticIntegralFactor(
    double min_time, double max_time, Eigen::Matrix3d weight) {
  size_t min_s = trajectory_->computeTIndex(min_time).second;
  size_t max_s = trajectory_->computeTIndex(max_time).second;

  std::vector<double> times;
  for (size_t i = min_s; i <= max_s; i++) {
    double timestamp = trajectory_->minTime() + i * trajectory_->getDt();
    times.push_back(timestamp);
  }
  times.back() += 1e-10;
  std::cout << YELLOW << "[AddQuadraticIntegralFactor] "
            << "from " << trajectory_->minTime() + min_s * trajectory_->getDt()
            << " to " << trajectory_->minTime() + max_s * trajectory_->getDt()
            << RESET << std::endl;

  if (times.empty()) return;

  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{times.front(), times.back()}}, spline_meta);

  using Functor = QuadraticIntegralFactor<_N, 2>;
  Functor* functor = new Functor(times, spline_meta, weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add R3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(3 * _N *
                                 times.size());  // times.size() 段区间积分

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddAngularVelocityConvexHullFactor(
    double time, double weight) {
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta({{time, time}}, spline_meta);

  using Functor = AngularVelocityConvexHullFactor<_N>;
  Functor* functor = new Functor(spline_meta, weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }

  size_t v_kont_num = spline_meta.NumParameters() - spline_meta.segments.size();
  cost_function->SetNumResiduals(v_kont_num);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddLoopClosureEdgeMeasurement(
    const RelativePoseData& measurement, const double pos_weight,
    const double rot_weight) {
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta(
      {{measurement.target_timestamp, measurement.target_timestamp},
       {measurement.source_timestamp, measurement.source_timestamp}},
      spline_meta);

  using Functor = LoopClosureEdgesFactor<_N>;
  Functor* functor =
      new Functor(measurement, spline_meta, pos_weight, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->AddParameterBlock(4);  // R_LtoI
  cost_function->AddParameterBlock(3);  // p_LinI

  cost_function->SetNumResiduals(6);

  std::vector<double*> vec;

  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  vec.emplace_back(calib_param_->so3_LtoI.data());
  problem_->AddParameterBlock(calib_param_->so3_LtoI.data(), 4,
                              local_parameterization);
  problem_->SetParameterBlockConstant(calib_param_->so3_LtoI.data());
  vec.emplace_back(calib_param_->p_LinI.data());
  problem_->AddParameterBlock(calib_param_->p_LinI.data(), 3);
  problem_->SetParameterBlockConstant(calib_param_->p_LinI.data());

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddNeighborEdgsMeasurement(
    const RelativePoseData& measurement, const double pos_weight,
    const double rot_weight) {
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta(
      {{measurement.target_timestamp, measurement.target_timestamp},
       {measurement.source_timestamp, measurement.source_timestamp}},
      spline_meta);

  using Functor = RelativeTrajectoryPoseFactor<_N>;
  Functor* functor =
      new Functor(measurement, spline_meta, pos_weight, rot_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(6);

  std::vector<double*> vec;

  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::AddVelocityConstraintMeasurement(
    const VelocityData& measurement, const double vel_weight,
    const double gyro_weight) {
  SplineMeta<_N> spline_meta;
  trajectory_->CaculateSplineMeta(
      {{measurement.timestamp, measurement.timestamp}}, spline_meta);

  using Functor = VelocityConstraintFactor<_N>;
  Functor* functor =
      new Functor(measurement, spline_meta, gyro_weight, vel_weight);
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

  /// add so3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(4);
  }
  /// add vec3 knots
  for (int i = 0; i < spline_meta.NumParameters(); i++) {
    cost_function->AddParameterBlock(3);
  }

  cost_function->SetNumResiduals(6);

  std::vector<double*> vec;
  AddControlPoints(spline_meta, vec);
  AddControlPoints(spline_meta, vec, true);

  problem_->AddResidualBlock(cost_function, NULL, vec);
}

template <int _N>
void TrajectoryEstimator<_N>::SetTrajectorControlPointVariable(
    double min_time, double max_time) {
  size_t min_s = trajectory_->computeTIndex(min_time).second;
  size_t max_s = trajectory_->computeTIndex(max_time).second;

  std::cout << "[SetTrajectorControlPointVariable]: " << min_s << ", "
            << max_s + _N - 1 << "\n";
  for (size_t i = min_s; i < (max_s + _N); i++) {
    problem_->AddParameterBlock(trajectory_->getKnotSO3(i).data(), 4,
                                local_parameterization);
    problem_->SetParameterBlockVariable(trajectory_->getKnotSO3(i).data());

    problem_->AddParameterBlock(trajectory_->getKnotPos(i).data(), 3);
    problem_->SetParameterBlockVariable(trajectory_->getKnotPos(i).data());
  }
}

template <int _N>
void TrajectoryEstimator<_N>::SetKeyScanConstant(double max_time) {
  //  std::pair<double, size_t> min_i_s = trajectory_->computeTIndex(min_time);
  std::pair<double, size_t> max_i_s = trajectory_->computeTIndex(max_time);

  if (max_i_s.first < 0.5) {
    fixed_control_point_index_ = max_i_s.second + _N - 2;
  } else {
    fixed_control_point_index_ = max_i_s.second + _N - 1;
  }

  //  std::cout << BLUE
  //            << " fixed_control_point_index_ : " <<
  //            fixed_control_point_index_
  //            << RESET << std::endl;
}

template <int _N>
void TrajectoryEstimator<_N>::LockIMUState(bool lock_ab, bool lock_wb,
                                           bool lock_g) {
  if (lock_ab && problem_->HasParameterBlock(calib_param_->acce_bias.data())) {
    problem_->AddParameterBlock(calib_param_->acce_bias.data(), 3);
    problem_->SetParameterBlockConstant(calib_param_->acce_bias.data());
  }
  if (lock_wb && problem_->HasParameterBlock(calib_param_->gyro_bias.data())) {
    problem_->AddParameterBlock(calib_param_->gyro_bias.data(), 3);
    problem_->SetParameterBlockConstant(calib_param_->gyro_bias.data());
  }
  if (lock_g && problem_->HasParameterBlock(calib_param_->g_refine.data())) {
    problem_->AddParameterBlock(calib_param_->g_refine.data(), 2);
    problem_->SetParameterBlockConstant(calib_param_->g_refine.data());
  }
}

template <int _N>
void TrajectoryEstimator<_N>::LockExtrinsicParam(bool lock_P, bool lock_R) {
  if (lock_P) {
    problem_->AddParameterBlock(calib_param_->p_LinI.data(), 3);
    problem_->SetParameterBlockConstant(calib_param_->p_LinI.data());
  }
  if (lock_R) {
    problem_->AddParameterBlock(calib_param_->so3_LtoI.data(), 4,
                                local_parameterization);
    problem_->SetParameterBlockConstant(calib_param_->so3_LtoI.data());
  }
}

template <int _N>
ceres::Solver::Summary TrajectoryEstimator<_N>::Solve(int max_iterations,
                                                      bool progress,
                                                      int num_threads,
                                                      double max_time_cost) {
  ceres::Solver::Options options;
  
  options.minimizer_type = ceres::TRUST_REGION;
  //  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  //  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  // options.linear_solver_type = ceres::DENSE_QR;
  // options.trust_region_strategy_type = ceres::DOGLEG;
  // options.dense_linear_algebra_library_type = ceres::CUDA;
    //  options.trust_region_strategy_type = ceres::DOGLEG;
  //    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

  //    options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

  options.minimizer_progress_to_stdout = progress;

  if (num_threads < 1) {
    num_threads = std::thread::hardware_concurrency();
  }
  // LOG(WARNING)<<"Using "<<num_threads<< " threads for front-end optimization.";
  options.num_threads = num_threads;
  options.max_num_iterations = max_iterations;
  options.max_solver_time_in_seconds = max_time_cost;//~the scan rate is about 10hz
  
  if (callbacks_.size() > 0) {
    for (auto& cb : callbacks_) {
      options.callbacks.push_back(cb.get());
    }

    if (callback_needs_state_) options.update_state_every_iteration = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem_.get(), &summary);

  // update state
  calib_param_->UpdateExtrinicParam();
  // calib_param_->UpdateGravity();
  //  getCovariance();
  return summary;
}

template <int _N>
bool TrajectoryEstimator<_N>::getCovariance() {
  ceres::Covariance::Options options;
  // options.algorithm_type = ceres::DENSE_SVD;
  options.apply_loss_function = false;
  ceres::Covariance covariance(options);

  if (!problem_->HasParameterBlock(calib_param_->p_LinI.data()) ||
      !problem_->HasParameterBlock(calib_param_->so3_LtoI.data())) {
    return false;
  }

  std::vector<const double*> vec;
  vec.emplace_back(calib_param_->p_LinI.data());
  vec.emplace_back(calib_param_->so3_LtoI.data());

  if (!covariance.Compute(vec, problem_.get())) {
    std::cout
        << "[CovarianceMatrixInTangentSpace] J^TJ is a rank deficient matrix\n";
    return false;
  }

  double m2cm = 100;
  double rad2degree = 180.0 / M_PI;

  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
  covariance.GetCovarianceMatrixInTangentSpace(vec, cov.data());

  Eigen::VectorXd diag;
  diag = cov.diagonal();
  diag.head<6>() = diag.head<6>().cwiseSqrt();
  diag.segment<3>(0) *= m2cm;
  diag.segment<3>(3) *= rad2degree;

  std::cout << std::fixed << std::setprecision(9);
  std::cout << "[CovarianceMatrixInTangentSpace] \n" << cov << std::endl;
  std::cout << YELLOW;
  std::cout << "[std] pos (cm)    : " << diag.segment<3>(0).transpose()
            << std::endl;
  std::cout << "[std] rot (degree): " << diag.segment<3>(3).transpose() << RESET
            << std::endl;
  return true;
}

}  // namespace clins

#endif
