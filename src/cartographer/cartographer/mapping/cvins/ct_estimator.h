#pragma once
#include <iostream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <mutex>

#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "parameters.h"
#include "feature_manager.h"
#include "utility.h"
#include "utils/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"

#include "factor/imu_factor.h"
#include "factor/state_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include "odometry/imu_state_estimator.h"
#include <odometry/camera_state_estimator.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/loop_closure_data.h>
#include <sensor_data/calibration.hpp>
#include <trajectory/se3_trajectory.hpp>
#include <trajectory/trajectory_estimator.hpp>

namespace cvins{

using namespace clins;

class CtEstimator{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using TrajectoryN = Trajectory<4>;
  typedef std::shared_ptr<CtEstimator> Ptr;

  CtEstimator(const ParameterServer* ps,
             std::shared_ptr<TrajectoryN> trajectory,
             std::shared_ptr<CalibParamManager> calib_param,
             std::shared_ptr<FeatureManager> feature_manager);
  ~CtEstimator(){};

  void AddIMUData(IMUData data) {
    data.timestamp -= trajectory_->GetDataStartTime();
    imu_data_.emplace_back(data);
    if (trajectory_init_) {
      imu_state_estimator_->FeedIMUData(data);
      imu_integrator_for_camera_->FeedIMUData(data);
    }
  }

  void ClearImuData(){
    CHECK(trajectory_init_ == false);
    imu_data_.clear();
  }

  // implicitely implemented via feature manager.
  void AddCameraData(){}
  
  // caution: currently the pose data is in tracking frame.
  void AddLidarData(const std::vector<PoseData>& anchor_poses){
    lidar_anchors_ = anchor_poses;
  }
  
  // for LIO dynamic initialization. Deprecated.
  void InitializeDynamic(
      const std::deque<double>& timestamps, 
      const std::deque<Eigen::Vector3d>& ps, 
      const std::deque<Eigen::Vector3d>& vs,
      const std::deque<Eigen::Matrix3d>& rs,
      const std::deque<Eigen::Vector3d>& bas,
      const std::deque<Eigen::Vector3d>& bgs);
  
  // for LIO initialization.
  void Initialize(double scan_time, 
                  const Eigen::Quaterniond& q,
                  const Eigen::Vector3d& ba, 
                  const Eigen::Vector3d& bg);

  bool BuildProblemAndSolveVI(int iteration = 100);
  bool BuildProblemAndSolveVI(
      const std::vector<State> cam_poses, int iteration = 100);
  
  bool BuildProblemAndSolveLI(int iteration = 100);

  bool BuildProblemAndSolveLVI(int iteration = 100);
  
  void IntegrateIMUMeasurement(double scan_min, double scan_max);

  void IntegrateIMUForCamera(double image_time);
  
  std::shared_ptr<ImuStateEstimator> GetCameraStateIntegrator(){
    return imu_integrator_for_camera_;
  }
  void SetVisualDepthUpperBound(float d){
    v_depth_upper_bound_ = d;
  }
  void SetVisualDepthLowerBound(float d){
    v_depth_lower_bound_ = d;
  }
  float GetVisualDepthUpperBound(){
    return v_depth_upper_bound_;
  }
  float GetVisualDepthLowerBound(){
    return v_depth_lower_bound_;
  }

private:
  inline Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d& Vec) {
    // Returns skew-symmetric form of a 3-d vector
    Eigen::Matrix3d M;
    M << 0, -Vec(2), Vec(1),
      Vec(2), 0, -Vec(0),
      -Vec(1), Vec(0), 0;
    return M;
  }

  void InitIMUData(double feature_time);
  
  void SetInitialPoseRotation(Eigen::Quaterniond q) {
    init_pose.orientation.setQuaternion(q);
  }
  void AddCameraErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator);
  
  void AddCameraErrorTermsNoFeature(
      std::shared_ptr<TrajectoryEstimator<4>> estimator);
  
  void AddImuErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator,
      std::shared_ptr<ImuStateEstimator> imu_integrator,
      bool enable_integrated_pose,
      bool enable_predicted_pose,
      double wp,
      double wv,
      double wq);
  
  void AddLiDARErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator);
  
  void ErasePastImuData(double t);

  void AddStartTimePose(std::shared_ptr<TrajectoryEstimator<4>> estimator);
  
  void UpdateFeatureDepth();

  void UpdateTrajectoryProperty();
  
  void LogVisualError();

  void MarkVisualOutliers();

  void triangulate(const Eigen::Matrix<double, 3, 4>& pose_ref, 
                   const Eigen::Vector2d& point_ref, 
                   const std::vector<Eigen::Matrix<double, 3, 4>>& poses_obs,
                   const std::vector<Eigen::Vector2d>& points_obs, 
                   Eigen::Vector3d& point_3d);
  void triangulate();
  ///////////////////////////////////
  const ParameterServer* ps_;
  
  std::shared_ptr<TrajectoryN> trajectory_ = nullptr;
  CalibParamManager::Ptr calib_param_ = nullptr;
  std::shared_ptr<FeatureManager> feature_manager_ = nullptr;
  std::shared_ptr<ImuStateEstimator> imu_state_estimator_ = nullptr;
  std::shared_ptr<ImuStateEstimator> imu_integrator_for_camera_ = nullptr;

  std::vector<IMUData> imu_data_ = {};
  std::vector<IMUData> cache_imu_data_ = {};
  std::vector<std::pair<double, IMUBias>> cache_imu_bias_;

  double active_time_lower_, active_time_upper_;
  double cur_image_time_;

  bool trajectory_init_ = false;

  PoseData init_pose;
  std::vector<PoseData> lidar_anchors_ = {};
  double para_Feature[NUM_OF_F][SIZE_FEATURE];

  bool use_corner_feature_ = false;
  bool use_imu_orientation_ = true;
  
  size_t img_frame_nr_ = 0;
  size_t matching_window_size_ = 7;

  float v_depth_upper_bound_ = 150.;
  float v_depth_lower_bound_ = 0.5;
};
}