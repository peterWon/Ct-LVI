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

class LviEstimator{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using TrajectoryN = Trajectory<4>;
  typedef std::shared_ptr<LviEstimator> Ptr;

  LviEstimator(const ParameterServer* ps,
             std::shared_ptr<TrajectoryN> trajectory,
             std::shared_ptr<CalibParamManager> calib_param,
             std::shared_ptr<FeatureManager> feature_manager);
  ~LviEstimator(){};
 
  void AddIMUData(IMUData data) {
    data.timestamp -= trajectory_->GetDataStartTime();
    imu_data_.emplace_back(data);
    if (trajectory_init_) {
      imu_state_estimator_->FeedIMUData(data);
      imu_integrator_for_camera_->FeedIMUData(data);
      
      // for discret-time vio initialization.
      // if(!first_img) return;// no image come in yet
      // if (!first_imu){
      //   first_imu = true;
      //   acc_0 = data.accel;
      //   gyr_0 = data.gyro;
      //   last_imu_t = data.timestamp;
      // }

      // if (!pre_integrations[frame_count_]){
      //   pre_integrations[frame_count_] = new IntegrationBase{
      //     acc_0, gyr_0, Bas[frame_count_], Bgs[frame_count_],
      //     ps_->G, ps_->ACC_N, ps_->GYR_N, ps_->ACC_W, ps_->GYR_W};
      // }
      // if (frame_count_ != 0){
      //   double dt = data.timestamp - last_imu_t;
      //   last_imu_t = data.timestamp;
      //   pre_integrations[frame_count_]->push_back(
      //     dt, data.accel, data.gyro);
      //   tmp_pre_integration->push_back(dt, data.accel, data.gyro);

      //   dt_buf[frame_count_].push_back(dt);
      //   linear_acceleration_buf[frame_count_].push_back(data.accel);
      //   angular_velocity_buf[frame_count_].push_back(data.gyro);

      //   // int j = frame_count;
      //   // //推算窗口内各帧的初始位姿
      //   // //未看到显示初始化g，默认取值为０？那位置的推算是否合理？
      //   // //在初始化解算了重力向量后，会重新计算窗口内的Position和Rotation,此后的旋转和g就都是在惯性系下的了         
      //   // Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g_;
      //   // Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
      //   // Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
      //   // Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g_;
      //   // Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
      //   // Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
      //   // Vs[j] += dt * un_acc;
      // } 
      // acc_0 = data.accel; 
      // gyr_0 = data.gyro;
    }
  }

  void AddCameraData(const cvins::VinsFrameFeature& feature);
  
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

  bool BuildProblemAndSolveVI(int iteration = 100, bool lio_failed = false);
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
  void ClearImuData(){
    CHECK(trajectory_init_ == false);
    imu_data_.clear();
  }
private:
  void clearState();
  bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(
    Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);

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
    init_pose.position.setZero();
  }
  
  void triangulate();

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

  void slideWindow();
  void slideWindowNew();
  void slideWindowOld();

  /* int getFeatureCount(){
    int cnt = 0;
    for (auto &it : feature_manager_->feature){
      it.used_num = it.feature_per_frame.size();
      if (it.used_num >= 4 && it.start_frame < WINDOW_SIZE - 2
          && it.estimated_depth > 0){
        cnt++;
      }
    }
    return cnt;
  }

  VectorXd getDepthVector(){
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature_manager_->feature){
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 4 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      //未三角化或未匹配到合适的激光深度的点不参与优化
      if (it_per_id.estimated_depth > 0)
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
  }

  void setDepth(const VectorXd &x){
    int feature_index = -1;
    for (auto &it_per_id : feature_manager_->feature){
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 4 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      
      //未三角化或未匹配到合适的激光深度的点不会参与优化
      if(it_per_id.estimated_depth < 0) continue;

      it_per_id.estimated_depth = 1.0 / x(++feature_index);
      if (it_per_id.estimated_depth < 0){
        it_per_id.solve_flag = 2;
        it_per_id.is_depth_associated = false;
      }else
        it_per_id.solve_flag = 1;
    }
  } */

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

  double active_time_lower_ = 0.;
  double active_time_upper_ = 0.;
  double cur_image_time_ = -1.;
  double last_outlier_ratio = 0.;

  bool trajectory_init_ = false;
  bool initial_move_flag_ = false;

  PoseData init_pose;
  std::vector<PoseData> lidar_anchors_ = {};
  double para_Feature[NUM_OF_F][SIZE_FEATURE];

  bool use_corner_feature_ = false;
  bool use_imu_orientation_ = true;
  
  size_t img_frame_nr_ = 0;
  size_t matching_window_size_ = 7;

  float v_depth_upper_bound_ = 150.;
  float v_depth_lower_bound_ = 0.5;
  
  //////////////////////////////////////////////////////////
  enum MarginalizationFlag{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
  };
  const int camera_id_ = 0;
  int frame_count_ = 0;
  double td_ = 0.;
  MarginalizationFlag marg_flag_;

  double Headers[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  
  bool meet_initial_movement = false;
  enum SolverFlag
  {
      INITIAL,
      NON_LINEAR
  };
  SolverFlag solver_flag;
  Matrix3d ric[NUM_OF_CAM];
  Vector3d tic[NUM_OF_CAM];

  Vector3d g_;
  IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
  bool first_imu = false;
  bool first_img = false;
  double last_imu_t;
  Vector3d acc_0, gyr_0;
  vector<double> dt_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
  map<double, ImageFrame> all_image_frame={};
  IntegrationBase *tmp_pre_integration = nullptr;
  MotionEstimator m_estimator;
};
}