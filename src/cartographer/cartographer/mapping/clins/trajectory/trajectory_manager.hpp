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

#ifndef TRAJECTORY_MANAGER_HPP
#define TRAJECTORY_MANAGER_HPP

#include <algorithm>
#include <odometry/imu_state_estimator.h>
#include <odometry/camera_state_estimator.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/loop_closure_data.h>
#include <utils/tic_toc.h>
#include <sensor_data/calibration.hpp>
#include <trajectory/se3_trajectory.hpp>
#include <trajectory/trajectory_estimator.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <glog/logging.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>

#include <cartographer/mapping/sfm/sfm.h>
#include <cartographer/transform/rigid_transform.h>

namespace clins {

template <int _N>
class TrajectoryManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using TrajectoryN = Trajectory<_N>;

  typedef std::shared_ptr<TrajectoryManager<_N>> Ptr;

  TrajectoryManager(std::shared_ptr<TrajectoryN> trajectory,
                    std::shared_ptr<CalibParamManager> calib_param,
                    std::shared_ptr<ImuStateEstimator> state_estimator,
                    std::shared_ptr<ImuStateEstimator> state_estimator_camera,
                    size_t window_size)
      : trajectory_(trajectory),
        calib_param_(calib_param),
        imu_state_estimator_(state_estimator),
        imu_integrator_for_camera_(state_estimator_camera),
        matching_window_size_(window_size) {}

  void SetTrajectory(std::shared_ptr<TrajectoryN> trajectory) {
    trajectory_ = trajectory;
  }

  void AddIMUData(IMUData data) {
    data.timestamp -= trajectory_->GetDataStartTime();
    imu_data_.emplace_back(data);
    if (trajectory_init_) {
      imu_state_estimator_->FeedIMUData(data);
      imu_integrator_for_camera_->FeedIMUData(data);
      // cache_imu_data_.emplace_back(data);
    }
  }

  void AddImageData(
      double timestamp,
      const std::vector<Eigen::Vector2f>& tracked_feature,
      const std::vector<size_t>& tracked_id,
      const std::vector<Eigen::Vector3f>& new_feature_3d,
      const std::vector<size_t>& new_id,
      const cartographer::transform::Rigid3d& cam_pose_predict){
    
    std::shared_ptr<sfm::View> view = std::make_shared<sfm::View>(
        img_frame_nr_++, timestamp);

    // 1st step: create landmark which are first time observed.
    for(size_t i = 0; i < new_id.size(); ++i){
      size_t id = new_id[i];
      std::shared_ptr<sfm::Landmark> lm = std::make_shared<sfm::Landmark>();
      lm->set_id(id);
      Eigen::Vector2d uv;

      if(new_feature_3d[i][2] < 0){
        uv << new_feature_3d[i][0], new_feature_3d[i][1];
        lm->set_inverse_depth(-1.);
      }else{
        double inverse_depth = 1.0 / new_feature_3d[i][2];
        uv << new_feature_3d[i][0] * inverse_depth, 
            new_feature_3d[i][1] * inverse_depth;
        lm->set_inverse_depth(inverse_depth);
      }
      
      // create and assign reference frame to the feature when it was first time observed.
      std::shared_ptr<sfm::Observation> obs = view->CreateObservation(lm, uv);
      lm->set_reference(obs);

      landmarks_.insert(lm);
    }
    
    // 2nd step: create observations for tracked landmarks
    for(size_t i = 0; i < tracked_id.size(); ++i){
      size_t id = tracked_id[i];
      
      auto lm = std::find_if(landmarks_.begin(), landmarks_.end(), [&id](
          const std::shared_ptr<sfm::Landmark>& a){
        return id == a->id();
      });
      if(lm != landmarks_.end()){
        view->CreateObservation(*lm, tracked_feature[i].cast<double>());
        if((*lm)->inverse_depth() > 0) { 
          // cache data for PnP estimation
        }else{
          // features which are without proper depth guess.
          CHECK((*lm)->reference());
          CHECK((*lm)->reference()->view());

          // ensure enough parallax
          if((*lm)->observations().size() < 4) continue;

          SE3d pose_ref;
          double t0 = (*lm)->reference()->view()->t0();
          if(!trajectory_->GetCameraPose(t0, pose_ref)) continue;
          
          Eigen::Vector3d pt_3d_w, pt_3d_ref;
          Eigen::Matrix<double, 3, 4> epose_ref, epose_obs;
          epose_ref.block(0, 0, 3, 3) = 
              pose_ref.unit_quaternion().toRotationMatrix();
          epose_ref.block(0, 3, 3, 1) = pose_ref.translation();
          epose_obs.block(0, 0, 3, 3) = 
              cam_pose_predict.rotation().toRotationMatrix();
          epose_obs.block(0, 3, 3, 1) = cam_pose_predict.translation();
          
          // triangulate
          triangulate(epose_ref, epose_obs, (*lm)->reference()->uv(), 
              tracked_feature[i].cast<double>(), pt_3d_w);
          pt_3d_ref = pose_ref.inverse() * pt_3d_w;

          double cache_size = 1.0; // triangulated points has some errors
          if(pt_3d_ref[2] < v_depth_lower_bound_ - cache_size 
              || pt_3d_ref[2] > v_depth_upper_bound_ + cache_size) continue;
          
          // check reprojection error
          Eigen::Vector3d pt_obs = cam_pose_predict.inverse() * pt_3d_w;
          Eigen::Vector2d err;
          err << pt_obs[0]/pt_obs[2] - tracked_feature[i][0],
                 pt_obs[1]/pt_obs[2] - tracked_feature[i][1];
          if(err.norm() < 1.){
            (*lm)->set_inverse_depth(1. / pt_3d_ref[2]);
          }
        }
      }else{
        // maybe filtered when it was first time observed while tracked later.
        // or whose referenced landmark has been removed.
        // LOG(ERROR)<<"Something wrong.";
      }
    }
    

    views_.push_back(view);
  
    // LOG(INFO) <<"landmark size: "<< landmarks_.size();
    // LOG(INFO) <<"view size: " <<views_.size();
    // erase views which are outside of the matching window and with no landmark.
    if(views_.size() > matching_window_size_){
      landmarks_in_window_.clear();
      size_t oldest_reference_frame_nr = views_.back()->frame_nr();
      for(int i = views_.size() - 1; i >= views_.size() - matching_window_size_ 
          && i>=0; --i){
        for(const auto& obs: views_[i]->observations()){
          landmarks_in_window_.insert(obs->landmark());
          size_t fn = obs->landmark()->reference()->view()->frame_nr();
          if(fn < oldest_reference_frame_nr){
            oldest_reference_frame_nr = fn;
          }
        }
      }
      // 保留至窗口内观测的最早的reference.
      // for(auto it = views_.begin(); it < views_.end(); ){
      //   if((*it)->frame_nr() < oldest_reference_frame_nr){
      //     it = views_.erase(it);
      //   }else{
      //     it++;
      //   }
      // }

      // only keep landmarks in the window.
      auto& first_view = views_.front();
      // LOG(INFO)<<views_.front()->frame_nr() <<"->"<< views_.back()->frame_nr();
      // LOG(INFO)<<views_.front()->t0() <<"->"<< views_.back()->t0();

      // slide reference
      for(auto it = landmarks_.begin(); it!=landmarks_.end(); it++){
        CHECK((*it)->reference());
        CHECK((*it)->reference()->view());
        if((*it)->reference()->view()->frame_nr() > first_view->frame_nr()) continue;

        bool slide_ok = false;
        const auto& lm_obs = (*it)->observations();        
        for(int i = 0; i < lm_obs.size(); ++i){
          const auto& obs = lm_obs[i];
          CHECK(obs);
          CHECK(obs->view());
          if(obs == (*it)->reference()) continue;
          if(obs->view()->frame_nr() <= first_view->frame_nr()) continue;

          // assign new reference and associated variables.
          if((*it)->inverse_depth() > 0.){
            double t_ref = (*it)->reference()->view()->t0();
            double t_obs = obs->view()->t0();
            SE3d pose_ref, pose_obs;
            if(trajectory_->GetCameraPose(t_ref, pose_ref)
                && trajectory_->GetCameraPose(t_obs, pose_obs)){
              Eigen::Vector3d pt_ref, pt_obs;
              double d = 1. / (*it)->inverse_depth();
              pt_ref << (*it)->reference()->uv()[0] * d,
                        (*it)->reference()->uv()[1] * d, d;
              pt_obs = pose_obs.inverse() * pose_ref * pt_ref;
              if(pt_obs[2] > v_depth_lower_bound_ 
                  && pt_obs[2] < v_depth_upper_bound_){
                // silde succeed.
                (*it)->set_inverse_depth(1. / pt_obs[2]);
                (*it)->set_reference(obs);
                slide_ok = true;
                break;
              }
            }
          }else{// depth not estimated, just slide reference.
            (*it)->set_inverse_depth(-1.); // resort to triangulation
            (*it)->set_reference(obs);
            slide_ok = true;
            break;
          }
        }

        // remove observations which are associated with the invalid landmark.
        if(!slide_ok){
          for(auto v: views_){
            for(auto obs: v->observations()){
              if(obs->landmark() == (*it)){
                v->RemoveObservation(obs);
              }
            }
          }
        }
      }
      
      // // now, remove the first view and the observations of landmarks.
      for(auto& obs: views_.front()->observations()){
        views_.front()->RemoveObservation(obs);
      }
      views_.pop_front();
      
      // remove landmarks out of scope or that with no observations.
      for(auto it_lm=landmarks_.begin(); it_lm!=landmarks_.end();){
        auto it = std::find(
          landmarks_in_window_.begin(), landmarks_in_window_.end(), *it_lm);
        if(it == landmarks_in_window_.end() ){//|| !(*it_lm)->reference()
          it_lm = landmarks_.erase(it_lm);
        }else{
          it_lm++;
        }
      }
      int num_depth_estimated = 0;
      for(const auto& lm: landmarks_){
        if(lm->inverse_depth() > 0) num_depth_estimated++;
      }
      if(calib_param_->enable_debug){
        LOG(INFO)<<"Landmark and Inited landmarks: "
          <<landmarks_.size()<<", "<<num_depth_estimated;
      }
    } 
  }

  
  void InitIMUData(double feature_time) {
    double traj_start_time;
    for (size_t i = imu_data_.size() - 1; i >= 0; i--) {
      if (imu_data_[i].timestamp <= feature_time) {
        traj_start_time = imu_data_[i].timestamp;
        break;
      }
    }

    /// remove imu data
    auto iter = imu_data_.begin();
    while (iter != imu_data_.end()) {
      if (iter->timestamp < traj_start_time) {
        iter = imu_data_.erase(iter);
      } else {
        iter->timestamp -= traj_start_time;
        iter++;
      }
    }

    //    std::cout << "IMU data first timestamp : " <<
    //    imu_data_.front().timestamp
    //              << std::endl;
    LOG(WARNING)<<"IMU data size: "<<imu_data_.size();
    trajectory_->setDataStartTime(traj_start_time);
    imu_state_estimator_->SetFirstIMU(imu_data_.front());
    imu_integrator_for_camera_->SetFirstIMU(imu_data_.front());
    for (size_t i = 1; i < imu_data_.size(); i++) {
      imu_state_estimator_->FeedIMUData(imu_data_[i]);
      imu_integrator_for_camera_->FeedIMUData(imu_data_[i]);
    }
    trajectory_init_ = true;
  }

  std::shared_ptr<TrajectoryN> get_trajectory() { return trajectory_; }

  bool BuildProblemAndSolve(
      const Eigen::aligned_vector<PointCorrespondence>& point_measurement,
      int iteration = 100);
  
  bool BuildProblemAndSolve(
      const std::vector<PoseData>& lidar_pose, int iteration = 100);

  bool BuildProblemAndSolveLVI(
      const std::vector<PoseData>& lidar_pose, int iteration = 100);

  bool BuildProblemAndSolveLI(
      const std::vector<PoseData>& lidar_pose, int iteration = 100);

  bool BuildProblemAndSolveVI(int iteration = 100); 

  void UpdateTrajectoryProperty();

  void IntegrateIMUMeasurement(double scan_min, double scan_max);

  void IntegrateIMUForCamera(double image_time);

  void EstimateIMUBias();

  void OptimiseLoopClosure(const LoopClosureOptimizationParam& param,
                           const LoopClosureWeights& weights);

  void SetInitialPoseRotation(Eigen::Quaterniond q) {
    init_pose.orientation.setQuaternion(q);
  }

  void SetUseCornerFeature(bool use_corner) {
    use_corner_feature_ = use_corner;
  }

  void SetUseIMUOrientation(bool use_orientation) {
    use_imu_orientation_ = use_orientation;
  }

  void PlotIMUMeasurement(std::string cache_path) {
    // TrajectoryViewer::PublishIMUData<_N>(trajectory_, cache_imu_data_,
    //                                      cache_imu_bias_, cache_path);
  }

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
  //ref. MVG p242
  void triangulate(const Eigen::Matrix<double, 3, 4> &Pose0, 
                   const Eigen::Matrix<double, 3, 4> &Pose1,
                   const Eigen::Vector2d &point0, 
                   const Eigen::Vector2d &point1, 
                   Eigen::Vector3d &point_3d){
    Eigen::Matrix<double, 4, 4> design_matrix 
        = Eigen::Matrix<double, 4, 4>::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
  }
  
  void triangulate(const Eigen::Matrix<double, 3, 4>& pose_ref, 
                   const Eigen::Vector2d& point_ref, 
                   const std::vector<Eigen::Matrix<double, 3, 4>>& poses_obs,
                   const std::vector<Eigen::Vector2d>& points_obs, 
                   Eigen::Vector3d& point_3d){
    size_t num = poses_obs.size();
    CHECK(poses_obs.size() == points_obs.size());
    Eigen::MatrixXd  A(4 * num, 4);
    for(int i = 0; i < poses_obs.size(); ++i){
      const Eigen::Vector2d& point_obs = points_obs.at(i);
      const Eigen::Matrix<double, 3, 4>& pose_obs = poses_obs.at(i);
      A.row(4 * i) = point_ref[0] * pose_ref.row(2) - pose_ref.row(0);
      A.row(4 * i + 1) = point_ref[1] * pose_ref.row(2) - pose_ref.row(1);
      A.row(4 * i + 2) = point_obs[0] * pose_obs.row(2) - pose_obs.row(0);
      A.row(4 * i + 3) = point_obs[1] * pose_obs.row(2) - pose_obs.row(1);
    }
    
    Eigen::Vector4d point_4d =
        A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = point_4d(0) / point_4d(3);
    point_3d(1) = point_4d(1) / point_4d(3);
    point_3d(2) = point_4d(2) / point_4d(3);
  }

  void SetPriorCorrespondence();

  void AddStartTimePose(std::shared_ptr<TrajectoryEstimator<_N>> estimator);

  std::shared_ptr<TrajectoryN> trajectory_;

  CalibParamManager::Ptr calib_param_;

  std::shared_ptr<ImuStateEstimator> imu_state_estimator_;
  std::shared_ptr<ImuStateEstimator> imu_integrator_for_camera_;

  Eigen::aligned_vector<PointCorrespondence> point_prior_database_;
  Eigen::aligned_vector<IMUData> imu_data_;
  Eigen::aligned_vector<IMUData> cache_imu_data_;
  std::vector<std::pair<double, IMUBias>> cache_imu_bias_;

  double cor_min_time_, cor_max_time_;
  double cur_image_time_;

  bool trajectory_init_ = false;

  PoseData init_pose;

  bool use_corner_feature_ = false;
  bool use_imu_orientation_ = true;

  std::vector<ReferenceFeature> ref_features_;
  std::vector<ObservationFeature> obs_features_;
  
  size_t img_frame_nr_ = 0;
  size_t matching_window_size_ = 7;

  float v_depth_upper_bound_ = 150.;
  float v_depth_lower_bound_ = 1.;

  std::deque<std::shared_ptr<sfm::View>> views_ = {};
  std::set<std::shared_ptr<sfm::Landmark>> landmarks_ = {};
  std::set<std::shared_ptr<sfm::Landmark>> landmarks_in_window_ = {};

  struct RelativeCamMotion{
    std::shared_ptr<sfm::View> reference_view_;
    std::shared_ptr<sfm::View> observation_view_;
    Eigen::Matrix3d relative_rot_;
    Eigen::Vector3d relative_pos_;
  };
  std::deque<RelativeCamMotion> relative_cam_poses_; 
};

template <int _N>
void TrajectoryManager<_N>::SetPriorCorrespondence() {
  double prior_time = trajectory_->GetActiveTime();
  if (prior_time <= trajectory_->minTime()) {
    return;
  }

  // https://stackoverflow.com/questions/991335/
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < prior_time) {
      iter = imu_data_.erase(iter);
    } else {
      ++iter;
    }
  }
}

template <int _N>
void TrajectoryManager<_N>::AddStartTimePose(
    std::shared_ptr<TrajectoryEstimator<_N>> estimator) {
  size_t kont_idx = trajectory_->computeTIndex(cor_min_time_).second;
  if (kont_idx < _N) {
    init_pose.timestamp = trajectory_->minTime();

    double rot_weight = 100;
    double pos_weight = 100;
    estimator->AddPoseMeasurement(init_pose, rot_weight, pos_weight);
  }
}


template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolve(
    const std::vector<PoseData>& local_pose, int iteration) {

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;

  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for(const auto& lp: local_pose){
    estimator->AddPoseMeasurement(lp, rot_weight, pos_weight);
  }
  

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cor_max_time_) break;
    //    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  //  double lambda = 1.0;
  //  Eigen::Matrix3d a2_weight = lambda * Eigen::Matrix3d::Identity();
  //  estimator->AddQuadraticIntegralFactor(cor_max_time_,
  //  trajectory_->maxTime()-1e-9, a2_weight); std::cout << "motion weight: " <<
  //  lambda << std::endl;

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, 1);
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);

  calib_param_->CheckIMUBias();
  return true;
}

template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolveLI(
    const std::vector<PoseData>& local_pose, int iteration) {

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_imu_vel_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;

  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;

  double imu_rot_weight = calib_param_->global_opt_imu_rot_weight;
  double imu_pos_weight = calib_param_->global_opt_imu_pos_weight;


  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  //Add scan matching pose
  //TODO: input lidar pose rather than IMU pose to refine extrinsics online.
  for(const auto& lp: local_pose){
    estimator->AddPoseMeasurement(lp, rot_weight, pos_weight);
  }
  
  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cur_image_time_) break;
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }

  for (const auto& state : imu_state_estimator_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);
  
  calib_param_->CheckIMUBias();

  return true;
}

template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolveLVI(
    const std::vector<PoseData>& local_pose, int iteration) {

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_imu_vel_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;

  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;

  double imu_rot_weight = calib_param_->global_opt_imu_rot_weight;
  double imu_pos_weight = calib_param_->global_opt_imu_pos_weight;

  double cam_weight = calib_param_->global_opt_cam_uv_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  //Add scan matching pose
  //TODO: input lidar pose rather than IMU pose to refine extrinsics online.
  for(const auto& lp: local_pose){
    estimator->AddPoseMeasurement(lp, rot_weight, pos_weight);
  }
  
  //Add camera reprojection loss term
  size_t visual_term_num = 0;
  for(const auto& landmark: landmarks_){
    if(landmark->inverse_depth() < 1e-3) continue;
    if(landmark->observations().size() < 3) continue;
    const auto& ref = landmark->reference();
    // LOG(INFO)<<"landmark->observations() "<<landmark->observations().size();
    for(const auto& obs: landmark->observations()){
      if(obs == ref) continue;
      // LOG(INFO)<<"Obs-ref: "<<obs->view()->frame_nr() - ref->view()->frame_nr()
      // <<", "<<obs->view()->t0() - ref->view()->t0();
      estimator->AddCameraMeasurement(ref, obs, cam_weight,
          v_depth_upper_bound_, v_depth_lower_bound_);
      visual_term_num++;
    }
  }

  size_t num_view_node = 0;

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cur_image_time_) break;
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cur_image_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }
  
  for (const auto& state : imu_state_estimator_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
    /* PoseData pose;
    pose.timestamp = state.timestamp;
    pose.position = state.p;
    pose.orientation = SO3d(state.q);
    estimator->AddPoseMeasurement(pose, imu_rot_weight, imu_pos_weight); */
  }

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
    /* PoseData pose;
    pose.timestamp = state.timestamp;
    pose.position = state.p;
    pose.orientation = SO3d(state.q);
    estimator->AddPoseMeasurement(pose, imu_rot_weight, imu_pos_weight); */
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);
  
  // 检验重投影误差, 多数在0.1以内，少数outlier剔除
  for(int i = views_.size() - 1; i >= 0 
      && i >= views_.size() - matching_window_size_; --i){
    if(views_[i]->t0() < trajectory_->GetForcedFixedTime()) break;
    SE3d pose_ref_se3, pose_obs_se3;
    double t_obs = views_[i]->t0();
    for(auto obs: views_[i]->observations()){
      double t_ref = obs->landmark()->reference()->view()->t0();
      if(!trajectory_->GetCameraPose(t_ref, pose_ref_se3)) continue;  
      if(!trajectory_->GetCameraPose(t_obs, pose_obs_se3)) continue;  
      
      Eigen::Vector3d point_3d, point_3d_ref, point_3d_obs;
      point_3d_ref << obs->landmark()->reference()->uv()[0],
                      obs->landmark()->reference()->uv()[1], 1.0;
      
      point_3d_ref /= obs->landmark()->inverse_depth();
      
      point_3d = pose_ref_se3 * point_3d_ref;
      point_3d_obs = pose_obs_se3.inverse() * point_3d;
      Eigen::Vector2d pt_ref_2d, pt_obs_2d;
      
      pt_obs_2d << point_3d_obs[0] / point_3d_obs[2], 
                  point_3d_obs[1] / point_3d_obs[2];
      auto obs_err = (pt_obs_2d - obs->uv()).transpose();
      
      if(calib_param_->enable_debug){
        LOG(INFO) <<"obs_err: "<<obs_err;
      }

      float current_depth = 1. / obs->landmark()->inverse_depth();
      if(std::abs(obs_err[0]) / current_depth > 0.2 || 
         std::abs(obs_err[1]) / current_depth > 0.2){
          views_[i]->RemoveObservation(obs);
      }
    }
  }
  
  if(calib_param_->enable_debug){
    LOG(INFO) <<"visual_term_num: "<<visual_term_num;
    LOG(INFO) <<"estimated gyro bias: "<<calib_param_->gyro_bias.transpose();
    LOG(INFO) <<"estimated acce bias: "<<calib_param_->acce_bias.transpose();
  }
  
  calib_param_->CheckIMUBias();

  // Lock landmarks which have enough observations after optimization.
  // for(auto& lm: landmarks_){
  //   if(lm->observations().size() >= 3){
  //     lm->Lock(true);
  //   }
  // }

  return true;
}

template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolveVI(int iteration) {

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_imu_vel_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;
  double imu_rot_weight = calib_param_->global_opt_imu_rot_weight;
  double imu_pos_weight = calib_param_->global_opt_imu_pos_weight;

  double cam_weight = calib_param_->global_opt_cam_uv_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
   
   //Add camera reprojection loss term
  size_t visual_term_num = 0;

  for(const auto& landmark: landmarks_){
    if(landmark->inverse_depth() < 1e-3) continue;
    if(landmark->observations().size() < 3) continue;
    const auto& ref = landmark->reference();
    // LOG(INFO)<<"landmark->observations() "<<landmark->observations().size();
    for(const auto& obs: landmark->observations()){
      if(obs == ref) continue;
      // LOG(INFO)<<"Obs-ref: "<<obs->view()->frame_nr() - ref->view()->frame_nr()
      // <<", "<<obs->view()->t0() - ref->view()->t0();
      estimator->AddCameraMeasurement(ref, obs, cam_weight,
          v_depth_upper_bound_, v_depth_lower_bound_);
      visual_term_num++;
    }
  }
  
  

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cur_image_time_) break;
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cur_image_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }
  
  for (const auto& state : imu_integrator_for_camera_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
    PoseData pose;
    pose.timestamp = state.timestamp;
    pose.position = state.p;
    pose.orientation = SO3d(state.q);
    estimator->AddPoseMeasurement(pose, imu_rot_weight, imu_pos_weight);
  }

  for (const auto& state : imu_integrator_for_camera_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
    PoseData pose;
    pose.timestamp = state.timestamp;
    pose.position = state.p;
    pose.orientation = SO3d(state.q);
    estimator->AddPoseMeasurement(pose, imu_rot_weight, imu_pos_weight);
  }

  AddStartTimePose(estimator);
  
  // LOG(INFO)<<imu_data_.size()<<","<<imu_integrator_for_camera_->GetIntegrateState().size()<<","<<imu_integrator_for_camera_->GetVirtualIMUState().size();
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);
  
  // 检验重投影误差, 多数在0.1以内，少数outlier剔除
  int outlier = 0;
  for(int i = views_.size() - 1; i >= 0 
      && i >= views_.size() - matching_window_size_; --i){
    // if(views_[i]->t0() < trajectory_->GetForcedFixedTime()) break;
    SE3d pose_ref_se3, pose_obs_se3;
    double t_obs = views_[i]->t0();
    for(auto obs: views_[i]->observations()){
      double t_ref = obs->landmark()->reference()->view()->t0();
      if(!trajectory_->GetCameraPose(t_ref, pose_ref_se3)) continue;  
      if(!trajectory_->GetCameraPose(t_obs, pose_obs_se3)) continue;  
      
      Eigen::Vector3d point_3d, point_3d_ref, point_3d_obs;
      point_3d_ref << obs->landmark()->reference()->uv()[0],
                      obs->landmark()->reference()->uv()[1], 1.0;
      
      point_3d_ref /= obs->landmark()->inverse_depth();
      
      point_3d = pose_ref_se3 * point_3d_ref;
      point_3d_obs = pose_obs_se3.inverse() * point_3d;
      Eigen::Vector2d pt_ref_2d, pt_obs_2d;
      
      pt_obs_2d << point_3d_obs[0] / point_3d_obs[2], 
                  point_3d_obs[1] / point_3d_obs[2];
      auto obs_err = (pt_obs_2d - obs->uv()).transpose();
      
      // if(calib_param_->enable_debug){
      //   LOG(INFO)<<"obs_err "<<obs_err;
      // }
      float current_depth = 1. / obs->landmark()->inverse_depth();
      if(std::abs(obs_err[0]) / current_depth > 0.3 || 
          std::abs(obs_err[1]) / current_depth > 0.3){
        views_[i]->RemoveObservation(obs);
        outlier++;
      }
      if(std::abs(obs_err[0]) / current_depth < 0.05 && 
          std::abs(obs_err[1]) / current_depth < 0.05){
        obs->landmark()->Lock(true);
      }
    }
  }
  
  
  if(calib_param_->enable_debug){
    LOG(INFO)<<"Added and outlier: "<<visual_term_num<<", "<<outlier;
    LOG(INFO) <<"estimated gyro bias: "<<calib_param_->gyro_bias.transpose();
    LOG(INFO) <<"estimated acce bias: "<<calib_param_->acce_bias.transpose();
  }
  
  calib_param_->CheckIMUBias();

  return true;
}


template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolve(
    const Eigen::aligned_vector<PointCorrespondence>& point_measurement,
    int iteration) {
  if (point_measurement.empty() || imu_data_.empty()) {
    std::cout << "[BuildProblemAndSolve] input empty data "
              << point_measurement.size() << ", " << imu_data_.size()
              << std::endl;
    return false;
  }

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);


  double feature_weight = calib_param_->global_opt_lidar_weight;
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for (const auto& v : point_measurement) {
    estimator->AddLoamMeasurement(v, feature_weight);
  }

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cor_max_time_) break;
    //    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  //  double lambda = 1.0;
  //  Eigen::Matrix3d a2_weight = lambda * Eigen::Matrix3d::Identity();
  //  estimator->AddQuadraticIntegralFactor(cor_max_time_,
  //  trajectory_->maxTime()-1e-9, a2_weight); std::cout << "motion weight: " <<
  //  lambda << std::endl;

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, 1);
  }

  AddStartTimePose(estimator);


  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);

  calib_param_->CheckIMUBias();
  return true;
}

template <int _N>
void TrajectoryManager<_N>::UpdateTrajectoryProperty() {
  trajectory_->UpdateActiveTime(cor_max_time_);
  trajectory_->SetForcedFixedTime(cor_min_time_ - 0.1);

  // TrajectoryViewer::PublishSplineTrajectory<4>(
  //     trajectory_, trajectory_->minTime(), cor_max_time_, 0.02);

  cache_imu_bias_.push_back(
      std::make_pair(cor_min_time_, calib_param_->GetIMUBias()));
}

template <int _N>
void TrajectoryManager<_N>::IntegrateIMUForCamera(double image_time) {
  cur_image_time_ = image_time;
  //更新上一推算时刻的状态为优化后的状态
  if(trajectory_->GetTrajQuality(
      imu_integrator_for_camera_->GetLatestTimestamp())){
    imu_integrator_for_camera_->GetLatestIMUState<_N>(trajectory_);
  }
  //推算到当前图像的时刻
  imu_integrator_for_camera_->Propagate(image_time);

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(cur_image_time_, last_kont);

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_imu_vel_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for (const auto& state : imu_integrator_for_camera_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  for (const auto& v : imu_data_) {
    if (v.timestamp >= cur_image_time_) break;
    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
  }

  //这里用了IMU的绝对orientation测量值,６轴IMU不具备
  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  imu_integrator_for_camera_->Predict(trajectory_->maxTime() - 1e-9, 0.01);

  for (const auto& state : imu_integrator_for_camera_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);
}

template <int _N>
void TrajectoryManager<_N>::IntegrateIMUMeasurement(double scan_time_min,
                                                    double scan_time_max) {
  if (imu_data_.empty()) {
    std::cout << "[IntegrateIMUMeasurement] IMU data empty! " << std::endl;
    return;
  }

  cor_min_time_ = scan_time_min;
  cor_max_time_ = scan_time_max;

  //推算上一帧点云进入时所推算到的时刻IMU在世界系下的状态
  imu_state_estimator_->GetLatestIMUState<_N>(trajectory_);
  //推算到当前点云的时刻
  imu_state_estimator_->Propagate(scan_time_max);

  SetPriorCorrespondence();//删除activetime之前的IMU数据, this->imu_data_

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(scan_time_max, last_kont);

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_imu_vel_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for (const auto& state : imu_state_estimator_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  for (const auto& v : imu_data_) {
    if (v.timestamp >= scan_time_max) break;
    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
  }

  //这里用了IMU的绝对orientation测量值,６轴IMU不具备
  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  //  double lambda = 1.0;
  //  Eigen::Matrix3d a2_weight = lambda * Eigen::Matrix3d::Identity();
  //  estimator->AddQuadraticIntegralFactor(
  //      scan_time_max, trajectory_->maxTime() - 1e-9, a2_weight);

  imu_state_estimator_->Predict(trajectory_->maxTime() - 1e-9, 0.01);

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);

  //  TrajectoryViewer::PublishIMUData<4>(trajectory_, imu_data_);
  // TrajectoryViewer::PublishSplineTrajectory<4>(
  //     trajectory_, trajectory_->minTime(), scan_time_max, 0.02);
}

template <int _N>
void TrajectoryManager<_N>::EstimateIMUBias() {
  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  estimator->LockTrajectory(true);
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;

  for (const auto& v : imu_data_) {
    if (v.timestamp >= cor_max_time_) break;
    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
  }
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true);

  ceres::Solver::Summary summary = estimator->Solve(50, true);
  //  calib_param_->ShowIMUBias();
}

template <int _N>
void TrajectoryManager<_N>::OptimiseLoopClosure(
    const LoopClosureOptimizationParam& param,
    const LoopClosureWeights& weights) {
  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  estimator->LockTrajectory(false);
  estimator->SetKeyScanConstant(param.history_key_frame_max_time);

  for (const auto& v : param.velocity_constraint) {
    estimator->AddVelocityConstraintMeasurement(v, weights.velocity_weight,
                                                weights.gyro_weight);
  }

  for (const auto& p : param.pose_graph_key_pose) {
    estimator->AddLiDARPoseMeasurement(p, weights.pose_graph_edge_rot_weight,
                                       weights.pose_graph_edge_pos_weight);
  }

  estimator->LockExtrinsicParam(true, true);

  ceres::Solver::Summary summary = estimator->Solve(100, true);

  // TrajectoryViewer::PublishSplineTrajectory<4>(
  //     trajectory_, trajectory_->minTime(), trajectory_->maxTime(), 0.02);
}

}  // namespace clins

#endif
