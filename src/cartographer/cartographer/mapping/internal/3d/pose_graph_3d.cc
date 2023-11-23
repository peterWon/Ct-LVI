/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/3d/pose_graph_3d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "cartographer/common/make_unique.h"
#include "cartographer/common/math.h"
#include "cartographer/transform/transform.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"

#include "glog/logging.h"


#include "cartographer/mapping/clins/camera_models/CameraFactory.h"
#include "cartographer/mapping/clins/odometry/camera_state_estimator.h"


#include "cartographer/mapping/cvins/initial/initial_sfm.h"
// #include "cartographer/mapping/cvins/initial/solve_5pts.h"
#include "cartographer/mapping/clins/utils/tic_toc.h"


#include <chrono>   
using namespace std;
using namespace chrono;

namespace cartographer {
namespace mapping {

PoseGraph3D::PoseGraph3D(
    const proto::PoseGraphOptions& options,
    std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem,
    common::ThreadPool* thread_pool)
    : options_(options),
      optimization_problem_(std::move(optimization_problem)),
      constraint_builder_(options_.constraint_builder_options(), thread_pool) {
  optimization_task_ = common::make_unique<common::Task>();

  // TODO: move to optition params.
  voc_.reset(new DBoW3::Vocabulary(options_.dbow_voc_filename()));
  db_.reset(new DBoW3::Database(*voc_, false, 0));  
  fdetector_ = cv::ORB::create();
  camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
    options_.camera_intrinsics_filename());
  
  // depth_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  // depth_cloud_ds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  
  pointsArray.resize(num_bins_);
  for(int i = 0; i < num_bins_; ++i)
    pointsArray[i].resize(num_bins_);
}

PoseGraph3D::~PoseGraph3D() {
  WaitForAllComputations();
  // common::MutexLocker locker(&mutex_);
  // CHECK(work_queue_ == nullptr);
}

std::vector<SubmapId> PoseGraph3D::InitializeGlobalSubmapPoses(
    const int trajectory_id, const common::Time time,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  CHECK(!insertion_submaps.empty());
  const auto& submap_data = optimization_problem_->submap_data();
  if (insertion_submaps.size() == 1) {
    // If we don't already have an entry for the first submap, add one.
    if (submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
      if (initial_trajectory_poses_.count(trajectory_id) > 0) {
        trajectory_connectivity_state_.Connect(
            trajectory_id,
            initial_trajectory_poses_.at(trajectory_id).to_trajectory_id, time);
      }
      optimization_problem_->AddSubmap(
          trajectory_id,
          ComputeLocalToGlobalTransform(global_submap_poses_, trajectory_id) *
              insertion_submaps[0]->local_pose());
    }
    CHECK_EQ(1, submap_data.SizeOfTrajectoryOrZero(trajectory_id));
    const SubmapId submap_id{trajectory_id, 0};
    CHECK(submap_data_.at(submap_id).submap == insertion_submaps.front());
    return {submap_id};
  }
  CHECK_EQ(2, insertion_submaps.size());
  const auto end_it = submap_data.EndOfTrajectory(trajectory_id);
  CHECK(submap_data.BeginOfTrajectory(trajectory_id) != end_it);
  const SubmapId last_submap_id = std::prev(end_it)->id;
  if (submap_data_.at(last_submap_id).submap == insertion_submaps.front()) {
    // In this case, 'last_submap_id' is the ID of 'insertions_submaps.front()'
    // and 'insertions_submaps.back()' is new.
    const auto& first_submap_pose = submap_data.at(last_submap_id).global_pose;
    optimization_problem_->AddSubmap(
        trajectory_id, first_submap_pose *
                           insertion_submaps[0]->local_pose().inverse() *
                           insertion_submaps[1]->local_pose());
    return {last_submap_id,
            SubmapId{trajectory_id, last_submap_id.submap_index + 1}};
  }
  CHECK(submap_data_.at(last_submap_id).submap == insertion_submaps.back());
  const SubmapId front_submap_id{trajectory_id,
                                 last_submap_id.submap_index - 1};
  CHECK(submap_data_.at(front_submap_id).submap == insertion_submaps.front());
  return {front_submap_id, last_submap_id};
}

NodeId PoseGraph3D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  const transform::Rigid3d optimized_pose(
      GetLocalToGlobalTransform(trajectory_id) * constant_data->local_pose);

  common::MutexLocker locker(&mutex_);
  AddTrajectoryIfNeeded(trajectory_id);
  const NodeId node_id = trajectory_nodes_.Append(
      trajectory_id, TrajectoryNode{constant_data, optimized_pose});
  ++num_trajectory_nodes_;

  // Test if the 'insertion_submap.back()' is one we never saw before.
  if (submap_data_.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
      std::prev(submap_data_.EndOfTrajectory(trajectory_id))->data.submap !=
          insertion_submaps.back()) {
    // We grow 'submap_data_' as needed. This code assumes that the first
    // time we see a new submap is as 'insertion_submaps.back()'.
    const SubmapId submap_id =
        submap_data_.Append(trajectory_id, InternalSubmapData());
    submap_data_.at(submap_id).submap = insertion_submaps.back();
  }

  // We have to check this here, because it might have changed by the time we
  // execute the lambda.
  const bool newly_finished_submap = insertion_submaps.front()->finished();
  AddWorkItem([=]() REQUIRES(mutex_) {
    ComputeConstraintsForNode(node_id, insertion_submaps,
                              newly_finished_submap);
  });
  return node_id;
}

void PoseGraph3D::AddWorkItem(const std::function<void()>& work_item) {
  if (work_queue_ == nullptr) {
    work_item();
  } else {
    work_queue_->push_back(work_item);
  }
}

void PoseGraph3D::AddTrajectoryIfNeeded(const int trajectory_id) {
  trajectory_connectivity_state_.Add(trajectory_id);
  // Make sure we have a sampler for this trajectory.
  if (!global_localization_samplers_[trajectory_id]) {
    global_localization_samplers_[trajectory_id] =
        common::make_unique<common::FixedRatioSampler>(
            options_.global_sampling_ratio());
  }
}

void PoseGraph3D::AddImuData(const int trajectory_id,
                             const sensor::ImuData& imu_data) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([=]() REQUIRES(mutex_) {
    optimization_problem_->AddImuData(trajectory_id, imu_data);
  });
}

void PoseGraph3D::ComputeConstraintsForImage(
      int trajectory_id, const sensor::ImageFeatureData& img_data,
      const std::shared_ptr<Submap>& matching_submap){
  TicToc tic_toc;
  if(trajectory_nodes_.empty()) return;
  const transform::Rigid3d& local_pose = img_data.pose_in_local;
  bool add_loop_kf = false;
  bool add_locate_kf = false;
  if(!img_keyframes_.empty()){
    // 两类关键帧，时空粗分辨率的用于回环，时空高分辨率（但比原始分辨率低）的用于回环定位
    const double rad2deg = 180. / M_PI;
    double pos_thresh_coarse = options_.loop_position_threshold_coarse(); 
    double rot_thresh_coarse = 
            options_.loop_rotation_threshold_coarse()*rad2deg;
    double pos_thresh_fine = options_.loop_position_threshold_fine(); 
    double rot_thresh_fine = 
            options_.loop_rotation_threshold_fine() * rad2deg;

    const transform::Rigid3d last_kf_pose = img_keyframes_.back().local_pose_;
    Eigen::Vector3d pos_diff = 
          local_pose.translation() - last_kf_pose.translation();
    Eigen::Vector3d rot_diff = (last_kf_pose.rotation().conjugate() * 
      local_pose.rotation()).toRotationMatrix().eulerAngles(2,1,0) * rad2deg;
    if(pos_diff.norm() > pos_thresh_coarse 
          || std::abs(rot_diff[0]) > rot_thresh_coarse
          || std::abs(rot_diff[1]) > rot_thresh_coarse
          || std::abs(rot_diff[2]) > rot_thresh_coarse){
      add_loop_kf = true;
    }
    if(pos_diff.norm() > pos_thresh_fine 
          || std::abs(rot_diff[0]) > rot_thresh_fine
          || std::abs(rot_diff[1]) > rot_thresh_fine
          || std::abs(rot_diff[2]) > rot_thresh_fine){
      add_locate_kf = true;
    }
  }else{
    add_loop_kf = add_locate_kf = true;
  }
  
  if(!add_locate_kf) return;

  ImageKeyframe kf;
  kf.time_ = img_data.time;
  kf.id_ = cur_image_kf_id_++;
  kf.local_pose_ = local_pose;
  kf.keypoints_.clear();
  // img_data.img.copyTo(kf.image_);
  /* for(int i = 0; i < img_data.features_uv.size(); i++){
    cv::KeyPoint kp;
    kp.pt.x = img_data.features_uv[i].x;
    kp.pt.y = img_data.features_uv[i].y;
    kf.keypoints_.emplace_back(kp);
  } */
  // fdetector_->compute(img_data.img, kf.keypoints_, kf.descriptors_);
  fdetector_->detectAndCompute(
      img_data.img, cv::Mat(), kf.keypoints_, kf.descriptors_);

  kf.submap_id_ = FindSubmapId(matching_submap);
  if(kf.submap_id_.submap_index == -1) return;
  
  img_keyframes_.push_back(kf);

  DBoW3::QueryResults ret;
  db_->query(kf.descriptors_, ret, 4, loop_kf_id_ - 10);
  // after quering, update the database
  if(add_loop_kf){
    loop_kf_index_.push_back(img_keyframes_.size() - 1);
    db_->add(kf.descriptors_);
    loop_kf_id_++;
  }
  
  if(ret.empty()) return;

  bool find_loop = false;
  if(ret.size() >= 1 && ret[0].Score > 0.05){
    for (size_t i = 1; i < ret.size(); ++i){
      if (ret[i].Score > 0.015){          
        find_loop = true;
        int tmp_index = ret[i].Id;
      }
    }
  }  
  
  // loop closure detected.
  if(find_loop){
    int min_index = -1;
    for(unsigned int i = 0; i < ret.size(); i++){
      if(min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
        min_index = ret[i].Id;
    }
    
    if(min_index == -1) return;

    int loop_index = loop_kf_index_[min_index];
    const auto& loop_frame = img_keyframes_[loop_index];
    
    // skip adjacent submap.
    if(kf.submap_id_.trajectory_id == loop_frame.submap_id_.trajectory_id){
      int id_dst = kf.submap_id_.submap_index 
                    - loop_frame.submap_id_.submap_index;
      if(std::abs(id_dst) < 2) return;
    } 
    
    LOG(WARNING)<<"Visual loop detected between submap "<<kf.submap_id_.submap_index<<" and submap "<<loop_frame.submap_id_.submap_index;
    
    // LOG(WARNING)<<"submap_id.submap_index: " << submap_id.submap_index;
    const transform::Rigid3d global_submap_pose =
        optimization_problem_->submap_data().at(loop_frame.submap_id_).global_pose;
    const transform::Rigid3d local_submap_pose =
        submap_data_.at(loop_frame.submap_id_).submap->local_pose();
    const transform::Rigid3d global_submap_pose_inverse =
        global_submap_pose.inverse();
    std::vector<std::pair<NodeId, TrajectoryNode> > submap_nodes;
    for (const NodeId& submap_node_id : submap_data_.at(loop_frame.submap_id_).node_ids) {  
      submap_nodes.push_back({submap_node_id,
          TrajectoryNode{trajectory_nodes_.at(submap_node_id).constant_data,
                        /* global_submap_pose_inverse *
                        trajectory_nodes_.at(submap_node_id).global_pose */
                local_submap_pose.inverse() 
                * trajectory_nodes_.at(submap_node_id).constant_data->local_pose}
      });
    }
    constraint_builder_.DispatchScanMatcherConstruction(
      loop_frame.submap_id_, local_submap_pose, submap_nodes, 
      submap_data_.at(loop_frame.submap_id_).submap.get());
    
    
    const transform::Rigid3d global_kf_submap_pose =
        optimization_problem_->submap_data().at(kf.submap_id_).global_pose;
    const transform::Rigid3d local_kf_submap_pose =
        submap_data_.at(kf.submap_id_).submap->local_pose();
    const transform::Rigid3d global_kf_submap_pose_inverse =
        global_kf_submap_pose.inverse();
    std::vector<std::pair<NodeId, TrajectoryNode> > kf_submap_nodes;
    for (const NodeId& submap_node_id : submap_data_.at(kf.submap_id_).node_ids) {  
      kf_submap_nodes.push_back({submap_node_id,
          TrajectoryNode{trajectory_nodes_.at(submap_node_id).constant_data,
                        /* global_submap_pose_inverse *
                        trajectory_nodes_.at(submap_node_id).global_pose */
                local_kf_submap_pose.inverse() 
                * trajectory_nodes_.at(submap_node_id).constant_data->local_pose}
      });
    }
    constraint_builder_.DispatchScanMatcherConstruction(
      kf.submap_id_, local_kf_submap_pose, kf_submap_nodes, 
      submap_data_.at(kf.submap_id_).submap.get());
    // 根据局部窗口，三角化特征点，PnP反定位回环帧, BA优化回环位姿
    // 1) 建立局部窗口内的匹配关系
    /*CHECK(kf.descriptors_.rows == kf.keypoints_.size());
    
    cv::BFMatcher matcher(NORM_HAMMING);
    const int query_size = kf.descriptors_.rows;
    const int half_winsize = 4;
    const int window_size = half_winsize * 2 + 1;
    
    // bundle ajustment variables
    // why possessing local_parameterization as a class member 
    // would crash the bundle adjustement?
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = 
        new ceres::QuaternionParameterization();
    
    Eigen::Matrix3d c_Rotation[window_size + 1];
    Eigen::Vector3d c_Translation[window_size + 1];
    Eigen::Quaterniond c_Quat[window_size + 1];
    double c_rotation[window_size + 1][4];
    double c_translation[window_size + 1][3];

    // i: queryId; k: index in img_keyframes_; j: trainId
    // corres: current keypoints->reference key frames->ikj
    Eigen::Vector3i ikj;
    std::vector<std::vector<Eigen::Vector3i>> corres;
    corres.resize(query_size, {});
    
    for(int index = -half_winsize, i = 0; 
        index < half_winsize; ++index, ++i){
      int k = index + loop_index;
      if(k < 0 || k >= img_keyframes_.size()) return; // no enough frame
      
      std::vector<cv::DMatch> matches;
      matcher.match(kf.descriptors_, 
                    img_keyframes_[k].descriptors_, matches);
      double min_dist = 10000, max_dist = 0;
      for(int j = 0; j < kf.descriptors_.rows; ++j){
        double dist = matches[j].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
      }
      
      std::vector<cv::DMatch> good_matches;
      for(int j = 0; j < kf.descriptors_.rows; ++j){
        if(matches[j].distance <= max(2 * min_dist, 30.0)){
          CHECK(j == matches[j].queryIdx);
          corres[j].push_back(
            Eigen::Vector3i(matches[j].queryIdx, k, matches[j].trainIdx));
          good_matches.push_back(matches[j]);
        }
      }

      // filling double array for ceres
      const auto& pose_in_cam = img_keyframes_[k].local_pose_.inverse();
      c_Quat[i] = pose_in_cam.rotation();
      c_Rotation[i] = pose_in_cam.rotation().toRotationMatrix();
      c_Translation[i] = pose_in_cam.translation();
      
      c_translation[i][0] = c_Translation[i].x();
      c_translation[i][1] = c_Translation[i].y();
      c_translation[i][2] = c_Translation[i].z();
      c_rotation[i][0] = c_Quat[i].w();
      c_rotation[i][1] = c_Quat[i].x();
      c_rotation[i][2] = c_Quat[i].y();
      c_rotation[i][3] = c_Quat[i].z();
      problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
      problem.AddParameterBlock(c_translation[i], 3);
      if (index == 0){
        problem.SetParameterBlockConstant(c_rotation[i]);
      }
      if (i == 0 || index == 0 || i == window_size - 1){
        problem.SetParameterBlockConstant(c_translation[i]);
      }
      // draw correspondence
      // cv::Mat img_goodmatch;
      // cv::drawMatches(kf.image_, kf.keypoints_, img_keyframes_[k].image_,
      //   img_keyframes_[k].keypoints_, good_matches, img_goodmatch);
      // std::string filename = "/home/wz/Desktop/match_loop/"+std::to_string(kf.id_)+"-"+std::to_string(k)+".jpg";
      // cv::imwrite(filename, img_goodmatch);
    }

    // add current frame to the problem.
    problem.AddParameterBlock(
        c_rotation[window_size], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[window_size], 3);

    // 2) find covisible points and perform triangulation.
    std::vector<cv::Point3f> pts_3_vec;
    std::vector<cv::Point2f> pts_2_vec;
    int cf_idx = 0;
    for(int i = 0; i < query_size; ++i){
      if(corres[i].size() < 3) continue;
      cv::Point2d pt_i, pt_j;
      Eigen::Vector3d un_pt_i, un_pt_j;
      
      Eigen::MatrixXd A(2 * corres[i].size(), 4);
      
      int ref_frameId = corres[i][0][1];
      int ref_trainId = corres[i][0][2];
      pt_i = img_keyframes_[ref_frameId].keypoints_[ref_trainId].pt;
      camera_->liftProjective(Eigen::Vector2d(pt_i.x, pt_i.y), un_pt_i);
      double di = un_pt_i[2];
      un_pt_i /= di;
      
      const auto& pose_ref = img_keyframes_[ref_frameId].local_pose_;
      Eigen::Vector3d t0 = pose_ref.translation();
      Eigen::Matrix3d R0 = Eigen::Matrix3d(pose_ref.rotation());

      for(int j = 0; j < corres[i].size(); ++j){
        int frameIdx = corres[i][j][1];
        int trainIdx = corres[i][j][2];
        pt_j = img_keyframes_[frameIdx].keypoints_[trainIdx].pt;
        camera_->liftProjective(Eigen::Vector2d(pt_j.x, pt_j.y), un_pt_j);
        double dj = un_pt_j[2];
        un_pt_j /= dj;

        const auto& pose_j = img_keyframes_[frameIdx].local_pose_;
        Eigen::Matrix<double, 3, 4> P1;
        Eigen::Vector3d t1 = pose_j.translation();
        Eigen::Matrix3d R1 = Eigen::Matrix3d(pose_j.rotation());
        
        // camera to world
        Eigen::Vector3d t = R0.transpose() * (t1 - t0);
        Eigen::Matrix3d R = R0.transpose() * R1;

        // world to camera
        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;
        
        Eigen::Vector3d f = un_pt_j.normalized();
        A.row(2 * j) = f[0] * P.row(2) - f[2] * P.row(0);
        A.row(2 * j + 1) = f[1] * P.row(2) - f[2] * P.row(1);
      }
      Eigen::Vector4d point_4d =
          A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
      Eigen::Vector3d point_3d, pt_3d_w;
      point_3d(0) = point_4d(0) / point_4d(3);
      point_3d(1) = point_4d(1) / point_4d(3);
      point_3d(2) = point_4d(2) / point_4d(3);
      
      if(point_3d[2] < 0.) continue;

      cv::Point2d pt_query;
      Eigen::Vector3d un_pt_query;
      int query_id = corres[i][0][0];
      pt_query = kf.keypoints_[query_id].pt;
      camera_->liftProjective(
          Eigen::Vector2d(pt_query.x, pt_query.y), un_pt_query);
      double d_query = un_pt_query[2];
      un_pt_query /= d_query;
      
      pt_3d_w = pose_ref * point_3d;
      pts_3_vec.emplace_back(cv::Point3d(pt_3d_w[0], pt_3d_w[1], pt_3d_w[2]));
      pts_2_vec.emplace_back(cv::Point2d(un_pt_query[0], un_pt_query[1]));
      
      // set initial guess for the features
      ceres_features_[cf_idx][0] = pt_3d_w[0];
      ceres_features_[cf_idx][1] = pt_3d_w[1];
      ceres_features_[cf_idx][2] = pt_3d_w[2];
      // add factors
      for(int j = 0; j < corres[i].size(); ++j){
        int frameIdx = corres[i][j][1];
        int trainIdx = corres[i][j][2];
        pt_j = img_keyframes_[frameIdx].keypoints_[trainIdx].pt;
        camera_->liftProjective(Eigen::Vector2d(pt_j.x, pt_j.y), un_pt_j);
        double dj = un_pt_j[2];
        un_pt_j /= dj;

        ceres::CostFunction* cost_function = 
            cvins::ReprojectionError3D::Create(un_pt_j.x(), un_pt_j.y());
        
        int index_in_window = frameIdx - loop_index + half_winsize;
        problem.AddResidualBlock(cost_function, NULL, 
          c_rotation[index_in_window], c_translation[index_in_window],
          ceres_features_[cf_idx]);	 
      }
      ceres::CostFunction* cost_function_query = cvins::ReprojectionError3D
            ::Create(un_pt_query.x(), un_pt_query.y());
      problem.AddResidualBlock(cost_function_query, NULL, 
            c_rotation[window_size], c_translation[window_size], 
            ceres_features_[cf_idx]);
      cf_idx++;
    }
      
    

    // 3) 3d->2d PnP     
    cv::Mat r, rvec, t, D, tmp_r, inliers;
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial(0,0,0);
    R_initial.setIdentity();
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    
    

    // without initial guess, the last parameter should be false.
    bool pnp_succ = false;
    if(pts_2_vec.size() > 10){
      pnp_succ = cv::solvePnP(pts_3_vec, pts_2_vec, K, D, rvec, t, false);
    }
    if(!pnp_succ){
      if(!options_.enable_so3_visual_loop_search()) return;
      // only attempt to construct so3 constraints by the two visited frames.
      std::vector<cv::DMatch> matches;
      matcher.match(kf.descriptors_, 
                    img_keyframes_[loop_index].descriptors_, matches);
      double min_dist = 10000, max_dist = 0;
      for(int j = 0; j < kf.descriptors_.rows; ++j){
        double dist = matches[j].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
      }
      
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> matched_pts;
      std::vector<cv::DMatch> good_matches;
      for(int j = 0; j < kf.descriptors_.rows; ++j){
        if(matches[j].distance <= max(2 * min_dist, 30.0)){
          cv::Point2f pi = kf.keypoints_.at(matches[j].queryIdx).pt;
          cv::Point2f pj = kf.keypoints_.at(matches[j].trainIdx).pt;
          Eigen::Vector3d pi_3d, pj_3d;
          camera_->liftProjective(Eigen::Vector2d(pi.x, pi.y), pi_3d);
          camera_->liftProjective(Eigen::Vector2d(pj.x, pj.y), pj_3d);
          double di = pi_3d[2];
          pi_3d /= di;  
          double dj = pj_3d[2];
          pj_3d /= dj;  
          // matched_pts.push_back({pi_3d.head<2>(), pj_3d.head<2>()});
          matched_pts.push_back({pj_3d.head<2>(), pi_3d.head<2>()});
        }
      }
      Eigen::Matrix3d rot;
      Eigen::Vector3d pos;
      if(clins::CameraMotionEstimator::SolveRelativeRT(matched_pts, rot, pos)){
        transform::Rigid3d kf_in_local(pos, Eigen::Quaterniond(rot));
        transform::Rigid3d diff_pnp = kf_in_local.inverse() * kf.local_pose_;
        LOG(INFO)<<"RT succeed.";
        LOG(INFO)<< "Rot diff "<< diff_pnp.rotation().toRotationMatrix()
                    .eulerAngles(2,1,0).transpose() * 180./ M_PI;
        LOG(INFO)<< "Rot Transpose diff "<< diff_pnp.rotation().conjugate().toRotationMatrix()
                    .eulerAngles(2,1,0).transpose() * 180./ M_PI;

        // add an so(3) constraint.
        constraints::SE3SubmapConstraint so3_loop;
        so3_loop.ref_submap = img_keyframes_[loop_index].submap_id_;
        const transform::Rigid3d global_ref_submap_pose 
                = optimization_problem_->submap_data().at(
                        so3_loop.ref_submap).global_pose;
        
        so3_loop.is_translation_usable = false;
        so3_loop.obs_submap = kf.submap_id_;
        transform::Rigid3d ref_pose = img_keyframes_[loop_index].local_pose_;
        so3_loop.T_refmap_in_global = global_ref_submap_pose;
        so3_loop.T_ref_in_refmap = submap_data_.at(
            so3_loop.ref_submap).submap->local_pose().inverse() * ref_pose;
        so3_loop.T_obs_in_obsmap = submap_data_.at(
            so3_loop.obs_submap).submap->local_pose().inverse() * local_pose; 

        transform::Rigid3d kf_in_local_est(pos, Eigen::Quaterniond(rot));
        so3_loop.T_obs_ref = ref_pose.inverse() * kf_in_local_est;

        so3_loop.nodes_in_submap.clear();
        const transform::Rigid3d local_obs_submap_pose 
            = submap_data_.at(so3_loop.obs_submap).submap->local_pose();
        for (const NodeId& node_id : submap_data_.at(kf.submap_id_).node_ids){  
          if(trajectory_nodes_.at(node_id).time() > kf.time_) break;
          so3_loop.nodes_in_submap.push_back({node_id, TrajectoryNode{
            trajectory_nodes_.at(node_id).constant_data,
            local_obs_submap_pose.inverse() * 
            trajectory_nodes_.at(node_id).constant_data->local_pose}
          });
        }
        constraint_builder_.DispatchScanMatcherConstruction(so3_loop);
        return;
      }
    }

    // test pnp results
    cv::Rodrigues(rvec, r);
    Eigen::Matrix3d r_pnp;
    cv::cv2eigen(r, r_pnp);
    Eigen::Vector3d t_pnp;
    cv::cv2eigen(t, t_pnp);

    transform::Rigid3d kf_in_local_pnp(t_pnp, Eigen::Quaterniond(r_pnp));
    transform::Rigid3d kf_in_cam = kf_in_local_pnp.inverse();
    
    // transform::Rigid3d diff_pnp = kf_in_local.inverse() * kf.local_pose_;
    // LOG(INFO)<<"PnP succeed.";
    // LOG(INFO)<< "Pos diff "<< diff_pnp.translation().transpose();
    // LOG(INFO)<< "Rot diff "<< diff_pnp.rotation().toRotationMatrix()
    //             .eulerAngles(2,1,0).transpose() * 180./ M_PI;

    // set initial guess of current frame.
    c_Quat[window_size] = kf_in_cam.rotation();
    c_Translation[window_size] = kf_in_cam.translation();
    c_translation[window_size][0] = c_Translation[window_size].x();
    c_translation[window_size][1] = c_Translation[window_size].y();
    c_translation[window_size][2] = c_Translation[window_size].z();
    c_rotation[window_size][0] = c_Quat[window_size].w();
    c_rotation[window_size][1] = c_Quat[window_size].x();
    c_rotation[window_size][2] = c_Quat[window_size].y();
    c_rotation[window_size][3] = c_Quat[window_size].z();
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type == ceres::CONVERGENCE 
      || summary.final_cost < 5e-03){
      
      Eigen::Quaterniond q_kf_in_local;
      Eigen::Vector3d t_kf_in_local;
      q_kf_in_local.w() = c_rotation[window_size][0]; 
      q_kf_in_local.x() = c_rotation[window_size][1]; 
      q_kf_in_local.y() = c_rotation[window_size][2]; 
      q_kf_in_local.z() = c_rotation[window_size][3]; 
      q_kf_in_local = q_kf_in_local.inverse();
      t_kf_in_local = -1 * (q_kf_in_local * Eigen::Vector3d(
                    c_translation[window_size][0], 
                    c_translation[window_size][1], 
                    c_translation[window_size][2]));
      transform::Rigid3d kf_in_local_opt(t_kf_in_local, q_kf_in_local);
      transform::Rigid3d diff = kf_in_local_opt.inverse() * kf.local_pose_;
    
      // find the corresponding submaps and perform submap-to-submap matching.
      constraints::SE3SubmapConstraint se3_loop;
      se3_loop.is_translation_usable = true;
      se3_loop.ref_submap = img_keyframes_[loop_index].submap_id_;
      se3_loop.obs_submap = kf.submap_id_;
      
      const transform::Rigid3d global_ref_submap_pose 
          = optimization_problem_->submap_data().at(
                        se3_loop.ref_submap).global_pose;

      // 相对运动关系（闭合差）从local系推算而得的.
      transform::Rigid3d ref_pose = img_keyframes_[loop_index].local_pose_;
      se3_loop.T_refmap_in_global = global_ref_submap_pose;
      se3_loop.T_ref_in_refmap = submap_data_.at(
          se3_loop.ref_submap).submap->local_pose().inverse() * ref_pose;
      se3_loop.T_obs_in_obsmap = submap_data_.at(
          se3_loop.obs_submap).submap->local_pose().inverse() * local_pose; 
      se3_loop.T_obs_ref = ref_pose.inverse() * kf_in_local_opt;

      se3_loop.nodes_in_submap.clear();
      
      const transform::Rigid3d local_obs_submap_pose 
          = submap_data_.at(se3_loop.obs_submap).submap->local_pose();
      for (const NodeId& node_id : submap_data_.at(kf.submap_id_).node_ids){  
        if(trajectory_nodes_.at(node_id).time() > kf.time_) break;
        se3_loop.nodes_in_submap.push_back({node_id, TrajectoryNode{
          trajectory_nodes_.at(node_id).constant_data,
          local_obs_submap_pose.inverse() * 
          trajectory_nodes_.at(node_id).constant_data->local_pose}
        });
      }
      constraint_builder_.DispatchScanMatcherConstruction(se3_loop);
      LOG(INFO)<<"Add visual constraint cost: "<<tic_toc.toc();
      LOG(INFO)<<"Bundle adjustment succeed!";
      LOG(INFO)<< "Pos diff "<< diff.translation().transpose();
      LOG(INFO)<< "Rot diff "<< diff.rotation().toRotationMatrix()
                  .eulerAngles(2,1,0).transpose() * 180./ M_PI;
      LOG(INFO)<<"Dispatch "<<se3_loop.nodes_in_submap.size()
                <<" visual work items of "
                <<submap_data_.at(kf.submap_id_).node_ids.size() << " nodes.";
    }else{
      if(!options_.enable_so3_visual_loop_search()) return;
    
      transform::Rigid3d diff_pnp = kf_in_local_pnp.inverse() * kf.local_pose_;
      LOG(INFO)<<"PnP succeed.";
      LOG(INFO)<< "Rot diff "<< diff_pnp.rotation().toRotationMatrix()
                  .eulerAngles(2,1,0).transpose() * 180./ M_PI;
      // add so3 loop
      constraints::SE3SubmapConstraint so3_loop;
      so3_loop.ref_submap = img_keyframes_[loop_index].submap_id_;
      const transform::Rigid3d global_ref_submap_pose 
              = optimization_problem_->submap_data().at(
                      so3_loop.ref_submap).global_pose;
      
      so3_loop.is_translation_usable = false;
      so3_loop.obs_submap = kf.submap_id_;
      transform::Rigid3d ref_pose = img_keyframes_[loop_index].local_pose_;
      so3_loop.T_refmap_in_global = global_ref_submap_pose;
      so3_loop.T_ref_in_refmap = submap_data_.at(
          so3_loop.ref_submap).submap->local_pose().inverse() * ref_pose;
      so3_loop.T_obs_in_obsmap = submap_data_.at(
          so3_loop.obs_submap).submap->local_pose().inverse() * local_pose; 

      so3_loop.T_obs_ref = ref_pose.inverse() * kf_in_local_pnp;

      so3_loop.nodes_in_submap.clear();
      const transform::Rigid3d local_obs_submap_pose 
          = submap_data_.at(so3_loop.obs_submap).submap->local_pose();
      for (const NodeId& node_id : submap_data_.at(kf.submap_id_).node_ids){  
        if(trajectory_nodes_.at(node_id).time() > kf.time_) break;
        so3_loop.nodes_in_submap.push_back({node_id, TrajectoryNode{
          trajectory_nodes_.at(node_id).constant_data,
          local_obs_submap_pose.inverse() * 
          trajectory_nodes_.at(node_id).constant_data->local_pose}
        });
      }
      constraint_builder_.DispatchScanMatcherConstruction(so3_loop);
    }*/
  }
}

void PoseGraph3D::AddImageData(int trajectory_id, 
                               const sensor::ImageFeatureData& img_data,
                               const std::shared_ptr<Submap>& matching_submap) {
  // disabled backend
  if(options_.optimize_every_n_nodes() == 0) return;
  // Todo, add option to enable/disable visual loop detection.
  // AddWorkItem([=]() REQUIRES(mutex_) {
  //   ComputeConstraintsForImage(trajectory_id, img_data, matching_submap);
  // }); 
}


void PoseGraph3D::AddOdometryData(const int trajectory_id,
                                  const sensor::OdometryData& odometry_data) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([=]() REQUIRES(mutex_) {
    optimization_problem_->AddOdometryData(trajectory_id, odometry_data);
  });
}

void PoseGraph3D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([=]() REQUIRES(mutex_) {
    optimization_problem_->AddFixedFramePoseData(trajectory_id,
                                                 fixed_frame_pose_data);
  });
}

void PoseGraph3D::AddLandmarkData(int trajectory_id,
                                  const sensor::LandmarkData& landmark_data)
    EXCLUDES(mutex_) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([=]() REQUIRES(mutex_) {
    for (const auto& observation : landmark_data.landmark_observations) {
      landmark_nodes_[observation.id].landmark_observations.emplace_back(
          PoseGraphInterface::LandmarkNode::LandmarkObservation{
              trajectory_id, landmark_data.time,
              observation.landmark_to_tracking_transform,
              observation.translation_weight, observation.rotation_weight});
    }
  });
}

void PoseGraph3D::ComputeConstraint(const NodeId& node_id,
                                    const SubmapId& submap_id,
                                    const SubmapId& node_submap_id) {
  CHECK(submap_data_.at(submap_id).state == SubmapState::kFinished);

  const transform::Rigid3d global_node_pose =
      optimization_problem_->node_data().at(node_id).global_pose;
  //  submap to match with
  const transform::Rigid3d global_submap_pose =
      optimization_problem_->submap_data().at(submap_id).global_pose;
  const transform::Rigid3d global_submap_pose_inverse =
      global_submap_pose.inverse();
  
  std::vector<TrajectoryNode> submap_nodes;
  Eigen::Vector3d avg_pos;
  avg_pos<<0.,0.,0.;
  for (const NodeId& submap_node_id : submap_data_.at(submap_id).node_ids) {
    const transform::Rigid3d tmp_pose =
      optimization_problem_->node_data().at(submap_node_id).global_pose;
    avg_pos += tmp_pose.translation();
    submap_nodes.push_back(
        TrajectoryNode{trajectory_nodes_.at(submap_node_id).constant_data,
                       global_submap_pose_inverse *
                           trajectory_nodes_.at(submap_node_id).global_pose});
  }
  CHECK(!submap_nodes.empty());
  //外层过滤，如果两者几何位置相距太大，直接跳过
  avg_pos /= submap_nodes.size();
  double dist_to_center = (global_node_pose.translation() - avg_pos).norm();
  if(dist_to_center > options_.max_radius_eable_loop_detection()) return;

  const common::Time node_time = GetLatestNodeTime(node_id, submap_id);
  const common::Time last_connection_time =
      trajectory_connectivity_state_.LastConnectionTime(
          node_id.trajectory_id, submap_id.trajectory_id);
  // modified by wz, naive implementation for easier loop closure within a same trajectory
  if (node_id.trajectory_id == submap_id.trajectory_id ||
      node_time < last_connection_time + common::FromSeconds(
        options_.global_constraint_search_after_n_seconds())) {
    // If the node and the submap belong to the same trajectory or if there has
    // been a recent global constraint that ties that node's trajectory to the
    // submap's trajectory, it suffices to do a match constrained to a local
    // search window.
    if(std::abs(node_submap_id.submap_index - submap_id.submap_index) 
        < options_.num_close_submaps_loop_with_initial_value()){
      // constraint_builder_.MaybeAddConstraint(
      //     submap_id, submap_data_.at(submap_id).submap.get(), node_id,
      //     trajectory_nodes_.at(node_id).constant_data.get(), submap_nodes,
      //     global_node_pose, global_submap_pose);
    }else{
      // constraint_builder_.MaybeAddGlobalConstraint(
      //   submap_id, submap_data_.at(submap_id).submap.get(), node_id,
      //   trajectory_nodes_.at(node_id).constant_data.get(), submap_nodes,
      //   global_node_pose.rotation(), global_submap_pose.rotation());
    }
  } else if (global_localization_samplers_[node_id.trajectory_id]->Pulse()) {
    // In this situation, 'global_node_pose' and 'global_submap_pose' have
    // orientations agreeing on gravity. Their relationship regarding yaw is
    // arbitrary. Finding the correct yaw component will be handled by the
    // matching procedure in the FastCorrelativeScanMatcher, and the given yaw
    // is essentially ignored./
    // constraint_builder_.MaybeAddGlobalConstraint(
    //     submap_id, submap_data_.at(submap_id).submap.get(), node_id,
    //     trajectory_nodes_.at(node_id).constant_data.get(), submap_nodes,
    //     global_node_pose.rotation(), global_submap_pose.rotation());
  }
}



void PoseGraph3D::ComputeConstraint(const NodeId& node_id,
                                    const SubmapId& submap_id) {
  CHECK(submap_data_.at(submap_id).state == SubmapState::kFinished);

  const transform::Rigid3d global_node_pose =
      optimization_problem_->node_data().at(node_id).global_pose;

  const transform::Rigid3d global_submap_pose =
      optimization_problem_->submap_data().at(submap_id).global_pose;

  const transform::Rigid3d global_submap_pose_inverse =
      global_submap_pose.inverse();

  const common::Time node_time = GetLatestNodeTime(node_id, submap_id);
  const common::Time last_connection_time =
      trajectory_connectivity_state_.LastConnectionTime(
          node_id.trajectory_id, submap_id.trajectory_id);
  
  std::vector<TrajectoryNode> submap_nodes;
  for (const NodeId& submap_node_id : submap_data_.at(submap_id).node_ids) {
    submap_nodes.push_back(
        TrajectoryNode{trajectory_nodes_.at(submap_node_id).constant_data,
                       global_submap_pose_inverse *
                           trajectory_nodes_.at(submap_node_id).global_pose});
  }
  CHECK(!submap_nodes.empty());
  if (node_id.trajectory_id == submap_id.trajectory_id ||
      node_time < last_connection_time + common::FromSeconds(
                  options_.global_constraint_search_after_n_seconds())) {
    // If the node and the submap belong to the same trajectory or if there has
    // been a recent global constraint that ties that node's trajectory to the
    // submap's trajectory, it suffices to do a match constrained to a local
    // search window.
    // constraint_builder_.MaybeAddConstraint(
    //       submap_id, submap_data_.at(submap_id).submap.get(), node_id,
    //       trajectory_nodes_.at(node_id).constant_data.get(), submap_nodes,
    //       global_node_pose, global_submap_pose);
  } else if (global_localization_samplers_[node_id.trajectory_id]->Pulse()) {
    // In this situation, 'global_node_pose' and 'global_submap_pose' have
    // orientations agreeing on gravity. Their relationship regarding yaw is
    // arbitrary. Finding the correct yaw component will be handled by the
    // matching procedure in the FastCorrelativeScanMatcher, and the given yaw
    // is essentially ignored.
    // constraint_builder_.MaybeAddGlobalConstraint(
    //     submap_id, submap_data_.at(submap_id).submap.get(), node_id,
    //     trajectory_nodes_.at(node_id).constant_data.get(), submap_nodes,
    //     global_node_pose.rotation(), global_submap_pose.rotation());
  }
}

void PoseGraph3D::ComputeConstraintsForOldNodes(const SubmapId& submap_id) {
  const auto& submap_data = submap_data_.at(submap_id);
  for (const auto& node_id_data : optimization_problem_->node_data()) {
    const NodeId& node_id = node_id_data.id;
    if (submap_data.node_ids.count(node_id) == 0) {
      // ComputeConstraint(node_id, submap_id);
    }
  }
}

void PoseGraph3D::ComputeConstraintsForNode(
    const NodeId& node_id,
    std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
    const bool newly_finished_submap) {
  const auto& constant_data = trajectory_nodes_.at(node_id).constant_data;
  const std::vector<SubmapId> submap_ids = InitializeGlobalSubmapPoses(
      node_id.trajectory_id, constant_data->time, insertion_submaps);
  CHECK_EQ(submap_ids.size(), insertion_submaps.size());
  const SubmapId matching_id = submap_ids.front();
  const transform::Rigid3d& local_pose = constant_data->local_pose;
  const transform::Rigid3d global_pose =
      optimization_problem_->submap_data().at(matching_id).global_pose *
      insertion_submaps.front()->local_pose().inverse() * local_pose;
  optimization_problem_->AddTrajectoryNode(
      matching_id.trajectory_id,
      optimization::NodeSpec3D{constant_data->time, local_pose, global_pose});
  for (size_t i = 0; i < insertion_submaps.size(); ++i) {
    const SubmapId submap_id = submap_ids[i];
    // Even if this was the last node added to 'submap_id', the submap will only
    // be marked as finished in 'submap_data_' further below.
    CHECK(submap_data_.at(submap_id).state == SubmapState::kActive);
    submap_data_.at(submap_id).node_ids.emplace(node_id);
    const transform::Rigid3d constraint_transform =
        insertion_submaps[i]->local_pose().inverse() * local_pose;
    constraints_.push_back(
        Constraint{submap_id,
                   node_id,
                   {constraint_transform, options_.matcher_translation_weight(),
                    options_.matcher_rotation_weight()},
                   Constraint::INTRA_SUBMAP});
  }

  /* const transform::Rigid3d global_node_pose 
    = optimization_problem_->node_data().at(node_id).global_pose;
  if((global_node_pose.translation() 
    - last_node_pose_to_find_constraint_.translation()).norm() 
      < options_.nodes_space_to_perform_loop_detection()){
    // nothing to do
  }else{
    for (const auto& submap_id_data : submap_data_) {
      if (submap_id_data.data.state == SubmapState::kFinished) {
        CHECK_EQ(submap_id_data.data.node_ids.count(node_id), 0);
        ComputeConstraint(node_id, submap_id_data.id, matching_id);
      }
    }
    last_node_pose_to_find_constraint_ = global_node_pose;
  } */ 
  
  // Enable submap-to-submap loop detection and constriant construction.
  if (newly_finished_submap && options_.optimize_every_n_nodes() > 0) {
    const SubmapId finished_submap_id = submap_ids.front();
    InternalSubmapData& finished_submap_data =
        submap_data_.at(finished_submap_id);
    CHECK(finished_submap_data.state == SubmapState::kActive);
    finished_submap_data.state = SubmapState::kFinished;
    ComputeConstraintsForSubmap(finished_submap_id);
  }
  
  // Enable scan-context loop detection and constriant construction.
  // if (options_.optimize_every_n_nodes() > 0) {
  //   constraint_builder_.DispatchScanContextMatching(
  //       node_id, submap_ids.front(), trajectory_nodes_.at(node_id).constant_data);
  // }

  constraint_builder_.NotifyEndOfNode();
  ++num_nodes_since_last_loop_closure_;
  CHECK(!run_loop_closure_);
  if (options_.optimize_every_n_nodes() > 0 &&
      num_nodes_since_last_loop_closure_ > options_.optimize_every_n_nodes()) {
    DispatchOptimization();
  }
}

// 原来的设计是当需要执行优化的时候，前端发过来的所有计算任务（work_item）都会被阻塞
// 只有当执行完一次Optimization后才会顺次执行,这样能确保全局地图的一致性
void PoseGraph3D::DispatchOptimization() {
  run_loop_closure_ = true;
  // If there is a 'work_queue_' already, some other thread will take care.
  if (work_queue_ == nullptr) {
    work_queue_ = common::make_unique<std::deque<std::function<void()>>>();
      auto optimization_task = common::make_unique<common::Task>();
      optimization_task->SetWorkItem([=]() EXCLUDES(mutex_) { 
        HandleWorkQueue(constraint_builder_.GetConstraints());
    });
    auto optimization_task_handle =
      constraint_builder_.GetThreadPool()->Schedule(
        std::move(optimization_task));
    // optimization_task_->AddDependency(optimization_task_handle);
    tasks_tracker_.push_back(optimization_task_handle);
  }else{
    LOG(WARNING) << "Remaining work items: " << work_queue_->size();
  }
}

common::Time PoseGraph3D::GetLatestNodeTime(const NodeId& node_id,
                                            const SubmapId& submap_id) const {
  common::Time time = trajectory_nodes_.at(node_id).constant_data->time;
  const InternalSubmapData& submap_data = submap_data_.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *submap_data_.at(submap_id).node_ids.rbegin();
    time = std::max(
        time, trajectory_nodes_.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

void PoseGraph3D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  CHECK_EQ(constraint.tag, PoseGraphInterface::Constraint::INTER_SUBMAP);
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  trajectory_connectivity_state_.Connect(constraint.node_id.trajectory_id,
                                         constraint.submap_id.trajectory_id,
                                         time);
}

void PoseGraph3D::HandleWorkQueue(
    const constraints::ConstraintBuilder3D::Result& result) {
  cartographer::common::TicToc tic_toc;
  tic_toc.Tic();
  {
    common::MutexLocker locker(&mutex_);
    for(const auto& constraint: result){
      bool has_added = false;
      for(int i = 0; i < constraints_.size(); ++i){
        const auto& constraint_i = constraints_[i];
        if(constraint_i.submap_id ==  constraint.submap_id
          && constraint_i.node_id ==  constraint.node_id){
          has_added = true;
          break;
        }
      }
      if(!has_added){
        constraints_.push_back(constraint);
      }
    }
  }
  
  {
    RunOptimization();
  }
  
  if (global_slam_optimization_callback_) {
    std::map<int, NodeId> trajectory_id_to_last_optimized_node_id;
    std::map<int, SubmapId> trajectory_id_to_last_optimized_submap_id;
    {
      common::MutexLocker locker(&mutex_);
      const auto& submap_data = optimization_problem_->submap_data();
      const auto& node_data = optimization_problem_->node_data();
      for (const int trajectory_id : node_data.trajectory_ids()) {
        trajectory_id_to_last_optimized_node_id[trajectory_id] =
            std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
        trajectory_id_to_last_optimized_submap_id[trajectory_id] =
            std::prev(submap_data.EndOfTrajectory(trajectory_id))->id;
      }
    }
    global_slam_optimization_callback_(
        trajectory_id_to_last_optimized_submap_id,
        trajectory_id_to_last_optimized_node_id);
  }
  common::MutexLocker locker(&mutex_);
  for (const Constraint& constraint : result) {
    UpdateTrajectoryConnectivity(constraint);
  }
  TrimmingHandle trimming_handle(this);
  for (auto& trimmer : trimmers_) {
    trimmer->Trim(&trimming_handle);
  }
  trimmers_.erase(
      std::remove_if(trimmers_.begin(), trimmers_.end(),
                     [](std::unique_ptr<PoseGraphTrimmer>& trimmer) {
                       return trimmer->IsFinished();
                     }),
      trimmers_.end());
  num_nodes_since_last_loop_closure_ = 0;
  run_loop_closure_ = false;
  sum_t_cost_ += tic_toc.Toc();
  while (!run_loop_closure_) {
    if(!work_queue_) return;
    if (work_queue_->empty()) {
      work_queue_.reset();
      return;
    }
    work_queue_->front()();
    work_queue_->pop_front();
  }
  // LOG(INFO) << "Remaining work items in queue: " << work_queue_->size();
}

void PoseGraph3D::WaitForAllComputations() {
  while (1) {
    if(!constraint_builder_.AllTaskFinished()){
      LOG(INFO) << "Waiting for constraints computing tasks finish...";
      sleep(2.0);
      continue;
    }else if(!AllTaskFinished()){
      LOG(INFO) << "Waiting for optimization finish..., Remaining task size: "
                << tasks_tracker_.size();
      sleep(2.0);
      continue;
    }else{
      LOG(INFO) << "All tasks finished!";
      break;
    }
  }
}

void PoseGraph3D::FinishTrajectory(const int trajectory_id) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([this, trajectory_id]() REQUIRES(mutex_) {
    CHECK_EQ(finished_trajectories_.count(trajectory_id), 0);
    finished_trajectories_.insert(trajectory_id);

    for (const auto& submap : submap_data_.trajectory(trajectory_id)) {
      submap_data_.at(submap.id).state = SubmapState::kFinished;
    }
    CHECK(!run_loop_closure_);
    DispatchOptimization();
  });
}

bool PoseGraph3D::IsTrajectoryFinished(const int trajectory_id) const {
  return finished_trajectories_.count(trajectory_id) > 0;
}

void PoseGraph3D::FreezeTrajectory(const int trajectory_id) {
  common::MutexLocker locker(&mutex_);
  trajectory_connectivity_state_.Add(trajectory_id);
  AddWorkItem([this, trajectory_id]() REQUIRES(mutex_) {
    CHECK_EQ(frozen_trajectories_.count(trajectory_id), 0);
    frozen_trajectories_.insert(trajectory_id);
  });
}

bool PoseGraph3D::IsTrajectoryFrozen(const int trajectory_id) const {
  return frozen_trajectories_.count(trajectory_id) > 0;
}

void PoseGraph3D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose, const proto::Submap& submap) {
  if (!submap.has_submap_3d()) {
    return;
  }

  const SubmapId submap_id = {submap.submap_id().trajectory_id(),
                              submap.submap_id().submap_index()};
  std::shared_ptr<const Submap3D> submap_ptr =
      std::make_shared<const Submap3D>(submap.submap_3d());

  common::MutexLocker locker(&mutex_);
  AddTrajectoryIfNeeded(submap_id.trajectory_id);
  submap_data_.Insert(submap_id, InternalSubmapData());
  submap_data_.at(submap_id).submap = submap_ptr;
  // Immediately show the submap at the 'global_submap_pose'.
  global_submap_poses_.Insert(submap_id,
                              optimization::SubmapSpec3D{global_submap_pose});
  AddWorkItem([this, submap_id, global_submap_pose]() REQUIRES(mutex_) {
    submap_data_.at(submap_id).state = SubmapState::kFinished;
    optimization_problem_->InsertSubmap(submap_id, global_submap_pose);
  });
}

void PoseGraph3D::AddNodeFromProto(const transform::Rigid3d& global_pose,
                                   const proto::Node& node) {
  const NodeId node_id = {node.node_id().trajectory_id(),
                          node.node_id().node_index()};
  std::shared_ptr<const TrajectoryNode::Data> constant_data =
      std::make_shared<const TrajectoryNode::Data>(FromProto(node.node_data()));

  common::MutexLocker locker(&mutex_);
  AddTrajectoryIfNeeded(node_id.trajectory_id);
  trajectory_nodes_.Insert(node_id, TrajectoryNode{constant_data, global_pose});

  AddWorkItem([this, node_id, global_pose]() REQUIRES(mutex_) {
    const auto& constant_data = trajectory_nodes_.at(node_id).constant_data;
    optimization_problem_->InsertTrajectoryNode(
        node_id,
        optimization::NodeSpec3D{constant_data->time, constant_data->local_pose,
                                 global_pose});
  });
}

void PoseGraph3D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
  TrajectoryData trajectory_data;
  trajectory_data.gravity_constant = data.gravity_constant();
  trajectory_data.imu_calibration = {
      {data.imu_calibration().w(), data.imu_calibration().x(),
       data.imu_calibration().y(), data.imu_calibration().z()}};
  if (data.has_fixed_frame_origin_in_map()) {
    trajectory_data.fixed_frame_origin_in_map =
        transform::ToRigid3(data.fixed_frame_origin_in_map());
  }

  const int trajectory_id = data.trajectory_id();
  common::MutexLocker locker(&mutex_);
  AddWorkItem([this, trajectory_id, trajectory_data]() REQUIRES(mutex_) {
    optimization_problem_->SetTrajectoryData(trajectory_id, trajectory_data);
  });
}

void PoseGraph3D::AddNodeToSubmap(const NodeId& node_id,
                                  const SubmapId& submap_id) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([this, node_id, submap_id]() REQUIRES(mutex_) {
    submap_data_.at(submap_id).node_ids.insert(node_id);
  });
}

void PoseGraph3D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([this, constraints]() REQUIRES(mutex_) {
    for (const auto& constraint : constraints) {
      CHECK(trajectory_nodes_.Contains(constraint.node_id));
      CHECK(submap_data_.Contains(constraint.submap_id));
      CHECK(trajectory_nodes_.at(constraint.node_id).constant_data != nullptr);
      CHECK(submap_data_.at(constraint.submap_id).submap != nullptr);
      switch (constraint.tag) {
        case Constraint::Tag::INTRA_SUBMAP:
          CHECK(submap_data_.at(constraint.submap_id)
                    .node_ids.emplace(constraint.node_id)
                    .second);
          break;
        case Constraint::Tag::INTER_SUBMAP:
          UpdateTrajectoryConnectivity(constraint);
          break;
      }
      constraints_.push_back(constraint);
    }
    LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
  });
}

void PoseGraph3D::AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) {
  common::MutexLocker locker(&mutex_);
  // C++11 does not allow us to move a unique_ptr into a lambda.
  PoseGraphTrimmer* const trimmer_ptr = trimmer.release();
  AddWorkItem([this, trimmer_ptr]()
                  REQUIRES(mutex_) { trimmers_.emplace_back(trimmer_ptr); });
}

void PoseGraph3D::RunFinalOptimization() {
  // Find loop of the last submap (may still be active).
  for(const auto& submap: submap_data_){
    if(submap.data.state == SubmapState::kActive){
      ComputeConstraintsForSubmap(submap.id);
    }
  }

  // wait for all tasks finished, then optimize again.
  WaitForAllComputations();

  // run optimization again.
  auto optimization_task = common::make_unique<common::Task>();
  optimization_task->SetWorkItem([=]() EXCLUDES(mutex_) {
    optimization_problem_->SetMaxNumIterations(
        options_.max_num_final_iterations());
    optimization_problem_->SetMaxNumIterations(
        options_.optimization_problem_options()
            .ceres_solver_options()
            .max_num_iterations()); 
    HandleWorkQueue(constraint_builder_.GetConstraints());
  });
  auto optimization_task_handle =
    constraint_builder_.GetThreadPool()->Schedule(
      std::move(optimization_task));
  tasks_tracker_.push_back(optimization_task_handle);
  
  // wait for the last optimization task in background threads finish.
  WaitForAllComputations();
  LOG(WARNING)<<"Average cost time for each node is: "<<(constraint_builder_.GetCostTime() + sum_t_cost_) / constraint_builder_.GetNumFinishedNodes();
  // std::cout << "\r\x1b[KOptimizing: Done.     " << std::endl;
}

void PoseGraph3D::LogResidualHistograms() const {
  common::Histogram rotational_residual;
  common::Histogram translational_residual;
  for (const Constraint& constraint : constraints_) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
      const cartographer::transform::Rigid3d optimized_node_to_map =
          trajectory_nodes_.at(constraint.node_id).global_pose;
      const cartographer::transform::Rigid3d node_to_submap_constraint =
          constraint.pose.zbar_ij;
      const cartographer::transform::Rigid3d optimized_submap_to_map =
          global_submap_poses_.at(constraint.submap_id).global_pose;
      const cartographer::transform::Rigid3d optimized_node_to_submap =
          optimized_submap_to_map.inverse() * optimized_node_to_map;
      const cartographer::transform::Rigid3d residual =
          node_to_submap_constraint.inverse() * optimized_node_to_submap;
      rotational_residual.Add(
          common::NormalizeAngleDifference(transform::GetAngle(residual)));
      translational_residual.Add(residual.translation().norm());
    }
  }
  LOG(INFO) << "Translational residuals histogram:\n"
            << translational_residual.ToString(10);
  LOG(INFO) << "Rotational residuals histogram:\n"
            << rotational_residual.ToString(10);
}

void PoseGraph3D::RunOptimization() {
  if (optimization_problem_->submap_data().empty()) {
    return;
  }

  // No other thread is accessing the optimization_problem_, constraints_,
  // frozen_trajectories_ and landmark_nodes_ when executing the Solve. Solve is
  // time consuming, so not taking the mutex before Solve to avoid blocking
  // foreground processing.
  optimization_problem_->Solve(constraints_, frozen_trajectories_,
                               landmark_nodes_);

  common::MutexLocker locker(&mutex_);
  const auto& submap_data = optimization_problem_->submap_data();
  const auto& node_data = optimization_problem_->node_data();
  for (const int trajectory_id : node_data.trajectory_ids()) {
    for (const auto& node : node_data.trajectory(trajectory_id)) {
      trajectory_nodes_.at(node.id).global_pose = node.data.global_pose;
    }

    // Extrapolate all point cloud poses that were not included in the
    // 'optimization_problem_' yet.
    const auto local_to_new_global =
        ComputeLocalToGlobalTransform(submap_data, trajectory_id);
    const auto local_to_old_global =
        ComputeLocalToGlobalTransform(global_submap_poses_, trajectory_id);
    const transform::Rigid3d old_global_to_new_global =
        local_to_new_global * local_to_old_global.inverse();

    const NodeId last_optimized_node_id =
        std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
    auto node_it = std::next(trajectory_nodes_.find(last_optimized_node_id));
    for (; node_it != trajectory_nodes_.EndOfTrajectory(trajectory_id);
         ++node_it) {
      auto& mutable_trajectory_node = trajectory_nodes_.at(node_it->id);
      mutable_trajectory_node.global_pose =
          old_global_to_new_global * mutable_trajectory_node.global_pose;
    }
  }
  for (const auto& landmark : optimization_problem_->landmark_data()) {
    landmark_nodes_[landmark.first].global_landmark_pose = landmark.second;
  }
  global_submap_poses_ = submap_data;

  // Log the histograms for the pose residuals.
  if (options_.log_residual_histograms()) {
    LogResidualHistograms();
  }
}

MapById<NodeId, TrajectoryNode> PoseGraph3D::GetTrajectoryNodes() const {
  common::MutexLocker locker(&mutex_);
  return trajectory_nodes_;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph3D::GetTrajectoryNodePoses()
    const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  common::MutexLocker locker(&mutex_);
  for (const auto& node_id_data : trajectory_nodes_) {
    common::optional<TrajectoryNodePose::ConstantPoseData> constant_pose_data;
    if (node_id_data.data.constant_data != nullptr) {
      constant_pose_data = TrajectoryNodePose::ConstantPoseData{
          node_id_data.data.constant_data->time,
          node_id_data.data.constant_data->local_pose};
    }
    node_poses.Insert(
        node_id_data.id,
        TrajectoryNodePose{node_id_data.data.global_pose, constant_pose_data});
  }
  return node_poses;
}

std::map<std::string, transform::Rigid3d> PoseGraph3D::GetLandmarkPoses()
    const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  common::MutexLocker locker(&mutex_);
  for (const auto& landmark : landmark_nodes_) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

void PoseGraph3D::SetLandmarkPose(const std::string& landmark_id,
                                  const transform::Rigid3d& global_pose) {
  common::MutexLocker locker(&mutex_);
  AddWorkItem([=]() REQUIRES(mutex_) {
    landmark_nodes_[landmark_id].global_landmark_pose = global_pose;
  });
}

sensor::MapByTime<sensor::ImuData> PoseGraph3D::GetImuData() const {
  common::MutexLocker locker(&mutex_);
  return optimization_problem_->imu_data();
}

sensor::MapByTime<sensor::OdometryData> PoseGraph3D::GetOdometryData() const {
  common::MutexLocker locker(&mutex_);
  return optimization_problem_->odometry_data();
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph3D::GetFixedFramePoseData() const {
  common::MutexLocker locker(&mutex_);
  return optimization_problem_->fixed_frame_pose_data();
}

std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
PoseGraph3D::GetLandmarkNodes() const {
  common::MutexLocker locker(&mutex_);
  return landmark_nodes_;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph3D::GetTrajectoryData() const {
  common::MutexLocker locker(&mutex_);
  return optimization_problem_->trajectory_data();
}

std::vector<PoseGraphInterface::Constraint> PoseGraph3D::constraints() const {
  common::MutexLocker locker(&mutex_);
  return constraints_;
}

void PoseGraph3D::SetInitialTrajectoryPose(const int from_trajectory_id,
                                           const int to_trajectory_id,
                                           const transform::Rigid3d& pose,
                                           const common::Time time) {
  common::MutexLocker locker(&mutex_);
  initial_trajectory_poses_[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph3D::GetInterpolatedGlobalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  CHECK_GT(trajectory_nodes_.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = trajectory_nodes_.lower_bound(trajectory_id, time);
  if (it == trajectory_nodes_.BeginOfTrajectory(trajectory_id)) {
    return trajectory_nodes_.BeginOfTrajectory(trajectory_id)->data.global_pose;
  }
  if (it == trajectory_nodes_.EndOfTrajectory(trajectory_id)) {
    return std::prev(trajectory_nodes_.EndOfTrajectory(trajectory_id))
        ->data.global_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.global_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.global_pose},
             time)
      .transform;
}

transform::Rigid3d PoseGraph3D::GetInterpolatedLocalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  CHECK_GT(trajectory_nodes_.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = trajectory_nodes_.lower_bound(trajectory_id, time);
  if (it == trajectory_nodes_.BeginOfTrajectory(trajectory_id)) {
    return trajectory_nodes_.BeginOfTrajectory(trajectory_id)->data.constant_data->local_pose;
  }
  if (it == trajectory_nodes_.EndOfTrajectory(trajectory_id)) {
    return std::prev(trajectory_nodes_.EndOfTrajectory(trajectory_id))
        ->data.constant_data->local_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.constant_data->local_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.constant_data->local_pose},
             time)
      .transform;
}

transform::Rigid3d PoseGraph3D::GetLocalToGlobalTransform(
    const int trajectory_id) const {
  common::MutexLocker locker(&mutex_);
  return ComputeLocalToGlobalTransform(global_submap_poses_, trajectory_id);
}

std::vector<std::vector<int>> PoseGraph3D::GetConnectedTrajectories() const {
  return trajectory_connectivity_state_.Components();
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapData(
    const SubmapId& submap_id) const {
  common::MutexLocker locker(&mutex_);
  return GetSubmapDataUnderLock(submap_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetAllSubmapData() const {
  common::MutexLocker locker(&mutex_);
  return GetSubmapDataUnderLock();
}

MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph3D::GetAllSubmapPoses() const {
  common::MutexLocker locker(&mutex_);
  MapById<SubmapId, SubmapPose> submap_poses;
  for (const auto& submap_id_data : submap_data_) {
    auto submap_data = GetSubmapDataUnderLock(submap_id_data.id);
    submap_poses.Insert(
        submap_id_data.id,
        PoseGraphInterface::SubmapPose{submap_data.submap->num_range_data(),
                                       submap_data.pose});
  }
  return submap_poses;
}

transform::Rigid3d PoseGraph3D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
    const int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = initial_trajectory_poses_.find(trajectory_id);
    if (it != initial_trajectory_poses_.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  return global_submap_poses.at(last_optimized_submap_id).global_pose *
         submap_data_.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  const auto it = submap_data_.find(submap_id);
  if (it == submap_data_.end()) {
    return {};
  }
  auto submap = it->data.submap;
  if (global_submap_poses_.Contains(submap_id)) {
    // We already have an optimized pose.
    return {submap, global_submap_poses_.at(submap_id).global_pose};
  }
  // We have to extrapolate.
  return {submap, ComputeLocalToGlobalTransform(global_submap_poses_,
                                                submap_id.trajectory_id) *
                      submap->local_pose()};
}

PoseGraph3D::TrimmingHandle::TrimmingHandle(PoseGraph3D* const parent)
    : parent_(parent) {}

int PoseGraph3D::TrimmingHandle::num_submaps(const int trajectory_id) const {
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  return submap_data.SizeOfTrajectoryOrZero(trajectory_id);
}

std::vector<SubmapId> PoseGraph3D::TrimmingHandle::GetSubmapIds(
    int trajectory_id) const {
  std::vector<SubmapId> submap_ids;
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  for (const auto& it : submap_data.trajectory(trajectory_id)) {
    submap_ids.push_back(it.id);
  }
  return submap_ids;
}
MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::TrimmingHandle::GetOptimizedSubmapData() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : parent_->submap_data_) {
    if (submap_id_data.data.state != SubmapState::kFinished ||
        !parent_->global_submap_poses_.Contains(submap_id_data.id)) {
      continue;
    }
    submaps.Insert(
        submap_id_data.id,
        SubmapData{
            submap_id_data.data.submap,
            parent_->global_submap_poses_.at(submap_id_data.id).global_pose});
  }
  return submaps;
}

const MapById<NodeId, TrajectoryNode>&
PoseGraph3D::TrimmingHandle::GetTrajectoryNodes() const {
  return parent_->trajectory_nodes_;
}

const std::vector<PoseGraphInterface::Constraint>&
PoseGraph3D::TrimmingHandle::GetConstraints() const {
  return parent_->constraints_;
}

bool PoseGraph3D::TrimmingHandle::IsFinished(const int trajectory_id) const {
  return parent_->IsTrajectoryFinished(trajectory_id);
}

void PoseGraph3D::TrimmingHandle::MarkSubmapAsTrimmed(
    const SubmapId& submap_id) {
  // TODO(hrapp): We have to make sure that the trajectory has been finished
  // if we want to delete the last submaps.
  CHECK(parent_->submap_data_.at(submap_id).state == SubmapState::kFinished);

  // Compile all nodes that are still INTRA_SUBMAP constrained once the submap
  // with 'submap_id' is gone.
  std::set<NodeId> nodes_to_retain;
  for (const Constraint& constraint : parent_->constraints_) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
        constraint.submap_id != submap_id) {
      nodes_to_retain.insert(constraint.node_id);
    }
  }
  // Remove all 'constraints_' related to 'submap_id'.
  std::set<NodeId> nodes_to_remove;
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->constraints_) {
      if (constraint.submap_id == submap_id) {
        if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
            nodes_to_retain.count(constraint.node_id) == 0) {
          // This node will no longer be INTRA_SUBMAP contrained and has to be
          // removed.
          nodes_to_remove.insert(constraint.node_id);
        }
      } else {
        constraints.push_back(constraint);
      }
    }
    parent_->constraints_ = std::move(constraints);
  }
  // Remove all 'constraints_' related to 'nodes_to_remove'.
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->constraints_) {
      if (nodes_to_remove.count(constraint.node_id) == 0) {
        constraints.push_back(constraint);
      }
    }
    parent_->constraints_ = std::move(constraints);
  }

  // Mark the submap with 'submap_id' as trimmed and remove its data.
  CHECK(parent_->submap_data_.at(submap_id).state == SubmapState::kFinished);
  parent_->submap_data_.Trim(submap_id);
  parent_->constraint_builder_.DeleteScanMatcher(submap_id);
  parent_->optimization_problem_->TrimSubmap(submap_id);

  // Remove the 'nodes_to_remove' from the pose graph and the optimization
  // problem.
  for (const NodeId& node_id : nodes_to_remove) {
    parent_->trajectory_nodes_.Trim(node_id);
    parent_->optimization_problem_->TrimTrajectoryNode(node_id);
  }
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : submap_data_) {
    submaps.Insert(submap_id_data.id,
                   GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

void PoseGraph3D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  global_slam_optimization_callback_ = callback;
}

SubmapId PoseGraph3D::FindSubmapId(const std::shared_ptr<Submap>& submap) const{
  for(const auto& entry: submap_data_){
    if(entry.data.submap == submap){
      return entry.id;
    }
  }
  SubmapId id;
  id.trajectory_id = -1;
  id.submap_index = -1;
  return id;
}

void PoseGraph3D::ComputeConstraintsForSubmap(
    const SubmapId& submap_id){
  // LOG(WARNING)<<"submap_id.submap_index: " << submap_id.submap_index;
  const transform::Rigid3d global_submap_pose =
      optimization_problem_->submap_data().at(submap_id).global_pose;
  const transform::Rigid3d local_submap_pose =
      submap_data_.at(submap_id).submap->local_pose();
  const transform::Rigid3d global_submap_pose_inverse =
      global_submap_pose.inverse();
  std::vector<std::pair<NodeId, TrajectoryNode> > submap_nodes;
  for (const NodeId& submap_node_id : submap_data_.at(submap_id).node_ids) {  
    submap_nodes.push_back({submap_node_id,
        TrajectoryNode{trajectory_nodes_.at(submap_node_id).constant_data,
                       /* global_submap_pose_inverse *
                       trajectory_nodes_.at(submap_node_id).global_pose */
              local_submap_pose.inverse() 
              * trajectory_nodes_.at(submap_node_id).constant_data->local_pose}
    });
  }
  constraint_builder_.DispatchScanMatcherConstruction(
    submap_id, local_submap_pose, submap_nodes, 
    submap_data_.at(submap_id).submap.get());
}


void PoseGraph3D::AssociateDepthForVisualFeatures(
    const transform::Rigid3d& cam_pose_in_local,
    const std::vector<Eigen::Vector2f>& features_2d, 
    std::vector<Eigen::Vector3f>& features_3d){
  if(features_2d.empty()) return;
  Eigen::Vector3f pt(0,0,-1);
  features_3d.resize(features_2d.size(), pt);

  // 0.5 project undistorted normalized (z) 2d features onto a unit sphere
  pcl::PointCloud<pcl::PointXYZ>::Ptr features_3d_sphere(
    new pcl::PointCloud<pcl::PointXYZ>());
  for (size_t i = 0; i < features_2d.size(); ++i){
    // normalize the 2d feature on the normalized plane to a unit sphere
    Eigen::Vector3f feature_cur(features_2d[i][0], features_2d[i][1], 1.0); 
    feature_cur.normalize(); 
    pcl::PointXYZ p;
    p.x = feature_cur(0);
    p.y = feature_cur(1);
    p.z = feature_cur(2);
    features_3d[i] = Vector3f(features_2d[i][0], features_2d[i][1], -1.0);
    features_3d_sphere->push_back(p);
  }

  // 3. project depth cloud on a range image, filter points satcked 
  // in the same region
  // currently only cover the space in front of the camera (-90 ~ 90)
  float bin_res = 180.0 / (float)num_bins_; 
  cv::Mat rangeImage = cv::Mat(
    num_bins_, num_bins_, CV_32F, cv::Scalar::all(FLT_MAX));
  
  transform::Rigid3f cam_pose_inv = cam_pose_in_local.inverse().cast<float>();
  int w = camera_->imageWidth();
  int h = camera_->imageHeight();
  int side = std::min(w, h);
  for(int k = 0; k < depth_clouds_.size(); ++k){
    const auto& cloud_k = depth_clouds_.at(k);
    for(size_t i = 0; i < cloud_k.size(); ++i){
      Eigen::Vector3f p_local;
      p_local << cloud_k.points[i].x, 
                 cloud_k.points[i].y, cloud_k.points[i].z;
      Eigen::Vector3f p_3d_cam = cam_pose_inv * p_local;
      
      // filter points not in camera view
      if(p_3d_cam[2] < 0.3) continue;
      
      // 2d point on the normalized plane
      Eigen::Vector2f p_2d_norm;
      p_2d_norm << p_3d_cam[0] / p_3d_cam[2], p_3d_cam[1] / p_3d_cam[2];
      
      // filter points not in camera view
      Eigen::Vector2d uv;
      camera_->spaceToPlane(p_3d_cam.cast<double>(), uv);
      if(uv[0] < 0 || uv[1] < 0 || uv[0] > w-1 || uv[1] > h-1) continue;
      if(std::abs(std::atan2(p_3d_cam[2], p_3d_cam[0])) < M_PI / 18.
        || std::abs(std::atan2(p_3d_cam[2], p_3d_cam[1])) < M_PI / 18.) continue;
      // find row id in range image
      float row_angle = std::atan2(
        p_3d_cam[1], p_3d_cam[2]) * 180.0 / M_PI + 90.0; 
      int row_id = std::round(row_angle / bin_res);
      // find column id in range image
      float col_angle = std::atan2(
        p_3d_cam[0], p_3d_cam[2]) * 180.0 / M_PI + 90.0; 
      int col_id = std::round(col_angle / bin_res);
      
      // id may be out of boundary
      if (row_id < 0 || row_id >= num_bins_ 
        || col_id < 0 || col_id >= num_bins_)
        continue;

      // only keep points that's closer
      float dist = p_3d_cam.norm();
      if (dist < rangeImage.at<float>(row_id, col_id)){
        rangeImage.at<float>(row_id, col_id) = dist;
        pcl::PointXYZ ptmp(p_3d_cam[0], p_3d_cam[1], p_3d_cam[2]);
        pointsArray[row_id][col_id] = ptmp;
      }
    }
  }
  
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_unit_sphere(
      new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> cloud_unit_sphere_ranges;

  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>());

  // 7. find the feature depth using kd-tree
   std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;
  float dist_sq_threshold = std::pow(std::sin(bin_res / 180.0 * M_PI) * 5.0, 2);
  for(size_t i = 0; i < features_3d_sphere->size(); ++i){
    depth_cloud_unit_sphere->clear();
    cloud_unit_sphere_ranges.clear();
    // get index on the sphere
    float row_angle = std::atan2(features_3d_sphere->points[i].y, 
        features_3d_sphere->points[i].z) * 180.0 / M_PI + 90.0; 
    int row_id = std::round(row_angle / bin_res);
    float col_angle = std::atan2(features_3d_sphere->points[i].x, 
        features_3d_sphere->points[i].z) * 180.0 / M_PI + 90.0; 
    int col_id = std::round(col_angle / bin_res);
    
    // id may be out of boundary
    if (row_id < 0 || row_id >= num_bins_ 
        || col_id < 0 || col_id >= num_bins_)
      continue;
    
    // retrieve points lie in a local spherical window.
    for(int r = -5; r < 5; ++r){
      for(int c = -5; c < 5; ++c){
        int ri = row_id + r;
        int ci = col_id + c;
        if (ri < 0 || ri >= num_bins_ 
          || ci < 0 || ci >= num_bins_) continue;
        if(rangeImage.at<float>(ri, ci) != FLT_MAX){
          pcl::PointXYZ p = pointsArray[ri][ci];
          float range = Range(p);
          p.x /= range;
          p.y /= range;
          p.z /= range;
          depth_cloud_unit_sphere->push_back(p);
          cloud_unit_sphere_ranges.push_back(range);
        }
      }
    }

    if(depth_cloud_unit_sphere->size() < 5) continue;
    kdtree->setInputCloud(depth_cloud_unit_sphere);
    kdtree->nearestKSearch(
      features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);
   
    if(pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold){
      float r1 = cloud_unit_sphere_ranges[pointSearchInd[0]];
      Eigen::Vector3f A(
          depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
          depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
          depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

      float r2 = cloud_unit_sphere_ranges[pointSearchInd[1]];
      Eigen::Vector3f B(
          depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
          depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
          depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

      float r3 = cloud_unit_sphere_ranges[pointSearchInd[2]];
      Eigen::Vector3f C(
          depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
          depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
          depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

      // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
      Eigen::Vector3f V(features_3d_sphere->points[i].x,
                        features_3d_sphere->points[i].y,
                        features_3d_sphere->points[i].z);

      Eigen::Vector3f N = (A - B).cross(B - C);
      float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) 
              / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

      float min_depth = min(r1, min(r2, r3));
      float max_depth = max(r1, max(r2, r3));
      if(max_depth - min_depth > 2 || s <= 0.5){
        continue;
      }else if(s - max_depth > 0){
        s = max_depth;
      }else if(s - min_depth < 0){
        s = min_depth;
      }
      
      // convert feature into cartesian space if depth is available
      Eigen::Vector3f p_3d = Eigen::Vector3f(
        features_3d_sphere->points[i].x * s,
        features_3d_sphere->points[i].y * s,
        features_3d_sphere->points[i].z * s);
      if(p_3d.norm() > 100 || p_3d.norm() < 0.2){
        continue;
      }
      features_3d[i] = p_3d;
      // LOG(INFO)<<features_3d[i].transpose();
    }
  }


  // 4. filter invalid points from range image
  /* pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_local_filter(
      new pcl::PointCloud<pcl::PointXYZ>());
  for(int i = 0; i < num_bins_; ++i){
    for(int j = 0; j < num_bins_; ++j){
      if(rangeImage.at<float>(i, j) != FLT_MAX)
        depth_cloud_local_filter->push_back(pointsArray[i][j]);
    }
  }

  // 5. project depth cloud onto a unit sphere
  pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_unit_sphere(
      new pcl::PointCloud<pcl::PointXYZ>());
  for(size_t i = 0; i < depth_cloud_local_filter->size(); ++i){
    pcl::PointXYZ p = depth_cloud_local_filter->points[i];
    float range = std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
    p.x /= range;
    p.y /= range;
    p.z /= range;
    depth_cloud_unit_sphere->push_back(p);
  }
  
  if(depth_cloud_unit_sphere->size() < 10) return;
  
  // 6. create a kd-tree using the spherical depth cloud
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(
    new pcl::KdTreeFLANN<pcl::PointXYZ>());
  kdtree->setInputCloud(depth_cloud_unit_sphere);

  // 7. find the feature depth using kd-tree
  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;
  float dist_sq_threshold = std::pow(std::sin(bin_res / 180.0 * M_PI) * 5.0, 2);
  for(size_t i = 0; i < features_3d_sphere->size(); ++i){
    kdtree->nearestKSearch(
      features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);
    if(pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold){
      float r1 = Range(depth_cloud_local_filter->points[pointSearchInd[0]]);
      Eigen::Vector3f A(
          depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
          depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
          depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

      float r2 = Range(depth_cloud_local_filter->points[pointSearchInd[1]]);
      Eigen::Vector3f B(
          depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
          depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
          depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

      float r3 = Range(depth_cloud_local_filter->points[pointSearchInd[2]]);
      Eigen::Vector3f C(
          depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
          depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
          depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

      // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
      Eigen::Vector3f V(features_3d_sphere->points[i].x,
                        features_3d_sphere->points[i].y,
                        features_3d_sphere->points[i].z);

      Eigen::Vector3f N = (A - B).cross(B - C);
      float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) 
              / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

      float min_depth = min(r1, min(r2, r3));
      float max_depth = max(r1, max(r2, r3));
      if(max_depth - min_depth > 2 || s <= 0.5){
        continue;
      }else if(s - max_depth > 0){
        s = max_depth;
      }else if(s - min_depth < 0){
        s = min_depth;
      }
      
      // convert feature into cartesian space if depth is available
      Eigen::Vector3f p_3d = Eigen::Vector3f(
        features_3d_sphere->points[i].x * s,
        features_3d_sphere->points[i].y * s,
        features_3d_sphere->points[i].z * s);
      if(p_3d.norm() > 100 || p_3d.norm() < 0.5) continue;
      features_3d[i] = p_3d;
    }
  } */
}

void PoseGraph3D::DrawDepthPointsAll(
    const cv::Mat& image, 
    const transform::Rigid3d& cam_pose_in_local){
  auto local_pose_inv = cam_pose_in_local.inverse();
  std::vector<cv::Point2f> points_2d;
  std::vector<float> points_distance;
  int w = camera_->imageWidth();
  int h = camera_->imageHeight();
  for(const auto& pc: depth_clouds_){
    for(const pcl::PointXYZ& pt: pc.points){
      Eigen::Vector3d pl, p_3d_cam;
      pl << pt.x, pt.y, pt.z;
      p_3d_cam = local_pose_inv * pl;
      if(p_3d_cam[2] < 0.3) continue;
      Eigen::Vector2d p_2d;
      camera_->spaceToPlane(p_3d_cam, p_2d);
      if(p_2d[0] < 0 || p_2d[1] < 0 || p_2d[0] > w-1 || p_2d[1] > h-1) continue;

      points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
      points_distance.push_back(p_3d_cam.norm());
    }

    cv::Mat showImage, circleImage;
    cv::cvtColor(image, showImage, cv::COLOR_GRAY2RGB);
    circleImage = showImage.clone();
    for (int i = 0; i < (int)points_2d.size(); ++i){
      float r, g, b;
      GetColor(points_distance[i], 50.0, r, g, b);
      cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
    }
    cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image
    cv::imwrite("/home/wz/Desktop/depth_debug/"+std::to_string(cur_image_kf_id_)+".jpg", showImage);
  }
}


void PoseGraph3D::GetColor(
    float p, float np, float&r, float&g, float&b){
  float inc = 6.0 / np;
  float x = p * inc;
  r = 0.0f; g = 0.0f; b = 0.0f;
  if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
  else if (4 <= x && x <= 5) r = x - 4;
  else if (1 <= x && x <= 2) r = 1.0f - (x - 1);

  if (1 <= x && x <= 3) g = 1.0f;
  else if (0 <= x && x <= 1) g = x - 0;
  else if (3 <= x && x <= 4) g = 1.0f - (x - 3);

  if (3 <= x && x <= 5) b = 1.0f;
  else if (2 <= x && x <= 3) b = x - 2;
  else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
  r *= 255.0;
  g *= 255.0;
  b *= 255.0;
}

}  // namespace mapping
}  // namespace cartographer
