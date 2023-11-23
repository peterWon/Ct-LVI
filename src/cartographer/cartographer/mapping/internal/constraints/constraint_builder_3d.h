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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_3D_H_

#include <array>
#include <deque>
#include <functional>
#include <limits>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/histogram.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/math.h"
#include "cartographer/common/mutex.h"
#include "cartographer/common/task.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/common/tic_toc.h"
#include "cartographer/mapping/3d/submap_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/ceres_scan_matcher_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/fast_correlative_scan_matcher_3d.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/sensor/point_cloud.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "cartographer/mapping/internal/3d/scan_context/Scancontext.h"
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>


namespace cartographer {
namespace mapping {
namespace constraints {

struct SE3SubmapConstraint{
  // reference submap ID of the observation image and reference image.
  SubmapId ref_submap;
  SubmapId obs_submap;

  transform::Rigid3d T_refmap_in_global;
  // transform::Rigid3d T_obsmap_in_global;
  
  // reference image in reference submap.
  transform::Rigid3d T_ref_in_refmap;
  // observation image in observation submap.
  transform::Rigid3d T_obs_in_obsmap;
  // se3 transform between the observation frame and reference frame.
  transform::Rigid3d T_obs_ref;

  // if this variable is set false, then only the RO(3) part takes effect.
  bool is_translation_usable;
  
  // record th node id and constant data pointers in the observation submap.
  std::vector<std::pair<NodeId, TrajectoryNode>> nodes_in_submap;
};

// Asynchronously computes constraints.
//
// Intermingle an arbitrary number of calls to 'MaybeAddConstraint',
// 'MaybeAddGlobalConstraint', and 'NotifyEndOfNode', then call 'WhenDone' once.
// After all computations are done the 'callback' will be called with the result
// and another MaybeAdd(Global)Constraint()/WhenDone() cycle can follow.
//
// This class is thread-safe.
class ConstraintBuilder3D {
 public:
  using Constraint = mapping::PoseGraphInterface::Constraint;
  using Result = std::vector<Constraint>;

  ConstraintBuilder3D(const proto::ConstraintBuilderOptions& options,
                      common::ThreadPoolInterface* thread_pool);
  ~ConstraintBuilder3D();

  ConstraintBuilder3D(const ConstraintBuilder3D&) = delete;
  ConstraintBuilder3D& operator=(const ConstraintBuilder3D&) = delete;
  
  // 回环检测接口,每完成一张新的子地图时触发
  void DispatchScanMatcherConstruction(
      const SubmapId& submap_id,
      const transform::Rigid3d& global_submap_pose,
      const std::vector<std::pair<NodeId, TrajectoryNode>>& submap_nodes, 
      const Submap3D* submap);
  
  // 回环检测接口,每查找到一次视觉回环时触发
  void DispatchScanMatcherConstruction(
      const SE3SubmapConstraint& se3_constraint);
  
  // 回环检测接口,每添加一帧关键帧触发，单独起作用，使用ScanContext为引擎
  void DispatchScanContextMatching(const NodeId& node_id, 
      const SubmapId& node_submap_id,
      std::shared_ptr<const TrajectoryNode::Data> constant_data);
  // Must be called after all computations related to one node have been added.
  void NotifyEndOfNode();

  // Returns the number of consecutive finished nodes.
  int GetNumFinishedNodes();
  double GetCostTime();
  
  // Delete data related to 'submap_id'.
  void DeleteScanMatcher(const SubmapId& submap_id);

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);
  
  common::ThreadPoolInterface* GetThreadPool(){
      common::MutexLocker locker(&mutex_);
      return thread_pool_;
  }
  Result GetConstraints() {
    common::MutexLocker locker(&mutex_);
    Result result;
    for (const std::unique_ptr<Constraint>& constraint : constraints_) {
      if (constraint == nullptr) continue;
      result.push_back(*constraint);
    }
    return result;
  }

  bool AllTaskFinished(){
    common::MutexLocker locker(&mutex_);
    for(auto task = tasks_tracker_.begin(); task !=tasks_tracker_.end(); ){
      if(task->expired()){
        tasks_tracker_.erase(task);
      }else{
        return false;
      }
    }
    return true;
  }

  // Not used anymore.
  // Registers the 'callback' to be called with the results, after all
  // computations triggered by 'MaybeAdd*Constraint' have finished.
  // 'callback' is executed in the 'ThreadPool'.
  void WhenDone(const std::function<void(const Result&)>& callback);

 private:
  enum InitialPoseType {
    LASER_LOOP_5DOF=1,
    LASER_LOOP_6DOF=2,
    VISUAL_LOOP_6DOF,
  };
  struct SubmapScanMatcher {
    const HybridGrid* high_resolution_hybrid_grid;
    const HybridGrid* low_resolution_hybrid_grid;
    transform::Rigid3d global_submap_pose; //retrieve gravity_aligned

    std::unique_ptr<scan_matching::FastCorrelativeScanMatcher3D>
        fast_correlative_scan_matcher;
    std::weak_ptr<common::Task> creation_task_handle;
    
    // wz: add for loop closure searching 
    cv::Mat prj_grid = cv::Mat();
    double ox, oy, resolution;
    std::vector<cv::KeyPoint> key_points = {};
    cv::Mat descriptors = cv::Mat();
    
    // only keep tracking of matched submaps with smaller submap_id
    std::map<SubmapId, transform::Rigid3d> matched_submaps_5dof={};
    std::map<SubmapId, transform::Rigid3d> matched_submaps_6dof={};

    std::vector<std::pair<NodeId, TrajectoryNode>> nodes_in_submap={};
  };

  
  void ExtractFeaturesForSubmap(const SubmapId& submap_id) REQUIRES(mutex_);
  void ComputeConstraintsBetweenSubmaps(
      const SubmapId& submap_id_from) EXCLUDES(mutex_);
  void ComputeConstraintsForVisualLoop(
      const SE3SubmapConstraint& visual_loop) EXCLUDES(mutex_);
  
  void merge_nearby_scans(const NodeId& node_id, 
                          pcl::PointCloud<pcl::PointXYZ>::Ptr pc){
    const int k_nearby_scan_num_to_merge = 1;
    transform::Rigid3f local_pose_inv = 
      nodes_data_[node_id].constant_data->local_pose.inverse().cast<float>();

    for(int i=-k_nearby_scan_num_to_merge; i<k_nearby_scan_num_to_merge; ++i){
      NodeId ni;
      ni.trajectory_id = node_id.trajectory_id;
      ni.node_index = i+node_id.node_index;
      if(nodes_data_.find(ni) != nodes_data_.end()){
        sensor::PointCloud scan_in_pc = sensor::TransformPointCloud(
          nodes_data_[ni].constant_data->low_resolution_point_cloud, local_pose_inv);
        std::transform(scan_in_pc.begin(), scan_in_pc.end(), 
          std::back_inserter(pc->points), [](const Eigen::Vector3f& pt){
            return pcl::PointXYZ(pt[0], pt[1], pt[2]);
          });
      }
    }
  }

  bool ICPAlignment(const NodeId& src_id, const NodeId& tgt_id, 
               transform::Rigid3f& pose){
    // parse pointclouds
    int historyKeyframeSearchNum = 25;
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(
        new pcl::PointCloud<pcl::PointXYZ>());
    merge_nearby_scans(src_id, src);
    merge_nearby_scans(tgt_id, tgt); 

    // ICP Settings
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaxCorrespondenceDistance(150); 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(
        new pcl::PointCloud<pcl::PointXYZ>());
    icp.align(*unused_result);
 
    float fit_thresh = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > fit_thresh) {
      return false;
    } 
    auto relative_pose = icp.getFinalTransformation();
    
    Eigen::Matrix3f R = relative_pose.block(0,0,3,3);
    Eigen::Vector3f t = relative_pose.block(0,3,3,1);
    pose = transform::Rigid3f(t, Eigen::Quaternionf(R));
    return true;
  }

  void ComputeScanContextConstraints(
        const NodeId& node_id, 
        const SubmapId& node_submap_id,
        std::shared_ptr<const TrajectoryNode::Data> constant_data){
    // 缓存节点指针用于建立匹配约束
    nodes_data_[node_id] = InternalNodeData();
    nodes_data_[node_id].submap_id = node_submap_id;
    nodes_data_[node_id].constant_data = constant_data;
    scManager_.setSCdistThres(0.2);
    scManager_.setMaximumRadius(80);
    // 确认是否为关键帧，暂时用最Naive的逻辑
    if(node_id.node_index - last_keyframe_index_ > 10){
      // 若是关键帧，则调用Scan-Context检测回环
      pcl::PointCloud<pcl::PointXYZ>::Ptr pc(
        new pcl::PointCloud<pcl::PointXYZ>());
      std::transform(constant_data->low_resolution_point_cloud.begin(),
          constant_data->low_resolution_point_cloud.end(), 
          std::back_inserter(pc->points), 
          [](const Eigen::Vector3f& pt){
            return pcl::PointXYZ(pt[0], pt[1], pt[2]);
          });
      int scid = scManager_.makeAndSaveScancontextAndKeys(*pc);
      this->scid_nid_map_[scid] = node_id;
      auto detectResult = scManager_.detectLoopClosureID(); 
      int SCclosestHistoryFrameID = detectResult.first;

      // 计算回环约束
      if(SCclosestHistoryFrameID != -1 && 
          scManager_.polarcontext_vkeys_.size() - SCclosestHistoryFrameID > 50){ 
        
        const NodeId prev_node_idx = scid_nid_map_[SCclosestHistoryFrameID];
        LOG(WARNING)<<"Detected Loop by Scan Context!";
        
        transform::Rigid3f rel_pose;
        if(ICPAlignment(node_id, prev_node_idx, rel_pose)){
          constraints_.emplace_back();
          auto* const constraint = &constraints_.back();
          constraint->reset(new Constraint{            
            nodes_data_[prev_node_idx].submap_id,
            node_id,
            {rel_pose.cast<double>(), options_.loop_closure_translation_weight(),
            options_.loop_closure_rotation_weight()},
            Constraint::INTER_SUBMAP});
          LOG(WARNING)<<"Added Scan Context Constraint.";
        }
      }
      last_keyframe_index_ = node_id.node_index;
    }
   }

  // Runs in a background thread and does computations for an additional
  // constraint.
  // As output, it may create a new Constraint in 'constraint'.
  void ComputeConstraint(const SubmapId& submap_id, const NodeId& node_id,
                         /*submap_id the node belongs to*/
                         const SubmapId& node_submap_id,
                         std::unique_ptr<Constraint>* constraint)
      EXCLUDES(mutex_);

  void ComputeConstraint(const SubmapId& submap_id, 
                         const NodeId& node_id,
                         const SE3SubmapConstraint& visual_loop,
                         std::unique_ptr<Constraint>* constraint)
      EXCLUDES(mutex_);

  void RunWhenDoneCallback() EXCLUDES(mutex_);

  const proto::ConstraintBuilderOptions options_;
  common::ThreadPoolInterface* thread_pool_;
  common::Mutex mutex_;

  // 'callback' set by WhenDone().
  std::unique_ptr<std::function<void(const Result&)>> when_done_
      GUARDED_BY(mutex_);

  // TODO(gaschler): Use atomics instead of mutex to access these counters.
  // Number of the node in reaction to which computations are currently
  // added. This is always the number of nodes seen so far, even when older
  // nodes are matched against a new submap.
  int num_started_nodes_ GUARDED_BY(mutex_) = 0;

  int num_finished_nodes_ GUARDED_BY(mutex_) = 0;

  std::unique_ptr<common::Task> finish_node_task_ GUARDED_BY(mutex_);
  std::unique_ptr<common::Task> when_done_task_ GUARDED_BY(mutex_);
  std::vector<std::weak_ptr<common::Task>> tasks_tracker_ GUARDED_BY(mutex_);

  // Constraints currently being computed in the background. A deque is used to
  // keep pointers valid when adding more entries. Constraint search results
  // with below-threshold scores are also 'nullptr'.
  std::deque<std::unique_ptr<Constraint>> constraints_ GUARDED_BY(mutex_);

  std::map<SubmapId, std::set<NodeId>> computed_constraints_ GUARDED_BY(mutex_);

  // Map of dispatched or constructed scan matchers by 'submap_id'.
  std::map<SubmapId, SubmapScanMatcher> submap_scan_matchers_
      GUARDED_BY(mutex_);

  common::FixedRatioSampler sampler_;
  scan_matching::CeresScanMatcher3D ceres_scan_matcher_;

  // Histograms of scan matcher scores.
  common::Histogram score_histogram_ GUARDED_BY(mutex_);
  common::Histogram rotational_score_histogram_ GUARDED_BY(mutex_);
  common::Histogram low_resolution_score_histogram_ GUARDED_BY(mutex_);
  
  // added by wz
  const int min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> surf_detector_ 
    = cv::xfeatures2d::SURF::create(min_hessian);
  cv::Ptr<cv::DescriptorMatcher> surf_matcher_ 
    = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

  // For experiment only
  double sum_t_cost_ = 0.;
  double avg_t_cost_ = 0.;

  SC2::SCManager scManager_;
  std::map<int, NodeId> scid_nid_map_;
  size_t last_keyframe_index_ = 0;
  struct InternalNodeData{
    SubmapId submap_id;//submap_id the node belongs to.
    std::shared_ptr<const TrajectoryNode::Data> constant_data;
  };
  std::map<NodeId, InternalNodeData> nodes_data_;  
};

}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_3D_H_
