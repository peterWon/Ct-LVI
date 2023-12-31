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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_

#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/mutex.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/3d/submap_3d.h"
#include "cartographer/mapping/internal/constraints/constraint_builder_3d.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"
#include "cartographer/mapping/internal/trajectory_connectivity_state.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/pose_graph_trimmer.h"
#include "cartographer/sensor/fixed_frame_pose_data.h"
#include "cartographer/sensor/landmark_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/transform.h"
#include "cartographer/mapping/2d/probability_grid.h"
// #include "cartographer/mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d.h"
// #include "cartographer/mapping/proto/scan_matching/fast_correlative_scan_matcher_options_2d.pb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "cartographer/mapping/dbow/DBoW3.h"
#include "cartographer/mapping/dbow/DescManip.h"
#include "cartographer/mapping/clins/camera_models/Camera.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace cartographer {
namespace mapping {

// Implements the loop closure method called Sparse Pose Adjustment (SPA) from
// Konolige, Kurt, et al. "Efficient sparse pose adjustment for 2d mapping."
// Intelligent Robots and Systems (IROS), 2010 IEEE/RSJ International Conference
// on (pp. 22--29). IEEE, 2010.
//
// It is extended for submapping in 3D:
// Each node has been matched against one or more submaps (adding a constraint
// for each match), both poses of nodes and of submaps are to be optimized.
// All constraints are between a submap i and a node j.
struct MatchPair{
  int descriptor_id;
  Eigen::Vector2d uv_a;
  Eigen::Vector2d uv_b;
};

class PoseGraph3D : public PoseGraph {
 public:
  PoseGraph3D(
      const proto::PoseGraphOptions& options,
      std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem,
      common::ThreadPool* thread_pool);
  ~PoseGraph3D() override;

  PoseGraph3D(const PoseGraph3D&) = delete;
  PoseGraph3D& operator=(const PoseGraph3D&) = delete;

  // Adds a new node with 'constant_data'. Its 'constant_data->local_pose' was
  // determined by scan matching against 'insertion_submaps.front()' and the
  // node data was inserted into the 'insertion_submaps'. If
  // 'insertion_submaps.front().finished()' is 'true', data was inserted into
  // this submap for the last time.
  NodeId AddNode(
      std::shared_ptr<const TrajectoryNode::Data> constant_data,
      int trajectory_id,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps)
      EXCLUDES(mutex_);

  void AddImuData(int trajectory_id, const sensor::ImuData& imu_data) override
      EXCLUDES(mutex_);
  void AddImageData(int trajectory_id, const sensor::ImageFeatureData& img_data,
        const std::shared_ptr<Submap>& matching_submap) override EXCLUDES(mutex_);
  void AddOdometryData(int trajectory_id,
                       const sensor::OdometryData& odometry_data) override
      EXCLUDES(mutex_);
  void AddFixedFramePoseData(
      int trajectory_id,
      const sensor::FixedFramePoseData& fixed_frame_pose_data) override
      EXCLUDES(mutex_);
  void AddLandmarkData(int trajectory_id,
                       const sensor::LandmarkData& landmark_data) override
      EXCLUDES(mutex_);

  void FinishTrajectory(int trajectory_id) override;
  bool IsTrajectoryFinished(int trajectory_id) const override REQUIRES(mutex_);
  void FreezeTrajectory(int trajectory_id) override;
  bool IsTrajectoryFrozen(int trajectory_id) const override REQUIRES(mutex_);
  void AddSubmapFromProto(const transform::Rigid3d& global_submap_pose,
                          const proto::Submap& submap) override;
  void AddNodeFromProto(const transform::Rigid3d& global_pose,
                        const proto::Node& node) override;
  void SetTrajectoryDataFromProto(const proto::TrajectoryData& data) override;
  void AddNodeToSubmap(const NodeId& node_id,
                       const SubmapId& submap_id) override;
  void AddSerializedConstraints(
      const std::vector<Constraint>& constraints) override;
  void AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) override;
  void RunFinalOptimization() override;
  std::vector<std::vector<int>> GetConnectedTrajectories() const override;
  PoseGraph::SubmapData GetSubmapData(const SubmapId& submap_id) const
      EXCLUDES(mutex_) override;
  MapById<SubmapId, SubmapData> GetAllSubmapData() const
      EXCLUDES(mutex_) override;
  MapById<SubmapId, SubmapPose> GetAllSubmapPoses() const
      EXCLUDES(mutex_) override;
  transform::Rigid3d GetLocalToGlobalTransform(int trajectory_id) const
      EXCLUDES(mutex_) override;
  MapById<NodeId, TrajectoryNode> GetTrajectoryNodes() const override
      EXCLUDES(mutex_);
  MapById<NodeId, TrajectoryNodePose> GetTrajectoryNodePoses() const override
      EXCLUDES(mutex_);
  std::map<std::string, transform::Rigid3d> GetLandmarkPoses() const override
      EXCLUDES(mutex_);
  void SetLandmarkPose(const std::string& landmark_id,
                       const transform::Rigid3d& global_pose) override
      EXCLUDES(mutex_);
  sensor::MapByTime<sensor::ImuData> GetImuData() const override
      EXCLUDES(mutex_);
  sensor::MapByTime<sensor::OdometryData> GetOdometryData() const override
      EXCLUDES(mutex_);
  sensor::MapByTime<sensor::FixedFramePoseData> GetFixedFramePoseData()
      const override EXCLUDES(mutex_);
  std::map<std::string /* landmark ID */, PoseGraph::LandmarkNode>
  GetLandmarkNodes() const override EXCLUDES(mutex_);
  std::map<int, TrajectoryData> GetTrajectoryData() const override;

  std::vector<Constraint> constraints() const override EXCLUDES(mutex_);
  void SetInitialTrajectoryPose(int from_trajectory_id, int to_trajectory_id,
                                const transform::Rigid3d& pose,
                                const common::Time time) override
      EXCLUDES(mutex_);
  void SetGlobalSlamOptimizationCallback(
      PoseGraphInterface::GlobalSlamOptimizationCallback callback) override;
  transform::Rigid3d GetInterpolatedGlobalTrajectoryPose(
      int trajectory_id, const common::Time time) const REQUIRES(mutex_);

  transform::Rigid3d GetInterpolatedLocalTrajectoryPose(
      int trajectory_id, const common::Time time) const REQUIRES(mutex_);

 protected:
  // Waits until we caught up (i.e. nothing is waiting to be scheduled), and
  // all computations have finished.
  void WaitForAllComputations() EXCLUDES(mutex_);
  void AssociateDepthForVisualFeatures(
    const transform::Rigid3d& cam_pose_in_local,
    const std::vector<Eigen::Vector2f>& features_2d, 
    std::vector<Eigen::Vector3f>& features_3d);
  float Range(const pcl::PointXYZ& p){
      return std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);};
  void DrawDepthPointsAll(const cv::Mat& image_data, 
    const transform::Rigid3d& cam_pose_in_local);
  void GetColor(float p, float np, float&r, float&g, float&b);
 private:
  // The current state of the submap in the background threads. When this
  // transitions to kFinished, all nodes are tried to match against this submap.
  // Likewise, all new nodes are matched against submaps which are finished.
  enum class SubmapState { kActive, kFinished };
  struct InternalSubmapData {
    std::shared_ptr<const Submap3D> submap;

    // IDs of the nodes that were inserted into this map together with
    // constraints for them. They are not to be matched again when this submap
    // becomes 'finished'.
    std::set<NodeId> node_ids;

    SubmapState state = SubmapState::kActive;
    
    // wz: add for loop closure searching 
    cv::Mat prj_grid = cv::Mat();
    double ox, oy;
    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors = cv::Mat();
    // only keep tracking of matched submaps with smaller submap_id
    std::map<SubmapId, transform::Rigid3d> matched_submaps;
  };

  MapById<SubmapId, SubmapData> GetSubmapDataUnderLock() const REQUIRES(mutex_);
  
  SubmapId FindSubmapId(const std::shared_ptr<Submap>& submap) const REQUIRES(mutex_);
  // Handles a new work item.
  void AddWorkItem(const std::function<void()>& work_item) REQUIRES(mutex_);

  // Adds connectivity and sampler for a trajectory if it does not exist.
  void AddTrajectoryIfNeeded(int trajectory_id) REQUIRES(mutex_);

  // Grows the optimization problem to have an entry for every element of
  // 'insertion_submaps'. Returns the IDs for the 'insertion_submaps'.
  std::vector<SubmapId> InitializeGlobalSubmapPoses(
      int trajectory_id, const common::Time time,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps)
      REQUIRES(mutex_);

  // Adds constraints for a node, and starts scan matching in the background.
  void ComputeConstraintsForNode(
      const NodeId& node_id,
      std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
      bool newly_finished_submap) REQUIRES(mutex_);
      
  // Adds constraints for a node of image, and starts scan matching in the background.
  void ComputeConstraintsForImage(
      int trajectory_id, const sensor::ImageFeatureData& img_data,
      const std::shared_ptr<Submap>& matching_submap) REQUIRES(mutex_);
  
  // Find constraints for a newly finished submap.
  void ComputeConstraintsForSubmap(const SubmapId& submap_id) REQUIRES(mutex_);

  // Computes constraints for a node and submap pair.
  void ComputeConstraint(const NodeId& node_id, const SubmapId& submap_id, 
    const SubmapId& node_submap_id)
      REQUIRES(mutex_);
  void ComputeConstraint(const NodeId& node_id, const SubmapId& submap_id)
      REQUIRES(mutex_);

  // Adds constraints for older nodes whenever a new submap is finished.
  void ComputeConstraintsForOldNodes(const SubmapId& submap_id)
      REQUIRES(mutex_);

  // Runs the optimization, executes the trimmers and processes the work queue.
  void HandleWorkQueue(const constraints::ConstraintBuilder3D::Result& result)
      REQUIRES(mutex_);
  
  bool AllTaskFinished(){
    for(auto task = tasks_tracker_.begin(); task !=tasks_tracker_.end(); ){
      if(task->expired()){
        tasks_tracker_.erase(task);
      }else{
        return false;
      }
    }
    return true;
  }

  // Runs the optimization. Callers have to make sure, that there is only one
  // optimization being run at a time.
  void RunOptimization() EXCLUDES(mutex_);

  // Computes the local to global map frame transform based on the given
  // 'global_submap_poses'.
  transform::Rigid3d ComputeLocalToGlobalTransform(
      const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
      int trajectory_id) const REQUIRES(mutex_);

  PoseGraph::SubmapData GetSubmapDataUnderLock(const SubmapId& submap_id) const
      REQUIRES(mutex_);

  common::Time GetLatestNodeTime(const NodeId& node_id,
                                 const SubmapId& submap_id) const
      REQUIRES(mutex_);

  // Logs histograms for the translational and rotational residual of node
  // poses.
  void LogResidualHistograms() const REQUIRES(mutex_);

  // Updates the trajectory connectivity structure with a new constraint.
  void UpdateTrajectoryConnectivity(const Constraint& constraint)
      REQUIRES(mutex_);

  // Schedules optimization (i.e. loop closure) to run.
  void DispatchOptimization() REQUIRES(mutex_);


  const proto::PoseGraphOptions options_;
  GlobalSlamOptimizationCallback global_slam_optimization_callback_;
  mutable common::Mutex mutex_;

  // If it exists, further work items must be added to this queue, and will be
  // considered later.
  std::unique_ptr<std::deque<std::function<void()>>> work_queue_
      GUARDED_BY(mutex_);
  std::unique_ptr<common::Task> optimization_task_ GUARDED_BY(mutex_);
  std::vector<std::weak_ptr<common::Task>> tasks_tracker_ GUARDED_BY(mutex_);
  // How our various trajectories are related.
  TrajectoryConnectivityState trajectory_connectivity_state_;

  // We globally localize a fraction of the nodes from each trajectory.
  std::unordered_map<int, std::unique_ptr<common::FixedRatioSampler>>
      global_localization_samplers_ GUARDED_BY(mutex_);
  
  // Number of nodes added since last loop closure.
  int num_nodes_since_last_loop_closure_ GUARDED_BY(mutex_) = 0;
  
  // Whether the optimization has to be run before more data is added.
  bool run_loop_closure_ GUARDED_BY(mutex_) = false;
  
  // just for paper experiment
  double sum_t_cost_ GUARDED_BY(mutex_) = 0.0;

  // Current optimization problem.
  std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem_;
  constraints::ConstraintBuilder3D constraint_builder_ GUARDED_BY(mutex_);
  std::vector<Constraint> constraints_ GUARDED_BY(mutex_);

  // Submaps get assigned an ID and state as soon as they are seen, even
  // before they take part in the background computations.
  MapById<SubmapId, InternalSubmapData> submap_data_ GUARDED_BY(mutex_);
  
  // Data that are currently being shown. 
  MapById<NodeId, TrajectoryNode> trajectory_nodes_ GUARDED_BY(mutex_);
  int num_trajectory_nodes_ GUARDED_BY(mutex_) = 0;

  // Global submap poses currently used for displaying data.
  MapById<SubmapId, optimization::SubmapSpec3D> global_submap_poses_
      GUARDED_BY(mutex_);

  // Global landmark poses with all observations.
  std::map<std::string /* landmark ID */, PoseGraph::LandmarkNode>
      landmark_nodes_ GUARDED_BY(mutex_);

  transform::Rigid3d last_node_pose_to_find_constraint_;

  // List of all trimmers to consult when optimizations finish.
  std::vector<std::unique_ptr<PoseGraphTrimmer>> trimmers_ GUARDED_BY(mutex_);

  // Set of all frozen trajectories not being optimized.
  std::set<int> frozen_trajectories_ GUARDED_BY(mutex_);

  // Set of all finished trajectories.
  std::set<int> finished_trajectories_ GUARDED_BY(mutex_);

  // Set of all initial trajectory poses.
  std::map<int, InitialTrajectoryPose> initial_trajectory_poses_
      GUARDED_BY(mutex_);
  
  // For visual loop closure.
  Eigen::Matrix3d R_CtoI_;
  Eigen::Vector3d t_CtoI_;
  camodocal::CameraPtr camera_;
  std::unique_ptr<DBoW3::Vocabulary> voc_ = nullptr;
  std::unique_ptr<DBoW3::Database> db_ = nullptr;
  cv::Ptr<cv::Feature2D> fdetector_;

  struct ImageKeyframe{
    size_t id_;
    SubmapId submap_id_;
    common::Time time_;
    transform::Rigid3d local_pose_; //camera pose in local frame
    transform::Rigid3d global_pose_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    cv::Mat image_; // for debug only, to remove
  };
  size_t cur_image_kf_id_ = 0;
  size_t loop_kf_id_ = 0;
  std::vector<ImageKeyframe> img_keyframes_;
  std::vector<int> loop_kf_index_;
  // ceres::LocalParameterization* local_parameterization; 
          
  // pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_ds_;
  std::deque<pcl::PointCloud<pcl::PointXYZ>> depth_clouds_;
  pcl::VoxelGrid<pcl::PointXYZ> depth_filter_;
  std::vector<std::vector<pcl::PointXYZ>> pointsArray;
  const size_t num_bins_ = 360;
  double ceres_features_[1000][3];

  // Allows querying and manipulating the pose graph by the 'trimmers_'. The
  // 'mutex_' of the pose graph is held while this class is used.
  class TrimmingHandle : public Trimmable {
   public:
    TrimmingHandle(PoseGraph3D* parent);
    ~TrimmingHandle() override {}

    int num_submaps(int trajectory_id) const override;
    std::vector<SubmapId> GetSubmapIds(int trajectory_id) const override;
    MapById<SubmapId, SubmapData> GetOptimizedSubmapData() const override
        REQUIRES(parent_->mutex_);
    const MapById<NodeId, TrajectoryNode>& GetTrajectoryNodes() const override
        REQUIRES(parent_->mutex_);
    const std::vector<Constraint>& GetConstraints() const override
        REQUIRES(parent_->mutex_);
    void MarkSubmapAsTrimmed(const SubmapId& submap_id)
        REQUIRES(parent_->mutex_) override;
    bool IsFinished(int trajectory_id) const override REQUIRES(parent_->mutex_);

   private:
    PoseGraph3D* const parent_;
  };
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_
