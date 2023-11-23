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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_3D_LOCAL_TRAJECTORY_BUILDER_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_3D_LOCAL_TRAJECTORY_BUILDER_3D_H_

#include <chrono>
#include <memory>

#include "cartographer/common/time.h"
#include "cartographer/common/fixed_ratio_sampler.h"

//https://stackoverflow.com/questions/42504592/flann-util-serialization-h-class-stdunordered-mapunsigned-int-stdvectorun
#include "cartographer/mapping/clins/odometry/imu_state_estimator.h"
#include "cartographer/mapping/clins/odometry/inertial_initializer.h"
#include "cartographer/mapping/clins/trajectory/trajectory_manager.hpp"

#include "cartographer/mapping/3d/submap_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/ceres_scan_matcher_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/real_time_correlative_scan_matcher_3d.h"
#include "cartographer/mapping/internal/motion_filter.h"
#include "cartographer/mapping/internal/range_data_collator.h"
#include "cartographer/mapping/pose_extrapolator.h"
#include "cartographer/mapping/proto/3d/local_trajectory_builder_options_3d.pb.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/imu_data.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/sensor/image_data.h"
#include "cartographer/transform/rigid_transform.h"

#include "cartographer/mapping/internal/3d/range_data_synchronizer.h"

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
#include <pclomp/ndt_omp.h>

#include "cartographer/mapping/internal/3d/initialization/rigid3d_with_preintegrator.h"
#include "cartographer/mapping/internal/3d/initialization/imu_lidar_initializer.h"

#include "cartographer/mapping/msckf/msckf.h"
#include "cartographer/mapping/msckf/corner_detector.h"
#include "cartographer/mapping/cvins/parameters.h"
#include "cartographer/mapping/cvins/feature_tracker.h"
#include "cartographer/mapping/cvins/estimator_interface.h"
#include "cartographer/mapping/cvins/ct_estimator.h"
#include "cartographer/mapping/cvins/lvi_estimator.h"

namespace cartographer {
namespace mapping {

// Wires up the local SLAM stack (i.e. pose extrapolator, scan matching, etc.)
// without loop closure.

class LocalTrajectoryBuilder3D {
 public:
  struct InsertionResult {
    std::shared_ptr<const mapping::TrajectoryNode::Data> constant_data;
    std::vector<std::shared_ptr<const mapping::Submap3D>> insertion_submaps;
  };
  struct MatchingResult {
    common::Time time;
    transform::Rigid3d local_pose;
    sensor::RangeData range_data_in_local;
    // 'nullptr' if dropped by the motion filter.
    std::unique_ptr<const InsertionResult> insertion_result;
  };

  explicit LocalTrajectoryBuilder3D(
      const mapping::proto::LocalTrajectoryBuilderOptions3D& options,
      const std::vector<std::string>& expected_range_sensor_ids);
  ~LocalTrajectoryBuilder3D();

  LocalTrajectoryBuilder3D(const LocalTrajectoryBuilder3D&) = delete;
  LocalTrajectoryBuilder3D& operator=(const LocalTrajectoryBuilder3D&) = delete;

  void AddImuData(const sensor::ImuData& imu_data);
  // Returns 'MatchingResult' when range data accumulation completed,
  // otherwise 'nullptr'.  `TimedPointCloudData::time` is when the last point in
  // `range_data` was acquired, `TimedPointCloudData::ranges` contains the
  // relative time of point with respect to `TimedPointCloudData::time`.
  std::unique_ptr<MatchingResult> AddRangeData(
      const std::string& sensor_id,
      const sensor::TimedPointCloudData& range_data);
  void AddOdometryData(const sensor::OdometryData& odometry_data);
  void AddImageData(
    const sensor::ImageData& image_data, 
    std::shared_ptr<Submap>& matching_submap,
    sensor::ImageFeatureData& img_feature_data);

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);

private:
  /*****************************************************************/
  void InitializeStatic(double scan_time); 
  void GetColor(float p, float np, float&r, float&g, float&b);
  void MatchByICP(
    pcl::PointCloud<pcl::PointXYZI>::Ptr pre_scan,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan);
  void InitilizeByICP(const common::Time time,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan);
  void MatchByNDT(
    pcl::PointCloud<pcl::PointXYZI>::Ptr pre_scan,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan,
    const Eigen::Matrix4f& initial_guess,
    Eigen::Matrix3f& R, Eigen::Vector3f& t);
  void InitializeByNDT(double scan_time, 
    const sensor::TimedPointCloudOriginData& synchronized_data
    /* pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan */);
  bool AlignWithWorld();
  void InitCircularBuffers();
  
 private:
  pcl::PointCloud<pcl::PointXYZI>::Ptr cvtPointCloud(
    const cartographer::sensor::TimedPointCloudOriginData& cloud);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cvtPointCloudAndDiscrew(
    const cartographer::sensor::TimedPointCloudOriginData& cloud,
    const transform::Rigid3d& rel_trans);
  float Range(const pcl::PointXYZ& p){
    return std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);};
  std::unique_ptr<MatchingResult> AddAccumulatedRangeData(
      common::Time time,
      const transform::Rigid3d& pose_prediction,
      const sensor::RangeData& filtered_range_data_in_tracking);

  std::unique_ptr<InsertionResult> InsertIntoSubmap(
      common::Time time, const sensor::RangeData& filtered_range_data_in_local,
      const sensor::RangeData& filtered_range_data_in_tracking,
      const sensor::PointCloud& high_resolution_point_cloud_in_tracking,
      const sensor::PointCloud& low_resolution_point_cloud_in_tracking,
      const transform::Rigid3d& pose_estimate,
      const Eigen::Quaterniond& gravity_alignment);
  
  void InterpolatePose(
    const double s, 
    const transform::Rigid3d& relative_transform,
    transform::Rigid3d& pose_t);


  clins::PoseData ToPoseData(double t, const transform::Rigid3d& pose){
    clins::PoseData res;
    res.timestamp = t;
    res.position = pose.translation();
    res.orientation = clins::SO3d(pose.rotation());
    return res;
  }
  
  transform::Rigid3d ToRigid3d(const Sophus::SE3<double>& se3_pose){
    Eigen::Vector3d pos(se3_pose.translation());
    Eigen::Quaterniond rot(se3_pose.unit_quaternion());
    return  transform::Rigid3d(pos, rot);             
  }
  
  inline Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d& Vec) {
    // Returns skew-symmetric form of a 3-d vector
    Eigen::Matrix3d M;
    M << 0, -Vec(2), Vec(1),
      Vec(2), 0, -Vec(0),
      -Vec(1), Vec(0), 0;
    return M;
  }
  
  void AssociateDepthForVisualFeatures(
    const sensor::ImageData& image_data, 
    const transform::Rigid3d& cam_pose_in_local,
    const std::vector<Eigen::Vector2f>& features_2d, 
    std::vector<Eigen::Vector3f>& features_3d);
    
  void DrawDepthPointsAll(
    const sensor::ImageData& image_data, 
    const transform::Rigid3d& cam_pose_in_local);

  void DrawDepthPointsAssociated(
    const sensor::ImageData& image_data, 
    const std::vector<Eigen::Vector3f>& fetures_3d);
  
  void NdtOmpInit(double ndt_resolution);
  void SurfelFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud);
  int checkPlaneType(const Eigen::Vector3d& eigen_value,
                     const Eigen::Matrix3d& eigen_vector,
                     const double& p_lambda);
  bool fitPlane(const GPointCloud::Ptr& cloud,
                Eigen::Vector4d &coeffs,
                GPointCloud::Ptr cloud_inliers);

  const mapping::proto::LocalTrajectoryBuilderOptions3D options_;
  mapping::ActiveSubmaps3D active_submaps_;

  mapping::MotionFilter motion_filter_;
  std::unique_ptr<scan_matching::RealTimeCorrelativeScanMatcher3D>
      real_time_correlative_scan_matcher_;
  std::unique_ptr<scan_matching::CeresScanMatcher3D> ceres_scan_matcher_;

  std::unique_ptr<mapping::PoseExtrapolator> extrapolator_;

  int num_accumulated_ = 0;
  sensor::RangeData accumulated_range_data_;
  std::chrono::steady_clock::time_point accumulation_started_;

  // RangeDataCollator range_data_collator_;
  RangeDataSynchronizer range_data_synchronizer_;
  
  int debug_img_num = 0;
  int debug_scan_num = 0;
  int random_drop_num = 5;
/**************************************************************/
  double scan_period_;
  bool eable_mannually_discrew_;
  int frames_for_static_initialization_ = 7;
  int accumulated_frame_num = 0;
  common::Time time_point_cloud_;
  double last_imu_time_ini_ = -1.;
  bool system_initialized_ = false;
  bool first_scan_to_insert_ = true;

  //IMU initialization variables
  Eigen::Vector3d P_, V_, Ba_, Bg_;
  Eigen::Matrix3d R_;

  //For window optimization initialization
  int buffered_scan_count_ = 0;
  int frames_for_dynamic_initialization_ = 7;
  double last_stamp_ = -1.0;
  double dt_ = 0.1;
  pcl::PointCloud<pcl::PointXYZI>::Ptr last_scan_;
  Eigen::Vector3f linear_velocity_;
  Eigen::Vector3f angular_velocity_;
  
  std::deque<sensor::ImuData> init_imu_buffer_static_;
  sensor::ImuData last_imu_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_pre_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_cur_;
  pcl::ApproximateVoxelGrid<pcl::PointXYZI> voxel_filter_pre_;
  pcl::ApproximateVoxelGrid<pcl::PointXYZI> voxel_filter_cur_;
  pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt_;

  // Initilization by linear alignment
  std::shared_ptr<IntegrationBase> init_integrator_;
  // gtsam::imuBias::ConstantBias zero_imu_bias_;
  Eigen::Vector3d INIT_BA = Eigen::Vector3d::Zero();
  Eigen::Vector3d INIT_BW = Eigen::Vector3d::Zero();
  IMUNoise imu_noise_;
  
  // These variables should be with size of frames_for_dynamic_initialization_ + 1
  std::deque<Rigid3dWithVINSPreintegrator> all_laser_transforms_;
  std::deque<double> dynamic_init_stamps_;
  std::deque<double> static_init_stamps_;
  std::deque<Eigen::Vector3d> Ps_;
  std::deque<Eigen::Matrix3d> Rs_;
  std::deque<Eigen::Vector3d> Vs_;
  std::deque<Eigen::Vector3d> Bas_;
  std::deque<Eigen::Vector3d> Bgs_;

  Eigen::Vector3d g_vec_; // always be ~(0,0,-9.8)
  Eigen::Vector3d g_vec_est_B_; // in body(IMU) frame
  Eigen::Vector3d g_vec_est_G_; // in Global(ENU) frame
  transform::Rigid3d transform_lb_ = transform::Rigid3d::Identity();

  // for vio tracker
  // msckf_mono::MSCKF<float> msckf_;
  
  clins::CalibParamManager::Ptr calib_param_;
  clins::Trajectory<4>::Ptr trajectory_;
  std::shared_ptr<cvins::FeatureManager> feature_manager_vi_;
  std::shared_ptr<cvins::FeatureManager> feature_manager_lvi_;
  // cvins::CtEstimator::Ptr lvio_estimator_;
  cvins::LviEstimator::Ptr lvio_estimator_;
  cvins::ParameterServer* ps_;
  
  std::deque<double> depth_stamps_;
  std::deque<pcl::PointCloud<pcl::PointXYZ>> depth_clouds_;
  pcl::SACSegmentation<pcl::PointXYZ> seg_;

  // for depth debug only.
  std::deque<double> img_stamps_;
  std::deque<cv::Mat> images_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_ds_;
  pclomp::NormalDistributionsTransform<
      pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp_;

  pcl::VoxelGrid<pcl::PointXYZ> depth_filter_;
  std::vector<std::vector<pcl::PointXYZ>> pointsArray;

  std::shared_ptr<cartographer::common::FixedRatioSampler> image_sampler_;
  size_t current_tracked_id_max_;

  // for debug only, to remove
  bool last_scan_match_failed_ = false;
  std::shared_ptr<cvins::FeatureTracker> feature_tracker_;
  std::shared_ptr<cvins::VinsInterface> vins_;
  std::vector<cvins::State> vins_states_;
  std::thread vins_process_;

  // std::shared_ptr<corner_detector::TrackHandler> tracker_tmp_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_3D_LOCAL_TRAJECTORY_BUILDER_3D_H_
