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

#include "cartographer/mapping/internal/3d/local_trajectory_builder_3d.h"

#include <memory>

#include "cartographer/common/make_unique.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/3d/scan_matching/rotational_scan_matcher.h"
#include "cartographer/mapping/proto/3d/local_trajectory_builder_options_3d.pb.h"
#include "cartographer/mapping/clins/sensor_data/imu_data.h"



#include <chrono>   
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
using namespace std;
using namespace chrono;
static auto* kLocalSlamLatencyMetric = metrics::Gauge::Null();
static auto* kRealTimeCorrelativeScanMatcherScoreMetric =
    metrics::Histogram::Null();
static auto* kCeresScanMatcherCostMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualDistanceMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualAngleMetric = metrics::Histogram::Null();


LocalTrajectoryBuilder3D::LocalTrajectoryBuilder3D(
    const mapping::proto::LocalTrajectoryBuilderOptions3D& options,
    const std::vector<std::string>& expected_range_sensor_ids)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options.motion_filter_options()),
      real_time_correlative_scan_matcher_(
          common::make_unique<scan_matching::RealTimeCorrelativeScanMatcher3D>(
              options_.real_time_correlative_scan_matcher_options())),
      ceres_scan_matcher_(
          common::make_unique<scan_matching::CeresScanMatcher3D>(
              options_.ceres_scan_matcher_options())),
      accumulated_range_data_{Eigen::Vector3f::Zero(), {}, {}},
      range_data_synchronizer_(expected_range_sensor_ids) {
  
  scan_period_ = options_.scan_period();
  eable_mannually_discrew_ = options_.eable_mannually_discrew();
  frames_for_static_initialization_ = options_.frames_for_static_initialization();
  frames_for_dynamic_initialization_ = options_.frames_for_dynamic_initialization();
  // g_est_win_size_ = options_.frames_for_online_gravity_estimate();
  const float imuAccNoise = options_.imu_options().acc_noise();
  const float imuGyrNoise = options_.imu_options().gyr_noise();
  const float imuAccBiasN = options_.imu_options().acc_bias_noise();
  const float imuGyrBiasN = options_.imu_options().gyr_bias_noise();
  const float prior_pose_n = options_.imu_options().prior_pose_noise();
  const float prior_gravity_noise = options_.imu_options().prior_gravity_noise();
  const float ceres_pose_n_t = options_.imu_options().ceres_pose_noise_t();
  const float ceres_pose_n_r = options_.imu_options().ceres_pose_noise_r();
  const float ceres_pose_n_t_1 = 
    options_.imu_options().ceres_pose_noise_t_drift();
  const float ceres_pose_n_r_1 = 
    options_.imu_options().ceres_pose_noise_r_drift();
  
  // for vins initial integrator initialization
  imu_noise_.ACC_N = imuAccNoise;
  imu_noise_.ACC_W = imuAccBiasN;
  imu_noise_.GYR_N = imuGyrNoise;
  imu_noise_.GYR_W = imuGyrBiasN;
  init_integrator_.reset(new IntegrationBase(INIT_BA, INIT_BW, imu_noise_));
  
  InitCircularBuffers();
  
  std::string cfg_file = options_.lvio_config_filename();
  YAML::Node node = YAML::LoadFile(cfg_file);
  ps_ = new cvins::ParameterServer();
  ps_->readParameters(node);
  
  // tracker_tmp_.reset(new corner_detector::TrackHandler(cfg_file)); 
  
  // init vins feature tracker
  feature_tracker_.reset(new cvins::FeatureTracker(cfg_file));
  feature_tracker_->setMaxCount(ps_->MAX_CNT);
  if(!ps_->mask.empty()){
    cv::Mat mask_img = cv::imread(ps_->mask, CV_LOAD_IMAGE_GRAYSCALE);
    feature_tracker_->setFisheyeMask(mask_img);
  }
  
  feature_manager_vi_.reset(new cvins::FeatureManager(ps_));

  // vins estimator and running thread
  vins_.reset(new cvins::VinsInterface(ps_, feature_manager_vi_));
  vins_process_ = std::thread(&cvins::VinsInterface::Run, vins_.get());

  calib_param_ = std::make_shared<clins::CalibParamManager>(ps_);

  CHECK(ps_->knot_space > 0);
  trajectory_ = std::make_shared<clins::Trajectory<4>>(ps_->knot_space);
  trajectory_->SetCalibParam(calib_param_);
  
  // assign the same feature manager to LVI estimator.
  lvio_estimator_.reset(new cvins::LviEstimator(
      ps_, trajectory_, calib_param_, feature_manager_vi_));
  
  image_sampler_.reset(new common::FixedRatioSampler(
      ps_->image_sampling_ratio));
  
  depth_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  depth_cloud_ds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  
  pointsArray.resize(ps_->num_bins);
  for(int i = 0; i < ps_->num_bins; ++i)
    pointsArray[i].resize(ps_->num_bins);
  
  CHECK(ps_->seg_threshold > 0);
  seg_.setOptimizeCoefficients(true);
  seg_.setModelType(pcl::SACMODEL_PLANE);
  seg_.setMethodType(pcl::SAC_RANSAC);
  seg_.setDistanceThreshold(ps_->seg_threshold);
  current_tracked_id_max_ = 0;
  
  CHECK(ps_->ndt_resolution > 0);
  NdtOmpInit(ps_->ndt_resolution);
}


LocalTrajectoryBuilder3D::~LocalTrajectoryBuilder3D() {
  if(vins_process_.joinable()){
    vins_process_.join();
  }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LocalTrajectoryBuilder3D::cvtPointCloud(
    const cartographer::sensor::TimedPointCloudOriginData& point_cloud){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud;
  pcl_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
  pcl_cloud->resize(point_cloud.ranges.size());
  for (size_t i = 0; i < point_cloud.ranges.size(); ++i) {
    const auto& range = point_cloud.ranges.at(i);
    pcl_cloud->points[i].x = range.point_time[0];
    pcl_cloud->points[i].y = range.point_time[1];
    pcl_cloud->points[i].z = range.point_time[2];
    pcl_cloud->points[i].intensity =  range.point_time[3];
  }
  
  return pcl_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LocalTrajectoryBuilder3D::
    cvtPointCloudAndDiscrew(
      const cartographer::sensor::TimedPointCloudOriginData& point_cloud,
      const transform::Rigid3d& rel_trans
      /*rel_trans is the transform from last frame to current frame*/){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud;
  pcl_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
  pcl_cloud->resize(point_cloud.ranges.size());
  const double sample_period = scan_period_;
  transform::Rigid3d tracking_pose_in_start, tracking_pose_in_end;
  for (size_t i = 0; i < point_cloud.ranges.size(); ++i) {
    const auto& range = point_cloud.ranges.at(i);
    double s = (sample_period + range.point_time[3]) / sample_period;
    // pose in last frame
    InterpolatePose(s, rel_trans, tracking_pose_in_start);
    // transform to current frame
    tracking_pose_in_end = rel_trans.inverse() * tracking_pose_in_start;
    Eigen::Vector3d pt, pt_in_end;
    pt << range.point_time[0], range.point_time[1],range.point_time[2];
    pt_in_end = tracking_pose_in_end * pt;
    pcl_cloud->points[i].x = pt_in_end[0];
    pcl_cloud->points[i].y = pt_in_end[1];
    pcl_cloud->points[i].z = pt_in_end[2];
    pcl_cloud->points[i].intensity =  range.point_time[3];
  }
  
  return pcl_cloud;
}


void LocalTrajectoryBuilder3D::AddImuData(const sensor::ImuData& imu_data) {
  if(!system_initialized_){
    if(options_.enable_ndt_initialization() && init_integrator_){
      double imu_time = common::ToSecondsStamp(imu_data.time);
      double dt = (last_imu_time_ini_ < 0) 
          ? (1.0 / 500.0) : (imu_time - last_imu_time_ini_);
      last_imu_time_ini_ = imu_time;
      init_integrator_->push_back(
        dt, imu_data.linear_acceleration, imu_data.angular_velocity);
    }else{
      init_imu_buffer_static_.push_back(imu_data);
    }
  }
  
  // anyway, we add imu for lvio logic.
  double imu_time = common::ToSecondsStamp(imu_data.time);
  clins::IMUData data;
  data.timestamp = imu_time;
  data.gyro = Eigen::Vector3d(imu_data.angular_velocity.x(), 
                              imu_data.angular_velocity.y(),
                              imu_data.angular_velocity.z());
  data.accel = Eigen::Vector3d(imu_data.linear_acceleration.x(),
                              imu_data.linear_acceleration.y(),
                              imu_data.linear_acceleration.z());

  lvio_estimator_->AddIMUData(data);
  // vins_->AddImuData(imu_data);
}

void LocalTrajectoryBuilder3D::InitializeStatic(double scan_time){
  Eigen::Vector3d accel_accum;
  Eigen::Vector3d gyro_accum;
  int num_readings = 0;

  accel_accum.setZero();
  gyro_accum.setZero();

  for(const auto& entry : init_imu_buffer_static_){
    accel_accum += entry.linear_acceleration;
    gyro_accum += entry.angular_velocity;
    num_readings++;
  }
  Eigen::Vector3d accel_mean = accel_accum / num_readings;
  Eigen::Vector3d gyro_mean = gyro_accum / num_readings;
  
  g_vec_ << 0.0, 0.0, ps_->GRAVITY_NORM;
  P_.setZero();
  V_.setZero();
  //frame I to frame G
  R_ = Eigen::Quaternion<double>::FromTwoVectors(accel_mean, -g_vec_);

  // test codes, to remove
 /*  Eigen::Matrix3d I;
  I.setIdentity();
  Eigen::Vector3d accel_mean_norm = accel_mean.normalized();
  Eigen::Vector3d g_vec_norm = g_vec_.normalized();
  Eigen::Vector3d c = accel_mean_norm.cross(-g_vec_norm);
  Eigen::Matrix3d c_hat = vectorToSkewSymmetric(c);
  double s = (1.+accel_mean_norm.dot(g_vec_norm)) / (c.norm()*c.norm());

  Eigen::Matrix3d Rot = I + c_hat + c_hat * c_hat * s; */
  
  Ba_ = R_.transpose() * g_vec_ + accel_mean;
  Bg_ = gyro_mean;
  
  lvio_estimator_->Initialize(scan_time, Eigen::Quaterniond(R_), Ba_, Bg_);
  
  // std::deque<Eigen::Vector3d> Ps, Vs, Bas, Bgs;
  // std::deque<Eigen::Matrix3d> Rs;
  // for(int i = 0; i < static_init_stamps_.size(); ++i){
  //   Ps.push_back(Eigen::Vector3d(0., 0., 0.));
  //   Vs.push_back(Eigen::Vector3d(0., 0., 0.));
  //   Bas.push_back(Ba_);
  //   Bgs.push_back(Bg_);
  //   Rs.push_back(R_);
  // }
  // lvio_estimator_->InitializeDynamic(static_init_stamps_, Ps, Vs, Rs, Bas, Bgs);

  system_initialized_ = true;
  LOG(INFO)<<"P: "<<P_.transpose();
  LOG(INFO)<<"V: "<<V_.transpose();
  LOG(INFO)<<"Ba: "<<Ba_.transpose();
  LOG(INFO)<<"Bg: "<<Bg_.transpose();
  LOG(INFO)<<"R: "<<R_;
  LOG(INFO)<<"Static initialization finished successfully!";
}

void LocalTrajectoryBuilder3D::InitializeByNDT(
  double scan_time,
  const sensor::TimedPointCloudOriginData& synchronized_data){
  Rigid3dWithVINSPreintegrator plt;
  // double stamp = common::ToSecondsStamp(time);
  double stamp = scan_time;
  
  if(!last_scan_){
    if(last_imu_time_ini_ < 0) return; // still no imu come in
    last_stamp_ = stamp;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan = cvtPointCloud(
        synchronized_data);
    last_scan_ = cur_scan;
    linear_velocity_ << 0., 0., 0.;
    angular_velocity_ << 0., 0., 0.;
    INIT_BA << 0,0,0;
    INIT_BW << 0,0,0;
    voxel_filter_pre_.setLeafSize(0.2, 0.2, 0.2);
    voxel_filter_cur_.setLeafSize(0.2, 0.2, 0.2);
  
    // Setting scale dependent NDT parameters
    // Setting minimum transformation difference for termination condition.
    ndt_.setTransformationEpsilon(0.01);
    // Setting maximum step size for More-Thuente line search.
    ndt_.setStepSize(0.1);
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    ndt_.setResolution(1.0);

    // Setting max number of registration iterations.
    ndt_.setMaximumIterations(35);
    // plt.transform = transform::Rigid3d::Identity();
    // plt.pre_integration = nullptr;
    CHECK(buffered_scan_count_ == 0)
      << "buffered_scan_count_ must be zero here!";  
    all_laser_transforms_[buffered_scan_count_] = plt;
    dynamic_init_stamps_[buffered_scan_count_] = stamp;
    buffered_scan_count_++;
    init_integrator_->resetIntegration(INIT_BA, INIT_BW, imu_noise_);
    return;
  }else{
    dt_ = stamp - last_stamp_;
    
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    // set initial alignment estimate
    // Eigen::Quaternionf init_rotation_q =
    //     transform::AngleAxisVectorToRotationQuaternion(
    //         Eigen::Vector3f(angular_velocity_ * dt_)).normalized();
    Eigen::Quaternionf init_q = init_integrator_->deltaQij().cast<float>();
    Eigen::Matrix3f init_R(init_q);
    Eigen::Vector3f init_t(linear_velocity_ * dt_);
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    
    init_guess.block(0,0,3,3) = init_R;
    init_guess.block(0,3,3,1) = init_t;
    
    // discrewing points or not?
    // transform::Rigid3d rel_trans = transform::Rigid3d(
    //     linear_velocity_.cast<double>() * dt_, init_integrator_->delta_q);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan = cvtPointCloudAndDiscrew(
    //     synchronized_data, rel_trans);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan = cvtPointCloud(
        synchronized_data);
    MatchByNDT(last_scan_, cur_scan, init_guess, R, t);
    last_scan_ = cur_scan;
   
    // insert laser transform pair 
    plt.transform = transform::Rigid3d(t.cast<double>(),
      Eigen::Quaterniond(all_laser_transforms_[buffered_scan_count_ - 1]
        .transform.rotation().toRotationMatrix() 
        * R.cast<double>()));
    
    std::shared_ptr<IntegrationBase> imu_int_init_pt;
    imu_int_init_pt.reset(new IntegrationBase(*init_integrator_));
    plt.pre_integration = imu_int_init_pt;
  
    all_laser_transforms_[buffered_scan_count_] = plt;
    init_integrator_->resetIntegration(INIT_BA, INIT_BW, imu_noise_);
 
    // update velocity and stamp
    linear_velocity_ = t / dt_;
    angular_velocity_ = transform::RotationQuaternionToAngleAxisVector(
          Eigen::Quaternionf(R)) / dt_;
    last_stamp_ = stamp;
    dynamic_init_stamps_[buffered_scan_count_] = stamp;
    buffered_scan_count_++;
  }

  if(buffered_scan_count_ == frames_for_dynamic_initialization_ + 1){
    if(!AlignWithWorld()){
      LOG(ERROR)<<"Initialization failed! Perform re-initialization...";
      
      buffered_scan_count_ = 0;
      last_scan_ = nullptr;
      lvio_estimator_->ClearImuData();
      InitCircularBuffers();
    }else{
      // init succeeded, initialize the reference frame's state
      P_ = Ps_[frames_for_dynamic_initialization_];
      V_ = Vs_[frames_for_dynamic_initialization_];
      R_ = Rs_[frames_for_dynamic_initialization_];
      Ba_ = Bas_[frames_for_dynamic_initialization_];
      Bg_ = Bgs_[frames_for_dynamic_initialization_];
      
      lvio_estimator_->InitializeDynamic(
        dynamic_init_stamps_, Ps_, Vs_, Rs_, Bas_, Bgs_);
      // lvio_estimator_->Initialize(scan_time, Eigen::Quaterniond(R_), Ba_, Bg_);
      system_initialized_ = true;
      LOG(INFO)<<"Dynamic initialization finished successfully!";
    }
  }
}



std::unique_ptr<LocalTrajectoryBuilder3D::MatchingResult>
LocalTrajectoryBuilder3D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  sensor::TimedPointCloudOriginData synchronized_data 
    = range_data_synchronizer_.AddRangeData(
      sensor_id, unsynchronized_data, eable_mannually_discrew_);
  if (synchronized_data.ranges.empty()) {
    // LOG(INFO) << "Range data collator filling buffer.";
    return nullptr;
  }
  const common::Time& time = synchronized_data.time;
  time_point_cloud_ = time;
  
  CHECK(!synchronized_data.ranges.empty());
  CHECK_LE(synchronized_data.ranges.back().point_time[3], 0.1f);
  
  //cartographer默认以最后一个点作为timestamp, TO CHECK THIS
  const common::Time time_first_point = time 
    + common::FromSeconds(synchronized_data.ranges.front().point_time[3]);
  
  double scan_time = common::ToSecondsStamp(time_first_point);
  double scan_begin_time = scan_time;
  double scan_end_time = common::ToSecondsStamp(time);

  if(!system_initialized_){
    if(options_.enable_ndt_initialization()){
      InitializeByNDT(scan_time, synchronized_data);
    }else{
      static_init_stamps_.push_back(scan_time);
      if(accumulated_frame_num++ > frames_for_static_initialization_){
        InitializeStatic(scan_time);
      }
    }
    return nullptr;
  }


  if (scan_time < trajectory_->GetDataStartTime() ||
      trajectory_->GetDataStartTime() == 0) {
    LOG(WARNING) << "skip scan : " << scan_time;
    return nullptr;
  }
  
  // step1: Check msg
  double traj_start_t = trajectory_->GetDataStartTime();
  // step2: Integrate IMU measurements to initialize trajectory
  if (scan_begin_time < 0 || scan_end_time < 0) {
    LOG(ERROR) << "Somthing wrong?";
    return nullptr;//impossible?
  } else {
    lvio_estimator_->IntegrateIMUMeasurement(
      scan_begin_time - traj_start_t, scan_end_time - traj_start_t);
  }

  // step3: Undistort Scan
  if (num_accumulated_ == 0) {
    accumulation_started_ = std::chrono::steady_clock::now();
  }

  std::vector<sensor::TimedPointCloudOriginData::RangeMeasurement> hits =
      sensor::VoxelFilter(0.5f * options_.voxel_filter_size())
          .Filter(synchronized_data.ranges);
  
  std::vector<transform::Rigid3f> hits_poses;
  hits_poses.reserve(hits.size());

  //即使是插入子地图的第一帧，也是经过初始化步骤的，同样可以进行相对运动的矫正
  first_scan_to_insert_ = false; 
  if(first_scan_to_insert_){
    //若以最后一个点的时间戳为准，这个点在连续轨迹中的值有可能取不到，会存在误差
    auto se3_pose = trajectory_->GetIMUPose(scan_end_time - traj_start_t);
    hits_poses = std::vector<transform::Rigid3f>(
      hits.size(), ToRigid3d(se3_pose).cast<float>());
    first_scan_to_insert_ = false;
  }else{
    if(std::abs(hits.front().point_time[3]) < 1e-3){
      //激光雷达驱动没有提供单点的时间戳
      auto se3_pose = trajectory_->GetIMUPose(scan_end_time - traj_start_t);
      hits_poses = std::vector<transform::Rigid3f>(
        hits.size(), ToRigid3d(se3_pose).cast<float>());
      LOG(WARNING)<<"Not de-skewing!";
    }else{
      for (const auto& hit : hits) {
        //TODO(wz):确定最后的点的时间差有多大
        auto se3_pose = trajectory_->GetIMUPose(
          scan_end_time + hit.point_time[3] - traj_start_t);
        hits_poses.push_back(ToRigid3d(se3_pose).cast<float>());
      }
    }
  }

  if (num_accumulated_ == 0) {
    // 'accumulated_range_data_.origin' is not used.
    accumulated_range_data_ = sensor::RangeData{{}, {}, {}};
  }
          
  for (size_t i = 0; i < hits.size(); ++i) {
    const Eigen::Vector3f hit_in_local =
        hits_poses[i] * hits[i].point_time.head<3>();
    const Eigen::Vector3f origin_in_local =
        hits_poses[i] * synchronized_data.origins.at(hits[i].origin_index);
    const Eigen::Vector3f delta = hit_in_local - origin_in_local;
    const float range = delta.norm();
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit_in_local);
      } else {
        // We insert a ray cropped to 'max_range' as a miss for hits beyond the
        // maximum range. This way the free space up to the maximum range will
        // be updated.
        accumulated_range_data_.misses.push_back(
            origin_in_local + options_.max_range() / range * delta);
      }
    }
  }
  ++num_accumulated_;
  
  if (num_accumulated_ >= options_.num_accumulated_range_data()) {
    num_accumulated_ = 0;

    size_t ref_index = hits_poses.size() / 2;
    transform::Rigid3f current_pose = hits_poses.back();
    
    const sensor::RangeData filtered_range_data = {
        current_pose.translation(),
        sensor::VoxelFilter(options_.voxel_filter_size())
            .Filter(accumulated_range_data_.returns),
        sensor::VoxelFilter(options_.voxel_filter_size())
            .Filter(accumulated_range_data_.misses)};
  
   // return the raw data for Scan Context computing, todo, remove 
   const sensor::PointCloud raw_range_data_in_tracking = sensor::TransformRangeData(
      accumulated_range_data_, current_pose.inverse()).returns;
    // return AddAccumulatedRangeData(
    //   time, current_pose.cast<double>(), 
    //   sensor::TransformRangeData(filtered_range_data, current_pose.inverse()));

    auto pose_prediction = current_pose.cast<double>();
    auto filtered_range_data_in_tracking = sensor::TransformRangeData(
      filtered_range_data, current_pose.inverse());
    if (filtered_range_data_in_tracking.returns.empty()) {
      LOG(WARNING) << "Dropped empty range data.";
      return nullptr;
    }

    std::shared_ptr<const mapping::Submap3D> matching_submap =
        active_submaps_.submaps().front();
    transform::Rigid3d initial_ceres_pose =
        matching_submap->local_pose().inverse() * pose_prediction;
    sensor::AdaptiveVoxelFilter adaptive_voxel_filter(
        options_.high_resolution_adaptive_voxel_filter_options());
    const sensor::PointCloud high_resolution_point_cloud_in_tracking =
        adaptive_voxel_filter.Filter(filtered_range_data_in_tracking.returns);
    if (high_resolution_point_cloud_in_tracking.empty()) {
      LOG(WARNING) << "Dropped empty high resolution point cloud data.";
      return nullptr;
    }
    if (options_.use_online_correlative_scan_matching()) {
      // We take a copy since we use 'initial_ceres_pose' as an output argument.
      const transform::Rigid3d initial_pose = initial_ceres_pose;
      double score = real_time_correlative_scan_matcher_->Match(
          initial_pose, high_resolution_point_cloud_in_tracking,
          matching_submap->high_resolution_hybrid_grid(), &initial_ceres_pose);
      kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
    }

    transform::Rigid3d pose_observation_in_submap;
    ceres::Solver::Summary summary;

    sensor::AdaptiveVoxelFilter low_resolution_adaptive_voxel_filter(
        options_.low_resolution_adaptive_voxel_filter_options());
    const sensor::PointCloud low_resolution_point_cloud_in_tracking =
        low_resolution_adaptive_voxel_filter.Filter(
            filtered_range_data_in_tracking.returns);
    if (low_resolution_point_cloud_in_tracking.empty()) {
      LOG(WARNING) << "Dropped empty low resolution point cloud data.";
      return nullptr;
    }
    // TicToc tic;
    ceres_scan_matcher_->Match((matching_submap->local_pose().inverse() *
        pose_prediction).translation(),
        initial_ceres_pose,
        {{&high_resolution_point_cloud_in_tracking,
          &matching_submap->high_resolution_hybrid_grid()},
        {&low_resolution_point_cloud_in_tracking,
          &matching_submap->low_resolution_hybrid_grid()}},
        &pose_observation_in_submap, &summary);
    // about 0.003s
    // LOG(INFO) << "scan match cost " <<tic.toc()<<" seconds.";
   
    kCeresScanMatcherCostMetric->Observe(summary.final_cost);
    double residual_distance = (pose_observation_in_submap.translation() -
                                initial_ceres_pose.translation())
                                  .norm();
    kScanMatcherResidualDistanceMetric->Observe(residual_distance);
    double residual_angle = pose_observation_in_submap.rotation().angularDistance(
        initial_ceres_pose.rotation());
    
    kScanMatcherResidualAngleMetric->Observe(residual_angle);
    transform::Rigid3d pose_estimate =
        matching_submap->local_pose() * pose_observation_in_submap;
    
    // plane segmentation
    // tic.tic();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
    std::transform(low_resolution_point_cloud_in_tracking.begin(),
        low_resolution_point_cloud_in_tracking.end(), 
        std::back_inserter(cloud->points), 
        [](const Eigen::Vector3f& pt){
          return pcl::PointXYZ(pt[0], pt[1], pt[2]);
        });
    
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object
    seg_.setInputCloud(cloud);
    seg_.segment(*inliers, *coefficients);
    
    // sampling anchors to perform optimization
    std::vector<clins::PoseData> pose_anchors;
    
    size_t hit_pose_num = hits_poses.size();
    size_t step = hit_pose_num / ps_->num_anchor;
    auto pose_pre_inv = pose_prediction.inverse();
    for(int i = 0; i < ps_->num_anchor; i++){
      size_t idx = i * step;
      if(idx > hits_poses.size()){
        idx = hits_poses.size()-1;
      }
      pose_anchors.emplace_back(
        ToPoseData(hits[idx].point_time[3] + scan_end_time - traj_start_t, 
            pose_estimate * pose_pre_inv * (hits_poses[idx].cast<double>())));
    }
    
    if(!summary.IsSolutionUsable()){
      pose_anchors = {};
      last_scan_match_failed_ = true;
      LOG(WARNING)<<"Failed to perform scan matching!";
    }else{
      last_scan_match_failed_ = false;
    }
    
    float ratio = float(inliers->indices.size()) / float(cloud->size());
    if(ratio > ps_->seg_inlier_ratio){
      pose_anchors = {};
      last_scan_match_failed_ = true;
      LOG(WARNING)<<"Plane detected, scan matching may degenerated.";
    }else{
      last_scan_match_failed_ = false;
    }
    
    // to remove
    debug_scan_num++;
    if(debug_scan_num > 200 && debug_scan_num % 50 < ps_->random_drop_num){// 
      pose_anchors = {};
      last_scan_match_failed_ = true;
      LOG(WARNING)<<"Random drop this scan.";
    }else{
      last_scan_match_failed_ = false;
    }
    
    if(last_scan_match_failed_){
     
    }else{
      // TicToc tic;
      lvio_estimator_->AddLidarData(pose_anchors);
      if(ps_->optimize_lvi_together){
        lvio_estimator_->BuildProblemAndSolveLVI(30);  
      }else{
        lvio_estimator_->BuildProblemAndSolveLI(50);
      }
    }
    
    
    // after optimization, the position error and the rotation error are almost close to zeros.
    /* Eigen::Vector3d error_pos, error_rot;
    error_pos << 0,0,0;
    error_rot << 0,0,0;
    for(int i = 0; i<pose_anchors.size(); i++){
      auto pose = trajectory_->GetIMUPose(pose_anchors[i].timestamp);
      error_pos += pose.translation() - pose_anchors[i].position;
      error_rot += (pose.unit_quaternion().conjugate() * pose_anchors[i].orientation.unit_quaternion()).toRotationMatrix().eulerAngles(0,1,2);
    }
    error_pos /= pose_anchors.size();
    error_rot /= pose_anchors.size();
    LOG(INFO)<<"Err pos: "<<error_pos.transpose();
    LOG(INFO)<<"Err rot: "<<error_rot.transpose()*180./M_PI; */
    
    //using scan_end_time to assign data to submap and node
    auto opt_pose = this->ToRigid3d(
      trajectory_->GetIMUPose(scan_end_time - traj_start_t));
    transform::Rigid3f local_pose_tmp = opt_pose.cast<float>();

    const Eigen::Quaterniond gravity_alignment = opt_pose.rotation();
    sensor::RangeData filtered_range_data_in_local = sensor::TransformRangeData(
        filtered_range_data_in_tracking, local_pose_tmp);
  
    depth_stamps_.push_back(scan_time);
    depth_clouds_.emplace_back();

    std::transform(filtered_range_data_in_local.returns.begin(),
        filtered_range_data_in_local.returns.end(), 
        std::back_inserter(depth_clouds_.back().points), 
        [](const Eigen::Vector3f& pt){
          return pcl::PointXYZ(pt[0], pt[1], pt[2]);
        });
    // std::transform(high_resolution_point_cloud_in_tracking.begin(),
    //     high_resolution_point_cloud_in_tracking.end(), 
    //     std::back_inserter(depth_clouds_.back().points), 
    //     [&local_pose_tmp](const Eigen::Vector3f& pt){
    //       auto p = local_pose_tmp * pt;
    //       return pcl::PointXYZ(p[0], p[1], p[2]);
    //     });

    // update depth map for depth association of visual features.
    while (!depth_stamps_.empty()){
      if (scan_time - depth_stamps_.front() > 5.0){
        depth_clouds_.pop_front();
        depth_stamps_.pop_front();
      } else {
        break;
      }
    }
    
    // about 0.05s for 25 scans, about 150 surfels found with voxel size 1m.
    // pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(
    // new pcl::PointCloud<pcl::PointXYZ>());
    // for(int i = 0; i < depth_clouds_.size(); i+=2){
    //   *map_cloud += depth_clouds_.at(i);
    // }
    // if(!depth_clouds_.empty()){
    //   *map_cloud += depth_clouds_.back();
    //   SurfelFitting(map_cloud);
    // }
    

    // depth_cloud_->clear();
    // for(size_t i = 0; i < depth_clouds_.size(); ++i)
    //   *depth_cloud_ += depth_clouds_[i];

    // // downsample global cloud
    // depth_filter_.setLeafSize(0.2, 0.2, 0.2);
    // depth_filter_.setInputCloud(depth_cloud_);
    // depth_filter_.filter(*depth_cloud_ds_);
    // *depth_cloud_ = *depth_cloud_ds_;
    // LOG(INFO) << "update depth cloud cost " <<tic.toc()<<" seconds.";
  
    
    /* if(debug_img_num > 0 && debug_img_num < 500 && !images_.empty()){
      std::vector<cv::Point2f> points_2d;
      std::vector<float> points_distance;
      
      clins::SE3d cam_pose, cam_pose_inv;
      trajectory_->GetCameraPose(img_stamps_->front(), cam_pose);
      cam_pose_inv = cam_pose.inverse();
      int w = track_handler_->get_camera()->imageWidth();
      int h = track_handler_->get_camera()->imageHeight();
      for (size_t i = 0; i < depth_cloud_->size(); ++i){
        // convert points from 3D to 2D
        Eigen::Vector3d p_3d(depth_cloud_->points[i].x,
                            depth_cloud_->points[i].y,
                            depth_cloud_->points[i].z);
        
        Eigen::Vector3d p_3d_cam = cam_pose_inv * p_3d;
        Eigen::Vector2d p_2d;
        if(p_3d_cam[2] < 0) continue;

        track_handler_->get_camera()->spaceToPlane(p_3d_cam, p_2d);
        
        if(p_2d[0] < 0 || p_2d[1] < 0 || p_2d[0] > w-1 || p_2d[1] > h-1) continue;
        
        points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
        points_distance.push_back(p_3d_cam.norm());
      }

      cv::Mat showImage, circleImage;
      cv::cvtColor(images_.front(), showImage, cv::COLOR_GRAY2RGB);
      circleImage = showImage.clone();
      for (int i = 0; i < (int)points_2d.size(); ++i){
        float r, g, b;
        GetColor(points_distance[i], 50.0, r, g, b);
        cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
      }
      cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image
      cv::imwrite("/home/wz/Desktop/depth_images/"+std::to_string(debug_img_num)+".jpg", showImage);
    } */
    
    // return local slam result    
    std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
        time, filtered_range_data_in_local, filtered_range_data_in_tracking,
        high_resolution_point_cloud_in_tracking, raw_range_data_in_tracking
        /* low_resolution_point_cloud_in_tracking */, opt_pose, gravity_alignment);
    auto duration = std::chrono::steady_clock::now() - accumulation_started_;
    kLocalSlamLatencyMetric->Set(
        std::chrono::duration_cast<std::chrono::seconds>(duration).count());
    return common::make_unique<MatchingResult>(MatchingResult{
        time, opt_pose, std::move(filtered_range_data_in_local),
        std::move(insertion_result)});
  }
  return nullptr;
}

std::unique_ptr<LocalTrajectoryBuilder3D::MatchingResult>
LocalTrajectoryBuilder3D::AddAccumulatedRangeData(
    const common::Time time,
    const transform::Rigid3d& pose_prediction,
    const sensor::RangeData& filtered_range_data_in_tracking) {
  if (filtered_range_data_in_tracking.returns.empty()) {
    LOG(WARNING) << "Dropped empty range data.";
    return nullptr;
  }

  std::shared_ptr<const mapping::Submap3D> matching_submap =
      active_submaps_.submaps().front();
  transform::Rigid3d initial_ceres_pose =
      matching_submap->local_pose().inverse() * pose_prediction;
  sensor::AdaptiveVoxelFilter adaptive_voxel_filter(
      options_.high_resolution_adaptive_voxel_filter_options());
  const sensor::PointCloud high_resolution_point_cloud_in_tracking =
      adaptive_voxel_filter.Filter(filtered_range_data_in_tracking.returns);
  if (high_resolution_point_cloud_in_tracking.empty()) {
    LOG(WARNING) << "Dropped empty high resolution point cloud data.";
    return nullptr;
  }
  if (options_.use_online_correlative_scan_matching()) {
    // We take a copy since we use 'initial_ceres_pose' as an output argument.
    const transform::Rigid3d initial_pose = initial_ceres_pose;
    double score = real_time_correlative_scan_matcher_->Match(
        initial_pose, high_resolution_point_cloud_in_tracking,
        matching_submap->high_resolution_hybrid_grid(), &initial_ceres_pose);
    kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
  }

  transform::Rigid3d pose_observation_in_submap;
  ceres::Solver::Summary summary;

  sensor::AdaptiveVoxelFilter low_resolution_adaptive_voxel_filter(
      options_.low_resolution_adaptive_voxel_filter_options());
  const sensor::PointCloud low_resolution_point_cloud_in_tracking =
      low_resolution_adaptive_voxel_filter.Filter(
          filtered_range_data_in_tracking.returns);
  if (low_resolution_point_cloud_in_tracking.empty()) {
    LOG(WARNING) << "Dropped empty low resolution point cloud data.";
    return nullptr;
  }
  ceres_scan_matcher_->Match(
      (matching_submap->local_pose().inverse() * pose_prediction).translation(),
      initial_ceres_pose,
      {{&high_resolution_point_cloud_in_tracking,
        &matching_submap->high_resolution_hybrid_grid()},
       {&low_resolution_point_cloud_in_tracking,
        &matching_submap->low_resolution_hybrid_grid()}},
      &pose_observation_in_submap, &summary);
  kCeresScanMatcherCostMetric->Observe(summary.final_cost);
  double residual_distance = (pose_observation_in_submap.translation() -
                              initial_ceres_pose.translation())
                                 .norm();
  kScanMatcherResidualDistanceMetric->Observe(residual_distance);
  double residual_angle = pose_observation_in_submap.rotation().angularDistance(
      initial_ceres_pose.rotation());
  
  kScanMatcherResidualAngleMetric->Observe(residual_angle);
  transform::Rigid3d pose_estimate =
      matching_submap->local_pose() * pose_observation_in_submap;


  const Eigen::Quaterniond gravity_alignment = pose_estimate.rotation();
  sensor::RangeData filtered_range_data_in_local = sensor::TransformRangeData(
      filtered_range_data_in_tracking, pose_estimate.cast<float>());
  std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
      time, filtered_range_data_in_local, filtered_range_data_in_tracking,
      high_resolution_point_cloud_in_tracking,
      low_resolution_point_cloud_in_tracking, pose_estimate, gravity_alignment);
  auto duration = std::chrono::steady_clock::now() - accumulation_started_;
  kLocalSlamLatencyMetric->Set(
      std::chrono::duration_cast<std::chrono::seconds>(duration).count());
  return common::make_unique<MatchingResult>(MatchingResult{
      time, pose_estimate, std::move(filtered_range_data_in_local),
      std::move(insertion_result)});
}

void LocalTrajectoryBuilder3D::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator we cannot add odometry data.
    LOG(INFO) << "Extrapolator not yet initialized.";
    return;
  }
  extrapolator_->AddOdometryData(odometry_data);
}

void LocalTrajectoryBuilder3D::AssociateDepthForVisualFeatures(
    const sensor::ImageData& image_data, 
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
  float bin_res = 180.0 / (float)ps_->num_bins; 
  cv::Mat rangeImage = cv::Mat(
    ps_->num_bins, ps_->num_bins, CV_32F, cv::Scalar::all(FLT_MAX));
  
  transform::Rigid3f cam_pose_inv = cam_pose_in_local.inverse().cast<float>();
  int w = feature_tracker_->get_camera()->imageWidth();
  int h = feature_tracker_->get_camera()->imageHeight();
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
      feature_tracker_->get_camera()->spaceToPlane(p_3d_cam.cast<double>(), uv);
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
      if (row_id < 0 || row_id >= ps_->num_bins 
        || col_id < 0 || col_id >= ps_->num_bins)
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
  
  /*
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
    if (row_id < 0 || row_id >= ps_->num_bins 
        || col_id < 0 || col_id >= ps_->num_bins)
      continue;
    
    // retrieve points lie in a local spherical window.
    for(int r = -5; r < 5; ++r){
      for(int c = -5; c < 5; ++c){
        int ri = row_id + r;
        int ci = col_id + c;
        if (ri < 0 || ri >= ps_->num_bins 
          || ci < 0 || ci >= ps_->num_bins) continue;
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
      if(p_3d.norm() > lvio_estimator_->GetVisualDepthUpperBound() 
        || p_3d.norm() < lvio_estimator_->GetVisualDepthLowerBound()){
        continue;
      }
      features_3d[i] = p_3d;
      // LOG(INFO)<<features_3d[i].transpose();
    }
  } */ 


  // 4. filter invalid points from range image
  pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_local_filter(
      new pcl::PointCloud<pcl::PointXYZ>());
  for(int i = 0; i < ps_->num_bins; ++i){
    for(int j = 0; j < ps_->num_bins; ++j){
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
  
  // LOG(INFO)<<"depth_cloud size: "<<depth_cloud_->size();
  // LOG(INFO)<<"depth_cloud_unit_sphere size: "<<depth_cloud_unit_sphere->size();

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
      if(p_3d.norm() > lvio_estimator_->GetVisualDepthUpperBound() 
        || p_3d.norm() < lvio_estimator_->GetVisualDepthLowerBound()) continue;
      features_3d[i] = p_3d;
      // LOG(INFO)<<features_3d[i].transpose();
    }
  }
}

void LocalTrajectoryBuilder3D::DrawDepthPointsAll(
    const sensor::ImageData& image_data, 
    const transform::Rigid3d& cam_pose_in_local){
  if(debug_img_num++ < 600) return;
  if(debug_img_num > 800) return;

  auto local_pose_inv = cam_pose_in_local.inverse();
  std::vector<cv::Point2f> points_2d;
  std::vector<float> points_distance;
  int w = feature_tracker_->get_camera()->imageWidth();
  int h = feature_tracker_->get_camera()->imageHeight();
  for(const auto& pc: depth_clouds_){
    for(const pcl::PointXYZ& pt: pc.points){
      Eigen::Vector3d pl, p_3d_cam;
      pl << pt.x, pt.y, pt.z;
      p_3d_cam = local_pose_inv * pl;
      if(p_3d_cam[2] < 0.3) continue;
      Eigen::Vector2d p_2d;
      feature_tracker_->get_camera()->spaceToPlane(p_3d_cam, p_2d);
      if(p_2d[0] < 0 || p_2d[1] < 0 || p_2d[0] > w-1 || p_2d[1] > h-1) continue;

      points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
      points_distance.push_back(p_3d_cam.norm());
    }

    cv::Mat showImage, circleImage;
    cv::cvtColor(image_data.img, showImage, cv::COLOR_GRAY2RGB);
    circleImage = showImage.clone();
    for (int i = 0; i < (int)points_2d.size(); ++i){
      float r, g, b;
      GetColor(points_distance[i], 50.0, r, g, b);
      cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
    }
    cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image
    cv::imwrite("/home/wz/Desktop/depth_debug/"+std::to_string(debug_img_num)+".jpg", showImage);
    // cv::imshow("depth", showImage);
    // cv::waitKey(3);
  }
}

void LocalTrajectoryBuilder3D::DrawDepthPointsAssociated(
    const sensor::ImageData& image_data, 
    const std::vector<Eigen::Vector3f>& fetures_3d){
  // if(debug_img_num++ < 10) return;
  // if(debug_img_num > 300) return;

  std::vector<cv::Point2f> points_2d;
  std::vector<float> points_distance;
  int w = ps_->COL;
  int h = ps_->ROW;
  
  for(const auto& pt: fetures_3d){
    Eigen::Vector3d p_3d_cam;
    p_3d_cam << pt[0], pt[1], pt[2];
    if(p_3d_cam[2] < 0.3) continue;
    Eigen::Vector2d p_2d;
    feature_tracker_->get_camera()->spaceToPlane(p_3d_cam, p_2d);
    if(p_2d[0] < 0 || p_2d[1] < 0 || p_2d[0] > w-1 || p_2d[1] > h-1) continue;

    points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
    points_distance.push_back(p_3d_cam.norm());
  }

  cv::Mat showImage, circleImage;
  cv::cvtColor(image_data.img, showImage, cv::COLOR_GRAY2RGB);
  circleImage = showImage.clone();
  for (int i = 0; i < (int)points_2d.size(); ++i){
    float r, g, b;
    GetColor(points_distance[i], 50.0, r, g, b);
    cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
  }
  cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image
  // cv::imwrite("/home/wz/Desktop/depth_debug/"+std::to_string(debug_img_num)+".jpg", showImage);
  cv::imshow("depth", showImage);
  cv::waitKey(3);
}

void LocalTrajectoryBuilder3D::GetColor(
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

// VINS test
void LocalTrajectoryBuilder3D::AddImageData(
    const sensor::ImageData& image_data,
    std::shared_ptr<Submap>& matching_submap,
    sensor::ImageFeatureData& img_feature_data) {
  
  // LIO must be initialized first.
  if(!system_initialized_) return;
  
  double timestamp = common::ToSecondsStamp(image_data.time);
  if(timestamp < trajectory_->GetDataStartTime()) return;
  
  double t = timestamp - trajectory_->GetDataStartTime();
  
  // Drop to save computation.
  bool enabled = false;
  if (image_sampler_->Pulse()) {
    enabled = true;
  }
  TicToc tic_tracker;
  feature_tracker_->trackImage(image_data.img, timestamp, enabled);
  
  /*corner_detector::OutFeatureVector features1, features2;
  corner_detector::IdVector ids1, ids2;
  tracker_tmp_->set_current_image(image_data.img, timestamp);
  tracker_tmp_->tracked_features(features1, ids1);
  tracker_tmp_->new_features(features2, ids2);
  cv::Mat img_tracked = tracker_tmp_->get_track_image();
  
   current_tracked_id_max_++;
  if(current_tracked_id_max_ > 100 && current_tracked_id_max_ < 500){
    cv::imwrite("/home/wz/Desktop/track_debug/"+std::to_string(current_tracked_id_max_)+".jpg", image_data.img);
    cv::imwrite("/home/wz/Desktop/track_debug/"+std::to_string(current_tracked_id_max_)+"-track.jpg", img_tracked);
  }
  return; */

  if(!enabled) return;

  const auto& ids = feature_tracker_->current_ids();
  const auto& cur_features = feature_tracker_->current_un_pts();
  // feature_tracker_->drawTrackedPoints("filename");

  // Integrate IMU measurements to estimate camera pose.
  lvio_estimator_->IntegrateIMUForCamera(t);
  auto imu_state = 
      lvio_estimator_->GetCameraStateIntegrator()->GetLatestState();
  clins::SO3d q_ItoG = clins::SO3d(imu_state.q);
  clins::SE3d T_ItoG = clins::SE3d(q_ItoG, imu_state.p); 
  transform::Rigid3d pose_in_local = ToRigid3d(T_ItoG * calib_param_->se3_CtoI);
  
  cvins::VinsFrameFeature img_feature;
  img_feature.timestamp = timestamp;
  const auto& un_pts = feature_tracker_->current_un_pts();
  const auto& cur_pts = feature_tracker_->current_pts();
  const auto& pts_velocity = feature_tracker_->current_pts_velocity();
 
  std::vector<Eigen::Vector2f> features_2d = {};
  std::vector<Eigen::Vector3f> features_3d = {};
  std::transform(un_pts.begin(), un_pts.end(), 
    std::back_inserter(features_2d), [](const cv::Point2f& cv_pt){
      return Eigen::Vector2f(cv_pt.x, cv_pt.y);
    });
  
  if(ps_->enable_depth_assiciation){
    TicToc tic_ass;
    if(!last_scan_match_failed_){
      AssociateDepthForVisualFeatures(
        image_data, pose_in_local, features_2d, features_3d);
    }
    // LOG(INFO)<<"Association cost "<<tic_ass.toc();
    // DrawDepthPointsAll(image_data, pose_in_local);
    // DrawDepthPointsAssociated(image_data, features_3d);
  }
  
  for(int i = 0; i < ids.size(); ++i){
    cvins::VinsFeature pt;
    pt.feature_id = ids[i];
    pt.x = un_pts[i].x;
    pt.y = un_pts[i].y;
    pt.z = 1.;
    pt.u = cur_pts[i].x;
    pt.v = cur_pts[i].y;
    pt.velocity_x = pts_velocity[i].x;
    pt.velocity_y = pts_velocity[i].y;
    // if the point is successfully associated, its depth will be positive.
    if(ps_->enable_depth_assiciation){
      if(!last_scan_match_failed_){
        pt.depth = features_3d[i][2]; 
      }else{
        pt.depth = -1.; 
      }
    }else{
      pt.depth = -1.; 
    }
    
    img_feature.features.push_back(pt);
  }

  // TicToc tic_vio; 
  lvio_estimator_->AddCameraData(img_feature);
  if(ps_->optimize_lvi_together){
    if(last_scan_match_failed_){
      lvio_estimator_->BuildProblemAndSolveVI(30, true); 
    }
  }else{
    lvio_estimator_->BuildProblemAndSolveVI(30, last_scan_match_failed_);  
  }
  
  matching_submap = active_submaps_.submaps().front();
  img_feature_data.time = image_data.time;
  image_data.img.copyTo(img_feature_data.img);

  // return optimized camera pose
  clins::SE3d se3_pose;
  if(!trajectory_->GetCameraPose(t, se3_pose)) return;
  img_feature_data.pose_in_local = ToRigid3d(se3_pose);
  img_feature_data.features_id = ids;
  img_feature_data.features_uv = cur_pts;
}


std::unique_ptr<LocalTrajectoryBuilder3D::InsertionResult>
LocalTrajectoryBuilder3D::InsertIntoSubmap(
    const common::Time time,
    const sensor::RangeData& filtered_range_data_in_local,
    const sensor::RangeData& filtered_range_data_in_tracking,
    const sensor::PointCloud& high_resolution_point_cloud_in_tracking,
    const sensor::PointCloud& low_resolution_point_cloud_in_tracking,
    const transform::Rigid3d& pose_estimate,
    const Eigen::Quaterniond& gravity_alignment) {
  if (motion_filter_.IsSimilar(time, pose_estimate)) {
    return nullptr;
  }
  // Querying the active submaps must be done here before calling
  // InsertRangeData() since the queried values are valid for next insertion.
  std::vector<std::shared_ptr<const mapping::Submap3D>> insertion_submaps;
  for (const std::shared_ptr<mapping::Submap3D>& submap :
       active_submaps_.submaps()) {
    insertion_submaps.push_back(submap);
  }
  active_submaps_.InsertRangeData(filtered_range_data_in_local,
                                  gravity_alignment);
  const Eigen::VectorXf rotational_scan_matcher_histogram =
      scan_matching::RotationalScanMatcher::ComputeHistogram(
          sensor::TransformPointCloud(
              filtered_range_data_in_tracking.returns,
              transform::Rigid3f::Rotation(gravity_alignment.cast<float>())),
          options_.rotational_histogram_size());
  return common::make_unique<InsertionResult>(
      InsertionResult{std::make_shared<const mapping::TrajectoryNode::Data>(
                          mapping::TrajectoryNode::Data{
                              time,
                              gravity_alignment,
                              {},  // 'filtered_point_cloud' is only used in 2D.
                              high_resolution_point_cloud_in_tracking,
                              low_resolution_point_cloud_in_tracking,
                              rotational_scan_matcher_histogram,
                              pose_estimate}),
                      std::move(insertion_submaps)});
}

void LocalTrajectoryBuilder3D::RegisterMetrics(
    metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_internal_3d_local_trajectory_builder_latency",
      "Duration from first incoming point cloud in accumulation to local slam "
      "result");
  kLocalSlamLatencyMetric = latency->Add({});
  auto score_boundaries = metrics::Histogram::FixedWidth(0.05, 20);
  auto* scores = family_factory->NewHistogramFamily(
      "mapping_internal_3d_local_trajectory_builder_scores",
      "Local scan matcher scores", score_boundaries);
  kRealTimeCorrelativeScanMatcherScoreMetric =
      scores->Add({{"scan_matcher", "real_time_correlative"}});
  auto cost_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 100);
  auto* costs = family_factory->NewHistogramFamily(
      "mapping_internal_3d_local_trajectory_builder_costs",
      "Local scan matcher costs", cost_boundaries);
  kCeresScanMatcherCostMetric = costs->Add({{"scan_matcher", "ceres"}});
  auto distance_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 10);
  auto* residuals = family_factory->NewHistogramFamily(
      "mapping_internal_3d_local_trajectory_builder_residuals",
      "Local scan matcher residuals", distance_boundaries);
  kScanMatcherResidualDistanceMetric =
      residuals->Add({{"component", "distance"}});
  kScanMatcherResidualAngleMetric = residuals->Add({{"component", "angle"}});
}

/******************************************************************************/



void LocalTrajectoryBuilder3D::InterpolatePose(
    const double s, 
    const transform::Rigid3d& relative_transform,
    transform::Rigid3d& pose_t){
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity().slerp(
    s, relative_transform.rotation());
  Eigen::Vector3d t = s * relative_transform.translation();
  pose_t = transform::Rigid3d(t, q);
}



// void LocalTrajectoryBuilder3D::TrimStatesCache(const common::Time& time){
//   while(1){
//     if(predicted_states_.empty()) break;
//     if(common::ToSeconds(predicted_states_.front().first - time) < 0){
//       predicted_states_.pop_front();
//     }else{
//       break;
//     }
//   }
// }


void LocalTrajectoryBuilder3D::MatchByICP(
  pcl::PointCloud<pcl::PointXYZI>::Ptr pre_scan,
   pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan){
  // ICP Settings
  static pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setMaxCorrespondenceDistance(3);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  // Align clouds
  auto start = system_clock::now();
  icp.setInputSource(cur_scan);
  icp.setInputTarget(pre_scan);
  pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(
    new pcl::PointCloud<pcl::PointXYZI>());
  icp.align(*unused_result);
  auto end = system_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  LOG(WARNING) <<  "ICP Cost " << double(duration.count()) 
      * microseconds::period::num / microseconds::period::den << "s.";
  LOG(INFO)<<icp.hasConverged()<<", "<<icp.getFitnessScore();
  if (icp.hasConverged() == false || icp.getFitnessScore() > 0.3)
      return;
  // Get pose transformation
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame = icp.getFinalTransformation();
  float noiseScore = icp.getFitnessScore();
}

void LocalTrajectoryBuilder3D::MatchByNDT(
    pcl::PointCloud<pcl::PointXYZI>::Ptr pre_scan,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_scan,
    const Eigen::Matrix4f& init_guess,
    Eigen::Matrix3f& R, Eigen::Vector3f& t){
   // Filtering input scan to roughly 10% of original size 
   // to increase speed of registration.
  filtered_cloud_pre_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  voxel_filter_pre_.setInputCloud(pre_scan);
  voxel_filter_pre_.filter(*filtered_cloud_pre_);
  
  auto start = system_clock::now();
  filtered_cloud_cur_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  voxel_filter_cur_.setInputCloud (cur_scan);
  voxel_filter_cur_.filter (*filtered_cloud_cur_);

  // Setting point cloud to be aligned.
  ndt_.setInputSource(filtered_cloud_cur_);
  // Setting point cloud to be aligned to.
  ndt_.setInputTarget(filtered_cloud_pre_);

  // Calculating required rigid transform to align the input cloud 
  // to the target cloud.
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(
    new pcl::PointCloud<pcl::PointXYZI>);
  ndt_.align(*output_cloud, init_guess);

  auto trans = ndt_.getFinalTransformation();
  R = trans.block(0,0,3,3).cast<float>();
  t = trans.block(0,3,3,1).cast<float>();
  
  // lins::Transform transform_to_start;
  // transform_to_start = all_laser_transforms_[
  //     buffered_scan_count_-1].second.transform.rot.toRotationMatrix() * R;
  
  // *filtered_cloud_pre_ += *TransformPointCloudToStart(
  //   filtered_cloud_cur_, transform_to_start);
  auto end = system_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
}

bool LocalTrajectoryBuilder3D::AlignWithWorld() {
  // Check IMU observibility, adapted from VINS-mono
{
  Rigid3dWithVINSPreintegrator laser_trans_j;
  Eigen::Vector3d sum_g;

  for (size_t i = 0; i < frames_for_dynamic_initialization_; ++i) {
    laser_trans_j = all_laser_transforms_[i + 1];
    if(laser_trans_j.pre_integration == nullptr) continue;// first frame maybe nullptr
    double dt = laser_trans_j.pre_integration->deltaTij();
    Eigen::Vector3d tmp_g = laser_trans_j.pre_integration->deltaVij() / dt;
    sum_g += tmp_g;
  }

  Vector3d aver_g;
  aver_g = sum_g * 1.0 / (frames_for_dynamic_initialization_);
  double var = 0;

  for (size_t i = 0; i < frames_for_dynamic_initialization_; ++i) {
    laser_trans_j = all_laser_transforms_[i + 1];
    if(laser_trans_j.pre_integration == nullptr) continue;
    double dt = laser_trans_j.pre_integration->deltaTij();
    Eigen::Vector3d tmp_g = laser_trans_j.pre_integration->deltaVij() / dt;
    var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
  }

  var = sqrt(var / (frames_for_dynamic_initialization_));
  
  if (var < 0.25) {
    LOG(WARNING) << "IMU variation: " 
      << var << " is not sufficient, maybe using InitializeStatic instead.";
    return false;
  }
}
  //所有的Vs都在Laser坐标系下，此处的g所采用的坐标系参考为北东地
  Eigen::Vector3d g_vec_in_laser, g_vec_in_base;
  bool init_result = Initializer::Initialization(
    all_laser_transforms_, transform_lb_, 
    abs(ps_->GRAVITY_NORM), 
    Vs_, Bgs_, g_vec_in_laser);
  if (!init_result) {
    return false;
  }
  
  // update Position and Rotation
  for (size_t i = 0; i < frames_for_dynamic_initialization_ + 1; ++i) {
    const auto& trans_li = all_laser_transforms_[i].transform;
    auto trans_bi = trans_li * transform_lb_;
    Ps_[i] = trans_bi.translation();
    Rs_[i] = trans_bi.rotation().normalized().toRotationMatrix();
    Vs_[i] = Rs_[i] * Vs_[i];
  }
  
  g_vec_ << 0.0, 0.0, ps_->GRAVITY_NORM;
  
  // frame I to frame G
  // 对照Foster的预积分，感觉上VINS论文中（1）式中的重力项差了一个负号，
  // 否则静止时该式不成立（IMU是敏感不到重力的）
  // 所以这里所求出来的重力向量要取反
  g_vec_in_base = -g_vec_in_laser;
  R_ = Eigen::Quaternion<double>::FromTwoVectors(
    g_vec_in_base.normalized(), g_vec_.normalized());
  Eigen::Vector3d g_est = R_ * g_vec_in_base;
  LOG(WARNING) << "g_est: " << g_est[0] << "," << g_est[1] << "," << g_est[2];
  LOG(WARNING) << "g_vec_in_base: " << g_vec_in_base[0] <<","
               << g_vec_in_base[1] << "," << g_vec_in_base[2];

  // Align with world frame
  for (int i = 0; i < frames_for_dynamic_initialization_ + 1; i++) {
    Ps_[i] = (R_ * Ps_[i]).eval();
    Rs_[i] = (R_ * Rs_[i]).eval();
    Vs_[i] = (R_ * Vs_[i]).eval();
  }

  LOG(WARNING) << "Imu initialization successful!";
  return true;
}

void LocalTrajectoryBuilder3D::InitCircularBuffers(){
  Eigen::Vector3d zero_vec(0., 0., 0.);
  size_t window_size = frames_for_dynamic_initialization_ + 1;
  all_laser_transforms_.resize(window_size);
  Ps_.resize(window_size);
  Rs_.resize(window_size);
  Vs_.resize(window_size);
  Bas_.resize(window_size);
  Bgs_.resize(window_size);
  dynamic_init_stamps_.resize(window_size);
  for(size_t i = 0; i < window_size; ++i){
    all_laser_transforms_[i] = Rigid3dWithVINSPreintegrator();
    Ps_[i] = zero_vec;
    Rs_[i] = Eigen::Matrix3d::Identity();
    Vs_[i] = zero_vec;
    Bas_[i] = zero_vec;
    Bgs_[i] = zero_vec;
  }
}

void LocalTrajectoryBuilder3D::NdtOmpInit(double ndt_resolution) {
  ndt_omp_ = pclomp::NormalDistributionsTransform<
      pcl::PointXYZ, pcl::PointXYZ>::Ptr(
          new pclomp::NormalDistributionsTransform<
                    pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp_->setResolution(ndt_resolution);
  ndt_omp_->setNumThreads(4);
  ndt_omp_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  ndt_omp_->setTransformationEpsilon(1e-3);
  ndt_omp_->setStepSize(0.01);
  ndt_omp_->setMaximumIterations(50);
}

void LocalTrajectoryBuilder3D::SurfelFitting(
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud){
  TicToc tic;
  if(!map_cloud) return;
  ndt_omp_->setInputTarget(map_cloud);

  Eigen::Vector3i counter(0,0,0);
  int plane_num = 0;
  for (const auto &v : ndt_omp_->getTargetCells().getLeaves()) {
    auto leaf = v.second;

    if (leaf.nr_points < 10)
      continue;
    double p_lambda_ = 0.7;
    int plane_type = checkPlaneType(leaf.getEvals(), leaf.getEvecs(), p_lambda_);
    if (plane_type < 0)
      continue;

    Eigen::Vector4d surfCoeff;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(
        new pcl::PointCloud<pcl::PointXYZ>());
    if (!fitPlane(leaf.pointList_.makeShared(), surfCoeff, cloud_inliers))
      continue;
    plane_num++;
    // counter(plane_type) += 1;
    // SurfelPlane surfplane;
    // surfplane.cloud = leaf.pointList_;
    // surfplane.cloud_inlier = *cloud_inliers;
    // surfplane.p4 = surfCoeff;
    // surfplane.Pi = -surfCoeff(3) * surfCoeff.head<3>();

    // GPoint min, max;
    // pcl::getMinMax3D(surfplane.cloud, min, max);
    // surfplane.boxMin = Eigen::Vector3d(min.x, min.y, min.z);
    // surfplane.boxMax = Eigen::Vector3d(max.x, max.y, max.z);

    // surfel_planes_.push_back(surfplane);
  }
  LOG(INFO)<<"Sufel mapping cost: "<<tic.toc()<<" of "
    <<depth_clouds_.size()<< "submaps, find " <<plane_num;
}


bool LocalTrajectoryBuilder3D::fitPlane(const GPointCloud::Ptr& cloud,
                                 Eigen::Vector4d &coeffs,
                                 GPointCloud::Ptr cloud_inliers) {
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::SACSegmentation<GPoint> seg;    /// Create the segmentation object
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.05);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () < 20) {
    return false;
  }

  for(int i = 0; i < 4; i++) {
    coeffs(i) = coefficients->values[i];
  }

  pcl::copyPointCloud<GPoint> (*cloud, *inliers, *cloud_inliers);
  return true;
}


int LocalTrajectoryBuilder3D::checkPlaneType(const Eigen::Vector3d& eigen_value,
                                      const Eigen::Matrix3d& eigen_vector,
                                      const double& p_lambda) {
  Eigen::Vector3d sorted_vec;
  Eigen::Vector3i ind;
  Eigen::sort_vec(eigen_value, sorted_vec, ind);
  //公式13，eigen_value由ndt提供
  double p = 2*(sorted_vec[1] - sorted_vec[2]) /
             (sorted_vec[2] + sorted_vec[1] + sorted_vec[0]);

  if (p < p_lambda) {
    return -1;
  }

  int min_idx = ind[2];
  Eigen::Vector3d plane_normal = eigen_vector.block<3,1>(0, min_idx);
  plane_normal = plane_normal.array().abs();

  Eigen::sort_vec(plane_normal, sorted_vec, ind);
  return ind[2];
}
/* 
pcl::PointCloud<pcl::PointXYZI>::Ptr LocalTrajectoryBuilder3D::
    TransformPointCloudToStart(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn, 
                               const lins::Transform& tsf) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(
    new pcl::PointCloud<pcl::PointXYZI>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
  Eigen::Affine3f transCur = tsf.transform();
  #pragma omp parallel for num_threads(4)
  for (int i = 0; i < cloudSize; ++i){
    const auto &pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) 
        * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
    cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) 
        * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
    cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) 
        * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }
  return cloudOut;
} */

}  // namespace mapping
}  // namespace cartographer
