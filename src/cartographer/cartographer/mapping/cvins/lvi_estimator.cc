
#include "lvi_estimator.h"
#include "glog/logging.h"

namespace cvins{

LviEstimator::LviEstimator(const ParameterServer* ps,
   std::shared_ptr<TrajectoryN> trajectory,
   std::shared_ptr<CalibParamManager> calib_param,
   std::shared_ptr<FeatureManager> feature_manager): 
      ps_(ps),
      trajectory_(trajectory),
      calib_param_(calib_param),
      feature_manager_(feature_manager){
  imu_state_estimator_.reset(new clins::ImuStateEstimator(ps));
  imu_integrator_for_camera_.reset(new clins::ImuStateEstimator(ps));
  clearState();
}

void LviEstimator::AddCameraData(const VinsFrameFeature& image){
  if(ps_->optimize_lvi_together && frame_count_==WINDOW_SIZE){
    slideWindow();
  }

  // transform feature
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> pts;
  for (unsigned int i = 0; i < image.features.size(); i++){
    const VinsFeature vf = image.features.at(i);

    Eigen::Matrix<double, 8, 1> xyz_uv_velocity_d;
    xyz_uv_velocity_d << vf.x, vf.y, vf.z, vf.u, vf.v, 
                        vf.velocity_x, vf.velocity_y, vf.depth;
    pts[vf.feature_id].emplace_back(camera_id_, xyz_uv_velocity_d);
  }
  
  double header = image.timestamp;
  Headers[frame_count_] = header;

  if(!first_img) first_img = true;

  ImageFrame imageframe(pts, header);
  imageframe.pre_integration = tmp_pre_integration;
  all_image_frame.insert(make_pair(header, imageframe));
  tmp_pre_integration = new IntegrationBase{
    acc_0, gyr_0, Bas[frame_count_], Bgs[frame_count_],
    ps_->G, ps_->ACC_N, ps_->GYR_N, ps_->ACC_W, ps_->GYR_W};

  // add features to manager
  if (feature_manager_->addFeatureCheckParallax(frame_count_, pts, td_, header))
    marg_flag_ = MARGIN_OLD;
  else
    marg_flag_ = MARGIN_SECOND_NEW;
  
  
  // when call this function, the LIO trajectory has been initialized,
  // we can triangulate features which has no depth associated 
  // or that failed to estimate in optimization.
  if(frame_count_ == WINDOW_SIZE){
    for(int i = 0; i < WINDOW_SIZE + 1; i++){
      SE3d pose_imu = trajectory_->GetIMUPose(
        Headers[i] - trajectory_->GetDataStartTime());
      Rs[i] = pose_imu.unit_quaternion().toRotationMatrix();
      Ps[i] = pose_imu.translation();
    }
    feature_manager_->triangulate(Rs, Ps, &(ps_->TIC[0]), &(ps_->RIC[0]));
  }

  if(ps_->optimize_lvi_together && frame_count_ < WINDOW_SIZE){
    frame_count_++;
  }
}

void LviEstimator::InitIMUData(double feature_time) {
  double traj_start_time = feature_time;
  
  for (int i = imu_data_.size() - 1; i >= 0; i--) {
    if (imu_data_[i].timestamp < feature_time) {
      // LOG(INFO)<<i<<","<<imu_data_[i].timestamp;
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

  LOG(WARNING)<<"IMU data size: "<<imu_data_.size()<<", "<<traj_start_time;
  trajectory_->setDataStartTime(traj_start_time);
  imu_state_estimator_->SetFirstIMU(imu_data_.front());
  imu_integrator_for_camera_->SetFirstIMU(imu_data_.front());
  for (size_t i = 1; i < imu_data_.size(); i++) {
    imu_state_estimator_->FeedIMUData(imu_data_[i]);
    imu_integrator_for_camera_->FeedIMUData(imu_data_[i]);
  }
  trajectory_init_ = true;
}

void LviEstimator::ErasePastImuData(double t) {
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < t) {
      iter = imu_data_.erase(iter);
    } else {
      ++iter;
    }
  }
}

void LviEstimator::AddStartTimePose(
    std::shared_ptr<TrajectoryEstimator<4>> estimator) {
  size_t kont_idx = trajectory_->computeTIndex(active_time_lower_).second;
  if (kont_idx < 4) {
    init_pose.timestamp = trajectory_->minTime();

    double rot_weight = 1000;
    double pos_weight = 1000;
    estimator->AddPoseMeasurement(init_pose, rot_weight, pos_weight);
  }
}


void LviEstimator::AddLiDARErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator){
  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;
  for(const auto& lp: lidar_anchors_){
    estimator->AddPoseMeasurement(lp, rot_weight, pos_weight);
  }
}

void LviEstimator::LogVisualError(){
  int f_m_cnt = 0;
  int feature_index = -1;
  double t0 = trajectory_->GetDataStartTime();
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    ++feature_index;
    
    // 共视点在i时刻坐标系下的归一化坐标
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double t_i = it_per_id.feature_per_frame[0].timestamp_;
    SE3d pose_i, pose_j;
    if(!trajectory_->GetCameraPose(t_i - t0, pose_i)) continue;
    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      if (imu_i == imu_j){
        continue;
      }
      // 共视点在j时刻坐标系下的归一化坐标
      Vector3d pts_j = it_per_frame.point;
      double t_j = it_per_frame.timestamp_;
      if(!trajectory_->GetCameraPose(t_j - t0, pose_j)) continue;
      Eigen::Vector3d pt_i_3d = pts_i * it_per_id.estimated_depth;
      Eigen::Vector3d pt_j_3d = pose_j.inverse() * pose_i * pt_i_3d;
      LOG(INFO)<<"Reproject error: "<<pt_j_3d[0]/pt_j_3d[2]-pts_j[0]
                                    <<" "<<pt_j_3d[1]/pt_j_3d[2]-pts_j[1];
      f_m_cnt++;
    }
  }
}

// void LviEstimator::AddCameraErrorTermsNoFeature(
//       std::shared_ptr<TrajectoryEstimator<4>> estimator){
//   int f_m_cnt = 0;
//   int f_m_cnt_true = 0;
//   int feature_index = -1;
//   double weight = calib_param_->global_opt_cam_uv_weight;
//   double t0 = trajectory_->GetDataStartTime();
//   Vector2d uv_var(1.5, 1.5);

//   //对每一个feature会有一个对应的雅克比矩阵和残差向量(2m,m为观察到该特征的相机数量)
//   for (auto &it_per_id : feature_manager_->feature){
//     it_per_id.used_num = it_per_id.feature_per_frame.size();
//     if (!(it_per_id.used_num >= 4 && it_per_id.start_frame < WINDOW_SIZE - 2))
//       continue;

//     ++feature_index;
    
//     MatrixXd H_f_j = MatrixXd::Zero(2 * it_per_id.used_num, 3); 
//     MatrixXd H_x_j = MatrixXd::Zero(2 * it_per_id.used_num, 6 * WINDOW_SIZE);
//     MatrixXd R_j = (uv_var.replicate(it_per_id.used_num, 1)).asDiagonal();

//     VectorXd r_j = VectorXd::Constant(2 * it_per_id.used_num,
//                        std::numeric_limits<_S>::quiet_NaN());
//     // 共视点在i时刻坐标系下的归一化坐标
//     int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
//     Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    
//     // get currently estimated point in world frame.
//     double t_i = it_per_id.feature_per_frame[0].timestamp_;
//     Eigen::Vector3d p_f_ci, p_f_G;
//     SE3d pose_ci;
//     if(!trajectory_->GetCameraPose(t_i - t0, pose_ci)) continue;
//     p_f_G = pose_ci * p_f_ci;

//     for (auto &it_per_frame : it_per_id.feature_per_frame){
//       imu_j++;
//       int index = imu_j - it_per_id.start_frame;
//       int cam_state_idx = imu_j;
//       CHECK(cam_state_idx >= 0 && cam_state_idx < WINDOW_SIZE);
      
//       // 共视点在j时刻坐标系下的归一化坐标
//       Eigen::Vector3d pts_j = it_per_frame.point;
//       double t_j = it_per_frame.timestamp_;
//       f_m_cnt++;

//       Eigen::Vector3d p_f_cj;
//       SE3d pose_cj;
//       if(!trajectory_->GetCameraPose(t_j - t0, pose_cj)) continue;
//       Eigen::Matrix3d R_G_cj = 
//           pose_cj.unit_quaternion().conjugate().toRotationMatrix();
//       p_f_cj = R_G_cj * (p_f_G - pose_cj.translation());
      
//       double X, Y, Z;
//       X = p_f_cj(0);
//       Y = p_f_cj(1);
//       Z = p_f_cj(2);
      
//       Eigen::Matrix<double, 2, 3> A;
//       A << 1, 0, -X / Z, 0, 1, -Y / Z;
//       A *= 1 / Z;
      
//       Eigen::Vector2d residual;
//       residual << X / Z - pts_j(0)/pts_j(2), Y / Z - pts_j(1)/pts_j(2);

//       Eigen::Matrix<double, 2, 6> J_x;
//       J_x.block(0, 0, 2, 3) = A * vectorToSkewSymmetric(-p_f_cj);
//       J_x.block(0, 3, 2, 3) = -A * R_G_cj;
      
//       Eigen::Matrix<double, 2, 3> J_f = A * R_G_cj;
      
//       H_f_j.template block<2, 3>(2 * index, 0) = J_f;
//       H_x_j.template block<2, 6>(2 * index, 6 * cam_state_idx) = J_x;
//       r_j.template block<2, 1>(2 * index, 0) = residual;
//     }

//     // marginalize feature.
//     int jacobian_row_size = 2 * it_per_id.used_num;
//     Eigen::JacobiSVD<MatrixXd> svd_helper(
//         H_f_j, Eigen::ComputeFullU | Eigen::ComputeThinV);
//     auto A_j = svd_helper.matrixU().rightCols(jacobian_row_size - 3);
//     auto H_o_j = A_j.transpose() * H_x_j;
//     auto r_o_j = A_j.transpose() * r_j;
//     auto R_o_j = A_j.transpose() * R_j * A_j;

//     /* HouseholderQR<MatrixXd> qr(H_o);
//     MatrixXd Q = qr.householderQ();
//     MatrixXd R = qr.matrixQR().template triangularView<Upper>();

//     VectorXd nonZeroRows = R.rowwise().any();
//     int numNonZeroRows = nonZeroRows.sum();

//     MatrixXd T_H = MatrixXd::Zero(numNonZeroRows, R.cols());
//     MatrixXd Q_1 = MatrixXd::Zero(Q.rows(), numNonZeroRows);

//     size_t counter = 0;
//     for (size_t r_ind = 0; r_ind < R.rows(); r_ind++) {
//       if (nonZeroRows(r_ind) == 1.0) {
//         T_H.row(counter) = R.row(r_ind);
//         Q_1.col(counter) = Q.col(r_ind);
//         counter++;
//         if (counter > numNonZeroRows) {
//           //ROS_ERROR("More non zero rows than expected in QR decomp");
//         }
//       }
//     }

//     VectorXd r_n = Q_1.transpose() * r_o;
//     MatrixXd R_n = Q_1.transpose() * R_o * Q_1 ;*/
//   }
// }

bool LviEstimator::BuildProblemAndSolveLI(int iteration) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  
  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  AddLiDARErrorTerms(estimator);
  AddImuErrorTerms(estimator, imu_state_estimator_, true, true,
    ps_->imu_pos_weight_li, ps_->imu_vel_weight_li, ps_->imu_rot_weight_li);

  AddStartTimePose(estimator);
  
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  calib_param_->CheckIMUBias();
  UpdateTrajectoryProperty();

  return true;
}


void LviEstimator::triangulate(){
  //cartographer::common::MutexLocker locker(&mutex_);
  ////std::scoped_lock lock(mtx_);
  int num_succ = 0;
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 
        && it_per_id.start_frame < WINDOW_SIZE - 2)){
      continue;
    }  

    if (it_per_id.estimated_depth > 0)
      continue;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
  
    Eigen::Matrix<double, 3, 4> pose_i, pose_j;
    Eigen::Vector2d pt_i, pt_j;
    std::vector<Eigen::Matrix<double, 3, 4>> poses_obs;
    std::vector<Eigen::Vector2d> pts_obs;
    SE3d  se_i, se_j;
    double t_i = it_per_id.feature_per_frame.front().timestamp_ 
                 - trajectory_->GetDataStartTime();
    if(!trajectory_->GetCameraPose(t_i, se_i)) continue;
    pose_i.block(0,0,3,3) = se_i.unit_quaternion().toRotationMatrix();
    pose_i.block(0,3,3,1) = se_i.translation();
    pt_i  = it_per_id.feature_per_frame.front().point.head<2>();
    
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = se_i.translation();
    Eigen::Matrix3d R0 = se_i.unit_quaternion().toRotationMatrix();
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      double t_j = it_per_frame.timestamp_ - trajectory_->GetDataStartTime();
      trajectory_->GetCameraPose(t_j, se_j);//todo: may not valid?

      Eigen::Vector3d t1 = se_j.translation();
      Eigen::Matrix3d R1 = se_j.unit_quaternion().toRotationMatrix();;
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
    }
    CHECK(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(
      svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    it_per_id.estimated_depth = svd_V[2] / svd_V[3];

    if (it_per_id.estimated_depth < 0.1){
      it_per_id.estimated_depth = ps_->INIT_DEPTH;
      // it_per_id.estimated_depth = -1.;
    }else{
      num_succ++;
    }
  }
  // LOG(INFO)<<"Triangulated "<< num_succ << " points.";
}

void LviEstimator::AddCameraErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator){
  // when call this function, the LIO trajectory has been initialized,
  // we can triangulate features which has no depth associated 
  // or that failed to estimate in VIO.
  // triangulate();

  double weight = calib_param_->global_opt_cam_uv_weight;
  
  // vector2double();
  VectorXd dep_bef = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    para_Feature[i][0] = dep_bef(i);
  
  // add reprojection errors
  int f_m_cnt_new = 0;
  int f_m_cnt_true = 0;
  int feature_index = -1;
  int no_depth_num = 0;
  double t0 = trajectory_->GetDataStartTime();
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    ++feature_index;

    // TODO: check this
    if(it_per_id.estimated_depth < 0) {
      no_depth_num++;
      continue;
    }

    // 共视点在i时刻坐标系下的归一化坐标
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double t_i = it_per_id.feature_per_frame[0].timestamp_;

    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      if (imu_i == imu_j){
        continue;
      }
      // 共视点在j时刻坐标系下的归一化坐标
      Vector3d pts_j = it_per_frame.point;
      double t_j = it_per_frame.timestamp_;
      
      if(true){//(f_m_cnt_new++) % step == 0
        // if this feature is optimized in VIO, we lock its depth 
        // and just optimize the associated poses here for higher efficiency.
        bool lock_depth = it_per_id.solve_flag == 1;
        double weight_uv = weight;
        if(it_per_frame.to_reweight && ps_->reweight_outlier){
          weight_uv *= ps_->reweight_outlier_scale;
        }

        // LOG(INFO)<<t_i - t0<<","<<t_j - t0;
        estimator->AddCameraMeasurement(t_i - t0, t_j - t0, pts_i, pts_j, 
          para_Feature[feature_index], weight_uv, false, 
          ps_->visual_depth_upper_bound, ps_->visual_depth_lower_bound);
        f_m_cnt_true++;
      }
    }
  }
  // LOG(INFO)<<"Added "<<f_m_cnt_true<<" visual terms.";
  // LOG(INFO)<<"feature_num "<<feature_index;
  // LOG(INFO)<<"no_depth_num "<<no_depth_num;
}

void LviEstimator::AddImuErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator,
      std::shared_ptr<ImuStateEstimator> imu_integrator,
      bool enable_integrated_pose,
      bool enable_predicted_pose,
      double wp,
      double wv,
      double wq){
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double ba_weight = calib_param_->global_opt_imu_ba_weight;
  double bg_weight = calib_param_->global_opt_imu_bg_weight;
  // double rot_weight = calib_param_->global_opt_imu_rot_weight;
  // double pos_weight = calib_param_->global_opt_imu_pos_weight;
  // double vel_weight = calib_param_->global_opt_imu_vel_weight;

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= active_time_upper_) break;
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     ba_weight, bg_weight);
  }
  
  for (const auto& state : imu_integrator->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, wv);
    if(enable_integrated_pose){
      clins::PoseData pose;
      pose.timestamp = state.timestamp;
      pose.position = state.p;
      pose.orientation = SO3d(state.q);
      estimator->AddPoseMeasurement(pose, wq, wp);
    }
  }

  for (const auto& state : imu_integrator->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, wv);
    if(enable_predicted_pose){
      clins::PoseData pose;
      pose.timestamp = state.timestamp;
      pose.position = state.p;
      pose.orientation = SO3d(state.q);
      estimator->AddPoseMeasurement(pose, wq, wp);
    }
  }
}

bool LviEstimator::BuildProblemAndSolveVI(int iteration, bool lio_failed) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  if(frame_count_ < WINDOW_SIZE){
    if(!ps_->optimize_lvi_together){
      frame_count_++;
    }
    
    UpdateTrajectoryProperty();
    return false;
  }
  
  if(imu_integrator_for_camera_->GetCurrentMotionState() 
      == MotionState::start_motionless){
    clearState();
    UpdateTrajectoryProperty(); 
    return false;
  }
  
  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    if(lio_failed){
      estimator->SetKeyScanConstant(active_time_upper_ - ps_->vio_time_window);
    }else{
      estimator->SetKeyScanConstant(active_time_lower_);
    }
  }
  
  AddImuErrorTerms(estimator, imu_integrator_for_camera_, true, true,
    ps_->imu_pos_weight_vi, ps_->imu_vel_weight_vi, ps_->imu_rot_weight_vi);
  AddStartTimePose(estimator);
  AddCameraErrorTerms(estimator);
  
  estimator->LockIMUState(false, false, true); 
  estimator->LockExtrinsicParam(true, true);
  
  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  // whether to update depth
  UpdateFeatureDepth();
  MarkVisualOutliers();
  feature_manager_->removeFailures();

  calib_param_->CheckIMUBias();
  UpdateTrajectoryProperty();
  
  if(!ps_->optimize_lvi_together){
    slideWindow();
  }
  
  return true;
}

bool LviEstimator::BuildProblemAndSolveLVI(int iteration) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  if(imu_integrator_for_camera_->GetCurrentMotionState() 
      == MotionState::start_motionless){
    clearState();
  }

  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  AddLiDARErrorTerms(estimator);
  AddImuErrorTerms(estimator, imu_state_estimator_, true, true,
    ps_->imu_pos_weight_li, ps_->imu_vel_weight_li, ps_->imu_rot_weight_li);
  AddStartTimePose(estimator);
  AddCameraErrorTerms(estimator);
  
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  calib_param_->CheckIMUBias();
  
  UpdateFeatureDepth();
  MarkVisualOutliers();
  feature_manager_->removeFailures();

  if(ps_->remove_outlier){
    feature_manager_->removeOutlier();
  }

  UpdateTrajectoryProperty();
  
  // The features in feature manager may have been changed when optimizating.
  // TODO: these codes are not thread-safe, the feature manager may have been
  // updated by VINS when performing LVI optimization above.
  if(ps_->enable_debug){
    LogVisualError();
  }
  
  return true;
}



void LviEstimator::MarkVisualOutliers(){
  // std::scoped_lock lock(feature_manager_->mtx_);
  int f_m_cnt = 0;
  int outlier_cnt = 0;
  int feature_index = -1;
  double t0 = trajectory_->GetDataStartTime();
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    
    // if(it_per_id.estimated_depth < 0) continue;//not inited yet
    
    ++feature_index;
    // 共视点在i时刻坐标系下的归一化坐标
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double t_i = it_per_id.feature_per_frame[0].timestamp_;
    SE3d pose_i, pose_j;
    if(!trajectory_->GetCameraPose(t_i - t0, pose_i)) continue;
    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      if (imu_i == imu_j){
        continue;
      }
      // 共视点在j时刻坐标系下的归一化坐标
      Vector3d pts_j = it_per_frame.point;
      double t_j = it_per_frame.timestamp_;
      if(!trajectory_->GetCameraPose(t_j - t0, pose_j)) continue;
      Eigen::Vector3d pt_i_3d = pts_i * it_per_id.estimated_depth;
      Eigen::Vector3d pt_j_3d = pose_j.inverse() * pose_i * pt_i_3d;
      Eigen::Vector2d rep_err;
      rep_err << pt_j_3d[0] / pt_j_3d[2] - pts_j[0],
                 pt_j_3d[1] / pt_j_3d[2] - pts_j[1];
      f_m_cnt++;
      if(std::abs(rep_err[0]) > ps_->reproject_outlier_threshold
          || std::abs(rep_err[1]) > ps_->reproject_outlier_threshold){
        it_per_frame.to_reweight = true;
        outlier_cnt++;
      }else{
        it_per_frame.to_reweight = false;
      }
    }
  }
  last_outlier_ratio = float(outlier_cnt) / float(f_m_cnt);
  if(last_outlier_ratio > ps_->outlier_ratio_threshold_to_reset){
    clearState();
    LOG(INFO)<<"outlier and observation size: "<<outlier_cnt<<", "<<f_m_cnt;
  }
}

void LviEstimator::UpdateFeatureDepth(){
  // The features in feature manager may have been changed by other thread.
  // so we just update those that still exist in the feature bucket.
  VectorXd dep_aft = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    dep_aft(i) = para_Feature[i][0];
  feature_manager_->setDepth(dep_aft);
}


void LviEstimator::UpdateTrajectoryProperty() {
  trajectory_->UpdateActiveTime(active_time_upper_);
  trajectory_->SetForcedFixedTime(active_time_lower_ - 0.1);
}

void LviEstimator::IntegrateIMUForCamera(double img_timestamp) {
  if (imu_data_.empty()) {
    LOG(WARNING) << "[IntegrateIMUForCamera] IMU data empty! " << std::endl;
    return;
  }
  // LOG(INFO)<<img_timestamp;
  cur_image_time_ = img_timestamp;
  active_time_upper_ = cur_image_time_;

  //推算上一帧图像进入时所推算到的时刻IMU在世界系下的状态
  // LOG(INFO) << "Min-max and lateset time: " << trajectory_->minTime()<<"-"<<trajectory_->maxTime()<<" "<<imu_state_estimator_->GetLatestTimestamp();
  imu_integrator_for_camera_->GetLatestIMUState<4>(trajectory_);

  //推算到当前时刻
  imu_integrator_for_camera_->Propagate(active_time_upper_);

  //删除之前的IMU数据, this->imu_data_
  // ErasePastImuData(trajectory_->GetActiveTime());

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(active_time_upper_, last_kont);
    
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  imu_integrator_for_camera_->Predict(trajectory_->maxTime() - 1e-9, 0.01);

  AddImuErrorTerms(estimator, imu_integrator_for_camera_, true, true,
    ps_->imu_pos_weight_vi, ps_->imu_vel_weight_vi, ps_->imu_rot_weight_vi);
  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);
}

void LviEstimator::IntegrateIMUMeasurement(double scan_min, double scan_max) {
  if (imu_data_.empty()) {
    LOG(WARNING) << "[IntegrateIMUForCamera] IMU data empty! " << std::endl;
    return;
  }
  // LOG(INFO)<<img_timestamp;
  active_time_lower_ = scan_min;
  active_time_upper_ = scan_max;

  //推算上一帧图像进入时所推算到的时刻IMU在世界系下的状态
  // LOG(INFO) << "Min-max and lateset time: " << trajectory_->minTime()<<"-"<<trajectory_->maxTime()<<" "<<imu_state_estimator_->GetLatestTimestamp();
  imu_state_estimator_->GetLatestIMUState<4>(trajectory_);

  //推算到当前时刻
  imu_state_estimator_->Propagate(active_time_upper_);

  //删除之前的IMU数据, this->imu_data_
  ErasePastImuData(trajectory_->GetActiveTime());

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(active_time_upper_, last_kont);
    
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  imu_state_estimator_->Predict(trajectory_->maxTime() - 1e-9, 0.01);

  AddImuErrorTerms(estimator, imu_state_estimator_, true, true,
    ps_->imu_pos_weight_li, ps_->imu_vel_weight_li, ps_->imu_rot_weight_li);
  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);
}

// void LviEstimator::UpdateInitStatusForTrajectory(){
//   // update data in calibration parameter manager
//   // 以窗口内第一帧初始化时，需要再优化一遍窗口内的轨迹；
//   // 以最后一帧初始化时，会不会使得前面的点找不到对应的knot，对于后续进入的帧约束不够
//   Eigen::Vector3d const_gravity(0, 0, 9.805);//todo
//   calib_param_->SetGravity(const_gravity);
//   calib_param_->SetAccelBias(Bas[0]);
//   calib_param_->SetGyroBias(Bgs[0]);
//   this->InitIMUData(Headers[0]);
//   this->SetInitialPoseRotation(Eigen::Quaterniond(Rs[0]));
//   LOG(INFO)<<Headers[WINDOW_SIZE];
//   for(int i = 0; i < WINDOW_SIZE + 1; ++i){
//     Headers[i] -= trajectory_->GetDataStartTime();
//   }
//   LOG(INFO)<<Headers[WINDOW_SIZE];

//   // LOG(INFO)<<"Initialize ct-trajectory succeed!"; 
//   // LOG(INFO)<<"Ba: "<<Bas[0].transpose();
//   // LOG(INFO)<<"Bg: "<<Bgs[0].transpose();
//   // LOG(INFO)<<"Pos: "<<Ps[0].transpose();
//   // LOG(INFO)<<"Vel: "<<Vs[0].transpose();
//   // LOG(INFO)<<"Rot: "<<Rs[0];

//   SE3d last_kont = trajectory_->getLastKnot();
//   trajectory_->extendKnotsTo(Headers[WINDOW_SIZE], last_kont);

//   std::shared_ptr<TrajectoryEstimator<4>> estimator =
//       std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);

//   double gyro_weight = calib_param_->global_opt_gyro_weight;
//   double accel_weight = calib_param_->global_opt_acce_weight;
//   double ba_weight = calib_param_->global_opt_imu_ba_weight;
//   double bg_weight = calib_param_->global_opt_imu_bg_weight;

//   double rot_weight = calib_param_->global_opt_cam_rot_weight;
//   double pos_weight = calib_param_->global_opt_cam_pos_weight;

//   // set active segments
//   estimator->LockTrajectory(false);
  
//   // add reprojection factors
//   /* VectorXd dep_bef = feature_manager_->getDepthVector();
//   for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
//     para_Feature[i][0] = dep_bef(i);

//   int f_m_cnt = 0;
//   int feature_index = -1;
//   for (auto &it_per_id : feature_manager_->feature){
//     it_per_id.used_num = it_per_id.feature_per_frame.size();
//     if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
//       continue;

//     ++feature_index;
    
//     // 共视点在i时刻坐标系下的归一化坐标
//     int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
//     Vector3d pts_i = it_per_id.feature_per_frame[0].point;
//     double t_i = it_per_id.feature_per_frame[0].timestamp_;

//     for (auto &it_per_frame : it_per_id.feature_per_frame){
//       imu_j++;
//       if (imu_i == imu_j){
//         continue;
//       }
//       // 共视点在j时刻坐标系下的归一化坐标
//       Vector3d pts_j = it_per_frame.point;
//       double t_j = it_per_frame.timestamp_;
//       double weight = calib_param_->global_opt_cam_uv_weight;
//       estimator->AddCameraMeasurement(t_i, t_j, pts_i, pts_j, 
//           para_Feature[feature_index], 
//           weight, v_depth_upper_bound_, v_depth_lower_bound_);
          
//       f_m_cnt++;
//     }
//   } */
  
//   IMUBias imu_bias = calib_param_->GetIMUBias();
//   for (const auto& v : imu_data_) {
//     if (v.timestamp >= Headers[WINDOW_SIZE]) break;
//     estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
//                                      ba_weight, bg_weight);
//   }
  
//   for(int i = 0; i < WINDOW_SIZE + 1; ++i){
//     PoseData pose;
//     pose.timestamp = Headers[i];
//     pose.position = Ps[i];
//     pose.orientation = SO3d(Eigen::Quaterniond(Rs[i]));
//     estimator->AddPoseMeasurement(pose, rot_weight, pos_weight);
//   }

//   AddStartTimePose(estimator);

//   estimator->LockExtrinsicParam(true, true);
//   estimator->LockIMUState(true, true, true); 

//   ceres::Solver::Summary summary = estimator->Solve(50, false);
  
//   // update times here and propagate the imu state extropolator
//   cur_image_time_ = Headers[WINDOW_SIZE];
//   active_time_upper_ = Headers[WINDOW_SIZE];
//   active_time_lower_ = Headers[WINDOW_SIZE - 2];
  
//   // 推算到当前帧,下一帧进入时会通过优化后的轨迹更新状态，本质上只使用了相对位姿
//   imu_state_estimator_->Propagate(Headers[WINDOW_SIZE]);
  
//   /* LOG(INFO)<<imu_state_estimator_->GetLatestTimestamp();
//   double s = 180./M_PI;
//   for(int i = 0; i < WINDOW_SIZE + 1; ++i){
//     SE3d pose = trajectory_->GetIMUPose(Headers[i]);
//     LOG(INFO) << (pose.translation() - Ps[i]).transpose();
//     LOG(INFO) << (pose.unit_quaternion().conjugate() * Eigen::Quaterniond(Rs[i])).  
//         toRotationMatrix().eulerAngles(0,1,2).transpose() * s;
//   } */

//   // double2vector();
//   /* VectorXd dep_aft = feature_manager_->getDepthVector();
//   for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
//     dep_aft(i) = para_Feature[i][0];
//   feature_manager_->setDepth(dep_aft); */
// }

void LviEstimator::InitializeDynamic(
      const std::deque<double>& timestamps,
      const std::deque<Eigen::Vector3d>& ps, 
      const std::deque<Eigen::Vector3d>& vs,
      const std::deque<Eigen::Matrix3d>& rs,
      const std::deque<Eigen::Vector3d>& bas,
      const std::deque<Eigen::Vector3d>& bgs){
  // update data in calibration parameter manager
  // 以窗口内第一帧初始化时，需要再优化一遍窗口内的轨迹；
  // 以最后一帧初始化时，会不会使得前面的点找不到对应的knot，对于后续进入的帧约束不够
  Eigen::Vector3d const_gravity(0, 0, ps_->GRAVITY_NORM);//todo
  calib_param_->SetGravity(const_gravity);
  calib_param_->SetAccelBias(bas.back());
  calib_param_->SetGyroBias(bgs.back());
  this->InitIMUData(timestamps.back());
  this->SetInitialPoseRotation(Eigen::Quaterniond(rs.back()));

  // std::vector<double> ts;
  // for(int i = 0; i < timestamps.size(); ++i){
  //   ts.push_back(timestamps[i] - trajectory_->GetDataStartTime());
  // }

  // LOG(INFO)<<ts[0]<<","<<ts.back()<<","<<trajectory_->GetDataStartTime();

  /* SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(ts.back(), last_kont);

  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);

  // set active segments
  estimator->LockTrajectory(false);
  
  double acce_weight = calib_param_->global_opt_acce_weight;
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= ts.back()) break;
    estimator->AddIMUMeasurement(v, gyro_weight, acce_weight);
  }

  for(int i = 0; i < ps.size(); ++i){
    PoseData pose;
    pose.timestamp = ts[i];
    pose.position = ps[i];
    pose.orientation = SO3d(Eigen::Quaterniond(rs[i]));
    estimator->AddPoseMeasurement(pose, rot_weight, pos_weight);
  }

  AddStartTimePose(estimator);
  
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true); 

  ceres::Solver::Summary summary = estimator->Solve(50, false);
  
  // update times here and propagate the imu state extropolator
  active_time_upper_ = ts.back();
  // active_time_lower_ = active_time_upper_ - 0.1;
  active_time_lower_ = max(0., active_time_upper_ - 0.1);
  
  // 推算到当前帧,下一帧进入时会通过优化后的轨迹更新状态，本质上只使用了相对位姿
  imu_state_estimator_->Propagate(ts.back()); */
  
  // LOG(INFO)<<"Finish intial trajectory fitting.";
  /* LOG(INFO)<<imu_state_estimator_->GetLatestTimestamp();
  double s = 180./M_PI;
  for(int i = 0; i < WINDOW_SIZE + 1; ++i){
    SE3d pose = trajectory_->GetIMUPose(Headers[i]);
    LOG(INFO) << (pose.translation() - Ps[i]).transpose();
    LOG(INFO) << (pose.unit_quaternion().conjugate() * Eigen::Quaterniond(Rs[i])).  
        toRotationMatrix().eulerAngles(0,1,2).transpose() * s;
  } */
}



void LviEstimator::Initialize(
      double scan_time, const Eigen::Quaterniond& q,
      const Eigen::Vector3d& ba, const Eigen::Vector3d& bg){
  // update data in calibration parameter manager
  Eigen::Vector3d const_gravity(0, 0, ps_->GRAVITY_NORM);
  calib_param_->SetGravity(const_gravity);
  calib_param_->SetAccelBias(ba);
  calib_param_->SetGyroBias(bg);
  InitIMUData(scan_time);
  SetInitialPoseRotation(q);
  trajectory_init_ = true;
}

//////////////////////////////////////////////////////////////////////////
void LviEstimator::slideWindow(){
  TicToc t_margin;
  if (marg_flag_ == MARGIN_OLD){
    double t_0 = Headers[0];
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count_ == WINDOW_SIZE){
      for (int i = 0; i < WINDOW_SIZE; i++){
        Rs[i].swap(Rs[i + 1]);
        Headers[i] = Headers[i + 1];
        Ps[i].swap(Ps[i + 1]);
        Vs[i].swap(Vs[i + 1]);
        Bas[i].swap(Bas[i + 1]);
        Bgs[i].swap(Bgs[i + 1]);
      }
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
      Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
      Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

      /* delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{
        acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
        ps_->G, ps_->ACC_N, ps_->GYR_N, ps_->ACC_W, ps_->GYR_W};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      if (true || solver_flag == INITIAL){
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);
        delete it_0->second.pre_integration;
        it_0->second.pre_integration = nullptr;

        for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); 
            it != it_0; ++it){
          if (it->second.pre_integration)
            delete it->second.pre_integration;
          it->second.pre_integration = NULL;
        }

        all_image_frame.erase(all_image_frame.begin(), it_0);
        all_image_frame.erase(t_0);
      } */
      slideWindowOld();
    }else{
      slideWindowOld();
    }
  }else{
    if (frame_count_ == WINDOW_SIZE){
      /* for (unsigned int i = 0; i < dt_buf[frame_count_].size(); i++){
        double tmp_dt = dt_buf[frame_count_][i];
        Vector3d tmp_linear_acceleration = 
            linear_acceleration_buf[frame_count_][i];
        Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count_][i];

        pre_integrations[frame_count_ - 1]->push_back(
            tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

        dt_buf[frame_count_ - 1].push_back(tmp_dt);
        linear_acceleration_buf[frame_count_ - 1].push_back(
            tmp_linear_acceleration);
        angular_velocity_buf[frame_count_ - 1].push_back(tmp_angular_velocity);
      } */

      Headers[WINDOW_SIZE- 1] = Headers[WINDOW_SIZE];
      Ps[WINDOW_SIZE - 1] = Ps[WINDOW_SIZE];
      Vs[WINDOW_SIZE - 1] = Vs[WINDOW_SIZE];
      Rs[WINDOW_SIZE - 1] = Rs[WINDOW_SIZE];
      Bas[WINDOW_SIZE - 1] = Bas[WINDOW_SIZE];
      Bgs[WINDOW_SIZE - 1] = Bgs[WINDOW_SIZE];
      
      /* delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{
          acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
          ps_->G, ps_->ACC_N, ps_->GYR_N, ps_->ACC_W, ps_->GYR_W};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear(); */

      slideWindowNew();
    }else{
      slideWindowNew();
    }
  }
}

// real marginalization is removed in solve_ceres()
void LviEstimator::slideWindowNew(){
  feature_manager_->removeFront(frame_count_);
}

// real marginalization is removed in solve_ceres()
void LviEstimator::slideWindowOld(){
  bool shift_depth = solver_flag == NON_LINEAR ? true : false;
  // bool shift_depth = frame_count_ == WINDOW_SIZE ? true: false;
  if (shift_depth){
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ps_->RIC[0];
    R1 = Rs[0] * ps_->RIC[0];
    P0 = back_P0 + back_R0 * ps_->TIC[0];
    P1 = Ps[0] + Rs[0] * ps_->TIC[0];
    feature_manager_->removeBackShiftDepth(R0, P0, R1, P1);
  } else {
    feature_manager_->removeBack();
  }
}


bool LviEstimator::initialStructure(){
  TicToc t_sfm;
  //check imu observibility
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++; 
        frame_it != all_image_frame.end(); frame_it++){
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; 
        frame_it != all_image_frame.end(); frame_it++){
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    // LOG(WARNING)<<"IMU variation: "<<var;
    if(var < 0.25){
      LOG(INFO)<<"IMU excitation not enouth!";
      return false;
    }
  }
  
  // global sfm
  Quaterniond Q[frame_count_ + 1];
  Vector3d T[frame_count_ + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;
  for (auto &it_per_id : feature_manager_->feature){
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(
        make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  } 
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)){
    LOG(WARNING)<<"Not enough features or parallax; Move device around";
    return false;
  }
  
  GlobalSFM sfm;
  if(!sfm.construct(frame_count_ + 1, Q, T, l, relative_R, relative_T,
        sfm_f, sfm_tracked_points)){
    LOG(WARNING)<<"global SFM failed!";
    marg_flag_ = MARGIN_OLD;
    return false;
  }
  // 全局SfM里面没有解算PnP的过程吗，为什么这里要重新计算一次？ 因为此处为优化后的值？
  // solve pnp for all frame
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin( );
  for (int i = 0; frame_it != all_image_frame.end( ); frame_it++){
    // provide initial guess
    
    cv::Mat r, rvec, t, D, tmp_r;
    if((frame_it->first) == Headers[i]){
      frame_it->second.is_key_frame = true;
      frame_it->second.R = Q[i].toRotationMatrix() * ps_->RIC[0].transpose();
      frame_it->second.T = T[i];
      i++;
      
      continue;
    }
    
    if((frame_it->first) > Headers[i]){
      i++;
    }
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = - R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);
    
    
    
    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto &id_pts : frame_it->second.points){
      int feature_id = id_pts.first;
      for (auto &i_p : id_pts.second){
        it = sfm_tracked_points.find(feature_id);
        if(it != sfm_tracked_points.end()){
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
    if(pts_3_vector.size() < 6){
      LOG(WARNING)<<"Not enough points for solve pnp! points size: "
                  <<pts_3_vector.size();
      return false;
    }
    if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)){
      LOG(WARNING)<<"solve pnp fail!";
      return false;
    }
    
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp,tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * ps_->RIC[0].transpose();
    frame_it->second.T = T_pnp;
  }

  
  if (visualInitialAlign())
    return true;
  else{
    LOG(WARNING)<<"misalign visual structure with IMU";
    return false;
  }
}

bool LviEstimator::visualInitialAlign(){
  TicToc t_g;
  VectorXd x;
  
  //solve scale
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g_, x, *ps_);
  if(!result){
    LOG(WARNING)<<"solve g failed!";
    return false;
  }
  
  // change state
  for (int i = 0; i <= frame_count_; i++){
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    Vector3d Pi = all_image_frame[Headers[i]].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  // VectorXd dep = feature_manager_->getDepthVector();
  // for (int i = 0; i < dep.size(); i++)
  //   dep[i] = -1;
  feature_manager_->clearDepth();

  //triangulat on cam pose , no tic
  Vector3d TIC_TMP[NUM_OF_CAM];
  for(int i = 0; i < NUM_OF_CAM; i++)
    TIC_TMP[i].setZero();
  ric[0] = ps_->RIC[0];
  // feature_manager_->setRic(ric);
  feature_manager_->triangulate(Rs, Ps, &(TIC_TMP[0]), &(ps_->RIC[0]));

  //根据估计的bias重新积分，根据估计的尺度因子恢复尺度
  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++){
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  for (int i = frame_count_; i >= 0; i--)
    Ps[i] = s * Ps[i] - Rs[i] * ps_->TIC[0] - (s * Ps[0] - Rs[0] * ps_->TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); 
      frame_i != all_image_frame.end(); frame_i++){
    if(frame_i->second.is_key_frame){
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth *= s;
  }

  // 以第一帧的方向为起始方向
  Matrix3d R0 = Utility::g2R(g_);
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g_ = R0 * g_;
  //Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count_; i++){
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  LOG(INFO)<<"g0  " << g_.transpose();
  return true;
}

bool LviEstimator::relativePose(
    Matrix3d &relative_R, Vector3d &relative_T, int &l){
  // find previous frame which contians enough correspondance and parallex 
  // with newest frame
  for (int i = 0; i < WINDOW_SIZE; i++){
    vector<pair<Vector3d, Vector3d>> corres;
    corres = feature_manager_->getCorresponding(i, WINDOW_SIZE);
    if (corres.size() > 20){
      double sum_parallax = 0;
      double average_parallax;
      for (int j = 0; j < int(corres.size()); j++){
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());
      if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(
          corres, relative_R, relative_T)){
        l = i;
        return true;
      }
    }
  }
  return false;
}


void LviEstimator::clearState(){
  for (int i = 0; i < WINDOW_SIZE + 1; i++){
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr)
      delete pre_integrations[i];
    pre_integrations[i] = nullptr;
  }
  for (int i = 0; i < NUM_OF_CAM; i++){
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  for (auto &it : all_image_frame){
    if (it.second.pre_integration != nullptr){
      delete it.second.pre_integration;
      it.second.pre_integration = nullptr;
    }
  }

  first_imu = false,
  frame_count_ = 0;
  solver_flag = INITIAL;
  all_image_frame.clear();
  // td = ps_->TD;

  if (tmp_pre_integration != nullptr)
    delete tmp_pre_integration;
  // if (last_marginalization_info != nullptr)
  //   delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  // last_marginalization_info = nullptr;
  // last_marginalization_parameter_blocks.clear();
  
  feature_manager_->clearState();
  
  last_outlier_ratio = 0;
  // failure_occur = 0;
  // relocalization_info = 0;

  // drift_correct_r = Matrix3d::Identity();
  // drift_correct_t = Vector3d::Zero();
}

}
