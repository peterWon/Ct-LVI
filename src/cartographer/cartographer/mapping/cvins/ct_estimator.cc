#include "ct_estimator.h"
#include "glog/logging.h"

namespace cvins{

CtEstimator::CtEstimator(const ParameterServer* ps,
   std::shared_ptr<TrajectoryN> trajectory,
   std::shared_ptr<CalibParamManager> calib_param,
   std::shared_ptr<FeatureManager> feature_manager): 
      ps_(ps),
      trajectory_(trajectory),
      calib_param_(calib_param),
      feature_manager_(feature_manager){
  imu_state_estimator_.reset(new clins::ImuStateEstimator(ps));
  imu_integrator_for_camera_.reset(new clins::ImuStateEstimator(ps));
}


void CtEstimator::InitIMUData(double feature_time) {
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

void CtEstimator::ErasePastImuData(double t) {
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < t) {
      iter = imu_data_.erase(iter);
    } else {
      ++iter;
    }
  }
}

void CtEstimator::AddStartTimePose(
    std::shared_ptr<TrajectoryEstimator<4>> estimator) {
  size_t kont_idx = trajectory_->computeTIndex(active_time_lower_).second;
  if (kont_idx < 4) {
    init_pose.timestamp = trajectory_->minTime();

    double rot_weight = 1000;
    double pos_weight = 1000;
    estimator->AddPoseMeasurement(init_pose, rot_weight, pos_weight);
  }
}


void CtEstimator::AddLiDARErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator){
  double rot_weight = calib_param_->global_opt_lidar_rot_weight;
  double pos_weight = calib_param_->global_opt_lidar_pos_weight;
  for(const auto& lp: lidar_anchors_){
    estimator->AddPoseMeasurement(lp, rot_weight, pos_weight);
  }
}

void CtEstimator::LogVisualError(){
  // std::scoped_lock lock(feature_manager_->mtx_);
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

void CtEstimator::MarkVisualOutliers(){
  // std::scoped_lock lock(feature_manager_->mtx_);
  int f_m_cnt = 0;
  int outlier_cnt = 0;
  int feature_index = -1;
  double t0 = trajectory_->GetDataStartTime();
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    

    ++feature_index;
    // if(it_per_id.estimated_depth < 0) continue;//not inited yet

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
  LOG(INFO)<<"outlier and observation size: "<<outlier_cnt<<", "<<f_m_cnt;
}

// void CtEstimator::AddCameraErrorTermsNoFeature(
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

void CtEstimator::triangulate(const Eigen::Matrix<double, 3, 4>& pose_ref, 
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

void CtEstimator::triangulate(){
  //cartographer::common::MutexLocker locker(&mutex_);
  ////std::scoped_lock lock(mtx_);
  int num_succ = 0;
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 4 
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
    /* for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;
      if(imu_i == imu_j) continue;
      
      double t_j = it_per_frame.timestamp_ - trajectory_->GetDataStartTime();
      if(!trajectory_->GetCameraPose(t_j, se_j)) continue;
      pose_j.block(0,0,3,3) = se_j.unit_quaternion().toRotationMatrix();
      pose_j.block(0,3,3,1) = se_j.translation();
      pt_j  = it_per_frame.point.head<2>();
      poses_obs.emplace_back(pose_j);
      pts_obs.emplace_back(pt_j);
    }

    Eigen::Vector3d pt_3d_w, pt_3d_i;
    triangulate(pose_i, pt_i, poses_obs, pts_obs, pt_3d_w);
    pt_3d_i = se_i.inverse() * pt_3d_w;
    it_per_id.estimated_depth = pt_3d_i[2]; */

    if (it_per_id.estimated_depth < 0.1){
      it_per_id.estimated_depth = ps_->INIT_DEPTH;
    }else{
      num_succ++;
    }
  }
  // LOG(INFO)<<"Triangulated "<< num_succ << " points.";
}

void CtEstimator::AddCameraErrorTerms(
      std::shared_ptr<TrajectoryEstimator<4>> estimator){
  // std::scoped_lock lock(feature_manager_->mtx_);
  
  // when call this function, the LIO trajectory has been initialized,
  // we can triangulate features which has no depth associated 
  // or that failed to estimate in VIO.
  triangulate();

  double weight = calib_param_->global_opt_cam_uv_weight;
  
  // vector2double();
  VectorXd dep_bef = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    para_Feature[i][0] = dep_bef(i);
  
  // add reprojection errors
  int f_m_cnt_new = 0;
  int f_m_cnt_true = 0;
  int feature_index = -1;
  double t0 = trajectory_->GetDataStartTime();
  for (auto &it_per_id : feature_manager_->feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    
    ++feature_index;
    
    // TODO: check this
    // if(it_per_id.estimated_depth < 0) continue;

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
}

void CtEstimator::AddImuErrorTerms(
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


bool CtEstimator::BuildProblemAndSolveVI(int iteration) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);

  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(active_time_lower_);
  }
  
  AddImuErrorTerms(estimator, imu_integrator_for_camera_, true, true,
    ps_->imu_pos_weight_vi, ps_->imu_vel_weight_vi, ps_->imu_rot_weight_vi);
  AddStartTimePose(estimator);

  std::scoped_lock lk(feature_manager_->mtx_);
  AddCameraErrorTerms(estimator);
  
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  // whether to update depth
  UpdateFeatureDepth();
  MarkVisualOutliers();

  calib_param_->CheckIMUBias();
  UpdateTrajectoryProperty();
  
  return true;
}

bool CtEstimator::BuildProblemAndSolveLVI(int iteration) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  
  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  AddLiDARErrorTerms(estimator);
  // TODO(wz): check weights
  AddImuErrorTerms(estimator, imu_state_estimator_, true, true,
    ps_->imu_pos_weight_li, ps_->imu_vel_weight_li, ps_->imu_rot_weight_li);
  AddStartTimePose(estimator);
  
  // Todo(WZ): codes below are time-consuming.
  std::scoped_lock lock(feature_manager_->mtx_);
  AddCameraErrorTerms(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  calib_param_->CheckIMUBias();
  
  UpdateFeatureDepth();
  MarkVisualOutliers();

  if(ps_->remove_outlier){
    feature_manager_->removeOutlier();
  }

  // The features in feature manager may have been changed when optimizating.
  // TODO: these codes are not thread-safe, the feature manager may have been
  // updated by VINS when performing LVI optimization above.
  if(ps_->enable_debug){
    LogVisualError();
  }
  
  UpdateTrajectoryProperty();
  return true;
}

bool CtEstimator::BuildProblemAndSolveVI(
    const std::vector<State> cam_poses, int iteration) {
  std::shared_ptr<TrajectoryEstimator<4>> estimator =
      std::make_shared<TrajectoryEstimator<4>>(trajectory_, calib_param_);
  
  // set active segments
  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }
  
  double rot_weight = calib_param_->global_opt_cam_rot_weight;
  double pos_weight = calib_param_->global_opt_cam_pos_weight;
  for(const State& cp: cam_poses){
    PoseData pose;
    pose.timestamp = cp.timestamp;
    pose.position = cp.P;
    pose.orientation = SO3d(cp.Q);
    estimator->AddPoseMeasurement(pose, rot_weight, pos_weight);
  }
  AddImuErrorTerms(estimator, imu_state_estimator_, true, true,
    ps_->imu_pos_weight_vi, ps_->imu_vel_weight_vi, ps_->imu_rot_weight_vi);
  AddStartTimePose(estimator);
  // std::scoped_lock lock(feature_manager_->mtx_);
  // AddCameraErrorTerms(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  ceres::Solver::Summary summary = estimator->Solve(
      iteration, false, -1, ps_->max_time_cost);
  
  calib_param_->CheckIMUBias();
  UpdateTrajectoryProperty();

  return true;
}

bool CtEstimator::BuildProblemAndSolveLI(int iteration) {
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


void CtEstimator::UpdateFeatureDepth(){
  // The features in feature manager may have been changed by other thread.
  // so we just update those that still exist in the feature bucket.
  VectorXd dep_aft = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    dep_aft(i) = para_Feature[i][0];
  feature_manager_->setDepth(dep_aft);
}


void CtEstimator::UpdateTrajectoryProperty() {
  trajectory_->UpdateActiveTime(active_time_upper_);
  double fix_time = std::max(active_time_lower_ - 0.1, 0.);
  trajectory_->SetForcedFixedTime(fix_time);
}

void CtEstimator::IntegrateIMUForCamera(double img_timestamp) {
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
  imu_integrator_for_camera_->Propagate(cur_image_time_);

  //删除之前的IMU数据, this->imu_data_
  // ErasePastImuData(trajectory_->GetActiveTime());

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(cur_image_time_, last_kont);
    
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

void CtEstimator::IntegrateIMUMeasurement(double scan_min, double scan_max) {
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

  AddImuErrorTerms(estimator, imu_state_estimator_, false, false,
    ps_->imu_pos_weight_li, ps_->imu_vel_weight_li, ps_->imu_rot_weight_li);
  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);
}

// void CtEstimator::UpdateInitStatusForTrajectory(){
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

void CtEstimator::InitializeDynamic(
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
  calib_param_->SetAccelBias(bas.front());
  calib_param_->SetGyroBias(bgs.front());
  this->InitIMUData(timestamps.front());
  this->SetInitialPoseRotation(Eigen::Quaterniond(rs.front()));

  std::vector<double> ts;
  for(int i = 0; i < timestamps.size(); ++i){
    ts.push_back(timestamps[i] - trajectory_->GetDataStartTime());
  }

  // LOG(INFO)<<ts[0]<<","<<ts.back()<<","<<trajectory_->GetDataStartTime();

  SE3d last_kont = trajectory_->getLastKnot();
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
  imu_state_estimator_->Propagate(ts.back());
  imu_integrator_for_camera_->Propagate(ts.back());
  
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


void CtEstimator::Initialize(
      double scan_time, const Eigen::Quaterniond& q,
      const Eigen::Vector3d& ba, const Eigen::Vector3d& bg){
  // update data in calibration parameter manager
  Eigen::Vector3d const_gravity(0, 0, ps_->g_norm_for_debug);
  calib_param_->SetGravity(const_gravity);
  calib_param_->SetAccelBias(ba);
  calib_param_->SetGyroBias(bg);
  InitIMUData(scan_time);
  SetInitialPoseRotation(q);
  trajectory_init_ = true;
}

}
