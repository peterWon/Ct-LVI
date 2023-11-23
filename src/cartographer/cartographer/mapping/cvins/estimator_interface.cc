#include "estimator_interface.h"
#include <glog/logging.h>


namespace cvins{

std::vector<double> VinsInterface::GetImageStampsInWindow(){
  mtx_estimator.lock();
  std::vector<double> stamps;
  stamps.resize(WINDOW_SIZE+1);
  for(int i = 0; i < stamps.size(); ++i){
    stamps[i] = estimator_->Headers[i]; 
  }
  mtx_estimator.unlock();
  return stamps;
}

void VinsInterface::SetEvaluatedStates(
    const std::map<double, State>& states_in_local){
  mtx_estimator.lock();
  estimator_->states_in_local_.clear();
  estimator_->states_in_local_ = states_in_local;
  mtx_estimator.unlock();
}

bool VinsInterface::Initialized(){
  return estimator_->solver_flag == Estimator::SolverFlag::NON_LINEAR;
}

bool VinsInterface::GetCurrentStates(double& timestamp, 
    Eigen::Vector3d& p, Eigen::Vector3d& v, Eigen::Quaterniond& q,
    Eigen::Vector3d& ba, Eigen::Vector3d& bg){
  std::scoped_lock lk(mtx_state);
  if(estimator_->solver_flag == Estimator::SolverFlag::NON_LINEAR){
    timestamp = latest_time_;
    p = tmp_P;
    v = tmp_V;
    q = tmp_Q;
    ba = tmp_Ba;
    bg = tmp_Bg;

    /* nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time::now();
    odometry.header.frame_id = "map";
    odometry.child_frame_id = "map";

    tmp_Q.normalize();
    odometry.pose.pose.position.x = tmp_P.x();
    odometry.pose.pose.position.y = tmp_P.y();
    odometry.pose.pose.position.z = tmp_P.z();
    
    odometry.pose.pose.orientation.x = tmp_Q.x();
    odometry.pose.pose.orientation.y = tmp_Q.y();
    odometry.pose.pose.orientation.z = tmp_Q.z();
    odometry.pose.pose.orientation.w = tmp_Q.w();
    odometry.twist.twist.linear.x = tmp_V.x();
    odometry.twist.twist.linear.y = tmp_V.y();
    odometry.twist.twist.linear.z = tmp_V.z();
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id = "map";
    pose_stamped.pose = odometry.pose.pose;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "map";
    path.poses.push_back(pose_stamped);
    pub_path.publish(path); */

    return true;
  }else{
    return false;
  }
}

bool VinsInterface::GetCurrentStates(std::vector<State>& states){
  std::scoped_lock lk(mtx_state);
  states.clear();
  if(estimator_->solver_flag == Estimator::SolverFlag::NON_LINEAR){
    State pvq;
    for(int i = 0; i < WINDOW_SIZE+1; ++i){
      pvq.timestamp = estimator_->Headers[i];
      pvq.P = estimator_->Ps[i];
      pvq.V = estimator_->Vs[i];
      pvq.Q = Eigen::Quaterniond(estimator_->Rs[i]);
      pvq.Ba = estimator_->Bas[i];
      pvq.Bg = estimator_->Bgs[i];
      states.push_back(pvq);
    }
    pvq.timestamp = latest_time_;
    pvq.P = tmp_P;
    pvq.V = tmp_V;
    pvq.Q = tmp_Q;
    pvq.Ba = tmp_Ba;
    pvq.Bg = tmp_Bg;
    states.push_back(pvq);
    return true;
  }else{
    return false;
  }
}

void VinsInterface::GetFeaturesInWindow(
    const std::shared_ptr<FeatureManager>& fm){
  // mtx_feature.lock();
  if(estimator_->solver_flag != Estimator::SolverFlag::NON_LINEAR) return;
  fm->feature.clear();
  for(const auto& feat: estimator_->f_manager->feature){
    fm->feature.emplace_back(FeaturePerId(feat));
  }
  fm->last_track_num = estimator_->f_manager->last_track_num;
  // mtx_feature.unlock();
}

void VinsInterface::AddImuData(const cartographer::sensor::ImuData& imu_data){
  if (cartographer::common::ToSecondsStamp(imu_data.time) <= last_imu_t_){
    LOG(WARNING)<<"imu message in disorder!";
    return;
  }
  last_imu_t_ = cartographer::common::ToSecondsStamp(imu_data.time);

  mtx_m_buf.lock();
  imu_buf_.push(imu_data);
  mtx_m_buf.unlock();
  con.notify_one();

  {
    std::lock_guard<std::mutex> lg(mtx_state);
    Predict(imu_data);    
     
  }
}

void VinsInterface::AddImageData(const VinsFrameFeature& img_feature){
  if (!init_feature_){
    //skip the first detected feature, which doesn't contain optical flow speed
    init_feature_ = true;
    return;
  }
  mtx_m_buf.lock();
  feature_buf_.push(img_feature);
  mtx_m_buf.unlock();
  con.notify_one();
}

std::vector<std::pair<std::vector<cartographer::sensor::ImuData>,
    VinsFrameFeature>> VinsInterface::getMeasurements(){
  std::vector<std::pair<std::vector<cartographer::sensor::ImuData>,   
    VinsFrameFeature>> meas;

  while (true){
    if (imu_buf_.empty() || feature_buf_.empty())
      return meas; 
    
    if (cartographer::common::ToSecondsStamp(imu_buf_.back().time) 
          < feature_buf_.front().timestamp + estimator_->td){
      // LOG(WARNING)<<"Wait for imu, only should happen at the beginning";
      sum_of_wait++;
      return meas;
    }

    if (!(cartographer::common::ToSecondsStamp(imu_buf_.front().time) 
        < feature_buf_.front().timestamp + estimator_->td)){
      LOG(WARNING)<<"Throw img, only should happen at the beginning";
      feature_buf_.pop();
      continue;
    }
    VinsFrameFeature img = feature_buf_.front();
    feature_buf_.pop();

    std::vector<cartographer::sensor::ImuData> IMUs;
    while (cartographer::common::ToSecondsStamp(imu_buf_.front().time)
        < img.timestamp + estimator_->td){
      IMUs.emplace_back(imu_buf_.front());
      imu_buf_.pop();
    }
    IMUs.emplace_back(imu_buf_.front());
    if (IMUs.empty())
      LOG(WARNING)<<"no imu between two image";
    meas.emplace_back(IMUs, img);
  }
  return meas;
}

void VinsInterface::Run(){
  while (true){
    // <ImuReadings, Feature>...
    std::vector<std::pair<std::vector<cartographer::sensor::ImuData>,
        VinsFrameFeature>> measurements;
    std::unique_lock<std::mutex> lk(mtx_m_buf);
    con.wait(lk, [&]{
      return (measurements = getMeasurements()).size() != 0;
    });
    lk.unlock();
    mtx_estimator.lock();
    for (auto &measurement : measurements){
      const VinsFrameFeature& img = measurement.second;
      double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
      // LOG(INFO)<<measurement.first.size();
      for (const auto &imu : measurement.first){
        double t = cartographer::common::ToSecondsStamp(imu.time);
        double img_t = img.timestamp + estimator_->td;
        if (t <= img_t){ 
          if (current_time_ < 0)
            current_time_ = t;
          double dt = t - current_time_;
          CHECK(dt >= 0);
          current_time_ = t;
          dx = imu.linear_acceleration[0];
          dy = imu.linear_acceleration[1];
          dz = imu.linear_acceleration[2];
          rx = imu.angular_velocity[0];
          ry = imu.angular_velocity[1];
          rz = imu.angular_velocity[2];
          estimator_->processIMU(
            dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }else{
          double dt_1 = img_t - current_time_;
          double dt_2 = t - img_t;
          current_time_ = img_t;
          CHECK(dt_1 >= 0);
          CHECK(dt_2 >= 0);
          CHECK(dt_1 + dt_2 > 0);
          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          dx = w1 * dx + w2 * imu.linear_acceleration[0];
          dy = w1 * dy + w2 * imu.linear_acceleration[1];
          dz = w1 * dz + w2 * imu.linear_acceleration[2];
          rx = w1 * rx + w2 * imu.angular_velocity[0];
          ry = w1 * ry + w2 * imu.angular_velocity[1];
          rz = w1 * rz + w2 * imu.angular_velocity[2];
          estimator_->processIMU(
            dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }
      }
      // set relocalization frame
      /* sensor_msgs::PointCloudConstPtr relo_msg = NULL;
      while (!relo_buf.empty()){
        relo_msg = relo_buf.front();
        relo_buf.pop();
      }if (relo_msg != NULL){
        vector<Vector3d> match_points;
        double frame_stamp = relo_msg->header.stamp.toSec();
        for (unsigned int i = 0; i < relo_msg->points.size(); i++){
            Vector3d u_v_id;
            u_v_id.x() = relo_msg->points[i].x;
            u_v_id.y() = relo_msg->points[i].y;
            u_v_id.z() = relo_msg->points[i].z;
            match_points.push_back(u_v_id);
        }
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
        Matrix3d relo_r = relo_q.toRotationMatrix();
        int frame_index;
        frame_index = relo_msg->channels[0].values[7];
        estimator_->setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
      } */


      std::map<int, std::vector<
          std::pair<int, Eigen::Matrix<double, 8, 1>>>> image;
      for (unsigned int i = 0; i < img.features.size(); i++){
        const VinsFeature vf = img.features.at(i);
    
        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_d;
        xyz_uv_velocity_d << vf.x, vf.y, vf.z, vf.u, vf.v, 
                           vf.velocity_x, vf.velocity_y, vf.depth;
        image[vf.feature_id].emplace_back(camera_id_, xyz_uv_velocity_d);
      }
      estimator_->processImage(image, img.timestamp);

      // double whole_t = t_s.toc();
      // printStatistics(estimator_, whole_t);
    }
    mtx_estimator.unlock();
    mtx_m_buf.lock();
    mtx_state.lock();
    if (estimator_->solver_flag == Estimator::SolverFlag::NON_LINEAR){
      Update();
    }
    mtx_state.unlock();
    mtx_m_buf.unlock();
  }
}

void VinsInterface::Update(){
  latest_time_ = current_time_;
  tmp_P = estimator_->Ps[WINDOW_SIZE];
  tmp_Q = estimator_->Rs[WINDOW_SIZE];
  tmp_V = estimator_->Vs[WINDOW_SIZE];
  tmp_Ba = estimator_->Bas[WINDOW_SIZE];
  tmp_Bg = estimator_->Bgs[WINDOW_SIZE];
  acc_0 = estimator_->acc_0;
  gyr_0 = estimator_->gyr_0;

  std::queue<cartographer::sensor::ImuData> tmp_imu_buf = imu_buf_;
  for (; !tmp_imu_buf.empty(); tmp_imu_buf.pop()){
    Predict(tmp_imu_buf.front());
  }
}

void VinsInterface::Predict(const cartographer::sensor::ImuData& imu){
  double t = cartographer::common::ToSecondsStamp(imu.time);
  if (init_imu_){
    latest_time_ = t;
    init_imu_ = false;
    return;
  }
  double dt = t - latest_time_;
  latest_time_ = t;
  
  double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
  dx = imu.linear_acceleration[0];
  dy = imu.linear_acceleration[1];
  dz = imu.linear_acceleration[2];
  rx = imu.angular_velocity[0];
  ry = imu.angular_velocity[1];
  rz = imu.angular_velocity[2];
  Eigen::Vector3d linear_acceleration{dx, dy, dz};
  Eigen::Vector3d angular_velocity{rx, ry, rz};

  Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator_->g_;

  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
  tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

  Eigen::Vector3d un_acc_1 = 
      tmp_Q * (linear_acceleration - tmp_Ba) - estimator_->g_;

  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

  tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
  tmp_V = tmp_V + dt * un_acc;

  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

}