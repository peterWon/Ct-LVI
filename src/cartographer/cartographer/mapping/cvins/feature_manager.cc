#include "feature_manager.h"

namespace cvins{

int FeaturePerId::endFrame(){
  return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(const ParameterServer * ps): ps_(ps){}

void FeatureManager::clearState(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  feature.clear();
}

int FeatureManager::getFeatureCount(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  int cnt = 0;
  for (auto &it : feature){
    it.used_num = it.feature_per_frame.size();

    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2
        /* && it.estimated_depth > 0 */){
      cnt++;
    }
  }
  return cnt;
}


bool FeatureManager::addFeatureCheckParallax(
    int frame_count, 
    const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, 
    double td, double timestamp){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  double parallax_sum = 0;
  int parallax_num = 0;
  last_track_num = 0;
  for (auto &id_pts : image){
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td, timestamp);

    int feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(), 
        [feature_id](const FeaturePerId &it){
      return it.feature_id == feature_id;
    });

    if (it == feature.end()){
      feature.push_back(FeaturePerId(feature_id, frame_count, f_per_fra.depth));
      feature.back().feature_per_frame.push_back(f_per_fra);
    }else if (it->feature_id == feature_id){
      it->feature_per_frame.push_back(f_per_fra);
      // it would be fine if the carrier is in slow　moving
      // if((!it->is_depth_associated) && (f_per_fra.depth > 0)){
      //   it->is_depth_associated = true;
      //   it->estimated_depth = f_per_fra.depth;
      // }
      last_track_num++;
    }
  }

  if (frame_count < 2 || last_track_num < 20)
    return true;

  for (auto &it_per_id : feature){
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 
        >= frame_count - 1){
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0){
    return true;
  }else{
    return parallax_sum / parallax_num >= ps_->MIN_PARALLAX;
  }
}

void FeatureManager::debugShow(){
  for (auto &it : feature){
    assert(it.feature_per_frame.size() != 0);
    assert(it.start_frame >= 0);
    assert(it.used_num >= 0);

    // ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
    int sum = 0;
    for (auto &j : it.feature_per_frame){
      // ROS_DEBUG("%d,", int(j.is_used));
      sum += j.is_used;
      // printf("(%lf,%lf) ",j.point(0), j.point(1));
    }
    // ROS_ASSERT(it.used_num == sum);
  }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(
    int frame_count_l, int frame_count_r){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature){
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r){
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;

      b = it.feature_per_frame[idx_r].point;
      
      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

void FeatureManager::setDepth(const VectorXd &x){
  // //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  int feature_index = -1;
  for (auto &it_per_id : feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    
    //未三角化或未匹配到合适的激光深度的点不会参与优化
    // if(it_per_id.estimated_depth < 0) continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth < 0){
      it_per_id.solve_flag = 2;
      it_per_id.is_depth_associated = false;
    }else
      it_per_id.solve_flag = 1;
  }
}

void FeatureManager::removeFailures(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  for (auto it = feature.begin(), it_next = feature.begin();
      it != feature.end(); it = it_next){
    it_next++;
    if (it->solve_flag == 2)
      feature.erase(it);
  }
}

void FeatureManager::clearDepth(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  int feature_index = -1;
  for (auto &it_per_id : feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
    // it_per_id.estimated_depth = 1.0 / x(++feature_index);
    it_per_id.estimated_depth = -1.;
    it_per_id.is_depth_associated = false;
  }
}

VectorXd FeatureManager::getDepthVector(){
  //cartographer::common::MutexLocker locker(&mutex_);
  VectorXd dep_vec(getFeatureCount());
  
  //std::scoped_lock lock(mtx_);
  int feature_index = -1;
  for (auto &it_per_id : feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    // if (it_per_id.estimated_depth > 0)
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    //未三角化或未匹配到合适的激光深度的点不参与优化?
    // else
    //   dep_vec(++feature_index) = 1. / ps_->INIT_DEPTH;
    // dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
  }
  return dep_vec;
}

void FeatureManager::triangulate(
    const Matrix3d Rs[], const Vector3d Ps[],
    const Vector3d tic[], const Matrix3d ric[]){
  //cartographer::common::MutexLocker locker(&mutex_);
  ////std::scoped_lock lock(mtx_);
  for (auto &it_per_id : feature){
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;

    if (it_per_id.estimated_depth > 0)
        continue;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    assert(NUM_OF_CAM == 1);
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto &it_per_frame : it_per_id.feature_per_frame){
      imu_j++;

      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j)
        continue;
    }
    assert(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(
      svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    //it_per_id->estimated_depth = -b / A;
    //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

    it_per_id.estimated_depth = svd_method;

    if (it_per_id.estimated_depth < 0.1){
      it_per_id.estimated_depth = ps_->INIT_DEPTH;
      // it_per_id.estimated_depth = -1.;
    }
  }
}

void FeatureManager::removeOutlier(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  int i = -1;
  for (auto it = feature.begin(), it_next = feature.begin();
      it != feature.end(); it = it_next){
    it_next++;
    i += it->used_num != 0;
    if (it->used_num != 0 && it->is_outlier == true){
      feature.erase(it);
    }
  }
}

void FeatureManager::removeBackShiftDepth(
  Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, 
  Eigen::Matrix3d new_R, Eigen::Vector3d new_P){
  for (auto it = feature.begin(), it_next = feature.begin();
      it != feature.end(); it = it_next){
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else{
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2){
        feature.erase(it);
        continue;
      }else{
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = ps_->INIT_DEPTH;
      }
    }
  }
}

/* void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, 
    Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  for (auto it = feature.begin(), it_next = feature.begin();
        it != feature.end(); it = it_next){
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else{
      // feature point and depth in old local camera frame
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      double depth = -1;
      if (it->feature_per_frame[0].depth > 0)
        // if lidar depth available at this frame for feature
        depth = it->feature_per_frame[0].depth;
      else if (it->estimated_depth > 0)
        // if estimated depth available
        depth = it->estimated_depth;

      // delete current feature in the old local camera frame
      it->feature_per_frame.erase(it->feature_per_frame.begin());

      if (it->feature_per_frame.size() < 2){
        // delete feature from feature manager
        feature.erase(it);
        continue;
      }else{
        // feature in cartisian space in old local camera frame
        Eigen::Vector3d pts_i = uv_i * depth; 
        // feautre in cartisian space in world frame
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; 
        // feature in cartisian space in shifted local camera frame
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); 
        double dep_j = pts_j(2);

        // after deletion, the feature has lidar depth in the first of the remaining frame
        if (it->feature_per_frame[0].depth > 0){
          it->estimated_depth = it->feature_per_frame[0].depth;
          it->is_depth_associated = true;
        }else if (dep_j > 0){
          // calculated depth in the current frame
          it->estimated_depth = dep_j;
          it->is_depth_associated = false;
        }else {
          // non-positive depth, invalid
          // it->estimated_depth = ps_->INIT_DEPTH;
          it->is_depth_associated = false;
        }
      }
    }
  }
} */

void FeatureManager::removeBack(){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  for (auto it = feature.begin(), it_next = feature.begin();
      it != feature.end(); it = it_next){
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else{
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count){
  //cartographer::common::MutexLocker locker(&mutex_);
  //std::scoped_lock lock(mtx_);
  for (auto it = feature.begin(), it_next = feature.begin(); 
    it != feature.end(); it = it_next){
    it_next++;

    if (it->start_frame == frame_count){
      it->start_frame--;
    }else{
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1)
          continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0)
          feature.erase(it);
    }
  }
}

double FeatureManager::compensatedParallax2(
    const FeaturePerId &it_per_id, int frame_count){
  //check the second last frame is keyframe or not
  //parallax betwwen seconde last frame and third last frame
  const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[
    frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[
    frame_count - 1 - it_per_id.start_frame];

  double ans = 0;
  Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  Vector3d p_i_comp;

  p_i_comp = p_i;
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  ans = max(ans, sqrt(min(du * du + dv * dv, 
                          du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}

}