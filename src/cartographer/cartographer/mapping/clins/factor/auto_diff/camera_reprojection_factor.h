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

#ifndef AUTO_DIFF_CAMERA_FEATURE_FACTOR
#define AUTO_DIFF_CAMERA_FEATURE_FACTOR

#include <memory>
#include <basalt/spline/ceres_spline_helper_jet.h>

#include <Eigen/Core>
#include <sophus/so3.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// #include <camera_models/Camera.h>
#include <cartographer/mapping/sfm/sfm.h>

namespace clins {

using namespace basalt;


template <int _N>
class ReprojectionFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionFactor(const std::shared_ptr<sfm::Observation>& ref,
                     const std::shared_ptr<sfm::Observation>& obs,
                     const SplineMeta<_N>& spline_meta,
                     double weight)
      : reference_(ref), measurement_(obs),
        spline_meta_(spline_meta), weight_(weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    
    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    const T eps = T(1e-15);
    T time_offset = T(sKnots[Kont_offset + 2][0]);

    T t[2];
    t[0] = T(reference_->view()->t0());
    t[1] = T(measurement_->view()->t0());
    
    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    Vec3T p_IkinG[2];
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &p_IkinG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &p_IkinG[1]);
    
    
    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_CinI(sKnots[Kont_offset + 1]);

    // T inverse_depth = T(reference_->landmark()->inverse_depth());
    T inverse_depth = T(sKnots[Kont_offset + 3][0]);

    // Here, I0 is the reference frame.
    // SO3T R_IkToI0 = R_IkToG[0].inverse() * R_IkToG[1];
    // Vec3T p_IkinI0 = R_IkToG[0].inverse() * (p_IkinG[1] - p_IkinG[0]);
    
    Eigen::Quaternion<T> Q_Ref2G = R_IkToG[0].unit_quaternion();
    Eigen::Quaternion<T> Q_Obs2G = R_IkToG[1].unit_quaternion();

    Eigen::Quaternion<T> Q_CtoI = R_CtoI.unit_quaternion();
    Eigen::Quaternion<T> Q_ItoC = Q_CtoI.conjugate();
    
    Vec3T p_IinC = Q_ItoC * (-p_CinI);
    Vec2T uv = reference_->uv().cast<T>();
    Vec3T pts_camera_i;
    pts_camera_i << uv[0]/inverse_depth, uv[1]/inverse_depth, 1./inverse_depth;
    
    Vec3T pts_imu_i = Q_CtoI * pts_camera_i + p_CinI;
    Vec3T pts_w = Q_Ref2G * pts_imu_i + p_IkinG[0];
    Vec3T pts_imu_j = Q_Obs2G.inverse() * (pts_w - p_IkinG[1]);
    Vec3T pts_camera_j = Q_ItoC * (pts_imu_j - p_CinI);
    T dep_j = pts_camera_j.z();
    Vec2T uv_hat;
    Vec2T uv_j = measurement_->uv().cast<T>();
    uv_hat << pts_camera_j[0] / dep_j - uv_j[0],
              pts_camera_j[1] / dep_j - uv_j[1]; 

    //assume that the uv are on the normalized plane. global shutter camera.
    /* Vec2T y = reference_->uv().cast<T>();
    Vec3T y_norm;
    y_norm << y[0], y[1], 1.0;
    // camera_->liftProjective(y, y_norm);
    
    //空间点归一化平面坐标转到IMU系
    Vec3T X_ref = Q_CtoI * (y_norm - inverse_depth * p_IinC);
    //IMU系转到世界系
    Vec3T X = Q_Ref2G * X_ref + p_IkinG[0] * inverse_depth;
    //观测帧IMU坐标系下的归一化坐标
    Vec3T X_obs = Q_Obs2G.conjugate() * (X - inverse_depth * p_IkinG[1]);
    //观测帧下的相空间归一化坐标, q_ct, p_ct是IMU到Camera!
    Vec3T X_camera = Q_ItoC * X_obs + p_IinC * inverse_depth;
    
    Vec2T uv_obs = measurement_->uv().cast<T>();
    Vec2T uv_hat;
    uv_hat << X_camera[0] / X_camera[2], X_camera[1] / X_camera[2];
    uv_hat = uv_hat - uv_obs; */

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    residuals.template block<2, 1>(0, 0) = Eigen::Matrix<T, 2, 1>(uv_hat);

    residuals = T(weight_) * residuals;

    return true;
  }

 private:
  std::shared_ptr<sfm::Observation> reference_;
  std::shared_ptr<sfm::Observation> measurement_;
  SplineMeta<_N> spline_meta_;
  // camodocal::CameraPtr camera_;
  double weight_;
  double inv_dt_;
};


template <int _N>
class ReprojectionFactorV1 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionFactorV1(const double t_ref,
                      const double t_obs,
                      const Eigen::Vector3d& pt_ref,
                      const Eigen::Vector3d& pt_obs,
                      const SplineMeta<_N>& spline_meta,
                      double weight)
      : t_ref_(t_ref), t_obs_(t_obs),
         reference_(pt_ref), measurement_(pt_obs),
        spline_meta_(spline_meta), weight_(weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    
    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    const T eps = T(1e-15);
    T time_offset = T(sKnots[Kont_offset + 2][0]);

    T t[2];
    t[0] = T(t_ref_);
    t[1] = T(t_obs_);
    
    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    Vec3T p_IkinG[2];
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &p_IkinG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &p_IkinG[1]);
    
    
    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_CinI(sKnots[Kont_offset + 1]);

    T inverse_depth = T(sKnots[Kont_offset + 3][0]);
    
    Eigen::Quaternion<T> Q_Ref2G = R_IkToG[0].unit_quaternion();
    Eigen::Quaternion<T> Q_Obs2G = R_IkToG[1].unit_quaternion();

    Eigen::Quaternion<T> Q_CtoI = R_CtoI.unit_quaternion();
    Eigen::Quaternion<T> Q_ItoC = Q_CtoI.conjugate();
    
    Vec3T p_IinC = Q_ItoC * (-p_CinI);
    
    //assume that the uv are on the normalized plane. global shutter camera.
    // Vec2T y = reference_.head<2>().cast<T>();
    // Vec3T y_norm;
    // y_norm << y[0], y[1], 1.0;
    
    Vec3T pts_camera_i = reference_.cast<T>() / inverse_depth;
    Vec3T pts_imu_i = Q_CtoI * pts_camera_i + p_CinI;
    Vec3T pts_w = Q_Ref2G * pts_imu_i + p_IkinG[0];
    Vec3T pts_imu_j = Q_Obs2G.inverse() * (pts_w - p_IkinG[1]);
    Vec3T pts_camera_j = Q_ItoC * (pts_imu_j - p_CinI);
    T dep_j = pts_camera_j.z();
    Vec2T uv_hat;
    uv_hat << pts_camera_j[0] / dep_j - T(measurement_[0]/measurement_[2]),
              pts_camera_j[1] / dep_j - T(measurement_[1]/measurement_[2]); 

    //空间点归一化平面坐标转到IMU系
    // Vec3T X_ref_i = Q_CtoI * (y_norm - inverse_depth * p_IinC);
    // //IMU系转到世界系
    // Vec3T X = Q_Ref2G * X_ref_i + p_IkinG[0] * inverse_depth;
    // //观测帧IMU坐标系下的归一化坐标
    // Vec3T X_obs_i = Q_Obs2G.conjugate() * (X - inverse_depth * p_IkinG[1]);
    // //观测帧下的相空间归一化坐标, q_ct, p_ct是IMU到Camera!
    // Vec3T X_obs_c = Q_ItoC * X_obs_i + p_IinC * inverse_depth;
    
    // Vec2T uv_hat;
    // uv_hat << X_obs_c[0] / X_obs_c[2], X_obs_c[1] / X_obs_c[2];
    // uv_hat = uv_hat - measurement_.head<2>().cast<T>();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    residuals.template block<2, 1>(0, 0) = Eigen::Matrix<T, 2, 1>(uv_hat);

    residuals = T(weight_) * residuals;

    return true;
  }

 private:
  double t_ref_;
  double t_obs_;
  Eigen::Vector3d reference_;
  Eigen::Vector3d measurement_;
  SplineMeta<_N> spline_meta_;
  double weight_;
  double inv_dt_;
};


template <int _N>
class CameraPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraPoseFactor(const PoseData& pose_data, const SplineMeta<_N>& spline_meta,
                  double pos_weight, double rot_weight, bool estimate_scale)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight),
        estimate_scale_(estimate_scale) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t Kont_offset = 2 * spline_meta_.NumParameters();

    T t_offset = sKnots[Kont_offset + 2][0];
    T t = T(pose_data_.timestamp) + t_offset;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_IkToG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T p_IkinG;
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(sKnots + P_offset, u,
                                                         inv_dt_, &p_IkinG);

    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_CinI(sKnots[Kont_offset + 1]);
    
    T scale;
    if(estimate_scale_){
      scale = sKnots[Kont_offset + 3][0];
    }

    SO3T R_CkToG = R_IkToG * R_CtoI;
    Vec3T p_CkinG = R_IkToG * p_CinI + p_IkinG;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_CkToG * pose_data_.orientation.inverse()).log();
    
    if(estimate_scale_){
      residuals.template block<3, 1>(3, 0) = 
        T(pos_weight_) * (scale * p_CkinG - pose_data_.position);
    }else{
      residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_CkinG - pose_data_.position);
    }
    
    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
  bool estimate_scale_;
};

template <int _N>
class RelativeCameraPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativeCameraPoseFactor(const double t_ref, 
                   const double t_obs,
                   const Eigen::Matrix3d& relative_rot, 
                   const Eigen::Vector3d& relative_pos, 
                   const SplineMeta<_N>& spline_meta,
                   double pos_weight, 
                   double rot_weight, 
                   bool estimate_scale)
      : t_ref_(t_ref),
        t_obs_(t_obs),
        relative_rot_(relative_rot),
        relative_pos_(relative_pos),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight),
        estimate_scale_(estimate_scale) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t Kont_offset = 2 * spline_meta_.NumParameters();

    T t_offset = sKnots[Kont_offset + 2][0]; //time offset
    T t_ref = T(t_ref_) + t_offset;
    T t_obs = T(t_obs_) + t_offset;

    size_t R_offset_ref, R_offset_obs;  // should be zero if not estimate time offset
    size_t P_offset_ref, P_offset_obs;
    T u_ref, u_obs;
    spline_meta_.ComputeSplineIndex(t_ref, R_offset_ref, u_ref);
    spline_meta_.ComputeSplineIndex(t_obs, R_offset_obs, u_obs);
    P_offset_ref = R_offset_ref + spline_meta_.NumParameters();
    P_offset_obs = R_offset_obs + spline_meta_.NumParameters();

    SO3T R_IkToG_ref, R_IkToG_obs;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset_ref, u_ref, inv_dt_, &R_IkToG_ref);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset_obs, u_obs, inv_dt_, &R_IkToG_obs);

    Vec3T p_IkinG_ref, p_IkinG_obs;
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset_ref, u_ref, inv_dt_, &p_IkinG_ref);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset_obs, u_obs, inv_dt_, &p_IkinG_obs);

    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]); //rot
    Eigen::Map<Vec3T const> const p_CinI(sKnots[Kont_offset + 1]); //pos
    
    T scale;
    if(estimate_scale_) scale = sKnots[Kont_offset + 3][0]; //scale
    

    SO3T R_CkToG_ref = R_IkToG_ref * R_CtoI;
    SO3T R_CkToG_obs = R_IkToG_obs * R_CtoI;
    Vec3T p_CkinG_ref = R_IkToG_ref * p_CinI + p_IkinG_ref;
    Vec3T p_CkinG_obs = R_IkToG_obs * p_CinI + p_IkinG_obs;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (
          (R_CkToG_ref.inverse() * R_CkToG_obs)*
          SO3T(relative_rot_.cast<T>()).inverse()).log();
    
    if(estimate_scale_){
      residuals.template block<3, 1>(3, 0) = 
        T(pos_weight_) * (scale * relative_pos_.cast<T>() - 
          R_CkToG_ref.inverse() * (p_CkinG_obs - p_CkinG_ref));
    }else{
      residuals.template block<3, 1>(3, 0) = 
        T(pos_weight_) * (relative_pos_.cast<T>() - 
          R_CkToG_ref.inverse() * (p_CkinG_obs - p_CkinG_ref));
    }
    
    return true;
  }

 private:
  double t_ref_;
  double t_obs_;
  Eigen::Matrix3d relative_rot_;
  Eigen::Vector3d relative_pos_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
  bool estimate_scale_;
};

template <int _N>
class RelativeCameraRotationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativeCameraRotationFactor(const double t_ref, 
                   const double t_obs,
                   const Eigen::Matrix3d& relative_rot, 
                   const SplineMeta<_N>& spline_meta,
                   double rot_weight)
      : t_ref_(t_ref),
        t_obs_(t_obs),
        relative_rot_(relative_rot),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight){
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;
    
    // caution: the offset should be consistant with the order defined in "AddCameraRelativeRotationMeasurement" of TrajectoryEstimator.
    size_t Kont_offset = spline_meta_.NumParameters();

    T t_offset = sKnots[Kont_offset + 1][0]; //time offset
    T t_ref = T(t_ref_) + t_offset;
    T t_obs = T(t_obs_) + t_offset;

    size_t R_offset_ref, R_offset_obs;  // should be zero if not estimate time offset
    T u_ref, u_obs;
    spline_meta_.ComputeSplineIndex(t_ref, R_offset_ref, u_ref);
    spline_meta_.ComputeSplineIndex(t_obs, R_offset_obs, u_obs);

    SO3T R_IkToG_ref, R_IkToG_obs;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset_ref, u_ref, inv_dt_, &R_IkToG_ref);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset_obs, u_obs, inv_dt_, &R_IkToG_obs);

    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]); //rot
    
    SO3T R_CkToG_ref = R_IkToG_ref * R_CtoI;
    SO3T R_CkToG_obs = R_IkToG_obs * R_CtoI;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (
          (R_CkToG_ref.inverse() * R_CkToG_obs)*
          SO3T(relative_rot_.cast<T>()).inverse()).log();
    
    return true;
  }

 private:
  double t_ref_;
  double t_obs_;
  Eigen::Matrix3d relative_rot_;
  SplineMeta<_N> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class CameraOrientationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraOrientationFactor(const PoseData& pose_data,
                         const SplineMeta<_N>& spline_meta, double rot_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    T t = T(pose_data_.timestamp);

    size_t R_offset;  // should be zero if not estimate time offset
    T u;
    spline_meta_.ComputeSplineIndex(t, R_offset, u);

    SO3T R_IkToG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    int Kont_offset = spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_CtoI(sKnots[Kont_offset]);

    SO3T R_CkToG = R_IkToG * R_CtoI;

    Eigen::Map<Tangent> residuals(sResiduals);
    residuals =
        T(rot_weight_) * (R_CkToG * pose_data_.orientation.inverse()).log();

    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

}  // namespace clins

#endif
