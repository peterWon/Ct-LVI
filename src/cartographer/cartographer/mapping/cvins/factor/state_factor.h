#ifndef __STATE_FACTOR__
#define __STATE_FACTOR__

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

namespace cvins{
  class NodeStateCostFunction{
     public:
    static ceres::CostFunction* CreateAutoDiffCostFunction(
        const Eigen::Vector3d& P, 
        const Eigen::Quaterniond& Q,
        const Eigen::Vector3d& V, 
        const Eigen::Vector3d& Ba, 
        const Eigen::Vector3d& Bg,
        const std::vector<double>& weights) {
      return new ceres::AutoDiffCostFunction<
          NodeStateCostFunction, 
          ceres::DYNAMIC, /* residuals */
          7, /* translation and rotation variables */
          9 /* velocity and bias variables */>(
            new NodeStateCostFunction(P, Q, V, Ba, Bg, weights), 13);
    }
    
    template <typename T>
    bool operator()(const T* const pose, 
                    const T* const speed,
                    T* const residual) const {
      size_t offset = 0;
      Eigen::Matrix<T,3,1> pos(pose[0], pose[1], pose[2]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rp(residual+offset);
      rp = T(weights_[0]) * (pos - P_.cast<T>()); 
      offset += 3;
      
      Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
      residual[offset] = T(weights_[1]) * (Q_.cast<T>().angularDistance(q)); 
      offset += 1;
      
      Eigen::Matrix<T,3,1> vel(speed[0], speed[1], speed[2]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rv(residual+offset);
      rv = T(weights_[2]) * (vel - V_.cast<T>());
      offset += 3;

      Eigen::Matrix<T,3,1> bias_acc(speed[3], speed[4], speed[5]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rba(residual+offset);
      rba = T(weights_[3]) * (bias_acc - Ba_.cast<T>());
      offset += 3;
      
      Eigen::Matrix<T,3,1> bias_gyr(speed[6], speed[7], speed[8]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rbg(residual+offset);
      rbg = T(weights_[4]) * (bias_gyr - Bg_.cast<T>());

      return true;
    }

  private:
    NodeStateCostFunction(const Eigen::Vector3d& P, 
        const Eigen::Quaterniond& Q, const Eigen::Vector3d& V, 
        const Eigen::Vector3d& Ba, const Eigen::Vector3d& Bg,
        const std::vector<double>& weights)
        : P_(P), Q_(Q), V_(V), Ba_(Ba), Bg_(Bg), weights_(weights) {}

    NodeStateCostFunction(const NodeStateCostFunction&) = delete;
    NodeStateCostFunction& operator=(
        const NodeStateCostFunction&) = delete;

    const Eigen::Vector3d P_;
    const Eigen::Quaterniond Q_;
    const Eigen::Vector3d V_;
    const Eigen::Vector3d Ba_;
    const Eigen::Vector3d Bg_;
    const std::vector<double> weights_;
  };
}

#endif//__STATE_FACTOR__