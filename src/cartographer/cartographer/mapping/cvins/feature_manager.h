#ifndef __CVINS_FEATURE_MANAGER_H__
#define __CVINS_FEATURE_MANAGER_H__

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <mutex>
#include <eigen3/Eigen/Dense>

#include "parameters.h"
#include "cartographer/common/mutex.h"

using namespace std;
using namespace Eigen;

namespace cvins{
  ///////////////////////////
  struct State{
    double timestamp = -1.;
    Eigen::Vector3d P;
    Eigen::Vector3d V;
    Eigen::Quaterniond Q;
    Eigen::Vector3d Ba;
    Eigen::Vector3d Bg;
  };  

  struct VinsFeature{
    int feature_id;
    double x;
    double y;
    double z;
    double u;
    double v;
    double velocity_x;
    double velocity_y;
    double depth;
  };

  struct VinsFrameFeature{
    double timestamp;
    std::vector<VinsFeature> features;
  };

  ////////////////////////////
  class FeaturePerFrame{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, 
        double td, double timestamp){
      point.x() = _point(0);
      point.y() = _point(1);
      point.z() = _point(2);
      uv.x() = _point(3);
      uv.y() = _point(4);
      velocity.x() = _point(5); 
      velocity.y() = _point(6); 
      depth = _point(7); 
      cur_td = td;
      timestamp_ = timestamp;

      parallax = 0.;
      dep_gradient = 0.;
      z = 0.;
      is_used = false;
      to_reweight = false;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double depth;

    double z;
    bool is_used;
    bool to_reweight;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
    double timestamp_;
  };

  class FeaturePerId{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    
    bool is_depth_associated = false;
    
    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame, double depth_associated)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(depth_associated), solve_flag(0){
      is_depth_associated = depth_associated > 0 ? true : false;
      is_outlier = false;
    }

    FeaturePerId(const FeaturePerId& obj):feature_id(obj.feature_id){
      // feature_id = obj.feature_id;
      start_frame = obj.start_frame;
      feature_per_frame = obj.feature_per_frame;

      used_num = obj.used_num;
      is_outlier = obj.is_outlier;
      is_margin = obj.is_margin;
      estimated_depth = obj.estimated_depth;
      solve_flag = obj.solve_flag; 
      // 0 haven't solve yet; 1 solve succ; 2 solve fail;
      
      is_depth_associated = obj.is_depth_associated;
      
      gt_p = obj.gt_p;
    }
    
    int endFrame();
  };

  class FeatureManager{
  public:
    FeatureManager(const ParameterServer* ps);


    void clearState() /*REQUIRES(mutex_)*/;

    int getFeatureCount() /*REQUIRES(mutex_)*/;

    bool addFeatureCheckParallax (
      int frame_count, 
      const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, 
      double td, double timestamp) /*REQUIRES(mutex_)*/;
    
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(
      int frame_count_l, int frame_count_r) /*REQUIRES(mutex_)*/;

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x) /*REQUIRES(mutex_)*/;
    void removeFailures() /*REQUIRES(mutex_)*/;
    void clearDepth() /*REQUIRES(mutex_)*/;
    VectorXd getDepthVector() /*REQUIRES(mutex_)*/;
    void triangulate(const Matrix3d Rs[], const Vector3d Ps[], 
                    const Vector3d tic[], const Matrix3d ric[]) /*REQUIRES(mutex_)*/;
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, 
                              Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
                              /*REQUIRES(mutex_)*/;
    void removeBack() /*REQUIRES(mutex_)*/;
    void removeFront(int frame_count) /*REQUIRES(mutex_)*/;
    void removeOutlier() /*REQUIRES(mutex_)*/;
    
    list<FeaturePerId> feature /*GUARDED_BY(mutex_)*/;
    int last_track_num /*GUARDED_BY(mutex_)*/;
    std::mutex mtx_;
  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const ParameterServer* ps_;
    // mutable cartographer::common::Mutex mutex_;
  };
}
#endif