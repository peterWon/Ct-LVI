#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "glog/logging.h"
#include "cartographer/sensor/imu_data.h"


// for debug, to remove
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

namespace cvins{

class VinsInterface{
public:
  VinsInterface(const ParameterServer* ps, 
                const std::shared_ptr<FeatureManager> fm){
    estimator_.reset(new Estimator(ps, fm));
    nh = ros::NodeHandle("~");
    pub_path = nh.advertise<nav_msgs::Path>("vins_path", 1000);
  }
  ~VinsInterface(){}
  
  bool Initialized();

  // for initialization and relocalization.
  std::vector<double> GetImageStampsInWindow();
  void SetEvaluatedStates(const std::map<double, State>& states_in_local);
  
  void AddImuData(const cartographer::sensor::ImuData& imu_data);
  void AddImageData(const VinsFrameFeature& img_feature);

  void Run();
  
  void GetFeatures2D();
  void GetFeatures3D();

  bool GetCurrentStates(double& timestamp, 
      Eigen::Vector3d& p, Eigen::Vector3d& v, Eigen::Quaterniond& q,
      Eigen::Vector3d& ba, Eigen::Vector3d& bg);
  bool GetCurrentStates(std::vector<State>& states);
  void GetFeaturesInWindow(const std::shared_ptr<FeatureManager>& fm);
private:
  ros::NodeHandle nh;
  ros::Publisher pub_odometry, pub_path;
  nav_msgs::Path path;
   
  std::vector<std::pair<std::vector<cartographer::sensor::ImuData>,
    VinsFrameFeature>> getMeasurements();

  void Update();
  void Predict(const cartographer::sensor::ImuData& imu_msg);

  std::shared_ptr<Estimator> estimator_;

  // used for cache features from VIO, for LVI retrivel.
  // FeatureManager feature_mgr_cache_;

  const int camera_id_ = 0;//currently, we only process one camera data.

  double current_time_ = -1;
  std::queue<cartographer::sensor::ImuData> imu_buf_;
  std::queue<VinsFrameFeature> feature_buf_;
  
  int sum_of_wait = 0;
  
  std::condition_variable con;
  std::mutex mtx_state;
  std::mutex mtx_feature;
  std::mutex mtx_m_buf;
  std::mutex mtx_estimator;

  double latest_time_;
  Eigen::Vector3d tmp_P;
  Eigen::Quaterniond tmp_Q;
  Eigen::Vector3d tmp_V;
  Eigen::Vector3d tmp_Ba;
  Eigen::Vector3d tmp_Bg;
  Eigen::Vector3d acc_0;
  Eigen::Vector3d gyr_0;

  bool init_feature_ = false;
  bool init_imu_ = true;
  bool inited_ = false;
  double last_imu_t_ = 0;

};

}