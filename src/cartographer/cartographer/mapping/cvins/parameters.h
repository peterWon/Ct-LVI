#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>

namespace cvins{

static const int NUM_OF_CAM = 1;
static const int WINDOW_SIZE = 10;
static const int NUM_OF_F = 1000;

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

class ParameterServer{
  //TODO(wz): singleton
public:
  ParameterServer(){}
  ~ParameterServer(){}

  //////////////////VINS/////////////////////
  double GRAVITY_NORM = -9.805;
  double FOCAL_LENGTH = 460.;
  int ESTIMATE_EXTRINSIC = 0;

  double INIT_DEPTH = 5.;
  double MIN_PARALLAX = 10.;

  double ACC_N, ACC_W;
  double GYR_N, GYR_W;

  std::vector<Eigen::Matrix3d> RIC;
  std::vector<Eigen::Vector3d> TIC;
  Eigen::Vector3d G = Eigen::Vector3d(0,0,9.805);

  double BIAS_ACC_THRESHOLD = 2.5;
  double BIAS_GYR_THRESHOLD = 1.0;
  double SOLVER_TIME = 0.04;
  int NUM_ITERATIONS = 8;

  double TD = 0.;
  double TR = 0.;
  int ESTIMATE_TD = 0;
  int ROLLING_SHUTTER = 0;
  double ROW = 0;
  double COL = 0;
  
  bool ENABLE_STATE_FACTOR = false;
  bool FIX_DEPTH_ASSOCIATED = false;

  double WEIGHT_POS = 30;
  double WEIGHT_ROT = 30;
  double WEIGHT_VEL = 10;
  double WEIGHT_BA = 10;
  double WEIGHT_BG = 10;
  
  // feature tracker
  int MAX_CNT = 150;
  
  ///////////CalibParamManager/////////////
  Eigen::Vector3d p_LinI;
  Eigen::Quaterniond q_LtoI;
  
  Eigen::Vector3d p_CinI;
  Eigen::Quaterniond q_CtoI;

  double time_offset_lidar = 0.;
  double time_offset_camera = 0.;

  /// opt weight
  double global_opt_gyro_weight;

  double global_opt_acce_weight;

  double global_opt_lidar_weight;

  double global_opt_lidar_rot_weight;
  double global_opt_lidar_pos_weight;
  
  double global_opt_imu_rot_weight;
  double global_opt_imu_pos_weight;
  double global_opt_imu_vel_weight;
  
  // for test only
  double imu_rot_weight_li;
  double imu_pos_weight_li;
  double imu_vel_weight_li;
  
  double imu_rot_weight_vi;
  double imu_pos_weight_vi;
  double imu_vel_weight_vi;
  
  double global_opt_imu_ba_weight;
  double global_opt_imu_bg_weight;

  double global_opt_cam_rot_weight;
  double global_opt_cam_pos_weight;
  double global_opt_cam_uv_weight;

  double gyro_bias_uppper_bound;
  double acce_bias_uppper_bound;

  bool enable_debug;

  //////////////ImuStateEstimator//////////////////////

  double accel_excite_threshold = 0.5;
  double gyro_excite_threshold = 0.5;
  double sample_num = 5;

  ////////////////////////////////////////
  double knot_space = 0.1;
  int num_anchor = 30;
  int num_bins = 180;
  double image_sampling_ratio = 1.0;
  double max_time_cost = 0.05;
  
  
  std::string mask = "";
  
  double visual_depth_upper_bound = 150.;
  double visual_depth_lower_bound = 0.5;
  
  bool remove_outlier = false;
  bool reweight_outlier = false;
  double reweight_outlier_scale = 0.3;
  double reproject_outlier_threshold = 0.3;

  double seg_threshold = 0.1;
  double seg_inlier_ratio = 0.8;

  int random_drop_num = 0;
  double ndt_resolution = 1.0;

  int visual_terms_number = 200;

  bool enable_imu_interpolated_pose = false;
  bool enable_imu_predicted_pose = false;
 
  // deprecated 
  int n_grid_rows = 5;
  int n_grid_cols = 5;
  double ransac_threshold = 0.00000002;

  double g_norm_for_debug = -9.805;

  double vio_time_window = 0.3;
  double outlier_ratio_to_lock = 0.1;
  double xy_movement_to_enable_vio = 0.5;
  double start_movement_excitation = 1.0;
  double outlier_ratio_threshold_to_reset = 1.0;
  bool enable_depth_assiciation = true;
  bool optimize_lvi_together = false;
  ////////////////////////////////////////
  void readParameters(const YAML::Node& cfg);
};
  
}//end namespace cvins