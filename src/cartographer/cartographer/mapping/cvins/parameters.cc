#include "parameters.h"
#include "glog/logging.h"

namespace cvins{

void ParameterServer::readParameters(const YAML::Node& cfg){
  const auto& vins_node = cfg["VINS"];
  FOCAL_LENGTH = vins_node["focal_length"].as<double>();
  // WINDOW_SIZE = vins_node["window_size"].as<int>();
  // NUM_OF_F = vins_node["num_of_f"].as<int>();

  INIT_DEPTH = vins_node["init_depth"].as<double>();
  MIN_PARALLAX = vins_node["min_parallax"].as<double>();
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

  ACC_N = vins_node["acc_n"].as<double>();
  ACC_W = vins_node["acc_w"].as<double>();
  GYR_N = vins_node["gyr_n"].as<double>();
  GYR_W = vins_node["gyr_w"].as<double>();
  BIAS_ACC_THRESHOLD = vins_node["bias_acc_threshold"].as<double>();
  BIAS_GYR_THRESHOLD = vins_node["bias_gyr_threshold"].as<double>();
  SOLVER_TIME = vins_node["solver_time"].as<double>();
  NUM_ITERATIONS = vins_node["num_iterations"].as<int>();
  int ESTIMATE_TD = vins_node["estimate_td"].as<int>();
  int ROLLING_SHUTTER = vins_node["rolling_shutter"].as<int>();
  TD = vins_node["td"].as<double>();
  TR = vins_node["tr"].as<double>();
  
  G.z() = vins_node["g_norm"].as<double>();
  
  ENABLE_STATE_FACTOR = vins_node["enable_state_factor"].as<bool>();
  FIX_DEPTH_ASSOCIATED = vins_node["fix_depth_associated"].as<bool>();
  
  WEIGHT_POS = vins_node["weight_pos"].as<double>();
  WEIGHT_ROT = vins_node["weight_rot"].as<double>();
  WEIGHT_VEL = vins_node["weight_vel"].as<double>();
  WEIGHT_BA = vins_node["weight_ba"].as<double>();
  WEIGHT_BG = vins_node["weight_bg"].as<double>();

  // parameters not in 'VINS' scope
  COL = cfg["image_width"].as<double>();
  ROW = cfg["image_height"].as<double>();
  
  std::vector<double> params_vec;
  params_vec.resize(9);
  for (size_t i = 0; i < params_vec.size(); i++) {
    params_vec.at(i) = cfg["extrinsic_CtoI"]["Rot"][i].as<double>();
  }
  RIC.resize(NUM_OF_CAM);
  RIC[0] << params_vec[0], params_vec[1], params_vec[2], params_vec[3],
        params_vec[4], params_vec[5], params_vec[6], params_vec[7],
        params_vec[8];
  params_vec.resize(3);
  for (size_t i = 0; i < params_vec.size(); i++) {
    params_vec.at(i) = cfg["extrinsic_CtoI"]["Trans"][i].as<double>();
  } 
  //todo
  TIC.resize(NUM_OF_CAM);
  TIC[0] << params_vec[0], params_vec[1], params_vec[2];

  MAX_CNT = vins_node["max_cnt"].as<int>();
  ///////////////////////////////////////////////////////////////////
  if (cfg["extrinsic_LtoI"]) {
    std::vector<double> params_vec;
    params_vec.resize(3);
    for (size_t i = 0; i < params_vec.size(); i++) {
      params_vec.at(i) = cfg["extrinsic_LtoI"]["Trans"][i].as<double>();
    }
    p_LinI << params_vec[0], params_vec[1], params_vec[2];

    params_vec.resize(9);
    for (size_t i = 0; i < params_vec.size(); i++) {
      params_vec.at(i) = cfg["extrinsic_LtoI"]["Rot"][i].as<double>();
    }
    Eigen::Matrix3d rot;
    rot << params_vec[0], params_vec[1], params_vec[2], params_vec[3],
        params_vec[4], params_vec[5], params_vec[6], params_vec[7],
        params_vec[8];

    q_LtoI = Eigen::Quaterniond(rot);
    q_LtoI.normalized();
  }
  if (cfg["extrinsic_CtoI"]) {
    std::vector<double> params_vec;
    params_vec.resize(3);
    for (size_t i = 0; i < params_vec.size(); i++) {
      params_vec.at(i) 
        = cfg["extrinsic_CtoI"]["Trans"][i].as<double>();
    }
    p_CinI << params_vec[0], params_vec[1], params_vec[2];

    params_vec.resize(9);
    for (size_t i = 0; i < params_vec.size(); i++) {
      params_vec.at(i) = cfg["extrinsic_CtoI"]["Rot"][i].as<double>();
    }
    Eigen::Matrix3d rot;
    rot << params_vec[0], params_vec[1], params_vec[2], params_vec[3],
        params_vec[4], params_vec[5], params_vec[6], params_vec[7],
        params_vec[8];

    q_CtoI = Eigen::Quaterniond(rot);
    q_CtoI.normalized();
  }
  if (cfg["time_offset_lidar"]) {
    time_offset_lidar = cfg["time_offset_lidar"].as<double>();
  }
  if (cfg["time_offset_camera"]) {
    time_offset_camera = cfg["time_offset_camera"].as<double>();
  }
  /// estimate weight param
  global_opt_gyro_weight = cfg["gyro_weight"].as<double>();
  global_opt_acce_weight = cfg["accel_weight"].as<double>();
  global_opt_lidar_weight = cfg["lidar_weight"].as<double>();
  
  global_opt_lidar_rot_weight = cfg["lidar_rot_weight"].as<double>();
  global_opt_lidar_pos_weight = cfg["lidar_pos_weight"].as<double>();
  global_opt_cam_rot_weight = cfg["cam_rot_weight"].as<double>();
  global_opt_cam_pos_weight = cfg["cam_pos_weight"].as<double>();
  
  global_opt_cam_uv_weight = cfg["cam_uv_weight"].as<double>();
  global_opt_imu_pos_weight = cfg["imu_pos_weight"].as<double>();
  global_opt_imu_vel_weight = cfg["imu_vel_weight"].as<double>();
  global_opt_imu_rot_weight = cfg["imu_rot_weight"].as<double>();
  global_opt_imu_ba_weight = cfg["imu_ba_weight"].as<double>();
  global_opt_imu_bg_weight = cfg["imu_bg_weight"].as<double>();
  gyro_bias_uppper_bound = cfg["gyro_bias_uppper_bound"].as<double>();
  acce_bias_uppper_bound = cfg["acce_bias_uppper_bound"].as<double>();
  

  imu_rot_weight_li = cfg["imu_rot_weight_li"].as<double>();
  imu_pos_weight_li = cfg["imu_pos_weight_li"].as<double>();
  imu_vel_weight_li = cfg["imu_vel_weight_li"].as<double>();
  imu_rot_weight_vi = cfg["imu_rot_weight_vi"].as<double>();
  imu_pos_weight_vi = cfg["imu_pos_weight_vi"].as<double>();
  imu_vel_weight_vi = cfg["imu_vel_weight_vi"].as<double>();

  max_time_cost = cfg["max_time_cost"].as<double>();

  reproject_outlier_threshold = cfg["reproject_outlier_threshold"].as<double>();
  reweight_outlier_scale = cfg["reweight_outlier_scale"].as<double>();
  remove_outlier = cfg["remove_outlier"].as<bool>();
  reweight_outlier = cfg["reweight_outlier"].as<bool>();
  
  enable_debug = cfg["enable_debug"].as<bool>();
  //////////////ImuStateEstimator//////////////////////
  if (cfg["accel_excite_threshold"]) {
    accel_excite_threshold = cfg["accel_excite_threshold"].as<double>();
  }
  if (cfg["gyro_excite_threshold"]) {
    gyro_excite_threshold = cfg["gyro_excite_threshold"].as<double>();
  }
  if (cfg["sample_num"]) {
    sample_num = cfg["sample_num"].as<double>();
  }
  if (cfg["visual_terms_number"]) {
    visual_terms_number = cfg["visual_terms_number"].as<int>();
  }
  ///////////////////LocalTrajectoryBuilder////////////////////////////////
  num_anchor = cfg["num_anchor"].as<int>();
  num_bins = cfg["num_bins"].as<int>();
  knot_space =  cfg["knot_space"].as<double>();
  enable_imu_predicted_pose =  cfg["enable_imu_predicted_pose"].as<bool>();
  enable_imu_interpolated_pose =  cfg["enable_imu_interpolated_pose"].as<bool>();
  image_sampling_ratio =  cfg["image_sampling_ratio"].as<double>();
  mask =  cfg["mask"].as<std::string>();
  visual_depth_upper_bound =  cfg["visual_depth_upper_bound"].as<double>();
  visual_depth_lower_bound =  cfg["visual_depth_lower_bound"].as<double>();
  seg_threshold =  cfg["seg_threshold"].as<double>();
  seg_inlier_ratio =  cfg["seg_inlier_ratio"].as<double>();

  random_drop_num =  cfg["random_drop_num"].as<int>();
  ndt_resolution =  cfg["ndt_resolution"].as<double>();

  GRAVITY_NORM =  cfg["gravity_norm"].as<double>();
  g_norm_for_debug =  cfg["g_norm_for_debug"].as<double>();
  vio_time_window =  cfg["vio_time_window"].as<double>();
  outlier_ratio_to_lock =  cfg["outlier_ratio_to_lock"].as<double>();
  xy_movement_to_enable_vio =  cfg["xy_movement_to_enable_vio"].as<double>();
  start_movement_excitation =  cfg["start_movement_excitation"].as<double>();
  outlier_ratio_threshold_to_reset =  cfg["outlier_ratio_threshold_to_reset"].as<double>();
  enable_depth_assiciation =  cfg["enable_depth_assiciation"].as<bool>();
  optimize_lvi_together =  cfg["optimize_lvi_together"].as<bool>();

}
  
}