-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "basic_config_3d.lua"

options.tracking_frame = "imu"
options.sensor_type = "ouster"
options.num_point_clouds = 1

options.use_image = true
options.enable_cache_range_data = false

POSE_GRAPH.optimization_problem.huber_scale = 1e5
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.linear_xy_search_window = 2.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.linear_z_search_window = 2.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.angular_search_window = math.rad(30.)

TRAJECTORY_BUILDER_3D.submaps.num_range_data= 100
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_range = 15.

POSE_GRAPH.optimize_every_n_nodes = 0
POSE_GRAPH.max_radius_eable_loop_detection = 5.
POSE_GRAPH.num_close_submaps_loop_with_initial_value = 30

TRAJECTORY_BUILDER_3D.min_range = 1.0
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.2

TRAJECTORY_BUILDER_3D.enable_gravity_factor = false
TRAJECTORY_BUILDER_3D.imu.prior_gravity_noise = 0.1
 
TRAJECTORY_BUILDER_3D.scan_period = 0.1
TRAJECTORY_BUILDER_3D.eable_mannually_discrew = false
TRAJECTORY_BUILDER_3D.frames_for_static_initialization = 10
TRAJECTORY_BUILDER_3D.frames_for_dynamic_initialization = 7
TRAJECTORY_BUILDER_3D.enable_ndt_initialization = false

TRAJECTORY_BUILDER_3D.imu.acc_noise= 0.0365432018302e1
TRAJECTORY_BUILDER_3D.imu.gyr_noise= 0.00367396706572e1
TRAJECTORY_BUILDER_3D.imu.acc_bias_noise= 0.000433
TRAJECTORY_BUILDER_3D.imu.gyr_bias_noise= 2.66e-05
TRAJECTORY_BUILDER_3D.imu.gravity= 9.80511

TRAJECTORY_BUILDER_3D.imu.ceres_pose_noise_t = 0.05
TRAJECTORY_BUILDER_3D.imu.ceres_pose_noise_r = 0.05
TRAJECTORY_BUILDER_3D.imu.ceres_pose_noise_t_drift = 0.01
TRAJECTORY_BUILDER_3D.imu.ceres_pose_noise_r_drift = 0.01

TRAJECTORY_BUILDER_3D.imu.prior_pose_noise = 0.05
TRAJECTORY_BUILDER_3D.imu.prior_vel_noise = 0.05
TRAJECTORY_BUILDER_3D.imu.prior_bias_noise = 1e-03

TRAJECTORY_BUILDER_3D.lvio_config_filename = "/home/wz/study_hub/ros_env/ct-lvio-release/src/ctlvi/config/lvio/viral.yaml"
POSE_GRAPH.camera_intrinsics_filename = "/home/wz/study_hub/ros_env/ct-lvio-release/src/ctlvi/config/lvio/viral.yaml"
POSE_GRAPH.dbow_voc_filename = "/home/wz/study_hub/ros_env/ct-lvio-release/orbvoc.dbow3"
POSE_GRAPH.enable_so3_visual_loop_search = false
return options