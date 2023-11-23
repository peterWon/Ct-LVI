/*
 * Copyright 2017 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "cartographer/io/proto_stream.h"
#include "cartographer/io/proto_stream_deserializer.h"
#include "cartographer/io/submap_painter.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/3d/submap_3d.h"
#include "cartographer/mapping/proto/pose_graph.pb.h"
#include "cartographer/mapping/proto/serialization.pb.h"
#include "cartographer/mapping/proto/submap.pb.h"
#include "cartographer/mapping/proto/trajectory_builder_options.pb.h"
#include "cartographer_ros/ros_map.h"
#include "cartographer_ros/submap.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "cartographer_ros/time_conversion.h"

DEFINE_string(pbstream_filename, "",
              "Filename of a pbstream to save a kitti trahectory from.");
DEFINE_string(traj_filestem, "traj", "Stem of the output file.");

namespace cartographer_ros {
namespace {

void Run(const std::string& pbstream_filename, 
         const std::string& traj_filestem) {
  ::cartographer::io::ProtoStreamReader reader(pbstream_filename);
  ::cartographer::io::ProtoStreamDeserializer deserializer(&reader);
  
  Eigen::Matrix4d H, H_init, T, Pose_imu;
  H = Eigen::Matrix4d::Identity();
  T = Eigen::Matrix4d::Identity();
  H_init = Eigen::Matrix4d::Identity();
  Pose_imu = Eigen::Matrix4d::Identity();
  
  // Eigen::AngleAxisd rotation_vector(-3.5 * M_PI / 180.0 , Eigen::Vector3d(0,0,1));
  // Eigen::Matrix3d R_init = rotation_vector.matrix();
  // H_init.block(0,0,3,3) = R_init;

  bool init_flag = true;
  // T = T_velo_to_cam * T_imu_to_velo;//
  //真值是以相机坐标系为基准的，x:水平向右; y:竖直向下；ｚ:水平向前
  // T << 0,0,1,0,
  //      -1,0,0,0,
  //      0,-1,0,0,
  //      0,0,0,1;
  /********************************************/
  std::ofstream ofs(traj_filestem+".viral");
  if(!ofs.is_open()) {
    LOG(ERROR)<<"Open viral result file failed!";  
    return;
  }
  ofs.setf(std::ios::scientific, std::ios::floatfield);
  ofs.precision(6);
  LOG(INFO) << "Loading trajectory nodes from serialized data.";

  uint32_t seq = 0;
  ofs << "\%time,field.header.seq,field.header.stamp,field.pose.position.x,field.pose.position.y,field.pose.position.z,field.pose.orientation.x,field.pose.orientation.y,field.pose.orientation.z,field.pose.orientation.w\n";

  ::cartographer::mapping::proto::SerializedData proto;
  const auto& pose_graph = deserializer.pose_graph();
  for(const auto&traj: pose_graph.trajectory()){
    for(const auto& node: traj.node()){
      const ::cartographer::transform::Rigid3d global_pose =
            ::cartographer::transform::ToRigid3(node.pose());
      
      cartographer::common::Time time = cartographer::common::FromUniversal(
        node.timestamp());
      // auto t64 = cartographer::common::ToUniversal(time);
      ros::Time stamp = cartographer_ros::ToRos(time); 
      
      int64_t uts_timestamp = ::cartographer::common::ToUniversal(time);
      int64_t ns_since_unix_epoch =
          (uts_timestamp -
          ::cartographer::common::kUtsEpochOffsetFromUnixEpochInSeconds *
              10000000ll) *
          100ll;     
      // LOG(INFO)<<stamp.toNSec() - ns_since_unix_epoch;
      //TODO(wz): make sure seq is not usded in alignment.
      ofs << ns_since_unix_epoch <<"," << seq++ 
          << "," << ns_since_unix_epoch << ","
          << global_pose.translation().x() << ","
          << global_pose.translation().y() << ","
          << global_pose.translation().z() << ","
          << global_pose.rotation().x() << ","
          << global_pose.rotation().y() << ","
          << global_pose.rotation().z() << ","
          << global_pose.rotation().w() << "\n";
    }
  }

  // CHECK(reader.eof());
  ofs.close();
  LOG(INFO) << "Exported trajectory poses to viral format."; 
}

}  // namespace
}  // namespace cartographer_ros

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_pbstream_filename.empty()) << "-pbstream_filename is missing.";
  CHECK(!FLAGS_traj_filestem.empty()) << "-traj_filestem is missing.";

  ::cartographer_ros::Run(FLAGS_pbstream_filename, FLAGS_traj_filestem);
}
