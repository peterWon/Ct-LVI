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
#include <Eigen/Core>
#include <Eigen/Dense>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "tf_bridge.h"

DEFINE_string(calib_file_dir, "", "Dir of the calibration file.");

namespace cartographer_ros {
namespace {
using namespace std;
void Split(const string& s, vector<string>& tokens, const char& delim = ' ') {
  tokens.clear();
  size_t lastPos = s.find_first_not_of(delim, 0);
  size_t pos = s.find(delim, lastPos);
  while (lastPos != string::npos) {
      tokens.emplace_back(s.substr(lastPos, pos - lastPos));
      lastPos = s.find_first_not_of(delim, pos);
      pos = s.find(delim, lastPos);
  }
}

void ReadTransform(const std::string& calib_file, Eigen::Matrix4d& T){
  std::ifstream ifs(calib_file);
  if(!ifs.is_open()){
    LOG(ERROR)<<"Open calib file failed!";
    return;
  }
  std::string line;
  std::vector<std::string> substrs = {};
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  while(getline(ifs, line)){
    if(line.empty()) continue;
    if(line.at(0) == 'R'){
      Split(line, substrs);
      if(substrs.size() != 10){
        LOG(ERROR)<<"R has invalid size, check it!";
        return;
      }
      for(int i = 1; i < 10; i++){
        istringstream os(substrs[i]);
        double d;
        os >> d;
        R.row((i-1)/3)[(i-1)%3] = d;
      }
    }else if(line.at(0) == 'T'){
      Split(line, substrs);
      if(substrs.size() != 4){
        LOG(ERROR)<<"T has invalid size, check it!";
        return;
      }
      for(int i = 1; i < 4; i++){
        istringstream os(substrs[i]);
        double d;
        os >> d;
        t[i-1] = d;
      }
    }
  }
  ifs.close();
  
  T.block(0,0,3,3) = R;
  T.block(0,3,3,1) = t;
  T.block(3,0,1,4) << 0,0,0,1;
}

void Run(const std::string& calib_file_dir) {  
  std::string calib_imu_to_velo = calib_file_dir+"/calib_imu_to_velo.txt";
  // std::string calib_velo_to_cam = calib_file_dir+"/calib_velo_to_cam.txt";
  Eigen::Matrix4d T_imu_to_velo;
  ReadTransform(calib_imu_to_velo, T_imu_to_velo);
  Eigen::Matrix4d T_velo_to_imu = T_imu_to_velo.inverse();
  Eigen::Vector3d rpy = T_velo_to_imu.block<3,3>(0,0).eulerAngles(0,1,2);
  Eigen::Vector3d t = T_velo_to_imu.block<3,1>(0,3);

  LOG(INFO) << "Translation from velodyne to imu is: "; 
  LOG(INFO) << t[0] << "," << t[1] << "," <<t[2]; 
  LOG(INFO) << "Rotation(roll, pitch, yaw) from velodyne to imu is: "; 
  LOG(INFO) << rpy[0] << "," << rpy[1] << "," << rpy[2]; 
}

}  // namespace
}  // namespace cartographer_ros

Eigen::Quaterniond yprToQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Eigen::Quaterniond q;
    q.w() = cy * cp * cr + sy * sp * sr;
    q.x() = cy * cp * sr - sy * sp * cr;
    q.y() = sy * cp * sr + cy * sp * cr;
    q.z() = sy * cp * cr - cy * sp * sr;

    return q;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  using namespace Eigen;
  
  /*newer college*/
  // Eigen::Quaterniond q(0,0,0,-1);
  // Eigen::Matrix3d R(q);
  // Eigen::Vector3d rpy = R.eulerAngles(0,1,2);
  // LOG(INFO) << "Rotation(roll, pitch, yaw) from velodyne to imu is: "; 
  // LOG(INFO) << rpy[0] << "," << rpy[1] << "," << rpy[2]; 
  
  // /*KAIST*/
  // Eigen::Matrix4d T_i2b, T_v2b;
  // T_i2b.setIdentity();
  // T_v2b.setIdentity();
  // // Eigen::Vector3d eulerAngle(yaw,pitch,roll);
  // Eigen::Vector3d eulerAngle(0,0,0);
  // Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
  // Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1),Vector3d::UnitY()));
  // Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(0), Vector3d::UnitZ())); 
  // R = yawAngle * pitchAngle * rollAngle;
  // Eigen::Vector3d t;
  // t << -0.07, 0, 1.7;
  // T_v2b.block(0,0,3,3)=R;
  // T_v2b.block(0,3,3,1)=t;

  // Eigen::Matrix4d T_b2v = T_v2b.inverse();
  // Eigen::Vector3d rpy_b2v = T_b2v.block<3,3>(0,0).eulerAngles(0,1,2);
  // Eigen::Vector3d t_b2v = T_b2v.block<3,1>(0,3);
  // LOG(INFO)<<t_b2v[0]<<","<<t_b2v[1]<<","<<t_b2v[2];
  // LOG(INFO)<<rpy_b2v[0]<<","<<rpy_b2v[1]<<","<<rpy_b2v[2];
  
  //LIO-SAM campus, lidar to imu
  // Eigen::Matrix3d R;
  // R << -1, 0, 0,  0, 1, 0, 0, 0, -1;
  // Eigen::Vector3d rpy_campus = R.eulerAngles(0,1,2);
  // LOG(INFO)<<rpy_campus[0]<<","<<rpy_campus[1]<<","<<rpy_campus[2];

  // VIRAL
  /* Eigen::Matrix3d R;
  Eigen::Matrix4d T, T_inv;
  Eigen::Vector3d t;
  T << 0.02183084, -0.01312053,  0.99967558,  0.00552943,
           0.99975965,  0.00230088, -0.02180248, -0.12431302,
          -0.00201407,  0.99991127,  0.01316761,  0.01614686,
        0, 0, 0, 1;
  T_inv = T.inverse();
  Eigen::Vector3d rpy_l2i = T_inv.block<3, 3>(0,0).eulerAngles(0,1,2);
  Eigen::Vector3d t_l2i = T_inv.block<3,1>(0,3);
  LOG(INFO)<<T_inv;
  LOG(INFO)<<t_l2i[0]<<","<<t_l2i[1]<<","<<t_l2i[2];
  LOG(INFO)<<rpy_l2i[0]<<","<<rpy_l2i[1]<<","<<rpy_l2i[2];
  
  Eigen::Quaterniond q = yprToQuaternion(rpy_l2i[2],rpy_l2i[1],rpy_l2i[0]);
  LOG(INFO)<<q.toRotationMatrix(); */
  // CHECK(!FLAGS_calib_file_dir.empty()) << "-calib_file_dir is missing.";

  // ::cartographer_ros::Run(FLAGS_calib_file_dir);
  

  // Eigen::Matrix4d T_cam2imu;
  // Eigen::Matrix3d R_cam2imu;
  // T_cam2imu << 0.00014894, -0.00731846,  0.99997321,  0.07328566,
  //          -0.9999806,  -0.00622816,  0.00010336, -0.09614561,
  //         0.00622723, -0.99995382, -0.00731925, -0.01113022,
  //          0.00000000,  0.00000000,  0.00000000,  1.00000000;
  // R_cam2imu = T_cam2imu.block(0,0,3,3);

  
  // Eigen::Vector3d euler = R_cam2imu.eulerAngles(2,1,0);
  // //此处的顺序等价于
  // // R_cam2imu = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
  // //             * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
  // //             * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  
  // LOG(INFO)<<euler[2]<<","<<euler[1]<<","<<euler[0];

  Eigen::Vector3d rpy, pos;
  pos<<-0.0353649, -0.205563, 0.078487;
  rpy<<3.1318, -3.0805, -3.0383;
  Eigen::Quaterniond q = yprToQuaternion(rpy[2],rpy[1],rpy[0]);
  Eigen::Matrix3d R = q.toRotationMatrix();

  LOG(INFO)<<R.transpose();
  LOG(INFO)<<R.transpose() * (-pos);
}
