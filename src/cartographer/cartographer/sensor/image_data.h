/*
 * Copyright 2016 The Cartographer Authors
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

#ifndef CARTOGRAPHER_SENSOR_IMAGE_DATA_H_
#define CARTOGRAPHER_SENSOR_IMAGE_DATA_H_

#include "cartographer/common/port.h"
#include "cartographer/common/time.h"
#include "cartographer/transform/rigid_transform.h"
#include "opencv2/opencv.hpp"
#include "cartographer/sensor/proto/sensor.pb.h"

namespace cartographer {
namespace sensor {
struct ImageData {
  common::Time time;
  cv::Mat img;
};

struct ImageFeatureData {
  common::Time time;
  cv::Mat img;//to remove

  // below are computed from img by local SLAM
  int frame_id;
  transform::Rigid3d pose_in_local;
  std::vector<int> features_id = {};
  std::vector<cv::Point2f> features_uv = {};
};

// TODO(wz)
// Converts 'image_data' to a proto::ImageData.
// proto::ImageData ToProto(const ImageData& image_data);

// // Converts 'proto' to an ImageData.
// ImageData FromProto(const proto::ImageData& proto);

}  // namespace sensor
}  // namespace cartographer

#endif  // CARTOGRAPHER_SENSOR_IMAGE_DATA_H_
