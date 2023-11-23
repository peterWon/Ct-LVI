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

#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <Eigen/Core>

namespace clins {


struct ReferenceFeature {
  double timestamp = -1;
  
  bool initialized = false;
  double depth = -1;
  
  size_t id = 0;
  Eigen::Vector2f pt;
};

struct ObservationFeature {
  double timestamp = -1;
  
  size_t ref_id = 0;
  double ref_timestamp = -1;
  
  Eigen::Vector2f pt;
};


}  // namespace clins

#endif
