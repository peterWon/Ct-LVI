#ifndef __CAMERA_STATE_ESTIMATOR_H__
#define __CAMERA_STATE_ESTIMATOR_H__
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "utils/eigen_utils.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

namespace clins {
class CameraMotionEstimator{
  
public:
  CameraMotionEstimator(){}
  static bool SolveRelativeRT(
    const vector<pair<Vector2d, Vector2d>> &corres, Matrix3d &R, Vector3d &T);
};

}  // namespace clins
#endif //__CAMERA_STATE_ESTIMATOR_H__