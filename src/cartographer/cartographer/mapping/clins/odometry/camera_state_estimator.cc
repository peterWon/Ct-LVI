
#include "odometry/camera_state_estimator.h"
#include <glog/logging.h>


namespace cv {
  void decomposeEssentialMat(const Mat& E, Mat& _R1, Mat& _R2, Mat& _t ){
    CV_Assert(E.cols == 3 && E.rows == 3);

    Mat D, U, Vt;
    SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    Mat R1, R2, t;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;

    R1.copyTo(_R1);
    R2.copyTo(_R2);
    t.copyTo(_t);
  }

  int recoverPose(const Mat& E_, const vector<Point2d>& points1_,
                  const vector<Point2d>& points2_, 
                  const Mat& cameraMatrix_,
                  Mat& _R, Mat& _t){
    int npoints = points1_.size();

    double fx = cameraMatrix_.at<double>(0,0);
    double fy = cameraMatrix_.at<double>(1,1);
    double cx = cameraMatrix_.at<double>(0,2);
    double cy = cameraMatrix_.at<double>(1,2);
    
    vector<Point2d> points1 = points1_;
    vector<Point2d> points2 = points2_;
    for(int i = 0; i < npoints; ++i){
      points1[i].x = (points1[i].x - cx) / fx;
      points2[i].x = (points2[i].x - cx) / fx;
      points1[i].y = (points1[i].y - cy) / fy;
      points2[i].y = (points2[i].y - cy) / fy;
    }
    
    Mat E, R1, R2, t;
    E_.copyTo(E);
    E.convertTo(E, CV_64F);
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    Mat Q;
    triangulatePoints(P0, P1, points1, points2, Q);
    Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;
    triangulatePoints(P0, P2, points1, points2, Q);

    Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;
    
    triangulatePoints(P0, P3, points1, points2, Q);
    Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;
    
    triangulatePoints(P0, P4, points1, points2, Q);
    Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;
    
    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();
    // If _mask is given, then use it to filter outliers.
    // if (!_mask.empty()){
    //   Mat mask = _mask.getMat();
    //   CV_Assert(mask.size() == mask1.size());
    //   bitwise_and(mask, mask1, mask1);
    //   bitwise_and(mask, mask2, mask2);
    //   bitwise_and(mask, mask3, mask3);
    //   bitwise_and(mask, mask4, mask4);
    // }
    // if (_mask.empty() && _mask.needed()){
    //   _mask.create(mask1.size(), CV_8U);
    // }

    // CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());
    
    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);
    
    if (good1 >= good2 && good1 >= good3 && good1 >= good4){
      R1.copyTo(_R);
      t.copyTo(_t);
      // if (_mask.needed()) mask1.copyTo(_mask);
      return good1;
    }else if (good2 >= good1 && good2 >= good3 && good2 >= good4){
      R2.copyTo(_R);
      t.copyTo(_t);
      // if (_mask.needed()) mask2.copyTo(_mask);
      return good2;
    }else if (good3 >= good1 && good3 >= good2 && good3 >= good4){
      t = -t;
      R1.copyTo(_R);
      t.copyTo(_t);
      // if (_mask.needed()) mask3.copyTo(_mask);
      return good3;
    }else{
      t = -t;
      R2.copyTo(_R);
      t.copyTo(_t);
      // if (_mask.needed()) mask4.copyTo(_mask);
      return good4;
    }
  }
}

namespace clins {
bool CameraMotionEstimator::SolveRelativeRT(
    const vector<pair<Vector2d, Vector2d>> &corres, 
    Matrix3d &Rotation, Vector3d &Translation){
  if (corres.size() >= 15){
    vector<cv::Point2d> ll, rr;
    for (int i = 0; i < int(corres.size()); i++){
      ll.push_back(cv::Point2d(corres[i].first(0), corres[i].first(1)));
      rr.push_back(cv::Point2d(corres[i].second(0), corres[i].second(1)));
    }
    cv::Mat mask;
    cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 1.0, 0.99, mask);
    if(E.cols!=3 || E.rows != 3) return false;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) 
        << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans);
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++){   
      T(i) = trans.at<double>(i, 0);
      for (int j = 0; j < 3; j++)
        R(i, j) = rot.at<double>(i, j);
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;
    if(inlier_cnt > 12)
      return true;
    else
      return false;
  }
  return false;
}


}  // namespace clins
