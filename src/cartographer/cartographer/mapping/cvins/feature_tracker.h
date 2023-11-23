#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"

namespace cvins{
  using namespace std;
  using namespace camodocal;
  using namespace Eigen;

  bool inBorder(const camodocal::CameraPtr camera, const cv::Point2f &pt);

  void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
  void reduceVector(vector<int> &v, vector<uchar> status);

  class FeatureTracker{
  public:
    FeatureTracker(const std::string& camera_intrinsic_path);
    void setFisheyeMask(const cv::Mat& _fisheye_mask);
    void setMaxCount(int count){
      MAX_CNT_ = count;
    }
    
    // update
    void trackImage(const cv::Mat &_img, double _cur_time, bool enabled);
    
    // access
    const vector<cv::Point2f>& current_un_pts() const{
      return cur_un_pts;
    }
    const vector<cv::Point2f>& current_pts() const{
      return cur_pts;
    }
    const vector<cv::Point2f>& current_pts_velocity() const{
      return pts_velocity;
    }
    const vector<int>& current_ids() const{
      return ids;
    }
    
    void get_equalized_img(cv::Mat& img){
      cur_img.copyTo(img);
    };

    camodocal::CameraPtr get_camera(){
      return m_camera;
    }
    // debug
    void drawTrackedPoints(const string &filename);
    void showUndistortion(const string &name);

    
  private:

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    //TODO(wz): check this, it would be dangerous when n_id grows too big!
    static int n_id;

    int ROW_;
    int COL_;
    bool FISHEYE_ = false;
    bool EQUALIZE_ = true;
    double MIN_DIST_ = 20.;
    double F_THRESHOLD_ = 1.0;
    int MAX_CNT_ = 150; 
    float FOCAL_LENGTH_ = 460.;
  };
}// namespace cvins