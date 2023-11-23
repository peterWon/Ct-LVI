#include "cartographer/mapping/cvins/feature_tracker.h"
#include "glog/logging.h"

namespace cvins{


int FeatureTracker::n_id = 0;

bool inBorder(const camodocal::CameraPtr camera, const cv::Point2f &pt){
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x 
    && img_x < camera->imageWidth() - BORDER_SIZE 
    && BORDER_SIZE <= img_y && img_y < camera->imageHeight() - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status){
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status){
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


FeatureTracker::FeatureTracker(const std::string& camera_intrinsic_path){
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(
      camera_intrinsic_path);
  assert(m_camera);
  ROW_= m_camera->imageHeight();
  COL_ = m_camera->imageWidth();
  FISHEYE_ = false;//TODO
}

void FeatureTracker::setFisheyeMask(const cv::Mat& _fisheye_mask){
  if(!_fisheye_mask.empty()){
    fisheye_mask = _fisheye_mask.clone();
    FISHEYE_ = true;
  }else{
    FISHEYE_ = false;
  }
}

void FeatureTracker::setMask(){
  if(FISHEYE_)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(ROW_, COL_, CV_8UC1, cv::Scalar(255));
 
  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for(unsigned int i = 0; i < forw_pts.size(); i++){
    cnt_pts_id.push_back(
      make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
  }

  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](
      const pair<int, pair<cv::Point2f, int>> &a, 
      const pair<int, pair<cv::Point2f, int>> &b){
    return a.first > b.first;
  });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for(auto &it : cnt_pts_id){
    if (mask.at<uchar>(it.second.first) == 255){
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST_, 0, -1);
    }
  }
}

void FeatureTracker::addPoints(){
  for (auto &p : n_pts){
    forw_pts.push_back(p);
    ids.push_back(n_id++);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::trackImage(
    const cv::Mat &_img, double _cur_time, bool enabled){
  cv::Mat img;
  cur_time = _cur_time;

  if (EQUALIZE_){
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(_img, img);
  }else{
    img = _img;
  }
      

  if (forw_img.empty()){
    prev_img = cur_img = forw_img = img;
  }else{
    forw_img = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0){
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(
      cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(m_camera, forw_pts[i]))
        status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt)
    n++;

  if (enabled){
    rejectWithF();
    setMask();

    int n_max_cnt = MAX_CNT_ - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0){//跟踪到的特征点不够MAX_CNT时，生成新的点
      if(mask.empty())
        LOG(WARNING) << "mask is empty.";
      if (mask.type() != CV_8UC1)
        LOG(WARNING) << "mask type wrong.";
      if (mask.size() != forw_img.size())
        LOG(WARNING) << "wrong size."<<mask.size()<<" vs. "<<forw_img.size();
      cv::goodFeaturesToTrack(
        forw_img, n_pts, MAX_CNT_ - forw_pts.size(), 0.01, MIN_DIST_, mask);
    }else{
      n_pts.clear();
    }
      
    addPoints();
  }
  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;

  // move update ID here. in case only one camera. 
  // bool completed = false;
  // for (unsigned int i = 0;; i++){
  //   completed |= updateID(i);
  //   if (!completed)
  //     break;
  // }
}

void FeatureTracker::rejectWithF(){
  if (forw_pts.size() >= 8){
    vector<cv::Point2f> un_cur_pts(
      cur_pts.size()), un_forw_pts(forw_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++){
      Eigen::Vector3d tmp_p;
      m_camera->liftProjective(
        Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      // tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      // tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      tmp_p.x() = tmp_p.x() / tmp_p.z();
      tmp_p.y() = tmp_p.y() / tmp_p.z();
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(
        Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
      tmp_p.x() = tmp_p.x() / tmp_p.z();
      tmp_p.y() = tmp_p.y() / tmp_p.z();
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(
      un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD_, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
  }
}

bool FeatureTracker::updateID(unsigned int i){
  if (i < ids.size()){
    if (ids[i] == -1)
      ids[i] = n_id++;
    return true;
  }else {
    return false;
  }  
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file){
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name){
  cv::Mat undistortedImg(ROW_ + 600, COL_ + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < COL_; i++){
    for (int j = 0; j < ROW_; j++){
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
    }
  }
  for (int i = 0; i < int(undistortedp.size()); i++){
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH_ + COL_ / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH_ + ROW_ / 2;
    pp.at<float>(2, 0) = 1.0;
    
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW_ + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL_ + 600){
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
    else{
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::drawTrackedPoints(const string &filename){
  cv::Mat show;
  show = cur_img.clone();
  cvtColor(show, show, CV_GRAY2BGR);
  for(const auto& pt: cur_un_pts){
    Eigen::Vector3d pt3d;
    Eigen::Vector2d uv;
    pt3d << pt.x, pt.y, 1.;
    m_camera->spaceToPlane(pt3d, uv);
    cv::circle(show, cv::Point2f(uv[0], uv[1]), 5, cv::Scalar(0,255,0));
  }
  // cv::imwrite(filename, show);
  cv::imshow("track", show);
  cv::waitKey(5);
}

void FeatureTracker::undistortedPoints(){
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  for (unsigned int i = 0; i < cur_pts.size(); i++){
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(
      make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty()){
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++){
      if (ids[i] != -1){
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end()){
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        }
        else{
          pts_velocity.push_back(cv::Point2f(0, 0));
        }
      }else{
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  }else{
    for (unsigned int i = 0; i < cur_pts.size(); i++){
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}

  
} // namespace cvins