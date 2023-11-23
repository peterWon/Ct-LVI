//
// Created by hannes on 2018-02-16.
//

#ifndef KONTIKIV2_VIEW_IMPL_H
#define KONTIKIV2_VIEW_IMPL_H

#include "view.h"
#include "landmark.h"
#include "observation.h"

namespace sfm {

View::View(size_t frame, double t0)
    : frame_nr_(frame), t0_(t0){};

View::~View() {
  // Make sure a View that is destroyed propagate observations removed
  std::vector<std::shared_ptr<Observation>> obs_copy(observations_);
  for (auto obs : obs_copy) {
    RemoveObservation(obs);
  }
}

size_t View::frame_nr() const {
  return frame_nr_;
}

void View::set_frame_nr(size_t fnr) {
  frame_nr_ = fnr;
}

double View::t0() const {
  return t0_;
}

void View::set_t0(double t0) {
  t0_ = t0;
}


std::vector<std::shared_ptr<Observation>> View::observations() const {
  return observations_;
}

std::shared_ptr<Observation> View::CreateObservation(std::shared_ptr<Landmark> landmark, const Eigen::Vector2d &uv) {
  // Create and store the new observation
  auto obs = std::make_shared<Observation>(uv, landmark, shared_from_this());
  observations_.push_back(obs);

  // Let the Landmark know about its new observation
  landmark->AddObservation(obs);

  return obs;
}

void View::RemoveObservation(std::shared_ptr<Observation> obs) {
  // Find and remove the landmark from the view's list
  auto it = std::find(observations_.begin(), observations_.end(), obs);
  if (it!=observations_.end()) {
    observations_.erase(it);
  } else {
    // throw std::runtime_error("Observation does not beloing to this view");
  }

  // Tell the landmark to remove it from its list
  obs->landmark()->RemoveObservation(obs);
}

} // namespace sfm

#endif //KONTIKIV2_VIEW_IMPL_H
