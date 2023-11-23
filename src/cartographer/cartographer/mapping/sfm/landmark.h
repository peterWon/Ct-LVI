//
// Created by hannes on 2016-12-01.
//

#ifndef KONTIKI_LANDMARK_H
#define KONTIKI_LANDMARK_H

#include <memory>
#include <vector>

#include <Eigen/Core>

namespace sfm {

class Observation;
class View;

class Landmark {
  friend View;

  static size_t new_id() {
    static size_t next_id = -1;
    ++next_id;
    return next_id;
  }

 public:
  Landmark();

  size_t id() const;
  void set_id(size_t id);

  void set_reference(std::shared_ptr<Observation> new_ref);
  std::shared_ptr<Observation> reference() const;

  std::vector<std::shared_ptr<Observation>> observations() const;

  double inverse_depth() const;
  void set_inverse_depth(double x);
  double* inverse_depth_ptr();


  bool IsLocked() const;
  void Lock(bool flag);
  void RemoveObservation(std::shared_ptr<Observation> obs);
  void AddObservation(std::shared_ptr<Observation> obs);
 protected:
  size_t id_;
  double inverse_depth_;
  bool locked_;
  // std::weak_ptr<Observation> reference_observation_;
  // std::vector<std::weak_ptr<Observation>> observations_={};
  std::shared_ptr<Observation> reference_observation_;
  std::vector<std::shared_ptr<Observation>> observations_={};
};

} // namespace sfm
#endif //KONTIKI_LANDMARK_H
