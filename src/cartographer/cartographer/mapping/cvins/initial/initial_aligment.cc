#include "initial/initial_alignment.h"

namespace cvins{

//目的是利用IMU预积分所得的第k和第k+1帧之间的旋转矩阵与视觉所解的两帧之间的旋转阵残差，
//以bias作为变量来优化求解
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, 
                        Vector3d* Bgs){
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin(); 
      next(frame_i) != all_image_frame.end(); frame_i++){
    frame_j = next(frame_i);
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    //从相机求得的i,j帧之间的相对旋转
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    //旋转对于Bias的雅克比
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(
        O_R, O_BG);
    //残差，只用四元数的虚部进行计算
    tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() 
        * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  //Cholesky分解求解Bias
  delta_bg = A.ldlt().solve(b);

  for (int i = 0; i <= WINDOW_SIZE; i++)
    Bgs[i] += delta_bg;

  //利用求解得到的Bias重新预积分
  for (frame_i = all_image_frame.begin(); 
      next(frame_i) != all_image_frame.end( ); frame_i++){
    frame_j = next(frame_i);
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}

MatrixXd TangentBasis(Vector3d &g0){
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if(a == tmp)
      tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

//由于速度和尺度求解过程所解得的g没有强约束，因此通过限制其大小的尺度，
//建立局部坐标系，对重力进行精细调整
void RefineGravity(map<double, ImageFrame> &all_image_frame, 
                   Vector3d &g, VectorXd &x, const cvins::ParameterServer& ps){
  Vector3d g0 = g.normalized() * ps.G.norm();
  Vector3d lx, ly;
  //VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for(int k = 0; k < 4; k++){
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin(); 
        next(frame_i) != all_image_frame.end(); frame_i++, i++){
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() 
          * dt * dt / 2 * Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() 
          * (frame_j->second.T - frame_i->second.T) / 100.0;     
      tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p 
          + frame_i->second.R.transpose() * frame_j->second.R * ps.TIC[0] 
          - ps.TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = 
          frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() 
          * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v 
          - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * ps.G.norm();
    //double s = x(n_state - 1);
  }   
  g = g0;
}

//利用预积分所得的位移变化（非传统意义）和速度变化值和经过帧的理想位置之间的差异来构建残差项。
//建立误差方程JtJx=Jb，通过Cholesky分解来求解理想参数。
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, 
                     Vector3d &g, VectorXd &x,
                     const ParameterServer& ps){
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin(); 
      next(frame_i) != all_image_frame.end(); frame_i++, i++){
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() 
        * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() 
        * (frame_j->second.T - frame_i->second.T) / 100.0;  
    tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + 
      frame_i->second.R.transpose() * frame_j->second.R * ps.TIC[0] - ps.TIC[0];

    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = 
        frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() 
        * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();

    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  g = x.segment<3>(n_state - 4);

  if(fabs(g.norm() - ps.G.norm()) > 1.0 || s < 0){
    return false;
  } 

  RefineGravity(all_image_frame, g, x, ps);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;

  if(s < 0.0 )
    return false;   
  else
    return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, 
                        Vector3d* Bgs, Vector3d &g, VectorXd &x,
                        const ParameterServer& ps){
  solveGyroscopeBias(all_image_frame, Bgs);

  if(LinearAlignment(all_image_frame, g, x, ps))
    return true;
  else 
    return false;
}
}
