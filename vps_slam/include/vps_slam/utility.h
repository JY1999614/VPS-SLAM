#ifndef UTILITY_H_
#define UTILITY_H_

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "vps_slam/MultiSurfels.h"
#include "vps_slam/SurfelWithCovariance.h"

#include <Eigen/Eigen>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <ceres/ceres.h>

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <algorithm>
#include <unordered_set>
#include <fstream>

using namespace std;

Eigen::Vector3d gravity(0, 0, -9.81);

typedef pcl::PointXYZI PointType;

struct StatesGroup
{
    double time;
    Eigen::Vector3d pos_end;
    Eigen::Quaterniond quat_end;
    Eigen::Vector3d vel_end;
    Eigen::Vector3d bias_a;
    Eigen::Vector3d bias_g;
    Eigen::Matrix<double, 15, 15> cov; // P R V Ba Bg

    StatesGroup()
    {
        this->time = 0;
        this->pos_end = Eigen::Vector3d::Zero();
        this->quat_end = Eigen::Quaterniond::Identity();
        this->vel_end = Eigen::Vector3d::Zero();
        this->bias_a = Eigen::Vector3d::Zero();
        this->bias_g = Eigen::Vector3d::Zero();
        this->cov = Eigen::Matrix<double, 15, 15>::Identity() * 0.000001;
    }

    StatesGroup(const double &time)
    {
        this->time = time;
        this->pos_end = Eigen::Vector3d::Zero();
        this->quat_end = Eigen::Quaterniond::Identity();
        this->vel_end = Eigen::Vector3d::Zero();
        this->bias_a = Eigen::Vector3d::Zero();
        this->bias_g = Eigen::Vector3d::Zero();
        this->cov = Eigen::Matrix<double, 15, 15>::Identity() * 0.000001;
    }

    StatesGroup(const StatesGroup &b)
    {
        this->time = b.time;
        this->pos_end = b.pos_end;
        this->quat_end = b.quat_end;
        this->vel_end = b.vel_end;
        this->bias_a = b.bias_a;
        this->bias_g = b.bias_g;
        this->cov = b.cov;
    }

    StatesGroup &operator=(const StatesGroup &b)
    {
        this->time = b.time;
        this->pos_end = b.pos_end;
        this->quat_end = b.quat_end;
        this->vel_end = b.vel_end;
        this->bias_a = b.bias_a;
        this->bias_g = b.bias_g;
        this->cov = b.cov;
        return *this;
    }
};

class Utility
{
    public:
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }
    
    // Hat (skew) operator
    template <typename T>
    static Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1> &vec)
    {
        Eigen::Matrix<T, 3, 3> mat;
        mat << 0, -vec(2), vec(1),
               vec(2), 0, -vec(0),
               -vec(1), vec(0), 0;
        return mat;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
               q(2), typename Derived::Scalar(0), -q(0),
               -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
        ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(q.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
        ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(p.vec());
        return ans;
    }

    template<typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> LeftQuatMatrix(const Eigen::QuaternionBase<Derived> &q) {
        Eigen::Matrix<typename Derived::Scalar, 4, 4> m;
        Eigen::Matrix<typename Derived::Scalar, 3, 1> vq = q.vec();
        typename Derived::Scalar q4 = q.w();
        m.block(0, 0, 3, 3) << q4 * Eigen::Matrix3d::Identity() + skewSymmetric(vq);
        m.block(3, 0, 1, 3) << -vq.transpose();
        m.block(0, 3, 3, 1) << vq;
        m(3, 3) = q4;
        return m;
    }

    // Convert from quaternion to rotation vector
    template <typename T>
    static Eigen::Matrix<T, 3, 1> quaternionToRotationVector(const Eigen::Quaternion<T> &qua)
    {
        Eigen::Matrix<T, 3, 1> rotation_vec;
        Eigen::Matrix<T, 3, 3> mat = qua.toRotationMatrix();
        Eigen::AngleAxis<T> angle_axis;
        angle_axis.fromRotationMatrix(mat);
        rotation_vec = angle_axis.angle() * angle_axis.axis();
        return rotation_vec;
    }

    // Left Jacobian matrix
    template <typename T>
    static Eigen::Matrix<T, 3, 3> Jleft(const Eigen::Quaternion<T> &qua)
    {
        Eigen::Matrix<T, 3, 3> mat;
        Eigen::Matrix<T, 3, 1> rotation_vec = quaternionToRotationVector(qua);
        double theta_norm = rotation_vec.norm();
        mat = Eigen::Matrix<T, 3, 3>::Identity() +
              (1 - cos(theta_norm)) / (theta_norm * theta_norm + 1e-10) * hat(rotation_vec) +
              (theta_norm - sin(theta_norm)) / (theta_norm * theta_norm * theta_norm + 1e-10) * hat(rotation_vec) * hat(rotation_vec);
        return mat;
    }

    template <typename T>
    static Eigen::Matrix<T, 3, 3> Jleft_inv(const Eigen::Quaternion<T> &qua)
    {
        Eigen::Matrix<T, 3, 3> mat;
        Eigen::Matrix<T, 3, 1> rotation_vec = quaternionToRotationVector(qua);
        double theta_norm = rotation_vec.norm();
        mat = Eigen::Matrix<T, 3, 3>::Identity() - 0.5 * hat(rotation_vec) +
              (1.0 / (theta_norm * theta_norm + 1e-10) + (1 + cos(theta_norm)) / (2 * theta_norm * sin(theta_norm) + 1e-10)) * hat(rotation_vec) * hat(rotation_vec);
        return mat;
    }
};

class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

#endif // UTILITY_H_