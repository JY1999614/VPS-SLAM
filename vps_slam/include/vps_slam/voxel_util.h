#ifndef VOXEL_UTIL_H_
#define VOXEL_UTIL_H_

#include "vps_slam/utility.h"

#define HASH_P 116101
#define MAX_N 10000000000

vector<Eigen::Vector3d> axes = {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)};

class VOXEL_LOC
{
  public:
    int64_t x, y, z;
    
    VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
        : x(vx), y(vy), z(vz) {}
    
    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

class VOXEL_LAYER_LOC
{
  public:
    int8_t l, x, y, z;

    VOXEL_LAYER_LOC(int8_t vl = 0, int8_t vx = 0, int8_t vy = 0, int8_t vz = 0)
        : l(vl), x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LAYER_LOC &other) const {
        return (l == other.l && x == other.x && y == other.y && z == other.z);
    }

    VOXEL_LAYER_LOC operator*(float scalar) const {
        return VOXEL_LAYER_LOC(
            l, 
            static_cast<int8_t>(std::floor(x * scalar)), 
            static_cast<int8_t>(std::floor(y * scalar)), 
            static_cast<int8_t>(std::floor(z * scalar))
        );
    }
    
    VOXEL_LAYER_LOC operator+(const VOXEL_LAYER_LOC &other) const {
        return VOXEL_LAYER_LOC(l + other.l, x + other.x, y + other.y, z + other.z);
    }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

template <> struct hash<VOXEL_LAYER_LOC> {
  int64_t operator()(const VOXEL_LAYER_LOC &s) const {
    using std::hash;
    using std::size_t;
    int64_t layer_value = (s.l == 0) ? 0 : (1LL << (3 * (s.l - 1)));
    return (layer_value + 4 * s.x + 2 * s.y + s.z);
  }
};
} // namespace std

typedef struct pointWithCov
{
    Eigen::Vector3d point;
    Eigen::Matrix3d cov;
} pointWithCov;

typedef struct Surfel
{
    Eigen::Vector3d normal;
    Eigen::Vector3d center;
    double radius = 0;
    Eigen::Matrix<double, 7, 7> surfel_cov;
    bool is_surfel = false;
} Surfel;

typedef struct SurfelInfo
{
    int points_size = 0;
    Eigen::Matrix3cd evecs;
    Eigen::Vector3d evalsReal;
    Eigen::Matrix3f::Index evalsMin;
    Eigen::Vector3d normal;
    Eigen::Matrix<double, 7, 7> surfel_cov;
} SurfelInfo;

typedef struct ptsf
{
    // point parameters
    Eigen::Vector3d point;
    Eigen::Matrix3d point_cov;
    // surfel parameters
    Eigen::Vector3d normal;
    Eigen::Vector3d center;
    double radius;
    Eigen::Matrix<double, 7, 7> surfel_cov;
} ptsf;

typedef struct sfsf
{
    // scan surfel parameters
    Eigen::Vector3d center_scan;
    Eigen::Vector3d normal_scan;
    Eigen::Matrix<double, 7, 7> surfel_scan_cov;
    // world surfel parameters
    Eigen::Vector3d center_map;
    Eigen::Vector3d normal_map;
    Eigen::Matrix<double, 7, 7> surfel_map_cov;
} sfsf;

class PointsChildVoxel
{
public:
    std::vector<pointWithCov> points_;
    shared_ptr<Surfel> surfel_ptr_;
    shared_ptr<SurfelInfo> surfel_info_ptr_;
    int layer_;
    float flat_threshold_;
    double half_length_;
    double voxel_vertex_[3];

    PointsChildVoxel(int layer, float flat_threshold)
        : layer_(layer), flat_threshold_(flat_threshold)
    {
        points_.clear();
        surfel_ptr_.reset(new Surfel);
        surfel_info_ptr_.reset(new SurfelInfo);
    }

    void init_surfel()
    {
        surfel_ptr_->center = Eigen::Vector3d::Zero();
        surfel_ptr_->surfel_cov = Eigen::Matrix<double, 7, 7>::Zero();
        surfel_info_ptr_->points_size = points_.size();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

        for (auto pv : points_)
        {
            surfel_ptr_->center += pv.point;
            covariance += pv.point * pv.point.transpose();
        }
        surfel_ptr_->center = surfel_ptr_->center / surfel_info_ptr_->points_size;
        covariance = covariance / surfel_info_ptr_->points_size - 
                     surfel_ptr_->center * surfel_ptr_->center.transpose();

        Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;

        // surfel uncertainty calculation
        Eigen::Matrix3d J_Q;
        J_Q << 1.0 / surfel_info_ptr_->points_size, 0, 0,
               0, 1.0 / surfel_info_ptr_->points_size, 0,
               0, 0, 1.0 / surfel_info_ptr_->points_size;
        double curve = 2 * (evalsReal(evalsMid) - evalsReal(evalsMin)) / (evalsReal(evalsMax) + evalsReal(evalsMid) + evalsReal(evalsMin));
        if (evalsReal(evalsMin) < flat_threshold_ && curve > 0.3)
        {
            surfel_ptr_->is_surfel = true;
            surfel_ptr_->normal << evecs.real()(0, evalsMin),
                                   evecs.real()(1, evalsMin),
                                   evecs.real()(2, evalsMin);
            int edge_point_id = -1;
            surfel_ptr_->radius = 0.0;
            for (int i = 0; i < surfel_info_ptr_->points_size; i++)
            {
                double dis = (points_[i].point - surfel_ptr_->center).norm();
                if (surfel_ptr_->radius < dis)
                {
                    edge_point_id = i;
                    surfel_ptr_->radius = dis;
                }
            }
            // adjust the direction of the normal vector
            double d = surfel_ptr_->center.transpose() * surfel_ptr_->normal;
            if (d > 0) surfel_ptr_->normal = -surfel_ptr_->normal;

            surfel_info_ptr_->evecs = evecs;
            surfel_info_ptr_->evalsReal = evalsReal;
            surfel_info_ptr_->evalsMin = evalsMin;

            for (int i = 0; i < surfel_info_ptr_->points_size; ++i)
            {
                Eigen::Matrix<double, 7, 3> J;
                Eigen::Matrix3d F;
                for (int m = 0; m < 3; ++m)
                {
                    if (m != (int)evalsMin)
                    {
                        Eigen::Matrix<double, 1, 3> F_m = 
							(points_[i].point - surfel_ptr_->center).transpose() /
							(surfel_info_ptr_->points_size * (evalsReal[evalsMin] - evalsReal[m])) *
							(evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
							 evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
						F.row(m) = F_m;
                    }
                    else
                    {
                        Eigen::Matrix<double, 1, 3> F_m;
                        F_m << 0, 0, 0;
                        F.row(m) = F_m;
                    }
                }
                J.block<3, 3>(0, 0) = evecs.real() * F;
                J.block<3, 3>(3, 0) = J_Q;
                if (i == edge_point_id)
                    J.block<1, 3>(6, 0) = (1.0 - 1.0 / surfel_info_ptr_->points_size) * Eigen::Vector3d::Identity().transpose();
                else
                    J.block<1, 3>(6, 0) = -1.0 / surfel_info_ptr_->points_size * Eigen::Vector3d::Identity().transpose();
                surfel_ptr_->surfel_cov += J * points_[i].cov * J.transpose();
            }
        }
    }

    void split_surfel(const shared_ptr<Surfel> surfel, const shared_ptr<SurfelInfo> surfel_info)
    {
        surfel_ptr_->normal = surfel->normal;
        surfel_ptr_->center = Eigen::Vector3d::Zero();
        surfel_ptr_->radius = 0.0;
        surfel_ptr_->surfel_cov = Eigen::Matrix<double, 7, 7>::Zero();
        surfel_ptr_->is_surfel = true;
        int points_size = points_.size();

        // calculate center
        for (auto pv : points_)
        {
            surfel_ptr_->center += pv.point;
        }
        surfel_ptr_->center = surfel_ptr_->center / points_size;

        // calculate radius
        int edge_point_id = -1;
        for (int i = 0; i < points_size; i++)
        {
            double dis = (points_[i].point - surfel_ptr_->center).norm();
            if (surfel_ptr_->radius < dis)
            {
                edge_point_id = i;
                surfel_ptr_->radius = dis;
            }
        }
        
        // calculate surfel_cov
        Eigen::Matrix3d J_Q;
        J_Q << 1.0 / points_size, 0, 0,
               0, 1.0 / points_size, 0,
               0, 0, 1.0 / points_size;
        for (int i = 0; i < points_size; i++)
        {
            Eigen::Matrix<double, 7, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++)
            {
                if (m != (int)surfel_info->evalsMin)
                {
                    Eigen::Matrix<double, 1, 3> F_m = 
                        (points_[i].point - surfel_ptr_->center).transpose() /
                        ((points_size) * (surfel_info->evalsReal[surfel_info->evalsMin] - surfel_info->evalsReal[m])) *
                        (surfel_info->evecs.real().col(m) * surfel_info->evecs.real().col(surfel_info->evalsMin).transpose() +
                         surfel_info->evecs.real().col(surfel_info->evalsMin) * surfel_info->evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                }
                else
                {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = surfel_info->evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;
            if (i == edge_point_id)
                J.block<1, 3>(6, 0) = (1.0 - 1.0 / points_size) * Eigen::Vector3d::Identity().transpose();
            else
                J.block<1, 3>(6, 0) = -1.0 / points_size * Eigen::Vector3d::Identity().transpose();
            surfel_ptr_->surfel_cov += J * points_[i].cov * J.transpose();
        }
        surfel_ptr_->surfel_cov.block<3, 3>(0, 0) = surfel->surfel_cov.block<3, 3>(0, 0);
    }
};

class PointsRootVoxel
{
public:
    std::vector<pointWithCov> points_;
    std::unordered_map<VOXEL_LAYER_LOC, shared_ptr<PointsChildVoxel>> voxels_;
    int max_layer_;
    int layer_point_size_;
    float flat_threshold_;
    float max_voxel_size_;
    float min_voxel_size_;
    double voxel_vertex_[3];

    PointsRootVoxel(int max_layer, int layer_point_size, float flat_threshold)
        : max_layer_(max_layer),layer_point_size_(layer_point_size), flat_threshold_(flat_threshold)
    {
        points_.clear();
    }

    ~PointsRootVoxel()
    {
        voxels_.clear();
    }

    void surfel_fitting()
    {
        if ((int)points_.size() < layer_point_size_) return;

        shared_ptr<PointsChildVoxel> root_voxel(new PointsChildVoxel(0, flat_threshold_));
        root_voxel->half_length_ = max_voxel_size_ / 2.0;
        root_voxel->voxel_vertex_[0] = voxel_vertex_[0];
        root_voxel->voxel_vertex_[1] = voxel_vertex_[1];
        root_voxel->voxel_vertex_[2] = voxel_vertex_[2];
        root_voxel->points_ = points_;

        queue<pair<VOXEL_LAYER_LOC, shared_ptr<PointsChildVoxel>>> voxels_que;
        VOXEL_LAYER_LOC position(0, 0, 0, 0);
        voxels_que.push(make_pair(position, root_voxel));

        while (!voxels_que.empty())
        {
            VOXEL_LAYER_LOC curr_pos = voxels_que.front().first;
            shared_ptr<PointsChildVoxel> curr_voxel = voxels_que.front().second;
            voxels_que.pop();

            curr_voxel->init_surfel();
            if (curr_voxel->surfel_ptr_->is_surfel)
            {
                voxels_[curr_pos] = curr_voxel;
            }
            else
            {
                if (curr_voxel->layer_ >= max_layer_) continue;

                // cut curr voxel
                unordered_map<VOXEL_LAYER_LOC, shared_ptr<PointsChildVoxel>> leaves;
                for (auto pv : curr_voxel->points_)
                {
                    int xyz[3] = {0, 0, 0};
                    if (pv.point[0] >= curr_voxel->voxel_vertex_[0] + curr_voxel->half_length_)
                        xyz[0] = 1;
                    if (pv.point[1] >= curr_voxel->voxel_vertex_[1] + curr_voxel->half_length_)
                        xyz[1] = 1;
                    if (pv.point[2] >= curr_voxel->voxel_vertex_[2] + curr_voxel->half_length_)
                        xyz[2] = 1;
                    
                    VOXEL_LAYER_LOC layer_pos(1, (int8_t)xyz[0], (int8_t)xyz[1], (int8_t)xyz[2]);
                    if (leaves.find(layer_pos) == leaves.end())
                    {
                        shared_ptr<PointsChildVoxel> child_voxel(new PointsChildVoxel(curr_voxel->layer_ + 1, flat_threshold_));
                        leaves[layer_pos] = child_voxel;
                        leaves[layer_pos]->half_length_ = curr_voxel->half_length_ / 2.0,
                        leaves[layer_pos]->voxel_vertex_[0] = curr_voxel->voxel_vertex_[0] + xyz[0] * curr_voxel->half_length_;
                        leaves[layer_pos]->voxel_vertex_[1] = curr_voxel->voxel_vertex_[1] + xyz[1] * curr_voxel->half_length_;
                        leaves[layer_pos]->voxel_vertex_[2] = curr_voxel->voxel_vertex_[2] + xyz[2] * curr_voxel->half_length_;
                    }
                    leaves[layer_pos]->points_.push_back(pv);
                }
                for (auto iter = leaves.begin(); iter != leaves.end(); ++iter)
                {
                    if ((int)iter->second->points_.size() >= layer_point_size_)
                    {
                        VOXEL_LAYER_LOC child_pos = curr_pos * 2 + iter->first;
                        voxels_que.push(make_pair(child_pos, iter->second));
                    }
                }
            }
        }
    }

    void surfel_splitting()
    {
        if (voxels_.empty()) return;

        vector<pair<VOXEL_LAYER_LOC, shared_ptr<PointsChildVoxel>>> voxels_vec;
        for (auto iter = voxels_.begin(); iter != voxels_.end(); ++iter)
        {
            if (iter->second->layer_ < max_layer_)
            {
                voxels_vec.push_back(make_pair(iter->first, iter->second));
            }
        }
        for (int i = 0; i < (int)voxels_vec.size(); i++)
        {
            VOXEL_LAYER_LOC curr_pos = voxels_vec[i].first;
            shared_ptr<PointsChildVoxel> curr_voxel = voxels_vec[i].second;
            // remove from map
            voxels_.erase(curr_pos);
            // split to minimal voxel
            unordered_map<VOXEL_LAYER_LOC, shared_ptr<PointsChildVoxel>> leaves;
            for (auto pv : curr_voxel->points_)
            {
                float loc_xyz[3];
                for (int j = 0; j < 3; j++)
                {
                    loc_xyz[j] = (pv.point[j] - voxel_vertex_[j]) / min_voxel_size_;
                }
                VOXEL_LAYER_LOC child_pos((int8_t)max_layer_, (int8_t)loc_xyz[0], (int8_t)loc_xyz[1], (int8_t)loc_xyz[2]);
                if (leaves.find(child_pos) == leaves.end())
                {
                    shared_ptr<PointsChildVoxel> child_voxel(new PointsChildVoxel(max_layer_, flat_threshold_));
                    leaves[child_pos] = child_voxel;
                    leaves[child_pos]->half_length_ = min_voxel_size_ / 2.0,
                    leaves[child_pos]->voxel_vertex_[0] = voxel_vertex_[0] + child_pos.x * min_voxel_size_;
                    leaves[child_pos]->voxel_vertex_[1] = voxel_vertex_[1] + child_pos.y * min_voxel_size_;
                    leaves[child_pos]->voxel_vertex_[2] = voxel_vertex_[2] + child_pos.z * min_voxel_size_;
                }
                leaves[child_pos]->points_.push_back(pv);
            }
            for (auto iter = leaves.begin(); iter != leaves.end(); ++iter)
            {
                if ((int)iter->second->points_.size() >= 3)
                {
                    iter->second->split_surfel(curr_voxel->surfel_ptr_, curr_voxel->surfel_info_ptr_);
                    voxels_[iter->first] = iter->second;
                }
            }
        }
    }
};

class SurfelChildVoxel
{
public:
    std::vector<shared_ptr<Surfel>> surfels_;
    shared_ptr<Surfel> surfel_map_;
    shared_ptr<Surfel> surfel_cur_;
    int layer_;
    float half_length_;
    double voxel_vertex_[3];
    bool surfel_occupy_;

    SurfelChildVoxel(int layer) : layer_(layer)
    {
        surfels_.clear();
        surfel_map_.reset(new Surfel);
        surfel_cur_.reset(new Surfel);
        surfel_occupy_ = false;
    }

    ~SurfelChildVoxel()
    {
        surfels_.clear();
    }

    bool checkTwoSurfelsConsistency(const shared_ptr<Surfel> surfel_1, const shared_ptr<Surfel> surfel_2)
    {
        double cos_theta = surfel_1->normal.transpose() * surfel_2->normal;
        if (cos_theta < 0.96) return false;
        double distance = surfel_1->normal.transpose() * (surfel_2->center - surfel_1->center);
        if (abs(distance) > 0.05) return false;
        return true;
    }

    void fuse_surfel()
    {
        // calculate transform matrix
        Eigen::Vector3d rot_v = surfel_map_->normal.cross(Eigen::Vector3d(1, 0, 0));
        rot_v.normalize();
        double cos_rot = surfel_map_->normal.transpose() * Eigen::Vector3d(1, 0, 0);
        double sin_rot = sqrt(1 - cos_rot * cos_rot);
        Eigen::Matrix3d rot_v_hat = Utility::hat(rot_v);
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin_rot * rot_v_hat + (1 - cos_rot) * rot_v_hat * rot_v_hat;
        Eigen::Matrix3d R_inverse = R.inverse();
        Eigen::Matrix<double, 7, 7> J_new_old = Eigen::Matrix<double, 7, 7>::Identity();
        J_new_old.block<3, 3>(0, 0) = R;
        J_new_old.block<3, 3>(3, 3) = R;
        Eigen::Matrix<double, 7, 7> J_old_new = Eigen::Matrix<double, 7, 7>::Identity();
        J_old_new.block<3, 3>(0, 0) = R_inverse;
        J_old_new.block<3, 3>(3, 3) = R_inverse;
        // transform to new coordinate system
        Surfel surfel_1_trans, surfel_2_trans;
        surfel_1_trans.normal = R * surfel_map_->normal;
        surfel_1_trans.center = R * surfel_map_->center;
        surfel_1_trans.radius = surfel_map_->radius;
        surfel_1_trans.surfel_cov = J_new_old * surfel_map_->surfel_cov * J_new_old.transpose();
        surfel_2_trans.normal = R * surfel_cur_->normal;
        surfel_2_trans.center = R * surfel_cur_->center;
        surfel_2_trans.radius = surfel_cur_->radius;
        surfel_2_trans.surfel_cov = J_new_old * surfel_cur_->surfel_cov * J_new_old.transpose();

        // transform to polar coordinate system
        Eigen::Matrix<double, 6, 1> surfel_1_p = Eigen::Matrix<double, 6, 1>::Zero();
        surfel_1_p(0) = acos(surfel_1_trans.normal.z());
        surfel_1_p(1) = atan2(surfel_1_trans.normal.y(), surfel_1_trans.normal.x());
        surfel_1_p.block<3, 1>(2, 0) = surfel_1_trans.center;
        surfel_1_p(5) = surfel_1_trans.radius;
        Eigen::Matrix<double, 6, 7> J_p_w_1 = Eigen::Matrix<double, 6, 7>::Zero();
        J_p_w_1(0, 2) = -1 * pow((1 - surfel_1_trans.normal.z() * surfel_1_trans.normal.z()), -0.5);
        J_p_w_1(1, 0) = -1 * surfel_1_trans.normal.y() / (surfel_1_trans.normal.x() * surfel_1_trans.normal.x() + surfel_1_trans.normal.y() * surfel_1_trans.normal.y());
        J_p_w_1(1, 1) = surfel_1_trans.normal.x() / (surfel_1_trans.normal.x() * surfel_1_trans.normal.x() + surfel_1_trans.normal.y() * surfel_1_trans.normal.y());
        J_p_w_1.block<4, 4>(2, 3) = Eigen::Matrix4d::Identity();
        Eigen::Matrix<double, 6, 6> sf_cov_1_p = J_p_w_1 * surfel_1_trans.surfel_cov * J_p_w_1.transpose();
        Eigen::Matrix<double, 6, 1> surfel_2_p = Eigen::Matrix<double, 6, 1>::Zero();
        surfel_2_p(0) = acos(surfel_2_trans.normal.z());
        surfel_2_p(1) = atan2(surfel_2_trans.normal.y(), surfel_2_trans.normal.x());
        surfel_2_p.block<3, 1>(2, 0) = surfel_2_trans.center;
        surfel_2_p(5) = surfel_2_trans.radius;
        Eigen::Matrix<double, 6, 7> J_p_w_2 = Eigen::Matrix<double, 6, 7>::Zero();
        J_p_w_2(0, 2) = -1 * pow((1 - surfel_2_trans.normal.z() * surfel_2_trans.normal.z()), -0.5);
        J_p_w_2(1, 0) = -1 * surfel_2_trans.normal.y() / (surfel_2_trans.normal.x() * surfel_2_trans.normal.x() + surfel_2_trans.normal.y() * surfel_2_trans.normal.y());
        J_p_w_2(1, 1) = surfel_2_trans.normal.x() / (surfel_2_trans.normal.x() * surfel_2_trans.normal.x() + surfel_2_trans.normal.y() * surfel_2_trans.normal.y());
        J_p_w_2.block<4, 4>(2, 3) = Eigen::Matrix4d::Identity();
        Eigen::Matrix<double, 6, 6> sf_cov_2_p = J_p_w_2 * surfel_2_trans.surfel_cov * J_p_w_2.transpose();
        
        // fuse surfels
        Eigen::Matrix<double, 6, 1> surfel_fuse_p = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 6> sf_cov_fuse_p = Eigen::Matrix<double, 6, 6>::Zero();
        // calculate weight
        double a1_cov_1_inv = 1.0 / sf_cov_1_p(0, 0);
        double a1_cov_2_inv = 1.0 / sf_cov_2_p(0, 0);
        double a1_cov_inv_sum = a1_cov_1_inv + a1_cov_2_inv;
        double a2_cov_1_inv = 1.0 / sf_cov_1_p(1, 1);
        double a2_cov_2_inv = 1.0 / sf_cov_2_p(1, 1);
        double a2_cov_inv_sum = a2_cov_1_inv + a2_cov_2_inv;
        double c_cov_1_inv = 1.0 / (sf_cov_1_p.block<3, 3>(2, 2).diagonal().sum());
        double c_cov_2_inv = 1.0 / (sf_cov_2_p.block<3, 3>(2, 2).diagonal().sum());
        double c_cov_inv_sum = c_cov_1_inv + c_cov_2_inv;
        double r_cov_1_inv = 1.0 / sf_cov_1_p(5, 5);
        double r_cov_2_inv = 1.0 / sf_cov_2_p(5, 5);
        double r_cov_inv_sum = r_cov_1_inv + r_cov_2_inv;
        // fuse normal
        surfel_fuse_p(0, 0) = (1.0 / a1_cov_inv_sum) * (a1_cov_1_inv * surfel_1_p(0, 0) + a1_cov_2_inv * surfel_2_p(0, 0));
        surfel_fuse_p(1, 0) = (1.0 / a2_cov_inv_sum) * (a2_cov_1_inv * surfel_1_p(1, 0) + a2_cov_2_inv * surfel_2_p(1, 0));
        // fuse center
        surfel_fuse_p.block<3, 1>(2, 0) = (1.0 / c_cov_inv_sum) * 
            (c_cov_1_inv * surfel_1_p.block<3, 1>(2, 0) + c_cov_2_inv * surfel_2_p.block<3, 1>(2, 0));
        // fuse radius
        surfel_fuse_p(5, 0) = (1.0 / r_cov_inv_sum) * 
            (r_cov_1_inv * ((surfel_1_p.block<3, 1>(2, 0) - surfel_fuse_p.block<3, 1>(2, 0)).norm() + surfel_1_p(5, 0)) +
             r_cov_2_inv * ((surfel_2_p.block<3, 1>(2, 0) - surfel_fuse_p.block<3, 1>(2, 0)).norm() + surfel_2_p(5, 0)));
        // calculate covariance
        Eigen::Matrix<double, 1, 3> A_1 = (r_cov_1_inv / r_cov_inv_sum) * 
            (surfel_1_p.block<3, 1>(2, 0) - surfel_fuse_p.block<3, 1>(2, 0)).transpose().normalized();
        Eigen::Matrix<double, 1, 3> A_2 = (r_cov_2_inv / r_cov_inv_sum) *
            (surfel_2_p.block<3, 1>(2, 0) - surfel_fuse_p.block<3, 1>(2, 0)).transpose().normalized();
        Eigen::Matrix<double, 1, 3> A_sum = A_1 + A_2;
        Eigen::Matrix<double, 6, 6> J_1 = Eigen::Matrix<double, 6, 6>::Zero();
        J_1(0, 0) = a1_cov_1_inv / a1_cov_inv_sum;
        J_1(1, 1) = a2_cov_1_inv / a2_cov_inv_sum;
        J_1.block<3, 3>(2, 2) = c_cov_1_inv / c_cov_inv_sum * Eigen::Matrix3d::Identity();
        J_1.block<1, 3>(5, 2) = A_1 - A_sum * J_1.block<3, 3>(2, 2);
        J_1(5, 5) = r_cov_1_inv / r_cov_inv_sum;
        Eigen::Matrix<double, 6, 6> J_2 = Eigen::Matrix<double, 6, 6>::Zero();
        J_2(0, 0) = a1_cov_2_inv / a1_cov_inv_sum;
        J_2(1, 1) = a2_cov_2_inv / a2_cov_inv_sum;
        J_2.block<3, 3>(2, 2) = c_cov_2_inv / c_cov_inv_sum * Eigen::Matrix3d::Identity();
        J_2.block<1, 3>(5, 2) = A_2 - A_sum * J_2.block<3, 3>(2, 2);
        J_2(5, 5) = r_cov_2_inv / r_cov_inv_sum;
        sf_cov_fuse_p = J_1 * sf_cov_1_p * J_1.transpose() + J_2 * sf_cov_2_p * J_2.transpose();

        // transform back to world coordinate system
        Surfel surfel_fuse_trans;
        surfel_fuse_trans.normal.x() = sin(surfel_fuse_p(0)) * cos(surfel_fuse_p(1));
        surfel_fuse_trans.normal.y() = sin(surfel_fuse_p(0)) * sin(surfel_fuse_p(1));
        surfel_fuse_trans.normal.z() = cos(surfel_fuse_p(0));
        surfel_fuse_trans.center = surfel_fuse_p.block<3, 1>(2, 0);
        surfel_fuse_trans.radius = surfel_fuse_p(5);
        Eigen::Matrix<double, 7, 6> J_w_p = Eigen::Matrix<double, 7, 6>::Zero();
        J_w_p(0, 0) = cos(surfel_fuse_p(0)) * cos(surfel_fuse_p(1));
		J_w_p(0, 1) = -sin(surfel_fuse_p(0)) * sin(surfel_fuse_p(1));
		J_w_p(1, 0) = cos(surfel_fuse_p(0)) * sin(surfel_fuse_p(1));
		J_w_p(1, 1) = sin(surfel_fuse_p(0)) * cos(surfel_fuse_p(1));
		J_w_p(2, 0) = -sin(surfel_fuse_p(0));
		J_w_p.block<4, 4>(3, 2) = Eigen::Matrix4d::Identity();
        surfel_fuse_trans.surfel_cov = J_w_p * sf_cov_fuse_p * J_w_p.transpose();
        // map surfel
        surfel_map_->normal = R_inverse * surfel_fuse_trans.normal;
        surfel_map_->center = R_inverse * surfel_fuse_trans.center;
        surfel_map_->radius = surfel_fuse_trans.radius;
        surfel_map_->surfel_cov = J_old_new * surfel_fuse_trans.surfel_cov * J_old_new.transpose();
        surfel_map_->is_surfel = true;
        // adjust the direction of the normal vector
        if (surfel_map_->normal.transpose() * surfel_cur_->normal < 0) surfel_map_->normal = -surfel_map_->normal;
    }
    
    void split_surfel()
    {
        // determine the distance from the surfel center to the voxel center
        Eigen::Vector3d voxel_center(voxel_vertex_[0] + half_length_, voxel_vertex_[1] + half_length_, voxel_vertex_[2] + half_length_);
        double distance = (surfel_map_->center - voxel_center).norm();
        if (distance < half_length_ / 2)
        {
            // split surfel regions
            vector<Eigen::Vector3d> split_lines;
            for (int i = 0; i < 3; i++)
            {
                // exclude parallel axis
                double cos_axis = surfel_map_->normal.transpose() * axes[i];
                if (cos_axis > 0.966) continue;
                // calculate split line
                Eigen::Vector3d line = surfel_map_->normal.cross(axes[i]);
                line.normalize();
                // adjust direction
                if (!split_lines.empty())
                {
                    double cos_lines = line.transpose() * split_lines.back();
                    if (cos_lines < 0) line = -line;
                }
                split_lines.push_back(line);
            }
            split_lines.push_back(-split_lines[0]);

            for (int i = 0; i < (int)split_lines.size() - 1; i++)
            {
                shared_ptr<Surfel> split_surfel_1(new Surfel);
                shared_ptr<Surfel> split_surfel_2(new Surfel);
                split_surfel_1->is_surfel = true;
                split_surfel_2->is_surfel = true;
                split_surfel_1->normal = surfel_map_->normal;
                split_surfel_2->normal = surfel_map_->normal;
                // calculate center and radius
                Eigen::Vector3d direction = 0.5 * (split_lines[i] + split_lines[i + 1]);
                direction.normalize();
                double half_theta = 0.5 * acos(split_lines[i].transpose() * split_lines[i + 1]);
                split_surfel_1->radius = sin(half_theta) / (sin(half_theta) + 1) * surfel_map_->radius;
                split_surfel_2->radius = split_surfel_1->radius;
                double step = surfel_map_->radius - split_surfel_1->radius;
                split_surfel_1->center = surfel_map_->center + step * direction;
                split_surfel_2->center = surfel_map_->center - step * direction;
                // the uncertainty of the split surfel remains unchanged
                split_surfel_1->surfel_cov = surfel_map_->surfel_cov;
                split_surfel_2->surfel_cov = surfel_map_->surfel_cov;
                // add surfels
                surfels_.push_back(split_surfel_1);
                surfels_.push_back(split_surfel_2);
            }
        }
        else
        {
            // sample points
            vector<vector<Eigen::Vector3d>> points_cluster(8);
            Eigen::Vector3d rot_v = surfel_map_->normal.cross(Eigen::Vector3d(0, 0, 1));
            rot_v.normalize();
            double cos_rot = surfel_map_->normal.transpose() * Eigen::Vector3d(0, 0, 1);
            double sin_rot = sqrt(1 - cos_rot * cos_rot);
            Eigen::Matrix3d rot_v_hat = Utility::hat(rot_v);
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin_rot * rot_v_hat + (1 - cos_rot) * rot_v_hat * rot_v_hat;
            Eigen::Matrix3d R_inverse = R.inverse();
            double step = half_length_ / 10;
            for (double x = -surfel_map_->radius; x <= surfel_map_->radius; x += step)
            {
                for (double y = -surfel_map_->radius; y <= surfel_map_->radius; y += step)
                {
                    Eigen::Vector3d point_s(x, y, 0);
                    double norm = point_s.norm();
                    if (norm > surfel_map_->radius) continue;
                    Eigen::Vector3d point_w = R_inverse * point_s + surfel_map_->center;
                    int xyz[3] = {0, 0, 0};
                    if (point_w.x() >= voxel_center.x())
                        xyz[0] = 1;
                    if (point_w.y() >= voxel_center.y())
                        xyz[1] = 1;
                    if (point_w.z() >= voxel_center.z())
                        xyz[2] = 1;
                    int id = xyz[0] * 4 + xyz[1] * 2 + xyz[2];

                    points_cluster[id].push_back(point_w);
                }
            }
            // calculate split line
            vector<pair<Eigen::Vector3d, Eigen::Vector3d>> line_pos_normal;
            for (int i = 0; i < 3; i++)
            {
                // exclude parallel axis
                double cos_axis = surfel_map_->normal.transpose() * axes[i];
                if (cos_axis > 0.966) continue;
                // calculate direction
                Eigen::Vector3d direction = surfel_map_->normal.cross(axes[i]);
                direction.normalize();
                // calculate point
                Eigen::Vector3d point;
                if (axes[i].x() == 1)
                {
                    if (surfel_map_->normal.y() > surfel_map_->normal.z())
                    {
                        double y = surfel_map_->center.y() + surfel_map_->normal.x() / surfel_map_->normal.y() * (surfel_map_->center.x() - voxel_center.x());
                        point = Eigen::Vector3d(voxel_center.x(), y, surfel_map_->center.z());
                    }
                    else
                    {
                        double z = surfel_map_->center.z() + surfel_map_->normal.x() / surfel_map_->normal.z() * (surfel_map_->center.x() - voxel_center.x());
                        point = Eigen::Vector3d(voxel_center.x(), surfel_map_->center.y(), z);
                    }
                }
                else if (axes[i].y() == 1)
                {
                    if (surfel_map_->normal.x() > surfel_map_->normal.z())
                    {
                        double x = surfel_map_->center.x() + surfel_map_->normal.y() / surfel_map_->normal.x() * (surfel_map_->center.y() - voxel_center.y());
                        point = Eigen::Vector3d(x, voxel_center.y(), surfel_map_->center.z());
                    }
                    else
                    {
                        double z = surfel_map_->center.z() + surfel_map_->normal.y() / surfel_map_->normal.z() * (surfel_map_->center.y() - voxel_center.y());
                        point = Eigen::Vector3d(surfel_map_->center.x(), voxel_center.y(), z);
                    }
                }
                else
                {
                    if (surfel_map_->normal.x() > surfel_map_->normal.y())
                    {
                        double x = surfel_map_->center.x() + surfel_map_->normal.z() / surfel_map_->normal.x() * (surfel_map_->center.z() - voxel_center.z());
                        point = Eigen::Vector3d(x, surfel_map_->center.y(), voxel_center.z());
                    }
                    else
                    {
                        double y = surfel_map_->center.y() + surfel_map_->normal.z() / surfel_map_->normal.y() * (surfel_map_->center.z() - voxel_center.z());
                        point = Eigen::Vector3d(surfel_map_->center.x(), y, voxel_center.z());
                    }
                }
                line_pos_normal.push_back(make_pair(point, direction));
            }
            // calculate surfel
            for (int i = 0; i < (int)points_cluster.size(); i++)
            {
                int cluster_size = points_cluster[i].size();
                if (cluster_size < 5) continue;

                shared_ptr<Surfel> surfel_child(new Surfel);
                surfel_child->normal = surfel_map_->normal;
                surfel_child->is_surfel = true;
                surfel_child->surfel_cov = surfel_map_->surfel_cov;
                surfel_child->center = Eigen::Vector3d::Zero();
                for (int j = 0; j < cluster_size; j++)
                {
                    surfel_child->center += points_cluster[i][j] / cluster_size;
                }
                surfel_child->radius = surfel_map_->radius - (surfel_child->center - surfel_map_->center).norm();
                for (int j = 0; j < (int)line_pos_normal.size(); j++)
                {
                    double dis = (surfel_child->center - line_pos_normal[j].first).norm();
                    double pro = line_pos_normal[j].second.transpose() * (surfel_child->center - line_pos_normal[j].first);
                    double this_radius = sqrt(dis * dis - pro * pro);
                    if (this_radius < surfel_child->radius)
                    {
                        surfel_child->radius = this_radius;
                    }
                }
                if (surfel_child->radius <= half_length_ / 4) continue;
                surfels_.push_back(surfel_child);
            }
        }
    }

    void merge_surfel()
    {
        int surfel_num = surfels_.size();
        if (surfel_num == 1)
        {
            *surfel_cur_ = *surfels_[0];
            return;
        }
        // calculate transform matrix
        Eigen::Vector3d rot_v = surfels_[0]->normal.cross(Eigen::Vector3d(1, 0, 0));
        rot_v.normalize();
        double cos_rot = surfels_[0]->normal.transpose() * Eigen::Vector3d(1, 0, 0);
        double sin_rot = sqrt(1 - cos_rot * cos_rot);
        Eigen::Matrix3d rot_v_hat = Utility::hat(rot_v);
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin_rot * rot_v_hat + (1 - cos_rot) * rot_v_hat * rot_v_hat;
        Eigen::Matrix3d R_inverse = R.inverse();
        Eigen::Matrix<double, 7, 7> J_new_old = Eigen::Matrix<double, 7, 7>::Identity();
        J_new_old.block<3, 3>(0, 0) = R;
        J_new_old.block<3, 3>(3, 3) = R;
        Eigen::Matrix<double, 7, 7> J_old_new = Eigen::Matrix<double, 7, 7>::Identity();
        J_old_new.block<3, 3>(0, 0) = R_inverse;
        J_old_new.block<3, 3>(3, 3) = R_inverse;
        // prepare
        vector<Eigen::Matrix<double, 6, 1>> surfel_p_vec;
        vector<Eigen::Matrix<double, 6, 6>> sf_cov_p_vec;
        for (auto sf : surfels_)
        {
            // transform to new coordinate system
            Surfel surfel_trans;
            surfel_trans.normal = R * sf->normal;
            surfel_trans.center = R * sf->center;
            surfel_trans.radius = sf->radius;
            surfel_trans.surfel_cov = J_new_old * sf->surfel_cov * J_new_old.transpose();
            // transform to polar coordinate system
            Eigen::Matrix<double, 6, 1> surfel_p = Eigen::Matrix<double, 6, 1>::Zero();
            surfel_p(0) = acos(surfel_trans.normal.z());
            surfel_p(1) = atan2(surfel_trans.normal.y(), surfel_trans.normal.x());
            surfel_p.block<3, 1>(2, 0) = surfel_trans.center;
            surfel_p(5) = surfel_trans.radius;
            Eigen::Matrix<double, 6, 7> J_p_w = Eigen::Matrix<double, 6, 7>::Zero();
            J_p_w(0, 2) = -1 * pow((1 - surfel_trans.normal.z() * surfel_trans.normal.z()), -0.5);
            J_p_w(1, 0) = -1 * surfel_trans.normal.y() / (surfel_trans.normal.x() * surfel_trans.normal.x() + surfel_trans.normal.y() * surfel_trans.normal.y());
            J_p_w(1, 1) = surfel_trans.normal.x() / (surfel_trans.normal.x() * surfel_trans.normal.x() + surfel_trans.normal.y() * surfel_trans.normal.y());
            J_p_w.block<4, 4>(2, 3) = Eigen::Matrix4d::Identity();
            Eigen::Matrix<double, 6, 6> sf_cov_p = J_p_w * surfel_trans.surfel_cov * J_p_w.transpose();
            // save parameter
            surfel_p_vec.push_back(surfel_p);
            sf_cov_p_vec.push_back(sf_cov_p);
        }

        // merge surfels
        Eigen::Matrix<double, 6, 1> surfel_merge_p = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 6> sf_cov_merge_p = Eigen::Matrix<double, 6, 6>::Zero();
        // calculate weight
        vector<double> a1_cov_inv_vec, a2_cov_inv_vec, c_cov_inv_vec, r_cov_inv_vec;
        double a1_cov_inv_sum = 0, a2_cov_inv_sum = 0, c_cov_inv_sum = 0, r_cov_inv_sum = 0;
        for (int i = 0; i < surfel_num; i++)
        {
            double a1_cov_inv = 1.0 / sf_cov_p_vec[i](0, 0);
            double a2_cov_inv = 1.0 / sf_cov_p_vec[i](1, 1);
            double c_cov_inv = 1.0 / (sf_cov_p_vec[i].block<3, 3>(2, 2).diagonal().sum());
            double r_cov_inv = 1.0 / sf_cov_p_vec[i](5, 5);
            a1_cov_inv_vec.push_back(a1_cov_inv);
            a2_cov_inv_vec.push_back(a2_cov_inv);
            c_cov_inv_vec.push_back(c_cov_inv);
            r_cov_inv_vec.push_back(r_cov_inv);
            a1_cov_inv_sum += a1_cov_inv;
            a2_cov_inv_sum += a2_cov_inv;
            c_cov_inv_sum += c_cov_inv;
            r_cov_inv_sum += r_cov_inv;
        }
        // calculate normal and center
        for (int i = 0; i < surfel_num; i++)
        {
            surfel_merge_p(0, 0) += a1_cov_inv_vec[i] / a1_cov_inv_sum * surfel_p_vec[i](0, 0);
            surfel_merge_p(1, 0) += a2_cov_inv_vec[i] / a2_cov_inv_sum * surfel_p_vec[i](1, 0);
            surfel_merge_p.block<3, 1>(2, 0) += c_cov_inv_vec[i] / c_cov_inv_sum * surfel_p_vec[i].block<3, 1>(2, 0);
        }
        // calculate radius
        vector<Eigen::Matrix<double, 1, 3>> A_vec;
        Eigen::Matrix<double, 1, 3> A_sum = Eigen::Matrix<double, 1, 3>::Zero();
        for (int i = 0; i < surfel_num; i++)
        {
            surfel_merge_p(5, 0) += (r_cov_inv_vec[i] / r_cov_inv_sum) * ((surfel_p_vec[i].block<3, 1>(2, 0) - surfel_merge_p.block<3, 1>(2, 0)).norm() + surfel_p_vec[i](5, 0));
            Eigen::Matrix<double, 1, 3> A = (r_cov_inv_vec[i] / r_cov_inv_sum) * (surfel_p_vec[i].block<3, 1>(2, 0) - surfel_merge_p.block<3, 1>(2, 0)).transpose().normalized();
            A_vec.push_back(A);
            A_sum += A;
        }
        // calculate covariance
        for (int i = 0; i < surfel_num; i++)
        {
            Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
            J(0, 0) = a1_cov_inv_vec[i] / a1_cov_inv_sum;
            J(1, 1) = a2_cov_inv_vec[i] / a2_cov_inv_sum;
            J.block<3, 3>(2, 2) = c_cov_inv_vec[i] / c_cov_inv_sum * Eigen::Matrix3d::Identity();
            J.block<1, 3>(5, 2) = A_vec[i] - A_sum * J.block<3, 3>(2, 2);
            J(5, 5) = r_cov_inv_vec[i] / r_cov_inv_sum;
            sf_cov_merge_p += J * sf_cov_p_vec[i] * J.transpose();
        }

        // transform to world coordinate system
        Surfel surfel_merge_trans;
        surfel_merge_trans.normal.x() = sin(surfel_merge_p(0)) * cos(surfel_merge_p(1));
        surfel_merge_trans.normal.y() = sin(surfel_merge_p(0)) * sin(surfel_merge_p(1));
        surfel_merge_trans.normal.z() = cos(surfel_merge_p(0));
        surfel_merge_trans.center = surfel_merge_p.block<3, 1>(2, 0);
        surfel_merge_trans.radius = surfel_merge_p(5);
        Eigen::Matrix<double, 7, 6> J_w_p = Eigen::Matrix<double, 7, 6>::Zero();
        J_w_p(0, 0) = cos(surfel_merge_p(0)) * cos(surfel_merge_p(1));
		J_w_p(0, 1) = -sin(surfel_merge_p(0)) * sin(surfel_merge_p(1));
		J_w_p(1, 0) = cos(surfel_merge_p(0)) * sin(surfel_merge_p(1));
		J_w_p(1, 1) = sin(surfel_merge_p(0)) * cos(surfel_merge_p(1));
		J_w_p(2, 0) = -sin(surfel_merge_p(0));
		J_w_p.block<4, 4>(3, 2) = Eigen::Matrix4d::Identity();
        surfel_merge_trans.surfel_cov = J_w_p * sf_cov_merge_p * J_w_p.transpose();
        // curr surfel
        surfel_cur_->normal = R_inverse * surfel_merge_trans.normal;
        surfel_cur_->center = R_inverse * surfel_merge_trans.center;
        surfel_cur_->radius = surfel_merge_trans.radius;
        surfel_cur_->surfel_cov = J_old_new * surfel_merge_trans.surfel_cov * J_old_new.transpose();
        surfel_cur_->is_surfel = true;
        // adjust the direction of the normal vector
        if (surfel_cur_->normal.transpose() * surfels_[0]->normal < 0) surfel_cur_->normal = -surfel_cur_->normal;
    }
};

class SurfelRootVoxel
{
public:
    std::vector<shared_ptr<Surfel>> surfels_;
    std::unordered_map<VOXEL_LAYER_LOC, shared_ptr<SurfelChildVoxel>> voxels_;
    std::unordered_set<VOXEL_LAYER_LOC> occupy_ids_;
    int max_layer_;
    vector<double> layer_voxel_size_;
    double voxel_vertex_[3];
    int update_frame_id_;

    SurfelRootVoxel(int max_layer, vector<double> layer_voxel_size) : max_layer_(max_layer), layer_voxel_size_(layer_voxel_size)
    {
        surfels_.clear();
        update_frame_id_ = -1;
    }

    ~SurfelRootVoxel()
    {
        surfels_.clear();
        voxels_.clear();
        occupy_ids_.clear();
    }

    void voxel_updating()
    {
        unordered_set<VOXEL_LAYER_LOC> active_voxel_id;
        for (auto sf : surfels_)
        {
            for (int layer = 0; layer <= max_layer_; layer++)
            {
                float loc_xyz[3];
                for (int i = 0; i < 3; i++)
                {
                    loc_xyz[i] = (sf->center[i] - voxel_vertex_[i]) / layer_voxel_size_[layer];
                }
                VOXEL_LAYER_LOC position((int8_t)layer, (int8_t)loc_xyz[0], (int8_t)loc_xyz[1], (int8_t)loc_xyz[2]);
                // record search path
                if (occupy_ids_.find(position) == occupy_ids_.end())
                {
                    occupy_ids_.insert(position);
                }
                // record voxel node for surfels fusing
                if (layer == max_layer_)
                {
                    shared_ptr<SurfelChildVoxel> voxel_node(new SurfelChildVoxel(layer));
                    voxels_[position] = voxel_node;
                    voxels_[position]->surfel_occupy_ = false;
                    voxels_[position]->half_length_ = layer_voxel_size_[layer] / 2.0;
                    voxels_[position]->voxel_vertex_[0] = voxel_vertex_[0] + position.x * layer_voxel_size_[layer];
                    voxels_[position]->voxel_vertex_[1] = voxel_vertex_[1] + position.y * layer_voxel_size_[layer];
                    voxels_[position]->voxel_vertex_[2] = voxel_vertex_[2] + position.z * layer_voxel_size_[layer];
                }
                if (voxels_.find(position) != voxels_.end())
                {
                    active_voxel_id.insert(position);
                    voxels_[position]->surfels_.push_back(sf);
                    break;
                }
            }
        }
        surfels_.clear();

        // filter surfels
        queue<VOXEL_LAYER_LOC> voxels_id_que;
        for (auto iter = active_voxel_id.begin(); iter != active_voxel_id.end(); iter++)
        {
            shared_ptr<SurfelChildVoxel> curr_voxel = voxels_[*iter];
            // try to filter curr surfels
            double min_trace = INT_MAX;
            deque<pair<double, shared_ptr<Surfel>>> filter_surfels;
            for (auto sf : curr_voxel->surfels_)
            {
                double trace = sf->surfel_cov.block<6, 6>(0, 0).diagonal().sum();
                if (trace < min_trace)
                {
                    min_trace = trace;
                    filter_surfels.push_front(make_pair(trace, sf));
                }
                else
                {
                    filter_surfels.push_back(make_pair(trace, sf));
                }
            }
            curr_voxel->surfels_.clear();
            for (auto p : filter_surfels)
            {
                if (p.first / min_trace > 5) continue;
                curr_voxel->surfels_.push_back(p.second);
            }
            // add id to queue
            voxels_id_que.push(*iter);
        }

        // fuse surfels
        unordered_set<VOXEL_LAYER_LOC> parent_voxel_id_set;
        while (!voxels_id_que.empty())
        {
            VOXEL_LAYER_LOC curr_voxel_id = voxels_id_que.front();
            voxels_id_que.pop();
            shared_ptr<SurfelChildVoxel> curr_voxel = voxels_[curr_voxel_id];

            // try to merge curr surfels
            bool able_merged = true;
            for (int i = 1; i < (int)curr_voxel->surfels_.size(); i++)
            {
                if (curr_voxel->checkTwoSurfelsConsistency(curr_voxel->surfels_[0], curr_voxel->surfels_[i]) == false)
                {
                    able_merged = false;
                    break;
                }
            }
            if (able_merged)
            {
                // merge
                if (curr_voxel->layer_ < max_layer_)
                {
                    for (int l = max_layer_; l > curr_voxel->layer_; l--)
                    {
                        unordered_map<VOXEL_LAYER_LOC, shared_ptr<SurfelChildVoxel>> merge_voxels;
                        for (auto sf : curr_voxel->surfels_)
                        {
                            float loc_xyz[3];
                            for (int i = 0; i < 3; i++)
                            {
                                loc_xyz[i] = (sf->center[i] - voxel_vertex_[i]) / layer_voxel_size_[l];
                            }
                            VOXEL_LAYER_LOC position((int8_t)l, (int8_t)loc_xyz[0], (int8_t)loc_xyz[1], (int8_t)loc_xyz[2]);
                            if (merge_voxels.find(position) == merge_voxels.end())
                            {
                                shared_ptr<SurfelChildVoxel> voxel_node(new SurfelChildVoxel(l));
                                merge_voxels[position] = voxel_node;
                            }
                            merge_voxels[position]->surfels_.push_back(sf);
                        }
                        curr_voxel->surfels_.clear();
                        for (auto iter = merge_voxels.begin(); iter != merge_voxels.end(); ++iter)
                        {
                            iter->second->merge_surfel();
                            curr_voxel->surfels_.push_back(iter->second->surfel_cur_);
                        }
                    }
                }
                curr_voxel->merge_surfel();
            }

            // try to fuse
            if (able_merged)
            {
                if (curr_voxel->surfel_map_->is_surfel)
                {
                    if (curr_voxel->checkTwoSurfelsConsistency(curr_voxel->surfel_map_, curr_voxel->surfel_cur_))
                    {
                        curr_voxel->fuse_surfel();
                    }
                    else
                    {
                        double trace_map = curr_voxel->surfel_map_->surfel_cov.block<6, 6>(0, 0).diagonal().sum();
                        double trace_cur = curr_voxel->surfel_cur_->surfel_cov.block<6, 6>(0, 0).diagonal().sum();
                        if (trace_cur < trace_map)
                        {
                            *(curr_voxel->surfel_map_) = *(curr_voxel->surfel_cur_);
                        }
                    }
                }
                else
                {
                    *(curr_voxel->surfel_map_) = *(curr_voxel->surfel_cur_);
                    curr_voxel->surfel_occupy_ = true;
                }
                // recored parent id
                if (curr_voxel_id.l > 0)
                {
                    VOXEL_LAYER_LOC parent_voxel_id = curr_voxel_id * 0.5;
                    parent_voxel_id.l = parent_voxel_id.l - 1;
                    parent_voxel_id_set.insert(parent_voxel_id);
                }
            }
            else
            {
                // split
                if (curr_voxel_id.l < max_layer_)
                {
                    unordered_map<VOXEL_LAYER_LOC, shared_ptr<SurfelChildVoxel>> leaves;
                    // process current surfels
                    for (auto sf : curr_voxel->surfels_)
                    {
                        int xyz[3] = {0, 0, 0};
                        if (sf->center[0] >= curr_voxel->voxel_vertex_[0] + curr_voxel->half_length_)
                            xyz[0] = 1;
                        if (sf->center[1] >= curr_voxel->voxel_vertex_[1] + curr_voxel->half_length_)
                            xyz[1] = 1;
                        if (sf->center[2] >= curr_voxel->voxel_vertex_[2] + curr_voxel->half_length_)
                            xyz[2] = 1;
                        VOXEL_LAYER_LOC layer_pos(1, (int8_t)xyz[0], (int8_t)xyz[1], (int8_t)xyz[2]);
                        if (leaves.find(layer_pos) == leaves.end())
                        {
                            shared_ptr<SurfelChildVoxel> child_voxel(new SurfelChildVoxel(curr_voxel->layer_ + 1));
                            leaves[layer_pos] = child_voxel;
                            leaves[layer_pos]->half_length_ = curr_voxel->half_length_ / 2.0;
                            leaves[layer_pos]->voxel_vertex_[0] = curr_voxel->voxel_vertex_[0] + xyz[0] * curr_voxel->half_length_;
                            leaves[layer_pos]->voxel_vertex_[1] = curr_voxel->voxel_vertex_[1] + xyz[1] * curr_voxel->half_length_;
                            leaves[layer_pos]->voxel_vertex_[2] = curr_voxel->voxel_vertex_[2] + xyz[2] * curr_voxel->half_length_;
                        }
                        leaves[layer_pos]->surfels_.push_back(sf);
                    }
                    curr_voxel->surfels_.clear();
                    // split map surfel
                    curr_voxel->split_surfel();
                    curr_voxel->surfel_map_->is_surfel = false;
                    curr_voxel->surfel_occupy_ = false;
                    for (auto sf : curr_voxel->surfels_)
                    {
                        // check boundary
                        if (sf->center[0] <= curr_voxel->voxel_vertex_[0] || sf->center[0] > curr_voxel->voxel_vertex_[0] + 2 * curr_voxel->half_length_)
                            continue;
                        if (sf->center[1] <= curr_voxel->voxel_vertex_[1] || sf->center[1] > curr_voxel->voxel_vertex_[1] + 2 * curr_voxel->half_length_)
                            continue;
                        if (sf->center[2] <= curr_voxel->voxel_vertex_[2] || sf->center[2] > curr_voxel->voxel_vertex_[2] + 2 * curr_voxel->half_length_)
                            continue;
                        // fill voxel
                        int xyz[3] = {0, 0, 0};
                        if (sf->center[0] >= curr_voxel->voxel_vertex_[0] + curr_voxel->half_length_)
                            xyz[0] = 1;
                        if (sf->center[1] >= curr_voxel->voxel_vertex_[1] + curr_voxel->half_length_)
                            xyz[1] = 1;
                        if (sf->center[2] >= curr_voxel->voxel_vertex_[2] + curr_voxel->half_length_)
                            xyz[2] = 1;
                        VOXEL_LAYER_LOC layer_pos(1, (int8_t)xyz[0], (int8_t)xyz[1], (int8_t)xyz[2]);
                        if (leaves.find(layer_pos) == leaves.end())
                        {
                            shared_ptr<SurfelChildVoxel> child_voxel(new SurfelChildVoxel(curr_voxel->layer_ + 1));
                            leaves[layer_pos] = child_voxel;
                            leaves[layer_pos]->half_length_ = curr_voxel->half_length_ / 2.0;
                            leaves[layer_pos]->voxel_vertex_[0] = curr_voxel->voxel_vertex_[0] + xyz[0] * curr_voxel->half_length_;
                            leaves[layer_pos]->voxel_vertex_[1] = curr_voxel->voxel_vertex_[1] + xyz[1] * curr_voxel->half_length_;
                            leaves[layer_pos]->voxel_vertex_[2] = curr_voxel->voxel_vertex_[2] + xyz[2] * curr_voxel->half_length_;
                        }
                        *(leaves[layer_pos]->surfel_map_) = *sf;
                        leaves[layer_pos]->surfel_occupy_ = true;
                    }
                    voxels_.erase(curr_voxel_id);
                    // process the child voxels
                    for (auto iter = leaves.begin(); iter != leaves.end(); ++iter)
                    {
                        VOXEL_LAYER_LOC child_voxel_id = curr_voxel_id * 2 + iter->first;
                        voxels_[child_voxel_id] = iter->second;
                        occupy_ids_.insert(child_voxel_id);
                        if (!voxels_[child_voxel_id]->surfels_.empty())
                        {
                            voxels_id_que.push(child_voxel_id);
                        }
                    }
                }
                // current voxel is a min leaf voxel, can not further split
                else
                {
                    curr_voxel->surfel_map_->is_surfel = false;
                }
            }
            curr_voxel->surfels_.clear();
        }

        // merge voxels
        queue<VOXEL_LAYER_LOC> parent_voxel_id_que;
        for (auto iter = parent_voxel_id_set.begin(); iter != parent_voxel_id_set.end(); ++iter)
        {
            parent_voxel_id_que.push(*iter);
        }
        while (!parent_voxel_id_que.empty())
        {
            VOXEL_LAYER_LOC curr_pos = parent_voxel_id_que.front();
            parent_voxel_id_que.pop();
            parent_voxel_id_set.erase(curr_pos);

            // record child voxel id
            vector<VOXEL_LAYER_LOC> child_id_vec;
            for (int dx = 0; dx < 2; dx++)
            {
                for (int dy = 0; dy < 2; dy++)
                {
                    for (int dz = 0; dz < 2; dz++)
                    {
                        VOXEL_LAYER_LOC child_id(curr_pos.l + 1, curr_pos.x * 2 + dx, curr_pos.y * 2 + dy, curr_pos.z * 2 + dz);
                        if (occupy_ids_.find(child_id) != occupy_ids_.end())
                        {
                            child_id_vec.push_back(child_id);
                        }
                    }
                }
            }

            // try to merge
            shared_ptr<SurfelChildVoxel> curr_voxel(new SurfelChildVoxel(curr_pos.l));
            int count = 0;
            for (auto id : child_id_vec)
            {
                if (voxels_.find(id) == voxels_.end() || voxels_[id]->surfel_map_->is_surfel == false)
                {
                    count = -1;
                    break;
                }
                curr_voxel->surfels_.push_back(voxels_[id]->surfel_map_);
                count++;
            }
            if (count > 3)
            {
                // check consistency
                bool surfel_consist = true;
                for (int i = 1; i < count; i++)
                {
                    if (curr_voxel->checkTwoSurfelsConsistency(curr_voxel->surfels_[0], curr_voxel->surfels_[i]) == false)
                    {
                        surfel_consist = false;
                        break;
                    }
                }

                // merge voxels
                if (surfel_consist)
                {
                    voxels_[curr_pos] = curr_voxel;
                    voxels_[curr_pos]->half_length_ = layer_voxel_size_[curr_pos.l] / 2.0;
                    voxels_[curr_pos]->voxel_vertex_[0] = voxel_vertex_[0] + curr_pos.x * layer_voxel_size_[curr_pos.l];
                    voxels_[curr_pos]->voxel_vertex_[1] = voxel_vertex_[1] + curr_pos.y * layer_voxel_size_[curr_pos.l];
                    voxels_[curr_pos]->voxel_vertex_[2] = voxel_vertex_[2] + curr_pos.z * layer_voxel_size_[curr_pos.l];
                    // merge surfels then add
                    curr_voxel->merge_surfel();
                    *(curr_voxel->surfel_map_) = *(curr_voxel->surfel_cur_);
                    curr_voxel->surfel_occupy_ = true;
                    curr_voxel->surfels_.clear();
                    // delete child voxels
                    for (auto id : child_id_vec)
                    {
                        voxels_.erase(id);
                        occupy_ids_.erase(id);
                    }
                    // record parent id
                    if (curr_pos.l > 0)
                    {
                        VOXEL_LAYER_LOC parent_pos = curr_pos * 0.5;
                        parent_pos.l = parent_pos.l - 1;
                        if (parent_voxel_id_set.find(parent_pos) == parent_voxel_id_set.end())
                        {
                            parent_voxel_id_set.insert(parent_pos);
                            parent_voxel_id_que.push(parent_pos);
                        }
                    }
                }
            }
        }
    }
};

pointWithCov transformLidarPvToBody(const pointWithCov &p_lidar, Eigen::Matrix3d &R_i_l, Eigen::Vector3d &t_i_l)
{
    pointWithCov p_body;

    // transform to body frame
    Eigen::Vector3d pos_body;
    Eigen::Matrix3d cov_body;
    p_body.point = R_i_l * p_lidar.point + t_i_l;
    p_body.cov = R_i_l * p_lidar.cov * R_i_l.transpose();

    return p_body;
}

void transformBodySfToWorld(const shared_ptr<Surfel> sf_body, const StatesGroup &state, shared_ptr<Surfel> sf_world)
{
    // transform from body frame to world frame
    Eigen::Matrix3d R = state.quat_end.toRotationMatrix();
    sf_world->normal = R * sf_body->normal;
    sf_world->center = R * sf_body->center + state.pos_end;
    sf_world->radius = sf_body->radius;
    sf_world->is_surfel = true;

    // calculate covariance
    Eigen::Matrix3d t_cov = state.cov.block<3, 3>(0, 0);
    Eigen::Matrix3d R_cov = state.cov.block<3, 3>(3, 3);
    Eigen::Vector3d center_trans = R * sf_body->center;
    
    Eigen::Matrix<double, 7, 7> F_s = Eigen::Matrix<double, 7, 7>::Identity();
    F_s.block<3, 3>(0, 0) = R;
    F_s.block<3, 3>(3, 3) = R;
    Eigen::Matrix<double, 7, 3> F_r = Eigen::Matrix<double, 7, 3>::Zero();
    F_r.block<3, 3>(0, 0) = -Utility::hat(sf_world->normal) * Utility::Jleft(state.quat_end);
    F_r.block<3, 3>(3, 0) = -Utility::hat(center_trans) * Utility::Jleft(state.quat_end);
    Eigen::Matrix<double, 7, 3> F_t = Eigen::Matrix<double, 7, 3>::Zero();
    F_t.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();

    sf_world->surfel_cov = F_s * sf_body->surfel_cov * F_s.transpose() +
                           F_r * R_cov * F_r.transpose() +
                           F_t * t_cov * F_t.transpose();
}

void updateVoxelMap(const vector<shared_ptr<Surfel>> &input_sf, const StatesGroup &state, const int curr_frame_id, const int delta_frame_num,
                    const int max_layer, const vector<double> &layer_voxel_size, std::unordered_map<VOXEL_LOC, shared_ptr<SurfelRootVoxel>> &voxel_map)
{
    unordered_set<VOXEL_LOC> voxel_loc;
    for (auto sf : input_sf)
    {
        shared_ptr<Surfel> surfel_ptr(new Surfel);
        transformBodySfToWorld(sf, state, surfel_ptr);
        float loc_xyz[3];
        for (int i = 0; i < 3; i++)
        {
            loc_xyz[i] = surfel_ptr->center[i] / layer_voxel_size[0];
        }
        VOXEL_LOC position((int64_t)(floor(loc_xyz[0])), (int64_t)(floor(loc_xyz[1])), (int64_t)(floor(loc_xyz[2])));
        auto iter = voxel_map.find(position);
        if (iter == voxel_map.end())
        {
            shared_ptr<SurfelRootVoxel> root_voxel(new SurfelRootVoxel(max_layer, layer_voxel_size));
            voxel_map[position] = root_voxel;
            voxel_map[position]->voxel_vertex_[0] = position.x * layer_voxel_size[0];
            voxel_map[position]->voxel_vertex_[1] = position.y * layer_voxel_size[0];
            voxel_map[position]->voxel_vertex_[2] = position.z * layer_voxel_size[0];
        }
        voxel_map[position]->surfels_.push_back(surfel_ptr);
        voxel_loc.insert(position);
    }
    for (auto iter = voxel_loc.begin(); iter != voxel_loc.end(); ++iter)
    {
        if (voxel_map[*iter]->update_frame_id_ < curr_frame_id - delta_frame_num)
        {
            voxel_map[*iter]->voxels_.clear();
        }
        voxel_map[*iter]->update_frame_id_ = curr_frame_id;
        voxel_map[*iter]->voxel_updating();
    }
}

void buildSingleResidual(const shared_ptr<Surfel> sf, const shared_ptr<Surfel> sf_world, 
                         const shared_ptr<SurfelRootVoxel> root_voxel, bool &is_success,
                         double &prob, sfsf &single_sfsf)
{
    for (auto iter = root_voxel->voxels_.begin(); iter != root_voxel->voxels_.end(); ++iter)
    {
        if (iter->second->surfel_map_->is_surfel == false) continue;
        Surfel &surfel_map = *(iter->second->surfel_map_);
        // check angle
        double theta = acos(sf_world->normal.transpose() * surfel_map.normal);
        if (theta > M_PI_4) continue;
        double k_theta = -1.0 / sqrt(1.0 - pow((sf_world->normal.transpose() * surfel_map.normal), 2.0));
        double sigma_theta = k_theta * k_theta * sf_world->normal.transpose() * surfel_map.surfel_cov.block<3, 3>(0, 0) * sf_world->normal;
        sigma_theta += k_theta * k_theta * surfel_map.normal.transpose() * sf_world->surfel_cov.block<3, 3>(3, 3) * surfel_map.normal;
        if (theta > 3 * sqrt(sigma_theta)) continue;
        // check distance
        double dis_to_surfel = fabs(surfel_map.normal.transpose() * (sf_world->center - surfel_map.center));
        Eigen::Matrix<double, 1, 6> J_nc;
        J_nc.block<1, 3>(0, 0) = (sf_world->center - surfel_map.center).transpose();
        J_nc.block<1, 3>(0, 3) = -surfel_map.normal.transpose();
        double sigma_dis = J_nc * surfel_map.surfel_cov.block<6, 6>(0, 0) * J_nc.transpose();
        sigma_dis += surfel_map.normal.transpose() * sf_world->surfel_cov.block<3, 3>(3, 3) * surfel_map.normal;
        if (dis_to_surfel > 3 * sqrt(sigma_dis)) continue;
        // check radius
        double dis_to_center = (sf_world->center - surfel_map.center).norm();
        double range_dis = sqrt(dis_to_center * dis_to_center - dis_to_surfel * dis_to_surfel);
        Eigen::Matrix<double, 1, 6> J_mnc;
        J_mnc.block<1, 3>(0, 0) = -dis_to_surfel / range_dis * (sf_world->center - surfel_map.center).transpose();
        J_mnc.block<1, 3>(0, 3) = (-1.0 / range_dis) * ((sf_world->center - surfel_map.center).transpose() - dis_to_surfel * surfel_map.normal.transpose());
        Eigen::Matrix<double, 1, 3> J_cc;
        J_cc = -J_mnc.block<1, 3>(0, 3);
        double sigma_radius = J_mnc * surfel_map.surfel_cov.block<6, 6>(0, 0) * J_mnc.transpose();
        sigma_radius += J_cc * sf_world->surfel_cov.block<3, 3>(3, 3) * J_cc.transpose();
        double sigma_total = sigma_radius + surfel_map.surfel_cov(6, 6);
        if (range_dis > (surfel_map.radius + 3 * sqrt(sigma_total))) continue;
        // record the residual
        is_success = true;
        double this_prob = exp(-0.5 * dis_to_surfel * dis_to_surfel / sigma_dis) / (sqrt(sigma_dis));
        if (this_prob > prob)
        {
            prob = this_prob;
            single_sfsf.normal_scan = sf->normal;
            single_sfsf.center_scan = sf->center;
            single_sfsf.surfel_scan_cov = sf->surfel_cov;
            single_sfsf.normal_map = surfel_map.normal;
            single_sfsf.center_map = surfel_map.center;
            single_sfsf.surfel_map_cov = surfel_map.surfel_cov;
        }
    }
}

void buildResidualList(const unordered_map<VOXEL_LOC, shared_ptr<SurfelRootVoxel>> &voxel_map,
                       const double voxel_size, const StatesGroup &state,
                       const vector<shared_ptr<Surfel>> &sf_list, vector<sfsf> &sfsf_list)
{
    sfsf_list.clear();
    for (auto sf : sf_list)
    {
        shared_ptr<Surfel> sf_world(new Surfel);
        transformBodySfToWorld(sf, state, sf_world);
        float loc_xyz[3];
        for (int i = 0; i < 3; i++)
        {
            loc_xyz[i] = sf_world->center[i] / voxel_size;
        }
        VOXEL_LOC position((int64_t)(floor(loc_xyz[0])), (int64_t)(floor(loc_xyz[1])), (int64_t)(floor(loc_xyz[2])));
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end())
        {
            shared_ptr<SurfelRootVoxel> curr_voxel = iter->second;
            sfsf single_sfsf;
            bool is_success = false;
            double prob = 0;
            buildSingleResidual(sf, sf_world, curr_voxel, is_success, prob, single_sfsf);
            if (!is_success)
            {
                VOXEL_LOC near_position = position;
                if (sf_world->center[0] > (curr_voxel->voxel_vertex_[0] + 0.75 * voxel_size))
                    near_position.x += 1;
                else if (sf_world->center[0] < (curr_voxel->voxel_vertex_[0] + 0.25 * voxel_size))
                    near_position.x -= 1;
                if (sf_world->center[1] > (curr_voxel->voxel_vertex_[1] + 0.75 * voxel_size))
                    near_position.y += 1;
                else if (sf_world->center[1] < (curr_voxel->voxel_vertex_[1] + 0.25 * voxel_size))
                    near_position.y -= 1;
                if (sf_world->center[2] > (curr_voxel->voxel_vertex_[2] + 0.75 * voxel_size))
                    near_position.z += 1;
                else if (sf_world->center[2] < (curr_voxel->voxel_vertex_[2] + 0.25 * voxel_size))
                    near_position.z -= 1;
                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end())
                {
                    buildSingleResidual(sf, sf_world, iter_near->second, is_success, prob, single_sfsf);
                }
            }
            if (is_success)
            {
                sfsf_list.push_back(single_sfsf);
            }
        }
    }
}

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin) v = vmin;

    if (v > vmax) v = vmax;

    double dr, dg, db;

    if (v < 0.1242)
    {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    }
    else if (v < 0.3747)
    {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    }
    else if (v < 0.6253)
    {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    }
    else if (v < 0.8758)
    {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    }
    else
    {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t)(255 * dr);
    g = (uint8_t)(255 * dg);
    b = (uint8_t)(255 * db);
}

void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q)
{
    Eigen::Matrix3d rot;
    rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
        z_vec(1), z_vec(2);
    Eigen::Matrix3d rotation = rot.transpose();
    Eigen::Quaterniond eq(rotation);
    eq.normalize();
    q.w = eq.w();
    q.x = eq.x();
    q.y = eq.y();
    q.z = eq.z();
}

void pubSingleSurfel(visualization_msgs::MarkerArray &surfel_pub,
                    const std::string surfel_ns, const shared_ptr<Surfel> single_surfel,
                    const float alpha, const Eigen::Vector3d rgb, const int id)
{
    visualization_msgs::Marker surfel;
    surfel.header.frame_id = "lidar_init";
    surfel.header.stamp = ros::Time();
    surfel.ns = surfel_ns;
    surfel.id = id;
    surfel.type = visualization_msgs::Marker::CYLINDER;
    surfel.action = visualization_msgs::Marker::ADD;
    surfel.pose.position.x = single_surfel->center[0];
    surfel.pose.position.y = single_surfel->center[1];
    surfel.pose.position.z = single_surfel->center[2];
    geometry_msgs::Quaternion q;
    Eigen::Vector3d x_normal, y_normal;
    x_normal = Eigen::Vector3d(single_surfel->normal.x() + 1.0, single_surfel->normal.y(), single_surfel->normal.z());
    x_normal.normalize();
    y_normal = single_surfel->normal.cross(x_normal);
    x_normal = y_normal.cross(single_surfel->normal);
    calcVectQuation(x_normal, y_normal, single_surfel->normal, q);
    surfel.pose.orientation = q;
    surfel.scale.x = 2.0 * single_surfel->radius;
    surfel.scale.y = 2.0 * single_surfel->radius;
    surfel.scale.z = 0.05;
    surfel.color.a = alpha;
    surfel.color.r = rgb(0);
    surfel.color.g = rgb(1);
    surfel.color.b = rgb(2);
    surfel.lifetime = ros::Duration(1.0);
    surfel_pub.markers.push_back(surfel);
}

void pubVoxelSurfel(const vector<shared_ptr<Surfel>> &surfel_list, const ros::Publisher &voxel_surfel_pub)
{
    double max_trace = 0.15;
    double pow_num = 0.125;
    double alpha = 1.0;
    visualization_msgs::MarkerArray voxel_surfel;
    voxel_surfel.markers.reserve(1000000);
    for (int i = 0; i < (int)surfel_list.size(); i++)
    {
        Eigen::Vector3d surfel_cov = surfel_list[i]->surfel_cov.block<3, 3>(0, 0).diagonal();
        double trace = surfel_cov.sum();
        if (trace >= max_trace)
        {
            trace = max_trace;
        }
        trace = trace * (1.0 / max_trace);
        trace = pow(trace, pow_num);
        uint8_t r, g, b;
        mapJet(trace, 0, 1, r, g, b);
        Eigen::Vector3d surfel_rgb(r / 256.0, g / 256.0, b / 256.0);
        pubSingleSurfel(voxel_surfel, "prob_surfel", surfel_list[i], alpha, surfel_rgb, i);
    }
    voxel_surfel_pub.publish(voxel_surfel);
}

void GetVoxelMapSurfel(const shared_ptr<SurfelRootVoxel> root_voxel, const int pub_max_voxel_layer,
                       std::vector<shared_ptr<Surfel>> &surfel_list)
{
    for (auto iter = root_voxel->voxels_.begin(); iter != root_voxel->voxels_.end(); ++iter)
    {
        if (iter->first.l > pub_max_voxel_layer) continue;

        if (iter->second->surfel_map_->is_surfel)
        {
            surfel_list.push_back(iter->second->surfel_map_);
        }
    }
}

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, 
                 const float degree_inc, Eigen::Matrix3d &cov)
{
    float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    float range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3d direction(pb);
    direction.normalize();
    Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0),
         base_vector1(1), base_vector2(1),
         base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * Utility::hat(direction) * N;
    cov = direction * range_var * direction.transpose() +
          A * direction_var * A.transpose();
}

#endif // VOXEL_UTIL_H_