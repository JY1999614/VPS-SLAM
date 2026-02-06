#include "vps_slam/voxel_util.h"

class SurfelExtraction
{
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_imu;
    ros::Subscriber sub_cloud;

    ros::Publisher pub_full_cloud;
    ros::Publisher pub_surfel_with_cov;
    ros::Publisher pub_surfel_vis;

    mutex mtx_buffer;

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    queue<sensor_msgs::PointCloud2ConstPtr> cloudBuf;
    vector<sensor_msgs::ImuConstPtr> imu_vec;

    double currTime;
    pcl::PointCloud<PointType>::Ptr currCloud;
    pcl::PointCloud<PointType>::Ptr currCloudDS;
    pcl::PointCloud<PointType>::Ptr currCloudOut;

    Eigen::Vector3d bias_g;
    Eigen::Quaterniond delta_q_imu;

    vector<pointWithCov> pv_body_list;
    vector<shared_ptr<Surfel>> sf_body_list;

    pcl::UniformSampling<PointType> UniformSamplingCloud;

    thread surfel_extraction_thread;

    // topic
    string imu_topic = "/imu/data";
    // imu
    vector<double> extrinT, extrinR;
    Eigen::Matrix3d Lid_matrix_to_IMU;
    Eigen::Quaterniond Lid_rot_to_IMU;
    Eigen::Vector3d Lid_offset_to_IMU;
    // noise_model
    double ranging_cov;
    double angle_cov;
    // voxel
    int max_layer;
    int layer_point_size;
    double max_voxel_size;
    double flat_threshold;
    double min_voxel_size;

public:
    SurfelExtraction();
    ~SurfelExtraction();
    void allocateMemory();
    void imuHandler(const sensor_msgs::ImuConstPtr& msg);
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
    void SurfelExtractionThread();
    void processIMU();
    void integrateIMU(double dt, const Eigen::Vector3d &angular_velocity);
    void adjustDistortion();
    void calculateBodyCov();
    void extractBodySurfel();
    void publishPointAndSurfel();
    pcl::PointCloud<PointType>::Ptr transformCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Vector3d t, Eigen::Quaterniond q);
    void toMultiSurfelsMsg(const vector<shared_ptr<Surfel>> &ps_list, vps_slam::MultiSurfels &msg);
};

SurfelExtraction::SurfelExtraction()
{
    // topic
    nh.param<string>("topic/imu_topic", imu_topic, "/imu/data");
    // imu
    nh.param<vector<double>>("imu/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("imu/extrinsic_R", extrinR, vector<double>());
    Lid_offset_to_IMU << extrinT[0], extrinT[1], extrinT[2];
    Lid_matrix_to_IMU << extrinR[0], extrinR[1], extrinR[2],
                         extrinR[3], extrinR[4], extrinR[5],
                         extrinR[6], extrinR[7], extrinR[8];
    Lid_rot_to_IMU = Lid_matrix_to_IMU;
    // noise_model
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    // voxel
    nh.param<int>("voxel/max_layer", max_layer, 3);
    nh.param<int>("voxel/layer_point_size", layer_point_size, 5);
    nh.param<double>("voxel/max_voxel_size", max_voxel_size, 3.0);
    nh.param<double>("voxel/flat_threshold", flat_threshold, 0.01);
    min_voxel_size = max_voxel_size / (pow(2.0, max_layer));

    sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 1000, &SurfelExtraction::imuHandler, this);
    sub_cloud = nh.subscribe<sensor_msgs::PointCloud2>("laser_cloud_full", 100, &SurfelExtraction::cloudHandler, this);

    pub_full_cloud = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_full_2", 100);
    pub_surfel_with_cov = nh.advertise<vps_slam::MultiSurfels>("surfels_with_cov", 100);
    pub_surfel_vis = nh.advertise<visualization_msgs::MarkerArray>("scan_surfels_vis", 100);

    surfel_extraction_thread = thread(&SurfelExtraction::SurfelExtractionThread, this);

    allocateMemory();
}

SurfelExtraction::~SurfelExtraction()
{
    cout << "--SurfelExtraction: exit!!!" << endl;
}

void SurfelExtraction::allocateMemory()
{
    currCloud.reset(new pcl::PointCloud<PointType>());
    currCloudDS.reset(new pcl::PointCloud<PointType>());
    currCloudOut.reset(new pcl::PointCloud<PointType>());

    UniformSamplingCloud.setRadiusSearch(0.5);
}

void SurfelExtraction::imuHandler(const sensor_msgs::ImuConstPtr& msg)
{
    mtx_buffer.lock();
    imuBuf.push(msg);
    mtx_buffer.unlock();
}

void SurfelExtraction::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mtx_buffer.lock();
    cloudBuf.push(msg);
    mtx_buffer.unlock();
}

void SurfelExtraction::SurfelExtractionThread()
{
    cout << "--SurfelExtractionThread begain!!!" << endl;
    
    ros::Rate rate(20);
    while (ros::ok())
    {
        if (!cloudBuf.empty())
        {
            TicToc t_whole;

            mtx_buffer.lock();

            currTime = cloudBuf.front()->header.stamp.toSec();

            // processs imu
            if (imuBuf.empty() || imuBuf.back()->header.stamp.toSec() < currTime)
            {
                // waiting for imu message
                mtx_buffer.unlock();
                cout << "--SurfelExtraction: waiting for imu message!!!" << endl;
                continue;
            }
            else if (imuBuf.front()->header.stamp.toSec() >= currTime)
            {
                cloudBuf.pop();
                mtx_buffer.unlock();
                cout << "--SurfelExtraction: pop current lasercloud, then waiting for imu message!!!" << endl;
                continue;
            }
            imu_vec.clear();
            while (imuBuf.front()->header.stamp.toSec() < currTime)
            {
                imu_vec.push_back(imuBuf.front());
                imuBuf.pop();
            }
            imu_vec.push_back(imuBuf.front());

            // process pointcloud
            currCloud->clear();
            pcl::fromROSMsg(*cloudBuf.front(), *currCloud);
            cloudBuf.pop();

            mtx_buffer.unlock();

            // 1. Integrate imu data
            processIMU();
            // 2. Adjust point cloud distortion
            adjustDistortion();
            // 3. Calculate point covariance and transform to body frame
            calculateBodyCov();
            // 4. Extract surfel in body frame
            extractBodySurfel();
            // 5. Publish point cloud and surfel
            publishPointAndSurfel();

            cout << "--SurfelExtraction: whole time: " << t_whole.toc() << endl << endl;
        }

        rate.sleep();
    }
}

void SurfelExtraction::processIMU()
{
    // imu integration
    static bool first_imu = true;
    static double current_time = -1;
    double rx = 0, ry = 0, rz = 0;
    if (first_imu)
    {
        Eigen::Vector3d sum_gyr = Eigen::Vector3d::Zero();
        for (auto &imu_msg : imu_vec)
        {
            Eigen::Vector3d curr_gyr;
            curr_gyr << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
            sum_gyr += curr_gyr;
        }
        bias_g = sum_gyr / imu_vec.size();
        first_imu = false;
    }

    // imu message interpolation
    delta_q_imu = Eigen::Quaterniond::Identity();
    for (auto &imu_msg : imu_vec)
    {
        double t = imu_msg->header.stamp.toSec();
        if (t <= currTime)
        {
            if (current_time < 0) current_time = t;
            double dt = t - current_time;
            ROS_ASSERT(dt >= 0);
            current_time = t;
            rx = imu_msg->angular_velocity.x;
            ry = imu_msg->angular_velocity.y;
            rz = imu_msg->angular_velocity.z;
            integrateIMU(dt, Eigen::Vector3d(rx, ry, rz));
        }
        else
        {
            double dt_1 = currTime - current_time;
            double dt_2 = t - currTime;
            current_time = currTime;
            ROS_ASSERT(dt_1 >= 0);
            ROS_ASSERT(dt_2 >= 0);
            ROS_ASSERT(dt_1 + dt_2 >= 0);
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);
            rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
            ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
            rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
            integrateIMU(dt_1, Eigen::Vector3d(rx, ry, rz));
        }
    }
}

void SurfelExtraction::integrateIMU(double dt, const Eigen::Vector3d &angular_velocity)
{
    static bool first_imu = true;
    static Eigen::Vector3d gyr_0;
    if (first_imu)
    {
        gyr_0 = angular_velocity;
        first_imu = false;
    }

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - bias_g;
    delta_q_imu = delta_q_imu * Utility::deltaQ(un_gyr * dt);
    delta_q_imu.normalize();

    gyr_0 = angular_velocity;
}

void SurfelExtraction::adjustDistortion()
{
    Eigen::Quaterniond delta_q_lid = Lid_rot_to_IMU.inverse() * delta_q_imu * Lid_rot_to_IMU;
    Eigen::Quaterniond delta_q_inverse = delta_q_lid.inverse();

    for (int i = 0; i < (int)currCloud->points.size(); i++)
    {
        double s = 1 - currCloud->points[i].intensity;
        Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, delta_q_inverse);
        Eigen::Vector3d point(currCloud->points[i].x, currCloud->points[i].y, currCloud->points[i].z);
        Eigen::Vector3d point_end = q_point_last * point;
        currCloud->points[i].x = point_end.x();
        currCloud->points[i].y = point_end.y();
        currCloud->points[i].z = point_end.z();
        currCloud->points[i].intensity = i;
    }
}

void SurfelExtraction::calculateBodyCov()
{
    // calculate points covariance for extract surfel
    pv_body_list.clear();
    for (int i = 0; i < (int)currCloud->points.size(); i++)
    {
        // calculate lidar point covariance in lidar frame
        pointWithCov pv, pv_body;
        pv.point << currCloud->points[i].x, currCloud->points[i].y, currCloud->points[i].z;
        if (pv.point[2] == 0) pv.point[2] = 0.001;
        calcBodyCov(pv.point, ranging_cov, angle_cov, pv.cov);
        // transform to body frame
        pv_body = transformLidarPvToBody(pv, Lid_matrix_to_IMU, Lid_offset_to_IMU);
        pv_body_list.push_back(pv_body);
    }
}

void SurfelExtraction::extractBodySurfel()
{
    // Project points to voxelscan
    std::unordered_map<VOXEL_LOC, shared_ptr<PointsRootVoxel>> voxel_scan;
    for (int i = 0; i < (int)pv_body_list.size(); i++)
    {
        pointWithCov this_point = pv_body_list[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = this_point.point[j] / max_voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = voxel_scan.find(position);
        if (iter == voxel_scan.end())
        {
            shared_ptr<PointsRootVoxel> root_voxel(new PointsRootVoxel(max_layer, layer_point_size, flat_threshold));
            voxel_scan[position] = root_voxel;
            voxel_scan[position]->voxel_vertex_[0] = position.x * max_voxel_size;
            voxel_scan[position]->voxel_vertex_[1] = position.y * max_voxel_size;
            voxel_scan[position]->voxel_vertex_[2] = position.z * max_voxel_size;
            voxel_scan[position]->max_voxel_size_ = max_voxel_size;
            voxel_scan[position]->min_voxel_size_ = min_voxel_size;
        }
        voxel_scan[position]->points_.push_back(this_point);
    }

    // Calculate surfel parameter
    sf_body_list.clear();
    for (auto iter = voxel_scan.begin(); iter != voxel_scan.end(); ++iter)
    {
        iter->second->surfel_fitting();
        iter->second->surfel_splitting();
        for (auto sf = iter->second->voxels_.begin(); sf != iter->second->voxels_.end(); ++sf)
        {
            sf_body_list.push_back(sf->second->surfel_ptr_);
        }
    }
}

void SurfelExtraction::publishPointAndSurfel()
{
    // downsample points
    UniformSamplingCloud.setInputCloud(currCloud);
    UniformSamplingCloud.filter(*currCloudDS);

    // publish point cloud in body frame
    pcl::PointCloud<PointType>::Ptr currCloudOut(new pcl::PointCloud<PointType>());
    *currCloudOut += *transformCloud(currCloudDS, Lid_offset_to_IMU, Lid_rot_to_IMU);

    sensor_msgs::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*currCloudOut, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time().fromSec(currTime);
    laserCloudMsg.header.frame_id = "lidar_init";
    pub_full_cloud.publish(laserCloudMsg);
    
    // publish surfels with covariance
    vps_slam::MultiSurfels surfelsWithCovMsg;
    toMultiSurfelsMsg(sf_body_list, surfelsWithCovMsg);
    surfelsWithCovMsg.header.stamp = ros::Time().fromSec(currTime);
    surfelsWithCovMsg.header.frame_id = "lidar_init";
    pub_surfel_with_cov.publish(surfelsWithCovMsg);

    // visualize scan surfels
    pubVoxelSurfel(sf_body_list, pub_surfel_vis);
}

pcl::PointCloud<PointType>::Ptr SurfelExtraction::transformCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Vector3d t, Eigen::Quaterniond q)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;
    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        Eigen::Vector3d point_curr(pointFrom->x, pointFrom->y, pointFrom->z);
        Eigen::Vector3d point_w = q * point_curr + t;
        cloudOut->points[i].x = point_w.x();
        cloudOut->points[i].y = point_w.y();
        cloudOut->points[i].z = point_w.z();
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
}

void SurfelExtraction::toMultiSurfelsMsg(const vector<shared_ptr<Surfel>> &ps_list, vps_slam::MultiSurfels &msg)
{
    for (auto ps_ptr : ps_list)
    {
        vps_slam::SurfelWithCovariance ps_msg;
        ps_msg.normal.x = ps_ptr->normal.x();
        ps_msg.normal.y = ps_ptr->normal.y();
        ps_msg.normal.z = ps_ptr->normal.z();
        ps_msg.center.x = ps_ptr->center.x();
        ps_msg.center.y = ps_ptr->center.y();
        ps_msg.center.z = ps_ptr->center.z();
        ps_msg.radius = ps_ptr->radius;
        for (int i = 0; i < 49; i++)
        {
            ps_msg.surfel_cov[i] = ps_ptr->surfel_cov(i);
        }
        msg.surfels.push_back(ps_msg);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SurfelExtraction");
    SurfelExtraction SE;
    ROS_INFO("\033[1;32m---->\033[0m SurfelExtraction Started.");
    ros::spin();
    return 0;
}
