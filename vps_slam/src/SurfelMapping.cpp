#include "vps_slam/voxel_util.h"
#include "vps_slam/PreIntegration.h"
#include "constraintFactor.hpp"

class SurfelMapping
{
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_imu;
    ros::Subscriber sub_full_cloud;
    ros::Subscriber sub_surfel_with_cov;

    ros::Publisher pub_odom_aftmapped;
    ros::Publisher pub_path_aftmapped;
    ros::Publisher pub_surfel_vis;

    mutex mtx_buffer;

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    queue<sensor_msgs::PointCloud2ConstPtr> cloudBuf;
    queue<vps_slam::MultiSurfelsConstPtr> surfelsBuf;
    vector<sensor_msgs::ImuConstPtr> imu_vec;

    double timeCloud, timeSurfels;
    pcl::PointCloud<PointType>::Ptr currCloud;
    vector<shared_ptr<Surfel>> sf_body_list;
    StatesGroup curr_state;

    // slide window
    deque<StatesGroup> state_window;
    deque<PreIntegration*> pre_integration_window;
    deque<pcl::PointCloud<PointType>::Ptr> cloud_window;
    deque<vector<shared_ptr<Surfel>>> sf_body_list_window;
    double **para_t;
    double **para_q;
    double **para_speed_bias;

    // mapping
    int keyFrameNum = 0;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses;
    std::map<int, StatesGroup> statesKeyFrames;
    std::map<int, pcl::PointCloud<PointType>::Ptr> cloudKeyFrames;
    std::map<int, vector<shared_ptr<Surfel>>> surfelsKeyFrames;
    std::map<int, float> travel_distance; // m
    std::unordered_map<VOXEL_LOC, shared_ptr<SurfelRootVoxel>> voxel_map;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    nav_msgs::Path pathAftMapped;
    thread surfel_mapping_thread;

    ofstream f_pose_tum, f_covariance;
    ofstream f_time;

    double time_whole, time_feature, time_optimization, time_covariance, time_mapping;

    // topic
    string imu_topic = "/imu/data";
    // voxel
    int max_layer;
    double max_voxel_size;
    vector<double> layer_voxel_size;
    // mapping
    int window_size;
    int voxelRebuildDeltaFrameNum;
    double keyframeAddingAngle;
    double keyframeAddingDistance;
    

public:
    SurfelMapping();
    ~SurfelMapping();
    void allocateMemory();
    void imuHandler(const sensor_msgs::ImuConstPtr& msg);
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
    void surfelsHandler(const vps_slam::MultiSurfelsConstPtr &msg);
    void surfelMappingThread();
    void slideWindow();
    void processIMU();
    void integrateIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void scan2MapOptimization();
    void saveKeyFramesAndMap();
    void publishOdometryAndPath();
    void pubVoxelMap(const std::unordered_map<VOXEL_LOC, shared_ptr<SurfelRootVoxel>> &voxel_map, const int pub_max_voxel_layer, const ros::Publisher &voxel_surfel_pub);
    void fromMultiSurfelsMsg(const vps_slam::MultiSurfels &msg, vector<shared_ptr<Surfel>> &sf_list);
};

SurfelMapping::SurfelMapping()
{
    // topic
    nh.param<string>("topic/imu_topic", imu_topic, "/imu/data");
    // voxel
    nh.param<int>("voxel/max_layer", max_layer, 3);
    nh.param<double>("voxel/max_voxel_size", max_voxel_size, 3.0);
    for (int i = 0; i <= max_layer; i++)
    {
        double curr_voxel_size = max_voxel_size / (pow(2.0, i));
        layer_voxel_size.push_back(curr_voxel_size);
    }
    // mapping
    nh.param<int>("mapping/window_size", window_size, 3);
    nh.param<int>("mapping/voxelRebuildDeltaFrameNum", voxelRebuildDeltaFrameNum, 50);
    nh.param<double>("mapping/keyframeAddingAngle", keyframeAddingAngle, 15.0);
    nh.param<double>("mapping/keyframeAddingDistance", keyframeAddingDistance, 0.3);

    sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 1000, &SurfelMapping::imuHandler, this);
    sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("laser_cloud_full_2", 100, &SurfelMapping::cloudHandler, this);
    sub_surfel_with_cov = nh.subscribe<vps_slam::MultiSurfels>("surfels_with_cov", 100, &SurfelMapping::surfelsHandler, this);

    pub_odom_aftmapped = nh.advertise<nav_msgs::Odometry>("odom_aft_mapped", 100);
    pub_path_aftmapped = nh.advertise<nav_msgs::Path>("path_aft_mapped", 100);
    pub_surfel_vis = nh.advertise<visualization_msgs::MarkerArray>("map_surfels_vis", 2);

    laserOdometryTrans.frame_id_ = "lidar_init";
    laserOdometryTrans.child_frame_id_ = "aft_mapped";

    f_pose_tum.open("/home/jy/SLAM/experiment/result/odometry/pose_myodom.txt", fstream::out);
    f_covariance.open("/home/jy/SLAM/experiment/result/odometry/covariance.txt", fstream::out);

    f_time.open("/home/jy/SLAM/experiment/result/odometry/time.txt", fstream::out);
    
    surfel_mapping_thread = thread(&SurfelMapping::surfelMappingThread, this);

    allocateMemory();
}

SurfelMapping::~SurfelMapping()
{
    f_pose_tum.close();
    f_covariance.close();
    f_time.close();
    cout << "--SurfelMapping: exit!!!" << endl;
}

void SurfelMapping::allocateMemory()
{
    currCloud.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses.reset(new pcl::PointCloud<PointType>());

    para_t = new double *[window_size];
    para_q = new double *[window_size];
    para_speed_bias = new double *[window_size];
    for (int i = 0; i < window_size; i++)
    {
        para_t[i] = new double[3];
        para_q[i] = new double[4];
        para_speed_bias[i] = new double[9];
    }
}

void SurfelMapping::imuHandler(const sensor_msgs::ImuConstPtr& msg)
{
    mtx_buffer.lock();
    imuBuf.push(msg);
    mtx_buffer.unlock();
}

void SurfelMapping::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mtx_buffer.lock();
    cloudBuf.push(msg);
    mtx_buffer.unlock();
}

void SurfelMapping::surfelsHandler(const vps_slam::MultiSurfelsConstPtr &msg)
{
    mtx_buffer.lock();
    surfelsBuf.push(msg);
    mtx_buffer.unlock();
}

void SurfelMapping::surfelMappingThread()
{
    cout << "--surfelMappingThread begain!!!" << endl;

    ros::Rate rate(20);
    while (ros::ok())
    {
        if (!surfelsBuf.empty() && !cloudBuf.empty())
        {
            mtx_buffer.lock();

            while (!cloudBuf.empty() && cloudBuf.front()->header.stamp.toSec() < surfelsBuf.front()->header.stamp.toSec())
                cloudBuf.pop();
            if (cloudBuf.empty())
            {
                mtx_buffer.unlock();
                continue;
            }
            
            timeSurfels = surfelsBuf.front()->header.stamp.toSec();
            timeCloud = cloudBuf.front()->header.stamp.toSec();

            if (timeCloud != timeSurfels)
            {
                cout << "--SurfelMapping: unsync messeage!" << endl;
                mtx_buffer.unlock();
                continue;
            }
            
            // processs imu
            if (imuBuf.empty() || imuBuf.back()->header.stamp.toSec() < timeCloud)
            {
                // waiting for imu message
                mtx_buffer.unlock();
                cout << "--SurfelMapping: waiting for imu message!!!" << endl;
                continue;
            }
            else if (imuBuf.front()->header.stamp.toSec() >= timeCloud)
            {
                surfelsBuf.pop();
                mtx_buffer.unlock();
                cout << "--SurfelMapping: pop current surfel frame, then waiting for imu message!!!" << endl;
                continue;
            }
            imu_vec.clear();
            while (imuBuf.front()->header.stamp.toSec() < timeCloud)
            {
                imu_vec.push_back(imuBuf.front());
                imuBuf.pop();
            }
            imu_vec.push_back(imuBuf.front());

            // process surfels and cloud
            sf_body_list.clear();
            fromMultiSurfelsMsg(*surfelsBuf.front(), sf_body_list);
            surfelsBuf.pop();

            currCloud->clear();
            pcl::fromROSMsg(*cloudBuf.front(), *currCloud);
            cloudBuf.pop();
            
            mtx_buffer.unlock();

            TicToc t_whole;
            slideWindow();
            scan2MapOptimization();
            TicToc t_mapping;
            saveKeyFramesAndMap();
            time_mapping = t_mapping.toc();
            time_whole = t_whole.toc();
            publishOdometryAndPath();
            pubVoxelMap(voxel_map, max_layer, pub_surfel_vis);
            f_time << std::fixed << std::setprecision(9) << time_whole << " " << time_feature << " " << time_optimization << " " << time_covariance << " " << time_mapping << std::endl;
            time_whole = 0;
            time_feature = 0;
            time_optimization = 0;
            time_covariance = 0;
            time_mapping = 0;
            cout << "--SurfelMapping: whole time: " << t_whole.toc() << endl << endl;
        }

        rate.sleep();
    }
}

void SurfelMapping::slideWindow()
{
    // Add current frame to slide window
    // 1. state
    processIMU();
    // 2. pointcloud
    pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*currCloud, *thisKeyFrame);
    cloud_window.push_back(thisKeyFrame);
    // 3. surfels
    sf_body_list_window.push_back(sf_body_list);

    // Pop old frame
    if ((int)state_window.size() > window_size)
    {
        state_window.pop_front();
        delete pre_integration_window.front();
        pre_integration_window.pop_front();
        cloud_window.pop_front();
        sf_body_list_window.pop_front();
    }
}

void SurfelMapping::processIMU()
{
    // imu integration
    static bool first_imu = true;
    static double current_time = -1;
    double ax = 0, ay = 0, az = 0, rx = 0, ry = 0, rz = 0;
    if (first_imu)
    {
        Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d sum_gyr = Eigen::Vector3d::Zero();
        for (auto &imu_msg : imu_vec)
        {
            Eigen::Vector3d curr_acc, curr_gyr;
            curr_acc << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;
            curr_gyr << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
            sum_acc += curr_acc;
            sum_gyr += curr_gyr;
        }
        gravity = sum_acc / imu_vec.size();
        curr_state.bias_g = sum_gyr / imu_vec.size();
        first_imu = false;
    }
    else
    {
        curr_state = state_window.back();
    }
    curr_state.time = timeCloud;

    // imu message interpolation
    for (auto &imu_msg : imu_vec)
    {
        double t = imu_msg->header.stamp.toSec();
        if (t <= timeCloud)
        {
            if (current_time < 0) current_time = t;
            double dt = t - current_time;
            ROS_ASSERT(dt >= 0);
            current_time = t;
            ax = imu_msg->linear_acceleration.x;
            ay = imu_msg->linear_acceleration.y;
            az = imu_msg->linear_acceleration.z;
            rx = imu_msg->angular_velocity.x;
            ry = imu_msg->angular_velocity.y;
            rz = imu_msg->angular_velocity.z;
            integrateIMU(dt, Eigen::Vector3d(ax, ay, az), Eigen::Vector3d(rx, ry, rz));
        }
        else
        {
            double dt_1 = timeCloud - current_time;
            double dt_2 = t - timeCloud;
            current_time = timeCloud;
            ROS_ASSERT(dt_1 >= 0);
            ROS_ASSERT(dt_2 >= 0);
            ROS_ASSERT(dt_1 + dt_2 >= 0);
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);
            ax = w1 * ax + w2 * imu_msg->linear_acceleration.x;
            ax = w1 * ay + w2 * imu_msg->linear_acceleration.y;
            ax = w1 * az + w2 * imu_msg->linear_acceleration.z;
            rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
            ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
            rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
            integrateIMU(dt_1, Eigen::Vector3d(ax, ay, az), Eigen::Vector3d(rx, ry, rz));
        }
    }

    state_window.push_back(curr_state);
}

void SurfelMapping::integrateIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity)
{
    static bool first_imu = true;
    static Eigen::Vector3d acc_0, gyr_0;
    if (first_imu)
    {
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        first_imu = false;
    }
    if (pre_integration_window.size() <= state_window.size())
    {
        PreIntegration *pre_integration = new PreIntegration(acc_0, gyr_0, curr_state.bias_a, curr_state.bias_g);
        pre_integration->g_vec = gravity;
        pre_integration_window.push_back(pre_integration);
    }

    pre_integration_window.back()->push_back(dt, linear_acceleration, angular_velocity);

    Eigen::Vector3d un_acc_0 = curr_state.quat_end * (acc_0 - curr_state.bias_a) - gravity;
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - curr_state.bias_g;
    Eigen::Matrix3d R_curr = curr_state.quat_end.toRotationMatrix() * Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Eigen::Vector3d un_acc_1 = curr_state.quat_end * (linear_acceleration - curr_state.bias_a) - gravity;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Eigen::Vector3d t_curr = curr_state.pos_end + curr_state.vel_end * dt + 0.5 * un_acc * dt * dt;
    Eigen::Vector3d v_curr = curr_state.vel_end + un_acc * dt;

    // calculate covariance
    Eigen::Vector3d w_x = 0.5 * (gyr_0 + angular_velocity) - curr_state.bias_g;
    Eigen::Vector3d a_0_x = acc_0 - curr_state.bias_a;
    Eigen::Vector3d a_1_x = linear_acceleration - curr_state.bias_a;
    Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;
    R_w_x << 0, -w_x(2), w_x(1),
             w_x(2), 0, -w_x(0),
             -w_x(1), w_x(0), 0;
    R_a_0_x << 0, -a_0_x(2), a_0_x(1),
               a_0_x(2), 0, -a_0_x(0),
               -a_0_x(1), a_0_x(0), 0;
    R_a_1_x << 0, -a_1_x(2), a_1_x(1),
               a_1_x(2), 0, -a_1_x(0),
               -a_1_x(1), a_1_x(0), 0;

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3) = -0.25 * curr_state.quat_end.toRotationMatrix() * R_a_0_x * dt * dt +
                          -0.25 * R_curr * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(0, 9) = -0.25 * (curr_state.quat_end.toRotationMatrix() + R_curr) * dt * dt;
    F.block<3, 3>(0, 12) = 0.25 * R_curr * R_a_1_x * dt * dt * dt;
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
    F.block<3, 3>(3, 12) = -1.0 * Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(6, 3) = -0.5 * curr_state.quat_end.toRotationMatrix() * R_a_0_x * dt +
                          -0.5 * R_curr * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9) = -0.5 * (curr_state.quat_end.toRotationMatrix() + R_curr) * dt;
    F.block<3, 3>(6, 12) = 0.5 * R_curr * R_a_1_x * dt * dt;
    F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
    V.block<3, 3>(0, 0) = 0.25 * curr_state.quat_end.toRotationMatrix() * dt * dt;
    V.block<3, 3>(0, 3) = -0.125 * R_curr * R_a_1_x * dt * dt * dt;
    V.block<3, 3>(0, 6) = 0.25 * R_curr * dt * dt;
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(3, 9) = 0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(6, 0) = 0.5 * curr_state.quat_end.toRotationMatrix() * dt;
    V.block<3, 3>(6, 3) = -0.25 * R_curr * R_a_1_x * dt * dt;
    V.block<3, 3>(6, 6) = 0.5 * R_curr * dt;
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12) = Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(12, 15) = Eigen::Matrix3d::Identity() * dt;

    curr_state.cov = F * curr_state.cov * F.transpose() + V * pre_integration_window.back()->noise * V.transpose();
    curr_state.pos_end = t_curr;
    curr_state.vel_end = v_curr;
    curr_state.quat_end = R_curr;
    curr_state.quat_end.normalize();

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void SurfelMapping::scan2MapOptimization()
{
    if (cloudKeyPoses->points.empty()) return;

    for (int iterCount = 0; iterCount < 1; iterCount++)
    {
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization = new ceres::QuaternionParameterization();
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);

        // eigen to double
        for (int i = 0; i < (int)state_window.size(); i++)
        {
            // cout << "bef state_window[" << i << "]: " << state_window[i].pos_end.transpose() << endl;
            para_t[i][0] = state_window[i].pos_end.x();
            para_t[i][1] = state_window[i].pos_end.y();
            para_t[i][2] = state_window[i].pos_end.z();
            para_q[i][0] = state_window[i].quat_end.w();
            para_q[i][1] = state_window[i].quat_end.x();
            para_q[i][2] = state_window[i].quat_end.y();
            para_q[i][3] = state_window[i].quat_end.z();
            para_speed_bias[i][0] = state_window[i].vel_end.x();
            para_speed_bias[i][1] = state_window[i].vel_end.y();
            para_speed_bias[i][2] = state_window[i].vel_end.z();
            para_speed_bias[i][3] = state_window[i].bias_a.x();
            para_speed_bias[i][4] = state_window[i].bias_a.y();
            para_speed_bias[i][5] = state_window[i].bias_a.z();
            para_speed_bias[i][6] = state_window[i].bias_g.x();
            para_speed_bias[i][7] = state_window[i].bias_g.y();
            para_speed_bias[i][8] = state_window[i].bias_g.z();
            problem.AddParameterBlock(para_t[i], 3);
            problem.AddParameterBlock(para_q[i], 4, q_parameterization);
            problem.AddParameterBlock(para_speed_bias[i], 9);
        }

        // find lidar residual
        TicToc t_lidar;
        time_feature = 0;
        vector<vector<sfsf>> sfsf_list_window;
        for (int i = 0; i < (int)sf_body_list_window.size(); i++)
        {
            TicToc t_feature;
            vector<sfsf> sfsf_list;
            buildResidualList(voxel_map, max_voxel_size, state_window[i], sf_body_list_window[i], sfsf_list);
            time_feature += t_feature.toc();
            for (auto sfsf : sfsf_list)
            {
                ceres::CostFunction *lidar_factor = LidarSurfelSurfelFactor::Create(sfsf.normal_scan, sfsf.center_scan, sfsf.normal_map, sfsf.center_map, 1);
                problem.AddResidualBlock(lidar_factor, loss_function, para_q[i], para_t[i]);
            }
            // cout << "--residual: " << sfsf_list.size() << " sfsf in " << sf_body_list_window[i].size()  << " surfels" << endl;
            sfsf_list_window.push_back(sfsf_list);
        }
        // cout << "--SurfelMapping: lidar residuals find time: " << t_lidar.toc() << endl;

        // add imu factor
        for (int i = 0; i < (int)pre_integration_window.size() - 1; i++)
        {
            ImuFactor *imu_factor = new ImuFactor(pre_integration_window[i+1]);
            problem.AddResidualBlock(imu_factor, NULL, para_t[i], para_q[i], para_speed_bias[i],
                                     para_t[i+1], para_q[i+1], para_speed_bias[i+1]);
        }
        
        TicToc t_optimization;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = false;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 1e-4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // cout << "--SurfelMapping: ceres solve time: " << t_solver.toc() << endl;

        time_optimization = t_optimization.toc();

        // double to eigen
        for (int i = 0; i < (int)state_window.size(); i++)
        {
            state_window[i].pos_end.x() = para_t[i][0];
            state_window[i].pos_end.y() = para_t[i][1];
            state_window[i].pos_end.z() = para_t[i][2];
            state_window[i].quat_end.w() = para_q[i][0];
            state_window[i].quat_end.x() = para_q[i][1];
            state_window[i].quat_end.y() = para_q[i][2];
            state_window[i].quat_end.z() = para_q[i][3];
            state_window[i].vel_end.x() = para_speed_bias[i][0];
            state_window[i].vel_end.y() = para_speed_bias[i][1];
            state_window[i].vel_end.z() = para_speed_bias[i][2];
            state_window[i].bias_a.x() = para_speed_bias[i][3];
            state_window[i].bias_a.y() = para_speed_bias[i][4];
            state_window[i].bias_a.z() = para_speed_bias[i][5];
            state_window[i].bias_g.x() = para_speed_bias[i][6];
            state_window[i].bias_g.y() = para_speed_bias[i][7];
            state_window[i].bias_g.z() = para_speed_bias[i][8];
            state_window[i].quat_end.normalize();
            state_window[i].cov = Eigen::Matrix<double, 15, 15>::Zero();
            // cout << "aft state_window[" << i << "]: " << state_window[i].pos_end.transpose() << endl;
        }

        // calculate covariance
        TicToc t_covariance;
        int N = state_window.size();
        Eigen::MatrixXd H_1 = Eigen::MatrixXd::Zero(15 * N, 15 * N);
        Eigen::MatrixXd cov_tmp = Eigen::MatrixXd::Zero(15 * N, 15 * N);
        // from Lidar surfel
        for (int i = 0; i < (int)state_window.size(); i++)
        {
            for (int j = 0; j < (int)sfsf_list_window[i].size(); j++)
            {
                sfsf this_sfsf = sfsf_list_window[i][j];
                // J_lid
                Eigen::Matrix<double, 2, 6> J_lid = Eigen::Matrix<double, 2, 6>::Zero();
                Eigen::Matrix3d tmp = state_window[i].quat_end.toRotationMatrix() * Utility::hat(this_sfsf.normal_map) * Utility::Jleft(state_window[i].quat_end);
                J_lid.block<1, 3>(0, 0) = this_sfsf.normal_map.transpose();
                J_lid.block<1, 3>(0, 3) = this_sfsf.center_scan.transpose() * tmp;
                J_lid.block<1, 3>(1, 3) = this_sfsf.normal_scan.transpose() * tmp;
                // add H_1
                H_1.block<6, 6>(i * 15, i * 15) += 2 * J_lid.transpose() * J_lid;
                // J_surfel
                Eigen::Matrix<double, 2, 14> J_surfel = Eigen::Matrix<double, 2, 14>::Zero();
                J_surfel.block<1, 3>(1, 0) = this_sfsf.normal_map.transpose() * state_window[i].quat_end.toRotationMatrix();
                J_surfel.block<1, 3>(0, 3) = J_surfel.block<1, 3>(1, 0);
                J_surfel.block<1, 3>(0, 7) = (state_window[i].quat_end.toRotationMatrix() * this_sfsf.center_scan + state_window[i].pos_end - this_sfsf.center_map).transpose();
                J_surfel.block<1, 3>(1, 7) = (state_window[i].quat_end.toRotationMatrix() * this_sfsf.normal_scan).transpose();
                J_surfel.block<1, 3>(0, 10) = -this_sfsf.normal_map.transpose();
                // calculate H_2
                Eigen::Matrix<double, 14, 6> H_2;
                H_2 = 2 * J_surfel.transpose() * J_lid;
                // cov_surfel
                Eigen::Matrix<double, 14, 14> cov_surfel = Eigen::Matrix<double, 14, 14>::Zero();
                cov_surfel.block<7, 7>(0, 0) = this_sfsf.surfel_scan_cov;
                cov_surfel.block<7, 7>(7, 7) = this_sfsf.surfel_map_cov;
                // propagate covariance
                cov_tmp.block<6, 6>(i * 15, i * 15) += H_2.transpose() * cov_surfel * H_2;
            }
        }
        // from IMU measurements
        for (int i = 1; i < (int)state_window.size(); i++)
        {
            double dt = pre_integration_window[i]->sum_dt;
            // J_imu
            Eigen::Matrix<double, 3, 3> F = Utility::Jleft_inv(pre_integration_window[i]->residual_q) * 
                                            pre_integration_window[i]->corrected_delta_q.toRotationMatrix().inverse();
            Eigen::Vector3d f_1 = state_window[i].pos_end - state_window[i-1].pos_end - state_window[i-1].vel_end * dt + 0.5 * pre_integration_window[i]->g_vec * dt * dt;
            Eigen::Vector3d f_2 = state_window[i].vel_end - state_window[i-1].vel_end + pre_integration_window[i]->g_vec * dt;
            Eigen::Matrix<double, 15, 30> J_imu = Eigen::Matrix<double, 15, 30>::Zero();
            J_imu.block<3, 3>(0, 0) = -state_window[i-1].quat_end.toRotationMatrix().inverse();
            J_imu.block<3, 3>(0, 3) = state_window[i-1].quat_end.toRotationMatrix().inverse() * Utility::hat(f_1) * Utility::Jleft(state_window[i-1].quat_end);
            J_imu.block<3, 3>(3, 3) = -F * state_window[i-1].quat_end.toRotationMatrix().inverse() * Utility::Jleft(state_window[i-1].quat_end);
            J_imu.block<3, 3>(6, 3) = state_window[i-1].quat_end.toRotationMatrix().inverse() * Utility::hat(f_2) * Utility::Jleft(state_window[i-1].quat_end);
            J_imu.block<3, 3>(0, 6) = J_imu.block<3, 3>(0, 0) * dt;
            J_imu.block<3, 3>(6, 6) = J_imu.block<3, 3>(0, 0);
            J_imu.block<3, 3>(0, 9) = -pre_integration_window[i]->jacobian.block<3, 3>(O_P, O_BA);
            J_imu.block<3, 3>(6, 9) = -pre_integration_window[i]->jacobian.block<3, 3>(O_V, O_BA);
            J_imu.block<3, 3>(9, 9) = -Eigen::Matrix3d::Identity();
            J_imu.block<3, 3>(0, 12) = -pre_integration_window[i]->jacobian.block<3, 3>(O_P, O_BG);
            J_imu.block<3, 3>(3, 12) = J_imu.block<3, 3>(3, 3) * pre_integration_window[i]->jacobian.block<3, 3>(O_R, O_BG);
            J_imu.block<3, 3>(6, 12) = -pre_integration_window[i]->jacobian.block<3, 3>(O_V, O_BG);
            J_imu.block<3, 3>(12, 12) = -Eigen::Matrix3d::Identity();
            J_imu.block<3, 3>(0, 15) = state_window[i-1].quat_end.toRotationMatrix().inverse();
            J_imu.block<3, 3>(3, 18) = F * state_window[i-1].quat_end.toRotationMatrix().inverse() * Utility::Jleft(state_window[i].quat_end);
            J_imu.block<3, 3>(6, 21) = J_imu.block<3, 3>(0, 15);
            J_imu.block<3, 3>(9, 24) = Eigen::Matrix3d::Identity();
            J_imu.block<3, 3>(12, 27) = Eigen::Matrix3d::Identity();
            // J_pre
            Eigen::Matrix<double, 15, 15> J_pre = Eigen::Matrix<double, 15, 15>::Zero();
            J_pre.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            J_pre.block<3, 3>(3, 3) = -F * Utility::Jleft(pre_integration_window[i]->corrected_delta_q);
            J_pre.block<3, 3>(6, 6) = -Eigen::Matrix3d::Identity();
            // add H_1
            H_1.block<30, 30>(i * 15 - 15, i * 15 - 15) += 2 * J_imu.transpose() * pre_integration_window[i]->covariance.inverse() * J_imu;
            // caculate H_3
            Eigen::Matrix<double, 15, 30> H_3;
            H_3 = 2 * J_pre.transpose() * pre_integration_window[i]->covariance.inverse() * J_imu;
            // propagate covariance
            cov_tmp.block<30, 30>(i * 15 - 15, i * 15 - 15) += H_3.transpose() * pre_integration_window[i]->covariance * H_3;
        }
        Eigen::MatrixXd H_1_inv = H_1.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd window_state_cov = H_1_inv * cov_tmp * H_1_inv.transpose();
        for (int i = 0; i < (int)state_window.size(); i++)
        {
            state_window[i].cov = window_state_cov.block<15, 15>(i * 15, i * 15);
            // cout << "state_window[" << i << "] cov:" << endl << state_window[i].cov << endl;
        }
        time_covariance = t_covariance.toc();
    }
}

void SurfelMapping::saveKeyFramesAndMap()
{
    if (!cloudKeyPoses->points.empty() && (int)state_window.size() < window_size) return;

    // check distance
    Eigen::Vector3d delta_pos;
    Eigen::Quaterniond delta_quat;
    Eigen::Vector3d delta_ypr;
    if (!cloudKeyPoses->points.empty())
    {
        int frame_id = cloudKeyPoses->back().intensity;
        delta_pos = state_window.front().pos_end - statesKeyFrames.at(frame_id).pos_end;
        delta_quat = statesKeyFrames.at(frame_id).quat_end.inverse() * state_window.front().quat_end;
        delta_ypr = Utility::R2ypr(delta_quat.toRotationMatrix());
        if (abs(delta_ypr[0]) < keyframeAddingAngle &&
            abs(delta_ypr[1]) < keyframeAddingAngle &&
            abs(delta_ypr[2]) < keyframeAddingAngle &&
            delta_pos.norm() < keyframeAddingDistance)
        {
            return;
        }
    }

    if (keyFrameNum == 0)
    {
        StatesGroup first_state;
        first_state.bias_g = curr_state.bias_g;
        first_state.time = curr_state.time;
        curr_state = first_state;
        state_window.clear();
        state_window.push_back(curr_state);
    }

    PointType thisPose;
    thisPose.x = state_window.front().pos_end.x();
    thisPose.y = state_window.front().pos_end.y();
    thisPose.z = state_window.front().pos_end.z();
    thisPose.intensity = keyFrameNum;

    // if (cloudKeyPoses->points.empty())
    // {
    //     travel_distance[keyFrameNum] = 0;
    // }
    // else
    // {
    //     float delta_distance = delta_pos.norm();
    //     float dis_temp = travel_distance.rbegin()->second + delta_distance;
    //     travel_distance[keyFrameNum] = dis_temp;
    // }

    TicToc t_update;
    updateVoxelMap(sf_body_list_window.front(), state_window.front(), keyFrameNum, 
                   voxelRebuildDeltaFrameNum, max_layer, layer_voxel_size, voxel_map);
    // cout << "--SurfelMapping: update time: " << t_update.toc() << endl;

    cloudKeyPoses->points.push_back(thisPose);
    statesKeyFrames[keyFrameNum] = state_window.front();
    // cloudKeyFrames[keyFrameNum] = cloud_window.front();
    // surfelsKeyFrames[keyFrameNum] = sf_body_list_window.front();

    keyFrameNum++;
}

void SurfelMapping::publishOdometryAndPath()
{
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "lidar_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time().fromSec(state_window.front().time);
    odomAftMapped.pose.pose.orientation.x = state_window.front().quat_end.x();
    odomAftMapped.pose.pose.orientation.y = state_window.front().quat_end.y();
    odomAftMapped.pose.pose.orientation.z = state_window.front().quat_end.z();
    odomAftMapped.pose.pose.orientation.w = state_window.front().quat_end.w();
    odomAftMapped.pose.pose.position.x = state_window.front().pos_end.x();
    odomAftMapped.pose.pose.position.y = state_window.front().pos_end.y();
    odomAftMapped.pose.pose.position.z = state_window.front().pos_end.z();
    pub_odom_aftmapped.publish(odomAftMapped);

    geometry_msgs::PoseStamped poseAftMapped;
    poseAftMapped.header = odomAftMapped.header;
    poseAftMapped.pose = odomAftMapped.pose.pose;
    pathAftMapped.header.stamp = odomAftMapped.header.stamp;
    pathAftMapped.header.frame_id = "lidar_init";
    pathAftMapped.poses.push_back(poseAftMapped);
    pub_path_aftmapped.publish(pathAftMapped);

    laserOdometryTrans.stamp_ = ros::Time().fromSec(state_window.front().time);
    laserOdometryTrans.setRotation(tf::Quaternion(state_window.front().quat_end.x(), state_window.front().quat_end.y(), state_window.front().quat_end.z(), state_window.front().quat_end.w()));
    laserOdometryTrans.setOrigin(tf::Vector3(state_window.front().pos_end.x(), state_window.front().pos_end.y(), state_window.front().pos_end.z()));
    tfBroadcaster.sendTransform(laserOdometryTrans);

    f_covariance << std::fixed << std::setprecision(9) << state_window.front().time << " "
                 << state_window.front().cov(0, 0) << " " << state_window.front().cov(1, 1) << " " << state_window.front().cov(2, 2) << " "
                 << state_window.front().cov(3, 3) << " " << state_window.front().cov(4, 4) << " " << state_window.front().cov(5, 5) << std::endl;
                //  << state_window.front().cov(3, 3) << " " << state_window.front().cov(4, 4) << " " << state_window.front().cov(5, 5) << " " << state_window.front().cov(5, 5) << std::endl;
    
    f_pose_tum << std::fixed << std::setprecision(9) << state_window.front().time << " " << odomAftMapped.pose.pose.position.x << " " << odomAftMapped.pose.pose.position.y << " " 
               << odomAftMapped.pose.pose.position.z << " " << odomAftMapped.pose.pose.orientation.x << " " << odomAftMapped.pose.pose.orientation.y << " "
               << odomAftMapped.pose.pose.orientation.z << " " << odomAftMapped.pose.pose.orientation.w << std::endl;
}

void SurfelMapping::pubVoxelMap(const std::unordered_map<VOXEL_LOC, shared_ptr<SurfelRootVoxel>> &voxel_map, 
                                const int pub_max_voxel_layer, const ros::Publisher &voxel_surfel_pub)
{
    // if (voxel_surfel_pub.getNumSubscribers() <= 0) return;
    vector<shared_ptr<Surfel>> pub_surfel_list;
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
    {
        GetVoxelMapSurfel(iter->second, pub_max_voxel_layer, pub_surfel_list);
    }
    pubVoxelSurfel(pub_surfel_list, voxel_surfel_pub);
    cout << "map size: " << pub_surfel_list.size() << endl;
}

void SurfelMapping::fromMultiSurfelsMsg(const vps_slam::MultiSurfels &msg, vector<shared_ptr<Surfel>> &sf_list)
{
    for (int i = 0; i < (int)msg.surfels.size(); i++)
    {
        shared_ptr<Surfel> ps_ptr(new Surfel);
        ps_ptr->normal.x() = msg.surfels[i].normal.x;
        ps_ptr->normal.y() = msg.surfels[i].normal.y;
        ps_ptr->normal.z() = msg.surfels[i].normal.z;
        ps_ptr->center.x() = msg.surfels[i].center.x;
        ps_ptr->center.y() = msg.surfels[i].center.y;
        ps_ptr->center.z() = msg.surfels[i].center.z;
        ps_ptr->radius = msg.surfels[i].radius;
        ps_ptr->is_surfel = true;
        for (int j = 0; j < 49; j++)
        {
            ps_ptr->surfel_cov(j) = msg.surfels[i].surfel_cov[j];
        }
        sf_list.push_back(ps_ptr);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SurfelMapping");
    SurfelMapping SM;
    ROS_INFO("\033[1;32m---->\033[0m SurfelMapping Started.");
    ros::spin();
    return 0;
}