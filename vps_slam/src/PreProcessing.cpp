#include "vps_slam/utility.h"

class PreProcessing
{
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_cloud;
    ros::Publisher pub_cloud;

    std_msgs::Header cloudHeader;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr laserCloudFull;

    double startOrientation, endOrientation, orientationDiff;

    pcl::UniformSampling<PointType> UniformSamplingFilter;

    // topic
    string lidar_topic = "/velodyne_points";

public:
    PreProcessing();
    ~PreProcessing();
    void allocateMemory();
    void resetParameters();
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& msg);
    void findStartEndAngle();
    void projectPointCloud();
    void publishCloud();
};

PreProcessing::PreProcessing()
{
    // topic
    nh.param<string>("topic/lidar_topic", lidar_topic, "/velodyne_points");

    sub_cloud = nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic, 100, &PreProcessing::cloudHandler, this);
    pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_full", 100);

    allocateMemory();
    resetParameters();
}

PreProcessing::~PreProcessing()
{
    cout << "--PreProcessing: exit!!!" << endl;
}

void PreProcessing::allocateMemory()
{
    laserCloudIn.reset(new pcl::PointCloud<PointType>());
    laserCloudFull.reset(new pcl::PointCloud<PointType>());

    UniformSamplingFilter.setRadiusSearch(0.1);
}

void PreProcessing::resetParameters()
{
    laserCloudIn->clear();
    laserCloudFull->clear();
}

void PreProcessing::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    TicToc t_whole;

    // 1. Convert ros message to pcl point cloud
    copyPointCloud(msg);
    // 2. Start and end angle of a scan
    findStartEndAngle();
    // 3. Range image projection
    projectPointCloud();
    // 4. Publish all clouds
    publishCloud();
    // 5. Reset parameters for next iteration
    resetParameters();

    cout << "--PreProcessing: whole time: " << t_whole.toc() << endl << endl;
}

void PreProcessing::copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    cloudHeader = msg->header;
    pcl::fromROSMsg(*msg, *laserCloudIn);
    // Remove Nan points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
}

void PreProcessing::findStartEndAngle()
{
    // start and end orientation of this cloud
    startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
    endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                            laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
    if (endOrientation - startOrientation > 3 * M_PI)
        endOrientation -= 2 * M_PI;
    else if (endOrientation - startOrientation < M_PI)
        endOrientation += 2 * M_PI;
    orientationDiff = endOrientation - startOrientation;
}

void PreProcessing::projectPointCloud()
{
    // range image projection
    int cloudSize;
    float horizonAngle, relTime, range;
    PointType thisPoint;
    bool halfPassed = false;

    cloudSize = laserCloudIn->points.size();

    for (int i = 0; i < cloudSize; i++)
    {
        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;

        horizonAngle = -atan2(thisPoint.y, thisPoint.x);
        if (!halfPassed)
        {
            if (horizonAngle < startOrientation - M_PI / 2)
                horizonAngle += 2 * M_PI;
            else if (horizonAngle > startOrientation + M_PI * 3 / 2)
                horizonAngle -= 2 * M_PI;

            if (horizonAngle - startOrientation > M_PI)
                halfPassed = true;
        }
        else
        {
            horizonAngle += 2 * M_PI;
            if (horizonAngle < endOrientation - M_PI * 3 / 2)
                horizonAngle += 2 * M_PI;
            else if (horizonAngle > endOrientation + M_PI / 2)
                horizonAngle -= 2 * M_PI;
        }
        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        if (range < 1.0) continue;
        // -0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
        relTime = (horizonAngle - startOrientation) / orientationDiff;
        thisPoint.intensity = relTime;
        laserCloudFull->points.push_back(thisPoint);
    }
}

void PreProcessing::publishCloud()
{
    pcl::PointCloud<PointType>::Ptr laserCloudFullDS(new pcl::PointCloud<PointType>());
    UniformSamplingFilter.setInputCloud(laserCloudFull);
    UniformSamplingFilter.filter(*laserCloudFullDS);

    cout << laserCloudFull->size() << " --> " << laserCloudFullDS->size() << endl;

    sensor_msgs::PointCloud2 laserCloudTemp;
    pcl::toROSMsg(*laserCloudFullDS, laserCloudTemp);
    laserCloudTemp.header.stamp = cloudHeader.stamp;
    laserCloudTemp.header.frame_id = "lidar_init";
    pub_cloud.publish(laserCloudTemp);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "PreProcessing");
    PreProcessing PP;
    ROS_INFO("\033[1;32m---->\033[0m PreProcessing Started.");
    ros::spin();
    return 0;
}