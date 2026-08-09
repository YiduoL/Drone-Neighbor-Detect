#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE {
    AVIA = 1, VELO16, OUST64, HESAIxt32, UNILIDAR
}; //{1, 2, 3, 4}
enum TIME_UNIT {
    SEC = 0, MS = 1, US = 2, NS = 3
};
enum Feature {
    Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint
};
enum Surround {
    Prev, Next
};
enum E_jump {
    Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind
};

const bool time_list_cut_frame(PointType &x, PointType &y);

struct orgtype {
    double range;
    double dista;
    double angle[2];
    double intersect;
    E_jump edj[2];
    Feature ftype;

    orgtype() {
        range = 0;
        edj[Prev] = Nr_nor;
        edj[Next] = Nr_nor;
        ftype = Nor;
        intersect = 2;
    }
};

namespace velodyne_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        uint16_t ring;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, intensity, intensity)
                                          (float, time, time)
                                          (std::uint16_t, ring, ring)
)

/**
 * @brief Unilidar Point Type
 */
namespace unilidar_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D
        PCL_ADD_INTENSITY
        std::uint16_t ring;
        float time;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace unilidar_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(unilidar_ros::Point,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, intensity, intensity)
                                    (std::uint16_t, ring, ring)
                                    (float, time, time)
                                    )

namespace hesai_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        double timestamp;
        uint16_t ring;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, intensity, intensity)
                                          (double, timestamp, timestamp)
                                          (std::uint16_t, ring, ring)
)

namespace ouster_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, intensity, intensity)
                                          // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                          (std::uint32_t, t, t)
                                          (std::uint16_t, reflectivity, reflectivity)
                                          (std::uint8_t, ring, ring)
                                          (std::uint16_t, ambient, ambient)
                                          (std::uint32_t, range, range)
)

class Preprocess {
public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Preprocess();

    ~Preprocess();

    void process(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    void process(const sensor_msgs::msg::PointCloud2::SharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    void set(bool feat_en, int lid_type, double bld, int pfilt_num);

    // sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud;
    PointCloudXYZI pl_full, pl_corn, pl_surf;
    // Near-field (<= nearfield_near_range), zero-downsample, bypasses point_filter_num.
    // Populated only by avia_handler. Does NOT participate in EKF update unless
    // nearfield_join_update is set. See laserMapping.cpp flush_nearfield_upto().
    PointCloudXYZI pl_nearfield;
    PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
    vector<orgtype> typess[128]; //maximum 128 line lidar
    float time_unit_scale;
    int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
    double blind;
    bool given_offset_time;
    //ros::Publisher pub_full, pub_surf, pub_corn;

    bool nearfield_enable = false;
    double nearfield_near_range = 3.5;
    double nearfield_blind_override = 0.1;
    bool nearfield_join_update = false;
    // Near-field membership is a CYLINDER (horizontal radius nearfield_near_range,
    // vertical half-height nearfield_z_half_height), not a sphere -- a sphere pulls in
    // ceiling/floor structure whenever the platform gets close enough vertically, even
    // though a neighbor drone is never going to be >1m above/below in the same near-field
    // horizontal radius. This check is in the sensor's own (near-level-flight) body frame,
    // not world/gravity frame -- cheap and matches this function's existing per-point
    // body-frame math, but will under/over-cover during aggressive roll/pitch.
    double nearfield_z_half_height = 1.0;

    // Far-field points are decimated at capture time (before EKF processing, not just
    // before publishing) so Jetson doesn't pay compute for points that get thrown away
    // anyway. Budget is shared with the near field so total output stays near
    // far_field_sampling_target_total regardless of how many near-field points this
    // frame has (near-field count varies a lot frame to frame). Within that budget,
    // far-field points are weighted-sampled favoring closer range over farther --
    // large distant flat surfaces (ceiling, far walls) subtend more solid angle and so
    // naturally dominate a frame's point budget under uniform sampling; range weighting
    // suppresses that without needing to know what a "ceiling" is (unlike a height-based
    // filter, this needs no attitude/gravity-frame assumption, so it doesn't degrade
    // during non-level flight the way the near-field cylinder's body-frame z gate can).
    bool far_field_sampling_enable = false;
    int far_field_sampling_target_total = 5000;


private:
    void avia_handler(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg);

    void oust64_handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    void velodyne_handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    void unilidar_handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    void hesai_handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);

    void pub_func(PointCloudXYZI &pl, const rclcpp::Time &ct);

    int
    plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);

    bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex,
                     Eigen::Vector3d &curr_direct);

    bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);

    int group_size;
    double disA, disB, inf_bound;
    double limit_maxmid, limit_midmin, limit_maxmin;
    double p2l_ratio;
    double jump_up_limit, jump_down_limit;
    double cos160;
    double edgea, edgeb;
    double smallp_intersect, smallp_ratio;
    double vx, vy, vz;
};
