from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare("point_lio"), "config", "mid360.yaml"]
    )

    point_lio = Node(
        package="point_lio",
        executable="pointlio_mapping",
        name="laserMapping",
        output="screen",
        parameters=[
            config,
            {
                # Calibration used for mutual_avoidance_uav1.
                "mapping.extrinsic_T": [-0.019391, -0.000278, 0.080926],
                "mapping.extrinsic_R": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                "preprocess.blind": 0.05,
                "use_imu_as_input": False,
                "prop_at_freq_of_imu": True,
                "check_satu": True,
                "init_map_size": 10,
                "point_filter_num": 3,
                "space_down_sample": True,
                "filter_size_surf": 0.5,
                "filter_size_map": 0.5,
                "cube_side_length": 1000.0,
                "runtime_pos_log_enable": False,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("point_lio"), "rviz_cfg", "loam_livox.rviz"]
            ),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            point_lio,
            rviz,
        ]
    )
