#!/usr/bin/env python3
"""Debug: subscribe to /nearfield/deskewed_world + /aft_mapped_to_init, print the
near-field point cloud's centroid (in WORLD frame) alongside ego position for N
consecutive frames, to see whether the near-field content is moving relative to ego
(a real dynamic object) or staying fixed (static structure -- which DUFOMap should
correctly never flag as dynamic)."""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2


def cloud_to_xyz(msg):
    pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.dtype.names is not None:
        return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
    return pts.astype(np.float32)


class Dumper(Node):
    def __init__(self, n_frames):
        super().__init__("debug_cloud_dump2")
        self.n_frames = n_frames
        self.count = 0
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        near_sub = message_filters.Subscriber(self, PointCloud2, "/nearfield/deskewed_world", qos_profile=qos)
        odom_sub = message_filters.Subscriber(self, Odometry, "/aft_mapped_to_init", qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([near_sub, odom_sub], queue_size=30, slop=0.05)
        self.sync.registerCallback(self.cb)
        self.done = False

    def cb(self, near_msg, odom_msg):
        near_pts = cloud_to_xyz(near_msg)
        p = odom_msg.pose.pose.position
        stamp = near_msg.header.stamp.sec + near_msg.header.stamp.nanosec / 1e9
        if len(near_pts) > 0:
            centroid = near_pts.mean(axis=0)
            dist_from_ego = np.linalg.norm(centroid - np.array([p.x, p.y, p.z]))
            print(f"t={stamp:.3f} n={len(near_pts):5d} ego=({p.x:.2f},{p.y:.2f},{p.z:.2f}) "
                  f"near_centroid_world=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) "
                  f"dist_from_ego={dist_from_ego:.2f}", flush=True)
        else:
            print(f"t={stamp:.3f} n=0 ego=({p.x:.2f},{p.y:.2f},{p.z:.2f})", flush=True)
        self.count += 1
        if self.count >= self.n_frames:
            self.done = True


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rclpy.init()
    node = Dumper(n)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
