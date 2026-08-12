#!/usr/bin/env python3
"""Live (real ROS2 subscriber, NOT offline bag-read) causal DUFOMap detection timing.

Why this exists instead of just running causal_vs_batch.py on Jetson: that script reads
bags via rosbag2_py's SequentialReader, but ROS2 Foxy never packaged rosbag2_py at all
(only Galactic+ did -- confirmed via `apt-cache search rosbag2` on the Jetson, which
lists ros-galactic-rosbag2-py / ros-rolling-rosbag2-py but no ros-foxy-rosbag2-py).
Building rosbag2_py from source for Foxy would be its own multi-hour undertaking
(similar scope to detection/dufomap_custom's from-source build), so instead: subscribe
to Point-LIO's live topics directly with rclpy + message_filters (both ARE available on
Foxy), and feed frames via `ros2 bag play <bag>` (a plain C++ CLI tool -- no rosbag2_py
needed to play a bag, only to read one back in Python). This is arguably more
representative of the real deployed pipeline anyway (a live subscriber consuming
published topics, not an offline batch read).

Mirrors causal_vs_batch.py's run_causal() exactly (segment-before-run, history-only,
same near/far range filtering, same CSV schema) so results are directly comparable.
Frame 0 has no history -- labeled all-static by convention, same as the offline script.

Usage:
  # terminal 1:
  python3 causal_live.py <out_dir> [--max-frames N] [--max-seconds S]
  # terminal 2, once terminal 1 prints "waiting for frames...":
  ros2 bag play <bag_path> --topics /nearfield/deskewed_world /cloud_registered /aft_mapped_to_init
Stops on --max-frames / --max-seconds, or Ctrl-C (writes partial results either way).
"""
import argparse
import csv
import json
import os
import signal
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2

from dufomap import dufomap

MIN_RANGE = 0.1
DEFAULT_MAX_RANGE = 50.0
FRAME_BUDGET_MS = 90.0  # ~1/10.5 Hz Mid360 frame rate, same budget causal_vs_batch.py uses


def cloud_to_xyz(msg):
    pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # sensor_msgs_py's read_points() return type differs across ROS2 distro versions:
    # some yield numpy structured scalars (so np.array(list(...)) keeps field names,
    # and pts["x"] works), this Jetson's older ros-foxy-sensor-msgs-py (2.0.5) yields
    # plain tuples instead, so np.array(list(...)) already collapses to a plain (N,3)
    # float array with no field names -- indexing by name then raises IndexError.
    # Handle both.
    if pts.dtype.names is not None:
        return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
    return pts.astype(np.float32)


class CausalLiveNode(Node):
    def __init__(self, args):
        super().__init__("causal_live_detect")
        self.args = args
        self.dm = dufomap(args.voxel, args.d_s, args.d_p, num_threads=0)
        self.rows = []
        self.frame_idx = 0
        self.done = False
        # Per-point (not just per-frame-count) near-field records, only kept when
        # --save-points is set -- lets a later comparison match individual points by
        # world-frame position (not by array index, which two independent live runs
        # of the same bag aren't guaranteed to preserve) and compute real
        # precision/recall of one run's labels against another's, not just whether
        # the aggregate per-frame counts happen to agree.
        self.point_records = [] if args.save_points else None

        # RELIABLE to match Point-LIO's own publishers (default QoS is reliable for
        # these topics in this fork) -- a mismatched QoS (e.g. defaulting to
        # best-effort here) would silently drop every message with no error.
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        near_sub = message_filters.Subscriber(self, PointCloud2, args.near_topic, qos_profile=qos)
        far_sub = message_filters.Subscriber(self, PointCloud2, args.far_topic, qos_profile=qos)
        odom_sub = message_filters.Subscriber(self, Odometry, args.odom_topic, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [near_sub, far_sub, odom_sub], queue_size=30, slop=args.slop)
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(
            f"subscribed near={args.near_topic} far={args.far_topic} odom={args.odom_topic}, "
            f"waiting for frames... (play a bag with `ros2 bag play <bag>` now)")

    def on_frame(self, near_msg, far_msg, odom_msg):
        if self.done:
            return
        stamp = near_msg.header.stamp.sec + near_msg.header.stamp.nanosec / 1e9

        near_pts = cloud_to_xyz(near_msg)
        far_pts = cloud_to_xyz(far_msg)
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        pose = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        pose_xyz = np.array(pose[:3], dtype=np.float32)

        dn = np.linalg.norm(near_pts - pose_xyz, axis=1) if len(near_pts) else np.zeros(0)
        near_pts = near_pts[(dn > MIN_RANGE) & (dn < self.args.max_range)]
        df = np.linalg.norm(far_pts - pose_xyz, axis=1) if len(far_pts) else np.zeros(0)
        far_pts = far_pts[(df > MIN_RANGE) & (df < self.args.max_range)]

        i = self.frame_idx
        t0 = time.time()
        if i == 0:
            near_labels = np.zeros(len(near_pts), dtype=np.uint8)
            far_labels = np.zeros(len(far_pts), dtype=np.uint8)
        else:
            near_labels = self.dm.segment(near_pts, pose, cloud_transform=False) if len(near_pts) else np.zeros(0, dtype=np.uint8)
            far_labels = self.dm.segment(far_pts, pose, cloud_transform=False) if len(far_pts) else np.zeros(0, dtype=np.uint8)
        t_seg = time.time()
        self.dm.run(far_pts, pose, cloud_transform=False)
        t_run = time.time()

        self.rows.append({
            "frame_idx": i, "t": stamp,
            "n_near": len(near_pts), "n_far": len(far_pts),
            "n_dynamic_near": int(near_labels.sum()), "n_dynamic_far": int(far_labels.sum()),
            "t_segment_ms": (t_seg - t0) * 1000, "t_run_ms": (t_run - t_seg) * 1000,
        })
        if self.point_records is not None and len(near_pts):
            self.point_records.append((stamp, near_pts, near_labels))
        if i % 50 == 0:
            self.get_logger().info(
                f"frame {i}: near={len(near_pts)} far={len(far_pts)} "
                f"dyn_near={int(near_labels.sum())} "
                f"seg={((t_seg - t0) * 1000):.2f}ms run={((t_run - t_seg) * 1000):.2f}ms")
        self.frame_idx += 1

        if self.args.max_frames and self.frame_idx >= self.args.max_frames:
            self.done = True
        if self.args.max_seconds and self.rows and \
                (self.rows[-1]["t"] - self.rows[0]["t"]) >= self.args.max_seconds:
            self.done = True

    def write_results(self):
        if not self.rows:
            self.get_logger().warn("no frames received -- nothing to write "
                                    "(check topic names / that a bag was played)")
            return
        os.makedirs(self.args.out_dir, exist_ok=True)
        fieldnames = ["frame_idx", "t", "n_near", "n_far", "n_dynamic_near",
                      "n_dynamic_far", "t_segment_ms", "t_run_ms"]
        csv_path = os.path.join(self.args.out_dir, "causal_live_per_frame.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)
        self.get_logger().info(f"wrote {csv_path} ({len(self.rows)} frames)")

        t_seg = np.array([r["t_segment_ms"] for r in self.rows[1:]])  # skip frame 0
        t_run = np.array([r["t_run_ms"] for r in self.rows])
        t_total = t_seg + t_run[1:]
        cn = np.array([r["n_dynamic_near"] for r in self.rows])

        summary = {
            "n_frames": len(self.rows),
            "params": {"voxel": self.args.voxel, "d_s": self.args.d_s, "d_p": self.args.d_p,
                       "max_range": self.args.max_range},
            "near_dynamic_mean": float(cn.mean()), "near_dynamic_median": float(np.median(cn)),
            "timing_ms": {
                "segment_mean": float(t_seg.mean()), "segment_p95": float(np.percentile(t_seg, 95)),
                "segment_p99": float(np.percentile(t_seg, 99)),
                "run_mean": float(t_run.mean()), "run_p95": float(np.percentile(t_run, 95)),
                "run_p99": float(np.percentile(t_run, 99)),
                "total_mean": float(t_total.mean()), "total_p95": float(np.percentile(t_total, 95)),
                "total_p99": float(np.percentile(t_total, 99)),
                "frac_over_budget_pct": float((t_total > FRAME_BUDGET_MS).mean() * 100),
            },
        }
        summary_path = os.path.join(self.args.out_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f"wrote {summary_path}")
        self.get_logger().info(
            f"LIVE detection timing (real Jetson, real subscriber): "
            f"segment mean={t_seg.mean():.2f}ms p99={np.percentile(t_seg, 99):.2f}ms | "
            f"run mean={t_run.mean():.2f}ms p99={np.percentile(t_run, 99):.2f}ms | "
            f"total mean={t_total.mean():.2f}ms p99={np.percentile(t_total, 99):.2f}ms "
            f"(budget={FRAME_BUDGET_MS}ms, {summary['timing_ms']['frac_over_budget_pct']:.1f}% over)")

        if self.point_records is not None:
            # Flat arrays + per-frame (start,count) index, keyed by this frame's
            # timestamp -- lets a comparison script match frames across two
            # independent runs by nearest timestamp (not by array index, which
            # doesn't survive two separately-executed live runs) and then match
            # individual points within a frame by world-frame xyz proximity.
            all_t, all_xyz, all_label, frame_t, frame_start, frame_count = [], [], [], [], [], []
            offset = 0
            for stamp, pts, labels in self.point_records:
                frame_t.append(stamp)
                frame_start.append(offset)
                frame_count.append(len(pts))
                all_xyz.append(pts)
                all_label.append(labels)
                offset += len(pts)
            npz_path = os.path.join(self.args.out_dir, "near_points.npz")
            np.savez_compressed(
                npz_path,
                xyz=np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3), dtype=np.float32),
                label=np.concatenate(all_label, axis=0) if all_label else np.zeros((0,), dtype=np.uint8),
                frame_t=np.array(frame_t, dtype=np.float64),
                frame_start=np.array(frame_start, dtype=np.int64),
                frame_count=np.array(frame_count, dtype=np.int64),
            )
            self.get_logger().info(f"wrote {npz_path} ({offset} near-field points across "
                                    f"{len(self.point_records)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--near-topic", default="/nearfield/deskewed_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE)
    ap.add_argument("--save-points", action="store_true",
                     help="also save per-point near-field xyz+label to near_points.npz, "
                          "for cross-run point-level precision/recall comparison")
    ap.add_argument("--slop", type=float, default=0.05,
                     help="max timestamp difference (s) allowed when syncing near/far/odom")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = unbounded (Ctrl-C to stop)")
    ap.add_argument("--max-seconds", type=float, default=0, help="0 = unbounded (Ctrl-C to stop)")
    args = ap.parse_args()

    rclpy.init()
    node = CausalLiveNode(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C received, writing partial results...")
    finally:
        node.write_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
