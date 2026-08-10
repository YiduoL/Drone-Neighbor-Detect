#!/usr/bin/env python3
"""Test: does the causal detector specifically fail to detect the OTHER drone while it's
HOVERING (a known theoretical weakness of occupancy/ray-casting dynamic detectors -- an
object that stops moving eventually looks statistically like real static structure)?

Self-contained: runs the causal detector itself (same method as visualize_confusion.py --
segment() using only past-frame history, then run() to integrate the current frame) over
the host flight, rather than depending on a previously-saved evaluation run. Uses the
same default parameters as the rest of this repo's final configuration (resolution=0.15,
d_s=0.2, d_p=2, C2-refined near-field input) so results are directly comparable to the
numbers reported in PIPELINE.md.

No cross-bag time sync needed (none is available -- see background_subtraction.py's
docstring) -- this works purely spatially: the other flight's own trajectory carries its
own relative-time, so its own speed at every point on its trajectory can be computed and
classified as "hovering" or "moving" independent of the host flight. Each of the host's
pseudo-GT label==2 points ("this is the other drone", see build_pseudo_labels.py) is then
matched to the nearest point on the other trajectory and tagged hovering/moving, and
causal-detector recall is compared between the two groups.

Usage: python3 hovering_hypothesis_test.py --host swarm1 [--hover-speed 0.15]
"""
import argparse
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import scipy.spatial

from dufomap import dufomap

BASE = os.environ.get("DETECTION_DATA_DIR", "./data")
C2_BAGS = {
    "swarm1": os.environ.get("SWARM1_C2_BAG", "./data/pointlio_lidar_1_c2_output"),
    "swarm2": os.environ.get("SWARM2_C2_BAG", "./data/pointlio_lidar_2_c2_output"),
}
REFERENCE_MAP = os.environ.get("REFERENCE_MAP", "./data/reference_map.pcd")
MIN_RANGE, MAX_RANGE = 0.1, 50.0
BG_THRESHOLD = 0.12
OTHER_DRONE_THRESHOLD = 0.35


def read_cloud_topic(bag_path, topic):
    # Use each message's own header.stamp, not the bag's recorded receive time -- a bag
    # recorded via a forced SIGKILL (see run_pointlio.sh) into an mcap with no message
    # index can be read back out of true receive order (rosbag2 warns "attempted to read
    # in receive timestamp order with no message index" in this case), which silently
    # corrupts any time-based logic downstream. header.stamp is set by the publisher and
    # is reliably monotonic; sorting by it here is a second safety net against that.
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while r.has_next():
        _, data, _ = r.read_next()
        msg = deserialize_message(data, PointCloud2)
        hs = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pts = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
        out.append((hs, pts))
    out.sort(key=lambda item: item[0])
    t0 = out[0][0] if out else 0.0
    return [(hs - t0, pts) for hs, pts in out]


def read_pose_topic_with_time(bag_path, topic="/aft_mapped_to_init"):
    # Sorted by header.stamp, not bag receive order -- see read_cloud_topic's comment
    # elsewhere in this repo. Matters even more here: poses get matched to point clouds
    # BY ARRAY INDEX, so an out-of-order read silently misaligns pose-to-cloud pairing
    # rather than just shifting a display timestamp.
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    rows = []
    while r.has_next():
        _, data, _ = r.read_next()
        msg = deserialize_message(data, Odometry)
        hs = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rows.append((hs, [p.x, p.y, p.z], [p.x, p.y, p.z, q.w, q.x, q.y, q.z]))
    rows.sort(key=lambda item: item[0])
    t0 = rows[0][0] if rows else 0.0
    times = np.array([hs - t0 for hs, _, _ in rows])
    pos = np.array([pos for _, pos, _ in rows])
    poses7 = [pose7 for _, _, pose7 in rows]
    return times, pos, poses7


def kabsch_transform(src, dst):
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    src0, dst0 = src - src_c, dst - dst_c
    H = src0.T @ dst0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    resid = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
    print(f"Kabsch T_far residual: mean={resid.mean():.4f}m max={resid.max():.4f}m")
    return T


def run_causal_and_collect(bag_path, ref_tree, other_traj_tree, T_far,
                           near_topic, far_topic, odom_topic,
                           resolution, d_s, d_p):
    """Runs the causal detector over the whole flight, returns (tp_pts, fn_pts) in the
    shared reference frame, matching evaluate-style scripts elsewhere in this repo."""
    R, t = T_far[:3, :3], T_far[:3, 3]
    near = read_cloud_topic(bag_path, near_topic)
    far = read_cloud_topic(bag_path, far_topic)
    _, _, poses = read_pose_topic_with_time(bag_path, odom_topic)
    n = min(len(near), len(far), len(poses))

    dm = dufomap(resolution, d_s, d_p, num_threads=0)
    tp_pts, fn_pts = [], []
    for i in range(n):
        _, near_pts = near[i]
        _, far_pts = far[i]
        pose = poses[i]
        pose_xyz = np.array(pose[:3], dtype=np.float32)

        dn = np.linalg.norm(near_pts - pose_xyz, axis=1)
        near_pts_f = near_pts[(dn > MIN_RANGE) & (dn < MAX_RANGE)]
        df = np.linalg.norm(far_pts - pose_xyz, axis=1)
        far_pts_f = far_pts[(df > MIN_RANGE) & (df < MAX_RANGE)]

        if i == 0 or len(near_pts_f) == 0:
            dyn_labels = np.zeros(len(near_pts_f), dtype=np.uint8)
        else:
            dyn_labels = dm.segment(near_pts_f, pose, cloud_transform=False)
        dm.run(far_pts_f, pose, cloud_transform=False)

        if len(near_pts_f) == 0:
            continue
        aligned = (R @ near_pts_f.T).T + t
        d_ref, _ = ref_tree.query(aligned, k=1)
        is_fg = d_ref > BG_THRESHOLD
        other_dists, _ = other_traj_tree.query(aligned, k=1)
        is_gt = is_fg & (other_dists < OTHER_DRONE_THRESHOLD)
        is_det = dyn_labels.astype(bool)

        tp_mask = is_gt & is_det
        fn_mask = is_gt & ~is_det
        if tp_mask.any():
            tp_pts.append(aligned[tp_mask])
        if fn_mask.any():
            fn_pts.append(aligned[fn_mask])

        if i % 500 == 0:
            print(f"  frame {i}/{n}")

    tp_pts = np.concatenate(tp_pts, axis=0) if tp_pts else np.zeros((0, 3))
    fn_pts = np.concatenate(fn_pts, axis=0) if fn_pts else np.zeros((0, 3))
    return tp_pts, fn_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"])
    ap.add_argument("--near-topic", default="/nearfield/refined_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--hover-speed", type=float, default=0.15,
                    help="m/s below which we call the other drone 'hovering'")
    args = ap.parse_args()

    host = args.host
    other = "swarm2" if host == "swarm1" else "swarm1"
    host_bag = C2_BAGS[host]
    other_bag = C2_BAGS[other]

    print(f"=== computing {other}'s own speed profile (its own relative time, no cross-bag sync needed) ===")
    other_times, other_pos_orig, _ = read_pose_topic_with_time(other_bag)
    speed = np.zeros(len(other_pos_orig))
    dt = np.diff(other_times)
    dp = np.linalg.norm(np.diff(other_pos_orig, axis=0), axis=1)
    speed[1:] = dp / np.maximum(dt, 1e-3)
    is_hover = speed < args.hover_speed
    print(f"{other}: {len(other_pos_orig)} poses, {is_hover.sum()} ({100*is_hover.mean():.1f}%) "
          f"below {args.hover_speed} m/s")

    print(f"\n=== recovering T_far for {host} via Kabsch ===")
    bg = np.load(f"{BASE}/{host}_bgsub/bgsubtract_result.npz")
    traj_aligned_full = bg["traj_aligned"]
    _, orig_traj_full, _ = read_pose_topic_with_time(host_bag)
    n_traj = min(len(orig_traj_full), len(traj_aligned_full))
    T_far = kabsch_transform(orig_traj_full[:n_traj], traj_aligned_full[:n_traj])

    ref_pcd = o3d.io.read_point_cloud(REFERENCE_MAP)
    ref_tree = scipy.spatial.cKDTree(np.asarray(ref_pcd.points))
    other_bg = np.load(f"{BASE}/{other}_bgsub/bgsubtract_result.npz")
    other_traj_aligned = other_bg["traj_aligned"]
    n_common = min(len(other_traj_aligned), len(is_hover))
    other_traj_tree = scipy.spatial.cKDTree(other_traj_aligned[:n_common])

    print(f"\n=== running causal detector on {host} ({host_bag}) ===")
    tp_pts, fn_pts = run_causal_and_collect(
        host_bag, ref_tree, other_traj_tree, T_far,
        args.near_topic, args.far_topic, args.odom_topic,
        args.resolution, args.d_s, args.d_p)
    print(f"{host}: tp={len(tp_pts)} fn={len(fn_pts)} "
          f"overall recall={len(tp_pts)/max(len(tp_pts)+len(fn_pts),1):.3f}")

    gt_pts = np.concatenate([tp_pts, fn_pts], axis=0)
    is_tp_flat = np.concatenate([np.ones(len(tp_pts), dtype=bool),
                                 np.zeros(len(fn_pts), dtype=bool)])
    _, nearest_idx = other_traj_tree.query(gt_pts, k=1)
    gt_is_hover_match = is_hover[:n_common][nearest_idx]
    print(f"of {len(gt_pts)} GT 'other-drone' points, "
          f"{gt_is_hover_match.sum()} ({100*gt_is_hover_match.mean():.1f}%) matched "
          f"to a point on {other}'s trajectory where it was hovering (speed<{args.hover_speed}m/s)")

    def recall_for_subset(mask, label):
        n_tp = int((is_tp_flat & mask).sum())
        n_total = int(mask.sum())
        if n_total == 0:
            print(f"{label}: no GT points")
            return
        print(f"{label}: n_gt={n_total} recall={n_tp/n_total:.3f}")

    print(f"\n=== recall comparison: GT points near a HOVERING vs MOVING {other} ===")
    recall_for_subset(gt_is_hover_match, "hovering-matched GT points")
    recall_for_subset(~gt_is_hover_match, "moving-matched GT points")


if __name__ == "__main__":
    main()
