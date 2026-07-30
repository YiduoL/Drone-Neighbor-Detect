#!/usr/bin/env python3
"""Diagnostic that motivated the cylindrical near-field gate (see PIPELINE.md):
exclude near-field points above a height cutoff (in the shared reference frame, where a
ceiling/overhead structure was found to dominate false positives) BEFORE they're even
counted as detection candidates. No voxel gating, no clustering, no tracking -- just
measures how much of the precision problem this one cut alone accounts for, and whether
it costs any recall (it should not, if the cutoff is chosen correctly: a real target
should never be found above the cutoff height).

Usage: python3 zcut_diagnostic.py --host swarm1 [--z-max 4.0] [--resolution 0.15]
"""
import argparse
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import scipy.spatial
import open3d as o3d

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
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    t0 = None
    while r.has_next():
        _, data, t = r.read_next()
        if t0 is None:
            t0 = t
        msg = deserialize_message(data, PointCloud2)
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pts = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
        out.append(((t - t0) / 1e9, pts))
    return out


def read_pose_topic(bag_path, topic):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while r.has_next():
        _, data, t = r.read_next()
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        out.append([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"])
    ap.add_argument("--near-topic", default="/nearfield/refined_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--z-max", type=float, default=4.0,
                    help="drop near-field points above this z (shared/reference frame) before classification")
    args = ap.parse_args()

    host_name = args.host
    other_name = "swarm2" if host_name == "swarm1" else "swarm1"
    bag_path = C2_BAGS[host_name]
    out_dir = f"{BASE}/{host_name}_bgsub"

    print(f"=== recovering T_far for {host_name} via Kabsch ===")
    bg = np.load(f"{out_dir}/bgsubtract_result.npz")
    traj_aligned_full = bg["traj_aligned"]
    orig_poses_full = read_pose_topic(bag_path, args.odom_topic)
    orig_traj_full = np.array([p[:3] for p in orig_poses_full])
    n_traj = min(len(orig_traj_full), len(traj_aligned_full))
    T_far = kabsch_transform(orig_traj_full[:n_traj], traj_aligned_full[:n_traj])
    R, t = T_far[:3, :3], T_far[:3, 3]

    ref_pcd = o3d.io.read_point_cloud(REFERENCE_MAP)
    ref_tree = scipy.spatial.cKDTree(np.asarray(ref_pcd.points))
    other_traj_aligned = np.load(f"{BASE}/{other_name}_bgsub/bgsubtract_result.npz")["traj_aligned"]
    other_traj_tree = scipy.spatial.cKDTree(other_traj_aligned)

    print(f"\n=== loading frames from {bag_path} ===")
    near = read_cloud_topic(bag_path, args.near_topic)
    far = read_cloud_topic(bag_path, args.far_topic)
    poses = orig_poses_full
    n = min(len(near), len(far), len(poses))
    print(f"total frames: {n}, z_max cut = {args.z_max}m (shared frame)")

    dm = dufomap(args.resolution, args.d_s, args.d_p, num_threads=0)

    raw_tp = raw_fn = raw_fp = 0
    cut_tp = cut_fn = cut_fp = 0
    n_dropped_by_cut = 0
    n_dropped_gt = 0  # how many REAL gt points would we lose by cutting z>4?

    for i in range(n):
        t_rel, near_pts = near[i]
        _, far_pts = far[i]
        pose = poses[i]
        pose_xyz = np.array(pose[:3], dtype=np.float32)

        dn = np.linalg.norm(near_pts - pose_xyz, axis=1)
        near_mask = (dn > MIN_RANGE) & (dn < MAX_RANGE)
        near_pts_f = near_pts[near_mask]
        df = np.linalg.norm(far_pts - pose_xyz, axis=1)
        far_mask = (df > MIN_RANGE) & (df < MAX_RANGE)
        far_pts_f = far_pts[far_mask]

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

        raw_tp += int((is_gt & is_det).sum())
        raw_fn += int((is_gt & ~is_det).sum())
        raw_fp += int((~is_fg & is_det).sum())

        below_cut = aligned[:, 2] <= args.z_max
        n_dropped_by_cut += int((~below_cut).sum())
        n_dropped_gt += int((is_gt & ~below_cut).sum())

        is_det_cut = is_det & below_cut
        cut_tp += int((is_gt & is_det_cut).sum())
        cut_fn += int((is_gt & ~is_det_cut).sum())
        cut_fp += int((~is_fg & is_det_cut).sum())

        if i % 500 == 0:
            print(f"  frame {i}/{n} t={t_rel:.1f}s raw_fp={raw_fp} cut_fp={cut_fp}")

    print(f"\ntotal near-field points dropped by z>{args.z_max}m cut: {n_dropped_by_cut}")
    print(f"of which were real GT (other-drone) points lost: {n_dropped_gt}")

    raw_recall = raw_tp / max(raw_tp + raw_fn, 1)
    raw_precision = raw_tp / max(raw_tp + raw_fp, 1)
    print(f"\n=== RAW (no z cut) ===")
    print(f"recall={raw_recall:.3f} precision={raw_precision:.3f} tp={raw_tp} fn={raw_fn} fp={raw_fp}")

    cut_recall = cut_tp / max(cut_tp + cut_fn, 1)
    cut_precision = cut_tp / max(cut_tp + cut_fp, 1)
    print(f"\n=== WITH z>{args.z_max}m CUT ===")
    print(f"recall={cut_recall:.3f} precision={cut_precision:.3f} tp={cut_tp} fn={cut_fn} fp={cut_fp}")


if __name__ == "__main__":
    main()
