#!/usr/bin/env python3
"""Shared data loading, pseudo-GT construction, and unified metrics for the parameter
ablation scripts (ablation_resolution_dp.py, ablation_ds.py) in this directory.

cloud_transform semantics confirmed from dufomap's own source (dufomap/__init__.py
docstring): points already in world frame -> cloud_transform=False throughout.
"""
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import scipy.spatial

BASE = os.environ.get("DETECTION_DATA_DIR", "./data")  # background_subtraction.py outputs
BAGS = {
    "swarm1": os.environ.get("SWARM1_BAG", "./data/pointlio_lidar_1_output"),
    "swarm2": os.environ.get("SWARM2_BAG", "./data/pointlio_lidar_2_output"),
}
REFERENCE_MAP = os.environ.get("REFERENCE_MAP", "./data/reference_map.pcd")
MIN_RANGE, MAX_RANGE = 0.1, 50.0
BG_THRESHOLD = 0.12
OTHER_DRONE_THRESHOLD = 0.35
COLD_START_CUTOFF_S = 90.0
DIST_BUCKETS = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]


def read_cloud_topic(bag_path, topic, max_frames=None):
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
        if max_frames is not None and len(out) >= max_frames:
            break
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


def read_pose_topic(bag_path, topic):
    # Sorted by header.stamp, not bag receive order -- see read_cloud_topic's comment.
    # Matters even more here: poses get matched to point clouds BY ARRAY INDEX elsewhere
    # in this codebase, so an out-of-order read here silently misaligns pose-to-cloud
    # pairing rather than just shifting a display timestamp.
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while r.has_next():
        _, data, _ = r.read_next()
        msg = deserialize_message(data, Odometry)
        hs = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        out.append((hs, [p.x, p.y, p.z, q.w, q.x, q.y, q.z]))
    out.sort(key=lambda item: item[0])
    return [pose for _, pose in out]


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
    return T, resid.mean()


_SCENARIO_CACHE = {}


class Scenario:
    """Everything needed to run causal detection + score it against pseudo-GT for one bag.

    Process-cached by (host_name, max_frames) -- re-running many parameter combos over the
    same bag/frame-count (e.g. an ablation sweep) should only hit the disk/bag once, not
    once per combo. Poses are always loaded in FULL (not capped) since the Kabsch T_far
    recovery needs the whole trajectory to match against the saved traj_aligned.
    """

    def __new__(cls, host_name, near_topic="/nearfield/deskewed_world",
                far_topic="/cloud_registered", odom_topic="/aft_mapped_to_init",
                max_frames=None):
        key = (host_name, near_topic, far_topic, odom_topic, max_frames)
        if key in _SCENARIO_CACHE:
            return _SCENARIO_CACHE[key]
        obj = super().__new__(cls)
        obj._init(host_name, near_topic, far_topic, odom_topic, max_frames)
        _SCENARIO_CACHE[key] = obj
        return obj

    def _init(self, host_name, near_topic, far_topic, odom_topic, max_frames):
        self.host_name = host_name
        bag_path = BAGS[host_name]
        print(f"[{host_name}] loading frames from {bag_path} (max_frames={max_frames})")
        self.near = read_cloud_topic(bag_path, near_topic, max_frames=max_frames)
        self.far = read_cloud_topic(bag_path, far_topic, max_frames=max_frames)
        self.poses = read_pose_topic(bag_path, odom_topic)  # full trajectory, needed for Kabsch
        self.n = min(len(self.near), len(self.far), len(self.poses))
        if max_frames is not None:
            self.n = min(self.n, max_frames)
        print(f"[{host_name}] using {self.n} frames")

        out_dir = f"{BASE}/{host_name}_bgsub"
        bg = np.load(f"{out_dir}/bgsubtract_result.npz")
        traj_aligned_full = bg["traj_aligned"]
        orig_traj_full = np.array([p[:3] for p in self.poses])
        n_traj = min(len(orig_traj_full), len(traj_aligned_full))
        T_far, resid = kabsch_transform(orig_traj_full[:n_traj], traj_aligned_full[:n_traj])
        print(f"[{host_name}] Kabsch T_far residual: {resid:.5f}m (should be ~0)")
        self.R, self.t = T_far[:3, :3], T_far[:3, 3]

        self.ref_pcd = o3d.io.read_point_cloud(REFERENCE_MAP)
        self.ref_tree = scipy.spatial.cKDTree(np.asarray(self.ref_pcd.points))
        other_name = "swarm2" if host_name == "swarm1" else "swarm1"
        other_traj_aligned = np.load(f"{BASE}/{other_name}_bgsub/bgsubtract_result.npz")["traj_aligned"]
        self.other_traj_tree = scipy.spatial.cKDTree(other_traj_aligned)

    def frame(self, i):
        """Returns (t, near_pts_f, far_pts_f, pose, dn_near, df_far) range-filtered."""
        t, near_pts = self.near[i]
        _, far_pts = self.far[i]
        pose = self.poses[i]
        pose_xyz = np.array(pose[:3], dtype=np.float32)
        dn = np.linalg.norm(near_pts - pose_xyz, axis=1)
        near_mask = (dn > MIN_RANGE) & (dn < MAX_RANGE)
        df = np.linalg.norm(far_pts - pose_xyz, axis=1)
        far_mask = (df > MIN_RANGE) & (df < MAX_RANGE)
        return t, near_pts[near_mask], far_pts[far_mask], pose, dn[near_mask], df[far_mask]

    def gt_for_near(self, near_pts_f, t):
        """Returns (gt_positive, is_fg) boolean arrays for a frame's near-field points."""
        aligned = (self.R @ near_pts_f.T).T + self.t
        d_ref, _ = self.ref_tree.query(aligned, k=1)
        is_fg = d_ref > BG_THRESHOLD
        other_dists, _ = self.other_traj_tree.query(aligned, k=1)
        is_other = is_fg & (other_dists < OTHER_DRONE_THRESHOLD)
        return is_other, is_fg


class MetricsAccumulator:
    """Collects per-frame TP/FN/FP/TN + timing, produces the unified metrics table row."""

    def __init__(self):
        self.rows = []  # per-frame: t, dn (ranges of gt points), gt, tp, fn, fp_bg, tn_bg, t_seg_ms, t_run_ms

    def add_frame(self, t, dn_gt_pts, gt_positive_mask, pred_positive_mask, gt_negative_bg_mask,
                  t_seg_ms=None, t_run_ms=None):
        tp = int((gt_positive_mask & pred_positive_mask).sum())
        fn = int((gt_positive_mask & ~pred_positive_mask).sum())
        fp = int((gt_negative_bg_mask & pred_positive_mask).sum())
        tn = int((gt_negative_bg_mask & ~pred_positive_mask).sum())
        self.rows.append({"t": t, "dn": dn_gt_pts[gt_positive_mask].tolist(),
                          "tp_ranges": dn_gt_pts[gt_positive_mask & pred_positive_mask].tolist(),
                          "fn_ranges": dn_gt_pts[gt_positive_mask & ~pred_positive_mask].tolist(),
                          "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                          "t_seg_ms": t_seg_ms, "t_run_ms": t_run_ms})

    def summarize(self, method_name, host_name):
        tp = sum(r["tp"] for r in self.rows)
        fn = sum(r["fn"] for r in self.rows)
        fp = sum(r["fp"] for r in self.rows)
        tn = sum(r["tn"] for r in self.rows)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)

        pre90 = [r for r in self.rows if r["t"] < COLD_START_CUTOFF_S]
        post90 = [r for r in self.rows if r["t"] >= COLD_START_CUTOFF_S]
        tp_pre, fn_pre = sum(r["tp"] for r in pre90), sum(r["fn"] for r in pre90)
        tp_post, fn_post = sum(r["tp"] for r in post90), sum(r["fn"] for r in post90)
        recall_pre90 = tp_pre / max(tp_pre + fn_pre, 1)
        recall_post90 = tp_post / max(tp_post + fn_post, 1)

        all_tp_ranges = np.concatenate([r["tp_ranges"] for r in self.rows]) if self.rows else np.zeros(0)
        all_fn_ranges = np.concatenate([r["fn_ranges"] for r in self.rows]) if self.rows else np.zeros(0)
        bucket_recalls = {}
        for lo, hi in DIST_BUCKETS:
            n_tp_b = int(((all_tp_ranges >= lo) & (all_tp_ranges < hi)).sum())
            n_fn_b = int(((all_fn_ranges >= lo) & (all_fn_ranges < hi)).sum())
            bucket_recalls[f"{lo}-{hi}m"] = n_tp_b / max(n_tp_b + n_fn_b, 1)

        seg_times = [r["t_seg_ms"] for r in self.rows if r["t_seg_ms"] is not None]
        run_times = [r["t_run_ms"] for r in self.rows if r["t_run_ms"] is not None]

        # cold-start convergence: first t after which a trailing 20-frame recall window stays >0.85
        conv_s = None
        ts = np.array([r["t"] for r in self.rows])
        window = 20
        if len(self.rows) > window * 2:
            tp_arr = np.array([r["tp"] for r in self.rows])
            gt_arr = np.array([r["tp"] + r["fn"] for r in self.rows])
            tp_roll = np.convolve(tp_arr, np.ones(window), mode="valid")
            gt_roll = np.convolve(gt_arr, np.ones(window), mode="valid")
            recall_roll = tp_roll / np.maximum(gt_roll, 1)
            for i in range(len(recall_roll)):
                if recall_roll[i] > 0.85 and np.all(recall_roll[i:] > 0.75):
                    conv_s = float(ts[i + window - 1])
                    break

        return {
            "method": method_name, "host": host_name,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "precision": precision, "fpr": fpr,
            "recall_pre90": recall_pre90, "recall_post90": recall_post90,
            "bucket_recalls": bucket_recalls,
            "cold_start_convergence_s": conv_s,
            "seg_ms_mean": float(np.mean(seg_times)) if seg_times else None,
            "seg_ms_p95": float(np.percentile(seg_times, 95)) if seg_times else None,
            "run_ms_mean": float(np.mean(run_times)) if run_times else None,
            "run_ms_p95": float(np.percentile(run_times, 95)) if run_times else None,
        }


def markdown_table(summaries):
    cols = ["method", "host", "recall", "precision", "fpr", "recall_pre90", "recall_post90",
           "cold_start_convergence_s"]
    bucket_keys = list(summaries[0]["bucket_recalls"].keys()) if summaries else []
    header = "| " + " | ".join(cols + bucket_keys + ["seg_ms_mean", "run_ms_mean"]) + " |"
    sep = "|" + "---|" * (len(cols) + len(bucket_keys) + 2)
    lines = [header, sep]
    for s in summaries:
        vals = []
        for c in cols:
            v = s[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        for bk in bucket_keys:
            vals.append(f"{s['bucket_recalls'][bk]:.3f}")
        vals.append(f"{s['seg_ms_mean']:.2f}" if s['seg_ms_mean'] is not None else "-")
        vals.append(f"{s['run_ms_mean']:.2f}" if s['run_ms_mean'] is not None else "-")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
