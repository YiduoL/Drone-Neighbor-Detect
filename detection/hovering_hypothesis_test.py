#!/usr/bin/env python3
"""Test: does the causal detector specifically fail to detect the OTHER drone while it's
HOVERING (a known theoretical weakness of occupancy/ray-casting dynamic detectors -- an
object that stops moving eventually looks statistically like real static structure)?

No cross-bag time sync needed for this test (we still don't have one) -- it works purely
spatially: the other flight's own trajectory carries its own relative-time, so we can
compute its own speed at every point on its trajectory and classify each as "hovering"
or "moving" independent of the host flight. Then for each of the host's pseudo-GT
label==2 points ("this is the other drone", see build_pseudo_labels.py), find which
point on the other trajectory it matched to (same KD-tree query build_pseudo_labels.py
uses) and tag it hovering/moving. Compare causal-detector recall between the two groups.

Usage: python3 hovering_hypothesis_test.py --host swarm1 [--hover-speed 0.15]
"""
import argparse
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
import scipy.spatial

BASE = os.environ.get("DETECTION_DATA_DIR", "./data")
BAGS = {
    "swarm1": os.environ.get("SWARM1_BAG", "./data/pointlio_lidar_1_output"),
    "swarm2": os.environ.get("SWARM2_BAG", "./data/pointlio_lidar_2_output"),
}


def read_pose_topic_with_time(bag_path, topic="/aft_mapped_to_init"):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    times, pos = [], []
    t0 = None
    while r.has_next():
        _, data, t = r.read_next()
        if t0 is None:
            t0 = t
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose.position
        times.append((t - t0) / 1e9)
        pos.append([p.x, p.y, p.z])
    return np.array(times), np.array(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"])
    ap.add_argument("--hover-speed", type=float, default=0.15,
                    help="m/s below which we call the other drone 'hovering'")
    args = ap.parse_args()

    host = args.host
    other = "swarm2" if host == "swarm1" else "swarm1"

    print(f"=== computing {other}'s own speed profile (its own relative time, no cross-bag sync needed) ===")
    other_times, other_pos_orig = read_pose_topic_with_time(BAGS[other])
    speed = np.zeros(len(other_pos_orig))
    dt = np.diff(other_times)
    dp = np.linalg.norm(np.diff(other_pos_orig, axis=0), axis=1)
    speed[1:] = dp / np.maximum(dt, 1e-3)
    is_hover = speed < args.hover_speed
    print(f"{other}: {len(other_pos_orig)} poses, {is_hover.sum()} ({100*is_hover.mean():.1f}%) "
          f"below {args.hover_speed} m/s")
    # find contiguous hovering segments >2s for a sanity printout
    seg_start = None
    print("hovering segments (>2s):")
    for i in range(len(is_hover)):
        if is_hover[i] and seg_start is None:
            seg_start = i
        elif not is_hover[i] and seg_start is not None:
            if other_times[i] - other_times[seg_start] > 2.0:
                print(f"  t=[{other_times[seg_start]:.1f}, {other_times[i]:.1f}]s "
                      f"({other_times[i]-other_times[seg_start]:.1f}s)")
            seg_start = None

    other_bg = np.load(f"{BASE}/{other}_bgsub/bgsubtract_result.npz")
    other_traj_aligned = other_bg["traj_aligned"]
    n_common = min(len(other_traj_aligned), len(is_hover))
    other_traj_tree = scipy.spatial.cKDTree(other_traj_aligned[:n_common])

    print(f"\n=== loading {host}'s GT labels + causal detector predictions ===")
    labels = np.load(f"{BASE}/{host}_bgsub/near_field_labels.npz")
    gt_pts = labels["points"][labels["label"] == 2]
    print(f"{host}: {len(gt_pts)} GT 'other-drone' points")

    _, nearest_idx = other_traj_tree.query(gt_pts, k=1)
    gt_is_hover_match = is_hover[:n_common][nearest_idx]
    print(f"of these, {gt_is_hover_match.sum()} ({100*gt_is_hover_match.mean():.1f}%) matched "
          f"to a point on {other}'s trajectory where it was hovering (speed<{args.hover_speed}m/s)")

    tp_pts = np.load(f"{BASE}/{host}_bgsub/causal_vs_labels_points.npz")["tp_pts"]
    fn_pts = np.load(f"{BASE}/{host}_bgsub/causal_vs_labels_points.npz")["fn_pts"]
    print(f"{host}: causal_vs_labels_points.npz has tp={len(tp_pts)} fn={len(fn_pts)}")

    tp_tree = scipy.spatial.cKDTree(tp_pts) if len(tp_pts) else None
    fn_tree = scipy.spatial.cKDTree(fn_pts) if len(fn_pts) else None

    def recall_for_subset(mask, label):
        subset = gt_pts[mask]
        if len(subset) == 0:
            print(f"{label}: no GT points")
            return
        d_tp, _ = tp_tree.query(subset, k=1) if tp_tree is not None else (np.full(len(subset), np.inf), None)
        d_fn, _ = fn_tree.query(subset, k=1) if fn_tree is not None else (np.full(len(subset), np.inf), None)
        is_tp = d_tp < 0.01  # exact point match (same coordinates, just re-identifying which)
        is_fn = d_fn < 0.01
        n_tp, n_fn = int(is_tp.sum()), int(is_fn.sum())
        recall = n_tp / max(n_tp + n_fn, 1)
        print(f"{label}: n_gt={len(subset)} matched(tp+fn)={n_tp+n_fn} recall={recall:.3f}")

    print(f"\n=== recall comparison: GT points near a HOVERING vs MOVING {other} ===")
    recall_for_subset(gt_is_hover_match, "hovering-matched GT points")
    recall_for_subset(~gt_is_hover_match, "moving-matched GT points")


if __name__ == "__main__":
    main()
