#!/usr/bin/env python3
"""Build pseudo-ground-truth labels for near-field points: label 0/1/2 combining
background subtraction (background_subtraction.py) with spatial proximity to the OTHER
flight's registered trajectory, indexed the same way as background_subtraction.py's
saved `near_pts`.

  0 = background (close to the static reference map)
  1 = foreground but not close to the other drone's trajectory (unexplained -- likely
      registration residue or missed structure in the reference map)
  2 = foreground AND within --threshold of the other flight's trajectory (labeled as
      "the other drone" -- a SPATIAL candidate only, not time-confirmed; see the
      caveat below)

Caveat: without a shared clock between the two flights' recording devices, "close to
the other trajectory" means close to any point the other drone visited over its ENTIRE
flight, not "where it was at the same instant" -- this produces some false positives
from the platforms coincidentally revisiting the same physical location at different
times. Combining this with the background-subtraction signal (must ALSO be far from
the static reference map) reduces, but does not eliminate, this effect.

Depends on background_subtraction.py having already been run for BOTH flights (writes
`bgsubtract_result.npz` per flight: near_pts / is_fg / traj_aligned, all already
registered into the same shared reference frame).

Usage: python3 build_pseudo_labels.py --host swarm1 [--threshold 0.35]
"""
import argparse
import os

import numpy as np
import open3d as o3d

BASE = os.environ.get("DETECTION_DATA_DIR", "./data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"])
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    host = args.host
    other = "swarm2" if host == "swarm1" else "swarm1"

    bg = np.load(f"{BASE}/{host}_bgsub/bgsubtract_result.npz")
    near_pts = bg["near_pts"]
    is_fg = bg["is_fg"]

    other_bg = np.load(f"{BASE}/{other}_bgsub/bgsubtract_result.npz")
    other_traj_pcd = o3d.geometry.PointCloud()
    other_traj_pcd.points = o3d.utility.Vector3dVector(other_bg["traj_aligned"])
    tree = o3d.geometry.KDTreeFlann(other_traj_pcd)

    label = np.zeros(len(near_pts), dtype=np.uint8)
    label[is_fg] = 1
    fg_idx = np.where(is_fg)[0]
    fg_pts = near_pts[is_fg]

    dists = np.empty(len(fg_pts))
    for i, p in enumerate(fg_pts):
        _, idx, d2 = tree.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(d2[0])
    is_other = dists < args.threshold
    label[fg_idx[is_other]] = 2

    out_path = f"{BASE}/{host}_bgsub/near_field_labels.npz"
    np.savez(out_path, points=near_pts, label=label)
    n0, n1, n2 = (label == 0).sum(), (label == 1).sum(), (label == 2).sum()
    print(f"{host}: wrote {out_path}")
    print(f"  label=0 background: {n0} ({100*n0/len(label):.1f}%)")
    print(f"  label=1 foreground/unexplained: {n1} ({100*n1/len(label):.1f}%)")
    print(f"  label=2 other-drone (spatial candidate, NOT time-confirmed): "
          f"{n2} ({100*n2/len(label):.2f}%)")


if __name__ == "__main__":
    main()
