#!/usr/bin/env python3
"""d_s ablation for the causal DUFOMap detector. resolution and d_p were already swept
(ablation_resolution_dp.py) -- d_s (inflate-hits distance) had never been touched, always
left at the paper's default of 0.2. Fixes resolution=0.15 (chosen for its much higher
recall) and d_p=2 (the clear winner from the resolution/d_p ablation), and sweeps d_s
over a few values.

Usage: python3 ablation_ds.py [--max-frames N] [--out results_ds.json]
"""
import argparse
import json
import time

import numpy as np
from dufomap import dufomap

from common_eval import Scenario, MetricsAccumulator, markdown_table, BAGS

RESOLUTION_FIXED = 0.15
D_P_FIXED = 2


def run_one(host_name, d_s, max_frames=None):
    scen = Scenario(host_name, max_frames=max_frames)
    dm = dufomap(RESOLUTION_FIXED, d_s, D_P_FIXED, num_threads=0)
    acc = MetricsAccumulator()

    t_start = time.time()
    for i in range(scen.n):
        t, near_f, far_f, pose, dn, df = scen.frame(i)

        t0 = time.time()
        if i == 0 or len(near_f) == 0:
            near_labels = np.zeros(len(near_f), dtype=np.uint8)
        else:
            near_labels = dm.segment(near_f, pose, cloud_transform=False)
        t_seg = time.time()
        dm.run(far_f, pose, cloud_transform=False)
        t_run = time.time()

        if len(near_f) == 0:
            continue
        gt_positive, is_fg = scen.gt_for_near(near_f, t)
        pred_positive = near_labels.astype(bool)
        gt_negative_bg = ~is_fg
        acc.add_frame(t, dn, gt_positive, pred_positive, gt_negative_bg,
                      t_seg_ms=(t_seg - t0) * 1000, t_run_ms=(t_run - t_seg) * 1000)

        if i % 500 == 0:
            print(f"  [{host_name} res={RESOLUTION_FIXED} d_p={D_P_FIXED} d_s={d_s}] frame {i}/{scen.n}")

    print(f"[{host_name} d_s={d_s}] loop done in {time.time()-t_start:.1f}s")
    return acc.summarize(f"d_s_ablation(res={RESOLUTION_FIXED},d_p={D_P_FIXED},d_s={d_s})", host_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default="results_ds.json")
    ap.add_argument("--d-s-values", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    args = ap.parse_args()

    summaries = []
    for d_s in args.d_s_values:
        for host_name in BAGS.keys():
            s = run_one(host_name, d_s, max_frames=args.max_frames)
            summaries.append(s)
            print(json.dumps(s, indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nwrote {args.out}")

    md = markdown_table(summaries)
    print("\n" + md)
    with open(args.out.replace(".json", ".md"), "w") as f:
        f.write(md)
    print(f"wrote {args.out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
