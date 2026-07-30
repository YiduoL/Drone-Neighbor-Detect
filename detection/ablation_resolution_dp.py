#!/usr/bin/env python3
"""Parameter ablation for the causal DUFOMap detector: d_p in {1, 2} x resolution in
{0.1, 0.15}, with d_s and hit_extension left at the paper's defaults (0.2, True).

Usage: python3 ablation_resolution_dp.py [--max-frames N] [--out results.json]
"""
import argparse
import json
import time

import numpy as np
from dufomap import dufomap

from common_eval import Scenario, MetricsAccumulator, markdown_table, BAGS

D_S_DEFAULT = 0.2


def run_one(host_name, resolution, d_p, max_frames=None):
    scen = Scenario(host_name, max_frames=max_frames)
    dm = dufomap(resolution, D_S_DEFAULT, d_p, num_threads=0)
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
            print(f"  [{host_name} res={resolution} d_p={d_p}] frame {i}/{scen.n}")

    print(f"[{host_name} res={resolution} d_p={d_p}] loop done in {time.time()-t_start:.1f}s")
    return acc.summarize(f"baseline(res={resolution},d_p={d_p})", host_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default="results_resolution_dp.json")
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.1, 0.15])
    ap.add_argument("--d-ps", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    summaries = []
    for resolution in args.resolutions:
        for d_p in args.d_ps:
            for host_name in BAGS.keys():
                s = run_one(host_name, resolution, d_p, max_frames=args.max_frames)
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
