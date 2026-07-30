#!/usr/bin/env python3
"""Stage 1: causal (real-time-style) DUFOMap dynamic-point detection vs. the official
two-pass batch reference, on our own two-drone Swarmbag flights.

cloud_transform semantics (confirmed from dufomap's own source, not guessed --
dufomap/__init__.py docstring: "whether need to transform the point cloud to the world
frame; if the point cloud is already in the world frame, set it to False"): our near/far
points are already world-frame Point-LIO output, so cloud_transform=False throughout.

Design (per spec):
- run() only ever integrates FAR-FIELD points into the background map (near-field is
  the detection target's own neighborhood -- letting near-field points into the
  background model would let a slow-moving target get absorbed into "background").
- segment() is called separately on near-field and far-field points each frame, counted
  separately -- the near-field count is what we actually care about (detection target
  range), far-field is a sanity/noise-floor reference.

CAUSAL mode: for frame i, segment(near_i) and segment(far_i) using the map built from
frames 0..i-1 ONLY, then run(far_i) to add this frame to history. Frame 0 has no
history (cold start -- labeled all-static by convention, not a real judgment).

BATCH mode (official reference usage, see dufomap main.py): run(far_i) for ALL frames
first, THEN segment(near_i)/segment(far_i) for each frame against the complete map.
This uses future frames' information -- not real-time, kept only as a comparison point.

Usage: python3 run_causal_vs_batch.py <bag_path> <out_dir> [--max-seconds 120]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2

from dufomap import dufomap

MIN_RANGE = 0.1
DEFAULT_MAX_RANGE = 50.0
FRAME_BUDGET_MS = 90.0  # ~1/10.5 Hz Mid360 frame rate we've been seeing


def read_cloud_topic(bag_path, topic, storage_id="mcap"):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
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


def read_pose_topic(bag_path, topic, storage_id="mcap"):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
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


def load_frames(bag_path, near_topic, far_topic, odom_topic, max_range, max_seconds):
    near = read_cloud_topic(bag_path, near_topic)
    far = read_cloud_topic(bag_path, far_topic)
    poses = read_pose_topic(bag_path, odom_topic)
    n = min(len(near), len(far), len(poses))
    if max_seconds is not None:
        n = min(n, sum(1 for t, _ in near[:n] if t <= max_seconds))
    print(f"near={len(near)} far={len(far)} poses={len(poses)} frames (using {n}, max_seconds={max_seconds})")

    frames = []
    for i in range(n):
        t, near_pts = near[i]
        _, far_pts = far[i]
        pose = poses[i]
        pose_xyz = np.array(pose[:3], dtype=np.float32)
        dn = np.linalg.norm(near_pts - pose_xyz, axis=1)
        near_pts = near_pts[(dn > MIN_RANGE) & (dn < max_range)]
        df = np.linalg.norm(far_pts - pose_xyz, axis=1)
        far_pts = far_pts[(df > MIN_RANGE) & (df < max_range)]
        frames.append((t, near_pts, far_pts, pose))
    return frames


def run_causal(frames, voxel, d_s, d_p):
    dm = dufomap(voxel, d_s, d_p, num_threads=0)
    rows = []
    labeled = []  # (t, near_pts, near_labels, far_pts, far_labels)
    for i, (t, near_pts, far_pts, pose) in enumerate(frames):
        t0 = time.time()
        if i == 0:
            near_labels = np.zeros(len(near_pts), dtype=np.uint8)
            far_labels = np.zeros(len(far_pts), dtype=np.uint8)
        else:
            near_labels = dm.segment(near_pts, pose, cloud_transform=False) if len(near_pts) else np.zeros(0, dtype=np.uint8)
            far_labels = dm.segment(far_pts, pose, cloud_transform=False) if len(far_pts) else np.zeros(0, dtype=np.uint8)
        t_seg = time.time()
        dm.run(far_pts, pose, cloud_transform=False)
        t_run = time.time()
        rows.append({
            "frame_idx": i, "t": t,
            "n_dynamic_near": int(near_labels.sum()), "n_dynamic_far": int(far_labels.sum()),
            "t_segment_ms": (t_seg - t0) * 1000, "t_run_ms": (t_run - t_seg) * 1000,
        })
        labeled.append((t, near_pts, near_labels, far_pts, far_labels))
        if i % 200 == 0:
            print(f"  causal frame {i}/{len(frames)}: dyn_near={int(near_labels.sum())} dyn_far={int(far_labels.sum())}")
    return rows, labeled


def run_batch(frames, voxel, d_s, d_p):
    dm = dufomap(voxel, d_s, d_p, num_threads=0)
    t0 = time.time()
    for i, (t, near_pts, far_pts, pose) in enumerate(frames):
        dm.run(far_pts, pose, cloud_transform=False)
        if i % 500 == 0:
            print(f"  batch run: frame {i}/{len(frames)}")
    print(f"  batch STEP1 (run all) done in {time.time()-t0:.1f}s")

    rows = []
    labeled = []
    t0 = time.time()
    for i, (t, near_pts, far_pts, pose) in enumerate(frames):
        near_labels = dm.segment(near_pts, pose, cloud_transform=False) if len(near_pts) else np.zeros(0, dtype=np.uint8)
        far_labels = dm.segment(far_pts, pose, cloud_transform=False) if len(far_pts) else np.zeros(0, dtype=np.uint8)
        rows.append({
            "frame_idx": i, "t": t,
            "n_dynamic_near": int(near_labels.sum()), "n_dynamic_far": int(far_labels.sum()),
        })
        labeled.append((t, near_pts, near_labels, far_pts, far_labels))
    print(f"  batch STEP2 (segment all) done in {time.time()-t0:.1f}s")
    return rows, labeled


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def cold_start_seconds(causal_rows, batch_rows, window=20, tol=0.5):
    """First time causal's rolling-mean near-field dynamic count gets within `tol` (as a
    ratio) of batch's rolling-mean for the same window, and stays there. Rough, printed
    as a diagnostic, not a hard scientific claim."""
    cn = np.array([r["n_dynamic_near"] for r in causal_rows], dtype=float)
    bn = np.array([r["n_dynamic_near"] for r in batch_rows], dtype=float)
    ts = np.array([r["t"] for r in causal_rows])
    n = len(cn)
    if n < window * 2:
        return None
    kernel = np.ones(window) / window
    cn_roll = np.convolve(cn, kernel, mode="valid")
    bn_roll = np.convolve(bn, kernel, mode="valid")
    bn_roll_safe = np.maximum(bn_roll, 1.0)
    ratio = np.abs(cn_roll - bn_roll) / bn_roll_safe
    converged = ratio < tol
    for i in range(len(converged)):
        if converged[i] and np.all(converged[i:]):
            return float(ts[i + window - 1])
    return None


def svg_line_chart(series_list, colors, labels, out_path, title, xlabel="frame", ylabel="dynamic points (near-field)"):
    """Hand-rolled SVG polyline chart -- no matplotlib/GUI dependency."""
    W, H, PAD = 900, 400, 50
    all_y = np.concatenate([np.array(s) for s in series_list])
    ymax = max(float(all_y.max()), 1.0)
    n = max(len(s) for s in series_list)

    def pts_to_path(series):
        pts = []
        for i, y in enumerate(series):
            x = PAD + (W - 2 * PAD) * i / max(n - 1, 1)
            yy = H - PAD - (H - 2 * PAD) * (y / ymax)
            pts.append(f"{x:.1f},{yy:.1f}")
        return "M " + " L ".join(pts)

    lines = ""
    legend = ""
    for s, c, lab in zip(series_list, colors, labels):
        lines += f'<path d="{pts_to_path(s)}" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.85"/>\n'
    for idx, (c, lab) in enumerate(zip(colors, labels)):
        legend += f'<rect x="{PAD + idx*160}" y="10" width="12" height="12" fill="{c}"/>' \
                   f'<text x="{PAD + idx*160 + 18}" y="20" font-size="12" font-family="sans-serif">{lab}</text>\n'

    svg = f"""<svg width="{W}" height="{H+30}" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="white"/>
<text x="{PAD}" y="{H+20}" font-size="13" font-family="sans-serif">{title}</text>
<g transform="translate(0,30)">
{legend}
<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#999"/>
<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#999"/>
<text x="5" y="{PAD}" font-size="11" font-family="sans-serif">{ymax:.0f}</text>
<text x="5" y="{H-PAD}" font-size="11" font-family="sans-serif">0</text>
{lines}
</g>
</svg>"""
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"wrote {out_path}")


def viewer_html(labeled_causal, traj, out_path, title):
    """Two-panel slider: left = current frame near-field colored (gray static / red
    dynamic, causal judgment), right = accumulated near-field dynamic trail over time."""
    def flat(a):
        return np.asarray(a).flatten().round(4).tolist()

    all_pts, all_colors, frame_counts = [], [], []
    dyn_flat, dyn_cum = [], []
    dyn_running = 0
    for t, near_pts, near_labels, far_pts, far_labels in labeled_causal:
        all_pts.append(near_pts)
        colors = np.where(near_labels[:, None] == 1, np.array([1.0, 0.15, 0.1]), np.array([0.75, 0.75, 0.78])) \
            if len(near_pts) else np.zeros((0, 3))
        all_colors.append(colors)
        frame_counts.append(len(near_pts))
        dyn_pts = near_pts[near_labels == 1] if len(near_pts) else np.zeros((0, 3))
        if len(dyn_pts):
            dyn_flat.append(dyn_pts)
            dyn_running += len(dyn_pts)
        dyn_cum.append(dyn_running)

    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    all_colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3))
    dyn_flat = np.concatenate(dyn_flat, axis=0) if dyn_flat else np.zeros((0, 3))
    cum_counts = np.cumsum(frame_counts).tolist()
    frame_times = [round(t, 3) for t, *_ in labeled_causal]

    bbox_src = np.concatenate([all_pts, traj], axis=0) if len(all_pts) else traj
    mins = bbox_src.min(axis=0)
    maxs = bbox_src.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 10px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 16px; align-items:center; flex-wrap: wrap; font-size:13px; }}
  .swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; }}
  html, body {{ height:100%; }}
  #panels {{ display:flex; width:100%; height: calc(100vh - 90px); position:relative; }}
  .panel {{ flex:1; position:relative; border-right:1px solid #ccc; overflow:hidden; }}
  .panel-title {{ position:absolute; top:8px; left:12px; z-index:10; font-size:13px;
                  color:#0a7a0a; background:rgba(255,255,255,0.85); padding:2px 6px; border-radius:4px; }}
  #syncOverlay {{ position:absolute; top:0; left:0; width:100%; height:100%; z-index:20; }}
  canvas {{ display:block; }}
  #sliderRow {{ padding: 8px 20px; display:flex; gap:12px; align-items:center; background:#f7f7f7; }}
  #sliderRow input[type=range] {{ flex:1; }}
</style></head>
<body>
<div id="controls">
  <span><span class="swatch" style="background:#bfbfbf;"></span> Near-field: static (causal judgment)</span>
  <span><span class="swatch" style="background:#ff2619;"></span> Near-field: dynamic (causal judgment, history up to this frame only)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> Own trajectory</span>
  <span style="color:#888;">frame 0: cold start, no history yet, treated as static by convention</span>
</div>
<div id="panels">
  <div class="panel" id="leftCanvas"><div class="panel-title">Current frame near-field (causal judgment)</div></div>
  <div class="panel" id="rightCanvas"><div class="panel-title">Accumulated near-field dynamic-point trail</div></div>
  <div id="syncOverlay"></div>
</div>
<div id="sliderRow">
  <button id="playBtn">&#9654; Play</button>
  <input type="range" id="frameSlider" min="0" max="{len(labeled_causal)-1}" value="0">
  <span id="frameLabel">frame 0 / {len(labeled_causal)-1}</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const ALL_PTS = new Float32Array({json.dumps(flat(all_pts))});
const ALL_COLORS = new Float32Array({json.dumps(flat(all_colors))});
const DYN_PTS = new Float32Array({json.dumps(flat(dyn_flat))});
const FRAME_COUNTS = {json.dumps(frame_counts)};
const CUM_COUNTS = {json.dumps(cum_counts)};
const DYN_CUM_COUNTS = {json.dumps(dyn_cum)};
const FRAME_TIMES = {json.dumps(frame_times)};
const TRAJ_FLAT = new Float32Array({json.dumps(flat(traj))});
const BBOX_CENTER = {json.dumps(center)};
const BBOX_RADIUS = {radius};

function makePanel(container) {{
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  scene.add(new THREE.AxesHelper(Math.max(0.5, BBOX_RADIUS * 0.1)));
  scene.add(new THREE.GridHelper(BBOX_RADIUS * 2.5, 20, 0xaaaaaa, 0xdddddd));
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  return {{ scene, renderer, container }};
}}
const left = makePanel(document.getElementById('leftCanvas'));
const right = makePanel(document.getElementById('rightCanvas'));

const trajGeom = new THREE.BufferGeometry();
trajGeom.setAttribute('position', new THREE.BufferAttribute(TRAJ_FLAT, 3));
const trajMat = new THREE.LineBasicMaterial({{ color: 0x2266ff }});
left.scene.add(new THREE.Line(trajGeom, trajMat));
right.scene.add(new THREE.Line(trajGeom.clone(), trajMat));

const curGeom = new THREE.BufferGeometry();
const curPoints = new THREE.Points(curGeom, new THREE.PointsMaterial({{ size: 0.06, vertexColors: true }}));
left.scene.add(curPoints);

const dynGeom = new THREE.BufferGeometry();
const dynPoints = new THREE.Points(dynGeom, new THREE.PointsMaterial({{ size: 0.09, color: 0xff2619 }}));
right.scene.add(dynPoints);

function setFrame(idx) {{
  const start = idx === 0 ? 0 : CUM_COUNTS[idx-1];
  const end = CUM_COUNTS[idx];
  curGeom.setAttribute('position', new THREE.BufferAttribute(ALL_PTS.subarray(start*3, end*3), 3));
  curGeom.setAttribute('color', new THREE.BufferAttribute(ALL_COLORS.subarray(start*3, end*3), 3));
  curGeom.attributes.position.needsUpdate = true;
  curGeom.attributes.color.needsUpdate = true;
  const dynEnd = DYN_CUM_COUNTS[idx];
  const dynStart = idx === 0 ? 0 : DYN_CUM_COUNTS[idx-1];
  dynGeom.setAttribute('position', new THREE.BufferAttribute(DYN_PTS.subarray(0, dynEnd*3), 3));
  dynGeom.attributes.position.needsUpdate = true;
  document.getElementById('frameLabel').textContent =
    `frame ${{idx}} / {len(labeled_causal)-1}, t=${{FRAME_TIMES[idx].toFixed(2)}}s, near-field pts=${{FRAME_COUNTS[idx]}}, dynamic=${{dynEnd-dynStart}}`;
}}
setFrame(0);

const [cx, cy, cz] = BBOX_CENTER;
const sharedCamera = new THREE.PerspectiveCamera(60, left.container.clientWidth / left.container.clientHeight, 0.01, 2000);
sharedCamera.up.set(0, 0, 1);
sharedCamera.position.set(cx + BBOX_RADIUS, cy - BBOX_RADIUS, cz + BBOX_RADIUS * 0.6);
const syncOverlay = document.getElementById('syncOverlay');
const sharedControls = new THREE.OrbitControls(sharedCamera, syncOverlay);
sharedControls.target.set(cx, cy, cz);

function animate() {{
  requestAnimationFrame(animate);
  sharedControls.update();
  left.renderer.render(left.scene, sharedCamera);
  right.renderer.render(right.scene, sharedCamera);
}}
animate();

window.addEventListener('resize', () => {{
  sharedCamera.aspect = left.container.clientWidth / left.container.clientHeight;
  sharedCamera.updateProjectionMatrix();
  left.renderer.setSize(left.container.clientWidth, left.container.clientHeight);
  right.renderer.setSize(right.container.clientWidth, right.container.clientHeight);
}});

const slider = document.getElementById('frameSlider');
slider.addEventListener('input', () => setFrame(parseInt(slider.value)));
let playing = false, playTimer = null;
document.getElementById('playBtn').addEventListener('click', (e) => {{
  playing = !playing;
  e.target.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    playTimer = setInterval(() => {{
      let v = parseInt(slider.value) + 1;
      if (v > slider.max) v = 0;
      slider.value = v;
      setFrame(v);
    }}, 50);
  }} else {{ clearInterval(playTimer); }}
}});
</script></body></html>"""
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path")
    ap.add_argument("out_dir")
    ap.add_argument("--near-topic", default="/nearfield/deskewed_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== loading frames from {args.bag_path} ===")
    frames = load_frames(args.bag_path, args.near_topic, args.far_topic, args.odom_topic,
                         args.max_range, args.max_seconds)
    traj = np.array([f[3][:3] for f in frames])

    print("\n=== CAUSAL (segment-before-run, history-only) ===")
    causal_rows, causal_labeled = run_causal(frames, args.voxel, args.d_s, args.d_p)
    write_csv(os.path.join(args.out_dir, "causal_per_frame.csv"), causal_rows,
              ["frame_idx", "t", "n_dynamic_near", "n_dynamic_far", "t_segment_ms", "t_run_ms"])

    print("\n=== BATCH (official two-pass reference) ===")
    batch_rows, batch_labeled = run_batch(frames, args.voxel, args.d_s, args.d_p)
    write_csv(os.path.join(args.out_dir, "batch_per_frame.csv"), batch_rows,
              ["frame_idx", "t", "n_dynamic_near", "n_dynamic_far"])

    cn = np.array([r["n_dynamic_near"] for r in causal_rows])
    bn = np.array([r["n_dynamic_near"] for r in batch_rows])
    print(f"\ncausal near-field dynamic/frame: mean={cn.mean():.1f} median={np.median(cn):.0f} p90={np.percentile(cn,90):.0f}")
    print(f"batch  near-field dynamic/frame: mean={bn.mean():.1f} median={np.median(bn):.0f} p90={np.percentile(bn,90):.0f}")

    cold_start = cold_start_seconds(causal_rows, batch_rows)
    print(f"\ncold-start convergence (causal rolling-mean within 50% of batch's, window=20 frames): "
          f"{cold_start:.1f}s" if cold_start is not None else "\ncold-start convergence: NOT REACHED within this recording")

    t_seg = np.array([r["t_segment_ms"] for r in causal_rows[1:]])  # skip frame 0, no segment call
    t_run = np.array([r["t_run_ms"] for r in causal_rows])
    t_total = t_seg + t_run[1:]
    print(f"\ncausal per-frame timing (WSL2 x86, NOT Jetson): "
          f"segment mean={t_seg.mean():.2f}ms p95={np.percentile(t_seg,95):.2f}ms | "
          f"run mean={t_run.mean():.2f}ms p95={np.percentile(t_run,95):.2f}ms | "
          f"total mean={t_total.mean():.2f}ms p95={np.percentile(t_total,95):.2f}ms "
          f"(budget={FRAME_BUDGET_MS}ms)")
    frac_over_budget = (t_total > FRAME_BUDGET_MS).mean() * 100
    print(f"frames exceeding {FRAME_BUDGET_MS}ms budget: {frac_over_budget:.1f}%")

    summary = {
        "bag": args.bag_path, "n_frames": len(frames),
        "params": {"voxel": args.voxel, "d_s": args.d_s, "d_p": args.d_p,
                   "max_range": args.max_range, "max_seconds": args.max_seconds},
        "causal_near_mean": float(cn.mean()), "causal_near_median": float(np.median(cn)),
        "batch_near_mean": float(bn.mean()), "batch_near_median": float(np.median(bn)),
        "cold_start_seconds": cold_start,
        "timing_ms": {"segment_mean": float(t_seg.mean()), "segment_p95": float(np.percentile(t_seg, 95)),
                      "run_mean": float(t_run.mean()), "run_p95": float(np.percentile(t_run, 95)),
                      "total_mean": float(t_total.mean()), "total_p95": float(np.percentile(t_total, 95)),
                      "frac_over_budget_pct": float(frac_over_budget)},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'summary.json')}")

    svg_line_chart([cn, bn], ["#ff2619", "#2266dd"], ["causal (history-only)", "batch (official, uses future)"],
                   os.path.join(args.out_dir, "near_dynamic_count_comparison.svg"),
                   f"Near-field dynamic points/frame: causal vs batch -- {os.path.basename(args.bag_path)}")

    viewer_html(causal_labeled, traj, os.path.join(args.out_dir, "causal_near_field_viewer.html"),
               f"Causal DUFOMap near-field detection result -- {os.path.basename(args.bag_path)}")


if __name__ == "__main__":
    main()
