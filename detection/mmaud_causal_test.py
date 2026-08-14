#!/usr/bin/env python3
"""Run the causal DUFOMap detector on MMAUD (ground-fixed 360-degree LiDAR observing a
drone at range, with real annotated GT position -- unlike Swarmbag, this has a single
clock and genuine per-frame time-synced ground truth, no cross-bag sync problem).

MMAUD's val split is cut into 16 short (~5s) clips, most too short on their own for the
causal map to warm up past cold start (frame 0 has no history and is skipped by
convention). Several clips are actually back-to-back continuations of the same
recording (near-zero gap between end of one and start of the next); chaining ALL 16
clips in chronological order gives ~794 frames (~80s) of continuous history for one
fixed sensor position, instead of 50 frames (~5s) per isolated clip.

Sensor is stationary -> identity pose for every frame (matches this project's
established "static platform doesn't need SLAM" pattern).

Usage: python3 mmaud_causal_test.py [--resolution 0.15] [--d-s 0.2] [--d-p 2]
       [--match-radius 1.0] [--out mmaud_overlay.html]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from dufomap import dufomap

MMAUD_ROOT = os.environ.get("MMAUD_ROOT", "./data/MMAUD/val")
GT_CSV = os.environ.get("MMAUD_GT_CSV", "./data/MMAUD/validation_ref_new.csv")

# Chronological order (by first-frame timestamp), not numeric seq order -- needed to
# chain the back-to-back clips (seq0003+04+05, seq0006+07+08, seq0010+11+12,
# seq0013+14+15 each have <0.2s gap between them, i.e. one continuous recording).
CHRONOLOGICAL_SEQS = [
    "seq0002", "seq0001", "seq0016", "seq0003", "seq0004", "seq0005",
    "seq0006", "seq0007", "seq0008", "seq0009", "seq0010", "seq0011",
    "seq0012", "seq0013", "seq0014", "seq0015",
]


def load_frames(seq, topic):
    d = os.path.join(MMAUD_ROOT, seq, topic)
    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    frames = []
    for f in files:
        ts = float(os.path.basename(f).replace(".npy", ""))
        pts = np.load(f).astype(np.float32)
        nz = ~np.all(pts == 0, axis=1)
        frames.append((ts, pts[nz]))
    return frames


def parse_position(s):
    return np.array([float(x) for x in s.strip("[]").split(",")])


def flat(a):
    return np.asarray(a, dtype=np.float64).flatten().round(4).tolist()


def write_overlay_html(out_html, seg_pts, seg_colors, seg_times, gt_marker_pos,
                       gt_marker_hit, tp, fn, fp):
    frame_counts = [len(p) for p in seg_pts]
    cum_counts = np.cumsum(frame_counts).tolist() if frame_counts else [0]
    all_pts = np.concatenate(seg_pts, axis=0) if seg_pts else np.zeros((0, 3))
    all_colors = np.concatenate(seg_colors, axis=0) if seg_colors else np.zeros((0, 3))
    n_frames = len(seg_pts)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    bbox_src = all_pts if len(all_pts) else np.zeros((1, 3))
    mins, maxs = bbox_src.min(axis=0), bbox_src.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>MMAUD causal detection -- TP={tp} FN={fn} FP={fp} (frame-level)</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 8px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 14px; align-items:center; flex-wrap: wrap; font-size:13px; }}
  .swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; }}
  html, body {{ height:100%; }}
  #view {{ width:100%; height: calc(100vh - 84px); }}
  canvas {{ display:block; }}
  #sliderRow {{ padding: 8px 20px; display:flex; gap:12px; align-items:center; background:#f7f7f7; }}
  #sliderRow input[type=range] {{ flex:1; }}
</style></head>
<body>
<div id="controls">
  <span><b>MMAUD, all 16 val clips chained chronologically (stationary sensor)</b></span>
  <span><span class="swatch" style="background:#cccccc;"></span> Background (this frame)</span>
  <span><span class="swatch" style="background:#22a622;"></span> Detected dynamic, near GT (hit)</span>
  <span><span class="swatch" style="background:#ff2619;"></span> Detected dynamic, NOT near GT (FP)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> GT drone position marker (green=hit this frame, orange=missed)</span>
  <span style="color:#888;">frame-level recall={recall:.3f} precision={precision:.3f} (TP={tp} FN={fn} FP={fp})</span>
</div>
<div id="view"></div>
<div id="sliderRow">
  <button id="playBtn">&#9654; Play</button>
  <input type="range" id="frameSlider" min="0" max="{max(n_frames-1,0)}" value="0">
  <span id="frameLabel">frame 0 / {max(n_frames-1,0)}</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const ALL_PTS = new Float32Array({json.dumps(flat(all_pts))});
const ALL_COLORS = new Float32Array({json.dumps(flat(all_colors))});
const FRAME_COUNTS = {json.dumps(frame_counts)};
const CUM_COUNTS = {json.dumps(cum_counts)};
const FRAME_TIMES = {json.dumps([round(t, 3) for t in seg_times])};
const GT_MARKER_POS = {json.dumps(flat(np.asarray(gt_marker_pos)))};
const GT_MARKER_HIT = {json.dumps(gt_marker_hit)};
const BBOX_CENTER = {json.dumps(center)};
const BBOX_RADIUS = {radius};

const container = document.getElementById('view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
scene.add(new THREE.AxesHelper(Math.max(0.5, BBOX_RADIUS * 0.1)));
scene.add(new THREE.GridHelper(BBOX_RADIUS * 2.5, 20, 0xaaaaaa, 0xdddddd));
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const [cx, cy, cz] = BBOX_CENTER;
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 2000);
camera.up.set(0, 0, 1);
camera.position.set(cx + BBOX_RADIUS, cy - BBOX_RADIUS, cz + BBOX_RADIUS * 0.6);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);

const ptsGeom = new THREE.BufferGeometry();
const ptsMat = new THREE.PointsMaterial({{ size: 0.15, vertexColors: true }});
const ptsObj = new THREE.Points(ptsGeom, ptsMat);
scene.add(ptsObj);

const markerGeom = new THREE.SphereGeometry(0.3, 16, 16);
const markerMat = new THREE.MeshBasicMaterial({{ color: 0x2266ff }});
const marker = new THREE.Mesh(markerGeom, markerMat);
scene.add(marker);

function setFrame(idx) {{
  if (FRAME_COUNTS.length === 0) return;
  const start = idx === 0 ? 0 : CUM_COUNTS[idx-1];
  const end = CUM_COUNTS[idx];
  ptsGeom.setAttribute('position', new THREE.BufferAttribute(ALL_PTS.subarray(start*3, end*3), 3));
  ptsGeom.setAttribute('color', new THREE.BufferAttribute(ALL_COLORS.subarray(start*3, end*3), 3));
  ptsGeom.attributes.position.needsUpdate = true;
  ptsGeom.attributes.color.needsUpdate = true;

  marker.position.set(GT_MARKER_POS[idx*3], GT_MARKER_POS[idx*3+1], GT_MARKER_POS[idx*3+2]);
  marker.material.color.set(GT_MARKER_HIT[idx] ? 0x22a622 : 0xff9900);

  document.getElementById('frameLabel').textContent =
    `frame ${{idx}} / {max(n_frames-1,0)}, t=${{FRAME_TIMES[idx] !== undefined ? FRAME_TIMES[idx].toFixed(2) : '?'}}s, points=${{FRAME_COUNTS[idx]}}, hit=${{GT_MARKER_HIT[idx]}}`;
}}
setFrame(0);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
window.addEventListener('resize', () => {{
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
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
    }}, 100);
  }} else {{ clearInterval(playTimer); }}
}});
</script></body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="lidar_360", choices=["lidar_360", "livox_avia"])
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--match-radius", type=float, default=1.0,
                    help="how close a detected dynamic point must be to GT to count as a hit")
    ap.add_argument("--min-range", type=float, default=0.2)
    ap.add_argument("--max-range", type=float, default=100.0)
    ap.add_argument("--out", default="mmaud_overlay.html")
    args = ap.parse_args()

    print(f"=== loading {len(CHRONOLOGICAL_SEQS)} clips, chained chronologically ===")
    frames = []
    t_offset = 0.0
    for seq in CHRONOLOGICAL_SEQS:
        seq_frames = load_frames(seq, args.topic)
        t0 = seq_frames[0][0]
        for t, pts in seq_frames:
            frames.append((t, t - t0 + t_offset, pts))  # (real_t, relative_t, pts)
        t_offset += (seq_frames[-1][0] - t0) + 0.1
        print(f"  {seq}: {len(seq_frames)} frames")
    print(f"total: {len(frames)} frames chained")

    gt = pd.read_csv(GT_CSV)
    gt["pos"] = gt.Position.apply(parse_position)

    dm = dufomap(args.resolution, args.d_s, args.d_p, num_threads=0)
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # stationary sensor, identity pose

    tp = fn = fp = 0
    seg_pts, seg_colors, seg_times = [], [], []
    gt_marker_pos, gt_marker_hit = [], []

    for i, (real_t, rel_t, pts) in enumerate(frames):
        d = np.linalg.norm(pts, axis=1)
        pts_f = pts[(d > args.min_range) & (d < args.max_range)]

        if i == 0 or len(pts_f) == 0:
            labels = np.zeros(len(pts_f), dtype=np.uint8)
        else:
            labels = dm.segment(pts_f, pose, cloud_transform=False)
        is_det = labels.astype(bool)
        # Only integrate points NOT classified dynamic this frame into the map --
        # unlike Swarmbag (fixed physical near-field region always excluded), MMAUD's
        # target ranges from ~2m to ~60m so there's no fixed region to exclude. Feeding
        # a point's own map-confirming evidence back in right after calling it dynamic
        # is exactly how a slow/hovering target gets absorbed into "background" within
        # a couple of frames; self-excluding by this frame's own classification avoids
        # that without needing any spatial prior on where the target is.
        dm.run(pts_f[~is_det], pose, cloud_transform=False)

        dyn_pts = pts_f[is_det]

        # nearest-time GT match (MMAUD has a real, single, synced clock)
        gi = (gt.Timestamp - real_t).abs().idxmin()
        gp = gt.loc[gi, "pos"]

        hit = False
        near_gt_mask = np.zeros(len(pts_f), dtype=bool)
        if len(dyn_pts) > 0:
            dists = np.linalg.norm(dyn_pts - gp, axis=1)
            near_gt_mask_dyn = dists < args.match_radius
            hit = bool(near_gt_mask_dyn.any())
            near_gt_mask[is_det] = near_gt_mask_dyn
        if hit:
            tp += 1
        else:
            fn += 1
        fp += int((is_det & ~near_gt_mask).sum())

        colors = np.tile(np.array([0.75, 0.75, 0.78]), (len(pts_f), 1))
        colors[is_det & near_gt_mask] = [0.13, 0.65, 0.13]
        colors[is_det & ~near_gt_mask] = [1.0, 0.15, 0.1]

        seg_pts.append(pts_f)
        seg_colors.append(colors)
        seg_times.append(rel_t)
        gt_marker_pos.append(gp)
        gt_marker_hit.append(hit)

        if i % 100 == 0:
            print(f"  frame {i}/{len(frames)} t={rel_t:.1f}s n_pts={len(pts_f)} "
                  f"n_dynamic={int(is_det.sum())} hit={hit}")

    print(f"\n=== frame-level recall={tp/max(tp+fn,1):.3f} precision={tp/max(tp+fp,1):.3f} "
          f"(TP={tp} FN={fn} FP={fp}, {len(frames)} frames) ===")

    write_overlay_html(args.out, seg_pts, seg_colors, seg_times,
                       gt_marker_pos, gt_marker_hit, tp, fn, fp)


if __name__ == "__main__":
    main()
