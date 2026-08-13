#!/usr/bin/env python3
"""Same shape as visualize_confusion.py's write_segment_html (one Three.js HTML page
per fixed-length time segment, left panel = current frame, right panel = accumulated
over the segment, shared camera, play/slider) but without the ground-truth confusion
categories -- this bag has no reference map / other-drone trajectory to compare
against, so points are colored by near-field (orange) vs far-field (gray) instead of
TP/FN/FP. Per-frame point budget: near+far combined, randomly subsampled down to
--target-total (default 5000, matching far_field_sampling's own per-frame budget in
config/mid360.yaml) -- this is a display-only cap, doesn't touch the underlying data.

Usage: python3 visualize_frame_and_map.py <bag_path> <out_dir> [--segment-seconds 30]
"""
import argparse
import json
import os
from collections import deque

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from dufomap import dufomap

MIN_RANGE, MAX_RANGE = 0.1, 50.0


def read_topic(bag_path, topic, msg_type, storage_id="mcap"):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while r.has_next():
        _, data, t = r.read_next()
        msg = deserialize_message(data, msg_type)
        hs = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9 if hasattr(msg, "header") else t / 1e9
        out.append((hs, msg))
    out.sort(key=lambda item: item[0])
    return out


def cloud_to_xyz(msg):
    pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.dtype.names is not None:
        return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
    return pts.astype(np.float32)


def flat(a):
    return np.asarray(a, dtype=np.float64).flatten().round(4).tolist()


def write_segment_html(out_html, seg_near_pts, seg_near_colors, seg_cluster_pts,
                        seg_cluster_colors, seg_gray_pts, seg_gray_colors, seg_times,
                        traj, seg_lo, seg_hi, n_near_total, n_cluster_total):
    """Left panel: this frame's points, causally-detected-dynamic near-field points
    highlighted red (larger, rendered on top), everything else (static near-field +
    far-field context) gray. Right panel: same, accumulated over the segment. This is
    the detector's real output (dm.segment()/dm.run() called causally, frame 0 has no
    history yet by convention), not just near-field zone membership -- being *in* the
    near-field cylinder doesn't mean *detected as dynamic*; most of it is static
    background (walls, floor) that the causal detector correctly leaves gray. A third
    category (orange) is the cluster-supplement signal -- points segment() called
    static that a small-isolated-persistent-blob heuristic flags as candidate dynamic
    anyway, since segment()'s free-space-history check structurally can't catch a
    target that's been sitting still since before mapping started (see argparse help
    in main() for the tuning/validation behind this)."""
    near_frame_counts = [len(p) for p in seg_near_pts]
    near_cum_counts = np.cumsum(near_frame_counts).tolist() if near_frame_counts else [0]
    all_near_pts = np.concatenate(seg_near_pts, axis=0) if seg_near_pts else np.zeros((0, 3))
    all_near_colors = np.concatenate(seg_near_colors, axis=0) if seg_near_colors else np.zeros((0, 3))

    cluster_frame_counts = [len(p) for p in seg_cluster_pts]
    cluster_cum_counts = np.cumsum(cluster_frame_counts).tolist() if cluster_frame_counts else [0]
    all_cluster_pts = np.concatenate(seg_cluster_pts, axis=0) if seg_cluster_pts else np.zeros((0, 3))
    all_cluster_colors = np.concatenate(seg_cluster_colors, axis=0) if seg_cluster_colors else np.zeros((0, 3))

    gray_frame_counts = [len(p) for p in seg_gray_pts]
    gray_cum_counts = np.cumsum(gray_frame_counts).tolist() if gray_frame_counts else [0]
    all_gray_pts = np.concatenate(seg_gray_pts, axis=0) if seg_gray_pts else np.zeros((0, 3))
    all_gray_colors = np.concatenate(seg_gray_colors, axis=0) if seg_gray_colors else np.zeros((0, 3))

    bbox_src = np.concatenate([all_near_pts, all_cluster_pts, all_gray_pts, traj], axis=0) \
        if (len(all_near_pts) + len(all_cluster_pts) + len(all_gray_pts)) else traj
    if len(bbox_src) == 0:
        bbox_src = traj
    mins, maxs = bbox_src.min(axis=0), bbox_src.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)
    n_frames_disp = len(seg_times)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>[{seg_lo},{seg_hi})s -- {n_near_total} detected-dynamic pts, {n_cluster_total} cluster-supplement pts</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 8px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 14px; align-items:center; flex-wrap: wrap; font-size:13px; }}
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
  <span><b>[{seg_lo},{seg_hi})s</b></span>
  <span><span class="swatch" style="background:#ff2619;"></span> Detected dynamic (causal, history up to this frame only)</span>
  <span><span class="swatch" style="background:#ff9900;"></span> Cluster supplement (small/isolated/persistent blob segment() missed -- see docs/lidar1_segments/index.html)</span>
  <span><span class="swatch" style="background:#cccccc;"></span> Static (near-field background + far-field context)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> Trajectory</span>
  <span style="color:#888;">{n_near_total} detected-dynamic + {n_cluster_total} cluster-supplement pts this segment, ~5000 pts/frame budget (near+far); frame 0: cold start, no history yet, static by convention</span>
</div>
<div id="labelControls" style="padding:8px 20px;background:#fff7e0;border-bottom:1px solid #e0c060;display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px;">
  <b>Labeling</b> (click = single point; <b>Shift+drag</b> = box-select a whole cluster; either panel works, right/accumulated auto-resolves which frame each point came from; plain drag still orbits):
  <label><input type="radio" name="labelMode" value="off" checked> Off</label>
  <label><input type="radio" name="labelMode" value="fn"> Missed detection (FN) <span class="swatch" style="background:#ffd400;"></span></label>
  <label><input type="radio" name="labelMode" value="fp"> Wrong detection (FP) <span class="swatch" style="background:#ff00e6;"></span></label>
  <span>pick radius (m): <input type="number" id="pickRadius" value="0.15" min="0.02" max="2" step="0.01" style="width:60px;"></span>
  <button id="undoBtn">Undo last</button>
  <button id="clearBtn">Clear all</button>
  <button id="exportBtn">Export labels (JSON)</button>
  <span id="labelCounts" style="color:#555;">FN: 0, FP: 0</span>
</div>
<div id="labelList" style="max-height:120px;overflow-y:auto;padding:4px 20px;font-size:12px;background:#fffdf5;border-bottom:1px solid #eee;display:none;"></div>
<div id="panels">
  <div class="panel" id="leftCanvas"><div class="panel-title">Current frame</div></div>
  <div class="panel" id="rightCanvas"><div class="panel-title">Accumulated (this segment)</div></div>
  <div id="syncOverlay"></div>
  <div id="selectionBox" style="position:absolute;border:2px dashed #ff8800;background:rgba(255,136,0,0.15);display:none;pointer-events:none;z-index:25;"></div>
</div>
<div id="depthPanel" style="display:none;position:fixed;right:20px;bottom:90px;z-index:60;background:#fff;border:2px solid #ff8800;border-radius:6px;padding:12px 16px;font-size:13px;box-shadow:0 2px 10px rgba(0,0,0,0.25);width:280px;">
  <b>Trim depth (drag box also grabs points behind the cluster)</b>
  <div style="color:#888;margin:4px 0;">Cyan preview = will be labeled. Narrow near/far to drop background.</div>
  <div style="display:flex;gap:8px;align-items:center;margin:6px 0;">
    <label style="flex:1;">near (m) <input type="number" id="depthNear" step="0.05" style="width:70px;"></label>
    <label style="flex:1;">far (m) <input type="number" id="depthFar" step="0.05" style="width:70px;"></label>
  </div>
  <div id="depthCount" style="color:#555;margin-bottom:8px;">0 points selected</div>
  <div style="display:flex;gap:8px;justify-content:flex-end;">
    <button id="depthResetBtn">Reset range</button>
    <button id="depthCancelBtn">Cancel</button>
    <button id="depthApplyBtn" style="font-weight:bold;">Apply</button>
  </div>
</div>
<div id="sliderRow">
  <button id="playBtn">&#9654; Play</button>
  <input type="range" id="frameSlider" min="0" max="{max(n_frames_disp-1,0)}" value="0">
  <span id="frameLabel">frame 0 / {max(n_frames_disp-1,0)}</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const ALL_NEAR_PTS = new Float32Array({json.dumps(flat(all_near_pts))});
const ALL_NEAR_COLORS = new Float32Array({json.dumps(flat(all_near_colors))});
const NEAR_FRAME_COUNTS = {json.dumps(near_frame_counts)};
const NEAR_CUM_COUNTS = {json.dumps(near_cum_counts)};
const ALL_CLUSTER_PTS = new Float32Array({json.dumps(flat(all_cluster_pts))});
const ALL_CLUSTER_COLORS = new Float32Array({json.dumps(flat(all_cluster_colors))});
const CLUSTER_FRAME_COUNTS = {json.dumps(cluster_frame_counts)};
const CLUSTER_CUM_COUNTS = {json.dumps(cluster_cum_counts)};
const ALL_GRAY_PTS = new Float32Array({json.dumps(flat(all_gray_pts))});
const ALL_GRAY_COLORS = new Float32Array({json.dumps(flat(all_gray_colors))});
const GRAY_FRAME_COUNTS = {json.dumps(gray_frame_counts)};
const GRAY_CUM_COUNTS = {json.dumps(gray_cum_counts)};
const FRAME_TIMES = {json.dumps([round(t, 3) for t in seg_times])};
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

const curGrayGeom = new THREE.BufferGeometry();
const curGrayMat = new THREE.PointsMaterial({{ size: 0.05, vertexColors: true }});
const curGrayPoints = new THREE.Points(curGrayGeom, curGrayMat);
left.scene.add(curGrayPoints);
const accGrayGeom = new THREE.BufferGeometry();
const accGrayMat = new THREE.PointsMaterial({{ size: 0.05, vertexColors: true }});
const accGrayPoints = new THREE.Points(accGrayGeom, accGrayMat);
right.scene.add(accGrayPoints);

const curNearGeom = new THREE.BufferGeometry();
const curNearMat = new THREE.PointsMaterial({{ size: 0.12, vertexColors: true, sizeAttenuation: true }});
const curNearPoints = new THREE.Points(curNearGeom, curNearMat);
left.scene.add(curNearPoints);
const accNearGeom = new THREE.BufferGeometry();
const accNearMat = new THREE.PointsMaterial({{ size: 0.12, vertexColors: true, sizeAttenuation: true }});
const accNearPoints = new THREE.Points(accNearGeom, accNearMat);
right.scene.add(accNearPoints);

const curClusterGeom = new THREE.BufferGeometry();
const curClusterMat = new THREE.PointsMaterial({{ size: 0.12, vertexColors: true, sizeAttenuation: true }});
const curClusterPoints = new THREE.Points(curClusterGeom, curClusterMat);
left.scene.add(curClusterPoints);
const accClusterGeom = new THREE.BufferGeometry();
const accClusterMat = new THREE.PointsMaterial({{ size: 0.12, vertexColors: true, sizeAttenuation: true }});
const accClusterPoints = new THREE.Points(accClusterGeom, accClusterMat);
right.scene.add(accClusterPoints);

function sliceInto(geom, allPts, allColors, start, end) {{
  geom.setAttribute('position', new THREE.BufferAttribute(allPts.subarray(start*3, end*3), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(allColors.subarray(start*3, end*3), 3));
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
}}

function setFrame(idx) {{
  if (NEAR_FRAME_COUNTS.length === 0) return;
  const nStart = idx === 0 ? 0 : NEAR_CUM_COUNTS[idx-1];
  const nEnd = NEAR_CUM_COUNTS[idx];
  const cStart = idx === 0 ? 0 : CLUSTER_CUM_COUNTS[idx-1];
  const cEnd = CLUSTER_CUM_COUNTS[idx];
  const gStart = idx === 0 ? 0 : GRAY_CUM_COUNTS[idx-1];
  const gEnd = GRAY_CUM_COUNTS[idx];

  sliceInto(curNearGeom, ALL_NEAR_PTS, ALL_NEAR_COLORS, nStart, nEnd);
  sliceInto(curClusterGeom, ALL_CLUSTER_PTS, ALL_CLUSTER_COLORS, cStart, cEnd);
  sliceInto(curGrayGeom, ALL_GRAY_PTS, ALL_GRAY_COLORS, gStart, gEnd);
  sliceInto(accNearGeom, ALL_NEAR_PTS, ALL_NEAR_COLORS, 0, nEnd);
  sliceInto(accClusterGeom, ALL_CLUSTER_PTS, ALL_CLUSTER_COLORS, 0, cEnd);
  sliceInto(accGrayGeom, ALL_GRAY_PTS, ALL_GRAY_COLORS, 0, gEnd);

  const nPts = NEAR_FRAME_COUNTS[idx] + CLUSTER_FRAME_COUNTS[idx] + GRAY_FRAME_COUNTS[idx];
  document.getElementById('frameLabel').textContent =
    `frame ${{idx}} / {max(n_frames_disp-1,0)}, t=${{FRAME_TIMES[idx] !== undefined ? FRAME_TIMES[idx].toFixed(2) : '?'}}s, points=${{nPts}} (${{NEAR_FRAME_COUNTS[idx]}} dynamic, ${{CLUSTER_FRAME_COUNTS[idx]}} cluster-supplement)`;
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

// ---- Interactive labeling: click a point to mark it as a missed detection (FN,
// should be red but wasn't) or a wrong detection (FP, is red but shouldn't be); or
// Shift+drag a box to label a whole cluster at once (dynamic objects show up as
// clusters, not lone points). Labels are stored as (buffer, flat-index) and applied
// directly as a color overwrite on the shared ALL_NEAR_COLORS/ALL_GRAY_COLORS arrays
// -- curXxxGeom and accXxxGeom attributes are subarray() VIEWS into those same arrays
// (not copies), so a write through one is visible in both panels and survives frame
// changes without any extra bookkeeping. Undo restores the original color.
//
// Frame attribution: the right/accumulated panel's geometries always span [0, cumEnd)
// starting at 0, so a point's position there IS its global flat index -- binary-search
// that against the per-frame cumulative-count array (same one setFrame() slices with)
// to recover which frame it came from. The left/current-frame panel only ever shows
// one frame, so every point there belongs to whatever the slider is on.
const raycaster = new THREE.Raycaster();
let labelMode = 'off';
document.querySelectorAll('input[name="labelMode"]').forEach(r => {{
  r.addEventListener('change', () => {{ labelMode = r.value; }});
}});

function labelColorHex(kind) {{ return kind === 'fn' ? 0xffd400 : 0xff00e6; }}
const FN_COLOR = new THREE.Color(0xffd400), FP_COLOR = new THREE.Color(0xff00e6);

function markGeomsDirty() {{
  curNearGeom.attributes.color.needsUpdate = true;
  curClusterGeom.attributes.color.needsUpdate = true;
  curGrayGeom.attributes.color.needsUpdate = true;
  accNearGeom.attributes.color.needsUpdate = true;
  accClusterGeom.attributes.color.needsUpdate = true;
  accGrayGeom.attributes.color.needsUpdate = true;
}}

function colorArrFor(bufName) {{
  if (bufName === 'near') return ALL_NEAR_COLORS;
  if (bufName === 'cluster') return ALL_CLUSTER_COLORS;
  return ALL_GRAY_COLORS;
}}

const labels = [];
function addLabelAtIndex(bufName, flatIdx, frame, t, x, y, z, kind, panel) {{
  const arr = colorArrFor(bufName);
  const orig = [arr[flatIdx*3], arr[flatIdx*3+1], arr[flatIdx*3+2]];
  const c = kind === 'fn' ? FN_COLOR : FP_COLOR;
  arr[flatIdx*3] = c.r; arr[flatIdx*3+1] = c.g; arr[flatIdx*3+2] = c.b;
  labels.push({{ frame, t, x, y, z, kind, panel, buf: bufName, flatIdx, orig }});
}}

function renderLabelList() {{
  const counts = {{ fn: 0, fp: 0 }};
  labels.forEach(l => counts[l.kind]++);
  document.getElementById('labelCounts').textContent = `FN: ${{counts.fn}}, FP: ${{counts.fp}}`;
  const listEl = document.getElementById('labelList');
  if (labels.length === 0) {{ listEl.style.display = 'none'; listEl.innerHTML = ''; return; }}
  listEl.style.display = 'block';
  listEl.innerHTML = labels.map((l, i) =>
    `<div>#${{i}} <b>${{l.kind.toUpperCase()}}</b> frame ${{l.frame}} t=${{l.t.toFixed(2)}}s (${{l.x.toFixed(2)}}, ${{l.y.toFixed(2)}}, ${{l.z.toFixed(2)}}) [${{l.panel}}]` +
    ` <a href="#" data-i="${{i}}" class="rmLabel" style="color:#c00;margin-left:6px;">remove</a></div>`
  ).join('');
  listEl.querySelectorAll('.rmLabel').forEach(a => {{
    a.addEventListener('click', (e) => {{ e.preventDefault(); removeLabel(parseInt(a.dataset.i)); }});
  }});
}}

function removeLabel(i) {{
  const l = labels[i];
  if (!l) return;
  const arr = l.buf === 'near' ? ALL_NEAR_COLORS : ALL_GRAY_COLORS;
  arr[l.flatIdx*3] = l.orig[0]; arr[l.flatIdx*3+1] = l.orig[1]; arr[l.flatIdx*3+2] = l.orig[2];
  labels.splice(i, 1);
  markGeomsDirty();
  renderLabelList();
}}

document.getElementById('undoBtn').addEventListener('click', () => {{ if (labels.length) removeLabel(labels.length - 1); }});
document.getElementById('clearBtn').addEventListener('click', () => {{ while (labels.length) removeLabel(0); }});
document.getElementById('exportBtn').addEventListener('click', () => {{
  const payload = {{
    segment_seconds: [{seg_lo}, {seg_hi}],
    labels: labels.map(l => ({{ frame: l.frame, t: l.t, x: l.x, y: l.y, z: l.z, kind: l.kind, panel: l.panel }})),
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'labels_seg_{seg_lo:04d}_{seg_hi:04d}.json';
  a.click();
  URL.revokeObjectURL(url);
}});

function frameForFlatIndex(flatIdx, cumCounts) {{
  let lo = 0, hi = cumCounts.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (cumCounts[mid] > flatIdx) hi = mid; else lo = mid + 1;
  }}
  return lo;
}}

function panelAt(clientX, clientY) {{
  const leftRect = left.container.getBoundingClientRect();
  const rightRect = right.container.getBoundingClientRect();
  if (clientX >= leftRect.left && clientX <= leftRect.right && clientY >= leftRect.top && clientY <= leftRect.bottom) {{
    return {{ panel: 'left', rect: leftRect }};
  }}
  if (clientX >= rightRect.left && clientX <= rightRect.right && clientY >= rightRect.top && clientY <= rightRect.bottom) {{
    return {{ panel: 'right', rect: rightRect }};
  }}
  return null;
}}

// panel -> near/gray slices of the global flat arrays visible in that panel right now
// i.e. which slice of the global flat arrays is actually visible in that panel right now.
function visibleRanges(panel) {{
  const idx = parseInt(slider.value);
  const nEnd = NEAR_CUM_COUNTS[idx], cEnd = CLUSTER_CUM_COUNTS[idx], gEnd = GRAY_CUM_COUNTS[idx];
  if (panel === 'left') {{
    const nStart = idx === 0 ? 0 : NEAR_CUM_COUNTS[idx-1];
    const cStart = idx === 0 ? 0 : CLUSTER_CUM_COUNTS[idx-1];
    const gStart = idx === 0 ? 0 : GRAY_CUM_COUNTS[idx-1];
    return {{ near: [nStart, nEnd], cluster: [cStart, cEnd], gray: [gStart, gEnd] }};
  }}
  return {{ near: [0, nEnd], cluster: [0, cEnd], gray: [0, gEnd] }};
}}

let downPos = null, downButton = -1, downShift = false, boxGesture = null;
const selectionBox = document.getElementById('selectionBox');
const panelsEl = document.getElementById('panels');

syncOverlay.addEventListener('pointerdown', (e) => {{
  if (pendingCandidates) return; // depth-trim panel open, ignore new gestures until resolved
  downPos = [e.clientX, e.clientY];
  downButton = e.button;
  downShift = e.shiftKey;
  if (downShift && downButton === 0 && labelMode !== 'off') {{
    const hit = panelAt(e.clientX, e.clientY);
    if (hit) {{
      boxGesture = hit;
      sharedControls.enabled = false;
      selectionBox.style.display = 'block';
    }}
  }}
}});

syncOverlay.addEventListener('pointermove', (e) => {{
  if (!boxGesture || !downPos) return;
  const panelsRect = panelsEl.getBoundingClientRect();
  const x1 = downPos[0], y1 = downPos[1], x2 = e.clientX, y2 = e.clientY;
  selectionBox.style.left = (Math.min(x1, x2) - panelsRect.left) + 'px';
  selectionBox.style.top = (Math.min(y1, y2) - panelsRect.top) + 'px';
  selectionBox.style.width = Math.abs(x2 - x1) + 'px';
  selectionBox.style.height = Math.abs(y2 - y1) + 'px';
}});

syncOverlay.addEventListener('pointerup', (e) => {{
  if (!downPos) return;
  const startPos = downPos;
  const dx = e.clientX - startPos[0], dy = e.clientY - startPos[1];
  downPos = null;

  if (boxGesture) {{
    const {{ panel, rect }} = boxGesture;
    boxGesture = null;
    sharedControls.enabled = true;
    selectionBox.style.display = 'none';
    if (Math.hypot(dx, dy) < 4) return; // negligible drag, ignore
    doBoxSelect(panel, rect, {{ x1: startPos[0], y1: startPos[1], x2: e.clientX, y2: e.clientY }});
    return;
  }}

  if (downButton !== 0 || labelMode === 'off') return;
  if (Math.hypot(dx, dy) > 4) return; // was a drag/orbit, not a click

  const hit0 = panelAt(e.clientX, e.clientY);
  if (!hit0) return;
  const {{ panel, rect }} = hit0;
  const objs = panel === 'left'
    ? [curNearPoints, curClusterPoints, curGrayPoints]
    : [accNearPoints, accClusterPoints, accGrayPoints];
  const ndc = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.params.Points.threshold = parseFloat(document.getElementById('pickRadius').value) || 0.15;
  raycaster.setFromCamera(ndc, sharedCamera);
  const hits = raycaster.intersectObjects(objs);
  if (!hits.length) return;
  const hit = hits[0];
  const pt = hit.point;
  let bufName = 'gray';
  if (hit.object === curNearPoints || hit.object === accNearPoints) bufName = 'near';
  else if (hit.object === curClusterPoints || hit.object === accClusterPoints) bufName = 'cluster';
  const cumFor = {{ near: NEAR_CUM_COUNTS, cluster: CLUSTER_CUM_COUNTS, gray: GRAY_CUM_COUNTS }};

  let flatIdx, frameIdx;
  if (panel === 'left') {{
    const idx = parseInt(slider.value);
    const vis = visibleRanges('left');
    flatIdx = vis[bufName][0] + hit.index;
    frameIdx = idx;
  }} else {{
    flatIdx = hit.index;
    frameIdx = frameForFlatIndex(flatIdx, cumFor[bufName]);
  }}
  const t = FRAME_TIMES[frameIdx] !== undefined ? FRAME_TIMES[frameIdx] : 0;
  addLabelAtIndex(bufName, flatIdx, frameIdx, t, pt.x, pt.y, pt.z, labelMode, panel);
  markGeomsDirty();
  renderLabelList();
}});

// Projects every point currently visible in `panel` through the shared camera and
// keeps the ones landing inside the drawn box (in NDC space, relative to that panel's
// own rect -- not the combined #panels rect). Cost is one Vector3.project() per
// candidate point; for the accumulated panel on a busy segment that can be a few
// hundred thousand points, which is a noticeable but one-off synchronous pause, not a
// per-frame cost.
// A screen-space rect actually selects a view FRUSTUM (everything between the near and
// far clip planes projecting into that rect), which grabs background points sitting
// behind the intended cluster along the same line of sight -- e.g. a wall behind a
// person. Rather than commit immediately, gather every candidate with its camera
// distance, then let the user trim near/far cutoffs in a small panel (with a live cyan
// preview) before anything is actually labeled -- this turns the 2D drag + depth trim
// into an effective 3D box (frustum slab) selection without needing a true 3D drag
// gesture, which a 2D mouse can't express directly.
let pendingCandidates = null, pendingPanel = null;

function doBoxSelect(panel, rect, xy) {{
  const nx1 = ((Math.min(xy.x1, xy.x2) - rect.left) / rect.width) * 2 - 1;
  const nx2 = ((Math.max(xy.x1, xy.x2) - rect.left) / rect.width) * 2 - 1;
  const ny1 = -((Math.max(xy.y1, xy.y2) - rect.top) / rect.height) * 2 + 1;
  const ny2 = -((Math.min(xy.y1, xy.y2) - rect.top) / rect.height) * 2 + 1;

  const vis = visibleRanges(panel);
  const idx = parseInt(slider.value);
  const v = new THREE.Vector3();
  const candidates = [];

  function scan(ptsArr, range, bufName, cumCounts) {{
    const [start, end] = range;
    for (let i = start; i < end; i++) {{
      const x = ptsArr[i*3], y = ptsArr[i*3+1], z = ptsArr[i*3+2];
      v.set(x, y, z);
      v.project(sharedCamera);
      if (v.z < -1 || v.z > 1) continue;
      if (v.x < nx1 || v.x > nx2 || v.y < ny1 || v.y > ny2) continue;
      const frameIdx = panel === 'left' ? idx : frameForFlatIndex(i, cumCounts);
      const t = FRAME_TIMES[frameIdx] !== undefined ? FRAME_TIMES[frameIdx] : 0;
      const dist = sharedCamera.position.distanceTo(new THREE.Vector3(x, y, z));
      const arr = colorArrFor(bufName);
      candidates.push({{ i, bufName, x, y, z, frameIdx, t, dist,
                          orig: [arr[i*3], arr[i*3+1], arr[i*3+2]] }});
    }}
  }}
  scan(ALL_NEAR_PTS, vis.near, 'near', NEAR_CUM_COUNTS);
  scan(ALL_CLUSTER_PTS, vis.cluster, 'cluster', CLUSTER_CUM_COUNTS);
  scan(ALL_GRAY_PTS, vis.gray, 'gray', GRAY_CUM_COUNTS);
  if (candidates.length === 0) return;

  pendingCandidates = candidates;
  pendingPanel = panel;
  openDepthPanel();
}}

const PREVIEW_COLOR = new THREE.Color(0x00eaff);

function applyPreview() {{
  const near = parseFloat(document.getElementById('depthNear').value);
  const far = parseFloat(document.getElementById('depthFar').value);
  let n = 0;
  for (const c of pendingCandidates) {{
    const arr = colorArrFor(c.bufName);
    const inRange = c.dist >= near && c.dist <= far;
    const col = inRange ? PREVIEW_COLOR : {{ r: c.orig[0], g: c.orig[1], b: c.orig[2] }};
    arr[c.i*3] = col.r; arr[c.i*3+1] = col.g; arr[c.i*3+2] = col.b;
    if (inRange) n++;
  }}
  markGeomsDirty();
  document.getElementById('depthCount').textContent = `${{n}} / ${{pendingCandidates.length}} points selected`;
}}

function openDepthPanel() {{
  const dists = pendingCandidates.map(c => c.dist);
  const lo = Math.min(...dists), hi = Math.max(...dists);
  document.getElementById('depthNear').value = (Math.max(lo - 0.01, 0)).toFixed(2);
  document.getElementById('depthFar').value = (hi + 0.01).toFixed(2);
  document.getElementById('depthPanel').style.display = 'block';
  applyPreview();
}}

function closeDepthPanel(revert) {{
  if (revert) {{
    for (const c of pendingCandidates) {{
      const arr = colorArrFor(c.bufName);
      arr[c.i*3] = c.orig[0]; arr[c.i*3+1] = c.orig[1]; arr[c.i*3+2] = c.orig[2];
    }}
    markGeomsDirty();
  }}
  pendingCandidates = null;
  pendingPanel = null;
  document.getElementById('depthPanel').style.display = 'none';
}}

document.getElementById('depthNear').addEventListener('input', applyPreview);
document.getElementById('depthFar').addEventListener('input', applyPreview);
document.getElementById('depthResetBtn').addEventListener('click', openDepthPanel);
document.getElementById('depthCancelBtn').addEventListener('click', () => closeDepthPanel(true));
document.getElementById('depthApplyBtn').addEventListener('click', () => {{
  const near = parseFloat(document.getElementById('depthNear').value);
  const far = parseFloat(document.getElementById('depthFar').value);
  const panel = pendingPanel;
  const kept = pendingCandidates.filter(c => c.dist >= near && c.dist <= far);
  // Revert everything to original first (clears the cyan preview), then commit real
  // labels only for the kept subset.
  for (const c of pendingCandidates) {{
    const arr = colorArrFor(c.bufName);
    arr[c.i*3] = c.orig[0]; arr[c.i*3+1] = c.orig[1]; arr[c.i*3+2] = c.orig[2];
  }}
  for (const c of kept) {{
    addLabelAtIndex(c.bufName, c.i, c.frameIdx, c.t, c.x, c.y, c.z, labelMode, panel);
  }}
  pendingCandidates = null;
  pendingPanel = null;
  document.getElementById('depthPanel').style.display = 'none';
  markGeomsDirty();
  renderLabelList();
}});
</script></body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path")
    ap.add_argument("out_dir")
    ap.add_argument("--near-topic", default="/nearfield/deskewed_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--storage-id", default="mcap")
    ap.add_argument("--segment-seconds", type=float, default=30.0)
    ap.add_argument("--target-total", type=int, default=5000,
                     help="per-frame display budget, near+far combined, randomly "
                          "subsampled down to this cap -- matches far_field_sampling's "
                          "own per-frame budget in config/mid360.yaml. Detected-dynamic "
                          "points are never subsampled (they're the point of the plot).")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=None,
                     help="stop after this many seconds (default: whole bag)")
    # Cluster-supplement: segment()'s seenFree() check structurally cannot flag a
    # target that's been sitting in roughly the same spot since before mapping
    # started -- that patch of space was never observed free, so there's never a
    # conflict to detect (see detection/dufomap_custom/README.md). Validated against
    # ~11.5k hand-labeled missed-detection points (docs/lidar1_segments/labels_seg_*.json,
    # 0-120s): recall 90.9%, ~81% of flagged points verified (not noise), at
    # gap=0.6/persist=3 -- tuned for recall since that's the priority for this
    # application (a missed neighbor drone is worse than an extra highlighted point).
    ap.add_argument("--cluster-eps", type=float, default=0.2)
    ap.add_argument("--cluster-min-samples", type=int, default=2)
    ap.add_argument("--max-cluster-pts", type=int, default=60)
    ap.add_argument("--max-bbox-diag", type=float, default=1.0)
    ap.add_argument("--isolation-gap", type=float, default=0.6,
                     help="min distance (m) from candidate cluster to nearest other "
                          "near+far point -- rejects clipped corners of walls/furniture "
                          "that DBSCAN happens to split off as a small blob")
    ap.add_argument("--persist-min-frames", type=int, default=3,
                     help="min number of frames (incl. current) a similarly-positioned "
                          "candidate must appear in within --persist-window seconds")
    ap.add_argument("--persist-radius", type=float, default=0.3)
    ap.add_argument("--persist-window", type=float, default=0.4)
    # Bootstrap static map: most flights hover a few seconds at the start before moving
    # (mentor's suggestion) -- while stationary, far-field returns come from one fixed
    # vantage point, so anything seen there is unambiguously static (background/
    # furniture), no reference-map recording or cross-bag ICP registration needed, same
    # coordinate frame by construction. Used to suppress false dynamic/cluster-supplement
    # flags later in the flight for structure that's already confirmed static from this
    # window -- catches exactly the "platform sees furniture from a new angle it's never
    # observed free from before" false positive found by hand on lidar_2_ros2 (a fixed
    # piece of furniture at x~-7.9 got fragmented into several simultaneously-active
    # false tracks once the platform approached it from a direction not covered by this
    # bootstrap window -- so this fixes cases where the furniture happens to be visible
    # from the start pose, not a universal fix; see also --isolation-gap etc. above).
    ap.add_argument("--bootstrap-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bootstrap-xy-radius", type=float, default=0.2,
                     help="stationary window = longest initial run with horizontal "
                          "(x,y) distance from the start pose under this (m) -- z is "
                          "excluded, it's noisier at rest (see config/mid360.yaml "
                          "plane_thr history) without indicating real motion")
    ap.add_argument("--bootstrap-gap", type=float, default=0.12,
                     help="a later near-field point within this distance (m) of the "
                          "bootstrap static cloud is reclassified as background, "
                          "overriding a dynamic/cluster-supplement flag -- same "
                          "threshold swarm_bgsubtract.py uses against the external "
                          "reference map")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"reading {args.near_topic} / {args.far_topic} / {args.odom_topic} from {args.bag_path}")
    near = read_topic(args.bag_path, args.near_topic, PointCloud2, args.storage_id)
    far = read_topic(args.bag_path, args.far_topic, PointCloud2, args.storage_id)
    odom = read_topic(args.bag_path, args.odom_topic, Odometry, args.storage_id)
    n = min(len(near), len(far), len(odom))
    t0 = near[0][0]
    print(f"near={len(near)} far={len(far)} odom={len(odom)}, using {n} synced frames")

    dm = dufomap(args.voxel, args.d_s, args.d_p, num_threads=0)
    print(f"causal detector: voxel={args.voxel} d_s={args.d_s} d_p={args.d_p}")
    print(f"cluster supplement: eps={args.cluster_eps} isolation_gap={args.isolation_gap} "
          f"persist_min_frames={args.persist_min_frames}")

    bootstrap_tree = None
    if args.bootstrap_enable:
        pos0 = None
        stationary_end = 0.0
        for i in range(n):
            p = odom[i][1].pose.pose.position
            xy = np.array([p.x, p.y], dtype=np.float32)
            if pos0 is None:
                pos0 = xy
            if np.linalg.norm(xy - pos0) < args.bootstrap_xy_radius:
                stationary_end = near[i][0] - t0
            else:
                break
        bootstrap_pts = []
        for i in range(n):
            if near[i][0] - t0 > stationary_end:
                break
            bootstrap_pts.append(cloud_to_xyz(far[i][1]))
        bootstrap_pts = np.concatenate(bootstrap_pts, axis=0) if bootstrap_pts else np.zeros((0, 3))
        n_raw = len(bootstrap_pts)
        if n_raw > 0:
            # Voxel-downsample (grid snap + unique) before building the tree -- a
            # multi-minute stationary window accumulates millions of raw points, way
            # more density than the bootstrap_gap (0.12m) query needs; this cuts the
            # tree size ~20-50x with no change to query accuracy at that threshold.
            grid = np.round(bootstrap_pts / (args.bootstrap_gap / 2)).astype(np.int64)
            _, keep_idx = np.unique(grid, axis=0, return_index=True)
            bootstrap_pts = bootstrap_pts[keep_idx]
            bootstrap_tree = cKDTree(bootstrap_pts)
        print(f"bootstrap static map: stationary window [0, {stationary_end:.2f}]s, "
              f"{n_raw} far-field pts accumulated -> {len(bootstrap_pts)} after voxel "
              f"downsample, gap={args.bootstrap_gap}m")

    cluster_history = deque()  # (t, centroid) of confirmed-candidate clusters, sliding window

    segments = {}
    for i in range(n):
        t_rel = near[i][0] - t0
        if args.max_seconds is not None and t_rel > args.max_seconds:
            break
        near_pts = cloud_to_xyz(near[i][1])
        far_pts = cloud_to_xyz(far[i][1])
        p = odom[i][1].pose.pose.position
        q = odom[i][1].pose.pose.orientation
        pose = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        pose_xyz = np.array([p.x, p.y, p.z], dtype=np.float32)

        dn = np.linalg.norm(near_pts - pose_xyz, axis=1) if len(near_pts) else np.zeros(0)
        near_pts = near_pts[(dn > MIN_RANGE) & (dn < MAX_RANGE)]
        df = np.linalg.norm(far_pts - pose_xyz, axis=1) if len(far_pts) else np.zeros(0)
        far_pts = far_pts[(df > MIN_RANGE) & (df < MAX_RANGE)]

        # Causal: segment() with history built from frames 0..i-1 only, THEN run() adds
        # this frame's far-field to the map -- matches causal_vs_batch.py/causal_live.py
        # exactly. Frame 0 has no history yet (cold start, static by convention).
        if i == 0 or len(near_pts) == 0:
            dyn_labels = np.zeros(len(near_pts), dtype=np.uint8)
        else:
            dyn_labels = dm.segment(near_pts, pose, cloud_transform=False)
        dm.run(far_pts, pose, cloud_transform=False)

        dyn_mask = dyn_labels.astype(bool)

        # Bootstrap static map: anything near a point seen during the initial stationary
        # window is confirmed background (see argparse help above) -- overrides both
        # segment() and the cluster-supplement pass below, since a point can't be both
        # "this exact spot was empty a moment ago" (bootstrap) and "genuinely dynamic".
        bootstrap_bg_mask = np.zeros(len(near_pts), dtype=bool)
        if bootstrap_tree is not None and len(near_pts):
            d_bs, _ = bootstrap_tree.query(near_pts, k=1)
            bootstrap_bg_mask = d_bs < args.bootstrap_gap
            dyn_mask = dyn_mask & ~bootstrap_bg_mask

        # Cluster supplement: a small, isolated (>= isolation_gap from anything else),
        # temporally-persistent blob within near_pts, independent of segment()'s
        # free-space-history check -- catches targets segment() structurally can't
        # (see argparse help above). Only runs against points segment() already called
        # static; points it already called dynamic stay red, not re-colored orange.
        while cluster_history and t_rel - cluster_history[0][0] > args.persist_window:
            cluster_history.popleft()
        cluster_mask = np.zeros(len(near_pts), dtype=bool)
        not_dyn_or_bg = ~dyn_mask & ~bootstrap_bg_mask
        candidate_near = near_pts[not_dyn_or_bg]
        candidate_local_idx = np.where(not_dyn_or_bg)[0]
        if len(candidate_near) >= args.cluster_min_samples:
            db = DBSCAN(eps=args.cluster_eps, min_samples=args.cluster_min_samples).fit(candidate_near)
            labels = db.labels_
            env_pts = np.concatenate([near_pts, far_pts], axis=0) if len(far_pts) else near_pts
            tree = cKDTree(env_pts) if len(env_pts) else None
            for lbl in set(labels):
                if lbl == -1:
                    continue
                mask = labels == lbl
                if mask.sum() > args.max_cluster_pts:
                    continue
                pts = candidate_near[mask]
                if np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) > args.max_bbox_diag:
                    continue
                cluster_local_idx = candidate_local_idx[mask]
                min_gap = np.inf
                if tree is not None:
                    for pt in pts:
                        dists, idxs = tree.query(pt, k=min(mask.sum() + 5, len(env_pts)))
                        for dd, ii in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                            if ii < len(near_pts) and ii in cluster_local_idx:
                                continue
                            min_gap = min(min_gap, dd)
                            break
                if min_gap < args.isolation_gap:
                    continue
                centroid = pts.mean(axis=0)
                persist_count = sum(1 for (ht, hc) in cluster_history
                                     if np.linalg.norm(hc - centroid) < args.persist_radius) + 1
                cluster_history.append((t_rel, centroid))
                if persist_count >= args.persist_min_frames:
                    cluster_mask[cluster_local_idx] = True

        dyn_pts = near_pts[dyn_mask]
        dyn_colors = np.tile(np.array([1.0, 0.15, 0.1]), (len(dyn_pts), 1))

        cluster_pts = near_pts[cluster_mask]
        cluster_colors = np.tile(np.array([1.0, 0.6, 0.0]), (len(cluster_pts), 1))

        static_mask = ~dyn_mask & ~cluster_mask
        static_near_pts = near_pts[static_mask]
        static_pts = np.concatenate([static_near_pts, far_pts], axis=0)
        static_colors = np.tile(np.array([0.78, 0.78, 0.8]), (len(static_pts), 1))

        # Detected-dynamic (red) and cluster-supplement (orange) points always kept in
        # full (they're the point of the plot); everything static (near+far combined)
        # subsampled so the total is ~target_total per frame, matching
        # far_field_sampling's own live per-frame budget -- display-only, doesn't touch
        # any actual detection data (dm.run() above already saw the full-resolution
        # far_pts).
        gray_budget = max(args.target_total - len(dyn_pts) - len(cluster_pts), 0)
        if args.target_total > 0 and len(static_pts) > gray_budget:
            keep = rng.choice(len(static_pts), gray_budget, replace=False)
            static_pts, static_colors = static_pts[keep], static_colors[keep]

        seg_idx = int(t_rel // args.segment_seconds)
        seg = segments.setdefault(seg_idx, {"near_pts": [], "near_colors": [],
                                             "cluster_pts": [], "cluster_colors": [],
                                             "gray_pts": [], "gray_colors": [],
                                             "times": [], "traj": [], "n_near": 0, "n_cluster": 0})
        seg["near_pts"].append(dyn_pts)
        seg["near_colors"].append(dyn_colors)
        seg["cluster_pts"].append(cluster_pts)
        seg["cluster_colors"].append(cluster_colors)
        seg["gray_pts"].append(static_pts)
        seg["gray_colors"].append(static_colors)
        seg["times"].append(t_rel)
        seg["traj"].append(pose_xyz)
        seg["n_near"] += len(dyn_pts)
        seg["n_cluster"] += len(cluster_pts)

        if i % 200 == 0:
            print(f"  frame {i}/{n} t={t_rel:.1f}s seg={seg_idx} dyn={len(dyn_pts)} cluster_supp={len(cluster_pts)}")

    print(f"{len(segments)} segments")
    for seg_idx in sorted(segments.keys()):
        seg = segments[seg_idx]
        seg_lo, seg_hi = seg_idx * args.segment_seconds, (seg_idx + 1) * args.segment_seconds
        traj_arr = np.array(seg["traj"])
        out_html = f"{args.out_dir}/seg_{int(seg_lo):04d}_{int(seg_hi):04d}.html"
        write_segment_html(out_html, seg["near_pts"], seg["near_colors"],
                            seg["cluster_pts"], seg["cluster_colors"],
                            seg["gray_pts"], seg["gray_colors"], seg["times"],
                            traj_arr, int(seg_lo), int(seg_hi), seg["n_near"], seg["n_cluster"])


if __name__ == "__main__":
    main()
