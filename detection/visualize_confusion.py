#!/usr/bin/env python3
"""Visualize the causal detector's output against pseudo-GT, one Three.js HTML file per
fixed-length time segment (map must still build causally from frame 0 across the whole
flight -- only the display is segmented, not the causal pass):
  - left panel: current frame's near+far-field points, TP/FP colored, gray otherwise
  - right panel: same, accumulated over the segment
  - the OTHER flight's trajectory drawn in the same shared (reference-map-registered)
    frame, so a false-positive cluster's location can be checked against where the
    other drone was actually flying
  - an interactive XYZ bounding-box clip (six range sliders), implemented with Three.js
    clipping planes (GPU-side, no geometry rebuild on every slider move -- stays smooth
    even with a few million accumulated points) so a specific region can be isolated

Usage: python3 visualize_confusion.py --host swarm1 [--resolution 0.15]
       [--segment-seconds 60]
"""
import argparse
import json
import os
import time

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


def write_segment_html(out_html, seg_pts, seg_colors, seg_times, traj,
                       tp, fn, fp, host, other_name, seg_lo, seg_hi):
    """Left panel: ONLY this frame's DETECTED (dynamic-labeled) points -- the detector's
    own honest output (TP green / FP red), with all other near-field and far-field
    points shown in gray for spatial context -- no GT-derived oracle markers. Right
    panel: the same, accumulated over time."""
    frame_counts = [len(p) for p in seg_pts]
    cum_counts = np.cumsum(frame_counts).tolist() if frame_counts else [0]
    all_pts = np.concatenate(seg_pts, axis=0) if seg_pts else np.zeros((0, 3))
    all_colors = np.concatenate(seg_colors, axis=0) if seg_colors else np.zeros((0, 3))

    def flat(a):
        return np.asarray(a, dtype=np.float64).flatten().round(4).tolist()

    bbox_src = np.concatenate([all_pts, traj], axis=0) if len(all_pts) else traj
    if len(bbox_src) == 0:
        bbox_src = traj
    mins = bbox_src.min(axis=0)
    maxs = bbox_src.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)
    n_frames_disp = len(seg_pts)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{host} [{seg_lo},{seg_hi})s -- TP={tp} FN={fn} FP={fp}</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 8px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 14px; align-items:center; flex-wrap: wrap; font-size:13px; }}
  .swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; }}
  html, body {{ height:100%; }}
  #bboxRow {{ padding: 6px 20px; background:#fafafa; border-bottom:1px solid #ddd;
              display:flex; gap:18px; align-items:center; flex-wrap:wrap; font-size:12px; }}
  #bboxRow .axis {{ display:flex; gap:6px; align-items:center; }}
  #bboxRow input[type=range] {{ width:120px; }}
  #panels {{ display:flex; width:100%; height: calc(100vh - 150px); position:relative; }}
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
  <span><span class="swatch" style="background:#cccccc;"></span> Background / far-field (context)</span>
  <span><span class="swatch" style="background:#22a622;"></span> TP={tp} (detector says dynamic, GT agrees)</span>
  <span><span class="swatch" style="background:#ff2619;"></span> FP={fp} (detector says dynamic, GT says background)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> {host} own trajectory</span>
  <span style="color:#888;">recall={recall:.3f} (FN={fn} not shown -- by definition not flagged) precision={precision:.3f}</span>
</div>
<div id="bboxRow">
  <span><b>Bounding-box clip:</b></span>
  <div class="axis">X <input type="range" id="xmin"> - <input type="range" id="xmax"> <span id="xlabel"></span></div>
  <div class="axis">Y <input type="range" id="ymin"> - <input type="range" id="ymax"> <span id="ylabel"></span></div>
  <div class="axis">Z <input type="range" id="zmin"> - <input type="range" id="zmax"> <span id="zlabel"></span></div>
  <button id="resetBbox">Reset</button>
</div>
<div id="panels">
  <div class="panel" id="leftCanvas"><div class="panel-title">Current frame</div></div>
  <div class="panel" id="rightCanvas"><div class="panel-title">Accumulated (this segment)</div></div>
  <div id="syncOverlay"></div>
</div>
<div id="sliderRow">
  <button id="playBtn">&#9654; Play</button>
  <input type="range" id="frameSlider" min="0" max="{max(n_frames_disp-1,0)}" value="0">
  <span id="frameLabel">frame 0 / {max(n_frames_disp-1,0)}</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const ALL_PTS = new Float32Array({json.dumps(flat(all_pts))});
const ALL_COLORS = new Float32Array({json.dumps(flat(all_colors))});
const FRAME_COUNTS = {json.dumps(frame_counts)};
const CUM_COUNTS = {json.dumps(cum_counts)};
const FRAME_TIMES = {json.dumps([round(t, 3) for t in seg_times])};
const TRAJ_FLAT = new Float32Array({json.dumps(flat(traj))});
const BBOX_CENTER = {json.dumps(center)};
const BBOX_RADIUS = {radius};
const DATA_MIN = {json.dumps(mins.tolist())};
const DATA_MAX = {json.dumps(maxs.tolist())};

function makePanel(container) {{
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  scene.add(new THREE.AxesHelper(Math.max(0.5, BBOX_RADIUS * 0.1)));
  scene.add(new THREE.GridHelper(BBOX_RADIUS * 2.5, 20, 0xaaaaaa, 0xdddddd));
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.localClippingEnabled = true;
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  return {{ scene, renderer, container }};
}}
const left = makePanel(document.getElementById('leftCanvas'));
const right = makePanel(document.getElementById('rightCanvas'));

// six clipping planes forming an axis-aligned box; updated live from sliders, no
// geometry rebuild needed (cheap, GPU-side clip test per point)
const clipPlanes = [
  new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),   // x >= xmin
  new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),  // x <= xmax
  new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),   // y >= ymin
  new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),  // y <= ymax
  new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),   // z >= zmin
  new THREE.Plane(new THREE.Vector3(0, 0, -1), 0),  // z <= zmax
];
function updateClipPlanes(xmin, xmax, ymin, ymax, zmin, zmax) {{
  clipPlanes[0].constant = -xmin;
  clipPlanes[1].constant = xmax;
  clipPlanes[2].constant = -ymin;
  clipPlanes[3].constant = ymax;
  clipPlanes[4].constant = -zmin;
  clipPlanes[5].constant = zmax;
}}

const trajGeom = new THREE.BufferGeometry();
trajGeom.setAttribute('position', new THREE.BufferAttribute(TRAJ_FLAT, 3));
const trajMat = new THREE.LineBasicMaterial({{ color: 0x2266ff, clippingPlanes: clipPlanes }});
left.scene.add(new THREE.Line(trajGeom, trajMat));
right.scene.add(new THREE.Line(trajGeom.clone(), trajMat));

const curGeom = new THREE.BufferGeometry();
const curMat = new THREE.PointsMaterial({{ size: 0.09, vertexColors: true, clippingPlanes: clipPlanes }});
const curPoints = new THREE.Points(curGeom, curMat);
left.scene.add(curPoints);

const accGeom = new THREE.BufferGeometry();
const accMat = new THREE.PointsMaterial({{ size: 0.06, vertexColors: true, clippingPlanes: clipPlanes }});
const accPoints = new THREE.Points(accGeom, accMat);
right.scene.add(accPoints);

function setFrame(idx) {{
  if (FRAME_COUNTS.length === 0) return;
  const start = idx === 0 ? 0 : CUM_COUNTS[idx-1];
  const end = CUM_COUNTS[idx];
  curGeom.setAttribute('position', new THREE.BufferAttribute(ALL_PTS.subarray(start*3, end*3), 3));
  curGeom.setAttribute('color', new THREE.BufferAttribute(ALL_COLORS.subarray(start*3, end*3), 3));
  curGeom.attributes.position.needsUpdate = true;
  curGeom.attributes.color.needsUpdate = true;

  accGeom.setAttribute('position', new THREE.BufferAttribute(ALL_PTS.subarray(0, end*3), 3));
  accGeom.setAttribute('color', new THREE.BufferAttribute(ALL_COLORS.subarray(0, end*3), 3));
  accGeom.attributes.position.needsUpdate = true;
  accGeom.attributes.color.needsUpdate = true;

  document.getElementById('frameLabel').textContent =
    `frame ${{idx}} / {max(n_frames_disp-1,0)}, t=${{FRAME_TIMES[idx] !== undefined ? FRAME_TIMES[idx].toFixed(2) : '?'}}s, points=${{FRAME_COUNTS[idx]}}`;
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

// --- bbox range sliders ---
const AXES = ['x','y','z'];
const SLIDER_STEPS = 500;
function setupAxis(axis, idx) {{
  const lo = DATA_MIN[idx], hi = DATA_MAX[idx];
  const minEl = document.getElementById(axis+'min');
  const maxEl = document.getElementById(axis+'max');
  [minEl, maxEl].forEach(el => {{ el.min = 0; el.max = SLIDER_STEPS; el.step = 1; }});
  minEl.value = 0;
  maxEl.value = SLIDER_STEPS;
  function toWorld(v) {{ return lo + (hi - lo) * (v / SLIDER_STEPS); }}
  function update() {{
    let a = parseInt(minEl.value), b = parseInt(maxEl.value);
    if (a > b) {{ [a, b] = [b, a]; }}
    const wa = toWorld(a), wb = toWorld(b);
    document.getElementById(axis+'label').textContent = `[${{wa.toFixed(2)}}, ${{wb.toFixed(2)}}]`;
    return [wa, wb];
  }}
  minEl.addEventListener('input', applyBbox);
  maxEl.addEventListener('input', applyBbox);
  update();
  return update;
}}
let axisUpdaters = {{}};
function applyBbox() {{
  const [xmin, xmax] = axisUpdaters.x();
  const [ymin, ymax] = axisUpdaters.y();
  const [zmin, zmax] = axisUpdaters.z();
  updateClipPlanes(xmin, xmax, ymin, ymax, zmin, zmax);
}}
AXES.forEach((a, i) => {{ axisUpdaters[a] = setupAxis(a, i); }});
applyBbox();
document.getElementById('resetBbox').addEventListener('click', () => {{
  ['x','y','z'].forEach(a => {{
    document.getElementById(a+'min').value = 0;
    document.getElementById(a+'max').value = SLIDER_STEPS;
  }});
  applyBbox();
}});
</script></body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB) TP={tp} FN={fn} FP={fp} "
          f"recall={recall:.3f} precision={precision:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"])
    ap.add_argument("--bag-path", default=None,
                    help="override C2_BAGS[--host] (e.g. to point at a reprocessed bag "
                         "with a different nearfield config, like the cylinder near-field fix)")
    ap.add_argument("--out-suffix", default="",
                    help="appended to output filenames so a rerun doesn't overwrite the baseline")
    ap.add_argument("--near-topic", default="/nearfield/refined_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--d-s", type=float, default=0.2)
    ap.add_argument("--d-p", type=int, default=2)
    ap.add_argument("--segment-seconds", type=float, default=60.0)
    ap.add_argument("--target-total", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    host_name = args.host
    other_name = "swarm2" if host_name == "swarm1" else "swarm1"
    bag_path = args.bag_path or C2_BAGS[host_name]
    out_dir = f"{BASE}/{host_name}_bgsub/segments_bbox{args.out_suffix}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== recovering T_far for {host_name} (C2 bag) via Kabsch ===")
    bg = np.load(f"{BASE}/{host_name}_bgsub/bgsubtract_result.npz")
    traj_aligned_full = bg["traj_aligned"]
    orig_poses_full = read_pose_topic(bag_path, args.odom_topic)
    orig_traj_full = np.array([p[:3] for p in orig_poses_full])
    n_traj = min(len(orig_traj_full), len(traj_aligned_full))
    T_far = kabsch_transform(orig_traj_full[:n_traj], traj_aligned_full[:n_traj])
    R, t = T_far[:3, :3], T_far[:3, 3]

    ref_pcd = o3d.io.read_point_cloud(REFERENCE_MAP)
    ref_tree = scipy.spatial.cKDTree(np.asarray(ref_pcd.points))
    other_bg = np.load(f"{BASE}/{other_name}_bgsub/bgsubtract_result.npz")
    other_traj_aligned = other_bg["traj_aligned"]  # already in the shared reference frame
    other_traj_tree = scipy.spatial.cKDTree(other_traj_aligned)

    print(f"\n=== loading {args.near_topic} / {args.far_topic} / {args.odom_topic} from {bag_path} ===")
    near = read_cloud_topic(bag_path, args.near_topic)
    far = read_cloud_topic(bag_path, args.far_topic)
    poses = orig_poses_full
    n = min(len(near), len(far), len(poses))
    print(f"total frames: {n}, segment length: {args.segment_seconds}s")

    dm = dufomap(args.resolution, args.d_s, args.d_p, num_threads=0)

    t_start = time.time()
    segments = {}

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

        seg_idx = int(t_rel // args.segment_seconds)
        seg = segments.setdefault(seg_idx, {"pts": [], "colors": [], "times": [], "traj": [],
                                            "tp": 0, "fn": 0, "fp": 0})

        aligned_pose = (R @ pose_xyz) + t  # own trajectory point, in shared frame
        seg["traj"].append(aligned_pose)

        if len(near_pts_f) > 0:
            aligned = (R @ near_pts_f.T).T + t
            d_ref, _ = ref_tree.query(aligned, k=1)
            is_fg = d_ref > BG_THRESHOLD
            other_dists, _ = other_traj_tree.query(aligned, k=1)
            is_gt = is_fg & (other_dists < OTHER_DRONE_THRESHOLD)
            is_det = dyn_labels.astype(bool)

            tp_mask = is_gt & is_det
            fn_mask = is_gt & ~is_det
            fp_mask = ~is_fg & is_det
            seg["tp"] += int(tp_mask.sum()); seg["fn"] += int(fn_mask.sum())
            seg["fp"] += int(fp_mask.sum())

            # All near-field points shown for context, gray by default; the detector's
            # own flagged-dynamic points (TP/FP, its real honest output) stand out in
            # color. No GT-derived oracle markers (that was the earlier "other drone
            # trajectory" mistake).
            near_colors = np.tile(np.array([0.7, 0.7, 0.72]), (len(near_pts_f), 1))
            near_colors[tp_mask] = [0.13, 0.65, 0.13]
            near_colors[fp_mask] = [1.0, 0.15, 0.1]
            disp_near = aligned
        else:
            near_colors = np.zeros((0, 3))
            disp_near = np.zeros((0, 3))

        far_pts_disp_aligned = (R @ far_pts_f.T).T + t if len(far_pts_f) else np.zeros((0, 3))
        far_colors_disp = np.tile(np.array([0.88, 0.88, 0.9]), (len(far_pts_disp_aligned), 1))

        disp_pts = np.concatenate([disp_near, far_pts_disp_aligned], axis=0)
        disp_colors = np.concatenate([near_colors, far_colors_disp], axis=0)
        seg["pts"].append(disp_pts)
        seg["colors"].append(disp_colors)
        seg["times"].append(t_rel)

        if i % 500 == 0:
            print(f"  frame {i}/{n} t={t_rel:.1f}s seg={seg_idx}")

    print(f"causal loop done in {time.time()-t_start:.1f}s, {len(segments)} segments")

    for seg_idx in sorted(segments.keys()):
        seg = segments[seg_idx]
        seg_lo, seg_hi = seg_idx * args.segment_seconds, (seg_idx + 1) * args.segment_seconds
        traj_arr = np.array(seg["traj"])

        out_html = f"{out_dir}/{host_name}_seg_{int(seg_lo):04d}_{int(seg_hi):04d}.html"
        write_segment_html(out_html, seg["pts"], seg["colors"], seg["times"],
                           traj_arr, seg["tp"], seg["fn"], seg["fp"],
                           host_name, other_name, int(seg_lo), int(seg_hi))


if __name__ == "__main__":
    main()
