#!/usr/bin/env python3
"""Static sanity-check viewer: one flight's own SLAM map (the accumulated far-field
point cloud built by Point-LIO from that flight's own bag, i.e. the "map this flight
built"), with BOTH flights' trajectories overlaid, all already in the shared
reference-map-registered frame (traj_aligned / far_aligned from
background_subtraction.py's saved bgsubtract_result.npz -- no extra registration
needed here). Point size and an XYZ bounding-box clip are both interactively
adjustable, same clipping-plane approach as visualize_confusion.py.

Purpose: a raw, detector-independent look at where the two flights' trajectories
actually were relative to each other and to the site structure -- useful as a sanity
check against the causal detector's TP/FN/FP output (e.g. checking whether the
trajectories really do come close at the times/places the detector claims).

Usage: python3 trajectory_overlay.py --host swarm1 [--voxel 0.15]
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

BASE = os.environ.get("DETECTION_DATA_DIR", "./data")


def flat(a):
    return np.asarray(a, dtype=np.float64).flatten().round(4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="swarm1", choices=["swarm1", "swarm2"],
                    help="whose own SLAM map to show as the background scene")
    ap.add_argument("--voxel", type=float, default=0.15,
                    help="voxel downsample size (m) for the background map -- the raw "
                         "accumulated far-field cloud is tens of millions of points, "
                         "far too many to embed directly")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    host_name = args.host
    other_name = "swarm2" if host_name == "swarm1" else "swarm1"
    host = np.load(f"{BASE}/{host_name}_bgsub/bgsubtract_result.npz")
    other = np.load(f"{BASE}/{other_name}_bgsub/bgsubtract_result.npz")
    out_html = args.out or f"{BASE}/{host_name}_bgsub/trajectory_overlay_{host_name}.html"

    host_traj = host["traj_aligned"]
    other_traj = other["traj_aligned"]

    print(f"downsampling {host_name}'s own accumulated map "
          f"({len(host['far_aligned'])} pts) at voxel={args.voxel}m ...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(host["far_aligned"])
    map_pts = np.asarray(pcd.voxel_down_sample(args.voxel).points)
    print(f"-> {len(map_pts)} pts")

    bbox_src = np.concatenate([map_pts, host_traj, other_traj], axis=0)
    mins, maxs = bbox_src.min(axis=0), bbox_src.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Trajectory overlay -- map from {host_name}'s own SLAM, both trajectories highlighted</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 8px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 16px; align-items:center; flex-wrap: wrap; font-size:13px; }}
  .swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; }}
  html, body {{ height:100%; }}
  #bboxRow {{ padding: 6px 20px; background:#fafafa; border-bottom:1px solid #ddd;
              display:flex; gap:18px; align-items:center; flex-wrap:wrap; font-size:12px; }}
  #bboxRow .axis {{ display:flex; gap:6px; align-items:center; }}
  #bboxRow input[type=range] {{ width:120px; }}
  #view {{ width:100%; height: calc(100vh - 92px); }}
  canvas {{ display:block; }}
</style></head>
<body>
<div id="controls">
  <span><span class="swatch" style="background:#999999;"></span> {host_name}'s own SLAM map ({len(map_pts)} pts, voxel={args.voxel}m)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> {host_name} trajectory ({len(host_traj)} poses)</span>
  <span><span class="swatch" style="background:#ff2619;"></span> {other_name} trajectory ({len(other_traj)} poses)</span>
  <span>Map pt size <input type="range" id="mapSize" min="0.01" max="0.15" step="0.005" value="0.03"> <span id="mapSizeLabel"></span></span>
  <span>Traj pt size <input type="range" id="trajSize" min="0.02" max="0.4" step="0.01" value="0.08"> <span id="trajSizeLabel"></span></span>
</div>
<div id="bboxRow">
  <span><b>Bounding-box clip:</b></span>
  <div class="axis">X <input type="range" id="xmin"> - <input type="range" id="xmax"> <span id="xlabel"></span></div>
  <div class="axis">Y <input type="range" id="ymin"> - <input type="range" id="ymax"> <span id="ylabel"></span></div>
  <div class="axis">Z <input type="range" id="zmin"> - <input type="range" id="zmax"> <span id="zlabel"></span></div>
  <button id="resetBbox">Reset</button>
</div>
<div id="view"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const MAP_FLAT = new Float32Array({json.dumps(flat(map_pts))});
const HOST_TRAJ_FLAT = new Float32Array({json.dumps(flat(host_traj))});
const OTHER_TRAJ_FLAT = new Float32Array({json.dumps(flat(other_traj))});
const BBOX_CENTER = {json.dumps(center)};
const BBOX_RADIUS = {radius};
const DATA_MIN = {json.dumps(mins.tolist())};
const DATA_MAX = {json.dumps(maxs.tolist())};

const container = document.getElementById('view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
scene.add(new THREE.AxesHelper(Math.max(0.5, BBOX_RADIUS * 0.1)));
scene.add(new THREE.GridHelper(BBOX_RADIUS * 2.5, 20, 0xaaaaaa, 0xdddddd));

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.localClippingEnabled = true;
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const [cx, cy, cz] = BBOX_CENTER;
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 2000);
camera.up.set(0, 0, 1);
camera.position.set(cx + BBOX_RADIUS, cy - BBOX_RADIUS, cz + BBOX_RADIUS * 0.6);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);

// six clipping planes forming an axis-aligned box, GPU-side (no geometry rebuild)
const clipPlanes = [
  new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
  new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
  new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),
  new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
  new THREE.Plane(new THREE.Vector3(0, 0, -1), 0),
];
function updateClipPlanes(xmin, xmax, ymin, ymax, zmin, zmax) {{
  clipPlanes[0].constant = -xmin; clipPlanes[1].constant = xmax;
  clipPlanes[2].constant = -ymin; clipPlanes[3].constant = ymax;
  clipPlanes[4].constant = -zmin; clipPlanes[5].constant = zmax;
}}

function cloud(flatArr, color, size) {{
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flatArr, 3));
  const mat = new THREE.PointsMaterial({{ size: size, color: color, clippingPlanes: clipPlanes }});
  return new THREE.Points(g, mat);
}}
const mapPoints = cloud(MAP_FLAT, 0x999999, 0.03);
scene.add(mapPoints);

const hostTrajGeom = new THREE.BufferGeometry();
hostTrajGeom.setAttribute('position', new THREE.BufferAttribute(HOST_TRAJ_FLAT, 3));
const hostTrajLine = new THREE.Line(hostTrajGeom, new THREE.LineBasicMaterial({{ color: 0x2266ff, clippingPlanes: clipPlanes }}));
scene.add(hostTrajLine);
const hostTrajPts = cloud(HOST_TRAJ_FLAT, 0x2266ff, 0.08);
scene.add(hostTrajPts);

const otherTrajGeom = new THREE.BufferGeometry();
otherTrajGeom.setAttribute('position', new THREE.BufferAttribute(OTHER_TRAJ_FLAT, 3));
const otherTrajLine = new THREE.Line(otherTrajGeom, new THREE.LineBasicMaterial({{ color: 0xff2619, clippingPlanes: clipPlanes }}));
scene.add(otherTrajLine);
const otherTrajPts = cloud(OTHER_TRAJ_FLAT, 0xff2619, 0.08);
scene.add(otherTrajPts);

const mapSizeEl = document.getElementById('mapSize');
const trajSizeEl = document.getElementById('trajSize');
function updatePointSizes() {{
  const ms = parseFloat(mapSizeEl.value), ts = parseFloat(trajSizeEl.value);
  mapPoints.material.size = ms;
  hostTrajPts.material.size = ts;
  otherTrajPts.material.size = ts;
  document.getElementById('mapSizeLabel').textContent = ms.toFixed(3);
  document.getElementById('trajSizeLabel').textContent = ts.toFixed(2);
}}
mapSizeEl.addEventListener('input', updatePointSizes);
trajSizeEl.addEventListener('input', updatePointSizes);
updatePointSizes();

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
    print(f"wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
