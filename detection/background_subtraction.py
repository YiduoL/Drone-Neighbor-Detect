#!/usr/bin/env python3
"""Background subtraction for a two-drone flight, using a pre-built static reference
map (a scan of the empty site) as the background model -- not the flight's own
accumulated map. Registers the flight's far-field point cloud (/cloud_registered) to the
reference map via global (RANSAC/FPFH) + ICP registration (far-field is used for
registration rather than near-field alone, since it has much better spatial overlap with
the reference map), then applies the resulting transform to the near-field (0.1-3.5 m,
causal C1) cloud and classifies each near-field point as background (close to the
reference map) or foreground (far from it -- candidate: the other drone, self-hit noise,
or registration residue).

Default threshold (0.12 m) was chosen from an error-heatmap analysis where the
distance-to-reference distribution has a clear knee around there: background points
cluster under 0.05 m, and a long, roughly uniform tail starts around 0.1-0.15 m.

Usage: python3 background_subtraction.py <flight_bag> <out_dir> [--threshold 0.12]
"""
import argparse
import os

import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2

DEFAULT_REFERENCE_MAP = "reference_map.pcd"  # override with --reference-map
VOXEL_REG = 0.15
VOXEL_ICP = 0.05


def load_topic_accumulated(bag_path, topic):
    so = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    all_pts = []
    while reader.has_next():
        _, data, t = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if pts.size == 0:
            continue
        pts = np.column_stack([pts["x"], pts["y"], pts["z"]]) if pts.dtype.names else pts
        all_pts.append(pts.astype(np.float64))
    pts = np.concatenate(all_pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"{topic}: {len(pts)} accumulated points")
    return pcd


def load_trajectory(bag_path, topic):
    so = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    pts = []
    while reader.has_next():
        _, data, t = reader.read_next()
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose.position
        pts.append([p.x, p.y, p.z])
    return np.array(pts)


def prep_for_registration(pcd, voxel_size):
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return down, fpfh


def global_register(src_down, src_fpfh, tgt_down, tgt_fpfh, voxel_size):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    print(f"global registration: fitness={result.fitness:.3f} inlier_rmse={result.inlier_rmse:.4f}m")
    return result.transformation


def refine_icp(src, tgt, init_transform, voxel_size):
    src_d = src.voxel_down_sample(voxel_size)
    tgt_d = tgt.voxel_down_sample(voxel_size)
    tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, voxel_size * 2, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.transformation, result.fitness, result.inlier_rmse


def sanity_check_alignment(src_pcd, tgt_pcd, transform, label, max_nn_median=0.5):
    aligned = o3d.geometry.PointCloud(src_pcd)
    aligned.transform(transform)
    src_pts = np.asarray(aligned.points)
    d = np.asarray(aligned.compute_point_cloud_distance(tgt_pcd))
    print(f"[sanity check] {label}: aligned bbox x[{src_pts[:,0].min():.1f},{src_pts[:,0].max():.1f}] "
          f"y[{src_pts[:,1].min():.1f},{src_pts[:,1].max():.1f}] "
          f"z[{src_pts[:,2].min():.1f},{src_pts[:,2].max():.1f}]")
    print(f"[sanity check] {label}: nearest-neighbor distance to reference: "
          f"median={np.median(d):.3f}m mean={d.mean():.3f}m")
    if np.median(d) > max_nn_median:
        raise RuntimeError(
            f"{label}: registration looks WRONG (median NN distance {np.median(d):.2f}m, "
            f"expected < {max_nn_median}m) -- refusing to report results from this alignment.")
    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("out_dir")
    ap.add_argument("--reference-map", default=DEFAULT_REFERENCE_MAP,
                    help="static reference map (.pcd) of the empty site, built by "
                         "raw-scan accumulation while the sensor was stationary")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--near-topic", default="/nearfield/deskewed_world")
    ap.add_argument("--far-topic", default="/cloud_registered")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading reference map: {args.reference_map}")
    ref_pcd = o3d.io.read_point_cloud(args.reference_map)
    print(f"reference map: {len(ref_pcd.points)} points")

    far_pcd = load_topic_accumulated(args.flight_bag, args.far_topic)
    near_pcd = load_topic_accumulated(args.flight_bag, args.near_topic)
    traj = load_trajectory(args.flight_bag, args.odom_topic)

    ref_down, ref_fpfh = prep_for_registration(ref_pcd, VOXEL_REG)

    print("\n--- global registration using far-field cloud ---")
    far_down, far_fpfh = prep_for_registration(far_pcd, VOXEL_REG)
    T_global = global_register(far_down, far_fpfh, ref_down, ref_fpfh, VOXEL_REG)

    print("\n--- ICP refine ---")
    T_far, fitness, inlier_rmse = refine_icp(far_pcd, ref_pcd, T_global, VOXEL_ICP)
    print(f"ICP fitness={fitness:.3f} inlier_rmse={inlier_rmse:.4f}m")
    far_aligned = sanity_check_alignment(far_pcd, ref_pcd, T_far, "far-field")

    print("\n--- applying transform to near-field + trajectory ---")
    near_aligned = sanity_check_alignment(near_pcd, ref_pcd, T_far, "near-field",
                                          max_nn_median=2.0)  # near-field naturally has more outliers (that's the point)
    traj_h = np.concatenate([traj, np.ones((len(traj), 1))], axis=1)
    traj_aligned = (T_far @ traj_h.T).T[:, :3]

    print("\n--- background/foreground classification (near-field vs reference) ---")
    near_pts = np.asarray(near_aligned.points)
    d = np.asarray(near_aligned.compute_point_cloud_distance(ref_pcd))
    is_fg = d > args.threshold
    print(f"near-field: {len(near_pts)} points, {is_fg.sum()} ({100*is_fg.mean():.1f}%) foreground "
          f"(>{args.threshold}m from reference map)")

    np.savez(os.path.join(args.out_dir, "bgsubtract_result.npz"),
             near_pts=near_pts, is_fg=is_fg, traj_aligned=traj_aligned,
             far_aligned=np.asarray(far_aligned.points))
    print(f"wrote {os.path.join(args.out_dir, 'bgsubtract_result.npz')}")

    # --- visualization: reference (gray, downsampled) + near-field background (gray) +
    # near-field foreground (red) + trajectory (blue) ---
    import json
    ref_down_viz = ref_pcd.voxel_down_sample(0.05)
    ref_pts_viz = np.asarray(ref_down_viz.points)
    bg_pts = near_pts[~is_fg]
    fg_pts = near_pts[is_fg]

    def flat(a):
        return np.asarray(a).flatten().round(4).tolist()

    all_bbox = np.concatenate([ref_pts_viz, near_pts, traj_aligned], axis=0)
    mins = all_bbox.min(axis=0)
    maxs = all_bbox.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    radius = max(float(np.linalg.norm(maxs - mins) / 2), 1.0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Background Subtraction Result -- {os.path.basename(args.flight_bag)}</title>
<style>
  body {{ margin:0; background:#fff; color:#222; font-family: sans-serif; }}
  #controls {{ padding: 10px 20px; background:#f0f0f0; border-bottom: 1px solid #ccc;
               display:flex; gap: 20px; align-items:center; flex-wrap: wrap; font-size:13px; }}
  .swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; }}
  html, body {{ height:100%; }}
  #view {{ width:100%; height: calc(100vh - 46px); }}
  canvas {{ display:block; }}
</style></head>
<body>
<div id="controls">
  <span><span class="swatch" style="background:#dddddd;"></span> Reference background map ({len(ref_pts_viz)}pts)</span>
  <span><span class="swatch" style="background:#999999;"></span> Near-field: matched background ({len(bg_pts)}pts)</span>
  <span><span class="swatch" style="background:#ff2619;"></span> Near-field: foreground candidate ({len(fg_pts)}pts, {100*is_fg.mean():.1f}%)</span>
  <span><span class="swatch" style="background:#2266ff;"></span> Own trajectory</span>
  <span style="color:#888;">threshold={args.threshold}m, ICP inlier_rmse={inlier_rmse:.3f}m</span>
</div>
<div id="view"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const REF_FLAT = new Float32Array({json.dumps(flat(ref_pts_viz))});
const BG_FLAT = new Float32Array({json.dumps(flat(bg_pts))});
const FG_FLAT = new Float32Array({json.dumps(flat(fg_pts))});
const TRAJ_FLAT = new Float32Array({json.dumps(flat(traj_aligned))});
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

function cloud(flat, color, size) {{
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({{ size: size, color: color }}));
}}
scene.add(cloud(REF_FLAT, 0xdddddd, 0.02));
scene.add(cloud(BG_FLAT, 0x999999, 0.03));
scene.add(cloud(FG_FLAT, 0xff2619, 0.07));

const trajGeom = new THREE.BufferGeometry();
trajGeom.setAttribute('position', new THREE.BufferAttribute(TRAJ_FLAT, 3));
scene.add(new THREE.Line(trajGeom, new THREE.LineBasicMaterial({{ color: 0x2266ff }})));

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
</script></body></html>"""
    out_html = os.path.join(args.out_dir, "bgsubtract_viz.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
