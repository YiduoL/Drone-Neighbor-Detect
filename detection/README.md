# Detection: Causal Background Subtraction + Dynamic-Point Detection

Real-time (causal) neighbor-drone detection on top of the near-field point stream
produced by the OS-Deskew stage (see the top-level [`PIPELINE.md`](../PIPELINE.md)).
Uses [DUFOMap](https://github.com/KTH-RPL/dufomap) (RA-L 2024) as the underlying
ray-casting occupancy detector, reformulated as a **causal** (history-only,
no look-ahead) online detector.

## Status and scope

This is a validation study, not a deployed real-time system. What has been shown:

- The causal reformulation (§1 below) is architecturally online and its per-frame cost
  on a desktop x86 machine is well within a 10 Hz frame budget.
- It has **not** been benchmarked on the actual Jetson deployment target.
- It has **not** been wired up as a live ROS2 node subscribing to a real-time topic --
  all evaluation here reads recorded bags offline.
- The background-subtraction step used to build evaluation labels (§2) is itself an
  offline batch process (ICP registration against a pre-built static reference map) --
  it is **not** part of the real-time detection path; it exists only to produce
  ground-truth labels for evaluating the causal detector.

## 1. Causal DUFOMap

DUFOMap's reference usage (see its own `main.py`) is a two-pass batch method: integrate
every frame into the map first, then classify. That uses future frames' information to
judge past ones and is not real-time-compatible. Here, `run()` (map integration) and
`segment()` (classification) are called in causal order per frame: for frame *i*,
`segment()` is called using only the map built from frames `0..i-1`, and `run()`
integrates frame *i* only afterward. Frame 0 has no history (cold start; treated as
all-static by convention).

`run()` only ever integrates far-field points into the background map -- the near-field
region is the detection target's own neighborhood, and letting near-field points into
the background model would let a slow-moving or hovering target get absorbed into
"background" over time.

## 2. Evaluation methodology

Two real two-drone flights (LiDAR mounted on one drone, the other drone as the target)
were used, each processed through the OS-Deskew pipeline first.

**Pseudo-ground-truth construction** (no manual annotation): each flight's far-field
point cloud is ICP-registered to a static reference map of the site (a scan recorded
with the sensor stationary), placing both flights' trajectories into one shared
coordinate frame. A near-field point is labeled "the other drone" if it is (a) far
enough from the reference map to be foreground, and (b) spatially close to the other
flight's registered trajectory. Caveat: the two recording devices' clocks were never
synchronized, so "close to the other trajectory" means close to any point the other
drone visited over its *entire* flight, not at the same instant -- this is a real
source of label noise (see `build_pseudo_labels.py`'s docstring), only partially offset
by requiring agreement with the background-subtraction signal.

**Metrics**: standard point-level TP/FP/FN/TN against the pseudo-GT above, plus
recall broken down by elapsed time (cold-start behavior) and by range.

## 3. Key findings

- **Parameter ablation** (`ablation_resolution_dp.py`, `ablation_ds.py`): `d_p=2`
  is clearly better than `d_p=1` (much higher precision, small recall cost).
  `resolution` and `d_s` trade recall against precision without a value that
  dominates on both flights simultaneously; the paper's defaults were kept.
- **Cold start is spatially local, not just temporal.** A flight that had already
  achieved near-perfect precision for 180 s collapsed to near-zero precision the
  moment it entered a part of the space it had never observed before (new altitude,
  new region) -- even though 180 s of *total* flight time had already elapsed. The
  causal map's confidence is per-location, not a single global "warm-up timer";
  re-entering unexplored space re-triggers the same cold-start behavior regardless of
  total elapsed time.
- **Root cause of the false-positive burst above: a ceiling/overhead structure**,
  confirmed against the reference map (94% of false positives fell in a narrow height
  band matching a high-density structural layer in the reference map; recall was
  unaffected). See `zcut_diagnostic.py` for the diagnostic that isolated this and
  motivated the cylindrical near-field gate fix (`PIPELINE.md` §2.3) -- removing the
  points above the cutoff cost zero recall (no real target was ever observed there) and
  recovered nearly all of the lost precision, more simply and effectively than a
  voxel-gating/clustering/tracking approach that was also tried and discarded in favor
  of the simpler fix.
- **Hovering does not hurt detection** (`hovering_hypothesis_test.py`): a common
  theoretical concern for occupancy-based dynamic detectors is that a stationary
  object eventually looks like static structure. Measured recall was in fact *higher*
  for pseudo-GT points matched to a hovering segment of the other drone's trajectory
  (88%) than to a moving segment (76%) -- plausibly because a hovering target is
  observed repeatedly from a stable vantage point, giving the ray-casting logic more
  consistent evidence than a fast-moving target sampled briefly at each location.

## 4. Scripts

| Script | Purpose |
|---|---|
| `background_subtraction.py` | Register a flight to a static reference map; classify near-field points as background/foreground. Offline, used only for building evaluation labels. |
| `build_pseudo_labels.py` | Combine background subtraction with the other flight's registered trajectory into per-point pseudo-GT labels. |
| `common_eval.py` | Shared data loading, pseudo-GT lookup, and metrics accumulation used by the ablation scripts. |
| `ablation_resolution_dp.py` | Sweep DUFOMap `resolution` x `d_p`. |
| `ablation_ds.py` | Sweep DUFOMap `d_s` at the resolution/d_p chosen above. |
| `causal_vs_batch.py` | Compare the causal reformulation against DUFOMap's official batch usage: dynamic-point counts, per-frame timing, cold-start convergence. Needs no pseudo-GT. |
| `zcut_diagnostic.py` | Diagnostic: what does simply dropping near-field points above a height cutoff do to precision/recall? Motivated the cylindrical near-field gate. |
| `hovering_hypothesis_test.py` | Compares causal-detector recall for pseudo-GT points matched to a hovering vs. moving segment of the other drone's trajectory. |
| `visualize_confusion.py` | Three.js viewer: per-segment TP/FP visualization with an interactive bounding-box clip and the other flight's trajectory for context. |

## 5. Data layout

Scripts read paths from environment variables (with `./data`-relative defaults) rather
than hardcoded absolute paths:

```bash
export DETECTION_DATA_DIR=/path/to/output/dir
export SWARM1_BAG=/path/to/pointlio_lidar_1_output      # C1-only bag
export SWARM2_BAG=/path/to/pointlio_lidar_2_output
export SWARM1_C2_BAG=/path/to/pointlio_lidar_1_c2_output  # C1+C2 bag
export SWARM2_C2_BAG=/path/to/pointlio_lidar_2_c2_output
export REFERENCE_MAP=/path/to/reference_map.pcd
```

Expected pipeline order, per flight:

```bash
python3 background_subtraction.py $SWARM1_BAG $DETECTION_DATA_DIR/swarm1_bgsub
python3 background_subtraction.py $SWARM2_BAG $DETECTION_DATA_DIR/swarm2_bgsub
python3 build_pseudo_labels.py --host swarm1
python3 build_pseudo_labels.py --host swarm2
python3 ablation_resolution_dp.py
python3 causal_vs_batch.py $SWARM1_BAG $DETECTION_DATA_DIR/swarm1_out
python3 visualize_confusion.py --host swarm1
```

## 6. Dependencies

```bash
pip install dufomap open3d scipy scikit-learn pandas
```

Also requires `rosbag2_py` / `rclpy` (from a sourced ROS2 install) to read bag files.
