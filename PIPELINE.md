# System Pipeline: Ego-Motion Compensation + Neighbor-Drone Detection

Two stages, built and validated together: **Deskew Pipeline** (Part I) produces a clean,
full-resolution near-field point stream from a moving platform; **Detection** (Part II)
consumes that stream to causally flag points belonging to a nearby drone. Sensor:
Livox Mid-360. Target platform: NVIDIA Jetson. Application: near-field (≤3.5 m)
neighbor-drone detection for a drone swarm, where a single target may be represented by
as few as 5-15 LiDAR points.

```
/livox/lidar ──► [Part I: Deskew Pipeline, this repo root] ──► /nearfield/deskewed_world (C1)
                  Point-LIO fork + cylindrical           /nearfield/refined_world   (C2)
                  near-field bypass + optional                    │
                  fixed-lag smoothed refinement                   ▼
                                                     [Part II: Detection, detection/]
                                                     causal DUFOMap: per-point
                                                     static/dynamic classification
                                                                    │
                                                                    ▼
                                                     candidate neighbor-drone points
```

**Status at a glance:** Part I is implemented and validated (see §4). Part II's core
algorithm is implemented and validated offline against recorded flights (see
[`detection/README.md`](detection/README.md)) **and** has been deployed as a live ROS2
node (`detection/causal_live.py`) and benchmarked on the Jetson target — see Part II §3
and [`RUNTIME_OPTIMIZATION.md`](RUNTIME_OPTIMIZATION.md) for real Jetson latency numbers
and the concurrent-workload (Point-LIO + detector running together) figures that
actually matter for deployment.

---

# Part I: Deskew Pipeline

**One-line summary:** existing LIO systems perform second-pass deskewing purely to
improve odometry / registration accuracy (estimation-side, frame-level). This work
instead treats deskewing as an output product for downstream perception: a fixed-lag
smoother refines Point-LIO's per-point state stream and performs a second, full-resolution
deskew pass, validated against a downstream drone-detection metric rather than ATE alone.

Built on a fork of [Point-LIO](https://github.com/hku-mars/Point-LIO) (ROS2 port of
[dfloreaa/point_lio_ros2](https://github.com/dfloreaa/point_lio_ros2)).

---

## 1. Problem

A LiDAR frame is a ~0.1 s integration window; every point in that window is captured at
a slightly different sensor pose, but the raw scan packages them as if captured
simultaneously. Static structure appears smeared whenever the platform moves, and the
error grows with range (position error ≈ r·δθ — a 1° attitude residual becomes 10.5 cm
at 6 m).

The downstream task here — detecting a sparse, small target within 3.5 m — has stricter
requirements on output point-cloud quality than typical LIO applications, and two hard
constraints:

- **Full resolution.** Any decimation risks dropping the few points that make up a valid
  target. Point-LIO's default real-time decimation (`point_filter_num > 1`) is not
  acceptable on the points that matter most.
- **Real-time on Jetson.** Removing decimation globally (`point_filter_num = 1`
  everywhere) is not feasible at the required frame rate.

Existing second-pass deskewing methods across the LIO literature (LOAM/F-LOAM,
FAST-LIO2, DLIO, the continuous-time family — CLINS/SLICT/Coco-LIC/CT-ICP, AC-LIO,
ADC-LIO) are all estimation-oriented: deskewing is a preprocessing step for scan
registration, evaluated by pose accuracy, and applied to the decimated point subset used
for matching. Output point-cloud quality for downstream consumers has not been treated
as a first-class concern.

## 2. Approach

Two complementary passes, both structured as pure additions to Point-LIO — neither
feeds back into the state estimator, so the core filter's behavior and accuracy are
unaffected regardless of whether either is enabled.

```
                          /livox/lidar (full-resolution CustomMsg)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                        │
           point_filter_num decimation              near-field bypass
                    │                                 (cylindrical gate,
                    ▼                                  see §2.3)
         Point-LIO EKF per-point update                      │
         (unchanged core estimator)                          │
                    │                                        ▼
                    ▼                          per-point causal transform
              state stream (t, R, p, v)         using the state AT THAT INSTANT
                    │                                        │
                    │                                        ├──► /nearfield/deskewed_world   (C1, zero extra latency)
                    ▼                                        │
        ┌─────────────────────────┐                          │
        │  fixed-lag RTS smoother │ ── smoothed trajectory ───┤
        │  (C2, optional)         │                           │
        └─────────────────────────┘                           ▼
                                                  /nearfield/refined_world  (C2, latency = lag)
```

### 2.1 C1 — Decoupled full-resolution compensation

The state-estimation path (decimated, for real-time cost control) and the deskewing
output path (full resolution) are decoupled. Every near-field point receives per-point
causal compensation using Point-LIO's state at its own capture instant — the same
per-point state stream the filter already produces internally, but no longer discarded
for points that would otherwise be decimated away.

- Near-field points are routed into a separate buffer at extraction time, bypassing
  `point_filter_num` entirely. They do **not** participate in the EKF measurement
  update unless explicitly enabled (`nearfield.join_update`, default off) — a nearby
  dynamic target (e.g. a neighbor drone) can therefore never pollute state estimation.
- Output is published per frame: zero additional latency relative to the core filter.

### 2.2 C2 — Fixed-lag smoothed second-pass deskew (optional)

Point-LIO's causal state estimate at time *t* only conditions on data up to *t*; a
short window of future observations could improve the estimate at *t*, but a purely
causal filter never uses it (the standard filtering-vs-smoothing gap). C2 runs a
lightweight fixed-lag smoother over the state stream and re-deskews the buffered
near-field points once the smoothing window closes:

- **Position:** independent constant-velocity model per axis (x/y/z), using the
  causal filter's own position and velocity as noisy observations.
- **Rotation:** each node's rotation is expressed relative to the window's first node
  in the tangent space (`Log(R_ref⁻¹ R_i)`); each of the three tangent components is
  smoothed independently with a random-walk model, then mapped back with `Exp`.
- Implemented as a standalone class (`src/FixedLagSmoother.hpp`), independent of the
  Point-LIO main loop and unit-tested in isolation before integration.
- Output latency is fixed and configurable (`os_deskew.lag`, default 0.1 s). C2 is a
  read-only consumer of the causal state stream — it never writes back into the
  estimator, so it can be enabled or disabled with zero risk to the core filter.

Two output rates are published so downstream consumers choose per their own latency
budget: causal (`/nearfield/deskewed_*`, C1, zero extra latency) for latency-critical
control/avoidance, and smoothed (`/nearfield/refined_*`, C2, latency = lag) for
detection/mapping where a short fixed delay is acceptable.

### 2.3 Near-field region: cylinder, not sphere

The near-field bypass originally admitted any point within a spherical radius
(`nearfield.near_range`) of the sensor. This pulls in ceiling/floor structure whenever
the platform's altitude brings it within range of an overhead or underfoot surface —
producing spurious dynamic-point detections downstream at grazing incidence, since a
real neighbor drone is never far above or below the platform at that horizontal range.
The near-field region is now a cylinder: horizontal radius `nearfield.near_range`
(default 3.5 m) combined with a vertical half-height `nearfield.z_half_height` (default
1.0 m), evaluated in the sensor's own body frame. This removes ceiling/floor points from
the near-field stream at the source, with zero cost to detection recall (validated: the
cut removes points only in a region no real target ever occupies).

## 3. Status

| Component | Status |
|---|---|
| C1 — decoupled full-resolution near-field deskew | Implemented, validated |
| C2 — fixed-lag RTS smoothed refinement (v1) | Implemented, validated |
| Cylindrical near-field gate | Implemented, validated |
| Downstream detection (causal DUFOMap) | Validated offline (see Part II); deployed live on Jetson (`detection/causal_live.py`), see [`RUNTIME_OPTIMIZATION.md`](RUNTIME_OPTIMIZATION.md) |

**Primary use case is zero-latency avoidance, which consumes C1 only.** C2 adds a fixed
0.1 s latency in exchange for a smoother trajectory and is available for
detection/mapping consumers that can tolerate it; it is not the default path.

Both are disabled by default in `config/mid360.yaml` (`nearfield.enable: false`,
`os_deskew.enable: false`) — with both off, behavior is identical to upstream Point-LIO.

## 4. Validation

**Regression (state estimation unaffected):** with `nearfield.enable`/`os_deskew.enable`
toggled on vs. off, `/aft_mapped_to_init` trajectories agree to within a few millimeters
(mean ~2.7 mm, max ~7.8 mm) — near-field extraction and the smoother do not perturb the
core filter, as expected from the read-only design.

**Latency:** the last C2 (`refined`) message is delayed from the corresponding causal
message by exactly the configured lag (measured: 0.100 s for `lag: 0.1`).

**Correction magnitude vs. platform motion:** a pilot study binned near-field points by
IMU angular velocity and compared causal vs. a smoothed estimate. Correction magnitude
increases with rotational rate, consistent with the r·δθ error model:

| Angular rate | mean \|Δp\| | p95 \|Δp\| | max \|Δp\| |
|---|---|---|---|---|
| < 0.5 rad/s | 1.0 cm | 3.7 cm | 14.1 cm |
| 0.5–2.0 rad/s | 6.5 cm | 21.0 cm | 46.3 cm |
| > 2.0 rad/s | 18.4 cm | 47.3 cm | 78.0 cm |

**Ceiling false-positive fix:** switching the near-field region from a sphere to a
cylinder (§2.3) removed a systematic false-positive source in downstream dynamic-point
detection with no measurable recall cost — see the detection-side repository for the
full evaluation.

## 5. Scope and limitations

- C2 does not feed back into the state estimator by design: it improves the deskewed
  *output*, not odometry accuracy. ATE is identical with C2 on or off.
- C2's process/observation noise parameters are currently fixed constants, not adapted
  from the causal filter's own reported covariance or from the actual dispersion of
  data in each smoothing window. This is the main open area for refinement.
- Rotation interpolation between smoothed nodes uses tangent-space linear interpolation
  rather than SLERP; adequate for the small angles and short windows involved here.
- Far-field decimation is unchanged from upstream Point-LIO; the compensation described
  here applies to the near-field region only.

## 6. Build

Standard ROS2 (Jazzy) colcon workspace, depends on `livox_ros_driver2`:

```bash
colcon build --packages-select point_lio
source install/setup.bash
```

## 7. Run

```bash
ros2 launch point_lio mapping_mid360.launch.py
```

Enable near-field / C2 output via `config/mid360.yaml`:

```yaml
nearfield:
    enable: true
    near_range: 3.5
    z_half_height: 1.0
os_deskew:
    enable: true      # optional, C1 alone is sufficient for zero-latency use cases
    lag: 0.1
```

## 8. Known issues

- `ikd-Tree` is referenced in `.gitmodules` at the repo root but the code actually
  compiled is `include/ikd-Tree/` (tracked directly, not a submodule) — the root-level
  submodule entry is stale and unused.
- This fork's `pointlio_mapping` node does not respond to SIGINT/SIGTERM once input bag
  playback ends; it spins in a busy loop instead of exiting. Use `pkill -9` after
  playback completes. If recording its output with `ros2 bag record`, the resulting
  mcap will be missing its footer metadata — recoverable with
  `ros2 bag reindex <dir> -s mcap`.

---

# Part II: Detection

Code and full write-up: [`detection/`](detection/README.md). Summary below.

## 1. Approach

[DUFOMap](https://github.com/KTH-RPL/dufomap) (RA-L 2024) is used as the underlying
ray-casting occupancy detector, reformulated as **causal**: DUFOMap's own reference
usage integrates every frame into its map first and classifies afterward (a two-pass
batch method that uses future frames to judge past ones). Here, for frame *i*,
classification (`segment()`) uses only the map built from frames `0..i-1`; the frame is
integrated into the map (`run()`) only afterward — no look-ahead.

## 2. Evaluation

Two real two-drone flights (LiDAR on one drone, the other as the target) were processed
through Part I, then evaluated against pseudo-ground-truth labels built by registering
each flight to a static reference map of the site and cross-referencing the other
flight's registered trajectory (see `detection/README.md` §2 for the full methodology
and its caveats — no manual annotation was used, and the pseudo-labels have a known,
documented source of noise from the two recording devices' clocks never being
synchronized).

## 3. What "causal" does and does not mean here

- **Architecturally causal / real-time-compatible:** the algorithm never uses future
  data.
- **Benchmarked live on the actual Jetson target** (`detection/causal_live.py`, a real
  ROS2 node subscribing to Point-LIO's published topics — not an offline bag read).
  With `run()` (map integration) backgrounded off the critical path (see
  [`RUNTIME_OPTIMIZATION.md`](RUNTIME_OPTIMIZATION.md) — `segment()` for frame *i* is
  causally independent of frame *i*'s own `run()`, so integration can execute
  concurrently while later work proceeds, verified bit-identical output either way),
  the detector's own critical-path cost is under 1ms/frame on Jetson, running
  concurrently with Point-LIO. **Point-LIO's own per-frame cost, not the detector's, is
  the dominant term in combined-workload latency** — see
  [`RUNTIME_OPTIMIZATION.md`](RUNTIME_OPTIMIZATION.md) for the current end-to-end
  number and the concurrent-workload caveats that don't show up in either component
  benchmarked alone.
- **Background subtraction (used only to build evaluation labels) is offline**, an ICP
  registration against a pre-built static reference map. It is not part of, and is not
  needed for, the causal detector's own real-time path.

## 4. Key results

- Parameter ablation selected `d_p=2` (`d_p=1` gives much worse precision for a small
  recall gain); `resolution`/`d_s` trade recall against precision without a value that
  wins on both evaluated flights.
- A precision collapse (from ~1.00 down to ~0.01 in the worst 60 s segment) was traced
  to a **spatially local cold start**: it began exactly when the platform entered a
  part of the space it had never observed before, independent of total elapsed flight
  time (180 s of prior flight time gave no protection once a new region was entered).
- The false positives above were further traced to a **ceiling/overhead structure**:
  94% fell in a narrow height band matching a high-density layer of the reference map,
  with zero true detections there. This motivated the cylindrical near-field gate
  (Part I §2.3): removing points above a height cutoff recovered nearly all lost
  precision at zero recall cost. A separate voxel-gating/clustering/tracking approach
  was also tried and discarded in favor of this simpler, more effective source-level
  fix.
- **Hovering does not hurt detection**: after the cylindrical near-field gate fix,
  recall is 0.996 for pseudo-GT points matched to a hovering segment of the other
  drone's trajectory (n=2707) and 0.996 for points matched to a moving segment
  (n=8131) — no measurable gap, contrary to the usual theoretical concern for
  occupancy-based dynamic-point detectors that a stationary target eventually looks
  like static structure. (A pre-fix run, when the ceiling issue above still
  depressed overall recall, showed a spurious hovering/moving split; it disappeared
  once the ceiling issue was fixed.)

## 5. Scope and limitations

- Pseudo-GT labels are spatial-proximity-based, not time-synchronized between the two
  flights (no shared clock was available) — a documented, only partially mitigated
  source of label noise.
- Evaluated on two flights at one site; no cross-site generalization evidence yet.
- The offline evaluation (§2, pseudo-GT metrics) and the live Jetson deployment (§3)
  have not yet been run as a single combined pass — recall/precision numbers above are
  from the offline path, latency numbers are from the live path. Re-running the
  pseudo-GT evaluation through `causal_live.py` itself (not just the offline
  `causal_vs_batch.py`) would close this gap.
