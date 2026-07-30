# Drone-Neighbor-Detect

Neighbor-drone detection for drone swarms using an onboard Livox Mid-360 LiDAR. Two
stages, see [`PIPELINE.md`](PIPELINE.md) for the full design and validation of both:

- **Ego-motion compensation** (this repo's root): a modified
  [Point-LIO](https://github.com/hku-mars/Point-LIO) (ROS2 port, forked from
  [dfloreaa/point_lio_ros2](https://github.com/dfloreaa/point_lio_ros2)) with an added
  near-field deskewing path, producing clean full-resolution input for detection.
- **Detection** ([`detection/`](detection/README.md)): causal background subtraction
  and dynamic-point detection on top of the deskewed near-field stream.

## Deskew pipeline summary

- **C1 (zero extra latency):** near-field points (0.1-3.5 m, cylindrical gate ±1 m in
  height) bypass the main EKF update and decimation entirely, and are deskewed
  per-point using the causal state at that instant — no dropped points, no added
  latency, and near targets can never pollute state estimation.
- **C2 (optional, +0.1 s latency):** a fixed-lag RTS smoother refines the causal state
  stream and re-deskews the buffered near-field points once the window closes —
  smoother output, at the cost of a fixed delay. Read-only with respect to the
  estimator; never feeds back into it.

Both are disabled by default in `config/mid360.yaml`; with both off, this fork behaves
identically to upstream Point-LIO.

## Build

Standard ROS2 (Jazzy) colcon workspace, depends on `livox_ros_driver2`:

```bash
colcon build --packages-select point_lio
source install/setup.bash
```

## Run

```bash
ros2 launch point_lio mapping_mid360.launch.py
```
