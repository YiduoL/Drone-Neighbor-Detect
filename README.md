# Drone-Neighbor-Detect

无人机集群邻机检测(drone-swarm neighbor detection):机载 Livox Mid360 LiDAR,实时
识别附近的其他无人机。

这部分是 **ego-motion 去畸变**——基于 [Point-LIO](https://github.com/hku-mars/Point-LIO)
(ROS2版本fork自 [dfloreaa/point_lio_ros2](https://github.com/dfloreaa/point_lio_ros2))
加了近场专用的去畸变通道,给后面的检测部分(背景剔除 + 动态目标检出)提供干净的
输入点云。详见 [`OS-DESKEW.md`](OS-DESKEW.md)。

## 核心改动:OS-Deskew

- **C1(零延迟)**:近场(0.1-3.5m内)点云绕开主 EKF 状态估计流程和降采样,直接用
  当前帧位姿因果去畸变、原样输出——不丢点、不引入延迟、不会被近距离动态目标
  (比如另一台无人机)污染主状态估计。
- **C2(可选,+0.1s延迟)**:在C1基础上加一个固定滞后窗口的RTS平滑
  (`src/FixedLagSmoother.hpp`),用未来数据把位姿再修正一遍,更平滑但有延迟,纯附加
  不反馈回主状态估计。

默认(`config/mid360.yaml`)两个功能都关闭,不影响原版 Point-LIO 行为。

## 构建

标准 ROS2 (Jazzy) colcon 工作区,依赖 `livox_ros_driver2`:

```bash
colcon build --packages-select point_lio
source install/setup.bash
```

## 运行

```bash
ros2 launch point_lio mapping_mid360.launch.py
```
