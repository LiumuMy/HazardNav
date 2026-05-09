# disaster_sim

灾害扩散仿真与梯度引导自主导航包。机器人不知道灾害源的位置，完全依靠自身算法趋近源头：在 SLAM 占用栅格上计算扩散场（测地模式用墙阻隔 BFS + 指数衰减，扩散模式用 Jacobi 迭代），然后在膨胀可达域上 BFS 搜索场值最高的格子，沿梯度方向绕墙前进直到到达判定触发。

## v2 新增功能（hazard_control_panel_node）

### 控制面板（tkinter GUI）

| 功能 | 说明 |
|------|------|
| 污染源 X/Y 滑块 | 拖动滑块实时调整污染源坐标，替代手动输入 |
| 位置预设 | Save / Load / Delete 一键保存/切换/删除预设位置，自动持久化至 `~/.hazard_control_panel_presets.json` |
| CSV 轨迹记录 | 开启后自动写轨迹 CSV 到指定目录，关闭时自动保存 |
| 目录选择 | Browse 按钮打开目录选择器指定 CSV 保存路径 |
| 机器人出生点控制 | X / Y / Yaw 三轴滑块 + Apply 按钮，通过 `/initialpose` 发布机器人初始位姿 |

### v1 已有功能

- `source_x`、`source_y`、`C0`、`source_type`、`algorithm` 参数均可通过界面直接修改并 Apply 到 ROS
- 实时显示机器人当前采样浓度（`/hazard/sample`）
- `ros2 launch disaster_sim disaster_nav.launch.py` 默认启动控制面板

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `source_type` | `fire` | 灾害类型：`fire` / `gas` / `pollution` |
| `source_x` | `6.0` | 灾害源 x 坐标 (m) |
| `source_y` | `4.0` | 灾害源 y 坐标 (m) |
| `algorithm` | `geodesic` | 扩散算法：`geodesic` / `diffusion` |
| `C0` | `600.0` | 源强（℃ / ppm） |
| `use_rviz` | `true` | 是否启动 RViz |
| `use_control_panel` | `true` | 是否打开 tkinter 控制面板 |

## 安装

```bash
git clone https://github.com/LiumuMy/HazardNav.git
```

将 `HazardNav/HazardNav/src/` 下的三个包 (`nav_slam`, `disaster_sim`, `gazebo_modele`) 放入你的 ROS 2 工作空间的 `src/` 目录，然后：

```bash
cd <your_ros2_ws>
colcon build --symlink-install
source install/setup.bash
```

## 仿真

```bash
ros2 launch disaster_sim disaster_nav.launch.py
```

示例：

```bash
ros2 launch disaster_sim disaster_nav.launch.py
```

## 依赖

- ROS 2 (Humble / Iron)
- Gazebo + gazebo_ros_pkgs
- Nav2
- Python: `numpy`, `scipy`, `sensor_msgs_py`
