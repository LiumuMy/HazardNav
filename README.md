# HazardNav

基于 ROS 2 的灾害场景自主导航仿真平台。在 Gazebo 中模拟火源/毒气/污染源扩散，结合 SLAM 建图、A\* 路径规划与纯追踪 (Pure Pursuit) 控制，实现机器人自主搜索灾害源。

## 工作原理

机器人**不知道灾害源的位置**，完全依靠**自身算法**趋近源头：首先，灾害源节点在 SLAM 地图的占用栅格上计算扩散场——在测地距离模式下用墙阻隔的 BFS 计算每格到源的最短路径长度，再用指数衰减 \(C = C_0 \cdot e^{-\alpha \cdot d}\) 建模浓度场；或者在扩散模式下用 Jacobi 迭代求解拉普拉斯方程，使场自然渗透过门缝和开口。机器人上的梯度探索节点订阅该浓度场，在膨胀后的可达地图上做 BFS 遍历所有可通行的格子，挑出场值最高的格子作为目标，沿父链回溯取中间点发给 A\* 绕墙规划，再由纯追踪控制器跟踪 B 样条平滑后的路径。与此同时，点云实时障碍检测会在机器人贴墙卡住时触发墙边跟随，绕过去后恢复场引导。整个过程形成闭环：**感知场 → 搜索最高场值可达格 → 绕墙前往 → 到达判定**，机器人不需要任何源位置先验知识。

## 核心特性

- **物理扩散场仿真** — 支持测地距离 (Geodesic) 算法，扩散场受墙体阻隔以至于达到不穿墙的效果
- **梯度引导自主探索** — 无需手动指定目标点，机器人自动沿浓度/温度梯度搜索灾害源
- **SLAM + A\* + 纯追踪** — 完整的建图→全局规划→局部跟踪导航管线，含 B 样条路径平滑
- **点云坐标变换** — 将 LiDAR 点云从传感器坐标系实时转换到 map 坐标系
- **一键启动** — 单条 launch 命令集成 Gazebo、导航全管线、扩散仿真、RViz 与控制面板

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

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `source_type` | `fire` | 灾害类型：`fire` / `gas` / `pollution` |
| `source_x` | `6.0` | 灾害源 x 坐标 (m) |
| `source_y` | `4.0` | 灾害源 y 坐标 (m) |
| `algorithm` | `geodesic` | 扩散算法：`geodesic` / `diffusion` |
| `C0` | `600.0` | 源强（℃ / ppm） |
| `use_rviz` | `true` | 是否启动 RViz |
| `use_control_panel` | `true` | 是否打开 tkinter 控制面板 |

示例：

```bash
ros2 launch disaster_sim disaster_nav.launch.py \
    source_type:=gas source_x:=8.0 source_y:=3.0 algorithm:=diffusion
```

## 项目结构

```
HazardNav/HazardNav/src/
├── nav_slam/          # SLAM建图 + A*规划 + 纯追踪控制 + 点云变换
├── disaster_sim/      # 灾害扩散仿真 + 梯度探索 + 控制面板 + 轨迹记录
└── gazebo_modele/     # Gazebo仿真环境 (URDF模型 + 2D/3D世界)
```

## 依赖

- ROS 2 (Humble / Iron)
- Gazebo + gazebo_ros_pkgs
- Nav2
- Python: `numpy`, `scipy`, `sensor_msgs_py`

## 许可

Apache-2.0 License
