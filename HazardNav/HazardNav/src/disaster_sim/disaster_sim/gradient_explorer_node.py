#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradient_explorer_node
----------------------

纯标量场驱动的自主探索节点：机器人不掌握任何源位置先验，仅依靠
/hazard/field（来自 hazard_source_node 的灾害浓度/温度场）和
/combined_grid（SLAM 建立的障碍栅格）自动生成 /goal_pose，
交由原 nav_slam.astar 做 A* 绕墙规划、再由 start_nav（纯追踪）执行。

工作闭环:
  /hazard/field   ──►┐
  /combined_grid  ──►│  gradient_explorer_node
  /odom           ──►│  ─► /goal_pose  ─► astar ─► /path  ─► start_nav ─► /cmd_vel
  /hazard/sample  ──►┘

订阅:
  /hazard/field     nav_msgs/OccupancyGrid    归一化的扩散场 (0~100)
  /combined_grid    nav_msgs/OccupancyGrid    SLAM 建立的障碍栅格 (墙=100)
  /odom             nav_msgs/Odometry         机器人位姿
  /hazard/sample    std_msgs/Float32          机器人当前读数（触发停车）
  /hazard/source_gt geometry_msgs/PointStamped 源真值位置（仅作信息日志，不参与决策）

发布:
  /goal_pose       geometry_msgs/PoseStamped  自动生成的中间目标（原 astar 入口）

决策层核心（纯场驱动，无先验）：

  A. 主路径：场最高可达点 BFS（field_max_BFS）
     在膨胀后的 /combined_grid 可达域上做 BFS，遍历所有可达格子并比较
     /hazard/field 值，挑出场最高的格子 G*。沿 BFS 父链回溯取离机器
     人 ≥ goal_lookahead_m 的中间点发 /goal_pose。
     物理意义：机器人始终朝"场上坡"方向移动，在 SLAM 地图约束下
     自动绕过障碍逼近浓度最高的可达位置——即源的位置。

  B. Fallback：局部 Sobel 梯度射线
     当 BFS 可达域为空（如刚启动地图尚未建立）时，用 Sobel 算子计算
     /hazard/field 的局部梯度，朝梯度方向取可达中间点。纯局部信息，
     不做全局搜索，但在地图未就绪时能驱动机器人开始运动。

  C. 到达判定：
     sample >= arrival_sample_threshold → 判定到达，锁定当前位姿为
     goal 并停止下发新目标。/hazard/source_gt 的订阅保留用于日志输出，
     不参与任何决策逻辑。

调参建议:
  - goal_lookahead_m: 1.5~2.5m，越大中间目标越远，A* 重规划频率越低
  - search_radius_m: 8~12m，覆盖中等规模室内地图；大场景调至 15~20m
  - arrival_sample_threshold: 配合 C0 和 alpha 使用，详见 hazard_params.yaml
  - max_bfs_cells: 80000（600x600 栅格约 150~300ms）；大地图改大以扩展探索范围
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
import struct
import time

from disaster_sim.diffusion_core import (
    world_to_grid,
    grid_to_world,
    OBSTACLE_VALUE,
)


class GradientExplorerNode(Node):
    """纯场驱动探索器：不依赖源位置先验，自动沿浓度梯度前进。"""

    def __init__(self) -> None:
        super().__init__('gradient_explorer_node')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('update_period_sec', 0.5)
        self.declare_parameter('goal_lookahead_m', 1.8)
        self.declare_parameter('search_radius_m', 8.0)
        self.declare_parameter('obstacle_clearance_cells', 3)
        self.declare_parameter('arrival_sample_threshold', 50.0)
        self.declare_parameter('arrival_distance_m', 0.8)
        self.declare_parameter('min_field_to_move', 1e-3)
        self.declare_parameter('max_bfs_cells', 80000)
        self.declare_parameter('min_field_gain', 0.5)
        self.declare_parameter('bfs_relax_pad_on_stuck', True)
        self.declare_parameter('verbose_goal_log', True)

        self.goal_topic = str(self.get_parameter('goal_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.update_period = float(self.get_parameter('update_period_sec').value)
        self.goal_lookahead = float(self.get_parameter('goal_lookahead_m').value)
        self.search_radius = float(self.get_parameter('search_radius_m').value)
        self.obstacle_clearance = int(
            self.get_parameter('obstacle_clearance_cells').value)
        self.arrival_sample_threshold = float(
            self.get_parameter('arrival_sample_threshold').value)
        self.arrival_distance = float(self.get_parameter('arrival_distance_m').value)
        self.min_field_to_move = float(self.get_parameter('min_field_to_move').value)
        self.max_bfs_cells = int(self.get_parameter('max_bfs_cells').value)
        self.min_field_gain = float(self.get_parameter('min_field_gain').value)
        self.bfs_relax_pad_on_stuck = bool(
            self.get_parameter('bfs_relax_pad_on_stuck').value)
        self.verbose_goal_log = bool(
            self.get_parameter('verbose_goal_log').value)

        qos = QoSProfile(depth=5)
        self.sub_field = self.create_subscription(
            OccupancyGrid, '/hazard/field', self._field_cb, qos)
        self.sub_grid = self.create_subscription(
            OccupancyGrid, '/combined_grid', self._grid_cb, qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos)
        self.sub_sample = self.create_subscription(
            Float32, '/hazard/sample', self._sample_cb, qos)
        # 源真值订阅保留，仅作信息日志；不参与任何决策
        self.sub_src_gt = self.create_subscription(
            PointStamped, '/hazard/source_gt', self._src_gt_cb, qos)

        self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, qos)
        self.pub_cmd_vel = self.create_publisher(
            Twist, '/cmd_vel', QoSProfile(depth=3))

        self._field_msg: Optional[OccupancyGrid] = None
        self._grid_msg: Optional[OccupancyGrid] = None
        self._robot_xy: Tuple[float, float] = (0.0, 0.0)
        self._latest_sample: float = 0.0
        self._arrived = False
        self._arrival_anchor: Optional[Tuple[float, float]] = None
        self._latest_source_xy: Optional[Tuple[float, float]] = None
        self._last_goal_log: Optional[Tuple[str, float, float]] = None
        self._last_goal_log_stamp: float = 0.0

        # 自探索绕墙状态机
        self._wall_explore_last_pos: Optional[Tuple[float, float]] = None  # 上次 goal 位置
        self._wall_explore_last_robot_pos: Optional[Tuple[float, float]] = None
        self._wall_explore_cooldown = 0               # 冷却计时（tick数）
        self._wall_explore_in_progress = False        # 是否处于绕墙探索中
        self._wall_explore_last_wall_explore_time: float = 0.0
        self._wall_explore_max_rays = 16              # 射线数量
        self._wall_explore_max_range_cells = 20       # 射线最大长度（格）
        self._wall_explore_min_free_cells = 3         # 绕墙方向最小自由通道（格）
        self._wall_explore_stuck_threshold_m = 0.15   # 机器人移动阈值

        # 实时点云障碍检测 + 墙边跟随
        self._pc_msg: Optional[PointCloud2] = None
        self._pc_recent_points: List[Tuple[float, float]] = []  # (x, y) in robot frame
        self._wall_follow_active = False
        self._wall_follow_dir = 0                      # 1=左侧墙, -1=右侧墙
        self._wall_follow_straight_counter = 0        # 连续直行计数
        self._wall_follow_dir_change_counter = 0       # 方向切换冷却
        self._last_wall_follow_cmd_time = 0.0
        self._pc_sub = self.create_subscription(
            PointCloud2, '/mapokk', self._pc_cb, 10)

        self.timer = self.create_timer(self.update_period, self._tick)

        self.get_logger().info(
            f"gradient_explorer_node (field_driven) ready: "
            f"lookahead={self.goal_lookahead}m "
            f"search_radius={self.search_radius}m "
            f"arrival_threshold={self.arrival_sample_threshold} "
            f"arrival_distance={self.arrival_distance}m "
            f"max_bfs_cells={self.max_bfs_cells} "
            f"obstacle_clearance={self.obstacle_clearance}"
        )

    # --------------------------------------------------------------
    # 自探索绕墙：射线探测通道宽度
    # --------------------------------------------------------------
    def _cast_free_channel(
        self,
        r: int, c: int,
        dr: int, dc: int,
        blocked: np.ndarray,
        max_range: int,
    ) -> int:
        """从 (r,c) 沿 (dr,dc) 方向返回不受障碍阻挡的连续自由格数。"""
        cnt = 0
        for k in range(1, max_range + 1):
            nr = r + dr * k
            nc = c + dc * k
            if not (0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]):
                break
            if blocked[nr, nc]:
                break
            cnt += 1
        return cnt

    def _nearest_obstacle_distance(
        self, rr: int, rc: int, blocked: np.ndarray, max_range: int
    ) -> float:
        """返回机器人到最近膨胀墙的格数。"""
        for d in range(1, max_range + 1):
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    if abs(dr) != d and abs(dc) != d:
                        continue
                    nr, nc = rr + dr, rc + dc
                    if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                        if blocked[nr, nc]:
                            return float(d)
        return float(max_range)

    def _find_best_wall_passage(
        self,
        rr: int, rc: int,
        blocked: np.ndarray,
        res: float, ox: float, oy: float,
        h: int, w: int,
    ) -> Optional[Tuple[float, float]]:
        """
        在机器人四周发出射线，计算每个方向沿墙走的通道宽度，
        返回最宽通道对应的世界坐标目标点。
        排除机器人后方（朝来的反方向），优先侧向绕墙。
        """
        n_rays = self._wall_explore_max_rays
        max_range = self._wall_explore_max_range_cells
        candidates: List[Tuple[int, int, int, int, int]] = []  # (free, dr, dc, pass_r, pass_c)

        for i in range(n_rays):
            angle = 2.0 * math.pi * i / n_rays
            dr = int(round(math.sin(angle)))
            dc = int(round(math.cos(angle)))
            if dr == 0 and dc == 0:
                continue

            free = self._cast_free_channel(rr, rc, dr, dc, blocked, max_range)
            if free < self._wall_explore_min_free_cells:
                continue

            # 通道末端的格子（绕墙后的落脚点）
            pass_r = rr + dr * free
            pass_c = rc + dc * free
            if not (0 <= pass_r < h and 0 <= pass_c < w):
                continue
            if blocked[pass_r, pass_c]:
                continue

            # 朝场梯度方向加权（选同时能靠近源的方向）
            candidates.append((free, dr, dc, pass_r, pass_c))

        if not candidates:
            return None

        # 选通道最宽的方向
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_free, best_dr, best_dc, target_r, target_c = candidates[0]

        tx, ty = grid_to_world(target_r, target_c, ox, oy, res)
        self.get_logger().warn(
            f"[WALL_EXPLORE] direction=({best_dr},{best_dc}) "
            f"free_cells={best_free} → goal=({tx:.2f},{ty:.2f})"
        )
        return float(tx), float(ty)

    # --------------------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------------------
    def _field_cb(self, msg: OccupancyGrid) -> None:
        self._field_msg = msg

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        self._grid_msg = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        self._robot_orient_quat = msg.pose.pose.orientation

    def _sample_cb(self, msg: Float32) -> None:
        self._latest_sample = float(msg.data)

    def _src_gt_cb(self, msg: PointStamped) -> None:
        # 仅记录，用于日志输出；不在任何决策逻辑中使用
        self._latest_source_xy = (float(msg.point.x), float(msg.point.y))

    def _pc_cb(self, msg: PointCloud2) -> None:
        """解析 PointCloud2，提取 map 坐标系下所有点的 (x, y)，存入列表。"""
        self._pc_msg = msg
        points = []
        offset = 0
        while offset + 16 <= len(msg.data):
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            points.append((float(x), float(y), float(z)))
            offset += msg.point_step
        self._pc_recent_points = [(px, py) for px, py, pz in points]

    # --------------------------------------------------------------
    # 实时墙边跟随（基于点云，无激光雷达）
    # --------------------------------------------------------------
    def _wall_follow(self) -> bool:
        """
        基于点云实时墙边跟随。
        返回 True 表示正在执行墙边跟随（此时发布 cmd_vel 不发 goal_pose）。
        返回 False 表示墙边跟随结束，可以继续正常导航。
        """
        now = time.monotonic()
        if now - self._last_wall_follow_cmd_time < 0.3:
            return True
        self._last_wall_follow_cmd_time = now

        pts = self._pc_recent_points
        if not pts:
            self._wall_follow_active = False
            return False

        rx, ry = self._robot_xy
        q = self._robot_orient_quat if hasattr(self, '_robot_orient_quat') else None

        # 机器人朝向
        if q is not None:
            yaw = self._quaternion_to_yaw(q)
        else:
            yaw = 0.0

        # 转换到机器人局部坐标系
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
        local_pts = []
        for px, py in pts:
            dx, dy = px - rx, py - ry
            lx = dx * cos_yaw - dy * sin_yaw
            ly = dx * sin_yaw + dy * cos_yaw
            local_pts.append((lx, ly))

        # 分类：左(ly>0)、右(ly<0)、前(lx>0)、后(lx<0)
        left = [lp for lp in local_pts if lp[1] > 0.05 and abs(lp[0]) < 2.0]
        right = [lp for lp in local_pts if lp[1] < -0.05 and abs(lp[0]) < 2.0]
        front = [lp for lp in local_pts if lp[0] > 0.05 and abs(lp[1]) < 1.5]
        rear = [lp for lp in local_pts if lp[0] < -0.05]

        # 距离
        left_dist = min((abs(lp[1]) for lp in left), default=3.0)
        right_dist = min((abs(lp[1]) for lp in right), default=3.0)
        front_dist = min((lp[0] for lp in front), default=3.0)

        # 判断墙在左侧还是右侧
        if left_dist < right_dist:
            wall_side = 1   # 墙在左侧（左侧更近）
            wall_dist = left_dist
        else:
            wall_side = -1  # 墙在右侧
            wall_dist = right_dist

        # 障碍检测阈值
        FRONT_STOP = 0.6
        WALL_FAR = 2.5
        WALL_NEAR = 0.4

        if front_dist < FRONT_STOP:
            # 前方有墙，旋转（朝墙的方向旋转）
            turn_dir = wall_side
            angular = turn_dir * 0.6
            linear = 0.0
            self.get_logger().warn(
                f"[WALL_FOLLOW] wall ahead dist={front_dist:.2f}m, "
                f"turning {'left' if turn_dir > 0 else 'right'}"
            )
        elif wall_dist > WALL_FAR:
            # 离墙太远，向墙靠近
            turn_dir = -wall_side
            angular = turn_dir * 0.4
            linear = 0.15
            self._wall_follow_straight_counter = 0
        elif wall_dist < WALL_NEAR:
            # 离墙太近，远离墙
            turn_dir = wall_side
            angular = turn_dir * 0.4
            linear = 0.15
            self._wall_follow_straight_counter = 0
        else:
            # 贴着墙直行
            angular = 0.0
            linear = 0.2
            self._wall_follow_straight_counter += 1

        # 发布 cmd_vel
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.pub_cmd_vel.publish(cmd)
        self._wall_follow_active = True
        return True

    def _quaternion_to_yaw(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # --------------------------------------------------------------
    # Main tick
    # --------------------------------------------------------------
    def _tick(self) -> None:
        self._maybe_release_arrival()
        if self._check_arrival():
            self._publish_hold_goal()
            return
        if self._field_msg is None:
            return

        # 冷却中：等待机器人绕过去
        if self._wall_explore_cooldown > 0:
            self._wall_explore_cooldown -= 1
            return

        # 绕墙探索完成，准备重置
        if self._wall_explore_in_progress:
            self._wall_explore_in_progress = False
            self._wall_explore_last_pos = None
            self._wall_explore_last_robot_pos = None
            return

        goal = self._compute_goal()
        if goal is None:
            return

        # ---- 自探索绕墙检测（基于膨胀地图 BFS + 点云实时障碍）----
        should_wall_explore = False
        if self._grid_msg is not None:
            field_msg = self._field_msg
            w = int(field_msg.info.width)
            h = int(field_msg.info.height)
            ox = float(field_msg.info.origin.position.x)
            oy = float(field_msg.info.origin.position.y)
            res = float(field_msg.info.resolution)
            obstacle_grid = np.array(
                self._grid_msg.data, dtype=np.int16).reshape(h, w)
            blocked = self._inflate_obstacles(
                obstacle_grid, self.obstacle_clearance)
            rr, rc = world_to_grid(
                self._robot_xy[0], self._robot_xy[1],
                ox, oy, res, w, h)
            rr = min(max(rr, 0), h - 1)
            rc = min(max(rc, 0), w - 1)
            near_wall = self._nearest_obstacle_distance(
                rr, rc, blocked,
                max_range=10) < 4.0  # < 4 格认为靠近墙

            last_pos = self._wall_explore_last_pos
            last_robot = self._wall_explore_last_robot_pos
            robot_moved = (
                last_robot is None
                or math.hypot(
                    self._robot_xy[0] - last_robot[0],
                    self._robot_xy[1] - last_robot[1]
                ) > self._wall_explore_stuck_threshold_m
            )
            goal_stable = (
                last_pos is not None
                and math.hypot(goal[0] - last_pos[0], goal[1] - last_pos[1])
                < self._wall_explore_stuck_threshold_m
            )

            if near_wall and not robot_moved and goal_stable:
                self.get_logger().warn(
                    f"[WALL_EXPLORE] near_wall={near_wall} "
                    f"robot_moved={robot_moved} goal_stable={goal_stable} "
                    f"→ triggering wall-follow"
                )
                should_wall_explore = True

            self._wall_explore_last_pos = (goal[0], goal[1])
            self._wall_explore_last_robot_pos = self._robot_xy

        # ---- 执行墙边跟随（实时点云计算障碍）----
        if should_wall_explore:
            self._wall_follow()
            # 冷却后退出墙边跟随
            if not hasattr(self, '_wall_follow_cooldown'):
                self._wall_follow_cooldown = 0
            self._wall_follow_cooldown += 1
            if self._wall_follow_cooldown > 40:
                self._wall_follow_cooldown = 0
                self._wall_explore_in_progress = False
                self._wall_explore_last_pos = None
                self._wall_explore_last_robot_pos = None
                self.get_logger().warn(
                    "[WALL_FOLLOW] wall-follow timeout, resuming normal navigation"
                )
            return

        # 正常退出墙边跟随时重置状态
        if hasattr(self, '_wall_follow_cooldown') and self._wall_follow_cooldown > 0:
            self._wall_follow_cooldown = 0
            self._wall_follow_active = False

        self._publish_goal(*goal)

    # --------------------------------------------------------------
    # Arrival / release
    # --------------------------------------------------------------
    def _maybe_release_arrival(self) -> None:
        if not self._arrived:
            return
        if self._latest_sample < self.arrival_sample_threshold:
            self.get_logger().info(
                f"[RESUME] sample dropped to {self._latest_sample:.2f} "
                f"< {self.arrival_sample_threshold:.2f}; resuming gradient tracking"
            )
            self._arrived = False
            self._arrival_anchor = None

    def _check_arrival(self) -> bool:
        if self._arrived:
            return True
        if self._latest_sample >= self.arrival_sample_threshold:
            self._arrived = True
            self._arrival_anchor = self._robot_xy
            self.get_logger().info(
                f"[ARRIVAL] sample={self._latest_sample:.2f} >= "
                f"{self.arrival_sample_threshold:.2f}, locking goal at "
                f"({self._arrival_anchor[0]:.2f},{self._arrival_anchor[1]:.2f})"
            )
            return True
        return False

    def _publish_hold_goal(self) -> None:
        if self._arrival_anchor is None:
            self._arrival_anchor = self._robot_xy
        self._publish_goal(*self._arrival_anchor)

    def _publish_goal(self, x: float, y: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    # --------------------------------------------------------------
    # Goal computation: field_max_BFS primary, Sobel fallback
    # --------------------------------------------------------------
    def _compute_goal(self) -> Optional[Tuple[float, float]]:
        """
        决策顺序（纯场驱动，无源位置先验）：
           A. 场最高可达点 BFS（主路径）
           B. Sobel 局部梯度射线（无地图时兜底）
        """
        field_msg = self._field_msg
        grid_msg = self._grid_msg
        if field_msg is None:
            return None

        info = field_msg.info
        w = int(info.width)
        h = int(info.height)
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        if w == 0 or h == 0 or res <= 0.0:
            return None

        field = (np.array(field_msg.data, dtype=np.int16)
                   .reshape(h, w).astype(np.float32))
        if field.max() < self.min_field_to_move:
            return None

        if (grid_msg is not None
                and grid_msg.info.width == w
                and grid_msg.info.height == h):
            obstacle_grid = np.array(
                grid_msg.data, dtype=np.int16).reshape(h, w)
            has_grid = True
        else:
            obstacle_grid = np.zeros((h, w), dtype=np.int16)
            has_grid = False

        rx, ry = self._robot_xy
        rr, rc = world_to_grid(rx, ry, ox, oy, res, w, h)
        rr = min(max(rr, 0), h - 1)
        rc = min(max(rc, 0), w - 1)

        # ---- A: 场最高可达点 BFS（主路径，纯场驱动）----
        if has_grid:
            bfs_goal = self._compute_bfs_goal(
                field, obstacle_grid, rr, rc, res, ox, oy, w, h,
                pad=self.obstacle_clearance,
            )
            if bfs_goal is not None:
                self._log_goal('field_seeking', bfs_goal[0], bfs_goal[1], '')
                return bfs_goal
            if self.bfs_relax_pad_on_stuck and self.obstacle_clearance > 1:
                bfs_goal2 = self._compute_bfs_goal(
                    field, obstacle_grid, rr, rc, res, ox, oy, w, h,
                    pad=max(0, self.obstacle_clearance - 2),
                )
                if bfs_goal2 is not None:
                    self._log_goal(
                        'field_seeking_relaxed',
                        bfs_goal2[0], bfs_goal2[1], '')
                    return bfs_goal2

        # ---- B: Sobel 局部梯度射线（兜底）----
        fallback = self._compute_sobel_goal(
            field, obstacle_grid, rr, rc, res, ox, oy, w, h)
        if fallback is not None:
            self._log_goal('sobel', fallback[0], fallback[1], '')
        return fallback

    def _log_goal(self, method: str, gx: float, gy: float, dbg: str) -> None:
        """节流日志：method 变化或目标位移 > 0.5m 或超过 3s 才再打一次。"""
        if not self.verbose_goal_log:
            return
        now = time.monotonic()
        prev = self._last_goal_log
        if prev is not None:
            pm, px, py = prev
            same = (pm == method
                    and math.hypot(gx - px, gy - py) < 0.5
                    and now - self._last_goal_log_stamp < 3.0)
            if same:
                return
        src_info = ''
        if self._latest_source_xy is not None:
            sx, sy = self._latest_source_xy
            src_info = f"(src_gt=({sx:.2f},{sy:.2f}) NOTE: NOT used for navigation) "
        extra = f" {dbg}" if dbg else ''
        self.get_logger().info(
            f"[GOAL] method={method} {src_info}"
            f"robot=({self._robot_xy[0]:.2f},{self._robot_xy[1]:.2f}) "
            f"goal=({gx:.2f},{gy:.2f}){extra}"
        )
        self._last_goal_log = (method, gx, gy)
        self._last_goal_log_stamp = now

    # --------------------------------------------------------------
    # A: 场最高可达点 BFS（主路径）
    # --------------------------------------------------------------
    def _inflate_obstacles(
        self, obstacle_grid: np.ndarray, pad: int
    ) -> np.ndarray:
        """对墙格子膨胀 pad 个栅格（不处理未知=-1），返回 bool 阻挡矩阵。"""
        occ = (obstacle_grid == OBSTACLE_VALUE)
        if pad <= 0:
            return occ
        out = occ.copy()
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(occ, dr, axis=0), dc, axis=1)
                if dr > 0:
                    shifted[:dr, :] = False
                elif dr < 0:
                    shifted[dr:, :] = False
                if dc > 0:
                    shifted[:, :dc] = False
                elif dc < 0:
                    shifted[:, dc:] = False
                out |= shifted
        return out

    def _compute_bfs_goal(
        self,
        field: np.ndarray,
        obstacle_grid: np.ndarray,
        rr: int, rc: int,
        res: float, ox: float, oy: float,
        w: int, h: int,
        pad: int,
    ) -> Optional[Tuple[float, float]]:
        """
        在膨胀后的 SLAM 地图可达域上 BFS，遍历所有可达格子并比较
        /hazard/field 值，挑出场最高的格子 G*，沿父链回溯取中间目标。

        纯标量场驱动：不使用任何源位置信息。
        """
        blocked = self._inflate_obstacles(obstacle_grid, pad)
        if blocked[rr, rc]:
            start_allowed = True
        else:
            start_allowed = True
        if obstacle_grid[rr, rc] == OBSTACLE_VALUE:
            return None

        max_cells_r = max(3, int(self.search_radius / res))
        max_bfs = int(self.max_bfs_cells)

        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        parent[(rr, rc)] = None
        queue: deque = deque()
        queue.append((rr, rc))

        start_field_val = float(field[rr, rc])
        best_rc = (rr, rc)
        best_field = start_field_val
        visited = 0
        _ = start_allowed  # lint

        while queue:
            cr, cc = queue.popleft()
            visited += 1
            if visited > max_bfs:
                break

            cv = float(field[cr, cc])
            if cv > best_field:
                best_field = cv
                best_rc = (cr, cc)

            if abs(cr - rr) > max_cells_r or abs(cc - rc) > max_cells_r:
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = cr + dr
                nc = cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in parent:
                    continue
                if blocked[nr, nc]:
                    continue
                parent[(nr, nc)] = (cr, cc)
                queue.append((nr, nc))

        if best_rc == (rr, rc):
            return None
        if (best_field - start_field_val) < self.min_field_gain:
            return None

        # 沿父链回溯取 lookahead 处的中间点
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = best_rc
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        if len(path) <= 1:
            return None

        lookahead_cells = max(1, int(self.goal_lookahead / res))
        chosen = path[-1]
        for pr, pc in path[1:]:
            d = max(abs(pr - rr), abs(pc - rc))
            if d >= lookahead_cells:
                chosen = (pr, pc)
                break

        tx, ty = grid_to_world(chosen[0], chosen[1], ox, oy, res)
        return float(tx), float(ty)

    # --------------------------------------------------------------
    # B: Fallback — v1 local Sobel ray
    # --------------------------------------------------------------
    def _compute_sobel_goal(
        self,
        field: np.ndarray,
        obstacle_grid: np.ndarray,
        rr: int, rc: int,
        res: float, ox: float, oy: float,
        w: int, h: int,
    ) -> Optional[Tuple[float, float]]:
        radius_cells = max(3, int(self.search_radius / res))
        r0 = max(0, rr - radius_cells)
        r1 = min(h, rr + radius_cells + 1)
        c0 = max(0, rc - radius_cells)
        c1 = min(w, rc + radius_cells + 1)
        window = field[r0:r1, c0:c1]
        if window.size == 0:
            return None

        gy, gx = np.gradient(window)
        local_r = min(max(rr - r0, 0), window.shape[0] - 1)
        local_c = min(max(rc - c0, 0), window.shape[1] - 1)
        vx = float(gx[local_r, local_c])
        vy = float(gy[local_r, local_c])
        norm = math.hypot(vx, vy)

        if norm < 1e-6:
            idx = np.unravel_index(np.argmax(window), window.shape)
            target_r = r0 + int(idx[0])
            target_c = c0 + int(idx[1])
            dx = (target_c - rc) * res
            dy = (target_r - rr) * res
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                return None
            ux = dx / dist
            uy = dy / dist
        else:
            ux = vx / norm
            uy = vy / norm

        return self._pick_reachable_point(
            rr, rc, ux, uy, res, ox, oy, w, h, obstacle_grid
        )

    def _pick_reachable_point(
        self,
        rr: int, rc: int,
        ux: float, uy: float,
        res: float, ox: float, oy: float,
        w: int, h: int,
        obstacle_grid: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        step = max(1, int(round(0.5 / res)))
        max_cells = max(1, int(self.goal_lookahead / res))
        best_rc = None
        for k in range(step, max_cells + step, step):
            nr = int(round(rr + uy * k))
            nc = int(round(rc + ux * k))
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if self._is_cell_blocked(nr, nc, obstacle_grid):
                break
            best_rc = (nr, nc)
        if best_rc is None:
            for k in range(1, max_cells + 1):
                nr = int(round(rr + uy * k))
                nc = int(round(rc + ux * k))
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if self._is_cell_blocked(nr, nc, obstacle_grid):
                    break
                best_rc = (nr, nc)
            if best_rc is None:
                return None
        tx, ty = grid_to_world(best_rc[0], best_rc[1], ox, oy, res)
        return float(tx), float(ty)

    def _is_cell_blocked(self, r: int, c: int, obstacle_grid: np.ndarray) -> bool:
        if obstacle_grid[r, c] == OBSTACLE_VALUE:
            return True
        pad = self.obstacle_clearance
        if pad <= 0:
            return False
        h, w = obstacle_grid.shape
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        patch = obstacle_grid[r0:r1, c0:c1]
        return bool(np.any(patch == OBSTACLE_VALUE))


def main(args=None):
    rclpy.init(args=args)
    node = GradientExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
