#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hazard_source_node
------------------

灾害源头模拟与发布节点。在 Gazebo 仿真世界中虚拟一个灾害源
(可切换为 fire 热源 / gas 有毒气体 / pollution 污染物)，
并基于 OccupancyGrid 的墙体信息计算一个"不会穿墙"的扩散场。

订阅:
  /combined_grid      nav_msgs/OccupancyGrid   原项目 map_pub 发布的栅格地图
  /odom               nav_msgs/Odometry        机器人位姿，用于采样"机器人处读数"

发布:
  /hazard/field       nav_msgs/OccupancyGrid   归一化后的扩散场 (0~100)
                                                便于直接在 RViz 里用 Map 图层显示热力
  /hazard/sample      std_msgs/Float32         机器人所在位置的浓度/温度
  /hazard/source_gt   geometry_msgs/PointStamped  源真值位置 (调试/可视化)
  /hazard/markers     visualization_msgs/MarkerArray  RViz 可视化 (源+标签)

参数:
  source_type         'fire' | 'gas' | 'pollution'
  source_x, source_y  源位置 (map 坐标系, m)
  algorithm           'geodesic' | 'diffusion'   (决策 C: 两种都实现, 可切换)
  C0                  源强度 (℃ / ppm / μg·m⁻³, 取决于类型)
  alpha               测地模型的衰减率 (1/m)
  diffusion_iters     Jacobi 迭代次数
  diffusion_decay     Jacobi 每步损耗率
  publish_rate_hz     发布频率

通信全部采用 ROS 2 标准消息, 代码结构模块化, 纯 Python。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32, ColorRGBA, Header
from geometry_msgs.msg import PointStamped, Point, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray

from disaster_sim.diffusion_core import (
    compute_geodesic_field,
    compute_diffusion_field,
    world_to_grid,
    OBSTACLE_VALUE,
)


SOURCE_TYPES = ('fire', 'gas', 'pollution')

# 三种灾害的默认物理量纲提示 (仅用于日志/标签, 不影响数值计算)
TYPE_UNIT = {
    'fire': '°C',
    'gas': 'ppm',
    'pollution': 'ug/m3',
}
TYPE_COLOR = {
    'fire':      ColorRGBA(r=1.0, g=0.35, b=0.0, a=0.9),
    'gas':       ColorRGBA(r=0.2, g=1.0,  b=0.2, a=0.9),
    'pollution': ColorRGBA(r=0.6, g=0.3,  b=0.9, a=0.9),
}


class HazardSourceNode(Node):
    """灾害源模拟器: 基于占用栅格计算物理扩散场并对外发布。"""

    def __init__(self) -> None:
        super().__init__('hazard_source_node')

        self.declare_parameter('source_type', 'fire')
        self.declare_parameter('source_x', 5.0)
        self.declare_parameter('source_y', 5.0)
        self.declare_parameter('algorithm', 'geodesic')
        self.declare_parameter('C0', 500.0)
        self.declare_parameter('alpha', 0.25)
        self.declare_parameter('diffusion_iters', 250)
        self.declare_parameter('diffusion_decay', 0.015)
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('downsample', 4)

        self._reload_params()

        qos = QoSProfile(depth=5)
        self.sub_grid = self.create_subscription(
            OccupancyGrid, '/combined_grid', self._grid_cb, qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos)

        self.pub_field = self.create_publisher(
            OccupancyGrid, '/hazard/field', qos)
        self.pub_sample = self.create_publisher(
            Float32, '/hazard/sample', qos)
        self.pub_src = self.create_publisher(
            PointStamped, '/hazard/source_gt', qos)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/hazard/markers', qos)

        self._latest_grid: Optional[OccupancyGrid] = None
        self._cached_field: Optional[np.ndarray] = None
        self._cached_grid_signature = None
        self._robot_xy = (0.0, 0.0)

        period = 1.0 / max(0.1, float(self.publish_rate_hz))
        self.timer = self.create_timer(period, self._publish_tick)

        self.get_logger().info(
            f"hazard_source_node ready: type={self.source_type} "
            f"algo={self.algorithm} source=({self.source_x:.2f},{self.source_y:.2f}) "
            f"C0={self.C0} unit={TYPE_UNIT.get(self.source_type,'')}"
        )

    def _reload_params(self) -> None:
        self.source_type = str(self.get_parameter('source_type').value).lower()
        if self.source_type not in SOURCE_TYPES:
            self.get_logger().warn(
                f"unknown source_type={self.source_type}, fallback to 'fire'")
            self.source_type = 'fire'
        self.source_x = float(self.get_parameter('source_x').value)
        self.source_y = float(self.get_parameter('source_y').value)
        self.algorithm = str(self.get_parameter('algorithm').value).lower()
        if self.algorithm not in ('geodesic', 'diffusion'):
            self.algorithm = 'geodesic'
        self.C0 = float(self.get_parameter('C0').value)
        self.alpha = float(self.get_parameter('alpha').value)
        self.diffusion_iters = int(self.get_parameter('diffusion_iters').value)
        self.diffusion_decay = float(self.get_parameter('diffusion_decay').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.downsample = max(1, int(self.get_parameter('downsample').value))

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        self._latest_grid = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

    def _grid_signature(self, msg: OccupancyGrid):
        info = msg.info
        arr = np.asarray(msg.data, dtype=np.int16)
        wall_count = int((arr == OBSTACLE_VALUE).sum())
        value_sum = int(arr.sum(dtype=np.int64))
        return (
            info.width, info.height, round(info.resolution, 6),
            round(info.origin.position.x, 6),
            round(info.origin.position.y, 6),
            wall_count, value_sum,
        )

    def _compute_field(self, grid_msg: OccupancyGrid) -> np.ndarray:
        """基于占用栅格 + 源参数 计算浮点扩散场 (单位: 与 C0 一致)。"""
        info = grid_msg.info
        w = int(info.width)
        h = int(info.height)
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)

        grid = np.array(grid_msg.data, dtype=np.int16).reshape(h, w)
        obstacle = np.where(grid == OBSTACLE_VALUE, OBSTACLE_VALUE, 0).astype(np.int16)

        sr, sc = world_to_grid(
            self.source_x, self.source_y, ox, oy, res, w, h)

        if obstacle[sr, sc] == OBSTACLE_VALUE:
            self.get_logger().warn(
                f"source ({self.source_x:.2f},{self.source_y:.2f}) "
                f"falls on an obstacle cell, field will be empty."
            )

        if self.algorithm == 'diffusion':
            field = compute_diffusion_field(
                obstacle, (sr, sc),
                C0=self.C0,
                iters=self.diffusion_iters,
                decay=self.diffusion_decay,
                downsample=self.downsample,
            )
        else:
            field = compute_geodesic_field(
                obstacle, (sr, sc),
                C0=self.C0,
                alpha=self.alpha,
                resolution=res,
                downsample=self.downsample,
            )
        return field

    def _ensure_field(self) -> Optional[np.ndarray]:
        if self._latest_grid is None:
            return None
        sig = self._grid_signature(self._latest_grid)
        param_sig = (
            self.source_type, self.source_x, self.source_y,
            self.algorithm, self.C0, self.alpha,
            self.diffusion_iters, self.diffusion_decay,
            self.downsample,
        )
        full_sig = (sig, param_sig)
        if self._cached_field is None or self._cached_grid_signature != full_sig:
            self._cached_field = self._compute_field(self._latest_grid)
            self._cached_grid_signature = full_sig
        return self._cached_field

    def _publish_tick(self) -> None:
        self._reload_params()
        self._publish_markers_and_source()
        if self._latest_grid is None:
            return
        field = self._ensure_field()
        if field is None:
            return
        self._publish_field(field, self._latest_grid)
        self._publish_sample(field, self._latest_grid)

    def _publish_field(self, field: np.ndarray, grid_msg: OccupancyGrid) -> None:
        max_v = float(field.max()) if field.size else 0.0
        if max_v <= 1e-6:
            norm = np.zeros_like(field, dtype=np.int8)
        else:
            norm = np.clip((field / max_v) * 100.0, 0, 100).astype(np.int8)

        out = OccupancyGrid()
        out.header = Header()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.info = grid_msg.info
        out.data = norm.flatten().tolist()
        self.pub_field.publish(out)

    def _publish_sample(self, field: np.ndarray, grid_msg: OccupancyGrid) -> None:
        info = grid_msg.info
        w = int(info.width)
        h = int(info.height)
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        rx, ry = self._robot_xy
        row, col = world_to_grid(rx, ry, ox, oy, res, w, h)
        val = float(field[row, col])
        self.pub_sample.publish(Float32(data=val))

    def _publish_markers_and_source(self) -> None:
        stamp = self.get_clock().now().to_msg()

        src = PointStamped()
        src.header.stamp = stamp
        src.header.frame_id = self.frame_id
        src.point.x = self.source_x
        src.point.y = self.source_y
        src.point.z = 0.0
        self.pub_src.publish(src)

        color = TYPE_COLOR.get(self.source_type, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9))
        unit = TYPE_UNIT.get(self.source_type, '')

        marr = MarkerArray()

        sphere = Marker()
        sphere.header.stamp = stamp
        sphere.header.frame_id = self.frame_id
        sphere.ns = 'hazard_source'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(x=self.source_x, y=self.source_y, z=0.3)
        sphere.pose.orientation.w = 1.0
        sphere.scale = Vector3(x=0.6, y=0.6, z=0.6)
        sphere.color = color
        marr.markers.append(sphere)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.frame_id
        text.ns = 'hazard_source'
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(x=self.source_x, y=self.source_y, z=1.0)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.5
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = f"{self.source_type.upper()} C0={self.C0:.1f}{unit}"
        marr.markers.append(text)

        self.pub_markers.publish(marr)


def main(args=None):
    rclpy.init(args=args)
    node = HazardSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
