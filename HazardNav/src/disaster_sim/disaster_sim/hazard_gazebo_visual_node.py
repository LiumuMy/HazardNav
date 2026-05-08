#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hazard_gazebo_visual_node
-------------------------

在 Gazebo Classic 仿真世界里把"灾害源"作为一个可见实体 spawn 出来，
与 hazard_source_node 发布的 ROS 浓度/温度场在位置上保持一致。

职责分工 (很重要, 别混):
  - Gazebo 只负责"把源渲染出来"的视觉占位, 不做气体/热量的物理扩散。
  - 扩散场仍由 hazard_source_node 在 OccupancyGrid 上计算, 墙=100 严格阻隔。
  - 本节点只调用两个标准服务:
        /spawn_entity   (gazebo_msgs/SpawnEntity)
        /delete_entity  (gazebo_msgs/DeleteEntity)
  - 所有服务调用都是 **异步** 的 (call_async + done_callback), 因为 timer
    回调内部不能再次 spin, 否则会死锁在 spin_until_future_complete.

参数:
    source_type  fire | gas | pollution
    source_x     m
    source_y     m
    entity_name  在 Gazebo 里显示的实体名 (默认 hazard_source_viz)
    update_period_sec  参数变化重新 spawn 的检查周期 (默认 2.0)
    ground_offset_z    source 在 Gazebo 里的离地高度 (默认按类型取值)

退出时会尝试 delete_entity, 避免下次启动时 "entity already exists".
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from gazebo_msgs.srv import SpawnEntity, DeleteEntity


SOURCE_TYPES = ('fire', 'gas', 'pollution')


def build_source_sdf(source_type: str, entity_name: str) -> str:
    """
    根据灾害类型生成一段 Gazebo Classic 兼容的 SDF, 一眼能看出是哪种源:
        fire       : 橙红色自发光球 + 顶部小火柱, 半径 0.5 m
        gas        : 半透明绿色大球 "气体云",      半径 0.9 m
        pollution  : 半透明紫色球 + 内嵌小暗球,    半径 0.7 m
    所有模型都 <static>true</static>, Gazebo 物理引擎不参与刚体动力学。
    """
    stype = (source_type or 'fire').lower()
    if stype not in SOURCE_TYPES:
        stype = 'fire'

    if stype == 'fire':
        radius = 0.5
        ambient = '1.0 0.35 0.05 1'
        diffuse = '1.0 0.55 0.1 1'
        emissive = '1.0 0.4 0.05 1'
        transparency = 0
        extras = f"""
      <visual name='flame_core'>
        <pose>0 0 {radius + 0.25} 0 0 0</pose>
        <geometry><cylinder><radius>{radius * 0.55}</radius><length>0.5</length></cylinder></geometry>
        <transparency>0.2</transparency>
        <material>
          <ambient>1.0 0.7 0.1 1</ambient>
          <diffuse>1.0 0.9 0.2 1</diffuse>
          <emissive>1.0 0.8 0.2 1</emissive>
        </material>
      </visual>
        """
    elif stype == 'gas':
        radius = 0.9
        ambient = '0.2 1.0 0.2 0.35'
        diffuse = '0.3 1.0 0.3 0.35'
        emissive = '0.1 0.5 0.1 0.4'
        transparency = 0.55
        extras = ''
    else:
        radius = 0.7
        ambient = '0.55 0.2 0.85 0.55'
        diffuse = '0.65 0.3 0.95 0.55'
        emissive = '0.35 0.1 0.55 0.35'
        transparency = 0.4
        extras = f"""
      <visual name='pollution_core'>
        <pose>0 0 0 0 0 0</pose>
        <geometry><sphere><radius>{radius * 0.4}</radius></sphere></geometry>
        <material>
          <ambient>0.15 0.05 0.25 1</ambient>
          <diffuse>0.2 0.1 0.3 1</diffuse>
          <emissive>0.1 0.0 0.2 1</emissive>
        </material>
      </visual>
        """

    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{entity_name}">
    <static>true</static>
    <link name="link">
      <visual name="halo">
        <pose>0 0 0 0 0 0</pose>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <transparency>{transparency}</transparency>
        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
          <emissive>{emissive}</emissive>
        </material>
      </visual>
      {extras}
    </link>
  </model>
</sdf>"""
    return sdf


def default_ground_offset(source_type: str) -> float:
    """不同类型源在地面上方的默认放置高度, 让球不至于埋在地里。"""
    return {
        'fire': 0.55,
        'gas': 0.95,
        'pollution': 0.75,
    }.get(source_type, 0.6)


# 状态机: 控制 spawn/delete 的异步流程, 避免 timer 内部 spin 死锁
STATE_IDLE = 0
STATE_SPAWNING = 1
STATE_DELETING = 2


class HazardGazeboVisualNode(Node):
    """把灾害源作为 Gazebo 实体 spawn/delete, 跟踪参数变化自动刷新。"""

    def __init__(self) -> None:
        super().__init__('hazard_gazebo_visual_node')

        self.declare_parameter('source_type', 'fire')
        self.declare_parameter('source_x', 5.0)
        self.declare_parameter('source_y', 5.0)
        self.declare_parameter('entity_name', 'hazard_source_viz')
        self.declare_parameter('update_period_sec', 2.0)
        self.declare_parameter('ground_offset_z', -1.0)
        self.declare_parameter('spawn_timeout_sec', 30.0)

        self._last_key: Optional[Tuple[str, float, float]] = None
        self._pending_key: Optional[Tuple[str, float, float]] = None
        self._state = STATE_IDLE
        self._lock = threading.Lock()
        self._shutdown = False

        cb_group = ReentrantCallbackGroup()
        self.spawn_client = self.create_client(
            SpawnEntity, '/spawn_entity', callback_group=cb_group)
        self.delete_client = self.create_client(
            DeleteEntity, '/delete_entity', callback_group=cb_group)

        period = float(self.get_parameter('update_period_sec').value)
        self.timer = self.create_timer(
            max(0.5, period), self._tick, callback_group=cb_group)

        self.get_logger().info(
            "hazard_gazebo_visual_node ready: will spawn/update the hazard "
            "source as a Gazebo entity mirroring source_type/x/y params."
        )

    def _read_key(self) -> Tuple[str, float, float, str, float]:
        stype = str(self.get_parameter('source_type').value).lower()
        if stype not in SOURCE_TYPES:
            stype = 'fire'
        sx = float(self.get_parameter('source_x').value)
        sy = float(self.get_parameter('source_y').value)
        name = str(self.get_parameter('entity_name').value) or 'hazard_source_viz'
        z_param = float(self.get_parameter('ground_offset_z').value)
        z = z_param if z_param >= 0.0 else default_ground_offset(stype)
        return stype, sx, sy, name, z

    def _tick(self) -> None:
        if self._shutdown:
            return
        with self._lock:
            if self._state != STATE_IDLE:
                return
            stype, sx, sy, name, z = self._read_key()
            key = (stype, round(sx, 3), round(sy, 3))
            if self._last_key == key:
                return

            if not self.spawn_client.service_is_ready():
                # /spawn_entity 还没起来, 下一个 tick 再试
                self.get_logger().debug(
                    "/spawn_entity not ready yet, will retry on next tick")
                return

            self._pending_key = key
            if self._last_key is not None:
                self._state = STATE_DELETING
                self._async_delete(name, next_step='respawn',
                                   respawn_args=(stype, name, sx, sy, z))
            else:
                self._state = STATE_SPAWNING
                self._async_spawn(stype, name, sx, sy, z)

    def _async_spawn(self, stype: str, name: str,
                     x: float, y: float, z: float) -> None:
        req = SpawnEntity.Request()
        req.name = name
        req.xml = build_source_sdf(stype, name)
        req.robot_namespace = ''
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        req.initial_pose.orientation.w = 1.0
        req.reference_frame = 'world'

        fut = self.spawn_client.call_async(req)

        def _done(f):
            try:
                resp = f.result()
            except Exception as e:
                self.get_logger().warn(f"spawn_entity exception: {e}")
                resp = None
            with self._lock:
                if resp is None:
                    self.get_logger().warn("spawn_entity returned no response")
                elif not resp.success:
                    self.get_logger().warn(
                        f"spawn_entity failed: {resp.status_message}")
                else:
                    self.get_logger().info(
                        f"spawned hazard source in Gazebo: type={stype} "
                        f"name={name} at ({x:.2f},{y:.2f},{z:.2f})")
                    self._last_key = self._pending_key
                self._pending_key = None
                self._state = STATE_IDLE

        fut.add_done_callback(_done)

    def _async_delete(self, name: str, next_step: Optional[str] = None,
                      respawn_args: Optional[Tuple[str, str, float, float, float]] = None) -> None:
        if not self.delete_client.service_is_ready():
            with self._lock:
                self._state = STATE_IDLE
                self._pending_key = None
            return
        req = DeleteEntity.Request()
        req.name = name
        fut = self.delete_client.call_async(req)

        def _done(f):
            try:
                resp = f.result()
            except Exception as e:
                self.get_logger().warn(f"delete_entity exception: {e}")
                resp = None
            if resp is not None and resp.success:
                self.get_logger().info(f"deleted Gazebo entity '{name}'")
            if next_step == 'respawn' and respawn_args is not None:
                with self._lock:
                    self._state = STATE_SPAWNING
                stype, ename, rx, ry, rz = respawn_args
                self._async_spawn(stype, ename, rx, ry, rz)
            else:
                with self._lock:
                    self._state = STATE_IDLE
                    self._pending_key = None

        fut.add_done_callback(_done)

    def teardown_blocking(self, executor) -> None:
        """外部调用: 结束前同步等待 delete, 避免下次启动 name already exists."""
        self._shutdown = True
        if self._last_key is None:
            return
        try:
            name = str(self.get_parameter('entity_name').value) or 'hazard_source_viz'
        except Exception:
            return
        if not self.delete_client.service_is_ready():
            return
        req = DeleteEntity.Request()
        req.name = name
        fut = self.delete_client.call_async(req)
        # 只在 spin 线程退出之前 (executor 尚在 spin), 这里不用 spin_until_future_complete,
        # 改成简单的条件轮询, executor 另一个线程在驱动 future.
        import time
        deadline = time.time() + 3.0
        while not fut.done() and time.time() < deadline:
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = HazardGazeboVisualNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.teardown_blocking(executor)
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
