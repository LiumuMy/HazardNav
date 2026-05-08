#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_logger_node
----------------------

把机器人趋源航行过程中的关键数据写进一份 CSV, 用于事后分析 / 画图 / 写报告.

每隔 `period_sec` (默认 0.5s) 记录一行:

    timestamp_sec,  x,  y,  source_x,  source_y,
    dist_to_source,  sample,  source_type,  arrived

订阅:
  /odom             nav_msgs/Odometry           机器人当前位姿 -> (x, y)
  /hazard/sample    std_msgs/Float32            机器人处读数
  /hazard/source_gt geometry_msgs/PointStamped  源真值位置 (可选, 没有就用参数里的)

参数:
  output_csv        CSV 输出路径 (若给出则直接使用, 优先级最高)
  output_dir        CSV 目录 (空串则回退 ~). output_csv 为空时, 文件名自动
                    生成为 <output_dir>/disaster_trajectory_<timestamp>.csv.
                    disaster_nav.launch.py 默认会把该值指向工作区 src/csv/,
                    这样所有次次实验的 CSV 都集中落在同一目录便于归档.
  period_sec        记录周期 (s), 默认 0.5
  append            true: 追加到已有文件; false: 总是覆盖; 默认 false
  source_type       记录到 CSV 的第 N 列 (只是打标签, 不参与计算), 默认 'fire'
  fallback_source_x 若没订阅到 /hazard/source_gt 时使用的源 x
  fallback_source_y 若没订阅到 /hazard/source_gt 时使用的源 y
  arrival_sample_threshold  当采样 >= 此值 -> arrived 列写 1

CSV 一打开就能直接用 Excel / pandas:
    df = pd.read_csv('disaster_trajectory_xxx.csv')
    df.plot(x='x', y='y')
    df.plot(x='timestamp_sec', y='sample')
"""

from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


CSV_HEADER = [
    'timestamp_sec',
    'x',
    'y',
    'source_x',
    'source_y',
    'dist_to_source',
    'sample',
    'source_type',
    'arrived',
]


def _default_csv_path(out_dir: str = '') -> str:
    """按 out_dir + 时间戳自动拼一个 csv 文件名; out_dir 空则回退用户家目录。"""
    base = out_dir.strip() if out_dir else ''
    if not base:
        base = os.path.expanduser('~')
    base = os.path.expanduser(base)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base, f'disaster_trajectory_{ts}.csv')


class TrajectoryLoggerNode(Node):
    """把 odom + hazard sample + 源位置合成一条 CSV 记录。"""

    def __init__(self) -> None:
        super().__init__('trajectory_logger_node')

        self.declare_parameter('output_csv', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('period_sec', 0.5)
        self.declare_parameter('append', False)
        self.declare_parameter('source_type', 'fire')
        self.declare_parameter('fallback_source_x', 6.0)
        self.declare_parameter('fallback_source_y', 4.0)
        self.declare_parameter('arrival_sample_threshold', 350.0)

        out = str(self.get_parameter('output_csv').value).strip()
        out_dir = str(self.get_parameter('output_dir').value).strip()
        self.output_csv = out if out else _default_csv_path(out_dir)
        self.period_sec = max(0.05, float(self.get_parameter('period_sec').value))
        self.append = bool(self.get_parameter('append').value)
        self.source_type = str(self.get_parameter('source_type').value)
        self.fallback_src = (
            float(self.get_parameter('fallback_source_x').value),
            float(self.get_parameter('fallback_source_y').value),
        )
        self.arrival_threshold = float(
            self.get_parameter('arrival_sample_threshold').value)

        self._robot_xy: Optional[Tuple[float, float]] = None
        self._sample: Optional[float] = None
        self._source_gt: Optional[Tuple[float, float]] = None
        self._start_time = time.time()
        self._row_count = 0

        qos = QoSProfile(depth=10)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos)
        self.sub_sample = self.create_subscription(
            Float32, '/hazard/sample', self._sample_cb, qos)
        self.sub_src = self.create_subscription(
            PointStamped, '/hazard/source_gt', self._src_cb, qos)

        self._open_csv()
        self.timer = self.create_timer(self.period_sec, self._tick)

        self.get_logger().info(
            f"trajectory_logger_node ready: writing -> {self.output_csv} "
            f"period={self.period_sec}s append={self.append}"
        )

    def _open_csv(self) -> None:
        out_dir = os.path.dirname(os.path.abspath(self.output_csv))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        mode = 'a' if self.append and os.path.isfile(self.output_csv) else 'w'
        self._fh = open(self.output_csv, mode, newline='', encoding='utf-8')
        self._writer = csv.writer(self._fh)
        if mode == 'w':
            self._writer.writerow(CSV_HEADER)
            self._fh.flush()

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

    def _sample_cb(self, msg: Float32) -> None:
        self._sample = float(msg.data)

    def _src_cb(self, msg: PointStamped) -> None:
        self._source_gt = (float(msg.point.x), float(msg.point.y))

    def _current_source(self) -> Tuple[float, float]:
        return self._source_gt if self._source_gt is not None else self.fallback_src

    def _tick(self) -> None:
        if self._robot_xy is None:
            return
        rx, ry = self._robot_xy
        sx, sy = self._current_source()
        dist = math.hypot(rx - sx, ry - sy)
        sample = self._sample if self._sample is not None else float('nan')
        arrived = 1 if (self._sample is not None
                        and self._sample >= self.arrival_threshold) else 0
        ts = round(time.time() - self._start_time, 3)

        self._writer.writerow([
            f'{ts:.3f}',
            f'{rx:.4f}',
            f'{ry:.4f}',
            f'{sx:.4f}',
            f'{sy:.4f}',
            f'{dist:.4f}',
            f'{sample:.4f}',
            self.source_type,
            arrived,
        ])
        self._fh.flush()
        self._row_count += 1
        if self._row_count % 20 == 1:
            self.get_logger().info(
                f"[log] t={ts:.2f}s pose=({rx:.2f},{ry:.2f}) "
                f"dist={dist:.2f}m sample={sample:.2f} "
                f"arrived={arrived} rows={self._row_count}"
            )

    def close(self) -> None:
        try:
            if getattr(self, '_fh', None) is not None:
                self._fh.close()
                self.get_logger().info(
                    f"csv closed: {self.output_csv} "
                    f"({self._row_count} rows)"
                )
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
