#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灾害扩散场的物理计算核心，不依赖 ROS，方便单元测试与复用。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Tuple
import numpy as np
OBSTACLE_VALUE = 100
def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w
def compute_geodesic_distance(
    grid: np.ndarray,
    source_rc: Tuple[int, int],
) -> np.ndarray:
    """
    8 邻接 BFS 计算每个格子到源的"测地距离" (单位: 格数)。
    墙格 (grid == 100) 不可通过，墙格自身距离置为 +inf。

    对角步长按 sqrt(2) 计，普通步长 1。
    返回 np.ndarray[H, W]，dtype=float32，不可达位置为 +inf。
    """
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    sr, sc = source_rc
    if not _in_bounds(sr, sc, h, w):
        return dist
    if grid[sr, sc] == OBSTACLE_VALUE:
        return dist
    SQRT2 = math.sqrt(2.0)
    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    )

    dist[sr, sc] = 0.0
    queue: deque = deque()
    queue.append((sr, sc))

    while queue:
        r, c = queue.popleft()
        d0 = dist[r, c]
        for dr, dc, step in neighbors:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, h, w):
                continue
            if grid[nr, nc] == OBSTACLE_VALUE:
                continue
            nd = d0 + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                queue.append((nr, nc))
    return dist


def _downsample_obstacle_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """
    把占用栅格按 factor 倍降采样，用 'max' 聚合以保留墙体。
    只要一个 sub-block 中存在任何墙格 (==100)，聚合格即视为墙，
    这样"墙阻隔扩散"的语义在降采样后不会被稀释。
    """
    if factor <= 1:
        return grid
    h, w = grid.shape
    nh = h // factor
    nw = w // factor
    if nh == 0 or nw == 0:
        return grid
    cropped = grid[: nh * factor, : nw * factor]
    blocks = cropped.reshape(nh, factor, nw, factor)
    is_wall = (blocks == OBSTACLE_VALUE).any(axis=(1, 3))
    small = np.where(is_wall, OBSTACLE_VALUE, 0).astype(grid.dtype)
    return small


def _upsample_field(small: np.ndarray, factor: int, target_shape: Tuple[int, int]) -> np.ndarray:
    """最近邻上采样, 把降采样后的 field 放回原栅格尺寸。"""
    if factor <= 1:
        return small
    big = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    th, tw = target_shape
    h, w = big.shape
    if h < th or w < tw:
        padded = np.zeros(target_shape, dtype=small.dtype)
        padded[:h, :w] = big
        return padded
    return big[:th, :tw]


def compute_geodesic_field(
    grid: np.ndarray,
    source_rc: Tuple[int, int],
    C0: float,
    alpha: float,
    resolution: float = 1.0,
    downsample: int = 1,
) -> np.ndarray:
    """
    基于测地距离的指数衰减浓度/温度场:

        C(x) = C0 * exp(-alpha * d_geodesic(x, source) * resolution)

    - resolution : 每格对应的物理距离 (m/cell)，用于把 alpha 的单位保持在 1/m
    - downsample : BFS 前先按此倍率降采样以加速 (默认 1 = 不降采样)，
                   墙体按"任一子格为墙则聚合格为墙"聚合, 保留阻隔语义, 最后上采样回原尺寸。
                   典型值: 栅格 >= 400x400 时用 4, >=1000x1000 时用 8。
    - 墙体/墙后不可达 → 测地距离 +inf → 浓度严格为 0
    - 返回 np.ndarray[H, W]，dtype=float32
    """
    if downsample <= 1:
        dist = compute_geodesic_distance(grid, source_rc)
        field = C0 * np.exp(-alpha * dist * resolution)
        field[~np.isfinite(dist)] = 0.0
        field[grid == OBSTACLE_VALUE] = 0.0
        return field.astype(np.float32)

    small_grid = _downsample_obstacle_grid(grid, downsample)
    sr, sc = source_rc
    s_sr = min(max(sr // downsample, 0), small_grid.shape[0] - 1)
    s_sc = min(max(sc // downsample, 0), small_grid.shape[1] - 1)

    small_dist = compute_geodesic_distance(small_grid, (s_sr, s_sc))
    step_m = float(resolution) * float(downsample)
    small_field = C0 * np.exp(-alpha * small_dist * step_m)
    small_field[~np.isfinite(small_dist)] = 0.0
    small_field[small_grid == OBSTACLE_VALUE] = 0.0

    field = _upsample_field(small_field.astype(np.float32), downsample, grid.shape)
    field[grid == OBSTACLE_VALUE] = 0.0
    return field


def compute_diffusion_field(
    grid: np.ndarray,
    source_rc: Tuple[int, int],
    C0: float,
    iters: int = 200,
    decay: float = 0.02,
    downsample: int = 1,
) -> np.ndarray:
    """
    基于扩散方程 (稳态 Laplacian) 的 Jacobi 数值迭代:

        C_new = mean(4-neighbors(C_old)) * (1 - decay)

    边界条件:
      - 墙格 (grid == 100) 恒为 0 (Dirichlet)
      - 源格 恒为 C0 (Dirichlet 激励)

    decay 是每步的体积损耗项，用来模拟扩散介质中的衰减，
    让远离源的浓度随传播衰减 (没有这一项，稳态解在无限域是常数)。

    注意:
      - 墙体天然阻隔扩散，因为墙格始终为 0，Laplacian 更新会把
        墙后的值拉回 0，完全符合"不穿墙"的要求。
      - iters 越大越接近稳态，默认 200 在常见地图尺寸下就足够。
      - downsample : 迭代前先按此倍率降采样以加速 (默认 1 = 不降采样)。
    返回 np.ndarray[H, W]，dtype=float32
    """
    target_shape = grid.shape
    work_grid = _downsample_obstacle_grid(grid, downsample) if downsample > 1 else grid
    sr, sc = source_rc
    if downsample > 1:
        sr = min(max(sr // downsample, 0), work_grid.shape[0] - 1)
        sc = min(max(sc // downsample, 0), work_grid.shape[1] - 1)

    h, w = work_grid.shape
    field = np.zeros((h, w), dtype=np.float32)

    if not _in_bounds(sr, sc, h, w):
        return field.astype(np.float32) if downsample <= 1 else _upsample_field(field, downsample, target_shape)
    if work_grid[sr, sc] == OBSTACLE_VALUE:
        return field.astype(np.float32) if downsample <= 1 else _upsample_field(field, downsample, target_shape)

    obstacle_mask = (work_grid == OBSTACLE_VALUE)
    field[sr, sc] = float(C0)

    keep = 1.0 - max(0.0, min(1.0, float(decay)))

    for _ in range(int(iters)):
        up = np.roll(field, -1, axis=0)
        down = np.roll(field, 1, axis=0)
        left = np.roll(field, -1, axis=1)
        right = np.roll(field, 1, axis=1)
        up[-1, :] = 0.0
        down[0, :] = 0.0
        left[:, -1] = 0.0
        right[:, 0] = 0.0

        new_field = 0.25 * (up + down + left + right) * keep
        new_field[obstacle_mask] = 0.0
        new_field[sr, sc] = float(C0)
        field = new_field

    field[obstacle_mask] = 0.0
    if downsample > 1:
        field = _upsample_field(field.astype(np.float32), downsample, target_shape)
        field[grid == OBSTACLE_VALUE] = 0.0
    return field.astype(np.float32)


def world_to_grid(
    x: float, y: float,
    origin_x: float, origin_y: float,
    resolution: float,
    width: int, height: int,
) -> Tuple[int, int]:
    """
    地图坐标 (x, y) -> 栅格 (row, col)。
    约定: OccupancyGrid row 对应 y，col 对应 x，
    与原项目 astar.map_callback 的写法保持一致。
    返回的 (row, col) 可能越界，调用方应自行判断。
    """
    col = int((x - origin_x) / resolution)
    row = int((y - origin_y) / resolution)
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))
    return row, col


def grid_to_world(
    row: int, col: int,
    origin_x: float, origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    """
    栅格 (row, col) -> 地图坐标 (x, y)，取格子中心。
    """
    x = (col + 0.5) * resolution + origin_x
    y = (row + 0.5) * resolution + origin_y
    return x, y
