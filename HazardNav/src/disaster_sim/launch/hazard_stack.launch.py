#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hazard_stack.launch.py
----------------------

模块化启动(阶段 2): 只启动 disaster_sim 的灾害场与交互能力。
必须先在另一个终端运行 base_nav_stack.launch.py, 等 Gazebo 与 nav_slam
的节点全部 ready 后再启动本 launch.

包含:
  - hazard_source_node
  - gradient_explorer_node
  - hazard_gazebo_visual_node (可选, 需要 Gazebo 已就绪)
  - hazard_control_panel_node (可选)
  - trajectory_logger_node (可选)

警告:
  - 本 launch 不做 pkill, 以免误杀 base_nav_stack 的 gzserver/nav_slam.
  - 如果 base_nav_stack 没启动, hazard_gazebo_visual_node 会因 /spawn_entity
    不可用而告警, 关掉 use_gazebo_viz:=false 可临时绕过.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _guess_workspace_csv_dir(share_dir: str) -> str:
    try:
        ws_root = os.path.abspath(
            os.path.join(share_dir, '..', '..', '..', '..'))
        cand = os.path.join(ws_root, 'src', 'csv')
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    return ''


def generate_launch_description():
    pkg_disaster = get_package_share_directory('disaster_sim')

    default_params = os.path.join(pkg_disaster, 'config', 'hazard_params.yaml')
    default_csv_dir = _guess_workspace_csv_dir(pkg_disaster)

    source_type = LaunchConfiguration('source_type')
    source_x = LaunchConfiguration('source_x')
    source_y = LaunchConfiguration('source_y')
    algorithm = LaunchConfiguration('algorithm')
    C0 = LaunchConfiguration('C0')
    arrival_thresh = LaunchConfiguration('arrival_thresh')
    use_gazebo_viz = LaunchConfiguration('use_gazebo_viz')
    use_control_panel = LaunchConfiguration('use_control_panel')
    log_trajectory = LaunchConfiguration('log_trajectory')
    trajectory_csv = LaunchConfiguration('trajectory_csv')
    trajectory_dir = LaunchConfiguration('trajectory_dir')
    params_file = LaunchConfiguration('params_file')

    args = [
        DeclareLaunchArgument('source_type', default_value='fire',
                              description='fire | gas | pollution'),
        DeclareLaunchArgument('source_x', default_value='6.0',
                              description='hazard source x in map frame (m)'),
        DeclareLaunchArgument('source_y', default_value='4.0',
                              description='hazard source y in map frame (m)'),
        DeclareLaunchArgument('algorithm', default_value='geodesic',
                              description='geodesic | diffusion'),
        DeclareLaunchArgument('C0', default_value='600.0',
                              description='source strength; unit depends on source_type'),
        DeclareLaunchArgument('arrival_thresh', default_value='350.0',
                              description='sample threshold that triggers STOP.'),
        DeclareLaunchArgument('use_gazebo_viz', default_value='true',
                              description='spawn hazard source visual entity in Gazebo'),
        DeclareLaunchArgument('use_control_panel', default_value='true',
                              description='open tkinter GUI for live parameter tweaking'),
        DeclareLaunchArgument('log_trajectory', default_value='true',
                              description='log trajectory to CSV'),
        DeclareLaunchArgument('trajectory_csv', default_value='',
                              description='CSV path; empty -> <trajectory_dir>/disaster_trajectory_<timestamp>.csv'),
        DeclareLaunchArgument('trajectory_dir', default_value=default_csv_dir,
                              description='directory for auto-named CSV files'),
        DeclareLaunchArgument('params_file', default_value=default_params),
    ]

    hazard_node = Node(
        package='disaster_sim',
        executable='hazard_source_node',
        name='hazard_source_node',
        output='screen',
        respawn=False,
        parameters=[
            params_file,
            {
                'source_type': source_type,
                'source_x': source_x,
                'source_y': source_y,
                'algorithm': algorithm,
                'C0': C0,
            },
        ],
    )

    explorer_node = Node(
        package='disaster_sim',
        executable='gradient_explorer_node',
        name='gradient_explorer_node',
        output='screen',
        respawn=False,
        parameters=[
            params_file,
            {
                'arrival_sample_threshold': arrival_thresh,
                'use_source_goal': False,
            },
        ],
    )

    gazebo_viz_node = Node(
        package='disaster_sim',
        executable='hazard_gazebo_visual_node',
        name='hazard_gazebo_visual_node',
        output='screen',
        respawn=False,
        condition=IfCondition(use_gazebo_viz),
        parameters=[
            {
                'source_type': source_type,
                'source_x': source_x,
                'source_y': source_y,
            },
        ],
    )

    control_panel_node = Node(
        package='disaster_sim',
        executable='hazard_control_panel_node',
        name='hazard_control_panel_node',
        output='screen',
        respawn=False,
        condition=IfCondition(use_control_panel),
    )

    trajectory_logger_node = Node(
        package='disaster_sim',
        executable='trajectory_logger_node',
        name='trajectory_logger_node',
        output='screen',
        respawn=False,
        condition=IfCondition(log_trajectory),
        parameters=[
            {
                'output_csv': trajectory_csv,
                'output_dir': trajectory_dir,
                'source_type': source_type,
                'fallback_source_x': source_x,
                'fallback_source_y': source_y,
                'arrival_sample_threshold': arrival_thresh,
            },
        ],
    )

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)
    ld.add_action(hazard_node)
    ld.add_action(explorer_node)
    ld.add_action(gazebo_viz_node)
    ld.add_action(control_panel_node)
    ld.add_action(trajectory_logger_node)
    return ld
