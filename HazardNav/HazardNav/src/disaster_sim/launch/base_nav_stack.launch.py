#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_nav_stack.launch.py
------------------------

模块化启动(阶段 1): 只启动底座仿真与原导航栈，不包含 disaster_sim 节点。

包含:
  - gazebo_modele/gazebo.launch.py
  - nav_slam: astar, map_pub, odom_map_tf, points_pub_map, start_nav
  - (可选) rviz2

参数:
  use_rviz       是否启动 RViz,   默认 true
  rviz_config    RViz 配置文件路径
  auto_clean     启动前是否自动清理残留 gzserver/gzclient/ros 节点,
                 默认 true (强烈建议). 关掉请传 auto_clean:=false.
                 关掉后如果端口 11345 已被其它 gzserver 占用, Gazebo
                 会启动即崩溃 (bind: Address already in use), 窗口看
                 起来根本没打开.

典型用途:
  - 先确认机器人底盘、TF、激光、A*/纯追踪都正常
  - 再在第二个终端启动 hazard_stack.launch.py 叠加灾害场能力

注意:
  - auto_clean 会 pkill 用户自己的 gzserver/gzclient/nav_slam/lib/
    disaster_sim/lib 进程. 如果你的其它工作依赖同名进程, 请 auto_clean:=false.
  - 为了让 pkill 有时间释放端口, Gazebo/nav/rviz 启动会被延迟约 3s.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PRE_CLEAN_CMD = (
    'echo "[base_nav_stack] auto_clean=true, killing stale gazebo/ros nodes..."; '
    'pkill -KILL -u "$USER" -f gzserver 2>/dev/null; '
    'pkill -KILL -u "$USER" -f gzclient 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "disaster_sim/lib" 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "nav_slam/lib" 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "gazebo --verbose" 2>/dev/null; '
    'sleep 2; '
    'echo "[base_nav_stack] pre-clean done."; '
    'true'
)


def generate_launch_description():
    pkg_disaster = get_package_share_directory('disaster_sim')
    pkg_gazebo = get_package_share_directory('gazebo_modele')
    get_package_share_directory('nav_slam')

    default_rviz = os.path.join(pkg_disaster, 'config', 'rviz_hazard.rviz')

    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    auto_clean = LaunchConfiguration('auto_clean')

    args = [
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument(
            'auto_clean', default_value='true',
            description='pkill stale gazebo/ros processes before launch, '
                        'then delay ~3s. Set false to skip cleaning.'),
    ]

    pre_clean = ExecuteProcess(
        cmd=['bash', '-c', PRE_CLEAN_CMD],
        output='screen',
        condition=IfCondition(auto_clean),
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo, 'launch', 'gazebo.launch.py')),
    )

    nav_nodes = GroupAction(actions=[
        Node(package='nav_slam', executable='astar',
             name='astar', output='screen', respawn=False),
        Node(package='nav_slam', executable='map_pub',
             name='map_pub', output='screen', respawn=False),
        Node(package='nav_slam', executable='odom_map_tf',
             name='odom_map_tf', output='screen', respawn=False),
        Node(package='nav_slam', executable='points_pub_map',
             name='points_pub_map', output='screen', respawn=False),
        Node(package='nav_slam', executable='start_nav',
             name='start_nav', output='screen', respawn=False),
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
    )

    # 固定小延迟, 让 pre_clean 的 pkill + sleep 有时间释放端口.
    # 即使 auto_clean=false, 延迟几秒也能给上一个终端的残留自然退出机会.
    delayed_gazebo = TimerAction(period=3.0, actions=[gazebo_launch])
    delayed_nav = TimerAction(period=3.5, actions=[nav_nodes])
    delayed_rviz = TimerAction(period=5.0, actions=[rviz_node])

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)
    ld.add_action(pre_clean)
    ld.add_action(delayed_gazebo)
    ld.add_action(delayed_nav)
    ld.add_action(delayed_rviz)
    return ld
